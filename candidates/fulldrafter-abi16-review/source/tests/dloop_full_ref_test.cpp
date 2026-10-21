#include "rad_test.h"
#include "mine/dloop_full.h"
#include "core/format/safetensors.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

/* The source-model fidelity fixture (scripts/generate_dloop_full_fixture.py). The build normally
 * passes the directory; the fallback keeps a misconfigured build compiling so the case FAILS with a
 * path in the message instead of taking the whole binary down at compile time. */
#ifdef RAD_DLOOP_FULL_FIXTURES
#define FULL_FIXTURE_DIR RAD_DLOOP_FULL_FIXTURES
#else
#define FULL_FIXTURE_DIR "tests/fixtures/dloop_full_tiny"
#endif

/* Contract and independent source-model parity tests for the full-context host oracle:
 * context beyond 32, chunk/path append, slot reuse, refusal before mutation, backbone,
 * selector and DARTree. Numeric fixtures come from the unchanged PyTorch implementation. */
namespace {
constexpr int L=5;                    /* five layers is the production count, even at tiny width */

RadDLoopFullConfig tiny(){
    RadDLoopFullConfig cfg{};
    RadDLoopConfig& m=cfg.model;
    m.n_seq=2;m.max_seqs=4;m.tap_rows=8;m.world_size=2;
    m.hidden_size=64;m.intermediate_size=128;m.num_layers=L;m.num_q_heads=4;m.num_kv_heads=2;
    m.head_dim=16;m.context_features_dim=96;m.target_width=32;m.proposal_length=8;
    m.context_window=0;m.active_vocab=509;m.vocab_size=512;m.selector_rank=8;
    m.conv_kernel_size=2;m.conv_group_size=16;m.vocab_size_shard=256;m.pool_size=64;
    m.budget=15;m.width=12;m.eps=1e-6f;m.rope_theta=1e7f;m.depth_bonus=-.2f;
    cfg.context_capacity=48;           /* > 32 on purpose: the old ring geometry is not the bound */
    return cfg;
}
std::vector<RadParam> params(const RadDLoopFullConfig& c){
    std::vector<RadParam> p;
#define P(k) {RadParam v{};v.key=#k;v.kind=RAD_P_INT;v.ival=c.model.k;p.push_back(v);}
    P(n_seq);P(max_seqs);P(tap_rows);P(world_size);P(hidden_size);P(intermediate_size);P(num_layers);
    P(num_q_heads);P(num_kv_heads);P(head_dim);P(context_features_dim);P(target_width);P(proposal_length);
    P(context_window);P(active_vocab);P(vocab_size);P(selector_rank);P(conv_kernel_size);P(conv_group_size);
    P(vocab_lo);P(vocab_size_shard);P(pool_size);P(budget);P(width);
#undef P
    {RadParam v{};v.key="context_capacity";v.kind=RAD_P_INT;v.ival=c.context_capacity;p.push_back(v);}
    RadParam variant{};variant.key="variant";variant.kind=RAD_P_STR;variant.sval=c.model.pruned?"pruned":"fixed";p.push_back(variant);
#define F(k) {RadParam v{};v.key=#k;v.kind=RAD_P_F64;v.dval=c.model.k;p.push_back(v);}
    F(eps);F(rope_theta);F(depth_bonus);
#undef F
    RadParam dtype{};dtype.key="dtype";dtype.kind=RAD_P_STR;dtype.sval="bf16";p.push_back(dtype);
    return p;
}
int64_t scratch_elements(const RadDLoopFullConfig& c,int op){
    return rad_dloop_full_scratch_elements(c,op);
}

struct Call {
    RadDLoopFullConfig c;std::vector<RadParam> p;std::vector<RadTensor> t;
    std::vector<std::vector<uint64_t>> storage;std::vector<float> scratch;int op;
    Call(int operation,RadDLoopFullConfig config):c(config),p(params(c)),op(operation){
        auto* schema=rad_dloop_full_schema_at(op);t.resize(schema->n_operands);storage.resize(t.size());
        for(size_t i=0;i<t.size();++i){
            RadOpdDesc d{};if(rad_dloop_full_shape(op,p.data(),(int)p.size(),(int)i,&d))throw std::runtime_error("bad shape");
            if(d.flags&RAD_OPD_F_ABSENT)continue;
            int64_t n=1;t[i].dtype=d.dtype;t[i].rank=d.rank;
            for(int j=(int)d.rank-1;j>=0;--j){t[i].shape[j]=d.shape[j];t[i].stride[j]=n;n*=d.shape[j];}
            storage[i].assign(size_t((n*(d.dtype==RAD_BF16?2:4)+7)/8)+64,0);t[i].data=storage[i].data();}
        scratch.assign(size_t(c.model.n_seq*scratch_elements(c,op))+8,0.0f);
    }
    RadArgs args(){RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
        a.scratch=scratch.data();a.scratch_bytes=(int64_t)scratch.size()*4;return a;}
    int run(){auto a=args();const RadLaunchFn f[]={rad_ref_dloop_full_append,rad_ref_dloop_full_base,
        rad_ref_dloop_full_dartree};return f[op](&a,nullptr);}
    template<class T>T* data(int i){return static_cast<T*>(t[i].data);}
    void copy(int i,const void* src,size_t bytes){std::memcpy(t[i].data,src,bytes);}
    void bind(int i,void* ptr,uint32_t dtype,std::initializer_list<int64_t> shape){
        RadTensor v{};v.data=ptr;v.dtype=dtype;v.rank=(uint32_t)shape.size();int j=0;for(auto n:shape)v.shape[j++]=n;
        int64_t s=1;for(int k=(int)v.rank-1;k>=0;--k){v.stride[k]=s;s*=v.shape[k];}t[i]=v;}
    /* Every byte a launch may write, for the "nothing moved" checks. */
    std::vector<uint32_t> snapshot_outputs(){
        std::vector<uint32_t> out;
        auto* schema=rad_dloop_full_schema_at(op);
        for(int i=0;i<(int)t.size();++i){
            if(!t[i].data||schema->operands[i].role==RAD_OPD_IN)continue;
            if(op==RAD_DLOOP_FULL_APPEND&&i==9){
                const int* len=data<int32_t>(9);
                for(int q=0;q<c.model.max_seqs;++q)out.push_back((uint32_t)len[q]);
                continue;
            }
            if(op==RAD_DLOOP_FULL_APPEND&&i==8){out.push_back(cache_digest());continue;}
            const auto* p32=static_cast<const uint32_t*>(t[i].data);
            int64_t n=1;for(unsigned k=0;k<t[i].rank;++k)n*=t[i].shape[k];
            for(int64_t j=0;j<n;++j)out.push_back(p32[j]);
        }
        return out;
    }
    uint32_t cache_digest(){
        const auto* p32=static_cast<const uint32_t*>(t[8].data);
        int64_t n=1;for(unsigned k=0;k<t[8].rank;++k)n*=t[8].shape[k];
        uint32_t h=2166136261u;for(int64_t j=0;j<n;++j)h=(h^p32[j])*16777619u;
        return h;
    }
};
/* Non-trivial but exactly representable taps, so a mis-placed row shows up as a mismatch. */
void fill_taps(Call& call){
    for(int plane=0;plane<3;++plane){
        auto* rows=call.data<uint16_t>(plane);
        for(int r=0;r<call.c.model.tap_rows;++r)
            for(int j=0;j<call.c.model.target_width;++j)
                rows[(int64_t)r*call.c.model.target_width+j]=rad_f32_to_bf16((float)(plane*1000+r*10)+(j%7)*0.25f);
    }
}
void fill_weights(Call& call){
    const int64_t n=rad_dloop_weight_elements(call.c.model);
    auto* w=call.data<uint16_t>(7);
    for(int64_t j=0;j<n;++j)w[j]=rad_f32_to_bf16((float)((j%23)-11)*0.03125f);
}
void fill_slots(Call& call,const std::vector<int32_t>& slots){
    for(size_t s=0;s<slots.size();++s)call.data<int32_t>(5)[s]=slots[s];
}
void set_span(Call& call,int s,int32_t start,int32_t count){
    call.data<int32_t>(4)[2*s]=start;call.data<int32_t>(4)[2*s+1]=count;
}
}

/* ---------------------------------------------------------------- config and geometry */
TEST(full_config_requires_zero_window_and_capacity){
    auto p=params(tiny());RadDLoopFullConfig out{};
    REQUIRE_EQ(rad_dloop_full_config(p.data(),(int)p.size(),&out),RAD_OK);
    CHECK_EQ(out.context_capacity,48);CHECK_EQ(out.model.context_window,0);CHECK_EQ(out.model.num_layers,L);
    /* context_window is not a window any more: any non-zero value is refused, not reinterpreted. */
    for(auto& v:p)if(!std::strcmp(v.key,"context_window"))v.ival=32;
    CHECK_EQ(rad_dloop_full_config(p.data(),(int)p.size(),&out),RAD_E_UNSUPPORTED);
    for(auto& v:p)if(!std::strcmp(v.key,"context_window"))v.ival=0;
    /* capacity is required */
    std::vector<RadParam> missing=p;missing.erase(std::remove_if(missing.begin(),missing.end(),
        [](const RadParam& v){return !std::strcmp(v.key,"context_capacity");}),missing.end());
    CHECK_EQ(rad_dloop_full_config(missing.data(),(int)missing.size(),&out),RAD_E_UNSUPPORTED);
    /* three tap planes: F is 3*W and nothing else */
    for(auto& v:p)if(!std::strcmp(v.key,"context_features_dim"))v.ival=2*32;
    CHECK_EQ(rad_dloop_full_config(p.data(),(int)p.size(),&out),RAD_E_UNSUPPORTED);
}
TEST(full_shapes_carry_the_layer_axis_and_no_window){
    auto p=params(tiny());const int D=tiny().model.num_kv_heads*tiny().model.head_dim;
    auto shape=[&](int op,int operand){RadOpdDesc d{};CHECK_EQ(rad_dloop_full_shape(op,p.data(),(int)p.size(),operand,&d),RAD_OK);return d;};
    const auto cache=shape(RAD_DLOOP_FULL_BASE,0);
    CHECK_EQ(cache.rank,4u);CHECK_EQ(cache.dtype,RAD_F32);
    CHECK_EQ(cache.shape[0],4);CHECK_EQ(cache.shape[1],L);CHECK_EQ(cache.shape[2],48);
    CHECK_EQ(cache.shape[3],2*D);
    const auto len=shape(RAD_DLOOP_FULL_BASE,1);CHECK_EQ(len.shape[0],4);
    const auto path=shape(RAD_DLOOP_FULL_APPEND,3);CHECK_EQ(path.shape[0],2);CHECK_EQ(path.shape[1],16);
    const auto span=shape(RAD_DLOOP_FULL_APPEND,4);CHECK_EQ(span.shape[0],2);CHECK_EQ(span.shape[1],2);
    for(int plane=0;plane<3;++plane){const auto tap=shape(RAD_DLOOP_FULL_APPEND,plane);
        CHECK_EQ(tap.rank,2u);CHECK_EQ(tap.shape[0],8);CHECK_EQ(tap.shape[1],32);}
    CHECK(shape(RAD_DLOOP_FULL_APPEND,5).flags&RAD_OPD_F_IDX_UNIQUE);
    const auto opt=shape(RAD_DLOOP_FULL_DARTREE,17);
    CHECK(opt.flags&RAD_OPD_F_ABSENT);CHECK_EQ(opt.shape[0],2);CHECK_EQ(opt.shape[1],33);
    CHECK_EQ(shape(RAD_DLOOP_FULL_BASE,2).rank,1u);      /* slots */
}
TEST(full_args_reject_a_mismatched_cache_and_a_short_scratch){
    Call call(RAD_DLOOP_FULL_BASE,tiny());
    call.data<int32_t>(1)[0]=4;call.data<int32_t>(2)[0]=0;call.data<int32_t>(2)[1]=1;
    RadDLoopFullConfig out{};
    auto a=call.args();REQUIRE_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_BASE,&out),RAD_OK);
    CHECK_EQ(out.context_capacity,call.c.context_capacity);
    a=call.args();call.t[0].shape[2]++;
    CHECK_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_BASE,&out),RAD_E_SHAPE);
    call.t[0].shape[2]--;
    a=call.args();call.t[0].stride[3]++;
    CHECK_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_BASE,&out),RAD_E_SHAPE);
    call.t[0].stride[3]--;
    a=call.args();a.scratch_bytes=4;
    CHECK_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_BASE,&out),RAD_E_SCRATCH);
    /* length aliasing an output is refused: a mutable operand must not overlap another. */
    call.bind(6,call.t[0].data,RAD_F32,{2,8,32});
    a=call.args();CHECK_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_BASE,&out),RAD_E_INVAL);
}

/* ---------------------------------------------------------------- append behaviour */
TEST(full_append_grows_past_thirty_two_and_appends_chunks){
    auto c=tiny();c.context_capacity=48;Call call(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(call);fill_weights(call);fill_slots(call,{3,1});
    call.data<int32_t>(6)[0]=1;call.data<int32_t>(6)[1]=1;
    set_span(call,0,0,8);set_span(call,1,0,8);
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[3],8);CHECK_EQ(call.data<int32_t>(9)[1],8);
    /* Five more chunks of eight: 48 positions, well past the ring's 32, and no wrap. */
    for(int chunk=1;chunk<6;++chunk){
        call.data<int32_t>(6)[0]=0;call.data<int32_t>(6)[1]=0;
        set_span(call,0,0,8);set_span(call,1,0,8);
        REQUIRE_EQ(call.run(),RAD_OK);
        CHECK_EQ(call.data<int32_t>(9)[3],(chunk+1)*8);
    }
    CHECK_EQ(call.data<int32_t>(9)[3],48);
    /* Position 0 was written by the LAST chunk's first row, i.e. tap row 0. One fresh append of
     * exactly that row must leave the same bytes there -- which is what "no wrap, no truncation"
     * means for a capacity that the old ring geometry could not name. */
    Call fresh(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(fresh);fill_weights(fresh);fill_slots(fresh,{3,1});
    fresh.data<int32_t>(6)[0]=1;fresh.data<int32_t>(6)[1]=0;
    set_span(fresh,0,0,1);set_span(fresh,1,0,0);
    REQUIRE_EQ(fresh.run(),RAD_OK);
    const int D=c.model.num_kv_heads*c.model.head_dim;
    for(int layer=0;layer<L;++layer)
        for(int d=0;d<2*D;++d)
            CHECK_EQ(call.data<float>(8)[rad_dloop_full_kv_index(c,3,layer,0,d)],
                     fresh.data<float>(8)[rad_dloop_full_kv_index(c,3,layer,0,d)]);
    /* A further chunk overflows capacity and must change nothing at all. */
    auto before=call.snapshot_outputs();
    set_span(call,0,0,8);
    CHECK_EQ(call.run(),RAD_E_INVAL);
    CHECK(call.snapshot_outputs()==before);
}
TEST(full_append_accepts_a_full_sixteen_row_path_and_rejects_a_dirty_tail){
    /* A complete root-plus-fifteen accepted path fills all sixteen columns and carries NO -1:
     * the schema has exactly sixteen places, so a full path is legal. */
    auto c=tiny();c.model.tap_rows=20;Call call(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(call);fill_weights(call);fill_slots(call,{0,1});
    call.data<int32_t>(6)[0]=1;call.data<int32_t>(6)[1]=1;
    for(int j=0;j<16;++j)call.data<int32_t>(3)[j]=(int32_t)j;
    set_span(call,0,0,-1);set_span(call,1,0,0);
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[0],16);
    /* A row index at the tap plane's bound is out of range even though sixteen columns is legal. */
    int32_t* path=call.data<int32_t>(3);
    const int32_t saved=path[15];
    path[15]=c.model.tap_rows;
    {
        auto before=call.snapshot_outputs();
        CHECK_EQ(call.run(),RAD_E_INVAL);
        CHECK(call.snapshot_outputs()==before);
    }
    path[15]=saved;
    /* A full sixteen must also fit the remaining capacity; 48 - 16 leaves room, so a second
     * sixteen-row path still lands, and the third does not. */
    call.data<int32_t>(6)[0]=0;
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[0],32);
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[0],48);
    {
        auto before=call.snapshot_outputs();
        CHECK_EQ(call.run(),RAD_E_INVAL);      /* capacity, not truncation */
        CHECK(call.snapshot_outputs()==before);
    }
    /* A positive value BEHIND a -1 is malformed: the tail is not "free to ignore". */
    Call dirty(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(dirty);fill_weights(dirty);fill_slots(dirty,{0,1});
    dirty.data<int32_t>(6)[0]=1;dirty.data<int32_t>(6)[1]=0;
    std::fill(dirty.data<int32_t>(3),dirty.data<int32_t>(3)+32,-1);
    dirty.data<int32_t>(3)[0]=3;dirty.data<int32_t>(3)[1]=-1;dirty.data<int32_t>(3)[2]=5;
    set_span(dirty,0,0,-1);set_span(dirty,1,0,0);
    auto snapshot=dirty.snapshot_outputs();
    CHECK_EQ(dirty.run(),RAD_E_INVAL);
    CHECK(dirty.snapshot_outputs()==snapshot);
    /* A row index outside the tap plane behind a valid prefix is refused too. */
    dirty.data<int32_t>(3)[2]=-1;dirty.data<int32_t>(3)[3]=c.model.tap_rows;
    CHECK_EQ(dirty.run(),RAD_E_INVAL);
    dirty.data<int32_t>(3)[3]=-1;
    REQUIRE_EQ(dirty.run(),RAD_OK);
    CHECK_EQ(dirty.data<int32_t>(9)[0],1);
}
TEST(full_append_chunk_equals_single_row_paths){
    auto c=tiny();Call chunk(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(chunk);fill_weights(chunk);fill_slots(chunk,{2,1});
    chunk.data<int32_t>(6)[0]=1;chunk.data<int32_t>(6)[1]=1;
    set_span(chunk,0,2,3);set_span(chunk,1,0,0);
    REQUIRE_EQ(chunk.run(),RAD_OK);
    CHECK_EQ(chunk.data<int32_t>(9)[2],3);
    /* The same three rows as three one-row path appends. A chunk and a path are two spellings of
     * one contract, so the resulting cache must be byte-identical. */
    Call rows(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(rows);fill_weights(rows);fill_slots(rows,{2,1});
    std::fill(rows.data<int32_t>(3),rows.data<int32_t>(3)+32,-1);
    const int32_t order[3]={2,3,4};
    for(int j=0;j<3;++j){
        rows.data<int32_t>(6)[0]=j==0?1:0;rows.data<int32_t>(6)[1]=0;
        std::fill(rows.data<int32_t>(3),rows.data<int32_t>(3)+32,-1);
        rows.data<int32_t>(3)[0]=order[j];
        set_span(rows,0,0,-1);set_span(rows,1,0,0);
        REQUIRE_EQ(rows.run(),RAD_OK);
    }
    CHECK_EQ(rows.data<int32_t>(9)[2],3);
    const int64_t cache_n=(int64_t)c.model.max_seqs*L*c.context_capacity*2*c.model.num_kv_heads*c.model.head_dim;
    CHECK(std::equal(chunk.data<float>(8),chunk.data<float>(8)+cache_n,rows.data<float>(8)));
    /* A different path order is a different history: descending rows must not land the same. */
    Call swapped(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(swapped);fill_weights(swapped);fill_slots(swapped,{2,1});
    swapped.data<int32_t>(6)[0]=1;swapped.data<int32_t>(6)[1]=0;
    std::fill(swapped.data<int32_t>(3),swapped.data<int32_t>(3)+32,-1);
    swapped.data<int32_t>(3)[0]=4;swapped.data<int32_t>(3)[1]=3;swapped.data<int32_t>(3)[2]=2;
    set_span(swapped,0,0,-1);set_span(swapped,1,0,0);
    REQUIRE_EQ(swapped.run(),RAD_OK);
    float delta=0;
    const int D=c.model.num_kv_heads*c.model.head_dim;
    for(int layer=0;layer<L;++layer)for(int pos=0;pos<3;++pos)for(int d=0;d<2*D;++d)
        delta=std::max(delta,std::abs(chunk.data<float>(8)[rad_dloop_full_kv_index(c,2,layer,pos,d)]-
                                      swapped.data<float>(8)[rad_dloop_full_kv_index(c,2,layer,pos,d)]));
    CHECK(delta>1e-4f);
    /* Every one of the sixteen columns names a row past the tap plane: refused before any write. A
     * full sixteen is legal (see the full-path test), so the fault here is the ROW, not the absence
     * of a terminator. */
    auto before=rows.snapshot_outputs();
    std::fill(rows.data<int32_t>(3),rows.data<int32_t>(3)+32,c.model.tap_rows);
    CHECK_EQ(rows.run(),RAD_E_INVAL);
    CHECK(rows.snapshot_outputs()==before);
}
TEST(full_append_refuses_malformed_metadata_before_mutating){
    auto c=tiny();Call call(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(call);fill_weights(call);fill_slots(call,{0,1});
    call.data<int32_t>(6)[0]=1;call.data<int32_t>(6)[1]=1;
    set_span(call,0,0,4);set_span(call,1,0,4);
    call.data<int32_t>(9)[0]=0;call.data<int32_t>(9)[1]=0;
    const auto check_refused=[&](){
        auto before=call.snapshot_outputs();
        CHECK_EQ(call.run(),RAD_E_INVAL);
        CHECK(call.snapshot_outputs()==before);};
    /* duplicate slot inside one append */
    call.data<int32_t>(5)[1]=0;check_refused();call.data<int32_t>(5)[1]=1;
    /* slot out of range */
    call.data<int32_t>(5)[1]=9;check_refused();call.data<int32_t>(5)[1]=1;
    /* reset is 0/1 only */
    call.data<int32_t>(6)[1]=2;check_refused();call.data<int32_t>(6)[1]=0;
    /* negative count other than -1 */
    set_span(call,0,0,-2);check_refused();set_span(call,0,0,4);
    /* span past the tap rows */
    set_span(call,0,6,4);check_refused();set_span(call,0,0,4);
    /* start beyond tap_rows */
    set_span(call,0,9,1);check_refused();set_span(call,0,0,4);
    /* a path row out of range */
    std::fill(call.data<int32_t>(3),call.data<int32_t>(3)+32,-1);
    call.data<int32_t>(3)[0]=99;set_span(call,0,0,-1);
    check_refused();set_span(call,0,0,4);
    /* capacity overflow is an error, never a truncation. reset must be 0 here or the stored
     * length is ignored and the row count is all that is left to check. */
    call.data<int32_t>(6)[0]=0;
    call.data<int32_t>(9)[0]=45;set_span(call,0,0,4);
    check_refused();
    /* a length that is already past capacity */
    call.data<int32_t>(9)[0]=49;set_span(call,0,0,1);
    check_refused();
    /* a length that is exactly full leaves no room for another row */
    call.data<int32_t>(9)[0]=48;set_span(call,0,0,1);
    check_refused();
    call.data<int32_t>(9)[0]=0;set_span(call,0,0,4);
    /* one broken sequence must stop the whole batch: the first is fine, the second is not. */
    set_span(call,0,0,4);set_span(call,1,0,-3);
    check_refused();
    /* and now a legal append still works */
    set_span(call,1,0,4);
    REQUIRE_EQ(call.run(),RAD_OK);CHECK_EQ(call.data<int32_t>(9)[0],4);CHECK_EQ(call.data<int32_t>(9)[1],4);
}
TEST(full_append_empty_chunk_and_slot_reuse){
    auto c=tiny();Call call(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(call);fill_weights(call);
    /* Sequence 0 owns slot 1, sequence 1 owns slot 3. */
    fill_slots(call,{1,3});
    call.data<int32_t>(6)[0]=1;call.data<int32_t>(6)[1]=1;
    set_span(call,0,0,3);set_span(call,1,0,0);      /* an empty chunk is legal */
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[1],3);CHECK_EQ(call.data<int32_t>(9)[3],0);
    /* Slot 1 gets a new occupant: reset clears the three positions, then one row lands. */
    call.data<int32_t>(6)[0]=1;call.data<int32_t>(6)[1]=0;
    set_span(call,0,0,1);set_span(call,1,0,0);
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[1],1);
    /* Slot 3 is a different slot and keeps its own history: the same append leaves it alone. */
    call.data<int32_t>(6)[0]=0;
    set_span(call,0,0,1);set_span(call,1,0,2);
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[1],2);CHECK_EQ(call.data<int32_t>(9)[3],2);
    /* Reset with an empty chunk clears slot 1 and stops there; slot 3, still appending two rows
     * per launch, carries on: reset is per slot, not per batch. */
    call.data<int32_t>(6)[0]=1;set_span(call,0,0,0);
    REQUIRE_EQ(call.run(),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[1],0);CHECK_EQ(call.data<int32_t>(9)[3],4);
}
TEST(full_append_leaves_the_cache_tail_byte_identical){
    auto c=tiny();Call call(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(call);fill_weights(call);fill_slots(call,{2,0});
    call.data<int32_t>(6)[0]=1;call.data<int32_t>(6)[1]=1;
    set_span(call,0,0,5);set_span(call,1,0,0);
    const int D=c.model.num_kv_heads*c.model.head_dim;
    auto cell=[&](int slot,int layer,int pos)->float*{return call.data<float>(8)+
        rad_dloop_full_kv_index(c,slot,layer,pos,0);};
    /* Poison everything past position 5 in slot 2 and everything in slot 0: the launch must not
     * touch a byte outside [0,length). */
    for(int layer=0;layer<L;++layer)for(int pos=5;pos<c.context_capacity;++pos)
        for(int d=0;d<2*D;++d)cell(2,layer,pos)[d]=-777.0f;
    for(int layer=0;layer<L;++layer)for(int pos=0;pos<c.context_capacity;++pos)
        for(int d=0;d<2*D;++d)cell(0,layer,pos)[d]=-555.0f;
    REQUIRE_EQ(call.run(),RAD_OK);
    for(int layer=0;layer<L;++layer){
        for(int pos=5;pos<c.context_capacity;++pos)
            for(int d=0;d<2*D;++d)CHECK_EQ(cell(2,layer,pos)[d],-777.0f);
        for(int pos=0;pos<c.context_capacity;++pos)
            for(int d=0;d<2*D;++d)CHECK_EQ(cell(0,layer,pos)[d],-555.0f);
    }
}

/* ---------------------------------------------------------------- base over a long prefix */
TEST(full_base_block_rows_use_their_own_q_rows){
    /* Regression: the block attention pass must give row t its own Q row. An implementation that
     * added t*H twice (caller and callee both offsetting) reads past the block for later rows and
     * silently reuses row 0's query -- which shows up here as all draft rows agreeing. */
    auto c=tiny();Call call(RAD_DLOOP_FULL_BASE,c);
    const int64_t n=rad_dloop_weight_elements(c.model);
    auto* w=call.data<uint16_t>(4);
    for(int64_t j=0;j<n;++j)w[j]=rad_f32_to_bf16((float)((j%23)-11)*0.03125f);
    for(int i=0;i<c.model.vocab_size_shard;++i)for(int j=0;j<c.model.target_width;++j)
        call.data<uint16_t>(5)[(int64_t)i*c.model.target_width+j]=rad_f32_to_bf16((float)((i+j)%13)*0.0625f);
    const int D=c.model.num_kv_heads*c.model.head_dim;
    const int64_t kv=(int64_t)c.model.max_seqs*L*c.context_capacity*2*D;
    for(int64_t j=0;j<kv;++j)call.data<float>(0)[j]=(float)((j%17)-8)*0.125f;
    call.data<int32_t>(2)[0]=0;call.data<int32_t>(2)[1]=1;
    call.data<int32_t>(1)[0]=6;call.data<int32_t>(1)[1]=6;
    REQUIRE_EQ(call.run(),RAD_OK);
    /* The anchor embedding differs per row, so every draft row must produce its own y. A run whose
     * rows were all row 0 would be constant across t. */
    auto* anchor=call.data<uint16_t>(3);
    for(int t=0;t<c.model.target_width;++t)anchor[t]=rad_f32_to_bf16((float)t*0.125f);
    REQUIRE_EQ(call.run(),RAD_OK);
    float spread=0;
    for(int t=1;t<8;++t)for(int j=0;j<32;++j)
        spread=std::max(spread,std::abs(call.data<float>(6)[t*32+j]-call.data<float>(6)[j]));
    CHECK(spread>1e-4f);
    /* And the whole y buffer is finite: an out-of-range read would not be. */
    for(int t=0;t<8;++t)for(int j=0;j<32;++j)CHECK(std::isfinite(call.data<float>(6)[t*32+j]));
}
TEST(full_append_arena_covers_the_projection_row){
    auto c=tiny();
    c.model.n_seq=1;c.model.max_seqs=1;c.model.tap_rows=1;c.model.num_kv_heads=1;
    Call call(RAD_DLOOP_FULL_APPEND,c);
    fill_taps(call);fill_weights(call);
    call.data<int32_t>(5)[0]=0;call.data<int32_t>(6)[0]=1;
    set_span(call,0,0,1);
    const int64_t elements=rad_dloop_full_scratch_elements(c,RAD_DLOOP_FULL_APPEND);
    constexpr float canary=-12345.0f;
    call.scratch.assign(size_t(elements)+64,canary);
    auto args=call.args();args.scratch_bytes=elements*4;
    REQUIRE_EQ(rad_ref_dloop_full_append(&args,nullptr),RAD_OK);
    CHECK_EQ(call.data<int32_t>(9)[0],1);
    CHECK(std::all_of(call.scratch.begin()+elements,call.scratch.end(),
                      [](float value){return value==canary;}));
}
TEST(full_base_reads_the_whole_prefix_and_refuses_an_empty_one){
    auto c=tiny();Call call(RAD_DLOOP_FULL_BASE,c);
    const int64_t n=rad_dloop_weight_elements(c.model);
    auto* w=call.data<uint16_t>(4);
    for(int64_t j=0;j<n;++j)w[j]=rad_f32_to_bf16((float)((j%23)-11)*0.03125f);
    for(int i=0;i<c.model.vocab_size_shard;++i)for(int j=0;j<c.model.target_width;++j)
        call.data<uint16_t>(5)[(int64_t)i*c.model.target_width+j]=rad_f32_to_bf16((float)((i+j)%13)*0.0625f);
    const int D=c.model.num_kv_heads*c.model.head_dim;
    const int64_t kv=(int64_t)c.model.max_seqs*L*c.context_capacity*2*D;
    for(int64_t j=0;j<kv;++j)call.data<float>(0)[j]=(float)((j%17)-8)*0.125f;
    /* Position 40 of slot 1 is the LAST verified position, so a causal implementation could reach
     * it and a windowed one would still see it. Position 37 is early: only a full prefix does. */
    auto* late=call.data<float>(0)+rad_dloop_full_kv_index(c,1,0,40,0);
    auto* early=call.data<float>(0)+rad_dloop_full_kv_index(c,1,0,37,0);
    call.data<int32_t>(2)[0]=1;call.data<int32_t>(2)[1]=3;
    call.data<int32_t>(1)[1]=41;call.data<int32_t>(1)[3]=48;
    REQUIRE_EQ(call.run(),RAD_OK);
    std::vector<float> before(call.data<float>(6),call.data<float>(6)+2*8*32);
    float late_delta=0,early_delta=0;
    for(int d=0;d<2*D;++d)late[d]+=1.5f;
    REQUIRE_EQ(call.run(),RAD_OK);
    for(int j=0;j<8*32;++j)late_delta=std::max(late_delta,std::abs(before[j]-call.data<float>(6)[8*32+j]));
    for(int d=0;d<2*D;++d){late[d]-=1.5f;early[d]+=1.5f;}
    REQUIRE_EQ(call.run(),RAD_OK);
    for(int j=0;j<8*32;++j)early_delta=std::max(early_delta,std::abs(before[j]-call.data<float>(6)[8*32+j]));
    CHECK(late_delta>1e-4f);
    CHECK(early_delta>1e-4f);      /* a ring of 32 could not have seen this one */
    /* Every draft row has its candidates. */
    for(int t=0;t<8;++t)for(int j=0;j<64;++j)CHECK(call.data<int32_t>(7)[t*64+j]>=0);
    /* An empty prefix is a drafter with no context: refused, not silently zeroed. */
    call.data<int32_t>(1)[3]=0;
    auto snapshot=call.snapshot_outputs();
    CHECK_EQ(call.run(),RAD_E_INVAL);
    CHECK(call.snapshot_outputs()==snapshot);
    /* A prefix past capacity is refused as well. */
    call.data<int32_t>(1)[3]=c.context_capacity+1;
    CHECK_EQ(call.run(),RAD_E_INVAL);
    /* Two sequences reading the SAME slot is not a fault: it is one prefix read twice, which is
     * exactly what a batched drafter does when the caller groups by prefix. */
    call.data<int32_t>(1)[3]=4;call.data<int32_t>(2)[1]=1;
    CHECK_EQ(call.run(),RAD_OK);
    /* A draft with no verified prefix is not a model state: length 0 is refused, not zeroed. */
    call.data<int32_t>(2)[1]=3;call.data<int32_t>(1)[1]=0;
    CHECK_EQ(call.run(),RAD_E_INVAL);
    call.data<int32_t>(1)[1]=1;
    REQUIRE_EQ(call.run(),RAD_OK);
}

/* ---------------------------------------------------------------- dartree over the cache */
namespace {
void fill_pool(Call& call){
    auto c=call.c;const int M=c.model.proposal_length,K=c.model.pool_size,W=c.model.target_width,R=c.model.selector_rank;
    const int64_t rows=(int64_t)c.model.n_seq*M*K;
    for(int64_t i=0;i<rows;++i){
        /* Strictly increasing within a candidate row, so the sorted-pool contract holds; the row
         * offset keeps different rows apart. Kept inside the active vocabulary. */
        const int64_t row=i/K,j=i%K;
        call.data<int32_t>(6)[i]=(int32_t)(1+j*3+(row%4)*4);
        call.data<float>(7)[i]=(float)((i*7)%29)*0.25f-3.0f;
        auto* head=call.data<uint16_t>(8)+i*W;auto* embed=call.data<uint16_t>(9)+i*W;
        auto* pred=call.data<uint16_t>(10)+i*R;auto* succ=call.data<uint16_t>(11)+i*R;
        for(int jj=0;jj<W;++jj){head[jj]=rad_f32_to_bf16((float)((i+jj)%17)*0.03125f);
            embed[jj]=rad_f32_to_bf16((float)((i*2+jj)%13)*0.0625f);}
        for(int r=0;r<R;++r){pred[r]=rad_f32_to_bf16((float)((r%5)-2)*0.125f);
            succ[r]=rad_f32_to_bf16((float)((r%3)-1)*0.25f);}
    }
    for(int s=0;s<c.model.n_seq;++s)for(int r=0;r<R;++r)
        call.data<uint16_t>(12)[(int64_t)s*R+r]=rad_f32_to_bf16((float)((r%7)-3)*0.125f);
}
}
TEST(full_dartree_uses_the_cache_and_prunes_to_budget){
    auto c=tiny();c.model.pruned=1;
    for(int variant=0;variant<2;++variant){
        auto cfg=c;cfg.model.pruned=variant;
        Call call(RAD_DLOOP_FULL_DARTREE,cfg);
        const int64_t n=rad_dloop_weight_elements(cfg.model);
        auto* w=call.data<uint16_t>(4);
        for(int64_t j=0;j<n;++j)w[j]=rad_f32_to_bf16((float)((j%23)-11)*0.03125f);
        const int64_t kv=(int64_t)cfg.model.max_seqs*L*cfg.context_capacity*2*cfg.model.num_kv_heads*cfg.model.head_dim;
        for(int64_t j=0;j<kv;++j)call.data<float>(0)[j]=(float)((j%17)-8)*0.125f;
        call.data<int32_t>(2)[0]=1;call.data<int32_t>(2)[1]=2;
        call.data<int32_t>(1)[1]=36;call.data<int32_t>(1)[2]=48;    /* past 32: the ring bound */
        fill_pool(call);
        std::array<int32_t,2> anchors{7,9};
        std::array<int32_t,2*33> packed{};
        call.bind(16,anchors.data(),RAD_I32,{2});
        call.bind(17,packed.data(),RAD_I32,{2,33});
        REQUIRE_EQ(call.run(),RAD_OK);
        for(int s=0;s<cfg.model.n_seq;++s){
            const int count=call.data<int32_t>(15)[s];
            CHECK(count>=1);CHECK(count<=cfg.model.budget);CHECK(count<=15);
            CHECK_EQ(packed[s*33],count+1);CHECK_EQ(packed[s*33+1],anchors[s]);CHECK_EQ(packed[s*33+17],-1);
            for(int j=0;j<count;++j){
                const int parent=call.data<int32_t>(14)[s*16+j];
                CHECK(parent>=-1);CHECK(parent<j);
                for(int k=0;k<j;++k)CHECK(parent!=call.data<int32_t>(14)[s*16+k]||
                    call.data<int32_t>(13)[s*16+j]!=call.data<int32_t>(13)[s*16+k]);
                CHECK_EQ(packed[s*33+j+2],call.data<int32_t>(13)[s*16+j]);
                CHECK_EQ(packed[s*33+j+18],parent+1);
            }
            for(int j=count;j<16;++j)CHECK_EQ(call.data<int32_t>(13)[s*16+j],-1);
        }
    }
}
TEST(full_dartree_refuses_pool_metadata_before_mutating){
    auto c=tiny();Call call(RAD_DLOOP_FULL_DARTREE,c);
    const int64_t n=rad_dloop_weight_elements(c.model);
    auto* w=call.data<uint16_t>(4);
    for(int64_t j=0;j<n;++j)w[j]=rad_f32_to_bf16((float)((j%23)-11)*0.03125f);
    call.data<int32_t>(2)[0]=0;call.data<int32_t>(1)[0]=8;
    fill_pool(call);
    const auto check_refused=[&](){
        auto before=call.snapshot_outputs();
        CHECK_EQ(call.run(),RAD_E_INVAL);
        CHECK(call.snapshot_outputs()==before);};
    /* An id outside the active vocabulary. */
    call.data<int32_t>(6)[0]=c.model.active_vocab;check_refused();
    call.data<int32_t>(6)[0]=5;
    /* Unsorted within one row. */
    call.data<int32_t>(6)[1]=call.data<int32_t>(6)[0];check_refused();
    call.data<int32_t>(6)[1]=call.data<int32_t>(6)[0]+1;
    /* A non-finite unary score. */
    call.data<float>(7)[3]=std::numeric_limits<float>::quiet_NaN();check_refused();
    call.data<float>(7)[3]=0.5f;
    /* A BF16 NaN in a gathered teacher row would poison the tree. */
    call.data<uint16_t>(8)[5]=0x7fc1u;check_refused();call.data<uint16_t>(8)[5]=0x3c00u;
    call.data<uint16_t>(12)[2]=0x7fc1u;check_refused();call.data<uint16_t>(12)[2]=0x3c00u;
    /* An invalid anchor id. */
    const std::array<int32_t,2> bad{1,c.model.active_vocab};
    call.bind(16,const_cast<int32_t*>(bad.data()),RAD_I32,{2});
    check_refused();
    /* An empty cache cannot be a tree's context: a draft with length 0 is not a state. */
    call.t[16].data=nullptr;
    call.data<int32_t>(1)[0]=0;check_refused();call.data<int32_t>(1)[0]=8;
    /* anchor and packed_tree are both or neither: the anchor alone is a schema disagreement. */
    RadDLoopFullConfig out{};
    const std::array<int32_t,2> anchors{1,2};
    call.bind(16,const_cast<int32_t*>(anchors.data()),RAD_I32,{2});
    std::array<int32_t,2*33> packed{};
    call.t[17].data=nullptr;
    auto a=call.args();CHECK_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_DARTREE,&out),RAD_E_INVAL);
    /* Now both: the pair is complete and the call must be accepted. */
    call.bind(17,packed.data(),RAD_I32,{2,33});
    a=call.args();CHECK_EQ(rad_dloop_full_args(&a,RAD_DLOOP_FULL_DARTREE,&out),RAD_OK);
    REQUIRE_EQ(call.run(),RAD_OK);
    for(int s=0;s<2;++s){CHECK_EQ(packed[s*33],call.data<int32_t>(15)[s]+1);CHECK_EQ(packed[s*33+1],anchors[s]);}
}
/* ---------------------------------------------------------------- source-model fidelity
 *
 * The goldens here come from scripts/generate_dloop_full_fixture.py, which drives the UPSTREAM
 * research model (prepare_context / forward_cached / build_dartree / _candidate_pool) in torch and
 * writes its numbers. It calls nothing in this C++ oracle and reads no mine_ref output, so agreeing
 * with these values is evidence about the model contract rather than a tautology -- which is the
 * whole point of the case.
 *
 * Geometry is the generator's published tiny set: H64/FF128/L5/Q4/KV2/Dh16/F96/W32/M8/K64/R8,
 * vocab 512/509, shard 256, TP2, context_capacity 40, tap_rows 34. The four-chunk plan therefore
 * runs well past a 32-token ring and lands a full sixteen-row path.
 *
 * Tolerances are the exporter's own for F32 outputs (atol 3e-5, rtol 2e-5); discrete results
 * (ids, parents, node counts) are compared EXACTLY. Nothing is loosened here. */
namespace {
struct FullFixture {
    rad::SafeTensorsFile student,teacher,full;
    std::vector<uint16_t> packed;
    RadDLoopFullConfig cfg{};
    FullFixture(){
        const std::string root=FULL_FIXTURE_DIR;
        const auto path=[&](const char* name){return root+"/"+name;};
        if(student.open(path("dloop_sidecar.safetensors"))||
           teacher.open(path("teacher-matrices.safetensors"))||
           full.open(path("full.safetensors")))
            throw std::runtime_error("required full-context fixture unavailable under "+root);
        RadDLoopConfig& m=cfg.model;
        m.n_seq=4;m.max_seqs=4;m.tap_rows=34;m.world_size=2;
        m.hidden_size=64;m.intermediate_size=128;m.num_layers=5;m.num_q_heads=4;m.num_kv_heads=2;
        m.head_dim=16;m.context_features_dim=96;m.target_width=32;m.proposal_length=8;
        m.context_window=0;m.active_vocab=509;m.vocab_size=512;m.selector_rank=8;
        m.conv_kernel_size=2;m.conv_group_size=16;m.vocab_size_shard=256;m.pool_size=64;
        m.budget=15;m.width=12;m.eps=1e-6f;m.rope_theta=1e7f;m.depth_bonus=-.2f;
        cfg.context_capacity=40;
        /* The sidecar holds the student weights as individual tensors; the op reads one dense blob,
         * so pack them in rad_dloop_weight_offset order. */
        packed.resize(rad_dloop_weight_elements(m));
        const auto add=[&](int w,int layer=0){
            std::string name="dloop.";
            if(w>=DL_INPUT_NORM&&w<=DL_DOWN_PROJ)name+="layers."+std::to_string(layer)+".";
            name+=rad_dloop_weight_name(w);
            const auto* t=student.find(name);
            if(!t||t->dtype!=RAD_BF16||t->numel()!=rad_dloop_weight_size(m,w))
                throw std::runtime_error("bad packed weight: "+name);
            std::memcpy(packed.data()+rad_dloop_weight_offset(m,w,layer),t->data,t->bytes);
        };
        for(int w=0;w<=DL_MASK_TOKEN;++w)add(w);
        for(int l=0;l<m.num_layers;++l)for(int w=DL_INPUT_NORM;w<=DL_DOWN_PROJ;++w)add(w,l);
        for(int w=DL_FINAL_NORM;w<=DL_SELECTOR_PROJ;++w)add(w);
    }
    const rad::StTensor& get(const char* key){
        const auto* t=full.find(key);
        if(!t)throw std::runtime_error(std::string("missing fixture tensor: ")+key);
        return *t;
    }
    template<class T>const T* g(const char* key){return reinterpret_cast<const T*>(get(key).data);}
    const uint16_t* teacher_rows(const char* key){
        const auto* t=teacher.find(key);
        if(!t)throw std::runtime_error(std::string("missing teacher tensor: ")+key);
        return reinterpret_cast<const uint16_t*>(t->data);
    }
    const uint16_t* student_rows(const char* key){
        const auto* t=student.find(key);
        if(!t)throw std::runtime_error(std::string("missing student tensor: ")+key);
        return reinterpret_cast<const uint16_t*>(t->data);
    }
    void near(const char* what,const float* actual,const char* key,float atol,float rtol){
        const auto& golden=get(key);
        const int64_t n=golden.numel();
        const auto* expected=reinterpret_cast<const float*>(golden.data);
        float worst=0;int64_t bad=0;
        for(int64_t i=0;i<n;++i){
            const float delta=std::abs(actual[i]-expected[i]);
            worst=std::max(worst,delta);
            if(!std::isfinite(actual[i])||delta>atol+rtol*std::abs(expected[i]))++bad;
        }
        std::fprintf(stderr,"    fidelity %-14s n=%lld worst_abs=%g atol=%g rtol=%g bad=%lld\n",
            what,(long long)n,worst,atol,rtol,(long long)bad);
        CHECK_EQ(bad,0);
    }
};

/* One append call per chunk, all four sharing ONE cache and ONE length vector -- which is what a
 * chunked prefill is, and the only way the plan's reset/append semantics are exercised. */
struct AppendPlan {
    FullFixture& fx;
    std::vector<Call> calls;
    explicit AppendPlan(FullFixture& fixture):fx(fixture){
        const RadDLoopFullConfig& c=fx.cfg;
        const int64_t chunks=fx.get("chunk_span").shape[0];
        calls.reserve((size_t)chunks);
        for(int64_t i=0;i<chunks;++i)calls.emplace_back(RAD_DLOOP_FULL_APPEND,c);
        Call& shared=calls.front();
        std::fill(shared.data<int32_t>(9),shared.data<int32_t>(9)+c.model.max_seqs,0);
        std::fill(shared.data<float>(8),shared.data<float>(8)+rad_tensor_numel(&shared.t[8]),0.0f);
        for(size_t i=1;i<calls.size();++i){calls[i].t[8]=shared.t[8];calls[i].t[9]=shared.t[9];}
        const int32_t* span=fx.g<int32_t>("chunk_span");
        const int32_t* reset=fx.g<int32_t>("chunk_reset");
        const int32_t* path=fx.g<int32_t>("chunk_path");
        const int32_t* slots=fx.g<int32_t>("slots");
        const int n_seq=c.model.n_seq;
        for(size_t chunk=0;chunk<calls.size();++chunk){
            Call& call=calls[chunk];
            call.copy(0,fx.g<uint16_t>("tap0"),(size_t)c.model.tap_rows*c.model.target_width*2);
            call.copy(1,fx.g<uint16_t>("tap1"),(size_t)c.model.tap_rows*c.model.target_width*2);
            call.copy(2,fx.g<uint16_t>("tap2"),(size_t)c.model.tap_rows*c.model.target_width*2);
            call.copy(7,fx.packed.data(),fx.packed.size()*2);
            for(int s=0;s<n_seq;++s){
                call.data<int32_t>(5)[s]=slots[s];
                call.data<int32_t>(4)[2*s]=span[(chunk*n_seq+s)*2];
                call.data<int32_t>(4)[2*s+1]=span[(chunk*n_seq+s)*2+1];
                call.data<int32_t>(6)[s]=reset[chunk*n_seq+s];
                for(int j=0;j<16;++j)call.data<int32_t>(3)[s*16+j]=path[(chunk*n_seq+s)*16+j];
            }
        }
    }
    void drive(){
        for(size_t i=0;i<calls.size();++i){
            REQUIRE_EQ(calls[i].run(),RAD_OK);
            if(i+1<calls.size())calls[i+1].t[8]=calls[i].t[8];   /* the next chunk continues */
            if(i+1<calls.size())calls[i+1].t[9]=calls[i].t[9];
        }
    }
    RadTensor cache() const {return calls.back().t[8];}
    RadTensor length() const {return calls.back().t[9];}
};
}

/* A missing or malformed fixture is a FAILURE, never a skip: the contract cases above always check
 * something, so the binary can never be reported as "nothing was checked", and a silently passing
 * fidelity case is the one outcome the plan forbids. Catching here also keeps an unreadable fixture
 * from aborting the process and taking the other cases down with it. */
template<class F>
void fixture_case(const char* file,int line,F body) {
    try { body(); }
    catch (const std::exception& e) {
        std::fprintf(stderr,"    full-context fixture unavailable: %s\n",e.what());
        ::radtest::fail(file,line,std::string("full-context fixture: ")+e.what());
    }
}

TEST(full_fixture_append_matches_prepare_context_and_the_plan){
  fixture_case(__FILE__,__LINE__,[&]{
    FullFixture fx;AppendPlan plan(fx);plan.drive();
    const auto c=fx.cfg;
    const int32_t* lengths=fx.g<int32_t>("length");
    const int32_t* slots=fx.g<int32_t>("slots");
    const auto* actual=static_cast<const int32_t*>(plan.length().data);
    /* Length is slot-scoped and the plan's own: a batch-wide count would fail here. */
    for(int s=0;s<c.model.n_seq;++s)CHECK_EQ(actual[slots[s]],lengths[slots[s]]);
    /* The whole cache against prepare_context's layout: [slot, layer, position, 2D], K first and
     * already rotated at its absolute position, V second. */
    fx.near("cache",static_cast<const float*>(plan.cache().data),"cache",3e-5f,2e-5f);
  });
}
TEST(full_fixture_base_matches_forward_cached_and_both_shard_topk){
  fixture_case(__FILE__,__LINE__,[&]{
    FullFixture fx;AppendPlan plan(fx);plan.drive();
    const auto c=fx.cfg;
    const int32_t* slots=fx.g<int32_t>("slots");
    const int64_t* anchors=fx.g<int64_t>("anchors");   /* the fixture stores anchors as I64 */
    const auto cache=plan.cache(),length=plan.length();
    for(int rank=0;rank<c.model.world_size;++rank){
        auto cfg=c;cfg.model.vocab_lo=rank*c.model.vocab_size_shard;
        Call base(RAD_DLOOP_FULL_BASE,cfg);
        base.t[0]=cache;base.t[1]=length;
        base.copy(4,fx.packed.data(),fx.packed.size()*2);
        base.copy(5,fx.teacher_rows("lm_head.weight")+(int64_t)cfg.model.vocab_lo*c.model.target_width,
            (size_t)c.model.vocab_size_shard*c.model.target_width*2);
        const uint16_t* embed=fx.teacher_rows("model.language_model.embed_tokens.weight");
        for(int s=0;s<c.model.n_seq;++s){
            base.data<int32_t>(2)[s]=slots[s];
            std::memcpy(base.data<uint16_t>(3)+(int64_t)s*c.model.target_width,
                embed+(int64_t)anchors[s]*c.model.target_width,(size_t)c.model.target_width*2);
        }
        REQUIRE_EQ(base.run(),RAD_OK);
        fx.near("y_base",base.data<float>(6),"y_base",3e-5f,2e-5f);
        /* The shard top-k is the op's order (descending score, ties by lower id); the generator
         * refuses to emit a fixture whose cutoff is a tie the two sides could split. */
        const char* ids_key=rank==0?"rank0.local_ids":"rank1.local_ids";
        const char* values_key=rank==0?"rank0.local_logits":"rank1.local_logits";
        const int32_t* golden_ids=fx.g<int32_t>(ids_key);
        const int64_t rows=(int64_t)c.model.n_seq*c.model.proposal_length;
        CHECK(std::equal(base.data<int32_t>(7),base.data<int32_t>(7)+rows*c.model.pool_size,golden_ids));
        fx.near("local_logits",base.data<float>(8),values_key,3e-5f,2e-5f);
    }
  });
}
TEST(full_fixture_dartree_matches_build_dartree_both_variants){
  fixture_case(__FILE__,__LINE__,[&]{
    FullFixture fx;AppendPlan plan(fx);plan.drive();
    const auto c=fx.cfg;
    const int M=c.model.proposal_length,K=c.model.pool_size,W=c.model.target_width,R=c.model.selector_rank;
    const int32_t* slots=fx.g<int32_t>("slots");
    const int64_t* anchors=fx.g<int64_t>("anchors");   /* the fixture stores anchors as I64 */
    const int32_t* pool_ids=fx.g<int32_t>("pool_ids");
    const uint16_t* head=fx.teacher_rows("lm_head.weight");
    const uint16_t* embed=fx.teacher_rows("model.language_model.embed_tokens.weight");
    const uint16_t* pred_rows=fx.student_rows("dloop.selector.pred_codebook.weight");
    const uint16_t* succ_rows=fx.student_rows("dloop.selector.succ_codebook.weight");
    const int64_t rows=(int64_t)c.model.n_seq*M*K;
    const auto cache=plan.cache(),length=plan.length();
    /* pool_ids/unary come from the FIXTURE (build_dartree's own _candidate_pool), never from a pool
     * launch: re-deriving them with the C++ pool would fold that op into this expectation. The five
     * gathered ROW operands are not in the fixture either -- they are gather OUTPUTS, so they are
     * built here from the teacher shards, each rank contributing only its own rows (the all-reduce
     * is a bitwise OR because the other contributor is zero), which is what the tree reads in
     * serving. */
    for(int variant=0;variant<2;++variant){
        auto cfg=c;cfg.model.pruned=variant;
        Call tree(RAD_DLOOP_FULL_DARTREE,cfg);
        tree.t[0]=cache;tree.t[1]=length;
        tree.copy(4,fx.packed.data(),fx.packed.size()*2);
        for(int s=0;s<c.model.n_seq;++s){
            tree.data<int32_t>(2)[s]=slots[s];
            std::memcpy(tree.data<uint16_t>(3)+(int64_t)s*W,embed+(int64_t)anchors[s]*W,(size_t)W*2);
        }
        tree.copy(6,pool_ids,(size_t)rows*4);
        tree.copy(7,fx.g<float>("pool_unary"),(size_t)rows*4);
        /* y_base (operand 5) is both the tree's ROOT hidden state and an IN operand, so it is read
         * and never written. */
        tree.copy(5,fx.g<float>("y_base"),(size_t)c.model.n_seq*M*W*4);
        for(int plane=0;plane<4;++plane){
            const int width=plane<2?W:R;
            const uint16_t* table=plane==0?head:plane==1?embed:plane==2?pred_rows:succ_rows;
            auto* dst=tree.data<uint16_t>(8+plane);
            for(int64_t i=0;i<rows;++i){
                uint16_t* row=dst+i*width;
                std::fill(row,row+width,0);
                for(int rank=0;rank<c.model.world_size;++rank){
                    const int shard_lo=rank*c.model.vocab_size_shard;
                    const int id=pool_ids[i]-shard_lo;
                    if(id>=0&&id<c.model.vocab_size_shard&&pool_ids[i]<c.model.active_vocab) {
                        const uint16_t* src=table+(int64_t)pool_ids[i]*width;
                        for(int j=0;j<width;++j)row[j]=(uint16_t)(row[j]|src[j]);
                    }
                }
            }
        }
        /* anchor_pred is the ANCHOR's PREDECESSOR embedding: upstream passes the anchor as the
         * root's predecessor id, and `score_candidates` builds pred_emb from the pred codebook.
         * The ring pool oracle fills anchor_pred the same way (ref_dloop.cpp: operand 13 <- the
         * pred_shard operand). */
        for(int s=0;s<c.model.n_seq;++s){
            uint16_t* row=tree.data<uint16_t>(12)+(int64_t)s*R;
            std::fill(row,row+R,0);
            for(int rank=0;rank<c.model.world_size;++rank){
                const int shard_lo=rank*c.model.vocab_size_shard;
                if(anchors[s]>=shard_lo&&anchors[s]<shard_lo+c.model.vocab_size_shard) {
                    const uint16_t* src=pred_rows+(int64_t)anchors[s]*R;
                    for(int j=0;j<R;++j)row[j]=(uint16_t)(row[j]|src[j]);
                }
            }
        }
        const std::array<int32_t,4> anchor_copy={(int32_t)anchors[0],(int32_t)anchors[1],
            (int32_t)anchors[2],(int32_t)anchors[3]};
        tree.bind(16,const_cast<int32_t*>(anchor_copy.data()),RAD_I32,{4});
        std::array<int32_t,4*33> packed{};
        tree.bind(17,packed.data(),RAD_I32,{4,33});
        REQUIRE_EQ(tree.run(),RAD_OK);
        /* y_base is an IN operand: a second launch on the same inputs must see the same bytes, and
         * a launch that wrote its per-depth backbone output back into the operand would violate
         * that. Snapshotting the operand across a second identical run is the contract check. */
        const std::vector<float> y_base_before(fx.g<float>("y_base"),
            fx.g<float>("y_base")+rad_tensor_numel(&tree.t[5]));
        REQUIRE_EQ(tree.run(),RAD_OK);
        CHECK(std::equal(y_base_before.begin(),y_base_before.end(),
            static_cast<const float*>(tree.t[5].data)));
        const std::string name=variant?"pruned":"fixed";
        const int32_t* golden_nodes=fx.g<int32_t>((name+".n_nodes").c_str());
        const int64_t* tokens=fx.g<int64_t>((name+".tokens").c_str());
        const int64_t* parents=fx.g<int64_t>((name+".parents").c_str());
        for(int s=0;s<c.model.n_seq;++s){
            const int count=tree.data<int32_t>(15)[s];
            CHECK_EQ(count,golden_nodes[s]);
            for(int j=0;j<count;++j){
                CHECK_EQ(tree.data<int32_t>(13)[s*16+j],tokens[s*16+j+1]);
                CHECK_EQ(tree.data<int32_t>(14)[s*16+j],parents[s*16+j+1]-1);
            }
            CHECK_EQ(packed[s*33],count+1);
            CHECK_EQ(packed[s*33+1],anchors[s]);
            for(int j=count;j<16;++j)CHECK_EQ(tree.data<int32_t>(13)[s*16+j],-1);
        }
    }
  });
}
RAD_TEST_MAIN()
