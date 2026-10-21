/* Explicit GPU integration test. Compile while training runs; execute only after
 * the devices are released. Every op resolves through the ordinary plugin registry. */
#include "core/plugin/registry.h"
#include "core/device/device.h"
#include "core/format/safetensors.h"
#include "mine/dloop_full.h"
#include "mine/dloop_ops.h"
#include "mine/tree_gdn.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
/* Optional byte-for-byte A/B artifact from the device outputs, before the next stage consumes them.
 * This complements (not replaces) the tolerance-based host oracle and records top-k ordering too. */
void record_output(const void* data,size_t bytes) {
    const char* path=std::getenv("RADIANCE_DLOOP_OUTPUT_RECORD");
    if(!path)return;
    static std::ofstream out(path,std::ios::binary|std::ios::trunc);
    require(bool(out),"open device output recording");
    out.write(reinterpret_cast<const char*>(&bytes),sizeof(bytes));
    out.write(static_cast<const char*>(data),std::streamsize(bytes));
    out.flush();
    require(bool(out),"write device output recording");
}
bool bench_enabled() {
    static const bool on=[] {const char* v=std::getenv("RADIANCE_DLOOP_BENCH");return v&&std::atoi(v)>0;}();
    return on;
}
void check(int status,const std::string& what) {
    if(status!=RAD_OK) {
        const char* detail=rad_dev_last_error();
        throw std::runtime_error(what+": "+rad_strerror(status)+"; "+(detail?detail:""));
    }
}
size_t element_bytes(uint32_t dtype) {
    if(dtype==RAD_BF16)return 2;
    if(dtype==RAD_F32||dtype==RAD_I32)return 4;
    if(dtype==RAD_I64)return 8;
    throw std::runtime_error("unsupported test dtype");
}
size_t span_bytes(const RadTensor& t) {
    if(!t.rank)return 0;
    int64_t span=1;
    for(unsigned d=0;d<t.rank;++d){require(t.shape[d]>0&&t.stride[d]>0,"invalid test tensor extent");span+=(t.shape[d]-1)*t.stride[d];}
    return size_t(span)*element_bytes(t.dtype);
}
RadTensor tensor(void* data,uint32_t dtype,std::initializer_list<int64_t> shape) {
    RadTensor t{};t.data=data;t.dtype=dtype;
    for(int64_t d:shape)t.shape[t.rank++]=d;
    int64_t stride=1;for(int i=int(t.rank)-1;i>=0;--i){t.stride[i]=stride;stride*=t.shape[i];}return t;
}
RadParam pi(const char* key,int64_t value){RadParam p{};p.key=key;p.kind=RAD_P_INT;p.ival=value;return p;}
RadParam pf(const char* key,double value){RadParam p{};p.key=key;p.kind=RAD_P_F64;p.dval=value;return p;}
std::vector<RadParam> parameters(const RadDLoopConfig& c,int context_capacity=-1) {
    std::vector<RadParam> p;
#define P(k) p.push_back(pi(#k,c.k));
    P(n_seq);P(max_seqs);P(tap_rows);P(world_size);P(hidden_size);P(intermediate_size);P(num_layers);
    P(num_q_heads);P(num_kv_heads);P(head_dim);P(context_features_dim);P(target_width);P(proposal_length);
    P(context_window);P(active_vocab);P(vocab_size);P(selector_rank);P(conv_kernel_size);P(conv_group_size);
    P(vocab_lo);P(vocab_size_shard);P(pool_size);P(budget);P(width);
#undef P
    if(context_capacity>=0)p.push_back(pi("context_capacity",context_capacity));
    RadParam variant{};variant.key="variant";variant.kind=RAD_P_STR;variant.sval=c.pruned?"pruned":"fixed";p.push_back(variant);
    p.push_back(pf("eps",c.eps));p.push_back(pf("rope_theta",c.rope_theta));p.push_back(pf("depth_bonus",c.depth_bonus));
    RadParam dtype{};dtype.key="dtype";dtype.kind=RAD_P_STR;dtype.sval="bf16";p.push_back(dtype);return p;
}
/* Full-context op ids are their own table: RAD_DLOOP_FULL_* index it directly. The name is
 * checked here so a schema table reordered under the harness fails loudly instead of silently
 * launching the wrong kernel under the right operand list. */
const RadOpSchema* full_schema(int op) {
    static const char* names[]={"dloop_full_append","dloop_full_base","dloop_full_dartree"};
    require(op>=0&&op<RAD_DLOOP_FULL_OPS,"full-context op id out of range");
    const RadOpSchema* s=rad_dloop_full_schema_at(op);
    require(s!=nullptr&&std::strcmp(s->op,names[op])==0,"full-context op schema name mismatch");
    return s;
}
RadOpdDesc full_shape(int op,int operand,const std::vector<RadParam>& p) {
    RadOpdDesc d{};check(rad_dloop_full_shape(op,p.data(),int(p.size()),operand,&d),"operand shape");
    return d;
}
struct Call {
    std::string op;
    std::vector<RadParam> params;
    std::vector<RadTensor> t;
    std::vector<std::vector<uint64_t>> storage;
    int rank=0,world=1;
    Call(std::string name,std::vector<RadParam> p,int operands):op(std::move(name)),params(std::move(p)),t(operands),storage(operands){}
    Call(int op_id,const RadDLoopConfig& c):Call(rad_dloop_schema_at(op_id)->op,parameters(c),rad_dloop_schema_at(op_id)->n_operands) {
        rank=c.vocab_lo/c.vocab_size_shard;world=2;
        const RadShapeFn shapes[]={rad_dloop_ctx_append_shape,rad_dloop_base_shape,rad_dloop_pool_shape,rad_dloop_dartree_shape};
        const auto* schema=rad_dloop_schema_at(op_id);
        for(size_t i=0;i<t.size();++i) {
            RadOpdDesc d{};check(shapes[op_id](params.data(),int(params.size()),int(i),&d),"operand shape");
            if(d.flags&RAD_OPD_F_ABSENT)continue;
            t[i].dtype=d.dtype;t[i].rank=d.rank;int64_t stride=1;
            for(int j=int(d.rank)-1;j>=0;--j){t[i].shape[j]=d.shape[j];t[i].stride[j]=stride;stride*=d.shape[j];}
            /* The sidecar blob and teacher/codebook shards are bound from the fixture below. */
            const std::string name=schema->operands[i].name;
            if(name!="weights"&&name.find("_shard")==std::string::npos)allocate(int(i));
        }
    }
    /* `op_id` is RAD_DLOOP_FULL_*: the three full-context ops allocate every operand from
     * `rad_dloop_full_shape` with `context_capacity` in the parameter list, and the same
     * weights/shards/teacher bindings are supplied by the fixture afterwards. */
    Call(int op_id,int capacity,const RadDLoopConfig& c)
        :Call(full_schema(op_id)->op,parameters(c,capacity),full_schema(op_id)->n_operands) {
        rank=c.vocab_lo/c.vocab_size_shard;world=2;
        for(size_t i=0;i<t.size();++i) {
            const RadOpdDesc d=full_shape(op_id,int(i),params);
            if(d.flags&RAD_OPD_F_ABSENT)continue;
            t[i].dtype=d.dtype;t[i].rank=d.rank;int64_t stride=1;
            for(int j=int(d.rank)-1;j>=0;--j){t[i].shape[j]=d.shape[j];t[i].stride[j]=stride;stride*=d.shape[j];}
            const std::string name=full_schema(op_id)->operands[i].name;
            if(name!="weights"&&name.find("_shard")==std::string::npos)allocate(int(i));
        }
    }
    void allocate(int i){storage[i].resize((span_bytes(t[i])+7)/8);t[i].data=storage[i].data();}
    template<class T>T* data(int i){return static_cast<T*>(t[i].data);}
    template<class T>const T* data(int i)const{return static_cast<const T*>(t[i].data);}
    void bind(int i,void* data,uint32_t dtype,std::initializer_list<int64_t> shape){t[i]=tensor(data,dtype,shape);}
    void copy(int i,const void* source,size_t bytes){require(t[i].data&&bytes<=span_bytes(t[i]),"copy exceeds operand");std::memcpy(t[i].data,source,bytes);}
    RadArgs args(){RadArgs a{};a.t=t.data();a.n_t=int(t.size());a.p=params.data();a.n_p=int(params.size());a.rank=rank;a.world_size=world;return a;}
};
struct DeviceBuffer {
    void* data=nullptr;
    explicit DeviceBuffer(size_t bytes){if(bytes){data=rad_dev_alloc(int64_t(bytes),RAD_MEM_DEVICE);require(data!=nullptr,"device allocation failed");}}
    ~DeviceBuffer(){if(data)rad_dev_free(data,RAD_MEM_DEVICE);}
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept:data(other.data){other.data=nullptr;}
};
struct Session {
    RadStream stream=nullptr;
    Session() {
        check(rad_dev_set(0),"select GPU 0");RadDeviceProps props{};check(rad_dev_props(0,&props),"GPU properties");
        require(!props.is_host_backend&&std::string(props.arch)=="gfx1201","GPU test requires a real gfx1201 device; no host fallback");
        check(rad_stream_create(&stream,0),"create engine stream");
    }
    ~Session(){if(stream)rad_stream_destroy(stream);}
};
float from_bf(uint16_t value){return rad_bf16_to_f32(value);}
void near(const float* actual,const float* expected,size_t count,float atol,float rtol,const std::string& label) {
    float worst=0;
    for(size_t i=0;i<count;++i) {
        if(actual[i]==expected[i])continue;
        const float delta=std::abs(actual[i]-expected[i]);worst=std::max(worst,delta);
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i])||delta>atol+rtol*std::abs(expected[i]))
            throw std::runtime_error(label+" mismatch at "+std::to_string(i)+": got "+std::to_string(actual[i])+", expected "+std::to_string(expected[i]));
    }
    std::printf("  FLOAT_PASS %s count=%zu worst_abs=%g\n",label.c_str(),count,worst);
}
struct Harness {
    Session session;
    rad::Registry registry;
    std::map<std::string,int> coverage;
    std::string oracle_path;
    /* Per-scope coverage gates: each pipeline reports its own before the run may print ALL_PASS. */
    bool ring_covered=false, full_covered=false;
    Harness(const char* mine,const char* oracle):oracle_path(oracle) {
        registry.set_code_device(0);check(registry.load_plugin(mine,0),"load mine");check(registry.load_plugin(oracle,1),"load mine_ref");
        check(registry.check_complete(),"complete plugin schemas");
        require(registry.plugin("mine")&&registry.plugin("mine_ref"),"expected mine and mine_ref plugin identities");
    }
    void run(Call& call) {
        const auto* schema=registry.schema(call.op);require(schema&&schema->n_operands==int(call.t.size()),"missing/mismatched op schema: "+call.op);
        rad::Geometry geometry;
        for(const auto& p:call.params) {
            if(p.kind==RAD_P_INT)geometry.set_i(p.key,p.ival);
            else if(p.kind==RAD_P_F64)geometry.set_f(p.key,p.dval);
            else if(p.kind==RAD_P_STR)geometry.set_s(p.key,p.sval);
            else throw std::runtime_error("unresolved test parameter");
        }
        std::string error;check(registry.validate(call.op,call.params.data(),int(call.params.size()),&error),"schema: "+error);
        rad::SelectionTrace why;
        auto device=registry.resolve(call.op,geometry,RAD_DOMAIN_DEVICE,&why);
        require(bool(device)&&device.row->plugin=="mine","missing mine device op: "+call.op+" "+why.reason);
        auto oracle=registry.resolve(call.op,geometry,RAD_DOMAIN_HOST,&why);
        require(bool(oracle)&&oracle.row->plugin=="mine_ref","missing mine_ref op: "+call.op+" "+why.reason);
        check(registry.init_instance(device,call.rank,call.world),"device instance");check(registry.init_instance(oracle,call.rank,call.world),"oracle instance");
        std::vector<DeviceBuffer> buffers;buffers.reserve(call.t.size());std::vector<RadTensor> dt=call.t;
        for(size_t i=0;i<call.t.size();++i) {
            const size_t bytes=call.t[i].data?span_bytes(call.t[i]):0;buffers.emplace_back(bytes);dt[i].data=buffers.back().data;
            if(bytes)check(rad_memcpy_async(dt[i].data,call.t[i].data,int64_t(bytes),session.stream),"upload operand");
        }
        check(rad_stream_sync(session.stream),"finish uploads");
        RadArgs host=call.args();host.p=oracle.geom.params();host.n_p=oracle.geom.n_params();host.instance=oracle.instance;
        const int64_t hs=oracle.row->info->scratch?oracle.row->info->scratch(&host):0;require(hs>=0,"negative reference scratch");
        std::vector<uint64_t> scratch((size_t(hs)+7)/8);host.scratch=scratch.data();host.scratch_bytes=hs;
        RadArgs gpu=call.args();gpu.t=dt.data();gpu.p=device.geom.params();gpu.n_p=device.geom.n_params();gpu.instance=device.instance;
        const int64_t ds=device.row->info->scratch?device.row->info->scratch(&gpu):0;require(ds>=0,"negative device scratch");
        DeviceBuffer device_scratch{size_t(ds)};gpu.scratch=device_scratch.data;gpu.scratch_bytes=ds;
        check(oracle.row->info->launch(&host,nullptr),"reference launch "+call.op);
        check(device.row->info->launch(&gpu,session.stream),"device launch "+call.op);check(rad_stream_sync(session.stream),"device completion "+call.op);
        std::vector<std::vector<uint64_t>> downloaded(call.t.size());
        /* dloop_base and dloop_full_base both end in a shard top-k whose cutoff is a tie set, so
         * their last two operands are downloaded and compared by that rule below. */
        const bool topk_base=call.op=="dloop_base"||call.op=="dloop_full_base";
        const int ids_operand=call.op=="dloop_base"?9:7,logits_operand=ids_operand+1;
        for(size_t i=0;i<call.t.size();++i) {
            const bool immutable_ple_history=call.op=="ngram_ids_tree"&&i==1;
            if(!call.t[i].data||(!immutable_ple_history&&schema->operands[i].role!=RAD_OPD_OUT&&schema->operands[i].role!=RAD_OPD_INOUT))continue;
            if(topk_base&&(int(i)==ids_operand||int(i)==logits_operand)) {
                const size_t bytes=span_bytes(call.t[i]);downloaded[i].resize((bytes+7)/8);
                check(rad_memcpy_async(downloaded[i].data(),dt[i].data,int64_t(bytes),session.stream),"download operand");check(rad_stream_sync(session.stream),"finish download");
                record_output(downloaded[i].data(),bytes);
                continue;   // compared together below: a top-k is a set with a cutoff
            }
            const size_t bytes=span_bytes(call.t[i]);std::vector<uint64_t> actual((bytes+7)/8);
            check(rad_memcpy_async(actual.data(),dt[i].data,int64_t(bytes),session.stream),"download operand");check(rad_stream_sync(session.stream),"finish download");
            record_output(actual.data(),bytes);
            const std::string name=call.op+"."+schema->operands[i].name;
            if(call.t[i].dtype==RAD_F32) {
                const bool ctx=call.op=="dloop_base"&&i==7;
                near(reinterpret_cast<const float*>(actual.data()),call.data<float>(int(i)),bytes/4,ctx?2e-5f:3e-5f,ctx?1e-5f:2e-5f,name);
            }
            else if(call.t[i].dtype==RAD_BF16&&(call.op.rfind("gdn_",0)==0||(call.op=="ple_conv_tree"&&i==8))) {
                // The GDN tree ops are declared verlustfrei-nicht-bitgenau (mine/tree_gdn_plan.txt):
                // device and host transcendentals differ in the last F32 bits, and the BF16 rounding
                // of the result may then land one BF16 ulp apart. More than one ulp is a mismatch.
                // PLE permits this bound only for y; saved raw inputs and both histories stay exact.
                const auto* a=reinterpret_cast<const uint16_t*>(actual.data());const auto* b=call.data<uint16_t>(int(i));
                const auto key=[](uint16_t x){return (x&0x8000)?-int(x&0x7fff):int(x&0x7fff);};
                size_t differ=0;int worst=0;
                for(size_t j=0;j<bytes/2;++j) {
                    if(a[j]==b[j])continue;
                    const float av=from_bf(a[j]),bv=from_bf(b[j]);const int ulp=std::abs(key(a[j])-key(b[j]));
                    ++differ;worst=std::max(worst,ulp);
                    require(std::isfinite(av)&&std::isfinite(bv)&&ulp<=1,name+" BF16 mismatch at "+std::to_string(j)+
                            " device="+std::to_string(av)+" host="+std::to_string(bv)+" ulp="+std::to_string(ulp));
                }
                std::printf("  BF16_PASS %s count=%zu differ=%zu worst_ulp=%d\n",name.c_str(),bytes/2,differ,worst);
            }else require(std::memcmp(actual.data(),call.t[i].data,bytes)==0,name+" discrete/bit-copy mismatch");
            // The next stage consumes DEVICE results, not the just-computed CPU oracle.
            std::memcpy(call.t[i].data,actual.data(),bytes);
        }
        if(topk_base)compare_topk(call,downloaded[ids_operand],downloaded[logits_operand],ids_operand,logits_operand);
        bench(call,device.row->info->launch,gpu);
        ++coverage[call.op];std::printf("OP_PASS %s rank=%d kernel=%s\n",call.op.c_str(),call.rank,device.row->info->name);
    }
    /* RADIANCE_DLOOP_BENCH=N: after the checked launch, time N more back-to-back device launches of
     * the same call (median of 5 batches, stream synced inside the timed region). Inputs are the ones
     * just checked; INOUT state keeps evolving across the repeats, which is fine for timing and is
     * never compared or passed on. Weights are reused, so a kernel whose weights fit in the last-level
     * cache reads them from cache after the first launch: an upper bound on speed, as rad-kbench says. */
    template<class Launch>void bench(const Call& call,Launch launch,RadArgs& gpu) {
        static const int iters=[]{const char* v=std::getenv("RADIANCE_DLOOP_BENCH");return v?std::atoi(v):0;}();
        if(iters<=0)return;
        std::vector<double> ms;
        for(int rep=0;rep<5;++rep) {
            const auto t0=std::chrono::steady_clock::now();
            for(int i=0;i<iters;++i)check(launch(&gpu,session.stream),"bench launch "+call.op);
            check(rad_stream_sync(session.stream),"bench completion "+call.op);
            ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/iters);
        }
        std::sort(ms.begin(),ms.end());
        std::printf("BENCH %s rank=%d n_seq=%lld n_head_k=%lld n_head_v=%lld median_ms=%.4f min_ms=%.4f\n",call.op.c_str(),call.rank,
                    (long long)param_i(call,"n_seq"),(long long)param_i(call,"n_head_k"),(long long)param_i(call,"n_head_v"),ms[2],ms[0]);
    }
    static int64_t param_i(const Call& call,const char* key) {
        for(const auto& p:call.params)if(p.kind==RAD_P_INT&&std::strcmp(p.key,key)==0)return p.ival;
        return -1;
    }
    /* Timing-only sibling of run(): resolves and launches the device row, uploads the
     * operands once and leaves them on the device across the timed repeats. No reference
     * launch and no comparison, so a BENCH case may cover a geometry no golden carries. */
    void run_device_only(Call& call) {
        const auto* schema=registry.schema(call.op);require(schema&&schema->n_operands==int(call.t.size()),"missing/mismatched op schema: "+call.op);
        rad::Geometry geometry;
        for(const auto& p:call.params) {
            if(p.kind==RAD_P_INT)geometry.set_i(p.key,p.ival);
            else if(p.kind==RAD_P_F64)geometry.set_f(p.key,p.dval);
            else if(p.kind==RAD_P_STR)geometry.set_s(p.key,p.sval);
            else throw std::runtime_error("unresolved test parameter");
        }
        std::string error;check(registry.validate(call.op,call.params.data(),int(call.params.size()),&error),"schema: "+error);
        rad::SelectionTrace why;
        auto device=registry.resolve(call.op,geometry,RAD_DOMAIN_DEVICE,&why);
        require(bool(device)&&device.row->plugin=="mine","missing mine device op: "+call.op+" "+why.reason);
        check(registry.init_instance(device,call.rank,call.world),"device instance");
        std::vector<DeviceBuffer> buffers;buffers.reserve(call.t.size());std::vector<RadTensor> dt=call.t;
        for(size_t i=0;i<call.t.size();++i) {
            const size_t bytes=call.t[i].data?span_bytes(call.t[i]):0;buffers.emplace_back(bytes);dt[i].data=buffers.back().data;
            if(bytes)check(rad_memcpy_async(dt[i].data,call.t[i].data,int64_t(bytes),session.stream),"upload operand");
        }
        check(rad_stream_sync(session.stream),"finish uploads");
        RadArgs gpu=call.args();gpu.t=dt.data();gpu.p=device.geom.params();gpu.n_p=device.geom.n_params();gpu.instance=device.instance;
        const int64_t ds=device.row->info->scratch?device.row->info->scratch(&gpu):0;require(ds>=0,"negative device scratch");
        DeviceBuffer device_scratch{size_t(ds)};gpu.scratch=device_scratch.data;gpu.scratch_bytes=ds;
        check(device.row->info->launch(&gpu,session.stream),"device launch "+call.op);
        check(rad_stream_sync(session.stream),"device completion "+call.op);
        if(std::getenv("RADIANCE_DLOOP_OUTPUT_RECORD")) {
            for(size_t i=0;i<call.t.size();++i) {
                if(!call.t[i].data||(schema->operands[i].role!=RAD_OPD_OUT&&schema->operands[i].role!=RAD_OPD_INOUT))continue;
                const size_t bytes=span_bytes(call.t[i]);std::vector<uint64_t> actual((bytes+7)/8);
                check(rad_memcpy_async(actual.data(),dt[i].data,int64_t(bytes),session.stream),"download benchmark output");
                check(rad_stream_sync(session.stream),"finish benchmark output download");
                record_output(actual.data(),bytes);
            }
        }
        bench(call,device.row->info->launch,gpu);
    }

    /* A refusal check for ops whose device kernel gates on metadata the host can also see:
     * the DEVICE launch must happen (the host oracle would refuse first and prove nothing), must
     * not fault, and must leave every OUT/INOUT byte identical to what was uploaded. This is the
     * vehicle for malformed GDN tree metadata: same schema/resolve path as run_device_only, plus
     * the sentinel download run_device_only never does. */
    void run_refused(Call& call,const char* what,bool invalid_capacity=false) {
        const auto* schema=registry.schema(call.op);require(schema&&schema->n_operands==int(call.t.size()),"missing/mismatched op schema: "+call.op);
        rad::Geometry geometry;
        for(const auto& p:call.params) {
            if(p.kind==RAD_P_INT)geometry.set_i(p.key,p.ival);
            else if(p.kind==RAD_P_F64)geometry.set_f(p.key,p.dval);
            else if(p.kind==RAD_P_STR)geometry.set_s(p.key,p.sval);
            else throw std::runtime_error("unresolved test parameter");
        }
        std::string error;check(registry.validate(call.op,call.params.data(),int(call.params.size()),&error),"schema: "+error);
        rad::SelectionTrace why;
        auto device=registry.resolve(call.op,geometry,RAD_DOMAIN_DEVICE,&why);
        require(bool(device)&&device.row->plugin=="mine","missing mine device op: "+call.op+" "+why.reason);
        check(registry.init_instance(device,call.rank,call.world),"device instance");
        std::vector<DeviceBuffer> buffers;buffers.reserve(call.t.size());std::vector<RadTensor> dt=call.t;
        for(size_t i=0;i<call.t.size();++i) {
            const size_t bytes=call.t[i].data?span_bytes(call.t[i]):0;buffers.emplace_back(bytes);dt[i].data=buffers.back().data;
            if(bytes)check(rad_memcpy_async(dt[i].data,call.t[i].data,int64_t(bytes),session.stream),"upload operand");
        }
        check(rad_stream_sync(session.stream),"finish uploads");
        RadArgs gpu=call.args();gpu.t=dt.data();gpu.p=device.geom.params();gpu.n_p=device.geom.n_params();gpu.instance=device.instance;
        const int64_t ds=device.row->info->scratch?device.row->info->scratch(&gpu):0;require(ds>=0,"negative device scratch");
        std::vector<RadParam> invalid_params;
        if(invalid_capacity) {
            invalid_params.assign(gpu.p,gpu.p+gpu.n_p);
            for(auto& param:invalid_params)if(!std::strcmp(param.key,"node_capacity"))param.ival=66;
            gpu.p=invalid_params.data();
        }
        DeviceBuffer device_scratch{size_t(ds)};gpu.scratch=device_scratch.data;gpu.scratch_bytes=ds;
        const int status=device.row->info->launch(&gpu,session.stream);
        require(status==(invalid_capacity?RAD_E_UNSUPPORTED:RAD_OK),std::string(what)+": unexpected refusal status "+std::to_string(status));
        check(rad_stream_sync(session.stream),"device completion "+call.op);
        const bool ple=call.op=="ngram_ids_tree"||call.op=="ple_conv_tree"||call.op=="ple_tree_commit";
        for(size_t i=0;i<call.t.size();++i) {
            if(!call.t[i].data||(!ple&&schema->operands[i].role!=RAD_OPD_OUT&&schema->operands[i].role!=RAD_OPD_INOUT))continue;
            const size_t bytes=span_bytes(call.t[i]);std::vector<uint64_t> actual((bytes+7)/8);
            check(rad_memcpy_async(actual.data(),dt[i].data,int64_t(bytes),session.stream),"download operand");check(rad_stream_sync(session.stream),"finish download");
            require(std::memcmp(actual.data(),call.t[i].data,bytes)==0,
                    std::string(what)+": refused launch touched an operand byte");
        }
        std::printf("REFUSAL_PASS %s rank=%d status=%d (%s untouched)\n",what,call.rank,status,ple?"all operands":"outputs");
    }
    /* dloop_base's local top-K over F32 logits that device and host sum in different orders. Common
     * IDs must agree within the float bound. Two candidates whose logits sit within that bound of the
     * row's K-th logit may trade places across the cutoff; anything else in the set differing is a
     * mismatch. The next stage then consumes the device's set, as for every other operand. */
    static void compare_topk(Call& call,const std::vector<uint64_t>& dev_ids_raw,const std::vector<uint64_t>& dev_logits_raw,
                             int ids_operand,int logits_operand) {
        const auto* di=reinterpret_cast<const int32_t*>(dev_ids_raw.data());const auto* dl=reinterpret_cast<const float*>(dev_logits_raw.data());
        const auto* hi=call.data<int32_t>(ids_operand);const auto* hl=call.data<float>(logits_operand);
        const size_t total=span_bytes(call.t[ids_operand])/4,k=size_t(call.t[ids_operand].shape[call.t[ids_operand].rank-1]),rows=total/k;
        compare_topk_sets(call.op,hi,hl,di,dl,rows,k);
        std::memcpy(call.t[ids_operand].data,dev_ids_raw.data(),total*4);std::memcpy(call.t[logits_operand].data,dev_logits_raw.data(),total*4);
    }
    /* The cutoff-set rule itself, on two already-extracted (ids, logits) planes. `hi`/`hl` is the
     * reference side, `di`/`dl` the side under test. */
    static void compare_topk_sets(const std::string& op,const int32_t* hi,const float* hl,
                                  const int32_t* di,const float* dl,size_t rows,size_t k) {
        const auto bound=[](float x){return 3e-5f+2e-5f*std::abs(x);};
        size_t tie_rows=0,swapped=0;float worst=0;
        for(size_t r=0;r<rows;++r) {
            std::map<int32_t,float> host,dev;
            for(size_t j=0;j<k;++j){host[hi[r*k+j]]=hl[r*k+j];dev[di[r*k+j]]=dl[r*k+j];}
            const float cut_h=hl[r*k+k-1],cut_d=dl[r*k+k-1];bool tie=false;
            for(const auto& [id,v]:host) {
                const auto it=dev.find(id);
                if(it!=dev.end()) {
                    const float delta=std::abs(it->second-v);worst=std::max(worst,delta);
                    require(std::isfinite(it->second)&&delta<=bound(v),op+".local_logits mismatch row "+std::to_string(r)+" id "+std::to_string(id));
                }else {
                    require(std::abs(v-cut_h)<=bound(cut_h)&&std::abs(v-cut_d)<=2*bound(cut_h),op+".local_ids mismatch row "+std::to_string(r)+": reference id "+std::to_string(id)+" logit "+std::to_string(v)+" not at the cutoff "+std::to_string(cut_h));
                    tie=true;++swapped;
                }
            }
            for(const auto& [id,v]:dev)if(!host.count(id))
                require(std::abs(v-cut_d)<=bound(cut_d)&&std::abs(v-cut_h)<=2*bound(cut_d),op+".local_ids mismatch row "+std::to_string(r)+": candidate id "+std::to_string(id)+" logit "+std::to_string(v)+" not at the cutoff "+std::to_string(cut_d));
            tie_rows+=tie;
        }
        std::printf("  TOPK_PASS %s.local_ids rows=%zu cutoff_tie_rows=%zu swapped=%zu worst_abs=%g\n",op.c_str(),rows,tie_rows,swapped,worst);
    }
    /* One shard of a full-context base against its golden, by the same rule. Both sides go through a
     * GPU round trip, so neither is bit-exact: agreeing on the *set* inside the float bound is the
     * claim the op can actually make. */
    static void compare_topk_golden(const std::string& op,const Call& base,int ids_operand,int logits_operand,
                                    const int32_t* gids,const float* glogits) {
        const size_t total=span_bytes(base.t[ids_operand])/4;
        const size_t k=size_t(base.t[ids_operand].shape[base.t[ids_operand].rank-1]),rows=total/k;
        compare_topk_sets(op,gids,glogits,base.data<int32_t>(ids_operand),base.data<float>(logits_operand),rows,k);
    }
    /* The pool's own selection: `K` retained ids per row, sorted by token id (dloop_ops.h), so the
     * cutoff is not the last column. Two planes agree when every common id carries the same unary
     * value within the float bound AND no id kept on one side alone scores above the other side's
     * lowest retained value -- the latter is exactly what would make the retained SET differ. */
    static void compare_pool_golden(const Call& pool,int ids_operand,int logits_operand,
                                    const int32_t* gids,const float* glogits) {
        const auto* pi=pool.data<int32_t>(ids_operand);const auto* pv=pool.data<float>(logits_operand);
        const size_t total=span_bytes(pool.t[ids_operand])/4;
        const size_t k=size_t(pool.t[ids_operand].shape[pool.t[ids_operand].rank-1]),rows=total/k;
        const auto bound=[](float x){return 3e-5f+2e-5f*std::abs(x);};
        float worst=0;size_t extra=0;
        for(size_t r=0;r<rows;++r) {
            std::map<int32_t,float> gold,dev;
            for(size_t j=0;j<k;++j){gold[gids[r*k+j]]=glogits[r*k+j];dev[pi[r*k+j]]=pv[r*k+j];}
            float floor_gold=INFINITY,floor_dev=INFINITY;
            for(const auto& [id,v]:gold)floor_gold=std::min(floor_gold,v);
            for(const auto& [id,v]:dev)floor_dev=std::min(floor_dev,v);
            for(const auto& [id,v]:gold) {
                const auto it=dev.find(id);
                if(it!=dev.end()) {
                    const float delta=std::abs(it->second-v);worst=std::max(worst,delta);
                    require(std::isfinite(it->second)&&delta<=bound(v),
                            "pool.local_logits mismatch row "+std::to_string(r)+" id "+std::to_string(id));
                }else {
                    require(v<=floor_dev+bound(floor_dev),
                            "pool.local_ids mismatch row "+std::to_string(r)+": golden id "+std::to_string(id)+
                            " scores "+std::to_string(v)+", above the retained floor "+std::to_string(floor_dev));
                    ++extra;
                }
            }
            for(const auto& [id,v]:dev)if(!gold.count(id)) {
                require(v<=floor_gold+bound(floor_gold),
                        "pool.local_ids mismatch row "+std::to_string(r)+": device id "+std::to_string(id)+
                        " scores "+std::to_string(v)+", above the golden floor "+std::to_string(floor_gold));
                ++extra;
            }
        }
        std::printf("  POOL_SET_PASS rows=%zu cutoff_swapped=%zu worst_abs=%g\n",rows,extra,worst);
    }
    /* Resolve, upload and launch the device row with NO reference launch, then require the launch to
     * refuse and leave every mutable operand byte-identical. The download happens BEFORE the status
     * check, because "it returned an error" and "it changed nothing" are two different promises and
     * a kernel can keep only one of them. This is the transaction dloop_full.h states: a malformed
     * slot, path, span or a capacity overflow leaves every operand untouched.
     *
     * A refusal the DECLARATIVE validator makes is just as good and is accepted here: the operand
     * carries an index extent, so a slot past max_seqs is refused before any kernel runs, and that
     * path must not be reported as a success either. */
    void expect_refused(Call& call) {
        const auto* schema=registry.schema(call.op);require(schema&&schema->n_operands==int(call.t.size()),"missing/mismatched op schema: "+call.op);
        std::vector<uint64_t> before(call.t.size());
        for(size_t i=0;i<call.t.size();++i)
            if(call.t[i].data&&(schema->operands[i].role==RAD_OPD_OUT||schema->operands[i].role==RAD_OPD_INOUT)) {
                const size_t bytes=span_bytes(call.t[i]);
                before[i]=bytes;   // snapshot in place below; the host operand is the input here
            }
        std::string error;
        if(registry.validate(call.op,call.params.data(),int(call.params.size()),&error)!=RAD_OK) {
            require(!error.empty(),"an invalid call must name the reason it was refused");
            std::printf("  FULL_APPEND_REFUSED_PASS declarative: %s\n",error.c_str());
            return;
        }
        rad::Geometry geometry;
        for(const auto& p:call.params) {
            if(p.kind==RAD_P_INT)geometry.set_i(p.key,p.ival);
            else if(p.kind==RAD_P_F64)geometry.set_f(p.key,p.dval);
            else if(p.kind==RAD_P_STR)geometry.set_s(p.key,p.sval);
        }
        rad::SelectionTrace why;
        auto device=registry.resolve(call.op,geometry,RAD_DOMAIN_DEVICE,&why);
        require(bool(device)&&device.row->plugin=="mine","missing mine device op: "+call.op+" "+why.reason);
        check(registry.init_instance(device,call.rank,call.world),"device instance");
        std::vector<DeviceBuffer> buffers;buffers.reserve(call.t.size());std::vector<RadTensor> dt=call.t;
        std::vector<std::vector<uint64_t>> host_before(call.t.size());
        for(size_t i=0;i<call.t.size();++i) {
            const size_t bytes=call.t[i].data?span_bytes(call.t[i]):0;buffers.emplace_back(bytes);dt[i].data=buffers.back().data;
            if(before[i])
                host_before[i].assign(reinterpret_cast<const uint64_t*>(call.t[i].data),
                                      reinterpret_cast<const uint64_t*>(call.t[i].data)+(before[i]+7)/8);
            if(bytes)check(rad_memcpy_async(dt[i].data,call.t[i].data,int64_t(bytes),session.stream),"upload operand");
        }
        check(rad_stream_sync(session.stream),"finish uploads");
        RadArgs gpu=call.args();gpu.t=dt.data();gpu.p=device.geom.params();gpu.n_p=device.geom.n_params();gpu.instance=device.instance;
        const int64_t ds=device.row->info->scratch?device.row->info->scratch(&gpu):0;require(ds>=0,"negative device scratch");
        DeviceBuffer device_scratch{size_t(ds)};gpu.scratch=device_scratch.data;gpu.scratch_bytes=ds;
        const int status=device.row->info->launch(&gpu,session.stream);
        check(rad_stream_sync(session.stream),"finish refused launch "+call.op);
        require(status!=RAD_OK,"device op accepted an invalid call: "+call.op);
        for(size_t i=0;i<call.t.size();++i) {
            const bool mutable_operand=schema->operands[i].role==RAD_OPD_OUT||schema->operands[i].role==RAD_OPD_INOUT;
            if(!mutable_operand||!call.t[i].data)continue;
            const size_t bytes=span_bytes(call.t[i]);std::vector<uint64_t> actual((bytes+7)/8);
            check(rad_memcpy_async(actual.data(),dt[i].data,int64_t(bytes),session.stream),"download refused operand");
            check(rad_stream_sync(session.stream),"finish refused download");
            require(actual.size()>=host_before[i].size(),"refusal download shorter than the operand");
            require(std::memcmp(actual.data(),host_before[i].data(),host_before[i].size()*8)==0,
                    "refused "+call.op+" mutated operand '"+schema->operands[i].name+"'");
        }
        std::printf("  FULL_APPEND_REFUSED_PASS device\n");
    }
    /* Every op a scope is supposed to cover. A run that silently skipped one of the three new
     * full-context ops is a failure, not a smaller pass: they are the contract this file exists for,
     * so ALL_PASS must not be reachable without them. The ring list stays what it was, and the full
     * list is separate rather than merged, because the two contracts are exercised by two passes
     * over two different fixtures. */
    void require_coverage(bool full) {
        const char* ring[]={"dloop_ctx_append","dloop_base","dloop_pool","dloop_dartree",
            "gdn_conv_tree","gdn_recurrent_tree","gdn_tree_commit",
            "ngram_ids_tree","ple_conv_tree","ple_tree_commit"};
        const char* full_ops[]={"dloop_full_append","dloop_full_base","dloop_full_dartree",
            "dloop_pool","gdn_conv_tree","gdn_recurrent_tree","gdn_tree_commit","ple_conv_tree","ple_tree_commit"};
        const char* const* ops=full?full_ops:ring;
        const size_t count=full?sizeof(full_ops)/sizeof(full_ops[0]):sizeof(ring)/sizeof(ring[0]);
        for(size_t i=0;i<count;++i){auto it=coverage.find(ops[i]);require(it!=coverage.end()&&it->second>0,std::string("skipped op: ")+ops[i]);}
        (full?full_covered:ring_covered)=true;
    }
    /* Both scopes must have reported their own coverage before ALL_PASS: a full-context run that
     * never exercised the ring ops would otherwise pass by simply not looking at them. */
    void require_both_scopes()const {
        require(ring_covered&&full_covered,"both the ring and the full-context pipelines must report coverage");
    }
};
const rad::StTensor& get(const rad::SafeTensorsFile& file,const std::string& name) {
    const auto* t=file.find(name);require(t!=nullptr,"missing tensor "+name+" in "+file.path());return *t;
}
struct Fixture {
    rad::SafeTensorsFile student,teacher,golden;
    RadDLoopConfig c{};std::vector<uint16_t> packed;
    Fixture(const std::string& golden_dir,const std::string& sidecar_dir,const std::string& teacher_file) {
        check(student.open(sidecar_dir+"/dloop_sidecar.safetensors"),"open student");check(teacher.open(teacher_file),"open teacher");
        check(golden.open(golden_dir+"/golden.safetensors"),"open golden");
        std::ifstream manifest(sidecar_dir+"/dloop_sidecar.json");require(bool(manifest),"missing sidecar JSON");nlohmann::json j;manifest>>j;const auto& cfg=j.at("config");
#define C(k) c.k=cfg.at(#k).get<int>();
        C(hidden_size);C(intermediate_size);C(num_layers);C(num_q_heads);C(num_kv_heads);C(head_dim);C(context_features_dim);C(target_width);
        C(proposal_length);C(context_window);C(active_vocab);C(vocab_size);C(selector_rank);C(conv_kernel_size);C(conv_group_size);
#undef C
        c.eps=cfg.at("eps").get<float>();c.rope_theta=cfg.at("rope_theta").get<float>();c.depth_bonus=-.2f;
        const auto& ctx=get(golden,"context");require(ctx.dtype==RAD_BF16&&ctx.shape.size()==3,"invalid context tensor");
        c.n_seq=int(ctx.shape[0]);c.max_seqs=c.n_seq;c.tap_rows=c.n_seq*c.context_window;c.world_size=2;
        c.vocab_size_shard=c.vocab_size/2;c.pool_size=64;c.budget=15;c.width=12;
        require(ctx.shape[1]==c.context_window&&ctx.shape[2]==c.context_features_dim&&c.n_seq>0,"context/config mismatch");
        const auto& emb=get(teacher,"model.language_model.embed_tokens.weight");const auto& head=get(teacher,"lm_head.weight");
        require(emb.dtype==RAD_BF16&&head.dtype==RAD_BF16&&emb.shape==std::vector<int64_t>{c.vocab_size,c.target_width}&&head.shape==emb.shape,"teacher geometry mismatch");
        auto expected=[&](const rad::SafeTensorsFile& file,const std::string& name,uint32_t dtype,std::initializer_list<int64_t> shape) {
            const auto& t=get(file,name);require(t.dtype==dtype&&t.shape==std::vector<int64_t>(shape),"fixture shape/dtype mismatch: "+name);
        };
        expected(golden,"anchors",RAD_I64,{c.n_seq});expected(golden,"context_lengths",RAD_I32,{c.n_seq});
        expected(golden,"ctxh",RAD_F32,{c.n_seq,c.context_window,c.hidden_size});
        expected(golden,"y_base",RAD_F32,{c.n_seq,c.proposal_length,c.target_width});
        expected(golden,"pool_ids",RAD_I64,{c.n_seq,c.proposal_length,c.pool_size});
        expected(golden,"pool_unary",RAD_F32,{c.n_seq,c.proposal_length,c.pool_size});
        for(const char* variant:{"fixed","pruned"}) {
            expected(golden,std::string(variant)+".n_nodes",RAD_I32,{c.n_seq});
            for(const char* field:{"tokens","parents","depths"})expected(golden,std::string(variant)+"."+field,RAD_I64,{c.n_seq,16});
        }
        expected(student,"dloop.selector.pred_codebook.weight",RAD_BF16,{c.vocab_size,c.selector_rank});
        expected(student,"dloop.selector.succ_codebook.weight",RAD_BF16,{c.vocab_size,c.selector_rank});
        packed.resize(size_t(rad_dloop_weight_elements(c)));
        auto add=[&](int w,int layer=0) {
            std::string name="dloop.";if(w>=DL_INPUT_NORM&&w<=DL_DOWN_PROJ)name+="layers."+std::to_string(layer)+".";name+=rad_dloop_weight_name(w);
            const auto& t=get(student,name);require(t.dtype==RAD_BF16&&t.numel()==rad_dloop_weight_size(c,w),"packed weight mismatch: "+name);
            std::memcpy(packed.data()+rad_dloop_weight_offset(c,w,layer),t.data,t.bytes);
        };
        for(int w=0;w<=DL_MASK_TOKEN;++w)add(w);
        for(int layer=0;layer<c.num_layers;++layer)for(int w=DL_INPUT_NORM;w<=DL_DOWN_PROJ;++w)add(w,layer);
        for(int w=DL_FINAL_NORM;w<=DL_SELECTOR_PROJ;++w)add(w);
    }
    template<class T>const T* g(const std::string& name)const{return reinterpret_cast<const T*>(get(golden,name).data);}
    const uint16_t* table(const char* name)const{return reinterpret_cast<const uint16_t*>(get(teacher,name).data);}
    void weights(Call& call,int operand){call.bind(operand,packed.data(),RAD_BF16,{int64_t(packed.size())});}
    void base(Call& call,const Call& append) {
        call.t[0]=append.t[4];call.t[1]=append.t[5];call.t[2]=append.t[6];call.t[3]=append.t[2];weights(call,5);
        call.bind(6,const_cast<uint16_t*>(table("lm_head.weight"))+int64_t(call.rank)*c.vocab_size_shard*c.target_width,RAD_BF16,{c.vocab_size_shard,c.target_width});
        for(int s=0;s<c.n_seq;++s) {
            const int64_t anchor=g<int64_t>("anchors")[s];require(anchor>=0&&anchor<c.active_vocab,"golden anchor out of range");
            std::memcpy(call.data<uint16_t>(4)+s*c.target_width,table("model.language_model.embed_tokens.weight")+anchor*c.target_width,size_t(c.target_width)*2);
        }
    }
    void pool(Call& call,const Call& base0,const Call* base1) {
        const int rows=c.n_seq*c.proposal_length*c.pool_size;
        call.copy(0,base0.t[9].data,size_t(rows)*4);call.copy(1,base0.t[10].data,size_t(rows)*4);
        if(base1){std::memcpy(call.data<int32_t>(0)+rows,base1->t[9].data,size_t(rows)*4);std::memcpy(call.data<float>(1)+rows,base1->t[10].data,size_t(rows)*4);}
        else {std::fill(call.data<int32_t>(0)+rows,call.data<int32_t>(0)+2*rows,-1);std::fill(call.data<float>(1)+rows,call.data<float>(1)+2*rows,-INFINITY);}
        for(int s=0;s<c.n_seq;++s)call.data<int32_t>(2)[s]=int32_t(g<int64_t>("anchors")[s]);
        const int lo=call.rank*c.vocab_size_shard;
        call.bind(3,const_cast<uint16_t*>(table("lm_head.weight"))+int64_t(lo)*c.target_width,RAD_BF16,{c.vocab_size_shard,c.target_width});
        call.bind(4,const_cast<uint16_t*>(table("model.language_model.embed_tokens.weight"))+int64_t(lo)*c.target_width,RAD_BF16,{c.vocab_size_shard,c.target_width});
        call.bind(5,const_cast<uint8_t*>(get(student,"dloop.selector.pred_codebook.weight").data)+int64_t(lo)*c.selector_rank*2,RAD_BF16,{c.vocab_size_shard,c.selector_rank});
        call.bind(6,const_cast<uint8_t*>(get(student,"dloop.selector.succ_codebook.weight").data)+int64_t(lo)*c.selector_rank*2,RAD_BF16,{c.vocab_size_shard,c.selector_rank});
    }
};
void check_tree_golden(Fixture& f,Call& tree,const std::string& variant) {
    const auto& c=f.c;
    for(int s=0;s<c.n_seq;++s) {
        const int count=tree.data<int32_t>(15)[s];require(count==f.g<int32_t>(variant+".n_nodes")[s]&&count<=15,"golden tree count mismatch");
        int depth[16]{};const int* packed=tree.data<int32_t>(17)+s*33;
        require(packed[0]==count+1&&packed[1]==f.g<int64_t>("anchors")[s]&&packed[17]==-1,"packed root mismatch");
        for(int i=0;i<count;++i) {
            const int parent=tree.data<int32_t>(14)[s*16+i],token=tree.data<int32_t>(13)[s*16+i];require(parent>=-1&&parent<i,"invalid tree parent");
            depth[i]=parent<0?1:depth[parent]+1;
            require(token==f.g<int64_t>(variant+".tokens")[s*16+i+1]&&parent==f.g<int64_t>(variant+".parents")[s*16+i+1]-1&&
                depth[i]==f.g<int64_t>(variant+".depths")[s*16+i+1],"golden tree token/parent/depth mismatch");
            require(packed[2+i]==token&&packed[18+i]==parent+1,"packed draft mismatch");
        }
        for(int i=count;i<16;++i)require(tree.data<int32_t>(13)[s*16+i]==-1&&tree.data<int32_t>(14)[s*16+i]==-1,"tree padding mismatch");
    }
}
/* ============================================================ the full-context fixture
 *
 * One directory, two contracts: `tests/fixtures/dloop_tiny` keeps the two-layer/ring ops it already
 * has, and `tests/fixtures/dloop_full_tiny` carries the five-layer full-context one -- a version-2
 * manifest, the same student/teacher files, and `full.safetensors` with operands AND expectations.
 * The generator is scripts/generate_dloop_full_fixture.py; nothing here reads a real checkpoint.
 *
 * Every binding below comes from the manifest's own config and from `full.json`, so a fixture whose
 * geometry has drifted from its expectations fails at load rather than at comparison. */
struct FullFixture {
    rad::SafeTensorsFile student,teacher,full;
    RadDLoopConfig c{};int capacity=0;std::vector<uint16_t> packed;
    /* One recorded non-last root-to-leaf chain per sequence that has one; `sequence` indexes the
     * fixture's own sequence order, which is the parent array the chain must be walked through. */
    struct VerifyPath { int sequence;std::vector<int32_t> path; };
    std::vector<VerifyPath> verify_paths;
    /* True for a REAL L5 fixture (`mode: "real"` in full.json): the recorded tap planes are the
     * full prefixes of real documents, so the append plan and the tap row count come from the
     * recording rather than from a synthetic four-chunk plan. */
    bool real_mode=false;
    FullFixture(const std::string& dir,const std::string& sidecar_dir,const std::string& teacher_file) {
        check(student.open(sidecar_dir+"/dloop_sidecar.safetensors"),"open full student");
        check(teacher.open(teacher_file),"open full teacher");
        check(full.open(dir+"/full.safetensors"),"open full goldens");
        std::ifstream manifest(sidecar_dir+"/dloop_sidecar.json");require(bool(manifest),"missing full sidecar JSON");
        nlohmann::json j;manifest>>j;
        require(j.at("version").get<int>()==2,"full fixture requires the version-2 sidecar contract");
        /* `source_promoted` is a PROVENANCE FACT, not a gate: the loader accepts either value, and a
         * promoted artifact is exactly what a serving run is supposed to load. All this harness
         * needs is that the key is present and boolean. */
        require(j.at("source_promoted").is_boolean(),"source_promoted must be a boolean");
        const auto& cfg=j.at("config");
        require(cfg.at("context_window").get<int>()==0,"full fixture requires context_window 0");
#define C(k) c.k=cfg.at(#k).get<int>();
        C(hidden_size);C(intermediate_size);C(num_layers);C(num_q_heads);C(num_kv_heads);C(head_dim);
        C(context_features_dim);C(target_width);C(proposal_length);C(context_window);C(active_vocab);
        C(vocab_size);C(selector_rank);C(conv_kernel_size);C(conv_group_size);
#undef C
        capacity=j.at("context_capacity").get<int>();
        c.eps=cfg.at("eps").get<float>();c.rope_theta=cfg.at("rope_theta").get<float>();c.depth_bonus=-.2f;
        /* The drive plan is the fixture's own: the sequence count comes from the span plane, and the
         * other planes are checked against it below rather than against a number spelled here. A
         * REAL fixture carries no chunk planes -- its taps ARE the recorded full prefix of each
         * document, so the pipeline compares the cache directly instead of replaying an append plan. */
        std::ifstream meta(dir+"/full.json");require(bool(meta),"missing full.json");nlohmann::json m;meta>>m;
        real_mode=m.value("mode","tiny")=="real";
        const auto& tap0=get(full,"tap0");
        require(tap0.dtype==RAD_BF16&&tap0.shape.size()==2&&tap0.shape[1]==c.target_width,
                "full tap plane geometry mismatch");
        /* The planes are rank 2 in both modes, so `tap_rows` is simply their row count: the tiny
         * fixture's capacity plane or the REAL one's concatenated prefixes. The sequence count comes
         * from the plane that states it -- the chunk span for the tiny fixture, the lengths for the
         * real one -- and each mode's own shape check below confirms that choice. */
        c.tap_rows=int(tap0.shape[0]);
        c.n_seq=real_mode?int(get(full,"length").shape[0]):int(get(full,"chunk_span").shape[1]);
        require((real_mode?get(full,"length").dtype==RAD_I32:get(full,"chunk_span").dtype==RAD_I32),
                "the fixture's sequence-count plane has the wrong dtype");
        c.max_seqs=c.n_seq;c.world_size=2;c.pool_size=64;c.budget=15;c.width=12;
        c.vocab_size_shard=c.vocab_size/2;
        require(c.n_seq>0&&c.tap_rows>=1&&capacity>=1,"full fixture geometry is empty");
        /* THE GEOMETRY THE DEVICE ROWS ARE GATED ON, asserted for the TINY fixture only: the REAL
         * fixture is the production geometry, which has its own (production) rows, and the geometry
         * check below is the one that matters there. */
        if(!real_mode) {
            const struct { const char* name;int actual,wanted; } geometry[] = {
                {"hidden_size",c.hidden_size,64},{"intermediate_size",c.intermediate_size,128},
                {"num_q_heads",c.num_q_heads,4},{"num_kv_heads",c.num_kv_heads,2},{"head_dim",c.head_dim,16},
                {"context_features_dim",c.context_features_dim,96},{"target_width",c.target_width,32},
                {"selector_rank",c.selector_rank,8},{"vocab_size",c.vocab_size,512},
                {"active_vocab",c.active_vocab,509}};
            for(const auto& entry:geometry)
                require(entry.actual==entry.wanted,
                        std::string("full fixture geometry disagrees with the published tiny row: ")+entry.name);
        } else {
            /* The REAL rows in mine/dloop_full.cpp gate the serving geometry: L5, H768, FF2304,
             * Q12/KV4/Dh64, F7680, W2560, R256, vocab 248320/248077. A real fixture that disagrees
             * would resolve to no row at all, and the message would not say why. */
            const struct { const char* name;int actual,wanted; } geometry[] = {
                {"hidden_size",c.hidden_size,768},{"intermediate_size",c.intermediate_size,2304},
                {"num_layers",c.num_layers,5},
                {"num_q_heads",c.num_q_heads,12},{"num_kv_heads",c.num_kv_heads,4},{"head_dim",c.head_dim,64},
                {"context_features_dim",c.context_features_dim,7680},{"target_width",c.target_width,2560},
                {"selector_rank",c.selector_rank,256},{"vocab_size",c.vocab_size,248320},
                {"active_vocab",c.active_vocab,248077}};
            for(const auto& entry:geometry)
                require(entry.actual==entry.wanted,
                        std::string("real fixture geometry disagrees with the production row: ")+entry.name);
            /* The >32 requirement: the point of the full-context contract is a prefix no ring
             * fixture could hold, so a real fixture with a capacity at or below the ring window
             * would prove nothing. */
            require(capacity>32,"a real full-context fixture must exceed the ring window of 32");
        }
        require(c.vocab_size_shard==c.vocab_size/2,"full fixture requires a two-way vocabulary shard");
        auto expected=[&](const rad::SafeTensorsFile& file,const std::string& name,uint32_t dtype,
                          std::initializer_list<int64_t> shape) {
            const auto& t=get(file,name);
            require(t.dtype==dtype&&t.shape==std::vector<int64_t>(shape),"full fixture shape/dtype mismatch: "+name);
        };
        /* The tap planes are ALWAYS rank 2, BF16 `[rows, W]` -- that is the op's operand, and a
         * padded `[B, C, F]` block would be the wrong shape. What differs is the row count: the tiny
         * fixture is a capacity plane, the REAL one is the concatenated prefixes. */
        expected(full,"tap0",RAD_BF16,{c.tap_rows,c.target_width});
        expected(full,"tap1",RAD_BF16,{c.tap_rows,c.target_width});
        expected(full,"tap2",RAD_BF16,{c.tap_rows,c.target_width});
        expected(full,"anchors",RAD_I64,{c.n_seq});
        expected(full,"cache",RAD_F32,{c.max_seqs,c.num_layers,capacity,2*c.num_kv_heads*c.head_dim});
        expected(full,"length",RAD_I32,{c.max_seqs});
        expected(full,"y_base",RAD_F32,{c.n_seq,c.proposal_length,c.target_width});
        if(real_mode) {
            /* THE CANONICAL REAL LAYOUT: `context_span[s] = [start, count]` with `start` the prefix
             * sum, so the spans tile the planes exactly once. Checked here rather than in the
             * pipeline, because a fixture whose spans overlap or fall short would otherwise show up
             * as a cache mismatch far from its cause. */
            expected(full,"context_span",RAD_I32,{c.n_seq,2});
            const int32_t* span=g32("context_span");
            const int64_t plane=gdim("tap0",0);
            require(plane==gdim("tap1",0)&&plane==gdim("tap2",0),"the three planes must be equally long");
            int64_t cursor=0;
            for(int s=0;s<c.n_seq;++s) {
                require(span[2*s]==cursor,"context_span is not a prefix-sum tiling at sequence "+std::to_string(s));
                require(span[2*s+1]>0,"a real sequence needs a nonempty prefix");
                cursor+=span[2*s+1];
            }
            require(cursor==plane,"context_span must cover the tap planes exactly");
        } else {
            /* The synthetic tiny fixture carries a drive plan; the REAL one does not, because its
             * taps are the recorded full prefix and there is nothing to replay. */
            const int chunks=int(get(full,"chunk_span").shape[0]);
            require(chunks>=2,"the tiny fixture needs at least a reset chunk and a follow-up chunk");
            expected(full,"chunk_span",RAD_I32,{chunks,c.n_seq,2});
            expected(full,"chunk_reset",RAD_I32,{chunks,c.n_seq});
            expected(full,"chunk_path",RAD_I32,{chunks,c.n_seq,16});
            expected(full,"slots",RAD_I32,{c.n_seq});
        }
        for(int rank=0;rank<2;++rank) {
            expected(full,"rank"+std::to_string(rank)+".local_ids",RAD_I32,{c.n_seq*c.proposal_length,c.pool_size});
            expected(full,"rank"+std::to_string(rank)+".local_logits",RAD_F32,{c.n_seq*c.proposal_length,c.pool_size});
        }
        expected(full,"pool_ids",RAD_I32,{c.n_seq*c.proposal_length,c.pool_size});
        expected(full,"pool_unary",RAD_F32,{c.n_seq*c.proposal_length,c.pool_size});
        for(const char* variant:{"fixed","pruned"}) {
            expected(full,std::string(variant)+".n_nodes",RAD_I32,{c.n_seq});
            for(const char* field:{"tokens","parents","depths"})expected(full,std::string(variant)+"."+field,RAD_I64,{c.n_seq,16});
        }
        expected(student,"dloop.selector.pred_codebook.weight",RAD_BF16,{c.vocab_size,c.selector_rank});
        expected(student,"dloop.selector.succ_codebook.weight",RAD_BF16,{c.vocab_size,c.selector_rank});
        const auto& emb=get(teacher,"model.language_model.embed_tokens.weight");const auto& head=get(teacher,"lm_head.weight");
        require(emb.dtype==RAD_BF16&&head.dtype==RAD_BF16&&emb.shape==std::vector<int64_t>{c.vocab_size,c.target_width}&&head.shape==emb.shape,
                "full teacher geometry mismatch");
        packed.resize(size_t(rad_dloop_weight_elements(c)));
        auto add=[&](int w,int layer=0) {
            std::string name="dloop.";if(w>=DL_INPUT_NORM&&w<=DL_DOWN_PROJ)name+="layers."+std::to_string(layer)+".";name+=rad_dloop_weight_name(w);
            const auto& t=get(student,name);require(t.dtype==RAD_BF16&&t.numel()==rad_dloop_weight_size(c,w),"full packed weight mismatch: "+name);
            std::memcpy(packed.data()+rad_dloop_weight_offset(c,w,layer),t.data,t.bytes);
        };
        for(int w=0;w<=DL_MASK_TOKEN;++w)add(w);
        for(int layer=0;layer<c.num_layers;++layer)for(int w=DL_INPUT_NORM;w<=DL_DOWN_PROJ;++w)add(w,layer);
        for(int w=DL_FINAL_NORM;w<=DL_SELECTOR_PROJ;++w)add(w);
        require(m.at("context_capacity").get<int>()==capacity,"full.json capacity disagrees with the manifest");
        for(const auto& entry:m.at("non_last_paths")) {
            VerifyPath path;path.sequence=entry.at("sequence").get<int>();
            path.path=entry.at("path").get<std::vector<int32_t>>();
            require(path.sequence>=0&&path.sequence<c.n_seq,"recorded path sequence out of range");
            verify_paths.push_back(std::move(path));
        }
        require(!verify_paths.empty(),"full fixture records no non-last branch path");
    }
    template<class T>const T* g(const std::string& name)const{return reinterpret_cast<const T*>(get(full,name).data);}
    const int32_t* g32(const std::string& name)const{return g<int32_t>(name);}
    const int64_t* g64(const std::string& name)const{return g<int64_t>(name);}
    int64_t gdim(const std::string& name,int axis)const{return get(full,name).shape[axis];}
    /* The slot each sequence uses. The tiny fixture states it (`slots`), the real one does not, so
     * the identity map is the honest fallback there: every sequence owns its own slot and the
     * goldens were produced from that same assignment. */
    std::vector<int32_t> slots()const {
        if(!real_mode) {
            const int32_t* table=g32("slots");
            return std::vector<int32_t>(table,table+c.n_seq);
        }
        std::vector<int32_t> identity(c.n_seq);
        for(int s=0;s<c.n_seq;++s)identity[s]=s;
        return identity;
    }
    const uint16_t* table(const char* name)const{return reinterpret_cast<const uint16_t*>(get(teacher,name).data);}
    const std::vector<VerifyPath>& paths()const{return verify_paths;}
    void weights(Call& call,int operand){call.bind(operand,packed.data(),RAD_BF16,{int64_t(packed.size())});}
    /* The student blob and this rank's head shard, exactly as the ring fixture binds them. */
    void shard(Call& call,int operand) {
        call.bind(operand,const_cast<uint16_t*>(table("lm_head.weight"))+int64_t(call.rank)*c.vocab_size_shard*c.target_width,
                  RAD_BF16,{c.vocab_size_shard,c.target_width});
    }
    /* Anchor embeddings out of the teacher table, for the sequences of one call. */
    void anchors(Call& call,int operand) {
        const auto* emb=table("model.language_model.embed_tokens.weight");
        for(int s=0;s<c.n_seq;++s) {
            const int64_t anchor=g64("anchors")[s];require(anchor>=0&&anchor<c.active_vocab,"full golden anchor out of range");
            std::memcpy(call.data<uint16_t>(operand)+s*c.target_width,emb+anchor*c.target_width,size_t(c.target_width)*2);
        }
    }
    /* Bind the three tap planes and the student blob on an append call. The planes' row count is the
     * fixture's own: the tiny fixture is a `[tap_rows, W]` capacity plane, while the REAL one is the
     * CONCATENATED prefixes, `[sum(lengths), W]`. Reading the extent off the tensor keeps both
     * honest instead of assuming the capacity. */
    void bind_append(Call& step) {
        for(int plane=0;plane<3;++plane) {
            const std::string name="tap"+std::to_string(plane);
            require(get(full,name).shape==std::vector<int64_t>{gdim(name,0),c.target_width},
                    "tap plane is not a dense [rows, target_width] tensor: "+name);
            step.bind(plane,const_cast<uint16_t*>(g<uint16_t>(name)),RAD_BF16,{gdim(name,0),c.target_width});
        }
        weights(step,7);
    }
    /* ONE append call, everything in it: every sequence resets its slot and appends its whole
     * recorded prefix as a single contiguous chunk. The real fixture has no synthetic plan because
     * the taps ARE the recorded documents, so this is the honest form of the append for it -- and it
     * still exercises the append op on the REAL geometry, which is the point. */
    void append_full_prefix(Harness& h,std::vector<float>& cache,std::vector<int32_t>& length) {
        const size_t cache_elems=size_t(c.max_seqs)*c.num_layers*capacity*2*c.num_kv_heads*c.head_dim;
        const size_t length_elems=size_t(c.max_seqs);
        Call step(RAD_DLOOP_FULL_APPEND,capacity,c);
        bind_append(step);
        std::fill(step.data<int32_t>(3),step.data<int32_t>(3)+int64_t(c.n_seq)*16,-1);
        /* THE FIXTURE'S OWN SPANS. The real tap planes are the sequences' prefixes CONCATENATED, so
         * each sequence reads a different range: `context_span[s] = [start, count]` with `start` the
         * prefix sum of the preceding lengths. Binding a single `[0, count]` for every sequence would
         * feed all of them the first sequence's rows -- which is why the span is data, not a
         * constant of the harness. */
        const int32_t* span=g32("context_span");
        for(int s=0;s<c.n_seq;++s) {
            const int32_t start=span[2*s],count=span[2*s+1];
            require(start>=0&&count>0&&int64_t(start)+count<=c.tap_rows,
                    "recorded span is outside the tap plane");
            step.data<int32_t>(4)[2*s]=start;step.data<int32_t>(4)[2*s+1]=count;
            step.data<int32_t>(5)[s]=s;                                        // slot = sequence
            step.data<int32_t>(6)[s]=1;                                        // reset: new occupant
        }
        std::memcpy(step.data<float>(8),cache.data(),cache_elems*4);
        std::memcpy(step.data<int32_t>(9),length.data(),length_elems*4);
        h.run(step);
        cache.assign(step.data<float>(8),step.data<float>(8)+cache_elems);
        std::copy(step.data<int32_t>(9),step.data<int32_t>(9)+length_elems,length.begin());
        std::puts("FULL_APPEND_REAL_PREFIX_PASS");
    }
    /* Replay the tiny fixture's four-chunk plan, checking the tail promise and the refusals on the
     * way: the whole plan is the fixture's own, so a harness that disagreed with it would fail
     * instead of quietly testing a different drive. */
    void replay_plan(Harness& h,std::vector<float>& cache,std::vector<int32_t>& length) {
        const int chunks=int(get(full,"chunk_span").shape[0]);
        const size_t cache_elems=size_t(c.max_seqs)*c.num_layers*capacity*2*c.num_kv_heads*c.head_dim;
        const size_t length_elems=size_t(c.max_seqs);
        for(int index=0;index<chunks;++index) {
            Call step(RAD_DLOOP_FULL_APPEND,capacity,c);
            bind_append(step);
            std::memcpy(step.data<int32_t>(3),g32("chunk_path")+int64_t(index)*c.n_seq*16,size_t(c.n_seq)*16*4);
            std::memcpy(step.data<int32_t>(4),g32("chunk_span")+int64_t(index)*c.n_seq*2,size_t(c.n_seq)*2*4);
            std::memcpy(step.data<int32_t>(5),g32("slots"),size_t(c.n_seq)*4);
            std::memcpy(step.data<int32_t>(6),g32("chunk_reset")+int64_t(index)*c.n_seq,size_t(c.n_seq)*4);
            std::memcpy(step.data<float>(8),cache.data(),cache_elems*4);
            std::memcpy(step.data<int32_t>(9),length.data(),length_elems*4);
            if(index==chunks-1) {
                /* Two refusals before the last real chunk, each on its own copy of the same operand
                 * set: (a) a slot outside max_seqs, (b) a path append whose first row is past the tap
                 * capacity. Both must be refused with every mutable operand byte-identical, and the
                 * refusal is checked on the DEVICE result, not on the host input. */
                const struct { int operand,entry,value;const char* what; } faults[] = {
                    {5,0,c.max_seqs+3,"slot outside max_seqs"},
                    {4,1,c.tap_rows+1,"path row past tap capacity"}};
                for(const auto& fault:faults) {
                    Call refused(RAD_DLOOP_FULL_APPEND,capacity,c);
                    for(size_t i=0;i<refused.t.size();++i) {
                        if(!refused.t[i].data)continue;
                        const size_t bytes=span_bytes(refused.t[i]);
                        require(bytes<=span_bytes(step.t[i]),"refusal copy exceeds the operand");
                        std::memcpy(refused.t[i].data,step.t[i].data,bytes);
                    }
                    if(fault.operand==5)refused.data<int32_t>(5)[fault.entry]=fault.value;
                    else {
                        refused.data<int32_t>(4)[2*fault.entry]=0;refused.data<int32_t>(4)[2*fault.entry+1]=-1;
                        refused.data<int32_t>(3)[fault.entry*16]=fault.value;
                    }
                    h.expect_refused(refused);
                    std::printf("  FULL_APPEND_REFUSED_PASS %s\n",fault.what);
                }
            }
            h.run(step);
            /* Nothing outside the appended range may change: for every slot and every layer, the
             * positions at and above that slot's new length are byte-identical to what went in. That
             * is the "never truncated, never wrapped" promise checked after every chunk, not only at
             * the end where the final golden could hide a transient overwrite. FullDevice confirmed
             * the store addresses only [len0, len0+limit) per slot, so this check must hold. */
            const int64_t row=int64_t(c.num_layers)*capacity*2*c.num_kv_heads*c.head_dim;
            for(int s=0;s<c.n_seq;++s) {
                const int slot=step.data<int32_t>(5)[s],new_length=step.data<int32_t>(9)[slot];
                require(new_length<=capacity,"full append exceeded the capacity");
                const size_t tail=size_t(capacity-new_length)*2*c.num_kv_heads*c.head_dim;
                for(int layer=0;layer<c.num_layers;++layer) {
                    const int64_t base=(int64_t)slot*row+(int64_t)layer*capacity*2*c.num_kv_heads*c.head_dim
                                       +(int64_t)new_length*2*c.num_kv_heads*c.head_dim;
                    require(std::memcmp(step.data<float>(8)+base,cache.data()+base,tail*4)==0,
                            "full append wrote past length[slot] in slot "+std::to_string(slot)+" layer "+std::to_string(layer));
                }
            }
            cache.assign(step.data<float>(8),step.data<float>(8)+cache_elems);
            std::copy(step.data<int32_t>(9),step.data<int32_t>(9)+length_elems,length.begin());
        }
    }
};
void gdn_pipeline(Harness& h,const Call& tree,int sequences,int ring) {
    constexpr int heads=2,dim=128,channels=512,width=4,history_length=7;
    std::vector<int32_t> cu(sequences+1),parents,idx(sequences),ctx(sequences),path(sequences*16,-1),length(sequences);
    for(int s=0;s<sequences;++s)cu[s+1]=cu[s]+static_cast<const int32_t*>(tree.t[15].data)[s]+1;
    const int tokens=cu.back();parents.resize(tokens);
    for(int s=0;s<sequences;++s) {
        idx[s]=sequences-1-s;ctx[s]=3+s;parents[cu[s]]=-1;
        for(int i=1;i<cu[s+1]-cu[s];++i)parents[cu[s]+i]=static_cast<const int32_t*>(tree.t[14].data)[s*16+i-1]+1;
        std::vector<int> chain;for(int node=cu[s+1]-cu[s]-1;node>=0;node=parents[cu[s]+node])chain.push_back(node);
        std::reverse(chain.begin(),chain.end());length[s]=int32_t(chain.size());std::copy(chain.begin(),chain.end(),path.begin()+s*16);
    }
    auto p=std::vector<RadParam>{pi("q_len",16),pi("head_k",128),pi("head_v",128),pi("n_head_k",1),pi("n_head_v",heads),pi("conv_width",width),
        pi("node_capacity",16),pi("history_ring",ring),pi("n_seq",sequences),pf("scale",.125),pf("eps",1e-6),pf("l2_eps",1e-6),pf("softplus_thr",20)};
    std::vector<uint16_t> x(tokens*channels),weights(channels*width),bias(channels),history(sequences*channels*history_length),q(tokens*dim,0x7fc1),k(q),
        v(tokens*heads*dim,0x7fc1),raw(tokens*channels,0x7fc1),a(tokens*heads),b(a),z(v.size()),out(v.size(),0x7fc1);
    std::vector<float> state(sequences*heads*dim*dim),factors(tokens*heads*257,-123),alog(heads,-2),dt(heads,-.5f),norm(dim,1);
    auto fill=[](std::vector<uint16_t>& dst,int modulo,float step){for(size_t i=0;i<dst.size();++i)dst[i]=rad_f32_to_bf16(float(int(i%modulo)-modulo/2)*step);};
    fill(x,37,.03125f);fill(weights,7,.0625f);fill(history,23,.015625f);fill(a,9,.125f);fill(b,7,.125f);fill(z,13,.125f);
    for(size_t i=0;i<state.size();++i)state[i]=float(int(i%31)-15)*.0009765625f;
    Call conv("gdn_conv_tree",p,12);
    conv.t={tensor(x.data(),RAD_BF16,{tokens,channels}),tensor(weights.data(),RAD_BF16,{channels,width}),tensor(bias.data(),RAD_BF16,{channels}),
        tensor(history.data(),RAD_BF16,{sequences,channels,history_length}),tensor(idx.data(),RAD_I32,{sequences}),tensor(cu.data(),RAD_I32,{sequences+1}),
        tensor(parents.data(),RAD_I32,{tokens}),tensor(ctx.data(),RAD_I32,{sequences}),tensor(q.data(),RAD_BF16,{tokens,dim}),tensor(k.data(),RAD_BF16,{tokens,dim}),
        tensor(v.data(),RAD_BF16,{tokens,heads*dim}),tensor(raw.data(),RAD_BF16,{tokens,channels})};
    h.run(conv);
    Call recurrent("gdn_recurrent_tree",p,15);
    recurrent.t={conv.t[8],conv.t[9],conv.t[10],tensor(a.data(),RAD_BF16,{tokens,heads}),tensor(b.data(),RAD_BF16,{tokens,heads}),
        tensor(alog.data(),RAD_F32,{heads}),tensor(dt.data(),RAD_F32,{heads}),tensor(state.data(),RAD_F32,{sequences,heads,dim,dim}),conv.t[4],conv.t[5],conv.t[6],
        tensor(z.data(),RAD_BF16,{tokens,heads*dim}),tensor(norm.data(),RAD_F32,{dim}),tensor(out.data(),RAD_BF16,{tokens,heads*dim}),tensor(factors.data(),RAD_F32,{tokens,heads,257})};
    h.run(recurrent);
    Call commit("gdn_tree_commit",p,11);
    commit.t={recurrent.t[7],conv.t[4],recurrent.t[14],conv.t[3],conv.t[4],conv.t[11],tensor(path.data(),RAD_I32,{sequences,16}),
        tensor(length.data(),RAD_I32,{sequences}),conv.t[7],conv.t[5],conv.t[6]};h.run(commit);
}
/* RADIANCE_DLOOP_BENCH only: the qwen4exp TP2 decode geometry the trunk actually runs, which
 * no golden carries. arch/common/rad_block_gdn_fp8.h passes linear_num_key_heads/value_heads
 * divided by the world size, so rank 0 and rank 1 each own n_head_k=8 of 16 and n_head_v=24 of
 * 48; head_k/head_v are 128 and the conv width is 4 (qwen4exp.ssm.conv_kernel). Rows are the
 * full 16-node tree of four sequences, the shape mine/dloop_kernels.md sizes the batch for.
 * Random operands and timing only: no reference launch, no comparison, no coverage claim. */
void gdn_bench_real_geometry(Harness& h) {
    constexpr int sequences=4,heads_k=8,heads_v=24,dim=128,width=4,history_length=7;
    std::vector<int32_t> cu(sequences+1),idx(sequences),ctx(sequences),parents,path(sequences*16,-1),length(sequences);
    for(int s=0;s<sequences;++s)cu[s+1]=cu[s]+16;
    const int tokens=cu.back();parents.resize(tokens);
    /* The fixed-variant topology of the real golden: node 0 is the root, and nodes 2i-1 and 2i
     * hang off node i-1 -- every node but the last has a child, so each branch replays ancestors. */
    for(int s=0;s<sequences;++s) {
        idx[s]=s;ctx[s]=64+s;parents[cu[s]]=-1;
        for(int i=1;i<16;++i)parents[cu[s]+i]=(i-1)/2;
        std::vector<int> chain;for(int node=15;node>=0;node=parents[cu[s]+node])chain.push_back(node);
        std::reverse(chain.begin(),chain.end());length[s]=int32_t(chain.size());std::copy(chain.begin(),chain.end(),path.begin()+s*16);
    }
    for(int rank=0;rank<2;++rank) {
        const int hv=heads_v,hk=heads_k;
        const int d=128*(2*hk+hv);
        auto p=std::vector<RadParam>{pi("q_len",16),pi("head_k",128),pi("head_v",128),pi("n_head_k",hk),pi("n_head_v",hv),pi("conv_width",width),
            pi("node_capacity",16),pi("history_ring",0),pi("n_seq",sequences),pf("scale",.125),pf("eps",1e-6),pf("l2_eps",1e-6),pf("softplus_thr",20)};
        std::vector<uint16_t> x(tokens*d),weights(d*width),bias(d),history(sequences*d*history_length),
            q(tokens*hk*dim,0x3f80),k(q),v(tokens*hv*dim,0x3f80),raw(tokens*d,0x3f80),a(tokens*hv),b(a),z(v.size()),out(v.size());
        std::vector<float> state(sequences*hv*dim*dim),factors(tokens*hv*257,-123),alog(hv,-2),dt(hv,-.5f),norm(dim,1);
        auto fill=[](std::vector<uint16_t>& dst,int modulo,float step){for(size_t i=0;i<dst.size();++i)dst[i]=rad_f32_to_bf16(float(int(i%modulo)-modulo/2)*step);};
        fill(x,37,.03125f);fill(weights,7,.0625f);fill(history,23,.015625f);fill(a,9,.125f);fill(b,7,.125f);fill(z,13,.125f);
        for(size_t i=0;i<state.size();++i)state[i]=float(int(i%31)-15)*.0009765625f;
        Call conv("gdn_conv_tree",p,12);
        conv.rank=rank;
        conv.t={tensor(x.data(),RAD_BF16,{tokens,d}),tensor(weights.data(),RAD_BF16,{d,width}),tensor(bias.data(),RAD_BF16,{d}),
            tensor(history.data(),RAD_BF16,{sequences,d,history_length}),tensor(idx.data(),RAD_I32,{sequences}),tensor(cu.data(),RAD_I32,{sequences+1}),
            tensor(parents.data(),RAD_I32,{tokens}),tensor(ctx.data(),RAD_I32,{sequences}),tensor(q.data(),RAD_BF16,{tokens,hk*dim}),tensor(k.data(),RAD_BF16,{tokens,hk*dim}),
            tensor(v.data(),RAD_BF16,{tokens,hv*dim}),tensor(raw.data(),RAD_BF16,{tokens,d})};
        h.run_device_only(conv);
        Call recurrent("gdn_recurrent_tree",p,15);
        recurrent.rank=rank;
        recurrent.t={conv.t[8],conv.t[9],conv.t[10],tensor(a.data(),RAD_BF16,{tokens,hv}),tensor(b.data(),RAD_BF16,{tokens,hv}),
            tensor(alog.data(),RAD_F32,{hv}),tensor(dt.data(),RAD_F32,{hv}),tensor(state.data(),RAD_F32,{sequences,hv,dim,dim}),conv.t[4],conv.t[5],conv.t[6],
            tensor(z.data(),RAD_BF16,{tokens,hv*dim}),tensor(norm.data(),RAD_F32,{dim}),tensor(out.data(),RAD_BF16,{tokens,hv*dim}),tensor(factors.data(),RAD_F32,{tokens,hv,257})};
        h.run_device_only(recurrent);
        Call commit("gdn_tree_commit",p,11);
        commit.rank=rank;
        commit.t={recurrent.t[7],conv.t[4],recurrent.t[14],conv.t[3],conv.t[4],conv.t[11],tensor(path.data(),RAD_I32,{sequences,16}),
            tensor(length.data(),RAD_I32,{sequences}),conv.t[7],conv.t[5],conv.t[6]};
        h.run_device_only(commit);

        /* REFUSAL PROBES: malformed metadata must be refused by the SAME gate on device and
         * host (rad_gdn_tree_metadata is shared), leave every output byte untouched, and not
         * fault. run() proves all three at once: it launches the device row, syncs (a fault is a
         * device error), launches the oracle, and downloads every OUT/INOUT for comparison --
 * untouched device bytes equal the untouched host bytes, so a byte that moved is a FAIL.
         * Cases: a parent outside the sequence (999), nnode past node_capacity (cu claims 32),
         * and a negative bos (cu starts at -1) -- which, without the range guard, reads
         * parent[-1] before the verdict. */
        auto refusal=[&](const char* what,int tokens_bad,std::vector<int32_t> cu_bad,
                         std::vector<int32_t> parents_bad) {
            /* EVERY operand is fresh: reusing the good call's tensors would compare the good
             * run's own writes, not a refusal's. Weights/history can be shared (they are INPUTS
             * nothing writes), but q/k/v/raw/x are this probe's own sentinels. */
            std::vector<uint16_t> x_bad(size_t(tokens_bad)*d,0x3f80),w_bad(size_t(d)*width,0x3f80),
                b_bad(size_t(d),0x3f80),hist_bad(size_t(d)*history_length,0x3f80),
                q_bad(size_t(tokens_bad)*hk*dim,0x3f80),k_bad(size_t(tokens_bad)*hk*dim,0x3f80),
                v_bad(size_t(tokens_bad)*hv*dim,0x3f80),raw_bad(size_t(tokens_bad)*d,0x3f80);
            std::vector<int32_t> idx_bad(1),ctx_bad(1);
            idx_bad[0]=0;ctx_bad[0]=64;
            /* ONE sequence: the sentinel compare then covers exactly the sequence whose metadata
             * is corrupt. A multi-sequence call refuses only ITS bad sequences and computes the
             * rest, which the whole-output compare cannot express. */
            std::vector<RadParam> p1=p;
            for(auto& q:p1)if(q.kind==RAD_P_INT&&std::strcmp(q.key,"n_seq")==0)q.ival=1;
            Call bad("gdn_conv_tree",p1,12);
            bad.rank=rank;
            bad.t={tensor(x_bad.data(),RAD_BF16,{tokens_bad,d}),tensor(w_bad.data(),RAD_BF16,{d,width}),
                tensor(b_bad.data(),RAD_BF16,{d}),tensor(hist_bad.data(),RAD_BF16,{1,d,history_length}),
                tensor(idx_bad.data(),RAD_I32,{1}),tensor(cu_bad.data(),RAD_I32,{2}),
                tensor(parents_bad.data(),RAD_I32,{tokens_bad}),tensor(ctx_bad.data(),RAD_I32,{1}),
                tensor(q_bad.data(),RAD_BF16,{tokens_bad,hk*dim}),tensor(k_bad.data(),RAD_BF16,{tokens_bad,hk*dim}),
                tensor(v_bad.data(),RAD_BF16,{tokens_bad,hv*dim}),tensor(raw_bad.data(),RAD_BF16,{tokens_bad,d})};
            h.run_refused(bad,what);
        };
        {
            std::vector<int32_t> cu_ok{0,16};
            std::vector<int32_t> par_bad(parents.begin(),parents.begin()+16);par_bad[2]=999;
            refusal("parent 999",16,cu_ok,par_bad);
        }
        {
            const int tokens_big=32;
            std::vector<int32_t> cu_big{0,32};
            std::vector<int32_t> par_big(tokens_big,-1);for(int i=1;i<32;++i)par_big[i]=(i-1)/2;
            refusal("nnode 32",tokens_big,cu_big,par_big);
        }
        {
            std::vector<int32_t> cu_neg{-1,15};
            std::vector<int32_t> par_neg(16,-1);for(int i=1;i<16;++i)par_neg[i]=(i-1)/2;
            refusal("bos -1",16,cu_neg,par_neg);
        }
    }
}
void ple_pipeline_case(Harness& h,int ring,int capacity,Harness& linear,bool frozen) {
    constexpr int sequences=2,slots=3,n=8,heads=4,ngram=3,width=4,dilation=3,hist=9,tok_len=5,conv_len=13,eos=2;
    const bool wide=capacity==65;
    const int first=wide?65:5,second=wide?37:5,tokens=first+second+4;
    constexpr uint16_t untouched=0x43aa;
    std::vector<int32_t> cu{2,2+first,2+first+second},parents(tokens,-1),tok_idx{2,-99,1,-99},conv_idx{1,-99,-99,2,-99,-99},
        ctx{wide?5:0,wide?19:11},path(sequences*capacity,-1),length(sequences,3),tok(tokens),tok_history(slots*tok_len),ids(tokens*heads,-71);
    std::vector<int64_t> mult{INT64_MAX-6,INT64_MIN+11,6364136223846793005LL},vocab{97,101,103,107},offsets{0,97,198,301};
    std::vector<uint16_t> x(tokens*n),weights(n*width),residual(tokens*n),history(slots*n*conv_len),y(tokens*n,untouched),raw(y);
    auto fill=[](std::vector<uint16_t>& dst,int modulo,float scale) {
        for(size_t i=0;i<dst.size();++i)dst[i]=rad_f32_to_bf16(float(int(i%modulo)-modulo/2)*scale);
    };
    fill(x,31,.03125f);fill(weights,11,.0625f);fill(residual,17,.015625f);fill(history,29,.03125f);
    for(int i=0;i<tokens;++i)tok[i]=7+6*i;
    for(size_t i=0;i<tok_history.size();++i)tok_history[i]=41+int32_t(i);
    for(int s=0;s<sequences;++s) {
        if(wide) {
            // An eight-deep spine and siblings: node64 branches from node7, not node63.
            for(int node=1;node<cu[s+1]-cu[s];++node)parents[cu[s]+node]=node<9?node-1:(node-9)%8;
            const int leaf=s?32:64;
            parents[cu[s]+leaf]=s?3:7;length[s]=s?5:9;
            for(int k=0;k<length[s]-1;++k)path[s*capacity+k]=k;
            path[s*capacity+length[s]-1]=leaf;
        } else {
            const int topology[]{-1,0,0,1,2};
            std::copy(topology,topology+5,parents.begin()+cu[s]);
            // Select 0->1->3, not the final enumerated branch 0->2->4.
            path[s*capacity]=0;path[s*capacity+1]=1;path[s*capacity+2]=3;
        }
    }
    tok[cu[0]+1]=eos;tok[cu[1]+2]=-17;
    auto p=std::vector<RadParam>{pi("q_len",capacity),pi("n",n),pi("heads",heads),pi("ngram",ngram),pi("eos",eos),pi("width",width),
        pi("dilation",dilation),pi("node_capacity",capacity),pi("history_ring",ring),pi("n_seq",sequences)};
    Call ids_call("ngram_ids_tree",p,10);
    ids_call.t={tensor(tok.data(),RAD_I32,{tokens}),ring?tensor(tok_history.data(),RAD_I32,{slots,1,tok_len}):tensor(tok_history.data(),RAD_I32,{slots,tok_len}),
        tensor(tok_idx.data(),RAD_I32,{sequences,2}),tensor(cu.data(),RAD_I32,{sequences+1}),tensor(parents.data(),RAD_I32,{tokens}),
        tensor(ctx.data(),RAD_I32,{sequences}),tensor(mult.data(),RAD_I64,{ngram}),tensor(vocab.data(),RAD_I64,{heads}),
        tensor(offsets.data(),RAD_I64,{heads}),tensor(ids.data(),RAD_I32,{tokens,heads})};
    Call conv("ple_conv_tree",p,10);
    conv.t={tensor(x.data(),RAD_BF16,{tokens,n}),tensor(weights.data(),RAD_BF16,{n,width}),tensor(history.data(),RAD_BF16,{slots,n,conv_len}),
        tensor(conv_idx.data(),RAD_I32,{sequences,3}),ids_call.t[3],ids_call.t[4],ids_call.t[5],tensor(residual.data(),RAD_BF16,{tokens,n}),
        tensor(y.data(),RAD_BF16,{tokens,n}),tensor(raw.data(),RAD_BF16,{tokens,n})};
    Call commit("ple_tree_commit",p,11);
    commit.t={ids_call.t[1],ids_call.t[2],conv.t[2],conv.t[3],ids_call.t[0],conv.t[9],ids_call.t[3],ids_call.t[4],ids_call.t[5],
        tensor(path.data(),RAD_I32,{sequences,capacity}),tensor(length.data(),RAD_I32,{sequences})};
    int replayed_nodes=0;
    auto compare_linear=[&]() {
        // Gather ancestry independently, then omit parents: the small GPU sees a linear sequence,
        // not the wide tree's node IDs or capacity walk. Frozen replay is opt-in, never implied.
        for(int s=0;s<sequences;++s)for(int target=0;target<cu[s+1]-cu[s];++target) {
            std::vector<int> ancestry;
            for(int node=target;node>=0;node=parents[cu[s]+node]) {
                require(ancestry.size()<16,"PLE gathered oracle path exceeds frozen capacity");
                ancestry.push_back(node);
            }
            std::reverse(ancestry.begin(),ancestry.end());
            const int count=int(ancestry.size());
            std::vector<int32_t> lt(count),li(count*heads,-71),lc{0,count},lctx{ctx[s]},
                ti{tok_idx[s*2],-99},ci{conv_idx[s*3],-99,-99};
            std::vector<uint16_t> lx(count*n),lr(count*n),ly(count*n,untouched),lraw(ly);
            auto th=tok_history;auto ch=history;
            for(int k=0;k<count;++k) {
                const int src=cu[s]+ancestry[k];lt[k]=tok[src];
                std::copy_n(x.data()+src*n,n,lx.data()+k*n);
                std::copy_n(residual.data()+src*n,n,lr.data()+k*n);
            }
            auto lp=p;
            for(auto& param:lp) {
                if(!std::strcmp(param.key,"node_capacity")||!std::strcmp(param.key,"q_len"))param.ival=16;
                if(!std::strcmp(param.key,"n_seq"))param.ival=1;
            }
            Call lid("ngram_ids_tree",lp,10);
            lid.t={tensor(lt.data(),RAD_I32,{count}),ring?tensor(th.data(),RAD_I32,{slots,1,tok_len}):tensor(th.data(),RAD_I32,{slots,tok_len}),
                tensor(ti.data(),RAD_I32,{1,2}),tensor(lc.data(),RAD_I32,{2}),RadTensor{},tensor(lctx.data(),RAD_I32,{1}),
                ids_call.t[6],ids_call.t[7],ids_call.t[8],tensor(li.data(),RAD_I32,{count,heads})};
            Call lconv("ple_conv_tree",lp,10);
            lconv.t={tensor(lx.data(),RAD_BF16,{count,n}),conv.t[1],tensor(ch.data(),RAD_BF16,{slots,n,conv_len}),
                tensor(ci.data(),RAD_I32,{1,3}),lid.t[3],RadTensor{},lid.t[5],tensor(lr.data(),RAD_BF16,{count,n}),
                tensor(ly.data(),RAD_BF16,{count,n}),tensor(lraw.data(),RAD_BF16,{count,n})};
            linear.run(lid);linear.run(lconv);
            const int dst=cu[s]+target;
            require(std::equal(li.end()-heads,li.end(),ids.begin()+dst*heads),"PLE gathered linear GPU IDs mismatch");
            require(std::equal(ly.end()-n,ly.end(),y.begin()+dst*n),"PLE gathered linear GPU convolution is not byte-exact");
            require(std::equal(lraw.end()-n,lraw.end(),raw.begin()+dst*n),"PLE gathered linear GPU saved raw mismatch");
            require(th==tok_history&&ch==history,"PLE gathered linear verification mutated history");
            ++replayed_nodes;
        }
    };
    auto verify=[&]() {
        const auto before_tok=tok_history;const auto before_conv=history;
        std::fill(ids.begin(),ids.end(),-71);std::fill(y.begin(),y.end(),untouched);std::fill(raw.begin(),raw.end(),untouched);
        h.run(ids_call);h.run(conv);
        require(tok_history==before_tok&&history==before_conv,"PLE verify mutated committed history");
        for(int t=0;t<tokens;++t) {
            const bool selected=t>=cu.front()&&t<cu.back();
            for(int j=0;j<n;++j)require(raw[t*n+j]==(selected?x[t*n+j]:untouched),"PLE raw copy or absolute subrange mismatch");
            if(!selected) {
                for(int j=0;j<n;++j)require(y[t*n+j]==untouched,"PLE wrote y outside absolute subrange");
                for(int j=0;j<heads;++j)require(ids[t*heads+j]==-71,"PLE wrote ids outside absolute subrange");
            }
        }
        compare_linear();
    };
    auto commit_selected=[&]() {
        auto expected_tok=tok_history;auto expected_conv=history;
        for(int s=0;s<sequences;++s) {
            const int ts=tok_idx[s*2],cs=conv_idx[s*3];
            for(int i=0;i<ngram-1;++i)if(ctx[s]-(ngram-1)+i<0)expected_tok[ts*tok_len+i]=eos;
            for(int j=0;j<n;++j) {
                const int base=(cs*n+j)*conv_len;
                if(!ring)for(int i=0;i<hist;++i)if(ctx[s]-hist+i<0)expected_conv[base+i]=0;
                for(int k=0;k<length[s];++k) {
                    const uint16_t value=raw[(cu[s]+path[s*capacity+k])*n+j];
                    if(ring)expected_conv[base+(ctx[s]+k)%conv_len]=value;
                    else {std::move(expected_conv.begin()+base+1,expected_conv.begin()+base+hist,expected_conv.begin()+base);expected_conv[base+hist-1]=value;}
                }
            }
            for(int k=0;k<length[s];++k) {
                for(int i=0;i<ngram-2;++i)expected_tok[ts*tok_len+i]=expected_tok[ts*tok_len+i+1];
                expected_tok[ts*tok_len+ngram-2]=tok[cu[s]+path[s*capacity+k]];
            }
        }
        h.run(commit);
        require(tok_history==expected_tok&&history==expected_conv,"PLE non-last path commit or untouched slot mismatch");
        for(int s=0;s<sequences;++s)ctx[s]+=length[s];
    };
    verify();commit_selected();

    // A long linear prefill must leave the same ring/canonical window that tree verification reads.
    constexpr int prefill_rows=73,prefill_bos=2,prefill_tokens=prefill_rows+5;
    std::vector<int32_t> prefill_cu{prefill_bos,prefill_bos+prefill_rows};
    std::vector<uint16_t> prefill_x(prefill_tokens*n),prefill_y(prefill_tokens*n,untouched);
    fill(prefill_x,43,.015625f);
    auto prefill_params=p;
    for(auto& param:prefill_params) {
        if(!std::strcmp(param.key,"q_len"))param.ival=prefill_rows;
        if(!std::strcmp(param.key,"n_seq"))param.ival=1;
    }
    RadParam mode{};mode.key="mode";mode.kind=RAD_P_STR;mode.sval="prefill";prefill_params.push_back(mode);
    Call prefill("ple_conv_tree",prefill_params,10);
    prefill.t={tensor(prefill_x.data(),RAD_BF16,{prefill_tokens,n}),conv.t[1],conv.t[2],tensor(conv_idx.data()+3,RAD_I32,{1,3}),
        tensor(prefill_cu.data(),RAD_I32,{2}),RadTensor{},tensor(ctx.data()+1,RAD_I32,{1}),RadTensor{},
        tensor(prefill_y.data(),RAD_BF16,{prefill_tokens,n}),RadTensor{}};
    auto expected_history=history;
    for(int j=0;j<n;++j) {
        const int base=(conv_idx[3]*n+j)*conv_len;
        if(ring)for(int t=0;t<prefill_rows;++t)expected_history[base+(ctx[1]+t)%conv_len]=prefill_x[(prefill_bos+t)*n+j];
        else for(int i=0;i<hist;++i)expected_history[base+i]=prefill_x[(prefill_bos+prefill_rows-hist+i)*n+j];
    }
    h.run(prefill);require(history==expected_history,"PLE long prefill final history mismatch");
    for(int t=0;t<prefill_tokens;++t)if(t<prefill_bos||t>=prefill_bos+prefill_rows)
        for(int j=0;j<n;++j)require(prefill_y[t*n+j]==untouched,"PLE prefill wrote outside absolute subrange");
    // The unchanged ngram prefill path materializes these last two canonical token IDs.
    for(int i=0;i<ngram-1;++i)tok_history[tok_idx[2]*tok_len+i]=101+prefill_rows-(ngram-1)+i;
    ctx[1]+=prefill_rows;fill(x,23,.0625f);
    verify();commit_selected();
    if(wide) {
        for(Call* call:{&ids_call,&conv,&commit})h.run_refused(*call,"PLE capacity66",true);
        const auto good_parents=parents;
        // Verification gates each sequence separately; invalidate both to require every byte unchanged.
        for(int s=0;s<sequences;++s)parents[cu[s]+1]=1;
        h.run_refused(ids_call,"PLE self-parent IDs");
        h.run_refused(conv,"PLE self-parent convolution");
        h.run_refused(commit,"PLE self-parent commit");
        std::copy(good_parents.begin(),good_parents.end(),parents.begin());
        const int good=path[capacity+1];
        path[capacity+1]=64; // Only sequence two is bad: commit must refuse the whole transaction.
        h.run_refused(commit,"PLE sequence-two out-of-range path64");
        path[capacity+1]=3; // In range, but skips the required parent1.
        h.run_refused(commit,"PLE sequence-two disconnected path");
        path[capacity+1]=good;
    }
    std::printf("PLE_TREE_PIPELINE_PASS history_ring=%d capacity=%d path_pitch=%d nodes=%d,%d selected_node=%d "
                "non_last_branch=1 prefill_rows=%d verifies=2 commits=2 exact_linear_nodes=%d oracle=%s negative_token=1 eos_cut=1\n",
                ring,capacity,capacity,first,second,wide?64:3,prefill_rows,replayed_nodes,frozen?"frozen_gpu":"current_small_gpu");
}
void ple_pipeline(Harness& h,int ring) {
    std::unique_ptr<Harness> frozen;
    if(const char* path=std::getenv("RADIANCE_PLE_FROZEN_MINE")) {
        require(*path,"RADIANCE_PLE_FROZEN_MINE must name the original PLE GPU plugin");
        frozen=std::make_unique<Harness>(path,h.oracle_path.c_str());
    }
    for(int capacity:{16,65})ple_pipeline_case(h,ring,capacity,frozen?*frozen:h,bool(frozen));
}
/* The full-context tree writes tree_tok/tree_parent/n_nodes at 13..15 and the optional anchor and
 * packed tree at 16..17, exactly as the ring tree does, so this comparison has the same shape. What
 * it pins beyond the ring check is that these trees were built over the FULL verified prefix the
 * cache operand holds -- a tree that had silently dropped history would not reproduce them. */
void check_full_tree_golden(FullFixture& f,Call& tree,const std::string& variant) {
    const auto& c=f.c;
    for(int s=0;s<c.n_seq;++s) {
        const int count=tree.data<int32_t>(15)[s];
        require(count==f.g32(variant+".n_nodes")[s]&&count<=15,"full golden tree count mismatch");
        int depth[16]{};const int* packed=tree.data<int32_t>(17)+s*33;
        require(packed[0]==count+1&&packed[1]==f.g64("anchors")[s]&&packed[17]==-1,"full packed root mismatch");
        for(int i=0;i<count;++i) {
            const int parent=tree.data<int32_t>(14)[s*16+i],token=tree.data<int32_t>(13)[s*16+i];
            require(parent>=-1&&parent<i,"invalid full tree parent");
            depth[i]=parent<0?1:depth[parent]+1;
            require(token==f.g64(variant+".tokens")[s*16+i+1]&&parent==f.g64(variant+".parents")[s*16+i+1]-1&&
                    depth[i]==f.g64(variant+".depths")[s*16+i+1],
                    "full golden tree token/parent/depth mismatch: "+variant+" sequence "+std::to_string(s));
            require(packed[2+i]==token&&packed[18+i]==parent+1,"full packed draft mismatch");
        }
        for(int i=count;i<16;++i)
            require(tree.data<int32_t>(13)[s*16+i]==-1&&tree.data<int32_t>(14)[s*16+i]==-1,"full tree padding mismatch");
    }
}
/* The non-last path the fixture records. Each chain is a root-to-leaf list of DRAFT indices; it is
 * walked from its leaf back through the DEVICE's own parent array and must end at the anchor
 * (parent -1). `full.json` records only chains whose generator-side check found a hop that is not
 * its parent's last child in node order, so a tree that merely re-emitted its newest branch fails
 * this check as a chain mismatch rather than passing by luck. */
void check_non_last_path(FullFixture& f,const Call& tree) {
    for(const auto& entry:f.paths()) {
        const auto& path=entry.path;
        require(!path.empty(),"recorded non-last path is empty");
        const int32_t* parents=tree.data<int32_t>(14)+int64_t(entry.sequence)*16;
        int node=path.back();
        require(node>=0&&node<15,"recorded path leaf out of range");
        for(size_t hop=path.size();hop-->1;) {
            require(path[hop-1]>=0&&path[hop-1]<15,"recorded path node out of range");
            require(parents[node]==path[hop-1],
                    "recorded non-last path is not sequence "+std::to_string(entry.sequence)+
                    "'s parent chain at hop "+std::to_string(hop));
            node=path[hop-1];
        }
        require(parents[node]==-1,"recorded non-last path does not end at the anchor");
    }
    std::printf("  FULL_NON_LAST_PATH_PASS paths=%zu\n",f.paths().size());
}
/* Float tolerance against the PyTorch golden. Device vs mine_ref (Harness::run) keeps its own tight
 * bound everywhere; this one compares two implementations with different reduction orders. The tiny
 * fixture's dots are short. On the real checkpoint (K up to 7680) the independent host reference
 * mine_ref already sits 6.5e-5 (y_base) and 4.1e-5 (pool_unary) from PyTorch
 * (reports/dloop_engine/real_ref_check_185000.log), so the real bound is about three times that.
 * Pool IDs and both trees stay exact either way. */
struct GoldenTol { float atol, rtol; };
constexpr GoldenTol kTinyTol{3e-5f,2e-5f}, kRealTol{2e-4f,2e-4f};
void pipeline(Harness& h,Fixture& f,const char* marker,GoldenTol tol) {
    h.coverage.clear();auto c=f.c;Call append(RAD_DLOOP_CTX_APPEND,c);
    append.bind(0,const_cast<uint8_t*>(get(f.golden,"context").data),RAD_BF16,{c.tap_rows,c.context_features_dim});
    for(int s=0;s<c.n_seq;++s)append.data<int32_t>(2)[s]=c.n_seq-1-s;
    for(int phase=0;phase<2;++phase) {
        std::fill(append.data<int32_t>(1),append.data<int32_t>(1)+c.n_seq*16,-1);
        for(int s=0;s<c.n_seq;++s) {
            const int length=f.g<int32_t>("context_lengths")[s];require(length>=1&&length<=c.context_window,"golden context length invalid");
            append.data<int32_t>(3)[s]=phase?0:1;
            for(int j=0;j<16&&phase*16+j<length;++j)append.data<int32_t>(1)[s*16+j]=s*c.context_window+c.context_window-length+phase*16+j;
        }
        h.run(append);
    }
    Call base0(RAD_DLOOP_BASE,c);f.base(base0,append);h.run(base0);
    near(base0.data<float>(7),f.g<float>("ctxh"),size_t(c.n_seq)*c.context_window*c.hidden_size,2e-5f,1e-5f,"golden.ctxh");
    near(base0.data<float>(8),f.g<float>("y_base"),size_t(c.n_seq)*c.proposal_length*c.target_width,tol.atol,tol.rtol,"golden.y_base");
    c.vocab_lo=c.vocab_size_shard;Call base1(RAD_DLOOP_BASE,c);f.base(base1,append);h.run(base1);
    near(base1.data<float>(8),f.g<float>("y_base"),size_t(c.n_seq)*c.proposal_length*c.target_width,tol.atol,tol.rtol,"golden.y_base.rank1");
    c.vocab_lo=0;
    {Call single(RAD_DLOOP_POOL,c);f.pool(single,base0,nullptr);h.run(single);std::puts("POOL_SINGLE_CONTRIBUTOR_PASS");}
    Call pool0(RAD_DLOOP_POOL,c);f.pool(pool0,base0,&base1);h.run(pool0);
    c.vocab_lo=c.vocab_size_shard;Call pool1(RAD_DLOOP_POOL,c);f.pool(pool1,base0,&base1);h.run(pool1);
    const int rows=c.n_seq*c.proposal_length*c.pool_size;
    for(int i=0;i<rows;++i)require(pool0.data<int32_t>(7)[i]==f.g<int64_t>("pool_ids")[i]&&pool1.data<int32_t>(7)[i]==pool0.data<int32_t>(7)[i],"golden pool ID mismatch");
    near(pool0.data<float>(8),f.g<float>("pool_unary"),rows,tol.atol,tol.rtol,"golden.pool_unary");
    // Both logical TP shards ran on one physical GPU. Transport is host-emulated;
    // one owner plus exact BF16 zero makes the reconstructed payload bitwise OR.
    for(int plane=9;plane<=13;++plane) {
        const size_t count=span_bytes(pool0.t[plane])/2;
        for(size_t i=0;i<count;++i){const auto a=pool0.data<uint16_t>(plane)[i],b=pool1.data<uint16_t>(plane)[i];require(a==0||b==0,"overlapping pool owners");pool0.data<uint16_t>(plane)[i]=uint16_t(a|b);}
    }
    std::puts("POOL_TWO_SHARD_MERGE_PASS");
    for(int variant=0;variant<2;++variant) {
        c.vocab_lo=0;c.pruned=variant;Call tree(RAD_DLOOP_DARTREE,c);
        tree.t[0]=base0.t[7];tree.t[1]=base0.t[2];tree.t[2]=base0.t[3];tree.t[3]=base0.t[4];tree.t[4]=base0.t[5];tree.t[5]=base0.t[8];
        for(int i=6;i<=12;++i)tree.t[i]=pool0.t[i+1];
        std::vector<int32_t> anchors(c.n_seq),packed(c.n_seq*33);
        for(int s=0;s<c.n_seq;++s)anchors[s]=int32_t(f.g<int64_t>("anchors")[s]);
        tree.bind(16,anchors.data(),RAD_I32,{c.n_seq});tree.bind(17,packed.data(),RAD_I32,{c.n_seq,33});h.run(tree);
        check_tree_golden(f,tree,variant?"pruned":"fixed");gdn_pipeline(h,tree,c.n_seq,variant);
    }
    if(bench_enabled())gdn_bench_real_geometry(h);
    for(int ring=0;ring<2;++ring)ple_pipeline(h,ring);
    h.require_coverage(false);std::puts(marker);std::fflush(stdout);
}
/* ============================================================ the full-context pipeline
 *
 * append (the fixture's four chunks: two contiguous ones, a non-monotone 13-row path, a FULL
 * sixteen-row path with no -1 terminator, and empty chunks) -> base -> pool -> tree, per rank, on
 * the five-layer fixture geometry with a context longer than any ring fixture carries. What is
 * checked:
 *
 *   append   the golden cache and length after every chunk; nothing outside [0,length) written
 *            (no wrap, no truncate); and the metadata transaction -- a slot outside the batch and
 *            a path row past the tap capacity are both refused with every operand byte-identical.
 *   base     y_base within the float bound and this rank's shard top-k by the same cutoff rule the
 *            ring ops use.
 *   pool     the merged IDs and unary values, then the exact BF16 ownership merge. The pool call
 *            itself keeps its own RING tags (see `pool_call`): it is geometry-neutral and its
 *            arithmetic never reads a layer count or a context width.
 *   tree     both variants' token/parent/depth arrays and packed trees, plus the fixture's own
 *            non-last root-to-leaf path walked back through the device's parent array.
 *
 * The drive plan is the fixture's own (`full.safetensors` carries span, reset, path and slots), so
 * a harness that disagrees with the golden's plan fails instead of quietly testing another one. */
void full_pipeline(Harness& h,FullFixture& f,const char* marker,GoldenTol tol) {
    h.coverage.clear();auto c=f.c;
    const size_t cache_elems=size_t(c.max_seqs)*c.num_layers*f.capacity*2*c.num_kv_heads*c.head_dim;
    const size_t length_elems=size_t(c.max_seqs);
    /* The state the base/tree stages read. In the tiny mode it is rebuilt by replaying the fixture's
     * chunk plan; in the real mode the recorded full prefix IS the state, so it is seeded directly
     * from the golden cache and there is nothing to replay -- but the APPEND OPS are still exercised,
     * through the full-prefix append below, so the real geometry proves them too. */
    std::vector<float> cache(cache_elems,0.0f);
    std::vector<int32_t> length(length_elems,0);
    if(f.real_mode) {
        f.append_full_prefix(h,cache,length);
    } else {
        f.replay_plan(h,cache,length);
    }
    near(cache.data(),f.g<float>("cache"),cache_elems,tol.atol,tol.rtol,"golden.full_cache");
    for(size_t i=0;i<length_elems;++i)require(length[i]==f.g32("length")[i],"full golden length mismatch");
    std::puts("FULL_APPEND_PASS");

    const std::vector<int32_t> slots=f.slots();
    c.vocab_lo=0;
    Call base0(RAD_DLOOP_FULL_BASE,f.capacity,c);
    std::memcpy(base0.data<float>(0),cache.data(),cache_elems*4);
    std::memcpy(base0.data<int32_t>(1),length.data(),length_elems*4);
    std::memcpy(base0.data<int32_t>(2),slots.data(),slots.size()*4);
    f.anchors(base0,3);f.weights(base0,4);f.shard(base0,5);
    h.run(base0);
    near(base0.data<float>(6),f.g<float>("y_base"),size_t(c.n_seq)*c.proposal_length*c.target_width,tol.atol,tol.rtol,"golden.full_y_base.rank0");
    c.vocab_lo=c.vocab_size_shard;
    Call base1(RAD_DLOOP_FULL_BASE,f.capacity,c);
    std::memcpy(base1.data<float>(0),base0.data<float>(0),cache_elems*4);
    std::memcpy(base1.data<int32_t>(1),base0.data<int32_t>(1),length_elems*4);
    std::memcpy(base1.data<int32_t>(2),slots.data(),slots.size()*4);
    f.anchors(base1,3);f.weights(base1,4);f.shard(base1,5);
    h.run(base1);
    near(base1.data<float>(6),f.g<float>("y_base"),size_t(c.n_seq)*c.proposal_length*c.target_width,tol.atol,tol.rtol,"golden.full_y_base.rank1");
    /* Both ranks' local top-k against their own golden table, by the cutoff-set rule the ring ops
     * use: y_base's FP32 dot products reduce in a different order on device, host and PyTorch, so
     * two ids within the float bound of the cutoff may legitimately trade places. */
    const int rows=c.n_seq*c.proposal_length*c.pool_size;
    Harness::compare_topk_golden("rank0",base0,7,8,f.g32("rank0.local_ids"),f.g<float>("rank0.local_logits"));
    Harness::compare_topk_golden("rank1",base1,7,8,f.g32("rank1.local_ids"),f.g<float>("rank1.local_logits"));
    /* The device's own top-k, not the golden's, feeds the pool stage below. */
    std::puts("FULL_BASE_PASS");

    c.vocab_lo=0;
    /* THE POOL STEP KEEPS ITS OWN CONTRACT. `dloop_pool` is geometry-neutral and unchanged: its
     * schema takes the ring op's 29 parameters and its validator requires context_window 1..32, so
     * it is called with a RING parameter set built from the same fixture geometry -- the full-context
     * ops are the three that take `context_window=0` plus `context_capacity`. Only the base rows it
     * consumes come from the full path; that is the whole point of the stage. */
    auto pool_call=[&]() {
        /* THE POOL'S GEOMETRY TAGS ARE THE RING ONES, and they are tags and nothing else: pool's
         * arithmetic reads S, M, K, Vs, R, W and active_vocab -- never a layer count, a hidden width
         * or a context feature count. Its declared rows are gated on num_layers 2 and
         * context_window 32 (mine/dloop_rows.cpp COMMON); the rest of the tags are the production or
         * tiny set matching the fixture. Passing the full-context values here resolves to no row at
         * all, which is the mismatch this comment exists to prevent someone "fixing" later.
         *
         * The tensors that matter come from this fixture: M, K, Vs, R and W all match, so the rows
         * it compares are the rows the full base produced. */
        RadDLoopConfig ring=c;
        ring.context_window=32;ring.num_layers=2;
        ring.context_features_dim=c.context_features_dim==96?48:c.context_features_dim;
        ring.n_seq=c.n_seq;ring.max_seqs=c.max_seqs;
        return Call(RAD_DLOOP_POOL,ring);
    };
    /* Each rank's shard top-k goes in at its own offset, the anchor table supplies the predecessor
     * for the root, and the gathered teacher/codebook rows stay zero-masked outside the owner so the
     * all-reduce is a bitwise OR. */
    auto fill_pool=[&](Call& pool,const Call& low,const Call& high) {
        pool.copy(0,low.t[7].data,size_t(rows)*4);std::memcpy(pool.data<int32_t>(0)+rows,high.t[7].data,size_t(rows)*4);
        pool.copy(1,low.t[8].data,size_t(rows)*4);std::memcpy(pool.data<float>(1)+rows,high.t[8].data,size_t(rows)*4);
        for(int s=0;s<c.n_seq;++s)pool.data<int32_t>(2)[s]=int32_t(f.g64("anchors")[s]);
        const int lo=pool.rank*c.vocab_size_shard;
        pool.bind(3,const_cast<uint16_t*>(f.table("lm_head.weight"))+int64_t(lo)*c.target_width,RAD_BF16,{c.vocab_size_shard,c.target_width});
        pool.bind(4,const_cast<uint16_t*>(f.table("model.language_model.embed_tokens.weight"))+int64_t(lo)*c.target_width,RAD_BF16,{c.vocab_size_shard,c.target_width});
        pool.bind(5,const_cast<uint8_t*>(get(f.student,"dloop.selector.pred_codebook.weight").data)+int64_t(lo)*c.selector_rank*2,RAD_BF16,{c.vocab_size_shard,c.selector_rank});
        pool.bind(6,const_cast<uint8_t*>(get(f.student,"dloop.selector.succ_codebook.weight").data)+int64_t(lo)*c.selector_rank*2,RAD_BF16,{c.vocab_size_shard,c.selector_rank});
    };
    Call pool0=pool_call();fill_pool(pool0,base0,base1);h.run(pool0);
    c.vocab_lo=c.vocab_size_shard;
    Call pool1=pool_call();fill_pool(pool1,base0,base1);h.run(pool1);
    /* The pool keeps K ids per row sorted by token id, so its retained SET is compared by the floor
     * rule (two ids at the floor may swap) rather than by position; the ring fixture does the same
     * through its exact-id check plus a merged unary bound. */
    Harness::compare_pool_golden(pool0,7,8,f.g32("pool_ids"),f.g<float>("pool_unary"));
    for(int plane=9;plane<=13;++plane) {
        const size_t count=span_bytes(pool0.t[plane])/2;
        for(size_t i=0;i<count;++i) {
            const auto a=pool0.data<uint16_t>(plane)[i],b=pool1.data<uint16_t>(plane)[i];
            require(a==0||b==0,"overlapping full pool owners");
            pool0.data<uint16_t>(plane)[i]=uint16_t(a|b);
        }
    }
    std::puts("FULL_POOL_TWO_SHARD_MERGE_PASS");

    for(int variant=0;variant<2;++variant) {
        c.vocab_lo=0;c.pruned=variant;
        Call tree(RAD_DLOOP_FULL_DARTREE,f.capacity,c);
        /* The full tree's operand positions match the ring tree's after the cache/length/slots/
         * anchor/weights/y_base head, so the pool outputs are spliced in by position. */
        std::memcpy(tree.data<float>(0),cache.data(),cache_elems*4);
        std::memcpy(tree.data<int32_t>(1),length.data(),length_elems*4);
        std::memcpy(tree.data<int32_t>(2),slots.data(),slots.size()*4);
        f.anchors(tree,3);f.weights(tree,4);
        std::memcpy(tree.data<float>(5),base0.data<float>(6),size_t(c.n_seq)*c.proposal_length*c.target_width*4);
        /* Operands 6..12 (pool_ids..anchor_pred) are the pool op's 7..13, position for position. */
        for(int i=6;i<=12;++i)tree.t[i]=pool0.t[i+1];
        std::vector<int32_t> anchors(c.n_seq),packed(c.n_seq*33);
        for(int s=0;s<c.n_seq;++s)anchors[s]=int32_t(f.g64("anchors")[s]);
        tree.bind(16,anchors.data(),RAD_I32,{c.n_seq});tree.bind(17,packed.data(),RAD_I32,{c.n_seq,33});
        h.run(tree);
        const std::string name=variant?"pruned":"fixed";
        check_full_tree_golden(f,tree,name);
        if(!variant)check_non_last_path(f,tree);
        gdn_pipeline(h,tree,c.n_seq,variant);
    }
    for(int ring=0;ring<2;++ring)ple_pipeline(h,ring);
    h.require_coverage(true);std::puts(marker);std::fflush(stdout);
}
/* RADIANCE_DLOOP_DECODE_BENCH=L: the decode step the engine actually issues, timed and nothing else.
 * One sequence (n_seq 1), a verified prefix of L positions, the real sidecar weights and the real
 * teacher head shard, and the three full-context ops launched on the device only -- no host oracle,
 * no golden, no comparison, so it is no parity claim. Everything data-dependent is synthetic but
 * VALID (sorted unique pool ids, finite scores, in-range anchor): the fixed DARTree's frontier is
 * 1,2,2,2,2,2,2,2 whatever the data, so the kernel mix and the grids are the engine's own. The cache
 * capacity is L plus room for the repeats' appends. Output: BENCH lines, one per op and rank. */
/* RADIANCE_DLOOP_DECODE_VERIFY also checks these single-sequence shapes against the host oracle.
 * The multi-sequence golden alone does not exercise decode-specific kernel dispatch. */
void decode_bench(Harness& h,FullFixture& f,int context) {
    auto c=f.c;
    c.n_seq=1;c.max_seqs=1;c.tap_rows=16;c.pruned=0;
    /* RADIANCE_DLOOP_BENCH_BUDGET overrides the window the bench drafts with, so a trace can
     * match the engine's configured budget: the fixture's 15 is the model default, not the
     * engine's, and the fixed variant's quota walk derives its whole frontier from it. */
    if(const char* b=std::getenv("RADIANCE_DLOOP_BENCH_BUDGET")) {
        const int budget=std::atoi(b);
        require(budget>=1&&budget<=15,"RADIANCE_DLOOP_BENCH_BUDGET must be 1..15");
        c.budget=budget;
    }
    const int capacity=context+256,M=c.proposal_length,K=c.pool_size,W=c.target_width;
    uint32_t seed=12345u;
    auto rnd=[&]() {seed=seed*1664525u+1013904223u;return float(int((seed>>9)&0x3fff)-0x2000)/float(0x2000);};
    auto fill_bf=[&](Call& call,int operand,float scale) {
        auto* p=call.data<uint16_t>(operand);const size_t n=span_bytes(call.t[operand])/2;
        for(size_t i=0;i<n;++i)p[i]=rad_f32_to_bf16(rnd()*scale);
    };
    std::printf("DECODE_BENCH context=%d capacity=%d n_seq=1 budget=%d rows=9 per node, fixed tree frontier\n",context,capacity,c.budget);
    const bool verify=std::getenv("RADIANCE_DLOOP_DECODE_VERIFY")!=nullptr;
    auto run=[&](Call& call) {if(verify)h.run(call);else h.run_device_only(call);};
    for(int rank=0;rank<2;++rank) {
        c.vocab_lo=rank*c.vocab_size_shard;
        {
            Call app(RAD_DLOOP_FULL_APPEND,capacity,c);
            for(int plane=0;plane<3;++plane)fill_bf(app,plane,1.0f);
            for(int i=0;i<16;++i)app.data<int32_t>(3)[i]=-1;
            app.data<int32_t>(4)[0]=0;app.data<int32_t>(4)[1]=3;     /* three verified rows: a typical accepted run */
            app.data<int32_t>(5)[0]=0;app.data<int32_t>(6)[0]=0;
            f.weights(app,7);app.data<int32_t>(9)[0]=context;
            run(app);
        }
        {
            Call base(RAD_DLOOP_FULL_BASE,capacity,c);
            base.data<int32_t>(1)[0]=context;base.data<int32_t>(2)[0]=0;
            fill_bf(base,3,0.5f);f.weights(base,4);f.shard(base,5);
            run(base);
        }
        {
            Call tree(RAD_DLOOP_FULL_DARTREE,capacity,c);
            tree.data<int32_t>(1)[0]=context;tree.data<int32_t>(2)[0]=0;
            fill_bf(tree,3,0.5f);f.weights(tree,4);
            for(int i=0;i<M*W;++i)tree.data<float>(5)[i]=rnd();
            for(int m=0;m<M;++m)for(int j=0;j<K;++j) {
                tree.data<int32_t>(6)[m*K+j]=1000*(m+1)+13*j;
                tree.data<float>(7)[m*K+j]=-0.05f*float(j)+0.001f*float(m);
            }
            fill_bf(tree,8,0.5f);fill_bf(tree,9,0.5f);fill_bf(tree,10,0.5f);fill_bf(tree,11,0.5f);fill_bf(tree,12,0.5f);
            std::array<int32_t,1> anchor{5};std::array<int32_t,33> packed{};
            tree.bind(16,anchor.data(),RAD_I32,{1});tree.bind(17,packed.data(),RAD_I32,{1,33});
            run(tree);
        }
    }
    if(verify)std::puts("DLOOP_GPU_DECODE_ORACLE_PASS");
}
}
int main(int argc,char** argv) {
    try {
        require(argc==4,"usage: dloop_gpu_parity_test MINE_SO MINE_REF_SO TINY_FIXTURE_DIR");
        Harness h(argv[1],argv[2]);
        /* `argv[3]` is the ring fixture. DLOOP_FULL_FIXTURES names the full-context fixture (its
         * sibling `dloop_full_tiny` by default), so both contracts are exercised from one invocation.
         * The REAL full-context fixture is a SEPARATE directory selected by the same variable when
         * dloop_testrun.sh runs the real sidecar; it is not this ring golden, and the two never
         * share a fixture. */
        const std::string tiny=argv[3];
        const char* full_env=std::getenv("DLOOP_FULL_FIXTURES");
        const std::string full=full_env&&*full_env?full_env:tiny+"/../dloop_full_tiny";
        /* RADIANCE_DLOOP_REAL_ONLY=1 is for profiling the real-geometry ops under a tracer: it skips
         * the two tiny pipelines (so the trace holds only the production kernels) and ends with a
         * marker that is NOT ALL_PASS, because a skipped pipeline is never a pass. */
        const bool real_only=std::getenv("RADIANCE_DLOOP_REAL_ONLY")!=nullptr||std::getenv("RADIANCE_DLOOP_DECODE_BENCH")!=nullptr;
        if(!real_only) {
            {Fixture fixture(tiny,tiny,tiny+"/teacher-matrices.safetensors");pipeline(h,fixture,"DLOOP_GPU_PARITY_TINY_PASS",kTinyTol);}
            /* The tiny full-context fixture is mandatory: the tiny golden is the RING contract, and this
             * one is the full-context ops' own contract. A missing directory is a failure, not a silently
             * smaller pass, because ALL_PASS must not be reachable without the new ops. */
            {FullFixture fixture(full,full,full+"/teacher-matrices.safetensors");
             full_pipeline(h,fixture,"DLOOP_GPU_PARITY_FULL_PASS",kTinyTol);}
        }
        /* RADIANCE_DLOOP_DECODE_BENCH=L replaces every pipeline: it needs the real golden only to load
         * the real weights, times the engine's decode shape, and ends with a marker that is not a pass. */
        if(const char* decode=std::getenv("RADIANCE_DLOOP_DECODE_BENCH")) {
            const char* golden=std::getenv("RADIANCE_DLOOP_FULL_GOLDEN");const char* sidecar=std::getenv("RADIANCE_DLOOP_SIDECAR");
            const char* teacher=std::getenv("RADIANCE_TEACHER_MATRICES");
            require(golden&&sidecar&&teacher,"the decode bench needs golden, sidecar and teacher paths");
            FullFixture fixture(golden,sidecar,teacher);
            decode_bench(h,fixture,std::atoi(decode));
            std::puts("DLOOP_GPU_DECODE_BENCH_DONE");return 0;
        }
        /* The REAL sidecar, when the driver provides one. Its goldens are generated for the SERVING
         * geometry, so it runs the full-context pipeline (never the ring one) and reports the REAL
         * marker only from the full path: a ring PASS here would prove nothing about the three new
         * ops. `RADIANCE_DLOOP_FULL_GOLDEN` is the real full-context golden directory. */
        if(const char* golden=std::getenv("RADIANCE_DLOOP_FULL_GOLDEN")) {
            const char* sidecar=std::getenv("RADIANCE_DLOOP_SIDECAR");const char* teacher=std::getenv("RADIANCE_TEACHER_MATRICES");
            require(*golden&&sidecar&&*sidecar&&teacher&&*teacher,
                    "real full-context parity requires golden, sidecar and teacher paths");
            FullFixture fixture(golden,sidecar,teacher);
            require(fixture.real_mode,"RADIANCE_DLOOP_FULL_GOLDEN must name a REAL full-context golden directory");
            full_pipeline(h,fixture,"DLOOP_GPU_PARITY_REAL_PASS",kRealTol);
        }
        if(real_only) {require(std::getenv("RADIANCE_DLOOP_FULL_GOLDEN"),"RADIANCE_DLOOP_REAL_ONLY needs the real golden");
                       std::puts("DLOOP_GPU_PARITY_REAL_ONLY_DONE");return 0;}
        h.require_both_scopes();
        std::puts("DLOOP_GPU_PARITY_ALL_PASS");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"DLOOP_GPU_PARITY_FAIL: %s\n",e.what());return 1;}
}
