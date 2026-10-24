/* Pinned libr4d qsa_work -> qsa_block_key -> qsa_tail_store for EVERY accepted
 * token versus one mine qsa_tree_commit. No CPU oracle, skip, or selector fallback
 * can satisfy the byte-exact GPU comparison. --bench reports local replay wall
 * latency (host dispatch plus GPU completion), not engine throughput or queue cost. */
#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "abi/rad_plugin.h"
#include "core/runtime/ctx.h"
#include "arch/common/rad_qsa.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
constexpr size_t guard=64;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void check(int rc, const char* message) {
    if (rc!=RAD_OK) throw std::runtime_error(std::string(message)+": "+rad_strerror(rc));
}
size_t bytes(uint32_t dtype) { return dtype==RAD_BF16 ? 2 : 4; }
struct Session {
    RadStream stream=nullptr;
    Session() {
        check(rad_dev_set(0),"select GPU");
        RadDeviceProps p{};
        check(rad_dev_props(0,&p),"GPU properties");
        require(!p.is_host_backend && std::string(p.arch)=="gfx1201", "requires real gfx1201; no skip pass");
        check(rad_stream_create(&stream,0),"create stream");
    }
    ~Session() { if (stream) rad_stream_destroy(stream); }
};
struct Buffer {
    RadTensor t{};
    std::string name;
    bool mapped=false;
    std::vector<uint8_t> initial, writable;
};
struct Scene {
    int n=128, ratio=4, ring=7, seqs=0, tokens=0, blocks=32, pages=0, capacity=16;
    const char* mode=nullptr;
    std::vector<Buffer> b;
    int add(const char* name, uint32_t type, std::initializer_list<int64_t> shape,
            bool pitched=false, int inner=1, bool mapped=false, int extra_row=0) {
        Buffer v;
        v.name=name; v.mapped=mapped; v.t.dtype=type;
        for (int64_t dim:shape) v.t.shape[v.t.rank++]=dim;
        int64_t stride=inner;
        for (int d=int(v.t.rank)-1; d>=0; --d) {
            v.t.stride[d]=stride;
            stride*=v.t.shape[d];
            if (d>0 && pitched) stride+=8;
            if (d==1) stride+=extra_row;
        }
        v.initial.resize(2*guard+size_t(stride)*bytes(type),0xa5);
        v.writable.resize(v.initial.size(),0);
        b.push_back(std::move(v));
        return int(b.size())-1;
    }
    size_t at(int id, std::initializer_list<int64_t> index) const {
        size_t offset=guard;
        int d=0;
        for (int64_t i:index) offset+=size_t(i*b[size_t(id)].t.stride[d++])*bytes(b[size_t(id)].t.dtype);
        return offset;
    }
    template<class T> void put(int id, std::initializer_list<int64_t> index, T value) {
        auto& v=b[size_t(id)];
        const size_t offset=at(id,index);
        require(offset+sizeof(T)<=v.initial.size()-guard,"fixture out of bounds");
        std::memcpy(v.initial.data()+offset,&value,sizeof value);
    }
    template<class T> T get(int id, std::initializer_list<int64_t> index) const {
        T value;
        std::memcpy(&value,b[size_t(id)].initial.data()+at(id,index),sizeof value);
        return value;
    }
    void allow(int id, std::initializer_list<int64_t> index) {
        auto& v=b[size_t(id)];
        std::fill_n(v.writable.begin()+at(id,index),bytes(v.t.dtype),uint8_t(1));
    }
    void fill(int id) {
        auto& v=b[size_t(id)];
        int64_t count=1;
        for (unsigned d=0; d<v.t.rank; ++d) count*=v.t.shape[d];
        uint32_t rng=uint32_t(id+1)*0x9e3779b9u;
        for (int64_t i=0; i<count; ++i) {
            int64_t rem=i;
            size_t offset=guard;
            for (int d=int(v.t.rank)-1; d>=0; --d) {
                offset+=size_t((rem%v.t.shape[d])*v.t.stride[d])*bytes(v.t.dtype);
                rem/=v.t.shape[d];
            }
            rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
            const float x=std::ldexp(float(int(rng&511)-256)/256.0f,int((rng>>9)%9)-4);
            if (v.t.dtype==RAD_BF16) {
                const uint16_t q=rad_f32_to_bf16(x);
                std::memcpy(v.initial.data()+offset,&q,2);
            } else if (v.t.dtype==RAD_F32) std::memcpy(v.initial.data()+offset,&x,4);
            else {
                const int32_t q=-1;
                std::memcpy(v.initial.data()+offset,&q,4);
            }
        }
    }
};

Scene fixture(const std::vector<int>& lengths, int ring, int position, bool pitched,
              int dim=128, bool absent_slot=false, bool absent_page=false, bool rank2=false,
              const char* mode=nullptr, int capacity=16, int path_width=0) {
    Scene s;
    s.n=dim; s.ring=ring; s.seqs=int(lengths.size()); s.mode=mode; s.capacity=capacity;
    if (!path_width) path_width=capacity;
    const int prefix=pitched ? 2 : 0;
    s.tokens=prefix+capacity*s.seqs+(pitched ? 3 : 0);
    s.blocks=std::max(s.blocks,(position+s.seqs+capacity+s.ratio-1)/s.ratio+1);
    s.pages=2*s.seqs*8+1; // Long logical histories still use a compact, noncontiguous page pool.
    s.add("k",RAD_BF16,{s.tokens,s.n},false,1,false,4*s.n+(pitched ? 8 : 0));
    s.add("pos",RAD_I32,{s.tokens},false,pitched ? 2 : 1);
    s.add("paths",RAD_I32,{s.seqs,path_width},pitched,pitched ? 2 : 1,true);
    s.add("lengths",RAD_I32,{s.seqs},false,pitched ? 2 : 1,true);
    s.add("starts",RAD_I32,{s.seqs},false,pitched ? 2 : 1,true);
    s.add("tail",RAD_BF16,{s.seqs+2,1,s.ring,s.n},pitched);
    s.add("sidx",RAD_I32,{s.seqs,3},pitched);
    s.add("bt",RAD_I32,{s.seqs,s.blocks},pitched);
    s.add("weight",RAD_F32,{s.n});
    if (rank2) s.add("block_keys",RAD_BF16,{s.pages,s.n},false,1,false,pitched ? 16 : 0);
    else s.add("block_keys",RAD_BF16,{s.pages,2,s.ratio,s.n/(2*s.ratio)},false,1,false,pitched ? 16 : 0);
    s.add("stage",RAD_BF16,{2*s.ratio,s.n},pitched);
    s.add("page",RAD_I32,{2});
    if (mode) s.add("bpos",RAD_I32,{2,3});
    else s.add("bpos",RAD_I32,{2});
    s.add("nc",RAD_I32,{1});
    s.add("cu",RAD_I32,{2});
    for (int i=0; i<int(s.b.size()); ++i) s.fill(i);
    if (mode) for (int w=0; w<2; ++w) {
        s.put(12,{w,1},int32_t(1234+w));
        s.put(12,{w,2},int32_t(5678+w));
    }
    s.put(14,{0},int32_t(0)); s.put(14,{1},int32_t(1));
    const int order[]={0,3,7,2,15,1,6,12,4,14,5,10,8,13,9,11};
    for (int d=0; d<s.n; ++d) s.put(8,{d},.5f+float(d%17)*.0078125f);
    for (int seq=0; seq<s.seqs; ++seq) {
        const int start=prefix+capacity*seq, count=lengths[size_t(seq)], p0=position+seq;
        const int slot=absent_slot && seq==s.seqs-1 ? -1 : s.seqs-seq;
        s.put(3,{seq},int32_t(count)); s.put(4,{seq},int32_t(start)); s.put(6,{seq,0},int32_t(slot));
        for (int b=0; b<s.blocks; ++b)
            s.put(7,{seq,b},int32_t((2*(seq*8+b%8)+3)%s.pages));
        if (absent_page) for (int b=0; b<s.blocks; b+=2) s.put(7,{seq,b},int32_t(-1));
        for (int j=0; j<path_width; ++j) {
            const int node=j<16 ? order[j] : j;
            s.put(2,{seq,j},int32_t(node));
            s.put(1,{start+node},int32_t(p0+j));
        }
        if (slot<0) continue;
        for (int j=0; j<count; ++j) {
            const int p=p0+j;
            for (int d=0; d<s.n; ++d) s.allow(5,{slot,0,p%s.ring,d});
            if ((p+1)%s.ratio) continue;
            const int page=s.get<int32_t>(7,{seq,p/s.ratio});
            if (page<0) continue;
            for (int d=0; d<s.n; ++d) {
                if (rank2) s.allow(9,{page,d});
                else s.allow(9,{page,d/(s.ratio*(s.n/(2*s.ratio))),
                               (d/(s.n/(2*s.ratio)))%s.ratio,d%(s.n/(2*s.ratio))});
            }
        }
    }
    return s;
}
struct Side {
    std::vector<void*> memory;
    std::vector<RadTensor> t;
    std::vector<bool> mapped;
    Side(const Session& session, const Scene& scene) {
        for (const auto& b:scene.b) {
            const int kind=b.mapped ? RAD_MEM_HOST_MAPPED : RAD_MEM_DEVICE;
            void* p=rad_dev_alloc(int64_t(b.initial.size()),kind);
            require(p!=nullptr,"allocation failed");
            memory.push_back(p); mapped.push_back(b.mapped);
            RadTensor v=b.t;
            v.data=static_cast<uint8_t*>(p)+guard;
            t.push_back(v);
            if (b.mapped) std::memcpy(p,b.initial.data(),b.initial.size());
            else check(rad_memcpy_async(p,b.initial.data(),int64_t(b.initial.size()),session.stream),"upload");
        }
        check(rad_stream_sync(session.stream),"finish uploads");
    }
    ~Side() {
        for (size_t i=0; i<memory.size(); ++i)
            rad_dev_free(memory[i],mapped[i] ? RAD_MEM_HOST_MAPPED : RAD_MEM_DEVICE);
    }
    Side(const Side&)=delete;
    Side& operator=(const Side&)=delete;
    std::vector<std::vector<uint8_t>> download(const Session& session, const Scene& scene) const {
        std::vector<std::vector<uint8_t>> result;
        for (size_t i=0; i<memory.size(); ++i) {
            result.emplace_back(scene.b[i].initial.size());
            if (!mapped[i]) check(rad_memcpy_async(result.back().data(),memory[i],
                int64_t(result.back().size()),session.stream),"download");
        }
        check(rad_stream_sync(session.stream),"finish downloads");
        for (size_t i=0; i<memory.size(); ++i)
            if (mapped[i]) std::memcpy(result[i].data(),memory[i],result[i].size());
        return result;
    }
};
rad::Geometry geometry(const Scene& s) {
    rad::Geometry g;
    g.set_i("M",s.seqs); g.set_i("n",s.n); g.set_i("ratio",s.ratio); g.set_i("ring",s.ring);
    g.set_i("rotary_dim",s.n/2); g.set_i("block_size",s.ratio); g.set_i("max_tokens",s.tokens);
    g.set_f("eps",1e-6); g.set_f("wadd",1); g.set_f("theta",10000000); g.set_f("scale",1);
    return g;
}
const rad::KernelRow* pin(rad::Registry& reg, const char* op, const char* name) {
    const auto* rows=reg.rows_for(op);
    require(rows!=nullptr,std::string("missing op ")+op);
    const rad::KernelRow* found=nullptr;
    for (const auto* row:*rows) if (row->plugin=="libr4d" && row->info && row->info->name &&
        !std::strcmp(row->info->name,name)) {
        require(!found,"duplicate pinned libr4d row"); found=row;
    }
    require(found && found->info->launch && found->info->domain==RAD_DOMAIN_DEVICE,
            std::string("missing actual libr4d device row ")+name);
    std::printf("ROW plugin=libr4d op=%s kernel=%s\n",op,name);
    return found;
}
rad::Resolved bind(rad::Registry& reg, const rad::KernelRow* row, const rad::Geometry& g) {
    rad::Resolved r;
    r.row=row; r.geom=g;
    check(reg.init_instance(r,0,1),"initialize pinned row");
    return r;
}
rad::Resolved candidate(rad::Registry& reg, const rad::Geometry& g) {
    const auto* schema=reg.schema("qsa_tree_commit");
    require(schema && schema->n_operands==10 && schema->operands[5].role==RAD_OPD_INOUT &&
            schema->operands[9].role==RAD_OPD_INOUT && schema->operands[8].role==RAD_OPD_WEIGHT,
            "missing schema, INOUT state contract, or weight role");
    std::string error;
    const int status=reg.validate("qsa_tree_commit",g.params(),g.n_params(),&error);
    check(status,error.c_str());
    auto r=reg.resolve("qsa_tree_commit",g,RAD_DOMAIN_DEVICE);
    require(r.row && r.row->plugin=="mine" && r.row->info && r.row->info->name &&
            !std::strcmp(r.row->info->name,"mine_qsa_tree_commit"),"wrong selected candidate");
    check(reg.init_instance(r,0,1),"initialize candidate");
    return r;
}
int launch(const rad::Resolved& r, const RadTensor* t, int count, RadStream stream) {
    RadArgs a{};
    a.t=t; a.n_t=count; a.p=r.geom.params(); a.n_p=r.geom.n_params(); a.instance=r.instance;
    a.rank=0; a.world_size=1;
    require(!r.row->info->scratch || r.row->info->scratch(&a)==0,"unexpected device scratch");
    return r.row->info->launch(&a,stream);
}
struct Rows { const rad::KernelRow *work, *key, *tail; };
struct Old { rad::Resolved work,key,tail; };
Old old_rows(rad::Registry& reg, const Rows& rows, const rad::Geometry& g) {
    return {bind(reg,rows.work,g),bind(reg,rows.key,g),bind(reg,rows.tail,g)};
}
int replay(const Session& session, const Scene& scene, const Side& side, const Old& old,
           bool stale_columns=false) {
    int launches=0;
    RadTensor bpos=side.t[12];
    // Mirror the architecture's corrected scalar column, retaining the [work,3]
    // backing pitch. The negative witness deliberately passes the original view.
    if (scene.mode && !stale_columns) bpos.shape[1]=1;
    for (int seq=0; seq<scene.seqs; ++seq) {
        for (int j=0; j<scene.get<int32_t>(3,{seq}); ++j) {
            const int tok=scene.get<int32_t>(4,{seq})+scene.get<int32_t>(2,{seq,j});
            RadTensor k=side.t[0], pos=side.t[1], si=side.t[6], bt=side.t[7];
            k.data=static_cast<uint16_t*>(k.data)+int64_t(tok)*k.stride[0]; k.shape[0]=1;
            pos.data=static_cast<int32_t*>(pos.data)+int64_t(tok)*pos.stride[0]; pos.shape[0]=1;
            si.data=static_cast<int32_t*>(si.data)+int64_t(seq)*si.stride[0]; si.shape[0]=1;
            bt.data=static_cast<int32_t*>(bt.data)+int64_t(seq)*bt.stride[0]; bt.shape[0]=1;
            const RadTensor work[]={k,pos,side.t[14],side.t[5],si,bt,RadTensor{},
                                    side.t[10],side.t[11],bpos,side.t[13]};
            const RadTensor key[]={side.t[10],side.t[8],side.t[11],bpos,side.t[9]};
            const RadTensor tail[]={k,pos,side.t[14],side.t[5],si};
            check(launch(old.work,work,11,session.stream),"old qsa_work");
            check(launch(old.key,key,5,session.stream),"old qsa_block_key");
            check(launch(old.tail,tail,5,session.stream),"old qsa_tail_store");
            launches+=3;
        }
    }
    return launches;
}
size_t compare(const Scene& s, const std::vector<std::vector<uint8_t>>& actual,
               const std::vector<std::vector<uint8_t>>& expected, bool refused) {
    size_t compared=0;
    for (size_t id=0; id<10; ++id) {
        const auto& b=s.b[id];
        for (size_t i=0; i<b.initial.size(); ++i) {
            if (id==5 || id==9) {
                require(actual[id][i]==expected[id][i],b.name+" byte mismatch at "+std::to_string(i));
                ++compared;
            }
            if (refused || !b.writable[i])
                require(actual[id][i]==b.initial[i],b.name+" changed unowned byte "+std::to_string(i));
        }
    }
    return compared;
}
size_t rounded_pool_channels(const Scene& s) {
    size_t rounded=0;
    for (int seq=0; seq<s.seqs; ++seq) {
        const int count=s.get<int32_t>(3,{seq}), slot=s.get<int32_t>(6,{seq,0});
        if (!count || slot<0) continue;
        const int start=s.get<int32_t>(4,{seq});
        const int p0=s.get<int32_t>(1,{start+s.get<int32_t>(2,{seq,0})});
        for (int j=0; j<count; ++j) {
            const int p=p0+j;
            if ((p+1)%s.ratio || s.get<int32_t>(7,{seq,p/s.ratio})<0) continue;
            for (int d=0; d<s.n; ++d) {
                float sum=0;
                for (int i=0; i<s.ratio; ++i) {
                    const int q=p-s.ratio+1+i;
                    const uint16_t raw=q<p0 ? s.get<uint16_t>(5,{slot,0,q%s.ring,d}) :
                        s.get<uint16_t>(0,{start+s.get<int32_t>(2,{seq,q-p0}),d});
                    sum+=rad_bf16_to_f32(raw);
                }
                const float mean=sum/float(s.ratio);
                rounded+=mean!=rad_bf16_to_f32(rad_f32_to_bf16(mean));
            }
        }
    }
    return rounded;
}
size_t run_case(const Session& session, rad::Registry& reg, const Rows& rows, const Scene& scene,
                int& old_launches, size_t& changed_keys, size_t& pool_witnesses) {
    Side expected(session,scene), actual(session,scene);
    const auto g=geometry(scene);
    auto old=old_rows(reg,rows,g);
    if (scene.mode) {
        auto kg=g;
        kg.set_s("mode",scene.mode); kg.set_s("sections","11 11 10");
        old.key=bind(reg,rows.key,kg);
    }
    auto fused=candidate(reg,g);
    const int count=replay(session,scene,expected,old);
    check(launch(fused,actual.t.data(),10,session.stream),"fused commit");
    const auto want=expected.download(session,scene), got=actual.download(session,scene);
    const size_t compared=compare(scene,got,want,false);
    compare(scene,want,want,false); // The old chain must also preserve every unowned state byte.
    if (scene.mode) {
        // This is a numerical defect witness, NOT an oracle for the fused row:
        // the original [work,3] operands read poisoned components never written
        // by qsa_work with no rotary operand. Corrected replay must differ.
        Side stale(session,scene);
        replay(session,scene,stale,old,true);
        const auto bad=stale.download(session,scene);
        require(bad[5]==want[5],"stale-column witness changed tail semantics");
        size_t differing=0;
        for (size_t i=0; i<bad[9].size(); ++i) differing+=bad[9][i]!=want[9][i];
        require(differing>0,"stale rotary columns did not change any block key");
        for (int w=0; w<2; ++w) for (int c=1; c<3; ++c) {
            const size_t offset=scene.at(12,{w,c});
            require(!std::memcmp(want[12].data()+offset,scene.b[12].initial.data()+offset,4) &&
                    !std::memcmp(bad[12].data()+offset,scene.b[12].initial.data()+offset,4),
                    "qsa_work unexpectedly wrote a missing rotary component");
        }
        std::printf("STALE_COLUMNS_WITNESS mode=%s sections='11 11 10' bpos_backing=2x3 "
                    "corrected_columns=1 ratio=4 ring=11 rot=64 differing_key_bytes=%zu\n",
                    scene.mode,differing);
    }
    for (size_t i=0; i<want[9].size(); ++i) changed_keys+=want[9][i]!=scene.b[9].initial[i];
    old_launches+=count;
    pool_witnesses+=rounded_pool_channels(scene);
    std::printf("CASE_PASS seqs=%d ring=%d n=%d tokens=%d mode=%s old_launches=%d fused_launches=1 bytes=%zu\n",
                scene.seqs,scene.ring,scene.n,scene.tokens,scene.mode ? scene.mode : "neox",count,compared);
    return compared;
}

const rad::arch::QsaIndexer* active_indexer=nullptr;
void indexer_step(RadCtx* ctx, const RadBatch* batch) {
    active_indexer->commit(ctx,batch);
}

std::vector<std::vector<uint8_t>> indexer_commit(
        const Session& session, rad::Registry& reg, const Rows& rows, Scene scene, int pitch) {
    // Use the engine's mapped layout, including the suffix cu and per-sequence starts.
    const int metadata=scene.add("commit_metadata",RAD_I32,{scene.seqs*pitch+2+2*scene.seqs},
                                 false,1,true);
    scene.fill(metadata);
    for (int s=0; s<scene.seqs; ++s) {
        for (int j=0; j<pitch; ++j)
            scene.put(metadata,{s*pitch+j},scene.get<int32_t>(2,{s,j}));
        scene.put(metadata,{scene.seqs*pitch+2+s},scene.get<int32_t>(3,{s}));
        scene.put(metadata,{scene.seqs*pitch+2+scene.seqs+s},scene.get<int32_t>(4,{s}));
    }
    scene.put(metadata,{scene.seqs*pitch},int32_t(0));
    scene.put(metadata,{scene.seqs*pitch+1},int32_t(1));
    Side actual(session,scene);
    rad::Program program;
    program.buffers.resize(1); program.weights.resize(2); program.ops.resize(1);
    rad::arch::QsaIndexer indexer;
    auto add_buffer=[&](int id, bool qk=false) {
        const auto& source=scene.b[size_t(id)];
        rad::BufferInfo b;
        b.name=source.name; b.decl.dtype=source.t.dtype; b.decl.rank=source.t.rank;
        b.decl.domain=RAD_DOMAIN_DEVICE;
        for (unsigned d=0; d<source.t.rank; ++d) b.decl.shape[d]=source.t.shape[d];
        if (qk) b.decl.shape[1]=5*scene.n;
        b.bytes=int64_t(source.initial.size()-2*guard);
        b.arena_offset=program.arena_bytes;
        program.arena_bytes=(program.arena_bytes+b.bytes+255)/256*256;
        program.buffers.push_back(b);
        return rad_buf(program.buffers.size()-1);
    };
    indexer.w.qk=add_buffer(0,true); indexer.w.stage=add_buffer(10);
    indexer.w.page=add_buffer(11); indexer.w.bpos=add_buffer(12); indexer.w.nc=add_buffer(13);
    auto add_op=[&](const char* name, const rad::Resolved& resolved) {
        rad::OpInfo op;
        op.op=name; op.schema=reg.schema(name); op.index=int32_t(program.ops.size());
        op.base=resolved.geom; op.ranged_key="M"; op.range_lo=1; op.range_hi=scene.seqs;
        rad::Band band;
        band.hi=scene.seqs; band.dom[RAD_DOMAIN_DEVICE]=resolved;
        op.bands.push_back(band); program.ops.push_back(op);
        return rad_op(program.ops.size()-1);
    };
    const auto g=geometry(scene);
    const auto old=old_rows(reg,rows,g);
    indexer.op_work=add_op("qsa_work",old.work);
    indexer.op_bkey=add_op("qsa_block_key",old.key);
    indexer.op_tail=add_op("qsa_tail_store",old.tail);
    indexer.op_tree_commit=add_op("qsa_tree_commit",candidate(reg,g));
    indexer.op_rope=indexer.op_work; // on(); projection/normalization are outside this commit seam.
    indexer.c.tree_state=true; indexer.c.heads=4; indexer.c.head_dim=scene.n;
    indexer.c.ratio=scene.ratio; indexer.c.ring=scene.ring; indexer.c.topk=16384;
    indexer.c.kv_tail=1; indexer.kv_bk=2; indexer.w_kn=1;
    auto& weight=program.weights[1];
    weight.name="k_norm"; weight.decl.dtype=RAD_F32; weight.decl.rank=1;
    weight.decl.shape[0]=scene.n; weight.ptr=actual.t[8].data;
    rad::CtxDesc desc;
    desc.program=&program; desc.stream=session.stream; desc.arch_step=indexer_step;
    desc.record_passes=false; desc.kv.resize(3);
    for (const auto& binding:{std::pair<int,int>{1,5},{2,9}}) {
        auto& pool=desc.kv[size_t(binding.first)];
        const auto& t=actual.t[size_t(binding.second)];
        pool.base=t.data; pool.dtype=t.dtype; pool.rank=t.rank; pool.layer_slot={0};
        pool.bytes=int64_t(scene.b[size_t(binding.second)].initial.size()-2*guard);
        pool.layer_stride=pool.bytes;
        for (unsigned d=0; d<t.rank; ++d) pool.shape[d]=t.shape[d];
    }
    rad::Ctx ctx;
    check(ctx.init(desc),"initialize real QSA architecture context");
    const rad_buf buffers[]={indexer.w.qk,indexer.w.stage,indexer.w.page,indexer.w.bpos,indexer.w.nc};
    const int sources[]={0,10,11,12,13};
    for (int i=0; i<5; ++i)
        check(rad_memcpy_async(ctx.buf_ptr(buffers[i]),actual.t[size_t(sources[i])].data,
              program.buffers[buffers[i]].bytes,session.stream),"upload architecture buffer");
    // Scene stores raw K at column 0 of a pitched row; the architecture consumes
    // [four query heads, one raw key head], so populate that fifth head explicitly.
    check(rad_memcpy_2d_async(static_cast<uint16_t*>(ctx.buf_ptr(indexer.w.qk))+4*scene.n,
          int64_t(5*scene.n)*sizeof(uint16_t),actual.t[0].data,
          actual.t[0].stride[0]*sizeof(uint16_t),int64_t(scene.n)*sizeof(uint16_t),
          scene.tokens,session.stream),"populate architectural raw key head");
    RadKVGroupBatch kv[2]{};
    kv[0].group=1; kv[0].state_index=static_cast<int32_t*>(actual.t[6].data);
    kv[0].state_index_pitch=3;
    kv[1].group=2; kv[1].block_table=static_cast<int32_t*>(actual.t[7].data);
    kv[1].block_table_pitch=scene.blocks;
    RadBatch batch{};
    batch.phase=RAD_PHASE_DECODE; batch.n_tok=scene.tokens; batch.n_seq=scene.seqs;
    batch.positions=static_cast<int32_t*>(actual.t[1].data);
    batch.spec_parent=batch.positions; // commit only checks that this is a tree pass.
    batch.n_kv_groups=2; batch.kv=kv; batch.n_draft_out=scene.seqs*pitch;
    batch.draft_out_ids=static_cast<int32_t*>(actual.t[size_t(metadata)].data);
    batch.num_accepted=batch.draft_out_ids+batch.n_draft_out+2;
    batch.max_q_len=scene.capacity; batch.max_ctx_len=47069;
    active_indexer=&indexer;
    const int status=ctx.run_step(&batch);
    active_indexer=nullptr;
    check(status,ctx.step_error());
    require(program.ops[indexer.op_tree_commit].issued,
            "QsaIndexer did not select the fused commit");
    for (rad_op op:{indexer.op_work,indexer.op_bkey,indexer.op_tail})
        require(!program.ops[op].issued,"QsaIndexer issued per-token replay");
    const auto got=actual.download(session,scene);
    require(got[size_t(metadata)]==scene.b[size_t(metadata)].initial,"commit metadata changed");
    return got;
}

size_t indexer_wide_pitch(const Session& session, rad::Registry& reg, const Rows& rows) {
    // Node 15 in a compact tree and node 64 in a wide tree carry identical accepted raw keys.
    // Different counts, an empty sequence, and nonzero starts expose cross-sequence pitch errors.
    Scene narrow=fixture({9,0,4,6},11,47065,false,128,false,false,false,nullptr,65);
    for (int s=0; s<narrow.seqs; ++s) {
        const int start=narrow.get<int32_t>(4,{s}), count=narrow.get<int32_t>(3,{s});
        for (int j=0; j<count; ++j) {
            const int node=j==count-1 ? 15 : j;
            narrow.put(2,{s,j},int32_t(node));
            narrow.put(1,{start+node},int32_t(47065+s+j));
        }
    }
    Scene wide=narrow;
    for (int s=0; s<wide.seqs; ++s) {
        const int start=wide.get<int32_t>(4,{s});
        const int count=wide.get<int32_t>(3,{s});
        if (count) wide.put(2,{s,count-1},int32_t(64));
        wide.put(1,{start+64},narrow.get<int32_t>(1,{start+15}));
        for (int d=0; d<wide.n; ++d) {
            wide.put(0,{start+64,d},narrow.get<uint16_t>(0,{start+15,d}));
            wide.put(0,{start+15,d},uint16_t(0x7f7f)); // A truncated/misindexed node must disagree.
        }
    }
    Side oracle(session,narrow);
    const int old_launches=replay(session,narrow,oracle,old_rows(reg,rows,geometry(narrow)));
    const auto want=oracle.download(session,narrow);
    auto compare_arm=[&](const Scene& scene, int pitch) {
        std::printf("QSA_INDEXER_COMMIT_BEGIN pitch=%d branch=fused\n",pitch);
        std::fflush(stdout);
        try {
            const auto got=indexer_commit(session,reg,rows,scene,pitch);
            const size_t bytes=compare(scene,got,want,false);
            std::printf("QSA_INDEXER_COMMIT_ARM_PASS pitch=%d bytes=%zu tolerance=0\n",pitch,bytes);
            return bytes;
        } catch (const std::exception& error) {
            throw std::runtime_error("QsaIndexer pitch="+std::to_string(pitch)+": "+error.what());
        }
    };
    const size_t compact_bytes=compare_arm(narrow,16);
    const size_t compared=compact_bytes+compare_arm(wide,65);
    size_t changed_tail=0, changed_keys=0;
    for (size_t i=0; i<want[5].size(); ++i) changed_tail+=want[5][i]!=narrow.b[5].initial[i];
    for (size_t i=0; i<want[9].size(); ++i) changed_keys+=want[9][i]!=narrow.b[9].initial[i];
    require(changed_tail && changed_keys,"QsaIndexer regression did not materialize state and keys");
    std::printf("QSA_INDEXER_COMMIT_PASS pitches=16,65 seqs=4 lengths=9,0,4,6 selected_node=64 "
                "bytes=%zu changed_tail_bytes=%zu changed_key_bytes=%zu tolerance=0 "
                "oracle=pinned-libr4d-GPU actual=QsaIndexer::commit "
                "old_launches=%d fused_launches=1 launch_saving=%d\n",
                compared,changed_tail,changed_keys,old_launches,old_launches-1);
    return compared;
}

size_t indexer_long_paths(const Session& session, rad::Registry& reg, const Rows& rows) {
    size_t compared=0;
    for (const auto& lengths:{std::vector<int>{65},std::vector<int>{4,0,65,9}}) {
        const Scene scene=fixture(lengths,11,47065,false,128,false,false,false,nullptr,65);
        Side oracle(session,scene);
        const int old_launches=replay(session,scene,oracle,old_rows(reg,rows,geometry(scene)));
        const auto want=oracle.download(session,scene);
        const auto got=indexer_commit(session,reg,rows,scene,65);
        const size_t bytes=compare(scene,got,want,false);
        compare(scene,want,want,false);
        compared+=bytes;
        std::printf("QSA_INDEXER_LONG_COMMIT_PASS pitch=65 seqs=%d max_path=65 ring=11 "
                    "bytes=%zu old_launches=%d fused_launches=1 launch_saving=%d "
                    "tolerance=0 oracle=pinned-libr4d-GPU actual=QsaIndexer::commit\n",
                    scene.seqs,bytes,old_launches,old_launches-1);
    }
    return compared;
}

size_t refusals(const Session& session, rad::Registry& reg, int capacity=16) {
    const char* faults[]={"last-length","last-path","last-start","last-position","last-slot",
        "last-page","duplicate-slot","duplicate-page","tail-alias","key-dtype","table-stride",
        "output-width","unsupported-ring"};
    size_t count=0;
    for (int fault=0; fault<13; ++fault) {
        const int length=capacity==16 ? 4 : capacity;
        Scene scene=fixture({length,length},7,3,true,128,false,false,false,nullptr,capacity);
        const int last=scene.seqs-1, start=scene.get<int32_t>(4,{last});
        if (fault==0) scene.put(3,{last},int32_t(capacity+1));
        if (fault==1) scene.put(2,{last,3},int32_t(capacity));
        if (fault==2) scene.put(4,{last},int32_t(-1));
        if (fault==3) scene.put(1,{start+scene.get<int32_t>(2,{last,3})},int32_t(100));
        if (fault==4) scene.put(6,{last,0},int32_t(scene.seqs+2));
        if (fault==5) scene.put(7,{last,1},int32_t(scene.pages));
        if (fault==6) scene.put(6,{last,0},scene.get<int32_t>(6,{0,0}));
        if (fault==7) scene.put(7,{last,1},scene.get<int32_t>(7,{0,0}));
        Side actual(session,scene);
        auto g=geometry(scene);
        if (fault==12) g.set_i("ring",33);
        auto fused=candidate(reg,g);
        if (fault==8) actual.t[5].data=actual.t[0].data;
        if (fault==9) actual.t[0].dtype=RAD_F32;
        if (fault==10) actual.t[7].stride[1]=2;
        if (fault==11) actual.t[9].shape[3]-=1;
        const int rc=launch(fused,actual.t.data(),10,session.stream);
        if (fault<8) require(rc==RAD_OK,"device metadata must refuse asynchronously without sync");
        else require(rc<0,"invalid argument accepted");
        const auto got=actual.download(session,scene);
        std::vector<std::vector<uint8_t>> initial;
        for (const auto& b:scene.b) initial.push_back(b.initial);
        compare(scene,got,initial,true);
        std::printf("REFUSAL_PASS capacity=%d fault=%s launch_status=%d all_state_untouched=1\n",
                    capacity,faults[fault],rc);
        ++count;
    }
    return count;
}

size_t wide_limit_refusals(const Session& session, rad::Registry& reg) {
    const char* faults[]={"logical-path-length","unsupported-path-width","logical-node-capacity","sequence-end"};
    for (int fault=0; fault<4; ++fault) {
        Scene scene=fixture({9,9},11,3,true,128,false,false,false,nullptr,65,
                            fault==0 || fault==2 ? 16 : 65);
        auto g=geometry(scene);
        if (fault==0) scene.put(3,{scene.seqs-1},int32_t(17));
        if (fault==2) scene.put(2,{scene.seqs-1,8},int32_t(64));
        if (fault==3) {
            scene.put(2,{0,8},int32_t(64));
            scene.put(4,{1},scene.get<int32_t>(4,{0})+int32_t(64));
        }
        Side actual(session,scene);
        if (fault==1) actual.t[2].shape[1]=32;
        auto fused=candidate(reg,g);
        const int rc=launch(fused,actual.t.data(),10,session.stream);
        require(fault==1 ? rc==RAD_E_SHAPE : rc==RAD_OK,"wrong wide-limit refusal status");
        const auto got=actual.download(session,scene);
        std::vector<std::vector<uint8_t>> initial;
        for (const auto& b:scene.b) initial.push_back(b.initial);
        compare(scene,got,initial,true);
        std::printf("REFUSAL_PASS capacity=65 fault=%s launch_status=%d all_state_untouched=1\n",
                    faults[fault],rc);
    }
    return 4;
}
struct HostSide {
    std::vector<std::vector<uint8_t>> memory;
    std::vector<RadTensor> t;
    explicit HostSide(const Scene& scene) {
        for (int i=0; i<10; ++i) {
            memory.push_back(scene.b[size_t(i)].initial);
            RadTensor v=scene.b[size_t(i)].t;
            v.data=memory.back().data()+guard;
            t.push_back(v);
        }
    }
};
int bf16_ulp(uint16_t a, uint16_t b) {
    const int x=(a&0x8000) ? 32768-int(a&0x7fff) : 32768+int(a);
    const int y=(b&0x8000) ? 32768-int(b&0x7fff) : 32768+int(b);
    return x>y ? x-y : y-x;
}
void host_smoke(const Session& session, rad::Registry& reg, const Rows& rows, int capacity=16) {
    const Scene scene=fixture({4,capacity},7,3,true,128,false,false,false,nullptr,capacity);
    auto g=geometry(scene);
    // Exclude libm RoPE differences from this host-only bound. The primary GPU
    // oracle still exercises rotary_dim=64, including absolute position47065.
    g.set_i("rotary_dim",0);
    Side expected(session,scene);
    auto old=old_rows(reg,rows,g);
    replay(session,scene,expected,old);
    const auto want=expected.download(session,scene);
    const auto* available=reg.rows_for("qsa_tree_commit");
    const rad::KernelRow* row=nullptr;
    require(available!=nullptr,"missing host-smoke op");
    for (const auto* r:*available) if (r->plugin=="mine_ref" && r->info &&
        r->info->domain==RAD_DOMAIN_HOST && r->info->name &&
        !std::strcmp(r->info->name,"ref_qsa_tree_commit")) {
        require(!row,"duplicate host oracle"); row=r;
    }
    require(row && row->info->launch && row->info->scratch,"missing scratch-backed host oracle");
    auto host=bind(reg,row,g);
    HostSide good(scene);
    RadArgs a{};
    a.t=good.t.data(); a.n_t=10; a.p=g.params(); a.n_p=g.n_params();
    a.instance=host.instance; a.rank=0; a.world_size=1;
    const int64_t need=row->info->scratch(&a);
    require(need>0 && need<65536,"unexpected host scratch bound");
    std::vector<uint32_t> scratch(size_t((need+3)/4)+32,0xa5a5a5a5u);
    a.scratch=scratch.data()+16; a.scratch_bytes=need;
    check(row->info->launch(&a,nullptr),"host winner replay");
    size_t values=0, tail_bytes=0;
    int maximum_ulp=0;
    for (size_t id=0; id<10; ++id) {
        const auto& b=scene.b[id];
        for (size_t i=0; i<b.initial.size(); ++i) {
            if (id==5) {
                require(good.memory[id][i]==want[id][i],"host tail is not byte-exact");
                ++tail_bytes;
            }
            if (id==9 && b.writable[i] && !(i&1)) {
                uint16_t actual, reference;
                std::memcpy(&actual,good.memory[id].data()+i,2);
                std::memcpy(&reference,want[id].data()+i,2);
                require(std::isfinite(rad_bf16_to_f32(actual)) && std::isfinite(rad_bf16_to_f32(reference)),
                        "nonfinite host block key");
                const int distance=bf16_ulp(actual,reference);
                // Pooling and reduction order match. CPU 1/sqrtf and GPU rsqrtf
                // may straddle one BF16 rounding boundary; no wider bound or
                // claim of host/GPU bit identity is made, and failures stay failures.
                require(distance<=1,"host normalized block key exceeds one BF16 ULP");
                maximum_ulp=std::max(maximum_ulp,distance); ++values;
            }
            if (!b.writable[i])
                require(good.memory[id][i]==b.initial[i],"host changed input, padding, or unowned state");
        }
    }
    require(values>0 && tail_bytes>0 && rounded_pool_channels(scene)>0,"empty host numeric coverage");
    for (size_t i=0; i<16; ++i)
        require(scratch[i]==0xa5a5a5a5u && scratch[scratch.size()-1-i]==0xa5a5a5a5u,
                "host scratch canary changed");
    HostSide short_scratch(scene);
    a.t=short_scratch.t.data(); a.scratch_bytes=need-1;
    require(row->info->launch(&a,nullptr)==RAD_E_SCRATCH,"host accepted short scratch");
    for (size_t i=0; i<10; ++i)
        require(short_scratch.memory[i]==scene.b[i].initial,"host short-scratch refusal wrote state");
    HostSide invalid(scene);
    const int32_t bad=capacity+1;
    std::memcpy(invalid.memory[3].data()+scene.at(3,{scene.seqs-1}),&bad,4);
    const auto before=invalid.memory;
    a.t=invalid.t.data(); a.scratch_bytes=need;
    require(row->info->launch(&a,nullptr)==RAD_E_INVAL,"host accepted invalid last sequence");
    require(invalid.memory==before,"host invalid-last-sequence refusal wrote any operand");
    std::printf("QSA_HOST_ORACLE_PASS capacity=%d cases=1 refusals=2 tail_bytes=%zu block_values=%zu "
                "max_bf16_ulp=%d allowed_bf16_ulp=1 rotary_dim=0 oracle=pinned-libr4d-GPU "
                "inputs_and_canaries_exact=1\n",capacity,tail_bytes,values,maximum_ulp);
}
void bench(const Session& session, rad::Registry& reg, const Rows& rows) {
    const Scene scene=fixture({4},7,6,false);
    Side baseline(session,scene), candidate_side(session,scene);
    auto g=geometry(scene);
    auto old=old_rows(reg,rows,g);
    auto fused=candidate(reg,g);
    constexpr int repeats=200;
    for (int i=0; i<5; ++i) {
        replay(session,scene,baseline,old);
        check(launch(fused,candidate_side.t.data(),10,session.stream),"fused warmup");
    }
    check(rad_stream_sync(session.stream),"warmup complete");
    auto t0=std::chrono::steady_clock::now();
    for (int i=0; i<repeats; ++i) replay(session,scene,baseline,old);
    check(rad_stream_sync(session.stream),"old local timing complete");
    auto t1=std::chrono::steady_clock::now();
    for (int i=0; i<repeats; ++i)
        check(launch(fused,candidate_side.t.data(),10,session.stream),"fused local timing");
    check(rad_stream_sync(session.stream),"fused local timing complete");
    auto t2=std::chrono::steady_clock::now();
    const double old_us=std::chrono::duration<double,std::micro>(t1-t0).count()/repeats;
    const double new_us=std::chrono::duration<double,std::micro>(t2-t1).count()/repeats;
    std::printf("QSA_LOCAL_WALL_TIMING repeats=%d seqs=1 path=4 old_us=%.3f fused_us=%.3f "
                "scope=host-dispatch-plus-GPU-completion not_engine_throughput=1 not_pure_queue_cost=1\n",
                repeats,old_us,new_us);
}
}

int main(int argc, char** argv) {
    try {
        std::string mine="./mine.so", libr4d="./libr4d.so", reference;
        bool timing=false;
        for (int i=1; i<argc; ++i) {
            const std::string arg=argv[i];
            if (arg=="--help") {
                std::puts("qsa-tree-commit-gpu-test [--mine PATH] [--libr4d PATH] [--ref PATH] [--bench]"); return 0;
            }
            if (arg=="--bench") { timing=true; continue; }
            require((arg=="--mine" || arg=="--libr4d" || arg=="--ref") && i+1<argc,
                    "unknown/incomplete argument "+arg);
            if (arg=="--mine") mine=argv[++i];
            else if (arg=="--libr4d") libr4d=argv[++i];
            else reference=argv[++i];
        }
        Session session;
        rad::Registry reg;
        reg.set_code_device(0);
        check(reg.load_plugin(mine,0),"load mine");
        check(reg.load_plugin(libr4d,1),"load libr4d");
        if (!reference.empty()) check(reg.load_plugin(reference,2),"load host oracle mine_ref");
        check(reg.check_complete(),"schema completeness");
        require(reg.plugin("mine") && reg.plugin("libr4d"),"wrong plugin identity");
        std::printf("PLUGIN candidate=%s oracle=%s\n",mine.c_str(),libr4d.c_str());
        Rows rows{pin(reg,"qsa_work","qsa_work_bf16"),pin(reg,"qsa_block_key","qsa_block_key_bf16"),
                  pin(reg,"qsa_tail_store","qsa_tail_store_bf16")};
        size_t cases=0, compared=0, changed_keys=0, pool_witnesses=0;
        int old_launches=0;
        for (int count:{1,3,4,16}) for (int ring:{4,7}) {
            compared+=run_case(session,reg,rows,fixture({count},ring,ring-1,false),old_launches,changed_keys,pool_witnesses); ++cases;
            compared+=run_case(session,reg,rows,fixture({count,0,1,count},ring,3,true),old_launches,changed_keys,pool_witnesses); ++cases;
        }
        compared+=run_case(session,reg,rows,fixture({16,3},19,0,true,64,false,true,true),old_launches,changed_keys,pool_witnesses); ++cases;
        compared+=run_case(session,reg,rows,fixture({16,4},7,37,true,256,true),old_launches,changed_keys,pool_witnesses); ++cases;
        compared+=run_case(session,reg,rows,fixture({4,16},7,47065,true),old_launches,changed_keys,pool_witnesses); ++cases;
        for (const char* mode:{"imrope","mrope"}) {
            compared+=run_case(session,reg,rows,
                fixture({4,16},11,47065,false,128,false,false,false,mode),
                old_launches,changed_keys,pool_witnesses);
            ++cases;
        }
        require(cases==21,"existing narrow cases did not all run");
        for (int ring:{4,11,32}) {
            compared+=run_case(session,reg,rows,
                fixture({65},ring,47065,false,128,false,false,false,nullptr,65),
                old_launches,changed_keys,pool_witnesses); ++cases;
        }
        compared+=run_case(session,reg,rows,
            fixture({4,0,65,9},11,47065,true,128,false,true,false,nullptr,65),
            old_launches,changed_keys,pool_witnesses); ++cases;
        Scene wide_node=fixture({9,0,4,6},11,47065,true,128,false,false,false,nullptr,65);
        for (int s=0; s<wide_node.seqs; ++s) {
            const int count=wide_node.get<int32_t>(3,{s}), start=wide_node.get<int32_t>(4,{s});
            if (count) {
                wide_node.put(2,{s,count-1},int32_t(64));
                wide_node.put(1,{start+64},int32_t(47065+s+count-1));
            }
        }
        compared+=run_case(session,reg,rows,wide_node,old_launches,changed_keys,pool_witnesses); ++cases;
        const size_t rejected=refusals(session,reg)+refusals(session,reg,65)+wide_limit_refusals(session,reg);
        const size_t indexer_bytes=indexer_wide_pitch(session,reg,rows)+indexer_long_paths(session,reg,rows);
        require(cases==26 && rejected==30 && compared>0 && old_launches>0 && changed_keys>0 &&
                pool_witnesses>0,"incomplete/zero coverage");
        if (!reference.empty()) for (int capacity:{16,65}) host_smoke(session,reg,rows,capacity);
        if (timing) bench(session,reg,rows);
        std::printf("QSA_TREE_COMMIT_GPU_PASS indexer_bytes=%zu cases=%zu refusals=%zu bytes=%zu changed_key_bytes=%zu "
                    "pooled_BF16_rounding_witnesses=%zu old_launches=%d fused_launches=%zu "
                    "tolerance=0 oracle=pinned-libr4d-GPU\n",
                    indexer_bytes,cases,rejected,compared,changed_keys,pool_witnesses,old_launches,cases);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"QSA_TREE_COMMIT_GPU_FAIL: %s\n",error.what()); return 1;
    }
}
