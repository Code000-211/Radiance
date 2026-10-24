#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include "rad_tree_candidate.h"
#include "../../vendor/nlohmann/json.hpp"

namespace radiance {

/* Loader and packing are host-only. No device discovery is needed to validate a sidecar. */
struct DLoopSidecar {
    using Json = nlohmann::json;
    struct Tensor { std::string name; std::vector<int64_t> shape; };
    Json config;
    /* The cache bound the manifest declared, as a top-level key. This is C: the number of verified
     * positions each slot's persistent K/V cache can hold, and the value dloop_full_append's
     * `context_capacity` parameter must take. Never a sliding window -- an append that would
     * exceed it is refused by the op rather than dropping a prefix row. */
    int64_t context_capacity = 0;
    std::vector<uint16_t> backbone, pred, succ;

    /* THE SERVING GEOMETRY, VERSION 2: five layers and context_window 0.
     *
     * context_window 0 is not "a short window": the drafter reads the whole verified prefix out of
     * its own persistent K/V cache, so no window exists to name. The cache's ALLOCATION BOUND is
     * `context_capacity`, which is a top-level manifest key and not a config field, because it is
     * a property of the deployment's --max-ctx rather than of the trained model. A loader that
     * accepted a non-zero window here would silently be validating a model nobody serves.
     *
     * num_layers 5 is the production depth. The L2 ring model that version 1 described is a
     * different contract entirely and is no longer loadable: the ops changed shape with it. */
    static Json production_config() {
        return {{"hidden_size",768},{"intermediate_size",2304},{"num_layers",5},
                {"num_q_heads",12},{"num_kv_heads",4},{"head_dim",64},
                {"context_features_dim",7680},{"target_width",2560},{"proposal_length",8},
                {"context_window",0},{"active_vocab",248077},{"vocab_size",248320},
                {"conv_kernel_size",2},{"conv_group_size",16},{"selector_rank",256},
                {"selector_top_k",16},{"eps",1e-6},{"rope_theta",10000000.0}};
    }

    /* The `config` keys version 2 requires, in full: the eighteen the geometry is stated with plus
     * `use_kv_cache`. NAMED RATHER THAN COUNTED, and that is the point -- a size comparison passes
     * a manifest that renamed one key and added another, while the loader would then be validating
     * a config it only half understood.
     *
     * `context_capacity` is deliberately NOT here: it is a top-level key, because it describes the
     * deployment's cache allocation and not the trained model. A serving geometry with a smaller
     * --max-ctx than the cache bound is legal; one asking for more is refused at load. */
    static std::vector<std::string> config_keys() {
        std::vector<std::string> v;
        const Json c = production_config();
        for (auto it = c.begin(); it != c.end(); ++it) v.push_back(it.key());
        v.push_back("use_kv_cache");
        return v;
    }

    struct TeacherLayout { uint64_t head=0, embedding=0, tensor_bytes=0; };
    static bool teacher_layout(const std::string& path,int64_t vocab,int64_t width,
                               TeacherLayout& layout,std::string& err) {
        try {
            if (vocab<=0 || width<=0 || vocab>std::numeric_limits<int64_t>::max()/width/2) {
                err="DLoop teacher: invalid matrix geometry";return false;
            }
            std::ifstream f(path,std::ios::binary|std::ios::ate);
            if (!f || f.tellg()<8) {err="DLoop teacher: cannot open valid tensor file "+path;return false;}
            const uint64_t bytes=static_cast<uint64_t>(f.tellg());
            f.seekg(0);uint64_t hn=0;f.read(reinterpret_cast<char*>(&hn),8);
            if (hn>bytes-8 || hn>16*1024*1024) {err="DLoop teacher: invalid header length";return false;}
            std::string text(static_cast<size_t>(hn),'\0');f.read(text.data(),static_cast<std::streamsize>(hn));
            if (!f) {err="DLoop teacher: truncated header";return false;}
            const auto h=Json::parse(text);
            TeacherLayout found;found.tensor_bytes=static_cast<uint64_t>(vocab*width*2);
            for (const char* name:{"lm_head.weight","model.language_model.embed_tokens.weight"}) {
                const auto& t=h.at(name);
                const auto off=t.at("data_offsets").get<std::vector<uint64_t>>();
                if (t.at("dtype")!="BF16" || t.at("shape")!=Json::array({vocab,width}) ||
                    off.size()!=2 || off[0]>off[1] || off[1]>bytes-8-hn ||
                    off[1]-off[0]!=found.tensor_bytes) {
                    err=std::string("DLoop teacher: wrong shape/dtype/extent for ")+name;return false;
                }
                for (const auto& d:t.at("shape"))
                    if (!d.is_number_integer()) {err="DLoop teacher: noninteger shape";return false;}
                if (std::strcmp(name,"lm_head.weight")==0) found.head=8+hn+off[0];
                else found.embedding=8+hn+off[0];
            }
            const uint64_t gap=found.head>found.embedding?found.head-found.embedding:found.embedding-found.head;
            if (gap<found.tensor_bytes) {err="DLoop teacher: overlapping matrix extents";return false;}
            layout=found;return true;
        } catch (const std::exception& e) {err=std::string("DLoop teacher: ")+e.what();return false;}
    }
    static bool validate_teacher(const std::string& path,int64_t vocab,int64_t width,std::string& err) {
        TeacherLayout layout;
        return teacher_layout(path,vocab,width,layout,err);
    }

    static std::vector<Tensor> tensors(const Json& c) {
        const int64_t h=c.at("hidden_size"), f=c.at("context_features_dim"),
            w=c.at("target_width"), ff=c.at("intermediate_size"),
            kv=c.at("num_kv_heads").get<int64_t>()*c.at("head_dim").get<int64_t>(),
            r=c.at("selector_rank"), v=c.at("vocab_size"),
            ng=h/c.at("conv_group_size").get<int64_t>();
        std::vector<Tensor> ts;
        auto add=[&](std::string n,std::vector<int64_t> s){ ts.push_back({"dloop."+n,std::move(s)}); };
        add("context_proj.weight",{h,f}); add("context_norm.weight",{h});
        add("token_proj.weight",{h,w}); add("mask_token",{h});
        for (int i=0;i<c.at("num_layers").get<int>();++i) {
            const std::string p="layers."+std::to_string(i)+".";
            add(p+"input_norm.weight",{h});
            add(p+"attention_conv.base_kernel",{2,2,h});
            add(p+"attention_conv.kernel_projection.weight",{4*ng,h});
            add(p+"attention.q_proj.weight",{h,h}); add(p+"attention.k_proj.weight",{kv,h});
            add(p+"attention.v_proj.weight",{kv,h}); add(p+"attention.o_proj.weight",{h,h});
            add(p+"post_attention_norm.weight",{h});
            add(p+"mlp_conv.base_kernel",{2,2,h});
            add(p+"mlp_conv.kernel_projection.weight",{4*ng,h});
            add(p+"mlp.gate_proj.weight",{ff,h}); add(p+"mlp.up_proj.weight",{ff,h});
            add(p+"mlp.down_proj.weight",{h,ff});
        }
        add("final_norm.weight",{h}); add("output_proj.weight",{w,h});
        add("selector.hidden_proj.weight",{r,w});
        add("selector.pred_codebook.weight",{v,r}); add("selector.succ_codebook.weight",{v,r});
        return ts;
    }

    static bool load(const std::string& dir, const Json& expected, int rank, int world,
                     DLoopSidecar& out, std::string& err) {
        auto fail=[&](const std::string& s){err="DLoop sidecar: "+s; return false;};
        try {
            if (dir.empty()) return fail("RADIANCE_DLOOP_SIDECAR must name a directory");
            if (world<1 || rank<0 || rank>=world) return fail("invalid tensor-parallel rank");
            std::ifstream mf(dir+"/dloop_sidecar.json");
            if (!mf) return fail("cannot open "+dir+"/dloop_sidecar.json");
            Json m; mf>>m;
            if (m.at("format")!="radiance-dloop-sidecar" || !m.at("version").is_number_integer() ||
                m.at("version")!=2 || m.at("architecture")!="dloop_diffusion")
                return fail("unsupported format/version/architecture (this build serves version 2)");
            if (!m.contains("config") || !m.at("config").is_object())
                return fail("manifest carries no config object");
            /* EVERY REQUIRED KEY, BY NAME. A manifest missing `context_window` would otherwise be
             * compared against a geometry that has one and fail as "config does not match", which
             * is true and says nothing about which key was wrong. */
            const Json& mc=m.at("config");
            for (const std::string& k:config_keys())
                if (!mc.contains(k)) return fail("config is missing required key: "+k);
            if (config_keys().size()!=mc.size())
                return fail("config carries "+std::to_string(mc.size())+" keys; version 2 states "+
                            std::to_string(config_keys().size()));
            if (!mc.at("use_kv_cache").is_boolean()) return fail("config field must be boolean: use_kv_cache");
            for (const std::string& k:config_keys()) {
                const Json& v=mc.at(k);
                const bool integral=production_config().contains(k) &&
                                    production_config().at(k).is_number_integer();
                if (integral && !v.is_number_integer()) return fail("config field must be integer: "+k);
            }
            Json actual=mc,geometry=expected;
            for (Json* config:{&actual,&geometry}) config->erase("use_kv_cache");
            if (actual!=geometry) return fail("config does not match the declared DLoop geometry");
            /* THE TWO FIELDS A CONFIG COMPARISON CANNOT CARRY, because they are the ones the old
             * contract stated differently: a non-zero window is a model this build cannot serve,
             * and a layer count that is not the five the full-context ops project is a different
             * backbone. Both are named rather than left to surface as a tensor mismatch. */
            if (mc.at("context_window").get<int64_t>()!=0)
                return fail("context_window must be 0: version 2 reads the whole verified prefix");
            if (mc.at("num_layers").get<int64_t>()!=5)
                return fail("num_layers must be 5 for the full-context serving geometry");
            /* THE CACHE BOUND, top-level and REQUIRED. It is an allocation bound and never a
             * window: the drafter refuses an append past it rather than dropping a prefix row. A
             * missing key is the version-1 manifest, and saying so beats a null-dereference in the
             * first op's validator. */
            /* THE SAME UPPER BOUND THE OP VALIDATOR STATES (dloop_full_config). The cache is
             * F32[Q,5,C,2D] per rank, so a capacity far past any real context is not a bigger
             * deployment but an allocation nobody asked for -- and a bound the loader and the
             * validator disagree about is one of them accepting a number the other will refuse at
             * the first launch. */
            if (!m.contains("context_capacity") || !m.at("context_capacity").is_number_integer() ||
                m.at("context_capacity").get<int64_t>()<=0 ||
                m.at("context_capacity").get<int64_t>()>1048576)
                return fail("context_capacity must be an integer in [1, 1048576]");
            const int64_t capacity=m.at("context_capacity").get<int64_t>();
            /* NO SILENT NARROWING OF A SERVING RUN: a --max-ctx the cache cannot hold is a
             * deployment mistake, and the alternative to refusing it is an engine that fails closed
             * at a chunk boundary nobody can predict. THE TWO NUMBERS ARE CHECKED WHERE BOTH ARE
             * KNOWN -- the block's declare has the engine's context and refuses there, naming both
             * -- so this loader validates the manifest against itself and adds no key of its own. */
            /* PROVENANCE, NOT A GATE. The key says where this artifact came from and stays a
             * boolean; `true` means an operator promoted it, which is a fact about the file and not
             * a reason to refuse it. A loader that refused a promoted sidecar would be refusing the
             * artifact the export path exists to produce -- the only thing a promotion changes is
             * which directory it was copied into. Whether a checkpoint may be SERVED is a decision
             * the deployment makes, and it is not this manifest's to make. */
            if (!m.contains("source_promoted") || !m.at("source_promoted").is_boolean())
                return fail("source_promoted must be a boolean");
            const Json defaults={{"variant","fixed"},{"budget",15},{"width",12},{"candidates",64},{"depth_bonus",-0.2}};
            if (!m.contains("tree_defaults") || m.at("tree_defaults")!=defaults)
                return fail("tree_defaults do not match version 2");
            if (!m.at("source_checkpoint").is_string() || !m.at("source_sha256").is_string() ||
                !m.at("completed_steps").is_number_integer() || m.at("completed_steps").get<int64_t>()<0)
                return fail("invalid source provenance");
            const auto sha=m.at("source_sha256").get<std::string>();
            if (sha.size()!=64 || sha.find_first_not_of("0123456789abcdefABCDEF")!=std::string::npos)
                return fail("invalid source SHA256");
            const auto ts=tensors(expected);
            if (!m.at("tensors").is_array() || m.at("tensors").size()!=ts.size()) return fail("tensor count mismatch");
            std::ifstream sf(dir+"/dloop_sidecar.safetensors",std::ios::binary|std::ios::ate);
            if (!sf) return fail("cannot open tensor file");
            const auto end=sf.tellg();
            if (end<8) return fail("truncated tensor file");
            const uint64_t file_bytes=static_cast<uint64_t>(end);
            sf.seekg(0); uint64_t hn=0; sf.read(reinterpret_cast<char*>(&hn),8);
            if (hn>file_bytes-8 || hn>16*1024*1024) return fail("invalid tensor header length");
            std::string hdr(static_cast<size_t>(hn),'\0'); sf.read(hdr.data(),static_cast<std::streamsize>(hn));
            const Json header=Json::parse(hdr);
            size_t nt=header.size()-(header.contains("__metadata__")?1:0);
            if (nt!=ts.size()) return fail("safetensors tensor count mismatch");
            const int64_t vocab=expected.at("vocab_size"), r=expected.at("selector_rank");
            if (vocab%world) return fail("vocabulary cannot be evenly sharded");
            DLoopSidecar loaded; loaded.config=expected; loaded.context_capacity=capacity;
            std::vector<std::pair<uint64_t,uint64_t>> extents;
            for (size_t i=0;i<ts.size();++i) {
                const auto& t=ts[i]; const auto& mt=m.at("tensors").at(i);
                if (mt.at("name")!=t.name || mt.at("shape")!=t.shape || mt.at("dtype")!="BF16")
                    return fail("manifest tensor mismatch at "+t.name);
                if (!header.contains(t.name)) return fail("missing tensor "+t.name);
                const auto& ht=header.at(t.name);
                if (ht.at("dtype")!="BF16" || ht.at("shape")!=t.shape) return fail("shape/dtype mismatch: "+t.name);
                for (const auto& d:mt.at("shape")) if (!d.is_number_integer()) return fail("noninteger manifest shape: "+t.name);
                for (const auto& d:ht.at("shape")) if (!d.is_number_integer()) return fail("noninteger tensor shape: "+t.name);
                const auto off=ht.at("data_offsets").get<std::vector<uint64_t>>();
                uint64_t n=1; for (int64_t d:t.shape) {
                    if (d<=0 || n>std::numeric_limits<uint64_t>::max()/static_cast<uint64_t>(d)) return fail("invalid shape: "+t.name);
                    n*=static_cast<uint64_t>(d);
                }
                if (off.size()!=2 || off[0]>off[1] || off[1]>file_bytes-8-hn || off[1]-off[0]!=n*2)
                    return fail("invalid data extent: "+t.name);
                extents.emplace_back(off[0],off[1]);
                uint64_t skip=0;
                std::vector<uint16_t>* dst=&loaded.backbone;
                if (i+2>=ts.size()) {
                    dst=(i+2==ts.size())?&loaded.pred:&loaded.succ;
                    n=static_cast<uint64_t>(vocab/world*r); skip=static_cast<uint64_t>(rank)*n*2;
                }
                const size_t at=dst->size(); dst->resize(at+static_cast<size_t>(n));
                sf.seekg(static_cast<std::streamoff>(8+hn+off[0]+skip));
                sf.read(reinterpret_cast<char*>(dst->data()+at),static_cast<std::streamsize>(n*2));
                if (!sf) return fail("truncated tensor payload: "+t.name);
            }
            std::sort(extents.begin(),extents.end()); uint64_t next=0;
            for (auto [lo,hi]:extents) {if (lo!=next) return fail("non-contiguous or overlapping tensor payload"); next=hi;}
            if (next!=file_bytes-8-hn) return fail("trailing tensor payload bytes");
            out=std::move(loaded); return true;
        } catch (const std::exception& e) { return fail(e.what()); }
    }
};

/* Root-inclusive packed row: [count, tokens[capacity], parents[capacity]], capacity at most
 * RAD_TREE_MAX_NODES. The root is the current anchor and has parent -1; every other parent
 * precedes its child. */
inline bool dloop_tree_from_packed(CandidateTree& tree,uint32_t anchor,
                                   const int32_t* row,int capacity,int vocab) {
    tree.reset();
    if (!row || capacity<1 || capacity>static_cast<int>(RAD_TREE_MAX_NODES) ||
        row[0]<1 || row[0]>capacity) return false;
    /* THE ANCHOR IS NODE 0, and it must be the token the sequence actually committed: the
     * proposal hangs off it, so a mismatch is a proposal for a different step. */
    if (row[1]<0 || row[1]>=vocab || static_cast<uint32_t>(row[1])!=anchor ||
        row[1+capacity]!=-1) return false;
    const int nodes=row[0];
    if (!tree.add_node(0,-1,static_cast<uint32_t>(row[1]),0)) return false;
    for (int i=1;i<nodes;++i) {
        const int tok=row[1+i], parent=row[1+capacity+i];
        if (tok<0 || tok>=vocab) return false;
        if (parent<0 || parent>=i) return false;
        const int depth=tree.nodes[parent].depth+1;
        if (!tree.add_node(i,parent,static_cast<uint32_t>(tok),depth)) return false;
    }
    return true;
}

/* DLoop emits each depth in descending path-score order; first child is the greedy child. */
inline CandidateTree dloop_greedy_chain(const CandidateTree& tree) {
    CandidateTree chain;
    if (tree.empty()) return chain;
    chain.add_node(0,-1,tree.nodes[0].token_id,0);
    int parent=0;
    for (;;) {
        int child=-1;
        for (size_t i=static_cast<size_t>(parent+1);i<tree.count;++i)
            if (tree.nodes[i].parent_id==parent) {child=static_cast<int>(i);break;}
        if (child<0) return chain;
        const int next=static_cast<int>(chain.count);
        chain.add_node(next,next-1,tree.nodes[child].token_id,next);
        parent=child;
    }
}

} // namespace radiance

#ifndef RAD_DLOOP_LOADER_ONLY
#include "rad_arch.h"
#include <memory>

namespace rad::arch {
/* THE FULL-CONTEXT DLOOP DRAFTER, five layers deep and reading the whole verified prefix.
 *
 * THIS BLOCK IS THE ENGINE'S HALF OF A CONTRACT THE KERNELS AND THE HOST ORACLE ALSO IMPLEMENT
 * (mine/dloop_full.h owns it; this file only issues against it). Three things are worth stating
 * here because they are what the old ring block got wrong:
 *
 *   NO RING. The old design kept a 32-row BF16 ring of teacher taps per slot and re-projected the
 *   whole window into the backbone's context on every base and every tree depth. That is gone,
 *   along with `dloop_ctxh`, `dloop_ring`, `dloop_ring_head` and `dloop_ring_length`. In their
 *   place are ONE F32 K/V cache [Q,L,C,2D] per rank and ONE I32 length [Q], written once per
 *   verified chunk by dloop_full_append and only read afterwards. A tree depth therefore costs no
 *   context repacking: the cache already holds every historical K and V.
 *
 *   THE APPEND IS DRIVEN BY ENGINE METADATA, NOT BY THE BATCH'S OWN ROW ORDER. `dloop_context_span`
 *   (I32[Q,2]) says, per slot, which part of the tap planes this step contributes a chunk of; the
 *   block copies the engine's staged `tree_path_rows` and `tree_path_reset` into the device
 *   operands. See step() for why the tap planes ARE the tree_path_rows buffer.
 *
 *   THE CACHE IS SIZED BY THE SIDECAR, AND REFUSED IF IT IS TOO SMALL. `context_capacity` is an
 *   allocation bound and never a window: a chunk that would cross it is an error the ops raise
 *   before touching a byte, and this block refuses to declare at all when the manifest's bound is
 *   under the context the engine was configured for. Truncating a prefix to fit is the one thing
 *   the full-context design exists to stop.
 */
struct DLoopBlock {
    bool enabled = false;
    /* Engine-facing metadata. `tree_path_rows`/`tree_path_reset` keep their old names and roles:
     * the engine stages the winning path's trunk-row indices into the first, and the reset flags
     * into the second, exactly as it did for the ring backend. The append op reads them. */
    rad_buf paths=0, reset=0, cache=0, length=0, span=0, anchor=0, anchor_embed=0;
    /* THE THREE TAP PLANES, in TRUNK-ROW coordinates: one row per token of the step, which is
     * `g.max_tok` rows and not `max_logit_rows`. They are declared and captured HERE rather than
     * borrowed from the KLD/FullDrafter recording for the reason engine.cpp's tap capture exists:
     * the old `kld_tap*` buffers are gathered through `out_ids`, so they hold one row per SAMPLED
     * position and are simply absent on a step that samples nothing. A DLoop append needs every
     * verified token of every chunk -- including the middle chunks of a long prefill, which write
     * no output row at all -- in the row space the winning-path indices are expressed in. */
    rad_buf tap0=0, tap1=0, tap2=0;
    rad_buf y=0,ids=0,logits=0,gids=0,glogits=0,pool=0,unary=0;
    rad_buf rows[5]={}, tok=0,parent=0,count=0,packed=0;
    rad_op append=0,base=0,merge=0,tree=0,gather[2]={},reduce[5]={};
    rad_op anchor_gather=0,embed_gather=0;
    std::shared_ptr<void> weights,pred,succ;
    std::shared_ptr<void> teacher_head_storage,teacher_embed_storage;
    int64_t nw=0,vs=0,lo=0,q=0,taps=0,world=0,budget=15;
    /* The sidecar's context_capacity, which is the C the ops are parameterised with and the extent
     * the cache was declared at. */
    int64_t cache_capacity=0;
    /* The tap plane column count, kept so step() can state the operand's second extent without
     * re-deriving it from a constant that lives in two places. */
    int64_t tap_width=2560;
    const uint16_t* teacher_head=nullptr;
    const uint16_t* teacher_embed=nullptr;

    static bool requested() {
        const char* p=std::getenv("RADIANCE_DLOOP_DRAFTER");
        return p && std::strcmp(p,"1")==0;
    }
    static int tree_budget() {
        const char* p=std::getenv("RADIANCE_DARTREE_BUDGET");
        if (!p) return 15;
        char* end=nullptr; const long n=std::strtol(p,&end,10);
        return end!=p && !*end && n>=1 && n<=15 ? static_cast<int>(n) : -1;
    }
    /* Load the version-2 sidecar and keep the cache bound it declared. The manifest's
     * `context_capacity` is what the ops are parameterised with and what the cache is sized for,
     * so it is carried here rather than re-derived from the engine's --max-ctx: the two are
     * checked against each other at declare, and a mismatch is a refusal rather than whichever
     * one happened to win. */
    int load(const char* dir,int rank,int ranks,std::string& err) {
        radiance::DLoopSidecar sc;
        if (!radiance::DLoopSidecar::load(dir?dir:"",radiance::DLoopSidecar::production_config(),
                                         rank,ranks,sc,err)) return RAD_E_FORMAT;
        RadStream stream=nullptr;
        int rc=rad_stream_create(&stream,0);
        if (rc<0) return rc;
        auto upload=[&](const std::vector<uint16_t>& src,std::shared_ptr<void>& dst) -> int {
            void* p=rad_dev_alloc(static_cast<int64_t>(src.size()*2),RAD_MEM_DEVICE);
            if (!p) return RAD_E_NOMEM;
            dst=std::shared_ptr<void>(p,[](void* x){rad_dev_free(x,RAD_MEM_DEVICE);});
            return rad_memcpy_async(p,src.data(),static_cast<int64_t>(src.size()*2),stream);
        };
        rc=upload(sc.backbone,weights);
        if (rc>=0) rc=upload(sc.pred,pred);
        if (rc>=0) rc=upload(sc.succ,succ);
        const int sync=rad_stream_sync(stream); rad_stream_destroy(stream);
        nw=static_cast<int64_t>(sc.backbone.size());
        cache_capacity=sc.context_capacity;
        return rc<0?rc:sync;
    }

    /* BF16-only and rank-local: never construct FullDrafter's unused coarse head or peer replica. */
    int load_teacher(const std::string& path,int rank,int ranks,int64_t vocab,int64_t width,
                     std::string& err) {
        if (ranks<=0 || rank<0 || rank>=ranks || vocab<=0 || vocab%ranks) {
            err="DLoop teacher: invalid vocabulary shard";return RAD_E_INVAL;
        }
        radiance::DLoopSidecar::TeacherLayout layout;
        if (!radiance::DLoopSidecar::teacher_layout(path,vocab,width,layout,err)) return RAD_E_FORMAT;
        std::ifstream f(path,std::ios::binary);
        if (!f) {err="DLoop teacher: cannot reopen tensor file";return RAD_E_FORMAT;}
        constexpr int64_t chunk_bytes=1024*1024;
        std::shared_ptr<void> staging(rad_dev_alloc(chunk_bytes,RAD_MEM_HOST_PINNED),
                                     [](void* p){rad_dev_free(p,RAD_MEM_HOST_PINNED);});
        if (!staging) {err="DLoop teacher: pinned staging allocation failed";return RAD_E_NOMEM;}
        RadStream stream=nullptr;
        int rc=rad_stream_create(&stream,0);
        if (rc<0) {err="DLoop teacher: upload stream creation failed";return rc;}
        auto upload=[&](uint64_t offset,int64_t bytes,std::shared_ptr<void>& dst)->int {
            dst=std::shared_ptr<void>(rad_dev_alloc(bytes,RAD_MEM_DEVICE),
                                     [](void* p){rad_dev_free(p,RAD_MEM_DEVICE);});
            if (!dst) {err="DLoop teacher: BF16 matrix allocation failed";return RAD_E_NOMEM;}
            f.seekg(static_cast<std::streamoff>(offset));
            for (int64_t at=0;at<bytes;at+=chunk_bytes) {
                const int64_t n=std::min(chunk_bytes,bytes-at);
                f.read(static_cast<char*>(staging.get()),static_cast<std::streamsize>(n));
                if (!f) {err="DLoop teacher: truncated BF16 matrix payload";return RAD_E_FORMAT;}
                const int copied=rad_memcpy_async(static_cast<char*>(dst.get())+at,staging.get(),n,stream);
                if (copied<0) {err="DLoop teacher: BF16 matrix upload failed";return copied;}
                const int synced=rad_stream_sync(stream);
                if (synced<0) {err="DLoop teacher: BF16 matrix upload synchronization failed";return synced;}
            }
            return RAD_OK;
        };
        std::shared_ptr<void> embedding,local_head;
        const int64_t head_bytes=static_cast<int64_t>(layout.tensor_bytes/ranks);
        rc=upload(layout.embedding,static_cast<int64_t>(layout.tensor_bytes),embedding);
        if (rc>=0) rc=upload(layout.head+static_cast<uint64_t>(rank)*head_bytes,head_bytes,local_head);
        rad_stream_destroy(stream);
        if (rc<0) return rc;
        teacher_embed_storage=std::move(embedding);teacher_head_storage=std::move(local_head);
        teacher_embed=static_cast<const uint16_t*>(teacher_embed_storage.get());
        teacher_head=static_cast<const uint16_t*>(teacher_head_storage.get());
        return RAD_OK;
    }

    /* DECLARE. The tap planes arrive from the architecture plugin, which is where the geometry that
     * decides their width lives; this block only states that it needs them, at TRUNK width, in the
     * coordinate its winning-path indices are expressed in.
     *
     * THE TEACHER HEAD IS NOT AN OPERAND HERE. It is a raw device pointer this block already owns
     * (loaded by load_teacher) and is passed at ISSUE as `RAD_P_T2(teacher_head, RAD_BF16, vs,
     * 2560)`, exactly as the ring block did. Declaring a buffer for it would be a second, competing
     * description of storage the drafter allocated itself -- and the only buffer in the plugin that
     * is head-shaped is the KLD tap, which is a different thing entirely.
     *
     * THE KLD RECORDING'S `kld_tap*` BUFFERS CANNOT SERVE AS THE TAPS, and the reason is worth
     * stating because they look like they should. Those are gathered through `out_ids`, so they
     * hold one row per SAMPLED position and are not written at all on a step that samples nothing
     * -- which is every middle chunk of a long prefill, and those chunks are exactly the ones a
     * full-context append must not skip. Filling them at full width would also change what the
     * `--kld-taps` consumer records. */
    int declare(RadBuilder* b,Names& nm,const Geom& g,rad_buf taps0,rad_buf taps1,rad_buf taps2) {
        q=g.max_seqs; taps=g.max_tok; world=g.world;
        vs=248320/world; lo=g.rank*vs; budget=tree_budget();
        tap0=taps0; tap1=taps1; tap2=taps2;
        if (!tap0 || !tap1 || !tap2) {
            rad_note(b,"DLoop: the three trunk-row tap planes are required operands");
            return RAD_E_INVAL;
        }
        /* THE CACHE BOUND, AND WHERE IT COMES FROM IN EACH PHASE.
         *
         * A SERVING DECLARE has `load()` behind it and takes the sidecar's `context_capacity`,
         * because that number IS the manifest's. A SHAPE PROBE has no sidecar (the plugin skips the
         * load deliberately -- it must not touch a device) so the field is 0, and the engine's own
         * context is the only bound available. Deriving it from `g.max_ctx` is not a fallback that
         * weakens the check: the block's job here is to declare buffers of the right SHAPE, and the
         * context the engine will ever verify against is exactly the number that sizes them.
         *
         * WHAT MUST NOT HAPPEN is what a bare `cache_capacity<=0` rejection did: the probe declared
         * nothing at all, so the whole plugin was recorded as failed and a shape question could no
         * longer be asked. */
        const bool probing = cache_capacity<=0;
        if (probing) {
            if (g.max_ctx<=0 || g.max_ctx>1048576) {
                rad_note(b,"DLoop: shape probe needs a usable context to size the cache "
                           "(max_ctx %lld)",(long long)g.max_ctx);
                return RAD_E_INVAL;
            }
            cache_capacity=g.max_ctx;
        }
        /* THE VALIDATOR'S OWN BOUND, applied before a cache of this size is declared. */
        if (cache_capacity>1048576) {
            rad_note(b,"DLoop: context_capacity %lld exceeds the %d the ops accept",
                     (long long)cache_capacity,1048576);
            return RAD_E_INVAL;
        }
        const char* variant=std::getenv("RADIANCE_DARTREE_VARIANT");
        if (!variant) variant="fixed";
        if (budget<0 || (std::strcmp(variant,"fixed") && std::strcmp(variant,"pruned"))) {
            rad_note(b,"DLoop requires RADIANCE_DARTREE_BUDGET=1..15 and VARIANT=fixed|pruned");
            return RAD_E_INVAL;
        }
        /* FAIL CLOSED ON THE CACHE BOUND, BEFORE ANY OPERAND IS DECLARED.
         *
         * The engine's --max-ctx is the longest context this deployment will ever verify, and the
         * sidecar's context_capacity is what the drafter's cache can hold. A serving run whose
         * context can exceed the cache has exactly one correct outcome -- refuse to start -- and
         * three wrong ones that all look alike at the first overflow: silently drop the prefix
         * tail, wrap onto the oldest positions, or fault mid-chunk. The ops themselves raise on a
         * range past C; this refuses earlier, where the message can name both numbers.
         *
         * A SHAPE PROBE SKIPS THIS ONE TEST AND ONLY THIS ONE. It has no sidecar, so its capacity
         * was derived from `g.max_ctx` above and the two are equal by construction -- the check
         * would pass trivially, and skipping it is what keeps the probe from having to invent a
         * manifest to declare a buffer. */
        if (!probing && g.max_ctx>0 && cache_capacity<g.max_ctx) {
            rad_note(b,"DLoop: the sidecar's context_capacity %lld is below this deployment's "
                       "context %lld -- the persistent cache cannot hold the verified prefix",
                     (long long)cache_capacity,(long long)g.max_ctx);
            return RAD_E_INVAL;
        }
        auto buf=[&](const char* n,uint32_t dt,std::initializer_list<int64_t> shape) {
            return decl_b(b,nm.f("%s",n),dt,shape,RAD_BUF_PERSIST);
        };
        paths=buf("tree_path_rows",RAD_I32,{q,16}); reset=buf("tree_path_reset",RAD_I32,{q});
        if (g.max_tok<=0 || q<=0 || world<=0) {
            rad_note(b,"DLoop: invalid serving geometry (max_tok %lld, max_seqs %lld, world %d)",
                     (long long)g.max_tok,(long long)q,world);
            return RAD_E_INVAL;
        }
        /* THE ENGINE-FACING METADATA, both declared PERSIST and both written by the host every
         * step that appends. `dloop_context_span` is [start,count] a slot: count >= 0 is a
         * contiguous chunk of the tap planes starting at `start`, and count == -1 says "use the
         * -1-terminated `path` instead", which is the decode case where the engine has a winning
         * node path and no contiguous run. */
        span=buf("dloop_context_span",RAD_I32,{q,2});
        length=buf("dloop_cache_length",RAD_I32,{q});
        /* F32 [Q,L,C,2D], D = KV heads * head_dim, K first then V, K already rotated at its
         * absolute position. F32 and not BF16: a re-quantised cache is a different model and the
         * whole point of the persistent prefix is that it is the same arithmetic as the oracle's.
         *
         * SIZED FROM THE SIDECAR'S BOUND AND NEVER FROM A WINDOW. At the production geometry this
         * is hundreds of MiB a rank, which is why the manifest states it and why it is allocated
         * once at declare rather than per step. */
        const int64_t kv_dim=2*4*64;                 /* 2 * num_kv_heads * head_dim */
        cache=buf("dloop_kv_cache",RAD_F32,{q,5,cache_capacity,kv_dim});
        if (!cache || !span || !length) return RAD_E_INVAL;
        /* The tap planes' second extent, taken from the geometry the plugin handed us rather than
         * restated: `step` states an operand extent from it and a mismatch would be a silently
         * wrong row pitch. The production geometry is pinned by the sidecar loader. */
        tap_width=2560;
        anchor=buf("dloop_anchor",RAD_I32,{q}); anchor_embed=buf("dloop_anchor_embed",RAD_BF16,{q,2560});
        y=buf("dloop_y",RAD_F32,{q,8,2560});
        ids=buf("dloop_ids",RAD_I32,{q,8,64}); logits=buf("dloop_logits",RAD_F32,{q,8,64});
        gids=buf("dloop_gids",RAD_I32,{world,q,8,64}); glogits=buf("dloop_glogits",RAD_F32,{world,q,8,64});
        pool=buf("dloop_pool",RAD_I32,{q,8,64}); unary=buf("dloop_unary",RAD_F32,{q,8,64});
        rows[0]=buf("dloop_pool_head",RAD_BF16,{q,8,64,2560});
        rows[1]=buf("dloop_pool_embed",RAD_BF16,{q,8,64,2560});
        rows[2]=buf("dloop_pool_pred",RAD_BF16,{q,8,64,256});
        rows[3]=buf("dloop_pool_succ",RAD_BF16,{q,8,64,256});
        rows[4]=buf("dloop_anchor_pred",RAD_BF16,{q,256});
        tok=buf("dloop_tree_tok",RAD_I32,{q,16}); parent=buf("dloop_tree_parent",RAD_I32,{q,16});
        count=buf("dloop_tree_count",RAD_I32,{q}); packed=buf("dloop_tree",RAD_I32,{q,33});
        const RadParam params[]={
            RAD_RANGE("n_seq",1,q),RAD_INT("max_seqs",q),RAD_INT("tap_rows",taps),RAD_INT("world_size",world),
            RAD_INT("hidden_size",768),RAD_INT("intermediate_size",2304),RAD_INT("num_layers",5),
            RAD_INT("num_q_heads",12),RAD_INT("num_kv_heads",4),RAD_INT("head_dim",64),
            RAD_INT("context_features_dim",7680),RAD_INT("target_width",2560),RAD_INT("proposal_length",8),
            /* ZERO, AND REQUIRED TO BE. The window is not configurable here: context_window 0 means
             * the op reads the whole cached prefix at length[slot], and dloop_full's validator
             * refuses any other value. */
            RAD_INT("context_window",0),
            RAD_INT("active_vocab",248077),RAD_INT("vocab_size",248320),
            RAD_INT("selector_rank",256),RAD_INT("conv_kernel_size",2),RAD_INT("conv_group_size",16),
            RAD_INT("vocab_lo",lo),RAD_INT("vocab_size_shard",vs),RAD_INT("pool_size",64),
            RAD_INT("budget",budget),RAD_INT("width",12),
            RAD_INT("context_capacity",static_cast<int64_t>(cache_capacity)),
            RAD_STR("variant",variant),
            RAD_F64("eps",1e-6),RAD_F64("rope_theta",1e7),RAD_F64("depth_bonus",-0.2),RAD_STR("dtype","bf16")};
        auto op=[&](const char* n){return rad_decl_op(b,n,params,30,nullptr,0);};
        /* THE POOL KEEPS ITS OWN PARAMETER SET, and not for tidiness: its validator and its device
         * rows state `context_window == 32` and there is no `context_capacity` in its 29-name
         * schema. Its arithmetic never touches the backbone or the context -- it ranks candidates
         * by the teacher codebooks -- so the plan's rule is that it keeps the geometry-neutral
         * parameters it always had. Declaring it with this block's 30-name full-context set would
         * be a schema mismatch at declare and an unresolvable row in the serving path. */
        const RadParam pool_params[]={
            RAD_RANGE("n_seq",1,q),RAD_INT("max_seqs",q),RAD_INT("tap_rows",g.max_logit_rows),RAD_INT("world_size",world),
            RAD_INT("hidden_size",768),RAD_INT("intermediate_size",2304),RAD_INT("num_layers",2),
            RAD_INT("num_q_heads",12),RAD_INT("num_kv_heads",4),RAD_INT("head_dim",64),
            RAD_INT("context_features_dim",7680),RAD_INT("target_width",2560),RAD_INT("proposal_length",8),
            RAD_INT("context_window",32),RAD_INT("active_vocab",248077),RAD_INT("vocab_size",248320),
            RAD_INT("selector_rank",256),RAD_INT("conv_kernel_size",2),RAD_INT("conv_group_size",16),
            RAD_INT("vocab_lo",lo),RAD_INT("vocab_size_shard",vs),RAD_INT("pool_size",64),
            RAD_INT("budget",budget),RAD_INT("width",12),RAD_STR("variant",variant),
            RAD_F64("eps",1e-6),RAD_F64("rope_theta",1e7),RAD_F64("depth_bonus",-0.2),RAD_STR("dtype","bf16")};
        auto pool_op=[&](const char* n){return rad_decl_op(b,n,pool_params,29,nullptr,0);};
        /* APPEND, in the op's operand order: tap0, tap1, tap2, path, span, slots, reset, weights,
         * cache, length. `slots` and `weights` are raw device pointers with no declared buffer
         * behind them, so they carry no liveness; the `rw` list below is the buffer handles
         * alone.
         *
         * `path` IS `tree_path_rows`, and its values are TRUNK-ROW indices: the engine stages the
         * winning node path as row numbers into the tap planes, which is the same coordinate the
         * tap capture writes them in. The op reads `span` first and only consults `path` when the
         * span says count == -1.
         *
         * `cache` and `length` are INOUT: the op advances the verified prefix and never rewrites
         * a historical row. */
        append=rw(b,op("dloop_full_append"),{tap0,tap1,tap2,paths,span,reset,cache,length},
                                                                          {cache,length});
        anchor_gather=rw(b,RAD_OP(b,"gather_rows",RAD_PARAMS(RAD_RANGE("M",1,q),RAD_INT("n",4),RAD_STR("dtype","u8")),RAD_NOWEIGHTS),{},{anchor});
        embed_gather=rw(b,RAD_OP(b,"gather_rows",RAD_PARAMS(RAD_RANGE("M",1,q),RAD_INT("n",2560),RAD_STR("dtype","bf16")),RAD_NOWEIGHTS),{anchor},{anchor_embed});
        /* BASE READS THE CACHE -- the cache operand IS the context, and there is no ctxh buffer
         * for it to write. `head_shard` and `weights` are raw device pointers the block owns, so
         * they are passed at issue and carry no declared liveness. */
        base=rw(b,op("dloop_full_base"),{cache,length,anchor_embed},{y,ids,logits});
        for (int i=0;i<2 && world>1;++i)
            gather[i]=rw(b,RAD_OP(b,"all_gather",RAD_PARAMS(RAD_INT("world_size",world),
                RAD_RANGE("numel",512,q*512),RAD_INT("row",0),RAD_STR("dtype","f32")),RAD_NOWEIGHTS),
                {i?logits:ids},{i?glogits:gids});
        merge=rw(b,pool_op("dloop_pool"),{world>1?gids:ids,world>1?glogits:logits,anchor},
                 {pool,unary,rows[0],rows[1],rows[2],rows[3],rows[4]});
        for (int i=0;i<5 && world>1;++i) {
            const int64_t width=i<2?8*64*2560:(i<4?8*64*256:256);
            reduce[i]=rw(b,RAD_OP(b,"all_reduce",RAD_PARAMS(RAD_INT("world_size",world),
                RAD_RANGE("numel",width,q*width),RAD_STR("dtype","bf16"),RAD_INT("exact",1)),RAD_NOWEIGHTS),
                {rows[i]},{rows[i]});
        }
        /* THE TREE READS THE CACHE, NOT A PRE-PROJECTED CONTEXT. `ctxh` is gone: the op takes
         * cache/length and attends over length[slot] cached positions plus the whole block, so a
         * tree depth costs no context repacking. */
        tree=rw(b,op("dloop_full_dartree"),
                {cache,length,anchor_embed,y,pool,unary,rows[0],rows[1],rows[2],rows[3],rows[4],anchor},
                {tok,parent,count,packed});
        if (!append || !anchor_gather || !embed_gather || !base || !merge || !tree) return RAD_E_INVAL;
        if (world>1) {
            for (int i=0;i<2;++i) if (!gather[i]) return RAD_E_INVAL;
            for (int i=0;i<5;++i) if (!reduce[i]) return RAD_E_INVAL;
        }
        RadDrafterDecl d{}; d.name="dloop"; d.kind=RAD_DRAFT_BLOCK; d.depth=static_cast<int>(budget);
        d.block=budget+1; d.proposal=tok; d.proposal_pitch=16; d.mask_token=151643;
        RAD_ARCH_TRY(rad_declare_drafter(b,&d));
        RAD_ARCH_TRY(rad_declare_drafter_tree(b,packed,16));
        enabled=true; return RAD_OK;
    }

    /* APPEND: issued on the dedicated tree-commit release, which is the one point in the step
     * where the HOST has already read the sampler's answer back and can therefore say which rows
     * of this step's taps are VERIFIED.
     *
     * That is the whole reason this is not issued with the trunk's forward pass. A tap plane is a
     * by-product of a forward pass, and the rows are only known to be real once acceptance has
     * run: before that, the verify step's rows past the winning node are candidates the target
     * rejected, and appending them would write positions into the persistent cache that never
     * happened. `dloop_context_span` and `tree_path_rows` are staged by collect_step with exactly
     * the accepted set, so this pass appends the verified prefix and nothing else.
     *
     * `batch` here is the engine's commit batch, and the append reads three things from it:
     * `seq_ids` (the scheduler's StepEntry::card) for slots, and `n_tok` for the tap-plane row
     * count. The PATH and the RESET flags do NOT come from this batch: they come from the declared
     * `tree_path_rows` and `tree_path_reset` buffers, which the step loop uploaded. That is a
     * coordinate distinction, not a style one -- see the block comment on the issue below.
     *
     * PREFILL INCLUDED, WHICH IS THE OTHER HALF OF THE FIX. `span` carries count >= 0 for a
     * contiguous chunk -- every prefill chunk, including the middle ones that sample nothing --
     * and count == -1 for the decode case, where the verified set is a node path and not a run.
     * The reset flags come from the engine and are 1 only at a true sequence start. */
    void step(RadCtx* c,const RadBatch* batch) const {
        const int64_t s=batch->n_seq;
        if (s<=0) return;
        if (batch->draft_pass==-2) {
            /* The commit release. Everything the append needs was staged by the host before this
             * release and is on the device by now: `span` and `reset` were uploaded by the step
             * loop beside the path, and the tap planes hold this step's recorded rows.
             *
             * PATH COMES FROM THE DECLARED BUFFER, NOT FROM `batch->draft_out_ids`.
             *
             * This is a coordinate correction, not a tidy-up. `draft_out_ids` on this release
             * points at the FIRST `S*P` ints of the engine's staging region, P=n_draft_out/S,
             * whose values are relative node ids -- the same array the QSA and GDN commits read, and
             * they must keep reading it, because a node id is what their parent-chain walks index.
             * The TRUNK ROWS this op needs are staged further along that same region and are
             * uploaded into the declared `tree_path_rows` buffer (core/engine.cpp, the
             * upload of `absolute`). So the append reads `paths`, which is that buffer,
             * and the two consumers disagree about nothing: one reads node ids where they live, the
             * other reads rows where they live.
             *
             * Reading `draft_out_ids` here would append node 0,1,2... as though they were tap rows
             * -- correct only on a single-sequence step whose winner happens to start at row 0,
             * which is precisely the case a test that never batches cannot tell apart.
             *
             * `taps` is the step's token count -- the rows a tap plane actually carries, which is
             * `n_tok` and not the declared `max_tok`. A span that names a row past it is refused
             * by the op, which is what keeps a partial append from reading a row no pass wrote.
             *
             * `reset` IS READ FROM THE DECLARED BUFFER and not from `num_accepted`: on this
             * release `num_accepted` carries the accepted path LENGTHS, which the QSA commit reads
             * through the same pointer (`rad_qsa.h`, `lengths = batch->num_accepted`). The reset
             * flags are the engine's own array and arrive through `tree_path_reset`. */
            RAD_ISSUE_N(c,append,s,praw2(rad_buf_ptr(c,tap0),RAD_BF16,batch->n_tok,tap_width),
                        praw2(rad_buf_ptr(c,tap1),RAD_BF16,batch->n_tok,tap_width),
                        praw2(rad_buf_ptr(c,tap2),RAD_BF16,batch->n_tok,tap_width),
                        brows(paths,s),
                        praw2(rad_buf_ptr(c,span),RAD_I32,s,2),
                        praw(batch->seq_ids,RAD_I32,s),
                        brows(reset,s),
                        praw(weights.get(),RAD_BF16,nw),
                        RAD_B(cache),RAD_B(length));
            return;
        }
        if (batch->draft_pass<0) return;   /* the block context pass: it does not append */
        const int64_t t=batch->n_tok;
        RAD_ISSUE_N(c,anchor_gather,s,praw2(batch->token_ids,RAD_U8,t,4),
                    praw(batch->cu_seqlens,RAD_I32,s),praw2(rad_buf_ptr(c,anchor),RAD_U8,s,4));
        RAD_ISSUE_N(c,embed_gather,s,praw2(teacher_embed,RAD_BF16,248320,2560),
                    brows(anchor,s),brows(anchor_embed,s));
        RAD_ISSUE_N(c,base,s,RAD_B(cache),RAD_B(length),praw(batch->seq_ids,RAD_I32,s),
                    brows(anchor_embed,s),praw(weights.get(),RAD_BF16,nw),
                    praw2(teacher_head,RAD_BF16,vs,2560),brows(y,s),brows(ids,s),brows(logits,s));
        if (world>1) {
            /* all_gather is a byte copy. F32 transport preserves I32 token-ID bits exactly. */
            RAD_ISSUE_N(c,gather[0],s*512,praw(rad_buf_ptr(c,ids),RAD_F32,s*512),
                        praw(rad_buf_ptr(c,gids),RAD_F32,world*s*512));
            RAD_ISSUE_N(c,gather[1],s*512,praw(rad_buf_ptr(c,logits),RAD_F32,s*512),
                        praw(rad_buf_ptr(c,glogits),RAD_F32,world*s*512));
        }
        RadOperand gi=praw2(rad_buf_ptr(c,world>1?gids:ids),RAD_I32,world,s*512);
        RadOperand gl=praw2(rad_buf_ptr(c,world>1?glogits:logits),RAD_F32,world,s*512);
        RAD_ISSUE_N(c,merge,s,gi,gl,brows(anchor,s),praw2(teacher_head,RAD_BF16,vs,2560),
                    praw2(teacher_embed+lo*2560,RAD_BF16,vs,2560),
                    praw2(pred.get(),RAD_BF16,vs,256),praw2(succ.get(),RAD_BF16,vs,256),
                    brows(pool,s),brows(unary,s),brows(rows[0],s),brows(rows[1],s),
                    brows(rows[2],s),brows(rows[3],s),brows(rows[4],s));
        for (int i=0;i<5 && world>1;++i) {
            const int64_t width=i<2?8*64*2560:(i<4?8*64*256:256);
            RAD_ISSUE_N(c,reduce[i],s*width,praw(rad_buf_ptr(c,rows[i]),RAD_BF16,s*width),RAD_NONE);
        }
        RAD_ISSUE_N(c,tree,s,RAD_B(cache),RAD_B(length),praw(batch->seq_ids,RAD_I32,s),
                    brows(anchor_embed,s),praw(weights.get(),RAD_BF16,nw),brows(y,s),brows(pool,s),
                    brows(unary,s),brows(rows[0],s),brows(rows[1],s),brows(rows[2],s),brows(rows[3],s),
                    brows(rows[4],s),brows(tok,s),brows(parent,s),brows(count,s),brows(anchor,s),brows(packed,s));
    }
};
} // namespace rad::arch
#endif
