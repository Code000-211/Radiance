#define RAD_DLOOP_LOADER_ONLY
#include "../../arch/common/rad_block_dloop.h"
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace fs=std::filesystem;
using Sidecar=radiance::DLoopSidecar;
static int failures=0;
static void check(bool ok,const char* what) {
    if (!ok) { std::cerr<<what<<'\n'; ++failures; }
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    const fs::path fixture=argv[1];
    Sidecar::Json manifest;
    {std::ifstream f(fixture/"dloop_sidecar.json"); f>>manifest;}
    const auto config=manifest.at("config");
    std::string err; Sidecar sc,rank0,rank1;
    check(Sidecar::load(fixture.string(),config,0,1,sc,err),err.c_str());
    check(Sidecar::load(fixture.string(),config,0,2,rank0,err),err.c_str());
    check(Sidecar::load(fixture.string(),config,1,2,rank1,err),err.c_str());
    check(Sidecar::validate_teacher((fixture/"teacher-matrices.safetensors").string(),512,32,err),
          "rejected good teacher matrices");
    check(!Sidecar::validate_teacher((fixture/"teacher-matrices.safetensors").string(),512,64,err),
          "accepted wrong teacher width");
    check(rank0.backbone==sc.backbone && rank1.backbone==sc.backbone,"backbone differs across ranks");
    std::vector<uint16_t> pred=rank0.pred; pred.insert(pred.end(),rank1.pred.begin(),rank1.pred.end());
    check(pred==sc.pred,"pred codebook shard order differs");
    std::vector<uint16_t> succ=rank0.succ; succ.insert(succ.end(),rank1.succ.begin(),rank1.succ.end());
    check(succ==sc.succ,"succ codebook shard order differs");
    const auto dir=fs::current_path()/("radiance_dloop_loader_"+std::to_string(::getpid()));
    fs::create_directory(dir);
    fs::copy_file(fixture/"dloop_sidecar.safetensors",dir/"dloop_sidecar.safetensors");
    auto reject=[&](Sidecar::Json bad,const char* label) {
        {std::ofstream f(dir/"dloop_sidecar.json"); f<<bad;}
        check(!Sidecar::load(dir.string(),config,0,1,sc,err) && err.starts_with("DLoop sidecar:"),label);
    };
    auto bad=manifest; bad["version"]=1; reject(bad,"accepted obsolete ring sidecar");
    bad=manifest; bad["version"]=2.0; reject(bad,"accepted noninteger version");
    bad=manifest; bad["architecture"]="other"; reject(bad,"accepted invalid architecture");
    bad=manifest; bad["config"]["hidden_size"]=65; reject(bad,"accepted wrong config");
    bad=manifest; bad["config"]["hidden_size"]=64.0; reject(bad,"accepted noninteger config");
    bad=manifest; bad["config"]["unknown"]=1; reject(bad,"accepted unknown config field");
    bad=manifest; bad["config"]["num_layers"]=2; reject(bad,"accepted obsolete two-layer model");
    bad=manifest; bad["config"]["context_window"]=32; reject(bad,"accepted truncated-context model");
    for (const auto& capacity:{Sidecar::Json(0),Sidecar::Json(-1),Sidecar::Json(40.0),Sidecar::Json(true),
                              Sidecar::Json(1048577),Sidecar::Json(nullptr)}) {
        bad=manifest;bad["context_capacity"]=capacity;
        reject(bad,"accepted invalid context capacity");
    }
    bad=manifest;bad.erase("context_capacity");reject(bad,"accepted missing context capacity");
    auto promoted=manifest;promoted["source_promoted"]=true;
    {std::ofstream f(dir/"dloop_sidecar.json");f<<promoted;}
    check(Sidecar::load(dir.string(),config,0,1,sc,err),"rejected promoted version-2 sidecar");
    bad=manifest; bad["tensors"][0]["name"]="dloop.missing"; reject(bad,"accepted wrong tensor name");
    bad=manifest; bad["tensors"][0]["dtype"]="F32"; reject(bad,"accepted wrong tensor dtype");
    bad=manifest; bad["tensors"][0]["shape"]={64,49}; reject(bad,"accepted wrong tensor shape");
    bad=manifest; bad["tensors"].erase(bad["tensors"].begin()); reject(bad,"accepted missing tensor");
    bad=manifest; bad["tree_defaults"]["budget"]=16; reject(bad,"accepted oversized tree default");
    for (bool cache:{false,true}) {
        auto cached=manifest;cached["config"]["use_kv_cache"]=cache;
        {std::ofstream f(dir/"dloop_sidecar.json");f<<cached;}
        check(Sidecar::load(dir.string(),config,0,1,sc,err),"rejected boolean use_kv_cache");
    }
    for (const auto& cache:{Sidecar::Json(0),Sidecar::Json(1),Sidecar::Json("true"),Sidecar::Json(nullptr)}) {
        bad=manifest;bad["config"]["use_kv_cache"]=cache;
        reject(bad,"accepted nonboolean use_kv_cache");
    }
    {std::ofstream f(dir/"dloop_sidecar.json"); f<<manifest;}
    fs::resize_file(dir/"dloop_sidecar.safetensors",9);
    check(!Sidecar::load(dir.string(),config,0,1,sc,err),"accepted truncated tensor file");
    fs::remove_all(dir);

    int32_t row[33]; std::fill(std::begin(row),std::end(row),-1);
    row[0]=6; const int32_t toks[]={10,11,12,13,14,15}, parents[]={-1,0,0,2,1,3};
    std::copy(std::begin(toks),std::end(toks),row+1);
    std::copy(std::begin(parents),std::end(parents),row+17);
    radiance::CandidateTree tree;
    check(radiance::dloop_tree_from_packed(tree,10,row,16,509),"rejected valid branching tree");
    check(tree.count==6 && tree.nodes[5].ancestor_bitmask==45,"wrong ancestry from parents");
    const auto chain=radiance::dloop_greedy_chain(tree);
    check(chain.count==3 && chain.nodes[1].token_id==11 && chain.nodes[2].token_id==14,"greedy chain did not choose first/highest-scoring child");
    row[17+3]=3;
    check(!radiance::dloop_tree_from_packed(tree,10,row,16,509),"accepted self parent");
    row[17+3]=2; row[0]=17;
    check(!radiance::dloop_tree_from_packed(tree,10,row,16,509),"accepted oversized tree");
    row[0]=6; row[1]=11;
    check(!radiance::dloop_tree_from_packed(tree,10,row,16,509),"accepted wrong anchor");

    // Same GPU wire layout at65: the host derives depth/ancestry from parents, never proposals.
    int32_t wide[131];
    std::fill(std::begin(wide),std::end(wide),-1);
    wide[0]=65;
    for (int i=0;i<65;++i) {wide[1+i]=10+i;wide[66+i]=i-1;}
    check(radiance::dloop_tree_from_packed(tree,10,wide,65,509),"rejected65-node packed chain");
    check(tree.count==65 && tree.nodes[64].ancestor_bitmask.count()==65,"lost wide ancestry");
    for (int node:{15,31,63,64}) {
        check(tree.nodes[node].parent_id==node-1 && tree.nodes[node].depth==node,"wrong wide parent/depth");
        check(tree.nodes[node].ancestor_bitmask.test(node),"lost high self bit");
    }
    check(!tree.add_node(65,64,75,65),"accepted66th host node");
    wide[0]=66;
    check(!radiance::dloop_tree_from_packed(tree,10,wide,65,509),"accepted count66 at capacity65");
    check(!radiance::dloop_tree_from_packed(tree,10,wide,66,509),"accepted declared capacity66");
    wide[0]=65;wide[66+64]=64;
    check(!radiance::dloop_tree_from_packed(tree,10,wide,65,509),"accepted wide self parent");
    wide[66+64]=15;
    check(radiance::dloop_tree_from_packed(tree,10,wide,65,509),"rejected wide fork");
    check(tree.nodes[64].depth==16 && tree.nodes[64].ancestor_bitmask.count()==17 &&
          tree.nodes[64].ancestor_bitmask.test(15) && !tree.nodes[64].ancestor_bitmask.test(63),
          "wide fork inherited non-ancestor bits");
    std::cout<<"DLoop loader/tree checks: "<<(failures?"FAIL":"PASS")<<" ("<<failures<<" failures)\n";
    return failures?1:0;
}
