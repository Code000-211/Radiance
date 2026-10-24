#include "core/device/device.h"
#include "core/plugin/registry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
void require(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
void check(int rc, const std::string& s) { require(rc == RAD_OK,s+": "+rad_strerror(rc)); }
constexpr int sequences = 4, guard = 16, sentinel = -777777;
constexpr int cand_ld = 11, hist_ld = 10, hist_rd = sequences*hist_ld+7;

struct Buffer {
    std::vector<int32_t> host, got;
    void* device = nullptr;
    explicit Buffer(int n) : host(n+2*guard,sentinel), got(host) {
        device = rad_dev_alloc(static_cast<int64_t>(host.size()*4),RAD_MEM_DEVICE);
        require(device != nullptr,"allocation");
    }
    ~Buffer() { if (device) rad_dev_free(device,RAD_MEM_DEVICE); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    int32_t* h() { return host.data()+guard; }
    void* d() { return static_cast<int32_t*>(device)+guard; }
    void upload(RadStream stream) {
        check(rad_memcpy_async(device,host.data(),host.size()*4,stream),"upload");
    }
    void download(RadStream stream) {
        check(rad_memcpy_async(got.data(),device,got.size()*4,stream),"download");
        check(rad_stream_sync(stream),"sync");
    }
    void compare(RadStream stream, const char* name) {
        download(stream);
        require(got == host,std::string(name)+" differs, including padding/guards");
    }
};
RadTensor matrix(void* p, int rows, int cols, int ld) {
    RadTensor t{}; t.data = p; t.dtype = RAD_I32; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = cols; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
RadTensor vector(void* p, int n, int stride) {
    RadTensor t{}; t.data = p; t.dtype = RAD_I32; t.rank = 1;
    t.shape[0] = n; t.stride[0] = stride; return t;
}
/* nr = max_rounds: 3 is the frozen legacy geometry (vocab 100, capacity<=16); 8 is the
 * fixed MTP8/64 geometry (capacity<=65). The wider vocabulary keeps 8 rounds of distinct ids live. */
int vocab_of(int nr) { return nr == 3 ? 100 : 1000; }
rad::Geometry geometry(int cap, int round, int nr) {
    rad::Geometry g;
    g.set_i("M",sequences); g.set_i("pitch",8); g.set_i("capacity",cap);
    g.set_i("max_rounds",nr); g.set_i("n_vocab",vocab_of(nr)); g.set_i("round",round);
    return g;
}
rad::Resolved resolve(rad::Registry& reg, int cap, int round, int domain, int nr,
                      bool merge = false) {
    rad::SelectionTrace why;
    auto g = geometry(cap,round,nr);
    if (merge) g.set_i("world_size",2);
    auto r = reg.resolve(merge ? "mtp_merge_tree_accum" : "mtp_tree_accum",g,domain,&why);
    require(r.row && r.row->info,"missing mtp_tree_accum row: "+why.reason);
    const char* name = merge
        ? (domain == RAD_DOMAIN_HOST ? "ref_mtp_merge_tree_accum" : "mine_mtp_merge_tree_accum")
        : (domain == RAD_DOMAIN_HOST ? "ref_mtp_tree_accum" : "mine_mtp_tree_accum");
    require(std::strcmp(r.row->info->name,name) == 0,"wrong row identity");
    require(r.row->plugin == (domain == RAD_DOMAIN_HOST ? "mine_ref" : "mine"),"wrong plugin");
    check(reg.init_instance(r,0,1),"init");
    return r;
}
template<size_t N>
int launch(rad::Resolved& r, std::array<RadTensor,N>& t, RadStream stream) {
    RadArgs a{}; a.t = t.data(); a.n_t = static_cast<int>(N); a.p = r.geom.params();
    a.n_p = r.geom.n_params(); a.world_size = N == 7 ? 2 : 1; a.instance = r.instance;
    return r.row->info->launch(&a,stream);
}

/* Independent oracle: literal old Engine::make_serial_tree topology using complete
 * round snapshots, never the accumulator's persistent history or host reference code. */
using Rounds = std::array<std::array<int32_t,8>,8>;
std::vector<int32_t> old_tree(const Rounds& rounds, int nr, int root, int cap, int vocab) {
    std::vector<int32_t> out(1+2*cap,-1);
    out[0] = 0;
    if (root < 0 || root >= vocab) return out;
    for (int r=0;r<nr;++r) for (int j=0;j<8 && rounds[r][j]>=0;++j)
        if (rounds[r][j] >= vocab) return out;
    std::vector<int32_t> tok{root}, par{-1};
    int spine = 0;
    for (int r=0;r<nr && static_cast<int>(tok.size())<cap;++r) {
        if (rounds[r][0] < 0) break;
        par.push_back(static_cast<int>(tok.size())-1);
        tok.push_back(rounds[r][0]); ++spine;
    }
    for (int r=0;r<spine;++r) for (int j=1;j<8 && static_cast<int>(tok.size())<cap;++j) {
        const int alt = rounds[r][j];
        if (alt < 0) break;
        bool dup = alt == rounds[r][0];
        for (size_t k=0;k<tok.size();++k) dup = dup || (par[k] == r && tok[k] == alt);
        if (!dup) { tok.push_back(alt); par.push_back(r); }
    }
    out[0] = static_cast<int32_t>(tok.size());
    std::copy(tok.begin(),tok.end(),out.begin()+1);
    std::copy(par.begin(),par.end(),out.begin()+1+cap);
    return out;
}

/* variant 3 (nr=8 only): no duplicate candidates, so a full 8-round step fills
 * 8 spine + 56 alternates + root = 65 nodes exactly. */
size_t run(rad::Registry& reg, RadStream stream, int cap, int variant, int nr) {
    const int out_ld = 1+2*cap+5, vocab = vocab_of(nr);
    Buffer cand(sequences*cand_ld), tokens(16), ids(sequences*2);
    Buffer hist(nr*hist_rd), packed(sequences*out_ld);
    cand.upload(stream); tokens.upload(stream); ids.upload(stream);
    hist.upload(stream); packed.upload(stream);
    check(rad_stream_sync(stream),"initial upload");
    std::array<Rounds,sequences> snapshots{};
    size_t launches = 0;
    bool full_seen = false;
    // Long -> short -> intermediate steps reuse the SAME allocations, including row shrink/grow.
    const int steps[] = {nr,1,nr == 3 ? 2 : 5,nr};
    const int rows[] = {4,2,4,1};
    for (int step=0;step<4;++step) {
        const int M = rows[step];
        for (int s=0;s<sequences;++s) {
            ids.h()[2*s] = 3-s;
            tokens.h()[2*(3-s)] = 5+step*4+s;
            if (variant == 2 && s == 1) ids.h()[2*s] = 20; // OOB gather index
            if (variant == 2 && s == 2) tokens.h()[2*(3-s)] = vocab; // invalid anchor
            if (variant == 2 && s == 3) ids.h()[2*s] = -1;
        }
        ids.upload(stream); tokens.upload(stream);
        for (int round=1;round<=steps[step];++round) {
            for (int s=0;s<M;++s) {
                auto& cr = snapshots[s][round-1];
                for (int k=0;k<8;++k) cr[k] = 20+round*10+s+k;
                if (variant != 3) { cr[2] = cr[0]; cr[4] = cr[1]; } // spine duplicate and repeated sibling
                if (variant == 1 && s == 0 && round == 2) cr[0] = -1;
                if (variant == 1 && s == 1) { cr[3] = -1; cr[4] = vocab+1; } // ignored tail
                if (variant == 1 && s == 2 && round == 1) cr[5] = vocab; // invalid live token
                if (variant == 1 && s == 3) cr[0] = vocab; // invalid spine
                std::copy(cr.begin(),cr.end(),cand.h()+s*cand_ld);
            }
            cand.upload(stream);
            auto h = resolve(reg,cap,round,RAD_DOMAIN_HOST,nr);
            auto d = resolve(reg,cap,round,RAD_DOMAIN_DEVICE,nr);
            auto bind = [&](bool device) {
                auto ptr = [&](Buffer& b) -> void* { return device ? b.d() : b.h(); };
                std::array<RadTensor,5> t{};
                t[0] = matrix(ptr(cand),M,8,cand_ld);
                if (round == 1) {
                    t[1] = vector(ptr(tokens),8,2); t[2] = vector(ptr(ids),M,2);
                }
                t[3] = matrix(ptr(hist),nr,sequences,hist_rd);
                t[3].rank = 3; t[3].shape[2] = 8;
                t[3].stride[1] = hist_ld; t[3].stride[2] = 1;
                t[4] = matrix(ptr(packed),M,1+2*cap,out_ld);
                return t;
            };
            auto ht = bind(false), dt = bind(true);
            // Metadata refusals must not launch or mutate any device bytes.
            if (round == 1 && step == 0) {
                auto bad = dt; bad[0].stride[1] = 2;
                require(launch(d,bad,stream) != RAD_OK,"accepted non-dense candidates");
                bad = dt; bad[3].shape[0] = 2;
                require(launch(d,bad,stream) != RAD_OK,"accepted truncated history");
                bad = dt; bad[4].data = bad[0].data;
                require(launch(d,bad,stream) != RAD_OK,"accepted output/input overlap");
                bad = dt; bad[2].shape[0] = M-1;
                require(launch(d,bad,stream) != RAD_OK,"accepted short gather indices");
                bad = dt; bad[1].stride[0] = INT64_MAX;
                require(launch(d,bad,stream) != RAD_OK,"accepted overflowing span");
                bad = dt; bad[0].dtype = RAD_F32;
                require(launch(d,bad,stream) != RAD_OK,"accepted wrong candidate dtype");
            }
            if (round == 1 && step == 0) {
                auto reject_geometry = [&](const char* key, int64_t value) {
                    auto g = geometry(cap,round,nr);
                    g.set_i(key,value);
                    RadArgs a{}; a.t = dt.data(); a.n_t = 5;
                    a.p = g.params(); a.n_p = g.n_params(); a.world_size = 1;
                    require(d.row->info->launch(&a,stream) != RAD_OK,
                            std::string("accepted malformed geometry: ")+key);
                };
                reject_geometry("capacity",1); reject_geometry("capacity",nr == 3 ? 17 : 66);
                reject_geometry("round",0); reject_geometry("round",nr+1);
                reject_geometry("max_rounds",4); reject_geometry("max_rounds",nr == 3 ? 8 : 3);
                reject_geometry("pitch",7);
                reject_geometry("n_vocab",0); reject_geometry("M",M-1);
            }
            check(launch(h,ht,nullptr),"host launch");
            check(launch(d,dt,stream),"device launch");
            for (int s=0;s<M;++s) {
                int root = -1;
                const int index = ids.h()[2*s];
                if (index >= 0 && index < 8) root = tokens.h()[2*index];
                const auto expected = old_tree(snapshots[s],round,root,cap,vocab);
                require(std::equal(expected.begin(),expected.end(),packed.h()+s*out_ld),
                        "old builder mismatch cap="+std::to_string(cap)+" round="+std::to_string(round));
                if (variant == 3 && cap == 65 && round == 8 && expected[0] == 65) {
                    full_seen = true;
                    // Negative control: one mutated parent must not satisfy the comparison.
                    auto mutated = expected;
                    mutated[1+cap+64] = 0;
                    require(!std::equal(mutated.begin(),mutated.end(),packed.h()+s*out_ld),
                            "mutated topology accepted");
                }
                for (int r=0;r<nr;++r) for (int k=0;k<8;++k)
                    require(hist.h()[r*hist_rd+s*hist_ld+k] ==
                            (r<round ? snapshots[s][r][k] : -1),
                            "history retained a previous-step candidate");
            }
            packed.compare(stream,"packed"); hist.compare(stream,"history");
            cand.compare(stream,"candidate input"); tokens.compare(stream,"token input");
            ids.compare(stream,"index input");
            ++launches;
        }
    }
    require(variant != 3 || cap != 65 || full_seen,"full 65-node tree never produced");
    return launches;
}

RadTensor history(void* data, int row_ld, int round_ld, int nr) {
    auto t = matrix(data,nr,sequences,round_ld);
    t.rank = 3; t.shape[2] = 8; t.stride[1] = row_ld; t.stride[2] = 1;
    return t;
}

/* The old chain is the pinned libr4d merge, followed by the unfused device
 * accumulator. Its downloaded candidates also drive the literal old_tree oracle. */
size_t run_merge(rad::Registry& reg, RadStream stream, int cap, int variant, bool padded, int nr) {
    const int vocab = vocab_of(nr);
    const int gl = padded ? 37 : 32, cl = padded ? 11 : 8;
    const int hl = padded ? 10 : 8, hr = sequences*hl+(padded ? 7 : 0);
    const int ol = 1+2*cap+(padded ? 5 : 0);
    Buffer gathered(sequences*gl), idx(sequences*cl), val(sequences*cl);
    Buffer tokens(16), ids(sequences*2), hist(nr*hr), packed(sequences*ol);
    Buffer old_gathered(sequences*32), old_idx(sequences*8), old_val(sequences*8);
    Buffer old_hist(nr*hr), old_packed(sequences*ol);
    for (Buffer* b : {&gathered,&idx,&val,&tokens,&ids,&hist,&packed,
                      &old_gathered,&old_idx,&old_val,&old_hist,&old_packed}) b->upload(stream);

    rad::Geometry mg; mg.set_i("M",sequences); mg.set_i("R",8);
    mg.set_i("world_size",2); mg.set_s("dtype","bf16");
    rad::SelectionTrace why;
    auto merge = reg.resolve("row_topk_merge",mg,RAD_DOMAIN_DEVICE,&why);
    require(merge.row && merge.row->info && merge.row->plugin == "libr4d" &&
            std::strcmp(merge.row->info->name,"rowtopk_merge") == 0,
            "missing actual pinned libr4d rowtopk_merge: "+why.reason);
    check(reg.init_instance(merge,0,1),"init libr4d merge");
    std::array<Rounds,sequences> snapshots{};
    const int steps[] = {nr,1,nr == 3 ? 2 : 5,nr}, rows[] = {4,2,4,1};
    size_t launches = 0;
    for (int step=0;step<4;++step) {
        const int M = rows[step];
        for (int s=0;s<sequences;++s) {
            ids.h()[2*s] = 3-s;
            tokens.h()[2*(3-s)] = 5+step*4+s;
            if (variant == 3 && s == 1) ids.h()[2*s] = 20;
            if (variant == 3 && s == 2) tokens.h()[2*(3-s)] = vocab;
        }
        tokens.upload(stream); ids.upload(stream);
        for (int round=1;round<=steps[step];++round) {
            for (int s=0;s<M;++s) {
                uint32_t value_bits[16]{};
                int32_t candidate[16]{};
                for (int k=0;k<16;++k) {
                    candidate[k] = 20+(s*17+round*7+k*5)%70;
                    const float f = static_cast<float>((k*11+round*3+s)%13-6);
                    std::memcpy(&value_bits[k],&f,4);
                }
                if (variant == 0) {
                    candidate[9] = candidate[1]; // same id, distinct value
                    candidate[10] = candidate[2]; value_bits[10] = value_bits[2]; // equal key
                } else if (variant == 1) {
                    const uint32_t special[] = {0x7fc00001u,0x7fc00002u,0x7f800000u,0u,
                        0x80000000u,0xff800000u,0xffc00001u,0xffc00002u,
                        0x7f800001u,0xff800001u,0x3f800000u,0xbf800000u,
                        1u,0x80000001u,0x7fffffffu,0xffffffffu};
                    for (int k=0;k<16;++k) {
                        value_bits[k] = special[(k+round-1)%16];
                        if (s == 1 && k>=8) candidate[k] = -1;
                        if (s == 2) value_bits[k] = 0xff800001u+static_cast<uint32_t>(k);
                        if (s == 3) value_bits[k] = k%2 ? 0x80000000u : 0u;
                    }
                } else if (variant == 2) {
                    for (int k=0;k<16;++k) {
                        candidate[k] = s == 0 || (s == 2 && k>=4) ? -7 : 31+k%2;
                        if (s == 1) candidate[k] = 31;
                        value_bits[k] = s == 2 && k%2 ? 0x80000000u : 0u;
                    }
                } else {
                    candidate[0] = s == 0 ? INT32_MAX : vocab;
                    value_bits[0] = 0x7f800000u;
                    candidate[1] = -2; value_bits[1] = 0x7fffffffu;
                }
                for (int k=0;k<16;++k) {
                    gathered.h()[s*gl+2*k] = candidate[k];
                    std::memcpy(gathered.h()+s*gl+2*k+1,&value_bits[k],4);
                }
                std::copy_n(gathered.h()+s*gl,32,old_gathered.h()+s*32);
            }
            gathered.upload(stream); old_gathered.upload(stream);
            auto h = resolve(reg,cap,round,RAD_DOMAIN_HOST,nr,true);
            auto d = resolve(reg,cap,round,RAD_DOMAIN_DEVICE,nr,true);
            auto old = resolve(reg,cap,round,RAD_DOMAIN_DEVICE,nr);
            auto bind = [&](bool device) {
                auto ptr = [&](Buffer& b) -> void* { return device ? b.d() : b.h(); };
                std::array<RadTensor,7> t{};
                t[0] = matrix(ptr(gathered),M,16,gl); t[0].rank = 3; t[0].dtype = RAD_F32;
                t[0].shape[2] = 2; t[0].stride[1] = 2; t[0].stride[2] = 1;
                t[1] = matrix(ptr(idx),M,8,cl);
                t[2] = matrix(ptr(val),M,8,cl); t[2].dtype = RAD_F32;
                if (round == 1) {
                    t[3] = vector(ptr(tokens),8,2); t[4] = vector(ptr(ids),M,2);
                }
                t[5] = history(ptr(hist),hl,hr,nr); t[6] = matrix(ptr(packed),M,1+2*cap,ol);
                return t;
            };
            auto ht = bind(false), dt = bind(true);
            if (round == 1 && step == 0) {
                auto reject = [&](std::array<RadTensor,7> bad, const char* reason) {
                    require(launch(d,bad,stream) != RAD_OK,std::string("accepted ")+reason);
                    auto host_bad = ht;
                    // Apply the same metadata mutation to the host row without dereferencing device bytes.
                    for (size_t i=0;i<bad.size();++i) {
                        void* hp = host_bad[i].data;
                        host_bad[i] = bad[i]; host_bad[i].data = hp;
                    }
                    require(launch(h,host_bad,nullptr) != RAD_OK,std::string("host accepted ")+reason);
                };
                auto bad = dt; bad[0].shape[1] = 15; reject(bad,"short gathered row");
                bad = dt; bad[0].stride[1] = 3; reject(bad,"non-dense pairs");
                bad = dt; bad[0].dtype = RAD_I32; reject(bad,"wrong gathered dtype");
                bad = dt; bad[1].shape[1] = 9; reject(bad,"wrong output width");
                bad = dt; bad[2].stride[1] = 2; reject(bad,"non-dense values");
                bad = dt; bad[2].dtype = RAD_I32; reject(bad,"wrong value dtype");
                bad = dt; bad[5].shape[0] = 2; reject(bad,"truncated history");
                bad = dt; bad[6].stride[0] = INT64_MAX; reject(bad,"overflowing packed span");
                // Every output must refuse aliases with every other operand.
                for (int w : {1,2,5,6}) for (int other=0;other<7;++other) {
                    if (w == other) continue;
                    bad = dt; bad[w].data = bad[other].data;
                    require(launch(d,bad,stream) != RAD_OK,"accepted output alias");
                    auto bh = ht; bh[w].data = bh[other].data;
                    require(launch(h,bh,nullptr) != RAD_OK,"host accepted output alias");
                }
                for (const auto& p : std::array<std::pair<const char*,int64_t>,8>{
                         {{"world_size",1},{"pitch",7},{"capacity",1},{"max_rounds",4},
                          {"n_vocab",0},{"round",nr+1},{"M",M-1},
                          {"capacity",nr == 3 ? 17 : 66}}}) {
                    auto g = geometry(cap,round,nr); g.set_i("world_size",2); g.set_i(p.first,p.second);
                    RadArgs a{}; a.t = dt.data(); a.n_t = 7; a.p = g.params(); a.n_p = g.n_params();
                    require(d.row->info->launch(&a,stream) != RAD_OK,"accepted malformed fusion geometry");
                }
            }
            std::array<RadTensor,3> mt{};
            mt[0] = matrix(old_gathered.d(),M,16,32); mt[0].rank = 3; mt[0].dtype = RAD_F32;
            mt[0].shape[2] = 2; mt[0].stride[1] = 2; mt[0].stride[2] = 1;
            mt[1] = matrix(old_idx.d(),M,8,8);
            mt[2] = matrix(old_val.d(),M,8,8); mt[2].dtype = RAD_F32;
            check(launch(merge,mt,stream),"pinned libr4d merge");
            std::array<RadTensor,5> ot = {mt[1],dt[3],dt[4],
                history(old_hist.d(),hl,hr,nr),matrix(old_packed.d(),M,1+2*cap,ol)};
            check(launch(old,ot,stream),"unfused device accumulator");
            check(launch(h,ht,nullptr),"host fusion");
            check(launch(d,dt,stream),"device fusion");
            old_idx.download(stream); old_val.download(stream);
            old_hist.download(stream); old_packed.download(stream);
            for (int s=0;s<M;++s) {
                auto& cr = snapshots[s][round-1];
                std::copy_n(old_idx.got.data()+guard+s*8,8,cr.begin());
                require(std::equal(cr.begin(),cr.end(),idx.h()+s*cl),"libr4d candidate ids differ");
                require(std::equal(old_val.got.begin()+guard+s*8,old_val.got.begin()+guard+s*8+8,
                                   val.h()+s*cl),"libr4d candidate value bits differ");
                const int index = ids.h()[2*s];
                const int root = index >= 0 && index < 8 ? tokens.h()[2*index] : -1;
                const auto expected = old_tree(snapshots[s],round,root,cap,vocab);
                require(std::equal(expected.begin(),expected.end(),packed.h()+s*ol),"old tree oracle differs");
            }
            require(old_hist.got == hist.host,"unfused device history differs");
            require(old_packed.got == packed.host,"unfused device packed tree differs");
            idx.compare(stream,"merged ids"); val.compare(stream,"merged value bits");
            hist.compare(stream,"merged history"); packed.compare(stream,"merged packed tree");
            gathered.compare(stream,"gathered input"); tokens.compare(stream,"merge tokens");
            ids.compare(stream,"merge indices");
            ++launches;
        }
    }
    return launches;
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 7 && std::string(argv[1]) == "--mine" && std::string(argv[3]) == "--ref" &&
                std::string(argv[5]) == "--libr4d",
                "usage: mtp_tree_accum_gpu_test --mine PATH --ref PATH --libr4d PATH");
        check(rad_dev_set(0),"set device");
        RadDeviceProps props{}; check(rad_dev_props(0,&props),"device properties");
        require(!props.is_host_backend,"real GPU required; host execution is not a pass");
        rad::Registry reg; reg.set_code_device(0);
        check(reg.load_plugin(argv[2],0),"load mine");
        check(reg.load_plugin(argv[4],1),"load mine_ref");
        check(reg.load_plugin(argv[6],2),"load pinned libr4d");
        check(reg.check_complete(),"complete schemas");
        RadStream stream = nullptr; check(rad_stream_create(&stream,0),"create stream");
        size_t launches = 0;
        size_t merge_launches = 0;
        try {
            for (int cap : {2,8,16}) for (int variant=0;variant<3;++variant)
                launches += run(reg,stream,cap,variant,3);
            for (int cap : {2,16,65}) for (int variant=0;variant<4;++variant)
                launches += run(reg,stream,cap,variant,8);
            for (int cap : {2,8,16}) for (int variant=0;variant<4;++variant)
                for (bool padded : {false,true})
                    merge_launches += run_merge(reg,stream,cap,variant,padded,3);
            for (int cap : {2,16,65}) for (int variant=0;variant<4;++variant)
                for (bool padded : {false,true})
                    merge_launches += run_merge(reg,stream,cap,variant,padded,8);
        } catch (...) { rad_stream_destroy(stream); throw; }
        rad_stream_destroy(stream);
        require(launches > 0 && merge_launches > 0,"zero launches");
        std::printf("ALL_PASS accumulator_configurations=21 accumulator_launches=%zu "
                    "merge_configurations=48 merge_launches=%zu max_rounds=3,8 capacity<=65\n",
                    launches,merge_launches);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
