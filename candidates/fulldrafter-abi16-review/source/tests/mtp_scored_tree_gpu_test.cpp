#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "arch/common/rad_mtp_tree_config.h"
#define RAD_DLOOP_LOADER_ONLY
#include "arch/common/rad_block_dloop.h"
#undef RAD_DLOOP_LOADER_ONLY

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
constexpr int seqs = 3, guard = 16, vocab = 64;
constexpr uint32_t sentinel = 0xd5f12345u;
constexpr int gl = 40, sl = 12, cl = 12, hl = 12, hr = seqs*hl+4;
using Eight = std::array<int32_t,8>;
using Logs = std::array<double,8>;
using Costs = std::array<double,8>;
const Costs flat_costs = {1,1,1,1,1,1,1,1};
const Costs steep_costs = {1,1.01,100,100,100,100,100,100};

void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
void check(int rc, const std::string& why) {
    require(rc == RAD_OK,why+": "+rad_strerror(rc));
}
uint32_t bits(float x) { uint32_t b; std::memcpy(&b,&x,4); return b; }
float value(uint32_t b) { float x; std::memcpy(&x,&b,4); return x; }
void close(double actual, double expected, const std::string& why, double atol = 0.00015) {
    if (std::isinf(expected)) {
        require(actual == expected,why+" infinity mismatch");
    } else {
        require(std::isfinite(actual) && std::abs(actual-expected) <= atol*(1+std::abs(expected)),
                why+" actual="+std::to_string(actual)+" expected="+std::to_string(expected));
    }
}

template<class T = uint32_t>
struct Buffer {
    std::vector<T> host, got;
    std::vector<bool> writable;
    void* device = nullptr;
    explicit Buffer(int n) : host(n+2*guard,static_cast<T>(sentinel)), got(host),
                             writable(host.size(),false) {
        device = rad_dev_alloc(static_cast<int64_t>(host.size()*sizeof(T)),RAD_MEM_DEVICE);
        require(device != nullptr,"device allocation");
    }
    ~Buffer() { if (device) rad_dev_free(device,RAD_MEM_DEVICE); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* h() { return host.data()+guard; }
    T* g() { return got.data()+guard; }
    void* d() { return static_cast<T*>(device)+guard; }
    void mark(int offset, int n) {
        std::fill_n(writable.begin()+guard+offset,n,true);
    }
    void upload(RadStream stream) {
        check(rad_memcpy_async(device,host.data(),host.size()*sizeof(T),stream),"upload");
    }
    void download(RadStream stream) {
        check(rad_memcpy_async(got.data(),device,got.size()*sizeof(T),stream),"download");
        check(rad_stream_sync(stream),"download sync");
    }
    void immutable(RadStream stream, const std::string& name) {
        download(stream);
        require(host == got,name+" input/padding/guard mutation");
    }
    void untouched(RadStream stream, const std::string& name) {
        immutable(stream,name);
        require(std::all_of(host.begin(),host.end(),[](T x) { return x == static_cast<T>(sentinel); }),
                name+" host mutated refused operands");
    }
    void guards(const std::string& name) const {
        for (size_t i=0;i<host.size();++i) if (!writable[i]) {
            require(host[i] == static_cast<T>(sentinel),name+" host padding/guard mutation");
            require(got[i] == static_cast<T>(sentinel),name+" device padding/guard mutation");
        }
    }
};
RadTensor matrix(void* p, int rows, int cols, int ld, uint32_t dtype = RAD_I32) {
    RadTensor t{}; t.data = p; t.dtype = dtype; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = cols; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
RadTensor vector(void* p, int n, int stride = 1) {
    RadTensor t{}; t.data = p; t.dtype = RAD_I32; t.rank = 1;
    t.shape[0] = n; t.stride[0] = stride; return t;
}
RadTensor history(void* p, int rounds, uint32_t dtype = RAD_I32) {
    auto t = matrix(p,rounds,seqs,hr,dtype);
    t.rank = 3; t.shape[2] = 8; t.stride[1] = hl; t.stride[2] = 1; return t;
}
RadTensor gathered_tensor(void* p, int M, int ld = gl) {
    auto t = matrix(p,M,16,ld,RAD_F32);
    t.rank = 3; t.shape[2] = 2; t.stride[1] = 2; t.stride[2] = 1; return t;
}
rad::Geometry geometry(int cap, int round, int rounds, int policy, const Costs& costs) {
    rad::Geometry g;
    g.set_i("M",seqs); g.set_i("pitch",8); g.set_i("capacity",cap);
    g.set_i("max_rounds",rounds); g.set_i("n_vocab",vocab); g.set_i("round",round);
    g.set_i("world_size",2); g.set_i("policy",policy);
    if (policy == 2) for (int i=0;i<8;++i)
        g.set_f("cost"+std::to_string(i),costs[i]);
    return g;
}
rad::Resolved resolve(rad::Registry& reg, const char* op, const rad::Geometry& g,
                      int domain, const char* row, const char* plugin) {
    rad::SelectionTrace why;
    auto r = reg.resolve(op,g,domain,&why);
    require(r.row && r.row->info,"missing "+std::string(op)+": "+why.reason);
    require(r.row->plugin == plugin && std::strcmp(r.row->info->name,row) == 0,
            "unexpected execution row for "+std::string(op));
    check(reg.init_instance(r,0,1),"init "+std::string(op));
    return r;
}
template<size_t N>
int launch(rad::Resolved& r, std::array<RadTensor,N>& t, RadStream stream,
           const rad::Geometry* override_geometry = nullptr, int count = static_cast<int>(N)) {
    const auto& g = override_geometry ? *override_geometry : r.geom;
    RadArgs a{}; a.t = t.data(); a.n_t = count; a.p = g.params(); a.n_p = g.n_params();
    a.world_size = 2; a.instance = r.instance;
    return r.row->info->launch(&a,stream);
}

// Full local vectors, in double precision. No top-k normalization or production helper.
struct Distribution {
    std::array<double,4> stats{};
    std::vector<double> logits;
};
Distribution full_vector(const Buffer<uint16_t>& coarse, const Buffer<>& idx,
                         const Buffer<>& val, int row, int N, int ld, int off, int nv) {
    Distribution out;
    out.stats[0] = -std::numeric_limits<double>::infinity();
    for (int j=0;j<N && off+j<nv;++j) {
        double x = value(static_cast<uint32_t>(coarse.host[guard+row*ld+j])<<16);
        for (int k=0;k<8;++k)
            if (static_cast<int32_t>(idx.host[guard+row*cl+k]) == off+j)
                x = value(val.host[guard+row*cl+k]);
        out.logits.push_back(x);
        out.stats[0] = std::max(out.stats[0],x);
    }
    if (std::isfinite(out.stats[0]))
        for (double x : out.logits) out.stats[1] += std::exp(x-out.stats[0]);
    return out;
}

struct Normalizer {
    int N, ld, nv;
    Buffer<uint16_t> coarse;
    Buffer<> idx, val, stats;
    Normalizer(int n, int n_vocab) : N(n), ld(n+6), nv(n_vocab), coarse(seqs*ld),
                                    idx(seqs*cl), val(seqs*cl), stats(seqs*8) {
        for (int s=0;s<seqs;++s) stats.mark(s*8,4);
    }
    std::array<Distribution,seqs> run(rad::Registry& reg, RadStream stream, int off,
                                     std::array<std::array<float,8>,seqs> exact,
                                     std::array<Eight,seqs> ids,
                                     const std::array<float,seqs>& background,
                                     std::array<std::array<float,8>,seqs>& gathered_stats,
                                     int rank) {
        for (int s=0;s<seqs;++s) {
            for (int j=0;j<N;++j) {
                // Padding has a huge value and must never enter the denominator.
                const float f = off+j<nv ? background[s] : 8192.0f;
                coarse.h()[s*ld+j] = static_cast<uint16_t>(bits(f)>>16);
            }
            for (int k=0;k<8;++k) {
                idx.h()[s*cl+k] = static_cast<uint32_t>(ids[s][k]);
                val.h()[s*cl+k] = bits(exact[s][k]);
            }
        }
        coarse.upload(stream); idx.upload(stream); val.upload(stream); stats.upload(stream);
        rad::Geometry g; g.set_i("M",seqs); g.set_i("N",N);
        g.set_i("n_vocab",nv); g.set_i("vocab_off",off);
        auto h = resolve(reg,"mtp_hybrid_lse",g,RAD_DOMAIN_HOST,"ref_mtp_hybrid_lse","mine_ref");
        auto d = resolve(reg,"mtp_hybrid_lse",g,RAD_DOMAIN_DEVICE,"mine_mtp_hybrid_lse","mine");
        auto bind = [&](bool device) {
            return std::array<RadTensor,4>{
                matrix(device ? coarse.d() : coarse.h(),seqs,N,ld,RAD_BF16),
                matrix(device ? idx.d() : idx.h(),seqs,8,cl),
                matrix(device ? val.d() : val.h(),seqs,8,cl,RAD_F32),
                matrix(device ? stats.d() : stats.h(),seqs,4,8,RAD_F32)};
        };
        std::array<Distribution,seqs> expected;
        for (int s=0;s<seqs;++s) expected[s] = full_vector(coarse,idx,val,s,N,ld,off,nv);
        auto ht = bind(false), dt = bind(true);
        check(launch(h,ht,nullptr),"host full-vector LSE");
        check(launch(d,dt,stream),"GPU full-vector LSE");
        stats.download(stream);
        for (int s=0;s<seqs;++s) {
            for (int k=0;k<4;++k) {
                close(value(stats.h()[s*8+k]),expected[s].stats[k],"host full-vector statistics");
                close(value(stats.g()[s*8+k]),expected[s].stats[k],"GPU full-vector statistics");
                gathered_stats[s][rank*4+k] = value(stats.g()[s*8+k]);
            }
        }
        stats.guards("LSE statistics");
        coarse.immutable(stream,"coarse BF16"); idx.immutable(stream,"local ids");
        val.immutable(stream,"local exact values");
        return expected;
    }
};

struct Candidate { int id; uint32_t image; };
// Numeric comparison is independent of the device's float-image/~id key network.
std::array<Candidate,8> merge_finite(const Buffer<>& gathered, int row) {
    std::vector<Candidate> all;
    for (int k=0;k<16;++k) {
        Candidate c{static_cast<int32_t>(gathered.host[guard+row*gl+2*k]),
                    gathered.host[guard+row*gl+2*k+1]};
        if (c.id >= 0 && c.id < vocab) all.push_back(c);
    }
    std::sort(all.begin(),all.end(),[](const Candidate& a, const Candidate& b) {
        const float av = value(a.image), bv = value(b.image);
        if (av != bv) return av > bv;
        if (std::signbit(av) != std::signbit(bv)) return !std::signbit(av);
        return a.id < b.id;
    });
    all.erase(std::unique(all.begin(),all.end(),[](const Candidate& a, const Candidate& b) {
        return a.id == b.id && a.image == b.image;
    }),all.end());
    std::array<Candidate,8> out;
    out.fill({-1,bits(-std::numeric_limits<float>::infinity())});
    std::copy_n(all.begin(),std::min<size_t>(8,all.size()),out.begin());
    return out;
}

struct Node { int token, parent, depth, candidate; double logpath; };
bool order(const Node& a, const Node& b) {
    if (a.logpath != b.logpath) return a.logpath > b.logpath;
    return std::tie(a.depth,a.token,a.candidate) < std::tie(b.depth,b.token,b.candidate);
}
std::vector<int32_t> pack_nodes(const std::vector<Node>& nodes, std::vector<int> chosen,
                              int root, int cap) {
    std::sort(chosen.begin(),chosen.end(),[&](int a, int b) { return order(nodes[a],nodes[b]); });
    std::vector<int32_t> out(1+2*cap,-1), positions(nodes.size(),-1);
    out[0] = static_cast<int32_t>(chosen.size()+1); out[1] = root;
    for (size_t i=0;i<chosen.size();++i) positions[chosen[i]] = static_cast<int>(i)+1;
    for (size_t i=0;i<chosen.size();++i) {
        const auto& n = nodes[chosen[i]];
        const int parent = n.parent < 0 ? 0 : positions[n.parent];
        require(parent >= 0 && parent < static_cast<int>(i)+1,"oracle ancestor closure");
        out[2+i] = n.token; out[2+cap+i] = parent;
    }
    return out;
}

// Exhaust all ancestor-closed subsets. This is not the production ranked-node selector.
// Maximizing summed log weights at each cardinality yields the globally highest nodes;
// all child probabilities are <= their greedy-spine ancestor probabilities.
std::vector<int32_t> subset_oracle(const std::vector<Eight>& ids,
                                   const std::vector<Logs>& logs, int rounds,
                                   int root, int cap, int policy, const Costs& costs) {
    std::vector<Node> nodes;
    int parent = -1;
    for (int r=0;r<rounds;++r) {
        int greedy = -1;
        for (int k=0;k<8 && ids[r][k]>=0;++k) {
            if (ids[r][k] >= vocab || !std::isfinite(logs[r][k])) continue;
            bool duplicate = false;
            for (const auto& n : nodes)
                duplicate = duplicate || (n.parent == parent && n.token == ids[r][k]);
            if (duplicate) continue;
            nodes.push_back({ids[r][k],parent,r+1,k,logs[r][k]});
            if (k == 0) greedy = static_cast<int>(nodes.size())-1;
        }
        if (greedy < 0) break;
        parent = greedy;
    }
    require(nodes.size() <= 20,"oracle fixture exceeds exhaustive-subset bound");
    std::vector<int> tie_order(nodes.size());
    for (size_t i=0;i<nodes.size();++i) tie_order[i] = static_cast<int>(i);
    std::sort(tie_order.begin(),tie_order.end(),[&](int a, int b) { return order(nodes[a],nodes[b]); });
    std::array<long double,8> best;
    best.fill(-std::numeric_limits<long double>::infinity());
    std::array<uint64_t,8> masks{};
    auto preferred = [&](uint64_t a, uint64_t b) {
        for (int i : tie_order) if (((a>>i)&1u) != ((b>>i)&1u)) return ((a>>i)&1u) != 0;
        return false;
    };
    const uint64_t end = uint64_t{1}<<nodes.size();
    for (uint64_t mask=0;mask<end;++mask) {
        int count = 0; long double sum = 0; bool closed = true;
        for (size_t i=0;i<nodes.size();++i) if ((mask>>i)&1u) {
            ++count; sum += nodes[i].logpath;
            if (nodes[i].parent >= 0 && !((mask>>nodes[i].parent)&1u)) { closed = false; break; }
        }
        if (!closed || count >= cap) continue;
        if (sum > best[count] || (sum == best[count] && preferred(mask,masks[count]))) {
            best[count] = sum; masks[count] = mask;
        }
    }
    int n = std::min<int>(cap-1,static_cast<int>(nodes.size()));
    while (n && !std::isfinite(best[n])) --n;
    if (policy == 2) {
        int winner = 0; long double utility = 1/costs[0];
        for (int k=1;k<=n;++k) {
            long double accepted = 1;
            for (size_t i=0;i<nodes.size();++i)
                if ((masks[k]>>i)&1u) accepted += std::exp(static_cast<long double>(nodes[i].logpath));
            const long double candidate = accepted/costs[k];
            if (candidate > utility) { winner = k; utility = candidate; }
        }
        n = winner;
    }
    std::vector<int> chosen;
    for (size_t i=0;i<nodes.size();++i) if ((masks[n]>>i)&1u) chosen.push_back(static_cast<int>(i));
    return pack_nodes(nodes,chosen,root,cap);
}
std::vector<int32_t> legacy_oracle(const std::vector<Eight>& rounds, int nr, int root, int cap) {
    std::vector<int32_t> out(1+2*cap,-1), tokens{root}, parents{-1};
    int spine = 0;
    for (int r=0;r<nr && static_cast<int>(tokens.size())<cap;++r) {
        if (rounds[r][0] < 0) break;
        parents.push_back(static_cast<int>(tokens.size())-1); tokens.push_back(rounds[r][0]); ++spine;
    }
    for (int r=0;r<spine;++r) for (int k=1;k<8 && static_cast<int>(tokens.size())<cap;++k) {
        const int id = rounds[r][k]; if (id < 0) break;
        bool duplicate = id == rounds[r][0];
        for (size_t j=0;j<tokens.size();++j) duplicate = duplicate || (parents[j] == r && tokens[j] == id);
        if (!duplicate) { tokens.push_back(id); parents.push_back(r); }
    }
    out[0] = static_cast<int32_t>(tokens.size());
    std::copy(tokens.begin(),tokens.end(),out.begin()+1);
    std::copy(parents.begin(),parents.end(),out.begin()+1+cap);
    return out;
}

struct Fixture {
    int cap, rounds, ol;
    Buffer<> gathered, stats, idx, val, tokens, ids, hist, loghist, packed;
    Fixture(int c, int r) : cap(c), rounds(r), ol(1+2*c+3), gathered(seqs*gl),
        stats(seqs*sl), idx(seqs*cl), val(seqs*cl), tokens(seqs*2), ids(seqs*2),
        hist(r*hr), loghist(r*hr), packed(seqs*ol) {
        for (int s=0;s<seqs;++s) {
            idx.mark(s*cl,8); val.mark(s*cl,8); packed.mark(s*ol,1+2*cap);
            for (int round=0;round<rounds;++round) {
                hist.mark(round*hr+s*hl,8); loghist.mark(round*hr+s*hl,8);
            }
        }
    }
    void upload(RadStream stream) {
        for (Buffer<>* b : {&gathered,&stats,&idx,&val,&tokens,&ids,&hist,&loghist,&packed}) b->upload(stream);
    }
    std::array<RadTensor,9> bind(bool device, int M, int round) {
        auto p = [&](Buffer<>& b) -> void* { return device ? b.d() : b.h(); };
        std::array<RadTensor,9> t = {gathered_tensor(p(gathered),M),matrix(p(stats),M,8,sl,RAD_F32),
            matrix(p(idx),M,8,cl),matrix(p(val),M,8,cl,RAD_F32),{}, {},
            history(p(hist),rounds),history(p(loghist),rounds,RAD_F32),
            matrix(p(packed),M,1+2*cap,ol)};
        if (round == 1) { t[4] = vector(p(tokens),seqs,2); t[5] = vector(p(ids),M,2); }
        return t;
    }
};

size_t run_sequence(rad::Registry& reg, RadStream stream, int cap, int rounds,
                    int policy, const Costs& costs) {
    Fixture f(cap,rounds);
    Normalizer norm(32,vocab);
    Buffer<> oldidx(seqs*cl), oldval(seqs*cl), oldhist(3*hr), oldpacked(seqs*f.ol);
    f.upload(stream); oldidx.upload(stream); oldval.upload(stream);
    oldhist.upload(stream); oldpacked.upload(stream);
    std::array<std::vector<Eight>,seqs> snapshots;
    std::array<std::vector<Logs>,seqs> log_snapshots;
    std::array<std::vector<int32_t>,seqs> expected_packed;
    std::array<std::array<Candidate,8>,seqs> previous_merged;
    for (int s=0;s<seqs;++s) {
        snapshots[s].resize(rounds); log_snapshots[s].resize(rounds);
        expected_packed[s].assign(1+2*cap,static_cast<int32_t>(sentinel));
        previous_merged[s].fill({static_cast<int32_t>(sentinel),sentinel});
        for (int r=0;r<rounds;++r) {
            snapshots[s][r].fill(static_cast<int32_t>(sentinel));
            log_snapshots[s][r].fill(value(sentinel));
        }
    }
    size_t launches = 0;
    for (int step=0;step<3;++step) {
        const int M = step == 1 ? 1 : seqs;
        const int nr = step == 1 ? 1 : rounds;
        for (int s=0;s<seqs;++s) {
            f.ids.h()[2*s] = seqs-1-s;
            f.tokens.h()[2*(seqs-1-s)] = 5+step*3+s;
        }
        f.ids.upload(stream); f.tokens.upload(stream);
        for (int round=1;round<=nr;++round) {
            std::array<std::array<float,8>,seqs> stats{};
            std::array<std::array<Distribution,seqs>,2> full;
            for (int rank=0;rank<2;++rank) {
                std::array<std::array<float,8>,seqs> exact{};
                std::array<Eight,seqs> local_ids;
                std::array<float,seqs> background = {-20,-20,128};
                for (int s=0;s<seqs;++s) {
                    local_ids[s].fill(-1);
                    const float primary = s == 0 ? (round == 3 ? 0.50f : 0.98f)
                                                 : (round == 1 ? 0.50f : 0.96f);
                    const float secondary = s == 0 ? (round == 3 ? 0.49f : 0.015f)
                                                   : (round == 1 ? 0.49f : 0.03f);
                    if (rank == 0) {
                        local_ids[s][0] = 10+round*3;
                        local_ids[s][1] = 11+round*3;
                        exact[s][0] = s == 2 ? -128.0f : std::log(primary);
                        exact[s][1] = s == 2 ? -129.0f : std::log(1-primary-secondary);
                    } else {
                        local_ids[s][0] = 40+round*3;
                        exact[s][0] = s == 2 ? -128.0f : std::log(secondary);
                    }
                    for (int k=0;k<8;++k) {
                        f.gathered.h()[s*gl+2*(rank*8+k)] = static_cast<uint32_t>(local_ids[s][k]);
                        f.gathered.h()[s*gl+2*(rank*8+k)+1] = bits(exact[s][k]);
                    }
                }
                if (policy != 0) {
                    full[rank] = norm.run(reg,stream,rank*32,exact,local_ids,background,stats,rank);
                    launches += 2;
                }
            }
            for (int s=0;s<seqs;++s) for (int k=0;k<8;++k)
                f.stats.h()[s*sl+k] = bits(stats[s][k]);
            f.gathered.upload(stream); f.stats.upload(stream);
            const auto g = geometry(cap,round,rounds,policy,costs);
            auto h = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                             "ref_mtp_scored_merge_tree_accum","mine_ref");
            auto d = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                             "mine_mtp_scored_merge_tree_accum","mine");
            auto ht = f.bind(false,M,round), dt = f.bind(true,M,round);
            if (policy == 0) { ht[1] = {}; ht[7] = {}; dt[1] = {}; dt[7] = {}; }
            check(launch(h,ht,nullptr),"host scored merge/tree");
            check(launch(d,dt,stream),"GPU scored merge/tree");
            f.idx.download(stream); f.val.download(stream); f.hist.download(stream);
            f.loghist.download(stream); f.packed.download(stream);
            for (int s=0;s<M;++s) {
                if (round == 1) for (int r=0;r<rounds;++r) {
                    snapshots[s][r].fill(-1);
                    log_snapshots[s][r].fill(-std::numeric_limits<double>::infinity());
                }
                const auto merged = merge_finite(f.gathered,s);
                const double maximum = std::max(full[0][s].stats[0],full[1][s].stats[0]);
                double sum = 0;
                for (int rank=0;rank<2;++rank) for (double x : full[rank][s].logits)
                    sum += std::exp(x-maximum);
                const double logz = policy == 0 ? 0 : maximum+std::log(sum);
                const double prefix = round == 1 ? 0 : log_snapshots[s][round-2][0];
                for (int k=0;k<8;++k) {
                    const uint32_t id = static_cast<uint32_t>(merged[k].id);
                    require(f.idx.h()[s*cl+k] == id && f.idx.g()[s*cl+k] == id,"independent merged id");
                    require(f.val.h()[s*cl+k] == merged[k].image && f.val.g()[s*cl+k] == merged[k].image,
                            "independent merged value image");
                    previous_merged[s][k] = merged[k];
                    snapshots[s][round-1][k] = merged[k].id;
                    log_snapshots[s][round-1][k] = merged[k].id < 0
                        ? -std::numeric_limits<double>::infinity()
                        : static_cast<float>(prefix+value(merged[k].image)-logz);
                }
                const int root = 5+step*3+s;
                expected_packed[s] = policy == 0 ? legacy_oracle(snapshots[s],round,root,cap)
                    : subset_oracle(snapshots[s],log_snapshots[s],round,root,cap,policy,costs);
                if (policy == 1 && cap == 5 && round == 3 && step == 0 && s < 2) {
                    const std::vector<int32_t> parents = s == 0 ? std::vector<int32_t>{-1,0,1,2,2}
                                                               : std::vector<int32_t>{-1,0,0,1,3};
                    require(std::equal(parents.begin(),parents.end(),expected_packed[s].begin()+1+cap),
                            "fixture must distinguish late branching from early uncertainty");
                }
            }
            for (int s=0;s<seqs;++s) {
                for (int k=0;k<8;++k) {
                    const auto& previous = previous_merged[s][k];
                    require(f.idx.h()[s*cl+k] == static_cast<uint32_t>(previous.id) &&
                            f.idx.g()[s*cl+k] == static_cast<uint32_t>(previous.id),
                            "merged ids changed outside live rows");
                    require(f.val.h()[s*cl+k] == previous.image && f.val.g()[s*cl+k] == previous.image,
                            "merged values changed outside live rows");
                }
                for (int k=0;k<1+2*cap;++k) {
                    const uint32_t expected = static_cast<uint32_t>(expected_packed[s][k]);
                    require(f.packed.h()[s*f.ol+k] == expected && f.packed.g()[s*f.ol+k] == expected,
                            "independent packed-tree oracle policy="+std::to_string(policy)+
                            " cap="+std::to_string(cap)+" round="+std::to_string(round)+" row="+std::to_string(s));
                }
                for (int r=0;r<rounds;++r) for (int k=0;k<8;++k) {
                    const int offset = r*hr+s*hl+k;
                    const uint32_t expected = static_cast<uint32_t>(snapshots[s][r][k]);
                    require(f.hist.h()[offset] == expected && f.hist.g()[offset] == expected,
                            "all-round history reset or preservation");
                    if (policy == 0) {
                        require(f.loghist.h()[offset] == sentinel && f.loghist.g()[offset] == sentinel,
                                "legacy policy touched unused log history");
                    } else {
                        close(value(f.loghist.h()[offset]),log_snapshots[s][r][k],"host cumulative logq");
                        close(value(f.loghist.g()[offset]),log_snapshots[s][r][k],"GPU cumulative logq");
                    }
                }
                std::vector<int32_t> row(1+2*cap);
                for (int k=0;k<1+2*cap;++k) row[k] = static_cast<int32_t>(f.packed.g()[s*f.ol+k]);
                radiance::CandidateTree parsed;
                const int root = 5+(s < M ? step : step-1)*3+s;
                require(radiance::dloop_tree_from_packed(parsed,static_cast<uint32_t>(root),
                                                        row.data(),cap,vocab),
                        "real packed-tree parser refused actual GPU output");
                require(parsed.count == f.packed.g()[s*f.ol],"parser changed actual target row count");
            }
            if (policy == 0 && round <= 3) {
                auto oldg = geometry(cap,round,3,0,costs);
                auto old = resolve(reg,"mtp_merge_tree_accum",oldg,RAD_DOMAIN_DEVICE,
                                   "mine_mtp_merge_tree_accum","mine");
                std::array<RadTensor,7> ot = {dt[0],matrix(oldidx.d(),M,8,cl),
                    matrix(oldval.d(),M,8,cl,RAD_F32),dt[4],dt[5],history(oldhist.d(),3),
                    matrix(oldpacked.d(),M,1+2*cap,f.ol)};
                check(launch(old,ot,stream),"frozen default TP merged path");
                oldidx.download(stream); oldval.download(stream); oldhist.download(stream); oldpacked.download(stream);
                for (int s=0;s<M;++s) {
                    require(std::equal(oldidx.g()+s*cl,oldidx.g()+s*cl+8,f.idx.g()+s*cl),"default TP ids changed");
                    require(std::equal(oldval.g()+s*cl,oldval.g()+s*cl+8,f.val.g()+s*cl),"default TP values changed");
                    require(std::equal(oldpacked.g()+s*f.ol,oldpacked.g()+s*f.ol+1+2*cap,
                                       f.packed.g()+s*f.ol),"default TP packed topology changed");
                    for (int r=0;r<3;++r)
                        require(std::equal(oldhist.g()+r*hr+s*hl,oldhist.g()+r*hr+s*hl+8,
                                           f.hist.g()+r*hr+s*hl),"default TP history changed");
                }
                ++launches;
            }
            if (policy == 2 && cap == 8 && round == 3 && step == 0) {
                const int expected_count = costs == flat_costs ? 8 : 2;
                require(f.packed.g()[0] == static_cast<uint32_t>(expected_count),
                        "cost table must change actual root-inclusive packed count");
                require(f.packed.g()[2*f.ol] == 1,
                        "cost objective must permit zero proposals/root-only count1");
            }
            if (policy == 1 && rounds == 5 && round == 5 && M == seqs)
                require(value(f.loghist.g()[4*hr+2*hl]) < -1000,"log-domain underflow fixture not reached");
            for (Buffer<>* b : {&f.idx,&f.val,&f.hist,&f.loghist,&f.packed}) b->guards("tree outputs");
            f.gathered.immutable(stream,"gathered pairs"); f.stats.immutable(stream,"two-rank statistics");
            f.tokens.immutable(stream,"anchors"); f.ids.immutable(stream,"anchor gather indices");
            launches += 2;
        }
    }
    return launches;
}

size_t normalization_outliers(rad::Registry& reg, RadStream stream) {
    Normalizer norm(37,65);
    std::array<std::array<float,8>,seqs> stats{};
    std::array<std::array<Distribution,seqs>,2> expected;
    std::array<double,seqs> top8_mass{};
    for (int rank=0;rank<2;++rank) {
        std::array<std::array<float,8>,seqs> exact{};
        std::array<Eight,seqs> ids;
        for (int s=0;s<seqs;++s) for (int k=0;k<8;++k) {
            ids[s][k] = rank*37+k;
            exact[s][k] = s == 0 ? -4.0f : (s == 1 ? 3.0f : -2.0f);
        }
        exact[1][0] = rank == 0 ? -32.0f : 12.0f; // Exact rescoring reverses coarse outliers.
        ids[2][7] = rank == 0 ? -1 : 70; // Invalid and padding replacement ids are ignored.
        expected[rank] = norm.run(reg,stream,rank*37,exact,ids,{0,4,-2},stats,rank);
        for (int s=0;s<seqs;++s) for (int k=0;k<8;++k)
            if (ids[s][k]>=0 && ids[s][k]<65) top8_mass[s] += std::exp(static_cast<double>(exact[s][k]));
    }
    for (int s=0;s<seqs;++s) {
        const double max = std::max(expected[0][s].stats[0],expected[1][s].stats[0]);
        double sum = 0;
        for (const auto& rank : expected) for (double x : rank[s].logits) sum += std::exp(x-max);
        const double gpu_max = std::max(stats[s][0],stats[s][4]);
        const double gpu_sum = stats[s][1]*std::exp(stats[s][0]-gpu_max)
                             + stats[s][5]*std::exp(stats[s][4]-gpu_max);
        const double probe = s == 1 ? 12.0 : (s == 0 ? -4.0 : -2.0);
        const double q = std::exp(probe-gpu_max)/gpu_sum;
        require(std::abs(q-std::exp(probe-max)/sum) <= 0.00002*std::exp(probe-max)/sum,
                "normalized two-rank q differs from full double vector");
        if (s == 0)
            require(q < std::exp(probe)/top8_mass[s]/100,"dominant tail must defeat top8-only normalization");
    }
    return 4;
}

size_t normalization_real_vocab(rad::Registry& reg, RadStream stream) {
    constexpr int local_vocab = 124160, model_vocab = 248320;
    Normalizer norm(local_vocab,model_vocab);
    std::array<std::array<float,8>,seqs> stats{};
    std::array<std::array<Distribution,seqs>,2> full;
    for (int rank=0;rank<2;++rank) {
        std::array<std::array<float,8>,seqs> exact{};
        std::array<Eight,seqs> ids;
        for (int s=0;s<seqs;++s) for (int k=0;k<8;++k) {
            ids[s][k] = rank*local_vocab+k;
            exact[s][k] = s == 0 ? -6.0f : -4.0f;
        }
        exact[1][0] = rank == 0 ? -80.0f : 16.0f;
        if (rank == 1) { ids[2][0] = model_vocab-1; exact[2][0] = 12; }
        full[rank] = norm.run(reg,stream,rank*local_vocab,exact,ids,
                             rank == 0 ? std::array<float,seqs>{-2,0,-4}
                                       : std::array<float,seqs>{0,2,-4},stats,rank);
    }
    for (int s=0;s<seqs;++s) {
        const double max = std::max(full[0][s].stats[0],full[1][s].stats[0]);
        double sum = 0;
        for (const auto& rank : full) for (double x : rank[s].logits) sum += std::exp(x-max);
        const double gm = std::max(stats[s][0],stats[s][4]);
        const double gs = stats[s][1]*std::exp(stats[s][0]-gm)+stats[s][5]*std::exp(stats[s][4]-gm);
        const double probe = s == 0 ? -6 : (s == 1 ? 16 : 12);
        const double expected = std::exp(probe-max)/sum;
        const double actual = std::exp(probe-gm)/gs;
        require(std::abs(actual-expected) <= 0.00003*expected,
                "124160-column TP2 normalized q differs from independent double vector");
        if (s == 0) require(actual < 0.000001,"deployment-size dominant tail was discarded");
        if (s == 2) require(actual > 0.9,"model vocabulary tail was confused with tokenizer padding");
    }
    return 4;
}

size_t frozen_merge_images(rad::Registry& reg, RadStream stream) {
    Fixture f(8,3);
    Buffer<> tight(seqs*32), oldidx(seqs*8), oldval(seqs*8);
    const std::array<uint32_t,16> special = {
        0x7fc00001u,0x7fc00002u,0x7f800000u,0u,0x80000000u,0xff800000u,
        0xffc00001u,0xffc00002u,0x7f800001u,0xff800001u,0x3f800000u,0xbf800000u,
        1u,0x80000001u,0x7fffffffu,0xffffffffu};
    for (int s=0;s<seqs;++s) {
        f.tokens.h()[s*2] = 5+s; f.ids.h()[s*2] = s;
        for (int rank=0;rank<2;++rank) {
            f.stats.h()[s*sl+rank*4] = bits(128);
            f.stats.h()[s*sl+rank*4+1] = bits(64);
            f.stats.h()[s*sl+rank*4+2] = 0; f.stats.h()[s*sl+rank*4+3] = 0;
        }
        for (int k=0;k<16;++k) {
            int id = 10+k;
            uint32_t image = s == 1 ? special[k] : bits(-static_cast<float>(k));
            if (s == 0 && k == 8) { id = 10; image = bits(0); } // Equal key removed simultaneously.
            if (s == 0 && k == 9) { id = 10; image = bits(-1); } // Same id/different key survives merge.
            if (s == 0 && k == 10) { id = 11; image = 0x80000000u; }
            if (s == 2 && k >= 8) { id = vocab+k; image = bits(1000.0f+k); }
            f.gathered.h()[s*gl+2*k] = static_cast<uint32_t>(id);
            f.gathered.h()[s*gl+2*k+1] = image;
        }
        std::copy_n(f.gathered.h()+s*gl,32,tight.h()+s*32);
    }
    f.upload(stream); tight.upload(stream); oldidx.upload(stream); oldval.upload(stream);
    auto g = geometry(8,1,3,0,flat_costs);
    auto h = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                     "ref_mtp_scored_merge_tree_accum","mine_ref");
    auto d = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                     "mine_mtp_scored_merge_tree_accum","mine");
    rad::Geometry mg; mg.set_i("M",seqs); mg.set_i("R",8);
    mg.set_i("world_size",2); mg.set_s("dtype","bf16");
    auto old = resolve(reg,"row_topk_merge",mg,RAD_DOMAIN_DEVICE,"rowtopk_merge","libr4d");
    std::array<RadTensor,3> ot = {gathered_tensor(tight.d(),seqs,32),
        matrix(oldidx.d(),seqs,8,8),matrix(oldval.d(),seqs,8,8,RAD_F32)};
    check(launch(old,ot,stream),"actual pinned float-image merge");
    auto ht = f.bind(false,seqs,1), dt = f.bind(true,seqs,1);
    check(launch(h,ht,nullptr),"host float-image merge"); check(launch(d,dt,stream),"GPU float-image merge");
    oldidx.download(stream); oldval.download(stream); f.idx.download(stream); f.val.download(stream);
    f.hist.download(stream); f.loghist.download(stream); f.packed.download(stream);
    for (int s=0;s<2;++s) for (int k=0;k<8;++k) {
        require(f.idx.h()[s*cl+k] == oldidx.g()[s*8+k] && f.idx.g()[s*cl+k] == oldidx.g()[s*8+k],
                "frozen float-image/~id and simultaneous duplicate-key ids changed");
        require(f.val.h()[s*cl+k] == oldval.g()[s*8+k] && f.val.g()[s*cl+k] == oldval.g()[s*8+k],
                "frozen float-image merge value bits changed");
    }
    require(oldidx.g()[2*8] >= vocab,"dominant invalid gathered id fixture did not reach pinned merge");
    for (int k=0;k<8;++k) {
        require(f.idx.h()[2*cl+k] == static_cast<uint32_t>(10+k) &&
                f.idx.g()[2*cl+k] == static_cast<uint32_t>(10+k),
                "padding id reached scored merge output/next MTP embedding");
        require(f.val.h()[2*cl+k] == bits(-static_cast<float>(k)) &&
                f.val.g()[2*cl+k] == bits(-static_cast<float>(k)),"padding merge value filter");
    }
    std::vector<int32_t> children;
    for (int k=0;k<8;++k) {
        const int token = static_cast<int32_t>(oldidx.g()[k]);
        if (token >= 0 && std::find(children.begin(),children.end(),token) == children.end())
            children.push_back(token);
    }
    require(children.size()<8,"duplicate-parent/token fixture did not reach merge output");
    g.set_i("policy",1);
    auto sh = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                      "ref_mtp_scored_merge_tree_accum","mine_ref");
    auto sd = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                      "mine_mtp_scored_merge_tree_accum","mine");
    check(launch(sh,ht,nullptr),"host score duplicate collapse");
    check(launch(sd,dt,stream),"GPU score duplicate collapse");
    f.packed.download(stream);
    require(f.packed.h()[0] == children.size()+1 && f.packed.g()[0] == children.size()+1,
            "score policy retained duplicate nodes with the same parent/token");
    for (size_t k=0;k<children.size();++k) {
        require(f.packed.h()[2+k] == static_cast<uint32_t>(children[k]) &&
                f.packed.g()[2+k] == static_cast<uint32_t>(children[k]),"score duplicate token ordering");
        require(f.packed.h()[10+k] == 0 && f.packed.g()[10+k] == 0,"duplicate collapse changed root parent");
    }
    f.idx.download(stream); f.val.download(stream); f.hist.download(stream); f.loghist.download(stream);
    for (Buffer<>* b : {&f.idx,&f.val,&f.hist,&f.loghist,&f.packed}) b->guards("float-image outputs");
    f.gathered.immutable(stream,"float-image gathered input"); tight.immutable(stream,"pinned gathered input");
    return 5;
}

size_t legacy_unused_scores(rad::Registry& reg, RadStream stream, int rounds) {
    size_t launches = 0;
    for (int variant=0;variant<3;++variant) {
        Fixture f(8,rounds);
        const uint32_t poison = 0x7fc54321u;
        for (int s=0;s<seqs;++s) {
            for (int k=0;k<8;++k) f.stats.h()[s*sl+k] = poison;
            for (int r=0;r<rounds;++r) for (int k=0;k<8;++k)
                f.loghist.h()[r*hr+s*hl+k] = poison;
        }
        const auto stats_before = f.stats.host, logs_before = f.loghist.host;
        f.upload(stream);
        std::array<std::vector<Eight>,seqs> snapshots;
        for (auto& row : snapshots) row.resize(rounds);
        for (int step=0;step<2;++step) {
            const int nr = step == 0 ? rounds : 1;
            for (int s=0;s<seqs;++s) {
                f.ids.h()[2*s] = seqs-1-s;
                f.tokens.h()[2*(seqs-1-s)] = 5+step*3+s;
                for (auto& row : snapshots[s]) row.fill(-1);
            }
            f.ids.upload(stream); f.tokens.upload(stream);
            for (int round=1;round<=nr;++round) {
                for (int s=0;s<seqs;++s) for (int k=0;k<16;++k) {
                    f.gathered.h()[s*gl+2*k] = 20+(round*3+s+k)%35;
                    f.gathered.h()[s*gl+2*k+1] = bits(3.0f-0.25f*k);
                }
                f.gathered.upload(stream);
                const auto g = geometry(8,round,rounds,0,flat_costs);
                auto h = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                                 "ref_mtp_scored_merge_tree_accum","mine_ref");
                auto d = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                                 "mine_mtp_scored_merge_tree_accum","mine");
                auto ht = f.bind(false,seqs,round), dt = f.bind(true,seqs,round);
                auto unused = [&](std::array<RadTensor,9>& t) {
                    if (variant == 0) { t[1] = {}; t[7] = {}; }
                    if (variant == 2) {
                        // Even invalid metadata and output aliases are irrelevant in ID-only mode.
                        t[1] = t[8]; t[1].dtype = RAD_BF16; t[1].stride[0] = -1;
                        t[7] = t[6]; t[7].rank = 0; t[7].stride[0] = INT64_MAX;
                    }
                };
                unused(ht); unused(dt);
                check(launch(h,ht,nullptr),"host legacy absent/poisoned score operands");
                check(launch(d,dt,stream),"GPU legacy absent/poisoned score operands");
                f.idx.download(stream); f.val.download(stream); f.hist.download(stream); f.packed.download(stream);
                for (int s=0;s<seqs;++s) {
                    const auto merged = merge_finite(f.gathered,s);
                    for (int k=0;k<8;++k) {
                        snapshots[s][round-1][k] = merged[k].id;
                        require(f.idx.h()[s*cl+k] == static_cast<uint32_t>(merged[k].id) &&
                                f.idx.g()[s*cl+k] == static_cast<uint32_t>(merged[k].id),
                                "legacy merge used poisoned score operands");
                        require(f.val.h()[s*cl+k] == merged[k].image && f.val.g()[s*cl+k] == merged[k].image,
                                "legacy merged value image changed without score operands");
                    }
                    const auto expected = legacy_oracle(snapshots[s],round,5+step*3+s,8);
                    for (int k=0;k<17;++k)
                        require(f.packed.h()[s*f.ol+k] == static_cast<uint32_t>(expected[k]) &&
                                f.packed.g()[s*f.ol+k] == static_cast<uint32_t>(expected[k]),
                                "legacy topology depends on absent/poisoned normalization or log history");
                    for (int r=0;r<rounds;++r) for (int k=0;k<8;++k) {
                        const uint32_t expected_id = static_cast<uint32_t>(snapshots[s][r][k]);
                        require(f.hist.h()[r*hr+s*hl+k] == expected_id &&
                                f.hist.g()[r*hr+s*hl+k] == expected_id,
                                "legacy ID-only history reset/round bounds");
                    }
                }
                f.stats.immutable(stream,"unused legacy statistics");
                f.loghist.immutable(stream,"unused legacy log history");
                require(f.stats.host == stats_before && f.loghist.host == logs_before,
                        "host legacy implementation wrote unused score operands");
                for (Buffer<>* b : {&f.idx,&f.val,&f.hist,&f.packed}) b->guards("ID-only legacy outputs");
                f.gathered.immutable(stream,"ID-only gathered input");
                launches += 2;
            }
        }
    }
    return launches;
}

size_t depth_ties_and_zero_probability(rad::Registry& reg, RadStream stream, int rounds) {
    Fixture f(8,rounds);
    Normalizer norm(32,vocab);
    for (int s=0;s<seqs;++s) { f.tokens.h()[2*s] = 7+s; f.ids.h()[2*s] = s; }
    f.upload(stream);
    size_t launches = 0;
    for (int round=1;round<=rounds;++round) {
        std::array<std::array<float,8>,seqs> st{};
        for (int rank=0;rank<2;++rank) {
            std::array<std::array<float,8>,seqs> exact{};
            std::array<Eight,seqs> ids;
            for (int s=0;s<seqs;++s) {
                ids[s].fill(-1);
                if (rank == 0) {
                    ids[s][0] = 30-round; ids[s][1] = 4; ids[s][2] = 3;
                    exact[s][1] = exact[s][2] = -std::numeric_limits<float>::infinity();
                }
                for (int k=0;k<8;++k) {
                    f.gathered.h()[s*gl+2*(rank*8+k)] = static_cast<uint32_t>(ids[s][k]);
                    f.gathered.h()[s*gl+2*(rank*8+k)+1] = bits(exact[s][k]);
                }
            }
            norm.run(reg,stream,rank*32,exact,ids,
                     {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()},st,rank);
            launches += 2;
        }
        for (int s=0;s<seqs;++s) for (int k=0;k<8;++k) f.stats.h()[s*sl+k] = bits(st[s][k]);
        f.gathered.upload(stream); f.stats.upload(stream);
        const auto g = geometry(8,round,rounds,1,flat_costs);
        auto h = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                         "ref_mtp_scored_merge_tree_accum","mine_ref");
        auto d = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                         "mine_mtp_scored_merge_tree_accum","mine");
        auto ht = f.bind(false,seqs,round), dt = f.bind(true,seqs,round);
        check(launch(h,ht,nullptr),"host depth ties/zero probability");
        check(launch(d,dt,stream),"GPU depth ties/zero probability");
        f.packed.download(stream); f.idx.download(stream); f.val.download(stream);
        f.hist.download(stream); f.loghist.download(stream);
        for (int s=0;s<seqs;++s) {
            std::vector<int32_t> expected(17,-1), tokens{7+s}, parents{-1};
            // Literal q=1 spine, then q=0 ties: depth precedes token id.
            for (int r=1;r<=round;++r) { tokens.push_back(30-r); parents.push_back(r-1); }
            for (int r=0;r<round;++r) for (int token : {3,4}) if (tokens.size()<8) {
                tokens.push_back(token); parents.push_back(r);
            }
            expected[0] = static_cast<int32_t>(tokens.size());
            std::copy(tokens.begin(),tokens.end(),expected.begin()+1);
            std::copy(parents.begin(),parents.end(),expected.begin()+9);
            for (int k=0;k<17;++k)
                require(f.packed.h()[s*f.ol+k] == static_cast<uint32_t>(expected[k]) &&
                        f.packed.g()[s*f.ol+k] == static_cast<uint32_t>(expected[k]),
                        "equal logq depth/token ordering, zero probability, or ancestor closure");
            require(value(f.loghist.h()[(round-1)*hr+s*hl]) == 0 &&
                    value(f.loghist.g()[(round-1)*hr+s*hl]) == 0,"q=1 cumulative log must remain zero");
            require(value(f.loghist.h()[(round-1)*hr+s*hl+1]) == -std::numeric_limits<float>::infinity() &&
                    value(f.loghist.g()[(round-1)*hr+s*hl+1]) == -std::numeric_limits<float>::infinity(),
                    "q=0 must remain valid negative-infinite log probability");
        }
        for (Buffer<>* b : {&f.idx,&f.val,&f.hist,&f.loghist,&f.packed}) b->guards("zero-probability outputs");
        launches += 2;
    }
    return launches;
}

void malformed(rad::Registry& reg, RadStream stream, int rounds) {
    Fixture f(8,rounds); f.upload(stream);
    const auto g = geometry(8,rounds,rounds,2,flat_costs);
    auto h = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                     "ref_mtp_scored_merge_tree_accum","mine_ref");
    auto d = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                     "mine_mtp_scored_merge_tree_accum","mine");
    auto ht = f.bind(false,seqs,rounds), dt = f.bind(true,seqs,rounds);
    auto refuse = [&](std::array<RadTensor,9> bh, std::array<RadTensor,9> bd, const std::string& why,
                      const rad::Geometry* badg = nullptr, int n = 9) {
        require(launch(h,bh,nullptr,badg,n) != RAD_OK,"host accepted "+why);
        require(launch(d,bd,stream,badg,n) != RAD_OK,"GPU accepted "+why);
    };
    auto change = [&](int operand, auto mutation, const char* why) {
        auto bh = ht, bd = dt; mutation(bh[operand]); mutation(bd[operand]); refuse(bh,bd,why);
    };
    for (int i : {0,1,2,3,6,7,8}) {
        change(i,[](RadTensor& t) { t.data = nullptr; },"null required operand");
        change(i,[](RadTensor& t) { t.rank = 1; },"wrong operand rank");
        change(i,[](RadTensor& t) { t.dtype = RAD_BF16; },"wrong operand dtype");
        change(i,[](RadTensor& t) { t.stride[0] = INT64_MAX; },"overflowing tensor span");
        change(i,[](RadTensor& t) { t.stride[0] = -1; },"negative stride");
        change(i,[](RadTensor& t) { t.shape[0] = 0; },"empty required extent");
        change(i,[](RadTensor& t) { t.data = static_cast<char*>(t.data)+1; },"misaligned data");
    }
    change(0,[](RadTensor& t) { t.shape[1] = 15; },"short gathered pair list");
    change(0,[](RadTensor& t) { t.shape[2] = 3; },"wrong gathered pair width");
    change(0,[](RadTensor& t) { t.stride[1] = 3; },"non-dense gathered pairs");
    change(0,[](RadTensor& t) { t.stride[2] = 2; },"non-dense gathered pair elements");
    change(1,[](RadTensor& t) { t.shape[1] = 7; },"short two-rank statistics");
    change(1,[](RadTensor& t) { t.stride[1] = 2; },"non-dense statistics");
    change(1,[](RadTensor& t) { t.stride[0] = 13; },"unaligned statistics row");
    change(1,[](RadTensor& t) { t.data = static_cast<char*>(t.data)+4; },"unaligned statistics base");
    for (int i : {2,3}) {
        change(i,[](RadTensor& t) { t.shape[1] = 9; },"wrong merged output width");
        change(i,[](RadTensor& t) { t.stride[1] = 2; },"non-dense merged output");
    }
    for (int i : {6,7}) {
        change(i,[&](RadTensor& t) { t.shape[0] = rounds-1; },"truncated last history round");
        change(i,[&](RadTensor& t) { t.shape[0] = rounds+1; },"extra history round");
        change(i,[](RadTensor& t) { t.shape[2] = 7; },"short history pitch");
        change(i,[](RadTensor& t) { t.stride[2] = 2; },"non-dense history candidates");
        change(i,[](RadTensor& t) { t.stride[1] = 7; },"overlapping history rows");
        change(i,[](RadTensor& t) { t.stride[0] = 8; },"overlapping history rounds");
    }
    change(8,[](RadTensor& t) { t.shape[1] = 16; },"missing final packed parent");
    change(8,[](RadTensor& t) { t.stride[1] = 2; },"non-dense packed tree");
    for (int w : {2,3,6,7,8}) for (int other : {0,1,2,3,6,7,8}) if (w != other) {
        auto bh = ht, bd = dt; bh[w].data = bh[other].data; bd[w].data = bd[other].data;
        refuse(bh,bd,"output alias");
    }
    for (const auto& p : std::array<std::pair<const char*,int64_t>,12>{
            {{"M",seqs-1},{"M",0},{"pitch",7},{"capacity",1},{"capacity",9},
             {"max_rounds",4},{"round",0},{"round",rounds+1},{"n_vocab",0},
             {"world_size",1},{"policy",-1},{"policy",3}}}) {
        auto badg = g; badg.set_i(p.first,p.second); refuse(ht,dt,p.first,&badg);
    }
    for (double badcost : {0.0,-1.0,std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        for (int i=0;i<8;++i) {
            auto badg = g; badg.set_f("cost"+std::to_string(i),badcost);
            refuse(ht,dt,"non-positive/non-finite cost",&badg);
        }
    }
    auto missing_costs = geometry(8,rounds,rounds,1,flat_costs); missing_costs.set_i("policy",2);
    refuse(ht,dt,"missing cost table",&missing_costs);
    auto integer_cost = g; integer_cost.set_i("cost0",1);
    refuse(ht,dt,"non-F64 cost parameter",&integer_cost);
    refuse(ht,dt,"short operand list",nullptr,8);
    refuse(ht,dt,"extra operand",nullptr,10);
    // Round-one anchors and gather ids are mandatory even in the five-round geometry.
    auto firstg = geometry(8,1,rounds,2,flat_costs);
    auto firsth = f.bind(false,seqs,1), firstd = f.bind(true,seqs,1);
    for (int i : {4,5}) {
        auto bh = firsth, bd = firstd; bh[i] = {}; bd[i] = {};
        refuse(bh,bd,"missing round-one anchor operand",&firstg);
        bh = firsth; bd = firstd; bh[i].stride[0] = INT64_MAX; bd[i].stride[0] = INT64_MAX;
        refuse(bh,bd,"overflowing anchor span",&firstg);
    }
    auto bh = firsth, bd = firstd; bh[5].shape[0] = seqs-1; bd[5].shape[0] = seqs-1;
    refuse(bh,bd,"short round-one gather ids",&firstg);
    for (int w : {2,3,6,7,8}) for (int other : {4,5}) {
        bh = firsth; bd = firstd; bh[w].data = bh[other].data; bd[w].data = bd[other].data;
        refuse(bh,bd,"output overlaps round-one anchor operand",&firstg);
    }
    // Refusal is observable: every allocation, including every guard, remains unchanged.
    for (Buffer<>* b : {&f.gathered,&f.stats,&f.idx,&f.val,&f.tokens,&f.ids,&f.hist,&f.loghist,&f.packed})
        b->untouched(stream,"refused launch");
}

void malformed_lse(rad::Registry& reg, RadStream stream) {
    Normalizer n(37,65);
    n.coarse.upload(stream); n.idx.upload(stream); n.val.upload(stream); n.stats.upload(stream);
    rad::Geometry g; g.set_i("M",seqs); g.set_i("N",37); g.set_i("n_vocab",65); g.set_i("vocab_off",37);
    auto h = resolve(reg,"mtp_hybrid_lse",g,RAD_DOMAIN_HOST,"ref_mtp_hybrid_lse","mine_ref");
    auto d = resolve(reg,"mtp_hybrid_lse",g,RAD_DOMAIN_DEVICE,"mine_mtp_hybrid_lse","mine");
    auto bind = [&](bool dev) {
        return std::array<RadTensor,4>{matrix(dev ? n.coarse.d() : n.coarse.h(),seqs,37,n.ld,RAD_BF16),
            matrix(dev ? n.idx.d() : n.idx.h(),seqs,8,cl),matrix(dev ? n.val.d() : n.val.h(),seqs,8,cl,RAD_F32),
            matrix(dev ? n.stats.d() : n.stats.h(),seqs,4,8,RAD_F32)};
    };
    auto ht = bind(false), dt = bind(true);
    auto reject = [&](std::array<RadTensor,4> bh, std::array<RadTensor,4> bd,
                      const rad::Geometry* badg = nullptr, int count = 4) {
        require(launch(h,bh,nullptr,badg,count) != RAD_OK,"host accepted malformed LSE");
        require(launch(d,bd,stream,badg,count) != RAD_OK,"GPU accepted malformed LSE");
    };
    auto change = [&](int i, auto mutation) {
        auto bh = ht, bd = dt; mutation(bh[i]); mutation(bd[i]); reject(bh,bd);
    };
    for (int i=0;i<4;++i) {
        change(i,[](RadTensor& t) { t.data = nullptr; });
        change(i,[](RadTensor& t) { t.rank = 3; });
        change(i,[](RadTensor& t) { t.stride[1] = 2; });
        change(i,[](RadTensor& t) { t.stride[0] = INT64_MAX; });
        change(i,[](RadTensor& t) { --t.shape[0]; });
    }
    change(0,[](RadTensor& t) { t.dtype = RAD_F32; });
    change(0,[](RadTensor& t) { --t.shape[1]; });
    change(1,[](RadTensor& t) { t.dtype = RAD_F32; });
    change(1,[](RadTensor& t) { t.shape[1] = 7; });
    change(2,[](RadTensor& t) { t.dtype = RAD_I32; });
    change(3,[](RadTensor& t) { t.shape[1] = 3; });
    change(3,[](RadTensor& t) { t.stride[0] = 9; });
    change(3,[](RadTensor& t) { t.data = static_cast<char*>(t.data)+4; });
    for (int i=0;i<3;++i) {
        auto bh = ht, bd = dt; bh[3].data = bh[i].data; bd[3].data = bd[i].data; reject(bh,bd);
    }
    for (const auto& p : std::array<std::pair<const char*,int64_t>,6>{
            {{"M",0},{"N",0},{"N",INT64_MAX},{"n_vocab",0},{"vocab_off",-1},{"vocab_off",INT64_MAX}}}) {
        auto badg = g; badg.set_i(p.first,p.second); reject(ht,dt,&badg);
    }
    reject(ht,dt,nullptr,3);
    n.coarse.untouched(stream,"refused LSE coarse"); n.idx.untouched(stream,"refused LSE ids");
    n.val.untouched(stream,"refused LSE values"); n.stats.untouched(stream,"refused LSE statistics");
}

void invalid_statistics(rad::Registry& reg, RadStream stream) {
    for (int rounds : {3,5}) for (int policy : {1,2}) {
        Fixture f(8,rounds);
        for (int s=0;s<seqs;++s) {
            f.tokens.h()[2*s] = 5+s; f.ids.h()[2*s] = s;
            for (int k=0;k<16;++k) {
                f.gathered.h()[s*gl+2*k] = k<3 ? 10+k : static_cast<uint32_t>(-1);
                f.gathered.h()[s*gl+2*k+1] = bits(-static_cast<float>(k));
            }
            for (int rank=0;rank<2;++rank) {
                f.stats.h()[s*sl+4*rank] = bits(0);
                f.stats.h()[s*sl+4*rank+1] = bits(1);
                f.stats.h()[s*sl+4*rank+2] = bits(0);
                f.stats.h()[s*sl+4*rank+3] = bits(0);
            }
        }
        f.stats.h()[1] = bits(0);
        f.stats.h()[sl] = bits(std::numeric_limits<float>::quiet_NaN());
        f.stats.h()[2*sl+5] = bits(-1);
        f.upload(stream);
        const auto g = geometry(8,1,rounds,policy,flat_costs);
        auto h = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_HOST,
                         "ref_mtp_scored_merge_tree_accum","mine_ref");
        auto d = resolve(reg,"mtp_scored_merge_tree_accum",g,RAD_DOMAIN_DEVICE,
                         "mine_mtp_scored_merge_tree_accum","mine");
        auto ht = f.bind(false,seqs,1), dt = f.bind(true,seqs,1);
        check(launch(h,ht,nullptr),"host runtime-invalid statistics");
        check(launch(d,dt,stream),"GPU runtime-invalid statistics");
        f.packed.download(stream); f.hist.download(stream); f.loghist.download(stream);
        f.idx.download(stream); f.val.download(stream);
        for (int s=0;s<seqs;++s) for (int k=0;k<17;++k) {
            const uint32_t expected = k == 0 ? 0u : static_cast<uint32_t>(-1);
            require(f.packed.h()[s*f.ol+k] == expected && f.packed.g()[s*f.ol+k] == expected,
                    "invalid normalization did not clear complete packed tree");
        }
        for (Buffer<>* b : {&f.idx,&f.val,&f.hist,&f.loghist,&f.packed}) b->guards("invalid-stat output");
    }
}

struct SavedEnv {
    std::array<const char*,3> names = {"RADIANCE_MTP_TREE_POLICY","RADIANCE_MTP_TREE_ROUNDS","RADIANCE_MTP_TREE_COSTS"};
    std::array<std::string,3> values;
    std::array<bool,3> present{};
    SavedEnv() {
        for (int i=0;i<3;++i) {
            const char* p = std::getenv(names[i]); present[i] = p != nullptr;
            if (p) values[i] = p;
            require(unsetenv(names[i]) == 0,"clear test environment");
        }
    }
    ~SavedEnv() {
        for (int i=0;i<3;++i) {
            if (present[i]) setenv(names[i],values[i].c_str(),1);
            else unsetenv(names[i]);
        }
    }
    void set(const char* policy, const char* rounds, const char* costs) {
        const char* inputs[] = {policy,rounds,costs};
        for (int i=0;i<3;++i)
            require((inputs[i] ? setenv(names[i],inputs[i],1) : unsetenv(names[i])) == 0,"set test environment");
    }
};
void configuration() {
    SavedEnv env;
    auto parse = [&](const char* policy, const char* rounds, const char* costs, bool valid,
                     int expected_policy = 0, int expected_rounds = 3) {
        env.set(policy,rounds,costs);
        rad::MtpTreeConfig c{}; const char* error = nullptr;
        const bool ok = rad::parse_mtp_tree_config(c,&error);
        require(ok == valid,"startup policy/cost configuration acceptance");
        if (ok) {
            require(c.policy == expected_policy && c.rounds == expected_rounds,"startup policy/default rounds");
            if (expected_policy == 2) for (int i=0;i<8;++i)
                close(c.costs[i],i+1,"F64 startup cost table",0);
        } else require(error && *error,"configuration refusal needs an error");
    };
    parse(nullptr,nullptr,nullptr,true);
    parse("off","3",nullptr,true);
    parse("score","5",nullptr,true,1,5);
    parse("cost","5","1,2,3,4,5,6,7,8",true,2,5);
    parse(nullptr,"8",nullptr,true,0,8);
    parse("off","8",nullptr,true,0,8);
    // Fixed MTP8/64 only: no scored/cost/pruned variant, no other widths.
    parse("score","8",nullptr,false);
    parse("cost","8","1,2,3,4,5,6,7,8",false);
    parse("pruned","8",nullptr,false);
    for (const char* r : {"7", "9", "15", "08", "64"}) parse("off",r,nullptr,false);
    for (const char* p : {"", "on", "Score", "legacy", " score", "score "}) parse(p,nullptr,nullptr,false);
    for (const char* r : {"", "0", "2", "4", "6", "03", "3x"}) parse("score",r,nullptr,false);
    for (const char* c : {"", "1,2,3,4,5,6,7", "1,2,3,4,5,6,7,8,9", "1,2,3,4,5,6,7,8,",
                          "1,,3,4,5,6,7,8", "0,2,3,4,5,6,7,8", "1,2,-3,4,5,6,7,8",
                          "1,2,3,nan,5,6,7,8", "1,2,3,4,inf,6,7,8", "1,2,3,4,5,6,7,1e999",
                          "1,2,3,4,5,6,7,8x"}) parse("cost","3",c,false);
    parse("cost",nullptr,nullptr,false);
    parse("off",nullptr,"1,2,3,4,5,6,7,8",false);
    parse("score",nullptr,"1,2,3,4,5,6,7,8",false);
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 7 && std::string(argv[1]) == "--mine" && std::string(argv[3]) == "--ref" &&
                std::string(argv[5]) == "--libr4d",
                "usage: mtp_scored_tree_gpu_test --mine PATH --ref PATH --libr4d PATH");
        configuration();
        check(rad_dev_set(0),"set device");
        RadDeviceProps props{}; check(rad_dev_props(0,&props),"device properties");
        require(!props.is_host_backend,"real GPU required; host-only execution is not a pass");
        rad::Registry reg; reg.set_code_device(0);
        check(reg.load_plugin(argv[2],0),"load mine"); check(reg.load_plugin(argv[4],1),"load mine_ref");
        check(reg.load_plugin(argv[6],2),"load pinned libr4d"); check(reg.check_complete(),"complete schemas");
        RadStream stream = nullptr; check(rad_stream_create(&stream,0),"create stream");
        size_t launches = 0;
        try {
            malformed_lse(reg,stream);
            launches += normalization_outliers(reg,stream);
            launches += normalization_real_vocab(reg,stream);
            launches += frozen_merge_images(reg,stream);
            for (int rounds : {3,5}) {
                malformed(reg,stream,rounds);
                launches += legacy_unused_scores(reg,stream,rounds);
                launches += depth_ties_and_zero_probability(reg,stream,rounds);
                for (int cap=2;cap<=8;++cap) {
                    launches += run_sequence(reg,stream,cap,rounds,0,flat_costs);
                    launches += run_sequence(reg,stream,cap,rounds,1,flat_costs);
                }
                launches += run_sequence(reg,stream,8,rounds,2,flat_costs);
                launches += run_sequence(reg,stream,8,rounds,2,steep_costs);
            }
            invalid_statistics(reg,stream);
        } catch (...) { rad_stream_destroy(stream); throw; }
        rad_stream_destroy(stream);
        require(launches > 0,"zero behavioral launches");
        std::printf("ALL_PASS scored_tree_launches=%zu capacities=2..8 max_rounds=3,5 policies=0,1,2\n",launches);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
