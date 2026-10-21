// Public-operation regressions shared by the host and real-device executables.
// The GPU target compiles this file with RAD_MTP_PRUNED_GPU_TEST and its own main.
#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "abi/rad_plugin.h"
#include "arch/common/rad_mtp_tree_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int capacity = 4, vocab = 512, meta_cols = 85, score_cols = 41, packed_cols = 17;
constexpr size_t guard_bytes = 64;
constexpr uint32_t sentinel = 0x5a5a5a5au;

void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error(why);
}
void check(int rc, const std::string& why) {
    require(rc == RAD_OK, why + ": " + rad_strerror(rc));
}
void near(double actual, double expected, const std::string& why, double tolerance = 0.00003) {
    if (std::isinf(expected)) require(actual == expected, why + ": infinity mismatch");
    else require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)),
                 why + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
}

// Framed, aligned tensor storage. Only logical output elements may change; input bytes,
// outer-row padding, and both frames are checked after every launch, including refusals.
struct Tensor {
    RadTensor desc{};
    std::vector<uint32_t> storage;
    std::vector<uint8_t> logical;
    size_t width = 0;

    Tensor() = default; // An absent operand is a present descriptor with null data.
    Tensor(uint32_t dtype, std::initializer_list<int64_t> shape,
           std::initializer_list<int64_t> strides = {}) {
        desc.dtype = dtype;
        desc.rank = static_cast<uint32_t>(shape.size());
        std::copy(shape.begin(), shape.end(), desc.shape);
        if (strides.size()) {
            require(strides.size() == shape.size(), "fixture stride rank");
            std::copy(strides.begin(), strides.end(), desc.stride);
        } else {
            int64_t stride = 1;
            for (int d = static_cast<int>(desc.rank) - 1; d >= 0; --d) {
                desc.stride[d] = stride;
                stride *= desc.shape[d];
            }
        }
        width = dtype == RAD_BF16 ? 2 : dtype == RAD_F8E4M3 ? 1 : 4;
        int64_t extent = 1, count = 1;
        for (uint32_t d = 0; d < desc.rank; ++d) {
            extent += (desc.shape[d] - 1) * desc.stride[d];
            count *= desc.shape[d];
        }
        const size_t bytes = 2 * guard_bytes + static_cast<size_t>(extent) * width;
        storage.assign((bytes + 3) / 4, sentinel);
        logical.assign(storage.size() * 4, 0);
        for (int64_t i = 0; i < count; ++i) {
            int64_t n = i, offset = 0;
            for (int d = static_cast<int>(desc.rank) - 1; d >= 0; --d) {
                offset += (n % desc.shape[d]) * desc.stride[d];
                n /= desc.shape[d];
            }
            std::fill_n(logical.begin() + guard_bytes + static_cast<size_t>(offset) * width, width, 1);
        }
    }
    void* body() { return storage.empty() ? nullptr : reinterpret_cast<uint8_t*>(storage.data()) + guard_bytes; }
    const void* body() const { return storage.empty() ? nullptr : reinterpret_cast<const uint8_t*>(storage.data()) + guard_bytes; }
    template<class T> T& at(int64_t offset) { return static_cast<T*>(body())[offset]; }
    template<class T> T at(int64_t offset) const { return static_cast<const T*>(body())[offset]; }
    int64_t offset(int64_t a, int64_t b = 0, int64_t c = 0, int64_t d = 0) const {
        return a * desc.stride[0] + b * desc.stride[1] + c * desc.stride[2] + d * desc.stride[3];
    }
    void i(int64_t row, int64_t col, int value) { at<int32_t>(offset(row, col)) = value; }
    int i(int64_t row, int64_t col = 0) const { return at<int32_t>(offset(row, col)); }
    void f(int64_t row, int64_t col, float value) { at<float>(offset(row, col)) = value; }
    float f(int64_t row, int64_t col = 0) const { return at<float>(offset(row, col)); }
};

struct DeviceStorage {
    void* p = nullptr;
    explicit DeviceStorage(size_t bytes) {
        if (bytes) {
            p = rad_dev_alloc(static_cast<int64_t>(bytes), RAD_MEM_DEVICE);
            require(p != nullptr, "device tensor allocation");
        }
    }
    ~DeviceStorage() { if (p) rad_dev_free(p, RAD_MEM_DEVICE); }
    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;
};

struct Runner {
    rad::Registry& registry;
    RadStream stream;
    bool device;
    size_t launches = 0, cases = 0;

    int invoke(const char* op, const rad::Geometry& geometry, std::vector<Tensor*> operands,
               std::initializer_list<int> outputs,
               const std::function<void(std::vector<RadTensor>&)>& malformed = {},
               const rad::Geometry* launch_geometry = nullptr) {
        rad::SelectionTrace trace;
        auto row = registry.resolve(op, geometry, device ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST, &trace);
        require(row.row && row.row->info, std::string("missing ") + op + ": " + trace.reason);
        const bool gather = std::string(op) == "gather_rows", enter = std::string(op) == "mtp_enter";
        const bool conventional = gather || enter;
        const std::string plugin = conventional ? (device ? "libr4d" : "libref") : (device ? "mine" : "mine_ref");
        const std::string name = conventional ? (device ? (gather ? "gather_rows_bf16" : "mtp_enter_bf16") : std::string("ref_") + op)
                                              : std::string(device ? "mine_" : "ref_") + op;
        require(row.row->plugin == plugin && row.row->info->name == name,
                std::string("wrong execution row for ") + op + ": " + row.row->plugin);
        check(registry.init_instance(row, 0, 2), std::string("initialize ") + op);
        std::vector<RadTensor> tensors;
        std::vector<std::vector<uint32_t>> before;
        std::vector<std::unique_ptr<DeviceStorage>> buffers;
        for (Tensor* tensor : operands) {
            before.push_back(tensor->storage);
            RadTensor t = tensor->desc;
            if (device) {
                buffers.emplace_back(new DeviceStorage(tensor->storage.size() * 4));
                if (buffers.back()->p) {
                    check(rad_memcpy_async(buffers.back()->p, tensor->storage.data(),
                                           static_cast<int64_t>(tensor->storage.size() * 4), stream), "upload framed tensor");
                    t.data = static_cast<uint8_t*>(buffers.back()->p) + guard_bytes;
                }
            } else t.data = tensor->body();
            tensors.push_back(t);
        }
        if (malformed) malformed(tensors);
        const auto& g = launch_geometry ? *launch_geometry : geometry;
        RadArgs args{};
        args.t = tensors.data(); args.n_t = static_cast<int>(tensors.size());
        args.p = g.params(); args.n_p = g.n_params(); args.instance = row.instance;
        args.rank = 0; args.world_size = 2;
        const int64_t scratch_bytes = row.row->info->scratch ? row.row->info->scratch(&args) : 0;
        require(scratch_bytes >= 0 && (enter || scratch_bytes == 0), "unexpected operation scratch");
        std::unique_ptr<DeviceStorage> device_scratch;
        std::vector<uint32_t> host_scratch;
        if (scratch_bytes) {
            if (device) {
                device_scratch.reset(new DeviceStorage(static_cast<size_t>(scratch_bytes)));
                args.scratch = device_scratch->p;
            } else {
                host_scratch.resize(static_cast<size_t>((scratch_bytes + 3) / 4));
                args.scratch = host_scratch.data();
            }
            args.scratch_bytes = scratch_bytes;
        }
        const int rc = row.row->info->launch(&args, device ? stream : nullptr);
        ++launches;
        if (device) {
            for (size_t i = 0; i < operands.size(); ++i) if (buffers[i]->p)
                check(rad_memcpy_async(operands[i]->storage.data(), buffers[i]->p,
                                       static_cast<int64_t>(operands[i]->storage.size() * 4), stream), "download framed tensor");
            check(rad_stream_sync(stream), std::string("synchronize ") + op);
        }
        for (size_t i = 0; i < operands.size(); ++i) {
            const bool output = rc == RAD_OK && std::find(outputs.begin(), outputs.end(), static_cast<int>(i)) != outputs.end();
            const auto* original = reinterpret_cast<const uint8_t*>(before[i].data());
            const auto* actual = reinterpret_cast<const uint8_t*>(operands[i]->storage.data());
            for (size_t b = 0; b < before[i].size() * 4; ++b)
                require((output && operands[i]->logical[b]) || original[b] == actual[b],
                        std::string(op) + ": input/padding/frame mutation at operand " + std::to_string(i) + " byte " + std::to_string(b));
        }
        return rc;
    }
};

rad::Geometry selector_geometry(int round) {
    rad::Geometry g; g.set_i("M", capacity); g.set_i("round", round); g.set_i("n_vocab", vocab); return g;
}
rad::Geometry pack_geometry(double beta) {
    rad::Geometry g; g.set_i("M", capacity); g.set_f("beta", beta); return g;
}

struct Node { int token = -1, parent = -1; double score = -std::numeric_limits<double>::infinity(); };
using Tree = std::array<Node, 41>;
struct Scene {
    Tensor absent, meta{RAD_I32, {capacity, meta_cols}, {92, 1}};
    Tensor scores{RAD_F32, {capacity, score_cols}, {48, 1}};
    Tensor packed{RAD_I32, {capacity, packed_cols}, {24, 1}};
    Tensor tokens{RAD_I32, {6}}, positions{RAD_I32, {6}}, anchor_rows{RAD_I32, {capacity}};
    std::array<Tree, capacity> expected{};
    Scene() {
        const int anchors[] = {4, 1, 5, 2};
        for (int i = 0; i < 6; ++i) { tokens.i(i, 0, 400 + i); positions.i(i, 0, 3 + i); }
        for (int s = 0; s < capacity; ++s) {
            anchor_rows.i(s, 0, anchors[s]);
            expected[s][0] = {tokens.i(anchors[s]), -1, 0};
        }
    }
    void known_tree() {
        for (int s = 0; s < capacity; ++s) {
            meta.i(s, 0, 1); meta.i(s, 1, positions.i(anchor_rows.i(s))); meta.i(s, 2, expected[s][0].token);
            for (int n = 0; n <= 40; ++n) {
                const int depth = n ? (n - 1) / 8 + 1 : 0;
                const int parent = depth <= 1 ? 0 : 1 + (depth - 2) * 8 + (n - 1) % 8;
                // Reverse token order makes token-ID tie breaking disagree with old-ID order.
                if (n) expected[s][n] = {200 - n, parent, -10.0 * depth};
                meta.i(s, 3 + n, expected[s][n].token);
                meta.i(s, 44 + n, n ? parent : -1);
                scores.f(s, n, static_cast<float>(expected[s][n].score));
            }
        }
    }
    void check_level(int s, int round) const {
        require(meta.i(s) == 1, "valid frontier was invalidated");
        require(meta.i(s, 1) == positions.i(anchor_rows.i(s)), "anchor position mapping");
        require(meta.i(s, 2) == expected[s][0].token, "anchor token mapping");
        for (int n = 0; n <= round * 8; ++n) {
            require(meta.i(s, 3 + n) == expected[s][n].token, "frontier token at node " + std::to_string(n));
            require(meta.i(s, 44 + n) == expected[s][n].parent, "frontier parent at node " + std::to_string(n));
            near(scores.f(s, n), expected[s][n].score, "candidate-set path score");
        }
    }
};

void offer(Tensor& gathered, int row, int pair, int token, float logit) {
    const int64_t offset = gathered.offset(row, pair, 0);
    gathered.at<int32_t>(offset) = token;
    gathered.at<float>(offset + 1) = logit;
}
Tensor candidates(int rows, int round, bool ties = false) {
    Tensor out(RAD_F32, {rows, 16, 2}, {40, 2, 1});
    for (int row = 0; row < rows; ++row) for (int pair = 0; pair < 16; ++pair) {
        // Both TP ranks contribute to the winning eight, in neither rank nor token order.
        const int rank = (pair * 5 + 3) % 16;
        const int token = round * 64 + rank;
        float value = -0.15f * rank;
        if (round > 1) value = row % 8 == 0 ? (rank < 8 ? 0.0f : -50.0f)
                                                          : (rank == 0 ? 20.0f : -20.0f - rank);
        offer(out, row, pair, token, ties ? 0.0f : value);
    }
    return out;
}

// Exhaustive edge enumeration in long double, separate from production helpers and
// metadata. Only the eight merged candidates form the probability denominator.
void extend_expected(Scene& scene, const Tensor& gathered, const std::vector<int>& ids, int round) {
    struct Edge { int token, parent, rank; double score; };
    for (size_t q = 0; q < ids.size(); ++q) {
        const int s = ids[q], parents = round == 1 ? 1 : 8;
        std::vector<Edge> edges;
        for (int j = 0; j < parents; ++j) {
            std::vector<std::pair<float, int>> offered;
            const int row = static_cast<int>(q) * parents + j;
            for (int k = 0; k < 16; ++k) {
                const int64_t off = gathered.offset(row, k, 0);
                offered.emplace_back(gathered.at<float>(off + 1), gathered.at<int32_t>(off));
            }
            std::sort(offered.begin(), offered.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            const long double maximum = offered.front().first;
            long double denominator = 0;
            for (int k = 0; k < 8; ++k) denominator += std::exp(static_cast<long double>(offered[k].first) - maximum);
            const int parent = round == 1 ? 0 : 1 + (round - 2) * 8 + j;
            for (int k = 0; k < 8; ++k) {
                const long double local = static_cast<long double>(offered[k].first) - maximum - std::log(denominator);
                const float path = static_cast<float>(scene.expected[s][parent].score + local);
                edges.push_back({offered[k].second, parent, k, path});
            }
        }
        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.parent != b.parent) return a.parent < b.parent;
            return a.rank < b.rank;
        });
        for (int k = 0; k < 8; ++k) scene.expected[s][1 + (round - 1) * 8 + k] = {edges[k].token, edges[k].parent, edges[k].score};
    }
}
Tensor request_ids(const std::vector<int>& ids) {
    Tensor t(RAD_I32, {static_cast<int64_t>(ids.size())});
    for (size_t i = 0; i < ids.size(); ++i) t.i(static_cast<int64_t>(i), 0, ids[i]);
    return t;
}
void expand(Runner& runner, Scene& scene, Tensor& gathered, Tensor& ids, int round) {
    check(runner.invoke("mtp_pruned_expand", selector_geometry(round),
                        {&gathered, round == 1 ? &scene.absent : &ids,
                         round == 1 ? &scene.tokens : &scene.absent,
                         round == 1 ? &scene.anchor_rows : &scene.absent,
                         round == 1 ? &scene.positions : &scene.absent,
                         &scene.meta, &scene.scores, &scene.packed}, {5, 6, 7}), "expand");
}
struct Prepared {
    Tensor tokens, positions, gather, ancestors, slots, prefix;
    explicit Prepared(int rows) : tokens(RAD_I32, {rows}), positions(RAD_I32, {rows}), gather(RAD_I32, {rows}),
                                 ancestors(RAD_I32, {rows, 4}, {7, 1}), slots(RAD_I32, {rows}), prefix(RAD_I32, {rows}) {}
    std::vector<Tensor*> bind(Tensor& ids, Tensor& meta) {
        return {&ids, &meta, &tokens, &positions, &gather, &ancestors, &slots, &prefix};
    }
};
void prepare(Runner& runner, Scene& scene, Tensor& ids, const std::vector<int>& mapping, int round, Prepared& out) {
    check(runner.invoke("mtp_pruned_prepare", selector_geometry(round), out.bind(ids, scene.meta), {2, 3, 4, 5, 6, 7}), "prepare");
    const int depth = round - 1;
    for (size_t q = 0; q < mapping.size(); ++q) for (int j = 0; j < 8; ++j) {
        const int row = static_cast<int>(q) * 8 + j, node = 1 + (depth - 1) * 8 + j, s = mapping[q];
        const auto& tree = scene.expected[s];
        require(out.tokens.i(row) == tree[node].token, "prepare frontier token");
        require(out.positions.i(row) == scene.positions.i(scene.anchor_rows.i(s)) + depth, "prepare absolute RoPE position");
        require(out.prefix.i(row) == scene.positions.i(scene.anchor_rows.i(s)) + 1, "prepare committed prefix");
        require(out.slots.i(row) == static_cast<int>(q) * 32 + node - 1, "tile-local provisional slot");
        const int gather = round == 2 ? s : static_cast<int>(q) * 8 + tree[node].parent - (1 + (depth - 2) * 8);
        require(out.gather.i(row) == gather, "prepare hidden parent index");
        std::vector<int> path;
        for (int n = node; n; n = tree[n].parent) path.push_back(n);
        std::reverse(path.begin(), path.end());
        for (int d = 0; d < 4; ++d)
            require(out.ancestors.i(row, d) == (d < depth ? static_cast<int>(q) * 32 + path[d] - 1 : -1), "chronological ancestor slots");
    }
}

void expect_packed(const Scene& scene, int s, const std::vector<int>& selected) {
    require(selected.size() == 7 && scene.packed.i(s) == 8, "exact root-plus-seven budget");
    require(scene.packed.i(s, 1) == scene.expected[s][0].token && scene.packed.i(s, 9) == -1, "packed root sentinel");
    for (int k = 0; k < 7; ++k) {
        const int node = selected[k], parent = scene.expected[s][node].parent;
        const auto found = std::find(selected.begin(), selected.end(), parent);
        const int parent_index = parent == 0 ? 0 : static_cast<int>(found - selected.begin()) + 1;
        require(parent == 0 || (found != selected.end() && parent_index <= k), "oracle selection must be ancestor closed");
        require(scene.packed.i(s, 2 + k) == scene.expected[s][node].token, "packed selected token");
        require(scene.packed.i(s, 10 + k) == parent_index, "packed parent remapping");
    }
}
void pack(Runner& runner, Scene& scene, Tensor& ids, double beta) {
    check(runner.invoke("mtp_pruned_pack", pack_geometry(beta), {&ids, &scene.meta, &scene.scores, &scene.packed}, {3}), "pack");
}
std::vector<int> best_seven(const Tree& tree, double beta) {
    std::vector<int> ranked;
    for (int n = 1; n <= 40; ++n) ranked.push_back(n);
    std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
        const double x = tree[a].score + beta * ((a - 1) / 8 + 1);
        const double y = tree[b].score + beta * ((b - 1) / 8 + 1);
        return x != y ? x > y : a < b;
    });
    ranked.resize(7);
    std::sort(ranked.begin(), ranked.end());
    return ranked;
}

float load_kv(const Tensor& tensor, int page, int head, int token, int column) {
    const int64_t offset = tensor.offset(page, head, token, column);
    return tensor.desc.dtype == RAD_BF16 ? rad_bf16_to_f32(tensor.at<uint16_t>(offset))
                                       : rad_fp8e4m3_to_f32(tensor.at<uint8_t>(offset));
}
void store_kv(Tensor& tensor, int page, int head, int token, int column, float value) {
    const int64_t offset = tensor.offset(page, head, token, column);
    if (tensor.desc.dtype == RAD_BF16) tensor.at<uint16_t>(offset) = rad_f32_to_bf16(value);
    else tensor.at<uint8_t>(offset) = rad_f32_to_fp8e4m3(value);
}
struct Overlay {
    int requests, dim, heads, kv_heads, depth;
    uint32_t dtype;
    Tensor committed, table, provisional, out;
    Overlay(int q, int d, int h, int hk, int level, uint32_t dt)
        : requests(q), dim(d), heads(h), kv_heads(hk), depth(level), dtype(dt),
          committed(dt, {q * 4, hk, 4, 2 * d}, {hk * (8 * d + 12) + 8, 8 * d + 12, 2 * d, 1}),
          table(RAD_I32, {q, 4}, {7, 1}),
          provisional(dt, {8 * q, hk, 4, 2 * d}, {hk * (8 * d + 12) + 8, 8 * d + 12, 2 * d, 1}),
          out(RAD_BF16, {8 * q, h * d}, {h * d + 8, 1}) {
        for (int s = 0; s < q; ++s) for (int b = 0; b < 4; ++b) table.i(s, b, s * 4 + (3 - b));
        for (int p = 0; p < q * 4; ++p) for (int hkv = 0; hkv < hk; ++hkv)
            for (int t = 0; t < 4; ++t) for (int x = 0; x < d; ++x) {
                store_kv(committed, p, hkv, t, x, float((p * 3 + t * 5 + x + hkv) % 11 - 5) / 8);
                store_kv(committed, p, hkv, t, d + x, float((p * 7 + t * 3 + x * 2 + hkv) % 17 - 8) / 4);
            }
        for (int slot = 0; slot < q * 32; ++slot) for (int hkv = 0; hkv < hk; ++hkv)
            for (int x = 0; x < d; ++x) {
                store_kv(provisional, slot / 4, hkv, slot % 4, x, float((slot * 7 + x * 3 + hkv) % 13 - 6) / 8);
                store_kv(provisional, slot / 4, hkv, slot % 4, d + x, float((slot * 5 + x + hkv) % 19 - 9) / 4);
            }
    }
    rad::Geometry geometry() const {
        rad::Geometry g; g.set_i("M", requests * 8 + 8); g.set_i("N", capacity);
        g.set_i("head_dim", dim); g.set_i("n_head", heads); g.set_i("n_head_kv", kv_heads);
        g.set_i("block_size", 4); g.set_i("depth", depth); g.set_i("max_ctx", 16);
        g.set_s("q_dtype", "bf16"); g.set_s("kv_dtype", dtype == RAD_BF16 ? "bf16" : "fp8_e4m3"); return g;
    }
    std::vector<Tensor*> bind(Tensor& q, Tensor& prefix, Tensor& ancestors, Tensor& meta, Tensor& ids) {
        return {&q, &committed, &table, &prefix, &provisional, &ancestors, &out, &meta, &ids};
    }
    // Materialize each branch as its own chronological dense sequence. This oracle
    // neither calls the host row nor reads target tree masks or production helpers.
    std::vector<double> oracle(const Tensor& q, const Tensor& prefix, const Tensor& ancestors) const {
        std::vector<double> result(static_cast<size_t>(requests * 8 * heads * dim));
        for (int row = 0; row < requests * 8; ++row) for (int h = 0; h < heads; ++h) {
            const int hk = h / (heads / kv_heads), committed_length = prefix.i(row);
            std::vector<double> logits, values;
            for (int t = 0; t < committed_length + depth; ++t) {
                const bool base = t < committed_length;
                const int slot = base ? t : ancestors.i(row, t - committed_length);
                const int page = base ? table.i(row / 8, slot / 4) : slot / 4;
                const Tensor& kv = base ? committed : provisional;
                double dot = 0;
                for (int x = 0; x < dim; ++x) {
                    dot += static_cast<double>(rad_bf16_to_f32(q.at<uint16_t>(q.offset(row, h * dim + x)))) * load_kv(kv, page, hk, slot % 4, x);
                    values.push_back(load_kv(kv, page, hk, slot % 4, dim + x));
                }
                logits.push_back(dot / std::sqrt(static_cast<double>(dim)));
            }
            const double maximum = *std::max_element(logits.begin(), logits.end());
            double denominator = 0;
            for (double logit : logits) denominator += std::exp(logit - maximum);
            for (int x = 0; x < dim; ++x) {
                double value = 0;
                for (size_t t = 0; t < logits.size(); ++t) value += std::exp(logits[t] - maximum) * values[t * dim + x] / denominator;
                result[static_cast<size_t>((row * heads + h) * dim + x)] = value;
            }
        }
        return result;
    }
    void run(Runner& runner, Tensor& q, Tensor& prefix, Tensor& ancestors, Tensor& meta, Tensor& ids) {
        check(runner.invoke("mtp_overlay_attn", geometry(), bind(q, prefix, ancestors, meta, ids), {6, 7}), "overlay attention");
    }
    void compare(const std::vector<double>& expected) const {
        for (int row = 0; row < requests * 8; ++row) for (int x = 0; x < heads * dim; ++x) {
            const double rounded = rad_bf16_to_f32(rad_f32_to_bf16(static_cast<float>(expected[static_cast<size_t>(row * heads * dim + x)])));
            // A BF16 ulp plus FP32 reduction error, not target-attention bit parity.
            near(rad_bf16_to_f32(out.at<uint16_t>(out.offset(row, x))), rounded, "independent branch attention", 0.008);
        }
    }
    std::vector<uint16_t> output() const {
        std::vector<uint16_t> result;
        for (int r = 0; r < requests * 8; ++r) for (int x = 0; x < heads * dim; ++x)
            result.push_back(out.at<uint16_t>(out.offset(r, x)));
        return result;
    }
};

Tensor hidden_bank(int rows, int width) {
    Tensor hidden(RAD_BF16, {rows, width}, {width + 8, 1});
    for (int r = 0; r < rows; ++r) for (int x = 0; x < width; ++x)
        hidden.at<uint16_t>(hidden.offset(r, x)) = rad_f32_to_bf16(float(((r + 1) * (x % 5 + 1) * 3 + x) % 29 - 14) / 8);
    return hidden;
}
void gather(Runner& runner, Tensor& bank, Tensor& ids, Tensor& out) {
    rad::Geometry g; g.set_i("M", out.desc.shape[0] + 8); g.set_i("n", out.desc.shape[1]); g.set_s("dtype", "bf16");
    check(runner.invoke("gather_rows", g, {&bank, &ids, &out}, {2}), "actual hidden carry gather");
}
struct MtpEntry {
    static constexpr int n = 128, hc = 2, width = n * hc;
    Tensor embedding, hidden_gain{RAD_F32, {width}}, embedding_gain{RAD_F32, {n}};
    Tensor hidden_fc{RAD_BF16, {n, n}}, embedding_fc{RAD_BF16, {n, n}}, out;
    explicit MtpEntry(const Tensor& token_ids)
        : embedding(RAD_BF16, {token_ids.desc.shape[0], n}, {n + 8, 1}),
          out(RAD_BF16, {token_ids.desc.shape[0], width}, {width + 8, 1}) {
        for (int i = 0; i < width; ++i) hidden_gain.at<float>(i) = 1 + float(i % 3) / 8;
        for (int i = 0; i < n; ++i) {
            embedding_gain.at<float>(i) = 0.875f + float(i % 2) / 4;
            for (int j = 0; j < n; ++j) {
                hidden_fc.at<uint16_t>(hidden_fc.offset(i, j)) = rad_f32_to_bf16(i == j ? 1 : j == (i + 1) % n ? -0.25f : 0);
                embedding_fc.at<uint16_t>(embedding_fc.offset(i, j)) = rad_f32_to_bf16(j == (3 * i + 1) % n ? 0.5f : 0);
            }
        }
        for (int r = 0; r < token_ids.desc.shape[0]; ++r) for (int i = 0; i < n; ++i)
            embedding.at<uint16_t>(embedding.offset(r, i)) = rad_f32_to_bf16(float((token_ids.i(r) * 3 + i * 5) % 23 - 11) / 8);
    }
    void run(Runner& runner, Tensor& hidden) {
        rad::Geometry g;
        g.set_i("M", hidden.desc.shape[0]); g.set_i("n", n); g.set_i("hc", hc); g.set_i("ngroup", hc);
        g.set_f("eps", 0.000001); g.set_f("wadd", 0); g.set_s("dtype", "bf16"); g.set_s("fc", "bf16");
        check(runner.invoke("mtp_enter", g, {&hidden, &embedding, &hidden_gain, &embedding_gain,
                                            &hidden_fc, &embedding_fc, &out}, {6}), "actual MTP entry");
    }
    Tensor oracle(const Tensor& hidden) const {
        Tensor expected(RAD_BF16, {hidden.desc.shape[0], width}, {width + 8, 1});
        auto bf = [](float x) { return rad_bf16_to_f32(rad_f32_to_bf16(x)); };
        for (int r = 0; r < hidden.desc.shape[0]; ++r) {
            double embedding_ss = 0;
            for (int i = 0; i < n; ++i) {
                const double v = rad_bf16_to_f32(embedding.at<uint16_t>(embedding.offset(r, i)));
                embedding_ss += v * v;
            }
            const float es = 1 / std::sqrt(static_cast<float>(embedding_ss / n) + 0.000001f);
            for (int c = 0; c < hc; ++c) {
                double hidden_ss = 0;
                for (int i = 0; i < n; ++i) {
                    const double v = rad_bf16_to_f32(hidden.at<uint16_t>(hidden.offset(r, c * n + i)));
                    hidden_ss += v * v;
                }
                const float hs = 1 / std::sqrt(static_cast<float>(hidden_ss / n) + 0.000001f);
                auto normalized = [&](int i) {
                    return bf(rad_bf16_to_f32(hidden.at<uint16_t>(hidden.offset(r, c * n + i))) * hs * hidden_gain.at<float>(c * n + i));
                };
                for (int i = 0; i < n; ++i) {
                    const int j = (3 * i + 1) % n;
                    const float embedded = bf(rad_bf16_to_f32(embedding.at<uint16_t>(embedding.offset(r, j))) * es * embedding_gain.at<float>(j));
                    // Closed form for the constructed two-tap/permutation matrices.
                    const float value = normalized(i) - 0.25f * normalized((i + 1) % n) + 0.5f * embedded;
                    expected.at<uint16_t>(expected.offset(r, c * n + i)) = rad_f32_to_bf16(value);
                }
            }
        }
        return expected;
    }
};

void hidden_attention_composition(Runner& runner, Scene& scene, Tensor& ids,
                                  const std::vector<int>& mapping, Prepared& prepared, int round) {
    constexpr int width = MtpEntry::width;
    const int rows = static_cast<int>(mapping.size()) * 8;
    Tensor bank = hidden_bank(round == 2 ? capacity : rows, width);
    Tensor hidden(RAD_BF16, {rows, width}, {width + 8, 1});
    gather(runner, bank, prepared.gather, hidden);
    Tensor expected_hidden = hidden;
    bool differs_from_same_row = false;
    for (int r = 0; r < rows; ++r) {
        const int s = mapping[static_cast<size_t>(r / 8)], depth = round - 1;
        const int node = 1 + (depth - 1) * 8 + r % 8;
        const int parent_row = round == 2 ? s : (r / 8) * 8 + scene.expected[s][node].parent - (1 + (depth - 2) * 8);
        differs_from_same_row |= parent_row != r;
        for (int x = 0; x < width; ++x) {
            const uint16_t expected = bank.at<uint16_t>(bank.offset(parent_row, x));
            require(hidden.at<uint16_t>(hidden.offset(r, x)) == expected, "gather_rows selected wrong hidden parent");
            expected_hidden.at<uint16_t>(expected_hidden.offset(r, x)) = expected;
        }
    }
    require(differs_from_same_row, "hidden fixture cannot detect same-row gather substitution");
    MtpEntry entry(prepared.tokens);
    const Tensor expected_q = entry.oracle(expected_hidden);
    entry.run(runner, hidden);
    for (int r = 0; r < rows; ++r) for (int x = 0; x < width; ++x)
        near(rad_bf16_to_f32(entry.out.at<uint16_t>(entry.out.offset(r, x))),
             rad_bf16_to_f32(expected_q.at<uint16_t>(expected_q.offset(r, x))), "independent MTP entry", 0.008);
    Overlay overlay(static_cast<int>(mapping.size()), 128, 2, 1, round - 1, RAD_BF16);
    const auto oracle = overlay.oracle(expected_q, prepared.prefix, prepared.ancestors);
    overlay.run(runner, entry.out, prepared.prefix, prepared.ancestors, scene.meta, ids);
    overlay.compare(oracle);
    const auto correct = overlay.output();
    const auto clean_kv = overlay.provisional.storage;
    const int sibling = prepared.ancestors.i(1, round - 2);
    for (int x = 0; x < 128; ++x) store_kv(overlay.provisional, sibling / 4, 0, sibling % 4, 128 + x, 32);
    overlay.run(runner, entry.out, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto sibling_output = overlay.output();
    require(std::equal(correct.begin(), correct.begin() + width, sibling_output.begin()), "real MTP attention leaked sibling KV");
    require(!std::equal(correct.begin() + width, correct.begin() + 2 * width, sibling_output.begin() + width), "real MTP attention ignored visible own KV");
    overlay.compare(overlay.oracle(expected_q, prepared.prefix, prepared.ancestors));
    overlay.provisional.storage = clean_kv;
    const int ancestor = prepared.ancestors.i(0, 0);
    for (int x = 0; x < 128; ++x) store_kv(overlay.provisional, ancestor / 4, 0, ancestor % 4, 128 + x, -32);
    overlay.run(runner, entry.out, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto ancestor_output = overlay.output();
    require(!std::equal(correct.begin(), correct.begin() + width, ancestor_output.begin()), "real MTP attention ignored ancestor KV");
    overlay.compare(overlay.oracle(expected_q, prepared.prefix, prepared.ancestors));
    overlay.provisional.storage = clean_kv;
    const int original = prepared.gather.i(0);
    // A non-spine parent specifically challenges the old greedy-first carry path.
    const int wrong_parent = round >= 3 && original % 8 ? (original / 8) * 8
                                                        : (original + 1) % static_cast<int>(bank.desc.shape[0]);
    prepared.gather.i(0, 0, wrong_parent);
    gather(runner, bank, prepared.gather, hidden);
    entry.run(runner, hidden);
    overlay.run(runner, entry.out, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto wrong_parent_output = overlay.output();
    require(!std::equal(correct.begin(), correct.begin() + width, wrong_parent_output.begin()),
            "greedy-spine/wrong-parent hidden carry did not reach attention output");
    prepared.gather.i(0, 0, original);
    ++runner.cases;
}

void frontier_and_tiles(Runner& runner, bool ties) {
    Scene scene;
    const std::vector<int> all{0, 1, 2, 3};
    Tensor ids = request_ids(all), root = candidates(capacity, 1, ties);
    extend_expected(scene, root, all, 1);
    expand(runner, scene, root, ids, 1);
    for (int s : all) {
        scene.check_level(s, 1);
        require(scene.packed.i(s) == 0, "root expansion must not publish a partial tree");
        if (ties) for (int n = 1; n <= 8; ++n) {
            require(scene.meta.i(s, 3 + n) == 63 + n, "merged token-ID ascending tie");
            near(scene.scores.f(s, n), -std::log(8.0), "top8-only normalization, not top16");
        }
    }
    for (const std::vector<int>& tile : {std::vector<int>{3, 1}, std::vector<int>{0, 2}}) {
        Tensor tile_ids = request_ids(tile);
        for (int round = 2; round <= 5; ++round) {
            Prepared prepared(16);
            prepare(runner, scene, tile_ids, tile, round, prepared);
            hidden_attention_composition(runner, scene, tile_ids, tile, prepared, round);
            Tensor offered = candidates(16, round, ties);
            extend_expected(scene, offered, tile, round);
            expand(runner, scene, offered, tile_ids, round);
            for (int s : tile) scene.check_level(s, round);
            if (round == 2) {
                for (int s : tile) require(scene.expected[s][9].parent == (ties ? 1 : 2), "fixture must distinguish non-spine expansion");
                if (ties) for (int s : tile) for (int n = 9; n <= 16; ++n)
                    require(scene.meta.i(s, 44 + n) == 1, "frontier ties must prefer parent then candidate rank");
            }
        }
        const auto before = scene.packed.storage;
        pack(runner, scene, tile_ids, -0.2);
        for (int s : tile) expect_packed(scene, s, best_seven(scene.expected[s], -0.2));
        for (int s : all) if (std::find(tile.begin(), tile.end(), s) == tile.end())
            for (int c = 0; c < packed_cols; ++c) {
                const size_t word = guard_bytes / 4 + static_cast<size_t>(scene.packed.offset(s, c));
                require(scene.packed.storage[word] == before[word], "pack modified another tile's request");
            }
    }
    ++runner.cases;
}

void normalization_and_refusals(Runner& runner) {
    const std::vector<int> all{0, 1, 2, 3};
    Tensor ids = request_ids(all);
    Scene a, b;
    Tensor first = candidates(capacity, 1), second = first;
    for (int r = 0; r < capacity; ++r) for (int pair = 0; pair < 16; ++pair) {
        const int64_t off = second.offset(r, pair, 0);
        if (second.at<int32_t>(off) >= 72) second.at<float>(off + 1) = -10000;
    }
    expand(runner, a, first, ids, 1); expand(runner, b, second, ids, 1);
    require(a.scores.storage == b.scores.storage, "discarded candidates changed normalized scores");
    require(a.meta.storage == b.meta.storage, "discarded candidates changed selected nodes");
    for (int kind = 0; kind < 7; ++kind) {
        Scene scene;
        Tensor offered = candidates(capacity, 1);
        const int64_t off = offered.offset(1, 15, 0);
        if (kind == 0) offered.at<int32_t>(off) = -1;
        if (kind == 1) offered.at<int32_t>(off) = vocab;
        if (kind == 2) offered.at<int32_t>(off) = offered.at<int32_t>(offered.offset(1, 0, 0));
        if (kind == 3) offered.at<float>(off + 1) = std::numeric_limits<float>::quiet_NaN();
        if (kind == 4) offered.at<float>(off + 1) = std::numeric_limits<float>::infinity();
        if (kind == 5) offered.at<float>(off + 1) = -std::numeric_limits<float>::infinity();
        if (kind == 6) scene.anchor_rows.i(1, 0, 6);
        expand(runner, scene, offered, ids, 1);
        require(scene.meta.i(1) == 0 && scene.packed.i(1) == 0, "bad offered candidate/anchor must invalidate request");
        require(scene.meta.i(0) == 1 && scene.meta.i(2) == 1, "bad metadata leaked across requests");
    }
    // Finite logits may underflow path storage to -inf; this is not bad input.
    Scene extreme;
    Tensor offered = candidates(capacity, 1);
    for (int r = 0; r < capacity; ++r) for (int pair = 0; pair < 16; ++pair)
        offered.at<float>(offered.offset(r, pair, 0) + 1) = pair == 0 ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    expand(runner, extreme, offered, ids, 1);
    require(extreme.meta.i(0) == 1 && extreme.scores.f(0, 1) == 0, "finite extreme winner invalid");
    require(extreme.scores.f(0, 2) == -std::numeric_limits<float>::infinity(), "finite extreme path must retain -inf");
    for (int kind = 0; kind < 9; ++kind) {
        Scene scene;
        Tensor input = candidates(capacity, 1);
        std::vector<Tensor*> operands{&input, &scene.absent, &scene.tokens, &scene.anchor_rows,
                                      &scene.positions, &scene.meta, &scene.scores, &scene.packed};
        const int rc = runner.invoke("mtp_pruned_expand", selector_geometry(1), operands, {5, 6, 7}, [&](auto& t) {
            if (kind == 0) t[0].dtype = RAD_I32;
            if (kind == 1) t[0].shape[1] = 15;
            if (kind == 2) t[0].stride[1] = 3;
            if (kind == 3) t[5].shape[1] = 84;
            if (kind == 4) t[6].stride[1] = 2;
            if (kind == 5) t[7].data = t[5].data;
            if (kind == 6) t[1] = t[3]; // Root request_ids must be absent.
            if (kind == 7) t[4].shape[0] = 5;
            if (kind == 8) t[5].data = static_cast<uint8_t*>(t[5].data) + 1;
        });
        require(rc != RAD_OK, "malformed expand descriptors accepted");
    }
    ++runner.cases;
}

void pack_ties_and_invalid_trees(Runner& runner) {
    Tensor ids = request_ids({3, 1});
    for (double beta : {0.0, -0.2}) {
        Scene scene; scene.known_tree();
        for (int s = 0; s < capacity; ++s) {
            for (int n = 1; n <= 6; ++n) scene.expected[s][n].score = 0;
            scene.expected[s][7].score = -0.1f;
            for (int n : {9, 17, 25, 33}) scene.expected[s][n].score = 0;
            for (int n = 1; n <= 40; ++n) scene.scores.f(s, n, static_cast<float>(scene.expected[s][n].score));
        }
        pack(runner, scene, ids, beta);
        const std::vector<int> literal = beta == 0 ? std::vector<int>{1, 2, 3, 4, 5, 6, 9} : std::vector<int>{1, 2, 3, 4, 5, 6, 7};
        for (int s : {3, 1}) expect_packed(scene, s, literal);
    }
    for (bool infinite : {false, true}) {
        Scene scene; scene.known_tree();
        for (int s = 0; s < capacity; ++s) for (int n = 1; n <= 40; ++n) {
            const float score = infinite ? -std::numeric_limits<float>::infinity() : 0.0f;
            scene.scores.f(s, n, score); scene.expected[s][n].score = score;
        }
        pack(runner, scene, ids, infinite ? 0 : -std::numeric_limits<double>::max());
        for (int s : {3, 1}) expect_packed(scene, s, {1, 2, 3, 4, 5, 6, 7});
    }
    for (int kind = 0; kind < 9; ++kind) {
        Scene scene; scene.known_tree();
        if (kind == 0) scene.meta.i(3, 0, 0);
        if (kind == 1) scene.meta.i(3, 44 + 17, 1); // Skips a complete level.
        if (kind == 2) scene.meta.i(3, 44 + 9, 9);  // Self cycle.
        if (kind == 3) scene.meta.i(3, 3 + 40, -1);
        if (kind == 4) scene.scores.f(3, 40, std::numeric_limits<float>::quiet_NaN());
        if (kind == 5) scene.scores.f(3, 40, std::numeric_limits<float>::infinity());
        if (kind == 6) scene.scores.f(3, 9, 0); // Child exceeds parent path score.
        if (kind == 7) scene.scores.f(3, 0, -1);
        if (kind == 8) scene.meta.i(3, 44, 0);
        pack(runner, scene, ids, -0.2);
        require(scene.packed.i(3) == 0, "malformed tree must fail closed, not drop to smaller budget");
        require(scene.packed.i(1) == 8, "bad tree invalidated another request");
    }
    for (double beta : {0.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        Scene scene; scene.known_tree();
        const auto bad = pack_geometry(beta);
        require(runner.invoke("mtp_pruned_pack", pack_geometry(0), {&ids, &scene.meta, &scene.scores, &scene.packed}, {3}, {}, &bad) != RAD_OK,
                "invalid beta accepted");
    }
    ++runner.cases;
}

void prepare_invalid_metadata(Runner& runner) {
    for (int kind = 0; kind < 4; ++kind) {
        Scene scene; scene.known_tree();
        Tensor ids = request_ids({3, 1});
        if (kind == 0) scene.meta.i(3, 0, 0);
        if (kind == 1) scene.meta.i(3, 44 + 17, 17);
        if (kind == 2) ids.i(0, 0, capacity);
        if (kind == 3) ids.i(0, 0, -1);
        Prepared output(16);
        check(runner.invoke("mtp_pruned_prepare", selector_geometry(4), output.bind(ids, scene.meta), {2, 3, 4, 5, 6, 7}), "prepare invalid metadata");
        for (int row = 0; row < 8; ++row) {
            require(output.tokens.i(row) == 0 && output.positions.i(row) == 0 && output.gather.i(row) == 0,
                    "invalid prepare must produce safe gather inputs");
            require(output.slots.i(row) == -1 && output.prefix.i(row) == 0, "invalid prepare must disable provisional writes");
            for (int d = 0; d < 4; ++d) require(output.ancestors.i(row, d) == -1, "invalid prepare ancestry");
        }
        require(output.slots.i(8) >= 32, "invalid prepare leaked into adjacent tile request");
    }
    Scene scene; scene.known_tree(); Tensor ids = request_ids({3, 1}); Prepared out(16);
    for (int kind = 0; kind < 4; ++kind) {
        require(runner.invoke("mtp_pruned_prepare", selector_geometry(3), out.bind(ids, scene.meta), {2, 3, 4, 5, 6, 7}, [&](auto& t) {
            if (kind == 0) t[5].shape[1] = 3;
            if (kind == 1) t[4].data = t[2].data;
            if (kind == 2) t[6].shape[0] = 15;
            if (kind == 3) t[0].dtype = RAD_F32;
        }) != RAD_OK, "malformed prepare descriptors accepted");
    }
    for (int bad_id : {-1, capacity}) {
        Tensor bad = request_ids({bad_id, 1});
        pack(runner, scene, bad, -0.2); // Must not access outside global buffers.
        require(scene.packed.i(1) == 8, "bad pack request id affected valid request");
    }
    ++runner.cases;
}

void overlay_oracle_and_poison(Runner& runner, int dim, int depth, uint32_t dtype) {
    Scene scene; scene.known_tree();
    Tensor ids = request_ids({3, 1});
    // Distinct branch ancestry at every level; final node is the current frontier row.
    for (int s : {3, 1}) for (int d = 2; d <= 4; ++d) for (int j = 0; j < 8; ++j) {
        const int node = 1 + (d - 1) * 8 + j;
        const int parent = 1 + (d - 2) * 8 + (j + d + 1) % 8;
        scene.meta.i(s, 44 + node, parent); scene.expected[s][node].parent = parent;
    }
    Prepared prepared(16);
    prepare(runner, scene, ids, {3, 1}, depth + 1, prepared);
    Tensor q = hidden_bank(16, 2 * dim);
    Overlay overlay(2, dim, 2, 1, depth, dtype);
    const auto original_meta = scene.meta.storage;
    overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
    overlay.compare(overlay.oracle(q, prepared.prefix, prepared.ancestors));
    require(scene.meta.storage == original_meta, "valid overlay mutated persistent tree metadata");
    const auto baseline = overlay.output();
    const auto original_provisional = overlay.provisional.storage;
    const auto original_committed = overlay.committed.storage;
    const int sibling = prepared.ancestors.i(1, depth - 1);
    for (int x = 0; x < dim; ++x)
        store_kv(overlay.provisional, sibling / 4, 0, sibling % 4, dim + x, 32);
    overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto sibling_result = overlay.output();
    require(std::equal(baseline.begin(), baseline.begin() + 2 * dim, sibling_result.begin()), "sibling KV entered branch0 attention");
    require(!std::equal(baseline.begin() + 2 * dim, baseline.begin() + 4 * dim, sibling_result.begin() + 2 * dim), "sibling poison did not reach its own branch");
    require(std::equal(baseline.begin() + 8 * 2 * dim, baseline.end(), sibling_result.begin() + 8 * 2 * dim), "provisional KV crossed real-request boundary");
    overlay.compare(overlay.oracle(q, prepared.prefix, prepared.ancestors));
    overlay.provisional.storage = original_provisional;
    const int ancestor = prepared.ancestors.i(0, 0);
    for (int x = 0; x < dim; ++x)
        store_kv(overlay.provisional, ancestor / 4, 0, ancestor % 4, dim + x, -32);
    overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto ancestor_result = overlay.output();
    require(!std::equal(baseline.begin(), baseline.begin() + 2 * dim, ancestor_result.begin()), "visible ancestor KV poison did not reach branch0");
    overlay.compare(overlay.oracle(q, prepared.prefix, prepared.ancestors));
    overlay.provisional.storage = original_provisional;
    // A committed position just beyond prefix must remain invisible, even in a live page.
    const int invisible = prepared.prefix.i(0);
    const int page = overlay.table.i(0, invisible / 4);
    for (int x = 0; x < dim; ++x) store_kv(overlay.committed, page, 0, invisible % 4, dim + x, 32);
    overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto outside_prefix = overlay.output();
    require(std::equal(baseline.begin(), baseline.begin() + 2 * dim, outside_prefix.begin()), "attention read beyond committed prefix");
    overlay.committed.storage = original_committed;
    const int other_page = overlay.table.i(1, 0);
    for (int x = 0; x < dim; ++x) store_kv(overlay.committed, other_page, 0, 0, dim + x, 32);
    overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
    const auto other_committed = overlay.output();
    require(std::equal(baseline.begin(), baseline.begin() + 8 * 2 * dim, other_committed.begin()), "committed KV crossed request boundary");
    require(!std::equal(baseline.begin() + 8 * 2 * dim, baseline.end(), other_committed.begin() + 8 * 2 * dim), "committed poison did not reach owning request");
    overlay.compare(overlay.oracle(q, prepared.prefix, prepared.ancestors));
    ++runner.cases;
}

void overlay_multiple_kv_heads(Runner& runner) {
    for (uint32_t dtype : {uint32_t(RAD_BF16), uint32_t(RAD_F8E4M3)}) {
        Scene scene; scene.known_tree(); Tensor ids = request_ids({3, 0, 2}); Prepared prepared(24);
        prepare(runner, scene, ids, {3, 0, 2}, 3, prepared);
        Tensor q = hidden_bank(24, 4 * 17);
        Overlay overlay(3, 17, 4, 2, 2, dtype);
        overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
        overlay.compare(overlay.oracle(q, prepared.prefix, prepared.ancestors));
        ++runner.cases;
    }
}

void overlay_invalid_metadata(Runner& runner) {
    for (int kind = 0; kind < 10; ++kind) {
        Scene scene; scene.known_tree(); Tensor ids = request_ids({3, 1}); Prepared prepared(16);
        prepare(runner, scene, ids, {3, 1}, 4, prepared);
        Tensor q = hidden_bank(16, 16); Overlay overlay(2, 8, 2, 1, 3, RAD_BF16);
        if (kind == 0) prepared.prefix.i(0, 0, -1);
        if (kind == 1) prepared.prefix.i(0, 0, 17);
        if (kind == 2) overlay.table.i(0, 0, -1);
        if (kind == 3) overlay.table.i(0, 0, 8);
        if (kind == 4) prepared.ancestors.i(0, 0, 32); // Another request's ancestor.
        if (kind == 5) prepared.ancestors.i(0, 1, prepared.ancestors.i(0, 0)); // Repeated level.
        if (kind == 6) prepared.ancestors.i(0, 2, 17); // Current node must be slot16, not sibling17.
        if (kind == 7) prepared.ancestors.i(0, 3, 24); // Unused ancestry is not -1.
        if (kind == 8) prepared.ancestors.i(0, 0, 64);
        if (kind == 9) scene.meta.i(3, 0, 0);
        overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, ids);
        for (int x = 0; x < 16; ++x)
            require(overlay.out.at<uint16_t>(overlay.out.offset(0, x)) == 0, "bad overlay metadata must zero query before unchecked KV reads");
        require(scene.meta.i(3) == 0, "bad overlay metadata must invalidate persistent global request status");
        require(scene.meta.i(1) == 1, "bad overlay metadata invalidated another request");
        pack(runner, scene, ids, -0.2);
        require(scene.packed.i(3) == 0 && scene.packed.i(1) == 8, "overlay failure did not propagate through final pack");
    }
    Scene scene; scene.known_tree(); Tensor ids = request_ids({3, 1}); Prepared prepared(16);
    prepare(runner, scene, ids, {3, 1}, 3, prepared);
    Tensor q = hidden_bank(16, 16); Overlay overlay(2, 8, 2, 1, 2, RAD_BF16);
    for (int kind = 0; kind < 10; ++kind) {
        require(runner.invoke("mtp_overlay_attn", overlay.geometry(), overlay.bind(q, prepared.prefix, prepared.ancestors, scene.meta, ids), {6, 7}, [&](auto& t) {
            if (kind == 0) t[0].dtype = RAD_F32;
            if (kind == 1) t[0].shape[0] = 15;
            if (kind == 2) t[1].shape[2] = 3;
            if (kind == 3) t[4].shape[0] = 15;
            if (kind == 4) t[5].stride[1] = 2;
            if (kind == 5) t[6].data = t[0].data;
            if (kind == 6) t[7].shape[1] = 84;
            if (kind == 7) t[8].dtype = RAD_F32;
            if (kind == 8) t[2].shape[0] = 1;
            if (kind == 9) t[1].dtype = RAD_F32;
        }) != RAD_OK, "malformed overlay descriptors accepted");
    }
    for (int kind = 0; kind < 9; ++kind) {
        auto bad_geometry = overlay.geometry();
        if (kind == 0) bad_geometry.set_i("M", 15);
        if (kind == 1) bad_geometry.set_i("head_dim", 257);
        if (kind == 2) bad_geometry.set_i("n_head", 3), bad_geometry.set_i("n_head_kv", 2);
        if (kind == 3) bad_geometry.set_i("block_size", 8);
        if (kind == 4) bad_geometry.set_i("depth", 5);
        if (kind == 5) bad_geometry.set_i("max_ctx", 0);
        if (kind == 6) bad_geometry.set_s("q_dtype", "f32");
        if (kind == 7) bad_geometry.set_s("kv_dtype", "f32");
        if (kind == 8) bad_geometry.set_i("N", 0);
        require(runner.invoke("mtp_overlay_attn", overlay.geometry(), overlay.bind(q, prepared.prefix, prepared.ancestors, scene.meta, ids),
                              {6, 7}, {}, &bad_geometry) != RAD_OK, "unsupported overlay geometry accepted");
    }
    for (int bad_id : {-1, capacity}) {
        Tensor bad_ids = request_ids({bad_id, 1});
        for (int s = 0; s < capacity; ++s) scene.meta.i(s, 0, 1);
        overlay.run(runner, q, prepared.prefix, prepared.ancestors, scene.meta, bad_ids);
        for (int x = 0; x < 16; ++x) require(overlay.out.at<uint16_t>(overlay.out.offset(0, x)) == 0, "invalid global request id must not load KV");
        for (int s = 0; s < capacity; ++s) require(scene.meta.i(s) == 0, "unaddressable request id must invalidate available statuses");
        pack(runner, scene, ids, -0.2);
        require(scene.packed.i(3) == 0 && scene.packed.i(1) == 0, "invalid global id did not fail closed through pack");
    }
    ++runner.cases;
}
void configuration(Runner& runner) {
    struct SavedEnv {
        std::array<const char*, 3> names{{"RADIANCE_MTP_TREE_POLICY", "RADIANCE_MTP_TREE_ROUNDS", "RADIANCE_MTP_TREE_COSTS"}};
        std::array<std::string, 3> values;
        std::array<bool, 3> present{};
        SavedEnv() {
            for (size_t i = 0; i < names.size(); ++i) {
                const char* value = std::getenv(names[i]);
                present[i] = value != nullptr;
                if (value) values[i] = value;
            }
        }
        ~SavedEnv() {
            for (size_t i = 0; i < names.size(); ++i) {
                if (present[i]) setenv(names[i], values[i].c_str(), 1);
                else unsetenv(names[i]);
            }
        }
        void set(const char* policy, const char* rounds, const char* costs) {
            const char* entries[] = {policy, rounds, costs};
            for (size_t i = 0; i < names.size(); ++i)
                require((entries[i] ? setenv(names[i], entries[i], 1) : unsetenv(names[i])) == 0, "set test configuration environment");
        }
    } env;
    auto parse = [&](const char* policy, const char* rounds, const char* costs, bool valid,
                     int expected_policy = 0, int expected_rounds = 3) {
        env.set(policy, rounds, costs);
        rad::MtpTreeConfig config{};
        const char* error = nullptr;
        require(rad::parse_mtp_tree_config(config, &error) == valid, "startup policy acceptance");
        if (valid) {
            require(!error && config.policy == expected_policy && config.rounds == expected_rounds, "startup policy/default rounds");
            require(config.pruned() == (expected_policy == 3), "startup pruned policy classification");
        } else require(error && *error, "configuration refusal must explain the error");
    };
    parse(nullptr, nullptr, nullptr, true);
    parse("off", nullptr, nullptr, true);
    parse("pruned", "5", nullptr, true, 3, 5);
    parse("pruned", "3", nullptr, false);
    parse("pruned", nullptr, nullptr, false);
    for (const char* policy : {"off", "score", "pruned"})
        parse(policy, "5", "1,2,3,4,5,6,7,8", false);
    parse("cost", "3", "1,2,3,4,5,6,7,8", true, 2, 3);
    ++runner.cases;
}
} // namespace

int rad_run_mtp_pruned_behavior(rad::Registry& registry, RadStream stream, bool device) {
    Runner runner{registry, stream, device};
    configuration(runner);
    frontier_and_tiles(runner, false);
    frontier_and_tiles(runner, true);
    normalization_and_refusals(runner);
    pack_ties_and_invalid_trees(runner);
    prepare_invalid_metadata(runner);
    for (uint32_t dtype : {uint32_t(RAD_BF16), uint32_t(RAD_F8E4M3)})
        for (int depth = 1; depth <= 4; ++depth)
            overlay_oracle_and_poison(runner, depth % 2 ? 8 : 80, depth, dtype);
    overlay_oracle_and_poison(runner, 256, 4, RAD_BF16);
    overlay_multiple_kv_heads(runner);
    overlay_invalid_metadata(runner);
    require(runner.cases > 0 && runner.launches > 0, "zero behavioral cases/launches");
    std::printf("ALL_PASS pruned_mtp_%s cases=%zu launches=%zu W=8 K=8 D=5 B=7\n",
                device ? "gpu" : "host", runner.cases, runner.launches);
    return 0;
}

#ifndef RAD_MTP_PRUNED_GPU_TEST
int main(int argc, char** argv) {
    try {
        require(argc == 5 && std::string(argv[1]) == "--ref" && std::string(argv[3]) == "--libref",
                "usage: mtp_pruned_tree_test --ref PATH --libref PATH");
        rad::Registry registry;
        check(registry.load_plugin(argv[2], 0), "load mine_ref");
        check(registry.load_plugin(argv[4], 1), "load libref");
        check(registry.check_complete(), "complete host schemas");
        return rad_run_mtp_pruned_behavior(registry, nullptr, false);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
#endif
