/* Independent libref attn_paged oracle: gather the original prefix plus each node's
 * ancestors into separately shuffled pages, then run ordinary one-row attention.
 * The candidate is a pinned real GPU row, never mine_ref or a selector fallback.
 * Reordered FP32 attention is checked with the existing rad-kbench BF16 attention
 * rel_l2 budget (2e-2), fixed before running, independently for every query head. */
#include "abi/rad_plugin.h"
#include "core/device/device.h"
#include "core/plugin/registry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr size_t guard = 64;
constexpr int heads = 12, dim = 256, cols = heads * dim, block = 4;
// tools/rad_kbench.cpp, g_tol: attn_ / bf16. Do not tune against candidate output.
constexpr double rel_l2_limit = 2e-2;
constexpr const char* op = "attn_paged_tree";
constexpr const char* kernel = "mine_attn_paged_tree_h256_gqa12_b4";
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void check(int rc, const std::string& message) {
    if (rc != RAD_OK) throw std::runtime_error(message + ": " + rad_strerror(rc));
}
size_t width(uint32_t dtype) {
    return dtype == RAD_BF16 ? 2 : dtype == RAD_I32 ? 4 : 1;
}
RadTensor tensor(void* data, uint32_t dtype, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    for (int64_t n : shape) t.shape[t.rank++] = n;
    int64_t stride = 1;
    for (int d = int(t.rank) - 1; d >= 0; --d) {
        t.stride[d] = stride;
        stride *= t.shape[d];
    }
    return t;
}
struct Session {
    RadStream stream = nullptr;
    Session() {
        check(rad_dev_set(0), "select GPU");
        RadDeviceProps p{};
        check(rad_dev_props(0, &p), "GPU properties");
        require(!p.is_host_backend && std::string(p.arch) == "gfx1201",
                "requires real gfx1201; missing GPU is a failure, not a skip");
        check(rad_stream_create(&stream, 0), "create stream");
    }
    ~Session() { if (stream) rad_stream_destroy(stream); }
};
struct Buffer {
    RadTensor t{};
    std::vector<uint8_t> initial, writable;
    std::string name;
};
struct Scene {
    std::vector<Buffer> b;
    std::vector<int> lengths, prefixes, starts;
    int rows = 0, max_ctx = 0, q_len = 0;
    bool bushy = false;
    int add(const char* name, uint32_t dtype, std::initializer_list<int64_t> shape,
            bool pitched = false) {
        Buffer v;
        v.name = name;
        v.t = tensor(nullptr, dtype, shape);
        if (pitched) {
            // KV slot/element are dense; only its block/head pitches may differ.
            if (v.t.rank == 4) {
                v.t.stride[1] += 16;
                v.t.stride[0] = v.t.stride[1] * v.t.shape[1] + 32;
            } else v.t.stride[0] += 7;
        }
        const size_t bytes = size_t(v.t.stride[0] * v.t.shape[0]) * width(dtype);
        v.initial.resize(2 * guard + bytes, 0xa5);
        v.writable.resize(v.initial.size(), 0);
        b.push_back(std::move(v));
        return int(b.size()) - 1;
    }
    size_t at(int id, std::initializer_list<int64_t> index) const {
        size_t offset = guard;
        int d = 0;
        for (int64_t i : index) offset += size_t(i * b[size_t(id)].t.stride[d++]) * width(b[size_t(id)].t.dtype);
        return offset;
    }
    template<class T> void put(int id, std::initializer_list<int64_t> index, T value) {
        const size_t off = at(id, index);
        require(off + sizeof(T) <= b[size_t(id)].initial.size() - guard, "fixture write out of bounds");
        std::memcpy(b[size_t(id)].initial.data() + off, &value, sizeof value);
    }
    template<class T> T get(int id, std::initializer_list<int64_t> index) const {
        T value;
        std::memcpy(&value, b[size_t(id)].initial.data() + at(id, index), sizeof value);
        return value;
    }
    uint8_t kv(int seq, int logical, int d) const {
        const int page = get<int32_t>(2, {seq, logical / block});
        return get<uint8_t>(1, {page, 0, logical % block, d});
    }
    rad::Geometry geometry() const {
        rad::Geometry g;
        g.set_i("q_len", q_len); g.set_i("max_tokens", rows);
        g.set_i("max_seqs", int(lengths.size())); g.set_i("max_ctx", max_ctx);
        g.set_i("head_dim", dim); g.set_i("gqa", heads); g.set_i("n_head", heads);
        g.set_i("block_size", block); g.set_i("causal", 1); g.set_i("window", 0);
        g.set_s("q_dtype", "bf16"); g.set_s("kv_dtype", "fp8_e4m3");
        return g;
    }
};
uint32_t random_bits(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return seed;
}
std::vector<int32_t> shuffled_pages(int count, uint32_t seed) {
    std::vector<int32_t> pages(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) pages[size_t(i)] = i;
    for (int i = count - 1; i > 0; --i)
        std::swap(pages[size_t(i)], pages[size_t(random_bits(seed) % uint32_t(i + 1))]);
    return pages;
}
Scene fixture(std::vector<int> lengths, std::vector<int> prefixes, bool bushy, bool pitched) {
    require(lengths.size() == prefixes.size(), "fixture sequence mismatch");
    Scene s;
    s.lengths = std::move(lengths); s.prefixes = std::move(prefixes); s.bushy = bushy;
    int logical_pages = 0;
    for (size_t seq = 0; seq < s.lengths.size(); ++seq) {
        s.starts.push_back(s.rows);
        s.rows += s.lengths[seq];
        s.q_len = std::max(s.q_len, s.lengths[seq]);
        s.max_ctx = std::max(s.max_ctx, s.prefixes[seq] + s.lengths[seq]);
        logical_pages += (s.prefixes[seq] + s.lengths[seq] + 3) / 4;
    }
    const int pages = logical_pages + 3, blocks = (s.max_ctx + 3) / 4;
    s.add("q", RAD_BF16, {s.rows, heads, dim});
    s.add("kv", RAD_F8E4M3, {pages, 1, block, 2 * dim}, pitched);
    s.add("block_table", RAD_I32, {int(s.lengths.size()), blocks}, pitched);
    s.add("seqused", RAD_I32, {int(s.lengths.size())});
    s.add("cu_seqlens", RAD_I32, {int(s.lengths.size()) + 1});
    s.add("spec_parent", RAD_I32, {s.rows});
    s.add("out", RAD_BF16, {s.rows, heads, dim});
    uint32_t seed = 0x1632a9b5u;
    for (int row = 0; row < s.rows; ++row) for (int h = 0; h < heads; ++h) for (int d = 0; d < dim; ++d) {
        const float q = float(int(random_bits(seed) >> 24) - 128) / 128.f;
        s.put(0, {row, h, d}, rad_f32_to_bf16(q));
        s.put(6, {row, h, d}, uint16_t(0x7fc1)); // NaN: an unwritten row cannot pass.
        const size_t off = s.at(6, {row, h, d});
        std::fill_n(s.b[6].writable.begin() + off, 2, uint8_t(1));
    }
    // Fill every physical page independently; no duplicate page can hide sequence leakage.
    for (int p = 0; p < pages; ++p) for (int slot = 0; slot < block; ++slot) for (int d = 0; d < 2 * dim; ++d) {
        const uint32_t r = random_bits(seed);
        const uint8_t code = uint8_t((d < dim ? 0x20 : 0x30) + ((r >> 24) % 24) + ((r >> 16) & 0x80));
        s.put(1, {p, 0, slot, d}, code); // finite OCP E4M3, not host-requantised floats
    }
    int page = 0;
    const auto physical = shuffled_pages(pages, 0xf275e123u);
    for (int seq = 0; seq < int(s.lengths.size()); ++seq) {
        const int n = s.lengths[size_t(seq)], start = s.starts[size_t(seq)];
        const int ctx = n + s.prefixes[size_t(seq)];
        s.put(3, {seq}, int32_t(ctx)); s.put(4, {seq}, int32_t(start));
        for (int b = 0; b < blocks; ++b) s.put(2, {seq, b}, int32_t(-1));
        // A bijection across sequences, with three unused pages and shuffled physical blocks.
        for (int b = 0; b < (ctx + 3) / 4; ++b) s.put(2, {seq, b}, physical[size_t(page++)]);
        for (int j = 0; j < n; ++j) {
            // Small trees branch at the root; MTP8/64 has seven siblings per spine level.
            const int parent = j == 0 ? -1 : !bushy ? j - 1 :
                               n <= 16 ? 0 : j <= 8 ? j - 1 : (j - 9) / 7;
            s.put(5, {start + j}, int32_t(parent));
        }
    }
    s.put(4, {int(s.lengths.size())}, int32_t(s.rows));
    return s;
}
const rad::KernelRow* pin(rad::Registry& reg, const char* operation, const char* plugin,
                          const char* name, int domain) {
    const auto* rows = reg.rows_for(operation);
    require(rows != nullptr, std::string("missing op ") + operation);
    const rad::KernelRow* found = nullptr;
    for (const auto* row : *rows) if (row->plugin == plugin && row->info && row->info->name &&
            !std::strcmp(row->info->name, name)) {
        require(!found, "duplicate pinned kernel"); found = row;
    }
    require(found && found->info->launch && found->info->domain == domain,
            std::string("missing actual row ") + plugin + "/" + name);
    std::printf("ROW plugin=%s op=%s kernel=%s domain=%d\n", plugin, operation, name, domain);
    return found;
}
struct Counts {
    size_t cases = 0, launches = 0, oracle_rows = 0, checked_rows = 0, elements = 0;
    size_t negative_launches = 0, negative_heads = 0, invalid_launches = 0, refusals = 0;
    size_t canary_bytes = 0, repeated_launches = 0;
    double worst_rel_l2 = 0, max_abs = 0;
};
std::vector<uint16_t> oracle(const Scene& s, const rad::KernelRow* row, Counts& counts) {
    std::vector<uint16_t> expected(size_t(s.rows) * cols);
    for (int seq = 0; seq < int(s.lengths.size()); ++seq) for (int j = 0; j < s.lengths[size_t(seq)]; ++j) {
        std::vector<int> path;
        for (int p = j; p >= 0; p = s.get<int32_t>(5, {s.starts[size_t(seq)] + p})) path.push_back(p);
        std::reverse(path.begin(), path.end());
        const int prefix = s.prefixes[size_t(seq)], ctx = prefix + int(path.size());
        const int pages = (ctx + 3) / 4;
        std::vector<uint8_t> kv(size_t(pages) * block * 2 * dim, 0x7f);
        auto bt = shuffled_pages(pages, 0x789521edu ^ uint32_t(j));
        for (int k = 0; k < ctx; ++k) {
            const int original = k < prefix ? k : prefix + path[size_t(k - prefix)];
            const size_t destination = size_t(bt[size_t(k / block)] * block + k % block) * 2 * dim;
            for (int d = 0; d < 2 * dim; ++d) kv[destination + size_t(d)] = s.kv(seq, original, d);
        }
        int32_t used = ctx;
        const int token = s.starts[size_t(seq)] + j;
        RadTensor q = s.b[0].t;
        q.data = const_cast<uint8_t*>(s.b[0].initial.data()) + s.at(0, {token, 0, 0});
        q.shape[0] = 1;
        RadTensor t[] = {q, tensor(kv.data(), RAD_F8E4M3, {pages, 1, block, 2 * dim}),
            tensor(bt.data(), RAD_I32, {1, pages}), tensor(&used, RAD_I32, {1}),
            RadTensor{}, RadTensor{}, RadTensor{},
            tensor(expected.data() + size_t(token) * cols, RAD_BF16, {1, heads, dim})};
        rad::Geometry g = s.geometry();
        g.set_i("q_len", 1); g.set_i("max_tokens", 1); g.set_i("max_seqs", 1); g.set_i("max_ctx", ctx);
        RadArgs a{};
        a.t = t; a.n_t = 8; a.p = g.params(); a.n_p = g.n_params(); a.rank = 0; a.world_size = 1;
        check(row->info->launch(&a, nullptr), "independent libref attn_paged");
        ++counts.oracle_rows;
    }
    return expected;
}
struct Device {
    void* data = nullptr;
    explicit Device(size_t bytes) {
        data = rad_dev_alloc(int64_t(bytes), RAD_MEM_DEVICE);
        require(data != nullptr, "device allocation failed");
    }
    ~Device() { if (data) rad_dev_free(data, RAD_MEM_DEVICE); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other) noexcept : data(other.data) { other.data = nullptr; }
};
struct Run {
    const Session& session;
    const Scene& scene;
    rad::Resolved resolved;
    std::vector<Device> memory;
    std::vector<RadTensor> t;
    std::vector<uint8_t> scratch_initial;
    Device scratch;
    RadArgs a{};
    static size_t scratch_size(const rad::Resolved& r) {
        RadArgs args{}; args.p = r.geom.params(); args.n_p = r.geom.n_params();
        require(r.row->info->scratch != nullptr, "missing candidate scratch contract");
        const int64_t bytes = r.row->info->scratch(&args);
        require(bytes > 0, "empty candidate scratch");
        return size_t(bytes);
    }
    Run(const Session& session_, const Scene& s, rad::Registry& registry, const rad::KernelRow* row)
        : session(session_), scene(s), resolved{}, scratch(2 * guard + scratch_bytes(s, row)) {
        resolved.row = row; resolved.geom = s.geometry();
        check(registry.init_instance(resolved, 0, 1), "initialize pinned candidate");
        for (const auto& b : s.b) {
            memory.emplace_back(b.initial.size());
            RadTensor v = b.t;
            v.data = static_cast<uint8_t*>(memory.back().data) + guard;
            t.push_back(v);
            check(rad_memcpy_async(memory.back().data, b.initial.data(), int64_t(b.initial.size()), session.stream), "upload");
        }
        scratch_initial.resize(2 * guard + scratch_size(resolved), 0xa5);
        a.t = t.data(); a.n_t = int(t.size()); a.p = resolved.geom.params(); a.n_p = resolved.geom.n_params();
        a.instance = resolved.instance; a.rank = 0; a.world_size = 1;
        a.scratch = static_cast<uint8_t*>(scratch.data) + guard;
        a.scratch_bytes = int64_t(scratch_initial.size() - 2 * guard);
        check(rad_stream_sync(session.stream), "finish uploads");
    }
    static size_t scratch_bytes(const Scene& s, const rad::KernelRow* row) {
        rad::Resolved r; r.row = row; r.geom = s.geometry(); return scratch_size(r);
    }
    void reset_output() {
        const auto& out = scene.b[6].initial;
        check(rad_memcpy_async(memory[6].data, out.data(), int64_t(out.size()), session.stream), "poison output");
        check(rad_memcpy_async(scratch.data, scratch_initial.data(), int64_t(scratch_initial.size()), session.stream), "poison scratch");
    }
    void parents(const std::vector<uint8_t>& bytes) {
        check(rad_memcpy_async(memory[5].data, bytes.data(), int64_t(bytes.size()), session.stream), "upload parent mutation");
    }
    std::vector<uint16_t> download(Counts& counts, const std::vector<uint8_t>* parents_override = nullptr,
                                   bool refused = false) {
        std::vector<uint16_t> output(size_t(scene.rows) * cols);
        for (size_t id = 0; id < scene.b.size(); ++id) {
            const auto& b = scene.b[id];
            std::vector<uint8_t> actual(b.initial.size());
            check(rad_memcpy_async(actual.data(), memory[id].data, int64_t(actual.size()), session.stream), "download operand");
            check(rad_stream_sync(session.stream), "finish operand download");
            const auto& initial = id == 5 && parents_override ? *parents_override : b.initial;
            for (size_t i = 0; i < actual.size(); ++i) if (refused || !b.writable[i]) {
                require(actual[i] == initial[i], b.name + " changed unowned byte " + std::to_string(i));
                ++counts.canary_bytes;
            }
            if (id == 6) std::memcpy(output.data(), actual.data() + guard, output.size() * 2);
        }
        std::vector<uint8_t> scratch_copy(scratch_initial.size());
        check(rad_memcpy_async(scratch_copy.data(), scratch.data, int64_t(scratch_copy.size()), session.stream), "download scratch");
        check(rad_stream_sync(session.stream), "finish scratch download");
        for (size_t i = 0; i < scratch_copy.size(); ++i) if (refused || i < guard || i >= scratch_copy.size() - guard) {
            require(scratch_copy[i] == scratch_initial[i], "scratch canary changed"); ++counts.canary_bytes;
        }
        return output;
    }
    void launch() {
        check(resolved.row->info->launch(&a, session.stream), "pinned GPU launch");
        check(rad_stream_sync(session.stream), "GPU completion");
    }
};
double head_error(const uint16_t* actual, const uint16_t* expected, double* max_abs = nullptr) {
    double numerator = 0, denominator = 0;
    for (int d = 0; d < dim; ++d) {
        const float a = rad_bf16_to_f32(actual[d]), b = rad_bf16_to_f32(expected[d]);
        require(std::isfinite(a) && std::isfinite(b), "nonfinite or unwritten attention output");
        const double delta = double(a) - double(b);
        if (max_abs) *max_abs = std::max(*max_abs, std::abs(delta));
        numerator += delta * delta; denominator += double(b) * double(b);
    }
    return denominator > 0 ? std::sqrt(numerator / denominator) : numerator > 0 ? INFINITY : 0;
}
void compare(const Scene& s, const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected,
             Counts& counts, int zero_sequence = -1) {
    for (int seq = 0; seq < int(s.lengths.size()); ++seq) for (int j = 0; j < s.lengths[size_t(seq)]; ++j) {
        const int row = s.starts[size_t(seq)] + j;
        for (int h = 0; h < heads; ++h) {
            const size_t off = size_t(row) * cols + size_t(h) * dim;
            if (seq == zero_sequence) {
                for (int d = 0; d < dim; ++d) require(actual[off + size_t(d)] == 0, "invalid parent sequence did not zero output");
            } else {
                const double error = head_error(actual.data() + off, expected.data() + off, &counts.max_abs);
                counts.worst_rel_l2 = std::max(counts.worst_rel_l2, error);
                require(error <= rel_l2_limit, "attention mismatch row=" + std::to_string(row) +
                    " head=" + std::to_string(h) + " rel_l2=" + std::to_string(error));
            }
            counts.elements += dim;
        }
        ++counts.checked_rows;
    }
}
void mutate(std::vector<uint8_t>& bytes, const Scene& s, int row, int32_t parent) {
    std::memcpy(bytes.data() + s.at(5, {row}), &parent, sizeof parent);
}
void run_case(const Session& session, rad::Registry& reg, const rad::KernelRow* candidate,
              const rad::KernelRow* ref, const Scene& s, Counts& counts) {
    const auto expected = oracle(s, ref, counts);
    const auto selected = reg.resolve(op, s.geometry(), RAD_DOMAIN_DEVICE);
    require(selected.row == candidate, "selector did not choose the pinned real GPU row");
    Run run(session, s, reg, candidate);
    for (int replay = 0; replay < 3; ++replay) {
        run.reset_output(); run.launch(); ++counts.launches;
        if (replay) ++counts.repeated_launches;
        compare(s, run.download(counts), expected, counts);
    }
    // A valid but wrong parent must fail the SAME numerical predicate, not a special tighter one.
    if (s.lengths.back() > 2) {
        const int seq = int(s.lengths.size()) - 1;
        const int local = s.lengths[size_t(seq)] - 1;
        const int token = s.starts[size_t(seq)] + local;
        const int parent = s.get<int32_t>(5, {token});
        auto changed = s.b[5].initial;
        mutate(changed, s, token, parent == 0 ? local - 1 : 0);
        run.parents(changed); run.reset_output(); run.launch();
        ++counts.launches; ++counts.negative_launches;
        const auto wrong = run.download(counts, &changed);
        size_t witnesses = 0;
        for (int h = 0; h < heads; ++h) {
            const size_t off = size_t(token) * cols + size_t(h) * dim;
            if (head_error(wrong.data() + off, expected.data() + off) > rel_l2_limit) ++witnesses;
        }
        require(witnesses > 0, "wrong-parent negative did not fail the numeric comparison");
        counts.negative_heads += witnesses;
        run.parents(s.b[5].initial);
    }
    // Device metadata validation: root, negative, self, forward, and out-of-sequence parent.
    if (s.bushy && s.lengths.size() == 2) {
        const int seq = 1, start = s.starts[1], n = s.lengths[1];
        for (int bad : {0, -2, 9, 10, n}) {
            auto changed = s.b[5].initial;
            mutate(changed, s, start + (bad == 0 ? 0 : 9), int32_t(bad));
            run.parents(changed); run.reset_output(); run.launch();
            ++counts.launches; ++counts.invalid_launches;
            compare(s, run.download(counts, &changed), expected, counts, seq);
        }
        run.parents(s.b[5].initial);
        // Cap 66 is rejected by selection AND by a direct pinned launch; no output/scratch writes.
        auto invalid = s.geometry(); invalid.set_i("q_len", 66);
        require(!reg.resolve(op, invalid, RAD_DOMAIN_DEVICE).row, "selector accepted q_len=66");
        run.reset_output();
        run.a.p = invalid.params(); run.a.n_p = invalid.n_params();
        require(candidate->info->launch(&run.a, session.stream) != RAD_OK, "pinned row accepted q_len=66");
        run.download(counts, nullptr, true); ++counts.refusals;
        run.a.p = run.resolved.geom.params(); run.a.n_p = run.resolved.geom.n_params();
        for (int operand : {0, 1, 6}) {
            run.reset_output();
            RadTensor saved = run.t[size_t(operand)];
            run.t[size_t(operand)].shape[run.t[size_t(operand)].rank - 1] -= 1;
            require(candidate->info->launch(&run.a, session.stream) != RAD_OK, "pinned row accepted malformed shape");
            run.t[size_t(operand)] = saved;
            run.download(counts, nullptr, true); ++counts.refusals;
        }
    }
    ++counts.cases;
    std::printf("CASE nodes=%d sequences=%zu rows=%d prefix0=%d bushy=%d pitched=%d repeats=3\n",
        s.q_len, s.lengths.size(), s.rows, s.prefixes[0], int(s.bushy), int(s.b[1].t.stride[0] != 2048));
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string mine = "./mine.so", libref = "./libref.so";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::puts("attn-paged-tree-gpu-test --mine PATH --libref PATH"); return 0;
            }
            require((arg == "--mine" || arg == "--libref") && i + 1 < argc, "unknown/incomplete argument " + arg);
            if (arg == "--mine") mine = argv[++i]; else libref = argv[++i];
        }
        Session session;
        rad::Registry reg;
        reg.set_code_device(0);
        check(reg.load_plugin(mine, 0), "load mine"); check(reg.load_plugin(libref, 1), "load libref");
        check(reg.check_complete(), "schema completeness");
        require(reg.plugin("mine") && reg.plugin("libref"), "unexpected plugin identity");
        const auto* schema = reg.schema(op);
        require(schema && schema->n_operands == 7, "missing seven-operand tree attention schema");
        const auto* candidate = pin(reg, op, "mine", kernel, RAD_DOMAIN_DEVICE);
        const auto* ref = pin(reg, "attn_paged", "libref", "ref_attn_paged", RAD_DOMAIN_HOST);
        std::printf("PLUGIN candidate=%s oracle=%s tolerance=%.8g per-head-rel-l2\n",
                    mine.c_str(), libref.c_str(), rel_l2_limit);
        Counts counts;
        for (int n : {1, 16, 32, 64, 65})
            run_case(session, reg, candidate, ref, fixture({n}, {n == 1 ? 0 : 5}, false, n % 2 != 0), counts);
        for (int n : {5, 6, 8, 16})
            run_case(session, reg, candidate, ref, fixture({n}, {5}, true, true), counts);
        run_case(session, reg, candidate, ref, fixture({65}, {0}, true, true), counts);
        run_case(session, reg, candidate, ref, fixture({65, 33}, {0, 7}, true, true), counts);
        run_case(session, reg, candidate, ref, fixture({32, 16}, {129, 3}, false, false), counts);
        std::printf("ATTN_PAGED_TREE_GPU_PASS cases=%zu gpu_launches=%zu repeated_launches=%zu oracle_rows=%zu "
            "checked_rows=%zu elements=%zu wrong_parent_launches=%zu negative_heads=%zu invalid_parent_launches=%zu "
            "refusals=%zu canary_bytes=%zu worst_head_rel_l2=%.9g max_abs=%.9g limit=%.9g oracle=libref-attn_paged-gathered-paths\n",
            counts.cases, counts.launches, counts.repeated_launches, counts.oracle_rows, counts.checked_rows,
            counts.elements, counts.negative_launches, counts.negative_heads, counts.invalid_launches,
            counts.refusals, counts.canary_bytes, counts.worst_rel_l2, counts.max_abs, rel_l2_limit);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ATTN_PAGED_TREE_GPU_FAIL: %s\n", e.what()); return 1;
    }
}
