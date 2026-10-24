/* Exact GPU oracle: pinned mine conv + recurrent, then pinned libr4d quant_act_i8g.
 * --baseline loads the pre-change mine plugin into a separate registry. No host oracle,
 * selector fallback, missing GPU, missing row, or zero-case run can count as a pass.
 * --wide checks 33/65-node exact rows against the frozen 16-node plugin by chunked paths. */
#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "abi/rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
constexpr size_t pad = 64;
constexpr int hk = 8, hv = 24, dim = 128, channels = (2 * hk + hv) * dim;
constexpr const char* fused_op = "gdn_conv_recurrent_tree_i8g";
const char* selected_kernel = "mine_gdn_conv_recurrent_tree_i8g";

void require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error(std::string(message));
}
void check(int rc, const std::string& message) {
    if (rc != RAD_OK) {
        const char* detail = rad_dev_last_error();
        throw std::runtime_error(message + ": " + rad_strerror(rc) + "; " + (detail ? detail : ""));
    }
}
size_t width(uint32_t dtype) {
    if (dtype == RAD_I8) return 1;
    if (dtype == RAD_BF16) return 2;
    require(dtype == RAD_F32 || dtype == RAD_I32, "unsupported fixture dtype");
    return 4;
}
struct Session {
    RadStream stream = nullptr;
    Session() {
        check(rad_dev_set(0), "select GPU");
        RadDeviceProps props{};
        check(rad_dev_props(0, &props), "GPU properties");
        require(!props.is_host_backend && std::string(props.arch) == "gfx1201",
                "requires real gfx1201; no host/skip pass");
        check(rad_stream_create(&stream, 0), "create stream");
    }
    ~Session() { if (stream) rad_stream_destroy(stream); }
};
struct Buffer {
    RadTensor tensor{};
    size_t offset = pad;
    std::vector<uint8_t> initial, writable;
    std::string name;
};
struct Scene {
    std::vector<Buffer> buffers;
    int begin = 0, end = 0, sequences = 0;
    int add(const char* name, uint32_t dtype, std::initializer_list<int64_t> shape,
            bool pitched = false, int inner = 1, int prefix = 0) {
        Buffer b;
        b.name = name;
        b.tensor.dtype = dtype;
        for (int64_t n : shape) b.tensor.shape[b.tensor.rank++] = n;
        int64_t stride = inner;
        for (int i = int(b.tensor.rank) - 1; i >= 0; --i) {
            b.tensor.stride[i] = stride;
            stride *= b.tensor.shape[i];
            if (pitched && i > 0) stride += 8;
        }
        b.offset += size_t(prefix) * width(dtype);
        b.initial.resize(pad + size_t(prefix + stride) * width(dtype) + pad, 0xa5);
        b.writable.resize(b.initial.size(), 0);
        buffers.push_back(std::move(b));
        return int(buffers.size()) - 1;
    }
    size_t at(int id, std::initializer_list<int64_t> index) const {
        const auto& b = buffers[size_t(id)];
        size_t off = b.offset;
        int axis = 0;
        for (int64_t i : index) off += size_t(i * b.tensor.stride[axis++]) * width(b.tensor.dtype);
        return off;
    }
    template<class T> void put(int id, std::initializer_list<int64_t> index, T value) {
        auto& b = buffers[size_t(id)];
        const size_t off = at(id, index);
        require(off + sizeof(T) <= b.initial.size() - pad, "fixture write outside payload");
        std::memcpy(b.initial.data() + off, &value, sizeof value);
    }
    template<class T> T get(int id, std::initializer_list<int64_t> index,
                             const std::vector<uint8_t>& bytes) const {
        T value;
        std::memcpy(&value, bytes.data() + at(id, index), sizeof value);
        return value;
    }
    void fill(int id, float step, int period, uint32_t seed = 0) {
        auto& b = buffers[size_t(id)];
        int64_t count = 1;
        for (unsigned d = 0; d < b.tensor.rank; ++d) count *= b.tensor.shape[d];
        const bool random = seed != 0;
        for (int64_t flat = 0; flat < count; ++flat) {
            int64_t rem = flat;
            size_t off = b.offset;
            for (int d = int(b.tensor.rank) - 1; d >= 0; --d) {
                off += size_t((rem % b.tensor.shape[d]) * b.tensor.stride[d]) * width(b.tensor.dtype);
                rem /= b.tensor.shape[d];
            }
            if (random) seed = 1664525u * seed + 1013904223u;
            const float value = random ? float(int(seed >> 8) - 8388608) * (step / 8388608.f) :
                                        float(int(flat % period) - period / 2) * step;
            if (b.tensor.dtype == RAD_BF16) {
                const uint16_t v = rad_f32_to_bf16(value);
                std::memcpy(b.initial.data() + off, &v, 2);
            } else {
                require(b.tensor.dtype == RAD_F32, "fill only floating tensors");
                std::memcpy(b.initial.data() + off, &value, 4);
            }
        }
    }
    void allow(int id) {
        auto& b = buffers[size_t(id)];
        int64_t count = 1;
        for (unsigned d = 1; d < b.tensor.rank; ++d) count *= b.tensor.shape[d];
        for (int row = begin; row < end; ++row) {
            for (int64_t flat = 0; flat < count; ++flat) {
                int64_t rem = flat;
                size_t off = b.offset + size_t(row * b.tensor.stride[0]) * width(b.tensor.dtype);
                for (int d = int(b.tensor.rank) - 1; d > 0; --d) {
                    off += size_t((rem % b.tensor.shape[d]) * b.tensor.stride[d]) * width(b.tensor.dtype);
                    rem /= b.tensor.shape[d];
                }
                std::fill_n(b.writable.begin() + off, width(b.tensor.dtype), uint8_t(1));
            }
        }
    }
};

Scene fixture(const std::vector<int>& nodes, bool pitched, bool ring, bool boundary,
              bool absent_parent = false) {
    Scene s;
    s.sequences = int(nodes.size());
    s.begin = pitched ? 2 : 0;
    s.end = s.begin;
    for (int n : nodes) s.end += n;
    const int tokens = s.end + (pitched ? 2 : 0), slots = s.sequences + 1;
    // The first twenty buffers are exactly the fused schema's positional operands.
    s.add("x", RAD_BF16, {tokens, channels}, pitched, pitched ? 2 : 1);
    s.add("weights", RAD_BF16, {channels, 4}, pitched);
    s.add("bias", RAD_BF16, {channels});
    s.add("history", RAD_BF16, {slots, channels, 7}, pitched);
    s.add("history_idx", RAD_I32, {s.sequences, 2}, true, 1, pitched ? 2 : 0);
    s.add("cu", RAD_I32, {s.sequences + 1}, false, pitched ? 2 : 1, pitched ? 2 : 0);
    s.add("parent", RAD_I32, {tokens}, false, pitched ? 2 : 1);
    s.add("ctx_before", RAD_I32, {s.sequences}, false, pitched ? 2 : 1, pitched ? 2 : 0);
    s.add("a", RAD_BF16, {tokens, hv}, pitched);
    s.add("b", RAD_BF16, {tokens, hv}, pitched);
    s.add("A_log", RAD_F32, {hv});
    s.add("dt_bias", RAD_F32, {hv});
    s.add("state", RAD_F32, {slots, hv, dim, dim}, pitched);
    s.add("state_idx", RAD_I32, {s.sequences, 2}, true, 1, pitched ? 2 : 0);
    s.add("z", RAD_BF16, {tokens, hv * dim}, pitched);
    s.add("norm_w", RAD_F32, {dim});
    s.add("saved_raw_x", RAD_BF16, {tokens, channels}, pitched);
    s.add("factors", RAD_F32, {tokens, hv, 257}, pitched);
    s.add("o_q", RAD_I8, {tokens, hv * dim}, pitched);
    s.add("o_s", RAD_F32, {tokens, hv}, pitched);
    s.add("q", RAD_BF16, {tokens, hk * dim}, pitched);
    s.add("k", RAD_BF16, {tokens, hk * dim}, pitched);
    s.add("v", RAD_BF16, {tokens, hv * dim}, pitched);
    s.add("o_bf16", RAD_BF16, {tokens, hv * dim}, pitched);
    s.fill(0, .03125f, 37); s.fill(1, .0625f, 7); s.fill(2, .015625f, 11);
    s.fill(3, .015625f, 23); s.fill(8, .125f, 9); s.fill(9, .125f, 7);
    s.fill(12, .0009765625f, 31); s.fill(14, .125f, 13);
    for (int h = 0; h < hv; ++h) { s.put(10, {h}, -2.f); s.put(11, {h}, -.5f); }
    for (int d = 0; d < dim; ++d) s.put(15, {d}, 1.f);
    int bos = s.begin;
    s.put(5, {0}, int32_t(bos));
    for (int seq = 0; seq < s.sequences; ++seq) {
        s.put(4, {seq, 0}, int32_t(s.sequences - seq));
        s.put(13, {seq, 0}, int32_t(seq)); // Deliberately different history/state slot maps.
        s.put(7, {seq}, int32_t(ring ? 19 + seq : seq));
        for (int node = 0; node < nodes[size_t(seq)]; ++node)
            s.put(6, {bos + node}, int32_t(node ? (node - 1) / 2 : -1));
        bos += nodes[size_t(seq)];
        s.put(5, {seq + 1}, int32_t(bos));
    }
    // Genuine zero quant groups alongside nonzero heads.
    for (int t = s.begin; t < s.end; ++t)
        for (int d = 0; d < dim; ++d) s.put(14, {t, d}, uint16_t(0));
    if (boundary) {
        s.fill(0, 0.f, 1); s.fill(1, 0.f, 1); s.fill(3, 0.f, 1);
        s.fill(8, 0.f, 1); s.fill(9, 0.f, 1); s.fill(12, 0.f, 1);
        for (int d = 0; d < channels; ++d) s.put(2, {d}, rad_f32_to_bf16(1.f));
        for (int t = s.begin; t < s.end; ++t)
            for (int d = 0; d < hv * dim; ++d) s.put(14, {t, d}, rad_f32_to_bf16(1.f));
        const float targets[] = {127.f, .5f, 1.5f, 2.5f, -.5f, -1.5f, -2.5f,
                                 .498046875f, .50390625f, -.498046875f, -.50390625f};
        const float silu_one = 1.f / (1.f + std::exp(-1.f));
        for (int d = 0; d < dim; ++d)
            s.put(15, {d}, targets[d % (sizeof(targets) / sizeof(targets[0]))] / silu_one);
    }
    if (absent_parent) s.buffers[6].tensor.rank = 0;
    for (int id = 16; id < 24; ++id) s.allow(id);
    return s;
}

struct Side {
    std::vector<void*> allocations;
    std::vector<RadTensor> tensors;
    Side(const Session& session, const Scene& scene) {
        for (const auto& b : scene.buffers) {
            void* p = rad_dev_alloc(int64_t(b.initial.size()), RAD_MEM_DEVICE);
            require(p != nullptr, "device allocation failed");
            allocations.push_back(p);
            RadTensor t = b.tensor;
            t.data = t.rank ? static_cast<uint8_t*>(p) + b.offset : nullptr;
            tensors.push_back(t);
            check(rad_memcpy_async(p, b.initial.data(), int64_t(b.initial.size()), session.stream), "upload");
        }
        check(rad_stream_sync(session.stream), "finish uploads");
    }
    ~Side() { for (void* p : allocations) rad_dev_free(p, RAD_MEM_DEVICE); }
    Side(const Side&) = delete;
    Side& operator=(const Side&) = delete;
    std::vector<std::vector<uint8_t>> download(const Session& session, const Scene& scene) const {
        std::vector<std::vector<uint8_t>> out;
        for (size_t i = 0; i < allocations.size(); ++i) {
            out.emplace_back(scene.buffers[i].initial.size());
            check(rad_memcpy_async(out.back().data(), allocations[i], int64_t(out.back().size()), session.stream), "download");
        }
        check(rad_stream_sync(session.stream), "finish downloads");
        return out;
    }
};
rad::Geometry geometry(int sequences, bool ring, int tokens) {
    rad::Geometry g;
    g.set_i("q_len", 16); g.set_i("node_capacity", 16);
    g.set_i("head_k", dim); g.set_i("head_v", dim);
    g.set_i("n_head_k", hk); g.set_i("n_head_v", hv);
    g.set_i("conv_width", 4); g.set_i("history_ring", ring ? 1 : 0); g.set_i("n_seq", sequences);
    g.set_i("max_tokens", tokens);
    g.set_f("scale", .125); g.set_f("eps", 1e-6); g.set_f("l2_eps", 1e-6); g.set_f("softplus_thr", 20);
    return g;
}
const rad::KernelRow* pin(rad::Registry& reg, const char* op, const char* name, const char* plugin) {
    const auto* rows = reg.rows_for(op);
    require(rows != nullptr, std::string("missing op ") + op);
    const rad::KernelRow* hit = nullptr;
    for (const auto* r : *rows) {
        if (r->plugin != plugin || !r->info || !r->info->name || std::strcmp(r->info->name, name)) continue;
        require(!hit, std::string("duplicate pinned row ") + name);
        hit = r;
    }
    require(hit && hit->info->launch && hit->info->domain == RAD_DOMAIN_DEVICE,
            std::string("missing pinned device row ") + plugin + "/" + name);
    std::printf("ROW plugin=%s op=%s kernel=%s\n", plugin, op, name);
    return hit;
}
rad::Resolved bind(rad::Registry& reg, const rad::KernelRow* row, const rad::Geometry& g) {
    rad::Resolved r;
    r.row = row; r.geom = g;
    check(reg.init_instance(r, 0, 1), "initialize pinned row");
    return r;
}
rad::Resolved candidate(rad::Registry& reg, const rad::Geometry& g) {
    const auto* schema = reg.schema(fused_op);
    require(schema && schema->n_operands == 20, "missing/mismatched fused schema");
    std::string error;
    check(reg.validate(fused_op, g.params(), g.n_params(), &error), "validate fused geometry " + error);
    rad::SelectionTrace trace;
    auto r = reg.resolve(fused_op, g, RAD_DOMAIN_DEVICE, &trace);
    require(r.row && r.row->plugin == "mine" && r.row->info && r.row->info->name &&
            !std::strcmp(r.row->info->name, selected_kernel),
            "fused selector did not select exact mine device row: " + trace.reason);
    check(reg.init_instance(r, 0, 1), "initialize selected candidate");
    std::printf("ROW role=candidate plugin=mine kernel=%s\n", r.row->info->name);
    return r;
}
int launch(const rad::Resolved& r, const std::vector<RadTensor>& tensors, RadStream stream) {
    RadArgs a{};
    a.t = tensors.data(); a.n_t = int(tensors.size());
    a.p = r.geom.params(); a.n_p = r.geom.n_params(); a.instance = r.instance;
    a.rank = 0; a.world_size = 1;
    require(!r.row->info->scratch || r.row->info->scratch(&a) == 0, "unexpected scratch requirement");
    return r.row->info->launch(&a, stream);
}
std::vector<RadTensor> pick(const Side& side, std::initializer_list<int> ids) {
    std::vector<RadTensor> t;
    for (int i : ids) t.push_back(side.tensors[size_t(i)]);
    return t;
}
size_t compare(const Scene& scene, const std::vector<std::vector<uint8_t>>& actual,
               const std::vector<std::vector<uint8_t>>& expected, bool reference, bool refused) {
    size_t bytes = 0;
    for (size_t id = 0; id < scene.buffers.size(); ++id) {
        const auto& b = scene.buffers[id];
        for (size_t i = 0; i < b.initial.size(); ++i) {
            const bool output = id >= 16 && id <= 19;
            if (output && !refused) {
                if (actual[id][i] != expected[id][i])
                    throw std::runtime_error(b.name + " byte mismatch at " + std::to_string(i));
                ++bytes;
            }
            const bool may_write = !refused && b.writable[i] && (output || (reference && id >= 20));
            if (!may_write && actual[id][i] != b.initial[i])
                throw std::runtime_error(b.name + " changed unowned byte " + std::to_string(i));
        }
    }
    return bytes;
}
void boundary_coverage(const Scene& scene, const std::vector<std::vector<uint8_t>>& ref) {
    size_t ties = 0, below = 0, above = 0;
    for (int h = 0; h < hv; ++h) {
        const float scale = scene.get<float>(19, {scene.begin, h}, ref[19]);
        require(scale > 0 && std::isfinite(scale), "invalid boundary scale");
        for (int d = 0; d < dim; ++d) {
            const float v = rad_bf16_to_f32(scene.get<uint16_t>(23, {scene.begin, h * dim + d}, ref[23]));
            const float ratio = v / scale;
            const float frac = std::abs(ratio) - std::floor(std::abs(ratio));
            if (frac == .5f) {
                ++ties;
                const auto code = scene.get<int8_t>(18, {scene.begin, h * dim + d}, ref[18]);
                require(int(code) == int(std::nearbyint(ratio)), "baseline does not round ties to even");
            }
            if (frac > .49f && frac < .5f) ++below;
            if (frac > .5f && frac < .51f) ++above;
        }
    }
    require(ties && below && above, "boundary fixture did not reach exact ties and both neighboring BF16 values");
    std::printf("BOUNDARY_PASS ties=%zu below=%zu above=%zu\n", ties, below, above);
}

struct Rows {
    const rad::KernelRow *conv, *recurrent, *quant, *commit;
};
void unfused(const Session& session, rad::Registry& reg, rad::Registry& baseline,
             const Rows& rows, const Scene& scene, Side& reference, const rad::Geometry& g) {
    auto conv = bind(baseline, rows.conv, g), recur = bind(baseline, rows.recurrent, g);
    rad::Geometry qg;
    qg.set_i("M", scene.end - scene.begin); qg.set_i("n", hv * dim); qg.set_i("group", dim);
    auto quant = bind(reg, rows.quant, qg);
    check(launch(conv, pick(reference, {0,1,2,3,4,5,6,7,20,21,22,16}), session.stream), "baseline conv");
    check(launch(recur, pick(reference, {20,21,22,8,9,10,11,12,13,5,6,14,15,23,17}), session.stream), "baseline recurrence");
    auto qt = pick(reference, {23,18,19});
    for (auto& t : qt) {
        t.data = static_cast<uint8_t*>(t.data) + size_t(scene.begin * t.stride[0]) * width(t.dtype);
        t.shape[0] = scene.end - scene.begin;
    }
    check(launch(quant, qt, session.stream), "baseline group128 quant");
}

size_t run_case(const Session& session, rad::Registry& reg, rad::Registry& baseline,
                const Rows& rows, const std::vector<int>& nodes, bool pitched, bool ring,
                bool boundary, bool absent_parent) {
    Scene scene = fixture(nodes, pitched, ring, boundary, absent_parent);
    // Upload BOTH arms before ANY launcher; host state/history can never alias an oracle write.
    Side actual(session, scene), reference(session, scene);
    const auto g = geometry(scene.sequences, ring, int(scene.buffers[0].tensor.shape[0]));
    auto fused = candidate(reg, g);
    unfused(session, reg, baseline, rows, scene, reference, g);
    check(launch(fused, std::vector<RadTensor>(actual.tensors.begin(), actual.tensors.begin() + 20), session.stream), "fused launch");
    const auto expected = reference.download(session, scene), result = actual.download(session, scene);
    const size_t bytes = compare(scene, result, expected, false, false);
    compare(scene, expected, expected, true, false);
    if (boundary) boundary_coverage(scene, expected);
    else {
        for (int t = scene.begin; t < scene.end; ++t) {
            require(scene.get<float>(19, {t, 0}, expected[19]) == 1.f, "zero group scale not one");
            for (int d = 0; d < dim; ++d)
                require(scene.get<int8_t>(18, {t, d}, expected[18]) == 0, "zero group code not zero");
        }
    }
    std::printf("COMPARE_PASS nodes=");
    for (int n : nodes) std::printf("%d,", n);
    std::printf(" hk=8 hv=24 group=128 pitched=%d ring=%d boundary=%d parent_absent=%d bytes=%zu tolerance=0\n",
                pitched, ring, boundary, absent_parent, bytes);
    return bytes;
}

size_t refusals(const Session& session, rad::Registry& reg) {
    size_t count = 0;
    const char* faults[] = {"parent-last-sequence", "negative-bos", "too-many-nodes", "history-slot",
                            "state-slot", "negative-context", "width", "group", "output-shape", "output-stride",
                            "output-input-alias", "output-output-alias"};
    for (size_t fault = 0; fault < sizeof(faults) / sizeof(faults[0]); ++fault) {
        Scene scene = fixture({8, 8}, true, false, false);
        auto g = geometry(scene.sequences, false, int(scene.buffers[0].tensor.shape[0]));
        if (fault == 0) scene.put(6, {scene.end - 1}, int32_t(999));
        if (fault == 1) scene.put(5, {0}, int32_t(-1));
        if (fault == 2) { g.set_i("q_len", 1); g.set_i("node_capacity", 1); }
        if (fault == 3) scene.put(4, {1, 0}, int32_t(99));
        if (fault == 4) scene.put(13, {1, 0}, int32_t(-1));
        if (fault == 5) scene.put(7, {1}, int32_t(-1));
        if (fault == 6) g.set_i("conv_width", 3);
        if (fault == 7) g.set_i("group", 64);
        if (fault == 8) --scene.buffers[19].tensor.shape[1];
        if (fault == 9) scene.buffers[18].tensor.stride[0] = 1;
        Side actual(session, scene);
        if (fault == 10) actual.tensors[18].data = actual.tensors[0].data;
        if (fault == 11) actual.tensors[19].data = actual.tensors[18].data;
        auto row = bind(reg, pin(reg, fused_op, selected_kernel, "mine"), g);
        const int rc = launch(row, std::vector<RadTensor>(actual.tensors.begin(), actual.tensors.begin() + 20), session.stream);
        // Device metadata verdicts may return RAD_OK after enqueueing a no-write kernel.
        if (fault >= 6) require(rc != RAD_OK, "invalid geometry/tensor accepted: " + std::string(faults[fault]));
        const auto out = actual.download(session, scene);
        compare(scene, out, out, false, true);
        std::printf("REFUSAL_PASS fault=%s launch_status=%d all_buffers_untouched=1\n", faults[fault], rc);
        ++count;
    }
    return count;
}

// Full FP32 mantissas in state/weights, rather than periodic binary fractions.
void randomize(Scene& scene,uint32_t seed) {
    for (int id:{0,3}) scene.fill(id,2.f,1,seed+uint32_t(id));
    for (int id:{1,2,12}) scene.fill(id,.25f,1,seed+uint32_t(id));
    for (int id:{8,9,14}) scene.fill(id,4.f,1,seed+uint32_t(id));
    for (int id:{10,11,15}) scene.fill(id,1.f,1,seed+uint32_t(id));
}

// The old Bole failures were stateful: verify and commit repeatedly, then move
// cancellation below any fixed prefix. All comparisons here are byte-exact.
// The 36/13-key aligned anchors make the serial FP32 dot round to an exact
// power of two even though the real squared norm differs from one. Beta-zero
// prefixes move that cancellation deeper; the oracle must actually reach zero.
void stress_case(const Session& session, rad::Registry& reg, rad::Registry& baseline,
                 const Rows& rows, const std::vector<int>& nodes, bool ring, bool chain,
                 int cols=0, int cancel_node=0, bool extreme=false, bool zero_decay=false,
                 bool negative=false, uint32_t seed=0) {
    Scene scene=fixture(nodes,true,ring,false,chain);
    if (seed) randomize(scene,seed);
    if (extreme) {
        scene.fill(8,0,1);
        for (int tok=scene.begin;tok<scene.end;++tok) for (int h=0;h<hv;++h)
            scene.put(9,{tok,h},rad_f32_to_bf16(-100.f));
        for (int h=0;h<hv;++h) { scene.put(10,{h},0.f);scene.put(11,{h},69.0775528f); }
        const int slots=int(scene.buffers[12].tensor.shape[0]);
        for (int slot=0;slot<slots;++slot) for (int h=0;h<hv;++h)
            for (int row=0;row<128;++row) for (int col=0;col<128;++col)
                scene.put(12,{slot,h,row,col},1e30f);
    }
    if (cols) {
        for (int id:{0,1,2,3,8,12}) scene.fill(id,0,1);
        for (int h=0;h<hk;++h) for (int col=0;col<cols;++col) {
            scene.put(2,{h*128+col},rad_f32_to_bf16(1.f));
            scene.put(2,{(hk+h)*128+col},rad_f32_to_bf16(1.f));
        }
        for (int h=0;h<hv;++h) { scene.put(10,{h},0.f);scene.put(11,{h},0.f); }
        int bos=scene.begin;
        for (int n:nodes) {
            for (int node=0;node<n;++node) for (int h=0;h<hv;++h)
                scene.put(9,{bos+node,h},rad_f32_to_bf16(node<cancel_node ? -200.f : 100.f));
            bos+=n;
        }
    }
    if (zero_decay) {
        scene.fill(9,0,1);
        for (int h=0;h<hv;++h) { scene.put(10,{h},0.f);scene.put(11,{h},0.f); }
        for (int tok=scene.begin;tok<scene.end;++tok) for (int h=0;h<hv;++h)
            scene.put(8,{tok,h},rad_f32_to_bf16(tok%2 ? -200.f : 200.f));
    }
    if (negative) {
        scene.fill(1,0,1); // The wrong parent must affect recurrence, not convolution.
        for (int ch=0;ch<channels;++ch) scene.put(1,{ch,3},rad_f32_to_bf16(1.f));
    }
    scene.add("accepted_path",RAD_I32,{scene.sequences,16});
    scene.add("accepted_length",RAD_I32,{scene.sequences});
    Scene actual_scene=scene;
    if (negative) {
        int bos=scene.begin;
        for (int n:nodes) {
            for (int node=0;node<n;++node) actual_scene.put(6,{bos+node},int32_t(node-1));
            bos+=n;
        }
    }
    Side actual(session,actual_scene),reference(session,scene);
    auto g=geometry(scene.sequences,ring,int(scene.buffers[0].tensor.shape[0]));
    g.set_f("scale",1.0/std::sqrt(128.0));
    auto upload=[&](Side& side,int id) {
        const auto& bytes=scene.buffers[size_t(id)].initial;
        check(rad_memcpy_async(side.allocations[size_t(id)],bytes.data(),int64_t(bytes.size()),session.stream),"stress upload");
    };
    if (cols) {
        unfused(session,reg,baseline,rows,scene,reference,g);
        const auto probe=reference.download(session,scene);
        const int slots=int(scene.buffers[12].tensor.shape[0]);
        for (int h=0;h<hv;++h) for (int col=0;col<128;++col) {
            const float anchor=cols==1 ? (col==0 ? 1048576.f : 0.f) :
                std::ldexp(scene.get<float>(17,{scene.begin,h,col},probe[17]),20);
            for (int slot=0;slot<slots;++slot) for (int row=0;row<128;++row)
                scene.put(12,{slot,h,row,col},anchor);
        }
        actual_scene.buffers[12].initial=scene.buffers[12].initial;
        upload(actual,12);upload(reference,12);
    }
    auto fused=candidate(reg,g);
    auto commit=bind(reg,pin(reg,"gdn_tree_commit","mine_gdn_tree_commit","mine"),g);
    auto old_commit=bind(baseline,rows.commit,g);
    for (int round=0;round<4;++round) {
        unfused(session,reg,baseline,rows,scene,reference,g);
        check(launch(fused,std::vector<RadTensor>(actual.tensors.begin(),actual.tensors.begin()+20),session.stream),"stress verify");
        auto result=actual.download(session,actual_scene),expected=reference.download(session,scene);
        if (cols>1 && !round) {
            for (int h=0;h<hv;++h) {
                require(scene.get<float>(19,{scene.begin+cancel_node,h},expected[19])==1.f,"oracle did not reach cancellation");
                for (int col=0;col<128;++col)
                    require(scene.get<int8_t>(18,{scene.begin+cancel_node,h*128+col},expected[18])==0,"oracle cancellation code nonzero");
                if (cancel_node && h)
                    require(scene.get<float>(19,{scene.begin,h},expected[19])!=1.f,"oracle prefix already zero");
            }
        }
        compare(actual_scene,result,expected,false,false);
        compare(scene,expected,expected,true,false);
        int bos=scene.begin;
        for (int seq=0;seq<scene.sequences;++seq) {
            const int n=nodes[size_t(seq)],endpoint=(round==0 || round==3) ? 0 : (round==1 ? n-1 : n/2);
            std::vector<int> path;
            for (int p=endpoint;p>=0;p=chain ? p-1 : scene.get<int32_t>(6,{bos+p},scene.buffers[6].initial)) path.push_back(p);
            std::reverse(path.begin(),path.end());
            for (size_t j=0;j<path.size();++j) scene.put(24,{seq,int64_t(j)},int32_t(path[j]));
            scene.put(25,{seq},int32_t(path.size()));
            bos+=n;
        }
        for (int id:{24,25}) {
            actual_scene.buffers[size_t(id)].initial=scene.buffers[size_t(id)].initial;
            upload(actual,id);upload(reference,id);
        }
        check(launch(commit,pick(actual,{12,13,17,3,4,16,24,25,7,5,6}),session.stream),"stress commit");
        check(launch(old_commit,pick(reference,{12,13,17,3,4,16,24,25,7,5,6}),session.stream),"oracle commit");
        result=actual.download(session,actual_scene);expected=reference.download(session,scene);
        for (int id:{3,12}) {
            require(result[size_t(id)]==expected[size_t(id)],"committed state/history differs");
            scene.buffers[size_t(id)].initial=actual_scene.buffers[size_t(id)].initial=expected[size_t(id)];
        }
        for (int seq=0;seq<scene.sequences;++seq)
            scene.put(7,{seq},int32_t(scene.get<int32_t>(7,{seq},scene.buffers[7].initial)+
                                     scene.get<int32_t>(25,{seq},scene.buffers[25].initial)));
        actual_scene.buffers[7].initial=scene.buffers[7].initial;
        upload(actual,7);upload(reference,7);
    }
    std::printf("EXACT_STRESS_PASS nodes=%d sequences=%d ring=%d chain=%d cols=%d cancel_node=%d extreme=%d zero_decay=%d seed=%u commits=4 tolerance=0\n",
                nodes.back(),scene.sequences,ring,chain,cols,cancel_node,extreme,zero_decay,seed);
}

void stress(const Session& session,rad::Registry& reg,rad::Registry& baseline,const Rows& rows,bool negative) {
    if (negative) {
        stress_case(session,reg,baseline,rows,{16},false,false,0,0,false,false,true);
        throw std::runtime_error("wrong-parent negative control did not fail");
    }
    for (int n:{1,7,16}) for (bool chain:{false,true})
        stress_case(session,reg,baseline,rows,{n},false,chain);
    stress_case(session,reg,baseline,rows,{7,1,16},true,false);
    stress_case(session,reg,baseline,rows,{16},true,true,0,0,true);
    stress_case(session,reg,baseline,rows,{16},false,true,1);
    stress_case(session,reg,baseline,rows,{16},true,false,0,0,false,true);
    for (int cols:{36,13}) for (int node:{0,1,3,7,15}) for (bool chain:{false,true})
        stress_case(session,reg,baseline,rows,{16},false,chain,cols,node);
    for (uint32_t seed:{1u,0x12345678u}) for (int n:{8,16}) for (bool chain:{false,true})
        stress_case(session,reg,baseline,rows,{n},true,chain,0,0,false,false,false,seed);
    std::puts("EXACT_STRESS_GPU_PASS cases=38 commits=152 tolerance=0 oracle=pinned-original");
}

// Wide (<=65 node) exact verification. The frozen plugin only accepts 16 nodes, so
// each target's root-inclusive ancestor path is replayed as linear <=16-node chunks:
// pinned old conv+recurrence+libr4d quant, then pinned old commit carries FP32 state,
// convolution history and ctx into the next chunk. Every node of every sequence is a
// prefix of some leaf path, so all candidate rows get an independent expected value.
constexpr int wide_cap = 65;
struct Slot { std::vector<uint8_t> state, history; int ctx = 0; };
using Expect = std::vector<std::vector<std::vector<uint8_t>>>; // [token][output 16..19]

// Axis-0 entry (token row or slot) including pitch padding; identical in two pitched scenes.
std::vector<uint8_t> slab(const Scene& s,int id,int row,const std::vector<uint8_t>& bytes) {
    const auto& t=s.buffers[size_t(id)].tensor;
    const auto from=bytes.begin()+std::ptrdiff_t(s.at(id,{row}));
    return std::vector<uint8_t>(from,from+std::ptrdiff_t(size_t(t.stride[0])*width(t.dtype)));
}
void put_slab(const Scene& s,int id,int row,const std::vector<uint8_t>& v,std::vector<uint8_t>& bytes) {
    const auto& t=s.buffers[size_t(id)].tensor;
    require(v.size()==size_t(t.stride[0])*width(t.dtype),"oracle/candidate slab layout differs");
    std::memcpy(bytes.data()+s.at(id,{row}),v.data(),v.size());
}
int parent_of(const Scene& s,int bos,int node) {
    return s.buffers[6].tensor.rank ? s.get<int32_t>(6,{bos+node},s.buffers[6].initial) : node-1;
}
std::vector<int> ancestry(const Scene& s,int bos,int node) {
    std::vector<int> path;
    for (int p=node;p>=0;p=parent_of(s,bos,p)) path.push_back(bos+p);
    std::reverse(path.begin(),path.end());
    return path;
}
Slot slot_of(const Scene& s,int seq) {
    const auto& b=s.buffers;
    return {slab(s,12,s.get<int32_t>(13,{seq,0},b[13].initial),b[12].initial),
            slab(s,3,s.get<int32_t>(4,{seq,0},b[4].initial),b[3].initial),
            s.get<int32_t>(7,{seq},b[7].initial)};
}

// Expected rows for `path` (absolute rows of `wide`) from the frozen 16-node GPU plugin.
// Returns the committed slot after the whole path when commit_all, else after the last full chunk.
Slot frozen_path(const Session& session,rad::Registry& reg,rad::Registry& baseline,const Rows& rows,
                 const Scene& wide,bool ring,const std::vector<int>& path,Slot slot,Expect& want,
                 bool commit_all,int& chunks) {
    for (size_t at=0;at<path.size();at+=16) {
        const int len=int(std::min<size_t>(16,path.size()-at));
        const bool commit=commit_all || at+size_t(len)<path.size();
        Scene o=fixture({len},true,ring,false,true); // absent parent: linear chunk
        for (int id:{1,2,10,11,15}) {
            require(o.buffers[size_t(id)].initial.size()==wide.buffers[size_t(id)].initial.size(),
                    "oracle weight layout differs");
            o.buffers[size_t(id)].initial=wide.buffers[size_t(id)].initial;
        }
        for (int j=0;j<len;++j) for (int id:{0,8,9,14})
            put_slab(o,id,o.begin+j,slab(wide,id,path[at+size_t(j)],wide.buffers[size_t(id)].initial),
                     o.buffers[size_t(id)].initial);
        put_slab(o,12,0,slot.state,o.buffers[12].initial);  // fixture state_idx[0]=0
        put_slab(o,3,1,slot.history,o.buffers[3].initial);  // fixture history_idx[0]=1
        o.put(7,{0},int32_t(slot.ctx));
        o.add("accepted_path",RAD_I32,{1,16});
        o.add("accepted_length",RAD_I32,{1});
        for (int j=0;j<len;++j) o.put(24,{0,j},int32_t(j));
        o.put(25,{0},int32_t(len));
        Side side(session,o);
        const auto g=geometry(1,ring,int(o.buffers[0].tensor.shape[0]));
        unfused(session,reg,baseline,rows,o,side,g);
        if (commit)
            check(launch(bind(baseline,rows.commit,g),pick(side,{12,13,17,3,4,16,24,25,7,5,6}),session.stream),
                  "frozen chunk commit");
        const auto out=side.download(session,o);
        ++chunks;
        for (int j=0;j<len;++j) {
            for (int h=0;h<hv;++h) {
                const float scale=o.get<float>(19,{o.begin+j,h},out[19]);
                require(scale>0 && std::isfinite(scale),"frozen oracle row not written");
            }
            std::vector<std::vector<uint8_t>> row;
            for (int id=16;id<20;++id) row.push_back(slab(o,id,o.begin+j,out[size_t(id)]));
            auto& w=want[size_t(path[at+size_t(j)])];
            if (w.empty()) w=std::move(row);
            else require(w==row,"frozen paths disagree on a shared ancestor");
        }
        if (commit) {
            slot.state=slab(o,12,0,out[12]);
            slot.history=slab(o,3,1,out[3]);
            slot.ctx+=len;
        }
    }
    return slot;
}

// One wide transaction batch: `rounds` verify+commit cycles; every node row of every round
// byte-compared, committed state/history byte-compared. absent=true uses a null parent
// (trees must be chains). negative rewires the last node of the last sequence to a wrong
// but valid parent in the candidate only; it must fail in compare after the wide launch.
size_t wide_case(const Session& session,rad::Registry& reg,rad::Registry& baseline,const Rows& rows,
                 const std::vector<std::vector<int>>& trees,bool ring,bool absent,uint32_t seed,
                 int rounds,bool negative) {
    std::vector<int> nodes;
    for (const auto& t:trees) nodes.push_back(int(t.size()));
    Scene scene=fixture(nodes,true,ring,false,absent);
    randomize(scene,seed);
    int bos=scene.begin;
    for (const auto& tree:trees) {
        for (int i=0;i<int(tree.size());++i) {
            require(i ? tree[size_t(i)]>=0 && tree[size_t(i)]<i : tree[0]==-1,"invalid wide tree");
            require(!absent || tree[size_t(i)]==i-1,"absent parent requires a chain");
            if (!absent) scene.put(6,{bos+i},int32_t(tree[size_t(i)]));
        }
        bos+=int(tree.size());
    }
    scene.add("accepted_path",RAD_I32,{scene.sequences,wide_cap});
    scene.add("accepted_length",RAD_I32,{scene.sequences});
    Scene wrong;
    if (negative) {
        require(!absent,"negative control needs explicit parents");
        wrong=scene;
        const int p=scene.get<int32_t>(6,{scene.end-1},scene.buffers[6].initial);
        require(p>0,"negative control needs a non-root parent");
        wrong.put(6,{scene.end-1},int32_t(p-1));
    }
    const Scene& live=negative ? wrong : scene;
    Side actual(session,live);
    auto g=geometry(scene.sequences,ring,int(scene.buffers[0].tensor.shape[0]));
    g.set_i("q_len",wide_cap);g.set_i("node_capacity",wide_cap);
    auto fused=candidate(reg,g);
    auto commit=bind(reg,pin(reg,"gdn_tree_commit","mine_gdn_tree_commit","mine"),g);
    auto upload=[&](int id) {
        const auto& bytes=scene.buffers[size_t(id)].initial;
        check(rad_memcpy_async(actual.allocations[size_t(id)],bytes.data(),int64_t(bytes.size()),session.stream),"wide upload");
    };
    size_t verified=0;
    int chunks=0,longest=0;
    for (int round=0;round<rounds;++round) {
        Expect want(size_t(scene.buffers[0].tensor.shape[0]));
        std::vector<Slot> base;
        bos=scene.begin;
        for (int seq=0;seq<scene.sequences;++seq) {
            const int n=nodes[size_t(seq)];
            base.push_back(slot_of(scene,seq));
            std::vector<bool> inner(size_t(n),false);
            for (int i=1;i<n;++i) inner[size_t(parent_of(scene,bos,i))]=true;
            for (int leaf=0;leaf<n;++leaf) if (!inner[size_t(leaf)]) {
                const auto path=ancestry(scene,bos,leaf);
                longest=std::max(longest,int(path.size()));
                frozen_path(session,reg,baseline,rows,scene,ring,path,base.back(),want,false,chunks);
            }
            bos+=n;
        }
        check(launch(fused,std::vector<RadTensor>(actual.tensors.begin(),actual.tensors.begin()+20),session.stream),"wide verify");
        const auto result=actual.download(session,live);
        std::vector<std::vector<uint8_t>> expected;
        for (const auto& b:live.buffers) expected.push_back(b.initial);
        for (int t=scene.begin;t<scene.end;++t) {
            require(want[size_t(t)].size()==4,"frozen oracle missed a wide node");
            for (int k=0;k<4;++k) put_slab(scene,16+k,t,want[size_t(t)][size_t(k)],expected[size_t(16+k)]);
            ++verified;
        }
        if (negative)
            std::printf("WIDE_NEGATIVE_REACHED kernel=%s nodes=%d wrong_parent_node=%d launch_status=0\n",
                        fused.row->info->name,nodes.back(),nodes.back()-1);
        compare(live,result,expected,false,false);
        if (negative) throw std::runtime_error("wide wrong-parent negative control did not fail");
        std::vector<Slot> committed;
        bos=scene.begin;
        for (int seq=0;seq<scene.sequences;++seq) {
            const int n=nodes[size_t(seq)],endpoint=round==0 ? n-1 : round==1 ? n/2 : 0;
            const auto path=ancestry(scene,bos,endpoint);
            for (size_t j=0;j<path.size();++j) scene.put(24,{seq,int64_t(j)},int32_t(path[j]-bos));
            scene.put(25,{seq},int32_t(path.size()));
            committed.push_back(frozen_path(session,reg,baseline,rows,scene,ring,path,base[size_t(seq)],want,true,chunks));
            require(committed.back().ctx==base[size_t(seq)].ctx+int(path.size()),"frozen commit did not cover path");
            bos+=n;
        }
        upload(24);upload(25);
        auto before=actual.download(session,scene);
        check(launch(commit,pick(actual,{12,13,17,3,4,16,24,25,7,5,6}),session.stream),"wide commit");
        const auto after=actual.download(session,scene);
        for (int seq=0;seq<scene.sequences;++seq) {
            put_slab(scene,12,scene.get<int32_t>(13,{seq,0},scene.buffers[13].initial),committed[size_t(seq)].state,before[12]);
            put_slab(scene,3,scene.get<int32_t>(4,{seq,0},scene.buffers[4].initial),committed[size_t(seq)].history,before[3]);
            scene.put(7,{seq},int32_t(committed[size_t(seq)].ctx));
        }
        for (size_t id=0;id<after.size();++id)
            require(after[id]==before[id],scene.buffers[id].name+" differs after wide commit");
        for (int id:{3,12}) scene.buffers[size_t(id)].initial=after[size_t(id)];
        upload(3);upload(7);upload(12);
    }
    std::printf("WIDE_PASS nodes=");
    for (int n:nodes) std::printf("%d,",n);
    std::printf(" ring=%d absent_parent=%d seed=%u rounds=%d rows=%zu longest_path=%d frozen_chunks=%d tolerance=0\n",
                ring,absent,seed,rounds,verified,longest,chunks);
    return verified;
}

// Capacity 66 must not select or launch; legacy serial rows must refuse 65.
size_t wide_refusals(const Session& session,rad::Registry& reg) {
    size_t count=0;
    const char* faults[]={"select-cap66","exact-direct-cap66","commit-direct-cap66","legacy-conv-65","legacy-recurrent-65"};
    for (int fault=0;fault<5;++fault) {
        Scene scene=fixture({wide_cap},true,false,false);
        scene.add("accepted_path",RAD_I32,{1,wide_cap+1});
        scene.add("accepted_length",RAD_I32,{1});
        scene.put(24,{0,0},int32_t(0));scene.put(25,{0},int32_t(1));
        auto g=geometry(1,false,int(scene.buffers[0].tensor.shape[0]));
        const int cap=fault<3 ? wide_cap+1 : wide_cap;
        g.set_i("q_len",cap);g.set_i("node_capacity",cap);
        Side actual(session,scene);
        int rc=RAD_E_UNSUPPORTED;
        if (fault==0) {
            rad::SelectionTrace trace;
            require(!reg.resolve(fused_op,g,RAD_DOMAIN_DEVICE,&trace).row,"capacity 66 selected a device row");
        } else if (fault==1) {
            rc=launch(bind(reg,pin(reg,fused_op,selected_kernel,"mine"),g),
                      std::vector<RadTensor>(actual.tensors.begin(),actual.tensors.begin()+20),session.stream);
        } else if (fault==2) {
            rc=launch(bind(reg,pin(reg,"gdn_tree_commit","mine_gdn_tree_commit","mine"),g),
                      pick(actual,{12,13,17,3,4,16,24,25,7,5,6}),session.stream);
        } else if (fault==3) {
            rc=launch(bind(reg,pin(reg,"gdn_conv_tree","mine_gdn_conv_tree","mine"),g),
                      pick(actual,{0,1,2,3,4,5,6,7,20,21,22,16}),session.stream);
        } else {
            rc=launch(bind(reg,pin(reg,"gdn_recurrent_tree","mine_gdn_recurrent_tree","mine"),g),
                      pick(actual,{20,21,22,8,9,10,11,12,13,5,6,14,15,23,17}),session.stream);
        }
        require(rc!=RAD_OK,std::string("wide geometry accepted: ")+faults[fault]);
        const auto out=actual.download(session,scene);
        compare(scene,out,out,false,true);
        std::printf("WIDE_REFUSAL_PASS fault=%s launch_status=%d all_buffers_untouched=1\n",faults[fault],rc);
        ++count;
    }
    return count;
}

void wide(const Session& session,rad::Registry& reg,rad::Registry& baseline,const Rows& rows,bool negative) {
    auto tree=[](int n,char kind) {
        std::vector<int> p(size_t(n),-1);
        for (int i=1;i<n;++i) {
            const int round=(i-1)/8,k=(i-1)%8,prev=1+(round-1)*8; // MTP8: 8 proposals per depth
            p[size_t(i)]=kind=='h' ? (i-1)/2 : kind=='d' ? (i%8 ? i-1 : i-5) : kind=='c' ? i-1 :
                         round==0 ? 0 : prev+(k%4);
        }
        return p;
    };
    if (negative) {
        // Node 64's true path is >16 nodes deep, so the oracle is chunked.
        wide_case(session,reg,baseline,rows,{tree(wide_cap,'d')},false,false,1u,1,true);
        throw std::runtime_error("wide wrong-parent negative control did not fail");
    }
    size_t rows_verified=0;
    rows_verified+=wide_case(session,reg,baseline,rows,{tree(wide_cap,'m')},false,false,1u,3,false);
    rows_verified+=wide_case(session,reg,baseline,rows,{tree(33,'h'),tree(wide_cap,'d'),tree(1,'c')},
                             true,false,0x12345678u,3,false);
    rows_verified+=wide_case(session,reg,baseline,rows,{tree(wide_cap,'c')},false,true,7u,3,false);
    rows_verified+=wide_case(session,reg,baseline,rows,{tree(wide_cap,'c')},true,true,0x9e3779b9u,3,false);
    const size_t refused=wide_refusals(session,reg);
    require(rows_verified==size_t(3*(65+33+65+1+65+65)) && refused==5,"zero/incomplete wide coverage");
    std::printf("WIDE_GPU_PASS cases=4 rows=%zu commits=12 refusals=%zu tolerance=0 oracle=frozen16-chunked\n",
                rows_verified,refused);
}

// Same synchronized steady-clock window as rad-kbench: dispatch latency, not
// isolated hardware duration. Alternate arm order and report every repetition.
void benchmark(const Session& session,rad::Registry& reg,rad::Registry& baseline) {
    constexpr int iters=64,reps=9;
    for (int n:{4,8,16}) for (bool chain:{false,true}) {
        Scene scene=fixture({n},false,false,false,chain);
        Side actual(session,scene),reference(session,scene);
        auto g=geometry(1,false,n);
        g.set_i("q_len",n);g.set_i("node_capacity",n);
        auto row=candidate(reg,g);
        auto old=bind(baseline,pin(baseline,fused_op,"mine_gdn_conv_recurrent_tree_i8g","mine"),g);
        auto tensors=std::vector<RadTensor>(actual.tensors.begin(),actual.tensors.begin()+20);
        auto old_tensors=std::vector<RadTensor>(reference.tensors.begin(),reference.tensors.begin()+20);
        auto measure=[&](const rad::Resolved& r,const std::vector<RadTensor>& t) {
            RadArgs a{};a.t=t.data();a.n_t=int(t.size());a.p=r.geom.params();a.n_p=r.geom.n_params();
            a.instance=r.instance;a.rank=0;a.world_size=1;
            const auto start=std::chrono::steady_clock::now();
            for (int i=0;i<iters;++i) check(r.row->info->launch(&a,session.stream),"timed launch");
            check(rad_stream_sync(session.stream),"timed synchronization");
            return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iters;
        };
        const auto warm=std::chrono::steady_clock::now();
        while (std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-warm).count()<100) {
            measure(old,old_tensors);measure(row,tensors);
        }
        std::vector<double> a,b;
        for (int rep=0;rep<reps;++rep) {
            double x,y;
            if (rep%2) { y=measure(row,tensors);x=measure(old,old_tensors); }
            else { x=measure(old,old_tensors);y=measure(row,tensors); }
            a.push_back(x);b.push_back(y);
            std::printf("BENCH_SAMPLE nodes=%d chain=%d rep=%d baseline_us=%.6f candidate_us=%.6f iterations=%d\n",n,chain,rep,x,y,iters);
        }
        std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
        std::printf("BENCH_MEDIAN nodes=%d chain=%d baseline_us=%.6f candidate_us=%.6f baseline_spread_pct=%.3f candidate_spread_pct=%.3f scope=synchronized-launch\n",
                    n,chain,a[reps/2],b[reps/2],100*(a.back()-a.front())/a[reps/2],100*(b.back()-b.front())/b[reps/2]);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string mine = "./mine.so", libr4d = "./libr4d.so", old;
        bool exact=false, run_stress=false, negative=false, bench=false, run_wide=false, wide_negative=false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::puts("gdn-tree-fusion-gpu-test [--mine PATH] [--libr4d PATH] [--baseline OLD_MINE_SO] [--exact] [--stress] [--negative-flat] [--bench] [--wide] [--wide-negative]");
                return 0;
            }
            if (arg=="--exact") { exact=true;continue; }
            if (arg=="--stress") { run_stress=true;continue; }
            if (arg=="--negative-flat") { negative=true;continue; }
            if (arg=="--bench") { bench=true;continue; }
            if (arg=="--wide") { run_wide=true;continue; }
            if (arg=="--wide-negative") { wide_negative=true;continue; }
            require((arg == "--mine" || arg == "--libr4d" || arg == "--baseline") && i + 1 < argc,
                    "unknown/incomplete argument: " + arg);
            if (arg == "--mine") mine = argv[++i];
            else if (arg == "--libr4d") libr4d = argv[++i];
            else old = argv[++i];
        }
        if (exact) selected_kernel="mine_gdn_bole_exact_i8g";
        require(!(run_wide || wide_negative) || exact, "--wide/--wide-negative require --exact");
        require(!(exact || run_stress || negative || bench || run_wide || wide_negative) ||
                    (!old.empty() && !std::filesystem::equivalent(old,mine)),
                "exact/stress/bench/wide runs require a separate frozen --baseline");
        Session session;
        rad::Registry reg, baseline;
        reg.set_code_device(0); baseline.set_code_device(0);
        check(reg.load_plugin(mine, 0), "load candidate mine");
        check(reg.load_plugin(libr4d, 1), "load libr4d");
        check(reg.check_complete(), "candidate schemas");
        check(baseline.load_plugin(old.empty() ? mine : old, 0), "load separate baseline mine registry");
        check(baseline.check_complete(), "baseline schemas");
        require(reg.plugin("mine") && reg.plugin("libr4d") && baseline.plugin("mine"), "plugin identity mismatch");
        std::printf("PLUGIN candidate=%s baseline=%s quant=%s baseline_mode=%s\n", mine.c_str(),
                    (old.empty() ? mine : old).c_str(), libr4d.c_str(), old.empty() ? "retained-unfused" : "pinned-old-plugin");
        Rows rows{pin(baseline, "gdn_conv_tree", "mine_gdn_conv_tree", "mine"),
                  pin(baseline, "gdn_recurrent_tree", "mine_gdn_recurrent_tree", "mine"),
                  pin(reg, "quant_act_i8g", "quant_act_i8g", "libr4d"),
                  pin(baseline, "gdn_tree_commit", "mine_gdn_tree_commit", "mine")};
        if (negative) { stress(session,reg,baseline,rows,true);return 1; }
        if (wide_negative) { wide(session,reg,baseline,rows,true);return 1; }
        size_t cases = 0, bytes = 0;
        for (int n : {1, 8, 16}) {
            bytes += run_case(session, reg, baseline, rows, {n}, false, false, false, false); ++cases;
            bytes += run_case(session, reg, baseline, rows, {n, 1, n}, true, true, false, false); ++cases;
        }
        bytes += run_case(session, reg, baseline, rows, {1}, false, false, true, false); ++cases;
        bytes += run_case(session, reg, baseline, rows, {8, 16}, true, false, false, true); ++cases;
        bytes += run_case(session, reg, baseline, rows, {0, 7, 0, 1}, true, true, false, false); ++cases;
        const size_t refused = refusals(session, reg);
        require(cases == 9 && bytes > 0 && refused == 12, "zero/incomplete selected coverage");
        std::printf("GDN_TREE_FUSION_GPU_PASS cases=%zu refusals=%zu bytes=%zu tolerance=0 oracle=GPU finite-cases-only\n",
                    cases, refused, bytes);
        if (run_stress) stress(session,reg,baseline,rows,false);
        if (run_wide) wide(session,reg,baseline,rows,false);
        if (bench) benchmark(session,reg,baseline);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GDN_TREE_FUSION_GPU_FAIL: %s\n", error.what());
        return 1;
    }
}
