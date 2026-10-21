// Compare two ACTUAL device plugins, never a reimplementation of the coupling law.
// Run from the worktree root; build/run argv in README.md. No engine registration.
#include <hip/hip_runtime.h>
#include "../../../mine/spechub.h"
#ifdef HC_PROBE_AQL
#include "rad_device.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void hip(hipError_t rc, const char* why) {
    if (rc != hipSuccess) throw std::runtime_error(std::string(why) + ": " + hipGetErrorString(rc));
}
struct Plugin {
    void* handle;
    const RadKernelInfo* row = nullptr;
    Plugin(const char* path, const char* name) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        require(handle != nullptr, std::string("dlopen ") + path + ": " + (handle ? "" : dlerror()));
        auto abi = reinterpret_cast<uint32_t (*)()>(dlsym(handle, "rad_plugin_abi_version"));
        auto count = reinterpret_cast<int (*)()>(dlsym(handle, "rad_kernel_count"));
        auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(handle, "rad_kernel_at"));
        require(abi && count && at && abi() == RAD_ABI_VERSION, "plugin ABI/exports mismatch");
        for (int i = 0; i < count(); ++i) {
            const auto* r = at(i);
            if (r && r->name && !std::strcmp(r->name, name)) {
                require(!row, "duplicate row"); row = r;
            }
        }
        require(row && row->launch && row->op && !std::strcmp(row->op, "mtp_couple_verify") &&
                row->domain == RAD_DOMAIN_DEVICE, "actual MTP verify row not loaded");
        Dl_info origin{};
        char expected[PATH_MAX], actual[PATH_MAX];
        require(dladdr(reinterpret_cast<const void*>(row->launch), &origin) && origin.dli_fname &&
                realpath(path, expected) && realpath(origin.dli_fname, actual) &&
                !std::strcmp(expected, actual), "row launch not from requested plugin");
        std::printf("loaded=%s row=%s origin=%s\n", path, name, actual);
    }
    ~Plugin() { if (handle) dlclose(handle); }
    Plugin(const Plugin&) = delete;
    Plugin& operator=(const Plugin&) = delete;
};
struct Buffer {
    void* ptr = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n) { hip(hipMalloc(&ptr, n), "hipMalloc"); }
    ~Buffer() { if (ptr && hipFree(ptr) != hipSuccess) std::fprintf(stderr, "hipFree failed\n"); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
constexpr uint32_t Guard = 0xdeadbeefu;
constexpr int Pad = 16;
constexpr int Vocab = 2048;
struct Row {
    std::vector<int32_t> ids;
    std::vector<float> val;
    RadMtpCoupleState state{};
    RadSampleParams params{};
    const char* label;
    bool applicable = true;
};
RadMtpCoupleState state(bool cancellation = false) {
    int32_t ids[8] = {0,1,2,3,4,5,6,7};
    float val[8] = {0,0,-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    if (cancellation) {
        val[1] = static_cast<float>(-std::log(2.0));
        for (int i = 2; i < 8; ++i) val[i] = -37;
    }
    RadMtpCoupleState s{};
    s.status = 1; s.round = 1; s.draw.mode = 1;
    require(rad_mtp_couple_build_q(ids, val, Vocab, &s), "fixture Q construction failed");
    s.x = 0;
    return s;
}
Row make(int n, uint64_t seed, int x = 0, bool cancellation = false) {
    Row r;
    r.ids.assign(n, -1); r.val.assign(n, -INFINITY);
    r.ids[0] = 0; r.val[0] = -1;
    if (n > 1) { r.ids[1] = 1; r.val[1] = 0; }
    if (n > 2) { r.ids[2] = 2; r.val[2] = 0; }
    if (n > 3) { r.ids[3] = 7; r.val[3] = -2; }
    r.state = state(cancellation); r.state.x = x;
    r.params.seed = r.state.draw.seed = seed;
    r.params.pos = r.state.pos = 5;
    r.label = "ordinary";
    return r;
}
// Dense, finite, unique target IDs; Q IDs span the entire ordered target.
// Deliberately favor or disfavor the bound proposal without deriving expected tokens.
Row dense(int n, uint64_t seed, bool favor) {
    Row r = make(n, seed);
    int32_t qids[8];
    float qval[8];
    for (int j = 0; j < n; ++j) {
        const uint32_t h = static_cast<uint32_t>(seed) * 747796405u +
                           static_cast<uint32_t>(j) * 2891336453u;
        r.ids[j] = 1 + static_cast<int32_t>((static_cast<uint32_t>(j) * 1093u +
                                              static_cast<uint32_t>(seed) * 17u) % (Vocab - 1));
        r.val[j] = -7.f + static_cast<float>((h ^ (h >> 16)) % 500u) / 100.f;
    }
    for (int k = 0; k < 8; ++k) {
        qids[k] = r.ids[k * (n - 1) / 7];
        qval[k] = -2.f + static_cast<float>((seed + 3 * k) % 9) / 3.f;
    }
    qval[0] = favor ? -1.f : 2.f;
    r.val[0] = favor ? 5.f : -14.f;
    r.state = {};
    r.state.status = 1; r.state.round = 1; r.state.draw.mode = 1;
    require(rad_mtp_couple_build_q(qids, qval, Vocab, &r.state), "dense Q construction failed");
    r.state.x = qids[0];
    r.state.draw.seed = r.params.seed = seed;
    r.state.pos = r.params.pos = 5;
    r.label = favor ? "dense-favored" : "dense-disfavored";
    return r;
}
RadTensor tensor(void* p, int dtype, int rows, int cols, int pitch) {
    RadTensor t{};t.data = p;t.dtype = dtype;t.rank = 2;
    t.shape[0] = rows;t.shape[1] = cols;t.stride[0] = pitch;t.stride[1] = 1;
    return t;
}
struct Arm {
    std::array<std::vector<uint32_t>,5> host;
    std::array<Buffer*,5> dev{};
    std::array<RadTensor,5> tensors{};
    RadParam params[3] = {RAD_INT("M",1), RAD_INT("n_cand",1), RAD_INT("n_vocab",Vocab)};
    RadArgs args{};
    Arm(const std::vector<Row>& rows, int n) {
        const int M = static_cast<int>(rows.size());
        const int pitch[5] = {n+4, n+4, 26, 40, 2};
        const int width[5] = {n, n, 24, RAD_MTP_COUPLE_STATE_WORDS, 1};
        const int dt[5] = {RAD_I32, RAD_F32, RAD_U32, RAD_U32, RAD_I32};
        params[0].ival = M; params[1].ival = n;
        for (int i = 0; i < 5; ++i) {
            host[i].assign(2 * Pad + M * pitch[i], Guard);
            for (int r = 0; r < M; ++r) {
                void* dst = host[i].data() + Pad + r * pitch[i];
                const Row& row = rows[r];
                if (i == 0) std::memcpy(dst, row.ids.data(), n * 4);
                if (i == 1) std::memcpy(dst, row.val.data(), n * 4);
                if (i == 2) std::memcpy(dst, &row.params, sizeof row.params);
                if (i == 3) std::memcpy(dst, &row.state, sizeof row.state);
                if (i == 4) { const int32_t ordinary = -5; std::memcpy(dst, &ordinary, 4); }
            }
            dev[i] = new Buffer(host[i].size() * 4);
            hip(hipMemcpy(dev[i]->ptr, host[i].data(), dev[i]->bytes, hipMemcpyHostToDevice), "upload");
            tensors[i] = tensor(static_cast<uint32_t*>(dev[i]->ptr) + Pad, dt[i], M, width[i], pitch[i]);
        }
        // The serving token operand accepts the [M,1] sampler plane.
        args.t = tensors.data();args.n_t = 5;args.p = params;args.n_p = 3;
    }
    ~Arm() { for (Buffer* b : dev) delete b; }
    std::array<std::vector<uint32_t>,5> get() const {
        auto result = host;
        for (int i = 0; i < 5; ++i)
            hip(hipMemcpy(result[i].data(), dev[i]->ptr, dev[i]->bytes, hipMemcpyDeviceToHost), "download");
        return result;
    }
};
int token(const std::array<std::vector<uint32_t>,5>& x, int r) {
    return static_cast<int32_t>(x[4][Pad + 2*r]);
}
// Compare ALL bytes including guards, padding and immutable state; only token cells may change.
// This function is deliberately the same one used by the negative GPU control.
bool identical(const std::array<std::vector<uint32_t>,5>& a,
               const std::array<std::vector<uint32_t>,5>& b) { return a == b; }
struct Stream {
    hipStream_t value = nullptr;
    Stream() {
#ifdef HC_PROBE_AQL
        RadStream native = nullptr;
        require(rad_stream_create(&native, 0) == RAD_OK, "AQL stream creation failed");
        value = reinterpret_cast<hipStream_t>(native);
#else
        hip(hipStreamCreate(&value), "stream create");
#endif
    }
    void sync() const {
#ifdef HC_PROBE_AQL
        require(rad_stream_sync(reinterpret_cast<RadStream>(value)) == RAD_OK, "AQL stream sync failed");
#else
        hip(hipStreamSynchronize(value), "kernel sync");
#endif
    }
    ~Stream() {
#ifdef HC_PROBE_AQL
        if (value) rad_stream_destroy(reinterpret_cast<RadStream>(value));
#else
        if (value && hipStreamDestroy(value) != hipSuccess) std::fprintf(stderr, "stream destroy failed\n");
#endif
    }
};
void launch(const Plugin& plug, Arm& arm, const Stream& stream) {
    const int rc = plug.row->launch(&arm.args, reinterpret_cast<RadStream>(stream.value));
    if (rc != RAD_OK)
        throw std::runtime_error(std::string(plug.row->name) + " launch rc=" + std::to_string(rc));
}
void checked_run(const Plugin& plug, Arm& arm, const Stream& stream) {
    launch(plug, arm, stream);
    hip(hipGetLastError(), "kernel launch");
    stream.sync();
}
#ifdef HC_PROBE_AQL
struct Events {
    RadEvent start = nullptr, stop = nullptr;
    Events() {
        require(rad_event_create(&start) == RAD_OK, "AQL start event creation failed");
        if (rad_event_create(&stop) != RAD_OK) {
            rad_event_destroy(start);
            throw std::runtime_error("AQL stop event creation failed");
        }
    }
    ~Events() { rad_event_destroy(start); rad_event_destroy(stop); }
};
double measure(const Plugin& plug, Arm& arm, const Stream& stream, Events& e, int iters) {
    float ms = 0;
    const auto native = reinterpret_cast<RadStream>(stream.value);
    require(rad_event_record(e.start, native) == RAD_OK, "AQL start record failed");
    for (int i = 0; i < iters; ++i) launch(plug, arm, stream);
    require(rad_event_record(e.stop, native) == RAD_OK, "AQL stop record failed");
    require(rad_event_sync(e.stop) == RAD_OK, "AQL stop sync failed");
    require(rad_event_elapsed_ms(e.start, e.stop, &ms) == RAD_OK, "AQL event timing failed");
    require(ms > 0 && std::isfinite(ms), "invalid AQL event duration");
    return static_cast<double>(ms) * 1000 / iters;
}
void spreads(const char* label, std::vector<double> samples) {
    std::printf(" timing=%s samples_us=", label);
    for (size_t i = 0; i < samples.size(); ++i)
        std::printf("%s%.6f", i ? "," : "", samples[i]);
    std::sort(samples.begin(), samples.end());
    auto quantile = [&](double q) {
        const double pos = q * (samples.size() - 1);
        const auto lo = static_cast<size_t>(pos);
        const auto hi = std::min(lo + 1, samples.size() - 1);
        return samples[lo] + (pos - lo) * (samples[hi] - samples[lo]);
    };
    std::printf(" min_us=%.6f p10_us=%.6f median_us=%.6f p90_us=%.6f max_us=%.6f\n",
                samples.front(), quantile(.1), quantile(.5), quantile(.9), samples.back());
}
#endif
struct Counts { int cases = 0, accepted = 0, rejected = 0, unchanged = 0; };
void compare_case(const Plugin& baseline, const Plugin& candidate, const Stream& stream,
                  std::vector<Row> rows, int n, Counts& counts, bool control = false) {
    const int M = static_cast<int>(rows.size());
    Arm ref(rows, n), test(rows, n);
    checked_run(baseline, ref, stream);
    checked_run(candidate, test, stream);
    const auto frozen = ref.get(), got = test.get();
    require(identical(frozen, got), "frozen/candidate byte difference at " + std::string(rows[0].label) +
            " n=" + std::to_string(n));
    for (int i = 0; i < 4; ++i)
        require(frozen[i] == ref.host[i] && got[i] == test.host[i], "input/state/guard changed");
    for (int r = 0; r < M; ++r) {
        const int t = token(got, r);
        if (!rows[r].applicable) require(t == -5, std::string(rows[r].label) + " overwrote ordinary token");
        else require(t >= 0 && t < Vocab &&
                     std::find(rows[r].ids.begin(), rows[r].ids.end(), t) != rows[r].ids.end(),
                     std::string(rows[r].label) + " returned invalid target token");
        if (t == -5) ++counts.unchanged;
        else if (t == rows[r].state.x) ++counts.accepted;
        else ++counts.rejected;
        ++counts.cases;
    }
    // Full token-plane check, except for the M possible writes.
    auto normalized = got[4];
    for (int r = 0; r < M; ++r) normalized[Pad + 2*r] = test.host[4][Pad + 2*r];
    require(normalized == test.host[4], "token guard/padding overwritten");
    hip(hipMemcpy(test.dev[4]->ptr, test.host[4].data(), test.dev[4]->bytes,
                  hipMemcpyHostToDevice), "poison token plane before describe");
    auto poisoned = got;
    poisoned[4] = test.host[4];
    require(candidate.row->describe, "candidate missing describe");
    RadLaunchDesc desc{};
    int nd = -1;
    require(candidate.row->describe(&test.args, reinterpret_cast<RadStream>(stream.value), &desc, 0, &nd) == RAD_E_FULL && nd == 0,
            "describe capacity refusal failed");
    require(candidate.row->describe(&test.args, reinterpret_cast<RadStream>(stream.value), &desc, 1, &nd) == RAD_OK && nd == 1 &&
            desc.grid[0] == static_cast<uint32_t>(M) && desc.block[0] == 128 && desc.shared_bytes == 0,
            "describe launch geometry mismatch");
    require(identical(poisoned, test.get()), "describe changed poisoned output");
    hip(hipLaunchKernel(desc.func, dim3(desc.grid[0],desc.grid[1],desc.grid[2]),
        dim3(desc.block[0],desc.block[1],desc.block[2]), desc.params, desc.shared_bytes, stream.value), "described dispatch");
    stream.sync();
    require(identical(got, test.get()), "described dispatch changed output");
    hip(hipMemcpy(ref.dev[4]->ptr, ref.host[4].data(), ref.dev[4]->bytes,
                  hipMemcpyHostToDevice), "poison baseline token plane before repeat");
    hip(hipMemcpy(test.dev[4]->ptr, test.host[4].data(), test.dev[4]->bytes,
                  hipMemcpyHostToDevice), "poison candidate token plane before repeat");
    checked_run(baseline, ref, stream);checked_run(candidate, test, stream);
    require(identical(frozen, ref.get()) && identical(got, test.get()), "repeat changed output");
    if (control) {
        // P=Q is uniform on {0,1}; changing only the candidate's bound x changes its GPU token.
        auto corrupted = rows;
        corrupted[0].state.x = 1;
        Arm negative(corrupted, n);
        checked_run(candidate, negative, stream);
        const auto changed = negative.get();
        require(token(frozen, 0) != token(changed, 0) && frozen[4] != changed[4],
                "negative GPU control did not change consumer token / did not fail token comparator");
        std::printf("negative_control=PASS original=%d changed=%d\n", token(frozen,0), token(changed,0));
    }
    std::printf("case=%s rows=%d n=%d frozen/candidate/describe/repeat=byte_exact\n", rows[0].label, M, n);
}
void duplicate_order_control(const Plugin& base, const Plugin& probe, const Stream& stream, Counts& counts) {
    uint64_t seed = 0;
    for (uint64_t s = 1; s < 100000; ++s)
        if (rad_spechub_uniform(s, 5, RAD_SAMPLE_KEY_ACCEPT) > 0.5 &&
            rad_spechub_uniform(s, 5, RAD_SAMPLE_KEY_RESID) < 0.1) { seed = s; break; }
    require(seed != 0, "no deterministic duplicate-order seed");
    Row row = make(8, seed, 1);
    row.ids = {0,2,0,1,-1,-1,-1,-1};
    row.val = {-1000,-1,0,-3,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    row.label = "underflowed-first-Q-duplicate";
    compare_case(base, probe, stream, {row}, 8, counts);
    Arm first({row}, 8);
    checked_run(base, first, stream);
    require(token(first.get(), 0) == 0, "first zero-weight Q occurrence did not carry residual mass");
    row.val[0] = -INFINITY; // Same P bits; this entry now SKIPS the seen bit.
    row.label = "skipped-first-Q-duplicate";
    compare_case(base, probe, stream, {row}, 8, counts);
    Arm skipped({row}, 8);
    checked_run(base, skipped, stream);
    require(token(skipped.get(), 0) == 2, "skipping first Q occurrence did not change consumer token");
    std::printf("duplicate_order_control=PASS seed=%llu first=0 skipped=2\n",
                static_cast<unsigned long long>(seed));
}
void refusal(const Plugin& candidate, const Stream& stream) {
    auto rows = std::vector<Row>{make(1025, 11)};
    Arm a(rows, 1025);
    require(candidate.row->launch(&a.args, reinterpret_cast<RadStream>(stream.value)) == RAD_E_UNSUPPORTED,
            "1025 candidates not refused at launch");
    RadOpdDesc d{};
    require(candidate.row->opd_shape && candidate.row->opd_shape(a.params, 3, 0, &d) == RAD_E_UNSUPPORTED,
            "1025 candidates not refused at shape");
    RadLaunchDesc desc{};int nd = 0;
    require(candidate.row->describe(&a.args, reinterpret_cast<RadStream>(stream.value), &desc, 1, &nd) == RAD_E_UNSUPPORTED,
            "1025 candidates not refused at describe");
}
struct Options {
    int device = -1, warmup = 100, iters = 200, repeats = 9;
    int live_candidates = 1024;
    bool check_only = false;
    const char* candidate_row = "coupled_sampler_probe_verify";
};
Options options(int argc, char** argv) {
    require(argc >= 5, "usage: compare FROZEN_MINE_SO CANDIDATE_SO --device 0|1 [--candidate-row NAME] [--live-candidates N] [--check-only] [--warmup N --iters N --repeats N]");
    Options o;
    for (int i = 3; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check-only") { o.check_only = true; continue; }
        require(i + 1 < argc, "missing option value: " + key);
        if (key == "--candidate-row") { o.candidate_row = argv[++i]; continue; }
        char* end = nullptr;
        const long value = std::strtol(argv[++i], &end, 10);
        require(end && *end == '\0' && value >= 0 && value <= 1000000, "invalid option value: " + key);
        if (key == "--device") o.device = static_cast<int>(value);
        else if (key == "--warmup") o.warmup = static_cast<int>(value);
        else if (key == "--iters") o.iters = static_cast<int>(value);
        else if (key == "--repeats") o.repeats = static_cast<int>(value);
        else if (key == "--live-candidates") o.live_candidates = static_cast<int>(value);
        else require(false, "unknown option: " + key);
    }
    require(o.device == 0 || o.device == 1, "explicit --device 0|1 required");
    require(o.live_candidates >= 1 && o.live_candidates <= 1024, "live candidates must be 1..1024");
    require(o.warmup > 0 && o.iters > 0 && o.repeats >= 3 && o.repeats <= 101,
            "warmup/iters must be positive; repeats must be 3..101");
    return o;
}
#ifdef HC_PROBE_AQL
void time_dense(const Plugin& base, const Plugin& probe, const Stream& stream,
                const std::vector<Row>& rows, const Options& o, const char* mix, Events& events) {
    const int M = static_cast<int>(rows.size());
    Arm ref(rows, 1024), test(rows, 1024);
    checked_run(base, ref, stream);
    checked_run(probe, test, stream);
    const auto frozen = ref.get(), got = test.get();
    require(identical(frozen, got), "timing fixture baseline/candidate mismatch before timing");
    for (int i = 0; i < 4; ++i)
        require(frozen[i] == ref.host[i] && got[i] == test.host[i], "timing fixture changed input/state/guards");
    for (int i = 0; i < o.warmup; ++i) { launch(base, ref, stream); launch(probe, test, stream); }
    stream.sync();
    std::vector<double> a, b;
    a.reserve(o.repeats); b.reserve(o.repeats);
    for (int repeat = 0; repeat < o.repeats; ++repeat) {
        if (repeat & 1) {
            b.push_back(measure(probe, test, stream, events, o.iters));
            a.push_back(measure(base, ref, stream, events, o.iters));
        } else {
            a.push_back(measure(base, ref, stream, events, o.iters));
            b.push_back(measure(probe, test, stream, events, o.iters));
        }
    }
    stream.sync();
    hip(hipGetLastError(), "timed kernel launch");
    require(identical(frozen, ref.get()) && identical(got, test.get()),
            "timed operations changed output, input/state, pitch or guards");
    std::printf("fixture M=%d n=1024 live=%d mix=%s warmup_each=%d timed_each=%d repeats=%d "
                "total_launches=%d order=alternating native_AQL_live_submission_events\n",
                M, o.live_candidates, mix, o.warmup, o.iters * o.repeats, o.repeats,
                2 * (1 + o.warmup + o.iters * o.repeats));
    spreads("frozen_mine", a);
    spreads("candidate", b);
}
#endif
void run(const char* frozen, const char* candidate, const Options& o) {
    hip(hipSetDevice(o.device), "device selection");
#ifdef HC_PROBE_AQL
    require(rad_dev_set(o.device) == RAD_OK, "AQL device selection failed");
    RadDeviceProps native_props{};
    require(rad_dev_props(o.device, &native_props) == RAD_OK && !native_props.is_host_backend,
            "native comparator requires accelerator backend");
#endif
    hipDeviceProp_t properties{};
    hip(hipGetDeviceProperties(&properties, o.device), "device properties");
    require(std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0, "gfx1201 required");
    std::printf("device=%d name=%s arch=%s\n", o.device, properties.name, properties.gcnArchName);
    Plugin base(frozen, "mine_mtp_couple_verify"), probe(candidate, o.candidate_row);
    Stream stream;
    Counts counts;
    std::vector<Row> rows;
    for (int seed = 0; seed < 64; ++seed) rows.push_back(make(8, seed, seed & 1));
    compare_case(base, probe, stream, rows, 8, counts);
    rows = {make(8, 0xdfddea36362e095full, 0, true)};
    rows[0].ids = {2,3,4,5,6,7,0,1};
    const float second = static_cast<float>(-std::log(2.0));
    rows[0].val = {-37,-37,-37,-37,-37,-37,0,second};
    rows[0].label = "cancellation";
    for (int x = 0; x < 8; ++x) {
        rows[0].state.x = x;
        compare_case(base, probe, stream, rows, 8, counts);
    }
    rows = {make(8, 101), make(8, 102), make(8, 103), make(8, 104), make(8, 105),
            make(8, 106), make(8, 107), make(8, 108), make(8, 109), make(8, 110)};
    rows[0].ids = {0,1,0,2,1,-1,-1,-1}; rows[0].val = {-1000,-0.5f,0,0,-1,-INFINITY,-INFINITY,-INFINITY};rows[0].label="underflow-first-duplicate";
    rows[1].ids[0] = -1;rows[1].val[1] = -INFINITY;rows[1].label="holes-minus-infinity";
    rows[2].params.flags = RAD_SP_GREEDY;rows[2].label="greedy";rows[2].applicable=false;
    ++rows[3].state.pos;rows[3].label="stale-position";rows[3].applicable=false;
    ++rows[4].state.draw.seed;rows[4].label="stale-seed";rows[4].applicable=false;
    rows[5].val[1] = NAN;rows[5].label="nan-target";rows[5].applicable=false;
    rows[6].val[1] = INFINITY;rows[6].label="infinite-target";rows[6].applicable=false;
    rows[7].ids[1] = Vocab;rows[7].label="out-of-range-target";rows[7].applicable=false;
    std::fill(rows[8].ids.begin(),rows[8].ids.end(),-1);rows[8].label="empty-target";rows[8].applicable=false;
    rows[9].state.ids[1] = 0;rows[9].label="invalid-state";rows[9].applicable=false;
    compare_case(base, probe, stream, rows, 8, counts);
    duplicate_order_control(base, probe, stream, counts);
    rows = {make(4, 15)};
    rows[0].ids = {0,1,-1,-1};rows[0].val = {0,0,-INFINITY,-INFINITY};
    rows[0].label = "equal-law-negative-control";
    compare_case(base, probe, stream, rows, 4, counts, true);
    compare_case(base, probe, stream, {make(1, 31)}, 1, counts);
    compare_case(base, probe, stream, {make(129, 91),make(129, 92),make(129, 93)}, 129, counts);
    compare_case(base, probe, stream, {make(1024, 201),make(1024, 202)}, 1024, counts);
    if (!std::strcmp(o.candidate_row, "mine_mtp_couple_verify"))
        compare_case(base, probe, stream, {dense(1025, 301, false),dense(1025, 302, true)}, 1025, counts);
    else
        refusal(probe, stream);
    require(counts.accepted > 0 && counts.rejected > 0 && counts.unchanged >= 8,
            "accepted/rejected/unbound outcomes not all exercised");
    // Validate all dense rows before recording any time. Each M has both branch outcomes.
    for (int n : {129, 1024}) for (int M : {1, 4, 32}) {
        Counts dense_counts;
        for (int mix = 0; mix < 2; ++mix) for (int seed = 0; seed < 3; ++seed) {
            rows.clear();
            for (int r = 0; r < M; ++r)
                rows.push_back(dense(n, 1001 + seed * 97 + r * 13 + mix * 1009,
                                     (r + mix) % 2 == 0));
            compare_case(base, probe, stream, rows, n, dense_counts);
        }
        require(dense_counts.accepted > 0 && dense_counts.rejected > 0 &&
                dense_counts.cases == M * 6, "dense acceptance/rejection coverage missing");
        counts.cases += dense_counts.cases;
        counts.accepted += dense_counts.accepted;
        counts.rejected += dense_counts.rejected;
        std::printf("dense_coverage n=%d M=%d accepted=%d rejected=%d\n",
                    n, M, dense_counts.accepted, dense_counts.rejected);
    }
#ifdef HC_PROBE_AQL
    if (!o.check_only) {
        Events events;
        for (int M : {1, 4, 32}) for (int mix = 0; mix < 2; ++mix) {
            rows.clear();
            for (int r = 0; r < M; ++r)
                rows.push_back(dense(1024, 1001 + r * 13 + mix * 1009, (r + mix) % 2 == 0));
            for (auto& row : rows)
                for (int j = o.live_candidates; j < 1024; ++j) {
                    row.ids[j] = -1;
                    row.val[j] = -INFINITY;
                }
            time_dense(base, probe, stream, rows, o, mix ? "reject-first" : "accept-first", events);
        }
    }
#endif
    std::printf("PASS cases=%d accepted=%d rejected=%d untouched=%d width_contract=PASS timing=%s\n",
                counts.cases, counts.accepted, counts.rejected, counts.unchanged,
#ifdef HC_PROBE_AQL
                o.check_only ? "disabled" : "native_AQL_live_submission_events");
#else
                "disabled_ordinary_HIP_check_only");
#endif
}
} // namespace
int main(int argc, char** argv) {
    try {
        const Options o = options(argc, argv);
        run(argv[1], argv[2], o);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
