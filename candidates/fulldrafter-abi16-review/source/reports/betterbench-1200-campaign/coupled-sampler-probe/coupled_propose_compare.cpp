// Compare the ACTUAL frozen/current production MTP proposal row against the ACTUAL cooperative
// proposal probe plugin, both loaded as DSOs. This never reimplements the proposal law.
//
// What is compared, byte for byte, over every plane including guards and row pitch padding:
//   * the whole `state` plane (every temporal slab of 152-byte RadMtpCoupleState rows),
//   * `idx` column 0 (the sampled x) and every other idx word, which must NOT move,
//   * `val`, `seq_ids` and `draw` (read-only inputs, must NOT move),
//   * and, downstream, the exact `token` cells the frozen production verifier emits from each
//     arm's produced state.
//
// WHY WIDTHS 20/129/1024/1025 APPEAR: the proposal shortlist is frozen at eight columns
// (mine/spechub.cpp: rad_mtp_couple_propose_shape is [M,8]; rad_mtp_couple_build_q walks eight
// slots), so a proposal-only width boundary does not exist. The boundaries belong to the
// CONSUMER: a produced state is only correct if the frozen verifier turns it into the same token
// at every candidate width it admits, including 1025 where mine/spechub.hip falls back to its
// general single-thread algorithm. That is the contract the proposal row serves, and it is the
// reachable width sweep this comparator can actually exercise.
//
// ROW COUNT: the probe uses one eight-lane group per row and four rows per block, so M = 1, 4 and
// 32 cross its block boundaries (32 = 8 full blocks).
//
// Timing: matched alternating launch pairs of the two real rows, reported as launch microseconds
// only, with no claim about engine throughput, acceptance or scoring.
//
// Run from the worktree root. Compile argv is in the report.
#include <hip/hip_runtime.h>
#include "../../../mine/spechub.h"
#ifdef PROPOSE_COMPARE_AQL
#include "rad_device.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    Plugin(const char* path, const char* name, const char* op)
        : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        require(handle != nullptr, std::string("dlopen ") + path + ": " + dlerror());
        auto abi = reinterpret_cast<uint32_t (*)()>(dlsym(handle, "rad_plugin_abi_version"));
        auto count = reinterpret_cast<int (*)()>(dlsym(handle, "rad_kernel_count"));
        auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(handle, "rad_kernel_at"));
        require(abi && count && at && abi() == RAD_ABI_VERSION, "plugin ABI/exports mismatch");
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* r = at(i);
            if (r && r->name && !std::strcmp(r->name, name)) {
                require(!row, "duplicate row name in plugin");
                row = r;
            }
        }
        require(row && row->launch && row->op && !std::strcmp(row->op, op) &&
                row->domain == RAD_DOMAIN_DEVICE, std::string("actual row not loaded: ") + name);
        Dl_info origin{};
        char expected[PATH_MAX], actual[PATH_MAX];
        require(dladdr(reinterpret_cast<const void*>(row->launch), &origin) && origin.dli_fname &&
                realpath(path, expected) && realpath(origin.dli_fname, actual) &&
                !std::strcmp(expected, actual), "row launch not from the requested plugin");
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
constexpr int Pad = 16;  // guard words before and after every plane
constexpr int Vocab = 4096;
constexpr int Slots = 8;  // the proposal shortlist width, frozen by the op schema
constexpr int StateWords = RAD_MTP_COUPLE_STATE_WORDS;
constexpr int DrawWords = RAD_MTP_COUPLE_DRAW_WORDS;
constexpr int ParamWords = 24;  // the RadSampleParams wire width the verifier reads

struct Options {
    int device = -1, warmup = 50, iters = 200, repeats = 9;
    bool check_only = false;
    const char* candidate_row = "coupled_propose_cooperative_probe";
};

struct Stream {
    hipStream_t value = nullptr;
    Stream() {
#ifdef PROPOSE_COMPARE_AQL
        RadStream native = nullptr;
        require(rad_stream_create(&native, 0) == RAD_OK, "AQL stream creation failed");
        value = reinterpret_cast<hipStream_t>(native);
#else
        hip(hipStreamCreate(&value), "stream create");
#endif
    }
    void sync() const {
#ifdef PROPOSE_COMPARE_AQL
        require(rad_stream_sync(reinterpret_cast<RadStream>(value)) == RAD_OK, "AQL stream sync failed");
#else
        hip(hipStreamSynchronize(value), "kernel sync");
#endif
    }
    ~Stream() {
#ifdef PROPOSE_COMPARE_AQL
        if (value) rad_stream_destroy(reinterpret_cast<RadStream>(value));
#else
        if (value && hipStreamDestroy(value) != hipSuccess) std::fprintf(stderr, "stream destroy failed\n");
#endif
    }
};

// ---------------------------------------------------------------- proposal fixtures
struct P {
    std::array<int32_t, Slots> ids{};
    std::array<float, Slots> val{};
    RadMtpCoupleDraw draw{};
    int32_t card = 0;
    const char* label = "dense8";
};

// Every proposal branch the production row has is reachable from one of these kinds.
P propose_row(const std::string& kind, int index, uint64_t seed) {
    P r;
    for (int i = 0; i < Slots; ++i) { r.ids[i] = i; r.val[i] = -static_cast<float>(i); }
    r.card = index;
    r.draw = RadMtpCoupleDraw{seed, 900 + static_cast<uint64_t>(index),
                              100u + static_cast<uint64_t>(index), index, 1};
    if (kind == "sparse2") {
        r.ids = {3, 1, -1, -1, -1, -1, -1, -1};
        r.val = {0.f, -0.5f, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        r.label = "sparse2";
    } else if (kind == "holes") {
        r.ids = {-1, 2, -1, 7, -1, -1, -1, 5};
        r.val = {-INFINITY, 0.f, -INFINITY, -1.5f, -INFINITY, -INFINITY, -INFINITY, -0.25f};
        r.label = "holes";
    } else if (kind == "duplicate") {
        r.ids = {0, 1, 1, 2, -1, -1, -1, -1};  // slot 2 repeats id 1: status -1
        r.val = {0.f, -0.5f, -0.25f, -2.f, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        r.label = "duplicate";
    } else if (kind == "underflow-first") {
        // exp(-800) and below are exactly zero in FP64, so the `q > 0` drop must shorten `count`
        // below the six live slots. Slot 0 keeps full mass and is the only surviving candidate.
        r.ids = {0, 1, 2, 3, 4, 5, -1, -1};
        r.val = {0.f, -400.f, -800.f, -1200.f, -1600.f, -2000.f, -INFINITY, -INFINITY};
        r.label = "underflow-first";
    } else if (kind == "greedy") {
        r.draw.mode = 0;
        r.card = 999;  // mode 0 must not consult seq_ids at all
        r.label = "greedy";
    } else if (kind == "stale-card") {
        r.draw.card = index + 1000;
        r.label = "stale-card";
    } else if (kind == "nan") {
        r.val[2] = NAN;
        r.label = "nan";
    } else if (kind == "plus-inf") {
        r.val[2] = INFINITY;
        r.label = "plus-inf";
    } else if (kind == "out-of-range") {
        r.ids[2] = Vocab;
        r.label = "out-of-range";
    } else if (kind == "empty") {
        r.ids = {3, 5, -1, -1, -1, -1, -1, -1};
        r.val = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        r.label = "empty-support";
    } else if (kind == "wide-ids") {
        r.ids = {4095, 0, 2047, 1, 1024, 2, 3, 4};
        r.val = {0.f, -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f};
        r.label = "wide-ids";
    }
    return r;
}

RadTensor make_tensor(void* base, int dtype, int rank, std::array<int64_t, 3> shape,
                      std::array<int64_t, 3> stride) {
    RadTensor t{};
    t.data = base;
    t.dtype = static_cast<uint32_t>(dtype);
    t.rank = static_cast<uint32_t>(rank);
    for (int i = 0; i < rank; ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

struct ProposeArm {
    int M = 0, rounds = 0, round = 1;
    int pitch[5] = {};
    std::array<std::vector<uint32_t>, 5> host;
    std::array<Buffer*, 5> dev{};
    std::array<RadTensor, 5> tensors{};
    RadParam params[4];
    RadArgs args{};
    ProposeArm(const std::vector<P>& rows, int max_rounds) {
        M = static_cast<int>(rows.size());
        rounds = max_rounds;
        pitch[0] = Slots + 2;        // idx: two guard words per row, so an OOB column is visible
        pitch[1] = Slots + 2;        // val: same
        pitch[2] = 1;                // seq_ids
        pitch[3] = DrawWords;        // draw, wire plane
        pitch[4] = StateWords + 2;   // state: two guard words per row; even, so 8-byte aligned
        const int64_t sr = static_cast<int64_t>(M) * pitch[4];
        const size_t words[5] = {
            static_cast<size_t>(Pad) + static_cast<size_t>(M) * pitch[0] + Pad,
            static_cast<size_t>(Pad) + static_cast<size_t>(M) * pitch[1] + Pad,
            static_cast<size_t>(Pad) + static_cast<size_t>(M) + Pad,
            static_cast<size_t>(Pad) + static_cast<size_t>(M) * pitch[3] + Pad,
            static_cast<size_t>(Pad) + static_cast<size_t>(rounds) * sr + Pad};
        for (int i = 0; i < 5; ++i) {
            host[i].assign(words[i], Guard);
            for (int r = 0; r < M; ++r) {
                uint32_t* base = host[i].data() + Pad;
                if (i == 0) std::memcpy(base + static_cast<size_t>(r) * pitch[0], rows[r].ids.data(), Slots * 4);
                if (i == 1) std::memcpy(base + static_cast<size_t>(r) * pitch[1], rows[r].val.data(), Slots * 4);
                if (i == 2) base[r] = static_cast<uint32_t>(rows[r].card);
                if (i == 3) std::memcpy(base + static_cast<size_t>(r) * pitch[3], &rows[r].draw, sizeof rows[r].draw);
            }
            dev[i] = new Buffer(host[i].size() * 4);
            hip(hipMemcpy(dev[i]->ptr, host[i].data(), dev[i]->bytes, hipMemcpyHostToDevice), "upload");
        }
        uint32_t* p0 = static_cast<uint32_t*>(dev[0]->ptr) + Pad;
        uint32_t* p4 = static_cast<uint32_t*>(dev[4]->ptr) + Pad;
        tensors[0] = make_tensor(p0, RAD_I32, 2, {M, Slots, 0}, {pitch[0], 1, 0});
        tensors[1] = make_tensor(static_cast<uint32_t*>(dev[1]->ptr) + Pad, RAD_F32, 2, {M, Slots, 0},
                                 {pitch[1], 1, 0});
        tensors[2] = make_tensor(static_cast<uint32_t*>(dev[2]->ptr) + Pad, RAD_I32, 1, {M, 0, 0}, {1, 0, 0});
        tensors[3] = make_tensor(static_cast<uint32_t*>(dev[3]->ptr) + Pad, RAD_U32, 2, {M, DrawWords, 0},
                                 {pitch[3], 1, 0});
        tensors[4] = make_tensor(p4, RAD_U32, 3, {rounds, M, StateWords}, {sr, pitch[4], 1});
        params[0] = RAD_INT("M", M);
        params[1] = RAD_INT("n_vocab", Vocab);
        params[2] = RAD_INT("round", round);
        params[3] = RAD_INT("max_rounds", rounds);
        args.t = tensors.data(); args.n_t = 5; args.p = params; args.n_p = 4;
    }
    ~ProposeArm() { for (Buffer* b : dev) delete b; }
    ProposeArm(const ProposeArm&) = delete;
    ProposeArm& operator=(const ProposeArm&) = delete;
    std::array<std::vector<uint32_t>, 5> get() const {
        auto out = host;
        for (int i = 0; i < 5; ++i)
            hip(hipMemcpy(out[i].data(), dev[i]->ptr, dev[i]->bytes, hipMemcpyDeviceToHost), "download");
        return out;
    }
    // The engine freezes `round` per op and issues one op per temporal round.
    void select_round(int r) { round = r; params[2] = RAD_INT("round", r); }
    size_t round_pitch() const { return static_cast<size_t>(M) * pitch[4]; }
    RadMtpCoupleState state_at(const std::vector<uint32_t>& plane, int slab, int row) const {
        RadMtpCoupleState s{};
        std::memcpy(&s, plane.data() + Pad + static_cast<size_t>(slab - 1) * round_pitch() +
                                   static_cast<size_t>(row) * pitch[4], sizeof s);
        return s;
    }
};

void launch(const Plugin& plug, ProposeArm& arm, const Stream& stream) {
    const int rc = plug.row->launch(&arm.args, reinterpret_cast<RadStream>(stream.value));
    if (rc != RAD_OK)
        throw std::runtime_error(std::string(plug.row->name) + " launch round=" +
                                 std::to_string(arm.round) + " rc=" + std::to_string(rc));
}
void checked_run(const Plugin& plug, ProposeArm& arm, const Stream& stream) {
    launch(plug, arm, stream);
    hip(hipGetLastError(), "kernel launch");
    stream.sync();
}

// The row may write idx column 0 and exactly its own 152-byte state row of one slab. Everything
// else -- other idx columns, val, seq_ids, draw, every state pitch pad word and every guard -- must
// be untouched.
void check_writes(const ProposeArm& arm, const std::array<std::vector<uint32_t>, 5>& before,
                  const std::array<std::vector<uint32_t>, 5>& after, int slab) {
    require(after[0].size() == before[0].size(), "idx plane resized");
    const size_t ip = static_cast<size_t>(arm.pitch[0]);
    for (size_t w = 0; w < after[0].size(); ++w) {
        const long off = static_cast<long>(w) - Pad;
        const bool column0 = off >= 0 && static_cast<size_t>(off) % ip == 0 &&
                             static_cast<size_t>(off) / ip < static_cast<size_t>(arm.M);
        if (!column0) require(after[0][w] == before[0][w], "idx non-column-0 word or guard changed");
    }
    for (int i = 1; i < 4; ++i) require(after[i] == before[i], "read-only input plane changed");
    const size_t rp = arm.round_pitch(), sp = static_cast<size_t>(arm.pitch[4]);
    require(after[4].size() == before[4].size(), "state plane resized");
    for (size_t w = 0; w < after[4].size(); ++w) {
        const long off = static_cast<long>(w) - Pad;
        bool own = false;
        if (off >= 0) {
            const size_t u = static_cast<size_t>(off);
            const size_t which = (u / rp) + 1, within = u % rp;
            own = which == static_cast<size_t>(slab) &&
                  within / sp < static_cast<size_t>(arm.M) && within % sp < StateWords;
        }
        if (!own) require(after[4][w] == before[4][w], "state pad/guard/other slab changed");
    }
}

struct ProposeOutcome {
    std::array<std::vector<uint32_t>, 5> image;
    int sampled = 0, greedy = 0, malformed = 0, dropped = 0;
};

ProposeOutcome run_proposal(const Plugin& plug, ProposeArm& arm, const Stream& stream) {
    ProposeOutcome out;
    for (int slab = 1; slab <= arm.rounds; ++slab) {
        const auto before = arm.get();
        arm.select_round(slab);
        checked_run(plug, arm, stream);
        out.image = arm.get();
        check_writes(arm, before, out.image, slab);
    }
    out.sampled = out.greedy = out.malformed = out.dropped = 0;
    for (int r = 0; r < arm.M; ++r) {
        const RadMtpCoupleState s = arm.state_at(out.image[4], arm.rounds, r);
        require(s.pos == 0, "device wrote state.pos");
        require(s.round == arm.rounds, "state.round does not name the launched round");
        const int32_t col0 = static_cast<int32_t>(
            out.image[0][Pad + static_cast<size_t>(r) * arm.pitch[0]]);
        if (s.status == 1) {
            ++out.sampled;
            require(s.count >= 1 && s.count <= 8, "sampled state count out of range");
            bool has_x = false;
            for (int k = 0; k < s.count; ++k) has_x = has_x || s.ids[k] == s.x;
            require(has_x, "state.x not among the stored Q ids");
            require(col0 == s.x, "idx column 0 does not carry x");
            for (int k = s.count; k < 8; ++k)
                require(s.ids[k] == -1 && s.q[k] == 0, "state tail not cleared");
            for (int k = 0; k < s.count; ++k) require(s.q[k] > 0, "stored q is not positive");
            if (s.count < 8) ++out.dropped;
        } else if (s.status == 0) {
            ++out.greedy;
            require(s.count == 0 && col0 == 0, "greedy row changed column 0 or kept a count");
        } else {
            ++out.malformed;
            require(s.status == -1 && s.count == 0, "malformed row kept a count");
            require(col0 == 0, "malformed row changed column 0");
        }
    }
    return out;
}

// ---------------------------------------------------------------- consumption fixtures
struct VerifyArm {
    int M = 0, n = 0;
    std::vector<std::vector<int32_t>> ids;
    std::vector<std::vector<float>> val;
    std::vector<RadSampleParams> params;
    std::vector<RadMtpCoupleState> state;
    std::array<std::vector<uint32_t>, 5> host;
    std::array<Buffer*, 5> dev{};
    std::array<RadTensor, 5> tensors{};
    RadParam p[3];
    RadArgs args{};
    int pitch[5] = {};

    VerifyArm(int rows, int cand) : M(rows), n(cand), ids(rows, std::vector<int32_t>(cand, -1)),
        val(rows, std::vector<float>(cand, -INFINITY)), params(rows), state(rows) {
        pitch[0] = n; pitch[1] = n; pitch[2] = ParamWords; pitch[3] = StateWords; pitch[4] = 1;
        for (int i = 0; i < 5; ++i) {
            const size_t words = static_cast<size_t>(Pad) + static_cast<size_t>(M) * pitch[i] + Pad;
            host[i].assign(words, Guard);
        }
        p[0] = RAD_INT("M", M); p[1] = RAD_INT("n_cand", n); p[2] = RAD_INT("n_vocab", Vocab);
    }
    ~VerifyArm() { for (Buffer* b : dev) delete b; }
    VerifyArm(const VerifyArm&) = delete;
    VerifyArm& operator=(const VerifyArm&) = delete;

    // Build the wire images from the fixture vectors, add the guards, and upload.
    void sync_planes() {
        for (int i = 0; i < 5; ++i) {
            std::fill(host[i].begin(), host[i].end(), Guard);
            for (int r = 0; r < M; ++r) {
                uint32_t* base = host[i].data() + Pad + static_cast<size_t>(r) * pitch[i];
                if (i == 0) std::memcpy(base, ids[r].data(), static_cast<size_t>(n) * 4);
                if (i == 1) std::memcpy(base, val[r].data(), static_cast<size_t>(n) * 4);
                if (i == 2) std::memcpy(base, &params[r], sizeof(RadSampleParams));
                if (i == 3) std::memcpy(base, &state[r], sizeof(RadMtpCoupleState));
                if (i == 4) base[0] = static_cast<uint32_t>(-5);
            }
        }
    }
    void upload() {
        const int dt[5] = {RAD_I32, RAD_F32, RAD_U32, RAD_U32, RAD_I32};
        for (int i = 0; i < 5; ++i) {
            dev[i] = new Buffer(host[i].size() * 4);
            hip(hipMemcpy(dev[i]->ptr, host[i].data(), dev[i]->bytes, hipMemcpyHostToDevice), "upload");
            uint32_t* base = static_cast<uint32_t*>(dev[i]->ptr) + Pad;
            if (i == 4) {
                tensors[i] = make_tensor(base, dt[i], 1, {M, 0, 0}, {1, 0, 0});
            } else {
                tensors[i] = make_tensor(base, dt[i], 2, {M, pitch[i], 0}, {pitch[i], 1, 0});
            }
        }
        args.t = tensors.data(); args.n_t = 5; args.p = p; args.n_p = 3;
    }
    std::array<std::vector<uint32_t>, 5> get() const {
        auto out = host;
        for (int i = 0; i < 5; ++i)
            hip(hipMemcpy(out[i].data(), dev[i]->ptr, dev[i]->bytes, hipMemcpyDeviceToHost), "download");
        return out;
    }
};

// Deterministic target rows bound to a produced state. `sparse` keeps six live candidates; `dense`
// fills every column. Every Q id is placed so the coupling has mass to compare, and the remaining
// live slots carry finite logits.
void fill_candidates(VerifyArm& arm, int row, const RadMtpCoupleState& s, bool sparse, uint64_t seed) {
    for (int j = 0; j < arm.n; ++j) { arm.ids[row][j] = -1; arm.val[row][j] = -INFINITY; }
    for (int k = 0; k < s.count; ++k) {
        const int at = static_cast<int>((static_cast<long>(k) * 7 + 3) % arm.n);
        arm.ids[row][at] = s.ids[k];
        arm.val[row][at] = -1.f + static_cast<float>((seed + 3u * static_cast<uint64_t>(k)) % 7u) / 4.f;
    }
    int live = 0;
    for (int j = 0; j < arm.n; ++j) live += arm.ids[row][j] >= 0;
    const int want = sparse ? std::min(6, arm.n) : arm.n;
    for (int j = 0; j < arm.n && live < want; ++j) {
        if (arm.ids[row][j] >= 0) continue;
        const uint32_t h = static_cast<uint32_t>(seed) * 747796405u +
                           static_cast<uint32_t>(j) * 2891336453u;
        arm.ids[row][j] = 3000 + static_cast<int32_t>((static_cast<uint64_t>(j) * 37u + seed) % 700u);
        arm.val[row][j] = -6.f + static_cast<float>((h ^ (h >> 16)) % 400u) / 100.f;
        ++live;
    }
}

struct TokenOutcome {
    std::vector<int32_t> token;
    int acted = 0, accepted = 0, rejected = 0, untouched = 0;
};

// Run the frozen production verifier over one arm's produced state at one candidate width.
TokenOutcome consume(const Plugin& verify, const ProposeOutcome& produced, const ProposeArm& arm,
                     const Stream& stream, int n, bool sparse, uint64_t seed,
                     const std::vector<int32_t>* override_x) {
    VerifyArm va(arm.M, n);
    for (int r = 0; r < arm.M; ++r) {
        RadMtpCoupleState s = arm.state_at(produced.image[4], arm.rounds, r);
        s.pos = 7;
        if (override_x) {
            s.x = (*override_x)[static_cast<size_t>(r)];
            require(rad_mtp_couple_state_valid(s, Vocab), "control state is not valid");
        }
        va.state[r] = s;
        va.params[r] = RadSampleParams{};
        va.params[r].seed = s.draw.seed;
        va.params[r].pos = s.pos;
        fill_candidates(va, r, s, sparse, seed + static_cast<uint64_t>(r));
    }
    va.sync_planes();
    va.upload();
    const int rc = verify.row->launch(&va.args, reinterpret_cast<RadStream>(stream.value));
    require(rc == RAD_OK, std::string(verify.row->name) + " launch n_cand=" + std::to_string(n) +
            " rc=" + std::to_string(rc));
    hip(hipGetLastError(), "verify launch");
    stream.sync();
    const auto img = va.get();
    for (int i = 0; i < 4; ++i) require(img[i] == va.host[i], "verify changed a read-only plane");
    TokenOutcome out;
    out.token.assign(static_cast<size_t>(arm.M), -5);
    for (int r = 0; r < arm.M; ++r) {
        const int t = static_cast<int32_t>(img[4][Pad + static_cast<size_t>(r)]);
        out.token[static_cast<size_t>(r)] = t;
        if (t == -5) { ++out.untouched; continue; }
        ++out.acted;
        require(t >= 0 && t < Vocab, "verify emitted an out-of-range token");
        require(std::find(va.ids[r].begin(), va.ids[r].end(), t) != va.ids[r].end(),
                "verify emitted a token outside the candidate set");
        if (t == va.state[r].x) ++out.accepted; else ++out.rejected;
    }
    for (size_t w = 0; w < img[4].size(); ++w) {
        const long off = static_cast<long>(w) - Pad;
        const bool cell = off >= 0 && off < arm.M;
        if (!cell) require(img[4][w] == va.host[4][w], "token guard/pitch word changed");
    }
    return out;
}

// Build a two-candidate state on the host with the production helpers, so controls do not depend
// on a device launch.
RadMtpCoupleState host_state(uint64_t seed, int x) {
    const int32_t ids[8] = {0, 1, -1, -1, -1, -1, -1, -1};
    const float val[8] = {0.f, 0.f, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    RadMtpCoupleState s{};
    s.draw.mode = 1;
    s.draw.seed = seed;
    s.round = 1;
    require(rad_mtp_couple_build_q(ids, val, Vocab, &s), "control Q construction failed");
    s.x = x;
    s.pos = 7;
    require(rad_mtp_couple_state_valid(s, Vocab), "control state is not valid");
    return s;
}

// ---------------------------------------------------------------- cases
struct Counts { int cases = 0, sampled = 0, greedy = 0, malformed = 0, dropped = 0, tokens = 0; };

void compare_case(const Plugin& base, const Plugin& probe, const Stream& stream,
                  const std::string& kind, int M, int rounds, Counts& counts) {
    std::vector<P> rows;
    for (int r = 0; r < M; ++r)
        rows.push_back(propose_row(kind, r, 0x5eed0000ull + static_cast<uint64_t>(r) * 7919u +
                                   static_cast<uint64_t>(rounds)));
    ProposeArm a(rows, rounds), b(rows, rounds);
    const auto A = run_proposal(base, a, stream);
    const auto B = run_proposal(probe, b, stream);
    require(A.image == B.image, "frozen/probe byte difference: " + kind + " M=" + std::to_string(M) +
            " rounds=" + std::to_string(rounds));
    require(A.sampled == B.sampled && A.greedy == B.greedy && A.malformed == B.malformed &&
            A.dropped == B.dropped, "branch census differs between arms");
    counts.cases += M;
    counts.sampled += B.sampled;
    counts.greedy += B.greedy;
    counts.malformed += B.malformed;
    counts.dropped += B.dropped;
    std::printf("case=%s M=%d rounds=%d sampled=%d greedy=%d malformed=%d underflow_dropped=%d "
                "state+idx+inputs+guards=byte_exact\n",
                kind.c_str(), M, rounds, B.sampled, B.greedy, B.malformed, B.dropped);
}

void refusal(const Plugin& base, const Plugin& probe, const Stream& stream) {
    const std::vector<P> rows = {propose_row("dense8", 0, 11)};
    auto both = [&](const char* why, ProposeArm& a) {
        const int rb = base.row->launch(&a.args, reinterpret_cast<RadStream>(stream.value));
        const int rp = probe.row->launch(&a.args, reinterpret_cast<RadStream>(stream.value));
        require(rb != RAD_OK && rb == rp, std::string("refusal disagreement: ") + why +
                " frozen=" + std::to_string(rb) + " probe=" + std::to_string(rp));
        std::printf("refusal=%s rc=%d both_arms=agree\n", why, rb);
    };
    {
        ProposeArm a(rows, RAD_MTP_COUPLE_MAX_ROUNDS + 1);
        both("max_rounds>8", a);
    }
    {
        ProposeArm a(rows, 1);
        a.select_round(9);  // round beyond max_rounds
        both("round>max_rounds", a);
    }
    {
        ProposeArm a(rows, 1);
        a.args.n_t = 4;
        both("operand count", a);
    }
    {
        ProposeArm a(rows, 1);
        a.tensors[3].data = static_cast<uint32_t*>(a.dev[2]->ptr) + 1;  // 4-byte-aligned draw
        both("draw alignment", a);
    }
    {
        ProposeArm a(rows, 1);
        a.tensors[4].data = a.tensors[3].data;  // aliasing wire planes
        both("operand overlap", a);
    }
    {
        ProposeArm a(rows, 1);
        a.tensors[0].shape[0] = a.M + 1;  // live rows beyond the M band
        both("live rows beyond M band", a);
    }
}

void describe_check(const Plugin& probe, const Stream& stream, int M) {
    require(probe.row->describe, "probe row missing describe");
    std::vector<P> rows;
    for (int r = 0; r < M; ++r) rows.push_back(propose_row("sparse2", r, 4242u + static_cast<uint64_t>(r)));
    ProposeArm arm(rows, 1);
    arm.select_round(1);
    const auto before = arm.get();
    checked_run(probe, arm, stream);
    const auto launched = arm.get();
    RadLaunchDesc desc{};
    int nd = -1;
    require(probe.row->describe(&arm.args, reinterpret_cast<RadStream>(stream.value), &desc, 0, &nd) ==
            RAD_E_FULL && nd == 0, "describe capacity refusal failed");
    require(probe.row->describe(&arm.args, reinterpret_cast<RadStream>(stream.value), &desc, 1, &nd) ==
            RAD_OK && nd == 1, "describe did not emit exactly one launch");
    const uint32_t want_grid = static_cast<uint32_t>((M + 3) / 4);
    require(desc.grid[0] == want_grid && desc.grid[1] == 1 && desc.grid[2] == 1 &&
            desc.block[0] == 32 && desc.block[1] == 1 && desc.block[2] == 1 && desc.shared_bytes == 0,
            "describe launch geometry mismatch");
    check_writes(arm, before, launched, 1);
    // Poison the writable planes, then dispatch the described launch: it must reproduce the same
    // bytes, so a describe/launch divergence cannot hide.
    hip(hipMemcpy(arm.dev[0]->ptr, arm.host[0].data(), arm.dev[0]->bytes, hipMemcpyHostToDevice),
        "poison idx before described dispatch");
    hip(hipMemcpy(arm.dev[4]->ptr, arm.host[4].data(), arm.dev[4]->bytes, hipMemcpyHostToDevice),
        "poison state before described dispatch");
    hip(hipLaunchKernel(desc.func, dim3(desc.grid[0], desc.grid[1], desc.grid[2]),
                        dim3(desc.block[0], desc.block[1], desc.block[2]), desc.params,
                        desc.shared_bytes, stream.value), "described dispatch");
    hip(hipGetLastError(), "described dispatch");
    stream.sync();
    require(arm.get() == launched, "described dispatch differs from the launch path");
    std::printf("describe_geometry M=%d grid=%u block=%u lds=%u dispatch=byte_exact\n",
                M, desc.grid[0], desc.block[0], desc.shared_bytes);
}

// Positive identity first, then one single-ulp input change: the byte comparator must fire.
void proposal_negative_control(const Plugin& base, const Plugin& probe, const Stream& stream) {
    std::vector<P> rows;
    for (int r = 0; r < 4; ++r) rows.push_back(propose_row("dense8", r, 313u + static_cast<uint64_t>(r)));
    ProposeArm a(rows, 1), b(rows, 1), c(rows, 1);
    const auto A = run_proposal(base, a, stream);
    const auto B = run_proposal(probe, b, stream);
    const auto C = run_proposal(probe, c, stream);
    require(A.image == B.image && B.image == C.image, "control fixture unstable before mutation");
    std::vector<P> mutated = rows;
    mutated[0].val[0] = std::nextafter(mutated[0].val[0], 1.0f);  // one ulp on the maximum score
    ProposeArm d(mutated, 1);
    const auto D = run_proposal(probe, d, stream);
    require(D.image != C.image, "negative control did not fire: 1-ulp input change left every byte identical");
    size_t state_words = 0, idx_words = 0;
    for (size_t i = 0; i < D.image[4].size(); ++i) state_words += D.image[4][i] != C.image[4][i];
    for (size_t i = 0; i < D.image[0].size(); ++i) idx_words += D.image[0][i] != C.image[0][i];
    require(state_words > 0, "negative control changed idx but not the produced state");
    std::printf("negative_control=input_ulp PASS changed_state_words=%zu changed_idx_words=%zu\n",
                state_words, idx_words);
}

// Only the bound x changes; the frozen verifier must emit a different token cell.
void token_negative_control(const Plugin& verify, const Stream& stream) {
    constexpr int M = 4;
    std::vector<P> rows;
    for (int r = 0; r < M; ++r) rows.push_back(propose_row("sparse2", r, 777u + static_cast<uint64_t>(r)));
    ProposeArm arm(rows, 1);
    ProposeOutcome produced;
    produced.image.assign(5, {});
    produced.image[4].assign(static_cast<size_t>(Pad) + arm.round_pitch() + Pad, Guard);
    for (int r = 0; r < M; ++r) {
        const RadMtpCoupleState s = host_state(0xfeed0000ull + static_cast<uint64_t>(r), 0);
        std::memcpy(produced.image[4].data() + Pad + static_cast<size_t>(r) * arm.pitch[4], &s, sizeof s);
    }
    std::vector<int32_t> xs;
    for (int r = 0; r < M; ++r) xs.push_back(1);
    const auto plain = consume(verify, produced, arm, stream, 20, true, 31, nullptr);
    const auto changed = consume(verify, produced, arm, stream, 20, true, 31, &xs);
    require(plain.acted == M && changed.acted == M, "token control fixture did not exercise the coupling");
    require(plain.token != changed.token,
            "negative control did not fire: a different bound x produced identical token cells");
    std::printf("negative_control=bound_x PASS plain_x0=%d changed_x1=%d acted=%d/%d cells_differ\n",
                plain.token[0], changed.token[0], plain.acted, changed.acted);
}

// ---------------------------------------------------------------- timing
#ifdef PROPOSE_COMPARE_AQL
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
double measure(const Plugin& plug, ProposeArm& arm, const Stream& stream, Events& e, int iters) {
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
        const double pos = q * (static_cast<double>(samples.size()) - 1);
        const auto lo = static_cast<size_t>(pos);
        const auto hi = std::min(lo + 1, samples.size() - 1);
        return samples[lo] + (pos - static_cast<double>(lo)) * (samples[hi] - samples[lo]);
    };
    std::printf(" min_us=%.6f p10_us=%.6f median_us=%.6f p90_us=%.6f max_us=%.6f\n",
                samples.front(), quantile(.1), quantile(.5), quantile(.9), samples.back());
}
void time_rows(const Plugin& base, const Plugin& probe, const Stream& stream, int M,
               const Options& o) {
    std::vector<P> rows;
    for (int r = 0; r < M; ++r) rows.push_back(propose_row("dense8", r, 9001u + static_cast<uint64_t>(r)));
    ProposeArm a(rows, 1), b(rows, 1);
    a.select_round(1);
    b.select_round(1);
    const auto A = run_proposal(base, a, stream);
    const auto B = run_proposal(probe, b, stream);
    require(A.image == B.image, "timing fixture mismatch before timing");
    Events events;
    for (int i = 0; i < o.warmup; ++i) { launch(base, a, stream); launch(probe, b, stream); }
    stream.sync();
    std::vector<double> fa, fb;
    fa.reserve(static_cast<size_t>(o.repeats));
    fb.reserve(static_cast<size_t>(o.repeats));
    for (int repeat = 0; repeat < o.repeats; ++repeat) {
        if (repeat & 1) {
            fb.push_back(measure(probe, b, stream, events, o.iters));
            fa.push_back(measure(base, a, stream, events, o.iters));
        } else {
            fa.push_back(measure(base, a, stream, events, o.iters));
            fb.push_back(measure(probe, b, stream, events, o.iters));
        }
    }
    stream.sync();
    hip(hipGetLastError(), "timed kernel launch");
    require(run_proposal(base, a, stream).image == A.image, "timed frozen run changed output");
    require(run_proposal(probe, b, stream).image == B.image, "timed probe run changed output");
    std::printf("fixture M=%d shortlist=%d warmup_each=%d timed_each=%d repeats=%d total_launches=%d "
                "order=alternating_launch_pairs\n",
                M, Slots, o.warmup, o.iters * o.repeats, o.repeats,
                2 * (1 + o.warmup + o.iters * o.repeats));
    spreads("frozen_mine", fa);
    spreads("probe", fb);
    std::printf("note=isolated proposal-row launch microseconds; not an engine throughput claim\n");
}
#endif

Options options(int argc, char** argv) {
    require(argc >= 5, "usage: compare FROZEN_MINE_SO PROBE_SO --device 0|1 "
                       "[--candidate-row NAME] [--check-only] [--warmup N --iters N --repeats N]");
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
        else require(false, "unknown option: " + key);
    }
    require(o.device == 0 || o.device == 1, "explicit --device 0|1 required");
    require(o.warmup > 0 && o.iters > 0 && o.repeats >= 3 && o.repeats <= 101,
            "warmup/iters must be positive; repeats must be 3..101");
    return o;
}

void run(const char* frozen, const char* probe_path, const Options& o) {
    hip(hipSetDevice(o.device), "device selection");
#ifdef PROPOSE_COMPARE_AQL
    require(rad_dev_set(o.device) == RAD_OK, "AQL device selection failed");
    RadDeviceProps native_props{};
    require(rad_dev_props(o.device, &native_props) == RAD_OK && !native_props.is_host_backend,
            "native comparator requires an accelerator backend");
#endif
    hipDeviceProp_t properties{};
    hip(hipGetDeviceProperties(&properties, o.device), "device properties");
    require(std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0, "gfx1201 required");
    std::printf("device=%d name=%s arch=%s\n", o.device, properties.name, properties.gcnArchName);
    Plugin base(frozen, "mine_mtp_couple_propose", "mtp_couple_propose");
    Plugin probe(probe_path, o.candidate_row, "mtp_couple_propose");
    Plugin verify(frozen, "mine_mtp_couple_verify", "mtp_couple_verify");
    Stream stream;
    Counts counts;

    // Every proposal branch at the probe's row-count boundaries.
    const char* kinds[] = {"dense8", "sparse2", "holes", "duplicate", "underflow-first", "greedy",
                           "stale-card", "nan", "plus-inf", "out-of-range", "empty", "wide-ids"};
    for (int M : {1, 4, 32})
        for (const char* kind : kinds) compare_case(base, probe, stream, kind, M, 1, counts);
    // Every state-plane slab: the engine issues one op per temporal round.
    for (int rounds : {2, 5, 8})
        for (int M : {1, 4}) compare_case(base, probe, stream, "dense8", M, rounds, counts);

    refusal(base, probe, stream);
    for (int M : {1, 4, 32}) describe_check(probe, stream, M);
    require(counts.sampled > 0 && counts.greedy > 0 && counts.malformed > 0 && counts.dropped > 0,
            "proposal branch coverage incomplete: sampled/greedy/malformed/underflow are all required");

    // Downstream: identical proposal state must yield identical token cells at every candidate
    // width the frozen verifier admits, sparse and dense.
    for (int M : {1, 4, 32}) {
        std::vector<P> rows;
        for (int r = 0; r < M; ++r) rows.push_back(propose_row("dense8", r, 606u + static_cast<uint64_t>(r)));
        ProposeArm a(rows, 1), b(rows, 1);
        const auto A = run_proposal(base, a, stream);
        const auto B = run_proposal(probe, b, stream);
        require(A.image == B.image, "consumption fixture state mismatch");
        for (int n : {20, 129, 1024, 1025}) {
            for (bool sparse : {true, false}) {
                const uint64_t seed = 5150u + static_cast<uint64_t>(n);
                const auto ta = consume(verify, A, a, stream, n, sparse, seed, nullptr);
                const auto tb = consume(verify, B, b, stream, n, sparse, seed, nullptr);
                require(ta.token == tb.token, "consumer token cells differ at n_cand=" + std::to_string(n));
                require(ta.acted == tb.acted && ta.accepted == tb.accepted &&
                        ta.rejected == tb.rejected, "consumer outcome split differs at n_cand=" +
                        std::to_string(n));
                require(tb.acted > 0, "consumer wrote no token at n_cand=" + std::to_string(n));
                counts.tokens += tb.acted;
                std::printf("consumer M=%d n_cand=%d support=%s acted=%d accepted=%d rejected=%d "
                            "tokens=identical\n", M, n, sparse ? "sparse" : "dense", tb.acted,
                            tb.accepted, tb.rejected);
            }
        }
    }

    // Negative controls, last, so a control failure cannot be confused with a positive-case failure.
    proposal_negative_control(base, probe, stream);
    token_negative_control(verify, stream);

#ifdef PROPOSE_COMPARE_AQL
    if (!o.check_only) {
        for (int M : {1, 32}) time_rows(base, probe, stream, M, o);
    }
#endif
    std::printf("PASS rows=%d sampled=%d greedy=%d malformed=%d underflow_dropped=%d consumer_tokens=%d "
                "state_bytes=%d timing=%s\n",
                counts.cases, counts.sampled, counts.greedy, counts.malformed, counts.dropped,
                counts.tokens, StateWords * 4,
#ifdef PROPOSE_COMPARE_AQL
                o.check_only ? "disabled" : "alternating_launch_pairs");
#else
                "disabled_ordinary_HIP_check_only");
#endif
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = options(argc, argv);
        run(argv[1], argv[2], o);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
