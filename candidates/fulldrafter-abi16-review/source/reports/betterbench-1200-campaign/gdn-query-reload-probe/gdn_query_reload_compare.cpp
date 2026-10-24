// Frozen libr4d versus the isolated compact CV candidate on an engine-owned native AQL stream.
// Check-only compares actual launches and described dispatches from identical pre-step state.
// Paired timing is available only after the byte comparisons and refusal controls pass.
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_device.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <pthread.h>
#include <sched.h>
#include <limits.h>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_fail = 0, g_ok = 0;

void ok(const char* what, bool cond, const std::string& detail = std::string()) {
    if (cond) { ++g_ok; std::printf("  ok    %s\n", what); return; }
    ++g_fail;
    std::printf("  FAIL  %s%s%s\n", what, detail.empty() ? "" : " -- ", detail.c_str());
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap; va_start(ap, f); std::vsnprintf(b, sizeof b, f, ap); va_end(ap);
    return b;
}

void ck(hipError_t e, const char* what, int line) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL %s (line %d): %s\n", what, line, hipGetErrorString(e));
        std::exit(1);
    }
}
#define CK(e, what) ck((e), (what), __LINE__)

struct Stream {
    RadStream value = nullptr;
    Stream() {
        if (rad_stream_create(&value, 0) != RAD_OK) throw std::runtime_error("native stream");
    }
    void sync() {
        if (rad_stream_sync(value) != RAD_OK) throw std::runtime_error("native sync");
    }
    ~Stream() { if (value) rad_stream_destroy(value); }
};
struct Events {
    RadEvent start = nullptr, stop = nullptr;
    Events() {
        if (rad_event_create(&start) != RAD_OK || rad_event_create(&stop) != RAD_OK)
            throw std::runtime_error("native event");
    }
    ~Events() { if (start) rad_event_destroy(start); if (stop) rad_event_destroy(stop); }
};

// ------------------------------------------------------------------ plugin surface
struct Plugin {
    void* h = nullptr;
    std::string path;
    int (*kernel_count)() = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    uint32_t (*abi)() = nullptr;
};

bool load(Plugin& p, const char* path) {
    p.path = path;
    p.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!p.h) { std::printf("  FAIL  dlopen(%s): %s\n", path, dlerror()); ++g_fail; return false; }
    p.abi          = (uint32_t (*)())dlsym(p.h, "rad_plugin_abi_version");
    p.kernel_count = (int (*)())dlsym(p.h, "rad_kernel_count");
    p.kernel_at    = (const RadKernelInfo* (*)(int))dlsym(p.h, "rad_kernel_at");
    if (!p.abi || !p.kernel_count || !p.kernel_at) {
        std::printf("  FAIL  %s lacks the plugin ABI exports\n", path); ++g_fail; return false;
    }
    if (p.abi() != RAD_ABI_VERSION) {
        std::printf("  FAIL  %s is ABI %u, this comparator is %u\n", path, p.abi(),
                    (unsigned)RAD_ABI_VERSION);
        ++g_fail; return false;
    }
    return true;
}

const RadKernelInfo* find_row(const Plugin& p, const char* name) {
    const int n = p.kernel_count();
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* r = p.kernel_at(i);
        if (r && r->name && std::strcmp(r->name, name) == 0) return r;
    }
    return nullptr;
}

bool owns(const Plugin& p, const void* symbol) {
    Dl_info info{};
    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname) return false;
    char module[PATH_MAX], requested[PATH_MAX];
    return realpath(p.path.c_str(), requested) && realpath(info.dli_fname, module) &&
           std::strcmp(module, requested) == 0;
}

// ------------------------------------------------------------------ host helpers
// Round-to-nearest-even, the selftest's own host mirror. It is NOT the kernel's f2bf (round
// half away); this only builds inputs that are already exactly representable, where they agree.
uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// An avalanche hash, so a wrong pitch or lane map reads unrelated numbers instead of a shifted
// copy of the right ones.
uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

RadTensor T(void* d, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t; std::memset(&t, 0, sizeof t);
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    int i = (int)shape.size() - 1;
    for (auto it = shape.end(); it != shape.begin();) {
        --it;
        t.shape[i] = *it;
        t.stride[i] = acc;
        acc *= *it;
        --i;
    }
    return t;
}

RadParam PI(const char* k, long long v) {
    RadParam p; std::memset(&p, 0, sizeof p);
    p.key = k; p.kind = RAD_P_INT; p.ival = v; return p;
}
RadParam PS(const char* k, const char* v) {
    RadParam p; std::memset(&p, 0, sizeof p);
    p.key = k; p.kind = RAD_P_STR; p.sval = v; return p;
}
RadParam PF(const char* k, double v) {
    RadParam p; std::memset(&p, 0, sizeof p);
    p.key = k; p.kind = RAD_P_F64; p.dval = v; return p;
}

void* dalloc(size_t bytes) {
    void* p = nullptr;
    if (bytes == 0) bytes = 16;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    CK(hipMemset(p, 0, bytes), "hipMemset");
    return p;
}

std::vector<uint8_t> fetch(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    CK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    return h;
}

// ------------------------------------------------------------------ the fixture
struct Geo {
    int64_t K = 128, V = 128, H = 24, Hg = 8, N = 4, W = 4, MQ = 16;
    int64_t SL, conv_dim, in_w, COLS = 6, SSLOTS, CSLOTS = 5, Td = 0;
    std::vector<int32_t> cu, lens, nacc, cidx;
    explicit Geo(std::vector<int32_t> windows = { 4, 2, 3, 0 })
        : N((int64_t)windows.size()), lens(std::move(windows)), nacc((size_t)N, 1), cidx((size_t)N) {
        CSLOTS = N + 1;
        for (int64_t n = 0; n < N; ++n) cidx[(size_t)n] = (int32_t)((n + N - 1) % N);
        SL = W - 2 + MQ;
        conv_dim = 2 * Hg * K + H * V;
        in_w = conv_dim + H * V;
        SSLOTS = N * COLS;
        cu.push_back(0);
        for (int64_t n = 0; n < N; ++n) { Td += lens[(size_t)n]; cu.push_back((int32_t)Td); }
    }
    size_t st_b()  const { return (size_t)(SSLOTS * H * V * K) * 4; }
    size_t cs_b()  const { return (size_t)(CSLOTS * conv_dim * SL) * 2; }
    size_t o_b()   const { return (size_t)(Td * H * V) * 2; }
    size_t oq_b()  const { return (size_t)(Td * H * V); }
    size_t os_b()  const { return (size_t)(Td * H) * 4; }
};

struct Inputs {
    std::vector<uint16_t> hx, hab, hw, hcs;
    std::vector<float> hst, halog, hdtb, hnw;
    std::vector<int32_t> hsidx;
};

void build_inputs(const Geo& g, uint32_t seed, Inputs& in) {
    auto rnd = [&seed]() {
        seed = mix32(seed + 0x9e3779b9u);
        return (float)(seed & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
    };
    auto bf = [&](size_t n, float s) {
        std::vector<uint16_t> v(n);
        for (auto& e : v) e = h_to_bf16(s * rnd());
        return v;
    };
    in.hw  = bf((size_t)(g.conv_dim * g.W), 0.5f);
    in.hcs = bf((size_t)(g.CSLOTS * g.conv_dim * g.SL), 1.0f);
    in.hx  = bf((size_t)(g.Td * g.in_w), 1.0f);
    in.hab = bf((size_t)(g.Td * 2 * g.H), 1.0f);
    in.hst.resize((size_t)(g.SSLOTS * g.H * g.V * g.K));
    for (auto& e : in.hst) e = 0.1f * rnd();
    in.halog.resize((size_t)g.H); in.hdtb.resize((size_t)g.H); in.hnw.resize((size_t)g.V);
    for (int64_t h = 0; h < g.H; ++h) {
        in.halog[(size_t)h] = -2.0f + 0.25f * (float)h;
        in.hdtb[(size_t)h]  = -1.0f + 0.5f * (float)h;
    }
    for (auto& e : in.hnw) e = 1.0f + 0.25f * rnd();
    in.hsidx.assign((size_t)(g.N * g.COLS), -1);
    for (int64_t n = 0; n < g.N; ++n)
        for (int64_t c = 0; c < g.COLS; ++c)
            in.hsidx[(size_t)(n * g.COLS + c)] =
                g.cidx[(size_t)n] < 0 ? -1 : (int32_t)(n * g.COLS + c);
    // Bootstrap replays one zeroed factor; later steps replay only real proposals.
    for (int n = 0; n < g.N; ++n)
        for (int c = 3; c < 6; ++c)
            std::fill_n(in.hst.begin() + (size_t)(n * g.COLS + c) * g.H * g.V * g.K,
                        (size_t)(g.H * g.V * g.K), 0.0f);
}

// ------------------------------------------------------------------ one row's device buffers
struct Buffers {
    void* in = nullptr; void* ab = nullptr; void* w = nullptr;
    void* alog = nullptr; void* dtb = nullptr; void* nw = nullptr;
    void* cidx = nullptr; void* sidx = nullptr; void* nacc = nullptr; void* cu = nullptr;
    void* cs = nullptr; void* st = nullptr; void* ctr_storage = nullptr; void* ctr = nullptr;
    void* o = nullptr; void* oq = nullptr; void* os = nullptr;

    bool alloc(const Geo& g) {
        in   = dalloc((size_t)(g.Td * g.in_w) * 2);
        ab   = dalloc((size_t)(g.Td * 2 * g.H) * 2);
        w    = dalloc((size_t)(g.conv_dim * g.W) * 2);
        alog = dalloc((size_t)g.H * 4);
        dtb  = dalloc((size_t)g.H * 4);
        nw   = dalloc((size_t)g.V * 4);
        cidx = dalloc(sizeof(int32_t) * (size_t)g.N);
        sidx = dalloc(sizeof(int32_t) * (size_t)(g.N * g.COLS));
        nacc = dalloc(sizeof(int32_t) * (size_t)g.N);
        cu   = dalloc(sizeof(int32_t) * (size_t)(g.N + 1));
        cs   = dalloc(g.cs_b());
        st   = dalloc(g.st_b());
        ctr_storage = dalloc((size_t)(g.N * g.Hg) * 4 + 32);
        if (ctr_storage) {
            ctr = (char*)ctr_storage + 16;
            CK(hipMemset(ctr_storage, 0xa7, 16), "counter front guard");
            CK(hipMemset((char*)ctr + (size_t)(g.N * g.Hg) * 4, 0xa7, 16),
               "counter rear guard");
        }
        o    = dalloc(g.o_b());
        oq   = dalloc(g.oq_b());
        os   = dalloc(g.os_b());
        return in && ab && w && alog && dtb && nw && cidx && sidx && nacc && cu && cs && st &&
               ctr && o && oq && os;
    }
    void free_all() {
        void** ps[] = { &in, &ab, &w, &alog, &dtb, &nw, &cidx, &sidx, &nacc, &cu,
                        &cs, &st, &ctr_storage, &o, &oq, &os };
        for (void** p : ps) if (*p) { CK(hipFree(*p), "hipFree"); *p = nullptr; }
        ctr = nullptr;
    }
};

// State and convolution history are uploaded only once per arm. A later step uploads
// its own input rows and the acceptance counts of the PREVIOUS step, never a synthetic state.
void upload_step(const Geo& g, const Inputs& in, Buffers& b) {
    CK(hipMemcpy(b.w, in.hw.data(), in.hw.size() * 2, hipMemcpyHostToDevice), "w");
    CK(hipMemcpy(b.in, in.hx.data(), in.hx.size() * 2, hipMemcpyHostToDevice), "x");
    CK(hipMemcpy(b.ab, in.hab.data(), in.hab.size() * 2, hipMemcpyHostToDevice), "ab");
    CK(hipMemcpy(b.alog, in.halog.data(), in.halog.size() * 4, hipMemcpyHostToDevice), "alog");
    CK(hipMemcpy(b.dtb, in.hdtb.data(), in.hdtb.size() * 4, hipMemcpyHostToDevice), "dtb");
    CK(hipMemcpy(b.nw, in.hnw.data(), in.hnw.size() * 4, hipMemcpyHostToDevice), "nw");
    CK(hipMemcpy(b.cidx, g.cidx.data(), g.cidx.size() * 4, hipMemcpyHostToDevice), "cidx");
    CK(hipMemcpy(b.sidx, in.hsidx.data(), in.hsidx.size() * 4, hipMemcpyHostToDevice), "sidx");
    CK(hipMemcpy(b.nacc, g.nacc.data(), g.nacc.size() * 4, hipMemcpyHostToDevice), "nacc");
    CK(hipMemcpy(b.cu, g.cu.data(), g.cu.size() * 4, hipMemcpyHostToDevice), "cu");
}

void upload_inputs(const Geo& g, const Inputs& in, Buffers& b) {
    CK(hipMemcpy(b.cs, in.hcs.data(), in.hcs.size() * 2, hipMemcpyHostToDevice), "cs");
    CK(hipMemcpy(b.st, in.hst.data(), in.hst.size() * 4, hipMemcpyHostToDevice), "st");
    upload_step(g, in, b);
}

// Poison every output plane before both paths. The T0 sequence has no output rows but still
// commits its anchor; the full state comparison detects a lost T0 commit.
void poison_outputs(const Geo& g, Buffers& b) {
    CK(hipMemset(b.o,  0xA5, g.o_b()),  "o poison");
    CK(hipMemset(b.oq, 0x5A, g.oq_b()), "oq poison");
    CK(hipMemset(b.os, 0x3C, g.os_b()), "os poison");
}

// Every live (token, head) must write all three planes. The T0 sequence has no output row.
bool real_tokens_written(const Geo& g, Buffers& b) {
    const auto o = fetch(b.o, g.o_b()), oq = fetch(b.oq, g.oq_b());
    const auto os = fetch(b.os, g.os_b());
    for (int64_t t = 0; t < g.Td; ++t) for (int64_t h = 0; h < g.H; ++h) {
        const size_t cell = (size_t)(t * g.H + h);
        if (std::all_of(o.begin() + cell * g.V * 2, o.begin() + (cell + 1) * g.V * 2,
                        [](uint8_t x){ return x == 0xa5; }) ||
            std::all_of(oq.begin() + cell * g.V, oq.begin() + (cell + 1) * g.V,
                        [](uint8_t x){ return x == 0x5a; }) ||
            std::all_of(os.begin() + cell * 4, os.begin() + (cell + 1) * 4,
                        [](uint8_t x){ return x == 0x3c; }))
            return false;
    }
    return true;
}

// Describe and launch use the same fixture, stream and scratch. Descriptor values stay live
// until dispatch; no graph is inferred from a successful description alone.
int run_row(const RadKernelInfo* row, const Geo& g, Buffers& b, Stream& stream,
            bool described = false, int invalid = 0, const Plugin* origin = nullptr) {
    RadTensor tx = T(b.in, RAD_BF16, { g.Td, g.conv_dim });
    tx.stride[0] = g.in_w;                                    // column view of a wider row
    RadTensor tz = T(b.in, RAD_BF16, { g.Td, g.H * g.V });
    tz.data = (char*)b.in + (size_t)g.conv_dim * 2;
    tz.stride[0] = g.in_w;
    const RadTensor tcw = T(b.w, RAD_BF16, { g.conv_dim, g.W });
    RadTensor ta = T(b.ab, RAD_BF16, { g.Td, g.H });
    ta.stride[0] = 2 * g.H;
    RadTensor tb = T(b.ab, RAD_BF16, { g.Td, g.H });
    tb.data = (char*)b.ab + (size_t)g.H * 2;
    tb.stride[0] = 2 * g.H;
    const RadTensor talog = T(b.alog, RAD_F32, { g.H });
    const RadTensor tdtb  = T(b.dtb, RAD_F32, { g.H });
    const RadTensor tnw   = T(b.nw, RAD_F32, { g.V });
    const RadTensor tcidx = T(b.cidx, RAD_I32, { g.N });
    const RadTensor tsidx = T(b.sidx, RAD_I32, { g.N, g.COLS });
    const RadTensor tnacc = T(b.nacc, RAD_I32, { g.N });
    const RadTensor tcu   = T(b.cu, RAD_I32, { g.N + 1 });
    const RadTensor tcs   = T(b.cs, RAD_BF16, { g.CSLOTS, g.conv_dim, g.SL });
    const RadTensor tst   = T(b.st, RAD_F32, { g.SSLOTS, g.H, g.V, g.K });
    const RadTensor to    = T(b.o, RAD_BF16, { g.Td, g.H, g.V });
    const RadTensor toq   = T(b.oq, RAD_I8, { g.Td, g.H * g.V });
    const RadTensor tos   = T(b.os, RAD_F32, { g.Td, g.H });

    RadTensor absent; std::memset(&absent, 0, sizeof absent);

    std::array<RadTensor, 18> ts = {{
        tx, tcw, absent, tcs, tcidx, ta, tb, talog, tdtb, tst, tsidx, tnacc, tcu,
        tz, tnw, to, toq, tos
    }};
    if (invalid == 1) ts[9].dtype = RAD_F16;
    if (invalid == 2) ts[16].dtype = RAD_F8E4M3;
    if (invalid == 3) ts[0].stride[0] = INT32_MAX + (int64_t)1;
    if (invalid == 7) ts[11].data = nullptr;
    std::array<RadParam, 9> ps = {{
        PI("q_len", g.MQ), PI("head_k", g.K), PI("head_v", g.V), PI("conv_width", g.W),
        PI("n_head_v", g.H), PF("scale", 1.0 / std::sqrt((double)g.K)), PF("eps", 1e-6), PS("act", "sigmoid"),
        PS("state_form", "anchor")
    }};
    if (invalid == 5) ps[0] = PI("q_len", 17);
    if (invalid == 6) ps[8] = PS("state_form", "per_candidate");
    RadArgs a; std::memset(&a, 0, sizeof a);
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.scratch = b.ctr; a.scratch_bytes = (int64_t)(g.N * g.Hg) * 4 - (invalid == 4);
    a.world_size = 1;
    if (!invalid && row->scratch &&
        row->scratch(&a) != (int64_t)(g.N * g.Hg) * 4) return RAD_E_SCRATCH;
    if (!described) return row->launch(&a, stream.value);
    if (!row->describe) return RAD_E_UNSUPPORTED;
    RadLaunchDesc desc{};
    int n = -1;
    if (invalid) return row->describe(&a, stream.value, &desc, 1, &n);
    int err = row->describe(&a, stream.value, &desc, 0, &n);
    if (err != RAD_E_FULL || n != 0) return RAD_E_INVAL;
    err = row->describe(&a, stream.value, &desc, 1, &n);
    if (err != RAD_OK || n != 1 || desc.n_params != 20 ||
        (origin && !owns(*origin, desc.func))) return RAD_E_INVAL;
    return hipLaunchKernel(desc.func, dim3(desc.grid[0], desc.grid[1], desc.grid[2]),
                           dim3(desc.block[0], desc.block[1], desc.block[2]),
                           desc.params, desc.shared_bytes,
                           (hipStream_t)stream.value) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

bool all_equal(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
               const char* step, std::string* why) {
    if (a.size() != b.size()) { *why = fmt("%s %s: size mismatch", step, what); return false; }
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { if (!diff) first = i; ++diff; }
    if (!diff) return true;
    *why = fmt("%s %s: %zu of %zu bytes differ, first at %zu", step, what, diff, a.size(), first);
    return false;
}

// nacc counts accepted tokens from the PREVIOUS step. Live CV calls need at least one
// to index their convolution history; only a T0 following T0 may accept zero.
// Every arm keeps its own state from the seed through rollback, full acceptance and T0.
void run_case(const Plugin& frozen, const Plugin& cand, Stream& stream, int steps) {
    std::array<Geo, 5> schedule = {
        Geo({3, 3, 3, 3}), Geo({1, 3, 1, 0}), Geo({3, 1, 3, 0}),
        Geo({16, 1, 3, 0}), Geo({1, 1, 1, 0})
    };
    schedule[0].nacc = {1, 1, 1, 1}; // bootstrap: zeroed factor blocks
    schedule[1].nacc = {1, 1, 3, 3};
    schedule[2].nacc = {1, 3, 1, 0};
    schedule[3].nacc = {3, 1, 2, 0};
    schedule[4].nacc = {16, 1, 3, 0};
    const Geo capacity({16, 16, 16, 16});
    Inputs initial;
    build_inputs(schedule[0], 0x51ed270bu, initial);
    Buffers bf, bc, bd;
    if (!bf.alloc(capacity) || !bc.alloc(capacity) || !bd.alloc(capacity)) {
        ok("causal T0/T1/T3/T16 trajectory", false, "allocation");
        bf.free_all(); bc.free_all(); bd.free_all(); return;
    }
    const RadKernelInfo* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const RadKernelInfo* rc = find_row(cand, "gdn_conv_recurrent_update_k128_v128_bf16_qreload");
    upload_inputs(schedule[0], initial, bf);
    upload_inputs(schedule[0], initial, bc);
    upload_inputs(schedule[0], initial, bd);
    std::string why;
    for (int step = 0; step < steps && why.empty(); ++step) {
        const Geo& g = step < (int)schedule.size() ? schedule[(size_t)step] : schedule.back();
        const Geo& previous = step <= 1 ? schedule[0]
                              : schedule[(size_t)std::min(step - 1, (int)schedule.size() - 1)];
        if (step >= (int)schedule.size()) {
            // The steady T1/T0 continuation accepts only factors written by the prior T1.
            schedule.back().nacc = {1, 1, 1, 0};
        }
        if (step) {
            Inputs next;
            build_inputs(g, 0x51ed270bu + (uint32_t)step, next);
            upload_step(g, next, bf);
            upload_step(g, next, bc);
            upload_step(g, next, bd);
        }
        for (size_t n = 0; n < g.nacc.size(); ++n)
            if (g.nacc[n] < 0 || g.nacc[n] > previous.lens[n]) {
                why = fmt("step %d sequence %zu: accepted %d of %d", step, n,
                          g.nacc[n], previous.lens[n]);
                break;
            }
        if (!why.empty()) break;
        const size_t anchor = (size_t)((g.N - 1) * g.COLS + 2) * g.H * g.V * g.K * 4;
        std::vector<uint8_t> before;
        if (step == 1) before = fetch(bf.st, capacity.st_b());
        poison_outputs(g, bf); poison_outputs(g, bc); poison_outputs(g, bd);
        const int ef = run_row(rf, g, bf, stream);
        const int ec = run_row(rc, g, bc, stream);
        const int ed = run_row(rc, g, bd, stream, true, 0, &cand);
        if (ef != RAD_OK || ec != RAD_OK || ed != RAD_OK) {
            why = fmt("step %d: frozen=%d candidate=%d described=%d", step, ef, ec, ed);
            break;
        }
        stream.sync();
        if (!real_tokens_written(g, bf) || !real_tokens_written(g, bc) ||
            !real_tokens_written(g, bd)) {
            why = fmt("step %d: output, quant or scale row not written", step); break;
        }
        const struct { const char* name; size_t bytes; void* Buffers::*field; } planes[] = {
            { "o", g.o_b(), &Buffers::o }, { "o_q", g.oq_b(), &Buffers::oq },
            { "o_scale", g.os_b(), &Buffers::os },
            { "state/anchor/factors", capacity.st_b(), &Buffers::st },
            { "conv_state", capacity.cs_b(), &Buffers::cs }
        };
        const std::string s = fmt("step %d", step);
        for (const auto& p : planes) {
            const auto f = fetch(bf.*(p.field), p.bytes);
            if (!all_equal(p.name, f, fetch(bc.*(p.field), p.bytes), s.c_str(), &why) ||
                !all_equal(p.name, f, fetch(bd.*(p.field), p.bytes), "described", &why))
                break;
        }
        if (step == 0 && why.empty()) {
            const size_t slot = (size_t)((g.N - 1) * g.COLS + 4) * g.H * g.V * g.K;
            const size_t last = slot + (size_t)(g.lens.back() - 1) * (4 * g.K + g.V + 8);
            const auto state = fetch(bf.st, capacity.st_b());
            if (std::memcmp(state.data() + last * 4, initial.hst.data() + last,
                            (size_t)(4 * g.K + g.V + 8) * 4) == 0)
                why = "seed did not write the final factor block for the later T0 replay";
        }
        if (step == 1 && why.empty()) {
            const auto state = fetch(bf.st, capacity.st_b());
            if (std::memcmp(state.data() + anchor, before.data() + anchor,
                            (size_t)(g.H * g.V * g.K) * 4) == 0)
                why = "T0 did not commit the preceding T3 accepted factors";
        }
        for (const auto* arm : { &bf, &bc, &bd }) {
            const auto ctr = fetch(arm->ctr, (size_t)(g.N * g.Hg) * 4);
            const auto guard = fetch(arm->ctr_storage, ctr.size() + 32);
            if (!std::all_of(ctr.begin(), ctr.end(), [](uint8_t x){ return x == 0; }) ||
                !std::all_of(guard.begin(), guard.begin() + 16,
                             [](uint8_t x){ return x == 0xa7; }) ||
                !std::all_of(guard.end() - 16, guard.end(),
                             [](uint8_t x){ return x == 0xa7; }))
                why = fmt("step %d: arrival counter or guard corrupted", step);
        }
    }
    ok("causal T0/T1/T3/T16 trajectory", why.empty(), why);
    bf.free_all(); bc.free_all(); bd.free_all();
}

// CONTROL 1: the frozen row must advance BOTH caches, or every later equality is two untouched
// buffers.
void check_moved(const Plugin& frozen, Stream& stream) {
    const Geo g;
    Inputs in;
    build_inputs(g, 0x51ed270bu, in);
    Buffers b;
    if (!b.alloc(g)) { ok("control: state moved", false, "hipMalloc"); b.free_all(); return; }
    upload_inputs(g, in, b);
    const RadKernelInfo* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const int e = rf ? run_row(rf, g, b, stream) : -1;
    stream.sync();
    const std::vector<uint8_t> st = fetch(b.st, g.st_b());
    const std::vector<uint8_t> cs = fetch(b.cs, g.cs_b());
    const uint8_t* st0 = (const uint8_t*)in.hst.data();
    const uint8_t* cs0 = (const uint8_t*)in.hcs.data();
    const bool st_moved = std::memcmp(st.data(), st0, g.st_b()) != 0;
    const bool cs_moved = std::memcmp(cs.data(), cs0, g.cs_b()) != 0;

    std::string why = e != RAD_OK ? fmt("rc=%d", e)
                    : !st_moved    ? "the recurrent state never moved"
                    : !cs_moved    ? "the conv state never moved" : "";
    ok("control: the frozen row advances BOTH state caches", why.empty(), why);
    b.free_all();
}

// CONTROL 2: one changed input byte must move the frozen row's output, or the comparison path
// itself is dead and every equality is vacuous.
void check_sensitive(const Plugin& frozen, Stream& stream) {
    const Geo g;
    Inputs a, b;
    build_inputs(g, 0x51ed270bu, a);
    build_inputs(g, 0x51ed270bu, b);
    // Flip the SIGN of one bf16 q channel of token 0. A low-bit change could round back to the
    // same bf16 output and turn this control into a false failure; a sign flip cannot. The
    // channel feeds the folded convolution, so the q this op normalises and accumulates changes.
    b.hx[0] = (uint16_t)(b.hx[0] ^ 0x8000u);

    Buffers ba, bb;
    if (!ba.alloc(g) || !bb.alloc(g)) {
        ok("control: comparison path is live", false, "hipMalloc");
        ba.free_all(); bb.free_all(); return;
    }
    upload_inputs(g, a, ba);
    upload_inputs(g, b, bb);
    const RadKernelInfo* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const int ea = rf ? run_row(rf, g, ba, stream) : -1;
    const int eb = rf ? run_row(rf, g, bb, stream) : -1;
    stream.sync();
    const std::vector<uint8_t> oa = fetch(ba.o, g.o_b());
    const std::vector<uint8_t> ob = fetch(bb.o, g.o_b());
    std::string why = (ea != RAD_OK || eb != RAD_OK) ? fmt("rc=%d/%d", ea, eb)
                    : oa == ob ? "a changed q byte did not change the output" : "";
    ok("control: a one-byte input change moves the frozen row's output", why.empty(), why);
    auto changed = oa;
    changed[0] ^= 1;
    CK(hipMemcpy(bb.o, changed.data(), changed.size(), hipMemcpyHostToDevice),
       "negative control consumer output");
    std::string rejected;
    ok("one consumer output byte differs and comparison rejects",
       !all_equal("o", oa, fetch(bb.o, g.o_b()), "negative control", &rejected),
       rejected);
    ba.free_all();
    bb.free_all();
}

void check_refusals(const Plugin& candidate, Stream& stream) {
    const Geo g;
    Inputs in;
    build_inputs(g, 0x51ed270bu, in);
    Buffers b;
    if (!b.alloc(g)) { ok("refusal fixture", false, "allocation"); b.free_all(); return; }
    upload_inputs(g, in, b);
    const auto* row = find_row(candidate, "gdn_conv_recurrent_update_k128_v128_bf16_qreload");
    const int expected[] = { RAD_E_DTYPE, RAD_E_DTYPE, RAD_E_STRIDE, RAD_E_SCRATCH,
                             RAD_E_SHAPE, RAD_E_UNSUPPORTED, RAD_E_INVAL };
    for (int i = 1; i <= 7; ++i) {
        const int launch = run_row(row, g, b, stream, false, i);
        const int described = run_row(row, g, b, stream, true, i);
        ok(fmt("unsupported case %d refuses launch and describe", i).c_str(),
           launch == expected[i - 1] && described == expected[i - 1],
           fmt("launch=%d describe=%d expected=%d", launch, described, expected[i - 1]));
    }
    stream.sync();
    b.free_all();
}

double measure(const RadKernelInfo* row, const Geo& g, Buffers& b, Stream& stream,
               Events& events, int iters, const Plugin* described = nullptr) {
    if (rad_event_record(events.start, stream.value) != RAD_OK)
        throw std::runtime_error("start event");
    for (int i = 0; i < iters; ++i)
        if (run_row(row, g, b, stream, described != nullptr, 0, described) != RAD_OK)
            throw std::runtime_error("timed launch");
    if (rad_event_record(events.stop, stream.value) != RAD_OK ||
        rad_event_sync(events.stop) != RAD_OK)
        throw std::runtime_error("stop event");
    float ms = 0;
    if (rad_event_elapsed_ms(events.start, events.stop, &ms) != RAD_OK ||
        !std::isfinite(ms) || ms <= 0)
        throw std::runtime_error("event duration");
    return ms * 1000.0 / iters;
}

void spreads(const char* name, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    auto quantile = [&](double p) {
        const double pos = p * (samples.size() - 1);
        const size_t lo = (size_t)pos, hi = std::min(lo + 1, samples.size() - 1);
        return samples[lo] + (pos - lo) * (samples[hi] - samples[lo]);
    };
    std::printf("%s min=%.6f p10=%.6f median=%.6f p90=%.6f max=%.6f us\n",
                name, samples.front(), quantile(.1), quantile(.5), quantile(.9),
                samples.back());
}

void paired_timing(const Plugin& frozen, const Plugin& candidate, Stream& stream,
                   int warmup, int iters, int repeats, int sequences) {
    Geo seed(std::vector<int32_t>((size_t)sequences, 3));
    Geo g(std::vector<int32_t>((size_t)sequences, 1));
    Inputs initial, next;
    build_inputs(seed, 0x51ed270bu, initial);
    build_inputs(g, 0x51ed270cu, next);
    Buffers bf, bc, bd;
    if (!bf.alloc(seed) || !bc.alloc(seed) || !bd.alloc(seed))
        throw std::runtime_error("timing allocation");
    const auto* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const auto* rc = find_row(candidate, "gdn_conv_recurrent_update_k128_v128_bf16_qreload");
    for (Buffers* b : { &bf, &bc, &bd }) {
        upload_inputs(seed, initial, *b);
        poison_outputs(seed, *b);
    }
    if (run_row(rf, seed, bf, stream) != RAD_OK ||
        run_row(rc, seed, bc, stream) != RAD_OK ||
        run_row(rc, seed, bd, stream, true, 0, &candidate) != RAD_OK)
        throw std::runtime_error("timing seed");
    stream.sync();
    if (!real_tokens_written(seed, bf) || !real_tokens_written(seed, bc) ||
        !real_tokens_written(seed, bd)) throw std::runtime_error("timing seed output");
    auto compare = [&](const char* phase) {
        std::string why;
        const struct { const char* name; size_t bytes; void* Buffers::*field; } planes[] = {
            { "o", g.o_b(), &Buffers::o }, { "o_q", g.oq_b(), &Buffers::oq },
            { "o_scale", g.os_b(), &Buffers::os }, { "state/anchor/factors", seed.st_b(), &Buffers::st },
            { "conv_state", seed.cs_b(), &Buffers::cs }
        };
        for (const auto& p : planes) {
            const auto f = fetch(bf.*(p.field), p.bytes);
            if (!all_equal(p.name, f, fetch(bc.*(p.field), p.bytes), phase, &why) ||
                !all_equal(p.name, f, fetch(bd.*(p.field), p.bytes), phase, &why)) break;
        }
        for (const Buffers* b : { &bf, &bc, &bd }) {
            const auto ctr = fetch(b->ctr, (size_t)(g.N * g.Hg) * 4);
            const auto guard = fetch(b->ctr_storage, ctr.size() + 32);
            if (!std::all_of(ctr.begin(), ctr.end(), [](uint8_t x){ return x == 0; }) ||
                !std::all_of(guard.begin(), guard.begin() + 16,
                             [](uint8_t x){ return x == 0xa7; }) ||
                !std::all_of(guard.end() - 16, guard.end(),
                             [](uint8_t x){ return x == 0xa7; }))
                why = fmt("%s: arrival counter or guard corrupted", phase);
        }
        if (!why.empty()) throw std::runtime_error(why);
    };
    // One proposal from the T3 seed is accepted. Every later T1 accepts its sole proposal.
    // Keep the three distinct resulting states; never restore a post-step snapshot.
    for (Buffers* b : { &bf, &bc, &bd }) {
        upload_step(g, next, *b);
        poison_outputs(g, *b);
    }
    for (int i = 0; i < warmup; ++i) {
        if (run_row(rf, g, bf, stream) != RAD_OK ||
            run_row(rc, g, bc, stream) != RAD_OK ||
            run_row(rc, g, bd, stream, true, 0, &candidate) != RAD_OK)
            throw std::runtime_error("warmup");
        stream.sync();
        compare("warmup");
    }
    if (!real_tokens_written(g, bf) || !real_tokens_written(g, bc) ||
        !real_tokens_written(g, bd)) throw std::runtime_error("timing warmup output");
    ok("timing history agrees before measurement", true);
    Events events;
    std::vector<double> baseline, probe, described;
    for (int r = 0; r < repeats; ++r) {
        if (r & 1) {
            described.push_back(measure(rc, g, bd, stream, events, iters, &candidate));
            probe.push_back(measure(rc, g, bc, stream, events, iters));
            baseline.push_back(measure(rf, g, bf, stream, events, iters));
        } else {
            baseline.push_back(measure(rf, g, bf, stream, events, iters));
            probe.push_back(measure(rc, g, bc, stream, events, iters));
            described.push_back(measure(rc, g, bd, stream, events, iters, &candidate));
        }
        compare("after timed batch");
    }
    ok("timed batches preserve output, state, counters and guards", true);
    if (!g_fail) {
        std::printf("native AQL event submission, seeded T1 N=%lld H=%lld Hg=%lld iterations=%d repeats=%d (not kernel-only)\n",
                    (long long)g.N, (long long)g.H, (long long)g.Hg, iters, repeats);
        spreads("frozen", baseline); spreads("candidate", probe); spreads("described", described);
    }
    bf.free_all(); bc.free_all(); bd.free_all();
}

}  // namespace

int main(int argc, char** argv) {
    const char* frozen_path = "build_hip/radiance_home/kernels/libr4d.so";
    const char* cand_path = "/tmp/gdn-query-reload.so";
    int device = 0, steps = 5, warmup = 10, iters = 100, repeats = 7;
    int sequences = 1;
    bool timing = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--timing")) { timing = true; continue; }
        if (!std::strcmp(argv[i], "--check-only")) { timing = false; continue; }
        if (i + 1 == argc) return 2;
        if (!std::strcmp(argv[i], "--frozen")) { frozen_path = argv[++i]; continue; }
        if (!std::strcmp(argv[i], "--candidate")) { cand_path = argv[++i]; continue; }
        char* end = nullptr;
        const long value = std::strtol(argv[++i], &end, 10);
        if (!end || *end || value < 0 || value > 1000000) return 2;
        if (!std::strcmp(argv[i - 1], "--steps")) steps = (int)value;
        else if (!std::strcmp(argv[i - 1], "--device")) device = (int)value;
        else if (!std::strcmp(argv[i - 1], "--warmup")) warmup = (int)value;
        else if (!std::strcmp(argv[i - 1], "--iters")) iters = (int)value;
        else if (!std::strcmp(argv[i - 1], "--repeats")) repeats = (int)value;
        else if (!std::strcmp(argv[i - 1], "--sequences")) sequences = (int)value;
        else return 2;
    }
    if (steps < 5 || warmup < 1 || iters < 1 || repeats < 3 || repeats > 101 ||
        sequences < 1 || sequences > 8) return 2;
    const char* anchor = std::getenv("RADIANCE_GDN_ANCHOR");
    if (!anchor || anchor[0] != '1') {
        std::fprintf(stderr, "FAIL: set RADIANCE_GDN_ANCHOR=1\n");
        return 1;
    }
    try {
        CK(hipSetDevice(device), "hipSetDevice");
        if (rad_dev_set(device) != RAD_OK) throw std::runtime_error("native AQL device");
        RadDeviceProps props{};
        if (rad_dev_props(device, &props) != RAD_OK || props.is_host_backend)
            throw std::runtime_error("native GPU backend");
        // This same thread calls both plugin launches and dispatches described kernels.
        const int cpu = sched_getcpu();
        if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("launch CPU selection");
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        CPU_SET(cpu, &allowed);
        if (pthread_setaffinity_np(pthread_self(), sizeof(allowed), &allowed))
            throw std::runtime_error("launch thread affinity");
        Stream stream;
        Plugin frozen, cand;
        if (!load(frozen, frozen_path) || !load(cand, cand_path)) return 1;
        const auto* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
        const auto* rc = find_row(cand, "gdn_conv_recurrent_update_k128_v128_bf16_qreload");
        ok("two rows from their own DSOs with one op",
           rf && rc && owns(frozen, (const void*)rf->launch) &&
           owns(cand, (const void*)rc->launch) && rf->launch != rc->launch &&
           rf->op && rc->op &&
           !std::strcmp(rf->op, "gdn_conv_recurrent_update") &&
           !std::strcmp(rc->op, "gdn_conv_recurrent_update") &&
           rc->describe && rc->scratch);
        if (g_fail) return 1;
        check_refusals(cand, stream);
        if (!g_fail) { check_moved(frozen, stream); check_sensitive(frozen, stream); }
        if (!g_fail) run_case(frozen, cand, stream, steps);
        if (!g_fail && timing) paired_timing(frozen, cand, stream, warmup, iters, repeats, sequences);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%d ok, %d failure(s)\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}