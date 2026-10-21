// Frozen libr4d versus the isolated CV candidate on an engine-owned native AQL stream.
// Check-only compares actual launches and described dispatches from identical pre-step state.
// Paired timing runs only after the byte comparisons and refusal controls pass.
//
// WHY THE TOKEN COUNT IS AN ARGUMENT. The folded kernel pays its per-token work -- the query
// reload, the LDS barriers, the conv prologue and the factor replay -- once per token in a
// sequence. A T=1 comparison cannot see any of it: that shape is the one that hides exactly the
// cost this comparator exists to attribute. The deployed MTP3 verify step runs T=4, so every
// timing arm is seeded at T=max(3,tokens) and then measured at T=tokens with an acceptance count
// equal to that width, which is valid against the factor rows the previous step actually wrote.
// The seed runs outside the events; no arm is ever restored to a completed snapshot.
//
// BOTH CANDIDATE GENERATIONS ARE SUPPORTED WITHOUT A FLAG. A row that describes one dispatch is
// the existing probe; a row that describes two is the scratch-clear prototype (counter clear, then
// the main kernel). Capacity 0, 1 and 2 are exercised on every described call, every returned
// descriptor's origin is checked, and all of them are launched in the order the row returned them.
// The main descriptor stays inside the ABI (20 parameters, 224 bytes of blob) while the clear has
// its own arity (2 parameters) -- so the arity is read from the descriptor, never assumed.
//
// --poison-counters fills the candidate's counter payload (guards excluded) before every numeric
// step and before each timed batch, outside the events and never between iterations of a timed
// batch. With it enabled the row must also answer a parameters-only scratch declaration with an
// honest, non-negative, covering bound: that is what shows the emitted clear handles arbitrary
// reused scratch and not merely a fresh zero.
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_device.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cstdarg>
#include <cstdint>
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
bool g_bias = false;       // --conv-bias: nonzero bf16 conv bias fixture
bool g_poison = false;     // --poison-counters: scratch-clear prototype, reused scratch
int  g_dispatches = 0;     // dispatch count the candidate's describe reported (1 or 2)

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
    bool bias = false;
    std::vector<int32_t> cu, lens, nacc, cidx;
    explicit Geo(std::vector<int32_t> windows = { 4, 2, 3, 0 })
        : N((int64_t)windows.size()), bias(g_bias), lens(std::move(windows)),
          nacc((size_t)N, 1), cidx((size_t)N) {
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
    std::vector<uint16_t> hx, hab, hw, hcs, hcb;
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
    in.hcb.clear();
    if (g.bias) {
        // A NONZERO bias on every one of the conv_dim channels: a zero bias makes the optional
        // operand indistinguishable from its absence, which would make the bias fixture vacuous.
        in.hcb.resize((size_t)g.conv_dim);
        for (auto& e : in.hcb) {
            const uint16_t v = h_to_bf16(0.5f * rnd());
            e = (v & 0x7fffu) ? v : (uint16_t)0x3c00;
        }
    }
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

// ------------------------------------------------------------------ one arm's device buffers
struct Buffers {
    void* in = nullptr; void* ab = nullptr; void* w = nullptr;
    void* alog = nullptr; void* dtb = nullptr; void* nw = nullptr;
    void* cidx = nullptr; void* sidx = nullptr; void* nacc = nullptr; void* cu = nullptr;
    void* cs = nullptr; void* st = nullptr; void* ctr_storage = nullptr; void* ctr = nullptr;
    void* o = nullptr; void* oq = nullptr; void* os = nullptr; void* cb = nullptr;

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
        if (g.bias) cb = dalloc((size_t)g.conv_dim * 2);
        o    = dalloc(g.o_b());
        oq   = dalloc(g.oq_b());
        os   = dalloc(g.os_b());
        return in && ab && w && alog && dtb && nw && cidx && sidx && nacc && cu && cs && st &&
               ctr && o && oq && os && (!g.bias || cb);
    }
    void free_all() {
        void** ps[] = { &in, &ab, &w, &alog, &dtb, &nw, &cidx, &sidx, &nacc, &cu,
                        &cs, &st, &ctr_storage, &o, &oq, &os, &cb };
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
    if (g.bias && b.cb)
        CK(hipMemcpy(b.cb, in.hcb.data(), in.hcb.size() * 2, hipMemcpyHostToDevice), "conv_b");
    upload_step(g, in, b);
}

// Poison every output plane before both paths. The T0 sequence has no output rows but still
// commits its anchor; the full state comparison detects a lost T0 commit.
void poison_outputs(const Geo& g, Buffers& b) {
    CK(hipMemset(b.o,  0xA5, g.o_b()),  "o poison");
    CK(hipMemset(b.oq, 0x5A, g.oq_b()), "oq poison");
    CK(hipMemset(b.os, 0x3C, g.os_b()), "os poison");
}

// Scratch belongs to the ordered operation, not to an initially zeroed allocation. The front and
// rear guards are deliberately left alone: they are the control that the payload write stopped
// where it should.
void poison_counters(const Geo& g, Buffers& b) {
    CK(hipMemset(b.ctr, 0x5b, (size_t)(g.N * g.Hg) * 4), "counter poison");
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

// The complement of real_tokens_written over a row range: a padded sequence must leave exactly the
// poison it was given, in all three planes.
bool poison_intact(const Geo& g, const Buffers& b, int64_t t0, int64_t t1) {
    const auto o = fetch(b.o, g.o_b()), oq = fetch(b.oq, g.oq_b());
    const auto os = fetch(b.os, g.os_b());
    for (int64_t t = t0; t < t1; ++t) for (int64_t h = 0; h < g.H; ++h) {
        const size_t cell = (size_t)(t * g.H + h);
        if (!std::all_of(o.begin() + cell * g.V * 2, o.begin() + (cell + 1) * g.V * 2,
                         [](uint8_t x){ return x == 0xa5; }) ||
            !std::all_of(oq.begin() + cell * g.V, oq.begin() + (cell + 1) * g.V,
                         [](uint8_t x){ return x == 0x5a; }) ||
            !std::all_of(os.begin() + cell * 4, os.begin() + (cell + 1) * 4,
                         [](uint8_t x){ return x == 0x3c; }))
            return false;
    }
    return true;
}

// The same three planes over an ACTIVE row range: those rows must no longer hold their poison.
bool rows_written(const Geo& g, const Buffers& b, int64_t t0, int64_t t1) {
    const auto o = fetch(b.o, g.o_b()), oq = fetch(b.oq, g.oq_b());
    const auto os = fetch(b.os, g.os_b());
    for (int64_t t = t0; t < t1; ++t) for (int64_t h = 0; h < g.H; ++h) {
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

// Describe and launch use the same fixture, stream and scratch. Descriptor values stay live until
// dispatch; no graph is inferred from a successful description alone.
//
// CAPACITY IS PART OF THE CONTRACT, NOT A DETAIL. Capacity 0 must report a full sink and record
// nothing. Capacity 1 records whatever fits: the one-dispatch rows return RAD_OK with n == 1 and
// the two-dispatch row returns RAD_E_FULL with n == 1, because its second emit hits the cap. The
// arity therefore comes from a capacity-2 describe, and every returned descriptor is origin-checked
// and then launched in the returned order -- a clear that is described but never dispatched would
// leave the next numeric step reading another step's arrivals.
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
    const RadTensor tcb = g.bias ? T(b.cb, RAD_BF16, { g.conv_dim }) : absent;

    std::array<RadTensor, 18> ts = {{
        tx, tcw, tcb, tcs, tcidx, ta, tb, talog, tdtb, tst, tsidx, tnacc, tcu,
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
    const int64_t need = (int64_t)(g.N * g.Hg) * 4;
    a.scratch = b.ctr; a.scratch_bytes = need - (invalid == 4);
    a.world_size = 1;
    // A covering bound, not an exact one: the one-dispatch probe returns the need itself and the
    // clear prototype returns its fixed pool. Both must cover this launch.
    if (!invalid && row->scratch && row->scratch(&a) < need) return RAD_E_SCRATCH;

    if (!described) return row->launch(&a, stream.value);
    if (!row->describe) return RAD_E_UNSUPPORTED;

    std::array<RadLaunchDesc, 2> d{};
    int n = -1;
    if (invalid) return row->describe(&a, stream.value, d.data(), 2, &n);
    int err = row->describe(&a, stream.value, d.data(), 0, &n);
    if (err != RAD_E_FULL || n != 0) return RAD_E_INVAL;
    err = row->describe(&a, stream.value, d.data(), 1, &n);
    if ((err != RAD_OK && err != RAD_E_FULL) || n != 1) return RAD_E_INVAL;
    const bool full_at_one = err == RAD_E_FULL;
    err = row->describe(&a, stream.value, d.data(), 2, &n);
    if (err != RAD_OK || n != (full_at_one ? 2 : 1)) return RAD_E_INVAL;
    g_dispatches = n;

    const int main_i = n - 1;
    if (d[(size_t)main_i].n_params != 20 ||
        d[(size_t)main_i].param_size[18] != sizeof(void*) ||
        d[(size_t)main_i].param_size[19] != 72 ||
        (origin && !owns(*origin, d[(size_t)main_i].func)))
        return RAD_E_INVAL;
    void* main_ptr = nullptr;
    std::memcpy(&main_ptr, d[(size_t)main_i].params[18], sizeof main_ptr);
    if (main_ptr != b.ctr) return RAD_E_INVAL;

    if (n == 2) {
        // The clear has its own arity: (unsigned int* counters, int count). Its counter pointer is
        // the same scratch the main kernel receives, and its grid covers every (sequence, head).
        void* clear_ptr = nullptr;
        int clear_count = 0;
        if (d[0].n_params != 2 || d[0].param_size[0] != sizeof(clear_ptr) ||
            d[0].param_size[1] != sizeof(clear_count) ||
            d[0].func == d[1].func || (origin && !owns(*origin, d[0].func)))
            return RAD_E_INVAL;
        std::memcpy(&clear_ptr, d[0].params[0], sizeof clear_ptr);
        std::memcpy(&clear_count, d[0].params[1], sizeof clear_count);
        if (clear_ptr != b.ctr || clear_count != g.N * g.Hg) return RAD_E_INVAL;
        if ((uint64_t)d[0].grid[0] * d[0].block[0] < (uint64_t)g.N * g.Hg) return RAD_E_INVAL;
    }
    for (int i = 0; i < n; ++i) {
        auto& k = d[(size_t)i];
        if (hipLaunchKernel(k.func, dim3(k.grid[0], k.grid[1], k.grid[2]),
                            dim3(k.block[0], k.block[1], k.block[2]),
                            k.params, k.shared_bytes,
                            (hipStream_t)stream.value) != hipSuccess)
            return RAD_E_DEVICE;
    }
    return RAD_OK;
}

bool counters_clean(const Geo& g, const Buffers& b, std::string* why) {
    const auto ctr = fetch(b.ctr, (size_t)(g.N * g.Hg) * 4);
    const auto guard = fetch(b.ctr_storage, ctr.size() + 32);
    if (!std::all_of(ctr.begin(), ctr.end(), [](uint8_t x){ return x == 0; }) ||
        !std::all_of(guard.begin(), guard.begin() + 16, [](uint8_t x){ return x == 0xa7; }) ||
        !std::all_of(guard.end() - 16, guard.end(), [](uint8_t x){ return x == 0xa7; })) {
        *why = "arrival counter or guard corrupted";
        return false;
    }
    return true;
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
    if (g_bias && (!bf.cb || !bc.cb || !bd.cb ||
                   bf.cb == bc.cb || bc.cb == bd.cb || bf.cb == bd.cb))
        ok("each arm has its own conv bias buffer", false, "shared or missing bias pointer");
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
        // The candidate and its described copy see a scratch payload nobody zeroed, and the clear
        // the clear-prototype row emits is what has to deal with it. Never inside an event pair,
        // and never between two iterations of one batch.
        if (g_poison) { poison_counters(g, bc); poison_counters(g, bd); }
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
            std::string bad;
            if (why.empty() && !counters_clean(g, *arm, &bad))
                why = fmt("step %d: %s", step, bad.c_str());
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

// CONTROL 3 (--conv-bias only): the optional bias operand has to be read. Two frozen launches on
// identical inputs, one with the operand and one without, on two arms that own their bias
// buffers, must disagree -- otherwise the fixture cannot show the bias path is exercised at all.
void check_bias(const Plugin& frozen, Stream& stream) {
    Geo with({4, 2, 3, 0});
    with.bias = true;
    Geo without = with;
    without.bias = false;
    const Geo capacity = with;
    Inputs in;
    build_inputs(with, 0x51ed270bu, in);
    Buffers ba, bb;
    if (!ba.alloc(capacity) || !bb.alloc(capacity)) {
        ok("control: conv bias is live", false, "hipMalloc");
        ba.free_all(); bb.free_all(); return;
    }
    // Both arms are allocated for the bias fixture and uploaded from the same host values, so the
    // contents are equal and the device pointers are not.
    upload_inputs(with, in, ba);
    upload_inputs(with, in, bb);
    const bool nonzero = std::all_of(in.hcb.begin(), in.hcb.end(),
                                     [](uint16_t v){ return (v & 0x7fffu) != 0; });
    ok("conv bias fixture is nonzero on every conv_dim channel and per-arm",
       nonzero && ba.cb && bb.cb && ba.cb != bb.cb,
       fmt("bias=%p/%p channels=%lld", ba.cb, bb.cb, (long long)with.conv_dim));
    const RadKernelInfo* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const int ea = rf ? run_row(rf, with, ba, stream) : -1;
    const int eb = rf ? run_row(rf, without, bb, stream) : -1;
    stream.sync();
    const auto oa = fetch(ba.o, with.o_b());
    const auto ob = fetch(bb.o, with.o_b());
    std::string why = (ea != RAD_OK || eb != RAD_OK) ? fmt("rc=%d/%d", ea, eb)
                    : oa == ob ? "the conv bias operand did not change the output" : "";
    ok("control: the conv bias operand reaches the convolution", why.empty(), why);
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

// SAFE padding. One sequence is padded exactly the way the engine pads a short row: a negative
// conv slot and all three anchor columns negative, while its three factor columns stay real
// allocated slots -- which is where the original marker read lands, so that read stays inside the
// allocation. Launching a negative conv slot with LIVE anchors is a different case and is never
// launched here: the fused kernel would follow its anchor path with a slot that does not exist.
// The padded sequence must leave every output row, every state column and every unreferenced conv
// slot untouched, its active neighbours must still advance, and the candidate must agree byte for
// byte with the frozen row on the skip.
void check_padding(const Plugin& frozen, const Plugin& cand, Stream& stream) {
    Geo g({3, 3, 3});
    g.cidx = {1, 0, -1};
    g.nacc = {1, 1, 1};
    const Geo capacity = g;
    Inputs in;
    build_inputs(g, 0x51ed270bu, in);
    const int64_t pad = g.N - 1;
    for (int64_t c = 0; c < 3; ++c) in.hsidx[(size_t)(pad * g.COLS + c)] = -1;
    for (int64_t c = 3; c < g.COLS; ++c)
        in.hsidx[(size_t)(pad * g.COLS + c)] = (int32_t)(pad * g.COLS + c);

    Buffers bf, bc, bd;
    if (!bf.alloc(capacity) || !bc.alloc(capacity) || !bd.alloc(capacity)) {
        ok("padded sequence (cidx=-1, anchor columns -1)", false, "allocation");
        bf.free_all(); bc.free_all(); bd.free_all(); return;
    }
    for (Buffers* b : { &bf, &bc, &bd }) {
        upload_inputs(g, in, *b);
        poison_outputs(g, *b);
    }
    if (g_poison) { poison_counters(g, bc); poison_counters(g, bd); }
    const std::vector<uint8_t> st0 = fetch(bf.st, capacity.st_b());
    const std::vector<uint8_t> cs0 = fetch(bf.cs, capacity.cs_b());
    const RadKernelInfo* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const RadKernelInfo* rc = find_row(cand, "gdn_conv_recurrent_update_k128_v128_bf16_qreload");
    const int ef = run_row(rf, g, bf, stream);
    const int ec = run_row(rc, g, bc, stream);
    const int ed = run_row(rc, g, bd, stream, true, 0, &cand);
    std::string why;
    if (ef != RAD_OK || ec != RAD_OK || ed != RAD_OK)
        why = fmt("frozen=%d candidate=%d described=%d", ef, ec, ed);
    stream.sync();

    const size_t slot = (size_t)(g.H * g.V * g.K) * 4;
    const size_t pad_base = (size_t)(pad * g.COLS) * slot;
    const size_t cs_slot = (size_t)(g.conv_dim * g.SL) * 2;
    const int64_t t0 = g.cu[(size_t)pad], t1 = g.cu[(size_t)pad + 1];
    for (const auto* arm : { &bf, &bc, &bd }) {
        if (why.empty() && !poison_intact(g, *arm, t0, t1))
            why = "the padded sequence wrote an output row";
        if (why.empty() && !rows_written(g, *arm, 0, t0))
            why = "an active neighbour left its output rows unwritten";
        if (why.empty()) {
            const auto st = fetch(arm->st, capacity.st_b());
            if (std::memcmp(st.data() + pad_base, st0.data() + pad_base,
                            (size_t)g.COLS * slot) != 0)
                why = "the padded sequence moved a state column";
            const auto cs = fetch(arm->cs, capacity.cs_b());
            if (std::memcmp(cs.data() + 2 * cs_slot, cs0.data() + 2 * cs_slot, cs_slot) != 0)
                why = "a padded sequence wrote an unreferenced conv slot";
            for (int64_t n = 0; n < g.N && why.empty(); ++n) {
                if (g.cidx[(size_t)n] < 0) continue;
                if (std::memcmp(st.data() + (size_t)(n * g.COLS) * slot,
                                st0.data() + (size_t)(n * g.COLS) * slot,
                                (size_t)g.COLS * slot) == 0)
                    why = fmt("active sequence %lld did not advance", (long long)n);
            }
        }
    }
    if (why.empty()) {
        const struct { const char* name; size_t bytes; void* Buffers::*field; } planes[] = {
            { "o", g.o_b(), &Buffers::o }, { "o_q", g.oq_b(), &Buffers::oq },
            { "o_scale", g.os_b(), &Buffers::os },
            { "state/anchor/factors", capacity.st_b(), &Buffers::st },
            { "conv_state", capacity.cs_b(), &Buffers::cs }
        };
        for (const auto& p : planes) {
            const auto f = fetch(bf.*(p.field), p.bytes);
            if (!all_equal(p.name, f, fetch(bc.*(p.field), p.bytes), "padded", &why) ||
                !all_equal(p.name, f, fetch(bd.*(p.field), p.bytes), "padded described", &why))
                break;
        }
    }
    for (const auto* arm : { &bf, &bc, &bd }) {
        std::string bad;
        if (why.empty() && !counters_clean(g, *arm, &bad))
            why = fmt("padded: %s", bad.c_str());
    }
    ok("padded sequence (cidx=-1, anchor columns -1) leaves outputs and state untouched; neighbours advance",
       why.empty(), why);
    bf.free_all(); bc.free_all(); bd.free_all();
}

// The parameters-only declaration the runtime makes before any tensor address exists. A row that
// answers with a negative or a too-small bound cannot be given a reusable scratch payload, so the
// clear-prototype check is only meaningful against an honest covering answer.
void check_cold_scratch(const RadKernelInfo* row) {
    std::array<RadParam, 4> ps = {{
        PI("head_k", 128), PI("head_v", 128), PI("conv_width", 4), PI("q_len", 16)
    }};
    RadArgs a{}; std::memset(&a, 0, sizeof a);
    a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
    const int64_t bytes = row->scratch ? row->scratch(&a) : -1;
    ok("cold declare scratch (parameters only) returns a covering bound",
       bytes >= 0 && bytes >= 8 * 8 * (int64_t)sizeof(unsigned int),
       fmt("scratch=%lld", (long long)bytes));
}

double measure(const RadKernelInfo* row, const Geo& g, Buffers& b, Stream& stream,
               Events& events, int iters, const Plugin* described = nullptr,
               bool poison = false) {
    // Outside the event interval and before any pending launch. Never between iterations: the
    // batch must measure the row's own ordered behavior, not a host-side refill each time.
    if (poison) poison_counters(g, b);
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

// The served width, measured. Every arm carries its own independent state through the seed and
// every later launch: the seed runs at T=max(3,tokens) with one accepted factor, the measured step
// runs at T=tokens and accepts that whole width, which is valid because the previous step wrote
// exactly that many factor rows. No arm is reset to a completed snapshot, so the state a timed
// launch reads is the state the previous launch left.
void paired_timing(const Plugin& frozen, const Plugin& candidate, Stream& stream,
                   int warmup, int iters, int repeats, int sequences, int tokens) {
    const int seed_t = std::max(3, tokens);
    Geo seed(std::vector<int32_t>((size_t)sequences, seed_t));
    seed.nacc.assign((size_t)sequences, 1);
    Geo g(std::vector<int32_t>((size_t)sequences, tokens));
    g.nacc.assign((size_t)sequences, tokens);
    const Geo capacity(std::vector<int32_t>((size_t)sequences, std::max(seed_t, tokens)));
    Inputs initial, next;
    build_inputs(seed, 0x51ed270bu, initial);
    build_inputs(g, 0x51ed270cu, next);
    Buffers bf, bc, bd;
    if (!bf.alloc(capacity) || !bc.alloc(capacity) || !bd.alloc(capacity))
        throw std::runtime_error("timing allocation");
    const auto* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const auto* rc = find_row(candidate, "gdn_conv_recurrent_update_k128_v128_bf16_qreload");
    for (Buffers* b : { &bf, &bc, &bd }) {
        upload_inputs(seed, initial, *b);
        poison_outputs(capacity, *b);
    }
    if (g_poison) { poison_counters(seed, bc); poison_counters(seed, bd); }
    if (run_row(rf, seed, bf, stream) != RAD_OK ||
        run_row(rc, seed, bc, stream) != RAD_OK ||
        run_row(rc, seed, bd, stream, true, 0, &candidate) != RAD_OK)
        throw std::runtime_error("timing seed");
    stream.sync();
    if (!real_tokens_written(seed, bf) || !real_tokens_written(seed, bc) ||
        !real_tokens_written(seed, bd)) throw std::runtime_error("timing seed output");
    auto compare = [&](const char* phase) {
        std::string why;
        // The planes are compared at their ALLOCATED extent: the seed wrote the wider rows, and a
        // measured step that scribbled into them would otherwise sit outside every comparison.
        const struct { const char* name; size_t bytes; void* Buffers::*field; } planes[] = {
            { "o", capacity.o_b(), &Buffers::o }, { "o_q", capacity.oq_b(), &Buffers::oq },
            { "o_scale", capacity.os_b(), &Buffers::os },
            { "state/anchor/factors", capacity.st_b(), &Buffers::st },
            { "conv_state", capacity.cs_b(), &Buffers::cs }
        };
        for (const auto& p : planes) {
            const auto f = fetch(bf.*(p.field), p.bytes);
            if (!all_equal(p.name, f, fetch(bc.*(p.field), p.bytes), phase, &why) ||
                !all_equal(p.name, f, fetch(bd.*(p.field), p.bytes), phase, &why)) break;
        }
        for (const Buffers* b : { &bf, &bc, &bd }) {
            std::string bad;
            if (why.empty() && !counters_clean(capacity, *b, &bad))
                why = fmt("%s: %s", phase, bad.c_str());
        }
        if (!why.empty()) throw std::runtime_error(why);
    };
    // The measured step, uploaded once. Its acceptance count is the width the seed itself wrote,
    // and every later step repeats the same width, so the bound never moves between launches.
    for (Buffers* b : { &bf, &bc, &bd }) {
        upload_step(g, next, *b);
        poison_outputs(g, *b);
    }
    for (int i = 0; i < warmup; ++i) {
        // Reused scratch before every numeric step, outside any pending launch. Not between the
        // timed iterations: the batch below measures the row's own ordered behavior.
        if (g_poison) { poison_counters(capacity, bc); poison_counters(capacity, bd); }
        if (run_row(rf, g, bf, stream) != RAD_OK ||
            run_row(rc, g, bc, stream) != RAD_OK ||
            run_row(rc, g, bd, stream, true, 0, &candidate) != RAD_OK)
            throw std::runtime_error("warmup");
        stream.sync();
        compare("warmup");
    }
    if (!real_tokens_written(g, bf) || !real_tokens_written(g, bc) ||
        !real_tokens_written(g, bd)) throw std::runtime_error("timing warmup output");
    ok(fmt("T=%d timing history agrees before measurement", tokens).c_str(), true);
    Events events;
    std::vector<double> baseline, probe, described;
    for (int r = 0; r < repeats; ++r) {
        if (r & 1) {
            described.push_back(measure(rc, g, bd, stream, events, iters, &candidate, g_poison));
            probe.push_back(measure(rc, g, bc, stream, events, iters, nullptr, g_poison));
            baseline.push_back(measure(rf, g, bf, stream, events, iters));
        } else {
            baseline.push_back(measure(rf, g, bf, stream, events, iters));
            probe.push_back(measure(rc, g, bc, stream, events, iters, nullptr, g_poison));
            described.push_back(measure(rc, g, bd, stream, events, iters, &candidate, g_poison));
        }
        compare("after timed batch");
    }
    ok("timed batches preserve output, state, counters and guards", true);
    if (!g_fail) {
        std::printf("native AQL event submission, seeded T=%d measured T=%d N=%lld H=%lld Hg=%lld "
                    "bias=%d poison=%d iterations=%d repeats=%d (not kernel-only)\n",
                    seed_t, tokens, (long long)g.N, (long long)g.H, (long long)g.Hg,
                    g_bias ? 1 : 0, g_poison ? 1 : 0, iters, repeats);
        spreads("frozen", baseline); spreads("candidate", probe); spreads("described", described);
    }
    bf.free_all(); bc.free_all(); bd.free_all();
}

// One described call on a real fixture, to learn whether this row is the one- or two-dispatch
// generation before the run decides which checks apply.
int detect_dispatches(const RadKernelInfo* row, const Plugin& origin, Stream& stream) {
    Geo g;
    Inputs in;
    build_inputs(g, 0x51ed270bu, in);
    Buffers b;
    if (!b.alloc(g)) { b.free_all(); return -1; }
    upload_inputs(g, in, b);
    poison_outputs(g, b);
    const int e = run_row(row, g, b, stream, true, 0, &origin);
    stream.sync();
    b.free_all();
    return e == RAD_OK ? g_dispatches : -1;
}

}  // namespace

int main(int argc, char** argv) {
    const char* frozen_path = "build_hip/radiance_home/kernels/libr4d.so";
    const char* cand_path = "/tmp/gdn-query-reload.so";
    int device = 0, steps = 5, warmup = 10, iters = 100, repeats = 7;
    int sequences = 1, tokens = 1;
    bool timing = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--timing")) { timing = true; continue; }
        if (!std::strcmp(argv[i], "--check-only")) { timing = false; continue; }
        if (!std::strcmp(argv[i], "--conv-bias")) { g_bias = true; continue; }
        if (!std::strcmp(argv[i], "--poison-counters")) { g_poison = true; continue; }
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
        else if (!std::strcmp(argv[i - 1], "--tokens")) tokens = (int)value;
        else return 2;
    }
    if (steps < 5 || warmup < 1 || iters < 1 || repeats < 3 || repeats > 101 ||
        sequences < 1 || sequences > 8 || tokens < 1 || tokens > 16) return 2;
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
        const int kind = detect_dispatches(rc, cand, stream);
        ok("candidate describes its real dispatches at capacity 0/1/2",
           kind == 1 || kind == 2, fmt("dispatches=%d", kind));
        if (g_poison && kind == 2) check_cold_scratch(rc);
        if (g_poison && kind != 2)
            ok("--poison-counters requires the two-dispatch clear prototype", false,
               fmt("dispatches=%d", kind));
        if (!g_fail) {
            check_refusals(cand, stream);
            if (!g_fail) check_moved(frozen, stream);
            if (!g_fail) check_sensitive(frozen, stream);
            if (!g_fail && g_bias) check_bias(frozen, stream);
            if (!g_fail) run_case(frozen, cand, stream, steps);
            if (!g_fail) check_padding(frozen, cand, stream);
            // Timing is a measurement of a run whose correctness checks passed, so it takes its
            // own guard rather than inheriting the one from before them.
            if (!g_fail && timing)
                paired_timing(frozen, cand, stream, warmup, iters, repeats, sequences, tokens);
        }
        std::printf("fixture: sequences N=%d tokens T=%d conv_bias=%s poison=%s dispatches=%d\n",
                    sequences, tokens, g_bias ? "on" : "off", g_poison ? "on" : "off", kind);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%d ok, %d failure(s)\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
