// gdn_query_reload_compare -- frozen libr4d's folded GDN decode row against
// candidates/gdn-query-reload, on radiance's own ABI 26 launch path.
//
// Both libraries are dlopen'd, their rows are found by name, and each `launch(const RadArgs*,
// RadStream)` is called directly, so the comparison is of the two real dispatches and not of a
// description of them: ABI 26 has no RadDescribeFn.
//
// WHAT IS COMPARED, byte for byte and at the allocation's full extent:
//   o (bf16), o_q (i8), o_scale (f32), the recurrent state cache (f32), the conv state cache (bf16)
// over a multi-step trajectory in which the acceptance count of each step is what the previous
// step actually wrote. Both arms keep their own state from a shared seed; neither is reset to a
// snapshot, so a step reads what the step before it left.
//
// CONTROLS, because an equality over two untouched buffers proves nothing:
//   * each arm's caches must have MOVED, and a single flipped input bit must move the output;
//   * a comparison that is handed one changed byte must reject it;
//   * the candidate's scratch is poisoned with 0x5b (guards 0xa7) before every launch, and after
//     the launch the N*Hg counters must be zero and both guards untouched -- that is the emitted
//     clear actually running, not a promise about a fresh allocation.
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_device.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int  g_ok = 0, g_fail = 0;
bool g_poison = false;

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

// ------------------------------------------------------------------ host helpers
// Round-to-nearest-even. The inputs it builds are exactly representable where the kernel's own
// f2bf rounds differently, so this mirrors only what both agree on.
uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

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
    int64_t K = 128, V = 128, H = 24, Hg = 8, N = 3, W = 4, MQ = 4;
    int64_t SL, conv_dim, in_w, COLS = 6, SSLOTS, CSLOTS, Td = 0;
    std::vector<int32_t> cu, lens, nacc, cidx;
    explicit Geo(std::vector<int32_t> windows = { 4, 2, 3 })
        : N((int64_t)windows.size()), lens(std::move(windows)),
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
    size_t st_b() const { return (size_t)(SSLOTS * H * V * K) * 4; }
    size_t cs_b() const { return (size_t)(CSLOTS * conv_dim * SL) * 2; }
    size_t o_b()  const { return (size_t)(Td * H * V) * 2; }
    size_t oq_b() const { return (size_t)(Td * H * V); }
    size_t os_b() const { return (size_t)(Td * H) * 4; }
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
            in.hsidx[(size_t)(n * g.COLS + c)] = (int32_t)(n * g.COLS + c);
}

constexpr int kGuard = 16;                       // bytes of scratch guard on either side

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
        ctr_storage = dalloc((size_t)(g.N * g.Hg) * 4 + 2 * kGuard);
        if (ctr_storage) {
            ctr = (char*)ctr_storage + kGuard;
            CK(hipMemset(ctr_storage, 0xa7, kGuard), "counter front guard");
            CK(hipMemset((char*)ctr + (size_t)(g.N * g.Hg) * 4, 0xa7, kGuard), "counter rear guard");
        }
        o  = dalloc(g.o_b());
        oq = dalloc(g.oq_b());
        os = dalloc(g.os_b());
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

void poison_outputs(const Geo& g, Buffers& b) {
    CK(hipMemset(b.o,  0xA5, g.o_b()),  "o poison");
    CK(hipMemset(b.oq, 0x5A, g.oq_b()), "oq poison");
    CK(hipMemset(b.os, 0x3C, g.os_b()), "os poison");
}

void poison_counters(const Geo& g, Buffers& b) {
    CK(hipMemset(b.ctr, 0x5b, (size_t)(g.N * g.Hg) * 4), "counter poison");
}

// ------------------------------------------------------------------ one launch
int run_row(const RadKernelInfo* row, const Geo& g, Buffers& b, Stream& stream,
            int invalid = 0) {
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
    const RadTensor tcb = absent;                            // the bias path is a separate probe

    std::array<RadTensor, 18> ts = {{
        tx, tcw, tcb, tcs, tcidx, ta, tb, talog, tdtb, tst, tsidx, tnacc, tcu,
        tz, tnw, to, toq, tos
    }};
    if (invalid == 1) ts[9].dtype = RAD_I32;                  // state cache type refused
    if (invalid == 2) ts[0].stride[0] = (int64_t)INT32_MAX + 1;   // pitch does not fit int32
    if (invalid == 7) ts[11].data = nullptr;                  // num_accepted absent
    std::array<RadParam, 9> ps = {{
        PI("q_len", g.MQ), PI("head_k", g.K), PI("head_v", g.V), PI("conv_width", g.W),
        PI("n_head_v", g.H), PF("scale", 1.0 / 16.0), PF("eps", 1e-6), PS("act", "sigmoid"),
        PS("state_form", "anchor")
    }};
    if (invalid == 5) ps[0] = PI("q_len", 17);
    if (invalid == 6) ps[8] = PS("state_form", "per_candidate");
    RadArgs a; std::memset(&a, 0, sizeof a);
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int64_t need = (int64_t)(g.N * g.Hg) * 4;
    a.scratch = b.ctr;
    a.scratch_bytes = need - (invalid == 4);
    a.world_size = 1;
    if (!invalid && row->scratch && row->scratch(&a) < need) return RAD_E_SCRATCH;
    return row->launch(&a, stream.value);
}

// ------------------------------------------------------------------ comparisons
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

bool counters_clean(const Geo& g, const Buffers& b, std::string* why) {
    const auto ctr = fetch(b.ctr, (size_t)(g.N * g.Hg) * 4);
    const auto guard = fetch(b.ctr_storage, ctr.size() + 2 * kGuard);
    if (!std::all_of(ctr.begin(), ctr.end(), [](uint8_t x){ return x == 0; })) {
        *why = "an arrival counter is not zero after the launch";
        return false;
    }
    if (!std::all_of(guard.begin(), guard.begin() + kGuard, [](uint8_t x){ return x == 0xa7; }) ||
        !std::all_of(guard.end() - kGuard, guard.end(), [](uint8_t x){ return x == 0xa7; })) {
        *why = "the clear wrote past its N*Hg counters";
        return false;
    }
    return true;
}

struct Trajectory {
    std::array<Geo, 4> schedule;
    Trajectory()
        : schedule{ Geo({3, 3, 3}), Geo({1, 3, 1}), Geo({3, 1, 3}), Geo({4, 4, 2}) } {
        // Each step accepts what the PREVIOUS step wrote, so the factor read is a real replay.
        schedule[0].nacc = {1, 1, 1};
        schedule[1].nacc = {3, 3, 3};
        schedule[2].nacc = {1, 3, 1};
        schedule[3].nacc = {3, 1, 3};
    }
};

// CONTROL 1: both caches must move, or every later equality compares two untouched buffers.
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
    const auto st = fetch(b.st, g.st_b());
    const auto cs = fetch(b.cs, g.cs_b());
    const bool st_moved = std::memcmp(st.data(), in.hst.data(), g.st_b()) != 0;
    const bool cs_moved = std::memcmp(cs.data(), in.hcs.data(), g.cs_b()) != 0;
    std::string why = e != RAD_OK ? fmt("rc=%d", e)
                    : !st_moved    ? "the recurrent state never moved"
                    : !cs_moved    ? "the conv state never moved" : "";
    ok("control: the frozen row advances BOTH state caches", why.empty(), why);
    b.free_all();
}

// CONTROL 2: one flipped input bit must move the frozen row's output.
void check_sensitive(const Plugin& frozen, Stream& stream) {
    const Geo g;
    Inputs a, b;
    build_inputs(g, 0x51ed270bu, a);
    build_inputs(g, 0x51ed270bu, b);
    b.hx[0] = (uint16_t)(b.hx[0] ^ 0x8000u);   // a sign flip cannot round back
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
    const auto oa = fetch(ba.o, g.o_b());
    const auto ob = fetch(bb.o, g.o_b());
    std::string why = (ea != RAD_OK || eb != RAD_OK) ? fmt("rc=%d/%d", ea, eb)
                    : oa == ob ? "a changed q byte did not change the output" : "";
    ok("control: a one-bit input change moves the frozen row's output", why.empty(), why);
    auto changed = oa;
    changed[0] ^= 1;
    CK(hipMemcpy(bb.o, changed.data(), changed.size(), hipMemcpyHostToDevice), "negative control");
    std::string rejected;
    ok("one consumer byte differs and the comparison rejects it",
       !all_equal("o", oa, fetch(bb.o, g.o_b()), "negative control", &rejected), rejected);
    ba.free_all(); bb.free_all();
}

void check_refusals(const Plugin& cand, Stream& stream) {
    const Geo g;
    Inputs in;
    build_inputs(g, 0x51ed270bu, in);
    Buffers b;
    if (!b.alloc(g)) { ok("refusal fixture", false, "allocation"); b.free_all(); return; }
    upload_inputs(g, in, b);
    const auto* row = find_row(cand, "gdn_query_reload_conv_recurrent_k128_v128_bf16");
    struct Case { int id; int expected; const char* what; };
    const Case cases[] = {
        { 1, RAD_E_DTYPE,       "an integer state cache is refused" },
        { 2, RAD_E_STRIDE,      "an x pitch past int32 is refused" },
        { 4, RAD_E_SCRATCH,     "scratch one byte short of N*Hg*4 is refused" },
        { 5, RAD_E_SHAPE,       "q_len 17 is refused" },
        { 6, RAD_E_UNSUPPORTED, "state_form per_candidate is refused" },
        { 7, RAD_E_INVAL,       "an absent num_accepted is refused" },
    };
    for (const auto& c : cases) {
        const int rc = run_row(row, g, b, stream, c.id);
        ok(c.what, rc == c.expected, fmt("rc=%d expected=%d", rc, c.expected));
    }
    stream.sync();
    b.free_all();
}

// The trajectory proper. Both arms run on their own buffers from one seed.
void run_case(const Plugin& frozen, const Plugin& cand, Stream& stream, int steps) {
    Trajectory traj;
    const Geo capacity({4, 4, 4});
    Inputs initial;
    build_inputs(traj.schedule[0], 0x51ed270bu, initial);
    Buffers bf, bc;
    if (!bf.alloc(capacity) || !bc.alloc(capacity)) {
        ok("frozen vs candidate trajectory", false, "allocation");
        bf.free_all(); bc.free_all(); return;
    }
    const RadKernelInfo* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
    const RadKernelInfo* rc = find_row(cand, "gdn_query_reload_conv_recurrent_k128_v128_bf16");
    upload_inputs(traj.schedule[0], initial, bf);
    upload_inputs(traj.schedule[0], initial, bc);
    std::string why;
    for (int step = 0; step < steps && why.empty(); ++step) {
        const Geo& g = step < (int)traj.schedule.size() ? traj.schedule[(size_t)step]
                                                        : traj.schedule.back();
        if (step) {
            Inputs next;
            build_inputs(g, 0x51ed270bu + (uint32_t)step, next);
            upload_step(g, next, bf);
            upload_step(g, next, bc);
        }
        poison_outputs(g, bf); poison_outputs(g, bc);
        if (g_poison) poison_counters(g, bc);
        const int ef = run_row(rf, g, bf, stream);
        const int ec = run_row(rc, g, bc, stream);
        if (ef != RAD_OK || ec != RAD_OK) {
            why = fmt("step %d: frozen=%d candidate=%d", step, ef, ec);
            break;
        }
        stream.sync();
        if (g_poison && why.empty()) {
            std::string bad;
            if (!counters_clean(g, bc, &bad)) why = fmt("step %d: %s", step, bad.c_str());
        }
        const struct { const char* name; size_t bytes; void* Buffers::*field; } planes[] = {
            { "o", g.o_b(), &Buffers::o }, { "o_q", g.oq_b(), &Buffers::oq },
            { "o_scale", g.os_b(), &Buffers::os },
            { "state/anchor/factors", capacity.st_b(), &Buffers::st },
            { "conv_state", capacity.cs_b(), &Buffers::cs }
        };
        const std::string s = fmt("step %d", step);
        for (const auto& p : planes) {
            if (why.empty() &&
                !all_equal(p.name, fetch(bf.*(p.field), p.bytes),
                           fetch(bc.*(p.field), p.bytes), s.c_str(), &why))
                break;
        }
    }
    ok("frozen vs candidate: o, o_q, o_scale, state and conv_state are byte-identical across the "
       "whole trajectory", why.empty(), why);
    bf.free_all(); bc.free_all();
}

// The scratch hook is asked with parameters only, before any tensor address exists.
void check_cold_scratch(const RadKernelInfo* row) {
    std::array<RadParam, 4> ps = {{
        PI("head_k", 128), PI("head_v", 128), PI("conv_width", 4), PI("q_len", 16)
    }};
    RadArgs a{}; std::memset(&a, 0, sizeof a);
    a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
    const int64_t bytes = row->scratch ? row->scratch(&a) : -1;
    ok("cold declare scratch (parameters only) returns a covering 16 MiB-ceiling bound",
       bytes == 4096 * (int64_t)sizeof(unsigned int), fmt("scratch=%lld", (long long)bytes));
}

}  // namespace

int main(int argc, char** argv) {
    const char* frozen_path = nullptr;
    const char* cand_path = nullptr;
    int device = 0, steps = 6;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--poison-counters")) { g_poison = true; continue; }
        if (i + 1 == argc) return 2;
        if (!std::strcmp(argv[i], "--frozen")) { frozen_path = argv[++i]; continue; }
        if (!std::strcmp(argv[i], "--candidate")) { cand_path = argv[++i]; continue; }
        char* end = nullptr;
        const long v = std::strtol(argv[++i], &end, 10);
        if (!end || *end || v < 0 || v > 1000000) return 2;
        if (!std::strcmp(argv[i - 1], "--device")) device = (int)v;
        else if (!std::strcmp(argv[i - 1], "--steps") && v >= 1) steps = (int)v;
        else return 2;
    }
    if (!frozen_path || !cand_path) {
        std::fprintf(stderr, "usage: %s --frozen libr4d.so --candidate gdn_query_reload.so "
                             "[--device N] [--steps N] [--poison-counters]\n", argv[0]);
        return 2;
    }
    try {
        CK(hipSetDevice(device), "hipSetDevice");
        if (rad_dev_set(device) != RAD_OK) throw std::runtime_error("native AQL device");
        RadDeviceProps props{};
        if (rad_dev_props(device, &props) != RAD_OK || props.is_host_backend)
            throw std::runtime_error("native GPU backend");
        Stream stream;
        Plugin frozen, cand;
        if (!load(frozen, frozen_path) || !load(cand, cand_path)) return 1;
        const auto* rf = find_row(frozen, "gdn_conv_recurrent_update_k128_v128_bf16");
        const auto* rc = find_row(cand, "gdn_query_reload_conv_recurrent_k128_v128_bf16");
        ok("both rows are found, from different DSOs, on the same op",
           rf && rc && rf->launch != rc->launch && rf->op && rc->op &&
           !std::strcmp(rf->op, "gdn_conv_recurrent_update") &&
           !std::strcmp(rc->op, "gdn_conv_recurrent_update"));
        ok("the candidate declares a scratch hook and no schema",
           rc && rc->scratch && cand.kernel_count() == 1);
        if (g_fail) return 1;
        check_cold_scratch(rc);
        check_refusals(cand, stream);
        if (!g_fail) check_moved(frozen, stream);
        if (!g_fail) check_sensitive(frozen, stream);
        if (!g_fail) run_case(frozen, cand, stream, steps);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%d ok, %d failure(s)\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
