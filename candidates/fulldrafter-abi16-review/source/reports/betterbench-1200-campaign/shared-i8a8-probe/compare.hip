// Standalone frozen-row versus private-row probe; parent builds and runs this, not the worker.
#include <hip/hip_runtime.h>
#include "rad_abi.h"
#include "rad_device.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* Frozen = "build_hip/radiance_home/kernels/libr4d.so";
constexpr const char* Candidate = "/tmp/shared-i8a8-probe.so";
void check(bool good, const std::string& msg) { if (!good) throw std::runtime_error(msg); }
#define HIP(call) do { const hipError_t e = (call); if (e != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e)); } while (0)
float bf(uint16_t v) { const uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t to_bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16); }
RadTensor tensor(void* p, uint32_t dtype, int64_t d0, int64_t d1) {
    RadTensor t{}; t.data = p; t.dtype = dtype; t.rank = 2;
    t.shape[0] = d0; t.shape[1] = d1; rad_tensor_pack(&t); return t;
}
RadParam integer(const char* key, long long v) { RadParam p{}; p.key = key; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam string(const char* key, const char* v) { RadParam p{}; p.key = key; p.kind = RAD_P_STR; p.sval = v; return p; }

struct Buffer {
    static constexpr size_t Guard = 64;
    size_t size;
    uint8_t* raw = nullptr;
    explicit Buffer(size_t n) : size(n) { HIP(hipMalloc((void**)&raw, size + 2 * Guard)); reset(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (raw) { const auto e = hipFree(raw); if (e != hipSuccess) std::fprintf(stderr, "hipFree: %s\n", hipGetErrorString(e)); } }
    void* ptr() const { return raw + Guard; }
    void reset() { HIP(hipMemset(raw, 0xa5, size + 2 * Guard)); HIP(hipMemset(ptr(), 0xcd, size)); }
    void put(const void* p) { HIP(hipMemcpy(ptr(), p, size, hipMemcpyHostToDevice)); }
    std::vector<uint8_t> get() const {
        std::vector<uint8_t> all(size + 2 * Guard);
        HIP(hipMemcpy(all.data(), raw, all.size(), hipMemcpyDeviceToHost));
        check(std::all_of(all.begin(), all.begin() + Guard, [](uint8_t x) { return x == 0xa5; }) &&
              std::all_of(all.end() - Guard, all.end(), [](uint8_t x) { return x == 0xa5; }), "buffer guard modified");
        return {all.begin() + Guard, all.end() - Guard};
    }
};
struct Plugin {
    void* h = nullptr;
    const RadKernelInfo* row = nullptr;
    explicit Plugin(const char* path, const char* name) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error(dlerror());
        auto sym = [&](const char* s) { dlerror(); void* p = dlsym(h, s); const char* e = dlerror();
            check(!e && p, std::string("symbol ") + s + ": " + (e ? e : "null")); return p; };
        auto abi = reinterpret_cast<uint32_t (*)()>(sym("rad_plugin_abi_version"));
        auto count = reinterpret_cast<int (*)()>(sym("rad_kernel_count"));
        auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(sym("rad_kernel_at"));
        check(abi() == RAD_ABI_VERSION, "plugin ABI mismatch");
        int matches = 0;
        for (int i = 0; i < count(); ++i) if (const auto* p = at(i))
            if (p->name && !std::strcmp(p->name, name)) { row = p; ++matches; }
        check(matches == 1 && row->launch && row->describe && row->op &&
              !std::strcmp(row->op, "gemm_nt_q") && row->domain == RAD_DOMAIN_DEVICE,
              "requested actual GEMM device row missing or duplicated");
        Dl_info info{};
        check(dladdr((const void*)row->launch, &info) && info.dli_fname, "unknown launch owner");
        char expected[4096], actual[4096];
        check(realpath(path, expected) && realpath(info.dli_fname, actual) &&
              !std::strcmp(expected, actual), "launch does not belong to requested DSO");
        std::printf("row=%s origin=%s abi=%u\n", name, actual, abi());
    }
    ~Plugin() { if (h && dlclose(h)) std::fprintf(stderr, "dlclose: %s\n", dlerror()); }
};

// Use the frozen row's *actual* layout and relayout, not a transcription of frag_src.
std::vector<uint8_t> stored(const RadKernelInfo* row, const std::vector<RadParam>& params,
                             int operand, const RadEncoding& enc, void* plane,
                             uint32_t dtype, int N, int K) {
    check(row->layout && row->relayout && row->unrelayout, "frozen relayout hooks absent");
    const int selected = operand == 2 ? 0 : 1;
    RadTensor t = tensor(plane, dtype, N, operand == 2 ? K : K / 128);
    RadLayout lay{};
    int rc = row->layout(params.data(), (int)params.size(), operand, &enc, &selected, &t, 1, &lay);
    check(rc == RAD_OK && lay.bytes == (operand == 2 ? (int64_t)N*K : (int64_t)N*(K/128)*2),
          "unexpected frozen stored-plane layout");
    std::vector<uint8_t> packed((size_t)lay.bytes);
    rc = row->relayout(params.data(), (int)params.size(), operand, &enc, &selected, &t, 1,
                       packed.data(), lay.bytes);
    check(rc == RAD_OK, "frozen relayout refused plane");
    std::vector<uint8_t> back((size_t)lay.bytes, 0xcd);
    t.data = back.data();
    rc = row->unrelayout(params.data(), (int)params.size(), operand, &enc, &selected,
                         packed.data(), lay.bytes, &t, 1);
    check(rc == RAD_OK && !std::memcmp(back.data(), plane, back.size()), "frozen layout roundtrip failed");
    return packed;
}
struct Call {
    std::vector<RadTensor> ts;
    std::vector<RadParam> ps;
    RadArgs args{};
    Call(int M, int N, int K, const Buffer& a, const Buffer& as, const Buffer& w,
         const Buffer& ws, const Buffer& y) {
        ts = {tensor(a.ptr(), RAD_I8, M, K), tensor(as.ptr(), RAD_F32, M, K / 128),
              tensor(w.ptr(), RAD_I8, N, K), tensor(ws.ptr(), RAD_BF16, N, K / 128),
              tensor(y.ptr(), RAD_BF16, M, N)};
        ps = {integer("M", M), integer("N", N), integer("K", K), integer("group", 128), string("dtype", "i8a8")};
        args.t = ts.data(); args.n_t = (int)ts.size(); args.p = ps.data(); args.n_p = (int)ps.size();
        args.world_size = 2;
    }
};
void launch(const Plugin& p, Call& c, RadStream stream) {
    const int rc = p.row->launch(&c.args, stream);
    check(rc == RAD_OK, std::string(p.row->name) + " launch refused: " + std::to_string(rc));
    HIP(hipGetLastError());
}
void refusal(const Plugin& p, Call& c, RadStream s) {
    auto fail = [&](int expected) {
        RadLaunchDesc d{}; int count = -1;
        check(p.row->launch(&c.args, s) == expected &&
              p.row->describe(&c.args, s, &d, 1, &count) == expected && count == 0,
              "launch/describe refusal mismatch");
    };
    auto m = c.ps[0]; c.ps[0] = integer("M", 8); fail(RAD_E_SHAPE); c.ps[0] = m;
    auto n = c.ps[1]; c.ps[1] = integer("N", 512); fail(RAD_E_SHAPE); c.ps[1] = n;
    auto k = c.ps[2]; c.ps[2] = integer("K", 640); fail(RAD_E_SHAPE); c.ps[2] = k;
    auto g = c.ps[3]; c.ps[3] = integer("group", 64); fail(RAD_E_SHAPE); c.ps[3] = g;
    auto dt = c.ts[2]; c.ts[2].dtype = RAD_U8; fail(RAD_E_DTYPE); c.ts[2] = dt;
    auto stride = c.ts[0]; c.ts[0].stride[0] += 8; fail(RAD_E_STRIDE); c.ts[0] = stride;
    auto scale = c.ts[3]; c.ts[3].shape[0] -= 16; fail(RAD_E_SHAPE); c.ts[3] = scale;
    auto ptr = c.ts[4]; c.ts[4].data = (uint8_t*)ptr.data + 1; fail(RAD_E_ALIGN); c.ts[4] = ptr;
    auto code = c.ps[4]; c.ps[4] = string("dtype", "fp8a8"); fail(RAD_E_SHAPE); c.ps[4] = code;
}
struct Stream {
    RadStream s = nullptr;
    Stream() { check(rad_stream_create(&s, 0) == RAD_OK, "AQL stream create failed"); }
    ~Stream() { if (s) rad_stream_destroy(s); }
    void sync() { check(rad_stream_sync(s) == RAD_OK, "AQL stream sync failed"); }
};
struct Events {
    RadEvent start = nullptr, stop = nullptr;
    Events() { check(rad_event_create(&start) == RAD_OK && rad_event_create(&stop) == RAD_OK,
                     "AQL event create failed"); }
    ~Events() { if (start) rad_event_destroy(start); if (stop) rad_event_destroy(stop); }
};
template <class F> double measure(F&& f, RadStream s, Events& e, int iters) {
    float ms = 0;
    check(rad_event_record(e.start, s) == RAD_OK, "event start failed");
    for (int i = 0; i < iters; ++i) f();
    check(rad_event_record(e.stop, s) == RAD_OK && rad_event_sync(e.stop) == RAD_OK &&
          rad_event_elapsed_ms(e.start, e.stop, &ms) == RAD_OK && ms > 0 && std::isfinite(ms),
          "event timing failed");
    return ms * 1000.0 / iters;
}
void timings(const char* name, std::vector<double> samples) {
    std::printf("%s samples_us=", name);
    for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", samples[i]);
    std::sort(samples.begin(), samples.end());
    std::printf(" median_us=%.6f\n", samples[samples.size()/2]);
}

// Independent integer dot over *canonical* signed codes. Each 128-K sum fits int32:
// 128 * 128 * 128 = 2097152. Fold with the frozen source's FP32 scale order.
bool compare(const std::vector<uint8_t>& bytes, const std::vector<int8_t>& a,
             const std::vector<int8_t>& w, const std::vector<float>& as,
             const std::vector<uint16_t>& ws, int M, int N, int K,
             const char* label, bool verbose) {
    const int G = K / 128;
    check(bytes.size() == (size_t)M * N * 2, "output byte count changed");
    size_t bad = 0, bf_mismatch = 0, first = 0;
    double worst = 0, worst_abs = 0;
    for (int m = 0; m < M; ++m) for (int col = 0; col < N; ++col) {
        float expected = 0, sum_abs = 0;
        for (int g = 0; g < G; ++g) {
            int32_t dot = 0;
            for (int j = 0; j < 128; ++j)
                dot += (int)a[(size_t)m*K + g*128 + j] * (int)w[(size_t)col*K + g*128 + j];
            const float term = (float)dot * (as[m*G + g] * bf(ws[col*G + g]));
            expected += term; sum_abs += std::fabs(term);
        }
        const size_t idx = (size_t)m*N + col;
        uint16_t bits; std::memcpy(&bits, bytes.data() + 2*idx, 2);
        const float actual = bf(bits);
        const uint16_t nearest = to_bf(expected);
        const float ulp = std::ldexp(1.0f, std::ilogb(std::max(std::fabs(expected),
                                                                 std::numeric_limits<float>::min())) - 7);
        const double error = std::fabs((double)actual - expected);
        // FP32 group-order discrepancy plus one final BF16 rounding interval; absolute near zero.
        const double bound = 1e-6 + 1e-5 * (double)sum_abs + (double)ulp;
        if (!std::isfinite(actual) || error > bound ||
            (sum_abs == 0 && bits != 0)) { if (!bad) first = idx; ++bad; }
        bf_mismatch += bits != nearest;
        worst = std::max(worst, error / bound); worst_abs = std::max(worst_abs, error);
    }
    if (verbose) std::printf("%s elements=%zu bad=%zu first=%ld bf16_vs_ref=%zu worst_abs=%.9g worst_bound_ratio=%.9g\n",
                             label, bytes.size()/2, bad, bad ? (long)first : -1L,
                             bf_mismatch, worst_abs, worst);
    return bad == 0;
}
void pair_errors(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) {
    check(x.size() == y.size(), "pair output sizes differ");
    size_t count = 0, first = 0; double worst = 0;
    for (size_t i = 0; i < x.size()/2; ++i) {
        uint16_t a, b; std::memcpy(&a, x.data()+2*i, 2); std::memcpy(&b, y.data()+2*i, 2);
        if (a != b) { if (!count) first = i; ++count; }
        worst = std::max(worst, std::fabs((double)bf(a) - bf(b)));
    }
    std::printf("frozen_vs_candidate BF16 mismatches=%zu first=%ld max_abs=%.9g\n",
                count, count ? (long)first : -1L, worst);
}
void run_case(const Plugin& old, const Plugin& cand, Stream& s, Events& e,
              int M, int N, int K, int mode, int warmup, int iters, int repeats, bool check_only) {
    const int G = K / 128;
    uint64_t rng = (uint64_t)M*8171 + (uint64_t)N*71 + mode*11 + 1;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    std::vector<int8_t> a((size_t)M*K), w((size_t)N*K);
    std::vector<float> as((size_t)M*G);
    std::vector<uint16_t> ws((size_t)N*G);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = mode == 1 ? (int8_t)(i % 4 == 0 ? -128 : i % 4 == 1 ? 127 : -127) :
               mode == 2 ? 0 : (int8_t)((int)(next() % 255) - 127);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = mode == 1 ? (int8_t)(i % 3 == 0 ? -128 : i % 3 == 1 ? 127 : -127) :
               mode == 2 ? 0 : (int8_t)((int)(next() % 255) - 127);
    for (auto& v : as) v = mode == 2 ? 1.f : (float)((int)(next()%2001)-1000) / 837.0f;
    for (auto& v : ws) v = to_bf(mode == 2 ? 1.f : (float)((int)(next()%2001)-1000) / 3103.0f);
    const std::vector<RadParam> lp = {integer("N", N), integer("K", K), integer("group", 128)};
    const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    auto packed_w = stored(old.row, lp, 2, enc, w.data(), RAD_I8, N, K);
    auto packed_s = stored(old.row, lp, 3, enc, ws.data(), RAD_BF16, N, K);
    Buffer d_a(a.size()), d_as(as.size()*4), d_w(packed_w.size()), d_ws(packed_s.size());
    Buffer y0((size_t)M*N*2), y1((size_t)M*N*2);
    d_a.put(a.data()); d_as.put(as.data()); d_w.put(packed_w.data()); d_ws.put(packed_s.data());
    Call before(M,N,K,d_a,d_as,d_w,d_ws,y0), after(M,N,K,d_a,d_as,d_w,d_ws,y1);
    Buffer scratch((size_t)std::max<int64_t>(1, old.row->scratch ? old.row->scratch(&before.args) : 0));
    before.args.scratch = scratch.ptr(); before.args.scratch_bytes = scratch.size;
    auto ref = [&] { launch(old, before, s.s); }; auto test = [&] { launch(cand, after, s.s); };
    ref(); test(); s.sync();
    auto ref_out = y0.get(), got = y1.get();
    std::printf("case M=%d N=%d K=%d mode=%s\n", M,N,K, mode == 0 ? "random" : mode == 1 ? "signed-extremes" : "zeros");
    check(compare(ref_out, a, w, as, ws, M,N,K,"frozen",true), "frozen row differs from independent reference");
    check(compare(got, a, w, as, ws, M,N,K,"candidate",true), "candidate differs from independent reference");
    pair_errors(ref_out,got);
    auto mutated = got; uint16_t infinity = 0x7f80; std::memcpy(mutated.data(), &infinity, 2);
    check(!compare(mutated,a,w,as,ws,M,N,K,"negative_control",false), "negative control was not detected");
    refusal(cand, after, s.s);
    RadLaunchDesc desc{}; int count = -1;
    check(cand.row->describe(&after.args,s.s,&desc,0,&count) == RAD_E_FULL && count == 0 &&
          cand.row->describe(&after.args,s.s,&desc,1,&count) == RAD_OK && count == 1,
          "candidate description count/capacity differs");
    check(y1.get() == got, "describe modified output");
    y1.reset();
    HIP(hipLaunchKernel(desc.func, dim3(desc.grid[0],desc.grid[1],desc.grid[2]),
                        dim3(desc.block[0],desc.block[1],desc.block[2]),
                        desc.params,desc.shared_bytes,(hipStream_t)s.s));
    s.sync(); check(y1.get() == got, "described dispatch changed output");
    y0.reset(); y1.reset();
    ref(); test(); s.sync();
    check(y0.get() == ref_out && y1.get() == got, "repeat changed output");
    check(d_a.get() == std::vector<uint8_t>((uint8_t*)a.data(),(uint8_t*)a.data()+a.size()) &&
          d_as.get() == std::vector<uint8_t>((uint8_t*)as.data(),(uint8_t*)as.data()+as.size()*4) &&
          d_w.get() == packed_w && d_ws.get() == packed_s, "input or packed weight modified");
    scratch.get(); // Also verify the frozen split-K scratch guards.
    if (!check_only && mode == 0) {
        for (int i=0;i<warmup;++i) { ref(); test(); } s.sync();
        std::vector<double> old_us, new_us;
        for (int i=0;i<repeats;++i) {
            if (i & 1) { new_us.push_back(measure(test,s.s,e,iters)); old_us.push_back(measure(ref,s.s,e,iters)); }
            else { old_us.push_back(measure(ref,s.s,e,iters)); new_us.push_back(measure(test,s.s,e,iters)); }
        }
        check(y0.get() == ref_out && y1.get() == got, "timing changed outputs");
        timings("frozen_aql",old_us); timings("candidate_aql",new_us);
    }
    std::printf("independent_reference,negative_control,refusals,describe,repeat=PASS\n");
}
int run(int argc, char** argv) {
    int device = 0, warmup = 100, iters = 200, repeats = 9; bool check_only = false;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"--check-only")) { check_only = true; continue; }
        check(i+1<argc,"option missing value");
        const std::string name=argv[i++]; char* end=nullptr;
        const long v=std::strtol(argv[i],&end,10);
        check(end && *end=='\0' && v>=0 && v<=1000000,"invalid option");
        if (name=="--device") device=(int)v;
        else if (name=="--warmup") warmup=(int)v;
        else if (name=="--iters") iters=(int)v;
        else if (name=="--repeats") repeats=(int)v;
        else check(false,"unknown option: "+name);
    }
    check(check_only || (warmup>0 && iters>0 && repeats>=3 && repeats<=101),"invalid timing count");
    HIP(hipSetDevice(device)); check(rad_dev_set(device)==RAD_OK,"native device selection failed");
    hipDeviceProp_t props{}; HIP(hipGetDeviceProperties(&props,device));
    check(!std::strncmp(props.gcnArchName,"gfx1201",7) && props.warpSize==32,"gfx1201 wave32 required");
    Plugin old(Frozen,"gemm_i8a8_nt_m16");
    Plugin gate(Candidate,"shared_i8a8_dot4_gateup_probe");
    Plugin down(Candidate,"shared_i8a8_dot4_down_probe");
    Stream stream; Events events;
    int cases=0;
    for (auto shape : {0,1}) {
        const int N=shape ? 2560:768, K=shape ? 384:2560;
        for (int M=1;M<=4;++M) for (int mode=0;mode<3;++mode) {
            run_case(old,shape ? down : gate,stream,events,M,N,K,mode,warmup,iters,repeats,check_only); ++cases;
        }
    }
    std::printf("PASS cases=%d timing=%s standalone, not sealed BetterBench\n",cases,check_only?"off":"AQL events");
    return 0;
}
} // namespace
int main(int argc,char** argv) {
    try { return run(argc,argv); }
    catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
