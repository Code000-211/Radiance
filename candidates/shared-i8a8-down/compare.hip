/* shared_i8a8_down_compare.hip -- frozen libr4d row vs the candidate row, byte for byte.
 *
 * Not part of the candidate plugin. Build it beside a build tree, run it under the GPU lock:
 *   - both DSOs are dlopen'd RTLD_NOW|RTLD_LOCAL and their launches called through the ABI;
 *   - the weight planes are canonical affine i8/bf16 [N][K] and [N][K/128];
 *   - the packed bytes come from each row's own layout/relayout hook, and the two are compared,
 *     so the claim "same layout bytes" is a measurement rather than a reading of the source;
 *   - the outputs of both rows are compared byte for byte at M=1..3 (the row's band; M=4
 *     belongs to the frozen row and must be refused here).
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <unistd.h>

namespace {

constexpr int N = 2560, K = 384, G = 128;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        std::exit(1);
    }
}
#define HIP(call) do { const hipError_t e_ = (call); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "FAIL: %s: %s\n", #call, hipGetErrorString(e_)); std::exit(1); } } while (0)

float bf(uint16_t v) { const uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t to_bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

RadTensor tensor(void* p, uint32_t dtype, int64_t d0, int64_t d1) {
    RadTensor t{}; t.data = p; t.dtype = dtype; t.rank = 2;
    t.shape[0] = d0; t.shape[1] = d1; rad_tensor_pack(&t); return t;
}
RadParam integer(const char* key, long long v) { RadParam p{}; p.key = key; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam str(const char* key, const char* v) { RadParam p{}; p.key = key; p.kind = RAD_P_STR; p.sval = v; return p; }

struct Buffer {
    size_t size = 0;
    void* p = nullptr;
    explicit Buffer(size_t n) : size(n) { HIP(hipMalloc(&p, n ? n : 1)); }
    ~Buffer() { if (p) hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const void* src) { HIP(hipMemcpy(p, src, size, hipMemcpyHostToDevice)); }
    std::vector<uint8_t> get() const {
        std::vector<uint8_t> v(size);
        HIP(hipMemcpy(v.data(), p, size, hipMemcpyDeviceToHost));
        return v;
    }
};

struct Plugin {
    void* h = nullptr;
    const RadKernelInfo* row = nullptr;
    Plugin(const char* path, const char* name) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "FAIL: dlopen %s: %s\n", path, dlerror()); std::exit(1); }
        auto sym = [&](const char* s) { dlerror(); void* q = dlsym(h, s);
            const char* e = dlerror();
            check(!e && q, std::string("symbol ") + s + (e ? e : " missing"));
            return q; };
        auto abi   = (uint32_t (*)())sym("rad_plugin_abi_version");
        auto count = (int (*)())sym("rad_kernel_count");
        auto at    = (const RadKernelInfo* (*)(int))sym("rad_kernel_at");
        check(abi() == RAD_ABI_VERSION, "plugin ABI version differs from this build's");
        int hits = 0;
        for (int i = 0; i < count(); ++i)
            if (const RadKernelInfo* r = at(i); r && r->name && !std::strcmp(r->name, name)) {
                row = r; ++hits;
            }
        check(hits == 1 && row && row->launch && row->layout && row->relayout && row->unrelayout &&
              row->op && !std::strcmp(row->op, "gemm_nt_q") && row->domain == RAD_DOMAIN_DEVICE,
              std::string("row ") + name + " missing, duplicated or not the expected device GEMM");
    }
    ~Plugin() { if (h) dlclose(h); }
};

struct Params {
    std::vector<RadParam> v;
    Params() {
        v = { integer("M", 1), integer("N", N), integer("K", K), integer("group", 128),
              str("dtype", "i8a8") };
    }
    void set_m(int m) { v[0] = integer("M", m); }
};

/* The stored bytes one row wants for a weight plane, produced by that row's own hook. */
std::vector<uint8_t> stored(const RadKernelInfo* row, const std::vector<RadParam>& p, int operand,
                            const RadEncoding& enc, const int* sel, void* plane, uint32_t dtype,
                            int64_t d0, int64_t d1, std::string* tag) {
    RadTensor t = tensor(plane, dtype, d0, d1);
    RadLayout lay{};
    const int rc = row->layout(p.data(), (int)p.size(), operand, &enc, sel, &t, 1, &lay);
    check(rc == RAD_OK && lay.bytes > 0, std::string(row->name) + ": layout refused");
    if (tag) *tag = lay.tag ? lay.tag : "";
    std::vector<uint8_t> packed((size_t)lay.bytes, 0xcd);
    check(row->relayout(p.data(), (int)p.size(), operand, &enc, sel, &t, 1, packed.data(),
                        lay.bytes) == RAD_OK,
          std::string(row->name) + ": relayout refused");
    /* Round trip: the bytes must decode back to the canonical plane, or the permutation is not
     * an invertible restatement of the same numbers. */
    std::vector<uint8_t> back((size_t)lay.bytes, 0xcd);
    t.data = back.data();
    check(row->unrelayout(p.data(), (int)p.size(), operand, &enc, sel, packed.data(), lay.bytes,
                          &t, 1) == RAD_OK && !std::memcmp(back.data(), plane, back.size()),
          std::string(row->name) + ": layout round trip is not the identity");
    return packed;
}

struct Call {
    std::vector<RadTensor> t;
    RadArgs a{};
    Call(int m, const Buffer& ab, const Buffer& asb, const Buffer& wb, const Buffer& wsb,
         const Buffer& yb) {
        t = { tensor(ab.p, RAD_I8, m, K), tensor(asb.p, RAD_F32, m, K / G),
              tensor(wb.p, RAD_I8, N, K), tensor(wsb.p, RAD_BF16, N, K / G),
              tensor(yb.p, RAD_BF16, m, N) };
        a.t = t.data(); a.n_t = (int)t.size();
    }
};

/* The independent signed-int reference: int32 per 128-K group, FP32 fold, one BF16 rounding.
 * `expected`, when asked for, receives the reference's own BF16 answer per element, so the host
 * self-check can run a known-good output through the same verdict. */
size_t reference(const std::vector<uint8_t>& y, const std::vector<int8_t>& a,
                 const std::vector<float>& asc, const std::vector<int8_t>& w,
                 const std::vector<uint16_t>& wsc, int m, int n, int k,
                 std::vector<uint16_t>* expected = nullptr) {
    size_t bad = 0;
    if (expected) expected->assign((size_t)m * n, 0);
    for (int i = 0; i < m; ++i)
        for (int c = 0; c < n; ++c) {
            float acc = 0;
            for (int g = 0; g < k / G; ++g) {
                int32_t dot = 0;
                for (int j = 0; j < G; ++j)
                    dot += (int)a[(size_t)i * k + g * G + j] * (int)w[(size_t)c * k + g * G + j];
                acc += (float)dot * (asc[(size_t)i * (k / G) + g] * bf(wsc[(size_t)c * (k / G) + g]));
            }
            uint16_t got; std::memcpy(&got, y.data() + 2 * ((size_t)i * n + c), 2);
            if (expected) (*expected)[(size_t)i * n + c] = to_bf(acc);
            /* A non-finite answer is an error, not a large error: `fabs(nan - acc) > ulp` is
             * false, so a NaN output would otherwise pass every comparison below. */
            if (!std::isfinite(bf(got)) || !std::isfinite(acc)) { ++bad; continue; }
            /* One BF16 rounding interval, plus the FP32 group-fold discrepancy a reordered sum
             * may produce. Not a gate tolerance: this harness' own reporting bound. */
            const double ulp = std::ldexp(1.0, std::ilogb(std::max(std::fabs(acc), 1e-30f)) - 8);
            if (std::fabs((double)bf(got) - acc) > 4 * ulp) ++bad;
        }
    return bad;
}

/* One (mode, M) case's verdict: the candidate's bytes against the frozen row's, and the frozen
 * row's against the independent integer reference. Both are failures. A pair of rows that agree
 * on the same wrong answer passes the first and must fail the second, which is why a reference
 * deviation cannot stay a printed number only. */
int judge(const std::vector<uint8_t>& bf_out, const std::vector<uint8_t>& bc_out,
          const std::vector<int8_t>& a, const std::vector<float>& asc,
          const std::vector<int8_t>& w, const std::vector<uint16_t>& wsc, int m, int n, int k) {
    size_t diff = 0, first = 0;
    double worst = 0;
    for (size_t i = 0; i < bc_out.size() / 2; ++i) {
        uint16_t u, v;
        std::memcpy(&u, bf_out.data() + 2 * i, 2);
        std::memcpy(&v, bc_out.data() + 2 * i, 2);
        if (u != v) { if (!diff) first = i; ++diff; }
        worst = std::max(worst, std::fabs((double)bf(u) - bf(v)));
    }
    const size_t ref_bad = reference(bf_out, a, asc, w, wsc, m, n, k);
    std::printf("M=%d bytes=%zu frozen_vs_candidate_diffs=%zu first=%ld max_abs=%.9g "
                "frozen_vs_int_reference_bad=%zu\n",
                m, bf_out.size(), diff, diff ? (long)first : -1L, worst, ref_bad);
    return (diff ? 1 : 0) + (ref_bad ? 1 : 0);
}
/* Host only, and deliberately before the first HIP call: the regression for the case this file
 * used to pass. Two rows that agree on the same wrong bytes pass the row-to-row comparison, so
 * the reference check is the only thing that can fail the run -- and it must. The third leg is
 * the control: the same two rows agreeing on the reference's own answer have to be accepted, or
 * "always fails" would satisfy this check.
 *
 *   compare.hip --self-check
 */
int self_check() {
    const int m = 1;
    std::vector<int8_t> a((size_t)K, 1), w((size_t)N * K, 1);
    std::vector<float> asc((size_t)(K / G), 1.0f);
    std::vector<uint16_t> wsc((size_t)N * (K / G), to_bf(1.0f));
    std::vector<uint16_t> expected;
    std::vector<uint8_t> probe((size_t)m * N * 2, 0);
    reference(probe, a, asc, w, wsc, m, N, K, &expected);

    /* One output value for every element: the inputs above are constant, so the reference is too. */
    auto same = [&](uint16_t v) {
        std::vector<uint8_t> b((size_t)m * N * 2);
        for (size_t i = 0; i < (size_t)m * N; ++i) std::memcpy(b.data() + 2 * i, &v, 2);
        return b;
    };
    struct Leg { const char* what; uint16_t v; bool want_fail; };
    const Leg legs[] = {
        { "both rows zero, reference is 384",            0x0000, true  },
        { "both rows NaN",                              0x7fc0, true  },
        { "both rows the reference's own answer",        expected[0], false },
    };
    int fails = 0;
    for (const Leg& l : legs) {
        const std::vector<uint8_t> y = same(l.v);
        const int bad = judge(y, y, a, asc, w, wsc, m, N, K);
        const bool got = bad != 0;
        std::printf("self_check %-40s -> %s (verdict %s)\n", l.what,
                    got ? "FAIL" : "ok", got == l.want_fail ? "as expected" : "WRONG");
        if (got != l.want_fail) ++fails;
    }
    std::printf("RESULT: %s\n", fails ? "SELF-CHECK FAILED" : "SELF-CHECK OK");
    return fails ? 1 : 0;
}

/* ============================================================== the timing run
 *
 *   compare.hip --timing <out.csv>                          one card, arms interleaved ABBA
 *   compare.hip --timing --rank frozen|cand <out.csv> <gofile>
 *                                                           one arm on this card, both cards
 *                                                           started together through <gofile>
 *
 * Both shapes time the same work and the same weights the correctness run above uses: `Rounds`
 * rounds of `Batches` batches of `PerBatch` launches at N=2560, K=384, group 128, the codes
 * rotating over `Planes` 983 040-byte planes, so a plane has left cache when it comes round
 * again. One event pair per batch, so the unit is launch-to-launch throughput; a single launch's
 * latency is the other question and this file does not answer it.
 *
 * The interleave is ABBA inside every round, so a drift over the round cancels instead of loading
 * one arm. The `--rank` shape exists because a --tp 2 step runs two ranks at once and the gain has
 * to survive their sharing the memory system; the two processes contend for it exactly as two
 * ranks do, and each writes its own medians. `*_FROZEN`/`*_CAND` come from the environment
 * (SI8_FROZEN, SI8_CAND) when the paths differ from the defaults.
 */
constexpr int Batches = 9, PerBatch = 256, Rounds = 9, Planes = 64;

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* One arm's planes, activations and outputs, packed by that arm's own layout hook. */
struct Arm {
    std::string name;
    const RadKernelInfo* row = nullptr;
    std::vector<std::unique_ptr<Buffer>> wb, wsb;
    std::unique_ptr<Buffer> a, as;
    std::vector<std::unique_ptr<Buffer>> y;
    std::vector<RadParam> params[3];
    RadArgs args[3]{};
};

int timing(const char* frozen_path, const char* cand_path, bool abba, const char* out_path,
           const char* arm_name, const char* gofile) {
    int dev = 0;
    HIP(hipGetDevice(&dev));
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props, dev));
    std::fprintf(stderr, "pid=%d device=%d %s arch=%s\n", (int)getpid(), dev, props.name,
                 props.gcnArchName);
    check(!std::strncmp(props.gcnArchName, "gfx1201", 7), "gfx1201 required");

    Plugin frozen(frozen_path, "gemm_i8a8_nt_m16");
    Plugin cand(cand_path, "shared_i8a8_down_m1_m3");
    hipStream_t stream = nullptr;
    HIP(hipStreamCreate(&stream));

    const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, G);
    const int sel_codes = 0, sel_scale = 1;

    /* Fixed seed, so every run of this file measures the same bytes. */
    uint64_t rng = 0x243f6a8885a308d3ull;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    std::vector<int8_t> codes((size_t)N * K), act((size_t)3 * K);
    std::vector<uint16_t> scales((size_t)N * (K / G));
    std::vector<float> asc((size_t)3 * (K / G));
    for (auto& v : codes)  v = (int8_t)((int)(next() % 255) - 127);
    for (auto& v : act)    v = (int8_t)((int)(next() % 255) - 127);
    for (auto& v : scales) v = to_bf((float)((int)(next() % 2001) - 1000) / 3103.0f);
    for (auto& v : asc)    v = (float)((int)(next() % 2001) - 1000) / 837.0f;

    auto make_arm = [&](const char* name, const RadKernelInfo* row) {
        Arm arm;
        arm.name = name;
        arm.row = row;
        Params p;
        const std::vector<uint8_t> fw = stored(row, p.v, 2, enc, &sel_codes, codes.data(), RAD_I8,
                                               N, K, nullptr);
        const std::vector<uint8_t> fs = stored(row, p.v, 3, enc, &sel_scale, scales.data(),
                                               RAD_BF16, N, K / G, nullptr);
        for (int i = 0; i < Planes; ++i) {
            arm.wb.push_back(std::make_unique<Buffer>(fw.size()));
            arm.wb.back()->put(fw.data());
            arm.wsb.push_back(std::make_unique<Buffer>(fs.size()));
            arm.wsb.back()->put(fs.data());
        }
        arm.a = std::make_unique<Buffer>(act.size());
        arm.a->put(act.data());
        arm.as = std::make_unique<Buffer>(asc.size() * 4);
        arm.as->put(asc.data());
        for (int m = 1; m <= 3; ++m) {
            arm.y.push_back(std::make_unique<Buffer>((size_t)m * N * 2));
            Params pm;
            pm.set_m(m);
            arm.params[m - 1] = pm.v;
            arm.args[m - 1].p = arm.params[m - 1].data();
            arm.args[m - 1].n_p = (int)arm.params[m - 1].size();
        }
        return arm;
    };

    std::vector<Arm> arms;
    if (abba) {
        arms.push_back(make_arm("frozen", frozen.row));
        arms.push_back(make_arm("cand", cand.row));
    } else {
        const bool use_cand = !std::strcmp(arm_name, "cand");
        arms.push_back(make_arm(use_cand ? "cand" : "frozen", use_cand ? cand.row : frozen.row));
    }

    /* One round: `Batches` batches, one event pair each, median µs per launch over the batches. */
    auto round_us = [&](Arm& arm, int m) {
        const int mi = m - 1;
        const int64_t need = arm.row->scratch ? arm.row->scratch(&arm.args[mi]) : 0;
        Buffer scratch((size_t)std::max<int64_t>(need, 1));
        std::vector<double> per;
        hipEvent_t e0, e1;
        HIP(hipEventCreate(&e0));
        HIP(hipEventCreate(&e1));
        for (int b = 0; b < Batches; ++b) {
            HIP(hipEventRecord(e0, stream));
            for (int i = 0; i < PerBatch; ++i) {
                const int plane = (b * PerBatch + i) % Planes;
                Call c(m, *arm.a, *arm.as, *arm.wb[plane], *arm.wsb[plane], *arm.y[mi]);
                c.a.p = arm.args[mi].p;
                c.a.n_p = arm.args[mi].n_p;
                c.a.scratch = scratch.p;
                c.a.scratch_bytes = need;
                check(arm.row->launch(&c.a, (RadStream)stream) == RAD_OK,
                      arm.name + ": launch refused");
            }
            HIP(hipEventRecord(e1, stream));
            HIP(hipEventSynchronize(e1));
            float ms = 0;
            HIP(hipEventElapsedTime(&ms, e0, e1));
            per.push_back((double)ms * 1000.0 / PerBatch);
        }
        HIP(hipEventDestroy(e0));
        HIP(hipEventDestroy(e1));
        HIP(hipGetLastError());
        return median(per);
    };

    FILE* out = std::fopen(out_path, "w");
    check(out != nullptr, std::string("cannot write ") + out_path);
    int fails = 0;

    if (abba) {
        for (Arm& arm : arms)
            for (int m = 1; m <= 3; ++m) round_us(arm, m);      /* warm */
        for (int m = 1; m <= 3; ++m) {
            std::vector<double> af, bf;
            for (int r = 0; r < Rounds; ++r) {
                const double a1 = round_us(arms[0], m);
                const double b1 = round_us(arms[1], m);
                const double b2 = round_us(arms[1], m);
                const double a2 = round_us(arms[0], m);
                af.push_back(a1);
                af.push_back(a2);
                bf.push_back(b1);
                bf.push_back(b2);
            }
            const auto [flo, fhi] = std::minmax_element(af.begin(), af.end());
            const auto [clo, chi] = std::minmax_element(bf.begin(), bf.end());
            const double fm = median(af), cm = median(bf);
            int faster = 0;
            for (size_t i = 0; i < af.size(); ++i) faster += bf[i] < af[i];
            const double spread = std::max(*fhi - *flo, *chi - *clo);
            std::printf("M=%d frozen_us=%.4f [%.4f..%.4f] cand_us=%.4f [%.4f..%.4f] "
                        "delta_us=%+.4f delta_pct=%+.2f cand_faster=%d/%zu spread_us=%.4f "
                        "gain_over_spread=%.2f\n",
                        m, fm, *flo, *fhi, cm, *clo, *chi, cm - fm, 100.0 * (cm - fm) / fm, faster,
                        af.size(), spread, spread > 0 ? std::fabs(cm - fm) / spread : 0.0);
            std::fprintf(out, "%d,%.4f,%.4f\n", m, fm, cm);
        }
    } else {
        Arm& arm = arms[0];
        for (int m = 1; m <= 3; ++m) round_us(arm, m);             /* warm */
        std::fprintf(stderr, "rank %s ready\n", arm.name.c_str());
        /* Both ranks start their timed rounds together: the parent touches <gofile> once both have
         * warmed, so the window they contend in is the window that is reported. */
        while (!std::ifstream(gofile).good()) {
            struct timespec ts { 0, 10 * 1000 * 1000 };
            nanosleep(&ts, nullptr);
        }
        for (int m = 1; m <= 3; ++m) {
            std::vector<double> v;
            for (int r = 0; r < Rounds; ++r) v.push_back(round_us(arm, m));
            const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
            std::printf("rank=%s M=%d median_us=%.4f [%.4f..%.4f]\n", arm.name.c_str(), m,
                        median(v), *lo, *hi);
            std::fprintf(out, "%s,%d,%.4f\n", arm.name.c_str(), m, median(v));
        }
    }

    std::fclose(out);
    HIP(hipStreamDestroy(stream));
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "--self-check")) return self_check();
    if (argc > 1 && !std::strcmp(argv[1], "--timing")) {
        const char* frozen_path = std::getenv("SI8_FROZEN") ? std::getenv("SI8_FROZEN") : "libr4d.so";
        const char* cand_path   = std::getenv("SI8_CAND") ? std::getenv("SI8_CAND") : "shared_i8a8_down.so";
        if (argc > 3 && !std::strcmp(argv[2], "--rank"))
            return timing(frozen_path, cand_path, false, argv[4], argv[3], argv[5]);
        check(argc > 2, "usage: --timing <out.csv> | --timing --rank frozen|cand <out.csv> <gofile>");
        return timing(frozen_path, cand_path, true, argv[2], nullptr, nullptr);
    }

    const char* frozen_path = argc > 1 ? argv[1] : "libr4d.so";
    const char* cand_path   = argc > 2 ? argv[2] : "shared_i8a8_down.so";

    int dev = 0;
    HIP(hipGetDevice(&dev));
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props, dev));
    std::printf("device %d %s arch=%s lanes=%d\n", dev, props.name, props.gcnArchName, props.warpSize);
    check(!std::strncmp(props.gcnArchName, "gfx1201", 7) && props.warpSize == 32,
          "gfx1201 wave32 required");

    Plugin frozen(frozen_path, "gemm_i8a8_nt_m16");
    Plugin cand(cand_path, "shared_i8a8_down_m1_m3");
    std::printf("frozen_row=%s candidate_row=%s\n", frozen.row->name, cand.row->name);

    hipStream_t stream = nullptr;
    HIP(hipStreamCreate(&stream));

    /* One weight and one activation set per mode, drawn from a fixed seed. */
    int fails = 0;
    for (int mode = 0; mode < 3; ++mode) {
        uint64_t rng = 0x9e3779b97f4a7c15ull ^ (uint64_t)mode;
        auto next = [&] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
        std::vector<int8_t> a((size_t)4 * K), w((size_t)N * K);
        std::vector<float> asc((size_t)4 * (K / G));
        std::vector<uint16_t> wsc((size_t)N * (K / G));
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = mode == 1 ? (int8_t)(i % 4 == 0 ? -128 : i % 4 == 1 ? 127 : -127)
                 : mode == 2 ? (int8_t)0 : (int8_t)((int)(next() % 255) - 127);
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = mode == 1 ? (int8_t)(i % 3 == 0 ? -128 : i % 3 == 1 ? 127 : -127)
                 : mode == 2 ? (int8_t)0 : (int8_t)((int)(next() % 255) - 127);
        for (auto& v : asc) v = mode == 2 ? 1.0f : (float)((int)(next() % 2001) - 1000) / 837.0f;
        for (auto& v : wsc) v = to_bf(mode == 2 ? 1.0f : (float)((int)(next() % 2001) - 1000) / 3103.0f);

        const char* mode_name = mode == 0 ? "random" : mode == 1 ? "signed-extremes" : "zeros";
        const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, G);
        const int sel_codes = 0, sel_scale = 1;

        Params p;
        std::string tag_w, tag_s, ctag_w, ctag_s;
        /* Frozen bytes are the reference layout; the candidate publishes the same hooks, so the
         * comparison of the two results is a real check of the compiled-in copy. */
        auto fw = stored(frozen.row, p.v, 2, enc, &sel_codes, w.data(), RAD_I8, N, K, &tag_w);
        auto fs = stored(frozen.row, p.v, 3, enc, &sel_scale, wsc.data(), RAD_BF16, N, K / G, &tag_s);
        auto cw = stored(cand.row, p.v, 2, enc, &sel_codes, w.data(), RAD_I8, N, K, &ctag_w);
        auto cs = stored(cand.row, p.v, 3, enc, &sel_scale, wsc.data(), RAD_BF16, N, K / G, &ctag_s);
        std::printf("mode=%s tags frozen=(\"%s\",\"%s\") candidate=(\"%s\",\"%s\") "
                    "codes_equal=%d scale_equal=%d bytes=%zu/%zu\n",
                    mode_name, tag_w.c_str(), tag_s.c_str(), ctag_w.c_str(), ctag_s.c_str(),
                    (int)(fw == cw), (int)(fs == cs), fw.size(), cw.size());
        if (fw != cw || fs != cs) { std::printf("  MISMATCH in stored weight bytes\n"); ++fails; }

        Buffer d_a(a.size()), d_asc(asc.size() * 4), d_w(fw.size()), d_ws(fs.size());
        d_a.put(a.data()); d_asc.put(asc.data()); d_w.put(fw.data()); d_ws.put(fs.data());

        for (int m = 1; m <= 3; ++m) {
            p.set_m(m);
            Buffer yf((size_t)m * N * 2), yc((size_t)m * N * 2);
            HIP(hipMemset(yf.p, 0xa5, yf.size));
            HIP(hipMemset(yc.p, 0xa5, yc.size));
            Call cf(m, d_a, d_asc, d_w, d_ws, yf), cc(m, d_a, d_asc, d_w, d_ws, yc);
            cf.a.p = p.v.data(); cf.a.n_p = (int)p.v.size();
            cc.a.p = p.v.data(); cc.a.n_p = (int)p.v.size();

            const int64_t need = frozen.row->scratch ? frozen.row->scratch(&cf.a) : 0;
            Buffer scratch((size_t)std::max<int64_t>(need, 1));
            cf.a.scratch = scratch.p; cf.a.scratch_bytes = need;

            const int rf = frozen.row->launch(&cf.a, (RadStream)stream);
            const int rc = cand.row->launch(&cc.a, (RadStream)stream);
            HIP(hipStreamSynchronize(stream));
            HIP(hipGetLastError());
            check(rf == RAD_OK && rc == RAD_OK,
                  std::string("launch refused: frozen=") + std::to_string(rf) +
                  " candidate=" + std::to_string(rc));

            fails += judge(yf.get(), yc.get(), a, asc, w, wsc, m, N, K);
        }
    }

    /* Negative control: one changed activation code must change the output. Without this the
     * zero-diff result above would also be produced by a comparison that never reached the
     * device. */
    {
        const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, G);
        const int sel_codes = 0, sel_scale = 1;
        std::vector<int8_t> a((size_t)K, 1), w((size_t)N * K, 1);
        std::vector<float> asc((size_t)(K / G), 1.0f);
        std::vector<uint16_t> wsc((size_t)N * (K / G), to_bf(1.0f));
        Params p;
        auto fw = stored(frozen.row, p.v, 2, enc, &sel_codes, w.data(), RAD_I8, N, K, nullptr);
        auto fs = stored(frozen.row, p.v, 3, enc, &sel_scale, wsc.data(), RAD_BF16, N, K / G, nullptr);
        Buffer d_w(fw.size()), d_ws(fs.size()), d_a(a.size()), d_asc(asc.size() * 4);
        d_w.put(fw.data()); d_ws.put(fs.data()); d_asc.put(asc.data());

        auto run = [&](const std::vector<int8_t>& act) {
            d_a.put(act.data());
            Buffer y((size_t)N * 2);
            HIP(hipMemset(y.p, 0xa5, y.size));
            Call c(1, d_a, d_asc, d_w, d_ws, y);
            c.a.p = p.v.data(); c.a.n_p = (int)p.v.size();
            check(cand.row->launch(&c.a, (RadStream)stream) == RAD_OK, "negative control launch refused");
            HIP(hipStreamSynchronize(stream));
            return y.get();
        };
        const auto base = run(a);
        /* A change large enough to move the BF16 result at this magnitude: the output here is
         * ~384, whose BF16 ulp is ~1.5, so a one-code change would be invisible. */
        for (auto& v : a) v = (int8_t)-1;
        const auto changed = run(a);
        std::printf("negative_control changed_input_differs=%d\n", (int)(base != changed));
        if (base == changed) ++fails;
    }

    /* The refusals that must survive: the gate/up geometry, and the fourth row, which the measured
     * band hands to the frozen row. */
    {
        struct Refusal { const char* what; int64_t n, k, m; };
        const Refusal rs[] = { { "gate/up geometry (N=768,K=2560,M=1)", 768, 2560, 1 },
                               { "fourth row (N=2560,K=384,M=4)",       2560, 384, 4 } };
        for (const Refusal& r : rs) {
            Params p;
            p.v[1] = integer("N", r.n);
            p.v[2] = integer("K", r.k);
            p.set_m((int)r.m);
            Buffer b1(1), b2(1), b3(1), b4(1), b5(1);
            Call c((int)r.m, b1, b2, b3, b4, b5);
            c.a.p = p.v.data(); c.a.n_p = (int)p.v.size();
            const int rc = cand.row->launch(&c.a, (RadStream)stream);
            std::printf("%-36s candidate launch status=%d refused=%d\n", r.what, rc, (int)(rc != RAD_OK));
            if (rc == RAD_OK) ++fails;
            /* The description has to agree, or a tool sizes buffers for a launch that refuses. */
            RadOpdDesc d{};
            const int rs2 = cand.row->opd_shape(p.v.data(), (int)p.v.size(), 0, &d);
            std::printf("%-36s candidate opd_shape status=%d refused=%d\n", r.what, rs2,
                        (int)(rs2 != RAD_OK));
            if (rs2 == RAD_OK) ++fails;
        }
    }

    HIP(hipStreamDestroy(stream));
    std::printf("%s\n", fails ? "RESULT: MISMATCH" : "RESULT: BYTE-IDENTICAL AND REFERENCE-CONSISTENT");
    return fails ? 1 : 0;
}
