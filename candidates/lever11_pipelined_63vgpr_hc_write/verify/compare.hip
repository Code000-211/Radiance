#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using U = unsigned short;
extern "C" __global__ void hc_baseline(const U*, U*, const U*, int, int, long, long, long);
extern "C" __global__ void hc_candidate(const U*, U*, const U*, int, int, long, long, long);
static void ok(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
struct Dev {
    U* p = nullptr;
    explicit Dev(size_t n) { ok(hipMalloc(&p, n * sizeof(U))); }
    ~Dev() {
        if (p && hipFree(p) != hipSuccess) {
            std::fputs("FAIL hipFree\n", stderr); std::abort();
        }
    }
};
static std::vector<U> fetch(Dev& d, size_t n) {
    std::vector<U> v(n); ok(hipMemcpy(v.data(), d.p, n * sizeof(U), hipMemcpyDeviceToHost));
    return v;
}
static void put(Dev& d, const std::vector<U>& v) {
    ok(hipMemcpy(d.p, v.data(), v.size() * sizeof(U), hipMemcpyHostToDevice));
}
static void require(bool yes, const char* msg) { if (!yes) throw std::runtime_error(msg); }
static bool one(int rows, int n, int hc, bool negative) {
    constexpr size_t guard = 16;
    const U edge = 0x5a5a;
    long hrow = (long)std::max(1, hc) * n + 16;
    long irow = std::max(1, hc) + 8;
    long yrow = n + 16;
    std::vector<U> h(2 * guard + rows * hrow, edge);
    std::vector<U> inj(2 * guard + rows * irow, edge);
    std::vector<U> y(2 * guard + rows * yrow, edge);
    constexpr U values[] = {0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80, 0x3b80, 0xbb80};
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < std::max(hc, 0); ++c) {
            inj[guard + r * irow + c] = values[(r * 7 + c * 3 + 4) % 8];
            for (int i = 0; i < n; ++i)
                h[guard + r * hrow + c * n + i] = values[(r * 5 + c * 7 + i) % 8];
        }
        for (int i = 0; i < n; ++i) y[guard + r * yrow + i] = values[(r + i * 3 + 4) % 8];
    }
    h[guard] = 0; inj[guard] = 0x3f80; y[guard] = 0x3f80;
    Dev dh(h.size()), di(inj.size()), dy(y.size());
    put(dh, h); put(di, inj); put(dy, y);
    U* hp = hc > 0 ? dh.p + guard : nullptr;
    const U* ip = hc > 0 ? di.p + guard : nullptr;
    const U* yp = dy.p + guard;
    void* args[] = {&ip, &hp, &yp, &n, &hc, &hrow, &irow, &yrow};
    const dim3 grid((n / 8 + 127) / 128, rows);
    ok(hipLaunchKernel((const void*)hc_baseline, grid, dim3(128), args, 0, nullptr));
    ok(hipDeviceSynchronize());
    const auto base = fetch(dh, h.size());
    if (hc > 0) require(base[guard] == 0x3f80, "baseline did not execute known 0+1*1 write");
    else require(base == h, "nonpositive hc changed guarded output");
    put(dh, h);
    if (negative) { inj[guard] = 0x4000; put(di, inj); }
    ok(hipLaunchKernel((const void*)hc_candidate, grid, dim3(128), args, 0, nullptr));
    ok(hipDeviceSynchronize());
    const auto got = fetch(dh, h.size());
    require(fetch(di, inj.size()) == inj, "injection input or guards mutated");
    require(fetch(dy, y.size()) == y, "reduced input or guards mutated");
    require(std::equal(got.begin(), got.begin() + guard, h.begin()), "prefix output guard overwritten");
    require(std::equal(got.end() - guard, got.end(), h.end() - guard), "suffix output guard overwritten");
    for (int r = 0; r < rows; ++r) {
        const size_t pad = guard + r * hrow + std::max(hc, 0) * n;
        require(std::equal(got.begin() + pad, got.begin() + guard + (r + 1) * hrow,
                           h.begin() + pad), "row padding overwritten");
    }
    if (negative) {
        require(got != base && got[guard] == 0x4000, "real input negative control did not fire");
        return true;
    }
    require(got == base, "candidate differs from independent baseline header");
    return true;
}
int main(int argc, char** argv) {
    try {
        const int card = argc > 1 ? std::atoi(argv[1]) : 0;
        ok(hipSetDevice(card));
        hipDeviceProp_t props{}; ok(hipGetDeviceProperties(&props, card));
        std::printf("device=%d arch=%s baseline=hc_baseline candidate=hc_candidate\n", card, props.gcnArchName);
        int cases = 0;
        for (int rows : {1, 4}) for (int n : {8, 24, 2560})
            for (int hc : {-1, 0, 1, 2, 4, 8, 16}) {
                one(rows, n, hc, false); ++cases;
            }
        one(1, 8, 1, true);
        std::printf("PASS cases=%d byte_exact guards=PASS immutable_inputs=PASS known_write=PASS negative_control=PASS\n", cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what()); return 1;
    }
}
