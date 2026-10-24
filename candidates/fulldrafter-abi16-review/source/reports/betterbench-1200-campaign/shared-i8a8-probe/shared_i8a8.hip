// Private experiment: one launch, no scratch, no serving registration.
// Weight is the frozen i8 fragment64 permutation; scales are tile-major [N/16][K/128][16].
#ifdef __FAST_MATH__
#error "The reordered FP32 fold requires precise floating-point compilation"
#endif
#include "../../../libr4d/r4d_emit.h"
#include "../../../libr4d/r4d_common.h"
#include <array>

namespace shared_i8a8 {
constexpr int Group = 128;

__global__ __launch_bounds__(32) void dot(const int8_t* __restrict__ w,
    const uint16_t* __restrict__ ws, const int8_t* __restrict__ a,
    const float* __restrict__ as, uint16_t* __restrict__ y,
    int M, int N, int K) {
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int groups = K / Group;
    float out[4] = {};
    // A lane owns one output column; each packed weight word serves every live token.
    for (int g = 0; g < groups; ++g) {
        int32_t sum[4] = {};
        for (int super = 0; super < 2; ++super) {
            const int tile = ((col / 16) * (K / 64) + g * 2 + super) * 1024;
            for (int frag = 0; frag < 4; ++frag) {
                for (int half = 0; half < 2; ++half) {
                    const int k = g * Group + super * 64 + frag * 16 + half * 8;
                    const int lane = (col & 15) + half * 16;
                    const uint32_t* wp = (const uint32_t*)(w + tile + lane * 32 + frag * 8);
                    const uint32_t w0 = wp[0], w1 = wp[1];
                    for (int m = 0; m < M; ++m) {
                        const uint32_t* ap = (const uint32_t*)(a + m * K + k);
                        sum[m] = __builtin_amdgcn_sudot4(true, w0, true, ap[0], sum[m], false);
                        sum[m] = __builtin_amdgcn_sudot4(true, w1, true, ap[1], sum[m], false);
                    }
                }
            }
        }
        const uint32_t bits = (uint32_t)ws[((col / 16) * groups + g) * 16 + col % 16] << 16;
        const float scale = __builtin_bit_cast(float, bits);
        for (int m = 0; m < M; ++m)
            out[m] += (float)sum[m] * (as[m * groups + g] * scale);
    }
    for (int m = 0; m < M; ++m) {
        y[m * N + col] = f32_to_bf16(out[m]);
    }
}

int body(const RadArgs* args, const R4dSink& sink) {
    if (!args || args->n_t < 5 || !args->t || !args->p) return RAD_E_INVAL;
    const auto* a = r4d_opt(args, 0);
    const auto* as = r4d_opt(args, 1);
    const auto* w = r4d_opt(args, 2);
    const auto* ws = r4d_opt(args, 3);
    const auto* y = r4d_opt(args, 4);
    if (!a || !as || !w || !ws || !y) return RAD_E_INVAL;
    if (a->dtype != RAD_I8 || as->dtype != RAD_F32 || w->dtype != RAD_I8 ||
        ws->dtype != RAD_BF16 || y->dtype != RAD_BF16) return RAD_E_DTYPE;
    const long long M = r4d_geti_or(args, "M", 0);
    const long long N = r4d_geti_or(args, "N", 0);
    const long long K = r4d_geti_or(args, "K", 0);
    if (M < 1 || M > 4 || !((N == 768 && K == 2560) || (N == 2560 && K == 384)) ||
        r4d_geti_or(args, "group", 0) != Group || !r4d_gets(args, "dtype") ||
        std::strcmp(r4d_gets(args, "dtype"), "i8a8") ||
        a->rank != 2 || as->rank != 2 || w->rank != 2 || ws->rank != 2 || y->rank != 2 ||
        a->shape[0] != M || a->shape[1] != K || as->shape[0] != M ||
        as->shape[1] != K / Group || w->shape[0] != N || w->shape[1] != K ||
        ws->shape[0] != N || ws->shape[1] != K / Group ||
        y->shape[0] != M || y->shape[1] != N) return RAD_E_SHAPE;
    // This row deliberately supports only the exact packed/tight geometry in the frozen container.
    if (r4d_row_pitch(a) != K || r4d_row_pitch(as) != K / Group ||
        r4d_row_pitch(w) != K || r4d_row_pitch(ws) != K / Group ||
        r4d_row_pitch(y) != N) return RAD_E_STRIDE;
    if (!r4d_aligned(a->data, 4) || !r4d_aligned(w->data, 4) ||
        !r4d_aligned(as->data, 4) || !r4d_aligned(ws->data, 2) ||
        !r4d_aligned(y->data, 2)) return RAD_E_ALIGN;
    return R4D_EMIT(sink, dot, dim3((unsigned)N / 32), dim3(32), 0,
                    (const int8_t*)w->data, (const uint16_t*)ws->data,
                    (const int8_t*)a->data, (const float*)as->data,
                    (uint16_t*)y->data, (int)M, (int)N, (int)K);
}
int gateup_body(const RadArgs* a, const R4dSink& sink) {
    if (!a || r4d_geti_or(a, "N", 0) != 768 || r4d_geti_or(a, "K", 0) != 2560)
        return RAD_E_SHAPE;
    return body(a, sink);
}
int down_body(const RadArgs* a, const R4dSink& sink) {
    if (!a || r4d_geti_or(a, "N", 0) != 2560 || r4d_geti_or(a, "K", 0) != 384)
        return RAD_E_SHAPE;
    return body(a, sink);
}
R4D_ENTRY(rad_shared_i8a8_gateup_probe, gateup_body)
R4D_ENTRY(rad_shared_i8a8_down_probe, down_body)

const RadKernelInfo* row(int i) {
    if (i < 0 || i > 1) return nullptr;
    static const RadConstraint constraints[2][6] = {
        {RAD_CGE("M", 1), RAD_CLE("M", 4), RAD_CEQ("N", 768),
         RAD_CEQ("K", 2560), RAD_CEQ("group", Group), RAD_CIN("dtype", "i8a8")},
        {RAD_CGE("M", 1), RAD_CLE("M", 4), RAD_CEQ("N", 2560),
         RAD_CEQ("K", 384), RAD_CEQ("group", Group), RAD_CIN("dtype", "i8a8")}
    };
    static const std::array<RadKernelInfo, 2> values = [] {
        RadKernelInfo r[2]{};
        for (int n = 0; n < 2; ++n) {
            r[n].name = n ? "shared_i8a8_dot4_down_probe" : "shared_i8a8_dot4_gateup_probe";
            r[n].op = "gemm_nt_q"; r[n].family = "gemm";
            r[n].computes = "signed i8 packed dot, int32 per 128 K, FP32 scale/fold, BF16 output";
            r[n].shape = n ? "M=1..4 N=2560 K=384 group=128 tight rows"
                           : "M=1..4 N=768 K=2560 group=128 tight rows";
            r[n].dtypes = "I8 A and blocked W; F32 A scales; BF16 W scales and Y";
            r[n].domain = RAD_DOMAIN_DEVICE; r[n].priority = 0;
            r[n].constraints = constraints[n]; r[n].n_constraints = 6;
            r[n].launch = n ? rad_shared_i8a8_down_probe : rad_shared_i8a8_gateup_probe;
            r[n].describe = n ? rad_shared_i8a8_down_probe_describe : rad_shared_i8a8_gateup_probe_describe;
        }
        return std::array<RadKernelInfo, 2>{r[0], r[1]};
    }();
    // No private layout hook: the comparator obtains stored planes via the frozen row's hooks.
    return &values[i];
}
} // namespace shared_i8a8

extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {RAD_PLUGIN_KERNEL, "shared_i8a8_probe", "0.0.0",
                                        "isolated small-M signed dot experiment", "gfx1201"};
    return &info;
}
extern "C" int rad_kernel_schema_count() { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count() { return 2; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return shared_i8a8::row(i);
}
