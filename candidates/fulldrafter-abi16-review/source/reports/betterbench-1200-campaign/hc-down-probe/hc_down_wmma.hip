// Standalone experimental HC read row; not registered in mine or the serving engine.
// numerik: verlustfrei-nicht-bitgenau. Only W_down's FP32 reduction tree changes:
// 128-code dot in WMMA FP32, then its existing FP32 scale, then FP32 group accumulation.
// E4M3 codes are represented exactly in BF16; scaled weights are NEVER narrowed.
// Kernel A repeats upstream's per-CTA grouped RMS and BF16 N construction;
// kernel B is the unchanged upstream hc_up_mix_launch, including quantization.
#ifdef __FAST_MATH__
#error "HC down probe requires strict floating point compilation"
#endif
#include "../../../mine/hc_up_wmma.hip"

namespace hc_down_probe {
constexpr int N = 2560, HC = 4, LR = 320, HN = N * HC, Group = 128;
constexpr int DownPitch = r4d_hc8::row_bytes(HN, Group);
constexpr int UpPitch = r4d_hc8::row_bytes(LR, LR / 4);
constexpr int Slots = 2 * N / Group;  // M=3 uses RG=2: two row groups of 20.
#ifndef HC_DOWN_K
#define HC_DOWN_K 128
#endif
constexpr int TileK = HC_DOWN_K, Pitch = TileK + 8;
static_assert(TileK == 128 || TileK == 256 || TileK == 512);
static_assert(HN % TileK == 0 && TileK % Group == 0 && DownPitch == 10560 && UpPitch == 336);

// Grid (21,M): 20 tiles of 16 columns plus one publisher/inject CTA per row.
// Scratch: [band*LR BF16 d][band*8 FP32 scales][40 u32 arrival counters].
// The publisher initializes counters before the next kernel starts, even with dirty scratch.
__global__ __launch_bounds__(kThreads)
void down(const unsigned short* __restrict__ h, const float* __restrict__ w,
          const unsigned char* __restrict__ dw, const unsigned short* __restrict__ iw,
          unsigned short* __restrict__ d, unsigned short* __restrict__ inj,
          float* __restrict__ scale, unsigned int* __restrict__ ctr,
          int rows, float eps, float wadd) {
    extern __shared__ __attribute__((aligned(16))) unsigned short sN[];
    __shared__ float part[kWaves][kHcMax], norms[kHcMax];
    __shared__ unsigned short stage[kWaves][kWave * 8];
    __shared__ __attribute__((aligned(16))) unsigned short a[16 * Pitch], b[16 * Pitch];
    const int r = blockIdx.y, tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
    if (r >= rows) return;
    const unsigned short* hp = h + r * HN;
    const bool publisher = blockIdx.x == LR / 16;
    // Match hc_read_a_kernel<true>'s 8-load-then-8-add order, wave_sum and
    // ascending eight-wave reduction. The tail is identical at N=2560.
    for (int c = 0; c < HC; ++c) {
        const int cb = c * N;
        float acc = 0.0f;
        int i = tid;
        for (; i + 7 * kThreads < N; i += 8 * kThreads) {
            unsigned short hv[8];
#pragma unroll
            for (int p = 0; p < 8; ++p) hv[p] = hp[cb + i + p * kThreads];
#pragma unroll
            for (int p = 0; p < 8; ++p) { const float v = bf2f(hv[p]); acc += v * v; }
        }
        for (; i < N; i += kThreads) { const float v = bf2f(hp[cb + i]); acc += v * v; }
        acc = wave_sum(acc);
        if (lane == 0) part[wave][c] = acc;
    }
    __syncthreads();
    if (tid < HC) {
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += part[k][tid];
        norms[tid] = __frsqrt_rn(ss / (float)N + eps);
        if (publisher) scale[r * kHcMax + tid] = norms[tid];
    }
    if (publisher && tid < Slots && r == 0) ctr[tid] = 0;
    __syncthreads();
    // F8's original non-swizzled N path. Preserve the BF16 rounding boundary.
    for (int c = 0; c < HC; ++c) {
        const int cb = c * N;
        const float sc = norms[c];
        int i = tid;
        for (; i + 7 * kThreads < N; i += 8 * kThreads) {
            float wv[8];
            unsigned short hv[8];
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                wv[p] = w[cb + i + p * kThreads];
                hv[p] = hp[cb + i + p * kThreads];
            }
#pragma unroll
            for (int p = 0; p < 8; ++p)
                sN[cb + i + p * kThreads] = f2bf(bf2f(hv[p]) * sc * (wv[p] + wadd));
        }
        for (; i < N; i += kThreads)
            sN[cb + i] = f2bf(bf2f(hp[cb + i]) * sc * (w[cb + i] + wadd));
    }
    __syncthreads();
    if (publisher) {
        for (int c = wave; c < HC; c += kWaves) {
            const float sum = wave_sum(wave_dot_lds<kDotU>(sN, iw + c * HN, HN, lane, stage[wave]));
            if (lane == 0) inj[r * HC + c] = f2bf(2.0f * sigmoidf_(sum * (1.0f / HC)));
        }
        return;
    }
    // Each WMMA dot consumes exactly one 128-code scale group. No scaled BF16 weights.
    // Wave 0 owns the 16 columns; the other waves cooperate in staging.
    v8f total = {};
    const int col = blockIdx.x * 16 + (lane & 15);
    const unsigned char* wr = dw + col * DownPitch;
    for (int k0 = 0; k0 < HN; k0 += TileK) {
        __syncthreads();
        for (int t = tid; t < 16 * (TileK / 8); t += kThreads) {
            const int row = t / (TileK / 8), k8 = (t % (TileK / 8)) * 8;
            *(uint4*)(a + row * Pitch + k8) = row == 0
                ? *(const uint4*)(sN + k0 + k8) : make_uint4(0, 0, 0, 0);
        }
        for (int t = tid; t < 16 * (TileK / 8); t += kThreads) {
            const int j = t / (TileK / 8), k8 = (t % (TileK / 8)) * 8;
            const int column = blockIdx.x * 16 + j;
            const uint2 codes = *(const uint2*)(dw + column * DownPitch + k0 + k8);
            float decoded[8];
            e4m3x8(codes.x, codes.y, decoded);
            union { uint4 v; unsigned short s[8]; } out;
#pragma unroll
            for (int e = 0; e < 8; ++e) out.s[e] = f2bf(decoded[e]);
            *(uint4*)(b + j * Pitch + k8) = out.v;
        }
        __syncthreads();
        if (wave == 0) {
#pragma unroll
            for (int sub = 0; sub < TileK / Group; ++sub) {
                const v8f raw = gemm<Group / 16>(a + sub * Group, Pitch, 0,
                                                  b + sub * Group, Pitch, 0, lane, v8f{});
                const float sc = ((const float*)(wr + HN))[(k0 + sub * Group) / Group];
#pragma unroll
                for (int e = 0; e < 8; ++e) total[e] = fmaf(raw[e], sc, total[e]);
            }
        }
    }
    if (wave == 0 && lane < 16)
        d[r * LR + col] = f2bf(siluf_(total[0] * (1.0f / HC)));
}

int64_t scratch(const RadArgs* a) {
    if (!a) return RAD_E_INVAL;
    const long long m = r4d_geti_or(a, "M", 0);
    if (m < 3 || m > 4 || r4d_geti_or(a, "n", 0) != N ||
        r4d_geti_or(a, "hc", 0) != HC || r4d_geti_or(a, "lowrank", 0) != LR ||
        r4d_geti_or(a, "inject", 0) != 1 || r4d_geti_or(a, "group", 0) != Group ||
        r4d_geti_or(a, "rotate", 0) < 0 || r4d_geti_or(a, "rotate", 0) > 1 ||
        !r4d_gets(a, "dtype") || std::strcmp(r4d_gets(a, "dtype"), "bf16") ||
        !r4d_gets(a, "mix") || std::strcmp(r4d_gets(a, "mix"), "e4m3")) return RAD_E_SHAPE;
    return m * (LR * 2 + kHcMax * 4) + Slots * 4;
}

bool dense(const RadTensor* t, uint32_t dt, int width, int rows, unsigned align) {
    if (!t || !t->data || t->dtype != dt || t->rank != 2 || t->shape[0] != rows ||
        t->shape[1] != width || ((uintptr_t)t->data & (align - 1))) return false;
    long long count = 0;
    return r4d_rows_pitch(t, width, &count) == width && count == rows;
}

int body(const RadArgs* a, const R4dSink& sk) {
    const int64_t need = scratch(a);
    if (need < 0) return (int)need;
    if (!a->t || a->n_t != 11) return RAD_E_INVAL;
    const auto* h = r4d_opt(a, HCR_H), *w = r4d_opt(a, HCR_W);
    const auto* dw = r4d_opt(a, HCR_DOWN), *up = r4d_opt(a, HCR_UP);
    const auto* iw = r4d_opt(a, HCR_INJW), *x = r4d_opt(a, HCR_X), *inj = r4d_opt(a, HCR_INJ);
    const auto* q = r4d_opt(a, HCR_Q), *qs = r4d_opt(a, HCR_S);
    const auto* rq = r4d_opt(a, HCR_RQ), *rqs = r4d_opt(a, HCR_RS);
    if (!h || !w || !dw || !up || !iw || !x || !inj) return RAD_E_INVAL;
    if (h->dtype != RAD_BF16 || w->dtype != RAD_F32 || dw->dtype != RAD_F8E4M3 ||
        up->dtype != RAD_F8E4M3 || iw->dtype != RAD_BF16 || x->dtype != RAD_BF16 ||
        inj->dtype != RAD_BF16) return RAD_E_DTYPE;
    const int rows = (int)h->shape[0], band = (int)r4d_geti_or(a, "M", 0);
    if (h->rank != 2 || rows < 3 || rows > band || rows > 4 ||
        w->rank != 1 || w->shape[0] != HN || iw->rank != 2 ||
        iw->shape[0] != HC || iw->shape[1] != HN || dw->rank != 2 ||
        dw->shape[0] != LR || dw->shape[1] != HN || up->rank != 2 ||
        up->shape[0] != HN || up->shape[1] != LR) return RAD_E_SHAPE;
    if (!dense(h, RAD_BF16, HN, rows, 16) || !dense(x, RAD_BF16, N, rows, 2) ||
        !dense(inj, RAD_BF16, HC, rows, 2) || !r4d_contig(w) || !r4d_contig(iw) ||
        !r4d_contig(dw) || !r4d_contig(up) ||
        ((uintptr_t)w->data & 3u) ||
        (((uintptr_t)iw->data | (uintptr_t)dw->data | (uintptr_t)up->data) & 15u))
        return RAD_E_STRIDE;
    if ((q != nullptr) != (qs != nullptr) || (rq != nullptr) != (rqs != nullptr) || (rq && !q) ||
        (bool)rq != (r4d_geti_or(a, "rotate", 0) == 1)) return RAD_E_INVAL;
    if (q && q->dtype != RAD_I8 && q->dtype != RAD_F8E4M3) return RAD_E_DTYPE;
    if (q && qs->dtype != RAD_F32) return RAD_E_DTYPE;
    if (rq && (rq->dtype != RAD_F8E4M3 || rqs->dtype != RAD_F32)) return RAD_E_DTYPE;
    if (q && (!dense(q, q->dtype, N, rows, 4) || !dense(qs, RAD_F32, N / Group, rows, 4)))
        return RAD_E_STRIDE;
    if (rq && (!dense(rq, RAD_F8E4M3, N, rows, 4) ||
               !dense(rqs, RAD_F32, N / Group, rows, 4))) return RAD_E_STRIDE;
    if (!a->scratch || a->scratch_bytes < need || ((uintptr_t)a->scratch & 15u)) return RAD_E_SCRATCH;
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    auto* d = (unsigned short*)a->scratch;
    auto* scales = (float*)(d + band * LR);
    auto* ctr = (unsigned int*)(scales + band * kHcMax);
    int rc = R4D_EMIT(sk, down, dim3(LR / 16 + 1, rows), dim3(kThreads),
                      HN * sizeof(unsigned short), (const unsigned short*)h->data,
                      (const float*)w->data, (const unsigned char*)dw->data,
                      (const unsigned short*)iw->data, d, (unsigned short*)inj->data,
                      scales, ctr, rows, eps, wadd);
    if (rc != RAD_OK) return rc;
    // kUpRows/HC=16: four CTAs per 128-channel group, one stream's counter
    // is initialized by kernel A and self-reset by upstream kernel B.
    return hc_up_mix_launch(up, UpPitch, true, q && q->dtype == RAD_I8, d,
        (const unsigned short*)h->data, (const float*)w->data, scales,
        (unsigned short*)x->data, q ? (unsigned char*)q->data : nullptr,
        q ? (float*)qs->data : nullptr, rq ? (unsigned char*)rq->data : nullptr,
        rq ? (float*)rqs->data : nullptr, q ? ctr : nullptr, rows, N, HC, LR,
        HN, N, N, N / Group, N, N / Group, wadd, sk);
}
R4D_ENTRY(rad_hc_down_probe, body)

const RadKernelInfo* row() {
    static const RadConstraint constraints[] = {
        RAD_CGE("M", 3), RAD_CLE("M", 4), RAD_CEQ("n", N), RAD_CEQ("hc", HC),
        RAD_CEQ("lowrank", LR), RAD_CEQ("inject", 1), RAD_CEQ("group", Group),
        RAD_CIN("dtype", "bf16"), RAD_CIN("mix", "e4m3"),
    };
    static const RadKernelInfo value = [] {
        RadKernelInfo r{};
        r.name = "hc_down_wmma_m3_m4_probe"; r.op = "hc_read"; r.family = "norm";
        r.computes = "BF16 normalized N; unscaled BF16 WMMA down; original inject/up/mix/quant";
        r.shape = "M=3..4 n=2560 hc=4 lowrank=320 inject=1 group=128 tight rows";
        r.dtypes = "BF16 H/inject/x; E4M3+FP32 per-group down/up; I8/E4M3 optional q";
        r.domain = RAD_DOMAIN_DEVICE; r.priority = 0;
        r.constraints = constraints; r.n_constraints = sizeof(constraints) / sizeof(constraints[0]);
        r.launch = rad_hc_down_probe; r.describe = rad_hc_down_probe_describe;
        r.scratch = scratch; r.layout = r4d_rad_layout_hc_e4m3;
        r.relayout = r4d_rad_relayout_hc_e4m3;
        r.unrelayout = r4d_rad_unrelayout_hc_e4m3; r.opd_shape = r4d_shape_hc_read;
        return r;
    }();
    return &value;
}
} // namespace hc_down_probe

// This DSO is an isolated experiment, not the actual serving mine plugin.
extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {RAD_PLUGIN_KERNEL, "hc_down_probe", "0.0.0", "isolated HC down probe", "gfx1201"};
    return &info;
}
// This row implements the existing hc_read schema; the installed libr4d owns it.
extern "C" int rad_kernel_schema_count() { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count() { return 1; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i == 0 ? hc_down_probe::row() : nullptr;
}
