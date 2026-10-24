// Standalone split-K HC down prototype; never registered in mine or the serving plugin.
// Row name is kept as hc_down_wmma_m3_m4_probe only because the frozen comparator selects
// the row by that name; it is still a private, unregistered experiment.
//
// numerik: verlustfrei-nicht-bitgenau. ONLY the FP32 reduction over the 80 independent
// 128-code down groups changes: wave w folds groups g = w, w+8, ... (its own ten) with the
// existing `fmaf(raw, scale, total)` in ascending g, and the eight wave partials are then
// summed in FP32 in ascending wave order. Inside a group nothing moves: the eight staged
// 16-K chunks each feed the SAME FP32 accumulator (`gemm<1>` with the running raw as its C),
// which is the instruction sequence gemm<8> issues, so the 128-code raw dot is bit-identical
// to the unsplit row; the group's existing FP32 scale is still applied to that FP32 raw dot
// after the group, and every intermediate format stays FP32 or exact E4M3-as-BF16.
// No scaled weight is narrowed, no accumulator is narrowed, E4M3 codes remain exact in BF16,
// and the original 256-thread RMS order, __frsqrt_rn, normalized-N f2bf boundary, inject
// reduction, down f2bf(silu(...)) boundary and the unchanged upstream up/mix/quant kernel
// are all untouched. This is a source-level declaration, not a parity measurement.
#ifdef __FAST_MATH__
#error "HC down probe requires strict floating point compilation"
#endif
#include "rad_abi.h"
#pragma GCC visibility push(hidden)
#include "../../../mine/hc_up_wmma.hip"
#pragma GCC visibility pop

// HC_PUBLISH_PARALLEL=0 (default): grid 21, the single publisher CTA 20 loops every row.
// HC_PUBLISH_PARALLEL=1: grid 20+M, publisher CTA 20+r publishes row r only, and CTR zeroing
// happens in publisher 0 alone, still ahead of the next kernel in this stream.
#ifndef HC_PUBLISH_PARALLEL
#define HC_PUBLISH_PARALLEL 0
#endif
static_assert(HC_PUBLISH_PARALLEL == 0 || HC_PUBLISH_PARALLEL == 1,
              "HC_PUBLISH_PARALLEL is the 0/1 publication-variant switch");

namespace hc_down_splitk {
constexpr int N = 2560, HC = 4, LR = 320, HN = N * HC, Group = 128;
constexpr int DownPitch = r4d_hc8::row_bytes(HN, Group);
constexpr int UpPitch = r4d_hc8::row_bytes(LR, LR / 4);
constexpr int Slots = 2 * N / Group;              // 40 arrival counters of the next kernel
constexpr int TileCols = 16;                      // down columns owned by one CTA
constexpr int kMaxRows = 4;                       // M band this row serves
constexpr int Chunk = 16;                         // K staged per subchunk: one WMMA step
constexpr int Pitch = 24;                         // padded row pitch of the 16-K A/B chunks
constexpr int Groups = HN / Group;                // 80 independent 128-code groups
constexpr int GroupsPerWave = Groups / kWaves;    // 10 groups per wave, one per iteration
constexpr bool kPar = HC_PUBLISH_PARALLEL != 0;

static_assert(kWave == 2 * TileCols, "one lane per 8-element item of a 16-row chunk");
static_assert(DownPitch == 10560 && UpPitch == 336);
static_assert(Groups % kWaves == 0 && Group % Chunk == 0 && Chunk % 8 == 0);
static_assert(Pitch >= Chunk + 8 && Pitch % 8 == 0, "fragment reads k and k+8, 8-byte aligned");

// Grid (20 + (kPar ? M : 1)): 20 CTAs own 16 down columns each; CTAs 20..20+M-1 publish.
// Scratch: [band*LR BF16 d][band*kHcMax FP32 scales][40 u32 arrival counters], unchanged.
// Across all eight waves, A and B each occupy 6 KiB (16x24 BF16 per wave).
// The two staging arrays total 12 KiB; only live-row partials occupy another 2 KiB.
__global__ __launch_bounds__(kThreads)
void down(const unsigned short* __restrict__ h, const float* __restrict__ w,
          const unsigned char* __restrict__ dw, const unsigned short* __restrict__ iw,
          unsigned short* __restrict__ d, unsigned short* __restrict__ inj,
          float* __restrict__ scale, unsigned int* __restrict__ ctr,
          int rows, float eps, float wadd) {
    extern __shared__ __attribute__((aligned(16))) unsigned short sN[];
    __shared__ float part[kWaves][kHcMax], norms[kMaxRows][kHcMax];
    __shared__ unsigned short stage[kWaves][kWave * 8];
    __shared__ float partial[kWaves][kMaxRows][TileCols];
    __shared__ __attribute__((aligned(16))) unsigned short a[kWaves][TileCols * Pitch];
    __shared__ __attribute__((aligned(16))) unsigned short b[kWaves][TileCols * Pitch];
    const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
    const int pub = (int)blockIdx.x - LR / TileCols;
    const bool publisher = pub >= 0;
    if (pub >= rows) return;                      // M > rows: this publisher owns no row
    if (pub == 0 && tid < Slots) ctr[tid] = 0;    // publisher 0 alone, before the next kernel

    // A parallel publisher needs only its own row's scale; a serial one needs every row.
    // Down CTAs always need every row. Both bounds are CTA-uniform.
    const int r0 = (kPar && publisher) ? pub : 0;
    const int r1 = (kPar && publisher) ? pub + 1 : rows;

    // Preserve hc_read_a_kernel<true>'s per-row eight-load, eight-add, wave_sum,
    // eight-wave reduction, and frsqrt order. M=3 with band=4 visits three rows.
    for (int r = r0; r < r1; ++r) {
        const unsigned short* hp = h + r * HN;
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
            norms[r][tid] = __frsqrt_rn(ss / (float)N + eps);
            if (publisher) scale[r * kHcMax + tid] = norms[r][tid];
        }
        __syncthreads();
    }

    if (publisher) {
        // Only a publisher needs the whole 10240-element normalized row in LDS; it reuses
        // the same storage per physical row. The inject dot keeps its original order.
        for (int r = r0; r < r1; ++r) {
            const unsigned short* hp = h + r * HN;
            for (int c = 0; c < HC; ++c) {
                const int cb = c * N;
                const float sc = norms[r][c];
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
            for (int c = wave; c < HC; c += kWaves) {
                const float sum = wave_sum(wave_dot_lds<kDotU>(sN, iw + c * HN, HN, lane, stage[wave]));
                if (lane == 0) inj[r * HC + c] = f2bf(2.0f * sigmoidf_(sum * (1.0f / HC)));
            }
            __syncthreads();
        }
        return;
    }

    // Down dot. Every wave owns ten independent 128-code groups and stages only its own
    // 16-row A and 16-column B chunk, one 8-element item per lane each. All eight waves run
    // the same GroupsPerWave*Group/Chunk iterations, so every block barrier is uniform.
    unsigned short* const aw = a[wave];
    unsigned short* const bw = b[wave];
    const int col = blockIdx.x * TileCols + (lane & 15);
    // One 8-element item per lane: item i covers row/column i>>1 and K half (i&1)*8, so the
    // 32 lanes fill a 16x16 chunk exactly. frag() then reads k = {0..3, 8..11} for lane>>4 == 0
    // and {4..7, 12..15} for lane>>4 == 1, which is the "two runs of four" layout.
    const int ir = lane >> 1, ik = (lane & 1) * 8;
    const float* const wsc = (const float*)(dw + col * DownPitch + HN);
    v8f total = {};
    for (int i = 0; i < GroupsPerWave; ++i) {
        const int g = wave + i * kWaves;
        v8f raw = {};
        for (int s = 0; s < Group / Chunk; ++s) {
            const int k0 = g * Group + s * Chunk;
            __syncthreads();
            union { uint4 v; unsigned short s[8]; } av, bv;
            if (ir < rows) {
                const unsigned short* hp = h + ir * HN + k0 + ik;
                const float* wp = w + k0 + ik;
                const float sc = norms[ir][(k0 + ik) / N];
                unsigned short hv[8];
                float wv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) { hv[e] = hp[e]; wv[e] = wp[e]; }
#pragma unroll
                for (int e = 0; e < 8; ++e) av.s[e] = f2bf(bf2f(hv[e]) * sc * (wv[e] + wadd));
            } else {
                av.v = make_uint4(0, 0, 0, 0);
            }
            *(uint4*)(aw + ir * Pitch + ik) = av.v;
            const int column = blockIdx.x * TileCols + ir;
            const uint2 codes = *(const uint2*)(dw + column * DownPitch + k0 + ik);
            float decoded[8];
            e4m3x8(codes.x, codes.y, decoded);
#pragma unroll
            for (int e = 0; e < 8; ++e) bv.s[e] = f2bf(decoded[e]);
            *(uint4*)(bw + ir * Pitch + ik) = bv.v;
            __syncthreads();
            // Running raw as C: the same mma chain gemm<Group/16> would issue over this group.
            raw = gemm<1>(aw, Pitch, 0, bw, Pitch, 0, lane, raw);
        }
        // The existing FP32 group scale applies to the FP32 128-code raw dot, after the group.
        const float sc = wsc[g];
#pragma unroll
        for (int e = 0; e < 8; ++e) total[e] = fmaf(raw[e], sc, total[e]);
    }
    // gfx1201 gemm: lane & 15 selects the column, output e selects A row e for lanes 0..15.
    // Only the live rows reach LDS; rows 8..15 of lanes 16..31 never store.
    if (lane < TileCols) {
#pragma unroll
        for (int rr = 0; rr < kMaxRows; ++rr)
            if (rr < rows) partial[wave][rr][lane] = total[rr];
    }
    __syncthreads();
    if (tid < kMaxRows * TileCols) {
        const int rr = tid / TileCols, lc = tid % TileCols;
        if (rr < rows) {
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < kWaves; ++k) acc += partial[k][rr][lc];
            d[rr * LR + blockIdx.x * TileCols + lc] = f2bf(siluf_(acc * (1.0f / HC)));
        }
    }
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
    int rc = R4D_EMIT(sk, down, dim3(LR / TileCols + (kPar ? band : 1)), dim3(kThreads),
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
R4D_ENTRY(rad_hc_down_splitk, body)

const RadKernelInfo* row() {
    static const RadConstraint constraints[] = {
        RAD_CGE("M", 3), RAD_CLE("M", 4), RAD_CEQ("n", N), RAD_CEQ("hc", HC),
        RAD_CEQ("lowrank", LR), RAD_CEQ("inject", 1), RAD_CEQ("group", Group),
        RAD_CIN("dtype", "bf16"), RAD_CIN("mix", "e4m3"),
    };
    static const RadKernelInfo value = [] {
        RadKernelInfo r{};
        r.name = "hc_down_wmma_m3_m4_probe"; r.op = "hc_read"; r.family = "norm";
        r.computes = "BF16 normalized N; eight-wave split-K unscaled BF16 WMMA down; original inject/up/mix/quant";
        r.shape = "M=3..4 n=2560 hc=4 lowrank=320 inject=1 group=128 tight rows";
        r.dtypes = "BF16 H/inject/x; E4M3+FP32 per-group down/up; I8/E4M3 optional q";
        r.domain = RAD_DOMAIN_DEVICE; r.priority = 0;
        r.constraints = constraints; r.n_constraints = sizeof(constraints) / sizeof(constraints[0]);
        r.launch = rad_hc_down_splitk; r.describe = rad_hc_down_splitk_describe;
        r.scratch = scratch; r.layout = r4d_rad_layout_hc_e4m3;
        r.relayout = r4d_rad_relayout_hc_e4m3;
        r.unrelayout = r4d_rad_unrelayout_hc_e4m3; r.opd_shape = r4d_shape_hc_read;
        return r;
    }();
    return &value;
}
} // namespace hc_down_splitk

// This DSO is an isolated experiment, not the actual serving mine plugin.
extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {RAD_PLUGIN_KERNEL, "hc_down_splitk", "0.0.0", "isolated split-K HC down probe", "gfx1201"};
    return &info;
}
// This row implements the existing hc_read schema; the installed libr4d owns it.
extern "C" int rad_kernel_schema_count() { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count() { return 1; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i == 0 ? hc_down_splitk::row() : nullptr;
}
