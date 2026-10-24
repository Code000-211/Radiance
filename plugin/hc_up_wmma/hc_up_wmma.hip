/* hc_up_wmma.hip -- the M=3/4 HC read with its up-dot reduction reordered onto WMMA.
 *
 * Ported from the ABI16 archive's `candidates/fulldrafter-abi16-review/source/mine/hc_up_wmma.hip`
 * (PR #1, 5c3bb1d) to this tree's ABI (26). The port is deliberately narrow: the down+inject
 * routine, the operand order, the mixing-weight rows and the layouts are upstream's, INCLUDED
 * here rather than restated, and only `up_mix` differs -- an unscaled BF16 WMMA dot per 80-column
 * slice with the FP32 scale applied after it. See plugin/hc_up_wmma/README.md.
 *
 * The upstream sources are pulled in for their device templates and host layouts and their entry
 * points must not become THIS library's exports: everything below `#pragma GCC visibility push`
 * is hidden, so loading this plugin cannot interpose on libr4d and the loader keeps both.
 */
#include <hip/hip_runtime.h>
#include "rad_abi.h"
#pragma GCC visibility push(hidden)
#include "../../libr4d/r4d_emit.h"
#include "../../libr4d/r4d_hc_bf16.hip"
#include "../../libr4d/r4d_fp8_layout.cpp"
#include "../../libr4d/r4d_shapes.cpp"
#pragma GCC visibility pop

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hc_up_wmma {
constexpr int N = 2560, HC = 4, LR = 320, HN = N * HC, Group = 128;
constexpr int UpPitch = r4d_hc8::row_bytes(LR, LR / 4);
constexpr int DownPitch = r4d_hc8::row_bytes(HN, Group);

// E4M3 codes are exact BF16 values. The FP32 scale is applied AFTER each
// 80-column dot; no scaled weight is narrowed to BF16.
__global__ __launch_bounds__(128)
void up_mix(const unsigned char* __restrict__ up, const unsigned short* __restrict__ d,
            const unsigned short* __restrict__ h, const float* __restrict__ w,
            const float* __restrict__ scale, unsigned short* __restrict__ x,
            int rows, float wadd) {
    constexpr int Channels = 32, Cols = HC * Channels, K = LR / 4, Pitch = K + 8;
    __shared__ __attribute__((aligned(16))) unsigned short a[16 * Pitch], b[Cols * Pitch];
    __shared__ unsigned short u[4 * Cols];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
    const int i0 = blockIdx.x * Channels;
    v8f total[2] = {}, pair[2] = {};
    for (int g = 0; g < 4; ++g) {
        __syncthreads();
        for (int t = tid; t < 16 * (K / 8); t += 128) {
            const int r = t / (K / 8), k8 = (t % (K / 8)) * 8;
            const uint4 v = r < rows ? *(const uint4*)(d + r * LR + g * K + k8)
                                    : make_uint4(0, 0, 0, 0);
            *(uint4*)(a + r * Pitch + k8) = v;
        }
        for (int t = tid; t < Cols * (K / 8); t += 128) {
            const int col = t / (K / 8), k8 = (t % (K / 8)) * 8;
            const int j = (col / Channels) * N + i0 + col % Channels;
            const uint2 codes = *(const uint2*)(up + j * UpPitch + g * K + k8);
            float decoded[8];
            e4m3x8(codes.x, codes.y, decoded);
            union { uint4 v; unsigned short s[8]; } out;
#pragma unroll
            for (int e = 0; e < 8; ++e) out.s[e] = f2bf(decoded[e]);
            *(uint4*)(b + col * Pitch + k8) = out.v;
        }
        __syncthreads();
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            const int btile = wave + tile * 4;
            const v8f raw = gemm<K / 16>(a, Pitch, 0, b, Pitch, btile, lane, v8f{});
            const int col = btile * 16 + (lane & 15);
            const int j = (col / Channels) * N + i0 + col % Channels;
            const float sc = ((const float*)(up + j * UpPitch + LR))[g];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = raw[e] * sc;
                if ((g & 1) == 0) pair[tile][e] = v;
                else {
                    pair[tile][e] += v;
                    if (g == 1) total[tile][e] = pair[tile][e];
                    else total[tile][e] += pair[tile][e];
                }
            }
        }
    }
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int col = (wave + tile * 4) * 16 + (lane & 15), r0 = 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (r0 + e < rows) u[(r0 + e) * Cols + col] = f2bf(sigmoidf_(total[tile][e]));
    }
    __syncthreads();
    for (int idx = tid; idx < rows * Channels; idx += 128) {
        const int r = idx / Channels, ii = idx % Channels, i = i0 + ii;
        float acc = 0.0f;
        for (int c = 0; c < HC; ++c) {
            const int j = c * N + i;
            const float nv = nbf(bf2f(h[r * HN + j]) * scale[r * kHcMax + c] * (w[j] + wadd));
            acc += bf2f(u[r * Cols + c * Channels + ii]) * nv;
        }
        x[r * N + i] = f2bf(acc * (1.0f / HC));
    }
}

template<bool I8>
__global__ __launch_bounds__(128)
void quant(const unsigned short* x, unsigned char* q, float* qs,
           unsigned char* rq, float* rqs, int rows) {
    const int r = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int group = blockIdx.x, base = group * Group;
    const unsigned short* xr = x + r * N + base;
    float m = 0.0f;
    for (int t = lane; t < Group; t += 32) m = fmaxf(m, fabsf(bf2f(xr[t])));
    m = r4d_wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * (N / Group) + group] = sc;
    const int c4 = lane * 4;
    *(unsigned*)(q + r * N + base + c4) =
        q8_pack4<I8>(bf2f(xr[c4]), bf2f(xr[c4 + 1]), bf2f(xr[c4 + 2]), bf2f(xr[c4 + 3]), inv);
    if (rq) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[c4 + j]);
        hc_rot_group(v, lane, rq + r * N + base, rqs + r * (N / Group) + group);
    }
}

int64_t scratch(const RadArgs* a) {
    if (!a) return RAD_E_INVAL;
    const auto m = r4d_geti_or(a, "M", 0);
    if (m < 3 || m > 4 || r4d_geti_or(a, "n", 0) != N ||
        r4d_geti_or(a, "hc", 0) != HC || r4d_geti_or(a, "lowrank", 0) != LR)
        return RAD_E_SHAPE;
    return m * (LR * 2 + kHcMax * 4);
}

bool dense(const RadTensor* t, uint32_t dtype, int width, int rows, unsigned align) {
    if (!t || t->dtype != dtype || ((uintptr_t)t->data & (align - 1))) return false;
    long long count = 0;
    return r4d_rows_pitch(t, width, &count) == width && count >= rows;
}

int body(const RadArgs* a, RadStream s) {
    const int64_t need = scratch(a);
    if (need < 0) return (int)need;
    if (!a->t || a->n_t != 11) return RAD_E_INVAL;
    const RadTensor* h = r4d_opt(a, HCR_H), *w = r4d_opt(a, HCR_W);
    const RadTensor* down = r4d_opt(a, HCR_DOWN), *up = r4d_opt(a, HCR_UP);
    const RadTensor* iw = r4d_opt(a, HCR_INJW), *x = r4d_opt(a, HCR_X), *inj = r4d_opt(a, HCR_INJ);
    const RadTensor* q = r4d_opt(a, HCR_Q), *qs = r4d_opt(a, HCR_S);
    const RadTensor* rq = r4d_opt(a, HCR_RQ), *rqs = r4d_opt(a, HCR_RS);
    if (!h || !w || !down || !up || !iw || !x || !inj) return RAD_E_INVAL;
    long long rows = 0;
    if (r4d_rows_pitch(h, HN, &rows) != HN || rows < 3 || rows > 4 || rows > r4d_geti_or(a, "M", 0))
        return RAD_E_SHAPE;
    if (!dense(h, RAD_BF16, HN, rows, 16) || !dense(x, RAD_BF16, N, rows, 2) ||
        !dense(inj, RAD_BF16, HC, rows, 2)) return RAD_E_STRIDE;
    if (w->dtype != RAD_F32 || iw->dtype != RAD_BF16 || down->dtype != RAD_F8E4M3 || up->dtype != RAD_F8E4M3)
        return RAD_E_DTYPE;
    if (r4d_numel(w) != HN || r4d_numel(iw) != HC * HN ||
        down->rank != 2 || down->shape[0] != LR || down->shape[1] != HN ||
        up->rank != 2 || up->shape[0] != HN || up->shape[1] != LR) return RAD_E_SHAPE;
    if (!r4d_contig(w) || !r4d_contig(iw) || !r4d_contig(down) || !r4d_contig(up) ||
        ((uintptr_t)w->data & 3u) || (((uintptr_t)iw->data | (uintptr_t)down->data | (uintptr_t)up->data) & 15u))
        return RAD_E_STRIDE;
    if ((q != nullptr) != (qs != nullptr) || (rq != nullptr) != (rqs != nullptr) || (rq && !q))
        return RAD_E_INVAL;
    if (q && q->dtype != RAD_I8 && q->dtype != RAD_F8E4M3) return RAD_E_DTYPE;
    if (q && (r4d_geti_or(a, "group", 0) != Group || !dense(q, q->dtype, N, rows, 4) ||
              !dense(qs, RAD_F32, N / Group, rows, 4))) return RAD_E_SHAPE;
    if (rq && (!dense(rq, RAD_F8E4M3, N, rows, 4) || !dense(rqs, RAD_F32, N / Group, rows, 4)))
        return RAD_E_SHAPE;
    if (!a->scratch || a->scratch_bytes < need || ((uintptr_t)a->scratch & 15u)) return RAD_E_SCRATCH;
    auto* d = (unsigned short*)a->scratch;
    auto* scales = (float*)(d + r4d_geti_or(a, "M", 0) * LR);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    // The base row's down ladder, verbatim (want_wg() is a cached device query, not a per-step
    // cost). The port's one claim is that ONLY the up-dot reduction changed, so the down launch has
    // to be the base row's own geometry -- otherwise a measured difference is the ladder's.
    long long ta = kDownTAMax;
    while (ta > 4 && ((LR + ta - 1) / ta) * rows < 2 * want_wg()) ta >>= 1;
    int rc = R4D_EMIT(s, hc_read_a_kernel<true>, dim3((LR + ta - 1) / ta + 1, rows),
        dim3(kThreads), HN * sizeof(unsigned short), (const unsigned short*)h->data,
        (const float*)w->data, (const void*)down->data, (long)DownPitch, (const unsigned short*)iw->data,
        d, (unsigned short*)inj->data, scales, (long)rows, (long)N, HC, (long)LR, ta,
        (long)HN, (long)HC, eps, wadd);
    if (rc != RAD_OK) return rc;
    rc = R4D_EMIT(s, up_mix, dim3(N / 32), dim3(128), 0, (const unsigned char*)up->data,
        d, (const unsigned short*)h->data, (const float*)w->data, scales,
        (unsigned short*)x->data, (int)rows, wadd);
    if (rc != RAD_OK || !q) return rc;
#define HC_UP_QUANT(I8_) \
    R4D_EMIT(s, quant<I8_>, dim3(N / Group), dim3(128), 0, (const unsigned short*)x->data, \
        (unsigned char*)q->data, (float*)qs->data, rq ? (unsigned char*)rq->data : nullptr, \
        rq ? (float*)rqs->data : nullptr, (int)rows)
    rc = q->dtype == RAD_I8 ? HC_UP_QUANT(true) : HC_UP_QUANT(false);
#undef HC_UP_QUANT
    return rc;
}


int launch(const RadArgs* a, RadStream s) { return body(a, s); }

const RadKernelInfo* row() {
    static const RadConstraint constraints[] = {
        RAD_CGE("M", 3), RAD_CLE("M", 4), RAD_CEQ("n", N), RAD_CEQ("hc", HC),
        RAD_CEQ("lowrank", LR), RAD_CEQ("inject", 1), RAD_CEQ("group", Group),
        RAD_CIN("dtype", "bf16"), RAD_CIN("mix", "e4m3"),
    };
    static const RadKernelInfo value = [] {
        RadKernelInfo r{};
        r.name = "hc_read_up_wmma_m3_m4"; r.op = "hc_read"; r.family = "norm";
        r.computes = "unchanged HC down/inject; unscaled BF16 WMMA up; stateless quantization";
        r.shape = "M=3..4, n=2560, hc=4, lowrank=320, inject=1, group=128; dense rows";
        r.dtypes = "bf16 activations; existing E4M3/F32 mixing rows; I8 or E4M3 quantized outputs";
        r.domain = RAD_DOMAIN_DEVICE;
        /* Selection is by plugin hierarchy first (core/plugin/select.cpp): the scan stops at the
         * first library with a matching row, so priority is only compared between rows of ONE
         * plugin. 100 cannot outbid libr4d across the boundary; it is kept high so a future
         * sibling row inside this library loses to this one. */
        r.priority = 100;
        r.constraints = constraints; r.n_constraints = sizeof(constraints) / sizeof(constraints[0]);
        r.launch = launch; r.scratch = scratch;
        r.layout = r4d_rad_layout_hc_e4m3; r.relayout = r4d_rad_relayout_hc_e4m3;
        r.unrelayout = r4d_rad_unrelayout_hc_e4m3; r.opd_shape = r4d_shape_hc_read;
        return r;
    }();
    return &value;
}
}  /* namespace hc_up_wmma */

/* Off unless RADIANCE_HC_UP_WMMA=1, so a library on the search path changes nothing by itself.
 * With the switch set the row is offered, and by the default path order this library comes first
 * (hc_up_wmma.so < libavx.so < libr4d.so), so the row is what `hc_read` resolves to. */
static bool hc_up_wmma_enabled() {
    const char* v = std::getenv("RADIANCE_HC_UP_WMMA");
    return v && std::strcmp(v, "1") == 0;
}

/* One line, once, when the row is first offered: the switch is the only thing standing between an
 * installed library and the unmeasured row, and it is otherwise invisible in operation. The
 * plugin ABI has no log hook -- RAD_INFO is core-internal (core/rad_internal.h) and a plugin links
 * only rad_abi -- so stderr is the channel a plugin has. */
/* Hidden with the rest of this library's internals: only the six rad_* plugin exports are dynamic
 * (see exports.cpp and README.md, "What it does not change"). */
#pragma GCC visibility push(hidden)
extern "C" int rad_hc_up_wmma_row_count(void) {
    if (!hc_up_wmma_enabled()) return 0;
    static const bool announced = [] {
        std::fprintf(stderr,
                     "[hc_up_wmma] RADIANCE_HC_UP_WMMA=1: offering hc_read_up_wmma_m3_m4 "
                     "(M=3..4, unmeasured on a card)\n");
        return true;
    }();
    (void)announced;
    return 1;
}

extern "C" const RadKernelInfo* rad_hc_up_wmma_row_at(int i) {
    return i == 0 && hc_up_wmma_enabled() ? hc_up_wmma::row() : nullptr;
}
#pragma GCC visibility pop
