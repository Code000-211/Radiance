// r4d_ple_gate_bf16.hip -- Per-Layer Embedding's dot-product gate.
//
// ============================== WHAT IT COMPUTES ==============================
//
//   K[c] = grouped_rmsnorm(k)[c] * (w_key   + wadd)     k is key_proj(embedding)
//   Q[c] = grouped_rmsnorm(q)[c] * (w_query + wadd)     q is the INCOMING hc stream
//   s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
//   s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))     <- the SIGNED square root
//   gv[c] = sigmoid(s[c]) * v                           v is value_proj(embedding), ONE row
//   gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)
//
// libref/ref_ple.cpp is the oracle and docs/OPS.md the contract. THE SIGN COMES FROM THE ORIGINAL,
// not from the clamped magnitude: torch's sign(0) is 0, so a dot of exactly zero stays zero instead
// of becoming sqrt(gate_eps), and a quantised activation row produces exact zeros.
//
// ============================== ONE WORKGROUP A (ROW, STREAM) ==============================
//
// Every reduction here is over ONE stream's `n` -- two for the norms, one for the dot, one for the
// gate's own rms -- and a workgroup is the widest thing that reduces without a second launch. So a
// workgroup owns one (row, stream) pair and each thread carries n/threads elements of it.
//
// AT DECODE THAT IS FOUR WORKGROUPS, which is 1.6% of this part, and it is the right shape anyway:
// the op moves 107 KB a row and is latency-bound long before it is occupancy-bound. Splitting the
// reduction across workgroups would buy parallelism the launch floor eats.
//
// ============================== WHY k AND q LIVE IN REGISTERS ==============================
//
// The naive form reads k and q once for the norm and again for the dot. Each thread instead keeps
// its own n/threads elements across both passes -- 10 values at this model's shape, 20 halves, ten
// VGPRs -- and the second read disappears. The same trick carries `gv` from where it is written to
// where its rms is taken.
//
// AND THAT IS NOT A SHORTCUT PAST THE NARROWING. `gv` is narrowed to bf16 BEFORE the rms sees it,
// because the reference writes a bf16 tensor and norms what was written; keeping the narrowed value
// in a register is the same bits as storing and reading it back. Keeping the f32 accumulator would
// be a different number -- the epilogue-fold rule the GDN projection states at more length.
//
// ============================== AN INVARIANT WORTH KNOWING ==============================
//
// gv[c] is a SCALAR times one shared v row, and norm_conv's rms is per stream -- so the gate divides
// straight back out and `gvn` does not depend on it at all. Exact only in exact arithmetic (gv is
// narrowed first), so it is a thing to CHECK a fused kernel against, never a thing to fold.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cstdint>

namespace {

enum { kPgThreads = 256 };

/* A value narrowed to bf16 and read back, which is what the reference's `.type_as(x)` does after
 * every norm. One definition, used four times below. */
__device__ __forceinline__ float pg_narrow(float v) {
    return bf16_to_f32(f32_to_bf16(v));
}

/* expf and NOT __expf. The fast intrinsic is a couple of ULP off, and this gate is compared against
 * libref's std::exp to a tolerance meant for summation order -- one op in one layer of the model is
 * not where to spend accuracy on an instruction. */
__device__ __forceinline__ float pg_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* A workgroup-wide sum through LDS, called four times.
 * Each wave reduces in registers/DPP (r4d_wave_sum) with 0 barriers, writes 8 wave sums to LDS,
 * wave 0 folds the 8 sums with r4d_xor_sum<8>, and broadcasts via lds[0].
 * Reduces barriers from 9 down to 2! */
__device__ __forceinline__ float pg_reduce(float v, float* lds) {
    const uint32_t t = threadIdx.x;
    const uint32_t wave = t / 32;
    const uint32_t lane = t % 32;
    v = r4d_wave_sum(v);
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    if (wave == 0) {
        float s = (lane < (kPgThreads / 32)) ? lds[lane] : 0.0f;
        s = r4d_xor_sum<8>(s);
        if (lane == 0) lds[0] = s;
    }
    __syncthreads();
    return lds[0];
}

__global__ __launch_bounds__(kPgThreads) void ple_gate_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ v, const float* __restrict__ wk,
    const float* __restrict__ wq, const float* __restrict__ wc,
    uint16_t* __restrict__ gv, uint16_t* __restrict__ gvn,
    uint32_t n, uint32_t hc, uint32_t k_ld, uint32_t q_ld, uint32_t v_ld,
    uint32_t g_ld, uint32_t gn_ld, float eps, float wadd, float geps, uint32_t per_thread) {

    __shared__ float lds[kPgThreads];
    /* `gv` is carried from where it is written to where its rms is taken. 16 is the widest this
     * kernel serves: n <= 16 * kPgThreads = 4096, which the row constrains. */
    float mine_k[16], mine_q[16], mine_gv[16];

    const uint32_t row = blockIdx.x / hc;
    const uint32_t c   = blockIdx.x % hc;
    const uint32_t t   = threadIdx.x;
    const uint32_t base = c * n;                 /* this stream's first column of the wide row */

    /* ---- the two norms, from values each thread then keeps ------------------------------------ */
    float sk = 0.0f, sq = 0.0f;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float kv = bf16_to_f32(k[(size_t)row * k_ld + base + d]);
        const float qv = bf16_to_f32(q[(size_t)row * q_ld + base + d]);
        mine_k[i] = kv;
        mine_q[i] = qv;
        sk += kv * kv;
        sq += qv * qv;
    }
    const float rk = rsqrtf(pg_reduce(sk, lds) / (float)n + eps);
    const float rq = rsqrtf(pg_reduce(sq, lds) / (float)n + eps);

    /* ---- the dot between them, over the NARROWED normed values -------------------------------- */
    float dot = 0.0f;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float kk = pg_narrow(mine_k[i] * rk * (wk[base + d] + wadd));
        const float qq = pg_narrow(mine_q[i] * rq * (wq[base + d] + wadd));
        dot += kk * qq;
    }
    float s = pg_reduce(dot, lds) * rsqrtf((float)n);

    /* THE SIGNED SQUARE ROOT, in the reference's order: clamp the MAGNITUDE, take the root, and
     * take the SIGN FROM THE ORIGINAL. sign(0) is 0, so an exactly-zero dot stays zero. */
    const float mag = sqrtf(fabsf(s) < geps ? geps : fabsf(s));
    s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);
    const float gate = pg_sigmoid(s);

    /* ---- the gated value, narrowed, and its own rms from the narrowed bytes ------------------- */
    float sg = 0.0f;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float x = pg_narrow(gate * bf16_to_f32(v[(size_t)row * v_ld + d]));
        mine_gv[i] = x;
        gv[(size_t)row * g_ld + base + d] = f32_to_bf16(x);
        sg += x * x;
    }
    const float rg = rsqrtf(pg_reduce(sg, lds) / (float)n + eps);

    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        gvn[(size_t)row * gn_ld + base + d] =
            f32_to_bf16(pg_narrow(mine_gv[i] * rg * (wc[base + d] + wadd)));
    }
}

enum { PG_K = 0, PG_Q, PG_V, PG_WK, PG_WQ, PG_WC, PG_GV, PG_GVN };

}  // namespace

extern "C" int r4d_rad_ple_gate(const RadArgs* a, RadStream s) {
    const RadTensor* k   = r4d_opt(a, PG_K);
    const RadTensor* q   = r4d_opt(a, PG_Q);
    const RadTensor* v   = r4d_opt(a, PG_V);
    const RadTensor* wk  = r4d_opt(a, PG_WK);
    const RadTensor* wq  = r4d_opt(a, PG_WQ);
    const RadTensor* wc  = r4d_opt(a, PG_WC);
    const RadTensor* gv  = r4d_opt(a, PG_GV);
    const RadTensor* gvn = r4d_opt(a, PG_GVN);
    if (!k || !q || !v || !wk || !wq || !wc || !gv || !gvn) return R4D_NO(RAD_E_INVAL);
    if (k->dtype != RAD_BF16 || q->dtype != RAD_BF16 || v->dtype != RAD_BF16 ||
        gv->dtype != RAD_BF16 || gvn->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wk->dtype != RAD_F32 || wq->dtype != RAD_F32 || wc->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);

    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long hn = hc * n;
    const long long per_thread = (n + kPgThreads - 1) / kPgThreads;
    if (per_thread > 16) return R4D_NO(RAD_E_UNSUPPORTED);

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float geps = (float)r4d_getf(a, "gate_eps", 1e-6);

    long long T = 0, tq = 0, tv = 0, tg = 0, tn = 0;
    const long long k_ld  = r4d_rows_pitch(k, hn, &T);
    const long long q_ld  = r4d_rows_pitch(q, hn, &tq);
    const long long v_ld  = r4d_rows_pitch(v, n, &tv);
    const long long g_ld  = r4d_rows_pitch(gv, hn, &tg);
    const long long gn_ld = r4d_rows_pitch(gvn, hn, &tn);
    if (k_ld < 0 || q_ld < 0 || v_ld < 0 || g_ld < 0 || gn_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* THE ROW COUNT COMES FROM THE OPERANDS AND EVERY ONE HAS TO AGREE: `M` is ranged, so the
     * value in the args is the band's upper bound, and a grid sized by it walks off the arena. */
    if (T <= 0 || tq != T || tv != T || tg != T || tn != T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(wk) < hn || r4d_numel(wq) < hn || r4d_numel(wc) < hn)
        return R4D_NO(RAD_E_SHAPE);

    hipLaunchKernelGGL(ple_gate_kernel, dim3((uint32_t)(T * hc)), dim3(kPgThreads), 0,
                       (hipStream_t)s,
                       (const uint16_t*)r4d_addr(k), (const uint16_t*)r4d_addr(q),
                       (const uint16_t*)r4d_addr(v), (const float*)r4d_addr(wk),
                       (const float*)r4d_addr(wq), (const float*)r4d_addr(wc),
                       (uint16_t*)r4d_addr(gv), (uint16_t*)r4d_addr(gvn),
                       (uint32_t)n, (uint32_t)hc, (uint32_t)k_ld, (uint32_t)q_ld, (uint32_t)v_ld,
                       (uint32_t)g_ld, (uint32_t)gn_ld, eps, wadd, geps, (uint32_t)per_thread);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
