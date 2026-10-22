/* plugin/spechub/spechub.hip -- the device launches for the two coupling ops.
 *
 * The kernels are one workgroup per sampled row, and the workgroup is the whole parallel machine
 * the verify needs: the cuts, the residual mass and the walk are all row-local. The arithmetic
 * itself is abi/rad_spechub.h (inline, host and device), so this file is a launch and nothing more
 * -- which is why the host reference and the kernel can be compared on the same bytes.
 *
 * STRICT FP64. The coupling sums FP64 masses in ascending candidate order and compares them against
 * a 53-bit uniform; --fast-math would let the compiler contract those sums and change the answer.
 * The #error below is the compile-time half of that, and CMakeLists sets -ffp-contract=off.
 */
#ifdef __FAST_MATH__
#error "the MTP coupling requires strict FP64 arithmetic: build without -ffast-math"
#endif

#include <hip/hip_runtime.h>

#include "spechub.h"
#include "../../libr4d/r4d_emit.h"

namespace {

/* One thread a row: both ops are row-local and the M is the number of sampled positions. */
__global__ void propose(RadMtpCoupleProposeCall c) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < c.M) rad_mtp_couple_propose_row(c, row);
}

/* ONE WORKGROUP A ROW, 128 threads. The verify's planes are shared: at n_cand 1024 the weights and
 * the residual masses are 16 KiB of FP64, and the mass walk reads them once per candidate rather
 * than once per thread. */
constexpr int kThreads = 128;
constexpr int64_t kMaxCand = 1024;

__global__ void couple_verify(const int32_t* idx, const float* values, const uint32_t* params,
                              const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                              int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                              int64_t ld4) {
    RadMtpCoupleVerifyCall c{ idx, values, params, state, token, M, n_cand, vocab,
                              { ld0, ld1, ld2, ld3, ld4 } };
    const int64_t row = blockIdx.x;
    if (row >= M) return;
    /* ABOVE THE NARROW BAND the general algorithm runs single-threaded: the shared planes are sized
     * for kMaxCand and a wider row would overrun them. That is the 1025-and-above fallback, and it is
     * the same FP64 order the cooperative path below keeps. */
    if (n_cand > kMaxCand) {
        if (threadIdx.x == 0) rad_mtp_couple_verify_row(c, row);
        return;
    }
    /* THE TWO COOPERATIVE PLANES. `weight` is P's unnormalised FP64 mass per candidate and `mass` is
     * the residual (P-Q)+ per candidate; both are written by the whole workgroup and read in
     * ASCENDING candidate order afterwards, which is the order the serial algorithm sums in. */
    __shared__ double weight[kMaxCand], mass[kMaxCand], pq[8], sum;
    __shared__ float  top;
    __shared__ int    eligible, first[8], ix, dominated;
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    const int32_t* ids = c.idx + row * c.ld[0];
    const float*   val = c.val + row * c.ld[1];
    /* The eligibility gate is the DECIDE's own: the same state, seed, position and target check, so
     * an unbound row writes nothing and keeps the ordinary token. */
    if (threadIdx.x == 0) {
        eligible = 0;
        if (s.status == 1 && !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed &&
            s.pos == p.pos && rad_mtp_couple_state_valid(s, c.vocab)) {
            float maximum = -INFINITY;
            bool  valid   = true;
            for (int64_t j = 0; j < c.n_cand; ++j) {
                if (ids[j] < 0) continue;
                if (ids[j] >= c.vocab || val[j] != val[j] || val[j] == INFINITY) {
                    valid = false;
                    break;
                }
                if (val[j] > maximum) maximum = val[j];
            }
            if (valid && maximum != -INFINITY) { top = maximum; eligible = 1; }
        }
    }
    __syncthreads();
    if (!eligible) return;
    /* THE FP64 MASS, done COOPERATIVELY: each thread takes a strided slice of the candidates and
     * writes exp(v - max) into its own slot. No cross-thread reduction happens here -- the sum below
     * is still walked in ascending order by one thread, which is what keeps the sum bit-identical to
     * the serial algorithm. */
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x)
        if (ids[j] >= 0 && val[j] != -INFINITY) weight[j] = ::exp((double)val[j] - top);
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int k = 0; k < s.count; ++k) { pq[k] = 0; first[k] = -1; }
        for (int64_t j = 0; j < c.n_cand; ++j) {
            if (ids[j] < 0 || val[j] == -INFINITY) continue;
            const double w = weight[j];
            sum += w;
            for (int k = 0; k < s.count; ++k)
                if (ids[j] == s.ids[k]) {
                    if (first[k] < 0) first[k] = (int)j;
                    pq[k] += w;
                    break;
                }
        }
        if (!(sum > 0) || !rad_spechub_finite(sum)) eligible = 0;
        if (eligible) {
            ix = -1;
            dominated = true;
            for (int k = 0; k < s.count; ++k) {
                pq[k] /= sum;
                dominated = dominated && pq[k] >= s.q[k];
                if (s.ids[k] == s.x) ix = k;
            }
            if (ix < 0) eligible = 0;
        }
    }
    __syncthreads();
    if (!eligible) return;
    /* THE RESIDUAL MASS, COOPERATIVELY. A Q token contributes max(0, P-Q) ONCE, at its first
     * occurrence (`first[k]`), which is the duplicate-first rule; every other candidate contributes
     * its own P. The plane is written per slot, so the walk below reads it in candidate order. */
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x) {
        double m = 0;
        if (ids[j] >= 0 && val[j] != -INFINITY) {
            bool in_q = false;
            for (int k = 0; k < s.count; ++k)
                if (s.ids[k] == ids[j]) {
                    in_q = true;
                    if (first[k] == (int)j) m = rad_spechub_positive(pq[k] - s.q[k]);
                    break;
                }
            if (!in_q) m = weight[j] / sum;
        }
        mass[j] = m;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    /* THE ORDERED WALK. One thread, ascending candidate index, exactly the serial algorithm's
     * arithmetic: the sums above and this walk are what the coupling's bit-exactness rests on, and
     * parallelising either would reorder them. */
    double resid = 0;
    for (int64_t j = 0; j < c.n_cand; ++j) resid += mass[j];
    int32_t out = -1;
    const double u_resid = rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int64_t j = 0; j < c.n_cand; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = weight[j];
                if (w > 0) { out = ids[j]; cdf -= w; if (cdf < 0) break; }
            }
    } else if (pq[ix] >= s.q[ix] ||
               rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT) * s.q[ix] < pq[ix]) {
        out = s.x;
    } else {
        double cdf = u_resid * resid;
        for (int64_t j = 0; j < c.n_cand; ++j) {
            if (ids[j] == s.x) continue;
            const double m = mass[j];
            if (m > 0) { out = ids[j]; cdf -= m; if (cdf < 0) break; }
        }
    }
    c.token[row * c.ld[4]] = out;
}

}  /* namespace */

int rad_mine_mtp_couple_propose_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, propose, dim3((unsigned)((c.M + 63) / 64)), dim3(64), 0, c);
}

int rad_mine_mtp_couple_verify_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    /* Separate pointer arguments let graph replay rebind each operand address independently: a
     * struct on the host would freeze the addresses it was built with. */
    return R4D_EMIT(stream, couple_verify, dim3((unsigned)c.M), dim3(kThreads), 0,
                    c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                    c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
}
