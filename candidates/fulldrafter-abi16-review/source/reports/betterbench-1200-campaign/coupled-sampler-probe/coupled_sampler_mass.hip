// Isolated second MTP verify experiment; coupled_sampler.hip remains the immutable first variant.
// Build without fast math, finite-math-only, or FP contraction; the serial FP64 order is the ABI.
#ifdef __FAST_MATH__
#error "coupled sampler requires strict FP64 arithmetic"
#endif
#include <hip/hip_runtime.h>
#include "../../../libr4d/r4d_emit.h"
// Reuse the production ABI validator, geometry and state checks without rebuilding mine.so.
#include "../../../mine/spechub.cpp"

namespace coupled_sampler_probe {
constexpr int MaxCand = 1024, Threads = 128;

// Separate pointer arguments let graph replay update each moving operand address.
__global__ void verify(const int32_t* idx, const float* values, const uint32_t* params,
                       const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                       int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                       int64_t ld4) {
    RadMtpCoupleVerifyCall c{idx, values, params, state, token, M, n_cand, vocab,
                             {ld0, ld1, ld2, ld3, ld4}};
    const int64_t row = blockIdx.x;
    if (row >= c.M) return;
    __shared__ double weight[MaxCand], mass[MaxCand], pq[8], sum;
    __shared__ float top;
    __shared__ int eligible, first[8], ix, dominated;
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    const int32_t* ids = c.idx + row * c.ld[0];
    const float* val = c.val + row * c.ld[1];
    if (threadIdx.x == 0) {
        eligible = 0;
        if (s.status == 1 && !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed &&
            s.pos == p.pos && rad_mtp_couple_state_valid(s, c.vocab)) {
            float maximum = -INFINITY;
            bool valid = true;
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
    if (!eligible) return;  // No eager exp and no output write on invalid/unbound rows.
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x)
        if (ids[j] >= 0 && val[j] != -INFINITY)
            weight[j] = ::exp(static_cast<double>(val[j]) - top);
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
                    if (first[k] < 0) first[k] = static_cast<int>(j);
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
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x) {
        double m = 0;
        if (ids[j] >= 0 && val[j] != -INFINITY) {
            bool in_q = false;
            for (int k = 0; k < s.count; ++k) if (s.ids[k] == ids[j]) {
                in_q = true;
                if (first[k] == j) m = rad_spechub_positive(pq[k] - s.q[k]);
                break;
            }
            if (!in_q) m = weight[j] / sum;
        }
        mass[j] = m;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    double resid = 0;
    for (int64_t j = 0; j < c.n_cand; ++j) resid += mass[j];
    int32_t out = -1;
    const double u_resid = rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int64_t j = 0; j < c.n_cand; ++j) if (ids[j] >= 0 && val[j] != -INFINITY) {
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

int shape(const RadParam* p, int n, int i, RadOpdDesc* d) {
    const int64_t nc = rad_param_geti(p, n, "n_cand", 0);
    if (nc < 1 || nc > MaxCand) return RAD_E_UNSUPPORTED;
    return rad_mtp_couple_verify_shape(p, n, i, d);
}
int body(const RadArgs* a, const R4dSink& sink) {
    const int64_t nc = a ? rad_param_geti(a->p, a->n_p, "n_cand", 0) : 0;
    if (nc < 1 || nc > MaxCand) return RAD_E_UNSUPPORTED;
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(sink, verify, dim3(static_cast<unsigned>(c.M)), dim3(Threads), 0,
                    c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                    c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
}
R4D_ENTRY(rad_coupled_sampler_probe, body)
const RadKernelInfo* row() {
    static const RadConstraint constraints[] = {RAD_CGE("n_cand", 1), RAD_CLE("n_cand", MaxCand)};
    static const RadKernelInfo info = [] {
        RadKernelInfo r = rad_mtp_couple_row(1, rad_coupled_sampler_probe);
        r.name = "coupled_sampler_probe_verify";
        r.shape = "M,n_cand=1..1024,n_vocab";
        r.priority = 0;
        r.constraints = constraints;
        r.n_constraints = 2;
        r.describe = rad_coupled_sampler_probe_describe;
        r.opd_shape = shape;
        return r;
    }();
    return &info;
}
} // namespace coupled_sampler_probe

extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {RAD_PLUGIN_KERNEL, "coupled_sampler_probe", "0.0.0", "isolated exact MTP verify probe", "gfx1201"};
    return &info;
}
extern "C" int rad_kernel_schema_count() { return 0; } // Existing mine schema, not a new op.
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count() { return 1; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i == 0 ? coupled_sampler_probe::row() : nullptr;
}
