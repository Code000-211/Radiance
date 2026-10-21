// Isolated MTP PROPOSAL experiment; mine/spechub.hip stays the production row and
// coupled_sampler_*.hip stay the verify experiments.
//
// The proposal row offers exactly one kind of genuinely independent per-row work: the eight
// FP64 exponentials exp(kv[j] - top) of one shortlist and the eight FP64 q[j] = w[j] / sum
// divisions. Each is a scalar function of that slot's own score and of a value every slot
// already holds, so evaluating them on eight lanes returns the same bits as evaluating them
// one after another. Nothing else moves:
//   * the validation order, the duplicate test, the maximum and the compaction of the kept
//     entries stay in one lane, in ascending slot order;
//   * the running sum `sum += w[j]` stays in that lane, in the original ascending order;
//   * the `q > 0` underflow drop, the stored ids/q packing, the 53-bit draft uniform
//     (seed ^ (KEY_DRAFT + round), base) and the inverse-CDF pick stay in that lane;
//   * greedy (mode 0) rows, malformed shortlists, a stale card and an empty support keep
//     column 0 and write status 0 / -1 exactly as before.
// No reordering, no reassociation and no new quantization: the ABI, the 152-byte
// RadMtpCoupleState layout and every RNG key are the production ones.
//
// Geometry: one group of eight lanes owns one row, four rows share a 32-thread block. A row
// still costs one workgroup slot rather than one lane, which is what makes the eight-way
// exponentials usable instead of paying 24 idle lanes per row. Per-row scratch is shared
// memory; the wire row and every heap allocation keep their production width.
//
// Build without fast math, finite-math-only or FP contraction; the serial FP64 order is the ABI.
#ifdef __FAST_MATH__
#error "coupled propose requires strict FP64 arithmetic"
#endif
#include <hip/hip_runtime.h>
#include "../../../libr4d/r4d_emit.h"
// Reuse the production schema, geometry, ABI validator and RNG helpers without rebuilding mine.so.
#include "../../../mine/spechub.cpp"

namespace coupled_propose_probe {
constexpr int Slots = 8;  // `mtp_couple_propose` freezes the shortlist at eight columns
constexpr int Group = 8;  // one lane per shortlist slot
constexpr int Rows = 4;   // rows per block
constexpr int Threads = Rows * Group;

__global__ void propose(RadMtpCoupleProposeCall c) {
    const int group = static_cast<int>(threadIdx.x) / Group;
    const int lane = static_cast<int>(threadIdx.x) % Group;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * Rows + group;
    const bool live = row < c.M;
    int32_t* idx = live ? c.idx + row * c.ld_idx : nullptr;
    const float* val = live ? c.val + row * c.ld_val : nullptr;

    // Per-row scratch, sized by the frozen eight-slot shortlist. Nothing is state, nothing is
    // heap, and no wire row grows.
    __shared__ int32_t kid[Rows][Slots];
    __shared__ float kv[Rows][Slots];
    __shared__ double w[Rows][Slots];
    __shared__ double qv[Rows][Slots];
    __shared__ RadMtpCoupleState st[Rows];
    __shared__ int ready[Rows], kept[Rows];
    __shared__ float top[Rows];
    __shared__ double total[Rows];

    // Phase 1, one lane: the production validation, duplicate and maximum order, verbatim.
    if (live && lane == 0) {
        RadMtpCoupleState& s = st[group];
        s = RadMtpCoupleState{};
        s.draw = *reinterpret_cast<const RadMtpCoupleDraw*>(c.draw + row * c.ld_draw);
        s.round = c.round;
        s.x = idx[0];
        for (int i = 0; i < Slots; ++i) s.ids[i] = -1;
        int n = 0, bad = 0;
        if (s.draw.mode == 1) {
            s.status = -1;
            if (s.draw.card != c.card[row]) bad = 1;
            for (int i = 0; i < Slots && !bad; ++i) {
                if (idx[i] < 0) continue;
                if (idx[i] >= c.vocab || val[i] != val[i] || val[i] == INFINITY) { bad = 1; break; }
                if (val[i] == -INFINITY) continue;
                for (int j = 0; j < n; ++j) if (kid[group][j] == idx[i]) { bad = 1; break; }
                if (bad) break;
                kid[group][n] = idx[i];
                kv[group][n++] = val[i];
            }
            if (!bad && n == 0) bad = 1;
        }
        kept[group] = n;
        ready[group] = (s.draw.mode == 1 && !bad) ? 1 : 0;
        if (ready[group]) {
            float mx = kv[group][0];
            for (int j = 1; j < n; ++j) if (kv[group][j] > mx) mx = kv[group][j];
            top[group] = mx;
        }
    }
    __syncthreads();
    // Phase 2, one lane per kept slot: the same FP64 exponential the serial row evaluates, on
    // the slot's own score. A scalar exp of one argument does not depend on who evaluates it.
    if (live && ready[group] && lane < kept[group])
        w[group][lane] = ::exp(static_cast<double>(kv[group][lane]) - top[group]);
    __syncthreads();
    // Phase 3, one lane: the ascending sum, unchanged order and unchanged addend count.
    if (live && lane == 0) {
        double t = 0;
        if (ready[group]) for (int j = 0; j < kept[group]; ++j) t += w[group][j];
        total[group] = t;
    }
    __syncthreads();
    // Phase 4, one lane per kept slot: the same w[j] / sum divide the serial row issues.
    if (live && ready[group] && lane < kept[group]) qv[group][lane] = w[group][lane] / total[group];
    __syncthreads();
    // Phase 5, one lane: the underflow drop, the pack, the 53-bit draw, the inverse-CDF pick and
    // the state store, all in the production ascending-slot order.
    if (live && lane == 0) {
        RadMtpCoupleState& s = st[group];
        if (ready[group]) {
            int count = 0;
            for (int j = 0; j < kept[group]; ++j) {
                const double q = qv[group][j];
                if (q > 0) { s.ids[count] = kid[group][j]; s.q[count++] = q; }
            }
            s.count = count;
            if (count > 0) {
                const double u = rad_spechub_uniform(s.draw.seed, s.draw.base,
                                    RAD_MTP_COUPLE_KEY_DRAFT + static_cast<uint64_t>(c.round));
                s.x = s.ids[rad_mtp_couple_pick(s, u)];
                s.status = 1;
                idx[0] = s.x;
            }
        }
        *reinterpret_cast<RadMtpCoupleState*>(c.state + (c.round - 1) * c.st_round +
                                              row * c.st_row) = s;
    }
}

int body(const RadArgs* a, const R4dSink& sink) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    const auto grid = dim3(static_cast<unsigned>((c.M + Rows - 1) / Rows));
    return R4D_EMIT(sink, propose, grid, dim3(Threads), 0, c);
}
R4D_ENTRY(rad_coupled_propose_probe, body)
const RadKernelInfo* row() {
    static const RadKernelInfo info = [] {
        RadKernelInfo r = rad_mtp_couple_row(0, rad_coupled_propose_probe);
        r.name = "coupled_propose_cooperative_probe";
        r.shape = "M,n_vocab,round,max_rounds<=8; shortlist=8; 4 rows/block x 8 lanes";
        r.priority = 0;
        r.describe = rad_coupled_propose_probe_describe;
        return r;
    }();
    return &info;
}
}  // namespace coupled_propose_probe

extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {RAD_PLUGIN_KERNEL, "coupled_propose_probe", "0.0.0",
                                       "isolated cooperative MTP proposal probe", "gfx1201"};
    return &info;
}
extern "C" int rad_kernel_schema_count() { return 0; }  // Existing mine schema, not a new op.
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count() { return 1; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i == 0 ? coupled_propose_probe::row() : nullptr;
}
