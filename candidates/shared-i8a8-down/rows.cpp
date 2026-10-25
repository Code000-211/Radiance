/* rows.cpp -- the candidate's whole declaration: one plugin identity, one op schema, one row.
 *
 * The schema is libr4d's own `gemm_nt_q` list, position for position and requiredness for
 * requiredness (libr4d/r4d_rows.cpp, pGemmNtQ / oGemmNtQ). The loader compares a re-declared op
 * against the schema of the plugin that fixed it and refuses a plugin that disagrees
 * (core/plugin/loader.cpp, schema_same): an out-of-tree override has to mean the same call. The
 * doc string and the operand names are diagnostics only and may differ.
 *
 * The row is absent unless RADIANCE_SHARED_I8A8_DOWN=1, so a build with the variable unset
 * publishes no new row and no new bucket boundary.
 */
#include "rad_abi.h"

#include "shared_i8a8_down.h"

#include <cstdlib>
#include <cstring>

namespace {

/* ================================================================== the op schema
 *
 * `M` is a value here rather than a range: a kernel plugin declares parameters, and the
 * architecture declares the range on the op (arch/common/rad_fp8.h). `pb1`/`pb2` are ParoQuant's
 * row parts, `split_n` the stored width a narrow row takes its K split from; this row accepts the
 * parameter and refuses a partition.
 */
const RadParamSpec kGemmNtQParams[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "N",       RAD_P_INT, RAD_REQUIRED },
    { "K",       RAD_P_INT, RAD_REQUIRED },
    { "group",   RAD_P_INT, RAD_REQUIRED },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
    { "pb1",     RAD_P_INT, RAD_OPTIONAL },
    { "pb2",     RAD_P_INT, RAD_OPTIONAL },
    { "split_n", RAD_P_INT, RAD_OPTIONAL },
};
const RadOperandSpec kGemmNtQOperands[] = {
    { "a",       RAD_OPD_IN,     0 },
    { "a_scale", RAD_OPD_IN,     1 },
    { "b",       RAD_OPD_WEIGHT, 0 },
    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "y",       RAD_OPD_OUT,    0 },
    { "a_sum",   RAD_OPD_IN,     1 },
    { "b_ref",   RAD_OPD_WEIGHT, 1 },
};

const RadOpSchema kSchemas[] = {
    { "gemm_nt_q", kGemmNtQParams, (int)(sizeof kGemmNtQParams / sizeof kGemmNtQParams[0]),
      kGemmNtQOperands, (int)(sizeof kGemmNtQOperands / sizeof kGemmNtQOperands[0]),
      "C[M,N] = A[M,K] @ dequant(Bq)[N,K]^T. This declaration exists so the candidate's row has a "
      "schema of its own; it is libr4d's, and an installed libr4d fixes the same one first." },
};

/* ================================================================== the row's bounds
 *
 * N and K are equalities, not bands: the arithmetic the row performs is written for one pair.
 * The M bound is the measured band (the .hip states the numbers), and it is an upper bound, not a
 * promise about one step's row count -- the issue path narrows the operands and the launch reads
 * the rows off them. It also cuts the bucket boundary: a four-token verify step lands in the band
 * above and goes to libr4d's wider row.
 */
const RadConstraint kBounds[] = {
    RAD_CGE("M", 1), RAD_CLE("M", 3), RAD_CEQ("N", 2560), RAD_CEQ("K", 384),
    RAD_CEQ("group", 128), RAD_CIN("dtype", "i8a8"),
};

const RadKernelInfo kRow = [] {
    RadKernelInfo r{};
    r.name    = "shared_i8a8_down_m1_m3";
    r.op      = "gemm_nt_q";
    r.family  = "gemm";
    r.computes = "signed i8 packed dot, int32 per 128 K, FP32 scale and fold, BF16 output";
    r.shape    = "M=1..3, N=2560, K=384, group=128; tight rows, frag64 weights";
    r.dtypes   = "i8 A and blocked W; f32 A scales; bf16 W scales and Y";
    r.domain   = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kBounds;
    r.n_constraints = (int)(sizeof kBounds / sizeof kBounds[0]);
    r.launch  = rad_shared_i8a8_down;
    r.opd_shape = rad_shared_i8a8_down_shape;
    /* libr4d's own int8 hooks, compiled into this DSO; see the layout note in the .hip. */
    r.layout     = r4d_rad_layout_i8_block;
    r.relayout   = r4d_rad_relayout_i8_block;
    r.unrelayout = r4d_rad_unrelayout_i8_block;
    /* No scratch: one launch, nothing staged. */
    return r;
}();

/* Default off. A build with the variable unset publishes exactly the rows it published before. */
bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("RADIANCE_SHARED_I8A8_DOWN");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}

const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "shared_i8a8_down", "0.1.0",
    "Shared expert down projection (gemm_nt_q i8a8) at the verify width; opt-in", "gfx1201",
};

}  // namespace

extern "C" SI8_EXPORT uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" SI8_EXPORT const RadPluginInfo* rad_plugin_info() { return &kInfo; }

extern "C" SI8_EXPORT int rad_kernel_schema_count() {
    return (int)(sizeof kSchemas / sizeof kSchemas[0]);
}
extern "C" SI8_EXPORT const RadOpSchema* rad_kernel_schema_at(int i) {
    return i >= 0 && i < rad_kernel_schema_count() ? &kSchemas[i] : nullptr;
}

extern "C" SI8_EXPORT int rad_kernel_count() { return enabled() ? 1 : 0; }
extern "C" SI8_EXPORT const RadKernelInfo* rad_kernel_at(int i) {
    return i == 0 && rad_kernel_count() ? &kRow : nullptr;
}

/* The row touches nothing but the five operands it is handed: no scratch, no counter, no state
 * kept between launches. Independent launches may therefore overlap (rad_abi.h,
 * rad_kernel_concurrent). */
extern "C" SI8_EXPORT int rad_kernel_concurrent(int i) { return i == 0 && enabled() ? 1 : 0; }
