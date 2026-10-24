/* plugin/hc_up_wmma/exports.cpp -- the ported M=3/4 HC-read row's host half.
 *
 * The whole kernel-plugin set (schema count/at, kernel count/at, ABI version, info) and nothing
 * else. The row, its device kernels and its launches are hc_up_wmma.hip; this file carries the op
 * schema the row answers for and the switch that keeps the library inert until asked.
 *
 * WHY THE SCHEMA IS RESTATED. Op selection is by hierarchy, not by priority: the FIRST library on
 * the search path that carries any row for an op fixes that op's schema, and later libraries serve
 * the rows it leaves unserved (docs/PLUGIN.md). This library only wins the M=3..4 band at all when
 * it comes first, so it must declare `hc_read` itself. The declaration
 * must match libr4d's row for row -- tests/hc_up_wmma_wiring_test.cpp compares the two libraries'
 * schemas field for field.
 *
 * A MISMATCH REFUSES THE LATER PLUGIN, WHICH IS libr4d. The loader compares schemas as it stages
 * libraries in hierarchy order (core/plugin/loader.cpp) and drops the one that disagrees, naming
 * both. With this library first -- the default path order, and the documented way to enable the
 * row -- a drifted port therefore refuses libr4d and every device row with it. That is loud but it
 * hits the wrong library, and it happens with or without RADIANCE_HC_UP_WMMA: the schema is
 * declared either way.
 *
 * OFF BY DEFAULT. `rad_kernel_count` is 0 unless RADIANCE_HC_UP_WMMA=1, so installing the library
 * changes nothing about how `hc_read` resolves.
 */
#include "hc_up_wmma.h"

/* The dispatch log is this DSO's own: the loader maps plugins RTLD_LOCAL, so libr4d's copy is not
 * importable and an unresolved symbol would refuse the library at load. Nothing here sets it.
 * Declared here rather than through libr4d/r4d_emit.h so this file needs no HIP header: it is also
 * the half a card-less tree compiles (plugin/hc_up_wmma/CMakeLists.txt).
 *
 * HIDDEN. These are libr4d's names, defined here so the include in hc_up_wmma.hip links; leaving
 * them dynamic would let a loader that maps this library RTLD_GLOBAL interpose libr4d's copies.
 * The project sets no -fvisibility=hidden, so the pragma is the only thing that keeps them in. */
#pragma GCC visibility push(hidden)
extern const void** r4d_dispatch_log;
extern int          r4d_dispatch_cap;
extern int          r4d_dispatch_n;
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n   = 0;
#pragma GCC visibility pop

namespace {

/* libr4d's p/oHcRead (libr4d/r4d_rows.cpp), restated: the same parameters in the same order, the
 * same operand slots. A port that added or reordered an operand would send `x` into `q`. */
#define PARAM(k, t, req) { (k), (t), (req), 0 }
const RadParamSpec kHcReadParams[] = {
    PARAM("M", RAD_P_INT, RAD_REQUIRED),    PARAM("n", RAD_P_INT, RAD_REQUIRED),
    PARAM("hc", RAD_P_INT, RAD_REQUIRED),   PARAM("lowrank", RAD_P_INT, RAD_REQUIRED),
    PARAM("eps", RAD_P_F64, RAD_REQUIRED),  PARAM("dtype", RAD_P_STR, RAD_REQUIRED),
    PARAM("wadd", RAD_P_F64, RAD_OPTIONAL), PARAM("inject", RAD_P_INT, RAD_OPTIONAL),
    PARAM("group", RAD_P_INT, RAD_OPTIONAL), PARAM("rotate", RAD_P_INT, RAD_OPTIONAL),
    PARAM("mix", RAD_P_STR, RAD_OPTIONAL),
};
#undef PARAM

const RadOperandSpec kHcReadOperands[] = {
    { "h", RAD_OPD_IN, 0 },      { "w", RAD_OPD_WEIGHT, 0 },
    { "mix_down", RAD_OPD_WEIGHT, 0 }, { "mix_up", RAD_OPD_WEIGHT, 0 },
    { "inject_w", RAD_OPD_WEIGHT, 1 }, { "x", RAD_OPD_OUT, 0 },
    { "inj", RAD_OPD_OUT, 1 },   { "q", RAD_OPD_OUT, 1 },
    { "scale", RAD_OPD_OUT, 1 }, { "rq", RAD_OPD_OUT, 1 },
    { "rscale", RAD_OPD_OUT, 1 },
};

const RadOpSchema kSchemas[] = {
    { "hc_read", kHcReadParams, (int)(sizeof(kHcReadParams) / sizeof(kHcReadParams[0])),
      kHcReadOperands, (int)(sizeof(kHcReadOperands) / sizeof(kHcReadOperands[0])),
      "the read half of a gated residual: a grouped rmsnorm, a low-rank sigmoid gate, and "
      "the mean of the gated streams -- plus the per-branch write gains and, optionally, "
      "the block-scaled fp8 form of the block input and its Hadamard-rotated twin." },
};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "hc_up_wmma", "0.1.0",
    "Opt-in HC-read row for n=2560/hc=4/lowrank=320/M=3..4: the up-dot reduction on WMMA",
    "gfx1201",
};

}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

extern "C" int rad_kernel_schema_count(void) {
    return (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return i >= 0 && i < rad_kernel_schema_count() ? &kSchemas[i] : nullptr;
}

/* A card-less build has the schema and no row: the library loads, declares `hc_read` exactly as
 * the device build does, and serves nothing, so no shape reaches a kernel that was not compiled. */
extern "C" int rad_kernel_count(void) {
#ifdef RAD_HC_UP_WMMA_WITH_DEVICE
    return rad_hc_up_wmma_row_count();
#else
    return 0;
#endif
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
#ifdef RAD_HC_UP_WMMA_WITH_DEVICE
    return rad_hc_up_wmma_row_at(i);
#else
    (void)i;
    return nullptr;
#endif
}
