/* plugin/qsatable/exports.cpp -- the plugin's export block.
 *
 * One op, `qsa_table`, with one device row and the host reference beside it. Reference is PER ROW:
 * `rad_plugin_reference` answers for the whole library, so returning 1 would mark the device row as
 * a reference too and the engine's reference gate would refuse to serve it. The device row reports 0
 * and the host row 1 through `rad_kernel_reference`.
 *
 * Nothing outside the plugin decides whether it is used: the architecture issues `qsa_table` only
 * with its own opt-in switch set (arch/common/rad_qsa.h), and the loader never reads that switch.
 */
#include "qsatable.h"

/* Only for the three dispatch-log globals it declares; a host build has no HIP headers. */
#ifdef RAD_QSATABLE_WITH_DEVICE
#include "../../libr4d/r4d_emit.h"
#endif

/* THE DISPATCH LOG IS THIS DSO'S OWN. The loader maps plugins RTLD_LOCAL, so libr4d's copy is not
 * importable and an unresolved symbol refuses the whole library at load. Defining them here leaves
 * the recorder inert: nothing in this plugin sets the log. */
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n   = 0;

namespace {
const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "qsatable", "0.0.1",
    "The QSA block table a query tile shares: one union in block order, one membership mask per row",
    "gfx1201",
};
}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
/* Mixed library: 0, and the per-row answer below does the classifying. */
extern "C" int rad_plugin_reference(void) { return 0; }

extern "C" int rad_kernel_reference(int i) {
#ifdef RAD_QSATABLE_WITH_DEVICE
    return i == 0 ? 0 : 1;
#else
    (void)i;
    return 1;
#endif
}

extern "C" int rad_kernel_schema_count(void) { return rad_qsatable_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return rad_qsatable_schema_at(i); }

extern "C" int rad_kernel_count(void) {
#ifdef RAD_QSATABLE_WITH_DEVICE
    return 2;
#else
    return 1;
#endif
}

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
#ifdef RAD_QSATABLE_WITH_DEVICE
    if (i == 0) {
        static const RadKernelInfo dev = rad_qsatable_row(rad_qsa_table_launch);
        return &dev;
    }
    if (i != 1) return nullptr;
#else
    if (i != 0) return nullptr;
#endif
    static const RadKernelInfo ref = [] {
        RadKernelInfo x = rad_qsatable_row(rad_ref_qsa_table);
        x.name = "ref_qsa_table";
        x.domain = RAD_DOMAIN_HOST;
        x.priority = 0;
        return x;
    }();
    return &ref;
}
