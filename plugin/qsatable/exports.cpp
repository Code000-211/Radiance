/* plugin/qsatable/exports.cpp -- the plugin's export block.
 *
 * One op, `qsa_table`, with one device row and the host reference beside it. Reference is PER ROW:
 * `rad_plugin_reference` answers for the whole library, so returning 1 would mark the device row as
 * a reference too and the engine's reference gate would refuse to serve it. The device row reports 0
 * and the host row 1 through `rad_kernel_reference`.
 *
 * Nothing outside the plugin decides whether they are used: the architecture issues them only with
 * its own opt-in switch set (arch/common/rad_qsa.h), and the loader never reads that switch.
 */
#include "qsaattn.h"
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

/* The reference rows' launches take the stream slot the device rows take, which the host ignores. */
namespace {
int ref_table_launch(const RadArgs* a, void*) { return rad_ref_qsa_table(a); }
int ref_attn_launch(const RadArgs* a, void*)  { return rad_ref_qsa_attn(a); }
}  /* namespace */

extern "C" int rad_kernel_reference(int i) {
#ifdef RAD_QSATABLE_WITH_DEVICE
    return (i == 0 || i == 2) ? 0 : 1;
#else
    /* Row i is the i-th reference: `ref_qsa_table` then `ref_qsa_attn`, the same two rows the
     * device build puts at 1 and 3. */
    return i == 0 || i == 1;
#endif
}

extern "C" int rad_kernel_schema_count(void) {
    return rad_qsatable_schema_count() + rad_qsaattn_schema_count();
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    const int n = rad_qsatable_schema_count();
    return i < n ? rad_qsatable_schema_at(i) : rad_qsaattn_schema_at(i - n);
}

/* Two ops, each with a device row and a host reference: qsa_table 0/1, qsa_attn 2/3 (host build:
 * references only, in the same order). */
extern "C" int rad_kernel_count(void) {
#ifdef RAD_QSATABLE_WITH_DEVICE
    return 4;
#else
    return 2;
#endif
}

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
#ifdef RAD_QSATABLE_WITH_DEVICE
    if (i == 0) {
        static const RadKernelInfo dev = rad_qsatable_row(rad_qsa_table_launch);
        return &dev;
    }
    if (i == 2) {
        static const RadKernelInfo dev = rad_qsaattn_row(rad_qsa_attn_launch);
        return &dev;
    }
    if (i != 1 && i != 3) return nullptr;
#else
    if (i != 0 && i != 1) return nullptr;
#endif
#ifdef RAD_QSATABLE_WITH_DEVICE
    const bool attn = i >= 2;
#else
    const bool attn = i == 1;
#endif
    static const RadKernelInfo ref_tab = [] {
        RadKernelInfo x = rad_qsatable_row(ref_table_launch);
        x.name = "ref_qsa_table";
        x.domain = RAD_DOMAIN_HOST;
        x.priority = 0;
        return x;
    }();
    static const RadKernelInfo ref_attn = [] {
        RadKernelInfo x = rad_qsaattn_row(ref_attn_launch);
        x.name = "ref_qsa_attn";
        x.domain = RAD_DOMAIN_HOST;
        x.priority = 0;
        return x;
    }();
    return attn ? &ref_attn : &ref_tab;
}
