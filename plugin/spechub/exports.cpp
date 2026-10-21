/* plugin/spechub/exports.cpp -- the plugin's export block.
 *
 * The whole kernel-plugin set (schema count/at, kernel count/at, ABI version, info) and nothing else.
 * The rows themselves, their schemas and their argument readers are in spechub.cpp; the launches are
 * spechub.hip (device) and ref_spechub.cpp (host).
 *
 * `rad_plugin_reference` is 1: this library's rows have a host reference beside them and are checked
 * against it, so the loader must rank it as a reference rather than let it win a selection by load
 * order. The flag, not the loader, is what puts a coupled row into serving.
 */
#include "spechub.h"

#include "../../libr4d/r4d_emit.h"   /* only for the three globals it declares */

/* THE DISPATCH LOG IS THIS DSO'S OWN. r4d_emit.h routes every launch through this recorder so a test
 * can name the functions it dispatched, but the loader maps plugins RTLD_LOCAL: a plugin cannot
 * import libr4d's copy, and an unresolved symbol refuses the whole library at load. Defining them
 * here gives r4d_emit.h exactly what it declares and leaves the recorder inert -- nothing in this
 * plugin sets the log. It is a tool's, never consulted on a serving path. */
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n   = 0;

namespace {
const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "spechub", "0.1.0",
    "Linear-MTP maximal coupling: a Q draw per round and the coupled target verify", "gfx1201",
};
}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_plugin_reference(void) { return 1; }

extern "C" int rad_kernel_schema_count(void) { return rad_spechub_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return rad_spechub_schema_at(i); }

/* Two device rows and the two host references beside them, in that order. A card-less machine loads
 * the same library and resolves only the reference rows. */
extern "C" int rad_kernel_count(void) {
#ifdef RAD_SPECHUB_WITH_DEVICE
    return 4;
#else
    return 2;
#endif
}

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    const int op = i & 1;
#ifdef RAD_SPECHUB_WITH_DEVICE
    if (i == 0 || i == 1) {
        static const RadKernelInfo r[2] = {
            rad_spechub_row(0, rad_mine_mtp_couple_propose_launch),
            rad_spechub_row(1, rad_mine_mtp_couple_verify_launch),
        };
        return &r[i];
    }
#endif
    static const RadKernelInfo ref[2] = {
        [] {
            RadKernelInfo x = rad_spechub_row(0, rad_ref_mtp_couple_propose);
            x.name = "ref_mtp_couple_propose";
            x.domain = RAD_DOMAIN_HOST;
            x.priority = 0;
            return x;
        }(),
        [] {
            RadKernelInfo x = rad_spechub_row(1, rad_ref_mtp_couple_verify);
            x.name = "ref_mtp_couple_verify";
            x.domain = RAD_DOMAIN_HOST;
            x.priority = 0;
            return x;
        }(),
    };
#ifdef RAD_SPECHUB_WITH_DEVICE
    if (i < 2 || i > 3) return nullptr;
    return &ref[op];
#else
    if (i < 0 || i > 1) return nullptr;
    return &ref[i];
#endif
}
