/* spechub_wiring_test.cpp -- what the loader and the reference gate see of the spechub plugin.
 *
 * The three spechub_*_test binaries call the abi helpers directly; none of them loads the library
 * or asks how its rows are classified, so none of them could see the bug this test exists for: a
 * plugin that declared itself a reference library had EVERY row marked reference, including its
 * two device rows, and the engine's reference gate then refused to serve `mtp_couple_propose` /
 * `mtp_couple_verify` at bringup (RAD_E_NOKERNEL) unless --debug-accept-reference-kernels was
 * passed. The switch alone could not start a coupled run.
 *
 * The check is on the real DSO, loaded the way the loader loads it (dlopen, the exported symbols,
 * nothing else): the classification the gate consumes is exactly `RadKernelInfo::domain` and the
 * reference answer, so that is what is asserted here. A host build (no RAD_SPECHUB_WITH_DEVICE)
 * has only the oracle rows and must keep being classified as reference -- that is the half of the
 * rule that must NOT be relaxed, and it is asserted too.
 */
#include "rad_test.h"

#include "rad_abi.h"

#include <dlfcn.h>

#include <map>
#include <string>
#include <vector>

namespace {

struct Plugin {
    void* handle = nullptr;
    const RadPluginInfo* (*info)() = nullptr;
    int (*reference)() = nullptr;
    int (*kern_ref)(int) = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;

    bool open(const char* path) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        info      = (const RadPluginInfo* (*)())dlsym(handle, "rad_plugin_info");
        reference = (int (*)())dlsym(handle, "rad_plugin_reference");
        kern_ref  = (int (*)(int))dlsym(handle, "rad_kernel_reference");
        count     = (int (*)())dlsym(handle, "rad_kernel_count");
        at        = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        return info && reference && count && at;
    }
    ~Plugin() { if (handle) dlclose(handle); }
};

/* The rule rad_reference_ops applies, over the rows as the loader would classify them: an op is on
 * the reference when no row for it is a non-reference row. */
std::vector<std::string> reference_only_ops(const Plugin& p, const std::vector<int>& is_ref) {
    std::map<std::string, bool> has_real;
    for (int i = 0; i < p.count(); ++i) {
        const RadKernelInfo* k = p.at(i);
        if (!k) continue;
        bool& real = has_real[k->op];
        real = real || !is_ref[(size_t)i];
    }
    std::vector<std::string> out;
    for (const auto& kv : has_real)
        if (!kv.second) out.push_back(kv.first);
    return out;
}

}  /* namespace */

TEST(spechub_wiring_the_device_rows_are_production_and_the_oracle_rows_are_reference) {
    Plugin p;
    REQUIRE(p.open(RAD_SPECHUB_LIB));

    CHECK_EQ(p.info()->kind, (int)RAD_PLUGIN_KERNEL);
    CHECK_EQ(std::string(p.info()->name), std::string("spechub"));

    /* The per-row answer must exist: without it the loader falls back to the library answer, which
     * is what mis-classified the device rows. */
    REQUIRE(p.kern_ref != nullptr);
    /* And the library as a whole is NOT a reference library -- otherwise the loader sorts it below
     * the kernel libraries it serves beside and every row inherits the flag. */
    CHECK_EQ(p.reference(), 0);

    std::vector<int> is_ref;
    int dev = 0, host = 0;
    for (int i = 0; i < p.count(); ++i) {
        const RadKernelInfo* k = p.at(i);
        REQUIRE(k != nullptr);
        const int r = p.kern_ref(i);
        is_ref.push_back(r);
        if (k->domain == RAD_DOMAIN_DEVICE) {
            ++dev;
            /* A device row is a kernel, never a reference: this is the line that decides whether a
             * coupled run may start at all. */
            CHECK_EQ(r, 0);
            CHECK(std::string(k->name).rfind("ref_", 0) != 0);
        } else {
            ++host;
            CHECK_EQ(k->domain, (int)RAD_DOMAIN_HOST);
            /* The host oracle beside it keeps its reference status. */
            CHECK_EQ(r, 1);
            CHECK(std::string(k->name).rfind("ref_", 0) == 0);
        }
    }
    CHECK_EQ(dev + host, p.count());
#ifdef RAD_SPECHUB_WITH_DEVICE
    CHECK_EQ(dev, 2);
    CHECK_EQ(host, 2);
#else
    CHECK_EQ(dev, 0);
    CHECK_EQ(host, 2);
#endif

    /* Both ops must have at least one production row on a device build. This is the property the
     * gate reads: with it, the coupling is servable without --debug-accept-reference-kernels. */
    const std::vector<std::string> refonly = reference_only_ops(p, is_ref);
#ifdef RAD_SPECHUB_WITH_DEVICE
    CHECK_EQ(refonly.size(), (size_t)0);
#else
    /* No card: both ops are oracle-only, and the gate must still see them as such. */
    CHECK_EQ(refonly.size(), (size_t)2);
#endif
}

RAD_TEST_MAIN()
