/* hc_up_wmma_wiring_test.cpp -- what the loader sees of the ported M=3/4 HC-read row.
 *
 * The row itself cannot be run here: it is a device kernel and this suite runs on the host backend
 * on every machine. What CAN be checked without a card is everything the engine decides before a
 * launch -- and every one of those answers is a place a port breaks silently:
 *
 *   - the library loads at this tree's ABI version and exports the six symbols the loader resolves;
 *   - it is INERT until RADIANCE_HC_UP_WMMA=1, so an installed library changes no resolution;
 *   - the row it then offers is a device row for `hc_read`, with the constraints that keep it off
 *     every shape libr4d must keep serving;
 *   - its scratch hook answers the same arithmetic the base row does for the band it claims, and
 *     refuses the bands it does not;
 *   - its `hc_read` schema is libr4d's, field for field. This library has to be listed BEFORE
 *     libr4d to win the op, and the first library to declare an op fixes its schema, so a drifted
 *     copy here would re-shape an op the whole tree is using.
 */
#include "rad_test.h"

#include "rad_abi.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Dso {
    void* handle = nullptr;
    uint32_t (*abi)(void) = nullptr;
    const RadPluginInfo* (*info)(void) = nullptr;
    int (*schema_count)(void) = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*count)(void) = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;

    bool open(const char* path) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        abi          = (uint32_t (*)(void))dlsym(handle, "rad_plugin_abi_version");
        info         = (const RadPluginInfo* (*)(void))dlsym(handle, "rad_plugin_info");
        schema_count = (int (*)(void))dlsym(handle, "rad_kernel_schema_count");
        schema_at    = (const RadOpSchema* (*)(int))dlsym(handle, "rad_kernel_schema_at");
        count        = (int (*)(void))dlsym(handle, "rad_kernel_count");
        at           = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        return abi && info && schema_count && schema_at && count && at;
    }
    ~Dso() { if (handle) dlclose(handle); }
};

/* The schema a library declares for `op`, or null. */
const RadOpSchema* find_schema(const Dso& d, const char* op) {
    for (int i = 0; i < d.schema_count(); ++i) {
        const RadOpSchema* s = d.schema_at(i);
        if (s && s->op && std::strcmp(s->op, op) == 0) return s;
    }
    return nullptr;
}

/* The row a library offers for `op`, or null. */
const RadKernelInfo* find_row(const Dso& d, const char* op) {
    for (int i = 0; i < d.count(); ++i) {
        const RadKernelInfo* k = d.at(i);
        if (k && k->op && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

/* A geometry as the resolved RadArgs carries it: only what the row's hooks read. */
struct Geometry {
    RadParam p[6];
    RadArgs  a{};

    Geometry(long long M, long long n, long long hc, long long lowrank) {
        p[0] = { "M",       RAD_P_INT, M,       0, nullptr, 0.0 };
        p[1] = { "n",       RAD_P_INT, n,       0, nullptr, 0.0 };
        p[2] = { "hc",      RAD_P_INT, hc,      0, nullptr, 0.0 };
        p[3] = { "lowrank", RAD_P_INT, lowrank, 0, nullptr, 0.0 };
        p[4] = { "eps",     RAD_P_F64, 0,       0, nullptr, 1e-6 };
        p[5] = { "dtype",   RAD_P_STR, 0,       0, "bf16",   0.0 };
        a.p   = p;
        a.n_p = 6;
    }
};

}  /* namespace */

/* Nothing about the port needs a card, but a card-less tree does not build the device half: the
 * row is then absent by construction rather than by the switch, and the rest still holds. */
TEST(hc_up_wmma_loads_and_is_inert_until_the_switch_is_set) {
    unsetenv("RADIANCE_HC_UP_WMMA");
    Dso d;
    if (!d.open(RAD_HC_UP_WMMA_LIB)) SKIP("hc_up_wmma: " RAD_HC_UP_WMMA_LIB " not built");
    CHECK_EQ(d.abi(), (uint32_t)RAD_ABI_VERSION);
    REQUIRE(d.info() != nullptr);
    CHECK_EQ(d.info()->kind, (uint32_t)RAD_PLUGIN_KERNEL);
    CHECK_EQ(std::string(d.info()->name), std::string("hc_up_wmma"));

    /* Off -- the whole point of an opt-in row. A library on the search path must not change what
     * anything resolves, and an operator who installs it and forgets the switch gets the base row. */
    CHECK_EQ(d.count(), 0);
    CHECK(d.at(0) == nullptr);

    setenv("RADIANCE_HC_UP_WMMA", "0", 1);
    CHECK_EQ(d.count(), 0);

    setenv("RADIANCE_HC_UP_WMMA", "1", 1);
#ifdef RAD_HC_UP_WMMA_WITH_DEVICE
    REQUIRE_EQ(d.count(), 1);
    const RadKernelInfo* k = d.at(0);
    REQUIRE(k != nullptr);
    CHECK_EQ(std::string(k->name), std::string("hc_read_up_wmma_m3_m4"));
    CHECK_EQ(std::string(k->op), std::string("hc_read"));
    CHECK_EQ(k->domain, (int)RAD_DOMAIN_DEVICE);
    CHECK(k->launch != nullptr);
    CHECK(k->scratch != nullptr);
    /* The layouts are libr4d's, pulled in by the port, not a reimplementation: the base row's
     * stored form is the one this row must read and produce. */
    CHECK(k->layout != nullptr);
    CHECK(k->relayout != nullptr);
    CHECK(k->unrelayout != nullptr);
    CHECK(k->opd_shape != nullptr);
#else
    CHECK_EQ(d.count(), 0);
#endif
    unsetenv("RADIANCE_HC_UP_WMMA");
}

TEST(hc_up_wmma_row_only_claims_the_shape_it_was_ported_for) {
    unsetenv("RADIANCE_HC_UP_WMMA");
    setenv("RADIANCE_HC_UP_WMMA", "1", 1);
    Dso d;
    if (!d.open(RAD_HC_UP_WMMA_LIB)) SKIP("hc_up_wmma: " RAD_HC_UP_WMMA_LIB " not built");
    const RadKernelInfo* k = find_row(d, "hc_read");
    if (!k) SKIP("hc_up_wmma: built without a device half");

    /* Every constraint the row carries must be a real one: libr4d serves M=1..4 at any of its
     * widths, and this row must never take a band it did not port. */
    auto has_eq = [k](const char* key, long long v) {
        for (int i = 0; i < k->n_constraints; ++i)
            if (std::strcmp(k->constraints[i].key, key) == 0)
                return k->constraints[i].op == RAD_C_EQ && k->constraints[i].ival == v;
        return false;
    };
    CHECK(has_eq("n", 2560));
    CHECK(has_eq("hc", 4));
    CHECK(has_eq("lowrank", 320));
    CHECK(has_eq("inject", 1));
    CHECK(has_eq("group", 128));
    bool m_ge = false, m_le = false;
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint& c = k->constraints[i];
        if (std::strcmp(c.key, "M") == 0 && c.op == RAD_C_GE && c.ival == 3) m_ge = true;
        if (std::strcmp(c.key, "M") == 0 && c.op == RAD_C_LE && c.ival == 4) m_le = true;
    }
    CHECK(m_ge);
    CHECK(m_le);
    unsetenv("RADIANCE_HC_UP_WMMA");
}

TEST(hc_up_wmma_scratch_is_the_base_rows_arithmetic_and_refuses_other_bands) {
    unsetenv("RADIANCE_HC_UP_WMMA");
    setenv("RADIANCE_HC_UP_WMMA", "1", 1);
    Dso d;
    if (!d.open(RAD_HC_UP_WMMA_LIB)) SKIP("hc_up_wmma: " RAD_HC_UP_WMMA_LIB " not built");
    const RadKernelInfo* k = find_row(d, "hc_read");
    if (!k || !k->scratch) SKIP("hc_up_wmma: built without a device half");

    /* `d` [M, lowrank] bf16 and the per-stream norm scales [M, kHcMax] f32 -- the base row's
     * buffer, because the down+inject half IS the base row's. kHcMax is 8 in libr4d. */
    Geometry g3(3, 2560, 4, 320), g4(4, 2560, 4, 320);
    CHECK_EQ(k->scratch(&g3.a), (int64_t)(3 * (320 * 2 + 8 * 4)));
    CHECK_EQ(k->scratch(&g4.a), (int64_t)(4 * (320 * 2 + 8 * 4)));

    /* Bands it does not claim answer a negative code, never a plausible size: a scratch of 0 would
     * launch the base-sized buffers into this kernel. */
    Geometry m5(5, 2560, 4, 320), wide(3, 4096, 4, 320), hc8(3, 2560, 8, 320),
             lr640(3, 2560, 4, 640), zero(0, 2560, 4, 320);
    CHECK(k->scratch(&m5.a) < 0);
    CHECK(k->scratch(&wide.a) < 0);
    CHECK(k->scratch(&hc8.a) < 0);
    CHECK(k->scratch(&lr640.a) < 0);
    CHECK(k->scratch(&zero.a) < 0);
    CHECK(k->scratch(nullptr) < 0);
    unsetenv("RADIANCE_HC_UP_WMMA");
}

/* The drift check. This library is useless unless it is listed before libr4d, and it is only
 * listed first because it declares `hc_read` itself; then its declaration is the one every caller
 * of `hc_read` is validated against. Same op, same parameters in the same order, same operand
 * slots -- or the engine would hand `x` a buffer meant for `q`. */
TEST(hc_up_wmma_declares_hc_read_exactly_as_libr4d_does) {
    Dso mine;
    if (!mine.open(RAD_HC_UP_WMMA_LIB)) SKIP("hc_up_wmma: " RAD_HC_UP_WMMA_LIB " not built");
    Dso base;
    if (!base.open(RAD_R4D_LIB)) SKIP("libr4d: " RAD_R4D_LIB " not built");

    const RadOpSchema* a = find_schema(mine, "hc_read");
    const RadOpSchema* b = find_schema(base, "hc_read");
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    REQUIRE_EQ(a->n_params, b->n_params);
    for (int i = 0; i < a->n_params; ++i) {
        CHECK_EQ(std::string(a->params[i].key), std::string(b->params[i].key));
        CHECK_EQ(a->params[i].type, b->params[i].type);
        CHECK_EQ(a->params[i].required, b->params[i].required);
        CHECK_EQ(a->params[i].role, b->params[i].role);
    }
    REQUIRE_EQ(a->n_operands, b->n_operands);
    for (int i = 0; i < a->n_operands; ++i) {
        CHECK_EQ(std::string(a->operands[i].name), std::string(b->operands[i].name));
        CHECK_EQ(a->operands[i].role, b->operands[i].role);
        CHECK_EQ(a->operands[i].optional, b->operands[i].optional);
    }
}

RAD_TEST_MAIN()
