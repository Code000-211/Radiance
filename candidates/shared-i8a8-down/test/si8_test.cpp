/* si8_test.cpp -- the half of this candidate that has nothing to do with a card.
 *
 * Every check here runs on a machine with no GPU: the DSO is loaded the way the engine loads it
 * (dlopen, RTLD_LOCAL, the exported symbols and nothing else), and the row's own `launch` entry
 * point is called with malformed calls that must be refused BEFORE the first device call. A
 * refusal that returns a negative code is the whole assertion; a call that gets as far as the
 * device returns RAD_E_DEVICE here, which this test treats as a failure, so it cannot pass by
 * accidentally launching.
 *
 * What is NOT here: the numerics. compare.hip beside this directory is the harness for those
 * (M = 1..3, random / signed extrema / zeros, against the frozen gemm_i8a8_nt_m16 and an
 * independent signed-integer reference), and it needs a card.
 *
 * Usage:
 *   si8_test --off  <path/to/shared_i8a8_down.so>   RADIANCE_SHARED_I8A8_DOWN unset
 *   si8_test --on   <path/to/shared_i8a8_down.so>   RADIANCE_SHARED_I8A8_DOWN=1
 *
 * Two processes, because the plugin reads the variable once into a function-local static: one
 * process cannot see both answers.
 */
#include "rad_abi.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

#define CHECK(cond) check((cond), #cond)

/* ================================================================== the plugin as a loader sees it */

struct Plugin {
    void* handle = nullptr;
    uint32_t (*abi_version)() = nullptr;
    const RadPluginInfo* (*info)() = nullptr;
    int (*schema_count)() = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    int (*concurrent)(int) = nullptr;

    bool open(const char* path) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            std::fprintf(stderr, "FAIL: dlopen %s: %s\n", path, dlerror());
            ++failures;
            return false;
        }
        abi_version  = (uint32_t (*)())dlsym(handle, "rad_plugin_abi_version");
        info         = (const RadPluginInfo* (*)())dlsym(handle, "rad_plugin_info");
        schema_count = (int (*)())dlsym(handle, "rad_kernel_schema_count");
        schema_at    = (const RadOpSchema* (*)(int))dlsym(handle, "rad_kernel_schema_at");
        count        = (int (*)())dlsym(handle, "rad_kernel_count");
        at           = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        concurrent   = (int (*)(int))dlsym(handle, "rad_kernel_concurrent");
        /* The ABI's export set. `rad_kernel_concurrent` is optional in rad_abi.h; every other
         * name here is required of any kernel plugin. */
        CHECK(abi_version && info && schema_count && schema_at && count && at);
        return abi_version && info && schema_count && schema_at && count && at;
    }
    ~Plugin() { if (handle) dlclose(handle); }
};

/* ================================================================== the op schema
 *
 * Position for position, requiredness for requiredness: libr4d's own pGemmNtQ / oGemmNtQ. The
 * loader refuses a plugin whose re-declaration of an op another plugin fixed disagrees, so a
 * schema that drifts here is a load failure on the served path, not a diagnostic. */
void check_schema(const Plugin& p) {
    CHECK(p.schema_count() == 1);
    const RadOpSchema* s = p.schema_at(0);
    if (!s) { CHECK(!"schema 0"); return; }
    CHECK(std::strcmp(s->op, "gemm_nt_q") == 0);
    static const char* const kParams[][2] = {
        { "M", "req" }, { "N", "req" }, { "K", "req" }, { "group", "req" },
        { "dtype", "req" }, { "pb1", "opt" }, { "pb2", "opt" }, { "split_n", "opt" },
    };
    CHECK(s->n_params == (int)(sizeof kParams / sizeof kParams[0]));
    for (int i = 0; i < s->n_params && i < (int)(sizeof kParams / sizeof kParams[0]); ++i) {
        CHECK(std::strcmp(s->params[i].key, kParams[i][0]) == 0);
        CHECK(s->params[i].required == (std::strcmp(kParams[i][1], "req") == 0));
        CHECK(s->params[i].type == (std::strcmp(kParams[i][0], "dtype") == 0 ? RAD_P_STR
                                                                           : RAD_P_INT));
    }
    static const char* const kOperands[] = { "a", "a_scale", "b", "b_scale", "y", "a_sum", "b_ref" };
    static const int kRoles[] = { RAD_OPD_IN, RAD_OPD_IN, RAD_OPD_WEIGHT, RAD_OPD_WEIGHT,
                                  RAD_OPD_OUT, RAD_OPD_IN, RAD_OPD_WEIGHT };
    CHECK(s->n_operands == 7);
    for (int i = 0; i < s->n_operands && i < 7; ++i) {
        CHECK(std::strcmp(s->operands[i].name, kOperands[i]) == 0);
        CHECK(s->operands[i].role == kRoles[i]);
        /* `a_scale` is optional (the bf16-activation reading of the op passes none), as are the
         * two trailing operands this row does not take. */
        CHECK(s->operands[i].optional ==
              (std::strcmp(kOperands[i], "a_scale") == 0 ||
               std::strcmp(kOperands[i], "a_sum") == 0 ||
               std::strcmp(kOperands[i], "b_ref") == 0));
    }
    CHECK(p.schema_at(-1) == nullptr);
    CHECK(p.schema_at(1) == nullptr);
}

/* The constraint table, read the way the selector reads it: three equalities and one set. */
bool constraint_holds(const RadKernelInfo* k, const char* key, long long v) {
    bool seen = false;
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint& c = k->constraints[i];
        if (std::strcmp(c.key, key) != 0) continue;
        seen = true;   /* `M` is a band: GE and LE both have to hold */
        switch (c.op) {
            case RAD_C_EQ: if (v != c.ival) return false; break;
            case RAD_C_LE: if (v >  c.ival) return false; break;
            case RAD_C_GE: if (v <  c.ival) return false; break;
            default: return false;
        }
    }
    return seen;
}

bool dtype_listed(const RadKernelInfo* k, const char* s) {
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint& c = k->constraints[i];
        if (c.op != RAD_C_IN || std::strcmp(c.key, "dtype") != 0) continue;
        const size_t n = std::strlen(s);
        for (const char* p = c.sval; (p = std::strstr(p, s)) != nullptr; ++p)
            if ((p == c.sval || p[-1] == ' ') && (p[n] == '\0' || p[n] == ' ')) return true;
        return false;
    }
    return false;
}

void check_row(const RadKernelInfo* k) {
    if (!k) { CHECK(!"row 0"); return; }
    CHECK(std::strcmp(k->op, "gemm_nt_q") == 0);
    CHECK(k->domain == RAD_DOMAIN_DEVICE);
    CHECK(k->launch && k->opd_shape && k->layout && k->relayout && k->unrelayout);
    CHECK(k->scratch == nullptr);          /* one launch, nothing staged */
    /* The geometry is an equality on N and K, not a band: the body is written for one pair. */
    CHECK(constraint_holds(k, "M", 1) && constraint_holds(k, "M", 3) &&
          !constraint_holds(k, "M", 0) && !constraint_holds(k, "M", 4) &&
          !constraint_holds(k, "M", 5));
    CHECK(constraint_holds(k, "N", 2560) && !constraint_holds(k, "N", 768));
    CHECK(constraint_holds(k, "K", 384) && !constraint_holds(k, "K", 256));
    CHECK(constraint_holds(k, "group", 128) && !constraint_holds(k, "group", 64));
    CHECK(dtype_listed(k, "i8a8"));
    CHECK(!dtype_listed(k, "i8a8h"));
    CHECK(!dtype_listed(k, "fp8a8"));
}

/* ================================================================== the body's refusals
 *
 * A well-formed call at the served geometry, with every field a refusal case perturbs. The buffers
 * exist so the alignment check has something to look at; nothing here is read, because every case
 * below is refused before the first device call. */
constexpr long long N = 2560, K = 384, Group = 128;

struct Operands {
    RadTensor t[7];
    RadParam p[8];
    RadArgs a;
    void* buf[5] = {};
    long long m = 1;

    explicit Operands(long long rows) : m(rows) {
        const long long kb = K / Group;
        auto take = [&](int i, size_t bytes) {
            buf[i] = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
            std::memset(buf[i], 0, ((bytes + 63) / 64) * 64);
            return buf[i];
        };
        std::memset(t, 0, sizeof t);
        /* a: i8 [m][K];  a_scale: f32 [m][K/128];  b: i8 [N][K];  b_scale: bf16 [N][K/128];
         * y: bf16 [m][N]. */
        t[0].data = take(0, (size_t)(m * K));     t[0].dtype = RAD_I8;   t[0].rank = 2;
        t[0].shape[0] = m;  t[0].shape[1] = K;    t[0].stride[0] = K;   t[0].stride[1] = 1;
        t[1].data = take(1, (size_t)(m * kb * 4)); t[1].dtype = RAD_F32; t[1].rank = 2;
        t[1].shape[0] = m;  t[1].shape[1] = kb;   t[1].stride[0] = kb;  t[1].stride[1] = 1;
        t[2].data = take(2, (size_t)(N * K));     t[2].dtype = RAD_I8;   t[2].rank = 2;
        t[2].shape[0] = N;  t[2].shape[1] = K;    t[2].stride[0] = K;   t[2].stride[1] = 1;
        t[3].data = take(3, (size_t)(N * kb * 2)); t[3].dtype = RAD_BF16; t[3].rank = 2;
        t[3].shape[0] = N;  t[3].shape[1] = kb;   t[3].stride[0] = kb;  t[3].stride[1] = 1;
        t[4].data = take(4, (size_t)(m * N * 2)); t[4].dtype = RAD_BF16; t[4].rank = 2;
        t[4].shape[0] = m;  t[4].shape[1] = N;    t[4].stride[0] = N;   t[4].stride[1] = 1;
        static const char* const keys[] = { "M", "N", "K", "group", "dtype", "pb1", "pb2", "split_n" };
        for (int i = 0; i < 8; ++i) {
            p[i].key = keys[i];
            p[i].kind = std::strcmp(keys[i], "dtype") == 0 ? RAD_P_STR : RAD_P_INT;
        }
        p[0].ival = m; p[1].ival = N; p[2].ival = K; p[3].ival = Group;
        p[4].sval = "i8a8"; p[5].ival = 0; p[6].ival = 0; p[7].ival = 0;
        std::memset(&a, 0, sizeof a);
        a.t = t; a.n_t = 5; a.p = p; a.n_p = 8;
    }
    ~Operands() { for (void* q : buf) std::free(q); }
    /* The launch's parameters are what the row is handed by the engine; `M` is the band's upper
     * bound there, which the body ignores on purpose. */
    Operands(const Operands&) = delete;
    Operands& operator=(const Operands&) = delete;
};

/* Each case names the change it makes and the code the body must answer with. A case that returns
 * RAD_E_DEVICE reached the device, which is precisely the bug these exist to catch; it is reported
 * as such rather than as a missing GPU. */
struct Negative {
    const char* what;
    int         want;
    void        (*make)(Operands*);
};

int call(const RadKernelInfo* k, const RadArgs* a) {
    return k->launch(a, (RadStream) nullptr);
}

void negatives(const RadKernelInfo* k) {
    static const Negative cases[] = {
        /* An absent optional operand is a null RadTensor.data (rad_abi.h), so "passed" means a
         * plane, not a longer array. */
        { "a_sum passed", RAD_E_UNSUPPORTED,
          [](Operands* o) { o->t[5].data = o->buf[0]; o->t[5].dtype = RAD_I32; o->a.n_t = 7; } },
        { "wrong a dtype", RAD_E_DTYPE, [](Operands* o) { o->t[0].dtype = RAD_F8E4M3; } },
        { "wrong y dtype", RAD_E_DTYPE, [](Operands* o) { o->t[4].dtype = RAD_F32; } },
        { "rank 1 output", RAD_E_SHAPE, [](Operands* o) { o->t[4].rank = 1; } },
        { "rank 3 weight", RAD_E_SHAPE, [](Operands* o) { o->t[2].rank = 3; } },
        { "K = 256 (rank-1 shape)", RAD_E_SHAPE,
          [](Operands* o) {
              o->t[2].shape[1] = 256; o->t[2].stride[0] = 256; o->p[2].ival = 256;
              o->t[3].shape[1] = 2;   o->t[3].stride[0] = 2;
              o->t[0].shape[1] = 256; o->t[0].stride[0] = 256;
              o->t[1].shape[1] = 2;   o->t[1].stride[0] = 2;
          } },
        { "N = 768 (gate/up shape)", RAD_E_SHAPE,
          [](Operands* o) {
              o->t[2].shape[0] = 768; o->t[3].shape[0] = 768; o->t[4].shape[1] = 768;
              o->t[4].stride[0] = 768; o->p[1].ival = 768;
          } },
        { "M = 5", RAD_E_SHAPE, [](Operands* o) { o->t[4].shape[0] = 5; } },
        { "M = 0", RAD_E_SHAPE, [](Operands* o) { o->t[4].shape[0] = 0; } },
        { "weight scale grid", RAD_E_SHAPE, [](Operands* o) { o->t[3].shape[1] = 4; } },
        { "N parameter disagrees", RAD_E_SHAPE, [](Operands* o) { o->p[1].ival = 2048; } },
        { "group 64", RAD_E_SHAPE, [](Operands* o) { o->p[3].ival = 64; } },
        { "dtype i8a8h", RAD_E_DTYPE, [](Operands* o) { o->p[4].sval = "i8a8h"; } },
        { "dtype missing", RAD_E_DTYPE, [](Operands* o) { o->p[4].sval = nullptr; } },
        { "pb1 partition", RAD_E_UNSUPPORTED, [](Operands* o) { o->p[5].ival = 1; } },
        { "pb2 partition", RAD_E_UNSUPPORTED, [](Operands* o) { o->p[6].ival = 2; } },
        { "loose activation pitch", RAD_E_STRIDE, [](Operands* o) { o->t[0].stride[0] = K + 16; } },
        { "loose output pitch", RAD_E_STRIDE, [](Operands* o) { o->t[4].stride[0] = N + 8; } },
        { "misaligned weight", RAD_E_ALIGN,
          [](Operands* o) { o->t[2].data = (char*)o->t[2].data + 1; } },
        { "misaligned scale plane", RAD_E_ALIGN,
          [](Operands* o) { o->t[3].data = (char*)o->t[3].data + 1; } },
    };
    for (const Negative& c : cases) {
        Operands o(1);
        c.make(&o);
        const int rc = call(k, &o.a);
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s -> %d (%s), want %d", c.what, rc, rad_strerror(rc),
                      c.want);
        check(rc == c.want, buf);
    }
    /* THE BAND'S OWN EDGE: four complete operand rows, every plane sized for four, exactly as a
     * four-token verify step would issue them. The M=5 case above changes only the output shape,
     * which the operand-completeness checks refuse as well, so it cannot stand in for this one:
     * only a complete four-row call tells the row cap apart from a truncated plane. With the cap
     * widened to four this call reaches the device (RAD_E_DEVICE under the test's hidden devices)
     * and the check fails. */
    {
        Operands o(4);
        const int rc = call(k, &o.a);
        char buf[160];
        std::snprintf(buf, sizeof buf, "M = 4, four full operand rows -> %d (%s), want %d", rc,
                      rad_strerror(rc), RAD_E_SHAPE);
        check(rc == RAD_E_SHAPE, buf);
    }
    /* A null RadArgs, and one carrying fewer operands than the body reads. */
    CHECK(call(k, nullptr) == RAD_E_INVAL);
    {
        Operands o(1);
        o.a.t = nullptr;
        CHECK(call(k, &o.a) == RAD_E_INVAL);
    }
}

/* The operand description is bounded on the same band as the body, so a tool that asks about four
 * rows is told this row will not run it instead of being sized buffers for a refusal. The
 * parameter list comes from Operands, so M, N, K and the group are the ones the body reads. */
void shape_refusals(const RadKernelInfo* k) {
    {
        Operands o(4);
        RadOpdDesc d{};
        const int rc = k->opd_shape(o.a.p, o.a.n_p, 4, &d);
        char buf[160];
        std::snprintf(buf, sizeof buf, "opd_shape M = 4 -> %d (%s), want %d", rc,
                      rad_strerror(rc), RAD_E_UNSUPPORTED);
        check(rc == RAD_E_UNSUPPORTED, buf);
    }
    /* The band's top row is still described, so the refusal above is the cap and not a stub. */
    {
        Operands o(3);
        RadOpdDesc d{};
        CHECK(k->opd_shape(o.a.p, o.a.n_p, 4, &d) == RAD_OK && d.shape[0] == 3);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool expect_rows = false;
    const char* path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--on") == 0) expect_rows = true;
        else if (std::strcmp(argv[i], "--off") == 0) expect_rows = false;
        else path = argv[i];
    }
    if (!path) {
        std::fprintf(stderr, "usage: si8_test (--on|--off) <shared_i8a8_down.so>\n");
        return 2;
    }
    const char* env = std::getenv("RADIANCE_SHARED_I8A8_DOWN");
    const bool on = env && std::strcmp(env, "1") == 0;
    if (on != expect_rows) {
        std::fprintf(stderr, "call this with RADIANCE_SHARED_I8A8_DOWN %s for --%s\n",
                     expect_rows ? "=1" : "unset", expect_rows ? "on" : "off");
        return 2;
    }

    Plugin p;
    if (!p.open(path)) return 1;
    CHECK(p.abi_version() == RAD_ABI_VERSION);
    check_schema(p);
    const RadPluginInfo* info = p.info();
    CHECK(info && info->kind == RAD_PLUGIN_KERNEL);
    CHECK(info && std::strcmp(info->name, "shared_i8a8_down") == 0);

    /* Opt-in, and the whole of it: with the variable unset the plugin declares its schema and no
     * kernel, so selection is exactly the build's existing one. */
    if (!expect_rows) {
        CHECK(p.count() == 0);
        CHECK(p.at(0) == nullptr);
        CHECK(!p.concurrent || p.concurrent(0) == 0);
        std::printf("off: 1 schema, 0 rows\n");
    } else {
        CHECK(p.count() == 1);
        const RadKernelInfo* k = p.at(0);
        check_row(k);
        CHECK(p.at(1) == nullptr);
        CHECK(p.concurrent && p.concurrent(0) == 1);
        if (k) {
            negatives(k);
            shape_refusals(k);
        }
        std::printf("on:  1 schema, 1 row\n");
    }

    if (failures) {
        std::fprintf(stderr, "RESULT: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("RESULT: OK\n");
    return 0;
}
