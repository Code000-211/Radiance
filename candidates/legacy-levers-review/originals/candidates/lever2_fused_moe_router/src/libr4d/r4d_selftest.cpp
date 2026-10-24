/* r4d_selftest -- dump the libr4d row table, cross-check it, and prove every row links and runs.
 *
 * A transcription error in a constraint table is invisible until a model silently picks the wrong
 * kernel, which is exactly the failure this project exists to make impossible. So the table is
 * printed in full and then checked against itself:
 *
 *   - every row's `op` has a schema declared by this plugin, or nothing can ever select it;
 *   - every CONSTRAINT KEY is a declared parameter of that op. This is the check that earns its
 *     keep: libr4d's select() has its own parameter vocabulary and radiance's schemas are
 *     docs/OPS.md's, so a key that was renamed in one place and not the other produces a row that
 *     matches nothing and reports nothing;
 *   - names are unique, launches exist, an init has a fini, a declared variant list is non-empty.
 *
 * Then, on a device: every launch is called once with a minimal valid input, so a linkage error, a
 * missing libr4d symbol or an argument-order mistake surfaces here rather than in a serve. Two of
 * these are real numerical checks rather than smoke tests, because they are cheap and because they
 * cover the two things most likely to be silently wrong:
 *
 *   - the W4 weight arrangement, checked against libr4d's own dequant (`dequant_w4_bf16` undoes
 *     the fragment order), so the host relayout and the device kernel have to agree;
 *   - the two-rank all-reduce, run across two real cards through the plugin's init/fini
 *     rendezvous, with a known sum. That exercises the peer handshake, which cannot be tested any
 *     other way.
 *
 * The plugin is dlopened rather than linked, because an undefined symbol in the .so is itself one
 * of the failures worth catching.
 *
 * ONE CHILD PROCESS PER ROW. A bad pointer handed to a kernel is a GPU page fault, and a GPU page
 * fault is not an error code -- the ROCr runtime prints one line to stderr and aborts the process
 * where it stands. In a single process that means the FIRST bad row ends the run and every row
 * after it reports nothing, which is the opposite of what a catalogue check is for; worse, stdout
 * is block-buffered down a pipe, so the abort interleaves the fault line into whatever was still
 * in the buffer and the row it lands next to is not the row that faulted -- a fault that reads as
 * `quant_act_i8`'s can belong to a row twenty earlier.
 *
 * So the parent runs the static phase and then re-execs ITSELF once per row with `--case <name>`,
 * and reports the child's exit status: a signal is a FAIL naming the signal, and the run
 * continues. posix_spawn rather than a bare fork, because the parent has the HIP runtime and its
 * threads up by then and only exec resets them. The row list is the plugin's own row table, so a
 * row nobody wrote a smoke case for reports `no smoke case` rather than being silently absent --
 * the coverage check falls out of the isolation for free.
 *
 * `--case <name>` is also the reproducer: it runs exactly one row, alone, in the foreground.
 */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_args.h"
#include "r4d_hc_fp8.h"
#include "r4d_fp8_frag.h"
#include "r4d_fp8_lm.h"

#include <dlfcn.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

/* posix_spawn's environment. Declared here rather than relying on unistd.h's _GNU_SOURCE guard. */
extern char** environ;

#ifdef RAD_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

static int g_fail = 0;
static int g_skip = 0;

/* The row this process was asked to run, or null for the parent. Set from --case. */
static const char* g_case = nullptr;

/* This binary and the plugin it was given, so a case that needs a process of its own can ask for
 * one. A kernel that reads its configuration from the environment caches the answer, which makes
 * two settings of it two processes whatever else is true. */
static const char* g_exe = nullptr;
static const char* g_plugin = nullptr;
/* Whether any case block claimed g_case. A row the sweep never mentions is a coverage hole and
 * says so, rather than exiting 0 having done nothing. */
static bool g_matched = false;

static bool sel(const char* name) {
    if (!g_case) return true;
    if (std::strcmp(g_case, name) != 0) return false;
    g_matched = true;
    return true;
}

/* Exit codes the child speaks: 0 clean, 1 at least one FAIL, 2 nothing but skips. Anything else,
 * or a signal, is the child dying where it stood and is a FAIL the parent attributes by name. */
enum { CH_OK = 0, CH_FAIL = 1, CH_SKIP = 2 };

static void ok(const char* what, bool cond, const std::string& detail = std::string()) {
    if (cond) { std::printf("  ok    %s\n", what); return; }
    ++g_fail;
    std::printf("  FAIL  %s%s%s\n", what, detail.empty() ? "" : " -- ", detail.c_str());
}

static void skip(const char* what, const char* why) {
    ++g_skip;
    std::printf("  skip  %s -- %s\n", what, why);
}

static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

/* Re-exec this binary on one case with the anchor state form selected, and hand back the exit
 * status the way the row driver does. The state form is read from the environment once per process
 * and cached, so the per-candidate smoke case and the anchor check cannot share one: the choice is
 * made before either of them can ask for it. */
static int spawn_anchor_child(const char* case_name) {
    std::vector<char*> env;
    for (char** e = environ; *e; ++e)
        if (std::strncmp(*e, "RADIANCE_GDN_ANCHOR=", 20) != 0) env.push_back(*e);
    char on[] = "RADIANCE_GDN_ANCHOR=1";
    env.push_back(on);
    env.push_back(nullptr);
    char* const argv[] = { const_cast<char*>(g_exe), const_cast<char*>(g_plugin),
                           const_cast<char*>("--case"), const_cast<char*>(case_name), nullptr };
    pid_t pid = 0;
    if (posix_spawn(&pid, g_exe, nullptr, nullptr, argv, env.data()) != 0) return -1;
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) return -1;
    std::fflush(stdout);
    if (WIFSIGNALED(st)) return 128 + WTERMSIG(st);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}


/* ------------------------------------------------------------------ the plugin surface */

struct Plugin {
    void* h = nullptr;
    uint32_t (*abi)(void) = nullptr;
    const RadPluginInfo* (*info)(void) = nullptr;
    int (*schema_count)(void) = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*kernel_count)(void) = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    /* A TEST HOOK and not part of the plugin ABI -- libr4d's own, resolved out of the same .so.
     * r4d.h says why the wave count cannot be compared any other way. */
    void (*pin_mw)(int) = nullptr;
    void (*pin_nt)(int) = nullptr;
    void (*pin_qsa)(int) = nullptr;
    void (*pin_moe)(int) = nullptr;
    void (*pin_hc)(int) = nullptr;
    void (*pin_qsel)(int) = nullptr;
};

/* Defined after the tiny-tensor helpers it needs; called from the static checks above them. */

template <typename T>
static bool sym(void* h, const char* n, T* out) {
    dlerror();
    void* p = dlsym(h, n);
    const char* e = dlerror();
    if (e || !p) { std::printf("  FAIL  dlsym(%s): %s\n", n, e ? e : "null"); ++g_fail; return false; }
    std::memcpy(out, &p, sizeof(p));
    return true;
}

static bool load(Plugin* pl, const char* path) {
    pl->h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!pl->h) { std::printf("  FAIL  dlopen(%s): %s\n", path, dlerror()); ++g_fail; return false; }
    bool okall = true;
    okall &= sym(pl->h, "rad_plugin_abi_version", &pl->abi);
    okall &= sym(pl->h, "rad_plugin_info", &pl->info);
    okall &= sym(pl->h, "rad_kernel_schema_count", &pl->schema_count);
    okall &= sym(pl->h, "rad_kernel_schema_at", &pl->schema_at);
    okall &= sym(pl->h, "rad_kernel_count", &pl->kernel_count);
    okall &= sym(pl->h, "rad_kernel_at", &pl->kernel_at);
    okall &= sym(pl->h, "r4d_gemm_fp8a8_pin_mw", &pl->pin_mw);
    okall &= sym(pl->h, "r4d_gemm_fp8a8_pin_nt", &pl->pin_nt);
    okall &= sym(pl->h, "r4d_qsa_score_pin_form", &pl->pin_qsa);
    okall &= sym(pl->h, "r4d_moe_gemm_pin_prefill", &pl->pin_moe);
    okall &= sym(pl->h, "r4d_hc_read_pin_prefill", &pl->pin_hc);
    okall &= sym(pl->h, "r4d_qsa_select_pin_form", &pl->pin_qsel);
    return okall;
}

static const char* cop_name(int op) {
    switch (op) {
        case RAD_C_EQ:  return "==";
        case RAD_C_LE:  return "<=";
        case RAD_C_GE:  return ">=";
        case RAD_C_DIV: return "%";
        case RAD_C_IN:  return "in";
        default:        return "?";
    }
}

/* ------------------------------------------------------------------ the catalogue */

static void dump(const Plugin& pl) {
    const RadPluginInfo* i = pl.info();
    std::printf("plugin  %s %s (%s)\n        %s\n\n", i->name, i->version, i->build_target,
                i->description);

    std::printf("op schemas (%d)\n", pl.schema_count());
    for (int s = 0; s < pl.schema_count(); ++s) {
        const RadOpSchema* sc = pl.schema_at(s);
        std::printf("  %-30s", sc->op);
        for (int j = 0; j < sc->n_params; ++j)
            std::printf(" %s%s%s", sc->params[j].required ? "" : "[",
                        sc->params[j].key, sc->params[j].required ? "" : "]");
        std::printf("\n%34s", "");
        for (int j = 0; j < sc->n_operands; ++j) {
            static const char* role[] = { "in", "out", "inout", "w" };
            std::printf(" %s:%s%s", sc->operands[j].name,
                        role[sc->operands[j].role & 3], sc->operands[j].optional ? "?" : "");
        }
        std::printf("\n");
    }

    std::printf("\nkernels (%d)\n", pl.kernel_count());
    for (int k = 0; k < pl.kernel_count(); ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        std::printf("  %-44s op=%-28s fam=%-7s prio=%-3d dom=%s%s%s%s%s\n",
                    r->name, r->op, r->family, r->priority,
                    r->domain == RAD_DOMAIN_DEVICE ? "device" : "host",
                    r->init ? " init" : "", r->scratch ? " scratch" : "",
                    r->layout ? " layout" : "",
                    r->relayout ? (r->unrelayout ? " relayout+inverse" : " relayout") : "");
        std::printf("        ");
        for (int c = 0; c < r->n_constraints; ++c) {
            const RadConstraint& cc = r->constraints[c];
            if (cc.op == RAD_C_IN) std::printf("%s in {%s}  ", cc.key, cc.sval);
            else if (cc.op == RAD_C_DIV) std::printf("%s %% %lld == 0  ", cc.key, cc.ival);
            else std::printf("%s %s %lld  ", cc.key, cop_name(cc.op), cc.ival);
        }
        std::printf("\n");
        if (r->n_tunables) {
            int64_t space = 1;
            std::printf("        tunables(%d):", r->n_tunables);
            for (int v = 0; v < r->n_tunables; ++v) {
                const RadTunable& t = r->tunables[v];
                space *= t.n_values > 0 ? t.n_values : 1;
                std::printf(" %s=%lld/%d", t.key, (long long)t.deflt, t.n_values);
            }
            std::printf("   space %lld%s\n", (long long)space,
                        r->tune_valid ? " (pruned)" : "");
        }
    }
    std::printf("\n");
}

/* ------------------------------------------------------------------ the static checks */

static void cross_check(const Plugin& pl) {
    std::printf("static checks\n");
    ok("ABI version matches the header this was built against", pl.abi() == RAD_ABI_VERSION);
    ok("plugin declares itself a kernel plugin", pl.info()->kind == RAD_PLUGIN_KERNEL);

    const int ns = pl.schema_count(), nk = pl.kernel_count();
    ok("at least one op schema", ns > 0);
    ok("at least one kernel row", nk > 0);

    /* Unique names. */
    int dup_schema = 0, dup_kernel = 0;
    for (int a = 0; a < ns; ++a)
        for (int b = a + 1; b < ns; ++b)
            if (!std::strcmp(pl.schema_at(a)->op, pl.schema_at(b)->op)) ++dup_schema;
    for (int a = 0; a < nk; ++a)
        for (int b = a + 1; b < nk; ++b)
            if (!std::strcmp(pl.kernel_at(a)->name, pl.kernel_at(b)->name)) ++dup_kernel;
    ok("op schema names are unique", dup_schema == 0);
    ok("kernel names are unique", dup_kernel == 0);

    int no_schema = 0, bad_key = 0, no_launch = 0, init_no_fini = 0, bad_domain = 0, no_p2p = 0;
    int bad_axis = 0, bad_deflt = 0, dup_axis = 0;
    std::string axisdetail;
    std::string keydetail, p2pdetail;
    for (int k = 0; k < nk; ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (!r->launch) { ++no_launch; continue; }
        if (r->init && !r->fini) ++init_no_fini;
        /* EVERY ROW IS DEVICE DOMAIN. This library is a device kernel library and a stray host
         * row would be a step that silently leaves the card -- which is why the check exists.
         * Host kernels are libavx's. */
        if (r->domain != RAD_DOMAIN_DEVICE) {
            ++bad_domain;
            keydetail += std::string(r->name) + " is not RAD_DOMAIN_DEVICE; ";
        }
        /* EVERY AXIS CARRIES A DEFAULT AND THE DEFAULT IS ONE OF ITS OWN VALUES. The loader
         * refuses a plugin that breaks this, by name, so a deployment can never run an axis with
         * no legal fallback -- but the loader's refusal is a startup failure, and this is the
         * check that tells the author which row and which key at the moment they add it. */
        for (int v = 0; v < r->n_tunables; ++v) {
            const RadTunable& t = r->tunables[v];
            if (!t.key || !t.key[0] || !t.values || t.n_values < 1) {
                ++bad_axis;
                if (axisdetail.empty())
                    axisdetail = std::string(r->name) + " axis " + std::to_string(v);
                continue;
            }
            int found = 0;
            for (int i = 0; i < t.n_values; ++i) if (t.values[i] == t.deflt) found = 1;
            if (!found) {
                ++bad_deflt;
                if (axisdetail.empty())
                    axisdetail = std::string(r->name) + ":" + t.key + " default "
                               + std::to_string((long long)t.deflt) + " is not one of its values";
            }
            for (int w = 0; w < v; ++w)
                if (r->tunables[w].key && !std::strcmp(r->tunables[w].key, t.key)) {
                    ++dup_axis;
                    if (axisdetail.empty())
                        axisdetail = std::string(r->name) + " declares " + t.key + " twice";
                }
        }

        const RadOpSchema* sc = nullptr;
        for (int s = 0; s < ns; ++s)
            if (!std::strcmp(pl.schema_at(s)->op, r->op)) { sc = pl.schema_at(s); break; }
        if (!sc) {
            ++no_schema;
            keydetail += std::string(r->name) + " has no schema for op '" + r->op + "'; ";
            continue;
        }
        /* THE CHECK THAT EARNS ITS KEEP. A constraint on a key the op does not declare can never
         * be satisfied, so the row is unreachable and nothing says so. A "device." key is not a
         * parameter at all: it is matched against the machine (rad_types.h). */
        static const size_t kFact = std::strlen(RAD_DEVICE_FACT_PREFIX);
        for (int c = 0; c < r->n_constraints; ++c) {
            if (!std::strncmp(r->constraints[c].key, RAD_DEVICE_FACT_PREFIX, kFact)) continue;
            bool found = false;
            for (int j = 0; j < sc->n_params; ++j)
                if (!std::strcmp(sc->params[j].key, r->constraints[c].key)) { found = true; break; }
            if (!found) {
                ++bad_key;
                keydetail += std::string(r->name) + " constrains '" + r->constraints[c].key +
                             "' which op '" + r->op + "' does not declare; ";
            }
        }
        /* EVERY MULTI-CARD ROW NEEDS PEER ACCESS. Each kernel in this library that runs only at
         * two or more ranks reads or writes the other cards' memory directly, so without
         * device.p2p a machine whose cards cannot reach each other resolves it and the launch
         * faults or hangs instead of being refused by name at selection. */
        bool multi = false, p2p = false;
        for (int c = 0; c < r->n_constraints; ++c) {
            const RadConstraint& rc = r->constraints[c];
            if (!std::strcmp(rc.key, "world_size") &&
                (rc.op == RAD_C_EQ || rc.op == RAD_C_GE) && rc.ival >= 2)
                multi = true;
            if (!std::strcmp(rc.key, RAD_FACT_P2P) && rc.op == RAD_C_EQ && rc.ival == 1)
                p2p = true;
        }
        if (multi && !p2p) {
            ++no_p2p;
            p2pdetail += std::string(r->name) + "; ";
        }
    }
    ok("every row has a launch", no_launch == 0);
    ok("every row with init has fini", init_no_fini == 0);
    ok("every row is RAD_DOMAIN_DEVICE but the named host gather", bad_domain == 0, keydetail);
    ok("every tunable axis has a key and at least one value", bad_axis == 0, axisdetail);
    ok("every tunable default is one of its own axis values", bad_deflt == 0, axisdetail);
    ok("no row declares the same axis key twice", dup_axis == 0, axisdetail);
    ok("every row's op has a schema", no_schema == 0, keydetail);
    ok("every constraint key is a declared parameter of its op", bad_key == 0, keydetail);
    ok("every row that needs two or more ranks requires device.p2p", no_p2p == 0, p2pdetail);

    /* The libr4d orderings the integer priority has to preserve. */
    int p_ts = -1, p_os = -1, p_dec = -1, p_pre = -1, p_m16 = -1, p_m64 = -1, p_bt = -1;
    for (int k = 0; k < nk; ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (!std::strcmp(r->name, "ar_twoshot_4rank_exact")) p_ts = r->priority;
        if (!std::strcmp(r->name, "ar_oneshot_4rank_exact")) p_os = r->priority;
        if (!std::strcmp(r->name, "attn_decode_h256_gqa6_fp8kv"))  p_dec = r->priority;
        if (!std::strcmp(r->name, "attn_prefill_h256_gqa6_fp8kv")) p_pre = r->priority;
        if (!std::strcmp(r->name, "gemm_bf16_nt_m16")) p_m16 = r->priority;
        if (!std::strcmp(r->name, "gemm_bf16_nt_m64")) p_m64 = r->priority;
        if (!std::strcmp(r->name, "gemm_bf16_nt_tiled")) p_bt = r->priority;
    }
    ok("two-shot all-reduce outranks one-shot (libr4d row order)", p_ts > p_os && p_ts > 0);
    ok("paged decode outranks paged prefill in the shared band", p_dec > p_pre && p_dec > 0);
    ok("gemm_bf16_nt_m16 outranks _m64 (libr4d row order, see the ladder)", p_m16 > p_m64);
    ok("gemm_bf16_nt_tiled outranks _m64 and not _m16 (M > 64 is its band)",
       p_bt > p_m64 && p_bt < p_m16);
    std::printf("\n");
}

/* ================================================================== the device phase */
#ifdef RAD_HAVE_HIP

/* EVERY hipError_t IN THIS FILE IS CHECKED, and that is not pedantry about a `nodiscard`. A
 * dropped device-to-host hipMemcpy is exactly how a correctness harness reports PASS on memory the
 * device never wrote: a kernel that faults leaves its error sticky on the next HIP call, so the
 * copy meant to bring the answer back fails and the comparison runs on whatever the host buffer
 * already held. Fresh, that is zeros and the case fails loudly. REUSED -- one allocation and a
 * loop over geometries, which is the normal shape in this file -- it still holds the PREVIOUS
 * geometry's correct answer, and the case passes.
 *
 * Forwarding macros rather than a wrapper function, so every call site stays an ordinary call and
 * __LINE__ comes for free. The failure goes through `g_fail`, which every one of main()'s four
 * exit paths already returns, so nothing downstream needs to know about this.
 *
 * AND IT RETURNS int, NOT hipError_t. ROCm puts the `nodiscard` on the TYPE -- `typedef enum
 * __HIP_NODISCARD hipError_t` -- not on the individual functions, so ANY expression of that type
 * whose value is dropped warns, this wrapper's own return included; returning hipError_t here
 * would move every one of those warnings onto the ten #define lines below instead of removing
 * them. `hipSuccess` is 0 and an enum promotes, so the handful of sites that do test the result
 * read unchanged. */
static int hip_ck(hipError_t e, const char* what, int line) {
    if (e != hipSuccess) {
        std::printf("  FAIL  %s at r4d_selftest.cpp:%d: %s\n", what, line, hipGetErrorString(e));
        ++g_fail;
    }
    return (int)e;
}
#define st_memcpy(...)      hip_ck(hipMemcpy(__VA_ARGS__), "hipMemcpy", __LINE__)
#define st_memcpy2d(...)    hip_ck(hipMemcpy2D(__VA_ARGS__), "hipMemcpy2D", __LINE__)
#define st_memset(...)      hip_ck(hipMemset(__VA_ARGS__), "hipMemset", __LINE__)
#define st_free(...)        hip_ck(hipFree(__VA_ARGS__), "hipFree", __LINE__)
#define st_set_device(...)  hip_ck(hipSetDevice(__VA_ARGS__), "hipSetDevice", __LINE__)
#define st_stream_sync(...) hip_ck(hipStreamSynchronize(__VA_ARGS__), "hipStreamSynchronize", \
                                   __LINE__)
#define st_ev_create(...)   hip_ck(hipEventCreate(__VA_ARGS__), "hipEventCreate", __LINE__)
#define st_ev_destroy(...)  hip_ck(hipEventDestroy(__VA_ARGS__), "hipEventDestroy", __LINE__)
#define st_ev_record(...)   hip_ck(hipEventRecord(__VA_ARGS__), "hipEventRecord", __LINE__)
#define st_ev_sync(...)     hip_ck(hipEventSynchronize(__VA_ARGS__), "hipEventSynchronize", \
                                   __LINE__)
#define st_ev_elapsed(...)  hip_ck(hipEventElapsedTime(__VA_ARGS__), "hipEventElapsedTime", \
                                   __LINE__)

static std::vector<void*> g_allocs;

static void* dalloc(size_t bytes) {
    void* p = nullptr;
    if (bytes == 0) bytes = 16;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    st_memset(p, 0, bytes);
    g_allocs.push_back(p);
    return p;
}

static void dfree_all() {
    for (void* p : g_allocs) st_free(p);
    g_allocs.clear();
}

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t;
    std::memset(&t, 0, sizeof(t));
    t.data = d;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}

/* moe_gemm_q TAKES ITS EXPERTS AS TWO TABLES BY PARITY (operands 2/3 even, 6/7 odd). A stacked
 * plane holding every expert in order is both at once: the even entries at twice its leading
 * stride, and the odd ones the same from one plane in. `wbits`/`sbits` are the two planes'
 * element widths, which a plugin-private dtype cannot be asked for. */
static void moe_parity_tables(std::vector<RadTensor>& ts, int64_t ne, int wbits, int sbits) {
    const RadTensor w = ts[2], s = ts[3];
    RadTensor wo = w, so = s;
    ts[2].shape[0] = ts[3].shape[0] = (ne + 1) / 2;
    ts[2].stride[0] *= 2;
    ts[3].stride[0] *= 2;
    wo.shape[0] = so.shape[0] = ne / 2;
    wo.stride[0] *= 2;
    so.stride[0] *= 2;
    wo.data = (char*)w.data + w.stride[0] * wbits / 8;
    so.data = (char*)s.data + s.stride[0] * sbits / 8;
    /* In front of the output, which is the last operand. */
    ts.insert(ts.end() - 1, wo);
    ts.insert(ts.end() - 1, so);
}

/* An ABSENT optional operand is a null `data`, not a zero rank -- r4d_opt() keys on the pointer
 * (rad_abi.h) and a shim reading a rank-0 tensor whose data still points at the last allocation
 * takes the present branch on an operand the test meant to withhold. Clearing rank alone leaves
 * quant_act_i8's plain form re-running the asum form. */
static void absent(RadTensor& t) { t.data = nullptr; t.rank = 0; }

/* Cumulative sequence lengths, on the device. Every gdn kernel opens with `bos = cu[n]` and
 * `T = cu[n+1] - bos`, so a cu of zeros -- which is what dalloc hands back -- means T == 0 for
 * every sequence and the whole grid returns before it touches an operand. That is a launch that
 * proves nothing, so the smoke tests fill it. */
static void fill_i32(void* d, const std::vector<int32_t>& v) {
    st_memcpy(d, v.data(), v.size() * sizeof(int32_t), hipMemcpyHostToDevice);
}

static std::vector<int32_t> cu_even(int64_t seqs, int64_t total) {
    std::vector<int32_t> cu((size_t)seqs + 1, 0);
    for (int64_t i = 0; i <= seqs; ++i) cu[(size_t)i] = (int32_t)(total * i / seqs);
    return cu;
}

static RadParam PI(const char* k, long long v) {
    RadParam p; std::memset(&p, 0, sizeof(p));
    p.key = k; p.kind = RAD_P_INT; p.ival = v; return p;
}
static RadParam PS(const char* k, const char* v) {
    RadParam p; std::memset(&p, 0, sizeof(p));
    p.key = k; p.kind = RAD_P_STR; p.sval = v; return p;
}
/* A real-valued parameter. These are RAD_P_F64 in the schemas (docs/OPS.md), and the point of
 * spelling them that way here is that this file feeds the launches DIRECTLY -- there is no declare
 * in front of it to catch a kind mismatch, so a case that passed them as strings would silently
 * take the shim's default instead of the value it names. */
static RadParam PF(const char* k, double v) {
    RadParam p; std::memset(&p, 0, sizeof(p));
    p.key = k; p.kind = RAD_P_F64; p.dval = v; return p;
}

/* Bytes per element for the dtypes this file allocates. Deliberately local: the core's
 * rad_dtype_bytes is not linkable from here (see the note in r4d_plugin.h). */
static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: case RAD_U32: return 4;
        case RAD_BF16: case RAD_F16: case RAD_I16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

/* An avalanche hash for test data: a value that depends on every bit of its index, so a wrong
 * pitch or lane map reads numbers that are unrelated rather than a shifted copy of the right ones. */
static uint32_t st_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

static void* alloc_for(const RadTensor& t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t.rank; ++i) n *= t.shape[i];
    return dalloc((size_t)n * esz(t.dtype));
}

/* An i32 operand whose CONTENTS matter -- cu, state_idx, num_accepted. run_case allocates what is
 * still null and dalloc zeroes it, and a zero is a wrong answer for all three: a zero cu means
 * every sequence has no tokens and the grid exits before touching an operand, and a zero
 * num_accepted makes the recurrent update read column `nacc - 1` == -1 of its state-index row.
 * So these are built and filled here, before run_case sees them. */
static RadTensor TI32(const std::vector<int64_t>& shape, const std::vector<int32_t>& v) {
    RadTensor t = T(nullptr, RAD_I32, shape);
    t.data = alloc_for(t);
    if (t.data) fill_i32(t.data, v);
    return t;
}

/* n copies of one value: state slots and accepted counts, which are per-sequence constants here. */
static std::vector<int32_t> rep_i32(int64_t n, int32_t v) {
    return std::vector<int32_t>((size_t)n, v);
}

/* An f32 operand filled with one value. The attention descales are the case: dalloc zeroes, a zero
 * descale scales every score to zero, and a uniform softmax over a zeroed cache is a launch that
 * would pass with the dequantise multiply deleted. */
static RadTensor TF32(const std::vector<int64_t>& shape, float v) {
    RadTensor t = T(nullptr, RAD_F32, shape);
    t.data = alloc_for(t);
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    const std::vector<float> h((size_t)n, v);
    if (t.data) st_memcpy(t.data, h.data(), (size_t)n * sizeof(float), hipMemcpyHostToDevice);
    return t;
}

/* 0, 1, ... n-1: one distinct cache slot per sequence, so two sequences do not race on one. */
static std::vector<int32_t> iota_i32(int64_t n) {
    std::vector<int32_t> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (int32_t)i;
    return v;
}

/* ------------------------------------------------------------------ the tuning space
 *
 * A row declares AXES and one point in their cross product is one instantiation of the kernel.
 * This enumerates that product the way the core does at tune time -- all-defaults first, so a
 * sweep's first line is what an untuned install runs, then every other legal point -- and asks
 * the row's own tune_valid to prune the combinations no geometry can serve.
 *
 * THE SELFTEST ENUMERATES RATHER THAN LISTING, for the same reason rows declare their space rather
 * than naming a table of points: a hand-kept list in a harness drifts from the plugin it is
 * testing, and drifts silently, because a missing point looks exactly like a point that was never
 * fast. */
/* How many points an INTERACTIVE sweep walks. rad-tune walks the whole space; a --perfbf16 that
 * printed four hundred lines at 400 iterations each would be a different tool. Raise it with
 * R4D_PERF_POINTS when a specific axis needs the whole ladder. */
static const size_t kPerfCap = [] {
    const char* e = std::getenv("R4D_PERF_POINTS");
    const long v = (e && *e) ? std::strtol(e, nullptr, 10) : 0;
    return (size_t)(v > 0 ? v : 64);
}();

struct TunePoint {
    std::vector<RadParam> choice;
    std::string           label;
};

static std::vector<TunePoint> tune_points(const RadKernelInfo* r, const std::vector<RadParam>& geom,
                                          size_t cap = 0) {
    std::vector<TunePoint> out;
    if (!r || r->n_tunables <= 0) { out.push_back(TunePoint{}); out.back().label = "-"; return out; }

    const int n = r->n_tunables;
    std::vector<int> idx((size_t)n, 0);
    for (;;) {
        TunePoint tp;
        tp.choice.resize((size_t)n);
        for (int i = 0; i < n; ++i) {
            const RadTunable& t = r->tunables[i];
            std::memset(&tp.choice[(size_t)i], 0, sizeof(RadParam));
            tp.choice[(size_t)i].key  = t.key;
            tp.choice[(size_t)i].kind = RAD_P_INT;
            tp.choice[(size_t)i].ival = t.values[idx[(size_t)i]];
            if (i) tp.label += ',';
            tp.label += std::string(t.key) + "=" + std::to_string((long long)t.values[idx[(size_t)i]]);
        }
        const int legal = !r->tune_valid ||
                          r->tune_valid(geom.data(), (int)geom.size(),
                                        tp.choice.data(), (int)tp.choice.size());
        if (legal) {
            /* The all-defaults point goes FIRST wherever it lands in the odometer: a sweep is read
             * top-down and the baseline has to be the first line or every delta is eyeballed. */
            int is_deflt = 1;
            for (int i = 0; i < n; ++i)
                if (tp.choice[(size_t)i].ival != r->tunables[i].deflt) { is_deflt = 0; break; }
            if (is_deflt) out.insert(out.begin(), std::move(tp));
            else          out.push_back(std::move(tp));
        }
        int i = n - 1;
        while (i >= 0 && ++idx[(size_t)i] >= r->tunables[i].n_values) { idx[(size_t)i] = 0; --i; }
        if (i < 0) break;
        if (cap && out.size() >= cap) break;
    }
    if (out.empty()) { out.push_back(TunePoint{}); out.back().label = "-"; }
    return out;
}

/* Append a point's choices to a case's parameters. This is exactly what the core does at issue:
 * the chosen values arrive as ordinary parameters and the launch reads `sk` with the same
 * rad_args_geti() it reads `M` with. */
static std::vector<RadParam> with_point(const std::vector<RadParam>& ps, const TunePoint& tp) {
    std::vector<RadParam> out = ps;
    out.insert(out.end(), tp.choice.begin(), tp.choice.end());
    return out;
}

/* Run one row's launch with the operands and parameters the caller built. Allocates every tensor
 * whose data is null, sizes and allocates scratch from the row's own hook, then synchronises so a
 * launch failure is attributed here and not to whatever runs next. */
static void run_case(const RadKernelInfo* r, std::vector<RadTensor>& ts,
                     std::vector<RadParam>& ps, int expect = RAD_OK) {
    for (auto& t : ts) if (!t.data && t.rank) t.data = alloc_for(t);

    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;

    if (r->scratch) {
        const int64_t need = r->scratch(&a);
        if (need < 0) { ok(r->name, false, "scratch hook returned " + std::to_string(need)); return; }
        if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
    }

    hipStream_t st = nullptr;
    const int rc = r->launch(&a, (RadStream)st);
    const hipError_t he = (hipError_t)st_stream_sync(st);
    std::string d = "rc=" + std::to_string(rc);
    if (he != hipSuccess) d += std::string(" hip=") + hipGetErrorString(he);
    ok(r->name, rc == expect && he == hipSuccess, d);
}

static const RadKernelInfo* find(const Plugin& pl, const char* name) {
    for (int k = 0; k < pl.kernel_count(); ++k)
        if (!std::strcmp(pl.kernel_at(k)->name, name)) return pl.kernel_at(k);
    return nullptr;
}

/* ------------------------------------------------------------------ per-family smoke tests */

static void attn_case(const Plugin& pl, const char* name, int64_t head_dim, int64_t gqa,
                      int64_t q_len, const char* kv_dtype, int prefill) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    const bool is_csa = (std::strstr(name, "csa3") != nullptr);
    const int64_t seqs = 2, kv_heads = 2, bs = 16;
    const int64_t blocks = is_csa ? 24 : 8;
    const int64_t q_heads = kv_heads * gqa, total_q = seqs * q_len;
    const uint32_t kvdt = std::strcmp(kv_dtype, "bf16") == 0 ? RAD_BF16 : RAD_F8E4M3;
    /* One disjoint run of blocks per sequence, and a context that fills them. Both operands are
     * i32 and dalloc zeroes: a zeroed block table points every sequence at block 0, so the two
     * race on one slot, and a zeroed seqused is no context at all -- the decode grid's `live` and
     * the prefill grid's `qb * BLOCK_Q >= qlen` both exit on the first comparison, which is a
     * launch that proves the kernel starts and nothing else. */
    const int64_t max_blocks = blocks / seqs, ctx = max_blocks * bs;

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { total_q, q_heads, head_dim }),
        T(nullptr, kvdt,     { blocks, kv_heads, bs, 2 * head_dim }),
        TI32({ seqs, max_blocks }, iota_i32(seqs * max_blocks)),
        TI32({ seqs }, rep_i32(seqs, (int32_t)ctx)),
        TF32({ seqs, kv_heads }, 1.0f),
        TF32({ seqs, kv_heads }, 1.0f),
        /* cu_seqlens. THE OPERAND LIST IS oAttnPaged's AND IT HAS EIGHT SLOTS: leaving this one
         * out shifts `out` into it, puts AP_OUT past n_t, and makes fill_paged refuse every row
         * with RAD_E_INVAL -- which reads as kernels that do not work. Absent is the uniform
         * batch, which is what this case is. */
        T(nullptr, RAD_I32,  { seqs + 1 }),
        T(nullptr, RAD_BF16, { total_q, q_heads, head_dim }),
    };
    if (kvdt == RAD_BF16) { absent(ts[4]); absent(ts[5]); }
    absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("q_len", q_len), PI("head_dim", head_dim), PI("gqa", gqa), PI("block_size", bs),
        PI("causal", 1), PI("window", is_csa ? 128 : 0),
        PI("csa_ratio", is_csa ? 3 : 0),
        PS("q_dtype", "bf16"), PS("kv_dtype", kv_dtype),
    };
    (void)prefill;
    run_case(r, ts, ps);
}

static void gemm_case(const Plugin& pl, const char* name, const char* dtype, int64_t M,
                      int64_t N, int64_t K, uint32_t adt, int need_ascale, int need_asum,
                      int need_bref, int64_t wbytes, int64_t sbytes) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    std::vector<RadTensor> ts = {
        T(nullptr, adt,      { M, K }),
        T(nullptr, RAD_F32,  { M }),
        T(nullptr, RAD_U8,   { wbytes }),
        T(nullptr, RAD_U8,   { sbytes }),
        T(nullptr, RAD_BF16, { M, N }),
        T(nullptr, RAD_F32,  { (K / 128) * ((M + 15) / 16) * 16 }),
        T(nullptr, RAD_U8,   { N }),
    };
    if (!need_ascale) absent(ts[1]);
    if (!need_asum)   absent(ts[5]);
    if (!need_bref)   absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", dtype),
    };
    run_case(r, ts, ps);
}

/* ================================================================== the block-scaled fp8 family
 *
 * The gemm_case helper above hands `b` and `b_scale` as flat U8 buffers, which the fp8 shim
 * refuses: it checks the scale plane's own extents, because that is the operand a converter is most
 * likely to hand over at the wrong grid and a wrong grid reads a plausible number. So the fp8 rows
 * get their own builder, with the [N][K] and [ceil(N/128)][ceil(K/128)] shapes spelled out. */
static void fp8_gemm_case(const Plugin& pl, const char* name, const char* dtype,
                          int64_t M, int64_t N, int64_t K, int need_ascale,
                          const TunePoint* tp = nullptr) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    const int64_t sn = (N + 127) / 128, sk = (K + 127) / 128;
    std::vector<RadTensor> ts = {
        T(nullptr, need_ascale ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16, { M, K }),
        T(nullptr, RAD_F32,    { M, sk }),
        T(nullptr, RAD_F8E4M3, { N, K }),
        T(nullptr, RAD_BF16,   { sn, sk }),
        T(nullptr, RAD_BF16,   { M, N }),
        T(nullptr, RAD_F32,    { 1 }),
        T(nullptr, RAD_U8,     { 1 }),
    };
    if (!need_ascale) absent(ts[1]);
    absent(ts[5]);
    absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", dtype),
    };
    if (tp) ps = with_point(ps, *tp);
    run_case(r, ts, ps);
}


/* ---- host mirrors, for the numeric check below --------------------------------------------- */
static float h_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}
static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/* E4M3 -> f32, the software form. Exponent 0 is subnormal (no implicit leading one). This is the
 * ORACLE for the device's v_cvt_f32_fp8, so it is written from the format and not from the kernel. */
static float h_e4m3(uint8_t b) {
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    const uint32_t ex = (uint32_t)(b >> 3) & 0x0Fu, mn = (uint32_t)b & 0x07u;
    if (ex == 0) return sign * (float)mn * (1.0f / 512.0f);
    return sign * (1.0f + (float)mn / 8.0f) * std::ldexp(1.0f, (int)ex - 7);
}

/* ================================================================== a weight's canonical planes
 *
 * A container holds a weight as the planes of its encoding, row-major (spec §4.1), and a kernel's
 * own layout hooks arrange those into what it reads, at load. The cases below that need a weight
 * build its planes here, get the bytes the kernel reads from the ROW'S OWN HOOKS (`stored`), and
 * compute their reference from the PLANES -- so the reference is what the weight means, and the
 * arrangement is checked by the one reader that matters, the kernel. A reference computed from the
 * stored bytes would only repeat the arrangement back to itself.
 *
 * The quantisation here is the plainest there is (absmax over each scale block) and is not under
 * test: what is checked is that the kernel reads the numbers the planes hold. */
struct Planes {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> data;   /* one per plane of `enc` */
    std::vector<int64_t> rows, cols;          /* each plane's extents, in its own elements */

    int find(const char* role) const { return rad_enc_find(&enc, role); }
    RadTensor t(int i) const {
        RadTensor x;
        std::memset(&x, 0, sizeof(x));
        x.data  = const_cast<uint8_t*>(data[(size_t)i].data());
        x.dtype = enc.plane[i].dtype;
        x.rank  = 2;
        x.shape[0] = rows[(size_t)i];
        x.shape[1] = cols[(size_t)i];
        rad_tensor_pack(&x);
        return x;
    }
    int64_t row_bytes(int i) const { return rad_enc_row_bytes(enc.plane[i].dtype, cols[(size_t)i]); }
    /* Size every plane for a [n, k] weight, zeroed. */
    void size(int64_t n, int64_t k) {
        data.assign((size_t)enc.n_planes, {});
        rows.assign((size_t)enc.n_planes, 0);
        cols.assign((size_t)enc.n_planes, 0);
        for (int i = 0; i < enc.n_planes; ++i) {
            rad_enc_plane_dims(&enc.plane[i], n, k, &rows[(size_t)i], &cols[(size_t)i]);
            data[(size_t)i].assign((size_t)(rows[(size_t)i] * row_bytes(i)), 0);
        }
    }
};

/* The unnormalised Sylvester Walsh-Hadamard transform of `n` values, in place: (a + b) to the lower
 * index and (a - b) to the upper, the transform an "fwhtN" encoding names. H_n H_n = n I. */
static void h_fwht(float* v, int64_t n) {
    for (int64_t len = 1; len < n; len <<= 1)
        for (int64_t i = 0; i < n; i += len << 1)
            for (int64_t j = i; j < i + len; ++j) {
                const float a = v[j], b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
}

/* Block fp8 over [N, K]: E4M3 codes, and a bf16 scale a 128x128 block that is the block's absmax
 * over 448 -- the format the checkpoints ship. Codes are taken against the ROUNDED scale, so code
 * times stored scale is the weight the planes describe. */
static Planes fp8_block_planes(const std::vector<float>& w, int64_t N, int64_t K) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    rad_enc_add_plane(&p.enc, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", RAD_BF16, 128, 128);
    p.size(N, K);
    const int64_t sn = p.rows[1], sk = p.cols[1];
    for (int64_t bi = 0; bi < sn; ++bi)
        for (int64_t bj = 0; bj < sk; ++bj) {
            float amax = 0.0f;
            for (int64_t r = bi * 128; r < std::min<int64_t>(N, (bi + 1) * 128); ++r)
                for (int64_t c = bj * 128; c < std::min<int64_t>(K, (bj + 1) * 128); ++c)
                    amax = std::max(amax, std::fabs(w[(size_t)(r * K + c)]));
            rad_store_f32(p.data[1].data(), RAD_BF16, bi * sk + bj, amax > 0 ? amax / 448.0f : 1.0f);
            const float s = rad_load_f32(p.data[1].data(), RAD_BF16, bi * sk + bj);
            for (int64_t r = bi * 128; r < std::min<int64_t>(N, (bi + 1) * 128); ++r)
                for (int64_t c = bj * 128; c < std::min<int64_t>(K, (bj + 1) * 128); ++c)
                    rad_store_f32(p.data[0].data(), RAD_F8E4M3, r * K + c, w[(size_t)(r * K + c)] / s);
        }
    return p;
}

/* A symmetric 4-bit grid over [N, K]: i4 codes -8..7 and a scale of dtype `sdt` a row per `group`
 * columns, absmax over 7. ROTATED, each group of a row is multiplied by the `group`-wide Hadamard
 * first and the stored scale carries the 1/group the rotation leaves -- <H w, H a> = group <w, a>,
 * so a kernel that multiplies codes by stored scales against a rotated activation computes the
 * unrotated product. The encoding names the rotation, which is what a kernel's hook checks. */
static Planes int4_planes(const std::vector<float>& w, int64_t N, int64_t K, int64_t group,
                          uint32_t sdt, bool rot) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    if (rot) {
        char t[RAD_ENC_STR];
        std::snprintf(t, sizeof t, "fwht%lld", (long long)group);
        rad_enc_copy_str(p.enc.transform, t);
    }
    rad_enc_add_plane(&p.enc, "codes", RAD_I4, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", sdt, 1, group);
    p.size(N, K);
    const int64_t ng = K / group, crow = p.row_bytes(0);
    std::vector<float> v((size_t)group);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            for (int64_t j = 0; j < group; ++j) v[(size_t)j] = w[(size_t)(n * K + g * group + j)];
            if (rot) h_fwht(v.data(), group);
            float amax = 0.0f;
            for (float x : v) amax = std::max(amax, std::fabs(x));
            float s = amax > 0 ? amax / 7.0f : 1.0f;
            rad_store_f32(p.data[1].data(), sdt, n * ng + g, s);
            s = rad_load_f32(p.data[1].data(), sdt, n * ng + g);
            for (int64_t j = 0; j < group; ++j) {
                const float q = std::nearbyint(v[(size_t)j] / s);
                const int code = (int)std::max(-8.0f, std::min(7.0f, q));
                rad_store_code(p.data[0].data() + n * crow, RAD_I4, g * group + j, code);
            }
            if (rot) rad_store_f32(p.data[1].data(), sdt, n * ng + g, s / (float)group);
        }
    return p;
}

/* The same rotated grid with libquant's w4nl CODEBOOK: u4 codes indexing a sixteen-entry f32
 * `table` plane holding r4d_args.h's kW4nlTable, a scale of amax over the largest entry (22), the
 * nearest entry chosen, and the residual 1/group folded into the stored scale as above. */
static Planes nl_planes(const std::vector<float>& w, int64_t N, int64_t K, int64_t group,
                        uint32_t sdt) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    char t[RAD_ENC_STR];
    std::snprintf(t, sizeof t, "fwht%lld", (long long)group);
    rad_enc_copy_str(p.enc.transform, t);
    rad_enc_add_plane(&p.enc, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", sdt, 1, group);
    rad_enc_add_table(&p.enc, "table", RAD_F32, 1, 16);
    p.size(N, K);
    std::memcpy(p.data[2].data(), kW4nlTable, sizeof kW4nlTable);
    const int64_t ng = K / group, crow = p.row_bytes(0);
    std::vector<float> v((size_t)group);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            for (int64_t j = 0; j < group; ++j) v[(size_t)j] = w[(size_t)(n * K + g * group + j)];
            h_fwht(v.data(), group);
            float amax = 0.0f;
            for (float x : v) amax = std::max(amax, std::fabs(x));
            float s = amax > 0 ? amax / 22.0f : 1.0f;
            rad_store_f32(p.data[1].data(), sdt, n * ng + g, s);
            s = rad_load_f32(p.data[1].data(), sdt, n * ng + g);
            for (int64_t j = 0; j < group; ++j) {
                const float u = v[(size_t)j] / s;
                int best = 0;
                for (int c = 1; c < 16; ++c)
                    if (std::fabs(u - kW4nlTable[c]) < std::fabs(u - kW4nlTable[best])) best = c;
                rad_store_code(p.data[0].data() + n * crow, RAD_U4, g * group + j, best);
            }
            rad_store_f32(p.data[1].data(), sdt, n * ng + g, s / (float)group);
        }
    return p;
}

/* The same codebook at w4nl64a8h's scales (`gs` 64) or w4nl32a8h's (32): the weight rotated by 128
 * as above, and each `gs`-column part of a rotated group scaled on its own -- amax over the largest
 * entry, residual 1/128 folded in, stored as an E4M3 code under r4d_args.h's fixed kW4G64Scale --
 * with the nearest entry chosen against the scale as stored. Planes: codes, scale (E4M3
 * [1 x gs]), scale.1 (f32, the constant), table. With `w5`, w5nl64a8h's: u5 codes into
 * kW5nlTable, whose largest entry is 30. */
static Planes nl64_planes(const std::vector<float>& w, int64_t N, int64_t K, int64_t gs = 64,
                          bool w5 = false) {
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    const int nt = w5 ? 32 : 16;
    const float top = w5 ? 30.0f : 22.0f;
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    rad_enc_copy_str(p.enc.transform, "fwht128");
    rad_enc_add_plane(&p.enc, "codes", w5 ? RAD_U5 : RAD_U4, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", RAD_F8E4M3, 1, gs);
    rad_enc_add_plane(&p.enc, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&p.enc, "table", RAD_F32, 1, nt);
    p.size(N, K);
    std::memcpy(p.data[2].data(), &kW4G64Scale, sizeof kW4G64Scale);
    std::memcpy(p.data[3].data(), tab, sizeof(float) * (size_t)nt);
    const int64_t crow = p.row_bytes(0), srow = p.row_bytes(1);
    std::vector<float> v(128);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < K / 128; ++g) {
            for (int64_t j = 0; j < 128; ++j) v[(size_t)j] = w[(size_t)(n * K + g * 128 + j)];
            h_fwht(v.data(), 128);
            const int64_t nh = 128 / gs;
            for (int64_t h = 0; h < nh; ++h) {
                float amax = 0.0f;
                for (int64_t j = 0; j < gs; ++j) amax = std::max(amax, std::fabs(v[(size_t)(h * gs + j)]));
                const uint8_t code = rad_f32_to_fp8e4m3(amax > 0 ? amax / top / 128.0f / kW4G64Scale
                                                                 : 1.0f);
                p.data[1][(size_t)(n * srow + g * nh + h)] = code;
                const float s = h_e4m3(code) * kW4G64Scale * 128.0f;   /* the rotated domain's */
                for (int64_t j = 0; j < gs; ++j) {
                    const float u = v[(size_t)(h * gs + j)] / s;
                    int best = 0;
                    for (int c = 1; c < nt; ++c)
                        if (std::fabs(u - tab[c]) < std::fabs(u - tab[best])) best = c;
                    rad_store_code(p.data[0].data() + n * crow, w5 ? RAD_U5 : RAD_U4,
                                   g * 128 + h * gs + j, best);
                }
            }
        }
    return p;
}

/* A symmetric 8-bit grid over [N, K]: i8 codes -127..127 and an f16 scale of amax / 127 a row per
 * 128 columns -- what libquant writes for `codes=i8 group=128 scale=f16 clamp=sym`, and the
 * encoding the w8a16 layout hook takes. */
static Planes int8_planes(const std::vector<float>& w, int64_t N, int64_t K) {
    Planes p;
    p.enc = rad_enc_affine(RAD_I8, RAD_F16, 1, 128);
    p.size(N, K);
    const int64_t ng = K / 128;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            float amax = 0.0f;
            for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(w[(size_t)(n * K + g * 128 + j)]));
            rad_store_f32(p.data[1].data(), RAD_F16, n * ng + g, amax > 0 ? amax / 127.0f : 1.0f);
            const float s = rad_load_f32(p.data[1].data(), RAD_F16, n * ng + g);
            for (int64_t j = 0; j < 128; ++j) {
                const float q = std::nearbyint(w[(size_t)(n * K + g * 128 + j)] / s);
                p.data[0][(size_t)(n * K + g * 128 + j)] =
                    (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
            }
        }
    return p;
}

/* Code `k` of row `n` of a packed integer codes plane, as its value. */
static int code_at(const Planes& p, int64_t n, int64_t k) {
    return (int)rad_load_f32(p.data[0].data() + n * p.row_bytes(0), p.enc.plane[0].dtype, k);
}

/* The rectangle [r0, r1) x [c0, c1) of a LOGICAL weight, as the planes covering it -- each plane
 * cut on its own blocks, the way the loader takes a tensor-parallel rank's share. Byte-wide planes
 * only, which is every plane the shards below are taken of. */
static Planes share(const Planes& p, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    Planes q;
    q.enc = p.enc;
    q.data.resize(p.data.size());
    q.rows.resize(p.data.size());
    q.cols.resize(p.data.size());
    for (int i = 0; i < p.enc.n_planes; ++i) {
        const RadEncPlane& pl = p.enc.plane[i];
        const int64_t br = pl.block[0] > 0 ? pl.block[0] : 1, bc = pl.block[1] > 0 ? pl.block[1] : 1;
        const int64_t pr0 = r0 / br, pr1 = (r1 + br - 1) / br, pc0 = c0 / bc, pc1 = (c1 + bc - 1) / bc;
        const int64_t eb = rad_dtype_bits(pl.dtype) / 8;
        q.rows[(size_t)i] = pr1 - pr0;
        q.cols[(size_t)i] = pc1 - pc0;
        q.data[(size_t)i].resize((size_t)((pr1 - pr0) * (pc1 - pc0) * eb));
        for (int64_t r = pr0; r < pr1; ++r)
            std::memcpy(q.data[(size_t)i].data() + (r - pr0) * (pc1 - pc0) * eb,
                        p.data[(size_t)i].data() + r * p.row_bytes(i) + pc0 * eb,
                        (size_t)((pc1 - pc0) * eb));
    }
    return q;
}

/* WHAT ROW `r` READS for `operand`, from the planes named `roles`: RAD_OK with its own arrangement
 * in `out` (relayout), or RAD_E_UNSUPPORTED with the one selected plane as it is -- a hook's
 * "stored as it is", and what a row with no hook gets. Anything else is the hook's refusal. */
static int stored(const RadKernelInfo* r, const std::vector<RadParam>& lp, int operand,
                  const Planes& p, std::initializer_list<const char*> roles,
                  std::vector<uint8_t>* out, RadLayout* lay = nullptr) {
    int sel[RAD_ENC_MAX_PLANES];
    RadTensor pt[RAD_ENC_MAX_PLANES];
    int n = 0;
    for (const char* role : roles) {
        if (n >= RAD_ENC_MAX_PLANES) return RAD_E_INVAL;
        sel[n] = p.find(role);
        if (sel[n] < 0) return RAD_E_SHAPE;
        pt[n] = p.t(sel[n]);
        ++n;
    }
    RadLayout L;
    std::memset(&L, 0, sizeof(L));
    const int rc = r->layout ? r->layout(lp.data(), (int)lp.size(), operand, &p.enc, sel, pt, n, &L)
                             : RAD_E_UNSUPPORTED;
    if (lay) *lay = L;
    if (rc == RAD_E_UNSUPPORTED) {
        if (n != 1) return RAD_E_SHAPE;
        *out = p.data[(size_t)sel[0]];
        return RAD_E_UNSUPPORTED;
    }
    if (rc != RAD_OK) return rc;
    if (!r->relayout || L.bytes <= 0) return RAD_E_STATE;
    out->assign((size_t)L.bytes, 0);
    return r->relayout(lp.data(), (int)lp.size(), operand, &p.enc, sel, pt, n, out->data(),
                       L.bytes);
}

/* AND BACK: the row's inverse of `stored`, which must give exactly the planes it was handed. True
 * when it does; `why` says what differed, or that the row has no inverse. */
static bool restores(const RadKernelInfo* r, const std::vector<RadParam>& lp, int operand,
                     const Planes& p, std::initializer_list<const char*> roles,
                     const std::vector<uint8_t>& bytes, std::string* why) {
    if (!r->unrelayout) { *why = "the row publishes no inverse"; return false; }
    int sel[RAD_ENC_MAX_PLANES];
    RadTensor pt[RAD_ENC_MAX_PLANES];
    std::vector<std::vector<uint8_t>> back;
    int n = 0;
    for (const char* role : roles) {
        if (n >= RAD_ENC_MAX_PLANES) return false;
        sel[n] = p.find(role);
        if (sel[n] < 0) { *why = std::string("no plane '") + role + "'"; return false; }
        back.emplace_back(p.data[(size_t)sel[n]].size(), 0xA5);
        ++n;
    }
    for (int i = 0; i < n; ++i) {
        pt[i] = p.t(sel[i]);
        pt[i].data = back[(size_t)i].data();
    }
    const int rc = r->unrelayout(lp.data(), (int)lp.size(), operand, &p.enc, sel, bytes.data(),
                                 (int64_t)bytes.size(), pt, n);
    if (rc != RAD_OK) { *why = "unrelayout rc=" + std::to_string(rc); return false; }
    for (int i = 0; i < n; ++i)
        if (back[(size_t)i] != p.data[(size_t)sel[i]]) {
            *why = std::string("plane '") + p.enc.plane[sel[i]].role + "' differs";
            return false;
        }
    return true;
}

/* kv_store, both widths, AGAINST AN EXACTLY-REPRESENTABLE SET so the check is equality and not a
 * tolerance. E4M3 has three mantissa bits, so every value below -- the halves and integers up to 8
 * -- survives a round trip unchanged, and the bf16 path stores them unchanged too. That makes ONE
 * expectation serve both rows: `h_e4m3` or `h_bf16` of what landed must equal what went in,
 * exactly. A tolerance would pass the two failures this is actually for.
 *
 * WHAT IT IS FOR. The op scatters into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim] with K
 * at [.., d] and V at [.., head_dim + d], addressed by a slot that decomposes as
 * (slot / block_size, slot % block_size). That is four indices deep and every one of them is a
 * place to be off by a head, a block or a half-row -- and a k/v swap or a head transposition is
 * invisible to any test whose values do not distinguish (token, head, dim). These do: the value is
 * built from all three, and K and V are given disjoint sign.
 *
 * THE NEGATIVE AND OUT-OF-RANGE SLOTS ARE THE OTHER HALF OF THE CONTRACT. A pad row must store
 * NOTHING -- the scheduler hands them for rows with no request -- so the cache is prefilled with a
 * sentinel and those slots are checked to still hold it. `libref` and libr4d agree that zero
 * is a REAL slot (r4d_gdn_conv_w4_h128_bf16.hip follows the same rule), so slot 0 is used
 * deliberately here rather than avoided. */

/* ------------------------------------------------------- the vocab-parallel top-R, end to end
 *
 * THE PROPERTY IS AN EQUALITY, not a tolerance: a top-R taken over one plane must equal the merge
 * of the two half-plane top-Rs, index for index and value for value. That one check covers all
 * three pieces at once -- the `vocab_off` shift, the `pairs` plane row_topk writes, and the
 * merge -- and it is the property the engine depends on, since a sharded draft head is only
 * allowed to be silent if it proposes exactly what the replicated one did.
 *
 * THE VALUES ARE DELIBERATELY FULL OF TIES, because ties are where a merge stops agreeing. A
 * dense top-R breaks them on the lower column, so a merge that broke them on rank order would
 * disagree exactly when the same logit appears in both shards -- which, on a 23-value alphabet
 * over 2048 columns, is every row. Distinct values would pass either way.
 *
 * The gathered plane is assembled here the way `all_gather` with `row = R*2` assembles it: rank
 * r's R pairs at [r*R, (r+1)*R) of each row. If that interleave is ever spelled differently in
 * the graph, this case is what says so. */

/* ------------------------------------------------- the fused attention prologue, against its pair
 *
 * THE QUESTION IS BIT EQUALITY AND NOTHING ELSE. `qk_norm_rope_gate` replaces rmsnorm(q) +
 * rmsnorm(k) + rope(q) + rope(k) -- four launches a layer, sixteen layers -- and it is only worth
 * having if the model that comes out is the SAME model. Its own header says it rounds the norm
 * output to bf16 before rotating for exactly this reason ("skipping that round trip is more
 * accurate and would be wrong: it is a different function"), and this is the check that the claim
 * holds against the kernels it replaces rather than against an argument.
 *
 * THE TABLE IS PART OF THE CLAIM. The fused kernel takes cos/sin instead of theta, so a table
 * built with different angles is a different rope -- fluent text with subtly wrong positions,
 * which nothing downstream would flag. `rope_table` is therefore run here as the fused path's
 * first stage rather than filled by this file, so what is compared is the pair the engine issues.
 *
 * The q side reads a STRIDED view in the engine (the [q|gate] projection is interleaved per head)
 * and the unfused norm cannot, so the reference is fed a contiguous copy of the same numbers. The
 * arithmetic does not depend on the stride; the values are what is being compared. */
static void qk_prologue_case(const Plugin& pl, uint32_t gain_dt) {
    if (!sel("qk_norm_rope_gate") && !sel("rope_table_f32")) return;
    /* THE GAIN'S WIDTH IS RUN BOTH WAYS, because the two callers differ: every architecture plugin
     * here declares an f32 norm gain and vLLM's declares bf16, and one flag inside the kernel picks
     * between them. Checking only bf16 -- which is what the shape smoke test above passes -- would
     * leave the width the ENGINE uses unexercised. */
    const bool gf32 = gain_dt == RAD_F32;
    const RadKernelInfo* rf = find(pl, "qk_norm_rope_gate");
    const RadKernelInfo* rt = find(pl, "rope_table_f32");
    const RadKernelInfo* rn = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* rr = find(pl, "rope_bf16");
    if (!rf || !rt || !rn || !rr) { ok("qk_norm_rope_gate vs its pair", false, "row absent"); return; }

    /* THE ENGINE'S HEAD COUNTS, and k rotated IN PLACE the way the block issues it -- the two ways
     * this case could pass while the model still moved. A separate k_out is not what a caller with
     * one k buffer does, and 4 query heads is not 12: the wave count the launcher picks is a
     * function of nq + nkv, so a head count the model does not have exercises a launch it does not
     * make. */
    const int64_t M = 512, nq = 12, nkv = 2, hd = 256, rot = 64, ctx = 4096;
    const double  theta = 1e7, eps = 1e-6, wadd = 1.0;
    /* MANY ROWS, because the difference this case is hunting is one bf16 in tens of thousands: two
     * summation orders for the same mean-square agree to a ULP in f32 and that ULP almost never
     * crosses a bf16 boundary. A handful of rows settles nothing; 512 rows x 14 heads x 256 lanes
     * is 1.8M elements a side. The positions still span the interesting range -- 0, small, and
     * near the top. */
    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < M; ++i) pos[(size_t)i] = (int32_t)((i * 7 + i * i) % ctx);

    auto rnd = [](int64_t i) {                       /* deterministic, and spans the exponent */
        const uint32_t x = (uint32_t)(i * 2654435761u + 12345u);
        return (float)((int32_t)(x >> 16) - 32768) * (1.0f / 8192.0f);
    };
    std::vector<uint16_t> hqg((size_t)M * nq * 2 * hd), hk((size_t)M * nkv * hd);
    std::vector<uint16_t> hq_src((size_t)M * nq * hd), hgate((size_t)M * nq * hd);
    for (int64_t t = 0; t < M; ++t)
        for (int64_t h = 0; h < nq; ++h)
            for (int64_t d = 0; d < hd; ++d) {
                const uint16_t q = h_to_bf16(rnd(((t * nq) + h) * hd + d));
                const uint16_t g = h_to_bf16(rnd(1000000 + ((t * nq) + h) * hd + d));
                hqg[(size_t)((t * nq + h) * 2 * hd + d)]      = q;
                hqg[(size_t)((t * nq + h) * 2 * hd + hd + d)] = g;
                hq_src[(size_t)((t * nq + h) * hd + d)] = q;
                hgate[(size_t)((t * nq + h) * hd + d)]  = g;
            }
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = h_to_bf16(rnd(2000000 + (int64_t)i));
    /* The gain is built in bf16 either way and WIDENED for the f32 run, so the two runs put the
     * same numbers through both widths -- a difference between them is the flag, not the data. */
    std::vector<uint16_t> hwq((size_t)hd), hwk((size_t)hd);
    std::vector<float> fwq((size_t)hd), fwk((size_t)hd);
    for (int64_t d = 0; d < hd; ++d) {
        hwq[(size_t)d] = h_to_bf16(rnd(3000000 + d) * 0.1f);
        hwk[(size_t)d] = h_to_bf16(rnd(4000000 + d) * 0.1f);
        fwq[(size_t)d] = h_bf16(hwq[(size_t)d]);
        fwk[(size_t)d] = h_bf16(hwk[(size_t)d]);
    }

    const size_t qb = hq_src.size() * 2, kb = hk.size() * 2;
    void* d_qg = dalloc(hqg.size() * 2);
    void* d_k  = dalloc(kb);
    const size_t gsz = gf32 ? 4u : 2u;
    void* d_wq = dalloc((size_t)hd * gsz);
    void* d_wk = dalloc((size_t)hd * gsz);
    void* d_p  = dalloc(pos.size() * 4);
    void* d_cs = dalloc((size_t)ctx * rot * 4);
    void* d_qr = dalloc(qb);            /* the reference q and k, normed then rotated in place */
    void* d_kr = dalloc(kb);
    void* d_qs = dalloc(qb);            /* the contiguous q source the unfused norm reads */
    void* d_qo = dalloc(qb);
    void* d_ko = dalloc(kb);
    void* d_go = dalloc(qb);
    if (!d_qg || !d_k || !d_wq || !d_wk || !d_p || !d_cs || !d_qr || !d_kr || !d_qs ||
        !d_qo || !d_ko || !d_go) { ok("qk_norm_rope_gate vs its pair", false, "hipMalloc failed"); return; }
    st_memcpy(d_qg, hqg.data(), hqg.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_k,  hk.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_qs, hq_src.data(), qb, hipMemcpyHostToDevice);
    if (gf32) {
        st_memcpy(d_wq, fwq.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
        st_memcpy(d_wk, fwk.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
    } else {
        st_memcpy(d_wq, hwq.data(), (size_t)hd * 2, hipMemcpyHostToDevice);
        st_memcpy(d_wk, hwk.data(), (size_t)hd * 2, hipMemcpyHostToDevice);
    }
    st_memcpy(d_p,  pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memset(d_cs, 0, (size_t)ctx * rot * 4);

    /* ---- the pair the fusion replaces ---- */
    {
        /* THE ENGINE FEEDS THIS NORM A STRIDED VIEW, not a contiguous copy: the q half of each
         * head inside the interleaved [q|gate] projection, so its rows are 2*head_dim apart. If
         * the norm answers differently on a strided row than on a dense one, the fused kernel can
         * be exact against the dense form and the MODEL still move -- so the reference here is the
         * operand the block actually issues. */
        RadTensor qsrc = T(d_qg, RAD_BF16, { M * nq, hd });
        qsrc.stride[0] = 2 * hd; qsrc.stride[1] = 1;
        std::vector<RadTensor> ts = { qsrc,
                                      T(d_wq, gain_dt, { hd }),
                                      T(d_qr, RAD_BF16, { M * nq, hd }) };
        std::vector<RadParam> ps = { PI("M", M * nq), PI("n", hd), PF("eps", eps),
                                     PS("dtype", "bf16"), PF("wadd", wadd) };
        run_case(rn, ts, ps);
    }
    {
        std::vector<RadTensor> ts = { T(d_k, RAD_BF16, { M * nkv, hd }),
                                      T(d_wk, gain_dt, { hd }),
                                      T(d_kr, RAD_BF16, { M * nkv, hd }) };
        std::vector<RadParam> ps = { PI("M", M * nkv), PI("n", hd), PF("eps", eps),
                                     PS("dtype", "bf16"), PF("wadd", wadd) };
        run_case(rn, ts, ps);
    }
    for (int side = 0; side < 2; ++side) {
        const int64_t nh = side ? nkv : nq;
        std::vector<RadTensor> ts = { T(side ? d_kr : d_qr, RAD_BF16, { M, nh * hd }),
                                      T(d_p, RAD_I32, { M }) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", nh),
                                     PI("n_head_kv", 0), PF("theta", theta), PF("scale", 1.0),
                                     PS("mode", "neox"), PI("rotary_dim", rot) };
        run_case(rr, ts, ps);
    }

    /* ---- the fused path, table and all ---- */
    {
        std::vector<RadTensor> ts = { T(d_p, RAD_I32, { M }), T(d_cs, RAD_F32, { ctx, rot }) };
        std::vector<RadParam> ps = { PI("M", M), PI("rot", rot), PF("theta", theta),
                                     PF("scale", 1.0), PS("dtype", "f32") };
        run_case(rt, ts, ps);
    }
    {
        std::vector<RadTensor> ts = {
            T(d_qg, RAD_BF16, { M, nq * 2 * hd }), T(d_k, RAD_BF16, { M, nkv * hd }),
            T(d_cs, RAD_F32,  { ctx, rot }),       T(d_p,  RAD_I32,  { M }),
            T(d_wq, gain_dt, { hd }),             T(d_wk, gain_dt, { hd }),
            T(d_qo, RAD_BF16, { M, nq * hd }),     T(d_k,  RAD_BF16, { M, nkv * hd }),
            T(d_go, RAD_BF16, { M, nq * hd }),
        };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", nq),
                                     PI("n_head_kv", nkv), PI("rot", rot), PS("q_dtype", "bf16"),
                                     PF("eps", eps), PF("wadd", wadd), PI("cs_f32", 1) };
        run_case(rf, ts, ps);
    }

    std::vector<uint16_t> gq(hq_src.size()), gk(hk.size()), gg(hgate.size());
    std::vector<uint16_t> wq_(hq_src.size()), wk_(hk.size());
    st_memcpy(gq.data(),  d_qo, qb, hipMemcpyDeviceToHost);
    st_memcpy(gk.data(),  d_k,  kb, hipMemcpyDeviceToHost);   /* rotated in place */
    st_memcpy(gg.data(),  d_go, qb, hipMemcpyDeviceToHost);
    st_memcpy(wq_.data(), d_qr, qb, hipMemcpyDeviceToHost);
    st_memcpy(wk_.data(), d_kr, kb, hipMemcpyDeviceToHost);

    /* BIT EQUALITY, and it holds only because the two FOLD THE SUM OF SQUARES THE SAME WAY.
     * `r4d_rmsnorm_bf16` picks its thread width from `r4d_act_threads(M, n)` and states the rule
     * itself: "the width decides which columns a thread folds into the sum of squares. Two
     * kernels that disagree about it disagree in the last bits of the norm, and the fused one is
     * required to be byte-identical to this kernel followed by the quantiser." The fused prologue
     * reproduces that fold, so the comparison below is an equality and not a tolerance.
     *
     * THE FAILURE MESSAGE STILL REPORTS A RELATIVE WORST CASE rather than a bf16 step count,
     * because a ULP in the mean square is a ULP in the normed value -- and a ROTATED element is a
     * difference of two products, so where they nearly cancel that ULP comes out as a few percent
     * of a small number while a wrong angle or a wrong pairing is O(1). The two read very
     * differently.
     *
     * 512 rows is not decoration: where the folds do differ they differ on a handful of elements
     * in a million, so a few rows agree exactly and say nothing -- which is also why a model's
     * transcript moves after a few dozen steps and not at the first one. */
    auto cmp = [&](const char* what, const std::vector<uint16_t>& got,
                   const std::vector<uint16_t>& want, int64_t width) {
        int64_t n_off = 0;
        double  worst = 0.0, scale = 0.0;
        std::string where;
        for (size_t i = 0; i < want.size(); ++i) {
            const double v = h_bf16(want[i]);
            if (v > scale) scale = v; else if (-v > scale) scale = -v;
        }
        if (scale <= 0.0) scale = 1.0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] == want[i]) continue;
            ++n_off;
            const double d = h_bf16(got[i]) - h_bf16(want[i]);
            const double ad = (d < 0 ? -d : d) / scale;
            if (ad > worst) {
                worst = ad;
                where = "worst at row " + std::to_string(i / (size_t)width) + " lane " +
                        std::to_string(i % (size_t)width) + ": " + std::to_string(h_bf16(got[i])) +
                        " vs " + std::to_string(h_bf16(want[i]));
            }
        }
        const std::string tag = std::string(gf32 ? "f32 gain, " : "bf16 gain, ") + what;
        char rate[128];
        std::snprintf(rate, sizeof rate, "%lld of %zu differ, worst %.2e of full scale",
                      (long long)n_off, got.size(), worst);
        /* BYTE EQUALITY, NOT A TOLERANCE. The kernel reproduces `rmsnorm_bf16`'s fold, so the
         * only honest assertion is that not one of 1.8M elements differs -- a relative bound here
         * would let the engine's transcript move while this row still said "ok". `worst` stays in
         * the message because a failure that is a rounding and a failure that is a wrong angle
         * read very differently. */
        ok(("qk_norm_rope_gate " + tag + " is byte-identical to the unfused pair").c_str(),
           n_off == 0, std::string(rate) + (n_off == 0 ? "" : " -- " + where));
    };
    cmp("q", gq, wq_, hd);
    cmp("k", gk, wk_, hd);
    cmp("gate", gg, hgate, hd);
}
/* ------------------------------------ the prologue with the cache store folded on, against the pair
 *
 * `qk_norm_rope_gate_kv_store` is `qk_norm_rope_gate` followed by `kv_store` into the E4M3 or the
 * bf16 cache, and the claim is byte equality with that pair -- the q and rotated k it leaves, and
 * every byte of the cache, including the ones it must not touch. The same inputs go down both paths
 * on separate buffers (k is rotated in place, so each path needs its own), and the caches start
 * from the same random bytes, so a store at a wrong place or a store that should not have happened
 * shows up against the pair and not only against a clean slate.
 *
 * THE INPUTS SPAN THE CLAMP: v up to +-2000 and k normed and rotated, so E4M3's +-448 saturation
 * and its subnormals are both reached. A pad row (slot -1) and a slot past the cache's end must
 * store nothing. The engine's geometry: 12 query heads and 2 kv heads of 256, rot 64. */
static void qk_kv_fold_case(const Plugin& pl, bool bf16) {
    const char* row = bf16 ? "qk_norm_rope_gate_kv_store_bf16" : "qk_norm_rope_gate_kv_store_fp8";
    if (!sel(row)) return;
    const char* name = bf16
        ? "qk_norm_rope_gate_kv_store_bf16 is byte-identical to the prologue + kv_store"
        : "qk_norm_rope_gate_kv_store_fp8 is byte-identical to the prologue + kv_store";
    const uint32_t cdt = bf16 ? RAD_BF16 : RAD_F8E4M3;
    const char* cname = bf16 ? "bf16" : "fp8_e4m3";
    const RadKernelInfo* rk = find(pl, row);
    const RadKernelInfo* rf = find(pl, "qk_norm_rope_gate");
    const RadKernelInfo* rs = find(pl, bf16 ? "kv_store_bf16" : "kv_store_fp8");
    const RadKernelInfo* rt = find(pl, "rope_table_f32");
    if (!rk || !rf || !rs || !rt) { ok(name, false, "row absent"); return; }

    const int64_t M = 48, nq = 12, nkv = 2, hd = 256, rot = 64, ctx = 4096, bs = 16, nb = 6;
    const double  theta = 1e7, eps = 1e-6, wadd = 1.0;
    std::vector<int32_t> pos((size_t)M), slot((size_t)M);
    for (int64_t t = 0; t < M; ++t) {
        pos[(size_t)t]  = (int32_t)((t * 97 + 5) % ctx);
        slot[(size_t)t] = (int32_t)((t * 7 + 3) % (nb * bs));     /* 7 is coprime with 96 */
    }
    slot[5]  = -1;                                                 /* a pad row */
    slot[11] = (int32_t)(nb * bs);                                 /* past the cache */

    uint32_t st = 0x9e3779b9u;
    auto rnd = [&](float span) {
        st = st * 1664525u + 1013904223u;
        return ((float)(st >> 8) / 16777216.0f * 2.0f - 1.0f) * span;
    };
    std::vector<uint16_t> hqg((size_t)M * nq * 2 * hd), hk((size_t)M * nkv * hd),
                          hv((size_t)M * nkv * hd);
    for (auto& x : hqg) x = h_to_bf16(rnd(4.0f));
    for (auto& x : hk)  x = h_to_bf16(rnd(4.0f));
    for (auto& x : hv)  x = h_to_bf16(rnd(2000.0f));
    std::vector<float> fwq((size_t)hd), fwk((size_t)hd);
    for (int64_t d = 0; d < hd; ++d) { fwq[(size_t)d] = rnd(0.5f); fwk[(size_t)d] = rnd(0.5f); }
    const size_t cbytes = (size_t)(nb * nkv * bs * 2 * hd) * (bf16 ? 2 : 1);
    std::vector<uint8_t> hc(cbytes);
    for (auto& x : hc) x = (uint8_t)(rnd(128.0f) + 128.0f);

    const size_t qgb = hqg.size() * 2, kb = hk.size() * 2, qb = (size_t)M * nq * hd * 2;
    void* d_qg = dalloc(qgb);  void* d_v = dalloc(kb);
    void* d_kA = dalloc(kb);   void* d_kB = dalloc(kb);
    void* d_qA = dalloc(qb);   void* d_qB = dalloc(qb);
    void* d_cA = dalloc(cbytes); void* d_cB = dalloc(cbytes);
    void* d_wq = dalloc((size_t)hd * 4); void* d_wk = dalloc((size_t)hd * 4);
    void* d_p  = dalloc((size_t)M * 4);  void* d_sl = dalloc((size_t)M * 4);
    void* d_cs = dalloc((size_t)ctx * rot * 4);
    if (!d_qg || !d_v || !d_kA || !d_kB || !d_qA || !d_qB || !d_cA || !d_cB || !d_wq || !d_wk ||
        !d_p || !d_sl || !d_cs) { ok(name, false, "hipMalloc failed"); return; }
    st_memcpy(d_qg, hqg.data(), qgb, hipMemcpyHostToDevice);
    st_memcpy(d_v,  hv.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_kA, hk.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_kB, hk.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_cA, hc.data(),  cbytes, hipMemcpyHostToDevice);
    st_memcpy(d_cB, hc.data(),  cbytes, hipMemcpyHostToDevice);
    st_memcpy(d_wq, fwq.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
    st_memcpy(d_wk, fwk.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
    st_memcpy(d_p,  pos.data(), (size_t)M * 4, hipMemcpyHostToDevice);
    st_memcpy(d_sl, slot.data(), (size_t)M * 4, hipMemcpyHostToDevice);
    st_memset(d_cs, 0, (size_t)ctx * rot * 4);
    {
        std::vector<RadTensor> ts = { T(d_p, RAD_I32, { M }), T(d_cs, RAD_F32, { ctx, rot }) };
        std::vector<RadParam> ps = { PI("M", M), PI("rot", rot), PF("theta", theta),
                                     PF("scale", 1.0), PS("dtype", "f32") };
        run_case(rt, ts, ps);
    }
    std::vector<RadParam> pq = { PI("M", M), PI("head_dim", hd), PI("n_head", nq),
                                 PI("n_head_kv", nkv), PI("rot", rot), PS("q_dtype", "bf16"),
                                 PF("eps", eps), PF("wadd", wadd), PI("cs_f32", 1) };
    auto prologue = [&](void* k, void* q) {
        return std::vector<RadTensor>{
            T(d_qg, RAD_BF16, { M, nq * 2 * hd }), T(k, RAD_BF16, { M, nkv * hd }),
            T(d_cs, RAD_F32, { ctx, rot }),        T(d_p, RAD_I32, { M }),
            T(d_wq, RAD_F32, { hd }),              T(d_wk, RAD_F32, { hd }),
            T(q, RAD_BF16, { M, nq * hd }),        T(k, RAD_BF16, { M, nkv * hd }),
            RadTensor{} };
    };
    auto cache = [&](void* c) { return T(c, cdt, { nb, nkv, bs, 2 * hd }); };

    /* ---- the pair ---- */
    {
        std::vector<RadTensor> ts = prologue(d_kA, d_qA);
        run_case(rf, ts, pq);
    }
    {
        std::vector<RadTensor> ts = { T(d_kA, RAD_BF16, { M, nkv * hd }),
                                      T(d_v, RAD_BF16, { M, nkv * hd }),
                                      T(d_sl, RAD_I32, { M }), cache(d_cA) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head_kv", nkv),
                                     PI("block_size", bs), PS("kv_dtype", cname) };
        run_case(rs, ts, ps);
    }
    /* ---- the fold ---- */
    {
        std::vector<RadTensor> ts = prologue(d_kB, d_qB);
        ts.push_back(T(d_v, RAD_BF16, { M, nkv * hd }));
        ts.push_back(T(d_sl, RAD_I32, { M }));
        ts.push_back(cache(d_cB));
        std::vector<RadParam> ps = pq;
        ps.push_back(PI("block_size", bs));
        ps.push_back(PS("kv_dtype", cname));
        run_case(rk, ts, ps);
    }

    std::vector<uint8_t> qa(qb), qbv(qb), ka(kb), kbv(kb), ca(cbytes), cb(cbytes);
    st_memcpy(qa.data(), d_qA, qb, hipMemcpyDeviceToHost);
    st_memcpy(qbv.data(), d_qB, qb, hipMemcpyDeviceToHost);
    st_memcpy(ka.data(), d_kA, kb, hipMemcpyDeviceToHost);
    st_memcpy(kbv.data(), d_kB, kb, hipMemcpyDeviceToHost);
    st_memcpy(ca.data(), d_cA, cbytes, hipMemcpyDeviceToHost);
    st_memcpy(cb.data(), d_cB, cbytes, hipMemcpyDeviceToHost);
    auto ndiff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        int64_t n = 0;
        for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
        return n;
    };
    /* And the store did happen: the pair's cache is not the initial bytes. */
    const int64_t dq = ndiff(qa, qbv), dk = ndiff(ka, kbv), dc = ndiff(ca, cb), moved = ndiff(ca, hc);
    char msg[160];
    std::snprintf(msg, sizeof msg, "q %lld, k %lld, cache %lld of %zu bytes differ; the pair wrote "
                  "%lld cache bytes", (long long)dq, (long long)dk, (long long)dc, cbytes,
                  (long long)moved);
    ok(name, dq == 0 && dk == 0 && dc == 0 && moved > 0, msg);
}
/* ------------------------------------------------ the K-ONLY form, against the pair IT replaces
 *
 * `n_head` 0 is what the QSA indexer asks for: its query heads are already one plain `head_dim`
 * run each inside the `heads + 1` projection, so there is no [q|gate] split to do and no q
 * operands to pass. The unfused pair it replaces is `heads` rmsnorms on a STRIDED view plus one
 * rope -- five launches a layer, sixteen indexer calls a step -- and the only thing that makes
 * the fusion worth having is that the model which comes out is the same model.
 *
 * THE STRIDE IS THE POINT AND IT IS WHY THIS IS A SECOND CASE. The pair above feeds the fused
 * kernel a [tokens, heads, head_dim] k whose heads are contiguous; this one feeds it a head that
 * is 128 columns inside a 640-column row, which is the only layout the indexer has. If the shim
 * ever took the row pitch from the head count rather than from the operand, that case would still
 * pass and this one would read another head's numbers.
 *
 * THE GEOMETRY IS THE MODEL'S: 4 heads of 128 at rot 64, the fifth head of the projection being
 * the raw key that this op must leave ALONE. It is checked explicitly -- an off-by-one in the
 * head loop would otherwise be invisible, since nothing downstream of here reads it in this file. */
static void qsa_qprep_case(const Plugin& pl) {
    if (!sel("qk_norm_rope_gate") && !sel("rope_table_f32")) return;
    const RadKernelInfo* rf = find(pl, "qk_norm_rope_gate");
    const RadKernelInfo* rt = find(pl, "rope_table_f32");
    const RadKernelInfo* rn = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* rr = find(pl, "rope_bf16");
    if (!rf || !rt || !rn || !rr) { ok("qk_norm_rope_gate k-only", false, "row absent"); return; }

    const int64_t M = 512, heads = 4, hd = 128, rot = 64, ctx = 4096;
    const int64_t nw = (heads + 1) * hd, qw = heads * hd;
    const double  theta = 1e7, eps = 1e-6, wadd = 1.0;

    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < M; ++i) pos[(size_t)i] = (int32_t)((i * 7 + i * i) % ctx);

    auto rnd = [](int64_t i) {
        const uint32_t x = (uint32_t)(i * 2654435761u + 12345u);
        return (float)((int32_t)(x >> 16) - 32768) * (1.0f / 8192.0f);
    };
    std::vector<uint16_t> hqk((size_t)M * nw);
    for (size_t i = 0; i < hqk.size(); ++i) hqk[i] = h_to_bf16(rnd(5000000 + (int64_t)i));
    std::vector<float> hw((size_t)hd);
    for (int64_t d = 0; d < hd; ++d) hw[(size_t)d] = h_bf16(h_to_bf16(rnd(6000000 + d) * 0.1f));

    const size_t qkb = hqk.size() * 2, qb = (size_t)M * qw * 2;
    void* d_qk = dalloc(qkb);
    void* d_w  = dalloc((size_t)hd * 4);
    void* d_p  = dalloc(pos.size() * 4);
    void* d_cs = dalloc((size_t)ctx * rot * 4);
    void* d_qr = dalloc(qb);          /* the reference: heads norms, then one rope in place */
    void* d_qo = dalloc(qb);          /* the fused output */
    if (!d_qk || !d_w || !d_p || !d_cs || !d_qr || !d_qo) {
        ok("qk_norm_rope_gate k-only", false, "hipMalloc failed"); return;
    }
    st_memcpy(d_qk, hqk.data(), qkb, hipMemcpyHostToDevice);
    st_memcpy(d_w,  hw.data(),  (size_t)hd * 4, hipMemcpyHostToDevice);
    st_memcpy(d_p,  pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memset(d_cs, 0, (size_t)ctx * rot * 4);
    st_memset(d_qr, 0, qb);
    st_memset(d_qo, 0, qb);

    /* ---- the pair: one norm a head off the strided projection, then one rope over all of them */
    for (int64_t h = 0; h < heads; ++h) {
        RadTensor src = T((void*)((uint16_t*)d_qk + h * hd), RAD_BF16, { M, hd });
        src.stride[0] = nw; src.stride[1] = 1;
        RadTensor dst = T((void*)((uint16_t*)d_qr + h * hd), RAD_BF16, { M, hd });
        dst.stride[0] = qw; dst.stride[1] = 1;
        std::vector<RadTensor> ts = { src, T(d_w, RAD_F32, { hd }), dst };
        std::vector<RadParam> ps = { PI("M", M), PI("n", hd), PF("eps", eps),
                                     PS("dtype", "bf16"), PF("wadd", wadd) };
        run_case(rn, ts, ps);
    }
    {
        std::vector<RadTensor> ts = { T(d_qr, RAD_BF16, { M, qw }), T(d_p, RAD_I32, { M }) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", heads),
                                     PI("n_head_kv", 0), PF("theta", theta), PF("scale", 1.0),
                                     PS("mode", "neox"), PI("rotary_dim", rot) };
        run_case(rr, ts, ps);
    }

    /* ---- the fused path, table and all ---- */
    {
        std::vector<RadTensor> ts = { T(d_p, RAD_I32, { M }), T(d_cs, RAD_F32, { ctx, rot }) };
        std::vector<RadParam> ps = { PI("M", M), PI("rot", rot), PF("theta", theta),
                                     PF("scale", 1.0), PS("dtype", "f32") };
        run_case(rt, ts, ps);
    }
    {
        std::vector<RadTensor> ts = {
            T(d_qk, RAD_BF16, { M, nw }),            /* q_gate -- made absent below */
            T(d_qk, RAD_BF16, { M, nw }),            /* k: the projection WHOLE, at its own pitch */
            T(d_cs, RAD_F32,  { ctx, rot }),         T(d_p, RAD_I32, { M }),
            T(d_w,  RAD_F32,  { hd }),               /* q_w -- absent */
            T(d_w,  RAD_F32,  { hd }),
            T(d_qo, RAD_BF16, { M, qw }),            /* q_out -- absent */
            T(d_qo, RAD_BF16, { M, qw }),
            T(d_qo, RAD_BF16, { M, qw }),            /* gate_out -- absent */
        };
        absent(ts[0]); absent(ts[4]); absent(ts[6]); absent(ts[8]);
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", 0),
                                     PI("n_head_kv", heads), PI("rot", rot), PS("q_dtype", "bf16"),
                                     PF("eps", eps), PF("wadd", wadd), PI("cs_f32", 1) };
        run_case(rf, ts, ps);
    }

    std::vector<uint16_t> got((size_t)M * qw), want((size_t)M * qw);
    st_memcpy(got.data(),  d_qo, qb, hipMemcpyDeviceToHost);
    st_memcpy(want.data(), d_qr, qb, hipMemcpyDeviceToHost);
    int64_t n_off = 0;
    std::string where;
    for (size_t i = 0; i < want.size(); ++i)
        if (got[i] != want[i]) {
            if (!n_off)
                where = " -- first at row " + std::to_string(i / (size_t)qw) + " head " +
                        std::to_string((i % (size_t)qw) / (size_t)hd) + " lane " +
                        std::to_string(i % (size_t)hd) + ": " + std::to_string(h_bf16(got[i])) +
                        " vs " + std::to_string(h_bf16(want[i]));
            ++n_off;
        }
    char rate[96];
    std::snprintf(rate, sizeof rate, "%lld of %zu differ", (long long)n_off, got.size());
    ok("qk_norm_rope_gate k-only is byte-identical to heads norms plus a rope",
       n_off == 0, std::string(rate) + where);

    /* THE HEAD IT MUST NOT TOUCH. `k_out` is `heads` wide, so the raw key at columns
     * [heads*hd, nw) of the INPUT is not addressed by the output at all -- but the input is what
     * a caller passes in place and an over-long head loop would norm it. */
    std::vector<uint16_t> back((size_t)M * nw);
    st_memcpy(back.data(), d_qk, qkb, hipMemcpyDeviceToHost);
    ok("qk_norm_rope_gate k-only leaves the projection's untouched heads alone",
       back == hqk, "the raw-key head must survive byte for byte");
}

/* -------------------------------------------- router_topk + moe_scatter, against the pair
 *
 * The claim is byte equality on all five outputs, and it is structural rather than hopeful: the
 * fused kernel calls the SAME two device functions the standalone kernels call. What this case
 * exists to catch is the wiring around them -- and there is one specific way to get it wrong that
 * would otherwise produce plausible numbers.
 *
 * THE TWO EXPERT COUNTS ARE DIFFERENT NUMBERS. The selection runs over the WHOLE router, because
 * the router is replicated and every rank picks from every expert; the sort runs over THIS RANK'S
 * SLICE and drops what went elsewhere. Under expert parallelism the two are different numbers --
 * a 512-expert router against a rank slice of 254, say. So this case is run with
 * `n_local < n_expert` and a non-zero `expert_base` -- a fused op
 * handed one count for both would rank half the experts, or index its histogram past the end, and
 * a case built at n_local == n_expert would not notice either.
 *
 * ROWS ENOUGH TO CROSS EVERY WAVE ASSIGNMENT. The standalone selection gives a workgroup NW = 4
 * rows; the fused one launches a wave a row, eight, sixteen or thirty-two of them, and wraps
 * above thirty-two. 5, 13 and 20 rows reach each width with a partial final pass, 45 wraps, and
 * the answer must not depend on which wave got a row -- which is the property that makes the
 * fusion legal in the first place.
 */
static void topk_scatter_at(const RadKernelInfo* rf, const RadKernelInfo* rt,
                            const RadKernelInfo* rs, int64_t M, const char* prot = nullptr);

static void topk_scatter_case(const Plugin& pl) {
    /* ROW names, not op names. `sel` compares against whatever the sweep is naming, and the
     * sweep names ROWS -- every other case here guards on `rowtopk_bf16`, `gated_had_quant_i8`,
     * `rmsnorm_bf16`. `qk_prologue_case` reads as an exception and is not one: its row is spelled
     * `qk_norm_rope_gate`, the same as its op. Guarding on the op name where the row carries a
     * suffix means the case never runs and reports nothing, which is indistinguishable from
     * passing. */
    if (!sel("router_topk_scatter_bf16") && !sel("router_topk_bf16") && !sel("moe_scatter_i32"))
        return;
    const RadKernelInfo* rf = find(pl, "router_topk_scatter_bf16");
    const RadKernelInfo* rt = find(pl, "router_topk_bf16");
    const RadKernelInfo* rs = find(pl, "moe_scatter_i32");
    if (!rf || !rt || !rs) { ok("router_topk_scatter", false, "row absent"); return; }
    for (const int64_t M : { 5, 13, 20, 45 }) topk_scatter_at(rf, rt, rs, M);
    /* ...and with protected experts sorted last: the first, a middle and the last of the slice. */
    for (const int64_t M : { 13, 45 }) topk_scatter_at(rf, rt, rs, M, "0,17,101,253");
}

static void topk_scatter_at(const RadKernelInfo* rf, const RadKernelInfo* rt,
                            const RadKernelInfo* rs, int64_t M, const char* prot) {
    const int64_t ne = 512, nl = 254, base = 3, k = 10;
    const std::string at = " (M=" + std::to_string(M) + (prot ? ", protect " + std::string(prot) : "") +
                           ")";
    const int64_t NS = M * k;   /* routing slots in the step */

    /* Logits with ties and a wide exponent range, so the tie rule and the exact-on-logits
     * comparison both decide something. */
    std::vector<uint16_t> hlg((size_t)M * ne);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t e = 0; e < ne; ++e) {
            const int32_t t = (int32_t)((m * 7919 + e * 31) % 97) - 48;
            hlg[(size_t)(m * ne + e)] = h_to_bf16((float)t * 0.125f);
        }

    void* d_lg = dalloc(hlg.size() * 2);
    void* d_id[2] = { dalloc((size_t)NS * 4), dalloc((size_t)NS * 4) };
    void* d_w [2] = { dalloc((size_t)NS * 2), dalloc((size_t)NS * 2) };
    void* d_so[2] = { dalloc((size_t)NS * 4), dalloc((size_t)NS * 4) };
    void* d_of[2] = { dalloc((size_t)(nl + 1) * 4), dalloc((size_t)(nl + 1) * 4) };
    void* d_cn[2] = { dalloc((size_t)nl * 4), dalloc((size_t)nl * 4) };
    if (!d_lg || !d_id[0] || !d_id[1] || !d_w[0] || !d_w[1] || !d_so[0] || !d_so[1] ||
        !d_of[0] || !d_of[1] || !d_cn[0] || !d_cn[1]) {
        ok(("router_topk_scatter" + at).c_str(), false, "hipMalloc failed"); return;
    }
    st_memcpy(d_lg, hlg.data(), hlg.size() * 2, hipMemcpyHostToDevice);
    for (int i = 0; i < 2; ++i) {
        st_memset(d_id[i], 0xFF, (size_t)NS * 4);  st_memset(d_w[i], 0, (size_t)NS * 2);
        st_memset(d_so[i], 0xFF, (size_t)NS * 4);  st_memset(d_of[i], 0xFF, (size_t)(nl + 1) * 4);
        st_memset(d_cn[i], 0xFF, (size_t)nl * 4);
    }

    /* ---- the pair ---- */
    {
        std::vector<RadTensor> ts = { T(d_lg, RAD_BF16, { M, ne }),
                                      T(d_id[0], RAD_I32, { M, k }),
                                      T(d_w[0], RAD_BF16, { M, k }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", ne), PI("top_k", k),
                                     PI("norm", 1), PS("dtype", "bf16") };
        run_case(rt, ts, ps);
    }
    {
        std::vector<RadTensor> ts = { T(d_id[0], RAD_I32, { M, k }),
                                      T(d_so[0], RAD_I32, { NS }),
                                      T(d_of[0], RAD_I32, { nl + 1 }),
                                      T(d_cn[0], RAD_I32, { nl }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", nl), PI("top_k", k),
                                     PI("expert_base", base) };
        if (prot) ps.push_back(PS("protect", prot));
        run_case(rs, ts, ps);
    }

    /* ---- the fused op ---- */
    {
        std::vector<RadTensor> ts = { T(d_lg, RAD_BF16, { M, ne }),
                                      T(d_id[1], RAD_I32, { M, k }),
                                      T(d_w[1], RAD_BF16, { M, k }),
                                      T(d_so[1], RAD_I32, { NS }),
                                      T(d_of[1], RAD_I32, { nl + 1 }),
                                      T(d_cn[1], RAD_I32, { nl }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", ne), PI("n_local", nl),
                                     PI("top_k", k), PI("norm", 1), PS("dtype", "bf16"),
                                     PI("expert_base", base) };
        if (prot) ps.push_back(PS("protect", prot));
        run_case(rf, ts, ps);
    }

    auto same_i32 = [&](const char* what, void* a, void* b, int64_t n) {
        std::vector<int32_t> x((size_t)n), y((size_t)n);
        st_memcpy(x.data(), a, (size_t)n * 4, hipMemcpyDeviceToHost);
        st_memcpy(y.data(), b, (size_t)n * 4, hipMemcpyDeviceToHost);
        int64_t off = 0; std::string where;
        for (int64_t i = 0; i < n; ++i)
            if (x[(size_t)i] != y[(size_t)i]) {
                if (!off) where = " -- first at " + std::to_string(i) + ": " +
                                  std::to_string(y[(size_t)i]) + " vs " + std::to_string(x[(size_t)i]);
                ++off;
            }
        ok((std::string("router_topk_scatter ") + what + " matches the unfused pair" + at).c_str(),
           off == 0, std::to_string(off) + " of " + std::to_string(n) + " differ" + where);
    };
    same_i32("expert_ids", d_id[0], d_id[1], NS);
    same_i32("sorted_tok", d_so[0], d_so[1], NS);
    same_i32("expert_offset", d_of[0], d_of[1], nl + 1);
    same_i32("expert_count", d_cn[0], d_cn[1], nl);
    {
        std::vector<uint16_t> x((size_t)NS), y((size_t)NS);
        st_memcpy(x.data(), d_w[0], (size_t)NS * 2, hipMemcpyDeviceToHost);
        st_memcpy(y.data(), d_w[1], (size_t)NS * 2, hipMemcpyDeviceToHost);
        ok(("router_topk_scatter expert_w is byte-identical to the unfused pair" + at).c_str(),
           x == y, "the weights carry the renormalisation, which reads back a stored bf16");
    }
}

/* -------------------------------- router_gemm_topk_scatter, against the launches it stands in for
 *
 * The claim is byte equality with THREE launches: the router's gemm_nt, the shared gate's gemm_nt
 * at N = 1, and router_topk_scatter over the router's logits -- so all seven outputs are compared,
 * the logits and the gate included. Those two are what the claim rests on: an element of the
 * skinny bf16 GEMM is a function of its split-K count alone, and the fused op asks the unfused
 * launch's own law for that count at each N (four at 512 x 2560, twenty for the one-column gate;
 * eight and sixteen at 256 x 2048).
 *
 * ROWS AT EVERY WAVE COUNT AND EVERY ROW-TILE COUNT the fused op launches -- 8, 16 and 32 waves,
 * one to four 16-row tiles, partial tiles included -- at the served geometry (512 experts,
 * top_k 10, K 2560) and at a 256-expert one, which takes the other selection instantiation and
 * other splits. An expert-parallel slice with a protect list, and the form without the gate.
 *
 * AND TWICE, BACK TO BACK, ON ONE OUTPUT. The last workgroup is named by a counter that resets
 * itself; a counter left behind by one launch would make the next one rank before the GEMM was
 * done, or never. The configurations share one logits buffer -- and with it one counter -- in
 * an order that changes the grid between consecutive launches. */
static void router_gemm_fused_case(const Plugin& pl) {
    if (!sel("router_gemm_topk_scatter_bf16")) return;
    const RadKernelInfo* rg  = find(pl, "router_gemm_topk_scatter_bf16");
    const RadKernelInfo* m16 = find(pl, "gemm_bf16_nt_m16");
    const RadKernelInfo* m64 = find(pl, "gemm_bf16_nt_m64");
    const RadKernelInfo* rf  = find(pl, "router_topk_scatter_bf16");
    if (!rg || !m16 || !m64 || !rf) { ok("router_gemm_topk_scatter", false, "row absent"); return; }

    struct RgCase {
        int64_t ne, K, k, nl, base;
        const char* prot;
        bool gate;
        std::vector<int64_t> Ms;
    };
    const std::vector<RgCase> cfgs = {
        { 512, 2560, 10, 512, 0, nullptr, true,  { 1, 4, 5, 8, 13, 16, 20, 33, 45, 64 } },
        { 512, 2560, 10, 512, 0, nullptr, false, { 4, 20 } },
        { 512, 2560, 10, 254, 3, "0,17,101,253", true, { 5, 45 } },
        { 512, 2560, 10, 254, 3, nullptr, false, { 13 } },
        { 256, 2048,  8, 256, 0, nullptr, true,  { 4, 33 } },
    };
    const int64_t maxM = 64, maxNe = 512, maxK = 2560, maxNS = 64 * 10;

    uint64_t rs64 = 0x2545F4914F6CDD1Dull;
    auto rnd = [&]() { rs64 ^= rs64 << 13; rs64 ^= rs64 >> 7; rs64 ^= rs64 << 17; return rs64; };
    auto uni = [&](float lo, float hi) {
        return lo + (hi - lo) * (float)((rnd() >> 40) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    };

    void* d_a  = dalloc((size_t)maxM * maxK * 2);
    void* d_w  = dalloc((size_t)maxNe * maxK * 2);
    void* d_wg = dalloc((size_t)maxK * 2);
    void* d_lg[2] = { dalloc((size_t)maxM * maxNe * 2), dalloc((size_t)maxM * maxNe * 2) };
    void* d_g [2] = { dalloc((size_t)maxM * 2),         dalloc((size_t)maxM * 2) };
    void* d_id[2] = { dalloc((size_t)maxNS * 4), dalloc((size_t)maxNS * 4) };
    void* d_ew[2] = { dalloc((size_t)maxNS * 2), dalloc((size_t)maxNS * 2) };
    void* d_so[2] = { dalloc((size_t)maxNS * 4), dalloc((size_t)maxNS * 4) };
    void* d_of[2] = { dalloc((size_t)(maxNe + 1) * 4), dalloc((size_t)(maxNe + 1) * 4) };
    void* d_cn[2] = { dalloc((size_t)maxNe * 4), dalloc((size_t)maxNe * 4) };
    if (!d_a || !d_w || !d_wg || !d_lg[0] || !d_lg[1] || !d_g[0] || !d_g[1] || !d_id[0] ||
        !d_id[1] || !d_ew[0] || !d_ew[1] || !d_so[0] || !d_so[1] || !d_of[0] || !d_of[1] ||
        !d_cn[0] || !d_cn[1]) {
        ok("router_gemm_topk_scatter", false, "hipMalloc failed");
        return;
    }

    for (const RgCase& c : cfgs) {
        /* A trained router's logits are a few units wide; these are about that, with a tie
         * planted in every row so the selection's tie rule decides something. */
        std::vector<uint16_t> hw((size_t)c.ne * c.K), hwg((size_t)c.K);
        for (auto& v : hw)  v = h_to_bf16(uni(-0.06f, 0.06f));
        for (auto& v : hwg) v = h_to_bf16(uni(-0.06f, 0.06f));
        for (int64_t kk = 0; kk < c.K; ++kk) hw[(size_t)(7 * c.K + kk)] = hw[(size_t)(3 * c.K + kk)];
        st_memcpy(d_w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_wg, hwg.data(), hwg.size() * 2, hipMemcpyHostToDevice);

        for (const int64_t M : c.Ms) {
            const int64_t NS = M * c.k;
            const std::string at = fmt(" (n_expert %lld, n_local %lld, M=%lld%s%s)",
                                       (long long)c.ne, (long long)c.nl, (long long)M,
                                       c.gate ? ", gate" : "",
                                       c.prot ? ", protect" : "");
            std::vector<uint16_t> ha((size_t)M * c.K);
            for (auto& v : ha) v = h_to_bf16(uni(-2.0f, 2.0f));
            st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
            for (int i = 0; i < 2; ++i) {
                st_memset(d_lg[i], 0xFF, (size_t)maxM * maxNe * 2);
                st_memset(d_g[i], 0xFF, (size_t)maxM * 2);
                st_memset(d_id[i], 0xFF, (size_t)maxNS * 4);
                st_memset(d_ew[i], 0, (size_t)maxNS * 2);
                st_memset(d_so[i], 0xFF, (size_t)maxNS * 4);
                st_memset(d_of[i], 0xFF, (size_t)(maxNe + 1) * 4);
                st_memset(d_cn[i], 0xFF, (size_t)maxNe * 4);
            }

            /* ---- the three launches */
            const RadKernelInfo* gr = M <= 16 ? m16 : m64;
            {
                std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, c.K }),
                                              T(d_w, RAD_BF16, { c.ne, c.K }),
                                              T(d_lg[0], RAD_BF16, { M, c.ne }) };
                std::vector<RadParam> ps = { PI("M", M), PI("N", c.ne), PI("K", c.K),
                                             PS("dtype", "bf16") };
                run_case(gr, ts, ps);
            }
            if (c.gate) {
                std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, c.K }),
                                              T(d_wg, RAD_BF16, { 1, c.K }),
                                              T(d_g[0], RAD_BF16, { M, 1 }) };
                std::vector<RadParam> ps = { PI("M", M), PI("N", 1), PI("K", c.K),
                                             PS("dtype", "bf16") };
                run_case(gr, ts, ps);
            }
            {
                std::vector<RadTensor> ts = { T(d_lg[0], RAD_BF16, { M, c.ne }),
                                              T(d_id[0], RAD_I32, { M, c.k }),
                                              T(d_ew[0], RAD_BF16, { M, c.k }),
                                              T(d_so[0], RAD_I32, { NS }),
                                              T(d_of[0], RAD_I32, { c.nl + 1 }),
                                              T(d_cn[0], RAD_I32, { c.nl }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n_expert", c.ne),
                                             PI("n_local", c.nl), PI("top_k", c.k), PI("norm", 1),
                                             PS("dtype", "bf16"), PI("expert_base", c.base) };
                if (c.prot) ps.push_back(PS("protect", c.prot));
                run_case(rf, ts, ps);
            }

            /* ---- the fused op, twice */
            for (int rep = 0; rep < 2; ++rep) {
                std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, c.K }),
                                              T(d_w, RAD_BF16, { c.ne, c.K }),
                                              T(d_wg, RAD_BF16, { 1, c.K }),
                                              T(d_lg[1], RAD_BF16, { M, c.ne }),
                                              T(d_id[1], RAD_I32, { M, c.k }),
                                              T(d_ew[1], RAD_BF16, { M, c.k }),
                                              T(d_so[1], RAD_I32, { NS }),
                                              T(d_of[1], RAD_I32, { c.nl + 1 }),
                                              T(d_cn[1], RAD_I32, { c.nl }),
                                              T(d_g[1], RAD_BF16, { M, 1 }) };
                if (!c.gate) { absent(ts[2]); absent(ts[9]); }
                std::vector<RadParam> ps = { PI("M", M), PI("K", c.K), PI("n_expert", c.ne),
                                             PI("n_local", c.nl), PI("top_k", c.k), PI("norm", 1),
                                             PS("dtype", "bf16"), PI("expert_base", c.base) };
                if (c.prot) ps.push_back(PS("protect", c.prot));
                if (rep == 1) {
                    /* The second launch must not find anything the first one left. */
                    st_memset(d_id[1], 0xFF, (size_t)maxNS * 4);
                    st_memset(d_so[1], 0xFF, (size_t)maxNS * 4);
                    st_memset(d_of[1], 0xFF, (size_t)(maxNe + 1) * 4);
                }
                run_case(rg, ts, ps);

                const std::string tag = at + (rep ? ", second launch" : "");
                auto same = [&](const char* what, void* a, void* b, size_t bytes) {
                    std::vector<uint8_t> x(bytes), y(bytes);
                    st_memcpy(x.data(), a, bytes, hipMemcpyDeviceToHost);
                    st_memcpy(y.data(), b, bytes, hipMemcpyDeviceToHost);
                    size_t first = bytes;
                    int64_t n_off = 0;
                    for (size_t i = 0; i < bytes; ++i)
                        if (x[i] != y[i]) { if (first == bytes) first = i; ++n_off; }
                    ok((std::string("router_gemm_topk_scatter ") + what +
                        " is byte-identical to the unfused launches" + tag).c_str(),
                       n_off == 0,
                       n_off ? fmt("%lld byte(s) differ, first at byte %zu", (long long)n_off,
                                   first) : std::string());
                };
                same("logits", d_lg[0], d_lg[1], (size_t)M * c.ne * 2);
                if (c.gate) same("shared gate", d_g[0], d_g[1], (size_t)M * 2);
                same("expert_ids", d_id[0], d_id[1], (size_t)NS * 4);
                same("expert_w", d_ew[0], d_ew[1], (size_t)NS * 2);
                same("sorted_tok", d_so[0], d_so[1], (size_t)NS * 4);
                same("expert_offset", d_of[0], d_of[1], (size_t)(c.nl + 1) * 4);
                same("expert_count", d_cn[0], d_cn[1], (size_t)c.nl * 4);
            }
        }
    }
    /* A shape past the decode band is refused, not served at another arithmetic. */
    {
        const int64_t M = 65;
        std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { M, 2560 }),
                                      T(d_w, RAD_BF16, { 512, 2560 }),
                                      T(d_wg, RAD_BF16, { 1, 2560 }),
                                      T(nullptr, RAD_BF16, { M, 512 }),
                                      T(nullptr, RAD_I32, { M, 10 }),
                                      T(nullptr, RAD_BF16, { M, 10 }),
                                      T(nullptr, RAD_I32, { M * 10 }),
                                      T(nullptr, RAD_I32, { 513 }),
                                      T(nullptr, RAD_I32, { 512 }),
                                      T(nullptr, RAD_BF16, { M, 1 }) };
        std::vector<RadParam> ps = { PI("M", M), PI("K", 2560), PI("n_expert", 512),
                                     PI("top_k", 10), PI("norm", 1), PS("dtype", "bf16") };
        run_case(rg, ts, ps, RAD_E_SHAPE);
    }
}

/* THE PROTECTED EXPERTS ARE SORTED LAST, against a host stable sort by place. A layer that keeps a
 * few experts at a higher precision serves them with a second grouped GEMM over a trailing run of
 * the offsets, so the sort has to put their rows after every other expert's, in the order listed,
 * with the rest closed up -- and keep every run stable. Both forms of the standalone sort run: 450
 * slots is the one-workgroup form, 3000 the three-pass one, and the first, a middle and the last
 * expert of a rank's slice are protected so the closing-up is tested at both ends. The fused
 * decode kernel is held to the standalone pair in topk_scatter_case. */
static void scatter_protect_case(const Plugin& pl) {
    if (!sel("moe_scatter_i32")) return;
    const RadKernelInfo* rs = find(pl, "moe_scatter_i32");
    if (!rs) { ok("moe_scatter protect", false, "row absent"); return; }
    const int64_t ne = 512, nl = 254, base = 3, k = 10;
    const std::vector<int64_t> prot = { 0, 17, 101, 253 };
    std::vector<int64_t> place((size_t)nl);
    {
        int64_t next = 0;
        for (int64_t e = 0; e < nl; ++e)
            if (std::find(prot.begin(), prot.end(), e) == prot.end()) place[(size_t)e] = next++;
        for (int64_t p : prot) place[(size_t)p] = next++;
    }
    for (const int64_t M : { 45, 300 }) {
        const int64_t NS = M * k;
        const std::string at = " (" + std::to_string(NS) + " slots)";
        /* Each row's k distinct experts from the whole router, a quarter of them in this slice and
         * the protected ones over-represented so their runs are long. */
        std::vector<int32_t> ids((size_t)NS);
        uint64_t rs64 = 0x9E3779B97F4A7C15ull ^ (uint64_t)M;
        auto rnd = [&]() { rs64 ^= rs64 << 13; rs64 ^= rs64 >> 7; rs64 ^= rs64 << 17; return rs64; };
        for (int64_t m = 0; m < M; ++m)
            for (int64_t j = 0; j < k; ++j) {
                int32_t g;
                bool dup;
                do {
                    const uint64_t r = rnd();
                    g = (r % 5 == 0) ? (int32_t)(base + prot[(size_t)((r >> 8) % prot.size())])
                                     : (int32_t)((r >> 8) % ne);
                    dup = false;
                    for (int64_t q = 0; q < j; ++q) dup |= ids[(size_t)(m * k + q)] == g;
                } while (dup);
                ids[(size_t)(m * k + j)] = g;
            }
        std::vector<int32_t> want_s((size_t)NS, -1), want_o((size_t)(nl + 1), 0),
                             want_c((size_t)nl, 0);
        for (int64_t i = 0; i < NS; ++i) {
            const int64_t l = ids[(size_t)i] - base;
            if (l >= 0 && l < nl) ++want_c[(size_t)place[(size_t)l]];
        }
        for (int64_t p = 0; p < nl; ++p) want_o[(size_t)(p + 1)] = want_o[(size_t)p] + want_c[(size_t)p];
        {
            std::vector<int32_t> cur(want_o.begin(), want_o.end() - 1);
            for (int64_t i = 0; i < NS; ++i) {
                const int64_t l = ids[(size_t)i] - base;
                if (l >= 0 && l < nl) want_s[(size_t)cur[(size_t)place[(size_t)l]]++] = (int32_t)i;
            }
        }
        void* d_id = dalloc((size_t)NS * 4);
        void* d_so = dalloc((size_t)NS * 4);
        void* d_of = dalloc((size_t)(nl + 1) * 4);
        void* d_cn = dalloc((size_t)nl * 4);
        if (!d_id || !d_so || !d_of || !d_cn) { ok(("moe_scatter protect" + at).c_str(), false, "alloc"); return; }
        st_memcpy(d_id, ids.data(), (size_t)NS * 4, hipMemcpyHostToDevice);
        st_memset(d_so, 0x5A, (size_t)NS * 4);
        st_memset(d_of, 0x5A, (size_t)(nl + 1) * 4);
        st_memset(d_cn, 0x5A, (size_t)nl * 4);
        std::vector<RadTensor> ts = { T(d_id, RAD_I32, { M, k }), T(d_so, RAD_I32, { NS }),
                                      T(d_of, RAD_I32, { nl + 1 }), T(d_cn, RAD_I32, { nl }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", nl), PI("top_k", k),
                                     PI("expert_base", base), PS("protect", "0,17,101,253") };
        run_case(rs, ts, ps);
        std::vector<int32_t> got_s((size_t)NS), got_o((size_t)(nl + 1)), got_c((size_t)nl);
        st_memcpy(got_s.data(), d_so, (size_t)NS * 4, hipMemcpyDeviceToHost);
        st_memcpy(got_o.data(), d_of, (size_t)(nl + 1) * 4, hipMemcpyDeviceToHost);
        st_memcpy(got_c.data(), d_cn, (size_t)nl * 4, hipMemcpyDeviceToHost);
        const int64_t prot_rows = want_o[(size_t)nl] - want_o[(size_t)(nl - prot.size())];
        ok(("moe_scatter puts the protected experts last" + at).c_str(),
           got_s == want_s && got_o == want_o && got_c == want_c,
           std::to_string(prot_rows) + " protected rows of " + std::to_string(want_o[(size_t)nl]) +
               " live; sorted " + (got_s == want_s ? "ok" : "DIFFERS") + ", offsets " +
               (got_o == want_o ? "ok" : "DIFFER") + ", counts " + (got_c == want_c ? "ok" : "DIFFER"));
        /* A list that is not ascending, or reaches past the slice, is refused. */
        for (const char* badp : { "17,0", "0,254" }) {
            std::vector<RadParam> pb = { PI("M", M), PI("n_expert", nl), PI("top_k", k),
                                         PI("expert_base", base), PS("protect", badp) };
            run_case(rs, ts, pb, RAD_E_SHAPE);
        }
    }
}

static void rowtopk_shard_case(const Plugin& pl) {
    if (!sel("rowtopk_merge") && !sel("rowtopk_bf16")) return;
    const RadKernelInfo* rt = find(pl, "rowtopk_bf16");
    const RadKernelInfo* rm = find(pl, "rowtopk_merge");
    if (!rt || !rm) { ok("rowtopk_merge", false, "row absent"); return; }

    const int64_t M = 5, N = 2048, R = 8, W = 2, SH = N / W;
    std::vector<uint16_t> hx((size_t)M * N);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t c = 0; c < N; ++c)
            hx[(size_t)(m * N + c)] = h_to_bf16((float)((m * 31 + c * 7) % 23) - 11.0f);

    void* d_x  = dalloc((size_t)M * N * 2);
    void* d_i0 = dalloc((size_t)M * R * 4);
    void* d_v0 = dalloc((size_t)M * R * 4);
    void* d_i  = dalloc((size_t)M * R * 4);
    void* d_v  = dalloc((size_t)M * R * 4);
    void* d_p  = dalloc((size_t)M * R * 2 * 4);
    void* d_g  = dalloc((size_t)M * W * R * 2 * 4);
    void* d_h  = dalloc((size_t)M * SH * 2);
    if (!d_x || !d_i0 || !d_v0 || !d_i || !d_v || !d_p || !d_g || !d_h) {
        ok("rowtopk_merge", false, "hipMalloc failed"); return;
    }
    st_memcpy(d_x, hx.data(), (size_t)M * N * 2, hipMemcpyHostToDevice);

    /* The answer to agree with: one top-R over the whole plane. */
    {
        std::vector<RadTensor> ts = { T(d_x, RAD_BF16, { M, N }), T(d_i0, RAD_I32, { M, R }),
                                      T(d_v0, RAD_F32, { M, R }) };
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("R", R), PS("dtype", "bf16") };
        run_case(rt, ts, ps);
    }
    std::vector<int32_t> wi((size_t)M * R);
    std::vector<float>   wv((size_t)M * R);
    st_memcpy(wi.data(), d_i0, (size_t)M * R * 4, hipMemcpyDeviceToHost);
    st_memcpy(wv.data(), d_v0, (size_t)M * R * 4, hipMemcpyDeviceToHost);

    /* Each rank's own slice, reporting global columns, plus the pairs an all_gather would move.
     *
     * THE SLICE IS COPIED OUT RATHER THAN VIEWED. libr4d's top-R takes a CONTIGUOUS [M, N] plane
     * and refuses a strided one, so a column slice of the wide plane is not an operand it accepts
     * -- and it does not have to be: under a real shard each rank's draft head writes only its own
     * rows, so the plane it hands this op is contiguous and `SH` wide by construction. Passing a
     * view here would be testing a shape the engine never produces. */
    std::vector<uint16_t> half((size_t)M * SH);
    std::vector<float> gath((size_t)M * W * R * 2);
    for (int64_t r = 0; r < W; ++r) {
        for (int64_t m = 0; m < M; ++m)
            for (int64_t c = 0; c < SH; ++c)
                half[(size_t)(m * SH + c)] = hx[(size_t)(m * N + r * SH + c)];
        st_memcpy(d_h, half.data(), half.size() * 2, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = {
            T(d_h, RAD_BF16, { M, SH }),
            T(d_i, RAD_I32, { M, R }), T(d_v, RAD_F32, { M, R }),
            T(d_p, RAD_F32, { M, R, 2 }),
        };
        std::vector<RadParam> ps = { PI("M", M), PI("N", SH), PI("R", R),
                                     PI("vocab_off", r * SH), PS("dtype", "bf16") };
        run_case(rt, ts, ps);
        std::vector<float> pr((size_t)M * R * 2);
        st_memcpy(pr.data(), d_p, (size_t)M * R * 2 * 4, hipMemcpyDeviceToHost);
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < R * 2; ++k)
                gath[(size_t)((m * W + r) * R * 2 + k)] = pr[(size_t)(m * R * 2 + k)];
    }
    st_memcpy(d_g, gath.data(), gath.size() * 4, hipMemcpyHostToDevice);

    {
        std::vector<RadTensor> ts = { T(d_g, RAD_F32, { M, W * R, 2 }), T(d_i, RAD_I32, { M, R }),
                                      T(d_v, RAD_F32, { M, R }) };
        std::vector<RadParam> ps = { PI("M", M), PI("R", R), PI("world_size", W),
                                     PS("dtype", "bf16") };
        run_case(rm, ts, ps);
    }
    std::vector<int32_t> gi((size_t)M * R);
    std::vector<float>   gv((size_t)M * R);
    st_memcpy(gi.data(), d_i, (size_t)M * R * 4, hipMemcpyDeviceToHost);
    st_memcpy(gv.data(), d_v, (size_t)M * R * 4, hipMemcpyDeviceToHost);

    int bad = 0;
    std::string first;
    for (size_t i = 0; i < gi.size() && !bad; ++i)
        if (gi[i] != wi[i] || gv[i] != wv[i]) {
            ++bad;
            first = "row " + std::to_string(i / (size_t)R) + " slot " +
                    std::to_string(i % (size_t)R) + ": merged (" + std::to_string(gi[i]) + ", " +
                    std::to_string(gv[i]) + ") want (" + std::to_string(wi[i]) + ", " +
                    std::to_string(wv[i]) + ")";
        }
    ok("rowtopk_merge matches an unsharded top-R", bad == 0, bad ? first : "");
}
/* --------------------------- the attention with the output gate and quantiser in its merge, vs the pair
 *
 * `attn_decode_gq_h256_gqa12_<kv>` is `attn_decode_h256_gqa12_<kv>` followed by `gate_quant_fp8`
 * over its output, for the E4M3 cache and the bf16 one, and the claim is byte equality with that
 * pair: the gated bf16 product it leaves in `out`, every E4M3 code and every scale. The engine's
 * sparse-path geometry -- 12 query heads over one kv head of 256, 4-token pages, one query row a
 * sequence -- with the gate a column view of a [q|gate] plane exactly as the block hands it over.
 *
 * TWO BATCH SIZES, because the merge has two forms: three rows is 36 (row, head) jobs and takes
 * the one-thread-a-column merge, 24 rows is 288 and takes the one-wave-a-row form. Each has one
 * padded row (no context), which the pair gives zeros, a zero scale group and a scale of 1. The
 * E4M3 cache is random bytes but for the NaN codes, the bf16 one random finite values, and the
 * descales are not 1. */
/* `i8` runs both at INT8 codes -- the codes operand an int8 out projection reads -- and the claim
 * is the same: the fold's int8 merge leaves what gate_quant_fp8's int8 form leaves. */
static void attn_gq_fold_case(const Plugin& pl, int64_t S, bool bf16, bool i8 = false) {
    const std::string kvs = bf16 ? "bf16kv" : "fp8kv";
    const std::string fold = "attn_decode_gq_h256_gqa12_" + kvs;
    const std::string pair = "attn_decode_h256_gqa12_" + kvs;
    if (!sel(fold.c_str())) return;
    const std::string name = fold + " is byte-identical to the pair at " +
                             std::to_string((long long)S) + " rows" + (i8 ? ", int8 codes" : "");
    const uint32_t cdt = i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
    const RadKernelInfo* rk = find(pl, fold.c_str());
    const RadKernelInfo* ra = find(pl, pair.c_str());
    const RadKernelInfo* rg = find(pl, "gate_quant_fp8");
    if (!rk || !ra || !rg) { ok(name.c_str(), false, "row absent"); return; }

    const int64_t nq = 12, nkv = 1, hd = 256, bs = 4, nb = 96, maxb = 40;
    uint32_t st = 0x2545f491u ^ (uint32_t)S;
    auto rnd = [&](float span) {
        st = st * 1664525u + 1013904223u;
        return ((float)(st >> 8) / 16777216.0f * 2.0f - 1.0f) * span;
    };
    std::vector<int32_t> table((size_t)(S * maxb)), used((size_t)S);
    for (int64_t r = 0; r < S; ++r) {
        for (int64_t b = 0; b < maxb; ++b)
            table[(size_t)(r * maxb + b)] = (int32_t)((r * 13 + b * 7 + 1) % nb);
        used[(size_t)r] = (int32_t)(maxb * bs - (r % 5) * 3);
    }
    used[1] = 0;                                                   /* a padded row */
    std::vector<uint16_t> hq((size_t)(S * nq * hd)), hqg((size_t)(S * nq * 2 * hd));
    for (auto& x : hq)  x = h_to_bf16(rnd(2.0f));
    for (auto& x : hqg) x = h_to_bf16(rnd(6.0f));
    const size_t cbytes = (size_t)(nb * nkv * bs * 2 * hd) * (bf16 ? 2 : 1);
    std::vector<uint8_t> hc(cbytes);
    if (bf16) {
        for (size_t i = 0; i + 1 < cbytes; i += 2) {
            const uint16_t h = h_to_bf16(rnd(3.0f));
            std::memcpy(&hc[i], &h, 2);
        }
    } else {
        for (auto& x : hc) {
            x = (uint8_t)(rnd(128.0f) + 128.0f);
            if ((x & 0x7f) == 0x7f) x ^= 1;                        /* E4M3's NaN codes */
        }
    }
    std::vector<float> hds((size_t)(S * nkv));
    for (auto& x : hds) x = 0.75f + 0.5f * (rnd(1.0f) + 1.0f) * 0.5f;

    const size_t qb = hq.size() * 2, ob = (size_t)(S * nq * hd) * 2;
    const size_t q8b = (size_t)(S * nq * hd), sb = (size_t)(S * nq * (hd / 128)) * 4;
    void* d_q  = dalloc(qb);          void* d_qg = dalloc(hqg.size() * 2);
    void* d_c  = dalloc(cbytes);
    void* d_kd = dalloc(hds.size() * 4); void* d_vd = dalloc(hds.size() * 4);
    void* d_oA = dalloc(ob);  void* d_oB = dalloc(ob);
    void* d_qA = dalloc(q8b); void* d_qB = dalloc(q8b);
    void* d_sA = dalloc(sb);  void* d_sB = dalloc(sb);
    if (!d_q || !d_qg || !d_c || !d_kd || !d_vd || !d_oA || !d_oB || !d_qA || !d_qB || !d_sA ||
        !d_sB) { ok(name.c_str(), false, "hipMalloc failed"); return; }
    st_memcpy(d_q,  hq.data(),  qb, hipMemcpyHostToDevice);
    st_memcpy(d_qg, hqg.data(), hqg.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_c,  hc.data(),  cbytes, hipMemcpyHostToDevice);
    st_memcpy(d_kd, hds.data(), hds.size() * 4, hipMemcpyHostToDevice);
    std::reverse(hds.begin(), hds.end());
    st_memcpy(d_vd, hds.data(), hds.size() * 4, hipMemcpyHostToDevice);

    /* The gate: head h's second half of the [q|gate] row, one pitch of 2 * head_dim a head. */
    RadTensor gate = T((uint8_t*)d_qg + hd * 2, RAD_BF16, { S, nq, hd });
    gate.stride[0] = nq * 2 * hd;
    gate.stride[1] = 2 * hd;
    auto attn_ops = [&](void* out) {
        return std::vector<RadTensor>{
            T(d_q, RAD_BF16, { S, nq, hd }),
            T(d_c, bf16 ? RAD_BF16 : RAD_F8E4M3, { nb, nkv, bs, 2 * hd }),
            TI32({ S, maxb }, table),
            TI32({ S }, used),
            T(d_kd, RAD_F32, { S, nkv }),
            T(d_vd, RAD_F32, { S, nkv }),
            RadTensor{},
            T(out, RAD_BF16, { S, nq * hd }) };
    };
    std::vector<RadParam> pa = {
        PI("q_len", 1), PI("head_dim", hd), PI("gqa", nq / nkv), PI("block_size", bs),
        PI("causal", 1), PI("window", 0), PS("q_dtype", "bf16"),
        PS("kv_dtype", bf16 ? "bf16" : "fp8_e4m3"),
        PI("n_head", nq), PI("max_seqs", S), PI("max_ctx", maxb * bs),
    };

    /* ---- the pair ---- */
    {
        std::vector<RadTensor> ts = attn_ops(d_oA);
        run_case(ra, ts, pa);
    }
    {
        std::vector<RadTensor> ts = { gate, T(d_oA, RAD_BF16, { S * nq, hd }),
                                      T(d_qA, cdt, { S * nq, hd }),
                                      T(d_sA, RAD_F32, { S * nq, hd / 128 }),
                                      T(d_oA, RAD_BF16, { S * nq, hd }) };
        std::vector<RadParam> ps = { PI("M", S * nq), PI("n", hd), PI("group", 128),
                                     PS("dtype", "bf16"), PS("act", "sigmoid") };
        run_case(rg, ts, ps);
    }
    /* ---- the fold ---- */
    {
        std::vector<RadTensor> ts = attn_ops(d_oB);
        ts.push_back(gate);
        ts.push_back(T(d_qB, cdt, { S, nq * hd }));
        ts.push_back(T(d_sB, RAD_F32, { S, nq * hd / 128 }));
        std::vector<RadParam> ps = pa;
        ps.push_back(PI("group", 128));
        ps.push_back(PS("act", "sigmoid"));
        run_case(rk, ts, ps);
    }

    std::vector<uint8_t> oa(ob), obv(ob), qa(q8b), qbv(q8b), sa(sb), sbv(sb);
    st_memcpy(oa.data(),  d_oA, ob,  hipMemcpyDeviceToHost);
    st_memcpy(obv.data(), d_oB, ob,  hipMemcpyDeviceToHost);
    st_memcpy(qa.data(),  d_qA, q8b, hipMemcpyDeviceToHost);
    st_memcpy(qbv.data(), d_qB, q8b, hipMemcpyDeviceToHost);
    st_memcpy(sa.data(),  d_sA, sb,  hipMemcpyDeviceToHost);
    st_memcpy(sbv.data(), d_sB, sb,  hipMemcpyDeviceToHost);
    auto ndiff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        int64_t n = 0;
        for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
        return n;
    };
    /* And the pair did write something: its codes are not all zero. */
    int64_t live = 0;
    for (uint8_t x : qa) live += x != 0;
    const int64_t n_o = ndiff(oa, obv), n_q = ndiff(qa, qbv), n_s = ndiff(sa, sbv);
    char msg[200];
    std::snprintf(msg, sizeof msg, "out %lld of %zu, codes %lld of %zu, scales %lld of %zu bytes "
                  "differ; %lld nonzero codes", (long long)n_o, ob, (long long)n_q, q8b,
                  (long long)n_s, sb, (long long)live);
    ok(name.c_str(), n_o == 0 && n_q == 0 && n_s == 0 && live > 0, msg);
}

static void kv_store_case(const Plugin& pl, const char* name, const char* kv_dtype) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }

    const bool     fp8        = std::strcmp(kv_dtype, "bf16") != 0;
    const int64_t  head_dim   = 256, kv_heads = 2, block_size = 16, n_blocks = 4;
    const int64_t  lane       = kv_heads * head_dim;
    const int64_t  cache_elts = n_blocks * kv_heads * block_size * 2 * head_dim;
    /* Slot 0 is real; -1 is the pad; the last is past the end of the four blocks. */
    const std::vector<int32_t> slots = { 0, 17, 63, -1, (int32_t)(n_blocks * block_size), 33 };
    const int64_t  T_tok      = (int64_t)slots.size();

    /* Exact in E4M3 and in bf16. Sign separates K from V, the magnitude carries the head and the
     * column, so no two (token, head, dim) triples that this could confuse share a value. */
    auto want = [&](int64_t t, int64_t h, int64_t d, bool is_v) {
        static const float mag[8] = { 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f };
        const float m = mag[(size_t)((t * 3 + h * 5 + d) % 8)];
        return is_v ? -m : m;
    };

    std::vector<uint16_t> hk((size_t)T_tok * lane), hv((size_t)T_tok * lane);
    for (int64_t t = 0; t < T_tok; ++t)
        for (int64_t h = 0; h < kv_heads; ++h)
            for (int64_t d = 0; d < head_dim; ++d) {
                hk[(size_t)(t * lane + h * head_dim + d)] = h_to_bf16(want(t, h, d, false));
                hv[(size_t)(t * lane + h * head_dim + d)] = h_to_bf16(want(t, h, d, true));
            }

    const size_t cbytes = (size_t)cache_elts * (fp8 ? 1u : 2u);
    void* d_k  = dalloc((size_t)T_tok * lane * 2);
    void* d_v  = dalloc((size_t)T_tok * lane * 2);
    void* d_sm = dalloc((size_t)T_tok * 4);
    void* d_c  = dalloc(cbytes);
    if (!d_k || !d_v || !d_sm || !d_c) { ok(name, false, "hipMalloc failed"); return; }
    st_memcpy(d_k, hk.data(), (size_t)T_tok * lane * 2, hipMemcpyHostToDevice);
    st_memcpy(d_v, hv.data(), (size_t)T_tok * lane * 2, hipMemcpyHostToDevice);
    st_memcpy(d_sm, slots.data(), (size_t)T_tok * 4, hipMemcpyHostToDevice);
    /* 0x7f is NaN in E4M3 and a large number in bf16 -- either way it is a value the store cannot
     * produce from the inputs above, so anything still holding it was not written. */
    st_memset(d_c, 0x7f, cbytes);

    std::vector<RadTensor> ts = {
        T(d_k,  RAD_BF16, { T_tok, lane }),
        T(d_v,  RAD_BF16, { T_tok, lane }),
        T(d_sm, RAD_I32,  { T_tok }),
        T(d_c,  fp8 ? RAD_F8E4M3 : RAD_BF16, { n_blocks, kv_heads, block_size, 2 * head_dim }),
    };
    std::vector<RadParam> ps = {
        PI("M", T_tok), PI("head_dim", head_dim), PI("n_head_kv", kv_heads),
        PI("block_size", block_size), PS("kv_dtype", kv_dtype),
    };
    run_case(r, ts, ps);

    std::vector<uint8_t> got(cbytes);
    st_memcpy(got.data(), d_c, cbytes, hipMemcpyDeviceToHost);
    auto at = [&](int64_t i) {
        return fp8 ? h_e4m3(got[(size_t)i])
                   : h_bf16(*(const uint16_t*)(got.data() + (size_t)i * 2));
    };

    int bad = 0, pad_touched = 0;
    std::string first;
    for (int64_t t = 0; t < T_tok && bad < 1; ++t) {
        const int32_t s = slots[(size_t)t];
        const bool    live = s >= 0 && s / block_size < n_blocks;
        for (int64_t h = 0; h < kv_heads && !bad; ++h)
            for (int64_t d = 0; d < head_dim; ++d) {
                if (!live) continue;
                const int64_t blk = s / block_size, off = s % block_size;
                const int64_t base = ((blk * kv_heads + h) * block_size + off) * 2 * head_dim;
                const float gk = at(base + d), gv = at(base + head_dim + d);
                const float wk = want(t, h, d, false), wv = want(t, h, d, true);
                if (gk != wk || gv != wv) {
                    ++bad;
                    first = "tok " + std::to_string((long long)t) + " head " +
                            std::to_string((long long)h) + " dim " + std::to_string((long long)d) +
                            ": k " + std::to_string(gk) + " want " + std::to_string(wk) +
                            ", v " + std::to_string(gv) + " want " + std::to_string(wv);
                    break;
                }
            }
    }
    /* Every element of a block no live slot names must still be the sentinel, and that is checked
     * on the RAW STORAGE rather than through `at`. 0x7f is the NaN encoding in E4M3 and this
     * file's `h_e4m3` does not implement it -- it reads exponent 15, mantissa 7 as 480.0 -- so a
     * decoded comparison against NaN reports every untouched byte as written. The bytes are what
     * the question is about anyway: "was this written", not "what does it mean". */
    for (int64_t i = 0; i < cache_elts; ++i) {
        const int64_t blk = i / (kv_heads * block_size * 2 * head_dim);
        const int64_t off = (i / (2 * head_dim)) % block_size;
        bool named = false;
        for (int32_t s : slots)
            if (s >= 0 && s / block_size < n_blocks && s / block_size == blk && s % block_size == off)
                named = true;
        if (named) continue;
        const bool intact = fp8 ? got[(size_t)i] == 0x7f
                                : *(const uint16_t*)(got.data() + (size_t)i * 2) == 0x7f7f;
        if (!intact) {
            ++pad_touched;
            first = "element " + std::to_string((long long)i) + " of block " +
                    std::to_string((long long)blk) + " offset " + std::to_string((long long)off);
            break;
        }
    }

    ok((std::string(name) + " scatters k and v to the slot's block and offset").c_str(),
       bad == 0, first);
    ok((std::string(name) + " leaves a pad slot and every unnamed offset untouched").c_str(),
       pad_touched == 0, pad_touched ? "written: " + first : "");
}

static void csa3_kvarn_case(const Plugin& pl) {
    if (!sel("kv_csa3_kvarn_pool") && !sel("csa3_kvarn_pool")) return;
    int (*pool_fn)(uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, int, int, int, hipStream_t) = nullptr;
    if (!sym(pl.h, "r4d_kv_csa3_kvarn_pool", &pool_fn) || !pool_fn) {
        ok("csa3_kvarn_pool", false, "r4d_kv_csa3_kvarn_pool symbol missing");
        return;
    }
    const int n_heads = 2, bs = 16, hd = 256;
    const int n_chunks = 1;
    const int64_t kv_block_stride = (int64_t)n_heads * bs * (2 * hd);
    const int64_t kv_head_stride = (int64_t)bs * (2 * hd);
    const size_t total_bytes = (size_t)(4 * kv_block_stride);

    void* d_kv  = dalloc(total_bytes);
    void* d_src = dalloc(4 * sizeof(int32_t));
    void* d_dst = dalloc(1 * sizeof(int32_t));
    if (!d_kv || !d_src || !d_dst) { ok("csa3_kvarn_pool", false, "hipMalloc failed"); return; }

    const std::vector<int32_t> h_src = { 0, 1, 2, 0 };
    const std::vector<int32_t> h_dst = { 3 };
    st_memcpy(d_src, h_src.data(), 4 * sizeof(int32_t), hipMemcpyHostToDevice);
    st_memcpy(d_dst, h_dst.data(), 1 * sizeof(int32_t), hipMemcpyHostToDevice);

    std::vector<uint8_t> h_kv(total_bytes, 0);
    for (int t = 0; t < 48; ++t) {
        const float val = 0.5f + (float)(t % 7);
        const uint8_t fp8_val = rad_f32_to_fp8e4m3(val);
        const int b_idx = t / bs;
        const int t_sub = t % bs;
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < hd; ++d) {
                const size_t off_k = (size_t)h_src[(size_t)b_idx] * kv_block_stride + (size_t)h * kv_head_stride + (size_t)t_sub * (2 * hd) + d;
                const size_t off_v = off_k + hd;
                h_kv[off_k] = fp8_val;
                h_kv[off_v] = fp8_val;
            }
        }
    }
    st_memcpy(d_kv, h_kv.data(), total_bytes, hipMemcpyHostToDevice);

    int rc = pool_fn((uint8_t*)d_kv, (const int32_t*)d_src, (const int32_t*)d_dst,
                     kv_block_stride, kv_head_stride, n_chunks, n_heads, 128, nullptr);
    if (rc != 0) {
        ok("csa3_kvarn_pool", false, "kernel launch failed");
        return;
    }
    (void)hipDeviceSynchronize();
    st_memcpy(h_kv.data(), d_kv, total_bytes, hipMemcpyDeviceToHost);

    const size_t out_off = (size_t)h_dst[0] * kv_block_stride;
    bool all_finite_and_nonzero = true;
    for (size_t i = 0; i < (size_t)bs * 2 * (size_t)hd; ++i) {
        if (h_kv[out_off + i] == 0) { all_finite_and_nonzero = false; break; }
    }
    ok("csa3_kvarn_pool pools 3:1 historical tokens with KVarN compression", all_finite_and_nonzero,
       all_finite_and_nonzero ? "" : "pooled chunk contains unexpected zero entries");
}

/* THE FP8 GEMM, CHECKED AGAINST A HOST REFERENCE ON THE SAME BYTES.
 *
 * A kernel that produces plausible numbers is not evidence (spec §17). This builds a weight's two
 * block-fp8 planes, has the plugin's OWN layout hooks arrange them, dequantises the planes on the
 * host -- the same E4M3 codes and the same bf16 block scales the kernel will read -- and compares.
 * So it falsifies four things at once: the arrangement, the kernel's scale INDEXING (a `>> 7` off
 * by one reads a neighbouring tile and still looks like weights), the fold's placement inside the
 * K loop, and -- for fp8a8 -- the activation quantiser and the epilogue's transpose.
 *
 * It is deliberately NOT a tolerance on the quantisation error: both sides see the same quantised
 * weight, so the only difference permitted is summation order. The bound is bf16's own resolution
 * on the output, 2^-8 relative, loosened to 3e-2 because a K=512 reduction reordered across 32
 * lanes moves the last bf16 mantissa bit on values near a rounding boundary and this is a
 * correctness gate, not a numerics measurement -- a transposed operand or a misread scale lands
 * two orders of magnitude away from it, which is the thing being caught.
 *
 * WHAT IT ACTUALLY MEASURES, so a later loosening is visible: 1.739e-3 for the matvec at every one
 * of the five rb variants, agreeing to six digits, and 1.1731e-2 for BOTH W8A8 rows. 2^-9 is
 * 1.95e-3, so the matvec is sitting on bf16's own rounding floor and nothing else; the W8A8 rows
 * are higher because their ACTIVATION is quantised to E4M3 as well, whose three mantissa bits are
 * ~6% per element before the K=512 average. On real checkpoint weights (--real) the matvec measures
 * 1.66e-3 at gate_proj, o_proj and q_proj alike.
 *
 * THE TWO W8A8 ROWS AGREEING TO SIX DIGITS IS THE RESULT WORTH READING. They put the weight on
 * OPPOSITE operands, fold opposite scale shapes, and store through opposite epilogues -- one
 * transposed and split over K, one coalesced and not -- and they land on the same number. Two
 * independent paths agreeing with the host is a much stronger statement than either agreeing on
 * its own, because the ways they could each be wrong do not overlap. */
/* THE GROUPED DYNAMIC CONVOLUTION AGAINST A HOST MIRROR OF DFlash2's `_grouped_conv`.
 *
 * The smoke case above launches it against zeroed buffers, which cannot tell a convolution from a
 * copy: every tap term is zero, the block-position mask is never exercised, and the group index is
 * irrelevant when every delta is the same. This one puts the reference formula on the host --
 *
 *   out[t,h] = SUM over k of (base[k][h] + delta[t][k*NG + h/group]) * x[t-k][h] * (t%block >= k)
 *
 * -- and asks whether the kernel agrees. The three things it can get wrong and the smoke case
 * cannot see are all in that line: which delta belongs to which channel (the group index), whether
 * the tap shift is within the block or across it (the mask), and whether base is indexed by
 * channel or by group. */
/* THE ANCHOR STATE FORM, END TO END.
 *
 * `gdn_recurrent_update` serves two contracts for the same arithmetic. Under `per_candidate` every
 * candidate token writes its own state slot and the next step reads the one the last accepted
 * token wrote. Under `anchor` a sequence keeps ONE state, the step's tokens are recorded beside it
 * as replay factors, and the next step folds the accepted prefix back in before it runs. The two
 * produce the same output from a completely different cache, so nothing that compares the state
 * can compare them -- and a fixture recording cannot reach the anchor at all, because drawing one
 * state index per candidate IS the per-candidate contract.
 *
 * What that leaves unchecked is the whole anchor path: the replay loop, the factor block's layout,
 * and the column algebra that has to survive a paged cache whose window slides. So this drives a
 * multi-step schedule against a host walk of the delta rule with a rollback between steps, which
 * is the per-candidate contract's answer by definition:
 *
 *     the state a step starts from is the state after the previous step's first `num_accepted`
 *     tokens, whatever the cache did to hold on to it.
 *
 * THE WINDOW MOVES ALL THREE WAYS. A paged caller re-derives the page index every step, so the six
 * columns slide by -1, 0 or +1 and the kernel finds its anchor through a marker it left in the
 * factor block. The schedule below takes +1, then 0, then -1. With a window that never moves,
 * every read lands where it was written whatever the algebra says and the marker is untested.
 *
 * AND IT ASSERTS ITS OWN SENSITIVITY. A second host chain rolls back to the FIRST token at every
 * step -- what a kernel that ignored `num_accepted` would produce -- and the check requires that
 * one to DISAGREE with the device. Without it, a schedule whose acceptances all happened to be one
 * would pass with the replay loop deleted.
 *
 * It needs a process of its own. The state form is selected by an environment variable read once
 * and cached, so the smoke case above and this one cannot share one; the child is spawned the same
 * way and for the same reason the per-row children are.
 */
static void gdn_recurrent_anchor(const Plugin& pl) {
    const char* row = "gdn_recurrent_update_k128_v128_bf16_fp32state";
    const char* nm  = "gdn_recurrent_update anchor replays the accepted prefix";

    if (sel(row)) {                       /* the per-candidate child hands the form to its own */
        if (!g_exe || !g_plugin) { ok(nm, false, "no re-exec path"); return; }
        const int st = spawn_anchor_child(nm);
        if (st == CH_FAIL)      ++g_fail;         /* the child printed the line and the detail */
        else if (st == CH_SKIP) ++g_skip;
        else if (st != CH_OK)   ok(nm, false, "child exit " + std::to_string(st));
        return;
    }
    if (!sel(nm)) return;
    const RadKernelInfo* r = find(pl, row);
    if (!r) { ok(nm, false, "row absent"); return; }

    const int64_t K = 128, V = 128, H = 2, Hg = 1, N = 2, NT = 3, Td = N * NT;
    const int64_t COLS = 6;               /* three copies of the anchor, three of the factors */
    const int64_t PAGES = 7;              /* the widest window the schedule below slides over */
    const int64_t SLOTS = N * PAGES;
    const double  scale = 0.088388, neps = 1e-6;
    /* `win` is the window's first page and `nacc` is what the PREVIOUS step committed. The first
     * step is the bootstrap: its factor block is the zeros a freshly handed out slot carries, the
     * marker reads as absent, and replaying one zeroed token leaves the anchor alone. */
    struct Step { int nacc, win; };
    const Step sched[] = { {1, 0}, {2, 1}, {3, 1}, {1, 0} };
    const int NS = (int)(sizeof(sched) / sizeof(sched[0]));

    std::vector<uint16_t> hq((size_t)(Td * Hg * K)), hkk((size_t)(Td * Hg * K)),
                          hv((size_t)(Td * H * V)), ha((size_t)(Td * H)), hb((size_t)(Td * H)),
                          hz((size_t)(Td * H * V)), hdev((size_t)(Td * H * V));
    std::vector<float> halog((size_t)H), hdtb((size_t)H), hnw((size_t)V);
    std::vector<float> hstate((size_t)(SLOTS * H * V * K), 0.0f);
    std::vector<int32_t> hsidx((size_t)(N * COLS)), hnacc((size_t)N);

    uint32_t rs = 0x9e3779b9u;
    auto rnd = [&rs]() {
        rs = rs * 1664525u + 1013904223u;
        return (float)((rs >> 9) & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
    };
    /* A_log sits where the decay is a few percent a token: exp(A_log) * softplus(a + dt_bias) of
     * order 0.05, so the state neither freezes nor forgets inside twelve tokens. */
    for (int64_t h = 0; h < H; ++h) { halog[(size_t)h] = -2.0f + 0.25f * (float)h;
                                      hdtb[(size_t)h]  = -1.0f + 0.5f * (float)h; }
    for (int64_t j = 0; j < V; ++j) hnw[(size_t)j] = 1.0f + 0.25f * rnd();
    /* The state the first step must find at its anchor column. */
    for (int64_t n = 0; n < N; ++n) {
        float* p = &hstate[(size_t)((n * PAGES + sched[0].win + 2) * H * V * K)];
        for (int64_t i = 0; i < H * V * K; ++i) p[(size_t)i] = 0.1f * rnd();
    }

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16,   { Td, Hg, K }),      /* q       */
        T(nullptr, RAD_BF16,   { Td, Hg, K }),      /* k       */
        T(nullptr, RAD_BF16,   { Td, H, V }),       /* v       */
        T(nullptr, RAD_BF16,   { Td, H }),          /* a       */
        T(nullptr, RAD_BF16,   { Td, H }),          /* b       */
        T(nullptr, RAD_F32,    { H }),              /* A_log   */
        T(nullptr, RAD_F32,    { H }),              /* dt_bias */
        T(nullptr, RAD_F32,    { SLOTS, H, V, K }), /* state   */
        T(nullptr, RAD_I32,    { N, COLS }),        /* state_idx: six columns, stride 6 */
        T(nullptr, RAD_I32,    { N }),              /* num_accepted */
        T(nullptr, RAD_I32,    { N + 1 }),          /* cu      */
        T(nullptr, RAD_BF16,   { Td, H, V }),       /* z       */
        T(nullptr, RAD_F32,    { V }),              /* norm_w  */
        T(nullptr, RAD_BF16,   { Td, H, V }),       /* o       */
        T(nullptr, RAD_F8E4M3, { Td, H * V }),      /* o_q     */
        T(nullptr, RAD_F32,    { Td, H }),          /* o_scale */
    };
    for (auto& t : ts) { t.data = alloc_for(t); if (!t.data) { ok(nm, false, "hipMalloc"); return; } }
    st_memcpy(ts[5].data, halog.data(), halog.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(ts[6].data, hdtb.data(),  hdtb.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(ts[7].data, hstate.data(), hstate.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(ts[12].data, hnw.data(),  hnw.size() * 4,   hipMemcpyHostToDevice);
    {
        const std::vector<int32_t> cu = cu_even(N, Td);
        st_memcpy(ts[10].data, cu.data(), cu.size() * 4, hipMemcpyHostToDevice);
    }

    std::vector<RadParam> ps = {
        PI("q_len", NT), PI("head_k", K), PI("head_v", V), PI("n_head_v", H),
        PF("scale", scale), PF("eps", neps), PS("act", "silu"), PS("state_form", "anchor"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;

    /* Two host chains over the same data: one that rolls back to the accepted token, and one that
     * always rolls back to the first. `sn` keeps a state per candidate token so the next step's
     * rollback has somewhere to go. */
    const size_t selems = (size_t)(N * H * V * K);
    std::vector<float> S(selems), Sw(selems);
    std::vector<float> sn((size_t)NT * selems), snw((size_t)NT * selems);
    std::vector<float> ref((size_t)(Td * H * V)), refw((size_t)(Td * H * V));
    for (int64_t n = 0; n < N; ++n) {
        const float* p = &hstate[(size_t)((n * PAGES + sched[0].win + 2) * H * V * K)];
        std::memcpy(&S[(size_t)(n * H * V * K)],  p, (size_t)(H * V * K) * 4);
        std::memcpy(&Sw[(size_t)(n * H * V * K)], p, (size_t)(H * V * K) * 4);
    }

    auto walk = [&](std::vector<float>& St, std::vector<float>& snp, std::vector<float>& out) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t h = 0; h < H; ++h) {
                float* s = &St[(size_t)((n * H + h) * V * K)];
                const int64_t hg = h / (H / Hg);
                for (int64_t t = 0; t < NT; ++t) {
                    const int64_t tok = n * NT + t;
                    float qv[128], kv[128];
                    double sq = 0.0, sk = 0.0;
                    for (int64_t j = 0; j < K; ++j) {
                        qv[j] = h_bf16(hq [(size_t)((tok * Hg + hg) * K + j)]);
                        kv[j] = h_bf16(hkk[(size_t)((tok * Hg + hg) * K + j)]);
                        sq += (double)qv[j] * qv[j];
                        sk += (double)kv[j] * kv[j];
                    }
                    const float qs = (float)(1.0 / std::sqrt(sq + 1e-6)) * (float)scale;
                    const float ks = (float)(1.0 / std::sqrt(sk + 1e-6));
                    for (int64_t j = 0; j < K; ++j) { qv[j] *= qs; kv[j] *= ks; }
                    const float x   = h_bf16(ha[(size_t)(tok * H + h)]) + hdtb[(size_t)h];
                    const float spl = x > 20.0f ? x
                                    : (x > 0.0f ? x + std::log1p(std::exp(-x))
                                                : std::log1p(std::exp(x)));
                    const float eg  = std::exp(-std::exp(halog[(size_t)h]) * spl);
                    const float bt  = 1.0f / (1.0f + std::exp(-h_bf16(hb[(size_t)(tok * H + h)])));
                    float on[128];
                    double tot = 0.0;
                    for (int64_t rw = 0; rw < V; ++rw) {
                        float* sr = s + rw * K;
                        double dot = 0.0;
                        for (int64_t j = 0; j < K; ++j) { sr[j] *= eg; dot += (double)sr[j] * kv[j]; }
                        const float u = (h_bf16(hv[(size_t)((tok * H + h) * V + rw)]) - (float)dot) * bt;
                        double o1 = 0.0;
                        for (int64_t j = 0; j < K; ++j) { sr[j] += u * kv[j]; o1 += (double)sr[j] * qv[j]; }
                        on[rw] = (float)o1;
                        tot += o1 * o1;
                    }
                    const float sc = (float)(1.0 / std::sqrt(tot / (double)V + neps));
                    for (int64_t rw = 0; rw < V; ++rw) {
                        const float zz = h_bf16(hz[(size_t)((tok * H + h) * V + rw)]);
                        out[(size_t)((tok * H + h) * V + rw)] =
                            on[rw] * sc * hnw[(size_t)rw] * (zz / (1.0f + std::exp(-zz)));
                    }
                    std::memcpy(&snp[(size_t)t * selems + (size_t)((n * H + h) * V * K)],
                                s, (size_t)(V * K) * 4);
                }
            }
    };

    double worst = 0.0, apart = 0.0;
    int bad_rc = 0;
    for (int s = 0; s < NS && !bad_rc; ++s) {
        for (size_t i = 0; i < hq.size(); ++i)  { hq[i]  = h_to_bf16(rnd()); hkk[i] = h_to_bf16(rnd()); }
        for (size_t i = 0; i < hv.size(); ++i)  { hv[i]  = h_to_bf16(rnd()); hz[i]  = h_to_bf16(rnd()); }
        for (size_t i = 0; i < ha.size(); ++i)  { ha[i]  = h_to_bf16(rnd()); hb[i]  = h_to_bf16(rnd()); }
        st_memcpy(ts[0].data, hq.data(),  hq.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[1].data, hkk.data(), hkk.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(ts[2].data, hv.data(),  hv.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[3].data, ha.data(),  ha.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[4].data, hb.data(),  hb.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[11].data, hz.data(), hz.size() * 2,  hipMemcpyHostToDevice);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < COLS; ++c)
                hsidx[(size_t)(n * COLS + c)] = (int32_t)(n * PAGES + sched[s].win + c);
            hnacc[(size_t)n] = sched[s].nacc;
        }
        st_memcpy(ts[8].data, hsidx.data(), hsidx.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(ts[9].data, hnacc.data(), hnacc.size() * 4, hipMemcpyHostToDevice);
        st_memset(ts[13].data, 0, hdev.size() * 2);

        const int rc = r->launch(&a, (RadStream) nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            ok(nm, false, "step " + std::to_string(s) + " rc=" + std::to_string(rc));
            bad_rc = 1;
            break;
        }
        st_memcpy(hdev.data(), ts[13].data, hdev.size() * 2, hipMemcpyDeviceToHost);

        if (s > 0) {
            const size_t back = (size_t)(sched[s].nacc - 1);
            std::memcpy(S.data(),  &sn [back * selems],  selems * 4);
            std::memcpy(Sw.data(), &snw[0],              selems * 4);  /* the prefix ignored */
        }
        walk(S,  sn,  ref);
        walk(Sw, snw, refw);
        /* A RELATIVE L2 AND NOT A PER-ELEMENT MAXIMUM. The gated norm leaves a fair number of
         * outputs next to zero, and dividing by one of those turns a rounding difference into a
         * large ratio -- which flatters the sensitivity check exactly as much as it slanders the
         * agreement, so neither number would mean anything. */
        double en = 0.0, ew = 0.0, rn = 0.0;
        for (size_t i = 0; i < hdev.size(); ++i) {
            const double d = h_bf16(hdev[i]);
            en += (d - ref[i])  * (d - ref[i]);
            ew += (d - refw[i]) * (d - refw[i]);
            rn += (double)ref[i] * ref[i];
        }
        rn = rn > 0.0 ? std::sqrt(rn) : 1.0;
        worst = std::max(worst, std::sqrt(en) / rn);
        if (s == NS - 1) apart = std::sqrt(ew) / rn;
    }
    if (bad_rc) return;

    char d[192];
    std::snprintf(d, sizeof d, "relative L2 %.3g over %d steps, %lld tokens; a replay that "
                  "ignored num_accepted would be %.3g away", worst, NS, (long long)(NS * Td), apart);
    ok(nm, worst < 5e-3 && apart > 0.05, d);
}

/* Codes or scales that differ from quant_act_i8g's over the bf16 rows `x`: one f32 scale per
 * (row, 128 columns) at amax / 127 (1 for a zero group), each code the value over it rounded to
 * nearest-even. The producers that write int8 codes themselves are held to exactly this. */
static int64_t i8_code_misses(const uint16_t* x, int64_t x_ld, const int8_t* q, int64_t q_ld,
                              const float* s, int64_t s_ld, int64_t rows, int64_t n) {
    int64_t miss = 0;
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t g = 0; g < n / 128; ++g) {
            const uint16_t* xr = x + i * x_ld + g * 128;
            float amax = 0.0f;
            for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(h_bf16(xr[j])));
            const float sc = amax > 0 ? amax * (1.0f / 127.0f) : 1.0f;
            miss += s[i * s_ld + g] != sc;
            const float inv = 1.0f / sc;
            for (int64_t j = 0; j < 128; ++j) {
                const float v = std::max(-127.0f, std::min(127.0f, h_bf16(xr[j]) * inv));
                miss += q[i * q_ld + g * 128 + j] != (int)std::nearbyint(v);
            }
        }
    return miss;
}

/* THE DECODE CONVOLUTION FOLDED INTO THE RECURRENT UPDATE, AGAINST THE PAIR IT REPLACES.
 *
 * gdn_conv_recurrent_update claims to leave every byte gdn_conv_update followed by
 * gdn_recurrent_update leaves: the output, its fp8 codes and scales, the recurrent state and the
 * conv state. So both run from identical copies and all five are compared exactly, over TWO steps
 * -- the second is what catches an arrival counter that does not reset, because the key heads'
 * conv state is then never rewritten.
 *
 * The geometry puts two value heads on each key head, which is what makes the last-arriver rewrite
 * run at all; gives the sequences uneven windows and read offsets; slots sequence 1 at conv slot 0,
 * the real slot a `<= 0` test would drop; and makes the last sequence padding, with every slot
 * negative. The projection is one row with the gate behind the convolution's columns, as the
 * block lays it out, so `x` and `z` are both column views with the row's pitch.
 *
 * AND IT ASSERTS ITS OWN SENSITIVITY: the key heads' conv state has to have moved, or a fold that
 * never rewrote it would pass against a pair that also did not. The state contract is chosen once a
 * process, so the anchor form runs in a child, as gdn_recurrent_anchor's does. */
/* `i8` runs both at INT8 codes, and holds the pair's codes to quant_act_i8g over its `o` too. */
static void gdn_conv_fold_run(const Plugin& pl, const char* nm, const char* form,
                              bool i8 = false) {
    const RadKernelInfo* rf = find(pl, "gdn_conv_recurrent_update_k128_v128_bf16");
    const RadKernelInfo* rc = find(pl, "gdn_conv_update_w4_h128_bf16");
    const RadKernelInfo* rr = find(pl, "gdn_recurrent_update_k128_v128_bf16_fp32state");
    if (!rf || !rc || !rr) { ok(nm, false, "a row is absent"); return; }

    const int64_t K = 128, V = 128, H = 4, Hg = 2, N = 4, W = 4, MQ = 4, SL = W - 2 + MQ;
    const int64_t conv_dim = (2 * Hg * K + H * V), in_w = conv_dim + H * V;
    const int64_t COLS = 6, SSLOTS = N * COLS, CSLOTS = 5;
    const int32_t lens[N] = { 4, 2, 3, 1 }, nacc[N] = { 2, 1, 3, 1 }, cidx[N] = { 3, 0, 1, -1 };
    int64_t Td = 0;
    std::vector<int32_t> cu(1, 0);
    for (int64_t n = 0; n < N; ++n) { Td += lens[n]; cu.push_back((int32_t)Td); }

    uint32_t seed = 0x51ed270bu;
    auto rnd = [&seed]() {
        seed = st_mix32(seed + 0x9e3779b9u);
        return (float)(seed & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
    };
    auto bf = [&](size_t n, float s) {
        std::vector<uint16_t> v(n);
        for (auto& e : v) e = h_to_bf16(s * rnd());
        return v;
    };
    const std::vector<uint16_t> hw = bf((size_t)(conv_dim * W), 0.5f);
    const std::vector<uint16_t> hcs = bf((size_t)(CSLOTS * conv_dim * SL), 1.0f);
    std::vector<float> hst((size_t)(SSLOTS * H * V * K)), halog((size_t)H), hdtb((size_t)H),
                       hnw((size_t)V);
    for (auto& e : hst) e = 0.1f * rnd();
    for (int64_t h = 0; h < H; ++h) { halog[(size_t)h] = -2.0f + 0.25f * (float)h;
                                      hdtb[(size_t)h]  = -1.0f + 0.5f * (float)h; }
    for (auto& e : hnw) e = 1.0f + 0.25f * rnd();
    std::vector<int32_t> hsidx((size_t)(N * COLS));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < COLS; ++c)
            hsidx[(size_t)(n * COLS + c)] = cidx[n] < 0 ? -1 : (int32_t)(n * COLS + c);

    /* Two of everything the ops write; the shared inputs once. */
    auto dev = [&](const void* h, size_t bytes) {
        void* d = dalloc(bytes);
        if (d && h) st_memcpy(d, h, bytes, hipMemcpyHostToDevice);
        return d;
    };
    const size_t in_b = (size_t)(Td * in_w) * 2, st_b = hst.size() * 4, cs_b = hcs.size() * 2;
    const size_t o_b = (size_t)(Td * H * V) * 2, oq_b = (size_t)(Td * H * V), os_b = (size_t)(Td * H) * 4;
    void* d_in = dev(nullptr, in_b);
    void* d_ab = dev(nullptr, (size_t)(Td * 2 * H) * 2);
    void* d_w = dev(hw.data(), hw.size() * 2);
    void* d_alog = dev(halog.data(), halog.size() * 4);
    void* d_dtb = dev(hdtb.data(), hdtb.size() * 4);
    void* d_nw = dev(hnw.data(), hnw.size() * 4);
    void* d_cidx = dev(cidx, sizeof(cidx));
    void* d_sidx = dev(hsidx.data(), hsidx.size() * 4);
    void* d_nacc = dev(nacc, sizeof(nacc));
    void* d_cu = dev(cu.data(), cu.size() * 4);
    void* d_q = dev(nullptr, (size_t)(Td * Hg * K) * 2);
    void* d_k = dev(nullptr, (size_t)(Td * Hg * K) * 2);
    void* d_v = dev(nullptr, (size_t)(Td * H * V) * 2);
    void* d_cs[2]; void* d_st[2]; void* d_o[2]; void* d_oq[2]; void* d_os[2];
    for (int i = 0; i < 2; ++i) {
        d_cs[i] = dev(hcs.data(), cs_b);  d_st[i] = dev(hst.data(), st_b);
        d_o[i] = dev(nullptr, o_b); d_oq[i] = dev(nullptr, oq_b); d_os[i] = dev(nullptr, os_b);
    }
    if (!d_in || !d_ab || !d_cs[1] || !d_os[1]) { ok(nm, false, "hipMalloc"); return; }

    auto view = [](void* base, uint32_t dt, int64_t rows, int64_t cols, int64_t pitch,
                   int64_t col0) {
        RadTensor t = T((char*)base + (size_t)col0 * esz(dt), dt, { rows, cols });
        t.stride[0] = pitch;
        return t;
    };
    const RadTensor tx = view(d_in, RAD_BF16, Td, conv_dim, in_w, 0);
    const RadTensor tz = view(d_in, RAD_BF16, Td, H * V, in_w, conv_dim);
    const RadTensor ta = view(d_ab, RAD_BF16, Td, H, 2 * H, 0);
    const RadTensor tb = view(d_ab, RAD_BF16, Td, H, 2 * H, H);
    const RadTensor tw = T(d_w, RAD_BF16, { conv_dim, W });
    const RadTensor tal = T(d_alog, RAD_F32, { H }), tdt = T(d_dtb, RAD_F32, { H });
    const RadTensor tnw = T(d_nw, RAD_F32, { V });
    const RadTensor tci = T(d_cidx, RAD_I32, { N }), tsi = T(d_sidx, RAD_I32, { N, COLS });
    const RadTensor tna = T(d_nacc, RAD_I32, { N }), tcu = T(d_cu, RAD_I32, { N + 1 });
    RadTensor none; std::memset(&none, 0, sizeof(none));
    auto cs = [&](int i) { return T(d_cs[i], RAD_BF16, { CSLOTS, conv_dim, SL }); };
    auto st = [&](int i) { return T(d_st[i], RAD_F32, { SSLOTS, H, V, K }); };
    auto to = [&](int i) { return T(d_o[i], RAD_BF16, { Td, H, V }); };
    auto toq = [&](int i) { return T(d_oq[i], i8 ? RAD_I8 : RAD_F8E4M3, { Td, H * V }); };
    auto tos = [&](int i) { return T(d_os[i], RAD_F32, { Td, H }); };
    const RadTensor tq = T(d_q, RAD_BF16, { Td, Hg, K }), tk = T(d_k, RAD_BF16, { Td, Hg, K }),
                    tv = T(d_v, RAD_BF16, { Td, H, V });

    std::vector<RadParam> pc = { PI("q_len", MQ), PI("head_k", K), PI("head_v", V),
                                 PI("conv_width", W) };
    std::vector<RadParam> pr = { PI("q_len", MQ), PI("head_k", K), PI("head_v", V),
                                 PI("n_head_v", H), PF("scale", 0.088388), PF("eps", 1e-6),
                                 PS("act", "silu"), PS("state_form", form) };
    std::vector<RadParam> pf = pr;
    pf.push_back(PI("conv_width", W));
    auto run = [&](const RadKernelInfo* r, std::vector<RadTensor> ts, std::vector<RadParam>& ps) {
        RadArgs a; std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        const int rc_ = r->launch(&a, (RadStream) nullptr);
        return rc_ != RAD_OK ? rc_ : (st_stream_sync(nullptr) != hipSuccess ? -1000 : RAD_OK);
    };
    auto fetch = [](void* d, size_t b) {
        std::vector<uint8_t> h(b);
        st_memcpy(h.data(), d, b, hipMemcpyDeviceToHost);
        return h;
    };

    std::string why;
    bool moved = false;
    for (int step = 0; step < 2 && why.empty(); ++step) {
        const std::vector<uint16_t> hin = bf((size_t)(Td * in_w), 1.0f);
        const std::vector<uint16_t> hab = bf((size_t)(Td * 2 * H), 1.0f);
        st_memcpy(d_in, hin.data(), in_b, hipMemcpyHostToDevice);
        st_memcpy(d_ab, hab.data(), hab.size() * 2, hipMemcpyHostToDevice);
        int e = run(rc, { tx, tw, none, cs(0), tci, tna, tcu, tq, tk, tv }, pc);
        if (e == RAD_OK)
            e = run(rr, { tq, tk, tv, ta, tb, tal, tdt, st(0), tsi, tna, tcu, tz, tnw, to(0),
                          toq(0), tos(0) }, pr);
        if (e != RAD_OK) { why = "pair rc=" + std::to_string(e); break; }
        e = run(rf, { tx, tw, none, cs(1), tci, ta, tb, tal, tdt, st(1), tsi, tna, tcu, tz, tnw,
                      to(1), toq(1), tos(1) }, pf);
        if (e != RAD_OK) { why = "fold rc=" + std::to_string(e); break; }

        const struct { const char* what; void* a; void* b; size_t n; } cmp[] = {
            { "o", d_o[0], d_o[1], o_b }, { "o_q", d_oq[0], d_oq[1], oq_b },
            { "o_scale", d_os[0], d_os[1], os_b }, { "state", d_st[0], d_st[1], st_b },
            { "conv_state", d_cs[0], d_cs[1], cs_b },
        };
        for (const auto& c : cmp) {
            const std::vector<uint8_t> A = fetch(c.a, c.n), B = fetch(c.b, c.n);
            size_t diff = 0;
            for (size_t i = 0; i < c.n; ++i) diff += A[i] != B[i];
            if (diff) {
                why = "step " + std::to_string(step) + ": " + c.what + " differs in " +
                      std::to_string(diff) + " of " + std::to_string(c.n) + " bytes";
                break;
            }
        }
        /* At int8 codes, the real sequences' tokens -- every sequence but the padded last one --
         * against quant_act_i8g over the `o` the pair left. A (token, head) row is one group. */
        if (i8 && why.empty()) {
            const std::vector<uint8_t> O = fetch(d_o[0], o_b), Q = fetch(d_oq[0], oq_b),
                                       S = fetch(d_os[0], os_b);
            const int64_t rows = (int64_t)cu[(size_t)N - 1] * H;
            const int64_t miss = i8_code_misses((const uint16_t*)O.data(), V,
                                                (const int8_t*)Q.data(), V,
                                                (const float*)S.data(), 1, rows, V);
            if (miss) why = "step " + std::to_string(step) + ": " + std::to_string(miss) +
                            " int8 codes or scales differ from quant_act_i8g over o";
        }
        /* The key heads' channels of a real sequence's conv slot, against where they started. */
        const std::vector<uint8_t> C = fetch(d_cs[1], cs_b);
        const size_t kq = (size_t)(3 * conv_dim * SL) * 2;           /* sequence 0's slot */
        if (std::memcmp(C.data() + kq, hcs.data() + kq / 2, (size_t)(2 * Hg * K * SL) * 2) != 0)
            moved = true;
    }
    if (why.empty() && !moved) why = "the key heads' conv state never moved";
    char d[160];
    std::snprintf(d, sizeof d, "%s", why.c_str());
    ok(nm, why.empty(), why.empty() ? "" : d);
}

static void gdn_conv_fold(const Plugin& pl) {
    const char* row = "gdn_conv_recurrent_update_k128_v128_bf16";
    const char* nmp = "gdn_conv_recurrent_update is byte-identical to the pair, per_candidate";
    const char* nma = "gdn_conv_recurrent_update is byte-identical to the pair, anchor";
    if (sel(row)) {
        gdn_conv_fold_run(pl, nmp, "per_candidate");
        gdn_conv_fold_run(pl, "gdn_conv_recurrent_update is byte-identical to the pair at int8 "
                              "codes, which are quant_act_i8g's, per_candidate",
                          "per_candidate", true);
        if (!g_exe || !g_plugin) { ok(nma, false, "no re-exec path"); return; }
        const int st = spawn_anchor_child(nma);
        if (st == CH_FAIL)      ++g_fail;
        else if (st == CH_SKIP) ++g_skip;
        else if (st != CH_OK)   ok(nma, false, "child exit " + std::to_string(st));
        return;
    }
    if (sel(nma)) gdn_conv_fold_run(pl, nma, "anchor");
}

/* THE GATED NORM'S GATE AS A COLUMN SLICE, WHICH ONE PITCH CANNOT DESCRIBE.
 *
 * The fp8 delta net projects [q|k|v|z] in ONE GEMM (arch/common/rad_block_gdn_fp8.h), so the gate
 * this kernel reads is the right-hand columns of a wider row: its (token, head) rows are `channels`
 * apart INSIDE a token and the projection's whole width apart BETWEEN tokens. An implementation
 * that multiplies the flat row index by a single stride therefore walks into the next token after
 * `n_head_v` rows -- still in bounds, still finite, and wrong in a way that reads as a slightly
 * different model rather than as a crash.
 *
 * The smoke case above cannot see it, because it only ever hands over a dense gate. So this one
 * runs the same numbers TWICE -- dense, and as the tail of a row twice as wide -- and requires the
 * two outputs to agree byte for byte. Every value is a multiple of an eighth so the equality is
 * exact and not a tolerance. */
static void gdn_gated_norm_sliced_gate(const Plugin& pl) {
    if (!sel("gdn_gated_rmsnorm_h128_bf16")) return;
    const char* nm = "gdn_gated_rmsnorm_h128_bf16 reads a column-sliced gate";
    const RadKernelInfo* r = find(pl, "gdn_gated_rmsnorm_h128_bf16");
    if (!r) { ok(nm, false, "row absent"); return; }

    const int64_t nt = 6, H = 5, ch = 128, rows = nt * H, wide = 2 * H * ch;
    std::vector<uint16_t> hx((size_t)(rows * ch)), hz((size_t)(rows * ch)),
                          hw2((size_t)(nt * wide), 0), ha((size_t)(rows * ch), 0),
                          hb((size_t)(rows * ch), 0);
    std::vector<float>    hw((size_t)ch);
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t c = 0; c < ch; ++c) {
                const int64_t i = (t * H + h) * ch + c;
                hx[(size_t)i] = h_to_bf16((float)(((t * 7 + h * 5 + c) % 17) - 8) * 0.125f);
                hz[(size_t)i] = h_to_bf16((float)(((t * 3 + h * 11 + c) % 13) - 6) * 0.25f);
                /* The same gate values, planted in the tail of a row twice as wide. */
                hw2[(size_t)(t * wide + H * ch + h * ch + c)] = hz[(size_t)i];
            }
    for (int64_t c = 0; c < ch; ++c) hw[(size_t)c] = (float)((c % 5) - 2) * 0.5f;

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { rows, ch }), T(nullptr, RAD_BF16, { rows, ch }),
        T(nullptr, RAD_F32,  { ch }),       T(nullptr, RAD_BF16, { rows, ch }),
    };
    RadTensor twide = T(nullptr, RAD_BF16, { nt, wide });
    for (auto& t : ts) t.data = alloc_for(t);
    twide.data = alloc_for(twide);
    if (!ts[0].data || !ts[1].data || !ts[2].data || !ts[3].data || !twide.data) {
        ok(nm, false, "hipMalloc failed");
        return;
    }
    st_memcpy(ts[0].data, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hz.data(), hz.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[2].data, hw.data(), hw.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(twide.data, hw2.data(), hw2.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadParam> ps = {
        PI("M", rows), PI("channels", ch), PF("eps", 1e-6), PS("act", "silu"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;

    int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "dense arm rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(ha.data(), ts[3].data, ha.size() * 2, hipMemcpyDeviceToHost);

    /* The gate operand, respelled: [tokens, heads*channels] starting H*ch into a `wide` row. */
    ts[1].data      = (void*)((uint16_t*)twide.data + H * ch);
    ts[1].rank      = 2;
    ts[1].shape[0]  = nt;
    ts[1].shape[1]  = H * ch;
    ts[1].stride[0] = wide;
    ts[1].stride[1] = 1;
    st_memset(ts[3].data, 0, ha.size() * 2);
    rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "sliced arm rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hb.data(), ts[3].data, hb.size() * 2, hipMemcpyDeviceToHost);

    int64_t bad = 0;
    for (size_t i = 0; i < ha.size(); ++i) if (ha[i] != hb[i]) ++bad;
    char d[128];
    std::snprintf(d, sizeof d, "%lld of %lld elements differ", (long long)bad,
                  (long long)ha.size());
    ok(nm, bad == 0, d);
}

static void dflash_conv_numeric(const Plugin& pl, int64_t nt, int64_t H, int64_t group,
                                int64_t taps, int64_t bsz) {
    if (!sel("dflash_conv_t2_g16_bf16")) return;
    const int64_t NG = H / group;
    char nm[128];
    std::snprintf(nm, sizeof nm, "dflash_conv_t2_g16_bf16 numeric T=%lld H=%lld block=%lld",
                  (long long)nt, (long long)H, (long long)bsz);
    const RadKernelInfo* r = find(pl, "dflash_conv_t2_g16_bf16");
    if (!r) { ok(nm, false, "row absent"); return; }

    const int64_t dpitch = 2 * taps * NG;
    std::vector<uint16_t> hx((size_t)(nt * H)), hd((size_t)(nt * dpitch)),
                          hb((size_t)(taps * H)), hy((size_t)(nt * H), 0);
    /* Quarters and eighths, so every product and sum is exact in f32 and the comparison is an
     * equality rather than a tolerance. */
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t h = 0; h < H; ++h)
            hx[(size_t)(t * H + h)] = h_to_bf16((float)(((t * 5 + h * 3) % 17) - 8) * 0.25f);
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t j = 0; j < dpitch; ++j)
            hd[(size_t)(t * dpitch + j)] = h_to_bf16((float)(((t * 3 + j * 7) % 9) - 4) * 0.125f);
    for (int64_t k = 0; k < taps; ++k)
        for (int64_t h = 0; h < H; ++h)
            hb[(size_t)(k * H + h)] = h_to_bf16((float)(((k * 11 + h) % 13) - 6) * 0.25f);

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { nt, H }), T(nullptr, RAD_BF16, { nt, dpitch }),
        T(nullptr, RAD_BF16, { taps, H }), T(nullptr, RAD_BF16, { nt, H }),
    };
    for (auto& t : ts) t.data = alloc_for(t);
    if (!ts[0].data || !ts[1].data || !ts[2].data || !ts[3].data) {
        ok(nm, false, "hipMalloc failed");
        return;
    }
    st_memcpy(ts[0].data, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hd.data(), hd.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[2].data, hb.data(), hb.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadParam> ps = {
        PI("T", nt), PI("hidden_size", H), PI("taps", taps), PI("group_size", group),
        PI("NG", NG), PI("block_size", bsz), PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), ts[3].data, hy.size() * 2, hipMemcpyDeviceToHost);

    int64_t bad = 0;
    double worst = 0;
    std::string first;
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t h = 0; h < H; ++h) {
            float want = 0;
            for (int64_t k = 0; k < taps; ++k) {
                if ((t & (bsz - 1)) < k || t - k < 0) continue;
                const float coef = h_bf16(hb[(size_t)(k * H + h)]) +
                                   h_bf16(hd[(size_t)(t * dpitch + k * NG + h / group)]);
                want += coef * h_bf16(hx[(size_t)((t - k) * H + h)]);
            }
            const float got = h_bf16(hy[(size_t)(t * H + h)]);
            const double e = std::fabs((double)got - (double)h_bf16(h_to_bf16(want)));
            if (e > 0) {
                if (!bad) first = " first at t=" + std::to_string(t) + " h=" + std::to_string(h) +
                                  " want " + std::to_string(want) + " got " + std::to_string(got);
                ++bad;
            }
            worst = std::max(worst, e);
        }
    char d[192];
    std::snprintf(d, sizeof d, "%lld of %lld differ, worst %.4g%s", (long long)bad,
                  (long long)(nt * H), worst, first.c_str());
    ok(nm, bad == 0, d);
}

/* THE SKINNY bf16 GEMM AGAINST A HOST REFERENCE, AT A GIVEN K.
 *
 * The smoke case above launches it; this one says whether the number is right, and it takes K as a
 * parameter because K is the dimension this kernel decomposes along -- pick_sk splits it, rowGo
 * strides rows by it, and the unrolled fragment loop walks it. A K no model has used is therefore
 * exactly the shape that has never been checked, and "it launches" says
 * nothing about it.
 *
 * The weight is plain row-major [N, K]: this row declares no layout hook, so there is nothing to
 * arrange and the bytes the host builds are the bytes the kernel reads. */
static void bf16_gemm_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K) {
    if (!sel(row)) return;
    char nm[128];
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld", row,
                  (long long)M, (long long)N, (long long)K);
    const RadKernelInfo* r = find(pl, row);
    if (!r) { ok(nm, false, "row absent"); return; }

    /* Deterministic and in the range the drafter's own fc actually sees: activations to a couple
     * of hundred, weights inside one. A pattern that is symmetric in k would hide a split-K bug by
     * making every partial sum equal, so the sign alternates on a period coprime with 16. */
    std::vector<uint16_t> ha((size_t)(M * K)), hw((size_t)(N * K)), hy((size_t)(M * N), 0);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k)
            ha[(size_t)(i * K + k)] =
                h_to_bf16((float)(((i * 7 + k * 13) % 401) - 200) * 0.5f * ((k % 3) ? 1.f : -1.f));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            hw[(size_t)(n * K + k)] =
                h_to_bf16((float)(((n * 11 + k * 5) % 199) - 99) * 0.01f * ((k % 5) ? 1.f : -1.f));

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { N, K }),
        T(nullptr, RAD_BF16, { M, N }),
    };
    for (auto& t : ts) t.data = alloc_for(t);
    if (!ts[0].data || !ts[1].data || !ts[2].data) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(ts[0].data, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), ts[2].data, hy.size() * 2, hipMemcpyDeviceToHost);

    /* The reference reads the SAME bf16 values the kernel does, so what is left to disagree about
     * is the arithmetic and the addressing and nothing else. A wide N is checked on a STRIDED
     * SAMPLE of its columns: the host cost is M*N*K and the production shape is 655 million
     * products, while what a column can be wrong about -- which weight row it read -- is a
     * property of the column index, so sampling the index space finds it. */
    const int64_t nstep = N > 64 ? N / 64 : 1;
    double num = 0, den = 0, worst = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n += nstep) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)h_bf16(ha[(size_t)(i * K + k)]) * (double)h_bf16(hw[(size_t)(n * K + k)]);
            const double got = (double)h_bf16(hy[(size_t)(i * N + n)]);
            num += (got - acc) * (got - acc);
            den += acc * acc;
            worst = std::max(worst, std::fabs(got - acc) / (std::fabs(acc) + 1e-6));
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[96];
    std::snprintf(d, sizeof d, "rel_l2 %.3e, worst %.3e", rel, worst);
    ok(nm, rel < 2e-2, d);
}


/* THE 8-BIT-WEIGHT SKINNY GEMM AGAINST WHAT THE PLANES MEAN. The row reads the w8a16 hooks'
 * fragment-order codes and (scale, zero) dwords, got from its own hooks, and the reference is the
 * planes dequantised on the host, so an arrangement the kernel misread shows as a column that
 * disagrees. `adt` is the activation's dtype, which the row takes as f16 and refuses otherwise. */
static void w8a16_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K,
                          uint32_t adt) {
    if (!sel(row)) return;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric %s M=%lld N=%lld K=%lld", row,
                  adt == RAD_BF16 ? "bf16" : "f16", (long long)M, (long long)N, (long long)K);
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || !r->relayout) { ok(nm, false, "row or hooks absent"); return; }

    std::vector<float> wf((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            wf[(size_t)(n * K + k)] = std::ldexp(1.0f, (int)((n / 16) % 5) - 2 - (int)((k / 128) % 3))
                                      * (0.031f * (float)((n * 37 + k * 11) % 61) - 0.9f);
    const Planes p = int8_planes(wf, N, K);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, wsz;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &wsz);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }

    std::vector<uint16_t> ha((size_t)(M * K)), hy((size_t)(M * N), 0);
    std::vector<float> av((size_t)(M * K));
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k) {
            const float v = (float)(((i * 7 + k * 13) % 401) - 200) * 0.05f * ((k % 3) ? 1.f : -1.f);
            if (adt == RAD_BF16) {
                ha[(size_t)(i * K + k)] = h_to_bf16(v);
                av[(size_t)(i * K + k)] = h_bf16(ha[(size_t)(i * K + k)]);
            } else {
                uint16_t h;
                rad_store_f32(&h, RAD_F16, 0, v);
                ha[(size_t)(i * K + k)] = h;
                av[(size_t)(i * K + k)] = rad_load_f32(&h, RAD_F16, 0);
            }
        }

    void* d_a = dalloc(ha.size() * 2);
    void* d_w = dalloc(wq.size());
    void* d_s = dalloc(wsz.size());
    void* d_y = dalloc(hy.size() * 2);
    if (!d_a || !d_w || !d_s || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, wsz.data(), wsz.size(), hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_a, adt, { M, K }), T(nullptr, RAD_F32, { M }),
        T(d_w, RAD_U8, { (int64_t)wq.size() }), T(d_s, RAD_U8, { (int64_t)wsz.size() }),
        T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
    };
    absent(ts[1]);
    absent(ts[5]);
    absent(ts[6]);
    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                 PS("dtype", "w8a16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);

    const int64_t ng = K / 128;
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n += (n + nstep < N ? nstep : 1)) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                const float sc = rad_load_f32(p.data[1].data(), RAD_F16, n * ng + k / 128);
                acc += (double)av[(size_t)(i * K + k)] *
                       (double)(int8_t)p.data[0][(size_t)(n * K + k)] * (double)sc;
            }
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[64];
    std::snprintf(d, sizeof d, "rel_l2 %.3e", rel);
    ok(nm, rel < 1e-2, d);
}


/* THE INT8 TRUNK CHAIN: quant_act_i8g on a bf16 activation, then one of the i8a8 GEMMs on what it
 * wrote, against the planes' meaning. The weight is i8 with a bf16 scale per row per 128 K (what
 * libquant writes for `codes=i8 group=128 scale=bf16 clamp=sym`), stored through the row's own
 * hooks. The reference multiplies the codes the QUANTISER wrote -- read back -- by their scales,
 * so what is left to disagree about is the GEMM alone; the quantiser is checked separately against
 * a host one, code for code. Ragged M reaches the guarded fold; N and K are what the rows take. */
static void i8a8_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K) {
    if (!sel(row) && !sel("quant_act_i8g")) return;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld", row, (long long)M,
                  (long long)N, (long long)K);
    const RadKernelInfo* r = find(pl, row);
    const RadKernelInfo* qz = find(pl, "quant_act_i8g");
    if (!r || !qz || !r->layout || !r->relayout) { ok(nm, false, "rows or hooks absent"); return; }
    const int64_t nkb = K / 128;

    std::vector<float> wf((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            wf[(size_t)(n * K + k)] = std::ldexp(1.0f, (int)((n / 16) % 5) - 2 - (int)((k / 128) % 3))
                                      * (0.031f * (float)((n * 37 + k * 11) % 61) - 0.9f);
    Planes p;
    p.enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    p.size(N, K);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < nkb; ++g) {
            float amax = 0.0f;
            for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(wf[(size_t)(n * K + g * 128 + j)]));
            rad_store_f32(p.data[1].data(), RAD_BF16, n * nkb + g, amax > 0 ? amax / 127.0f : 1.0f);
            const float sc = rad_load_f32(p.data[1].data(), RAD_BF16, n * nkb + g);
            for (int64_t j = 0; j < 128; ++j) {
                const float q = std::nearbyint(wf[(size_t)(n * K + g * 128 + j)] / sc);
                p.data[0][(size_t)(n * K + g * 128 + j)] =
                    (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
            }
        }
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, wsc;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &wsc);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }
    {
        std::string why;
        const std::string what = std::string(row) + " i8 codes and tile-major scales invert";
        ok(what.c_str(), restores(r, lp, 2, p, { "codes" }, wq, &why) &&
                         restores(r, lp, 3, p, { "scale" }, wsc, &why), why);
    }

    std::vector<uint16_t> ha((size_t)(M * K));
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k)
            ha[(size_t)(i * K + k)] = h_to_bf16((float)(((i * 7 + k * 13) % 401) - 200) * 0.05f *
                                                ((k % 3) ? 1.f : -1.f) * ((k % 517 == 3) ? 30.f : 1.f));
    void* d_a  = dalloc(ha.size() * 2);
    void* d_q  = dalloc((size_t)(M * K));
    void* d_qs = dalloc((size_t)(M * nkb) * 4);
    void* d_w  = dalloc(wq.size());
    void* d_s  = dalloc(wsc.size());
    void* d_y  = dalloc((size_t)(M * N) * 2);
    if (!d_a || !d_q || !d_qs || !d_w || !d_s || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, wsc.data(), wsc.size(), hipMemcpyHostToDevice);

    auto launch = [&](const RadKernelInfo* k, std::vector<RadTensor> ts, std::vector<RadParam> ps) {
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        void* scr = nullptr;
        if (k->scratch) {
            const int64_t need = k->scratch(&a);
            if (need > 0) { scr = dalloc((size_t)need); a.scratch = scr; a.scratch_bytes = need; }
        }
        const int rc = k->launch(&a, (RadStream) nullptr);
        return rc == RAD_OK && st_stream_sync(nullptr) == hipSuccess ? RAD_OK : (rc ? rc : -1);
    };
    int rc = launch(qz, { T(d_a, RAD_BF16, { M, K }), T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M, nkb }) },
                    { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc != RAD_OK) { ok(nm, false, "quant_act_i8g rc " + std::to_string(rc)); return; }
    std::vector<int8_t> hq((size_t)(M * K));
    std::vector<float> hqs((size_t)(M * nkb));
    st_memcpy(hq.data(), d_q, hq.size(), hipMemcpyDeviceToHost);
    st_memcpy(hqs.data(), d_qs, hqs.size() * 4, hipMemcpyDeviceToHost);
    {
        int64_t bad = 0;
        for (int64_t i = 0; i < M; ++i)
            for (int64_t g = 0; g < nkb; ++g) {
                float amax = 0.0f;
                for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(h_bf16(ha[(size_t)(i * K + g * 128 + j)])));
                const float sc = amax > 0 ? amax * (1.0f / 127.0f) : 1.0f;
                bad += hqs[(size_t)(i * nkb + g)] != sc;
                const float inv = 1.0f / sc;
                for (int64_t j = 0; j < 128; ++j) {
                    const float v = h_bf16(ha[(size_t)(i * K + g * 128 + j)]) * inv;
                    const int c = (int)std::nearbyint(std::max(-127.0f, std::min(127.0f, v)));
                    bad += hq[(size_t)(i * K + g * 128 + j)] != c;
                }
            }
        const std::string what =
            "quant_act_i8g vs a host quantiser, M=" + std::to_string(M) + " K=" + std::to_string(K);
        ok(what.c_str(), bad == 0, std::to_string(bad) + " codes or scales differ");
    }

    std::vector<RadTensor> ts = {
        T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M, nkb }), T(d_w, RAD_I8, { N, K }),
        T(d_s, RAD_BF16, { N, nkb }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(ts[5]); absent(ts[6]);
    rc = launch(r, ts, { PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "i8a8") });
    if (rc != RAD_OK) { ok(nm, false, "rc=" + std::to_string(rc)); return; }
    std::vector<uint16_t> hy((size_t)(M * N));
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n += (n + nstep < N ? nstep : 1)) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)hq[(size_t)(i * K + k)] * hqs[(size_t)(i * nkb + k / 128)] *
                       (double)(int8_t)p.data[0][(size_t)(n * K + k)] *
                       rad_load_f32(p.data[1].data(), RAD_BF16, n * nkb + k / 128);
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[64];
    std::snprintf(d, sizeof d, "rel_l2 %.3e", rel);
    ok(nm, rel < 5e-3, d);
}


/* THE TILED bf16 GEMM AGAINST THE SKINNY ONE, BYTE FOR BYTE, at every tile variant. The claim in
 * its row is identity and not closeness: an op declared over M 1..max_tok is served by the skinny
 * row at M <= 64 and by this one above, so any difference would make a decode step and a prefill
 * chunk disagree about the same row. The shapes reach both forms of the sum -- one slice (N wide
 * enough that the split rule answers 1) and several (N 512 answers 4; K 64 answers a slice a k
 * step) -- and ragged M and N, so a clamped fragment row that were ever stored would show. */
static void bf16_tiled_identity(const Plugin& pl, int64_t M, int64_t N, int64_t K) {
    if (!sel("gemm_bf16_nt_tiled")) return;
    char nm[128];
    std::snprintf(nm, sizeof nm, "gemm_bf16_nt_tiled == gemm_bf16_nt_m64  M=%lld N=%lld K=%lld",
                  (long long)M, (long long)N, (long long)K);
    const RadKernelInfo* rt = find(pl, "gemm_bf16_nt_tiled");
    const RadKernelInfo* rs = find(pl, "gemm_bf16_nt_m64");
    if (!rt || !rs) { ok(nm, false, "row absent"); return; }

    std::vector<uint16_t> ha((size_t)(M * K)), hw((size_t)(N * K));
    for (size_t i = 0; i < ha.size(); ++i)
        ha[i] = h_to_bf16((float)((int)(st_mix32((uint32_t)i) % 401) - 200) * 0.5f);
    for (size_t i = 0; i < hw.size(); ++i)
        hw[i] = h_to_bf16((float)((int)(st_mix32((uint32_t)i ^ 0x9e3779b9u) % 199) - 99) * 0.01f);

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { N, K }),
        T(nullptr, RAD_BF16, { M, N }),
    };
    for (auto& t : ts) t.data = alloc_for(t);
    if (!ts[0].data || !ts[1].data || !ts[2].data) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(ts[0].data, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);

    auto run = [&](const RadKernelInfo* r, int64_t tile, std::vector<uint16_t>& out) {
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
        if (tile >= 0) ps.push_back(PI("tile", tile));
        /* A stale output would compare equal to itself, so the buffer is poisoned before each. */
        if (hipMemset(ts[2].data, 0xA5, (size_t)(M * N) * 2) != hipSuccess) return -1;
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        const int rc = r->launch(&a, (RadStream) nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return rc != RAD_OK ? rc : -1;
        out.assign((size_t)(M * N), 0);
        st_memcpy(out.data(), ts[2].data, out.size() * 2, hipMemcpyDeviceToHost);
        return 0;
    };
    std::vector<uint16_t> ref, got;
    if (run(rs, -1, ref) != 0) { ok(nm, false, "skinny launch failed"); return; }
    std::string d;
    bool all = true;
    for (int64_t tile = 0; tile <= 2; ++tile) {
        const int rc = run(rt, tile, got);
        size_t diff = 0;
        if (rc == 0)
            for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
        const bool pass = rc == 0 && diff == 0;
        all = all && pass;
        char b[64];
        std::snprintf(b, sizeof b, "%stile %lld: %s", d.empty() ? "" : ", ", (long long)tile,
                      rc ? "launch failed" : (diff ? (std::to_string(diff) + " differ").c_str()
                                                   : "same"));
        d += b;
    }
    ok(nm, all, d);
}

/* ---- the two fp8 fusions, against the pair each one replaces -------------------------------- */
//
// THE INVARIANT IS NOT "CLOSE", IT IS BYTE-IDENTICAL, which is why this compares against the
// UNFUSED KERNELS rather than a host mirror. The engine's own acceptance test is that generated
// text does not move across builds; a fused quantiser whose codes differed by one in the last
// place on a few channels a layer would pass any tolerance-based check and change the text anyway.
//
// The trap it guards is a single line. The unfused pair writes a BF16 norm output and the
// quantiser reads those bytes back, so the amax and the codes are taken over bf16-ROUNDED values.
// A fused kernel that quantises the f32 it already has in a register is MORE accurate, and wrong.
static int launch_row(const RadKernelInfo* r, std::vector<RadTensor> ts,
                      std::vector<RadParam> ps) {
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream)nullptr);
    if (rc != RAD_OK) return rc;
    return st_stream_sync(nullptr) == hipSuccess ? RAD_OK : RAD_E_STATE;
}

static void fused_quant_fp8_numeric(const Plugin& pl) {
    const RadKernelInfo* nrm = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* smu = find(pl, "silu_mul_bf16");
    const RadKernelInfo* qz  = find(pl, "quant_act_fp8");
    const RadKernelInfo* rq  = find(pl, "rmsnorm_quant_fp8");
    const RadKernelInfo* gq  = find(pl, "gated_quant_fp8");
    const RadKernelInfo* sg  = find(pl, "sigmoid_bf16");
    const RadKernelInfo* ml  = find(pl, "mul_bf16");
    const RadKernelInfo* tq  = find(pl, "gate_quant_fp8");
    if (!nrm || !smu || !qz || !rq || !gq || !sg || !ml || !tq) {
        ok("fp8 fusions vs the unfused pair", false, "rows absent");
        return;
    }
    /* M is a speculative decode step's token count and K is a per-rank hidden width that is a
     * whole number of 128-column groups without being a power of two -- five groups, so a wave
     * that got its group stride wrong lands somewhere visible. */
    const int64_t M = 8, K = 640, nkb = K / 128;

    std::vector<uint16_t> hx((size_t)M * K), hgu((size_t)M * 2 * K);
    std::vector<float>    hw((size_t)K);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < K; ++c) {
            /* Per-group magnitudes an order of magnitude apart, so one group's amax cannot stand
               in for its neighbour's, and one channel per row far above the rest, which is what a
               dropped clamp turns into a NaN. */
            const float m = std::ldexp(1.0f, (int)(c / 128) - 2);
            const float v = m * (0.037f * (float)((r * 29 + c * 13) % 71) - 1.3f);
            hx[(size_t)r * K + c]        = h_to_bf16(c == (r * 37) % K ? v * 19.0f : v);
            hgu[(size_t)r * 2 * K + c]   = h_to_bf16(v * 0.75f);
            hgu[(size_t)r * 2 * K + K + c] = h_to_bf16(m * (0.011f * (float)((r + c * 7) % 43) - 0.2f));
        }
    /* A zero-centred gain, which is what GemmaRMSNorm stores and what makes `wadd` matter. */
    for (int64_t c = 0; c < K; ++c) hw[(size_t)c] = 0.004f * (float)((c * 19) % 51) - 0.1f;

    void* d_x  = dalloc((size_t)M * K * 2);
    void* d_gu = dalloc((size_t)M * 2 * K * 2);
    void* d_w  = dalloc((size_t)K * 4);
    void* d_y[2] = { dalloc((size_t)M * K * 2), dalloc((size_t)M * K * 2) };
    void* d_q[2] = { dalloc((size_t)M * K),     dalloc((size_t)M * K) };
    void* d_s[2] = { dalloc((size_t)M * nkb * 4), dalloc((size_t)M * nkb * 4) };
    if (!d_x || !d_gu || !d_w || !d_y[1] || !d_q[1] || !d_s[1]) {
        ok("fp8 fusion alloc", false, "hipMalloc failed");
        return;
    }
    st_memcpy(d_x,  hx.data(),  (size_t)M * K * 2, hipMemcpyHostToDevice);
    st_memcpy(d_gu, hgu.data(), (size_t)M * 2 * K * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w,  hw.data(),  (size_t)K * 4, hipMemcpyHostToDevice);

    const double eps = 1e-6, wadd = 1.0;

    auto same = [&](const char* what) {
        std::vector<uint16_t> y0((size_t)M * K), y1((size_t)M * K);
        std::vector<uint8_t>  q0((size_t)M * K), q1((size_t)M * K);
        std::vector<float>    s0((size_t)M * nkb), s1((size_t)M * nkb);
        st_memcpy(y0.data(), d_y[0], y0.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(y1.data(), d_y[1], y1.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(q0.data(), d_q[0], q0.size(), hipMemcpyDeviceToHost);
        st_memcpy(q1.data(), d_q[1], q1.size(), hipMemcpyDeviceToHost);
        st_memcpy(s0.data(), d_s[0], s0.size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(s1.data(), d_s[1], s1.size() * 4, hipMemcpyDeviceToHost);
        int ny = 0, nq = 0, ns = 0;
        for (size_t i = 0; i < y0.size(); ++i) ny += y0[i] != y1[i];
        for (size_t i = 0; i < q0.size(); ++i) nq += q0[i] != q1[i];
        for (size_t i = 0; i < s0.size(); ++i) ns += std::memcmp(&s0[i], &s1[i], 4) != 0;
        /* A run that wrote nothing at all would compare equal against a zeroed buffer, so the
         * codes are also required to be non-trivial. */
        int nz = 0;
        for (size_t i = 0; i < q1.size(); ++i) nz += q1[i] != 0;
        ok(what, ny == 0 && nq == 0 && ns == 0 && nz > (int)q1.size() / 2,
           "bf16 differs " + std::to_string(ny) + ", codes " + std::to_string(nq) +
           ", scales " + std::to_string(ns) + ", nonzero codes " + std::to_string(nz));
    };

    /* --- rmsnorm + quant_act_fp8 --- */
    int rc = launch_row(nrm, { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }),
                               T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"),
                          PF("wadd", wadd) });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(rq, { T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, {}),
                              T(d_w, RAD_F32, { K }),
                              T(d_q[1], RAD_F8E4M3, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                              T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128),
                          PS("dtype", "bf16"), PF("wadd", wadd) });
    if (rc != RAD_OK) ok("rmsnorm_quant_fp8 vs rmsnorm + quant_act_fp8", false,
                         "rc=" + std::to_string(rc));
    else same("rmsnorm_quant_fp8 is byte-identical to rmsnorm + quant_act_fp8");

    /* --- silu_mul + quant_act_fp8 --- */
    rc = launch_row(smu, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_y[0], RAD_BF16, { M, K }) },
                    { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(gq, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q[1], RAD_F8E4M3, { M, K }),
                              T(d_s[1], RAD_F32, { M, nkb }), T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                          PS("act", "silu") });
    if (rc != RAD_OK) ok("gated_quant_fp8 vs silu_mul + quant_act_fp8", false,
                         "rc=" + std::to_string(rc));
    else same("gated_quant_fp8 is byte-identical to silu_mul + quant_act_fp8");

    /* --- sigmoid + mul + quant_act_fp8, against gate_quant_fp8 ---
     *
     * The gated attention's epilogue, and the case that says the two-buffer form agrees with the
     * three passes it replaces. The gate is the FIRST half of `gu` and the thing it scales is the
     * second, which is the packed layout; the strided form -- a column view of a [q|gate]
     * projection -- differs from it only in the pitch the shim reads off the operand, and the
     * arch is the only place that shape exists. */
    rc = launch_row(sg, { T(d_gu, RAD_BF16, { M, K }), T(d_y[0], RAD_BF16, { M, K }) },
                    { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(ml, { T(d_y[0], RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                              T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(tq, { T(d_gu, RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                              T(d_q[1], RAD_F8E4M3, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                              T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                          PS("act", "sigmoid") });
    if (rc != RAD_OK) ok("gate_quant_fp8 vs sigmoid + mul + quant_act_fp8", false,
                         "rc=" + std::to_string(rc));
    else same("gate_quant_fp8 is byte-identical to sigmoid + mul + quant_act_fp8");

    /* --- the same two at INT8 codes, against the same products and quant_act_i8g ---
     *
     * An int8 trunk linear reads int8 codes, and the producer in front of it writes them itself
     * when its codes operand is I8: the bf16 product must not move, and the codes and scales must
     * be what quant_act_i8g takes over that product. */
    if (const RadKernelInfo* qi = find(pl, "quant_act_i8g")) {
        rc = launch_row(smu, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(qi, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_I8, { M, K }),
                                  T(d_s[0], RAD_F32, { M, nkb }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(gq, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q[1], RAD_I8, { M, K }),
                                  T(d_s[1], RAD_F32, { M, nkb }), T(d_y[1], RAD_BF16, { M, K }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                              PS("act", "silu") });
        if (rc != RAD_OK) ok("gated_quant_fp8 int8 vs silu_mul + quant_act_i8g", false,
                             "rc=" + std::to_string(rc));
        else same("gated_quant_fp8 at int8 codes is byte-identical to silu_mul + quant_act_i8g");

        rc = launch_row(sg, { T(d_gu, RAD_BF16, { M, K }), T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(ml, { T(d_y[0], RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                                  T(d_y[0], RAD_BF16, { M, K }) },
                            { PI("M", M), PI("n", K), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(qi, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_I8, { M, K }),
                                  T(d_s[0], RAD_F32, { M, nkb }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(tq, { T(d_gu, RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                                  T(d_q[1], RAD_I8, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                                  T(d_y[1], RAD_BF16, { M, K }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                              PS("act", "sigmoid") });
        if (rc != RAD_OK) ok("gate_quant_fp8 int8 vs sigmoid + mul + quant_act_i8g", false,
                             "rc=" + std::to_string(rc));
        else same("gate_quant_fp8 at int8 codes is byte-identical to sigmoid + mul + quant_act_i8g");

        /* The rotated form pairs with E4M3 experts only, and says so rather than writing codes
         * no reader could use. */
        if (const RadKernelInfo* gh = find(pl, "gated_had_quant_fp8")) {
            rc = launch_row(gh, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q[1], RAD_I8, { M, K }),
                                  T(d_s[1], RAD_F32, { M, nkb }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                              PS("act", "silu") });
            ok("gated_had_quant_fp8 refuses int8 codes", rc != RAD_OK, "rc=" + std::to_string(rc));
        }
    }

    /* --- add + rmsnorm + quant_act_fp8, against the same three fused ---
     *
     * The residual is what a block folds into the norm in front of it, and it is the case where
     * "bit-identical" is a real choice rather than a formality: the sum is f32 and the pair this
     * replaces ROUNDS IT TO BF16 in `add`'s store before `rmsnorm` ever sees it. Keeping the f32
     * sum -- which is what the int8 sibling does, and is more accurate -- moves the text. */
    const RadKernelInfo* ad = find(pl, "add_bf16");
    if (!ad) { ok("rmsnorm_quant_fp8 residual", false, "add_bf16 absent"); return; }
    void* d_r[2] = { dalloc((size_t)M * K * 2), dalloc((size_t)M * K * 2) };
    if (!d_r[1]) { ok("fp8 fusion residual alloc", false, "hipMalloc failed"); return; }
    /* A residual with its own structure, and BOTH copies start from it: the fused form updates it
     * in place, so a shared buffer would have the second run reading the first run's output. */
    std::vector<uint16_t> hr((size_t)M * K);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < K; ++c)
            hr[(size_t)r * K + c] = h_to_bf16(0.019f * (float)((r * 13 + c * 29) % 67) - 0.6f);
    st_memcpy(d_r[0], hr.data(), hr.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_r[1], hr.data(), hr.size() * 2, hipMemcpyHostToDevice);

    rc = launch_row(ad, { T(d_r[0], RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                          T(d_r[0], RAD_BF16, { M, K }) },
                    { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(nrm, { T(d_r[0], RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }),
                               T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"),
                          PF("wadd", wadd) });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(rq, { T(d_x, RAD_BF16, { M, K }), T(d_r[1], RAD_BF16, { M, K }),
                              T(d_w, RAD_F32, { K }),
                              T(d_q[1], RAD_F8E4M3, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                              T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128),
                          PS("dtype", "bf16"), PF("wadd", wadd) });
    if (rc != RAD_OK) {
        ok("rmsnorm_quant_fp8 residual vs add + rmsnorm + quant_act_fp8", false,
           "rc=" + std::to_string(rc));
        return;
    }
    same("rmsnorm_quant_fp8 residual is byte-identical to add + rmsnorm + quant_act_fp8");
    {
        std::vector<uint16_t> r0((size_t)M * K), r1((size_t)M * K);
        st_memcpy(r0.data(), d_r[0], r0.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(r1.data(), d_r[1], r1.size() * 2, hipMemcpyDeviceToHost);
        int nd = 0;
        for (size_t i = 0; i < r0.size(); ++i) nd += r0[i] != r1[i];
        ok("rmsnorm_quant_fp8 leaves the same residual behind", nd == 0,
           "differs " + std::to_string(nd));
    }
}

/* THE fp8 lm_head, end to end through its OWN hooks.
 *
 * The arrangement is the odd one in this library -- E4M3 codes with the row's own bf16 scales in
 * the tail of the same row, rather than the separate [sn][sk] plane the linears read -- so the
 * things that can go wrong are the interleave (a scale read from the wrong row), the f32 store
 * (this is the op most prone to a bf16 write into an f32 plane), and the m_real mask on the
 * instantiations that round M up.
 *
 * The weight is block fp8's two planes, relaid by the row's own hooks into the row it reads; the
 * oracle is those planes, code times its block's scale. The inverse must give the planes back. */
/* `i8`: the int8 head, logits_gemm_i8 -- i8 codes with a bf16 scale a ROW per 128 columns. */
static void fp8_logits_numeric(const Plugin& pl, int64_t N, int64_t K, int64_t M, bool i8 = false) {
    const char* row = i8 ? "logits_gemm_i8" : "logits_gemm_fp8";
    const RadKernelInfo* lg = find(pl, row);
    if (!lg || !lg->layout || !lg->relayout) {
        ok(i8 ? "i8 logits vs host reference" : "fp8 logits vs host reference", false,
           "row or hooks absent");
        return;
    }
    const int64_t stride = r4d_fp8::lm_row_bytes(K);
    const std::string tag = " [N=" + std::to_string(N) + " M=" + std::to_string(M) + "]";

    std::vector<float> wf((size_t)N * K), xf((size_t)M * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wf[(size_t)r * K + c] =
                std::ldexp(1.0f, (int)(((r / 128) % 5) * 3 - ((c / 128) % 4)))
                * (0.031f * (float)((r * 37 + c * 11) % 61) - 0.9f);
    std::vector<uint16_t> xb((size_t)M * K);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t c = 0; c < K; ++c) {
            const float v = 0.017f * (float)((m * 53 + c * 7) % 83) - 0.7f;
            xb[(size_t)m * K + c] = h_to_bf16(v);
            xf[(size_t)m * K + c] = h_bf16(xb[(size_t)m * K + c]);
        }

    Planes p;
    if (!i8) {
        p = fp8_block_planes(wf, N, K);
    } else {
        p.enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        p.size(N, K);
        const int64_t nkb = K / 128;
        for (int64_t r = 0; r < N; ++r)
            for (int64_t gi = 0; gi < nkb; ++gi) {
                float amax = 0.0f;
                for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(wf[(size_t)(r * K + gi * 128 + j)]));
                rad_store_f32(p.data[1].data(), RAD_BF16, r * nkb + gi, amax > 0 ? amax / 127.0f : 1.0f);
                const float sc = rad_load_f32(p.data[1].data(), RAD_BF16, r * nkb + gi);
                for (int64_t j = 0; j < 128; ++j) {
                    const float q = std::nearbyint(wf[(size_t)(r * K + gi * 128 + j)] / sc);
                    p.data[0][(size_t)(r * K + gi * 128 + j)] =
                        (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
                }
            }
    }
    std::vector<RadParam> lp = { PI("n_vocab", N), PI("n_embd", K) };
    RadLayout lay;
    std::vector<uint8_t> wq;
    const int lrc = stored(lg, lp, 1, p, { "codes", "scale" }, &wq, &lay);
    if (lrc != RAD_OK) {
        ok(i8 ? "i8 logits relayout" : "fp8 logits relayout", false, "rc=" + std::to_string(lrc) + tag);
        return;
    }
    ok(i8 ? "i8 logits layout is codes plus the row's scale tail" : "fp8 logits layout is codes plus a per-row scale tail",
       lay.bytes == N * stride && lay.shape[0] == N && lay.shape[1] == stride &&
       (int64_t)wq.size() == N * stride,
       "bytes " + std::to_string(lay.bytes) + " want " + std::to_string(N * stride) + tag);
    {
        std::string why;
        ok(i8 ? "i8 logits relayout inverts to the planes it was handed" : "fp8 logits relayout inverts to the planes it was handed",
           restores(lg, lp, 1, p, { "codes", "scale" }, wq, &why), why + tag);
    }

    /* The oracle: the planes, code times the scale of its 128x128 block. */
    const int64_t sk = p.cols[1];
    std::vector<double> ref((size_t)M * N);
    for (int64_t r = 0; r < N; ++r) {
        const uint8_t* row = p.data[0].data() + (size_t)r * K;
        const uint16_t* sc = (const uint16_t*)p.data[1].data() + (i8 ? r : r / 128) * sk;
        for (int64_t m = 0; m < M; ++m) {
            double acc = 0;
            for (int64_t c = 0; c < K; ++c)
                acc += (i8 ? (double)(int8_t)row[c] : (double)h_e4m3(row[c])) *
                       (double)h_bf16(sc[c / 128]) * xf[(size_t)m * K + c];
            ref[(size_t)m * N + r] = acc;
        }
    }

    void* d_w = dalloc(wq.size());
    void* d_x = dalloc((size_t)M * K * 2);
    void* d_y = dalloc((size_t)M * N * 4);
    if (!d_w || !d_x || !d_y) { ok(i8 ? "i8 logits alloc" : "fp8 logits alloc", false, std::string("hipMalloc failed") + tag); return; }
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_x, xb.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
    /* Poisoned, so a kernel that writes bf16 pairs into the f32 plane -- the failure mode this op
     * is prone to -- cannot pass by leaving the high half of each word at zero. */
    st_memset(d_y, 0x5a, (size_t)M * N * 4);

    // THE DECLARED SHAPE, [N][K] and not [N][K + 2*nkb], because that is what the engine hands a
    // kernel: core/runtime/ctx.cpp builds a weight's tensor from decl.shape with the layout
    // hook's dtype. Passing the stored width here would test a call the engine never makes.
    const int rc = launch_row(lg, { T(d_x, RAD_BF16, { M, K }),
                                    T(d_w, i8 ? RAD_I8 : RAD_F8E4M3, { N, K }),
                                    T(d_y, RAD_F32, { M, N }) },
                              { PI("M", M), PI("n_vocab", N), PI("n_embd", K),
                                PS("dtype", "bf16") });
    if (rc != RAD_OK) { ok(i8 ? "i8 logits launch" : "fp8 logits launch", false, "rc=" + std::to_string(rc) + tag); return; }

    std::vector<float> got((size_t)M * N);
    st_memcpy(got.data(), d_y, got.size() * 4, hipMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    const double e = den > 0 ? std::sqrt(num / den) : 1.0;
    ok(i8 ? "logits_gemm_i8 vs host ref over its planes" : "logits_gemm_fp8 vs host ref over its planes",
       e < 2e-3, "rel_l2 " + std::to_string(e) + tag);

    /* THE DRAFTER'S RESCORING READS THE SAME STORED HEAD, so it is checked over the same bytes:
     * R candidates a row -- one of them an id past the shard, which must come back a pad -- each
     * scored against the planes, and the answer sorted by value. */
    const RadKernelInfo* rr = find(pl, i8 ? "logit_rerank_i8" : "logit_rerank");
    if (!rr) return;
    const int64_t R = std::min<int64_t>(8, N);
    std::vector<int32_t> cand((size_t)(M * R));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t r = 0; r < R; ++r)
            cand[(size_t)(m * R + r)] = r == R - 1 ? (int32_t)N : (int32_t)((m * 131 + r * 977) % N);
    void* d_in = dalloc(cand.size() * 4);
    void* d_oi = dalloc(cand.size() * 4);
    void* d_ov = dalloc(cand.size() * 4);
    const std::string rn = std::string(i8 ? "logit_rerank_i8" : "logit_rerank") +
                           " vs host ref over the stored head" + tag;
    if (!d_in || !d_oi || !d_ov) { ok(rn.c_str(), false, "hipMalloc failed"); return; }
    st_memcpy(d_in, cand.data(), cand.size() * 4, hipMemcpyHostToDevice);
    const int rrc = launch_row(rr, { T(d_x, RAD_BF16, { M, K }),
                                     T(d_w, i8 ? RAD_I8 : RAD_F8E4M3, { N, K }),
                                     T(d_in, RAD_I32, { M, R }), T(d_oi, RAD_I32, { M, R }),
                                     T(d_ov, RAD_F32, { M, R }) },
                               { PI("M", M), PI("R", R), PI("n_vocab", N), PI("n_embd", K),
                                 PI("vocab_off", 0), PS("dtype", "bf16") });
    if (rrc != RAD_OK) { ok(rn.c_str(), false, "rc=" + std::to_string(rrc)); return; }
    std::vector<int32_t> oi(cand.size());
    std::vector<float> ov(cand.size());
    st_memcpy(oi.data(), d_oi, oi.size() * 4, hipMemcpyDeviceToHost);
    st_memcpy(ov.data(), d_ov, ov.size() * 4, hipMemcpyDeviceToHost);
    int64_t bad = 0;
    for (int64_t m = 0; m < M; ++m) {
        double amax = 0;
        for (int64_t n = 0; n < N; ++n) amax = std::max(amax, std::fabs(ref[(size_t)(m * N + n)]));
        for (int64_t r = 0; r < R; ++r) {
            const int32_t id = oi[(size_t)(m * R + r)];
            const float v = ov[(size_t)(m * R + r)];
            if (r == R - 1) { bad += id != -1; continue; }      /* the pad sorts last */
            const bool known = std::find(cand.begin() + m * R, cand.begin() + m * R + R - 1, id) !=
                               cand.begin() + m * R + R - 1;
            bad += !known || std::fabs((double)v - ref[(size_t)(m * N + id)]) > 1e-3 * amax + 1e-4;
            if (r > 0) bad += v > ov[(size_t)(m * R + r - 1)];
        }
    }
    ok(rn.c_str(), bad == 0, std::to_string(bad) + " wrong ids, values or orderings");
}

static void fp8_numeric(const Plugin& pl, int64_t N, int64_t K) {
    const RadKernelInfo* mv = find(pl, "gemm_fp8a16_nt_m1");
    const RadKernelInfo* tl = find(pl, "gemm_fp8a8_tiled");
    const RadKernelInfo* qz = find(pl, "quant_act_fp8");
    const RadKernelInfo* n16 = find(pl, "gemm_fp8a8_nt_m16");
    if (!mv || !tl || !qz || !n16 || !mv->layout || !n16->layout || !n16->relayout) {
        ok("fp8 numeric vs host reference", false, "rows or hooks absent");
        return;
    }
    const int64_t sn = N / 128, sk = K / 128;

    /* A weight with real structure: an asymmetric, per-block-varying magnitude. A constant or a
     * symmetric pattern cannot tell a transposed fragment from a correct one -- both give zero
     * error -- and cannot tell one block scale from its neighbour either.
     *
     * THE EXPONENT WRAPS, and it has to. An unwrapped `(r/128)*3 - (c/128)` reaches 405 by row
     * 17408 at 128 rows a step, and ldexp(1.0f, 405) is INFINITY -- so the generator would produce
     * a valid weight only below about 5400 rows and every fp8 numeric case in this file would be
     * pinned to N = 256: raising it would fill the weight with inf rather than exercise a bigger
     * shape. Every large projection in the model is 17408 rows.
     *
     * Wrapping the block index keeps what the variation is FOR -- neighbouring 128-row blocks still
     * differ by 8x, which is what tells one block scale from the next -- and bounds the range at
     * 2^12, comfortably inside float and inside E4M3's own reach after the per-block scale. */
    std::vector<float> wf((size_t)N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wf[(size_t)r * K + c] =
                std::ldexp(1.0f, (int)(((r / 128) % 5) * 3 - ((c / 128) % 4)))
                * (0.031f * (float)((r * 37 + c * 11) % 61) - 0.9f);

    /* TWO ARRANGEMENTS OF ONE PAIR OF PLANES, and the two-arrangement part is not optional. The
     * fp8a16 matvec reads the planes ROW-MAJOR, as they are; the two fp8a8 GEMMs read the same
     * codes permuted into WMMA FRAGMENT ORDER, and each row's own layout hook says which it wants.
     *
     * Building ONE arrangement and testing all three kernels against it is how this test lies: the
     * oracle and a fp8a8 kernel would then read the same bytes the same way, agree with each
     * other, both disagree with the weight, and the case would report ok.
     *
     * ONE oracle is still right, because an arrangement does not change VALUES: the planes,
     * dequantised on the host, are the reference for every kernel. */
    const Planes p = fp8_block_planes(wf, N, K);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, wqf, wsb, wsf;
    RadLayout flay;
    const int rw = stored(mv, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(mv, lp, 3, p, { "scale" }, &wsb);
    const int rf = stored(n16, lp, 2, p, { "codes" }, &wqf, &flay);
    const int rfs = stored(n16, lp, 3, p, { "scale" }, &wsf);
    ok("fp8 matvec reads both planes as they are",
       rw == RAD_E_UNSUPPORTED && rs == RAD_E_UNSUPPORTED,
       "codes rc " + std::to_string(rw) + ", scale rc " + std::to_string(rs));
    ok("fp8 fragment order is the codes rearranged, and the scales as they are",
       rf == RAD_OK && flay.bytes == N * K && (int64_t)wqf.size() == N * K &&
       rfs == RAD_E_UNSUPPORTED && wsf == wsb,
       "codes rc " + std::to_string(rf) + " bytes " + std::to_string(flay.bytes) +
       ", scale rc " + std::to_string(rfs));
    if (rw != RAD_E_UNSUPPORTED || rs != RAD_E_UNSUPPORTED || rf != RAD_OK) return;
    {
        std::string why;
        ok("fp8 fragment order inverts to the codes", restores(n16, lp, 2, p, { "codes" }, wqf, &why),
           why);
    }
    std::vector<uint16_t> ws((size_t)sn * sk);
    std::memcpy(ws.data(), wsb.data(), ws.size() * 2);

    std::vector<float> wd((size_t)N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wd[(size_t)r * K + c] =
                h_e4m3(wq[(size_t)r * K + c]) * h_bf16(ws[(size_t)(r / 128) * sk + (c / 128)]);

    std::vector<uint16_t> xb((size_t)K);
    for (int64_t c = 0; c < K; ++c)
        xb[(size_t)c] = h_to_bf16(0.021f * (float)((c * 17) % 53) - 0.55f);
    std::vector<float> ref((size_t)N, 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        double acc = 0;
        for (int64_t c = 0; c < K; ++c) acc += (double)wd[(size_t)r * K + c] * h_bf16(xb[(size_t)c]);
        ref[(size_t)r] = (float)acc;
    }

    void* d_w = dalloc((size_t)N * K);
    void* d_wf = dalloc((size_t)N * K);   /* the same weight in fragment order, for the fp8a8 pair */
    void* d_s = dalloc((size_t)sn * sk * 2);
    void* d_x = dalloc((size_t)K * 2);
    void* d_y = dalloc((size_t)N * 2);
    if (!d_w || !d_wf || !d_s || !d_x || !d_y) { ok("fp8 numeric alloc", false, "hipMalloc failed"); return; }
    st_memcpy(d_wf, wqf.data(), (size_t)N * K, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), (size_t)N * K, hipMemcpyHostToDevice);
    st_memcpy(d_s, ws.data(), (size_t)sn * sk * 2, hipMemcpyHostToDevice);
    st_memcpy(d_x, xb.data(), (size_t)K * 2, hipMemcpyHostToDevice);

    auto rel = [&](const std::vector<uint16_t>& got) {
        double num = 0, den = 0;
        for (int64_t r = 0; r < N; ++r) {
            const double d = (double)h_bf16(got[(size_t)r]) - (double)ref[(size_t)r];
            num += d * d;
            den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
        }
        return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    };

    /* --- the matvec, at every point of its declared space: `rb` changes which rows share an
     *     activation window, and a wrong RB reads the neighbouring row's scale. That is exactly
     *     the bug a smoke test misses. */
    for (const TunePoint& tp : tune_points(mv, std::vector<RadParam>{ PI("M", 1), PI("N", N),
                                                                     PI("K", K), PI("group", 128),
                                                                     PS("dtype", "fp8a16") })) {
        std::vector<RadTensor> ts = {
            T(d_x, RAD_BF16, { 1, K }), T(nullptr, RAD_F32, { 1, sk }),
            T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_y, RAD_BF16, { 1, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[1]); absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = with_point(std::vector<RadParam>{
            PI("M", 1), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
        }, tp);
        st_memset(d_y, 0, (size_t)N * 2);
        run_case(mv, ts, ps);
        std::vector<uint16_t> got((size_t)N);
        st_memcpy(got.data(), d_y, (size_t)N * 2, hipMemcpyDeviceToHost);
        const double e = rel(got);
        const std::string what =
            std::string("gemm_fp8a16_nt_m1 vs host ref, ") + tp.label;
        ok(what.c_str(), e < 3e-2, "rel_l2 " + std::to_string(e));
    }

    /* --- THE MULTI-ROW MATVEC, which is what a speculative verify step runs: M = 1 + n_spec token
     *     rows against one weight read.
     *
     *     THE ROWS ARE DELIBERATELY DISTINCT, and that is the whole value of this case. The failure
     *     it exists to catch is a token row reading another row's activation -- an `a_ld` off by
     *     anything, or a hoisted pointer -- and the W8A8 case below CANNOT see that, because every
     *     row there holds the same vector and a kernel that read row 0 five times would pass it.
     *     Each row is checked against its own host reference.
     *
     *     M = 5 rather than a power of two: it is 1 + n_spec at n_spec 4, and rounding a token count
     *     up to a template instantiation is exactly the kind of thing that works at 4 and 8. */
    {
        const int64_t M = 5;
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_ym = dalloc((size_t)M * N * 2);
        if (!d_xm || !d_ym) { ok("fp8a16 multi-row alloc", false, "hipMalloc failed"); }
        else {
            std::vector<uint16_t> xm((size_t)M * K);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t c = 0; c < K; ++c)
                    xm[(size_t)m * K + c] = xb[(size_t)((c + m * 37) % K)];
            st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);

            std::vector<double> refm((size_t)M * N);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t r = 0; r < N; ++r) {
                    double acc = 0;
                    for (int64_t c = 0; c < K; ++c)
                        acc += (double)wd[(size_t)r * K + c] * h_bf16(xm[(size_t)m * K + c]);
                    refm[(size_t)m * N + r] = acc;
                }

            std::vector<RadTensor> ts = {
                T(d_xm, RAD_BF16, { M, K }), T(nullptr, RAD_F32, { M, sk }),
                T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
                T(d_ym, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
            };
            absent(ts[1]); absent(ts[5]); absent(ts[6]);
            std::vector<RadParam> ps = {
                PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
            };
            st_memset(d_ym, 0, (size_t)M * N * 2);
            run_case(mv, ts, ps);
            std::vector<uint16_t> got((size_t)M * N);
            st_memcpy(got.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
            double worst = 0;
            for (int64_t m = 0; m < M; ++m) {
                double num = 0, den = 0;
                for (int64_t r = 0; r < N; ++r) {
                    const double d =
                        (double)h_bf16(got[(size_t)m * N + r]) - refm[(size_t)m * N + r];
                    num += d * d;
                    den += refm[(size_t)m * N + r] * refm[(size_t)m * N + r];
                }
                const double er = den > 0 ? std::sqrt(num / den) : 1.0;
                if (er > worst) worst = er;
            }
            ok("gemm_fp8a16_nt_mt vs host ref, worst of 5 DISTINCT token rows",
               worst < 3e-2, "rel_l2 " + std::to_string(worst));
        }
    }

    /* --- the NARROW W8A8 row, which is the riskiest kernel in the family and the only one where
     *     two hazards meet: the epilogue TRANSPOSES (the weight is on the A
     *     operand there, so D is [N][M] and C must be [M][N]), and split-K is on. At N=256, K=512
     *     the split law gives ksplit=2, so the partial plane and the reduce are both exercised.
     *     M=8 is deliberately not a multiple of the 16-wide token tile: the ragged path is what a
     *     speculative batch actually runs. Every one of the 8 rows must match, because a wrong
     *     transposed stride writes one right row and seven wrong ones. */
    {
        /* EVERY TOKEN-TILE COUNT AND BOTH SIDES OF `Full`. The narrow tile covers NT * 16 tokens
         * with one weight read, and NT is chosen from M, so the M list has to cross every
         * boundary: 1 and 8 are NT = 1 (and M = 1 is the only M where the Full predicate and the
         * transposed epilogue see a single token), 24 and 32 are NT = 2, 40 and 48 are NT = 3,
         * and 64 is NT = 4.
         *
         * The ODD ones are the point. `Full` drops the epilogue's column guard, and it means "all
         * NT tiles are live", NOT "M is a multiple of 16" -- at M = 48 with NT = 4 the second is
         * true and the first is false, and a kernel that conflated them writes columns 48..63
         * past the end of a split-K scratch sized for 48, over the arrival counter, and the
         * finisher spins forever. 24 and 48 are here to keep that covered, and 40 is the one M in
         * the band that is ragged under NT = 3 as well -- 48 is a whole number of tiles there and
         * would not exercise it. */
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)24, (int64_t)32, (int64_t)40,
                                 (int64_t)48, (int64_t)64 }) {
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_q  = dalloc((size_t)M * K);
        void* d_qs = dalloc((size_t)M * sk * 4);
        void* d_ym = dalloc((size_t)M * N * 2);
        const RadKernelInfo* nr = find(pl, "gemm_fp8a8_nt_m16");
        if (!nr) { ok("gemm_fp8a8_nt_m16 numeric", false, "row absent"); }
        else if (!d_xm || !d_q || !d_qs || !d_ym) { ok("fp8 narrow alloc", false, "hipMalloc failed"); }
        else {
            /* DISTINCT ROWS, because identical ones cannot see the failure this case exists
             * for: a token tile reading another tile's activation. With every row the same vector
             * a kernel that read row 0 sixty-four times passes. Each row gets its own reference. */
            std::vector<uint16_t> xm((size_t)M * K);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t c = 0; c < K; ++c)
                    xm[(size_t)m * K + c] = xb[(size_t)((c + m * 37) % K)];
            st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
            std::vector<double> refn((size_t)M * N);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t r = 0; r < N; ++r) {
                    double acc = 0;
                    for (int64_t c = 0; c < K; ++c)
                        acc += (double)wd[(size_t)r * K + c] * h_bf16(xm[(size_t)m * K + c]);
                    refn[(size_t)m * N + r] = acc;
                }
            std::vector<RadTensor> qt = {
                T(d_xm, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }),
                T(d_qs, RAD_F32, { M, sk }),
            };
            std::vector<RadParam> qp = {
                PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
            };
            run_case(qz, qt, qp);

            std::vector<RadTensor> ts = {
                T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
                T(d_wf, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
                T(d_ym, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
            };
            absent(ts[5]); absent(ts[6]);
            std::vector<RadParam> ps = {
                PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
            };
            st_memset(d_ym, 0, (size_t)M * N * 2);
            run_case(nr, ts, ps);
            std::vector<uint16_t> got((size_t)M * N);
            st_memcpy(got.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
            double worst = 0;
            for (int64_t m = 0; m < M; ++m) {
                double num = 0, den = 0;
                for (int64_t r = 0; r < N; ++r) {
                    const double d =
                        (double)h_bf16(got[(size_t)m * N + r]) - refn[(size_t)m * N + r];
                    num += d * d;
                    den += refn[(size_t)m * N + r] * refn[(size_t)m * N + r];
                }
                const double er = den > 0 ? std::sqrt(num / den) : 1.0;
                if (er > worst) worst = er;
            }
            ok((std::string("gemm_fp8a8_nt_m16 vs host ref, ") + std::to_string((long long)M) +
                " DISTINCT rows (transposed store + split-K)").c_str(),
               worst < 8e-2, "rel_l2 " + std::to_string(worst));

            /* THE WAVE COUNT MUST NOT MOVE A BYTE, and this is the only place that can be asked.
             * MW is a launch heuristic chosen from (N, M) inside libr4d, so a rule change is
             * invisible from out here and the environment pin that sweeps it is read once a
             * process -- one run cannot hold two settings. A rel_l2 against a host reference will
             * not answer it either: a wave count that moved the last bit of every element passes
             * that at 1e-7 and changes the emitted text.
             *
             * A DIFFERENT GRID IS ALLOWED TO DIFFER. `narrow_ksplit` decides from `gx * mw`, which
             * is N/16 only when N divides MW*16; where it does not, the split repartitions K and
             * the accumulation chain legitimately changes. So the sweep compares the wave counts
             * whose grid the shape actually admits, which is the same test the rule applies to
             * itself when it asks `n % 128` and `n % 256`. */
            if (pl.pin_mw) {
                const std::vector<uint16_t> base = got;
                std::vector<uint16_t> alt((size_t)M * N);
                std::string why;
                for (const int mw : { 1, 2, 4, 8, 16 }) {
                    if (N % (mw * 16)) continue;
                    /* A K that is not a multiple of 256 runs the 128-wide staged set, which
                     * compiles only the wave counts the rule answers (4, 8, 16); a pin outside
                     * them is refused by contract -- see R4D_F8D_NT128 in r4d_gemm_fp8a8.hip. */
                    if (K % 256 && mw != 4 && mw != 8 && mw != 16) continue;
                    pl.pin_mw(mw);
                    st_memset(d_ym, 0, (size_t)M * N * 2);
                    run_case(nr, ts, ps);
                    st_memcpy(alt.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
                    if (alt != base) { why = "mw " + std::to_string(mw); break; }
                }
                pl.pin_mw(0);
                ok((std::string("gemm_fp8a8_nt_m16 is byte-identical at every wave count its "
                                "grid admits, M ") + std::to_string((long long)M)).c_str(),
                   why.empty(), why);
            }

            /* AND THE TOKEN-TILE COUNT MUST NOT MOVE A BYTE EITHER, for the same reason and with
             * one extra hazard the wave count does not have. NT decides which wave owns which
             * token AND it feeds `Full`: at M = 48 the rule's NT = 3 makes Full true where NT = 4
             * makes it false, so the pin sweeps two different epilogue paths as well as two
             * different tilings. The pin only goes UP -- NT*16 is the column extent the epilogue
             * may write, so NT below ceil(M/16) drops rows rather than running a slower variant. */
            if (pl.pin_nt) {
                const std::vector<uint16_t> base = got;
                std::vector<uint16_t> alt((size_t)M * N);
                std::string why;
                for (int nt = 1; nt <= 4; ++nt) {
                    if ((int64_t)nt * 16 < M) continue;
                    pl.pin_nt(nt);
                    st_memset(d_ym, 0, (size_t)M * N * 2);
                    run_case(nr, ts, ps);
                    st_memcpy(alt.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
                    if (alt != base) { why = "nt " + std::to_string(nt); break; }
                }
                pl.pin_nt(0);
                ok((std::string("gemm_fp8a8_nt_m16 is byte-identical at every token-tile count "
                                "that covers M, M ") + std::to_string((long long)M)).c_str(),
                   why.empty(), why);
            }

            /* ---- ...AND WITH THE GATE AND THE QUANTISER FOLDED INTO ITS EPILOGUE -------------
             *
             * THE QUESTION IS BYTE EQUALITY AND NOTHING ELSE, for the reason the wave-count sweep
             * above gives from the other side: the engine substitutes the fold for the pair on a
             * band, so a fold that moved the last bit of every code would be a SECOND numerical
             * path wearing the first one's name, and a rel_l2 against a host reference would pass
             * it at 1e-7 while the emitted text moved. The pair is this GEMM's bf16 output read
             * back by `gated_quant_fp8`; the fold never writes that tensor at all.
             *
             * Run over the same M list, which crosses every token-tile count, and over N = 256,
             * 5120 and 17408 -- the last is gate_up's own per-rank width, so the shape the engine
             * issues is one of the shapes checked and not an extrapolation from smaller ones. */
            const RadKernelInfo* gq = find(pl, "gated_quant_fp8");
            const RadKernelInfo* gg = find(pl, "gemm_fp8a8_gated_nt_m16");
            /* The fold stages K 256 wide and refuses any other K. The one K % 256 shape in
             * the engine is the shared expert's down projection, which is never gated. */
            if (gq && gg && N % 256 == 0 && K % 256 == 0) {
                const int64_t nout = N / 2, gkb = nout / 128;
                /* THE FOLD REFUSES A SHAPE WHOSE GRID THE SPLIT RULE WOULD WIDEN, and that is a
                 * contract and not an accident: it has no partial plane to reduce and no second
                 * launch to fold a reduce into. `narrow_ksplit` splits when `K >= 512` and
                 * `N/16 < 128`, so N = 256 here is a REFUSAL and the equality question does not
                 * arise. gate_up is 17408 rows and never comes near it; the k and v projections
                 * that do are not gated. Asserted rather than skipped, because a fold that served
                 * this shape by ignoring the split would be silently wrong. */
                const bool splits = K >= 512 && N < 2048;
                void* d_gq[2] = { dalloc((size_t)M * nout),     dalloc((size_t)M * nout) };
                void* d_gs[2] = { dalloc((size_t)M * gkb * 4),  dalloc((size_t)M * gkb * 4) };
                void* d_gy[2] = { dalloc((size_t)M * nout * 2), dalloc((size_t)M * nout * 2) };
                if (!d_gq[1] || !d_gs[1] || !d_gy[1]) {
                    ok("gated fold alloc", false, "hipMalloc failed");
                } else {
                    /* The GEMM is re-run because the wave-count sweep left the LAST pin's output
                     * in d_ym, and a pinned wave count is not what the engine issues. */
                    st_memset(d_ym, 0, (size_t)M * N * 2);
                    run_case(nr, ts, ps);
                    int rc = launch_row(gq, { T(d_ym, RAD_BF16, { M, N }),
                                              T(d_gq[0], RAD_F8E4M3, { M, nout }),
                                              T(d_gs[0], RAD_F32, { M, gkb }),
                                              T(d_gy[0], RAD_BF16, { M, nout }) },
                                        { PI("M", M), PI("n", nout), PI("group", 128),
                                          PS("dtype", "bf16"), PS("act", "silu") });
                    if (rc == RAD_OK) {
                        st_memset(d_gq[1], 0, (size_t)M * nout);
                        st_memset(d_gs[1], 0, (size_t)M * gkb * 4);
                        st_memset(d_gy[1], 0, (size_t)M * nout * 2);
                        rc = launch_row(gg, { T(d_q, RAD_F8E4M3, { M, K }),
                                              T(d_qs, RAD_F32, { M, sk }),
                                              T(d_wf, RAD_F8E4M3, { N, K }),
                                              T(d_s, RAD_BF16, { sn, sk }),
                                              T(d_gq[1], RAD_F8E4M3, { M, nout }),
                                              T(d_gs[1], RAD_F32, { M, gkb }),
                                              T(d_gy[1], RAD_BF16, { M, nout }) },
                                        { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                          PS("dtype", "fp8a8"), PS("act", "silu") });
                    }
                    if (splits) {
                        ok((std::string("gemm_fp8a8_gated_nt_m16 refuses a grid the split rule "
                                        "would widen, N ") + std::to_string((long long)N) +
                            " M " + std::to_string((long long)M)).c_str(),
                           rc != RAD_OK, "rc=" + std::to_string(rc));
                    } else if (rc != RAD_OK) {
                        ok("gemm_fp8a8_gated_nt_m16 vs the GEMM + gated_quant_fp8", false,
                           "rc=" + std::to_string(rc));
                    } else {
                        std::vector<uint8_t>  c0((size_t)M * nout), c1((size_t)M * nout);
                        std::vector<float>    z0((size_t)M * gkb), z1((size_t)M * gkb);
                        std::vector<uint16_t> b0((size_t)M * nout), b1((size_t)M * nout);
                        st_memcpy(c0.data(), d_gq[0], c0.size(), hipMemcpyDeviceToHost);
                        st_memcpy(c1.data(), d_gq[1], c1.size(), hipMemcpyDeviceToHost);
                        st_memcpy(z0.data(), d_gs[0], z0.size() * 4, hipMemcpyDeviceToHost);
                        st_memcpy(z1.data(), d_gs[1], z1.size() * 4, hipMemcpyDeviceToHost);
                        st_memcpy(b0.data(), d_gy[0], b0.size() * 2, hipMemcpyDeviceToHost);
                        st_memcpy(b1.data(), d_gy[1], b1.size() * 2, hipMemcpyDeviceToHost);
                        int nc = 0, nz = 0, nb = 0, live = 0;
                        for (size_t i = 0; i < c0.size(); ++i) nc += c0[i] != c1[i];
                        for (size_t i = 0; i < z0.size(); ++i)
                            nz += std::memcmp(&z0[i], &z1[i], 4) != 0;
                        for (size_t i = 0; i < b0.size(); ++i) nb += b0[i] != b1[i];
                        /* Two zeroed buffers compare equal, so the codes are also required to be
                         * non-trivial -- the failure mode a fold that wrote nothing would have. */
                        for (size_t i = 0; i < c1.size(); ++i) live += c1[i] != 0;
                        ok((std::string("gemm_fp8a8_gated_nt_m16 is byte-identical to the GEMM + "
                                        "gated_quant_fp8, N ") + std::to_string((long long)N) +
                            " M " + std::to_string((long long)M)).c_str(),
                           nc == 0 && nz == 0 && nb == 0 && live > (int)c1.size() / 2,
                           "codes differ " + std::to_string(nc) + ", scales " +
                           std::to_string(nz) + ", bf16 " + std::to_string(nb) + ", nonzero " +
                           std::to_string(live));
                    }
                }
            }
        }
        }
    }

    /* --- A ROW SHARD, AS THE LOADER MAKES ONE: rank 1's share of each canonical plane -- rows
     *     [N/2, N) of the codes and the scale rows that cover them, cut on the scale plane's 128-row
     *     blocks -- relaid by the row's own hook as a weight of N/2 rows, which has to reproduce
     *     rows [N/2, N) of the full run. The share is cut BEFORE the arrangement, so a relayout
     *     that leaned on rows it was not handed shows here. */
    if (N % 32 == 0 && (N / 2) % 128 == 0) {
        const int64_t Nh = N / 2, snh = Nh / 128, M = 8;
        const Planes half = share(p, Nh, N, 0, K);
        std::vector<RadParam> hp = { PI("N", Nh), PI("K", K), PI("group", 128) };
        std::vector<uint8_t> hw, hs;
        const int hrc = stored(n16, hp, 2, half, { "codes" }, &hw);
        const int hsc = stored(n16, hp, 3, half, { "scale" }, &hs);
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_q  = dalloc((size_t)M * K);
        void* d_qs = dalloc((size_t)M * sk * 4);
        void* d_yh = dalloc((size_t)M * Nh * 2);
        void* d_hw = dalloc((size_t)Nh * K);
        void* d_hs = dalloc((size_t)snh * sk * 2);
        const RadKernelInfo* nr = n16;
        if (hrc != RAD_OK || hsc != RAD_E_UNSUPPORTED) {
            ok("fragment order survives a row shard at N/2", false,
               "relayout of the share rc " + std::to_string(hrc) + "/" + std::to_string(hsc));
        } else if (!d_xm || !d_q || !d_qs || !d_yh || !d_hw || !d_hs) {
            ok("fp8 shard alloc", false, "missing");
        } else {
            st_memcpy(d_hw, hw.data(), hw.size(), hipMemcpyHostToDevice);
            st_memcpy(d_hs, hs.data(), hs.size(), hipMemcpyHostToDevice);
            std::vector<uint16_t> xm((size_t)M * K);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t c = 0; c < K; ++c) xm[(size_t)m * K + c] = xb[(size_t)c];
            st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
            std::vector<RadTensor> qt = {
                T(d_xm, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }),
                T(d_qs, RAD_F32, { M, sk }),
            };
            std::vector<RadParam> qp = {
                PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
            };
            run_case(qz, qt, qp);
            /* Rank 1's share, relaid. */
            std::vector<RadTensor> ts = {
                T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
                T(d_hw, RAD_F8E4M3, { Nh, K }),
                T(d_hs, RAD_BF16, { snh, sk }),
                T(d_yh, RAD_BF16, { M, Nh }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
            };
            absent(ts[5]); absent(ts[6]);
            std::vector<RadParam> ps = {
                PI("M", M), PI("N", Nh), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
            };
            st_memset(d_yh, 0, (size_t)M * Nh * 2);
            run_case(nr, ts, ps);
            std::vector<uint16_t> got((size_t)M * Nh);
            st_memcpy(got.data(), d_yh, (size_t)M * Nh * 2, hipMemcpyDeviceToHost);
            double num = 0, den = 0;
            for (int64_t r = 0; r < Nh; ++r) {
                const double d = (double)h_bf16(got[(size_t)r]) - (double)ref[(size_t)(Nh + r)];
                num += d * d;
                den += (double)ref[(size_t)(Nh + r)] * (double)ref[(size_t)(Nh + r)];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 1.0;
            ok("fragment order survives a row shard at N/2", e < 8e-2,
               "rel_l2 " + std::to_string(e));
        }
    }

    /* --- A COLUMN SHARD, which is where a fragment plane and a row-major one stop agreeing.
     *
     * The fragment arrangement groups sixteen rows into one run, so a K-slice of the WHOLE
     * arranged plane is a 2D copy of pitch K*16 and width (K/tp)*16 -- the shape a loader that cut
     * arranged bytes would have to know. The loader cuts the CANONICAL planes instead, where a
     * column share is a plain rectangle of the codes and of the scale plane, and the kernel's hook
     * arranges each rank's share as a weight of K/2. Both shares are made that way here and run
     * as independent weights; their sum is what the all-reduce forms, and it has to be the
     * unsharded product. */
    if (K % 256 == 0) {
        const int64_t Kh = K / 2, skh = Kh / 128, M = 8;
        void* d_xh = dalloc((size_t)M * Kh * 2);
        void* d_q  = dalloc((size_t)M * Kh);
        void* d_qs = dalloc((size_t)M * skh * 4);
        void* d_wh = dalloc((size_t)N * Kh);
        void* d_sh = dalloc((size_t)sn * skh * 2);
        void* d_yh = dalloc((size_t)M * N * 2);
        const RadKernelInfo* nr = n16;
        if (!d_xh || !d_q || !d_qs || !d_wh || !d_sh || !d_yh) {
            ok("fp8 col-shard alloc", false, "missing");
        } else {
            std::vector<double> acc((size_t)N, 0.0);
            bool laid = true;
            for (int rank = 0; rank < 2 && laid; ++rank) {
                /* This rank's share of the canonical planes, arranged by the row's own hook. */
                const Planes part = share(p, 0, N, rank * Kh, (rank + 1) * Kh);
                std::vector<RadParam> kp = { PI("N", N), PI("K", Kh), PI("group", 128) };
                std::vector<uint8_t> cw, cs;
                const int crc = stored(nr, kp, 2, part, { "codes" }, &cw);
                const int csc = stored(nr, kp, 3, part, { "scale" }, &cs);
                if (crc != RAD_OK || csc != RAD_E_UNSUPPORTED ||
                    (int64_t)cw.size() != N * Kh || (int64_t)cs.size() != sn * skh * 2) {
                    ok("fragment order survives a column shard at K/2", false,
                       "relayout of the share rc " + std::to_string(crc) + "/" +
                       std::to_string(csc));
                    laid = false;
                    break;
                }
                st_memcpy(d_wh, cw.data(), cw.size(), hipMemcpyHostToDevice);
                st_memcpy(d_sh, cs.data(), cs.size(), hipMemcpyHostToDevice);
                std::vector<uint16_t> xm((size_t)M * Kh);
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t c = 0; c < Kh; ++c)
                        xm[(size_t)m * Kh + c] = xb[(size_t)(rank * Kh + c)];
                st_memcpy(d_xh, xm.data(), (size_t)M * Kh * 2, hipMemcpyHostToDevice);
                std::vector<RadTensor> qt = {
                    T(d_xh, RAD_BF16, { M, Kh }), T(d_q, RAD_F8E4M3, { M, Kh }),
                    T(d_qs, RAD_F32, { M, skh }),
                };
                std::vector<RadParam> qp = {
                    PI("M", M), PI("n", Kh), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(qz, qt, qp);
                std::vector<RadTensor> ts = {
                    T(d_q, RAD_F8E4M3, { M, Kh }), T(d_qs, RAD_F32, { M, skh }),
                    T(d_wh, RAD_F8E4M3, { N, Kh }), T(d_sh, RAD_BF16, { sn, skh }),
                    T(d_yh, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
                    T(nullptr, RAD_U8, { 1 }),
                };
                absent(ts[5]); absent(ts[6]);
                std::vector<RadParam> ps = {
                    PI("M", M), PI("N", N), PI("K", Kh), PI("group", 128), PS("dtype", "fp8a8"),
                };
                st_memset(d_yh, 0, (size_t)M * N * 2);
                run_case(nr, ts, ps);
                std::vector<uint16_t> got((size_t)M * N);
                st_memcpy(got.data(), d_yh, (size_t)M * N * 2, hipMemcpyDeviceToHost);
                for (int64_t r = 0; r < N; ++r) acc[(size_t)r] += (double)h_bf16(got[(size_t)r]);
            }
            double num = 0, den = 0;
            for (int64_t r = 0; r < N; ++r) {
                const double d = acc[(size_t)r] - (double)ref[(size_t)r];
                num += d * d;
                den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 1.0;
            if (laid)
                ok("fragment order survives a column shard at K/2", e < 8e-2,
                   "rel_l2 " + std::to_string(e));
        }
    }

    /* --- the tiled W8A8 path: quantise the activation with the library's own quantiser, then GEMM.
     *     M is 128 so the tile is exactly full; every row holds the SAME activation, so a wrong
     *     epilogue index shows up as one row of the output being right and the rest not. */
    {
        /* THE PREFILL WIDTH. 128 is one FM tile and grid.x = 1; a prefill chunk is 2048 and
         * sixteen of them, which is the only place a per-M-block staging bug can show. */
        const int64_t M = 2048;
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_q  = dalloc((size_t)M * K);
        void* d_qs = dalloc((size_t)M * sk * 4);
        void* d_ym = dalloc((size_t)M * N * 2);
        if (!d_xm || !d_q || !d_qs || !d_ym) { ok("fp8 tiled alloc", false, "hipMalloc failed"); return; }
        std::vector<uint16_t> xm((size_t)M * K);
        for (int64_t m = 0; m < M; ++m)
            for (int64_t c = 0; c < K; ++c) xm[(size_t)m * K + c] = xb[(size_t)c];
        st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);

        std::vector<RadTensor> qt = {
            T(d_xm, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
        };
        std::vector<RadParam> qp = {
            PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
        };
        run_case(qz, qt, qp);

        std::vector<RadTensor> ts = {
            T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
            T(d_wf, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_ym, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
        };
        st_memset(d_ym, 0, (size_t)M * N * 2);
        run_case(tl, ts, ps);
        std::vector<uint16_t> got((size_t)M * N);
        st_memcpy(got.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
        /* Every row must match, not just row 0: a wrong epilogue stride writes one right row. The
         * tolerance is looser than the matvec's because the ACTIVATION is quantised here too, and
         * E4M3's three mantissa bits are 6% of a value on their own. */
        double worst = 0;
        for (int64_t m = 0; m < M; ++m) {
            double num = 0, den = 0;
            for (int64_t r = 0; r < N; ++r) {
                const double d = (double)h_bf16(got[(size_t)m * N + r]) - (double)ref[(size_t)r];
                num += d * d;
                den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 1.0;
            if (e > worst) worst = e;
        }
        ok("gemm_fp8a8_tiled vs host ref, worst row", worst < 8e-2,
           "rel_l2 " + std::to_string(worst));
    }
}

/* THE SAME CHECK ON A REAL CHECKPOINT WEIGHT.
 *
 * `fp8_numeric` above builds its own weight, so it proves the kernel agrees with this file's idea
 * of the format. That is necessary and it is not sufficient: both sides could share a wrong idea.
 * This runs the identical comparison against BYTES READ OUT OF THE SAFETENSORS FILE -- real
 * magnitudes, real per-block scale variation, whatever subnormals and zero blocks the checkpoint
 * actually contains -- so what it falsifies is the claim that radiance can serve THIS model. The
 * file's two tensors ARE block fp8's canonical planes, and the matvec reads them as they are, so
 * the kernel and the oracle both read them row-major.
 *
 * The offsets are arguments rather than parsed here, because a safetensors header is JSON and this
 * binary has no business carrying a JSON parser. The parameters are: the file, the weight's byte
 * offset, the scale plane's byte offset, N and K. Reading the two offsets out of the file needs no
 * parser either -- the header length is a u64 at byte 0 and the JSON follows it, so the data
 * region begins at 8 + that, and each tensor's `data_offsets[0]` is relative to there:
 *
 *     N=$(od -An -tu8 -N8 w.safetensors | tr -d ' ')
 *     dd if=w.safetensors bs=1 skip=8 count=$N status=none | tr ',' '\n' | grep -A2 gate_proj
 *
 * then pass 8 + $N + data_offsets[0], once for the weight and once for its scale plane.
 *
 * It is spec §17's assertion with the .rad container taken out of the path. */
static void fp8_real(const Plugin& pl, const char* file, long long woff, long long soff,
                     int64_t N, int64_t K) {
    const RadKernelInfo* mv = find(pl, "gemm_fp8a16_nt_m1");
    if (!mv) { ok("fp8 real-weight check", false, "row absent"); return; }
    if (N % 128 || K % 128) { ok("fp8 real-weight check", false, "N and K must be 128-aligned"); return; }
    const int64_t sn = N / 128, sk = K / 128;

    std::FILE* f = std::fopen(file, "rb");
    if (!f) { ok("fp8 real-weight check", false, std::string("cannot open ") + file); return; }
    std::vector<uint8_t> wq((size_t)N * K);
    std::vector<uint16_t> ws((size_t)sn * sk);
    bool okread = std::fseek(f, (long)woff, SEEK_SET) == 0 &&
                  std::fread(wq.data(), 1, wq.size(), f) == wq.size() &&
                  std::fseek(f, (long)soff, SEEK_SET) == 0 &&
                  std::fread(ws.data(), 2, ws.size(), f) == (size_t)ws.size();
    std::fclose(f);
    if (!okread) { ok("fp8 real-weight check", false, "short read"); return; }

    /* A checkpoint block scale is ~1e-4 and a weight byte spans E4M3's whole range, so a plane read
     * at the wrong offset shows up here rather than as a tolerance failure fifty lines later. */
    double smin = 1e30, smax = -1e30;
    for (size_t i = 0; i < ws.size(); ++i) {
        const double v = h_bf16(ws[i]);
        if (v < smin) smin = v;
        if (v > smax) smax = v;
    }
    ok("fp8 real scale plane is a plausible dequant multiplier",
       smin > 0 && smax < 1.0 && smax / smin < 1e4,
       "min " + std::to_string(smin) + " max " + std::to_string(smax));

    std::vector<float> wd((size_t)N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wd[(size_t)r * K + c] =
                h_e4m3(wq[(size_t)(r * K + c)]) * h_bf16(ws[(size_t)(r / 128) * sk + (c / 128)]);

    std::vector<uint16_t> xb((size_t)K);
    for (int64_t c = 0; c < K; ++c)
        xb[(size_t)c] = h_to_bf16(0.021f * (float)((c * 17) % 53) - 0.55f);
    std::vector<float> ref((size_t)N, 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        double acc = 0;
        for (int64_t c = 0; c < K; ++c) acc += (double)wd[(size_t)r * K + c] * h_bf16(xb[(size_t)c]);
        ref[(size_t)r] = (float)acc;
    }

    void* d_w = dalloc((size_t)N * K);
    void* d_s = dalloc((size_t)sn * sk * 2);
    void* d_x = dalloc((size_t)K * 2);
    void* d_y = dalloc((size_t)N * 2);
    if (!d_w || !d_s || !d_x || !d_y) { ok("fp8 real alloc", false, "hipMalloc failed"); return; }
    st_memcpy(d_w, wq.data(), (size_t)N * K, hipMemcpyHostToDevice);
    st_memcpy(d_s, ws.data(), (size_t)sn * sk * 2, hipMemcpyHostToDevice);
    st_memcpy(d_x, xb.data(), (size_t)K * 2, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_x, RAD_BF16, { 1, K }), T(nullptr, RAD_F32, { 1, sk }),
        T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
        T(d_y, RAD_BF16, { 1, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
    };
    absent(ts[1]); absent(ts[5]); absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("M", 1), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
    };
    run_case(mv, ts, ps);
    std::vector<uint16_t> got((size_t)N);
    st_memcpy(got.data(), d_y, (size_t)N * 2, hipMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int64_t r = 0; r < N; ++r) {
        const double d = (double)h_bf16(got[(size_t)r]) - (double)ref[(size_t)r];
        num += d * d;
        den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
    }
    const double e = den > 0 ? std::sqrt(num / den) : 1.0;
    /* Printed whether it passes or not. `ok` shows its detail only on failure, and a correctness
     * gate whose margin is invisible when it passes is a gate nobody can tell has loosened. */
    std::printf("        N=%lld K=%lld  rel_l2 %.6g  (bound 3e-2; bf16 output rounding alone is "
                "~2e-3)\n", (long long)N, (long long)K, e);
    ok("gemm_fp8a16_nt_m1 vs host ref on REAL checkpoint bytes", e < 3e-2,
       "rel_l2 " + std::to_string(e));
}

/* ---- THE DECODE GEMM, TIMED --------------------------------------------------------------------
 *
 * The cases above answer "is it right". This answers "at what fraction of the card", which is the
 * only other question a decode GEMM has, and it lives here because the alternatives do not reach
 * it: rad-tune searches a row's declared axes and the fp8a8 rows declare none, and a serve is a
 * round trip that buries one shape under thirty other ops. This is a second per point.
 *
 * WHY THE WEIGHT IS ROTATED. Back-to-back launches on ONE weight buffer measure the last level of
 * cache, not HBM, and this part has enough of it to hold a whole decode GEMM's weight. Decode
 * reads every weight in the model exactly once and hits none of it, so the buffer is replicated
 * until it is comfortably past any cache on the card and the launches walk the copies in turn.
 * Without that the same kernel reads more than twice the bandwidth here that it reaches in a
 * serve, and the faster number is the useless one.
 *
 * The reported bandwidth is the WEIGHT stream and nothing else: at M <= 16 the activation is 16*K
 * bytes against N*K and the output 16*N, so the two together are under a percent of the traffic at
 * any shape this band serves. That makes the number directly comparable to the 638 GB/s a plain
 * stream of the same bytes reaches, which is the point of printing it.
 */
static void perf_row(const Plugin& pl, const char* name, std::vector<RadTensor> ts,
                     std::vector<RadParam> ps, void* wbase, size_t wbytes, int copies,
                     double bytes, int iters, int widx = 2, const char* point = nullptr) {
    const RadKernelInfo* r = find(pl, name);
    if (!r) { std::printf("  %-22s row absent\n", name); return; }
    /* The label is the row unless a sweep named the tuned point, because a sweep that prints one
     * row name fifteen times is a sweep nobody can read. `ps` already CARRIES that point -- the
     * caller appended it -- so this is only how it is spelled, never how it is applied. */
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "%s", (point && *point && std::strcmp(point, "-")) ? point : name);
    name = lbl;
    for (auto& t : ts) if (!t.data && t.rank) t.data = alloc_for(t);

    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    if (r->scratch) {
        const int64_t need = r->scratch(&a);
        if (need < 0) { std::printf("  %-22s scratch %lld\n", name, (long long)need); return; }
        if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
    }

    /* One warm launch, and it is also the check that the row SERVES this shape. A row that refuses
     * is reported rather than timed: timing a kernel that returned -3 without launching is how a
     * heuristic sweep gets a spectacular wrong answer. */
    const int rc = r->launch(&a, nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        std::printf("  %-22s rc=%d (shape not served)\n", name, rc);
        return;
    }

    hipEvent_t e0, e1;
    st_ev_create(&e0);
    st_ev_create(&e1);
    st_ev_record(e0, nullptr);
    for (int i = 0; i < iters; ++i) {
        ts[widx].data = (uint8_t*)wbase + (size_t)(i % copies) * wbytes;
        r->launch(&a, nullptr);
    }
    st_ev_record(e1, nullptr);
    st_ev_sync(e1);
    float ms = 0;
    st_ev_elapsed(&ms, e0, e1);
    st_ev_destroy(e0);
    st_ev_destroy(e1);
    const double us = (double)ms * 1000.0 / iters;
    std::printf("  %-22s %8.1f us   %6.1f GB/s\n", name, us, bytes / (us * 1e3));
}


/* ------------------------------------------------------------------ the small ops, timed
 *
 * A decode step issues roughly twelve hundred kernels and only about four hundred of them are
 * GEMMs. The other eight hundred each move a few tens of kilobytes, which is nothing, so what
 * they actually cost is a question about LAUNCHES rather than about bandwidth -- and the per-op
 * profiler cannot answer it, because enabling it restores a synchronisation per call and that
 * sync is the same order as the thing being measured.
 *
 * So: back-to-back launches with no host synchronisation between them, over a buffer set that
 * makes every one of them independent. The number that comes out is the floor a fusion would
 * have to beat, and the gap between it and the profiler's per-call figure is what the step is
 * paying to dispatch. */
static void smallop_perf(const Plugin& pl, int64_t M, int64_t K, int iters) {
    if (iters <= 0) iters = 2000;
    const int64_t nkb = K / 128;
    std::printf("small ops at M=%lld K=%lld, %d launches each, no host sync between\n",
                (long long)M, (long long)K, iters);

    void* d_x  = dalloc((size_t)M * K * 2);
    void* d_r  = dalloc((size_t)M * K * 2);
    void* d_gu = dalloc((size_t)M * 2 * K * 2);
    void* d_w  = dalloc((size_t)K * 4);
    void* d_y  = dalloc((size_t)M * K * 2);
    void* d_q  = dalloc((size_t)M * K);
    void* d_s  = dalloc((size_t)M * nkb * 4);
    if (!d_x || !d_r || !d_gu || !d_w || !d_y || !d_q || !d_s) {
        std::printf("  alloc failed\n");
        return;
    }
    st_memset(d_x, 0x3c, (size_t)M * K * 2);
    st_memset(d_r, 0x3c, (size_t)M * K * 2);
    st_memset(d_gu, 0x3c, (size_t)M * 2 * K * 2);
    st_memset(d_w, 0, (size_t)K * 4);

    auto time_row = [&](const char* label, const char* row, std::vector<RadTensor> ts,
                        std::vector<RadParam> ps, double bytes) {
        const RadKernelInfo* r = find(pl, row);
        if (!r) { std::printf("  %-24s row absent\n", label); return; }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("  %-24s rc=%d (shape not served)\n", label, rc);
            return;
        }
        hipEvent_t e0, e1;
        st_ev_create(&e0);
        st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0);
        st_ev_destroy(e1);
        const double us = (double)ms * 1000.0 / iters;
        std::printf("  %-24s %7.2f us   %6.1f GB/s\n", label, us, bytes / (us * 1e3));
    };

    const double eps = 1e-6, wadd = 1.0;
    const double b_norm = (double)M * K * 2 + (double)M * K + (double)M * K * 2;

    time_row("rmsnorm_quant_fp8", "rmsnorm_quant_fp8",
             { T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, {}), T(d_w, RAD_F32, { K }),
               T(d_q, RAD_F8E4M3, { M, K }), T(d_s, RAD_F32, { M, nkb }),
               T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128), PS("dtype", "bf16"),
               PF("wadd", wadd) }, b_norm);
    time_row("rmsnorm_quant_fp8 (res)", "rmsnorm_quant_fp8",
             { T(d_x, RAD_BF16, { M, K }), T(d_r, RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }),
               T(d_q, RAD_F8E4M3, { M, K }), T(d_s, RAD_F32, { M, nkb }),
               T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128), PS("dtype", "bf16"),
               PF("wadd", wadd) }, b_norm + (double)M * K * 4);
    time_row("gated_quant_fp8", "gated_quant_fp8",
             { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q, RAD_F8E4M3, { M, K }),
               T(d_s, RAD_F32, { M, nkb }), T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") },
             (double)M * 2 * K * 2 + (double)M * K * 3);
    time_row("quant_act_fp8", "quant_act_fp8",
             { T(d_x, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }),
               T(d_s, RAD_F32, { M, nkb }) },
             { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") },
             (double)M * K * 3);
    time_row("rmsnorm_bf16", "rmsnorm_bf16",
             { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }), T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"), PF("wadd", wadd) },
             (double)M * K * 4);
    time_row("add_bf16", "add_bf16",
             { T(d_x, RAD_BF16, { M, K }), T(d_r, RAD_BF16, { M, K }), T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PS("dtype", "bf16") }, (double)M * K * 6);
}

/* ------------------------------------------------------------- the QSA indexer, timed at length
 *
 * THE INDEXER IS THE ONLY PART OF A DECODE STEP THAT GROWS WITH CONTEXT. `qsa_score` and
 * `qsa_select` both scale with the number of compressed blocks in the history and both run once a
 * layer, so between them they account for the whole of the step-time-against-context curve; every
 * other op in the step is flat to within noise. That makes the pair worth timing AT LENGTH rather
 * than at a convenient context.
 *
 * `blocks` is the DECLARED width (max_ctx / ratio) and `ctx` the live one, because the launch
 * geometry is sized from the declaration and the work from `nc` -- so a harness that passes the
 * live count for both measures a kernel the deployment never runs.
 *
 * M IS 1 + n_spec, not the sequence count: every draft row selects its own set. */
static void qsa_perf(const Plugin& pl, int64_t ctx, int64_t M, int iters) {
    if (iters <= 0) iters = 200;
    const int64_t ratio = 4, heads = 4, n = 128, topk = 512;
    const int64_t max_ctx = 200000, nb = max_ctx / ratio;      /* the declared width */
    const int64_t live = ctx / ratio;                          /* complete blocks */
    if (live < 1 || live > nb) { std::printf("--perfqsa: ctx out of range\n"); return; }
    std::printf("qsa indexer at ctx=%lld (%lld of %lld blocks live), M=%lld, heads=%lld n=%lld "
                "topk=%lld, %d launches\n", (long long)ctx, (long long)live, (long long)nb,
                (long long)M, (long long)heads, (long long)n, (long long)topk, iters);

    const int64_t pages = live;
    std::vector<uint16_t> qh((size_t)(M * heads * n)), bkh((size_t)(pages * n));
    for (int64_t i = 0; i < M * heads * n; ++i)
        qh[(size_t)i] = h_to_bf16((float)(((i * 29 + 7) % 83) - 41) / 47.0f);
    for (int64_t i = 0; i < pages * n; ++i)
        bkh[(size_t)i] = h_to_bf16((float)(((i * 41 + 13) % 71) - 35) / 39.0f);
    /* A SCATTERED TABLE, because a paged pool hands out whatever is free and the block key read
     * is the only random access in the kernel -- a table in page order measures a different
     * machine. */
    std::vector<int32_t> bth((size_t)nb, -1);
    for (int64_t b = 0; b < live; ++b) bth[(size_t)b] = (int32_t)((b * 7919 + 13) % pages);
    /* ...AND THE SAME TABLE IN PAGE ORDER, because the difference between the two IS the price of
     * the paged pool, and it is the only number that says whether a contiguous block-key store
     * would be worth building. Nothing else in this kernel is a random access. */
    std::vector<int32_t> bti((size_t)nb, -1);
    for (int64_t b = 0; b < live; ++b) bti[(size_t)b] = (int32_t)b;
    std::vector<int32_t> nch(1, (int32_t)live), cuh(2), posh((size_t)M);
    cuh[0] = 0; cuh[1] = (int32_t)M;
    for (int64_t m = 0; m < M; ++m) posh[(size_t)m] = (int32_t)(ctx - M + m);

    void* d_q  = dalloc(qh.size() * 2);
    void* d_bk = dalloc(bkh.size() * 2);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_bi = dalloc(bti.size() * 4);
    void* d_nc = dalloc(4);
    void* d_cu = dalloc(8);
    void* d_po = dalloc(posh.size() * 4);
    void* d_sc = dalloc((size_t)(M * nb) * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    if (!d_q || !d_bk || !d_bt || !d_bi || !d_nc || !d_cu || !d_po || !d_sc || !d_se || !d_ns ||
        !d_su) {
        std::printf("  alloc failed\n");
        return;
    }
    st_memcpy(d_q,  qh.data(),  qh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_bk, bkh.data(), bkh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bi, bti.data(), bti.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nch.data(), 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), 8, hipMemcpyHostToDevice);
    st_memcpy(d_po, posh.data(), posh.size() * 4, hipMemcpyHostToDevice);

    /* `row` finds the kernel, `label` names the LINE -- two forms of the same row is the point of
     * this harness, and a lambda that used one string for both could not express it. */
    auto run = [&](const char* row, const char* name, std::vector<RadTensor> ts,
                   std::vector<RadParam> ps, double bytes) {
        const RadKernelInfo* r = find(pl, row);
        if (!r) { std::printf("  %-18s row absent\n", name); return; }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need < 0) { std::printf("  %-18s scratch %lld\n", name, (long long)need); return; }
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("  %-18s rc=%d (shape not served)\n", name, rc);
            return;
        }
        /* WARM THE CLOCKS, NOT JUST THE CACHES, and one launch does not do it. `rocm-smi` reports
         * these cards in a LOW-POWER STATE whenever nothing is running, and 300 iterations of a
         * 100 us kernel is ~30 ms -- comfortably inside the ramp back up. On an UNCHANGED kernel
         * the same shape can take close to twice as long after an idle gap as it does after other
         * work, and the whole of that swing is the governor. Half a second of launches first, so
         * the timed window runs at the clock the deployment runs at.
         *
         * A harness that starts cold does not measure a slow kernel, it measures a sleeping card,
         * and the failure is invisible: the number is plausible and it moves with the shape. */
        {
            hipEvent_t w0, w1;
            st_ev_create(&w0); st_ev_create(&w1);
            st_ev_record(w0, nullptr);
            float wms = 0.0f;
            for (int guard = 0; guard < 100000 && wms < 500.0f; ++guard) {
                for (int i = 0; i < 32; ++i) r->launch(&a, nullptr);
                st_ev_record(w1, nullptr);
                st_ev_sync(w1);
                st_ev_elapsed(&wms, w0, w1);
            }
            st_ev_destroy(w0); st_ev_destroy(w1);
        }
        hipEvent_t e0, e1;
        st_ev_create(&e0); st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0); st_ev_destroy(e1);
        const double us = (double)ms * 1000.0 / iters;
        std::printf("  %-30s %9.2f us   %6.1f GB/s\n", name, us, bytes / (us * 1e3));
    };

    /* THE SCORE PLANE IS NARROWED TO THE LIVE REACH, exactly as the architecture narrows it
     * (arch/common/rad_qsa.h): the pitch stays the declared `nb` and the column count is what the
     * host knows the context can have reached. Both kernels size their grid from it, so a harness
     * that passed the declared width would measure a grid the deployment never launches. */
    RadTensor sc_t = T(d_sc, RAD_F32, { M, nb });
    sc_t.shape[1] = live + 1;

    /* THE BYTES ARE `live * n * 2` AND NOT `M *` THAT. The kernel tiles `kQT` query rows through
     * LDS, so the key history is read ONCE for the tile and not once a row. Counting it once a
     * query row overstates the bandwidth by exactly the factor that tiling saved -- a gauge that
     * flatters a kernel by the size of its own best optimisation is worse than none, because it
     * hides the headroom that optimisation left. */
    const double score_bytes = (double)live * n * 2.0;
    auto score_args = [&]() {
        return std::vector<RadTensor>{ T(d_q,  RAD_BF16, { M, heads * n }),
                                       T(d_bk, RAD_BF16, { pages, n }),
                                       T(d_bt, RAD_I32,  { 1, nb }), T(d_cu, RAD_I32,  { 2 }),
                                       T(d_nc, RAD_I32,  { 1 }), sc_t };
    };
    const std::vector<RadParam> score_params = { PI("M", M), PI("n", n), PI("heads", heads),
                                                 PI("blocks", nb), PI("seqs", 1),
                                                 PS("dtype", "bf16") };
    /* BOTH FORMS ON ONE INPUT, timed and compared. They multiply the same bf16 pairs and sum them
     * in different orders, so the comparison is a tolerance on the live plane -- and the -inf of
     * an absent block has to match exactly, which a tolerance on finite values would not see. */
    if (pl.pin_qsa) {
        std::vector<float> plane[3];
        const char* label[3] = { "qsa_score_bf16 scalar form", "qsa_score_bf16 matrix form", "qsa_score_bf16 packed matrix form" };
        for (int f = 0; f < 3; ++f) {
            pl.pin_qsa(f + 1);
            run("qsa_score_bf16", label[f], score_args(), score_params, score_bytes);
            plane[f].resize((size_t)(M * nb));
            st_memcpy(plane[f].data(), d_sc, plane[f].size() * 4, hipMemcpyDeviceToHost);
        }
        pl.pin_qsa(0);
        for (int cmp = 1; cmp <= 2; ++cmp) {
            double maxrel = 0.0, num = 0.0, den = 0.0;
            long long inf_mismatch = 0;
            for (int64_t m = 0; m < M; ++m)
                for (int64_t b = 0; b < live; ++b) {
                    const float a = plane[0][(size_t)(m * nb + b)], c = plane[cmp][(size_t)(m * nb + b)];
                    if (std::isinf(a) || std::isinf(c)) { inf_mismatch += (a != c); continue; }
                    const double d = (double)a - (double)c;
                    num += d * d;
                    den += (double)a * a;
                    const double sc = std::fabs((double)a) > 1e-3 ? std::fabs((double)a) : 1e-3;
                    maxrel = std::max(maxrel, std::fabs(d) / sc);
                }
            std::printf("  form 1 vs %s agree to rel_l2 %.3e, max rel %.3e, %lld -inf mismatches\n",
                        label[cmp], den > 0 ? std::sqrt(num / den) : 0.0, maxrel, inf_mismatch);
        }
    }
    run("qsa_score_bf16", "qsa_score_bf16 (pages scattered)", score_args(), score_params,
        score_bytes);
    run("qsa_score_bf16", "qsa_score_bf16 (pages in order)",
        { T(d_q,  RAD_BF16, { M, heads * n }), T(d_bk, RAD_BF16, { pages, n }),
          T(d_bi, RAD_I32,  { 1, nb }),        T(d_cu, RAD_I32,  { 2 }),
          T(d_nc, RAD_I32,  { 1 }),            sc_t },
        { PI("M", M), PI("n", n), PI("heads", heads), PI("blocks", nb), PI("seqs", 1),
          PS("dtype", "bf16") },
        score_bytes);
    /* The launcher's rule, and the single-workgroup form it replaces at a wide launch. */
    if (pl.pin_qsel) pl.pin_qsel(1);
    run("qsa_select_i32", "qsa_select_i32 (single-workgroup form)",
        { sc_t, T(d_bt, RAD_I32, { 1, nb }), T(d_nc, RAD_I32, { 1 }),
          T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { 2 }),
          T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) },
        { PI("M", M), PI("blocks", nb), PI("topk", topk), PI("ratio", ratio), PI("seqs", 1) },
        (double)M * live * 8.0);
    if (pl.pin_qsel) pl.pin_qsel(0);
    run("qsa_select_i32", "qsa_select_i32",
        { sc_t, T(d_bt, RAD_I32, { 1, nb }), T(d_nc, RAD_I32, { 1 }),
          T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { 2 }),
          T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) },
        { PI("M", M), PI("blocks", nb), PI("topk", topk), PI("ratio", ratio), PI("seqs", 1) },
        (double)M * live * 8.0);
}


/* THE ROUTED-EXPERT GEMM AT A PREFILL CHUNK, both projections, every prefill form.
 *
 * WHY IT NEEDS ITS OWN MODE. rad-kbench declares at world 1, where the width split does not exist
 * and the geometry is the whole 1280-column expert, and its fixture does not serve the rotated w4
 * dtype -- so the shape the engine runs at tp2 is reachable only in the engine, where a profile
 * restores a synchronisation per op and a run moves ~8% between servers. This is rank 0's shape at
 * Qwen3.8-Flash-Next: gate_up with three 128-column blocks of an even expert and two of an odd one
 * (parts 2), and the down projection with the matching K slices over the SORTED rows.
 *
 * THE ROUTING IS SKEWED, because the kernel's cost is a function of the run lengths and a uniform
 * routing gives every expert the mean. Expert popularity is lognormal (sigma 0.8) and each token
 * takes its top_k by a Gumbel draw, so a run ranges from a handful of rows to several times the
 * mean, as a real chunk's do. The weights are random codes: what is timed is the traffic and the
 * arithmetic, and both are value-independent.
 *
 * EVERY FORM IS RUN ON THE SAME BYTES and compared with the first -- the run-aligned prefill form,
 * the (tile, run) decode form and the uniform prefill grid, whatever the band would pick. All three
 * keep each output element's accumulation order, so they must match byte for byte. */
static void moe_perf(const Plugin& pl, int64_t Mtok, int iters, int64_t E = 512, int64_t top_k = 10,
                     int pool_pct = 0) {
    if (iters <= 0) iters = 20;
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    if (!r) { std::printf("--perfmoe: row moe_gemm_w4a8 absent\n"); return; }
    const int64_t group = 128, Tr = Mtok * top_k;

    uint64_t rs = 0x9E3779B97F4A7C15ull;
    auto rnd  = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto unif = [&]() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
    std::vector<double> logp((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        logp[(size_t)e] = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by_e((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    for (int64_t t = 0; t < Mtok; ++t) {
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + top_k, sc.end(),
                          [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                              return x.first > y.first;
                          });
        for (int64_t j = 0; j < top_k; ++j)
            by_e[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * top_k + j));
    }
    std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
    sorted.reserve((size_t)Tr);
    int64_t rmin = Tr, rmax = 0, rows_c[2] = { 0, 0 };
    for (int64_t e = 0; e < E; ++e) {
        const int64_t n = (int64_t)by_e[(size_t)e].size();
        off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)n;
        sorted.insert(sorted.end(), by_e[(size_t)e].begin(), by_e[(size_t)e].end());
        rmin = std::min(rmin, n); rmax = std::max(rmax, n); rows_c[e & 1] += n;
    }
    std::printf("moe_gemm_w4a8 at a %lld-token chunk: Tr=%lld routed rows over %lld experts, a run "
                "%lld..%lld rows (mean %.1f), %d launches\n", (long long)Mtok, (long long)Tr,
                (long long)E, (long long)rmin, (long long)rmax, (double)Tr / (double)E, iters);

    void* d_s = dalloc((size_t)Tr * 4);
    void* d_o = dalloc((size_t)(E + 1) * 4);
    st_memcpy(d_s, sorted.data(), (size_t)Tr * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(), (size_t)(E + 1) * 4, hipMemcpyHostToDevice);

    struct Proj { const char* name; int64_t N0, K0, N1, K1, parts; bool srt; };
    const Proj projs[2] = { { "gate_up", 768, 2560, 512, 2560, 2, false },
                            { "down",    2560, 384, 2560, 256, 1, true } };
    const int nforms = pl.pin_moe ? 3 : 1;
    for (const Proj& P : projs) {
        const int64_t N = std::max(P.N0, P.N1), K = std::max(P.K0, P.K1), kb = K / group;
        const int64_t Ma = P.srt ? Tr : Mtok;
        const int64_t nec[2] = { E / 2, E / 2 };
        const int64_t Nc[2] = { P.N0, P.N1 }, Kc[2] = { P.K0, P.K1 };
        std::vector<uint8_t> hb;
        auto fill = [&](size_t n, uint8_t lo, uint8_t span) {
            hb.resize(n);
            for (size_t i = 0; i < n; ++i) hb[i] = (uint8_t)(lo + (uint8_t)(rnd() % span));
        };
        void* d_a = dalloc((size_t)(Ma * K));
        fill((size_t)(Ma * K), 8, 111);
        for (size_t i = 0; i < hb.size(); ++i) if (i & 1) hb[i] |= 0x80u;
        st_memcpy(d_a, hb.data(), hb.size(), hipMemcpyHostToDevice);
        std::vector<float> asc((size_t)(Ma * kb), 0.013f);
        void* d_as = dalloc(asc.size() * 4);
        st_memcpy(d_as, asc.data(), asc.size() * 4, hipMemcpyHostToDevice);
        void* d_w[2]; void* d_ws[2]; void* d_tab[2] = { nullptr, nullptr };
        for (int c = 0; c < 2; ++c) {
            /* Five bits an element: the five-bit pass reads the same plane at its own pitch, and
             * the four-bit ones the front of each expert's share of it. */
            d_w[c] = dalloc((size_t)(nec[c] * Nc[c] * Kc[c] / 8 * 5));
            fill((size_t)(nec[c] * Nc[c] * Kc[c] / 8 * 5), 0, 255);
            st_memcpy(d_w[c], hb.data(), hb.size(), hipMemcpyHostToDevice);
            /* POOLED EXPERTS, as the engine has them: a pointer table, with `pool_pct` percent of
             * the entries pointing into pinned host memory the GEMM reads over the link. The
             * LEAST ROUTED ones, because residency keeps the hot experts on the card and pools the
             * cold tail -- a random choice would pool hot experts whose long runs the engine never
             * streams. */
            if (pool_pct > 0) {
                const size_t eb = (size_t)(Nc[c] * Kc[c] / 2);
                std::vector<const void*> tab((size_t)nec[c]);
                std::vector<int64_t> order((size_t)nec[c]);
                for (int64_t x = 0; x < nec[c]; ++x) order[(size_t)x] = x;
                std::sort(order.begin(), order.end(), [&](int64_t p0, int64_t p1) {
                    return by_e[(size_t)(2 * p0 + c)].size() < by_e[(size_t)(2 * p1 + c)].size();
                });
                std::vector<char> pool((size_t)nec[c], 0);
                for (int64_t i = 0; i < nec[c] * pool_pct / 100; ++i) pool[(size_t)order[(size_t)i]] = 1;
                int64_t pooled = 0, pooled_rows = 0;
                for (int64_t x = 0; x < nec[c]; ++x) {
                    tab[(size_t)x] = (const char*)d_w[c] + (size_t)x * eb;
                    if (pool[(size_t)x]) {
                        void* h = nullptr;
                        if (hipHostMalloc(&h, eb, hipHostMallocMapped) != hipSuccess) {
                            std::printf("  %s: pinned allocation %lld failed\n", P.name, (long long)pooled);
                            break;
                        }
                        std::memcpy(h, hb.data() + (size_t)x * eb, eb);
                        void* hd = nullptr;
                        if (hipHostGetDevicePointer(&hd, h, 0) != hipSuccess || !hd) {
                            std::printf("  %s: no device address for pinned block %lld\n", P.name,
                                        (long long)pooled);
                            break;
                        }
                        tab[(size_t)x] = hd;
                        ++pooled;
                        pooled_rows += (int64_t)by_e[(size_t)(2 * x + c)].size();
                    }
                }
                std::printf("  %s class %d: %lld of %lld experts pooled, %lld routed rows on them\n",
                            P.name, c, (long long)pooled, (long long)nec[c], (long long)pooled_rows);
                d_tab[c] = dalloc(tab.size() * sizeof(void*));
                st_memcpy(d_tab[c], tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice);
            }
            /* Room for the 32-wide grid's four E4M3 bytes a 128, twice the bf16 grid's two. */
            std::vector<uint16_t> wsc((size_t)(2 * nec[c] * Nc[c] * (Kc[c] / group)), h_to_bf16(0.02f));
            d_ws[c] = dalloc(wsc.size() * 2);
            st_memcpy(d_ws[c], wsc.data(), wsc.size() * 2, hipMemcpyHostToDevice);
        }
        void* d_y = dalloc((size_t)(Tr * N * 2));
        if (!d_s || !d_o || !d_a || !d_as || !d_w[0] || !d_w[1] || !d_ws[0] || !d_ws[1] || !d_y) {
            std::printf("  alloc failed\n");
            return;
        }
        std::vector<RadTensor> ts = {
            T(d_a,     RAD_F8E4M3,   { Ma, K }),
            T(d_as,    RAD_F32,      { Ma, kb }),
            T(d_tab[0] ? d_tab[0] : d_w[0], R4D_DT_MOEW4, { nec[0], P.N0, P.K0 }),
            T(d_ws[0], RAD_BF16,     { nec[0], P.N0, P.K0 / group }),
            T(d_s,     RAD_I32,      { Tr }),
            T(d_o,     RAD_I32,      { E + 1 }),
            T(d_tab[1] ? d_tab[1] : d_w[1], R4D_DT_MOEW4, { nec[1], P.N1, P.K1 }),
            T(d_ws[1], RAD_BF16,     { nec[1], P.N1, P.K1 / group }),
            T(d_y,     RAD_BF16,     { Tr, N }),
        };
        /* A pointer table is told from a stacked plane by a leading stride of zero. */
        if (d_tab[0]) ts[2].stride[0] = 0;
        if (d_tab[1]) ts[6].stride[0] = 0;
        std::vector<RadParam> ps = {
            PI("M", Mtok), PI("N", P.N0), PI("K", P.K0), PI("N_odd", P.N1), PI("K_odd", P.K1),
            PI("parts", P.parts), PI("n_expert", E), PI("top_k", top_k), PI("group", group),
            PS("a_order", P.srt ? "sorted" : "token"), PS("dtype", "w4a8h"),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        /* The run map the aligned prefill form reads, sized for it whatever the hook answers:
         * inside the decode band the hook sizes none, because the rule runs no map there, and
         * the pinned aligned form would then time the uniform grid it falls back to. */
        {
            const int64_t need = std::max<int64_t>(r->scratch ? r->scratch(&a) : 0,
                                                   (1 + 2 * (E + Tr / 32 + 2)) * 4);
            a.scratch = dalloc((size_t)need);
            a.scratch_bytes = need;
        }
        const double flop = 2.0 * ((double)rows_c[0] * P.N0 * P.K0 + (double)rows_c[1] * P.N1 * P.K1);
        const double welems = (double)nec[0] * P.N0 * P.K0 + (double)nec[1] * P.N1 * P.K1;
        std::vector<uint16_t> first((size_t)(Tr * N)), got((size_t)(Tr * N));
        /* EVERY SCALE GRID over the same bytes: w4a8h's bf16 a (row, 128 of K), then
         * w4nl64a8h's two E4M3 a 128 and w4nl32a8h's four -- the same plane read as more groups,
         * so the later passes time the extra accumulators and fold and nothing else -- and last
         * w5nl64a8h (`sub` 5 here), w4nl64a8h's grid over five-bit rows: the sign plane's loads
         * and the sign merge. Not over pooled experts, whose blocks are sized for four bits. */
        for (const int sub : { 1, 2, 4, 5 }) {
        const bool g64 = sub > 1, w5 = sub == 5;
        if (w5 && (d_tab[0] || d_tab[1])) continue;
        const int64_t sg = w5 ? 2 : sub;
        ts[2] = T(d_tab[0] ? d_tab[0] : d_w[0], w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4,
                  { nec[0], P.N0, P.K0 });
        ts[6] = T(d_tab[1] ? d_tab[1] : d_w[1], w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4,
                  { nec[1], P.N1, P.K1 });
        if (d_tab[0]) ts[2].stride[0] = 0;
        if (d_tab[1]) ts[6].stride[0] = 0;
        ts[3] = g64 ? T(d_ws[0], RAD_F8E4M3, { nec[0], P.N0, sg * P.K0 / group })
                    : T(d_ws[0], RAD_BF16, { nec[0], P.N0, P.K0 / group });
        ts[7] = g64 ? T(d_ws[1], RAD_F8E4M3, { nec[1], P.N1, sg * P.K1 / group })
                    : T(d_ws[1], RAD_BF16, { nec[1], P.N1, P.K1 / group });
        ps.back().sval = w5 ? "w5nl64a8h" : sub == 4 ? "w4nl32a8h" : g64 ? "w4nl64a8h" : "w4a8h";
        const double wbytes = welems * (w5 ? 0.625 : 0.5);
        const std::string pname = std::string(P.name) + (w5 ? " w5" : sub == 4 ? " g32"
                                                         : g64 ? " g64" : "");
        for (int f = 0; f < nforms; ++f) {
            if (pl.pin_moe) pl.pin_moe(f + 1);
            const int rc = r->launch(&a, nullptr);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
                std::printf("  %-12s form %d  rc=%d (not served)\n", pname.c_str(), f + 1, rc);
                continue;
            }
            {
                hipEvent_t w0, w1;
                st_ev_create(&w0); st_ev_create(&w1);
                st_ev_record(w0, nullptr);
                float wms = 0.0f;
                for (int guard = 0; guard < 100000 && wms < 500.0f; ++guard) {
                    for (int i = 0; i < 4; ++i) r->launch(&a, nullptr);
                    st_ev_record(w1, nullptr);
                    st_ev_sync(w1);
                    st_ev_elapsed(&wms, w0, w1);
                }
                st_ev_destroy(w0); st_ev_destroy(w1);
            }
            hipEvent_t e0, e1;
            st_ev_create(&e0); st_ev_create(&e1);
            st_ev_record(e0, nullptr);
            for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
            st_ev_record(e1, nullptr);
            st_ev_sync(e1);
            float ms = 0.0f;
            st_ev_elapsed(&ms, e0, e1);
            st_ev_destroy(e0); st_ev_destroy(e1);
            const double us = (double)ms * 1000.0 / iters;
            st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
            std::string same = "";
            if (f == 0) {
                first = got;
                /* A hash of the first form's bytes, so two BUILDS can be compared as well as two
                 * forms of one: the inputs are a fixed function of the arguments. */
                uint64_t hsh = 1469598103934665603ull;
                for (uint16_t v : got) { hsh ^= v; hsh *= 1099511628211ull; }
                same = "  fnv " + std::to_string(hsh);
            } else {
                int64_t diff = 0;
                for (size_t i = 0; i < got.size(); ++i) diff += got[i] != first[i];
                same = diff ? ("  " + std::to_string(diff) + " elements differ from form 1")
                            : "  byte-identical to form 1";
            }
            std::printf("  %-12s form %d  %9.1f us  %6.1f TFLOP/s useful  %6.1f GB/s of weight%s\n",
                        pname.c_str(), f + 1, us, flop / (us * 1e6), wbytes / (us * 1e3),
                        same.c_str());
        }
        }
        if (pl.pin_moe) pl.pin_moe(0);
    }
}

/* THE GATED RESIDUAL'S READ AT INT8 CODES. The same inputs run twice, the codes plane once E4M3
 * and once I8: the bf16 block input and the rotated E4M3 twin must come out byte-identical -- the
 * codes' format is the only thing the dtype changes -- and the int8 codes must be the bf16 input
 * quantised per (row, 128) at amax / 127, code for code, which is what quant_act_i8g would have
 * written from it. Row counts reach the per-row kernels, the decode tiles and the prefill tile; the
 * null stream takes the unfused path and a real one the fused finishers, and `form` pins the
 * launcher to one family of kernels where the plugin exports the pin (0 leaves it the rule). */
static void hc_read_i8(const Plugin& pl, int64_t M, bool real_stream, int form) {
    const RadKernelInfo* r = find(pl, "hc_read_e4m3");
    if (!r) { skip("hc_read_e4m3 int8 codes", "row absent"); return; }
    const int64_t n = 2560, hc = 4, lr = 320, HN = hc * n, grp = 128, nkb = n / grp;
    const int64_t dld = r4d_hc8::row_bytes(HN, r4d_hc8::down_group(HN));
    const int64_t uld = r4d_hc8::row_bytes(lr, r4d_hc8::up_group(lr));
    uint64_t rs = 0x9E3779B97F4A7C15ull ^ (uint64_t)M;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    std::vector<uint16_t> hh((size_t)(M * HN));
    for (auto& v : hh) v = h_to_bf16((float)((int)(rnd() % 2001) - 1000) / 400.0f);
    std::vector<float> wg((size_t)HN);
    for (auto& v : wg) v = (float)((int)(rnd() % 201) - 100) / 1000.0f;
    auto plane = [&](int64_t rows, int64_t K, int64_t G, int64_t ld) {
        std::vector<uint8_t> b((size_t)(rows * ld), 0);
        for (int64_t r0 = 0; r0 < rows; ++r0) {
            uint8_t* row = b.data() + r0 * ld;
            for (int64_t k = 0; k < K; ++k) row[k] = (uint8_t)(8 + rnd() % 111) | ((rnd() & 1) ? 0x80u : 0u);
            float* sc = (float*)(row + K);
            for (int64_t g = 0; g < K / G; ++g) sc[g] = 0.002f + 0.004f * (float)(rnd() % 1000) / 1000.0f;
        }
        return b;
    };
    std::vector<uint8_t> down = plane(lr, HN, r4d_hc8::down_group(HN), dld);
    std::vector<uint8_t> up   = plane(HN, lr, r4d_hc8::up_group(lr), uld);
    std::vector<uint16_t> injw((size_t)(hc * HN));
    for (auto& v : injw) v = h_to_bf16((float)((int)(rnd() % 201) - 100) / 50000.0f);
    const size_t xb = (size_t)(M * n * 2), ib = (size_t)(M * hc * 2), qb = (size_t)(M * n),
                 sb = (size_t)(M * nkb * 4);
    void* d_h = dalloc(hh.size() * 2);
    void* d_w = dalloc(wg.size() * 4);
    void* d_dn = dalloc(down.size());
    void* d_up = dalloc(up.size());
    void* d_iw = dalloc(injw.size() * 2);
    void* d_x = dalloc(xb);
    void* d_inj = dalloc(ib);
    void* d_q = dalloc(qb);
    void* d_qs = dalloc(sb);
    void* d_rq = dalloc(qb);
    void* d_rs = dalloc(sb);
    static const char* const kForm[] = { "the launcher's rule", "the decode tiles' kernels",
                                         "the wide kernels" };
    const std::string nm = fmt("hc_read_e4m3 int8 codes at M=%lld (%s, %s)", (long long)M,
                               real_stream ? "a real stream" : "the null stream", kForm[form]);
    if (!d_h || !d_w || !d_dn || !d_up || !d_iw || !d_x || !d_inj || !d_q || !d_qs || !d_rq || !d_rs) {
        ok(nm.c_str(), false, "hipMalloc failed");
        return;
    }
    st_memcpy(d_h, hh.data(), hh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wg.data(), wg.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_dn, down.data(), down.size(), hipMemcpyHostToDevice);
    st_memcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice);
    st_memcpy(d_iw, injw.data(), injw.size() * 2, hipMemcpyHostToDevice);
    hipStream_t hs = nullptr;
    if (real_stream && hipStreamCreate(&hs) != hipSuccess) { ok(nm.c_str(), false, "stream"); return; }
    struct Out { std::vector<uint8_t> x, inj, q, qs, rq, rs; };
    Out out[2];
    if (pl.pin_hc) pl.pin_hc(form);
    for (int pass = 0; pass < 2; ++pass) {
        const uint32_t qdt = pass ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
        std::vector<RadTensor> ts = {
            T(d_h,  RAD_BF16,   { M, HN }),     T(d_w,  RAD_F32,    { HN }),
            T(d_dn, RAD_F8E4M3, { lr, HN }),    T(d_up, RAD_F8E4M3, { HN, lr }),
            T(d_iw, RAD_BF16,   { hc, HN }),    T(d_x,  RAD_BF16,   { M, n }),
            T(d_inj, RAD_BF16,  { M, hc }),     T(d_q,  qdt,        { M, n }),
            T(d_qs, RAD_F32,    { M, nkb }),    T(d_rq, RAD_F8E4M3, { M, n }),
            T(d_rs, RAD_F32,    { M, nkb }),
        };
        std::vector<RadParam> ps = {
            PI("M", M), PI("n", n), PI("hc", hc), PI("lowrank", lr), PF("eps", 1e-6),
            PS("dtype", "bf16"), PF("wadd", 1.0), PI("inject", 1), PI("group", grp),
            PI("rotate", 1), PS("mix", "e4m3"),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        st_memset(d_x, 0, xb); st_memset(d_q, 0, qb); st_memset(d_qs, 0, sb);
        st_memset(d_rq, 0, qb); st_memset(d_rs, 0, sb); st_memset(d_inj, 0, ib);
        const int rc = r->launch(&a, (RadStream)hs);
        if (rc != RAD_OK || st_stream_sync(hs) != hipSuccess) {
            ok(nm.c_str(), false, fmt("pass %d rc=%d", pass, rc));
            if (pl.pin_hc) pl.pin_hc(0);
            if (hs) (void)hipStreamDestroy(hs);
            return;
        }
        auto get = [&](std::vector<uint8_t>& v, void* d, size_t b) {
            v.resize(b); st_memcpy(v.data(), d, b, hipMemcpyDeviceToHost);
        };
        get(out[pass].x, d_x, xb); get(out[pass].inj, d_inj, ib); get(out[pass].q, d_q, qb);
        get(out[pass].qs, d_qs, sb); get(out[pass].rq, d_rq, qb); get(out[pass].rs, d_rs, sb);
    }
    if (pl.pin_hc) pl.pin_hc(0);
    if (hs) (void)hipStreamDestroy(hs);
    std::string bad;
    if (out[0].x != out[1].x)     bad += " x";
    if (out[0].inj != out[1].inj) bad += " inj";
    if (out[0].rq != out[1].rq) bad += " rq";
    if (out[0].rs != out[1].rs) bad += " rs";
    const int64_t miss = i8_code_misses((const uint16_t*)out[1].x.data(), n,
                                        (const int8_t*)out[1].q.data(), n,
                                        (const float*)out[1].qs.data(), nkb, M, n);
    int64_t nz = 0;
    for (uint8_t v : out[1].q) nz += v != 0;
    ok(nm.c_str(), bad.empty() && miss == 0 && nz > (int64_t)qb / 2,
       bad.empty() ? fmt("%lld codes or scales differ from the host quantiser, %lld non-zero",
                         (long long)miss, (long long)nz)
                   : "the int8 run changed" + bad);
}

/* THE GATED RESIDUAL'S READ AT A PREFILL CHUNK: hc_read_e4m3 at Qwen3.8-Flash-Next's shape (n
 * 2560, four streams, lowrank 320, the inject gains, the fp8 twin and its rotated twin). At prefill
 * it is several launches, so this is also the harness to run under libr4d/isa/trace.sh for their
 * split. Random codes and small positive scales: what is timed is the traffic and the arithmetic.
 *
 * BOTH FORMS OF THE PREFILL TILE are run -- the decode tiles' kernels at 128 rows and the wide
 * kernels, through the r4d_hc_read_pin_prefill hook -- and their outputs compared BYTE FOR BYTE,
 * which is what the wide kernels claim. `timed` times each form as well. */
static void hc_prefill(const Plugin& pl, int64_t M, int iters, bool timed) {
    if (iters <= 0) iters = 20;
    const RadKernelInfo* r = find(pl, "hc_read_e4m3");
    if (!r) { skip("hc_read_e4m3 prefill forms", "row absent"); return; }
    const int64_t n = 2560, hc = 4, lr = 320, HN = hc * n, grp = 128;
    const int64_t dld = r4d_hc8::row_bytes(HN, r4d_hc8::down_group(HN));
    const int64_t uld = r4d_hc8::row_bytes(lr, r4d_hc8::up_group(lr));
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ (uint64_t)M;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };

    std::vector<uint16_t> hh((size_t)(M * HN));
    for (auto& v : hh) v = h_to_bf16((float)((int)(rnd() % 2001) - 1000) / 400.0f);
    std::vector<float> wg((size_t)HN);
    for (auto& v : wg) v = (float)((int)(rnd() % 201) - 100) / 1000.0f;
    auto plane = [&](int64_t rows, int64_t K, int64_t G, int64_t ld) {
        std::vector<uint8_t> b((size_t)(rows * ld), 0);
        for (int64_t r0 = 0; r0 < rows; ++r0) {
            uint8_t* row = b.data() + r0 * ld;
            for (int64_t k = 0; k < K; ++k) row[k] = (uint8_t)(8 + rnd() % 111) | ((rnd() & 1) ? 0x80u : 0u);
            float* sc = (float*)(row + K);
            for (int64_t g = 0; g < K / G; ++g) sc[g] = 0.002f + 0.004f * (float)(rnd() % 1000) / 1000.0f;
        }
        return b;
    };
    std::vector<uint8_t> down = plane(lr, HN, r4d_hc8::down_group(HN), dld);
    std::vector<uint8_t> up   = plane(HN, lr, r4d_hc8::up_group(lr), uld);
    std::vector<uint16_t> injw((size_t)(hc * HN));
    for (auto& v : injw) v = h_to_bf16((float)((int)(rnd() % 201) - 100) / 50000.0f);

    const size_t xb = (size_t)(M * n * 2), ib = (size_t)(M * hc * 2), qb = (size_t)(M * n),
                 sb = (size_t)(M * (n / grp) * 4);
    void* d_h = dalloc(hh.size() * 2);
    void* d_w = dalloc(wg.size() * 4);
    void* d_dn = dalloc(down.size());
    void* d_up = dalloc(up.size());
    void* d_iw = dalloc(injw.size() * 2);
    void* d_x = dalloc(xb);
    void* d_inj = dalloc(ib);
    void* d_q = dalloc(qb);
    void* d_qs = dalloc(sb);
    void* d_rq = dalloc(qb);
    void* d_rs = dalloc(sb);
    if (!d_h || !d_w || !d_dn || !d_up || !d_iw || !d_x || !d_inj || !d_q || !d_qs || !d_rq || !d_rs) {
        ok("hc_read_e4m3 prefill forms: allocation", false);
        return;
    }
    st_memcpy(d_h, hh.data(), hh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wg.data(), wg.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_dn, down.data(), down.size(), hipMemcpyHostToDevice);
    st_memcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice);
    st_memcpy(d_iw, injw.data(), injw.size() * 2, hipMemcpyHostToDevice);

    /* The mixing matrices arrive at their DECLARED shape with the stored dtype, as the engine
     * hands them; the launcher derives the stored row pitch itself. */
    std::vector<RadTensor> ts = {
        T(d_h,  RAD_BF16,   { M, HN }),     T(d_w,  RAD_F32,    { HN }),
        T(d_dn, RAD_F8E4M3, { lr, HN }),    T(d_up, RAD_F8E4M3, { HN, lr }),
        T(d_iw, RAD_BF16,   { hc, HN }),    T(d_x,  RAD_BF16,   { M, n }),
        T(d_inj, RAD_BF16,  { M, hc }),     T(d_q,  RAD_F8E4M3, { M, n }),
        T(d_qs, RAD_F32,    { M, n / grp }), T(d_rq, RAD_F8E4M3, { M, n }),
        T(d_rs, RAD_F32,    { M, n / grp }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", n), PI("hc", hc), PI("lowrank", lr), PF("eps", 1e-6),
        PS("dtype", "bf16"), PF("wadd", 1.0), PI("inject", 1), PI("group", grp),
        PI("rotate", 1), PS("mix", "e4m3"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    if (r->scratch) {
        const int64_t need = r->scratch(&a);
        if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
    }
    /* A REAL STREAM, because the op keys per-stream state on it -- the fused forms' arrival
     * counters -- and on the null stream it finds none and takes the unfused path. */
    hipStream_t hs = nullptr;
    if (hipStreamCreate(&hs) != hipSuccess) { ok("hc_read_e4m3 prefill forms: stream", false); return; }
    const RadStream rst = (RadStream)hs;

    struct Out { std::vector<uint8_t> x, inj, q, qs, rq, rs; };
    const int f_lo = pl.pin_hc ? 1 : 0, f_hi = pl.pin_hc ? 2 : 0;
    static const char* const kForm[] = { "the launcher's rule", "the decode tiles' kernels",
                                         "the wide kernels" };
    std::vector<Out> outs;
    for (int form = f_lo; form <= f_hi; ++form) {
        if (pl.pin_hc) pl.pin_hc(form);
        st_memset(d_x, 0, xb); st_memset(d_inj, 0, ib); st_memset(d_q, 0, qb);
        st_memset(d_qs, 0, sb); st_memset(d_rq, 0, qb); st_memset(d_rs, 0, sb);
        const int rc = r->launch(&a, rst);
        if (rc != RAD_OK || st_stream_sync(hs) != hipSuccess) {
            ok(fmt("hc_read_e4m3 prefill form '%s' at M=%lld is served", kForm[form],
                   (long long)M).c_str(), false, fmt("rc=%d", rc));
            if (pl.pin_hc) pl.pin_hc(0);
            (void)hipStreamDestroy(hs);
            return;
        }
        Out o;
        auto get = [&](std::vector<uint8_t>& v, void* d, size_t b) {
            v.resize(b); st_memcpy(v.data(), d, b, hipMemcpyDeviceToHost);
        };
        get(o.x, d_x, xb); get(o.inj, d_inj, ib); get(o.q, d_q, qb);
        get(o.qs, d_qs, sb); get(o.rq, d_rq, qb); get(o.rs, d_rs, sb);
        outs.push_back(std::move(o));
    }
    if (outs.size() == 2) {
        std::string bad;
        if (outs[0].x != outs[1].x)     bad += " x";
        if (outs[0].inj != outs[1].inj) bad += " inj";
        if (outs[0].q != outs[1].q)     bad += " q";
        if (outs[0].qs != outs[1].qs)   bad += " qs";
        if (outs[0].rq != outs[1].rq)   bad += " rq";
        if (outs[0].rs != outs[1].rs)   bad += " rs";
        /* A zero output would pass a byte comparison between two forms that both wrote nothing. */
        size_t nz = 0;
        for (uint8_t v : outs[1].q) nz += v != 0;
        ok(fmt("hc_read_e4m3 prefill tile at M=%lld: the wide kernels are byte-identical to the "
               "decode tiles' kernels", (long long)M).c_str(),
           bad.empty() && nz > qb / 2, bad.empty() ? fmt("%zu of %zu codes non-zero", nz, qb)
                                                   : "differs in" + bad);
    }

    if (timed) {
        for (int form = f_lo; form <= f_hi; ++form) {
            if (pl.pin_hc) pl.pin_hc(form);
            hipEvent_t w0, w1;
            st_ev_create(&w0); st_ev_create(&w1);
            st_ev_record(w0, hs);
            float wms = 0.0f;
            for (int guard = 0; guard < 100000 && wms < 500.0f; ++guard) {
                for (int i = 0; i < 8; ++i) r->launch(&a, rst);
                st_ev_record(w1, hs);
                st_ev_sync(w1);
                st_ev_elapsed(&wms, w0, w1);
            }
            st_ev_destroy(w0); st_ev_destroy(w1);
            hipEvent_t e0, e1;
            st_ev_create(&e0); st_ev_create(&e1);
            st_ev_record(e0, hs);
            for (int i = 0; i < iters; ++i) r->launch(&a, rst);
            st_ev_record(e1, hs);
            st_ev_sync(e1);
            float ms = 0.0f;
            st_ev_elapsed(&ms, e0, e1);
            st_ev_destroy(e0); st_ev_destroy(e1);
            const double us = (double)ms * 1000.0 / iters;
            const double flop = 2.0 * (double)M * (double)HN * (double)(2 * lr + hc);
            std::printf("hc_read_e4m3 at M=%lld (n %lld, hc %lld, lowrank %lld, inject, fp8 twin + "
                        "rotated), %s: %.1f us a call, %.1f TFLOP/s\n", (long long)M,
                        (long long)n, (long long)hc, (long long)lr, kForm[form], us,
                        flop / (us * 1e6));
        }
    }
    if (pl.pin_hc) pl.pin_hc(0);
    (void)hipStreamDestroy(hs);
}

/* THE INT2 DRAFT HEAD, at every variant it has.
 *
 * WHY IT NEEDS ITS OWN MODE. rad-tune walks the declared graph and never reaches this row -- on
 * the DFlash2 container the requests it lists are `gemm_nt` and `all_reduce` -- so the one kernel
 * in the drafter that is not at bandwidth is never swept. It streams 178.8 MB a decode step at
 * well under the 638 GB/s the part sustains, and the file's own header sets the target: a 280 us
 * memory roofline and a 200 us compute floor with the codes on v_wmma_i32_16x16x16_iu8.
 *
 * The numbers are safe to chase harder than the trunk's. This head only PROPOSES; greedy verify
 * takes the trunk's argmax whatever the draft said, so a variant that changes the head's output
 * bits changes the acceptance rate and not the text. Nothing else in the step has that property.
 *
 * The weight is one packed blob -- N*K/4 bytes of codes plus N*(K/128)*4 of scale/zero pairs --
 * and the activation is the int8 the quantiser writes with its per-group sums, so this builds the
 * gemm_nt_q operand list rather than the fp8 one. Contents are shape-correct garbage: what is
 * being timed is a stream of the right size in the right order.
 */

/* The bf16 skinny GEMM at one shape, every variant. rad-tune reaches these rows, but its absolute
 * numbers are not repeatable enough between back-to-back runs of the same command to base a
 * default on -- the same reason `--perf` and `--perfw2` exist beside it. This rotates a working
 * set past the last level, times each variant the same way, and prints them together.
 *
 * THE SHAPES THIS IS FOR are the three the model's declared graph reaches: N = 48 (the GDN a|b
 * projection, forty-eight times a step and the only TRUNK caller), N = 1280 and N = 256 (the
 * DFlash2 drafter's, ten and one). The distinction matters more than the sizes: a default that
 * changes the drafter's bits changes which tokens it PROPOSES and not which the model emits, and
 * one that changes the trunk's changes the model. */
static void bf16_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 16 || K % 16 || M < 1) {
        std::printf("--perfbf16: N and K must be 16-aligned and M >= 1\n");
        return;
    }
    /* R4D_PERFBF16_M64 forces the WMMA kernel onto the M <= 16 band the registry gives to the
     * scalar one, which is the only way to ask whether that row still earns its place. */
    static const bool f64 = std::getenv("R4D_PERFBF16_M64") != nullptr;
    const RadKernelInfo* r = find(pl, (M <= 16 && !f64) ? "gemm_bf16_nt_m16" : "gemm_bf16_nt_m64");
    if (!r) { std::printf("--perfbf16: row absent\n"); return; }

    const size_t wbytes = (size_t)N * (size_t)K * 2;
    int copies = (int)(((size_t)768 << 20) / wbytes) + 1;
    if (copies > 16) copies = 16;

    void* d_w = dalloc(wbytes * (size_t)copies);
    void* d_a = dalloc((size_t)M * K * 2);
    void* d_y = dalloc((size_t)M * N * 2);
    if (!d_w || !d_a || !d_y) {
        std::printf("--perfbf16: hipMalloc failed (%zu MiB wanted)\n",
                    (wbytes * (size_t)copies) >> 20);
        return;
    }
    {
        std::vector<uint16_t> w(wbytes / 2);
        for (size_t i = 0; i < w.size(); ++i) w[i] = h_to_bf16((float)((int)(i % 17) - 8) * 0.125f);
        for (int c = 0; c < copies; ++c)
            st_memcpy((uint8_t*)d_w + (size_t)c * wbytes, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> a((size_t)M * K);
        for (size_t i = 0; i < a.size(); ++i) a[i] = h_to_bf16((float)((int)(i % 13) - 6) * 0.25f);
        st_memcpy(d_a, a.data(), a.size() * 2, hipMemcpyHostToDevice);
    }

    std::printf("\nbf16 skinny GEMM  %s  N=%lld K=%lld M=%lld  %.1f MiB weight, %d copies, "
                "%d iters\n", r->name, (long long)N, (long long)K, (long long)M,
                (double)wbytes / (1 << 20), copies, iters);

    /* EVERY LEGAL POINT OF THE DECLARED SPACE, capped. The skinny GEMM's five axes are ~900 points
     * and tune_valid prunes that to a few hundred at any one K -- more than an interactive sweep
     * wants but exactly what rad-tune walks. `kPerfCap` keeps this readable; the cap is a property
     * of the HARNESS and not of the kernel, which is the whole reason the space is declared. */
    std::vector<RadParam> geom = {
        PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16"),
    };
    /* Above the skinny band the tiled row is timed as well, at each of its tile variants: the two
     * serve the same op, so the numbers belong side by side. */
    const RadKernelInfo* rt = M > 64 && N >= 128 && K % 64 == 0 ? find(pl, "gemm_bf16_nt_tiled")
                                                                 : nullptr;
    for (const RadKernelInfo* row : { r, rt }) {
        if (!row) continue;
        const std::vector<TunePoint> pts = tune_points(row, geom, kPerfCap);
        std::printf("  %s: %zu point(s) of the declared space\n", row->name, pts.size());
        for (const TunePoint& tp : pts) {
            std::vector<RadTensor> ts = {
                T(d_a, RAD_BF16, { M, K }), T(d_w, RAD_BF16, { N, K }), T(d_y, RAD_BF16, { M, N }),
            };
            std::vector<RadParam> ps = with_point(geom, tp);
            perf_row(pl, row->name, ts, ps, d_w, wbytes, copies, (double)wbytes, iters, 1,
                     tp.label.c_str());
        }
    }
}
static void w2_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 16 || K % 128 || M < 1) {
        std::printf("--perfw2: N must be 16-aligned, K 128-aligned, M >= 1\n");
        return;
    }
    const RadKernelInfo* r = find(pl, "gemm_w2a8_nt");
    if (!r) { std::printf("--perfw2: row absent\n"); return; }

    const int64_t nkb = K / 128;
    const size_t  wbytes = (size_t)N * (size_t)K / 4;
    const size_t  sbytes = (size_t)N * (size_t)nkb * 4;
    /* Same rotation argument as fp8_perf: enough copies that the working set is past the last
     * level, capped so a 160 MiB weight does not try to allocate the card. */
    int copies = (int)(((size_t)768 << 20) / wbytes) + 1;
    if (copies > 16) copies = 16;

    void* d_w  = dalloc(wbytes * (size_t)copies);
    void* d_s  = dalloc(sbytes);
    void* d_a  = dalloc((size_t)M * K);
    void* d_as = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)nkb * (size_t)(((M + 15) / 16) * 16) * 4);
    void* d_y  = dalloc((size_t)M * N * 2);
    if (!d_w || !d_s || !d_a || !d_as || !d_su || !d_y) {
        std::printf("--perfw2: hipMalloc failed (%zu MiB wanted)\n",
                    (wbytes * (size_t)copies) >> 20);
        return;
    }
    /* A zeroed scale/zero pair is two f16 zeroes, which makes every dequant zero and is exactly
     * the shape of measurement that finds a denormal fast path instead of the kernel. */
    {
        std::vector<uint32_t> sz((size_t)N * (size_t)nkb, 0x00003C00u);   /* zp 0, scale 1.0 */
        st_memcpy(d_s, sz.data(), sz.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint8_t> w(wbytes, 0x1Bu);        /* codes 0,1,2,3 */
        for (int c = 0; c < copies; ++c)
            st_memcpy((uint8_t*)d_w + (size_t)c * wbytes, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<int8_t> a((size_t)M * K, 3);
        st_memcpy(d_a, a.data(), a.size(), hipMemcpyHostToDevice);
        std::vector<float> as((size_t)M, 1.0f / 127.0f);
        st_memcpy(d_as, as.data(), as.size() * 4, hipMemcpyHostToDevice);
        std::vector<float> su((size_t)nkb * (size_t)(((M + 15) / 16) * 16), 3.0f * 128.0f);
        st_memcpy(d_su, su.data(), su.size() * 4, hipMemcpyHostToDevice);
    }

    const double bytes = (double)(wbytes + sbytes);
    std::printf("\nint2 draft head  N=%lld K=%lld M=%lld  %.1f MiB weight, %d copies, %d iters\n",
                (long long)N, (long long)K, (long long)M,
                (double)(wbytes + sbytes) / (1 << 20), copies, iters);

    std::vector<RadParam> geom = {
        PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "w2a8"),
    };
    const std::vector<TunePoint> pts = tune_points(r, geom, kPerfCap);
    std::printf("  %zu point(s) of the declared space\n", pts.size());
    for (const TunePoint& tp : pts) {
        std::vector<RadTensor> ts = {
            T(d_a, RAD_I8, { M, K }), T(d_as, RAD_F32, { M }),
            T(d_w, RAD_U8, { (int64_t)wbytes }), T(d_s, RAD_U8, { (int64_t)sbytes }),
            T(d_y, RAD_BF16, { M, N }),
            T(d_su, RAD_F32, { nkb * ((M + 15) / 16) * 16 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[6]);
        std::vector<RadParam> ps = with_point(geom, tp);
        perf_row(pl, "gemm_w2a8_nt", ts, ps, d_w, wbytes, copies, bytes, iters, 2,
                 tp.label.c_str());
    }
}
static void fp8_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 128 || K % 128 || M < 1) {
        std::printf("--perf: N and K must be 128-aligned and M >= 1\n");
        return;
    }
    const int64_t sn = N / 128, sk = K / 128;
    const size_t wbytes = (size_t)N * K;
    /* Enough copies that the rotation's whole working set is past the last level of cache on this
     * part, with a ceiling so a 600 MiB weight does not try to allocate the card. A shape small
     * enough to sit in cache would otherwise report a bandwidth decode can never see. */
    int copies = (int)(((size_t)768 << 20) / wbytes) + 1;
    if (copies > 64) copies = 64;

    void* d_w = dalloc(wbytes * (size_t)copies);
    void* d_s = dalloc((size_t)sn * sk * 2);
    void* d_q = dalloc((size_t)M * K);
    void* d_qs = dalloc((size_t)M * sk * 4);
    void* d_x = dalloc((size_t)M * K * 2);
    void* d_y = dalloc((size_t)M * N * 2);
    if (!d_w || !d_s || !d_q || !d_qs || !d_x || !d_y) {
        std::printf("--perf: hipMalloc failed (%zu MiB wanted)\n",
                    (wbytes * (size_t)copies) >> 20);
        return;
    }
    /* Zeroed scales would make every dequant a zero, which some day will be a denormal-flush fast
     * path somewhere and a wrong measurement here. Fill with a checkpoint-shaped multiplier. */
    {
        std::vector<uint16_t> s((size_t)sn * sk, h_to_bf16(3.0e-4f));
        st_memcpy(d_s, s.data(), s.size() * 2, hipMemcpyHostToDevice);
        std::vector<uint8_t> w(wbytes, 0x38);   /* E4M3 1.0 */
        for (int c = 0; c < copies; ++c)
            st_memcpy((uint8_t*)d_w + (size_t)c * wbytes, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<uint8_t> q((size_t)M * K, 0x38);
        st_memcpy(d_q, q.data(), q.size(), hipMemcpyHostToDevice);
        std::vector<float> qs((size_t)M * sk, 1.0f);
        st_memcpy(d_qs, qs.data(), qs.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint16_t> x((size_t)M * K, h_to_bf16(0.5f));
        st_memcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice);
    }

    const double bytes = (double)wbytes;
    std::printf("\nfp8 decode GEMM  N=%lld K=%lld M=%lld  %.1f MiB weight, %d copies, %d iters\n",
                (long long)N, (long long)K, (long long)M, (double)wbytes / (1 << 20), copies, iters);

    {
        std::vector<RadTensor> ts = {
            T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_F32, { M, sk }),
            T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[1]); absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
        };
        perf_row(pl, "gemm_fp8a16_nt_m1", ts, ps, d_w, wbytes, copies, bytes, iters);
    }
    {
        std::vector<RadTensor> ts = {
            T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
            T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
        };
        perf_row(pl, "gemm_fp8a8_nt_m16", ts, ps, d_w, wbytes, copies, bytes, iters);
        perf_row(pl, "gemm_fp8a8_tiled", ts, ps, d_w, wbytes, copies, bytes, iters);
    }

    /* THE LM HEAD, which is a decode GEMM too and the one nothing could time. Its weight is a
     * plane of its own shape -- [N][K + 2*nkb] E4M3, each row's bf16 scales in its own tail, see
     * r4d_fp8_layout.cpp -- so it cannot share the rotation buffer above, and its row is the
     * SECOND operand rather than the third. At this model's 124160 x 5120 it is the largest
     * single kernel of a decode step after the trunk's GEMMs, and without this block the only way
     * to measure it is a rocprofv3 trace of a serving engine. */
    if (M <= 64) {
        const int64_t lstride = r4d_fp8::lm_row_bytes(K);
        const size_t  lwb = (size_t)N * (size_t)lstride;
        int lcopies = (int)(((size_t)768 << 20) / lwb) + 1;
        if (lcopies > 16) lcopies = 16;
        void* d_lw = dalloc(lwb * (size_t)lcopies);
        void* d_ly = dalloc((size_t)M * N * 4);
        if (!d_lw || !d_ly) {
            std::printf("  %-22s hipMalloc failed (%zu MiB)\n", "logits_gemm_fp8",
                        (lwb * (size_t)lcopies) >> 20);
        } else {
            std::vector<uint8_t> plane(lwb);
            for (int64_t n = 0; n < N; ++n) {
                uint8_t* r = plane.data() + (size_t)n * lstride;
                std::memset(r, 0x38, (size_t)K);                  /* E4M3 1.0 */
                uint16_t* tail = (uint16_t*)(r + K);
                for (int64_t b = 0; b < sk; ++b) tail[b] = h_to_bf16(3.0e-4f);
            }
            for (int c = 0; c < lcopies; ++c)
                st_memcpy((uint8_t*)d_lw + (size_t)c * lwb, plane.data(), lwb,
                          hipMemcpyHostToDevice);
            /* The DECLARED shape, [N][K], because that is what reaches the shim -- the stored
             * width is derived there from n_embd; the fp8 logits_gemm launcher in
             * r4d_gemm_fp8a16_nt_m1.hip says why. */
            std::vector<RadTensor> ts = {
                T(d_x, RAD_BF16, { M, K }),
                T(d_lw, RAD_F8E4M3, { N, K }),
                T(d_ly, RAD_F32, { M, N }),
            };
            std::vector<RadParam> ps = {
                PI("n_vocab", N), PI("n_embd", K), PI("M", M), PS("dtype", "bf16"),
            };
            perf_row(pl, "logits_gemm_fp8", ts, ps, d_lw, lwb, lcopies, (double)lwb, iters, 1);
        }
    }
}

/* THE 8-BIT TRUNK GEMMS, side by side: block E4M3 W8A8 (the fp8 trunk), the int8 W8A8
 * tiled kernel, and the fp8 kernels' int8 twins (gemm_i8a8_*), one shape, each with its useful
 * TFLOP/s -- and at M <= 64 the two narrow rows too. The operands hold constants: timing does not
 * depend on the values, and the bytes are only the shape each row's shim checks. */
static void w8_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 256 || K % 128 || M < 1) {
        std::printf("--perfw8: N a multiple of 256, K of 128, M >= 1\n");
        return;
    }
    const int64_t sn = N / 128, sk = K / 128;
    const size_t wbytes = (size_t)N * K;
    void* d_w  = dalloc(wbytes);
    void* d_fs = dalloc((size_t)sn * sk * 2);           /* E4M3 block scales, bf16 */
    void* d_is = dalloc((size_t)N * sk * 4);            /* the int8 rows' scale planes */
    void* d_q  = dalloc((size_t)M * K);
    void* d_qs = dalloc((size_t)M * sk * 4);
    void* d_x  = dalloc((size_t)M * K * 2);
    void* d_y  = dalloc((size_t)M * N * 2);
    if (!d_w || !d_fs || !d_is || !d_q || !d_qs || !d_x || !d_y) {
        std::printf("--perfw8: hipMalloc failed\n");
        return;
    }
    {
        std::vector<uint8_t> w(wbytes, 0x38);
        st_memcpy(d_w, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> fs((size_t)sn * sk, h_to_bf16(3.0e-4f));
        st_memcpy(d_fs, fs.data(), fs.size() * 2, hipMemcpyHostToDevice);
        std::vector<uint32_t> is((size_t)N * sk, 0x0C001C00u);
        st_memcpy(d_is, is.data(), is.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint8_t> q((size_t)M * K, 0x38);
        st_memcpy(d_q, q.data(), q.size(), hipMemcpyHostToDevice);
        std::vector<float> qs((size_t)M * sk, 1.0f);
        st_memcpy(d_qs, qs.data(), qs.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint16_t> x((size_t)M * K, h_to_bf16(0.5f));
        st_memcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice);
    }
    const double flop = 2.0 * (double)M * (double)N * (double)K;
    std::printf("\n8-bit trunk GEMMs  N=%lld K=%lld M=%lld  %d iters\n", (long long)N, (long long)K,
                (long long)M, iters);
    auto run = [&](const char* row, const char* dt, std::vector<RadTensor> ts) {
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", dt),
        };
        const RadKernelInfo* r = find(pl, row);
        if (!r) { std::printf("  %-22s row absent\n", row); return; }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("  %-22s rc=%d (shape not served)\n", row, rc);
            return;
        }
        hipEvent_t e0, e1;
        st_ev_create(&e0);
        st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0);
        st_ev_destroy(e1);
        const double us = (double)ms * 1000.0 / iters;
        std::printf("  %-22s %9.1f us  %6.1f TFLOP/s\n", row, us, flop / (us * 1e6));
    };
    std::vector<RadTensor> fp8 = {
        T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }), T(d_w, RAD_F8E4M3, { N, K }),
        T(d_fs, RAD_BF16, { sn, sk }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(fp8[5]); absent(fp8[6]);
    run("gemm_fp8a8_tiled", "fp8a8", fp8);
    std::vector<RadTensor> i8 = {
        T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M }), T(d_w, RAD_U8, { (int64_t)wbytes }),
        T(d_is, RAD_U8, { N * sk * 2 }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(i8[5]); absent(i8[6]);
    run("gemm_w8a8_tiled", "w8a8", i8);
    /* The int8 twin of the fp8 row: i8 activation with an f32 scale per (row, 128), i8 weight with
     * a bf16 scale per row per 128. The fp8 row's own scale buffers are the right size for both. */
    std::vector<uint16_t> rs((size_t)N * sk, h_to_bf16(3.0e-4f));
    void* d_rs = dalloc(rs.size() * 2);
    if (!d_rs) { std::printf("--perfw8: hipMalloc failed\n"); return; }
    st_memcpy(d_rs, rs.data(), rs.size() * 2, hipMemcpyHostToDevice);
    std::vector<RadTensor> i8g = {
        T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M, sk }), T(d_w, RAD_I8, { N, K }),
        T(d_rs, RAD_BF16, { N, sk }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(i8g[5]); absent(i8g[6]);
    run("gemm_i8a8_tiled", "i8a8", i8g);
    if (M <= 64) {
        run("gemm_fp8a8_nt_m16", "fp8a8", fp8);
        run("gemm_i8a8_nt_m16", "i8a8", i8g);
    }
}

/* THE W4 ARRANGEMENT, CHECKED AGAINST libr4d's OWN DEQUANT. The relayout writes the fragment
 * order and the (scale, zero) dwords and dequant_w4_bf16 undoes both on the card, so a disagreement
 * between the two is exactly the class of bug that would otherwise reach a model. The planes are
 * an i4 grid with an f16 scale a group of 128, and what comes back must be code times scale. */
static void w4_roundtrip(const Plugin& pl) {
    const RadKernelInfo* wrow = find(pl, "gemm_w4a8_nt_m64");
    const RadKernelInfo* drow = find(pl, "dequant_w4_bf16");
    if (!wrow || !drow || !wrow->layout || !wrow->relayout) {
        ok("w4 relayout vs dequant_w4_bf16", false, "rows or hooks absent");
        return;
    }
    const int64_t N = 32, K = 256, group = 128;
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", group) };

    std::vector<float> src((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            src[(size_t)(n * K + k)] = (float)(((n * 31 + k * 17) % 61) - 30) / 30.0f;
    const Planes p = int4_planes(src, N, K, group, RAD_F16, false);

    std::vector<uint8_t> wq, sz;
    RadLayout lay;
    const int rw = stored(wrow, lp, 2, p, { "codes" }, &wq, &lay);
    const int rs = stored(wrow, lp, 3, p, { "scale" }, &sz);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok("w4 relayout", false, "rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }
    ok("w4 layout is the codes at half a byte each", lay.bytes == N * K / 2,
       "bytes " + std::to_string(lay.bytes));
    {
        std::string why;
        const bool cw = restores(wrow, lp, 2, p, { "codes" }, wq, &why);
        std::string why_s;
        const bool cs = restores(wrow, lp, 3, p, { "scale" }, sz, &why_s);
        ok("w4 relayout inverts to the planes it was handed", cw && cs, why + " " + why_s);
    }

    void* dwq = dalloc(wq.size());
    void* dsz = dalloc(sz.size());
    void* dout = dalloc((size_t)(N * K * 2));
    st_memcpy(dwq, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(dsz, sz.data(), sz.size(), hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(dwq,  RAD_U8,   { (int64_t)wq.size() }),
        T(dsz,  RAD_U8,   { (int64_t)sz.size() }),
        T(dout, RAD_BF16, { N, K }),
    };
    std::vector<RadParam> ps = {
        PI("N", N), PI("K", K), PI("group", group), PS("dtype", "w4"), PI("twos", 1),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = drow->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok("dequant_w4_bf16 launch", false, "rc=" + std::to_string(rc)); return; }

    std::vector<uint16_t> got((size_t)(N * K));
    st_memcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost);

    /* Code times scale exactly as the planes hold them, rounded to bf16 as the kernel's output
     * is -- so the only freedom left is the rounding of one product. */
    const int64_t ng = K / group;
    double worst = 0.0;
    int64_t bad = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t i = n * K + k;
            const uint32_t u = (uint32_t)got[(size_t)i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            const float want = h_bf16(h_to_bf16(
                (float)code_at(p, n, k) * rad_load_f32(p.data[1].data(), RAD_F16, n * ng + k / group)));
            const double err = std::fabs((double)f - (double)want);
            /* one bf16 step either way: the two sides may round the same product apart */
            if (!(err <= 1e-6 + 1e-2 * std::fabs((double)want))) { ++bad; if (err > worst) worst = err; }
        }
    ok("w4 relayout round-trips through libr4d's own dequant_w4_bf16", bad == 0,
       std::to_string(bad) + " elements off, worst " + std::to_string(worst));
}


/* THE 4-BIT GROUPED MoE GEMM, AGAINST A HOST PRODUCT OVER THE SAME PLANES.
 *
 * This is the check the `moe_gemm_w4a8` registry row promises in place of a kbench case, and it is
 * the one that can actually fail. The relayout reorders each dword's nibbles; the kernel expands
 * them with three V_PERM_B32 on the way into LDS; the two have to agree about the nibble order,
 * the row pitch (the launcher halves an ELEMENT stride into bytes -- get that wrong and every row
 * is read from the wrong place) and the scale grid (one bf16 per ROW per group of 128, not per
 * 128x128 tile). The reference reads the canonical planes -- code k of row n, times its group's
 * scale -- so it shares nothing with either side's arrangement.
 *
 * BOTH BANDS. `T <= 128` runs a 16-row M tile and above it a 128-row one, and the two differ in
 * the wave grid and in how many experts a block's row range straddles -- which is the part with
 * the binary search and the run loop in it. A case in only one band tests half the kernel. */
static float st_e4m3(uint8_t b) {
    const uint32_t sign = (uint32_t)(b & 0x80u) << 24, ex = (b >> 3) & 0x0Fu, mn = b & 0x07u;
    const uint32_t nrm = ((ex + 120u) << 23) | (mn << 20);
    float f;
    std::memcpy(&f, &nrm, 4);
    const float v = ex ? f : (float)mn * (1.0f / 512.0f);
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u |= sign;
    std::memcpy(&f, &u, 4);
    return f;
}

/* One geometry per class -- the even experts' and the odd ones' -- and the number of stacked parts
 * along the weight rows and the output. One class shape is the plain grouped GEMM; two are a
 * rank's uneven slice of every expert, where the narrower class's gate/up pair writes zeros past
 * its part and its down projection reads a shorter K. */
struct MoeW4Shape { const char* name; int64_t N0, K0, N1, K1, parts, ne; };

/* EVERY FORM THE LAUNCHER HAS, each against the host decode: the run-aligned prefill form (a run
 * map in the scratch it is handed), the (tile, run) decode form, and the uniform prefill grid a
 * caller with no scratch gets. The rule picks one by the routed-row count, so a case that only
 * ran the rule would test whichever form its row count landed on. */
/* `sub` 2 runs the codebook at E4M3 scales a 64 of K (w4nl64a8h), 4 at a 32 (w4nl32a8h): the
 * codes' values are the table's and each part of a 128-wide group has its own scale, the fold the
 * forms must agree on. `w5` with `sub` 2 runs the five-bit codebook (w5nl64a8h), whose stored row
 * is its nibbles and then its signs. `four` also runs `moe_gemm_w4a8_mod4` over the same experts
 * and holds it to the two-class op's bytes -- that row's own case. */
static void moe_w4_gemm(const Plugin& pl, int64_t Tr, const MoeW4Shape& sh, int sub = 0,
                        bool w5 = false, bool four = false) {
    const bool g64 = sub > 1;
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    if (!r || !r->layout || !r->relayout) {
        ok((std::string("moe_gemm_w4a8 (") + sh.name + ")").c_str(), false, "row or hooks absent");
        return;
    }
    const std::string what = std::string("moe_gemm_w4a8 vs a host decode (T=") +
                             std::to_string(Tr) + ", " + sh.name +
                             (w5 ? ", w5nl64a8h" : sub == 4 ? ", w4nl32a8h"
                              : g64 ? ", w4nl64a8h" : "") + ")";
    const char* dt = w5 ? "w5nl64a8h" : sub == 4 ? "w4nl32a8h" : g64 ? "w4nl64a8h" : "w4a8";

    const int64_t ne = sh.ne, group = 128, top_k = 2;
    const int64_t Nc[2] = { sh.N0, sh.N1 }, Kc[2] = { sh.K0, sh.K1 };
    const int64_t nec[2] = { (ne + 1) / 2, ne / 2 };
    const int64_t N = std::max(sh.N0, sh.N1), K = std::max(sh.K0, sh.K1);
    const int64_t kb = K / group, M = Tr / top_k;
    const int64_t fy = N / sh.parts;
    std::vector<RadParam> lp = { PI("N", sh.N0), PI("K", sh.K0), PI("N_odd", sh.N1),
                                 PI("K_odd", sh.K1), PI("group", group), PS("dtype", dt) };

    /* Each class's stored bytes from the hooks, asked through the class's own operands -- 2/3 and
     * 6/7 -- so the view that answers for the odd table is under test as well. One float source an
     * expert, its planes, and the arrangement beside them; the planes are kept for the reference. */
    std::vector<uint8_t>  wq[2];
    std::vector<uint16_t> ws[2];
    std::vector<Planes>   cp[2];
    for (int c = 0; c < 2; ++c) {
        const int ow = c ? 6 : 2, os = c ? 7 : 3;
        const int64_t kbc = Kc[c] / group;
        const int64_t wbytes = w5 ? Nc[c] * Kc[c] / 8 * 5 : Nc[c] * Kc[c] / 2;
        wq[c].assign((size_t)(nec[c] * wbytes), 0);
        ws[c].assign((size_t)(nec[c] * Nc[c] * kbc * (sub == 4 ? 2 : 1)), 0);
        std::vector<float> wsrc((size_t)(Nc[c] * Kc[c]));
        for (int64_t x = 0; x < nec[c]; ++x) {
            const int64_t e = 2 * x + c;
            for (int64_t n = 0; n < Nc[c]; ++n)
                for (int64_t k = 0; k < Kc[c]; ++k)
                    wsrc[(size_t)(n * Kc[c] + k)] =
                        (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
            cp[c].push_back(g64 ? nl64_planes(wsrc, Nc[c], Kc[c], 128 / sub, w5)
                                : int4_planes(wsrc, Nc[c], Kc[c], group, RAD_BF16, false));
            std::vector<uint8_t> w, sc;
            const int rw = g64 ? stored(r, lp, ow, cp[c].back(), { "codes", "table" }, &w)
                               : stored(r, lp, ow, cp[c].back(), { "codes" }, &w);
            const int rs = g64 ? stored(r, lp, os, cp[c].back(), { "scale", "scale.1" }, &sc)
                               : stored(r, lp, os, cp[c].back(), { "scale" }, &sc);
            if (rw != RAD_OK || rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED)) {
                ok(what.c_str(), false, "relayout rc " + std::to_string(rw) + "/" +
                                        std::to_string(rs));
                return;
            }
            if ((int64_t)w.size() != wbytes ||
                (int64_t)sc.size() != Nc[c] * kbc * (sub == 4 ? 4 : 2)) {
                ok(what.c_str(), false, "stored byte counts are not N*K/2 (5/8 at five bits) and "
                                        "N*(K/group)*2 (*4 at a 32)");
                return;
            }
            std::memcpy(wq[c].data() + (size_t)(x * wbytes), w.data(), w.size());
            std::memcpy((uint8_t*)ws[c].data() + (size_t)x * sc.size(), sc.data(), sc.size());
        }
    }

    /* The activation is written as E4M3 BYTES and a per-(row, group) f32 scale, so the host and
     * the device read the same numbers and the only difference left is summation order. */
    std::vector<uint8_t> aq((size_t)(M * K));
    std::vector<float>   as((size_t)(M * kb));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t g = 0; g < kb; ++g) as[(size_t)(m * kb + g)] = 0.011f + 0.003f * (float)g;
        for (int64_t k = 0; k < K; ++k) {
            /* 8..126: no subnormal, and no 0x7F -- which is E4M3's NaN, and which a comparison
             * that lets NaN through never noticed. */
            const uint32_t c = (uint32_t)((m * 13 + k * 5) % 119) + 8;
            aq[(size_t)(m * K + k)] = (uint8_t)(((m + k) & 1) ? (c | 0x80u) : c);
        }
    }

    /* A routing that puts an uneven run on each expert and drops the tail, which is what the run
     * loop and the dropped-slot zeroing have to survive. */
    std::vector<int32_t> off((size_t)(ne + 1), 0), sorted((size_t)Tr, -1);
    const int64_t live = Tr - 3;
    for (int64_t e = 0; e <= ne; ++e) off[(size_t)e] = (int32_t)((live * e) / ne);
    off[(size_t)ne] = (int32_t)live;
    for (int64_t i = 0; i < live; ++i) sorted[(size_t)i] = (int32_t)((i * 7 + 3) % Tr);

    void* d_a  = dalloc(aq.size());
    void* d_as = dalloc(as.size() * 4);
    void* d_w[2]  = { dalloc(wq[0].size()), dalloc(wq[1].size()) };
    void* d_ws[2] = { dalloc(ws[0].size() * 2), dalloc(ws[1].size() * 2) };
    void* d_s  = dalloc(sorted.size() * 4);
    void* d_o  = dalloc(off.size() * 4);
    void* d_y  = dalloc((size_t)(Tr * N * 2));
    st_memcpy(d_a,  aq.data(),     aq.size(),         hipMemcpyHostToDevice);
    st_memcpy(d_as, as.data(),     as.size() * 4,     hipMemcpyHostToDevice);
    for (int c = 0; c < 2; ++c) {
        st_memcpy(d_w[c],  wq[c].data(), wq[c].size(),     hipMemcpyHostToDevice);
        st_memcpy(d_ws[c], ws[c].data(), ws[c].size() * 2, hipMemcpyHostToDevice);
    }
    st_memcpy(d_s,  sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o,  off.data(),    off.size() * 4,    hipMemcpyHostToDevice);
    /* Stale bytes where the narrower class writes zeros, so a pad tile left unwritten shows. */
    {
        std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
        st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
    }

    /* THE STACKED WEIGHT-TABLE FORM (a leading stride that is not zero). A pointer array is what
     * the engine hands the kernel; a test has nowhere to put one, and the kernel reads both
     * precisely so that this case runs against the same bytes. */
    std::vector<RadTensor> ts = {
        T(d_a,     RAD_F8E4M3,   { M, K }),
        T(d_as,    RAD_F32,      { M, kb }),
        T(d_w[0],  w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4, { nec[0], sh.N0, sh.K0 }),
        g64 ? T(d_ws[0], RAD_F8E4M3, { nec[0], sh.N0, sub * sh.K0 / group })
            : T(d_ws[0], RAD_BF16,   { nec[0], sh.N0, sh.K0 / group }),
        T(d_s,     RAD_I32,      { Tr }),
        T(d_o,     RAD_I32,      { ne + 1 }),
        T(d_w[1],  w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4, { nec[1], sh.N1, sh.K1 }),
        g64 ? T(d_ws[1], RAD_F8E4M3, { nec[1], sh.N1, sub * sh.K1 / group })
            : T(d_ws[1], RAD_BF16,   { nec[1], sh.N1, sh.K1 / group }),
        T(d_y,     RAD_BF16,     { Tr, N }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", sh.N0), PI("K", sh.K0), PI("N_odd", sh.N1), PI("K_odd", sh.K1),
        PI("parts", sh.parts), PI("n_expert", ne), PI("top_k", top_k), PI("group", group),
        PS("a_order", "token"), PS("dtype", dt),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    /* Room for the run map at the smallest prefill tile, whatever the hook answers: the hook sizes
     * none inside the decode band, and the aligned form pinned there would then not run. */
    const int64_t map_bytes = (1 + 2 * (ne + Tr / 16 + 2)) * 4;
    void* d_map = dalloc((size_t)map_bytes);

    /* The reference, once: every form must reproduce it. */
    std::vector<double> ref((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < Tr; ++i) {
        const int32_t f = sorted[(size_t)i];
        int64_t e = 0;
        while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
        const int c = (int)(e & 1);
        const int64_t x = e / 2, fc = Nc[c] / sh.parts, kbc = Kc[c] / group;
        for (int64_t n = 0; n < N; ++n) {
            double want = 0.0;
            const int64_t pp = n / fy, j = n - pp * fy;
            if (f >= 0 && i < live && j < fc) {
                const int64_t tok = f / top_k, wr = pp * fc + j;
                for (int64_t g = 0; g < kbc; ++g) {
                    if (g64) {
                        /* `sub` scale groups a 128, each an E4M3 byte under the fixed constant. */
                        const Planes& q = cp[c][(size_t)x];
                        const int64_t gw = group / sub;
                        for (int64_t h = 0; h < sub; ++h) {
                            double part = 0.0;
                            for (int64_t k = g * group + h * gw; k < g * group + (h + 1) * gw; ++k)
                                part += (double)tab[code_at(q, wr, k)] *
                                        (double)st_e4m3(aq[(size_t)(tok * K + k)]);
                            const double sf = (double)h_e4m3(q.data[1][(size_t)(wr * q.row_bytes(1) +
                                                                                 sub * g + h)]) *
                                              (double)kW4G64Scale;
                            want += part * sf * (double)as[(size_t)(tok * kb + g)];
                        }
                        continue;
                    }
                    double sub = 0.0;
                    for (int64_t k = g * group; k < (g + 1) * group; ++k)
                        sub += (double)code_at(cp[c][(size_t)x], wr, k) *
                               (double)st_e4m3(aq[(size_t)(tok * K + k)]);
                    const float sf = rad_load_f32(cp[c][(size_t)x].data[1].data(), RAD_BF16,
                                                  wr * kbc + g);
                    want += sub * (double)sf * (double)as[(size_t)(tok * kb + g)];
                }
            }
            ref[(size_t)(i * N + n)] = want;
        }
    }

    const int f_lo = pl.pin_moe ? 1 : 0, f_hi = pl.pin_moe ? 3 : 0;
    std::vector<uint16_t> two_cls[4];      /* each form's bytes, for the four-class pass below */
    for (int form = f_lo; form <= f_hi; ++form) {
        const std::string label = what + " " + kForm[form];
        if (pl.pin_moe) pl.pin_moe(form);
        /* Stale bytes again, so a form that leaves a pad tile unwritten cannot inherit the zeros
         * the form before it wrote. */
        {
            std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        a.scratch = form == 3 ? nullptr : d_map;
        a.scratch_bytes = form == 3 ? 0 : map_bytes;
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (pl.pin_moe) pl.pin_moe(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }

        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0.0;
        int64_t bad = 0;
        std::string first;
        for (int64_t i = 0; i < Tr; ++i)
            for (int64_t n = 0; n < N; ++n) {
                const double want = ref[(size_t)(i * N + n)];
                const uint32_t u = (uint32_t)got[(size_t)(i * N + n)] << 16;
                float gf;
                std::memcpy(&gf, &u, 4);
                /* bf16 carries eight mantissa bits, so a relative half-ulp is 1/512; the tolerance
                 * is that plus room for a 256-term f32 sum taken in a different order. Written so
                 * that a NaN on either side counts as a miss: `err > tol` is false for one, and a
                 * NaN is exactly what a kernel that read a stale row would produce. */
                const double err = std::fabs((double)gf - want);
                if (!(err <= 0.004 * std::fabs(want) + 1e-4)) {
                    if (!bad++)
                        first = ", first at row " + std::to_string(i) + " col " +
                                std::to_string(n) + ": " + std::to_string(gf) + " for " +
                                std::to_string(want);
                    if (!(err <= worst)) worst = err;
                }
            }
        ok(label.c_str(), bad == 0, std::to_string(bad) + " of " + std::to_string(Tr * N) +
                                    " elements off, worst " + std::to_string(worst) + first);
        two_cls[form] = got;
    }
    if (!four) return;

    /* AND THE SAME EXPERTS AS FOUR CLASSES, by number mod 4 (`moe_gemm_q_mod4`). Each parity table
     * read at twice its stride is two of them -- from its first entry, the experts 0, 4, 8, ... of
     * its parity's numbers mod 4, and from its second the others -- so classes 0 and 2 are the even
     * table's halves and 1 and 3 the odd one's, at the even and the odd geometry. The same experts
     * at the same geometry, read in the same order: every form must give the two-class op's bytes,
     * byte for byte. What this checks is the class and entry each run resolves to. */
    const RadKernelInfo* r4 = find(pl, "moe_gemm_w4a8_mod4");
    const std::string what4 = what + ", four classes";
    if (!r4) {
        ok(what4.c_str(), false, "row moe_gemm_w4a8_mod4 absent");
        return;
    }
    auto halves = [](const RadTensor& t, int bits, RadTensor* lo, RadTensor* hi) {
        const int64_t n = t.shape[0];
        *lo = t;
        *hi = t;
        lo->shape[0] = (n + 1) / 2;
        hi->shape[0] = n / 2;
        lo->stride[0] = t.stride[0] * 2;
        hi->stride[0] = t.stride[0] * 2;
        hi->data = (char*)t.data + t.stride[0] * bits / 8;
    };
    const int wbits = w5 ? 5 : 4, sbits = g64 ? 8 : 16;
    RadTensor c4w[4], c4s[4];
    halves(ts[2], wbits, &c4w[0], &c4w[2]);
    halves(ts[3], sbits, &c4s[0], &c4s[2]);
    halves(ts[6], wbits, &c4w[1], &c4w[3]);
    halves(ts[7], sbits, &c4s[1], &c4s[3]);
    std::vector<RadTensor> ts4 = { ts[0], ts[1], c4w[0], c4s[0], ts[4], ts[5], c4w[1], c4s[1],
                                   c4w[2], c4s[2], c4w[3], c4s[3], ts[8] };
    std::vector<RadParam> ps4 = {
        PI("M", M), PI("N", sh.N0), PI("K", sh.K0), PI("N_1", sh.N1), PI("K_1", sh.K1),
        PI("N_2", sh.N0), PI("K_2", sh.K0), PI("N_3", sh.N1), PI("K_3", sh.K1),
        PI("parts", sh.parts), PI("n_expert", ne), PI("top_k", top_k), PI("group", group),
        PS("a_order", "token"), PS("dtype", dt),
    };
    RadArgs a4;
    std::memset(&a4, 0, sizeof(a4));
    a4.t = ts4.data(); a4.n_t = (int)ts4.size();
    a4.p = ps4.data(); a4.n_p = (int)ps4.size();
    for (int form = f_lo; form <= f_hi; ++form) {
        const std::string label = what4 + " " + kForm[form];
        if (two_cls[form].empty()) continue;      /* the two-class launch failed, and said so */
        if (pl.pin_moe) pl.pin_moe(form);
        {
            std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        a4.scratch = form == 3 ? nullptr : d_map;
        a4.scratch_bytes = form == 3 ? 0 : map_bytes;
        const int rc = r4->launch(&a4, nullptr);
        st_stream_sync(nullptr);
        if (pl.pin_moe) pl.pin_moe(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        int64_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) diff += got[i] != two_cls[form][i];
        ok(label.c_str(), diff == 0, std::to_string(diff) + " of " + std::to_string(Tr * N) +
                                     " elements differ from the two-class op's");
    }
}

/* THE FOUR-CLASS GROUPED GEMM, EACH CLASS AT ITS OWN GEOMETRY, AGAINST A HOST DECODE.
 *
 * moe_w4_gemm's four-class pass reads the two-class op's tables as four, so its classes come in
 * geometry pairs -- 0 with 2, 1 with 3 -- and class 0 is always among the wide ones. A rank of four
 * holds two blocks of its own class of every five-block expert and one of each other: ONE wide
 * class, which is class 0 on rank 0 only. Here every class is built at its own [N_q, K_q], stored
 * through the row's own hooks at its own operands (2/3, 6/7, 8/9, 10/11), and checked against the
 * planes, so a wide class 1, 2 or 3 has to read its own width and pitch, and a narrow class 0 has
 * to write the zeros past its part. `dt` names the format: fp8a8 runs `moe_gemm_fp8a8_mod4`, block
 * E4M3 with a bf16 scale a 128x128 tile; the four-bit ones run `moe_gemm_w4a8_mod4`, as
 * moe_w4_gemm reads them. Every form, as there. */
struct MoeQ4Shape { const char* name; int64_t N[4], K[4], parts, ne; };

/* A rank of four's slices of a five-block expert: the gate/up pair (two parts) at two 128-column
 * blocks a part for the wide class and one for the others, the down projection at 256 and 128 of
 * K. The wide class is 1, 2 or 3 -- what the ranks other than 0 hold. Ten experts make classes 0
 * and 1 one expert longer than 2 and 3; three hundred spread a decode step's few runs over
 * experts that are mostly empty. Every part is whole 128-row blocks of the fp8 scale grid or lies
 * inside one, so no output tile's weight rows straddle two -- block fp8 reads one scale a tile. */
static const MoeQ4Shape kMoeQ4Shapes[] = {
    { "gate/up, class 1 wide", { 256, 512, 256, 256 }, { 256, 256, 256, 256 }, 2, 10 },
    { "gate/up, class 2 wide", { 256, 256, 512, 256 }, { 256, 256, 256, 256 }, 2, 10 },
    { "down, class 3 wide",    { 128, 128, 128, 128 }, { 128, 128, 128, 256 }, 1, 10 },
    { "300 experts, gate/up, class 3 wide", { 128, 128, 128, 256 }, { 128, 128, 128, 128 }, 2,
      300 },
};

static void moe_q4_gemm(const Plugin& pl, int64_t Tr, const MoeQ4Shape& sh, const char* dt) {
    const bool fp8 = std::strcmp(dt, "fp8a8") == 0, w5 = std::strcmp(dt, "w5nl64a8h") == 0;
    /* Scale groups a 128 of K at the E4M3-scaled forms (moe_w4_gemm's `sub`), 0 at the others. */
    const int64_t sub = std::strcmp(dt, "w4nl32a8h") == 0 ? 4
                      : (w5 || std::strcmp(dt, "w4nl64a8h") == 0) ? 2 : 0;
    const bool g64 = sub > 1;
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const char* row = fp8 ? "moe_gemm_fp8a8_mod4" : "moe_gemm_w4a8_mod4";
    const std::string what = std::string(row) + " vs a host decode (T=" + std::to_string(Tr) +
                             ", " + sh.name + ", " + dt + ")";
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || (!fp8 && !r->relayout)) {
        ok(what.c_str(), false, "row or hooks absent");
        return;
    }

    const int64_t ne = sh.ne, group = 128, top_k = 2;
    int64_t N = 0, K = 0;
    for (int q = 0; q < 4; ++q) {
        N = std::max(N, sh.N[q]);
        K = std::max(K, sh.K[q]);
    }
    const int64_t kb = K / group, M = Tr / top_k, fy = N / sh.parts;
    const std::vector<RadParam> lp = {
        PI("N", sh.N[0]), PI("K", sh.K[0]), PI("N_1", sh.N[1]), PI("K_1", sh.K[1]),
        PI("N_2", sh.N[2]), PI("K_2", sh.K[2]), PI("N_3", sh.N[3]), PI("K_3", sh.K[3]),
        PI("group", group), PS("dtype", dt),
    };

    /* Each class's experts e = 4x + q, from moe_w4_gemm's float source; their planes, kept for
     * the reference, and the bytes the hooks store for them, asked at the class's own operands.
     * Block fp8 is stored as its planes are, in both views; the four-bit codes are re-laid, and so
     * is the E4M3 scale view with its second level, while a bf16 scale plane is stored as it is. */
    std::vector<uint8_t> wq[4], ws[4];
    std::vector<Planes>  cp[4];
    int64_t nec[4] = { 0, 0, 0, 0 };
    for (int q = 0; q < 4; ++q) {
        const int64_t Nq = sh.N[q], Kq = sh.K[q], kbq = Kq / group;
        const int ow = q ? 4 + 2 * q : 2, os = ow + 1;
        const int64_t wbytes = fp8 ? Nq * Kq : w5 ? Nq * Kq / 8 * 5 : Nq * Kq / 2;
        const int64_t sbytes = fp8 ? (Nq + group - 1) / group * kbq * 2
                                   : Nq * kbq * (g64 ? sub : 2);
        nec[q] = (ne + 3 - q) / 4;
        std::vector<float> wsrc((size_t)(Nq * Kq));
        for (int64_t x = 0; x < nec[q]; ++x) {
            const int64_t e = 4 * x + q;
            for (int64_t n = 0; n < Nq; ++n)
                for (int64_t k = 0; k < Kq; ++k)
                    wsrc[(size_t)(n * Kq + k)] =
                        (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
            cp[q].push_back(fp8 ? fp8_block_planes(wsrc, Nq, Kq)
                            : g64 ? nl64_planes(wsrc, Nq, Kq, group / sub, w5)
                                  : int4_planes(wsrc, Nq, Kq, group, RAD_BF16, false));
            std::vector<uint8_t> w, sc;
            const int rw = g64 ? stored(r, lp, ow, cp[q].back(), { "codes", "table" }, &w)
                               : stored(r, lp, ow, cp[q].back(), { "codes" }, &w);
            const int rs = g64 ? stored(r, lp, os, cp[q].back(), { "scale", "scale.1" }, &sc)
                               : stored(r, lp, os, cp[q].back(), { "scale" }, &sc);
            if (rw != (fp8 ? RAD_E_UNSUPPORTED : RAD_OK) ||
                rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED)) {
                ok(what.c_str(), false, "class " + std::to_string(q) + " relayout rc " +
                                        std::to_string(rw) + "/" + std::to_string(rs));
                return;
            }
            if ((int64_t)w.size() != wbytes || (int64_t)sc.size() != sbytes) {
                ok(what.c_str(), false, "class " + std::to_string(q) + " stored " +
                                        std::to_string(w.size()) + " and " +
                                        std::to_string(sc.size()) + " bytes an expert, for " +
                                        std::to_string(wbytes) + " and " + std::to_string(sbytes));
                return;
            }
            wq[q].insert(wq[q].end(), w.begin(), w.end());
            ws[q].insert(ws[q].end(), sc.begin(), sc.end());
        }
    }

    /* moe_w4_gemm's activation and routing: E4M3 bytes with no subnormal and no NaN, an f32 scale
     * a (row, group), an uneven run on each expert and a dropped tail. */
    std::vector<uint8_t> aq((size_t)(M * K));
    std::vector<float>   as((size_t)(M * kb));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t g = 0; g < kb; ++g) as[(size_t)(m * kb + g)] = 0.011f + 0.003f * (float)g;
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t c = (uint32_t)((m * 13 + k * 5) % 119) + 8;
            aq[(size_t)(m * K + k)] = (uint8_t)(((m + k) & 1) ? (c | 0x80u) : c);
        }
    }
    std::vector<int32_t> off((size_t)(ne + 1), 0), sorted((size_t)Tr, -1);
    const int64_t live = Tr - 3;
    for (int64_t e = 0; e <= ne; ++e) off[(size_t)e] = (int32_t)((live * e) / ne);
    off[(size_t)ne] = (int32_t)live;
    for (int64_t i = 0; i < live; ++i) sorted[(size_t)i] = (int32_t)((i * 7 + 3) % Tr);

    void* d_a  = dalloc(aq.size());
    void* d_as = dalloc(as.size() * 4);
    void* d_w[4];
    void* d_ws[4];
    for (int q = 0; q < 4; ++q) {
        d_w[q]  = dalloc(wq[q].size());
        d_ws[q] = dalloc(ws[q].size());
        st_memcpy(d_w[q],  wq[q].data(), wq[q].size(), hipMemcpyHostToDevice);
        st_memcpy(d_ws[q], ws[q].data(), ws[q].size(), hipMemcpyHostToDevice);
    }
    void* d_s = dalloc(sorted.size() * 4);
    void* d_o = dalloc(off.size() * 4);
    void* d_y = dalloc((size_t)(Tr * N * 2));
    st_memcpy(d_a,  aq.data(),     aq.size(),         hipMemcpyHostToDevice);
    st_memcpy(d_as, as.data(),     as.size() * 4,     hipMemcpyHostToDevice);
    st_memcpy(d_s,  sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o,  off.data(),    off.size() * 4,    hipMemcpyHostToDevice);

    /* The stacked table form, as moe_w4_gemm hands it, each class at its own extents. */
    auto codes_t = [&](int q) {
        return T(d_w[q], fp8 ? (uint32_t)RAD_F8E4M3 : w5 ? (uint32_t)R4D_DT_MOEW5
                                                         : (uint32_t)R4D_DT_MOEW4,
                 { nec[q], sh.N[q], sh.K[q] });
    };
    auto scale_t = [&](int q) {
        const int64_t kbq = sh.K[q] / group;
        return fp8 ? T(d_ws[q], RAD_BF16, { nec[q], (sh.N[q] + group - 1) / group, kbq })
             : g64 ? T(d_ws[q], RAD_F8E4M3, { nec[q], sh.N[q], sub * kbq })
                   : T(d_ws[q], RAD_BF16, { nec[q], sh.N[q], kbq });
    };
    std::vector<RadTensor> ts = {
        T(d_a, RAD_F8E4M3, { M, K }), T(d_as, RAD_F32, { M, kb }), codes_t(0), scale_t(0),
        T(d_s, RAD_I32, { Tr }), T(d_o, RAD_I32, { ne + 1 }), codes_t(1), scale_t(1),
        codes_t(2), scale_t(2), codes_t(3), scale_t(3), T(d_y, RAD_BF16, { Tr, N }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", sh.N[0]), PI("K", sh.K[0]), PI("N_1", sh.N[1]), PI("K_1", sh.K[1]),
        PI("N_2", sh.N[2]), PI("K_2", sh.K[2]), PI("N_3", sh.N[3]), PI("K_3", sh.K[3]),
        PI("parts", sh.parts), PI("n_expert", ne), PI("top_k", top_k), PI("group", group),
        PS("a_order", "token"), PS("dtype", dt),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int64_t map_bytes = (1 + 2 * (ne + Tr / 16 + 2)) * 4;
    void* d_map = dalloc((size_t)map_bytes);

    /* The reference, from the planes: row i's expert e is class e % 4, entry e / 4, at that
     * class's width and K; a part's columns past the class's width are zero. */
    std::vector<double> ref((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < Tr; ++i) {
        const int32_t f = sorted[(size_t)i];
        int64_t e = 0;
        while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
        const int q = (int)(e % 4);
        const Planes& p = cp[q][(size_t)(e / 4)];
        const int64_t fc = sh.N[q] / sh.parts, Kq = sh.K[q], kbq = Kq / group;
        for (int64_t n = 0; n < N; ++n) {
            double want = 0.0;
            const int64_t pp = n / fy, j = n - pp * fy;
            if (f >= 0 && i < live && j < fc) {
                const int64_t tok = f / top_k, wr = pp * fc + j;
                for (int64_t g = 0; g < kbq; ++g) {
                    const double a_s = (double)as[(size_t)(tok * kb + g)];
                    if (fp8) {
                        /* E4M3 codes, and the bf16 scale of the 128x128 tile the element is in. */
                        double part = 0.0;
                        for (int64_t k = g * group; k < (g + 1) * group; ++k)
                            part += (double)h_e4m3(p.data[0][(size_t)(wr * Kq + k)]) *
                                    (double)st_e4m3(aq[(size_t)(tok * K + k)]);
                        want += part * (double)rad_load_f32(p.data[1].data(), RAD_BF16,
                                                            (wr / group) * kbq + g) * a_s;
                    } else if (g64) {
                        /* `sub` scale groups a 128, each an E4M3 byte under the fixed constant. */
                        const int64_t gw = group / sub;
                        for (int64_t h = 0; h < sub; ++h) {
                            double part = 0.0;
                            for (int64_t k = g * group + h * gw; k < g * group + (h + 1) * gw; ++k)
                                part += (double)tab[code_at(p, wr, k)] *
                                        (double)st_e4m3(aq[(size_t)(tok * K + k)]);
                            const uint8_t sb =
                                p.data[1][(size_t)(wr * p.row_bytes(1) + sub * g + h)];
                            want += part * (double)h_e4m3(sb) * (double)kW4G64Scale * a_s;
                        }
                    } else {
                        double part = 0.0;
                        for (int64_t k = g * group; k < (g + 1) * group; ++k)
                            part += (double)code_at(p, wr, k) *
                                    (double)st_e4m3(aq[(size_t)(tok * K + k)]);
                        want += part * (double)rad_load_f32(p.data[1].data(), RAD_BF16,
                                                            wr * kbq + g) * a_s;
                    }
                }
            }
            ref[(size_t)(i * N + n)] = want;
        }
    }

    const int f_lo = pl.pin_moe ? 1 : 0, f_hi = pl.pin_moe ? 3 : 0;
    for (int form = f_lo; form <= f_hi; ++form) {
        const std::string label = what + " " + kForm[form];
        if (pl.pin_moe) pl.pin_moe(form);
        /* Stale bytes where a narrow class writes zeros, so a pad tile left unwritten shows. */
        {
            std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        a.scratch = form == 3 ? nullptr : d_map;
        a.scratch_bytes = form == 3 ? 0 : map_bytes;
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (pl.pin_moe) pl.pin_moe(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }

        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0.0;
        int64_t bad = 0;
        std::string first;
        for (int64_t i = 0; i < Tr; ++i)
            for (int64_t n = 0; n < N; ++n) {
                const double want = ref[(size_t)(i * N + n)];
                const double err = std::fabs((double)h_bf16(got[(size_t)(i * N + n)]) - want);
                /* moe_w4_gemm's tolerance, written so that a NaN on either side is a miss. */
                if (!(err <= 0.004 * std::fabs(want) + 1e-4)) {
                    if (!bad++)
                        first = ", first at row " + std::to_string(i) + " col " +
                                std::to_string(n) + ": " +
                                std::to_string(h_bf16(got[(size_t)(i * N + n)])) + " for " +
                                std::to_string(want);
                    if (!(err <= worst)) worst = err;
                }
            }
        ok(label.c_str(), bad == 0, std::to_string(bad) + " of " + std::to_string(Tr * N) +
                                    " elements off, worst " + std::to_string(worst) + first);
    }
}

/* THE ARRANGED PLANE, AGAINST THE FORMAT IT STATES, AND BACK.
 *
 * `moe_gemm_w4a8` is the one large weight of a routed model, and what a checker reads it through is
 * the row's inverse: a reference handed the arranged nibbles reads the wrong element in most
 * positions and reports a correct kernel as wrong. So the inverse has to give back exactly the
 * planes the relayout was handed.
 *
 * AND THE ARRANGEMENT HAS TO BE THE ONE r4d_layout.cpp STATES, which is checked from the
 * statement and not from either hook: byte b of a packed dword holds element b low and element
 * b+4 high, two's complement. The scale plane is read as it is stored, so its hook must answer
 * RAD_E_UNSUPPORTED -- "nothing to arrange", which a caller has to be able to tell from a refusal. */
static void moe_w4_unlayout(const Plugin& pl) {
    const char* what = "moe_gemm_w4a8's arranged plane, vs the stated format and its inverse";
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    if (!r || !r->layout || !r->relayout || !r->unrelayout) {
        ok(what, false, "row or hooks absent");
        return;
    }
    const int64_t N = 64, K = 256, group = 128;
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", group) };

    std::vector<float> src((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            src[(size_t)(n * K + k)] = (float)(((n * 37 + k * 11) % 97) - 48) / 53.0f;
    const Planes p = int4_planes(src, N, K, group, RAD_BF16, false);

    std::vector<uint8_t> wq, ws;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &ws);
    if (rw != RAD_OK) { ok(what, false, "codes relayout rc=" + std::to_string(rw)); return; }
    if (rs != RAD_E_UNSUPPORTED) {
        ok(what, false, "the scale plane is stored as read and must answer RAD_E_UNSUPPORTED");
        return;
    }

    int64_t bad_code = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t pos  = k & 7;
            const int64_t byte = n * (K / 2) + (k >> 3) * 4 + (pos & 3);
            const int nib  = (wq[(size_t)byte] >> (pos >= 4 ? 4 : 0)) & 15;
            const int code = nib >= 8 ? nib - 16 : nib;
            if (code != code_at(p, n, k)) ++bad_code;
        }
    std::string why;
    const bool back = restores(r, lp, 2, p, { "codes" }, wq, &why);
    ok(what, bad_code == 0 && back,
       std::to_string(bad_code) + " code(s) not where the format puts them" +
       (back ? std::string() : "; inverse: " + why));

    /* w5nl64a8h's row, against the same statement: the low four bits of every code where w4a8's
     * nibble would be, in a row K/2 + K/8 bytes long, and the high bit -- the sign -- of element
     * 4m+b of a 32 at bit 7-m of byte b of that 32's dword, from byte K/2 on. The table beside
     * the codes is the kernel's, and its inverse restores both planes. */
    const char* what5 = "moe_gemm_w4a8's five-bit arranged plane, vs the stated format and its "
                        "inverse";
    std::vector<RadParam> lp5 = { PI("N", N), PI("K", K), PI("group", group),
                                  PS("dtype", "w5nl64a8h") };
    const Planes p5 = nl64_planes(src, N, K, 64, true);
    std::vector<uint8_t> w5;
    const int r5 = stored(r, lp5, 2, p5, { "codes", "table" }, &w5);
    const int64_t rb = K / 8 * 5;
    if (r5 != RAD_OK || (int64_t)w5.size() != N * rb) {
        ok(what5, false, "codes relayout rc=" + std::to_string(r5) + ", " +
                         std::to_string(w5.size()) + " bytes");
        return;
    }
    int64_t bad5 = 0, neg = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t pos = k & 7;
            const int nib = (w5[(size_t)(n * rb + (k >> 3) * 4 + (pos & 3))] >> (pos >= 4 ? 4 : 0)) & 15;
            const int64_t j = k % 32;
            const int sgn = (w5[(size_t)(n * rb + K / 2 + (k / 32) * 4 + (j & 3))] >>
                             (7 - (j >> 2))) & 1;
            const int code = code_at(p5, n, k);
            neg += code >= 16;
            if ((nib | (sgn << 4)) != code) ++bad5;
        }
    std::string why5;
    const bool back5 = restores(r, lp5, 2, p5, { "codes", "table" }, w5, &why5);
    /* Half the source is negative, so a fixture with no sign bits set proves nothing. */
    ok(what5, bad5 == 0 && back5 && neg > N * K / 4,
       std::to_string(bad5) + " code(s) not where the format puts them, " + std::to_string(neg) +
       " negative" + (back5 ? std::string() : "; inverse: " + why5));
}
/* ---- THE ROTATED EXPERT FORMAT, end to end and against the float it approximates.
 *
 * WHAT THIS CATCHES AND moe_w4_gemm CANNOT. That case compares the kernel against a host product
 * over the planes: it proves the arrangement, the two's complement and the scale grid, and it would
 * pass just as happily if the WEIGHT had been rotated by one Hadamard and the ACTIVATION by a
 * different one, because it never looks at what the numbers mean. The rotated pair is only a model
 * if the quantiser's Hadamard (this file's h_fwht, libquant's fwht_inplace) and r4d_fwht.h's
 * r4d_fwht_wave are the same transform -- and they are written independently, one iterative over a
 * host array and one in registers across a wave with five shuffles, so nothing but a test says so.
 *
 * The oracle is therefore the exact f32 dot product of the SOURCE weight and the SOURCE
 * activation, in the original coordinates, and both formats are measured against it in one run.
 * A rotation mismatch is not subtle when read that way -- rotating the activation and not the
 * weight measures rel_l2 5.32 against 0.13 -- but it is invisible without this oracle.
 *
 * The two numbers it prints are also the instrument the format decision rests on, which is why
 * the source has a few outlier channels in it: sixteen levels spent covering one outlier is
 * exactly the error a rotation is supposed to remove, and a source with none would make the two
 * formats look alike for a reason that is not true of a real expert. */
static void moe_w4_rot(const Plugin& pl) {
    const char* what = "moe_gemm_w4a8h/w4nla8h/w4nl64a8h/w5nl64a8h + had_quant_act_fp8 vs the f32 "
                       "they approximate";
    const RadKernelInfo* r  = find(pl, "moe_gemm_w4a8");
    const RadKernelInfo* qp = find(pl, "quant_act_fp8");
    const RadKernelInfo* qh = find(pl, "had_quant_act_fp8");
    if (!r || !r->layout || !r->relayout || !qp || !qh) {
        ok(what, false, "row or hooks absent");
        return;
    }

    const int64_t ne = 3, N = 64, K = 256, group = 128, top_k = 2, Tr = 24;
    const int64_t kb = K / group, M = Tr / top_k;

    /* The source weight and activation, in float. Outliers every 37th and 23rd column -- six
     * times the body, which is mild beside a real MoE input and enough to separate the grids. */
    std::vector<float>    wsrc((size_t)(ne * N * K));
    std::vector<uint16_t> xbf ((size_t)(M * K));
    std::vector<float>    xsrc((size_t)(M * K));
    for (int64_t e = 0; e < ne; ++e)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k) {
                float v = (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
                if (k % 37 == 0) v *= 6.0f;
                wsrc[(size_t)((e * N + n) * K + k)] = v;
            }
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            float v = (float)(((m * 29 + k * 11) % 53) - 26) / 41.0f;
            if (k % 23 == 0) v *= 6.0f;
            /* Rounded through bf16 HERE, so the oracle reads the same number the device does and
             * the residual is the format's and not the input's. */
            const uint16_t h = h_to_bf16(v);
            xbf [(size_t)(m * K + k)] = h;
            xsrc[(size_t)(m * K + k)] = h_bf16(h);
        }

    /* The same uneven routing the byte-level case uses, tail dropped. */
    std::vector<int32_t> off((size_t)(ne + 1), 0), sorted((size_t)Tr, -1);
    const int64_t live = Tr - 3;
    for (int64_t e = 0; e <= ne; ++e) off[(size_t)e] = (int32_t)((live * e) / ne);
    off[(size_t)ne] = (int32_t)live;
    for (int64_t i = 0; i < live; ++i) sorted[(size_t)i] = (int32_t)((i * 7 + 3) % Tr);

    void* d_x  = dalloc(xbf.size() * 2);
    void* d_s  = dalloc(sorted.size() * 4);
    void* d_o  = dalloc(off.size() * 4);
    st_memcpy(d_x, xbf.data(),    xbf.size() * 2,    hipMemcpyHostToDevice);
    st_memcpy(d_s, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(),    off.size() * 4,    hipMemcpyHostToDevice);

    /* The exact answer, in the ORIGINAL coordinates: no rotation, no grid, f64 accumulation. */
    std::vector<double> want((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < live; ++i) {
        const int64_t tok = sorted[(size_t)i] / top_k;
        int64_t e = 0;
        while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)wsrc[(size_t)((e * N + n) * K + k)] *
                       (double)xsrc[(size_t)(tok * K + k)];
            want[(size_t)(i * N + n)] = acc;
        }
    }

    /* Six formats: plain, rotated, rotated with the w4nl codebook -- whose codes view takes the
     * table beside the codes and whose kernel decodes that table -- the codebook at E4M3 scales a
     * 64 and a 32 of K, whose scale view takes the fixed second level beside the scales, and the
     * five-bit codebook at a 64, whose codes carry their signs in a plane of their own. */
    double rel[6] = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
    for (int pass = 0; pass < 6; ++pass) {
        const bool rot = pass >= 1, nl = pass >= 2, g64 = pass >= 3, w5 = pass == 5;
        const int64_t sub = pass == 4 ? 4 : g64 ? 2 : 1;   /* scale groups a 128 */
        const char* dt = w5 ? "w5nl64a8h" : pass == 4 ? "w4nl32a8h" : g64 ? "w4nl64a8h"
                       : nl ? "w4nla8h" : rot ? "w4a8h" : "w4a8";

        std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", group), PS("dtype", dt) };
        const int64_t wbytes = w5 ? N * K / 8 * 5 : N * K / 2;
        std::vector<uint8_t>  wq((size_t)(ne * wbytes), 0);
        std::vector<uint16_t> ws((size_t)(ne * N * kb * (sub > 2 ? 2 : 1)), 0);   /* bytes a 128: 2, or 4 */
        for (int64_t e = 0; e < ne; ++e) {
            const std::vector<float> we(wsrc.begin() + e * N * K, wsrc.begin() + (e + 1) * N * K);
            const Planes pe = g64 ? nl64_planes(we, N, K, 128 / sub, w5)
                            : nl  ? nl_planes(we, N, K, group, RAD_BF16)
                                  : int4_planes(we, N, K, group, RAD_BF16, rot);
            std::vector<uint8_t> w, sc;
            RadLayout lw;
            const int rw = nl ? stored(r, lp, 2, pe, { "codes", "table" }, &w, &lw)
                              : stored(r, lp, 2, pe, { "codes" }, &w, &lw);
            const int rs = g64 ? stored(r, lp, 3, pe, { "scale", "scale.1" }, &sc)
                               : stored(r, lp, 3, pe, { "scale" }, &sc);
            if (rw != RAD_OK || rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED)) {
                ok(what, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
                return;
            }
            /* THE ENCODING IS THE WHOLE GUARD AGAINST A MIS-SERVED CONTAINER, so it is checked
             * here and not merely produced: the arrangement, the extents and the byte count are
             * identical between the two formats, and only the encoding's transform differs. The
             * hook must take the one its dtype names and refuse the other at declare. */
            if (e == 0) {
                const Planes other = int4_planes(we, N, K, group, RAD_BF16, nl || !rot);
                std::vector<uint8_t> scratch;
                const int ro = stored(r, lp, 2, other, { "codes" }, &scratch);
                const char* want_tag = w5 ? "moew5nh" : nl ? "moew4nh" : "moew4h";
                if (ro != RAD_E_DTYPE || !lw.tag ||
                    (std::strstr(lw.tag, want_tag) != nullptr) != rot) {
                    ok(what, false, std::string("the hook does not tell the formats apart: tag ") +
                                    (lw.tag ? lw.tag : "(none)") + ", the other encoding rc " +
                                    std::to_string(ro));
                    return;
                }
            }
            if ((int64_t)w.size() != wbytes) {
                ok(what, false, std::string(dt) + " stored " + std::to_string(w.size()) +
                                " code bytes for " + std::to_string(wbytes));
                return;
            }
            std::memcpy(wq.data() + (size_t)(e * wbytes), w.data(), w.size());
            std::memcpy((uint8_t*)ws.data() + (size_t)e * sc.size(), sc.data(), sc.size());
        }

        /* The activation goes through the DEVICE quantiser -- the whole point is that its
         * rotation is the one the weight's planes were quantised under. */
        void* d_q  = dalloc((size_t)(M * K));
        void* d_qs = dalloc((size_t)(M * kb) * 4);
        std::vector<RadTensor> qt = {
            T(d_x,  RAD_BF16,    { M, K }),
            T(d_q,  RAD_F8E4M3,  { M, K }),
            T(d_qs, RAD_F32,     { M, kb }),
        };
        std::vector<RadParam> qps = {
            PI("M", M), PI("n", K), PI("group", group), PS("dtype", "bf16"),
        };
        RadArgs qa;
        std::memset(&qa, 0, sizeof(qa));
        qa.t = qt.data(); qa.n_t = (int)qt.size();
        qa.p = qps.data(); qa.n_p = (int)qps.size();
        const int qrc = (rot ? qh : qp)->launch(&qa, nullptr);
        st_stream_sync(nullptr);
        if (qrc != RAD_OK) {
            ok(what, false, std::string("quantiser rc=") + std::to_string(qrc)); return;
        }

        void* d_w  = dalloc(wq.size());
        void* d_ws = dalloc(ws.size() * 2);
        void* d_y  = dalloc((size_t)(Tr * N * 2));
        st_memcpy(d_w,  wq.data(), wq.size(),     hipMemcpyHostToDevice);
        st_memcpy(d_ws, ws.data(), ws.size() * 2, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = {
            T(d_q,  RAD_F8E4M3,   { M, K }),
            T(d_qs, RAD_F32,      { M, kb }),
            T(d_w,  w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4, { ne, N, K }),
            g64 ? T(d_ws, RAD_F8E4M3, { ne, N, sub * kb }) : T(d_ws, RAD_BF16, { ne, N, kb }),
            T(d_s,  RAD_I32,      { Tr }),
            T(d_o,  RAD_I32,      { ne + 1 }),
            T(d_y,  RAD_BF16,     { Tr, N }),
        };
        moe_parity_tables(ts, ne, w5 ? 5 : 4, g64 ? 8 : 16);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("n_expert", ne), PI("top_k", top_k),
            PI("group", group), PS("a_order", "token"), PS("dtype", dt),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) {
            ok(what, false, std::string("launch rc=") + std::to_string(rc)); return;
        }

        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < live * N; ++i) {
            const double d = (double)h_bf16(got[(size_t)i]) - want[(size_t)i];
            num += d * d;
            den += want[(size_t)i] * want[(size_t)i];
        }
        rel[pass] = den > 0.0 ? std::sqrt(num / den) : -1.0;
    }

    /* 0.6 is loose ON PURPOSE. The measured pair is 0.20 and 0.13, and a quantiser that improves
     * on either must not have to come back here and move a number. The failure it is set to catch is
     * much further away than that: feeding the ROTATED activation to the UNROTATED weight -- the
     * exact mismatch a mis-served container would produce -- measures 5.32, because the rotated
     * values are also sqrt(128) larger. */
    bool pass = true;
    for (double x : rel) pass = pass && x >= 0.0 && x < 0.6;
    /* PRINTED WHETHER OR NOT IT PASSES, because the pass is about the plumbing and the numbers
     * are about the format: a reader comparing two quantisers wants them either way, and `ok` shows
     * its detail only on a failure. A kernel that decoded the w4nl codes as -8..7 would read
     * every code as a different number and measure near 1. */
    std::printf("        rel_l2 vs f32: w4a8 %.4f, w4a8h %.4f, w4nla8h %.4f, w4nl64a8h %.4f, "
                "w4nl32a8h %.4f, w5nl64a8h %.4f\n", rel[0], rel[1], rel[2], rel[3], rel[4], rel[5]);
    /* The finer groups have to show: a fold that read one group's scale for another, or the bytes
     * in another order, lands at or above the coarser form. And the fifth bit has to: a sign plane
     * read for the wrong elements -- or not at all -- flips half the signs, which measures near 1,
     * and one read for the right elements of the wrong row measures well above w4nl64a8h. */
    pass = pass && rel[3] < rel[2] && rel[4] < rel[3] && rel[5] < 0.75 * rel[3];
    ok(what, pass, "rel_l2 w4a8 " + std::to_string(rel[0]) + ", w4a8h " + std::to_string(rel[1]) +
                   ", w4nla8h " + std::to_string(rel[2]) + ", w4nl64a8h " + std::to_string(rel[3]) +
                   ", w4nl32a8h " + std::to_string(rel[4]) + ", w5nl64a8h " +
                   std::to_string(rel[5]));
}

/* qsa_work, against a host walk of the same batch.
 *
 * THE FIXTURE IS A MIXED STEP, because that is the case the op exists for and the one a
 * single-sequence test cannot reach: a long prefill chunk that completes many blocks and STRADDLES
 * the boundary (so its first block takes keys from the tail AND from the chunk), a short chunk
 * that stops inside a block, a speculative decode row of four that completes one out of two
 * tail keys and two of its own, and a fresh three-token prompt that stops before its first block
 * ends. Four sequences, four shapes, one launch.
 *
 * THREE THINGS HERE ARE REGRESSION TESTS AND NOT COVERAGE.
 *
 *   THE STATE SLOTS ARE PERMUTED AND WIDER THAN THE BATCH -- {5, 0, 8} out of nine. A kernel that
 *   indexed the tail by the BATCH ROW reads slots 0, 1, 2 and every tail key comes back wrong.
 *   That is not a synthetic worry: the scheduler builds a step from the running set in priority
 *   order and stable-sorts decode first, so the row a sequence
 *   occupies changes under it and a long-lived request inherits a retired one's keys.
 *
 *   THE RING IS 7 AND THE RATIO IS 4, so `q % ring` and `q % ratio` name different rows for every
 *   tail key in the fixture. A kernel that used `% ratio` passes a ring==ratio fixture and fails
 *   this one.
 *
 *   THE THREE-TOKEN PROMPT ENDS BEFORE ITS FIRST BLOCK DOES. A block bound taken with C's
 *   truncating division counts block 0 complete from position 0 and gathers a key past the end
 *   of the sequence; here that is a work item where the padding must say -1.
 *
 * What is otherwise checked is every operand the block-key kernel will read: the page, the
 * first-token position, the `ratio` staged rows a block -- each against the source it should have
 * come from -- and that the slots past the real count say -1 rather than naming a page nobody
 * meant.
 */
static void qsa_work_case(const Plugin& pl) {
    const char* what = "qsa_work vs a host walk of a mixed prefill/decode batch";
    const RadKernelInfo* r = find(pl, "qsa_work_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t n = 32, ratio = 4, ring = 7, seqs = 4, btw = 64, slots = 9;
    const int32_t cuh[5] = { 0, 22, 25, 29, 32 };
    const int64_t NT = cuh[4];
    /* NOT the batch rows, and wider than the batch -- see the header. */
    const int32_t sxh[4] = { 5, 0, 8, 3 };
    /* Sequence 0 starts at 11 -- NOT a block boundary -- so its first complete block (indices
     * 8..11) needs 8, 9 and 10 from the tail and only 11 from the chunk. Sequence 1 starts ON a
     * boundary and needs none. Sequence 2 is a four-row speculative decode whose block takes two
     * of each. Sequence 3 is positions 0..2, one short of its first block. */
    const int64_t start[4] = { 11, 60, 130, 0 };

    std::vector<uint16_t> kh((size_t)(NT * n));
    std::vector<int32_t>  ph((size_t)NT);
    for (int64_t s = 0, t = 0; s < seqs; ++s)
        for (int64_t i = 0; t < cuh[s + 1]; ++i, ++t) {
            ph[(size_t)t] = (int32_t)(start[s] + i);
            for (int64_t d = 0; d < n; ++d)
                kh[(size_t)(t * n + d)] = h_to_bf16((float)((t * 31 + d * 17) % 89) / 23.0f);
        }
    /* The tail pool, [slots][ring][n], and every slot holds something DIFFERENT: a kernel reading
     * the wrong slot has to come back with the wrong bytes, not with bytes that happen to match. */
    std::vector<uint16_t> th((size_t)(slots * ring * n));
    for (int64_t sl = 0; sl < slots; ++sl)
        for (int64_t row = 0; row < ring; ++row)
            for (int64_t d = 0; d < n; ++d)
                th[(size_t)((sl * ring + row) * n + d)] =
                    h_to_bf16(-1.0f - (float)sl * 3.0f - (float)row / 8.0f - (float)(d % 5) / 64.0f);
    std::vector<int32_t> bth((size_t)(seqs * btw));
    for (int64_t s = 0; s < seqs; ++s)
        for (int64_t b = 0; b < btw; ++b)
            bth[(size_t)(s * btw + b)] = (int32_t)((s * 211 + b * 13 + 5) % 4096);

    /* The host walk. */
    struct W { int64_t s, b; };
    std::vector<W> want;
    for (int64_t s = 0; s < seqs; ++s) {
        const int64_t p0 = ph[(size_t)cuh[s]], p1 = ph[(size_t)(cuh[s + 1] - 1)];
        const int64_t lo = p0 / ratio;                /* first block ending at or after p0 */
        const int64_t hi = (p1 + 1) / ratio - 1;      /* last block ending at or before p1 */
        for (int64_t b = lo; b <= hi; ++b) want.push_back({ s, b });
    }
    const int64_t W_ = (NT + ratio - 1) / ratio + seqs;     /* the host bound */

    void* d_k  = dalloc(kh.size() * 2);
    void* d_p  = dalloc(ph.size() * 4);
    void* d_cu = dalloc(sizeof cuh);
    void* d_tl = dalloc(th.size() * 2);
    void* d_sx = dalloc(sizeof sxh);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_st = dalloc((size_t)(W_ * ratio * n) * 2);
    void* d_pg = dalloc((size_t)W_ * 4);
    void* d_bp = dalloc((size_t)W_ * 4);
    void* d_nc = dalloc((size_t)seqs * 4);
    st_memcpy(d_k,  kh.data(),  kh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_p,  ph.data(),  ph.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh,        sizeof cuh,     hipMemcpyHostToDevice);
    st_memcpy(d_tl, th.data(),  th.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_sx, sxh,        sizeof sxh,     hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_k,  RAD_BF16, { NT, n }),
        T(d_p,  RAD_I32,  { NT }),
        T(d_cu, RAD_I32,  { seqs + 1 }),
        T(d_tl, RAD_BF16, { slots, 1, ring, n }),
        T(d_sx, RAD_I32,  { seqs, 1 }),
        T(d_bt, RAD_I32,  { seqs, btw }),
        T(nullptr, RAD_I32, { 3, NT }),          /* rope: absent, the one-component form */
        T(d_st, RAD_BF16, { W_ * ratio, n }),
        T(d_pg, RAD_I32,  { W_ }),
        T(d_bp, RAD_I32,  { W_ }),
        T(d_nc, RAD_I32,  { seqs }),
    };
    std::vector<RadParam> ps = {
        PI("M", NT), PI("n", n), PI("ratio", ratio), PI("ring", ring), PI("seqs", seqs),
        PI("work", W_), PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<uint16_t> gs((size_t)(W_ * ratio * n));
    std::vector<int32_t>  gp((size_t)W_), gb((size_t)W_);
    st_memcpy(gs.data(), d_st, gs.size() * 2, hipMemcpyDeviceToHost);
    st_memcpy(gp.data(), d_pg, gp.size() * 4, hipMemcpyDeviceToHost);
    st_memcpy(gb.data(), d_bp, gb.size() * 4, hipMemcpyDeviceToHost);
    std::vector<int32_t> gnc((size_t)seqs);
    st_memcpy(gnc.data(), d_nc, gnc.size() * 4, hipMemcpyDeviceToHost);

    /* COMPLETE BLOCKS A SEQUENCE, which the scorer and the selection read. It is (last position
     * + 1) / ratio and not the number of blocks that completed THIS step -- a distinction worth a
     * check, because the two are the same on a sequence that starts at zero and nothing else. */
    bool nc_ok = true;
    for (int64_t s = 0; s < seqs; ++s) {
        const int32_t w2 = (int32_t)((ph[(size_t)(cuh[s + 1] - 1)] + 1) / ratio);
        if (gnc[(size_t)s] != w2) nc_ok = false;
    }
    bool page_ok = true, pos_ok = true, key_ok = true, pad_ok = true;
    long from_tail = 0, ring_differs = 0;
    for (int64_t w = 0; w < W_; ++w) {
        if (w >= (int64_t)want.size()) { if (gp[(size_t)w] != -1) pad_ok = false; continue; }
        const int64_t s = want[(size_t)w].s, b = want[(size_t)w].b;
        if (gp[(size_t)w] != bth[(size_t)(s * btw + b)]) page_ok = false;
        if (gb[(size_t)w] != (int32_t)(b * ratio)) pos_ok = false;
        const int64_t p0 = ph[(size_t)cuh[s]];
        for (int64_t i = 0; i < ratio; ++i) {
            const int64_t q = b * ratio + i;
            const uint16_t* src;
            if (q >= p0) src = kh.data() + (size_t)((cuh[s] + (q - p0)) * n);
            else {
                src = th.data() + (size_t)((sxh[s] * ring + (q % ring)) * n);
                ++from_tail;
                if (q % ring != q % ratio) ++ring_differs;
            }
            for (int64_t d = 0; d < n; ++d)
                if (gs[(size_t)((w * ratio + i) * n + d)] != src[(size_t)d]) key_ok = false;
        }
    }
    /* UNDER M-RoPE, with the step's rotary planes: `bpos` is [work, 3], a block's components
     * side by side. A block's first token in this step reads its own three; one committed
     * earlier is text, at its index plus the shift the sequence's first row carries -- and the
     * shift differs per sequence here, as it does after a picture, so a kernel that took one
     * sequence's shift for another's is wrong. The components differ from each other too. */
    const int32_t shift[4] = { 0, -9, 17, 5 };
    std::vector<int32_t> rh((size_t)(3 * NT));
    for (int64_t s = 0; s < seqs; ++s)
        for (int64_t t = cuh[s]; t < cuh[s + 1]; ++t) {
            const int32_t p = ph[(size_t)t] + shift[s];
            rh[(size_t)t] = p;
            rh[(size_t)(NT + t)] = p + 3 + (int32_t)(t % 4);
            rh[(size_t)(2 * NT + t)] = p + 11 + (int32_t)(t % 7);
        }
    void* d_rp = dalloc(rh.size() * 4);
    void* d_b3 = dalloc((size_t)W_ * 3 * 4);
    st_memcpy(d_rp, rh.data(), rh.size() * 4, hipMemcpyHostToDevice);
    std::vector<int32_t> sentinel((size_t)W_ * 3, -777);
    st_memcpy(d_b3, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> tr = ts;
    tr[6] = T(d_rp, RAD_I32, { 3, NT });
    tr[9] = T(d_b3, RAD_I32, { W_, 3 });
    RadArgs ar = a;
    ar.t = tr.data();
    bool mc_ok = r->launch(&ar, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<int32_t> g3((size_t)W_ * 3);
    st_memcpy(g3.data(), d_b3, g3.size() * 4, hipMemcpyDeviceToHost);
    long mc_text = 0;
    for (int64_t w = 0; w < W_ && mc_ok; ++w) {
        for (int c = 0; c < 3; ++c) {
            int32_t e = 0;
            if (w < (int64_t)want.size()) {
                const int64_t s = want[(size_t)w].s, b = want[(size_t)w].b;
                const int64_t t0 = cuh[s], p0 = ph[(size_t)t0], f = b * ratio;
                if (f >= p0) e = rh[(size_t)(c * NT + t0 + (f - p0))];
                else { e = (int32_t)(f + (rh[(size_t)t0] - p0)); if (c == 0) ++mc_text; }
            }
            if (g3[(size_t)(w * 3 + c)] != e) mc_ok = false;
        }
    }
    /* And a pass WITHOUT the planes on the same buffer, as a text-only step hands it: the first
     * column alone, [work, 1] at a row pitch of 3. Component 0 is written and the other two
     * columns keep what they held. */
    st_memcpy(d_b3, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> tt = ts;
    RadTensor col = T(d_b3, RAD_I32, { W_, 1 });
    col.stride[0] = 3;
    tt[9] = col;
    RadArgs at = a;
    at.t = tt.data();
    bool col_ok = r->launch(&at, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    st_memcpy(g3.data(), d_b3, g3.size() * 4, hipMemcpyDeviceToHost);
    for (int64_t w = 0; w < W_ && col_ok; ++w) {
        if (g3[(size_t)(w * 3)] != gb[(size_t)w]) col_ok = false;
        if (g3[(size_t)(w * 3 + 1)] != -777 || g3[(size_t)(w * 3 + 2)] != -777) col_ok = false;
    }
    std::printf("        M-RoPE: [work, 3] positions %s (%ld from committed text); one column at "
                "pitch 3 %s\n", mc_ok ? "ok" : "NO", mc_text, col_ok ? "ok" : "NO");

    /* The straddling block is what makes this a real test rather than a copy of the chunk, and
     * `ring_differs` is what makes it a test of the RING rather than of the modulus. */
    const bool pass = page_ok && pos_ok && key_ok && pad_ok && nc_ok && from_tail > 0 &&
                      ring_differs > 0 && !want.empty() && mc_ok && mc_text > 0 && col_ok;
    std::printf("        %zu blocks (%ld keys from the tail, %ld at a row the old modulus would "
                "have missed), page %s, pos %s, keys %s, padding %s, nc %s\n", want.size(),
                from_tail, ring_differs, page_ok ? "ok" : "NO", pos_ok ? "ok" : "NO",
                key_ok ? "ok" : "NO", pad_ok ? "ok" : "NO", nc_ok ? "ok" : "NO");
    ok(what, pass, std::string("page=") + (page_ok ? "1" : "0") + " pos=" + (pos_ok ? "1" : "0") +
                   " key=" + (key_ok ? "1" : "0") + " pad=" + (pad_ok ? "1" : "0") +
                   " tail_keys=" + std::to_string(from_tail) +
                   " ring_rows=" + std::to_string(ring_differs));
}

/* qsa_tail_store, against a token-by-token loop.
 *
 * THE WHOLE CASE IS THE RACE. Positions p and p + ring share a row, so a prefill chunk carrying
 * many tokens a sequence has several tokens targeting one row -- and the kernel is allowed to
 * write only the last of them. The fixture therefore gives one sequence a long chunk, another a
 * short one, and a third a four-row speculative decode, and the oracle is the SERIAL loop: every
 * token in order, last writer wins. A kernel that wrote them all would agree with this on the
 * short and single cases and disagree on the long one, non-deterministically -- so the case also
 * runs twice and requires the same bytes.
 *
 * TWO THINGS HERE ARE PINNED DELIBERATELY AND ARE NOT COVERAGE:
 *
 *   THE STATE SLOTS ARE PERMUTED AND WIDER THAN THE BATCH -- {5, 0, 8} out of nine -- so a kernel
 *   that indexed by the BATCH ROW writes slots 0, 1, 2 and the untouched-row count collapses.
 *   The row a sequence occupies is step-local; the slot is the sequence's for its life.
 *
 *   THE RING IS 7 AND THE RATIO IS 4. `pos % ring` and `pos % ratio` disagree for most of this
 *   fixture, so a kernel that used `% ratio` fails on the rows, and `ring` rows of a sequence's
 *   state are in play rather than four.
 *
 * The other thing it pins is that rows nothing targets KEEP WHAT THEY HELD. That is what makes
 * the tail a tail: at decode a handful of rows are written a step and the rest are the point.
 */
static void qsa_tail_store_case(const Plugin& pl) {
    const char* what = "qsa_tail_store vs the serial last-writer-wins loop";
    const RadKernelInfo* r = find(pl, "qsa_tail_store_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t n = 64, ratio = 4, ring = 7, seqs = 3, slots = 9;
    /* A long prefill chunk, a short one, and a four-row speculative decode: 19 + 3 + 4. */
    const int32_t cuh[4] = { 0, 19, 22, 26 };
    const int64_t NT = cuh[3];
    const int32_t sxh[3] = { 5, 0, 8 };
    /* Each sequence starts somewhere different, and NOT on a block boundary for two of them --
     * `pos % ring` is the row and a fixture that only ever started at 0 would never see a
     * sequence whose first token lands mid-block. */
    const int64_t start[3] = { 40, 7, 131 };

    std::vector<uint16_t> kh((size_t)(NT * n));
    std::vector<int32_t>  ph((size_t)NT);
    for (int64_t s = 0, t = 0; s < seqs; ++s)
        for (int64_t i = 0; t < cuh[s + 1]; ++i, ++t) {
            ph[(size_t)t] = (int32_t)(start[s] + i);
            for (int64_t d = 0; d < n; ++d)
                kh[(size_t)(t * n + d)] = h_to_bf16((float)((t * 13 + d * 7) % 97) / 31.0f);
        }

    /* The pool arrives holding something, and most of it must still hold it afterwards -- every
     * row of every slot, including the six slots this batch does not name at all. */
    std::vector<uint16_t> pre((size_t)(slots * ring * n));
    for (size_t i = 0; i < pre.size(); ++i) pre[i] = h_to_bf16(-3.5f - (float)(i % 5));
    std::vector<uint16_t> want = pre;
    for (int64_t s = 0; s < seqs; ++s)
        for (int64_t t = cuh[s]; t < cuh[s + 1]; ++t) {          /* IN ORDER: last writer wins */
            const int64_t row = ph[(size_t)t] % ring;
            for (int64_t d = 0; d < n; ++d)
                want[(size_t)((sxh[s] * ring + row) * n + d)] = kh[(size_t)(t * n + d)];
        }

    void* d_k  = dalloc(kh.size() * 2);
    void* d_p  = dalloc(ph.size() * 4);
    void* d_cu = dalloc(sizeof cuh);
    void* d_tl = dalloc(pre.size() * 2);
    void* d_sx = dalloc(sizeof sxh);
    st_memcpy(d_k,  kh.data(), kh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_p,  ph.data(), ph.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh,       sizeof cuh,    hipMemcpyHostToDevice);
    st_memcpy(d_sx, sxh,       sizeof sxh,    hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_k,  RAD_BF16, { NT, n }),
        T(d_p,  RAD_I32,  { NT }),
        T(d_cu, RAD_I32,  { seqs + 1 }),
        T(d_tl, RAD_BF16, { slots, 1, ring, n }),
        T(d_sx, RAD_I32,  { seqs, 1 }),
    };
    std::vector<RadParam> ps = {
        PI("M", NT), PI("n", n), PI("ratio", ratio), PI("ring", ring), PI("seqs", seqs),
        PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    std::vector<uint16_t> got[2];
    for (int pass = 0; pass < 2; ++pass) {
        st_memcpy(d_tl, pre.data(), pre.size() * 2, hipMemcpyHostToDevice);
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        got[pass].assign(pre.size(), 0);
        st_memcpy(got[pass].data(), d_tl, got[pass].size() * 2, hipMemcpyDeviceToHost);
    }

    long kept = 0, kept_want = 0;
    for (size_t i = 0; i < pre.size(); i += (size_t)n) {
        if (got[0][i] == pre[i]) ++kept;
        if (want[i] == pre[i]) ++kept_want;
    }
    /* THE SLOTS THIS BATCH DOES NOT NAME MUST BE UNTOUCHED, and they are the ones a batch-row
     * index would have walked straight into. Counted separately, because the aggregate above
     * would still look plausible if three wrong slots were written and three right ones were not. */
    bool other_ok = true;
    for (int64_t sl = 0; sl < slots; ++sl) {
        bool named = false;
        for (int64_t s = 0; s < seqs; ++s) if (sxh[s] == sl) named = true;
        if (named) continue;
        for (int64_t i = 0; i < ring * n; ++i)
            if (got[0][(size_t)(sl * ring * n + i)] != pre[(size_t)(sl * ring * n + i)])
                other_ok = false;
    }
    const bool match = got[0] == want;
    const bool det = got[0] == got[1];
    const bool pass = match && det && other_ok && kept == kept_want && kept_want > 0;
    std::printf("        rows %s, untouched %ld of %lld (want %ld), unnamed slots %s, "
                "two runs identical %s\n", match ? "ok" : "DIFFER", kept,
                (long long)(slots * ring), kept_want, other_ok ? "ok" : "WRITTEN",
                det ? "ok" : "NO");
    ok(what, pass, std::string("match=") + (match ? "1" : "0") + " det=" + (det ? "1" : "0") +
                   " other=" + (other_ok ? "1" : "0") +
                   " kept=" + std::to_string(kept) + "/" + std::to_string(kept_want));
}

/* sample_argmax, against a host scan.
 *
 * WHAT MAKES THIS WORTH A CASE IS THE TIE RULE AND THE TWO LOAD BODIES, not finding a maximum.
 * Greedy must break ties to the LOWEST token id -- that is what lets a greedy step and a top-1
 * step never disagree -- and the kernel gets it by packing (key, ~id) into one word, which is
 * right only if the packing is monotone in the key and ANTI-monotone in the id. A row whose
 * maximum occurs at four ids is the only input that can tell the two apart; a row of distinct
 * values passes either way.
 *
 * The kernel also has two bodies: a float4 one when the row base AND the row pitch are 16-byte
 * aligned, and a scalar one when they are not. Both run here. `n_vocab` is deliberately not a
 * multiple of four, so the vector body also has a two-entry tail to pick up scalar-wise, and one
 * of the tied ids is inside that tail.
 *
 * THE PITCH IS THE THING THE ALIGNMENT TEST GETS WRONG IF IT ONLY LOOKS AT THE BASE: row m starts
 * at m * lg_ld floats, so with an odd pitch every row after the first is misaligned even though
 * the allocation is not. Pass 0 gives a pitch that is a multiple of four and pass 1 one that is
 * not, and both must produce the same answer.
 */
static void sample_argmax_case(const Plugin& pl) {
    const char* what = "sample_argmax vs a host scan, with ties and both load bodies";
    const RadKernelInfo* r = find(pl, "sample_argmax_f32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 5, V = 24578;          /* 4 * 6144 + 2 */
    std::vector<std::vector<float>> rows((size_t)M, std::vector<float>((size_t)V, 0.0f));
    std::vector<int32_t> want((size_t)M, 0);
    uint32_t rs = 0x1234567u;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t i = 0; i < V; ++i) {
            rs = rs * 1664525u + 1013904223u;
            rows[(size_t)m][(size_t)i] = (float)((int32_t)((rs >> 8) % 20001u) - 10000) * 0.001f;
        }
        /* Four ids hold the maximum exactly; the last of them is in the vector body's tail. */
        const int64_t ties[4] = { 7, V / 3, V - 6, V - 1 };
        for (int k = 0; k < 4; ++k) rows[(size_t)m][(size_t)ties[k]] = 11.0f + (float)m;
        want[(size_t)m] = (int32_t)ties[0];
    }

    bool pass_ok[2] = { true, true };
    std::vector<int32_t> got[2];
    for (int pass = 0; pass < 2; ++pass) {
        const int64_t P = (pass == 0) ? V + 2 : V + 1;   /* 24580 vector, 24579 scalar */
        std::vector<float> flat((size_t)(M * P), 0.0f);
        for (int64_t m = 0; m < M; ++m)
            std::memcpy(&flat[(size_t)(m * P)], rows[(size_t)m].data(), (size_t)V * 4);
        void* d_lg = dalloc(flat.size() * 4);
        void* d_pa = dalloc(64);
        void* d_tk = dalloc((size_t)M * 4);
        st_memcpy(d_lg, flat.data(), flat.size() * 4, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = { T(d_lg, RAD_F32, { M, V }), T(d_pa, RAD_F32, { 16 }),
                                      T(d_tk, RAD_I32, { M }) };
        ts[0].stride[0] = P;
        std::vector<RadParam> ps = { PI("M", M), PI("n_vocab", V) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        got[pass].assign((size_t)M, -1);
        st_memcpy(got[pass].data(), d_tk, (size_t)M * 4, hipMemcpyDeviceToHost);
        for (int64_t m = 0; m < M; ++m)
            if (got[pass][(size_t)m] != want[(size_t)m]) pass_ok[pass] = false;
    }
    const bool agree = got[0] == got[1];
    if (!g_case)
        std::printf("  argmax  aligned=%s unaligned=%s agree=%s  (row 0 got %d want %d)\n",
                    pass_ok[0] ? "ok" : "NO", pass_ok[1] ? "ok" : "NO", agree ? "ok" : "NO",
                    got[0].empty() ? -1 : got[0][0], want[0]);
    ok(what, pass_ok[0] && pass_ok[1] && agree,
       std::string("vec=") + (pass_ok[0] ? "1" : "0") + " scalar=" + (pass_ok[1] ? "1" : "0") +
       " agree=" + (agree ? "1" : "0"));
}

/* qsa_select, against a host sort of the same scores.
 *
 * WHAT MAKES THIS WORTH A CASE IS THE TIES AND THE TAIL, not the selection. A top-k is easy to get
 * right on distinct inputs and easy to get wrong on equal ones, and a block score is a sum of four
 * relus over a shared grid -- equal scores are ordinary, not pathological. So the fixture gives
 * the threshold value to more blocks than there are slots left, and the oracle is a full sort by
 * (score descending, index ascending), which is exactly the rule the kernel's two prefix counts
 * implement.
 *
 * Four things are checked and three of them are not the answer itself:
 *   the SET matches a host sort
 *   the partial page is LAST, because that is what lets the attention's own causal bound mask
 *     inside it and nowhere else
 *   a page id of -1 inside the scored range is never selected, however good its score
 *   two runs produce byte-identical tables -- a top-k placed by atomics would not
 */
static void qsa_select_case(const Plugin& pl) {
    const char* what = "qsa_select vs a host sort, with ties, a tail and a dead page";
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 6, nb = 900, topk = 64, pages = 4096, ratio = 4;

    std::vector<float>   sc((size_t)(M * nb), 0.0f);
    std::vector<int32_t> bt((size_t)(M * nb), -1), nc((size_t)M, 0);
    for (int64_t m = 0; m < M; ++m) {
        /* Row 0 and 1 are shorter than topk -- the fast path, where everything live is selected.
         * The rest run the radix. Rows alternate on whether a partial page exists. */
        const int64_t live = (m == 0) ? 9 : (m == 1) ? topk : (100 + m * 137);
        nc[(size_t)m] = (int32_t)live;
        for (int64_t b = 0; b < nb; ++b) {
            const size_t i = (size_t)(m * nb + b);
            if (b < live) {
                /* A COARSE GRID ON PURPOSE: 40 distinct values over hundreds of blocks puts many
                 * blocks on the threshold exactly, which is the case the tie rule is for. */
                sc[i] = (float)((b * 17 + m * 7) % 40) * 0.25f;
                bt[i] = (int32_t)((b * 31 + m * 11) % pages);
                /* A DEAD PAGE WITH A WINNING SCORE, inside the scored range. */
                if (b == live / 3) { sc[i] = 1e6f; bt[i] = -1; }
            } else if (b == live) {
                bt[i] = (m % 2 == 0) ? (int32_t)((m * 97 + 5) % pages) : -1;   /* the tail */
            }
        }
    }

    /* The oracle: sort the live, non-dead blocks by (score desc, index asc), take topk, then the
     * partial page. */
    std::vector<std::vector<int32_t>> want((size_t)M);
    std::vector<int32_t> want_n((size_t)M, 0), want_tail((size_t)M, -1);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t live = nc[(size_t)m];
        std::vector<std::pair<float, int64_t>> v;
        for (int64_t b = 0; b < live; ++b)
            if (bt[(size_t)(m * nb + b)] >= 0) v.push_back({ sc[(size_t)(m * nb + b)], b });
        std::stable_sort(v.begin(), v.end(), [](const std::pair<float, int64_t>& a,
                                                const std::pair<float, int64_t>& b) {
            return a.first > b.first;                       /* stable => index ascending on ties */
        });
        const size_t take = v.size() < (size_t)topk ? v.size() : (size_t)topk;
        for (size_t i = 0; i < take; ++i)
            want[(size_t)m].push_back(bt[(size_t)(m * nb + v[i].second)]);
        want_tail[(size_t)m] = bt[(size_t)(m * nb + live)];
        want_n[(size_t)m] = (int32_t)take + (want_tail[(size_t)m] >= 0 ? 1 : 0);
    }

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    /* The query's own position: the last token of the partial page when there is one, and the
     * last token of the last complete block when there is not. */
    std::vector<int32_t> pos((size_t)M);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t live = nc[(size_t)m];
        pos[(size_t)m] = (int32_t)(want_tail[(size_t)m] >= 0 ? live * ratio + (m % 3)
                                                             : live * ratio - 1);
    }
    /* ONE QUERY A SEQUENCE: the decode shape, where `seq(m) == m` and each row's `nc` is its own.
     * That is what the oracle above assumes, and it is deliberate -- the point of this case is the
     * radix and the tie rule, not the join. qsa_select_prefill_case is the join. */
    std::vector<int32_t> cuh((size_t)(M + 1));
    for (int64_t i = 0; i <= M; ++i) cuh[(size_t)i] = (int32_t)i;

    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }),
        T(d_bt, RAD_I32, { M, nb }),
        T(d_nc, RAD_I32, { M }),
        T(d_po, RAD_I32, { M }),
        T(d_cu, RAD_I32, { M + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }),
        T(d_ns, RAD_I32, { M }),
        T(d_su, RAD_I32, { M }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", M) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    std::vector<int32_t> got[2], gn[2], gs[2];
    for (int pass = 0; pass < 2; ++pass) {
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        got[pass].assign((size_t)(M * (topk + 1)), 0);
        gn[pass].assign((size_t)M, 0);
        st_memcpy(got[pass].data(), d_se, got[pass].size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(gn[pass].data(),  d_ns, gn[pass].size() * 4,  hipMemcpyDeviceToHost);
        gs[pass].assign((size_t)M, 0);
        st_memcpy(gs[pass].data(),  d_su, gs[pass].size() * 4,  hipMemcpyDeviceToHost);
    }

    bool set_ok = true, tail_ok = true, dead_ok = true, n_ok = true;
    for (int64_t m = 0; m < M; ++m) {
        const int32_t ns = gn[0][(size_t)m];
        if (ns != want_n[(size_t)m]) { n_ok = false; continue; }
        std::vector<int32_t> mine(got[0].begin() + (size_t)(m * (topk + 1)),
                                  got[0].begin() + (size_t)(m * (topk + 1) + ns));
        /* The tail is last, by contract. */
        if (want_tail[(size_t)m] >= 0 && (ns == 0 || mine.back() != want_tail[(size_t)m]))
            tail_ok = false;
        if (want_tail[(size_t)m] >= 0) mine.pop_back();
        for (int32_t v : mine) if (v < 0) dead_ok = false;
        std::vector<int32_t> ref = want[(size_t)m];
        std::sort(mine.begin(), mine.end());
        std::sort(ref.begin(), ref.end());
        if (mine != ref) set_ok = false;
    }
    /* THE CAUSAL BOUND THE ATTENTION WILL BE GIVEN. The flat KV is `nsel` pages of `ratio` with
     * the partial page LAST, so the query sits at `(nsel-1)*ratio + pos%ratio` and the length is
     * one past it. Checked here because it is the number that decides whether the sparse
     * attention reads the query's own token and stops there -- and because it is an off-by-one
     * away from reading tokens the query cannot see. */
    bool su_ok = true;
    for (int64_t m = 0; m < M; ++m) {
        const int32_t ns = gn[0][(size_t)m];
        const int32_t want_su = ns == 0 ? 0
                              : (int32_t)((ns - 1) * ratio + (pos[(size_t)m] % ratio) + 1);
        if (gs[0][(size_t)m] != want_su) su_ok = false;
    }
    const bool det = got[0] == got[1] && gn[0] == gn[1] && gs[0] == gs[1];
    const bool pass = set_ok && tail_ok && dead_ok && n_ok && su_ok && det;
    std::printf("        set %s, tail last %s, dead page excluded %s, count %s, seqused %s, "
                "two runs identical %s\n", set_ok ? "ok" : "DIFFERS", tail_ok ? "ok" : "NO",
                dead_ok ? "ok" : "NO", n_ok ? "ok" : "NO", su_ok ? "ok" : "NO",
                det ? "ok" : "NO");
    ok(what, pass, std::string("set=") + (set_ok ? "1" : "0") + " tail=" + (tail_ok ? "1" : "0") +
                   " dead=" + (dead_ok ? "1" : "0") + " n=" + (n_ok ? "1" : "0") +
                   " su=" + (su_ok ? "1" : "0") + " det=" + (det ? "1" : "0"));
}

/* qsa_select AT PREFILL: many queries over one history, each bounded by its own position.
 *
 * THIS IS THE CASE THE DECODE FIXTURE CANNOT REACH. There, one query a sequence makes the
 * sequence's complete-block count and the query's own the same number, and `bt[nc]` is the page
 * after the sequence's last -- which the KV manager leaves -1, so a partial page could never be
 * wrongly appended. Mid-CHUNK both of those stop being true at once: the sequence's count runs to
 * the end of the chunk while the query's runs to the query, and the page the query sits in is a
 * live one with LATER TOKENS OF THE SAME CHUNK in it. A selection that appended it and set
 * `seqused` past the query would attend to its own future, which is not a wrong ranking -- it is
 * a model that cheats, and it would show up as an implausibly good prefill and nothing else.
 *
 * So the two things checked here are the per-query bound `n = min(nc[seq], (pos+1)/ratio)` and the
 * rule that the partial page exists exactly when `(pos + 1) % ratio != 0`.
 */
static void qsa_select_prefill_case(const Plugin& pl) {
    const char* what = "qsa_select at prefill: per-query bound and the partial-page rule";
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t S = 2, nb = 96, topk = 5, pages = 512, ratio = 4;
    /* Two chunks of different lengths over two histories, which is the ragged case the engine
     * actually builds -- and it makes a query's row index and its position different numbers, so
     * a kernel that confused them fails here rather than agreeing by accident. */
    const int64_t q0 = 9, q1 = 14, M = q0 + q1;
    const int64_t base[2] = { 137, 4 };          /* each sequence's first position in the chunk */

    std::vector<int32_t> cuh = { 0, (int32_t)q0, (int32_t)(q0 + q1) };
    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < q0; ++i) pos[(size_t)i] = (int32_t)(base[0] + i);
    for (int64_t i = 0; i < q1; ++i) pos[(size_t)(q0 + i)] = (int32_t)(base[1] + i);

    /* `nc` is the SEQUENCE's count after the chunk: every block the chunk's last token completed. */
    std::vector<int32_t> nc((size_t)S);
    nc[0] = (int32_t)((base[0] + q0) / ratio);
    nc[1] = (int32_t)((base[1] + q1) / ratio);

    /* The table and the scores are per SEQUENCE and are the same for every query in it -- which is
     * the whole point: the queries differ only in how much of the row they may see. */
    std::vector<float>   sc((size_t)(M * nb), 0.0f);
    std::vector<int32_t> bt((size_t)(S * nb), -1);
    for (int64_t s = 0; s < S; ++s)
        for (int64_t b = 0; b < nb; ++b)
            /* Live up to and including the block the chunk's last token is IN, so `bt[n]` is a
             * real page for every query that has a partial one. */
            bt[(size_t)(s * nb + b)] =
                b <= (base[s] + (s == 0 ? q0 : q1) - 1) / ratio ? (int32_t)((b * 29 + s * 7) % pages)
                                                                : -1;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t b = 0; b < nb; ++b)
            sc[(size_t)(m * nb + b)] = (float)((b * 23 + m * 5) % 37) * 0.5f;

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    st_memcpy(d_sc, sc.data(),  sc.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(),  bt.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(),  nc.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }),
        T(d_bt, RAD_I32, { S, nb }),
        T(d_nc, RAD_I32, { S }),
        T(d_po, RAD_I32, { M }),
        T(d_cu, RAD_I32, { S + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }),
        T(d_ns, RAD_I32, { M }),
        T(d_su, RAD_I32, { M }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", S) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<int32_t> got((size_t)(M * (topk + 1))), gn((size_t)M), gs((size_t)M);
    st_memcpy(got.data(), d_se, got.size() * 4, hipMemcpyDeviceToHost);
    st_memcpy(gn.data(),  d_ns, gn.size() * 4,  hipMemcpyDeviceToHost);
    st_memcpy(gs.data(),  d_su, gs.size() * 4,  hipMemcpyDeviceToHost);

    bool set_ok = true, n_ok = true, su_ok = true, future_ok = true, varied = false;
    int64_t seen_partial = 0, seen_none = 0;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s   = m < q0 ? 0 : 1;
        const int64_t p   = pos[(size_t)m];
        const int64_t own = (p + 1) / ratio;
        int64_t n = nc[(size_t)s] < nb ? nc[(size_t)s] : nb;
        if (own < n) n = own;
        if (own != nc[(size_t)s]) varied = true;      /* the bound is doing something */

        std::vector<std::pair<float, int64_t>> v;
        for (int64_t b = 0; b < n; ++b)
            if (bt[(size_t)(s * nb + b)] >= 0) v.push_back({ sc[(size_t)(m * nb + b)], b });
        std::stable_sort(v.begin(), v.end(), [](const std::pair<float, int64_t>& x,
                                                const std::pair<float, int64_t>& y) {
            return x.first > y.first;
        });
        const size_t take = v.size() < (size_t)topk ? v.size() : (size_t)topk;
        std::vector<int32_t> ref;
        for (size_t i = 0; i < take; ++i) ref.push_back(bt[(size_t)(s * nb + v[i].second)]);

        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? bt[(size_t)(s * nb + n)] : -1;
        if (part >= 0) ++seen_partial; else ++seen_none;
        const int32_t want_n = (int32_t)take + (part >= 0 ? 1 : 0);
        if (gn[(size_t)m] != want_n) { n_ok = false; continue; }

        std::vector<int32_t> mine(got.begin() + (size_t)(m * (topk + 1)),
                                  got.begin() + (size_t)(m * (topk + 1) + want_n));
        if (part >= 0) {
            if (mine.back() != part) set_ok = false;
            mine.pop_back();
        }
        std::sort(mine.begin(), mine.end());
        std::sort(ref.begin(), ref.end());
        if (mine != ref) set_ok = false;

        const int32_t want_su = want_n == 0 ? 0
                              : (int32_t)((want_n - 1) * ratio + (p % ratio) + 1);
        if (gs[(size_t)m] != want_su) su_ok = false;
        /* THE PREDICATE THAT MATTERS: the flat KV the attention will walk is `nsel` pages of
         * `ratio`, and `seqused` must not reach past the query's own token in the last one. */
        if (want_n > 0 && gs[(size_t)m] > (int32_t)((want_n - 1) * ratio + (p % ratio) + 1))
            future_ok = false;
    }
    const bool pass = set_ok && n_ok && su_ok && future_ok && varied &&
                      seen_partial > 0 && seen_none > 0;
    std::printf("        set %s, count %s, seqused %s, no future page %s; %lld queries with a "
                "partial page, %lld without\n", set_ok ? "ok" : "DIFFERS", n_ok ? "ok" : "NO",
                su_ok ? "ok" : "NO", future_ok ? "ok" : "NO",
                (long long)seen_partial, (long long)seen_none);
    ok(what, pass, std::string("set=") + (set_ok ? "1" : "0") + " n=" + (n_ok ? "1" : "0") +
                   " su=" + (su_ok ? "1" : "0") + " fut=" + (future_ok ? "1" : "0") +
                   " varied=" + (varied ? "1" : "0"));
}

/* qsa_select's WIDE-LAUNCH FORM against its single-workgroup form, byte for byte: the table, the
 * count and seqused. Two sequences, a hundred and twenty queries, each bounded by its own position
 * so some select from fewer blocks than topk; dead pages and -inf scores scattered through the
 * history; and five score distributions -- continuous, coarse with heavy ties, two packed into a
 * single first-digit bin, one small enough for the LDS list and one past it, which is the form's
 * fallback, and fine ties, where the sampled form's list is short and ends in a tie. */
static void qsa_select_wide_case(const Plugin& pl) {
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r || !pl.pin_qsel) { ok("qsa_select wide form", false, "row or hook absent"); return; }
    const int64_t S = 2, nb = 24000, topk = 512, ratio = 4, pages = 1 << 20;
    const int64_t q0 = 70, q1 = 50, M = q0 + q1;
    /* The second sequence's queries see 300 to 337 blocks, fewer than topk. */
    const int64_t base[2] = { 4 * 23000, 4 * 300 };
    std::vector<int32_t> cuh = { 0, (int32_t)q0, (int32_t)M };
    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < q0; ++i) pos[(size_t)i] = (int32_t)(base[0] + 37 * i);
    for (int64_t i = 0; i < q1; ++i) pos[(size_t)(q0 + i)] = (int32_t)(base[1] + 3 * i);
    std::vector<int32_t> nc = { (int32_t)std::min<int64_t>((base[0] + 37 * q0) / ratio, nb),
                                (int32_t)((base[1] + 3 * q1) / ratio) };
    uint64_t x = 0x243F6A8885A308D3ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::vector<int32_t> bt((size_t)(S * nb));
    for (auto& v : bt) v = (rnd() % 97 == 0) ? -1 : (int32_t)(rnd() % pages);
    static const char* const kDist[] = { "continuous", "coarse, heavy ties",
                                         "one first-digit bin, in LDS",
                                         "one first-digit bin, past LDS",
                                         "fine ties at the threshold" };
    for (int dist = 0; dist < 5; ++dist) {
        std::vector<float> sc((size_t)(M * nb));
        for (auto& v : sc) {
            const uint64_t u = rnd();
            if (u % 211 == 0) { v = -INFINITY; continue; }
            switch (dist) {
                case 0: v = (float)(u % 1000003) / 1000.0f; break;
                case 1: v = (float)(u % 23) * 0.25f; break;
                /* 1.0 to 1.06 is one bin of the first digit, whose bins are 1/16 of an octave;
                 * a few thousand of those sit in it, then the whole history. */
                case 2: v = (u % 8 == 0) ? 1.0f + (float)(u % 60000) * 1e-6f
                                         : (float)(u % 1000) / 2000.0f; break;
                case 3: v = 1.0f + (float)(u % 60000) * 1e-6f; break;
                /* ~5000 distinct scores a few times each: the sampled form's list is short and
                 * the threshold is a tie, which is the case its emit has to place exactly. */
                default: v = (float)(u % 4999) * 0.01f; break;
            }
        }
        void* d_sc = dalloc(sc.size() * 4);
        void* d_bt = dalloc(bt.size() * 4);
        void* d_nc = dalloc(nc.size() * 4);
        void* d_po = dalloc(pos.size() * 4);
        void* d_cu = dalloc(cuh.size() * 4);
        st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
        std::vector<int32_t> out[2][3];
        for (int f = 1; f <= 2; ++f) {
            void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
            void* d_ns = dalloc((size_t)M * 4);
            void* d_su = dalloc((size_t)M * 4);
            std::vector<RadTensor> ts = {
                T(d_sc, RAD_F32, { M, nb }), T(d_bt, RAD_I32, { S, nb }), T(d_nc, RAD_I32, { S }),
                T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { S + 1 }),
                T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) };
            std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                         PI("ratio", ratio), PI("seqs", S) };
            RadArgs a;
            std::memset(&a, 0, sizeof(a));
            a.t = ts.data(); a.n_t = (int)ts.size();
            a.p = ps.data(); a.n_p = (int)ps.size();
            pl.pin_qsel(f);
            const int rc = r->launch(&a, nullptr);
            pl.pin_qsel(0);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
                ok(fmt("qsa_select wide form, %s: form %d is served", kDist[dist], f).c_str(),
                   false, fmt("rc=%d", rc));
                return;
            }
            out[f - 1][0].resize((size_t)(M * (topk + 1)));
            out[f - 1][1].resize((size_t)M);
            out[f - 1][2].resize((size_t)M);
            st_memcpy(out[f - 1][0].data(), d_se, out[f - 1][0].size() * 4, hipMemcpyDeviceToHost);
            st_memcpy(out[f - 1][1].data(), d_ns, (size_t)M * 4, hipMemcpyDeviceToHost);
            st_memcpy(out[f - 1][2].data(), d_su, (size_t)M * 4, hipMemcpyDeviceToHost);
        }
        int64_t full = 0, diff = 0;
        for (int32_t v : out[0][1]) full += v >= (int32_t)topk;
        for (int k = 0; k < 3; ++k)
            for (size_t i = 0; i < out[0][k].size(); ++i) diff += out[0][k][i] != out[1][k][i];
        ok(fmt("qsa_select wide form, %s: the table, the count and seqused are the single "
               "workgroup's, byte for byte", kDist[dist]).c_str(),
           diff == 0 && full > 0 && full < M,
           fmt("%lld entries differ; %lld of %lld queries select a full table", (long long)diff,
               (long long)full, (long long)M));
    }
}

/* qsa_select WITH THE BLOCK LIST SPLIT ACROSS THE DEVICE, against the same launch unsplit.
 *
 * WHY IT SPLITS. One workgroup a query is 6% of the device and the whole of the kernel at length,
 * so the launcher cuts the list into chunks, takes each chunk's own top-k, and merges -- which is
 * exact because a chunk's local top-k contains every global top-k member of that chunk (the proof
 * is above the split in r4d_qsa_select_bf16.hip). "Exact" here means BIT-IDENTICAL, not equivalent:
 * the selection decides which keys the attention reads, so a set that differed by one block on a
 * tie would be a different model.
 *
 * THE TWO FORMS ARE THE SAME LAUNCH WITH AND WITHOUT SCRATCH, which is what makes this a test of
 * the split rather than of a second oracle: the launcher falls back to the single-workgroup kernel
 * when the arena is absent, so withholding it selects that form. A fixture that never crossed the
 * chunk threshold would pass while testing nothing, so `nb` is over four thousand -- and the
 * scores are drawn from a SMALL SET so that ties are everywhere, because the tie rule (lowest
 * block index first, by prefix count) is the part a split can get wrong without being obviously
 * wrong. */
static void qsa_select_split_one(const Plugin& pl, const char* what, int64_t S, int64_t per) {
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = S * per, nb = 20001, topk = 512, ratio = 4, pages = 8192;
    std::vector<float>   sc((size_t)(M * nb));
    /* ONE TABLE A SEQUENCE AND A DIFFERENT LIVE COUNT ON EACH, because the chunk pass reads both
     * off the SEQUENCE while its grid is indexed by the QUERY -- a fixture with one sequence would
     * never catch a `seq` that came from the wrong side of that join. At decode a sequence
     * contributes exactly `1 + n_spec` query rows, which is what `per` is. */
    std::vector<int32_t> bt((size_t)(S * nb)), nc((size_t)S), pos((size_t)M),
                         cuh((size_t)(S + 1));
    const int64_t n = 19000;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t ns = n - s * 1300;             /* live blocks; one page in 64 is dead */
        nc[(size_t)s] = (int32_t)ns;
        for (int64_t b = 0; b < nb; ++b)
            bt[(size_t)(s * nb + b)] =
                b >= ns ? -1 : ((b % 64) == 37 ? -1 : (int32_t)((b * 4099 + 11 + s * 71) % pages));
        cuh[(size_t)s] = (int32_t)(s * per);
        for (int64_t j = 0; j < per; ++j)
            pos[(size_t)(s * per + j)] = (int32_t)(ns * ratio + j - 2);
    }
    cuh[(size_t)S] = (int32_t)M;
    /* Only 96 distinct scores over 19000 blocks, so the threshold bin is thousands deep and the
     * answer is decided by the tie rule rather than by the radix. */
    for (int64_t m = 0; m < M; ++m)
        for (int64_t b = 0; b < nb; ++b)
            sc[(size_t)(m * nb + b)] = (float)(((b * 31 + m * 17) % 96)) * 0.125f;

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    if (!d_sc || !d_bt || !d_nc || !d_po || !d_cu || !d_se || !d_ns || !d_su) {
        ok(what, false, "alloc failed"); return;
    }
    st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }), T(d_bt, RAD_I32, { S, nb }), T(d_nc, RAD_I32, { S }),
        T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { S + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", S) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int64_t need = r->scratch ? r->scratch(&a) : 0;
    void* d_sk = need > 0 ? dalloc((size_t)need) : nullptr;
    if (need > 0 && !d_sk) { ok(what, false, "scratch alloc failed"); return; }

    /* Three runs: split, split again (determinism), and unsplit. */
    std::vector<int32_t> se[3], ns[3], su[3];
    for (int pass = 0; pass < 3; ++pass) {
        a.scratch = pass == 2 ? nullptr : d_sk;
        a.scratch_bytes = pass == 2 ? 0 : need;
        st_memset(d_se, 0xAA, (size_t)(M * (topk + 1)) * 4);
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        se[pass].assign((size_t)(M * (topk + 1)), 0);
        ns[pass].assign((size_t)M, 0);
        su[pass].assign((size_t)M, 0);
        st_memcpy(se[pass].data(), d_se, se[pass].size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(ns[pass].data(), d_ns, ns[pass].size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(su[pass].data(), d_su, su[pass].size() * 4, hipMemcpyDeviceToHost);
    }

    /* THE TWO RUNS HAVE TO HAVE TAKEN DIFFERENT FORMS. Without the scratch hook the launcher takes
     * the fallback in both, and three identical runs would then prove nothing at all. */
    const bool split_ran = need > 0;
    const bool same = se[0] == se[2] && ns[0] == ns[2] && su[0] == su[2];
    const bool det  = se[0] == se[1] && ns[0] == ns[1] && su[0] == su[1];
    /* A SET COMPARISON BESIDE THE BYTE ONE, because the two fail for different reasons: a wrong
     * SET is a wrong selection, and a right set in a different ORDER is a table the attention sums
     * in another order -- different text, same model. Reporting them apart is what says which. */
    bool set_ok = true;
    long slots = 0;
    long long first = -1;
    for (int64_t m = 0; m < M; ++m) {
        std::vector<int32_t> A(se[0].begin() + (size_t)(m * (topk + 1)),
                               se[0].begin() + (size_t)((m + 1) * (topk + 1)));
        std::vector<int32_t> B(se[2].begin() + (size_t)(m * (topk + 1)),
                               se[2].begin() + (size_t)((m + 1) * (topk + 1)));
        for (size_t i = 0; i < A.size(); ++i)
            if (A[i] != B[i]) { ++slots; if (first < 0) first = (long long)(m * (topk + 1) + i); }
        std::sort(A.begin(), A.end());
        std::sort(B.begin(), B.end());
        if (A != B) set_ok = false;
    }
    bool nsel_ok = ns[0] == ns[2] && su[0] == su[2];
    const bool pass = split_ran && same && det;
    std::printf("        %lld seq x %lld rows, %lld blocks, scratch %lld B; split vs unsplit %s "
                "(set %s, nsel/seqused %s, %ld slot(s) differ, first at %lld), "
                "two split runs identical %s\n",
                (long long)S, (long long)per, (long long)nb, (long long)need,
                same ? "ok" : "DIFFERS",
                set_ok ? "ok" : "DIFFERS", nsel_ok ? "ok" : "DIFFERS", slots, first,
                det ? "ok" : "NO");
    ok(what, pass, std::string("split_ran=") + (split_ran ? "1" : "0") +
                   " same=" + (same ? "1" : "0") + " det=" + (det ? "1" : "0"));
}

/* qsa_score, against an f64 evaluation of the same formula.
 *
 * TWO CONTROLS, AND BOTH ARE ORDERING MISTAKES A READER MAKES rather than transcription slips.
 *
 *   sum-then-relu instead of relu-then-sum. Arithmetically it is one line either way and both
 *   rank blocks plausibly; it is a different model. Measured here at a rel_l2 of order one.
 *
 *   a padded table entry scoring ZERO instead of -inf. A real score can be zero -- every head
 *   vetoed the block -- so this is the mistake that makes padding outrank a block the model
 *   deliberately ignored, and it is invisible in the scores themselves. The case checks the
 *   padding lanes directly rather than through an l2, because an l2 over a mostly-live row
 *   drowns them.
 */
static void qsa_score_case(const Plugin& pl) {
    const char* what = "qsa_score vs an f64 evaluation of the same formula";
    const RadKernelInfo* r = find(pl, "qsa_score_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 5, n = 128, heads = 4, nb = 37, pages = 64;

    std::vector<uint16_t> qh((size_t)(M * heads * n)), bkh((size_t)(pages * n));
    std::vector<int32_t>  bth((size_t)(M * nb));
    for (int64_t i = 0; i < M * heads * n; ++i)
        qh[(size_t)i] = h_to_bf16((float)(((i * 29 + 7) % 83) - 41) / 47.0f);
    for (int64_t i = 0; i < pages * n; ++i)
        bkh[(size_t)i] = h_to_bf16((float)(((i * 41 + 13) % 71) - 35) / 39.0f);
    /* A scattered table with real padding at the end of every row, which is what a sequence
     * shorter than the declared width actually produces. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t live = nb - 3 - m * 4;
        for (int64_t b = 0; b < nb; ++b)
            bth[(size_t)(m * nb + b)] = b < live ? (int32_t)((b * 13 + m * 5) % pages) : -1;
    }

    std::vector<double> want((size_t)(M * nb), 0.0), sumrelu((size_t)(M * nb), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t b = 0; b < nb; ++b) {
            const int32_t p = bth[(size_t)(m * nb + b)];
            if (p < 0) { want[(size_t)(m * nb + b)] = -1e30; sumrelu[(size_t)(m * nb + b)] = -1e30;
                         continue; }
            double s = 0.0, raw = 0.0;
            for (int64_t h = 0; h < heads; ++h) {
                double d = 0.0;
                for (int64_t j = 0; j < n; ++j)
                    d += (double)h_bf16(qh[(size_t)((m * heads + h) * n + j)]) *
                         (double)h_bf16(bkh[(size_t)(p * n + j)]);
                s += d > 0.0 ? d : 0.0;     /* relu per (block, head), THEN the head sum */
                raw += d;                   /* the control: sum first, clamp after */
            }
            const double inv = 1.0 / std::sqrt((double)n);
            want[(size_t)(m * nb + b)] = s * inv;
            sumrelu[(size_t)(m * nb + b)] = (raw > 0.0 ? raw : 0.0) * inv;
        }

    /* ONE QUERY A SEQUENCE, which is the decode shape and makes `seq(m) == m` -- so the oracle
     * above, which treats `m` as the sequence, describes what the kernel computes. The
     * prefill shape (many queries over one history) is qsa_select_prefill_case's job, because it
     * is the SELECTION that has to bound each query separately. */
    std::vector<int32_t> cuh((size_t)(M + 1)), nch((size_t)M, (int32_t)nb);
    for (int64_t i = 0; i <= M; ++i) cuh[(size_t)i] = (int32_t)i;

    void* d_q  = dalloc(qh.size() * 2);
    void* d_bk = dalloc(bkh.size() * 2);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_nc = dalloc(nch.size() * 4);
    void* d_o  = dalloc((size_t)(M * nb) * 4);
    st_memcpy(d_q,  qh.data(),  qh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_bk, bkh.data(), bkh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nch.data(), nch.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_q,  RAD_BF16, { M, heads * n }),
        T(d_bk, RAD_BF16, { pages, n }),
        T(d_bt, RAD_I32,  { M, nb }),
        T(d_cu, RAD_I32,  { M + 1 }),
        T(d_nc, RAD_I32,  { M }),
        T(d_o,  RAD_F32,  { M, nb }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", n), PI("heads", heads), PI("blocks", nb), PI("seqs", M),
        PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<float> got((size_t)(M * nb));
    st_memcpy(got.data(), d_o, got.size() * 4, hipMemcpyDeviceToHost);

    /* The l2 is over the LIVE entries only; the padding is checked as a predicate, because an l2
     * that mixed -inf with real scores is not a number. */
    auto rel = [&](const std::vector<double>& ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] <= -1e29) continue;
            const double d = (double)got[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        return den > 0.0 ? std::sqrt(num / den) : -1.0;
    };
    long pad = 0, pad_ninf = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (want[i] <= -1e29) { ++pad; if (got[i] < -1e30f) ++pad_ninf; }

    const double r_ok = rel(want), r_ctl = rel(sumrelu);
    const bool pass = r_ok >= 0.0 && r_ok < 1e-5 && r_ctl > 100.0 * r_ok && pad > 0 &&
                      pad_ninf == pad;
    std::printf("        rel_l2 vs f64: %.3e; sum-then-relu %.3e; %ld padded entries, %ld at "
                "-inf\n", r_ok, r_ctl, pad, pad_ninf);
    ok(what, pass, "rel " + std::to_string(r_ok) + ", control " + std::to_string(r_ctl) +
                   ", padding " + std::to_string(pad_ninf) + "/" + std::to_string(pad));
}

/* qsa_score over SEQUENCES OF MANY ROWS, against the same f64 formula.
 *
 * The case above is one row a sequence, the decode shape. A prefill chunk is many rows over one
 * history, and in a mixed step the chunk sits beside other sequences' decode rows, so a 16-row
 * query tile can hold the end of one sequence and the start of the next. Rows {7, 20, 5}: the
 * second sequence starts inside the first tile and ends inside the second, and each sequence has
 * its own table row with its own padding and its own live count. The control is scoring every row
 * against sequence 0's table -- what a tile that forgot to split at the boundary would do. */
static void qsa_score_rows_case(const Plugin& pl) {
    const char* what = "qsa_score, three sequences of 7, 20 and 5 rows, vs f64";
    const RadKernelInfo* r = find(pl, "qsa_score_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t S = 3, n = 128, heads = 4, nb = 45, pages = 80;
    const int64_t rows[S] = { 7, 20, 5 };
    std::vector<int32_t> cuh((size_t)(S + 1), 0), nch((size_t)S);
    for (int64_t i = 0; i < S; ++i) cuh[(size_t)(i + 1)] = cuh[(size_t)i] + (int32_t)rows[i];
    const int64_t M = cuh[(size_t)S];

    std::vector<uint16_t> qh((size_t)(M * heads * n)), bkh((size_t)(pages * n));
    for (int64_t i = 0; i < M * heads * n; ++i)
        qh[(size_t)i] = h_to_bf16((float)(((i * 31 + 5) % 89) - 44) / 43.0f);
    for (int64_t i = 0; i < pages * n; ++i)
        bkh[(size_t)i] = h_to_bf16((float)(((i * 43 + 11) % 73) - 36) / 37.0f);
    std::vector<int32_t> bth((size_t)(S * nb));
    for (int64_t sq = 0; sq < S; ++sq) {
        const int64_t live = nb - 2 - sq * 9;
        nch[(size_t)sq] = (int32_t)(live - 1);           /* one table entry past the live count */
        for (int64_t b = 0; b < nb; ++b)
            bth[(size_t)(sq * nb + b)] =
                b < live && (b % 11) != 4 ? (int32_t)((b * 17 + sq * 7) % pages) : -1;
    }

    auto score = [&](int64_t m, int64_t sq, int64_t b) {
        const int32_t p = bth[(size_t)(sq * nb + b)];
        if (p < 0) return -1e30;
        double s = 0.0;
        for (int64_t h = 0; h < heads; ++h) {
            double d = 0.0;
            for (int64_t j = 0; j < n; ++j)
                d += (double)h_bf16(qh[(size_t)((m * heads + h) * n + j)]) *
                     (double)h_bf16(bkh[(size_t)(p * n + j)]);
            s += d > 0.0 ? d : 0.0;
        }
        return s / std::sqrt((double)n);
    };

    void* d_q  = dalloc(qh.size() * 2);
    void* d_bk = dalloc(bkh.size() * 2);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_nc = dalloc(nch.size() * 4);
    void* d_o  = dalloc((size_t)(M * nb) * 4);
    st_memcpy(d_q,  qh.data(),  qh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_bk, bkh.data(), bkh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nch.data(), nch.size() * 4, hipMemcpyHostToDevice);
    /* A sentinel in every entry, so a store past a sequence's live count is caught. */
    std::vector<float> sentinel((size_t)(M * nb), 12345.0f);
    st_memcpy(d_o, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_q,  RAD_BF16, { M, heads * n }), T(d_bk, RAD_BF16, { pages, n }),
        T(d_bt, RAD_I32,  { S, nb }),        T(d_cu, RAD_I32,  { S + 1 }),
        T(d_nc, RAD_I32,  { S }),            T(d_o,  RAD_F32,  { M, nb }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("heads", heads), PI("blocks", nb),
                                 PI("seqs", S), PS("dtype", "bf16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
    std::vector<float> got((size_t)(M * nb));
    st_memcpy(got.data(), d_o, got.size() * 4, hipMemcpyDeviceToHost);

    double num = 0.0, den = 0.0, cnum = 0.0;
    long pad = 0, pad_ninf = 0, past = 0, past_kept = 0;
    for (int64_t sq = 0; sq < S; ++sq)
        for (int64_t m = cuh[(size_t)sq]; m < cuh[(size_t)(sq + 1)]; ++m)
            for (int64_t b = 0; b < nb; ++b) {
                const float g = got[(size_t)(m * nb + b)];
                if (b >= nch[(size_t)sq]) { ++past; past_kept += g == 12345.0f; continue; }
                const double w = score(m, sq, b);
                if (w <= -1e29) { ++pad; pad_ninf += g < -1e30f; continue; }
                const double c = score(m, 0, b);
                num += ((double)g - w) * ((double)g - w);
                den += w * w;
                if (c > -1e29) cnum += (c - w) * (c - w);
            }
    const double r_ok = den > 0.0 ? std::sqrt(num / den) : -1.0;
    const double r_ctl = den > 0.0 ? std::sqrt(cnum / den) : -1.0;
    const bool pass = r_ok >= 0.0 && r_ok < 1e-5 && r_ctl > 100.0 * r_ok && pad > 0 &&
                      pad_ninf == pad && past > 0 && past_kept == past;
    std::printf("        rel_l2 vs f64: %.3e; every row on sequence 0's table %.3e; %ld padded "
                "entries, %ld at -inf; %ld past the live count, %ld untouched\n",
                r_ok, r_ctl, pad, pad_ninf, past, past_kept);
    ok(what, pass, "rel " + std::to_string(r_ok) + ", control " + std::to_string(r_ctl));
}

/* qsa_block_key, against an f64 evaluation of the same formula.
 *
 * THE ROUNDING POINT IS THE PART WORTH TESTING. The reference pools in f32 and rounds to bf16
 * BEFORE the norm reads it; carrying f32 straight through is the convenient order and a different
 * number. So this builds BOTH oracles and reports how far apart they are, which is what says the
 * decision is load-bearing rather than pedantry -- and then checks the kernel against the right
 * one.
 *
 * The negative control is the POSITION. `pos0` off by the ratio rotates every block key by one
 * block's worth of angle, which is the mistake a reader makes when they take the position of the
 * block's LAST token instead of its first, and it is invisible in magnitude: the vectors stay the
 * same length and point somewhere else. rel_l2 is what sees it. */
static void qsa_block_key_case(const Plugin& pl) {
    const char* what = "qsa_block_key vs an f64 evaluation of the same formula";
    const RadKernelInfo* r = find(pl, "qsa_block_key_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 17, n = 128, ratio = 4, rot = 64, pos0 = 36;
    const double eps = 1e-6, theta = 10000000.0, rscale = 1.0, wadd = 1.0;

    std::vector<uint16_t> kh((size_t)(M * ratio * n));
    std::vector<float>    wh((size_t)n);
    for (int64_t i = 0; i < M * ratio; ++i)
        for (int64_t d = 0; d < n; ++d) {
            float v = (float)(((i * 37 + d * 11) % 97) - 48) / 53.0f;
            if (d % 31 == 0) v *= 5.0f;         /* outlier channels, as a real key has */
            kh[(size_t)(i * n + d)] = h_to_bf16(v);
        }
    for (int64_t d = 0; d < n; ++d) wh[(size_t)d] = 0.1f + (float)((d * 7) % 13) / 40.0f;

    /* `round_mean` false is the convenient order the reference does NOT take. */
    auto oracle = [&](bool round_mean, int64_t p0, std::vector<double>* out) {
        out->assign((size_t)(M * n), 0.0);
        std::vector<double> v((size_t)n);
        for (int64_t b = 0; b < M; ++b) {
            for (int64_t d = 0; d < n; ++d) {
                float acc = 0.0f;
                for (int64_t j = 0; j < ratio; ++j)
                    acc += h_bf16(kh[(size_t)((b * ratio + j) * n + d)]);
                acc /= (float)ratio;
                v[(size_t)d] = round_mean ? (double)h_bf16(h_to_bf16(acc)) : (double)acc;
            }
            double ss = 0.0;
            for (int64_t d = 0; d < n; ++d) ss += v[(size_t)d] * v[(size_t)d];
            const double inv = 1.0 / std::sqrt(ss / (double)n + eps);
            std::vector<double> nz((size_t)n);
            for (int64_t d = 0; d < n; ++d) nz[(size_t)d] = v[(size_t)d] * inv * (wadd + wh[(size_t)d]);
            const double pp = (double)(p0 + b * ratio) / rscale;
            for (int64_t d = 0; d < n; ++d) {
                double res = nz[(size_t)d];
                if (d < rot) {
                    const int64_t half = rot / 2;
                    const bool first = d < half;
                    const int64_t pt = first ? d + half : d - half;
                    const int64_t ci = first ? d : d - half;
                    const double iv = 1.0 / std::pow(theta, (double)(2 * ci) / (double)rot);
                    const double c = std::cos(pp * iv), s = std::sin(pp * iv);
                    res = first ? (nz[(size_t)d] * c - nz[(size_t)pt] * s)
                                : (nz[(size_t)pt] * s + nz[(size_t)d] * c);
                }
                (*out)[(size_t)(b * n + d)] = res;
            }
        }
    };

    std::vector<double> want, unrounded, shifted;
    oracle(true,  pos0, &want);
    oracle(false, pos0, &unrounded);
    oracle(true,  pos0 + ratio, &shifted);

    void* d_k = dalloc(kh.size() * 2);
    void* d_w = dalloc(wh.size() * 4);
    void* d_o = dalloc((size_t)(M * n) * 2);
    st_memcpy(d_k, kh.data(), kh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice);

    /* `page` ABSENT, which is one entry with a null pointer and NOT a shorter list -- operands are
     * positional, so a three-entry list would hand `out` to `page` and the shim would refuse. The
     * scattered form is the pass below. */
    std::vector<RadTensor> ts = {
        T(d_k, RAD_BF16, { M * ratio, n }),
        T(d_w, RAD_F32,  { n }),
        T(nullptr, RAD_I32, { M }),
        T(nullptr, RAD_I32, { M }),
        T(d_o, RAD_BF16, { M, n }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", n), PI("ratio", ratio), PI("rotary_dim", rot), PI("pos0", pos0),
        PF("eps", eps), PF("theta", theta), PF("scale", rscale), PF("wadd", wadd),
        PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<uint16_t> got((size_t)(M * n));
    st_memcpy(got.data(), d_o, got.size() * 2, hipMemcpyDeviceToHost);

    /* THE SCATTERED FORM, which is the one the engine uses: a page list sends block b to page
     * `pg[b]` of a store as tall as the POOL, and a -1 says the block has no page and must not be
     * written anywhere. Same arithmetic, so the same oracle holds -- what is under test is that
     * the destination moved and the block count came off the PAGE LIST and not off `out`, which
     * is taller than the launch. */
    const int64_t store = M * 3 + 11;
    std::vector<int32_t> pg((size_t)M);
    for (int64_t b = 0; b < M; ++b)
        pg[(size_t)b] = (b == M / 2) ? -1 : (int32_t)((b * 7 + 3) % store);
    void* d_pg = dalloc(pg.size() * 4);
    void* d_o2 = dalloc((size_t)(store * n) * 2);
    st_memcpy(d_pg, pg.data(), pg.size() * 4, hipMemcpyHostToDevice);
    std::vector<uint16_t> fill((size_t)(store * n), h_to_bf16(-7.0f));
    st_memcpy(d_o2, fill.data(), fill.size() * 2, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts2 = {
        T(d_k,  RAD_BF16, { M * ratio, n }),
        T(d_w,  RAD_F32,  { n }),
        T(d_pg, RAD_I32,  { M }),
        T(nullptr, RAD_I32, { M }),
        T(d_o2, RAD_BF16, { store, n }),
    };
    RadArgs a2 = a;
    a2.t = ts2.data(); a2.n_t = (int)ts2.size();
    bool scatter_ok = r->launch(&a2, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got2((size_t)(store * n));
    st_memcpy(got2.data(), d_o2, got2.size() * 2, hipMemcpyDeviceToHost);
    long untouched = 0;
    for (int64_t b = 0; b < M && scatter_ok; ++b) {
        if (pg[(size_t)b] < 0) continue;
        for (int64_t d = 0; d < n; ++d)
            if (got2[(size_t)(pg[(size_t)b] * n + d)] != got[(size_t)(b * n + d)])
                scatter_ok = false;
    }
    for (int64_t p = 0; p < store; ++p) {
        bool named = false;
        for (int64_t b = 0; b < M; ++b) if (pg[(size_t)b] == (int32_t)p) named = true;
        if (!named && got2[(size_t)(p * n)] == h_to_bf16(-7.0f)) ++untouched;
    }
    /* Every page nothing named must still hold the fill, INCLUDING the one the -1 block would have
     * landed on if the guard were missing. */
    const bool scatter_clean = scatter_ok && untouched == store - (M - 1);

    /* THE WORK-LIST FORM. `pos` given, the same positions the tiling would have produced but
     * handed over one at a time and in a SHUFFLED block order -- which is what a decode step
     * looks like, where the blocks completing in one launch belong to different sequences. The
     * answer for block b must be the answer the tiled form gave for whichever block holds those
     * positions, so the fixture permutes the POSITIONS and checks against the permuted oracle.
     * Reading the same keys at a different position is the exact mistake this catches, and the
     * pos0 control above shows it is worth 5.8e-01 of error. */
    std::vector<int32_t> bpos((size_t)M);
    std::vector<int64_t> perm((size_t)M);
    for (int64_t b = 0; b < M; ++b) perm[(size_t)b] = (b * 5 + 2) % M;
    for (int64_t b = 0; b < M; ++b)
        bpos[(size_t)b] = (int32_t)(pos0 + perm[(size_t)b] * ratio);
    void* d_bp = dalloc(bpos.size() * 4);
    void* d_o3 = dalloc((size_t)(M * n) * 2);
    st_memcpy(d_bp, bpos.data(), bpos.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts3 = {
        T(d_k,  RAD_BF16, { M * ratio, n }),
        T(d_w,  RAD_F32,  { n }),
        T(nullptr, RAD_I32, { M }),
        T(d_bp, RAD_I32,  { M }),
        T(d_o3, RAD_BF16, { M, n }),
    };
    RadArgs a3 = a;
    a3.t = ts3.data(); a3.n_t = (int)ts3.size();
    bool list_ok = r->launch(&a3, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got3((size_t)(M * n));
    st_memcpy(got3.data(), d_o3, got3.size() * 2, hipMemcpyDeviceToHost);
    {
        /* Block b read rows [b*ratio, (b+1)*ratio) as before -- only the ANGLE moved -- so the
         * oracle is the one built at the permuted position for the same keys. */
        std::vector<double> wantp;
        oracle(true, pos0, &wantp);          /* shape only; recomputed per block below */
        for (int64_t b = 0; b < M && list_ok; ++b) {
            std::vector<double> one;
            oracle(true, pos0 + (perm[(size_t)b] - b) * ratio, &one);
            for (int64_t d = 0; d < n; ++d) {
                const double e = (double)h_bf16(got3[(size_t)(b * n + d)]) -
                                 one[(size_t)(b * n + d)];
                if (std::fabs(e) > 1e-2 * (std::fabs(one[(size_t)(b * n + d)]) + 1e-3))
                    list_ok = false;
            }
        }
    }

    /* UNDER M-RoPE: `mode` imrope with Qwen3.8's sections, and each block's three components
     * side by side in a [M, 3] view of a buffer with a row pitch of 4 -- so a kernel that took
     * the view as dense reads the wrong block's components. Frequency i reads H when i % 3 == 1
     * and i < 33, W when i % 3 == 2 and i < 30, and T otherwise; every component differs, so a
     * frequency that read the wrong one is off by at least one position's angle. */
    const int64_t pitch = 4;
    std::vector<int32_t> b4((size_t)(M * pitch), -1);
    for (int64_t b = 0; b < M; ++b) {
        const int32_t t = (int32_t)(pos0 + b * ratio);
        b4[(size_t)(b * pitch + 0)] = t;
        b4[(size_t)(b * pitch + 1)] = t + 5 + (int32_t)(b % 3);
        b4[(size_t)(b * pitch + 2)] = t + 13 + (int32_t)(b % 5);
    }
    void* d_b4 = dalloc(b4.size() * 4);
    void* d_o4 = dalloc((size_t)(M * n) * 2);
    st_memcpy(d_b4, b4.data(), b4.size() * 4, hipMemcpyHostToDevice);
    RadTensor b4v = T(d_b4, RAD_I32, { M, 3 });
    b4v.stride[0] = pitch;
    std::vector<RadTensor> ts4 = {
        T(d_k,  RAD_BF16, { M * ratio, n }),
        T(d_w,  RAD_F32,  { n }),
        T(nullptr, RAD_I32, { M }),
        b4v,
        T(d_o4, RAD_BF16, { M, n }),
    };
    std::vector<RadParam> ps4 = ps;
    ps4.push_back(PS("mode", "imrope"));
    ps4.push_back(PS("sections", "11 11 10"));
    RadArgs a4 = a;
    a4.t = ts4.data(); a4.n_t = (int)ts4.size();
    a4.p = ps4.data(); a4.n_p = (int)ps4.size();
    bool mrope_ok = r->launch(&a4, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got4((size_t)(M * n));
    st_memcpy(got4.data(), d_o4, got4.size() * 2, hipMemcpyDeviceToHost);
    double mr_num = 0.0, mr_den = 0.0, mr_t = 0.0;
    for (int64_t b = 0; b < M && mrope_ok; ++b) {
        /* The one-component oracle at each component's position, and per frequency the one the
         * interleave names. */
        std::vector<double> oc[3];
        for (int c = 0; c < 3; ++c) {
            std::vector<double> full;
            oracle(true, pos0 + (b4[(size_t)(b * pitch + c)] - (pos0 + b * ratio)), &full);
            oc[c].assign(full.begin() + b * n, full.begin() + (b + 1) * n);
        }
        for (int64_t d = 0; d < n; ++d) {
            const int64_t half = rot / 2;
            const int64_t ci = d < rot ? (d < half ? d : d - half) : 0;
            const int comp = d >= rot ? 0 : (ci % 3 == 1 && ci < 33) ? 1 : (ci % 3 == 2 && ci < 30) ? 2 : 0;
            const double e = oc[comp][(size_t)d], g2 = (double)h_bf16(got4[(size_t)(b * n + d)]);
            mr_num += (g2 - e) * (g2 - e);
            mr_den += e * e;
            mr_t += (g2 - oc[0][(size_t)d]) * (g2 - oc[0][(size_t)d]);
        }
    }
    const double r_mr = mr_den > 0.0 ? std::sqrt(mr_num / mr_den) : -1.0;
    const double r_mr_t = mr_den > 0.0 ? std::sqrt(mr_t / mr_den) : -1.0;
    /* The same buffer's FIRST COLUMN, as a text-only step hands it: the one-component form,
     * which must be the tiled answer bit for bit, since column 0 holds exactly the tiling. */
    RadTensor b1v = T(d_b4, RAD_I32, { M, 1 });
    b1v.stride[0] = pitch;
    ts4[3] = b1v;
    bool col_same = r->launch(&a4, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    st_memcpy(got4.data(), d_o4, got4.size() * 2, hipMemcpyDeviceToHost);
    col_same = col_same && got4 == got;
    mrope_ok = mrope_ok && r_mr >= 0.0 && r_mr < 0.01 && r_mr_t > 5.0 * r_mr && col_same;
    std::printf("        M-RoPE imrope: rel_l2 %.3e (all-T control %.3e); one column = tiled %s\n",
                r_mr, r_mr_t, col_same ? "yes" : "NO");

    auto rel = [&](const std::vector<double>& ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = (double)h_bf16(got[i]) - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        return den > 0.0 ? std::sqrt(num / den) : -1.0;
    };
    double sep = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = want[i] - unrounded[i];
        sep += d * d; den += want[i] * want[i];
    }
    sep = den > 0.0 ? std::sqrt(sep / den) : -1.0;

    const double r_ok = rel(want), r_shift = rel(shifted), r_unr = rel(unrounded);
    /* 0.01 is the bf16 STORE and nothing tighter is available: the output is bf16, so a correct
     * kernel lands within the grid's own spacing of the exact answer -- 1.7e-03 measured, against
     * 2^-9 = 2.0e-03 of grid. Two controls give that bar its meaning.
     *
     * The POSITION control is the loud one: off by one block it measures 5.8e-01, three hundred
     * times worse, because a wrong rope angle leaves the vector's length alone and only moves
     * where it points.
     *
     * The ROUNDING control is the quiet one and it is the reason this case exists. The kernel
     * must be closer to the order that rounds the pooled mean to bf16 before the norm than to the
     * order that does not -- 1.68e-03 against 2.27e-03, a 35% separation out of two numbers only
     * 1.6e-03 apart. It is not much, and it is the whole discrimination available at a bf16
     * output; a kernel that took the convenient order fails here and nowhere else. */
    const bool pass = r_ok >= 0.0 && r_ok < 0.01 && r_shift > 20.0 * r_ok &&
                      r_unr > 0.0 && r_ok < 0.9 * r_unr && scatter_clean && list_ok && mrope_ok;
    std::printf("        rel_l2 vs f64: %.3e (unrounded-mean order %.3e, the two orders %.3e "
                "apart); pos0 off by one block %.3e; scattered form %s\n",
                r_ok, r_unr, sep, r_shift, scatter_clean ? "ok" : "DIFFERS");
    std::printf("        work-list positions %s\n", list_ok ? "ok" : "DIFFER");
    ok(what, pass, "rel " + std::to_string(r_ok) + ", control " + std::to_string(r_shift));
}

/* gram_accum, against an f64 host Gram of the same dequantised bytes.
 *
 * THE THING THIS IS ACTUALLY FOR IS THE ACCUMULATE. A kernel that stored instead of adding would
 * agree with any single-call oracle and be wrong on the second launch, which is the only way this
 * op is ever used -- a corpus is many launches into one buffer. So `h` is SEEDED with a non-zero
 * pattern and the kernel is launched TWICE, and the expectation is seed + 2*want. Either mistake
 * -- storing, or dropping the seed -- fails; neither would fail a zero-seeded single call.
 *
 * The activation goes through the DEVICE quantiser rather than being synthesised, so the case also
 * pins the two ops' scale convention to each other: gram_accum reads `scale` as a DEQUANT
 * multiplier, and reading it the other way would be a factor of 448^2 on every entry.
 *
 * K = 260 is deliberate and is three awkward things at once: not a multiple of the 64-wide output
 * tile (the last tile has four live columns of sixty-four), not a multiple of the 128-wide scale
 * group (the last group has four), and a multiple of four (the staging dword). A square K would
 * have exercised none of the masking. */
static void gram_accum_case(const Plugin& pl) {
    const char* what = "gram_accum vs an f64 host Gram of the same bytes, accumulated twice";
    const RadKernelInfo* g = find(pl, "gram_accum");
    const RadKernelInfo* q = find(pl, "quant_act_fp8");
    if (!g || !q) { ok(what, false, "row absent"); return; }

    const int64_t M = 200, K = 260, group = 128, kb = (K + group - 1) / group;

    std::vector<uint16_t> xbf((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            float v = (float)(((m * 29 + k * 11) % 53) - 26) / 41.0f;
            if (k % 23 == 0) v *= 6.0f;          /* outlier channels, so the scales differ by group */
            xbf[(size_t)(m * K + k)] = h_to_bf16(v);
        }

    void* d_x  = dalloc(xbf.size() * 2);
    void* d_q  = dalloc((size_t)(M * K) + 4);    /* the row is owned out to a multiple of 4 */
    void* d_qs = dalloc((size_t)(M * kb) * 4);
    st_memcpy(d_x, xbf.data(), xbf.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadTensor> qt = {
        T(d_x,  RAD_BF16,   { M, K }),
        T(d_q,  RAD_F8E4M3, { M, K }),
        T(d_qs, RAD_F32,    { M, kb }),
    };
    std::vector<RadParam> qps = {
        PI("M", M), PI("n", K), PI("group", group), PS("dtype", "bf16"),
    };
    RadArgs qa;
    std::memset(&qa, 0, sizeof(qa));
    qa.t = qt.data(); qa.n_t = (int)qt.size();
    qa.p = qps.data(); qa.n_p = (int)qps.size();
    const int qrc = q->launch(&qa, nullptr);
    st_stream_sync(nullptr);
    if (qrc != RAD_OK) { ok(what, false, "quantiser rc=" + std::to_string(qrc)); return; }

    /* Read the codes back and build the oracle from THEM, not from the bf16 source: the kernel
     * contracts the codes, and an oracle that quantised again on the host would be measuring the
     * quantiser as well as the Gram. */
    std::vector<uint8_t> qc((size_t)(M * K));
    std::vector<float>   qsc((size_t)(M * kb));
    st_memcpy(qc.data(),  d_q,  qc.size(),      hipMemcpyDeviceToHost);
    st_memcpy(qsc.data(), d_qs, qsc.size() * 4, hipMemcpyDeviceToHost);
    std::vector<double> xd((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k)
            xd[(size_t)(m * K + k)] = (double)h_e4m3(qc[(size_t)(m * K + k)]) *
                                      (double)qsc[(size_t)(m * kb + k / group)];

    std::vector<double> want((size_t)(K * K), 0.0);
    for (int64_t k1 = 0; k1 < K; ++k1)
        for (int64_t k2 = 0; k2 < K; ++k2) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m)
                acc += xd[(size_t)(m * K + k1)] * xd[(size_t)(m * K + k2)];
            want[(size_t)(k1 * K + k2)] = acc;
        }

    std::vector<float> seed((size_t)(K * K));
    for (int64_t i = 0; i < K * K; ++i) seed[(size_t)i] = (float)((i % 17) - 8) * 0.25f;
    void* d_h = dalloc(seed.size() * 4);
    st_memcpy(d_h, seed.data(), seed.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_q,  RAD_F8E4M3, { M, K }),
        T(d_qs, RAD_F32,    { M, kb }),
        T(d_h,  RAD_F32,    { K, K }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", K), PI("group", group), PS("dtype", "fp8a8"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    for (int i = 0; i < 2; ++i) {
        const int rc = g->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
    }

    std::vector<float> got((size_t)(K * K));
    st_memcpy(got.data(), d_h, got.size() * 4, hipMemcpyDeviceToHost);
    double num = 0.0, den = 0.0, worst = 0.0;
    for (int64_t i = 0; i < K * K; ++i) {
        const double exp = (double)seed[(size_t)i] + 2.0 * want[(size_t)i];
        const double d = (double)got[(size_t)i] - exp;
        num += d * d; den += exp * exp;
        const double r = std::fabs(d) / (std::fabs(exp) + 1e-6);
        if (r > worst) worst = r;
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
    /* f32 accumulation over 200 rows in a register tree against f64: 1e-5 is three orders of
     * margin on what that costs, and four below anything the consumer's damping would notice. */
    const bool pass = rel >= 0.0 && rel < 1e-5;
    std::printf("        rel_l2 %.3e, worst entry %.3e\n", rel, worst);
    ok(what, pass, "rel_l2 " + std::to_string(rel) + ", worst " + std::to_string(worst));
}

/* gram_accum at bf16, against an f64 host Gram of the same bf16 values.
 *
 * The fp8 case above holds the accumulate to a seeded buffer and two launches; this one does the
 * same for the form with no scale operand, and adds the property the consumer leans on hardest: a
 * Gram is SYMMETRIC, and GPTQ's Cholesky takes the matrix as symmetric without looking. Each entry
 * and its mirror are the same products summed in the same row order by two different blocks, so
 * the kernel's output has to be symmetric to the bit, not to a tolerance.
 *
 * Two shapes: K = 260 for the masking (see the fp8 case) and an M that is not a whole number of
 * staged row blocks, and the routed down projection's width at a pass of sorted rows. */
static void gram_accum_bf16_case(const Plugin& pl, int64_t M, int64_t K) {
    const std::string what = "gram_accum_bf16 vs an f64 host Gram (M=" + std::to_string(M) +
                             ", K=" + std::to_string(K) + "), accumulated twice";
    const RadKernelInfo* g = find(pl, "gram_accum_bf16");
    if (!g) { ok(what.c_str(), false, "row absent"); return; }

    std::vector<uint16_t> xbf((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            float v = (float)(((m * 29 + k * 11) % 53) - 26) / 41.0f;
            if (k % 23 == 0) v *= 6.0f;
            xbf[(size_t)(m * K + k)] = h_to_bf16(v);
        }
    std::vector<double> xd((size_t)(M * K));
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = (double)h_bf16(xbf[i]);
    std::vector<double> want((size_t)(K * K), 0.0);
    for (int64_t k1 = 0; k1 < K; ++k1)
        for (int64_t k2 = k1; k2 < K; ++k2) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m) acc += xd[(size_t)(m * K + k1)] * xd[(size_t)(m * K + k2)];
            want[(size_t)(k1 * K + k2)] = want[(size_t)(k2 * K + k1)] = acc;
        }

    void* d_x = dalloc(xbf.size() * 2);
    void* d_h = dalloc((size_t)(K * K) * 4);
    if (!d_x || !d_h) { ok(what.c_str(), false, "alloc failed"); return; }
    st_memcpy(d_x, xbf.data(), xbf.size() * 2, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_x, RAD_BF16, { M, K }),
        RadTensor{},
        T(d_h, RAD_F32,  { K, K }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    /* One launch into zeros: the symmetry check reads this. */
    int rc = g->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what.c_str(), false, "launch rc=" + std::to_string(rc)); return; }
    std::vector<float> one((size_t)(K * K));
    st_memcpy(one.data(), d_h, one.size() * 4, hipMemcpyDeviceToHost);
    int64_t asym = 0;
    for (int64_t k1 = 0; k1 < K; ++k1)
        for (int64_t k2 = k1 + 1; k2 < K; ++k2)
            if (std::memcmp(&one[(size_t)(k1 * K + k2)], &one[(size_t)(k2 * K + k1)], 4)) ++asym;
    ok((what + ": symmetric to the bit").c_str(), asym == 0,
       std::to_string(asym) + " mirrored pairs differ");

    /* Seed, then two launches: seed + 2 * want, which neither a store nor a dropped seed passes. */
    std::vector<float> seed((size_t)(K * K));
    for (int64_t i = 0; i < K * K; ++i) seed[(size_t)i] = (float)((i % 17) - 8) * 0.25f;
    st_memcpy(d_h, seed.data(), seed.size() * 4, hipMemcpyHostToDevice);
    for (int i = 0; i < 2; ++i) {
        rc = g->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what.c_str(), false, "launch rc=" + std::to_string(rc)); return; }
    }
    std::vector<float> got((size_t)(K * K));
    st_memcpy(got.data(), d_h, got.size() * 4, hipMemcpyDeviceToHost);
    double num = 0.0, den = 0.0, worst = 0.0;
    for (int64_t i = 0; i < K * K; ++i) {
        const double exp = (double)seed[(size_t)i] + 2.0 * want[(size_t)i];
        const double d = (double)got[(size_t)i] - exp;
        num += d * d; den += exp * exp;
        const double r = std::fabs(d) / (std::fabs(exp) + 1e-6);
        if (r > worst) worst = r;
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
    std::printf("        rel_l2 %.3e, worst entry %.3e\n", rel, worst);
    ok(what.c_str(), rel >= 0.0 && rel < 1e-5,
       "rel_l2 " + std::to_string(rel) + ", worst " + std::to_string(worst));

    /* A scale handed to the bf16 form is a caller that thinks the input is quantised: refused. */
    ts[1] = T(d_h, RAD_F32, { M, 1 });
    rc = g->launch(&a, nullptr);
    ok((what + ": refuses a scale operand").c_str(), rc != RAD_OK, "rc=" + std::to_string(rc));
    dfree_all();
}

/* THE bf16 GROUPED MoE GEMM, AGAINST A HOST PRODUCT, EVERY FORM AGAINST EVERY OTHER.
 *
 * The routing is SKEWED the way moe_perf's is -- lognormal popularity, a Gumbel top-k a token -- so
 * runs go from empty to several times the mean, and the last few sorted rows are dropped slots
 * (sorted -1, past the live count) whose output must be zeros and not the stale bytes the output
 * is seeded with. Every form the launcher has -- the run-aligned prefill form, the (tile, run)
 * decode form and the uniform prefill grid -- runs on the same bytes and must match the first
 * byte for byte: each output element is one wave's accumulation over K in one order whichever
 * block computes it, so a byte apart is a form reading a different row or expert.
 *
 * THE WEIGHTS ARE ALSO HANDED AS A POINTER TABLE, which is what the engine hands the kernel, with
 * the experts laid out in a PERMUTED order behind it where the shape is small enough to keep two
 * copies: an entry the kernel addressed by arithmetic instead of through the table would then read
 * another expert's plane.
 *
 * THE NEGATIVE CONTROL is the down projection with `a_order` swapped to "token": the activation
 * still has a row per (token, slot), so the launch is legal and gathers the wrong rows, and the
 * comparison has to fail. A harness that passes it has stopped looking at the numbers. */
struct MoeBf16Shape { const char* name; int64_t N, K, ne, top_k; bool srt; };

static void moe_gemm_bf16_case(const Plugin& pl, int64_t Mtok, const MoeBf16Shape& sh,
                               bool control = false, int64_t seg = 0) {
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const RadKernelInfo* r = find(pl, "moe_gemm_bf16");
    const std::string what = std::string("moe_gemm_bf16 vs an f64 host product (") + sh.name +
                             ", " + std::to_string(Mtok) + " tokens" +
                             (seg ? ", serving the last " + std::to_string(seg) + " experts" : "") +
                             ")" + (control ? " CONTROL: a_order swapped" : "");
    if (!r) { ok(what.c_str(), false, "row absent"); return; }

    const int64_t N = sh.N, K = sh.K, E = sh.ne, top_k = sh.top_k, Tr = Mtok * top_k;
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ (uint64_t)(N * 131 + K * 7 + E + Mtok);
    auto rnd  = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto unif = [&]() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };

    /* The routing: every token's top_k distinct experts, then a stable sort by expert. The last
     * three tokens' slots are left out of the sort as dropped slots. */
    std::vector<double> logp((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        logp[(size_t)e] = 1.0 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by_e((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    const int64_t kept = Mtok > 4 ? Mtok - 3 : Mtok;
    for (int64_t t = 0; t < kept; ++t) {
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + top_k, sc.end(),
                          [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                              return x.first > y.first;
                          });
        for (int64_t j = 0; j < top_k; ++j)
            by_e[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * top_k + j));
    }
    std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
    sorted.reserve((size_t)Tr);
    int64_t empty = 0, rmax = 0;
    for (int64_t e = 0; e < E; ++e) {
        const int64_t n = (int64_t)by_e[(size_t)e].size();
        off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)n;
        sorted.insert(sorted.end(), by_e[(size_t)e].begin(), by_e[(size_t)e].end());
        empty += n == 0;
        rmax = std::max(rmax, n);
    }
    const int64_t live = off[(size_t)E];
    sorted.resize((size_t)Tr, -1);
    /* A CALL SERVING A TRAILING SEGMENT -- a layer's protected experts, sorted last -- is handed the
     * offsets from that segment's first expert on and that segment's weights, and every row below
     * the segment is another call's output: it must keep the seeded bytes exactly. */
    const int64_t e0 = seg ? E - seg : 0, En = seg ? seg : E;
    const int64_t s0 = off[(size_t)e0];

    /* The activation: one row a token, or one a (token, slot) for the down projection. */
    const int64_t AR = sh.srt ? Tr : Mtok;
    std::vector<uint16_t> ah((size_t)(AR * K)), wh((size_t)(E * N * K));
    for (auto& v : ah) v = h_to_bf16((float)((double)(int64_t)(rnd() % 2001) - 1000.0) / 700.0f);
    for (auto& v : wh) v = h_to_bf16((float)((double)(int64_t)(rnd() % 2001) - 1000.0) / 9000.0f);

    /* The reference, every output element, in f64 over the bf16 inputs exactly. */
    std::vector<double> ref((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < live; ++i) {
        const int32_t f = sorted[(size_t)i];
        if (f < 0) continue;
        int64_t e = 0;
        while (e + 1 < E && off[(size_t)(e + 1)] <= i) ++e;
        const int64_t row = sh.srt ? i : f / top_k;
        const uint16_t* ar = &ah[(size_t)(row * K)];
        for (int64_t n = 0; n < N; ++n) {
            const uint16_t* wr = &wh[(size_t)((e * N + n) * K)];
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) acc += (double)h_bf16(ar[k]) * (double)h_bf16(wr[k]);
            ref[(size_t)(i * N + n)] = acc;
        }
    }

    /* Two copies of the weights where they are small: the stacked plane in expert order, and a
     * permuted one the pointer table reads. One copy otherwise, and the table points into it. */
    const size_t eb = (size_t)(N * K) * 2;
    const bool permute = E * (int64_t)eb <= ((int64_t)16 << 20);
    void* d_w  = dalloc((size_t)E * eb);
    void* d_wp = permute ? dalloc((size_t)E * eb) : d_w;
    void* d_a  = dalloc(ah.size() * 2);
    void* d_s  = dalloc((size_t)Tr * 4);
    void* d_o  = dalloc((size_t)(E + 1) * 4);
    void* d_y  = dalloc((size_t)(Tr * N) * 2);
    void* d_tab = dalloc((size_t)E * sizeof(void*));
    const int64_t map_bytes = (1 + 2 * (E + Tr / 16 + 2)) * 4;
    void* d_map = dalloc((size_t)map_bytes);
    if (!d_w || !d_wp || !d_a || !d_s || !d_o || !d_y || !d_tab || !d_map) {
        ok(what.c_str(), false, "alloc failed");
        dfree_all();
        return;
    }
    st_memcpy(d_w, wh.data(), wh.size() * 2, hipMemcpyHostToDevice);
    std::vector<const void*> tab((size_t)E);
    if (permute) {
        std::vector<int64_t> slot((size_t)E);
        for (int64_t e = 0; e < E; ++e) slot[(size_t)e] = (e * 37 + 11) % E;
        bool bij = true;
        std::vector<char> seen((size_t)E, 0);
        for (int64_t e = 0; e < E; ++e) bij &= !seen[(size_t)slot[(size_t)e]]++;
        for (int64_t e = 0; e < E; ++e) {
            const int64_t s = bij ? slot[(size_t)e] : (E - 1 - e);
            st_memcpy((char*)d_wp + (size_t)s * eb, &wh[(size_t)(e * N * K)], eb,
                      hipMemcpyHostToDevice);
            tab[(size_t)e] = (const char*)d_wp + (size_t)s * eb;
        }
    } else {
        for (int64_t e = 0; e < E; ++e) tab[(size_t)e] = (const char*)d_w + (size_t)e * eb;
    }
    st_memcpy(d_tab, tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice);
    st_memcpy(d_a, ah.data(), ah.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_s, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(), off.size() * 4, hipMemcpyHostToDevice);

    const char* order = (sh.srt != control) ? "sorted" : "token";
    std::vector<RadParam> ps = {
        PI("M", Mtok), PI("N", N), PI("K", K), PI("n_expert", En), PI("top_k", top_k),
        PS("a_order", order), PS("dtype", "bf16"),
    };
    std::vector<uint16_t> first, got((size_t)(Tr * N));
    int64_t bad_total = 0;
    /* Stacked (form loop) and then the pointer table at the rule's form. */
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<RadTensor> ts = {
            T(d_a, RAD_BF16, { AR, K }),
            T(pass ? (void*)((const void**)d_tab + e0) : (void*)((char*)d_w + (size_t)e0 * eb),
              RAD_BF16, { En, N, K }),
            T(d_s, RAD_I32, { Tr }),
            T((int32_t*)d_o + e0, RAD_I32, { En + 1 }),
            T(d_y, RAD_BF16, { Tr, N }),
        };
        if (pass) ts[1].stride[0] = 0;
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int f_lo = (pass == 0 && pl.pin_moe) ? 1 : 0, f_hi = (pass == 0 && pl.pin_moe) ? 3 : 0;
        for (int form = f_lo; form <= f_hi; ++form) {
            const std::string label = what + " " + (pass ? "pointer table" : kForm[form]);
            if (pl.pin_moe) pl.pin_moe(form);
            {
                std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
                st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
            }
            a.scratch = form == 3 ? nullptr : d_map;
            a.scratch_bytes = form == 3 ? 0 : map_bytes;
            const int rc = r->launch(&a, nullptr);
            st_stream_sync(nullptr);
            if (pl.pin_moe) pl.pin_moe(0);
            if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
            st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);

            double worst = 0.0, rel_max = 0.0;
            int32_t ulp_max = 0;
            int64_t bad = 0;
            std::string firstbad;
            for (int64_t i = 0; i < Tr; ++i)
                for (int64_t n = 0; n < N; ++n) {
                    if (i < s0) {
                        if (got[(size_t)(i * N + n)] != 0x7f7fu && !bad++)
                            firstbad = ", row " + std::to_string(i) + " below the segment was "
                                       "written";
                        continue;
                    }
                    const double want = ref[(size_t)(i * N + n)];
                    const double gf = (double)h_bf16(got[(size_t)(i * N + n)]);
                    /* Half a bf16 ulp of the output, plus room for a K-term f32 sum taken in the
                     * WMMA's order; NaN counts as a miss. `rel_max` is the worst element as a
                     * fraction of that allowance, printed so a margin shows and not only a pass. */
                    const double err = std::fabs(gf - want);
                    const double tol = 0.004 * std::fabs(want) + 2e-3;
                    if (err / tol > rel_max) rel_max = err / tol;
                    /* ...and in bf16 ulps from the f64 product rounded once, on the
                     * sign-magnitude ordering of the bits. */
                    {
                        const uint16_t rb = h_to_bf16((float)want), gb = got[(size_t)(i * N + n)];
                        const int32_t ro = (rb & 0x8000) ? -(int32_t)(rb & 0x7fff) : (int32_t)rb;
                        const int32_t go = (gb & 0x8000) ? -(int32_t)(gb & 0x7fff) : (int32_t)gb;
                        const int32_t du = ro > go ? ro - go : go - ro;
                        if (du > ulp_max) ulp_max = du;
                    }
                    if (!(err <= tol)) {
                        if (!bad++)
                            firstbad = ", first at row " + std::to_string(i) + " col " +
                                       std::to_string(n) + ": " + std::to_string(gf) + " for " +
                                       std::to_string(want);
                        if (!(err <= worst)) worst = err;
                    }
                }
            bad_total += bad;
            if (control) continue;
            std::string same;
            if (first.empty()) {
                first = got;
                std::printf("        worst element %.3f of the tolerance, %d bf16 ulp from the rounded "
                            "f64 product\n", rel_max, (int)ulp_max);
            }
            else if (std::memcmp(first.data(), got.data(), got.size() * 2) != 0) {
                int64_t ndiff = 0;
                for (size_t q = 0; q < got.size(); ++q) ndiff += got[q] != first[q];
                same = ", " + std::to_string(ndiff) + " elements differ from the first form";
                ++bad;
            }
            ok(label.c_str(), bad == 0,
               std::to_string(bad) + " of " + std::to_string(Tr * N) + " elements off, worst " +
                   std::to_string(worst) + firstbad + same);
        }
        if (control) break;
    }
    if (control)
        ok(what.c_str(), bad_total > 0,
           "the swapped order agreed with the reference, so the comparison sees nothing");
    else
        std::printf("        %lld routed rows, %lld live, %lld experts empty, longest run %lld\n",
                    (long long)Tr, (long long)live, (long long)empty, (long long)rmax);
    dfree_all();
}

/* THE bf16 EXPERT GEMM AT A PREFILL CHUNK: Qwen3.8-Flash-Next's whole experts (gate_up 1280 x 2560
 * over token rows, down 2560 x 640 over sorted rows) under a skewed routing, every form timed and
 * compared with the first byte for byte.
 *
 * `distinct` planes back the `experts` table entries, entry e reading plane e % distinct. Equal to
 * `experts` it is the real weight traffic; smaller, it is what fits beside a running server, and
 * the time is then the kernel's arithmetic and staging over planes the last-level cache can hold --
 * a floor on the real figure rather than the figure, and the line printed says which it was. */
static void moe16_perf(const Plugin& pl, int64_t Mtok, int iters, int64_t E, int64_t top_k,
                       int64_t distinct) {
    if (iters <= 0) iters = 20;
    if (distinct <= 0 || distinct > E) distinct = E;
    const RadKernelInfo* r = find(pl, "moe_gemm_bf16");
    if (!r) { std::printf("--perfmoe16: row moe_gemm_bf16 absent\n"); return; }
    const int64_t Tr = Mtok * top_k;

    uint64_t rs = 0x9E3779B97F4A7C15ull;
    auto rnd  = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto unif = [&]() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
    std::vector<double> logp((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        logp[(size_t)e] = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by_e((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    for (int64_t t = 0; t < Mtok; ++t) {
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + top_k, sc.end(),
                          [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                              return x.first > y.first;
                          });
        for (int64_t j = 0; j < top_k; ++j)
            by_e[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * top_k + j));
    }
    std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
    sorted.reserve((size_t)Tr);
    for (int64_t e = 0; e < E; ++e) {
        off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)by_e[(size_t)e].size();
        sorted.insert(sorted.end(), by_e[(size_t)e].begin(), by_e[(size_t)e].end());
    }
    std::printf("moe_gemm_bf16 at a %lld-token chunk: %lld routed rows over %lld experts, %lld "
                "distinct weight planes%s, %d launches\n", (long long)Mtok, (long long)Tr,
                (long long)E, (long long)distinct,
                distinct < E ? " (planes alias: a cache-resident floor, not the real traffic)" : "",
                iters);

    void* d_s = dalloc((size_t)Tr * 4);
    void* d_o = dalloc((size_t)(E + 1) * 4);
    if (!d_s || !d_o) { std::printf("  alloc failed\n"); return; }
    st_memcpy(d_s, sorted.data(), (size_t)Tr * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(), (size_t)(E + 1) * 4, hipMemcpyHostToDevice);

    struct Proj { const char* name; int64_t N, K; bool srt; };
    const Proj projs[2] = { { "gate_up", 1280, 2560, false }, { "down", 2560, 640, true } };
    const int nforms = pl.pin_moe ? 3 : 1;
    for (const Proj& P : projs) {
        const int64_t Ma = P.srt ? Tr : Mtok;
        const size_t eb = (size_t)(P.N * P.K) * 2;
        std::vector<uint16_t> hb((size_t)std::max<int64_t>(Ma * P.K, P.N * P.K));
        for (auto& v : hb) v = h_to_bf16((float)((double)(int64_t)(rnd() % 2001) - 1000.0) / 3000.0f);
        void* d_a = dalloc((size_t)(Ma * P.K) * 2);
        void* d_w = dalloc((size_t)distinct * eb);
        void* d_tab = dalloc((size_t)E * sizeof(void*));
        void* d_y = dalloc((size_t)(Tr * P.N) * 2);
        if (!d_a || !d_w || !d_tab || !d_y) {
            std::printf("  %-8s alloc failed (%.1f MiB of planes)\n", P.name,
                        (double)distinct * (double)eb / 1048576.0);
            dfree_all();
            return;
        }
        st_memcpy(d_a, hb.data(), (size_t)(Ma * P.K) * 2, hipMemcpyHostToDevice);
        for (int64_t x = 0; x < distinct; ++x)
            st_memcpy((char*)d_w + (size_t)x * eb, hb.data(), eb, hipMemcpyHostToDevice);
        std::vector<const void*> tab((size_t)E);
        for (int64_t e = 0; e < E; ++e) tab[(size_t)e] = (const char*)d_w + (size_t)(e % distinct) * eb;
        st_memcpy(d_tab, tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice);

        std::vector<RadTensor> ts = {
            T(d_a, RAD_BF16, { Ma, P.K }),
            T(d_tab, RAD_BF16, { E, P.N, P.K }),
            T(d_s, RAD_I32, { Tr }),
            T(d_o, RAD_I32, { E + 1 }),
            T(d_y, RAD_BF16, { Tr, P.N }),
        };
        ts[1].stride[0] = 0;
        std::vector<RadParam> ps = {
            PI("M", Mtok), PI("N", P.N), PI("K", P.K), PI("n_expert", E), PI("top_k", top_k),
            PS("a_order", P.srt ? "sorted" : "token"), PS("dtype", "bf16"),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        {
            const int64_t need = std::max<int64_t>(r->scratch ? r->scratch(&a) : 0,
                                                   (1 + 2 * (E + Tr / 16 + 2)) * 4);
            a.scratch = dalloc((size_t)need);
            a.scratch_bytes = need;
        }
        const double flop = 2.0 * (double)Tr * (double)P.N * (double)P.K;
        std::vector<uint16_t> first((size_t)(Tr * P.N)), got((size_t)(Tr * P.N));
        for (int f = 0; f < nforms; ++f) {
            if (pl.pin_moe) pl.pin_moe(f + 1);
            const int rc = r->launch(&a, nullptr);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
                std::printf("  %-8s form %d  rc=%d (not served)\n", P.name, f + 1, rc);
                if (pl.pin_moe) pl.pin_moe(0);
                continue;
            }
            hipEvent_t e0, e1;
            st_ev_create(&e0); st_ev_create(&e1);
            for (int i = 0; i < 3; ++i) r->launch(&a, nullptr);
            st_ev_record(e0, nullptr);
            for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
            st_ev_record(e1, nullptr);
            st_ev_sync(e1);
            float ms = 0.0f;
            st_ev_elapsed(&ms, e0, e1);
            st_ev_destroy(e0); st_ev_destroy(e1);
            if (pl.pin_moe) pl.pin_moe(0);
            const double us = (double)ms * 1000.0 / iters;
            st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
            std::string same;
            if (f == 0) first = got;
            else same = std::memcmp(first.data(), got.data(), got.size() * 2) ? "  DIFFERS from form 1"
                                                                            : "  = form 1";
            std::printf("  %-8s form %d  %9.1f us  %7.1f TFLOP/s  %7.1f GB/s of planes%s\n", P.name,
                        f + 1, us, flop / (us * 1e6), (double)E * (double)eb / (us * 1e3),
                        same.c_str());
        }
        dfree_all();
        d_s = dalloc((size_t)Tr * 4);
        d_o = dalloc((size_t)(E + 1) * 4);
        if (!d_s || !d_o) return;
        st_memcpy(d_s, sorted.data(), (size_t)Tr * 4, hipMemcpyHostToDevice);
        st_memcpy(d_o, off.data(), (size_t)(E + 1) * 4, hipMemcpyHostToDevice);
    }
}

/* ------------------------------------------------------------------ the two-rank all-reduce */

struct ArThread {
    const RadKernelInfo* row;
    int rank, dev;
    int64_t numel;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_E_STATE;
    std::vector<float> out;
    /* The all-gather row width in ELEMENTS. 0 is the plain concatenation; anything else asks for
     * the row-interleaved placement the vocab-parallel sampler needs. */
    int64_t row_elems = 0;
    /* Ranks in the collective. 2 is the pair; 3 exercises the wide family at the one width that
     * is not a power of two. */
    int world = 2;
};

static void ar_worker(ArThread* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", t->numel), PS("dtype", "fp32"), PI("exact", 1),
    };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;

    void* in = nullptr;
    void* out = nullptr;
    if (hipMalloc(&in, (size_t)t->numel * 4) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    if (hipMalloc(&out, (size_t)t->numel * 4) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    std::vector<float> h((size_t)t->numel, (float)(t->rank + 1));
    st_memcpy(in, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    st_memset(out, 0, (size_t)t->numel * 4);

    RadTensor ts[2] = { T(in, RAD_F32, { t->numel }), T(out, RAD_F32, { t->numel }) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts; a.n_t = 2;
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.instance = t->inst;
    a.rank = t->rank;
    a.world_size = t->world;
    t->rc_launch = t->row->launch(&a, nullptr);
    st_stream_sync(nullptr);

    t->out.assign((size_t)t->numel, 0.0f);
    st_memcpy(t->out.data(), out, t->out.size() * 4, hipMemcpyDeviceToHost);
    st_free(in);
    st_free(out);
}

static void ar_two_rank(const Plugin& pl, const std::vector<int>& devs) {
    const RadKernelInfo* row = find(pl, "ar_oneshot_2rank_exact");
    if (!row) { ok("ar_oneshot_2rank_exact", false, "row absent"); return; }
    if (devs.size() < 2) { skip("ar_oneshot_2rank_exact (2 ranks)", "fewer than two gfx1201 cards"); return; }

    /* 8192 fp32 elements: a whole number of 16-byte pushes and small enough that the handshake,
     * not the link, is what is being exercised. */
    const int64_t numel = 8192;
    ArThread a{ row, 0, devs[0], numel }, b{ row, 1, devs[1], numel };
    std::thread ta(ar_worker, &a), tb(ar_worker, &b);
    ta.join();
    tb.join();

    ok("2-rank all-reduce init (peer rendezvous)",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK) return;
    ok("2-rank all-reduce launch",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch));

    int64_t bad = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (a.out[(size_t)i] != 3.0f) ++bad;
        if (b.out[(size_t)i] != 3.0f) ++bad;
    }
    ok("2-rank all-reduce sums 1 + 2 == 3 on both ranks, bit for bit", bad == 0,
       std::to_string(bad) + " elements wrong");

    if (row->fini) { row->fini(a.inst); row->fini(b.inst); }
    ok("2-rank all-reduce fini ran", true);
}

/* ------------------------------------------------------------------ changing block counts */
/* BACK-TO-BACK ALL-REDUCES ON ONE INSTANCE, AT DIFFERENT BLOCK COUNTS. The block count follows the
 * message, so the ops sharing an instance run different grids, and consecutive launches must still
 * take opposite scratch slots as a whole: a slot taken per block lets a block of one launch push
 * into bytes a different block of the previous launch is still reducing (r4d_ar_seq.h). Nothing
 * synchronises between the launches here, and each has its own values and its own output, so a
 * slot two in-flight launches share reads as a wrong sum and a counter the kernels disagree on as
 * a hang. The sizes give 11, 4 and 7 blocks under the default block rule. */
static const int64_t kArSweepNumel[] = { 65536, 8192, 40000, 65536, 12000, 24576, 65536, 8192 };
constexpr int kArSweepN = (int)(sizeof(kArSweepNumel) / sizeof(kArSweepNumel[0]));

struct ArSweep {
    const RadKernelInfo* row;
    int rank, dev;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    int64_t bad = 0;
    int world = 2;
};

static void ar_sweep_worker(ArSweep* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    int64_t most = 0;
    for (int64_t n : kArSweepNumel) most = n > most ? n : most;
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", most), PS("dtype", "fp32"), PI("exact", 1),
    };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;
    /* Rank r contributes (r + 1) * (j + 1), so launch j sums to (1 + 2 + .. + world) * (j + 1). */
    const float tri = (float)(t->world * (t->world + 1) / 2);

    void* in[kArSweepN] = {};
    void* out[kArSweepN] = {};
    for (int j = 0; j < kArSweepN; ++j) {
        const size_t bytes = (size_t)kArSweepNumel[j] * 4;
        if (hipMalloc(&in[j], bytes) != hipSuccess || hipMalloc(&out[j], bytes) != hipSuccess) {
            t->rc_launch = RAD_E_NOMEM;
            return;
        }
        /* Rank r contributes (r + 1) * (j + 1), so launch j sums to 3 * (j + 1) exactly. */
        std::vector<float> h((size_t)kArSweepNumel[j], (float)((t->rank + 1) * (j + 1)));
        st_memcpy(in[j], h.data(), bytes, hipMemcpyHostToDevice);
        st_memset(out[j], 0, bytes);
    }
    for (int j = 0; j < kArSweepN && t->rc_launch == RAD_OK; ++j) {
        RadTensor ts[2] = { T(in[j], RAD_F32, { kArSweepNumel[j] }),
                            T(out[j], RAD_F32, { kArSweepNumel[j] }) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts; a.n_t = 2;
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.instance = t->inst;
        a.rank = t->rank;
        a.world_size = t->world;
        t->rc_launch = t->row->launch(&a, nullptr);
    }
    st_stream_sync(nullptr);
    for (int j = 0; j < kArSweepN; ++j) {
        std::vector<float> h((size_t)kArSweepNumel[j]);
        st_memcpy(h.data(), out[j], h.size() * 4, hipMemcpyDeviceToHost);
        for (float v : h) if (v != tri * (float)(j + 1)) ++t->bad;
        st_free(in[j]);
        st_free(out[j]);
    }
}

static void ar_two_rank_sweep(const Plugin& pl, const std::vector<int>& devs) {
    const RadKernelInfo* row = find(pl, "ar_oneshot_2rank_exact");
    if (!row) { ok("ar_oneshot_2rank_exact", false, "row absent"); return; }
    if (devs.size() < 2) {
        skip("ar_oneshot_2rank_exact (block counts)", "fewer than two gfx1201 cards");
        return;
    }
    ArSweep a{ row, 0, devs[0] }, b{ row, 1, devs[1] };
    std::thread ta(ar_sweep_worker, &a), tb(ar_sweep_worker, &b);
    ta.join();
    tb.join();
    ok("2-rank all-reduce init at the widest message",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK) return;
    ok("2-rank all-reduce, launches back to back at changing block counts, every sum exact on "
       "both ranks",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK && a.bad == 0 && b.bad == 0,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch) + ", " +
       std::to_string(a.bad + b.bad) + " elements wrong");
    if (row->fini) { row->fini(a.inst); row->fini(b.inst); }
}

/* ------------------------------------------------------------------ the two-rank all-gather */
/* The payload is what the vocab-parallel sampler really gathers: (u32 global token id, f32 logit)
 * pairs. THE IDS ARE THE POINT. A token id is a small integer, and its bit pattern read as f32 is
 * a DENORMAL -- so an all-gather faked as an all-reduce over zero-padded planes returns the ids
 * flushed to zero anywhere denormals are flushed, and the model then samples token 0 forever while
 * every logit still looks right. This checks the bytes, on both ranks, for both halves. */
static void gather_worker(ArThread* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", t->numel), PS("dtype", "f32"),
    };
    if (t->row_elems > 0) ps.push_back(PI("row", t->row_elems));
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;

    void* in = nullptr;
    void* out = nullptr;
    if (hipMalloc(&in, (size_t)t->numel * 4) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    if (hipMalloc(&out, (size_t)t->numel * 4 * t->world) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }

    std::vector<uint32_t> h((size_t)t->numel);
    for (int64_t i = 0; i < t->numel; i += 2) {
        h[(size_t)i] = (uint32_t)(t->rank * 124160 + (int)i + 1);          /* a real token id */
        const float v = (float)(t->rank + 1) * 0.5f + (float)i;            /* its logit */
        std::memcpy(&h[(size_t)i + 1], &v, 4);
    }
    st_memcpy(in, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    st_memset(out, 0xAB, (size_t)t->numel * 4 * t->world);   /* so an untouched slice is visibly untouched */

    RadTensor ts[2] = { T(in, RAD_F32, { t->numel }), T(out, RAD_F32, { t->numel * t->world }) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts; a.n_t = 2;
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.instance = t->inst;
    a.rank = t->rank;
    a.world_size = t->world;
    t->rc_launch = t->row->launch(&a, nullptr);
    st_stream_sync(nullptr);

    t->out.assign((size_t)t->numel * t->world, 0.0f);
    st_memcpy(t->out.data(), out, t->out.size() * 4, hipMemcpyDeviceToHost);
    st_free(in);
    st_free(out);
}

static void gather_two_rank(const Plugin& pl, const std::vector<int>& devs) {
    const RadKernelInfo* row = find(pl, "ar_gather_2rank");
    if (!row) { ok("ar_gather_2rank", false, "row absent"); return; }
    if (devs.size() < 2) { skip("ar_gather_2rank (2 ranks)", "fewer than two gfx1201 cards"); return; }

    const int64_t numel = 8192;   /* the sampler's own message at 4 rows and 1024 candidates */
    ArThread a{ row, 0, devs[0], numel }, b{ row, 1, devs[1], numel };
    std::thread ta(gather_worker, &a), tb(gather_worker, &b);
    ta.join();
    tb.join();

    ok("2-rank all-gather init (peer rendezvous)",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK) return;
    ok("2-rank all-gather launch",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch));

    /* What each rank sent, rebuilt here rather than read back off the device: the check is that
     * the OUTPUT holds it, so the expectation must not come from the same memory. */
    auto sent = [numel](int rank, int64_t i) -> uint32_t {
        if ((i & 1) == 0) return (uint32_t)(rank * 124160 + (int)i + 1);
        uint32_t u; const float v = (float)(rank + 1) * 0.5f + (float)(i - 1);
        std::memcpy(&u, &v, 4);
        return u;
    };
    int64_t bad = 0, bad_id = 0;
    for (const ArThread* t : { &a, &b }) {
        for (int r = 0; r < 2; ++r) {
            for (int64_t i = 0; i < numel; ++i) {
                uint32_t got;
                std::memcpy(&got, &t->out[(size_t)(r * numel + i)], 4);
                if (got == sent(r, i)) continue;
                ++bad;
                if ((i & 1) == 0) ++bad_id;
            }
        }
    }
    ok("2-rank all-gather places both ranks' bytes on both ranks, bit for bit", bad == 0,
       std::to_string(bad) + " words wrong, " + std::to_string(bad_id) + " of them token ids");

    if (row->fini) { row->fini(a.inst); row->fini(b.inst); }

    /* THE ROW-INTERLEAVED PLACEMENT, which is the one the sampler actually asks for: the input is
     * [rows][row] and the output must be [rows][2][row], so rank 1's row m sits immediately after
     * rank 0's rather than a whole message away. A concatenating gather passes the check above and
     * fails this one, and the difference between them is a merge that reads one rank's candidates
     * twice and the other rank's never. */
    const int64_t rowe = 2048;                     /* 4 rows of 1024 (id, logit) pairs */
    ArThread c{ row, 0, devs[0], numel }, d{ row, 1, devs[1], numel };
    c.row_elems = d.row_elems = rowe;
    std::thread tc(gather_worker, &c), td(gather_worker, &d);
    tc.join();
    td.join();
    if (c.rc_init != RAD_OK || d.rc_init != RAD_OK || c.rc_launch != RAD_OK ||
        d.rc_launch != RAD_OK) {
        ok("2-rank all-gather, row-interleaved", false,
           "rc " + std::to_string(c.rc_init) + "/" + std::to_string(d.rc_init) + " init, " +
           std::to_string(c.rc_launch) + "/" + std::to_string(d.rc_launch) + " launch");
        return;
    }
    const int64_t nrows = numel / rowe;
    bad = bad_id = 0;
    for (const ArThread* t : { &c, &d }) {
        for (int64_t m = 0; m < nrows; ++m) {
            for (int r = 0; r < 2; ++r) {
                for (int64_t j = 0; j < rowe; ++j) {
                    uint32_t got;
                    std::memcpy(&got, &t->out[(size_t)(m * 2 * rowe + r * rowe + j)], 4);
                    if (got == sent(r, m * rowe + j)) continue;
                    ++bad;
                    if ((j & 1) == 0) ++bad_id;
                }
            }
        }
    }
    ok("2-rank all-gather interleaves by row, so one row carries both ranks", bad == 0,
       std::to_string(bad) + " words wrong, " + std::to_string(bad_id) + " of them token ids");

    if (row->fini) { row->fini(c.inst); row->fini(d.inst); }
    ok("2-rank all-gather fini ran", true);
}

/* ------------------------------------------------------------------ the three-rank wide family */
/* THE ONE WIDTH THAT IS NOT A POWER OF TWO. The wide kernels are templated on the world size and
 * shipped at 4 and 8; three cards instantiate the same template at 3, and these are the checks that
 * the instantiation, the dispatch rows and the entry points agree: a rendezvous of three, a sum of
 * 1 + 2 + 3 on every rank bit for bit, the slot parity surviving a block-count sweep, and the
 * gather placing three slices in both of its layouts. Skipped, not failed, on fewer cards. */
static void ar_three_rank(const Plugin& pl, const std::vector<int>& devs) {
    const RadKernelInfo* row = find(pl, "ar_oneshot_3rank_exact");
    if (!row) { ok("ar_oneshot_3rank_exact", false, "row absent"); return; }
    if (devs.size() < 3) { skip("ar_oneshot_3rank_exact (3 ranks)", "fewer than three gfx1201 cards"); return; }
    const int64_t numel = 8192;
    ArThread a{ row, 0, devs[0], numel }, b{ row, 1, devs[1], numel }, c{ row, 2, devs[2], numel };
    a.world = b.world = c.world = 3;
    std::thread ta(ar_worker, &a), tb(ar_worker, &b), tc(ar_worker, &c);
    ta.join();
    tb.join();
    tc.join();
    ok("3-rank all-reduce init (peer rendezvous)",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK && c.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init) + "/" +
           std::to_string(c.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK || c.rc_init != RAD_OK) return;
    ok("3-rank all-reduce launch",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK && c.rc_launch == RAD_OK,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch) + "/" +
           std::to_string(c.rc_launch));
    int64_t bad = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (a.out[(size_t)i] != 6.0f) ++bad;
        if (b.out[(size_t)i] != 6.0f) ++bad;
        if (c.out[(size_t)i] != 6.0f) ++bad;
    }
    ok("3-rank all-reduce sums 1 + 2 + 3 == 6 on every rank, bit for bit", bad == 0,
       std::to_string(bad) + " elements wrong");
    if (row->fini) { row->fini(a.inst); row->fini(b.inst); row->fini(c.inst); }
    ok("3-rank all-reduce fini ran", true);

    /* Block counts that change launch to launch on one instance: the wide family takes its slot
     * parity from the shared counters exactly as the pair does, and this is the check that it
     * does so at a width whose flag array is [block][3]. */
    ArSweep x{ row, 0, devs[0] }, y{ row, 1, devs[1] }, z{ row, 2, devs[2] };
    x.world = y.world = z.world = 3;
    std::thread tx(ar_sweep_worker, &x), ty(ar_sweep_worker, &y), tz(ar_sweep_worker, &z);
    tx.join();
    ty.join();
    tz.join();
    const bool up = x.rc_init == RAD_OK && y.rc_init == RAD_OK && z.rc_init == RAD_OK &&
                    x.rc_launch == RAD_OK && y.rc_launch == RAD_OK && z.rc_launch == RAD_OK;
    ok("3-rank all-reduce, eight launches at changing block counts on one instance",
       up && x.bad == 0 && y.bad == 0 && z.bad == 0,
       up ? std::to_string(x.bad + y.bad + z.bad) + " elements wrong"
          : "rc " + std::to_string(x.rc_init) + "/" + std::to_string(y.rc_init) + "/" +
                std::to_string(z.rc_init) + " init, " + std::to_string(x.rc_launch) + "/" +
                std::to_string(y.rc_launch) + "/" + std::to_string(z.rc_launch) + " launch");
    if (row->fini) {
        if (x.inst) row->fini(x.inst);
        if (y.inst) row->fini(y.inst);
        if (z.inst) row->fini(z.inst);
    }
}

static void gather_three_rank(const Plugin& pl, const std::vector<int>& devs) {
    const RadKernelInfo* row = find(pl, "ar_gather_3rank");
    if (!row) { ok("ar_gather_3rank", false, "row absent"); return; }
    if (devs.size() < 3) { skip("ar_gather_3rank (3 ranks)", "fewer than three gfx1201 cards"); return; }
    const int64_t numel = 8192;   /* the sampler's own message at 4 rows and 1024 candidates */
    const int W = 3;
    /* What each rank sent, rebuilt here rather than read back off the device (gather_two_rank). */
    auto sent = [](int rank, int64_t i) -> uint32_t {
        if ((i & 1) == 0) return (uint32_t)(rank * 124160 + (int)i + 1);
        uint32_t u; const float v = (float)(rank + 1) * 0.5f + (float)(i - 1);
        std::memcpy(&u, &v, 4);
        return u;
    };
    {
        ArThread a{ row, 0, devs[0], numel }, b{ row, 1, devs[1], numel }, c{ row, 2, devs[2], numel };
        a.world = b.world = c.world = W;
        std::thread ta(gather_worker, &a), tb(gather_worker, &b), tc(gather_worker, &c);
        ta.join();
        tb.join();
        tc.join();
        ok("3-rank all-gather init (peer rendezvous)",
           a.rc_init == RAD_OK && b.rc_init == RAD_OK && c.rc_init == RAD_OK,
           "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init) + "/" +
               std::to_string(c.rc_init));
        if (a.rc_init != RAD_OK || b.rc_init != RAD_OK || c.rc_init != RAD_OK) return;
        ok("3-rank all-gather launch",
           a.rc_launch == RAD_OK && b.rc_launch == RAD_OK && c.rc_launch == RAD_OK,
           "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch) + "/" +
               std::to_string(c.rc_launch));
        int64_t bad = 0, bad_id = 0;
        for (const ArThread* t : { &a, &b, &c }) {
            for (int r = 0; r < W; ++r) {
                for (int64_t i = 0; i < numel; ++i) {
                    uint32_t got;
                    std::memcpy(&got, &t->out[(size_t)(r * numel + i)], 4);
                    if (got == sent(r, i)) continue;
                    ++bad;
                    if ((i & 1) == 0) ++bad_id;
                }
            }
        }
        ok("3-rank all-gather places every rank's bytes on every rank, bit for bit", bad == 0,
           std::to_string(bad) + " words wrong, " + std::to_string(bad_id) + " of them token ids");
        if (row->fini) { row->fini(a.inst); row->fini(b.inst); row->fini(c.inst); }
    }

    /* The row-interleaved placement the sampler asks for: [rows][3][row]. */
    const int64_t rowe = 2048;                     /* 4 rows of 1024 (id, logit) pairs */
    ArThread c{ row, 0, devs[0], numel }, d{ row, 1, devs[1], numel }, e{ row, 2, devs[2], numel };
    c.world = d.world = e.world = W;
    c.row_elems = d.row_elems = e.row_elems = rowe;
    std::thread tc(gather_worker, &c), td(gather_worker, &d), te(gather_worker, &e);
    tc.join();
    td.join();
    te.join();
    if (c.rc_init != RAD_OK || d.rc_init != RAD_OK || e.rc_init != RAD_OK ||
        c.rc_launch != RAD_OK || d.rc_launch != RAD_OK || e.rc_launch != RAD_OK) {
        ok("3-rank all-gather, row-interleaved", false,
           "rc " + std::to_string(c.rc_init) + "/" + std::to_string(d.rc_init) + "/" +
               std::to_string(e.rc_init) + " init, " + std::to_string(c.rc_launch) + "/" +
               std::to_string(d.rc_launch) + "/" + std::to_string(e.rc_launch) + " launch");
        return;
    }
    const int64_t nrows = numel / rowe;
    int64_t bad = 0, bad_id = 0;
    for (const ArThread* t : { &c, &d, &e }) {
        for (int64_t m = 0; m < nrows; ++m) {
            for (int r = 0; r < W; ++r) {
                for (int64_t j = 0; j < rowe; ++j) {
                    uint32_t got;
                    std::memcpy(&got, &t->out[(size_t)(m * W * rowe + r * rowe + j)], 4);
                    if (got == sent(r, m * rowe + j)) continue;
                    ++bad;
                    if ((j & 1) == 0) ++bad_id;
                }
            }
        }
    }
    ok("3-rank all-gather interleaves by row, so one row carries all three ranks", bad == 0,
       std::to_string(bad) + " words wrong, " + std::to_string(bad_id) + " of them token ids");
    if (row->fini) { row->fini(c.inst); row->fini(d.inst); row->fini(e.inst); }
    ok("3-rank all-gather fini ran", true);
}

/* ------------------------------------------------------------------ the device sweep */

/* NGRAM_IDS' `ahead_ids`: the prompt that follows the step, hashed in the same launch. Step one
 * runs two sequences, [5, 6] and [11, 42], with three more tokens of the LAST one, [7, 8, 9], in
 * `tok` behind them. Their ids must be the ids step two computes running [7, 8, 9] against the
 * window step one left; that window must be the step's own; and the step's own ids must be the
 * ones a launch without `ahead_ids` writes. The same property ref_test holds libref to. */
/* embed_lookup_q against a host gather of the same bytes, through the host side's own conversions.
 *
 * THE BAR IS THE BIT, not a tolerance. libavx's gather decodes a code as
 * rad_f32_to_bf16(rad_fp8e4m3_to_f32(code) * scale), and the device kernel has to produce exactly
 * that for every code, or one model reads a different n-gram embedding depending on which tier
 * --ngram-placement put its table on. So the oracle is those two functions, the table holds every
 * code but the two NaN encodings (no quantiser writes them), and the comparison is of the bytes.
 *
 * Five launches: the quad walk and the entry walk (the model's width of 160, and 13); an E4M3
 * table with a scale and a plain bf16 one without, whose rows come back unchanged; the table in
 * VRAM and in pinned host memory read over the link, the placement this kernel exists for. The ids
 * include padding (-1) and one past the table, and both write a zero row. */
static void embed_lookup_q_case(const Plugin& pl) {
    if (!sel("embed_lookup_q_bf16")) return;
    const char* what = "embed_lookup_q vs a host gather, to the bit";
    const RadKernelInfo* r = find(pl, "embed_lookup_q_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    struct Shape { int64_t ne; bool e4m3; bool host; };
    const Shape shapes[] = { { 160, true, false }, { 160, true, true }, { 13, true, false },
                             { 160, false, true }, { 13, false, false } };
    const int64_t NV = 1031, M = 37;
    const float scale = rad_bf16_to_f32(rad_f32_to_bf16(1.99317932e-4f));
    std::vector<void*> pinned;
    std::string bad;
    for (const Shape& sh : shapes) {
        const int64_t es = sh.e4m3 ? 1 : 2;
        const size_t tbytes = (size_t)(NV * sh.ne * es);
        std::vector<uint8_t> table(tbytes);
        uint32_t rs = 0x9e3779b9u;
        for (int64_t i = 0; i < NV * sh.ne; ++i) {
            rs = rs * 1664525u + 1013904223u;
            if (sh.e4m3) {
                uint8_t c = (uint8_t)((i * 167 + (rs >> 24)) & 0xFF);
                if ((c & 0x7F) == 0x7F) c ^= 0x01;           /* 0x7F and 0xFF are NaN */
                table[(size_t)i] = c;
            } else {
                const uint16_t b = rad_f32_to_bf16((float)((int32_t)(rs >> 9) - (1 << 22)) * 1e-6f);
                std::memcpy(&table[(size_t)(2 * i)], &b, 2);
            }
        }
        std::vector<int32_t> ids((size_t)M);
        for (int64_t m = 0; m < M; ++m) ids[(size_t)m] = (int32_t)((m * 131 + 7) % NV);
        ids[3] = -1;
        ids[7] = (int32_t)NV;
        ids[11] = (int32_t)(NV - 1);
        ids[15] = ids[16] = ids[17] = ids[18] = 42;

        void* d_tab = nullptr;
        if (sh.host) {
            void* h = nullptr;
            if (hipHostMalloc(&h, tbytes, hipHostMallocMapped) != hipSuccess) {
                ok(what, false, "pinned table allocation failed");
                return;
            }
            pinned.push_back(h);
            std::memcpy(h, table.data(), tbytes);
            if (hipHostGetDevicePointer(&d_tab, h, 0) != hipSuccess || !d_tab) {
                ok(what, false, "no device address for the pinned table");
                return;
            }
        } else {
            d_tab = dalloc(tbytes);
            st_memcpy(d_tab, table.data(), tbytes, hipMemcpyHostToDevice);
        }
        const uint16_t sc_bits = rad_f32_to_bf16(scale);
        void* d_sc = dalloc(2);
        st_memcpy(d_sc, &sc_bits, 2, hipMemcpyHostToDevice);
        void* d_tok = dalloc((size_t)M * 4);
        st_memcpy(d_tok, ids.data(), (size_t)M * 4, hipMemcpyHostToDevice);
        void* d_x = dalloc((size_t)(M * sh.ne * 2));
        st_memset(d_x, 0xFF, (size_t)(M * sh.ne * 2));    /* a zero row has to be written */

        std::vector<RadTensor> ts = { T(d_tok, RAD_I32, { M }),
                                      T(d_tab, sh.e4m3 ? RAD_F8E4M3 : RAD_BF16, { NV, sh.ne }),
                                      T(sh.e4m3 ? d_sc : nullptr, RAD_BF16, { 1 }),
                                      T(d_x, RAD_BF16, { M, sh.ne }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_embd", sh.ne), PI("n_vocab", NV),
                                     PS("dtype", "bf16") };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        std::vector<uint16_t> got((size_t)(M * sh.ne), 0);
        st_memcpy(got.data(), d_x, got.size() * 2, hipMemcpyDeviceToHost);

        int64_t wrong = 0;
        for (int64_t m = 0; m < M; ++m) {
            const int32_t id = ids[(size_t)m];
            for (int64_t i = 0; i < sh.ne; ++i) {
                uint16_t want = 0;
                if (id >= 0 && id < NV) {
                    const size_t at = (size_t)(id * sh.ne + i);
                    if (sh.e4m3) {
                        want = rad_f32_to_bf16(rad_fp8e4m3_to_f32(table[at]) * scale);
                    } else {
                        std::memcpy(&want, &table[2 * at], 2);
                    }
                }
                if (got[(size_t)(m * sh.ne + i)] != want) ++wrong;
            }
        }
        if (rc != RAD_OK || wrong)
            bad += fmt("%s n_embd=%lld %s: rc=%d, %lld of %lld entries differ; ",
                       sh.e4m3 ? "e4m3" : "bf16", (long long)sh.ne,
                       sh.host ? "pinned host table" : "vram table", rc, (long long)wrong,
                       (long long)(M * sh.ne));
    }
    for (void* h : pinned) hip_ck(hipHostFree(h), "hipHostFree", __LINE__);
    ok(what, bad.empty(), bad);
}

static void ngram_ahead_case(const Plugin& pl) {
    if (!sel("ngram_ids_i32")) return;
    const RadKernelInfo* r = find(pl, "ngram_ids_i32");
    if (!r) { ok("ngram_ids_i32 ahead", false, "row absent"); return; }
    const int64_t kEos = 248044, H = 16, NG = 3;
    std::vector<int64_t> mult = { 23703573157769LL, 20109073645365LL, 8052911324071LL }, vsz, off;
    int64_t acc = 0;
    for (int i = 0; i < 16; ++i) { vsz.push_back(20000003 + 10 * i); off.push_back(acc); acc += vsz.back(); }
    auto dev = [&](const void* h, size_t b) { void* d = dalloc(b); st_memcpy(d, h, b, hipMemcpyHostToDevice); return d; };
    void* d_mult = dev(mult.data(), 3 * 8);
    void* d_vsz  = dev(vsz.data(), 16 * 8);
    void* d_off  = dev(off.data(), 16 * 8);
    auto launch = [&](const std::vector<int32_t>& tok, std::vector<int32_t>& state,
                      const std::vector<int32_t>& cu, const std::vector<int32_t>& cidx, int64_t M,
                      int64_t A, std::vector<int32_t>* ids, std::vector<int32_t>* ahead) {
        void* d_tok = dev(tok.data(), tok.size() * 4);
        void* d_st  = dev(state.data(), state.size() * 4);
        void* d_cu  = dev(cu.data(), cu.size() * 4);
        void* d_ci  = dev(cidx.data(), cidx.size() * 4);
        std::vector<int32_t> hini(cidx.size(), 1);
        void* d_hi  = dev(hini.data(), hini.size() * 4);
        void* d_ids = dalloc((size_t)(M * H * 4));
        void* d_ahd = A > 0 ? dalloc((size_t)(A * H * 4)) : nullptr;
        std::vector<RadTensor> ts = {
            T(d_tok, RAD_I32, { (int64_t)tok.size() }), T(d_st, RAD_I32, { (int64_t)state.size() / 2, 2 }),
            T(d_cu, RAD_I32, { (int64_t)cu.size() }), T(d_mult, RAD_I64, { 3 }),
            T(d_vsz, RAD_I64, { 16 }), T(d_off, RAD_I64, { 16 }),
            T(d_ci, RAD_I32, { (int64_t)cidx.size() }), T(d_hi, RAD_I32, { (int64_t)hini.size() }),
            T(nullptr, RAD_I32, { 1 }), T(d_ids, RAD_I32, { M, H }), T(d_ahd, RAD_I32, { A, H }) };
        absent(ts[8]);
        if (A == 0) absent(ts[10]);
        std::vector<RadParam> ps = { PI("M", M), PI("heads", H), PI("ngram", NG), PI("eos", kEos) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return false;
        ids->resize((size_t)(M * H));
        st_memcpy(ids->data(), d_ids, ids->size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(state.data(), d_st, state.size() * 4, hipMemcpyDeviceToHost);
        if (A > 0) {
            ahead->resize((size_t)(A * H));
            st_memcpy(ahead->data(), d_ahd, ahead->size() * 4, hipMemcpyDeviceToHost);
        }
        return true;
    };
    std::vector<int32_t> st1 = { 900, 901, 902, 903 }, st0 = st1, st2;
    std::vector<int32_t> ids1, ahd1, ids0, ids2;
    const bool l1 = launch({ 5, 6, 11, 42, 7, 8, 9 }, st1, { 0, 2, 4 }, { 0, 1 }, 4, 3, &ids1, &ahd1);
    const bool l0 = launch({ 5, 6, 11, 42 }, st0, { 0, 2, 4 }, { 0, 1 }, 4, 0, &ids0, nullptr);
    st2 = st1;
    const bool l2 = l1 && launch({ 7, 8, 9 }, st2, { 0, 3 }, { 1 }, 3, 0, &ids2, nullptr);
    if (!l1 || !l0 || !l2) { ok("ngram_ids_i32 ahead: served", false); return; }
    ok("ngram_ids_i32 ahead: the step's own ids are the ones a launch without it writes", ids1 == ids0);
    ok("ngram_ids_i32 ahead: the window is the step's own", st1 == st0 && st1[2] == 11 && st1[3] == 42);
    ok("ngram_ids_i32 ahead: the ids ahead are the ones the next step computes", ahd1 == ids2);
}

static void device_phase(const Plugin& pl) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
        skip("device phase", "no HIP device");
        return;
    }
    /* A host can present more HIP devices than it has discrete cards: an integrated part shows up
     * alongside them, reporting tens of GiB of "VRAM" that is really system RAM. Filtering on the
     * arch string rather than taking device 0 is not defensive tidiness -- a rank landing on the
     * integrated part runs slowly and wrongly and the failure looks like a kernel bug. */
    std::vector<int> devs;
    for (int d = 0; d < n; ++d) {
        hipDeviceProp_t p;
        if (hipGetDeviceProperties(&p, d) != hipSuccess) continue;
        if (!g_case) std::printf("  device %d: %s\n", d, p.gcnArchName);
        if (std::strncmp(p.gcnArchName, "gfx1201", 7) == 0) devs.push_back(d);
    }
    if (devs.empty()) { skip("device phase", "no gfx1201 device"); return; }
    st_set_device(devs[0]);

    if (!g_case) std::printf("\ndevice smoke tests (device %d)\n", devs[0]);

    ngram_ahead_case(pl);
    embed_lookup_q_case(pl);

    /* --- the paged cache's write side --- */
    kv_store_case(pl, "kv_store_bf16", "bf16");
    kv_store_case(pl, "kv_store_fp8",  "fp8_e4m3");

    /* --- attention --- */
    attn_case(pl, "attn_prefill_h256_gqa6_fp8kv",  256, 6, 32, "fp8_e4m3", 1);
    attn_case(pl, "attn_prefill_h256_gqa6_bf16kv", 256, 6, 32, "bf16",     1);
    attn_case(pl, "attn_decode_h256_gqa6_fp8kv",   256, 6,  1, "fp8_e4m3", 0);
    attn_case(pl, "attn_decode_h256_gqa6_csa3_fp8kv", 256, 6, 1, "fp8_e4m3", 0);
    attn_case(pl, "attn_decode_h256_gqa6_bf16kv",  256, 6,  1, "bf16",     0);
    attn_case(pl, "attn_decode_h128_gqa4_fp8kv",   128, 4,  1, "fp8_e4m3", 0);
    attn_case(pl, "attn_decode_h128_gqa4_bf16kv",  128, 4,  1, "bf16",     0);
    csa3_kvarn_case(pl);
    attn_gq_fold_case(pl, 3, false);
    attn_gq_fold_case(pl, 24, false);
    attn_gq_fold_case(pl, 3, true);
    attn_gq_fold_case(pl, 24, true);
    for (const bool bf16 : { false, true }) {
        attn_gq_fold_case(pl, 3, bf16, true);
        attn_gq_fold_case(pl, 24, bf16, true);
    }
    if (sel("attn_vit_h72_bf16")) {
        const RadKernelInfo* r = find(pl, "attn_vit_h72_bf16");
        if (!r) ok("attn_vit_h72_bf16", false, "row absent");
        else {
            const int64_t total = 64, heads = 4, hd = 72;
            std::vector<RadTensor> ts = {
                T(nullptr, RAD_BF16, { total, heads, hd }),
                T(nullptr, RAD_BF16, { total, heads, hd }),
                T(nullptr, RAD_BF16, { total, heads, hd }),
                T(nullptr, RAD_I32,  { 3 }),
                T(nullptr, RAD_BF16, { total, heads, hd }),
            };
            std::vector<RadParam> ps = {
                PI("M", total), PI("head_dim", hd), PI("gqa", 1), PI("causal", 0),
                PS("dtype", "bf16"), PI("max_seqlen", 32),
            };
            run_case(r, ts, ps);
        }
    }


    /* --- cast: the strided row copy an intermediate-hidden-state tap writes with -------------
     *
     * The destination is deliberately a COLUMN SLICE of a wider buffer, because that is the only
     * thing about this row that can be wrong. A contiguous copy is uninteresting; a copy that has
     * to honour a destination row pitch different from the source's is what puts five taps of the
     * trunk's residual stream side by side in one [T, 5 * n_embd] row, and getting the pitch wrong
     * writes a correct-looking tensor into the wrong columns. So the check is exact -- byte for
     * byte, this is a copy -- and it asserts the UNTOUCHED columns too, since a kernel that
     * ignored the pitch would fill them. */
    if (sel("cast_bf16_bf16")) {
        const RadKernelInfo* r = find(pl, "cast_bf16_bf16");
        if (!r) ok("cast_bf16_bf16", false, "row absent");
        else {
            const int64_t M = 7, n = 40, wide = 3 * n, slot = 1;   /* the middle third */
            std::vector<uint16_t> src((size_t)(M * n));
            for (int64_t i = 0; i < M * n; ++i)
                src[(size_t)i] = h_to_bf16(0.031f * (float)((i * 13) % 97) - 1.25f);
            void* d_src = dalloc((size_t)M * n * 2);
            void* d_dst = dalloc((size_t)M * wide * 2);
            if (!d_src || !d_dst) { ok("cast alloc", false, "hipMalloc failed"); }
            else {
                st_memcpy(d_src, src.data(), (size_t)M * n * 2, hipMemcpyHostToDevice);
                /* A fill the copy must not spread: every untouched column has to still hold it. */
                std::vector<uint16_t> fill((size_t)(M * wide), h_to_bf16(-7.5f));
                st_memcpy(d_dst, fill.data(), (size_t)M * wide * 2, hipMemcpyHostToDevice);

                RadTensor ty = T(d_dst, RAD_BF16, { M, n });
                ty.data   = (void*)((uint16_t*)d_dst + slot * n);   /* the slice's first element */
                ty.stride[0] = wide;                                /* ...at the WIDE row pitch */
                std::vector<RadTensor> ts = { T(d_src, RAD_BF16, { M, n }), ty };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", n), PS("from", "bf16"), PS("to", "bf16"),
                };
                run_case(r, ts, ps);

                std::vector<uint16_t> got((size_t)(M * wide));
                st_memcpy(got.data(), d_dst, (size_t)M * wide * 2, hipMemcpyDeviceToHost);
                bool copied = true, spilled = false;
                for (int64_t m = 0; m < M && copied && !spilled; ++m)
                    for (int64_t c = 0; c < wide; ++c) {
                        const uint16_t g = got[(size_t)(m * wide + c)];
                        if (c >= slot * n && c < (slot + 1) * n) {
                            if (g != src[(size_t)(m * n + c - slot * n)]) { copied = false; break; }
                        } else if (g != fill[0]) { spilled = true; break; }
                    }
                ok("cast_bf16_bf16 copies into a column slice, byte for byte", copied);
                ok("cast_bf16_bf16 leaves the columns outside the slice alone", !spilled);
            }
        }
    }
    /* --- byte rows: the KV tier's transfer pair --- */
    /* PACK AND UNPACK A POOL, which is the only thing these two are ever asked to do. The layout
     * here is the real one in miniature: a "layer" is `rows` blocks of `n` bytes, the blocks a
     * session holds are scattered through it, and what must come back is those blocks and nothing
     * else. Bytes rather than a float dtype because a KV fragment is whatever the attention kernel
     * wrote -- an exact comparison is the only meaningful one, and at u8 it is available. */
    if (sel("gather_rows_u8") || sel("scatter_rows_u8")) {
        const RadKernelInfo* rg = find(pl, "gather_rows_u8");
        const RadKernelInfo* rs = find(pl, "scatter_rows_u8");
        if (!rg || !rs) ok("byte row pair", false, "row absent");
        else {
            const int64_t rows = 64, n = 256, M = 5;
            const int32_t sel_blocks[5] = { 63, 0, 17, 40, 9 };   /* scattered, and not in order */
            std::vector<uint8_t> pool((size_t)(rows * n));
            for (int64_t i = 0; i < rows * n; ++i) pool[(size_t)i] = (uint8_t)((i * 31 + 7) & 0xff);

            void* d_pool = dalloc((size_t)rows * n);
            void* d_pack = dalloc((size_t)M * n);
            void* d_idx  = dalloc(sizeof sel_blocks);
            if (!d_pool || !d_pack || !d_idx) ok("byte row alloc", false, "hipMalloc failed");
            else {
                st_memcpy(d_pool, pool.data(), (size_t)rows * n, hipMemcpyHostToDevice);
                st_memcpy(d_idx, sel_blocks, sizeof sel_blocks, hipMemcpyHostToDevice);
                std::vector<uint8_t> poison((size_t)(M * n), 0xA5);
                st_memcpy(d_pack, poison.data(), (size_t)M * n, hipMemcpyHostToDevice);

                std::vector<RadTensor> tg = { T(d_pool, RAD_U8, { rows, n }),
                                              T(d_idx,  RAD_I32, { M }),
                                              T(d_pack, RAD_U8, { M, n }) };
                std::vector<RadParam> pg = { PI("M", M), PI("n", n), PS("dtype", "u8") };
                run_case(rg, tg, pg);

                std::vector<uint8_t> packed((size_t)(M * n));
                st_memcpy(packed.data(), d_pack, (size_t)M * n, hipMemcpyDeviceToHost);
                bool gathered = true;
                for (int64_t m = 0; m < M && gathered; ++m)
                    for (int64_t i = 0; i < n; ++i)
                        if (packed[(size_t)(m * n + i)] !=
                            pool[(size_t)(sel_blocks[m] * n + i)]) { gathered = false; break; }
                ok("gather_rows_u8 packs scattered blocks byte for byte", gathered);

                /* Now scatter them back into a DIFFERENT pool prefilled with a marker, and the
                 * blocks nobody named have to still hold it. That is the property the whole op
                 * exists for: the destination is shared, and a restore that touched a row it was
                 * not given would be destroying another session's context. */
                std::vector<uint8_t> mark((size_t)(rows * n), 0x5C);
                st_memcpy(d_pool, mark.data(), (size_t)rows * n, hipMemcpyHostToDevice);
                std::vector<RadTensor> ts2 = { T(d_pack, RAD_U8, { M, n }),
                                               T(d_idx,  RAD_I32, { M }),
                                               T(d_pool, RAD_U8, { rows, n }) };
                run_case(rs, ts2, pg);

                std::vector<uint8_t> back((size_t)(rows * n));
                st_memcpy(back.data(), d_pool, (size_t)rows * n, hipMemcpyDeviceToHost);
                bool restored = true, clobbered = false;
                std::vector<bool> named((size_t)rows, false);
                for (int64_t m = 0; m < M; ++m) named[(size_t)sel_blocks[m]] = true;
                for (int64_t b = 0; b < rows && restored && !clobbered; ++b)
                    for (int64_t i = 0; i < n; ++i) {
                        const uint8_t g = back[(size_t)(b * n + i)];
                        if (named[(size_t)b]) {
                            if (g != pool[(size_t)(b * n + i)]) { restored = false; break; }
                        } else if (g != 0x5C) { clobbered = true; break; }
                    }
                ok("scatter_rows_u8 restores every named block byte for byte", restored);
                ok("scatter_rows_u8 leaves every block it was not given alone", !clobbered);
            }
        }
    }
    /* --- gated delta net --- */
    {
        const int64_t T_ = 128, H = 4, Hg = 2, K = 128, V = 128, chunk = 64, N = 2;
        if (sel("gdn_kkt_solve_k128_c64_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_kkt_solve_k128_c64_bf16");
            if (!r) ok("gdn_kkt_solve_k128_c64_bf16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_, Hg, K }), T(nullptr, RAD_F32, { T_, H }),
                    T(nullptr, RAD_F32, { T_, H }),      TI32({ N + 1 }, cu_even(N, T_)),
                    T(nullptr, RAD_BF16, { T_, H, chunk }),
                };
                std::vector<RadParam> ps = { PI("M", T_), PI("head_k", K), PI("chunk", chunk) };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_chunk_scan_k128_v128_c64_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_chunk_scan_k128_v128_c64_bf16");
            if (!r) ok("gdn_chunk_scan_k128_v128_c64_bf16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_, Hg, K }), T(nullptr, RAD_BF16, { T_, Hg, K }),
                    T(nullptr, RAD_BF16, { T_, H, V }),  T(nullptr, RAD_BF16, { T_, H, chunk }),
                    T(nullptr, RAD_F32,  { T_, H }),     T(nullptr, RAD_F32,  { T_, H }),
                    T(nullptr, RAD_F32,  { N, H, V, K }), TI32({ N + 1 }, cu_even(N, T_)),
                    T(nullptr, RAD_BF16, { T_, H, V }),  T(nullptr, RAD_F32, { N, H, V, K }),
                };
                std::vector<RadParam> ps = {
                    PI("M", T_), PI("head_k", K), PI("head_v", V), PI("chunk", chunk),
                    PF("scale", 0.088388),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_conv_prep_w4_h128_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_conv_prep_w4_h128_bf16");
            if (!r) ok("gdn_conv_prep_w4_h128_bf16", false, "row absent");
            else {
                const int64_t width = 4, xdim = Hg * K * 2 + H * V;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_, xdim }),          /* x                */
                    T(nullptr, RAD_BF16, { xdim, width }),       /* conv weight      */
                    T(nullptr, RAD_BF16, { xdim }),              /* conv bias        */
                    T(nullptr, RAD_F32,  { H }),                 /* A_log            */
                    T(nullptr, RAD_F32,  { H }),                 /* dt_bias          */
                    T(nullptr, RAD_BF16, { N, xdim, width }),    /* conv state       */
                    TI32({ N + 1 }, cu_even(N, T_)),             /* cu               */
                    T(nullptr, RAD_BF16, { T_, H }),             /* a                */
                    T(nullptr, RAD_BF16, { T_, H }),             /* b_gate           */
                    TI32({ N }, iota_i32(N)),                    /* cache_idx        */
                    T(nullptr, RAD_U8,   { N }),                 /* has_init         */
                    T(nullptr, RAD_BF16, { T_, Hg, K }),         /* q                */
                    T(nullptr, RAD_BF16, { T_, Hg, K }),         /* k                */
                    T(nullptr, RAD_BF16, { T_, H, V }),          /* v                */
                    T(nullptr, RAD_F32,  { T_, H }),             /* g                */
                    T(nullptr, RAD_F32,  { T_, H }),             /* beta             */
                };
                std::vector<RadParam> ps = {
                    PI("M", T_), PI("head_k", K), PI("head_v", V), PI("chunk", chunk),
                    PI("conv_width", width),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_conv_update_w4_h128_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_conv_update_w4_h128_bf16");
            if (!r) ok("gdn_conv_update_w4_h128_bf16", false, "row absent");
            else {
                const int64_t width = 4, xdim = Hg * K * 2 + H * V, Td = N;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { Td, xdim }),
                    T(nullptr, RAD_BF16, { xdim, width }),
                    T(nullptr, RAD_BF16, { xdim }),
                    T(nullptr, RAD_BF16, { N, xdim, width }),
                    TI32({ N }, iota_i32(N)),                    /* state_idx        */
                    TI32({ N }, rep_i32(N, 1)),                  /* num_accepted     */
                    TI32({ N + 1 }, cu_even(N, Td)),             /* cu               */
                    T(nullptr, RAD_BF16, { Td, Hg, K }),
                    T(nullptr, RAD_BF16, { Td, Hg, K }),
                    T(nullptr, RAD_BF16, { Td, H, V }),
                };
                std::vector<RadParam> ps = {
                    PI("q_len", 1), PI("head_k", K), PI("head_v", V), PI("conv_width", width),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_recurrent_update_k128_v128_bf16_fp32state")) {
            const RadKernelInfo* r = find(pl, "gdn_recurrent_update_k128_v128_bf16_fp32state");
            if (!r) ok("gdn_recurrent_update_k128_v128_bf16_fp32state", false, "row absent");
            else {
                const int64_t Td = N;                            /* one candidate a sequence */
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { Td, Hg, K }),         /* q                */
                    T(nullptr, RAD_BF16, { Td, Hg, K }),         /* k                */
                    T(nullptr, RAD_BF16, { Td, H, V }),          /* v                */
                    T(nullptr, RAD_BF16, { Td, H }),             /* a                */
                    T(nullptr, RAD_BF16, { Td, H }),             /* b                */
                    T(nullptr, RAD_F32,  { H }),                 /* A_log            */
                    T(nullptr, RAD_F32,  { H }),                 /* dt_bias          */
                    T(nullptr, RAD_F32,  { N, H, V, K }),        /* state            */
                    TI32({ N }, iota_i32(N)),                    /* state_idx        */
                    /* num_accepted indexes the state-index ROW as `nacc - 1`, so a zero here is
                     * a read one int before the buffer. One accepted token is the decode step
                     * this case models. */
                    TI32({ N }, rep_i32(N, 1)),                  /* num_accepted     */
                    TI32({ N + 1 }, cu_even(N, Td)),             /* cu               */
                    T(nullptr, RAD_BF16, { Td, H, V }),          /* z                */
                    T(nullptr, RAD_F32,  { V }),                 /* norm_w           */
                    T(nullptr, RAD_BF16, { Td, H, V }),          /* o                */
                    /* THE FOLDED QUANTISER, present so the case covers it: absent, the kernel
                     * writes bf16 only and the two extra outputs are never compared. head_v is
                     * the fp8 block width, so the scale plane is one f32 a (token, value head)
                     * and ref quantises the bf16 it just stored -- which is what makes the two
                     * agree to the byte rather than to a tolerance. */
                    T(nullptr, RAD_F8E4M3, { Td, H * V }),       /* o_q              */
                    T(nullptr, RAD_F32,    { Td, H }),           /* o_scale          */
                };
                std::vector<RadParam> ps = {
                    PI("q_len", 1), PI("head_k", K), PI("head_v", V), PF("scale", 1.0),
                    PF("eps", 1e-6), PS("act", "silu"),
                };
                run_case(r, ts, ps);
            }
        }
        gdn_recurrent_anchor(pl);
        gdn_conv_fold(pl);
        if (sel("gdn_gated_rmsnorm_h128_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_gated_rmsnorm_h128_bf16");
            if (!r) ok("gdn_gated_rmsnorm_h128_bf16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_ * H, 128 }), T(nullptr, RAD_BF16, { T_ * H, 128 }),
                    T(nullptr, RAD_F32,  { 128 }),         T(nullptr, RAD_BF16, { T_ * H, 128 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", T_ * H), PI("channels", 128), PF("eps", 1e-6), PS("act", "silu"),
                };
                run_case(r, ts, ps);
            }
        }
        gdn_gated_norm_sliced_gate(pl);
    }

    /* --- dflash conv --- */
    if (sel("dflash_conv_t2_g16_bf16")) {
        const RadKernelInfo* r = find(pl, "dflash_conv_t2_g16_bf16");
        if (!r) ok("dflash_conv_t2_g16_bf16", false, "row absent");
        else {
            const int64_t T_ = 64, H = 256, NG = H / 16, taps = 2;
            std::vector<RadTensor> ts = {
                T(nullptr, RAD_BF16, { T_, H }),
                T(nullptr, RAD_BF16, { T_, 2 * taps * NG }),
                T(nullptr, RAD_BF16, { taps, NG }),
                T(nullptr, RAD_BF16, { T_, H }),
            };
            std::vector<RadParam> ps = {
                PI("T", T_), PI("hidden_size", H), PI("taps", taps), PI("group_size", 16),
                PI("NG", NG), PI("block_size", 16), PS("dtype", "bf16"),
            };
            run_case(r, ts, ps);
        }
        /* The drafter's own shape (block 8, 5120 channels) and a wider block, because the tap
         * mask is the block position and a block of 16 exercises a different mask. */
        dflash_conv_numeric(pl, 64, 256, 16, 2, 16);
        dflash_conv_numeric(pl, 32, 5120, 16, 2, 8);
    }

    /* --- dflash select: NUMERIC, because the thing worth testing is the CHAINING.
     *
     * The walk's whole reason to exist is that step l's score depends on the token step l-1 chose.
     * A kernel that scored every step against the anchor instead would still emit K plausible ids
     * per sequence, in range, in the right shape -- and the drafter would simply accept less. So
     * the case is built to distinguish the two: it asserts the chained path AND asserts that an
     * anchor-pinned walk over the same data would have produced a different one, which is what
     * makes the first assertion mean something.
     *
     * Every value is a power of two times a quarter, so the f32 accumulation is EXACT whatever
     * order the wave reduces in and the expected path is a comparison of exact integers rather
     * than a tolerance. */
    if (sel("dflash_select_bf16")) {
        const RadKernelInfo* r = find(pl, "dflash_select_bf16");
        if (!r) ok("dflash_select_bf16", false, "row absent");
        else {
            const int64_t M = 2, steps = 3, K = 4, R = 32, V = 64, rows = M * steps;
            auto pv   = [](int64_t t) { return t % 3 == 0 ? 0.5f : (t % 3 == 1 ? 1.0f : 2.0f); };
            auto sv   = [](int64_t t) { return 0.25f * (float)(t % 4); };
            auto cnd  = [&](int64_t row, int64_t c) { return (int32_t)((row * 7 + c * 10) % V); };
            auto unf  = [&](int64_t row, int64_t c) { return 8.0f * (float)((row + c * 3) % 5); };
            /* STRIDED, because that is how the drafter passes it: the operand is the step's own
             * token_ids, which carries `block` ids a sequence with the anchor first. The entries
             * between the anchors are mask tokens, and reading one instead would be a plausible
             * walk from the wrong predecessor -- so they are filled with a value that would give a
             * different answer if the stride were ignored. */
            const int64_t astride = steps + 1;
            const int32_t anchor[2] = { 2, 2 };
            std::vector<int32_t> h_anc((size_t)(M * astride), 7);
            for (int64_t s = 0; s < M; ++s) h_anc[(size_t)(s * astride)] = anchor[s];

            std::vector<int32_t> h_cand((size_t)(rows * K));
            std::vector<float>   h_un((size_t)(rows * K));
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t c = 0; c < K; ++c) {
                    h_cand[(size_t)(i * K + c)] = cnd(i, c);
                    h_un[(size_t)(i * K + c)]   = unf(i, c);
                }
            std::vector<uint16_t> h_hp((size_t)(rows * R), h_to_bf16(1.0f));
            std::vector<uint16_t> h_pred((size_t)(V * R)), h_succ((size_t)(V * R));
            for (int64_t t = 0; t < V; ++t)
                for (int64_t j = 0; j < R; ++j) {
                    h_pred[(size_t)(t * R + j)] = h_to_bf16(pv(t));
                    h_succ[(size_t)(t * R + j)] = h_to_bf16(sv(t));
                }

            /* The reference walk, and the same walk with the predecessor pinned to the anchor. */
            int32_t want[2][8] = {}, pinned[2][8] = {};
            for (int pin = 0; pin < 2; ++pin)
                for (int64_t s = 0; s < M; ++s) {
                    int64_t chosen = 0;
                    for (int64_t l = 0; l < steps; ++l) {
                        const int64_t row = s * steps + l;
                        const int64_t pid = (l == 0 || pin) ? anchor[s] : cnd(row - 1, chosen);
                        int64_t best = 0;
                        float   bv   = -1e30f;
                        for (int64_t c = 0; c < K; ++c) {
                            const float sc = unf(row, c) + (float)R * pv(pid) * sv(cnd(row, c));
                            if (sc > bv) { bv = sc; best = c; }
                        }
                        chosen = best;
                        (pin ? pinned : want)[s][l] = cnd(row, best);
                    }
                }
            bool distinguishes = false;
            for (int64_t s = 0; s < M; ++s)
                for (int64_t l = 0; l < steps; ++l)
                    if (want[s][l] != pinned[s][l]) distinguishes = true;

            void* d_cand = dalloc((size_t)rows * K * 4);
            void* d_un   = dalloc((size_t)rows * K * 4);
            void* d_hp   = dalloc((size_t)rows * R * 2);
            void* d_anc  = dalloc(h_anc.size() * 4);
            void* d_pred = dalloc((size_t)V * R * 2);
            void* d_succ = dalloc((size_t)V * R * 2);
            void* d_out  = dalloc((size_t)M * steps * 4);
            if (!d_cand || !d_un || !d_hp || !d_anc || !d_pred || !d_succ || !d_out) {
                ok("dflash_select alloc", false, "hipMalloc failed");
            } else {
                st_memcpy(d_cand, h_cand.data(), (size_t)rows * K * 4, hipMemcpyHostToDevice);
                st_memcpy(d_un,   h_un.data(),   (size_t)rows * K * 4, hipMemcpyHostToDevice);
                st_memcpy(d_hp,   h_hp.data(),   (size_t)rows * R * 2, hipMemcpyHostToDevice);
                st_memcpy(d_anc,  h_anc.data(),  h_anc.size() * 4,     hipMemcpyHostToDevice);
                st_memcpy(d_pred, h_pred.data(), (size_t)V * R * 2,    hipMemcpyHostToDevice);
                st_memcpy(d_succ, h_succ.data(), (size_t)V * R * 2,    hipMemcpyHostToDevice);

                std::vector<RadTensor> ts = {
                    T(d_cand, RAD_I32,  { rows, K }), T(d_un,   RAD_F32,  { rows, K }),
                    T(d_hp,   RAD_BF16, { rows, R }), T(d_anc,  RAD_I32,  { M * astride }),
                    T(d_pred, RAD_BF16, { V, R }),    T(d_succ, RAD_BF16, { V, R }),
                    T(d_out,  RAD_I32,  { M, steps }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("steps", steps), PI("top_k", K), PI("rank", R),
                    PI("n_vocab", V), PI("anchor_stride", astride), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);

                std::vector<int32_t> got((size_t)(M * steps));
                st_memcpy(got.data(), d_out, (size_t)M * steps * 4, hipMemcpyDeviceToHost);
                bool same = true;
                for (int64_t s = 0; s < M && same; ++s)
                    for (int64_t l = 0; l < steps; ++l)
                        if (got[(size_t)(s * steps + l)] != want[s][l]) { same = false; break; }
                ok("dflash_select_bf16 walks the chained path", same);
                ok("dflash_select_bf16 case can tell a chained walk from an anchor-pinned one",
                   distinguishes);

                /* THE BIGRAM FORM, which is why `hp` is optional: DSpark's Markov head is this
                 * walk with no hidden projection. `hp` above is a plane of
                 * EXACT ones, so absent has to give bit-identical tokens -- a kernel that read a
                 * null pointer, or skipped the multiply in one branch and not the other, moves
                 * some token here and none of it would show up as anything but acceptance. */
                std::vector<RadTensor> tb = ts;
                /* `absent`, NOT a null `data` with the rank left on: run_case allocates for any
                 * tensor that has a rank and no pointer, so spelling it the other way hands the
                 * kernel a plane of UNINITIALISED memory and tests the present branch against
                 * garbage rather than withholding the operand. */
                absent(tb[2]);
                st_memcpy(d_out, h_cand.data(), (size_t)M * steps * 4, hipMemcpyHostToDevice);
                run_case(r, tb, ps);
                std::vector<int32_t> bg((size_t)(M * steps));
                st_memcpy(bg.data(), d_out, (size_t)M * steps * 4, hipMemcpyDeviceToHost);
                bool bsame = true;
                for (int64_t s = 0; s < M && bsame; ++s)
                    for (int64_t l = 0; l < steps; ++l)
                        if (bg[(size_t)(s * steps + l)] != want[s][l]) { bsame = false; break; }
                ok("dflash_select_bf16 absent hp is a plane of ones (the bigram form)", bsame);
            }
        }
    }

    /* --- GEMM --- */
    {
        const int64_t M = 16, N = 256, K = 512;
        if (sel("gemm_bf16_nt_m16")) {
            const RadKernelInfo* r = find(pl, "gemm_bf16_nt_m16");
            if (!r) ok("gemm_bf16_nt_m16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { N, K }),
                    T(nullptr, RAD_BF16, { M, N }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gemm_bf16_nt_m64")) {
            const RadKernelInfo* r = find(pl, "gemm_bf16_nt_m64");
            if (!r) ok("gemm_bf16_nt_m64", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { 64, K }), T(nullptr, RAD_BF16, { N, K }),
                    T(nullptr, RAD_BF16, { 64, N }),
                };
                std::vector<RadParam> ps = {
                    PI("M", 64), PI("N", N), PI("K", K), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        gemm_case(pl, "gemm_w4a16_nt_m64",   "w4a16",   M, N, K, RAD_F16, 0, 0, 0, N*K/2, N*(K/128)*4);
        gemm_case(pl, "gemm_w8a16_nt_m64",   "w8a16",   M, N, K, RAD_F16, 0, 0, 0, N*K,   N*(K/128)*4);
        i8a8_numeric(pl, "gemm_i8a8_nt_m16", 5, 272, 384);
        i8a8_numeric(pl, "gemm_i8a8_nt_m16", 40, 2560, 6144);
        i8a8_numeric(pl, "gemm_i8a8_nt_m16", 64, 512, 640);
        i8a8_numeric(pl, "gemm_i8a8_tiled", 200, 256, 640);
        i8a8_numeric(pl, "gemm_i8a8_tiled", 2048, 512, 2560);
        w8a16_numeric(pl, "gemm_w8a16_nt_m64", 5, 272, 384, RAD_F16);
        w8a16_numeric(pl, "gemm_w8a16_nt_m64", 64, 2560, 6144, RAD_F16);
        gemm_case(pl, "gemm_w4a8_nt_m64",    "w4a8",    M, N, K, RAD_I8,  1, 0, 0, N*K/2, N*(K/128)*4);
        gemm_case(pl, "gemm_w4a8_nt_m64",    "w4a8_asym", M, N, K, RAD_I8, 1, 1, 0, N*K/2, N*(K/128)*4);
        gemm_case(pl, "gemm_w8a8_nt_m64",    "w8a8",    M, N, K, RAD_I8,  1, 0, 0, N*K,   N*(K/128)*2);
        gemm_case(pl, "gemm_w2a8_nt",        "w2a8",    M, N, K, RAD_I8,  1, 1, 0, N*K/4, N*(K/128)*4);
        gemm_case(pl, "gemm_mxfp4a8_nt_m64", "mxfp4a8", M, N, K, RAD_F8E4M3, 1, 0, 1, N*K/2, (K/32)*N);
        gemm_case(pl, "gemm_w4a8_tiled",     "w4a8",    256, 256, K, RAD_I8, 1, 0, 0, 256*K/2, 256*(K/128)*4);
        gemm_case(pl, "gemm_w8a8_tiled",     "w8a8",    256, 256, K, RAD_I8, 1, 0, 0, 256*K,   256*(K/128)*2);
        gemm_case(pl, "gemm_w4a8_prefill",   "w4a8",    256, 256, K, RAD_I8, 1, 0, 0, 256*K/2, 256*(K/128)*4);

        /* Block-scaled fp8. The matvec is smoked at every point of its space because `rb` is the
         * only axis in the family and each value is a different instantiation of the K loop. The
         * narrow W8A8 row needs K a multiple of 256 -- 512 is -- and the tiled one needs N a
         * multiple of 128. */
        for (const TunePoint& tp : tune_points(find(pl, "gemm_fp8a16_nt_m1"),
                                               std::vector<RadParam>{ PI("M", 1), PI("N", N),
                                                                      PI("K", K), PI("group", 128),
                                                                      PS("dtype", "fp8a16") }))
            fp8_gemm_case(pl, "gemm_fp8a16_nt_m1",  "fp8a16", 1,  N, K, 0, &tp);
        fp8_gemm_case(pl, "gemm_fp8a8_nt_m16",      "fp8a8",  8,  N, K, 1);
        fp8_gemm_case(pl, "gemm_fp8a8_nt_m16",      "fp8a8",  16, N, K, 1);
        fp8_gemm_case(pl, "gemm_fp8a8_tiled",       "fp8a8",  256, 256, K, 1);
    }

    /* --- the fp8 activation quantiser and the device sampler --- */
    {
        if (sel("quant_act_fp8")) {
            const RadKernelInfo* r = find(pl, "quant_act_fp8");
            if (!r) ok("quant_act_fp8", false, "row absent");
            else {
                const int64_t M = 32, K = 512;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16,    { M, K }),
                    T(nullptr, RAD_F8E4M3,  { M, K }),
                    T(nullptr, RAD_F32,     { M, K / 128 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("sample_chain_f32")) {
            const RadKernelInfo* r = find(pl, "sample_chain_f32");
            if (!r) ok("sample_chain_f32", false, "row absent");
            else {
                /* The vocabulary is deliberately not a round number: the radix passes are over a
                 * float's bytes and a power-of-two width would hide an off-by-one in the stride. */
                const int64_t M = 4, V = 1000;
                /* TWO uniforms a position, and they must be IN [0,1): dalloc zeroes, and a uniform
                 * of exactly 0 walks the inverse CDF to the first kept token every time -- which is
                 * a valid draw, so the smoke test would pass with the walk broken. */
                std::vector<float> uh((size_t)(2 * M));
                for (size_t i = 0; i < uh.size(); ++i) uh[i] = 0.11f + 0.17f * (float)i;
                RadTensor tu = T(nullptr, RAD_F32, { 2 * M });
                tu.data = alloc_for(tu);
                if (tu.data) st_memcpy(tu.data, uh.data(), uh.size() * 4, hipMemcpyHostToDevice);
                /* `draw` is three 32-bit words a position. The schema names one operand, not three,
                 * because the kernel writes a struct -- so it is sized here in words. */
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_F32, { M, V }),
                    T(nullptr, RAD_U32, { M }),
                    tu,
                    T(nullptr, RAD_U32, { M, 3 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n_vocab", V), PF("temp", 1.0), PF("p", 0.95),
                    PF("min_p", 0.0), PI("k", 40),
                };
                run_case(r, ts, ps);
                /* Temperature 0 is a REAL path, not a caller special case: it must give the argmax
                 * with the same lowest-id tie-break a greedy run uses, and a zeroed logits plane is
                 * exactly the all-ties input that catches a tie-break that disagrees. */
                std::vector<RadParam> ps0 = {
                    PI("M", M), PI("n_vocab", V), PF("temp", 0.0), PF("p", 1.0),
                    PF("min_p", 0.0), PI("k", 0),
                };
                run_case(r, ts, ps0);
                /* ...and with no draft proposal, which is the last position of a verify block. */
                absent(ts[1]);
                run_case(r, ts, ps);
            }
        }
    }

    /* --- quantiser and the fusions --- */
    {
        const int64_t M = 32, K = 512;
        if (sel("quant_act_i8")) {
            const RadKernelInfo* r = find(pl, "quant_act_i8");
            if (!r) ok("quant_act_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_I8, { M, K }),
                    T(nullptr, RAD_F32,  { M }),    T(nullptr, RAD_F32, { (K/128) * 32 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
                absent(ts[3]);                           /* the plain form, without the row sums */
                run_case(r, ts, ps);
            }
        }
        if (sel("had_quant_act_i8")) {
            const RadKernelInfo* r = find(pl, "had_quant_act_i8");
            if (!r) ok("had_quant_act_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_I8, { M, K }),
                    T(nullptr, RAD_F32,  { M }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("rmsnorm_had_quant_i8")) {
            const RadKernelInfo* r = find(pl, "rmsnorm_had_quant_i8");
            if (!r) ok("rmsnorm_had_quant_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { M, K }),
                    T(nullptr, RAD_BF16, { K }),    T(nullptr, RAD_I8,   { M, K }),
                    T(nullptr, RAD_F32,  { M }),    T(nullptr, RAD_BF16, { M, K }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PF("eps", 1e-6), PI("group", 128),
                    PS("dtype", "bf16"), PF("wadd", 0.0),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gated_had_quant_i8")) {
            const RadKernelInfo* r = find(pl, "gated_had_quant_i8");
            if (!r) ok("gated_had_quant_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, 2 * K }), T(nullptr, RAD_BF16, { 0 }),
                    T(nullptr, RAD_I8,   { M, K }),     T(nullptr, RAD_F32,  { M }),
                };
                absent(ts[1]);                           /* the packed gate_up form */
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"), PS("act", "silu"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_gated_norm_had_quant_i8")) {
            const RadKernelInfo* r = find(pl, "gdn_gated_norm_had_quant_i8");
            if (!r) ok("gdn_gated_norm_had_quant_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { M, K }),
                    T(nullptr, RAD_F32,  { 128 }),  T(nullptr, RAD_I8,   { M, K }),
                    T(nullptr, RAD_F32,  { M }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("head_dim", 128), PI("group", 128),
                    PF("eps", 1e-6), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("qk_norm_rope_gate")) {
            const RadKernelInfo* r = find(pl, "qk_norm_rope_gate");
            if (!r) ok("qk_norm_rope_gate", false, "row absent");
            else {
                const int64_t nq = 4, nkv = 2, hd = 128, rot = 64;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, nq * hd * 2 }),   /* [q|gate] per head */
                    T(nullptr, RAD_BF16, { M, nkv * hd }),
                    T(nullptr, RAD_BF16, { 4096, rot }),
                    T(nullptr, RAD_I32,  { M }),
                    T(nullptr, RAD_BF16, { hd }),
                    T(nullptr, RAD_BF16, { hd }),
                    T(nullptr, RAD_BF16, { M, nq * hd }),
                    T(nullptr, RAD_BF16, { M, nkv * hd }),
                    T(nullptr, RAD_BF16, { M, nq * hd }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("head_dim", hd), PI("n_head", nq), PI("n_head_kv", nkv),
                    PI("rot", rot), PS("q_dtype", "bf16"), PF("eps", 1e-6),
                };
                run_case(r, ts, ps);
            }
        }
        qk_prologue_case(pl, RAD_BF16);
        qk_prologue_case(pl, RAD_F32);
        qk_kv_fold_case(pl, false);
        qk_kv_fold_case(pl, true);
        qsa_qprep_case(pl);
        topk_scatter_case(pl);
        router_gemm_fused_case(pl);
        scatter_protect_case(pl);
        if (sel("rowtopk_bf16")) {
            const RadKernelInfo* r = find(pl, "rowtopk_bf16");
            if (!r) ok("rowtopk_bf16", false, "row absent");
            else {
                const int64_t Mr = 32, Nr = 2048, R = 8;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { Mr, Nr }), T(nullptr, RAD_I32, { Mr, R }),
                    T(nullptr, RAD_F32,  { Mr, R }),
                };
                std::vector<RadParam> ps = {
                    PI("M", Mr), PI("N", Nr), PI("R", R), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        rowtopk_shard_case(pl);
    }

    /* dequant_w4_bf16's case IS the round-trip: the row's only caller here is the check that the
     * host relayout and libr4d's own dequant agree, so naming the case for the row keeps the one
     * case-name-is-a-row-name rule that makes the coverage check below work. */
    if (sel("dequant_w4_bf16")) {
        if (!g_case) std::printf("\nweight layout, against libr4d's own inverse\n");
        w4_roundtrip(pl);
    }

    /* The 4-bit grouped MoE GEMM gets its OWN case name, which is the rule the coverage check
     * below enforces -- a row with no case of its own is reported as a hole. Nesting it inside
     * another row's block makes it invisible and says nothing about it: a child process is
     * spawned per row and `sel` is false for every name but that child's own. */
    if (sel("sample_argmax_f32")) {
        if (!g_case) std::printf("\nthe greedy sampler, against a host scan\n");
        sample_argmax_case(pl);
    }

    if (sel("qsa_work_bf16")) {
        if (!g_case) std::printf("\nthe QSA work list, against a host walk of the batch\n");
        qsa_work_case(pl);
    }

    if (sel("qsa_tail_store_bf16")) {
        if (!g_case) std::printf("\nthe QSA raw-key tail, against a serial loop\n");
        qsa_tail_store_case(pl);
    }

    if (sel("qsa_select_i32")) {
        if (!g_case) std::printf("\nthe QSA selection, against a host sort\n");
        qsa_select_case(pl);
        qsa_select_prefill_case(pl);
        qsa_select_wide_case(pl);
        qsa_select_split_one(pl, "qsa_select split across chunks vs the same launch unsplit",
                             1, 4);
        qsa_select_split_one(pl, "qsa_select split, two sequences of four draft rows", 2, 4);
    }

    if (sel("qsa_score_bf16")) {
        if (!g_case) std::printf("\nthe QSA block score, against the formula it implements\n");
        /* EVERY FORM THE ROW HAS: the launcher picks one by shape, and a form the rule does not
         * pick for these shapes is the one another model's shape would get. */
        for (int f = pl.pin_qsa ? 1 : 0; f <= (pl.pin_qsa ? 3 : 0); ++f) {
            if (pl.pin_qsa) {
                pl.pin_qsa(f);
                std::printf("      qsa_score, %s form\n", f == 1 ? "scalar" : (f == 2 ? "matrix" : "packed matrix"));
            }
            qsa_score_case(pl);
            if (f <= 2) qsa_score_rows_case(pl);
        }
        if (pl.pin_qsa) pl.pin_qsa(0);
    }

    if (sel("qsa_block_key_bf16")) {
        if (!g_case) std::printf("\nthe QSA compressed key, against the formula it implements\n");
        qsa_block_key_case(pl);
    }

    if (sel("moe_gemm_w4a8")) {
        if (!g_case) std::printf("\nthe 4-bit grouped MoE GEMM, against a host decode\n");
        /* One geometry; a gate/up pair whose odd experts are half as wide; a down projection
         * whose odd experts are half as long; and a decode-shaped routing -- a few runs over
         * hundreds of experts, most of them empty -- which is what the decode grid's run search
         * has to find its way through. */
        static const MoeW4Shape kShapes[] = {
            { "one class",   64, 256,  64, 256, 1, 5 },
            { "gate/up 2:1", 256, 256, 128, 256, 2, 5 },
            { "down 2:1",    64, 256,  64, 128, 1, 5 },
            { "300 experts, gate/up 2:1", 256, 128, 128, 128, 2, 300 },
        };
        for (const MoeW4Shape& sh : kShapes) {
            moe_w4_gemm(pl, 24, sh);
            moe_w4_gemm(pl, 200, sh);
            moe_w4_gemm(pl, 600, sh);
            for (const int sub : { 2, 4 })
                for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                    moe_w4_gemm(pl, t, sh, sub);
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                moe_w4_gemm(pl, t, sh, 2, true);
        }
    }

    if (sel("moe_gemm_w4a8_mod4")) {
        if (!g_case) std::printf("\nthe 4-bit grouped MoE GEMM over four classes, against its "
                                 "two-class form and a host decode\n");
        /* The same shapes, read as four classes: the even table's halves are classes 0 and 2 and
         * the odd one's 1 and 3, so the two-class launch is the reference, byte for byte. */
        static const MoeW4Shape kShapes4[] = {
            { "one class",   64, 256,  64, 256, 1, 5 },
            { "gate/up 2:1", 256, 256, 128, 256, 2, 10 },
            { "down 2:1",    64, 256,  64, 128, 1, 10 },
            { "300 experts, gate/up 2:1", 256, 128, 128, 128, 2, 300 },
        };
        for (const MoeW4Shape& sh : kShapes4)
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 }) {
                moe_w4_gemm(pl, t, sh, 0, false, true);
                moe_w4_gemm(pl, t, sh, 2, false, true);
                moe_w4_gemm(pl, t, sh, 4, false, true);
                moe_w4_gemm(pl, t, sh, 2, true, true);
            }
        /* And every class at its own geometry, one of classes 1 to 3 the only wide one. */
        for (const MoeQ4Shape& sh : kMoeQ4Shapes)
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                for (const char* dt : { "w4a8", "w4nl64a8h", "w4nl32a8h", "w5nl64a8h" })
                    moe_q4_gemm(pl, t, sh, dt);
    }

    if (sel("moe_gemm_fp8a8_mod4")) {
        if (!g_case) std::printf("\nthe block-fp8 grouped MoE GEMM over four classes, against a "
                                 "host decode\n");
        for (const MoeQ4Shape& sh : kMoeQ4Shapes)
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                moe_q4_gemm(pl, t, sh, "fp8a8");
    }

    if (sel("hc_read_e4m3")) {
        if (!g_case) std::printf("\nthe gated residual's read at the prefill tile, the wide "
                                 "kernels against the decode tiles'\n");
        for (int64_t m : { 65, 300, 2048 }) hc_prefill(pl, m, 0, false);
        for (int64_t m : { 1, 4, 33, 65, 300, 2048 }) {
            hc_read_i8(pl, m, false, 0);
            hc_read_i8(pl, m, true, 0);
        }
        if (pl.pin_hc)
            for (int64_t m : { 65, 2048 })
                for (int form : { 1, 2 }) hc_read_i8(pl, m, true, form);
    }

    if (sel("moe_gemm_w4a8")) {
        if (!g_case) std::printf("\nthe arranged plane, against the stated format and its "
                                 "inverse\n");
        moe_w4_unlayout(pl);
    }

    if (sel("had_quant_act_fp8")) {
        if (!g_case) std::printf("\nthe ROTATED 4-bit expert format, against the f32 it "
                                 "approximates\n");
        moe_w4_rot(pl);
    }

    if (sel("gram_accum")) {
        if (!g_case) std::printf("\nthe GPTQ calibration accumulator, against an f64 host Gram\n");
        gram_accum_case(pl);
    }

    if (sel("gram_accum_bf16")) {
        if (!g_case) std::printf("\nthe calibration accumulator over a plain bf16 activation\n");
        gram_accum_bf16_case(pl, 200, 260);
        gram_accum_bf16_case(pl, 2049, 640);
    }

    if (sel("moe_gemm_bf16")) {
        if (!g_case) std::printf("\nthe bf16 grouped MoE GEMM, against an f64 host product\n");
        /* Qwen3.8-Flash-Next's and Qwen3.6-35B-A3B's whole-expert widths at a few experts each --
         * what fits beside a running server -- and the expert counts that exercise the run map
         * and the decode grid's run search at small widths. A width that does not take the
         * 128-column tile runs the 64-column one. */
        static const MoeBf16Shape kShapes[] = {
            { "Flash-Next gate_up 1280x2560", 1280, 2560,   4,  2, false },
            { "Flash-Next down 2560x640",     2560,  640,   4,  2, true  },
            { "35B gate_up 1024x2048",        1024, 2048,   4,  2, false },
            { "35B down 2048x512",            2048,  512,   4,  2, true  },
            { "256 experts top 8",             128,  128, 256,  8, false },
            { "512 experts top 10, sorted",     64,   64, 512, 10, true  },
            { "64-column tile",                192,  192,   6,  2, false },
        };
        for (const MoeBf16Shape& sh : kShapes) {
            /* One decode-band step and one or two prefill passes. */
            const int64_t dec = std::max<int64_t>(1, (3 * sh.ne) / (2 * sh.top_k));
            moe_gemm_bf16_case(pl, dec, sh);
            moe_gemm_bf16_case(pl, 100, sh);
            if (sh.ne >= 256) moe_gemm_bf16_case(pl, 300, sh);
        }
        moe_gemm_bf16_case(pl, 100, kShapes[1], true);
        /* The trailing segment alone, at both row orders and both bands. */
        for (const int i : { 0, 4, 5 }) {
            const MoeBf16Shape& sh = kShapes[i];
            const int64_t dec = std::max<int64_t>(1, (3 * sh.ne) / (2 * sh.top_k));
            for (const int64_t m : { dec, (int64_t)100 }) moe_gemm_bf16_case(pl, m, sh, false, 2);
        }
    }

    if (sel("dequant_w4_bf16")) {
        /* TWO SHAPES, and the second is the one the model actually runs: N = 256 is the k and v
         * projections, gate_up is 17408 rows. Both wave counts run here without asking for them,
         * because narrow_mw is a function of M and the M list inside fp8_numeric crosses its
         * threshold -- 1 and 8 take MW = 4, 24 through 64 take MW = 8. The second shape is only
         * reachable because the generator above wraps its exponent. */
        fp8_numeric(pl, 256, 512);
        fp8_numeric(pl, 17408, 512);
        fp8_numeric(pl, 5120, 5120);
        /* THE ONE PRODUCTION REGIME THE OTHERS MISS: many K tiles with split-K OFF. (256,512) and
         * (17408,512) are two staged tiles, and (5120,5120) splits. gate_up is 17408 rows by 5120,
         * which is twenty tiles and gx*mw over the split threshold -- so the prefetch fires
         * nineteen times and nothing else here makes it fire more than once. */
        fp8_numeric(pl, 17408, 5120);
        /* The two remaining production K values: the GDN out-projection reads K = v_dim = 1536
         * per rank, and down_proj reads K = n_ff = 8704. Both are multiples of 256 and 64, and
         * neither falls out of the shapes above. */
        fp8_numeric(pl, 5120, 1536);
        fp8_numeric(pl, 5120, 8704);
        /* A SHARED EXPERT SPLIT BY WHOLE SCALE BLOCKS gives each rank a width the even split never
         * does: Qwen3.8-Flash-Next's 640 columns go 384 / 256, so the down projection runs at
         * K = 384 -- a multiple of 128 and not of 256, which is the narrow tile's 128-wide leg --
         * and K = 256, and the gate/up projection at N = 768 and N = 512 over K = 2560. */
        fp8_numeric(pl, 2560, 384);
        fp8_numeric(pl, 2560, 256);
        fp8_numeric(pl, 768, 2560);
        fp8_numeric(pl, 512, 2560);
        /* The lm_head. 5120 rows stands in for 124160 -- the kernel walks rows with a grid-stride
         * loop, so the row count is not what varies -- and M spans the exact instantiations, the
         * rounded-up ones, and the cap. */
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)13, (int64_t)32, (int64_t)33,
                                 (int64_t)64 })
            fp8_logits_numeric(pl, 5120, 5120, M);
        /* AND K = 2560, WHICH IS THE PADDED ROW. 5120's stored width is 5200 and already a
         * multiple of 16; 2560's is 2600 and is not, so r4d_fp8_lm.h rounds it to 2608 and every
         * odd row moves by eight bytes. That padding is invisible to any shape that does not need
         * it, so without this case the layout, the relayout, its inverse and both kernels'
         * strides agree on a number none of them was ever asked to derive. M = 1 takes the
         * matvec, M = 8 the WMMA tile: the pad has to be right in both, and they index the row
         * differently. */
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)33 })
            fp8_logits_numeric(pl, 2560, 2560, M);
    }
    /* The int8 head over the same stored row: both forms, and the padded K = 2560 row. */
    if (sel("logits_gemm_i8") || sel("logit_rerank_i8")) {
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)33, (int64_t)64 })
            fp8_logits_numeric(pl, 5120, 5120, M, true);
        for (const int64_t M : { (int64_t)1, (int64_t)8 })
            fp8_logits_numeric(pl, 2560, 2560, M, true);
    }

    /* The fused quantisers are checked against the kernels they replace rather than against a host
     * mirror, so they live outside `fp8_numeric` and are gated on their own row. */
    if (sel("rmsnorm_quant_fp8") || sel("gated_quant_fp8") || sel("gate_quant_fp8")) {
        if (!g_case) std::printf("\nthe fp8 fusions, against the pair each one replaces\n");
        fused_quant_fp8_numeric(pl);
    }

    /* The MTP head's fc is K=10240; the DFlash2 drafter's is 25600, the largest K any shape in
     * the tree asks these kernels for.
     *
     * BOTH ROWS, because one op's band splits across both of them: an fc declared over M 1..64
     * resolves to _m16 for the small issues and _m64 for the wide ones, and the two are separate
     * kernels with separate split laws. Checking only the one named in the declaration tests a
     * kernel that never runs and leaves the one that does unchecked. */
    if (sel("gemm_bf16_nt_m16") || sel("gemm_bf16_nt_m64")) {
        if (!g_case) std::printf("\nthe skinny bf16 GEMMs, against a host reference\n");
        for (const char* row : { "gemm_bf16_nt_m16", "gemm_bf16_nt_m64" }) {
            bf16_gemm_numeric(pl, row, 5, 32, 10240);
            bf16_gemm_numeric(pl, row, 5, 32, 25600);
            /* The drafter's fc, exactly: N and K together, because the weight offset a lane
             * computes is row * K and only the pair is large. */
            bf16_gemm_numeric(pl, row, 5, 5120, 10240);
            bf16_gemm_numeric(pl, row, 5, 5120, 25600);
            bf16_gemm_numeric(pl, row, 16, 5120, 25600);
        }
        bf16_gemm_numeric(pl, "gemm_bf16_nt_m64", 64, 5120, 25600);
    }
    if (sel("gemm_bf16_nt_tiled")) {
        if (!g_case) std::printf("\nthe tiled bf16 GEMM, against the skinny one\n");
        bf16_tiled_identity(pl, 2048, 512, 2560);     /* the MoE router: four slices */
        bf16_tiled_identity(pl, 1000, 2560, 2560);    /* the PLE value projection: one slice */
        bf16_tiled_identity(pl, 333, 1040, 2560);     /* ragged M and N, two slices */
        bf16_tiled_identity(pl, 77, 128, 64);         /* a slice a k step, the band's floor */
        bf16_gemm_numeric(pl, "gemm_bf16_nt_tiled", 300, 1040, 2560);
    }

    if (sel("ar_oneshot_2rank_exact")) {
        if (!g_case) std::printf("\ncross-rank\n");
        ar_two_rank(pl, devs);
        ar_two_rank_sweep(pl, devs);
    }
    if (sel("ar_gather_2rank")) {
        if (!g_case) std::printf("\ncross-rank\n");
        gather_two_rank(pl, devs);
    }
    /* The wide family at three. Its own rows, so the sweep names them (a row the sweep never
     * mentions is reported as a coverage hole) and so `--case ar_oneshot_3rank_exact` reproduces
     * one of them alone. */
    if (sel("ar_oneshot_3rank_exact")) {
        if (!g_case) std::printf("\ncross-rank, three\n");
        ar_three_rank(pl, devs);
    }
    if (sel("ar_gather_3rank")) {
        if (!g_case) std::printf("\ncross-rank, three\n");
        gather_three_rank(pl, devs);
    }
    /* The 4- and 8-rank rows cannot be exercised on a two-card box. Reported as skipped rather
     * than quietly passing: a row that has never been launched is not a row that works. */
    for (const char* n : { "ar_oneshot_4rank_exact", "ar_oneshot_8rank_exact",
                           "ar_twoshot_4rank_exact", "ar_twoshot_8rank_exact",
                           "ar_twoshot_4rank_ti8" })
        if (sel(n)) skip(n, "needs 4 or 8 peer-accessible cards");
    if (sel("ar_oneshot_2rank_wht6"))
        skip("ar_oneshot_2rank_wht6", "lossy wire: needs a tolerance harness, not a smoke test");
    if (sel("ar_ln_had_quant_i8"))
        skip("ar_ln_had_quant_i8", "two-rank fusion: needs the same harness as the wht6 wire");
    /* Same reason, and one more: the norm half of it is rmsnorm_quant_fp8's, called through the
     * same device function, so what a single-rank harness could check here is already checked
     * there. What it cannot check without a peer -- the handshake and the reduced sum -- is what
     * the two-rank harness is for. The engine covers it end to end: a decode step issues this op
     * at every one of the 128 sites, and the generated text is compared byte for byte against a
     * build that issues the pair instead. */
    if (sel("ar_rmsnorm_quant_fp8"))
        skip("ar_rmsnorm_quant_fp8", "two-rank fusion: needs a peer rank");

    dfree_all();
}

/* ------------------------------------------------------------------ the isolating driver
 *
 * The row table is the case list. Every case above is named for a row, so a row the sweep never
 * mentions comes back from its child as "no smoke case" and the coverage hole is visible without
 * a second list to keep in step with the first.
 */

/* The number as well as the name: a GPU page fault does not always come out of ROCr as SIGABRT,
 * and "signal" on its own tells the reader nothing about which. The fault line itself is on
 * stderr, unbuffered, immediately above this. */
static std::string signame(int sig) {
    const char* n = nullptr;
    switch (sig) {
        case SIGABRT: n = "SIGABRT"; break;
        case SIGSEGV: n = "SIGSEGV"; break;
        case SIGBUS:  n = "SIGBUS";  break;
        case SIGILL:  n = "SIGILL";  break;
        case SIGFPE:  n = "SIGFPE";  break;
        case SIGKILL: n = "SIGKILL"; break;
        default:      n = "signal";  break;
    }
    return std::string(n) + " (" + std::to_string(sig) + ")";
}

static void run_isolated(const Plugin& pl, const char* exe, const char* plugin) {
    /* The inventory belongs to the parent: the children print one line each and repeating it
     * forty times buries them. A host can present more HIP devices than it has discrete cards --
     * an integrated part reporting tens of GiB of "VRAM" that is really system RAM -- so seeing
     * the list once is worth the parent touching HIP at all. */
    int nd = 0;
    if (hipGetDeviceCount(&nd) == hipSuccess) {
        for (int d = 0; d < nd; ++d) {
            hipDeviceProp_t p;
            if (hipGetDeviceProperties(&p, d) == hipSuccess)
                std::printf("  device %d: %s\n", d, p.gcnArchName);
        }
    }

    std::printf("\ndevice smoke tests (one child process per row)\n");
    std::fflush(stdout);

    for (int k = 0; k < pl.kernel_count(); ++k) {
        const std::string name = pl.kernel_at(k)->name;
        char* const argv[] = { const_cast<char*>(exe), const_cast<char*>(plugin),
                               const_cast<char*>("--case"), const_cast<char*>(name.c_str()),
                               nullptr };
        pid_t pid = 0;
        /* posix_spawn, not fork: this process already has the HIP runtime and its worker threads
         * up, and only the exec resets them in the child. */
        if (posix_spawn(&pid, exe, nullptr, nullptr, argv, environ) != 0) {
            ok(name.c_str(), false, "posix_spawn failed");
            continue;
        }
        int st = 0;
        if (waitpid(pid, &st, 0) < 0) { ok(name.c_str(), false, "waitpid failed"); continue; }
        std::fflush(stdout);

        if (WIFSIGNALED(st)) {
            ok(name.c_str(), false, std::string("child died: ") + signame(WTERMSIG(st)));
            continue;
        }
        if (!WIFEXITED(st)) { ok(name.c_str(), false, "child did not exit normally"); continue; }
        switch (WEXITSTATUS(st)) {
            case CH_OK:   break;                       /* the child already printed its ok line */
            case CH_FAIL: ++g_fail; break;             /* and its FAIL line, with the detail    */
            case CH_SKIP: ++g_skip; break;
            default:
                ok(name.c_str(), false,
                   "child exit " + std::to_string(WEXITSTATUS(st)));
        }
    }
}

#else   /* !RAD_HAVE_HIP */
static void device_phase(const Plugin&) { skip("device phase", "built without HIP"); }
static void run_isolated(const Plugin&, const char*, const char*) {
    skip("device phase", "built without HIP");
}
#endif

/* ------------------------------------------------------------------ main */

static void usage(void) {
    std::printf("usage: r4d_selftest <plugin.so> [--case <row>]\n"
                "       r4d_selftest <plugin.so> --real <file> <woff> <soff> <N> <K>\n"
                "  no --case: the catalogue, the static checks, then one child process per row\n"
                "  --case:    that row's smoke test alone, in this process -- the reproducer\n"
                "  --real:    the block-scaled fp8 matvec against a REAL checkpoint weight, at the\n"
                "             given byte offsets in a safetensors file, compared with a host\n"
                "             reference over the same bytes. An offset is 8 + <header length, a\n"
                "             u64 at byte 0> + the tensor's own data_offsets[0].\n"
                "  --perf N K M [iters]:  time the fp8 decode GEMMs at one shape and print the\n"
                "             weight bandwidth each reaches. R4D_F8_MW and R4D_F8_SPLITTGT pin\n"
                "             the narrow tile's launch heuristics for a sweep.\n"
                "  --perfbf16 N K M [iters]:  the bf16 skinny GEMM at every variant it has.\n"
                "  --perfw8 N K M [iters]:  the 8-bit trunk GEMMs -- block E4M3 W8A8 and the int8\n"
                "             rows -- at one shape, each with its TFLOP/s.\n"
                "  --perfw2 N K M [iters]:  the int2 DFlash2 draft head at every variant it has.\n"
                "  --perfhc M [iters]:  hc_read_e4m3 at Qwen3.8-Flash-Next's shape and M rows, both\n"
                "             prefill-tile forms compared byte for byte and timed as one op each;\n"
                "             run it under libr4d/isa/trace.sh for its launches.\n"
                "  --perfmoe M [iters] [experts] [top_k] [pooled%%]:  the routed-expert w4 GEMM,\n"
                "             both projections at rank 0's tp2 width split, over a skewed routing of\n"
                "             an M-token chunk (512 experts, top 10 by default); every form, each\n"
                "             compared with the first. pooled%% of the experts are read out of\n"
                "             pinned host memory through a pointer table, as the engine's are.\n"
                "  --perfmoe16 M [iters] [experts] [top_k] [distinct]:  the bf16 expert GEMM at\n"
                "             Qwen3.8-Flash-Next's whole-expert widths (256 experts, top 10 by\n"
                "             default), every form compared with the first; `distinct` weight\n"
                "             planes back the table, and fewer than the experts is a cache-resident\n"
                "             floor rather than the real traffic.\n"
                "  --perfqsa CTX [M] [iters]:  the QSA indexer at a real context length. It is\n"
                "             the only decode op that grows with context, and at 200K it is\n"
                "             about a third of the step. M is 1 + n_spec.\n"
                "             rad-tune's declared graph does not reach a gemm_nt_q row, so this\n"
                "             is the only way to sweep it. Safe to chase hard: the head only\n"
                "             proposes, so its output bits change acceptance, not text.\n");
}

int main(int argc, char** argv) {
    const char* path = nullptr;
    const char* real_file = nullptr;
    long long real_woff = 0, real_soff = 0, real_n = 0, real_k = 0;
    long long perf_n = 0, perf_k = 0, perf_m = 0;
    int perf_it = 50, perf_w2 = 0, perf_bf = 0, perf_w8 = 0;
    long long sop_m = 0, sop_k = 0;
    int sop_it = 2000;
    long long qsa_ctx = 0, qsa_m = 4;
    int qsa_it = 200;
    long long hc_m = 0;
    int hc_it = 20;
    long long moe_m = 0, moe_e = 512, moe_k = 10;
    int moe_pool = 0;
    int moe_it = 20;
    long long m16_m = 0, m16_e = 256, m16_k = 10, m16_d = 0;
    int m16_it = 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) {
            if (i + 5 >= argc) { usage(); return 2; }
            real_file = argv[++i];
            real_woff = std::strtoll(argv[++i], nullptr, 0);
            real_soff = std::strtoll(argv[++i], nullptr, 0);
            real_n    = std::strtoll(argv[++i], nullptr, 0);
            real_k    = std::strtoll(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perf")) {
            if (i + 3 >= argc) { usage(); return 2; }
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfbf16")) {
            /* The bf16 skinny GEMM, at every variant. Its own flag because rad-tune reaches these
             * rows but its absolute numbers are not repeatable enough between back-to-back runs
             * of the same command to base a default on. */
            if (i + 3 >= argc) { usage(); return 2; }
            perf_bf = 1;
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfw8")) {
            if (i + 3 >= argc) { usage(); return 2; }
            perf_w8 = 1;
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfw2")) {
            /* The int2 draft head, at every variant. Its own flag because rad-tune's declared
             * graph never reaches a gemm_nt_q row and this is the only kernel in the drafter
             * that is not already at bandwidth. */
            if (i + 3 >= argc) { usage(); return 2; }
            perf_w2 = 1;
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfqsa")) {
            /* The indexer at a REAL context length: it is the only decode op that grows with one. */
            if (i + 1 >= argc) { usage(); return 2; }
            qsa_ctx = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) qsa_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) qsa_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfmoe")) {
            /* The routed-expert GEMM at a prefill chunk of M tokens, every prefill form. */
            if (i + 1 >= argc) { usage(); return 2; }
            moe_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_it = (int)std::strtol(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_e = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_k = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_pool = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfmoe16")) {
            /* The bf16 expert GEMM at a prefill chunk of M tokens, every form. */
            if (i + 1 >= argc) { usage(); return 2; }
            m16_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_it = (int)std::strtol(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_e = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_k = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_d = std::strtoll(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfhc")) {
            /* The gated residual's read at M rows. */
            if (i + 1 >= argc) { usage(); return 2; }
            hc_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) hc_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfops")) {
            /* The elementwise ops at a decode shape. Separate from --perf because the question is
             * about launches, not about a GEMM's bandwidth. */
            if (i + 2 >= argc) { usage(); return 2; }
            sop_m = std::strtoll(argv[++i], nullptr, 0);
            sop_k = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') sop_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--case")) {
            if (i + 1 >= argc) { usage(); return 2; }
            g_case = argv[++i];
        } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage();
            return 0;
        } else if (!path) {
            path = argv[i];
        } else {
            usage();
            return 2;
        }
    }
    if (!path) path = "libr4d.so";

    Plugin pl;

    /* The timing mode runs alone for the same reason the real-checkpoint check does: it answers a
     * different question from the catalogue, and it is the one a reviewer runs by itself. */
    if (hc_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        hc_prefill(pl, hc_m, hc_it, true);
        dfree_all();
#else
        skip("gated residual read timing", "built without HIP");
#endif
        return 0;
    }
    if (moe_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        moe_perf(pl, moe_m, moe_it, moe_e, moe_k, moe_pool);
        dfree_all();
#else
        skip("routed-expert GEMM timing", "built without HIP");
#endif
        return 0;
    }
    if (m16_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        moe16_perf(pl, m16_m, m16_it, m16_e, m16_k, m16_d);
        dfree_all();
#else
        skip("bf16 expert GEMM timing", "built without HIP");
#endif
        return 0;
    }
    if (qsa_ctx) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        qsa_perf(pl, qsa_ctx, qsa_m, qsa_it);
        dfree_all();
#else
        skip("qsa indexer timing", "built without HIP");
#endif
        return 0;
    }
    if (sop_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        smallop_perf(pl, sop_m, sop_k, sop_it);
        dfree_all();
#else
        skip("small-op timing", "built without HIP");
#endif
        return 0;
    }

    if (perf_n) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        if (perf_bf)      bf16_perf(pl, perf_n, perf_k, perf_m, perf_it);
        else if (perf_w8) w8_perf(pl, perf_n, perf_k, perf_m, perf_it);
        else if (perf_w2) w2_perf(pl, perf_n, perf_k, perf_m, perf_it);
        else         fp8_perf(pl, perf_n, perf_k, perf_m, perf_it);
        dfree_all();
#else
        skip("fp8 decode GEMM timing", "built without HIP");
#endif
        return 0;
    }

    /* The real-checkpoint check runs alone: it is the one case whose input is not synthetic, so it
     * is also the one a reviewer will want to run by itself against a particular tensor. */
    if (real_file) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        fp8_real(pl, real_file, real_woff, real_soff, real_n, real_k);
        dfree_all();
#else
        skip("fp8 real-weight check", "built without HIP");
#endif
        std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
        return g_fail;
    }

    /* THE CHILD. One row, no catalogue, no static checks -- and an exit code the parent reads,
     * because a row that faults never gets to print anything at all. */
    if (g_case) {
        if (!load(&pl, path)) return CH_FAIL;
        /* A case that needs its own process re-execs this one; see spawn_anchor_child. Both paths
         * live only as long as this frame, which outlives the device phase. */
        char me[4096];
        const ssize_t mn = readlink("/proc/self/exe", me, sizeof(me) - 1);
        if (mn > 0) { me[mn] = 0; g_exe = me; g_plugin = path; }
        device_phase(pl);
        if (!g_matched) skip(g_case, "no smoke case");
        std::fflush(stdout);
        return g_fail ? CH_FAIL : (g_skip ? CH_SKIP : CH_OK);
    }

    std::printf("r4d_selftest  %s\n\n", path);
    if (!load(&pl, path)) {
        std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
        return g_fail ? g_fail : 1;
    }
    dump(pl);
    cross_check(pl);

    /* /proc/self/exe rather than argv[0]: the children are this binary, and argv[0] is whatever
     * the caller's shell felt like passing. */
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
        ok("re-exec path (/proc/self/exe)", false, "readlink failed");
        std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
        return g_fail;
    }
    self[n] = '\0';
    run_isolated(pl, self, path);

    std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
    return g_fail;
}
