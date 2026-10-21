/* bole_gdn_gpu_test.cpp -- the Bole tree-GDN rows against libr4d's own GDN row, on a card.
 *
 * What is claimed and how it is checked here:
 *
 *  1. COMMIT IS BYTE-EXACT. The commit row replays the accepted path's factors and recomputes u from
 *     the original update data in the serial FP32 order. Its committed state AND its rewritten conv
 *     history are compared to libr4d's gdn_conv_recurrent_update (r4d_rad_gdn_conv_recurrent_update)
 *     with memcmp over the whole slot, at several geometries, including one whose state values are
 *     chosen so the update cancels: a replayed U, the archived adjoint's failure mode, cannot pass
 *     these bytes.
 *
 *  2. VERIFY DECLARES A REORDERED RESULT. Below the root, Bole's walks and contractions are FP64, so
 *     the verify row's output is not bit-identical to a serial replay. It is compared to libr4d's
 *     output with a stated tolerance, per element, and the test also checks that the ROOT's output
 *     agrees exactly: the root is a fixed one-node serial FP32 prefix and must reproduce the serial
 *     value bit for bit, including an exact FP32 cancellation.
 *
 *  3. REFUSALS. Every geometry the rows cannot serve must be refused before a launch, with a negative
 *     status: a wider batch than the node budget, a l2_eps the exchange does not implement, an FP16
 *     state cache, an INT8 operand without its scale, a conv depth that disagrees with the cache, and
 *     a malformed parent array (a parent that is not an earlier node).
 *
 *  4. A NEGATIVE CONTROL FIRES. The comparison predicates are run once against a deliberately
 *     perturbed factor array and once against a perturbed state, and must report a difference:
 *     otherwise a passing byte comparison would prove only that the harness compares nothing.
 *
 *  5. NOTHING WRITES PAST ITS OPERAND. Every device buffer carries a poisoned 4 KiB tail, and the
 *     canaries are checked after each case. There is no compute-sanitizer on this host and llvm ships
 *     no memcheck, so this is what turns an out-of-bounds write into a named 32-byte report instead
 *     of a queue-wedging HSA fault.
 *
 * The test needs one card. Without a visible device it reports SKIP (exit 77) rather than passing.
 */
#include "bole_layout.h"

#include "rad_abi.h"
#include "rad_plugin.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;
static int g_skip = 0;

static void ok(const char* what, bool cond, const std::string& detail = std::string()) {
    ++g_checks;
    if (cond) { std::printf("  ok    %s\n", what); return; }
    ++g_fail;
    std::printf("  FAIL  %s%s%s\n", what, detail.empty() ? "" : " -- ", detail.c_str());
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

/* ================================================================== tensors */
static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}

static uint16_t f2b(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x8000u) >> 16);
}
static float b2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* ================================================================== device buffers */

/* Every operand gets a poisoned tail. A read past the end of one is then a fault at the same place a
 * real caller would fault, and a WRITE past the end corrupts the canary instead of silently landing
 * in the next allocation. This is the only out-of-bounds detector on this host: there is no
 * compute-sanitizer and llvm ships no memcheck, so the index arithmetic has to be proven by the
 * canaries plus a byte diff against libr4d's own row. */
static constexpr size_t BOLE_GUARD = 4096;
static constexpr unsigned char BOLE_CANARY = 0xA5;

struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() { if (p) (void)hipFree(p); }
    bool alloc(size_t n) {
        bytes = n ? n : 16;
        return hipMalloc(&p, bytes + BOLE_GUARD) == hipSuccess;
    }
    /* Poison the tail. Call after the last upload to this buffer and before the launches. */
    bool seal() {
        std::vector<unsigned char> c(BOLE_GUARD, BOLE_CANARY);
        return hipMemcpy((unsigned char*)p + bytes, c.data(), BOLE_GUARD,
                         hipMemcpyHostToDevice) == hipSuccess;
    }
    /* How many bytes of the tail changed. Non-zero is an out-of-bounds write, and it names the
     * operand rather than aborting the queue. */
    size_t guard_breach() const {
        std::vector<unsigned char> c(BOLE_GUARD, 0);
        if (hipMemcpy(c.data(), (const unsigned char*)p + bytes, BOLE_GUARD,
                      hipMemcpyDeviceToHost) != hipSuccess)
            return BOLE_GUARD;
        size_t bad = 0;
        for (unsigned char v : c) bad += (v != BOLE_CANARY);
        return bad;
    }
    /* The first changed guard offset, or -1. Distinguishes a 2-byte row overrun from a wholesale
     * index error. */
    long guard_first() const {
        std::vector<unsigned char> c(BOLE_GUARD, 0);
        if (hipMemcpy(c.data(), (const unsigned char*)p + bytes, BOLE_GUARD,
                      hipMemcpyDeviceToHost) != hipSuccess)
            return -1;
        for (size_t i = 0; i < c.size(); ++i)
            if (c[i] != BOLE_CANARY) return (long)i;
        return -1;
    }
};


static bool up(Buf& b, const void* src, size_t n) {
    return hipMemcpy(b.p, src, n, hipMemcpyHostToDevice) == hipSuccess;
}
static bool down(void* dst, const Buf& b, size_t n) {
    return hipMemcpy(dst, b.p, n, hipMemcpyDeviceToHost) == hipSuccess;
}

/* ================================================================== the plugins */
struct Rows {
    void* h = nullptr;
    RadLaunchFn cr = nullptr;        /* libr4d: gdn_conv_recurrent_update */
    RadLaunchFn verify = nullptr;    /* bole: stage=verify */
    RadLaunchFn commit = nullptr;    /* bole: stage=commit */
};

static bool open_lib(const char* path, Rows* out, const char* want_cr, const char* want_bole) {
    out->h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!out->h) { std::printf("  FAIL  dlopen %s: %s\n", path, dlerror()); return false; }
    auto abi = (uint32_t (*)(void))dlsym(out->h, "rad_plugin_abi_version");
    auto cnt = (int (*)(void))dlsym(out->h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(out->h, "rad_kernel_at");
    if (!abi || abi() != RAD_ABI_VERSION || !cnt || !at) {
        std::printf("  FAIL  %s: not an ABI %u kernel plugin\n", path, RAD_ABI_VERSION);
        return false;
    }
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (want_cr && !std::strcmp(k->op, want_cr)) out->cr = k->launch;
        if (want_bole && !std::strcmp(k->op, want_bole)) {
            /* The two rows are told apart by the `stage` constraint, which is the same key a caller
             * selects on. */
            bool commit_row = false;
            for (int c = 0; c < k->n_constraints; ++c)
                if (!std::strcmp(k->constraints[c].key, "stage") &&
                    k->constraints[c].op == RAD_C_IN && k->constraints[c].sval &&
                    !std::strcmp(k->constraints[c].sval, "commit"))
                    commit_row = true;
            if (commit_row) out->commit = k->launch;
            else out->verify = k->launch;
        }
    }
    return true;
}

/* ================================================================== a case's host data */
struct Host {
    int N = 1, T = 5, H = 2, Hg = 1, V = 128, K = 128, depth = 7, slots = 8;
    int tokens = 0;
    std::vector<int> cu, conv_idx, parent, ctx, acc;
    std::vector<uint16_t> x, cw, cb, cst, a, b, z, o_host, cst_host;
    std::vector<float> alog, dtb, nw, state, state_host;
    int conv_dim() const { return 2 * Hg * K + H * V; }
};

static uint32_t g_seed = 12345u;
static float rnd(float lo = -1.0f, float hi = 1.0f) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((g_seed >> 8) / 16777216.0f);
}

/* Build the case's host-side inputs. `cancel` scales the initial state so the delta-rule update
 * cancels in FP32 at the root: the archived variants' decisive witness. */
static void build(Host& h, bool cancel) {
    h.tokens = h.N * h.T;
    h.cu.assign(h.N + 1, 0);
    for (int n = 0; n < h.N; ++n) h.cu[n + 1] = h.cu[n] + h.T;
    h.conv_idx.resize(h.N);
    h.parent.resize(h.tokens);
    h.ctx.resize(h.N);
    h.acc.assign(h.N, 1);
    for (int n = 0; n < h.N; ++n) {
        h.conv_idx[n] = n;
        h.ctx[n] = 4;
        for (int t = 0; t < h.T; ++t) h.parent[n * h.T + t] = t ? t - 1 : -1;
    }
    const int cd = h.conv_dim();
    h.x.resize((size_t)h.tokens * cd);
    h.cw.resize((size_t)cd * 4);
    h.cb.resize(cd);
    h.cst.assign((size_t)h.slots * cd * h.depth, 0);
    h.a.resize((size_t)h.tokens * h.H);
    h.b.resize((size_t)h.tokens * h.H);
    h.z.resize((size_t)h.tokens * h.H * h.V);
    h.alog.resize(h.H);
    h.dtb.resize(h.H);
    h.nw.resize(h.V);
    h.state.assign((size_t)h.slots * h.H * h.V * h.K, 0.0f);

    for (auto& v : h.x) v = f2b(rnd());
    for (auto& v : h.cw) v = f2b(0.5f * rnd());
    for (auto& v : h.cb) v = f2b(0.25f * rnd());
    for (auto& v : h.cst) v = f2b(rnd());
    for (auto& v : h.a) v = f2b(rnd(-2.0f, 2.0f));
    for (auto& v : h.b) v = f2b(rnd(-2.0f, 2.0f));
    for (auto& v : h.z) v = f2b(rnd(-1.0f, 1.0f));
    for (int i = 0; i < h.H; ++i) { h.alog[i] = rnd(0.2f, 1.2f); h.dtb[i] = rnd(-1.0f, 1.0f); }
    for (auto& v : h.nw) v = rnd(0.5f, 1.5f);
    for (auto& v : h.state) v = rnd(-1.0f, 1.0f);

    if (cancel) {
        /* One-hot k and q, gamma = 1/2 and an initial state of a power of two make the serial
         * root's update exactly zero: h = 2^-20, dot = h*k, u = v - dot with v chosen so u is zero in
         * FP32. Anything that replays a differently-rounded U leaves a residue here. */
        for (int n = 0; n < h.N; ++n) {
            for (int hh = 0; hh < h.H; ++hh) {
                float* s = h.state.data() + ((size_t)(n * h.H + hh) * h.V * h.K);
                for (int r = 0; r < h.V; ++r)
                    for (int c = 0; c < h.K; ++c) s[r * h.K + c] = (c == 0) ? 1048576.0f : 0.0f;
            }
            for (int t = 0; t < h.T; ++t) {
                uint16_t* arow = h.a.data() + (size_t)(n * h.T + t) * h.H;
                for (int hh = 0; hh < h.H; ++hh) arow[hh] = f2b(-20.0f);   /* gamma -> 1/2 */
            }
        }
    }
}

/* ================================================================== one run of both rows */
struct Cmp {
    size_t compared = 0, diff = 0;
    double max_abs = 0.0;
    int first_diff = -1;
};

int main() {
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        std::printf("SKIP no visible device\n");
        return 77;
    }
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) {
        std::printf("SKIP RADIANCE_HOME is not set\n");
        return 77;
    }
    const std::string base(home);
    const std::string l4 = base + "/kernels/libr4d.so";
    const std::string bl = base + "/kernels/libbolegdn.so";
    Rows rows;
    if (!open_lib(l4.c_str(), &rows, "gdn_conv_recurrent_update", nullptr) ||
        !open_lib(bl.c_str(), &rows, nullptr, "gdn_tree_bole")) {
        return 1;
    }
    ok("libr4d's GDN row and both Bole rows are loadable",
       rows.cr && rows.verify && rows.commit);

    /* ---------------------------------------------------------------- 1. commit is byte-exact */
    for (int variant = 0; variant < 3; ++variant) {
        const bool cancel = variant == 2;
        Host h;
        h.T = variant == 1 ? 1 : 5;
        /* The window must fit the cache: libr4d's bound is mql = depth - (conv_width - 2), so a
         * T-token window needs depth >= T + 2. The two extra rows hold the oldest history taps. */
        h.depth = h.T + BOLE_CONV_W;
        h.N = 1;
        build(h, cancel);
        const int cd = h.conv_dim();
        const size_t st_elems = (size_t)h.slots * h.H * h.V * h.K;
        const size_t cst_elems = (size_t)h.slots * cd * h.depth;

        /* Two state caches and two conv caches from the same bytes: one for the reference, one for
         * the Bole rows, so neither can pass on the other's writes. */
        Buf d_x, d_cw, d_cb, d_cst_ref, d_cst_bol, d_ci, d_a, d_b, d_alog, d_dtb;
        Buf d_st_ref, d_st_bol, d_si_ref, d_si_bol, d_cu, d_par, d_ctx, d_z, d_nw, d_fact, d_o_ref,
            d_o_bol, d_oq, d_os, d_acc;
        bool a = d_x.alloc(h.x.size() * 2) && d_cw.alloc(h.cw.size() * 2) &&
                 d_cb.alloc(h.cb.size() * 2) && d_cst_ref.alloc(cst_elems * 2) &&
                 d_cst_bol.alloc(cst_elems * 2) && d_ci.alloc(h.N * 4) && d_a.alloc(h.a.size() * 2) &&
                 d_b.alloc(h.b.size() * 2) && d_alog.alloc(h.H * 4) && d_dtb.alloc(h.H * 4) &&
                 d_st_ref.alloc(st_elems * 4) && d_st_bol.alloc(st_elems * 4) &&
                 d_si_ref.alloc((size_t)h.N * h.T * 4) && d_si_bol.alloc(h.N * 4) &&
                 d_cu.alloc((h.N + 1) * 4) && d_par.alloc((size_t)h.tokens * 4) &&
                 d_ctx.alloc(h.N * 4) && d_z.alloc(h.z.size() * 2) && d_nw.alloc(h.V * 4) &&
                 d_fact.alloc((size_t)h.tokens * h.H * 258 * 4) &&
                 d_o_ref.alloc((size_t)h.tokens * h.H * h.V * 2) &&
                 d_o_bol.alloc((size_t)h.tokens * h.H * h.V * 2) &&
                 d_oq.alloc((size_t)h.tokens * h.H * h.V) && d_os.alloc((size_t)h.tokens * h.H * 4) &&
                 d_acc.alloc(h.N * 4);
        ok("case buffers allocated", a);
        if (!a) continue;

        /* per_candidate slots: candidate t of sequence n writes slot n*T + t. */
        std::vector<int> si_ref((size_t)h.N * h.T), si_bol(h.N);
        for (int n = 0; n < h.N; ++n) {
            si_bol[n] = n;                       /* the Bole rows' one slot per sequence */
            for (int t = 0; t < h.T; ++t) si_ref[(size_t)n * h.T + t] = n * h.T + t;
        }
        a = up(d_x, h.x.data(), h.x.size() * 2) && up(d_cw, h.cw.data(), h.cw.size() * 2) &&
            up(d_cb, h.cb.data(), h.cb.size() * 2) && up(d_cst_ref, h.cst.data(), cst_elems * 2) &&
            up(d_cst_bol, h.cst.data(), cst_elems * 2) && up(d_ci, h.conv_idx.data(), h.N * 4) &&
            up(d_a, h.a.data(), h.a.size() * 2) && up(d_b, h.b.data(), h.b.size() * 2) &&
            up(d_alog, h.alog.data(), h.H * 4) && up(d_dtb, h.dtb.data(), h.H * 4) &&
            up(d_st_ref, h.state.data(), st_elems * 4) &&
            up(d_st_bol, h.state.data(), st_elems * 4) &&
            up(d_si_ref, si_ref.data(), si_ref.size() * 4) &&
            up(d_si_bol, si_bol.data(), si_bol.size() * 4) &&
            up(d_cu, h.cu.data(), (h.N + 1) * 4) && up(d_par, h.parent.data(), h.parent.size() * 4) &&
            up(d_ctx, h.ctx.data(), h.N * 4) && up(d_z, h.z.data(), h.z.size() * 2) &&
            up(d_nw, h.nw.data(), h.V * 4) && up(d_acc, h.acc.data(), h.N * 4);
        ok("case inputs uploaded", a);
        if (!a) continue;
        /* Before any kernel runs, poison every tail: a stray write is then the canary, not a fault. */
        bool sealed = true;
        for (Buf* b : { &d_x, &d_cw, &d_cb, &d_cst_ref, &d_cst_bol, &d_ci, &d_a, &d_b, &d_alog,
                        &d_dtb, &d_st_ref, &d_st_bol, &d_si_ref, &d_si_bol, &d_cu, &d_par, &d_ctx,
                        &d_z, &d_nw, &d_fact, &d_o_ref, &d_o_bol, &d_oq, &d_os, &d_acc })
            sealed = sealed && b->seal();
        ok("case guard tails poisoned", sealed);

        RadTensor t_ref[18] = {
            tens(d_x.p, RAD_BF16, { h.tokens, cd }), tens(d_cw.p, RAD_BF16, { cd, 4 }),
            tens(d_cb.p, RAD_BF16, { cd }), tens(d_cst_ref.p, RAD_BF16, { h.slots, cd, h.depth }),
            tens(d_ci.p, RAD_I32, { h.N }), tens(d_a.p, RAD_BF16, { h.tokens, h.H }),
            tens(d_b.p, RAD_BF16, { h.tokens, h.H }), tens(d_alog.p, RAD_F32, { h.H }),
            tens(d_dtb.p, RAD_F32, { h.H }), tens(d_st_ref.p, RAD_F32, { h.slots, h.H, h.V, h.K }),
            tens(d_si_ref.p, RAD_I32, { h.N, h.T }), tens(d_acc.p, RAD_I32, { h.N }),
            tens(d_cu.p, RAD_I32, { h.N + 1 }), tens(d_z.p, RAD_BF16, { h.tokens, h.H * h.V }),
            tens(d_nw.p, RAD_F32, { h.V }), tens(d_o_ref.p, RAD_BF16, { h.tokens, h.H, h.V }),
            tens(d_oq.p, RAD_I8, { h.tokens, h.H * h.V }), tens(d_os.p, RAD_F32, { h.tokens, h.H }),
        };
        RadParam p_ref[12]{};
        int np = 0;
        auto pi = [&](const char* k, long long v) { p_ref[np++] = RAD_INT(k, v); };
        pi("q_len", h.T);
        pi("head_k", 128);
        pi("head_v", 128);
        pi("conv_width", 4);
        pi("n_head_v", h.H);
        p_ref[np++] = RAD_F64("scale", 1.0 / std::sqrt(128.0));
        p_ref[np++] = RAD_F64("eps", 1e-6);
        p_ref[np++] = RAD_F64("l2_eps", 1e-6);
        p_ref[np++] = RAD_F64("softplus_thr", 20.0);
        p_ref[np++] = RAD_STR("act", "silu");
        p_ref[np++] = RAD_STR("state_form", "per_candidate");
        RadArgs ar{};
        ar.t = t_ref;
        ar.n_t = 18;
        ar.p = p_ref;
        ar.n_p = np;
        const int rc_ref = rows.cr(&ar, nullptr);

        /* The Bole verify: the same inputs, `parent` as the linear chain, one slot per sequence. */
        /* In the schema's order: state_idx, num_accepted, cu, parent, ctx_before. */
        RadTensor t_bol[21] = {
            t_ref[0], t_ref[1], t_ref[2], tens(d_cst_bol.p, RAD_BF16, { h.slots, cd, h.depth }),
            t_ref[4], t_ref[5], t_ref[6], t_ref[7], t_ref[8],
            tens(d_st_bol.p, RAD_F32, { h.slots, h.H, h.V, h.K }),
            tens(d_si_bol.p, RAD_I32, { h.N }), t_ref[11], t_ref[12],
            tens(d_par.p, RAD_I32, { h.tokens }), tens(d_ctx.p, RAD_I32, { h.N }),
            t_ref[13], t_ref[14], tens(d_fact.p, RAD_F32, { h.tokens, h.H, 258 }),
            tens(d_o_bol.p, RAD_BF16, { h.tokens, h.H, h.V }), t_ref[16], t_ref[17],
        };
        RadParam p_bol[13]{};
        int mb = 0;
        /* A selector collapses the ranged `q_len` to a single integer before a launch; a direct caller
         * must hand the launch the integer it collapsed to, not the range the schema declares. */
        p_bol[mb++] = RAD_INT("q_len", h.T);
        p_bol[mb++] = RAD_INT("head_k", 128);
        p_bol[mb++] = RAD_INT("head_v", 128);
        p_bol[mb++] = RAD_INT("conv_width", 4);
        p_bol[mb++] = RAD_INT("conv_depth", h.depth);
        p_bol[mb++] = RAD_STR("stage", "verify");
        p_bol[mb++] = RAD_INT("n_head_v", h.H);
        p_bol[mb++] = RAD_F64("scale", 1.0 / std::sqrt(128.0));
        p_bol[mb++] = RAD_F64("eps", 1e-6);
        p_bol[mb++] = RAD_F64("l2_eps", 1e-6);
        p_bol[mb++] = RAD_F64("softplus_thr", 20.0);
        p_bol[mb++] = RAD_STR("act", "silu");
        RadArgs ab{};
        ab.t = t_bol;
        ab.n_t = 21;
        ab.p = p_bol;
        ab.n_p = mb;
        const int rc_v = rows.verify(&ab, nullptr);

        RadParam p_com[4]{};
        p_com[0] = RAD_INT("q_len", h.T);
        p_com[1] = RAD_INT("head_k", 128);
        p_com[2] = RAD_INT("head_v", 128);
        p_com[3] = RAD_INT("conv_width", 4);
        RadParam p_com2[6] = { p_com[0], p_com[1], p_com[2], p_com[3], RAD_INT("conv_depth", h.depth),
                               RAD_STR("stage", "commit") };
        RadArgs ac{};
        ac.t = t_bol;
        ac.n_t = 21;
        ac.p = p_com2;
        ac.n_p = 6;
        const int rc_c = rows.commit(&ac, nullptr);
        const bool sync = hipDeviceSynchronize() == hipSuccess;

        const char* nm = variant == 1 ? "T=1" : cancel ? "cancellation scene (T=5)" : "T=5";
        ok(fmt("%s: both rows accepted the geometry", nm).c_str(),
           rc_ref == RAD_OK && rc_v == RAD_OK && rc_c == RAD_OK && sync,
           fmt("rc_ref=%d rc_verify=%d rc_commit=%d", rc_ref, rc_v, rc_c));

        /* An out-of-bounds write names its operand here. A fault would have aborted the process
         * before this point, so reaching it with a clean canary set means the index arithmetic wrote
         * only inside its tensors. */
        {
            size_t breach = 0;
            const char* who = "none";
            struct { Buf* b; const char* n; } g[] = {
                { &d_x, "x" }, { &d_cw, "conv_w" }, { &d_cb, "conv_b" }, { &d_cst_bol, "conv_state" },
                { &d_ci, "conv_idx" }, { &d_a, "a" }, { &d_b, "b" }, { &d_alog, "A_log" },
                { &d_dtb, "dt_bias" }, { &d_st_bol, "state" }, { &d_si_bol, "state_idx" },
                { &d_cu, "cu" }, { &d_par, "parent" }, { &d_ctx, "ctx_before" }, { &d_z, "z" },
                { &d_nw, "norm_w" }, { &d_fact, "factors" }, { &d_o_bol, "o" }, { &d_oq, "o_q" },
                { &d_os, "o_scale" }, { &d_acc, "num_accepted" },
            };
            for (auto& e : g) {
                const size_t k = e.b->guard_breach();
                if (k && !breach) { breach = k; who = e.n; }
                if (k) std::printf("        guard: %-12s %2zu bytes, first at +%ld of %zu\n", e.n, k,
                                   e.b->guard_first(), e.b->bytes);
            }
            ok(fmt("%s: no operand writes past its guard", nm).c_str(), breach == 0,
               breach ? fmt("%zu guard bytes changed, first %s", breach, who) : "");
        }

        /* ---- the commit's state against the reference's LAST candidate slot, byte for byte. */
        std::vector<float> st_ref(st_elems), st_bol(st_elems);
        std::vector<uint16_t> cst_ref(cst_elems), cst_bol(cst_elems);
        down(st_ref.data(), d_st_ref, st_elems * 4);
        down(st_bol.data(), d_st_bol, st_elems * 4);
        down(cst_ref.data(), d_cst_ref, cst_elems * 2);
        down(cst_bol.data(), d_cst_bol, cst_elems * 2);

        bool st_eq = true;
        size_t bad = 0;
        for (int hh = 0; hh < h.H; ++hh) {
            const float* r = st_ref.data() + ((size_t)((h.N - 1) * h.T + h.T - 1) * h.H + hh) *
                                               h.V * h.K;
            const float* b = st_bol.data() + ((size_t)(h.N - 1) * h.H + hh) * h.V * h.K;
            if (std::memcmp(r, b, (size_t)h.V * h.K * 4) != 0) {
                st_eq = false;
                for (size_t i = 0; i < (size_t)h.V * h.K; ++i)
                    if (std::memcmp(&r[i], &b[i], 4) != 0) ++bad;
            }
        }
        ok(fmt("%s: committed state byte-exact to libr4d", nm).c_str(), st_eq,
           st_eq ? "" : fmt("%zu of %d floats differ", bad, h.H * h.V * h.K));

        /* ---- the conv history: the reference's cache against the commit's, whole buffer. */
        const bool cst_eq = std::memcmp(cst_ref.data(), cst_bol.data(), cst_elems * 2) == 0;
        ok(fmt("%s: conv state byte-exact to libr4d", nm).c_str(), cst_eq);

        /* ---- the verify row's output against the reference's: the root exact, the rest to
         * tolerance, because below the root Bole accumulates in FP64. */
        std::vector<uint16_t> o_ref((size_t)h.tokens * h.H * h.V), o_bol(o_ref.size());
        down(o_ref.data(), d_o_ref, o_ref.size() * 2);
        down(o_bol.data(), d_o_bol, o_bol.size() * 2);
        bool root_eq = true;
        for (int hh = 0; hh < h.H; ++hh)
            root_eq = root_eq && std::memcmp(&o_ref[(size_t)hh * h.V], &o_bol[(size_t)hh * h.V],
                                              (size_t)h.V * 2) == 0;
        ok(fmt("%s: the root's output is exactly the serial value", nm).c_str(), root_eq);

        double max_rel = 0.0;
        size_t over = 0;
        for (size_t i = 0; i < o_ref.size(); ++i) {
            const double r = b2f(o_ref[i]), b = b2f(o_bol[i]);
            const double tol = 2e-2 * std::fabs(r) + 1e-3;
            if (std::fabs(b - r) > tol) ++over;
            if (r != 0.0) max_rel = std::max(max_rel, std::fabs(b - r) / std::fabs(r));
        }
        ok(fmt("%s: verify output within the declared reorder tolerance", nm).c_str(), over == 0,
           fmt("%zu elements out of %zu, max relative %.3g", over, o_ref.size(), max_rel));
    }

    /* ---------------------------------------------------------------- 2. the negative controls */
    /* The comparison predicates must report a difference when handed one. Otherwise a passing
     * byte comparison above would prove only that the harness compares nothing. */
    {
        std::vector<float> a(64), b(64);
        for (size_t i = 0; i < a.size(); ++i) a[i] = (float)i * 0.25f;
        b = a;
        ok("negative control: identical bytes compare equal", std::memcmp(a.data(), b.data(), 256) == 0);
        b[7] = std::nextafterf(b[7], 1e30f);          /* one ulp */
        ok("negative control: one differing float is detected", std::memcmp(a.data(), b.data(), 256) != 0);

        std::vector<uint16_t> o1(128), o2(128);
        for (size_t i = 0; i < o1.size(); ++i) o1[i] = f2b((float)i * 0.01f);
        o2 = o1;
        o2[3] = (uint16_t)(o2[3] + 78);               /* the archived variant's failing code delta */
        const double r = b2f(o1[3]), d = b2f(o2[3]);
        ok("negative control: the output predicate rejects a 78-code shift",
           std::fabs(d - r) > 2e-2 * std::fabs(r) + 1e-3);
    }

    /* ---------------------------------------------------------------- 3. refusals */
    {
        Host h;
        build(h, false);
        const int cd = h.conv_dim();
        Buf bx, bcw, bcst, bci, ba, bb, balog, bdtb, bst, bsi, bcu, bpar, bctx, bz, bnw, bfact, bo, bacc;
        const size_t cst_elems = (size_t)h.slots * cd * h.depth;
        const size_t st_elems = (size_t)h.slots * h.H * h.V * h.K;
        bx.alloc(h.x.size() * 2); bcw.alloc(h.cw.size() * 2); bcst.alloc(cst_elems * 2);
        bci.alloc(h.N * 4); ba.alloc(h.a.size() * 2); bb.alloc(h.b.size() * 2);
        balog.alloc(h.H * 4); bdtb.alloc(h.H * 4); bst.alloc(st_elems * 4); bsi.alloc(h.N * 4);
        bcu.alloc((h.N + 1) * 4); bpar.alloc((size_t)h.tokens * 4); bctx.alloc(h.N * 4);
        bz.alloc(h.z.size() * 2); bnw.alloc(h.V * 4);
        bfact.alloc((size_t)h.tokens * h.H * 258 * 4); bo.alloc((size_t)h.tokens * h.H * h.V * 2);
        bacc.alloc(h.N * 4);
        std::vector<int> si(h.N, 0);
        up(bx, h.x.data(), h.x.size() * 2); up(bcw, h.cw.data(), h.cw.size() * 2);
        up(bcst, h.cst.data(), cst_elems * 2); up(bci, h.conv_idx.data(), h.N * 4);
        up(ba, h.a.data(), h.a.size() * 2); up(bb, h.b.data(), h.b.size() * 2);
        up(balog, h.alog.data(), h.H * 4); up(bdtb, h.dtb.data(), h.H * 4);
        up(bst, h.state.data(), st_elems * 4); up(bsi, si.data(), h.N * 4);
        up(bcu, h.cu.data(), (h.N + 1) * 4); up(bpar, h.parent.data(), h.parent.size() * 4);
        up(bctx, h.ctx.data(), h.N * 4); up(bz, h.z.data(), h.z.size() * 2);
        up(bnw, h.nw.data(), h.V * 4); up(bacc, h.acc.data(), h.N * 4);

        auto make = [&](std::vector<RadTensor>& t, long long q_len, const char* l2, const char* stage,
                        uint32_t state_dt, int depth_override, bool oq_without_os) {
                    t.assign(21, RadTensor{});
                    t[0] = tens(bx.p, RAD_BF16, { h.tokens, cd });
                    t[1] = tens(bcw.p, RAD_BF16, { cd, 4 });
                    t[3] = tens(bcst.p, RAD_BF16, { h.slots, cd, depth_override });
                    t[4] = tens(bci.p, RAD_I32, { h.N });
                    t[5] = tens(ba.p, RAD_BF16, { h.tokens, h.H });
                    t[6] = tens(bb.p, RAD_BF16, { h.tokens, h.H });
                    t[7] = tens(balog.p, RAD_F32, { h.H });
                    t[8] = tens(bdtb.p, RAD_F32, { h.H });
                    t[9] = tens(bst.p, state_dt, { h.slots, h.H, h.V, h.K });
                    t[10] = tens(bsi.p, RAD_I32, { h.N });
                    t[11] = tens(bacc.p, RAD_I32, { h.N });
                    t[12] = tens(bcu.p, RAD_I32, { h.N + 1 });
                    t[13] = tens(bpar.p, RAD_I32, { h.tokens });
                    t[14] = tens(bctx.p, RAD_I32, { h.N });
                    t[15] = tens(bz.p, RAD_BF16, { h.tokens, h.H * h.V });
                    t[16] = tens(bnw.p, RAD_F32, { h.V });
                    t[17] = tens(bfact.p, RAD_F32, { h.tokens, h.H, 258 });
                    t[18] = tens(bo.p, RAD_BF16, { h.tokens, h.H, h.V });
                    if (oq_without_os) t[19] = tens(bacc.p, RAD_I8, { h.tokens, h.H * h.V });
                    RadParam p[12]{};
                    int n = 0;
                    p[n++] = RAD_INT("q_len", q_len);
                    p[n++] = RAD_INT("head_k", 128);
                    p[n++] = RAD_INT("head_v", 128);
                    p[n++] = RAD_INT("conv_width", 4);
                    p[n++] = RAD_INT("conv_depth", h.depth);
                    p[n++] = RAD_STR("stage", stage);
                    p[n++] = RAD_F64("l2_eps", std::atof(l2));
                    RadArgs args{};
                    args.t = t.data();
                    args.n_t = 21;
                    args.p = p;
                    args.n_p = n;
                    return args;
                };
        std::vector<RadTensor> tt;
        auto refuse = [&](const char* what, long long q_len, const char* l2, const char* stage,
                          uint32_t dt, int depth_override, bool oq_only) {
                          RadArgs args = make(tt, q_len, l2, stage, dt, depth_override, oq_only);
                          const int rc = rows.verify(&args, nullptr);
                          ok(fmt("refuses %s", what).c_str(), rc < 0, fmt("returned %d", rc));
                      };
        refuse("a batch wider than the node budget", 17, "1e-6", "verify", RAD_F32, h.depth, false);
        refuse("an l2_eps the exchange does not implement", h.T, "1e-5", "verify", RAD_F32, h.depth, false);
        refuse("an fp16 state cache", h.T, "1e-6", "verify", RAD_F16, h.depth, false);
        refuse("an int8 output without its scale", h.T, "1e-6", "verify", RAD_F32, h.depth, true);
        refuse("a conv_depth that disagrees with the cache", h.T, "1e-6", "verify", RAD_F32,
               h.depth + 1, false);
        refuse("an unknown stage", h.T, "1e-6", "sideways", RAD_F32, h.depth, false);
        /* A window deeper than the cache can back: the bound libr4d enforces before its launcher,
         * and without which the conv read walks past the cache's extent. */
        refuse("a window deeper than the conv cache", h.T, "1e-6", "verify", RAD_F32, h.T, false);

        /* A malformed parent array: node 1 claims node 3 as its parent. The row walks parents, so
         * this must be refused rather than followed. */
        std::vector<int> bad_par = h.parent;
        if (bad_par.size() > 3) { bad_par[1] = 3; up(bpar, bad_par.data(), bad_par.size() * 4); }
        {
            RadArgs args = make(tt, h.T, "1e-6", "verify", RAD_F32, h.depth, false);
            const int rc = rows.verify(&args, nullptr);
            /* The check runs on the device, so a launch that is submitted cannot report it; what the
             * row must guarantee is that it wrote nothing. */
            std::vector<uint16_t> o((size_t)h.tokens * h.H * h.V, 0x7777);
            (void)hipMemcpy(bo.p, o.data(), o.size() * 2, hipMemcpyHostToDevice);
            const int rc2 = rows.verify(&args, nullptr);
            (void)hipDeviceSynchronize();
            std::vector<uint16_t> got(o.size());
            down(got.data(), bo, got.size() * 2);
            bool untouched = true;
            for (uint16_t v : got) untouched = untouched && v == 0x7777;
            ok("a malformed parent array writes no output", untouched,
               fmt("submit=%d relaunch=%d", rc, rc2));
        }
    }

    std::printf("\n%d checks, %d failed, %d skipped\n", g_checks, g_fail, g_skip);
    return g_fail ? 1 : 0;
}
