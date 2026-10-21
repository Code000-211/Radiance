/* ref_test.cpp -- libref, op by op, on shapes small enough to check by hand.
 *
 * THIS TEST IS THE FOUNDATION OF THE WHOLE CORRECTNESS STORY. libref is the oracle rad-kbench
 * runs every resolved kernel against (spec §17), so if ref is wrong, rad-kbench certifies wrong
 * kernels as correct and the engine ends with fluent wrong output and no way to bisect it. Every
 * expected value below is therefore computed on paper and written as a literal -- not produced by
 * a second call into the same code, which would only prove the plugin agrees with itself.
 *
 * Where an expectation could not be a literal (the seeded sampler's draw), the test checks the
 * PROPERTY instead: determinism, range, and that the generator is not constant.
 *
 * Two of the tests are cross-checks rather than literals, and they are the most valuable ones
 * here: the gated-delta-net chunked scan is run against the recurrent update on the same input,
 * because the two compute the same recurrence by completely different routes, and the paged
 * attention is run at q_len 2 against a hand-computed softmax.
 *
 * It reaches the plugin the way the engine does -- through RadKernelInfo::launch, with a RadArgs
 * built by hand -- so the operand order and the parameter names in libref/ref_registry.cpp
 * are under test too, not just the arithmetic.
 */
#include "rad_test.h"
#include "rad_abi.h"
#include "../libref/ref_common.h"
#include "rad_sample.h"

#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>
#include <dlfcn.h>

using namespace ref;

/* ================================================================== reaching the plugin
 * Weak references first: linked directly against the plugin sources, the symbols are right
 * there. Built by CMake it links rad_core only, the weak references stay null, and the plugin is
 * dlopen'd out of $RADIANCE_HOME exactly as the engine loads it. Both paths exercise the same
 * table. */
extern "C" {
__attribute__((weak)) int                  rad_kernel_count(void);
__attribute__((weak)) const RadKernelInfo* rad_kernel_at(int);
__attribute__((weak)) int                  rad_kernel_schema_count(void);
__attribute__((weak)) const RadOpSchema*   rad_kernel_schema_at(int);
__attribute__((weak)) uint32_t             rad_plugin_abi_version(void);
__attribute__((weak)) const RadPluginInfo* rad_plugin_info(void);
}

namespace {

struct Api {
    int                  (*kernel_count)(void)        = nullptr;
    const RadKernelInfo* (*kernel_at)(int)            = nullptr;
    int                  (*schema_count)(void)        = nullptr;
    const RadOpSchema*   (*schema_at)(int)            = nullptr;
    uint32_t             (*abi_version)(void)         = nullptr;
    const RadPluginInfo* (*info)(void)                = nullptr;
};

Api load_api() {
    Api a;
    if (&rad_kernel_count) {
        a.kernel_count = rad_kernel_count;
        a.kernel_at    = rad_kernel_at;
        a.schema_count = rad_kernel_schema_count;
        a.schema_at    = rad_kernel_schema_at;
        a.abi_version  = rad_plugin_abi_version;
        a.info         = rad_plugin_info;
        return a;
    }
    std::vector<std::string> paths;
    if (const char* home = std::getenv("RADIANCE_HOME"))
        paths.push_back(std::string(home) + "/kernels/libref.so");
    paths.push_back("radiance_home/kernels/libref.so");
    paths.push_back("build/radiance_home/kernels/libref.so");
    paths.push_back("./libref.so");
    for (auto& p : paths) {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        a.kernel_count = (int (*)(void))dlsym(h, "rad_kernel_count");
        a.kernel_at    = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        a.schema_count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
        a.schema_at    = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
        a.abi_version  = (uint32_t (*)(void))dlsym(h, "rad_plugin_abi_version");
        a.info         = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
        if (a.kernel_count) return a;
    }
    fprintf(stderr, "ref_test: libref not found. Set RADIANCE_HOME or link the plugin.\n");
    std::exit(2);
}

const Api& api() { static Api a = load_api(); return a; }

const RadKernelInfo* kern(const char* op) {
    for (int i = 0; i < api().kernel_count(); ++i) {
        const RadKernelInfo* k = api().kernel_at(i);
        if (k && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

const RadOpSchema* schema(const char* op) {
    for (int i = 0; i < api().schema_count(); ++i) {
        const RadOpSchema* s = api().schema_at(i);
        if (s && std::strcmp(s->op, op) == 0) return s;
    }
    return nullptr;
}

/* ================================================================== tiny tensors */
int64_t dt_bytes(uint32_t dt, int64_t n) {
    switch (dt) {
        case RAD_F32: case RAD_I32: case RAD_U32:            return n * 4;
        case RAD_I64:                                        return n * 8;
        case RAD_F16: case RAD_BF16: case RAD_I16:           return n * 2;
        case RAD_I8: case RAD_U8: case RAD_BOOL:
        case RAD_F8E4M3: case RAD_F8E5M2:                    return n;
        case RAD_I4: case RAD_FP4E2M1:                         return (n + 1) / 2;
        case RAD_I2:                                         return (n + 3) / 4;
        default:                                             return n * 4;
    }
}

struct T {
    std::vector<uint8_t> mem;
    RadTensor t{};

    T(uint32_t dt, std::vector<int64_t> shape) {
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t n = 1;
        for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; n *= shape[i]; }
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
        mem.assign((size_t)dt_bytes(dt, n) + 8, 0);
        t.data = mem.data();
    }
    T& f(std::vector<float> v) {
        for (size_t i = 0; i < v.size(); ++i) st_dt(t.dtype, t.data, (int64_t)i, v[i]);
        return *this;
    }
    T& raw(std::vector<uint8_t> v) {
        for (size_t i = 0; i < v.size(); ++i) mem[i] = v[i];
        return *this;
    }
    T& n(std::vector<int64_t> v) {
        for (size_t i = 0; i < v.size(); ++i) st_int(t.dtype, t.data, (int64_t)i, v[i]);
        return *this;
    }
    float    at(int64_t i) const  { return ld_dt(t.dtype, t.data, i); }
    int64_t  iat(int64_t i) const { return ld_int(t.dtype, t.data, i); }
};

/* ================================================================== the sampler's params row */
/* Every sampler op takes its scalars as an OPERAND of RadSampleParams rows, not as parameters --
 * a parameter is geometry and freezes at declare, so top_p as a parameter would make two requests
 * with different top_p two resolutions of one op (abi/rad_sample.h). This builds that
 * operand: identity defaults, then the case sets the one field it is about. */
struct SP {
    std::vector<RadSampleParams> rows;
    T                            t;

    explicit SP(int64_t n)
        : rows((size_t)n), t(RAD_U32, {n, (int64_t)(sizeof(RadSampleParams) / 4)}) {
        for (auto& r : rows) {
            std::memset(&r, 0, sizeof r);
            r.temp = 1.0f;
            r.top_p = 1.0f;
            r.min_p = 0.0f;
            r.typical_p = 1.0f;
            r.rep_penalty = 1.0f;
            r.mask_row = -1;
            /* The whole history. 0 would turn the history stages off, llama.cpp's meaning. */
            r.penalty_last_n = -1;
            r.dry_penalty_last_n = -1;
        }
    }
    RadSampleParams& operator[](size_t i) { return rows[i]; }
    /* Copied on demand, so a case can set fields after construction and before the call. */
    const T& operand() {
        std::memcpy(t.mem.data(), rows.data(), rows.size() * sizeof(RadSampleParams));
        return t;
    }
};

/* ================================================================== argument builder */
struct A {
    std::vector<RadTensor>  ts;
    std::vector<RadParam>   ps;
    std::deque<std::string> strs;      /* stable addresses, unlike a vector's */
    std::vector<float>      scratch;
    RadArgs                 a{};

    A& t(const T& x)   { ts.push_back(x.t); return *this; }
    /* A VIEW, not a fresh tensor: the shape and strides are the case's, which is how a column
     * slice of a wider plane is handed to a kernel (core/runtime/issue.cpp does the same). */
    A& v(const RadTensor& x) { ts.push_back(x); return *this; }
    A& tnull()         { ts.push_back(RadTensor{}); return *this; }
    A& i(const char* k, long long v) { ps.push_back(RAD_INT(k, v)); return *this; }
    /* A float parameter is RAD_P_F64 (docs/OPS.md). Spelled that way here on purpose: this test
     * reaches the launch directly, with no declare in front of it to catch a kind mismatch, so a
     * case that spelled it as anything else would prove nothing about the schema it claims to be
     * testing. */
    A& d(const char* k, double v) { ps.push_back(RAD_F64(k, v)); return *this; }
    A& s(const char* k, const char* v) {
        strs.emplace_back(v);
        ps.push_back(RAD_STR(k, strs.back().c_str()));
        return *this;
    }
    A& scr(int64_t n) { scratch.assign((size_t)n, 0.0f); return *this; }

    const RadArgs* get(int rank = 0, int world = 1) {
        a.t = ts.data();
        a.n_t = (int)ts.size();
        a.p = ps.data();
        a.n_p = (int)ps.size();
        a.scratch = scratch.empty() ? nullptr : scratch.data();
        a.scratch_bytes = (int64_t)scratch.size() * 4;
        a.instance = nullptr;
        a.rank = rank;
        a.world_size = world;
        return &a;
    }
};

int run(const char* op, A& args, int rank = 0, int world = 1) {
    const RadKernelInfo* k = kern(op);
    if (!k || !k->launch) return RAD_E_NOKERNEL;
    return k->launch(args.get(rank, world), nullptr);
}

}  /* namespace */

/* ================================================================== the plugin surface */
TEST(plugin_surface) {
    CHECK(api().abi_version != nullptr);
    CHECK_EQ((long long)api().abi_version(), (long long)RAD_ABI_VERSION);
    const RadPluginInfo* pi = api().info();
    CHECK(pi != nullptr);
    CHECK_EQ((long long)pi->kind, (long long)RAD_PLUGIN_KERNEL);
    CHECK(std::strcmp(pi->name, "libref") == 0);

    /* Every kernel row is host domain at priority 0 and has a launch and a schema. Ref sits last
     * in the hierarchy and must never outbid anything. */
    for (int i = 0; i < api().kernel_count(); ++i) {
        const RadKernelInfo* k = api().kernel_at(i);
        CHECK(k->launch != nullptr);
        CHECK_EQ(k->domain, RAD_DOMAIN_HOST);
        CHECK_EQ(k->priority, 0);
        if (!schema(k->op)) fprintf(stderr, "    no schema for op '%s'\n", k->op);
        CHECK(schema(k->op) != nullptr);
    }
}

/* The conventional vocabulary, spelled out. If an op disappears from the registry this test says
 * which one -- and an op leaving ref's list is an op that no longer has a fallback or an oracle,
 * which is a decision and not an oversight (docs/OPS.md). */
TEST(op_vocabulary) {
    static const char* kOps[] = {
        "rmsnorm", "rmsnorm_add", "hc_enter", "hc_read", "hc_write", "hc_mean", "mtp_enter",
        "layernorm", "add", "mul", "silu_mul", "gelu", "silu",
        "sigmoid", "gather_rows", "scatter_rows", "ngram_ids", "ple_gate", "ple_conv",
        "embed_lookup_q", "cast",
        "scale_rows",
        "softmax",
        "quant_act_i8", "quant_act_fp8", "had_quant_act_fp8", "had_quant_act_i8", "gram_accum",
        /* The fused fp8 forms. A host run needs EVERY op covered, so these are what stands
         * between RADIANCE_HOST_DOMAIN and "site 'host' is unavailable for this band". */
        "rmsnorm_quant_fp8", "gated_quant_fp8", "gemm_nt_q_gated", "dflash_select",
        "rmsnorm_had_quant_i8",
        "gated_had_quant_i8",
        "dequant",
        "gemm_nt", "gemm_nt_bias", "gemm_nt_q", "gemm_nt_q_bias", "grid_embed",
        "attn_paged", "attn_dense", "kv_store", "rope", "rope_table", "qk_norm_rope",
        "router_topk", "moe_scatter", "moe_gemm", "moe_gemm_q", "moe_gemm_q_mod4", "moe_gather",
        "row_topk",
        "row_topk_merge", "logit_rerank",
        "gdn_conv_prep", "gdn_conv_update", "gdn_kkt_solve", "gdn_chunk_scan",
        "gdn_recurrent_update", "gdn_conv_recurrent_update", "gdn_gated_rmsnorm",
        "all_reduce", "all_gather",
        "embed_lookup", "logits_gemm",
        "sample_penalties", "sample_dry", "sample_temp", "sample_topk", "sample_topp",
        "sample_minp", "sample_typical", "sample_xtc", "sample_mask", "sample_pick",
        "sample_argmax", "sample_merge_topk",
    };
    for (const char* op : kOps) {
        if (!kern(op)) fprintf(stderr, "    missing op '%s'\n", op);
        CHECK(kern(op) != nullptr);
    }
    CHECK_EQ(api().kernel_count(), (int)(sizeof(kOps) / sizeof(kOps[0])));

    /* The two structural constraints, and only those. Everything else must match anything. */
    CHECK_EQ(kern("attn_paged")->n_constraints, 1);
    CHECK(std::strcmp(kern("attn_paged")->constraints[0].key, "block_size") == 0);
    CHECK_EQ(kern("kv_store")->n_constraints, 1);
    CHECK_EQ(kern("gemm_nt")->n_constraints, 0);
    CHECK_EQ(kern("gdn_chunk_scan")->n_constraints, 0);
}

/* ================================================================== the numerics
 * Hand-written bit patterns, not round-trips: a round-trip through a broken pair of converters
 * passes. */
TEST(dtype_bf16) {
    CHECK_EQ(bf16_to_f32(0x3f80), 1.0f);
    CHECK_EQ(bf16_to_f32(0xbf80), -1.0f);
    CHECK_EQ(bf16_to_f32(0x4000), 2.0f);
    CHECK_EQ((long long)f32_to_bf16(1.0f), 0x3f80LL);
    CHECK_EQ((long long)f32_to_bf16(-2.0f), 0xc000LL);

    /* Round to nearest, ties to EVEN. 0x3f808000 is exactly halfway between bf16 0x3f80 (even)
     * and 0x3f81, so it rounds down; 0x3f818000 is halfway between 0x3f81 (odd) and 0x3f82, so it
     * rounds up. */
    float half_even, half_odd, above;
    uint32_t u;
    u = 0x3f808000u; std::memcpy(&half_even, &u, 4);
    u = 0x3f818000u; std::memcpy(&half_odd, &u, 4);
    u = 0x3f808001u; std::memcpy(&above, &u, 4);
    CHECK_EQ((long long)f32_to_bf16(half_even), 0x3f80LL);
    CHECK_EQ((long long)f32_to_bf16(half_odd),  0x3f82LL);
    CHECK_EQ((long long)f32_to_bf16(above),     0x3f81LL);

    CHECK_EQ((long long)f32_to_bf16(INFINITY), 0x7f80LL);
    CHECK((f32_to_bf16(std::nanf("")) & 0x7f80u) == 0x7f80u);
    CHECK((f32_to_bf16(std::nanf("")) & 0x007fu) != 0u);   /* still a NaN, not an infinity */
}

TEST(dtype_f16) {
    CHECK_EQ(f16_to_f32(0x3c00), 1.0f);
    CHECK_EQ(f16_to_f32(0xc000), -2.0f);
    CHECK_EQ(f16_to_f32(0x0400), std::ldexp(1.0f, -14));    /* smallest normal */
    CHECK_EQ(f16_to_f32(0x0001), std::ldexp(1.0f, -24));    /* smallest subnormal */
    CHECK_EQ((long long)f32_to_f16(1.0f), 0x3c00LL);
    CHECK_EQ((long long)f32_to_f16(std::ldexp(1.0f, -24)), 0x0001LL);
    CHECK_EQ((long long)f32_to_f16(70000.0f), 0x7c00LL);    /* overflow -> infinity */
    CHECK_EQ(f16_to_f32(f32_to_f16(0.5f)), 0.5f);
}

TEST(dtype_fp8) {
    CHECK_EQ(fp8e4m3_to_f32(0x38), 1.0f);                   /* e=7 m=0 */
    CHECK_EQ(fp8e4m3_to_f32(0xb8), -1.0f);
    CHECK_EQ(fp8e4m3_to_f32(0x7e), 448.0f);                 /* the largest finite */
    CHECK_EQ((long long)f32_to_fp8e4m3(1.0f), 0x38LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(448.0f), 0x7eLL);
    CHECK_EQ((long long)f32_to_fp8e4m3(1e30f), 0x7eLL);     /* saturates, never NaN */
    CHECK_EQ(fp8e5m2_to_f32(0x3c), 1.0f);

    /* THE BAND UNDER THE SMALLEST SUBNORMAL. 2^-9 is code 0x01 and 2^-10 is the tie between it and
     * zero: the tie goes to the even code, anything above it goes up, and only what is below it
     * flushes. */
    CHECK_EQ(fp8e4m3_to_f32(0x01), std::ldexp(1.0f, -9));
    CHECK_EQ((long long)f32_to_fp8e4m3(std::ldexp(1.0f, -10)), 0x00LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::nextafter(std::ldexp(1.0f, -10), 1.0f)), 0x01LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::ldexp(1.5f, -10)), 0x01LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(-std::ldexp(1.5f, -10)), 0x81LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::nextafter(std::ldexp(1.0f, -10), 0.0f)), 0x00LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::ldexp(3.0f, -10)), 0x02LL);   /* tie 1|2 -> even */
}

/* THE ENCODER AGAINST AN INDEPENDENT ROUNDING, at every point where rounding can go wrong.
 *
 * The reference is written from the format's definition and shares nothing with the encoder: the
 * 127 non-negative finite codes are decoded with ldexp, a value takes the nearest of them in
 * double precision, an exact tie takes the even code, and anything at or past 448 saturates.
 *
 * Every rounding boundary of e4m3 sits at or above mantissa bit 12 of an f32, normal or
 * subnormal, so the sweep takes every sign, every exponent and every value of the top eleven
 * mantissa bits, and under them the low twelve bits at each side of zero and of the half-way
 * point: every tie, the value one ulp either side of it, and every code's own value. */
TEST(dtype_fp8_encode_rounds_to_nearest_even) {
    double tab[127];
    for (int c = 0; c < 127; ++c) {
        const int e = (c >> 3) & 15, m = c & 7;
        tab[c] = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    }
    auto want = [&](float f) -> uint8_t {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint8_t s = (uint8_t)((u >> 24) & 0x80u);
        if (std::isnan(f)) return (uint8_t)(s | 0x7fu);
        const double a = std::fabs((double)f);
        if (a >= 448.0) return (uint8_t)(s | 0x7eu);
        int lo = 0, hi = 126;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (tab[mid] <= a) lo = mid; else hi = mid;
        }
        const double dl = a - tab[lo], dh = tab[hi] - a;
        const int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
        return (uint8_t)(s | c);
    };
    static const uint32_t kLow[] = { 0x000u, 0x001u, 0x7ffu, 0x800u, 0x801u, 0xfffu };
    long long bad = 0, n = 0;
    uint32_t first = 0;
    for (uint32_t hi = 0; hi < (1u << 20); ++hi) {           /* sign, exponent, top 11 mantissa */
        for (uint32_t lo : kLow) {
            const uint32_t u = (hi << 12) | lo;
            float f;
            std::memcpy(&f, &u, 4);
            ++n;
            if (f32_to_fp8e4m3(f) != want(f) && bad++ == 0) first = u;
        }
    }
    if (bad) fprintf(stderr, "    first mismatch at f32 bits 0x%08x\n", first);
    CHECK_EQ(bad, 0LL);
    CHECK_EQ(n, 6LL << 20);
}

TEST(dtype_packed) {
    /* w4: element 2i in the LOW nibble, value = (n ^ 8) - 8, which is plain 4-bit two's
     * complement -- codes 0..7 are 0..7 and codes 8..15 are -8..-1. */
    uint8_t b = 0x21;                     /* low 0x1 -> 1, high 0x2 -> 2 */
    CHECK_EQ(w4_code(&b, 0), 1.0f);
    CHECK_EQ(w4_code(&b, 1), 2.0f);
    uint8_t c = 0xf8;                     /* low 0x8 -> -8, high 0xf -> -1 */
    CHECK_EQ(w4_code(&c, 0), -8.0f);
    CHECK_EQ(w4_code(&c, 1), -1.0f);

    /* w2 is the same construction one radix down: 0->0, 1->1, 2->-2, 3->-1. */
    uint8_t d = 0xe4;                     /* 0b11_10_01_00, codes 0,1,2,3 from the low end */
    CHECK_EQ(w2_code(&d, 0), 0.0f);
    CHECK_EQ(w2_code(&d, 1), 1.0f);
    CHECK_EQ(w2_code(&d, 2), -2.0f);
    CHECK_EQ(w2_code(&d, 3), -1.0f);

    uint8_t m = 0xa2;                     /* low 0x2 = +1.0, high 0xa = -1.0 */
    CHECK_EQ(mxfp4_code(&m, 0), 1.0f);
    CHECK_EQ(mxfp4_code(&m, 1), -1.0f);
    CHECK_EQ(e8m0_to_f32(127), 1.0f);
    CHECK_EQ(e8m0_to_f32(128), 2.0f);
    CHECK_EQ(e8m0_to_f32(126), 0.5f);
}

/* ================================================================== elementwise and norms */
TEST(op_rmsnorm) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T w(RAD_F32, {4});    w.f({1, 1, 1, 1});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(w).t(y).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("rmsnorm", a));
    /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386128 */
    const float s = 1.0f / 2.7386128f;
    CHECK_NEAR(y.at(0), 1 * s, 1e-6);
    CHECK_NEAR(y.at(3), 4 * s, 1e-6);

    /* wadd is GemmaRMSNorm's `1 +`: a zero-centred gain of 0 must behave as a gain of 1. */
    T w0(RAD_F32, {4}); w0.f({0, 0, 0, 0});
    T y2(RAD_F32, {1, 4});
    A b; b.t(x).t(w0).t(y2).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32").d("wadd", 1);
    CHECK_OK(run("rmsnorm", b));
    CHECK_NEAR(y2.at(3), 4 * s, 1e-6);
}

TEST(op_rmsnorm_add) {
    T x(RAD_F32, {1, 2});   x.f({1, 3});
    T res(RAD_F32, {1, 2}); res.f({1, 1});
    T w(RAD_F32, {2});      w.f({1, 1});
    T y(RAD_F32, {1, 2}), ro(RAD_F32, {1, 2});
    A a; a.t(x).t(res).t(w).t(y).t(ro).i("M", 1).i("n", 2).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("rmsnorm_add", a));
    /* residual_out = [2, 4]; rms = sqrt((4+16)/2) = sqrt(10) = 3.1622777 */
    CHECK_EQ(ro.at(0), 2.0f);
    CHECK_EQ(ro.at(1), 4.0f);
    CHECK_NEAR(y.at(0), 2.0f / 3.1622777f, 1e-6);
    CHECK_NEAR(y.at(1), 4.0f / 3.1622777f, 1e-6);
}

/* ================================================================== hc_mean: the DFlash2 tap
 *
 * THE EXPECTATIONS ARE LITERALS AND THEY DISCRIMINATE. Every case below is chosen so a wrong op
 * gives a different number:
 *
 *   MEAN vs SUM     streams [1,0] and [3,3] average to [2, 1.5]; a sum gives [4, 3].
 *   MEAN vs RMS     the same pair averages 1.5 where an rms would give sqrt((0+9)/2) = 2.1213.
 *   SIGNED          1 and -1 cancel to 0, which a sum of squares or an rms cannot produce.
 *   NONUNIFORM      the four streams of one element are 1, 2, 3 and 10 -- an implementation that
 *                   mixed up the c and i axes averages a different set.
 *
 * bf16 stores 1.5, 2 and 0 exactly, so every literal below is exact rather than a tolerance. */
TEST(op_hc_mean) {
    /* MEAN, not SUM. h = [1, 0, 3, 3] is stream 0 [1,0] and stream 1 [3,3], so the element-wise
     * means are [2, 1.5]. A sum would give [4, 3]; an rms gate would give [2.1213, 2.1213]. */
    T h2(RAD_BF16, {1, 4}); h2.f({1, 0, 3, 3});
    T x2(RAD_BF16, {1, 2});
    A b; b.t(h2).t(x2).i("M", 1).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", b));
    CHECK_EQ(x2.at(0), 2.0f);
    CHECK_EQ(x2.at(1), 1.5f);

    /* SIGNED CANCELLATION: stream 0 [1, -1], stream 1 [-1, 1] -> [0, 0]. */
    T h3(RAD_BF16, {1, 4}); h3.f({1, -1, -1, 1});
    T x3(RAD_BF16, {1, 2});
    A c; c.t(h3).t(x3).i("M", 1).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", c));
    CHECK_EQ(x3.at(0), 0.0f);
    CHECK_EQ(x3.at(1), 0.0f);

    /* THE PRODUCTION hc 4 and the AXIS ORDER. The layout is h[m, c*N+i], so the four streams of
     * row 0 are [1,-1], [2,-1], [3,1] and [10,1]: their element-wise means are (1+2+3+10)/4 = 4
     * and (-1-1+1+1)/4 = 0. A kernel that read the row as [M, N, hc] -- the transposed reading,
     * which is the one mistake this layout invites -- would average (1,2,3,10) and (-1,-1,1,1)
     * together differently and land elsewhere. */
    T h4(RAD_BF16, {2, 8});
    h4.f({1, -1,  2, -1,  3, 1,  10, 1,        /* row 0 */
          0, 4,   0, 4,   0, 4,  0, 4});       /* row 1: every stream [0,4] */
    T x4(RAD_BF16, {2, 2});
    A d; d.t(h4).t(x4).i("M", 2).i("N", 2).i("hc", 4).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", d));
    CHECK_EQ(x4.at(0), 4.0f);          /* (1+2+3+10)/4 */
    CHECK_EQ(x4.at(1), 0.0f);          /* (-1-1+1+1)/4 */
    CHECK_EQ(x4.at(2), 0.0f);
    CHECK_EQ(x4.at(3), 4.0f);

    /* THE SIGN OF A ZERO IS PART OF THE ANSWER, and it is the one case a mean-vs-sum test cannot
     * see. The oracle's loop starts its accumulator at +0.0f, and +0 + (-0) is +0; a kernel that
     * seeded the sum from stream 0 would keep -0 through four negative zeros and store the
     * different bf16 bit 0x8000. A GPU probe found exactly that at 1/8/2048 rows, so the bits are
     * pinned here rather than the value. */
    T hz(RAD_BF16, {1, 4});
    hz.f({-0.0f, -0.0f, -0.0f, -0.0f});
    T xz(RAD_BF16, {1, 1});
    A z; z.t(hz).t(xz).i("M", 1).i("N", 1).i("hc", 4).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", z));
    /* Read the raw 16-bit word: CHECK_EQ on the float cannot tell +0 from -0. */
    CHECK_EQ((long long)((const uint16_t*)xz.t.data)[0], 0LL);
}

/* THE OUTPUT IS A COLUMN VIEW OF A WIDER PLANE, and the neighbouring taps must come out untouched.
 * That is the whole reason the op takes a pitch: a kernel that wrote a packed [M, N] would silently
 * overwrite tap t+1 with tap t's values, and the model would still produce text. */
TEST(op_hc_mean_writes_through_the_output_pitch) {
    const int64_t M = 2, N = 3, taps = 4;
    T h(RAD_BF16, {M, 4 * N});
    h.f({1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12,     /* row 0, streams laid out by N */
         2, 4, 6,  8, 10, 12, 14, 16, 18, 20, 22, 24}); /* row 1 */
    /* The plane itself, pre-filled with a sentinel so an out-of-column store is visible. */
    T plane(RAD_BF16, {M, taps * N});
    std::vector<float> sentinel((size_t)(M * taps * N), 7.0f);
    plane.f(sentinel);

    /* Tap 1 = columns N .. 2N-1, keeping the plane's taps*N as the row stride. */
    RadTensor view = plane.t;
    view.data = (uint8_t*)plane.t.data + (size_t)N * 2;
    view.shape[0] = M; view.shape[1] = N;
    view.stride[0] = taps * N; view.stride[1] = 1;

    A a; a.t(h).v(view).i("M", M).i("N", N).i("hc", 4).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", a));

    auto at = [&](int64_t m, int64_t i) { return plane.at(m * taps * N + i); };
    /* Row 0: (1+4+7+10)/4 = 5.5, (2+5+8+11)/4 = 6.5, (3+6+9+12)/4 = 7.5
     * Row 1: (2+8+14+20)/4 = 11, (4+10+16+22)/4 = 13, (6+12+18+24)/4 = 15 */
    CHECK_EQ(at(0, 0 * N + 0), 7.0f);  /* tap 0 untouched */
    CHECK_EQ(at(0, 0 * N + 2), 7.0f);
    CHECK_EQ(at(0, 1 * N + 0), 5.5f);
    CHECK_EQ(at(0, 1 * N + 1), 6.5f);
    CHECK_EQ(at(0, 1 * N + 2), 7.5f);
    CHECK_EQ(at(0, 2 * N + 0), 7.0f);  /* tap 2 untouched */
    CHECK_EQ(at(0, 3 * N + 2), 7.0f);
    CHECK_EQ(at(1, 0 * N + 1), 7.0f);
    CHECK_EQ(at(1, 1 * N + 0), 11.0f);
    CHECK_EQ(at(1, 1 * N + 2), 15.0f);
    CHECK_EQ(at(1, 2 * N + 1), 7.0f);
    /* And the last tap's last element, where an overrun past the plane would land. */
    CHECK_EQ(at(1, 3 * N + 2), 7.0f);
}

/* AN OUTPUT PITCH NARROWER THAN THE ROW IS REFUSED. A column view's stride[0] is the plane's
 * width, so it is at least the slice's own width; a smaller one would make row m's store land in
 * row m+1's columns, which is a silent overwrite of another row rather than a wrong answer. One
 * row is exempt because the pitch is never used there. */
TEST(op_hc_mean_refuses_an_output_pitch_narrower_than_the_row) {
    const int64_t N = 4, hc = 2;
    T h(RAD_BF16, {2, hc * N});
    h.f({1, 1, 1, 1,  2, 2, 2, 2,       /* row 0: streams [1,1,1,1] and [2,2,2,2] -> 1.5 */
         3, 3, 3, 3,  4, 4, 4, 4});     /* row 1: streams [3,3,3,3] and [4,4,4,4] -> 3.5 */

    /* A packed [2, N] output is legitimate -- pitch == N -- and is the control. */
    T ok(RAD_BF16, {2, N});
    A a; a.t(h).t(ok).i("M", 2).i("N", N).i("hc", hc).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", a));
    CHECK_EQ(ok.at(0), 1.5f);
    CHECK_EQ(ok.at(N), 3.5f);

    /* The same rows with stride[0] = N-1: row 1 would write into row 0's last column. */
    T narrow(RAD_BF16, {2, N});
    RadTensor bad = narrow.t;
    bad.stride[0] = N - 1;
    A b; b.t(h).v(bad).i("M", 2).i("N", N).i("hc", hc).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", b), RAD_E_SHAPE);

    /* And the input side: h's row pitch under hc*N. */
    RadTensor hbad = h.t;
    hbad.stride[0] = hc * N - 1;
    T x(RAD_BF16, {2, N});
    A c; c.v(hbad).t(x).i("M", 2).i("N", N).i("hc", hc).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", c), RAD_E_SHAPE);

    /* ONE row with a zero pitch is accepted: nothing reads it, and several callers describe a
     * single-row slice that way. */
    RadTensor one = h.t;
    one.shape[0] = 1;
    one.stride[0] = 0;
    T x1(RAD_BF16, {1, N});
    RadTensor gone = x1.t;
    gone.stride[0] = 0;
    A d; d.v(one).v(gone).i("M", 1).i("N", N).i("hc", hc).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", d));
    CHECK_EQ(x1.at(0), 1.5f);
    CHECK_EQ(x1.at(N - 1), 1.5f);
}

/* A pitched INPUT works too: a row of `h` whose streams are N apart but which sits at a column
 * offset of a wider buffer. The op reads `h`'s own stride, so this is the same arithmetic over the
 * same relative layout. */
TEST(op_hc_mean_reads_a_pitched_input) {
    const int64_t N = 2, hc = 2;
    T wide(RAD_BF16, {2, 2 * hc * N});   /* twice as wide as one row of h */
    /* The view's row pitch is the buffer's width (8), not hc*N (4), so each row's four elements
     * are followed by four that are not part of it. */
    std::vector<float> v((size_t)(2 * 2 * hc * N), -9.0f);
    v[0] = 1; v[1] = 3;  v[2] = 1; v[3] = 5;                       /* row 0: [1,3] [1,5] */
    v[8] = 2; v[9] = 4;  v[10] = 2; v[11] = 8;                     /* row 1, one pitch further */
    wide.f(v);

    RadTensor h = wide.t;
    h.shape[0] = 2; h.shape[1] = hc * N;
    h.stride[0] = 2 * hc * N; h.stride[1] = 1;

    T x(RAD_BF16, {2, N});
    A a; a.v(h).t(x).i("M", 2).i("N", N).i("hc", hc).s("dtype", "bf16");
    CHECK_OK(run("hc_mean", a));
    CHECK_EQ(x.at(0), 1.0f);    /* (1+1)/2 */
    CHECK_EQ(x.at(1), 4.0f);    /* (3+5)/2 */
    CHECK_EQ(x.at(2), 2.0f);    /* (2+2)/2 */
    CHECK_EQ(x.at(3), 6.0f);    /* (4+8)/2 */
}

/* INVALID GEOMETRY IS REFUSED, not computed. Each of these would otherwise produce a plausible
 * number: a width that disagrees with the parameter reads a different element, an f32 operand is
 * a different op, and an output aliasing the input is a self-overwrite. */
TEST(op_hc_mean_refuses_what_it_is_not) {
    T h(RAD_BF16, {1, 4}); h.f({1, 2, 3, 4});
    T x(RAD_BF16, {1, 2});

    /* The declared width is hc*N; a 4-wide `h` under N=3 describes a different call. */
    A a; a.t(h).t(x).i("M", 1).i("N", 3).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", a), RAD_E_SHAPE);

    /* An f32 stream. */
    T hf(RAD_F32, {1, 4}); hf.f({1, 2, 3, 4});
    A b; b.t(hf).t(x).i("M", 1).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", b), RAD_E_DTYPE);

    /* An f32 output under a bf16 call. */
    T xf(RAD_F32, {1, 2});
    A c; c.t(h).t(xf).i("M", 1).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", c), RAD_E_DTYPE);

    /* A dtype the op does not have. */
    A d; d.t(h).t(x).i("M", 1).i("N", 2).i("hc", 2).s("dtype", "f32");
    CHECK_EQ(run("hc_mean", d), RAD_E_DTYPE);

    /* hc 0 and n 0 are not geometries. */
    A e; e.t(h).t(x).i("M", 1).i("N", 2).i("hc", 0).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", e), RAD_E_SHAPE);
    A f; f.t(h).t(x).i("M", 1).i("N", 0).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", f), RAD_E_SHAPE);

    /* The output must not alias the input. */
    RadTensor self = h.t;
    self.shape[0] = 1; self.shape[1] = 2;
    A g; g.t(h).v(self).i("M", 1).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", g), RAD_E_INVAL);

    /* A required operand absent. */
    A i; i.t(h).tnull().i("M", 1).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", i), RAD_E_INVAL);

    /* More rows than the band admits. `M` is the ceiling, not the count. */
    T h2(RAD_BF16, {4, 4}); h2.f({1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
    T x2(RAD_BF16, {4, 2});
    A j; j.t(h2).t(x2).i("M", 2).i("N", 2).i("hc", 2).s("dtype", "bf16");
    CHECK_EQ(run("hc_mean", j), RAD_E_SHAPE);
}

TEST(op_layernorm) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T w(RAD_F32, {4});    w.f({1, 1, 1, 1});
    T b(RAD_F32, {4});    b.f({0, 0, 0, 0});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(w).t(b).t(y).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("layernorm", a));
    /* mean 2.5, biased var 1.25, sd 1.1180340 */
    CHECK_NEAR(y.at(0), -1.5f / 1.1180340f, 1e-6);
    CHECK_NEAR(y.at(3),  1.5f / 1.1180340f, 1e-6);

    /* The bias operand is optional -- a null tensor is not an error. */
    T y2(RAD_F32, {1, 4});
    A c; c.t(x).t(w).tnull().t(y2).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("layernorm", c));
    CHECK_NEAR(y2.at(0), -1.5f / 1.1180340f, 1e-6);
}

TEST(op_add_mul) {
    T p(RAD_F32, {2, 2}); p.f({1, 2, 3, 4});
    T q(RAD_F32, {2, 2}); q.f({10, 20, 30, 40});
    T y(RAD_F32, {2, 2});
    A a; a.t(p).t(q).t(y).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("add", a));
    CHECK_EQ(y.at(0), 11.0f);
    CHECK_EQ(y.at(3), 44.0f);

    A b; b.t(p).t(q).t(y).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("mul", b));
    CHECK_EQ(y.at(0), 10.0f);
    CHECK_EQ(y.at(3), 160.0f);

    /* One row broadcasts along rows; that is the bias-add case and the only broadcast there is. */
    T bias(RAD_F32, {2}); bias.f({100, 200});
    A c; c.t(p).t(bias).t(y).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("add", c));
    CHECK_EQ(y.at(0), 101.0f);
    CHECK_EQ(y.at(3), 204.0f);
}

TEST(op_silu_mul_and_unary) {
    T gu(RAD_F32, {1, 4}); gu.f({1, 2, 3, 4});      /* gate [1,2], up [3,4] */
    T y(RAD_F32, {1, 2});
    A a; a.t(gu).t(y).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("silu_mul", a));
    CHECK_NEAR(y.at(0), 0.73105858f * 3.0f, 1e-6);
    CHECK_NEAR(y.at(1), 1.76159416f * 4.0f, 1e-6);

    T x(RAD_F32, {2}); x.f({0, 1});
    T g(RAD_F32, {2}), s(RAD_F32, {2});
    A b; b.t(x).t(g).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("gelu", b));
    CHECK_EQ(g.at(0), 0.0f);
    CHECK_NEAR(g.at(1), 0.84134475f, 1e-6);          /* the exact erf gelu, not tanh */
    A c; c.t(x).t(s).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("silu", c));
    CHECK_NEAR(s.at(1), 0.73105858f, 1e-6);
}

/* The quantised gather. Three things, and the scale is the least interesting of them.
 *
 * The E4M3 codes are chosen so their VALUES are exact and writable as literals: 0x38 is 1.0,
 * 0x40 is 2.0, 0x3c is 1.5, 0xb8 is -1.0 (see abi/rad_plugin.h's decoder). So the expectation is
 * a literal and not a second call into the same conversion, which is the rule this file exists to
 * keep -- a test that computes its answer the way the code does proves only that the code agrees
 * with itself. */
TEST(op_embed_lookup_q) {
    T wte(RAD_F8E4M3, {4, 2});
    wte.raw({ 0x38, 0x40,      /* row 0:  1.0,  2.0 */
              0x3c, 0xb8,      /* row 1:  1.5, -1.0 */
              0x00, 0x38,      /* row 2:  0.0,  1.0 */
              0x40, 0x40 });   /* row 3:  2.0,  2.0 */
    T sc(RAD_BF16, {1}); sc.f({ 0.25f });
    T tok(RAD_I32, {3}); tok.n({ 1, 3, -1 });
    T x(RAD_BF16, {3, 2});

    A a; a.t(tok).t(wte).t(sc).t(x)
         .i("M", 3).i("n_embd", 2).i("n_vocab", 4).s("dtype", "bf16");
    CHECK_OK(run("embed_lookup_q", a));

    CHECK_EQ(x.at(0), 0.375f);      /*  1.5 * 0.25 */
    CHECK_EQ(x.at(1), -0.25f);      /* -1.0 * 0.25 */
    CHECK_EQ(x.at(2), 0.5f);        /*  2.0 * 0.25 */
    CHECK_EQ(x.at(3), 0.5f);
    /* A NEGATIVE ID IS PADDING AND WRITES A ZERO ROW -- the same convention embed_lookup has, and
     * the arm a vocab-sharded table depends on: every rank writes zeros for the ids outside its
     * slice and the all_reduce that follows sums to the right row. */
    CHECK_EQ(x.at(4), 0.0f);
    CHECK_EQ(x.at(5), 0.0f);

    /* ...and an id AT the table's row count is refused rather than read, because reading it would
     * be reading whatever the container put after the table. */
    T bad(RAD_I32, {1}); bad.n({ 4 });
    T y(RAD_BF16, {1, 2});
    A b; b.t(bad).t(wte).t(sc).t(y)
         .i("M", 1).i("n_embd", 2).i("n_vocab", 4).s("dtype", "bf16");
    CHECK(run("embed_lookup_q", b) != RAD_OK);
}

/* PLE's gate, on paper. n = 2 and hc = 2, which is the smallest shape where both asymmetries of
 * the op are visible: the rsqrt is per n-wide stream while the gain spans hc*n, and the single
 * `v` row is gated differently into each stream.
 *
 * The inputs are chosen so every norm is exactly 1: a stream of (1, -1) or (1, 1) has mean square
 * 1, so with eps 0 the rsqrt is 1 and the normed value is the input times its gain. That turns the
 * whole op into arithmetic that can be written down.
 *
 *   stream 0:  K = (1, -1),  Q = (1, 1)  ->  dot 0       -> s = 0, and sign(0) = 0 so it STAYS 0
 *   stream 1:  K = (1,  1),  Q = (1, 1)  ->  dot 2       -> 2/sqrt(2) = sqrt(2), then sqrt again
 *
 * The first is the case the clamp would get wrong: clamping the magnitude to 1e-6 and re-signing
 * the CLAMPED value gives 1e-3 where torch gives 0. */
TEST(op_ple_gate) {
    T k(RAD_F32, {1, 4}); k.f({ 1, -1,  1, 1 });
    T q(RAD_F32, {1, 4}); q.f({ 1,  1,  1, 1 });
    T v(RAD_F32, {1, 2}); v.f({ 4, 8 });
    T wk(RAD_F32, {4}); wk.f({ 1, 1, 1, 1 });
    T wq(RAD_F32, {4}); wq.f({ 1, 1, 1, 1 });
    T wc(RAD_F32, {4}); wc.f({ 1, 1, 1, 1 });
    T gv(RAD_F32, {1, 4}), gvn(RAD_F32, {1, 4});

    A a; a.t(k).t(q).t(v).t(wk).t(wq).t(wc).t(gv).t(gvn)
         .i("M", 1).i("n", 2).i("hc", 2).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("ple_gate", a));

    /* Stream 0: sigmoid(0) = 0.5 exactly. */
    CHECK_NEAR(gv.at(0), 0.5f * 4.0f, 1e-6);
    CHECK_NEAR(gv.at(1), 0.5f * 8.0f, 1e-6);
    /* Stream 1: s = 2/sqrt(2) = 1.41421356, then sqrt -> 1.18920712, sigmoid -> 0.76659. */
    const float g1 = 1.0f / (1.0f + std::exp(-1.18920712f));
    CHECK_NEAR(gv.at(2), g1 * 4.0f, 1e-5);
    CHECK_NEAR(gv.at(3), g1 * 8.0f, 1e-5);

    /* AND THE CONVOLUTION BRANCH DOES NOT SEE THE GATE AT ALL. This is a property of the model and
     * not of this case: gv[c] is the SCALAR g[c] times one shared v row, and norm_conv's rms is
     * taken per n-wide stream -- so the same scalar appears in the numerator and in the rms and
     * divides straight back out. gvn[c][d] is v[d]/rms(v) times its gain, whatever the gate did.
     *
     * Worth asserting because it is a sharp invariant: a kernel whose `gvn` varies with the gate
     * has fused something it should not have. (It is exact only in exact arithmetic -- gv is
     * narrowed before the norm reads it -- so this is an invariant to CHECK, not a shortcut to
     * take. Computing gvn from v directly would differ in the last bits.)
     *
     * Stream 0 holds (2, 4), so rms = sqrt(10) and the normed row is (2, 4)/sqrt(10). */
    const float r0 = 1.0f / std::sqrt((4.0f + 16.0f) / 2.0f);
    CHECK_NEAR(gvn.at(0), 2.0f * r0, 1e-5);
    CHECK_NEAR(gvn.at(1), 4.0f * r0, 1e-5);
    CHECK_NEAR(gvn.at(2), gvn.at(0), 1e-5);
    CHECK_NEAR(gvn.at(3), gvn.at(1), 1e-5);

    /* wadd is the `1 +` of Qwen4ExpTextRMSNorm: a zero-centred gain must behave as a gain of 1. */
    T w0(RAD_F32, {4}); w0.f({ 0, 0, 0, 0 });
    T gv2(RAD_F32, {1, 4}), gvn2(RAD_F32, {1, 4});
    A b; b.t(k).t(q).t(v).t(w0).t(w0).t(w0).t(gv2).t(gvn2)
         .i("M", 1).i("n", 2).i("hc", 2).d("eps", 0).s("dtype", "f32").d("wadd", 1);
    CHECK_OK(run("ple_gate", b));
    for (int i = 0; i < 4; ++i) CHECK_NEAR(gv2.at(i), gv.at(i), 1e-6);

    /* THE SIGN SURVIVES. Flipping stream 1's query makes the dot -2, and the gate must land below
     * a half rather than at zero -- which is what a signed root is for. */
    T qn(RAD_F32, {1, 4}); qn.f({ 1, 1, -1, -1 });
    T gv3(RAD_F32, {1, 4}), gvn3(RAD_F32, {1, 4});
    A c; c.t(k).t(qn).t(v).t(wk).t(wq).t(wc).t(gv3).t(gvn3)
         .i("M", 1).i("n", 2).i("hc", 2).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("ple_gate", c));
    const float gneg = 1.0f / (1.0f + std::exp(1.18920712f));
    CHECK_NEAR(gv3.at(2), gneg * 4.0f, 1e-5);
    CHECK(gv3.at(2) > 0.0f);            /* below half, not zero */
}

/* PLE's dilated convolution. Width 3 at dilation 2, so the taps are 4, 2 and 0 timesteps back and
 * the history is (3-1)*2 = 4 -- small enough to check by hand and still wide enough that a kernel
 * that ignored the dilation would read 2, 1, 0 and be caught.
 *
 * The weights are (100, 10, 1), so the accumulator READS OUT AS A DECIMAL NUMBER: a tap triple of
 * (a, b, c) accumulates to 100a + 10b + c and any misread tap changes a digit. silu is applied
 * after, so the expectation is silu of that number.
 */
TEST(op_ple_conv) {
    auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
    const int n = 1, wid = 3, dil = 2, hist = (wid - 1) * dil;   /* 4 */

    /* ---- COLD: the window in front of the sequence is ZEROS, not the slot's contents --------- */
    {
        T x(RAD_F32, {6, 1}); x.f({ 1, 2, 3, 4, 5, 6 });
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        T cs(RAD_F32, {1, 1, hist}); cs.f({ 7, 7, 7, 7 });   /* a recycled slot's leftovers */
        T cu(RAD_I32, {2}); cu.n({ 0, 6 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 0 });                  /* COLD */
        T y(RAD_F32, {6, 1});
        A a; a.t(x).t(w).t(cs).t(cu).tnull().t(ci).t(hi).tnull().t(y)
             .i("M", 6).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", a));
        /* t=0 reads x[-4], x[-2], x[0] = 0, 0, 1  -> 1
         * t=1 reads x[-3], x[-1], x[1] = 0, 0, 2  -> 2
         * t=2 reads x[-2], x[0],  x[2] = 0, 1, 3  -> 13
         * t=4 reads x[0],  x[2],  x[4] = 1, 3, 5  -> 135  <- every tap in the sequence */
        CHECK_NEAR(y.at(0), silu(1.0f), 1e-4);
        CHECK_NEAR(y.at(1), silu(2.0f), 1e-4);
        CHECK_NEAR(y.at(2), silu(13.0f), 1e-4);
        CHECK_NEAR(y.at(4), silu(135.0f), 1e-4);
        /* ...and the state left behind is the last `hist` inputs, at offset zero. */
        CHECK_EQ(cs.at(0), 3.0f);
        CHECK_EQ(cs.at(1), 4.0f);
        CHECK_EQ(cs.at(2), 5.0f);
        CHECK_EQ(cs.at(3), 6.0f);
    }

    /* ---- WARM: the same six tokens as two steps must equal them as one ----------------------- */
    {
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        T cs(RAD_F32, {1, 1, 4}); cs.f({ 0, 0, 0, 0 });
        T cu(RAD_I32, {2}); cu.n({ 0, 4 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T cold(RAD_I32, {1}); cold.n({ 0 });
        T warm(RAD_I32, {1}); warm.n({ 1 });

        T x1(RAD_F32, {4, 1}); x1.f({ 1, 2, 3, 4 });
        T y1(RAD_F32, {4, 1});
        A a; a.t(x1).t(w).t(cs).t(cu).tnull().t(ci).t(cold).tnull().t(y1)
             .i("M", 4).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", a));

        T x2(RAD_F32, {2, 1}); x2.f({ 5, 6 });
        T cu2(RAD_I32, {2}); cu2.n({ 0, 2 });
        T y2(RAD_F32, {2, 1});
        A b; b.t(x2).t(w).t(cs).t(cu2).tnull().t(ci).t(warm).tnull().t(y2)
             .i("M", 2).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", b));
        /* The continuation's t=0 is the whole sequence's t=4, which reads 1, 3, 5 -> 135. THIS is
         * the case a cache off by one fails: with history width-1 = 2 instead of (width-1)*dilation
         * = 4 it would have kept only (5, 6) and read zeros where the 1 belongs. */
        CHECK_NEAR(y2.at(0), silu(135.0f), 1e-4);
        CHECK_NEAR(y2.at(1), silu(246.0f), 1e-4);
    }

    /* ---- `resid` is added AFTER the silu, and `num_accepted` shifts the read ----------------- */
    {
        T x(RAD_F32, {1, 1}); x.f({ 1 });
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        /* A window of six for a read window of four: two slots of speculative slack. */
        T cs(RAD_F32, {1, 1, 6}); cs.f({ 9, 9, 2, 0, 3, 0 });
        T cu(RAD_I32, {2}); cu.n({ 0, 1 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 1 });
        T na(RAD_I32, {1}); na.n({ 3 });                  /* off = 2 */
        T rs(RAD_F32, {1, 1}); rs.f({ 1000 });
        T y(RAD_F32, {1, 1});
        A a; a.t(x).t(w).t(cs).t(cu).t(rs).t(ci).t(hi).t(na).t(y)
             .i("M", 1).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", a));
        /* off = num_accepted - 1 = 2, so the window is cs[2..5] = (2, 0, 3, 0) and t=0 reads
         * hist[0]=2, hist[2]=3, x[0]=1 -> 231. Read at offset 0 it would have been 9,2,1 = 921. */
        CHECK_NEAR(y.at(0), 1000.0f + silu(231.0f), 1e-3);
    }
}

/* PLE's n-gram ids, ON THE REAL CHECKPOINT CONSTANTS.
 *
 * The multipliers, vocab sizes and offsets below are the bytes of
 * `model.language_model.layers.1.ple.ple_embedding.*` in Qwen3.8-Flash-Next, read out of the
 * safetensors: a synthetic multiplier would not exercise the 10^13 magnitude that makes the
 * products fill 63 bits, which is where a hash goes wrong.
 *
 * THE EXPECTATIONS ARE PROPERTIES, NOT A SECOND COPY OF THE FORMULA. Recomputing
 * `t0*m0 ^ t1*m1 % v + o` here would only prove the plugin agrees with itself. What is checked is
 * what the REFERENCE's `_shift_right_ignore_eos` and its head blocking say, read off the python and
 * not off ref_ngram.cpp:
 *
 *   - a 2-gram head depends on t0 and t1 and NOT on t2; a 3-gram head depends on all three;
 *   - an EOS at p-1 makes both t1 and t2 fall back to eos, so the row equals one whose history is
 *     entirely eos;
 *   - an EOS at p-2 alone makes t2 fall back but leaves t1 -- and since t2 was that eos anyway,
 *     the row is UNCHANGED, which is the identity that reduces the reference's cummax to two
 *     compares;
 *   - the first rows of a sequence read `prev`, and the later ones read the batch;
 *   - two sequences in one batch do not see each other's tokens.
 */
TEST(op_ngram_ids) {
    const int64_t kEos = 248044, kHeads = 16, kNgram = 3;
    const int64_t mult[3] = { 23703573157769LL, 20109073645365LL, 8052911324071LL };
    const int64_t vsz[16] = { 20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                              20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                              20000153, 20000159, 20000161, 20000171 };
    int64_t offs[16];
    offs[0] = 0;
    for (int i = 1; i < 16; ++i) offs[i] = offs[i - 1] + vsz[i - 1];

    T tm(RAD_I64, {3});  tm.n({ mult[0], mult[1], mult[2] });
    T tv(RAD_I64, {16}); T to(RAD_I64, {16});
    { std::vector<int64_t> a(vsz, vsz + 16), b(offs, offs + 16); tv.n(a); to.n(b); }

    /* Every id lands inside its head's slice of the 320-million-row table -- the property that
     * makes the offsets meaningful, and the one a harness drawing small vocab sizes would miss. */
    auto in_slice = [&](const T& ids, int64_t row, int64_t h) {
        const int64_t v = ids.iat(row * kHeads + h);
        return v >= offs[h] && v < offs[h] + vsz[h];
    };

    /* `prev` is the WARM contents of a one-slot state window, oldest first. The state is INOUT, so
     * each call gets its own -- a shared one would carry the previous case's tail forward. */
    auto run4 = [&](std::vector<int64_t> tok, std::vector<int64_t> prev, T& ids) {
        const int64_t M = (int64_t)tok.size();
        T tt(RAD_I32, {M}); tt.n(tok);
        T ts(RAD_I32, {1, 2}); ts.n(prev);
        T tc(RAD_I32, {2}); tc.n({0, M});
        T ci(RAD_I32, {1}); ci.n({0});
        T hi(RAD_I32, {1}); hi.n({1});            /* WARM: read the window as given */
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids)
             .i("M", M).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        return run("ngram_ids", a);
    };

    /* --- the head blocks: 0..7 are the 2-gram, 8..15 the 3-gram ------------------------------ */
    T base(RAD_I32, {1, kHeads}), t2_changed(RAD_I32, {1, kHeads}),
      t1_changed(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { 700, 22 }, base));           /* t0=11 t1=22 t2=700 */
    CHECK_OK(run4({ 11 }, { 701, 22 }, t2_changed));     /* only t2 differs */
    CHECK_OK(run4({ 11 }, { 700, 23 }, t1_changed));     /* only t1 differs */
    for (int64_t h = 0; h < 8; ++h) {
        CHECK_EQ(base.iat(h), t2_changed.iat(h));        /* a 2-gram head cannot see t2 */
        CHECK(base.iat(h) != t1_changed.iat(h));
        CHECK(in_slice(base, 0, h));
    }
    for (int64_t h = 8; h < 16; ++h) {
        CHECK(base.iat(h) != t2_changed.iat(h));         /* a 3-gram head must */
        CHECK(in_slice(base, 0, h));
    }

    /* --- the EOS rule ------------------------------------------------------------------------ */
    T all_eos(RAD_I32, {1, kHeads}), eos_at_p1(RAD_I32, {1, kHeads}),
      eos_at_p2(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { kEos, kEos }, all_eos));
    CHECK_OK(run4({ 11 }, { 700, kEos }, eos_at_p1));    /* t1 is the eos: t2 must follow it */
    CHECK_OK(run4({ 11 }, { kEos, 22 }, eos_at_p2));     /* t2 is the eos: t1 survives */
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(all_eos.iat(h), eos_at_p1.iat(h));
    /* ...and an eos at p-2 changes nothing, because the value it substitutes is the value there. */
    T eos_p2_ref(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { kEos, 22 }, eos_p2_ref));
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(eos_at_p2.iat(h), eos_p2_ref.iat(h));

    /* --- `prev` vs the batch: a row deep enough in the step must not read `prev` at all ------- */
    T deep_a(RAD_I32, {3, kHeads}), deep_b(RAD_I32, {3, kHeads});
    CHECK_OK(run4({ 5, 6, 11 }, { 900, 901 }, deep_a));
    CHECK_OK(run4({ 5, 6, 11 }, { 902, 903 }, deep_b));
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(deep_a.iat(2 * kHeads + h), deep_b.iat(2 * kHeads + h));
    CHECK(deep_a.iat(0) != deep_b.iat(0));               /* but row 0 does */
    /* Row 2 of [5, 6, 11] sees exactly the history row 0 of [11] sees with prev = [5, 6]. */
    T from_prev(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { 5, 6 }, from_prev));
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(deep_a.iat(2 * kHeads + h), from_prev.iat(h));

    /* --- two sequences in one batch do not see each other, and each reads its OWN slot --------- */
    {
        T tt(RAD_I32, {4}); tt.n({ 5, 6, 11, 11 });      /* seq0 = [5,6,11], seq1 = [11] */
        T ts(RAD_I32, {2, 2}); ts.n({ 900, 901, 5, 6 });
        T tc(RAD_I32, {3}); tc.n({ 0, 3, 4 });
        T ci(RAD_I32, {2}); ci.n({ 0, 1 });
        T hi(RAD_I32, {2}); hi.n({ 1, 1 });
        T ids(RAD_I32, {4, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids)
             .i("M", 4).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        /* seq1's only row has a window of [5, 6] and token 11 -- the same history as `from_prev`,
         * even though the tokens 5 and 6 are also sitting in the batch two rows earlier. */
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(3 * kHeads + h), from_prev.iat(h));
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(2 * kHeads + h), from_prev.iat(h));
        /* AND EACH SLOT WAS WRITTEN WITH ITS OWN TAIL, at offset zero: seq0 ran [5,6,11] so its
         * window becomes (6, 11); seq1 ran [11] against (5, 6) so its window becomes (6, 11) too,
         * by a different route -- one value carried forward from the old window, one from `tok`. */
        CHECK_EQ(ts.iat(0), 6);   CHECK_EQ(ts.iat(1), 11);
        CHECK_EQ(ts.iat(2), 6);   CHECK_EQ(ts.iat(3), 11);
    }

    /* --- A COLD SLOT IS EOS, NOT ZERO, and not what the slot last held ------------------------ */
    {
        T tt(RAD_I32, {1}); tt.n({ 11 });
        T ts(RAD_I32, {1, 2}); ts.n({ 700, 22 });        /* a recycled slot's leftovers */
        T tc(RAD_I32, {2}); tc.n({ 0, 1 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 0 });                 /* COLD */
        T ids(RAD_I32, {1, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids)
             .i("M", 1).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        /* Identical to a WARM slot whose window is already all EOS -- which is the whole content
         * of "cold means eos". Zero would be a real token and would hash somewhere else. */
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(h), all_eos.iat(h));
    }

    /* --- `num_accepted` shifts the READ inside a deeper window -------------------------------- */
    {
        T tt(RAD_I32, {1}); tt.n({ 11 });
        T ts(RAD_I32, {1, 4}); ts.n({ 900, 901, 5, 6 }); /* two slots of speculative slack */
        T tc(RAD_I32, {2}); tc.n({ 0, 1 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 1 });
        T na(RAD_I32, {1}); na.n({ 3 });                 /* off = 2, so the window is (5, 6) */
        T ids(RAD_I32, {1, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).t(na).t(ids)
             .i("M", 1).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(h), from_prev.iat(h));
        /* ...and the write still lands at offset ZERO, not at the offset it read from. */
        CHECK_EQ(ts.iat(0), 6);   CHECK_EQ(ts.iat(1), 11);
    }

    /* --- `ahead_ids`: the prompt that follows, hashed as the step that runs it will ------------
     *
     * Step one is two sequences, [5, 6] and [11, 42], and `tok` carries three more tokens of the
     * LAST one, [7, 8, 9]. Their ids must be exactly the ids step two computes when it runs
     * [7, 8, 9] against the window step one leaves -- and that window, and the other sequence,
     * must be what they would have been with no `ahead_ids` at all. */
    {
        T tt(RAD_I32, {7}); tt.n({ 5, 6, 11, 42, 7, 8, 9 });
        T ts(RAD_I32, {2, 2}); ts.n({ 900, 901, 902, 903 });
        T tc(RAD_I32, {3}); tc.n({ 0, 2, 4 });
        T ci(RAD_I32, {2}); ci.n({ 0, 1 });
        T hi(RAD_I32, {2}); hi.n({ 1, 1 });
        T ids(RAD_I32, {4, kHeads}), ahead(RAD_I32, {3, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids).t(ahead)
             .i("M", 4).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        CHECK_EQ(ts.iat(0), 5);   CHECK_EQ(ts.iat(1), 6);    /* seq0's window: its own tail */
        CHECK_EQ(ts.iat(2), 11);  CHECK_EQ(ts.iat(3), 42);   /* seq1's: its own, not the ahead */

        /* The same step without `ahead_ids`: the same ids for the step's own rows. */
        T tt0(RAD_I32, {4}); tt0.n({ 5, 6, 11, 42 });
        T ts0(RAD_I32, {2, 2}); ts0.n({ 900, 901, 902, 903 });
        T ids0(RAD_I32, {4, kHeads});
        A a0; a0.t(tt0).t(ts0).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids0)
              .i("M", 4).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a0));
        for (int64_t i = 0; i < 4 * kHeads; ++i) CHECK_EQ(ids.iat(i), ids0.iat(i));

        /* Step two: seq1 runs [7, 8, 9] against the window step one left. */
        T tt2(RAD_I32, {3}); tt2.n({ 7, 8, 9 });
        T ts2(RAD_I32, {1, 2}); ts2.n({ ts.iat(2), ts.iat(3) });
        T tc2(RAD_I32, {2}); tc2.n({ 0, 3 });
        T ci2(RAD_I32, {1}); ci2.n({ 0 });
        T hi2(RAD_I32, {1}); hi2.n({ 1 });
        T ids2(RAD_I32, {3, kHeads});
        A a2; a2.t(tt2).t(ts2).t(tc2).t(tm).t(tv).t(to).t(ci2).t(hi2).tnull().t(ids2)
              .i("M", 3).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a2));
        for (int64_t i = 0; i < 3 * kHeads; ++i) CHECK_EQ(ahead.iat(i), ids2.iat(i));
    }
}

/* ============================================================ the PLE windows across a verify
 *
 * THE DEFINITION THE TWO TESTS BELOW CHECK AGAINST: a step's outputs are a function of the
 * COMMITTED history and of its own tokens, and of nothing else. Whatever a run of prefill chunks
 * and speculative verifies left in a rolling window, row t of any step must therefore equal the
 * last row of ONE cold prefill over (every committed token ++ that step's first t + 1 tokens). A
 * verify of n tokens that kept k of them commits its first k, and the next step's `num_accepted`
 * is k; a prefill chunk commits all of its tokens and the next step's `num_accepted` is 1, which
 * is what the scheduler hands both ops.
 *
 * The window itself is checked too, without going through the op a second time: after a verify
 * of n, the entries at offset k - 1 must be the last committed values for EVERY k the next step
 * could present, and after a prefill chunk the entries at offset zero must be.
 *
 * NO SINGLE LAYOUT PASSES THIS, which is why both kinds of step are in the chain. The tail of the
 * step at offset zero gives a next step that kept one token a history ending at the last REJECTED
 * draft; the shifted layout alone gives a short prefill chunk's next step a history ending at the
 * chunk's first token. The chains below include a two-token prefill chunk (shorter than a verify
 * window) and verifies accepting every count from one to all -- and one check that the history of
 * the rejected drafts really does give a different answer, without which agreement proves
 * nothing. */
namespace {

/* The n-gram op's constants: the real checkpoint's multipliers and a slice of its vocabulary. */
struct NgK {
    static constexpr int64_t kEos = 248044, kHeads = 16, kNgram = 3, kCtx = 2;
    T tm{RAD_I64, {3}}, tv{RAD_I64, {16}}, to{RAD_I64, {16}};
    NgK() {
        tm.n({ 23703573157769LL, 20109073645365LL, 8052911324071LL });
        std::vector<int64_t> v, o;
        int64_t acc = 0;
        for (int i = 0; i < 16; ++i) {
            v.push_back(20000003 + 10 * i);
            o.push_back(acc);
            acc += v.back();
        }
        tv.n(v);
        to.n(o);
    }
};

/* One launch over several sequences, sequence i on state slot `slots[i]`. `nacc` null is the
 * operand ABSENT -- the prefill form -- and not a row of ones. */
int ng_launch(const NgK& K, const std::vector<std::vector<int64_t>>& seqs, T& state,
              const std::vector<int64_t>& slots, const std::vector<int64_t>& hini,
              const std::vector<int64_t>* nacc, T& ids) {
    std::vector<int64_t> tok, cu{ 0 };
    for (const auto& s : seqs) {
        tok.insert(tok.end(), s.begin(), s.end());
        cu.push_back((int64_t)tok.size());
    }
    const int64_t M = (int64_t)tok.size(), S = (int64_t)seqs.size();
    T tt(RAD_I32, {M}); tt.n(tok);
    T tc(RAD_I32, {S + 1}); tc.n(cu);
    T ci(RAD_I32, {S}); ci.n(slots);
    T hi(RAD_I32, {S}); hi.n(hini);
    T na(RAD_I32, {S});
    if (nacc) na.n(*nacc);
    A a; a.t(tt).t(state).t(tc).t(K.tm).t(K.tv).t(K.to).t(ci).t(hi);
    if (nacc) a.t(na); else a.tnull();
    a.t(ids).i("M", M).i("heads", NgK::kHeads).i("ngram", NgK::kNgram).i("eos", NgK::kEos);
    return run("ngram_ids", a);
}

/* The definition: the last row of one cold prefill over `hist`. */
std::vector<int64_t> ng_fresh(const NgK& K, const std::vector<int64_t>& hist) {
    const int64_t L = (int64_t)hist.size();
    T st(RAD_I32, {1, NgK::kCtx});
    T ids(RAD_I32, {L, NgK::kHeads});
    std::vector<int64_t> row((size_t)NgK::kHeads, -1);
    if (ng_launch(K, { hist }, st, { 0 }, { 0 }, nullptr, ids) != RAD_OK) return row;
    for (int64_t h = 0; h < NgK::kHeads; ++h) row[(size_t)h] = ids.iat((L - 1) * NgK::kHeads + h);
    return row;
}

std::vector<int64_t> ng_row(const T& ids, int64_t r) {
    std::vector<int64_t> row((size_t)NgK::kHeads);
    for (int64_t h = 0; h < NgK::kHeads; ++h) row[(size_t)h] = ids.iat(r * NgK::kHeads + h);
    return row;
}

/* The last `n` values of `v`, cold-padded on the left with `cold`. */
template <class V>
std::vector<V> last_n(const std::vector<V>& v, int64_t n, V cold) {
    std::vector<V> out((size_t)n, cold);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t src = (int64_t)v.size() - n + i;
        if (src >= 0) out[(size_t)i] = v[(size_t)src];
    }
    return out;
}

/* A step of a chain: `n` tokens, `k` of them kept. A prefill keeps all of them. */
struct ChainStep { bool prefill; int n; int k; };
const ChainStep kChain[] = {
    { true, 3, 3 }, { true, 2, 2 },                    /* a cold chunk, then a SHORT warm one */
    { false, 4, 1 }, { false, 4, 4 }, { false, 4, 2 },  /* verifies of 1 + n_spec = 4 */
    { false, 1, 1 },                                    /* a decode that drafted nothing */
    { false, 4, 3 }, { false, 3, 1 }, { false, 4, 2 }, { false, 2, 2 },
};
constexpr int64_t kNspec = 3;

}  /* namespace */

TEST(ngram_window_follows_the_committed_history) {
    const NgK K;
    const int64_t W = NgK::kCtx + kNspec;       /* what RAD_KV_CONV sizes: ctx + n_spec */
    T st(RAD_I32, {1, W});
    st.n(std::vector<int64_t>((size_t)W, 777)); /* a recycled slot's leftovers */

    uint64_t lcg = 0x5EEDull;
    auto draw = [&]() {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        return (int64_t)((lcg >> 33) % 150000) + 1;
    };

    std::vector<int64_t> committed, wrong;   /* `wrong` keeps every draft, rejected or not */
    int64_t nacc_next = 1;
    bool cold = true, after_partial = false;
    int step_no = 0;
    for (const ChainStep& cs : kChain) {
        std::vector<int64_t> tok;
        for (int i = 0; i < cs.n; ++i) tok.push_back(draw());
        if (step_no == 6) tok[1] = NgK::kEos;     /* the EOS rule inside a verify, once */

        T ids(RAD_I32, {cs.n, NgK::kHeads});
        const std::vector<int64_t> na{ nacc_next };
        CHECK_OK(ng_launch(K, { tok }, st, { 0 }, { cold ? 0 : 1 }, cs.prefill ? nullptr : &na,
                           ids));

        /* Every row against the definition. */
        for (int t = 0; t < cs.n; ++t) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + t + 1);
            CHECK(ng_row(ids, t) == ng_fresh(K, h));
        }
        /* The sensitivity check: the history that keeps the rejected drafts is a DIFFERENT
         * answer, so agreement above is a statement about which history was read. */
        if (after_partial) {
            std::vector<int64_t> h = wrong;
            h.push_back(tok[0]);
            CHECK(ng_row(ids, 0) != ng_fresh(K, h));
        }

        /* The window left behind, at every offset the next step could read. */
        const int kmax = cs.prefill ? 1 : cs.n;
        for (int kk = 1; kk <= kmax; ++kk) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + (cs.prefill ? cs.n : kk));
            const std::vector<int64_t> want = last_n<int64_t>(h, NgK::kCtx, NgK::kEos);
            for (int64_t i = 0; i < NgK::kCtx; ++i) CHECK_EQ(st.iat(kk - 1 + i), want[(size_t)i]);
        }

        committed.insert(committed.end(), tok.begin(), tok.begin() + cs.k);
        wrong.insert(wrong.end(), tok.begin(), tok.end());
        after_partial = !cs.prefill && cs.k < cs.n;
        if (!after_partial) wrong = committed;
        nacc_next = cs.prefill ? 1 : cs.k;
        cold = false;
        ++step_no;
    }

    /* AND ONE CASE WRITTEN OUT: history {1, 2, 3}, a verify of {4, 100, 200, 300} that
     * kept only the 4. The next step's two ids in front must be (3, 4) -- not (200, 300), which
     * is the tail of drafts that were rejected. */
    {
        T s2(RAD_I32, {1, W});
        T i1(RAD_I32, {3, NgK::kHeads}), i2(RAD_I32, {4, NgK::kHeads});
        CHECK_OK(ng_launch(K, { { 1, 2, 3 } }, s2, { 0 }, { 0 }, nullptr, i1));
        const std::vector<int64_t> one{ 1 };
        CHECK_OK(ng_launch(K, { { 4, 100, 200, 300 } }, s2, { 0 }, { 1 }, &one, i2));
        CHECK_EQ(s2.iat(0), 3LL);
        CHECK_EQ(s2.iat(1), 4LL);
        T i3(RAD_I32, {1, NgK::kHeads});
        CHECK_OK(ng_launch(K, { { 9 } }, s2, { 0 }, { 1 }, &one, i3));
        CHECK(ng_row(i3, 0) == ng_fresh(K, { 1, 2, 3, 4, 9 }));
    }

    /* TWO SEQUENCES IN ONE LAUNCH, on swapped slots and with different acceptances: each reads
     * its own window at its own offset. */
    {
        T s2(RAD_I32, {2, W});
        const std::vector<int64_t> A0{ 11, 12, 13 }, B0{ 21, 22, 23, 24 };
        T i1(RAD_I32, {7, NgK::kHeads});
        CHECK_OK(ng_launch(K, { A0, B0 }, s2, { 1, 0 }, { 0, 0 }, nullptr, i1));
        const std::vector<int64_t> A1{ 31, 32, 33, 34 }, B1{ 41, 42, 43 };
        const std::vector<int64_t> ones{ 1, 1 };
        T i2(RAD_I32, {7, NgK::kHeads});
        CHECK_OK(ng_launch(K, { A1, B1 }, s2, { 1, 0 }, { 1, 1 }, &ones, i2));
        const std::vector<int64_t> kept{ 2, 3 };  /* A kept 31, 32; B kept 41, 42, 43 */
        T i3(RAD_I32, {2, NgK::kHeads});
        CHECK_OK(ng_launch(K, { { 50 }, { 60 } }, s2, { 1, 0 }, { 1, 1 }, &kept, i3));
        CHECK(ng_row(i3, 0) == ng_fresh(K, { 11, 12, 13, 31, 32, 50 }));
        CHECK(ng_row(i3, 1) == ng_fresh(K, { 21, 22, 23, 24, 41, 42, 43, 60 }));
    }

    /* A decode step whose window is too shallow for the step it is asked to leave behind is
     * refused, not truncated: ctx + n_spec slots hold a step of 1 + n_spec ids and no more. */
    {
        T s2(RAD_I32, {1, W});
        T ids(RAD_I32, {5, NgK::kHeads});
        const std::vector<int64_t> one{ 1 };
        CHECK_EQ(ng_launch(K, { { 1, 2, 3, 4, 5 } }, s2, { 0 }, { 1 }, &one, ids), RAD_E_SHAPE);
    }
}

namespace {

/* ple_conv at width 3 and dilation 2 -- history 4 -- over two channels, f32 throughout so the
 * comparison against the definition is exact rather than a tolerance. */
struct PcK {
    static constexpr int64_t kN = 2, kWid = 3, kDil = 2, kHist = (kWid - 1) * kDil;
    T w{RAD_F32, {kN, kWid}};
    PcK() { w.f({ 0.5f, -0.25f, 1.0f, 0.125f, 0.75f, -0.5f }); }
};

/* A token's input row: a function of the token, so a history of tokens is a history of rows. */
float pc_x(int64_t tok, int64_t c) { return (float)((tok * 37 + c * 11) % 101) * 0.0625f - 3.0f; }

int pc_launch(const PcK& K, const std::vector<std::vector<int64_t>>& seqs, T& state,
              const std::vector<int64_t>& slots, const std::vector<int64_t>& hini,
              const std::vector<int64_t>* nacc, T& y) {
    std::vector<int64_t> cu{ 0 };
    std::vector<float> xv;
    for (const auto& s : seqs) {
        for (int64_t t : s)
            for (int64_t c = 0; c < PcK::kN; ++c) xv.push_back(pc_x(t, c));
        cu.push_back(cu.back() + (int64_t)s.size());
    }
    const int64_t M = cu.back(), S = (int64_t)seqs.size();
    T x(RAD_F32, {M, PcK::kN}); x.f(xv);
    T tc(RAD_I32, {S + 1}); tc.n(cu);
    T ci(RAD_I32, {S}); ci.n(slots);
    T hi(RAD_I32, {S}); hi.n(hini);
    T na(RAD_I32, {S});
    if (nacc) na.n(*nacc);
    A a; a.t(x).t(K.w).t(state).t(tc).tnull().t(ci).t(hi);
    if (nacc) a.t(na); else a.tnull();
    a.t(y).i("M", M).i("n", PcK::kN).i("width", PcK::kWid).i("dilation", PcK::kDil)
     .s("dtype", "f32");
    return run("ple_conv", a);
}

std::vector<float> pc_fresh(const PcK& K, const std::vector<int64_t>& hist) {
    const int64_t L = (int64_t)hist.size();
    T st(RAD_F32, {1, PcK::kN, PcK::kHist});
    T y(RAD_F32, {L, PcK::kN});
    std::vector<float> row((size_t)PcK::kN, NAN);
    if (pc_launch(K, { hist }, st, { 0 }, { 0 }, nullptr, y) != RAD_OK) return row;
    for (int64_t c = 0; c < PcK::kN; ++c) row[(size_t)c] = y.at((L - 1) * PcK::kN + c);
    return row;
}

std::vector<float> pc_row(const T& y, int64_t r) {
    std::vector<float> row((size_t)PcK::kN);
    for (int64_t c = 0; c < PcK::kN; ++c) row[(size_t)c] = y.at(r * PcK::kN + c);
    return row;
}

}  /* namespace */

TEST(ple_conv_window_follows_the_committed_history) {
    const PcK K;
    const int64_t W = PcK::kHist + kNspec;      /* (width-1)*dilation + n_spec */
    T st(RAD_F32, {1, PcK::kN, W});
    st.f(std::vector<float>((size_t)(PcK::kN * W), 9.0f));

    uint64_t lcg = 0xC0FFEEull;
    auto draw = [&]() {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        return (int64_t)((lcg >> 33) % 1000);
    };

    std::vector<int64_t> committed, wrong;
    int64_t nacc_next = 1;
    bool cold = true, after_partial = false;
    for (const ChainStep& cs : kChain) {
        std::vector<int64_t> tok;
        for (int i = 0; i < cs.n; ++i) tok.push_back(draw());

        T y(RAD_F32, {cs.n, PcK::kN});
        const std::vector<int64_t> na{ nacc_next };
        CHECK_OK(pc_launch(K, { tok }, st, { 0 }, { cold ? 0 : 1 }, cs.prefill ? nullptr : &na, y));

        for (int t = 0; t < cs.n; ++t) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + t + 1);
            CHECK(pc_row(y, t) == pc_fresh(K, h));
        }
        if (after_partial) {
            std::vector<int64_t> h = wrong;
            h.push_back(tok[0]);
            CHECK(pc_row(y, 0) != pc_fresh(K, h));
        }

        const int kmax = cs.prefill ? 1 : cs.n;
        for (int kk = 1; kk <= kmax; ++kk) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + (cs.prefill ? cs.n : kk));
            const std::vector<int64_t> want = last_n<int64_t>(h, PcK::kHist, -1);
            for (int64_t c = 0; c < PcK::kN; ++c)
                for (int64_t i = 0; i < PcK::kHist; ++i) {
                    const int64_t tk = want[(size_t)i];
                    CHECK_EQ(st.at(c * W + kk - 1 + i), tk < 0 ? 0.0f : pc_x(tk, c));
                }
        }

        committed.insert(committed.end(), tok.begin(), tok.begin() + cs.k);
        wrong.insert(wrong.end(), tok.begin(), tok.end());
        after_partial = !cs.prefill && cs.k < cs.n;
        if (!after_partial) wrong = committed;
        nacc_next = cs.prefill ? 1 : cs.k;
        cold = false;
    }

    /* Two sequences in one launch, swapped slots, different acceptances. */
    {
        T s2(RAD_F32, {2, PcK::kN, W});
        const std::vector<int64_t> A0{ 11, 12, 13, 14, 15 }, B0{ 21, 22 };
        T y1(RAD_F32, {7, PcK::kN});
        CHECK_OK(pc_launch(K, { A0, B0 }, s2, { 1, 0 }, { 0, 0 }, nullptr, y1));
        const std::vector<int64_t> A1{ 31, 32, 33, 34 }, B1{ 41, 42, 43, 44 };
        const std::vector<int64_t> ones{ 1, 1 };
        T y2(RAD_F32, {8, PcK::kN});
        CHECK_OK(pc_launch(K, { A1, B1 }, s2, { 1, 0 }, { 1, 1 }, &ones, y2));
        const std::vector<int64_t> kept{ 1, 3 };
        T y3(RAD_F32, {2, PcK::kN});
        CHECK_OK(pc_launch(K, { { 50 }, { 60 } }, s2, { 1, 0 }, { 1, 1 }, &kept, y3));
        CHECK(pc_row(y3, 0) == pc_fresh(K, { 11, 12, 13, 14, 15, 31, 50 }));
        CHECK(pc_row(y3, 1) == pc_fresh(K, { 21, 22, 41, 42, 43, 60 }));
    }

    /* Too shallow for the window a decode step of this length leaves: refused, not truncated. */
    {
        T s2(RAD_F32, {1, PcK::kN, W});
        T y(RAD_F32, {5, PcK::kN});
        const std::vector<int64_t> one{ 1 };
        CHECK_EQ(pc_launch(K, { { 1, 2, 3, 4, 5 } }, s2, { 0 }, { 1 }, &one, y), RAD_E_SHAPE);
    }
}

TEST(op_cast) {
    T x(RAD_F32, {4}); x.f({2.5f, -1.5f, 200.0f, 1.4f});
    T y(RAD_I8, {4});
    A a; a.t(x).t(y).i("M", 1).i("n", 4).s("from", "f32").s("to", "i8");
    CHECK_OK(run("cast", a));
    CHECK_EQ(y.iat(0), 2LL);      /* ties to even */
    CHECK_EQ(y.iat(1), -2LL);
    CHECK_EQ(y.iat(2), 127LL);    /* saturates rather than wrapping */
    CHECK_EQ(y.iat(3), 1LL);

    /* Integer to integer must not pass through f32: 2^25 + 1 is not representable there. */
    T big(RAD_I64, {1}); big.n({33554433});
    T out(RAD_I32, {1});
    A b; b.t(big).t(out).i("M", 1).i("n", 1).s("from", "i64").s("to", "i32");
    CHECK_OK(run("cast", b));
    CHECK_EQ(out.iat(0), 33554433LL);
}

TEST(op_softmax) {
    T x(RAD_F32, {1, 2}); x.f({0.0f, 0.69314718f});   /* ln 2 */
    T y(RAD_F32, {1, 2});
    A a; a.t(x).t(y).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("softmax", a));
    CHECK_NEAR(y.at(0), 1.0f / 3.0f, 1e-6);
    CHECK_NEAR(y.at(1), 2.0f / 3.0f, 1e-6);
}

/* ================================================================== quantisation */
TEST(op_quant_act_i8) {
    T x(RAD_F32, {1, 4}); x.f({1, -2, 0.5f, 2});
    T q(RAD_I8, {1, 4}), s(RAD_F32, {1, 1});
    A a; a.t(x).t(q).t(s).i("M", 1).i("n", 4).i("group", 4);
    CHECK_OK(run("quant_act_i8", a));
    /* amax 2, so q = round(x * 63.5): 63.5 -> 64 (ties to even), -127, 31.75 -> 32, 127. */
    CHECK_EQ(q.iat(0), 64LL);
    CHECK_EQ(q.iat(1), -127LL);
    CHECK_EQ(q.iat(2), 32LL);
    CHECK_EQ(q.iat(3), 127LL);
    CHECK_NEAR(s.at(0), 2.0f / 127.0f, 1e-9);

    /* Two groups of two: each gets its own scale. */
    T q2(RAD_I8, {1, 4}), s2(RAD_F32, {1, 2});
    A b; b.t(x).t(q2).t(s2).i("M", 1).i("n", 4).i("group", 2);
    CHECK_OK(run("quant_act_i8", b));
    CHECK_NEAR(s2.at(0), 2.0f / 127.0f, 1e-9);
    CHECK_NEAR(s2.at(1), 2.0f / 127.0f, 1e-9);
    CHECK_EQ(q2.iat(2), 32LL);
}

TEST(op_had_quant_act_i8) {
    /* One rotation block of two: the butterfly is [a+b, a-b]. */
    T x(RAD_F32, {1, 2}); x.f({1, 3});
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1});
    A a; a.t(x).t(q).t(s).i("M", 1).i("n", 2).i("group", 2);
    CHECK_OK(run("had_quant_act_i8", a));
    /* rotated [4, -2], amax 4, q = round(v * 127/4) = [127, -63.5 -> -64] */
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), -64LL);
    /* the scale carries the 1/sqrt(group) the rotation left out */
    CHECK_NEAR(s.at(0), 4.0f / 127.0f / 1.41421356f, 1e-9);
}

TEST(op_rmsnorm_had_quant_i8) {
    /* Operand 1 is the optional `residual`; absent here, exercised below. */
    T x(RAD_F32, {1, 2}); x.f({1, 1});
    T w(RAD_F32, {2});    w.f({1, 1});
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1});
    A a; a.t(x).tnull().t(w).t(q).t(s).i("M", 1).i("n", 2).d("eps", 0).i("group", 2);
    CHECK_OK(run("rmsnorm_had_quant_i8", a));
    /* rms 1, normed [1,1], rotated [2,0], amax 2 */
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), 0LL);
    CHECK_NEAR(s.at(0), 2.0f / 127.0f / 1.41421356f, 1e-9);
}

/* The fused_add_rms_norm form. The residual add happens FIRST and is written back -- that is what
 * makes `residual` inout and what the next layer's shortcut reads -- and `out_bf16` carries the
 * post-norm, PRE-rotation tensor for a layer that also has a bf16 consumer of it. */
TEST(op_rmsnorm_had_quant_i8_residual) {
    T x(RAD_F32, {1, 2});   x.f({1, 1});
    T res(RAD_F32, {1, 2}); res.f({1, 1});
    T w(RAD_F32, {2});      w.f({1, 1});
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1}), ob(RAD_F32, {1, 2});
    A a; a.t(x).t(res).t(w).t(q).t(s).t(ob).i("M", 1).i("n", 2).d("eps", 0).i("group", 2);
    CHECK_OK(run("rmsnorm_had_quant_i8", a));
    /* residual becomes [2, 2]; rms 2; normed [1, 1]; rotated [2, 0]. */
    CHECK_EQ(res.at(0), 2.0f);
    CHECK_EQ(res.at(1), 2.0f);
    CHECK_NEAR(ob.at(0), 1.0f, 1e-6);
    CHECK_NEAR(ob.at(1), 1.0f, 1e-6);
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), 0LL);
    CHECK_NEAR(s.at(0), 2.0f / 127.0f / 1.41421356f, 1e-9);
}

TEST(op_gated_had_quant_i8) {
    /* `b` ABSENT IS THE PACKED FORM: `a` is [M, 2n], gate first half, up second. */
    T gu(RAD_F32, {1, 4}); gu.f({0, 0, 5, 7});     /* gate [0,0], up [5,7] */
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1});
    A a; a.t(gu).tnull().t(q).t(s).i("M", 1).i("n", 2).i("group", 2).s("act", "sigmoid");
    CHECK_OK(run("gated_had_quant_i8", a));
    /* sigmoid(0)*[5,7] = [2.5, 3.5]; rotated [6, -1]; amax 6 */
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), (long long)std::nearbyintf(-1.0f * 127.0f / 6.0f));
    CHECK_NEAR(s.at(0), 6.0f / 127.0f / 1.41421356f, 1e-9);

    /* The split form: the same numbers out of two operands, and `mode` as libr4d's integer
     * spelling of the same activation. The two readings have to agree or the schema's union of
     * them means two different ops. */
    T ga(RAD_F32, {1, 2}); ga.f({0, 0});
    T up(RAD_F32, {1, 2}); up.f({5, 7});
    T q2(RAD_I8, {1, 2}), s2(RAD_F32, {1});
    A b; b.t(ga).t(up).t(q2).t(s2).i("M", 1).i("n", 2).i("group", 2).i("mode", 1);
    CHECK_OK(run("gated_had_quant_i8", b));
    CHECK_EQ(q2.iat(0), 127LL);
    CHECK_EQ(q2.iat(1), (long long)std::nearbyintf(-1.0f * 127.0f / 6.0f));
    CHECK_NEAR(s2.at(0), 6.0f / 127.0f / 1.41421356f, 1e-9);
}

TEST(op_dequant) {
    T q(RAD_I8, {1, 4}); q.n({1, 2, 3, 4});
    T s(RAD_F32, {1, 2}); s.f({0.5f, 2.0f});
    T y(RAD_F32, {1, 4});
    A a; a.t(q).t(s).t(y).i("M", 1).i("n", 4).i("group", 2).s("dtype", "i8");
    CHECK_OK(run("dequant", a));
    CHECK_EQ(y.at(0), 0.5f);
    CHECK_EQ(y.at(1), 1.0f);
    CHECK_EQ(y.at(2), 6.0f);
    CHECK_EQ(y.at(3), 8.0f);
}

/* ================================================================== gemm */
TEST(op_gemm_nt) {
    /* 2x3 times a transposed 4x3, checkable by hand. */
    T a_(RAD_F32, {2, 3}); a_.f({1, 2, 3, 4, 5, 6});
    T b_(RAD_F32, {4, 3}); b_.f({1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1});
    T y(RAD_F32, {2, 4});
    A a; a.t(a_).t(b_).t(y).i("M", 2).i("N", 4).i("K", 3).s("dtype", "f32");
    CHECK_OK(run("gemm_nt", a));
    CHECK_EQ(y.at(0), 1.0f); CHECK_EQ(y.at(1), 2.0f);
    CHECK_EQ(y.at(2), 3.0f); CHECK_EQ(y.at(3), 6.0f);
    CHECK_EQ(y.at(4), 4.0f); CHECK_EQ(y.at(5), 5.0f);
    CHECK_EQ(y.at(6), 6.0f); CHECK_EQ(y.at(7), 15.0f);

    T bias(RAD_F32, {4}); bias.f({1, 2, 3, 4});
    T y2(RAD_F32, {2, 4});
    A b; b.t(a_).t(b_).t(bias).tnull().t(y2).i("M", 2).i("N", 4).i("K", 3).s("dtype", "f32");
    CHECK_OK(run("gemm_nt_bias", b));
    CHECK_EQ(y2.at(0), 2.0f);
    CHECK_EQ(y2.at(7), 19.0f);

    /* The same GEMM in bf16: every value here is exact in bf16, so the answer is exact too and
     * the bf16 path is checked by equality rather than by a tolerance. */
    T ab(RAD_BF16, {2, 3}); ab.f({1, 2, 3, 4, 5, 6});
    T bb(RAD_BF16, {4, 3}); bb.f({1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1});
    T yb(RAD_BF16, {2, 4});
    A c; c.t(ab).t(bb).t(yb).i("M", 2).i("N", 4).i("K", 3).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", c));
    CHECK_EQ(yb.at(3), 6.0f);
    CHECK_EQ(yb.at(7), 15.0f);
}

TEST(op_gemm_nt_q) {
    /* int8 x int8 with a per-row activation scale and a per-row weight scale. */
    T a_(RAD_I8, {2, 2}); a_.n({1, 2, 3, 4});
    T as(RAD_F32, {2});   as.f({0.5f, 1.0f});
    T b_(RAD_I8, {2, 2}); b_.n({1, 1, 2, 0});
    T bs(RAD_F32, {2});   bs.f({2.0f, 1.0f});
    T y(RAD_F32, {2, 2});
    A a; a.t(a_).t(as).t(b_).t(bs).t(y)
          .i("M", 2).i("N", 2).i("K", 2).i("group", 2).s("dtype", "w8a8");
    CHECK_OK(run("gemm_nt_q", a));
    CHECK_EQ(y.at(0), 3.0f);    /* (1*1 + 2*1) * 0.5 * 2 */
    CHECK_EQ(y.at(1), 1.0f);    /* (1*2 + 2*0) * 0.5 * 1 */
    CHECK_EQ(y.at(2), 14.0f);   /* (3*1 + 4*1) * 1.0 * 2 */
    CHECK_EQ(y.at(3), 6.0f);    /* (3*2 + 4*0) * 1.0 * 1 */

    /* A 4-bit weight, in the packed layout ref_common.h documents: element 2i in the low nibble,
     * value = (code ^ 8) - 8. Row 0 is byte 0x21 -> codes 1, 2 -> values 1, 2; row 1 is 0x0f ->
     * codes 0xf, 0x0 -> values -1, 0. */
    T bw(RAD_I4, {2, 2}); bw.raw({0x21, 0x0f});
    T bws(RAD_F32, {2});  bws.f({1.0f, 1.0f});
    T y2(RAD_F32, {2, 2});
    A b; b.t(a_).t(as).t(bw).t(bws).t(y2)
          .i("M", 2).i("N", 2).i("K", 2).i("group", 2).s("dtype", "w4a8");
    CHECK_OK(run("gemm_nt_q", b));
    CHECK_EQ(y2.at(0), 2.5f);   /* (1*1 + 2*2) * 0.5 */
    CHECK_EQ(y2.at(1), -0.5f);  /* (1*-1 + 2*0) * 0.5 */
    CHECK_EQ(y2.at(2), 11.0f);  /* (3*1 + 4*2) * 1.0 */
    CHECK_EQ(y2.at(3), -3.0f);

    /* mxfp4: e2m1 elements with an e8m0 block exponent. Codes 0x2 (+1) and 0x4 (+2); the byte is
     * 0x42 and the shared exponent 128 doubles both. */
    T bm(RAD_FP4E2M1, {1, 2}); bm.raw({0x42});
    T bms(RAD_U8, {1});      bms.n({128});
    T y3(RAD_F32, {2, 1});
    A c; c.t(a_).t(as).t(bm).t(bms).t(y3)
          .i("M", 2).i("N", 1).i("K", 2).i("group", 2).s("dtype", "mxfp4a8");
    CHECK_OK(run("gemm_nt_q", c));
    CHECK_EQ(y3.at(0), (1 * 2.0f + 2 * 4.0f) * 0.5f);
    CHECK_EQ(y3.at(1), (3 * 2.0f + 4 * 4.0f) * 1.0f);
}

/* A BLOCK GRID WHOSE LAST BLOCK IS PARTIAL, on both axes. K = 300 at 128 is three scales over 128,
 * 128 and 44 columns; N = 129 is two row blocks, the second of one row, and N = 300 is three. The
 * expected value of every output is written out from that definition: with every code 1, output
 * n is the sum over column groups of the group's width times scale[n / 128][g]. The block is NOT
 * the quotient of the extent by the block count (100 at 300) -- and a stream with fewer rows than
 * the operand that does not divide it is still a block grid, not a per-row stream to read past
 * its end. Checked with `group` given and without it, since the loop reads the grid off the
 * stream either way. */
TEST(op_gemm_nt_q_partial_block_grid) {
    const int64_t K = 300, G = 128, gk = 3;
    const int64_t width[3] = { 128, 128, 44 };
    for (int64_t N : { 129, 300 }) {
        const int64_t gn = (N + G - 1) / G;
        T a_(RAD_F32, {1, K}); a_.f(std::vector<float>((size_t)K, 1.0f));
        T b_(RAD_F32, {N, K}); b_.f(std::vector<float>((size_t)(N * K), 1.0f));
        std::vector<float> sv;
        for (int64_t r = 0; r < gn; ++r)
            for (int64_t g = 0; g < gk; ++g) sv.push_back((float)(1 + 10 * r + g));
        T bs(RAD_F32, {gn, gk}); bs.f(sv);
        for (int with_group = 0; with_group < 2; ++with_group) {
            T y(RAD_F32, {1, N});
            A a; a.t(a_).tnull().t(b_).t(bs).t(y).i("M", 1).i("N", N).i("K", K).s("dtype", "f32");
            if (with_group) a.i("group", G);
            CHECK_OK(run("gemm_nt_q", a));
            for (int64_t n = 0; n < N; ++n) {
                float want = 0.0f;
                for (int64_t g = 0; g < gk; ++g)
                    want += (float)width[g] * sv[(size_t)((n / G) * gk + g)];
                CHECK_EQ(y.at(n), want);
            }
        }
    }
}

/* ================================================================== the fp8 fused quantisers */

/* gemm_nt_q_gated against the pair it replaces, on one set of operands.
 *
 * Both libref and the libr4d row that serves this op claim to be "byte-identical to gemm_nt_q
 * followed by gated_quant_fp8". A fold that is a slightly different function from the pair is a
 * different model, so the claim is worth a test rather than a comment -- and the two paths meet
 * only at `quant_dot`, so agreement here is evidence and not a tautology: the pair's plane comes
 * through the TEMPLATED gemm_quant, with its loads dispatched at compile time, and the fused op
 * comes through the runtime-dispatched quant_dot.
 *
 * IT IS ALSO THE CASE THE PER-OP ORACLE CANNOT MAKE, which is why it is worth the operand sizes.
 * The container stores an fp8a8 weight in libr4d's WMMA fragment order, so handing libref the
 * engine's own weight bytes compares a permutation against a row-major reading and reports every
 * such kernel as failing from element zero. The operands drawn here are row-major on both sides.
 *
 * Not a literal expectation, for the reason the file header gives: what is under test is that two
 * routes through this plugin compute one function, and a third hand-computed number would only
 * restate one of them. The shapes are the smallest the fold's own rules admit -- a 128-column
 * scale group, a 128x128 weight-scale block -- so nothing here is a degenerate grid. */
TEST(op_gemm_nt_q_gated_is_the_pair) {
    const int64_t M = 3, K = 256, nout = 256, N = 2 * nout;
    const int64_t kb = K / 128, nb = N / 128, ng = nout / 128;

    /* One stream for both planes: a shared generator makes the weight and the activation
     * uncorrelated with each other, which a per-tensor pattern does not. */
    uint32_t rs = 0x9e3779b9u;
    auto rnd = [&]() {
        rs = rs * 1664525u + 1013904223u;
        return (float)((int32_t)((rs >> 13) & 0xffu) - 128) / 64.0f;
    };

    T a_(RAD_F8E4M3, {M, K});
    for (int64_t i = 0; i < M * K; ++i) st_dt(RAD_F8E4M3, a_.t.data, i, rnd() * 2.0f);
    T b_(RAD_F8E4M3, {N, K});
    for (int64_t i = 0; i < N * K; ++i) st_dt(RAD_F8E4M3, b_.t.data, i, rnd());
    T as(RAD_F32, {M, kb});
    for (int64_t i = 0; i < M * kb; ++i) st_dt(RAD_F32, as.t.data, i, 0.5f + 0.25f * (float)i);
    T bs(RAD_BF16, {nb, kb});
    for (int64_t i = 0; i < nb * kb; ++i) st_dt(RAD_BF16, bs.t.data, i, 0.75f + 0.125f * (float)i);

    /* The fold: no [M, N] plane is ever written. */
    T q1(RAD_F8E4M3, {M, nout}), s1(RAD_F32, {M, ng}), y1(RAD_BF16, {M, nout});
    A f; f.t(a_).t(as).t(b_).t(bs).t(q1).t(s1).t(y1)
          .i("M", M).i("N", N).i("K", K).i("group", 128)
          .s("dtype", "fp8a8").s("act", "silu");
    CHECK_OK(run("gemm_nt_q_gated", f));

    /* The pair, over the plane the fold does not write. `N` is the weight's rows on the GEMM and
     * `n` is half of it on the quantiser -- the one parameter a caller coming from gemm_nt_q gets
     * wrong here, and r4d_fuse.cpp's decomposition of this same op names it for the same reason. */
    T y(RAD_BF16, {M, N});
    A g; g.t(a_).t(as).t(b_).t(bs).t(y)
          .i("M", M).i("N", N).i("K", K).i("group", 128).s("dtype", "fp8a8");
    CHECK_OK(run("gemm_nt_q", g));

    T q2(RAD_F8E4M3, {M, nout}), s2(RAD_F32, {M, ng}), y2(RAD_BF16, {M, nout});
    A h; h.t(y).t(q2).t(s2).t(y2)
          .i("M", M).i("n", nout).i("group", 128).s("dtype", "bf16").s("act", "silu");
    CHECK_OK(run("gated_quant_fp8", h));

    /* TWO PLANES OF ZEROS WOULD AGREE, so the operands are asserted live before the comparison is
     * allowed to mean anything: an all-zero group takes scale 1 and codes 0 by the rule in
     * fp8_row, which is exactly what a silently-unwritten output looks like. */
    int64_t nz = 0;
    for (int64_t i = 0; i < M * nout; ++i) if (q1.mem[(size_t)i]) ++nz;
    REQUIRE(nz > M * nout / 2);

    int64_t bad_q = 0, bad_s = 0, bad_y = 0;
    for (int64_t i = 0; i < M * nout; ++i) {
        if (q1.mem[(size_t)i] != q2.mem[(size_t)i]) ++bad_q;
        if (y1.mem[(size_t)(2 * i)]     != y2.mem[(size_t)(2 * i)] ||
            y1.mem[(size_t)(2 * i + 1)] != y2.mem[(size_t)(2 * i + 1)]) ++bad_y;
    }
    for (int64_t i = 0; i < M * ng; ++i) if (s1.at(i) != s2.at(i)) ++bad_s;
    CHECK_EQ(bad_q, (int64_t)0);
    CHECK_EQ(bad_s, (int64_t)0);
    CHECK_EQ(bad_y, (int64_t)0);

    /* The gate half is the FIRST n weight rows and the up half the second, so swapping them is a
     * different answer. Without this the case passes on an implementation that reads either. */
    T q3(RAD_F8E4M3, {M, nout}), s3(RAD_F32, {M, ng});
    T yswap(RAD_BF16, {M, N});
    for (int64_t m = 0; m < M; ++m)
        for (int64_t i = 0; i < nout; ++i) {
            yswap.mem[(size_t)((m * N + i) * 2)]          = y.mem[(size_t)((m * N + nout + i) * 2)];
            yswap.mem[(size_t)((m * N + i) * 2 + 1)]      = y.mem[(size_t)((m * N + nout + i) * 2 + 1)];
            yswap.mem[(size_t)((m * N + nout + i) * 2)]   = y.mem[(size_t)((m * N + i) * 2)];
            yswap.mem[(size_t)((m * N + nout + i) * 2 + 1)] = y.mem[(size_t)((m * N + i) * 2 + 1)];
        }
    A k; k.t(yswap).t(q3).t(s3)
          .i("M", M).i("n", nout).i("group", 128).s("dtype", "bf16").s("act", "silu");
    CHECK_OK(run("gated_quant_fp8", k));
    int64_t differs = 0;
    for (int64_t i = 0; i < M * nout; ++i) if (q1.mem[(size_t)i] != q3.mem[(size_t)i]) ++differs;
    CHECK(differs > M * nout / 4);
}

TEST(op_vocab_edges) {
    T wte(RAD_F32, {3, 2}); wte.f({1, 2, 3, 4, 5, 6});
    T tok(RAD_I32, {3});    tok.n({2, 0, -1});
    T x(RAD_F32, {3, 2});
    A a; a.t(tok).t(wte).t(x).i("M", 3).i("n_embd", 2).i("n_vocab", 3).s("dtype", "f32");
    CHECK_OK(run("embed_lookup", a));
    CHECK_EQ(x.at(0), 5.0f); CHECK_EQ(x.at(1), 6.0f);
    CHECK_EQ(x.at(2), 1.0f); CHECK_EQ(x.at(3), 2.0f);
    CHECK_EQ(x.at(4), 0.0f); CHECK_EQ(x.at(5), 0.0f);   /* a negative id is padding */

    /* An id at or above n_vocab fails the launch rather than returning a plausible row. */
    T bad(RAD_I32, {1}); bad.n({7});
    T x2(RAD_F32, {1, 2});
    A b; b.t(bad).t(wte).t(x2).i("M", 1).i("n_embd", 2).i("n_vocab", 3).s("dtype", "f32");
    CHECK(run("embed_lookup", b) == RAD_E_INVAL);

    T h(RAD_F32, {1, 2});   h.f({1, 2});
    T lm(RAD_F32, {3, 2});  lm.f({1, 0, 0, 1, 1, 1});
    T lg(RAD_F32, {1, 3});
    A c; c.t(h).t(lm).t(lg).i("M", 1).i("n_vocab", 3).i("n_embd", 2).s("dtype", "f32");
    CHECK_OK(run("logits_gemm", c));
    CHECK_EQ(lg.at(0), 1.0f); CHECK_EQ(lg.at(1), 2.0f); CHECK_EQ(lg.at(2), 3.0f);
}

/* ================================================================== attention */
TEST(op_attn_paged) {
    /* Two query tokens, two query heads over one kv head, head_dim 2, one block of 2 slots.
     *   slot 0: K = [1,0]  V = [1,0]
     *   slot 1: K = [0,1]  V = [0,1]
     * Query head 0 takes q = [1,0] then [0,1]; head 1 takes q = [0,1] then [1,0]. */
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {2, 2, 2});
    q.f({ 1, 0,   0, 1,      /* token 0: head0 [1,0], head1 [0,1] */
          0, 1,   1, 0 });   /* token 1: head0 [0,1], head1 [1,0] */
    T kv(RAD_F32, {1, 1, 2, 4});
    kv.f({ 1, 0, 1, 0,       /* slot 0: K then V */
           0, 1, 0, 1 });    /* slot 1 */
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T su(RAD_I32, {1});    su.n({2});
    T out(RAD_F32, {2, 2, 2});
    A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(out)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", a));

    /* Token 0 sits at position 0 and sees key 0 only, so the softmax is trivial and both heads
     * return V0 = [1, 0]. */
    CHECK_NEAR(out.at(0), 1.0f, 1e-6); CHECK_NEAR(out.at(1), 0.0f, 1e-6);
    CHECK_NEAR(out.at(2), 1.0f, 1e-6); CHECK_NEAR(out.at(3), 0.0f, 1e-6);

    /* Token 1 sees both keys. Head 0, q = [0,1]: scores 0 and 1/sqrt(2); the weights are
     * exp(-0.70710678) = 0.49306869 and 1, summing to 1.49306869. */
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(out.at(4), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(5), 1.0f / sum, 1e-6);
    /* Head 1, q = [1,0]: the mirror image. */
    CHECK_NEAR(out.at(6), 1.0f / sum, 1e-6);
    CHECK_NEAR(out.at(7), w0 / sum, 1e-6);

    /* A window of 1 leaves token 1 seeing only its own key. */
    T out2(RAD_F32, {2, 2, 2});
    A b; b.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(out2)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 1).i("window", 1).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", b));
    CHECK_NEAR(out2.at(4), 0.0f, 1e-6);
    CHECK_NEAR(out2.at(5), 1.0f, 1e-6);
}

TEST(op_attn_paged_ragged) {
    /* A RAGGED batch, which is what a serving step actually looks like: sequence 0 is prefilling
     * two tokens while sequence 1 decodes one. total_q is 3 and n_seq is 2, so without cu_seqlens
     * this batch cannot even be expressed -- the op would have to assume 1.5 query rows each.
     *
     * One query head over one KV head, head_dim 2, one 2-slot block per sequence, both sequences
     * holding the same keys: slot 0 K = V = [1,0], slot 1 K = V = [0,1]. */
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {3, 1, 2});
    q.f({ 1, 0,        /* seq 0 token 0 */
          0, 1,        /* seq 0 token 1 */
          0, 1 });     /* seq 1, its only row, at position 1 */
    T kv(RAD_F32, {2, 1, 2, 4});
    kv.f({ 1, 0, 1, 0,   0, 1, 0, 1,      /* block 0 */
           1, 0, 1, 0,   0, 1, 0, 1 });   /* block 1 */
    T bt(RAD_I32, {2, 1}); bt.n({0, 1});
    T su(RAD_I32, {2});    su.n({2, 2});
    T cu(RAD_I32, {3});    cu.n({0, 2, 3});
    T out(RAD_F32, {3, 1, 2});
    A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().t(cu).t(out)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 1).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", a));

    /* Sequence 0's first token is at position 0 and sees key 0 only. */
    CHECK_NEAR(out.at(0), 1.0f, 1e-6); CHECK_NEAR(out.at(1), 0.0f, 1e-6);

    /* Its second is at position 1 and sees both. */
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(out.at(2), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(3), 1.0f / sum, 1e-6);

    /* Sequence 1 carries ONE row, so its position is ctx-1 = 1, not ctx-q_len+0 for the batch's
     * q_len of 2. Getting that wrong is the whole point of the test: with the batch maximum it
     * would sit at position 0 and return V0 = [1,0]. */
    CHECK_NEAR(out.at(4), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(5), 1.0f / sum, 1e-6);
}

TEST(op_attn_dense) {
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {2, 1, 2}); q.f({1, 0, 0, 1});
    T k(RAD_F32, {2, 1, 2}); k.f({1, 0, 0, 1});
    T v(RAD_F32, {2, 1, 2}); v.f({1, 0, 0, 1});
    T cu(RAD_I32, {2});      cu.n({0, 2});
    T o(RAD_F32, {2, 1, 2});
    A a; a.t(q).t(k).t(v).t(cu).t(o)
          .i("M", 2).i("head_dim", 2).i("gqa", 1).i("causal", 0).s("dtype", "f32");
    CHECK_OK(run("attn_dense", a));
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(o.at(0), 1.0f / sum, 1e-6);
    CHECK_NEAR(o.at(1), w0 / sum, 1e-6);
    CHECK_NEAR(o.at(2), w0 / sum, 1e-6);
    CHECK_NEAR(o.at(3), 1.0f / sum, 1e-6);
}

TEST(op_kv_store) {
    T k(RAD_F32, {2, 1, 2}); k.f({1, 2, 5, 6});
    T v(RAD_F32, {2, 1, 2}); v.f({3, 4, 7, 8});
    T sm(RAD_I32, {2});      sm.n({1, -1});         /* the second token has no slot */
    T kv(RAD_F32, {1, 1, 2, 4});
    A a; a.t(k).t(v).t(sm).t(kv)
          .i("M", 2).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2).s("kv_dtype", "f32");
    CHECK_OK(run("kv_store", a));
    /* K then V inside slot 1 of block 0. */
    CHECK_EQ(kv.at(4), 1.0f); CHECK_EQ(kv.at(5), 2.0f);
    CHECK_EQ(kv.at(6), 3.0f); CHECK_EQ(kv.at(7), 4.0f);
    /* Slot 0 was never mapped and must be untouched. */
    CHECK_EQ(kv.at(0), 0.0f); CHECK_EQ(kv.at(3), 0.0f);
}

/* THE ENGINE HANDS k AND v A COLUMN SLICE AND NOT A DENSE PLANE, which the case above cannot see
 * because it builds them dense.
 *
 * An architecture that does not de-interleave its projection declares the two KV halves over the
 * FUSED qkv buffer: [tokens, kv_heads * head_dim] whose row stride is the whole projection's
 * width. Deriving that stride from the shape instead of from the tensor puts token 0 in the right
 * slot and every token after it inside the PREVIOUS token's query rows -- a cache that looks
 * written, passes every extent check, and holds the wrong keys from the second token on. This is
 * that layout small enough to check by hand, and it belongs with the reference's own tests
 * because a reference has to be right on the operands an engine actually hands it and not only
 * on the ones a test draws. */
TEST(op_kv_store_reads_the_row_stride_and_not_the_shape) {
    /* A [3, 6] projection: columns 0..1 are the query, 2..3 the key, 4..5 the value. */
    T qkv(RAD_F32, {3, 6});
    qkv.f({ 90, 91,  1,  2,  3,  4,
            92, 93,  5,  6,  7,  8,
            94, 95,  9, 10, 11, 12 });
    T kslice(RAD_F32, {3, 2}), vslice(RAD_F32, {3, 2});
    kslice.t.data = (void*)((float*)qkv.t.data + 2); kslice.t.stride[0] = 6;
    vslice.t.data = (void*)((float*)qkv.t.data + 4); vslice.t.stride[0] = 6;

    T sm(RAD_I32, {3}); sm.n({0, 1, 2});
    T kv(RAD_F32, {1, 1, 4, 4});          /* one block, one kv head, four slots, K|V of width 2 */
    A a; a.t(kslice).t(vslice).t(sm).t(kv)
          .i("M", 3).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 4).s("kv_dtype", "f32");
    CHECK_OK(run("kv_store", a));
    for (int64_t t = 0; t < 3; ++t) {
        CHECK_EQ(kv.at(t * 4 + 0), 1.0f + 4 * (float)t);   /* k */
        CHECK_EQ(kv.at(t * 4 + 1), 2.0f + 4 * (float)t);
        CHECK_EQ(kv.at(t * 4 + 2), 3.0f + 4 * (float)t);   /* v */
        CHECK_EQ(kv.at(t * 4 + 3), 4.0f + 4 * (float)t);
    }
    /* Nothing reached the query columns and nothing reached the unmapped fourth slot. */
    CHECK_EQ(qkv.at(0), 90.0f);
    CHECK_EQ(kv.at(12), 0.0f); CHECK_EQ(kv.at(15), 0.0f);
}

TEST(op_rope) {
    /* head_dim 2 means one rotary pair with inv_freq 1, so the angle is the position itself. */
    T qkv(RAD_F32, {1, 6}); qkv.f({1, 0,  1, 0,  9, 9});   /* q, k, v */
    T pos(RAD_I32, {1});    pos.n({1});
    A a; a.t(qkv).t(pos).i("M", 1).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 10000).d("scale", 1).s("mode", "neox");
    CHECK_OK(run("rope", a));
    CHECK_NEAR(qkv.at(0), std::cos(1.0f), 1e-6);
    CHECK_NEAR(qkv.at(1), std::sin(1.0f), 1e-6);
    CHECK_NEAR(qkv.at(2), std::cos(1.0f), 1e-6);
    CHECK_NEAR(qkv.at(3), std::sin(1.0f), 1e-6);
    CHECK_EQ(qkv.at(4), 9.0f);          /* the value heads are left alone */
    CHECK_EQ(qkv.at(5), 9.0f);
}

/* THE EPILOGUE IS THE UNFUSED CHAIN: the biased product, then the activation of it, then the
 * residual -- each rounded where a separate tensor would be. */
TEST(op_gemm_nt_bias_epilogue) {
    T a_(RAD_F32, {1, 2}); a_.f({1, -1});
    T b_(RAD_F32, {3, 2}); b_.f({1, 0,  0, 1,  2, 1});
    T bias(RAD_F32, {3}); bias.f({0.5f, 0.25f, -1.0f});
    T res(RAD_F32, {1, 3}); res.f({10, 20, 30});
    const float pre[3] = { 1.5f, -0.75f, 0.0f };      /* a . b + bias */
    for (const char* act : { "gelu", "gelu_tanh" }) {
        T y(RAD_F32, {1, 3});
        A e; e.t(a_).t(b_).t(bias).t(res).t(y).i("M", 1).i("N", 3).i("K", 2).s("dtype", "f32")
              .s("act", act);
        CHECK_OK(run("gemm_nt_bias", e));
        for (int n = 0; n < 3; ++n) {
            const float x = pre[n];
            const float g = !std::strcmp(act, "gelu")
                ? 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)))
                : 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            CHECK_NEAR(y.at(n), res.at(n) + g, 1e-5);
        }
    }
    /* bf16 rounds at each step: 1.5 * 1.5 is exact, but the residual sum 258 + 2.25 is not. */
    T a2(RAD_BF16, {1, 1}); a2.f({1.5f});
    T b2(RAD_BF16, {1, 1}); b2.f({1.5f});
    T bias2(RAD_BF16, {1}); bias2.f({0.0f});
    T res2(RAD_BF16, {1, 1}); res2.f({258.0f});
    T y2(RAD_BF16, {1, 1});
    A f; f.t(a2).t(b2).t(bias2).t(res2).t(y2).i("M", 1).i("N", 1).i("K", 1).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt_bias", f));
    CHECK_EQ(y2.at(0), rad_bf16_to_f32(rad_f32_to_bf16(258.0f + 2.25f)));
}

/* THE MULTI-COMPONENT ROTARY. With every component of every token equal, the interleaved form is
 * NeoX to the bit; with them different, each frequency reads the component its section names. */
TEST(op_rope_multi_component) {
    const int64_t hd = 12, rot = 12;           /* six frequencies: sections 2 2 2 */
    std::vector<float> row(3 * hd);
    for (int64_t i = 0; i < 3 * hd; ++i) row[(size_t)i] = 0.1f * (float)(i + 1);
    T ref(RAD_F32, {1, 3 * hd}); ref.f(row);
    T pos1(RAD_I32, {1}); pos1.n({7});
    A a; a.t(ref).t(pos1).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 100).d("scale", 1).s("mode", "neox").i("rotary_dim", rot);
    CHECK_OK(run("rope", a));
    for (const char* mode : { "imrope", "mrope" }) {
        T got(RAD_F32, {1, 3 * hd}); got.f(row);
        T pos3(RAD_I32, {3, 1}); pos3.n({7, 7, 7});
        A b; b.t(got).t(pos3).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 1)
              .d("theta", 100).d("scale", 1).s("mode", mode).i("rotary_dim", rot)
              .s("sections", "2 2 2");
        CHECK_OK(run("rope", b));
        for (int64_t i = 0; i < 3 * hd; ++i) CHECK_EQ(got.at(i), ref.at(i));
    }
    /* Distinct components: frequency i of the interleaved form reads t, h, w, t, h, w for
     * sections 2 2 2 -- i % 3 == 1 and i < 6 is h, i % 3 == 2 and i < 6 is w. */
    {
        T got(RAD_F32, {1, 2 * hd}); got.f(std::vector<float>(2 * hd, 1.0f));
        T pos3(RAD_I32, {3, 1}); pos3.n({1, 2, 3});
        A b; b.t(got).t(pos3).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
              .d("theta", 100).d("scale", 1).s("mode", "imrope").i("rotary_dim", rot)
              .s("sections", "2 2 2");
        CHECK_OK(run("rope", b));
        const int comp[6] = { 0, 1, 2, 0, 1, 2 };
        const float p[3] = { 1, 2, 3 };
        for (int i = 0; i < 6; ++i) {
            const float inv = 1.0f / std::pow(100.0f, (float)(2 * i) / (float)rot);
            const float ang = p[comp[i]] * inv;
            CHECK_NEAR(got.at(i), std::cos(ang) - std::sin(ang), 1e-6);
            CHECK_NEAR(got.at(i + 6), std::sin(ang) + std::cos(ang), 1e-6);
        }
    }
    /* Axial: two sections of three, each restarting its ladder at theta^0. */
    {
        T got(RAD_F32, {1, 2 * hd}); got.f(std::vector<float>(2 * hd, 1.0f));
        T pos2(RAD_I32, {2, 1}); pos2.n({5, 9});
        A b; b.t(got).t(pos2).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
              .d("theta", 100).d("scale", 1).s("mode", "axial").i("rotary_dim", rot)
              .s("sections", "3 3");
        CHECK_OK(run("rope", b));
        for (int i = 0; i < 6; ++i) {
            const int c = i < 3 ? 0 : 1, j = i < 3 ? i : i - 3;
            const float inv = 1.0f / std::pow(100.0f, (float)(2 * j) / 6.0f);
            const float ang = (c ? 9.0f : 5.0f) * inv;
            CHECK_NEAR(got.at(i), std::cos(ang) - std::sin(ang), 1e-6);
        }
    }
}

/* ALIGNED CORNERS: a grid's first and last rows and columns sit exactly on the table's, so a
 * corner patch reads one table row whole; an interior one blends four. */
TEST(op_grid_embed) {
    const int64_t side = 3, n = 2;
    std::vector<float> tab((size_t)(side * side * n));
    for (int64_t r = 0; r < side * side; ++r)
        for (int64_t i = 0; i < n; ++i) tab[(size_t)(r * n + i)] = (float)(r * 10 + i);
    T table(RAD_F32, {side * side, n}); table.f(tab);
    /* Three rows of a 5 x 4 grid: the top-left corner, the bottom-right, and (2, 1). */
    T x(RAD_F32, {3, n}); x.f({0, 0,  100, 100,  0, 0});
    T crd(RAD_I32, {4, 3}); crd.n({0, 4, 2,   0, 3, 1,   5, 5, 5,   4, 4, 4});
    A a; a.t(x).t(table).t(crd).i("M", 3).i("n", n).i("side", side).s("dtype", "f32");
    CHECK_OK(run("grid_embed", a));
    CHECK_EQ(x.at(0), 0.0f);  CHECK_EQ(x.at(1), 1.0f);                /* table row 0 */
    CHECK_EQ(x.at(2), 180.0f); CHECK_EQ(x.at(3), 181.0f);             /* 100 + table row 8 */
    /* (2, 1) of 5 x 4 sits at table (1.0, 0.6667): rows 3 and 4 by 1/3 and 2/3. */
    const float want = (30.0f * (1.0f / 3.0f) + 40.0f * (2.0f / 3.0f));
    CHECK_NEAR(x.at(4), want, 1e-4);
}

TEST(op_qk_norm_rope) {
    /* At position 0 the rotation is the identity, so this isolates the per-head RMS norm. */
    T qkv(RAD_F32, {1, 6}); qkv.f({3, 4,  3, 4,  7, 7});
    T pos(RAD_I32, {1});    pos.n({0});
    T qw(RAD_F32, {2});     qw.f({1, 1});
    T kw(RAD_F32, {2});     kw.f({2, 2});
    A a; a.t(qkv).t(pos).t(qw).t(kw).i("M", 1).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 10000).d("eps", 0);
    CHECK_OK(run("qk_norm_rope", a));
    const float rms = std::sqrt((9.0f + 16.0f) / 2.0f);   /* 3.5355339 */
    CHECK_NEAR(qkv.at(0), 3.0f / rms, 1e-6);
    CHECK_NEAR(qkv.at(1), 4.0f / rms, 1e-6);
    CHECK_NEAR(qkv.at(2), 2.0f * 3.0f / rms, 1e-6);
    CHECK_NEAR(qkv.at(3), 2.0f * 4.0f / rms, 1e-6);
    CHECK_EQ(qkv.at(4), 7.0f);
}

/* ================================================================== mixture of experts */
TEST(op_router_topk) {
    T lg(RAD_F32, {1, 4}); lg.f({1, 2, 3, 0});
    T ids(RAD_I32, {1, 2}), w(RAD_F32, {1, 2});
    A a; a.t(lg).t(ids).t(w).i("M", 1).i("n_expert", 4).i("top_k", 2).i("norm", 0)
          .s("dtype", "f32");
    CHECK_OK(run("router_topk", a));
    CHECK_EQ(ids.iat(0), 2LL);
    CHECK_EQ(ids.iat(1), 1LL);
    const float sum = std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f + std::exp(-3.0f);
    CHECK_NEAR(w.at(0), 1.0f / sum, 1e-6);
    CHECK_NEAR(w.at(1), std::exp(-1.0f) / sum, 1e-6);

    /* Renormalised, the kept pair is the softmax of {3, 2} -- which is sigmoid(1) and its
     * complement, a number this test can state exactly. */
    T ids2(RAD_I32, {1, 2}), w2(RAD_F32, {1, 2});
    A b; b.t(lg).t(ids2).t(w2).i("M", 1).i("n_expert", 4).i("top_k", 2).i("norm", 1)
          .s("dtype", "f32");
    CHECK_OK(run("router_topk", b));
    CHECK_NEAR(w2.at(0), 0.73105858f, 1e-6);
    CHECK_NEAR(w2.at(1), 0.26894142f, 1e-6);
}

/* THE FOUR-CLASS GROUPED GEMM: expert e is entry e / 4 of class e % 4, each class at its own
 * width, and a narrower class's columns past its width are zero. Ten experts, so classes 0 and 1
 * hold three and 2 and 3 hold two -- the shape a layer with 510 quantised experts has -- and class
 * 1 is the wide one, as it is on rank 1. Every value is a small integer, so the sums are exact. */
TEST(op_moe_gemm_q_mod4) {
    const int64_t K = 3, top_k = 2, M = 3, ne = 10;
    const int64_t Nq[4] = { 2, 4, 2, 2 };
    auto wv = [](int64_t e, int64_t n, int64_t k) { return (float)(e * 100 + n * 10 + k); };
    std::vector<T> w, ws;
    w.reserve(4);
    ws.reserve(4);
    for (int q = 0; q < 4; ++q) {
        const int64_t cnt = (ne + 3 - q) / 4;
        w.emplace_back(RAD_F32, std::vector<int64_t>{ cnt, Nq[q], K });
        std::vector<float> v;
        for (int64_t j = 0; j < cnt; ++j)
            for (int64_t n = 0; n < Nq[q]; ++n)
                for (int64_t k = 0; k < K; ++k) v.push_back(wv(q + 4 * j, n, k));
        w.back().f(v);
        /* One unit scale an expert: the arithmetic under test is which weight a row reads. */
        ws.emplace_back(RAD_F32, std::vector<int64_t>{ cnt, 1, 1 });
        ws.back().f(std::vector<float>((size_t)cnt, 1.0f));
    }
    T act(RAD_F32, {M, K});
    act.f({ 1, 2, 3,  4, 5, 6,  7, 8, 9 });
    T as(RAD_F32, {M, 1});
    as.f({ 1, 1, 1 });
    /* Rows by expert: 1 owns rows 0-1, 2 row 2, 6 row 3, 9 rows 4-5, the rest none. */
    const std::vector<int64_t> eoff = { 0, 0, 2, 3, 3, 3, 3, 4, 4, 4, 6 };
    const std::vector<int64_t> flat = { 0, 2, 1, 4, 3, 5 };
    T sorted(RAD_I32, {6}); sorted.n(flat);
    T off(RAD_I32, {ne + 1}); off.n(eoff);
    T y(RAD_F32, {6, 4});
    A a;
    a.t(act).t(as).t(w[0]).t(ws[0]).t(sorted).t(off).t(w[1]).t(ws[1]).t(w[2]).t(ws[2]).t(w[3])
     .t(ws[3]).t(y)
     .i("M", M).i("N", Nq[0]).i("K", K).i("N_1", Nq[1]).i("K_1", K).i("N_2", Nq[2]).i("K_2", K)
     .i("N_3", Nq[3]).i("K_3", K).i("n_expert", ne).i("top_k", top_k).s("a_order", "token")
     .s("dtype", "f32");
    CHECK_OK(run("moe_gemm_q_mod4", a));
    for (int64_t i = 0; i < 6; ++i) {
        int64_t e = 0;
        while (eoff[(size_t)(e + 1)] <= i) ++e;
        const int64_t tok = flat[(size_t)i] / top_k;
        for (int64_t n = 0; n < 4; ++n) {
            float want = 0.0f;
            if (n < Nq[e % 4])
                for (int64_t k = 0; k < K; ++k) want += act.at(tok * K + k) * wv(e, n, k);
            CHECK_EQ(y.at(i * 4 + n), want);
        }
    }
}

/* THE WEIGHT SCALE GRID IS THE ONE THE DTYPE NAMES, in both grouped GEMMs: the checkpoint's
 * 128x128 tiles for block fp8, one scale a row per group of K for the four-bit experts -- a 128 at
 * w4a8/w4a8h/w4nla8h, a 64 at w4nl64a8h/w5nl64a8h, a 32 at w4nl32a8h. The oracle inverts a re-laid
 * weight into exactly this description (core/runtime/oracle.cpp), and the E4M3 scale views are
 * re-laid, so tiles here leave their scales nowhere to go. Then a case built from the description
 * itself: one expert of the last class at K = 128, whose two 64-wide halves scale by 2 and 3 under
 * one activation scale of 0.5 over the whole 128 -- a reading at any other grid gives another
 * number. */
TEST(op_moe_gemm_q_scale_grid_by_dtype) {
    struct Want { const char* dt; int64_t rows, cols; };
    /* N 256, K 512, group 128; the last class at N 128, K 256. */
    const Want wants[] = { { "fp8a8", 2, 4 },      { "w4a8", 256, 4 },     { "w4a8h", 256, 4 },
                           { "w4nla8h", 256, 4 },  { "w4nl64a8h", 256, 8 }, { "w5nl64a8h", 256, 8 },
                           { "w4nl32a8h", 256, 16 } };
    for (const char* op : { "moe_gemm_q", "moe_gemm_q_mod4" }) {
        const RadKernelInfo* k = kern(op);
        REQUIRE(k != nullptr && k->opd_shape != nullptr);
        const bool four = std::strcmp(op, "moe_gemm_q_mod4") == 0;
        const int last = four ? 11 : 7;
        for (const Want& w : wants) {
            A a;
            a.i("M", 2).i("N", 256).i("K", 512).i("n_expert", 8).i("top_k", 2).i("group", 128)
             .s("a_order", "token").s("dtype", w.dt);
            if (four) a.i("N_3", 128).i("K_3", 256);
            else      a.i("N_odd", 128).i("K_odd", 256);
            const RadArgs* r = a.get();
            RadOpdDesc d{}, dl{};
            REQUIRE_EQ(k->opd_shape(r->p, r->n_p, 3, &d), RAD_OK);
            REQUIRE_EQ(k->opd_shape(r->p, r->n_p, last, &dl), RAD_OK);
            CHECK_EQ(d.dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(d.rank, 3u);
            CHECK_EQ(d.shape[0], four ? 2LL : 4LL);
            if (d.shape[1] != w.rows || d.shape[2] != w.cols)
                std::fprintf(stderr, "    %s at %s: w_scale [%lld, %lld], want [%lld, %lld]\n", op,
                             w.dt, (long long)d.shape[1], (long long)d.shape[2],
                             (long long)w.rows, (long long)w.cols);
            CHECK_EQ(d.shape[1], w.rows);
            CHECK_EQ(d.shape[2], w.cols);
            /* the last class at its own geometry, on the same grid */
            CHECK_EQ(dl.shape[1], w.rows == 2 ? 1LL : 128LL);
            CHECK_EQ(dl.shape[2], w.cols / 2);
        }
    }

    for (const char* op : { "moe_gemm_q", "moe_gemm_q_mod4" }) {
        const RadKernelInfo* k = kern(op);
        REQUIRE(k != nullptr && k->opd_shape != nullptr);
        const bool four = std::strcmp(op, "moe_gemm_q_mod4") == 0;
        const int ncls = four ? 4 : 2, ne = ncls, n_opd = four ? 13 : 9;
        auto params = [&](A& x) {
            x.i("M", 1).i("N", 2).i("K", 128).i("n_expert", ne).i("top_k", 1).i("group", 128)
             .s("a_order", "token").s("dtype", "w4nl64a8h");
        };
        A p;
        params(p);
        const RadArgs* pr = p.get();
        std::deque<T> opd;
        for (int i = 0; i < n_opd; ++i) {
            RadOpdDesc d{};
            REQUIRE_EQ(k->opd_shape(pr->p, pr->n_p, i, &d), RAD_OK);
            opd.emplace_back(d.dtype, std::vector<int64_t>(d.shape, d.shape + d.rank));
        }
        /* the classes' (codes, scale) operand pairs, class 0 first */
        std::vector<std::pair<int, int>> cls = { { 2, 3 }, { 6, 7 } };
        if (four) cls.insert(cls.end(), { { 8, 9 }, { 10, 11 } });
        for (int q = 0; q < ncls; ++q) {
            T& w = opd[(size_t)cls[(size_t)q].first];
            T& s = opd[(size_t)cls[(size_t)q].second];
            REQUIRE_EQ(s.t.shape[1], 2LL);
            REQUIRE_EQ(s.t.shape[2], 2LL);
            const bool hit = q == ncls - 1;
            std::vector<float> wv(256);
            for (int64_t n = 0; n < 2; ++n)
                for (int64_t kk = 0; kk < 128; ++kk)
                    wv[(size_t)(n * 128 + kk)] = hit ? (float)(n + 1) : 7.0f;
            w.f(wv);
            s.f(hit ? std::vector<float>{ 2, 3, 2, 3 } : std::vector<float>{ 1, 1, 1, 1 });
        }
        opd[0].f(std::vector<float>(128, 1.0f));
        opd[1].f({ 0.5f });
        opd[4].n({ 0 });
        /* every expert empty but the last one, which owns the one row */
        std::vector<int64_t> off((size_t)ne + 1, 0);
        off[(size_t)ne] = 1;
        opd[5].n(off);
        A a;
        for (int i = 0; i < n_opd; ++i) a.t(opd[(size_t)i]);
        params(a);
        CHECK_OK(run(op, a));
        const T& y = opd[(size_t)n_opd - 1];
        CHECK_EQ(y.at(0), 160.0f);                   /* 0.5 * (64*1*2 + 64*1*3) */
        CHECK_EQ(y.at(1), 320.0f);                   /* 0.5 * (64*2*2 + 64*2*3) */
    }

    /* A dtype that names none of those grids -- a format of another op, or no dtype at all -- is
     * declined for every operand, the scale planes and the rest: a plane sized on a guessed grid
     * is a tensor of the wrong extent. */
    for (const char* op : { "moe_gemm_q", "moe_gemm_q_mod4" }) {
        const RadKernelInfo* k = kern(op);
        REQUIRE(k != nullptr && k->opd_shape != nullptr);
        for (const char* dt : { "w8a8", "mxfp4a8", "" }) {
            A a;
            a.i("M", 2).i("N", 256).i("K", 512).i("n_expert", 8).i("top_k", 2).i("group", 128)
             .s("a_order", "token");
            if (*dt) a.s("dtype", dt);
            const RadArgs* r = a.get();
            for (int opd : { 0, 2, 3 }) {
                RadOpdDesc d{};
                if (k->opd_shape(r->p, r->n_p, opd, &d) != RAD_E_UNSUPPORTED)
                    std::fprintf(stderr, "    %s at dtype '%s': operand %d described\n", op, dt,
                                 opd);
                CHECK_EQ(k->opd_shape(r->p, r->n_p, opd, &d), RAD_E_UNSUPPORTED);
            }
        }
    }
}

TEST(op_moe_chain) {
    /* Two tokens, two experts, top_k 2 -- every token visits every expert, so the whole chain is
     * checkable without any routing subtlety. */
    T ids(RAD_I32, {2, 2}); ids.n({1, 0, 0, 1});
    T sorted(RAD_I32, {4}), off(RAD_I32, {3}), cnt(RAD_I32, {2});
    A a; a.t(ids).t(sorted).t(off).t(cnt).i("M", 2).i("n_expert", 2).i("top_k", 2);
    CHECK_OK(run("moe_scatter", a));
    CHECK_EQ(cnt.iat(0), 2LL); CHECK_EQ(cnt.iat(1), 2LL);
    CHECK_EQ(off.iat(0), 0LL); CHECK_EQ(off.iat(1), 2LL); CHECK_EQ(off.iat(2), 4LL);
    /* expert 0 owns flat slots 1 and 2; expert 1 owns 0 and 3, in ascending flat order. */
    CHECK_EQ(sorted.iat(0), 1LL); CHECK_EQ(sorted.iat(1), 2LL);
    CHECK_EQ(sorted.iat(2), 0LL); CHECK_EQ(sorted.iat(3), 3LL);

    T act(RAD_F32, {2, 2}); act.f({1, 2, 3, 4});
    T w(RAD_F32, {2, 1, 2}); w.f({1, 0,  0, 1});     /* expert 0 picks dim 0, expert 1 dim 1 */
    T ye(RAD_F32, {4, 1});
    A b; b.t(act).t(w).t(sorted).t(off).t(ye)
          .i("M", 2).i("N", 1).i("K", 2).i("n_expert", 2).i("top_k", 2).s("dtype", "f32");
    CHECK_OK(run("moe_gemm", b));
    CHECK_EQ(ye.at(0), 1.0f);   /* sorted[0] = flat 1 -> token 0, expert 0 -> a[0][0] */
    CHECK_EQ(ye.at(1), 3.0f);
    CHECK_EQ(ye.at(2), 2.0f);
    CHECK_EQ(ye.at(3), 4.0f);

    T ew(RAD_F32, {2, 2}); ew.f({0.5f, 0.5f, 0.25f, 0.75f});
    T y(RAD_F32, {2, 1});
    A c; c.t(ye).t(ew).t(sorted).tnull().tnull().t(y)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32");
    CHECK_OK(run("moe_gather", c));
    /* token 0: slot 0 is at sorted position 2 (value 2), slot 1 at position 0 (value 1). */
    CHECK_NEAR(y.at(0), 0.5f * 2.0f + 0.5f * 1.0f, 1e-6);
    CHECK_NEAR(y.at(1), 0.25f * 3.0f + 0.75f * 4.0f, 1e-6);

    /* THE SHARED ARM FOLDED IN. Same routed result, plus `shared * act(gate)` -- the `scale_rows`
     * this op's output would otherwise be handed to. A pair or neither: one of the two alone is an error,
     * because a gate with nothing to gate is a caller who thinks they folded and did not. */
    T sh(RAD_F32, {2, 1}); sh.f({10.0f, 20.0f});
    T sg(RAD_F32, {2, 1}); sg.f({0.5f, 0.25f});
    T yf(RAD_F32, {2, 1});
    A d; d.t(ye).t(ew).t(sorted).t(sh).t(sg).t(yf)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32").s("act", "none");
    CHECK_OK(run("moe_gather", d));
    CHECK_NEAR(yf.at(0), 10.0f * 0.5f + 1.5f, 1e-6);
    CHECK_NEAR(yf.at(1), 20.0f * 0.25f + 3.75f, 1e-6);

    /* `act` applies to the SCALAR, not to the product: sigmoid(0) = 0.5 exactly. */
    T sg0(RAD_F32, {2, 1}); sg0.f({0.0f, 0.0f});
    A e; e.t(ye).t(ew).t(sorted).t(sh).t(sg0).t(yf)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32").s("act", "sigmoid");
    CHECK_OK(run("moe_gather", e));
    CHECK_NEAR(yf.at(0), 10.0f * 0.5f + 1.5f, 1e-6);
    CHECK_NEAR(yf.at(1), 20.0f * 0.5f + 3.75f, 1e-6);

    A f; f.t(ye).t(ew).t(sorted).t(sh).tnull().t(yf)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32");
    CHECK_EQ(run("moe_gather", f), RAD_E_INVAL);
}

TEST(op_moe_scatter_expert_base) {
    /* EXPERT PARALLELISM, which is what `expert_base` is for: four experts in the model, this
     * rank owns [2, 4), and the router -- replicated, so it picks from all four -- sends slots to
     * every one of them. The slots that went to ranks elsewhere have to be DROPPED here and the
     * ones that stayed have to be sorted by LOCAL index, because the offsets, the counts and the
     * weight table downstream are all this rank's.
     *
     * Two tokens, top_k 2: flat slots 0..3 route to experts 3, 0, 2, 1. Only slots 0 (expert 3 ->
     * local 1) and 2 (expert 2 -> local 0) are ours. */
    T ids(RAD_I32, {2, 2}); ids.n({3, 0, 2, 1});
    T sorted(RAD_I32, {4}), off(RAD_I32, {3}), cnt(RAD_I32, {2});
    A a; a.t(ids).t(sorted).t(off).t(cnt)
          .i("M", 2).i("n_expert", 2).i("top_k", 2).i("expert_base", 2);
    CHECK_OK(run("moe_scatter", a));
    CHECK_EQ(cnt.iat(0), 1LL); CHECK_EQ(cnt.iat(1), 1LL);
    CHECK_EQ(off.iat(0), 0LL); CHECK_EQ(off.iat(1), 1LL); CHECK_EQ(off.iat(2), 2LL);
    CHECK_EQ(sorted.iat(0), 2LL);   /* local expert 0 (global 2) owns flat slot 2 */
    CHECK_EQ(sorted.iat(1), 0LL);   /* local expert 1 (global 3) owns flat slot 0 */
    /* THE TAIL IS -1, which is what tells the GEMM there is no row there. Two of the four slots
     * belong to another rank and this one computes nothing for them; its partial is summed with
     * that rank's by the block's end-of-block all-reduce. */
    CHECK_EQ(sorted.iat(2), -1LL); CHECK_EQ(sorted.iat(3), -1LL);

    /* AND base 0 WITH THE FULL COUNT IS THE SINGLE-RANK CASE: every slot is kept. */
    T s2(RAD_I32, {4}), o2(RAD_I32, {5}), c2(RAD_I32, {4});
    A b; b.t(ids).t(s2).t(o2).t(c2).i("M", 2).i("n_expert", 4).i("top_k", 2);
    CHECK_OK(run("moe_scatter", b));
    CHECK_EQ(o2.iat(4), 4LL);
    CHECK_EQ(s2.iat(0), 1LL); CHECK_EQ(s2.iat(1), 3LL);
    CHECK_EQ(s2.iat(2), 2LL); CHECK_EQ(s2.iat(3), 0LL);
}

TEST(op_row_topk) {
    T x(RAD_F32, {1, 4}); x.f({3, 1, 3, 2});
    T idx(RAD_I32, {1, 2}), val(RAD_F32, {1, 2});
    A a; a.t(x).t(idx).t(val).i("M", 1).i("N", 4).i("R", 2).s("dtype", "f32");
    CHECK_OK(run("row_topk", a));
    CHECK_EQ(idx.iat(0), 0LL);      /* ties go to the LOWER column */
    CHECK_EQ(idx.iat(1), 2LL);
    CHECK_EQ(val.at(0), 3.0f);
    CHECK_EQ(val.at(1), 3.0f);
}

/* ================================================================== gated delta net */
TEST(op_gdn_conv_prep) {
    /* One sequence, two tokens, one head each of k and v, all width 1, conv width 2 with the
     * weight [0, 1] -- so the convolution is the identity on the current token and the silu is
     * the only thing that happens to it.
     * x is the fused row [q | k | v | a | b], five channels. */
    const int64_t conv_dim = 3, D = 5;
    T x(RAD_F32, {2, D});
    x.f({ 1, 1, 1, 0, 0,
          2, 2, 2, 0, 0 });
    T w(RAD_F32, {conv_dim, 2}); w.f({0, 1, 0, 1, 0, 1});
    T b(RAD_F32, {conv_dim});    b.f({0, 0, 0});
    T alog(RAD_F32, {1});        alog.f({0});
    T dtb(RAD_F32, {1});         dtb.f({0});
    T cs(RAD_F32, {1, conv_dim, 1});
    T cu(RAD_I32, {2});          cu.n({0, 2});
    T q(RAD_F32, {2, 1, 1}), k(RAD_F32, {2, 1, 1}), v(RAD_F32, {2, 1, 1});
    T g(RAD_F32, {2, 1}), beta(RAD_F32, {2, 1});
    /* Inputs first, outputs last: x, w, b, A_log, dt_bias, conv_state, cu, then the four optional
     * inputs (a, b_gate, cache_idx, has_init), then q, k, v, g, beta. `a` and `b_gate` are absent
     * here, so the gate and beta come out of the fused x row's last two columns. */
    A a; a.t(x).t(w).t(b).t(alog).t(dtb).t(cs).t(cu).tnull().tnull().tnull().tnull()
          .t(q).t(k).t(v).t(g).t(beta)
          .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2).i("conv_width", 2);
    CHECK_OK(run("gdn_conv_prep", a));

    /* v is not normalised, so it is exactly silu of the input. */
    CHECK_NEAR(v.at(0), 0.73105858f, 1e-6);
    CHECK_NEAR(v.at(1), 1.76159416f, 1e-6);
    /* q and k are l2-normalised over a one-element head, so they are 1 up to the epsilon. */
    CHECK_NEAR(q.at(0), 1.0f, 1e-4);
    CHECK_NEAR(k.at(1), 1.0f, 1e-4);
    /* g = -exp(A_log) * softplus(a + dt_bias) = -ln 2, cumulative within the chunk. */
    CHECK_NEAR(g.at(0), -0.69314718f, 1e-6);
    CHECK_NEAR(g.at(1), -1.38629436f, 1e-6);
    CHECK_NEAR(beta.at(0), 0.5f, 1e-6);
    /* The conv state keeps the sequence's last conv_width-1 RAW tokens. */
    CHECK_NEAR(cs.at(0), 2.0f, 1e-6);
    CHECK_NEAR(cs.at(2), 2.0f, 1e-6);

    /* THE SEPARATE GATE AND BETA PROJECTIONS. The same numbers must come out when `a` and
     * `b_gate` are named operands rather than the tail of x -- otherwise the schema's union of the
     * two readings means two different ops. `cache_idx` names the slot and `has_init` 0 says that
     * slot holds no history; here the conv weight is [0, 1] so the history is multiplied by zero
     * either way, and what this case proves is the OPERAND POSITIONS, not the zeroing. */
    T x2(RAD_F32, {2, conv_dim});
    x2.f({ 1, 1, 1,
           2, 2, 2 });
    T ta(RAD_F32, {2, 1});   ta.f({0, 0});
    T tb(RAD_F32, {2, 1});   tb.f({0, 0});
    T cidx(RAD_I32, {1});    cidx.n({0});
    T hini(RAD_U8, {1});     hini.n({0});
    T cs2(RAD_F32, {1, conv_dim, 1});
    for (int c = 0; c < conv_dim; ++c) st_dt(RAD_F32, cs2.t.data, c, 99.0f);   /* must be ignored */
    T q2(RAD_F32, {2, 1, 1}), k2(RAD_F32, {2, 1, 1}), v2(RAD_F32, {2, 1, 1});
    T g2(RAD_F32, {2, 1}), beta2(RAD_F32, {2, 1});
    A b2; b2.t(x2).t(w).t(b).t(alog).t(dtb).t(cs2).t(cu).t(ta).t(tb).t(cidx).t(hini)
            .t(q2).t(k2).t(v2).t(g2).t(beta2)
            .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2).i("conv_width", 2);
    CHECK_OK(run("gdn_conv_prep", b2));
    CHECK_NEAR(v2.at(0), 0.73105858f, 1e-6);
    CHECK_NEAR(v2.at(1), 1.76159416f, 1e-6);
    CHECK_NEAR(g2.at(0), -0.69314718f, 1e-6);
    CHECK_NEAR(g2.at(1), -1.38629436f, 1e-6);
    CHECK_NEAR(beta2.at(0), 0.5f, 1e-6);
}

TEST(op_gdn_conv_update_rollback) {
    /* THE SPECULATIVE ROLLBACK CONTRACT, which is the reason this op exists separately.
     * conv_width 3, so the state holds two history entries plus this step's tokens; a step of two
     * candidate tokens needs (width-2) + 2 = 3 slots. The state starts as [1, 2, 3] and
     * num_accepted = 2 means the read window is slots [1, 2] = [2, 3]. */
    const int64_t conv_dim = 3, D = 3, W = 3;
    T x(RAD_F32, {2, D});
    x.f({ 10, 10, 10,
          20, 20, 20 });
    T w(RAD_F32, {conv_dim, W}); w.f({1, 1, 1,  1, 1, 1,  1, 1, 1});
    T b(RAD_F32, {conv_dim});    b.f({0, 0, 0});
    T cs(RAD_F32, {1, conv_dim, 3});
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 3; ++i) st_dt(RAD_F32, cs.t.data, c * 3 + i, (float)(i + 1));
    T sidx(RAD_I32, {1}); sidx.n({0});
    T nacc(RAD_I32, {1}); nacc.n({2});
    T cu(RAD_I32, {2});   cu.n({0, 2});
    T q(RAD_F32, {2, 1, 1}), k(RAD_F32, {2, 1, 1}), v(RAD_F32, {2, 1, 1});
    A a; a.t(x).t(w).t(b).t(cs).t(sidx).t(nacc).t(cu).t(q).t(k).t(v)
          .i("q_len", 2).i("head_k", 1).i("head_v", 1).i("conv_width", 3);
    CHECK_OK(run("gdn_conv_update", a));

    /* Token 0's window is [state[1], state[2], x0] = [2, 3, 10] -> sum 15 -> silu(15). */
    CHECK_NEAR(v.at(0), 15.0f * (1.0f / (1.0f + std::exp(-15.0f))), 1e-4);
    /* Token 1's window is [state[2], x0, x1] = [3, 10, 20] -> 33. */
    CHECK_NEAR(v.at(1), 33.0f * (1.0f / (1.0f + std::exp(-33.0f))), 1e-3);

    /* And the buffer has shifted down by one: [old[2], x0, x1]. The invariant is that the last
     * accepted token of a step sits at slot (its index) + width - 2, so the NEXT step reads at
     * num_accepted - 1 and finds exactly the right history. */
    CHECK_NEAR(cs.at(0), 3.0f, 1e-6);
    CHECK_NEAR(cs.at(1), 10.0f, 1e-6);
    CHECK_NEAR(cs.at(2), 20.0f, 1e-6);
}

TEST(op_gdn_kkt_solve) {
    /* One chunk of two tokens, one head, head_k 1, both keys [1]. Then
     *     M[1][0] = beta1 * (k1 . k0) * exp(g1 - g0) = 0.5
     * and the inverse of the unit lower triangular I + M is I - M. */
    T k(RAD_F32, {2, 1, 1}); k.f({1, 1});
    T beta(RAD_F32, {2, 1}); beta.f({1.0f, 0.5f});
    T g(RAD_F32, {2, 1});    g.f({0, 0});
    T cu(RAD_I32, {2});      cu.n({0, 2});
    T A_(RAD_F32, {2, 1, 2});
    A a; a.t(k).t(beta).t(g).t(cu).t(A_).i("M", 2).i("head_k", 1).i("chunk", 2);
    CHECK_OK(run("gdn_kkt_solve", a));
    CHECK_NEAR(A_.at(0), 1.0f, 1e-6);
    CHECK_NEAR(A_.at(1), 0.0f, 1e-6);
    CHECK_NEAR(A_.at(2), -0.5f, 1e-6);
    CHECK_NEAR(A_.at(3), 1.0f, 1e-6);
}

TEST(op_gdn_chunk_scan_matches_recurrence) {
    /* THE CROSS-CHECK. The chunked scan and the plain recurrence compute the same thing by
     * completely different routes -- a triangular solve against a sequential update -- so running
     * both on one input and comparing is worth more than either checked alone.
     *
     * Two tokens, one head, head_k = head_v = 1, no gating (g = 0, beta = 1), q = k = [1].
     * By hand, recurrently:  S=0; u0 = 1*(1 - 0) = 1, S = 1, o0 = S*q0 = 1;
     *                             u1 = 1*(2 - 1) = 1, S = 2, o1 = 2.
     * The chunked form has to produce the same o and the same final state. */
    T q(RAD_F32, {2, 1, 1}); q.f({1, 1});
    T k(RAD_F32, {2, 1, 1}); k.f({1, 1});
    T v(RAD_F32, {2, 1, 1}); v.f({1, 2});
    T g(RAD_F32, {2, 1});    g.f({0, 0});
    T beta(RAD_F32, {2, 1}); beta.f({1, 1});
    T cu(RAD_I32, {2});      cu.n({0, 2});

    /* A comes from the kkt solve on the same input, so the two ops are chained the way a real
     * layer chains them. */
    T A_(RAD_F32, {2, 1, 2});
    A s; s.t(k).t(beta).t(g).t(cu).t(A_).i("M", 2).i("head_k", 1).i("chunk", 2);
    CHECK_OK(run("gdn_kkt_solve", s));
    CHECK_NEAR(A_.at(2), -1.0f, 1e-6);          /* beta1 * k1.k0 * e^0 = 1 */

    T h0(RAD_F32, {1, 1, 1, 1});
    T o(RAD_F32, {2, 1, 1}), ht(RAD_F32, {1, 1, 1, 1});
    A a; a.t(q).t(k).t(v).t(A_).t(g).t(beta).t(h0).t(cu).t(o).t(ht)
          .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2);
    CHECK_OK(run("gdn_chunk_scan", a));
    CHECK_NEAR(o.at(0), 1.0f, 1e-5);
    CHECK_NEAR(o.at(1), 2.0f, 1e-5);
    CHECK_NEAR(ht.at(0), 2.0f, 1e-5);

    /* A null h0 is a zero initial state, which is what a fresh sequence has. */
    T o2(RAD_F32, {2, 1, 1}), ht2(RAD_F32, {1, 1, 1, 1});
    A b; b.t(q).t(k).t(v).t(A_).t(g).t(beta).tnull().t(cu).t(o2).t(ht2)
          .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2);
    CHECK_OK(run("gdn_chunk_scan", b));
    CHECK_NEAR(o2.at(1), 2.0f, 1e-5);
}

TEST(op_gdn_chunk_scan_matches_recurrence_gated) {
    /* THE SAME CROSS-CHECK WITH THE GATING TURNED ON, AND ACROSS A CHUNK BOUNDARY.
     *
     * The test above runs g = 0, beta = 1 and head_k = head_v = 1 -- which is the identity case,
     * and the identity case cannot see the two things the chunked form actually has to get right:
     * that `g` reaches these ops as a PER-CHUNK INCLUSIVE CUMSUM (gdn_conv_prep writes it that way
     * and both kkt_solve and chunk_scan read differences of it), and that the state carried from
     * one chunk into the next is decayed by the whole chunk. Six tokens at chunk 4 is two chunks,
     * one of them partial, which is the shape a real prefill has.
     *
     * The expectation is a sequential recurrence written out here, not a second call into the
     * plugin: this compares two implementations, and a comparison of one implementation with
     * itself proves nothing.
     */
    const int64_t Tn = 6, C = 4, K = 2, V = 2;
    const float qd[Tn][K] = {{1.0f,0.0f},{0.5f,0.5f},{0.0f,1.0f},{1.0f,-0.5f},{0.25f,0.75f},{-1.0f,0.5f}};
    const float kd[Tn][K] = {{0.5f,0.5f},{1.0f,0.0f},{0.25f,-0.5f},{0.0f,1.0f},{0.75f,0.25f},{0.5f,-1.0f}};
    const float vd[Tn][V] = {{1.0f,2.0f},{-1.0f,0.5f},{0.25f,1.0f},{2.0f,-1.0f},{0.5f,0.5f},{1.0f,0.0f}};
    const float gr[Tn]    = {-0.1f, -0.4f, -0.2f, -0.7f, -0.3f, -0.5f};   /* raw log-decay */
    const float bt[Tn]    = { 0.9f,  0.3f,  0.6f,  0.2f,  0.8f,  0.5f};

    T q(RAD_F32, {Tn, 1, K}), k(RAD_F32, {Tn, 1, K}), v(RAD_F32, {Tn, 1, V});
    T g(RAD_F32, {Tn, 1}), beta(RAD_F32, {Tn, 1});
    std::vector<float> qf, kf, vf, gf, bf;
    for (int64_t t = 0; t < Tn; ++t) {
        for (int64_t d = 0; d < K; ++d) { qf.push_back(qd[t][d]); kf.push_back(kd[t][d]); }
        for (int64_t e = 0; e < V; ++e) vf.push_back(vd[t][e]);
        bf.push_back(bt[t]);
        /* the per-chunk inclusive cumsum, restarted at every chunk boundary */
        const float prev = (t % C == 0) ? 0.0f : gf.back();
        gf.push_back(prev + gr[t]);
    }
    q.f(qf); k.f(kf); v.f(vf); g.f(gf); beta.f(bf);
    T cu(RAD_I32, {2}); cu.n({0, (int)Tn});

    T A_(RAD_F32, {Tn, 1, C});
    A s; s.t(k).t(beta).t(g).t(cu).t(A_).i("M", Tn).i("head_k", K).i("chunk", C);
    CHECK_OK(run("gdn_kkt_solve", s));

    T o(RAD_F32, {Tn, 1, V}), ht(RAD_F32, {1, 1, V, K});
    A a; a.t(q).t(k).t(v).t(A_).t(g).t(beta).tnull().t(cu).t(o).t(ht)
          .i("M", Tn).i("head_k", K).i("head_v", V).i("chunk", C);
    CHECK_OK(run("gdn_chunk_scan", a));

    /* S *= e^g ; u = beta (v - S k) ; S += u k^T ; o = S q -- the delta rule, in order. */
    double S[V][K] = {{0, 0}, {0, 0}};
    for (int64_t t = 0; t < Tn; ++t) {
        const double dec = std::exp((double)gr[t]);
        double u[V];
        for (int64_t e = 0; e < V; ++e) {
            double sk = 0;
            for (int64_t d = 0; d < K; ++d) { S[e][d] *= dec; sk += S[e][d] * kd[t][d]; }
            u[e] = bt[t] * (vd[t][e] - sk);
        }
        for (int64_t e = 0; e < V; ++e) {
            double so = 0;
            for (int64_t d = 0; d < K; ++d) { S[e][d] += u[e] * kd[t][d]; so += S[e][d] * qd[t][d]; }
            CHECK_NEAR(o.at(t * V + e), (float)so, 2e-5);
        }
    }
    for (int64_t e = 0; e < V; ++e)
        for (int64_t d = 0; d < K; ++d)
            CHECK_NEAR(ht.at(e * K + d), (float)S[e][d], 2e-5);
}

TEST(op_gdn_recurrent_update) {
    /* One sequence, one token, one head, head_k 1 and head_v 2, so the gated norm over the head
     * is meaningful. A_log = 0 and a = 0 give g = -ln 2; b = 0 gives beta = 1/2. */
    T q(RAD_F32, {1, 1, 1}); q.f({1});
    T k(RAD_F32, {1, 1, 1}); k.f({1});
    T v(RAD_F32, {1, 1, 2}); v.f({3, 4});
    T av(RAD_F32, {1, 1});   av.f({0});
    T bv(RAD_F32, {1, 1});   bv.f({0});
    T alog(RAD_F32, {1});    alog.f({0});
    T dtb(RAD_F32, {1});     dtb.f({0});
    T state(RAD_F32, {2, 1, 2, 1});                   /* two slots */
    T sidx(RAD_I32, {1, 1}); sidx.n({1});
    T nacc(RAD_I32, {1});    nacc.n({1});
    T cu(RAD_I32, {2});      cu.n({0, 1});
    T z(RAD_F32, {1, 1, 2}); z.f({0, 0});
    T nw(RAD_F32, {2});      nw.f({1, 1});
    T o(RAD_F32, {1, 1, 2});
    /* `cu` sits between num_accepted and z -- the same slot the sibling gdn_conv_update puts it
     * in. It is REQUIRED: libr4d's kernel opens with cu[n] before any null test, so a schema that
     * omits it does not make it optional, it makes the launch fault. */
    A a; a.t(q).t(k).t(v).t(av).t(bv).t(alog).t(dtb).t(state).t(sidx).t(nacc).t(cu).t(z).t(nw)
          .t(o)
          .i("q_len", 1).i("head_k", 1).i("head_v", 2).d("eps", 0).s("act", "sigmoid");
    CHECK_OK(run("gdn_recurrent_update", a));

    /* The state is written to slot state_idx[0][0] = 1: u = beta * v = [1.5, 2], and the state
     * update is u k^T with k l2-normalised to 1. */
    CHECK_NEAR(state.at(2), 1.5f, 1e-4);
    CHECK_NEAR(state.at(3), 2.0f, 1e-4);
    /* o_raw = S q = [1.5, 2]; the gated norm divides by sqrt(mean(o^2)) = sqrt(3.125) and
     * multiplies by sigmoid(0) = 0.5. */
    const float sc = 1.0f / std::sqrt(3.125f);
    CHECK_NEAR(o.at(0), 1.5f * sc * 0.5f, 1e-4);
    CHECK_NEAR(o.at(1), 2.0f * sc * 0.5f, 1e-4);
}

TEST(op_gdn_gated_rmsnorm) {
    T x(RAD_F32, {1, 2}); x.f({3, 4});
    T z(RAD_F32, {1, 2}); z.f({0, 0});
    T w(RAD_F32, {2});    w.f({1, 1});
    T o(RAD_F32, {1, 2});
    A a; a.t(x).t(z).t(w).t(o).i("M", 1).i("channels", 2).d("eps", 0).s("act", "sigmoid");
    CHECK_OK(run("gdn_gated_rmsnorm", a));
    const float sc = 1.0f / std::sqrt(12.5f);
    CHECK_NEAR(o.at(0), 3.0f * sc * 0.5f, 1e-6);
    CHECK_NEAR(o.at(1), 4.0f * sc * 0.5f, 1e-6);
}

/* ================================================================== collectives */
TEST(op_collectives_world1) {
    T x(RAD_F32, {2}); x.f({1, 2});
    A a; a.t(x).i("world_size", 1).i("numel", 2).s("dtype", "f32").i("exact", 1);
    CHECK_OK(run("all_reduce", a));
    CHECK_EQ(x.at(0), 1.0f);        /* inout: the answer was already there */
    CHECK_EQ(x.at(1), 2.0f);

    T y(RAD_F32, {2});
    A b; b.t(x).t(y).i("world_size", 1).i("numel", 2).s("dtype", "f32");
    CHECK_OK(run("all_gather", b));
    CHECK_EQ(y.at(0), 1.0f);
    CHECK_EQ(y.at(1), 2.0f);
}

TEST(op_collectives_world2) {
    /* The fake multi-device host backend: two rank threads in one address space, which is exactly
     * how the engine runs tensor parallel (spec §1). */
    T x0(RAD_F32, {2}); x0.f({1, 2});
    T x1(RAD_F32, {2}); x1.f({10, 20});
    T g0(RAD_F32, {4}), g1(RAD_F32, {4});
    int st[2] = { 0, 0 };

    auto rank_body = [&](int r, T& x, T& gy) {
        A a; a.t(x).i("world_size", 2).i("numel", 2).s("dtype", "f32").i("exact", 1);
        int s = run("all_reduce", a, r, 2);
        A b; b.t(x).t(gy).i("world_size", 2).i("numel", 2).s("dtype", "f32");
        int s2 = run("all_gather", b, r, 2);
        st[r] = (s < 0) ? s : s2;
    };
    std::thread t0(rank_body, 0, std::ref(x0), std::ref(g0));
    std::thread t1(rank_body, 1, std::ref(x1), std::ref(g1));
    t0.join();
    t1.join();
    CHECK_OK(st[0]);
    CHECK_OK(st[1]);

    CHECK_EQ(x0.at(0), 11.0f); CHECK_EQ(x0.at(1), 22.0f);
    CHECK_EQ(x1.at(0), 11.0f); CHECK_EQ(x1.at(1), 22.0f);
    /* After the reduce both ranks hold the same values, so the gather is four copies of them. */
    CHECK_EQ(g0.at(0), 11.0f); CHECK_EQ(g0.at(2), 11.0f);
    CHECK_EQ(g1.at(1), 22.0f); CHECK_EQ(g1.at(3), 22.0f);
}

/* ================================================================== sampling */
/* Every scalar is a field of the params ROW, not a parameter: see SP above and
 * abi/rad_sample.h. These cases check that arithmetic against the schema core/sample/sampler.h
 * issues against. */
TEST(op_sample_penalties) {
    T lg(RAD_F32, {1, 4}); lg.f({1, -1, 2, 0});
    T hist(RAD_I32, {1, 3}); hist.n({0, 0, 2});
    SP sp(1);
    sp[0].rep_penalty = 2.0f; sp[0].freq_penalty = 0.5f; sp[0].pres_penalty = 1.0f;
    sp[0].hist_len = 3; sp[0].flags = RAD_SP_PENALTY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_penalties", a));
    /* token 0 appears twice: 1/2 - 2*0.5 - 1 = -1.5. token 2 appears once: 2/2 - 0.5 - 1 = -0.5. */
    CHECK_NEAR(lg.at(0), -1.5f, 1e-6);
    CHECK_NEAR(lg.at(1), -1.0f, 1e-6);
    CHECK_NEAR(lg.at(2), -0.5f, 1e-6);
    CHECK_NEAR(lg.at(3), 0.0f, 1e-6);

    /* A negative logit is MULTIPLIED by the repetition penalty, never divided -- dividing would
     * make it larger, which is the opposite of a penalty. */
    T l2(RAD_F32, {1, 2}); l2.f({-2, 0});
    T h2(RAD_I32, {1, 1}); h2.n({0});
    SP sp2(1);
    sp2[0].rep_penalty = 2.0f; sp2[0].hist_len = 1; sp2[0].flags = RAD_SP_PENALTY;
    A b; b.t(l2).t(sp2.operand()).t(h2).i("M", 1).i("n_vocab", 2);
    CHECK_OK(run("sample_penalties", b));
    CHECK_NEAR(l2.at(0), -4.0f, 1e-6);

    /* Without the flag the row is untouched, whatever the floats say: the flag is what lets a
     * kernel skip a stage for one sequence of a mixed batch. */
    T l3(RAD_F32, {1, 2}); l3.f({-2, 0});
    SP sp3(1);
    sp3[0].rep_penalty = 2.0f; sp3[0].hist_len = 1;      /* flags left at 0 */
    A e; e.t(l3).t(sp3.operand()).t(h2).i("M", 1).i("n_vocab", 2);
    CHECK_OK(run("sample_penalties", e));
    CHECK_EQ(l3.at(0), -2.0f);
}

TEST(op_sample_dry) {
    /* History "a b c a b": the last two tokens (a b) reproduce positions 0-1, so the token that
     * followed there -- c -- is what would extend the repeat, and it is the one penalised. */
    T lg(RAD_F32, {1, 4}); lg.f({0, 0, 0, 0});
    T hist(RAD_I32, {1, 5}); hist.n({0, 1, 2, 0, 1});
    SP sp(1);
    sp[0].dry_multiplier = 1.0f; sp[0].dry_base = 2.0f; sp[0].dry_allowed_length = 2;
    sp[0].hist_len = 5; sp[0].flags = RAD_SP_DRY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_dry", a));
    CHECK_NEAR(lg.at(2), -1.0f, 1e-6);     /* n == allowed_length, so base^0 = 1 */
    CHECK_EQ(lg.at(3), 0.0f);              /* nothing else is touched */
}

TEST(op_sample_temp_topk_argmax) {
    T lg(RAD_F32, {1, 4}); lg.f({1, 3, 2, 3});
    SP sp(1); sp[0].temp = 0.5f;
    A a; a.t(lg).t(sp.operand()).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_temp", a));
    CHECK_EQ(lg.at(0), 2.0f);
    CHECK_EQ(lg.at(1), 6.0f);

    T ci(RAD_I32, {1, 2}), cv(RAD_F32, {1, 2});
    SP spk(1); spk[0].top_k = 2;
    A b; b.t(lg).t(spk.operand()).t(ci).t(cv).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_topk", b));
    CHECK_EQ(ci.iat(0), 1LL);       /* the tie between index 1 and 3 goes to the lower first */
    CHECK_EQ(ci.iat(1), 3LL);
    CHECK_EQ(cv.at(0), 6.0f);

    /* A row's top_k narrower than the buffer leaves holes, which every later stage skips. */
    T ci2(RAD_I32, {1, 4}), cv2(RAD_F32, {1, 4});
    SP sp1(1); sp1[0].top_k = 1;
    A e; e.t(lg).t(sp1.operand()).t(ci2).t(cv2).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_topk", e));
    CHECK_EQ(ci2.iat(0), 1LL);
    CHECK_EQ(ci2.iat(1), -1LL);
    CHECK(cv2.at(1) == -INFINITY);

    T tk(RAD_I32, {1});
    SP spa(1);
    A c; c.t(lg).t(spa.operand()).t(tk).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_argmax", c));
    CHECK_EQ(tk.iat(0), 1LL);

    /* temp <= 0 leaves the row untouched: greedy is sample_argmax, not a special temperature. */
    SP spz(1); spz[0].temp = 0.0f;
    A d; d.t(lg).t(spz.operand()).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_temp", d));
    CHECK_EQ(lg.at(1), 6.0f);
}

TEST(op_sample_narrowing) {
    /* Two candidates whose probabilities are 3/4 and 1/4. */
    const float ln3 = 1.09861229f;
    T ci(RAD_I32, {1, 2}); ci.n({5, 7});
    T cv(RAD_F32, {1, 2}); cv.f({ln3, 0});
    SP sp(1); sp[0].top_p = 0.7f;
    A a; a.t(ci).t(cv).t(sp.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", a));
    CHECK_EQ(ci.iat(0), 5LL);
    CHECK_EQ(ci.iat(1), -1LL);              /* dropped: index -1, value -infinity */
    CHECK(cv.at(1) == -INFINITY);

    T ci2(RAD_I32, {1, 2}); ci2.n({5, 7});
    T cv2(RAD_F32, {1, 2}); cv2.f({ln3, 0});
    SP sp2(1); sp2[0].top_p = 0.9f;
    A b; b.t(ci2).t(cv2).t(sp2.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", b));
    CHECK_EQ(ci2.iat(1), 7LL);              /* 0.75 alone does not reach 0.9 */

    T ci3(RAD_I32, {1, 2}); ci3.n({5, 7});
    T cv3(RAD_F32, {1, 2}); cv3.f({ln3, 0});
    SP sp3(1); sp3[0].min_p = 0.5f;
    A c; c.t(ci3).t(cv3).t(sp3.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_minp", c));
    CHECK_EQ(ci3.iat(1), -1LL);             /* 1/3 of the peak is below 0.5 */

    T ci4(RAD_I32, {1, 2}); ci4.n({5, 7});
    T cv4(RAD_F32, {1, 2}); cv4.f({ln3, 0});
    SP sp4(1); sp4[0].min_p = 0.2f;
    A d; d.t(ci4).t(cv4).t(sp4.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_minp", d));
    CHECK_EQ(ci4.iat(1), 7LL);

    /* An identity setting is a no-op rather than a narrowing: top_p 1 keeps everything. */
    T ci5(RAD_I32, {1, 2}); ci5.n({5, 7});
    T cv5(RAD_F32, {1, 2}); cv5.f({ln3, 0});
    SP sp5(1);
    A f; f.t(ci5).t(cv5).t(sp5.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", f));
    CHECK_EQ(ci5.iat(1), 7LL);
}

TEST(op_sample_xtc) {
    /* Two candidates both above the threshold, and a probability of 1 so the draw always fires:
     * the MORE likely one is dropped and the least likely of those above survives. That is the
     * inversion the stage exists for. */
    T ci(RAD_I32, {1, 2}); ci.n({5, 7});
    T cv(RAD_F32, {1, 2}); cv.f({0.0f, 0.0f});
    SP sp(1); sp[0].xtc_probability = 1.0f; sp[0].xtc_threshold = 0.4f; sp[0].flags = RAD_SP_XTC;
    A a; a.t(ci).t(cv).t(sp.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_xtc", a));
    CHECK_EQ(ci.iat(0), -1LL);
    CHECK_EQ(ci.iat(1), 7LL);

    /* One candidate above the threshold is a no-op: dropping it would leave the row with no
     * choice of its own. */
    T ci2(RAD_I32, {1, 2}); ci2.n({5, 7});
    T cv2(RAD_F32, {1, 2}); cv2.f({0.0f, -80.0f});
    SP sp2(1); sp2[0].xtc_probability = 1.0f; sp2[0].xtc_threshold = 0.4f;
    A b; b.t(ci2).t(cv2).t(sp2.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_xtc", b));
    CHECK_EQ(ci2.iat(0), 5LL);
    CHECK_EQ(ci2.iat(1), 7LL);
}

TEST(op_sample_typical) {
    /* A uniform distribution is entirely typical: every candidate's surprisal IS the entropy, so
     * nothing is dropped whatever typical_p asks for. */
    T ci(RAD_I32, {1, 4}); ci.n({1, 2, 3, 4});
    T cv(RAD_F32, {1, 4}); cv.f({0, 0, 0, 0});
    SP sp(1); sp[0].typical_p = 0.5f;
    A a; a.t(ci).t(cv).t(sp.operand()).i("M", 1).i("n_cand", 4);
    CHECK_OK(run("sample_typical", a));
    for (int i = 0; i < 4; ++i) CHECK(ci.iat(i) >= 0);

    /* A PEAKED row: logits {4, 0, 0} are probabilities {0.964, 0.018, 0.018}, entropy 0.180.
     * The peak's surprisal is 0.037 -- distance 0.143 -- and the tail's is 4.017, distance 3.837.
     * So the peak is the TYPICAL one here and the tail is dropped, which is the opposite of what a
     * first reading of "exclude the most likely" suggests and is why this case is written down:
     * typical sampling is a distance from the entropy, not a rank.
     *
     * typical_p 0.1 is reached by the peak alone (0.964), so the cut lands at its distance. */
    T ci2(RAD_I32, {1, 3}); ci2.n({1, 2, 3});
    T cv2(RAD_F32, {1, 3}); cv2.f({4.0f, 0.0f, 0.0f});
    SP sp2(1); sp2[0].typical_p = 0.1f;
    A b; b.t(ci2).t(cv2).t(sp2.operand()).i("M", 1).i("n_cand", 3);
    CHECK_OK(run("sample_typical", b));
    CHECK_EQ(ci2.iat(0), 1LL);
    CHECK_EQ(ci2.iat(1), -1LL);
    CHECK_EQ(ci2.iat(2), -1LL);
}

TEST(op_sample_mask) {
    T lg(RAD_F32, {1, 4}); lg.f({1, 2, 3, 4});
    T bm(RAD_U32, {1, 1}); bm.n({0x5});     /* a SET bit means allowed: tokens 0 and 2 */
    SP sp(1); sp[0].mask_row = 0; sp[0].flags = RAD_SP_MASK;
    A a; a.t(lg).t(sp.operand()).t(bm).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_mask", a));
    CHECK_EQ(lg.at(0), 1.0f);
    CHECK(lg.at(1) == -INFINITY);
    CHECK_EQ(lg.at(2), 3.0f);
    CHECK(lg.at(3) == -INFINITY);

    /* mask_row -1 is an unconstrained sequence sharing the plane with a constrained one. */
    T l2(RAD_F32, {1, 4}); l2.f({1, 2, 3, 4});
    SP sp2(1); sp2[0].mask_row = -1; sp2[0].flags = RAD_SP_MASK;
    A b; b.t(l2).t(sp2.operand()).t(bm).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_mask", b));
    CHECK_EQ(l2.at(1), 2.0f);
}

TEST(op_sample_pick) {
    /* A degenerate distribution has one answer whatever the draw is. */
    T ci(RAD_I32, {1, 2}); ci.n({5, 7});
    T cv(RAD_F32, {1, 2}); cv.f({0.0f, -80.0f});
    T tk(RAD_I32, {1});
    SP sp(1); sp[0].seed = 12345; sp[0].pos = 0;
    A a; a.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_pick", a));
    CHECK_EQ(tk.iat(0), 5LL);

    /* Determinism: the same (seed, pos) gives the same token, every time. */
    T ci2(RAD_I32, {1, 2}); ci2.n({5, 7});
    T cv2(RAD_F32, {1, 2}); cv2.f({0.0f, 0.0f});
    int64_t first = -1;
    for (int rep = 0; rep < 4; ++rep) {
        T tk2(RAD_I32, {1});
        A b; b.t(ci2).t(cv2).t(sp.operand()).t(tk2).i("M", 1).i("n_cand", 2);
        CHECK_OK(run("sample_pick", b));
        if (rep == 0) first = tk2.iat(0);
        CHECK_EQ(tk2.iat(0), first);
    }
    CHECK(first == 5 || first == 7);

    /* And the stream is not constant: over a run of positions an even split produces both. */
    int saw5 = 0, saw7 = 0;
    for (int off = 0; off < 64; ++off) {
        SP spo(1); spo[0].seed = 12345; spo[0].pos = (uint64_t)off;
        T tk3(RAD_I32, {1});
        A c; c.t(ci2).t(cv2).t(spo.operand()).t(tk3).i("M", 1).i("n_cand", 2);
        CHECK_OK(run("sample_pick", c));
        if (tk3.iat(0) == 5) ++saw5; else ++saw7;
    }
    CHECK(saw5 > 8);
    CHECK(saw7 > 8);

    /* A dropped candidate is skipped, never picked. */
    T ci3(RAD_I32, {1, 2}); ci3.n({-1, 7});
    T cv3(RAD_F32, {1, 2}); cv3.f({-INFINITY, 0.0f});
    T tk4(RAD_I32, {1});
    A d; d.t(ci3).t(cv3).t(sp.operand()).t(tk4).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_pick", d));
    CHECK_EQ(tk4.iat(0), 7LL);
}

TEST(op_sample_merge_topk) {
    /* Two ranks' candidate sets, already carrying GLOBAL token ids, as (u32 id, f32 logit) pairs.
     * The merge is the k largest across both, descending. */
    T gt(RAD_U32, {1, 8});
    float* gf = (float*)gt.t.data;
    uint32_t* gi = (uint32_t*)gt.t.data;
    gi[0] = 11; gf[1] = 3.0f;      /* rank 0 */
    gi[2] = 12; gf[3] = 1.0f;
    gi[4] = 20; gf[5] = 4.0f;      /* rank 1 */
    gi[6] = 21; gf[7] = 2.0f;
    T ci(RAD_I32, {1, 3}), cv(RAD_F32, {1, 3});
    SP sp(1);
    A a; a.t(gt).t(sp.operand()).t(ci).t(cv).i("M", 1).i("n_cand", 3);
    CHECK_OK(run("sample_merge_topk", a));
    CHECK_EQ(ci.iat(0), 20LL); CHECK_EQ(cv.at(0), 4.0f);
    CHECK_EQ(ci.iat(1), 11LL); CHECK_EQ(cv.at(1), 3.0f);
    CHECK_EQ(ci.iat(2), 21LL); CHECK_EQ(cv.at(2), 2.0f);
}

/* ------------------------------------------------------------------ the history stages, per row */

/* EVERY ROW READS ITS OWN ROW OF THE HISTORY PLANE. hist_off is an offset within the row, so the
 * engine stages 0 for every row -- and a second sampled row (a second sequence, or a speculative
 * verify's next position) is penalised against its own tokens, not left alone. */
TEST(op_sample_history_stages_read_each_rows_own_row) {
    T lg(RAD_F32, {2, 4}); lg.f({1, 1, 1, 1,  1, 1, 1, 1});
    T hist(RAD_I32, {2, 3}); hist.n({0, 0, 0,  3, 3, 3});
    SP sp(2);
    for (size_t m = 0; m < 2; ++m) {
        sp[m].rep_penalty = 2.0f; sp[m].hist_len = 3; sp[m].flags = RAD_SP_PENALTY;
    }
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 2).i("n_vocab", 4);
    CHECK_OK(run("sample_penalties", a));
    CHECK_NEAR(lg.at(0), 0.5f, 1e-6);          /* row 0 penalises token 0 ... */
    CHECK_NEAR(lg.at(3), 1.0f, 1e-6);
    CHECK_NEAR(lg.at(4 + 3), 0.5f, 1e-6);      /* ... and row 1 token 3, out of ITS row */
    CHECK_NEAR(lg.at(4 + 0), 1.0f, 1e-6);

    /* The same for DRY: row 1's history "a b c a b" repeats, row 0's does not. */
    T l2(RAD_F32, {2, 4}); l2.f({0, 0, 0, 0,  0, 0, 0, 0});
    T h2(RAD_I32, {2, 5}); h2.n({0, 1, 2, 3, 0,  0, 1, 2, 0, 1});
    SP sd(2);
    for (size_t m = 0; m < 2; ++m) {
        sd[m].dry_multiplier = 1.0f; sd[m].dry_base = 2.0f; sd[m].dry_allowed_length = 2;
        sd[m].hist_len = 5; sd[m].flags = RAD_SP_DRY;
    }
    A b; b.t(l2).t(sd.operand()).t(h2).i("M", 2).i("n_vocab", 4);
    CHECK_OK(run("sample_dry", b));
    for (int i = 0; i < 4; ++i) CHECK_EQ(l2.at(i), 0.0f);
    CHECK_NEAR(l2.at(4 + 2), -1.0f, 1e-6);
}

/* A SPLIT VOCABULARY: the history holds GLOBAL ids and the row is this rank's shard, so a token's
 * column is its id less vocab_off, and an id another rank owns is not a column here at all. */
TEST(op_sample_history_stages_subtract_the_vocab_offset) {
    /* This rank holds ids 4..7. */
    T lg(RAD_F32, {1, 4}); lg.f({1, 1, 1, 1});
    T hist(RAD_I32, {1, 3}); hist.n({5, 1, 6});
    SP sp(1);
    sp[0].rep_penalty = 2.0f; sp[0].hist_len = 3; sp[0].flags = RAD_SP_PENALTY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4).i("vocab_off", 4);
    CHECK_OK(run("sample_penalties", a));
    CHECK_NEAR(lg.at(0), 1.0f, 1e-6);          /* id 4: not in the history */
    CHECK_NEAR(lg.at(1), 0.5f, 1e-6);          /* id 5 */
    CHECK_NEAR(lg.at(2), 0.5f, 1e-6);          /* id 6 */
    CHECK_NEAR(lg.at(3), 1.0f, 1e-6);          /* id 7 -- and id 1 is rank 0's, not column 1 */

    /* DRY: "4 5 7 4 5" repeats "4 5", so id 7 -- column 3 -- is the continuation. */
    T l2(RAD_F32, {1, 4}); l2.f({0, 0, 0, 0});
    T h2(RAD_I32, {1, 5}); h2.n({4, 5, 7, 4, 5});
    SP sd(1);
    sd[0].dry_multiplier = 1.0f; sd[0].dry_base = 2.0f; sd[0].dry_allowed_length = 2;
    sd[0].hist_len = 5; sd[0].flags = RAD_SP_DRY;
    A b; b.t(l2).t(sd.operand()).t(h2).i("M", 1).i("n_vocab", 4).i("vocab_off", 4);
    CHECK_OK(run("sample_dry", b));
    CHECK_NEAR(l2.at(3), -1.0f, 1e-6);
    CHECK_EQ(l2.at(0), 0.0f);
    CHECK_EQ(l2.at(1), 0.0f);
    CHECK_EQ(l2.at(2), 0.0f);
}

/* repeat_last_n and dry_penalty_last_n are llama.cpp's: 0 is OFF, negative is the whole history,
 * n > 0 is the last n tokens. */
TEST(op_sample_history_window_of_zero_turns_the_stage_off) {
    auto penal = [](int32_t last_n) {
        T lg(RAD_F32, {1, 4}); lg.f({1, 1, 1, 1});
        T hist(RAD_I32, {1, 3}); hist.n({0, 1, 2});
        SP sp(1);
        sp[0].rep_penalty = 2.0f; sp[0].hist_len = 3; sp[0].penalty_last_n = last_n;
        sp[0].flags = RAD_SP_PENALTY;
        A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4);
        CHECK_OK(run("sample_penalties", a));
        return std::vector<float>{ lg.at(0), lg.at(1), lg.at(2) };
    };
    const auto off = penal(0), all = penal(-1), tail = penal(2);
    CHECK_EQ(off[0], 1.0f); CHECK_EQ(off[1], 1.0f); CHECK_EQ(off[2], 1.0f);
    CHECK_NEAR(all[0], 0.5f, 1e-6); CHECK_NEAR(all[2], 0.5f, 1e-6);
    CHECK_EQ(tail[0], 1.0f);                   /* the window is the TAIL */
    CHECK_NEAR(tail[1], 0.5f, 1e-6); CHECK_NEAR(tail[2], 0.5f, 1e-6);

    T l2(RAD_F32, {1, 4}); l2.f({0, 0, 0, 0});
    T h2(RAD_I32, {1, 5}); h2.n({0, 1, 2, 0, 1});
    SP sd(1);
    sd[0].dry_multiplier = 1.0f; sd[0].dry_base = 2.0f; sd[0].dry_allowed_length = 2;
    sd[0].hist_len = 5; sd[0].dry_penalty_last_n = 0; sd[0].flags = RAD_SP_DRY;
    A b; b.t(l2).t(sd.operand()).t(h2).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_dry", b));
    CHECK_EQ(l2.at(2), 0.0f);
}

/* DRY PENALISES A TOKEN ONCE, BY ITS LONGEST REPEAT. History "a b x a b x a b": the suffix "a b"
 * also ends at position 1 (a repeat of 2) and at position 4 (a repeat of 5), and x follows both.
 * llama.cpp takes the longer: 1 * 2^(5-2) = 8. Summing the two would give 9. */
TEST(op_sample_dry_penalises_each_token_once_by_its_longest_repeat) {
    T lg(RAD_F32, {1, 3}); lg.f({0, 0, 0});
    T hist(RAD_I32, {1, 8}); hist.n({0, 1, 2, 0, 1, 2, 0, 1});
    SP sp(1);
    sp[0].dry_multiplier = 1.0f; sp[0].dry_base = 2.0f; sp[0].dry_allowed_length = 2;
    sp[0].hist_len = 8; sp[0].flags = RAD_SP_DRY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 3);
    CHECK_OK(run("sample_dry", a));
    CHECK_EQ(lg.at(2), -8.0f);
    CHECK_EQ(lg.at(0), 0.0f);
    CHECK_EQ(lg.at(1), 0.0f);
}

/* THE BREAKERS REACH THE KERNEL. A position the breaker plane marks holds a whole sequence
 * breaker, which DRY never penalises; and the row's dry_rep_limit caps how long a repeat may
 * count, down to "not at all" below allowed_length. */
TEST(op_sample_dry_honours_the_breaker_plane_and_the_repeat_limit) {
    auto dry = [](int32_t rep_limit, bool mark_x) {
        T lg(RAD_F32, {1, 3}); lg.f({0, 0, 0});
        T hist(RAD_I32, {1, 8}); hist.n({0, 1, 2, 0, 1, 2, 0, 1});
        T brk(RAD_U8, {1, 8});
        for (int i = 0; i < 8; ++i) st_int(RAD_U8, brk.t.data, i, 0);
        if (mark_x) { st_int(RAD_U8, brk.t.data, 2, 1); st_int(RAD_U8, brk.t.data, 5, 1); }
        SP sp(1);
        sp[0].dry_multiplier = 1.0f; sp[0].dry_base = 2.0f; sp[0].dry_allowed_length = 2;
        sp[0].hist_len = 8; sp[0].dry_rep_limit = rep_limit; sp[0].flags = RAD_SP_DRY;
        A a; a.t(lg).t(sp.operand()).t(hist).t(brk).i("M", 1).i("n_vocab", 3);
        CHECK_OK(run("sample_dry", a));
        return lg.at(2);
    };
    CHECK_EQ(dry(0, false), -8.0f);            /* the plane present and empty changes nothing */
    CHECK_EQ(dry(0, true), 0.0f);              /* x is a breaker */
    CHECK_EQ(dry(3, false), -2.0f);            /* the repeat of 5 counts as 3: 2^(3-2) */
    CHECK_EQ(dry(1, false), 0.0f);             /* below allowed_length nothing counts */
}

/* ------------------------------------------------------------------ greedy rows in a mixed batch */

/* A GREEDY ROW'S TOKEN IS THE ARGMAX'S. In a batch mixing greedy and sampled rows the argmax runs
 * over every row and so does the candidate chain, so every candidate stage has to treat a greedy
 * row as having no candidates -- or the pick replaces its argmax with a draw. */
TEST(op_sample_candidate_stages_leave_a_greedy_row_alone) {
    T lg(RAD_F32, {2, 4}); lg.f({0, 5, 0, 0,  -80, -80, 0, -80});
    SP sp(2);
    sp[0].flags = RAD_SP_GREEDY;
    sp[0].seed = 1; sp[1].seed = 1;

    T ci(RAD_I32, {2, 4}), cv(RAD_F32, {2, 4});
    A k; k.t(lg).t(sp.operand()).t(ci).t(cv).i("M", 2).i("n_vocab", 4);
    CHECK_OK(run("sample_topk", k));
    for (int i = 0; i < 4; ++i) CHECK_EQ(ci.iat(i), -1LL);   /* row 0: no candidates */
    CHECK_EQ(ci.iat(4), 2LL);                                /* row 1: its own set */

    T tk(RAD_I32, {2}); tk.n({1, -7});                      /* the argmax already wrote row 0 */
    A p; p.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 2).i("n_cand", 4);
    CHECK_OK(run("sample_pick", p));
    CHECK_EQ(tk.iat(0), 1LL);
    CHECK_EQ(tk.iat(1), 2LL);

    /* The narrowing stages leave a greedy row's set as it is, holes or not. */
    T c2(RAD_I32, {1, 2}); c2.n({5, 7});
    T v2(RAD_F32, {1, 2}); v2.f({0.0f, 0.0f});
    SP sg(1); sg[0].flags = RAD_SP_GREEDY; sg[0].top_p = 0.3f;
    A n; n.t(c2).t(v2).t(sg.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", n));
    CHECK_EQ(c2.iat(1), 7LL);

    /* And the merge writes a greedy row an empty set. */
    T gt(RAD_U32, {1, 4});
    uint32_t* gi = (uint32_t*)gt.t.data;
    float* gf = (float*)gt.t.data;
    gi[0] = 11; gf[1] = 3.0f; gi[2] = 20; gf[3] = 4.0f;
    T mi(RAD_I32, {1, 2}), mv(RAD_F32, {1, 2});
    A g; g.t(gt).t(sg.operand()).t(mi).t(mv).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_merge_topk", g));
    CHECK_EQ(mi.iat(0), -1LL);
    CHECK_EQ(mi.iat(1), -1LL);
}

/* THE PICK'S FALL-THROUGH NEVER LANDS ON A MASKED CANDIDATE. top-k fills its width from a masked
 * row, so the grammar's forbidden tokens sit at -infinity at the tail of the set, and a draw that
 * walks past every comparison -- float drift at the very top of the range -- takes the last
 * candidate. That has to be the last one the distribution allows. A NaN logit makes the sum NaN
 * and forces the fall-through on every draw, which is what makes the case checkable. */
TEST(op_sample_pick_falls_through_to_an_allowed_candidate) {
    T ci(RAD_I32, {1, 3}); ci.n({5, 6, 7});
    T cv(RAD_F32, {1, 3}); cv.f({0.0f, NAN, -INFINITY});
    SP sp(1); sp[0].seed = 3;
    for (uint64_t pos = 0; pos < 8; ++pos) {
        sp[0].pos = pos;
        T tk(RAD_I32, {1});
        A a; a.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 1).i("n_cand", 3);
        CHECK_OK(run("sample_pick", a));
        CHECK_EQ(tk.iat(0), 5LL);
    }
}

/* THE PICK'S DRAW IS rad_sample.h's, on the pick's key: the same uniform the host reference and
 * every other implementation take for (seed, pos). Two candidates at 1/4 and 3/4 split exactly at
 * u = 0.25, so the token says which side of it the draw fell. */
TEST(op_sample_pick_draws_the_specified_uniform) {
    const float ln3 = 1.09861229f;
    int agree = 0;
    for (uint64_t pos = 0; pos < 64; ++pos) {
        T ci(RAD_I32, {1, 2}); ci.n({5, 7});
        T cv(RAD_F32, {1, 2}); cv.f({0.0f, ln3});
        SP sp(1); sp[0].seed = 99; sp[0].pos = pos;
        T tk(RAD_I32, {1});
        A a; a.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 1).i("n_cand", 2);
        CHECK_OK(run("sample_pick", a));
        const float u = rad_sample_uniform01(99 ^ RAD_SAMPLE_KEY_PICK, pos);
        const int64_t want = u < 0.25f ? 5 : 7;
        if (std::fabs(u - 0.25f) > 1e-5f) { CHECK_EQ(tk.iat(0), want); ++agree; }
    }
    CHECK(agree > 60);
}

/* ================================================================== failure modes */
/* Nothing that did not do the work returns RAD_OK. These are the cases the engine turns into a
 * named abort rather than a silent fallback (spec §17). */
TEST(errors_are_values) {
    T x(RAD_F32, {4});
    A a; a.t(x).i("M", 1).i("n", 0).d("eps", 0).s("dtype", "f32");
    CHECK(run("rmsnorm", a) < 0);              /* n = 0 is a shape, not a no-op */

    A b; b.i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_EQ(run("rmsnorm", b), RAD_E_INVAL);  /* no operands at all */

    /* A rotation wider than the reference will do in one frame says so by name instead of
     * rotating a narrower block. */
    T big(RAD_F32, {1, 8192});
    T q(RAD_I8, {1, 8192}), s(RAD_F32, {1});
    A c; c.t(big).t(q).t(s).i("M", 1).i("n", 8192).i("group", 8192);
    CHECK_EQ(run("had_quant_act_i8", c), RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()

/* scatter_rows: the three rules that distinguish it from the gather it mirrors. Each one exists
 * because the destination is LIVE MEMORY -- a KV pool other sequences are in -- rather than a
 * fresh buffer the op owns. */
TEST(op_scatter_rows_writes_only_the_rows_it_is_given) {
    T v(RAD_F32, {2, 3});   v.f({ 10, 11, 12,  20, 21, 22 });
    T ix(RAD_I32, {2});     ix.n({ 3, 0 });
    T x(RAD_F32, {4, 3});   x.f({ 1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4 });

    A a; a.t(v).t(ix).t(x).i("M", 2).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("scatter_rows", a));

    CHECK_EQ(x.at(0), 20.0f);  CHECK_EQ(x.at(2), 22.0f);    /* row 0 <- v[1] */
    CHECK_EQ(x.at(9), 10.0f);  CHECK_EQ(x.at(11), 12.0f);   /* row 3 <- v[0] */
    /* THE UNNAMED ROWS ARE UNTOUCHED. This is the property the whole op is shaped around: rows 1
     * and 2 belong to somebody else and a scatter that rewrote or zeroed them would be silently
     * destroying another sequence's context. */
    CHECK_EQ(x.at(3), 2.0f);   CHECK_EQ(x.at(5), 2.0f);
    CHECK_EQ(x.at(6), 3.0f);   CHECK_EQ(x.at(8), 3.0f);
}

TEST(op_scatter_rows_skips_a_negative_index_rather_than_zeroing) {
    /* gather_rows reads a ZERO ROW for a negative index -- the padding convention a short prefill
     * chunk leaves behind. The mirror of that here would be a zero FILL, which against a live pool
     * is data loss and not padding, so the row is left alone instead. */
    T v(RAD_F32, {2, 2});   v.f({ 7, 7,  8, 8 });
    T ix(RAD_I32, {2});     ix.n({ -1, 1 });
    T x(RAD_F32, {3, 2});   x.f({ 5, 5,  6, 6,  9, 9 });

    A a; a.t(v).t(ix).t(x).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("scatter_rows", a));
    CHECK_EQ(x.at(0), 5.0f);   CHECK_EQ(x.at(1), 5.0f);   /* nothing written, not zeroed */
    CHECK_EQ(x.at(2), 8.0f);   CHECK_EQ(x.at(3), 8.0f);
    CHECK_EQ(x.at(4), 9.0f);   CHECK_EQ(x.at(5), 9.0f);
}

TEST(op_scatter_rows_refuses_an_index_past_the_destination) {
    /* The same rule gather_rows has, for the same reason turned around: reading past the operand
     * would be reading another sequence's state, and writing past it would be corrupting one. */
    T v(RAD_F32, {1, 2});   v.f({ 1, 2 });
    T ix(RAD_I32, {1});     ix.n({ 3 });
    T x(RAD_F32, {3, 2});   x.f({ 0, 0,  0, 0,  0, 0 });
    A a; a.t(v).t(ix).t(x).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_EQ(run("scatter_rows", a), RAD_E_INVAL);
}

TEST(op_scatter_rows_round_trips_opaque_bytes_through_u8) {
    /* THE DTYPE THE TIER TRAFFIC ACTUALLY USES. A KV fragment is whatever the attention kernel
     * wrote, so it is described as bytes and never interpreted -- gather it out, scatter it back,
     * and every byte has to survive. u8 is what makes that checkable against this reference at
     * all: ref's float path carries 0..255 exactly, where a bf16 NaN payload need not survive. */
    const int n = 8, rows = 4;
    T src(RAD_U8, {rows, n});
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i)
            st_dt(src.t.dtype, src.t.data, r * n + i, (float)((r * 37 + i * 11) & 0xff));

    T ix(RAD_I32, {2});   ix.n({ 2, 0 });
    T packed(RAD_U8, {2, n});
    A g; g.t(src).t(ix).t(packed).i("M", 2).i("n", n).s("dtype", "u8");
    CHECK_OK(run("gather_rows", g));

    T dst(RAD_U8, {rows, n});
    for (int i = 0; i < rows * n; ++i) st_dt(dst.t.dtype, dst.t.data, i, 0.0f);
    A s; s.t(packed).t(ix).t(dst).i("M", 2).i("n", n).s("dtype", "u8");
    CHECK_OK(run("scatter_rows", s));

    for (int i = 0; i < n; ++i) {
        CHECK_EQ(dst.at(2 * n + i), src.at(2 * n + i));   /* the two gathered rows came back */
        CHECK_EQ(dst.at(0 * n + i), src.at(0 * n + i));
        CHECK_EQ(dst.at(1 * n + i), 0.0f);                /* and the others are still untouched */
        CHECK_EQ(dst.at(3 * n + i), 0.0f);
    }
}
