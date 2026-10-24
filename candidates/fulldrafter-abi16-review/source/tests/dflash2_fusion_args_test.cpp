/* tests/dflash2_fusion_args_test.cpp -- the HOST-SIDE argument contracts of the three DFlash2
 * fusion candidates.
 *
 * WHY THIS FILE EXISTS. `rad_dflash_*_args` is the only part of these candidates that runs on the
 * host, and it is the part that decides whether a geometry, a stride, an alias or a narrowing is
 * legal. Nothing there dereferences a pointer, so a real call can be assembled from host arrays
 * and accepted or refused without a device, a stream or a launch. This test does exactly that: it
 * exercises a valid call per candidate and checks the validated call struct's geometry against
 * what was passed, then mutates ONE property at a time to prove the specific refusal. The k-store
 * case also re-accepts the same call with a bf16 gain, because that field is read from the
 * OPERAND rather than from a parameter.
 *
 * NO MOCK LAUNCH, NO MOCK SUCCESS. The operands are bounded host arrays. A valid case claims only
 * that validation accepted the call and described it correctly; it makes no claim about numerics,
 * which is what tests/dflash2_fusion_gpu_test.cpp is for.
 *
 * THE GATED OFFSET BOUND IS DECIDED FIRST, FROM METADATA. `product_fits(N, K)` divides rather than
 * multiplying and runs before any extent is read (mine/dflash_gated_bf16.cpp:173). The overflow
 * case therefore passes metadata-valid tensors with N = 2^60 and K = 16: N*K does not fit int64, so
 * a validator that formed the product would sign-overflow under UBSan, and one that skipped the
 * guard would accept a geometry whose 32-bit element offsets wrap in the kernel.
 *
 * Link: the three candidate .cpp files and the ABI headers only -- no core, no HIP, no device.
 */
#include "rad_test.h"

#include "dflash_conv_add.h"
#include "dflash_gated_bf16.h"
#include "dflash_k_norm_rope_store.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace {

RadTensor dense(void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t v : shape) t.shape[i++] = v;
    rad_tensor_pack(&t);
    return t;
}

RadTensor strided(void* p, uint32_t dt, std::initializer_list<int64_t> shape,
                  std::initializer_list<int64_t> stride) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t v : shape) t.shape[i++] = v;
    i = 0;
    for (int64_t v : stride) t.stride[i++] = v;
    return t;
}

RadArgs args(const RadTensor* t, int n_t, const RadParam* p, int n_p) {
    RadArgs a{};
    a.t = t;
    a.n_t = n_t;
    a.p = p;
    a.n_p = n_p;
    return a;
}

/* One entry of a parameter list, mutated in place by the cases below. */
void set_int(std::vector<RadParam>& p, const char* key, long long v) {
    for (auto& e : p)
        if (e.key && std::strcmp(e.key, key) == 0) { e.ival = v; return; }
    std::abort();
}

void set_f64(std::vector<RadParam>& p, const char* key, double v) {
    for (auto& e : p)
        if (e.key && std::strcmp(e.key, key) == 0) { e.dval = v; return; }
    std::abort();
}

void set_str(std::vector<RadParam>& p, const char* key, const char* v) {
    for (auto& e : p)
        if (e.key && std::strcmp(e.key, key) == 0) { e.sval = v; return; }
    std::abort();
}

std::vector<RadParam> params_conv(int64_t T) {
    return { RAD_INT("T", T), RAD_INT("hidden_size", 64), RAD_INT("taps", 2),
             RAD_INT("group_size", 16), RAD_INT("NG", 4), RAD_INT("block_size", 4),
             RAD_STR("dtype", "bf16") };
}

std::vector<RadParam> params_k_store(int64_t M) {
    return { RAD_INT("M", M), RAD_INT("n_head_kv", 2), RAD_INT("head_dim", 128),
             RAD_INT("block_size", 4), RAD_F64("eps", 1e-6), RAD_F64("wadd", 0.0),
             RAD_F64("theta", 10000.0), RAD_F64("scale", 1.0),
             RAD_STR("kv_dtype", "bf16") };
}

std::vector<RadParam> params_gated(int64_t M, int64_t N, int64_t K) {
    return { RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", "bf16") };
}

/* ================================================================ dflash_conv_add
 * Geometry: hidden_size 64, taps 2, NG 4, group_size 16, block_size 4, 2 rows of a T=8 band.
 * `delta` is the SLICE of the [T, 2, taps, NG] projection, so its row pitch is 16 and its
 * width 8. */
TEST(conv_add_accepts_and_describes_the_valid_call) {
    alignas(64) uint16_t x[128], delta[32], base[128], res[128], out[128];
    RadTensor t[5] = {
        dense(x,     RAD_BF16, {2, 64}),
        strided(delta, RAD_BF16, {2, 8}, {16, 1}),
        dense(base,  RAD_BF16, {2, 64}),
        dense(res,   RAD_BF16, {2, 64}),
        dense(out,   RAD_BF16, {2, 64}),
    };
    std::vector<RadParam> p = params_conv(8);
    RadArgs a = args(t, 5, p.data(), (int)p.size());

    RadDflashConvAddCall call{};
    REQUIRE_EQ(rad_dflash_conv_add_args(&a, &call), RAD_OK);
    CHECK_EQ(call.rows, 2);
    CHECK_EQ(call.H, 64);
    CHECK_EQ(call.NG, 4);
    CHECK_EQ(call.dpitch, 16);
    CHECK_EQ(call.x_pitch, 64);
    CHECK_EQ(call.r_pitch, 64);
    CHECK_EQ(call.o_pitch, 64);
    CHECK_EQ(call.block_size, 4);

    /* The in-place form the graph issues: out IS residual, same base and same pitch. Accepting it
     * is the outcome that matters; the alias refusal cases below prove the rule has teeth. */
    t[RAD_DFLASH_CONV_ADD_OUT] = dense(res, RAD_BF16, {2, 64});
    CHECK_EQ(rad_dflash_conv_add_args(&a, &call), RAD_OK);
}

TEST(conv_add_refuses_geometry_dtype_stride_alias_and_narrowing) {
    alignas(64) uint16_t x[1024], delta[64], base[1024], res[1024], out[1024];
    const RadTensor good[5] = {
        dense(x,     RAD_BF16, {2, 64}),
        strided(delta, RAD_BF16, {2, 8}, {16, 1}),
        dense(base,  RAD_BF16, {2, 64}),
        dense(res,   RAD_BF16, {2, 64}),
        dense(out,   RAD_BF16, {2, 64}),
    };
    const std::vector<RadParam> p = params_conv(8);
    RadDflashConvAddCall call{};
    RadTensor t[5];

    auto try_call = [&](std::vector<RadParam>& pp) {
        RadArgs a = args(t, 5, pp.data(), (int)pp.size());
        return rad_dflash_conv_add_args(&a, &call);
    };
    auto fresh = [&] { std::copy(std::begin(good), std::end(good), t); };
    std::vector<RadParam> q;

    /* outside taps 2 / group 16 / the 64-divisible width band / NG == H/group */
    fresh(); q = p; set_int(q, "taps", 1);        CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);
    fresh(); q = p; set_int(q, "group_size", 8);  CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);
    fresh(); q = p; set_int(q, "hidden_size", 48); CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);
    fresh(); q = p; set_int(q, "NG", 3);          CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);

    /* the dtype is a string AND on every operand */
    fresh(); q = p; set_str(q, "dtype", "f32");   CHECK_EQ(try_call(q), RAD_E_DTYPE);
    fresh(); t[RAD_DFLASH_CONV_ADD_X].dtype = RAD_F32;
    CHECK_EQ(try_call(q = p), RAD_E_DTYPE);

    /* THE PITCH NO SHAPE CHECK WOULD CATCH: a packed delta slice reads the other half's
     * coefficients (8 instead of the projection's 16). */
    fresh(); t[RAD_DFLASH_CONV_ADD_DELTA] = dense(delta, RAD_BF16, {2, 2, 2, 2});
    CHECK_EQ(try_call(q = p), RAD_E_STRIDE);

    /* a negative pitch is never a view */
    fresh(); t[RAD_DFLASH_CONV_ADD_X] = strided(x, RAD_BF16, {2, 64}, {-64, 1});
    CHECK_EQ(try_call(q = p), RAD_E_STRIDE);

    /* rows past the T band, and an extent that disagrees with hidden_size */
    fresh();
    t[RAD_DFLASH_CONV_ADD_X]        = dense(x,   RAD_BF16, {16, 64});
    t[RAD_DFLASH_CONV_ADD_RESIDUAL] = dense(res, RAD_BF16, {16, 64});
    t[RAD_DFLASH_CONV_ADD_OUT]      = dense(out, RAD_BF16, {16, 64});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);
    fresh(); t[RAD_DFLASH_CONV_ADD_RESIDUAL] = dense(res, RAD_BF16, {2, 32});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);

    /* NARROWING: the launch stores the pitch in an int, so > INT32_MAX is refused, not truncated */
    fresh(); t[RAD_DFLASH_CONV_ADD_X] =
        strided(x, RAD_BF16, {2, 64}, {(int64_t)INT32_MAX + 8, 1});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);

    /* ALIAS: out may be residual, must not touch x or the coefficients, and a partial overlap of
     * residual is not "in place" either. */
    fresh(); t[RAD_DFLASH_CONV_ADD_OUT] = dense(x, RAD_BF16, {2, 64});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
    fresh(); t[RAD_DFLASH_CONV_ADD_OUT] = dense(res + 16, RAD_BF16, {2, 64});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
    fresh();
    t[RAD_DFLASH_CONV_ADD_RESIDUAL] = dense(res, RAD_BF16, {2, 64});
    t[RAD_DFLASH_CONV_ADD_OUT]      = dense(res, RAD_BF16, {2, 64});
    t[RAD_DFLASH_CONV_ADD_OUT].stride[0] = 128;   /* same base, different pitch */
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
    fresh(); t[RAD_DFLASH_CONV_ADD_OUT] = dense(base, RAD_BF16, {2, 64});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);

    /* ALIGNMENT: a row the 16-byte loads cannot address */
    fresh(); t[RAD_DFLASH_CONV_ADD_OUT] = dense((uint8_t*)out + 2, RAD_BF16, {2, 64});
    CHECK_EQ(try_call(q = p), RAD_E_ALIGN);
    fresh(); t[RAD_DFLASH_CONV_ADD_X] = strided(x, RAD_BF16, {2, 64}, {68, 1});
    CHECK_EQ(try_call(q = p), RAD_E_ALIGN);
}

/* ====================================================== dflash_k_norm_rope_store
 * Geometry: 2 tokens, n_head_kv 2, head_dim 128, block_size 4 over 2 blocks, f32 gain. */
TEST(k_store_accepts_and_describes_the_valid_call) {
    alignas(64) uint16_t k[512], v[512], kv[4096];
    alignas(64) float w[128];
    alignas(64) int32_t pos[2] = {5, 9}, slot[2] = {1, 6};
    RadTensor t[6] = {
        dense(k,    RAD_BF16, {2, 256}),
        dense(v,    RAD_BF16, {2, 256}),
        dense(w,    RAD_F32,  {128}),
        dense(pos,  RAD_I32,  {2}),
        dense(slot, RAD_I32,  {2}),
        dense(kv,   RAD_BF16, {2, 2, 4, 256}),
    };
    std::vector<RadParam> p = params_k_store(4);
    RadArgs a = args(t, 6, p.data(), (int)p.size());

    RadDflashKStoreCall call{};
    REQUIRE_EQ(rad_dflash_k_store_args(&a, &call), RAD_OK);
    CHECK_EQ(call.rows, 2);
    CHECK_EQ(call.head_dim, 128);
    CHECK_EQ(call.n_head_kv, 2);
    CHECK_EQ(call.block_size, 4);
    CHECK_EQ(call.n_blocks, 2);
    CHECK_EQ(call.k_pitch, 256);
    CHECK_EQ(call.v_pitch, 256);
    CHECK_EQ(call.w_f32, 1);
    CHECK_EQ(call.eps, 1e-6f);
    CHECK_EQ(call.theta, 10000.0f);
    CHECK_EQ(call.rscale, 1.0f);

    /* The gain's width is the OPERAND's, not a parameter: the same call with a bf16 gain is
     * accepted and says so. */
    alignas(64) uint16_t wb[128];
    t[RAD_DKNS_W] = dense(wb, RAD_BF16, {128});
    CHECK_EQ(rad_dflash_k_store_args(&a, &call), RAD_OK);
    CHECK_EQ(call.w_f32, 0);
}

TEST(k_store_refuses_geometry_dtype_stride_alias_and_narrowing) {
    alignas(64) uint16_t k[2048], v[2048], kv[4096];
    alignas(64) float w[128];
    alignas(64) int32_t pos[2] = {5, 9}, slot[2] = {1, 6};
    const RadTensor good[6] = {
        dense(k,    RAD_BF16, {2, 256}),
        dense(v,    RAD_BF16, {2, 256}),
        dense(w,    RAD_F32,  {128}),
        dense(pos,  RAD_I32,  {2}),
        dense(slot, RAD_I32,  {2}),
        dense(kv,   RAD_BF16, {2, 2, 4, 256}),
    };
    const std::vector<RadParam> p = params_k_store(4);
    RadDflashKStoreCall call{};
    RadTensor t[6];

    auto try_call = [&](std::vector<RadParam>& pp) {
        RadArgs a = args(t, 6, pp.data(), (int)pp.size());
        return rad_dflash_k_store_args(&a, &call);
    };
    auto fresh = [&] { std::copy(std::begin(good), std::end(good), t); };
    std::vector<RadParam> q;

    fresh(); q = p; set_int(q, "head_dim", 64);   CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);
    fresh(); q = p; set_str(q, "kv_dtype", "f16"); CHECK_EQ(try_call(q), RAD_E_DTYPE);
    fresh(); q = p; set_f64(q, "theta", 0.0);     CHECK_EQ(try_call(q), RAD_E_INVAL);

    fresh(); t[RAD_DKNS_K].dtype = RAD_F32;       CHECK_EQ(try_call(q = p), RAD_E_DTYPE);
    fresh(); t[RAD_DKNS_W].dtype = RAD_I32;       CHECK_EQ(try_call(q = p), RAD_E_DTYPE);
    fresh(); t[RAD_DKNS_POS].dtype = RAD_F32;     CHECK_EQ(try_call(q = p), RAD_E_DTYPE);

    /* rank before extents: a rank-3 k is not the plane this row serves */
    fresh(); t[RAD_DKNS_K] = dense(k, RAD_BF16, {2, 2, 128});
    CHECK_EQ(try_call(q = p), RAD_E_UNSUPPORTED);

    /* rows past the M band (band = 1, two rows presented) */
    fresh();
    q = p; set_int(q, "M", 1);
    CHECK_EQ(try_call(q), RAD_E_SHAPE);
    fresh(); t[RAD_DKNS_POS] = dense(pos, RAD_I32, {1});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);

    /* a column step other than 1 */
    fresh(); t[RAD_DKNS_K] = strided(k, RAD_BF16, {2, 256}, {256, 2});
    CHECK_EQ(try_call(q = p), RAD_E_STRIDE);
    /* a pitched multi-head plane: the flat norm enumeration would walk the next head's bytes */
    fresh(); t[RAD_DKNS_K] = strided(k, RAD_BF16, {2, 256}, {264, 1});
    CHECK_EQ(try_call(q = p), RAD_E_STRIDE);
    /* the cache is walked packed, with an inner extent of 2*head_dim */
    fresh(); t[RAD_DKNS_KV] = strided(kv, RAD_BF16, {2, 2, 4, 256}, {4096, 2048, 512, 2});
    CHECK_EQ(try_call(q = p), RAD_E_STRIDE);
    fresh(); t[RAD_DKNS_KV] = dense(kv, RAD_BF16, {2, 2, 4, 128});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);

    /* NARROWING: block_size is stored in an int, so a value past INT32_MAX is refused rather than
     * wrapped -- and it is refused before any cache extent could produce the same code. */
    fresh();
    q = p; set_int(q, "block_size", (long long)INT32_MAX + 1);
    CHECK_EQ(try_call(q), RAD_E_SHAPE);

    /* ALIAS: k is written in place, so neither the cache nor v may be k's bytes */
    fresh(); t[RAD_DKNS_K] = dense(kv, RAD_BF16, {2, 256});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
    fresh(); t[RAD_DKNS_V] = dense(k, RAD_BF16, {2, 256});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
}

/* ============================================================ dflash_gated_bf16
 * Geometry: 2 rows of an M=8 band, gate_up N 2048, K 16. */
TEST(gated_accepts_and_describes_the_valid_call) {
    alignas(64) uint16_t x[32], w[2048 * 16], y[2 * 1024];
    RadTensor t[3] = {
        dense(x, RAD_BF16, {2, 16}),
        dense(w, RAD_BF16, {2048, 16}),
        dense(y, RAD_BF16, {2, 1024}),
    };
    std::vector<RadParam> p = params_gated(8, 2048, 16);
    RadArgs a = args(t, 3, p.data(), (int)p.size());

    RadDflashGatedCall call{};
    REQUIRE_EQ(rad_dflash_gated_bf16_args(&a, &call), RAD_OK);
    CHECK_EQ(call.M, 2);
    CHECK_EQ(call.K, 16);
    CHECK_EQ(call.N, 2048);
    CHECK_EQ(call.n, 1024);
}

TEST(gated_refuses_geometry_dtype_stride_alias_and_narrowing) {
    alignas(64) uint16_t x[32], w[2048 * 16], y[2 * 1024];
    const RadTensor good[3] = {
        dense(x, RAD_BF16, {2, 16}),
        dense(w, RAD_BF16, {2048, 16}),
        dense(y, RAD_BF16, {2, 1024}),
    };
    const std::vector<RadParam> p = params_gated(8, 2048, 16);
    RadDflashGatedCall call{};
    RadTensor t[3];

    auto try_call = [&](std::vector<RadParam>& pp) {
        RadArgs a = args(t, 3, pp.data(), (int)pp.size());
        return rad_dflash_gated_bf16_args(&a, &call);
    };
    auto fresh = [&] { std::copy(std::begin(good), std::end(good), t); };
    std::vector<RadParam> q;

    /* an odd N is a shape error; the declared bands are UNSUPPORTED */
    fresh(); q = p; set_int(q, "N", 2047);  CHECK_EQ(try_call(q), RAD_E_SHAPE);
    fresh(); q = p; set_int(q, "N", 1024);  CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);
    fresh(); q = p; set_int(q, "N", 2064);  CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);
    fresh(); q = p; set_int(q, "K", 24);    CHECK_EQ(try_call(q), RAD_E_UNSUPPORTED);

    fresh(); q = p; set_str(q, "dtype", "f16"); CHECK_EQ(try_call(q), RAD_E_DTYPE);
    fresh(); t[RAD_DFLASH_GATED_Y].dtype = RAD_F32;
    CHECK_EQ(try_call(q = p), RAD_E_DTYPE);

    /* rows past the M band (64), and extents that disagree with the parameters */
    fresh();
    t[RAD_DFLASH_GATED_A] = dense(x, RAD_BF16, {65, 16});
    t[RAD_DFLASH_GATED_Y] = dense(y, RAD_BF16, {65, 1024});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);
    fresh(); t[RAD_DFLASH_GATED_A] = dense(x, RAD_BF16, {2, 32});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);
    fresh(); t[RAD_DFLASH_GATED_Y] = dense(y, RAD_BF16, {2, 512});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);
    fresh(); t[RAD_DFLASH_GATED_B] = dense(w, RAD_BF16, {2048, 8});
    CHECK_EQ(try_call(q = p), RAD_E_SHAPE);

    /* a pitched A is not the contiguous operand the fragments address */
    fresh(); t[RAD_DFLASH_GATED_A] = strided(x, RAD_BF16, {2, 16}, {32, 1});
    CHECK_EQ(try_call(q = p), RAD_E_STRIDE);
    fresh(); t[RAD_DFLASH_GATED_B] = dense((uint8_t*)w + 2, RAD_BF16, {2048, 16});
    CHECK_EQ(try_call(q = p), RAD_E_ALIGN);

    /* all three pointers are __restrict__ in the kernel */
    fresh(); t[RAD_DFLASH_GATED_Y] = dense(w, RAD_BF16, {2, 1024});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
    fresh(); t[RAD_DFLASH_GATED_A] = dense(w, RAD_BF16, {2, 16});
    CHECK_EQ(try_call(q = p), RAD_E_INVAL);
}

/* THE 32-BIT ELEMENT-OFFSET BOUND IS DECIDED FROM METADATA, BEFORE ANY PRODUCT IS FORMED.
 *
 * The tensors below carry valid METADATA: rank 2, bf16, extents consistent with M, K and N,
 * strides spelled out densely. The backing arrays stay tiny, because rad_dflash_gated_bf16_args
 * dereferences no operand and the refusal must come from the parameters alone.
 *
 * N = 2^60, K = 16. N*K is 2^64, past INT64_MAX: a validator that formed the product before
 * bounding it would sign-overflow (UBSan traps). K is small on purpose -- `rows * K` still fits,
 * so every later extent guard in the function passes -- which makes product_fits(N, K) the ONE
 * check that can refuse this call. Remove it and validation reaches `const int64_t w_need =
 * N * K` (mine/dflash_gated_bf16.cpp:197) with the same UB, whether or not the row is accepted.
 *
 * That the predicate bounds the PRODUCT rather than either dimension is pinned by the accepted
 * call in gated_accepts_and_describes_the_valid_call (N = 2048, K = 16 -> 32768).
 */
TEST(gated_offset_bound_refuses_oversized_product_before_any_data_access) {
    alignas(64) uint16_t x[64], w[64], y[64];
    const int64_t N = INT64_C(1) << 60;
    const int64_t K = 16;
    RadDflashGatedCall call{};

    RadTensor t[3] = {
        dense(x,  RAD_BF16, {1, K}),
        strided(w, RAD_BF16, {N, K}, {K, 1}),
        dense(y,  RAD_BF16, {1, N / 2}),
    };
    std::vector<RadParam> p = params_gated(2, N, K);
    RadArgs a = args(t, 3, p.data(), (int)p.size());

    CHECK_EQ(rad_dflash_gated_bf16_args(&a, &call), RAD_E_SHAPE);
}

}  /* namespace */

RAD_TEST_MAIN()
