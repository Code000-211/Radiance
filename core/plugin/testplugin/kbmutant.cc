/* kbmutant.cc -- kernel libraries that are wrong on purpose, for tests/kbench_mutant_test.cpp.
 *
 * rad-kbench's verdicts are only worth something if a wrong kernel fails them. Each build of this
 * file (-DKBM_MUTANT=..., see CMakeLists.txt) is one library implementing a few libref ops, with
 * one defect: an output scaled past its tolerance, a NaN, an unwritten half, a write one byte past
 * an output, a refusal of a geometry its constraints accept, a description of an output that
 * disagrees with the reference's, an id off by one, a write into an input, and a preferred row that
 * cannot read its weight's encoding. KBM_OK is the same library without a defect.
 *
 * It declares no schemas: libref's own fix the ops. Device-domain rows run on the host backend's
 * memory, after draining the stream, so the guarded allocations and their margins are real; the
 * gather and scatter device rows instead queue their work on the stream as copies, which the
 * timing pass's gate can hold. Like
 * tierrows.cc, a computing library registered in the device domain must never be found by a real
 * run, so it lives under testkernels/ and is loaded only by name.
 */
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#define NELEM(a) (int)(sizeof(a) / sizeof((a)[0]))

#define KBM_OK      0
#define KBM_SCALE   1
#define KBM_NAN     2
#define KBM_HALF    3
#define KBM_OVERRUN 4
#define KBM_REFUSE  5
#define KBM_SHAPE   6
#define KBM_ARGMAX  7
#define KBM_INWRITE 8
#define KBM_ENC     9
#define KBM_PAIR    10

#ifndef KBM_MUTANT
#define KBM_MUTANT KBM_OK
#endif
#ifndef KBM_NAME
#define KBM_NAME "kbm_ok"
#endif

namespace {

int64_t numel(const RadTensor& t) { return rad_tensor_numel(&t); }

/* A device row runs where a device launch would: after everything enqueued before it. */
int order(RadStream s) { return s && rad_stream_sync(s) != RAD_OK ? RAD_E_DEVICE : RAD_OK; }

/* y = a + b, with this library's defect. */
int add_impl(const RadArgs* a, RadStream s, bool device) {
    if (order(s) != RAD_OK) return RAD_E_DEVICE;
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    if (KBM_MUTANT == KBM_REFUSE) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &y = a->t[1], &o = a->t[2];
    const int64_t n = numel(o);
    for (int64_t i = 0; i < n; ++i) {
        if (KBM_MUTANT == KBM_HALF && i >= n / 2) break;
        float v = rad_load_f32(x.data, x.dtype, i) + rad_load_f32(y.data, y.dtype, i);
        /* Three times the bf16 tolerance (6e-3). */
        if (KBM_MUTANT == KBM_SCALE) v *= 1.018f;
        rad_store_f32(o.data, o.dtype, i, v);
    }
    if (KBM_MUTANT == KBM_NAN) rad_store_f32(o.data, o.dtype, n / 3, std::nanf(""));
    if (KBM_MUTANT == KBM_OVERRUN && device)
        ((uint8_t*)o.data)[rad_dtype_bytes(o.dtype, n)] = 0x5A;
    if (KBM_MUTANT == KBM_INWRITE && !device)
        rad_store_f32(x.data, x.dtype, 0, rad_load_f32(x.data, x.dtype, 0) + 1.0f);
    return RAD_OK;
}
int add_host(const RadArgs* a, RadStream s)   { return add_impl(a, s, false); }
int add_device(const RadArgs* a, RadStream s) { return add_impl(a, s, true); }

/* y[i,:] = x[idx[i],:], byte rows. */
int gather_rows(const RadArgs* a, RadStream s) {
    if (order(s) != RAD_OK) return RAD_E_DEVICE;
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &idx = a->t[1], &y = a->t[2];
    if (x.rank != 2 || y.rank != 2 || idx.dtype != RAD_I32) return RAD_E_SHAPE;
    const int64_t row = rad_dtype_bytes(y.dtype, y.shape[1]);
    for (int64_t i = 0; i < y.shape[0]; ++i) {
        const int32_t r = ((const int32_t*)idx.data)[i];
        if (r < 0 || r >= x.shape[0]) return RAD_E_INVAL;
        std::memcpy((uint8_t*)y.data + i * row, (const uint8_t*)x.data + r * row, (size_t)row);
    }
    return RAD_OK;
}

/* The same gather as one stream copy a row: a device row whose work is on its stream, which the
 * timing pass can queue behind a gate (the rows above compute when called, after draining it). It
 * reads `idx` when called, which only the host backend's device memory allows. */
int gather_rows_device(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3 || !s) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &idx = a->t[1], &y = a->t[2];
    if (x.rank != 2 || y.rank != 2 || idx.dtype != RAD_I32) return RAD_E_SHAPE;
    const int64_t row = rad_dtype_bytes(y.dtype, y.shape[1]);
    for (int64_t i = 0; i < y.shape[0]; ++i) {
        const int32_t r = ((const int32_t*)idx.data)[i];
        if (r < 0 || r >= x.shape[0]) return RAD_E_INVAL;
        if (rad_memcpy_async((uint8_t*)y.data + i * row, (const uint8_t*)x.data + r * row, row, s) !=
            RAD_OK)
            return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* x[idx[i],:] = v[i,:], in place: on the host, and as stream copies (see gather_rows_device). */
int scatter_rows(const RadArgs* a, RadStream s) {
    if (order(s) != RAD_OK) return RAD_E_DEVICE;
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &v = a->t[0], &idx = a->t[1], &x = a->t[2];
    if (v.rank != 2 || x.rank != 2 || idx.dtype != RAD_I32) return RAD_E_SHAPE;
    const int64_t row = rad_dtype_bytes(v.dtype, v.shape[1]);
    for (int64_t i = 0; i < v.shape[0]; ++i) {
        const int32_t r = ((const int32_t*)idx.data)[i];
        if (r < 0 || r >= x.shape[0]) return RAD_E_INVAL;
        uint8_t* dst = (uint8_t*)x.data + r * row;
        const uint8_t* src = (const uint8_t*)v.data + i * row;
        /* A destination already holding its value was written by an earlier launch, so `x` was
         * not restored in between: refused, which is how kbench_mutant_test sees that the timing
         * pass restores in-place operands. Drawn rows never match by chance. */
        if (std::memcmp(dst, src, (size_t)row) == 0) return RAD_E_STATE;
        std::memcpy(dst, src, (size_t)row);
    }
    return RAD_OK;
}
int scatter_rows_device(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3 || !s) return RAD_E_INVAL;
    const RadTensor &v = a->t[0], &idx = a->t[1], &x = a->t[2];
    if (v.rank != 2 || x.rank != 2 || idx.dtype != RAD_I32) return RAD_E_SHAPE;
    const int64_t row = rad_dtype_bytes(v.dtype, v.shape[1]);
    for (int64_t i = 0; i < v.shape[0]; ++i) {
        const int32_t r = ((const int32_t*)idx.data)[i];
        if (r < 0 || r >= x.shape[0]) return RAD_E_INVAL;
        if (rad_memcpy_async((uint8_t*)x.data + r * row, (const uint8_t*)v.data + i * row, row, s) !=
            RAD_OK)
            return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* token[m] = the lower id of a row's largest logit. */
int sample_argmax(const RadArgs* a, RadStream s) {
    if (order(s) != RAD_OK) return RAD_E_DEVICE;
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &l = a->t[0], &tok = a->t[2];
    if (l.rank != 2 || l.dtype != RAD_F32 || tok.dtype != RAD_I32) return RAD_E_DTYPE;
    for (int64_t m = 0; m < l.shape[0]; ++m) {
        const float* row = (const float*)l.data + m * l.shape[1];
        int32_t best = 0;
        for (int64_t v = 1; v < l.shape[1]; ++v) if (row[v] > row[best]) best = (int32_t)v;
        if (KBM_MUTANT == KBM_ARGMAX) best += 1;
        ((int32_t*)tok.data)[m] = best;
    }
    return RAD_OK;
}

/* y = x * rsqrt(mean(x^2) + eps) * (w + wadd), as libref computes it. */
int rmsnorm(const RadArgs* a, RadStream s) {
    if (order(s) != RAD_OK) return RAD_E_DEVICE;
    if (!a || a->n_t < 3 || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const int64_t n = rad_args_geti_or(a, "n", 0);
    const double eps = rad_args_getf_or(a, "eps", 1e-6);
    const double wadd = rad_args_getf_or(a, "wadd", 0.0);
    if (n <= 0 || numel(x) % n) return RAD_E_SHAPE;
    for (int64_t r = 0; r < numel(x) / n; ++r) {
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float v = rad_load_f32(x.data, x.dtype, r * n + i);
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / (float)n + (float)eps);
        for (int64_t i = 0; i < n; ++i)
            rad_store_f32(y.data, y.dtype, r * n + i,
                          rad_load_f32(x.data, x.dtype, r * n + i) * inv *
                              (rad_load_f32(w.data, w.dtype, i) + (float)wadd));
    }
    return RAD_OK;
}

/* The pair's device row: the same add, off by a rounding-sized 1e-4 instead of by a defect, which
 * is inside bf16's tolerance. It exists so that one library holds both a definition and a kernel
 * (see rad_kernel_reference below): the check of the device row must apply its tolerance, not the
 * exactness of the library that ships the definition. */
int pair_add_off(const RadArgs* a, RadStream s) {
    if (order(s) != RAD_OK) return RAD_E_DEVICE;
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &y = a->t[1], &o = a->t[2];
    const int64_t n = numel(o);
    for (int64_t i = 0; i < n; ++i)
        rad_store_f32(o.data, o.dtype, i,
                      (rad_load_f32(x.data, x.dtype, i) + rad_load_f32(y.data, y.dtype, i)) *
                          1.0001f);
    return RAD_OK;
}

/* A preferred row that reads no encoding a recording carries: its layout hook refuses, so the
 * engine passes it over, and its launch would fail the comparison if a replay chose it anyway. */
[[maybe_unused]] int rmsnorm_wrong(const RadArgs* a, RadStream s) {
    const int rc = rmsnorm(a, s);
    if (rc == RAD_OK) rad_store_f32(a->t[2].data, a->t[2].dtype, 0, 1e6f);
    return rc;
}
[[maybe_unused]] int refuse_layout(const RadParam*, int, int, const RadEncoding*, const int*,
                                   const RadTensor*, int, RadLayout*) {
    return RAD_E_DTYPE;
}

/* `add` described with an output one column wider than the reference describes it. */
[[maybe_unused]] int add_shape_wide(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (operand < 0 || operand > 2) return RAD_E_SHAPE;
    std::memset(out, 0, sizeof *out);
    out->idx_const = -1;
    out->dtype = RAD_BF16;
    out->rank = 2;
    out->shape[0] = rad_param_getdim(p, n_p, "M", 0);
    out->shape[1] = rad_param_getdim(p, n_p, "n", 0) + (operand == 2 ? 1 : 0);
    return RAD_OK;
}

const RadConstraint kSmall[] = { RAD_CLE("M", 4) };

}  /* namespace */

extern "C" {

#if KBM_MUTANT == KBM_SHAPE
#define KBM_ADD_SHAPE add_shape_wide
#else
#define KBM_ADD_SHAPE nullptr
#endif

static const RadKernelInfo kKernels[] = {
#if KBM_MUTANT != KBM_PAIR
    { .name = "kbm_add", .op = "add", .family = "elem", .computes = "y = a + b",
      .shape = "any", .dtypes = "bf16", .domain = RAD_DOMAIN_HOST, .priority = 0,
      .launch = add_host, .opd_shape = KBM_ADD_SHAPE },
    { .name = "kbm_add_device", .op = "add", .family = "elem", .computes = "y = a + b",
      .shape = "any", .dtypes = "bf16", .domain = RAD_DOMAIN_DEVICE, .priority = 0,
      .launch = add_device, .opd_shape = KBM_ADD_SHAPE },
#endif
    /* Reached only at M <= 4, which the recorded cases do not use: --record-rows' target. */
    { .name = "kbm_add_small", .op = "add", .family = "elem", .computes = "y = a + b",
      .shape = "M <= 4", .dtypes = "bf16", .domain = RAD_DOMAIN_HOST, .priority = 5,
      .constraints = kSmall, .n_constraints = NELEM(kSmall), .launch = add_host,
      .opd_shape = KBM_ADD_SHAPE },
    { .name = "kbm_gather_rows", .op = "gather_rows", .family = "elem", .computes = "byte rows",
      .shape = "any", .dtypes = "any", .domain = RAD_DOMAIN_HOST, .priority = 0,
      .launch = gather_rows },
    { .name = "kbm_gather_rows_device", .op = "gather_rows", .family = "elem",
      .computes = "byte rows, one stream copy each", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 0, .launch = gather_rows_device },
    { .name = "kbm_scatter_rows", .op = "scatter_rows", .family = "elem", .computes = "byte rows",
      .shape = "any", .dtypes = "any", .domain = RAD_DOMAIN_HOST, .priority = 0,
      .launch = scatter_rows },
    { .name = "kbm_scatter_rows_device", .op = "scatter_rows", .family = "elem",
      .computes = "byte rows, one stream copy each", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 0, .launch = scatter_rows_device },
    { .name = "kbm_sample_argmax", .op = "sample_argmax", .family = "sample",
      .computes = "greedy", .shape = "any", .dtypes = "f32", .domain = RAD_DOMAIN_HOST,
      .priority = 0, .launch = sample_argmax },
#if KBM_MUTANT == KBM_ENC
    { .name = "kbm_rmsnorm_wide", .op = "rmsnorm", .family = "norm",
      .computes = "reads a layout no recording carries", .shape = "any", .dtypes = "bf16",
      .domain = RAD_DOMAIN_HOST, .priority = 10, .launch = rmsnorm_wrong,
      .layout = refuse_layout },
#endif
    { .name = "kbm_rmsnorm", .op = "rmsnorm", .family = "norm", .computes = "rmsnorm",
      .shape = "any", .dtypes = "bf16", .domain = RAD_DOMAIN_HOST, .priority = 0,
      .launch = rmsnorm },
#if KBM_MUTANT == KBM_PAIR
    /* The definition of `add` beside a kernel of it, in one library: the reference is a row, not a
     * library (plugin/qsatable/exports.cpp states the same case). Only the first of these is marked
     * by rad_kernel_reference below. */
    { .name = "kbm_pair_add", .op = "add", .family = "elem", .computes = "y = a + b",
      .shape = "any", .dtypes = "bf16", .domain = RAD_DOMAIN_HOST, .priority = 0,
      .launch = add_host },
    { .name = "kbm_pair_add_device", .op = "add", .family = "elem",
      .computes = "y = (a + b) * (1 + 1e-4)", .shape = "any", .dtypes = "bf16",
      .domain = RAD_DOMAIN_DEVICE, .priority = 0, .launch = pair_add_off },
#endif
};

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, KBM_NAME, "0.1",
    "a deliberately defective kernel library for rad-kbench's own tests", "host",
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
int                  rad_kernel_schema_count(void) { return 0; }
const RadOpSchema*   rad_kernel_schema_at(int) { return nullptr; }
int                  rad_kernel_count(void) { return NELEM(kKernels); }
const RadKernelInfo* rad_kernel_at(int i) {
    return (i < 0 || i >= NELEM(kKernels)) ? nullptr : &kKernels[i];
}
#if KBM_MUTANT == KBM_PAIR
/* Per row, never per library: the host add is this library's definition of the op and its device
 * add is a kernel to be measured against that definition. */
int rad_kernel_reference(int i) {
    return i >= 0 && i < NELEM(kKernels) && std::strcmp(kKernels[i].name, "kbm_pair_add") == 0;
}
#endif

}  /* extern "C" */
