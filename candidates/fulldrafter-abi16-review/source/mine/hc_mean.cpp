/* hc_mean.cpp -- the schema and the host-side validation for the DFlash2 tap reduction.
 *
 * THE SCHEMA IS libref's, SLOT FOR SLOT. The loader refuses a plugin whose schema for an op
 * disagrees with the one already registered, and arguments are positional, so a capital `N` here
 * and a lower-case `n` there would make the same call mean two things depending on which kernel
 * won selection. `hc_mean` is a new op and spells the width `N`, which is what its caller in
 * arch/qwen4exp_fp8 writes; the gated residual's three ops keep their `n`.
 *
 * THE OPERAND ORDER IS `h` THEN `x`. The output is NOT in-place and the tap plane is a different
 * buffer from the residual stream, so a caller cannot express the call the other way round.
 *
 * THIS FILE IS COMPILED INTO THE DEVICE PLUGIN ONLY. The host oracle is libref's ref_hc_mean, and
 * duplicating it here would be a second reference that rad-kbench could then pick between. What is
 * shared is the SCHEMA and the validation: a device kernel that accepted a call the oracle refuses
 * would be a divergence the oracle cannot see, so both are written once and the row below carries
 * mine's launch entry point in.
 */
#include "hc_mean.h"

#include <cstdint>
#include <cstring>

namespace {

/* The one geometry the device row is compiled for: hc 4, the DFlash2 hyper-connection count, and a
 * width that is a whole number of 8-wide bf16 groups. `N % 8 == 0` is what lets the kernel read
 * one `uint4`-sized group of each stream without a tail branch; N = 2560 is the real call and
 * every multiple of 8 below the alignment bound is the same code. */
constexpr int64_t kHc = 4;
constexpr int64_t kAlign = 8;

/* Elements a tensor occupies, from its OWN shape and strides -- a column view is measured over the
 * columns it names, not over the buffer it is a view of. */
int64_t span_elements(const RadTensor& t) {
    int64_t e = 1;
    for (uint32_t i = 0; i < t.rank; ++i) e += (t.shape[i] - 1) * t.stride[i];
    return e;
}

bool overlaps(const RadTensor& a, const RadTensor& b) {
    if (!a.data || !b.data) return false;
    const uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
    const int64_t ea = span_elements(a), eb = span_elements(b);
    const uint64_t xa = (uint64_t)ea * 2u, xb = (uint64_t)eb * 2u;
    return x < y + xb && y < x + xa;
}

static_assert(RAD_HC_MEAN_OPERANDS == 2, "the schema below lists two operands");

}  /* namespace */

int rad_hc_mean_schema_count(void) { return 1; }

/* ---- schema ---------------------------------------------------------------------------------- */
#define P_INT(k)  { k, RAD_P_INT, RAD_REQUIRED, 0 }
#define P_STR(k)  { k, RAD_P_STR, RAD_REQUIRED, 0 }
#define IN(n)     { n, RAD_OPD_IN,  0 }
#define OUT(n)    { n, RAD_OPD_OUT, 0 }

static const RadParamSpec kParams[] = { P_INT("M"), P_INT("N"), P_INT("hc"), P_STR("dtype") };
static const RadOperandSpec kOperands[] = { IN("h"), OUT("x") };
static const RadOpSchema kSchema = {
    "hc_mean", kParams, (int)(sizeof(kParams) / sizeof(kParams[0])),
    kOperands, (int)(sizeof(kOperands) / sizeof(kOperands[0])),
    "x[m,i] = mean over the hc streams of h[m, c*N+i] -- the capture's own reduction, summed in "
    "f32 and stored bf16 once, with no RMS, gate or weight. The output is normally a column view "
    "of a wider tap plane and is written through its own row stride."
};
#undef P_INT
#undef P_STR
#undef IN
#undef OUT

const RadOpSchema* rad_hc_mean_schema_at(int i) { return i == 0 ? &kSchema : nullptr; }

/* ---- operand description --------------------------------------------------------------------- */
/* `M` is the band ceiling and the operand names the call, exactly as libref's hook says: a tool
 * that sizes its buffers from this gets the band, which is what a test wants, and a launch reads
 * the rows from the tensor. The output is described as the packed [M, N] plane a tool allocates
 * for itself even though a real caller hands a column view of the tap plane -- a description names
 * extents, and a stride is the caller's (libref/ref_shapes.cpp says the same). */
int rad_hc_mean_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= 2) return RAD_E_INVAL;
    const int64_t M  = rad_param_getdim(p, n_p, "M", 0);
    const int64_t n  = rad_param_getdim(p, n_p, "N", 0);
    const int64_t hc = rad_param_getdim(p, n_p, "hc", 0);
    if (M <= 0 || n <= 0 || hc <= 0) return RAD_E_UNSUPPORTED;
    RadOpdDesc d{};
    d.dtype = RAD_BF16;
    d.rank = 2;
    d.shape[0] = M;
    d.shape[1] = operand == RAD_HC_MEAN_IN ? hc * n : n;
    d.fill = (uint32_t)RAD_FILL_NORMAL;
    d.idx_const = -1;
    *out = d;
    return RAD_OK;
}

/* ---- the call ---------------------------------------------------------------------------------- */
int rad_hc_mean_args(const RadArgs* a, RadHcMeanCall* call) {
    if (!a || !call || !a->t || a->n_t != 2) return RAD_E_INVAL;
    const RadTensor& h = a->t[RAD_HC_MEAN_IN];
    const RadTensor& x = a->t[RAD_HC_MEAN_OUT];
    if (!h.data || !x.data) return RAD_E_INVAL;

    const int64_t n  = rad_param_getdim(a->p, a->n_p, "N", 0);
    const int64_t hc = rad_param_getdim(a->p, a->n_p, "hc", 0);
    if (n <= 0 || hc <= 0) return RAD_E_SHAPE;
    /* Every offset this kernel forms is `row*pitch + (c*n + i)`, so the products are bounded here
     * once rather than trusted: an overflowing extent is a caller bug that the device would turn
     * into a wild pointer. */
    if (n > INT32_MAX || hc > INT32_MAX || hc > INT64_MAX / n) return RAD_E_SHAPE;
    /* rad_param_getdim rather than rad_args_geti: it reads a RANGE as its high bound, which is
     * what a tool calling the launch directly still holds, and it is the ABI's own reader for
     * exactly that case. A STRING spelling is accepted for the same reason the reference accepts
     * one -- a checker reaching the launch without a declare in front of it. */
    const char* dt = rad_param_gets(a->p, a->n_p, "dtype", "bf16");
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_DTYPE;
    if (h.dtype != RAD_BF16 || x.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (h.rank != 2 || x.rank != 2) return RAD_E_UNSUPPORTED;
    if (hc != kHc || n % kAlign != 0) return RAD_E_UNSUPPORTED;
    if (h.shape[1] != hc * n || x.shape[1] != n) return RAD_E_SHAPE;

    /* Rows are the operand's, and `M` is only the ceiling -- a decode step presents one row of a
     * max_tok arena, and reading `M` as the count walks rows nobody wrote (libref/ref_ops.h). */
    const int64_t rows = h.shape[0];
    const int64_t band = rad_param_getdim(a->p, a->n_p, "M", 0);
    if (rows <= 0 || rows > INT32_MAX || (band > 0 && rows > band) || x.shape[0] != rows)
        return RAD_E_SHAPE;

    /* THE COLUMN STEP HAS TO BE 1, and this is the one place the device row is narrower than the
     * oracle. The kernel reads a lane's 8 consecutive elements as one group and adds stream c at
     * +c*N, so a transposed or element-interleaved view would be read wrong -- and the reference,
     * which walks each element by stride, would be right. Refusing it is what keeps the two from
     * disagreeing; the ROW pitch is still honoured, which is the view the caller actually uses. */
    if (h.stride[1] != 1 || x.stride[1] != 1) return RAD_E_UNSUPPORTED;
    /* Every offset is `row*stride[0] + col`. A negative pitch is never a view. With more than one
     * row a pitch narrower than the row would make consecutive rows overlap -- a store to row m
     * landing in row m+1's data -- and a pitch large enough to overflow the product would wrap the
     * address. Both are refused. With ONE row the pitch is never used, so it may be zero, which is
     * what several callers hand a single-row slice. */
    if (h.stride[0] < 0 || x.stride[0] < 0) return RAD_E_SHAPE;
    if (rows > 1) {
        if (h.stride[0] < h.shape[1] || x.stride[0] < x.shape[1]) return RAD_E_SHAPE;
        if (rows > INT64_MAX / h.stride[0] || rows > INT64_MAX / x.stride[0]) return RAD_E_SHAPE;
    }
    /* The element offsets have to be expressible; a negative or overflowing one is a caller bug
     * that the device would turn into a wild pointer. */
    if (span_elements(h) <= 0 || span_elements(x) <= 0) return RAD_E_SHAPE;
    if (overlaps(h, x)) return RAD_E_INVAL;

    call->h = h.data;
    call->x = x.data;
    call->rows = rows;
    call->N = n;
    call->h_pitch = h.stride[0];
    call->x_pitch = x.stride[0];
    return RAD_OK;
}

/* ---- the row ----------------------------------------------------------------------------------- */
/* THE CONSTRAINTS ARE THE SPECIALISATION, stated so selection never has to find out the hard way.
 * Without them this row would be chosen for every hc_mean in the graph (device outranks host) and
 * then refuse hc != 4 at issue, which turns "the host reference serves this" into a failed step.
 * `N % 8 == 0` is RAD_CDIV's meaning: N is a whole number of 8-wide bf16 groups. */
static const RadConstraint kConstraints[] = {
    RAD_CEQ("hc", 4),
    RAD_CDIV("N", 8),
    RAD_CIN("dtype", "bf16"),
};

RadKernelInfo rad_hc_mean_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "mine_hc_mean_hc4";
    r.op = kSchema.op;
    r.family = "norm";
    r.computes = kSchema.doc;
    r.shape = "hc==4, N%8==0; the DFlash2 tap (N=2560), any row count up to max_tok";
    r.dtypes = "bf16 in, bf16 out";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kConstraints;
    r.n_constraints = (int)(sizeof(kConstraints) / sizeof(kConstraints[0]));
    r.launch = launch;
    r.opd_shape = rad_hc_mean_shape;
    return r;
}
