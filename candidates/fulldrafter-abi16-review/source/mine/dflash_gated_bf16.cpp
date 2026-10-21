/* dflash_gated_bf16.cpp -- the schema, the host-side validation and the declared chain for
 * DFlash2's BF16 gate/up projection with the SiLU-gated product fused into the epilogue.
 *
 * THIS FILE IS COMPILED INTO THE DEVICE PLUGIN ONLY. The host oracle for the unfused chain is
 * libr4d's own `gemm_nt` and `silu_mul` rows, and libref's reference GEMM and elementwise kernels;
 * duplicating them here would be a second reference a comparator could then pick between. What
 * this file shares is the SCHEMA and the VALIDATION, so a device kernel that accepted a call the
 * reference refuses is a divergence the comparator cannot see.
 *
 * THE PARAMETERS ARE `gemm_nt`'s, and that is deliberate: the fused op replaces one GEMM, so M, N
 * and K keep their meanings and `N` is the gate_up width, 2 * n_ff. The output is half that wide.
 * A caller porting a graph rewrites one op, not its arithmetic.
 */
#include "dflash_gated_bf16.h"

#include <cstdint>
#include <cstring>

namespace {

/* THE GEOMETRY THIS ROW IS COMPILED FOR, and every bound below has a reason rather than a
 * measured number behind it:
 *
 *   kRowMax   64    the skinny arm's decode band. `gemm_bf16_nt_m64` itself serves M <= 65536,
 *                   but this fusion holds TWO accumulator sets (gate AND up) live across the whole
 *                   K loop, so it stays on the band the DFlash2 query pass actually runs --
 *                   max_seqs * block, capped by the block the drafter declares. Above 64 rows the
 *                   tiled arm would be the right shape and is NOT implemented; the op then has no
 *                   row in that band, resolution fails closed, and the graph keeps the unfused
 *                   pair.
 *   kNMin     2048  the gate_up width that makes the split law answer SK == 1 by construction:
 *                   the law takes the smallest split whose `n_tiles * SK` reaches 128, and
 *                   2048/16 = 128 tiles already reaches it. SK != 1 would need a slice reduction
 *                   this kernel does not have, and refusing it here keeps a geometry that would
 *                   reorder the sum from ever reaching the launcher.
 *   kNStep    32    N divisible by 32, so the OUTPUT width n = N/2 is a whole number of 16-column
 *                   windows. A wave owns window t of the gate half and window t of the up half; a
 *                   ragged n would make the last gate window overrun into the up half's rows and
 *                   read the wrong weight rows. Every power-of-two n_ff meets this.
 *   kAlign    16    every fragment load in the kernel is an 8- or 16-byte access, and the skinny
 *                   bf16 row already requires 16-byte operands (r4d_gemm_args.h:387).
 *   kOffMax   INT32_MAX  the kernel forms 32-BIT element offsets for both operands (`rowGo(K, ...)`
 *                   for A and `woff_g + n*K` for W), formed as `unsigned` and added to a base
 *                   pointer, so the largest legal product of two dimensions is 2^31-1 elements.
 *                   The test divides instead of multiplying, so it cannot overflow itself. */
constexpr int64_t kRowMax = 64;
constexpr int64_t kNMin = 2048;
constexpr int64_t kNStep = 32;
constexpr int64_t kAlign = 16;
/* The largest product of two dimensions this row accepts, in ELEMENTS. It is the kernel's own
 * 32-bit offset limit; the shape below divides instead of multiplying so the test itself cannot
 * overflow. */
constexpr int64_t kOffMax = 2147483647;   /* INT32_MAX: a 32-bit element offset must be positive */

/* `a * b <= kOffMax` WITHOUT EVER FORMING `a * b`. Both arguments are required positive. This is
 * the only place the bound is decided, and both the call validator and the fuse hook use it, so a
 * geometry the launcher would refuse is refused here too -- which is what makes a CPU-side
 * boundary test exercise the real guard rather than a copy of it. */
bool product_fits(long long a, long long b) {
    if (a <= 0 || b <= 0) return false;
    if (a > kOffMax || b > kOffMax) return false;
    return b <= kOffMax / a;
}

/* Elements a tensor occupies, from its OWN shape and strides. Saturating: a span that would
 * overflow returns INT64_MAX rather than a wrapped number, so an overlap test on a nonsense
 * tensor refuses conservatively instead of reading as "no overlap". */
int64_t span_elements(const RadTensor& t) {
    int64_t e = 1;
    for (uint32_t i = 0; i < t.rank; ++i) {
        if (t.shape[i] <= 0 || t.stride[i] < 0) return INT64_MAX;
        if (t.shape[i] == 1) continue;
        if (t.stride[i] != 0 && (t.shape[i] - 1) > (INT64_MAX - e) / t.stride[i])
            return INT64_MAX;
        e += (t.shape[i] - 1) * t.stride[i];
    }
    return e;
}

/* Byte spans overlap? Both operands are bf16, so the multiplier is 2 either side and cancels. An
 * unknown span (INT64_MAX, saturated above) is treated as overlapping: the caller refuses. */
bool overlaps(const RadTensor& a, const RadTensor& b) {
    if (!a.data || !b.data) return false;
    const int64_t ea = span_elements(a), eb = span_elements(b);
    if (ea <= 0 || eb <= 0) return true;
    const uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
    const uint64_t xa = (uint64_t)ea * 2u, xb = (uint64_t)eb * 2u;
    return x < y + xb && y < x + xa;
}

static_assert(RAD_DFLASH_GATED_OPERANDS == 3, "the schema below lists three operands");

}  /* namespace */

int rad_dflash_gated_bf16_schema_count(void) { return 1; }

/* ---- schema ---------------------------------------------------------------------------------- */
#define P_INT(k)  { k, RAD_P_INT, RAD_REQUIRED, 0 }
#define P_STR(k)  { k, RAD_P_STR, RAD_REQUIRED, 0 }
#define IN(n)     { n, RAD_OPD_IN,  0 }
#define WGT(n)    { n, RAD_OPD_WEIGHT, 0 }
#define OUT(n)    { n, RAD_OPD_OUT, 0 }

static const RadParamSpec kParams[] = {
    P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
};
/* THE OPERAND ORDER IS `gemm_nt`'s, with `y` half as wide. `b` is the WEIGHT role, not an input:
 * issue resolves the handle and waits on the mover for a weight, and reads a plain operand's
 * buffer. */
static const RadOperandSpec kOperands[] = { IN("a"), WGT("b"), OUT("y") };
static const RadOpSchema kSchema = {
    "dflash_gated_bf16", kParams, (int)(sizeof(kParams) / sizeof(kParams[0])),
    kOperands, (int)(sizeof(kOperands) / sizeof(kOperands[0])),
    "y[m,i] = bf16(silu(bf16(c[m,i])) * bf16(c[m,n+i])), c = A @ W^T over the whole [2n, K] "
    "gate_up weight in one fp32 accumulation, with the gate half first. Both intermediate "
    "roundings the unfused `gemm_nt` -> `silu_mul` pair performs are kept; nothing is quantised."
};
#undef P_INT
#undef P_STR
#undef IN
#undef WGT
#undef OUT

const RadOpSchema* rad_dflash_gated_bf16_schema_at(int i) { return i == 0 ? &kSchema : nullptr; }

/* ---- operand description --------------------------------------------------------------------- */
/* `M` is the band ceiling and each operand names the call, exactly as hc_mean's hook does: a tool
 * sizing from this gets the band, and a launch reads the rows from the tensor. */
int rad_dflash_gated_bf16_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= RAD_DFLASH_GATED_OPERANDS) return RAD_E_INVAL;
    const int64_t M = rad_param_getdim(p, n_p, "M", 0);
    const int64_t N = rad_param_getdim(p, n_p, "N", 0);
    const int64_t K = rad_param_getdim(p, n_p, "K", 0);
    if (M <= 0 || N <= 0 || K <= 0 || (N & 1)) return RAD_E_UNSUPPORTED;
    /* A derived extent this hook hands out is a product the caller will allocate from, so an
     * over-large one is refused HERE rather than described and then refused at issue: a tool that
     * sized a buffer from `N * K` and then got RAD_E_SHAPE would have allocated on a bogus
     * number. `M * K`, `N * K` and `M * N` are the three products the operands imply. */
    if (!product_fits(M, K) || !product_fits(N, K) || !product_fits(M, N))
        return RAD_E_UNSUPPORTED;
    RadOpdDesc d{};
    d.dtype = RAD_BF16;
    d.rank = 2;
    d.shape[0] = M;
    d.shape[1] = operand == RAD_DFLASH_GATED_Y ? N / 2 : K;
    /* The weight's rows are the gate_up plane, not the output width. */
    if (operand == RAD_DFLASH_GATED_B) d.shape[0] = N;
    d.fill = (uint32_t)RAD_FILL_NORMAL;
    d.idx_const = -1;
    *out = d;
    return RAD_OK;
}

/* ---- the call ---------------------------------------------------------------------------------- */
int rad_dflash_gated_bf16_args(const RadArgs* a, RadDflashGatedCall* call) {
    if (!a || !call || !a->t || a->n_t != RAD_DFLASH_GATED_OPERANDS) return RAD_E_INVAL;
    const RadTensor& x = a->t[RAD_DFLASH_GATED_A];
    const RadTensor& w = a->t[RAD_DFLASH_GATED_B];
    const RadTensor& y = a->t[RAD_DFLASH_GATED_Y];
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;

    const int64_t N = rad_param_getdim(a->p, a->n_p, "N", 0);
    const int64_t K = rad_param_getdim(a->p, a->n_p, "K", 0);
    if (N <= 0 || K <= 0 || (N & 1)) return RAD_E_SHAPE;
    const int64_t n = N / 2;

    /* THE 32-BIT OFFSET BOUND COMES FIRST, before any product is formed or any tensor extent is
     * read. The kernel addresses both operands with 32-bit element offsets, so `M*K`, `N*K` and
     * the weight extent are only meaningful -- and only computable without overflow -- under this
     * bound. Deciding it here, and not in the launcher, is what makes the CPU-side boundary check
     * exercise the real guard; the launcher carries no second copy of it because one would be
     * dead code after this. */
    if (!product_fits(N, K)) return RAD_E_SHAPE;

    /* A STRING SPELLING IS ACCEPTED for the reason hc_mean accepts one: a checker reaching the
     * launch without a declare in front of it. */
    const char* dt = rad_param_gets(a->p, a->n_p, "dtype", "bf16");
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_DTYPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;

    if (x.rank != 2 || y.rank != 2) return RAD_E_UNSUPPORTED;
    if (N < kNMin || (N % kNStep) != 0 || (K % 16) != 0) return RAD_E_UNSUPPORTED;

    /* Rows are the OPERAND's and `M` is only the ceiling: a decode step presents one sequence's
     * block out of a max_tok arena, and reading `M` as the count walks rows nobody wrote. `rows`
     * is bounded by kRowMax below, which is what makes `rows * K` a safe product. */
    const int64_t rows = x.shape[0];
    const int64_t band = rad_param_getdim(a->p, a->n_p, "M", 0);
    if (rows <= 0 || (band > 0 && rows > band) || rows > kRowMax) return RAD_E_SHAPE;
    /* A's own offsets are `row * K + col`, so its extent is bounded by `rows * K`. */
    if (!product_fits(rows, K)) return RAD_E_SHAPE;
    if (x.shape[1] != K || y.shape[0] != rows || y.shape[1] != n) return RAD_E_SHAPE;
    /* The weight is an opaque plane to libr4d's shims and is checked here only for extent: the
     * layout hook that would interpret a quantised encoding does not exist on this row, and the
     * dtype check above already refused anything but a plain bf16 plane. `N * K` is safe to form
     * under product_fits above, and the span saturates rather than wrapping. */
    const int64_t w_need = N * K;
    if (span_elements(w) < w_need) return RAD_E_SHAPE;

    if (!rad_tensor_is_contiguous(&x) || !rad_tensor_is_contiguous(&w) ||
        !rad_tensor_is_contiguous(&y))
        return RAD_E_STRIDE;

    /* Every fragment address is `base + 16-byte-aligned offset`; the kernel does not check, so the
     * shim is where it is checked. */
    if (((uintptr_t)x.data & (kAlign - 1)) || ((uintptr_t)w.data & (kAlign - 1)) ||
        ((uintptr_t)y.data & (kAlign - 1)))
        return RAD_E_ALIGN;
    /* `__restrict__` is on all three pointers in the kernel, so no two operands may overlap --
     * not just the output against its inputs. An `a`/`w` overlap would be aliasing UB that no
     * output comparison could attribute. */
    if (overlaps(x, y) || overlaps(w, y) || overlaps(x, w)) return RAD_E_INVAL;

    call->a = x.data;
    call->w = w.data;
    call->y = y.data;
    call->M = rows;
    call->K = K;
    call->N = N;
    call->n = n;
    return RAD_OK;
}

/* ---- the declared chain ------------------------------------------------------------------------ */
/* `gemm_nt` into a temporary, then `silu_mul` over it at half the width. The temporary is the
 * [M, 2n] bf16 plane the unfused pair writes, so the chain describes exactly the two launches this
 * one replaces -- including the rounding both of them perform on it.
 *
 * A STEP'S `M` IS MIRRORED VERBATIM, kind and all, for the reason libr4d's own Step::mirror gives
 * (r4d_fuse.cpp:172-180): `M` is the RANGED parameter of every op in this vocabulary, and
 * re-emitting a range as its collapsed high bound would hand the step a different BAND from the
 * one the fused kernel was selected for -- which for a banded GEMM family is a different kernel
 * with a different summation order. `N` and `K` are fixed values in both ops and are emitted as
 * the ints they are; `silu_mul`'s `n` is the fused `N / 2`, which is what makes the second step's
 * width the output's.
 *
 * NOT CHECKABLE BY STOCK rad-kbench: tools/rad_kbench.cpp:2682-2686 resolves every step inside the
 * FUSED ROW'S OWN PLUGIN, and `mine` publishes neither `gemm_nt` nor `silu_mul`. The declaration is
 * still the honest description of the arithmetic; parity is the comparator's job. */
namespace {

RadParam pint(const char* k, long long v) {
    RadParam p{};
    p.key = k; p.kind = RAD_P_INT; p.ival = v;
    return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p{};
    p.key = k; p.kind = RAD_P_STR; p.sval = v;
    return p;
}

/* Copy a parameter of the fused op VERBATIM, kind and all, so a range stays a range. Bounded: a
 * step whose list overflows is a drift error, not something to write past. */
bool mirror(const RadParam* p, int n_p, const char* key, RadParam* dst) {
    const RadParam* f = rad_param_find(p, n_p, key);
    if (!f) return false;
    *dst = *f;
    return true;
}

/* One entry per operand of the STEP op, in its schema's order. */
bool wire(RadFuseStep* s, const int16_t* from, int n) {
    if (n > RAD_MAX_FUSE_OPD) return false;
    for (int i = 0; i < n; ++i) s->from[i] = from[i];
    s->n_from = n;
    return true;
}

}  /* namespace */

int rad_dflash_gated_bf16_fuse(const RadParam* p, int n_p, uint64_t present, int step,
                               RadFuseStep* out) {
    (void)present;   /* every operand is required in both ops, so presence cannot change the chain */
    if (!out) return RAD_E_INVAL;
    const int64_t M = rad_param_getdim(p, n_p, "M", 0);
    const int64_t N = rad_param_getdim(p, n_p, "N", 0);
    const int64_t K = rad_param_getdim(p, n_p, "K", 0);
    const char* dt = rad_param_gets(p, n_p, "dtype", 0);
    if (M <= 0 || N <= 0 || K <= 0 || (N & 1)) return RAD_E_UNSUPPORTED;
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_UNSUPPORTED;
    /* A step built at a geometry the step op cannot serve would resolve to nothing and read as
     * "this fusion has no equivalent", which is a different claim from "not at this shape". The
     * products are refused WITHOUT being formed, for the same reason the call validator does:
     * `N * K` on an extreme geometry is an overflow, not a large number. */
    if (!product_fits(M, K) || !product_fits(N, K) || !product_fits(M, N))
        return RAD_E_UNSUPPORTED;
    if (N < kNMin || (N % kNStep) != 0 || (K % 16) != 0 || M > kRowMax) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        /* The GEMM over the whole [N, K] gate_up plane, into the bf16 that the fold never
         * materialises. */
        *out = RadFuseStep{};
        out->op = "gemm_nt";
        if (!mirror(p, n_p, "M", &out->p[out->n_p])) return RAD_E_UNSUPPORTED;
        ++out->n_p;
        out->p[out->n_p++] = pint("N", (long long)N);
        out->p[out->n_p++] = pint("K", (long long)K);
        out->p[out->n_p++] = pstr("dtype", "bf16");
        const int16_t w[] = { RAD_DFLASH_GATED_A, RAD_DFLASH_GATED_B, RAD_FUSE_TMP(0) };
        return wire(out, w, 3) ? RAD_OK : RAD_E_INVAL;
    }
    case 1: {
        /* The gate, the multiply and the store over the accumulator plane, at half its width. */
        *out = RadFuseStep{};
        out->op = "silu_mul";
        if (!mirror(p, n_p, "M", &out->p[out->n_p])) return RAD_E_UNSUPPORTED;
        ++out->n_p;
        out->p[out->n_p++] = pint("n", (long long)(N / 2));
        out->p[out->n_p++] = pstr("dtype", "bf16");
        const int16_t w[] = { RAD_FUSE_TMP(0), RAD_DFLASH_GATED_Y };
        return wire(out, w, 2) ? RAD_OK : RAD_E_INVAL;
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ---- the row ----------------------------------------------------------------------------------- */
/* THE CONSTRAINTS ARE THE SPECIALISATION, stated so selection never has to find out the hard way.
 * `N >= 2048` is what makes the split law answer SK == 1, which is the condition under which this
 * kernel's accumulation is the unfused GEMM's own; a narrower gate_up would need a slice reduction
 * and is declined rather than served differently. `N % 32 == 0` is what makes the output width a
 * whole number of 16-column windows, which is how the gate and up halves are paired. `M <= 64` is
 * the skinny arm: above it the tiled arm would be the right shape and this row has no kernel for
 * it, so that band resolves to nothing and the graph keeps the pair. */
static const RadConstraint kConstraints[] = {
    RAD_CGE("M", 1),
    RAD_CLE("M", kRowMax),
    RAD_CDIV("K", 16),
    RAD_CGE("N", kNMin),
    RAD_CDIV("N", kNStep),
    RAD_CIN("dtype", "bf16"),
};

RadKernelInfo rad_dflash_gated_bf16_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "mine_dflash_gated_bf16";
    r.op = kSchema.op;
    r.family = "gemm";
    r.computes = kSchema.doc;
    r.shape = "bf16 A/W, gate_up N = 2*n with the gate half first; M <= 64, N >= 2048 and a "
              "multiple of 32, K a multiple of 16; SK == 1 by construction, so no cross-slice "
              "reduction exists";
    r.dtypes = "bf16 in, bf16 out; fp32 accumulate, two bf16 roundings kept";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kConstraints;
    r.n_constraints = (int)(sizeof(kConstraints) / sizeof(kConstraints[0]));
    r.launch = launch;
    r.opd_shape = rad_dflash_gated_bf16_shape;
    r.replaces = rad_dflash_gated_bf16_fuse;
    return r;
}
