/* dflash_conv_add.cpp -- schema, operand description, call validation and the declared fusion for
 * DFlash2's closing convolution plus its residual add.
 *
 * THE SCHEMA IS `dflash_conv`'s, PLUS THE RESIDUAL. The parameters are the convolution's own, in
 * its own spelling and order (`T`, `hidden_size`, `taps`, `group_size`, `NG`, `block_size`,
 * `dtype`), because the fused op is selected where that op was selected and a caller that has the
 * geometry already has it in this form. Nothing of `add`'s geometry is added: its `M` is `T` and
 * its `n` is `hidden_size`, so a second pair of keys would be the same two numbers under two names
 * and a way for them to disagree.
 *
 * THE OPERAND ORDER IS THE CONVOLUTION'S, WITH THE RESIDUAL AND THE OUTPUT AT THE END:
 * x, delta, base, residual, out. Appending rather than interleaving keeps the conv's four
 * operands in the positions the unfused op already names -- which is what makes the fusion hook
 * below a wiring of indices rather than a re-description -- and puts the two operands the fusion
 * adds where a reader looks for them.
 *
 * THE OUTPUT MAY ALIAS `residual`, AND MUST NOT ALIAS `x`. The unfused pair is issued with the
 * residual stream as the add's `a` and its `y` (arch/common/rad_block_dflash2.h:1126), while the
 * add's `b` is the bf16 plane the convolution wrote -- so an in-place form is the real call and
 * refusing it would refuse every site. `x` is different: tap 1 of row t reads row t-1 of the
 * INPUT, so an output overlapping it writes a row the tap set has not read yet.
 *
 * THIS FILE IS COMPILED INTO THE DEVICE PLUGIN ONLY. The host-domain reference for the chain is
 * the two unsfused rows' own (libr4d's dflash_conv is itself device-only, and `add` has libref's
 * host row), and duplicating that here would be a second reference to disagree with. What is
 * shared is the SCHEMA and the VALIDATION: a device kernel accepting a call the unfused pair
 * refuses is a divergence no oracle can see.
 */
#include "dflash_conv_add.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace {

/* The one geometry the device row is compiled for. taps 2 and group_size 16 are DFlash2's, and
 * hidden_size divisible by 64 is what makes a lane's 8-channel group, its 16 channels of group,
 * and the 16-byte vector loads line up: `hidden_size/8` lanes, a group of two lanes, and every row
 * start 16-byte aligned. It is the unfused row's band (`cDflashConv`), so no geometry arrives here
 * that `dflash_conv` would have served. */
constexpr int64_t kTaps  = 2;
constexpr int64_t kGroup = 16;
constexpr int64_t kAlign = 64;
constexpr int64_t kVec   = 8;

/* Elements a tensor occupies, from its OWN shape and strides -- a view is measured over what it
 * names, not over the buffer it looks into. */
int64_t span_elements(const RadTensor& t) {
    int64_t e = 1;
    for (uint32_t i = 0; i < t.rank; ++i) e += (t.shape[i] - 1) * t.stride[i];
    return e;
}

bool byte_overlap(const RadTensor& a, const RadTensor& b, int64_t ea, int64_t eb) {
    if (!a.data || !b.data) return false;
    const uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
    const uint64_t xa = (uint64_t)ea * 2u, xb = (uint64_t)eb * 2u;
    return x < y + xb && y < x + xa;
}

/* EVERY AXIS BUT THE LEADING ONE IS DENSE, so the tensor is rows of one width and the only pitch
 * is stride[0]. That is the reading libr4d's shims take of the same operands (r4d_args.h's
 * r4d_row_pitch), and it is deliberately the same one: the delta operand is a COLUMN SLICE whose
 * row pitch is the wider buffer's, and a helper that answered with the packed width would read the
 * other coefficient half and produce numbers that look like a convolution.
 *
 * `width` receives the elements one row spans. */
int64_t dense_row_pitch(const RadTensor& t, int64_t* width) {
    if (t.rank < 2) return -1;
    int64_t acc = 1;
    for (int i = (int)t.rank - 1; i >= 1; --i) {
        if (t.shape[i] != 1 && t.stride[i] != acc) return -1;
        acc *= t.shape[i];
    }
    *width = acc;
    return t.stride[0];
}

bool aligned16(const void* p) { return p && ((uintptr_t)p & 15u) == 0; }

}  /* namespace */

int rad_dflash_conv_add_schema_count(void) { return 1; }

/* ---- schema ---------------------------------------------------------------------------------- */
#define P_INT(k)  { k, RAD_P_INT, RAD_REQUIRED, 0 }
#define P_STR(k)  { k, RAD_P_STR, RAD_REQUIRED, 0 }
#define IN(n)     { n, RAD_OPD_IN,  0 }
#define OUT(n)    { n, RAD_OPD_OUT, 0 }

static const RadParamSpec kParams[] = {
    P_INT("T"), P_INT("hidden_size"), P_INT("taps"), P_INT("group_size"),
    P_INT("NG"), P_INT("block_size"), P_STR("dtype"),
};
static const RadOperandSpec kOperands[] = {
    IN("x"),
    IN("delta"),
    /* A WEIGHT, for the reason dflash_conv states: base_kernel is a checkpoint parameter, and the
     * role is what gives it a residency class and a prefetch interval. */
    { "base", RAD_OPD_WEIGHT, 0 },
    IN("residual"),
    OUT("out"),
};
static const RadOpSchema kSchema = {
    "dflash_conv_add", kParams, (int)(sizeof(kParams) / sizeof(kParams[0])),
    kOperands, (int)(sizeof(kOperands) / sizeof(kOperands[0])),
    "DFlash2's closing grouped convolution with its residual add in one launch: h2 = conv(x, "
    "delta, base) with the accumulator rounded to bf16, then out = residual + h2 rounded once. "
    "Its arithmetic target is the unfused `dflash_conv` followed by `add`, including the bf16 "
    "rounding between them; the equivalence has NOT been measured."
};
#undef P_INT
#undef P_STR
#undef IN
#undef OUT

const RadOpSchema* rad_dflash_conv_add_schema_at(int i) { return i == 0 ? &kSchema : nullptr; }

/* ---- operand description --------------------------------------------------------------------- */
int rad_dflash_conv_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= RAD_DFLASH_CONV_ADD_OPERANDS) return RAD_E_INVAL;
    const int64_t T    = rad_param_getdim(p, n_p, "T", 0);
    const int64_t H    = rad_param_getdim(p, n_p, "hidden_size", 0);
    const int64_t taps = rad_param_getdim(p, n_p, "taps", 0);
    const int64_t NG   = rad_param_getdim(p, n_p, "NG", 0);
    const int64_t grp  = rad_param_getdim(p, n_p, "group_size", 0);
    const char*   dt   = rad_param_gets(p, n_p, "dtype", nullptr);
    /* The entry point's own conditions, checked here so a geometry that cannot be served declines
     * rather than being described into a buffer nothing will read. */
    if (T <= 0 || H <= 0 || grp <= 0 || taps <= 0 || NG <= 0) return RAD_E_UNSUPPORTED;
    if (taps != kTaps || grp != kGroup || H % kAlign || NG != H / grp) return RAD_E_UNSUPPORTED;
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_UNSUPPORTED;

    RadOpdDesc d{};
    d.dtype = RAD_BF16;
    d.fill  = (uint32_t)RAD_FILL_NORMAL;
    d.idx_const = -1;
    switch (operand) {
    case RAD_DFLASH_CONV_ADD_X:
    case RAD_DFLASH_CONV_ADD_RESIDUAL:
    case RAD_DFLASH_CONV_ADD_OUT:
        d.rank = 2;
        d.shape[0] = T;
        d.shape[1] = H;
        break;
    case RAD_DFLASH_CONV_ADD_DELTA:
        /* THE PROJECTION, NOT THE SLICE: the operand a real call hands over is one half of this
         * at the projection's own row pitch (2*taps*NG), and a description of the half would
         * allocate a buffer whose natural pitch is half the one the kernel is handed. */
        d.rank = 4;
        d.shape[0] = T;
        d.shape[1] = 2;
        d.shape[2] = taps;
        d.shape[3] = NG;
        break;
    case RAD_DFLASH_CONV_ADD_BASE:
    default:
        d.rank = 2;
        d.shape[0] = taps;
        d.shape[1] = H;
        break;
    }
    *out = d;
    return RAD_OK;
}

/* ---- the call ---------------------------------------------------------------------------------- */
int rad_dflash_conv_add_args(const RadArgs* a, RadDflashConvAddCall* call) {
    if (!a || !call || !a->t || a->n_t != RAD_DFLASH_CONV_ADD_OPERANDS) return RAD_E_INVAL;
    const RadTensor& x   = a->t[RAD_DFLASH_CONV_ADD_X];
    const RadTensor& d   = a->t[RAD_DFLASH_CONV_ADD_DELTA];
    const RadTensor& b   = a->t[RAD_DFLASH_CONV_ADD_BASE];
    const RadTensor& r   = a->t[RAD_DFLASH_CONV_ADD_RESIDUAL];
    const RadTensor& o   = a->t[RAD_DFLASH_CONV_ADD_OUT];
    if (!x.data || !d.data || !b.data || !r.data || !o.data) return RAD_E_INVAL;

    const int64_t H    = rad_param_getdim(a->p, a->n_p, "hidden_size", 0);
    const int64_t taps = rad_param_getdim(a->p, a->n_p, "taps", 0);
    const int64_t NG   = rad_param_getdim(a->p, a->n_p, "NG", 0);
    const int64_t grp  = rad_param_getdim(a->p, a->n_p, "group_size", 0);
    const int64_t bsz  = rad_param_getdim(a->p, a->n_p, "block_size", 0);
    if (H <= 0 || taps <= 0 || NG <= 0 || grp <= 0 || bsz <= 0) return RAD_E_SHAPE;
    if (H > INT32_MAX || NG > INT32_MAX || bsz > INT32_MAX) return RAD_E_SHAPE;
    /* THE PARAMETERS ARE THE SPECIALISATION: a call outside the compiled geometry is refused here
     * rather than by the constraints alone, because a tool may reach a launch with no selection in
     * front of it. */
    if (taps != kTaps || grp != kGroup || H % kAlign || NG != H / grp) return RAD_E_UNSUPPORTED;
    /* THE OFF MASK IN THE KERNEL IS A BOUND, NOT AN OPTIMISATION. A lane's channel offset is
     * `(hv & 0xFFFFF) * 8`, so hidden_size above `8 * (1 << 20)` would alias lane offsets onto
     * each other -- silently, with every admitted call looking legitimate. H is capped here at
     * exactly that value, which also keeps every derived pitch inside `int`. */
    if (H > 8 * (INT64_C(1) << 20)) return RAD_E_UNSUPPORTED;

    const char* dt = rad_param_gets(a->p, a->n_p, "dtype", "bf16");
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_DTYPE;
    if (x.dtype != RAD_BF16 || d.dtype != RAD_BF16 || b.dtype != RAD_BF16 ||
        r.dtype != RAD_BF16 || o.dtype != RAD_BF16)
        return RAD_E_DTYPE;

    /* Rows are the OPERAND's, and `T` is only the ceiling of the band -- a decode step presents
     * one row of a max_tok arena, and reading `T` as the count walks rows nobody wrote. */
    if (x.rank != 2 || r.rank != 2 || o.rank != 2) return RAD_E_UNSUPPORTED;
    const int64_t rows = x.shape[0];
    const int64_t band = rad_param_getdim(a->p, a->n_p, "T", 0);
    if (rows <= 0 || rows > INT32_MAX || (band > 0 && rows > band)) return RAD_E_SHAPE;
    if (r.shape[0] != rows || o.shape[0] != rows) return RAD_E_SHAPE;
    if (x.shape[1] != H || r.shape[1] != H || o.shape[1] != H) return RAD_E_SHAPE;

    int64_t aw = 0, rw = 0, ow = 0;
    const int64_t x_pitch = dense_row_pitch(x, &aw);
    const int64_t r_pitch = dense_row_pitch(r, &rw);
    const int64_t o_pitch = dense_row_pitch(o, &ow);
    if (x_pitch < 0 || r_pitch < 0 || o_pitch < 0) return RAD_E_STRIDE;
    if (aw != H || rw != H || ow != H) return RAD_E_SHAPE;
    /* A negative pitch is never a view; a pitch narrower than the row would make consecutive rows
     * overlap (a store to row t landing in row t+1's data); a pitch that overflows the product
     * would wrap the address. All three are refused. */
    if (x_pitch < H || r_pitch < H || o_pitch < H) return RAD_E_SHAPE;
    if (rows > INT64_MAX / x_pitch || rows > INT64_MAX / r_pitch || rows > INT64_MAX / o_pitch)
        return RAD_E_SHAPE;
    /* THE LAUNCH NARROWS EVERY PITCH AND WIDTH TO `int`, so anything above INT32_MAX is refused
     * HERE rather than silently truncated at the argument struct. A truncated pitch is the worst
     * kind of wrong: the kernel reads a plausible row and produces a plausible number. */
    if (x_pitch > INT32_MAX || r_pitch > INT32_MAX || o_pitch > INT32_MAX) return RAD_E_SHAPE;
    /* Every lane loads one 16-byte vector at `row*pitch + off`, so both the pointer and the pitch
     * have to be 16-byte aligned. H is a multiple of 64, so the pitch rule only bites on a view
     * of a buffer whose row is not. */
    if (!aligned16(x.data) || !aligned16(r.data) || !aligned16(o.data)) return RAD_E_ALIGN;
    if (x_pitch % kVec || r_pitch % kVec || o_pitch % kVec) return RAD_E_ALIGN;

    /* ---- the coefficients: a slice of the [T, 2, taps, NG] projection ---- */
    int64_t dw = 0;
    const int64_t dpitch = dense_row_pitch(d, &dw);
    if (dpitch < 0) return RAD_E_STRIDE;
    /* THIS IS THE ONE CHECK NO SHAPE TEST WOULD MAKE. The slice's natural width is taps*NG and
     * the projection's row pitch is 2*taps*NG; passing the packed pitch reads the other side's
     * coefficients, which is a plausible convolution and not a shape error. */
    if (dpitch != 2 * taps * NG) return RAD_E_STRIDE;
    if (dpitch > INT32_MAX) return RAD_E_SHAPE;
    if (dw < taps * NG || dw % NG) return RAD_E_SHAPE;
    if (d.shape[0] < rows) return RAD_E_SHAPE;
    if (!aligned16(d.data) || dpitch % kVec) return RAD_E_ALIGN;

    /* ---- base: the tap row of base_kernel, contiguous ---- */
    if (!rad_tensor_is_contiguous(&b)) return RAD_E_STRIDE;
    if (rad_tensor_numel(&b) < taps * H) return RAD_E_SHAPE;
    if (!aligned16(b.data)) return RAD_E_ALIGN;

    /* ---- aliasing ---- */
    /* `out` MAY be `residual`, and when it is, it must be the SAME view -- same base, same pitch.
     * The graph issues the add in place (`x = x + h2`), so a partial overlap is not something a
     * caller produces; it is the case where lane j's store lands on lane j' 's unread residual,
     * and no per-lane ordering makes that defined. Refusing it keeps "in place" meaning in place
     * rather than "some overlap nobody reasoned about".
     *
     * `out` may NOT overlap `x` -- tap 1 reads row t-1 of the input, and a row already overwritten
     * is a convolution over its own output -- nor the coefficients, whose reads are pitched and
     * shared between rows. */
    const bool in_place = o.data == r.data;
    if (in_place && o_pitch != r_pitch) return RAD_E_INVAL;
    const int64_t ex = span_elements(x), ed = span_elements(d), eb = span_elements(b);
    const int64_t er = span_elements(r), eo = span_elements(o);
    if (ex <= 0 || ed <= 0 || eb <= 0 || er <= 0 || eo <= 0) return RAD_E_SHAPE;
    if (byte_overlap(o, x, eo, ex) || byte_overlap(o, d, eo, ed) || byte_overlap(o, b, eo, eb))
        return RAD_E_INVAL;
    if (!in_place && byte_overlap(o, r, eo, er)) return RAD_E_INVAL;

    call->x        = x.data;
    call->delta    = d.data;
    call->base     = b.data;
    call->residual = r.data;
    call->out      = o.data;
    call->rows     = rows;
    call->H        = H;
    call->NG       = NG;
    call->dpitch   = dpitch;
    call->x_pitch  = x_pitch;
    call->r_pitch  = r_pitch;
    call->o_pitch  = o_pitch;
    call->block_size = bsz;
    return RAD_OK;
}

/* ---- the declared fusion ----------------------------------------------------------------------- */
/* THE CHAIN, IN THE FUSED OP'S OWN OPERAND INDICES:
 *
 *   add temp          <- dflash_conv(x, delta, base)            the bf16 the add reads back
 *   out               <- add(residual, temp)
 *
 * The temporary is exactly the [T, hidden_size] plane the unfused pair materialises, so both
 * sides of a comparison leave the same bytes behind and the rounding between the two steps is the
 * rounding the fusion preserves.
 *
 * The steps' parameters are the steps' own spelling. `dflash_conv` takes this op's seven
 * parameters under the same names; `add` calls them `M` and `n`, and `T`/`hidden_size` are the
 * same two numbers, so they are emitted as INT with the band's HIGH bound -- what rad_param_getdim
 * would answer, and what the step's own selector buckets on.
 *
 * EVERY OPERAND OF THIS OP IS REQUIRED, so a `present` mask that omits one describes a call that
 * cannot be issued: the honest answer is RAD_E_UNSUPPORTED, not a chain with a hole in it. */
int rad_dflash_conv_add_fuse(const RadParam* p, int n_p, uint64_t present, int step,
                             RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    for (int i = 0; i < RAD_DFLASH_CONV_ADD_OPERANDS; ++i)
        if (!RAD_FUSE_HAS(present, i)) return RAD_E_UNSUPPORTED;

    const int64_t T    = rad_param_getdim(p, n_p, "T", 0);
    const int64_t H    = rad_param_getdim(p, n_p, "hidden_size", 0);
    const int64_t taps = rad_param_getdim(p, n_p, "taps", 0);
    const int64_t NG   = rad_param_getdim(p, n_p, "NG", 0);
    const int64_t grp  = rad_param_getdim(p, n_p, "group_size", 0);
    const char*   dt   = rad_param_gets(p, n_p, "dtype", nullptr);
    if (T <= 0 || H <= 0 || taps != kTaps || grp != kGroup || NG != H / grp || H % kAlign)
        return RAD_E_UNSUPPORTED;
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_UNSUPPORTED;

    /* Field by field rather than through rad_types.h's builders: those are compound literals, a C
     * construct that is only a GNU extension in C++, and this is ordinary host C++. */
    auto mirror = [&](const char* k) -> bool {
        if (out->n_p >= RAD_MAX_FUSE_PARAMS) return false;
        const RadParam* f = rad_param_find(p, n_p, k);
        if (!f) return false;
        out->p[out->n_p++] = *f;
        return true;
    };
    /* THE RANGED KEY HAS TO BE RE-EMITTED AS AN INT, and this is not the band loss libr4d's
     * `Step::mirror` warns about. A RadFuseStep is read back by writing each parameter into a
     * selector geometry, and that reader carries RAD_P_INT, RAD_P_STR and RAD_P_F64 -- a
     * RAD_P_RANGE in a step is silently dropped, so a mirrored `T` would reach the step's own
     * shape hook as absent and the temporary it sizes would be refused. rad_param_getdim's high
     * bound IS the value the fused op is resolved at, so the step carries the same band. */
    auto pint = [&](const char* k, long long v) -> bool {
        if (out->n_p >= RAD_MAX_FUSE_PARAMS) return false;
        RadParam q{};
        q.key = k;
        q.kind = RAD_P_INT;
        q.ival = v;
        out->p[out->n_p++] = q;
        return true;
    };
    auto wire = [&](std::initializer_list<int> from) -> int {
        if (from.size() > RAD_MAX_FUSE_OPD) return RAD_E_INVAL;
        for (int v : from) out->from[out->n_from++] = (int16_t)v;
        return RAD_OK;
    };

    *out = RadFuseStep{};
    switch (step) {
    case 0:
        /* The convolution, into the plane the pair materialises. */
        out->op = "dflash_conv";
        if (!pint("T", T) || !pint("hidden_size", H) || !mirror("taps") ||
            !mirror("group_size") || !mirror("NG") || !mirror("block_size") || !mirror("dtype"))
            return RAD_E_INVAL;
        /* x, delta, base, out */
        return wire({ RAD_DFLASH_CONV_ADD_X, RAD_DFLASH_CONV_ADD_DELTA, RAD_DFLASH_CONV_ADD_BASE,
                      RAD_FUSE_TMP(0) });
    case 1:
        /* The residual add, in place on `out` exactly as the block issues it. `add` spells the two
         * extents `M` and `n`; they are the same two numbers the convolution calls `T` and
         * `hidden_size`, so nothing is invented. */
        out->op = "add";
        if (!pint("M", T) || !pint("n", H) || !mirror("dtype")) return RAD_E_INVAL;
        /* a, b, y */
        return wire({ RAD_DFLASH_CONV_ADD_RESIDUAL, RAD_FUSE_TMP(0), RAD_DFLASH_CONV_ADD_OUT });
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ---- the row ----------------------------------------------------------------------------------- */
/* THE CONSTRAINTS ARE THE SPECIALISATION, stated so selection never has to find out the hard way.
 * They are `dflash_conv`'s own (`cDflashConv` in libr4d) and not a widening of them: this row is
 * chosen for exactly the geometries the unfused row would have served, so a model that resolves
 * it never finds a band the other would have taken. */
static const RadConstraint kConstraints[] = {
    RAD_CEQ("taps", 2),
    RAD_CEQ("group_size", 16),
    RAD_CDIV("hidden_size", 64),
    RAD_CIN("dtype", "bf16"),
    RAD_CGE("block_size", 1),
};

RadKernelInfo rad_dflash_conv_add_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "mine_dflash_conv_add_t2_g16_bf16";
    r.op = kSchema.op;
    r.family = "dflash";
    r.computes = kSchema.doc;
    r.shape = "taps 2, group_size 16, hidden_size % 64; delta is a slice of the [T, 2, taps, NG] "
              "projection so its row pitch is 2*taps*NG; out may alias residual and must not "
              "alias x";
    r.dtypes = "bf16 in and out, fp32 accumulate, bf16 rounding of the convolution before the add";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kConstraints;
    r.n_constraints = (int)(sizeof(kConstraints) / sizeof(kConstraints[0]));
    r.launch = launch;
    r.opd_shape = rad_dflash_conv_add_shape;
    r.replaces = rad_dflash_conv_add_fuse;
    return r;
}

/* ============================== WHAT `replaces` DOES *NOT* BUY HERE ==============================
 *
 * The hook above is the truthful chain, and on its own IT IS NOT EXECUTABLE BY STOCK rad-kbench.
 * The tool resolves each declared step with `row_in(THIS row's plugin, step_op, ...)`
 * (tools/rad_kbench.cpp:2682-2686), and `mine` has no `dflash_conv` and no `add` row -- by
 * design, since duplicating libr4d's kernels into this plugin would be a second copy of the
 * arithmetic the fusion is supposed to be checked against, and registering an `add` row here would
 * shadow libr4d's for every model that uses this plugin.
 *
 * So the chain is declared for the graph dump and for a comparator that selects the candidate row
 * by name and each unfused step from libr4d explicitly. Until that comparator exists and runs, NO
 * PARITY CLAIM RESTS ON THIS HOOK -- see reports/dflash2-conv-add-handoff.md, which states the
 * same blocker and the evidence still owed.
 */


