/* ref_shapes.cpp -- how big is operand i of this op at this geometry, and what may legally be in
 * it. libref's implementation of `RadKernelInfo::opd_shape` (rad_abi.h), for every op this
 * plugin registers.
 *
 * ============================== WHY THIS LIVES HERE ==============================
 *
 * A tool that wants to CALL an op cold -- which a correctness check and a benchmark both do -- has
 * to size its buffers somehow, and an op's schema names its operands and their roles but not their
 * extents. The alternative to this hook is a hand-written table inside `tools/` (opshapes.h), and
 * its own header says why that is the wrong home: it can only describe the ops whoever wrote the
 * table had heard of, it cannot be extended by anyone who does not patch the engine, and it is the
 * one place a tool knows something about an op. The component that knows an operand's extent is the
 * one that reads it. This file is ref answering for the whole conventional vocabulary, so an op
 * that stays inside docs/OPS.md is describable even when the kernel under test is not.
 *
 * ============================== ONE FUNCTION PER ROW, NAMED AFTER THE LAUNCH ==============
 *
 * `RadShapeFn` does not carry the op name -- it gets the resolved parameter list and an operand
 * index and nothing else -- so a single shared function would have to recover the op from a
 * parameter, and no op in this vocabulary carries its own name as one. The alternative is a small
 * per-op function in every registry row, and that is what this file supplies: for every entry point
 * `ref_X` there is a `ref_X_shape` beside it, so ref_registry.cpp's ROW() macro pastes the second
 * from the first (`fn ## _shape`) and no row names it explicitly. The 1:1 naming is load-bearing:
 * adding an op to the registry is one row plus one host implementation plus one shape function
 * whose name the macro already demands, so a new op that forgets to describe itself does not link.
 *
 * ============================== THE COUNT IS CHECKED, NOT ASSUMED ==============================
 *
 * A description must emit exactly as many operands as the schema declares, and that check is what
 * catches a description that has drifted from the schema it describes -- an op that grows an
 * optional operand and a description that does not follow leave the op silently skipped by a tool
 * whose whole job is to say what was checked. Padding with a guess would hide the drift behind a
 * number that looks like a result.
 *
 * The hook is called one operand at a time, so the property has to be rebuilt: every function below
 * fills a small local array with the WHOLE operand list and then hands out the one that was asked
 * for, and `Opd::finish` compares the length against `RadOpSchema::n_operands` read out of this
 * plugin's own schema table. A mismatch returns RAD_E_SHAPE and not RAD_E_UNSUPPORTED,
 * deliberately: RAD_E_UNSUPPORTED is "this geometry does not carry what the op needs" and is a
 * correct answer a caller skips and counts, while a count mismatch is a BUG in this file or in the
 * schema and deserves a distinguishable status.
 *
 * ============================== WHAT IS IN AN OPERAND ==============================
 *
 * The fill domain matters as much as the extent (rad_abi.h's own comment says the same at length):
 *
 *   * A gated-delta-net `g` is an intra-chunk cumulative sum of log-decays: non-increasing and
 *     never positive. The scan forms e^{g_i - g_j} with j <= i, which is <= 1 ONLY because g is
 *     monotone; handed normals the oracle overflows to 2e38 while the kernel does something else,
 *     and the comparison then measures the test rather than the kernel.
 *   * `beta` is a sigmoid output and lives in (0, 1). `A` is a chunk's triangular inverse and is
 *     zero above the diagonal within a chunk -- a dense random A puts two implementations on
 *     opposite sides of a convention (whether the strictly-upper part is read at all) that they
 *     never disagree about on a real one.
 *   * An index operand's RANGE is part of its meaning. A KV slot indexes n_blocks * block_size
 *     positions and a block-table entry indexes n_blocks, and neither is any operand's dim 0;
 *     drawing either from the wrong range tests the kernel's out-of-range handling and calls the
 *     result correctness.
 *   * A SCATTER destination has to be unique. Two tokens drawn onto one slot make the store a race
 *     and two implementations resolve it differently -- ref runs the token loop under OpenMP and a
 *     device kernel runs a workgroup a token -- so the comparison would measure the scheduler. A
 *     read-side index (a gather row, a block-table entry) may repeat and does.
 *   * `seqused` is a CONSTANT and not a draw. A random context length shorter than the query makes
 *     a causal attention read a prefix that does not exist, which both implementations would then
 *     agree about while neither computed what the model asks for.
 *   * An OPTIONAL operand a description deliberately does not pass still needs an entry -- the
 *     count check above is what catches drift -- so absence has a spelling, and a null
 *     RadTensor.data is what the ABI already means by it.
 *
 * ============================== WHAT THIS FILE MAY NOT ASSUME ==============================
 *
 * A kernel plugin links rad_abi and NOTHING ELSE (libref/CMakeLists.txt), so anything of core's
 * that is not in a header is unreachable: RAD_DEBUG is core's, and ref_common.h makes that trade.
 *
 * WHAT IS REACHABLE IS MORE THAN THE LINK RULE SUGGESTS. `rad_dtype_parse` and `rad_dtype_bytes`
 * are `static inline` in rad_types.h and `rad_param_*` in rad_plugin.h -- header-only, which is
 * the whole point of a header-only ABI -- so they compile into the plugin like any other inline.
 * The rule is: reach for the ABI's version first, and write a local one only where the ABI
 * genuinely has no answer. `gi` below is the one such place, and it says why.
 *
 * `rad_kernel_schema_count` / `rad_kernel_schema_at` ARE this plugin's own exports, defined in
 * ref_registry.cpp beside the table they read, so the count check costs no coupling at all.
 */
#include "ref_ops.h"

/* rad_param_find / rad_param_getdim / rad_param_gets / rad_dtype_parse. Included by name rather
 * than leaned on through ref_ops.h, because what this file needs from the ABI is exactly this
 * header. */
#include "rad_plugin.h"
#include "rad_sample.h"

#include <cstring>
#include <cstdlib>
#include <initializer_list>

namespace {

/* ================================================================== parameters
 * NOT ref_common.h's p_int(): those take a RadArgs, and this hook is handed a bare RadParam list
 * with no tensors and no scratch -- it is called before anything is allocated. Same readings
 * though, including the RAD_P_RANGE one: a caller that has not collapsed a range yet is describing
 * the band's upper bound, which is the extent the arena has to be sized for. */
/* rad_param_getdim IS the int-or-range reading above. What stays local is the ONE deliberate
 * leniency on top of it: a STRING spelling of an integer is accepted, because this is the oracle
 * and a geometry it can describe is a case that gets checked rather than skipped. libr4d's shape
 * hooks refuse the same spelling on purpose -- its launches read integers as RAD_P_INT only, so
 * describing a geometry its launch would then refuse turns a skip into a failure. The two
 * libraries disagreeing here is intended; both saying so where they differ is the point. */
static int64_t gi(const RadParam* p, int n_p, const char* k, int64_t dflt) {
    const RadParam* q = rad_param_find(p, n_p, k);
    if (q && q->kind == RAD_P_STR && q->sval) return (int64_t)std::strtoll(q->sval, nullptr, 10);
    return (int64_t)rad_param_getdim(p, n_p, k, (long long)dflt);
}

static const char* gs(const RadParam* p, int n_p, const char* k) {
    return rad_param_gets(p, n_p, k, nullptr);
}

/* ================================================================== dtype names
 * THE ABI's PARSER, deliberately, rather than a table of spellings here. `rad_dtype_parse` is
 * `static inline` in rad_types.h, which rad_abi.h includes, which is the one header this plugin
 * compiles against -- so it is reachable and it is the single definition. A local copy would mean
 * a dtype added to the ABI reads as RAD_DT_INVALID here and correctly everywhere else. */
static uint32_t dt_parse(const char* n) { return rad_dtype_parse(n); }

/* THE ACTIVATION DTYPE, and the fallback matters. Several ops carry no `dtype` parameter at all --
 * `rope` names a `mode` and a `theta` and nothing about the width, because the tensor it rotates in
 * place is whatever the caller declared -- and falling back to f32 there manufactures an f32
 * activation, which a shim that reads a bf16 rotation then refuses ON DTYPE: a check that reports
 * FAIL on a kernel the engine runs correctly. `q_dtype` and `kv_dtype` are attention's spellings of
 * the same thing, and bf16 is what every activation in this project is when nothing says otherwise.
 * A dtype string the table does not know -- "w4a8", "fp8a8", the QUANTISATION SCHEME rather than an
 * element width -- lands on bf16 for the same reason. */
static uint32_t act_dt(const RadParam* p, int n_p) {
    const char* s = gs(p, n_p, "dtype");
    if (!s) s = gs(p, n_p, "q_dtype");
    if (!s) s = gs(p, n_p, "kv_dtype");
    const uint32_t dt = dt_parse(s);
    return dt == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;
}

/* ================================================================== the operand list
 * Every function below builds the WHOLE list and then hands out the entry that was asked for. That
 * is what lets the count be checked against the schema (see the header), and it costs a couple of
 * hundred bytes of stack per call on a hook that is called once per operand before anything is
 * allocated. */
enum { REF_MAX_OPD = 24 };   /* the widest schema here is gdn_conv_prep's sixteen */

struct Opd {
    RadOpdDesc v[REF_MAX_OPD];
    int        n = 0;
    bool       over = false;
    RadOpdDesc sink{};       /* somewhere for an overflowing push to land; `over` is the answer */

    RadOpdDesc* push() {
        if (n >= REF_MAX_OPD) { over = true; return &sink; }
        RadOpdDesc& d = v[n++];
        std::memset(&d, 0, sizeof d);
        d.fill = (uint32_t)RAD_FILL_NORMAL;
        /* -1 is the ABI's "draw", and it is written on EVERY descriptor rather than left to the
         * caller's zeroing: a caller that memsets the struct and does not pre-set this field would
         * read a 0 as "the constant zero", which is a different operand. */
        d.idx_const = -1;
        return &d;
    }

    static void shape(RadOpdDesc* d, uint32_t dt, std::initializer_list<int64_t> s) {
        d->dtype = dt;
        d->rank  = 0;
        for (int64_t e : s) {
            if (d->rank >= (uint32_t)RAD_MAX_RANK) break;
            d->shape[d->rank++] = e;
        }
    }

    /* An ordinary float operand: standard normals. */
    void t(uint32_t dt, std::initializer_list<int64_t> s) { shape(push(), dt, s); }

    /* A float operand whose DOMAIN is narrower than a normal draw -- a sigmoid gate, a monotone
     * log-decay cumsum, a chunk-triangular factor. `chunk` is the width the reset happens at. */
    void t_fill(uint32_t dt, std::initializer_list<int64_t> s, uint32_t fill, int64_t chunk) {
        RadOpdDesc* d = push();
        shape(d, dt, s);
        d->fill = fill;
        d->fill_chunk = chunk;
    }

    /* An index operand. `max` is the EXCLUSIVE upper bound of the draw and 0 means "the leading
     * extent of the previous operand", which is the common case and saves every gather from
     * restating it. `konst` >= 0 makes every element that value instead. */
    void t_idx(std::initializer_list<int64_t> s, int64_t max, int64_t konst, uint32_t flags) {
        RadOpdDesc* d = push();
        shape(d, (uint32_t)RAD_I32, s);
        d->fill      = (uint32_t)RAD_FILL_INDEX;
        d->idx_max   = max;
        d->idx_const = konst;
        d->flags     = flags;
    }

    /* AN INDEX OPERAND THAT IS NOT i32. `t_idx` hardcodes the index dtype, which is right for a
     * gather row or a block-table entry and wrong for a WEIGHT whose values happen to be sizes or
     * offsets: those are i64 in every container and every plugin refuses anything else by name, so
     * describing one as i32 makes the op refuse its own description. */
    void t_idx_dt(uint32_t dt, std::initializer_list<int64_t> s, int64_t max, int64_t konst,
                  uint32_t flags) {
        RadOpdDesc* d = push();
        shape(d, dt, s);
        d->fill      = (uint32_t)RAD_FILL_INDEX;
        d->idx_max   = max;
        d->idx_const = konst;
        d->flags     = flags;
    }

    /* An OPTIONAL operand this description deliberately does not pass. */
    void skip() {
        RadOpdDesc* d = push();
        d->flags = (uint32_t)RAD_OPD_F_ABSENT;
    }

    int finish(const char* op, int operand, RadOpdDesc* out) const;
};

/* How many operands the schema declares for this op, out of this plugin's own table. -1 if the op
 * is not ours, which cannot happen for a row that named one of these functions but is answered
 * rather than assumed. */
static int schema_operands(const char* op) {
    const int n = rad_kernel_schema_count();
    for (int i = 0; i < n; ++i) {
        const RadOpSchema* s = rad_kernel_schema_at(i);
        if (s && s->op && std::strcmp(s->op, op) == 0) return s->n_operands;
    }
    return -1;
}

int Opd::finish(const char* op, int operand, RadOpdDesc* out) const {
    if (over) return RAD_E_SHAPE;
    const int want = schema_operands(op);
    /* THE DESCRIPTION AND THE SCHEMA DISAGREE. That is a bug in one of them, and answering the
     * operands that do line up would hide it behind a number that looks like a result. */
    if (want != n) return RAD_E_SHAPE;
    if (operand < 0 || operand >= n || !out) return RAD_E_SHAPE;
    /* A ZERO OR NEGATIVE EXTENT means the geometry did not carry a parameter this description
     * needed -- an `n` that was never set, a head count that is not in the schema. That is a
     * DECLINE and not a bug: the caller skips and counts it, which is the honest outcome, because a
     * tool that invents an extent tests the kernel's bounds handling and calls it correctness. */
    for (int i = 0; i < n; ++i) {
        if (v[i].flags & RAD_OPD_F_ABSENT) continue;
        if (v[i].rank == 0) return RAD_E_UNSUPPORTED;
        for (uint32_t d = 0; d < v[i].rank; ++d)
            if (v[i].shape[d] <= 0) return RAD_E_UNSUPPORTED;
    }
    *out = v[operand];
    return RAD_OK;
}

/* ================================================================== shared readings */

/* The scale-stream granularity `quant_act_i8`, `quant_act_fp8` and `dequant` share, taken from the
 * launch's own clamping (ref_quant.cpp) rather than restated: group <= 0 or wider than the row is
 * ONE SCALE PER ROW, which is what every int8 GEMM in libr4d consumes, and anything else is
 * ceil(n / group) groups. Note this is NOT the reading the `had_*` forms use -- there `group` is
 * the Walsh-Hadamard rotation WIDTH and the scale is per row whatever it says. */
static int64_t scale_groups(int64_t n, int64_t group) {
    int64_t g = group;
    if (g <= 0 || g > n) g = n;
    return g > 0 ? (n + g - 1) / g : 0;
}

/* THE GATED-DELTA-NET HEAD COUNTS, which are not parameters of any gdn op: the kernels read them
 * off the operands, so no op names them and the resolved geometry does not carry them. A
 * description that invented a head ratio would be describing a model nobody is running, so the
 * whole family DECLINES unless the caller has put them in the parameter list under these names --
 * they come from the container's `linear_num_key_heads` / `linear_num_value_heads`, which is where
 * the architecture plugin reads them from too. Absent, the caller skips and counts. */
static bool gdn_heads(const RadParam* p, int n_p, int64_t* Hk, int64_t* Hv) {
    *Hk = gi(p, n_p, "n_head_k", 0);
    *Hv = gi(p, n_p, "n_head_v", 0);
    return *Hk > 0 && *Hv > 0;
}

/* Sequences in the described batch. Not a parameter of anything either -- one sequence is what a
 * cold call describes and what every recipe in opshapes.h assumed -- but a caller that wants a
 * ragged batch can say so and the gdn and attention descriptions will follow. */
static int64_t n_seq_of(const RadParam* p, int n_p) {
    const int64_t s = gi(p, n_p, "n_seq", 1);
    return s > 0 ? s : 1;
}

/* THE SAMPLER'S PER-ROW PARAMETER PLANE, one RadSampleParams per sampled position
 * (abi/rad_sample.h). It is a BYTE BLOB viewed through a 32-bit dtype -- the builder can
 * only say "this many words a row" -- and both this plugin and libr4d accept u32, i32 or f32 for
 * it.
 *
 * IT IS DESCRIBED AS f32, AND THAT IS A DELIBERATE CHOICE ABOUT CONTENT rather than about size.
 * The struct is eleven floats and nine int32s, and the ABI's fill domains describe a BUFFER and
 * not a struct, so there is no spelling for "fill this like a request". Drawn as f32 normals every
 * float field -- temp, top_p, min_p, typical_p, the three penalties, the two XTC values -- lands in
 * a plausible range, and a third to a half of the rows land inside the interval that makes their
 * stage do work (temp > 0, 0 < top_p < 1, and so on), so `sample_temp`, `sample_topp`,
 * `sample_minp`, `sample_typical` and `sample_xtc` are really exercised on some rows and identity
 * on others, which is the mix a real batch has.
 *
 * WHAT IT CANNOT REACH, said plainly rather than left for someone to discover: the INTEGER fields
 * come out as the bit patterns of those same normals, so `hist_off`, `hist_len` and `mask_row` are
 * enormous, every op clamps them, and `sample_penalties`, `sample_dry` and `sample_mask` therefore
 * take their "nothing to do for this row" branch. A pass on those three says the plumbing agrees,
 * not that the arithmetic does. `flags` is a bit pattern as well, so roughly half the rows carry
 * RAD_SP_GREEDY and every candidate stage treats them as having no candidates -- consistently in
 * both implementations, so the comparison covers the other half. The alternative -- a word-typed
 * draw -- is worse in both directions: it makes every float field a denormal or a NaN, and
 * `1 / temp` an infinity.
 *
 * u32 would also be invisible to the tools for a second reason: opshapes.cpp's rad_widen has no
 * RAD_U32 case, so a u32 operand is compared as zeroes and every case passes vacuously. Every
 * word-typed operand this file describes is therefore i32. */
static int64_t sample_params_words() {
    return (int64_t)(sizeof(RadSampleParams) / 4);   /* 96 bytes, 24 words */
}

}  /* namespace */

/* ================================================================== elementwise and norms */

int ref_rmsnorm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { n });             /* w */
    o.t(D, { M, n });          /* y */
    return o.finish("rmsnorm", operand, out);
}

int ref_rmsnorm_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, n });          /* residual */
    o.t(D, { n });             /* w */
    o.t(D, { M, n });          /* y */
    o.t(D, { M, n });          /* residual_out */
    return o.finish("rmsnorm_add", operand, out);
}

/* The gated residual. `inject` and `group` DEFAULT ON, so a query that names neither describes the
 * shape every trunk connection actually runs; passing either as 0 describes one of the two
 * exceptions (the final mixer, and a read whose consumer is bf16). That is the same rule the
 * schema's two optional parameters carry, said once more here because a cold caller reads this
 * and not the schema. */
int ref_hc_read_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0), lr = gi(p, n_p, "lowrank", 0);
    const int64_t hn = hc * n;
    const int64_t inject = gi(p, n_p, "inject", 1);
    const int64_t group  = gi(p, n_p, "group", 0);
    const int64_t rotate = gi(p, n_p, "rotate", 0);
    Opd o;
    o.t(D, { M, hn });                          /* h */
    /* The gain is F32 in the container, not the activation dtype: it is 10240 numbers read once a
     * connection and the norm's arithmetic is f32 anyway, so narrowing it buys nothing. */
    o.t((uint32_t)RAD_F32, { hn });             /* w */
    o.t(D, { lr, hn });                         /* mix_down */
    o.t(D, { hn, lr });                         /* mix_up */
    if (inject) o.t(D, { hc, hn }); else o.skip();   /* inject_w */
    o.t(D, { M, n });                           /* x */
    if (inject) o.t(D, { M, hc }); else o.skip();    /* inj */
    if (group > 0) {
        o.t((uint32_t)RAD_F8E4M3, { M, n });                       /* q */
        o.t((uint32_t)RAD_F32, { M, scale_groups(n, group) });     /* scale */
    } else {
        o.skip();
        o.skip();
    }
    if (group > 0 && rotate) {
        o.t((uint32_t)RAD_F8E4M3, { M, n });                       /* rq */
        o.t((uint32_t)RAD_F32, { M, scale_groups(n, group) });     /* rscale */
    } else {
        o.skip();
        o.skip();
    }
    return o.finish("hc_read", operand, out);
}

int ref_hc_enter_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, hc * n });     /* h */
    return o.finish("hc_enter", operand, out);
}

int ref_hc_write_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, n });          /* y */
    o.t(D, { M, hc });         /* inj */
    o.t(D, { M, hc * n });     /* h, in place */
    return o.finish("hc_write", operand, out);
}

int ref_mtp_enter_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, hc * n });                      /* h, the wide stream */
    o.t(D, { M, n });                           /* e, the next token's embedding */
    /* Both gains are F32 in the container for the reason hc_read's is: read once a pass,
     * multiplied in f32, and narrowing them buys nothing. */
    o.t((uint32_t)RAD_F32, { hc * n });         /* w_h */
    o.t((uint32_t)RAD_F32, { n });              /* w_e */
    o.t(D, { n, n });                           /* fc_h */
    o.t(D, { n, n });                           /* fc_e */
    o.t(D, { M, hc * n });                      /* x */
    return o.finish("mtp_enter", operand, out);
}

int ref_grid_embed_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), side = gi(p, n_p, "side", 0);
    if (M <= 0 || n <= 0 || side <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M, n });              /* x, INOUT */
    o.t(D, { side * side, n });    /* table */
    /* row, column, height, width: drawn over the table's side, so some rows sit past their grid's
     * last row and exercise the clamped taps as well as the interior. */
    o.t_idx({ 4, M }, side, -1, 0);
    return o.finish("grid_embed", operand, out);
}

int ref_layernorm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { n });             /* w */
    /* `b` is optional and is PASSED: a bias-free layer norm is the same code with one branch not
     * taken, so describing the bias covers both and omitting it covers one. */
    o.t(D, { n });             /* b */
    o.t(D, { M, n });          /* y */
    return o.finish("layernorm", operand, out);
}

/* add / mul share a schema and a description. An operand carrying exactly n elements against a
 * taller output would broadcast along rows, and that is a second case rather than this one: `b` is
 * described at full extent, because the broadcast form cannot be told from a shape bug once the
 * buffers exist. */
static int binary_shape(const RadParam* p, int n_p, int operand, const char* op, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* a */
    o.t(D, { M, n });          /* b */
    o.t(D, { M, n });          /* y */
    return o.finish(op, operand, out);
}

int ref_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, "add", out);
}
int ref_mul_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, "mul", out);
}

int ref_silu_mul_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, 2 * n });      /* gate_up -- the gate is the FIRST half */
    o.t(D, { M, n });          /* y */
    return o.finish("silu_mul", operand, out);
}

static int unary_shape(const RadParam* p, int n_p, int operand, const char* op, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, n });          /* y */
    return o.finish(op, operand, out);
}

int ref_gelu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "gelu", out);
}
int ref_silu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "silu", out);
}
int ref_sigmoid_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "sigmoid", out);
}
int ref_softmax_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "softmax", out);
}

int ref_gather_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    /* The source is FOUR TIMES the gathered height on purpose, so the gather is not the identity --
     * the identity is the one case that passes while broken. `idx` then draws from the previous
     * operand's leading extent, which is exactly the 4M rows that are legal to read. */
    o.t(D, { 4 * M, n });                                  /* x */
    o.t_idx({ M }, 0, -1, 0);                              /* idx */
    o.t(D, { M, n });                                      /* y */
    return o.finish("gather_rows", operand, out);
}


/* scatter_rows. The destination is four times the scattered height for the same reason gather's
 * source is -- a scatter onto exactly M rows with M unique indices is a permutation, and a broken
 * kernel that ignored `idx` entirely would still touch every row and could still pass. Wider than
 * the write, and the rows nobody named have to come back unchanged. */
int ref_scatter_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                                      /* v */
    o.t_idx({ M }, 4 * M, -1, RAD_OPD_F_IDX_UNIQUE);       /* idx, distinct destinations */
    o.t(D, { 4 * M, n });                                  /* x, inout */
    return o.finish("scatter_rows", operand, out);
}
/* PLE's n-gram ids. THE THREE WEIGHTS ARE I64 AND ARE NOT DRAWN LIKE INDICES: `mult` is a
 * multiplier in the 10^13 range, `vocab_sizes` a prime near 2*10^7 and `offsets` its prefix sum,
 * and a harness that drew them from an index distribution would produce a modulo by a small
 * number -- which passes while saying nothing, because every id would land in the first rows of a
 * 320-million-row table. They are described here with their real extents and left to the ordinary
 * weight draw; the op's own arithmetic is exact for whatever arrives (see ref_ngram.cpp on the
 * wrapping multiply and the flooring modulo).
 *
 * `tok` and `state` are index operands bounded by nothing, because a token id is bounded by the
 * VOCABULARY and this op never sees it. Drawing them over the whole i32 range is the honest
 * description and is also the case that exercises the wrap. */
/* PLE's gate. `v` is the ONE narrow operand -- n wide against everything else's hc*n -- and that
 * asymmetry is the op: a single value row gated differently into each of the hc streams. */
int ref_ple_gate_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D  = act_dt(p, n_p);
    const int64_t M   = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    const int64_t hn  = hc * n;
    Opd o;
    o.t(D, { M, hn });        /* k */
    o.t(D, { M, hn });        /* q */
    o.t(D, { M, n });         /* v */
    /* The three gains are F32 in the container for the reason hc_read's are: they are read once a
     * token, the norm's arithmetic is f32 anyway, and narrowing them buys nothing. */
    o.t((uint32_t)RAD_F32, { hn });   /* w_key */
    o.t((uint32_t)RAD_F32, { hn });   /* w_query */
    o.t((uint32_t)RAD_F32, { hn });   /* w_conv */
    o.t(D, { M, hn });        /* gv */
    o.t(D, { M, hn });        /* gvn */
    return o.finish("ple_gate", operand, out);
}

/* PLE's dilated convolution, described as the DECODE step it is on every token: `num_accepted`
 * present, so the window the step leaves behind is the shifted one (ref_ple.cpp) and the state is
 * (width-1)*dilation - 1 + M deep -- exactly what a step of M tokens needs, the way the manager's
 * (width-1)*dilation + n_spec is exactly what a step of 1 + n_spec needs. A state at the read
 * window's own depth would be refused for any M above one.
 *
 * WARM, and read at the DEEPEST offset that depth allows (`num_accepted` = M): a cold window or a
 * read at offset zero would leave the offset arithmetic and the shifted history unexercised, and
 * those are what two implementations of this op can disagree about. One sequence, because a shape
 * query carries no sequence count. */
int ref_ple_conv_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D  = act_dt(p, n_p);
    const int64_t M   = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t wid = gi(p, n_p, "width", 0), dil = gi(p, n_p, "dilation", 1);
    if (M <= 0 || n <= 0 || wid <= 0 || dil <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = 1, hist = (wid - 1) * dil, depth = hist - 1 + M;
    Opd o;
    o.t(D, { M, n });                                /* x   -- gvn */
    o.t(D, { n, wid });                              /* w */
    o.t(D, { S, n, depth });                         /* conv_state, INOUT */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);           /* cu */
    o.t(D, { M, n });                                /* resid -- gv */
    o.t_idx({ S }, S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);           /* cache_idx */
    /* WARM, SAID rather than left absent: ref reads a null `has_init` as warm too, but a plugin
     * need not, and a description should not lean on a default. */
    o.t_idx({ S }, 0, 1, 0);                         /* has_init */
    o.t_idx({ S }, 0, M, 0);                         /* num_accepted: read at M - 1 */
    o.t(D, { M, n });                                /* y */
    return o.finish("ple_conv", operand, out);
}

int ref_ngram_ids_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M     = gi(p, n_p, "M", 0);
    const int64_t heads = gi(p, n_p, "heads", 0);
    const int64_t ngram = gi(p, n_p, "ngram", 0);
    const int64_t eos   = gi(p, n_p, "eos", 0);
    const int64_t ctx   = ngram > 0 ? ngram - 1 : 0;
    /* One sequence, because `cu` is the only thing that says otherwise and a shape query carries
     * no sequence count. The op reads `numel(cu) - 1`, so this describes the single-sequence
     * launch every harness makes and every multi-sequence launch is the same code path.
     *
     * A DECODE STEP, for ple_conv's reason (see its description above): `num_accepted` present,
     * a warm window read at the deepest offset, and a state `ctx - 1 + M` deep, which is what the
     * shifted window a step of M ids leaves behind needs. */
    const int64_t S = 1, depth = ctx - 1 + M;
    Opd o;
    o.t_idx({ M }, 0, -1, 0);                                        /* tok */
    o.t_idx({ S, depth }, 0, -1, 0);                                 /* state, INOUT */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);           /* cu */
    o.t((uint32_t)RAD_I64, { ngram });                               /* mult */
    /* NOT A NORMAL DRAW, or the op refuses its own description: `ids[h] = floor_mod(mixed, v) +
     * off[h]` and the launch checks `v <= 0 || off[h] < 0`, which a standard normal cast to i64
     * fails on most heads. The refusal looks exactly like a geometry the plugin does not serve,
     * so rad-kbench would skip ngram_ids on every plugin without saying why.
     *
     * The vocabulary is a CONSTANT because a draw cannot promise positive and the modulus has to
     * be; `eos` is a token the head's vocabulary must contain, so eos + 1 is a bound derived from
     * this op's own parameters rather than a number invented here. The offsets ARE drawn, because
     * they only have to be non-negative -- and they are what gives the `heads_per_ngram` heads of
     * one n-gram block different answers, since `mixed` is shared across them. `idx_max` 0 is the
     * leading extent of the previous operand, which is `heads`. */
    o.t_idx_dt((uint32_t)RAD_I64, { heads }, 0, eos + 1, 0);         /* vocab_sizes */
    o.t_idx_dt((uint32_t)RAD_I64, { heads }, 0, -1, 0);              /* offsets */
    o.t_idx({ S }, S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);           /* cache_idx */
    o.t_idx({ S }, 0, 1, 0);                                         /* has_init: warm */
    o.t_idx({ S }, 0, M, 0);                                         /* num_accepted: at M - 1 */
    o.t((uint32_t)RAD_I32, { M, heads });                            /* ids */
    return o.finish("ngram_ids", operand, out);
}

int ref_cast_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    /* `from` and `to` are the only ops in this vocabulary whose operand dtypes come from the
     * parameters rather than from a shared activation dtype, and an unparseable spelling falls back
     * to f32 rather than to the activation default -- a cast with no `from` is a cast of nothing in
     * particular, and f32 is the widest thing it could have meant. */
    uint32_t from = dt_parse(gs(p, n_p, "from"));
    uint32_t to   = dt_parse(gs(p, n_p, "to"));
    if (from == (uint32_t)RAD_DT_INVALID) from = (uint32_t)RAD_F32;
    if (to   == (uint32_t)RAD_DT_INVALID) to   = (uint32_t)RAD_F32;
    Opd o;
    o.t(from, { M, n });       /* x */
    o.t(to,   { M, n });       /* y */
    return o.finish("cast", operand, out);
}

/* ================================================================== quantisation and rotation */

int ref_quant_act_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    /* `asum` selects libr4d's _asum form and REF DOES NOT WRITE IT: ref implements no asymmetric
     * grid, so it has no per-group code sums to produce and leaves the buffer alone. Passing it
     * anyway would compare a zeroed reference against a real one and report a failure that is a gap
     * in the ORACLE rather than a fault in the kernel. */
    o.skip();                               /* asum */
    return o.finish("quant_act_i8", operand, out);
}

int ref_quant_act_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    /* `dtype` names the INPUT here, which is bf16; the output width is what the op name says. */
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale, a DEQUANT multiplier (amax / 448) */
    return o.finish("quant_act_fp8", operand, out);
}

/* IDENTICAL EXTENTS TO quant_act_fp8'S, and that is the point worth stating: `group` is BOTH the
 * rotation width and the scale granularity here, unlike the int8 rotated forms below, where it is
 * the rotation width alone and the scale is per row. So the description is the unrotated one --
 * the rotation changes the numbers, not the shapes. */
/* `n` is K on BOTH axes of the accumulator, which is the one thing a caller coming from every
 * other op in this file gets wrong: it is not [M, n]. */
int ref_gram_accum_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const char* dt = gs(p, n_p, "dtype");
    Opd o;
    if (dt && std::strcmp(dt, "bf16") == 0) {
        o.t((uint32_t)RAD_BF16, { M, n });  /* a, read as it stands */
        o.skip();                           /* a_scale: a plain activation has none */
    } else {
        const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
        o.t((uint32_t)RAD_F8E4M3, { M, n });    /* a */
        o.t((uint32_t)RAD_F32, { M, ng });      /* a_scale, a DEQUANT multiplier (amax / 448) */
    }
    o.t((uint32_t)RAD_F32, { n, n });       /* h, READ and added to */
    return o.finish("gram_accum", operand, out);
}

int ref_had_quant_act_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale, a DEQUANT multiplier (amax / 448) */
    return o.finish("had_quant_act_fp8", operand, out);
}

/* The rotated forms share a scale convention that is NOT quant_act_i8's, and it is the one place
 * this file corrects the table it replaces. `group` here is the WALSH-HADAMARD ROTATION WIDTH
 * (libr4d calls it `had`) and the int8 scale is per ROW, carrying the 1/sqrt(group) normalisation
 * -- libref/ref_quant.cpp says so at length and the launch refuses anything shorter than one
 * scale a row. Describing the scale as [M, n/group] instead is not fatal (it is merely larger than
 * the op writes) but it leaves the tail of every scale buffer untouched by both implementations,
 * which is a comparison of two zero fills dressed up as agreement. */

int ref_had_quant_act_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale -- one per row */
    return o.finish("had_quant_act_i8", operand, out);
}

int ref_rmsnorm_had_quant_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                       /* x */
    /* `residual` PRESENT is the fused_add_rms_norm form, and it is the form worth describing: the
     * add happens first, the sum is written back through the INOUT operand, and the norm reads the
     * NARROWED sum. That rounding is the contract the fused kernel has to match, and it is only
     * exercised when the residual is passed. */
    o.t(D, { M, n });                       /* residual, INOUT */
    o.t(D, { n });                          /* w */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale -- one per row */
    /* `out_bf16` is the post-norm, PRE-ROTATION tensor for a layer that also has a bf16 consumer.
     * Both plugins write it when it is present, so it is described rather than skipped. */
    o.t(D, { M, n });                       /* out_bf16 */
    return o.finish("rmsnorm_had_quant_i8", operand, out);
}

int ref_gated_had_quant_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    /* THE PACKED FORM: `b` absent means `a` is [M, 2n], the gate is its first half and the up its
     * second, which is vLLM's layout and the one a fused gate_up projection actually produces. Both
     * readings exist and only one can be described; this is the one the MLP issues. */
    o.t(D, { M, 2 * n });                   /* a */
    o.skip();                               /* b */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale -- one per row */
    return o.finish("gated_had_quant_i8", operand, out);
}

int ref_dequant_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t(D, { M, n });                       /* y */
    return o.finish("dequant", operand, out);
}

/* ================================================================== gemm */

int ref_gemm_nt_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    Opd o;
    o.t(D, { M, K });          /* a */
    o.t(D, { N, K });          /* b -- stored transposed, so both operands are k-contiguous */
    o.t(D, { M, N });          /* y */
    return o.finish("gemm_nt", operand, out);
}

int ref_gemm_nt_bias_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    Opd o;
    o.t(D, { M, K });          /* a */
    o.t(D, { N, K });          /* b */
    o.t(D, { N });             /* bias */
    /* `res` is optional and PASSED, the layernorm bias's reason: with it the description covers
     * the residual epilogue, and a caller's plain form is the same code with a branch untaken. */
    o.t(D, { M, N });          /* res */
    o.t(D, { M, N });          /* y */
    return o.finish("gemm_nt_bias", operand, out);
}

/* BLOCK-SCALED FP8 ONLY, and the restriction is the honest one rather than laziness.
 *
 * `gemm_nt_q` is one op over six weight formats and the SCALE GRID is what differs between them --
 * per row, per row-group, per output-row-block, an e8m0 exponent byte, with or without a zero point
 * or a reference exponent. Every one of those is a property of the kernel that reads it and not of
 * the schema, so a description that guessed would be sizing five kernels' scale streams wrong and
 * calling whatever came back correctness.
 *
 * The fp8 family is different: its grid is stated by the CHECKPOINT -- E4M3 [N][K] against a bf16
 * [ceil(N/128)][ceil(K/128)] plane, with `group` carrying the 128 -- so there is nothing left to
 * guess. It is also the family the production model runs on. `a_sum` and `b_ref` belong to the
 * asymmetric-integer and mxfp4 grids and are absent; ref implements neither. */
static int gemm_q_shape(const RadParam* p, int n_p, int operand, bool bias, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const int64_t group = gi(p, n_p, "group", 0);
    const char* q = gs(p, n_p, "dtype");
    const bool a8  = q && std::strcmp(q, "fp8a8")  == 0;
    const bool a16 = q && std::strcmp(q, "fp8a16") == 0;
    if ((!a8 && !a16) || N <= 0 || K <= 0 || group <= 0) return RAD_E_UNSUPPORTED;
    /* The bias form takes `a_scale` as a REQUIRED operand, so the fp8a16 reading -- a bf16
     * activation and no activation scale at all -- cannot be expressed there. Declining is the
     * right answer rather than passing a scale stream the op does not mean. */
    if (bias && !a8) return RAD_E_UNSUPPORTED;

    const int64_t kb = (K + group - 1) / group;
    const int64_t nb = (N + group - 1) / group;
    Opd o;
    if (a8) {
        o.t((uint32_t)RAD_F8E4M3, { M, K });        /* a */
        o.t((uint32_t)RAD_F32, { M, kb });          /* a_scale */
    } else {
        o.t((uint32_t)RAD_BF16, { M, K });          /* a */
        o.skip();                                   /* a_scale */
    }
    o.t((uint32_t)RAD_F8E4M3, { N, K });            /* b -- the checkpoint's own bytes */
    o.t((uint32_t)RAD_BF16, { nb, kb });            /* b_scale -- weight_scale_inv, per 128x128 */
    if (bias) {
        o.t((uint32_t)RAD_BF16, { N });             /* bias */
        o.t((uint32_t)RAD_BF16, { M, N });          /* y */
        return o.finish("gemm_nt_q_bias", operand, out);
    }
    o.t((uint32_t)RAD_BF16, { M, N });              /* y */
    o.skip();                                       /* a_sum */
    o.skip();                                       /* b_ref */
    return o.finish("gemm_nt_q", operand, out);
}

int ref_gemm_nt_q_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gemm_q_shape(p, n_p, operand, false, out);
}
int ref_gemm_nt_q_bias_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gemm_q_shape(p, n_p, operand, true, out);
}

/* ================================================================== attention
 *
 * The paged pair share one layout and it is stated once for the whole project in ref_ops.h:
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]   K first, V second in a slot
 *
 * The extents they need beyond the parameters are a block COUNT and a context length, and both are
 * this description's to choose rather than the model's: any cache big enough for the case will do,
 * so long as the indices land inside it. */

int ref_attn_paged_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    /* `q_len` is the ranged parameter and it is a QUERY LENGTH, not a token count -- one query row
     * at decode and thousands in a prefill chunk are different kernels (docs/OPS.md). */
    const int64_t M   = gi(p, n_p, "q_len", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 16);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    const int64_t nh  = gi(p, n_p, "n_head", gqa);
    if (M <= 0 || hd <= 0 || bs <= 0 || gqa <= 0 || nh <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
    const int64_t kvh = nh / gqa;

    /* THE CONTEXT DEPTH, AND IT IS NOT THE QUERY LENGTH.
     *
     * The default is one sequence whose context is exactly its own query -- a causal attention
     * over its own prefix, which is the case a prefill chunk runs. On its own that default checks
     * a decode of ten query rows against ten tokens of KV and never describes the paged path at
     * any real depth: no block table long enough to rebase, no split over KV, no sliding window
     * with anything to slide past.
     * `ctx` is an ordinary parameter the harness sets when it sweeps depth, exactly as it hands
     * over the GDN head counts, and no op declares it because the KERNEL reads its context off
     * `seqused` -- which is right for the kernel and leaves only the description short.
     *
     * It must be at least the query length: a context shorter than the query makes causal
     * attention read a prefix that does not exist. */
    const int64_t S = 1;
    int64_t ctx = gi(p, n_p, "ctx", 0);
    if (ctx < M) ctx = M;
    const int64_t max_blocks = (ctx + bs - 1) / bs;

    /* `q_dtype` and `kv_dtype` are separate in this schema and are read separately: an fp8 cache
     * beside a bf16 query is the shape the engine actually issues, and folding them into one dtype
     * describes a call nobody makes. */
    const uint32_t Q  = act_dt(p, n_p);
    const uint32_t kv = dt_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? Q : kv;

    Opd o;
    o.t(Q,  { S * M, nh, hd });                      /* q */
    o.t(KV, { max_blocks, kvh, bs, 2 * hd });        /* kv_cache */
    o.t_idx({ S, max_blocks }, max_blocks, -1, 0);   /* block_table -- indexes BLOCKS, may repeat */
    /* NOT a draw: a context shorter than the query makes causal attention read a prefix that does
     * not exist, and both implementations would agree about the garbage. */
    o.t_idx({ S }, 0, ctx, 0);                       /* seqused */
    o.skip();                                        /* k_descale -- no per-(seq,head) descales */
    o.skip();                                        /* v_descale */
    /* `cu_seqlens` absent is the RECTANGULAR reading: every sequence presents the same query
     * length, which at one sequence is the only thing it could mean. Present it would have to agree
     * with the query's own row count, and the value a tool fills it with is the tool's `M` rather
     * than this description's -- so describing it would couple the two. */
    o.skip();                                        /* cu_seqlens */
    o.t(Q,  { S * M, nh, hd });                      /* out */
    return o.finish("attn_paged", operand, out);
}

int ref_attn_dense_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    if (M <= 0 || hd <= 0 || gqa <= 0) return RAD_E_UNSUPPORTED;
    /* THE HEAD COUNT IS NOT IN THIS SCHEMA -- `gqa` is the RATIO and the count is read off `q` at
     * launch -- so it is chosen here, and `gqa` query heads against one KV head is the smallest
     * geometry that exercises the grouping at all. That is a choice about the case and not about
     * the model: this op's operands are all created by the caller, so any head count produces a
     * well-formed dense attention. */
    const int64_t nh = gqa, kvh = 1;
    const uint32_t D = act_dt(p, n_p);
    const int64_t S = n_seq_of(p, n_p);

    Opd o;
    o.t(D, { M, nh, hd });                           /* q */
    o.t(D, { M, kvh, hd });                          /* k */
    o.t(D, { M, kvh, hd });                          /* v */
    /* [0, ..., T]: the batch's shape and not a draw. */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* cu_seqlens */
    o.t(D, { M, nh, hd });                           /* out */
    return o.finish("attn_dense", operand, out);
}

int ref_kv_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t kvh = gi(p, n_p, "n_head_kv", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 0);
    if (M <= 0 || hd <= 0 || kvh <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    /* Two blocks of slack so the slot mapping is not a dense prefix: a scatter that only ever
     * writes block 0 agrees with anything about the block index. */
    const int64_t nb = (M + bs - 1) / bs + 2;
    /* ONLY THE CACHE TAKES `kv_dtype`. k and v are the step's bf16 activations whatever the cache
     * stores -- an E4M3 cache is written from bf16 keys -- and act_dt, which falls back on
     * kv_dtype for an op with no `dtype`, would describe an fp8 k no engine passes. */
    const uint32_t D  = (uint32_t)RAD_BF16;
    const uint32_t kv = dt_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? D : kv;

    Opd o;
    o.t(D, { M, kvh, hd });                          /* k */
    o.t(D, { M, kvh, hd });                          /* v */
    /* A SCATTER DESTINATION: distinct, because two tokens on one slot make the store a race and
     * ref (an OpenMP token loop) and a device kernel (a workgroup a token) resolve it differently.
     * The range is the whole cache in SLOTS, which is no operand's dim 0. */
    o.t_idx({ M }, nb * bs, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* slot_mapping */
    o.t(KV, { nb, kvh, bs, 2 * hd });                /* kv_cache, INOUT */
    return o.finish("kv_store", operand, out);
}

int ref_query_fwht_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, 256 });
    return o.finish("query_fwht", operand, out);
}

int ref_rope_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t nh  = gi(p, n_p, "n_head", 0);
    const int64_t nkv = gi(p, n_p, "n_head_kv", nh);
    if (M <= 0 || hd <= 0 || nh <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);   /* no `dtype` on this op: bf16, see act_dt */
    Opd o;
    /* The fused row: n_head queries and n_head_kv each of k and v, all head_dim wide. The v heads
     * are left alone, which is the whole reason this op takes the fused row rather than two
     * tensors. */
    o.t(D, { M, (nh + 2 * nkv) * hd });              /* qkv, INOUT */
    /* A multi-component mode with its sections takes a position per component, component-major;
     * the count is the sections' (three for the interleaved form, which always reads three). */
    const char* mode = gs(p, n_p, "mode");
    if (!mode) mode = "neox";
    const char* sec  = gs(p, n_p, "sections");
    int64_t nc = 0;
    for (const char* q = sec; q && *q; ) {
        char* e = nullptr;
        (void)std::strtoll(q, &e, 10);
        if (e == q) break;
        ++nc;
        q = e;
    }
    const bool multi = nc > 0 && (!std::strcmp(mode, "mrope") || !std::strcmp(mode, "imrope") ||
                                  !std::strcmp(mode, "axial"));
    if (multi) o.t_idx({ !std::strcmp(mode, "imrope") ? 3 : nc, M }, M, -1, 0);
    else       o.t_idx({ M }, 0, -1, 0);             /* positions */
    return o.finish("rope", operand, out);
}

int ref_qk_norm_rope_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t nh  = gi(p, n_p, "n_head", 0);
    const int64_t nkv = gi(p, n_p, "n_head_kv", nh);
    if (M <= 0 || hd <= 0 || nh <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    /* THIS IS NOT libr4d's `qk_norm_rope_gate`, which opshapes.h declines and is right to: that op
     * reads an INTERLEAVED strided view -- attn_q is [n_embd, 2 * head_dim * n_head] with query and
     * gate alternating per head, so it is not a contiguous split and no description built out of
     * the parameters could know it. The vocabulary's `qk_norm_rope` takes the same contiguous fused
     * [q|k|v] row `rope` does, so it is describable and is described. */
    o.t(D, { M, (nh + 2 * nkv) * hd });              /* qkv, INOUT */
    o.t_idx({ M }, 0, -1, 0);                        /* positions */
    o.t(D, { hd });                                  /* q_w -- one gain per head channel */
    o.t(D, { hd });                                  /* k_w */
    return o.finish("qk_norm_rope", operand, out);
}

int ref_rope_table_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t rot = gi(p, n_p, "rot", 0);
    if (M <= 0 || rot < 2 || (rot & 1)) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    /* THE TABLE HEIGHT IS THIS DESCRIPTION'S CHOICE and the position range has to be stated for it.
     * Each position writes its OWN row of a [max_ctx, rot] plane, `max_ctx` is not an operand of
     * this op, and a row no position names is left untouched -- so a table of M rows with positions
     * drawn from [0, M) is the case where every row is reachable. Left at the default the draw
     * would come from the previous operand's leading extent, and `positions` IS the first operand:
     * the caller would fall back to the container's vocabulary size and write nothing at all. */
    o.t_idx({ M }, M, -1, 0);                        /* positions */
    o.t(D, { M, rot });                              /* cos_sin -- cos in the first half, sin in the second */
    return o.finish("rope_table", operand, out);
}

/* ================================================================== mixture of experts */

int ref_scale_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                /* x */
    o.t(D, { M });                                   /* s -- one scalar a row */
    o.t(D, { M, n });                                /* add -- optional, described anyway: a tool
                                                      * calling this cold gets a buffer for it and
                                                      * exercises the fused form */
    o.t(D, { M, n });                                /* y */
    return o.finish("scale_rows", operand, out);
}

int ref_router_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    if (M <= 0 || ne <= 0 || k <= 0 || k > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });                               /* logits */
    o.t((uint32_t)RAD_I32, { M, k });                /* expert_ids */
    o.t(D, { M, k });                                /* expert_w */
    return o.finish("router_topk", operand, out);
}

int ref_moe_scatter_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    if (M <= 0 || ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    /* Expert ids index EXPERTS, which is no operand's dim 0, so the range is stated. A slot whose
     * expert is outside [0, n_expert) is dropped by the op and a draw inside it exercises the sort
     * rather than the dropping; the dropping is the caller's to construct if it wants it. */
    o.t_idx({ M, k }, ne, -1, 0);                    /* expert_ids */
    o.t((uint32_t)RAD_I32, { M * k });               /* sorted_tok -- token*top_k + slot, by expert */
    o.t((uint32_t)RAD_I32, { ne + 1 });              /* expert_offset */
    o.t((uint32_t)RAD_I32, { ne });                  /* expert_count */
    return o.finish("moe_scatter", operand, out);
}

/* `a_order` decides the activation's row count: one per token, or one per (token, slot) already
 * in the scatter's order. A described case that got this wrong would build the operand at the
 * extent the WRONG reading wants, and the fault it names would go unseen. */
static inline bool sorted_a_order(const RadParam* p, int n_p) {
    const char* s = gs(p, n_p, "a_order");
    return s && std::strcmp(s, "sorted") == 0;
}

int ref_moe_gemm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t N  = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    if (M <= 0 || N <= 0 || K <= 0 || ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    const int64_t AR = sorted_a_order(p, n_p) ? M * k : M;
    o.t(D, { AR, K });                               /* a -- see `a_order` */
    o.t(D, { ne, N, K });                            /* w */
    o.t_idx({ M * k }, M * k, -1, 0);                /* sorted_tok */
    /* `expert_offset` IS NOT A FREE DRAW and the ABI has no fill domain that says so: it is
     * moe_scatter's output, a non-decreasing prefix array ending at the live row count, and the
     * expert that owns a sorted row is found by searching it. Drawn at random it is a malformed
     * histogram, so two implementations that search it differently -- a binary search here, a
     * linear scan there -- can pick different experts and disagree without either being wrong.
     * The extents below are still exactly right, which is what a benchmark needs; a correctness
     * comparison over this operand proves less than it appears to, and that is worth knowing
     * before reading a result. */
    o.t_idx({ ne + 1 }, M * k + 1, -1, 0);           /* expert_offset */
    o.t(D, { M * k, N });                            /* y -- in SORTED order */
    return o.finish("moe_gemm", operand, out);
}

/* THE WEIGHT SCALE GRID IS THE ONE `dtype` NAMES, and the grouped GEMM serves more than one. The
 * block-fp8 experts (fp8a8) carry the checkpoint's: a scale a 128x128 tile, `group` holding the
 * 128. The four-bit ones (docs/MOE-W4.md) carry one a ROW per group of K -- all of `group` at
 * w4a8, w4a8h and w4nla8h, half of it at w4nl64a8h and w5nl64a8h, a quarter at w4nl32a8h. Those
 * strings are the served formats' own names (arch/common/rad_block_moe_fp8.h, kServed), so this
 * states a format and guesses at no kernel's stream. Any other dtype, or none, names no grid this
 * file knows: the answer is { 0, 0 } and the description declines (RAD_E_UNSUPPORTED), because a
 * scale plane sized on a guessed grid is a tensor of the wrong extent handed to the oracle.
 *
 * It matters past the extents. A weight its kernel stores re-laid goes back through that
 * kernel's inverse into the tensor THIS describes before the oracle hands it over
 * (core/runtime/oracle.cpp), and the E4M3 scale views are re-laid -- their two levels are one
 * operand -- so tiles here leave every such expert's scales with nothing to be inverted into.
 * The arithmetic reads whichever grid it is handed (ref_gemm.h derives the group from the
 * extents) and is unchanged. Weight rows and K elements a scale covers, or 0 when `group` is too
 * small to divide or the dtype names no grid. */
struct MoeScaleGrid { int64_t rows, k; };
static MoeScaleGrid moe_scale_grid(const RadParam* p, int n_p, int64_t g) {
    const char* d = gs(p, n_p, "dtype");
    auto is = [&](const char* s) { return d && std::strcmp(d, s) == 0; };
    if (is("fp8a8")) return { g, g };
    if (is("w4a8") || is("w4a8h") || is("w4nla8h")) return { 1, g };
    if (is("w4nl64a8h") || is("w5nl64a8h")) return { 1, g / 2 };
    if (is("w4nl32a8h")) return { 1, g / 4 };
    return { 0, 0 };
}

/* The quantised arm. Same planes, plus the two scale streams: the activation's is per row per
 * group of K (rblk 1) and the weight's is one plane per expert on the grid above -- for block fp8
 * [ceil(N/g), ceil(K/g)], the DeepSeek block grid the checkpoint ships. `w` is described as the
 * STACKED [ne, N, K] plane even though the engine issues a pointer table against it, because a
 * fixture has nowhere to put a pointer array -- ref_gemm.h's expert_entry() reads both forms and
 * rad-kbench therefore checks the arithmetic against the same bytes the engine would. */
int ref_moe_gemm_q_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t N0 = gi(p, n_p, "N", 0), K0 = gi(p, n_p, "K", 0);
    const int64_t N1 = gi(p, n_p, "N_odd", N0), K1 = gi(p, n_p, "K_odd", K0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    int64_t g = gi(p, n_p, "group", 0);
    if (M <= 0 || N0 <= 0 || K0 <= 0 || N1 <= 0 || K1 <= 0 || ne <= 0 || k <= 0)
        return RAD_E_UNSUPPORTED;
    /* The output is the wider class's width and the activation the longer class's K; the two
     * tables are the even and the odd experts. */
    const int64_t N = N0 > N1 ? N0 : N1, K = K0 > K1 ? K0 : K1;
    const int64_t ne0 = (ne + 1) / 2, ne1 = ne / 2;
    if (g <= 0) g = K;
    const MoeScaleGrid sg = moe_scale_grid(p, n_p, g);
    if (sg.rows <= 0 || sg.k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    const int64_t AR = sorted_a_order(p, n_p) ? M * k : M;
    o.t((uint32_t)RAD_F8E4M3, { AR, K });             /* a -- E4M3 codes, see `a_order` */
    o.t((uint32_t)RAD_F32, { AR, (K + g - 1) / g });   /* a_scale */
    o.t((uint32_t)RAD_F8E4M3, { ne0, N0, K0 });       /* w -- the even experts */
    o.t(D, { ne0, (N0 + sg.rows - 1) / sg.rows, (K0 + sg.k - 1) / sg.k });   /* w_scale */
    o.t_idx({ M * k }, M * k, -1, 0);                 /* sorted_tok */
    o.t_idx({ ne + 1 }, M * k + 1, -1, 0);            /* expert_offset -- see moe_gemm's note */
    o.t((uint32_t)RAD_F8E4M3, { ne1, N1, K1 });       /* w_odd -- the odd experts */
    o.t(D, { ne1, (N1 + sg.rows - 1) / sg.rows, (K1 + sg.k - 1) / sg.k });   /* w_odd_scale */
    o.t(D, { M * k, N });                             /* y -- in SORTED order */
    return o.finish("moe_gemm_q", operand, out);
}

/* The four-class form: class q at [N_q, K_q], defaulting to class 0's, holds the experts e with
 * e % 4 == q -- ceil((n - q) / 4) of them -- and `y` follows the last class. */
int ref_moe_gemm_q_mod4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t N0 = gi(p, n_p, "N", 0), K0 = gi(p, n_p, "K", 0);
    const int64_t Nq[4] = { N0, gi(p, n_p, "N_1", N0), gi(p, n_p, "N_2", N0), gi(p, n_p, "N_3", N0) };
    const int64_t Kq[4] = { K0, gi(p, n_p, "K_1", K0), gi(p, n_p, "K_2", K0), gi(p, n_p, "K_3", K0) };
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    int64_t g = gi(p, n_p, "group", 0);
    if (M <= 0 || ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    int64_t N = 0, K = 0;
    for (int q = 0; q < 4; ++q) {
        if (Nq[q] <= 0 || Kq[q] <= 0) return RAD_E_UNSUPPORTED;
        if (Nq[q] > N) N = Nq[q];
        if (Kq[q] > K) K = Kq[q];
    }
    if (g <= 0) g = K;
    const MoeScaleGrid sg = moe_scale_grid(p, n_p, g);
    if (sg.rows <= 0 || sg.k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    const int64_t AR = sorted_a_order(p, n_p) ? M * k : M;
    auto cls = [&](int q) {
        const int64_t nq = (ne + 3 - q) / 4;
        o.t((uint32_t)RAD_F8E4M3, { nq, Nq[q], Kq[q] });
        o.t(D, { nq, (Nq[q] + sg.rows - 1) / sg.rows, (Kq[q] + sg.k - 1) / sg.k });
    };
    o.t((uint32_t)RAD_F8E4M3, { AR, K });             /* a -- E4M3 codes, see `a_order` */
    o.t((uint32_t)RAD_F32, { AR, (K + g - 1) / g });   /* a_scale */
    cls(0);                                           /* w, w_scale -- class 0 */
    o.t_idx({ M * k }, M * k, -1, 0);                 /* sorted_tok */
    o.t_idx({ ne + 1 }, M * k + 1, -1, 0);            /* expert_offset */
    cls(1);
    cls(2);
    cls(3);
    o.t(D, { M * k, N });                             /* y -- in SORTED order */
    return o.finish("moe_gemm_q_mod4", operand, out);
}

int ref_moe_gather_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t k = gi(p, n_p, "top_k", 0);
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M * k, n });                            /* y_expert -- one row per (token, slot) */
    o.t(D, { M, k });                                /* expert_w */
    /* DISTINCT, and this one is a read-side index that still has to be: gather rebuilds the INVERSE
     * permutation of sorted_tok, so a repeated entry makes two sorted rows claim one (token, slot)
     * and the last writer wins -- serially here, by whichever workgroup lands last on a device. A
     * full permutation of [0, M*top_k) is what moe_scatter produces and is what is described. */
    o.t_idx({ M * k }, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* sorted_tok */
    o.t(D, { M, n });                                /* shared -- optional, described anyway */
    o.t(D, { M });                                   /* shared_gate */
    o.t(D, { M, n });                                /* y */
    return o.finish("moe_gather", operand, out);
}

int ref_row_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), R = gi(p, n_p, "R", 0);
    if (M <= 0 || N <= 0 || R <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, N });                                /* x */
    o.t((uint32_t)RAD_I32, { M, R });                /* idx */
    o.t((uint32_t)RAD_F32, { M, R });                /* val */
    /* `pairs` is the vocab-parallel plane an all_gather moves and it is described rather than
     * skipped, because it is an OUTPUT: nothing has to fill it, and comparing it is what checks the
     * packing convention -- an int32 id memcpy'd into the first float slot beside its value, with
     * id -1 for a slot the row could not fill. Both plugins require f32 here for exactly that
     * reason: the id is carried by BIT PATTERN and a word-typed plane could not hold the value. */
    o.t((uint32_t)RAD_F32, { M, R, 2 });             /* pairs */
    return o.finish("row_topk", operand, out);
}

int ref_row_topk_merge_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), R = gi(p, n_p, "R", 0);
    const int64_t W = gi(p, n_p, "world_size", 1);
    if (M <= 0 || R <= 0 || W <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    /* THE ONE PACKED PAIR PLANE THIS FILE CAN FILL, and it is worth saying why it works here and
     * not on `sample_merge_topk`. The plane is f32, so a normal draw puts a PROPER FLOAT in every
     * value lane -- which is the lane the merge orders on -- while the id lane is that same float's
     * bit pattern read as an int32: negative floats become negative ids and are dropped as pads,
     * positive ones become large but perfectly legal ids that the op only ever compares against
     * each other and copies out. Ties are then impossible and the answer is exact. */
    o.t((uint32_t)RAD_F32, { M, W * R, 2 });         /* gathered */
    o.t((uint32_t)RAD_I32, { M, R });                /* idx */
    o.t((uint32_t)RAD_F32, { M, R });                /* val */
    return o.finish("row_topk_merge", operand, out);
}



/* logit_rerank DESCRIBES ITSELF, AND THE READER THAT NEEDS IT INVERTS FIRST.
 *
 * The head operand is the lm_head, which libr4d STORES as its own E4M3 row plane (r4d_fp8_lm.h)
 * while the declaration says bf16. A SYNTHETIC case built from this description would hand libref
 * a bf16 weight and libr4d the identical bytes read as fp8 codes, so the two would compute from
 * different numbers -- a disagreement far outside any tolerance with no defect behind it. That is
 * why a recording made through the fp8 row is skipped by name, exactly as logits_gemm's is.
 *
 * Declining the description does not buy that skip and costs the OTHER reader everything. The op
 * oracle puts a relaid weight back through the device row's own RadUnrelayoutFn to its canonical
 * planes and decodes them -- for this plane that DEQUANTISES -- so the dense weight it hands the
 * reference is exactly what the kernel's arithmetic saw, and all it then needs is a logical form
 * to write it into. Without one, the draft-head rerank goes unchecked on the engine's own
 * operands, reported as "nothing could describe the logical form of 'output.weight'".
 * ref_logits_gemm_shape describes the same weight, for the same reason. */
int ref_logit_rerank_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0), R = gi(p, n_p, "R", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0), ne = gi(p, n_p, "n_embd", 0);
    const int64_t vo = gi(p, n_p, "vocab_off", 0);
    if (M <= 0 || R <= 0 || nv <= 0 || ne <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });                               /* x      */
    o.t(D, { nv, ne });                              /* head   */
    /* The candidate ids are GLOBAL, so the draw spans the whole vocabulary this rank can name and
     * the ones below its own slice are the pads the op reports as -1. */
    o.t_idx({ M, R }, vo + nv, -1, 0);                /* idx_in */
    o.t_idx({ M, R }, vo + nv, -1, 0);                /* idx    */
    o.t((uint32_t)RAD_F32, { M, R });                /* val    */
    /* f32 and [M, R, 2] for the same reason row_topk's is: the id rides in the first lane as a
     * bit pattern beside its value, so a word-typed plane could not hold the value. */
    o.t((uint32_t)RAD_F32, { M, R, 2 });             /* pairs  */
    return o.finish("logit_rerank", operand, out);
}
/* ================================================================== gated delta net
 *
 * The layouts are libref/ref_gdn.cpp's, stated there once:
 *
 *     q, k     [T, Hk, head_k]         v, o  [T, Hv, head_v]
 *     g, beta  [T, Hv]  fp32           A     [T, Hv, chunk]
 *     h0, ht   [N, Hv, head_v, head_k] fp32  cu    [N+1]
 *     conv w   [conv_dim, conv_width]        conv_state [n_slots, conv_dim, state_len]
 *
 * with conv_dim = 2 * Hk * head_k + Hv * head_v.
 *
 * THE HEAD COUNTS ARE NOT PARAMETERS of any of these ops -- the kernels read them off the operands
 * -- so the whole family declines unless the caller has supplied them (see gdn_heads above). A GDN
 * case run at an invented head ratio is a case of a model nobody is running. */

/* Everything the five gdn descriptions share, resolved once. Returns false to decline. */
/* ================================================================== the fp8 fusions
 *
 * libr4d fuses the fp8 activation quantiser into whichever kernel already had the row in
 * registers. The shapes are the unfused pair's, which is the whole point of the fusion: the same
 * operands, minus the bf16 tensor between them -- and `out_bf16` puts even that back, because the
 * codes are taken over its value and a caller may need to compare them.
 */
int ref_rmsnorm_quant_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t(D, { M, n });                       /* residual, inout -- the fused_add_rms_norm form */
    o.t(D, { n });                          /* w, the gain */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t((uint32_t)RAD_BF16, { M, n });      /* out_bf16 -- what the codes were taken over */
    return o.finish("rmsnorm_quant_fp8", operand, out);
}

int ref_gated_quant_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, 2 * n });                   /* gate_up, gate half first */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t((uint32_t)RAD_BF16, { M, n });      /* out_bf16 */
    return o.finish("gated_quant_fp8", operand, out);
}

/* `N` IS THE WEIGHT'S ROWS AND IT IS TWICE THE FUSED WIDTH, which is the one thing a caller coming
 * from gemm_nt_q gets wrong here: the quantised output is [M, N/2], not [M, N]. */
int ref_gemm_nt_q_gated_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const int64_t group = gi(p, n_p, "group", 0);
    const char* q = gs(p, n_p, "dtype");
    const bool a8 = q && std::strcmp(q, "fp8a8") == 0;
    if (!a8 || N <= 0 || K <= 0 || group <= 0 || (N & 1)) return RAD_E_UNSUPPORTED;
    const int64_t n  = N / 2;
    const int64_t kb = (K + group - 1) / group;
    const int64_t nb = (N + group - 1) / group;
    const int64_t ng = scale_groups(n, group);
    Opd o;
    o.t((uint32_t)RAD_F8E4M3, { M, K });    /* a */
    o.t((uint32_t)RAD_F32, { M, kb });      /* a_scale */
    o.t((uint32_t)RAD_F8E4M3, { N, K });    /* b */
    o.t((uint32_t)RAD_BF16, { nb, kb });    /* b_scale */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t((uint32_t)RAD_BF16, { M, n });      /* out_bf16 */
    return o.finish("gemm_nt_q_gated", operand, out);
}

/* DFlash2's candidate walk. `cand` holds TOKEN IDS drawn from the vocabulary and `anchor` one per
 * sequence, so both are bounded by n_vocab rather than by a row count -- a draw outside it would
 * make the codebook gather read past the plane, which is the one thing a description can prevent
 * here. The rows of `cand`/`unary`/`hp` are (sequence, step) pairs, M * steps of them. */
int ref_dflash_select_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M     = gi(p, n_p, "M", 0);
    const int64_t steps = gi(p, n_p, "steps", 0);
    const int64_t K     = gi(p, n_p, "top_k", 0);
    const int64_t R     = gi(p, n_p, "rank", 0);
    const int64_t nv    = gi(p, n_p, "n_vocab", 0);
    const int64_t astr  = gi(p, n_p, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_UNSUPPORTED;
    const int64_t rows = M * steps;
    Opd o;
    o.t_idx_dt((uint32_t)RAD_I32, { rows, K }, nv, -1, 0);   /* cand */
    o.t((uint32_t)RAD_F32, { rows, K });                     /* unary */
    o.t((uint32_t)RAD_BF16, { rows, R });                    /* hp, absent is a plane of ones */
    o.t_idx_dt((uint32_t)RAD_I32, { M * astr }, nv, -1, 0);  /* anchor */
    o.t((uint32_t)RAD_BF16, { nv, R });                      /* pred */
    o.t((uint32_t)RAD_BF16, { nv, R });                      /* succ */
    o.t_idx_dt((uint32_t)RAD_I32, { M, steps }, nv, -1, 0);  /* tokens */
    return o.finish("dflash_select", operand, out);
}

namespace {
struct GdnGeom {
    int64_t Hk, Hv, hk, hv, chunk, width, conv_dim, S, M;
    uint32_t D;
};

static bool gdn_geom(const RadParam* p, int n_p, const char* mkey, GdnGeom* g) {
    if (!gdn_heads(p, n_p, &g->Hk, &g->Hv)) return false;
    g->hk = gi(p, n_p, "head_k", 0);
    g->hv = gi(p, n_p, "head_v", g->hk);
    g->chunk = gi(p, n_p, "chunk", 64);
    g->width = gi(p, n_p, "conv_width", 4);
    g->M = gi(p, n_p, mkey, 0);
    g->S = n_seq_of(p, n_p);
    g->D = act_dt(p, n_p);
    if (g->hk <= 0 || g->hv <= 0 || g->chunk <= 0 || g->width <= 1 || g->M <= 0) return false;
    g->conv_dim = 2 * g->Hk * g->hk + g->Hv * g->hv;
    return true;
}

}  /* namespace */

int ref_gdn_conv_prep_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "M", &g)) return RAD_E_UNSUPPORTED;
    Opd o;
    /* T tokens over S sequences, one conv-state slot each. `state_len` is the manager's
     * conv_width - 1 + n_spec; with no speculation that is conv_width - 1, and the kernel reads its
     * window at the slot the last committed token left. */
    o.t(g.D, { g.M, g.conv_dim });                   /* x -- the fused projection row */
    o.t(g.D, { g.conv_dim, g.width });               /* w */
    o.skip();                                        /* b -- this checkpoint has no conv bias */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* A_log */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* dt_bias */
    o.t(g.D, { g.S, g.conv_dim, g.width - 1 });      /* conv_state, INOUT */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    /* `a` and `b_gate` PRESENT: the layer's separate gate and beta projections. Absent they would
     * be the last 2*Hv columns of the fused `x` row, which would make `x` wider and hide the
     * separate-projection path that every current caller takes. */
    o.t(g.D, { g.M, g.Hv });                         /* a */
    o.t(g.D, { g.M, g.Hv });                         /* b_gate */
    o.t_idx({ g.S }, g.S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* cache_idx, one a sequence */
    /* A COLD PREFILL, SAID rather than left absent: ref reads a null `has_init` as WARM and libr4d
     * as COLD, so the default is the one thing the two do not share. */
    o.t_idx({ g.S }, 0, 0, 0);                       /* has_init */
    o.t(g.D, { g.M, g.Hk, g.hk });                   /* q */
    o.t_fill(g.D, { g.M, g.Hk, g.hk }, (uint32_t)RAD_FILL_UNIT_ROW, 0);                  /* k */
    o.t(g.D, { g.M, g.Hv, g.hv });                   /* v */
    /* g and beta are OUTPUTS here and their domains are stated anyway: they are the same tensors
     * gdn_kkt_solve and gdn_chunk_scan take as INPUTS, and the domain is the op's contract for what
     * it produces rather than a property of who is filling the buffer. */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_GATE_CUMSUM, g.chunk);  /* g */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_SIGMOID, 0);            /* beta */
    return o.finish("gdn_conv_prep", operand, out);
}

int ref_gdn_conv_update_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "q_len", &g)) return RAD_E_UNSUPPORTED;
    /* The decode pair. `M` is q_len here, one row per candidate token, and the window is
     * conv_width - 2 + q_len deep so a rejection is a change of READ OFFSET rather than a recompute
     * (ref_ops.h's rollback contract). */
    const int64_t sl = g.width - 2 + g.M;
    Opd o;
    o.t(g.D, { g.S * g.M, g.conv_dim });             /* x */
    o.t(g.D, { g.conv_dim, g.width });               /* w */
    o.skip();                                        /* b */
    o.t(g.D, { g.S, g.conv_dim, sl });               /* conv_state, INOUT */
    o.t_idx({ g.S, g.M }, g.S, -1, 0);               /* state_idx */
    /* num_accepted is 1-BASED and all ones is the non-speculative decode: "none of the drafts
     * survived, only the token that was already committed". A draw here would move the read offset
     * to a slot this step's window does not cover. */
    o.t_idx({ g.S }, 0, 1, 0);                       /* num_accepted */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { g.S * g.M, g.Hk, g.hk });             /* q */
    o.t(g.D, { g.S * g.M, g.Hk, g.hk });             /* k */
    o.t(g.D, { g.S * g.M, g.Hv, g.hv });             /* v */
    return o.finish("gdn_conv_update", operand, out);
}

int ref_gdn_kkt_solve_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "M", &g)) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_fill(g.D, { g.M, g.Hk, g.hk }, (uint32_t)RAD_FILL_UNIT_ROW, 0);                  /* k */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_SIGMOID, 0);            /* beta */
    /* NON-INCREASING AND NEVER POSITIVE, reset at every chunk boundary. The solve forms
     * e^{g_i - g_j} with j <= i and that is <= 1 only because g is monotone. */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_GATE_CUMSUM, g.chunk);  /* g */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t_fill(g.D, { g.M, g.Hv, g.chunk }, (uint32_t)RAD_FILL_TRI_LOWER, g.chunk);         /* A */
    return o.finish("gdn_kkt_solve", operand, out);
}

int ref_gdn_chunk_scan_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "M", &g)) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(g.D, { g.M, g.Hk, g.hk });                   /* q */
    o.t_fill(g.D, { g.M, g.Hk, g.hk }, (uint32_t)RAD_FILL_UNIT_ROW, 0);                  /* k */
    o.t(g.D, { g.M, g.Hv, g.hv });                   /* v */
    /* ZERO ABOVE THE DIAGONAL WITHIN A CHUNK. A is the chunk's triangular inverse and the scan
     * multiplies by it; a dense random A puts two implementations on opposite sides of a convention
     * -- whether the strictly-upper part is read at all -- that they never disagree about on a real
     * one, because the same library's kkt_solve produced it. */
    o.t_fill(g.D, { g.M, g.Hv, g.chunk }, (uint32_t)RAD_FILL_TRI_LOWER, g.chunk);         /* A */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_GATE_CUMSUM, g.chunk);  /* g */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_SIGMOID, 0);            /* beta */
    o.t((uint32_t)RAD_F32, { g.S, g.Hv, g.hv, g.hk });               /* h0 */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { g.M, g.Hv, g.hv });                   /* o */
    o.t((uint32_t)RAD_F32, { g.S, g.Hv, g.hv, g.hk });               /* ht */
    o.t_idx({ g.S }, g.S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* state_idx */
    return o.finish("gdn_chunk_scan", operand, out);
}

int ref_gdn_recurrent_update_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "q_len", &g)) return RAD_E_UNSUPPORTED;
    const int64_t T = g.S * g.M;
    Opd o;
    o.t(g.D, { T, g.Hk, g.hk });                     /* q */
    o.t(g.D, { T, g.Hk, g.hk });                     /* k */
    o.t(g.D, { T, g.Hv, g.hv });                     /* v */
    /* `a` and `b` are the RAW gate and beta projections here, not the cumsum and the sigmoid: this
     * op applies softplus and sigmoid itself (o = -exp(A_log) . softplus(a + dt_bias)), so normals
     * are the right domain and the monotone-cumsum rule belongs to the prefill path's `g`. */
    o.t(g.D, { T, g.Hv });                           /* a */
    o.t(g.D, { T, g.Hv });                           /* b */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* A_log */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* dt_bias */
    /* One state slot per CANDIDATE TOKEN: the initial state is read from state_idx[s,
     * num_accepted-1] and every candidate writes its own slot, because which of them survives
     * verification is not known until after this layer has run. */
    o.t((uint32_t)RAD_F32, { T, g.Hv, g.hv, g.hk }); /* state, INOUT */
    o.t_idx({ g.S, g.M }, T, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);    /* state_idx */
    o.t_idx({ g.S }, 0, 1, 0);                       /* num_accepted -- none rejected */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { T, g.Hv, g.hv });                     /* z -- the output gate */
    o.t((uint32_t)RAD_F32, { g.hv });                /* norm_w */
    o.t(g.D, { T, g.Hv, g.hv });                     /* o */
    /* The out-projection's fp8 quantiser, folded into the epilogue. It is ALL-OR-NOTHING and it is
     * skipped: with it present the op quantises the bf16 `o` it just wrote, which is a second
     * contract on top of the recurrence, and describing it here would mean every gdn decode case
     * also carried an fp8 comparison it did not ask for. The unfused pair (this op then
     * quant_act_fp8) covers the same arithmetic and is separately describable. */
    o.skip();                                        /* o_q */
    o.skip();                                        /* o_scale */
    return o.finish("gdn_recurrent_update", operand, out);
}

int ref_gdn_conv_recurrent_update_shape(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "q_len", &g)) return RAD_E_UNSUPPORTED;
    /* The decode pair's two descriptions spliced: gdn_conv_update's inputs, then
     * gdn_recurrent_update's minus q, k and v. `conv_idx` is one slot a sequence. */
    const int64_t T = g.S * g.M, sl = g.width - 2 + g.M;
    Opd o;
    o.t(g.D, { T, g.conv_dim });                     /* x */
    o.t(g.D, { g.conv_dim, g.width });               /* conv_w */
    o.skip();                                        /* conv_b */
    o.t(g.D, { g.S, g.conv_dim, sl });               /* conv_state, INOUT */
    o.t_idx({ g.S }, g.S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* conv_idx */
    o.t(g.D, { T, g.Hv });                           /* a */
    o.t(g.D, { T, g.Hv });                           /* b */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* A_log */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* dt_bias */
    o.t((uint32_t)RAD_F32, { T, g.Hv, g.hv, g.hk }); /* state, INOUT */
    o.t_idx({ g.S, g.M }, T, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);    /* state_idx */
    o.t_idx({ g.S }, 0, 1, 0);                       /* num_accepted -- none rejected */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { T, g.Hv, g.hv });                     /* z */
    o.t((uint32_t)RAD_F32, { g.hv });                /* norm_w */
    o.t(g.D, { T, g.Hv, g.hv });                     /* o */
    o.skip();                                        /* o_q -- see gdn_recurrent_update */
    o.skip();                                        /* o_scale */
    return o.finish("gdn_conv_recurrent_update", operand, out);
}

int ref_gdn_gated_rmsnorm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    /* One row per (token, head): `M` is already tokens x heads and `channels` is head_v. No head
     * count is needed, which is why this is the one member of the family that never declines. */
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ch = gi(p, n_p, "channels", 0);
    if (M <= 0 || ch <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ch });                               /* x */
    o.t(D, { M, ch });                               /* z -- the gate */
    o.t((uint32_t)RAD_F32, { ch });                  /* w */
    o.t(D, { M, ch });                               /* o */
    return o.finish("gdn_gated_rmsnorm", operand, out);
}

/* ================================================================== collectives */

int ref_all_reduce_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t numel = gi(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { numel });                               /* x, INOUT */
    /* `y` is the optional SECOND DESTINATION. Absent means in place, which is the form docs/OPS.md
     * draws and the only one a one-shot kernel needs -- but libr4d's two-shot rows are NOT safe in
     * place (a rank gathers into the output while its own input is still being read for the
     * scatter) and cannot be described at all without it, so it is passed. */
    o.t(D, { numel });                               /* y */
    return o.finish("all_reduce", operand, out);
}

int ref_all_gather_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t numel = gi(p, n_p, "numel", 0);
    const int64_t W = gi(p, n_p, "world_size", 1);
    if (numel <= 0 || W <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { numel });                               /* x */
    o.t(D, { numel * W });                           /* y -- rank r's contribution at r * numel */
    return o.finish("all_gather", operand, out);
}

/* ================================================================== vocabulary edges */

int ref_embed_lookup_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t vo = gi(p, n_p, "vocab_offset", 0);
    if (M <= 0 || ne <= 0 || nv <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    /* THE RANGE IS STATED even though the table follows: `tokens` is operand ZERO, so there is no
     * previous operand to take a leading extent from and a caller falling back to something else
     * would draw ids the launch refuses (an id at or above n_vocab fails the launch by name).
     *
     * `vocab_offset + nv` rather than `nv`: ids are GLOBAL, so under a shard the bench must draw
     * across the slice boundary or it never exercises the zero-row arm that makes the following
     * all_reduce correct. At offset 0 the bound is `nv`. */
    o.t_idx({ M }, vo + nv, -1, 0);                  /* tokens */
    o.t(D, { nv, ne });                              /* wte */
    o.t(D, { M, ne });                               /* x */
    return o.finish("embed_lookup", operand, out);
}

/* The quantised gather. The TABLE is E4M3 and the OUTPUT is the activation dtype, which is the
 * whole reason `dtype` on this op names a combination ("w8a16") rather than one width -- and it is
 * why the two operands are described with different dtypes here instead of one shared D.
 *
 * `scale` is a single element, and a harness drawing it from a normal is exactly right: the op's
 * arithmetic is a multiply and nothing about it cares what the value is. What the value IS lives in
 * the checkpoint (Qwen3.8-Flash-Next-FP8's is 1.9932e-4) and is not this hook's business. */
int ref_embed_lookup_q_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t vo = gi(p, n_p, "vocab_offset", 0);
    if (M <= 0 || ne <= 0 || nv <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t_idx({ M }, vo + nv, -1, 0);                  /* tokens */
    o.t((uint32_t)RAD_F8E4M3, { nv, ne });           /* wte */
    o.t(D, { 1 });                                   /* scale */
    o.t(D, { M, ne });                               /* x */
    o.skip();                                        /* ahead: a later call's ids, optional */
    return o.finish("embed_lookup_q", operand, out);
}

int ref_logits_gemm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    if (M <= 0 || nv <= 0 || ne <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });                               /* x */
    o.t(D, { nv, ne });                              /* lm_head */
    /* f32 logits, always: the sampler chain reads them and every stage in it is f32. */
    o.t((uint32_t)RAD_F32, { M, nv });               /* logits */
    return o.finish("logits_gemm", operand, out);
}

/* ================================================================== sampling
 *
 * Every scalar a request sets is PER ROW and arrives in the `params` operand rather than as a
 * parameter -- see sample_params_words() above for what that plane is, how it is filled, and the
 * three stages a filled-from-normals row cannot reach. The rest of the sampler's operands are
 * plain:
 *
 *   logits     f32 [M, n_vocab]     cand_idx  i32 [M, n_cand]    token   i32 [M]
 *   history    i32 [M, max_hist]    cand_val  f32 [M, n_cand]    bitmask i32 [M, ceil(nv/32)]
 *
 * and the dtypes are not a choice: libr4d's shim requires exactly these
 * (libr4d/r4d_sample_stages_f32.hip checks each one by name) and ref's requires the same of the
 * two it can tell apart. Every word-typed operand here is i32 and never u32, because
 * opshapes.cpp's rad_widen has no RAD_U32 case -- a u32 operand is compared as zeroes, and a
 * vacuous pass is worse than a skip. */

/* The history window. Not a parameter of anything -- it is the packed history plane's own extent,
 * which the deployment sizes -- so it is chosen here. Sixty-four tokens is a plausible penalty
 * window and any width walks the same loops. */
enum { REF_SAMPLE_HIST = 64 };
/* The candidate width for `sample_topk`, whose parameters carry `n_vocab` and not `n_cand`: the
 * buffer's extent IS the deployment bound and the op reads it off the operand. Narrower than the
 * vocabulary and wide enough that the narrowing stages downstream have something to narrow. */
enum { REF_SAMPLE_CAND = 64 };

static int sample_vocab(const RadParam* p, int n_p, int64_t* M, int64_t* nv) {
    *M  = gi(p, n_p, "M", 0);
    *nv = gi(p, n_p, "n_vocab", 0);
    return (*M > 0 && *nv > 0) ? RAD_OK : RAD_E_UNSUPPORTED;
}

/* sample_penalties and sample_dry share their first three operands and their description:
 * logits, params, history. DRY's fourth, the optional breaker plane, is not passed: absent means
 * no token is exempt, and the params rows drawn here reach the row's early out before a word of
 * either plane is read. */
static int penalty_shape(const RadParam* p, int n_p, int operand, const char* op,
                         bool breakers, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits, INOUT */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params */
    /* Real token ids: the penalties look each one up in the row's logits and skip anything outside
     * [0, n_vocab), so a draw from the wrong range is a draw that does nothing. */
    o.t_idx({ M, (int64_t)REF_SAMPLE_HIST }, nv, -1, 0);            /* history */
    if (breakers) o.skip();                                         /* breakers, optional */
    return o.finish(op, operand, out);
}

int ref_sample_penalties_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return penalty_shape(p, n_p, operand, "sample_penalties", false, out);
}
int ref_sample_dry_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return penalty_shape(p, n_p, operand, "sample_dry", true, out);
}

int ref_sample_temp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits, INOUT */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params */
    return o.finish("sample_temp", operand, out);
}

int ref_sample_mask_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits, INOUT */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params */
    /* A SET BIT MEANS ALLOWED, word w bit b covering token 32w + b. The plane is shorter than M in
     * a real step -- a constrained sequence and an unconstrained one share it and `mask_row` is -1
     * for the second -- and one row per sequence is the widest it is ever built. */
    o.t((uint32_t)RAD_I32, { M, (nv + 31) / 32 });                  /* bitmask */
    return o.finish("sample_mask", operand, out);
}

int ref_sample_argmax_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params -- unread by argmax */
    o.t((uint32_t)RAD_I32, { M });                                  /* token */
    return o.finish("sample_argmax", operand, out);
}

int ref_sample_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const int64_t nc = nv < (int64_t)REF_SAMPLE_CAND ? nv : (int64_t)REF_SAMPLE_CAND;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params */
    o.t((uint32_t)RAD_I32, { M, nc });                              /* cand_idx */
    o.t((uint32_t)RAD_F32, { M, nc });                              /* cand_val -- LOGITS, not probs */
    /* The vocab-parallel pair plane, and it is described because it is an OUTPUT: nothing fills it,
     * comparing it checks the packing -- (u32 global id, f32 logit) with 0xffffffff for a hole --
     * and `vocab_off` is zero on a single-rank graph, which makes the ids the local ones and the
     * comparison exactly as strict. */
    o.t((uint32_t)RAD_I32, { M, 2 * nc });                          /* pairs */
    return o.finish("sample_topk", operand, out);
}

/* The four narrowing stages share a schema -- cand_idx, cand_val, params, all in place -- and
 * therefore one description.
 *
 * THE CANDIDATE SET IS DRAWN, NOT SORTED, and that is worth naming. The stages take the set as
 * sample_topk leaves it: descending, with index -1 / -infinity for a hole. The ABI has no
 * "descending" fill domain, so what these get is a random (id, value) plane. min-p, locally-typical
 * and XTC are per-candidate predicates and do not care. Top-p is a PREFIX rule -- keep the shortest
 * run whose cumulative probability reaches p, walking the set in candidate order -- and walking a
 * random order is still the same definition applied to the same bytes, so two implementations that
 * both walk it agree. One that sorted internally would not, and that would be a real disagreement
 * with the schema rather than an artefact of this description. */
static int narrow_shape(const RadParam* p, int n_p, int operand, const char* op, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nc = gi(p, n_p, "n_cand", 0);
    if (M <= 0 || nc <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    /* Every drawn index is >= 0, so every candidate arrives LIVE: a stage that dropped nothing
     * because the set was already empty is the vacuous case. The range is the candidate width
     * rather than the vocabulary, because the ids are only ever copied and compared here and a
     * wider draw would say nothing more. */
    o.t_idx({ M, nc }, nc, -1, 0);                                  /* cand_idx, INOUT */
    o.t((uint32_t)RAD_F32, { M, nc });                              /* cand_val, INOUT */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params */
    return o.finish(op, operand, out);
}

int ref_sample_topp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_topp", out);
}
int ref_sample_minp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_minp", out);
}
int ref_sample_typical_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_typical", out);
}
int ref_sample_xtc_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_xtc", out);
}

int ref_sample_pick_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nc = gi(p, n_p, "n_cand", 0);
    if (M <= 0 || nc <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ M, nc }, nc, -1, 0);                                  /* cand_idx */
    o.t((uint32_t)RAD_F32, { M, nc });                              /* cand_val */
    o.t((uint32_t)RAD_F32, { M, sample_params_words() });           /* params -- seed and pos */
    o.t((uint32_t)RAD_I32, { M });                                  /* token */
    return o.finish("sample_pick", operand, out);
}

/* sample_merge_topk is DECLINED, and it is the one op in this vocabulary whose extents are known
 * and whose description would still be a lie.
 *
 * `gathered` is [M, world_size * n_cand] pairs of (u32 global token id, f32 logit) packed into one
 * WORD-TYPED buffer -- both plugins require i32 or u32 for it, because the two halves have
 * different types and the buffer's dtype can only be one of them. The value lane is therefore read
 * by BIT PATTERN, and there is no fill domain in this ABI that puts a sensible float there: an
 * integer draw over the buffer makes every logit a DENORMAL, and a device that flushes denormals
 * and a host reference that does not then order the candidates differently. That is not a
 * hypothetical -- it is the failure r4d_selftest's two-rank all-gather exists to catch ("a token id
 * is a small integer, and its bit pattern read as f32 is a denormal ... the model then samples
 * token 0 forever while every logit still looks right"). A comparison built on it would be
 * measuring the float mode.
 *
 * `world_size` is not in this op's parameter list either, so the gathered width would have to be
 * invented on top of that. Two reasons, either sufficient.
 *
 * row_topk_merge is the same shape of operand and is NOT declined, because its plane is f32: there
 * the value lane is a real float and the id lane is the bit pattern, which is the way round that
 * works. See ref_row_topk_merge_shape. */
int ref_sample_merge_topk_shape(const RadParam*, int, int, RadOpdDesc*) {
    return RAD_E_UNSUPPORTED;
}
