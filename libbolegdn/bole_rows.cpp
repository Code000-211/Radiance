/* bole_rows.cpp -- the host half of the Bole tree-GDN plugin: the op schema, the two rows, the
 * operand validation and the launches.
 *
 * One op, two rows, chosen by the `stage` constraint:
 *
 *   stage = "verify"  gdn_tree_bole_verify  the Bole solver: root serial, the rest node-parallel.
 *                                            Writes per-node factors and per-node outputs.
 *   stage = "commit"  gdn_tree_bole_commit  replays those factors on the ACCEPTED path and rewrites
 *                                            the state and conv history. Byte-exact to a serial
 *                                            replay, by construction.
 *
 * Both refuse what they cannot serve, and refuse it before launching: a wider batch than BOLE_NODES,
 * a head size other than 128, a conv width other than 4, a l2_eps other than 1e-6 (the engine's own
 * recurrent row compiles it in), a state cache that is not FP32, an INT8 operand without its scale,
 * and a batch whose cu, slot or shape arrays are inconsistent. A refusal returns a negative RAD_E_*;
 * neither row returns success for work it did not do.
 *
 * Only abi/ is available to an out-of-tree plugin, so RadArgs is read through the inline helpers in
 * rad_abi.h / rad_plugin.h and every libr4d helper is a restatement (bole_device.h).
 */
#include "bole_layout.h"

#include "rad_abi.h"
#include "rad_plugin.h"

#include <string.h>

/* ================================================================== the op schema */
/* Parameters are the engine's gdn_conv_recurrent_update plus the tree's own. `q_len` is the ranged
 * one (docs/OPS.md: at most one parameter per op is a range; the selector collapses it before a
 * launch). `conv_depth` is the conv cache's history depth (conv_width - 1 + the window the caller
 * keeps), which the serial row derives from its own state tensors but a tree caller must state,
 * because the history arm indexes the cache by it. `stage` selects the row. */
static const RadParamSpec kBoleParams[] = {
    { "q_len",        RAD_P_INT, RAD_REQUIRED },
    { "head_k",       RAD_P_INT, RAD_REQUIRED },
    { "head_v",       RAD_P_INT, RAD_REQUIRED },
    { "conv_width",   RAD_P_INT, RAD_REQUIRED },
    { "conv_depth",   RAD_P_INT, RAD_REQUIRED },
    { "stage",        RAD_P_STR, RAD_REQUIRED },
    { "n_head_v",     RAD_P_INT, RAD_OPTIONAL },
    { "scale",        RAD_P_F64, RAD_OPTIONAL },
    { "eps",          RAD_P_F64, RAD_OPTIONAL },
    { "l2_eps",       RAD_P_F64, RAD_OPTIONAL },
    { "softplus_thr", RAD_P_F64, RAD_OPTIONAL },
    { "act",          RAD_P_STR, RAD_OPTIONAL },
};

static const RadOperandSpec kBoleOperands[] = {
    { "x",          RAD_OPD_IN,     0 },
    { "conv_w",     RAD_OPD_WEIGHT, 0 },
    { "conv_b",     RAD_OPD_WEIGHT, 1 },
    { "conv_state", RAD_OPD_INOUT,  0 },
    { "conv_idx",   RAD_OPD_IN,     0 },
    { "a",          RAD_OPD_IN,     0 },
    { "b",          RAD_OPD_IN,     0 },
    { "A_log",      RAD_OPD_WEIGHT, 0 },
    { "dt_bias",    RAD_OPD_WEIGHT, 0 },
    { "state",      RAD_OPD_INOUT,  0 },
    { "state_idx",  RAD_OPD_IN,     0 },
    { "num_accepted", RAD_OPD_IN,   0 },
    { "cu",         RAD_OPD_IN,     0 },
    { "parent",     RAD_OPD_IN,     1 },
    { "ctx_before", RAD_OPD_IN,     0 },
    { "z",          RAD_OPD_IN,     0 },
    { "norm_w",     RAD_OPD_WEIGHT, 0 },
    { "factors",    RAD_OPD_INOUT,  0 },
    { "o",          RAD_OPD_OUT,    0 },
    { "o_q",        RAD_OPD_OUT,    1 },
    { "o_scale",    RAD_OPD_OUT,    1 },
};

static const RadOpSchema kBoleSchemas[] = {
    { "gdn_tree_bole", kBoleParams, (int)(sizeof(kBoleParams) / sizeof(kBoleParams[0])),
      kBoleOperands, (int)(sizeof(kBoleOperands) / sizeof(kBoleOperands[0])),
      "Bole tree GDN state update: `parent` names a candidate tree, so candidates that share an "
      "ancestor are evaluated against it instead of against each other. stage=verify runs the Bole "
      "solver and writes per-node factors and outputs with REORDERED accumulation below the root; "
      "stage=commit replays the accepted path onto the state cache and is byte-identical to a "
      "serial replay. Opt-in and out of tree: the engine has no tree-layout seam, so nothing "
      "selects this op unless a caller names this plugin." },
};

/* The rows' constraint bands. Every value is a property of the compiled kernel, not a preference:
 * 128 is the compiled head width, 4 the conv tap count whose shifts are hard-coded, and 1..BOLE_NODES
 * the node capacity the LDS tables are sized for. */
static const RadConstraint kBoleVerifyBand[] = {
    RAD_CEQ("head_k", 128), RAD_CEQ("head_v", 128), RAD_CEQ("conv_width", 4),
    RAD_CIN("stage", "verify"), RAD_CGE("q_len", 1), RAD_CLE("q_len", BOLE_NODES),
};
static const RadConstraint kBoleCommitBand[] = {
    RAD_CEQ("head_k", 128), RAD_CEQ("head_v", 128), RAD_CEQ("conv_width", 4),
    RAD_CIN("stage", "commit"), RAD_CGE("q_len", 1), RAD_CLE("q_len", BOLE_NODES),
};

#define BOLE_ROW(a) (a), (int)(sizeof(a) / sizeof((a)[0]))

/* The launches live beside the kernels in bole_tree.hip; the validation lives below. Declared here
 * with the ABI's RadLaunchFn shape, and filled into the row table at the bottom of this file. */
static int bole_launch_verify(const RadArgs* a, RadStream s);
static int bole_launch_commit(const RadArgs* a, RadStream s);

/* ================================================================== operand validation */
/* One place, used by both rows, so a refusal cannot drift between them. */
struct BoleArgs {
    BoleCall f;
};

static int bole_prepare(const RadArgs* a, bool commit, BoleArgs* out) {
    if (!a) return RAD_E_INVAL;
    const RadTensor* x   = rad_arg_in(a, BX);
    const RadTensor* cw  = rad_arg_in(a, BCW);
    const RadTensor* cst = rad_arg_in(a, BCST);
    const RadTensor* ci  = rad_arg_in(a, BCIDX);
    const RadTensor* st  = rad_arg_in(a, BSTATE);
    const RadTensor* si  = rad_arg_in(a, BSIDX);
    const RadTensor* cu  = rad_arg_in(a, BCU);
    const RadTensor* cx  = rad_arg_in(a, BCTX);
    const RadTensor* ft  = rad_arg_in(a, BFACT);
    /* The commit row's inputs are the factors, the state and the conv history: it needs neither the
     * gate vectors nor the output epilogue. Everything else is the verify row's. */
    const RadTensor* acc = rad_arg_in(a, BNACC);
    if (!x || !cw || !cst || !ci || !st || !si || !cu || !cx || !ft) return RAD_E_INVAL;
    /* num_accepted is required: it is the conv cache row the last accepted token left, and the
     * serial row reads the same operand for the same reason. */
    if (!acc) return RAD_E_INVAL;
    if (!commit) {
        const RadTensor* ta  = rad_arg_in(a, BA);
        const RadTensor* tb  = rad_arg_in(a, BB);
        const RadTensor* alg = rad_arg_in(a, BALOG);
        const RadTensor* dtb = rad_arg_in(a, BDTB);
        const RadTensor* z   = rad_arg_in(a, BZ);
        const RadTensor* nw  = rad_arg_in(a, BNW);
        const RadTensor* o   = rad_arg_in(a, BO);
        if (!ta || !tb || !alg || !dtb || !z || !nw || !o) return RAD_E_INVAL;
        const RadTensor* oq = rad_arg_in(a, BOQ);
        const RadTensor* os = rad_arg_in(a, BOS);
        if ((oq != nullptr) != (os != nullptr)) return RAD_E_INVAL;
        if (oq && (oq->dtype != RAD_I8 || os->dtype != RAD_F32)) return RAD_E_DTYPE;
    }

    /* Geometry, from the resolved parameters: a range arrives collapsed by selection. */
    long long q_len = 0, hk = 0, hv = 0, cwid = 0, cdepth = 0;
    if (!rad_args_geti(a, "q_len", &q_len) || !rad_args_geti(a, "head_k", &hk) ||
        !rad_args_geti(a, "head_v", &hv) || !rad_args_geti(a, "conv_width", &cwid) ||
        !rad_args_geti(a, "conv_depth", &cdepth))
        return RAD_E_SCHEMA;
    if (hk != 128 || hv != 128) return RAD_E_SHAPE;
    if (cwid != BOLE_CONV_W) return RAD_E_UNSUPPORTED;
    if (q_len < 1 || q_len > BOLE_NODES) return RAD_E_UNSUPPORTED;
    if (cdepth < 1 || cdepth > 64) return RAD_E_UNSUPPORTED;

    /* The serial row derives the conv depth from the state tensors; here it is stated, so it is
     * checked against them instead. */
    if (cst->rank != 3 || cst->shape[2] != cdepth) return RAD_E_SHAPE;
    /* The window must fit the cache: `mql = shape[2] - (conv_width - 2)` is libr4d's own bound
     * (r4d_gdn_recurrent_update..., before ru_conv_recurrent_launch), and it is a hard bound here
     * too, not a preference. A node at depth d reads history entries up to
     * off + d - 1 + (conv_width - 1), so a window deeper than shape[2] - (conv_width - 2) walks past
     * the cache's extent -- a device memory fault, not a wrong number. */
    if (q_len > cdepth - (BOLE_CONV_W - 2)) return RAD_E_UNSUPPORTED;
    if (cu->rank != 1 || cu->shape[0] < 2) return RAD_E_SHAPE;
    const int nseq = (int)(cu->shape[0] - 1);
    if (si->shape[0] != nseq || ci->shape[0] != nseq || cx->shape[0] != nseq ||
        acc->shape[0] != nseq)
        return RAD_E_SHAPE;

    /* The factor tensor must hold one node's 258 floats per (token, head) of the widest sequence. */
    if (ft->rank < 2) return RAD_E_SHAPE;

    const float l2_eps = (float)rad_args_getf_or(a, "l2_eps", 1e-6);
    /* The l2 epsilon is compiled into the exchange, exactly as libr4d's recurrent row refuses any
     * other value rather than serving a geometry it would compute differently. */
    if (l2_eps != 1e-6f) return RAD_E_UNSUPPORTED;
    /* State width: this row serves the FP32 cache only. libr4d's row also serves FP16 and picks the
     * store type from the dtype; an FP16 cache is refused here rather than silently truncated. */
    if (st->dtype != RAD_F32) return RAD_E_DTYPE;

    BoleCall& f = out->f;
    memset(&f, 0, sizeof f);
    f.nseq = nseq;
    f.tokens = (int)x->shape[0];
    f.hk = (int)hk;
    f.hv = (int)hv;
    f.q_len = (int)q_len;
    f.conv_w = (int)cwid;
    f.scale = (float)rad_args_getf_or(a, "scale", 1.0);
    f.eps = (float)rad_args_getf_or(a, "eps", 0.0);
    f.l2_eps = l2_eps;
    f.softplus_thr = (float)rad_args_getf_or(a, "softplus_thr", 20.0);
    /* libr4d's ru_act: kind 0 is silu (z*sigmoid(z)), anything else sigmoid(z). */
    const char* act = rad_args_gets_or(a, "act", "silu");
    f.sigmoid = strcmp(act, "silu") != 0 ? 1 : 0;

    /* In the schema's order, index for index. A transposition here is invisible at compile time --
     * every entry is a RadTensor* -- and shows up as a valid call being refused on a dtype or a
     * shape that belongs to its neighbour. */
    const RadTensor* ops[BOLE_N_OPS] = {
        x, cw, rad_arg_in(a, BCB), cst, ci, rad_arg_in(a, BA), rad_arg_in(a, BB),
        rad_arg_in(a, BALOG), rad_arg_in(a, BDTB), st, si, acc, cu, rad_arg_in(a, BPARENT), cx,
        rad_arg_in(a, BZ), rad_arg_in(a, BNW), ft, rad_arg_in(a, BO), rad_arg_in(a, BOQ),
        rad_arg_in(a, BOS),
    };
    for (int i = 0; i < BOLE_N_OPS; ++i) {
        if (!ops[i]) continue;
        if (ops[i]->rank > 4) return RAD_E_SHAPE;
        f.t[i] = ops[i]->data;
        for (uint32_t d = 0; d < ops[i]->rank; ++d) f.s[i][d] = ops[i]->stride[d];
    }
    return RAD_OK;
}

static int bole_launch_verify(const RadArgs* a, RadStream s) {
    BoleArgs b;
    const int rc = bole_prepare(a, false, &b);
    return rc != RAD_OK ? rc : bole_verify_launch(&b.f, s);
}

static int bole_launch_commit(const RadArgs* a, RadStream s) {
    BoleArgs b;
    const int rc = bole_prepare(a, true, &b);
    return rc != RAD_OK ? rc : bole_commit_launch(&b.f, s);
}

/* The two rows. Built here rather than at the top so the names of the launches are already in
 * scope; `nullptr` converts to any function-pointer type, so a field inserted in the wrong place
 * would still compile (abi/rad_abi.h, "Keep this field last"). */
#define BOLE_VERIFY_ROW                                                                \
    { "gdn_tree_bole_verify_k128_v128", "gdn_tree_bole", "gdn",                        \
      "Bole tree GDN verify: the root serial in FP32 as the engine computes it, every " \
      "other node node-parallel in FP64 over its non-root ancestors, rounded-u "        \
      "feedback in node order, then the engine's gated norm and INT8 epilogue per node " \
      "[REORDERED accumulation below the root]",                                        \
      "head_k/head_v 128, conv width 4, 1..16 nodes a sequence, one workgroup per "      \
      "(sequence, head)",                                                                \
      "bf16 x / conv weights / conv state / a,b / z / out; f32 state; int8 o_q with f32 o_scale", \
      RAD_DOMAIN_DEVICE, 50, BOLE_ROW(kBoleVerifyBand), nullptr, 0, nullptr,             \
      nullptr, nullptr, bole_launch_verify, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }

#define BOLE_COMMIT_ROW                                                                \
    { "gdn_tree_bole_commit_k128_v128", "gdn_tree_bole", "gdn",                        \
      "Bole tree GDN commit: replays the accepted path's factors and recomputes u "    \
      "from the original update data in the serial FP32 order, so the committed state " \
      "is byte-identical to a serial replay; also shifts the conv history",            \
      "head_k/head_v 128, conv width 4, 1..16 nodes a sequence, one workgroup per "      \
      "(sequence, head)",                                                                \
      "f32 factors and state, bf16 conv state, both inout; no output row",               \
      RAD_DOMAIN_DEVICE, 50, BOLE_ROW(kBoleCommitBand), nullptr, 0, nullptr,             \
      nullptr, nullptr, bole_launch_commit, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }

static const RadKernelInfo kBoleRows[] = { BOLE_VERIFY_ROW, BOLE_COMMIT_ROW };

/* ================================================================== the plugin exports */
static const RadPluginInfo kBoleInfo = {
    RAD_PLUGIN_KERNEL, "libbolegdn", "0.1",
    "Bole tree-GDN state update (arXiv:2608.01651v1), ported from the ABI-16 vb-exact-commit "
    "variant. Opt-in: nothing selects this op unless a caller names this plugin.",
    "gfx1201",
};

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &kBoleInfo; }
int rad_kernel_schema_count(void) { return (int)(sizeof(kBoleSchemas) / sizeof(kBoleSchemas[0])); }
const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i >= 0 && i < rad_kernel_schema_count()) ? &kBoleSchemas[i] : nullptr;
}
int rad_kernel_count(void) { return 2; }
const RadKernelInfo* rad_kernel_at(int i) { return (i >= 0 && i < 2) ? &kBoleRows[i] : nullptr; }

}  /* extern "C" */
