/* avx_registry.cpp -- what libavx contains, in the two forms the core reads it in.
 *
 * ============================== THE SCHEMAS ARE libref's, DELIBERATELY IDENTICAL ==============================
 *
 * The first plugin in hierarchy order to declare an op FIXES its schema, and a later plugin
 * declaring the same op name with a different one is refused AT LOAD with both plugins named
 * (rad_abi.h). The check is core/plugin/loader.cpp:schema_same, and what it holds is: the
 * parameter list position for position -- key, type, requiredness -- and the operand list position
 * for position -- role and optionality. Operand NAMES are explicitly not load-bearing.
 *
 * So this table is libref's, slot for slot. It is COPIED rather than shared, and that is the ABI
 * boundary working as designed rather than a missed refactor: a plugin links abi/ and nothing
 * else, so there is no header for libref's table to live in that libavx could include without
 * making one plugin depend on another's source. What replaces the shared header is a TEST:
 * rad-avx-check opens both plugins and runs the loader's own comparison over all seventy-five
 * schemas before it checks a single number, and fails loudly on the first disagreement. A copy
 * that is checked on every run is a copy that cannot drift; a copy that is merely intended to
 * match is a copy that eventually does.
 *
 * ============================== WHY THE ROWS CARRY NO opd_shape ==============================
 *
 * `RadKernelInfo::opd_shape` answers "how big is operand i of this op at this geometry" for a tool
 * calling the op cold, and libref implements one for every op in the conventional vocabulary --
 * 88 KB of judgement calls about index ranges, fill domains and what a ragged batch means. libavx
 * computes exactly those ops at exactly those shapes, so libref's answers are libavx's answers,
 * and the ABI says so in as many words: "libref supplies it for the whole conventional vocabulary,
 * so an op that stays inside docs/OPS.md is describable even when the kernel under test is not."
 *
 * Writing a second copy would be writing a second set of judgement calls about the SAME ops, and
 * the two would disagree somewhere -- which is worse than having one, because a tool would then
 * size a buffer from one plugin and check it against the other. The cost is stated plainly: a
 * tool that has libavx and NOT libref cannot size libavx's buffers. That tool does not exist (both
 * are host plugins and ship together) and if it ever does, the fix is to implement the hook, not
 * to have guessed.
 *
 * ============================== AND NO TUNABLES ==============================
 *
 * ABI v8 has kernels declare tunable AXES and the core take the cross product (rad_abi.h). libavx
 * declares none, and the reason is that its axes are not free: the vector width is fixed by the
 * ISA level, which is fixed by the CPU, and the GEMM's blocking is chosen from the cache sizes
 * CPUID reports rather than from a search. There is one thing worth tuning -- the L2 block edge --
 * and it is read from the machine rather than measured per shape, because unlike a GPU tile it
 * does not interact with an occupancy limit. A declared axis whose best value is computable is a
 * declared axis that wastes the tuner's time.
 */
#include "avx_isa.h"
#include "avx_rows.h"

#include "rad_abi.h"
#include "rad_plugin.h"

/* ================================================================== parameter specs
 * The macro names are libref's too. Two files that must agree line for line should look the same
 * when they are diffed, and `diff libref/ref_registry.cpp libavx/avx_registry.cpp` over the schema
 * section is a real review step. */
#define P_INT(k)   { k, RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { k, RAD_P_INT, RAD_OPTIONAL }
#define P_STR(k)   { k, RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { k, RAD_P_STR, RAD_OPTIONAL }
#define P_INTD(k)  { k, RAD_P_INT, RAD_DERIVED }
#define P_CHUNK(k) { k, RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }
#define P_F64(k)   { k, RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { k, RAD_P_F64, RAD_OPTIONAL }

#define OPD(n)     { n, RAD_OPD_IN,     0 }
#define OPD_O(n)   { n, RAD_OPD_IN,     1 }
#define OUT(n)     { n, RAD_OPD_OUT,    0 }
#define OUT_O(n)   { n, RAD_OPD_OUT,    1 }
#define INOUT(n)   { n, RAD_OPD_INOUT,  0 }
#define INOUT_O(n) { n, RAD_OPD_INOUT,  1 }
#define WGT(n)     { n, RAD_OPD_WEIGHT, 0 }
#define WTAB(n)    { n, RAD_OPD_WTAB,   0 }
#define WGT_O(n)   { n, RAD_OPD_WEIGHT, 1 }

#define ARR(a)  (a), (int)(sizeof(a) / sizeof((a)[0]))

/* ---- elementwise and norms ---------------------------------------------------------------- */
static const RadParamSpec pRmsnorm[]   = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                           P_F64O("wadd") };
static const RadOperandSpec oRmsnorm[] = { OPD("x"), WGT("w"), OUT("y") };

static const RadParamSpec pHcRead[]    = { P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"),
                                           P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
                                           P_INTO("inject"), P_INTO("group"), P_INTO("rotate"),
                                           P_STRO("mix") };
static const RadOperandSpec oHcRead[]  = { OPD("h"), WGT("w"), WGT("mix_down"), WGT("mix_up"),
                                           WGT_O("inject_w"), OUT("x"), OUT_O("inj"), OUT_O("q"),
                                           OUT_O("scale"), OUT_O("rq"), OUT_O("rscale") };

static const RadParamSpec pNgramIds[]   = { P_INT("M"), P_INT("heads"), P_INT("ngram"),
                                            P_INT("eos") };
static const RadOperandSpec oNgramIds[] = { OPD("tok"), INOUT("state"), OPD("cu"), WGT("mult"),
                                            WGT("vocab_sizes"), WGT("offsets"),
                                            OPD_O("cache_idx"), OPD_O("has_init"),
                                            OPD_O("num_accepted"), OUT("ids"),
                                           OUT_O("ahead_ids") };

static const RadParamSpec pPleGate[]   = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                           P_STR("dtype"), P_F64O("wadd"), P_F64O("gate_eps") };
static const RadOperandSpec oPleGate[] = { OPD("k"), OPD("q"), OPD("v"), WGT("w_key"),
                                           WGT("w_query"), WGT("w_conv"), OUT("gv"), OUT("gvn") };

static const RadParamSpec pPleConv[]   = { P_INT("M"), P_INT("n"), P_INT("width"),
                                           P_INT("dilation"), P_STR("dtype") };
static const RadOperandSpec oPleConv[] = { OPD("x"), WGT("w"), INOUT("conv_state"), OPD("cu"),
                                           OPD_O("resid"), OPD_O("cache_idx"), OPD_O("has_init"),
                                           OPD_O("num_accepted"), OUT("y") };

static const RadParamSpec pHcWrite[]   = { P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype") };
static const RadOperandSpec oHcEnter[] = { OPD("x"), OUT("h") };
static const RadOperandSpec oHcWrite[] = { OPD("y"), OPD("inj"), INOUT("h") };
/* libref's schema, slot for slot: a capital `N` because this is a new op and its caller spells the
 * width that way; the gated residual's ops keep their `n`. See libref/ref_registry.cpp. */
static const RadParamSpec pHcMean[]   = { P_INT("M"), P_INT("N"), P_INT("hc"), P_STR("dtype") };
static const RadOperandSpec oHcMean[] = { OPD("h"), OUT("x") };

static const RadParamSpec pMtpEnter[]   = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                            P_STR("dtype"), P_F64O("wadd"), P_INTO("ngroup"),
                                            P_STRO("fc") };
static const RadOperandSpec oMtpEnter[] = { OPD("h"), OPD("e"), WGT("w_h"), WGT("w_e"),
                                            WGT("fc_h"), WGT("fc_e"), OUT("x") };

static const RadOperandSpec oRmsnormAdd[] = { OPD("x"), OPD("residual"), WGT("w"),
                                              OUT("y"), OUT("residual_out") };

static const RadParamSpec pLayernorm[]   = { P_INT("M"), P_INT("n"), P_F64("eps"),
                                             P_STR("dtype") };
static const RadOperandSpec oLayernorm[] = { OPD("x"), WGT("w"), WGT_O("b"), OUT("y") };
static const RadParamSpec pGridEmbed[]   = { P_INT("M"), P_INT("n"), P_INT("side"), P_STR("dtype") };
static const RadOperandSpec oGridEmbed[] = { INOUT("x"), WGT("table"), OPD("coord") };

static const RadParamSpec pMnD[]      = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oBinary[] = { OPD("a"), OPD("b"), OUT("y") };
static const RadOperandSpec oGateUp[] = { OPD("gate_up"), OUT("y") };
static const RadOperandSpec oUnary[]  = { OPD("x"), OUT("y") };
static const RadOperandSpec oGather[] = { OPD("x"), OPD("idx"), OUT("y") };
static const RadOperandSpec oScatterRows[] = { OPD("v"), OPD("idx"), INOUT("x") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

/* ---- quantisation and rotation ------------------------------------------------------------ */
static const RadParamSpec pQuant[]      = { P_INT("M"), P_INT("n"), P_INT("group"),
                                            P_STR("dtype") };
static const RadOperandSpec oQuant[]    = { OPD("x"), OUT("q"), OUT("scale"), OUT_O("asum") };
static const RadOperandSpec oHadQuant[] = { OPD("x"), OUT("q"), OUT("scale") };
static const RadOperandSpec oQuantFp8[] = { OPD("x"), OUT("q"), OUT("scale") };

/* The fp8 fusions: the unfused pair's operands minus the tensor between them, and `out_bf16` to
 * put even that back. */
static const RadParamSpec pRmsQuantFp8[]     = { P_INT("M"), P_INT("n"), P_F64("eps"),
                                                 P_INT("group"), P_STR("dtype"), P_F64O("wadd") };
static const RadOperandSpec oRmsQuantFp8[]   = { OPD("x"), INOUT_O("residual"), WGT("w"),
                                                 OUT("q"), OUT("scale"), OUT_O("out_bf16") };
static const RadParamSpec pGatedQuantFp8[]   = { P_INT("M"), P_INT("n"), P_INT("group"),
                                                 P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oGatedQuantFp8[] = { OPD("gate_up"), OUT("q"), OUT("scale"),
                                                 OUT_O("out_bf16") };

static const RadParamSpec pRmsHadQuant[]   = { P_INT("M"), P_INT("n"), P_F64("eps"),
                                               P_INT("group"), P_STR("dtype"), P_F64O("wadd") };
static const RadOperandSpec oRmsHadQuant[] = { OPD("x"), INOUT_O("residual"), WGT("w"),
                                               OUT("q"), OUT("scale"), OUT_O("out_bf16") };

static const RadParamSpec pGatedQuant[]   = { P_INT("M"), P_INT("n"), P_INT("group"),
                                              P_STR("dtype"), P_STRO("act"), P_INTO("mode") };
static const RadOperandSpec oGatedQuant[] = { OPD("a"), OPD_O("b"), OUT("q"), OUT("scale") };

/* `a_scale` is optional: dtype "bf16" is a plain activation and carries none. */
static const RadOperandSpec oGram[] = { OPD("a"), OPD_O("a_scale"), INOUT("h") };

static const RadParamSpec pDequant[]   = { P_INT("M"), P_INT("n"), P_INT("group"),
                                           P_STR("dtype") };
static const RadOperandSpec oDequant[] = { OPD("q"), OPD("scale"), OUT("y") };

/* ---- gemm --------------------------------------------------------------------------------- */
static const RadParamSpec pGemm[]       = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype") };
static const RadOperandSpec oGemm[]     = { OPD("a"), WGT("b"), OUT("y") };
static const RadParamSpec pGemmBias[]   = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
                                           P_STRO("act") };
static const RadOperandSpec oGemmBias[] = { OPD("a"), WGT("b"), WGT("bias"), OPD_O("res"),
                                            OUT("y") };

static const RadParamSpec pGemmQ[]        = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                              P_STR("dtype") };
static const RadOperandSpec oGemmQ[]      = { OPD("a"), OPD_O("a_scale"), WGT("b"), WGT("b_scale"),
                                              OUT("y"), OPD_O("a_sum"), WGT_O("b_ref") };
static const RadOperandSpec oGemmQBias[]  = { OPD("a"), OPD("a_scale"), WGT("b"), WGT("b_scale"),
                                              WGT("bias"), OUT("y") };
/* `act` REQUIRED, where the quantiser has it optional; `N` is the weight's rows, twice the output's
 * width. */
static const RadParamSpec pGemmQGated[]    = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                               P_STR("dtype"), P_STR("act") };
static const RadOperandSpec oGemmQGated[]  = { OPD("a"), OPD("a_scale"), WGT("b"), WGT("b_scale"),
                                               OUT("q"), OUT("scale"), OUT_O("out_bf16") };

/* ---- attention ---------------------------------------------------------------------------- */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"), P_INTD("block_size"), P_INT("causal"),
    P_INT("window"), P_STR("q_dtype"), P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
    P_INTO("n_head"), P_INTO("max_seqs")
};
static const RadOperandSpec oAttnPaged[] = {
    OPD("q"), OPD("kv_cache"), OPD("block_table"), OPD("seqused"),
    OPD_O("k_descale"), OPD_O("v_descale"), OPD_O("cu_seqlens"), OUT("out"),
    OPD_O("tree_ancestors")
};
static const RadConstraint cPagedBlock[] = { RAD_CGE("block_size", 1) };

static const RadParamSpec pAttnDense[]   = { P_INT("M"), P_INT("head_dim"), P_INT("gqa"),
                                             P_INT("causal"), P_STR("dtype"), P_F64O("scale"),
                                             P_INTO("max_seqlen") };
static const RadOperandSpec oAttnDense[] = { OPD("q"), OPD("k"), OPD("v"), OPD("cu_seqlens"),
                                             OUT("out") };

static const RadParamSpec pKvStore[]   = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                           P_INT("block_size"), P_STR("kv_dtype"),
                                           P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { OPD("k"), OPD("v"), OPD("slot_mapping"),
                                           INOUT("kv_cache") };

static const RadParamSpec pRope[]   = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                        P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                        P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections") };
static const RadOperandSpec oRope[] = { INOUT("qkv"), OPD("positions") };

static const RadParamSpec pRopeTable[]   = { P_INT("M"), P_INT("rot"), P_F64("theta"),
                                             P_F64("scale"), P_STR("dtype") };
static const RadOperandSpec oRopeTable[] = { OPD("positions"), OUT("cos_sin") };

static const RadParamSpec pQkNormRope[]   = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                              P_INT("n_head_kv"), P_F64("theta"), P_F64("eps"),
                                              P_F64O("scale"), P_STRO("mode"),
                                              P_INTO("rotary_dim"), P_F64O("wadd") };
static const RadOperandSpec oQkNormRope[] = { INOUT("qkv"), OPD("positions"), WGT("q_w"),
                                              WGT("k_w") };

/* ---- mixture of experts ------------------------------------------------------------------- */
static const RadParamSpec pScaleRows[]   = { P_INT("M"), P_INT("n"), P_STR("act"),
                                             P_STR("dtype") };
/* `add` is OPTIONAL and sits BEFORE `y`, which is the kind of detail a hand-kept copy gets wrong:
 * operands are positional, so reading `y` at index 2 writes the output into the residual the
 * caller passed. rad-avx-check compares this table against libref's on every run, which is what
 * catches exactly this drift. */
static const RadOperandSpec oScaleRows[] = { OPD("x"), OPD("s"), OPD_O("add"), OUT("y") };

static const RadParamSpec pRouter[]   = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                          P_INT("norm"), P_STR("dtype") };
static const RadOperandSpec oRouter[] = { OPD("logits"), OUT("expert_ids"), OUT("expert_w") };

static const RadParamSpec pScatter[]   = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                           P_INTO("expert_base"),
                                           P_STRO("protect") };
static const RadOperandSpec oScatter[] = { OPD("expert_ids"), OUT("sorted_tok"),
                                           OUT("expert_offset"), OUT("expert_count") };

static const RadParamSpec pMoeGemm[]   = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                           P_INT("top_k"), P_STR("a_order"), P_STR("dtype") };
static const RadOperandSpec oMoeGemm[] = { OPD("a"), WTAB("w"), OPD("sorted_tok"),
                                           OPD("expert_offset"), OUT("y") };

/* The experts are two tables by parity -- `w` the even ones at [N, K], `w_odd` the odd ones at
 * [N_odd, K_odd] -- with `parts` stacked parts along the rows and the output; libr4d's schema
 * says why. */
static const RadParamSpec pMoeGemmQ[]   = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                            P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                            P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"),
                                            P_INTO("parts") };
static const RadOperandSpec oMoeGemmQ[] = { OPD("a"), OPD_O("a_scale"), WTAB("w"), WTAB("w_scale"),
                                            OPD("sorted_tok"), OPD("expert_offset"),
                                            WTAB("w_odd"), WTAB("w_odd_scale"), OUT("y") };
/* The same over four classes by the expert's number mod 4, `w_q` class q at [N_q, K_q]. Mirrors
 * libref's and libr4d's schema exactly. */
static const RadParamSpec pMoeGemmQMod4[]   = { P_INT("M"), P_INT("N"), P_INT("K"),
                                                P_INT("n_expert"), P_INT("top_k"), P_INT("group"),
                                                P_STR("a_order"), P_STR("dtype"), P_INTO("N_1"),
                                                P_INTO("K_1"), P_INTO("N_2"), P_INTO("K_2"),
                                                P_INTO("N_3"), P_INTO("K_3"), P_INTO("parts") };
static const RadOperandSpec oMoeGemmQMod4[] = { OPD("a"), OPD_O("a_scale"), WTAB("w"),
                                                WTAB("w_scale"), OPD("sorted_tok"),
                                                OPD("expert_offset"), WTAB("w_1"),
                                                WTAB("w_1_scale"), WTAB("w_2"), WTAB("w_2_scale"),
                                                WTAB("w_3"), WTAB("w_3_scale"), OUT("y") };

/* `shared` + `shared_gate` present = the shared arm folded in: y = narrow(shared *
 * act(gate)) + narrow(gathered), the unfused pair's two narrows carried in the stores. `act` is
 * scale_rows' spelling, sigmoid or none, and applies to the SCALAR. Mirrors libref's schema
 * exactly -- the loader refuses the whole plugin on any disagreement. */
static const RadParamSpec pMoeGather[]   = { P_INT("M"), P_INT("n"), P_INT("top_k"),
                                             P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oMoeGather[] = { OPD("y_expert"), OPD("expert_w"), OPD("sorted_tok"),
                                             OPD_O("shared"), OPD_O("shared_gate"), OUT("y") };

static const RadParamSpec pRowTopk[]   = { P_INT("M"), P_INT("N"), P_INT("R"), P_INTO("vocab_off"),
                                           P_STR("dtype") };
static const RadOperandSpec oRowTopk[] = { OPD("x"), OUT("idx"), OUT("val"), OUT_O("pairs") };

static const RadParamSpec pRowTopkMerge[]   = { P_INT("M"), P_INT("R"), P_INT("world_size"),
                                                P_STR("dtype") };
static const RadOperandSpec oRowTopkMerge[] = { OPD("gathered"), OUT("idx"), OUT("val") };

/* Mirrors libref's row exactly: R candidates rescored against the full-precision head, ties to the
 * lower id, pads sort last. libref's shape hook declines to describe this op, so the checker skips
 * its geometries -- but the row must exist or the whole-plugin coverage gate fails. */
static const RadParamSpec pLogitRerank[] = { P_INT("M"), P_INT("R"), P_INT("n_vocab"),
                                             P_INT("n_embd"), P_INTO("vocab_off"),
                                             P_STR("dtype") };
static const RadOperandSpec oLogitRerank[] = { OPD("x"), WGT("head"), OPD("idx_in"),
                                               OUT("idx"), OUT("val"), OUT_O("pairs") };

/* ---- gated delta net ---------------------------------------------------------------------- */
static const RadParamSpec pConvPrep[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                          P_CHUNK("chunk"), P_INT("conv_width"),
                                          P_F64O("l2_eps"), P_F64O("softplus_thr") };
static const RadOperandSpec oConvPrep[] = {
    OPD("x"), WGT("w"), WGT_O("b"), WGT("A_log"), WGT("dt_bias"), INOUT("conv_state"), OPD("cu"),
    OPD_O("a"), OPD_O("b_gate"), OPD_O("cache_idx"), OPD_O("has_init"),
    OUT("q"), OUT("k"), OUT("v"), OUT("g"), OUT("beta")
};

static const RadParamSpec pConvUpdate[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                            P_INT("conv_width"), P_INTO("max_query_len") };
static const RadOperandSpec oConvUpdate[] = {
    OPD("x"), WGT("w"), WGT_O("b"), INOUT("conv_state"), OPD("state_idx"), OPD("num_accepted"),
    OPD("cu"), OUT("q"), OUT("k"), OUT("v")
};

static const RadParamSpec pKkt[]   = { P_INT("M"), P_INT("head_k"), P_CHUNK("chunk") };
static const RadOperandSpec oKkt[] = { OPD("k"), OPD("beta"), OPD("g"), OPD("cu"), OUT("A") };

static const RadParamSpec pChunkScan[]   = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                             P_CHUNK("chunk"), P_F64O("scale"),
                                             P_INTO("state_fp16") };
static const RadOperandSpec oChunkScan[] = { OPD("q"), OPD("k"), OPD("v"), OPD("A"), OPD("g"),
                                             OPD("beta"), OPD_O("h0"), OPD("cu"),
                                             OUT("o"), OUT_O("ht"), OPD_O("state_idx") };

static const RadParamSpec pRecurrent[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                           P_INTO("n_head_v"), P_F64O("scale"), P_F64O("eps"),
                                           P_F64O("l2_eps"), P_F64O("softplus_thr"),
                                           P_STRO("act"), P_INTO("state_fp16"),
                                           P_STRO("state_form") };
static const RadOperandSpec oRecurrent[] = {
    OPD("q"), OPD("k"), OPD("v"), OPD("a"), OPD("b"), WGT("A_log"), WGT("dt_bias"),
    INOUT("state"), OPD("state_idx"), OPD("num_accepted"), OPD("cu"), OPD_O("z"),
    WGT_O("norm_w"), OUT("o"), OUT_O("o_q"), OUT_O("o_scale")
};

/* gdn_conv_update folded into gdn_recurrent_update: the convolution's operands, then the
 * recurrence's minus q, k and v; the parameters are the union of the two ops' under their own
 * names. */
static const RadParamSpec pConvRecurrent[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                               P_INT("conv_width"), P_INTO("n_head_v"),
                                               P_F64O("scale"), P_F64O("eps"), P_F64O("l2_eps"),
                                               P_F64O("softplus_thr"), P_STRO("act"),
                                               P_INTO("state_fp16"), P_STRO("state_form") };
static const RadOperandSpec oConvRecurrent[] = {
    OPD("x"), WGT("conv_w"), WGT_O("conv_b"), INOUT("conv_state"), OPD("conv_idx"), OPD("a"),
    OPD("b"), WGT("A_log"), WGT("dt_bias"), INOUT("state"), OPD("state_idx"),
    OPD("num_accepted"), OPD("cu"), OPD("z"), WGT("norm_w"), OUT("o"), OUT_O("o_q"),
    OUT_O("o_scale")
};

static const RadParamSpec pGatedNorm[]   = { P_INT("M"), P_INT("channels"), P_F64("eps"),
                                             P_STR("act") };
static const RadOperandSpec oGatedNorm[] = { OPD("x"), OPD("z"), WGT("w"), OUT("o") };

/* ---- collectives -------------------------------------------------------------------------- */
static const RadParamSpec pAllReduce[]   = { P_INT("world_size"), P_INT("numel"), P_STR("dtype"),
                                             P_INT("exact"), P_INTO("min_bytes"), P_INTO("hops") };
static const RadOperandSpec oAllReduce[] = { INOUT("x"), OUT_O("y") };

static const RadParamSpec pAllGather[]   = { P_INT("world_size"), P_INT("numel"), P_STR("dtype"),
                                             P_INTO("row") };
static const RadOperandSpec oAllGather[] = { OPD("x"), OUT("y") };

/* ---- vocabulary edges --------------------------------------------------------------------- */
static const RadParamSpec pEmbed[]    = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                          P_STR("dtype"), P_INTO("vocab_offset"),
                                          P_F64O("wscale") };
static const RadOperandSpec oEmbed[]  = { OPD("tokens"), WGT("wte"), OUT("x") };
static const RadOperandSpec oEmbedQ[] = { OPD("tokens"), WGT("wte"), WGT_O("scale"), OUT("x"),
                                          OPD_O("ahead") };

static const RadParamSpec pLogits[]   = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                          P_STR("dtype") };
static const RadOperandSpec oLogits[] = { OPD("x"), WGT("lm_head"), OUT("logits") };

/* ---- sampling ----------------------------------------------------------------------------- */
static const RadParamSpec pMVocab[] = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]  = { P_INT("M"), P_INT("n_cand") };
/* libref's schema: the history stages take the rank's first vocabulary row, and DRY an optional
 * plane of breaker marks beside the history. */
static const RadParamSpec pPenal[]  = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };

static const RadOperandSpec oPenalties[] = { INOUT("logits"), OPD("params"), OPD("history") };
static const RadOperandSpec oDry[]       = { INOUT("logits"), OPD("params"), OPD("history"),
                                             OPD_O("breakers") };
static const RadOperandSpec oTemp[]      = { INOUT("logits"), OPD("params") };
static const RadOperandSpec oMask[]      = { INOUT("logits"), OPD("params"), OPD("bitmask") };
static const RadParamSpec pTopk[]        = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadOperandSpec oTopk[]      = { OPD("logits"), OPD("params"),
                                             OUT("cand_idx"), OUT("cand_val"), OUT_O("pairs") };
static const RadOperandSpec oArgmax[]    = { OPD("logits"), OPD("params"), OUT("token") };
static const RadOperandSpec oNarrow[]    = { INOUT("cand_idx"), INOUT("cand_val"), OPD("params") };
static const RadOperandSpec oPick[]      = { OPD("cand_idx"), OPD("cand_val"), OPD("params"),
                                             OUT("token") };
static const RadOperandSpec oMerge[]     = { OPD("gathered"), OPD("params"),
                                             OUT("cand_idx"), OUT("cand_val") };

/* The drafter's token walk over a candidate lattice. */
static const RadParamSpec pDflashSelect[]    = { P_INT("M"), P_INT("steps"), P_INT("top_k"),
                                                 P_INT("rank"), P_INT("n_vocab"),
                                                 P_INTO("anchor_stride"), P_STR("dtype") };
static const RadOperandSpec oDflashSelect[]  = { OPD("cand"), OPD("unary"), OPD_O("hp"),
                                                 OPD("anchor"), WGT("pred"), WGT("succ"),
                                                 OUT("tokens") };

/* ================================================================== the schema table
 * The doc strings are libavx's own and say what this plugin does with the op -- which of the two
 * plugins' strings a reader sees depends on which one fixed the schema first, and schema_same does
 * not compare them. Where the op is subtle the string points at libref's, which is the definition.
 */
static const RadOpSchema kSchemas[] = {
{ "rmsnorm", ARR(pRmsnorm), ARR(oRmsnorm),
  "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). The sum of squares is a four-accumulator vector "
  "reduction, so it does not sum in libref's order and does not claim to." },
{ "rmsnorm_add", ARR(pRmsnorm), ARR(oRmsnormAdd),
  "residual_out = x + residual, then y = rmsnorm(residual_out) * w. The variance is taken from the "
  "unrounded f32 sum and the multiply reads residual_out back AFTER narrowing -- both are libref's "
  "rule and both are observable, so both are reproduced exactly." },
{ "hc_enter", ARR(pHcWrite), ARR(oHcEnter),
  "h[s] = x for each of the hc streams -- the wide residual stream's first value." },
{ "hc_read", ARR(pHcRead), ARR(oHcRead),
  "The read half of a gated residual: grouped norm, low-rank gate, stream mix, and optionally the "
  "block-scaled fp8 form of x and its Hadamard-rotated twin. See docs/OPS.md for the three facts "
  "a second implementation has to match (the mix is a MEAN, there is a separate / hc inside each "
  "gate argument, and hc_write adds into the UNNORMED h). `mix` names the matrices' stored "
  "format; a host kernel reads them as the dense tensors it is handed." },
{ "hc_write", ARR(pHcWrite), ARR(oHcWrite), "h[s] += inj[s] * y, into the unnormed stream." },
{ "hc_mean", ARR(pHcMean), ARR(oHcMean),
  "x[m,i] = mean over the hc streams of h[m, c*N+i], summed in f32 and stored bf16 once -- the "
  "capture's own reduction, with no RMS, gate or weight. The output is normally a column view of "
  "a wider tap plane and is written through its own row stride." },
{ "mtp_enter", ARR(pMtpEnter), ARR(oMtpEnter),
  "MTP head entry: two norms, a per-sub-stream fc and a broadcast fc. `fc` names the two "
  "matrices' stored format; a host kernel reads them as the dense tensors it is handed." },
{ "layernorm", ARR(pLayernorm), ARR(oLayernorm), "layer norm with gain and optional bias." },
{ "grid_embed", ARR(pGridEmbed), ARR(oGridEmbed),
  "x += a side x side position table resampled bilinearly to the row's grid." },
{ "add", ARR(pMnD), ARR(oBinary), "elementwise sum." },
{ "mul", ARR(pMnD), ARR(oBinary),
  "elementwise product; `b` may have extent 1 on any axis `a` does not and is broadcast along it." },
{ "silu_mul", ARR(pMnD), ARR(oGateUp),
  "silu-gated product over a fused [M, 2n] gate_up: y = silu(gate) * up." },
{ "gelu", ARR(pMnD), ARR(oUnary), "exact erf gelu." },
{ "silu", ARR(pMnD), ARR(oUnary), "silu." },
{ "sigmoid", ARR(pMnD), ARR(oUnary), "logistic sigmoid." },
{ "ple_gate", ARR(pPleGate), ARR(oPleGate),
  "PLE gate: three grouped norms, a dot, a signed sqrt." },
{ "ple_conv", ARR(pPleConv), ARR(oPleConv),
  "PLE's dilated depthwise conv, silu and residual add." },
{ "ngram_ids", ARR(pNgramIds), ARR(oNgramIds), "hashed n-gram row ids for PLE." },
{ "gather_rows", ARR(pMnD), ARR(oGather), "y[i,:] = x[idx[i],:]." },
{ "scatter_rows", ARR(pMnD), ARR(oScatterRows),
  "x[idx[i],:] = v[i,:], the mirror of gather_rows: `x` is INOUT and the write is partial, a "
  "negative index writes nothing, and indices are distinct." },
{ "cast", ARR(pCast), ARR(oUnary),
  "dtype conversion. The narrowing direction is the ABI's RTNE, bit for bit -- a vectorised cast "
  "that rounded differently could not be checked against anything." },
{ "softmax", ARR(pMnD), ARR(oUnary), "row-wise softmax, max-subtracted." },
{ "quant_act_i8", ARR(pQuant), ARR(oQuant),
  "symmetric int8 activation quantiser, one f32 scale per `group` contiguous elements. `asum` is "
  "not produced -- libavx has no asymmetric grid to feed, exactly as libref has none." },
{ "quant_act_fp8", ARR(pQuant), ARR(oQuantFp8), "symmetric block-scaled E4M3 quantiser." },
{ "had_quant_act_fp8", ARR(pQuant), ARR(oQuantFp8),
  "width-`group` Hadamard, then the same symmetric block-scaled E4M3 quantiser. No normalisation: "
  "the 1/group is folded into the stored weight scale at convert (docs/OPS.md)." },
{ "gram_accum", ARR(pQuant), ARR(oGram),
  "h += x^T x over a dequantised E4M3 activation. `h` is [n, n] and INOUT." },
{ "had_quant_act_i8", ARR(pQuant), ARR(oHadQuant),
  "Walsh-Hadamard rotation at width `group`, then symmetric int8 with ONE scale per row carrying "
  "the 1/sqrt(group)." },
{ "rmsnorm_had_quant_i8", ARR(pRmsHadQuant), ARR(oRmsHadQuant),
  "RMS norm, optional residual add, rotation and int8 in one pass." },
{ "gated_had_quant_i8", ARR(pGatedQuant), ARR(oGatedQuant),
  "gated product, rotation and int8 in one pass." },
{ "dequant", ARR(pDequant), ARR(oDequant), "group-scaled dequantisation." },
{ "rmsnorm_quant_fp8", ARR(pRmsQuantFp8), ARR(oRmsQuantFp8),
  "RMS norm with an optional residual add, then the E4M3 quantiser, in one pass. The add is "
  "stored before the variance reads it, and the codes are taken over the bf16-narrowed row that "
  "`out_bf16` holds." },
{ "gated_quant_fp8", ARR(pGatedQuantFp8), ARR(oGatedQuantFp8),
  "act(gate) * up over a packed [M, 2n] plane, gate half first, then the E4M3 quantiser over the "
  "bf16-narrowed product. `act` is silu or sigmoid, silu by default." },
{ "gemm_nt", ARR(pGemm), ARR(oGemm),
  "C = A B^T, f32 accumulate, cache-blocked with a register-blocked microkernel. Sums in a "
  "different order from libref's straight loop by construction; the kbench tolerance for a bf16 "
  "gemm_nt (8e-3) is what that costs." },
{ "gemm_nt_bias", ARR(pGemmBias), ARR(oGemmBias),
  "C = act(A B^T + bias) + res, each step rounded to C's dtype." },
{ "gemm_nt_q", ARR(pGemmQ), ARR(oGemmQ),
  "C = A B^T with group-scaled quantised operands. `a_sum` and `b_ref` are libr4d's asymmetric and "
  "mxfp4 forms and are ignored here, as they are in libref." },
{ "gemm_nt_q_bias", ARR(pGemmQ), ARR(oGemmQBias), "quantised C = A B^T + bias." },
{ "gemm_nt_q_gated", ARR(pGemmQGated), ARR(oGemmQGated),
  "gemm_nt_q over a [N, K] gate_up plane with gated_quant_fp8's pass in the epilogue, byte for "
  "byte the pair. `N` is the weight's rows, twice the fused width; the output is [M, N/2]." },
{ "attn_paged", ARR(pAttnPaged), ARR(oAttnPaged),
  "paged causal / sliding-window attention, any query length, online-softmax so one pass over the "
  "cache suffices." },
{ "attn_dense", ARR(pAttnDense), ARR(oAttnDense), "dense varlen attention over cu_seqlens." },
{ "kv_store", ARR(pKvStore), ARR(oKvStore), "scatter k and v into the paged cache." },
{ "rope_table", ARR(pRopeTable), ARR(oRopeTable), "cos/sin at each position's own row." },
{ "rope", ARR(pRope), ARR(oRope), "rotary embedding in place." },
{ "qk_norm_rope", ARR(pQkNormRope), ARR(oQkNormRope), "per-head QK norm then rotary embedding." },
{ "scale_rows", ARR(pScaleRows), ARR(oScaleRows),
  "one scalar per row, optionally through an activation, broadcast across the row." },
{ "router_topk", ARR(pRouter), ARR(oRouter), "softmax then exact top-k routing." },
{ "moe_scatter", ARR(pScatter), ARR(oScatter),
  "stable counting sort of routing slots by expert." },
{ "moe_gemm", ARR(pMoeGemm), ARR(oMoeGemm), "grouped GEMM over the expert-sorted rows." },
{ "moe_gemm_q", ARR(pMoeGemmQ), ARR(oMoeGemmQ),
  "grouped GEMM over the expert-sorted rows, quantised operands." },
{ "moe_gemm_q_mod4", ARR(pMoeGemmQMod4), ARR(oMoeGemmQMod4),
  "grouped GEMM over the expert-sorted rows, quantised operands, four classes by number mod 4." },
{ "moe_gather", ARR(pMoeGather), ARR(oMoeGather), "weighted scatter back to token order." },
{ "row_topk", ARR(pRowTopk), ARR(oRowTopk), "exact per-row top-R." },
{ "row_topk_merge", ARR(pRowTopkMerge), ARR(oRowTopkMerge),
  "the global top-R of the gathered per-rank candidate sets." },
{ "logit_rerank", ARR(pLogitRerank), ARR(oLogitRerank),
  "R candidates a row rescored exactly against the full-precision head." },
{ "gdn_conv_prep", ARR(pConvPrep), ARR(oConvPrep),
  "gdn prefill preamble: conv, split, l2 norm, gate cumsum, beta." },
{ "gdn_conv_update", ARR(pConvUpdate), ARR(oConvUpdate),
  "gdn decode conv over the rolling speculative window." },
{ "gdn_kkt_solve", ARR(pKkt), ARR(oKkt), "gdn chunk preamble: the triangular inverse." },
{ "gdn_chunk_scan", ARR(pChunkScan), ARR(oChunkScan), "gdn chunked scan." },
{ "gdn_recurrent_update", ARR(pRecurrent), ARR(oRecurrent),
  "gdn decode recurrence against the paged state cache, in the PER-CANDIDATE reading libref "
  "implements. `state_form` names the reading; any other form, libr4d's two-slot \"anchor\" "
  "among them, is declined by name as libref declines it." },
{ "gdn_conv_recurrent_update", ARR(pConvRecurrent), ARR(oConvRecurrent),
  "gdn_conv_update followed by gdn_recurrent_update, with q, k and v never written: the operands "
  "are the convolution's (`conv_idx` is its state_idx) and then the recurrence's without those "
  "three." },
{ "gdn_gated_rmsnorm", ARR(pGatedNorm), ARR(oGatedNorm), "gated RMS norm over a head." },
{ "all_reduce", ARR(pAllReduce), ARR(oAllReduce),
  "sum across ranks, f32, ascending rank order." },
{ "all_gather", ARR(pAllGather), ARR(oAllGather), "gather every rank's shard." },
{ "embed_lookup", ARR(pEmbed), ARR(oEmbed), "embedding gather, vocab-parallel with an offset." },
{ "embed_lookup_q", ARR(pEmbed), ARR(oEmbedQ),
  "embedding gather, one scale for the table; without a scale the table is read as it is." },
{ "logits_gemm", ARR(pLogits), ARR(oLogits), "the lm_head projection." },
{ "sample_penalties", ARR(pPenal), ARR(oPenalties),
  "repetition, frequency and presence penalties." },
{ "sample_dry", ARR(pPenal), ARR(oDry), "DRY repetition-suffix penalty, once per token." },
{ "sample_temp", ARR(pMVocab), ARR(oTemp), "per-request temperature." },
{ "sample_topk", ARR(pTopk), ARR(oTopk), "exact top-k candidate set, descending, ties to the "
  "lower id." },
{ "sample_topp", ARR(pMCand), ARR(oNarrow), "nucleus narrowing in place." },
{ "sample_minp", ARR(pMCand), ARR(oNarrow), "min-p narrowing in place." },
{ "sample_typical", ARR(pMCand), ARR(oNarrow), "locally typical narrowing in place." },
{ "sample_xtc", ARR(pMCand), ARR(oNarrow), "exclude-top-choices narrowing in place." },
{ "sample_mask", ARR(pMVocab), ARR(oMask), "grammar bitmask application." },
{ "sample_pick", ARR(pMCand), ARR(oPick),
  "seeded inverse-CDF draw. The uniform is splitmix64 over (seed * 0x9E3779B97F4A7C15 + pos), "
  "top 24 bits as a float in [0,1) -- specified rather than left to a library, because the same "
  "stream has to come out of a host reference, a device kernel and a rerun." },
{ "sample_merge_topk", ARR(pMCand), ARR(oMerge), "the vocab-parallel candidate merge." },
{ "sample_argmax", ARR(pMVocab), ARR(oArgmax), "greedy, ties to the lower token id." },
{ "dflash_select", ARR(pDflashSelect), ARR(oDflashSelect),
  "a greedy walk over a candidate lattice: score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * "
  "hp[l][r] * succ[cand[l, c]][r], pid(0) the anchor and pid(l) the walk's own previous pick. "
  "`hp` absent is a plane of ones." },
};

/* ================================================================== the kernel table
 *
 * Every row is domain HOST at priority 0, unconstrained except where the op's own structure needs
 * a value the operands do not carry. That last clause is kv_store's `block_size`, and it is
 * libref's rule for the same reason: the constraint admits every value and refuses only a query
 * that does not NAME one, because an absent key fails a constraint and a cache cannot be walked
 * without knowing its block edge.
 *
 * SHAPE SAYS "any" ON EVERY ROW, and that is a claim about this plugin rather than a copied
 * phrase: every extent is read off the operands (avx_common.h), so there is no geometry these
 * refuse. Which ISA level runs is NOT part of the shape and deliberately does not appear in a
 * constraint -- a row that said `avx512` would be a row that vanishes from the hierarchy on a
 * machine without it, and the whole point of the dispatch is that the ROW is the same everywhere
 * and only the instruction stream underneath it changes. */
#define SHAPE_ANY "any: every extent is read off the operands, strides included"

#define ROW(nm, opname, fam, what, dts, cons, ncons, fn)                                          \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                  \
      cons, ncons, nullptr, 0, nullptr, nullptr, nullptr, fn, nullptr, nullptr, nullptr,          \
      nullptr, nullptr, nullptr }

#define SIMPLE(nm, opname, fam, what, dts, fn) ROW(nm, opname, fam, what, dts, nullptr, 0, fn)

#define DT_FLOAT "bf16 f16 f32 fp8_e4m3 fp8_e5m2 i8 i32, chosen per operand from its own dtype"
#define DT_ANY   "any dtype rad_types.h names, per operand"

/* The launch is the baseline-compiled thunk, never the level's own symbol -- see avx_dispatch.cpp
 * on why the row must be correct before rad_plugin_open has run. */
#define L(nm) avx_thunk_##nm
#define AVX_THUNK_DECL(nm) extern "C" int avx_thunk_##nm(const RadArgs*, RadStream);
AVX_OP_LIST(AVX_THUNK_DECL)
#undef AVX_THUNK_DECL

static const RadKernelInfo kKernels[] = {
SIMPLE("avx_rmsnorm", "rmsnorm", "norm", "root-mean-square norm with a gain", DT_FLOAT, L(rmsnorm)),
SIMPLE("avx_rmsnorm_add", "rmsnorm_add", "norm", "residual add then RMS norm", DT_FLOAT,
       L(rmsnorm_add)),
SIMPLE("avx_hc_enter", "hc_enter", "norm", "gated residual entry: hc copies of x", DT_FLOAT,
       L(hc_enter)),
SIMPLE("avx_hc_read", "hc_read", "norm",
       "gated residual read: grouped norm, low-rank gate, stream mix", DT_FLOAT, L(hc_read)),
SIMPLE("avx_hc_write", "hc_write", "norm", "gated residual write: h[s] += inj[s] * y", DT_FLOAT,
       L(hc_write)),
SIMPLE("avx_hc_mean", "hc_mean", "norm",
       "the capture's reduction: mean over the hc streams, bf16 in and out", "bf16 in, bf16 out",
       L(hc_mean)),
SIMPLE("avx_mtp_enter", "mtp_enter", "norm",
       "MTP head entry: two norms, a per-sub-stream fc and a broadcast fc", DT_FLOAT, L(mtp_enter)),
SIMPLE("avx_layernorm", "layernorm", "norm", "layer norm with gain and bias", DT_FLOAT,
       L(layernorm)),
SIMPLE("avx_grid_embed", "grid_embed", "elem",
       "bilinear resample of a learned position grid, added in", DT_FLOAT, L(grid_embed)),
SIMPLE("avx_add", "add", "elem", "elementwise sum", DT_FLOAT, L(add)),
SIMPLE("avx_mul", "mul", "elem", "elementwise product, broadcasting b", DT_FLOAT, L(mul)),
SIMPLE("avx_silu_mul", "silu_mul", "elem", "silu-gated product over a fused gate_up", DT_FLOAT,
       L(silu_mul)),
SIMPLE("avx_gelu", "gelu", "elem", "exact erf gelu", DT_FLOAT, L(gelu)),
SIMPLE("avx_silu", "silu", "elem", "silu", DT_FLOAT, L(silu)),
SIMPLE("avx_sigmoid", "sigmoid", "elem", "logistic sigmoid", DT_FLOAT, L(sigmoid)),
SIMPLE("avx_gather_rows", "gather_rows", "elem", "y[i,:] = x[idx[i],:]", DT_ANY, L(gather_rows)),
SIMPLE("avx_scatter_rows", "scatter_rows", "elem", "x[idx[i],:] = v[i,:]", DT_ANY,
       L(scatter_rows)),
SIMPLE("avx_ngram_ids", "ngram_ids", "vocab", "hashed n-gram row ids for PLE", DT_ANY,
       L(ngram_ids)),
SIMPLE("avx_ple_gate", "ple_gate", "norm", "PLE gate: three grouped norms, a dot, a signed sqrt",
       DT_FLOAT, L(ple_gate)),
SIMPLE("avx_ple_conv", "ple_conv", "norm", "PLE's dilated depthwise conv, silu and residual add",
       DT_FLOAT, L(ple_conv)),
SIMPLE("avx_cast", "cast", "elem", "dtype conversion", DT_ANY, L(cast)),
SIMPLE("avx_softmax", "softmax", "elem", "row-wise softmax", DT_FLOAT, L(softmax)),
SIMPLE("avx_scale_rows", "scale_rows", "elem", "one scalar per row, broadcast across the row",
       DT_FLOAT, L(scale_rows)),

SIMPLE("avx_quant_act_i8", "quant_act_i8", "quant", "symmetric int8 activation quantiser",
       "float in, i8 out, f32 scale", L(quant_act_i8)),
SIMPLE("avx_had_quant_act_i8", "had_quant_act_i8", "quant",
       "Walsh-Hadamard rotation then symmetric int8", "float in, i8 out, f32 scale",
       L(had_quant_act_i8)),
SIMPLE("avx_quant_act_fp8", "quant_act_fp8", "quant", "symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale", L(quant_act_fp8)),
SIMPLE("avx_had_quant_act_fp8", "had_quant_act_fp8", "quant",
       "width-`group` Hadamard, then the same symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale per group; n a multiple of group",
       L(had_quant_act_fp8)),
SIMPLE("avx_gram_accum", "gram_accum", "quant",
       "h += x^T x over a dequantised E4M3 activation, or a plain bf16 one",
       "fp8_e4m3 plus an f32 scale per group, or bf16 with no scale, in; an f32 [n][n] "
       "accumulator READ and added to",
       L(gram_accum)),
SIMPLE("avx_rmsnorm_had_quant_i8", "rmsnorm_had_quant_i8", "quant",
       "RMS norm, rotation and int8 in one pass", "float in, i8 out, f32 scale",
       L(rmsnorm_had_quant_i8)),
SIMPLE("avx_gated_had_quant_i8", "gated_had_quant_i8", "quant",
       "gated product, rotation and int8 in one pass", "float in, i8 out, f32 scale",
       L(gated_had_quant_i8)),
SIMPLE("avx_dequant", "dequant", "quant", "group-scaled dequantisation", DT_ANY, L(dequant)),
SIMPLE("avx_rmsnorm_quant_fp8", "rmsnorm_quant_fp8", "quant",
       "RMS norm, optional residual add, and the E4M3 quantiser in one pass",
       "float in, fp8_e4m3 out, f32 scale, bf16 out_bf16", L(rmsnorm_quant_fp8)),
SIMPLE("avx_gated_quant_fp8", "gated_quant_fp8", "quant",
       "act(gate) * up over a packed [M, 2n] plane, then the E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale, bf16 out_bf16", L(gated_quant_fp8)),

SIMPLE("avx_gemm_nt", "gemm_nt", "gemm", "C = A B^T, f32 accumulate, cache-blocked", DT_FLOAT,
       L(gemm_nt)),
SIMPLE("avx_gemm_nt_bias", "gemm_nt_bias", "gemm", "C = A B^T + bias", DT_FLOAT, L(gemm_nt_bias)),
SIMPLE("avx_gemm_nt_q", "gemm_nt_q", "gemm", "C = A B^T with group-scaled quantised operands",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; f32 accumulate", L(gemm_nt_q)),
SIMPLE("avx_gemm_nt_q_bias", "gemm_nt_q_bias", "gemm", "quantised C = A B^T + bias",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; f32 accumulate", L(gemm_nt_q_bias)),
SIMPLE("avx_gemm_nt_q_gated", "gemm_nt_q_gated", "gemm",
       "gemm_nt_q over a gate_up plane with the gated E4M3 quantiser in the epilogue",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; fp8_e4m3 out, f32 scale",
       L(gemm_nt_q_gated)),

ROW("avx_attn_paged", "attn_paged", "attn",
    "paged causal / sliding-window attention, any query length",
    "any q and out dtype; bf16 or fp8 kv cache with optional per-(seq,head) descales",
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])), L(attn_paged)),
SIMPLE("avx_attn_dense", "attn_dense", "attn", "dense varlen attention over cu_seqlens", DT_FLOAT,
       L(attn_dense)),
ROW("avx_kv_store", "kv_store", "attn", "scatter k and v into the paged cache", DT_ANY,
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])), L(kv_store)),
SIMPLE("avx_rope", "rope", "attn", "rotary embedding in place", DT_FLOAT, L(rope)),
SIMPLE("avx_rope_table", "rope_table", "attn", "cos/sin at each position's own row", DT_FLOAT,
       L(rope_table)),
SIMPLE("avx_qk_norm_rope", "qk_norm_rope", "attn", "per-head QK norm then rotary embedding",
       DT_FLOAT, L(qk_norm_rope)),

SIMPLE("avx_router_topk", "router_topk", "moe", "softmax then exact top-k routing", DT_FLOAT,
       L(router_topk)),
SIMPLE("avx_moe_scatter", "moe_scatter", "moe", "stable counting sort of routing slots by expert",
       "i32 indices", L(moe_scatter)),
SIMPLE("avx_moe_gemm", "moe_gemm", "moe", "grouped GEMM over the expert-sorted rows", DT_FLOAT,
       L(moe_gemm)),
SIMPLE("avx_moe_gemm_q", "moe_gemm_q", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands", DT_FLOAT, L(moe_gemm_q)),
SIMPLE("avx_moe_gemm_q_mod4", "moe_gemm_q_mod4", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands, four classes by number mod 4",
       DT_FLOAT, L(moe_gemm_q_mod4)),
SIMPLE("avx_moe_gather", "moe_gather", "moe", "weighted scatter back to token order", DT_FLOAT,
       L(moe_gather)),
SIMPLE("avx_row_topk", "row_topk", "moe", "exact per-row top-R", DT_FLOAT, L(row_topk)),
SIMPLE("avx_row_topk_merge", "row_topk_merge", "moe",
       "the global top-R of the gathered per-rank candidate sets", DT_FLOAT, L(row_topk_merge)),
SIMPLE("avx_logit_rerank", "logit_rerank", "moe",
       "R candidates a row rescored exactly against the full-precision head", DT_FLOAT,
       L(logit_rerank)),

SIMPLE("avx_gdn_conv_prep", "gdn_conv_prep", "gdn",
       "gdn prefill preamble: conv, split, l2 norm, gate cumsum, beta", DT_FLOAT,
       L(gdn_conv_prep)),
SIMPLE("avx_gdn_conv_update", "gdn_conv_update", "gdn",
       "gdn decode conv over the rolling speculative window", DT_FLOAT, L(gdn_conv_update)),
SIMPLE("avx_gdn_kkt_solve", "gdn_kkt_solve", "gdn", "gdn chunk preamble: the triangular inverse",
       DT_FLOAT, L(gdn_kkt_solve)),
SIMPLE("avx_gdn_chunk_scan", "gdn_chunk_scan", "gdn", "gdn chunked scan", DT_FLOAT,
       L(gdn_chunk_scan)),
SIMPLE("avx_gdn_recurrent_update", "gdn_recurrent_update", "gdn",
       "gdn decode recurrence against the paged state cache", DT_FLOAT, L(gdn_recurrent_update)),
SIMPLE("avx_gdn_conv_recurrent_update", "gdn_conv_recurrent_update", "gdn",
       "gdn_conv_update then gdn_recurrent_update, this plugin's own two kernels in turn", DT_FLOAT,
       L(gdn_conv_recurrent_update)),
SIMPLE("avx_gdn_gated_rmsnorm", "gdn_gated_rmsnorm", "gdn", "gated RMS norm over a head", DT_FLOAT,
       L(gdn_gated_rmsnorm)),

/* THE COLLECTIVES HAVE NO init/fini AND THAT IS NOT AN OVERSIGHT. libref's rows carry
 * ref_collective_init because its all-reduce establishes a shared rendezvous across ranks in one
 * process. libavx serves the ONE-RANK case only -- world_size 1, where an all-reduce is a copy and
 * an all-gather is a copy -- and refuses anything wider by name rather than pretending. A
 * multi-rank host collective is a real piece of work (shared memory, a barrier, an ordering rule)
 * and it is not this plugin's: a CPU kernel library has no business owning the wire. */
SIMPLE("avx_all_reduce", "all_reduce", "ar", "sum across ranks; world_size 1 only", DT_FLOAT,
       L(all_reduce)),
SIMPLE("avx_all_gather", "all_gather", "ar", "gather every rank's shard; world_size 1 only",
       DT_FLOAT, L(all_gather)),

SIMPLE("avx_embed_lookup", "embed_lookup", "vocab", "embedding gather", DT_FLOAT, L(embed_lookup)),
/* THE ONE ROW WITH A LAYOUT HOOK, and it only checks: the table is read as it is stored, and the
 * hook refuses a table encoded any other way (avx_ngram.cpp). An E4M3 table into bf16 goes through
 * the row cache; every other dtype pair takes the general gather. Its init refuses a process that
 * may not make the io_uring ring the cache's misses are read through, at declare and not at the
 * first request. */
{ "avx_embed_lookup_q", "embed_lookup_q", "vocab",
  "embedding gather, one scale for the table: an E4M3 table into bf16 through a shared row cache "
  "and direct reads of the container, a later call's rows read ahead",
  SHAPE_ANY, DT_ANY, RAD_DOMAIN_HOST, 0,
  nullptr, 0, nullptr, 0, nullptr,
  /* INIT, fini, LAUNCH, scratch, layout, relayout, opd_shape, unrelayout */
  avx_init_ngram, nullptr, L(embed_lookup_q), nullptr, avx_layout_ngram, nullptr, nullptr, nullptr,
  nullptr, nullptr },
SIMPLE("avx_logits_gemm", "logits_gemm", "vocab", "the lm_head projection", DT_FLOAT,
       L(logits_gemm)),

SIMPLE("avx_sample_penalties", "sample_penalties", "sample",
       "repetition, frequency and presence penalties", DT_FLOAT, L(sample_penalties)),
SIMPLE("avx_sample_dry", "sample_dry", "sample", "DRY repetition-suffix penalty", DT_FLOAT,
       L(sample_dry)),
SIMPLE("avx_sample_temp", "sample_temp", "sample", "per-request temperature", DT_FLOAT,
       L(sample_temp)),
SIMPLE("avx_sample_topk", "sample_topk", "sample", "exact top-k candidate set", DT_FLOAT,
       L(sample_topk)),
SIMPLE("avx_sample_topp", "sample_topp", "sample", "nucleus narrowing in place", DT_FLOAT,
       L(sample_topp)),
SIMPLE("avx_sample_minp", "sample_minp", "sample", "min-p narrowing in place", DT_FLOAT,
       L(sample_minp)),
SIMPLE("avx_sample_typical", "sample_typical", "sample", "locally typical narrowing in place",
       DT_FLOAT, L(sample_typical)),
SIMPLE("avx_sample_xtc", "sample_xtc", "sample", "exclude-top-choices narrowing in place",
       DT_FLOAT, L(sample_xtc)),
SIMPLE("avx_sample_merge_topk", "sample_merge_topk", "sample", "vocab-parallel candidate merge",
       DT_FLOAT, L(sample_merge_topk)),
SIMPLE("avx_sample_mask", "sample_mask", "sample", "grammar bitmask application", DT_FLOAT,
       L(sample_mask)),
SIMPLE("avx_sample_pick", "sample_pick", "sample", "seeded inverse-CDF draw", DT_FLOAT,
       L(sample_pick)),
SIMPLE("avx_sample_argmax", "sample_argmax", "sample", "greedy pick", DT_FLOAT, L(sample_argmax)),
SIMPLE("avx_dflash_select", "dflash_select", "draft",
       "DFlash2's greedy walk over the candidate lattice",
       "i32 candidates and tokens, float unary and codebooks", L(dflash_select)),
};

/* ================================================================== plugin exports */
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libavx",
    "0.1.0",
    "the conventional op vocabulary in hand-written x86 SIMD, dispatched by CPU feature at load: "
    "scalar / AVX / AVX2+FMA / AVX-512. Same ops as libref, same schemas, same answers to a "
    "tolerance -- the difference is the instruction stream",
    /* NOT a -march string. The binary runs everywhere x86-64 runs and picks its own path, so a
     * build_target naming one machine would be a claim the plugin does not make. */
    "host"
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

extern "C" int rad_kernel_schema_count(void) {
    return (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    return &kSchemas[i];
}
extern "C" int rad_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    return &kKernels[i];
}
