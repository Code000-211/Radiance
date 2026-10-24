/* avx_registry.cpp -- the op schemas and kernel rows libavx exports.
 *
 * Schemas match libref's exactly. The first plugin in hierarchy order to declare an op fixes its
 * schema; a later plugin declaring it differently is refused at load (rad_abi.h,
 * core/plugin/loader.cpp: schema_same). That compares parameters position by position (key, type,
 * requiredness) and operands (role, optionality), not operand names. The table is copied rather
 * than shared because a plugin links only abi/; rad-avx-check runs the loader's comparison against
 * libref over every schema before checking any number, so the copy cannot drift silently.
 *
 * No opd_shape. That hook sizes an op's operands for a tool calling it cold; libref implements it
 * for the whole conventional vocabulary (docs/OPS.md), and libavx computes the same ops at the same
 * shapes. A second set of answers could disagree with libref's. A tool with libavx but not libref
 * could not size buffers; both host plugins ship together.
 *
 * No tunables. The vector width is fixed by the ISA level, and the GEMM's M block is derived from K
 * against a fixed L2 byte target (avx_gemm.cpp), with no occupancy interaction to search over.
 */
#include "avx_isa.h"
#include "avx_planes.h"
#include "avx_rows.h"

#include "rad_abi.h"
#include "rad_planes.h"
#include "rad_plugin.h"

#include <cstring>

/* ================================================================== parameter specs
 * Same macro names as libref, so the schema sections of libref/ref_registry.cpp and this file can
 * be diffed directly. */
#define P_INT(k)   { k, RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { k, RAD_P_INT, RAD_OPTIONAL }
#define P_STR(k)   { k, RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { k, RAD_P_STR, RAD_OPTIONAL }
#define P_INTD(k)  { k, RAD_P_INT, RAD_DERIVED }
#define P_CHUNK(k) { k, RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }
#define P_CAP(k)   { k, RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY }
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
/* rmsnorm_add has no `mult`: no fused residual-norm site uses a scaled branch. */
static const RadParamSpec pRmsnormAdd[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                          P_F64O("wadd") };
static const RadParamSpec pRmsnorm[]   = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                           P_F64O("wadd"), P_F64O("mult") };
/* `w` is optional; absent means a gain of 1. */
static const RadOperandSpec oRmsnorm[] = { OPD("x"), WGT_O("w"), OUT("y") };

static const RadParamSpec pHcRead[]    = { P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"),
                                           P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
                                           P_INTO("inject"), P_INTO("group"), P_INTO("rotate"),
                                           P_STRO("mix") };
static const RadOperandSpec oHcRead[]  = { OPD("h"), WGT("w"), WGT("mix_down"), WGT("mix_up"),
                                           WGT_O("inject_w"), OUT("x"), OUT_O("inj"), OUT_O("q"),
                                           OUT_O("scale"), OUT_O("rq"), OUT_O("rscale") };

static const RadParamSpec pMhcPre[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
    P_INT("iterations"), P_F64("eps"), P_F64("hc_eps"), P_STR("dtype") };
static const RadOperandSpec oMhcPre[] = { OPD("h"), WGT("fn"), WGT("scale"), WGT("base"),
    OUT("x"), OUT("post"), OUT("comb") };
static const RadParamSpec pMhcPrePartial[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
    P_INT("chunk"), P_INT("first"), P_INT("chunks"), P_STR("dtype") };
static const RadOperandSpec oMhcPrePartial[] = { OPD("h"), WGT("fn"), OUT("part") };
static const RadParamSpec pMhcPreFinish[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
    P_INT("iterations"), P_F64("eps"), P_F64("hc_eps"), P_INT("chunk"), P_STR("dtype") };
static const RadOperandSpec oMhcPreFinish[] = { OPD("part"), OPD("h"), WGT("scale"), WGT("base"),
    OUT("x"), OUT("post"), OUT("comb") };
static const RadOperandSpec oMhcPost[] = { OPD("x"), OPD("h"), OPD("post"), OPD("comb"), OUT("out") };
static const RadOperandSpec oMhcContract[] = { OPD("h"), OUT("x") };
static const RadParamSpec pKdaUpdate[] = { P_INT("M"), P_INT("heads"), P_INT("head_k"),
    P_INT("head_v"), P_F64("lower_bound"), P_F64("scale"), P_F64("l2_eps"),
    P_STR("mode"), P_STR("dtype"), P_STRO("state_form") };
static const RadOperandSpec oKdaUpdate[] = { OPD("q"), OPD("k"), OPD("v"), OPD("a"), OPD("b"),
    WGT("A_log"), WGT("dt_bias"), INOUT("state"), OPD("state_idx"), OPD("cu"),
    OPD_O("num_accepted"), OPD_O("has_init"), OUT("o") };

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
static const RadParamSpec pCausalConv[]   = { P_INT("M"), P_INT("n"), P_INT("width"),
                                              P_STR("dtype") };
static const RadOperandSpec oCausalConv[] = { OPD("x"), WGT("w"), INOUT("conv_state"), OPD("cu"),
                                              OPD_O("cache_idx"), OPD_O("has_init"),
                                              OPD_O("num_accepted"), OUT("y") };

static const RadParamSpec pHcWrite[]   = { P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype") };
static const RadOperandSpec oHcEnter[] = { OPD("x"), OUT("h") };
static const RadOperandSpec oHcWrite[] = { OPD("y"), OPD("inj"), INOUT("h") };
/* The control vector's projection: one unit direction per layer (f32) and an int32 per-row switch. */
static const RadParamSpec pHcProject[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("scale"),
                                           P_STR("dtype") };
static const RadOperandSpec oHcProject[] = { INOUT("h"), OPD("v"), OPD("on") };

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

static const RadParamSpec pSwigluClamped[] = { P_INT("M"), P_INT("n"), P_F64("limit"), P_STR("dtype") };
static const RadParamSpec pDsaPool[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_STR("dtype"), P_INTO("block_size") };
static const RadParamSpec pMlaAttn[] = { P_INT("M"), P_INT("latent_dim"), P_INT("heads"),
    P_INT("blocks"), P_INT("block_size"), P_F64("scale"), P_STR("dtype") };
static const RadParamSpec pMlaProject[] = { P_INT("M"), P_INT("heads"), P_INT("head_dim"),
    P_INT("latent_dim"), P_STR("mode"), P_STR("dtype") };
static const RadOperandSpec oMlaAttn[] = { OPD("q"), OPD("kv"), OPD("bt"), OPD("length"), OUT("out") };
static const RadParamSpec pDsaScore[] = { P_INT("M"), P_INT("n"), P_INT("heads"),
    P_INT("blocks"), P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oDsaScore[] = { OPD("q"), OPD("q_scale"), OPD("gate"), OPD("k"),
    OPD_O("k_scale"), OPD("bt"), OPD("cu"), OPD("nc"), OUT("score") };
static const RadOperandSpec oDsaPool[] = { OPD("kg"), WGT("ape"), OUT("q"), OUT("scale"),
    OPD_O("page"), INOUT_O("cache") };
static const RadParamSpec pRouterSigmoid[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
    P_INT("norm"), P_F64("route_scale"), P_STR("dtype") };
static const RadOperandSpec oRouterSigmoid[] = { OPD("logits"), WGT("bias"), OUT("ids"), OUT("weights") };
static const RadParamSpec pMnD[]      = { P_INT("M"), P_INT("n"), P_STR("dtype") };
/* `add`'s `div` divides the rounded sum (libref's pAdd). */
static const RadParamSpec pAdd[]      = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("div") };
static const RadParamSpec pMulWeight[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_INTO("scalar") };
/* `cap` is required (as in libref's pMnCap). */
static const RadParamSpec pMnCap[]    = { P_INT("M"), P_INT("n"), P_F64("cap"), P_STR("dtype") };
static const RadOperandSpec oBinary[] = { OPD("a"), OPD("b"), OUT("y") };
static const RadOperandSpec oMulWeight[] = { OPD("x"), WGT("w"), OUT("y") };
static const RadParamSpec pActMul[]   = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };
static const RadParamSpec pSegMean[]  = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oSegMean[] = { OPD("x"), OPD("cu_seqlens"), OUT("y") };
static const RadParamSpec pPoolRows[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_F64O("mult"),
                                          P_STR("dtype") };
static const RadOperandSpec oGateUp[] = { OPD("gate_up"), OUT("y") };
static const RadOperandSpec oUnary[]  = { OPD("x"), OUT("y") };
/* The audio (USM conformer) tower's operators, as in libref. */
static const RadOperandSpec oClamp[]  = { OPD("x"), WGT("lo"), WGT("hi"), OUT("y") };
static const RadParamSpec pConv2d[]   = { P_INT("M"), P_INT("width"), P_INT("cin"), P_INT("cout"),
                                          P_INT("kh"), P_INT("kw"), P_INT("stride"), P_INT("pad"),
                                          P_INTO("cu_div"), P_INTO("valid_stride"),
                                          P_STR("dtype"), P_INTO("t_last"), P_STRO("act") };
static const RadOperandSpec oConv2d[] = { OPD("x"), WGT("w"), OPD("cu_seqlens"), OPD_O("valid"),
                                          OUT("y"), WGT_O("b") };
static const RadParamSpec pConv2dUnfold[] = { P_INT("M"), P_INT("width"), P_INT("cin"),
                                              P_INT("kh"), P_INT("kw"), P_INT("stride"),
                                              P_INT("pad"), P_INTO("cu_div"),
                                              P_INTO("valid_stride"), P_INTO("t_last"),
                                              P_STR("dtype") };
static const RadOperandSpec oConv2dUnfold[] = { OPD("x"), OPD("cu_seqlens"), OPD_O("valid"),
                                                OUT("y") };
static const RadParamSpec pConv1dDw[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_INTO("cu_div"),
                                          P_STR("dtype") };
static const RadOperandSpec oConv1dDw[] = { OPD("x"), WGT("w"), OPD("cu_seqlens"), OUT("y"),
                                            WGT_O("b") };
static const RadParamSpec pRelSin[]   = { P_INT("M"), P_INT("n"), P_F64O("max_timescale"),
                                          P_STR("dtype") };
static const RadOperandSpec oRelSin[] = { OUT("y") };
static const RadParamSpec pSegSin[]   = { P_INT("M"), P_INT("n"), P_INT("period"),
                                          P_F64O("max_timescale"), P_INTO("cu_div"), P_STR("dtype") };
static const RadOperandSpec oSegSin[] = { OPD("cu_seqlens"), OUT("y") };
static const RadParamSpec pAttnLocalRel[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                              P_INT("left"), P_INT("right"), P_F64("softcap"),
                                              P_INTO("cu_div"), P_STR("dtype") };
static const RadOperandSpec oAttnLocalRel[] = { OPD("q"), OPD("k"), OPD("v"), OPD("relk"),
                                                WGT("pds"), OPD("cu_seqlens"), OUT("out") };
/* `fill` has no input; it writes `value`. */
static const RadParamSpec pFill[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("value") };
static const RadOperandSpec oFill[] = { OUT("y") };
static const RadOperandSpec oScale[]  = { OPD("x"), WGT("s"), OUT("y") };
static const RadOperandSpec oGather[] = { OPD("x"), OPD("idx"), OUT("y") };
static const RadOperandSpec oScatterRows[] = { OPD("v"), OPD("idx"), INOUT("x") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

/* ---- quantisation and rotation ------------------------------------------------------------ */
static const RadParamSpec pQuant[]      = { P_INT("M"), P_INT("n"), P_INT("group"),
                                            P_STR("dtype") };
static const RadOperandSpec oQuant[]    = { OPD("x"), OUT("q"), OUT("scale"), OUT_O("asum") };
static const RadOperandSpec oHadQuant[] = { OPD("x"), OUT("q"), OUT("scale") };
static const RadOperandSpec oQuantFp8[] = { OPD("x"), OUT("q"), OUT("scale") };
static const RadParamSpec pQuantCols[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oQuantCols[] = { OPD("x"), OUT("y"), OUT("scale") };
/* ParoQuant's rotation, as libref declares it: the tables come from the "givens" encoding
 * (abi/rad_encoding.h) and are weight operands, placed with the weight they belong to. */
static const RadParamSpec pGivensRotate[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                             P_INT("stages"), P_INT("parts") };
static const RadParamSpec pGivensDense[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                             P_INT("parts"), P_STR("out") };
static const RadOperandSpec oGivensDense[] = { OPD("x"), WGT("tdense"), WGT("rscale"), OUT("q"),
                                               OUT("scale") };
static const RadOperandSpec oGivensRotate[] = { OPD("x"), WGT("pairs"), WGT("angle"), WGT("rscale"),
                                                OUT("y") };

/* The fp8 fusions: the unfused pair's operands minus the intermediate tensor, which `out_bf16`
 * can optionally still produce. */
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

/* `a_scale` is optional: a "bf16" activation has none. */
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

/* `split_n` is libr4d's (docs/OPS.md); accepted and ignored here. */
static const RadParamSpec pGemmQ[]        = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                              P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                              P_INTO("split_n") };
static const RadParamSpec pGemmQBias[]    = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                              P_STR("dtype") };
static const RadOperandSpec oGemmQ[]      = { OPD("a"), OPD_O("a_scale"), WGT("b"), WGT("b_scale"),
                                              OUT("y"), OPD_O("a_sum"), WGT_O("b_ref") };
static const RadOperandSpec oGemmQBias[]  = { OPD("a"), OPD("a_scale"), WGT("b"), WGT("b_scale"),
                                              WGT("bias"), OUT("y") };
/* Runtime LoRA, libref's schemas slot for slot (libref/ref_registry.cpp says what they mean). */
static const RadParamSpec pLoraShrink[]   = { P_INT("M"), P_INT("K"), P_INT("R"), P_INT("S"),
                                              P_INTO("group"), P_STR("dtype") };
static const RadOperandSpec oLoraShrink[] = { OPD("a"), OPD_O("a_scale"), OPD("slot"),
                                              OPD("lora_a"), OUT("t") };
static const RadParamSpec pLoraExpand[]   = { P_INT("M"), P_INT("N"), P_INT("R"), P_INT("S"),
                                              P_STR("dtype") };
static const RadOperandSpec oLoraExpand[] = { OPD("t"), OPD("slot"), OPD("lora_b"), OPD("scale"),
                                              INOUT("y") };
/* `act` is required here (optional in the quantiser); `N` is the weight's rows, twice the output
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
    OPD_O("k_descale"), OPD_O("v_descale"), OPD_O("cu_seqlens"), OUT("out")
};
static const RadConstraint cPagedBlock[] = { RAD_CGE("block_size", 1) };

static const RadParamSpec pAttnDense[]   = { P_INT("M"), P_INT("head_dim"), P_INT("gqa"),
                                             P_INT("causal"), P_STR("dtype"), P_F64O("scale"),
                                             P_INTO("max_seqlen"), P_INTO("window"),
                                             P_INTO("cu_div"), P_INTO("block"),
                                             P_INTO("window_right") };
static const RadOperandSpec oAttnDense[] = { OPD("q"), OPD("k"), OPD("v"), OPD("cu_seqlens"),
                                             OUT("out") };
static const RadParamSpec pAttnCross[]   = { P_INT("M"), P_INT("head_dim"), P_INT("gqa"),
                                             P_INT("causal"), P_STR("dtype"), P_F64O("scale") };
static const RadOperandSpec oAttnCross[] = { OPD("q"), OPD("k"), OPD("v"), OPD("cu_seqlens"),
                                             OPD("cu_k"), OUT("out") };
static const RadParamSpec pAttnNa3d[] = { P_INT("M"), P_INT("head_dim"), P_INT("kt"), P_INT("kh"),
                                          P_INT("kw"), P_STR("dtype"), P_F64O("scale") };
static const RadOperandSpec oAttnNa3d[] = { OPD("q"), OPD("k"), OPD("v"), OPD("geom"), OUT("out") };

static const RadParamSpec pKvStore[]   = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                           P_INT("block_size"), P_STR("kv_dtype"),
                                           P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { OPD("k"), OPD("v"), OPD("slot_mapping"),
                                           INOUT("kv_cache") };

static const RadParamSpec pRope[]   = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                        P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                        P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections"),
                                        P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                        P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                        P_STRO("ladder"), P_F64O("llama3_factor"),
                                        P_F64O("llama3_low"), P_F64O("llama3_high"),
                                        P_F64O("llama3_orig") };
static const RadOperandSpec oRope[] = { INOUT("qkv"), OPD("positions") };

static const RadParamSpec pRopeTable[]   = { P_INT("M"), P_INT("rot"), P_F64("theta"),
                                             P_F64("scale"), P_STR("dtype"),
                                             P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                             P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                             P_INTO("head_dim"), P_STRO("ladder") };
static const RadOperandSpec oRopeTable[] = { OPD("positions"), OUT("cos_sin") };

/* QSA's compressed key, libr4d's schema (libr4d/r4d_rows.cpp): `block_size` is `ratio` under the
 * name the block-key store's KV group uses; `page` and `pos` are optional. */
static const RadParamSpec pQsaBlockKey[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                             P_INT("rotary_dim"), P_INTO("pos0"),
                                             P_INTO("block_size"), P_F64("eps"), P_F64("theta"),
                                             P_F64O("scale"), P_F64O("wadd"), P_STR("dtype"),
                                             P_STRO("mode"), P_STRO("sections"),
                                             P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                             P_F64O("yarn_high"), P_F64O("yarn_magnitude") };
static const RadOperandSpec oQsaBlockKey[] = { OPD("k"), WGT("w"), OPD_O("page"), OPD_O("pos"),
                                               OUT("out") };

/* The rest of the QSA indexer, as libref declares it. `M` counts query rows (score, select) or
 * tokens (tail store, work list); `seqs` is the declared height of `cu`'s sequences, whose count is
 * cu's length less one. `work` bounds the blocks that can complete. */
static const RadParamSpec pQsaScore[] = { P_INT("M"), P_INT("n"), P_INT("heads"), P_INT("blocks"),
                                          P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaScore[] = { OPD("q"), OPD("bk"), OPD("bt"), OPD("cu"), OPD("nc"),
                                            OUT("score") };
static const RadParamSpec pQsaSelect[] = { P_INT("M"), P_INT("blocks"), P_INT("topk"),
                                           P_INT("ratio"), P_INT("seqs"), P_INTO("signed") };
static const RadOperandSpec oQsaSelect[] = { OPD("score"), OPD("bt"), OPD("nc"), OPD("pos"),
                                             OPD("cu"), OUT("sel"), OUT("nsel"), OUT("seqused"), OUT_O("selb") };
static const RadParamSpec pQsaTailStore[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("ring"),
                                              P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaTailStore[] = { OPD("k"), OPD("pos"), OPD("cu"), INOUT("state"),
                                                OPD("sidx") };
static const RadParamSpec pQsaWork[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("ring"),
                                         P_INT("seqs"), P_CAP("work"), P_STR("dtype") };
static const RadOperandSpec oQsaWork[] = { OPD("k"), OPD("pos"), OPD("cu"), OPD("state"),
                                           OPD("sidx"), OPD("bt"), OPD_O("rope"), OUT("stage"),
                                           OUT("page"), OUT("bpos"), OUT("nc") };

static const RadParamSpec pQkNormRope[]   = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                              P_INT("n_head_kv"), P_F64("theta"), P_F64("eps"),
                                              P_F64O("scale"), P_STRO("mode"),
                                              P_INTO("rotary_dim"), P_F64O("wadd"),
                                              P_INTO("v_norm"), P_STRO("sections"),
                                              P_STRO("comps") };
/* `positions` may be absent at `rotary_dim` 0, and an absent gain is a unit one (libref). */
static const RadOperandSpec oQkNormRope[] = { INOUT("qkv"), OPD_O("positions"), WGT_O("q_w"),
                                              WGT_O("k_w") };

/* ---- mixture of experts ------------------------------------------------------------------- */
static const RadParamSpec pScaleRows[]   = { P_INT("M"), P_INT("n"), P_STR("act"),
                                             P_STR("dtype") };
/* `add` is optional and comes before `y`; operands are positional, so `y` is index 3. */
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

/* Experts in two tables by parity: `w` the even ones at [N, K], `w_odd` the odd ones at
 * [N_odd, K_odd], with `parts` parts stacked along the rows and the output (see libr4d's
 * schema). */
static const RadParamSpec pMoeGemmQ[]   = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                            P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                            P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"),
                                            P_INTO("parts") };
static const RadOperandSpec oMoeGemmQ[] = { OPD("a"), OPD_O("a_scale"), WTAB("w"), WTAB("w_scale"),
                                            OPD("sorted_tok"), OPD("expert_offset"),
                                            WTAB("w_odd"), WTAB("w_odd_scale"), OUT("y") };
/* The same over four classes by expert number mod 4, `w_q` class q at [N_q, K_q], as in libref
 * and libr4d. */
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

/* With `shared` and `shared_gate`, the shared expert is folded in: y = narrow(shared * act(gate))
 * + narrow(gathered), the unfused pair's roundings. `act` is sigmoid or none, as in scale_rows,
 * applied to the per-token gate scalar. */
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
static const RadParamSpec pPairArgmax[]     = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oPairArgmax[]   = { OPD("pairs"), OUT("idx"), OUT_O("val") };

/* As libref: R candidates rescored against the full-precision head, ties to the lower id, pads
 * last. libref's shape hook does not describe this op, so the checker skips its geometries, but
 * the row is needed for the coverage gate. */
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
 * recurrence's minus q, k and v; parameters are the union of both ops'. */
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
                                             P_INT("exact"), P_INTO("min_bytes"), P_INTO("hops"),
                                             P_INTO("overlap") };
static const RadOperandSpec oAllReduce[] = { INOUT("x"), OUT_O("y") };

static const RadParamSpec pAllGather[]   = { P_INT("world_size"), P_INT("numel"), P_STR("dtype"),
                                             P_INTO("row") };
static const RadOperandSpec oAllGather[] = { OPD("x"), OUT("y") };

/* ---- image generation --------------------------------------------------------------------- */
static const RadParamSpec pLnMod[]   = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLnMod[] = { OPD("x"), OPD("s"), OPD("cu_seqlens"), OUT("y") };
static const RadParamSpec pTableAdd[] = { P_INT("M"), P_INT("n"), P_INT("L"), P_STR("dtype") };
static const RadOperandSpec oTableAdd[] = { OPD("x"), WGT("t"), OUT("y") };
static const RadParamSpec pMixRows[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_STR("dtype") };
static const RadParamSpec pChanStats[] = { P_INT("M"), P_INT("n"), P_STR("mode"), P_INTO("var"),
                                           P_F64O("eps"), P_STR("dtype") };
static const RadOperandSpec oChanStats[] = { OPD("x"), WGT("mean"), WGT("s"), OUT("y") };
static const RadOperandSpec oMixRows[] = { OPD("x"), WGT("w"), OUT("y") };
static const RadParamSpec pGateAdd[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oGateAdd[] = { OPD("y"), OPD("g"), OPD("cu_seqlens"), INOUT("x") };
static const RadParamSpec pLnModQ[]  = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                         P_INT("gated"), P_STR("dtype") };
static const RadOperandSpec oLnModQ[] = { OPD_O("y"), OPD_O("g"), OPD("s"), OPD("cu_seqlens"),
                                          INOUT("x"), OUT("q"), OUT("scale"), OUT_O("z") };
static const RadParamSpec pRmsMod[]  = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INTO("gated"),
                                         P_STR("dtype"), P_F64O("wadd"),
                                         P_INTO("norm_once"), P_STRO("norm") };
static const RadOperandSpec oRmsMod[] = { OPD_O("y"), OPD_O("g"), OPD("s"), WGT_O("w"),
                                          OPD("cu_seqlens"), INOUT("x"), OUT("z") };
static const RadParamSpec pRmsModQ[]  = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                          P_INT("gated"), P_STR("dtype"), P_F64O("wadd"),
                                         P_INTO("norm_once"), P_STRO("norm") };
static const RadOperandSpec oRmsModQ[] = { OPD_O("y"), OPD_O("g"), OPD("s"), WGT_O("w"),
                                           OPD("cu_seqlens"), INOUT("x"), OUT("q"), OUT("scale"),
                                           OUT_O("z") };
static const RadOperandSpec oFlowX0[] = { OPD("v"), OPD("t"), OPD("sigma"), OPD("sigma_next"),
                                          INOUT("x") };
static const RadParamSpec pTimestep[] = { P_INT("M"), P_INT("n"), P_F64("scale"), P_F64("max_period"),
                                          P_INTO("round"), P_STR("dtype") };
static const RadOperandSpec oTimestep[] = { OPD("t"), OUT("y") };
static const RadOperandSpec oGather2[]  = { OPD("a"), OPD("b"), OPD("idx"), OUT("y") };
static const RadParamSpec pLane[] = { P_INT("M"), P_INT("n"), P_INT("stride"), P_STR("dtype") };
static const RadOperandSpec oLaneGather[] = { OPD("src"), OPD("lane"), OPD("cu_seqlens"), OUT("y") };
static const RadOperandSpec oLaneStore[]  = { OPD("src"), OPD("lane"), INOUT("dst") };
static const RadOperandSpec oRandnLanes[] = { OPD("flags"), OPD("seed"), OPD("lane"),
                                              OPD("cu_seqlens"), INOUT("dst") };
static const RadOperandSpec oFlowEuler[]  = { OPD("v"), OPD("sigma"), OPD("sigma_next"), OPD("cfg"),
                                              OPD("pair"), OPD("lane"), OPD("flags"), OPD("seed"),
                                              OPD("cu_seqlens"), OPD_O("inp_lat"), OPD_O("inp_mask"),
                                              INOUT("state"), INOUT_O("preview") };
/* The k-diffusion sampler's input and step: `pred` "eps" or "v", the lanes f32. */
static const RadParamSpec pKdStep[] = { P_INT("M"), P_INT("n"), P_INT("stride"), P_STR("pred"),
                                        P_STR("dtype") };
static const RadOperandSpec oKdInput[] = { OPD("sigma"), OPD("pair"), OPD("lane"), OPD("flags"),
                                           OPD("seed"), OPD("cu_seqlens"), INOUT("state"),
                                           OUT("y") };
static const RadOperandSpec oKdStep[]  = { OPD("out"), OPD("sigma"), OPD("sigma_next"),
                                           OPD("sigma_prev"), OPD("cfg"), OPD("pair"), OPD("lane"),
                                           OPD("flags"), OPD("seed"), OPD("sampler"), OPD("step"),
                                           OPD("cu_seqlens"), INOUT("state"), INOUT("hist"),
                                           INOUT_O("preview") };
static const RadParamSpec pChanAffine[] = { P_INT("M"), P_INT("n"), P_STR("mul"), P_STR("add"),
                                            P_STR("mode"), P_STR("dtype"), P_INTO("round") };
static const RadParamSpec pConvNhwc[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                          P_INT("stride"), P_STR("pad"), P_INT("up"), P_STR("dtype") };
static const RadOperandSpec oConvNhwc[] = { OPD("x"), WGT("w"), WGT("b"), OUT("y") };
static const RadParamSpec pIm2col[] = { P_INT("M"), P_INT("cin"), P_INT("k"), P_INT("stride"),
                                        P_STR("pad"), P_INT("up"), P_STR("dtype") };
static const RadParamSpec pL2Norm[] = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };
static const RadOperandSpec oL2Norm[] = { OPD("x"), WGT("gamma"), OUT("y") };
static const RadParamSpec pSpaceDepth[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("ft"),
                                            P_STR("dtype") };
static const RadOperandSpec oSpaceDepth[] = { OPD("x"), INOUT("y") };
static const RadParamSpec pAttnPS[] = { P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"),
                                        P_INTD("block_size"), P_INT("causal"), P_STR("q_dtype"),
                                        P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
                                        P_INTO("n_head"), P_INTO("max_seqs"), P_STRO("qk"),
                                        P_STRO("pv") };
static const RadOperandSpec oAttnPS[] = { OPD("q"), OPD("k"), OPD("v"), OPD_O("kv_cache"),
                                          OPD_O("block_table"), OPD_O("seqused"), OPD("cu_seqlens"),
                                          OPD_O("k_descale"), OPD_O("v_descale"), OPD_O("k_i8"),
                                          OPD_O("k_scale"), OUT("out"), OPD_O("v_scale") };

/* ---- video generation (avx_video.cpp), as libref declares it ------------------------------ */
static const RadParamSpec pPatchUnfold[] = { P_INT("M"), P_INT("C"), P_INT("pt"), P_INT("ph"),
                                             P_INT("pw"), P_STR("mul"), P_STR("add"),
                                             P_STR("dtype") };
static const RadOperandSpec oPatchUnfold[] = { OPD("x"), OPD("geom"), OUT("y") };
static const RadParamSpec pVtileGather[] = { P_INT("M"), P_INT("n"), P_INT("tile"),
                                             P_INT("overlap"), P_INT("frames"), P_INT("period"),
                                             P_INT("n_reg"), P_STR("dtype") };
static const RadOperandSpec oVtileGather[] = { OPD("x"), WGT("reg"), OPD("geom"), OUT("y") };
static const RadParamSpec pVtileStitch[] = { P_INT("M"), P_INT("tile"), P_INT("overlap"),
                                             P_INT("scale"), P_INT("pt"), P_INT("skip"),
                                             P_INT("keep"), P_INT("blend"), P_INT("period"),
                                             P_STR("mean"), P_STR("std") };
static const RadOperandSpec oVtileStitch[] = { OPD("x"), OPD("geom"), INOUT("prev"), OUT("y") };
static const RadParamSpec pGridPos3[] = { P_INT("M"), P_INT("T"), P_INT("H"), P_INT("W"),
                                          P_INT("period"), P_F64("scale") };
static const RadOperandSpec oGridPos3[] = { OUT("y") };
/* ---- video sparse attention (avx_vsa.cpp), as libref declares it ------------------------- */
static const RadParamSpec pVsa[] = { P_INT("M"), P_INT("head_dim"), P_INT("tile_t"),
                                     P_INT("tile_h"), P_INT("tile_w"), P_INT("patch"),
                                     P_F64("sparsity"), P_F64O("scale") };
static const RadOperandSpec oVsaSelect[] = { OPD("q"), OPD("k"), OPD_O("v"), OPD("segs"),
                                             OPD("geom"), OUT("sel"), OUT_O("coarse") };
/* attn_vsa adds attn_prefix_self's int8 Q.K form at the end: `qk` "i8" with `k_i8` and `k_scale`,
 * quant_act_i8g's codes and scales of k at group head_dim. */
static const RadParamSpec pAttnVsa[] = { P_INT("M"), P_INT("head_dim"), P_INT("tile_t"),
                                         P_INT("tile_h"), P_INT("tile_w"), P_INT("patch"),
                                         P_F64("sparsity"), P_F64O("scale"), P_STRO("qk"),
                                         P_STRO("pv") };
static const RadOperandSpec oAttnVsa[] = { OPD("q"), OPD("k"), OPD("v"), OPD("segs"),
                                           OPD("geom"), OPD("sel"), OPD_O("coarse"),
                                           OPD_O("gate"), OUT("out"), OPD_O("k_i8"),
                                           OPD_O("k_scale"), OPD_O("v_scale") };
/* ---- the latent upscaler (avx_video.cpp), as libref declares it; patch_fold takes
 * patch_unfold's parameters and operands ------------------------------------------------------ */
static const RadParamSpec pConv3d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                        P_INT("kt"), P_STR("dtype"), P_STRO("tpad"),
                                        P_INTO("tpre") };
static const RadOperandSpec oConv3d[] = { OPD("x"), WGT("w"), WGT("b"), OPD("geom"),
                                          OPD_O("res"), OUT("y") };
static const RadParamSpec pGroupNorm[] = { P_INT("M"), P_INT("n"), P_INT("groups"), P_F64("eps"),
                                           P_STR("act"), P_STR("dtype") };
static const RadOperandSpec oGroupNorm[] = { OPD("x"), WGT("w"), WGT("b"), OPD_O("mod"),
                                             INOUT("stats"), OUT("y") };
static const RadParamSpec pTconvDw[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_STR("dtype") };
static const RadOperandSpec oTconvDw[] = { OPD("x"), WGT("w"), WGT("b"), OPD("geom"), OUT("y") };
static const RadOperandSpec oResize[] = { OPD("x"), OPD("geom"), OUT("y") };
static const RadParamSpec pFlowRenoise[] = { P_INT("M"), P_INT("n"), P_INT("stream"),
                                             P_STR("dtype") };
static const RadOperandSpec oFlowRenoise[] = { OPD("sigma"), OPD("seed"), INOUT("x") };

/* ---- LTX-2 (avx_ltx.cpp) */
static const RadParamSpec pRopeSplit[] = { P_INT("M"), P_INT("n_head"), P_INT("head_dim"),
                                           P_INT("dim"), P_INT("n_dims"), P_F64("theta"),
                                           P_STR("max"), P_INTO("head0"), P_STR("dtype") };
static const RadOperandSpec oRopeSplit[] = { INOUT("x"), OPD("positions") };
static const RadParamSpec pHeadGate[] = { P_INT("M"), P_INT("n_head"), P_INT("head_dim"),
                                          P_STR("dtype") };
static const RadOperandSpec oHeadGate[] = { OPD("g"), INOUT("x") };
static const RadParamSpec pLtxGuide[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oLtxGuide[] = { OPD("vc"), OPD_O("vu"), OPD_O("vs"), OPD_O("vm"),
                                            OPD("scales"), OPD("sched"), INOUT("stats"),
                                            INOUT("x") };
static const RadParamSpec pPixelNorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("form"),
                                           P_STRO("act"), P_INTO("ileave"), P_STR("dtype"),
                                           P_F64O("mult") };
static const RadOperandSpec oPixelNorm[] = { OPD("x"), OUT("y") };
static const RadParamSpec pD2s3d[] = { P_INT("M"), P_INT("C"), P_INT("st"), P_INT("sh"), P_INT("sw"),
                                       P_INTO("drop"), P_STR("dtype") };
static const RadOperandSpec oD2s3d[] = { OPD("x"), OPD("geom"), OUT("y") };
static const RadParamSpec pS2d3d[] = { P_INT("M"), P_INT("C"), P_INT("st"), P_INT("sh"), P_INT("sw"),
                                       P_INTO("tpre"), P_INTO("group"), P_STR("mode"),
                                       P_INTO("add"), P_STR("dtype") };
static const RadOperandSpec oS2d3d[] = { OPD("x"), OPD("geom"), INOUT("y") };
static const RadParamSpec pPixPatch[] = { P_INT("M"), P_INT("p"), P_INT("C"), P_STRO("order"),
                                          P_STR("dtype") };
static const RadOperandSpec oPixPatch[] = { OPD("x"), OUT("y") };
static const RadParamSpec pVolIndex[] = { P_INT("M") };
static const RadOperandSpec oVolIndex[] = { OPD("geom"), OUT("y") };
static const RadParamSpec pTileGrid[] = { P_INT("M"), P_INT("tile"), P_INT("stride"), P_STRO("rule"),
                                          P_INTO("min"), P_INTO("tile_t"), P_INTO("stride_t"),
                                          P_INTO("min_t"), P_INTO("tail") };
static const RadOperandSpec oTileGrid[] = { OPD("geom"), OUT("y") };
static const RadParamSpec pTileCrop[] = { P_INT("M"), P_INT("C"), P_STR("dtype") };
static const RadOperandSpec oTileCrop[] = { OPD("x"), OPD("geom"), OPD("box"), OUT("y") };
static const RadParamSpec pTileStitch[] = { P_INT("M"), P_INT("tile"), P_INT("stride"),
                                            P_STRO("rule"), P_INTO("min"), P_INTO("tile_t"),
                                            P_INTO("stride_t"), P_INTO("min_t"), P_INT("scale"),
                                            P_INTO("scale_t"), P_INT("p"), P_INT("C"),
                                            P_STRO("order"), P_STR("dtype") };
static const RadOperandSpec oTileStitch[] = { OPD("x"), OPD("geom"), OPD("first"), OUT("y") };
static const RadParamSpec pStftMel[] = { P_INT("M"), P_INT("hop"), P_INT("nfft"), P_INT("n_mel"),
                                         P_INT("ch"), P_STR("dtype") };
static const RadOperandSpec oStftMel[] = { OPD("x"), WGT("basis"), WGT("mel"), OUT("y") };
static const RadParamSpec pBweUp[] = { P_INT("M"), P_INT("ratio"), P_INT("ch"), P_STR("dtype") };
static const RadOperandSpec oBweUp[] = { OPD("x"), OPD("res"), OUT("y") };

/* ---- audio generation (avx_audio.cpp), as libref declares it ------------------------------ */
static const RadParamSpec pConv1d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                        P_INT("dil"), P_STRO("act"), P_STR("dtype"),
                                        P_INTO("causal") };
static const RadOperandSpec oConv1d[] = { OPD("x"), WGT("w"), OPD("cu_seqlens"), OPD_O("res"),
                                          OUT("y"), WGT_O("b") };
static const RadParamSpec pConvT1d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("stride"),
                                         P_STR("dtype"), P_INTO("pad"), P_INTO("k") };
static const RadOperandSpec oConvT1d[] = { OPD("x"), WGT("w"), WGT("b"), OPD("cu_seqlens"),
                                           OUT("y") };
static const RadParamSpec pSnake[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_INTO("log") };
static const RadOperandSpec oSnake[] = { OPD("x"), WGT("alpha"), OUT("y"), WGT_O("beta") };
static const RadParamSpec pSnakeAa[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("k"),
                                         P_INTO("log"), P_STR("dtype") };
static const RadOperandSpec oSnakeAa[] = { OPD("x"), WGT("alpha"), WGT("beta"), WGT("f_up"),
                                           WGT("f_down"), OPD("cu_seqlens"), OUT("y") };
static const RadParamSpec pCfgTopk[] = { P_INT("M"), P_INT("n_vocab"), P_INT("k"), P_INT("rep"),
                                         P_INTO("cond_k"), P_F64("cfg"), P_F64O("temp"),
                                         P_INT("stream"), P_STR("dtype"), P_INTO("sup_lo"),
                                         P_INTO("sup_hi"), P_F64O("penalty") };
static const RadOperandSpec oCfgTopk[] = { OPD("logits"), OPD("seed"), OPD("frame"), OUT("ids"),
                                           OPD_O("cfg"), OPD_O("lane"), INOUT_O("hist") };
static const RadParamSpec pRvqEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_codes"),
                                          P_INT("vocab"), P_INT("off0"), P_F64("scale"),
                                          P_STR("dtype"), P_INTO("round") };
static const RadOperandSpec oRvqEmbed[] = { OPD("codes"), WGT("t0"), WGT("t1"), OUT("y") };
static const RadParamSpec pFrameMix[] = { P_INT("M"), P_INT("n_embd"), P_INT("L"), P_INT("stride"),
                                          P_INTO("rep"), P_STR("dtype") };
static const RadOperandSpec oFrameMix[] = { OPD("hid"), WGT("logits"), WGT("scale"), OPD("lane"),
                                            OPD("frame"), OPD("status"), INOUT("dst") };
static const RadParamSpec pEndStatus[] = { P_INT("M"), P_INT("end"), P_INT("rep") };
static const RadOperandSpec oEndStatus[] = { OPD("ids"), OPD("lane"), INOUT("status") };
static const RadParamSpec pFrameStore[] = { P_INT("M"), P_INT("n_codes"), P_INT("stride"),
                                            P_INTO("rep"), P_INTO("skip") };
static const RadOperandSpec oFrameStore[] = { OPD("codes"), OPD("lane"), OPD("frame"),
                                              OPD("status"), INOUT("dst") };
static const RadParamSpec pFourier[] = { P_INT("M"), P_INT("n"), P_INTO("round"), P_STR("dtype") };
static const RadOperandSpec oFourier[] = { OPD("t"), WGT("w"), OUT("y") };
static const RadParamSpec pWinInput[] = { P_INT("M"), P_INT("n_lat"), P_INT("n_zero"),
                                          P_INT("n_cond"), P_INT("stride"), P_STR("dtype") };
static const RadOperandSpec oWinInput[] = { OPD("lat"), OPD("cond"), OPD("lat_cu"), OPD("win_lane"),
                                            OPD("pair"), OUT("y") };
static const RadParamSpec pWinPrep[] = { P_INT("M"), P_INT("n"), P_INT("stride"), P_INT("ov"),
                                         P_STR("dtype") };
static const RadOperandSpec oWinPrep[] = { OPD("flags"), OPD("t"), OPD("seed"), OPD("win_index"),
                                           OPD("win_lane"), OPD("pair"), OPD("lat_cu"),
                                           OPD("prev"), INOUT("noise"), INOUT("lat") };
static const RadParamSpec pWinFinish[] = { P_INT("M"), P_INT("n"), P_INT("n_cond"), P_INT("stride"),
                                           P_INT("ov"), P_STR("dtype") };
static const RadOperandSpec oWinFinish[] = { OPD("flags"), OPD("pair"), OPD("win_lane"),
                                             OPD("lat_cu"), OPD("cond"), INOUT("lat"),
                                             INOUT("prev"), INOUT("prev_cond") };
static const RadParamSpec pChanSplit[] = { P_INT("M"), P_INT("n"), P_INT("parts"), P_STR("dtype") };
static const RadParamSpec pInterleave[] = { P_INT("M"), P_STR("dtype") };
static const RadOperandSpec oInterleave2[] = { OPD("a"), OPD("b"), OUT("y") };
static const RadParamSpec pIota[] = { P_INT("M"), P_INT("start"), P_INT("step"),
                                      P_INTO("period") };
static const RadOperandSpec oIota[] = { OUT("y") };
static const RadParamSpec pCuRepeat[] = { P_INT("M"), P_INT("mult") };
static const RadOperandSpec oCuRepeat[] = { OPD("len"), OUT("y") };
static const RadParamSpec pIntAffine[] = { P_INT("M"), P_STR("src"), P_STR("mul"), P_STR("add") };
static const RadOperandSpec oIntAffine[] = { OPD("x"), OUT("y") };

/* ---- vocabulary edges --------------------------------------------------------------------- */
static const RadParamSpec pEmbed[]    = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                          P_STR("dtype"), P_INTO("vocab_offset"),
                                          P_F64O("wscale") };
static const RadParamSpec pEmbedM[]   = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                          P_STR("dtype"), P_INTO("vocab_offset"),
                                          P_F64O("wscale"), P_F64O("mult") };
static const RadOperandSpec oEmbed[]  = { OPD("tokens"), WGT("wte"), OUT("x") };
static const RadOperandSpec oEmbedQ[] = { OPD("tokens"), WGT("wte"), WGT_O("scale"), OUT("x"),
                                          OPD_O("ahead") };

static const RadParamSpec pLogits[]   = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                          P_STR("dtype") };
static const RadOperandSpec oLogits[] = { OPD("x"), WGT("lm_head"), OUT("logits") };

/* ---- sampling ----------------------------------------------------------------------------- */
static const RadParamSpec pMVocab[] = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]  = { P_INT("M"), P_INT("n_cand") };
/* As libref: the history stages take the rank's first vocabulary row (`vocab_off`), and DRY an
 * optional plane of breaker marks beside the history. */
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
static const RadParamSpec pLogprobs[]    = { P_INT("M"), P_INT("n_vocab"), P_INT("k"),
                                             P_INTO("vocab_off") };
static const RadOperandSpec oLogprobs[]  = { OPD("logits"), OPD("params"), OUT("stats") };
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
/* DFlash2's grouped convolution, as libref declares it. */
static const RadParamSpec pDflashConv[] = { P_INT("T"), P_INT("hidden_size"), P_INT("taps"),
                                            P_INT("group_size"), P_INT("NG"), P_INT("block_size"),
                                            P_STR("dtype") };
static const RadOperandSpec oDflashConv[] = { OPD("x"), OPD("delta"), WGT("base"), OUT("out") };

/* ================================================================== the schema table
 * Doc strings are libavx's own (schema_same does not compare them); which plugin's string a reader
 * sees depends on which fixed the schema first. libref's are the definition.
 */
static const RadOpSchema kSchemas[] = {
{ "rmsnorm", ARR(pRmsnorm), ARR(oRmsnorm),
  "y = x * rsqrt(mean(x^2) + eps) * (w + wadd); the sum of squares is a four-accumulator vector "
  "reduction, in a different order from libref's." },
{ "rmsnorm_p", ARR(pLayernorm), ARR(oUnary),
  "y = x * rsqrt(mean(x^2) + eps): rmsnorm with no gain, summed as rmsnorm sums." },
{ "rmsnorm_add", ARR(pRmsnormAdd), ARR(oRmsnormAdd),
  "residual_out = x + residual, then y = rmsnorm(residual_out) * w: the variance from the unrounded "
  "f32 sum, the multiply from residual_out after narrowing, as libref." },
{ "hc_enter", ARR(pHcWrite), ARR(oHcEnter),
  "h[s] = x for each of the hc streams -- the wide residual stream's first value." },
{ "mhc_pre", ARR(pMhcPre), ARR(oMhcPre), "Whole-residual RMS projection, sigmoid gates and Sinkhorn mixing." },
{ "mhc_pre_partial", ARR(pMhcPrePartial), ARR(oMhcPrePartial),
  "mhc_pre's projection over chunks [first, first+chunks) of `chunk` columns, zero-padded per chunk." },
{ "mhc_pre_finish", ARR(pMhcPreFinish), ARR(oMhcPreFinish),
  "mhc_pre from every chunk's partials: chunk sums, gates, Sinkhorn mixing, collapse." },
{ "mhc_post", ARR(pHcWrite), ARR(oMhcPost), "Source-to-destination manifold residual update." },
{ "mhc_contract", ARR(pHcWrite), ARR(oMhcContract), "Mean over manifold residual streams." },
{ "kda_update", ARR(pKdaUpdate), ARR(oKdaUpdate), "Bounded per-channel KDA with candidate-state rollback." },
{ "hc_read", ARR(pHcRead), ARR(oHcRead),
  "The read half of a gated residual: grouped norm, low-rank gate, stream mix, and optionally the "
  "block-scaled fp8 form of x and its Hadamard-rotated twin. See docs/OPS.md for the three facts "
  "a second implementation has to match (the mix is a MEAN, there is a separate / hc inside each "
  "gate argument, and hc_write adds into the UNNORMED h). `mix` names the matrices' stored "
  "format; a host kernel reads them as the dense tensors it is handed." },
{ "hc_write", ARR(pHcWrite), ARR(oHcWrite), "h[s] += inj[s] * y, into the unnormed stream." },
{ "hc_project", ARR(pHcProject), ARR(oHcProject),
  "The control vector's projection (llama.cpp's `--cvec-mode project`): h[s] -= scale * "
  "(h[s] . v) * v for each of the hc streams of a row whose flag is on. A zero flag is not read." },
{ "mtp_enter", ARR(pMtpEnter), ARR(oMtpEnter),
  "MTP head entry: two norms, a per-sub-stream fc and a broadcast fc. `fc` names the two "
  "matrices' stored format; a host kernel reads them as the dense tensors it is handed." },
{ "layernorm", ARR(pLayernorm), ARR(oLayernorm), "layer norm with gain and optional bias." },
{ "grid_embed", ARR(pGridEmbed), ARR(oGridEmbed),
  "x += a side x side position table resampled bilinearly to the row's grid." },
{ "grid_embed_aa", ARR(pGridEmbed), ARR(oGridEmbed),
  "x += the table resampled as torch's antialiased half-pixel bilinear, SigLIP2's." },
{ "add", ARR(pAdd), ARR(oBinary),
  "y = a + b; a `b` of exactly n elements against a taller output broadcasts along rows. With "
  "`div`, y = rd(rd(a + b) / div), bit for bit libref's." },
{ "mul", ARR(pMnD), ARR(oBinary), "y = a * b, with the same broadcast rule as `add`." },
{ "mul_weight", ARR(pMulWeight), ARR(oMulWeight),
  "y = x * w, `w` a weight of one element or one row, broadcast; f32 product rounded once." },
{ "act_mul", ARR(pActMul), ARR(oBinary),
  "y = act(a) * b, act silu / gelu / gelu_tanh, the activation rounded to y's dtype first." },
{ "seg_mean", ARR(pSegMean), ARR(oSegMean),
  "y[s] = the mean of x's rows in cu_seqlens segment s, summed in double, rounded once." },
{ "relu", ARR(pMnD), ARR(oUnary), "y = max(x, 0)." },
{ "clamp", ARR(pMnD), ARR(oClamp), "y = min(max(x, lo), hi), lo and hi one-element weights." },
{ "conv2d", ARR(pConv2d), ARR(oConv2d),
  "segment-wise 2-D convolution over (time, width), channels innermost, optional row mask." },
{ "conv1d_dw", ARR(pConv1dDw), ARR(oConv1dDw),
  "segment-wise causal depthwise convolution over time, an optional bias." },
{ "rel_sinusoid", ARR(pRelSin), ARR(oRelSin), "the sinusoidal relative-position table." },
{ "conv2d_unfold", ARR(pConv2dUnfold), ARR(oConv2dUnfold),
  "a convolution's receptive fields as rows, columns in the weight's order." },
{ "seg_sinusoid", ARR(pSegSin), ARR(oSegSin),
  "sinusoidal absolute positions restarting every `period` rows of a segment." },
{ "attn_local_rel", ARR(pAttnLocalRel), ARR(oAttnLocalRel),
  "local attention with a relative-position term and a softcap (Gemma 4's audio)." },
{ "pool_rows", ARR(pPoolRows), ARR(oUnary),
  "y[r] = the mean of x's rows [r*k, r*k + k), rounded to y's dtype, times mult, rounded again." },
{ "mla_attn", ARR(pMlaAttn), ARR(oMlaAttn),
  "Softmax over selected latent vectors used as both key and value; per-query page tables and visible lengths." },
{ "mla_project", ARR(pMlaProject), ARR(oGemm),
  "Per-head query absorption or value expansion through the shared [head,(K|V),latent] BF16 matrix." },
{ "dsa_score", ARR(pDsaScore), ARR(oDsaScore),
  "Per-head ReLU of FP8 dot products, then signed head gates and scales; invalid history is -inf." },
{ "gemm_nt_f32", ARR(pGemm), ARR(oGemm),
  "BF16 activations times F32 weights, F32 accumulation and output; preserves routing and index gate precision." },
{ "dsa_query", ARR(pMnD), ARR(oQuantFp8),
  "Normalized Hadamard, BF16 boundary and power-of-two E4M3 quantization per query head." },
{ "dsa_pool", ARR(pDsaPool), ARR(oDsaPool),
  "Channel-wise softmax pooling with learned position bias, BF16, normalized Hadamard, BF16, power-of-two FP8." },
{ "swiglu_clamped", ARR(pSwigluClamped), ARR(oGateUp),
  "silu(min(gate,limit))*clamp(up,-limit,limit); gate is the first half." },
{ "silu_mul", ARR(pMnD), ARR(oGateUp),
  "silu-gated product over a fused [M, 2n] gate_up: y = silu(gate) * up." },
{ "gelu_mul", ARR(pMnD), ARR(oGateUp),
  "gelu_mul over a fused [M, 2n] gate_up: y = gelu_tanh(gate) * up, gate the FIRST half, the tanh "
  "form (libref's act_gelu_tanh, std::tanh) rather than the exact-erf gelu." },
{ "gelu", ARR(pMnD), ARR(oUnary), "exact erf gelu." },
{ "silu", ARR(pMnD), ARR(oUnary), "silu." },
{ "sigmoid", ARR(pMnD), ARR(oUnary), "logistic sigmoid." },
{ "fill", ARR(pFill), ARR(oFill), "y = value over [M, n], rounded once to y's dtype (default 0)." },
{ "logit_softcap", ARR(pMnCap), ARR(oUnary), "cap * tanh(x / cap), the final-logit softcap." },
{ "scale", ARR(pMnD), ARR(oScale), "y = x * s, s a [1] weight broadcast over every element." },
{ "ple_gate", ARR(pPleGate), ARR(oPleGate),
  "PLE gate: three grouped norms, a dot, a signed sqrt." },
{ "ple_conv", ARR(pPleConv), ARR(oPleConv),
  "PLE's dilated depthwise conv, silu and residual add." },
{ "causal_conv1d", ARR(pCausalConv), ARR(oCausalConv),
  "depthwise causal conv with ple_conv's window, no activation and no residual (LFM2)." },
{ "ngram_ids", ARR(pNgramIds), ARR(oNgramIds), "hashed n-gram row ids for PLE." },
{ "gather_rows", ARR(pMnD), ARR(oGather), "y[i,:] = x[idx[i],:]." },
{ "scatter_rows", ARR(pMnD), ARR(oScatterRows),
  "x[idx[i],:] = v[i,:], the mirror of gather_rows: `x` is INOUT and the write is partial, a "
  "negative index writes nothing, and indices are distinct." },
{ "cast", ARR(pCast), ARR(oUnary),
  "dtype conversion; narrowing is the ABI's round-to-nearest-even, bit for bit." },
{ "softmax", ARR(pMnD), ARR(oUnary), "row-wise softmax, max-subtracted." },
{ "quant_act_i8", ARR(pQuant), ARR(oQuant),
  "symmetric int8, one f32 scale a row, codes in the int8 WMMA's step order; `asum` gets the "
  "per-`group` code sums (libref's description)." },
{ "quant_act_i8g", ARR(pQuant), ARR(oQuantFp8),
  "symmetric int8, one f32 scale per `group` contiguous elements, row-major." },
{ "quant_act_fp8", ARR(pQuant), ARR(oQuantFp8), "symmetric block-scaled E4M3 quantiser." },
{ "quant_cols_fp8", ARR(pQuantCols), ARR(oQuantCols), "E4M3 with one scale per column." },
{ "had_quant_act_fp8", ARR(pQuant), ARR(oQuantFp8),
  "width-`group` Hadamard, then the same symmetric block-scaled E4M3 quantiser. No normalisation: "
  "the 1/group is folded into the stored weight scale at convert (docs/OPS.md)." },
{ "had_quant_act_i8g", ARR(pQuant), ARR(oQuantFp8),
  "width-`group` Hadamard, then quant_act_i8g's symmetric int8 per group. No normalisation: the "
  "1/group is the i8a8h weight's stored scale (docs/OPS.md)." },
{ "gram_accum", ARR(pQuant), ARR(oGram),
  "h += x^T x over a dequantised E4M3 activation. `h` is [n, n] and INOUT." },
{ "givens_dense_quant", ARR(pGivensDense), ARR(oGivensDense),
  "ParoQuant's activation quantiser with the rotation as a dense per-group matrix: y = T (x / "
  "rscale) for each of `parts` rotated copies, then the block-scaled quantiser `out` names "
  "(\"i8\": amax / 127, round to nearest; \"fp8\": E4M3, amax / 448)." },
{ "givens_rotate", ARR(pGivensRotate), ARR(oGivensRotate),
  "ParoQuant's rotation without a quantiser: x / rscale rotated by `stages` rounds of Givens "
  "pairs, `parts` copies side by side, in y's own dtype." },
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
  "C = A B^T, f32 accumulate, cache-blocked with a register-blocked microkernel; it sums in a "
  "different order from libref's straight loop." },
{ "gemm_nt_bias", ARR(pGemmBias), ARR(oGemmBias),
  "C = act(A B^T + bias) + res, each step rounded to C's dtype." },
{ "gemm_nt_q", ARR(pGemmQ), ARR(oGemmQ),
  "C = A B^T with group-scaled quantised operands. `a_sum` and `b_ref` are libr4d's asymmetric and "
  "mxfp4 forms and are ignored here, as they are in libref." },
{ "gemm_nt_q_bias", ARR(pGemmQBias), ARR(oGemmQBias), "quantised C = A B^T + bias." },
{ "lora_shrink", ARR(pLoraShrink), ARR(oLoraShrink),
  "t[m, j] = sum_k x(m, k) * lora_a[slot[m], j, k]; a base row (slot outside [0, S)) is not "
  "written." },
{ "lora_expand", ARR(pLoraExpand), ARR(oLoraExpand),
  "y[m, n] += scale[slot[m], n] * sum_j t[m, j] * lora_b[slot[m], n, j]; a base row is neither "
  "read nor written." },
{ "lora_expand_mm", ARR(pLoraExpand), ARR(oLoraExpand),
  "lora_expand with t rounded to bf16, the sum over j in any order; a base row is neither read nor "
  "written." },
{ "gemm_nt_q_gated", ARR(pGemmQGated), ARR(oGemmQGated),
  "gemm_nt_q over a [N, K] gate_up plane with gated_quant_fp8's pass in the epilogue, byte for "
  "byte the pair. `N` is the weight's rows, twice the fused width; the output is [M, N/2]." },
{ "attn_paged", ARR(pAttnPaged), ARR(oAttnPaged),
  "paged causal / sliding-window attention, any query length, online-softmax so one pass over the "
  "cache suffices." },
{ "attn_dense", ARR(pAttnDense), ARR(oAttnDense),
  "dense varlen attention over cu_seqlens; a non-causal window may reach further right "
  "(`window_right`)." },
{ "attn_cross", ARR(pAttnCross), ARR(oAttnCross),
  "attn_dense with each entry's keys a run of their own (cu_k); causal counts from each run's "
  "start." },
{ "attn_na3d", ARR(pAttnNa3d), ARR(oAttnNa3d),
  "3-D neighborhood attention over a volume, libref's window and arithmetic." },
{ "kv_store", ARR(pKvStore), ARR(oKvStore), "scatter k and v into the paged cache." },
{ "rope_table", ARR(pRopeTable), ARR(oRopeTable), "cos/sin at each position's own row." },
{ "qsa_block_key", ARR(pQsaBlockKey), ARR(oQsaBlockKey),
  "QSA's compressed key: the mean of a block's `ratio` rows rounded to k's dtype, RMS-normed with "
  "gain (wadd + w), rotated at the block's position with `rope`'s ladder and YaRN; libref's "
  "arithmetic with the angles from one table a block." },
{ "qsa_score", ARR(pQsaScore), ARR(oQsaScore),
  "score[m][b] = (sum over heads of relu(q[m][h] . bk[bt[seq(m)][b]])) / sqrt(n) for the sequence's "
  "first min(nc, width) blocks, a negative page id scoring -inf; the dots in vector order, the "
  "heads summed in order. Columns past the bound are not written." },
{ "qsa_select", ARR(pQsaSelect), ARR(oQsaSelect),
  "the topk highest-scoring complete blocks as page ids, libref's order exactly: the blocks above "
  "the threshold in block order, then its ties in block order, then the partial page, then -1 to "
  "topk + 1; a radix select, linear in the blocks." },
{ "qsa_tail_store", ARR(pQsaTailStore), ARR(oQsaTailStore),
  "state[sidx[s][0]][pos[t] % ring] = k[t] for the last `ring` tokens of each sequence; the other "
  "rows of the tail keep what they held." },
{ "qsa_work", ARR(pQsaWork), ARR(oQsaWork),
  "the blocks completing in this batch: each block's `ratio` raw keys staged contiguously, its "
  "block-key page and first-token position; slots past the count get page -1. nc is each "
  "sequence's complete-block count." },
{ "rope", ARR(pRope), ARR(oRope), "rotary embedding in place." },
{ "qk_norm_rope", ARR(pQkNormRope), ARR(oQkNormRope), "per-head QK norm then rotary embedding." },
{ "scale_rows", ARR(pScaleRows), ARR(oScaleRows),
  "one scalar per row, optionally through an activation, broadcast across the row." },
{ "router_sigmoid_topk", ARR(pRouterSigmoid), ARR(oRouterSigmoid),
  "Single-group sigmoid routing; correction bias selects experts, original scores weight them." },
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
{ "pair_argmax", ARR(pPairArgmax), ARR(oPairArgmax),
  "the global top-1 of a row of gathered (id, value) pairs, one id a row." },
{ "logit_rerank", ARR(pLogitRerank), ARR(oLogitRerank),
  "R candidates a row rescored exactly against the full-precision head." },
{ "gdn_conv_prep", ARR(pConvPrep), ARR(oConvPrep),
  "gdn prefill preamble: conv, split, l2 norm, gate cumsum, beta." },
{ "gdn_conv_update", ARR(pConvUpdate), ARR(oConvUpdate),
  "gdn decode conv over the rolling speculative window." },
{ "gdn_kkt_solve", ARR(pKkt), ARR(oKkt), "gdn chunk preamble: the triangular inverse." },
{ "gdn_chunk_scan", ARR(pChunkScan), ARR(oChunkScan), "gdn chunked scan." },
{ "gdn_recurrent_update", ARR(pRecurrent), ARR(oRecurrent),
  "gdn decode recurrence against the paged state cache. `state_form` names the reading: "
  "\"per_candidate\" (one state per candidate token) or libr4d's two-slot \"anchor\" (one "
  "committed state and its tokens' replay factors), both as libref reads them; any other form is "
  "declined by name." },
{ "gdn_conv_recurrent_update", ARR(pConvRecurrent), ARR(oConvRecurrent),
  "gdn_conv_update followed by gdn_recurrent_update, with q, k and v never written: the operands "
  "are the convolution's (`conv_idx` is its state_idx) and then the recurrence's without those "
  "three." },
{ "gdn_gated_rmsnorm", ARR(pGatedNorm), ARR(oGatedNorm), "gated RMS norm over a head." },
{ "all_reduce", ARR(pAllReduce), ARR(oAllReduce),
  "sum across ranks, f32, ascending rank order." },
{ "all_gather", ARR(pAllGather), ARR(oAllGather), "gather every rank's shard." },
{ "ln_mod_quant_fp8", ARR(pLnModQ), ARR(oLnModQ),
  "gate_add (with `gated` 1), ln_mod and quant_act_fp8 in one pass, each value rounded as its own "
  "store would round it; z written when bound." },
{ "ln_mod", ARR(pLnMod), ARR(oLnMod),
  "y = bf16(bf16(LN(x)) * bf16(1 + s[seq])): a gainless layer norm and the row's entry's "
  "modulation row of `s` ([n_seq, >= n] at its own pitch)." },
{ "table_add", ARR(pTableAdd), ARR(oTableAdd),
  "y[r, l n + c] = x[r, c] + t[l, c], f32 sum rounded once." },
{ "chan_stats", ARR(pChanStats), ARR(oChanStats),
  "per-channel statistics held as weights: denorm rd(rd(x std) + mean), norm rd(rd(x - mean) / "
  "std)." },
{ "mix_rows", ARR(pMixRows), ARR(oMixRows),
  "y[r, c] = sum_i w[i] x[r k + i, c], f32 in ascending i, rounded once." },
{ "gate_add", ARR(pGateAdd), ARR(oGateAdd),
  "x = bf16(x + bf16(bf16(tanh(g[seq])) * y)), or with `act` none the gate g itself, bit for bit "
  "libref's." },
{ "timestep_embed", ARR(pTimestep), ARR(oTimestep),
  "the sinusoidal timestep embedding [cos | sin], libref's libm calls; `round` 1 takes t through "
  "bf16 at scale and back first." },
{ "gather_rows2", ARR(pMnD), ARR(oGather2),
  "y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]], zeros past either source." },
{ "lane_gather", ARR(pLane), ARR(oLaneGather),
  "y[cu[e] + k] = src[lane[e] * stride + k]; every entry checked before a row is written." },
{ "lane_store", ARR(pLane), ARR(oLaneStore),
  "dst[l * stride + k] = src[k] for every lane l listed." },
{ "randn_lanes", ARR(pLane), ARR(oRandnLanes),
  "an image's first step: its lane's rows become its seed's normal stream (abi/rad_philox.h), "
  "bit for bit libref's." },
{ "flow_euler", ARR(pLane), ARR(oFlowEuler),
  "one flow-matching Euler step per entry that moves a lane, with guidance, the preview and the "
  "repaint's kept region; libref's roundings bit for bit." },
{ "kd_input", ARR(pLane), ARR(oKdInput),
  "a k-diffusion sampler's model input per entry that moves a lane: a first step's noise times "
  "sqrt(1 + sigma^2) into the lane, then rd(x / sqrt(sigma^2 + 1)) at the entry's and its pair's "
  "rows; libref's arithmetic bit for bit." },
{ "kd_step", ARR(pKdStep), ARR(oKdStep),
  "one k-diffusion sampler step per entry that moves a lane: the guided clean estimate from an "
  "eps or v output, then Euler, Euler ancestral or DPM++ 2M; libref's arithmetic bit for bit." },
{ "chan_affine", ARR(pChanAffine), ARR(oUnary),
  "a per-channel affine from two lists of n bf16 values, `fma` or `inv`; `round` 0 keeps the "
  "lists and the arithmetic in f32, rounded once." },
{ "conv2d_nhwc", ARR(pConvNhwc), ARR(oConvNhwc),
  "an image convolution, channels innermost, as GEMMs over bands of patch rows; `up` 2 reads the "
  "input upsampled nearest-neighbour, `pad` same or down." },
{ "im2col_nhwc", ARR(pIm2col), ARR(oUnary),
  "conv2d_nhwc's receptive fields as rows, y [Hout * Wout, cin * k * k], column (ci * k + i) * k "
  "+ j the value conv2d_nhwc multiplies weight[o, ci, i, j] by; libref's copies bit for bit." },
{ "l2norm_chan", ARR(pL2Norm), ARR(oL2Norm),
  "each pixel's channels to unit L2, times sqrt(n), times gamma (three roundings), then `act`." },
{ "d2s_add", ARR(pSpaceDepth), ARR(oSpaceDepth),
  "y [2H, 2W * cout] += x [H, W * cin] repeated and dealt into 2x2 blocks (DupUp3D, one frame)." },
{ "s2d_avg_add", ARR(pSpaceDepth), ARR(oSpaceDepth),
  "y [H, W * cout] += the 2x2 blocks of x dealt into channels and averaged in groups (AvgDown3D, "
  "one frame)." },
{ "to_rgba8", ARR(pMnD), ARR(oUnary),
  "decoded pixels in [-1, 1] to bytes as diffusers' postprocess makes them, u8 out." },
{ "gelu_tanh", ARR(pMnD), ARR(oUnary), "the tanh approximation of gelu." },
{ "gelu_quick", ARR(pMnD), ARR(oUnary),
  "CLIP's quick_gelu, three roundings: a = rd(1.702 x), s = rd(sigmoid(a)), y = rd(x s)." },
{ "scatter_add_rows", ARR(pMnD), ARR(oScatterRows),
  "x[idx[i]] += v[i], rounded to x's dtype; a negative index adds nothing." },
{ "attn_prefix_self", ARR(pAttnPS), ARR(oAttnPS),
  "attention over `seqused[s]` prefix positions in the paged cache and the step's own rows, "
  "flash-attention tiles; `qk` \"i8\" is the int8 score form (docs/OPS.md)." },
{ "rms_mod", ARR(pRmsMod), ARR(oRmsMod),
  "the gated residual (no tanh), then an RMS norm with a gain times bf16(1 + scale row) plus the "
  "shift row, `s` holding [shift | scale]; libref's roundings." },
{ "rms_mod_quant_fp8", ARR(pRmsModQ), ARR(oRmsModQ),
  "rms_mod with its norm chain in bf16, then the E4M3 codes and scales of that row." },
{ "flow_x0_step", ARR(pMnD), ARR(oFlowX0),
  "x = r x + (1 - r)(x + (1 - t) v), r = sigma_next / sigma, each step one f32 rounding, bit for "
  "bit libref's." },
{ "patch_unfold", ARR(pPatchUnfold), ARR(oPatchUnfold),
  "patch rows to denormalised voxel rows, f32(f32(x * mul) + add), bit for bit libref's." },
{ "vtile_gather", ARR(pVtileGather), ARR(oVtileGather),
  "a video decoder's tiles out of the latent's voxel rows, then the register rows and zeros." },
{ "vtile_stitch", ARR(pVtileStitch), ARR(oVtileStitch),
  "decoded tiles blended into frames, the chunk overlap blended and kept, RGB bytes out; bit for "
  "bit libref's." },
{ "grid_pos3", ARR(pGridPos3), ARR(oGridPos3),
  "a video decoder's rotary positions over its tiles, [3, >= M] at the operand's row pitch, bit "
  "for bit libref's." },
{ "vsa_select", ARR(pVsa), ARR(oVsaSelect),
  "video sparse attention's tile selection over libref's tiles: the k video tiles of largest "
  "pooled score per video query tile, ascending, bit for bit libref's; with `coarse`, the softmax "
  "over every tile's score times the tiles' means of v." },
{ "attn_vsa", ARR(pAttnVsa), ARR(oAttnVsa),
  "video sparse attention over libref's tiles: prefix queries dense, video queries the prefix "
  "and their `sel` tiles, blocked online softmax; the gated coarse term on the output rounded "
  "to its dtype." },
{ "conv3d_ndhwc", ARR(pConv3d), ARR(oConv3d),
  "a 3-D convolution over voxel rows, `same` zero padding, PyTorch's [cout, cin, kt, k, k] "
  "weight, as GEMMs over bands of patch rows; with `res` (possibly y) rd(rd(acc + b) + res)." },
{ "group_norm", ARR(pGroupNorm), ARR(oGroupNorm),
  "a group norm over all rows, its statistics in f64 over fixed row blocks combined in order, "
  "then the optional [scale | shift] row and silu, one rounding; stats[0, 2 groups) the mean "
  "and rstd." },
{ "tconv_dw", ARR(pTconvDw), ARR(oTconvDw),
  "a depthwise convolution along a volume's frames, PyTorch's [n, 1, k, 1, 1] weight, zero "
  "frames outside the volume." },
{ "resize_bilinear", ARR(pMnD), ARR(oResize),
  "each frame of a volume resized as torch's align_corners=False bilinear at a size, bit for bit "
  "libref's." },
{ "patch_fold", ARR(pPatchUnfold), ARR(oPatchUnfold),
  "voxel rows to normalised patch rows, f32(f32(x * mul) + add), bit for bit libref's." },
{ "flow_renoise", ARR(pFlowRenoise), ARR(oFlowRenoise),
  "x = f32(f32(1 - s) x) + f32(s z), z the (seed, stream) normal stream at k n + c, bit for bit "
  "libref's." },
{ "rope_split", ARR(pRopeSplit), ARR(oRopeSplit),
  "LTX-2's split rotary over fractional coordinates, libref's arithmetic element by element." },
{ "head_gate", ARR(pHeadGate), ARR(oHeadGate),
  "x = rd(x rd(2 rd(sigmoid(g)))) per head, libref's." },
{ "ltx_guide", ARR(pLtxGuide), ARR(oLtxGuide),
  "LTX-2's guided sampler step in f32, libref's arithmetic with the deviations in f64." },
{ "pixel_norm", ARR(pPixelNorm), ARR(oPixelNorm),
  "an RMS norm over each row as bf16 tensors compute it, every step rounded; rsqrt or div form, "
  "optionally silu, `ileave` interleaving the output; libref's." },
{ "d2s3d", ARR(pD2s3d), ARR(oD2s3d), "depth to space over voxel rows, libref's copy." },
{ "s2d3d", ARR(pS2d3d), ARR(oS2d3d),
  "space to depth over voxel rows, a copy or the mean of each group, libref's." },
{ "pix_patch", ARR(pPixPatch), ARR(oPixPatch), "an image to patch rows, libref's copy." },
{ "patch_pix8", ARR(pPixPatch), ARR(oPixPatch), "patch rows to RGB bytes, libref's rule." },
{ "vol_index", ARR(pVolIndex), ARR(oVolIndex), "a volume's voxel coordinates, libref's." },
{ "tile_grid", ARR(pTileGrid), ARR(oTileGrid), "a tiled decode's boxes, libref's cut." },
{ "tile_crop", ARR(pTileCrop), ARR(oTileCrop), "a tile out of a volume, libref's copy." },
{ "tile_stitch", ARR(pTileStitch), ARR(oTileStitch),
  "decoded tiles blended to RGB bytes, libref's blends and rounding." },
{ "stft_logmel", ARR(pStftMel), ARR(oStftMel),
  "each channel's causal log-mel spectrogram, libref's roundings." },
{ "bwe_up", ARR(pBweUp), ARR(oBweUp),
  "a Hann-windowed sinc resampler plus a residual, clamped, libref's roundings." },
{ "conv1d", ARR(pConv1d), ARR(oConv1d),
  "a 1-D convolution over the sequences `cu_seqlens` bounds, channels innermost, PyTorch's "
  "[cout, cin, k] weight, dilation `dil`, centred or (`causal`) all on the left, zeros outside "
  "each sequence; `res` added after the bias, then `act` none, tanh or clamp. GEMMs over bands "
  "of patch rows." },
{ "conv_t1d", ARR(pConvT1d), ARR(oConvT1d),
  "a transposed 1-D convolution of kernel `k` >= stride (2 stride by default), padding `pad` "
  "((k - stride) / 2, or 0 the causal form), each sequence growing `stride` times; one GEMM a "
  "phase." },
{ "snake", ARR(pSnake), ARR(oSnake),
  "x + sin(alpha x)^2 / (alpha + 1e-9) per channel, or SnakeBeta's x + sin(alpha x)^2 / "
  "(beta + 1e-9) with `beta`, both read as logarithms with `log` 1; a polynomial sine." },
{ "snake_aa", ARR(pSnakeAa), ARR(oSnakeAa),
  "BigVGAN's alias-free SnakeBeta: replicate-padded upsampling by `f_up`, SnakeBeta, "
  "replicate-padded filtering by `f_down` at stride `ratio`; a polynomial sine." },
{ "cfg_topk_sample", ARR(pCfgTopk), ARR(oCfgTopk),
  "one guided top-k pick a lane by inverse CDF over rad_philox_uniform_s, with a suppressed id "
  "range, a repetition penalty over each lane's history and per-lane guidance; bit for bit "
  "libref's." },
{ "rvq_embed", ARR(pRvqEmbed), ARR(oRvqEmbed),
  "a frame's embedding from code-major residual codes, libref's roundings (`round` 1: one)." },
{ "frame_mix_store", ARR(pFrameMix), ARR(oFrameMix),
  "a frame's softmax-weighted mix of hidden planes, stored into its lane, bit for bit libref's." },
{ "end_status", ARR(pEndStatus), ARR(oEndStatus),
  "status[lane[i]] = ids[rep i] == end ? -1 : ids[rep i]." },
{ "frame_store", ARR(pFrameStore), ARR(oFrameStore),
  "a frame's codes into its lane's history, code-major rows over the lane's frames." },
{ "resample_nearest", ARR(pMnD), ARR(oUnary),
  "rows resampled nearest-neighbour, the row counts read off the operands." },
{ "fourier_embed", ARR(pFourier), ARR(oFourier),
  "[cos(2 pi t w) | sin(2 pi t w)], libref's libm calls and roundings." },
{ "win_input", ARR(pWinInput), ARR(oWinInput),
  "a denoiser's input [lat | zeros | cond] from each window entry's lane." },
{ "win_prep", ARR(pWinPrep), ARR(oWinPrep),
  "a window step's noise and overlap binding, bit for bit libref's." },
{ "win_finish", ARR(pWinFinish), ARR(oWinFinish),
  "a window's end and the overlap it hands on." },
{ "chan_split", ARR(pChanSplit), ARR(oUnary), "channel groups to sequences." },
{ "interleave2", ARR(pInterleave), ARR(oInterleave2), "two streams interleaved, y [rows, 2]." },
{ "iota", ARR(pIota), ARR(oIota), "y[i] = start + i * step." },
{ "cu_repeat", ARR(pCuRepeat), ARR(oCuRepeat), "y[j] = j * len[0] * mult." },
{ "int_affine", ARR(pIntAffine), ARR(oIntAffine),
  "y[j] = (src[j] < 0 ? 0 : x[src[j]]) * mul[j] + add[j] over three integer lists." },
{ "embed_lookup", ARR(pEmbedM), ARR(oEmbed),
  "embedding gather, vocab-parallel with an offset; `mult` multiplies every row (rounded once)." },
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
  "top 24 bits as a float in [0,1), so a host reference, a device kernel and a rerun draw the "
  "same stream." },
{ "sample_merge_topk", ARR(pMCand), ARR(oMerge), "the vocab-parallel candidate merge." },
{ "sample_argmax", ARR(pMVocab), ARR(oArgmax), "greedy, ties to the lower token id." },
{ "sample_logprobs", ARR(pLogprobs), ARR(oLogprobs),
  "the raw logits' max, sum of exp(l - max) and k largest (id, logit) pairs, for a row with "
  "RAD_SP_LOGPROBS." },
{ "dflash_conv", ARR(pDflashConv), ARR(oDflashConv),
  "DFlash2's grouped dynamic depthwise convolution: out[t][h] = sum over taps j, j == 0 or "
  "t % block_size >= j, of (base[j][h] + delta[t][j][h / group_size]) * x[t - j][h]." },
{ "dflash_select", ARR(pDflashSelect), ARR(oDflashSelect),
  "a greedy walk over a candidate lattice: score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * "
  "hp[l][r] * succ[cand[l, c]][r], pid(0) the anchor and pid(l) the walk's own previous pick. "
  "`hp` absent is a plane of ones." },
};

/* ================================================================== the kernel table
 *
 * Every row is domain HOST at priority 0 and unconstrained, except `block_size` on attn_paged and
 * kv_store (as in libref): the constraint admits any value but requires the key to be present,
 * since the cache cannot be walked without the block edge.
 *
 * Every row's shape is "any", since every extent is read from the operands (avx_common.h). The
 * ISA level is deliberately not a constraint: rows are identical on every CPU and only the code
 * underneath changes. */
#define SHAPE_ANY "any: every extent is read off the operands, strides included"

#define ROW(nm, opname, fam, what, dts, cons, ncons, fn)                                          \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                  \
      cons, ncons, nullptr, 0, nullptr, nullptr, nullptr, fn, nullptr, nullptr, nullptr,          \
      nullptr, nullptr, nullptr }

#define SIMPLE(nm, opname, fam, what, dts, fn) ROW(nm, opname, fam, what, dts, nullptr, 0, fn)

/* ---- the codebook experts' layout hooks, on moe_gemm_q and moe_gemm_q_mod4
 *
 * A codebook expert's codes view (`codes,table`) and two-level scale view (`scale,scale.1`) are
 * stored joined (abi/rad_planes.h); a view of one plane is stored as it is. The selections
 * accepted are libref's (libref/ref_moe.cpp: cb_check), so a weight libref refuses at declare is
 * refused here too, and both libraries store the same bytes under one tag. */
struct CbForm { int bits; int64_t group; uint32_t sdt; bool level; };
static CbForm cb_form(const RadParam* p, int n_p) {
    const RadParam* d = rad_param_find(p, n_p, "dtype");
    const char* s = d && d->kind == RAD_P_STR ? d->sval : nullptr;
    if (s && !std::strcmp(s, "w4nla8h"))   return CbForm{ 4, 128, RAD_BF16, false };
    if (s && !std::strcmp(s, "w4nl64a8h")) return CbForm{ 4, 64, RAD_F8E4M3, true };
    if (s && !std::strcmp(s, "w4nl32a8h")) return CbForm{ 4, 32, RAD_F8E4M3, true };
    if (s && !std::strcmp(s, "w5nl64a8h")) return CbForm{ 5, 64, RAD_F8E4M3, true };
    return CbForm{ 0, 0, 0, false };
}
static int cb_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                    const int* sel, int n_planes) {
    const CbForm f = cb_form(p, n_p);
    const bool codes = operand == 2 || operand == 6 || operand == 8 || operand == 10;
    const bool scale = operand == 3 || operand == 7 || operand == 9 || operand == 11;
    if (!f.bits || (!codes && !scale) || !enc || !sel || n_planes != 2 ||
        std::strcmp(enc->transform, "fwht128") != 0)
        return RAD_E_DTYPE;
    for (int i = 0; i < 2; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return RAD_E_SHAPE;
    const RadEncPlane& a = enc->plane[sel[0]];
    const RadEncPlane& b = enc->plane[sel[1]];
    if (codes)
        return !std::strcmp(a.role, "codes") && a.kind == RAD_PLANE_TILED &&
               a.dtype == (f.bits == 5 ? (uint32_t)RAD_U5 : (uint32_t)RAD_U4) &&
               a.block[0] == 1 && a.block[1] == 1 && !std::strcmp(b.role, "table") &&
               b.kind == RAD_PLANE_TABLE && b.dtype == RAD_F32 && b.extent[0] == 1 &&
               b.extent[1] == ((int64_t)1 << f.bits)
                   ? RAD_OK : RAD_E_DTYPE;
    return f.level && !std::strcmp(a.role, "scale") && a.kind == RAD_PLANE_TILED &&
           a.dtype == f.sdt && a.block[0] == 1 && a.block[1] == f.group &&
           !std::strcmp(b.role, "scale.1") && b.kind == RAD_PLANE_TILED && b.dtype == RAD_F32 &&
           b.block[0] == 0 && b.block[1] == 0
               ? RAD_OK : RAD_E_DTYPE;
}
static int avx_moe_cb_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                             const int* sel, const RadTensor* planes, int n_planes,
                             RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const int rc = cb_check(p, n_p, operand, enc, sel, n_planes);
    return rc < 0 ? rc : rad_planes_layout(enc, sel, planes, n_planes, out);
}
static int avx_moe_cb_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n_planes, void* dst,
                               int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const int rc = cb_check(p, n_p, operand, enc, sel, n_planes);
    return rc < 0 ? rc : rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}
static int avx_moe_cb_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const void* src, int64_t src_bytes,
                                 const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const int rc = cb_check(p, n_p, operand, enc, sel, n_planes);
    return rc < 0 ? rc : rad_planes_unrelayout(enc, sel, src, src_bytes, planes, n_planes);
}

#define ROW_CB(nm, opname, fam, what, dts, fn)                                                    \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                  \
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, fn, nullptr, avx_moe_cb_layout,          \
      avx_moe_cb_relayout, nullptr, avx_moe_cb_unrelayout, nullptr }

/* A row whose weight operands may be joined over several planes (avx_planes.h): libref's hooks for
 * the op, restated, so both libraries store and read one form. */
#define ROWL(nm, opname, fam, what, dts, fn, lay)                                                 \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                  \
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, fn, nullptr, avxp::lay,                  \
      avxp::avx_relayout_planes<avxp::lay>, nullptr, avxp::avx_unrelayout_planes<avxp::lay>,     \
      nullptr }

#define DT_FLOAT "bf16 f16 f32 fp8_e4m3 fp8_e5m2 i8 i32, chosen per operand from its own dtype"
#define DT_ANY   "any dtype rad_types.h names, per operand"
#define DT_FLOAT_CB DT_FLOAT "; or u4 / u5 codes into an f32 table, joined (w4nl* / w5nl*)"

/* Rows launch through the baseline-compiled thunks, which are valid before rad_plugin_open
 * (avx_dispatch.cpp). */
#define L(nm) avx_thunk_##nm
#define AVX_THUNK_DECL(nm) extern "C" int avx_thunk_##nm(const RadArgs*, RadStream);
AVX_OP_LIST(AVX_THUNK_DECL)
#undef AVX_THUNK_DECL

static const RadKernelInfo kKernels[] = {
SIMPLE("avx_rmsnorm", "rmsnorm", "norm", "root-mean-square norm with a gain", DT_FLOAT, L(rmsnorm)),
SIMPLE("avx_rmsnorm_p", "rmsnorm_p", "norm", "weightless RMS norm (no gain)", DT_FLOAT,
       L(rmsnorm_p)),
SIMPLE("avx_rmsnorm_add", "rmsnorm_add", "norm", "residual add then RMS norm", DT_FLOAT,
       L(rmsnorm_add)),
SIMPLE("avx_hc_enter", "hc_enter", "norm", "gated residual entry: hc copies of x", DT_FLOAT,
       L(hc_enter)),
SIMPLE("avx_mhc_pre", "mhc_pre", "norm", "manifold gates and Sinkhorn mixing", DT_FLOAT, L(mhc_pre)),
SIMPLE("avx_mhc_pre_partial", "mhc_pre_partial", "norm", "manifold projection partials", DT_FLOAT, L(mhc_pre_partial)),
SIMPLE("avx_mhc_pre_finish", "mhc_pre_finish", "norm", "manifold gates from projection partials", DT_FLOAT, L(mhc_pre_finish)),
SIMPLE("avx_mhc_post", "mhc_post", "norm", "manifold residual update", DT_FLOAT, L(mhc_post)),
SIMPLE("avx_mhc_contract", "mhc_contract", "norm", "mean of residual streams", DT_FLOAT, L(mhc_contract)),
SIMPLE("avx_kda_update", "kda_update", "gdn", "bounded KDA and candidate-state rollback", DT_FLOAT, L(kda_update)),
ROWL("avx_hc_read", "hc_read", "norm",
     "gated residual read: grouped norm, low-rank gate, stream mix",
     DT_FLOAT ", or mixing matrices of E4M3 / int8 codes with a scale plane, joined", L(hc_read),
     avx_layout_hc),
SIMPLE("avx_hc_write", "hc_write", "norm", "gated residual write: h[s] += inj[s] * y", DT_FLOAT,
       L(hc_write)),
SIMPLE("avx_hc_project", "hc_project", "norm",
       "control vector: h[s] -= scale * (h[s] . v) * v where the row's flag is on", DT_FLOAT,
       L(hc_project)),
ROWL("avx_mtp_enter", "mtp_enter", "norm",
     "MTP head entry: two norms, a per-sub-stream fc and a broadcast fc",
     DT_FLOAT ", or fc matrices of E4M3 / int8 codes with a scale plane, joined", L(mtp_enter),
     avx_layout_mtp),
SIMPLE("avx_layernorm", "layernorm", "norm", "layer norm with gain and bias", DT_FLOAT,
       L(layernorm)),
SIMPLE("avx_grid_embed", "grid_embed", "elem",
       "bilinear resample of a learned position grid, added in", DT_FLOAT, L(grid_embed)),
SIMPLE("avx_grid_embed_aa", "grid_embed_aa", "elem",
       "antialiased half-pixel bilinear resample of a position grid, added in", DT_FLOAT,
       L(grid_embed_aa)),
SIMPLE("avx_add", "add", "elem", "elementwise sum", DT_FLOAT, L(add)),
SIMPLE("avx_mul", "mul", "elem", "elementwise product, broadcasting b", DT_FLOAT, L(mul)),
SIMPLE("avx_act_mul", "act_mul", "elem", "act(a) * b over two operands", DT_FLOAT, L(act_mul)),
SIMPLE("avx_mul_weight", "mul_weight", "elem", "x * a scalar or one-row weight", DT_FLOAT,
       L(mul_weight)),
SIMPLE("avx_seg_mean", "seg_mean", "elem", "mean of each cu_seqlens segment's rows", DT_FLOAT,
       L(seg_mean)),
SIMPLE("avx_pool_rows", "pool_rows", "elem", "mean of every k consecutive rows, then a multiplier",
       DT_FLOAT, L(pool_rows)),
SIMPLE("avx_relu", "relu", "elem", "max(x, 0)", DT_FLOAT, L(relu)),
SIMPLE("avx_clamp", "clamp", "elem", "x clamped to two weight scalars", DT_FLOAT, L(clamp)),
SIMPLE("avx_conv2d", "conv2d", "elem", "segment-wise 2-D convolution, channels innermost",
       DT_FLOAT, L(conv2d)),
SIMPLE("avx_conv1d_dw", "conv1d_dw", "elem", "segment-wise causal depthwise convolution",
       DT_FLOAT, L(conv1d_dw)),
SIMPLE("avx_rel_sinusoid", "rel_sinusoid", "elem", "sinusoidal relative-position table",
       DT_FLOAT, L(rel_sinusoid)),
SIMPLE("avx_conv2d_unfold", "conv2d_unfold", "elem",
       "a convolution's receptive fields as rows, for a GEMM against its weight",
       DT_FLOAT, L(conv2d_unfold)),
SIMPLE("avx_seg_sinusoid", "seg_sinusoid", "elem",
       "sinusoidal absolute positions, restarting every `period` rows of a segment",
       DT_FLOAT, L(seg_sinusoid)),
SIMPLE("avx_attn_local_rel", "attn_local_rel", "attn",
       "local attention with a relative-position term and a softcap", DT_FLOAT,
       L(attn_local_rel)),
SIMPLE("avx_mla_attn", "mla_attn", "attn", "absorbed latent MQA attention", DT_FLOAT,
       L(mla_attn)),
SIMPLE("avx_dsa_score", "dsa_score", "attn", "weighted FP8 index scoring", DT_FLOAT,
       L(dsa_score)),
SIMPLE("avx_dsa_pool", "dsa_pool", "attn", "learned compressed FP8 index keys", DT_FLOAT,
       L(dsa_pool)),
SIMPLE("avx_dsa_query", "dsa_query", "attn", "normalized FP8 index queries", DT_FLOAT,
       L(dsa_query)),
SIMPLE("avx_gemm_nt_f32", "gemm_nt_f32", "gemm", "F32 routing projection", DT_FLOAT,
       L(gemm_nt_f32)),
SIMPLE("avx_mla_project", "mla_project", "attn", "absorbed MLA projections", DT_FLOAT,
       L(mla_project)),
SIMPLE("avx_swiglu_clamped", "swiglu_clamped", "elem", "asymmetrically clipped SwiGLU", DT_FLOAT,
       L(swiglu_clamped)),
SIMPLE("avx_silu_mul", "silu_mul", "elem", "silu-gated product over a fused gate_up", DT_FLOAT,
       L(silu_mul)),
SIMPLE("avx_gelu_mul", "gelu_mul", "elem", "gelu_tanh-gated product over a fused gate_up", DT_FLOAT,
       L(gelu_mul)),
SIMPLE("avx_gelu", "gelu", "elem", "exact erf gelu", DT_FLOAT, L(gelu)),
SIMPLE("avx_silu", "silu", "elem", "silu", DT_FLOAT, L(silu)),
SIMPLE("avx_sigmoid", "sigmoid", "elem", "logistic sigmoid", DT_FLOAT, L(sigmoid)),
SIMPLE("avx_fill", "fill", "elem", "constant fill", DT_FLOAT, L(fill)),
SIMPLE("avx_logit_softcap", "logit_softcap", "elem", "final-logit softcap: cap * tanh(x / cap)",
       DT_FLOAT, L(logit_softcap)),
SIMPLE("avx_scale", "scale", "elem", "a [1] weight broadcast over every element", DT_FLOAT, L(scale)),
SIMPLE("avx_gather_rows", "gather_rows", "elem", "y[i,:] = x[idx[i],:]", DT_ANY, L(gather_rows)),
SIMPLE("avx_scatter_rows", "scatter_rows", "elem", "x[idx[i],:] = v[i,:]", DT_ANY,
       L(scatter_rows)),
SIMPLE("avx_ngram_ids", "ngram_ids", "vocab", "hashed n-gram row ids for PLE", DT_ANY,
       L(ngram_ids)),
SIMPLE("avx_ple_gate", "ple_gate", "norm", "PLE gate: three grouped norms, a dot, a signed sqrt",
       DT_FLOAT, L(ple_gate)),
SIMPLE("avx_ple_conv", "ple_conv", "norm", "PLE's dilated depthwise conv, silu and residual add",
       DT_FLOAT, L(ple_conv)),
SIMPLE("avx_causal_conv1d", "causal_conv1d", "conv", "depthwise causal conv, no activation",
       DT_FLOAT, L(causal_conv1d)),
SIMPLE("avx_cast", "cast", "elem", "dtype conversion", DT_ANY, L(cast)),
SIMPLE("avx_softmax", "softmax", "elem", "row-wise softmax", DT_FLOAT, L(softmax)),
SIMPLE("avx_scale_rows", "scale_rows", "elem", "one scalar per row, broadcast across the row",
       DT_FLOAT, L(scale_rows)),

SIMPLE("avx_quant_act_i8", "quant_act_i8", "quant",
       "symmetric int8 activation quantiser, one scale a row, codes in the WMMA step order",
       "float in, i8 out, f32 scale", L(quant_act_i8)),
SIMPLE("avx_quant_act_i8g", "quant_act_i8g", "quant",
       "symmetric int8 activation quantiser, one scale per group, row-major",
       "float in, i8 out, f32 scale", L(quant_act_i8g)),
SIMPLE("avx_had_quant_act_i8", "had_quant_act_i8", "quant",
       "Walsh-Hadamard rotation then symmetric int8", "float in, i8 out, f32 scale",
       L(had_quant_act_i8)),
SIMPLE("avx_quant_act_fp8", "quant_act_fp8", "quant", "symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale", L(quant_act_fp8)),
SIMPLE("avx_quant_cols_fp8", "quant_cols_fp8", "quant", "E4M3 with one scale per column",
       "float in, fp8_e4m3 out, f32 scale", L(quant_cols_fp8)),
SIMPLE("avx_had_quant_act_fp8", "had_quant_act_fp8", "quant",
       "width-`group` Hadamard, then the same symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale per group; n a multiple of group",
       L(had_quant_act_fp8)),
SIMPLE("avx_had_quant_act_i8g", "had_quant_act_i8g", "quant",
       "width-`group` Hadamard, then the symmetric int8 quantiser, one scale per group",
       "float in, i8 out, f32 scale per group; n a multiple of group",
       L(had_quant_act_i8g)),
SIMPLE("avx_givens_rotate", "givens_rotate", "quant",
       "channel scale and Givens rotations per part, in the output's dtype",
       "float in, float out; int32 pairs, f32 angles and channel scales", L(givens_rotate)),
SIMPLE("avx_givens_dense_quant", "givens_dense_quant", "quant",
       "the rotation as a dense per-group matrix, then the block-scaled int8 or E4M3 quantiser",
       "float in, int8 or fp8_e4m3 out, f32 scale per group; f16 matrices and f32 channel scales",
       L(givens_dense_quant)),
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
ROWL("avx_gemm_nt_q", "gemm_nt_q", "gemm", "C = A B^T with group-scaled quantised operands",
     "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; u2 / u4 / u5 codes with an f16 scale "
     "and integer zero per group, joined; f32 accumulate", L(gemm_nt_q), avx_layout_gemm_q),
SIMPLE("avx_gemm_nt_q_bias", "gemm_nt_q_bias", "gemm", "quantised C = A B^T + bias",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; f32 accumulate", L(gemm_nt_q_bias)),
SIMPLE("avx_lora_shrink", "lora_shrink", "gemm", "t = x A[slot]^T per row, base rows untouched",
       "bf16 / f16 / f32 a, or fp8_e4m3 / fp8_e5m2 / i8 a with its f32 scale stream; bf16 lora_a; "
       "i32 slot; f32 t", L(lora_shrink)),
SIMPLE("avx_lora_expand", "lora_expand", "gemm",
       "y += scale * (t B[slot]^T) per row, base rows untouched",
       "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y", L(lora_expand)),
SIMPLE("avx_lora_expand_mm", "lora_expand_mm", "gemm",
       "y += scale * (bf16(t) B[slot]^T) per row, base rows untouched",
       "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y", L(lora_expand_mm)),
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
SIMPLE("avx_attn_cross", "attn_cross", "attn",
       "dense varlen attention, keys bounded apart from the queries (cu_k)", DT_FLOAT,
       L(attn_cross)),
SIMPLE("avx_attn_na3d", "attn_na3d", "attn", "3-D neighborhood attention over a volume", DT_FLOAT,
       L(attn_na3d)),
ROW("avx_kv_store", "kv_store", "attn", "scatter k and v into the paged cache", DT_ANY,
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])), L(kv_store)),
SIMPLE("avx_rope", "rope", "attn", "rotary embedding in place", DT_FLOAT, L(rope)),
SIMPLE("avx_rope_table", "rope_table", "attn", "cos/sin at each position's own row", DT_FLOAT,
       L(rope_table)),
SIMPLE("avx_qsa_block_key", "qsa_block_key", "attn",
       "QSA compressed key: pooled, normed and rotated at the block's position", DT_FLOAT,
       L(qsa_block_key)),
SIMPLE("avx_qk_norm_rope", "qk_norm_rope", "attn", "per-head QK norm then rotary embedding",
       DT_FLOAT, L(qk_norm_rope)),
SIMPLE("avx_qsa_score", "qsa_score", "attn",
       "QSA block scores: per-head relu dots summed over the heads", DT_FLOAT, L(qsa_score)),
SIMPLE("avx_qsa_select", "qsa_select", "attn",
       "QSA selection: the top-k blocks as a page-id table, partial page last",
       "f32 scores, i32 tables", L(qsa_select)),
SIMPLE("avx_qsa_tail_store", "qsa_tail_store", "attn",
       "QSA tail: the raw keys of open blocks into a per-sequence ring", DT_FLOAT,
       L(qsa_tail_store)),
SIMPLE("avx_qsa_work", "qsa_work", "attn",
       "QSA work list: completing blocks' keys, pages and positions", DT_FLOAT, L(qsa_work)),

SIMPLE("avx_router_topk", "router_topk", "moe", "softmax then exact top-k routing", DT_FLOAT,
       L(router_topk)),
SIMPLE("avx_router_sigmoid_topk", "router_sigmoid_topk", "moe", "sigmoid correction-bias routing", DT_FLOAT,
       L(router_sigmoid_topk)),
SIMPLE("avx_moe_scatter", "moe_scatter", "moe", "stable counting sort of routing slots by expert",
       "i32 indices", L(moe_scatter)),
SIMPLE("avx_moe_gemm", "moe_gemm", "moe", "grouped GEMM over the expert-sorted rows", DT_FLOAT,
       L(moe_gemm)),
ROW_CB("avx_moe_gemm_q", "moe_gemm_q", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands", DT_FLOAT_CB,
       L(moe_gemm_q)),
ROW_CB("avx_moe_gemm_q_mod4", "moe_gemm_q_mod4", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands, four classes by number mod 4",
       DT_FLOAT_CB, L(moe_gemm_q_mod4)),
SIMPLE("avx_moe_gather", "moe_gather", "moe", "weighted scatter back to token order", DT_FLOAT,
       L(moe_gather)),
SIMPLE("avx_row_topk", "row_topk", "moe", "exact per-row top-R", DT_FLOAT, L(row_topk)),
SIMPLE("avx_row_topk_merge", "row_topk_merge", "moe",
       "the global top-R of the gathered per-rank candidate sets", DT_FLOAT, L(row_topk_merge)),
SIMPLE("avx_pair_argmax", "pair_argmax", "moe",
       "the global top-1 of a row of gathered (id, value) pairs, one id a row", DT_FLOAT,
       L(pair_argmax)),
ROWL("avx_logit_rerank", "logit_rerank", "moe",
     "R candidates a row rescored exactly against the full-precision head",
     DT_FLOAT ", or a head of E4M3 / int8 codes with a scale plane, joined", L(logit_rerank),
     avx_layout_head),

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

/* No init/fini on the collectives: unlike libref's (ref_collective_init sets up a cross-rank
 * rendezvous), these serve world_size 1 only, where both are copies (see avx_misc.cpp). */
SIMPLE("avx_all_reduce", "all_reduce", "ar", "sum across ranks; world_size 1 only", DT_FLOAT,
       L(all_reduce)),
SIMPLE("avx_all_gather", "all_gather", "ar", "gather every rank's shard; world_size 1 only",
       DT_FLOAT, L(all_gather)),

SIMPLE("avx_ln_mod", "ln_mod", "norm", "gainless layer norm times (1 + an entry's scale row)",
       DT_FLOAT, L(ln_mod)),
SIMPLE("avx_gate_add", "gate_add", "elem", "x += act(gate row) * y, act tanh or none", DT_FLOAT,
       L(gate_add)),
SIMPLE("avx_table_add", "table_add", "elem", "a row plus each of L table rows", DT_FLOAT,
       L(table_add)),
SIMPLE("avx_mix_rows", "mix_rows", "elem", "a weighted sum of k consecutive rows", DT_FLOAT,
       L(mix_rows)),
SIMPLE("avx_chan_stats", "chan_stats", "elem", "per-channel statistics held as weights",
       DT_FLOAT, L(chan_stats)),
SIMPLE("avx_ln_mod_quant_fp8", "ln_mod_quant_fp8", "norm",
       "gated residual, adaptive norm and the E4M3 quantiser in one pass",
       "float in, fp8_e4m3 out, f32 scale, bf16 z", L(ln_mod_quant_fp8)),
SIMPLE("avx_timestep_embed", "timestep_embed", "elem", "sinusoidal timestep embedding", DT_FLOAT,
       L(timestep_embed)),
SIMPLE("avx_gather_rows2", "gather_rows2", "elem", "rows from two sources by a signed index",
       DT_ANY, L(gather_rows2)),
SIMPLE("avx_lane_gather", "lane_gather", "elem", "each entry's rows out of its lane", DT_ANY,
       L(lane_gather)),
SIMPLE("avx_lane_store", "lane_store", "elem", "one source's rows into several lanes", DT_ANY,
       L(lane_store)),
SIMPLE("avx_randn_lanes", "randn_lanes", "elem", "a lane's rows from its seed's normal stream",
       DT_FLOAT, L(randn_lanes)),
SIMPLE("avx_flow_euler", "flow_euler", "elem", "a flow-matching Euler step with guidance",
       DT_FLOAT, L(flow_euler)),
SIMPLE("avx_kd_input", "kd_input", "elem", "a k-diffusion sampler's model input", DT_FLOAT,
       L(kd_input)),
SIMPLE("avx_kd_step", "kd_step", "elem", "a k-diffusion sampler step with guidance", DT_FLOAT,
       L(kd_step)),
SIMPLE("avx_chan_affine", "chan_affine", "elem", "a per-channel affine from parameter lists",
       DT_FLOAT, L(chan_affine)),
SIMPLE("avx_conv2d_nhwc", "conv2d_nhwc", "conv",
       "an image convolution, channels innermost, as GEMMs over patch rows", DT_FLOAT,
       L(conv2d_nhwc)),
SIMPLE("avx_im2col_nhwc", "im2col_nhwc", "elem", "a convolution's receptive fields as rows",
       DT_FLOAT, L(im2col_nhwc)),
SIMPLE("avx_l2norm_chan", "l2norm_chan", "norm", "unit-L2 channel norm, scaled, optional silu",
       DT_FLOAT, L(l2norm_chan)),
SIMPLE("avx_d2s_add", "d2s_add", "elem", "an upsampling shortcut added in", DT_FLOAT, L(d2s_add)),
SIMPLE("avx_s2d_avg_add", "s2d_avg_add", "elem", "a downsampling shortcut added in", DT_FLOAT,
       L(s2d_avg_add)),
SIMPLE("avx_to_rgba8", "to_rgba8", "elem", "decoded pixels to bytes", DT_FLOAT, L(to_rgba8)),
SIMPLE("avx_gelu_tanh", "gelu_tanh", "elem", "tanh gelu", DT_FLOAT, L(gelu_tanh)),
SIMPLE("avx_gelu_quick", "gelu_quick", "elem", "quick gelu", DT_FLOAT, L(gelu_quick)),
SIMPLE("avx_scatter_add_rows", "scatter_add_rows", "elem", "x[idx[i]] += v[i]", DT_FLOAT,
       L(scatter_add_rows)),
SIMPLE("avx_attn_prefix_self", "attn_prefix_self", "attn",
       "attention over a paged prefix and the step's own rows, flash-attention tiles",
       "bf16 or f32 q, k, v and cache; int8 score form", L(attn_prefix_self)),
SIMPLE("avx_rms_mod", "rms_mod", "norm",
       "gated residual, RMS norm with a gain times (1 + scale row) plus a shift row", DT_FLOAT,
       L(rms_mod)),
SIMPLE("avx_rms_mod_quant_fp8", "rms_mod_quant_fp8", "norm",
       "gated residual, modulated RMS norm and the E4M3 quantiser in one pass", DT_FLOAT,
       L(rms_mod_quant_fp8)),
SIMPLE("avx_flow_x0_step", "flow_x0_step", "elem", "a sampler step toward the clean estimate",
       DT_FLOAT, L(flow_x0_step)),
SIMPLE("avx_patch_unfold", "patch_unfold", "elem", "patch rows to denormalised voxel rows",
       DT_FLOAT, L(patch_unfold)),
SIMPLE("avx_vtile_gather", "vtile_gather", "elem", "a video decoder's tiles out of the latent",
       DT_FLOAT, L(vtile_gather)),
SIMPLE("avx_vtile_stitch", "vtile_stitch", "elem",
       "decoded tiles blended into frames, the chunk overlap, bytes out", DT_FLOAT,
       L(vtile_stitch)),
SIMPLE("avx_grid_pos3", "grid_pos3", "elem", "a video decoder's rotary positions over its tiles",
       DT_FLOAT, L(grid_pos3)),
SIMPLE("avx_vsa_select", "vsa_select", "attn",
       "video sparse attention's top-k tiles by pooled score, and the coarse branch",
       "float q, k, v and coarse; i32 segs, geom and sel", L(vsa_select)),
SIMPLE("avx_attn_vsa", "attn_vsa", "attn",
       "block-sparse attention over the prefix and selected video tiles, the gated coarse term",
       "float q, k, v, coarse, gate and out; i32 segs, geom and sel", L(attn_vsa)),
SIMPLE("avx_conv3d_ndhwc", "conv3d_ndhwc", "conv",
       "a 3-D convolution over voxel rows, as GEMMs over patch rows", DT_FLOAT, L(conv3d_ndhwc)),
SIMPLE("avx_group_norm", "group_norm", "norm",
       "a group norm over all rows, optionally modulated, then an activation", DT_FLOAT,
       L(group_norm)),
SIMPLE("avx_tconv_dw", "tconv_dw", "conv", "a depthwise convolution along a volume's frames",
       DT_FLOAT, L(tconv_dw)),
SIMPLE("avx_resize_bilinear", "resize_bilinear", "elem",
       "each frame of a volume resized bilinearly", DT_FLOAT, L(resize_bilinear)),
SIMPLE("avx_patch_fold", "patch_fold", "elem", "voxel rows to normalised patch rows", DT_FLOAT,
       L(patch_fold)),
SIMPLE("avx_flow_renoise", "flow_renoise", "elem", "a clean latent re-noised to a sigma",
       DT_FLOAT, L(flow_renoise)),
SIMPLE("avx_rope_split", "rope_split", "elem", "LTX-2's split rotary over coordinates", DT_FLOAT,
       L(rope_split)),
SIMPLE("avx_head_gate", "head_gate", "elem", "each head times 2 sigmoid of its logit", DT_FLOAT,
       L(head_gate)),
SIMPLE("avx_ltx_guide", "ltx_guide", "elem", "LTX-2's guided sampler step", DT_FLOAT,
       L(ltx_guide)),
SIMPLE("avx_pixel_norm", "pixel_norm", "norm", "an RMS norm as bf16 tensors compute it", DT_FLOAT,
       L(pixel_norm)),
SIMPLE("avx_d2s3d", "d2s3d", "elem", "depth to space over voxel rows", DT_FLOAT, L(d2s3d)),
SIMPLE("avx_s2d3d", "s2d3d", "elem", "space to depth over voxel rows", DT_FLOAT, L(s2d3d)),
SIMPLE("avx_pix_patch", "pix_patch", "elem", "an image to patch rows", DT_FLOAT, L(pix_patch)),
SIMPLE("avx_patch_pix8", "patch_pix8", "elem", "patch rows to RGB bytes", DT_FLOAT, L(patch_pix8)),
SIMPLE("avx_vol_index", "vol_index", "elem", "a volume's voxel coordinates", DT_ANY, L(vol_index)),
SIMPLE("avx_tile_grid", "tile_grid", "elem", "a tiled decode's boxes", DT_ANY, L(tile_grid)),
SIMPLE("avx_tile_crop", "tile_crop", "elem", "a tile out of a volume", DT_FLOAT, L(tile_crop)),
SIMPLE("avx_tile_stitch", "tile_stitch", "elem", "decoded tiles blended to RGB bytes", DT_FLOAT,
       L(tile_stitch)),
SIMPLE("avx_stft_logmel", "stft_logmel", "elem", "a causal log-mel spectrogram", DT_FLOAT,
       L(stft_logmel)),
SIMPLE("avx_bwe_up", "bwe_up", "elem", "a sinc-resampled waveform plus a residual, clamped",
       DT_FLOAT, L(bwe_up)),

SIMPLE("avx_conv1d", "conv1d", "conv", "a 1-D convolution over sequences, as GEMMs over patch rows",
       DT_FLOAT, L(conv1d)),
SIMPLE("avx_conv_t1d", "conv_t1d", "conv", "a transposed 1-D convolution, one GEMM a phase",
       DT_FLOAT, L(conv_t1d)),
SIMPLE("avx_snake", "snake", "elem", "the snake activation", DT_FLOAT, L(snake)),
SIMPLE("avx_snake_aa", "snake_aa", "elem", "the alias-free SnakeBeta: upsample, snake, downsample",
       DT_FLOAT, L(snake_aa)),
SIMPLE("avx_cfg_topk_sample", "cfg_topk_sample", "sample", "a guided top-k pick a lane", DT_FLOAT,
       L(cfg_topk_sample)),
SIMPLE("avx_rvq_embed", "rvq_embed", "elem", "a frame's embedding from its residual codes",
       DT_FLOAT, L(rvq_embed)),
SIMPLE("avx_frame_mix_store", "frame_mix_store", "elem",
       "a frame's conditioning row into its lane", DT_FLOAT, L(frame_mix_store)),
SIMPLE("avx_end_status", "end_status", "elem", "a frame's status from its first pick", DT_ANY,
       L(end_status)),
SIMPLE("avx_frame_store", "frame_store", "elem", "a frame's codes into its lane's history", DT_ANY,
       L(frame_store)),
SIMPLE("avx_resample_nearest", "resample_nearest", "elem", "rows resampled nearest-neighbour",
       DT_ANY, L(resample_nearest)),
SIMPLE("avx_fourier_embed", "fourier_embed", "elem", "random Fourier features of a time",
       DT_FLOAT, L(fourier_embed)),
SIMPLE("avx_win_input", "win_input", "elem", "a denoiser's input from the window states",
       DT_FLOAT, L(win_input)),
SIMPLE("avx_win_prep", "win_prep", "elem", "a window step's noise and overlap", DT_FLOAT,
       L(win_prep)),
SIMPLE("avx_win_finish", "win_finish", "elem", "a window's end and the overlap it hands on",
       DT_FLOAT, L(win_finish)),
SIMPLE("avx_chan_split", "chan_split", "elem", "channel groups to sequences", DT_ANY,
       L(chan_split)),
SIMPLE("avx_interleave2", "interleave2", "elem", "two streams interleaved", DT_ANY,
       L(interleave2)),
SIMPLE("avx_iota", "iota", "elem", "an arithmetic index sequence", DT_ANY, L(iota)),
SIMPLE("avx_cu_repeat", "cu_repeat", "elem", "bounds of equal sequences", DT_ANY, L(cu_repeat)),
SIMPLE("avx_int_affine", "int_affine", "elem", "integers derived from device geometry", DT_ANY,
       L(int_affine)),

SIMPLE("avx_embed_lookup", "embed_lookup", "vocab", "embedding gather", DT_FLOAT, L(embed_lookup)),
/* Its layout hook only validates: the table is read as stored and other encodings are refused
 * (avx_ngram.cpp). An E4M3 table into bf16 goes through the row cache; other dtype pairs take
 * the general gather. The init hook refuses, at declare, a process that cannot create the
 * io_uring ring the cache misses are read through. */
{ "avx_embed_lookup_q", "embed_lookup_q", "vocab",
  "embedding gather, one scale for the table: an E4M3 table into bf16 through a shared row cache "
  "and direct reads of the container, a later call's rows read ahead",
  SHAPE_ANY, DT_ANY, RAD_DOMAIN_HOST, 0,
  nullptr, 0, nullptr, 0, nullptr,
  /* init, fini, launch, scratch, layout, relayout, opd_shape, unrelayout, replaces */
  avx_init_ngram, nullptr, L(embed_lookup_q), nullptr, avx_layout_ngram, nullptr, nullptr, nullptr,
  nullptr },
ROWL("avx_logits_gemm", "logits_gemm", "vocab", "the lm_head projection",
     DT_FLOAT ", or a head of E4M3 / int8 codes with a scale plane, joined", L(logits_gemm),
     avx_layout_head),

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
SIMPLE("avx_sample_logprobs", "sample_logprobs", "sample", "the raw distribution's summary",
       DT_FLOAT, L(sample_logprobs)),
SIMPLE("avx_dflash_conv", "dflash_conv", "draft",
       "DFlash2 grouped dynamic depthwise convolution", DT_FLOAT, L(dflash_conv)),
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
    "scalar / AVX / AVX2+FMA / AVX-512. Same ops and schemas as libref, the same answers to a "
    "tolerance",
    /* Not a -march string: the binary runs on any x86-64 and picks its own path at load. */
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
