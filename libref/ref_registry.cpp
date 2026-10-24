/* ref_registry.cpp -- libref's op schemas and kernel rows.
 *
 * The schemas are the authoritative copy of the conventional op vocabulary; docs/OPS.md is their
 * prose form (spec §17). The core never learns op names, but ref can only implement ops it knows,
 * so an architecture plugin using only these ops is guaranteed a fallback and an oracle. Adding an
 * op means a row here plus a host implementation; no core file changes.
 *
 * The first plugin in hierarchy order to declare an op fixes its schema, and ref sits last, so a
 * faster plugin's schema wins and ref's must agree. Arguments are positional, so disagreeing
 * schemas would silently give the same call different meanings (spec §2.3). Operand orders follow
 * docs/OPS.md; deviations are marked in the deviating row's doc string.
 *
 * The kernel rows are deliberately unconstrained: every extent is read off the operands, and ref
 * must match anything. The exception is kv_store, which must be told the paged block size and so
 * carries `block_size >= 1`; that admits every value and refuses only a query that omits the key
 * (an absent key fails a constraint, rad_util.cpp's constraint_holds). attn_paged carries the same
 * constraint, but there `block_size` is RAD_DERIVED, so it is skipped during matching and ref reads
 * the block size off the cache tensor.
 *
 * spec §7.2 has the block manager read the KV block size from the resolved attention kernel's
 * constraints. libr4d answers `block_size == 16`; ref answers "any", so a core resolving attention
 * to ref alone must pick, and 16 is the value the project is built around.
 *
 * Everything is domain HOST at priority 0; a device-side reference would be a separate row.
 */
#include "ref_common.h"
#include "ref_ops.h"
#include "ref_planes.h"

/* ================================================================== parameter specs */
#define P_INT(k)   { k, RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { k, RAD_P_INT, RAD_OPTIONAL }
#define P_STR(k)   { k, RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { k, RAD_P_STR, RAD_OPTIONAL }

/* Supplied by the kernel, not the caller (rad_abi.h, spec §7.2): constraints on it are skipped
 * during matching and the winner's RAD_C_EQ value is written into the resolved geometry. RAD_REQUIRED
 * would make it unselectable, since the caller cannot know it before resolution. */
#define P_INTD(k)  { k, RAD_P_INT, RAD_DERIVED }
/* The linear-state tile length. The scheduler cuts its steps on a multiple of it (rad_abi.h,
 * RAD_PROLE_SEQ_CHUNK); nothing else in the core reads a parameter by meaning. */
#define P_CHUNK(k) { k, RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }
/* A count the kernel is handed as a bound and never writes past its operands for (rad_abi.h,
 * RAD_PROLE_CAPACITY). */
#define P_CAP(k)   { k, RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY }

/* Real-valued parameters are RAD_P_F64 (docs/OPS.md, "The four rules"); the core refuses a string
 * spelling for these keys, so plugins cannot drift apart over formatting. Constraints never match
 * on a float. */
#define P_F64(k)   { k, RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { k, RAD_P_F64, RAD_OPTIONAL }

#define OPD(n)     { n, RAD_OPD_IN,     0 }
#define OPD_O(n)   { n, RAD_OPD_IN,     1 }
#define OUT(n)     { n, RAD_OPD_OUT,    0 }
#define OUT_O(n)   { n, RAD_OPD_OUT,    1 }
#define INOUT(n)   { n, RAD_OPD_INOUT,  0 }
#define INOUT_O(n) { n, RAD_OPD_INOUT,  1 }
#define WGT(n)     { n, RAD_OPD_WEIGHT, 0 }
/* A weight table: the issue names a run of declared weights and the kernel gets a device array of
 * pointers, one per expert (RAD_WTAB in abi/rad_runtime.h). */
#define WTAB(n)    { n, RAD_OPD_WTAB,   0 }
#define WGT_O(n)   { n, RAD_OPD_WEIGHT, 1 }

#define ARR(a)  (a), (int)(sizeof(a) / sizeof((a)[0]))

/* ---- elementwise and norms ---------------------------------------------------------------- */
/* `mult` scales the normed row before its one rounding (default 1). rmsnorm_add has none: nothing
 * fuses a scaled branch into the add. */
static const RadParamSpec pRmsnormAdd[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                          P_F64O("wadd") };
static const RadParamSpec pRmsnorm[]  = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                          P_F64O("wadd"), P_F64O("mult") };
/* `w` is optional: absent, the gain is one (a norm with no scale, such as Gemma 4's value norm and
 * its multimodal embedder's pre-projection norm). */
static const RadOperandSpec oRmsnorm[] = { OPD("x"), WGT_O("w"), OUT("y") };

/* The gated residual pair. The operand list is banded on `inject` and `group`: `inject` 0 is the
 * final mixer (no per-stream gate weight or output), `group` 0 a read whose consumer wants bf16
 * instead of block-scaled fp8. Both default on; only two sites in the model turn them off. */
static const RadParamSpec pHcRead[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"),
                                        P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
                                        P_INTO("inject"), P_INTO("group"), P_INTO("rotate"),
                                        P_STRO("mix") };
static const RadOperandSpec oHcRead[] = { OPD("h"), WGT("w"), WGT("mix_down"), WGT("mix_up"),
                                          WGT_O("inject_w"), OUT("x"), OUT_O("inj"), OUT_O("q"),
                                          OUT_O("scale"), OUT_O("rq"), OUT_O("rscale") };
/* ---- PLE / n-gram ---------------------------------------------------------------------
 *
 * `state` is a rolling window of committed ids with RAD_KV_CONV's shape and ple_conv's contract,
 * read at `num_accepted - 1`. Its cold value is eos, not zero (a real token); see ref_ngram.cpp. `eos`
 * is a parameter, not an operand: the shift rule is a model property and can be constrained on. */
static const RadParamSpec pNgramIds[] = { P_INT("M"), P_INT("heads"), P_INT("ngram"),
                                         P_INT("eos") };
static const RadOperandSpec oNgramIds[] = { OPD("tok"), INOUT("state"), OPD("cu"), WGT("mult"),
                                           WGT("vocab_sizes"), WGT("offsets"),
                                           OPD_O("cache_idx"), OPD_O("has_init"),
                                           OPD_O("num_accepted"), OUT("ids"),
                                           OUT_O("ahead_ids") };
/* PLE's gate. `k` and `q` arrive un-normed; the op owns all three norms, since the third is over
 * bytes the op itself wrote. */
static const RadParamSpec pPleGate[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                         P_STR("dtype"), P_F64O("wadd"), P_F64O("gate_eps") };
static const RadOperandSpec oPleGate[] = { OPD("k"), OPD("q"), OPD("v"), WGT("w_key"),
                                          WGT("w_query"), WGT("w_conv"),
                                          OUT("gv"), OUT("gvn") };
/* The dilated convolution that finishes the layer: one op for prefill and decode, taking both
 * contracts (`has_init` for prefill, `num_accepted` for decode; see ref_ple.cpp). */
static const RadParamSpec pPleConv[] = { P_INT("M"), P_INT("n"), P_INT("width"),
                                         P_INT("dilation"), P_STR("dtype") };
static const RadOperandSpec oPleConv[] = { OPD("x"), WGT("w"), INOUT("conv_state"), OPD("cu"),
                                          OPD_O("resid"), OPD_O("cache_idx"), OPD_O("has_init"),
                                          OPD_O("num_accepted"), OUT("y") };
static const RadParamSpec pCausalConv[] = { P_INT("M"), P_INT("n"), P_INT("width"),
                                            P_STR("dtype") };
static const RadOperandSpec oCausalConv[] = { OPD("x"), WGT("w"), INOUT("conv_state"), OPD("cu"),
                                             OPD_O("cache_idx"), OPD_O("has_init"),
                                             OPD_O("num_accepted"), OUT("y") };

static const RadParamSpec pHcWrite[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype") };
/* The control vector's projection (libref/ref_hc.cpp, core/cvec.h): one unit direction per layer and a
 * per-row switch. `v` is f32 whatever the stream's dtype is, because the engine uploads the table
 * in f32; `on` is an int32 row mask (core derive.cpp's `projection_on_rows`). */
static const RadParamSpec pHcProject[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("scale"),
                                           P_STR("dtype") };
static const RadOperandSpec oHcProject[] = { INOUT("h"), OPD("v"), OPD("on") };
static const RadOperandSpec oHcEnter[] = { OPD("x"), OUT("h") };
static const RadOperandSpec oHcWrite[] = { OPD("y"), OPD("inj"), INOUT("h") };
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
/* The MTP head's entry for a wide residual stream. `ngroup` is hc for a grouped norm over the
 * sub-streams, 1 for a flat one over the whole row (see ref_hc.cpp). */
static const RadParamSpec pMtpEnter[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
                                          P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
                                          P_INTO("ngroup"), P_STRO("fc") };
static const RadOperandSpec oMtpEnter[] = { OPD("h"), OPD("e"), WGT("w_h"), WGT("w_e"),
                                           WGT("fc_h"), WGT("fc_e"), OUT("x") };

static const RadOperandSpec oRmsnormAdd[] = { OPD("x"), OPD("residual"), WGT("w"),
                                              OUT("y"), OUT("residual_out") };

static const RadParamSpec pLayernorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLayernorm[] = { OPD("x"), WGT("w"), WGT_O("b"), OUT("y") };
/* A learned position table resampled to each row's grid and added in (a vision tower's). */
static const RadParamSpec pGridEmbed[] = { P_INT("M"), P_INT("n"), P_INT("side"), P_STR("dtype") };
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
static const RadParamSpec pMnD[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
/* `div` divides the rounded sum, rounded again (default 1, the plain sum): a mean of branches
 * summed pairwise, as a vocoder averages its residual blocks. */
static const RadParamSpec pAdd[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("div") };
/* `scalar` 1 says `w` is one element rather than one row; kernels read that off the operand, but a
 * shape description needs the parameter. */
static const RadParamSpec pMulWeight[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_INTO("scalar") };
/* `cap` is required: the one model that issues this op (Gemma 4) has a value for it. */
static const RadParamSpec pMnCap[] = { P_INT("M"), P_INT("n"), P_F64("cap"), P_STR("dtype") };
static const RadOperandSpec oBinary[]  = { OPD("a"), OPD("b"), OUT("y") };
static const RadOperandSpec oGateUp[]  = { OPD("gate_up"), OUT("y") };
static const RadOperandSpec oMulWeight[] = { OPD("x"), WGT("w"), OUT("y") };
static const RadParamSpec pActMul[] = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };
/* `M` is the step's token count and bands the op; the segment count is cu_seqlens' length. */
static const RadParamSpec pSegMean[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oSegMean[] = { OPD("x"), OPD("cu_seqlens"), OUT("y") };
/* `M` is the input row count, a multiple of `k`; y has M / k rows. */
static const RadParamSpec pPoolRows[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_F64O("mult"),
                                          P_STR("dtype") };
static const RadOperandSpec oUnary[]   = { OPD("x"), OUT("y") };
/* The audio (USM conformer) tower's operators. */
static const RadOperandSpec oClamp[]  = { OPD("x"), WGT("lo"), WGT("hi"), OUT("y") };
/* `M` is the input time rows. */
static const RadParamSpec pConv2d[]   = { P_INT("M"), P_INT("width"), P_INT("cin"), P_INT("cout"),
                                          P_INT("kh"), P_INT("kw"), P_INT("stride"), P_INT("pad"),
                                          P_INTO("cu_div"), P_INTO("valid_stride"),
                                          P_STR("dtype"), P_INTO("t_last"), P_STRO("act") };
static const RadOperandSpec oConv2d[] = { OPD("x"), WGT("w"), OPD("cu_seqlens"), OPD_O("valid"),
                                          OUT("y"), WGT_O("b") };
/* `M` counts input time rows; y is [output rows * Wo, cin * kh * kw]. */
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
/* `M` is the table's rows, one per distance. */
static const RadParamSpec pRelSin[]   = { P_INT("M"), P_INT("n"), P_F64O("max_timescale"),
                                          P_STR("dtype") };
static const RadOperandSpec oRelSin[] = { OUT("y") };
/* `M` is the output rows; `period` the rows a position cycle takes. */
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
static const RadOperandSpec oScale[]   = { OPD("x"), WGT("s"), OUT("y") };
static const RadOperandSpec oGather[]  = { OPD("x"), OPD("idx"), OUT("y") };
static const RadOperandSpec oScatterRows[] = { OPD("v"), OPD("idx"), INOUT("x") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

/* ---- quantisation and rotation ------------------------------------------------------------ */
static const RadParamSpec pQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
/* `asum`, quant_act_i8's optional third output: group-major per-row code sums an asymmetric
 * weight grid subtracts to remove the stored zero point. */
static const RadOperandSpec oQuant[]    = { OPD("x"), OUT("q"), OUT("scale"), OUT_O("asum") };
static const RadOperandSpec oHadQuant[] = { OPD("x"), OUT("q"), OUT("scale") };
/* No `asum`: it exists for asymmetric integer grids and fp8 is symmetric (as libr4d/r4d_rows.cpp
 * notes). */
static const RadOperandSpec oQuantFp8[]  = { OPD("x"), OUT("q"), OUT("scale") };
static const RadParamSpec pQuantCols[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oQuantCols[] = { OPD("x"), OUT("y"), OUT("scale") };
/* ParoQuant's rotation, one per part, to f16 for the 16-bit GEMMs. The tables are the "givens"
 * encoding's (abi/rad_encoding.h) and are weight operands, so they are placed with their weight. */
static const RadParamSpec pGivensRotate[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                             P_INT("stages"), P_INT("parts") };
static const RadParamSpec pGivensDense[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                             P_INT("parts"), P_STR("out") };
static const RadOperandSpec oGivensDense[] = { OPD("x"), WGT("tdense"), WGT("rscale"), OUT("q"),
                                               OUT("scale") };
static const RadOperandSpec oGivensRotate[] = { OPD("x"), WGT("pairs"), WGT("angle"), WGT("rscale"),
                                                OUT("y") };

/* ---- the fp8 fusions ----------------------------------------------------------------------
 * A fused op's operands are the unfused pair's minus the intermediate; optional `out_bf16` restores
 * it (the codes are taken over its value) so the fused form can be compared against the pair. */
static const RadParamSpec pRmsQuantFp8[] = { P_INT("M"), P_INT("n"), P_F64("eps"),
                                             P_INT("group"), P_STR("dtype"), P_F64O("wadd") };
static const RadOperandSpec oRmsQuantFp8[] = { OPD("x"), INOUT_O("residual"), WGT("w"),
                                               OUT("q"), OUT("scale"), OUT_O("out_bf16") };
static const RadParamSpec pGatedQuantFp8[] = { P_INT("M"), P_INT("n"), P_INT("group"),
                                               P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oGatedQuantFp8[] = { OPD("gate_up"), OUT("q"), OUT("scale"),
                                                 OUT_O("out_bf16") };
/* `act` is required here, unlike on the quantiser (docs/OPS.md; see ref_gemm_nt_q_gated). `N` is
 * the weight's rows, twice the fused width, so the output is [M, N/2]. */
static const RadParamSpec pGemmQGated[] = { P_INT("M"), P_INT("N"), P_INT("K"),
                                            P_INT("group"), P_STR("dtype"), P_STR("act") };
static const RadOperandSpec oGemmQGated[] = { OPD("a"), OPD("a_scale"), WGT("b"),
                                              WGT("b_scale"), OUT("q"), OUT("scale"),
                                              OUT_O("out_bf16") };
static const RadParamSpec pDflashSelect[] = { P_INT("M"), P_INT("steps"), P_INT("top_k"),
                                              P_INT("rank"), P_INT("n_vocab"),
                                              P_INTO("anchor_stride"), P_STR("dtype") };
static const RadOperandSpec oDflashSelect[] = { OPD("cand"), OPD("unary"), OPD_O("hp"),
                                                OPD("anchor"), WGT("pred"), WGT("succ"),
                                                OUT("tokens") };

static const RadParamSpec pRmsHadQuant[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                             P_STR("dtype"), P_F64O("wadd") };
static const RadOperandSpec oRmsHadQuant[] = { OPD("x"), INOUT_O("residual"), WGT("w"),
                                               OUT("q"), OUT("scale"), OUT_O("out_bf16") };

/* Callers pass gate and up as two operands, or one fused [M, 2n] `gate_up` as `a` with `b`
 * omitted. `mode` is libr4d's optional integer spelling of `act`. */
static const RadParamSpec pGatedQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                            P_STRO("act"), P_INTO("mode") };
static const RadOperandSpec oGatedQuant[] = { OPD("a"), OPD_O("b"), OUT("q"), OUT("scale") };

/* `h` is INOUT: the caller zeroes it once and each launch over the calibration corpus adds its
 * block. `a`/`a_scale` are gemm_nt_q's first two operands, the activation this op characterises;
 * `a_scale` is absent for a plain bf16 activation. */
static const RadOperandSpec oGram[] = { OPD("a"), OPD_O("a_scale"), INOUT("h") };

static const RadParamSpec pDequant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oDequant[] = { OPD("q"), OPD("scale"), OUT("y") };

/* ---- gemm --------------------------------------------------------------------------------- */
static const RadParamSpec pGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype") };
static const RadOperandSpec oGemm[]     = { OPD("a"), WGT("b"), OUT("y") };
/* The bias form carries an optional epilogue: an activation of the biased product, then a residual
 * add, each rounded to the output dtype as separate tensors would be. */
static const RadParamSpec pGemmBias[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
                                          P_STRO("act") };
static const RadOperandSpec oGemmBias[] = { OPD("a"), WGT("b"), WGT("bias"), OPD_O("res"),
                                            OUT("y") };

/* `split_n` is libr4d's (docs/OPS.md): ref never splits K, so it accepts and ignores it. */
static const RadParamSpec pGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                       P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                       P_INTO("split_n") };
/* The bias form has no parts (`pb1`/`pb2`); only the plain product supports them. */
static const RadParamSpec pGemmQBias[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                           P_STR("dtype") };
/* `a_sum` and `b_ref` are libr4d's, appended after `y` so docs/OPS.md's list stays a positional
 * prefix: asymmetric 4- and 2-bit grids subtract the activation's per-group code sums, and mxfp4
 * folds a per-output-row reference exponent. Neither fits a weight aux stream (RadLayout can
 * declare one but RadArgs cannot deliver it). Ref reads no zero point, so it refuses `a_sum`;
 * `b_ref` is derived from `b_scale`, so ref accepts it beside an e2m1 weight and ignores it. */
static const RadOperandSpec oGemmQ[]     = { OPD("a"), OPD_O("a_scale"), WGT("b"), WGT("b_scale"),
                                             OUT("y"), OPD_O("a_sum"), WGT_O("b_ref") };
static const RadOperandSpec oGemmQBias[] = { OPD("a"), OPD("a_scale"), WGT("b"), WGT("b_scale"),
                                             WGT("bias"), OUT("y") };

/* Runtime LoRA, the two halves of y += scale * (B . (A . x)) at a trunk GEMM, per row: `slot` names
 * each row's adapter and a row outside [0, S) is a base row neither op touches. `a`/`a_scale` are
 * the GEMM's own first two operands, read as gemm_nt_q reads them, and `y` is its output. The pools
 * (`lora_a` [S, R, K], `lora_b` [S, N, R], `scale` [S, N]) are inputs, not weights: the engine owns
 * them as program buffers and copies an adapter into a slot between steps. `group` is optional, 0
 * meaning no scale stream; `dtype` is the tensor dtype name (rad_dtype_name) of `a` for shrink and
 * of `y` for expand. */
static const RadParamSpec pLoraShrink[] = { P_INT("M"), P_INT("K"), P_INT("R"), P_INT("S"),
                                            P_INTO("group"), P_STR("dtype") };
static const RadOperandSpec oLoraShrink[] = { OPD("a"), OPD_O("a_scale"), OPD("slot"),
                                              OPD("lora_a"), OUT("t") };
static const RadParamSpec pLoraExpand[] = { P_INT("M"), P_INT("N"), P_INT("R"), P_INT("S"),
                                            P_STR("dtype") };
static const RadOperandSpec oLoraExpand[] = { OPD("t"), OPD("slot"), OPD("lora_b"), OPD("scale"),
                                              INOUT("y") };

/* ---- attention ---------------------------------------------------------------------------- */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"), P_INTD("block_size"), P_INT("causal"),
    P_INT("window"), P_STR("q_dtype"), P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
    /* For libr4d's scratch hook, which sizes its split-KV partial buffer at declare
     * (num_seqs * q_len * q_heads * splits rows; gqa is only the head ratio). Ref ignores them;
     * they are here because both plugins must agree on the parameter list. */
    P_INTO("n_head"), P_INTO("max_seqs")
};
static const RadOperandSpec oAttnPaged[] = {
    OPD("q"), OPD("kv_cache"), OPD("block_table"), OPD("seqused"),
    OPD_O("k_descale"), OPD_O("v_descale"), OPD_O("cu_seqlens"), OUT("out")
};
/* Effective on kv_store, where `block_size` is RAD_REQUIRED: admits every value and refuses only a
 * query that omits it. On attn_paged the key is RAD_DERIVED and the constraint is skipped
 * (core/plugin/select.cpp). See the file header for the spec §7.2 consequence. */
static const RadConstraint cPagedBlock[] = { RAD_CGE("block_size", 1) };

static const RadParamSpec pAttnDense[] = { P_INT("M"), P_INT("head_dim"), P_INT("gqa"),
                                           P_INT("causal"), P_STR("dtype"), P_F64O("scale"),
                                           P_INTO("max_seqlen"), P_INTO("window"),
                                           P_INTO("cu_div"), P_INTO("block"),
                                           P_INTO("window_right") };
static const RadOperandSpec oAttnDense[] = { OPD("q"), OPD("k"), OPD("v"), OPD("cu_seqlens"),
                                             OUT("out") };
static const RadParamSpec pAttnCross[] = { P_INT("M"), P_INT("head_dim"), P_INT("gqa"),
                                           P_INT("causal"), P_STR("dtype"), P_F64O("scale") };
static const RadOperandSpec oAttnCross[] = { OPD("q"), OPD("k"), OPD("v"), OPD("cu_seqlens"),
                                             OPD("cu_k"), OUT("out") };
static const RadParamSpec pAttnNa3d[] = { P_INT("M"), P_INT("head_dim"), P_INT("kt"), P_INT("kh"),
                                          P_INT("kw"), P_STR("dtype"), P_F64O("scale") };
static const RadOperandSpec oAttnNa3d[] = { OPD("q"), OPD("k"), OPD("v"), OPD("geom"), OUT("out") };

/* `k_scale` and `v_scale` are the descales an fp8 cache is read back with, so a store divides by
 * them; optional because a bf16 cache has none. */
static const RadParamSpec pKvStore[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                         P_INT("block_size"), P_STR("kv_dtype"),
                                         P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { OPD("k"), OPD("v"), OPD("slot_mapping"),
                                           INOUT("kv_cache") };

static const RadParamSpec pRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                      P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                      P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections"),
                                      P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                      P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                      P_STRO("ladder"), P_F64O("llama3_factor"),
                                      P_F64O("llama3_low"), P_F64O("llama3_high"),
                                      P_F64O("llama3_orig") };
static const RadOperandSpec oRope[] = { INOUT("qkv"), OPD("positions") };

/* The cos/sin plane a fused prologue reads instead of theta, at each position's own row of a
 * [max_ctx, rot] table. `rot`, `theta` and `scale` are spelled as on `rope`, which must compute the
 * same angles. */
static const RadParamSpec pRopeTable[] = { P_INT("M"), P_INT("rot"), P_F64("theta"),
                                           P_F64("scale"), P_STR("dtype"),
                                           P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                           P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                           P_INTO("head_dim"), P_STRO("ladder") };
static const RadOperandSpec oRopeTable[] = { OPD("positions"), OUT("cos_sin") };

/* QSA's compressed key, libr4d's schema exactly (libr4d/r4d_rows.cpp): `block_size` is `ratio`
 * spelled for the block-key store's KV group; `page` and `pos` are optional. */
static const RadParamSpec pQsaBlockKey[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                             P_INT("rotary_dim"), P_INTO("pos0"),
                                             P_INTO("block_size"), P_F64("eps"), P_F64("theta"),
                                             P_F64O("scale"), P_F64O("wadd"), P_STR("dtype"),
                                             P_STRO("mode"), P_STRO("sections"),
                                             P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                             P_F64O("yarn_high"), P_F64O("yarn_magnitude") };
static const RadOperandSpec oQsaBlockKey[] = { OPD("k"), WGT("w"), OPD_O("page"), OPD_O("pos"),
                                               OUT("out") };

/* The rest of the QSA indexer, libr4d's schemas exactly (libr4d/r4d_rows.cpp). `M` counts query
 * rows (score, select) or tokens (tail store, work list); `cu` maps a row to its sequence, whose
 * count is cu's length less one (`seqs` is the declared height). */
static const RadParamSpec pQsaScore[] = { P_INT("M"), P_INT("n"), P_INT("heads"), P_INT("blocks"),
                                          P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaScore[] = { OPD("q"), OPD("bk"), OPD("bt"), OPD("cu"), OPD("nc"),
                                            OUT("score") };
static const RadParamSpec pQsaSelect[] = { P_INT("M"), P_INT("blocks"), P_INT("topk"),
                                           P_INT("ratio"), P_INT("seqs"), P_INTO("signed") };
static const RadOperandSpec oQsaSelect[] = { OPD("score"), OPD("bt"), OPD("nc"), OPD("pos"),
                                             OPD("cu"), OUT("sel"), OUT("nsel"), OUT("seqused"), OUT("selb") };
static const RadParamSpec pQsaTailStore[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("ring"),
                                              P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaTailStore[] = { OPD("k"), OPD("pos"), OPD("cu"), INOUT("state"),
                                                OPD("sidx") };
/* `work` bounds the blocks that can complete; slots past the real count get page -1. */
static const RadParamSpec pQsaWork[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("ring"),
                                         P_INT("seqs"), P_CAP("work"), P_STR("dtype") };
static const RadOperandSpec oQsaWork[] = { OPD("k"), OPD("pos"), OPD("cu"), OPD("state"),
                                           OPD("sidx"), OPD("bt"), OPD_O("rope"), OUT("stage"),
                                           OUT("page"), OUT("bpos"), OUT("nc") };

/* DFlash2's grouped convolution, libr4d's schema exactly. */
static const RadParamSpec pDflashConv[] = { P_INT("T"), P_INT("hidden_size"), P_INT("taps"),
                                            P_INT("group_size"), P_INT("NG"), P_INT("block_size"),
                                            P_STR("dtype") };
static const RadOperandSpec oDflashConv[] = { OPD("x"), OPD("delta"), WGT("base"), OUT("out") };

/* `mode` "blocks" rotates each of `sections` (pair counts) as its own neox block of channels,
 * reading the position component `comps` names for it; positions are then [C, M]. */
static const RadParamSpec pQkNormRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                            P_INT("n_head_kv"), P_F64("theta"), P_F64("eps"),
                                            P_F64O("scale"), P_STRO("mode"), P_INTO("rotary_dim"),
                                            P_F64O("wadd"), P_INTO("v_norm"), P_STRO("sections"),
                                            P_STRO("comps") };
/* `positions` may be absent at `rotary_dim` 0 (the norm alone), and an absent gain is a unit
 * one. */
static const RadOperandSpec oQkNormRope[] = { INOUT("qkv"), OPD_O("positions"), WGT_O("q_w"),
                                              WGT_O("k_w") };

/* ---- mixture of experts ------------------------------------------------------------------- */
static const RadParamSpec pScaleRows[] = { P_INT("M"), P_INT("n"), P_STR("act"),
                                           P_STR("dtype") };
static const RadOperandSpec oScaleRows[] = { OPD("x"), OPD("s"), OPD_O("add"), OUT("y") };

static const RadParamSpec pRouter[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                        P_INT("norm"), P_STR("dtype") };
static const RadOperandSpec oRouter[] = { OPD("logits"), OUT("expert_ids"), OUT("expert_w") };

/* `expert_base` is this rank's first expert id, making expert parallelism a parameter rather than
 * a second op; absent (0) is the unsharded case. */
static const RadParamSpec pScatter[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                         P_INTO("expert_base"),
                                         P_STRO("protect") };
static const RadOperandSpec oScatter[] = { OPD("expert_ids"), OUT("sorted_tok"),
                                           OUT("expert_offset"), OUT("expert_count") };

/* `a_order` says which rows the activation has; a routed block's two grouped GEMMs differ. gate_up
 * reads the block input, one row per token, so sorted row i gathers `a[sorted_tok[i] / top_k]`.
 * down reads gate_up's output, already one row per (token, slot) in scatter order, so row i is row
 * i.
 *
 * It is stated rather than derived from the row count (M vs M*top_k): an operand whose meaning
 * changes with its extent is invisible to a checker, since a harness sizing it for one reading
 * never exercises the other. */
static const RadParamSpec pMoeGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                         P_INT("top_k"), P_STR("a_order"), P_STR("dtype") };
static const RadOperandSpec oMoeGemm[] = { OPD("a"), WTAB("w"), OPD("sorted_tok"),
                                           OPD("expert_offset"), OUT("y") };

/* The quantised arm is a separate op, like gemm_nt_q, because it adds operands (an activation
 * scale stream and a per-expert weight scale plane) and positional operands cannot vary with a
 * dtype string. `group` is the block edge, 128 for DeepSeek-style fp8 checkpoints. Both weight
 * operands are tables with one entry per expert.
 *
 * Experts are two tables by parity: `w`/`w_scale` the even ones at [N, K], `w_odd`/`w_odd_scale`
 * the odd ones at [N_odd, K_odd], expert e at entry e / 2. A table's entries share one geometry,
 * and a rank holding an uneven slice of each expert's width has two. `parts` stacked parts run
 * along both weight rows and output columns; the output is the wider class's width, and a narrower
 * class fills each part's leading columns and zeros the rest (as in libr4d's schema). */
static const RadParamSpec pMoeGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                          P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                          P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"),
                                          P_INTO("parts") };
static const RadOperandSpec oMoeGemmQ[] = { OPD("a"), OPD_O("a_scale"), WTAB("w"), WTAB("w_scale"),
                                            OPD("sorted_tok"), OPD("expert_offset"),
                                            WTAB("w_odd"), WTAB("w_odd_scale"), OUT("y") };
/* The same op over four classes by expert number mod 4: `w`/`w_scale` class 0 at [N, K],
 * `w_q`/`w_q_scale` class q at [N_q, K_q], expert e at entry e / 4 of class e % 4; how four ranks
 * share the extra block of a five-block expert (as in libr4d's schema). */
static const RadParamSpec pMoeGemmQMod4[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                              P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                              P_STR("dtype"), P_INTO("N_1"), P_INTO("K_1"),
                                              P_INTO("N_2"), P_INTO("K_2"), P_INTO("N_3"),
                                              P_INTO("K_3"), P_INTO("parts") };
static const RadOperandSpec oMoeGemmQMod4[] = { OPD("a"), OPD_O("a_scale"), WTAB("w"),
                                                WTAB("w_scale"), OPD("sorted_tok"),
                                                OPD("expert_offset"), WTAB("w_1"),
                                                WTAB("w_1_scale"), WTAB("w_2"), WTAB("w_2_scale"),
                                                WTAB("w_3"), WTAB("w_3_scale"), OUT("y") };

/* With `shared` and `shared_gate`, the shared expert is folded in: y = bf16(shared * act(gate)) +
 * bf16(gathered), scale_rows' fused gate-and-add. `act` (sigmoid or none) applies to the scalar. */
static const RadParamSpec pMoeGather[] = { P_INT("M"), P_INT("n"), P_INT("top_k"),
                                           P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oMoeGather[] = { OPD("y_expert"), OPD("expert_w"), OPD("sorted_tok"),
                                             OPD_O("shared"), OPD_O("shared_gate"), OUT("y") };

/* `vocab_off` and `pairs` are the vocab-parallel form: `x` is one rank's slice, `idx` carries
 * global columns, and `pairs` is the [M, R, 2] (int32 id, float value) plane an all_gather moves.
 * Must match libr4d's schema, or the loader refuses ref and the model fails to load. */
static const RadParamSpec pRowTopk[] = { P_INT("M"), P_INT("N"), P_INT("R"), P_INTO("vocab_off"),
                                         P_STR("dtype") };
static const RadOperandSpec oRowTopk[] = { OPD("x"), OUT("idx"), OUT("val"), OUT_O("pairs") };

static const RadParamSpec pRowTopkMerge[] = { P_INT("M"), P_INT("R"), P_INT("world_size"),
                                              P_STR("dtype") };
static const RadOperandSpec oRowTopkMerge[] = { OPD("gathered"), OUT("idx"), OUT("val") };
static const RadParamSpec pPairArgmax[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oPairArgmax[] = { OPD("pairs"), OUT("idx"), OUT_O("val") };

static const RadParamSpec pLogitRerank[] = { P_INT("M"), P_INT("R"), P_INT("n_vocab"),
                                            P_INT("n_embd"), P_INTO("vocab_off"), P_STR("dtype") };
static const RadOperandSpec oLogitRerank[] = { OPD("x"), WGT("head"), OPD("idx_in"),
                                              OUT("idx"), OUT("val"), OUT_O("pairs") };

/* ---- gated delta net ----------------------------------------------------------------------
 *
 * `cu` is an operand of every GDN op, decode included: the fast decode kernels read
 * `bos = cu[n]; T = cu[n+1] - bos` before any null test. `num_accepted` is required on both decode
 * ops (spec §10): the state has absorbed the rejected tokens, so a rejection is a change of read
 * offset. A non-speculative decode passes all ones. */
static const RadParamSpec pConvPrep[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                          P_CHUNK("chunk"), P_INT("conv_width"),
                                          P_F64O("l2_eps"), P_F64O("softplus_thr") };
/* Inputs first, outputs last, as docs/OPS.md's resolved table draws. `a` and `b_gate` are separate
 * gate and beta projections; absent, they are the last 2*head_v_count columns of `x`. `cache_idx`
 * is the conv state slot per sequence (a paged cache is not indexed by batch position), and
 * `has_init` says whether that slot holds history. */
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

static const RadParamSpec pKkt[] = { P_INT("M"), P_INT("head_k"), P_CHUNK("chunk") };
static const RadOperandSpec oKkt[] = { OPD("k"), OPD("beta"), OPD("g"), OPD("cu"), OUT("A") };

static const RadParamSpec pChunkScan[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
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

/* gdn_conv_update folded into gdn_recurrent_update: the convolution's inputs, then the
 * recurrence's minus the intermediate q, k and v. Parameters are the union of both ops'. */
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

static const RadParamSpec pGatedNorm[] = { P_INT("M"), P_INT("channels"), P_F64("eps"),
                                           P_STR("act") };
static const RadOperandSpec oGatedNorm[] = { OPD("x"), OPD("z"), WGT("w"), OUT("o") };

/* ---- collectives -------------------------------------------------------------------------- */
static const RadParamSpec pAllReduce[] = { P_INT("world_size"), P_INT("numel"), P_STR("dtype"),
                                           P_INT("exact"), P_INTO("min_bytes"),
                                           P_INTO("hops"), P_INTO("overlap") };
/* `y` is optional; absent means in place. libr4d's two-shot rows need it (see ref_all_reduce). */
static const RadOperandSpec oAllReduce[] = { INOUT("x"), OUT_O("y") };

static const RadParamSpec pAllGather[] = { P_INT("world_size"), P_INT("numel"), P_STR("dtype"),
                                           P_INTO("row") };
static const RadOperandSpec oAllGather[] = { OPD("x"), OUT("y") };

/* ---- vocabulary edges --------------------------------------------------------------------- */
/* `vocab_offset` makes the embedding vocab-parallel, like lm_head. Absent (or 0 with an unsharded
 * table) it is the plain gather.
 *
 * Present, `n_vocab` is this rank's row count (as for logits_gemm) and the table covers global ids
 * [vocab_offset, vocab_offset + n_vocab). Ids stay global; an id outside the slice writes a zero
 * row, as the negative padding id does. Each token lands on exactly one rank, so the following
 * all_reduce reconstructs x exactly.
 *
 * Without it the table is replicated per card: 248320 x 5120 bf16 is 2.43 GiB (bf16 even in the
 * fp8 checkpoint), memory taken from the KV cache. */
static const RadParamSpec pEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                       P_STR("dtype"), P_INTO("vocab_offset"),
                                       /* embed_lookup_q's: the scale its layout quantises with;
                                        * a parameter because a global scale needs the whole
                                        * table and the transform sees one block. */
                                       P_F64O("wscale") };
/* embed_lookup's list: pEmbed plus `mult`, the factor every gathered row is multiplied by (Gemma's
 * sqrt(hidden)). embed_lookup_q has its table scale instead. */
static const RadParamSpec pEmbedM[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                        P_STR("dtype"), P_INTO("vocab_offset"),
                                        P_F64O("wscale"), P_F64O("mult") };
static const RadOperandSpec oEmbed[] = { OPD("tokens"), WGT("wte"), OUT("x") };
/* The quantised table, one scale for all of it; a separate op since it adds an operand. */
static const RadOperandSpec oEmbedQ[] = { OPD("tokens"), WGT("wte"), WGT_O("scale"), OUT("x"),
                                          OPD_O("ahead") };

static const RadParamSpec pLogits[] = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                        P_STR("dtype") };
static const RadOperandSpec oLogits[] = { OPD("x"), WGT("lm_head"), OUT("logits") };

/* ---- sampling -------------------------------------------------------------------------------
 *
 * Every sampling scalar is per row and arrives as an operand. Parameters are geometry, matched and
 * frozen at declare (spec §2.2), so `top_p` as a parameter would allow one value per deployment.
 * `params` is [M] rows of RadSampleParams (abi/rad_sample.h), uploaded once per step.
 *
 * This is the schema core/sample/sampler.h issues against. It supersedes the one docs/OPS.md
 * prints (`rep`/`freq`/`pres`/`k`/`p` as parameters, `seed`/`offset` as extra sample_pick
 * operands); a plugin declaring that form cannot start, since declare refuses per-request values
 * as missing geometry. */
static const RadParamSpec pMVocab[] = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]  = { P_INT("M"), P_INT("n_cand") };
/* The history stages read global token ids and write this rank's columns, so under a split
 * vocabulary they take the rank's first row as `vocab_off` (as sample_topk does). Default 0. */
static const RadParamSpec pPenal[]  = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };

/* `history` is [M, max_hist] int32 of global token ids; row m's window is addressed by row m's
 * hist_off / hist_len within row m. */
static const RadOperandSpec oPenalties[] = { INOUT("logits"), OPD("params"), OPD("history") };
/* `breakers` is optional: [M, max_hist] u8 beside `history`, nonzero where that position's token
 * is a whole DRY sequence breaker and never penalised. Absent, no token is exempt. */
static const RadOperandSpec oDry[]       = { INOUT("logits"), OPD("params"), OPD("history"),
                                             OPD_O("breakers") };
static const RadOperandSpec oTemp[]      = { INOUT("logits"), OPD("params") };
/* `bitmask` is [rows, ceil(n_vocab/32)] uint32, addressed by the row's mask_row -- which is -1 for
 * an unconstrained sequence, so the plane is shared and shorter than M. */
static const RadOperandSpec oMask[]      = { INOUT("logits"), OPD("params"), OPD("bitmask") };
/* `pairs` is the optional vocab-parallel output: [M, 2 * n_cand] u32 of this rank's candidates as
 * (u32 global id, f32 logit), which sample_merge_topk reads after the all-gather. The kernel adds
 * `vocab_off`, so the gather needs no shard layout. */
static const RadParamSpec pTopk[]        = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadOperandSpec oTopk[]      = { OPD("logits"), OPD("params"),
                                             OUT("cand_idx"), OUT("cand_val"), OUT_O("pairs") };
static const RadOperandSpec oArgmax[]    = { OPD("logits"), OPD("params"), OUT("token") };
/* `stats` is [M, 2 + 2k]: max, sum of exp(l - max), then k (id, logit) pairs. `k` is geometry
 * (the sampler declares one width every request's top_logprobs fits); `vocab_off` is as on
 * sample_topk. */
static const RadParamSpec pLogprobs[]    = { P_INT("M"), P_INT("n_vocab"), P_INT("k"),
                                             P_INTO("vocab_off") };
static const RadOperandSpec oLogprobs[]  = { OPD("logits"), OPD("params"), OUT("stats") };
static const RadOperandSpec oNarrow[]    = { INOUT("cand_idx"), INOUT("cand_val"), OPD("params") };
static const RadOperandSpec oPick[]      = { OPD("cand_idx"), OPD("cand_val"), OPD("params"),
                                             OUT("token") };
/* `gathered` is [M, world_size * n_cand] (u32 id, f32 logit) pairs with global token ids, each rank
 * having added its vocab offset before the gather. */
static const RadOperandSpec oMerge[]     = { OPD("gathered"), OPD("params"),
                                             OUT("cand_idx"), OUT("cand_val") };

/* ---- image generation (ref_gen.cpp) --------------------------------------------------------- */
static const RadParamSpec pLnMod[]   = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLnMod[] = { OPD("x"), OPD("s"), OPD("cu_seqlens"), OUT("y") };
/* `act` is the gate's activation, "tanh" (the default) or "none". */
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
/* `s` is [n_seq, >= 2n], [shift | scale] in each row. */
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
/* `t`, `sigma` and `sigma_next` are one f32 value each. */
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
/* `round` 0 keeps the lists and the arithmetic in f32 (default 1, bf16 lists and roundings). */
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

/* ---- video generation (ref_video.cpp). `geom` is the request's latent geometry, i32 on the
 * device: (F, Hp, Wp) for patch_unfold, (frames, H, W) for vtile_gather and (frames, H, W, first,
 * last) for vtile_stitch. */
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
/* `y` is [3, >= M], component-major at its own row pitch. */
static const RadParamSpec pGridPos3[] = { P_INT("M"), P_INT("T"), P_INT("H"), P_INT("W"),
                                          P_INT("period"), P_F64("scale") };
static const RadOperandSpec oGridPos3[] = { OUT("y") };
/* ---- video sparse attention (ref_vsa.cpp). `segs` cuts the rows into dense segments and the
 * video grid, last; `geom` is the request's RadVideoGeom on the device (RAD_VGEOM_*). The two ops
 * share their parameters: attn_vsa recomputes k from `sparsity`. `v` of vsa_select is read for
 * `coarse` alone, and attn_vsa's `gate` comes with its `coarse`. */
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
/* ---- the latent upscaler (ref_video.cpp). `geom` is the request's volume, i32 on the device:
 * (T, H, W) for conv3d_ndhwc and tconv_dw, (T, h, w, H, W) for resize_bilinear and (F, Hp, Wp)
 * for patch_fold, which shares patch_unfold's parameters and operands. */
static const RadParamSpec pConv3d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                        P_INT("kt"), P_STR("dtype"), P_STRO("tpad"),
                                        P_INTO("tpre") };
static const RadOperandSpec oConv3d[] = { OPD("x"), WGT("w"), WGT("b"), OPD("geom"),
                                          OPD_O("res"), OUT("y") };
/* `mod` is one row [scale | shift]; `stats` the per-group mean and rstd, then scratch. */
static const RadParamSpec pGroupNorm[] = { P_INT("M"), P_INT("n"), P_INT("groups"), P_F64("eps"),
                                           P_STR("act"), P_STR("dtype") };
static const RadOperandSpec oGroupNorm[] = { OPD("x"), WGT("w"), WGT("b"), OPD_O("mod"),
                                             INOUT("stats"), OUT("y") };
static const RadParamSpec pTconvDw[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_STR("dtype") };
static const RadOperandSpec oTconvDw[] = { OPD("x"), WGT("w"), WGT("b"), OPD("geom"), OUT("y") };
static const RadOperandSpec oResize[] = { OPD("x"), OPD("geom"), OUT("y") };
/* `sigma` is one f32 value, `seed` two u32 words, low first. */
static const RadParamSpec pFlowRenoise[] = { P_INT("M"), P_INT("n"), P_INT("stream"),
                                             P_STR("dtype") };
static const RadOperandSpec oFlowRenoise[] = { OPD("sigma"), OPD("seed"), INOUT("x") };

/* ---- LTX-2 (ref_ltx.cpp) */
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

/* ---- audio generation (ref_audio.cpp) */
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
static const RadParamSpec pIntAffine[] = { P_INT("M"), P_STR("src"), P_STR("mul"), P_STR("add") };
static const RadOperandSpec oIntAffine[] = { OPD("x"), OUT("y") };
static const RadOperandSpec oCuRepeat[] = { OPD("len"), OUT("y") };

/* ================================================================== the schema table */
static const RadOpSchema kSchemas[] = {
{ "rmsnorm", ARR(pRmsnorm), ARR(oRmsnorm),
  "y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. `wadd` is not in docs/OPS.md's elementwise table: "
  "it is GemmaRMSNorm's `1 +`, which Qwen3.5, Qwen3-Next and Gemma all need, and it defaults to 0. "
  "Spelled `wadd` and not `w_add` because that is what libr4d's fused norms call it and one "
  "vocabulary cannot have two spellings of one parameter. `mult` (default 1) scales the row "
  "before its one rounding." },
{ "rmsnorm_p", ARR(pLayernorm), ARR(oUnary),
  "y = x * rsqrt(mean(x^2) + eps); an rmsnorm with NO gain tensor at all. wadd is 1 so the result "
  "is a plain normalisation. Gemma's attention normalises the value with a bare RMS (a weightless "
  "ggml_rms_norm) and the key with the k_norm gain in the same layer, which one op with a required "
  "gain cannot express." },
{ "rmsnorm_add", ARR(pRmsnormAdd), ARR(oRmsnormAdd),
  "residual_out = x + residual, then y = rmsnorm(residual_out) * w. The variance is taken from "
  "the unrounded f32 sum; the multiply reads residual_out back AFTER it has been narrowed, which "
  "is what every fused implementation does." },
{ "hc_enter", ARR(pHcWrite), ARR(oHcEnter),
  "h[s] = x for each of the hc streams -- the wide residual stream's first value. Shares `hc_write`'s parameter list." },
{ "mhc_pre", ARR(pMhcPre), ARR(oMhcPre),
  "Manifold hyper-connection read: RMS-scaled projection of the whole hc*n residual, sigmoid "
  "pre/post gates and Sinkhorn mixing. x has dtype, post and comb are f32. comb is indexed "
  "[source,destination]. First pass is row softmax + hc_eps then column normalization; later "
  "passes add hc_eps to both normalization denominators. hc is 2..8; iterations is 1..256." },
{ "mhc_pre_partial", ARR(pMhcPrePartial), ARR(oMhcPrePartial),
  "mhc_pre's projection split by residual columns. Chunk c covers columns [c*chunk, (c+1)*chunk) "
  "of the hc*n row; for chunks [first, first+chunks) part[m, (c-first)*S + j] is the sum over the "
  "chunk of h*fn[j] for j < hc*(hc+2) and of h*h for j = hc*(hc+2), with S = hc*(hc+2)+1 rounded "
  "up to a multiple of 4 and the slots past the last zero. chunk divides hc*n. Tensor-parallel "
  "ranks each compute their chunks and an all_gather (row = chunks*S) joins them." },
{ "mhc_pre_finish", ARR(pMhcPreFinish), ARR(oMhcPreFinish),
  "mhc_pre from mhc_pre_partial's matrix over all hc*n/chunk chunks: each coefficient and the sum "
  "of squares is the sum of its chunks' partials (the order is the implementation's), then "
  "mhc_pre's gates, Sinkhorn mixing and collapse of h. part and mhc_pre_partial over every chunk "
  "together are mhc_pre." },
{ "mhc_post", ARR(pHcWrite), ARR(oMhcPost),
  "out[dst,i] = post[dst]*x[i] + sum_src comb[src,dst]*h[src,i]. post and comb are f32; "
  "out must not alias h. No normalization of the residual stream." },
{ "mhc_contract", ARR(pHcWrite), ARR(oMhcContract),
  "x[i] = mean over hc streams h[stream,i], rounded once to dtype." },
{ "kda_update", ARR(pKdaUpdate), ARR(oKdaUpdate),
  "L2-normalized KDA: per-channel decay exp(lower_bound*sigmoid(exp(A_log)*(a+dt_bias))), "
  "beta=sigmoid(b), S*=decay; S+=beta*(v-S*k)*k; o=scale*S*q. State is f32 [slot,heads,V,K]. "
  "verify reads state_idx[seq,num_accepted-1] and writes each candidate state_idx[seq,t]; "
  "prefill reads/writes column zero, committing only the final state. cu offsets are absolute; "
  "has_init=0 starts from zero and a negative initial slot emits zero output. No output norm." },
{ "hc_read", ARR(pHcRead), ARR(oHcRead),
  "The read half of a gated residual. N = grouped_rmsnorm(h) -- one rsqrt per `n`-wide stream, "
  "one gain of hc*n over all of them, `w + wadd`; g = sigmoid(mix_up @ silu(mix_down @ N / hc)); "
  "x = mean over the hc streams of g * N. With `inject_w`, inj[s] = 2*sigmoid(inject_w[s] . N / hc) "
  "-- the per-branch gains hc_write consumes. With `q`, x is also emitted block-scaled fp8, "
  "quantised from the NARROWED x and not from the accumulator. With `rotate` (and `group`), `rq` "
  "and `rscale` carry had_quant_act_fp8 of the same stored x: each scale group rotated by an "
  "unnormalised Hadamard before its absmax, for a consumer whose weight carries the rotation. "
  "`mix` names the mixing matrices' STORED format, \"bf16\" or \"e4m3\"; the arithmetic is the "
  "same and an oracle reads either as the dense tensor its kernel's inverse returns." },
{ "hc_write", ARR(pHcWrite), ARR(oHcWrite),
  "The write half: h[s] += inj[s] * y, for each of the hc streams of h. In place, and `h` is the "
  "UNNORMED stream the paired hc_read was handed -- the normalisation never enters the residual." },
{ "hc_project", ARR(pHcProject), ARR(oHcProject),
  "The control vector's projection (llama.cpp's `--cvec-mode project`): for each of the hc "
  "streams of a row whose `on` flag is 1, h[s] -= scale * (h[s] . v) * v, with v that layer's unit "
  "direction and scale the norm the file stored. The dot is per stream, not on the stream mean. A "
  "row whose flag is 0 is not read and not written, so an unprojected request is byte-identical to "
  "a deployment with no vector loaded." },
{ "mtp_enter", ARR(pMtpEnter), ARR(oMtpEnter),
  "The MTP head's entry for a wide residual stream: x[k*n+i] = fc_h[i] . grouped_norm(h)[k] "
  "+ fc_e[i] . norm(e), for each of the hc sub-streams k. `ngroup` == hc is one rsqrt per "
  "sub-stream (what every gated residual in this architecture does) and 1 is a single rsqrt "
  "over the whole wide row; the checkpoint fits both and the acceptance rate decides. `fc` names "
  "the two matrices' STORED format, \"bf16\" or \"e4m3\"." },
{ "layernorm", ARR(pLayernorm), ARR(oLayernorm),
  "y = (x - mean) * rsqrt(var + eps) * w + b, biased variance. `b` is optional." },
{ "grid_embed", ARR(pGridEmbed), ARR(oGridEmbed),
  "x[m] += table resampled at row m's grid position: `table` is a side x side grid of rows, "
  "`coord` is [4, M] -- row, column, grid height, grid width -- and the position is sampled "
  "bilinearly with the grid's corners on the table's corners (align_corners), edge taps clamped. "
  "The taps are summed in f32 and rounded to x's dtype, then added to x and rounded again." },
{ "grid_embed_aa", ARR(pGridEmbed), ARR(oGridEmbed),
  "x += the side x side table resampled as torch's F.interpolate(bilinear, align_corners=False, "
  "antialias=True) resamples it to the row's grid -- SigLIP2's position embedding: per axis a "
  "triangle filter on half-pixel centres, widened by the scale when shrinking, its in-table taps "
  "normalised. Separable, f32, rounded once before the add; coordinates clamp into their grid." },
{ "add", ARR(pAdd), ARR(oBinary),
  "y = a + b. An operand carrying exactly n elements against a taller output broadcasts along "
  "rows; nothing else broadcasts. With `div`, y = rd(rd(a + b) / div), rd y's dtype: the last "
  "sum of a mean taken pairwise." },
{ "mul", ARR(pMnD), ARR(oBinary), "y = a * b, with the same one broadcast rule as `add`." },
{ "table_add", ARR(pTableAdd), ARR(oTableAdd),
  "y[r, l n + c] = x[r, c] + t[l, c]: a shared modulation row plus each of L learned rows, f32 sum "
  "rounded once; `M` is y's rows." },
{ "chan_stats", ARR(pChanStats), ARR(oChanStats),
  "a latent through per-channel statistics held as weights: `denorm` y = rd(rd(x std) + mean), "
  "`norm` y = rd(rd(x - mean) / std); std is `s`, or sqrt(s + eps) with `var` 1, each statistic "
  "rounded to y's dtype." },
{ "mix_rows", ARR(pMixRows), ARR(oMixRows),
  "y[r, c] = sum_i w[i] x[r k + i, c], i < k: a Linear(k, 1) across k consecutive rows, f32 sum in "
  "ascending i rounded once; `M` is y's rows." },
{ "mul_weight", ARR(pMulWeight), ARR(oMulWeight),
  "y = x * w, `w` a WEIGHT of one element (a scalar for the whole output) or of one row "
  "(broadcast down the rows): a learned gain on an activation. f32 product, rounded once." },
{ "act_mul", ARR(pActMul), ARR(oBinary),
  "y = act(a) * b, `act` \"silu\", \"gelu\" (exact erf), \"gelu_tanh\" or \"sigmoid\". The activation is "
  "rounded to y's dtype before the product, which is the unfused pair. `a` and `b` may be column "
  "slices of one fused row." },
{ "seg_mean", ARR(pSegMean), ARR(oSegMean),
  "y[s] = mean of x's rows [cu[s], cu[s+1]): one row a segment, summed in double and rounded "
  "once to y's dtype. An empty segment is a zero row." },
{ "pool_rows", ARR(pPoolRows), ARR(oUnary),
  "Row r of y is the mean of x's rows [r*k, r*k + k), summed in f32 and rounded to y's dtype, then "
  "times `mult` (default 1) and rounded again. M, the input rows, is a multiple of k." },
{ "relu", ARR(pMnD), ARR(oUnary), "y = max(x, 0), each operand at its own row pitch." },
{ "clamp", ARR(pMnD), ARR(oClamp),
  "y = min(max(x, lo), hi), `lo` and `hi` WEIGHTS of one element each: a clipped linear's range." },
{ "conv2d", ARR(pConv2d), ARR(oConv2d),
  "A 2-D convolution over (time, width), channels innermost: x[t, w*cin + c] -> y[t, w*cout + c], "
  "w [cout, cin, kh, kw] ([cout, cin, kw, kh] with t_last). Each cu_seqlens segment (rows "
  "cu/cu_div) is convolved on its own with zero padding `pad` at its edges; its output rows start "
  "at cu/(cu_div*stride). With `valid`, input row t reads as zeros unless valid[t * valid_stride] "
  "!= 0. f32 sum plus the optional bias `b`, one rounding; then `act` (none, gelu) rounded." },
{ "conv2d_unfold", ARR(pConv2dUnfold), ARR(oConv2dUnfold),
  "conv2d's receptive fields as rows: row (to, wo) holds, per input channel and kernel tap, what "
  "conv2d multiplies that tap's weight by (zero in the padding and on masked rows), its columns "
  "in the weight's order -- (ci, i, j), or (ci, j, i) with t_last -- so a GEMM against the weight "
  "as [cout, cin*kh*kw] is the convolution. Segments, cu_div and valid as conv2d's." },
{ "conv1d_dw", ARR(pConv1dDw), ARR(oConv1dDw),
  "Causal depthwise convolution over time: y[t, c] = sum_j w[c, 0, j] * x[t - (k-1) + j, c], rows "
  "before the segment's start (cu/cu_div) read as zeros, plus the optional bias b[c]. f32 sum, "
  "one rounding." },
{ "rel_sinusoid", ARR(pRelSin), ARR(oRelSin),
  "Row d: sin(d * inv[i]) then cos(d * inv[i]) over n/2 timescales, inv[i] = exp(i * "
  "-log(max_timescale) / max(n/2 - 1, 1)), each step in f32, rounded once." },
{ "seg_sinusoid", ARR(pSegSin), ARR(oSegSin),
  "Row r of segment s (rows cu/cu_div) takes position p = (r - cu[s]/cu_div) mod period: sin(p * "
  "inv[i]) then cos(p * inv[i]) over n/2 timescales, inv as rel_sinusoid's, each step in f32, "
  "rounded once. Rows no segment covers are left alone." },
{ "attn_local_rel", ARR(pAttnLocalRel), ARR(oAttnLocalRel),
  "Local attention with a relative-position term (Gemma 4's audio): query i of a segment sees "
  "keys j with 0 <= i - j <= left (`right` 0). qf = q * head_dim^-0.5/ln2 * softplus(pds); logit "
  "= qf.k * ln(1+e)/ln2 + qf.relk[i-j]; softcap * tanh(logit / softcap); softmax; .v. f32, one "
  "rounding." },
{ "mla_attn", ARR(pMlaAttn), ARR(oMlaAttn),
  "Softmax over selected latent vectors used as both key and value; per-query page tables and visible lengths." },
{ "mla_project", ARR(pMlaProject), ARR(oGemm),
  "Per-head query absorption or value expansion through the shared [head,(K|V),latent] BF16 matrix." },
{ "dsa_score", ARR(pDsaScore), ARR(oDsaScore),
  "Per-head ReLU of FP8 dot products, then signed head gates and scales; invalid history is -inf." },
{ "dsa_query", ARR(pMnD), ARR(oQuantFp8),
  "Normalized Hadamard, BF16 boundary and power-of-two E4M3 quantization per query head." },
{ "dsa_pool", ARR(pDsaPool), ARR(oDsaPool),
  "Channel-wise softmax pooling with learned position bias, BF16, normalized Hadamard, BF16, power-of-two FP8." },
{ "swiglu_clamped", ARR(pSwigluClamped), ARR(oGateUp),
  "silu(min(gate,limit))*clamp(up,-limit,limit); gate is the first half." },
{ "silu_mul", ARR(pMnD), ARR(oGateUp),
  "y[i] = silu(gate_up[i]) * gate_up[n + i]. The GATE IS THE FIRST HALF, which is the order "
  "rad-convert fuses gate and up into one weight in." },
{ "gelu_mul", ARR(pMnD), ARR(oGateUp),
  "y[i] = gelu_tanh(gate_up[i]) * gate_up[n + i], the tanh approximation (gemma's "
  "gelu_pytorch_tanh), and the GATE IS THE FIRST HALF as in silu_mul. Its own op, not gelu, because "
  "gated feed-forwards are trained with the tanh form and the unary gelu is the exact erf one." },
{ "gelu", ARR(pMnD), ARR(oUnary),
  "y = 0.5x(1 + erf(x/sqrt(2))). The exact erf form, not the tanh approximation." },
{ "silu", ARR(pMnD), ARR(oUnary), "y = x * sigmoid(x)." },
{ "sigmoid", ARR(pMnD), ARR(oUnary), "y = 1 / (1 + exp(-x))." },
{ "fill", ARR(pFill), ARR(oFill),
  "y = value, every element of an [M, n] extent (default 0), rounded once to y's dtype. The op a "
  "rank issues where it computes nothing and a reader downstream -- an all-reduce, a residual -- "
  "still expects a defined buffer." },
{ "logit_softcap", ARR(pMnCap), ARR(oUnary), "y = cap * tanh(x / cap); the final-logit softcap." },
{ "scale", ARR(pMnD), ARR(oScale),
  "y = x * s, s a [1] weight broadcast over every element; the per-tensor scalar (per-layer scalar, "
  "`mul` broadcasts a row, `scale_rows` a column-per-row)." },
{ "ple_gate", ARR(pPleGate), ARR(oPleGate),
  "Per-Layer Embedding's gate. K = grouped_rmsnorm(k) * (w_key + wadd) and Q likewise over `q`, "
  "one rsqrt per `n`-wide stream and one gain across all hc*n; s[c] = sum_d K[c][d]*Q[c][d] / "
  "sqrt(n); then the SIGNED SQUARE ROOT s = sign(s) * sqrt(max(|s|, gate_eps)), where the sign is "
  "the ORIGINAL's so that a dot of exactly zero stays zero; then gv[c] = sigmoid(s[c]) * v, one v "
  "row gated differently per stream. `gvn` is grouped_rmsnorm(gv) * (w_conv + wadd), taken over "
  "the bytes gv was narrowed to. BOTH outputs leave: the convolution that follows reads the normed "
  "copy and the residual add reads the un-normed one." },
{ "ple_conv", ARR(pPleConv), ARR(oPleConv),
  "y[t] = resid[t] + silu(sum_j w[c][j] * x[t - (width-1-j)*dilation]), a DEPTHWISE CAUSAL "
  "convolution whose taps are `dilation` apart -- at width 4 and dilation 3 they are 9, 6, 3 and 0 "
  "timesteps back, so the history a sequence carries is (width-1)*dilation and not width-1. The "
  "state is [n_slots, n, state_len] with slot i holding the value at relative position "
  "i - (width-1)*dilation, and `has_init` 0 means that window is ZEROS rather than whatever the "
  "slot last held. `num_accepted` shifts the READ offset within a deeper window and marks a "
  "decode step: the window is then rewritten as gdn_conv_update rewrites its own -- the history "
  "read shifted down by one, then every input of the step -- so a next step that kept k of this "
  "step's tokens reads at k - 1 and finds the history ending at the k-th; it needs "
  "(width-1)*dilation - 1 slots plus one a token. Absent, every token commits: the read is at "
  "offset zero and the last (width-1)*dilation inputs land at offset zero, which is the prefill "
  "chunk. `resid` absent drops the add and leaves the plain convolution." },
{ "causal_conv1d", ARR(pCausalConv), ARR(oCausalConv),
  "y[t] = sum_j w[c][j] * x[t - (width-1-j)], a DEPTHWISE CAUSAL convolution with no activation "
  "and no residual (LFM2's short convolution, whose gates the caller applies). The state, "
  "`has_init`, `cache_idx` and `num_accepted` are ple_conv's at dilation one: a window of width-1 "
  "values, read at `num_accepted - 1` and rewritten shifted by one with every input of a decode "
  "step behind it, or left as the chunk's last width-1 inputs when `num_accepted` is absent." },
{ "ngram_ids", ARR(pNgramIds), ARR(oNgramIds),
  "ids[m][h] = remainder(hash of the n-gram ending at m, vocab_sizes[h]) + offsets[h], where "
  "hash = t0*mult[0] ^ t1*mult[1] ^ ... over as many shifted tokens as head h's block needs -- "
  "heads are `ngram - 1` equal blocks, block b using the (b+2)-gram. A shift that would cross an "
  "EOS reads the EOS instead, and so does every deeper shift. `state` is a rolling window of the "
  "committed ids in front of the step -- ple_conv's contract exactly, read at `num_accepted - 1` "
  "and rewritten in the decode layout when `num_accepted` is present and the prefill layout when "
  "it is not -- except that a cold window is EOS and not zero, because zero is a real token. The "
  "gather that follows is `embed_lookup_q` over M * heads indices." },
{ "gather_rows", ARR(pMnD), ARR(oGather),
  "y[i, :] = x[idx[i], :]. `M` is a RANGE, so the rows gathered are `y`'s extent and not the "
  "declared worst case. A negative index writes a zero row, as embed_lookup's padding does; an "
  "index at or past x's row count is RAD_E_INVAL, because reading it would be reading another "
  "sequence's hidden state." },
{ "scatter_rows", ARR(pMnD), ARR(oScatterRows),
  "x[idx[i], :] = v[i, :]. The mirror of `gather_rows`, and `x` is INOUT because the write is "
  "PARTIAL: rows no index names keep what they held, which is what makes it usable against a pool "
  "other sequences are in. A negative index writes nothing, mirroring the zero row `gather_rows` "
  "reads for one. Indices MUST be distinct -- the schema says so with RAD_OPD_F_IDX_UNIQUE -- "
  "because two rows landing on one destination has no defined winner on a parallel machine, and "
  "an op whose oracle and whose kernel may legitimately disagree is not checkable." },
{ "cast", ARR(pCast), ARR(oUnary),
  "y = x, converted. `from` and `to` are for the selector; the bytes are described by the "
  "tensors' own dtypes and those are what ref reads." },
{ "softmax", ARR(pMnD), ARR(oUnary), "Row-wise, f32, with the maximum subtracted." },

{ "quant_act_i8", ARR(pQuant), ARR(oQuant),
  "Symmetric int8 with ONE SCALE PER ROW, max(amax, 1e-8) / 127, and codes x / scale rounded to "
  "nearest even, stored in the STEP ORDER the int8 WMMA reads (the middle two dwords of every "
  "16-byte step exchanged; n a multiple of 16). The activation of the w4a8, w8a8 and w2a8 GEMMs. "
  "`asum`, if passed, gets each row's code sums per `group` columns, group-major "
  "[n/group][ceil(M/16)*16]; rows past M are not written." },
{ "quant_act_i8g", ARR(pQuant), ARR(oQuantFp8),
  "quant_act_fp8 at int8 codes: per `group` contiguous elements a scale amax / 127 (1 for an "
  "all-zero group) and codes rounded to nearest even, row-major, scale [M, ceil(n/group)]. The "
  "activation of the i8a8 GEMMs." },
/* Not one of docs/OPS.md's original ops: ref adopts the schema libr4d/r4d_rows.cpp fixed, and
 * implements it so the production fp8 activation format has an oracle. */
{ "quant_act_fp8", ARR(pQuant), ARR(oQuantFp8),
  "Symmetric E4M3 per group of `group` contiguous elements: scale = amax / 448 (a DEQUANT "
  "multiplier, the same sense as a checkpoint's weight_scale_inv), q = clamp(x / scale, +-448), "
  "scale shaped [M, ceil(n/group)] in f32. An all-zero group gets scale 1, not scale 0: the GEMM "
  "multiplies this scale by the weight's block scale, and a zero here would erase that one. Rows "
  "are dim 0, so a rank-3 activation is one row per token and not one per head." },
{ "quant_cols_fp8", ARR(pQuantCols), ARR(oQuantCols),
  "E4M3 codes with one scale per column over all M rows: scale[c] = amax_r |x[r, c]| / 448 (1 for "
  "an all-zero column), y = clamp(x / scale, +-448). An attention's values for its fp8 P.V, where "
  "a column's scale multiplies out of the sum over keys." },
{ "rmsnorm_quant_fp8", ARR(pRmsQuantFp8), ARR(oRmsQuantFp8),
  "RMS norm with an optional residual add, then the E4M3 quantiser, in one pass. The residual "
  "add is STORED BEFORE THE VARIANCE READS IT, so the sum the norm is taken over is the bf16 "
  "one and not the f32 one -- that is what makes this the `add` + `rmsnorm` pair and not "
  "`rmsnorm_add`, which normalises the unrounded sum. `wadd` is added to every gain in f32. "
  "`out_bf16` holds exactly the value the codes were taken over." },
{ "gated_quant_fp8", ARR(pGatedQuantFp8), ARR(oGatedQuantFp8),
  "act(gate) * up over a packed [M, 2n] plane, gate half first, then the E4M3 quantiser. "
  "`act` is silu (down_proj's input) or sigmoid (a gated attention o_proj's) and defaults to "
  "silu, which is what docs/OPS.md's `silu_mul` names." },
{ "gemm_nt_q_gated", ARR(pGemmQGated), ARR(oGemmQGated),
  "gemm_nt_q over a [N, K] gate_up plane with gated_quant_fp8's pass in the epilogue, so the "
  "[M, N] bf16 tensor between them is never written. `N` is the WEIGHT's rows, twice the fused "
  "width; the output is [M, N/2]." },
{ "dflash_select", ARR(pDflashSelect), ARR(oDflashSelect),
  "A greedy walk over a candidate lattice: score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * "
  "hp[l][r] * succ[cand[l, c]][r], where pid(0) is the anchor token and pid(l) the walk's own "
  "previous pick. `hp` absent is a plane of ones, which drops the form to a plain low-rank "
  "bigram. Serial in l by construction -- the predecessor is not known until the previous step "
  "has been decided." },
{ "givens_dense_quant", ARR(pGivensDense), ARR(oGivensDense),
  "ParoQuant's activation quantiser with the rotation as a dense per-group matrix: y = T (x / "
  "rscale) for each of `parts` rotated copies, T the [128, 128] f16 matrix a (part, 128-column "
  "group) that its Givens stages compose to, then the block-scaled quantiser `out` names "
  "(\"i8\": scale amax / 127, round to nearest; \"fp8\": E4M3, amax / 448). q is [M, parts * n] "
  "and scale [M, parts * n / group]; tdense is [parts * n, 128]. The same activation as "
  "the sequential rotation, up to the f16 rounding of T." },
{ "givens_rotate", ARR(pGivensRotate), ARR(oGivensRotate),
  "ParoQuant's rotation without a quantiser: y = x / rscale rotated by `stages` rounds of Givens "
  "pairs, `parts` copies side by side ([M, parts * n]) in y's own dtype. For a weight whose GEMM "
  "reads a 16-bit activation. `group` is the width a pair stays inside." },
/* The rotated fp8 twin. Unlike had_quant_act_i8, the group is both rotation width and scale
 * granularity, and the factor of `group` is left entirely to the paired w4a8h weight plane, where
 * the division is an exact exponent shift on a bf16 (libref/ref_quant.cpp). */
{ "had_quant_act_fp8", ARR(pQuant), ARR(oQuantFp8),
  "A Walsh-Hadamard rotation of width `group` along n, then symmetric E4M3 per group: scale = "
  "amax(rotated) / 448, shaped [M, n/group] in f32. n must be a multiple of `group` -- a rotation "
  "block cannot straddle the end of a row. NOT interchangeable with quant_act_fp8 in either "
  "direction: the consumer must be a weight that was rotated the same way." },
/* Its int8 twin, the activation of the i8a8h GEMMs; the factor of `group` is the weight's here
 * too. */
{ "had_quant_act_i8g", ARR(pQuant), ARR(oQuantFp8),
  "quant_act_i8g with a Walsh-Hadamard rotation of width `group` along n first: scale = "
  "amax(rotated) / 127 per group, [M, n/group] in f32, n a multiple of `group`. Pairs with an i8 "
  "weight rotated the same way, its scale carrying the factor of `group` (dtype i8a8h)." },
/* The calibration op, the only one no forward pass issues: it runs under a calibration flag and
 * its output is read at convert time, in another process, through a file. `dtype` is the input's:
 * fp8a8 where the tap is a quantiser's output, bf16 without `a_scale` for a plain activation. */
{ "gram_accum", ARR(pQuant), ARR(oGram),
  "h[k1][k2] += sum over rows m of dequant(a)[m][k1] * dequant(a)[m][k2], in f32. The second "
  "moment of a layer's real input over a calibration corpus -- the Hessian a GPTQ packer needs. "
  "dequant(a) is a * a_scale for fp8a8 and `a` itself for bf16, which passes no a_scale. "
  "`h` is [n, n] and INOUT: it is READ and added to, not stored. NOT gemm_nt with a transposed "
  "operand; that op contracts the last axis of both sides and this one contracts the first." },
{ "had_quant_act_i8", ARR(pQuant), ARR(oHadQuant),
  "A Walsh-Hadamard rotation of width `group` along n, then symmetric int8 with ONE SCALE PER ROW "
  "carrying the 1/sqrt(group) normalisation. Here `group` is the rotation width, not the scale "
  "granularity -- see libref/ref_quant.cpp." },
{ "rmsnorm_had_quant_i8", ARR(pRmsHadQuant), ARR(oRmsHadQuant),
  "residual add (if `residual` is present), rmsnorm, the width-`group` rotation, then per-row "
  "int8. The fused form a w4a8 layer asks for before falling back to the unfused ops. `residual` "
  "is INOUT and receives x + residual; `out_bf16` is the post-norm tensor for a layer that has a "
  "bf16 consumer of it as well." },
{ "gated_had_quant_i8", ARR(pGatedQuant), ARR(oGatedQuant),
  "act(a) * b, then the rotation and the int8 quantiser. `b` ABSENT MEANS THE PACKED FORM: `a` is "
  "[M, 2n] and b is its second half, which is what the MLP actually produces. `act` is \"silu\" "
  "(down_proj's input) or \"sigmoid\" (a gated-attention o_proj's), and `mode` is libr4d's integer "
  "spelling of the same choice -- 0 silu, 1 sigmoid. `act` wins where both are given." },
{ "dequant", ARR(pDequant), ARR(oDequant),
  "y = q * scale, the inverse of quant_act_i8g. It does NOT undo a rotation." },

{ "gemm_nt", ARR(pGemm), ARR(oGemm), "y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate." },
{ "gemm_nt_f32", ARR(pGemm), ARR(oGemm),
  "BF16 activations times F32 weights, F32 accumulation and output; preserves routing and index gate precision." },
{ "gemm_nt_bias", ARR(pGemmBias), ARR(oGemmBias),
  "gemm_nt with a per-column bias added in f32, rounded to y's dtype; then, optionally, `act` of "
  "that (\"gelu\" exact, \"gelu_tanh\" the tanh approximation, \"none\"), rounded; then `res` "
  "added, rounded. Each step is what a separate tensor would hold." },
{ "gemm_nt_q", ARR(pGemmQ), ARR(oGemmQ),
  "gemm_nt with quantised operands. The group sub-sum is accumulated from the raw codes and "
  "scaled once per group, which is what an int8 or fp8 WMMA kernel does. Scale streams are "
  "[rows, groups] or [rows]; an mxfp4 weight's u8 scale stream is read as OCP e8m0." },
{ "gemm_nt_q_bias", ARR(pGemmQBias), ARR(oGemmQBias), "gemm_nt_q with a per-column bias." },
{ "lora_shrink", ARR(pLoraShrink), ARR(oLoraShrink),
  "t[m, j] = sum_k x(m, k) * lora_a[s, j, k] for each row m whose s = slot[m] is in [0, S), f32 "
  "accumulate, k ascending; x(m, k) is the value gemm_nt_q uses for that element of `a` (a code "
  "times its group's scale from `a_scale`, a float as it is). A row outside [0, S) is a base row "
  "and its `t` row is not written. A row's bytes do not depend on M or on any other row." },
{ "lora_expand", ARR(pLoraExpand), ARR(oLoraExpand),
  "y[m, n] = y[m, n] + scale[s, n] * sum_j t[m, j] * lora_b[s, n, j] for each row m whose s = "
  "slot[m] is in [0, S), f32 accumulate, j ascending, one rounding to y's dtype. A base row's `y` "
  "is neither read nor written, so it stays bit for bit what the GEMM wrote (not y + 0: that "
  "turns -0 into +0)." },
{ "lora_expand_mm", ARR(pLoraExpand), ARR(oLoraExpand),
  "lora_expand with t rounded to bf16 and the sum over j in any order: y[m, n] = y[m, n] + "
  "scale[s, n] * sum_j bf16(t[m, j]) * lora_b[s, n, j], f32 accumulate, one rounding to y's dtype. "
  "A base row's `y` is neither read nor written. A row's bytes may depend on M: the generators' "
  "expand, whose passes are many rows of one adapter." },

{ "attn_paged", ARR(pAttnPaged), ARR(oAttnPaged),
  "Paged attention over kv_cache[num_blocks, kv_heads, block_size, 2*head_dim] (K then V per "
  "slot), f32 softmax with the max subtracted, causal and sliding-window masking, q_len >= 1 so "
  "one kernel serves decode, speculative verify and prefill. seqused is the TOTAL key count "
  "including this step's query rows. k_descale / v_descale are optional and apply to an fp8 "
  "cache. `scale` is not in docs/OPS.md and defaults to 1/sqrt(head_dim)." },
{ "attn_dense", ARR(pAttnDense), ARR(oAttnDense),
  "Dense varlen attention over cu_seqlens segments (rows cu/cu_div). Nothing paged. `window` is "
  "attn_paged's rule over a segment's positions: causal, i - j < window; non-causal, |i - j| < "
  "window, or i - j < window and j - i < window_right with `window_right`; 0 none. `block` cuts "
  "a segment into blocks of that many rows from its start, each attending within itself; 0 "
  "none." },
{ "attn_cross", ARR(pAttnCross), ARR(oAttnCross),
  "attn_dense with each entry's keys a run of their own: queries rows [cu[s], cu[s+1]), keys and "
  "values rows [cu_k[s], cu_k[s+1]); with `causal` a query sees key j of its run when j <= its own "
  "index, both counted from their run's start. A query that sees no key is zero." },
{ "attn_na3d", ARR(pAttnNa3d), ARR(oAttnNa3d),
  "3-D neighborhood attention (NATTEN's na3d) over the voxels of geom = (T, H, W): a query sees the "
  "kt x kh x kw window centred on it, shifted inward at the border, each extent at most its axis." },
{ "kv_store", ARR(pKvStore), ARR(oKvStore),
  "Writes k and v into the paged cache at slot_mapping; a negative slot is skipped. The cache is "
  "INOUT rather than the OUT docs/OPS.md draws, because the op touches only the mapped slots." },
{ "rope_table", ARR(pRopeTable), ARR(oRopeTable),
  "cos/sin for the positions a step names, at each position's own row of a [max_ctx, rot] plane: "
  "rot values a row, cos for the first half and sin for the second. The angles are `rope`'s "
  "exactly. A row no position names is not written; a negative position writes nothing." },
{ "qsa_block_key", ARR(pQsaBlockKey), ARR(oQsaBlockKey),
  "QSA's compressed key, one vector per `ratio` rows of k: the block's mean accumulated in f32 and "
  "rounded to k's dtype, RMS-normed with gain (wadd + w), then NeoX-rotated over `rotary_dim` at "
  "the block's position -- `pos[b]` (with `mode` mrope/imrope and `sections`, a [>= M, 3] list's "
  "three components) or `pos0 + b*ratio` -- with YaRN's blend and magnitude when given. `page[b]`, "
  "when given, is the output row; a negative entry writes nothing." },
{ "qsa_score", ARR(pQsaScore), ARR(oQsaScore),
  "score[m][b] = (sum over heads of relu(q[m][h] . bk[bt[seq(m)][b]])) / sqrt(n) for the sequence's "
  "first min(nc, width) blocks; the relu is per (block, head) and the sum over heads, and a negative "
  "page id scores -inf. Columns past the bound are not written." },
{ "qsa_select", ARR(pQsaSelect), ARR(oQsaSelect),
  "The topk highest-scoring complete blocks a query has passed, as page ids in block order: ties to "
  "the lower block index, then the partial page when "
  "(pos + 1) % ratio != 0, then -1 to topk + 1. nsel counts the entries and seqused is the "
  "attention's length over them." },
{ "qsa_tail_store", ARR(pQsaTailStore), ARR(oQsaTailStore),
  "state[sidx[s][0]][pos[t] % ring] = k[t] for the last `ring` tokens of each sequence; the other "
  "rows of the tail keep what they held." },
{ "qsa_work", ARR(pQsaWork), ARR(oQsaWork),
  "The blocks completing in this batch: per block its `ratio` raw keys (this step's rows, or the "
  "tail's at q % ring) staged contiguously, its block-key page and its first-token position; slots "
  "past the count get page -1. nc is each sequence's complete-block count." },
{ "dflash_conv", ARR(pDflashConv), ARR(oDflashConv),
  "DFlash2's grouped dynamic depthwise convolution: out[t][h] = sum over taps j, j == 0 or "
  "t % block_size >= j, of (base[j][h] + delta[t][j][h / group_size]) * x[t - j][h]." },
{ "rope", ARR(pRope), ARR(oRope),
  "Rotary embedding in place over the q and k heads of a fused [q|k|v] row; the v heads are left "
  "alone. `mode` is \"neox\" or \"gptj\", or a multi-component form: \"mrope\" (contiguous "
  "`sections` of frequencies, one ladder), \"imrope\" (Qwen3-VL's interleaved sections) or "
  "\"axial\" (contiguous sections, each on its own ladder -- a 2-D rotary). `positions` is [M], "
  "or [C, M] component-major for a multi-component mode; with [M] every component is that one. "
  "An f32 `positions` holds the positions as they are (fractional coordinates), an integer one "
  "integers. "
  "`scale` divides the position; `rotary_dim` is optional and defaults to head_dim. `llama3_*` "
  "applies Llama 3's frequency scaling (factor, low and high frequency factors, the original "
  "length)." },
{ "qk_norm_rope", ARR(pQkNormRope), ARR(oQkNormRope),
  "Per-head RMS norm of q and k against their own gains, then rope. THE NORM OUTPUT IS ROUNDED "
  "TO THE TENSOR'S DTYPE BEFORE THE ROTATION, which is what the unfused pair does. `v_norm` 1 "
  "also RMS-norms each v head with a unit gain, unrotated. An absent gain is a unit gain "
  "without `wadd`; `rotary_dim` 0 is the norm alone, `positions` then optional. `mode` "
  "\"axial\" is rope's axial (its section ladders also over rank-1 positions), and f32 "
  "positions are read as rope reads them." },

{ "scale_rows", ARR(pScaleRows), ARR(oScaleRows),
  "y[m, i] = x[m, i] * act(s[m]): one scalar per ROW, broadcast ACROSS the row. `mul` broadcasts "
  "the other way -- one row down the rows -- and giving it both rules would make them ambiguous "
  "when rows == n. `act` is \"sigmoid\" or \"none\", and it applies to the SCALAR. With `add` "
  "present the form is y = bf16(x * act(s)) + add -- the product NARROWED before the sum, which "
  "is what makes the fused form bit-identical to the pair it replaces." },
{ "router_sigmoid_topk", ARR(pRouterSigmoid), ARR(oRouterSigmoid),
  "Single-group sigmoid routing; correction bias selects experts, original scores weight them." },
{ "router_topk", ARR(pRouter), ARR(oRouter),
  "Softmax over ALL n_expert logits, then the top_k by probability with ties to the lower expert "
  "id, then -- if `norm` -- renormalise the kept weights to sum to one." },
{ "moe_scatter", ARR(pScatter), ARR(oScatter),
  "Stable counting sort of the (token, slot) pairs by expert. sorted_tok holds the FLATTENED "
  "index token*top_k + slot, not the bare token id; a slot whose expert is outside the range this "
  "call owns is dropped and the tail of sorted_tok is filled with -1. THE RANGE IS "
  "[expert_base, expert_base + n_expert), which is how EXPERT PARALLELISM is expressed: under it "
  "the router stays replicated and picks from every expert in the model, each rank declares and "
  "sorts only its own slice, and the slots that went elsewhere simply contribute nothing to this "
  "rank's partial -- which the block's end-of-block all-reduce then sums. `expert_base` absent "
  "or 0 with n_expert the model's count is the whole-model case." },
{ "moe_gemm", ARR(pMoeGemm), ARR(oMoeGemm),
  "y[i] = a[row(i)] @ w[e]^T in SORTED order, e being the expert whose offset range contains i. "
  "`a_order` picks row(i): \"token\" gathers a[sorted_tok[i] / top_k] from an activation with one "
  "row per token, \"sorted\" reads row i of one that is already in the scatter's order. `w` is a "
  "WEIGHT TABLE of n_expert entries, each [N, K] -- a layer's experts are separate movement "
  "units, so base-plus-stride describes only the case where none was offloaded." },
{ "moe_gemm_q", ARR(pMoeGemmQ), ARR(oMoeGemmQ),
  "moe_gemm with quantised operands, `a_order` included: an activation scale stream per row per "
  "group of K -- with a row for every row `a` has -- and one "
  "weight scale plane per expert shaped [ceil(N/group), ceil(K/group)]. The group sub-sum is "
  "accumulated from the raw codes and scaled once, exactly as `gemm_nt_q` does. The experts are "
  "two tables by parity: `w` the even ones at [N, K], `w_odd` the odd ones at [N_odd, K_odd], "
  "expert e at entry e/2. The output is max(N, N_odd) wide in `parts` stacked parts, and a "
  "narrower class fills each part's leading columns and zeros the rest." },
{ "moe_gemm_q_mod4", ARR(pMoeGemmQMod4), ARR(oMoeGemmQMod4),
  "moe_gemm_q over FOUR weight-table classes by the expert's number mod 4: `w` class 0 at [N, K], "
  "`w_q` class q at [N_q, K_q], expert e at entry e/4 of class e%4. The output is as wide as the "
  "widest class in `parts` stacked parts, and a narrower class fills each part's leading columns "
  "and zeros the rest." },
{ "moe_gather", ARR(pMoeGather), ARR(oMoeGather),
  "Scatters the sorted expert output back and weights it. The sum over a token's slots runs in "
  "ASCENDING SLOT order, which is what makes it reproducible." },
{ "row_topk", ARR(pRowTopk), ARR(oRowTopk),
  "Exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every column it "
  "reports, so a rank's slice reports GLOBAL ids; `pairs` is the [M, R, 2] (int32 id, float value) "
  "plane an all_gather moves, with id -1 for a slot the row could not fill." },
{ "row_topk_merge", ARR(pRowTopkMerge), ARR(oRowTopkMerge),
  "The R largest of world_size all-gathered per-rank candidate sets, in the same descending "
  "(idx, val) form the single-rank top-R produces. `gathered` is [M, world_size * R, 2] pairs, "
  "rank r's at [r*R, (r+1)*R); id -1 is a pad and is dropped. Ties break on the LOWER id, which is "
  "what makes the merged answer agree with a top-R over the unsharded plane." },
{ "pair_argmax", ARR(pPairArgmax), ARR(oPairArgmax),
  "idx[m] = the id of the largest value among pairs[m, 0..n), each pair an int32 id in the first "
  "f32 slot and its value in the second (row_topk's `pairs`, all-gathered): ties to the lower "
  "id, a negative id skipped, -1 and -inf where nothing is valid. row_topk_merge at R = 1, "
  "written one element a row at the output's own pitch." },

{ "logit_rerank", ARR(pLogitRerank), ARR(oLogitRerank),
  "Rescore R candidate ids a row EXACTLY against the full-precision head and re-sort on those "
  "values: the second half of a coarse top-R. `idx_in` carries GLOBAL ids as row_topk reports "
  "them, `vocab_off` is this rank's first column, and an id outside its rows becomes a pad. Ties "
  "break on the LOWER id, the same rule row_topk and row_topk_merge use, so a tie inside a shard "
  "and a tie between shards resolve alike." },

{ "gdn_conv_prep", ARR(pConvPrep), ARR(oConvPrep),
  "The gated-delta-net prefill preamble: depthwise causal conv with silu and its state cache, the "
  "q/k/v split, the l2 norm over each q and k head, the gate with its per-chunk cumsum, and beta. "
  "`a` and `b_gate` carry the gate and beta projections; absent, they are the last 2*head_v_count "
  "columns of the fused `x` row [q|k|v|a|b]. `cache_idx` names the conv-state slot per sequence "
  "-- without it the state is indexed by the batch's sequence number, which a paged conv cache "
  "cannot be -- and a zero entry in `has_init` means that slot holds no history, so the window in "
  "front of the sequence's first token is zeros rather than whatever the slot last held." },
{ "gdn_conv_update", ARR(pConvUpdate), ARR(oConvUpdate),
  "The same convolution for a decode step against the rolling window: reads at num_accepted-1 and "
  "rewrites the buffer shifted down by one, so a speculative rejection is a change of read offset "
  "rather than a recompute. `num_accepted` is REQUIRED, because an op that lets the caller omit it "
  "is an op whose rollback contract is optional (spec §10); a non-speculative decode passes ones." },
{ "gdn_kkt_solve", ARR(pKkt), ARR(oKkt),
  "A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, by forward substitution in "
  "f32. The decay is applied per element, which cannot overflow at any gate span." },
{ "gdn_chunk_scan", ARR(pChunkScan), ARR(oChunkScan),
  "The chunked scan: W = beta(V - e^g K S^T), V' = A W, O = scale e^g Q S^T + sA V', "
  "S = e^{g_last} S + V'^T(e^{g_last-g} K). h0 and ht are [N, heads, head_v, head_k] and both are "
  "optional-ish: a null h0 is a zero initial state." },
{ "gdn_recurrent_update", ARR(pRecurrent), ARR(oRecurrent),
  "The decode recurrence against the paged state cache: the initial state comes from "
  "state_idx[s, num_accepted-1] and every candidate token writes its own slot. The gated RMS norm "
  "is folded into the epilogue, and so is the fp8 out-projection quantiser where o_q and o_scale "
  "are present: one group per (token, value head), because head_v is the fp8 block width. It "
  "quantises the bf16 `o` it just wrote and not the f32 behind it, which is what makes it agree "
  "to the byte with a standalone quant_act_fp8 over the same output. "
  "`state_form` says WHAT THE STATE CACHE MEANS and defaults to \"per_candidate\", which is what "
  "this description and ref describe: the initial state comes from state_idx[s, num_accepted-1] "
  "and every candidate token writes its own slot, so a step drafting n needs 1 + n distinct ones. "
  "\"anchor\" is a second contract the same op serves -- one committed state a sequence with the "
  "accepted prefix replayed onto it out of a scratch slot, two slots whatever the depth -- and it "
  "produces the SAME `o` from a COMPLETELY DIFFERENT `state`. A reference that implements one and "
  "is handed the other computes a correct answer to the other question, which is why the form is "
  "declared rather than inferred from the index row's width. ref serves per_candidate and "
  "DECLINES anchor; libr4d serves both. "
  "`cu` bounds the sequences exactly as it does on the sibling "
  "gdn_conv_update, which runs off the same tensor on the same step." },
{ "gdn_conv_recurrent_update", ARR(pConvRecurrent), ARR(oConvRecurrent),
  "gdn_conv_update followed by gdn_recurrent_update, with q, k and v never written: the operands "
  "are the convolution's (`conv_idx` is its state_idx) and then the recurrence's without those "
  "three. The rolling-window and state contracts are the two ops' own." },
{ "gdn_gated_rmsnorm", ARR(pGatedNorm), ARR(oGatedNorm),
  "o = x * rsqrt(mean_channels(x^2) + eps) * w * act(z). `act` is \"silu\" or \"sigmoid\"." },

{ "all_reduce", ARR(pAllReduce), ARR(oAllReduce),
  "Sum across ranks, f32, IN ASCENDING RANK ORDER so every rank gets the same bits. world_size 1 "
  "is a no-op. On the host backend the ranks are threads in one address space, so this is a "
  "rendezvous rather than a transfer. exact=0 is served exactly, so `min_bytes` -- the byte "
  "floor under which a lossy row must serve the message exactly anyway -- is accepted and "
  "ignored, as is `hops`, and so is `overlap` (the caller issues it beside compute). "
  "`y` absent is the in-place form; present, the sum lands in y and x is left alone." },
{ "all_gather", ARR(pAllGather), ARR(oAllGather),
  "y is [world_size * numel] with rank r's contribution at offset r * numel. `row` -- one row of "
  "a rank's contribution, in elements -- switches to the ROW-INTERLEAVED placement instead: x is "
  "[rows][row] and y is [rows][world_size][row], so one row of y carries every rank's words for "
  "that row. The vocab-parallel candidate merge reads exactly that." },

{ "embed_lookup", ARR(pEmbedM), ARR(oEmbed),
  "A gather from wte[n_vocab, n_embd]. A negative token id writes a zero row (padding); an id at "
  "or above n_vocab fails the launch. `mult`, when given, multiplies every gathered row and the "
  "product is rounded once to x's dtype (Gemma's sqrt(hidden)); absent it is a plain copy." },
{ "embed_lookup_q", ARR(pEmbed), ARR(oEmbedQ),
  "The same gather from a table quantised with ONE scale for all of it: x[m] = wte[id] * scale[0]. "
  "Qwen3.8-Flash-Next-FP8's 320-million-row n-gram embedding is stored that way -- E4M3 codes and a "
  "single bf16 `weight_scale` -- and it can be, because E4M3 carries its own exponent and a row's "
  "dynamic range is 2.3x. Padding and bounds behave exactly as embed_lookup's." },
{ "logits_gemm", ARR(pLogits), ARR(oLogits),
  "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T. Separate from gemm_nt because "
  "lm_head is its own access class and the sampler wants to fuse against it." },

{ "sample_penalties", ARR(pPenal), ARR(oPenalties),
  "Repetition (multiplicative, sign-dependent), frequency and presence penalties over the row's "
  "history. Applied once per DISTINCT token, with the count carrying the multiplicity. "
  "penalty_last_n windows the TAIL of the history; 0 turns the stage off for the row. History ids "
  "are global and a token's column is its id less vocab_off." },
{ "sample_dry", ARR(pPenal), ARR(oDry),
  "DRY: each token that would extend a suffix of the window which has occurred before is "
  "penalised ONCE, by multiplier * base^(L - allowed_length) for the longest such repeat L, "
  "repeats capped at the row's dry_rep_limit. Only lengths at or above allowed_length count, and "
  "a token the optional `breakers` plane marks is never penalised. dry_penalty_last_n windows the "
  "TAIL of the history; 0 turns the stage off for the row. Columns are ids less vocab_off." },
{ "sample_temp", ARR(pMVocab), ARR(oTemp),
  "logits /= temp, per row. temp <= 0 leaves the row untouched -- greedy decoding is "
  "sample_argmax, and a temperature of zero silently meaning something else is how a sampler "
  "chain becomes unreadable." },
{ "sample_topk", ARR(pTopk), ARR(oTopk),
  "The row's top_k largest logits, descending, ties to the lower token id. cand_val carries "
  "LOGITS. top_k <= 0 or wider than the buffer means the whole candidate width; a narrower one "
  "leaves index -1 / -infinity holes, which every later stage skips." },
{ "sample_topp", ARR(pMCand), ARR(oNarrow),
  "Narrows the candidate set in place to the shortest prefix whose cumulative probability reaches "
  "top_p, always keeping at least one. A dropped candidate is index -1, value -infinity. The "
  "candidates are assumed to arrive descending, as sample_topk leaves them." },
{ "sample_minp", ARR(pMCand), ARR(oNarrow),
  "Keeps candidates whose probability is at least min_p times the most likely one's." },
{ "sample_typical", ARR(pMCand), ARR(oNarrow),
  "Locally typical: keeps the candidates whose surprisal is closest to the distribution's entropy "
  "until their mass reaches typical_p. NOT a prefix -- the candidate order is by probability and "
  "this order is by |surprisal - H| -- so it is a threshold on the distance, ties kept." },
{ "sample_xtc", ARR(pMCand), ARR(oNarrow),
  "Exclude top choices: with probability xtc_probability, DROP every candidate at or above "
  "xtc_threshold except the least likely of them. The opposite of every other stage, and the "
  "point. A no-op unless at least two candidates clear the threshold." },
{ "sample_mask", ARR(pMVocab), ARR(oMask),
  "A SET BIT MEANS ALLOWED. Word w bit b covers token 32w + b; a cleared bit sets -infinity. The "
  "row's mask_row selects the plane row, or -1 for an unconstrained sequence." },
{ "sample_pick", ARR(pMCand), ARR(oPick),
  "Inverse-CDF draw over the candidate set. The uniform is splitmix64 over "
  "(seed * 0x9E3779B97F4A7C15 + pos), taking the top 24 bits as a float in [0,1) -- specified "
  "rather than left to a library, because the same stream has to come out of a host reference, a "
  "device kernel and a rerun." },
{ "sample_merge_topk", ARR(pMCand), ARR(oMerge),
  "The vocab-parallel merge: the k largest of the all-gathered per-rank candidate sets, in the "
  "same descending form the single-rank path produces. Token ids in `gathered` are already "
  "global." },
{ "sample_argmax", ARR(pMVocab), ARR(oArgmax), "Greedy, ties to the lower token id." },
{ "sample_logprobs", ARR(pLogprobs), ARR(oLogprobs),
  "For a row with RAD_SP_LOGPROBS: the raw logits' max, sum of exp(l - max) and k largest as (id, "
  "logit) pairs, descending, ties to the lower id. Other rows are not written." },
/* ---- image generation -------------------------------------------------------------------- */
{ "ln_mod_quant_fp8", ARR(pLnModQ), ARR(oLnModQ),
  "the DiT block's norm in front of a quantised linear, in one pass: with `gated` 1 the branch "
  "before it is added first, x = bf16(x + bf16(bf16(tanh(g[seq])) * y)); then z = ln_mod(x) and "
  "(q, scale) = quant_act_fp8(z, group) -- written when bound, z." },
{ "ln_mod", ARR(pLnMod), ARR(oLnMod),
  "y = bf16(bf16(LN(x)) * bf16(1 + s[seq])): a layer norm with no gain or bias, then the row's "
  "sequence's modulation row of `s` ([n_seq, >= n] at its own pitch). A diffusion transformer's "
  "adaptive norm without a shift." },
{ "gate_add", ARR(pGateAdd), ARR(oGateAdd),
  "x = bf16(x + bf16(bf16(tanh(g[seq])) * y)): a residual added through its sequence's gate row; "
  "with `act` none the gate is g itself, x = bf16(x + bf16(g[seq] * y))." },
{ "timestep_embed", ARR(pTimestep), ARR(oTimestep),
  "the sinusoidal timestep embedding, [cos | sin] of t' * exp(-ln(max_period) * i / (n/2)) with "
  "t' = scale * t; `round` 1 takes t through bf16 at scale and back first, as diffusers' "
  "pipeline hands it to its transformer." },
{ "gather_rows2", ARR(pMnD), ARR(oGather2),
  "y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]]: a sequence assembled from two sources." },
{ "lane_gather", ARR(pLane), ARR(oLaneGather),
  "y[cu[e] + k] = src[lane[e] * stride + k]: each entry's rows out of its lane of a per-image "
  "state." },
{ "lane_store", ARR(pLane), ARR(oLaneStore),
  "dst[l * stride + k] = src[k] for every lane l listed: one source's rows into several lanes." },
{ "randn_lanes", ARR(pLane), ARR(oRandnLanes),
  "an image's first step: its lane's rows become its seed's standard-normal stream "
  "(abi/rad_philox.h), element (k, c) the stream's k * n + c; entries without flag 1 are left." },
{ "flow_euler", ARR(pLane), ARR(oFlowEuler),
  "one Euler step of a flow-matching sampler per entry that moves a lane: guidance against the "
  "paired entry, x' = bf16(x + bf16(bf16(sigma_next - sigma) * v)), a preview of the clean "
  "latent with flag 4, the kept region of a repaint restored with flag 2 (docs/OPS.md)." },
{ "kd_input", ARR(pLane), ARR(oKdInput),
  "a k-diffusion sampler's model input per entry that moves a lane: on a first step the lane "
  "becomes its seed's normal stream times sqrt(1 + sigma^2); y = rd(x / sqrt(sigma^2 + 1)) at the "
  "entry's rows and at its pair's (docs/OPS.md)." },
{ "kd_step", ARR(pKdStep), ARR(oKdStep),
  "one k-diffusion sampler step over a discrete-time denoiser per entry that moves a lane: the "
  "clean estimate D from an eps or v output, guided against the pair, then Euler, Euler ancestral "
  "or DPM++ 2M by the entry's sampler, the lane f32; flag 4 previews D, `hist` keeps it "
  "(docs/OPS.md)." },
{ "chan_affine", ARR(pChanAffine), ARR(oUnary),
  "a per-channel affine from two parameter lists of n values (rounded to bf16): `fma` is "
  "bf16(bf16(x * mul) + add), `inv` bf16(bf16(x - add) / mul). `round` 0 keeps the lists and "
  "the arithmetic in f32, rounded once into y: rd(f32(x * mul) + add), rd(f32(x - add) / mul)." },
{ "conv2d_nhwc", ARR(pConvNhwc), ARR(oConvNhwc),
  "an image convolution, channels innermost: x [Hin, Win * cin] and y [Hout, Wout * cout] dense, "
  "the sizes read off them; PyTorch's [cout, cin, k, k] weight and a bias; `up` 2 reads the input "
  "upsampled nearest-neighbour; `pad` same (k/2 a side) or down (0 above and left, 1 below and "
  "right)." },
{ "im2col_nhwc", ARR(pIm2col), ARR(oUnary),
  "conv2d_nhwc's receptive fields as rows: x [Hin, Win * cin], y [Hout * Wout, cin * k * k], row "
  "oy * Wout + ox and column (ci * k + i) * k + j the value conv2d_nhwc multiplies weight[o, ci, "
  "i, j] by (0 in the padding), so y @ w.reshape(cout, cin * k * k)^T + b is the convolution; "
  "`up`, `pad` and `stride` as conv2d_nhwc reads them." },
{ "l2norm_chan", ARR(pL2Norm), ARR(oL2Norm),
  "each row normalised to unit L2 (eps 1e-12), times sqrt(n), times gamma -- three bf16 roundings "
  "-- then `act` none or silu: an image decoder's channel norm." },
{ "d2s_add", ARR(pSpaceDepth), ARR(oSpaceDepth),
  "y [2H, 2W * cout] += x [H, W * cin] repeated and dealt into 2x2 blocks: an upsampling "
  "shortcut (Wan's DupUp3D on one frame; ft 2 keeps the second frame)." },
{ "s2d_avg_add", ARR(pSpaceDepth), ARR(oSpaceDepth),
  "y [H, W * cout] += the 2x2 blocks of x [2H, 2W * cin] dealt into channels and averaged in "
  "groups: a downsampling shortcut (Wan's AvgDown3D on one frame; ft 2 counts the zero frame)." },
{ "to_rgba8", ARR(pMnD), ARR(oUnary),
  "decoded pixels in [-1, 1] to bytes as diffusers' postprocess makes them: "
  "round(clamp(bf16(bf16(clamp(x) / 2) + 0.5), 0, 1) * 255), u8 out." },
{ "gelu_tanh", ARR(pMnD), ARR(oUnary), "the tanh approximation of gelu." },
{ "gelu_quick", ARR(pMnD), ARR(oUnary),
  "CLIP's quick_gelu, three roundings: a = rd(1.702 x), s = rd(sigmoid(a)), y = rd(x s)." },
{ "scatter_add_rows", ARR(pMnD), ARR(oScatterRows),
  "x[idx[i]] += v[i], rounded to x's dtype; a negative index adds nothing." },
{ "attn_prefix_self", ARR(pAttnPS), ARR(oAttnPS),
  "attention over two key sources: `seqused[s]` prefix positions in the paged cache and the "
  "step's own rows `k`/`v` beside `q`; self row i sees every prefix key and its own rows j <= i "
  "with `causal`, all of them without. `qk` \"i8\" with `k_i8` [tokens, kv_heads * head_dim] and "
  "`k_scale` [tokens, kv_heads] selects the int8 Q.K form: they are the own keys as quant_act_i8g "
  "writes them at group head_dim (scale amax/127, 1 for an all-zero row; code "
  "rint(clamp(x * (1/scale), +-127))), the query and each prefix key are quantised by the same "
  "rule per (row, head), and a score is the exact integer dot product times both scales. `k` is "
  "then not read for the own rows." },
/* ---- video generation -------------------------------------------------------------------- */
{ "rms_mod", ARR(pRmsMod), ARR(oRmsMod),
  "a DiT's adaptive RMS norm with a shift: with `gated` 1 first x = bf16(x + bf16(g[seq] * y)) "
  "(no tanh), stored; then n1 = bf16(x * rsqrt(mean(x^2) + eps)), n2 = bf16(n1 * w), "
  "m = bf16(n2 * bf16(1 + scale[seq])), z = bf16(m + shift[seq]), `s` [n_seq, >= 2n] holding "
  "[shift | scale] at its own pitch." },
{ "rms_mod_quant_fp8", ARR(pRmsModQ), ARR(oRmsModQ),
  "rms_mod with the norm chain in bf16, then (q, scale) = quant_act_fp8(z, group) as "
  "ln_mod_quant_fp8 makes them; z written when bound." },
{ "flow_x0_step", ARR(pMnD), ARR(oFlowX0),
  "a sampler step toward a data-ward velocity's clean estimate, in f32: st = 1 - t, "
  "d = x + st v, r = sigma_next / sigma, x = r x + (1 - r) d, each product and sum rounded "
  "once; t, sigma and sigma_next one value each (MiniMax-H3's scheduler)." },
{ "patch_unfold", ARR(pPatchUnfold), ARR(oPatchUnfold),
  "patch rows [F Hp Wp, C pt ph pw] (geom = (F, Hp, Wp)) unfolded to voxel rows "
  "[(F pt) (Hp ph) (Wp pw), C] and denormalised, f32(f32(x * mul[c]) + add[c]) with `mul` and "
  "`add` lists of C f32 values." },
{ "vtile_gather", ARR(pVtileGather), ARR(oVtileGather),
  "a video decoder's tiles out of voxel rows [frames H W, n] (geom = (frames, H, W)): each axis "
  "split into tile-wide tiles at least `overlap` apart (diffusers' _split_tiles), tile k at rows "
  "[k period, (k + 1) period) of y as its frames tile tile voxels, then n_reg rows of `reg`, then "
  "zeros." },
{ "vtile_stitch", ARR(pVtileStitch), ARR(oVtileStitch),
  "a decoded chunk's tiles stitched into frames (each tile's kept range, blended with the tile "
  "above and then the tile to the left, both as decoded), the first `blend` kept frames blended "
  "with `prev` unless the chunk is first, prev refilled from the chunk's tail (also emitted "
  "when it is last), then bytes round(255 clamp(v std + mean, 0, 1)); geom = (frames, H, W, "
  "first, last), u8 out." },
{ "grid_pos3", ARR(pGridPos3), ARR(oGridPos3),
  "a video decoder's rotary positions over tiles of `period` rows: token (t, i, j) of a tile's "
  "T x H x W gets f32(scale) * (2 (c + 0.5) / n - 1) on each axis, in f32 as torch makes it, "
  "the rest of the period 0; y f32 [3, >= M] component-major at its own row pitch." },
{ "vsa_select", ARR(pVsa), ARR(oVsaSelect),
  "video sparse attention's tile selection. Tiles: each dense segment of `segs` in chunks of "
  "tile_t tile_h tile_w rows (P prefix tiles), then tile_t x tile_h x tile_w boxes over the "
  "F x Hp x Wp video grid (geom's frames, height / patch, width / patch), box (a, b, c) tile "
  "P + (a nb + b) nc + c (V of them). S = f32 tile means of q and k, dotted, times scale; a video "
  "query tile's sel[h][i][0, k) is its k = max(1, min(ceil((1 - sparsity) V), V)) (f64) video "
  "tiles of largest S, ties to the lower, ascending. With `coarse` (and `v`): softmax over all "
  "tiles of S times the tile means of v, rounded once." },
{ "attn_vsa", ARR(pAttnVsa), ARR(oAttnVsa),
  "video sparse attention: a prefix query tile sees every tile, a video one the prefix tiles and "
  "its k tiles of `sel` (k ascending video tiles, RAD_E_INVAL otherwise); f32 softmax with the "
  "max subtracted. With `coarse` and `gate`: out = rd(rd(o) + coarse[h][tile] gate), rd the "
  "output's dtype." },
{ "conv3d_ndhwc", ARR(pConv3d), ARR(oConv3d),
  "a 3-D convolution over voxel rows [T H W, cin] -> [T H W, cout] (geom = (T, H, W)): "
  "PyTorch's Conv3d weight [cout, cin, kt, k, k] and bias, stride 1, `same` zero padding (kt/2 "
  "frames, k/2 rows and columns a side; k and kt odd), f32 accumulation, rd(acc + b); with `res`, "
  "rd(rd(acc + b) + res), `res` possibly y itself." },
{ "group_norm", ARR(pGroupNorm), ARR(oGroupNorm),
  "PyTorch's GroupNorm over all M rows: group g is channels [g n/groups, (g+1) n/groups), its "
  "biased mean and variance in f64, rstd = 1 / sqrt(var + eps) in f32; v = (x - mean) rstd w + b, "
  "with `mod` (one row [scale | shift]) v (1 + scale) + shift, `act` none or silu, one rounding. "
  "stats[2g], stats[2g+1] get the mean and rstd; stats holds groups (2 + 3 1024) f32, the rest "
  "scratch. y may be x." },
{ "tconv_dw", ARR(pTconvDw), ARR(oTconvDw),
  "a depthwise convolution along the frames of voxel rows (geom = (T, H, W)): PyTorch's "
  "[n, 1, k, 1, 1] weight and bias, k odd, zero frames outside [0, T), the sum f32 in tap order, "
  "rd(acc + b)." },
{ "resize_bilinear", ARR(pMnD), ARR(oResize),
  "each frame of voxel rows [T h w, n] resized to [T H W, n] (geom = (T, h, w, H, W)) as torch's "
  "bilinear with align_corners=False at a size: src = max(f32(h / H) (i + 0.5) - 0.5, 0), its "
  "floor and the next row clamped, weights in f32, rounded once." },
{ "patch_fold", ARR(pPatchUnfold), ARR(oPatchUnfold),
  "patch_unfold's inverse: voxel rows [(F pt) (Hp ph) (Wp pw), C] folded to patch rows "
  "[F Hp Wp, C pt ph pw] (geom = (F, Hp, Wp)) and normalised, f32(f32(x * mul[c]) + add[c]) with "
  "`mul` and `add` lists of C f32 values." },
{ "flow_renoise", ARR(pFlowRenoise), ARR(oFlowRenoise),
  "a clean latent re-noised to sigma s: x = f32(f32(f32(1 - s) x) + f32(s z)), z element "
  "k n + c of the normal stream (seed, `stream`) (abi/rad_philox.h)." },
{ "rope_split", ARR(pRopeSplit), ARR(oRopeSplit),
  "LTX-2's rotary over fractional coordinates: dim/2 pairs over all heads, the first `pad` "
  "unrotated, then freq[k] = f32(theta^linspace(0,1,steps)[k] pi/2) with the n_dims coordinates "
  "interleaved; angle f32(f32(2 f32(c / max)) - 1) freq; each head's halves rotated against each "
  "other in f32, rounded to x's dtype. `head0` is the operand's first head." },
{ "head_gate", ARR(pHeadGate), ARR(oHeadGate),
  "x = rd(x rd(2 rd(sigmoid(g)))) per head: LTX-2's gated attention output." },
{ "pixel_norm", ARR(pPixelNorm), ARR(oPixelNorm),
  "an RMS norm over each row's n values as bf16 tensors compute it, every step rounded to y's "
  "dtype: m = rd(sum rd(x^2) / n), then x rd(rsqrt(rd(m + eps))) (`form` rsqrt) or "
  "x / rd(sqrt(rd(m + eps))) (div), optionally silu; `ileave` L writes x's row t L + l to "
  "columns c L + l of y's row t." },
{ "d2s3d", ARR(pD2s3d), ARR(oD2s3d),
  "depth to space over voxel rows: packed channel ((c st + a) sh + b) sw + d of x's row (t, i, j) "
  "to y's (t st + a - (st - 1), i sh + b, j sw + d), the first st - 1 frames dropped." },
{ "s2d3d", ARR(pS2d3d), ARR(oS2d3d),
  "space to depth over voxel rows, `tpre` leading frames prepended: each st x sh x sw box a packed "
  "row (`mode` copy), or each run of `group` packed channels averaged (mean); `add` adds into y." },
{ "pix_patch", ARR(pPixPatch), ARR(oPixPatch),
  "an image [H, W C] to patch rows [H/p, (W/p) C p p], channel c's (ph, pw) at c p p + pw p + ph "
  "(order wh) or c p p + ph p + pw (hw)." },
{ "patch_pix8", ARR(pPixPatch), ARR(oPixPatch),
  "patch rows [R, w C p p] to RGB bytes [R p, w p C] by to_rgba8's rule, pix_patch's channel "
  "order." },
{ "vol_index", ARR(pVolIndex), ARR(oVolIndex),
  "a volume's voxel coordinates (t, i, j), component-major: attention's rotary positions." },
{ "tile_grid", ARR(pTileGrid), ARR(oTileGrid),
  "the tiles of a grid as diffusers cuts them (`rule` stride: tiled_decode's, a tile every "
  "`stride` cells; intervals: _tile_intervals', a short last remnant merged), in space and "
  "optionally time, one row (t, h, w, t0, y0, x0, k, count) a tile." },
{ "tile_crop", ARR(pTileCrop), ARR(oTileCrop),
  "one tile's voxels out of a volume: y[(a, i, j)] = x[(t0 + a, y0 + i, x0 + j)], box "
  "(t, h, w, t0, y0, x0)." },
{ "tile_stitch", ARR(pTileStitch), ARR(oTileStitch),
  "decoded tiles to RGB bytes: diffusers' in-place blend_v then blend_h of each tile with the one "
  "above and the one to its left, then blend_t of each time band with the one before, each kept to "
  "its stride, then patch_pix8's bytes." },
{ "stft_logmel", ARR(pStftMel), ARR(oStftMel),
  "each channel's causal log-mel spectrogram from a windowed DFT basis and a mel filterbank, every "
  "stored tensor rounded to y's dtype: frames S / hop, columns h n_mel + m." },
{ "bwe_up", ARR(pBweUp), ARR(oBweUp),
  "y = clamp(rd(res + up(x)), -1, 1): x raised `ratio` times by a Hann-windowed sinc resampler "
  "(width ceil(6 / 0.99), taps in x's dtype, edges replicated)." },
{ "ltx_guide", ARR(pLtxGuide), ARR(oLtxGuide),
  "LTX-2's sampler step: each pass's clean estimate x - v sigma, the guidance deltas (CFG, STG, "
  "modality) added in f32, an optional rescale by the deviations' ratio, the velocity back and "
  "an Euler step of x, all f32. `scales` f32[4] (gs, stg, ms, rescale), `sched` f32[2]." },
{ "conv1d", ARR(pConv1d), ARR(oConv1d),
  "a 1-D convolution over the sequences `cu_seqlens` bounds, channels innermost: x [rows, cin], "
  "y [rows, cout], PyTorch's [cout, cin, k] weight and an optional bias, dilation `dil` and padding "
  "dil (k - 1) / 2 a side (k odd), or dil (k - 1) on the left with `causal` 1, zeros outside each "
  "sequence; `res` added after the bias, then `act` none, tanh or clamp (to [-1, 1])." },
{ "conv_t1d", ARR(pConvT1d), ARR(oConvT1d),
  "a transposed 1-D convolution of kernel `k` >= stride (2 stride by default), padding `pad` "
  "((k - stride) / 2 by default, 0 the causal form): each sequence of x [rows, cin] grows "
  "`stride` times into y [rows * stride, cout], output row o reading the input rows i with "
  "0 <= o + pad - i stride < k; PyTorch's [cin, cout, k] weight and a bias." },
{ "snake", ARR(pSnake), ARR(oSnake),
  "the snake activation per channel: x + sin(alpha x)^2 / (alpha + 1e-9), or with `beta` "
  "x + (1 / (beta + 1e-9)) sin(alpha x)^2; `log` 1 stores both as logarithms." },
{ "snake_aa", ARR(pSnakeAa), ARR(oSnakeAa),
  "BigVGAN's alias-free SnakeBeta over each sequence, f32: replicate-padded, upsampled `ratio` "
  "times by the transposed `f_up` filter, SnakeBeta, replicate-padded and filtered by `f_down` "
  "at stride `ratio` (docs/OPS.md has the paddings and crops); `log` 1 reads alpha and beta "
  "as logarithms." },
{ "cfg_topk_sample", ARR(pCfgTopk), ARR(oCfgTopk),
  "one pick a lane from `rep` branch rows of logits (1 or 2, conditional first): guidance "
  "u + cfg (c - u) (`cfg` [n] per lane when bound), a repetition `penalty` over the lane's "
  "history `hist` (cleared at frame 0, the pick recorded), ids in [sup_lo, sup_hi) never "
  "candidates, with `cond_k` > 0 only the entries where c reaches its cond_k-th largest, the k "
  "largest kept (ties to the lower index), over `temp`, softmax, drawn by inverse CDF with "
  "rad_philox_uniform_s(seed, 2^32 + frame, stream); written to every one of the lane's slots." },
{ "rvq_embed", ARR(pRvqEmbed), ARR(oRvqEmbed),
  "a frame's embedding from code-major residual codes [n_codes, rows]: t0[c0 + off0] plus the sum "
  "of t1[c_k + (k - 1) vocab], times `scale`, rounded as bf16(bf16(t0 + bf16(sum)) * scale), or "
  "with `round` 1 one f32 sum in codebook order times scale, rounded once." },
{ "frame_mix_store", ARR(pFrameMix), ARR(oFrameMix),
  "a frame's conditioning row: the softmax(logits)-weighted sum of hid's L planes times scale, "
  "stored at dst[lane * stride + frame - 1] for frame >= 1 and status[lane] >= 0." },
{ "end_status", ARR(pEndStatus), ARR(oEndStatus),
  "status[lane[i]] = ids[rep i] == end ? -1 : ids[rep i]." },
{ "frame_store", ARR(pFrameStore), ARR(oFrameStore),
  "a frame's codes into its lane's history: dst[(lane n_codes + k) stride + frame - skip] = "
  "codes[k, rep i] when frame >= skip and status[lane] >= 0." },
{ "resample_nearest", ARR(pMnD), ARR(oUnary),
  "rows resampled nearest-neighbour: output row j reads input row min(floor(j * in / out), "
  "in - 1) in f32, the row counts read off the operands." },
{ "fourier_embed", ARR(pFourier), ARR(oFourier),
  "[cos(2 pi t w) | sin(2 pi t w)] of t [M] f32 and a trained w [n / 2]; `round` 1 rounds each "
  "tensor a bf16 reference holds (t, 2 pi t, the product, each cosine and sine)." },
{ "win_input", ARR(pWinInput), ARR(oWinInput),
  "a denoiser's input for its window entries: [lat | zeros | cond] from each entry's lane of the "
  "latent and conditioning states, zeros for the conditioning of an unconditional entry "
  "(pair -2)." },
{ "win_prep", ARR(pWinPrep), ARR(oWinPrep),
  "a window step's start for each entry that moves its lane: flag 1 draws the window's noise "
  "(rad_philox_normal_s(seed, 1 + win_index, .)) into the lane, keeping its first `ov` rows with "
  "flag 16; flag 16 binds the first rows to the previous window's end, "
  "bf16(bf16(1 - bf16((1 - 1e-6) t)) noise + bf16(t prev))." },
{ "win_finish", ARR(pWinFinish), ARR(oWinFinish),
  "a window's end (flag 8): with flag 16 its first rows become the previous window's end "
  "exactly; rows [max(0, n - 2 ov), max(.., n - ov)) of the latent and the conditioning are kept "
  "as the next window's prev and prev_cond." },
{ "chan_split", ARR(pChanSplit), ARR(oUnary),
  "x [rows, parts * n] to y [parts * rows, n]: each channel group a sequence of its own." },
{ "interleave2", ARR(pInterleave), ARR(oInterleave2),
  "two streams interleaved, y [rows, 2] = (a[r], b[r])." },
{ "iota", ARR(pIota), ARR(oIota),
  "y[i] = start + i * step over y's elements, or start + (i % period) * step with `period`." },
{ "cu_repeat", ARR(pCuRepeat), ARR(oCuRepeat),
  "bounds of equal sequences, y[j] = j * len[0] * mult over y's elements." },
{ "int_affine", ARR(pIntAffine), ARR(oIntAffine),
  "y[j] = (src[j] < 0 ? 0 : x[src[j]]) * mul[j] + add[j] over three integer lists: shapes and "
  "bounds derived from a pass's device geometry." },
};

/* ================================================================== the kernel table */
/* One row per op. `shape` is the same on every row, since ref refuses no geometry; --debug-graph
 * prints it beside the warning that an op fell through to ref. */
#define SHAPE_ANY "any: every extent is read off the operands, strides included"

/* The operand-description hook (`RadKernelInfo::opd_shape`, rad_abi.h) is pasted from the launch
 * name as `fn ## _shape` (see ref_ops.h), so a row without one fails to link. The implementations
 * are in libref/ref_shapes.cpp.
 *
 * Every launch first passes ref_check_dtypes (below), so an operand libref cannot read or store
 * is refused rather than read as zeros.
 *
 * No tunables on any row: the oracle has one implementation per op, so its answer never depends on
 * a variant. The three nulls are tunables, n_tunables and tune_valid. */
static bool ref_readable(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_F16: case RAD_BF16: case RAD_F8E4M3: case RAD_F8E5M2:
        case RAD_I8: case RAD_U8: case RAD_BOOL: case RAD_I16: case RAD_U16: case RAD_I32:
        case RAD_U32: case RAD_I64: case RAD_I4: case RAD_I2: case RAD_FP4E2M1: case RAD_E8M0:
            return true;
        default:
            return false;
    }
}
static bool ref_storable(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_F16: case RAD_BF16: case RAD_F8E4M3: case RAD_F8E5M2:
        case RAD_I8: case RAD_U8: case RAD_BOOL: case RAD_I16: case RAD_U16: case RAD_I32:
        case RAD_U32: case RAD_I64: case RAD_I4: case RAD_I2: case RAD_FP4E2M1:
            return true;
        default:
            return false;
    }
}

/* A weight operand stored in the joined form of abi/rad_planes.h, or a table whose first entry is:
 * its dtype is the first plane's, and the kernel reads each plane by the dtype its entry names. */
static bool ref_joined_weight(const RadTensor& t, int role) {
    if (role != RAD_OPD_WEIGHT && role != RAD_OPD_WTAB) return false;
    const void* p = t.data;
    if (role == RAD_OPD_WTAB && t.rank >= 1 && t.stride[0] == 0)
        p = t.shape[0] > 0 ? ((void* const*)t.data)[0] : nullptr;
    return rad_planes_header(p) != nullptr;
}

/* RAD_E_DTYPE for a present operand whose dtype ref_common.h's ld_dt cannot read (an input) or
 * st_dt cannot store (an output): those return zero and drop the store, which would make a call
 * on an unsupported format return RAD_OK with a wrong answer. A joined weight is left to its
 * kernel, which refuses the planes it does not compute. */
static int ref_check_dtypes(const char* op, const RadArgs* a) {
    if (!a) return RAD_E_INVAL;
    const RadOpSchema* sc = nullptr;
    for (const RadOpSchema& s : kSchemas)
        if (std::strcmp(s.op, op) == 0) { sc = &s; break; }
    if (!sc) return RAD_E_NOKERNEL;
    const int n = a->n_t < sc->n_operands ? a->n_t : sc->n_operands;
    for (int i = 0; i < n; ++i) {
        const RadTensor& t = a->t[i];
        if (!t.data) continue;
        const int role = sc->operands[i].role;
        if (role != RAD_OPD_OUT && !ref_readable(t.dtype) && !ref_joined_weight(t, role))
            return RAD_E_DTYPE;
        if ((role == RAD_OPD_OUT || role == RAD_OPD_INOUT) && !ref_storable(t.dtype))
            return RAD_E_DTYPE;
    }
    return RAD_OK;
}

#define ROW(nm, opname, fam, what, dts, cons, ncons, init, fini, fn, scr)                        \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                 \
      cons, ncons, nullptr, 0, nullptr, init, fini,                                              \
      [](const RadArgs* a, RadStream s) -> int {                                                 \
          const int rc = ref_check_dtypes(opname, a);                                            \
          return rc < 0 ? rc : fn(a, s);                                                         \
      },                                                                                         \
      scr, nullptr, nullptr, fn##_shape }

#define SIMPLE(nm, opname, fam, what, dts, fn)                                                   \
    ROW(nm, opname, fam, what, dts, nullptr, 0, nullptr, nullptr, fn, nullptr)

/* A grouped GEMM row with the codebook experts' layout hooks (ref_moe.cpp). */
#define ROW_CB(nm, opname, fam, what, dts, fn)                                                   \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                 \
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,                                         \
      [](const RadArgs* a, RadStream s) -> int {                                                 \
          const int rc = ref_check_dtypes(opname, a);                                            \
          return rc < 0 ? rc : fn(a, s);                                                         \
      },                                                                                         \
      nullptr, ref_moe_cb_layout, ref_moe_cb_relayout, fn##_shape, ref_moe_cb_unrelayout }

/* gemm_nt_q's zero-point grids read their codes as `b`: u2, u4 or u5, which ld_dt does not read
 * and every other op refuses. The grid's own path reads them (ref_gemm.cpp: gemm_zp), so they pass
 * the check there and only there. */
static int ref_check_gemm_q(const char* op, const RadArgs* a) {
    const ref::ZpGrid* zg =
        a ? ref::zp_grid(rad_param_gets(a->p, a->n_p, "dtype", nullptr)) : nullptr;
    if (!zg || a->n_t < 3 || !a->t[2].data || a->t[2].dtype != zg->codes)
        return ref_check_dtypes(op, a);
    RadArgs b = *a;
    RadTensor t[16];
    if (a->n_t > 16) return RAD_E_INVAL;
    for (int i = 0; i < a->n_t; ++i) t[i] = a->t[i];
    t[2].data = nullptr;
    b.t = t;
    return ref_check_dtypes(op, &b);
}

/* A row whose weight operands may be joined over several planes (ref_planes.h): its layout hooks,
 * and the dtype check its launch passes first. */
#define ROWL(nm, opname, fam, what, dts, check, fn, lay)                                          \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                 \
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,                                         \
      [](const RadArgs* a, RadStream s) -> int {                                                 \
          const int rc = check(opname, a);                                                       \
          return rc < 0 ? rc : fn(a, s);                                                         \
      },                                                                                         \
      nullptr, lay, ref::ref_relayout_planes<lay>, fn##_shape, ref::ref_unrelayout_planes<lay> }

#define DT_FLOAT "bf16 f16 f32 fp8_e4m3 fp8_e5m2 i8 i32, chosen per operand from its own dtype"
#define DT_ANY   "any dtype rad_types.h names, per operand"
#define DT_FLOAT_CB DT_FLOAT "; or u4 / u5 codes into an f32 table, joined (w4nl* / w5nl*)"

static const RadKernelInfo kKernels[] = {
SIMPLE("ref_rmsnorm", "rmsnorm", "norm", "root-mean-square norm with a gain", DT_FLOAT,
       ref_rmsnorm),
SIMPLE("ref_rmsnorm_p", "rmsnorm_p", "norm", "weightless RMS norm (no gain, plain x/rms)", DT_FLOAT,
       ref_rmsnorm_p),
SIMPLE("ref_rmsnorm_add", "rmsnorm_add", "norm", "residual add then RMS norm", DT_FLOAT,
       ref_rmsnorm_add),
SIMPLE("ref_hc_enter", "hc_enter", "norm", "gated residual entry: hc copies of x", DT_FLOAT,
       ref_hc_enter),
SIMPLE("ref_mhc_pre", "mhc_pre", "norm", "manifold residual gates and Sinkhorn mixing", DT_FLOAT,
       ref_mhc_pre),
SIMPLE("ref_mhc_pre_partial", "mhc_pre_partial", "norm", "manifold residual projection partials", DT_FLOAT,
       ref_mhc_pre_partial),
SIMPLE("ref_mhc_pre_finish", "mhc_pre_finish", "norm", "manifold residual gates from projection partials", DT_FLOAT,
       ref_mhc_pre_finish),
SIMPLE("ref_mhc_post", "mhc_post", "norm", "manifold residual source-to-destination update", DT_FLOAT,
       ref_mhc_post),
SIMPLE("ref_mhc_contract", "mhc_contract", "norm", "mean of manifold residual streams", DT_FLOAT,
       ref_mhc_contract),
SIMPLE("ref_kda_update", "kda_update", "gdn", "bounded per-channel delta recurrence and candidate rollback", DT_FLOAT,
       ref_kda_update),
ROWL("ref_hc_read", "hc_read", "norm",
     "gated residual read: grouped norm, low-rank gate, stream mix",
     DT_FLOAT ", or mixing matrices of E4M3 / int8 codes with a scale plane, joined",
     ref_check_dtypes, ref_hc_read, ref::ref_layout_hc),
ROWL("ref_mtp_enter", "mtp_enter", "norm",
     "MTP head entry: two norms, a per-sub-stream fc and a broadcast fc",
     DT_FLOAT ", or fc matrices of E4M3 / int8 codes with a scale plane, joined",
     ref_check_dtypes, ref_mtp_enter, ref::ref_layout_mtp),
SIMPLE("ref_hc_write", "hc_write", "norm", "gated residual write: h[s] += inj[s] * y",
       DT_FLOAT, ref_hc_write),
SIMPLE("ref_hc_project", "hc_project", "norm",
       "control vector: h[s] -= scale * (h[s] . v) * v where the row's flag is on",
       DT_FLOAT, ref_hc_project),
SIMPLE("ref_layernorm", "layernorm", "norm", "layer norm with gain and bias", DT_FLOAT,
       ref_layernorm),
SIMPLE("ref_grid_embed", "grid_embed", "elem",
       "bilinear resample of a learned position grid, added in", DT_FLOAT, ref_grid_embed),
SIMPLE("ref_grid_embed_aa", "grid_embed_aa", "elem",
       "antialiased half-pixel bilinear resample of a position grid, added in", DT_FLOAT,
       ref_grid_embed_aa),
SIMPLE("ref_add", "add", "elem", "elementwise sum", DT_FLOAT, ref_add),
SIMPLE("ref_mul", "mul", "elem", "elementwise product", DT_FLOAT, ref_mul),
SIMPLE("ref_mla_attn", "mla_attn", "attn", "absorbed latent MQA attention", DT_FLOAT,
       ref_mla_attn),
SIMPLE("ref_dsa_score", "dsa_score", "attn", "weighted FP8 index scoring", DT_FLOAT,
       ref_dsa_score),
SIMPLE("ref_dsa_pool", "dsa_pool", "attn", "learned compressed FP8 index keys", DT_FLOAT,
       ref_dsa_pool),
SIMPLE("ref_dsa_query", "dsa_query", "attn", "normalized FP8 index queries", DT_FLOAT,
       ref_dsa_query),
SIMPLE("ref_gemm_nt_f32", "gemm_nt_f32", "gemm", "F32 routing projection", DT_FLOAT,
       ref_gemm_nt_f32),
SIMPLE("ref_mla_project", "mla_project", "attn", "absorbed MLA projections", DT_FLOAT,
       ref_mla_project),
SIMPLE("ref_swiglu_clamped", "swiglu_clamped", "elem", "asymmetrically clipped SwiGLU", DT_FLOAT,
       ref_swiglu_clamped),
SIMPLE("ref_silu_mul", "silu_mul", "elem", "silu-gated product over a fused gate_up", DT_FLOAT,
        ref_silu_mul),
SIMPLE("ref_act_mul", "act_mul", "elem", "act(a) * b over two operands", DT_FLOAT, ref_act_mul),
SIMPLE("ref_mul_weight", "mul_weight", "elem", "x * a scalar or one-row weight", DT_FLOAT,
       ref_mul_weight),
SIMPLE("ref_table_add", "table_add", "elem", "a row plus each of L table rows", DT_FLOAT,
       ref_table_add),
SIMPLE("ref_mix_rows", "mix_rows", "elem", "a weighted sum of k consecutive rows", DT_FLOAT,
       ref_mix_rows),
SIMPLE("ref_chan_stats", "chan_stats", "elem", "per-channel statistics held as weights",
       DT_FLOAT, ref_chan_stats),
SIMPLE("ref_seg_mean", "seg_mean", "elem", "mean of each cu_seqlens segment's rows", DT_FLOAT,
       ref_seg_mean),
SIMPLE("ref_pool_rows", "pool_rows", "elem", "mean of every k consecutive rows, then a multiplier",
       DT_FLOAT, ref_pool_rows),
SIMPLE("ref_relu", "relu", "elem", "max(x, 0)", DT_FLOAT, ref_relu),
SIMPLE("ref_clamp", "clamp", "elem", "x clamped to two weight scalars", DT_FLOAT, ref_clamp),
SIMPLE("ref_conv2d", "conv2d", "elem", "segment-wise 2-D convolution, channels innermost",
       DT_FLOAT, ref_conv2d),
SIMPLE("ref_conv2d_unfold", "conv2d_unfold", "elem",
       "a convolution's receptive fields as rows, for a GEMM against its weight", DT_FLOAT,
       ref_conv2d_unfold),
SIMPLE("ref_conv1d_dw", "conv1d_dw", "elem", "segment-wise causal depthwise convolution",
       DT_FLOAT, ref_conv1d_dw),
SIMPLE("ref_rel_sinusoid", "rel_sinusoid", "elem", "sinusoidal relative-position table", DT_FLOAT,
       ref_rel_sinusoid),
SIMPLE("ref_seg_sinusoid", "seg_sinusoid", "elem",
       "sinusoidal absolute positions, restarting every `period` rows of a segment", DT_FLOAT,
       ref_seg_sinusoid),
SIMPLE("ref_attn_local_rel", "attn_local_rel", "attn",
       "local attention with a relative-position term and a softcap", DT_FLOAT,
       ref_attn_local_rel),
SIMPLE("ref_gelu", "gelu", "elem", "exact erf gelu", DT_FLOAT, ref_gelu),
SIMPLE("ref_silu", "silu", "elem", "silu", DT_FLOAT, ref_silu),
SIMPLE("ref_sigmoid", "sigmoid", "elem", "logistic sigmoid", DT_FLOAT, ref_sigmoid),
SIMPLE("ref_fill", "fill", "elem", "constant fill", DT_FLOAT, ref_fill),
SIMPLE("ref_gelu_mul", "gelu_mul", "elem",
       "tanh-gelu-gated product over a fused gate_up (gemma's gelu_pytorch_tanh)", DT_FLOAT,
       ref_gelu_mul),
SIMPLE("ref_logit_softcap", "logit_softcap", "elem", "final-logit softcap: cap * tanh(x / cap)",
       DT_FLOAT, ref_logit_softcap),
SIMPLE("ref_scale", "scale", "elem", "per-tensor scalar scale: y = x * s (s a [1] weight)",
       DT_FLOAT, ref_scale),
SIMPLE("ref_ngram_ids", "ngram_ids", "vocab", "hashed n-gram row ids for PLE", DT_ANY,
       ref_ngram_ids),
SIMPLE("ref_ple_gate", "ple_gate", "norm", "PLE gate: three grouped norms, a dot, a signed sqrt",
       DT_FLOAT, ref_ple_gate),
SIMPLE("ref_ple_conv", "ple_conv", "norm", "PLE's dilated depthwise conv, silu and residual add",
       DT_FLOAT, ref_ple_conv),
SIMPLE("ref_causal_conv1d", "causal_conv1d", "conv", "depthwise causal conv, no activation",
       DT_FLOAT, ref_causal_conv1d),
SIMPLE("ref_gather_rows", "gather_rows", "elem", "y[i,:] = x[idx[i],:]", DT_ANY, ref_gather_rows),
SIMPLE("ref_scatter_rows", "scatter_rows", "elem", "x[idx[i],:] = v[i,:]", DT_ANY, ref_scatter_rows),
SIMPLE("ref_cast", "cast", "elem", "dtype conversion", DT_ANY, ref_cast),
SIMPLE("ref_softmax", "softmax", "elem", "row-wise softmax", DT_FLOAT, ref_softmax),

SIMPLE("ref_quant_act_i8", "quant_act_i8", "quant",
       "symmetric int8 activation quantiser, one scale a row, codes in the WMMA step order",
       "float in, i8 out, f32 scale", ref_quant_act_i8),
SIMPLE("ref_quant_act_i8g", "quant_act_i8g", "quant",
       "symmetric int8 activation quantiser, one scale per group, row-major",
       "float in, i8 out, f32 scale", ref_quant_act_i8g),
SIMPLE("ref_quant_act_fp8", "quant_act_fp8", "quant", "symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale", ref_quant_act_fp8),
SIMPLE("ref_quant_cols_fp8", "quant_cols_fp8", "quant", "E4M3 with one scale per column",
       "float in, fp8_e4m3 out, f32 scale", ref_quant_cols_fp8),
/* The fused fp8 forms, so a host-backend run can start on an fp8 model (see ref_quant.cpp). */
SIMPLE("ref_rmsnorm_quant_fp8", "rmsnorm_quant_fp8", "quant",
       "RMS norm, optional residual add, and the E4M3 quantiser in one pass",
       "bf16 in, fp8_e4m3 out, f32 scale", ref_rmsnorm_quant_fp8),
SIMPLE("ref_gated_quant_fp8", "gated_quant_fp8", "quant",
       "act(gate) * up over a packed [M, 2n] plane, then the E4M3 quantiser",
       "bf16 in, fp8_e4m3 out, f32 scale", ref_gated_quant_fp8),
SIMPLE("ref_gemm_nt_q_gated", "gemm_nt_q_gated", "gemm",
       "gemm_nt_q over a gate_up plane with the gated quantiser in the epilogue",
       "fp8_e4m3 operands, fp8_e4m3 out, f32 scale", ref_gemm_nt_q_gated),
SIMPLE("ref_dflash_select", "dflash_select", "draft",
       "DFlash2's greedy walk over the candidate lattice",
       "i32 candidates, bf16 codebooks, i32 tokens out", ref_dflash_select),
SIMPLE("ref_givens_rotate", "givens_rotate", "quant",
       "channel scale and Givens rotations per part, in the output's dtype",
       "float in, float out; int32 pairs, f32 angles and channel scales", ref_givens_rotate),
SIMPLE("ref_givens_dense_quant", "givens_dense_quant", "quant",
       "the rotation as a dense per-group matrix, then the block-scaled int8 or E4M3 quantiser",
       "float in, int8 or fp8_e4m3 out, f32 scale per group; f16 matrices and f32 channel scales",
       ref_givens_dense_quant),
SIMPLE("ref_had_quant_act_fp8", "had_quant_act_fp8", "quant",
       "width-`group` Hadamard, then the same symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale per group; n a multiple of group",
       ref_had_quant_act_fp8),
SIMPLE("ref_had_quant_act_i8g", "had_quant_act_i8g", "quant",
       "width-`group` Hadamard, then the symmetric int8 quantiser, one scale per group",
       "float in, i8 out, f32 scale per group; n a multiple of group",
       ref_had_quant_act_i8g),
SIMPLE("ref_gram_accum", "gram_accum", "quant",
       "h += x^T x over a dequantised E4M3 activation, or a plain bf16 one -- the Gram matrix "
       "GPTQ calibrates on",
       "fp8_e4m3 plus an f32 scale per group, or bf16 with no scale, in; an f32 [n][n] "
       "accumulator READ and added to",
       ref_gram_accum),
SIMPLE("ref_had_quant_act_i8", "had_quant_act_i8", "quant",
       "Walsh-Hadamard rotation then symmetric int8", "float in, i8 out, f32 scale",
       ref_had_quant_act_i8),
SIMPLE("ref_rmsnorm_had_quant_i8", "rmsnorm_had_quant_i8", "quant",
       "RMS norm, rotation and int8 in one pass", "float in, i8 out, f32 scale",
       ref_rmsnorm_had_quant_i8),
SIMPLE("ref_gated_had_quant_i8", "gated_had_quant_i8", "quant",
       "gated product, rotation and int8 in one pass", "float in, i8 out, f32 scale",
       ref_gated_had_quant_i8),
SIMPLE("ref_dequant", "dequant", "quant", "group-scaled dequantisation", DT_ANY, ref_dequant),

SIMPLE("ref_gemm_nt", "gemm_nt", "gemm", "C = A B^T, f32 accumulate", DT_FLOAT, ref_gemm_nt),
SIMPLE("ref_gemm_nt_bias", "gemm_nt_bias", "gemm", "C = A B^T + bias", DT_FLOAT,
       ref_gemm_nt_bias),
ROWL("ref_gemm_nt_q", "gemm_nt_q", "gemm", "C = A B^T with group-scaled quantised operands",
     "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; u2 / u4 / u5 codes with an f16 scale "
     "and integer zero per group, joined; f32 accumulate",
     ref_check_gemm_q, ref_gemm_nt_q, ref::ref_layout_gemm_q),
SIMPLE("ref_gemm_nt_q_bias", "gemm_nt_q_bias", "gemm", "quantised C = A B^T + bias",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; f32 accumulate", ref_gemm_nt_q_bias),
SIMPLE("ref_lora_shrink", "lora_shrink", "gemm", "t = x A[slot]^T per row, base rows untouched",
       "bf16 / f16 / f32 a, or fp8_e4m3 / fp8_e5m2 / i8 a with its f32 scale stream; bf16 lora_a; "
       "i32 slot; f32 t", ref_lora_shrink),
SIMPLE("ref_lora_expand", "lora_expand", "gemm",
       "y += scale * (t B[slot]^T) per row, base rows untouched",
       "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y", ref_lora_expand),
SIMPLE("ref_lora_expand_mm", "lora_expand_mm", "gemm",
       "y += scale * (bf16(t) B[slot]^T) per row, base rows untouched",
       "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y", ref_lora_expand_mm),

ROW("ref_attn_paged", "attn_paged", "attn",
    "paged causal / sliding-window attention, any query length",
    "any q and out dtype; bf16 or fp8 kv cache with optional per-(seq,head) descales",
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])),
    nullptr, nullptr, ref_attn_paged, nullptr),
SIMPLE("ref_attn_dense", "attn_dense", "attn", "dense varlen attention over cu_seqlens", DT_FLOAT,
       ref_attn_dense),
SIMPLE("ref_attn_cross", "attn_cross", "attn",
       "dense varlen attention, keys bounded apart from the queries (cu_k)", DT_FLOAT,
       ref_attn_cross),
SIMPLE("ref_attn_na3d", "attn_na3d", "attn", "3-D neighborhood attention over a volume", DT_FLOAT,
       ref_attn_na3d),
ROW("ref_kv_store", "kv_store", "attn", "scatter k and v into the paged cache", DT_ANY,
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])),
    nullptr, nullptr, ref_kv_store, nullptr),
SIMPLE("ref_rope", "rope", "attn", "rotary embedding in place", DT_FLOAT, ref_rope),
SIMPLE("ref_qk_norm_rope", "qk_norm_rope", "attn", "per-head QK norm then rotary embedding",
       DT_FLOAT, ref_qk_norm_rope),

SIMPLE("ref_scale_rows", "scale_rows", "elem", "one scalar per row, broadcast across the row",
       DT_FLOAT, ref_scale_rows),
SIMPLE("ref_router_sigmoid_topk", "router_sigmoid_topk", "moe", "sigmoid correction-bias routing", DT_FLOAT,
       ref_router_sigmoid_topk),
SIMPLE("ref_router_topk", "router_topk", "moe", "softmax then exact top-k routing", DT_FLOAT,
       ref_router_topk),
SIMPLE("ref_moe_scatter", "moe_scatter", "moe", "stable counting sort of routing slots by expert",
       "i32 indices", ref_moe_scatter),
SIMPLE("ref_moe_gemm", "moe_gemm", "moe", "grouped GEMM over the expert-sorted rows", DT_FLOAT,
       ref_moe_gemm),
ROW_CB("ref_moe_gemm_q", "moe_gemm_q", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands", DT_FLOAT_CB,
       ref_moe_gemm_q),
ROW_CB("ref_moe_gemm_q_mod4", "moe_gemm_q_mod4", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands, four classes by number mod 4",
       DT_FLOAT_CB, ref_moe_gemm_q_mod4),
SIMPLE("ref_moe_gather", "moe_gather", "moe", "weighted scatter back to token order", DT_FLOAT,
       ref_moe_gather),
SIMPLE("ref_rope_table", "rope_table", "attn", "cos/sin at each position's own row", DT_FLOAT, ref_rope_table),
SIMPLE("ref_qsa_block_key", "qsa_block_key", "attn",
       "QSA compressed key: pooled, normed and rotated at the block's position", DT_FLOAT,
       ref_qsa_block_key),
SIMPLE("ref_qsa_score", "qsa_score", "attn",
       "QSA block scores: per-head relu dots summed over the heads", DT_FLOAT, ref_qsa_score),
SIMPLE("ref_qsa_select", "qsa_select", "attn",
       "QSA selection: the top-k blocks as a page-id table, partial page last",
       "f32 scores, i32 tables", ref_qsa_select),
SIMPLE("ref_qsa_tail_store", "qsa_tail_store", "attn",
       "QSA tail: the raw keys of open blocks into a per-sequence ring", DT_FLOAT,
       ref_qsa_tail_store),
SIMPLE("ref_qsa_work", "qsa_work", "attn",
       "QSA work list: completing blocks' keys, pages and positions", DT_FLOAT, ref_qsa_work),
SIMPLE("ref_dflash_conv", "dflash_conv", "draft",
       "DFlash2 grouped dynamic depthwise convolution", DT_FLOAT, ref_dflash_conv),
SIMPLE("ref_row_topk", "row_topk", "moe", "exact per-row top-R", DT_FLOAT, ref_row_topk),
SIMPLE("ref_row_topk_merge", "row_topk_merge", "moe", "the global top-R of the gathered per-rank candidate sets", DT_FLOAT, ref_row_topk_merge),
SIMPLE("ref_pair_argmax", "pair_argmax", "moe", "the global top-1 of a row of gathered (id, value) pairs, one id a row", DT_FLOAT, ref_pair_argmax),
ROWL("ref_logit_rerank", "logit_rerank", "moe",
     "R candidates a row rescored exactly against the full-precision head",
     DT_FLOAT ", or a head of E4M3 / int8 codes with a scale plane, joined",
     ref_check_dtypes, ref_logit_rerank, ref::ref_layout_head),

SIMPLE("ref_gdn_conv_prep", "gdn_conv_prep", "gdn",
       "gdn prefill preamble: conv, split, l2 norm, gate cumsum, beta", DT_FLOAT,
       ref_gdn_conv_prep),
SIMPLE("ref_gdn_conv_update", "gdn_conv_update", "gdn",
       "gdn decode conv over the rolling speculative window", DT_FLOAT, ref_gdn_conv_update),
SIMPLE("ref_gdn_kkt_solve", "gdn_kkt_solve", "gdn", "gdn chunk preamble: the triangular inverse",
       DT_FLOAT, ref_gdn_kkt_solve),
SIMPLE("ref_gdn_chunk_scan", "gdn_chunk_scan", "gdn", "gdn chunked scan", DT_FLOAT,
       ref_gdn_chunk_scan),
SIMPLE("ref_gdn_recurrent_update", "gdn_recurrent_update", "gdn",
       "gdn decode recurrence against the paged state cache, per-candidate or anchor state",
       DT_FLOAT, ref_gdn_recurrent_update),
SIMPLE("ref_gdn_conv_recurrent_update", "gdn_conv_recurrent_update", "gdn",
       "gdn_conv_update then gdn_recurrent_update, composed from those two references", DT_FLOAT,
       ref_gdn_conv_recurrent_update),
SIMPLE("ref_gdn_gated_rmsnorm", "gdn_gated_rmsnorm", "gdn", "gated RMS norm over a head", DT_FLOAT,
       ref_gdn_gated_rmsnorm),

ROW("ref_all_reduce", "all_reduce", "ar", "sum across ranks, f32, ascending rank order", DT_FLOAT,
    nullptr, 0, ref_collective_init, ref_collective_fini, ref_all_reduce, ref_all_reduce_scratch),
ROW("ref_all_gather", "all_gather", "ar", "gather every rank's shard", DT_FLOAT,
    nullptr, 0, ref_collective_init, ref_collective_fini, ref_all_gather, nullptr),

SIMPLE("ref_embed_lookup", "embed_lookup", "vocab", "embedding gather", DT_FLOAT,
       ref_embed_lookup),
SIMPLE("ref_embed_lookup_q", "embed_lookup_q", "vocab", "embedding gather, one scale for the table",
       DT_ANY, ref_embed_lookup_q),
ROWL("ref_logits_gemm", "logits_gemm", "vocab", "the lm_head projection",
     DT_FLOAT ", or a head of E4M3 / int8 codes with a scale plane, joined",
     ref_check_dtypes, ref_logits_gemm, ref::ref_layout_head),

SIMPLE("ref_sample_penalties", "sample_penalties", "sample",
       "repetition, frequency and presence penalties", DT_FLOAT, ref_sample_penalties),
SIMPLE("ref_sample_temp", "sample_temp", "sample", "per-request temperature", DT_FLOAT,
       ref_sample_temp),
SIMPLE("ref_sample_topk", "sample_topk", "sample", "exact top-k candidate set", DT_FLOAT,
       ref_sample_topk),
SIMPLE("ref_sample_topp", "sample_topp", "sample", "nucleus narrowing in place", DT_FLOAT,
       ref_sample_topp),
SIMPLE("ref_sample_minp", "sample_minp", "sample", "min-p narrowing in place", DT_FLOAT,
       ref_sample_minp),
SIMPLE("ref_sample_mask", "sample_mask", "sample", "grammar bitmask application", DT_FLOAT,
       ref_sample_mask),
SIMPLE("ref_sample_pick", "sample_pick", "sample", "seeded inverse-CDF draw", DT_FLOAT,
       ref_sample_pick),
SIMPLE("ref_sample_argmax", "sample_argmax", "sample", "greedy pick", DT_FLOAT, ref_sample_argmax),
SIMPLE("ref_sample_logprobs", "sample_logprobs", "sample", "the raw distribution's summary",
       DT_FLOAT, ref_sample_logprobs),
SIMPLE("ref_sample_dry", "sample_dry", "sample", "DRY repetition-suffix penalty", DT_FLOAT,
       ref_sample_dry),
SIMPLE("ref_sample_typical", "sample_typical", "sample", "locally typical narrowing in place",
       DT_FLOAT, ref_sample_typical),
SIMPLE("ref_sample_xtc", "sample_xtc", "sample", "exclude-top-choices narrowing in place",
       DT_FLOAT, ref_sample_xtc),
SIMPLE("ref_sample_merge_topk", "sample_merge_topk", "sample",
       "vocab-parallel candidate merge", DT_FLOAT, ref_sample_merge_topk),
SIMPLE("ref_ln_mod", "ln_mod", "norm", "gainless layer norm times (1 + a sequence's scale row)",
       DT_FLOAT, ref_ln_mod),
SIMPLE("ref_gate_add", "gate_add", "elem", "x += act(gate row) * y, act tanh or none", DT_FLOAT,
       ref_gate_add),
SIMPLE("ref_ln_mod_quant_fp8", "ln_mod_quant_fp8", "norm",
       "gated residual, gainless layer norm times (1 + scale row), block-scaled E4M3",
       DT_FLOAT, ref_ln_mod_quant_fp8),
SIMPLE("ref_timestep_embed", "timestep_embed", "elem", "sinusoidal timestep embedding",
       DT_FLOAT, ref_timestep_embed),
SIMPLE("ref_gather_rows2", "gather_rows2", "elem", "rows from two sources by a signed index",
       DT_ANY, ref_gather_rows2),
SIMPLE("ref_lane_gather", "lane_gather", "elem", "each entry's rows out of its lane", DT_ANY,
       ref_lane_gather),
SIMPLE("ref_lane_store", "lane_store", "elem", "one source's rows into several lanes", DT_ANY,
       ref_lane_store),
SIMPLE("ref_randn_lanes", "randn_lanes", "elem", "a lane's rows from its seed's normal stream",
       DT_FLOAT, ref_randn_lanes),
SIMPLE("ref_flow_euler", "flow_euler", "elem", "a flow-matching Euler step with guidance",
       DT_FLOAT, ref_flow_euler),
SIMPLE("ref_kd_input", "kd_input", "elem", "a k-diffusion sampler's model input", DT_FLOAT,
       ref_kd_input),
SIMPLE("ref_kd_step", "kd_step", "elem", "a k-diffusion sampler step with guidance", DT_FLOAT,
       ref_kd_step),
SIMPLE("ref_chan_affine", "chan_affine", "elem", "a per-channel affine from parameter lists",
       DT_FLOAT, ref_chan_affine),
SIMPLE("ref_conv2d_nhwc", "conv2d_nhwc", "elem", "an image convolution, channels innermost",
       DT_FLOAT, ref_conv2d_nhwc),
SIMPLE("ref_im2col_nhwc", "im2col_nhwc", "elem", "a convolution's receptive fields as rows",
       DT_FLOAT, ref_im2col_nhwc),
SIMPLE("ref_l2norm_chan", "l2norm_chan", "norm", "unit-L2 channel norm, scaled, optional silu",
       DT_FLOAT, ref_l2norm_chan),
SIMPLE("ref_d2s_add", "d2s_add", "elem", "an upsampling shortcut added in", DT_FLOAT,
       ref_d2s_add),
SIMPLE("ref_s2d_avg_add", "s2d_avg_add", "elem", "a downsampling shortcut added in", DT_FLOAT,
       ref_s2d_avg_add),
SIMPLE("ref_to_rgba8", "to_rgba8", "elem", "decoded pixels to bytes", DT_FLOAT, ref_to_rgba8),
SIMPLE("ref_gelu_tanh", "gelu_tanh", "elem", "tanh gelu", DT_FLOAT, ref_gelu_tanh),
SIMPLE("ref_gelu_quick", "gelu_quick", "elem", "quick gelu", DT_FLOAT, ref_gelu_quick),
SIMPLE("ref_scatter_add_rows", "scatter_add_rows", "elem", "x[idx[i]] += v[i]", DT_FLOAT,
       ref_scatter_add_rows),
SIMPLE("ref_attn_prefix_self", "attn_prefix_self", "attn",
       "attention over a paged prefix and the step's own rows", DT_FLOAT, ref_attn_prefix_self),
SIMPLE("ref_rms_mod", "rms_mod", "norm",
       "gated residual, RMS norm with a gain times (1 + scale row) plus a shift row", DT_FLOAT,
       ref_rms_mod),
SIMPLE("ref_rms_mod_quant_fp8", "rms_mod_quant_fp8", "norm",
       "gated residual, modulated RMS norm, block-scaled E4M3", DT_FLOAT, ref_rms_mod_quant_fp8),
SIMPLE("ref_flow_x0_step", "flow_x0_step", "elem", "a sampler step toward the clean estimate",
       DT_FLOAT, ref_flow_x0_step),
SIMPLE("ref_patch_unfold", "patch_unfold", "elem", "patch rows to denormalised voxel rows",
       DT_FLOAT, ref_patch_unfold),
SIMPLE("ref_vtile_gather", "vtile_gather", "elem", "a video decoder's tiles out of the latent",
       DT_FLOAT, ref_vtile_gather),
SIMPLE("ref_vtile_stitch", "vtile_stitch", "elem",
       "decoded tiles blended into frames, the chunk overlap, bytes out", DT_FLOAT,
       ref_vtile_stitch),
SIMPLE("ref_grid_pos3", "grid_pos3", "elem", "a video decoder's rotary positions over its tiles",
       DT_FLOAT, ref_grid_pos3),
SIMPLE("ref_vsa_select", "vsa_select", "attn",
       "video sparse attention's top-k tiles by pooled score, and the coarse branch",
       "float q, k, v and coarse; i32 segs, geom and sel", ref_vsa_select),
SIMPLE("ref_attn_vsa", "attn_vsa", "attn",
       "block-sparse attention over the prefix and selected video tiles, the gated coarse term",
       "float q, k, v, coarse, gate and out; i32 segs, geom and sel", ref_attn_vsa),
SIMPLE("ref_conv3d_ndhwc", "conv3d_ndhwc", "conv",
       "a 3-D convolution over voxel rows, channels innermost", DT_FLOAT, ref_conv3d_ndhwc),
SIMPLE("ref_group_norm", "group_norm", "norm",
       "a group norm over all rows, optionally modulated, then an activation", DT_FLOAT,
       ref_group_norm),
SIMPLE("ref_tconv_dw", "tconv_dw", "conv", "a depthwise convolution along a volume's frames",
       DT_FLOAT, ref_tconv_dw),
SIMPLE("ref_resize_bilinear", "resize_bilinear", "elem",
       "each frame of a volume resized bilinearly", DT_FLOAT, ref_resize_bilinear),
SIMPLE("ref_patch_fold", "patch_fold", "elem", "voxel rows to normalised patch rows", DT_FLOAT,
       ref_patch_fold),
SIMPLE("ref_flow_renoise", "flow_renoise", "elem", "a clean latent re-noised to a sigma", DT_FLOAT,
       ref_flow_renoise),
SIMPLE("ref_rope_split", "rope_split", "elem", "LTX-2's split rotary over coordinates", DT_FLOAT,
       ref_rope_split),
SIMPLE("ref_head_gate", "head_gate", "elem", "each head times 2 sigmoid of its logit", DT_FLOAT,
       ref_head_gate),
SIMPLE("ref_ltx_guide", "ltx_guide", "elem", "LTX-2's guided sampler step", DT_FLOAT,
       ref_ltx_guide),
SIMPLE("ref_pixel_norm", "pixel_norm", "norm", "an RMS norm as bf16 tensors compute it", DT_FLOAT,
       ref_pixel_norm),
SIMPLE("ref_d2s3d", "d2s3d", "elem", "depth to space over voxel rows", DT_FLOAT, ref_d2s3d),
SIMPLE("ref_s2d3d", "s2d3d", "elem", "space to depth over voxel rows", DT_FLOAT, ref_s2d3d),
SIMPLE("ref_pix_patch", "pix_patch", "elem", "an image to patch rows", DT_FLOAT, ref_pix_patch),
SIMPLE("ref_patch_pix8", "patch_pix8", "elem", "patch rows to RGB bytes", DT_FLOAT, ref_patch_pix8),
SIMPLE("ref_vol_index", "vol_index", "elem", "a volume's voxel coordinates", DT_ANY, ref_vol_index),
SIMPLE("ref_tile_grid", "tile_grid", "elem", "a tiled decode's boxes", DT_ANY, ref_tile_grid),
SIMPLE("ref_tile_crop", "tile_crop", "elem", "a tile out of a volume", DT_FLOAT, ref_tile_crop),
SIMPLE("ref_tile_stitch", "tile_stitch", "elem", "decoded tiles blended to RGB bytes", DT_FLOAT,
       ref_tile_stitch),
SIMPLE("ref_stft_logmel", "stft_logmel", "elem", "a causal log-mel spectrogram", DT_FLOAT,
       ref_stft_logmel),
SIMPLE("ref_bwe_up", "bwe_up", "elem", "a sinc-resampled waveform plus a residual, clamped",
       DT_FLOAT, ref_bwe_up),
SIMPLE("ref_conv1d", "conv1d", "conv", "a 1-D convolution over sequences, channels innermost",
       DT_FLOAT, ref_conv1d),
SIMPLE("ref_conv_t1d", "conv_t1d", "conv", "a transposed 1-D convolution growing each sequence",
       DT_FLOAT, ref_conv_t1d),
SIMPLE("ref_snake", "snake", "elem", "the snake activation", DT_FLOAT, ref_snake),
SIMPLE("ref_snake_aa", "snake_aa", "elem", "the alias-free SnakeBeta: upsample, snake, downsample",
       DT_FLOAT, ref_snake_aa),
SIMPLE("ref_cfg_topk_sample", "cfg_topk_sample", "sample", "a guided top-k pick a lane",
       DT_FLOAT, ref_cfg_topk_sample),
SIMPLE("ref_rvq_embed", "rvq_embed", "elem", "a frame's embedding from its residual codes",
       DT_FLOAT, ref_rvq_embed),
SIMPLE("ref_frame_mix_store", "frame_mix_store", "elem", "a frame's conditioning row into its lane",
       DT_FLOAT, ref_frame_mix_store),
SIMPLE("ref_end_status", "end_status", "elem", "a frame's status from its first pick", DT_ANY,
       ref_end_status),
SIMPLE("ref_frame_store", "frame_store", "elem", "a frame's codes into its lane's history", DT_ANY,
       ref_frame_store),
SIMPLE("ref_resample_nearest", "resample_nearest", "elem", "rows resampled nearest-neighbour",
       DT_ANY, ref_resample_nearest),
SIMPLE("ref_fourier_embed", "fourier_embed", "elem", "random Fourier features of a time",
       DT_FLOAT, ref_fourier_embed),
SIMPLE("ref_win_input", "win_input", "elem", "a denoiser's input from the window states",
       DT_FLOAT, ref_win_input),
SIMPLE("ref_win_prep", "win_prep", "elem", "a window step's noise and overlap", DT_FLOAT,
       ref_win_prep),
SIMPLE("ref_win_finish", "win_finish", "elem", "a window's end and the overlap it hands on",
       DT_FLOAT, ref_win_finish),
SIMPLE("ref_chan_split", "chan_split", "elem", "channel groups to sequences", DT_ANY,
       ref_chan_split),
SIMPLE("ref_interleave2", "interleave2", "elem", "two streams interleaved", DT_ANY,
       ref_interleave2),
SIMPLE("ref_iota", "iota", "elem", "an arithmetic index sequence", DT_ANY, ref_iota),
SIMPLE("ref_cu_repeat", "cu_repeat", "elem", "bounds of equal sequences", DT_ANY, ref_cu_repeat),
SIMPLE("ref_int_affine", "int_affine", "elem", "integers derived from device geometry", DT_ANY,
       ref_int_affine),
};

/* ================================================================== plugin exports */
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libref",
    "0.1.0",
    "every op in the conventional vocabulary, naively and generically: the fallback that makes an "
    "unsupported model run at reduced speed instead of not at all, and the oracle rad-kbench runs "
    "the resolved kernels against (spec.md §17)",
    "host"
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
/* Declares this plugin the reference through the ABI (RadPluginReferenceFn); the engine knows no
 * library by name. */
extern "C" int rad_plugin_reference(void) { return 1; }

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
/* The rows that join the other ranks (ref_collective.cpp), via RadKernelCollectiveFn; covers the
 * all-gather as well as the all-reduce. */
extern "C" int rad_kernel_collective(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    return k && k->init == ref_collective_init;
}
