/* ref_registry.cpp -- what libref contains, in the two forms the core reads it in.
 *
 * THE SCHEMAS COME FIRST, and they are the authoritative copy of the conventional op vocabulary.
 * docs/OPS.md is their prose form (spec §17): op names are free strings and the core never learns
 * one, but ref cannot implement an op it has never heard of, so an architecture plugin that stays
 * inside this table is guaranteed to run somewhere and one that invents an op has no fallback and
 * no oracle for it. Adding a row here plus a host implementation beside it is the whole procedure
 * for adding an op; no core file changes.
 *
 * The first plugin in hierarchy order to declare an op FIXES its schema, and ref sits last, so in
 * practice a faster plugin's schema wins and ref's has to agree with it. Since arguments are
 * positional, two disagreeing schemas would make the same call mean different things depending on
 * which plugin won selection -- silent numerical garbage rather than a diagnosable failure (spec
 * §2.3). Every operand order below is therefore the one docs/OPS.md prints, in the order it prints
 * it, and the deviations are marked in the doc string of the row that deviates.
 *
 * THE KERNEL ROWS ARE DELIBERATELY UNCONSTRAINED. libr4d's rows state their geometry twice -- prose
 * and predicates -- because they are specialised and reject what they were not compiled for. These
 * are the opposite: every extent is read off the operands at launch, so there is nothing to
 * predicate on, and REF MUST MATCH ANYTHING because that is what "runs somewhere" means. The one
 * exception is structural rather than performance-related: kv_store has to be TOLD the paged block
 * size to walk the cache, so it carries `block_size >= 1`, which admits every value and refuses
 * only a query that does not name it (an absent key fails a constraint -- rad_util.cpp's
 * constraint_holds). attn_paged carries the same row, but there
 * `block_size` is RAD_DERIVED and a constraint on a derived key is skipped during matching, so it
 * decides nothing; ref reads the block size off the cache tensor instead.
 *
 * A CONSEQUENCE WORTH NAMING: spec §7.2 has the block manager read the KV block size off the
 * resolved attention kernel's constraints. libr4d answers `block_size == 16`; ref answers "any",
 * which is not a number. A core that resolves attention to ref alone therefore has to pick, and
 * 16 is the value the rest of the project is built around.
 *
 * Everything is domain HOST at priority 0. libref is portable host code, so host is the only
 * domain it declares; a device-side reference would be a second row, not a change to this one.
 */
#include "ref_common.h"
#include "ref_ops.h"

/* ================================================================== parameter specs */
#define P_INT(k)   { k, RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { k, RAD_P_INT, RAD_OPTIONAL }
#define P_STR(k)   { k, RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { k, RAD_P_STR, RAD_OPTIONAL }

/* Supplied by the KERNEL and not by the caller (rad_abi.h, spec §7.2). Constraints on it are
 * skipped during matching and the winning row's RAD_C_EQ value is written into the resolved
 * geometry afterwards. Declaring one RAD_REQUIRED makes it unselectable, because the caller cannot
 * know the value until a kernel is resolved and no kernel resolves until the value is known. */
#define P_INTD(k)  { k, RAD_P_INT, RAD_DERIVED }
/* The linear-state tile length. The scheduler cuts its steps on a multiple of it (rad_abi.h,
 * RAD_PROLE_SEQ_CHUNK); nothing else in the core reads a parameter by meaning. */
#define P_CHUNK(k) { k, RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }

/* Real-valued parameters are RAD_P_F64 (docs/OPS.md, "The four rules"). The core refuses the
 * string spelling for these keys, so a "%.9g" round trip -- exact for an IEEE single, and
 * unreadable in a graph dump -- cannot creep back in, and two plugins cannot drift apart over it.
 * Constraints never match on a float, by design: a tolerance is not a predicate. */
#define P_F64(k)   { k, RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { k, RAD_P_F64, RAD_OPTIONAL }

#define OPD(n)     { n, RAD_OPD_IN,     0 }
#define OPD_O(n)   { n, RAD_OPD_IN,     1 }
#define OUT(n)     { n, RAD_OPD_OUT,    0 }
#define OUT_O(n)   { n, RAD_OPD_OUT,    1 }
#define INOUT(n)   { n, RAD_OPD_INOUT,  0 }
#define INOUT_O(n) { n, RAD_OPD_INOUT,  1 }
#define WGT(n)     { n, RAD_OPD_WEIGHT, 0 }
/* A weight TABLE: the issue names a RUN of declared weights and the kernel is handed a
 * device array of pointers, one per expert. See RAD_WTAB in abi/rad_runtime.h. */
#define WTAB(n)    { n, RAD_OPD_WTAB,   0 }
#define WGT_O(n)   { n, RAD_OPD_WEIGHT, 1 }

#define ARR(a)  (a), (int)(sizeof(a) / sizeof((a)[0]))

/* ---- elementwise and norms ---------------------------------------------------------------- */
static const RadParamSpec pRmsnorm[]  = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                          P_F64O("wadd") };
static const RadOperandSpec oRmsnorm[] = { OPD("x"), WGT("w"), OUT("y") };

/* The GATED RESIDUAL pair. `inject` and `group` are the two switches the operand list is banded
 * on: `inject` 0 is the final mixer (no per-branch gate weight, no gate output) and `group` 0 is a
 * read whose consumer wants bf16 rather than block-scaled fp8. Both are OPTIONAL parameters that
 * DEFAULT ON, because every connection in a trunk layer wants both and the exceptions are two
 * sites in the whole model. */
static const RadParamSpec pHcRead[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"),
                                        P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
                                        P_INTO("inject"), P_INTO("group"), P_INTO("rotate"),
                                        P_STRO("mix") };
static const RadOperandSpec oHcRead[] = { OPD("h"), WGT("w"), WGT("mix_down"), WGT("mix_up"),
                                          WGT_O("inject_w"), OUT("x"), OUT_O("inj"), OUT_O("q"),
                                          OUT_O("scale"), OUT_O("rq"), OUT_O("rscale") };
/* ---- PLE / n-gram ---------------------------------------------------------------------
 *
 * `state` is a ROLLING WINDOW of committed ids -- RAD_KV_CONV's shape, ple_conv's contract, read
 * at `num_accepted - 1` and left in the layout that contract names. Its cold value is EOS and not
 * zero, because zero is a real token; see ref_ngram.cpp on why this is a state and not an input.
 * `eos` is a parameter rather than an operand because the shift rule is a property of the model,
 * not of the batch, and a constraint can be written against it. */
static const RadParamSpec pNgramIds[] = { P_INT("M"), P_INT("heads"), P_INT("ngram"),
                                         P_INT("eos") };
static const RadOperandSpec oNgramIds[] = { OPD("tok"), INOUT("state"), OPD("cu"), WGT("mult"),
                                           WGT("vocab_sizes"), WGT("offsets"),
                                           OPD_O("cache_idx"), OPD_O("has_init"),
                                           OPD_O("num_accepted"), OUT("ids"),
                                           OUT_O("ahead_ids") };
/* PLE's gate. `k` and `q` arrive UN-normed -- the op owns all three norms, because the third one
 * is taken over bytes the op itself wrote and splitting it would make two definitions of it. */
static const RadParamSpec pPleGate[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                         P_STR("dtype"), P_F64O("wadd"), P_F64O("gate_eps") };
static const RadOperandSpec oPleGate[] = { OPD("k"), OPD("q"), OPD("v"), WGT("w_key"),
                                          WGT("w_query"), WGT("w_conv"),
                                          OUT("gv"), OUT("gvn") };
/* The dilated convolution that finishes the layer. ONE op where the GDN has two, because the work
 * around it is the same at prefill and at decode -- so it takes the union of the two contracts:
 * `has_init` is prefill's question and `num_accepted` is decode's. */
static const RadParamSpec pPleConv[] = { P_INT("M"), P_INT("n"), P_INT("width"),
                                         P_INT("dilation"), P_STR("dtype") };
static const RadOperandSpec oPleConv[] = { OPD("x"), WGT("w"), INOUT("conv_state"), OPD("cu"),
                                          OPD_O("resid"), OPD_O("cache_idx"), OPD_O("has_init"),
                                          OPD_O("num_accepted"), OUT("y") };

static const RadParamSpec pHcWrite[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype") };
static const RadOperandSpec oHcEnter[] = { OPD("x"), OUT("h") };
static const RadOperandSpec oHcWrite[] = { OPD("y"), OPD("inj"), INOUT("h") };
/* The MTP head's entry, for a wide residual stream. `ngroup` is hc for a grouped norm over
 * the hc sub-streams and 1 for a flat one over the whole row -- see ref_hc.cpp on why the
 * checkpoint cannot say which and the acceptance rate can. */
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

static const RadParamSpec pMnD[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oBinary[]  = { OPD("a"), OPD("b"), OUT("y") };
static const RadOperandSpec oGateUp[]  = { OPD("gate_up"), OUT("y") };
static const RadOperandSpec oUnary[]   = { OPD("x"), OUT("y") };
static const RadOperandSpec oGather[]  = { OPD("x"), OPD("idx"), OUT("y") };
static const RadOperandSpec oScatterRows[] = { OPD("v"), OPD("idx"), INOUT("x") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

/* ---- quantisation and rotation ------------------------------------------------------------ */
static const RadParamSpec pQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
/* `asum` is libr4d's third output: the group-major per-row code sums an ASYMMETRIC weight grid
 * subtracts to remove the stored zero point. Optional, because whether the caller wants them is a
 * property of the call and not of the model. Ref does not produce them -- it has no asymmetric
 * grid to feed -- and leaves the buffer alone rather than filling it with something plausible. */
static const RadOperandSpec oQuant[]    = { OPD("x"), OUT("q"), OUT("scale"), OUT_O("asum") };
static const RadOperandSpec oHadQuant[] = { OPD("x"), OUT("q"), OUT("scale") };
/* No `asum`: that output exists for the asymmetric integer grids and an fp8 grid is symmetric by
 * construction (libr4d/r4d_rows.cpp says the same). */
static const RadOperandSpec oQuantFp8[]  = { OPD("x"), OUT("q"), OUT("scale") };

/* ---- the fp8 fusions ----------------------------------------------------------------------
 * A fused op's operands are the unfused pair's minus the tensor between them, and `out_bf16`
 * puts even that back: the codes are taken over its value, so a caller that wants to compare
 * the fused form against the pair has to be able to ask for it. */
static const RadParamSpec pRmsQuantFp8[] = { P_INT("M"), P_INT("n"), P_F64("eps"),
                                             P_INT("group"), P_STR("dtype"), P_F64O("wadd") };
static const RadOperandSpec oRmsQuantFp8[] = { OPD("x"), INOUT_O("residual"), WGT("w"),
                                               OUT("q"), OUT("scale"), OUT_O("out_bf16") };
static const RadParamSpec pGatedQuantFp8[] = { P_INT("M"), P_INT("n"), P_INT("group"),
                                               P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oGatedQuantFp8[] = { OPD("gate_up"), OUT("q"), OUT("scale"),
                                                 OUT_O("out_bf16") };
/* `act` is REQUIRED here and optional on the quantiser, for the reason docs/OPS.md gives: an
 * implementation is free to compile one gate, and an optional parameter would let it resolve
 * for another. `N` is the WEIGHT's rows -- twice the fused width -- so the output is [M, N/2]. */
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

/* Two callers spell this differently: the gate and the up as two operands with a `mode`, or one
 * fused `gate_up` with an `act`. Two operands with the second OPTIONAL expresses both -- a caller
 * with a fused [M, 2n] buffer passes it as `a` and omits `b` -- and `mode` stays as an optional
 * integer spelling of `act`, since it is a kernel plugin's name for it and not the vocabulary's. */
static const RadParamSpec pGatedQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                            P_STRO("act"), P_INTO("mode") };
static const RadOperandSpec oGatedQuant[] = { OPD("a"), OPD_O("b"), OUT("q"), OUT("scale") };

/* `h` is INOUT and that is the op: the caller zeroes it once and drives a whole calibration
 * corpus through the graph, so every launch adds its own block. `a`/`a_scale` rather than
 * `x`/`scale` because they are gemm_nt_q's first two operands in gemm_nt_q's order -- this op
 * exists to characterise exactly the operand that GEMM contracts. */
/* `a_scale` is optional: dtype "bf16" is a plain activation and carries none. */
static const RadOperandSpec oGram[] = { OPD("a"), OPD_O("a_scale"), INOUT("h") };

static const RadParamSpec pDequant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oDequant[] = { OPD("q"), OPD("scale"), OUT("y") };

/* ---- gemm --------------------------------------------------------------------------------- */
static const RadParamSpec pGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype") };
static const RadOperandSpec oGemm[]     = { OPD("a"), WGT("b"), OUT("y") };
/* The bias form carries a linear layer's usual epilogue: an activation of the biased product and a
 * residual added after it -- each step rounded to the output's dtype, as three separate tensors
 * would be. Both optional; absent they are the plain biased GEMM. */
static const RadParamSpec pGemmBias[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
                                          P_STRO("act") };
static const RadOperandSpec oGemmBias[] = { OPD("a"), WGT("b"), WGT("bias"), OPD_O("res"),
                                            OUT("y") };

static const RadParamSpec pGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                       P_STR("dtype") };
/* `a_sum` and `b_ref` are libr4d's, appended AFTER `y` so docs/OPS.md's list stays a positional
 * prefix: the asymmetric 4-bit and 2-bit grids subtract the activation's per-group code sums, and
 * mxfp4 folds a per-output-row reference exponent. Neither is expressible as a weight aux stream,
 * because RadLayout can declare one and RadArgs has no way to deliver it. Ref implements no grid
 * that needs either, so it ignores both -- which is what an optional operand means. */
static const RadOperandSpec oGemmQ[]     = { OPD("a"), OPD_O("a_scale"), WGT("b"), WGT("b_scale"),
                                             OUT("y"), OPD_O("a_sum"), WGT_O("b_ref") };
static const RadOperandSpec oGemmQBias[] = { OPD("a"), OPD("a_scale"), WGT("b"), WGT("b_scale"),
                                             WGT("bias"), OUT("y") };

/* ---- attention ---------------------------------------------------------------------------- */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"), P_INTD("block_size"), P_INT("causal"),
    P_INT("window"), P_STR("q_dtype"), P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
    /* Three the SCRATCH HOOK needs at declare, when an op has parameters and no tensors: libr4d's
     * split-KV partial buffer is num_seqs * q_len * q_heads * splits rows, and gqa gives the head
     * ratio rather than the count. Ref sizes no scratch and ignores all three; they are in its
     * schema because the two plugins must agree on this op's parameter list. */
    P_INTO("n_head"), P_INTO("max_seqs")
};
static const RadOperandSpec oAttnPaged[] = {
    OPD("q"), OPD("kv_cache"), OPD("block_table"), OPD("seqused"),
    OPD_O("k_descale"), OPD_O("v_descale"), OPD_O("cu_seqlens"), OUT("out")
};
/* kv_store's, and attn_paged's only in the sense that it is skipped there: `block_size` is
 * RAD_DERIVED on attn_paged now, and a constraint on a derived key the caller did not supply does
 * not take part in matching (core/plugin/select.cpp). Ref therefore answers "any" to spec §7.2's
 * question of what the paged block size is, which is not a number -- a core that resolves
 * attention to ref alone has to pick, and 16 is what the rest of the project is built around. On
 * kv_store, where the key stays RAD_REQUIRED, the constraint still does its job: it admits every
 * value and refuses only a query that does not name one. */
static const RadConstraint cPagedBlock[] = { RAD_CGE("block_size", 1) };

static const RadParamSpec pAttnDense[] = { P_INT("M"), P_INT("head_dim"), P_INT("gqa"),
                                           P_INT("causal"), P_STR("dtype"), P_F64O("scale"),
                                           P_INTO("max_seqlen") };
static const RadOperandSpec oAttnDense[] = { OPD("q"), OPD("k"), OPD("v"), OPD("cu_seqlens"),
                                             OUT("out") };

/* `k_scale` and `v_scale` are the descales an fp8 cache is read back with, so a store divides by
 * them; they are meaningless for a bf16 cache and optional for that reason. THE SCHEMA IS SHARED
 * ACROSS PLUGINS -- the loader refuses a plugin whose spelling of an op differs from the one that
 * fixed it first -- so they are here even though this reference store only serves bf16. */
static const RadParamSpec pKvStore[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                         P_INT("block_size"), P_STR("kv_dtype"),
                                         P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { OPD("k"), OPD("v"), OPD("slot_mapping"),
                                           INOUT("kv_cache") };

static const RadParamSpec pQueryFwht[] = { P_INT("M") };
static const RadOperandSpec oQueryFwht[] = { INOUT("q") };

static const RadParamSpec pRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                      P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                      P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections") };
static const RadOperandSpec oRope[] = { INOUT("qkv"), OPD("positions") };

/* The cos/sin plane a fused prologue reads instead of theta: written at each position's OWN row
 * of a [max_ctx, rot] table. `rot`, `theta` and `scale` are `rope`'s, spelled the same way,
 * because the two have to compute the same angles. */
static const RadParamSpec pRopeTable[] = { P_INT("M"), P_INT("rot"), P_F64("theta"),
                                           P_F64("scale"), P_STR("dtype") };
static const RadOperandSpec oRopeTable[] = { OPD("positions"), OUT("cos_sin") };

static const RadParamSpec pQkNormRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                            P_INT("n_head_kv"), P_F64("theta"), P_F64("eps"),
                                            P_F64O("scale"), P_STRO("mode"), P_INTO("rotary_dim"),
                                            P_F64O("wadd") };
static const RadOperandSpec oQkNormRope[] = { INOUT("qkv"), OPD("positions"), WGT("q_w"),
                                              WGT("k_w") };

/* ---- mixture of experts ------------------------------------------------------------------- */
static const RadParamSpec pScaleRows[] = { P_INT("M"), P_INT("n"), P_STR("act"),
                                           P_STR("dtype") };
static const RadOperandSpec oScaleRows[] = { OPD("x"), OPD("s"), OPD_O("add"), OUT("y") };

static const RadParamSpec pRouter[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                        P_INT("norm"), P_STR("dtype") };
static const RadOperandSpec oRouter[] = { OPD("logits"), OUT("expert_ids"), OUT("expert_w") };

/* `expert_base` is THIS RANK'S FIRST EXPERT ID, and it is what makes expert parallelism a
 * parameter rather than a second op. Absent (or 0) is the whole-model case every non-sharded
 * deployment has always issued. See the schema note. */
static const RadParamSpec pScatter[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                         P_INTO("expert_base"),
                                         P_STRO("protect") };
static const RadOperandSpec oScatter[] = { OPD("expert_ids"), OUT("sorted_tok"),
                                           OUT("expert_offset"), OUT("expert_count") };

/* `a_order` SAYS WHICH ROWS THE ACTIVATION HAS, and it is a parameter because the two grouped
 * GEMMs of one routed block disagree about it. The gate_up projection reads the block's INPUT,
 * which is one row per TOKEN, and a sorted row i must gather `a[sorted_tok[i] / top_k]`. The down
 * projection reads the FIRST projection's output, which is already one row per (token, slot) in
 * the scatter's order -- row i is row i, and gathering it again reads another slot's activation.
 *
 * It has to be said rather than derived. The row count would give it away -- M against M*top_k --
 * but an operand whose meaning changes with its extent is exactly the shape a checker cannot see:
 * if both implementations gather they agree with each other, and a harness that sizes the operand
 * the way the gather wants never exercises the other reading. */
static const RadParamSpec pMoeGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                         P_INT("top_k"), P_STR("a_order"), P_STR("dtype") };
static const RadOperandSpec oMoeGemm[] = { OPD("a"), WTAB("w"), OPD("sorted_tok"),
                                           OPD("expert_offset"), OUT("y") };

/* The quantised arm, and it is a SEPARATE OP for the same reason `gemm_nt_q` is separate from
 * `gemm_nt`: the operand list differs -- an activation scale stream and a weight scale plane per
 * expert -- and an op whose operand count depends on a dtype string is an op whose positional
 * arguments mean different things to different kernels. `group` is the block edge, 128 for every
 * DeepSeek-style fp8 checkpoint. Both weight operands are tables and both carry one entry per
 * expert, which is what lets the declaration split evenly across them. */
/* THE EXPERTS ARE TWO TABLES BY PARITY -- `w`/`w_scale` the even ones at [N, K], `w_odd`/
 * `w_odd_scale` the odd ones at [N_odd, K_odd], expert e at entry e / 2 -- because a table's
 * entries share one geometry and a rank holding an uneven slice of each expert's width has two.
 * `parts` stacked parts run along both the weight rows and the output columns; the output is the
 * wider class's width, and a narrower class fills each part's leading columns and zeros the rest.
 * libr4d's schema states the same. */
static const RadParamSpec pMoeGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                          P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                          P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"),
                                          P_INTO("parts") };
static const RadOperandSpec oMoeGemmQ[] = { OPD("a"), OPD_O("a_scale"), WTAB("w"), WTAB("w_scale"),
                                            OPD("sorted_tok"), OPD("expert_offset"),
                                            WTAB("w_odd"), WTAB("w_odd_scale"), OUT("y") };
/* THE SAME OP OVER FOUR CLASSES, by the expert's number mod 4: `w`/`w_scale` class 0 at [N, K],
 * `w_q`/`w_q_scale` class q at [N_q, K_q], expert e at entry e / 4 of class e % 4 -- how four
 * ranks take turns with the extra block of a five-block expert. libr4d's schema states the same. */
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

/* `shared` and `shared_gate` PRESENT = the shared arm folded in, `y = bf16(shared * act(gate)) +
 * bf16(gathered)`, which is scale_rows' fused gate-and-add taken one op further back. `act` is
 * that op's spelling, sigmoid or none, and applies to the SCALAR. */
static const RadParamSpec pMoeGather[] = { P_INT("M"), P_INT("n"), P_INT("top_k"),
                                           P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oMoeGather[] = { OPD("y_expert"), OPD("expert_w"), OPD("sorted_tok"),
                                             OPD_O("shared"), OPD_O("shared_gate"), OUT("y") };

/* `vocab_off` and `pairs` are the vocab-parallel form: `x` is one rank's slice, `idx` carries
 * GLOBAL columns, and `pairs` is the [M, R, 2] (int32 id, float value) plane an all_gather moves.
 * Both absent is the single-rank top-R unchanged. AN OP SCHEMA IS A CONTRACT ACROSS PLUGINS -- if
 * this and libr4d's disagree the loader refuses ref and the arch plugin's model stops loading with
 * a message that points nowhere near the cause. */
static const RadParamSpec pRowTopk[] = { P_INT("M"), P_INT("N"), P_INT("R"), P_INTO("vocab_off"),
                                         P_STR("dtype") };
static const RadOperandSpec oRowTopk[] = { OPD("x"), OUT("idx"), OUT("val"), OUT_O("pairs") };

static const RadParamSpec pRowTopkMerge[] = { P_INT("M"), P_INT("R"), P_INT("world_size"),
                                              P_STR("dtype") };
static const RadOperandSpec oRowTopkMerge[] = { OPD("gathered"), OUT("idx"), OUT("val") };

static const RadParamSpec pLogitRerank[] = { P_INT("M"), P_INT("R"), P_INT("n_vocab"),
                                            P_INT("n_embd"), P_INTO("vocab_off"), P_STR("dtype") };
static const RadOperandSpec oLogitRerank[] = { OPD("x"), WGT("head"), OPD("idx_in"),
                                              OUT("idx"), OUT("val"), OUT_O("pairs") };

/* ---- gated delta net ----------------------------------------------------------------------
 *
 * `cu` IS AN OPERAND OF EVERY GDN OP, DECODE INCLUDED, and `num_accepted` is REQUIRED on both
 * decode ops. The first is not a preference: the fast decode kernels open with
 * `bos = cu[n]; T = cu[n+1] - bos` before any null test, so a schema that omits `cu` does not make
 * it optional, it makes the launch fault. The second is
 * spec §10: the recurrent state and the conv window have already absorbed the rejected tokens, so
 * a rejection is a change of read offset, and an op that lets the caller omit `num_accepted` is an
 * op whose rollback contract is optional. A non-speculative decode passes all ones. */
static const RadParamSpec pConvPrep[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                          P_CHUNK("chunk"), P_INT("conv_width"),
                                          P_F64O("l2_eps"), P_F64O("softplus_thr") };
/* THE INPUTS COME FIRST AND THE OUTPUTS LAST, which is what docs/OPS.md's resolved table draws.
 * `a` and `b_gate` are the gate and beta projections a layer may hand over separately from `x`;
 * absent, they are the last 2*head_v_count columns of the fused `x` row. `cache_idx` is the conv
 * state slot per sequence -- a paged conv cache cannot be indexed by the batch's sequence number
 * -- and `has_init` says whether that slot holds a history at all. */
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

/* gdn_conv_update folded into gdn_recurrent_update: the convolution's inputs, then the recurrence's
 * minus q, k and v, which are the pair's intermediate. The parameters are the union of the two
 * ops' under their own names. */
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
                                           P_INTO("hops") };
/* `y` is an optional SECOND operand: absent means in place, which is what docs/OPS.md draws and
 * what every one-shot kernel does. libr4d's two-shot rows are not safe in place -- a rank gathers
 * into the output while its own input is still being read for the scatter -- so they need the
 * separate destination, and a schema without it cannot express their call at all. */
static const RadOperandSpec oAllReduce[] = { INOUT("x"), OUT_O("y") };

static const RadParamSpec pAllGather[] = { P_INT("world_size"), P_INT("numel"), P_STR("dtype"),
                                           P_INTO("row") };
static const RadOperandSpec oAllGather[] = { OPD("x"), OUT("y") };

/* ---- vocabulary edges --------------------------------------------------------------------- */
/* `vocab_offset` MAKES THE EMBEDDING VOCAB-PARALLEL, which is the symmetry lm_head has had all
 * along. Absent (or 0 with an unsharded table) it is exactly the gather it always was.
 *
 * Present, `n_vocab` is THIS RANK'S row count -- the same meaning it already carries for
 * logits_gemm -- and the table covers GLOBAL ids [vocab_offset, vocab_offset + n_vocab). Token ids
 * stay GLOBAL, so a rank subtracts the offset and writes a ZERO ROW for an id outside its slice,
 * exactly as it does for the negative padding id. Every token lands on exactly one rank, so
 * summing the ranks' outputs reconstructs x -- which is why the caller follows this with an
 * all_reduce and why the reduction is exact rather than approximate.
 *
 * The cost of NOT having it: the table is replicated on every card. 248320 x 5120 in bf16 is
 * 2.43 GiB, and it is bf16 even in the fp8 checkpoint. That is KV cache. */
static const RadParamSpec pEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                       P_STR("dtype"), P_INTO("vocab_offset"),
                                       /* embed_lookup_q's: the scale its LAYOUT quantises with.
                                        * A parameter because a global scale needs the whole table
                                        * and the transform sees one block. */
                                       P_F64O("wscale") };
static const RadOperandSpec oEmbed[] = { OPD("tokens"), WGT("wte"), OUT("x") };
/* The quantised table: one scale for all of it. A SEPARATE OP and not a dtype on the above --
 * the operand list differs and operands are positional, the same argument gemm_nt_q makes. */
static const RadOperandSpec oEmbedQ[] = { OPD("tokens"), WGT("wte"), WGT_O("scale"), OUT("x"),
                                          OPD_O("ahead") };

static const RadParamSpec pLogits[] = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                        P_STR("dtype") };
static const RadOperandSpec oLogits[] = { OPD("x"), WGT("lm_head"), OUT("logits") };

/* ---- sampling -------------------------------------------------------------------------------
 *
 * EVERY SCALAR IS PER ROW AND ARRIVES AS AN OPERAND, not as a parameter. A parameter is geometry:
 * the selector matches on it and freezes it at declare (spec §2.2), so `top_p` as a parameter
 * would make two requests with different top_p two different resolutions of one op, and a
 * deployment would have exactly one top_p. `params` is [M] rows of RadSampleParams
 * (abi/rad_sample.h), uploaded once a step.
 *
 * This is the schema core/sample/sampler.h issues against, and it is the correction of the one
 * docs/OPS.md prints -- `rep`/`freq`/`pres`/`k`/`p` as parameters, and `seed`/`offset` as two more
 * operands of sample_pick. A plugin declaring against that form cannot start: the parameters it
 * names are per-request values, so declare refuses them as missing geometry. */
static const RadParamSpec pMVocab[] = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]  = { P_INT("M"), P_INT("n_cand") };
/* The history stages read GLOBAL token ids and write this rank's columns, so under a split
 * vocabulary they take the rank's first row as `vocab_off`, the parameter sample_topk carries for
 * the same crossing in the other direction. Absent is 0: the whole vocabulary. */
static const RadParamSpec pPenal[]  = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };

/* `history` is [M, max_hist] int32 of global token ids; row m's window is addressed by row m's
 * hist_off / hist_len within row m. */
static const RadOperandSpec oPenalties[] = { INOUT("logits"), OPD("params"), OPD("history") };
/* `breakers` is optional: [M, max_hist] u8 beside `history`, nonzero where that position's token
 * is a whole DRY sequence breaker and therefore never penalised. Absent, no token is exempt. */
static const RadOperandSpec oDry[]       = { INOUT("logits"), OPD("params"), OPD("history"),
                                             OPD_O("breakers") };
static const RadOperandSpec oTemp[]      = { INOUT("logits"), OPD("params") };
/* `bitmask` is [rows, ceil(n_vocab/32)] uint32, addressed by the row's mask_row -- which is -1 for
 * an unconstrained sequence, so the plane is shared and shorter than M. */
static const RadOperandSpec oMask[]      = { INOUT("logits"), OPD("params"), OPD("bitmask") };
/* `pairs` is the VOCAB-PARALLEL path and is optional: [M, 2 * n_cand] u32 holding this rank's
 * candidates as (u32 GLOBAL id, f32 logit), which is exactly what sample_merge_topk reads once
 * the all-gather has put every rank's side by side. `vocab_off` is added by the kernel, so the
 * gather in between never has to know the shard layout. Absent, and the op is the single-rank
 * top-k it has always been. */
static const RadParamSpec pTopk[]        = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadOperandSpec oTopk[]      = { OPD("logits"), OPD("params"),
                                             OUT("cand_idx"), OUT("cand_val"), OUT_O("pairs") };
static const RadOperandSpec oArgmax[]    = { OPD("logits"), OPD("params"), OUT("token") };
static const RadOperandSpec oNarrow[]    = { INOUT("cand_idx"), INOUT("cand_val"), OPD("params") };
static const RadOperandSpec oPick[]      = { OPD("cand_idx"), OPD("cand_val"), OPD("params"),
                                             OUT("token") };
/* `gathered` is [M, world_size * n_cand] (u32 id, f32 logit) pairs with GLOBAL token ids -- each
 * rank adds its own vocab offset before the gather, so the merge needs no shard layout. */
static const RadOperandSpec oMerge[]     = { OPD("gathered"), OPD("params"),
                                             OUT("cand_idx"), OUT("cand_val") };

/* ================================================================== the schema table */
static const RadOpSchema kSchemas[] = {
{ "rmsnorm", ARR(pRmsnorm), ARR(oRmsnorm),
  "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). `wadd` is not in docs/OPS.md's elementwise table: "
  "it is GemmaRMSNorm's `1 +`, which Qwen3.5, Qwen3-Next and Gemma all need, and it defaults to 0. "
  "Spelled `wadd` and not `w_add` because that is what libr4d's fused norms call it and one "
  "vocabulary cannot have two spellings of one parameter." },
{ "rmsnorm_add", ARR(pRmsnorm), ARR(oRmsnormAdd),
  "residual_out = x + residual, then y = rmsnorm(residual_out) * w. The variance is taken from "
  "the unrounded f32 sum; the multiply reads residual_out back AFTER it has been narrowed, which "
  "is what every fused implementation does." },
{ "hc_enter", ARR(pHcWrite), ARR(oHcEnter),
  "h[s] = x for each of the hc streams -- the wide residual stream's first value. Shares `hc_write`'s parameter list." },
{ "hc_read", ARR(pHcRead), ARR(oHcRead),
  "The read half of a gated residual. N = grouped_rmsnorm(h) -- one rsqrt per `n`-wide stream, "
  "one gain of hc*n over all of them, `1 + wadd*w`; g = sigmoid(mix_up @ silu(mix_down @ N / hc)); "
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
{ "add", ARR(pMnD), ARR(oBinary),
  "y = a + b. An operand carrying exactly n elements against a taller output broadcasts along "
  "rows; nothing else broadcasts." },
{ "mul", ARR(pMnD), ARR(oBinary), "y = a * b, with the same one broadcast rule as `add`." },
{ "silu_mul", ARR(pMnD), ARR(oGateUp),
  "y[i] = silu(gate_up[i]) * gate_up[n + i]. The GATE IS THE FIRST HALF, which is the order "
  "rad-convert fuses gate and up into one weight in." },
{ "gelu", ARR(pMnD), ARR(oUnary),
  "y = 0.5x(1 + erf(x/sqrt(2))). The exact erf form, not the tanh approximation." },
{ "silu", ARR(pMnD), ARR(oUnary), "y = x * sigmoid(x)." },
{ "sigmoid", ARR(pMnD), ARR(oUnary), "y = 1 / (1 + exp(-x))." },
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
  "Symmetric int8 per group of `group` contiguous elements: q = round(x * 127 / amax), "
  "scale = amax / 127, scale shaped [M, ceil(n/group)]. group <= 0 or >= n is one scale per row. "
  "`asum` present selects libr4d's asymmetric arm and REF DECLINES THAT FORM: it writes one scale "
  "per ROW over the whole of n -- there `group` is the asum grouping width and not the scale "
  "granularity -- and stores q in fragment order, so neither of its planes means what this one's "
  "does." },
/* NOT one of docs/OPS.md's original ops: it comes from a kernel plugin, and ref adopts the schema
 * libr4d/r4d_rows.cpp fixed rather than inventing a second one. Ref implements it so the FP8 path
 * has an ORACLE -- without one, the activation format the production checkpoints ship in is the
 * one format rad-kbench cannot measure. */
{ "quant_act_fp8", ARR(pQuant), ARR(oQuantFp8),
  "Symmetric E4M3 per group of `group` contiguous elements: scale = amax / 448 (a DEQUANT "
  "multiplier, the same sense as a checkpoint's weight_scale_inv), q = clamp(x / scale, +-448), "
  "scale shaped [M, ceil(n/group)] in f32. An all-zero group gets scale 1, not scale 0: the GEMM "
  "multiplies this scale by the weight's block scale, and a zero here would erase that one. Rows "
  "are dim 0, so a rank-3 activation is one row per token and not one per head." },
/* The rotated fp8 twin, and its scale convention is NOT had_quant_act_i8's: the group is both the
 * rotation width and the scale granularity, and the residual factor of `group` is NOT normalised
 * here at all -- the weight plane it pairs with (dtype w4a8h) carries the whole of it, where the
 * division is an exponent shift on a bf16 and therefore exact. See libref/ref_quant.cpp. */
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
{ "had_quant_act_fp8", ARR(pQuant), ARR(oQuantFp8),
  "A Walsh-Hadamard rotation of width `group` along n, then symmetric E4M3 per group: scale = "
  "amax(rotated) / 448, shaped [M, n/group] in f32. n must be a multiple of `group` -- a rotation "
  "block cannot straddle the end of a row. NOT interchangeable with quant_act_fp8 in either "
  "direction: the consumer must be a weight that was rotated the same way." },
/* The calibration op. It is the only entry in this vocabulary that no forward pass issues: it
 * runs under a calibration flag and its output is read at CONVERT time, in another process,
 * through a file. `dtype` is the INPUT's -- fp8a8 where the tap is a quantiser's output, and
 * bf16, with no `a_scale`, where the consuming GEMM contracts a plain activation. */
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
  "y = q * scale, the inverse of quant_act_i8. It does NOT undo a rotation." },

{ "gemm_nt", ARR(pGemm), ARR(oGemm), "y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate." },
{ "gemm_nt_bias", ARR(pGemmBias), ARR(oGemmBias),
  "gemm_nt with a per-column bias added in f32, rounded to y's dtype; then, optionally, `act` of "
  "that (\"gelu\" exact, \"gelu_tanh\" the tanh approximation, \"none\"), rounded; then `res` "
  "added, rounded. Each step is what a separate tensor would hold." },
{ "gemm_nt_q", ARR(pGemmQ), ARR(oGemmQ),
  "gemm_nt with quantised operands. The group sub-sum is accumulated from the raw codes and "
  "scaled once per group, which is what an int8 or fp8 WMMA kernel does. Scale streams are "
  "[rows, groups] or [rows]; an mxfp4 weight's u8 scale stream is read as OCP e8m0." },
{ "gemm_nt_q_bias", ARR(pGemmQ), ARR(oGemmQBias), "gemm_nt_q with a per-column bias." },

{ "attn_paged", ARR(pAttnPaged), ARR(oAttnPaged),
  "Paged attention over kv_cache[num_blocks, kv_heads, block_size, 2*head_dim] (K then V per "
  "slot), f32 softmax with the max subtracted, causal and sliding-window masking, q_len >= 1 so "
  "one kernel serves decode, speculative verify and prefill. seqused is the TOTAL key count "
  "including this step's query rows. k_descale / v_descale are optional and apply to an fp8 "
  "cache. `scale` is not in docs/OPS.md and defaults to 1/sqrt(head_dim)." },
{ "attn_dense", ARR(pAttnDense), ARR(oAttnDense),
  "Dense varlen attention over cu_seqlens segments -- the vision tower's. Nothing paged." },
{ "kv_store", ARR(pKvStore), ARR(oKvStore),
  "Writes k and v into the paged cache at slot_mapping; a negative slot is skipped. The cache is "
  "INOUT rather than the OUT docs/OPS.md draws, because the op touches only the mapped slots." },
{ "query_fwht", ARR(pQueryFwht), ARR(oQueryFwht),
  "In-place wave-local FWHT-256 with 1/16 scaling for queries under KVarN." },
{ "rope_table", ARR(pRopeTable), ARR(oRopeTable),
  "cos/sin for the positions a step names, at each position's own row of a [max_ctx, rot] plane: "
  "rot values a row, cos for the first half and sin for the second. The angles are `rope`'s "
  "exactly. A row no position names is not written; a negative position writes nothing." },
{ "rope", ARR(pRope), ARR(oRope),
  "Rotary embedding in place over the q and k heads of a fused [q|k|v] row; the v heads are left "
  "alone. `mode` is \"neox\" or \"gptj\", or a multi-component form: \"mrope\" (contiguous "
  "`sections` of frequencies, one ladder), \"imrope\" (Qwen3-VL's interleaved sections) or "
  "\"axial\" (contiguous sections, each on its own ladder -- a 2-D rotary). `positions` is [M], "
  "or [C, M] component-major for a multi-component mode; with [M] every component is that one. "
  "`scale` divides the position; `rotary_dim` is optional and defaults to head_dim." },
{ "qk_norm_rope", ARR(pQkNormRope), ARR(oQkNormRope),
  "Per-head RMS norm of q and k against their own gains, then rope. THE NORM OUTPUT IS ROUNDED "
  "TO THE TENSOR'S DTYPE BEFORE THE ROTATION, which is what the unfused pair does." },

{ "scale_rows", ARR(pScaleRows), ARR(oScaleRows),
  "y[m, i] = x[m, i] * act(s[m]): one scalar per ROW, broadcast ACROSS the row. `mul` broadcasts "
  "the other way -- one row down the rows -- and giving it both rules would make them ambiguous "
  "when rows == n. `act` is \"sigmoid\" or \"none\", and it applies to the SCALAR. With `add` "
  "present the form is y = bf16(x * act(s)) + add -- the product NARROWED before the sum, which "
  "is what makes the fused form bit-identical to the pair it replaces." },
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
  "ignored, as is `hops`. "
  "`y` absent is the in-place form; present, the sum lands in y and x is left alone." },
{ "all_gather", ARR(pAllGather), ARR(oAllGather),
  "y is [world_size * numel] with rank r's contribution at offset r * numel. `row` -- one row of "
  "a rank's contribution, in elements -- switches to the ROW-INTERLEAVED placement instead: x is "
  "[rows][row] and y is [rows][world_size][row], so one row of y carries every rank's words for "
  "that row. The vocab-parallel candidate merge reads exactly that." },

{ "embed_lookup", ARR(pEmbed), ARR(oEmbed),
  "A gather from wte[n_vocab, n_embd]. A negative token id writes a zero row (padding); an id at "
  "or above n_vocab fails the launch." },
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
};

/* ================================================================== the kernel table */
/* One row per op. `shape` says the same thing on every row on purpose: there is no geometry
 * this refuses, and saying so in the row is what a --debug-graph dump prints next to the
 * warning that an op fell through to ref. */
#define SHAPE_ANY "any: every extent is read off the operands, strides included"

/* THE OPERAND DESCRIPTION IS PASTED OUT OF THE LAUNCH and is not a twelfth macro argument.
 * `RadKernelInfo::opd_shape` is the last member, after `transform`, and it answers "how big is
 * operand i of this op at this geometry, and what may legally be in it" for a tool that wants to
 * call the op cold (rad_abi.h). It cannot be one shared function, because the hook is not handed
 * the op name and no op in this vocabulary carries its own name as a parameter -- so every row
 * needs its own. Naming each one after its entry point (`ref_rmsnorm` -> `ref_rmsnorm_shape`) means
 * `fn ## _shape` produces it here, so no row below carries it as an argument of its own. It also
 * means an op added later cannot forget: the macro demands the symbol, so a
 * row without a shape hook fails to link rather than silently making its op undescribable. The
 * implementations, and every judgement call in them, are in libref/ref_shapes.cpp. */
/* No tunables, on every row, deliberately. libref is the ORACLE and the last resort: there is one
 * implementation of each op and nothing to choose between, so there is no axis to declare and
 * nothing for rad-tune to measure. A reference with tile variants would be a reference whose
 * answer depends on which one ran. The three nulls are tunables, n_tunables and tune_valid. */
#define ROW(nm, opname, fam, what, dts, cons, ncons, init, fini, fn, scr)                        \
    { nm, opname, fam, what, SHAPE_ANY, dts, RAD_DOMAIN_HOST, 0,                                 \
      cons, ncons, nullptr, 0, nullptr, init, fini, fn, scr, nullptr, nullptr, fn##_shape }

#define SIMPLE(nm, opname, fam, what, dts, fn)                                                   \
    ROW(nm, opname, fam, what, dts, nullptr, 0, nullptr, nullptr, fn, nullptr)

#define DT_FLOAT "bf16 f16 f32 fp8_e4m3 fp8_e5m2 i8 i32, chosen per operand from its own dtype"
#define DT_ANY   "any dtype rad_types.h names, per operand"

static const RadKernelInfo kKernels[] = {
SIMPLE("ref_rmsnorm", "rmsnorm", "norm", "root-mean-square norm with a gain", DT_FLOAT,
       ref_rmsnorm),
SIMPLE("ref_rmsnorm_add", "rmsnorm_add", "norm", "residual add then RMS norm", DT_FLOAT,
       ref_rmsnorm_add),
SIMPLE("ref_hc_enter", "hc_enter", "norm", "gated residual entry: hc copies of x", DT_FLOAT,
       ref_hc_enter),
SIMPLE("ref_hc_read", "hc_read", "norm", "gated residual read: grouped norm, low-rank gate, stream mix",
       DT_FLOAT, ref_hc_read),
SIMPLE("ref_mtp_enter", "mtp_enter", "norm",
       "MTP head entry: two norms, a per-sub-stream fc and a broadcast fc", DT_FLOAT,
       ref_mtp_enter),
SIMPLE("ref_hc_write", "hc_write", "norm", "gated residual write: h[s] += inj[s] * y",
       DT_FLOAT, ref_hc_write),
SIMPLE("ref_layernorm", "layernorm", "norm", "layer norm with gain and bias", DT_FLOAT,
       ref_layernorm),
SIMPLE("ref_grid_embed", "grid_embed", "elem",
       "bilinear resample of a learned position grid, added in", DT_FLOAT, ref_grid_embed),
SIMPLE("ref_add", "add", "elem", "elementwise sum", DT_FLOAT, ref_add),
SIMPLE("ref_mul", "mul", "elem", "elementwise product", DT_FLOAT, ref_mul),
SIMPLE("ref_silu_mul", "silu_mul", "elem", "silu-gated product over a fused gate_up", DT_FLOAT,
       ref_silu_mul),
SIMPLE("ref_gelu", "gelu", "elem", "exact erf gelu", DT_FLOAT, ref_gelu),
SIMPLE("ref_silu", "silu", "elem", "silu", DT_FLOAT, ref_silu),
SIMPLE("ref_sigmoid", "sigmoid", "elem", "logistic sigmoid", DT_FLOAT, ref_sigmoid),
SIMPLE("ref_ngram_ids", "ngram_ids", "vocab", "hashed n-gram row ids for PLE", DT_ANY,
       ref_ngram_ids),
SIMPLE("ref_ple_gate", "ple_gate", "norm", "PLE gate: three grouped norms, a dot, a signed sqrt",
       DT_FLOAT, ref_ple_gate),
SIMPLE("ref_ple_conv", "ple_conv", "norm", "PLE's dilated depthwise conv, silu and residual add",
       DT_FLOAT, ref_ple_conv),
SIMPLE("ref_gather_rows", "gather_rows", "elem", "y[i,:] = x[idx[i],:]", DT_ANY, ref_gather_rows),
SIMPLE("ref_scatter_rows", "scatter_rows", "elem", "x[idx[i],:] = v[i,:]", DT_ANY, ref_scatter_rows),
SIMPLE("ref_cast", "cast", "elem", "dtype conversion", DT_ANY, ref_cast),
SIMPLE("ref_softmax", "softmax", "elem", "row-wise softmax", DT_FLOAT, ref_softmax),

SIMPLE("ref_quant_act_i8", "quant_act_i8", "quant", "symmetric int8 activation quantiser",
       "float in, i8 out, f32 scale", ref_quant_act_i8),
SIMPLE("ref_quant_act_fp8", "quant_act_fp8", "quant", "symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale", ref_quant_act_fp8),
/* THE FUSED FORMS, AND THEY EXIST SO A HOST RUN IS POSSIBLE AT ALL. libr4d fuses the fp8
 * activation quantiser into whichever kernel already has the row in registers and declares the
 * fused op as its own; without host kernels for those names, RADIANCE_HOST_DOMAIN -- the only
 * thing that can answer "is the GRAPH right", because the per-op oracle reads the same graph on
 * both sides -- cannot start on an fp8 model. */
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
SIMPLE("ref_had_quant_act_fp8", "had_quant_act_fp8", "quant",
       "width-`group` Hadamard, then the same symmetric block-scaled E4M3 quantiser",
       "float in, fp8_e4m3 out, f32 scale per group; n a multiple of group",
       ref_had_quant_act_fp8),
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
SIMPLE("ref_gemm_nt_q", "gemm_nt_q", "gemm", "C = A B^T with group-scaled quantised operands",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; f32 accumulate", ref_gemm_nt_q),
SIMPLE("ref_gemm_nt_q_bias", "gemm_nt_q_bias", "gemm", "quantised C = A B^T + bias",
       "i8 / fp8 / float A; w4, w2, mxfp4, i8 or float B; f32 accumulate", ref_gemm_nt_q_bias),

ROW("ref_attn_paged", "attn_paged", "attn",
    "paged causal / sliding-window attention, any query length",
    "any q and out dtype; bf16 or fp8 kv cache with optional per-(seq,head) descales",
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])),
    nullptr, nullptr, ref_attn_paged, nullptr),
SIMPLE("ref_attn_dense", "attn_dense", "attn", "dense varlen attention over cu_seqlens", DT_FLOAT,
       ref_attn_dense),
ROW("ref_kv_store", "kv_store", "attn", "scatter k and v into the paged cache", DT_ANY,
    cPagedBlock, (int)(sizeof(cPagedBlock) / sizeof(cPagedBlock[0])),
    nullptr, nullptr, ref_kv_store, nullptr),
SIMPLE("ref_query_fwht", "query_fwht", "attn", "in-place FWHT-256 for queries", DT_FLOAT, ref_query_fwht),
SIMPLE("ref_rope", "rope", "attn", "rotary embedding in place", DT_FLOAT, ref_rope),
SIMPLE("ref_qk_norm_rope", "qk_norm_rope", "attn", "per-head QK norm then rotary embedding",
       DT_FLOAT, ref_qk_norm_rope),

SIMPLE("ref_scale_rows", "scale_rows", "elem", "one scalar per row, broadcast across the row",
       DT_FLOAT, ref_scale_rows),
SIMPLE("ref_router_topk", "router_topk", "moe", "softmax then exact top-k routing", DT_FLOAT,
       ref_router_topk),
SIMPLE("ref_moe_scatter", "moe_scatter", "moe", "stable counting sort of routing slots by expert",
       "i32 indices", ref_moe_scatter),
SIMPLE("ref_moe_gemm", "moe_gemm", "moe", "grouped GEMM over the expert-sorted rows", DT_FLOAT,
       ref_moe_gemm),
SIMPLE("ref_moe_gemm_q", "moe_gemm_q", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands", DT_FLOAT, ref_moe_gemm_q),
SIMPLE("ref_moe_gemm_q_mod4", "moe_gemm_q_mod4", "moe",
       "grouped GEMM over the expert-sorted rows, quantised operands, four classes by number mod 4",
       DT_FLOAT, ref_moe_gemm_q_mod4),
SIMPLE("ref_moe_gather", "moe_gather", "moe", "weighted scatter back to token order", DT_FLOAT,
       ref_moe_gather),
SIMPLE("ref_rope_table", "rope_table", "attn", "cos/sin at each position's own row", DT_FLOAT, ref_rope_table),
SIMPLE("ref_row_topk", "row_topk", "moe", "exact per-row top-R", DT_FLOAT, ref_row_topk),
SIMPLE("ref_row_topk_merge", "row_topk_merge", "moe", "the global top-R of the gathered per-rank candidate sets", DT_FLOAT, ref_row_topk_merge),
SIMPLE("ref_logit_rerank", "logit_rerank", "moe", "R candidates a row rescored exactly against the full-precision head", DT_FLOAT, ref_logit_rerank),

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
       "gdn decode recurrence against the paged state cache", DT_FLOAT, ref_gdn_recurrent_update),
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
SIMPLE("ref_logits_gemm", "logits_gemm", "vocab", "the lm_head projection", DT_FLOAT,
       ref_logits_gemm),

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
SIMPLE("ref_sample_dry", "sample_dry", "sample", "DRY repetition-suffix penalty", DT_FLOAT,
       ref_sample_dry),
SIMPLE("ref_sample_typical", "sample_typical", "sample", "locally typical narrowing in place",
       DT_FLOAT, ref_sample_typical),
SIMPLE("ref_sample_xtc", "sample_xtc", "sample", "exclude-top-choices narrowing in place",
       DT_FLOAT, ref_sample_xtc),
SIMPLE("ref_sample_merge_topk", "sample_merge_topk", "sample",
       "vocab-parallel candidate merge", DT_FLOAT, ref_sample_merge_topk),
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
/* The oracle and the last resort, said through the ABI as any library would say it
 * (RadPluginReferenceFn): the engine knows no library by name. */
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
