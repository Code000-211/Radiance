/* r4d_rows.cpp -- the libr4d row table, and the op schemas that go with it.
 *
 * This is the single declaration of what libr4d can serve: a kernel missing from this table is
 * invisible to callers, and a second table would drift.
 *
 * A row's constraints restate its entry point's rules, and each row says where its band came from.
 * Three things differ from the kernel side's spelling, each marked at the row:
 *
 *   1. Key names. Op schemas use docs/OPS.md's vocabulary (what architecture plugins write and
 *      libref implements). Where the kernel's select() differs, the predicate and value are the
 *      kernel's and only the key is renamed, with the kernel-side key in a comment: `K` -> `n` in
 *      the quantiser family, `had` -> `group` for the rotation width, and `q_dtype`/`kv_dtype` ->
 *      `dtype` for the dense vision kernel.
 *
 *   2. Preference between matching rows is `priority`, not position: the R4D_PRIO_* constants,
 *      each naming the kernel-side ordering it preserves (r4d_plugin.h).
 *
 *   3. Some bands come from the entry point itself (the tiled and prefill W4A8/W8A8 GEMMs, the
 *      asymmetric grids, `quant_act_i8_asum`, `gemm_w8a8_nt_m64`): read off its rejection tests
 *      and header rather than a kernel-side registry row.
 *
 * Every row is RAD_DOMAIN_DEVICE: libr4d is a gfx1201 kernel library with no host execution site.
 */

#include "r4d_plugin.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define ROW(a) (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Parameter kinds. Real-valued parameters (eps, scale, theta, softplus_thr, l2_eps, wadd) are
 * RAD_P_F64 (docs/OPS.md, "The four rules"). RAD_DERIVED is a value the kernel supplies, not the
 * caller (`block_size`, `chunk`); a constraint on one is skipped during matching and read back off
 * the winner (spec §7.2). */
#define P_INT(k)   { (k), RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { (k), RAD_P_INT, RAD_OPTIONAL }
#define P_INTD(k)  { (k), RAD_P_INT, RAD_DERIVED }
/* The linear-state tile length. The scheduler cuts its steps on a multiple of it (rad_abi.h,
 * RAD_PROLE_SEQ_CHUNK); nothing else in the core reads a parameter by meaning. */
#define P_CHUNK(k) { (k), RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }
#define P_STR(k)   { (k), RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { (k), RAD_P_STR, RAD_OPTIONAL }
#define P_F64(k)   { (k), RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { (k), RAD_P_F64, RAD_OPTIONAL }

/* Spelled as the kernel side spells its constraints, so a row compares line for line. */
#define C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }

/* Required by every kernel that touches a peer's memory: the collectives write the other ranks'
 * scratch and flags directly. The engine measures this at launch (rad_types.h, device.p2p). */
#define C_P2P       C_EQ(RAD_FACT_P2P, 1)

/* libr4d's R4D_AR_TWOSHOT_WIDE_MAX_ELEMS, restated because this file must not include r4d.h.
 * r4d_ar_entry.hip static_asserts the two against each other. */
#define R4D_AR_TWOSHOT_WIDE_MAX_ELEMS 268435456

/* The largest message, in elements (1 MiB of a 16-bit payload), a 4- or 8-rank exact all-reduce
 * keeps on the one-shot; above it the two-shot rows take it. The one-shot sends (ws-1)*N bytes per
 * rank in one hop, the two-shot 2*(ws-1)/ws*N in two, so the one-shot wins while the second hop's
 * latency outweighs the bytes saved. The value separates decode from prefill rather than marking a
 * measured crossover: on Qwen3.8-Flash-Next (5 KiB per row) the largest decode message is 640 KiB
 * (32 sequences verifying four tokens) and a 2048-token prefill chunk is 10 MiB. A multiple of 128
 * (see pAllReduce's `numel`). */
#define R4D_AR_WIDE_ONESHOT_MAX_ELEMS 524288
/* The same bound at eight ranks, where the one-shot's (ws-1)*N is seven times the message and
 * loses early: on eight R9700 the two-shot is faster from 16384 elements on (21.6 against 26.3
 * us), at a 48-row decode message of 4096 wide 3.4 times. A multiple of 64, the 8-rank
 * two-shot's DIV. */
#define R4D_AR_8RANK_ONESHOT_MAX_ELEMS 8192
#define R4D_XSTR(v) R4D_STR(v)
#define R4D_STR(v)  #v

/* ==================================================================================== schemas
 *
 * The first plugin in hierarchy order to declare an op fixes its schema (spec §2.3), and this
 * plugin sits ahead of `libref`. An operand or parameter that docs/OPS.md does not list is a
 * vocabulary change and is called out at the schema: ref must grow the same one, or the two
 * plugins are refused against each other at load.
 */

/* -------------------------------------------------------------------------------- attention */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"),
    P_INT("head_dim"),
    P_INT("gqa"),
    /* Supplied by the kernel (spec §7.2): the block manager reads the paged block size off the
     * resolved attention kernel, so the resolving query cannot state it. libr4d answers 16. */
    P_INTD("block_size"),
    P_INT("causal"),
    P_INT("window"),
    P_STR("q_dtype"),
    P_STR("kv_dtype"),
    /* The softmax scale, a model constant. Absent, the shim uses 1/sqrt(head_dim). */
    P_F64O("scale"),
    /* Not in docs/OPS.md. Needed at declare: the split-KV scratch hook sizes the arena before
     * there is a batch, and the split law reads the context bound. Absent, the shim uses the
     * block table's width, a safe upper bound. */
    P_INTO("max_ctx"),
    /* Also for the scratch hook, which runs at declare with parameters but no tensors: the
     * split-KV partial buffer is num_seqs * q_len * q_heads * splits rows, and gqa gives only the
     * head ratio. `max_seqs` is the bound, since the scratch is sized once for the worst step.
     * Absent, the shim falls back to the tensors, which exist only at issue. */
    P_INTO("n_head"),
    P_INTO("max_seqs"),
};
static const RadOperandSpec oAttnPaged[] = {
    { "q",           RAD_OPD_IN,  0 },
    { "kv_cache",    RAD_OPD_IN,  0 },
    { "block_table", RAD_OPD_IN,  0 },
    { "seqused",     RAD_OPD_IN,  0 },
    { "k_descale",   RAD_OPD_IN,  1 },   /* absent for a bf16 cache */
    { "v_descale",   RAD_OPD_IN,  1 },
    { "cu_seqlens",  RAD_OPD_IN,  1 },   /* absent = one q_len for the whole batch */
    { "out",         RAD_OPD_OUT, 0 },
};

/* attn_paged with gate_quant_fp8 folded into its merge: attn_paged's parameters plus the two of
 * gate_quant_fp8's that are not implied (its `n` is `head_dim`, its rows attn_paged's (token,
 * head) pairs), then attn_paged's operands, the gate, the codes and the scales. `out` holds the
 * gated bf16 product, as the in-place pair leaves it. */
static const RadParamSpec pAttnPagedGq[] = {
    P_INT("q_len"),
    P_INT("head_dim"),
    P_INT("gqa"),
    /* Required here, where attn_paged derives it: the caller passes on the page size read off
     * the resolved attn_paged kernel. */
    P_INT("block_size"),
    P_INT("causal"),
    P_INT("window"),
    P_STR("q_dtype"),
    P_STR("kv_dtype"),
    P_F64O("scale"),
    P_INTO("max_ctx"),
    P_INTO("n_head"),
    /* A capacity: the row launches by the block table's rows and checks its scratch against them,
     * so a declare may bound it lower. The gated attention block sizes this op per step (one
     * launch per step on the sparse path). */
    { "max_seqs", RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_CAPACITY },
    P_INT("group"),
    P_STR("act"),
};
static const RadOperandSpec oAttnPagedGq[] = {
    { "q",           RAD_OPD_IN,  0 },
    { "kv_cache",    RAD_OPD_IN,  0 },
    { "block_table", RAD_OPD_IN,  0 },
    { "seqused",     RAD_OPD_IN,  0 },
    { "k_descale",   RAD_OPD_IN,  1 },
    { "v_descale",   RAD_OPD_IN,  1 },
    { "cu_seqlens",  RAD_OPD_IN,  1 },
    { "out",         RAD_OPD_OUT, 0 },
    { "gate",        RAD_OPD_IN,  0 },
    { "q8",          RAD_OPD_OUT, 0 },
    { "q8_scale",    RAD_OPD_OUT, 0 },
};

static const RadParamSpec pAttnDense[] = {
    P_INT("M"),
    P_INT("head_dim"),
    P_INT("gqa"),
    P_INT("causal"),
    P_STR("dtype"),
    P_F64O("scale"),
    /* Selects the query-block height; a host-side bound on the cu_seqlens segments. Absent, the
     * shim passes the total token count, a valid bound that picks the tall arm (speed only). */
    P_INTO("max_seqlen"),
    /* attn_paged's window over a segment's own positions (docs/OPS.md); 0 is none. */
    P_INTO("window"),
    /* cu_seqlens read in units that many rows apart, and blocks of `block` rows from a segment's
     * start that attend only within themselves (a chunked encoder's windows). */
    P_INTO("cu_div"),
    P_INTO("block"),
    /* Non-causal only: the window's right side, key j visible when j - i < window_right; 0 is
     * `window`'s (docs/OPS.md). */
    P_INTO("window_right"),
};
static const RadOperandSpec oAttnDense[] = {
    { "q",          RAD_OPD_IN,  0 },
    { "k",          RAD_OPD_IN,  0 },
    { "v",          RAD_OPD_IN,  0 },
    { "cu_seqlens", RAD_OPD_IN,  0 },
    { "out",        RAD_OPD_OUT, 0 },
};
static const RadParamSpec pAttnCross[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("gqa"), P_INT("causal"), P_STR("dtype"),
    P_F64O("scale"),
};
static const RadOperandSpec oAttnCross[] = {
    { "q",          RAD_OPD_IN,  0 },
    { "k",          RAD_OPD_IN,  0 },
    { "v",          RAD_OPD_IN,  0 },
    { "cu_seqlens", RAD_OPD_IN,  0 },
    { "cu_k",       RAD_OPD_IN,  0 },
    { "out",        RAD_OPD_OUT, 0 },
};
static const RadParamSpec pAttnNa3d[] = { P_INT("M"), P_INT("head_dim"), P_INT("kt"), P_INT("kh"),
                                          P_INT("kw"), P_STR("dtype"), P_F64O("scale") };
static const RadOperandSpec oAttnNa3d[] = { { "q", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 },
                                            { "v", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 },
                                            { "out", RAD_OPD_OUT, 0 } };

/* Not docs/OPS.md's `qk_norm_rope` (qkv in place, `theta`): this kernel splits a packed [q|gate]
 * projection per head, writes q, k and the gate to three destinations, and takes a precomputed
 * cos/sin table. Hence its own op name. */
static const RadParamSpec pQkNormRopeGate[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "rot",        RAD_P_INT, RAD_REQUIRED },   /* rotary_dim, even and <= head_dim */
    { "q_dtype",    RAD_P_STR, RAD_REQUIRED },
    P_F64O("eps"),
    P_F64O("wadd"),                              /* GemmaRMSNorm's `1 +`; 0 for a plain norm */
    { "cs_f32",     RAD_P_INT, RAD_OPTIONAL },   /* cos_sin table is f32 rather than bf16 */
    { "pos_i64",    RAD_P_INT, RAD_OPTIONAL },
};
/* The q half is optional (the k-only form): with `n_head` 0 there is no `q_gate`, `q_w` or
 * `q_out`, and every head is read from `k` at its row pitch and written contiguously to `k_out`.
 * The k side is the plain [tokens, heads, head_dim] reader; the q side only adds the split. */
static const RadOperandSpec oQkNormRopeGate[] = {
    { "q_gate",    RAD_OPD_IN,     1 },
    { "k",         RAD_OPD_IN,     0 },
    { "cos_sin",   RAD_OPD_IN,     0 },
    { "positions", RAD_OPD_IN,     0 },
    { "q_w",       RAD_OPD_WEIGHT, 1 },
    { "k_w",       RAD_OPD_WEIGHT, 0 },
    { "q_out",     RAD_OPD_OUT,    1 },
    { "k_out",     RAD_OPD_OUT,    0 },
    { "gate_out",  RAD_OPD_OUT,    1 },
};
/* The prologue with kv_store folded in: its parameters and kv_store's, its operands, then
 * kv_store's `v`, `slot_mapping` and `kv_cache`. kv_store's `k` is `k_out`. */
static const RadParamSpec pQkNormRopeGateKv[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "rot",        RAD_P_INT, RAD_REQUIRED },
    { "q_dtype",    RAD_P_STR, RAD_REQUIRED },
    P_F64O("eps"),
    P_F64O("wadd"),
    { "cs_f32",     RAD_P_INT, RAD_OPTIONAL },
    { "pos_i64",    RAD_P_INT, RAD_OPTIONAL },
    { "block_size", RAD_P_INT, RAD_REQUIRED },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
    P_F64O("k_scale"),
    P_F64O("v_scale"),
};
static const RadOperandSpec oQkNormRopeGateKv[] = {
    { "q_gate",       RAD_OPD_IN,     1 },
    { "k",            RAD_OPD_IN,     0 },
    { "cos_sin",      RAD_OPD_IN,     0 },
    { "positions",    RAD_OPD_IN,     0 },
    { "q_w",          RAD_OPD_WEIGHT, 1 },
    { "k_w",          RAD_OPD_WEIGHT, 0 },
    { "q_out",        RAD_OPD_OUT,    1 },
    { "k_out",        RAD_OPD_OUT,    0 },
    { "gate_out",     RAD_OPD_OUT,    1 },
    { "v",            RAD_OPD_IN,     0 },
    { "slot_mapping", RAD_OPD_IN,     0 },
    { "kv_cache",     RAD_OPD_INOUT,  0 },
};
/* Builds the cos/sin table the fused prologue reads, each position at its own row of a
 * [max_ctx, rot] plane. `rot`, `theta` and `scale` are spelled as `rope`'s, since both must compute
 * the same angles. */
static const RadParamSpec pRopeTable[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "rot",   RAD_P_INT, RAD_REQUIRED },
    P_F64("theta"),
    P_F64("scale"),
    { "dtype",        RAD_P_STR, RAD_REQUIRED },
    { "yarn_factor",  RAD_P_F64, RAD_OPTIONAL },
    { "yarn_low",     RAD_P_F64, RAD_OPTIONAL },
    { "yarn_high",    RAD_P_F64, RAD_OPTIONAL },
    { "yarn_magnitude", RAD_P_F64, RAD_OPTIONAL },
    { "head_dim",     RAD_P_INT, RAD_OPTIONAL },   /* needed only by the `head` ladder */
    { "ladder",       RAD_P_STR, RAD_OPTIONAL },   /* the base ladder's width: rotary|head */
};
static const RadOperandSpec oRopeTable[] = {
    { "positions", RAD_OPD_IN,  0 },
    { "cos_sin",   RAD_OPD_OUT, 0 },
};

/* -------------------------------------------------------------------------- gated delta net */
static const RadParamSpec pGdnConvPrep[] = {
    P_INT("M"),
    P_INT("head_k"),
    P_INT("head_v"),
    /* The gate cumsum's chunk is the one gdn_chunk_scan tiles on, which the kernel owns, so it
     * is derived; a second source could sum a decay over the wrong span. libr4d answers 64. */
    P_CHUNK("chunk"),
    P_INT("conv_width"),
    /* The l2 norm's epsilon, fixed at 1e-6f in the kernel (FLA's L2NORM_EPS). The shim accepts
     * only that value. */
    P_F64O("l2_eps"),
    P_F64O("softplus_thr"),
};
/* Inputs first, outputs last. `a` and `b_gate` are the gate and beta projections, passed
 * separately from x (the qkv slice); `cache_idx` is each sequence's conv-state slot and `has_init`
 * whether it holds a history. All four are optional in the schema (libref reads a and b_gate out
 * of the fused x row), but this shim requires the first three (RAD_E_INVAL naming them). */
static const RadOperandSpec oGdnConvPrep[] = {
    { "x",          RAD_OPD_IN,     0 },
    { "w",          RAD_OPD_WEIGHT, 0 },
    { "b",          RAD_OPD_WEIGHT, 1 },
    { "A_log",      RAD_OPD_WEIGHT, 0 },
    { "dt_bias",    RAD_OPD_WEIGHT, 0 },
    { "conv_state", RAD_OPD_INOUT,  0 },
    { "cu",         RAD_OPD_IN,     0 },
    { "a",          RAD_OPD_IN,     1 },
    { "b_gate",     RAD_OPD_IN,     1 },
    { "cache_idx",  RAD_OPD_IN,     1 },
    { "has_init",   RAD_OPD_IN,     1 },
    { "q",          RAD_OPD_OUT,    0 },
    { "k",          RAD_OPD_OUT,    0 },
    { "v",          RAD_OPD_OUT,    0 },
    { "g",          RAD_OPD_OUT,    0 },
    { "beta",       RAD_OPD_OUT,    0 },
};

static const RadParamSpec pGdnConvUpdate[] = {
    P_INT("q_len"),
    P_INT("head_k"),
    P_INT("head_v"),
    P_INT("conv_width"),
    P_INTO("max_query_len"),
};
static const RadOperandSpec oGdnConvUpdate[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "w",            RAD_OPD_WEIGHT, 0 },
    { "b",            RAD_OPD_WEIGHT, 1 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "state_idx",    RAD_OPD_IN,     0 },
    /* Required: speculative rollback is part of the schema (spec §10). The conv window has
     * absorbed the rejected tokens, and a rejection changes the read offset. A non-speculative
     * decode passes all ones. */
    { "num_accepted", RAD_OPD_IN,     0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "q",            RAD_OPD_OUT,    0 },
    { "k",            RAD_OPD_OUT,    0 },
    { "v",            RAD_OPD_OUT,    0 },
};

static const RadParamSpec pGdnKktSolve[] = {
    P_INT("M"),
    P_INT("head_k"),
    P_CHUNK("chunk"),
};
static const RadOperandSpec oGdnKktSolve[] = {
    { "k",    RAD_OPD_IN,  0 },
    { "beta", RAD_OPD_IN,  0 },
    { "g",    RAD_OPD_IN,  0 },
    { "cu",   RAD_OPD_IN,  0 },
    { "A",    RAD_OPD_OUT, 0 },
};

static const RadParamSpec pGdnChunkScan[] = {
    P_INT("M"),
    P_INT("head_k"),
    P_INT("head_v"),
    P_CHUNK("chunk"),
    P_F64O("scale"),
    /* An fp16 SSM state is faster than fp32 and frees KV capacity; a deployment declares which. */
    P_INTO("state_fp16"),
};
static const RadOperandSpec oGdnChunkScan[] = {
    { "q",    RAD_OPD_IN,  0 },
    { "k",    RAD_OPD_IN,  0 },
    { "v",    RAD_OPD_IN,  0 },
    { "A",    RAD_OPD_IN,  0 },
    { "g",    RAD_OPD_IN,  0 },
    { "beta", RAD_OPD_IN,  0 },
    { "h0",   RAD_OPD_IN,  1 },
    { "cu",   RAD_OPD_IN,  0 },
    { "o",    RAD_OPD_OUT, 0 },
    { "ht",   RAD_OPD_OUT, 1 },
    /* The state slot each batch row owns. Optional: without it row n owns slot n. */
    { "state_idx", RAD_OPD_IN, 1 },
};

/* libr4d's `norm_eps` and `norm_act` are `eps` and `act` here, as on the unfused
 * gdn_gated_rmsnorm whose arithmetic the fold reproduces. `act` is a name ("silu" or "sigmoid") so
 * graph dumps are readable; the shim maps it to the entry point's integer. */
static const RadParamSpec pGdnRecurrentUpdate[] = {
    P_INT("q_len"),
    P_INT("head_k"),
    P_INT("head_v"),
    /* Stated rather than inferred from the value tensor: with 48 value heads against 16 key
     * heads, a flattened view gives a plausible wrong count. */
    P_INTO("n_head_v"),
    P_F64O("scale"),
    P_F64O("eps"),
    /* The qk l2 norm's epsilon, fixed at 1e-6f in the kernel; the shim accepts only that value. */
    P_F64O("l2_eps"),
    P_F64O("softplus_thr"),
    P_STRO("act"),
    P_INTO("state_fp16"),
    /* The state cache's meaning; both forms produce the same `o` from different `state`.
     * "per_candidate" writes one full state per candidate token (1 + n_spec slots); "anchor" keeps
     * one committed state per sequence and replays the accepted prefix onto it from a scratch slot.
     * Optional because an out-of-tree caller sharing this kernel selects the form by environment.
     * When declared it names the shape the pool was sized for, and the entry point refuses the
     * other. */
    P_STRO("state_form"),
};
static const RadOperandSpec oGdnRecurrentUpdate[] = {
    { "q",            RAD_OPD_IN,     0 },
    { "k",            RAD_OPD_IN,     0 },
    { "v",            RAD_OPD_IN,     0 },
    { "a",            RAD_OPD_IN,     0 },
    { "b",            RAD_OPD_IN,     0 },
    { "A_log",        RAD_OPD_WEIGHT, 0 },
    { "dt_bias",      RAD_OPD_WEIGHT, 0 },
    { "state",        RAD_OPD_INOUT,  0 },
    { "state_idx",    RAD_OPD_IN,     0 },
    /* Required, as on gdn_conv_update: the state has absorbed the rejected tokens (spec §10). A
     * non-speculative decode passes all ones. */
    { "num_accepted", RAD_OPD_IN,     0 },
    /* Required, in the same slot as on gdn_conv_update: the kernel reads `cu[n]` and `cu[n+1]`
     * before any null test. Both decode kernels run off one cu per step. */
    { "cu",           RAD_OPD_IN,     0 },
    { "z",            RAD_OPD_IN,     1 },
    { "norm_w",       RAD_OPD_WEIGHT, 1 },
    { "o",            RAD_OPD_OUT,    0 },
    /* The out projection's quantiser, optional and folded in, saving a quant_act_fp8 launch per
     * layer. Possible because head_v equals the fp8 block (128): the scale plane is one f32 per
     * (token, value head), which one workgroup owns. Absent, the kernel writes bf16 only and the
     * caller quantises separately (the prefill path). */
    { "o_q",          RAD_OPD_OUT,    1 },
    { "o_scale",      RAD_OPD_OUT,    1 },
};

/* The decode convolution folded into the recurrent update. gdn_recurrent_update is the only
 * reader of gdn_conv_update's q, k and v at decode, so this op takes the convolution's inputs and
 * the recurrent update's minus those three; the parameters are the union of both. `z` and `norm_w`
 * are required: the fold runs one workgroup per (sequence, value head), the shape the fused row
 * norm already forces. */
static const RadParamSpec pGdnConvRecurrentUpdate[] = {
    P_INT("q_len"),
    P_INT("head_k"),
    P_INT("head_v"),
    P_INT("conv_width"),
    P_INTO("n_head_v"),
    P_F64O("scale"),
    P_F64O("eps"),
    P_F64O("l2_eps"),
    P_F64O("softplus_thr"),
    P_STRO("act"),
    P_INTO("state_fp16"),
    P_STRO("state_form"),
};
static const RadOperandSpec oGdnConvRecurrentUpdate[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "conv_w",       RAD_OPD_WEIGHT, 0 },
    { "conv_b",       RAD_OPD_WEIGHT, 1 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "conv_idx",     RAD_OPD_IN,     0 },
    { "a",            RAD_OPD_IN,     0 },
    { "b",            RAD_OPD_IN,     0 },
    { "A_log",        RAD_OPD_WEIGHT, 0 },
    { "dt_bias",      RAD_OPD_WEIGHT, 0 },
    { "state",        RAD_OPD_INOUT,  0 },
    { "state_idx",    RAD_OPD_IN,     0 },
    { "num_accepted", RAD_OPD_IN,     0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "z",            RAD_OPD_IN,     0 },
    { "norm_w",       RAD_OPD_WEIGHT, 0 },
    { "o",            RAD_OPD_OUT,    0 },
    { "o_q",          RAD_OPD_OUT,    1 },
    { "o_scale",      RAD_OPD_OUT,    1 },
};

/* The control vector's projection (r4d_hc_bf16.hip, with the other gated-residual kernels): a bf16
 * stream, one unit f32 direction for every stream, an int32 switch per row. */
static const RadParamSpec pHcProject[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("scale"),
                                           P_STR("dtype") };
static const RadOperandSpec oHcProject[] = {
    { "h", RAD_OPD_INOUT, 0 }, { "v", RAD_OPD_IN, 0 }, { "on", RAD_OPD_IN, 0 },
};
static const RadConstraint cHcProject[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};
/* The gated residual. `hc_enter` and `hc_write` share a parameter list; `hc_read` adds the two
 * switches its operand list depends on: `inject` 0 is the final mixer and `group` 0 a read whose
 * consumer wants bf16. Both default on, as every trunk connection uses both. */
static const RadParamSpec pHcPlain[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype"),
};
static const RadOperandSpec oHcEnter[] = {
    { "x", RAD_OPD_IN, 0 }, { "h", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pMhcPre[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("iterations"), P_F64("eps"),
    P_F64("hc_eps"), P_STR("dtype"),
};
static const RadOperandSpec oMhcPre[] = {
    {"h", RAD_OPD_IN, 0}, {"fn", RAD_OPD_WEIGHT, 0}, {"scale", RAD_OPD_WEIGHT, 0},
    {"base", RAD_OPD_WEIGHT, 0}, {"x", RAD_OPD_OUT, 0}, {"post", RAD_OPD_OUT, 0},
    {"comb", RAD_OPD_OUT, 0},
};
static const RadParamSpec pMhcPrePartial[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("chunk"), P_INT("first"), P_INT("chunks"),
    P_STR("dtype"),
};
static const RadOperandSpec oMhcPrePartial[] = {
    {"h", RAD_OPD_IN, 0}, {"fn", RAD_OPD_WEIGHT, 0}, {"part", RAD_OPD_OUT, 0},
};
static const RadParamSpec pMhcPreFinish[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("iterations"), P_F64("eps"),
    P_F64("hc_eps"), P_INT("chunk"), P_STR("dtype"),
};
static const RadOperandSpec oMhcPreFinish[] = {
    {"part", RAD_OPD_IN, 0}, {"h", RAD_OPD_IN, 0}, {"scale", RAD_OPD_WEIGHT, 0},
    {"base", RAD_OPD_WEIGHT, 0}, {"x", RAD_OPD_OUT, 0}, {"post", RAD_OPD_OUT, 0},
    {"comb", RAD_OPD_OUT, 0},
};
static const RadOperandSpec oMhcPost[] = {
    {"x", RAD_OPD_IN, 0}, {"h", RAD_OPD_IN, 0}, {"post", RAD_OPD_IN, 0},
    {"comb", RAD_OPD_IN, 0}, {"out", RAD_OPD_OUT, 0},
};
static const RadOperandSpec oMhcContract[] = {
    {"h", RAD_OPD_IN, 0}, {"x", RAD_OPD_OUT, 0},
};
static const RadParamSpec pKdaUpdate[] = {
    P_INT("M"), P_INT("heads"), P_INT("head_k"), P_INT("head_v"), P_F64("lower_bound"),
    P_F64("scale"), P_F64("l2_eps"), P_STR("mode"), P_STR("dtype"),
    P_STRO("state_form"),
};
static const RadOperandSpec oKdaUpdate[] = {
    {"q", RAD_OPD_IN, 0}, {"k", RAD_OPD_IN, 0}, {"v", RAD_OPD_IN, 0},
    {"a", RAD_OPD_IN, 0}, {"b", RAD_OPD_IN, 0}, {"A_log", RAD_OPD_WEIGHT, 0},
    {"dt_bias", RAD_OPD_WEIGHT, 0}, {"state", RAD_OPD_INOUT, 0}, {"state_idx", RAD_OPD_IN, 0},
    {"cu", RAD_OPD_IN, 0}, {"num_accepted", RAD_OPD_IN, 1}, {"has_init", RAD_OPD_IN, 1},
    {"o", RAD_OPD_OUT, 0},
};
static const RadOperandSpec oHcWrite[] = {
    { "y", RAD_OPD_IN, 0 }, { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 },
};
/* hc_write with the preceding all-reduce folded in: `y` is also the message, so it is inout and
 * holds the reduced sum as the standalone collective leaves it. */
static const RadParamSpec pArHcWrite[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("world_size"), P_STR("dtype"),
    P_INTO("wire"),   /* 0 = exact bf16, 6 = rotated 6-bit */
};
static const RadOperandSpec oArHcWrite[] = {
    { "y", RAD_OPD_INOUT, 0 }, { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 },
};
/* moe_gather's parameters and operands, then ar_hc_write's. The gather's output is the
 * reduction's message, one operand written by the gather and reduced in place. */
static const RadParamSpec pArGatherHcWrite[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("top_k"), P_INT("world_size"), P_STR("dtype"),
    P_STRO("act"), P_INTO("wire"),
};
static const RadOperandSpec oArGatherHcWrite[] = {
    { "y_expert", RAD_OPD_IN, 0 }, { "expert_w", RAD_OPD_IN, 0 }, { "sorted_tok", RAD_OPD_IN, 0 },
    { "shared", RAD_OPD_IN, 1 },   { "shared_gate", RAD_OPD_IN, 1 },
    { "y", RAD_OPD_INOUT, 0 },     { "inj", RAD_OPD_IN, 0 },       { "h", RAD_OPD_INOUT, 0 },
};

/* The MTP head's entry, for a model whose residual stream is `hc` wide. `ngroup` (hc rsqrts, one
 * per sub-stream, or one over the whole row) is not in the checkpoint, so it is a parameter; it
 * defaults to the grouped form every gated residual here uses. */
static const RadParamSpec pMtpEnter[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
    P_INTO("ngroup"), P_STRO("fc"),
};
static const RadOperandSpec oMtpEnter[] = {
    { "h", RAD_OPD_IN, 0 },  { "e", RAD_OPD_IN, 0 },
    { "w_h", RAD_OPD_WEIGHT, 0 }, { "w_e", RAD_OPD_WEIGHT, 0 },
    { "fc_h", RAD_OPD_WEIGHT, 0 }, { "fc_e", RAD_OPD_WEIGHT, 0 },
    { "x", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pHcRead[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"), P_F64("eps"), P_STR("dtype"),
    P_F64O("wadd"), P_INTO("inject"), P_INTO("group"), P_INTO("rotate"), P_STRO("mix"),
};
static const RadOperandSpec oHcRead[] = {
    { "h",        RAD_OPD_IN,     0 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "mix_down", RAD_OPD_WEIGHT, 0 },
    { "mix_up",   RAD_OPD_WEIGHT, 0 },
    { "inject_w", RAD_OPD_WEIGHT, 1 },
    { "x",        RAD_OPD_OUT,    0 },
    { "inj",      RAD_OPD_OUT,    1 },
    { "q",        RAD_OPD_OUT,    1 },
    { "scale",    RAD_OPD_OUT,    1 },
    { "rq",       RAD_OPD_OUT,    1 },
    { "rscale",   RAD_OPD_OUT,    1 },
};

static const RadParamSpec pGdnGatedRmsNorm[] = {
    P_INT("M"),
    P_INT("channels"),
    P_F64("eps"),
    P_STR("act"),      /* "silu" or "sigmoid"; the shim maps it to libr4d's 0 / 1 */
};
static const RadOperandSpec oGdnGatedRmsNorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "z", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "o", RAD_OPD_OUT,    0 },
};

/* Not in docs/OPS.md: the DFlash2 drafter's grouped dynamic depthwise convolution. It has no
 * unfused equivalent and no ref implementation, so the drafter has no fallback (spec §17). */
static const RadParamSpec pDflashConv[] = {
    { "T",           RAD_P_INT, RAD_REQUIRED },
    { "hidden_size", RAD_P_INT, RAD_REQUIRED },
    { "taps",        RAD_P_INT, RAD_REQUIRED },
    { "group_size",  RAD_P_INT, RAD_REQUIRED },
    { "NG",          RAD_P_INT, RAD_REQUIRED },
    { "block_size",  RAD_P_INT, RAD_REQUIRED },
    { "dtype",       RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oDflashConv[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "delta", RAD_OPD_IN,  0 },
    /* A weight: `base_kernel` is a checkpoint parameter, and the role gives it a prefetch
     * interval and a residency class. */
    { "base",  RAD_OPD_WEIGHT, 0 },
    { "out",   RAD_OPD_OUT, 0 },
};

/* DFlash2's candidate selector, also without an equivalent in the vocabulary. A block-diffusion
 * drafter computes every position of a block in one pass, so position l+1 cannot condition on the
 * token chosen at l; the selector restores that dependence with a low-rank bilinear score on the
 * edge between consecutive positions and a greedy walk over the scores.
 *
 * `M` is ranged and counts sequences; `steps`, the draft depth, is fixed by the checkpoint's
 * trained block length. No ref implementation, so no host fallback (spec §17). */
static const RadParamSpec pDflashSelect[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "steps",   RAD_P_INT, RAD_REQUIRED },
    { "top_k",   RAD_P_INT, RAD_REQUIRED },
    { "rank",    RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    /* The stride between consecutive sequences' anchors inside `anchor`, so the operand can be
     * the step's token_ids (`block` ids per sequence, anchor first). Absent means a packed [M]
     * array. */
    { "anchor_stride", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oDflashSelect[] = {
    { "cand",   RAD_OPD_IN,     0 },
    { "unary",  RAD_OPD_IN,     0 },
    /* Optional; absent means ones, and the edge score drops from trilinear to the bigram
     * `<pred[prev], succ[cand]>`, DSpark's Markov head (see r4d_dflash_select_bf16.hip). */
    { "hp",     RAD_OPD_IN,     1 },
    { "anchor", RAD_OPD_IN,     0 },
    { "pred",   RAD_OPD_WEIGHT, 0 },
    { "succ",   RAD_OPD_WEIGHT, 0 },
    { "tokens", RAD_OPD_OUT,    0 },
};

/* ------------------------------------------------------------------------------ collectives */
static const RadParamSpec pAllReduce[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    /* The message, in elements. Blocks declare it as a range (rows times width), so the selector
     * cuts bands at the rows' bounds and resolves each at its upper bound; init() sizes the peer
     * scratch for that bound and the launch reads the real count off the tensor. RAD_C_DIV is thus
     * checked at a band's upper bound, so every bound here is a multiple of 128 (the largest DIV);
     * otherwise some band would have no DIV row to serve it. */
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "exact",      RAD_P_INT, RAD_REQUIRED },
    /* The byte floor below which a lossy row serves the message exactly. Absent means the shim's
     * default. A parameter because the crossover (where rotate/pack/unpack stops costing more than
     * the bytes it saves) depends on the link's bandwidth, which the deployment knows. Exact rows
     * ignore it. */
    { "min_bytes",  RAD_P_INT, RAD_OPTIONAL },
    /* A request for one-shot (1) or two-shot (2). Declared because docs/OPS.md's schema is the
     * union of every plugin's and libref accepts it; no row here reads it, since the message size
     * picks the two-shot. */
    { "hops",       RAD_P_INT, RAD_OPTIONAL },
    /* 1 when issued on the caller's second lane, beside compute (rad_lane). Picks the rows with a
     * narrow launch (tArLane); the result is unchanged. */
    { "overlap",    RAD_P_INT, RAD_OPTIONAL },
};
/* docs/OPS.md has `x` -> `x` (inout). Every kernel here is safe in place: the thread that
 * overwrites a word of the input is the one that read it (r4d_ar_entry.hip). `y` is an optional
 * second destination; absent means in place. */
static const RadOperandSpec oAllReduce[] = {
    { "x", RAD_OPD_INOUT, 0 },
    { "y", RAD_OPD_OUT,   1 },
};

/* No `exact` and no `hops`: a gather has no lossy wire and no second topology at two ranks.
 * `numel` is this rank's contribution (the output holds world_size times it) and, as for the
 * all-reduce, the bound init sizes the peer scratch from. */
static const RadParamSpec pAllGather[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    /* One row of this rank's contribution, in elements. Absent or 0: the message is one row and
     * `y` the plain concatenation, rank r's slice at r * numel. Present: row-interleaved, x
     * [rows][row] and y [rows][world_size][row], so the vocab-parallel candidate merge reads each
     * position's gathered candidates at one row stride. */
    { "row",        RAD_P_INT, RAD_OPTIONAL },
};
/* Both required (the third field is `optional`): the output is world_size times the input, so
 * there is no in-place form. */
static const RadOperandSpec oAllGather[] = {
    { "x", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};

/* ------------------------------------------------------------------------------------- GEMM */
static const RadParamSpec pGemmNt[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGemmNt[] = {
    { "a", RAD_OPD_IN,     0 },
    { "b", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT,    0 },
};

/* `pb1` and `pb2` are ParoQuant's, read by the mxfp4a8 rows and the 4- and 5-bit asymmetric rows
 * (w5a16, w4a16_asym, w4a8i8_asym, w5a8i8): weight rows [0, pb1), [pb1, pb2) and [pb2, N) each
 * meet their own copy of the activation, side by side in a row of `a` ([M, parts * K]). 0 leaves a
 * bound out. Absent on other grids, where a fused projection has one activation. */
/* `add_rmsnorm` is `add` (x += a) followed by `rmsnorm` of the x it wrote (docs/OPS.md). */
static const RadParamSpec pAddRmsNorm[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
    P_F64O("mult"),
};
static const RadOperandSpec oAddRmsNorm[] = {
    { "x", RAD_OPD_INOUT,  0 },
    { "a", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT,    0 },
};
static const RadParamSpec pGemmNtQ[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "pb1",   RAD_P_INT, RAD_OPTIONAL },
    { "pb2",   RAD_P_INT, RAD_OPTIONAL },
    /* The width whose K split a row slice of a wider linear takes (docs/OPS.md); only the narrow
     * fp8a8 / i8a8 rows split K by N, and they read it. */
    { "split_n", RAD_P_INT, RAD_OPTIONAL },
};
/* docs/OPS.md's `gemm_nt_q` is `a`, `a_scale`, `b`w, `b_scale`w -> `y`. Two of libr4d's six
 * quantised GEMMs need a seventh operand. RadLayout can declare aux streams beside a weight, but
 * RadArgs cannot deliver them to a launch (an ABI gap), so both are operands:
 *
 *   `a_sum`  per-row per-group sums of the int8 activation codes, which the asymmetric 4-bit and
 *            the 2-bit grids subtract to remove the stored zero point. Produced by
 *            `quant_act_i8`'s third output, so activation-side.
 *   `b_ref`  MXFP4's per-output-row reference exponent, against which the E8M0 block scale is
 *            folded so the inner loop needs no rescale. A weight aux in principle.
 *
 * Both follow `y`, so OPS.md's list stays a positional prefix and ref is unaffected until it
 * implements a grid that needs them. */
static const RadOperandSpec oGemmNtQ[] = {
    { "a",       RAD_OPD_IN,     0 },
    { "a_scale", RAD_OPD_IN,     1 },   /* absent for the f16-activation grids (w4a16, w8a16) */
    { "b",       RAD_OPD_WEIGHT, 0 },
    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "y",       RAD_OPD_OUT,    0 },
    { "a_sum",   RAD_OPD_IN,     1 },
    { "b_ref",   RAD_OPD_WEIGHT, 1 },
};
/* docs/OPS.md's `gemm_nt_q_bias`, ref's spelling: gemm_nt_q's first four operands, a per-column
 * bias, then `y`. No parts and no aux operands. */
static const RadParamSpec pGemmNtQBias[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGemmNtQBias[] = {
    { "a",       RAD_OPD_IN,     0 },
    { "a_scale", RAD_OPD_IN,     0 },
    { "b",       RAD_OPD_WEIGHT, 0 },
    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "bias",    RAD_OPD_WEIGHT, 0 },
    { "y",       RAD_OPD_OUT,    0 },
};

/* docs/OPS.md's `dequant` is activation-shaped (`M`(range) `n` `group` `dtype`). libr4d's
 * dequant_w4_bf16 unpacks a weight from WMMA fragment order to row-major bf16, for the prefill
 * path where W4A8 gains nothing. It is its own op with N/K keys, so a ref implementation of
 * `dequant` is never mistaken for the fragment-order inverse.
 *
 * libref declares `dequant`, not `dequant_w4`, so this schema is libr4d's alone (docs/OPS.md,
 * "Quantisation and rotation"). */
static const RadParamSpec pDequantW4[] = {
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    /* 1 = the symmetric grid's two's-complement nibble, 0 = the asymmetric grid's unsigned nibble
     * minus a per-group zero. They differ by bit 3, and the wrong one decodes silently wrong. */
    { "twos",  RAD_P_INT, RAD_REQUIRED },
};
static const RadOperandSpec oDequantW4[] = {
    { "wq", RAD_OPD_WEIGHT, 0 },
    { "ws", RAD_OPD_WEIGHT, 0 },
    { "y",  RAD_OPD_OUT,    0 },
};

/* ---- the routed mixture of experts ------------------------------------------------------------
 *
 * `moe_gemm_q` is a separate op from docs/OPS.md's `moe_gemm`, as `gemm_nt_q` is from `gemm_nt`:
 * it adds an activation scale stream and a per-expert weight scale plane, and an operand count must
 * not depend on a dtype string. libref declares both.
 *
 * Weight operands are RAD_OPD_WTAB with one entry per expert, so the declaration divides evenly
 * between them (core/build/rad_builder.cpp). */
static const RadParamSpec pScaleRows[] = {
    P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype"),
};
static const RadOperandSpec oScaleRows[] = {
    { "x", RAD_OPD_IN,  0 },
    { "s", RAD_OPD_IN,  0 },
    /* Present = the fused gate-and-add form, `y = bf16(x * act(s)) + add`: a gated shared expert
     * scales its output, then adds the routed arm. On the MoE block that pair runs on one rank
     * while the other waits in the block's all-reduce, so fusing saves a dispatch per layer. */
    { "add", RAD_OPD_IN, 1 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pRouterTopk[] = {
    P_INT("M"), P_INT("n_expert"), P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
};
static const RadOperandSpec oRouterTopk[] = {
    { "logits",     RAD_OPD_IN,  0 },
    { "expert_ids", RAD_OPD_OUT, 0 },
    { "expert_w",   RAD_OPD_OUT, 0 },
};
static const RadParamSpec pMoeScatter[] = {
    P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
    /* This rank's first expert id, so expert parallelism is a parameter, not a second op. Absent
     * or 0 is the whole-model case. */
    P_INTO("expert_base"),
    /* Experts sorted last: a comma list of local indices, ascending. A layer that keeps a few
     * experts at higher precision serves them with a second grouped GEMM over a contiguous run of
     * offsets, so their rows go after all others and offsets and counts are in sort order. Absent
     * is none. */
    P_STRO("protect"),
};
/* router_topk and moe_scatter as one op: the union of their parameters, and their operands with
 * the shared `expert_ids` named once, as an output. */
static const RadParamSpec pRouterTopkScatter[] = {
    P_INT("M"),
    /* The router's count, every expert in the model: the router is replicated and each rank
     * picks from all of them. */
    P_INT("n_expert"),
    /* This rank's slice, which the sort runs over. Absent means the whole router (no expert
     * parallelism). Under expert parallelism the two counts differ. */
    P_INTO("n_local"),
    P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
    P_INTO("expert_base"),
    P_STRO("protect"),
};
static const RadOperandSpec oRouterTopkScatter[] = {
    { "logits",        RAD_OPD_IN,  0 },
    { "expert_ids",    RAD_OPD_OUT, 0 },
    { "expert_w",      RAD_OPD_OUT, 0 },
    { "sorted_tok",    RAD_OPD_OUT, 0 },
    { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count",  RAD_OPD_OUT, 0 },
};
/* The router's GEMM before that pair, with the shared expert's one-column gate beside it:
 * `gemm_nt` [M, K] x [n_expert, K]^T, `router_topk_scatter` over its output, and, with `gate_b` and
 * `gate` present, `gemm_nt` at N = 1 over the same input. N is implied (n_expert, 1). The logits
 * stay an output: the routing report and debug paths read them after the step. */
static const RadParamSpec pRouterGemmTopkScatter[] = {
    P_INT("M"), P_INT("K"), P_INT("n_expert"), P_INTO("n_local"),
    P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
    P_INTO("expert_base"),
    P_STRO("protect"),
};
static const RadOperandSpec oRouterGemmTopkScatter[] = {
    { "a",             RAD_OPD_IN,     0 },
    { "b",             RAD_OPD_WEIGHT, 0 },
    { "gate_b",        RAD_OPD_WEIGHT, 1 },
    { "logits",        RAD_OPD_OUT,    0 },
    { "expert_ids",    RAD_OPD_OUT,    0 },
    { "expert_w",      RAD_OPD_OUT,    0 },
    { "sorted_tok",    RAD_OPD_OUT,    0 },
    { "expert_offset", RAD_OPD_OUT,    0 },
    { "expert_count",  RAD_OPD_OUT,    0 },
    { "gate",          RAD_OPD_OUT,    1 },
};
static const RadOperandSpec oMoeScatter[] = {
    { "expert_ids",    RAD_OPD_IN,  0 },
    { "sorted_tok",    RAD_OPD_OUT, 0 },
    { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count",  RAD_OPD_OUT, 0 },
};
/* moe_scatter with each slot's row in `sorted_tok` beside it: moe_gather_inv reads that table
 * instead of rebuilding it. Ops of their own, since the shared vocabulary fixes moe_scatter's and
 * moe_gather's operands. */
static const RadOperandSpec oMoeScatterInv[] = {
    { "expert_ids",    RAD_OPD_IN,  0 },
    { "sorted_tok",    RAD_OPD_OUT, 0 },
    { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count",  RAD_OPD_OUT, 0 },
    { "inverse",       RAD_OPD_OUT, 0 },
};
/* `a_order` ("token" or "sorted") says which rows the activation has: one per token, gathered
 * through sorted_tok, or one per (token, slot) already in scatter order. A routed block's two
 * grouped GEMMs differ (gate_up reads the block input, down reads gate_up's output), and the row
 * count alone cannot tell the readings apart.
 *
 * `moe_gemm`, the unquantised op, has the vocabulary's schema, which libref fixes. It has one
 * weight table: an unscaled plane needs no parity split (see moe_gemm_q). */
static const RadParamSpec pMoeGemm[] = {
    P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"), P_INT("top_k"),
    P_STR("a_order"), P_STR("dtype"),
};
static const RadOperandSpec oMoeGemm[] = {
    { "a",             RAD_OPD_IN,   0 },
    { "w",             RAD_OPD_WTAB, 0 },
    { "sorted_tok",    RAD_OPD_IN,   0 },
    { "expert_offset", RAD_OPD_IN,   0 },
    { "y",             RAD_OPD_OUT,  0 },
};
/* The experts are two tables by parity: `w`/`w_scale` hold the even experts, `w_odd`/`w_odd_scale`
 * the odd ones, expert e at entry e / 2 of its class. A table's entries share one geometry, and a
 * rank holding an uneven slice of every expert's width (a five-block expert split across two ranks
 * as three 128-column blocks of one parity and two of the other) has two. `N`/`K` are the even
 * class's; `N_odd`/`K_odd` default to them.
 *
 * `parts` is how many stacked parts the weight rows and output columns hold (two for a fused
 * gate/up). The output is as wide as the wider class; a narrower class fills the leading columns
 * of each output part and writes zeros after them. */
static const RadParamSpec pMoeGemmQ[] = {
    P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"), P_INT("top_k"), P_INT("group"),
    P_STR("a_order"), P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"), P_INTO("parts"),
};
static const RadOperandSpec oMoeGemmQ[] = {
    { "a",             RAD_OPD_IN,   0 },
    { "a_scale",       RAD_OPD_IN,   1 },
    { "w",             RAD_OPD_WTAB, 0 },
    { "w_scale",       RAD_OPD_WTAB, 0 },
    { "sorted_tok",    RAD_OPD_IN,   0 },
    { "expert_offset", RAD_OPD_IN,   0 },
    { "w_odd",         RAD_OPD_WTAB, 0 },
    { "w_odd_scale",   RAD_OPD_WTAB, 0 },
    { "y",             RAD_OPD_OUT,  0 },
};
/* The same GEMM over four classes by expert number mod 4: `w`/`w_scale` hold class 0 at [N, K],
 * `w_q`/`w_q_scale` class q at [N_q, K_q], expert e at entry e / 4 of class e % 4. Four ranks
 * cannot share a five-block expert's extra block evenly over two classes, but can over four: rank
 * r takes two blocks of class r and one of each other. A separate op, because the declaration
 * divides a table op's weights over its table operands by count alone
 * (core/build/rad_builder.cpp). `y` follows the last class. */
static const RadParamSpec pMoeGemmQMod4[] = {
    P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"), P_INT("top_k"), P_INT("group"),
    P_STR("a_order"), P_STR("dtype"), P_INTO("N_1"), P_INTO("K_1"), P_INTO("N_2"), P_INTO("K_2"),
    P_INTO("N_3"), P_INTO("K_3"), P_INTO("parts"),
};
static const RadOperandSpec oMoeGemmQMod4[] = {
    { "a",             RAD_OPD_IN,   0 },
    { "a_scale",       RAD_OPD_IN,   1 },
    { "w",             RAD_OPD_WTAB, 0 },
    { "w_scale",       RAD_OPD_WTAB, 0 },
    { "sorted_tok",    RAD_OPD_IN,   0 },
    { "expert_offset", RAD_OPD_IN,   0 },
    { "w_1",           RAD_OPD_WTAB, 0 },
    { "w_1_scale",     RAD_OPD_WTAB, 0 },
    { "w_2",           RAD_OPD_WTAB, 0 },
    { "w_2_scale",     RAD_OPD_WTAB, 0 },
    { "w_3",           RAD_OPD_WTAB, 0 },
    { "w_3_scale",     RAD_OPD_WTAB, 0 },
    { "y",             RAD_OPD_OUT,  0 },
};
static const RadParamSpec pMoeGather[] = {
    P_INT("M"), P_INT("n"), P_INT("top_k"), P_STR("dtype"),
    /* The shared arm's gate, present exactly when `shared` and `shared_gate` are: "sigmoid" or
     * "none", applied to the scalar, as in `scale_rows` (the op folded in). */
    P_STRO("act"),
};
static const RadOperandSpec oMoeGather[] = {
    { "y_expert",    RAD_OPD_IN,  0 },
    { "expert_w",    RAD_OPD_IN,  0 },
    { "sorted_tok",  RAD_OPD_IN,  0 },
    /* Present = the shared arm folded in, `y = bf16(shared * act(gate)) + gathered`, replacing
     * the following `scale_rows`. Both terms are rounded to bf16 before the sum, as in the unfused
     * pair. The shared arm runs on one rank, so this removes a dispatch per layer the other rank
     * would wait out in the all-reduce, and the routed arm's `moe` buffer is never written. */
    { "shared",      RAD_OPD_IN,  1 },
    { "shared_gate", RAD_OPD_IN,  1 },
    { "y",           RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oMoeGatherInv[] = {
    { "y_expert",    RAD_OPD_IN,  0 },
    { "expert_w",    RAD_OPD_IN,  0 },
    { "inverse",     RAD_OPD_IN,  0 },
    { "shared",      RAD_OPD_IN,  1 },
    { "shared_gate", RAD_OPD_IN,  1 },
    { "y",           RAD_OPD_OUT, 0 },
};

/* `vocab_off` and `pairs` are the vocab-parallel form, both absent on a single-rank graph (which
 * keeps it byte-identical to the unsplit form). With them, `x` is one rank's slice, `idx` carries
 * global columns, and `pairs` is an [M, R, 2] plane of (int32 id, float value) (`buf_pairs_` in
 * core/sample/sampler.h), which an all_gather moves so the ranks agree on a global top-R. */
static const RadParamSpec pRowTopk[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "N",         RAD_P_INT, RAD_REQUIRED },
    { "R",         RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",     RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oRowTopk[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "idx",   RAD_OPD_OUT, 0 },
    { "val",   RAD_OPD_OUT, 0 },
    { "pairs", RAD_OPD_OUT, 1 },
};
/* The other half of that form: the R largest of the all-gathered per-rank candidates, in the
 * (idx, val) shape the single-rank top-R produces, so `dflash_select` sees the same input either
 * way. `R` is the output width; the input holds world_size * R. */
static const RadParamSpec pRowTopkMerge[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "R",          RAD_P_INT, RAD_REQUIRED },
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
};
static const RadParamSpec pLogitRerank[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "R",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "n_embd",    RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",     RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oLogitRerank[] = {
    { "x",      RAD_OPD_IN,     0 },
    { "head",   RAD_OPD_WEIGHT, 0 },
    { "idx_in", RAD_OPD_IN,     0 },
    { "idx",    RAD_OPD_OUT,    0 },
    { "val",    RAD_OPD_OUT,    0 },
    { "pairs",  RAD_OPD_OUT,    1 },
};

static const RadOperandSpec oRowTopkMerge[] = {
    { "gathered", RAD_OPD_IN,  0 },
    { "idx",      RAD_OPD_OUT, 0 },
    { "val",      RAD_OPD_OUT, 0 },
};

/* ------------------------------------------------------- quantisation, norms and the fusions */
static const RadParamSpec pQuantAct[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
/* docs/OPS.md gives `x` -> `q`, `scale`. The optional third output selects libr4d's `_asum` form,
 * which also emits the [K/group][Mpad] group-major row sums the asymmetric weight grids subtract.
 * Whether the sums are wanted is a property of the call, so it is an operand of one op (spec
 * §2.1). */
static const RadOperandSpec oQuantAct[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
    { "asum",  RAD_OPD_OUT, 1 },
};

/* `quant_act_fp8`, a new op whose schema this plugin fixes. Shaped on docs/OPS.md's
 * `quant_act_i8` (same params, same first three operands), so switching activation format changes
 * only the op name. No `asum`: fp8 grids are symmetric. libref implements it (ref_quant.cpp), which
 * gives the block-scaled fp8 family an oracle (spec §17). */
static const RadParamSpec pQuantActFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQuantActFp8[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
};

/* `givens_rotate` is ParoQuant's activation side for the 16-bit GEMMs. libref declared this
 * schema first (the loader refuses two that differ). The tables are the "givens" encoding's, as
 * weight operands; see libref/ref_quant.cpp for the contract and r4d_givens_rotate.hip. */
static const RadParamSpec pGivensRotate[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "n",      RAD_P_INT, RAD_REQUIRED },
    { "group",  RAD_P_INT, RAD_REQUIRED },
    { "dtype",  RAD_P_STR, RAD_REQUIRED },
    { "stages", RAD_P_INT, RAD_REQUIRED },
    { "parts",  RAD_P_INT, RAD_REQUIRED },
};
static const RadOperandSpec oGivensRotate[] = {
    { "x",      RAD_OPD_IN,     0 },
    { "pairs",  RAD_OPD_WEIGHT, 0 },
    { "angle",  RAD_OPD_WEIGHT, 0 },
    { "rscale", RAD_OPD_WEIGHT, 0 },
    { "y",      RAD_OPD_OUT,    0 },
};

/* `givens_dense_quant` is ParoQuant's activation quantiser, the rotation done by a dense per-group
 * matrix on the matrix cores (r4d_givens_dense.hip). The matrices are the weight `tdense` (the
 * givens encoding's "t.dense" table, composed by rad-convert), the channel scale `rscale`. `out`
 * is the quantiser: "i8" (scale amax / 127) or "fp8" (E4M3, amax / 448), matching q's dtype. */
static const RadParamSpec pGivensDense[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "n",      RAD_P_INT, RAD_REQUIRED },
    { "group",  RAD_P_INT, RAD_REQUIRED },
    { "dtype",  RAD_P_STR, RAD_REQUIRED },
    { "parts",  RAD_P_INT, RAD_REQUIRED },
    { "out",    RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGivensDense[] = {
    { "x",      RAD_OPD_IN,     0 },
    { "tdense", RAD_OPD_WEIGHT, 0 },
    { "rscale", RAD_OPD_WEIGHT, 0 },
    { "q",      RAD_OPD_OUT,    0 },
    { "scale",  RAD_OPD_OUT,    0 },
};

/* `gram_accum`, a new op whose schema this plugin fixes: GPTQ's calibration half, `h += x^T x`
 * over an E4M3 activation, accumulated across every launch of a corpus.
 *
 * `h` is inout: a caller zeroes it once and each launch adds its block. The operands are named `a`
 * and `a_scale` as in `gemm_nt_q`, whose activation this characterises. `a_scale` is optional:
 * dtype "bf16" reads a plain activation with no scale (a model served at bf16). */
static const RadParamSpec pGramAccum[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGramAccum[] = {
    { "a",       RAD_OPD_IN,    0 },
    { "a_scale", RAD_OPD_IN,    1 },
    { "h",       RAD_OPD_INOUT, 0 },
};

/* `qsa_block_key`, a new op whose schema this plugin fixes: the QSA indexer's compressed key, one
 * vector per `ratio` tokens of key history, which the indexer scores against.
 *
 * `M` counts blocks and `k` is `ratio` times taller; unlike other ops, there is one output row per
 * `ratio` input rows. `pos0` is a parameter because blocks tile the token stream (block i starts at
 * `pos0 + i * ratio`); the optional `pos` operand serves non-contiguous blocks. */
static const RadParamSpec pQsaBlockKey[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    { "ratio",      RAD_P_INT, RAD_REQUIRED },
    { "rotary_dim", RAD_P_INT, RAD_REQUIRED },
    { "pos0",       RAD_P_INT, RAD_OPTIONAL },   /* the tiling; a `pos` operand overrides it */
    /* Equal to `ratio`; it sizes a KV group. `Builder::find_kv_consumer` binds a group to the
     * first op declared after it with the key `block_size`, and the block-key store's page is the
     * compress block. The shim refuses a mismatch. */
    { "block_size", RAD_P_INT, RAD_OPTIONAL },
    P_F64("eps"),
    P_F64("theta"),
    P_F64O("scale"),
    P_F64O("wadd"),
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    /* The layer's rotary when it has several position components: `mode` mrope or imrope and its
     * `sections`, against a [3, M] component-major `pos` list. Absent is NeoX. */
    { "mode",       RAD_P_STR, RAD_OPTIONAL },
    { "sections",   RAD_P_STR, RAD_OPTIONAL },
    { "yarn_factor", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_low", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_high", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_magnitude", RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oQsaBlockKey[] = {
    { "k",    RAD_OPD_IN,     0 },
    { "w",    RAD_OPD_WEIGHT, 0 },
    /* Optional; it changes where the result goes. The block-key store is the paged KV group of
     * docs/QSA.md, so a sequence's blocks live on arbitrary pages. Absent, the write is contiguous
     * (own scratch, tests). Present, `M` comes from this operand, since `out` is as tall as the
     * pool. */
    { "page", RAD_OPD_IN,     1 },
    /* Optional; turns the launch from a range into a list. `pos0 + b*ratio` serves a prefill
     * chunk, whose completing blocks are one sequence's contiguous run. At decode the completing
     * blocks belong to different sequences, so one launch per layer per step needs this list. */
    { "pos",  RAD_OPD_IN,     1 },
    { "out",  RAD_OPD_OUT,    0 },
};

/* `qsa_work`, a new op whose schema this plugin fixes: the gather that turns a batch into the
 * flat block list `qsa_block_key` reads.
 *
 * `M` counts tokens and `work` blocks (one per `ratio` tokens plus at most one straddling block
 * per sequence). `work` is a host bound on how many blocks can complete; the actual count is
 * device data, and slots past it get `page = -1`, which `qsa_block_key` skips. */
static const RadParamSpec pQsaWork[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "ratio", RAD_P_INT, RAD_REQUIRED },
    { "ring",  RAD_P_INT, RAD_REQUIRED },
    { "seqs",  RAD_P_INT, RAD_REQUIRED },
    /* A capacity: the launch is as wide as `page` (r4d_qsa_work_bf16.hip). */
    { "work",  RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQsaWork[] = {
    { "k",     RAD_OPD_IN,  0 },
    { "pos",   RAD_OPD_IN,  0 },
    { "cu",    RAD_OPD_IN,  0 },
    /* The tail is the state group's cache for this layer; `sidx` names the sequence's slot (the
     * batch row is step-local, the sequence is not). */
    { "state", RAD_OPD_IN,  0 },
    { "sidx",  RAD_OPD_IN,  0 },
    { "bt",    RAD_OPD_IN,  0 },
    /* Optional: the step's rotary components, [3, T] (RadBatch::rope_pos). Present, `bpos` holds
     * each block's first-token components as [3, work], which an M-RoPE layer's block key rotates
     * at, instead of its index. */
    { "rope",  RAD_OPD_IN,  1 },
    { "stage", RAD_OPD_OUT, 0 },
    { "page",  RAD_OPD_OUT, 0 },
    { "bpos",  RAD_OPD_OUT, 0 },
    /* Complete blocks per sequence, needed by the scorer and the selection and computed here on
     * the way, which saves a launch per layer per step. */
    { "nc",    RAD_OPD_OUT, 0 },
};

/* `qsa_tail_store`, a new op whose schema this plugin fixes: it keeps the raw index keys of
 * tokens that have not completed a block, so a later step can finish it.
 *
 * `cu` is the only batch operand that maps a token to its sequence (`slot_mapping` is a flat KV
 * slot), and it supplies the run-time sequence count; the declared height is max_seqs. `cu` gives
 * the step-local row, so the keys live in a per-sequence linear KV group: `state` is its cache for
 * this layer and `sidx` the sequence's slot. */
static const RadParamSpec pQsaTailStore[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "ratio", RAD_P_INT, RAD_REQUIRED },
    { "ring",  RAD_P_INT, RAD_REQUIRED },
    { "seqs",  RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQsaTailStore[] = {
    { "k",     RAD_OPD_IN,    0 },
    { "pos",   RAD_OPD_IN,    0 },
    { "cu",    RAD_OPD_IN,    0 },
    { "state", RAD_OPD_INOUT, 0 },
    { "sidx",  RAD_OPD_IN,    0 },
};

/* `qsa_select`, a new op whose schema this plugin fixes: it turns the scores into the page-id
 * table the attention reads.
 *
 * `nc`, the complete-block count, is separate from the table width: `bt[nc]`, when not -1, is the
 * partial page (tokens after the last complete block, where the query itself lives), which the
 * reference always attends. It is appended after the selection and never competes in it, so `sel`
 * is `topk + 1` wide. */
static const RadParamSpec pQsaSelect[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "blocks", RAD_P_INT, RAD_REQUIRED },
    { "topk",   RAD_P_INT, RAD_REQUIRED },
    { "ratio",  RAD_P_INT, RAD_REQUIRED },
    { "seqs",   RAD_P_INT, RAD_REQUIRED },
    /* The ranking: 0 (absent) ranks max(score, 0) with scores at or below -1e30 dead, the QSA
     * indexer's scores being sums of relus; 1 ranks signed finite scores, nonfinite ones dead, for
     * indexers whose head weights can be negative. */
    { "signed", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oQsaSelect[] = {
    { "score",   RAD_OPD_IN,  0 },
    { "bt",      RAD_OPD_IN,  0 },
    { "nc",      RAD_OPD_IN,  0 },
    { "pos",     RAD_OPD_IN,  0 },
    /* `M` counts queries while `bt`/`nc` are per sequence; `cu` joins them. At prefill a chunk's
     * queries share one history and each selects its own set (docs/QSA.md, "Per-query selection"). */
    { "cu",      RAD_OPD_IN,  0 },
    { "sel",     RAD_OPD_OUT, 0 },
    { "nsel",    RAD_OPD_OUT, 0 },
    /* With `seqused`, QSA needs no new attention kernel: `attn_paged` takes its block table and
     * per-sequence length as ordinary operands, so the sparse attention is the same op issued with
     * `sel` and this. Emitted here because it is one store of a value this kernel has. */
    { "seqused", RAD_OPD_OUT, 0 },
    /* The same entries as block indices, in the same order. A page id does not sort by block -- a
     * dead block drops out of the table and the prefix cache re-orders the rest -- so a table
     * shared by a tile of query rows, the rows' union in block order, is built from this. Additive:
     * `sel` is unchanged and a caller may leave it out. */
    { "selb",    RAD_OPD_OUT, 1 },
};

/* `qsa_score`, a new op whose schema this plugin fixes: the indexer's score of each compressed
 * block of key history for one query.
 *
 * `bt` holds page ids, not block indices. The attention KV group is paged at the compress ratio,
 * so a block is a page, the sequence's block table is the list scored, and the top-k over the
 * output is the block table the attention reads (docs/QSA.md). */
static const RadParamSpec pQsaScore[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "n",      RAD_P_INT, RAD_REQUIRED },
    { "heads",  RAD_P_INT, RAD_REQUIRED },
    { "blocks", RAD_P_INT, RAD_REQUIRED },
    { "seqs",   RAD_P_INT, RAD_REQUIRED },
    { "dtype",  RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQsaScore[] = {
    { "q",     RAD_OPD_IN,  0 },
    { "bk",    RAD_OPD_IN,  0 },
    { "bt",    RAD_OPD_IN,  0 },
    /* As in oQsaSelect: `M` counts queries and the table is the sequence's. */
    { "cu",    RAD_OPD_IN,  0 },
    /* The block loop's bound, required. `blocks` is a declared width sized for max_ctx (65536
     * padding entries per query on a short prompt); `nc` is the sequence's live count, an upper
     * bound for every query in the chunk, which `qsa_select` narrows per query. */
    { "nc",    RAD_OPD_IN,  0 },
    { "score", RAD_OPD_OUT, 0 },
};

/* `rmsnorm_quant_fp8` and `gated_quant_fp8`, new ops whose schemas this plugin fixes. Each is its
 * producer's operand list with the quantiser's two outputs added and the producer's bf16 output
 * made optional, so an architecture that fuses changes only the op name and adds `q` and `scale`.
 *
 * The residual is second, as in `rmsnorm_had_quant_i8`, so the two differ only by the rotation. It
 * folds `add` + `rmsnorm` into one launch: a block adds the previous block's delta into the
 * residual stream at the front of its own norm. Unlike the int8 sibling, it rounds the sum to bf16
 * and stores it before normalising, as the replaced pair does, which keeps the output
 * byte-identical. */
static const RadParamSpec pRmsNormQuantFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
};
static const RadOperandSpec oRmsNormQuantFp8[] = {
    { "x",        RAD_OPD_IN,     0 },
    { "residual", RAD_OPD_INOUT,  1 },   /* present = the fused_add_rms_norm form */
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },   /* layers with a bf16 consumer of the post-norm tensor */
};

/* The same op with the all-reduce in front, like `ar_ln_had_quant_i8` without the rotation: same
 * operands, plus `world_size` and `wire` (0 exact, 6 the rotated 6-bit payload). */
static const RadParamSpec pArRmsNormQuantFp8[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "group",      RAD_P_INT, RAD_REQUIRED },
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
    { "wire",       RAD_P_INT, RAD_OPTIONAL },   /* 0 = exact bf16, 6 = rotated 6-bit */
};
static const RadOperandSpec oArRmsNormQuantFp8[] = {
    { "x",        RAD_OPD_INOUT,  0 },   /* the message and the input; receives the reduced sum */
    { "residual", RAD_OPD_INOUT,  1 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
};

/* ar_rmsnorm_post_quant_fp8: the same with a sandwich post-norm of the reduced sum before the add
 * (Gemma 3's junction, arch/gemma3), exact wire only. Its own op: an optional seventh operand on
 * ar_rmsnorm_quant_fp8 would break every plugin that issues that op's six. */
static const RadOperandSpec oArRmsNormPostQuantFp8[] = {
    { "x",        RAD_OPD_INOUT,  0 },
    { "residual", RAD_OPD_INOUT,  0 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
    { "w_post",   RAD_OPD_WEIGHT, 0 },
};

/* `rmsnorm_quant_i8` is the norm with r4d_gemm_w4a8_nt_m64's activation producer on its pass:
 * int8 codes in the GEMM's fragment order and one f32 scale per row (quant_act_i8's grid, not
 * quant_act_i8g's per-group one nor the rotated rmsnorm_had_quant_i8's). Its first six operands are
 * rmsnorm_quant_fp8's. Two optional weights make it a sandwich-norm model's whole residual
 * junction: `w_post` normalises the branch output before the add, `scalar` multiplies the sum
 * after it (Gemma's per-layer scalar). Every stage rounds to bf16 where the replaced chain stores
 * bf16, so the op is byte-identical to it. */
static const RadParamSpec pRmsNormQuantI8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
};
static const RadOperandSpec oRmsNormQuantI8[] = {
    { "x",        RAD_OPD_IN,     0 },
    { "residual", RAD_OPD_INOUT,  1 },   /* present = the norm is of residual + (post-normed) x */
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
    { "w_post",   RAD_OPD_WEIGHT, 1 },   /* rmsnorm(x; w_post) before the add; needs the residual */
    { "scalar",   RAD_OPD_WEIGHT, 1 },   /* [1], multiplies the sum; needs the residual */
};

/* `qkv_norm_rope` is `qk_norm_rope` over separate q, k and v projections: the per-head q and k
 * norms, the weightless value norm and the rotary of q and k in one dispatch (Gemma 4's attention
 * prologue). `v_from_k` 1 is its global class (k == v): v is the value norm of the raw k, written
 * before the k norm rewrites it. Rotary parameters are `rope`'s (`ladder` included), norm
 * parameters `rmsnorm`'s; byte-identical to rmsnorm_p + rmsnorm + rmsnorm + rope + rope. */
static const RadParamSpec pQkvNormRope[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    P_F64("theta"),
    P_F64("eps"),
    P_F64O("scale"),
    { "mode",       RAD_P_STR, RAD_OPTIONAL },
    { "rotary_dim", RAD_P_INT, RAD_OPTIONAL },
    { "ladder",     RAD_P_STR, RAD_OPTIONAL },
    P_F64O("wadd"),
    { "v_from_k",   RAD_P_INT, RAD_OPTIONAL },
    /* 0 leaves v unnormed (Gemma 3; a Qwen3-shaped layer); absent, 1 */
    { "v_norm",     RAD_P_INT, RAD_OPTIONAL },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQkvNormRope[] = {
    { "q",         RAD_OPD_INOUT,  0 },
    { "k",         RAD_OPD_INOUT,  0 },
    { "v",         RAD_OPD_INOUT,  0 },   /* written from the raw k under v_from_k */
    { "positions", RAD_OPD_IN,     0 },
    { "q_w",       RAD_OPD_WEIGHT, 0 },
    { "k_w",       RAD_OPD_WEIGHT, 0 },
};
/* `qkv_norm_rope_kv_store` is qkv_norm_rope followed by kv_store in one launch: the prologue's
 * operands, then kv_store's slot mapping and paged cache; every parameter is one of the two ops'.
 * Byte-identical to the pair (docs/OPS.md). */
static const RadParamSpec pQkvNormRopeKv[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    P_F64("theta"),
    P_F64("eps"),
    P_F64O("scale"),
    { "mode",       RAD_P_STR, RAD_OPTIONAL },
    { "rotary_dim", RAD_P_INT, RAD_OPTIONAL },
    { "ladder",     RAD_P_STR, RAD_OPTIONAL },
    P_F64O("wadd"),
    { "v_from_k",   RAD_P_INT, RAD_OPTIONAL },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "v_norm",     RAD_P_INT, RAD_OPTIONAL },
    { "block_size", RAD_P_INT, RAD_REQUIRED },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
    P_F64O("k_scale"),
    P_F64O("v_scale"),
};
static const RadOperandSpec oQkvNormRopeKv[] = {
    { "q",            RAD_OPD_INOUT,  0 },
    { "k",            RAD_OPD_INOUT,  0 },
    { "v",            RAD_OPD_INOUT,  0 },
    { "positions",    RAD_OPD_IN,     0 },
    { "q_w",          RAD_OPD_WEIGHT, 0 },
    { "k_w",          RAD_OPD_WEIGHT, 0 },
    { "slot_mapping", RAD_OPD_IN,     0 },
    { "kv_cache",     RAD_OPD_INOUT,  0 },
};

/* `gelu_mul_quant_i8` is `gelu_mul` with quant_act_i8's quantiser on its pass: the down projection
 * input of a GELU-gated feed-forward on the w4a8 grid, byte-identical to the pair. */
static const RadParamSpec pGeluMulQuantI8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGeluMulQuantI8[] = {
    { "gate_up",  RAD_OPD_IN,  0 },
    { "q",        RAD_OPD_OUT, 0 },
    { "scale",    RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};

/* `ar_rmsnorm_quant_i8` is rmsnorm_quant_i8 with the two-rank exact all-reduce of `x` in front:
 * same operands with `x` inout (it receives the reduced sum), plus `world_size`. */
static const RadParamSpec pArRmsNormQuantI8[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
};
static const RadOperandSpec oArRmsNormQuantI8[] = {
    { "x",        RAD_OPD_INOUT,  0 },
    { "residual", RAD_OPD_INOUT,  1 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
    { "w_post",   RAD_OPD_WEIGHT, 1 },
    { "scalar",   RAD_OPD_WEIGHT, 1 },
};

static const RadParamSpec pGatedQuantFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_STRO("act"),
};
static const RadOperandSpec oGatedQuantFp8[] = {
    { "gate_up",  RAD_OPD_IN,  0 },
    { "q",        RAD_OPD_OUT, 0 },
    { "scale",    RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};
/* The gate_up GEMM with gated_quant_fp8's pass in its epilogue. Its own schema, not a flag on
 * `gemm_nt_q`, because it writes the quantiser's outputs instead of a `y`. `N` is the weight's
 * rows, 2 * the fused width (gate then up), the same plane the unfused GEMM reads. `act` is
 * required: the fold is compiled for silu only. */
static const RadParamSpec pGemmNtQGated[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "act",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGemmNtQGated[] = {
    { "a",        RAD_OPD_IN,     0 },
    { "a_scale",  RAD_OPD_IN,     0 },
    { "b",        RAD_OPD_WEIGHT, 0 },
    { "b_scale",  RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
};

/* The gate in a separate operand, as in a gated attention: the gate is a column slice of the
 * [q|gate] projection and `x` the attention output, two buffers with different pitches. A row is a
 * (token, head) pair, so both sources are a fixed stride apart and the scale plane lands where the
 * out projection reads it. */
static const RadParamSpec pGateQuantFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_STRO("act"),
};
static const RadOperandSpec oGateQuantFp8[] = {
    { "gate",     RAD_OPD_IN,  0 },
    { "x",        RAD_OPD_IN,  0 },
    { "q",        RAD_OPD_OUT, 0 },
    { "scale",    RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};

/* `sample_chain`, a new op, distinct from docs/OPS.md's eight-op sampler chain (`sample_penalties`,
 * `sample_temp`, `sample_topk`, `sample_topp`, `sample_minp`, `sample_mask`, `sample_pick`,
 * `sample_argmax`), which would be eight launches over a 248K-entry row per position. libr4d's is
 * one kernel walking the vocabulary several times inside one workgroup, L2-resident after the first
 * pass. The core does not fuse ("fusion is a selection, not a compiler pass", spec §2.3), so the
 * plugin declares the fused op.
 *
 * top_k, top_p, min_p and temperature change per request, so they are launch-time parameters,
 * never constraints (a constraint would rebuild the bucket table on every setting change). The
 * float ones are RAD_P_F64 per docs/OPS.md.
 *
 * `query` is the drafted token per position and `u` two uniforms per position. The three outputs
 * are a draw, a draw with `query` excluded and renormalised, and `query`'s filtered probability:
 * everything a rejection-sampling verify needs in one pass (spec §10).
 *
 * libref does not implement it; its oracle is llama.cpp's host sampler (spec §13). */
static const RadParamSpec pSampleChain[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },   /* positions in the verify block */
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "temp",    RAD_P_F64, RAD_OPTIONAL },
    { "p",       RAD_P_F64, RAD_OPTIONAL },   /* top_p */
    { "min_p",   RAD_P_F64, RAD_OPTIONAL },
    { "k",       RAD_P_INT, RAD_OPTIONAL },   /* top_k; 0 disables */
};
static const RadOperandSpec oSampleChain[] = {
    { "logits", RAD_OPD_IN,  0 },
    { "query",  RAD_OPD_IN,  1 },   /* the drafted token per position, or absent */
    { "u",      RAD_OPD_IN,  0 },   /* two uniforms per position */
    { "draw",   RAD_OPD_OUT, 0 },   /* [M] x {u32 tok, u32 tok_excl, f32 p_query} */
};

static const RadParamSpec pHadQuantAct[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },   /* libr4d `had`: the rotation width */
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oHadQuantAct[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
};

static const RadParamSpec pRmsNormHadQuant[] = {
    P_INT("M"),
    P_INT("n"),
    P_F64("eps"),
    P_INT("group"),
    P_STR("dtype"),
    /* GemmaRMSNorm (Qwen3.5, Qwen3-Next, Gemma) stores a zero-centred gamma and uses `1 + w`.
     * Forming that in bf16 costs 0.3% per channel, so libr4d adds it in fp32 after the load. 0 is
     * a plain RMS norm. */
    P_F64O("wadd"),
};
static const RadOperandSpec oRmsNormHadQuant[] = {
    { "x",        RAD_OPD_IN,     0 },
    { "residual", RAD_OPD_INOUT,  1 },   /* present = the fused_add_rms_norm form */
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },   /* layers with a bf16 consumer of the post-norm tensor */
};

/* `act` names the gate: "silu" for down_proj's input, "sigmoid" for a gated-attention o_proj's.
 * `mode` is libr4d's optional integer spelling of the same choice; `act` wins if both are given. */
static const RadParamSpec pGatedHadQuant[] = {
    P_INT("M"),
    P_INT("n"),
    P_INT("group"),
    P_STR("dtype"),
    P_STRO("act"),
    P_INTO("mode"),                         /* 0 = silu(a)*b, 1 = sigmoid(a)*b */
};
static const RadOperandSpec oGatedHadQuant[] = {
    { "a",     RAD_OPD_IN,  0 },
    { "b",     RAD_OPD_IN,  1 },   /* absent = vLLM's packed gate_up layout, b = a + n */
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
};

/* Not in docs/OPS.md. Like `rmsnorm_had_quant_i8`, a fusion with no unfused equivalent: the GDN
 * gated RMS norm is per head while the rotation and the int8 scale are per token. */
static const RadParamSpec pGdnGatedNormHadQuant[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED },
    { "n",        RAD_P_INT, RAD_REQUIRED },
    { "head_dim", RAD_P_INT, RAD_REQUIRED },
    { "group",    RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "dtype",    RAD_P_STR, RAD_REQUIRED },
    /* Z may be the interleaved qkvz projection rather than a compacted gate: element i of the
     * flattened row sits at (i / zblk) * zgrp + (i % zblk). 0 means a contiguous gate. */
    { "zgrp",     RAD_P_INT, RAD_OPTIONAL },
    { "zblk",     RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oGdnGatedNormHadQuant[] = {
    { "x",     RAD_OPD_IN,     0 },
    { "z",     RAD_OPD_IN,     0 },
    { "w",     RAD_OPD_WEIGHT, 0 },
    { "q",     RAD_OPD_OUT,    0 },
    { "scale", RAD_OPD_OUT,    0 },
};

/* Not in docs/OPS.md; spec §2.3's example of a fused op a plugin asks for first. Every
 * row-parallel linear's output is all-reduced and then normalised; fused, the all-reduce's bf16
 * output never exists. The unfused fallback is `all_reduce` then `rmsnorm_had_quant_i8`. */
static const RadParamSpec pArLnHadQuant[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "group",      RAD_P_INT, RAD_REQUIRED },
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
    { "wire",       RAD_P_INT, RAD_OPTIONAL },   /* 0 = exact bf16, 6 = rotated 6-bit */
};
static const RadOperandSpec oArLnHadQuant[] = {
    { "x",        RAD_OPD_IN,     0 },
    { "residual", RAD_OPD_INOUT,  1 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
};

/* ---------------------------------------------- the small model ops and the sampler chain -----
 *
 * Transcribed from libref/ref_registry.cpp field for field. The first plugin in hierarchy order to
 * declare an op fixes its schema, and a disagreeing later plugin is refused at load
 * (core/plugin/loader.cpp), since arguments are positional. libr4d loads first, so a change here
 * belongs in both files.
 *
 * The constraint rows for these ops state only what the kernels require (bf16 operands, and a
 * block size for kv_store); the shims still check every stride, pitch and extent per call. */
static const RadParamSpec pRmsNorm[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
    P_F64O("mult"),          /* scales the row before its one rounding */
};
/* `w` optional: absent, the gain is one. */
static const RadOperandSpec oRmsNorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 1 },
    { "y", RAD_OPD_OUT,    0 },
};

static const RadParamSpec pMnDtype[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadParamSpec pSwigluClamped[] = { P_INT("M"), P_INT("n"), P_F64("limit"),
                                               P_STR("dtype") };
static const RadParamSpec pSwigluClampedQuant[] = { P_INT("M"), P_INT("n"), P_F64("limit"),
                                                    P_INT("group"), P_STR("dtype") };
static const RadParamSpec pDsaPool[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_STR("dtype"),
                                         P_INTO("block_size") };
static const RadParamSpec pMlaAttn[] = { P_INT("M"),      P_INT("latent_dim"), P_INT("heads"),
                                         P_INT("blocks"), P_INT("block_size"), P_F64("scale"),
                                         P_STR("dtype") };
static const RadParamSpec pMlaProject[] = { P_INT("M"),          P_INT("heads"), P_INT("head_dim"),
                                            P_INT("latent_dim"), P_STR("mode"),  P_STR("dtype") };
static const RadOperandSpec oMlaAttn[] = { { "q", RAD_OPD_IN, 0 },
                                           { "kv", RAD_OPD_IN, 0 },
                                           { "bt", RAD_OPD_IN, 0 },
                                           { "length", RAD_OPD_IN, 0 },
                                           { "out", RAD_OPD_OUT, 0 } };
static const RadParamSpec pDsaScore[] = { P_INT("M"),      P_INT("n"),    P_INT("heads"),
                                          P_INT("blocks"), P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oDsaScore[] = {
    { "q", RAD_OPD_IN, 0 },  { "q_scale", RAD_OPD_IN, 0 }, { "gate", RAD_OPD_IN, 0 },
    { "k", RAD_OPD_IN, 0 },  { "k_scale", RAD_OPD_IN, 1 }, { "bt", RAD_OPD_IN, 0 },
    { "cu", RAD_OPD_IN, 0 }, { "nc", RAD_OPD_IN, 0 },      { "score", RAD_OPD_OUT, 0 }
};
static const RadOperandSpec oDsaPool[] = {
    { "kg", RAD_OPD_IN, 0 },     { "ape", RAD_OPD_WEIGHT, 0 }, { "q", RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 }, { "page", RAD_OPD_IN, 1 },    { "cache", RAD_OPD_INOUT, 1 }
};
static const RadParamSpec pRouterSigmoid[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
    P_INT("norm"), P_F64("route_scale"), P_STR("dtype") };
static const RadOperandSpec oRouterSigmoid[] = {
    { "logits", RAD_OPD_IN, 0 }, { "bias", RAD_OPD_WEIGHT, 0 },
    { "ids", RAD_OPD_OUT, 0 }, { "weights", RAD_OPD_OUT, 0 }
};
/* router_sigmoid_topk and moe_scatter as one op, with `ids` named once, and the `fill` of an
 * ungated shared arm's gate with 1 as an optional output. */
static const RadParamSpec pRouterSigmoidScatter[] = { P_INT("M"), P_INT("n_expert"),
    P_INTO("n_local"), P_INT("top_k"), P_INT("norm"), P_F64("route_scale"), P_STR("dtype"),
    P_INTO("expert_base"), P_STRO("protect") };
static const RadOperandSpec oRouterSigmoidScatter[] = {
    { "logits", RAD_OPD_IN, 0 },        { "bias", RAD_OPD_WEIGHT, 0 },
    { "ids", RAD_OPD_OUT, 0 },          { "weights", RAD_OPD_OUT, 0 },
    { "sorted_tok", RAD_OPD_OUT, 0 },   { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count", RAD_OPD_OUT, 0 }, { "gate", RAD_OPD_OUT, 1 },
    { "inverse", RAD_OPD_OUT, 1 }
};
/* `logit_softcap`'s, matching ref's `pMnCap` (spec §2.3). `cap` must be > 0; the op refuses a
 * non-positive cap (a division by zero in every lane). */
static const RadParamSpec pMnCap[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    P_F64("cap"),
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};

static const RadOperandSpec oBinary[] = {
    { "a", RAD_OPD_IN,  0 },
    { "b", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};
/* `add`'s: `div` divides the rounded sum, rounded again (absent, 1: the plain sum). */
static const RadParamSpec pAdd[] = {
    P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("div"),
};
static const RadParamSpec pActMul[] = {
    P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype"),
};
/* `M` is the step's token count and bands the op; the segment count is cu_seqlens' length. */
static const RadParamSpec pSegMean[] = {
    P_INT("M"), P_INT("n"), P_STR("dtype"),
};
static const RadOperandSpec oSegMean[] = {
    { "x",          RAD_OPD_IN,  0 },
    { "cu_seqlens", RAD_OPD_IN,  0 },
    { "y",          RAD_OPD_OUT, 0 },
};
/* docs/OPS.md's `qk_norm_rope`: a fused [q | k | v] row in place, theta and not a table. */
static const RadParamSpec pQkNormRope[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_INT("n_head_kv"), P_F64("theta"),
    P_F64("eps"), P_F64O("scale"), P_STRO("mode"), P_INTO("rotary_dim"), P_F64O("wadd"),
    P_INTO("v_norm"), P_STRO("sections"), P_STRO("comps"),
};
/* `M` is the input row count, a multiple of `k`. */
static const RadParamSpec pPoolRows[] = {
    P_INT("M"), P_INT("n"), P_INT("k"), P_F64O("mult"), P_STR("dtype"),
};
/* `positions` may be absent at `rotary_dim` 0 (the norm alone), and an absent gain is a unit
 * gain. */
static const RadOperandSpec oQkNormRope[] = {
    { "qkv",       RAD_OPD_INOUT,  0 },
    { "positions", RAD_OPD_IN,     1 },
    { "q_w",       RAD_OPD_WEIGHT, 1 },
    { "k_w",       RAD_OPD_WEIGHT, 1 },
};
/* `scalar` 1 says `w` is one element rather than one row, for the shape description; the shim
 * reads it off the operand. */
static const RadParamSpec pMulWeight[] = {
    P_INT("M"), P_INT("n"), P_STR("dtype"), P_INTO("scalar"),
};
/* The audio (USM conformer) tower's operators; libref's lists, field for field. */
static const RadOperandSpec oClamp[] = {
    { "x",  RAD_OPD_IN,     0 },
    { "lo", RAD_OPD_WEIGHT, 0 },
    { "hi", RAD_OPD_WEIGHT, 0 },
    { "y",  RAD_OPD_OUT,    0 },
};
static const RadParamSpec pConv2d[] = {
    P_INT("M"), P_INT("width"), P_INT("cin"), P_INT("cout"), P_INT("kh"), P_INT("kw"),
    P_INT("stride"), P_INT("pad"), P_INTO("cu_div"), P_INTO("valid_stride"), P_STR("dtype"),
    P_INTO("t_last"), P_STRO("act"),
};
static const RadOperandSpec oConv2d[] = {
    { "x",          RAD_OPD_IN,     0 },
    { "w",          RAD_OPD_WEIGHT, 0 },
    { "cu_seqlens", RAD_OPD_IN,     0 },
    { "valid",      RAD_OPD_IN,     1 },
    { "y",          RAD_OPD_OUT,    0 },
    { "b",          RAD_OPD_WEIGHT, 1 },
};
static const RadParamSpec pConv2dUnfold[] = {
    P_INT("M"), P_INT("width"), P_INT("cin"), P_INT("kh"), P_INT("kw"), P_INT("stride"),
    P_INT("pad"), P_INTO("cu_div"), P_INTO("valid_stride"), P_INTO("t_last"), P_STR("dtype"),
};
static const RadOperandSpec oConv2dUnfold[] = {
    { "x",          RAD_OPD_IN,  0 },
    { "cu_seqlens", RAD_OPD_IN,  0 },
    { "valid",      RAD_OPD_IN,  1 },
    { "y",          RAD_OPD_OUT, 0 },
};
static const RadParamSpec pConv1dDw[] = {
    P_INT("M"), P_INT("n"), P_INT("k"), P_INTO("cu_div"), P_STR("dtype"),
};
static const RadOperandSpec oConv1dDw[] = {
    { "x",          RAD_OPD_IN,     0 },
    { "w",          RAD_OPD_WEIGHT, 0 },
    { "cu_seqlens", RAD_OPD_IN,     0 },
    { "y",          RAD_OPD_OUT,    0 },
    { "b",          RAD_OPD_WEIGHT, 1 },
};
static const RadParamSpec pRelSin[] = {
    P_INT("M"), P_INT("n"), P_F64O("max_timescale"), P_STR("dtype"),
};
static const RadOperandSpec oRelSin[] = {
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pSegSin[] = {
    P_INT("M"), P_INT("n"), P_INT("period"), P_F64O("max_timescale"), P_INTO("cu_div"),
    P_STR("dtype"),
};
static const RadOperandSpec oSegSin[] = {
    { "cu_seqlens", RAD_OPD_IN,  0 },
    { "y",          RAD_OPD_OUT, 0 },
};
static const RadParamSpec pAttnLocalRel[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_INT("left"), P_INT("right"),
    P_F64("softcap"), P_INTO("cu_div"), P_STR("dtype"),
};
static const RadOperandSpec oAttnLocalRel[] = {
    { "q",          RAD_OPD_IN,     0 },
    { "k",          RAD_OPD_IN,     0 },
    { "v",          RAD_OPD_IN,     0 },
    { "relk",       RAD_OPD_IN,     0 },
    { "pds",        RAD_OPD_WEIGHT, 0 },
    { "cu_seqlens", RAD_OPD_IN,     0 },
    { "out",        RAD_OPD_OUT,    0 },
};
static const RadOperandSpec oMulWeight[] = {
    { "x", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT,    0 },
};
static const RadOperandSpec oGateUp[] = {
    { "gate_up", RAD_OPD_IN,  0 },
    { "y",       RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oUnary[] = {
    { "x", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};
/* `fill` has no input: `value` is the whole of what it writes. */
static const RadParamSpec pFill[] = {
    P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("value"),
};
static const RadOperandSpec oFill[] = {
    { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oScale[] = {
    { "x", RAD_OPD_IN,     0 },
    { "s", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT,    0 },
};
/* `cast` takes two dtype strings, `from` and `to`, as ref spells them. */
static const RadParamSpec pCast[] = {
    { "M",    RAD_P_INT, RAD_REQUIRED },
    { "n",    RAD_P_INT, RAD_REQUIRED },
    { "from", RAD_P_STR, RAD_REQUIRED },
    { "to",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGather[] = {
    { "x",   RAD_OPD_IN,  0 },
    { "idx", RAD_OPD_IN,  0 },
    { "y",   RAD_OPD_OUT, 0 },
};

static const RadOperandSpec oScatterRows[] = {
    { "v",   RAD_OPD_IN,    0 },
    { "idx", RAD_OPD_IN,    0 },
    { "x",   RAD_OPD_INOUT, 0 },
};

static const RadParamSpec pRope[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    P_F64("theta"),
    P_F64("scale"),
    { "mode",       RAD_P_STR, RAD_REQUIRED },
    { "rotary_dim", RAD_P_INT, RAD_OPTIONAL },
    { "sections",   RAD_P_STR, RAD_OPTIONAL },
    { "yarn_factor", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_low", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_high", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_magnitude", RAD_P_F64, RAD_OPTIONAL },
    { "ladder",     RAD_P_STR, RAD_OPTIONAL },   /* the base ladder's width: rotary|head */
    /* Llama 3's frequency scaling (abi/rad_yarn.h, RadLlama3). */
    { "llama3_factor", RAD_P_F64, RAD_OPTIONAL },
    { "llama3_low",    RAD_P_F64, RAD_OPTIONAL },
    { "llama3_high",   RAD_P_F64, RAD_OPTIONAL },
    { "llama3_orig",   RAD_P_F64, RAD_OPTIONAL },
};

/* ---------------------------------------------------------------------- the vision tower */
static const RadParamSpec pLayernorm[] = {
    P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
};
static const RadOperandSpec oLayernorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "b", RAD_OPD_WEIGHT, 1 },
    { "y", RAD_OPD_OUT,    0 },
};
static const RadParamSpec pGridEmbed[] = {
    P_INT("M"), P_INT("n"), P_INT("side"), P_STR("dtype"),
};
static const RadOperandSpec oGridEmbed[] = {
    { "x",     RAD_OPD_INOUT,  0 },
    { "table", RAD_OPD_WEIGHT, 0 },
    { "coord", RAD_OPD_IN,     0 },
};
static const RadParamSpec pGemmBias[] = {
    P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
    { "act", RAD_P_STR, RAD_OPTIONAL },
};
static const RadOperandSpec oGemmBias[] = {
    { "a",    RAD_OPD_IN,     0 },
    { "b",    RAD_OPD_WEIGHT, 0 },
    { "bias", RAD_OPD_WEIGHT, 0 },
    { "res",  RAD_OPD_IN,     1 },
    { "y",    RAD_OPD_OUT,    0 },
};
static const RadOperandSpec oRope[] = {
    { "qkv",       RAD_OPD_INOUT, 0 },
    { "positions", RAD_OPD_IN,    0 },
};

static const RadParamSpec pKvStore[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "block_size", RAD_P_INT, RAD_REQUIRED },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
    /* The descales the reader multiplies back by, so the store divides. Unused for a bf16 cache;
     * an fp8 one defaults them to 1.0, vLLM's default and what the attention kernels assume when
     * their descale operands are absent. They are one number per (sequence, head), so they cannot
     * be derived from the tokens written (r4d.h, r4d_kv_store_fp8). */
    P_F64O("k_scale"),
    P_F64O("v_scale"),
};
static const RadOperandSpec oKvStore[] = {
    { "k",            RAD_OPD_IN,    0 },
    { "v",            RAD_OPD_IN,    0 },
    { "slot_mapping", RAD_OPD_IN,    0 },
    { "kv_cache",     RAD_OPD_INOUT, 0 },
};

static const RadParamSpec pEmbed[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },   /* this rank's rows when vocab_offset is given */
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
    { "vocab_offset", RAD_P_INT, RAD_OPTIONAL },  /* first global row; vocab-parallel embedding */
    /* The scale embed_lookup_q's layout quantises with; embed_lookup_q shares this list, which
     * libref fixes. */
    { "wscale",  RAD_P_F64, RAD_OPTIONAL },
};
/* embed_lookup's own: pEmbed and `mult`, the factor every gathered row is multiplied by (Gemma's
 * sqrt(hidden)); libref's list, which embed_lookup_q does not share. */
static const RadParamSpec pEmbedM[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
    { "vocab_offset", RAD_P_INT, RAD_OPTIONAL },
    { "wscale",  RAD_P_F64, RAD_OPTIONAL },
    { "mult",    RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oEmbed[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "wte",    RAD_OPD_WEIGHT, 0 },
    { "x",      RAD_OPD_OUT,    0 },
};
/* The quantised gather's, libref's list (ref_registry.cpp): a one-element `scale`, absent for a
 * plain table, and `ahead`, the next prompt's ids for a host reader fetching rows from a file. */
static const RadOperandSpec oEmbedQ[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "wte",    RAD_OPD_WEIGHT, 0 },
    { "scale",  RAD_OPD_WEIGHT, 1 },
    { "x",      RAD_OPD_OUT,    0 },
    { "ahead",  RAD_OPD_IN,     1 },
};
static const RadParamSpec pPleGate[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED },
    { "n",        RAD_P_INT, RAD_REQUIRED },
    { "hc",       RAD_P_INT, RAD_REQUIRED },
    { "eps",      RAD_P_F64, RAD_REQUIRED },
    { "dtype",    RAD_P_STR, RAD_REQUIRED },
    { "wadd",     RAD_P_F64, RAD_OPTIONAL },
    { "gate_eps", RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oPleGate[] = {
    { "k",       RAD_OPD_IN,     0 },
    { "q",       RAD_OPD_IN,     0 },
    { "v",       RAD_OPD_IN,     0 },
    { "w_key",   RAD_OPD_WEIGHT, 0 },
    { "w_query", RAD_OPD_WEIGHT, 0 },
    { "w_conv",  RAD_OPD_WEIGHT, 0 },
    { "gv",      RAD_OPD_OUT,    0 },
    { "gvn",     RAD_OPD_OUT,    0 },
};

static const RadParamSpec pPleConv[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED },
    { "n",        RAD_P_INT, RAD_REQUIRED },
    { "width",    RAD_P_INT, RAD_REQUIRED },
    { "dilation", RAD_P_INT, RAD_REQUIRED },
    { "dtype",    RAD_P_STR, RAD_REQUIRED },
};
/* The optionals unite gdn_conv_prep's and gdn_conv_update's contracts: `has_init` serves
 * prefill, `num_accepted` decode. */
static const RadOperandSpec oPleConv[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "w",            RAD_OPD_WEIGHT, 0 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "resid",        RAD_OPD_IN,     1 },
    { "cache_idx",    RAD_OPD_IN,     1 },
    { "has_init",     RAD_OPD_IN,     1 },
    { "num_accepted", RAD_OPD_IN,     1 },
    { "y",            RAD_OPD_OUT,    0 },
};
/* ple_conv's operands without `resid`, its parameters without `dilation`. */
static const RadParamSpec pCausalConv[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "width", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oCausalConv[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "w",            RAD_OPD_WEIGHT, 0 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "cache_idx",    RAD_OPD_IN,     1 },
    { "has_init",     RAD_OPD_IN,     1 },
    { "num_accepted", RAD_OPD_IN,     1 },
    { "y",            RAD_OPD_OUT,    0 },
};

/* PLE's hashed n-gram ids. `state` is a rolling window of committed ids (RAD_KV_CONV's shape,
 * ple_conv's contract), except that a cold window holds EOS, not zero, since zero is a real token
 * id. */
static const RadParamSpec pNgramIds[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "heads", RAD_P_INT, RAD_REQUIRED },
    { "ngram", RAD_P_INT, RAD_REQUIRED },
    { "eos",   RAD_P_INT, RAD_REQUIRED },
};
static const RadOperandSpec oNgramIds[] = {
    { "tok",          RAD_OPD_IN,     0 },
    { "state",        RAD_OPD_INOUT,  0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "mult",         RAD_OPD_WEIGHT, 0 },
    { "vocab_sizes",  RAD_OPD_WEIGHT, 0 },
    { "offsets",      RAD_OPD_WEIGHT, 0 },
    { "cache_idx",    RAD_OPD_IN,     1 },
    { "has_init",     RAD_OPD_IN,     1 },
    { "num_accepted", RAD_OPD_IN,     1 },
    { "ids",          RAD_OPD_OUT,    0 },
    { "ahead_ids",    RAD_OPD_OUT,    1 },
};

static const RadParamSpec pLogits[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oLogits[] = {
    { "x",       RAD_OPD_IN,     0 },
    { "lm_head", RAD_OPD_WEIGHT, 0 },
    { "logits",  RAD_OPD_OUT,    0 },
};

/* Every sampler scalar is per row and arrives in `params`, an array of RadSampleParams
 * (abi/rad_sample.h). Parameters are geometry, frozen at declare, so only the two extents are
 * parameters. */
static const RadParamSpec pMVocab[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
};
static const RadParamSpec pMCand[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "n_cand", RAD_P_INT, RAD_REQUIRED },
};
/* The history stages read global token ids and write this rank's columns, so under a split
 * vocabulary they take the rank's first row, `vocab_off` (as sample_topk does). Absent is 0. */
static const RadParamSpec pPenal[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oPenalties[] = {
    { "logits",  RAD_OPD_INOUT, 0 },
    { "params",  RAD_OPD_IN,    0 },
    { "history", RAD_OPD_IN,    0 },
};
/* `breakers` is optional: a u8 plane shaped like `history`, nonzero where that position's token
 * is a whole DRY sequence breaker and so never penalised. */
static const RadOperandSpec oDry[] = {
    { "logits",   RAD_OPD_INOUT, 0 },
    { "params",   RAD_OPD_IN,    0 },
    { "history",  RAD_OPD_IN,    0 },
    { "breakers", RAD_OPD_IN,    1 },
};
static const RadOperandSpec oTemp[] = {
    { "logits", RAD_OPD_INOUT, 0 },
    { "params", RAD_OPD_IN,    0 },
};
static const RadOperandSpec oMask[] = {
    { "logits",  RAD_OPD_INOUT, 0 },
    { "params",  RAD_OPD_IN,    0 },
    { "bitmask", RAD_OPD_IN,    0 },
};
/* `pairs`, the optional vocab-parallel output: [M, 2 * n_cand] u32 of this rank's candidates as
 * (u32 global id, f32 logit), which sample_merge_topk reads after the all-gather. The kernel adds
 * `vocab_off`, so the gather needs no shard layout. Absent: the single-rank top-k. */
static const RadParamSpec pTopk[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oTopk[] = {
    { "logits",   RAD_OPD_IN,  0 },
    { "params",   RAD_OPD_IN,  0 },
    { "cand_idx", RAD_OPD_OUT, 0 },
    { "cand_val", RAD_OPD_OUT, 0 },
    { "pairs",    RAD_OPD_OUT, 1 },
};
static const RadOperandSpec oArgmax[] = {
    { "logits", RAD_OPD_IN,  0 },
    { "params", RAD_OPD_IN,  0 },
    { "token",  RAD_OPD_OUT, 0 },
};
/* `stats` is [M, 2 + 2k] f32: max, sum of exp(l - max), then k (id, logit) pairs. `k` is the
 * deployment's width, `vocab_off` sample_topk's. */
static const RadParamSpec pLogprobs[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "k",         RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oLogprobs[] = {
    { "logits", RAD_OPD_IN,  0 },
    { "params", RAD_OPD_IN,  0 },
    { "stats",  RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oNarrow[] = {
    { "cand_idx", RAD_OPD_INOUT, 0 },
    { "cand_val", RAD_OPD_INOUT, 0 },
    { "params",   RAD_OPD_IN,    0 },
};
static const RadOperandSpec oPick[] = {
    { "cand_idx", RAD_OPD_IN,  0 },
    { "cand_val", RAD_OPD_IN,  0 },
    { "params",   RAD_OPD_IN,  0 },
    { "token",    RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oMergeTopk[] = {
    { "gathered", RAD_OPD_IN,  0 },
    { "params",   RAD_OPD_IN,  0 },
    { "cand_idx", RAD_OPD_OUT, 0 },
    { "cand_val", RAD_OPD_OUT, 0 },
};

/* ------------------------------------------------------- pair_argmax, the joins, expert_scale
 *
 * pair_argmax is libref's schema restated. The other four ops, and attn_paged_qrow (attn_paged's
 * schema), are this library's own; libref deliberately has no row for them, so that where libr4d
 * is not loaded (CPU-only, a --kernels list without it) the architecture falls back to the chain
 * each replaces. docs/OPS.md lists all six (checked by the ops_doc test). */
static const RadParamSpec pPairArgmax[] = {
    P_INT("M"),
    P_INT("n"),
    P_STR("dtype"),
};
static const RadOperandSpec oPairArgmax[] = {
    { "pairs", RAD_OPD_IN,  0 },
    { "idx",   RAD_OPD_OUT, 0 },
    { "val",   RAD_OPD_OUT, 1 },
};
static const RadParamSpec pNormJoin[] = {
    P_INT("M"),
    P_INT("n"),
    P_F64("eps"),
    P_STR("dtype"),
};
static const RadOperandSpec oNormAddNorm[] = {
    { "a",   RAD_OPD_IN,     0 },
    { "w_a", RAD_OPD_WEIGHT, 0 },
    { "x",   RAD_OPD_INOUT,  0 },
    { "w_y", RAD_OPD_WEIGHT, 0 },
    { "y",   RAD_OPD_OUT,    0 },
};
static const RadParamSpec pNormAddNormRoute[] = {
    P_INT("M"),
    P_INT("n"),
    P_F64("eps"),
    P_STR("dtype"),
    P_F64("mult"),
    P_INT("group"),
};
static const RadOperandSpec oNormAddNormRoute[] = {
    { "a",   RAD_OPD_IN,     0 },
    { "w_a", RAD_OPD_WEIGHT, 0 },
    { "x",   RAD_OPD_INOUT,  0 },
    { "w_y", RAD_OPD_WEIGHT, 0 },
    { "y",   RAD_OPD_OUT,    0 },
    { "w_r", RAD_OPD_WEIGHT, 0 },
    { "r",   RAD_OPD_OUT,    0 },
    { "w_e", RAD_OPD_WEIGHT, 0 },
    { "q",   RAD_OPD_OUT,    0 },
    { "qs",  RAD_OPD_OUT,    0 },
};
static const RadOperandSpec oJoinAddScale[] = {
    { "d",   RAD_OPD_IN,     0 },
    { "w_d", RAD_OPD_WEIGHT, 0 },
    { "r",   RAD_OPD_IN,     0 },
    { "w_r", RAD_OPD_WEIGHT, 0 },
    { "w_j", RAD_OPD_WEIGHT, 0 },
    { "x",   RAD_OPD_INOUT,  0 },
    { "s",   RAD_OPD_WEIGHT, 0 },
    { "w_n", RAD_OPD_WEIGHT, 1 },
    { "y",   RAD_OPD_OUT,    1 },
};
static const RadParamSpec pExpertScale[] = {
    P_INT("M"),
    P_INT("n_expert"),
    P_INT("top_k"),
    P_STR("dtype"),
};
static const RadOperandSpec oExpertScale[] = {
    { "ids", RAD_OPD_IN,     0 },
    { "w_e", RAD_OPD_WEIGHT, 0 },
    { "ew",  RAD_OPD_INOUT,  0 },
};

/* ------------------------------------------------------------------------------- runtime LoRA
 * libref's schemas field for field (libref/ref_registry.cpp says what they mean): the loader
 * refuses a library whose schema differs from the first declarer's. The pools are plain inputs
 * (program buffers the engine fills between steps), and `dtype` is the tensor dtype name of `a`
 * (shrink) or `y` (expand), which the rows below match on. */
static const RadParamSpec pLoraShrink[] = {
    P_INT("M"),
    P_INT("K"),
    P_INT("R"),
    P_INT("S"),
    P_INTO("group"),
    P_STR("dtype"),
};
static const RadOperandSpec oLoraShrink[] = {
    { "a",       RAD_OPD_IN,  0 },
    { "a_scale", RAD_OPD_IN,  1 },
    { "slot",    RAD_OPD_IN,  0 },
    { "lora_a",  RAD_OPD_IN,  0 },
    { "t",       RAD_OPD_OUT, 0 },
};
static const RadParamSpec pLoraExpand[] = {
    P_INT("M"),
    P_INT("N"),
    P_INT("R"),
    P_INT("S"),
    P_STR("dtype"),
};
static const RadOperandSpec oLoraExpand[] = {
    { "t",      RAD_OPD_IN,    0 },
    { "slot",   RAD_OPD_IN,    0 },
    { "lora_b", RAD_OPD_IN,    0 },
    { "scale",  RAD_OPD_IN,    0 },
    { "y",      RAD_OPD_INOUT, 0 },
};

#define SCHEMA(op, p, o, doc) { (op), ROW(p), ROW(o), (doc) }

/* ---- image generation (r4d_gen.hip, r4d_attn_gen.hip) -------------------------------------
 * ref's spellings (libref/ref_registry.cpp), field for field. */
static const RadParamSpec pGenMn[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadParamSpec pLnMod[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLnMod[] = {
    { "x", RAD_OPD_IN, 0 }, { "s", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pLnModQ[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                        P_INT("gated"), P_STR("dtype") };
static const RadOperandSpec oLnModQ[] = {
    { "y", RAD_OPD_IN, 1 }, { "g", RAD_OPD_IN, 1 }, { "s", RAD_OPD_IN, 0 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 }, { "q", RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 }, { "z", RAD_OPD_OUT, 1 },
};
/* `act` is the gate's activation, "tanh" (the default) or "none". */
static const RadParamSpec pTableAdd[] = { P_INT("M"), P_INT("n"), P_INT("L"), P_STR("dtype") };
static const RadOperandSpec oTableAdd[] = {
    { "x", RAD_OPD_IN, 0 }, { "t", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pMixRows[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_STR("dtype") };
static const RadParamSpec pChanStats[] = { P_INT("M"), P_INT("n"), P_STR("mode"), P_INTO("var"),
                                           P_F64O("eps"), P_STR("dtype") };
static const RadOperandSpec oChanStats[] = {
    { "x", RAD_OPD_IN, 0 }, { "mean", RAD_OPD_WEIGHT, 0 }, { "s", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oMixRows[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pGateAdd[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_STRO("act") };
static const RadOperandSpec oGateAdd[] = {
    { "y", RAD_OPD_IN, 0 }, { "g", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "x", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pTimestep[] = { P_INT("M"), P_INT("n"), P_F64("scale"),
                                          P_F64("max_period"), P_INTO("round"), P_STR("dtype") };
static const RadOperandSpec oTimestep[] = { { "t", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oGather2[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pLane[] = { P_INT("M"), P_INT("n"), P_INT("stride"), P_STR("dtype") };
static const RadOperandSpec oLaneGather[] = {
    { "src", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oLaneStore[] = {
    { "src", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 }, { "dst", RAD_OPD_INOUT, 0 },
};
static const RadOperandSpec oRandnLanes[] = {
    { "flags", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "dst", RAD_OPD_INOUT, 0 },
};
static const RadOperandSpec oFlowEuler[] = {
    { "v", RAD_OPD_IN, 0 }, { "sigma", RAD_OPD_IN, 0 }, { "sigma_next", RAD_OPD_IN, 0 },
    { "cfg", RAD_OPD_IN, 0 }, { "pair", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 },
    { "flags", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "inp_lat", RAD_OPD_IN, 1 }, { "inp_mask", RAD_OPD_IN, 1 }, { "state", RAD_OPD_INOUT, 0 },
    { "preview", RAD_OPD_INOUT, 1 },
};
/* The k-diffusion sampler's input and step: `pred` "eps" or "v", the lanes f32. */
static const RadParamSpec pKdStep[] = { P_INT("M"), P_INT("n"), P_INT("stride"), P_STR("pred"),
                                        P_STR("dtype") };
static const RadOperandSpec oKdInput[] = {
    { "sigma", RAD_OPD_IN, 0 }, { "pair", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 },
    { "flags", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "state", RAD_OPD_INOUT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oKdStep[] = {
    { "out", RAD_OPD_IN, 0 }, { "sigma", RAD_OPD_IN, 0 }, { "sigma_next", RAD_OPD_IN, 0 },
    { "sigma_prev", RAD_OPD_IN, 0 }, { "cfg", RAD_OPD_IN, 0 }, { "pair", RAD_OPD_IN, 0 },
    { "lane", RAD_OPD_IN, 0 }, { "flags", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 },
    { "sampler", RAD_OPD_IN, 0 }, { "step", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "state", RAD_OPD_INOUT, 0 }, { "hist", RAD_OPD_INOUT, 0 }, { "preview", RAD_OPD_INOUT, 1 },
};
/* `round` 0 keeps the lists and the arithmetic in f32 (default 1, bf16 lists and roundings). */
static const RadParamSpec pChanAffine[] = { P_INT("M"), P_INT("n"), P_STR("mul"), P_STR("add"),
                                            P_STR("mode"), P_STR("dtype"), P_INTO("round") };
static const RadOperandSpec oGenUnary[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pConvNhwc[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                          P_INT("stride"), P_STR("pad"), P_INT("up"),
                                          P_STR("dtype") };
static const RadOperandSpec oConvNhwc[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pIm2col[] = { P_INT("M"), P_INT("cin"), P_INT("k"), P_INT("stride"),
                                        P_STR("pad"), P_INT("up"), P_STR("dtype") };
static const RadParamSpec pL2Norm[] = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };
static const RadOperandSpec oL2Norm[] = {
    { "x", RAD_OPD_IN, 0 }, { "gamma", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pSpaceDepth[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("ft"),
                                            P_STR("dtype") };
static const RadOperandSpec oSpaceDepth[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_INOUT, 0 } };
static const RadOperandSpec oScatterAdd[] = {
    { "v", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pAttnPS[] = { P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"),
                                        P_INTD("block_size"), P_INT("causal"), P_STR("q_dtype"),
                                        P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
                                        P_INTO("n_head"), P_INTO("max_seqs"), P_STRO("qk"),
                                        P_STRO("pv") };
static const RadOperandSpec oAttnPS[] = {
    { "q", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 0 },
    { "kv_cache", RAD_OPD_IN, 1 }, { "block_table", RAD_OPD_IN, 1 }, { "seqused", RAD_OPD_IN, 1 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "k_descale", RAD_OPD_IN, 1 }, { "v_descale", RAD_OPD_IN, 1 },
    { "k_i8", RAD_OPD_IN, 1 }, { "k_scale", RAD_OPD_IN, 1 }, { "out", RAD_OPD_OUT, 0 },
    { "v_scale", RAD_OPD_IN, 1 },
};

/* ---- audio generation (r4d_audio.hip); the same specs as libref's (ref_registry.cpp). */
static const RadParamSpec pConv1d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                        P_INT("dil"), P_STRO("act"), P_STR("dtype"),
                                        P_INTO("causal") };
static const RadOperandSpec oConv1d[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "res", RAD_OPD_IN, 1 }, { "y", RAD_OPD_OUT, 0 }, { "b", RAD_OPD_WEIGHT, 1 },
};
static const RadParamSpec pConvT1d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("stride"),
                                         P_STR("dtype"), P_INTO("pad"), P_INTO("k") };
static const RadOperandSpec oConvT1d[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pSnake[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_INTO("log") };
static const RadOperandSpec oSnake[] = {
    { "x", RAD_OPD_IN, 0 }, { "alpha", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
    { "beta", RAD_OPD_WEIGHT, 1 },
};
static const RadParamSpec pCfgTopk[] = { P_INT("M"), P_INT("n_vocab"), P_INT("k"), P_INT("rep"),
                                         P_INTO("cond_k"), P_F64("cfg"), P_F64O("temp"),
                                         P_INT("stream"), P_STR("dtype"), P_INTO("sup_lo"),
                                         P_INTO("sup_hi"), P_F64O("penalty") };
static const RadOperandSpec oCfgTopk[] = {
    { "logits", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 }, { "frame", RAD_OPD_IN, 0 },
    { "ids", RAD_OPD_OUT, 0 }, { "cfg", RAD_OPD_IN, 1 }, { "lane", RAD_OPD_IN, 1 },
    { "hist", RAD_OPD_INOUT, 1 },
};
static const RadParamSpec pRvqEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_codes"),
                                          P_INT("vocab"), P_INT("off0"), P_F64("scale"),
                                          P_STR("dtype"), P_INTO("round") };
static const RadOperandSpec oRvqEmbed[] = {
    { "codes", RAD_OPD_IN, 0 }, { "t0", RAD_OPD_WEIGHT, 0 }, { "t1", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pFrameMix[] = { P_INT("M"), P_INT("n_embd"), P_INT("L"), P_INT("stride"),
                                          P_INTO("rep"), P_STR("dtype") };
static const RadOperandSpec oFrameMix[] = {
    { "hid", RAD_OPD_IN, 0 }, { "logits", RAD_OPD_WEIGHT, 0 }, { "scale", RAD_OPD_WEIGHT, 0 },
    { "lane", RAD_OPD_IN, 0 }, { "frame", RAD_OPD_IN, 0 }, { "status", RAD_OPD_IN, 0 },
    { "dst", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pEndStatus[] = { P_INT("M"), P_INT("end"), P_INT("rep") };
static const RadOperandSpec oEndStatus[] = {
    { "ids", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 }, { "status", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pFrameStore[] = { P_INT("M"), P_INT("n_codes"), P_INT("stride"),
                                            P_INTO("rep"), P_INTO("skip") };
static const RadOperandSpec oFrameStore[] = {
    { "codes", RAD_OPD_IN, 0 }, { "lane", RAD_OPD_IN, 0 }, { "frame", RAD_OPD_IN, 0 },
    { "status", RAD_OPD_IN, 0 }, { "dst", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pFourier[] = { P_INT("M"), P_INT("n"), P_INTO("round"), P_STR("dtype") };
static const RadOperandSpec oFourier[] = {
    { "t", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pWinInput[] = { P_INT("M"), P_INT("n_lat"), P_INT("n_zero"),
                                          P_INT("n_cond"), P_INT("stride"), P_STR("dtype") };
static const RadOperandSpec oWinInput[] = {
    { "lat", RAD_OPD_IN, 0 }, { "cond", RAD_OPD_IN, 0 }, { "lat_cu", RAD_OPD_IN, 0 },
    { "win_lane", RAD_OPD_IN, 0 }, { "pair", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pWinPrep[] = { P_INT("M"), P_INT("n"), P_INT("stride"), P_INT("ov"),
                                         P_STR("dtype") };
static const RadOperandSpec oWinPrep[] = {
    { "flags", RAD_OPD_IN, 0 }, { "t", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 },
    { "win_index", RAD_OPD_IN, 0 }, { "win_lane", RAD_OPD_IN, 0 }, { "pair", RAD_OPD_IN, 0 },
    { "lat_cu", RAD_OPD_IN, 0 }, { "prev", RAD_OPD_IN, 0 }, { "noise", RAD_OPD_INOUT, 0 },
    { "lat", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pWinFinish[] = { P_INT("M"), P_INT("n"), P_INT("n_cond"), P_INT("stride"),
                                           P_INT("ov"), P_STR("dtype") };
static const RadOperandSpec oWinFinish[] = {
    { "flags", RAD_OPD_IN, 0 }, { "pair", RAD_OPD_IN, 0 }, { "win_lane", RAD_OPD_IN, 0 },
    { "lat_cu", RAD_OPD_IN, 0 }, { "cond", RAD_OPD_IN, 0 }, { "lat", RAD_OPD_INOUT, 0 },
    { "prev", RAD_OPD_INOUT, 0 }, { "prev_cond", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pChanSplit[] = { P_INT("M"), P_INT("n"), P_INT("parts"), P_STR("dtype") };
static const RadParamSpec pInterleave[] = { P_INT("M"), P_STR("dtype") };
static const RadOperandSpec oInterleave2[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pIota[] = { P_INT("M"), P_INT("start"), P_INT("step"),
                                      P_INTO("period") };
static const RadOperandSpec oIota[] = { { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pCuRepeat[] = { P_INT("M"), P_INT("mult") };
static const RadOperandSpec oCuRepeat[] = { { "len", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pIntAffine[] = { P_INT("M"), P_STR("src"), P_STR("mul"), P_STR("add") };
static const RadOperandSpec oIntAffine[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pSnakeAa[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("k"),
                                         P_INTO("log"), P_STR("dtype") };
static const RadOperandSpec oSnakeAa[] = {
    { "x", RAD_OPD_IN, 0 }, { "alpha", RAD_OPD_WEIGHT, 0 }, { "beta", RAD_OPD_WEIGHT, 0 },
    { "f_up", RAD_OPD_WEIGHT, 0 }, { "f_down", RAD_OPD_WEIGHT, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "y", RAD_OPD_OUT, 0 },
};

/* ---- video generation (r4d_video.hip); the same specs as libref's (ref_registry.cpp). `geom` is
 * the request's latent geometry, i32 on the device. */
static const RadParamSpec pRmsMod[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INTO("gated"),
                                        P_STR("dtype"), P_F64O("wadd"),
                                         P_INTO("norm_once"), P_STRO("norm") };
static const RadOperandSpec oRmsMod[] = {
    { "y", RAD_OPD_IN, 1 }, { "g", RAD_OPD_IN, 1 }, { "s", RAD_OPD_IN, 0 },
    { "w", RAD_OPD_WEIGHT, 1 }, { "cu_seqlens", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 },
    { "z", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pRmsModQ[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                         P_INT("gated"), P_STR("dtype"), P_F64O("wadd"),
                                         P_INTO("norm_once"), P_STRO("norm") };
static const RadOperandSpec oRmsModQ[] = {
    { "y", RAD_OPD_IN, 1 }, { "g", RAD_OPD_IN, 1 }, { "s", RAD_OPD_IN, 0 },
    { "w", RAD_OPD_WEIGHT, 1 }, { "cu_seqlens", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 },
    { "q", RAD_OPD_OUT, 0 }, { "scale", RAD_OPD_OUT, 0 }, { "z", RAD_OPD_OUT, 1 },
};
static const RadOperandSpec oFlowX0[] = {
    { "v", RAD_OPD_IN, 0 }, { "t", RAD_OPD_IN, 0 }, { "sigma", RAD_OPD_IN, 0 },
    { "sigma_next", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec pPatchUnfold[] = { P_INT("M"), P_INT("C"), P_INT("pt"), P_INT("ph"),
                                             P_INT("pw"), P_STR("mul"), P_STR("add"),
                                             P_STR("dtype") };
static const RadOperandSpec oPatchUnfold[] = {
    { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pVtileGather[] = { P_INT("M"), P_INT("n"), P_INT("tile"),
                                             P_INT("overlap"), P_INT("frames"), P_INT("period"),
                                             P_INT("n_reg"), P_STR("dtype") };
static const RadOperandSpec oVtileGather[] = {
    { "x", RAD_OPD_IN, 0 }, { "reg", RAD_OPD_WEIGHT, 0 }, { "geom", RAD_OPD_IN, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pVtileStitch[] = { P_INT("M"), P_INT("tile"), P_INT("overlap"),
                                             P_INT("scale"), P_INT("pt"), P_INT("skip"),
                                             P_INT("keep"), P_INT("blend"), P_INT("period"),
                                             P_STR("mean"), P_STR("std") };
static const RadOperandSpec oVtileStitch[] = {
    { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 }, { "prev", RAD_OPD_INOUT, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pGridPos3[] = { P_INT("M"), P_INT("T"), P_INT("H"), P_INT("W"),
                                          P_INT("period"), P_F64("scale") };
static const RadOperandSpec oGridPos3[] = { { "y", RAD_OPD_OUT, 0 } };

/* ---- a video latent upscaler (r4d_upscaler.hip); the same specs as libref's. `geom` is the
 * request's volume, i32 on the device; patch_fold takes patch_unfold's specs. */
static const RadParamSpec pConv3d[] = { P_INT("M"), P_INT("cin"), P_INT("cout"), P_INT("k"),
                                        P_INT("kt"), P_STR("dtype"), P_STRO("tpad"),
                                        P_INTO("tpre") };
static const RadOperandSpec oConv3d[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
    { "geom", RAD_OPD_IN, 0 }, { "res", RAD_OPD_IN, 1 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pGroupNorm[] = { P_INT("M"), P_INT("n"), P_INT("groups"), P_F64("eps"),
                                           P_STR("act"), P_STR("dtype") };
static const RadOperandSpec oGroupNorm[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
    { "mod", RAD_OPD_IN, 1 }, { "stats", RAD_OPD_INOUT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pTconvDw[] = { P_INT("M"), P_INT("n"), P_INT("k"), P_STR("dtype") };
static const RadOperandSpec oTconvDw[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
    { "geom", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oResizeBil[] = {
    { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pFlowRenoise[] = { P_INT("M"), P_INT("n"), P_INT("stream"),
                                             P_STR("dtype") };
static const RadOperandSpec oFlowRenoise[] = {
    { "sigma", RAD_OPD_IN, 0 }, { "seed", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 },
};

/* ---- LTX-2 (r4d_ltx.hip); the same specs as libref's. */
static const RadParamSpec pRopeSplit[] = { P_INT("M"), P_INT("n_head"), P_INT("head_dim"),
                                           P_INT("dim"), P_INT("n_dims"), P_F64("theta"),
                                           P_STR("max"), P_INTO("head0"), P_STR("dtype") };
static const RadOperandSpec oRopeSplit[] = {
    { "x", RAD_OPD_INOUT, 0 }, { "positions", RAD_OPD_IN, 0 },
};
static const RadParamSpec pHeadGate[] = { P_INT("M"), P_INT("n_head"), P_INT("head_dim"),
                                          P_STR("dtype") };
static const RadOperandSpec oHeadGate[] = { { "g", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 } };
static const RadParamSpec pLtxGuide[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oLtxGuide[] = {
    { "vc", RAD_OPD_IN, 0 }, { "vu", RAD_OPD_IN, 1 }, { "vs", RAD_OPD_IN, 1 },
    { "vm", RAD_OPD_IN, 1 }, { "scales", RAD_OPD_IN, 0 }, { "sched", RAD_OPD_IN, 0 },
    { "stats", RAD_OPD_INOUT, 0 }, { "x", RAD_OPD_INOUT, 0 },
};
/* LTX-2's VAE, audio VAE and vocoder (r4d_ltx.hip); `geom` is x's volume, i32 on the device. */
static const RadParamSpec pPixelNorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("form"),
                                           P_STRO("act"), P_INTO("ileave"), P_STR("dtype"),
                                           P_F64O("mult") };
static const RadOperandSpec oPixelNorm[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pD2s3d[] = { P_INT("M"), P_INT("C"), P_INT("st"), P_INT("sh"),
                                       P_INT("sw"), P_INTO("drop"), P_STR("dtype") };
static const RadOperandSpec oD2s3d[] = { { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 },
                                         { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pS2d3d[] = { P_INT("M"), P_INT("C"), P_INT("st"), P_INT("sh"),
                                       P_INT("sw"), P_INTO("tpre"), P_INTO("group"),
                                       P_STR("mode"), P_INTO("add"), P_STR("dtype") };
static const RadOperandSpec oS2d3d[] = { { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 },
                                         { "y", RAD_OPD_INOUT, 0 } };
static const RadParamSpec pPixPatch[] = { P_INT("M"), P_INT("p"), P_INT("C"), P_STRO("order"),
                                          P_STR("dtype") };
static const RadOperandSpec oPixPatch[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pVolIndex[] = { P_INT("M") };
static const RadOperandSpec oVolIndex[] = { { "geom", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pTileGrid[] = { P_INT("M"), P_INT("tile"), P_INT("stride"), P_STRO("rule"),
                                          P_INTO("min"), P_INTO("tile_t"), P_INTO("stride_t"),
                                          P_INTO("min_t"), P_INTO("tail") };
static const RadOperandSpec oTileGrid[] = { { "geom", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pTileCrop[] = { P_INT("M"), P_INT("C"), P_STR("dtype") };
static const RadOperandSpec oTileCrop[] = { { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 },
                                            { "box", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pTileStitch[] = { P_INT("M"), P_INT("tile"), P_INT("stride"),
                                            P_STRO("rule"), P_INTO("min"), P_INTO("tile_t"),
                                            P_INTO("stride_t"), P_INTO("min_t"), P_INT("scale"),
                                            P_INTO("scale_t"), P_INT("p"), P_INT("C"),
                                            P_STRO("order"), P_STR("dtype") };
static const RadOperandSpec oTileStitch[] = { { "x", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 },
                                              { "first", RAD_OPD_IN, 0 },
                                              { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pStftMel[] = { P_INT("M"), P_INT("hop"), P_INT("nfft"), P_INT("n_mel"),
                                         P_INT("ch"), P_STR("dtype") };
static const RadOperandSpec oStftMel[] = { { "x", RAD_OPD_IN, 0 }, { "basis", RAD_OPD_WEIGHT, 0 },
                                           { "mel", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pBweUp[] = { P_INT("M"), P_INT("ratio"), P_INT("ch"), P_STR("dtype") };
static const RadOperandSpec oBweUp[] = { { "x", RAD_OPD_IN, 0 }, { "res", RAD_OPD_IN, 0 },
                                         { "y", RAD_OPD_OUT, 0 } };

/* ---- video sparse attention (r4d_vsa.hip); the same specs as libref's. `segs` cuts the packed
 * rows into dense segments and the video grid, `geom` is the request's latent geometry; `sel` is
 * vsa_select's output and attn_vsa's input. `v` of vsa_select is read only for `coarse`. */
static const RadParamSpec pVsa[] = { P_INT("M"), P_INT("head_dim"), P_INT("tile_t"),
                                     P_INT("tile_h"), P_INT("tile_w"), P_INT("patch"),
                                     P_F64("sparsity"), P_F64O("scale") };
static const RadOperandSpec oVsaSelect[] = {
    { "q", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 1 },
    { "segs", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 },
    { "sel", RAD_OPD_OUT, 0 }, { "coarse", RAD_OPD_OUT, 1 },
};
/* attn_vsa adds attn_prefix_self's int8 Q.K form at the end of the shared list: `qk` "i8" with
 * `k_i8` [M, n_head head_dim] and `k_scale` [M, n_head], quant_act_i8g's codes and scales of k at
 * group head_dim; and the fp8 P.V form: `pv` "fp8" with `v` as quant_cols_fp8's E4M3 codes and
 * `v_scale` [n_head head_dim] their column scales. */
static const RadParamSpec pAttnVsa[] = { P_INT("M"), P_INT("head_dim"), P_INT("tile_t"),
                                         P_INT("tile_h"), P_INT("tile_w"), P_INT("patch"),
                                         P_F64("sparsity"), P_F64O("scale"), P_STRO("qk"),
                                         P_STRO("pv") };
static const RadOperandSpec oAttnVsa[] = {
    { "q", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 0 },
    { "segs", RAD_OPD_IN, 0 }, { "geom", RAD_OPD_IN, 0 }, { "sel", RAD_OPD_IN, 0 },
    { "coarse", RAD_OPD_IN, 1 }, { "gate", RAD_OPD_IN, 1 }, { "out", RAD_OPD_OUT, 0 },
    { "k_i8", RAD_OPD_IN, 1 }, { "k_scale", RAD_OPD_IN, 1 }, { "v_scale", RAD_OPD_IN, 1 },
};

/* quant_cols_fp8: E4M3 codes with one scale per column over every row (an attention's V for its
 * fp8 P.V); the same spec as libref's. */
static const RadParamSpec pQuantCols[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oQuantCols[] = {
    { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 }, { "scale", RAD_OPD_OUT, 0 },
};

static const RadOpSchema kSchemas[] = {
    SCHEMA("rmsnorm", pRmsNorm, oRmsNorm,
           "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). `wadd` is GemmaRMSNorm's `1 +`, added in "
           "f32 after the load because forming it in bf16 first costs 0.3% per channel."),
    SCHEMA("rmsnorm_p", pLayernorm, oUnary,
           "y = x * rsqrt(mean(x^2) + eps); an rmsnorm with NO gain tensor at all. Gemma's "
           "attention normalises the value with a bare RMS (a weightless ggml_rms_norm) and the "
           "key with the k_norm gain in the same layer, which one op with a required gain cannot "
           "express."),
    SCHEMA("add", pAdd, oBinary,
           "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
           "along rows, and nothing else broadcasts at all. With `div`, y = rd(rd(a + b) / div), "
           "rd y's dtype."),
    SCHEMA("mul", pMnDtype, oBinary, "y = a * b, with the same one broadcast rule as `add`."),
    SCHEMA("mul_weight", pMulWeight, oMulWeight,
           "y = x * w, `w` a WEIGHT of one element (a scalar for the whole output) or of one row "
           "(broadcast down the rows): a learned gain on an activation. f32 product, rounded once."),
    SCHEMA("mla_attn", pMlaAttn, oMlaAttn,
           "Softmax over selected latent vectors used as both key and value; per-query page tables and visible lengths."),
    SCHEMA("mla_project", pMlaProject, oGemmNt,
           "Per-head query absorption or value expansion through the shared [head,(K|V),latent] BF16 matrix."),
    SCHEMA("dsa_score", pDsaScore, oDsaScore,
           "Per-head ReLU of FP8 dot products, then signed head gates and scales; invalid history is -inf."),
    SCHEMA("dsa_pool", pDsaPool, oDsaPool,
           "Channel-wise softmax pooling with learned position bias, BF16, normalized Hadamard, BF16, power-of-two FP8."),
    SCHEMA("dsa_query", pMnDtype, oQuantActFp8,
           "Normalized Hadamard, BF16 boundary and power-of-two E4M3 quantization per query head."),
    SCHEMA("gemm_nt_f32", pGemmNt, oGemmNt,
           "BF16 activations times F32 weights, F32 accumulation and output; preserves routing and index gate precision."),
    SCHEMA("swiglu_clamped", pSwigluClamped, oGateUp,
           "silu(min(gate,limit))*clamp(up,-limit,limit); gate is the first half."),
    SCHEMA("swiglu_clamped_quant_fp8", pSwigluClampedQuant, oGatedQuantFp8,
           "`swiglu_clamped` then `quant_act_fp8` over its product, in one pass: the codes are "
           "taken over the BF16-ROUNDED product, which `out_bf16` receives when present."),
    SCHEMA("silu_mul", pMnDtype, oGateUp,
           "y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The GATE IS THE FIRST HALF, "
            "which is the order rad-convert fuses gate_proj and up_proj in."),
    SCHEMA("gelu_mul", pMnDtype, oGateUp,
           "y[.., i] = gelu_tanh(gate_up[.., i]) * gate_up[.., n + i], the tanh approximation "
           "(gemma's gelu_pytorch_tanh), and the GATE IS THE FIRST HALF as in silu_mul. Its own "
           "op, not gelu, because gated feed-forwards are trained with the tanh form and the "
           "unary gelu is the exact erf one."),
    SCHEMA("act_mul", pActMul, oBinary,
           "y = act(a) * b, the activation (silu, gelu, gelu_tanh) rounded to y's dtype before "
           "the product: silu_mul with its two halves in two operands."),
    SCHEMA("seg_mean", pSegMean, oSegMean,
           "Row s of y is the mean of x's rows [cu_seqlens[s], cu_seqlens[s+1]), summed in f64 "
           "and rounded once; an empty segment is zeros."),
    SCHEMA("qk_norm_rope", pQkNormRope, oQkNormRope,
           "In place over a fused [q | k | v] row: each q and k head RMS-normed against its gain "
           "(plus wadd) and rounded, then rotated (neox or gptj, rotary_dim of head_dim); with "
           "v_norm each v head is RMS-normed with a unit gain and not rotated. An absent gain is "
           "a unit gain without wadd; rotary_dim 0 is the norm alone, positions then optional; "
           "f32 positions are read as the values they hold."),
    SCHEMA("pool_rows", pPoolRows, oUnary,
           "Row r of y is the mean of x's rows [r*k, r*k + k), summed in f32 and rounded, then "
           "times `mult` and rounded again."),
    SCHEMA("sigmoid", pMnDtype, oUnary, "y = 1 / (1 + exp(-x))."),
    SCHEMA("relu", pMnDtype, oUnary, "y = max(x, 0)."),
    SCHEMA("silu", pMnDtype, oUnary, "y = x * sigmoid(x)."),
    SCHEMA("gelu", pMnDtype, oUnary,
           "y = 0.5x(1 + erf(x/sqrt(2))). The exact erf form, not the tanh approximation."),
    SCHEMA("clamp", pMnDtype, oClamp,
           "y = min(max(x, lo), hi), lo and hi WEIGHTS of one element each: a clipped linear's range."),
    SCHEMA("conv2d", pConv2d, oConv2d,
           "A 2-D convolution over (time, width), channels innermost, w [cout, cin, kh, kw] "
           "([cout, cin, kw, kh] with t_last). Each cu_seqlens segment (rows cu/cu_div) is "
           "convolved on its own with zero padding at its edges; its output rows start at "
           "cu/(cu_div*stride). With `valid`, input row t reads as zeros unless valid[t * "
           "valid_stride] != 0. f32 sum plus the optional bias `b`, one rounding; then `act` "
           "(none, gelu) rounded."),
    SCHEMA("conv2d_unfold", pConv2dUnfold, oConv2dUnfold,
           "conv2d's receptive fields as rows: row (to, wo) holds, per input channel and kernel "
           "tap, what conv2d multiplies that tap's weight by (zero in the padding and on masked "
           "rows), its columns in the weight's order -- (ci, i, j), or (ci, j, i) with t_last -- so "
           "a GEMM against the weight as [cout, cin*kh*kw] is the convolution."),
    SCHEMA("conv1d_dw", pConv1dDw, oConv1dDw,
           "Causal depthwise convolution over time, w [C, 1, k], rows before the segment's start "
           "(cu/cu_div) reading as zeros. f32 sum, one rounding."),
    SCHEMA("rel_sinusoid", pRelSin, oRelSin,
           "The sinusoidal relative-position table: row d is sin(d * inv[i]) then cos(d * inv[i]), "
           "inv[i] = exp(i * -log(max_timescale) / max(n/2 - 1, 1)), in f32, rounded once."),
    SCHEMA("seg_sinusoid", pSegSin, oSegSin,
           "Row r of segment s (rows cu/cu_div) takes position p = (r - cu[s]/cu_div) mod period: "
           "sin(p * inv[i]) then cos(p * inv[i]), inv as rel_sinusoid's, in f32, rounded once."),
    SCHEMA("attn_local_rel", pAttnLocalRel, oAttnLocalRel,
           "Local attention with a relative-position term and a softcap (Gemma 4's audio): query i "
           "of a segment sees keys j with 0 <= i - j <= left; qf = q * head_dim^-0.5/ln2 * "
           "softplus(pds); logit = qf.k * ln(1+e)/ln2 + qf.relk[i-j], softcapped; softmax; .v."),
    SCHEMA("fill", pFill, oFill,
           "y = value, every element of an [M, n] extent (default 0). The op a rank issues where "
           "it computes nothing and a reader downstream -- an all-reduce, a residual -- still "
           "expects a defined buffer: a wire shared across layers cannot be zeroed once a step."),
    SCHEMA("logit_softcap", pMnCap, oUnary,
           "y = cap * tanh(x / cap); the final-logit softcap. `cap` is required and > 0."),
    SCHEMA("scale", pMnDtype, oScale,
           "y = x * s, s a [1] weight broadcast over every element; the per-tensor scalar "
           "(per-layer scalar). `mul` broadcasts a row, `scale_rows` a column-per-row."),

    SCHEMA("cast", pCast, oUnary,
           "y = x, converted. `from` and `to` are for the selector; the bytes are described by the "
           "tensors' own dtypes and those are what the kernel reads."),
    SCHEMA("gather_rows", pMnDtype, oGather,
           "y[i, :] = x[idx[i], :]. A negative index writes a zero row -- the padding convention a "
           "step with fewer sampled rows than the buffer's extent leaves behind."),
    SCHEMA("scatter_rows", pMnDtype, oScatterRows,
           "x[idx[i], :] = v[i, :]. The mirror of gather_rows, with `x` INOUT because the write is "
           "PARTIAL -- rows no index names keep what they held, which is what makes it usable "
           "against a pool other sequences are in. A negative index writes NOTHING rather than a "
           "zero row: the destination is live memory and a zero fill there is data loss, not "
           "padding. Indices must be DISTINCT (RAD_OPD_F_IDX_UNIQUE); two rows landing on one "
           "destination has no defined winner on a parallel machine."),
    SCHEMA("rope", pRope, oRope,
           "rotary embedding in place over the Q and K heads of a fused projection; the V heads "
           "that follow are untouched. The multi-component modes (`mrope`, `imrope`, `axial`) "
           "take [C, M] component-major positions and `sections`; with [M] positions every "
           "component is that one, so mrope and imrope over a text batch are neox exactly."),
    SCHEMA("layernorm", pLayernorm, oLayernorm,
           "y = (x - mean) * rsqrt(var + eps) * w + b, biased variance, two passes. `b` optional."),
    SCHEMA("grid_embed", pGridEmbed, oGridEmbed,
           "x += a side x side table resampled bilinearly (corners aligned, taps clamped) to each "
           "row's grid; coord is [4, M] row, column, height, width. The taps are summed in f32 "
           "and rounded before the add."),
    SCHEMA("grid_embed_aa", pGridEmbed, oGridEmbed,
           "x += the side x side table resampled as torch's F.interpolate(bilinear, "
           "align_corners=False, antialias=True) resamples it to the row's grid -- SigLIP2's "
           "position embedding: per axis a triangle filter on half-pixel centres, widened by the "
           "scale when shrinking, its in-table taps normalised. Separable, f32, rounded once "
           "before the add; coordinates clamp into their grid."),
    SCHEMA("gemm_nt_bias", pGemmBias, oGemmBias,
           "y = res + act(a @ b^T + bias), rounded to bf16 after the biased product, after the "
           "activation and after the residual -- the unfused chain's three tensors."),
    SCHEMA("kv_store", pKvStore, oKvStore,
           "scatter k and v into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim] -- K at "
           "[.., d] and V at [.., head_dim + d]. A negative slot stores nothing."),
    SCHEMA("embed_lookup", pEmbedM, oEmbed,
           "the embedding gather. A negative token id writes a zero row, and so does one outside "
           "[vocab_offset, vocab_offset + n_vocab) -- which is what makes the table shardable "
           "across the vocabulary, the symmetry lm_head already had. Ids stay GLOBAL; summing the "
           "ranks reconstructs x, so the caller follows this with an exact all_reduce."),
    SCHEMA("embed_lookup_q", pEmbed, oEmbedQ,
           "the same gather from a table quantised with ONE scale for all of it: x[m] = "
           "wte[id] * scale[0], an absent scale being one. Padding and bounds behave exactly as "
           "embed_lookup's. `ahead` is the next prompt's ids, for a reader that fetches rows ahead "
           "of the step; a kernel that reads its table in place ignores it."),
    SCHEMA("ple_gate", pPleGate, oPleGate,
           "PLE's gate. K = grouped_rmsnorm(k) * (w_key + wadd) and Q likewise over `q`, one rsqrt "
           "per n-wide stream and one gain across all hc*n; s[c] = sum_d K[c][d]*Q[c][d] / sqrt(n); "
           "then the SIGNED square root with the sign taken from the ORIGINAL, so a dot of exactly "
           "zero stays zero; then gv[c] = sigmoid(s[c]) * v, one v row gated per stream. `gvn` is "
           "grouped_rmsnorm(gv) * (w_conv + wadd) over the bytes gv was narrowed to. Both outputs "
           "leave: the convolution reads the normed copy and the residual add the un-normed one."),
    SCHEMA("ple_conv", pPleConv, oPleConv,
           "y[t] = resid[t] + silu(sum_j w[c][j] * x[t - (width-1-j)*dilation]), a DEPTHWISE "
           "CAUSAL convolution whose taps are `dilation` apart -- at width 4 and dilation 3 they "
           "are 9, 6, 3 and 0 timesteps back, so the history a sequence carries is "
           "(width-1)*dilation and not width-1. The state is [n_slots, n, state_len] with slot i "
           "holding the value at relative position i - (width-1)*dilation, and `has_init` 0 means "
           "that window is ZEROS rather than whatever the slot last held. `num_accepted` shifts "
           "the READ offset within a deeper window and marks a decode step: the window is then "
           "rewritten as gdn_conv_update rewrites its own -- the history read shifted down by one, "
           "then every input of the step -- so a next step that kept k tokens reads at k - 1. "
           "Absent, every token commits: the read is at offset zero and the last "
           "(width-1)*dilation inputs land at offset zero, which is the prefill chunk. "
           "`resid` absent drops the add and leaves the plain convolution."),
    SCHEMA("causal_conv1d", pCausalConv, oCausalConv,
           "y[t] = sum_j w[c][j] * x[t - (width-1-j)], ple_conv at dilation one with no "
           "activation and no residual: LFM2's short convolution, whose gates the caller applies. "
           "The window, `has_init`, `cache_idx` and `num_accepted` are ple_conv's."),
    SCHEMA("ngram_ids", pNgramIds, oNgramIds,
           "ids[m][h] = remainder(hash of the n-gram ending at m, vocab_sizes[h]) + offsets[h]. "
           "Heads are `ngram - 1` equal blocks, block b using the (b+2)-gram. A shift that would "
           "cross an EOS reads the EOS instead, and so does every deeper shift. `state` is a "
           "rolling window of the committed ids in front of the step, read at `num_accepted - 1` "
           "and rewritten in ple_conv's decode layout when `num_accepted` is present and its "
           "prefill layout when it is not; a cold window is EOS and not zero. `ahead_ids` "
           "present: `tok` holds its row count more tokens after the step's, continuing the LAST "
           "sequence, and their ids -- as the step that runs them will compute them -- go there; "
           "the window is the step's own."),
    SCHEMA("logits_gemm", pLogits, oLogits,
           "the lm_head projection. gemm_nt's arithmetic under the vocabulary edge's parameter "
           "spelling; M is one row per SAMPLED position, `gather_rows` having run first."),
    SCHEMA("sample_penalties", pPenal, oPenalties,
           "repetition, frequency and presence penalties over the row's history, llama.cpp's "
           "order. The repetition penalty is multiplicative and sign-dependent: dividing a "
           "negative logit would make it larger. A token's column is its id less vocab_off."),
    SCHEMA("sample_dry", pPenal, oDry,
           "DRY: penalise, once, each token that would extend a suffix of the history that has "
           "occurred before, by its longest such repeat -- what stops a model looping on a phrase "
           "rather than on a token. Tokens the breaker plane marks are exempt."),
    SCHEMA("sample_temp", pMVocab, oTemp,
           "per-request temperature. temp <= 0 leaves the row untouched; greedy is sample_argmax."),
    SCHEMA("sample_mask", pMVocab, oMask,
           "A SET BIT MEANS ALLOWED. Word w bit b covers token 32w + b; a cleared bit sets "
           "-infinity. The row's mask_row selects the plane row, or -1 when unconstrained."),
    SCHEMA("sample_topk", pTopk, oTopk,
           "the exact top-k candidate set, descending, ties to the lower token id. The width is "
           "the buffer's; a row's own top_k may be narrower and leaves holes."),
    SCHEMA("sample_argmax", pMVocab, oArgmax, "greedy, ties to the lower token id."),
    SCHEMA("sample_logprobs", pLogprobs, oLogprobs,
           "for a row with RAD_SP_LOGPROBS: the raw logits' max, sum of exp(l - max) and k largest "
           "(id, logit) pairs, descending, ties to the lower id. Other rows are not written."),
    SCHEMA("sample_topp", pMCand, oNarrow,
           "nucleus: keep the shortest prefix whose cumulative probability reaches top_p, always "
           "at least one. The candidates are assumed descending, as sample_topk leaves them."),
    SCHEMA("sample_minp", pMCand, oNarrow,
           "keep candidates whose probability is at least min_p times the most likely one's."),
    SCHEMA("sample_typical", pMCand, oNarrow,
           "locally typical: a threshold on |surprisal - entropy|, widened until the mass reaches "
           "typical_p. NOT a prefix -- that order and the probability order disagree."),
    SCHEMA("sample_xtc", pMCand, oNarrow,
           "exclude top choices: with probability xtc_probability, DROP every candidate at or "
           "above xtc_threshold except the least likely of them. A no-op unless two clear it."),
    SCHEMA("sample_pick", pMCand, oPick,
           "the inverse-CDF draw over the candidate set, from the specified splitmix64 stream."),
    SCHEMA("sample_merge_topk", pMCand, oMergeTopk,
           "the vocab-parallel merge: the k largest of the all-gathered per-rank candidate sets, "
           "in the same descending form the single-rank path produces."),
    SCHEMA("attn_paged", pAttnPaged, oAttnPaged,
           "paged causal attention, varlen. The ranged parameter is q_len, not M: one query row "
           "at decode, up to 64/gqa at a speculative verify, thousands in a prefill chunk, and "
           "libr4d has a different kernel for each band."),
    /* attn_paged under its own name, for q read at its own row pitch. The windowed kernels
     * (r4d_attn_paged_window.hip) read q in place in a fused [q|k|v] projection; the other
     * attn_paged kernels read q dense and refuse a slice (RAD_E_STRIDE). Under this name the op
     * resolves to a kernel that takes the slice or to nothing, and in the latter case the
     * architecture lifts q into a dense block for attn_paged. */
    SCHEMA("attn_paged_qrow", pAttnPaged, oAttnPaged,
           "attn_paged, parameter for parameter and value for value, with q read at its own row "
           "pitch: a column slice of a wider row -- the q part of a fused [q|k|v] projection -- read "
           "where it lies rather than lifted into a dense block first."),
    SCHEMA("attn_paged_gate_quant", pAttnPagedGq, oAttnPagedGq,
           "attn_paged followed by gate_quant_fp8 over its output, in the split-KV merge: each "
           "merged (token, head) row is gated, rounded, written to `out` and quantised to E4M3 "
           "with one scale per 128 columns into `q8` / `q8_scale`."),
    SCHEMA("attn_dense", pAttnDense, oAttnDense,
           "dense attention over cu_seqlens segments (rows cu/cu_div), with attn_paged's window "
           "rule over a segment's positions and optional blocks of `block` rows that attend "
           "within themselves."),
    SCHEMA("attn_cross", pAttnCross, oAttnCross,
           "attn_dense with each entry's keys a run of their own: queries rows [cu[s], cu[s+1]), "
           "keys and values rows [cu_k[s], cu_k[s+1]); `causal` counts from each run's start."),
    SCHEMA("attn_na3d", pAttnNa3d, oAttnNa3d,
           "3-D neighborhood attention over the voxels of geom = (T, H, W): each query sees the "
           "kt x kh x kw window centred on it, shifted inward at the border."),
    SCHEMA("qk_norm_rope_gate", pQkNormRopeGate, oQkNormRopeGate,
           "split [q|gate] per head, QK RMS norm over the head, partial NeoX RoPE and the gate "
           "copy, in one pass. Not docs/OPS.md's qk_norm_rope: different operands, and a "
           "precomputed cos/sin table instead of theta. `n_head` 0 is the K-ONLY FORM -- no "
           "`q_gate`, no `q_w`, no `q_out` -- for an input that is already one plain head per "
           "`head_dim` columns with nothing interleaved. The QSA indexer's query heads are that "
           "shape, and it is how its `heads` norms plus a rope become one launch."),
    SCHEMA("qk_norm_rope_gate_kv_store", pQkNormRopeGateKv, oQkNormRopeGateKv,
           "qk_norm_rope_gate followed by kv_store, in one launch: each k head's block also writes "
           "its rotated k and its v into the paged cache at the token's slot. The `k` kv_store "
           "reads is `k_out`, so the tail adds only `v`, `slot_mapping` and `kv_cache`."),
    SCHEMA("rope_table", pRopeTable, oRopeTable,
           "the cos/sin `qk_norm_rope_gate` reads, written at each position's OWN row of a "
           "[max_ctx, rot] plane: rot values a row, cos for the first half and sin for the second. "
           "Only the rows a step names are touched, so a decode step fills kilobytes; one launch a "
           "step serves every layer. The angles are `rope`'s exactly -- the two must agree or the "
           "fusion is a different model."),
    SCHEMA("gdn_conv_prep", pGdnConvPrep, oGdnConvPrep,
           "gdn prefill preamble: causal conv with its state cache, qkv split, qk l2 norm, "
           "gating and the per-chunk gate cumsum."),
    SCHEMA("gdn_conv_update", pGdnConvUpdate, oGdnConvUpdate,
           "the same convolution for a decode step: a rolling speculative window over the conv "
           "state cache, read at the slot the last ACCEPTED token left."),
    SCHEMA("gdn_kkt_solve", pGdnKktSolve, oGdnKktSolve,
           "A = (I + strict_lower(diag(beta) K K^T e^dg))^-1 per chunk."),
    SCHEMA("gdn_chunk_scan", pGdnChunkScan, oGdnChunkScan,
           "gated delta net chunked scan: WY recompute, state recurrence and output."),
    SCHEMA("gdn_recurrent_update", pGdnRecurrentUpdate, oGdnRecurrentUpdate,
           "gdn decode recurrent update against the paged state cache, one state written per "
           "candidate token."),
    SCHEMA("gdn_conv_recurrent_update", pGdnConvRecurrentUpdate, oGdnConvRecurrentUpdate,
           "gdn_conv_update followed by gdn_recurrent_update, with the convolution computed in "
           "the recurrent update's prologue: q, k and v never reach memory. Byte-identical to "
           "the two ops it replaces."),
    SCHEMA("hc_enter", pHcPlain, oHcEnter,
           "h[s] = x for each of the hc streams -- the wide residual stream's first value."),
    SCHEMA("mhc_pre", pMhcPre, oMhcPre,
           "Whole-residual RMS-scaled projection, sigmoid gates and Sinkhorn mixing; coefficients f32."),
    SCHEMA("mhc_pre_partial", pMhcPrePartial, oMhcPrePartial,
           "mhc_pre's projection over chunks [first, first+chunks) of `chunk` residual columns: "
           "per chunk the hc*(hc+2) coefficient partials and the sum of squares, zero-padded to "
           "a multiple of four floats; tensor-parallel ranks each take their chunks."),
    SCHEMA("mhc_pre_finish", pMhcPreFinish, oMhcPreFinish,
           "mhc_pre from every chunk's partials: chunk sums, gates, Sinkhorn mixing and the "
           "collapse of h; the pair is mhc_pre."),
    SCHEMA("mhc_post", pHcPlain, oMhcPost,
           "out[dst,i] = post[dst]*x[i] + sum_src comb[src,dst]*h[src,i]; separate output."),
    SCHEMA("mhc_contract", pHcPlain, oMhcContract,
           "Mean over the hc residual streams, rounded once to dtype."),
    SCHEMA("kda_update", pKdaUpdate, oKdaUpdate,
           "Bounded per-channel KDA recurrence. verify stores candidate states; prefill commits final state. "
           "Absolute cu offsets, accepted-prefix read slot, optional cold initialization, no output norm."),
    SCHEMA("hc_read", pHcRead, oHcRead,
           "the read half of a gated residual: a grouped rmsnorm, a low-rank sigmoid gate, and "
           "the mean of the gated streams -- plus the per-branch write gains and, optionally, "
           "the block-scaled fp8 form of the block input and its Hadamard-rotated twin."),
    SCHEMA("hc_project", pHcProject, oHcProject,
           "the control vector's projection (llama.cpp's `--cvec-mode project`): for each of the "
           "hc streams of a row whose `on` flag is 1, h[s] -= scale * (h[s] . v) * v, v that "
           "layer's unit direction -- the same v for every stream -- and scale its stored norm. A "
           "row whose flag is 0 is neither read nor written, so an unprojected request's stream is "
           "the stock model's."),
    SCHEMA("hc_write", pHcPlain, oHcWrite,
           "the write half: h[s] += inj[s] * y, into the UNNORMED stream."),
    SCHEMA("mtp_enter", pMtpEnter, oMtpEnter,
           "The MTP head's entry for a wide residual stream: x[k*n+i] = fc_h[i] . "
           "grouped_norm(h)[k] + fc_e[i] . norm(e), over each of the hc sub-streams. One op "
           "because every intermediate is a different VIEW of the same bytes and this ABI narrows "
           "only dim 0 and the last dimension -- a row-pitch change is inexpressible, so the "
           "unfused spelling cannot be written at all."),
    SCHEMA("gdn_gated_rmsnorm", pGdnGatedRmsNorm, oGdnGatedRmsNorm,
           "out = rms(x) . w . act(z) over a row."),
    SCHEMA("dflash_conv", pDflashConv, oDflashConv,
           "DFlash2 grouped dynamic depthwise convolution, coefficient add and both taps in one "
           "pass."),
    SCHEMA("dflash_select", pDflashSelect, oDflashSelect,
           "DFlash2's candidate selector: the low-rank edge score between consecutive draft "
           "positions, and the greedy walk that turns a block of independent top-k sets into one "
           "path. Only the K edges leaving the previous pick are scored, because the walk is "
           "serial and the other K-1 rows of the score matrix are never read."),
    SCHEMA("all_reduce", pAllReduce, oAllReduce,
           "sum over the tensor-parallel group. `exact` separates the exact sum from the "
           "quantised wire and a caller that does not declare it gets no kernel. At 4 and 8 ranks "
           "the exact sum is one-shot up to " R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements "
           "(at 8 ranks " R4D_XSTR(R4D_AR_8RANK_ONESHOT_MAX_ELEMS) ") and two-shot above, chosen "
           "by the message's band; `hops` is accepted and no row of this plugin reads it."),
    SCHEMA("all_gather", pAllGather, oAllGather,
           "y is [world_size * numel] with rank r's contribution at offset r * numel. The bytes are "
           "moved and never summed: the vocab-parallel sampler's payload is (u32 token id, f32 "
           "logit) pairs and an id lane read as f32 is a denormal."),
    SCHEMA("gemm_nt", pGemmNt, oGemmNt,
           "C[M,N] = A[M,K] @ B[N,K]^T, unquantised."),
    SCHEMA("gemm_nt_q", pGemmNtQ, oGemmNtQ,
           "C[M,N] = A[M,K] @ dequant(Bq)[N,K]^T. `dtype` names the operand COMBINATION, so one "
           "op covers w4a16 through mxfp4a8 and an architecture plugin that changes quant changes "
           "one string."),
    SCHEMA("gemm_nt_q_bias", pGemmNtQBias, oGemmNtQBias,
           "gemm_nt_q with a per-column bias added to the f32 product before its one rounding."),
    SCHEMA("dequant_w4", pDequantW4, oDequantW4,
           "a packed 4-bit fragment-ordered weight back to row-major bf16."),
    SCHEMA("scale_rows", pScaleRows, oScaleRows,
           "y[m, i] = x[m, i] * act(s[m]): one scalar per ROW, broadcast ACROSS the row. `mul` "
           "broadcasts the other way -- one row down the rows -- and one op with both rules would "
           "be ambiguous at rows == n. `act` is \"sigmoid\" or \"none\" and applies to the SCALAR. "
           "With `add` present the form is y = bf16(x * act(s)) + add: the product is NARROWED "
           "before the sum, which is what makes the fused form bit-identical to the two ops it "
           "stands in for rather than merely close to them."),
    SCHEMA("router_sigmoid_topk", pRouterSigmoid, oRouterSigmoid,
           "Single-group sigmoid routing; correction bias selects experts, original scores weight them."),
    SCHEMA("router_sigmoid_topk_scatter", pRouterSigmoidScatter, oRouterSigmoidScatter,
           "`router_sigmoid_topk` then `moe_scatter` over its ids, in one launch, and with `gate` "
           "present the ungated shared arm's `fill` of its gate with 1; with `inverse` the sort is "
           "`moe_scatter_inv`. Every output is the bytes the separate launches write, including a "
           "row with fewer finite ranks than top_k. Bounded by the sort's single-workgroup form: "
           "M * top_k <= 1024, refused above."),
    SCHEMA("router_topk", pRouterTopk, oRouterTopk,
           "softmax over ALL n_expert logits, then the top_k of them by value with ties to the "
           "lower expert id, then -- if `norm` -- the kept weights renormalised to sum to one. The "
           "selection runs on the LOGITS: softmax is monotone, so the order is identical and the "
           "comparison is exact rather than through an exp."),
    SCHEMA("router_topk_scatter", pRouterTopkScatter, oRouterTopkScatter,
           "`router_topk` then `moe_scatter`, in one launch. Exactly those two and in that order: "
           "the sort reads `expert_ids` and nothing else, and `expert_ids` is what the selection "
           "writes, so the chain is closed and the launch between them was buying only a barrier. "
           "It exists because NEITHER HALF IS WORK -- at a decode shape each is a single "
           "256-thread workgroup ranking about a kilobyte, and each costs the one-workgroup "
           "dispatch floor. Bounded: `M * top_k` must fit the sort's single-workgroup form, and a "
           "caller past that gets a refusal rather than a silent fallback to a different cost."),
    SCHEMA("router_gemm_topk_scatter", pRouterGemmTopkScatter, oRouterGemmTopkScatter,
           "`gemm_nt` (the router, N = n_expert) then `router_topk_scatter` over its logits, in "
           "one launch -- and with `gate_b` and `gate` present, the shared expert's `gemm_nt` at "
           "N = 1 over the same input as well. The workgroup that finishes the GEMM last runs the "
           "selection and the sort. Byte-identical to the launches it stands in for: the same "
           "split-K WMMA arithmetic at the split the unfused GEMM's own law picks, and the same "
           "selection and sort. Decode only: M <= 64 and M * top_k within the sort's "
           "single-workgroup form."),
    SCHEMA("moe_scatter", pMoeScatter, oMoeScatter,
           "a STABLE counting sort of the (token, slot) pairs by expert. `sorted_tok` holds the "
           "FLATTENED index token*top_k + slot and not the bare token id, because gather has to "
           "find which of a token's routing weights applies to a row; a slot whose expert is "
           "outside [expert_base, expert_base + n_expert) is dropped and the tail is filled with "
           "-1 -- which is how EXPERT PARALLELISM is expressed: the router stays replicated and "
           "picks from every expert, each rank sorts only its own slice, and what went elsewhere "
           "contributes nothing to this rank's partial. Stable without an "
           "atomic ordering anything, so the sort -- and every row of the grouped GEMM's output -- "
           "does not depend on the scheduler. `protect` lists local experts sorted LAST, in that "
           "order, the rest closed up: `expert_offset` and `expert_count` are then in sort order, "
           "and a layer's protected experts are one trailing run a second grouped GEMM serves."),
    SCHEMA("moe_scatter_inv", pMoeScatter, oMoeScatterInv,
           "`moe_scatter`, also writing `inverse`: for every slot token*top_k + slot its row in "
           "`sorted_tok`, -1 for a dropped slot. That is the table `moe_gather` builds from "
           "`sorted_tok` before it sums, so `moe_gather_inv` reads it instead."),
    SCHEMA("moe_gather_inv", pMoeGather, oMoeGatherInv,
           "`moe_gather` reading the slot table `moe_scatter_inv` wrote instead of rebuilding it "
           "from `sorted_tok`: the same sums in ascending slot order, the same bytes."),
    SCHEMA("moe_gemm", pMoeGemm, oMoeGemm,
           "y[i] = a[row(i)] @ w[e]^T in SORTED order, e being the expert whose offset range "
           "contains i, both operands unquantised and one accumulator over K. `a_order` picks "
           "row(i): \"token\" gathers sorted_tok[i]/top_k, \"sorted\" reads i. `w` is ONE "
           "WEIGHT TABLE (RAD_OPD_WTAB) of n_expert entries, each [N, K]."),
    SCHEMA("moe_gemm_q", pMoeGemmQ, oMoeGemmQ,
           "y[i] = a[row(i)] @ w[e]^T in SORTED order, e being the expert whose offset range "
           "contains i, with both operands E4M3 and a scale per 128 along K. `a_order` picks "
           "row(i): \"token\" gathers sorted_tok[i]/top_k, \"sorted\" reads i. The weights are "
           "WEIGHT TABLES (RAD_OPD_WTAB), two by parity: `w`/`w_scale` hold the even experts at "
           "[N, K] and `w_odd`/`w_odd_scale` the odd ones at [N_odd, K_odd], expert e at entry "
           "e/2. The output is max(N, N_odd) wide in `parts` stacked parts, and a class narrower "
           "than that fills each part's leading columns and zeros the rest."),
    SCHEMA("moe_gemm_q_mod4", pMoeGemmQMod4, oMoeGemmQMod4,
           "moe_gemm_q over FOUR weight-table classes by the expert's number mod 4: `w`/`w_scale` "
           "hold class 0 at [N, K] and `w_q`/`w_q_scale` class q at [N_q, K_q], expert e at entry "
           "e/4 of class e%4. The output is as wide as the widest class in `parts` stacked parts, "
           "and a narrower class fills each part's leading columns and zeros the rest."),
    SCHEMA("moe_gather", pMoeGather, oMoeGather,
           "scatters the sorted expert output back to token order and weights it. The sum over a "
           "token's slots runs in ASCENDING SLOT order, which is what makes it reproducible -- "
           "walking the expert-sorted rows instead sums in expert-id order and the two differ in "
           "the last bits."),
    SCHEMA("row_topk", pRowTopk, oRowTopk,
           "exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every "
           "column reported so a rank's slice reports global ids, and `pairs` is the [M, R, 2] "
           "(int32 id, float value) plane an all_gather moves; a slot the row could not fill "
           "reports id -1 there and column 0 in `idx`, which is the unsharded contract unchanged."),
    SCHEMA("row_topk_merge", pRowTopkMerge, oRowTopkMerge,
           "the R largest of world_size all-gathered per-rank candidate sets, in the same "
           "descending (idx, val) form the single-rank top-R produces. `gathered` is "
           "[M, world_size * R, 2] pairs with rank r's at [r*R, (r+1)*R); id -1 is a pad and is "
           "dropped, and ties break on the LOWER id so the merged answer equals a top-R over the "
           "unsharded plane."),
    SCHEMA("logit_rerank", pLogitRerank, oLogitRerank,
           "R candidate ids a row rescored EXACTLY against the full-precision head and re-sorted "
           "on those values -- the second half of a coarse top-R. `idx_in` carries global ids as "
           "row_topk reports them, `vocab_off` is this rank's first column, and an id outside its "
           "rows becomes a pad. Ties break on the LOWER id, the rule row_topk and row_topk_merge "
           "use, so a tie inside a shard and one between shards resolve alike."),
    SCHEMA("rmsnorm_quant_fp8", pRmsNormQuantFp8, oRmsNormQuantFp8,
           "RMS norm and the fp8 activation quantiser in one pass over the row. The bf16 norm "
           "output the unfused pair wrote is still written, and the codes are taken over it."),
    SCHEMA("rmsnorm_had_quant_i8g", pRmsNormQuantFp8, oRmsNormQuantFp8,
           "rmsnorm_quant_fp8 with had_quant_act_i8g's quantiser on its pass: a `group`-wide "
           "Hadamard of the bf16 norm output, then int8 codes per (row, `group`). `out_bf16` "
           "receives the UNROTATED norm output. Pairs only with an i8 weight stored rotated "
           "(dtype i8a8h); byte-identical to the norm and had_quant_act_i8g."),
    SCHEMA("rmsnorm_quant_i8", pRmsNormQuantI8, oRmsNormQuantI8,
           "RMS norm and the per-row int8 quantiser the w4a8 skinny GEMM reads, in one pass; with "
           "a residual the norm is of residual + x (+ a post-norm of x, x a scalar), stored back."),
    SCHEMA("add_rmsnorm", pAddRmsNorm, oAddRmsNorm,
           "x = x + a, then y = rmsnorm(x) * (w + wadd) * mult over the x written: a residual add "
           "and the next norm in one pass, byte-identical to add then rmsnorm. The norm is taken "
           "over the bf16 sum it stores; rmsnorm_add normalises the unrounded one."),
    SCHEMA("qkv_norm_rope", pQkvNormRope, oQkvNormRope,
           "per-head q and k RMS norms, the weightless v norm and the rotary of q and k over separate "
           "q, k and v projections, in one dispatch; v from the raw k under v_from_k. n_head 0 "
           "leaves q unread (the k and v heads alone)."),
    SCHEMA("qkv_norm_rope_kv_store", pQkvNormRopeKv, oQkvNormRopeKv,
           "qkv_norm_rope, then kv_store of the finished k and v heads into the paged cache, in one "
           "launch."),
    SCHEMA("gelu_mul_quant_i8", pGeluMulQuantI8, oGeluMulQuantI8,
           "gelu(gate) * up (tanh form) and the per-row int8 quantiser the w4a8 skinny GEMM reads, in "
           "one pass."),
    SCHEMA("ar_rmsnorm_quant_i8", pArRmsNormQuantI8, oArRmsNormQuantI8,
           "the two- or four-rank exact all-reduce of x and rmsnorm_quant_i8 after it, in one kernel."),
    SCHEMA("gated_quant_fp8", pGatedQuantFp8, oGatedQuantFp8,
           "silu(a)*b or sigmoid(a)*b, then the same quantiser, in one pass."),
    SCHEMA("gated_had_quant_fp8", pGatedQuantFp8, oGatedQuantFp8,
           "the same, with a `group`-wide Hadamard between the product and the absmax. `out_bf16` "
           "still receives the UNROTATED product -- it exists for a bf16 consumer of the gate, and "
           "the rotation is a property of the quantised pair alone. Pairs with a weight stored at "
           "dtype w4a8h; substituting it for gated_quant_fp8, or the reverse, produces a model "
           "that runs and is wrong."),
    SCHEMA("gated_had_quant_i8g", pGatedQuantFp8, oGatedQuantFp8,
           "gated_had_quant_fp8 at int8 codes: silu(a)*b or sigmoid(a)*b, a `group`-wide Hadamard "
           "and had_quant_act_i8g's quantiser, in one pass. `out_bf16` receives the UNROTATED "
           "product. Pairs only with an i8 weight stored rotated (dtype i8a8h); byte-identical "
           "to the product and had_quant_act_i8g."),
    SCHEMA("gemm_nt_q_gated", pGemmNtQGated, oGemmNtQGated,
           "gemm_nt_q over a [2n, K] gate_up plane with gated_quant_fp8's pass folded into the "
           "epilogue: the outputs are the quantiser's E4M3 codes, its scale per (row, `group` "
           "columns) and optionally the bf16 product, and no [M, 2n] tensor is ever written. "
           "Byte-identical to the two ops it replaces, which is what makes it substitutable "
           "rather than a second numerical path."),
    SCHEMA("gate_quant_fp8", pGateQuantFp8, oGateQuantFp8,
           "the same, with the gate in its own operand rather than packed beside x."),
    SCHEMA("quant_act_fp8", pQuantActFp8, oQuantActFp8,
           "per-(row, `group`-column) symmetric E4M3 quantisation of a token-major activation, in "
           "PLAIN ROW-MAJOR order rather than quant_act_i8's fragment order. The scale is "
           "amax/448 -- a DEQUANT multiplier, the same sense as the checkpoint's "
           "weight_scale_inv, so the fp8 GEMMs multiply both scales and never divide."),
    SCHEMA("quant_act_i8g", pQuantActFp8, oQuantActFp8,
           "quant_act_fp8 at INT8 codes: per-(row, `group`-column) symmetric int8, PLAIN ROW-MAJOR, "
           "the scale amax/127 and a code the value over it rounded to nearest-even. What the int8 "
           "trunk GEMMs (dtype i8a8) read."),
    SCHEMA("had_quant_act_i8g", pQuantActFp8, oQuantActFp8,
           "quant_act_i8g with a `group`-wide Hadamard along K first; pairs with an i8 weight stored "
           "rotated, its scale carrying the residual factor of `group`."),
    SCHEMA("givens_rotate", pGivensRotate, oGivensRotate,
           "ParoQuant's rotation without a quantiser: y = x / rscale rotated by `stages` rounds of "
           "Givens pairs, `parts` copies side by side ([M, parts * n]) in y's own dtype. For a weight "
           "whose GEMM reads a 16-bit activation. `group` is the width a pair stays inside."),
    SCHEMA("givens_dense_quant", pGivensDense, oGivensDense,
           "ParoQuant's activation quantiser with the rotation as a dense per-group matrix: y = T (x / "
           "rscale) for each of `parts` rotated copies, T the [128, 128] f16 matrix a (part, 128-column "
           "group) that its Givens stages compose to, then the block-scaled quantiser `out` names "
           "(\"i8\": scale amax / 127, round to nearest; \"fp8\": E4M3, amax / 448). q is [M, parts * n] "
           "and scale [M, parts * n / group]; tdense is [parts * n, 128]. The same activation as "
           "the sequential rotation, up to the f16 rounding of T."),
    SCHEMA("had_quant_act_fp8", pQuantActFp8, oQuantActFp8,
           "the same, with a `group`-wide Hadamard along K applied BEFORE the absmax. Pairs with "
           "an expert weight stored at dtype w4a8h, which carries the matching rotation and the "
           "residual factor of `group` folded into its scale -- so the consuming GEMM is the "
           "unrotated one, unchanged. Substituting it for quant_act_fp8, or the reverse, produces "
           "a model that runs and is wrong."),
    SCHEMA("qsa_work", pQsaWork, oQsaWork,
           "the blocks that became complete this step, gathered into the three operands "
           "qsa_block_key takes: `ratio` consecutive rows of `stage` a block, its destination "
           "`page`, and its first-token position. A block's raw keys come from THIS step's `k` "
           "where the position is in the chunk and from the TAIL STATE where it is not, and that "
           "is the whole reason the gather exists -- the block-key kernel sees one plane and "
           "never learns there were two sources. `state` is the tail group's cache for this "
           "layer and `sidx[s][0]` is the sequence's slot in it; the row read is `q % ring`. "
           "`bt` is the BLOCK-KEY group's table, not the attention group's. Slots past the "
           "real count get page -1, and so does a sequence with no state slot."),
    SCHEMA("qsa_tail_store", pQsaTailStore, oQsaTailStore,
           "state[sidx[s][0]][pos[t] % ring] = k[t], for the tokens that have not completed a "
           "block yet. THE TAIL IS A LINEAR KV STATE, not a plane indexed by the batch row: the "
           "row is step-local, so a request that outlives an earlier one inherits its keys. `cu` "
           "is cu_seqlens and is how a token finds its SEQUENCE -- the batch's other per-token "
           "index is a flat KV slot and says nothing about which. ONLY THE LAST `ring` TOKENS OF "
           "A SEQUENCE'S CHUNK WRITE, which is not an optimisation: positions p and p + ring "
           "share a row, so writing every token would be two workgroups racing for one. THE RING "
           "IS `ratio + n_spec`: a rejected draft restarts the next step at a position this one "
           "has already passed, and a ring of `ratio` has overwritten the row it still owes. "
           "`state` is INOUT -- rows nothing targets keep what they held, which is the whole "
           "point across steps."),
    SCHEMA("qsa_select", pQsaSelect, oQsaSelect,
           "the `topk` highest-scoring blocks of a query's history, as a PAGE-ID table the paged "
           "attention reads directly. The PARTIAL page -- `bt[nc]`, the tokens after the last "
           "complete block, which the query itself lives in -- is appended LAST and is never a "
           "candidate: last is what makes the attention's own causal bound mask inside it and "
           "nowhere else, so the effective query position is (nsel-1)*ratio + pos%ratio. Ties "
           "break to the LOWER block index through a prefix count, never through atomic order. "
           "The table is in BLOCK ORDER (the partial page last), which is the order the attention "
           "sums in. `selb` repeats the entries as block indices: `sel` holds page ids and a page "
           "id does not sort by block, so a table shared by a query tile needs both. `signed` 1 "
           "ranks signed finite scores; absent or 0 ranks max(score, 0)."),
    SCHEMA("qsa_score", pQsaScore, oQsaScore,
           "score[m][b] = (sum over heads of relu(q[m][h] . bk[bt[m][b]])) / sqrt(n). THE RELU IS "
           "PER (BLOCK, HEAD) AND THE SUM IS OVER HEADS, in that order -- four independent votes, "
           "each of which can only argue FOR a block; summing first would let one head's large "
           "negative cancel another's positive. `bt` holds PAGE IDS, and a negative entry scores "
           "-inf rather than zero, because a real score can be zero and padding must not outrank "
           "a block the model chose to ignore."),
    SCHEMA("qsa_block_key", pQsaBlockKey, oQsaBlockKey,
           "the QSA indexer's compressed key: out[b] = rope(rmsnorm(mean of the `ratio` raw keys "
           "of block b), at the position of the block's FIRST token). The mean is accumulated in "
           "f32 and ROUNDED TO THE ACTIVATION DTYPE before the norm, which is what the reference "
           "does and is a different number from carrying f32 through. `M` is BLOCKS and `k` is "
           "`ratio` times taller; `pos0` is the first block's first-token position and the rest "
           "tile from it."),
    SCHEMA("gram_accum", pGramAccum, oGramAccum,
           "h[k1][k2] += sum over rows of dequant(a)[m][k1] * dequant(a)[m][k2] -- the second "
           "moment of a layer's real input, which is the Hessian GPTQ's error feedback needs. "
           "`h` is INOUT: a caller zeroes it once and drives a whole calibration corpus through "
           "the graph. Not gemm_nt with a transposed operand: that op contracts the LAST axis of "
           "both sides and this one contracts the FIRST."),
    SCHEMA("sample_chain", pSampleChain, oSampleChain,
           "the whole vLLM sampler chain in one workgroup a position -- temperature, softmax, "
           "top_k, top_p, min_p, renormalise, draw -- plus the exclusive draw and the accept "
           "probability a greedy-draft rejection-sampling verify needs. Not docs/OPS.md's eight-op "
           "chain: one launch over an L2-resident row instead of eight over a 248K vocabulary."),
    SCHEMA("quant_act_i8", pQuantAct, oQuantAct,
           "per-row symmetric int8 quantisation, written in the byte order the r4d GEMMs' A "
           "fragment reads. A third output selects the form that also emits the per-group row "
           "sums an asymmetric weight grid needs."),
    SCHEMA("had_quant_act_i8", pHadQuantAct, oHadQuantAct,
           "block Walsh-Hadamard along the row, then the same quantiser. The rotation is what "
           "makes one scale per row viable at all."),
    SCHEMA("rmsnorm_had_quant_i8", pRmsNormHadQuant, oRmsNormHadQuant,
           "residual add, RMS norm, rotation and int8 quantisation in one pass over the row."),
    SCHEMA("gated_had_quant_i8", pGatedHadQuant, oGatedHadQuant,
           "silu(a)*b or sigmoid(a)*b, then the rotation and the quantiser."),
    SCHEMA("gdn_gated_norm_had_quant_i8", pGdnGatedNormHadQuant, oGdnGatedNormHadQuant,
           "the GDN gated RMS norm over 128-wide heads, then a rotation and int8 quantisation "
           "over the flattened row."),
    SCHEMA("ar_ln_had_quant_i8", pArLnHadQuant, oArLnHadQuant,
           "the two-rank all-reduce and the residual add, RMS norm, rotation and quantisation "
           "that always follow it, in one kernel."),

    SCHEMA("ar_rmsnorm_post_quant_fp8", pArRmsNormQuantFp8, oArRmsNormPostQuantFp8,
           "ar_rmsnorm_quant_fp8 with a post-norm of the reduced sum against w_post before the "
           "residual add (a sandwich-norm junction), exact wire only, byte-identical to all_reduce, "
           "rmsnorm, rmsnorm_quant_fp8."),
    SCHEMA("ar_rmsnorm_quant_fp8", pArRmsNormQuantFp8, oArRmsNormQuantFp8,
           "the two-rank all-reduce and the residual add, RMS norm and block-scaled fp8 "
           "quantisation that always follow it, in one kernel. Both wires."),
    SCHEMA("ar_rmsnorm_had_quant_i8g", pArRmsNormQuantFp8, oArRmsNormQuantFp8,
           "the two-rank all-reduce of x and rmsnorm_had_quant_i8g after it, in one kernel; "
           "either wire, as ar_rmsnorm_quant_fp8."),
    SCHEMA("ar_hc_write", pArHcWrite, oArHcWrite,
           "the all-reduce and the gated residual's write that follows it, in one kernel. Both "
           "wires at two ranks, the exact one at four."),
    SCHEMA("ar_gather_hc_write", pArGatherHcWrite, oArGatherHcWrite,
           "moe_gather, then ar_hc_write, in one kernel: the routed experts' rows summed back to "
           "token order (with the shared arm folded when present) ARE the all-reduce's message, "
           "and the gated residual's write follows the reduction. Both wires at two ranks, the "
           "exact one at four."),
    SCHEMA("pair_argmax", pPairArgmax, oPairArgmax,
           "idx[m] = the id of the largest value among pairs[m, 0..n), each pair an int32 id in "
           "the first f32 slot and its value in the second (row_topk's `pairs`, all-gathered): ties "
           "to the lower id, a negative id skipped, -1 and -inf where nothing is valid. "
           "row_topk_merge at R = 1, written one element a row at the output's own pitch."),
    SCHEMA("norm_add_norm", pNormJoin, oNormAddNorm,
           "t = rmsnorm(a) * w_a; x = x + t; y = rmsnorm(x) * w_y, over [M, n] rows; every value "
           "the unfused rmsnorm, add, rmsnorm chain stores is rounded to the activation dtype at "
           "the same point. A post-norm, the residual add and the next pre-norm (Gemma's "
           "sandwich)."),
    SCHEMA("norm_add_norm_route", pNormAddNormRoute, oNormAddNormRoute,
           "norm_add_norm, then from the same rmsnorm of the x it writes r = rmsnorm(x) * w_r * "
           "mult and the E4M3 codes q of rmsnorm(x) * w_e with an f32 scale qs per (row, `group` "
           "columns): rmsnorm_quant_fp8's codes, taken over the bf16-rounded norm. A MoE layer's "
           "router input and its experts' fp8 input, written by the attention join that already "
           "holds the norm."),
    SCHEMA("join_add_scale", pNormJoin, oJoinAddScale,
           "x = (x + rmsnorm(rmsnorm(d) * w_d + rmsnorm(r) * w_r) * w_j) * s, over [M, n] rows, s "
           "one element; and with w_n and y, y = rmsnorm(x) * w_n over the x written. Rounding "
           "where the unfused chain rounds. Two arms' post-norms, their sum, its norm, the residual "
           "add, a per-layer scalar and the next layer's input norm (Gemma 4's feed-forward "
           "join)."),
    SCHEMA("expert_scale", pExpertScale, oExpertScale,
           "ew[t, k] = ew[t, k] * w_e[ids[t, k]] over [M, top_k], w_e an [n_expert] table, an id "
           "outside it reading 0; one product rounded once, so the embed_lookup + mul pair's value "
           "bit for bit. Gemma 4's per_expert_scale on the renormalised routing weights."),
    SCHEMA("lora_shrink", pLoraShrink, oLoraShrink,
           "t[m, j] = sum_k x(m, k) * lora_a[s, j, k] for each row m whose s = slot[m] is in "
           "[0, S), f32 accumulate; x(m, k) is the value gemm_nt_q uses for that element of `a`. "
           "A base row's `t` is not written; a row's bytes do not depend on M."),
    SCHEMA("lora_expand", pLoraExpand, oLoraExpand,
           "y[m, n] = y[m, n] + scale[s, n] * sum_j t[m, j] * lora_b[s, n, j] for each row m "
           "whose s = slot[m] is in [0, S), f32 accumulate, one rounding to y's dtype. A base "
           "row's `y` is neither read nor written."),
    SCHEMA("lora_expand_mm", pLoraExpand, oLoraExpand,
           "lora_expand with t rounded to bf16 and the sum over j in any order: y[m, n] = y[m, n] "
           "+ scale[s, n] * sum_j bf16(t[m, j]) * lora_b[s, n, j], f32 accumulate, one rounding "
           "to y's dtype; a base row's `y` is neither read nor written. A row's bytes may depend "
           "on M."),
    SCHEMA("ln_mod", pLnMod, oLnMod,
           "y = bf16(bf16(LN(x)) * bf16(1 + s[seq])): a layer norm with no gain or bias, then the "
           "row's sequence's modulation row of `s`."),
    SCHEMA("ln_mod_quant_fp8", pLnModQ, oLnModQ,
           "gated 1: x = bf16(x + bf16(bf16(tanh(g[seq])) * y)) first; then z = ln_mod(x) and "
           "(q, scale) = quant_act_fp8(z, group), z written when bound."),
    SCHEMA("table_add", pTableAdd, oTableAdd,
           "y[r, l n + c] = x[r, c] + t[l, c]: a shared modulation row plus each of L learned rows, "
           "f32 sum rounded once."),
    SCHEMA("chan_stats", pChanStats, oChanStats,
           "a latent through per-channel statistics held as weights: denorm rd(rd(x std) + mean), "
           "norm rd(rd(x - mean) / std); std is s, or sqrt(s + eps) with var 1."),
    SCHEMA("mix_rows", pMixRows, oMixRows,
           "y[r, c] = sum_i w[i] x[r k + i, c], i < k: a Linear(k, 1) across k consecutive rows."),
    SCHEMA("gate_add", pGateAdd, oGateAdd,
           "x = bf16(x + bf16(bf16(tanh(g[seq])) * y)): a residual through its sequence's gate; "
           "with `act` none the gate is g itself, x = bf16(x + bf16(g[seq] * y))."),
    SCHEMA("timestep_embed", pTimestep, oTimestep,
           "the sinusoidal timestep embedding, [cos | sin] of t' * exp(-ln(max_period) * i / (n/2))."),
    SCHEMA("gather_rows2", pGenMn, oGather2,
           "y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]]."),
    SCHEMA("lane_gather", pLane, oLaneGather, "y[cu[e] + k] = src[lane[e] * stride + k]."),
    SCHEMA("lane_store", pLane, oLaneStore, "dst[l * stride + k] = src[k] for every lane listed."),
    SCHEMA("randn_lanes", pLane, oRandnLanes,
           "an image's first step: its lane's rows become its seed's normal stream."),
    SCHEMA("flow_euler", pLane, oFlowEuler,
           "one Euler step of a flow-matching sampler per entry that moves a lane."),
    SCHEMA("kd_input", pLane, oKdInput,
           "a k-diffusion sampler's model input per entry that moves a lane: a first step's noise "
           "times sqrt(1 + sigma^2) into the lane, then rd(x / sqrt(sigma^2 + 1)) at its rows and "
           "its pair's."),
    SCHEMA("kd_step", pKdStep, oKdStep,
           "one k-diffusion sampler step per entry that moves a lane: the guided clean estimate "
           "from an eps or v output, then Euler, Euler ancestral or DPM++ 2M; the lanes f32."),
    SCHEMA("chan_affine", pChanAffine, oGenUnary,
           "a per-channel affine from two parameter lists of n values; `round` 0 keeps the lists "
           "and the arithmetic in f32, rounded once into y."),
    SCHEMA("conv2d_nhwc", pConvNhwc, oConvNhwc,
           "an image convolution, channels innermost, the sizes read off the operands."),
    SCHEMA("im2col_nhwc", pIm2col, oGenUnary,
           "conv2d_nhwc's receptive fields as rows: y [Ho * Wo, cin * k * k], column "
           "(ci * k + i) * k + j of row oy * Wo + ox the input conv2d_nhwc multiplies weight[o, ci, "
           "i, j] by, 0 in the padding."),
    SCHEMA("l2norm_chan", pL2Norm, oL2Norm,
           "unit-L2 per row, times sqrt(n), times gamma, then `act`: an image decoder's norm."),
    SCHEMA("d2s_add", pSpaceDepth, oSpaceDepth, "an upsampling shortcut added in."),
    SCHEMA("s2d_avg_add", pSpaceDepth, oSpaceDepth, "a downsampling shortcut added in."),
    SCHEMA("to_rgba8", pGenMn, oGenUnary, "decoded pixels in [-1, 1] to bytes."),
    SCHEMA("gelu_tanh", pGenMn, oGenUnary, "the tanh approximation of gelu."),
    SCHEMA("gelu_quick", pGenMn, oGenUnary,
           "CLIP's quick_gelu: a = rd(1.702 x), s = rd(sigmoid(a)), y = rd(x s)."),
    SCHEMA("scatter_add_rows", pGenMn, oScatterAdd, "x[idx[i]] += v[i]; a negative index adds nothing."),
    SCHEMA("attn_prefix_self", pAttnPS, oAttnPS,
           "attention over a paged prefix and the step's own rows, causal over the own rows or not. "
           "`qk` \"i8\" with `k_i8` and `k_scale` selects the INT8 Q.K form: the own keys as quant_act_i8g "
           "codes and scales at group head_dim, the query and the prefix keys quantised the same "
           "way inside the op, the score product exact in integers and P.V as in the bf16 form. "
           "`pv` \"fp8\" with `v_scale` selects the fp8 P.V form (prefix-free only): `v` as "
           "quant_cols_fp8's E4M3 codes and column scales, P as e4m3(448 p) against an exact "
           "running max, o = (sum w v_code) v_scale / (448 sum p)."),
    SCHEMA("conv1d", pConv1d, oConv1d,
           "a 1-D convolution over the sequences cu_seqlens bounds, channels innermost, zeros outside "
           "each; dilation `dil`, padding dil (k - 1) / 2 a side or, with `causal`, dil (k - 1) on "
           "the left; `res` added after the bias, then `act`."),
    SCHEMA("conv_t1d", pConvT1d, oConvT1d,
           "a transposed 1-D convolution of kernel `k` (2 stride by default, or stride), padding "
           "`pad` ((k - stride) / 2 by default, 0 the causal form): each sequence grows `stride` "
           "times."),
    SCHEMA("snake", pSnake, oSnake,
           "x + sin(alpha x)^2 / (alpha + 1e-9) per channel, or with `beta` x + (1 / (beta + "
           "1e-9)) sin(alpha x)^2; `log` reads both as logarithms."),
    SCHEMA("cfg_topk_sample", pCfgTopk, oCfgTopk,
           "a guided top-k pick a lane from `rep` branch rows, drawn with the frame's Philox stream; "
           "per-lane guidance, a repetition penalty over the lane's history and suppressed ids "
           "optional."),
    SCHEMA("rvq_embed", pRvqEmbed, oRvqEmbed,
           "a frame's embedding from code-major residual codes, the first codebook's table and the "
           "others' stacked one, times `scale`; `round` 1 rounds the f32 sum once."),
    SCHEMA("frame_mix_store", pFrameMix, oFrameMix,
           "a frame's conditioning row (softmax-weighted planes times scale) into its lane."),
    SCHEMA("end_status", pEndStatus, oEndStatus,
           "status[lane[i]] = ids[rep i] == end ? -1 : ids[rep i]."),
    SCHEMA("frame_store", pFrameStore, oFrameStore,
           "a frame's codes into its lane's history: dst[(lane n_codes + k) stride + frame - skip] "
           "= codes[k, rep i] when frame >= skip and status[lane] >= 0."),
    SCHEMA("resample_nearest", pGenMn, oGenUnary, "rows resampled nearest-neighbour."),
    SCHEMA("fourier_embed", pFourier, oFourier, "[cos | sin] of 2 pi t w for a trained w."),
    SCHEMA("win_input", pWinInput, oWinInput,
           "a denoiser's [lat | zeros | cond] rows from the window states."),
    SCHEMA("win_prep", pWinPrep, oWinPrep, "a window step's noise and overlap binding."),
    SCHEMA("win_finish", pWinFinish, oWinFinish, "a window's end and the overlap it hands on."),
    SCHEMA("chan_split", pChanSplit, oGenUnary, "channel groups to sequences."),
    SCHEMA("interleave2", pInterleave, oInterleave2, "two streams interleaved."),
    SCHEMA("iota", pIota, oIota, "y[i] = start + i * step."),
    SCHEMA("cu_repeat", pCuRepeat, oCuRepeat, "bounds of equal sequences, y[j] = j * len[0] * mult."),
    SCHEMA("int_affine", pIntAffine, oIntAffine,
           "y[j] = (src[j] < 0 ? 0 : x[src[j]]) * mul[j] + add[j] over three integer lists."),
    SCHEMA("snake_aa", pSnakeAa, oSnakeAa,
           "BigVGAN's alias-free SnakeBeta over each sequence, f32: replicate-padded, upsampled "
           "`ratio` times by `f_up`, SnakeBeta, filtered by `f_down` at stride `ratio`."),
    SCHEMA("rms_mod", pRmsMod, oRmsMod,
           "gated 1: x = bf16(x + bf16(g[seq] * y)) first (no tanh); then z = bf16(bf16(bf16(bf16("
           "x rsqrt(mean(x^2) + eps)) w) bf16(1 + scale[seq])) + shift[seq]), `s` holding "
           "[shift | scale]."),
    SCHEMA("rms_mod_quant_fp8", pRmsModQ, oRmsModQ,
           "rms_mod with the chain in bf16, then (q, scale) = quant_act_fp8(z, group); z written "
           "when bound."),
    SCHEMA("flow_x0_step", pGenMn, oFlowX0,
           "x = r x + (1 - r) (x + (1 - t) v), r = sigma_next / sigma, each step one f32 "
           "rounding."),
    SCHEMA("patch_unfold", pPatchUnfold, oPatchUnfold,
           "patch rows unfolded to voxel rows and denormalised with f32 lists."),
    SCHEMA("vtile_gather", pVtileGather, oVtileGather,
           "a video decoder's tiles out of voxel rows, each followed by its register tokens and "
           "zeros to the period."),
    SCHEMA("vtile_stitch", pVtileStitch, oVtileStitch,
           "a decoded chunk's tiles stitched into RGB bytes, blended with the previous chunk's "
           "tail."),
    SCHEMA("grid_pos3", pGridPos3, oGridPos3,
           "a video decoder's rotary positions over tiles of `period` rows, f32 [3, M]."),
    SCHEMA("conv3d_ndhwc", pConv3d, oConv3d,
           "a stride-1 `same` 3D convolution over NDHWC voxel rows of the device volume geom = "
           "(T, H, W), PyTorch's [cout, cin, kt, k, k] weight, f32 accumulation; with `res`, "
           "y = rd(rd(acc + b) + res), res may alias y."),
    SCHEMA("group_norm", pGroupNorm, oGroupNorm,
           "nn.GroupNorm over all of y's rows (biased variance, rstd in f32), then the optional "
           "modulation v (1 + mod[0:n]) + mod[n:2n] and `act` (none or silu), one rounding; "
           "stats[2g], stats[2g + 1] hold group g's mean and rstd, the rest is scratch."),
    SCHEMA("tconv_dw", pTconvDw, oTconvDw,
           "a depthwise convolution along the frames of the device volume geom = (T, H, W), "
           "frames outside zero, taps in ascending order in f32, one rounding after the bias."),
    SCHEMA("resize_bilinear", pGenMn, oResizeBil,
           "each frame resized as torch's bilinear interpolation at a size, align_corners=False, "
           "geom = (T, h, w, H, W), f32 and one rounding."),
    SCHEMA("patch_fold", pPatchUnfold, oPatchUnfold,
           "patch_unfold's inverse: voxel rows to f32 patch rows, x mul + add per channel."),
    SCHEMA("flow_renoise", pFlowRenoise, oFlowRenoise,
           "x = (1 - sigma) x + sigma z, z the (seed, stream) normal stream at element k n + c, "
           "each step one f32 rounding."),
    SCHEMA("rope_split", pRopeSplit, oRopeSplit,
           "LTX-2's split rotary over fractional coordinates: the pairs of all heads on one "
           "ladder, the first `pad` unrotated, the axes interleaved; each head's halves turned "
           "against each other in f32."),
    SCHEMA("head_gate", pHeadGate, oHeadGate,
           "x = rd(x rd(2 rd(sigmoid(g)))) for each head's values."),
    SCHEMA("ltx_guide", pLtxGuide, oLtxGuide,
           "LTX-2's sampler step: the passes' clean estimates combined (CFG, STG, modality), an "
           "optional rescale by the deviations' ratio, the velocity back and the Euler step, f32."),
    SCHEMA("pixel_norm", pPixelNorm, oPixelNorm,
           "an RMS norm over each row's n values as bf16 tensors compute it, every step rounded "
           "to y's dtype; rsqrt or div form, optionally silu; `ileave` L writes x's row t L + l "
           "to columns c L + l of y's row t."),
    SCHEMA("d2s3d", pD2s3d, oD2s3d,
           "depth to space over voxel rows, the first st - 1 frames dropped."),
    SCHEMA("s2d3d", pS2d3d, oS2d3d,
           "space to depth over voxel rows, `tpre` leading frames prepended: a packed row each "
           "box (copy) or each `group` packed channels averaged (mean); `add` adds into y."),
    SCHEMA("pix_patch", pPixPatch, oPixPatch,
           "an image [H, W C] to patch rows [H/p, (W/p) C p p], order wh or hw."),
    SCHEMA("patch_pix8", pPixPatch, oPixPatch,
           "patch rows [R, w C p p] to RGB bytes [R p, w p C] by to_rgba8's rule."),
    SCHEMA("vol_index", pVolIndex, oVolIndex,
           "a volume's voxel coordinates (t, i, j), component-major."),
    SCHEMA("tile_grid", pTileGrid, oTileGrid,
           "the tiles of a grid as diffusers cuts them (rule stride or intervals), in space and "
           "optionally time, one row (t, h, w, t0, y0, x0, k, count) a tile."),
    SCHEMA("tile_crop", pTileCrop, oTileCrop,
           "one tile's voxels out of a volume, box (t, h, w, t0, y0, x0)."),
    SCHEMA("tile_stitch", pTileStitch, oTileStitch,
           "decoded tiles to RGB bytes: diffusers' in-place blend_v then blend_h, then blend_t "
           "across time bands, each tile kept to its stride, patch_pix8's bytes."),
    SCHEMA("stft_logmel", pStftMel, oStftMel,
           "each channel's causal log-mel spectrogram from a windowed DFT basis and a mel "
           "filterbank, every stored tensor rounded to y's dtype."),
    SCHEMA("bwe_up", pBweUp, oBweUp,
           "y = clamp(rd(res + up(x)), -1, 1), x raised `ratio` times by a Hann-windowed sinc "
           "resampler."),
    SCHEMA("vsa_select", pVsa, oVsaSelect,
           "video sparse attention's selection over tiles of at most 64 rows (dense segments in "
           "chunks, the video grid in tile_t x tile_h x tile_w boxes): f32 tile means of q and k, "
           "scores scale qp.kp, and for each video query tile the k = max(1, min(ceil((1 - "
           "sparsity) V), V)) video tiles of largest score, ascending, ties to the lower tile; "
           "with `coarse`, bf16(softmax over every tile of the scores, times the v means)."),
    SCHEMA("attn_vsa", pAttnVsa, oAttnVsa,
           "block-sparse attention over vsa_select's tiles: a prefix query tile sees every tile, a "
           "video one the prefix tiles and its selected tiles; with `coarse` and `gate`, "
           "bf16(bf16(o) + coarse[tile] gate). `qk` \"i8\" with `k_i8` and `k_scale` selects the "
           "INT8 Q.K form: the keys as quant_act_i8g codes and scales at group head_dim, the query "
           "quantised the same way inside the op, a score the exact integer product times both "
           "scales, P.V as in the bf16 form. `pv` \"fp8\" with `v_scale` selects the fp8 P.V "
           "form: `v` as quant_cols_fp8's E4M3 codes and column scales, P as e4m3(448 p) against "
           "an exact running max, o = (sum w v_code) v_scale / (448 sum p)."),
    SCHEMA("quant_cols_fp8", pQuantCols, oQuantCols,
           "E4M3 codes with one scale per column over all M rows: scale = amax(|x[:, c]|) / 448 "
           "(1 for an all-zero column), y = e4m3(clamp(x / scale, +-448)), round to nearest even."),
};

/* ================================================================================ constraints
 *
 * Each row restates its entry point's admissibility rules, line for line where the keys agree;
 * the key renames are marked.
 */

/* ---- attention ------------------------------------------------------------------------------
 * `causal` is a constraint: the paged prefill kernels are causal and the vision kernel dense, and
 * neither serves the other's masking. */
static const RadConstraint cAttnPrefillFp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillBf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* q_len <= 10 is q_len*gqa <= 64 at the gqa 6 this kernel is compiled for. It is also the bucket
 * boundary splitting `attn_paged` into decode and prefill bands (spec §2.2). */
static const RadConstraint cAttnDecodeFp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
/* The fp8_v4 cache (E4M3 K, int4 V in groups of 32 with f16 scales) and the k6_v4 cache (int6 K,
 * int4 V); the row layouts are RAD_KVQ_*'s (abi/rad_builder.h). */
static const RadConstraint cAttnPrefillK6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cAttnDecodeK6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cAttnPrefillFp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cAttnDecodeFp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cAttnDecodeBf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The DFlash2 drafter: head_dim 128, 4 queries per kv head, and a sliding window, which this
 * kernel supports and the h256 pair above does not. */
static const RadConstraint cAttnDecodeH128Fp8[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH128Bf16[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Gemma 4 dense sliding window: head_dim 256, 2 queries per kv head, windowed decode
 * (r4d_attn_paged_h256_gqa2.hip). `window` is a launch argument (1024 in the model, 0 = full
 * attention). q_len <= 32 is q_len*gqa <= 64 at gqa 2, and the bucket boundary to prefill. */
static const RadConstraint cAttnDecodeH256Gqa2Fp8[] = {
    C_GE("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH256Gqa2Bf16[] = {
    C_GE("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Head_dim 128 at 8 queries per kv head under attn_paged_qrow: the gqa-4 kernel above with each kv
 * head mapped onto two query groups (R4DArgs::kv_rep), so q_len * 4 <= 64 bounds it where gqa 8's
 * own decode (q_len <= 8, no window) stops at 8. The 31B's DFlash drafter: 64 heads over 8 kv heads,
 * sliding layers over 2048 and a full non-causal one, at a block of 16. Under attn_paged_qrow, which
 * no host library declares, so a drafter asking for it learns whether a card serves it. */
static const RadConstraint cAttnQrowH128G8Fp8[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnQrowH128G8Bf16[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Gemma 4 dense sliding-window prefill (r4d_attn_paged_h256_gqa2.hip): the decode leg's geometry
 * without a q_len bound, serving q_len > 32. The kernel's WIN bit masks the window and skips tiles
 * below it. Without this row the prefill band would have no device kernel, which is fatal (spec
 * §17: libref is an opt-in fallback). */
static const RadConstraint cAttnPrefillH256Gqa2Fp8[] = {
    C_GE("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH256Gqa2Bf16[] = {
    C_GE("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Gemma 3 sliding window: head_dim 128, 2 queries per kv head (r4d_attn_paged_h128_gqa2w.hip).
 * window >= 1 only: window 0 at this geometry is the tuned full-attention pair above. Decode is
 * q_len <= 32 (q_len*gqa <= 64), prefill takes the rest. */
static const RadConstraint cAttnDecodeH128Gqa2wFp8[] = {
    C_GE("window", 1), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH128Gqa2wBf16[] = {
    C_GE("window", 1), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnPrefillH128Gqa2wFp8[] = {
    C_GE("window", 1), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH128Gqa2wBf16[] = {
    C_GE("window", 1), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Eight queries per KV head, at both head widths (r4d_attn_paged_gqa8.hip). q_len <= 8 is
 * `q_len * gqa <= 64` at gqa 8: a decode workgroup holds 64 rows, each query position claims gqa
 * of them. As on the gqa6 leg, it is also the decode/prefill bucket boundary (spec §2.2). */
static const RadConstraint cAttnPrefillH256G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH256G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The decode pair at gqa 8, head_dim 256. */
static const RadConstraint cAttnDecodeH256G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH256G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Four queries per KV head, head_dim 256 (r4d_attn_paged_h256_gqa4.hip): Qwen3.5-0.8B, 8 query
 * heads over 2. `q_len <= 16` is the same `q_len * gqa <= 64` bound, so every legal speculation
 * depth is a decode launch at this ratio. */
static const RadConstraint cAttnPrefillH256G4Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH256G4Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnDecodeH256G4Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH256G4Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The prefill pair at head_dim 64 (LFM2: 32 query heads over 8). No decode row exists at this
 * width (the split-KV kernel's lanes hold 128 or 256 columns), so the query-tiled kernel serves
 * every q_len. */
static const RadConstraint cAttnPrefillH64G4Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 64), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH64G4Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 64), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The windowed kernels (r4d_attn_paged_window.hip): head_dim 256 and 512, 1/2/4/8 queries per KV
 * head, any window, causal or not. Gemma 4's sliding layers are h256/gqa2 with a 1024-token
 * window, its full layers h512 at gqa 8 (TP1/TP2), 4 (TP4) or 2 (TP8). One kernel serves every
 * q_len (splitting the key range when there are too few tiles to fill the card), so `q_len` cuts
 * no band. `causal` and `window` are runtime fields; block_size is derived: 16.
 *
 * h256 at gqa 4 and 8 take a window only (C_GE 1), leaving the window-0 band to the gqa4 / gqa8
 * legs above. At h256/gqa2 these rows overlap the sliding leg's bands and rank below it
 * (R4D_PRIO_ATTN_WINDOW); every other geometry here is theirs alone. */
#define R4D_WIN_C(NAME, HD, G, KV, WMIN)                                                           \
    static const RadConstraint NAME[] = {                                                          \
        C_EQ("head_dim", HD), C_EQ("gqa", G), C_EQ("block_size", 16), C_GE("causal", 0),           \
        C_GE("window", WMIN), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", KV),                       \
    };
R4D_WIN_C(cWinH256G1Bf16, 256, 1, "bf16", 0) R4D_WIN_C(cWinH256G1Fp8, 256, 1, "fp8_e4m3", 0)
R4D_WIN_C(cWinH256G2Bf16, 256, 2, "bf16", 0) R4D_WIN_C(cWinH256G2Fp8, 256, 2, "fp8_e4m3", 0)
R4D_WIN_C(cWinH256G4Bf16, 256, 4, "bf16", 1) R4D_WIN_C(cWinH256G4Fp8, 256, 4, "fp8_e4m3", 1)
R4D_WIN_C(cWinH256G8Bf16, 256, 8, "bf16", 1) R4D_WIN_C(cWinH256G8Fp8, 256, 8, "fp8_e4m3", 1)
R4D_WIN_C(cWinH512G1Bf16, 512, 1, "bf16", 0) R4D_WIN_C(cWinH512G1Fp8, 512, 1, "fp8_e4m3", 0)
R4D_WIN_C(cWinH512G2Bf16, 512, 2, "bf16", 0) R4D_WIN_C(cWinH512G2Fp8, 512, 2, "fp8_e4m3", 0)
R4D_WIN_C(cWinH512G4Bf16, 512, 4, "bf16", 0) R4D_WIN_C(cWinH512G4Fp8, 512, 4, "fp8_e4m3", 0)
R4D_WIN_C(cWinH512G8Bf16, 512, 8, "bf16", 0) R4D_WIN_C(cWinH512G8Fp8, 512, 8, "fp8_e4m3", 0)
#undef R4D_WIN_C
/* Twelve queries per KV head, head_dim 256 (r4d_attn_paged_h256_gqa12.hip): Qwen3.8-Flash-Next's
 * QSA layers, 24 query heads over 2. `q_len <= 5` is the same `q_len * gqa <= 64` bound, tighter
 * than the legal speculation depths: a depth-7 verify (q_len 8) takes the prefill kernel, which
 * serves any q_len causally. */
static const RadConstraint cAttnPrefillH256G12Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH256G12Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnDecodeH256G12Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeGqH256G12Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint cAttnDecodeH256G12Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnDecodeGqH256G12Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
/* Six queries per KV head, head_dim 256, paged block 4 (r4d_attn_paged_h256_gqa6_bs4.hip): the
 * gqa12 leg's QSA layers at four ranks. The model's two KV heads are each replicated across a rank
 * pair, so a rank declares 6 queries over its 1 kv head. `q_len <= 10` is `q_len * gqa <= 64` at
 * 6, so a draft block of six or seven positions is a decode launch here. The page stays 4: it is
 * the indexer's block size.
 *
 * The `attn_paged_gate_quant` rows are registered here too (see attn_decode_gq_h256_gqa6_fp8kv):
 * the `_gq` entry points are instantiated at every ratio, and the block declares the fused op
 * beside its unfused pair, so an unregistered geometry would be reported as a band hole. */
static const RadConstraint cAttnPrefillH256G6Bs4Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH256G6Bs4Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnDecodeH256G6Bs4Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH256G6Bs4Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The gate and quantiser as the merge's epilogue at the four-rank ratio: the gqa12 pair's
 * constraints with ratio and verify width moved together (`q_len <= 10`, not `<= 5`). */
static const RadConstraint cAttnDecodeGqH256G6Bs4Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint cAttnDecodeGqH256G6Bs4Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
/* The fp8_v4 and k6_v4 caches on the QSA layers' two ratios (r4d_attn_paged_gqa8.hip's
 * R4D_ATTN_V4): the fp8 rows' constraints at those kv_dtype values. */
static const RadConstraint cAttnPrefillH256G12Fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cAttnDecodeH256G12Fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cAttnDecodeGqH256G12Fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint cAttnPrefillH256G12K6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cAttnDecodeH256G12K6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cAttnDecodeGqH256G12K6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 5), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint cAttnPrefillH256G6Bs4Fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cAttnDecodeH256G6Bs4Fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cAttnDecodeGqH256G6Bs4Fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint cAttnPrefillH256G6Bs4K6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cAttnDecodeH256G6Bs4K6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cAttnDecodeGqH256G6Bs4K6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_GE("causal", 0), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint cAttnPrefillH128G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH128G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Non-causal is served, the window is not.
 *
 * `causal` is a runtime field of R4DArgs: at cz == 0 the shared decode kernel bounds the key by
 * ctx - 1 instead of the row's own position, which is a block-diffusion drafter's block attending
 * to the whole context and itself (r4d_attn_decode_h256_gqa6.hip, klimit). DSpark issues a
 * non-causal gqa8 decode (16 heads over 2 kv at head_dim 128, over 7 noise rows).
 *
 * `window` stays C_EQ 0 because it is a template parameter: this dispatch instantiates WIN = 0,
 * so a windowed geometry would silently attend to everything. */
static const RadConstraint cAttnDecodeH128G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH128G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Two queries per KV head at head_dim 128 (Qwen3-1.7B, Qwen3-ASR's decoder): the gqa8 pair's
 * constraints with the ratio and the decode verify width changed together. A decode workgroup
 * holds 64 (query, head) rows, so q_len <= 32 at gqa 2. */
static const RadConstraint cAttnPrefillH128G2Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH128G2Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnDecodeH128G2Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH128G2Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 2), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* Key rename: libr4d writes C_IN("q_dtype","bf16") and C_IN("kv_dtype","bf16"); docs/OPS.md's
 * `attn_dense` has one `dtype`. The predicate is unchanged.
 *
 * No constraint on `window`: it is optional and vision towers omit it, and a constraint on an
 * absent key does not hold. The kernel refuses a non-zero window at launch (r4d_rad_attn_vit). */
static const RadConstraint cAttnVit[] = {
    C_EQ("head_dim", 72), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16"),
};

/* `cast` at bf16 -> bf16: a strided row copy, the device arm for an architecture that taps an
 * intermediate residual stream (DFlash2 conditions its drafter on the trunk's hidden states at five
 * layers) and must copy rows out of a buffer the next layer overwrites. Without it `cast` resolves
 * only to ref on the host. Only from == to == bf16: this kernel copies bytes, it does not
 * convert. */
static const RadConstraint cCastBf16[] = {
    C_IN("from", "bf16"), C_IN("to", "bf16"),
};
/* The audio attention's window: a query's keys are held in LDS, 64 of them at most. */
static const RadConstraint cAttnLocalRel[] = {
    C_IN("dtype", "bf16"), C_LE("left", 63), C_EQ("right", 0),
};
/* `cast` widening bf16 -> f32: an exact conversion, the form a pooled embedding leaves the
 * device in. */
static const RadConstraint cCastF32[] = {
    C_IN("from", "f32"), C_IN("to", "f32"),
};
static const RadConstraint cCastWiden[] = {
    C_IN("from", "bf16"), C_IN("to", "f32"),
};
/* `cast` narrowing bf16 -> f16, rounded to nearest: the f16 activation of a 4-bit weight's skinny
 * GEMM (w4a16_asym) where no rotation produces one. */
static const RadConstraint cCastF16[] = {
    C_IN("from", "bf16"), C_IN("to", "f16"),
};
/* `cast` narrowing f32 -> bf16, rounded to nearest even: an f32 latent into a bf16 network (the
 * MiniMax-H3 latent upscaler's input voxels). */
static const RadConstraint cCastNarrow[] = {
    C_IN("from", "f32"), C_IN("to", "bf16"),
};
/* Dense attention at the (head_dim, gqa) pairs r4d_attn_dense_bf16.hip compiles: an encoder's,
 * bidirectional with any window, or a text encoder's causal prompt. gqa 1 at 256 and 512 is
 * attn_dense_small_h256's and the wide kernel's. */
/* Causal or not at head_dim 64: a vision tower's heads, and a codec's sliding causal layers. */
static const RadConstraint cAttnDenseH64[] = {
    C_EQ("head_dim", 64), C_EQ("gqa", 1), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH256G2[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 2), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH256G4[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 4), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH256G8[] = {
    C_EQ("head_dim", 256), C_DIV("gqa", 8), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH512G2[] = {
    C_EQ("head_dim", 512), C_EQ("gqa", 2), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH512G4[] = {
    C_EQ("head_dim", 512), C_EQ("gqa", 4), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH512G8[] = {
    C_EQ("head_dim", 512), C_DIV("gqa", 8), C_IN("dtype", "bf16"),
};
/* A head in a wave's registers with a rotary pair in one lane: head_dim a multiple of 32 up to
 * 1024. The shim refuses a rotary_dim whose half is not a multiple of 32 (neox) by name. */
static const RadConstraint cQkNormRope[] = {
    C_LE("head_dim", 1024), C_DIV("head_dim", 2),
};

/* ---- gated delta net ----------------------------------------------------------------------- */
static const RadConstraint cGdnChunkScan[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64),
};
static const RadConstraint cGdnConvPrep[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64),
};
static const RadConstraint cGdnConvUpdate[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128),
};
static const RadConstraint cGdnKktSolve[] = {
    C_EQ("head_k", 128), C_EQ("chunk", 64),
};
static const RadConstraint cGdnRecurrentUpdate[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128),
};
/* q_len <= 8 is the register window: the fold preloads a sequence's tokens into registers and
 * indexes them at compile time, as the standalone update does. The entry point checks the conv
 * cache's depth against the same bound. */
static const RadConstraint cGdnConvRecurrentUpdate[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("conv_width", 4), C_GE("q_len", 1),
    C_LE("q_len", 8),
};
static const RadConstraint cGdnGatedRmsNorm[] = {
    C_EQ("channels", 128),
};

/* ---- all-gather -------------------------------------------------------------------------------
 * Issued only by the vocab-parallel sampler's candidate merge. No `exact` key: a gather moves its
 * payload and cannot be lossy. dtype gives the element size (the pairs are half u32), and both
 * spellings of each type are listed because the arch blocks declare `bf16`/`fp32` and
 * core/sample/sampler.cpp `f32` (r4d_ar_dtype_code accepts either). */
static const RadConstraint cArGather2[] = {
    C_EQ("world_size", 2), C_IN("dtype", "bf16 fp16 f16 fp32 f32"),
    C_P2P,
};
/* The same gather at a width whose handshake is the wide family's tree, not one peer flag; the
 * dtype set is cArGather2's. */
static const RadConstraint cArGather4[] = {
    C_EQ("world_size", 4), C_IN("dtype", "bf16 fp16 f16 fp32 f32"),
    C_P2P,
};
static const RadConstraint cArGather8[] = {
    C_EQ("world_size", 8), C_IN("dtype", "bf16 fp16 f16 fp32 f32"),
    C_P2P,
};

/* ---- all-reduce ------------------------------------------------------------------------------
 * `exact` separates the exact sum from the lossy wire; a request that does not declare it gets no
 * kernel. Message size separates one-shot from two-shot at widths that have both: the two-shot rows
 * start above R4D_AR_WIDE_ONESHOT_MAX_ELEMS, so the band at or under it resolves only the one-shot,
 * and above it both match and R4D_PRIO_AR_TWOSHOT > R4D_PRIO_AR_ONESHOT picks the two-shot. The GE
 * is one past the bound so the lower band ends on it, a multiple of every DIV here (see
 * pAllReduce's `numel`). The selector makes the choice, so the band report names it. The ti8 wire
 * needs no such key: exact=0 separates it, and it serves its own small messages exactly. */
static const RadConstraint cArExact[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_P2P,
};
static const RadConstraint cArGather3[] = {
    C_EQ("world_size", 3), C_IN("dtype", "bf16 fp16 f16 fp32 f32"),
    C_P2P,
};
static const RadConstraint cArExact3[] = {
    C_EQ("world_size", 3), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_P2P,
};
static const RadConstraint cArExact4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_P2P,
};
static const RadConstraint cArExact8[] = {
    C_EQ("world_size", 8), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_P2P,
};
/* No DIV: the three-rank two-shot shards unevenly (r4d_rad_ar_twoshot_3rank_exact), so the only
 * shape it needs is the 16-byte push ar_setup already checks. */
static const RadConstraint cArTsExact3[] = {
    C_EQ("world_size", 3), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_GE("numel", R4D_AR_WIDE_ONESHOT_MAX_ELEMS + 1), C_LE("numel", R4D_AR_TWOSHOT_WIDE_MAX_ELEMS),
    C_P2P,
};
static const RadConstraint cArTsExact4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R4D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R4D_AR_TWOSHOT_WIDE_MAX_ELEMS),
    C_P2P,
};
/* The same at `overlap` 1 (tArTs4Lane), so a pipelined prefill on the exact wire does not get the
 * wide launch above, which holds the card beside the other piece's work. */
static const RadConstraint cArTsExact4Lane[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R4D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R4D_AR_TWOSHOT_WIDE_MAX_ELEMS),
    C_EQ("overlap", 1), C_P2P,
};
static const RadConstraint cArTsExact8[] = {
    C_EQ("world_size", 8), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R4D_AR_8RANK_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R4D_AR_TWOSHOT_WIDE_MAX_ELEMS),
    C_P2P,
};
static const RadConstraint cArWht6[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
    C_P2P,
};
/* The same two at `overlap` 1: the rows below whose launch defaults leave the CUs to compute. */
static const RadConstraint cArExactLane[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"), C_EQ("overlap", 1),
    C_P2P,
};
static const RadConstraint cArWht6Lane[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
    C_EQ("overlap", 1), C_P2P,
};
/* DIV 32, not 96: the three-rank split is uneven in whole groups (r4d_rad_ar_twoshot_3rank_ti8). */
static const RadConstraint cArTi8Ts3[] = {
    C_EQ("world_size", 3), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 32),
    C_P2P,
};
static const RadConstraint cArTi8Ts4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 128),
    C_P2P,
};
/* The same at `overlap` 1 (tArTs4Lane). */
static const RadConstraint cArTi8Ts4Lane[] = {
    C_EQ("world_size", 4), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 128),
    C_EQ("overlap", 1), C_P2P,
};

/* ---- GEMM ------------------------------------------------------------------------------------
 * DFlash2's grouped dynamic depthwise convolution. block_size must be a power of two (the body
 * masks the position with block_size-1), which no constraint can express; the entry point
 * rejects others. */
static const RadConstraint cDflashConv[] = {
    C_EQ("taps", 2), C_EQ("group_size", 16), C_DIV("hidden_size", 128), C_IN("dtype", "bf16"),
};
static const RadConstraint cDflashSelect[] = {
    C_GE("steps", 1), C_GE("top_k", 1), C_LE("top_k", 32),
    C_DIV("rank", 32), C_LE("rank", 512), C_IN("dtype", "bf16"),
};
/* The sandwich-norm joins hold 24 elements a thread over a 256-thread block (n <= 6144), and the
 * route form quantises in whole 128-column groups. pair_argmax takes any row; expert_scale's top_k
 * is the router's, at most 64. */
static const RadConstraint cNormJoin[] = {
    C_GE("n", 1), C_LE("n", 6144), C_IN("dtype", "bf16"),
};
static const RadConstraint cNormJoinRoute[] = {
    C_GE("n", 128), C_LE("n", 6144), C_DIV("n", 128), C_EQ("group", 128), C_IN("dtype", "bf16"),
};
static const RadConstraint cPairArgmax[] = {
    C_GE("n", 1), C_IN("dtype", "f32 fp32"),
};
static const RadConstraint cExpertScale[] = {
    C_GE("top_k", 1), C_LE("top_k", 64), C_GE("n_expert", 1), C_IN("dtype", "bf16"),
};
/* Runtime LoRA. Any M, K and N: the scalar kernels loop over whatever extents arrive, and the grid
 * is M blocks (shrink) or M strips of 256 columns (expand). R and S are bounded only by the shims'
 * 32-bit index sanity. The dtype set is every activation the decode reads (an fp8 or i8 one with
 * its scale stream) and the three outputs a GEMM writes. */
static const RadConstraint cLoraShrink[] = {
    C_GE("R", 1), C_LE("R", 65536), C_GE("S", 1), C_LE("S", 1048576),
    C_IN("dtype", "bf16 f16 f32 fp8_e4m3 fp8_e5m2 i8"),
};
static const RadConstraint cLoraExpand[] = {
    C_GE("R", 1), C_LE("R", 65536), C_GE("S", 1), C_LE("S", 1048576),
    C_IN("dtype", "bf16 f16 f32"),
};
/* The tiled rows (r4d_lora_bf16.hip). Shrink is two rows because a constraint on an absent
 * parameter does not hold: a bf16 activation has no `group`, and a coded one is the WMMA's only
 * when its group is whole 32-wide steps (the scale is applied a step at a time). Anything else
 * (f16, f32, a coded activation with another group or none declared) is the scalar row's. */
static const RadConstraint cLoraShrinkWmma[] = {
    C_GE("R", 1), C_LE("R", 65536), C_GE("S", 1), C_LE("S", 1048576), C_IN("dtype", "bf16"),
};
static const RadConstraint cLoraShrinkWmmaQ[] = {
    C_GE("R", 1), C_LE("R", 65536), C_GE("S", 1), C_LE("S", 1048576),
    C_IN("dtype", "fp8_e4m3 fp8_e5m2 i8"), C_GE("group", 32), C_DIV("group", 32),
};

/* ---- the routed mixture of experts ----------------------------------------------------------
 *
 * The router's `n_expert` bound is a hard limit: the selection keeps a lane's share of a row's
 * logits in registers, and the scatter's small form keeps a one-word-per-expert LDS histogram,
 * both sized at r4d_moe.hip's kMaxExpert (see cRouterTopK). A wider model needs a different kernel.
 *
 * The GEMM's N%64 is the block's N tile and K%128 the fp8 scale block; `group` must equal that
 * block, so the selector, not the launch, declines any other quantisation. */
static const RadConstraint cScaleRows[] = {
    C_GE("M", 1), C_GE("n", 1), C_IN("dtype", "bf16"),
};
/* The gated residual. `n <= 7552` and `hc <= 4` bound hc*n at 30208, inside the 30576 bf16
 * elements kernel A can hold the normed row in after its 4384 bytes of static planes in 64 KiB
 * (kHcLdsMaxHN); no device fallback exists for a wider stream (libref is host domain). 7552 is the
 * widest multiple of 128 under that bound at hc = 4. `n` divisible by 128 is both the workgroup's
 * channel tile and the fp8 scale group, so a workgroup owns whole groups. */
static const RadConstraint cHcRead[] = {
    C_GE("M", 1), C_DIV("n", 128), C_LE("n", 7552), C_GE("hc", 2), C_LE("hc", 4),
    C_GE("lowrank", 1), C_LE("group", 128), C_LE("inject", 1), C_IN("dtype", "bf16"),
    C_IN("mix", "bf16"),
};
/* The mixing matrices as E4M3 rows (r4d_hc_fp8.h). `n` a multiple of 512 keeps hc*n a whole
 * number of the down dot's 512-element steps at any hc; `lowrank` a multiple of 64 up to 512 makes
 * the up row's quarter (its scale group) one to eight sixteen-code loads. */
static const RadConstraint cHcReadE4m3[] = {
    C_GE("M", 1), C_DIV("n", 512), C_LE("n", 7552), C_GE("hc", 2), C_LE("hc", 4),
    C_DIV("lowrank", 64), C_LE("lowrank", 512), C_LE("group", 128), C_LE("inject", 1),
    C_IN("dtype", "bf16"), C_IN("mix", "e4m3"),
};
static const RadConstraint cHcPlain[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};
static const RadConstraint cMhcPlain[] = {
    C_GE("M", 1),  C_LE("M", 65535), C_GE("n", 1),          C_LE("n", 16384),
    C_GE("hc", 2), C_LE("hc", 8),    C_IN("dtype", "bf16"),
};
/* GLM's split projection only: eight 2048-column chunks of a 4 x 4096 residual. */
static const RadConstraint cMhcPrePartial[] = {
    C_GE("M", 1),     C_LE("M", 65535), C_EQ("n", 4096),   C_EQ("hc", 4),     C_EQ("chunk", 2048),
    C_GE("first", 0), C_LE("first", 7), C_GE("chunks", 1), C_LE("chunks", 8), C_IN("dtype", "bf16"),
};
static const RadConstraint cMhcPreFinish[] = {
    C_GE("M", 1),        C_LE("M", 65535),      C_EQ("n", 4096),       C_EQ("hc", 4),
    C_EQ("chunk", 2048), C_IN("dtype", "bf16"), C_GE("iterations", 1), C_LE("iterations", 256),
};
static const RadConstraint cKdaUpdate[] = {
    C_GE("M", 1),          C_GE("heads", 1),    C_LE("heads", 65535),
    C_EQ("head_k", 128),   C_EQ("head_v", 128), C_IN("mode", "prefill verify"),
    C_IN("dtype", "bf16"),
};
static const RadConstraint cMhcPre[] = {
    C_GE("M", 1),          C_LE("M", 65535),      C_GE("n", 1),
    C_LE("n", 16384),      C_GE("hc", 2),         C_LE("hc", 8),
    C_IN("dtype", "bf16"), C_GE("iterations", 1), C_LE("iterations", 256),
};
static const RadConstraint cDsaPool[] = { C_GE("M", 1), C_EQ("n", 128), C_EQ("ratio", 4),
                                          C_IN("dtype", "bf16") };
static const RadConstraint cDsaQuery[] = { C_GE("M", 1), C_LE("M", 2147483644), C_EQ("n", 128),
                                           C_IN("dtype", "bf16") };
static const RadConstraint cFp8F32Scale[] = {
    C_GE("M", 1),    C_GE("N", 128),     C_DIV("N", 128),          C_GE("K", 128),
    C_DIV("K", 128), C_EQ("group", 128), C_IN("dtype", "fp8a8f32")
};
static const RadConstraint cGemmNtF32[] = { C_GE("M", 1),         C_LE("M", 65535),
                                            C_GE("N", 1),         C_LE("N", 512),
                                            C_GE("K", 1),         C_LE("K", 16384),
                                            C_IN("dtype", "bf16") };
static const RadConstraint cDsaScore[] = {
    C_GE("M", 1),      C_LE("M", 65535), C_EQ("n", 128),           C_EQ("heads", 32),
    C_GE("blocks", 1), C_GE("seqs", 1),  C_IN("dtype", "fp8_e4m3")
};
static const RadConstraint cMlaAttn[] = {
    C_GE("M", 1),      C_LE("M", 65535),  C_EQ("latent_dim", 512), C_GE("heads", 1),
    C_LE("heads", 64), C_GE("blocks", 1), C_EQ("block_size", 4),   C_IN("dtype", "bf16")
};
static const RadConstraint cMlaProject[] = { C_GE("M", 1),
                                             C_LE("M", 65535),
                                             C_GE("heads", 1),
                                             C_LE("heads", 64),
                                             C_EQ("head_dim", 256),
                                             C_EQ("latent_dim", 512),
                                             C_IN("mode", "absorb expand"),
                                             C_IN("dtype", "bf16") };
/* The matrix row outranks the wave row (mla_attn_bf16) at every M, the pairing walking the union
 * of both rows' pages: measured cold on gfx1201 (r4d_selftest --perfmla), a 2048-row chunk
 * with DSA tables (fill topk:16) takes 7.1 ms against 12.4 ms on the wave row and 13.0 ms on the
 * positional pairing, fresh 3.8 against 6.3 ms, a decode step of 8x6 rows 0.23 against 0.82 ms. */
static const RadConstraint cMlaMatrix[] = { C_GE("M", 1),
                                            C_LE("M", 65535),
                                            C_EQ("latent_dim", 512),
                                            C_GE("heads", 8),
                                            C_LE("heads", 64),
                                            C_DIV("heads", 8),
                                            C_GE("blocks", 1),
                                            C_EQ("block_size", 4),
                                            C_IN("dtype", "bf16") };
/* `n` is bounded by LDS: nh (hc*n) and ne (n) are both bf16 and live through the second phase;
 * 28672 elements is that budget at hc 4. */
static const RadConstraint cMtpEnter[] = {
    C_GE("M", 1), C_GE("n", 128), C_LE("n", 5734), C_GE("hc", 2), C_LE("hc", 4),
    C_IN("dtype", "bf16"), C_IN("fc", "bf16"),
};
/* The fc matrices as E4M3 rows, read by the tiled kernel at every row count: `n` a multiple of
 * its 64-column K slice and of the 128-column scale group. */
static const RadConstraint cMtpEnterE4m3[] = {
    C_GE("M", 1), C_DIV("n", 128), C_LE("n", 5734), C_GE("hc", 2), C_LE("hc", 4),
    C_IN("dtype", "bf16"), C_IN("fc", "e4m3"),
};
/* 512 is r4d_moe.hip's kMaxExpert, a hard limit: the router keeps a lane's share of the row's
 * logits in registers (16 per lane at 512, a separate instantiation) and both scatter forms keep an
 * n_expert + 1 histogram in LDS. Qwen3.8-Flash-Next routes 10 of 512. */
static const RadConstraint cRouterTopK[] = {
    C_GE("M", 1), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
    C_LE("top_k", 32), C_IN("dtype", "bf16"),
};
static const RadConstraint cRouterSigmoid[] = { C_GE("M", 1),        C_LE("M", 65535),
                                                C_GE("n_expert", 1), C_LE("n_expert", 512),
                                                C_GE("top_k", 1),    C_LE("top_k", 32),
                                                C_IN("dtype", "f32") };
/* The sort's single-workgroup form bounds M * top_k at 1024, which the shim checks. */
static const RadConstraint cRouterSigmoidScatter[] = { C_GE("M", 1),        C_LE("M", 1024),
                                                       C_GE("n_expert", 1), C_LE("n_expert", 512),
                                                       C_GE("top_k", 1),    C_LE("top_k", 32),
                                                       C_IN("dtype", "f32") };
/* The selection's bounds and the skinny GEMM's decode band: four 16-row tiles. */
static const RadConstraint cRouterGemmTopkScatter[] = {
    C_GE("M", 1), C_LE("M", 64), C_DIV("K", 16), C_GE("n_expert", 1), C_LE("n_expert", 512),
    C_GE("top_k", 1), C_LE("top_k", 32), C_IN("dtype", "bf16"),
};
static const RadConstraint cMoeScatter[] = {
    C_GE("M", 1), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
};
static const RadConstraint cMoeGemmQ[] = {
    C_GE("M", 1), C_DIV("N", 64), C_DIV("K", 128), C_GE("n_expert", 1), C_LE("n_expert", 65536),
    C_GE("top_k", 1), C_EQ("group", 128), C_IN("dtype", "fp8a8"),
};
static const RadConstraint cMoeMx[] = {
    C_GE("M", 1), C_LE("M", 32767), C_GE("N", 1), C_DIV("K", 128),
    C_GE("n_expert", 2), C_LE("n_expert", 512), C_GE("top_k", 1), C_LE("top_k", 32),
    C_EQ("group", 128), C_IN("dtype", "mxfp4a8"),
};
/* The same op at a four-bit expert weight: same band and operands, only the stored weight plane
 * and scale grid differ, so the dtype string alone separates the rows.
 *
 * `w4a8h` is the same row and launch: the weight was rotated by a 128-wide Hadamard along K before
 * quantisation, with the residual factor folded into the scale, so the kernel does not branch on
 * it. What differs is the layout tag (so a container converted one way is not served the other)
 * and the paired activation (`had_quant_act_fp8`, not `quant_act_fp8`).
 *
 * `w4nla8h` is the rotated plane with another code table: codes index libquant's w4nl levels
 * instead of -8..7, and the launcher passes the table (r4d_args.h). The layout hook checks the
 * container's table against the kernel's.
 *
 * `w4nl64a8h` is w4nla8h with one E4M3 scale per 64 of K under the fixed kW4G64Scale (the same
 * bytes as a bf16 scale per 128); `w4nl32a8h` one per 32, twice the bytes. `group` stays 128 (the
 * staged K block and the rotation), and the kernel sums each scale group separately.
 *
 * `w5nl64a8h` is w4nl64a8h at five bits: codes index libquant's 32-level w5nl table, whose high bit
 * is the sign; the kernel expands the low four as a w4nl magnitude and sets the sign from a bit
 * plane stored after each row's nibbles. */
static const RadConstraint cMoeGemmW4[] = {
    C_GE("M", 1), C_DIV("N", 64), C_DIV("K", 128), C_GE("n_expert", 1), C_LE("n_expert", 65536),
    C_GE("top_k", 1), C_EQ("group", 128),
    C_IN("dtype", "w4a8 w4a8h w4nla8h w4nl64a8h w4nl32a8h w5nl64a8h"),
};
/* The bf16 expert GEMM: 64-wide column tiles and 64-deep K passes, one table, no group. */
static const RadConstraint cMoeGemmBf16[] = {
    C_GE("M", 1), C_DIV("N", 64), C_DIV("K", 64), C_GE("n_expert", 1), C_LE("n_expert", 65536),
    C_GE("top_k", 1), C_IN("dtype", "bf16"),
};
static const RadConstraint cMoeGather[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("top_k", 1), C_IN("dtype", "bf16"),
};

/* R <= 32 is stage 2's PT (R4D_TOPK_PT2): a lane holds the whole answer, so no distribution of the
 * candidates over lanes can lose one. The merge and the rerank share the limit. */
static const RadConstraint cRowTopK[] = {
    C_DIV("N", 8), C_GE("M", 1), C_GE("R", 1), C_LE("R", 32), C_IN("dtype", "bf16"),
};
static const RadConstraint cRowTopKMerge[] = {
    C_GE("M", 1), C_GE("R", 1), C_LE("R", 32), C_GE("world_size", 1), C_IN("dtype", "bf16"),
};
static const RadConstraint cLogitRerank[] = {
    C_GE("M", 1), C_GE("R", 1), C_LE("R", 32), C_GE("n_vocab", 1), C_GE("n_embd", 1),
    C_IN("dtype", "bf16"),
};
static const RadConstraint cGemmNt[] = {
    C_LE("M", 16), C_DIV("K", 256), C_IN("dtype", "bf16"),
};
/* No 64-row cap: grid.y covers any row count, so the band is the grid's reach. A cap would force a
 * caller with a taller projection (the gated delta net's a|b) to slice it. */
static const RadConstraint cGemmNt64[] = {
    C_LE("M", 65536), C_DIV("K", 16), C_IN("dtype", "bf16"),
};
/* M > 64 is the prefill band, where the skinny row's per-wave fragment loads re-read the
 * activation once per N tile. N >= 128 because a 64- or 128-wide tile over a narrower N computes
 * mostly clamped rows; those shapes (the GDN a|b at 48, the shared-expert gate at 1) stay skinny. */
static const RadConstraint cGemmNtTiled[] = {
    C_GE("M", 65), C_LE("M", 65536), C_GE("N", 128), C_DIV("K", 64), C_IN("dtype", "bf16"),
};
static const RadConstraint cGemmNtW4[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a16"),
};
/* w5a16 serves any M: the dispatcher takes rows past 64 as chunks of 64. */
static const RadConstraint cGemmNtW5[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w5a16"),
};
static const RadConstraint cGemmNtW4Asym[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a16_asym"),
};
static const RadConstraint cGemmNtW4A8I8[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a8i8_asym"),
};
static const RadConstraint cGemmNtW5A8I8[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w5a8i8"),
};
static const RadConstraint cGemmNtW2A8[] = {
    C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w2a8"),
};
static const RadConstraint cGemmNtW8[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w8a16"),
};
static const RadConstraint cGemmNtW4A8[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a8 w4a8_asym"),
};
static const RadConstraint cGemmNtMxfp4A8[] = {
    C_LE("M", 64), C_DIV("K", 32), C_DIV("N", 16), C_IN("dtype", "mxfp4a8"),
};
/* ---- block-scaled fp8 -------------------------------------------------------------------------
 *
 * The `group` predicate has no kernel-side counterpart (the kernel's quantised GEMMs take the
 * group from a build flag), but docs/OPS.md's `gemm_nt_q` schema carries one and this family's
 * block belongs to the checkpoint. Any other group gets no kernel rather than a silent 128. */
static const RadConstraint cGemmNtFp8A16[] = {
    C_LE("M", 8), C_DIV("K", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a16"),
};
/* K DIV 128, not 256: the entry serves a K that is a multiple of 128 by staging a 128-byte tile
 * instead of a 256-byte one (narrow_fk). The resolver offers shapes by declared constraints, so the
 * row must claim the band it can serve. */
static const RadConstraint cGemmNtFp8A8M16[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a8"),
};
static const RadConstraint cGemmNtFp8A8Tiled[] = {
    C_DIV("K", R4D_FP8_BLOCK), C_DIV("N", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a8"),
};
/* gemm_nt_q_bias on the narrow tile: the unbiased narrow row's band. The weight is the same
 * fragment-order plane, so the same layout hooks serve it. */
static const RadConstraint cGemmNtQBiasFp8A8M16[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a8"),
};
/* The int8 twins: the same bands at dtype i8a8. `i8a8h` is the same launch over a weight rotated
 * by a 128-wide Hadamard, the 1/128 in its scale, as `w4a8h` is beside `w4a8`; only its layout
 * tag and its activation (had_quant_act_i8g) differ. */
static const RadConstraint cGemmNtI8A8M16[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "i8a8 i8a8h"),
};
static const RadConstraint cGemmNtI8A8Tiled[] = {
    C_DIV("K", R4D_FP8_BLOCK), C_DIV("N", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "i8a8 i8a8h"),
};
/* libr4d's `K DIV 4`, renamed to docs/OPS.md's `n` as in every quantiser row here. */
static const RadConstraint cQuantActFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* The rotated form needs n divisible by the group, not just by 4: a rotation block cannot
 * straddle the end of a row (a zero-padded ragged tail would mix real values and padding in one
 * butterfly, matching no rotated weight block). */
static const RadConstraint cHadQuantActFp8[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* ParoQuant's dense rotation: the same whole-group rule, and `out` one of the two quantisers its
 * epilogue has. */
static const RadConstraint cGivensDense[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"), C_IN("out", "i8 fp8"),
};

/* No bound on `seqs`: the declared value is a deployment maximum (rad-convert declares 256), and a
 * row refusing it would not resolve at convert, leaving the op's weights out of the container. The
 * real count comes from `cu_seqlens` at issue and the launcher checks it. */
static const RadConstraint cQsaWork[] = {
    C_GE("n", 1), C_GE("ratio", 1), C_GE("seqs", 1), C_GE("work", 1), C_IN("dtype", "bf16"),
};

static const RadConstraint cQsaTailStore[] = {
    C_GE("n", 1), C_GE("ratio", 1), C_GE("seqs", 1), C_IN("dtype", "bf16"),
};

/* No dtype: every operand is f32 or i32. */
static const RadConstraint cQsaSelect[] = { C_GE("blocks", 1), C_GE("topk", 1), C_GE("ratio", 1) };

/* A lane owns n/32 dims of both operands, so n is a wave multiple, at most eight of them. The
 * head bound is the register file: the query stays resident across the block loop. */
static const RadConstraint cQsaScore[] = {
    C_DIV("n", 32), C_GE("n", 32), C_LE("n", 256), C_GE("heads", 1), C_LE("heads", 8),
    C_GE("blocks", 1), C_IN("dtype", "bf16"),
};

/* The head dim is bounded by the LDS row the block reduction needs, and the rotary width by the
 * neox pairing (`rot/2` apart, so even and inside the head). The bounds are the kernel's. */
static const RadConstraint cQsaBlockKey[] = {
    C_GE("n", 32), C_LE("n", 256), C_DIV("n", 32), C_GE("ratio", 1),
    C_GE("rotary_dim", 0), C_IN("dtype", "bf16"),
};

/* The dtype is the input's, already quantised: `fp8a8`, not `bf16` as elsewhere in this family.
 * `n` a multiple of 4 is the staging slot (one dword of four codes); the 64-wide C tile masks a
 * ragged tile. */
static const RadConstraint cGramAccum[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "fp8a8"),
};
/* The bf16 form has no scale grid, so `group` constrains nothing; the staging slot is four
 * elements of a row. */
static const RadConstraint cGramAccumBf16[] = {
    C_DIV("n", 4), C_IN("dtype", "bf16"),
};
/* The fused fp8 forms carry the quantiser's own constraints; the int8 fusions below also need the
 * rotation's divisor. */
static const RadConstraint cRmsNormQuantFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};

/* `n DIV 8`, not 4: `x` is also the all-reduce's message, pushed as 16-byte words. world_size 2:
 * the handshake is one peer flag per block (as cArLnHadQuantI8). */
static const RadConstraint cArRmsNormQuantFp8[] = {
    C_DIV("n", 8), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
    C_EQ("world_size", 2),
    C_P2P,
};
/* The rotated int8 twins: `n DIV 128`, since a rotation block cannot straddle the end of a row
 * (which also gives the all-reduce's 16B words). */
static const RadConstraint cRmsNormHadQuantI8g[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
static const RadConstraint cArRmsNormHadQuantI8g[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
    C_EQ("world_size", 2),
    C_P2P,
};
/* hc_write's own bounds plus the all-reduce's: `n DIV 8` because `y` is pushed as 16B words, and
 * world_size 2 because the handshake is one peer flag per block. */
static const RadConstraint cArHcWrite[] = {
    C_GE("M", 1), C_DIV("n", 8), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
    C_EQ("world_size", 2),
    C_P2P,
};
/* The same at four ranks, with the wide family's handshake (r4d_ar_wide.h). Exact wire only: the
 * four-rank lossy wire is the ti8 two-shot, which has no fused form, so `wire` 6 misses selection
 * and the caller keeps the unfused pair. Callers must state `wire` 0, since a constraint on an
 * absent key does not hold; init and the entry refuse other wires. */
static const RadConstraint cArHcWrite4[] = {
    C_GE("M", 1), C_DIV("n", 8), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
    C_EQ("world_size", 4), C_EQ("wire", 0),
    C_P2P,
};
/* ar_hc_write's bounds and the gather's: top_k up to the 64 slots a row's table holds in LDS. */
static const RadConstraint cArGatherHcWrite[] = {
    C_GE("M", 1), C_DIV("n", 8), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
    C_EQ("world_size", 2), C_GE("top_k", 1), C_LE("top_k", 64),
    C_P2P,
};
/* The same at four ranks, exact wire only (see cArHcWrite4). */
static const RadConstraint cArGatherHcWrite4[] = {
    C_GE("M", 1), C_DIV("n", 8), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
    C_EQ("world_size", 4), C_EQ("wire", 0), C_GE("top_k", 1), C_LE("top_k", 64),
    C_P2P,
};
static const RadConstraint cGatedQuantFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
static const RadConstraint cSwigluClampedQuant[] = {
    C_GE("M", 1), C_LE("M", 65535), C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "bf16"),
};
/* n divisible by the group, as in cHadQuantActFp8. */
static const RadConstraint cGatedHadQuantFp8[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* The narrow GEMM's row plus the fold's own constraints. `N DIV 256` is `nout DIV 128` on the
 * weight's extent: a block emits exactly one 128-column scale group. `act IN silu` because only
 * that arm is compiled (a sigmoid gate reaches `gate_quant_fp8`).
 *
 * `N GE 2048` is the split-K rule: the fold has no partial plane, so its entry point refuses any
 * shape the narrow split rule would split, which is K >= 512 with N/16 under 128 waves, i.e. every
 * N under 2048 (the K = 256 exception is a residual width no model has). Claiming that band would
 * fail at issue instead of being reported at declare. */
static const RadConstraint cGemmNtQGatedFp8[] = {
    C_LE("M", 64), C_DIV("K", 256), C_DIV("N", 256), C_GE("N", 2048),
    C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "fp8a8"), C_IN("act", "silu"),
};
static const RadConstraint cGateQuantFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* libr4d's cSampleChain. The kernel has no geometry constraint (the radix passes run over a
 * float's 256 byte buckets), but a row with no predicate would match a query that named nothing. */
static const RadConstraint cSampleChain[] = {
    C_GE("n_vocab", 1),
};

static const RadConstraint cQuantActI8[] = {
    C_DIV("n", 16), C_IN("dtype", "bf16"),
};
/* `n DIV 16`: a k step of the fragment order is sixteen bytes of one row. The row is staged in LDS
 * as bf16 for the cross-thread fragment store; `n <= 30720` is the shims' 60 KiB bound, stated
 * here so a wider row is refused at selection and the caller can keep the unfused chain. */
static const RadConstraint cRmsNormQuantI8[] = {
    C_DIV("n", 16), C_LE("n", 30720), C_IN("dtype", "bf16"),
};
/* head_dim 256 or 512: a workgroup is one head, a thread one channel. */
static const RadConstraint cAddRmsNorm[] = {
    C_GE("n", 1), C_IN("dtype", "bf16"),
};
static const RadConstraint cQkvNormRope[] = {
    C_GE("head_dim", 256), C_LE("head_dim", 512), C_DIV("head_dim", 256), C_IN("dtype", "bf16"),
};
/* Gemma 3's prologue (arch/gemma3): head_dim 128, usually with v_norm 0. */
static const RadConstraint cQkvNormRopeH128[] = {
    C_EQ("head_dim", 128), C_IN("dtype", "bf16"),
};
static const RadConstraint cQkvNormRopeKv[] = {
    C_GE("head_dim", 256), C_LE("head_dim", 512), C_DIV("head_dim", 256), C_IN("dtype", "bf16"),
    C_GE("block_size", 1), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cQkvNormRopeKvBf16[] = {
    C_GE("head_dim", 256), C_LE("head_dim", 512), C_DIV("head_dim", 256), C_IN("dtype", "bf16"),
    C_GE("block_size", 1), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cGeluMulQuantI8[] = {
    C_DIV("n", 16), C_LE("n", 30720), C_IN("dtype", "bf16"),
};
/* x is the message, pushed as 16B words, and the row is staged in LDS for the fragment store. */
static const RadConstraint cArRmsNormQuantI8[] = {
    C_DIV("n", 16), C_LE("n", 30720), C_IN("dtype", "bf16"), C_EQ("world_size", 2), C_P2P,
};
/* The same at four ranks, on the wide family's transport. */
static const RadConstraint cArRmsNormQuantI84[] = {
    C_DIV("n", 16), C_LE("n", 30720), C_IN("dtype", "bf16"), C_EQ("world_size", 4), C_P2P,
};
static const RadConstraint cHadQuantActI8[] = {
    C_DIV("n", 16), C_DIV("n", 128), C_LE("group", 512), C_IN("dtype", "bf16"),
};
static const RadConstraint cRmsNormHadQuantI8[] = {
    C_DIV("n", 16), C_DIV("n", 128), C_LE("group", 512), C_IN("dtype", "bf16"),
};
/* world_size 2: the handshake is one peer flag per block. `n <= 16256` is the widest row whose
 * fp32 staging fits LDS beside the reduction planes at the widest workgroup
 * (r4d_ar_ln_blocks_min), rounded down to a multiple of 128. The compressed wire also stages every
 * row a block owns, bounding M by the flag capacity times the rows that fit; the shim checks that
 * product. */
static const RadConstraint cArLnHadQuantI8[] = {
    C_DIV("n", 16), C_DIV("n", 128), C_LE("n", 16256), C_LE("group", 512), C_IN("dtype", "bf16"),
    C_EQ("world_size", 2),
    C_P2P,
};
static const RadConstraint cQkNormRopeGate[] = {
    C_IN("q_dtype", "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32),
};
static const RadConstraint cQkNormRopeGateKv[] = {
    C_IN("q_dtype", "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32),
    C_GE("block_size", 1), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cQkNormRopeGateKvFp8v4[] = {
    C_IN("q_dtype", "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32),
    C_GE("block_size", 1), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint cQkNormRopeGateKvK6v4[] = {
    C_IN("q_dtype", "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32),
    C_GE("block_size", 1), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint cQkNormRopeGateKvBf16[] = {
    C_IN("q_dtype", "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32),
    C_GE("block_size", 1), C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cRopeTable[] = {
    C_GE("M", 1), C_GE("rot", 2), C_DIV("rot", 2), C_IN("dtype", "f32 bf16"),
};
static const RadConstraint cGatedHadQuantI8[] = {
    C_DIV("n", 16), C_DIV("n", 128), C_LE("group", 512), C_IN("dtype", "bf16"),
};
static const RadConstraint cGdnGatedNormHadQuantI8[] = {
    C_DIV("n", 128), C_LE("group", 512), C_EQ("head_dim", 128), C_IN("dtype", "bf16"),
};
static const RadConstraint cDequantW4Bf16[] = {
    C_DIV("N", 16), C_DIV("K", 64), C_IN("dtype", "w4"),
};

/* ---- the int4 / int8 / rotated-quant family, retained ------------------------------------------
 *
 * No architecture in this tree resolves a dense w4a8, w8a8, w4a16 or `*_had_quant_i8` row, and
 * `rad-quantize`'s w4_g128 output has no engine consumer. The rows stay because carrying a kernel
 * family is cheaper than rewriting it; `rad-kbench --bench` covers them and r4d_selftest checks
 * the fusion claims.
 */

/* ---- rows whose band is read off the entry point ---------------------------------------------
 * These constraints come from the entry point's own rejection tests and header rather than a
 * kernel-side row, a weaker source, so each is marked.
 */

/* r4d_gemm_w8a8_nt_m64: the 4-bit kernel's fragment order at twice the radix, one f16 scale per
 * (row, group of 128) in an f16-only plane. Rejection tests mirror the w4a8 kernel's (N % 16,
 * K % (SK*group)). Its MAX_M of 32768 is not structural; as a skinny kernel with a tiled sibling
 * above 64, it takes the w4a8 row's band. */
static const RadConstraint cGemmNtW8A8[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w8a8"),
};
/* r4d_gemm_w4a8_tiled / _asym_tiled. Entry point: N % 256, K % group, M >= 1. */
static const RadConstraint cGemmW4A8Tiled[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 256), C_IN("dtype", "w4a8 w4a8_asym"),
};
/* r4d_gemm_mxfp4a8_decode. Entry point: M <= 128, N % 16, K % 64, and N up to its counter block's
 * 36864. Above the skinny row by priority, so it takes the decode band; the skinny row keeps the
 * wider N up to M = 64, the tiled row everything else. */
static const RadConstraint cGemmMxfp4A8Decode[] = {
    C_LE("M", 128), C_LE("N", 36864), C_DIV("K", 64), C_DIV("N", 16), C_IN("dtype", "mxfp4a8"),
};
/* r4d_gemm_mxfp4a8_tiled. Entry point: N % 16 (the fragment-order weight), K % 64 (its narrowest
 * K slab), M >= 1. The decode and skinny rows outrank it in their bands by priority, so the band
 * starts at 1 rather than at a hand-picked crossover. */
static const RadConstraint cGemmMxfp4A8Tiled[] = {
    C_GE("M", 1), C_DIV("K", 64), C_DIV("N", 16), C_IN("dtype", "mxfp4a8"),
};
/* r4d_gemm_w8a8_tiled: the same kernel at radix 8. Its asym arm is not declared in r4d.h, so
 * `w8a8_asym` is absent from the dtype set. */
static const RadConstraint cGemmW8A8Tiled[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 256), C_IN("dtype", "w8a8"),
};
/* r4d_gemm_w4a8_prefill. Entry point: N % BN and K % group, BN being the variant's block width
 * (256 except for the two 1024-thread arms). N % 256 is the default's condition; the shim refuses
 * a variant whose BN does not divide N (the per-call half of spec §2.1). */
static const RadConstraint cGemmW4A8Prefill[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 256), C_IN("dtype", "w4a8"),
};

/* =================================================================================== tunables
 *
 * spec §15: constraint matching finds a kernel that is correct for a geometry; the tile
 * configuration depends on the shape and the machine. Below is the space each kernel can be
 * instantiated over, one axis at a time, plus cross-axis legality tests a product of independent
 * lists cannot express. Each axis carries the value an untuned install runs.
 *
 * Axes rather than named points: a hand-written point list has holes (thirty names over five axes
 * reach thirty of nine hundred points), and declaring the space lets the core tune a library it
 * was not compiled against.
 */

/* ------------------------------------------------------------------- the skinny GEMM's axes
 *
 * (WV, SK, MB, NPW, NT). Zero means "decide at issue" on the two axes with a law: SK is then the
 * shape's own ladder and MB covers every row tile in one block. The laws live in r4d_gemm_args.h
 * because they read M and the N tile count, known only at issue. A tuned value replaces the law.
 * A point such as "wv1_sk8_mb3_npw4" is wv=1 sk=8 mb=3 npw=4 nt=0; the space has nine hundred. */
static const int64_t kGemmWv[]  = { 1, 2, 4 };
static const int64_t kGemmSk[]  = { 0, 1, 2, 4, 5, 8, 10, 16, 20, 32 };
static const int64_t kGemmMb[]  = { 0, 1, 2, 3, 4 };
static const int64_t kGemmNpw[] = { 1, 2, 4 };
static const int64_t kGemmNt[]  = { 0, 1 };

/* The tile (BM, BN) is one axis because each value is one compiled arm. 0 is the shape's own
 * choice; see r4d_rad_gemm_bf16_tiled. */
static const int64_t kGemmBf16Tile[] = { 0, 1, 2 };
static const RadTunable tGemmBf16Tiled[] = {
  { "tile", kGemmBf16Tile, NELEM(kGemmBf16Tile), 0,
    "block tile: 1 = 128x128, 2 = 64x64. 0 = by the grid it gives" },
};
static const RadTunable tGemmSkinny[] = {
  { "wv",  kGemmWv,  NELEM(kGemmWv),  1, "waves per block; wv*sk*32 must stay inside the block bound" },
  { "sk",  kGemmSk,  NELEM(kGemmSk),  0, "split-K segment count, reduced in LDS. 0 = the shape's own ladder" },
  { "mb",  kGemmMb,  NELEM(kGemmMb),  0, "16-row tiles one block covers; grid.y is ceil(ceil(M/16)/mb). 0 = all of them" },
  { "npw", kGemmNpw, NELEM(kGemmNpw), 1, "N tiles per wave; widens the block to hide a dequantise, at MT*npw accumulators" },
  { "nt",  kGemmNt,  NELEM(kGemmNt),  1, "non-temporal weight loads" },
};

/* Geometry-only legality, so the tuner skips ~40% of the space without launching. wv*sk*32 is the
 * block's thread count, at most 1024 in hardware; the tighter per-MB/NPW bound depends on the row
 * and lives in r4d_gemm_args.h's block_bound(). K must divide into SK slices of the kernel's unit:
 * 16 for bf16, 128 where a quantised weight carries a scale per 128 K. cfg_for() refuses the same
 * combinations at issue. */
static int r4d_tune_valid_gemm(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    long long wv = 1, sk = 0;
    for (int i = 0; i < n_c; ++i) {
        if (c[i].kind != RAD_P_INT) continue;
        if      (!std::strcmp(c[i].key, "wv")) wv = c[i].ival;
        else if (!std::strcmp(c[i].key, "sk")) sk = c[i].ival;
    }
    if (sk <= 0) return 1;                       /* the law decides, legally */
    if (wv * sk * 32 > 1024) return 0;

    long long K = 0;
    const char* dt = nullptr;
    for (int i = 0; i < n_p; ++i) {
        if (p[i].kind == RAD_P_INT && !std::strcmp(p[i].key, "K")) K = p[i].ival;
        else if (p[i].kind == RAD_P_STR && !std::strcmp(p[i].key, "dtype")) dt = p[i].sval;
    }
    if (K <= 0) return 1;                        /* K is not pinned in this band: cannot say */
    const long long unit = (dt && !std::strcmp(dt, "bf16")) ? 16 : 128;
    return (K % (sk * unit)) == 0;
}

/* --------------------------------------------------------------- the tiled GEMM's tile ladder
 *
 * One axis, not a cross product: the tiled kernel switches over compiled template instantiations,
 * and only some (BM, BN) pairs have an arm. The values are arm ids, which r4d_cfg_tiled() maps to
 * (BM, BN). Only arms that compute a correct result are listed. */
static const int64_t kTileArms[] = { 0, 6, 10, 12, 57, 59, 114, 124, 125, 126, 128, 129, 138, 139, 146 };
static const RadTunable tGemmTiled[] = {
  { "tile", kTileArms, NELEM(kTileArms), 0,
    "compiled tile arm; 0 is BM 128 x BN 256, libr4d's own default" },
};

/* The prefill kernel's arms, likewise. */
static const int64_t kPrefillArms[] = { 0, 2, 3, 5, 6, 8, 9, 10, 13, 14, 27, 28, 29, 30 };
static const RadTunable tGemmPrefill[] = {
  { "tile", kPrefillArms, NELEM(kPrefillArms), 0,
    "compiled tile arm; 0 is BM 128 x BN 256, libr4d's own default" },
};

/* The MXFP4 tiled GEMM's one axis: compiled arms of one template. 0 is the M crossover (TN=4 from
 * M >= 2048), which needs M. */
static const int64_t kMxTn[] = { 0, 2, 4 };
static const RadTunable tGemmMxfp4Tiled[] = {
  { "tn", kMxTn, NELEM(kMxTn), 0,
    "16-column fragments per wave: 2 = 256x64 block, 4 = 256x128. 0 = TN 4 from M >= 2048" },
};

/* The MXFP4 decode GEMM's axes. 0 on each is the kernel's own rule (r4d_gemm_mxfp4a8_decode.hip);
 * `nt` 1 streams the weight past the caches, 2 loads it plainly. */
static const int64_t kMxDecKs[] = { 0, 1, 2, 4, 8, 16 };
static const int64_t kMxDecBk[] = { 0, 64, 128 };
static const int64_t kMxDecNt[] = { 0, 1, 2 };
static const RadTunable tGemmMxfp4Decode[] = {
  { "ks", kMxDecKs, NELEM(kMxDecKs), 0, "split-K workgroups a column block. 0 = the fill rule" },
  { "bk", kMxDecBk, NELEM(kMxDecBk), 0, "K slab in bytes. 0 = 128, or 64 for a wide unsplit DTM 4" },
  { "nt", kMxDecNt, NELEM(kMxDecNt), 0, "weight loads: 1 non-temporal, 2 plain. 0 = by weight size" },
};

/* ------------------------------------------------------------------------ split-KV attention
 * 0 is libr4d's split law (table in r4d_attn_paged_h256_gqa6.hip), which reads num_seqs and
 * max_ctx at issue. The fixed counts are for a tuner with a concrete batch. RadScratchFn receives
 * the same choice, so the arena is sized for what this picks. */
static const int64_t kSplits[] = { 0, 8, 16, 32, 64 };
static const RadTunable tAttnSplit[] = {
  { "splits", kSplits, NELEM(kSplits), 0, "split-KV segments per sequence. 0 = libr4d's split law" },
};

/* ------------------------------------------------------------------- all-reduce launch geometry
 *
 * Four independent axes of the handshake: nblocks, nthreads, drain and acq. (`pub` is 1 in every
 * arm but the four-rank junction's, tArQ84, where it is an axis.) nblocks 0 scales with the message
 * and clamps; drain 3 is
 * s_wait_storecnt, the correct wait for the fine-grained uncached IPC scratch.
 *
 * rad-tune cannot reach these rows: the two-rank kernels need a real peer over IPC and rad-tune
 * runs one process. A serving engine at --tp 2 can compare points via RADIANCE_TUNE, e.g.
 * `RADIANCE_TUNE=all_reduce:drain=1`. The defaults are the fastest point at both decode and
 * prefill message sizes.
 *
 * On the fp8 fused row only drain and acq apply: it forces `nthreads` to the norm's width for
 * bit-exactness, and pinning `nblocks` would undo r4d_ar_entry.hip's cap. No axis changes the
 * result. */
static const int64_t kArNblocks[]  = { 0, 2, 4, 8, 16, 24 };
static const int64_t kArNthreads[] = { 256, 512, 1024 };
static const int64_t kArDrain[]    = { 1, 3 };
static const int64_t kArAcq[]      = { 0, 1 };
/* `ts_min`: from this many elements the four-rank exact one-shot hands the message to the exact
 * two-shot on the same instance (the same per-element fold, so the same bytes): a rank then sends
 * 1.5x the message instead of 3x, for a second handshake. 32768 is about six rows of 5376, so an
 * 8-row DFlash2 verify's all-reduces take the two-shot too (on 4x R9700, 2026-10-17: 297 -> 303
 * tok/s code, long decode 447 -> 460, against 65536 that kept them one-shot); a 16-row verify's
 * already did. 0 keeps the one-shot to its band's top. Read only by that
 * adapter; declared by every all-reduce table so one RADIANCE_TUNE=all_reduce:ts_min=N pin is legal
 * for every row the op resolves to. */
static const int64_t kArTsMin[]    = { 0, 16384, 32768, 65536, 131072, 262144 };
static const RadTunable tAr[] = {
  { "nblocks",  kArNblocks,  NELEM(kArNblocks),  0, "blocks in the grid. 0 = scale with the message and clamp" },
  { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "threads per block; the fused fp8 row overrides this for bit-exactness" },
  { "drain",    kArDrain,    NELEM(kArDrain),    3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
  { "acq",      kArAcq,      NELEM(kArAcq),      0, "acquire fence after the flag read" },
  { "ts_min",   kArTsMin,    NELEM(kArTsMin),    32768, "elements from which the four-rank one-shot takes the exact two-shot; 0 = never" },
};
/* The four-rank fused junction (ar_rmsnorm_quant_i8_4rank): tAr's axes, its own value sets.
 * On 4x R9700 over PCIe P2P a junction costs about 22 us at one row of 5376 and 57 us at sixteen
 * (--profile-ops): it is bound by its handshakes, not its bytes.
 *
 * Every block takes one handshake, a system-scope release by its first
 * thread and a system-scope acquire by every wave. Both are needed: on 4x R9700 behind an EPYC
 * root complex, dropping either (an acq 2 and a pub 2 that were here, r4d_selftest 2026-10-16 and
 * -17) left the readers with stale scratch on nearly every element, so the peers' stores do not
 * reach them past the writer's L2 without the writeback. What can move is how many handshakes a
 * call takes: fewer blocks (2 or 4, each owning several rows) take fewer. The defaults are the
 * forms that have run on cards; the others are for RADIANCE_TUNE until they have. None of them
 * changes the result. */
static const int64_t kArQ84Tail[]  = { 1, 0 };
static const int64_t kArQ84Nblocks[] = { 0, 2, 4, 8, 16, 24 };
static const int64_t kArQ84Acq[]     = { 0, 1 };
static const int64_t kArQ84Pub[]     = { 1 };
static const RadTunable tArQ84[] = {
  { "nblocks",  kArQ84Nblocks, NELEM(kArQ84Nblocks), 0, "blocks in the grid. 0 = one a row, clamped" },
  { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "unused: the norm's width sets the block for bit-exactness" },
  { "drain",    kArDrain,    NELEM(kArDrain),    3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
  { "acq",      kArQ84Acq,   NELEM(kArQ84Acq),   0, "acquire after the flag read: 0 system invalidate a wave, 1 full fence" },
  { "pub",      kArQ84Pub,   NELEM(kArQ84Pub),   1, "release before the flag store: 1 one system writeback" },
  { "tail",     kArQ84Tail,  NELEM(kArQ84Tail),  1, "from 5 to 32 rows: the standalone two-shot with the junction as its tail, a block a row (no handshake or launch of its own); 0 = the one-shot" },
};
/* The four-rank two-shots beside compute: four blocks of 1024. A pipelined Flash-Next prefill at
 * TP4 (qwen4exp_fp8) runs each all-reduce of a 2048-row piece beside the other piece's MoE passes,
 * where the size-chosen 24 blocks hold a third of the card while they wait on the link. With every
 * all-reduce pinned, 4 blocks beat 8 by about 1.6 % at 18K-64K tokens, and 8 beat 16 and 24. */
static const int64_t kArNblocksLane4[] = { 0, 4, 8, 16, 24 };
static const RadTunable tArTs4Lane[] = {
  { "nblocks",  kArNblocksLane4, NELEM(kArNblocksLane4), 4, "blocks in the grid. 0 = scale with the message and clamp" },
  { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "threads per block" },
  { "drain",    kArDrain,    NELEM(kArDrain),    3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
  { "acq",      kArAcq,      NELEM(kArAcq),      0, "acquire fence after the flag read" },
  { "ts_min",   kArTsMin,    NELEM(kArTsMin),    32768, "unused here (tAr says why it is declared)" },
};
/* The same axes with narrow defaults, for a collective issued beside compute (`overlap` 1). The
 * wide launch (up to 24 blocks of 1024 threads spinning on peer flags) suits an all-reduce that has
 * the card to itself; beside a GEMM it slows the GEMM by 1.45-2.1x. Eight blocks of 256 threads
 * still keep the link busy and were fastest on Qwen-Image's two-rank pass (ahead of 8 x 512 and
 * 24 x 256). */
static const RadTunable tArLane[] = {
  { "nblocks",  kArNblocks,  NELEM(kArNblocks),  8, "blocks in the grid. 0 = scale with the message and clamp" },
  { "nthreads", kArNthreads, NELEM(kArNthreads), 256, "threads per block" },
  { "drain",    kArDrain,    NELEM(kArDrain),    3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
  { "acq",      kArAcq,      NELEM(kArAcq),      0, "acquire fence after the flag read" },
  { "ts_min",   kArTsMin,    NELEM(kArTsMin),    32768, "unused here (tAr says why it is declared)" },
};

/* --------------------------------------------------------------------------------- row top-k
 * NCH is a request (libr4d rounds the chunk width up to a multiple of eight columns), so the
 * scratch hook asks r4d_rowtopk_bf16_chunks(). PT1 is the stage-1 partial depth in waves; the
 * entry point compiles 2, 4, 8 and 16. `rc`, the candidates a chunk keeps, is bounded by pt1*32
 * and derived. */
static const int64_t kTopkNch[] = { 0, 1024, 2048, 4096 };
static const int64_t kTopkPt1[] = { 2, 4, 8, 16 };
static const RadTunable tTopk[] = {
  { "nch", kTopkNch, NELEM(kTopkNch), 0, "requested chunk width in columns. 0 = derive from R and N at issue" },
  { "pt1", kTopkPt1, NELEM(kTopkPt1), 8, "stage-1 partial depth in waves" },
};

/* ------------------------------------------------------------------------------- fp8 matvec
 * Rows per wave. 0 is libr4d's own rule. 8 is offered though the rule never returns it: `rows_dot`
 * is templated and 8 divides 128, and the rule's ceiling of 4 was not derived from the served
 * models' shapes. */
static const int64_t kMvRb[] = { 0, 1, 2, 4, 8 };
static const RadTunable tMv[] = {
  { "rb", kMvRb, NELEM(kMvRb), 0, "rows of the weight one wave owns. 0 = libr4d's own rule" },
};

/* rmsnorm_quant_i8 at decode heights (1024 threads, n <= 8192): 1 loads every operand of the row
 * into registers before the first reduction (rmsnorm_quant_i8_reg_kernel), 0 re-reads the row in
 * each pass. The same bytes either way; on 4x R9700 the first is 8.2 us a launch, the second 11. */
static const int64_t kRmsQ8Reg[] = { 1, 0 };
static const RadTunable tRmsQ8[] = {
  { "reg", kRmsQ8Reg, NELEM(kRmsQ8Reg), 1, "decode heights: 1 = the row's operands loaded before the reductions, 0 = re-read each pass" },
};

/* ==================================================================================== the rows */

/* ------------------------------------------- the small model ops and the sampler chain -------
 *
 * A constraint is a necessary condition on the geometry (spec §2.1), and these ops have almost
 * none. The kernels require the activation width (they read through a uint16_t), so `dtype` is
 * the row; the shims also check operand dtypes per call.
 *
 * kv_store's `block_size >= 1`, as in ref, admits every value and refuses only a query that names
 * none. The paged block size is derived from the resolved attention kernel (spec §7.2): 16.
 *
 * The sampler rows constrain only the extent, so they cannot match a query that named nothing (as
 * cSampleChain). n_cand is bounded because the top-k's bitonic sort runs in LDS over a
 * power-of-two padding. */
static const RadConstraint cBf16Row[]   = { C_IN("dtype", "bf16") };
static const RadConstraint cF32Row[]    = { C_IN("dtype", "f32") };
/* bf16 rows over the byte copy, which moves four bytes a thread: an even width. */
static const RadConstraint cBf16Row2[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
/* Opaque byte rows, four at a time: the vectorised copy's alignment, stated so the selector
 * refuses an unservable row at resolve rather than at launch. */
static const RadConstraint cU8Row4[]    = { C_IN("dtype", "u8"), C_DIV("n", 4) };
/* The n-gram hash reads no activation, so no dtype. The shift window lives in a register array;
 * `ngram` past 8 would spill it to scratch. */
static const RadConstraint cNgramIds[]  = { C_GE("ngram", 2), C_LE("ngram", 8), C_GE("heads", 1) };
/* PLE's gate. `n` is the per-stream width every reduction runs over within a workgroup; 4096 is
 * 16 values per thread at 256 threads, what the register arrays hold. */
static const RadConstraint cPleGate[]   = { C_IN("dtype", "bf16"), C_GE("n", 1), C_LE("n", 4096),
                                            C_GE("hc", 2) };
/* PLE's convolution. The kernel bounds (width-1)*dilation, the state write-back's register
 * window, at 16; a constraint cannot express a product, so these are the tightest separate caps
 * admitting every combination under it. */
static const RadConstraint cPleConv[]   = { C_IN("dtype", "bf16"), C_GE("width", 1),
                                            C_LE("width", 5), C_GE("dilation", 1),
                                            C_LE("dilation", 4) };
/* The same kernel undilated: the window is width-1 deep, and the taps are held in registers. */
static const RadConstraint cCausalConv[] = { C_IN("dtype", "bf16"), C_GE("width", 2),
                                             C_LE("width", 8) };
static const RadConstraint cKvStore[]   = { C_GE("block_size", 1), C_IN("kv_dtype", "bf16") };
static const RadConstraint cKvStoreFp8[] = { C_GE("block_size", 1), C_IN("kv_dtype", "fp8_e4m3") };
static const RadConstraint cKvStoreFp8v4[] = { C_GE("block_size", 1), C_DIV("head_dim", 32),
                                               C_IN("kv_dtype", "fp8_v4") };
static const RadConstraint cKvStoreK6v4[] = { C_GE("block_size", 1), C_DIV("head_dim", 32),
                                              C_IN("kv_dtype", "k6_v4") };
static const RadConstraint cRopeRow[]   = { C_GE("head_dim", 1),
                                            C_IN("mode", "neox mrope gptj imrope axial") };
/* The vision tower's rows. The GEMM stages 16-byte chunks, so K is a multiple of 8; the norm and
 * the position embedding take any width. */
static const RadConstraint cLayernorm[] = { C_GE("n", 1), C_IN("dtype", "bf16") };
static const RadConstraint cGridEmbed[] = { C_GE("n", 1), C_GE("side", 1), C_IN("dtype", "bf16") };
static const RadConstraint cGemmBias[]  = { C_GE("M", 1), C_DIV("K", 8), C_IN("dtype", "bf16") };
/* The f32 GEMM reads its rows a float at a time: any K. */
static const RadConstraint cGemmBiasF32[] = { C_GE("M", 1), C_IN("dtype", "f32") };
static const RadConstraint cSampleVoc[] = { C_GE("n_vocab", 1) };
static const RadConstraint cSampleCand[] = { C_GE("n_cand", 1), C_LE("n_cand", 4096) };
static const RadConstraint cSampleLogprobs[] = { C_GE("n_vocab", 1), C_GE("k", 1), C_LE("k", 64) };
/* The activation width the column loop vectorises over; M is unbounded -- the kernel chunks rows. */
static const RadConstraint cLogitsGemm[] = { C_DIV("n_embd", 8), C_IN("dtype", "bf16") };
/* The fp8 head: n_embd on the 128 scale block, and no cap on M. The row relayouts the weight, so
 * it must cover the whole range; above a cap the bf16 projection would read the E4M3 codes as bf16.
 * The kernel chunks above 32, and core/build/rad_builder.cpp refuses a capped arrangement.
 *
 * A row's constraints must match what its layout can serve. r4d_rad_layout_fp8_logits stores a
 * vocabulary row as `n_embd` codes plus its scales, read sixteen codes at a time through a uint4.
 * Unpadded, K = 128j gives a row of 130j bytes, 16-aligned only when K % 1024 == 0, so at n_embd
 * 2560 the row would win selection and then fail at layout with no fallback. r4d_fp8_lm.h pads the
 * stored row to 16 bytes instead (under 0.3% at K = 2560), so every multiple of 128 is served. */
static const RadConstraint cLogitsGemmFp8[] = { C_DIV("n_embd", 128), C_IN("dtype", "bf16") };

/* No tunable axes. */
#define NOTUNE nullptr, 0, nullptr

/* ---- image generation ------------------------------------------------------------------- */
/* Rows read 16 bytes at a time, so the widths are whole eights. */
static const RadConstraint cGenRows8[] = { C_IN("dtype", "bf16"), C_DIV("n", 8) };
static const RadConstraint cGenBf16[] = { C_IN("dtype", "bf16") };
/* A depth decoder's attention: a frame's handful of positions, one query head a key head. */
static const RadConstraint cAttnDenseSmallH256[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 1), C_IN("dtype", "bf16"), C_LE("max_seqlen", 32),
};
static const RadConstraint cAudioAny[] = { C_IN("dtype", "bf16 f32") };
/* ---- video generation (r4d_video.hip). The modulated norms keep a row's 8-wide chunks in
 * registers, four a thread (n <= 8192), and read 16 bytes at a time; the lists are kernel
 * arguments, 64 floats at most; the alias-free activation stages its filters in 32 floats of
 * LDS. */
static const RadConstraint cRmsMod[] = { C_IN("dtype", "bf16"), C_DIV("n", 8), C_LE("n", 8192) };
static const RadConstraint cRmsModQ[] = { C_IN("dtype", "bf16"), C_DIV("n", 128), C_LE("n", 8192),
                                          C_EQ("group", 128) };
static const RadConstraint cGenF32[] = { C_IN("dtype", "f32") };
static const RadConstraint cPatchUnfold[] = { C_IN("dtype", "f32"), C_LE("C", 64) };
static const RadConstraint cGenAnyFloat[] = { C_IN("dtype", "bf16 f32") };
static const RadConstraint cSnakeAa[] = { C_IN("dtype", "f32"), C_LE("k", 32) };
/* ---- a video latent upscaler (r4d_upscaler.hip). The convolution decodes 3 x 3 x 3 taps as
 * constants and reads 16-byte chunks of a voxel row; the group norm puts a thread on chunks of 8, 4
 * or 2 channels inside one group, a row's chunks spread over at most the 256-thread block (the
 * shim checks an even group width), and merges at most 256 groups; the temporal convolution keeps
 * its 5-frame window in registers; the fold's lists are kernel arguments, 64 floats at most. */
static const RadConstraint cConv3d[] = { C_IN("dtype", "bf16"), C_EQ("k", 3), C_EQ("kt", 3),
                                         C_DIV("cin", 8) };
static const RadConstraint cGroupNorm[] = { C_IN("dtype", "bf16 f32"), C_DIV("n", 8),
                                            C_LE("n", 8192), C_LE("groups", 256) };
static const RadConstraint cTconvDw[] = { C_IN("dtype", "bf16"), C_EQ("k", 5), C_DIV("n", 8) };
static const RadConstraint cResizeBil[] = { C_IN("dtype", "bf16 f32"), C_DIV("n", 8) };
static const RadConstraint cPatchFold[] = { C_IN("dtype", "bf16 f32"), C_LE("C", 64) };
static const RadConstraint cLnModQ[] = { C_IN("dtype", "bf16"), C_DIV("n", 128), C_LE("n", 8192),
                                         C_EQ("group", 128) };
/* The lane copies move 32-bit words, so any row of whole words: bf16 latents and an f32 mask. */
static const RadConstraint cGenLanes[] = { C_IN("dtype", "bf16 f32") };
/* gelu_quick: x and y of one dtype, any width and pitch. */
static const RadConstraint cGeluQuick[] = { C_IN("dtype", "bf16 f32") };
/* The two lists are kernel arguments, 64 floats each at most (r4d_gen.hip). */
static const RadConstraint cGenAffine[] = { C_IN("dtype", "bf16"), C_LE("n", 64) };
static const RadConstraint cGenAffineF32[] = { C_IN("dtype", "f32"), C_LE("n", 64) };
static const RadConstraint cGenTimestep[] = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cGenTimestepF32[] = { C_IN("dtype", "f32"), C_DIV("n", 2) };
/* The paged prefix at libr4d's block of 16, both caches bf16, up to eight query heads a KV head
 * (the shim takes 1, 2, 4 and 8). */
static const RadConstraint cAttnPrefixSelf[] = {
    C_EQ("head_dim", 128), C_LE("gqa", 8), C_EQ("block_size", 16), C_IN("q_dtype", "bf16"),
    C_IN("kv_dtype", "bf16"),
};
static const RadConstraint cAttnDenseH128[] = {
    C_EQ("head_dim", 128), C_LE("gqa", 8), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnCrossH64[] = {
    C_EQ("head_dim", 64), C_LE("gqa", 8), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnNa3d[] = { C_EQ("head_dim", 64), C_IN("dtype", "bf16 f32") };
static const RadConstraint cAttnDenseH384[] = {
    C_EQ("head_dim", 384), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH512[] = {
    C_EQ("head_dim", 512), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH768[] = {
    C_EQ("head_dim", 768), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16"),
};
static const RadConstraint cAttnDenseH1152[] = {
    C_EQ("head_dim", 1152), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16"),
};
/* Video sparse attention: head_dim 128 over 4 x 4 x 4 tiles. The dtypes, the tile capacity and
 * the optional pairs (v with coarse, coarse with gate) are the entry points' checks. */
static const RadConstraint cVsa[] = {
    C_EQ("head_dim", 128), C_EQ("tile_t", 4), C_EQ("tile_h", 4), C_EQ("tile_w", 4),
    C_GE("patch", 1),
};
/* A thread quantises eight columns: one 16-byte load and one 8-byte store a row. */
static const RadConstraint cQuantCols[] = { C_DIV("n", 8), C_IN("dtype", "bf16") };

static const RadKernelInfo kKernels[] = {
/* ---------------------------------- the small model ops (r4d_model_bf16.hip) ----------------
 *
 * The ops between the larger kernels. They need device rows because the engine's buffers live in
 * VRAM, libref is host domain, and `rad_issue` refuses an op with no launch for its placed domain:
 * a missing device add is a fatal "no device kernel for this band".
 *
 * R4D_PRIO_ONLY throughout: nothing else in this plugin implements them.
 */
{ "embed_lookup_bf16", "embed_lookup", "vocab",
  "embedding gather, one workgroup a token",
  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
  "i32 tokens, bf16 table and output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup, nullptr, nullptr, nullptr,
  r4d_shape_embed_lookup },

{ "embed_lookup_q_bf16", "embed_lookup_q", "vocab",
  "quantised embedding gather, four entries a thread",
  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
  "i32 tokens, an E4M3 table with a bf16 or f32 scale or a plain bf16 one, bf16 output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup_q, nullptr, r4d_rad_layout_embed_q, nullptr,
  r4d_shape_embed_lookup_q },

{ "ple_gate_bf16", "ple_gate", "norm",
  "PLE's dot-product gate, one workgroup a (row, stream) pair",
  "n up to 4096 -- every reduction is over one stream's n and lives in a workgroup",
  "bf16 activations, F32 norm gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPleGate), NOTUNE,
  nullptr, nullptr, r4d_rad_ple_gate },

{ "ple_conv_bf16", "ple_conv", "norm",
  "PLE's dilated depthwise conv, silu and residual add; one thread a channel",
  "width up to 5 and dilation up to 4 -- (width-1)*dilation is a register window",
  "bf16 activations, weights and conv state; i32 cu, slots, has_init and num_accepted",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPleConv), NOTUNE,
  nullptr, nullptr, r4d_rad_ple_conv },

{ "causal_conv1d_bf16", "causal_conv1d", "conv",
  "depthwise causal conv with no activation (ple_conv's kernel); one thread a channel",
  "width 2 to 8, undilated",
  "bf16 activations, weights and conv state; i32 cu, slots, has_init and num_accepted",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCausalConv), NOTUNE,
  nullptr, nullptr, r4d_rad_causal_conv1d },

{ "ngram_ids_i32", "ngram_ids", "vocab",
  "PLE's hashed n-gram row ids, one thread a token",
  "any heads and n-gram size up to 8; heads must divide into ngram-1 equal blocks",
  "i32 tokens and state, i64 multipliers and per-head vocab sizes and offsets",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cNgramIds), NOTUNE,
  nullptr, nullptr, r4d_rad_ngram_ids },

{ "gather_rows_bf16", "gather_rows", "elem",
  "y[i, :] = x[idx[i], :], one workgroup a row",
  "any n; source and destination row pitches are read off the operands",
  "i32 indices, bf16 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_gather_rows, nullptr, nullptr, nullptr,
  r4d_shape_gather_rows },

/* The byte forms, separate rows rather than a widened bf16 dtype. They move KV fragments (one
 * layer of one block) with arbitrary contents. libref's bf16 path loads through float, which need
 * not preserve a NaN payload; at u8 float carries every value exactly, so oracle and kernel agree
 * bit for bit. */
{ "gather_rows_u8", "gather_rows", "elem",
  "y[i, :] = x[idx[i], :] over opaque bytes, one workgroup a row, 4 bytes a thread",
  "n a multiple of 4; source and destination row pitches are read off the operands",
  "i32 indices, u8 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cU8Row4), NOTUNE,
  nullptr, nullptr, r4d_rad_gather_rows_u8, nullptr, nullptr, nullptr,
  r4d_shape_gather_rows_u8 },

{ "scatter_rows_u8", "scatter_rows", "elem",
  "x[idx[i], :] = v[i, :] over opaque bytes, one workgroup a row, 4 bytes a thread",
  "n a multiple of 4; distinct indices; the destination pitch is read off the operand",
  "i32 indices, u8 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cU8Row4), NOTUNE,
  nullptr, nullptr, r4d_rad_scatter_rows_u8, nullptr, nullptr, nullptr,
  r4d_shape_scatter_rows_u8 },

{ "scatter_rows_bf16", "scatter_rows", "elem",
  "x[idx[i], :] = v[i, :] for bf16 rows -- the u8 row's byte copy at twice the width",
  "any n; distinct indices; the destination pitch is read off the operand",
  "i32 indices, bf16 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row2), NOTUNE,
  nullptr, nullptr, r4d_rad_scatter_rows_bf16, nullptr, nullptr, nullptr,
  r4d_shape_scatter_rows_bf16 },

{ "cast_bf16_bf16", "cast", "elem",
  "y = x, bf16 to bf16 -- a strided row copy through libr4d's 2D copy",
  "any M and n; both row pitches are read off the operands, so a COLUMN SLICE of a wider "
  "destination is expressible and is what an intermediate-hidden-state tap writes into",
  "bf16 in, bf16 out, byte-for-byte",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCastBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_cast, nullptr, nullptr, nullptr,
  r4d_shape_cast },

{ "cast_f32_f32", "cast", "elem",
  "y = x, f32 to f32 -- a strided row copy through libr4d's 2D copy",
  "any M and n; both row pitches are read off the operands",
  "f32 in, f32 out, byte-for-byte",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCastF32), NOTUNE,
  nullptr, nullptr, r4d_rad_cast, nullptr, nullptr, nullptr,
  r4d_shape_cast },

{ "cast_bf16_f32", "cast", "elem",
  "y = x, bf16 widened to f32",
  "any M and n; both row pitches are read off the operands",
  "bf16 in, f32 out, exact",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCastWiden), NOTUNE,
  nullptr, nullptr, r4d_rad_cast_widen, nullptr, nullptr, nullptr,
  r4d_shape_cast },

{ "cast_bf16_f16", "cast", "elem",
  "y = x, bf16 narrowed to f16",
  "any M and n; both row pitches are read off the operands",
  "bf16 in, f16 out, rounded to nearest even (a value past f16's range becomes an infinity)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCastF16), NOTUNE,
  nullptr, nullptr, r4d_rad_cast_f16, nullptr, nullptr, nullptr,
  r4d_shape_cast },

{ "cast_f32_bf16", "cast", "elem",
  "y = x, f32 narrowed to bf16",
  "any M and n; both row pitches are read off the operands",
  "f32 in, bf16 out, rounded to nearest even",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCastNarrow), NOTUNE,
  nullptr, nullptr, r4d_rad_cast_bf16, nullptr, nullptr, nullptr,
  r4d_shape_cast },

{ "act_mul_bf16", "act_mul", "elem",
  "gated activation over two operands: act(a) rounded, times b",
  "any n; a, b and y each at their own row pitch, so both may be column slices of one buffer",
  "bf16 in and out, f32 arithmetic, libm silu / erf / tanh",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_act_mul, nullptr, nullptr, nullptr,
  r4d_shape_binary },

{ "seg_mean_bf16", "seg_mean", "elem",
  "mean pooling: one row a cu_seqlens segment",
  "any n; 32 columns by 8 row lanes a workgroup, one workgroup row a segment",
  "bf16 in and out, f64 sum, one rounding",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_seg_mean },

{ "qk_norm_rope_bf16", "qk_norm_rope", "attn",
  "per-head RMS norm of q and k (and optionally v, unit gain) with rotary on q and k, in place",
  "head_dim up to 1024, a multiple of 32 for neox (rotary_dim / 2 a multiple of 32) and gptj; "
  "blocks at any even head_dim and section size (sections of 16 or a multiple of 32 pairs, each "
  "on a 32-channel slice, in registers; others a lane a pair); one workgroup a token and a wave "
  "a head, sin/cos computed once for all its heads",
  "bf16 qkv, bf16 or f32 gains, i32 positions, f32 arithmetic, the norm rounded before the "
  "rotation",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRope), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope },

{ "relu_bf16", "relu", "elem",
  "max(x, 0)",
  "any n; both row pitches read off the operands",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_relu, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "silu_bf16", "silu", "elem",
  "x * sigmoid(x), the sigmoid rounded nowhere: libref's act_silu",
  "any n; both row pitches read off the operands",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_silu, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "silu_f32", "silu", "elem",
  "x * sigmoid(x) over f32 rows, flat over the extent",
  "any n; both row pitches read off the operands",
  "f32 in, f32 out or bf16 rounded once",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_silu, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "gelu_bf16", "gelu", "elem",
  "0.5 x (1 + erf(x / sqrt 2)), the exact form: libref's act_gelu",
  "any n; both row pitches read off the operands",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_gelu, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "clamp_bf16", "clamp", "elem",
  "x clamped to two one-element weights: a clipped linear's range",
  "any n; bf16 or f32 bounds",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_clamp },

{ "conv2d_bf16", "conv2d", "elem",
  "segment-wise 2-D convolution over (time, width), channels innermost, optional row mask, bias "
  "and gelu, either kernel layout",
  "any geometry; a workgroup an output time row, a thread an output element; sized for an audio "
  "subsampler (a few thousand frames, kernels of 3)",
  "bf16 x / w / y, i32 cu_seqlens and mask, f32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_conv2d },

{ "conv2d_unfold_bf16", "conv2d_unfold", "elem",
  "a convolution's receptive fields as rows, for a GEMM against its weight",
  "any geometry; a workgroup an output time row, a thread an unfolded element",
  "bf16 x / y, i32 cu_seqlens and mask",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_conv2d_unfold, nullptr, nullptr, nullptr },

{ "conv1d_dw_bf16", "conv1d_dw", "elem",
  "segment-wise causal depthwise convolution over time",
  "any channels and taps; a thread a (row, channel)",
  "bf16 x / w / y, i32 cu_seqlens, f32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_conv1d_dw },

{ "rel_sinusoid_bf16", "rel_sinusoid", "elem",
  "the sinusoidal relative-position table",
  "any rows and even width; a workgroup a row",
  "bf16 out, f32 libm sin / cos / exp",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_rel_sinusoid },

{ "seg_sinusoid_bf16", "seg_sinusoid", "elem",
  "sinusoidal absolute positions restarting every `period` rows of a segment (Whisper-style "
  "chunk positions)",
  "any rows and even width; a workgroup a row, its segment by a scan over cu_seqlens",
  "bf16 out, i32 cu_seqlens, f32 libm sin / cos / exp",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_seg_sinusoid, nullptr, nullptr, nullptr },

{ "attn_local_rel_bf16", "attn_local_rel", "attn",
  "local attention with a relative-position term and a softcap (Gemma 4's audio)",
  "any head_dim and heads, left <= 63, right 0; a workgroup a (row, head), each logit a "
  "workgroup sum",
  "bf16 q / k / v / relk / out, bf16 or f32 per-dim scale, f32 softmax",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnLocalRel), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_local_rel },

{ "pool_rows_bf16", "pool_rows", "elem",
  "mean of every k consecutive rows, then a multiplier: a vision pooler over windows its "
  "processor laid out consecutively",
  "any n and k; a workgroup an output row, a thread a column",
  "bf16 in and out, f32 sum, two roundings",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_pool_rows },

{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 y, x bf16 or (with a gain) f32, bf16 or f32 gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_rmsnorm },

{ "rmsnorm_p_bf16", "rmsnorm_p", "norm",
  "weightless root-mean-square norm (no gain), two passes over the row",
  "any n; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x / y, fp32 reduction",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_rmsnorm_p, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "layernorm_bf16", "layernorm", "norm",
  "layer norm with a gain and an optional bias, two passes over the row",
  "any n; mean and variance accumulate in f32 and only the store narrows",
  "bf16 or f32 x, bf16 y, f32 or bf16 gain and bias",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLayernorm), NOTUNE,
  nullptr, nullptr, r4d_rad_layernorm, nullptr, nullptr, nullptr,
  r4d_shape_layernorm },

{ "grid_embed_bf16", "grid_embed", "elem",
  "a learned position grid resampled bilinearly to each row's grid and added in",
  "any n and side; coord [4, M] dense; one workgroup a row",
  "bf16 x, bf16 or f32 table, i32 coord, f32 taps",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGridEmbed), NOTUNE,
  nullptr, nullptr, r4d_rad_grid_embed, nullptr, nullptr, nullptr,
  r4d_shape_grid_embed },

{ "grid_embed_aa_bf16", "grid_embed_aa", "elem",
  "a learned position grid resampled as torch's antialiased bilinear and added in (SigLIP2)",
  "side up to 1024, any n; coord [4, M] dense; one workgroup a row, its taps in LDS",
  "bf16 x, bf16 or f32 table, i32 coord, f32 taps",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGridEmbed), NOTUNE,
  nullptr, nullptr, r4d_rad_grid_embed_aa, nullptr, nullptr, nullptr,
  r4d_shape_grid_embed },

{ "gemm_bf16_nt_bias_tiled", "gemm_nt_bias", "gemm",
  "biased GEMM C = res + act(A @ W^T + bias) over 64-deep K tiles, the K tail staged as zeros, "
  "the activation and residual in the store: from 32 blocks of 128x128 the weight read straight "
  "into registers and the activation staged through LDS, below that both staged at 64x64",
  "any M, N; K a multiple of 8; A rows 16-byte aligned; `res` may be the output itself",
  "bf16 A / W / C / res, bf16 or f32 bias, fp32 accumulate, three RNE roundings in the store",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGemmBias), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_bias, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt_bias },

{ "gemm_f32_nt_bias", "gemm_nt_bias", "gemm",
  "biased f32 GEMM C = res + act(A @ W^T + bias): a register-blocked FMA tile of 128 rows by 128, "
  "96, 64 or 32 columns over LDS-staged 16-deep K slabs, the activation and residual in the store",
  "any M, N and K; A and C at their own row pitches; `res` may be the output itself",
  "f32 A / W, f32 or bf16 bias, f32 accumulate, C f32 or bf16 with res in C's dtype (each epilogue "
  "step rounded to C's dtype)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGemmBiasF32), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_f32_nt_bias, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt_bias },

{ "add_bf16", "add", "elem",
  "elementwise sum, with the one row-broadcast form; `div` divides the rounded sum",
  "any n; `b` with exactly n elements broadcasts along rows and nothing else does",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_add, nullptr, nullptr, nullptr,
  r4d_shape_binary },

{ "add_f32", "add", "elem",
  "elementwise f32 sum, the same broadcast rule and `div`, flat over the extent so a narrow row "
  "over a million rows keeps every lane busy",
  "any M and n; `y` may be `a` (an in-place residual)",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_add, nullptr, nullptr, nullptr,
  r4d_shape_binary },

{ "mul_bf16", "mul", "elem",
  "elementwise product, with the one row-broadcast form",
  "any n; the same broadcast rule as add",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_mul, nullptr, nullptr, nullptr,
  r4d_shape_binary },

{ "mul_weight_bf16", "mul_weight", "elem",
  "activation times a learned scalar or one-row gain",
  "any n; the weight is one element or exactly n",
  "bf16 in and out, a bf16 or f32 weight, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_mul_weight, nullptr, nullptr, nullptr,
  r4d_shape_mul_weight },

{ "mul_weight_f32", "mul_weight", "elem",
  "activation times a learned scalar or one-row gain into an f32 output (a layer scale on a "
  "branch joining an f32 residual)",
  "any n; the weight is one element or exactly n",
  "bf16 or f32 in, a bf16 or f32 weight, f32 out, the product rounded once",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_mul_weight, nullptr, nullptr, nullptr,
  r4d_shape_mul_weight },

{ "fill_bf16", "fill", "elem",
  "constant fill, a row a workgroup",
  "any M, n; a row pitch is honoured",
  "bf16 out; the value rounds to nearest even once",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_fill, nullptr, nullptr, nullptr,
  r4d_shape_fill },

{ "fill_f32", "fill", "elem",
  "constant fill, a row a workgroup",
  "any M, n; a row pitch is honoured",
  "f32 out, the value exactly",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_fill, nullptr, nullptr, nullptr,
  r4d_shape_fill },

{ "sigmoid_bf16", "sigmoid", "elem",
  "logistic sigmoid, grid-strided over the whole extent",
  "any extent; contiguous in and out",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_sigmoid, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "logit_softcap_bf16", "logit_softcap", "elem",
   "cap * tanh(x / cap), the final-logit softcap, a row spread over up to 64 blocks",
   "any extent; contiguous in and out",
   "bf16 in and out, fp32 arithmetic",
   RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
   ROW(cBf16Row), NOTUNE,
   nullptr, nullptr, r4d_rad_logit_softcap, nullptr, nullptr, nullptr,
   r4d_shape_unary },

{ "logit_softcap_f32", "logit_softcap", "elem",
   "cap * tanh(x / cap), the final-logit softcap, a row spread over up to 64 blocks",
   "any extent; contiguous in and out",
   "f32 in and out — the 1.3.0 sampling chain carries the logits as f32 from the gemm",
   RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
   ROW(cF32Row), NOTUNE,
   nullptr, nullptr, r4d_rad_logit_softcap, nullptr, nullptr, nullptr,
   r4d_shape_unary },

{ "scale_bf16", "scale", "elem",
   "x * s, s a [1] weight broadcast over the whole extent, grid-strided",
   "any extent; contiguous in and out",
   "bf16 in and out, fp32 arithmetic, s f32 or bf16",
   RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
   ROW(cBf16Row), NOTUNE,
   nullptr, nullptr, r4d_rad_scale, nullptr, nullptr, nullptr,
   r4d_shape_scale },

{ "mla_attn_matrix_bf16", "mla_attn", "attn", "head-tiled WMMA latent attention with FP32 softmax",
  "latent_dim=512, heads a multiple of 8 (heads%16==8 pairs two requests a tile), block_size=4",
  "bf16 operands; FP32 probabilities via high/low BF16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY + 1, ROW(cMlaMatrix), NOTUNE,
  nullptr, nullptr, r4d_rad_mla_attn_matrix, r4d_rad_mla_attn_matrix_scratch, nullptr, nullptr,
  r4d_shape_mla_attn },
{ "mla_attn_bf16", "mla_attn", "attn", "absorbed MLA with a shared latent cache and split history",
  "latent_dim=512, heads<=64, block_size=4", "bf16 Q/cache/output, f32 softmax and partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMlaAttn), NOTUNE,
  nullptr, nullptr, r4d_rad_mla_attn, r4d_rad_mla_attn_scratch, nullptr, nullptr,
  r4d_shape_mla_attn },
{ "gemm_fp8a8_f32scale", "gemm_nt_q", "gemm", "FP8 WMMA with canonical F32 block scales",
  "N and K multiples of 128; pitched activations and row-major weights", "E4M3 codes, F32 scales, BF16 output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cFp8F32Scale), NOTUNE,
  nullptr, nullptr, r4d_rad_fp8_f32scale, nullptr, r4d_layout_fp8_f32scale, nullptr,
  nullptr },
{ "moe_gemm_fp8a8_f32scale", "moe_gemm_q", "moe", "grouped FP8 WMMA with canonical F32 block scales",
  "equal parity widths; token/sorted activations; canonical pointer tables", "E4M3 codes, F32 scales, BF16 output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cFp8F32Scale), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_fp8_f32scale, nullptr, r4d_layout_fp8_f32scale, nullptr,
  nullptr },
{ "dsa_score_fp8", "dsa_score", "attn", "32-head FP8 WMMA index scoring over paged keys",
  "n=128, heads=32; ragged sequences and pitched tensors", "E4M3 codes, f32 scales and gates",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDsaScore), NOTUNE,
  nullptr, nullptr, r4d_rad_dsa_score, nullptr, nullptr, nullptr,
  r4d_shape_dsa_score },
{ "mla_project_bf16", "mla_project", "attn", "per-head absorbed MLA projections",
  "head_dim=256, latent_dim=512; pitched tensors", "bf16 input, weights and output; fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMlaProject), NOTUNE,
  nullptr, nullptr, r4d_rad_mla_project, nullptr, nullptr, nullptr,
  r4d_shape_mla_project },
{ "dsa_query_fp8", "dsa_query", "attn", "normalized index queries in wave-local registers",
  "n=128; flattened heads and pitched rows", "bf16 input, E4M3 codes and power-of-two f32 scale",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDsaQuery), NOTUNE,
  nullptr, nullptr, r4d_rad_dsa_query, nullptr, nullptr, nullptr,
  r4d_shape_dsa_query },
{ "gemm_nt_f32_bf16", "gemm_nt_f32", "gemm", "F32 routing and index gate projection",
  "N<=512, K<=16384; pitched rows", "bf16 input, f32 weights and output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cGemmNtF32), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_f32, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt_f32 },
{ "dsa_pool_fp8", "dsa_pool", "attn", "learned compressed index keys in wave-local registers",
  "n=128, ratio=4; pitched rows", "bf16 keys/gates, f32 position bias; FP8 codes and f32 scale",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDsaPool), NOTUNE,
  nullptr, nullptr, r4d_rad_dsa_pool, nullptr, nullptr, nullptr,
  r4d_shape_dsa_pool },
{ "swiglu_clamped_bf16", "swiglu_clamped", "elem",
  "asymmetrically clipped SwiGLU", "positive finite limit; pitched rows",
  "bf16 in and out, fp32 arithmetic", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_swiglu_clamped, nullptr, nullptr, nullptr,
  r4d_shape_silu_mul },
{ "swiglu_clamped_quant_fp8", "swiglu_clamped_quant_fp8", "quant",
  "asymmetrically clipped SwiGLU and the fp8 activation quantiser in one pass",
  "positive finite limit; n a multiple of 4; the gated operand is [M][2n], gate then up; "
  "group 128; pitched rows",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns) and an optional bf16 out; the codes "
  "are taken over the BF16-ROUNDED product",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSwigluClampedQuant), NOTUNE,
  nullptr, nullptr, r4d_rad_swiglu_clamped_quant_fp8, nullptr, nullptr, nullptr,
  r4d_shape_gated_quant_fp8, nullptr,
  /* `swiglu_clamped` + `quant_act_fp8` (r4d_fuse.cpp). */
  r4d_fuse_swiglu_clamped_quant_fp8 },
{ "silu_mul_bf16", "silu_mul", "elem",
  "silu-gated product over a fused [M, 2n] gate_up",
  "any n; the gate is the first half",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_silu_mul, nullptr, nullptr, nullptr,
  r4d_shape_silu_mul },

{ "gelu_mul_bf16", "gelu_mul", "elem",
  "gelu-gated product over a fused [M, 2n] gate_up",
  "any n; the gate is the first half",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_gelu_mul, nullptr, nullptr, nullptr,
  r4d_shape_silu_mul },

{ "rope_bf16", "rope", "attn",
  "rotary embedding in place, one thread a rotated pair",
  "any head_dim; rotary_dim even and no wider than head_dim -- the frequency exponent divides by "
  "rotary_dim, so a partial rotary is a complete rotation of its own width; neox, mrope (which is "
  "neox over one position component) and gptj pairings",
  "bf16 qkv, i32 positions, fp32 angles through the accurate sincosf",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRopeRow), NOTUNE,
  nullptr, nullptr, r4d_rad_rope, nullptr, nullptr, nullptr,
  r4d_shape_rope },

{ "kv_store_bf16", "kv_store", "attn",
  "scatter k and v into the paged cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim]; a negative or out-of-range slot "
  "stores nothing",
  "bf16 k / v / cache, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cKvStore), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_kv_store },

/* The write side of the fp8kv attention family. Four columns per thread, so head_dim must be a
 * multiple of 4. */
{ "kv_store_fp8", "kv_store", "attn",
  "scatter k and v into a paged E4M3 cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim]; head_dim a multiple of 4; a "
  "negative or out-of-range slot stores nothing",
  "bf16 k / v, fp8-e4m3 cache, i32 slot mapping, scalar k_scale / v_scale descales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cKvStoreFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store_fp8, nullptr, nullptr, nullptr,
  r4d_shape_kv_store },

{ "kv_store_fp8v4", "kv_store", "attn",
  "scatter k and v into a paged fp8_v4 cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 25*head_dim/16]; head_dim a multiple of 32; a "
  "negative or out-of-range slot stores nothing",
  "bf16 k / v, E4M3 K and int4 V in groups of 32 with f16 scales, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cKvStoreFp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store_fp8v4, nullptr, nullptr, nullptr,
  nullptr },

{ "kv_store_k6v4", "kv_store", "attn",
  "scatter k and v into a paged k6_v4 cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 22*head_dim/16]; head_dim a multiple of 32; a "
  "negative or out-of-range slot stores nothing",
  "bf16 k / v, int6 K and int4 V in groups of 32 with f16 scales, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cKvStoreK6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store_k6v4, nullptr, nullptr, nullptr,
  nullptr },

{ "logits_gemm_bf16", "logits_gemm", "vocab",
  "the lm_head projection: C[M, n_vocab] f32 = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T",
  "any M and any vocabulary; n_embd a multiple of 8. One workgroup a vocabulary column",
  "bf16 x and lm_head, F32 LOGITS OUT, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLogitsGemm), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm, nullptr, nullptr, nullptr,
  r4d_shape_logits_gemm },

{ "logits_gemm_fp8", "logits_gemm", "vocab",
  "the lm_head projection over an fp8 head: C[M, n_vocab] f32 = x[M, n_embd] @ dequant(W)^T, "
  "one wave a vocabulary row, the row's own scales in its tail",
  "any M -- one pass to 32 rows, instantiated exactly to 8 and rounded up to a multiple of 4 above "
  "it, then chunked; n_embd a multiple of 128, which is the group the scales are taken over. The "
  "stored row is padded to 16 bytes so the uint4 load stays aligned at any such n_embd",
  "bf16 x, E4M3 head with a bf16 scale per 128x128 tile stored PER ROW, F32 LOGITS OUT, fp32 "
  "accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LOGITS_FP8,
  ROW(cLogitsGemmFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm_fp8, nullptr,
  r4d_rad_layout_fp8_logits, r4d_rad_relayout_fp8_logits,
  nullptr, r4d_rad_unrelayout_fp8_logits },

{ "logits_gemm_i8", "logits_gemm", "vocab",
  "logits_gemm_fp8 over an INT8 head: the same stored row and the same two forms, i8 codes and each "
  "row's own scales in its tail",
  "logits_gemm_fp8's: any M, n_embd a multiple of 128",
  "bf16 x, i8 head with a bf16 scale per ROW per 128 columns stored in the row's tail, F32 LOGITS "
  "OUT, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LOGITS_FP8,
  ROW(cLogitsGemmFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm_i8, nullptr,
  r4d_rad_layout_i8_logits, r4d_rad_relayout_i8_logits,
  nullptr, r4d_rad_unrelayout_i8_logits },

/* ---------------------------- the decomposed sampler chain (r4d_sample_stages_f32.hip) ------
 *
 * Distinct from sample_chain, which fuses temperature, softmax, top_k, top_p, min_p and the draw
 * in vLLM's fixed order and produces the speculative verify's outputs. These are the ops the core
 * composes per request, including stages the fused form lacks: penalties, DRY, a grammar bitmask,
 * locally-typical, XTC (spec §2.3: fusion is a selection).
 */
{ "sample_penalties_f32", "sample_penalties", "sample",
  "repetition, frequency and presence penalties in one pass over the row's history",
  "any vocabulary; the history window is the TAIL of the row's span. Parallel over the history and "
  "race-free because only a token's FIRST occurrence writes it",
  "f32 logits in place, i32 history, per-row params",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleVoc), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_penalties, nullptr, nullptr, nullptr,
  r4d_shape_sample_penalties },

{ "sample_dry_f32", "sample_dry", "sample",
  "the DRY penalty over the row's history",
  "any vocabulary, a history row of at most 16384 tokens (one u16 repeat length each in LDS); ONE "
  "BLOCK A ROW, and each continuation token has exactly one writing thread -- its longest repeat's "
  "-- so no atomic sum and no order-dependent rounding",
  "f32 logits in place, i32 history, optional u8 breaker plane, per-row params",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleVoc), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_dry, nullptr, nullptr, nullptr,
  r4d_shape_sample_dry },

{ "sample_temp_f32", "sample_temp", "sample",
  "per-request temperature",
  "any vocabulary; a row with temp <= 0 or exactly 1 is skipped whole",
  "f32 logits in place, per-row params",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleVoc), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_temp, nullptr, nullptr, nullptr,
  r4d_shape_sample_temp },

{ "sample_mask_f32", "sample_mask", "sample",
  "the grammar bitmask -- a cleared bit sets -infinity",
  "any vocabulary; the plane is [rows, ceil(n_vocab/32)] and is shorter than M, addressed by the "
  "row's mask_row",
  "f32 logits in place, u32 bitmask, per-row params",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleVoc), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_mask, nullptr, nullptr, nullptr,
  r4d_shape_sample_mask },

{ "sample_topk_f32", "sample_topk", "sample",
  "the exact top-k candidate set, descending, ties to the lower token id",
  "any vocabulary; n_cand <= 4096, the width the LDS bitonic sort covers. Eight radix passes over "
  "a monotone key -- four for the k-th largest logit, four over the token id to break the tie at "
  "it -- then one sort of the survivors. Exact and deterministic, without sorting the vocabulary",
  "f32 logits, i32 candidate ids, f32 candidate LOGITS (not probabilities)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleVoc), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_topk, nullptr, nullptr, nullptr,
  r4d_shape_sample_topk },

{ "sample_argmax_f32", "sample_argmax", "sample",
  "greedy, ties to the lower token id",
  "any vocabulary; the (key, ~id) packing makes the tie rule fall out of an unsigned max",
  "f32 logits, i32 token",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleVoc), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_argmax, nullptr, nullptr, nullptr,
  r4d_shape_sample_argmax },

{ "sample_logprobs_f32", "sample_logprobs", "sample",
  "the raw distribution's summary for a logprobs row: max, sum of exp(l - max), the k largest",
  "any vocabulary; k <= 64, the LDS the sort runs in. sample_topk's radix select at the op's k, so "
  "the set and its tie rule are the top-k's; the maximum is its first entry and the sum one more "
  "pass. A row without RAD_SP_LOGPROBS leaves before reading anything",
  "f32 logits, f32 stats [M, 2 + 2k] with the ids as floats",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleLogprobs), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_logprobs, nullptr, nullptr, nullptr,
  r4d_shape_sample_logprobs },

{ "sample_topp_f32", "sample_topp", "sample",
  "nucleus narrowing in place",
  "n_cand <= 4096; the prefix sum is SEQUENTIAL in f32 in candidate order, because a parallel "
  "prefix sums in a different order and a candidate at the cut would survive in one and not the "
  "other",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_topp, nullptr, nullptr, nullptr,
  r4d_shape_sample_narrow },

{ "sample_minp_f32", "sample_minp", "sample",
  "min-p narrowing in place",
  "n_cand <= 4096; exp(v - max) >= p is the same test as prob >= p * max_prob without forming the "
  "normaliser",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_minp, nullptr, nullptr, nullptr,
  r4d_shape_sample_narrow },

{ "sample_typical_f32", "sample_typical", "sample",
  "locally typical narrowing in place",
  "n_cand <= 4096; quadratic in the candidate width, parallel over the outer index, and the "
  "reduction is a MINIMUM so it is order-independent",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_typical, nullptr, nullptr, nullptr,
  r4d_shape_sample_narrow },

{ "sample_xtc_f32", "sample_xtc", "sample",
  "exclude top choices, in place",
  "n_cand <= 4096; the draw is per row from the row's own stream, so it reproduces",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_xtc, nullptr, nullptr, nullptr,
  r4d_shape_sample_narrow },

{ "sample_pick_f32", "sample_pick", "sample",
  "the inverse-CDF draw over the candidate set",
  "n_cand <= 4096; the walk is SEQUENTIAL in f32 for the same reason top_p's is, and the last live "
  "candidate is the fallback because the probabilities do not sum to exactly 1 in f32",
  "i32 candidate ids, f32 candidate logits, i32 token",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_pick, nullptr, nullptr, nullptr,
  r4d_shape_sample_pick },

{ "sample_merge_topk_f32", "sample_merge_topk", "sample",
  "the vocab-parallel merge of the all-gathered per-rank candidate sets",
  "world_size * n_cand <= 4096; token ids in `gathered` are already GLOBAL, so no shard layout is "
  "needed",
  "u32 (id, logit-bits) pairs in, i32 candidate ids and f32 candidate logits out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  /* No `opd_shape`: see r4d_shapes.cpp (word-typed pairs, no `world_size`); ref declines it too. */
  nullptr, nullptr, r4d_rad_sample_merge_topk },


/* ---------------------------------------------------------------------------------- attention */
{ "attn_prefill_h256_gqa6_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa6_fp8v4", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillFp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_fp8v4, nullptr, nullptr, nullptr,
  nullptr },

{ "attn_prefill_h256_gqa6_k6v4", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 16, varlen",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillK6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_k6v4, nullptr, nullptr, nullptr,
  nullptr },

{ "attn_prefill_h256_gqa6_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 16, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa6_fp8kv", "attn_paged", "attn",
  "paged causal attention, split-KV (decode / speculative verify)",
  "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeFp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_fp8, r4d_rad_attn_decode_h256_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa6_fp8v4", "attn_paged", "attn",
  "paged causal attention, split-KV (decode / speculative verify)",
  "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeFp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_fp8v4, r4d_rad_attn_decode_h256_v4_scratch,
  nullptr, nullptr,
  nullptr },

{ "attn_decode_h256_gqa6_k6v4", "attn_paged", "attn",
  "paged causal attention, split-KV (decode / speculative verify)",
  "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeK6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_k6v4, r4d_rad_attn_decode_h256_k6_scratch,
  nullptr, nullptr,
  nullptr },

{ "attn_decode_h256_gqa6_bf16kv", "attn_paged", "attn",
  "paged causal attention, split-KV (decode / speculative verify)",
  "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeBf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_bf16, r4d_rad_attn_decode_h256_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa4_fp8kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, split-KV (decode / draft block)",
  "head_dim 128, 4 queries per kv head, paged block 16, q_len*gqa <= 64, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_fp8, r4d_rad_attn_decode_h128_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa4_bf16kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, split-KV (decode / draft block)",
  "head_dim 128, 4 queries per kv head, paged block 16, q_len*gqa <= 64, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_bf16, r4d_rad_attn_decode_h128_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa2_fp8kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 2 queries per kv head, paged block 16, varlen, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256Gqa2Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa2_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa2_bf16kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 2 queries per kv head, paged block 16, varlen, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256Gqa2Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa2_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa2_fp8kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, split-KV (decode / verify)",
  "head_dim 256, 2 queries per kv head, paged block 16, q_len*gqa <= 64, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256Gqa2Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa2_fp8, r4d_rad_attn_decode_h256_gqa2_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa2_bf16kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, split-KV (decode / verify)",
  "head_dim 256, 2 queries per kv head, paged block 16, q_len*gqa <= 64, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256Gqa2Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa2_bf16, r4d_rad_attn_decode_h256_gqa2_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h128_gqa2w_fp8kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, query-tiled (prefill / chunked prefill)",
  "head_dim 128, 2 queries per kv head, paged block 16, varlen, window >= 1 from the launch argument",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH128Gqa2wFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h128_gqa2w_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h128_gqa2w_bf16kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, query-tiled (prefill / chunked prefill)",
  "head_dim 128, 2 queries per kv head, paged block 16, varlen, window >= 1 from the launch argument",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH128Gqa2wBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h128_gqa2w_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa2w_fp8kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, split-KV (decode / verify)",
  "head_dim 128, 2 queries per kv head, paged block 16, q_len*gqa <= 64, window >= 1 from the "
  "launch argument",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128Gqa2wFp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa2w_fp8, r4d_rad_attn_decode_h128_gqa2w_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa2w_bf16kv", "attn_paged", "attn",
  "paged causal attention with a sliding window, split-KV (decode / verify)",
  "head_dim 128, 2 queries per kv head, paged block 16, q_len*gqa <= 64, window >= 1 from the "
  "launch argument",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128Gqa2wBf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa2w_bf16, r4d_rad_attn_decode_h128_gqa2w_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },
/* The same decode under attn_paged_qrow, reading q at its own row pitch: Gemma 4 31B's sliding
 * layers read q in the fused q|k|v row. It outranks the windowed qrow rows (R4D_PRIO_ONLY) in its
 * band, as attn_paged's does, since it takes 0.26-0.71 of their time there; above q_len 32 the
 * windowed kernel serves the op alone. No opd_shape, as for every qrow row: libref has none. */
{ "attn_qrow_decode_h256_gqa2_fp8kv", "attn_paged_qrow", "attn",
  "paged attention reading q at its own row pitch, sliding window, split-KV (decode / verify)",
  "head_dim 256, 2 queries per kv head, paged block 16, q_len*gqa <= 64, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query (any row pitch, a multiple of 4), fp8-e4m3 kv cache with per-(seq,head) descales, "
  "bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256Gqa2Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_qrow_decode_h256_gqa2_fp8,
  r4d_rad_attn_qrow_decode_h256_gqa2_scratch,
  nullptr, nullptr, nullptr },

{ "attn_qrow_decode_h256_gqa2_bf16kv", "attn_paged_qrow", "attn",
  "paged attention reading q at its own row pitch, sliding window, split-KV (decode / verify)",
  "head_dim 256, 2 queries per kv head, paged block 16, q_len*gqa <= 64, "
  "window taken from the launch argument (0 = full attention)",
  "bf16 query (any row pitch, a multiple of 4), bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256Gqa2Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_qrow_decode_h256_gqa2_bf16,
  r4d_rad_attn_qrow_decode_h256_gqa2_scratch,
  nullptr, nullptr, nullptr },

/* Head_dim 128 at gqa 8 under attn_paged_qrow (cAttnQrowH128G8Fp8 says why). */
{ "attn_qrow_h128_gqa8_fp8kv", "attn_paged_qrow", "attn",
  "paged attention, any window, causal or not, split-KV (draft block); q at any row pitch",
  "head_dim 128, 8 queries per kv head as two groups of 4 a kv head, paged block 16, q_len <= 16",
  "bf16 query (any row pitch, a multiple of 4), fp8-e4m3 kv cache with per-(seq,head) descales, "
  "bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnQrowH128G8Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_qrow_h128_gqa8_fp8, r4d_rad_attn_qrow_h128_gqa8_scratch,
  nullptr, nullptr, nullptr },

{ "attn_qrow_h128_gqa8_bf16kv", "attn_paged_qrow", "attn",
  "paged attention, any window, causal or not, split-KV (draft block); q at any row pitch",
  "head_dim 128, 8 queries per kv head as two groups of 4 a kv head, paged block 16, q_len <= 16",
  "bf16 query (any row pitch, a multiple of 4), bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnQrowH128G8Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_qrow_h128_gqa8_bf16, r4d_rad_attn_qrow_h128_gqa8_scratch,
  nullptr, nullptr, nullptr },

{ "attn_prefill_h256_gqa8_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 8 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G8Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa8_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa8_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 8 queries per kv head, paged block 16, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G8Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa8_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa8_fp8kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 8 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G8Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa8_fp8, r4d_rad_attn_decode_h256_gqa8_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa8_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 8 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G8Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa8_bf16, r4d_rad_attn_decode_h256_gqa8_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa4_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 4 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G4Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa4_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa4_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 4 queries per kv head, paged block 16, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G4Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa4_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa4_fp8kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 4 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G4Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa4_fp8, r4d_rad_attn_decode_h256_gqa4_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa4_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 4 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G4Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa4_bf16, r4d_rad_attn_decode_h256_gqa4_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h64_gqa4_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 64, 4 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH64G4Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h64_gqa4_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h64_gqa4_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 64, 4 queries per kv head, paged block 16, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH64G4Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h64_gqa4_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

/* The windowed kernels (r4d_attn_paged_window.hip), under attn_paged and, with the same launches
 * and constraints, under attn_paged_qrow. The qrow rows have no opd_shape: libref has no row for
 * that op, and tests/r4d_emu's attention cases already take q as a pitched slice. */
#define R4D_WIN_ROW(HD, G, KVN, KVS, C, KVDOC)                                                    \
{ "attn_window_h" #HD "_gqa" #G "_" #KVN, "attn_paged", "attn",                                   \
  "paged attention, any window, causal or not; split-KV with a merge where tiles are few",        \
  "head_dim " #HD ", " #G " queries per kv head, paged block 16, any q_len, varlen",              \
  "bf16 query, " KVDOC " kv cache, bf16 out, f32 partials",                                        \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_WINDOW,                                                         \
  ROW(C), NOTUNE,                                                                                  \
  nullptr, nullptr, r4d_rad_attn_window_h##HD##_gqa##G##_##KVS,                                   \
  r4d_rad_attn_window_h##HD##_scratch,                                                             \
  nullptr, nullptr,                                                                                \
  r4d_shape_attn_paged },                                                                          \
{ "attn_qrow_h" #HD "_gqa" #G "_" #KVN, "attn_paged_qrow", "attn",                                \
  "paged attention reading q at its own row pitch, any window, causal or not",                    \
  "head_dim " #HD ", " #G " queries per kv head, paged block 16, any q_len, varlen",              \
  "bf16 query (any row pitch), " KVDOC " kv cache, bf16 out, f32 partials",                        \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,                                                                \
  ROW(C), NOTUNE,                                                                                  \
  nullptr, nullptr, r4d_rad_attn_window_h##HD##_gqa##G##_##KVS,                                   \
  r4d_rad_attn_window_h##HD##_scratch,                                                             \
  nullptr, nullptr, nullptr },
#define R4D_WIN_PAIR(HD, GQ)                                                                       \
    R4D_WIN_ROW(HD, GQ, bf16kv, bf16, cWinH##HD##G##GQ##Bf16, "bf16")                              \
    R4D_WIN_ROW(HD, GQ, fp8kv, fp8, cWinH##HD##G##GQ##Fp8, "fp8-e4m3 (per-(seq,head) descales)")
R4D_WIN_PAIR(256, 1) R4D_WIN_PAIR(256, 2) R4D_WIN_PAIR(256, 4) R4D_WIN_PAIR(256, 8)
R4D_WIN_PAIR(512, 1) R4D_WIN_PAIR(512, 2) R4D_WIN_PAIR(512, 4) R4D_WIN_PAIR(512, 8)
#undef R4D_WIN_PAIR
#undef R4D_WIN_ROW

{ "attn_prefill_h256_gqa12_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 12 queries per kv head, paged block 4, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G12Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa12_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa12_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 12 queries per kv head, paged block 4, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G12Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa12_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa12_fp8kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 12 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G12Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_fp8, r4d_rad_attn_decode_h256_gqa12_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

/* The same decode dispatch with gate_quant_fp8 as the merge's epilogue: two launches instead of
 * three. */
{ "attn_decode_gq_h256_gqa12_fp8kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa12_fp8kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, fp8-e4m3 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G12Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa12_fp8, r4d_rad_attn_decode_h256_gqa12_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_decode_h256_gqa12_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 12 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G12Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_bf16, r4d_rad_attn_decode_h256_gqa12_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

/* The bf16 cache's decode with gate_quant_fp8 in the merge, as the fp8 cache's row above. */
{ "attn_decode_gq_h256_gqa12_bf16kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa12_bf16kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, bf16 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G12Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa12_bf16, r4d_rad_attn_decode_h256_gqa12_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant },

/* The TP4 leg of Flash-Next's sparse attention: the gqa12 rows' kernels at the ratio a four-rank
 * world hands a rank (see cAttnPrefillH256G6Bs4Fp8). `gqa` is per rank, so world 2 resolves only
 * the gqa12 rows and world 4 only these. */
{ "attn_prefill_h256_gqa6_bs4_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 4, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G6Bs4Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h256_gqa6_bs4_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 4, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G6Bs4Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa6_bs4_fp8kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 6 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G6Bs4Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_fp8, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h256_gqa6_bs4_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 6 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G6Bs4Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_bf16, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

/* The TP4 leg's sparse decode with the gate and quantiser in the merge, as the gqa12 pair above. */
{ "attn_decode_gq_h256_gqa6_bs4_fp8kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa6_bs4_fp8kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, fp8-e4m3 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G6Bs4Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_fp8, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_decode_gq_h256_gqa6_bs4_bf16kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa6_bs4_bf16kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, bf16 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G6Bs4Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_bf16, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant },

/* The fp8_v4 and k6_v4 caches on Flash-Next's QSA layers: gqa12 at one and two ranks, gqa6 with
 * 4-token pages at four. A token's row is 400 bytes (E4M3 K, int4 V, f16 group scales) or 352
 * (int6 K, int4 V); the layouts are RAD_KVQ_*'s (abi/rad_builder.h). */
{ "attn_prefill_h256_gqa12_fp8v4", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 12 queries per kv head, paged block 4, varlen",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G12Fp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa12_fp8v4, nullptr, nullptr, nullptr,
  nullptr },

{ "attn_decode_h256_gqa12_fp8v4", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 12 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G12Fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_fp8v4, r4d_rad_attn_decode_h256_gqa12_fp8v4_scratch,
  nullptr, nullptr,
  nullptr },

{ "attn_decode_gq_h256_gqa12_fp8v4", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa12_fp8v4's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G12Fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa12_fp8v4, r4d_rad_attn_decode_h256_gqa12_fp8v4_scratch,
  nullptr, nullptr,
  nullptr, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_prefill_h256_gqa12_k6v4", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 12 queries per kv head, paged block 4, varlen",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G12K6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa12_k6v4, nullptr, nullptr, nullptr,
  nullptr },

{ "attn_decode_h256_gqa12_k6v4", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 12 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G12K6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_k6v4, r4d_rad_attn_decode_h256_gqa12_k6v4_scratch,
  nullptr, nullptr,
  nullptr },

{ "attn_decode_gq_h256_gqa12_k6v4", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa12_k6v4's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G12K6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa12_k6v4, r4d_rad_attn_decode_h256_gqa12_k6v4_scratch,
  nullptr, nullptr,
  nullptr, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_prefill_h256_gqa6_bs4_fp8v4", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 4, varlen",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G6Bs4Fp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_fp8v4, nullptr, nullptr, nullptr,
  nullptr },

{ "attn_decode_h256_gqa6_bs4_fp8v4", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 6 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G6Bs4Fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_fp8v4, r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch,
  nullptr, nullptr,
  nullptr },

{ "attn_decode_gq_h256_gqa6_bs4_fp8v4", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa6_bs4_fp8v4's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, fp8_v4 kv cache (E4M3 K, int4 V in groups of 32 with f16 scales), bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G6Bs4Fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_fp8v4, r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch,
  nullptr, nullptr,
  nullptr, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_prefill_h256_gqa6_bs4_k6v4", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 4, varlen",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH256G6Bs4K6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_k6v4, nullptr, nullptr, nullptr,
  nullptr },

{ "attn_decode_h256_gqa6_bs4_k6v4", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 6 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G6Bs4K6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_k6v4, r4d_rad_attn_decode_h256_gqa6_k6v4_scratch,
  nullptr, nullptr,
  nullptr },

{ "attn_decode_gq_h256_gqa6_bs4_k6v4", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa6_bs4_k6v4's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, k6_v4 kv cache (int6 K, int4 V, each in groups of 32 with f16 scales), bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G6Bs4K6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_k6v4, r4d_rad_attn_decode_h256_gqa6_k6v4_scratch,
  nullptr, nullptr,
  nullptr, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_prefill_h128_gqa8_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 128, 8 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH128G8Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h128_gqa8_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h128_gqa8_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 128, 8 queries per kv head, paged block 16, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH128G8Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h128_gqa8_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa8_fp8kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 128, 8 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128G8Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa8_fp8, r4d_rad_attn_decode_h128_gqa8_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa8_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 128, 8 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128G8Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa8_bf16, r4d_rad_attn_decode_h128_gqa8_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h128_gqa2_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 128, 2 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH128G2Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h128_gqa2_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_prefill_h128_gqa2_bf16kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 128, 2 queries per kv head, paged block 16, varlen",
  "bf16 query, bf16 kv cache, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillH128G2Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h128_gqa2_bf16, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa2_fp8kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify)",
  "head_dim 128, 2 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128G2Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa2_fp8, r4d_rad_attn_decode_h128_gqa2_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_decode_h128_gqa2_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify)",
  "head_dim 128, 2 queries per kv head, paged block 16, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH128G2Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa2_bf16, r4d_rad_attn_decode_h128_gqa2_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged },

{ "attn_vit_h72_bf16", "attn_dense", "attn",
  "dense non-causal vision-encoder attention, varlen over images / windows",
  "head_dim 72 native (no padding to 128), multi-head (no GQA), contiguous q/k/v/o, "
  "one cu_seqlens segment per image",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnVit), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_vit },

{ "attn_dense_h64_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window or "
  "blocks, or causal with an optional window behind",
  "head_dim 64, multi-head (no GQA), q/k/v/o at their own token pitches (column slices of a "
  "fused projection allowed)",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH64), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

{ "attn_dense_h256_gqa2_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window, or "
  "causal with an optional window behind",
  "head_dim 256, two query heads a KV head in one workgroup, the key range cut to the window",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH256G2), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

{ "attn_dense_h256_gqa4_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window, or "
  "causal with an optional window behind",
  "head_dim 256, four query heads a KV head in one workgroup, the key range cut to the window",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH256G4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

{ "attn_dense_h256_gqa8_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window, or "
  "causal with an optional window behind",
  "head_dim 256, eight query heads a KV head in one workgroup (a multiple of eight as several), "
  "the key range cut to the window",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH256G8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

{ "attn_dense_h512_gqa2_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window, or "
  "causal with an optional window behind",
  "head_dim 512 split across two waves, two query heads a KV head in one workgroup",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH512G2), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

{ "attn_dense_h512_gqa4_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window, or "
  "causal with an optional window behind",
  "head_dim 512 split across two waves, four query heads a KV head in one workgroup",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH512G4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

{ "attn_dense_h512_gqa8_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens: bidirectional with an optional two-sided window, or "
  "causal with an optional window behind",
  "head_dim 512 split across two waves, eight query heads a KV head in one workgroup (a "
  "multiple of eight as several workgroups)",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH512G8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense },

/* -------------------------------------------------------------------------- gated delta net */
{ "gdn_chunk_scan_k128_v128_c64_bf16", "gdn_chunk_scan", "gdn",
  "gated delta net chunked scan: WY recompute, state recurrence and output in one kernel",
  "head_k 128, head_v 128, chunk 64, varlen, state resident in WMMA accumulators",
  "bf16 q/k/v/A, fp32 gate/beta, bf16 out; the state cache is f32 or f16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnChunkScan), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_chunk_scan },

/* Unvalidated: the DPP row_xmask exchanges in this translation unit (r4d_common.h r4d_xor_lane,
 * used by the conv) are compile-verified only, not measured or correctness-gated. */
{ "gdn_conv_prep_w4_h128_bf16", "gdn_conv_prep", "gdn",
  "gdn prefill preamble: causal conv + qkv split + qk l2norm + gating + per-chunk cumsum "
  "[UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "conv width 4, head_k/head_v 128, chunk 64, varlen, conv state read and rewritten in place",
  "bf16 x / weights / conv state / q,k,v out, fp32 a,b and gate/beta out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnConvPrep), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_conv_prep },

/* Unvalidated, as gdn_conv_prep above (same translation unit and exchanges). */
{ "gdn_conv_update_w4_h128_bf16", "gdn_conv_update", "gdn",
  "gdn decode conv: rolling speculative window over the conv state cache, writes q/k/v "
  "[UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "conv width 4, head_k/head_v 128, state cache width-1 + num_spec, read at accepted offset",
  "bf16 x / weights / conv state / q,k,v out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnConvUpdate), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_conv_update },

{ "gdn_kkt_solve_k128_c64_bf16", "gdn_kkt_solve", "gdn",
  "gdn chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^dg))^-1, gram never in HBM",
  "head_k 128, chunk 64, varlen; one gram per k head, applied to every v head of the group",
  "bf16 k and A out, fp32 beta / gate / gram / inverse",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnKktSolve), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_kkt_solve },

/* Unvalidated: the DPP row_xmask exchanges here replace 408 ds_bpermute, several inside the token
 * loop on the dependency chain, and are not measured or correctness-gated. */
{ "gdn_recurrent_update_k128_v128_bf16_fp32state", "gdn_recurrent_update", "gdn",
  "gdn decode recurrent update: gating, qk l2norm, delta-rule state update, gated rms norm and "
  "output [UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "head_k 128, head_v 128, paged state, one state written per candidate token",
  "bf16 q/k/v and out, fp32 a,b; the state cache is f32 or f16 and the operand says which",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnRecurrentUpdate), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_recurrent_update },

{ "gdn_conv_recurrent_update_k128_v128_bf16", "gdn_conv_recurrent_update", "gdn",
  "gdn decode: the recurrent update with gdn_conv_update folded into its prologue -- each "
  "workgroup convolves its key head's q and k and its value head's v into LDS, and the last "
  "workgroup of a key head to arrive rewrites that head's conv state",
  "head_k/head_v 128, conv width 4, a sequence's window at most 8 tokens, the fused row norm",
  "bf16 x / conv weights / conv state / a,b / out; the state cache is f32 or f16. "
  "Byte-identical to gdn_conv_update followed by gdn_recurrent_update",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnConvRecurrentUpdate), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_conv_recurrent_update, nullptr, nullptr, nullptr, nullptr,
  nullptr, r4d_fuse_gdn_conv_recurrent_update },

{ "kda_update_k128_v128_bf16", "kda_update", "gdn", "per-channel KDA recurrence and candidate rollback",
  "heads 1..65535, K=V=128, prefill and verify", "bf16 activations, f32 gates and state",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cKdaUpdate), NOTUNE,
  nullptr, nullptr, r4d_rad_kda_update, nullptr, nullptr, nullptr,
  r4d_shape_kda_update },
{ "mhc_pre_bf16", "mhc_pre", "norm", "manifold residual gates and Sinkhorn mixing",
  "M 1..65535, n 1..16384, hc 2..8, iterations 1..256", "bf16 activation, f32 weights and coefficients",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMhcPre), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_pre, nullptr, nullptr, nullptr,
  r4d_shape_mhc_pre },
{ "mhc_pre_partial_bf16", "mhc_pre_partial", "norm", "mHC projection partials of a chunk range",
  "n 4096, hc 4, chunk 2048 (GLM)", "bf16 activation, f32 weights and partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMhcPrePartial), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_pre_partial, nullptr, nullptr, nullptr,
  r4d_shape_mhc_pre_partial },
{ "mhc_pre_finish_bf16", "mhc_pre_finish", "norm", "mHC gates and Sinkhorn mixing from gathered partials",
  "n 4096, hc 4, chunk 2048 (GLM), iterations 1..256", "bf16 activation, f32 partials and coefficients",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMhcPreFinish), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_pre_finish, nullptr, nullptr, nullptr,
  r4d_shape_mhc_pre_finish },
{ "mhc_post_bf16", "mhc_post", "norm", "source-to-destination manifold residual update",
  "M 1..65535, n 1..16384, hc 2..8", "bf16 activation, f32 coefficients",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMhcPlain), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_post, nullptr, nullptr, nullptr,
  r4d_shape_mhc_post },
{ "mhc_contract_bf16", "mhc_contract", "norm", "mean over manifold residual streams",
  "M 1..65535, n 1..16384, hc 2..8", "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMhcPlain), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_contract, nullptr, nullptr, nullptr,
  r4d_shape_mhc_contract },

/* ---------------------------------------------------------------- the gated residual --------
 * r4d_hc_bf16.hip. Three rows; the read is two launches behind one entry point, because `x`
 * depends on every column of the low-rank waist and the kernel boundary is the grid-wide
 * barrier. */
{ "hc_enter_bf16", "hc_enter", "norm",
  "hc identical copies of x into the wide residual stream",
  "any M, any n, hc 2..8",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcPlain), NOTUNE,
  nullptr, nullptr, r4d_rad_hc_enter, nullptr, nullptr, nullptr,
  r4d_shape_hc_enter },

{ "hc_write_bf16", "hc_write", "norm",
  "h[s] += inj[s] * y, in place on the unnormed stream",
  "any M, any n, hc 2..8",
  "bf16 throughout",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcPlain), NOTUNE,
  nullptr, nullptr, r4d_rad_hc_write, nullptr, nullptr, nullptr,
  r4d_shape_hc_write },

{ "hc_project_bf16", "hc_project", "norm",
  "control vector: h[s] -= scale * (h[s] . v) * v where the row's flag is on",
  "any M, any n, hc 2..8",
  "bf16 stream, f32 direction, i32 switch",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcProject), NOTUNE,
  nullptr, nullptr, r4d_rad_hc_project, nullptr, nullptr, nullptr,
  r4d_shape_hc_project },

{ "mtp_enter_bf16", "mtp_enter", "norm",
  "MTP head entry: two norms, a per-sub-stream fc and one broadcast fc",
  "any M, n 128..5734, hc 2..4",
  "bf16 activations and fc, f32 gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMtpEnter), NOTUNE,
  nullptr, nullptr, r4d_rad_mtp_enter, r4d_rad_mtp_enter_scratch, nullptr, nullptr,
  r4d_shape_mtp_enter },

{ "mtp_enter_e4m3", "mtp_enter", "norm",
  "the MTP head's entry with fc_hidden and fc_embedding stored as E4M3 rows: half the bytes of the "
  "two square matrices every draft round reads",
  "any M, n a multiple of 128 up to 5734, hc 2..4",
  "bf16 activations, E4M3 fc rows with f32 128-column group scales in each row's tail, f32 gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMtpEnterE4m3), NOTUNE,
  nullptr, nullptr, r4d_rad_mtp_enter, r4d_rad_mtp_enter_scratch,
  r4d_rad_layout_mtp_e4m3, r4d_rad_relayout_mtp_e4m3,
  r4d_shape_mtp_enter, r4d_rad_unrelayout_mtp_e4m3 },

{ "hc_read_bf16", "hc_read", "norm",
  "grouped rmsnorm, the low-rank read gate, the stream mean, the per-branch write gains and the "
  "fp8 twin of the block input -- in two launches",
  "any M; n a multiple of 128 and at most 7552; hc 2..4; any lowrank; group 0 or 128",
  "bf16 stream and mixing matrices, f32 gain, bf16 out, E4M3 + f32 scale for the optional twin",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcRead), NOTUNE,
  nullptr, nullptr, r4d_rad_hc_read, r4d_rad_hc_read_scratch, nullptr, nullptr,
  r4d_shape_hc_read },

{ "hc_read_e4m3", "hc_read", "norm",
  "hc_read with the two mixing matrices stored as E4M3 rows -- half their bytes, the largest "
  "weight stream a decode step reads after the experts -- in the same two launches",
  "any M; n a multiple of 512 and at most 7552; hc 2..4; lowrank a multiple of 64 up to 512; "
  "group 0 or 128",
  "bf16 stream, E4M3 mixing matrices with f32 scales in each row (128-column groups for "
  "mix_down, a quarter row for mix_up), f32 gain, bf16 out, E4M3 + f32 scale for the twin",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcReadE4m3), NOTUNE,
  nullptr, nullptr, r4d_rad_hc_read, r4d_rad_hc_read_scratch,
  r4d_rad_layout_hc_e4m3, r4d_rad_relayout_hc_e4m3,
  r4d_shape_hc_read, r4d_rad_unrelayout_hc_e4m3 },

{ "gdn_gated_rmsnorm_h128_bf16", "gdn_gated_rmsnorm", "gdn",
  "gated rms norm over a row: out = rms(x) . w . act(z), for the prefill path only",
  "128 channels per row, one wave per row; silu or sigmoid gate",
  "bf16 x / z / out, fp32 weight",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnGatedRmsNorm), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_gated_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_gdn_gated_rmsnorm },

/* ---------------------------------------------------------------------------- all-reduce ----
 * The two-shot rows outrank the one-shots (see the all-reduce constraints above). */
{ "ar_twoshot_3rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch, per-block pipelined "
  "handshake",
  "exactly 3 ranks, all peer-accessible; messages above "
  R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements, two hops; shards as even as the word count "
  "allows (the first words % 3 take one more); shares the wide family's flag/seq/scratch set; in "
  "place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks "
  "and to ar_oneshot_3rank_exact",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact3), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_3rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_twoshot_4rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch, per-block pipelined "
  "handshake",
  "exactly 4 ranks, all peer-accessible; messages above "
  R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements, two hops; shares the wide family's "
  "flag/seq/scratch set; message must shard evenly over the ranks; in place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact4), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_4rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

/* The row above at `overlap` 1: the same entry point and the same arithmetic, a narrow launch
 * that leaves the CUs to the compute running beside it (tArTs4Lane). */
{ "ar_twoshot_4rank_exact_lane", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch; narrow launch beside "
  "compute",
  "exactly 4 ranks, all peer-accessible, overlap 1; messages above "
  R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements; message must shard evenly over the ranks; "
  "in place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks "
  "and to the wide row",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_TWOSHOT + 1,
  ROW(cArTsExact4Lane), ROW(tArTs4Lane), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_4rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_twoshot_8rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch, per-block pipelined "
  "handshake",
  "exactly 8 ranks, all peer-accessible; messages above "
  R4D_XSTR(R4D_AR_8RANK_ONESHOT_MAX_ELEMS) " elements, two hops; shares the wide family's "
  "flag/seq/scratch set; message must shard evenly over the ranks; in place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact8), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_8rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_gather_2rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch, per-block flag handshake",
  "exactly 2 ranks, peer-accessible; y is [2 * numel] with rank r's input at r * numel; "
  "scratch sized by the full message; cudagraph-safe",
  "any payload -- the bytes are moved, never summed, so the (u32 id, f32 logit) pairs the "
  "vocab-parallel sampler gathers survive intact",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArGather2), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_gather_2rank, nullptr, nullptr, nullptr,
  r4d_shape_all_gather },

{ "ar_gather_4rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch, per-block handshake",
  "exactly 4 ranks, all peer-accessible; y is [4 * numel] with rank r's input at r * numel; one "
  "scratch slot per (parity, source rank); shares the wide family's flag/seq/scratch set; "
  "cudagraph-safe",
  "any payload -- the bytes are moved, never summed, so the (u32 id, f32 logit) pairs the "
  "vocab-parallel sampler gathers survive intact",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArGather4), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_gather_4rank, nullptr, nullptr, nullptr,
  r4d_shape_all_gather },

{ "ar_gather_8rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch, per-block handshake",
  "exactly 8 ranks, all peer-accessible; y is [8 * numel] with rank r's input at r * numel; one "
  "scratch slot per (parity, source rank); shares the wide family's flag/seq/scratch set; "
  "cudagraph-safe",
  "any payload -- the bytes are moved, never summed, so the (u32 id, f32 logit) pairs the "
  "vocab-parallel sampler gathers survive intact",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArGather8), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_gather_8rank, nullptr, nullptr, nullptr,
  r4d_shape_all_gather },

{ "ar_oneshot_2rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch",
  "exactly 2 ranks, peer-accessible; scratch sized by the full message; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArExact), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_2rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_oneshot_2rank_wht6", "all_reduce", "ar",
  "all-reduce (sum) with a Walsh-Hadamard-rotated 6-bit wire payload -- lossy",
  "exactly 2 ranks, peer-accessible; groups of 64 elements; cudagraph-safe",
  "bf16 / fp16 payload, 6 bits + bf16 scale per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArWht6), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_2rank_wht6, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

/* The two above at `overlap` 1: same entry points and arithmetic, with a launch that leaves the
 * CUs to the compute running beside it (tArLane). */
{ "ar_oneshot_2rank_exact_lane", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch, narrow launch beside compute",
  "exactly 2 ranks, peer-accessible, overlap 1; scratch sized by the full message; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate, bit-identical across ranks and to the wide row",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT + 1,
  ROW(cArExactLane), ROW(tArLane), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_2rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_oneshot_2rank_wht6_lane", "all_reduce", "ar",
  "all-reduce (sum) with a Walsh-Hadamard-rotated 6-bit wire payload -- lossy; narrow launch "
  "beside compute",
  "exactly 2 ranks, peer-accessible, overlap 1; groups of 64 elements; cudagraph-safe",
  "bf16 / fp16 payload, 6 bits + bf16 scale per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT + 1,
  ROW(cArWht6Lane), ROW(tArLane), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_2rank_wht6, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_gather_3rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch, the wide family's per-block handshake",
  "exactly 3 ranks, all peer-accessible; y is [3 * numel] with rank r's input at r * numel, or "
  "row-interleaved [rows][3][row] when `row` is given; shares the wide family's "
  "flag/seq/scratch set; cudagraph-safe",
  "any payload -- the bytes are moved, never summed, so the (u32 id, f32 logit) pairs the "
  "vocab-parallel sampler gathers survive intact",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArGather3), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_gather_3rank, nullptr, nullptr, nullptr,
  r4d_shape_all_gather },
{ "ar_oneshot_3rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch, per-block handshake",
  "exactly 3 ranks, all peer-accessible; one scratch slot per (parity, source rank); in place; "
  "above " R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements ar_twoshot_3rank_exact outranks it; "
  "cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArExact3), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_3rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },
{ "ar_oneshot_4rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch, per-block handshake",
  "exactly 4 ranks, all peer-accessible; one scratch slot per (parity, source rank); in "
  "place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArExact4), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_4rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_oneshot_8rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch, per-block handshake",
  "exactly 8 ranks, all peer-accessible; one scratch slot per (parity, source rank); in "
  "place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArExact8), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_oneshot_8rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_twoshot_3rank_ti8", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather; both hops ride the tiered-int8 wire (double "
  "quant, the owner storing the decode of its own words) -- lossy",
  "exactly 3 ranks, all peer-accessible; groups of 32, split over the shards as evenly as they "
  "divide (the first groups % 3 take one more); under min_bytes the exact one-shot; cudagraph-safe",
  "bf16 / fp16 payload, int8 + fp16 scale + u32 record per group on the wire, fp32 accumulate in "
  "ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArTi8Ts3), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_3rank_ti8, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_twoshot_4rank_ti8", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather; both hops ride the tiered-int8 wire (double "
  "quant, the owner storing the decode of its own words) -- lossy",
  "exactly 4 ranks, all peer-accessible; groups of 32 per shard (numel tiles 128); cudagraph-safe",
  "bf16 / fp16 payload, int8 + fp16 scale + u32 record per group on the wire, fp32 accumulate in "
  "ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT,
  ROW(cArTi8Ts4), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_4rank_ti8, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

/* The row above at `overlap` 1: the same entry point and the same arithmetic, a launch that leaves
 * the CUs to the compute running beside it (tArTs4Lane). */
{ "ar_twoshot_4rank_ti8_lane", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather on the tiered-int8 wire -- lossy; narrow launch "
  "beside compute",
  "exactly 4 ranks, all peer-accessible, overlap 1; groups of 32 per shard (numel tiles 128); "
  "cudagraph-safe",
  "bf16 / fp16 payload, int8 + fp16 scale + u32 record per group on the wire, fp32 accumulate in "
  "ascending rank order, bit-identical across ranks and to the wide row",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_ONESHOT + 1,
  ROW(cArTi8Ts4Lane), ROW(tArTs4Lane), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_4rank_ti8, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

/* ---------------------------------------------------------------------------------- GEMM ---- */
{ "gemm_bf16_nt_m16", "gemm_nt", "gemm",
  "skinny GEMM C[M,N] = A[M,K] @ W[N,K]^T, split-K reduced in LDS",
  "M <= 16; K divisible by SK and K/SK by 256; N, K otherwise runtime",
  "bf16 A / W / C, fp32 accumulate, deterministic reduction",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_M16,
  ROW(cGemmNt), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_bf16_m16, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt },

{ "gemm_bf16_nt_m64", "gemm_nt", "gemm",
  "skinny GEMM C[M,N] = A[M,K] @ W[N,K]^T with 16x16x16 WMMA, split-K reduced in LDS",
  "any M -- MT row tiles a block and grid.y covers the rest; K divisible by SK*16; N, K runtime",
  "bf16 A / W / C, fp32 accumulate, deterministic reduction",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNt64), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_bf16_m64, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt },

{ "gemm_bf16_nt_tiled", "gemm_nt", "gemm",
  "the same GEMM at prefill row counts: eight waves a block over an LDS-staged 64-deep K tile, "
  "the next tile prefetched in registers while the current one is computed",
  "M > 64; N >= 128; K a multiple of 64; fragment rows past M or N clamped, never stored",
  "bf16 A / W / C, fp32 accumulate. BYTE-IDENTICAL to gemm_bf16_nt_m64: the same WMMA sequence "
  "a tile and the same split count, the slices summed in the same order from 0.f",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_BF16_TILED,
  ROW(cGemmNtTiled), ROW(tGemmBf16Tiled), nullptr,
  nullptr, nullptr, r4d_rad_gemm_bf16_tiled, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt },

{ "gemm_w4a16_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with a 4-bit weight, 16x16x16 WMMA, split-K reduced in LDS",
  "M <= 64; N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order",
  "f16 A, 4-bit W with an f16 scale and integer zero per 128 K, bf16 C, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW4), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w4a16_m64, nullptr, r4d_rad_layout_w4, r4d_rad_relayout_w4,
  nullptr, r4d_rad_unrelayout_w4 },

{ "gemm_w5a16_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with a 5-bit asymmetric weight (ParoQuant int5), 16x16x16 f16 WMMA, split-K reduced "
  "in LDS; rows past 64 in chunks of 64, a partitioned weight (pb1/pb2) one launch a part",
  "N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order, a nibble "
  "plane then a fifth-bit plane",
  "f16 A (parts side by side), 5-bit W with an f16 scale and integer zero per 128 K, bf16 C, fp32 "
  "accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW5), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w5a16_m64, nullptr, r4d_rad_layout_w5, r4d_rad_relayout_w5,
  nullptr, r4d_rad_unrelayout_w5 },

{ "gemm_w4a16_asym_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with a 4-bit ASYMMETRIC weight (z-lab's ParoQuant int4), the 5-bit row's kernel with "
  "no fifth-bit plane; rows past 64 in chunks of 64, a partitioned weight (pb1/pb2) one launch a part",
  "N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order, unsigned nibbles",
  "f16 A (parts side by side), 4-bit W with an f16 scale and integer zero per 128 K, bf16 C, fp32 "
  "accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW4Asym), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w4a16asym_m64, nullptr, r4d_rad_layout_w4, r4d_rad_relayout_w4,
  nullptr, r4d_rad_unrelayout_w4 },

{ "gemm_w4a8i8_tiled", "gemm_nt_q", "gemm",
  "the 4-bit ASYMMETRIC weight x int8 activation GEMM for prefill row counts (z-lab's ParoQuant int4): "
  "int8 WMMA over (code - zero), the weight and activation scales folded per 128 of K",
  "N a multiple of 16; K a multiple of 128; the activation scale per (row, 128 of K); weight in the "
  "f16 kernel's fragment order; a partitioned weight (pb1/pb2) one launch a part",
  "I8 A (parts side by side) with an f32 scale per (row, part, 128), 4-bit W with an f16 scale and "
  "integer zero per 128 K, bf16 C, int32 accumulate per 128 of K",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW4A8I8), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_w4a8i8_tiled, r4d_rad_gemm_mxfp4a8_tiled_scratch,
  r4d_rad_layout_w4, r4d_rad_relayout_w4, nullptr, r4d_rad_unrelayout_w4 },

{ "gemm_w5a8i8_tiled", "gemm_nt_q", "gemm",
  "the 5-bit ASYMMETRIC weight x int8 activation GEMM for prefill row counts (ParoQuant int5): int8 "
  "WMMA over (code - zero), the weight and activation scales folded per 128 of K",
  "N a multiple of 16; K a multiple of 128; the activation scale per (row, 128 of K); weight in the "
  "int5 kernel's two planes (nibbles, then fifth bits); a partitioned weight (pb1/pb2) one launch a part",
  "I8 A (parts side by side) with an f32 scale per (row, part, 128), 5-bit W with an f16 scale and "
  "integer zero per 128 K, bf16 C, int32 accumulate per 128 of K",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW5A8I8), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_w5a8i8_tiled, r4d_rad_gemm_mxfp4a8_tiled_scratch,
  r4d_rad_layout_w5, r4d_rad_relayout_w5, nullptr, r4d_rad_unrelayout_w5 },

{ "gemm_w8a16_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with an 8-bit weight, 16x16x16 WMMA, split-K reduced in LDS",
  "M <= 64; N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order, "
  "one packed block per 32 K",
  "f16 A, 8-bit W with an f16 scale and integer zero per 128 K, bf16 C, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW8), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w8a16_m64, nullptr, r4d_rad_layout_w8a16,
  r4d_rad_relayout_w8a16 },

{ "gemm_w4a8_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with a 4-bit weight and an 8-bit activation, 16x16x16 int8 WMMA, split-K reduced "
  "in LDS. dtype w4a8_asym takes the asymmetric grid, which reads the nibble UNSIGNED and needs "
  "the a_sum operand",
  "M <= 64; N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order "
  "and SHARED byte for byte with gemm_w4a16_nt_m64; activation byte-shuffled inside each k step",
  "int8 A with an f32 per-row scale, 4-bit W with an f16 scale per 128 K, bf16 C, int32 "
  "accumulate per group",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW4A8), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w4a8_m64, nullptr, r4d_rad_layout_w4, r4d_rad_relayout_w4,
  nullptr, r4d_rad_unrelayout_w4 },

/* ------------------------------------------------------------------------ block-scaled fp8
 *
 * These read block-scaled FP8 checkpoints as shipped: `<linear>.weight` F8_E4M3 [N][K] with
 * `<linear>.weight_scale_inv` BF16 on a [ceil(N/128)][ceil(K/128)] grid. No repacking or scale
 * conversion, so a .rad costs what the checkpoint does, instead of a bf16 dequantisation at twice
 * the weight bytes.
 */
{ "gemm_fp8a16_nt_m1", "gemm_nt_q", "gemm",
  "block-scaled FP8 matvec against the checkpoint's own weight bytes: one wave a row, 512 "
  "coalesced bytes a pass, RB rows a wave",
  "M <= 8 (libr4d's kMaxMt; one weight read serves all M rows); K a multiple of 128 (one scale "
  "per 128 columns, and a ragged last block reads past the plane); N runtime including a ragged "
  "wave",
  "bf16 A read DIRECTLY -- no quantise launch -- E4M3 W with a bf16 scale per 128x128 tile, bf16 "
  "C, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_M16,
  ROW(cGemmNtFp8A16), ROW(tMv), nullptr,
  nullptr, nullptr, r4d_rad_gemm_fp8a16_m1, nullptr,
  r4d_rad_layout_fp8_rowmajor, nullptr,
  r4d_shape_gemm_nt_q },

{ "gemm_fp8a8_nt_m16", "gemm_nt_q", "gemm",
  "block-scaled FP8 W8A8 GEMM at the speculative-verify width: 16-wide token tiles, NT up to 4 of "
  "them, 16x16x16 fp8 WMMA, split-K over the caller's scratch",
  "M <= 64, libr4d's max_m: NT takes up to 4 tiles of 16. K a multiple of 128 -- the narrow tile "
  "stages a whole 256-byte cacheline a row, and a K that is not a multiple of 128 makes a split "
  "slice overrun its own end; N a multiple of 16, "
  "because the weight lands on the A operand and the epilogue transposes on the way out",
  "E4M3 A with an f32 scale per (row, 128 K), E4M3 W with a bf16 scale per 128x128 tile, bf16 C. "
  "The split-K finisher sums its planes in INDEX order, so a run repeats exactly; but the split "
  "COUNT is chosen from M, so the same row computed at M=1 and at M=8 is not bit-identical. That "
  "is why a speculative decode diverges from a non-speculative one on a near-tie. NOTHING PINS IT "
  "TODAY: --deterministic reaches placement and the sampler, not this kernel",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_M16,
  ROW(cGemmNtFp8A8M16), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_fp8a8_m16, r4d_rad_gemm_fp8a8_m16_scratch,
  r4d_rad_layout_fp8_block, r4d_rad_relayout_fp8_block,
  r4d_shape_gemm_nt_q, r4d_rad_unrelayout_fp8_block },

{ "gemm_fp8a8_gated_nt_m16", "gemm_nt_q_gated", "gemm",
  "the narrow fp8 GEMM with the SwiGLU and the fp8 quantiser folded into its epilogue: the block's "
  "waves split in half over the gate rows and the up rows, so a block emits one whole 128-column "
  "scale group and the quantiser's amax closes inside it",
  "M <= 64; K a multiple of 256 for the reason the ungated narrow row gives; N a multiple of 256, "
  "which is the fused width being a multiple of the 128-column scale group; act silu only",
  "E4M3 A with an f32 scale per (row, 128 K), E4M3 W with a bf16 scale per 128x128 tile, and the "
  "QUANTISER's outputs: E4M3 codes, an f32 scale per (row, 128 columns) and an optional bf16 "
  "product. Byte-identical to gemm_nt_q followed by gated_quant_fp8 -- the accumulator is rounded "
  "to bf16 before the gate and the multiply, and the amax is an fmaxf reduction, which is exact "
  "and order-free however it is regrouped across the waves",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGemmNtQGatedFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_fp8a8_gated_m16, nullptr,
  r4d_rad_layout_fp8_block, r4d_rad_relayout_fp8_block,
  r4d_shape_gemm_nt_q_gated, r4d_rad_unrelayout_fp8_block,
  /* The byte-identity claim above, as a chain of two rows of this plugin (r4d_fuse.cpp). */
  r4d_fuse_gemm_nt_q_gated },

{ "gemm_fp8a8_tiled", "gemm_nt_q", "gemm",
  "the same GEMM for prefill row counts: 128x128 tile, eight waves of 2 weight x 4 activation "
  "tiles, the weight read straight into registers and the activation staged through LDS, one K "
  "tile ahead; a lane stores eight consecutive columns of one row",
  "any M; K and N multiples of 128 -- the fold at the end of the K loop assumes a whole scale "
  "block, and the weight's block scale is read as one scalar for the block's 128 N rows",
  "E4M3 A with an f32 scale per (row, 128 K), E4M3 W with a bf16 scale per 128x128 tile, bf16 C; "
  "each 128-K block's product folded into the f32 accumulator with one fma",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_TILED,
  ROW(cGemmNtFp8A8Tiled), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_fp8a8_tiled, nullptr,
  r4d_rad_layout_fp8_block, r4d_rad_relayout_fp8_block,
  r4d_shape_gemm_nt_q, r4d_rad_unrelayout_fp8_block },

{ "gemm_fp8a8_bias_nt_m16", "gemm_nt_q_bias", "gemm",
  "the narrow block-scaled FP8 W8A8 GEMM with a bias per output column in its store: the "
  "unbiased narrow tile, never split, writing C at its own row pitch",
  "M <= 64 (the narrow tile's NT of up to 4); K a multiple of 128 and N of 16, the unbiased "
  "narrow row's band; C may be a column slice of a wider buffer",
  "E4M3 A with an f32 scale per (row, 128 K), E4M3 W with a bf16 scale per 128x128 tile, a bf16 "
  "bias added to the f32 accumulator before its one rounding, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGemmNtQBiasFp8A8M16), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_fp8a8_bias_m16, nullptr,
  r4d_rad_layout_fp8_block, r4d_rad_relayout_fp8_block,
  r4d_shape_gemm_nt_q_bias, r4d_rad_unrelayout_fp8_block },

{ "gemm_i8a8_nt_m16", "gemm_nt_q", "gemm",
  "gemm_fp8a8_nt_m16 at INT8 codes: the same narrow tile, staging and split-K, with the iu8 WMMA "
  "for the fp8 one and the weight's scale per row",
  "the fp8 row's: M <= 64; K a multiple of 128; N a multiple of 16",
  "i8 A with an f32 scale per (row, 128 K), i8 W with a bf16 scale per ROW per 128 K, bf16 C; "
  "int32 accumulation within a 128-K block, folded to f32 at its end. dtype i8a8h: the weight "
  "rotated by a 128-wide Hadamard, the 1/128 in its scale, against had_quant_act_i8g's activation",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_M16,
  ROW(cGemmNtI8A8M16), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_i8a8_m16, r4d_rad_gemm_i8a8_m16_scratch,
  r4d_rad_layout_i8_block, r4d_rad_relayout_i8_block,
  nullptr, r4d_rad_unrelayout_i8_block },

{ "gemm_i8a8_tiled", "gemm_nt_q", "gemm",
  "gemm_fp8a8_tiled at INT8 codes: the same 128x128 prefill tile, the iu8 WMMA for the fp8 one and "
  "the weight's scale per row, eight a lane a tile in the fold",
  "the fp8 row's: any M; K and N multiples of 128",
  "i8 A with an f32 scale per (row, 128 K), i8 W with a bf16 scale per ROW per 128 K, bf16 C. "
  "dtype i8a8h: the weight rotated by a 128-wide Hadamard, the 1/128 in its scale, against "
  "had_quant_act_i8g's activation",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_TILED,
  ROW(cGemmNtI8A8Tiled), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_i8a8_tiled, nullptr,
  r4d_rad_layout_i8_block, r4d_rad_relayout_i8_block,
  nullptr, r4d_rad_unrelayout_i8_block },

/* Band read off the entry point's own rejection tests, not a kernel-side row. */
{ "gemm_w8a8_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with an 8-bit weight and an 8-bit activation: the promoted linears of the INT4 "
  "target (every down_proj, the attention o_proj) and lm_head",
  "M <= 64; N a multiple of 16; K divisible by SK*128; same fragment order as the 4-bit kernel at "
  "twice the radix; f16-ONLY scale plane, no zero point",
  "int8 A with an f32 per-row scale, int8 W with an f16 scale per 128 K, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW8A8), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w8a8_m64, nullptr, r4d_rad_layout_w8a8, r4d_rad_relayout_w8a8 },

{ "gemm_w2a8_nt", "gemm_nt_q", "gemm",
  "skinny GEMM with a 2-bit weight and an 8-bit activation -- the DFlash2 draft head",
  "N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order, one packed "
  "block per 128 K; activation and its group sums from quant_act_i8's asum form",
  "int8 A with an f32 per-row scale, UNSIGNED 2-bit W with an f16 scale and integer zero per "
  "128 K, bf16 C, int32 accumulate per group",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtW2A8), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_w2a8, nullptr, r4d_rad_layout_w2a8, r4d_rad_relayout_w2a8 },

{ "gemm_mxfp4a8_nt_m64", "gemm_nt_q", "gemm",
  "skinny GEMM with an OCP-MXFP4 weight and an fp8 activation, 16x16x16 fp8 WMMA",
  "M <= 64; N a multiple of 16; K divisible by SK*32; weight pre-permuted into fragment order "
  "(32 lanes x uint32 per n-tile/k-step), one E8M0 exponent per 32 K in [K/32][N] order, plus a "
  "per-row reference exponent",
  "e4m3 A with an f32 per-row scale, e2m1 W with an E8M0 scale per 32 K folded into the element "
  "against a per-row reference, bf16 C, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_SKINNY,
  ROW(cGemmNtMxfp4A8), ROW(tGemmSkinny), r4d_tune_valid_gemm,
  nullptr, nullptr, r4d_rad_gemm_mxfp4a8_m64, nullptr, r4d_rad_layout_mxfp4,
  r4d_rad_relayout_mxfp4 },

{ "gemm_mxfp4a8_decode", "gemm_nt_q", "gemm",
  "the OCP-MXFP4 GEMM for the DECODE band: split-K across workgroups with a fused last-arriver "
  "reduction, the weight unpacked through LDS, streaming weight loads",
  "M <= 128; N a multiple of 16; K a multiple of 64; the skinny kernel's fragment-order weight, "
  "[K/32][N] E8M0 exponents and per-row reference exponent; the split partial in the op's scratch",
  "e4m3 A with an f32 scale per row or per (row, 128), e2m1 W with an E8M0 scale per 32 K folded "
  "into the element against a per-row reference, bf16 C, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_MXDEC,
  ROW(cGemmMxfp4A8Decode), ROW(tGemmMxfp4Decode), nullptr,
  nullptr, nullptr, r4d_rad_gemm_mxfp4a8_decode, r4d_rad_gemm_mxfp4a8_decode_scratch,
  r4d_rad_layout_mxfp4, r4d_rad_relayout_mxfp4 },

{ "gemm_mxfp4a8_tiled", "gemm_nt_q", "gemm",
  "the OCP-MXFP4 GEMM tiled for PREFILL row counts: the activation re-tiled into fragment order in "
  "the op's scratch, its fragments read straight into registers, the weight unpacked once per K "
  "slab into LDS for a 256-row block",
  "any M; N a multiple of 16; K a multiple of 64; the skinny kernel's fragment-order weight, "
  "[K/32][N] E8M0 exponents and per-row reference exponent; the re-tiled activation in the op's "
  "scratch",
  "e4m3 A with an f32 scale per row or per (row, 128), e2m1 W with an E8M0 scale per 32 K folded "
  "into the element against a per-row reference, bf16 C, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_TILED,
  ROW(cGemmMxfp4A8Tiled), ROW(tGemmMxfp4Tiled), nullptr,
  nullptr, nullptr, r4d_rad_gemm_mxfp4a8_tiled, r4d_rad_gemm_mxfp4a8_tiled_scratch,
  r4d_rad_layout_mxfp4, r4d_rad_relayout_mxfp4 },

/* Band read off the entry point's own rejection tests. */
{ "gemm_w4a8_prefill", "gemm_nt_q", "gemm",
  "the 4-bit GEMM for PREFILL row counts, where the matrix pipe rather than the weight stream is "
  "the constraint: the activation is stored half-major in LDS so one ds_load_b128 fetches two k "
  "steps, and the group fold is plain v_fmac_f32 so the VOPD packer can pair it",
  "any M; N a multiple of the variant's BN (256 at the default); K a multiple of the scale group",
  "int8 A with an f32 per-row scale, signed 4-bit W with an f16 scale per 128 K, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_PREFILL,
  ROW(cGemmW4A8Prefill), ROW(tGemmPrefill), nullptr,
  nullptr, nullptr, r4d_rad_gemm_w4a8_prefill, nullptr, r4d_rad_layout_w4, r4d_rad_relayout_w4,
  nullptr, r4d_rad_unrelayout_w4 },

/* Band read off the entry point's own rejection tests. */
{ "gemm_w4a8_tiled", "gemm_nt_q", "gemm",
  "the 4-bit GEMM tiled for large M, where the skinny kernel gives a block 64 rows and so re-reads "
  "the weight M/64 times. dtype w4a8_asym takes the asymmetric grid and needs a_sum",
  "any M; N a multiple of 256; K a multiple of the scale group; same packed weight and same "
  "pre-shuffled activation as the skinny kernel",
  "int8 A with an f32 per-row scale, 4-bit W with an f16 scale per 128 K, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_TILED,
  ROW(cGemmW4A8Tiled), ROW(tGemmTiled), nullptr,
  nullptr, nullptr, r4d_rad_gemm_w4a8_tiled, nullptr, r4d_rad_layout_w4, r4d_rad_relayout_w4,
  nullptr, r4d_rad_unrelayout_w4 },

/* Band read off the entry point's own rejection tests. */
{ "gemm_w8a8_tiled", "gemm_nt_q", "gemm",
  "the 8-bit GEMM tiled for large M; inherits the 4-bit kernel's unpack-at-staging path, whose LDS "
  "buffer already holds bytes",
  "any M; N a multiple of 256; K a multiple of the scale group",
  "int8 A with an f32 per-row scale, int8 W with an f16 scale per 128 K, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_TILED,
  ROW(cGemmW8A8Tiled), ROW(tGemmTiled), nullptr,
  nullptr, nullptr, r4d_rad_gemm_w8a8_tiled, nullptr, r4d_rad_layout_w8a8, r4d_rad_relayout_w8a8 },

{ "dequant_w4_bf16", "dequant_w4", "gemm",
  "the packed 4-bit weight back to row-major bf16, undoing the WMMA fragment order",
  "N a multiple of 16, K a multiple of 64; group must divide K; one thread per packed dword",
  "4-bit in with an f16 scale and a zero in the dword's high half, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDequantW4Bf16), NOTUNE,
  nullptr, nullptr, r4d_rad_dequant_w4, nullptr, nullptr, nullptr,
  r4d_shape_dequant_w4 },

{ "scale_rows_bf16", "scale_rows", "elem",
  "one scalar a row, optionally through a sigmoid, broadcast across the row",
  "any M and n; the scalar stream is [M] or [M, 1]",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cScaleRows), NOTUNE,
  nullptr, nullptr, r4d_rad_scale_rows, nullptr, nullptr, nullptr,
  r4d_shape_scale_rows },

{ "router_sigmoid_topk_f32", "router_sigmoid_topk", "moe",
  "sigmoid correction-bias routing", "single group, experts <=512, top_k<=32",
  "f32 logits, bias and weights; i32 ids", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRouterSigmoid), NOTUNE,
  nullptr, nullptr, r4d_rad_router_sigmoid_topk, nullptr, nullptr, nullptr,
  r4d_shape_router_sigmoid_topk },
{ "router_sigmoid_topk_scatter_f32", "router_sigmoid_topk_scatter", "moe",
  "sigmoid correction-bias routing, the stable counting sort and the ungated shared gate in one "
  "launch", "single group, experts <=512, top_k<=32, M*top_k<=1024",
  "f32 logits, bias and weights; i32 ids, sorted slots, offsets and counts; bf16 gate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRouterSigmoidScatter), NOTUNE,
  nullptr, nullptr, r4d_rad_router_sigmoid_topk_scatter, nullptr, nullptr, nullptr,
  r4d_shape_router_sigmoid_topk_scatter, nullptr,
  /* `router_sigmoid_topk` + `moe_scatter` (+ `fill` with the gate) (r4d_fuse.cpp). */
  r4d_fuse_router_sigmoid_topk_scatter },
{ "router_topk_bf16", "router_topk", "moe",
  "softmax over a row of expert logits and the exact top-k of it, one wave a row",
  "n_expert <= 512, top_k <= min(n_expert, 32); bf16 logits and weights, i32 ids",
  "bf16 in, bf16 weights + i32 expert ids out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRouterTopK), NOTUNE,
  nullptr, nullptr, r4d_rad_router_topk, nullptr, nullptr, nullptr,
  r4d_shape_router_topk },

{ "router_topk_scatter_bf16", "router_topk_scatter", "moe",
  "the router's top-k and the stable counting sort that follows it, in one launch",
  "n_expert <= 512, top_k <= min(n_expert, 32), and M*top_k <= 1024 so the sort is the "
  "single-workgroup form; bf16 logits and weights, i32 everything else",
  "bf16 in, bf16 weights + i32 ids, sorted slots, offsets and counts out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRouterTopK), NOTUNE,
  nullptr, nullptr, r4d_rad_router_topk_scatter, nullptr, nullptr, nullptr,
  r4d_shape_router_topk_scatter },

{ "router_gemm_topk_scatter_bf16", "router_gemm_topk_scatter", "moe",
  "the router GEMM, its top-k and the stable counting sort in one launch -- the last workgroup "
  "of the GEMM ranks -- and optionally the shared expert's one-column gate beside it",
  "M <= 64 and M*top_k <= 1024; n_expert <= 512, top_k <= min(n_expert, 32); K a multiple of "
  "16 times the split the bf16 skinny GEMM's law picks",
  "bf16 activation and weights, fp32 accumulate; bf16 logits, gate and weights, i32 ids, "
  "sorted slots, offsets and counts out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRouterGemmTopkScatter), NOTUNE,
  r4d_rad_router_gemm_topk_scatter_init, r4d_rad_router_gemm_topk_scatter_fini,
  r4d_rad_router_gemm_topk_scatter, nullptr, nullptr, nullptr,
  r4d_shape_router_gemm_topk_scatter },

{ "moe_scatter_i32", "moe_scatter", "moe",
  "a stable counting sort of the routing slots by expert, with the offsets and the histogram",
  "n_expert <= 256; one launch at M*top_k <= 1024 and four above it, and the four exist to keep "
  "the sort stable without an atomic deciding the order",
  "i32 throughout",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeScatter), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_scatter, r4d_rad_moe_scatter_scratch, nullptr, nullptr,
  r4d_shape_moe_scatter },

{ "moe_scatter_inv_i32", "moe_scatter_inv", "moe",
  "moe_scatter's stable counting sort, also writing every slot's row in the sort",
  "n_expert <= 512; one launch at M*top_k <= 1024 and three above it",
  "i32 throughout",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeScatter), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_scatter_inv, r4d_rad_moe_scatter_scratch, nullptr, nullptr,
  r4d_shape_moe_scatter_inv },

{ "moe_gemm_mxfp4a8", "moe_gemm_q", "moe",
  "canonical MXFP4 experts decoded in packed words into exact FP8 WMMA tiles, scales folded per 32 values",
  "K multiple 128, parity tables, token and sorted order, parts and narrower odd experts",
  "E2M1 and E8M0 weights, E4M3 activations and f32 scales, bf16 output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cMoeMx), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_mxfp4, nullptr, r4d_rad_layout_moe_mxfp4, nullptr,
  r4d_shape_moe_mxfp4 },

{ "moe_gemm_fp8a8", "moe_gemm_q", "moe",
  "the grouped block-scaled fp8 GEMM: one contiguous range of expert-sorted rows a block, looping "
  "over the experts that range covers",
  "N a multiple of 64, K a multiple of 128, group 128; 16-row tiles a run a block while runs "
  "average under a row and a half, 32-row tiles a run a block above, with the M tiles past a "
  "run skipped",
  "E4M3 activation with an f32 scale per 128 of K, E4M3 weight with a bf16 scale per 128x128 "
  "tile, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmQ), NOTUNE,
  /* The row-major fp8 layout: the checkpoint's own bytes, shared with the fp8a16 matvec.
   * `moe_gemm_q` puts the weight at operand 2 and its scale at 3, as `gemm_nt_q` does, so the hook
   * needs no MoE arm. Not the dense fp8a8 GEMMs' fragment order: this kernel stages through LDS
   * because its rows are gathered, so a permutation buys nothing. rad-convert refuses a declared
   * fp8 weight with no layout hook, so the hook is what tells it the bytes are already right. */
  nullptr, nullptr, r4d_rad_moe_gemm_q, r4d_rad_moe_gemm_q_scratch,
  r4d_rad_layout_moe_fp8, nullptr,
  r4d_shape_moe_gemm_q },

{ "moe_gemm_w4a8", "moe_gemm_q", "moe",
  "the same grouped GEMM with the expert weight at FOUR BITS: row-major packed nibbles expanded to "
  "E4M3 on the way into LDS, so the WMMA inner loop is the fp8 kernel's untouched",
  "N a multiple of 64, K a multiple of 128, group 128; the same 16-row/32-row band, and the same "
  "argument for it",
  "E4M3 activation with an f32 scale per 128 of K, symmetric int4 weight (codes -8..7) with a "
  "bf16 scale per row per 128 of K, bf16 out; dtype w4a8h is the same plane with a 128-wide "
  "Hadamard rotation along K baked into both, and pairs with had_quant_act_fp8; dtype w4nla8h is "
  "w4a8h with the codes indexing libquant's w4nl levels, and w4nl64a8h / w4nl32a8h are w4nla8h "
  "with an E4M3 scale per row per 64 / 32 of K under a fixed 2^-13; w5nl64a8h is w4nl64a8h with "
  "five-bit codes into libquant's w5nl levels, a sign bit plane after each row's nibbles",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmW4), NOTUNE,
  /* No shape function: the weight is a plugin-private packed plane whose byte count
   * r4d_layout.cpp owns (see r4d_shapes.cpp). r4d_selftest checks the decode against the
   * relayout that produced the bytes.
   *
   * Four bits for capacity: Qwen3.8-Flash-Next's 512 experts over 48 layers are 112.8 GiB at
   * E4M3, too much for two 32 GB cards even with expert parallelism, and a routed weight cannot
   * be served from the file tier. At four bits the share fits VRAM plus a few GiB of pinned host.
   *
   * The unrelayout hook is still provided: it yields the canonical codes plane, so an oracle can
   * hand a dense reference the codes even though callers cannot allocate the packed plane. */
  nullptr, nullptr, r4d_rad_moe_gemm_w4a8, r4d_rad_moe_gemm_w4a8_scratch,
  r4d_rad_layout_moe_w4, r4d_rad_relayout_moe_w4, nullptr, r4d_rad_unrelayout_moe_w4 },

{ "moe_gemm_fp8a8_mod4", "moe_gemm_q_mod4", "moe",
  "moe_gemm_fp8a8 over four classes of experts by number mod 4: the same kernel, each run reading "
  "its class's table",
  "as moe_gemm_fp8a8, every class",
  "as moe_gemm_fp8a8",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmQ), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_q_mod4, r4d_rad_moe_gemm_q_scratch,
  r4d_rad_layout_moe_fp8_mod4, nullptr,
  r4d_shape_moe_gemm_q_mod4 },

{ "moe_gemm_w4a8_mod4", "moe_gemm_q_mod4", "moe",
  "moe_gemm_w4a8 over four classes of experts by number mod 4: the same kernel, each run reading "
  "its class's table",
  "as moe_gemm_w4a8, every class",
  "as moe_gemm_w4a8",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmW4), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_w4a8_mod4, r4d_rad_moe_gemm_w4a8_scratch,
  r4d_rad_layout_moe_w4_mod4, r4d_rad_relayout_moe_w4_mod4, nullptr,
  r4d_rad_unrelayout_moe_w4_mod4 },

{ "moe_gemm_bf16", "moe_gemm", "moe",
  "the grouped GEMM with unquantised experts: one run of expert-sorted rows a block, bf16 WMMA over "
  "an LDS-staged 64-deep K pass with the next one prefetched in registers",
  "N a multiple of 64, K a multiple of 64; the 16-row decode tiles and the 32-row run-aligned "
  "prefill tiles of moe_gemm_q, by the same rule",
  "bf16 activation, bf16 weight as the checkpoint stores it ([N, K] row-major, no layout hook), "
  "f32 accumulate over the whole of K, bf16 out rounded to nearest-even",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmBf16), NOTUNE,
  /* No layout hook and no shape function. The plane is the checkpoint's own bytes (identity
   * layout), so the file tier can stage an expert and hand it to this kernel unchanged. libref
   * describes the op for tools. */
  nullptr, nullptr, r4d_rad_moe_gemm_bf16, r4d_rad_moe_gemm_bf16_scratch },

{ "moe_gather_bf16", "moe_gather", "moe",
  "the weighted scatter back to token order, summing a token's slots in ascending slot order",
  "any M and n; the inverse permutation is built in scratch rather than searched",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGather), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gather, r4d_rad_moe_gather_scratch, nullptr, nullptr,
  r4d_shape_moe_gather },

{ "moe_gather_inv_bf16", "moe_gather_inv", "moe",
  "moe_gather over the slot table moe_scatter_inv wrote: no prepass and no scan of the sort",
  "any M and n; the decode kernel up to 512 slots, the prefill forms above, as moe_gather",
  "bf16 in and out, f32 or bf16 routing weights",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGather), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gather_inv, nullptr, nullptr, nullptr,
  r4d_shape_moe_gather_inv },

{ "rowtopk_bf16", "row_topk", "select",
  "exact per-row top-R of a bf16 matrix, descending, ties to the lower column",
  "any M, N a multiple of 8, R <= 32; two launches, caller-sized partial buffer",
  "bf16 in, f32 values + i32 columns out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRowTopK), ROW(tTopk), nullptr,
  nullptr, nullptr, r4d_rad_rowtopk, r4d_rad_rowtopk_scratch, nullptr, nullptr,
  r4d_shape_row_topk },

{ "logit_rerank", "logit_rerank", "select",
  "R candidates a row rescored exactly against the full-precision head",
  "any M, R <= 32, bf16 x and the E4M3 lm_head plane logits_gemm_fp8 stores; one block a row with "
  "the waves spread over the candidates, one launch, no scratch",
  "bf16 in, f32 values + i32 ids out, ties to the lower id",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLogitRerank), NOTUNE,
  nullptr, nullptr, r4d_rad_logit_rerank, nullptr,
  r4d_rad_layout_fp8_logits, r4d_rad_relayout_fp8_logits,
  /* No opd_shape, as for logits_gemm_fp8: the declared shape and the stored plane (scales in
   * each row's tail) differ, so a buffer of the declared shape would be read past its end. */
  nullptr, r4d_rad_unrelayout_fp8_logits },

{ "logit_rerank_i8", "logit_rerank", "select",
  "logit_rerank over an INT8 head: the same candidates rescored against logits_gemm_i8's rows",
  "logit_rerank's: any M, R <= 32, bf16 x and the i8 lm_head plane logits_gemm_i8 stores",
  "bf16 in, f32 values + i32 ids out, ties to the lower id",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLogitRerank), NOTUNE,
  nullptr, nullptr, r4d_rad_logit_rerank_i8, nullptr,
  r4d_rad_layout_i8_logits, r4d_rad_relayout_i8_logits,
  /* No opd_shape, for logit_rerank's reason: the declared shape and the stored row differ. */
  nullptr, r4d_rad_unrelayout_i8_logits },

{ "rowtopk_merge", "row_topk_merge", "select",
  "the global top-R of world_size all-gathered per-rank candidate sets",
  "any M, R <= 32; `gathered` is [M, world_size * R, 2] f32 pairs, an int32 id in the first slot; "
  "one wave a row, one launch, no scratch",
  "f32 pairs in, f32 values + i32 ids out, ties to the lower id",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRowTopKMerge), NOTUNE,
  nullptr, nullptr, r4d_rad_rowtopk_merge, nullptr, nullptr, nullptr,
  r4d_shape_row_topk_merge },

/* ------------------------------------------------- quantisation, norms and the fusions ------ */

/* The input quantiser of both fp8a8 GEMMs: `activation_scheme: "dynamic"` means the checkpoint
 * ships no activation scale, so one is computed every step. This standalone form serves inputs
 * that are some GEMM's output; where a producer holds the row, `rmsnorm_quant_fp8` or
 * `gated_quant_fp8` below quantises on its pass. */
{ "quant_act_fp8", "quant_act_fp8", "quant",
  "per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation",
  "n a multiple of 4 -- a lane owns four columns, so the store is one dword and the wave's is 128 "
  "contiguous bytes; group 128, which is the WEIGHT scale's block width and therefore a property "
  "of the checkpoint rather than a knob; a ragged last group is written zero-padded",
  "bf16 in, E4M3 out in PLAIN ROW-MAJOR order with an f32 scale per (row, 128 columns)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_act_fp8, nullptr, nullptr, nullptr,
  r4d_shape_quant_act_fp8 },

/* The rotated twin, for the four-bit expert weight only. At E4M3 a rotation is slightly worse
 * (four exponent bits already carry the outliers it removes); at four bits, sixteen levels spent
 * covering one outlier channel is the whole error. The trade pays off only with GPTQ: rotation
 * alone makes round-to-nearest int4 substantially worse. */
{ "had_quant_act_fp8", "had_quant_act_fp8", "quant",
  "a 128-wide Hadamard along K, then per-(row, 128-column) symmetric E4M3 quantisation",
  "n a multiple of 128 -- a rotation block cannot straddle the end of a row -- and group 128, "
  "which is both the rotation width and the weight scale's block width",
  "bf16 in, E4M3 out in PLAIN ROW-MAJOR order with an f32 scale per (row, 128 columns); the "
  "sqrt(128) the unnormalised transform leaves behind is NOT undone here, because <Hw,Ha> = "
  "128<w,a> and the whole factor is folded into the stored weight scale at convert",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHadQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_had_quant_act_fp8, nullptr, nullptr, nullptr,
  r4d_shape_quant_act_fp8 },

{ "givens_dense_quant", "givens_dense_quant", "quant",
  "ParoQuant's rotation as a dense 128 x 128 f16 matrix a (part, group) on the matrix cores, then "
  "per-(row, 128-column) symmetric quantisation to int8 or E4M3, once for each rotated part",
  "n a multiple of 128 and group 128; the matrices are the weight, composed from the givens tables at "
  "convert time",
  "bf16 in, int8 or E4M3 out in PLAIN ROW-MAJOR order, parts side by side, with an f32 scale (amax/127 "
  "or amax/448) per (row, part, 128 columns); f16 matrices and f32 channel scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGivensDense), NOTUNE,
  nullptr, nullptr, r4d_rad_givens_dense_quant, nullptr, r4d_rad_layout_gdense, r4d_rad_relayout_gdense },

{ "givens_rotate", "givens_rotate", "quant",
  "the channel scale and ParoQuant's Givens rotations, once for each rotated part, to f16",
  "n a multiple of 128 and group 128: a rotation group cannot straddle the end of a row",
  "bf16 in, f16 out, parts side by side; the tables are the weight's int32 pairs, f32 angles and "
  "f32 channel scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHadQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_givens_rotate, nullptr, r4d_rad_layout_givens_pairs,
  r4d_rad_relayout_givens_pairs, nullptr, r4d_rad_unrelayout_givens_pairs },

{ "quant_act_i8g", "quant_act_i8g", "quant",
  "per-(row, 128-column) symmetric int8 quantisation of a bf16 activation",
  "quant_act_fp8's: n a multiple of 4, group 128, a ragged last group written zero-padded",
  "bf16 in, int8 out in PLAIN ROW-MAJOR order with an f32 scale of amax/127 per (row, 128 columns)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_act_i8g },

{ "had_quant_act_i8g", "had_quant_act_i8g", "quant",
  "a 128-wide Hadamard along K, then per-(row, 128-column) symmetric int8 quantisation",
  "had_quant_act_fp8's: n a multiple of 128 and group 128",
  "bf16 in, int8 out in PLAIN ROW-MAJOR order with an f32 scale of amax/127 per (row, 128 "
  "columns); the factor of 128 of the unnormalised transform is left to the weight's stored "
  "scale (dtype i8a8h)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHadQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_had_quant_act_i8g },

/* QSA (docs/QSA.md). The work list: one workgroup per potential block. */
{ "qsa_work_bf16", "qsa_work", "attn",
  "gather the blocks completing this step into the flat list qsa_block_key reads",
  "any declared sequence maximum -- the real count comes off cu_seqlens at issue and the launcher "
  "refuses past 256 there; any token count, any ratio; "
  "`work` is a HOST bound and slots past the real count report page -1",
  "bf16 keys and a bf16 tail in, an i32 position, cu_seqlens and the block-key table; a bf16 "
  "staging plane out with an i32 page and first-token position a block",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaWork), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_work, nullptr, nullptr, nullptr,
  r4d_shape_qsa_work },

/* One workgroup per token; most exit on the second compare. */
{ "qsa_tail_store_bf16", "qsa_tail_store", "attn",
  "park the raw index keys of the tokens that have not completed a block, one slot a position",
  "any token count, any width, any ratio; the sequence count comes off cu_seqlens at run time "
  "and not off the tail's declared height",
  "bf16 keys in, an i32 position and an i32 cu_seqlens; the bf16 tail written in place",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaTailStore), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_tail_store, nullptr, nullptr, nullptr,
  r4d_shape_qsa_tail_store },

/* A four-digit radix select, one workgroup per query. Not a sort: attention is a softmax over a
 * set, and sorting 65536 scores per query per layer per step is too costly. The set is exact and
 * deterministic: ties break to the lower block index through a prefix count, never atomic order. */
{ "qsa_select_i32", "qsa_select", "attn",
  "the topk highest-scoring blocks as a page-id table, with the partial page appended last",
  "any M, any block count, any topk; a decode-shaped launch splits the block list across the "
  "device and merges, and falls back to one workgroup a query when the scratch is absent",
  "f32 scores and an i32 page table in; an i32 [M, topk+1] table, the same entries' block "
  "indices and a live count out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaSelect), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_select, r4d_rad_qsa_select_scratch, nullptr, nullptr,
  r4d_shape_qsa_select },

/* The indexer's score. One workgroup per query, one wave per block: a lane owns n/32 consecutive
 * dims of the query heads and the block key, so a head's dot product is one wave reduction and the
 * block key is read once for all heads (the head loop is inside the block loop). */
{ "qsa_score_bf16", "qsa_score", "attn",
  "the QSA indexer's block score: relu per (block, head), summed over heads, over 1/sqrt(n)",
  "n a multiple of 32 up to 256, at most 8 heads, any block count and any M",
  "bf16 queries and block keys, an i32 page-id table, f32 scores; a negative table entry scores "
  "-inf and not zero",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaScore), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_score, nullptr, nullptr, nullptr,
  r4d_shape_qsa_score },

/* The only query-independent piece of QSA, hence a kernel of its own: a block key is fixed once
 * its tokens exist, so it is built when the last arrives and read by every later query. */
{ "qsa_block_key_bf16", "qsa_block_key", "attn",
  "the QSA indexer's compressed key: mean of `ratio` raw keys, RMS-normed, rotated at the "
  "block's first position",
  "n a multiple of 32 up to 256 (a block reduction over one LDS row), an even rotary width no "
  "wider than the head, any `ratio` and any `M`",
  "bf16 keys in with an f32 gain; bf16 out, one row a BLOCK. The mean is rounded to bf16 before "
  "the norm, which is the reference's order and not the convenient one",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaBlockKey), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_block_key, nullptr, nullptr, nullptr,
  r4d_shape_qsa_block_key },

/* GPTQ's calibration half, the only op here no forward pass issues. It runs under a calibration
 * flag beside the quantiser whose output it characterises, and the 4-bit expert packer reads the
 * accumulated Hessians at convert time, from a file. So it is unresolved by the serving graph on
 * purpose, not an unwired kernel. */
{ "gram_accum", "gram_accum", "quant",
  "h += x^T x over an E4M3 activation, in f32 -- the Gram matrix a GPTQ packer needs",
  "n a multiple of 4 and group 128, because a 64-wide output tile must lie inside one scale "
  "group; any M, and the accumulator is READ as well as written",
  "E4M3 plus an f32 scale per (row, 128 columns) in; an f32 [n][n] accumulator in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGramAccum), NOTUNE,
  nullptr, nullptr, r4d_rad_gram_accum, nullptr, nullptr, nullptr,
  r4d_shape_gram_accum },

{ "gram_accum_bf16", "gram_accum", "quant",
  "h += x^T x over a plain bf16 activation, in f32 -- the Gram matrix of an unquantised GEMM's "
  "input, which a model served at bf16 records",
  "n a multiple of 4; any M, and the accumulator is READ as well as written",
  "bf16 in, no scale operand; an f32 [n][n] accumulator in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGramAccumBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_gram_accum_bf16, nullptr, nullptr, nullptr,
  r4d_shape_gram_accum },

/* The two fp8 fusions, counterparts of the int8 path's. A producer that holds the row (an RMS
 * norm, a gated product) writes the E4M3 pair on the same pass, saving the quantiser's launch and
 * its second read. The codes equal the unfused pair's, since both round to bf16 first. */
{ "rmsnorm_quant_fp8", "rmsnorm_quant_fp8", "quant",
  "RMS norm and the fp8 activation quantiser in one pass over the row, plus the bf16 norm output "
  "the unfused pair wrote",
  "n a multiple of 4; one workgroup per row and two passes over it, the second L1-hot; the gain is "
  "f32 or bf16 and the shim reads which off the operand; group 128; an optional "
  "residual is the fused_add_rms_norm form and may ALIAS out_bf16",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns) and an optional bf16 out; the codes "
  "are taken over the BF16-ROUNDED norm output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRmsNormQuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_rmsnorm_quant_fp8, nullptr, nullptr, nullptr,
  r4d_shape_rmsnorm_quant_fp8, nullptr,
  /* `add` + `rmsnorm` + `quant_act_fp8`, not `rmsnorm_add` (r4d_fuse.cpp). */
  r4d_fuse_rmsnorm_quant_fp8 },

/* The rotated int8 twin, for a trunk of weights stored at i8a8h: the same kernel with
 * had_quant_act_i8g's quantiser on its second pass. No fusion hook: the architecture declares it. */
{ "rmsnorm_had_quant_i8g", "rmsnorm_had_quant_i8g", "quant",
  "RMS norm, a 128-wide Hadamard and the per-(row, 128) int8 quantiser in one pass over the row, "
  "plus the bf16 norm output",
  "n a multiple of 128 -- a rotation block cannot straddle the end of a row; one workgroup per row "
  "and two passes over it, as rmsnorm_quant_fp8; the gain is f32 or bf16; group 128, both the "
  "rotation width and the scale's block width; an optional residual may ALIAS out_bf16",
  "bf16 in, int8 out with an f32 scale per (row, 128 columns) and an optional bf16 out that "
  "receives the UNROTATED norm output; the codes are taken over the rotated BF16-ROUNDED norm "
  "output, byte-identical to rmsnorm + had_quant_act_i8g",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRmsNormHadQuantI8g), NOTUNE,
  nullptr, nullptr, r4d_rad_rmsnorm_had_quant_i8g, nullptr, nullptr, nullptr,
  r4d_shape_rmsnorm_had_quant_i8g },

{ "gated_quant_fp8", "gated_quant_fp8", "quant",
  "an elementwise gated product and the fp8 activation quantiser in one pass; `act` silu is "
  "down_proj's input and sigmoid is a gated o_proj's",
  "n a multiple of 4; the gated operand is [M][2n], gate then up; no reduction across the row, so "
  "no barrier; group 128; strides in elements",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns) and an optional bf16 out; the codes "
  "are taken over the BF16-ROUNDED product",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGatedQuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_gated_quant_fp8, nullptr, nullptr, nullptr,
  r4d_shape_gated_quant_fp8, nullptr,
  /* `silu_mul` + `quant_act_fp8`, silu arm only (r4d_fuse.cpp). */
  r4d_fuse_gated_quant_fp8 },

/* The rotated twin, for a down projection whose weight is stored at w4a8h. No fusion hook:
 * nothing emits `silu_mul` + `had_quant_act_fp8`; the MoE block declares this op directly. */
{ "gated_had_quant_fp8", "gated_had_quant_fp8", "quant",
  "an elementwise gated product, a 128-wide Hadamard along n, and the fp8 activation quantiser, "
  "in one pass",
  "n a multiple of 128 -- a rotation block cannot straddle the end of a row; the gated operand is "
  "[M][2n], gate then up; no reduction across the row, so no barrier; group 128, which is both "
  "the rotation width and the scale's block width; strides in elements",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns) and an optional bf16 out that "
  "receives the UNROTATED product; the codes are taken over the rotated BF16-ROUNDED product, and "
  "the residual factor of 128 is folded into the stored weight scale at convert, not here",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGatedHadQuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_gated_had_quant_fp8, nullptr, nullptr, nullptr,
  r4d_shape_gated_quant_fp8 },

/* Its int8 twin, for a down projection whose weight is stored at i8a8h. */
{ "gated_had_quant_i8g", "gated_had_quant_i8g", "quant",
  "an elementwise gated product, a 128-wide Hadamard along n, and the per-(row, 128) int8 "
  "quantiser, in one pass",
  "n a multiple of 128 -- a rotation block cannot straddle the end of a row; the gated operand is "
  "[M][2n], gate then up; no reduction across the row, so no barrier; group 128, both the "
  "rotation width and the scale's block width; strides in elements",
  "bf16 in, int8 out with an f32 scale per (row, 128 columns) and an optional bf16 out that "
  "receives the UNROTATED product; the codes are taken over the rotated BF16-ROUNDED product, "
  "byte-identical to the product + had_quant_act_i8g",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGatedHadQuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_gated_had_quant_i8g, nullptr, nullptr, nullptr,
  r4d_shape_gated_had_quant_i8g },

{ "gate_quant_fp8", "gate_quant_fp8", "quant",
  "an elementwise gated product and the fp8 activation quantiser in one pass, with the gate in "
  "its own operand: a gated attention's gate is a column slice of the [q|gate] projection and "
  "cannot be packed beside the attention output it scales",
  "n a multiple of 4; gate and x are both [M][n] and may have different pitches; no reduction "
  "across the row, so no barrier; group 128; strides in elements",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns) and an optional bf16 out; the codes "
  "are taken over the BF16-ROUNDED product",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGateQuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_gate_quant_fp8, nullptr, nullptr, nullptr,
  r4d_shape_gate_quant_fp8, nullptr,
  /* `sigmoid` + `mul` + `quant_act_fp8`; the silu arm is declined (r4d_fuse.cpp). */
  r4d_fuse_gate_quant_fp8 },

/* ---------------------------------------------------------------------------------- sampling
 *
 * spec §13 requires sampling on device. This is the only implementation of sample_chain (libref
 * implements docs/OPS.md's host stages), so there is nothing to compare it against. */
{ "sample_chain_f32", "sample_chain", "sample",
  "temperature, softmax, top_k, top_p, min_p, renormalise and the draw in vLLM's order -- plus the "
  "exclusive draw and the accept probability a greedy-draft rejection-sampling verify needs -- in "
  "one workgroup a position",
  "any vocabulary and any M; logits read through explicit row and column strides, so either head "
  "layout is served without a transpose; the top_k and top_p cuts are four radix passes each over "
  "a monotone key, exact without sorting; n_vocab must be the FULL vocabulary, so a vocab-parallel "
  "caller gathers the other ranks' bands first",
  "f32 logits, f32 uniforms (two a position), u32 query tokens or absent, "
  "{u32 tok, u32 tok_excl, f32 p_query} out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSampleChain), NOTUNE,
  nullptr, nullptr, r4d_rad_sample_chain, nullptr, nullptr, nullptr,
  r4d_shape_sample_chain },

{ "quant_act_i8", "quant_act_i8", "quant",
  "per-row symmetric int8 quantisation of a bf16 activation, written in the byte order the r4d "
  "GEMMs' A fragment reads; a third output selects the form that also emits the per-group row sums",
  "K a multiple of 16; one workgroup per row",
  "bf16 in, int8 out with an f32 per-row scale",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQuantActI8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_act_i8, nullptr, nullptr, nullptr,
  r4d_shape_quant_act_i8 },

{ "rmsnorm_quant_i8", "rmsnorm_quant_i8", "quant",
  "RMS norm and quant_act_i8's quantiser in one pass over the row; optionally the residual "
  "junction in front of it -- a post-norm on x, the add into the residual and a scalar on the sum",
  "n a multiple of 16 and n*2 bytes within LDS; one workgroup per row at rmsnorm_bf16's width; "
  "the gains f32 or bf16, read off the operand; w_post and scalar need the residual",
  "bf16 in, int8 out in the w4a8 GEMM's fragment order with an f32 per-row scale, optional bf16 "
  "out; every stored intermediate rounded to bf16 as the unfused chain stores it",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRmsNormQuantI8), ROW(tRmsQ8), nullptr,
  nullptr, nullptr, r4d_rad_rmsnorm_quant_i8 },

{ "qkv_norm_rope_bf16", "qkv_norm_rope", "attn",
  "per-head RMS norms of separate q and k projections, the weightless norm of v (or of the raw k), "
  "and the rotary of q and k, in place, in one dispatch",
  "head_dim 256 or 512; neox, `rotary` or `head` ladder, one position component, no YaRN; one "
  "workgroup a (token, head), a thread a channel",
  "bf16 q/k/v and gains (or f32 gains), i32 positions; byte-identical to rmsnorm_p + rmsnorm x2 + "
  "rope x2, each norm taken by its standalone kernel's partition",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkvNormRope), NOTUNE,
  nullptr, nullptr, r4d_rad_qkv_norm_rope, nullptr, nullptr, nullptr, r4d_shape_qkv_norm_rope },

{ "qkv_norm_rope_h128_bf16", "qkv_norm_rope", "attn",
  "per-head RMS norms of separate q and k projections, the weightless norm of v unless v_norm is 0, "
  "and the rotary of q and k, in place, in one dispatch",
  "head_dim 128; neox, `rotary` or `head` ladder, one position component, no YaRN; one workgroup a "
  "(token, head), a thread a channel",
  "bf16 q/k/v and gains (or f32 gains), i32 positions; byte-identical to [rmsnorm_p +] rmsnorm x2 + "
  "rope x2, each norm taken by its standalone kernel's partition",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkvNormRopeH128), NOTUNE,
  nullptr, nullptr, r4d_rad_qkv_norm_rope },

/* The prologue and the cache store in one launch: a k (or v) head's block ends holding its token's
 * finished head, so it writes it into the cache too, converted as kv_store converts it. Saves the
 * store's launch a layer, which at decode is most of its cost. */
{ "qkv_norm_rope_kv_store_fp8", "qkv_norm_rope_kv_store", "attn",
  "qkv_norm_rope, and kv_store into an E4M3 paged cache from the same launch",
  "qkv_norm_rope's shapes; cache exactly [n_blocks, n_head_kv, block_size, 2*head_dim], one slot "
  "a row; a negative or out-of-range slot stores nothing",
  "bf16 q/k/v and gains (or f32 gains), i32 positions and slots, fp8-e4m3 cache, scalar "
  "k_scale / v_scale descales; byte-identical to qkv_norm_rope + kv_store_fp8",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkvNormRopeKv), NOTUNE,
  nullptr, nullptr, r4d_rad_qkv_norm_rope_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qkv_norm_rope_kv_store, nullptr, r4d_fuse_qkv_norm_rope_kv_store },

/* The same into a bf16 cache: the store is the values kv_store_bf16 copies. */
{ "qkv_norm_rope_kv_store_bf16", "qkv_norm_rope_kv_store", "attn",
  "qkv_norm_rope, and kv_store into a bf16 paged cache from the same launch",
  "qkv_norm_rope's shapes; cache exactly [n_blocks, n_head_kv, block_size, 2*head_dim], one slot "
  "a row; a negative or out-of-range slot stores nothing",
  "bf16 q/k/v, gains, cache; i32 positions and slots; byte-identical to qkv_norm_rope + "
  "kv_store_bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkvNormRopeKvBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_qkv_norm_rope_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qkv_norm_rope_kv_store, nullptr, r4d_fuse_qkv_norm_rope_kv_store },

{ "add_rmsnorm_bf16", "add_rmsnorm", "norm",
  "x += a, then the root-mean-square norm of the x written, one workgroup a row as rmsnorm_bf16",
  "any n; rmsnorm_bf16's width and column partition, so the sum of squares folds as it does",
  "bf16 x / a / y, f32 or bf16 gain, fp32 reduction; byte-identical to add_bf16 + rmsnorm_bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAddRmsNorm), NOTUNE,
  nullptr, nullptr, r4d_rad_add_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_add_rmsnorm, nullptr, r4d_fuse_add_rmsnorm },

{ "gelu_mul_quant_i8", "gelu_mul_quant_i8", "quant",
  "gelu(gate) * up, the tanh form, and quant_act_i8's quantiser in one pass over the row",
  "n a multiple of 16 and n*2 bytes within LDS; one workgroup per row",
  "bf16 gate_up in, int8 out in the w4a8 GEMM's fragment order with an f32 per-row scale, optional "
  "bf16 out; the codes taken over the bf16-rounded product",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGeluMulQuantI8), NOTUNE,
  nullptr, nullptr, r4d_rad_gelu_mul_quant_i8 },

{ "ar_rmsnorm_quant_i8", "ar_rmsnorm_quant_i8", "quant",
  "the two-rank exact all-reduce and the junction after it -- post-norm of the reduced sum, residual "
  "add, scalar, RMS norm and quant_act_i8's quantiser -- in one kernel",
  "exactly 2 ranks; n a multiple of 16, x unpitched (it is the message); one block a row; exact wire",
  "bf16 in, int8 out in the w4a8 GEMM's fragment order with an f32 per-row scale; the reduced sum "
  "rounded to bf16 as the standalone exact all-reduce stores it",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArRmsNormQuantI8), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_rmsnorm_quant_i8 },

/* Its arithmetic is the four-rank exact one-shot's fold followed by the two-rank form's junction,
 * each the code those two run. r4d_selftest's case of the same name holds it byte for byte to
 * all_reduce + rmsnorm_quant_i8 on a four-card box, and the engine to a decode step's text against
 * a build that issues the pair. Both have run on cards that way (the 31B's fused junctions, and
 * r4d_selftest on four R9700s). */
{ "ar_rmsnorm_quant_i8_4rank", "ar_rmsnorm_quant_i8", "quant",
  "the four-rank exact one-shot all-reduce and the junction after it -- post-norm of the reduced "
  "sum, residual add, scalar, RMS norm and quant_act_i8's quantiser -- in one kernel",
  "exactly 4 ranks; n a multiple of 16, x unpitched (it is the message); a block owns whole rows; "
  "exact wire; one-shot ((ranks - 1) x the message a rank) -- for decode heights; from 5 to 32 "
  "rows (axis `tail`) the standalone two-shot with the junction as its tail",
  "bf16 in, int8 out in the w4a8 GEMM's fragment order with an f32 per-row scale; the reduced sum "
  "rounded to bf16 as the standalone exact all-reduce stores it",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArRmsNormQuantI84), ROW(tArQ84), nullptr,
  r4d_rad_ar_hc_wide_init, r4d_rad_ar_fini, r4d_rad_ar_rmsnorm_quant_i8_4rank, nullptr, nullptr,
  nullptr, nullptr },

{ "had_quant_act_i8", "had_quant_act_i8", "quant",
  "block Hadamard along K then per-row symmetric int8 quantisation; the rotation is what makes "
  "int8 activations usable at all: unrotated, the per-row absmax is set by a few outlier channels "
  "and the rest of the row quantises to nearly nothing",
  "K a multiple of 128; rotation width a power of two dividing the PER-RANK K, so 512 is the "
  "largest that down_proj's 8704 admits; K*4 bytes must fit LDS; one workgroup per row",
  "bf16 in, int8 out with an f32 per-row scale that also carries the 1/sqrt(had) normalisation",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHadQuantActI8), NOTUNE,
  nullptr, nullptr, r4d_rad_had_quant_act_i8, nullptr, nullptr, nullptr,
  r4d_shape_had_quant_act_i8 },

{ "rmsnorm_had_quant_i8", "rmsnorm_had_quant_i8", "quant",
  "residual add, RMS norm, block Hadamard and per-row int8 quantisation in one pass over the row; "
  "optionally also writes the UNROTATED bf16 norm output for consumers that stay bf16",
  "K a multiple of 128; rotation width a power of two dividing the per-rank K; K*4 bytes must fit "
  "LDS; one workgroup per row; residual and out_bf16 may be null",
  "bf16 in, int8 out with an f32 per-row scale carrying 1/sqrt(had), optional bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRmsNormHadQuantI8), NOTUNE,
  nullptr, nullptr, r4d_rad_rmsnorm_had_quant_i8, nullptr, nullptr, nullptr,
  r4d_shape_rmsnorm_had_quant_i8 },

/* Unvalidated: the DPP row_xmask form of the fused all-reduce (the r4d_xor_lane exchanges in
 * r4d_ar_ln_had_quant_i8.hip) is compile-verified only, not measured or correctness-gated. */
{ "ar_ln_had_quant_i8", "ar_ln_had_quant_i8", "quant",
  "the two-rank all-reduce and the residual add, RMS norm, block Hadamard and per-row int8 "
  "quantisation that always follow it, in one kernel; the all-reduce's bf16 output never exists "
  "[UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "exactly 2 ranks; K a multiple of 128 and K*2 16B-aligned; rotation width a power of two "
  "dividing the per-rank K; K*4 bytes must fit LDS; a block owns whole ROWS, so the block count "
  "is clamped to M; residual and out_bf16 may be null",
  "bf16 in, int8 out with an f32 per-row scale carrying 1/sqrt(had), optional bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArLnHadQuantI8), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_ln_had_quant_i8, nullptr, nullptr, nullptr,
  r4d_shape_ar_ln_had_quant_i8 },


{ "ar_rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8", "quant",
  "the two-rank all-reduce and the residual add, RMS norm and block-scaled fp8 quantisation that "
  "always follow it, in one kernel; the all-reduce's output is never read back",
  "exactly 2 ranks; n a multiple of 8 and x unpitched, because x is the message and a block pushes "
  "its rows as 16B words; group 128; a block owns whole ROWS, so the block count is clamped to M; "
  "either wire, chosen by `wire` -- the rotated 6-bit payload additionally needs n to be a "
  "multiple of 512, so that a chunk never straddles a row; residual and out_bf16 may be null",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns) and an optional bf16 out; codes "
  "taken over the BF16-ROUNDED norm output, and the reduced sum rounded to bf16 exactly as the "
  "standalone all-reduce of the SAME wire stores it -- byte-identical to the pair it replaces",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArRmsNormQuantFp8), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_rmsnorm_quant_fp8, nullptr, nullptr, nullptr,
  r4d_shape_ar_rmsnorm_quant_fp8, nullptr,
  /* `all_reduce` + `rmsnorm_quant_fp8`, both wires (r4d_fuse.cpp). */
  r4d_fuse_ar_rmsnorm_quant_fp8 },

{ "ar_rmsnorm_had_quant_i8g", "ar_rmsnorm_had_quant_i8g", "quant",
  "the two-rank all-reduce and the residual add, RMS norm, 128-wide Hadamard and per-(row, 128) "
  "int8 quantisation that always follow it, in one kernel; the all-reduce's output is never read "
  "back",
  "exactly 2 ranks; n a multiple of 128 (a rotation block cannot straddle a row) and x unpitched; "
  "group 128; a block owns whole ROWS, so the block count is clamped to M; either wire, chosen by "
  "`wire` -- the rotated 6-bit payload additionally needs n to be a multiple of 512; residual and "
  "out_bf16 may be null",
  "bf16 in, int8 out with an f32 scale per (row, 128 columns) and an optional bf16 out of the "
  "UNROTATED norm output; x and the residual are ar_rmsnorm_quant_fp8's bytes, and the codes "
  "rmsnorm_had_quant_i8g's over the same norm output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArRmsNormHadQuantI8g), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_rmsnorm_had_quant_i8g, nullptr, nullptr, nullptr,
  r4d_shape_ar_rmsnorm_had_quant_i8g },

{ "ar_rmsnorm_post_quant_fp8", "ar_rmsnorm_post_quant_fp8", "quant",
  "the two-rank all-reduce, a post-norm of the reduced sum, the residual add, RMS norm and "
  "block-scaled fp8 quantisation in one kernel: a sandwich-norm junction (Gemma 3)",
  "exactly 2 ranks, the exact wire; n a multiple of 8 and x unpitched; group 128; a block owns whole "
  "rows; the residual is required",
  "bf16 in, E4M3 out with an f32 scale per (row, 128 columns); x receives the reduced sum",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArRmsNormQuantFp8), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_rmsnorm_quant_fp8, nullptr, nullptr, nullptr,
  nullptr },

{ "ar_hc_write", "ar_hc_write", "norm",
  "the two-rank all-reduce and the gated residual's write that always follows it, in one kernel: "
  "h[s] += inj[s] * y over the reduced y",
  "exactly 2 ranks; n a multiple of 8 and y unpitched, because y is the message and a block pushes "
  "it as 16B words; the stream pitch a multiple of 8; hc 2..8; either wire, chosen by `wire` -- "
  "the rotated 6-bit payload additionally needs M*n to be a multiple of 64",
  "bf16 throughout; the sum rounded to bf16 exactly as the standalone all-reduce of the SAME wire "
  "stores it and the write in hc_write's arithmetic over that bf16 value -- byte-identical to the "
  "pair it replaces",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArHcWrite), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_hc_write, nullptr, nullptr, nullptr,
  r4d_shape_ar_hc_write },

{ "ar_hc_write_4rank", "ar_hc_write", "norm",
  "the four-rank exact all-reduce and the gated residual's write that follows it, in one kernel: "
  "h[s] += inj[s] * y over the reduced y",
  "exactly 4 ranks, all peer-accessible; exact wire only; n a multiple of 8 and y unpitched, "
  "because y is the message and a block pushes it as 16B words; the stream pitch a multiple of 8; "
  "hc 2..8; the wide one-shot's handshake, slots and block partition, on an instance keyed by "
  "its own M bound -- the wide family's flag/seq/slot discipline is what lets it share a stream, "
  "or an instance where the keys agree, with the plain all-reduce; cudagraph-safe",
  "bf16 throughout; the sum in fp32 in ascending rank order and rounded to bf16 exactly as the "
  "four-rank exact all-reduce stores it, and the write in hc_write's arithmetic over that bf16 "
  "value -- byte-identical to the pair it replaces",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArHcWrite4), ROW(tAr), nullptr,
  r4d_rad_ar_hc_wide_init, r4d_rad_ar_fini, r4d_rad_ar_hc_write_4rank, nullptr, nullptr, nullptr,
  r4d_shape_ar_hc_write },

{ "ar_gather_hc_write", "ar_gather_hc_write", "norm",
  "moe_gather folded in front of ar_hc_write: each block sums its words' tokens back from the "
  "routed experts' rows, keeps the result as its message, and then reduces and writes exactly as "
  "ar_hc_write does",
  "exactly 2 ranks; n a multiple of 8, y unpitched and y_expert's pitch a multiple of 8, because "
  "both move as 16B words; top_k <= 64; the stream pitch a multiple of 8; hc 2..8; either wire, "
  "chosen by `wire` -- the rotated 6-bit payload additionally needs M*n to be a multiple of 64 and "
  "builds its slot table in LDS, widening the grid until a block's table fits",
  "bf16 throughout; the gather's sums and narrows are moe_gather's (round-to-nearest-even, the "
  "shared arm's product rounded before the add), the reduction and the write ar_hc_write's on the "
  "same wire -- byte-identical to the three kernels it replaces",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArGatherHcWrite), ROW(tAr), nullptr,
  r4d_rad_ar_ln_init, r4d_rad_ar_fini, r4d_rad_ar_gather_hc_write,
  r4d_rad_ar_gather_hc_write_scratch, nullptr, nullptr,
  r4d_shape_ar_gather_hc_write, nullptr, r4d_fuse_ar_gather_hc_write },

{ "ar_gather_hc_write_4rank", "ar_gather_hc_write", "norm",
  "moe_gather folded in front of the four-rank ar_hc_write: each block sums its words' tokens "
  "back from the routed experts' rows, keeps the result as its message and pushes it, and then "
  "reduces and writes exactly as ar_hc_write_4rank does",
  "exactly 4 ranks, all peer-accessible; exact wire only; n a multiple of 8, y unpitched and "
  "y_expert's pitch a multiple of 8, because both move as 16B words; top_k <= 64; the stream "
  "pitch a multiple of 8; hc 2..8; the wide one-shot's handshake and slots, on an instance keyed "
  "by its own M bound as ar_hc_write_4rank's is; the grid widened until a block's slot table fits "
  "in LDS, or the inverse table in the scratch where that would take more blocks than the "
  "reduction asks for; cudagraph-safe",
  "bf16 throughout; the gather's sums and narrows are moe_gather's (round-to-nearest-even, the "
  "shared arm's product rounded before the add), the sum in fp32 in ascending rank order and "
  "rounded to bf16 exactly as the four-rank exact all-reduce stores it, and the write in "
  "hc_write's arithmetic over that bf16 value -- byte-identical to the three kernels it replaces",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cArGatherHcWrite4), ROW(tAr), nullptr,
  r4d_rad_ar_hc_wide_init, r4d_rad_ar_fini, r4d_rad_ar_gather_hc_write_4rank,
  r4d_rad_ar_gather_hc_write_scratch, nullptr, nullptr,
  r4d_shape_ar_gather_hc_write, nullptr, r4d_fuse_ar_gather_hc_write },

{ "qk_norm_rope_gate", "qk_norm_rope_gate", "attn",
  "the gated-attention QKV preamble in one pass: split [q|gate] per head, QK RMS norm over the "
  "head, partial NeoX RoPE and the gate copy",
  "head_dim 64, 128 or 256; rotary_dim even and <= head_dim, with rotary_dim/2 a power-of-two "
  "multiple of head_dim/32 so the RoPE pair is one shuffle; NeoX style; bf16; n_head may be 0, "
  "which is the K-only form and serves an input with no gate interleaved",
  "bf16 in and out; the norm output is rounded to bf16 before the rotation, as the unfused "
  "reference does by storing it",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGate), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate },

/* The prologue and the fp8 cache store in one launch: a k head's block already holds its token's
 * rotated k one element per lane, so the store is two byte writes per lane. */
{ "qk_norm_rope_gate_kv_store_fp8", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into an E4M3 paged cache from the same launch",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] and "
  "one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, fp8-e4m3 cache, i32 slot mapping, scalar k_scale / v_scale descales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGateKv), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },

/* The fold into an fp8_v4 cache: E4M3 K, int4 V in groups of 32 (a wave) with f16 scales. */
{ "qk_norm_rope_gate_kv_store_fp8v4", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into an fp8_v4 paged cache from the same launch",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 25*head_dim/16], "
  "head_dim a multiple of 32, one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, E4M3 K and int4 V with f16 scales, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGateKvFp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store_fp8v4, nullptr, nullptr, nullptr,
  nullptr },

/* The fold into a k6_v4 cache: int6 K and int4 V in groups of 32 (a wave) with f16 scales. */
{ "qk_norm_rope_gate_kv_store_k6v4", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into a k6_v4 paged cache from the same launch",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 22*head_dim/16], "
  "head_dim a multiple of 32, one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, int6 K and int4 V with f16 scales, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGateKvK6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store_k6v4, nullptr, nullptr, nullptr,
  nullptr },

/* The same fold into a bf16 cache: the store is the two values the pair's kv_store_bf16 copies. */
{ "qk_norm_rope_gate_kv_store_bf16", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into a bf16 paged cache from the same launch",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] and "
  "one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, bf16 cache, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGateKvBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },

{ "rope_table_f32", "rope_table", "attn",
  "the cos/sin plane the fused prologue reads, at each position's own row",
  "rot even and >= 2; the table is [max_ctx, rot] and only the rows `positions` names are written, "
  "so a row nothing addresses keeps whatever it held; a negative position writes nothing",
  "f32 or bf16 table, i32 or i64 positions; the angles are rope_bf16's exactly",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRopeTable), NOTUNE,
  nullptr, nullptr, r4d_rad_rope_table, nullptr, nullptr, nullptr,
  r4d_shape_rope_table },

{ "gated_had_quant_i8", "gated_had_quant_i8", "quant",
  "an elementwise gated product, block Hadamard and per-row int8 quantisation in one pass; mode 0 "
  "is silu(a)*b for down_proj's input and mode 1 is sigmoid(a)*b for o_proj's",
  "K a multiple of 128; rotation width a power of two dividing the per-rank K; K*4 bytes must fit "
  "LDS; strides in elements, 0 means tight",
  "bf16 in, int8 out with an f32 per-row scale carrying 1/sqrt(had)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGatedHadQuantI8), NOTUNE,
  nullptr, nullptr, r4d_rad_gated_had_quant_i8, nullptr, nullptr, nullptr,
  r4d_shape_gated_had_quant_i8 },

{ "gdn_gated_norm_had_quant_i8", "gdn_gated_norm_had_quant_i8", "quant",
  "the GDN gated RMS norm over 128-wide heads, then a block Hadamard and per-row int8 "
  "quantisation over the flattened row -- out_proj's input on a linear-attention layer",
  "head 128; K a multiple of 128 and of the rotation width; the RMS is per HEAD while the rotation "
  "and the int8 scale are per TOKEN, so the reduction is segmented over 128/VPL lanes",
  "bf16 x and z, f32 norm weight, int8 out with an f32 per-row scale carrying 1/sqrt(had)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnGatedNormHadQuantI8), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_gated_norm_hq_i8, nullptr, nullptr, nullptr,
  r4d_shape_gdn_gated_norm_had_quant_i8 },

{ "dflash_conv_t2_g16_bf16", "dflash_conv", "dflash",
  "DFlash2 grouped dynamic depthwise conv: coefficient add, both taps and the block-position mask "
  "in one pass",
  "taps 2, group_size 16, hidden_size divisible by group_size*8, block_size a power of two; delta "
  "is a slice of the [T, 2, taps, NG] projection so its row pitch is 2*taps*NG; out must not alias x",
  "bf16 x / delta / base / out, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDflashConv), NOTUNE,
  nullptr, nullptr, r4d_rad_dflash_conv, nullptr, nullptr, nullptr,
  r4d_shape_dflash_conv },

{ "dflash_select_bf16", "dflash_select", "dflash",
  "DFlash2 candidate selector: the low-rank edge scores and the greedy walk over them, one "
  "workgroup a sequence and no launch between steps",
  "top_k <= 32 and rank a multiple of 32 and <= 512, both being LDS extents; cand/unary/hp are "
  "read at row s*steps+l, so the three planes share one row numbering",
  "i32 candidate ids, f32 unary logits, bf16 hp and both codebooks, i32 tokens out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDflashSelect), NOTUNE,
  nullptr, nullptr, r4d_rad_dflash_select, nullptr, nullptr, nullptr,
  r4d_shape_dflash_select },

/* pair_argmax, the sandwich-norm joins and expert_scale. The joins and expert_scale have no
 * opd_shape: libref has no row for them, and tests/r4d_emu holds their source to the replaced chain
 * bit for bit on the host. pair_argmax is described as libref describes it. */
{ "pair_argmax_f32", "pair_argmax", "select",
  "the top-1 of a row of (id, value) pairs, ties to the lower id, negative ids skipped",
  "any M and n; one wave a row; the outputs one element a row at their own pitch",
  "f32 pairs (int32 id bits, f32 value), i32 idx, f32 val",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPairArgmax), NOTUNE,
  nullptr, nullptr, r4d_rad_pair_argmax_f32, nullptr, nullptr, nullptr,
  r4d_shape_pair_argmax },

{ "norm_add_norm_bf16", "norm_add_norm", "norm",
  "post-norm, residual add, next pre-norm: one block a row, rounding where the chain rounds",
  "n <= 6144, any M, row pitches honoured",
  "bf16 activations, f32 gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cNormJoin), NOTUNE,
  nullptr, nullptr, r4d_rad_norm_add_norm_bf16 },

{ "norm_add_norm_route_bf16", "norm_add_norm_route", "norm",
  "norm_add_norm, and the routed arm's router input and fp8 expert input from the same norm",
  "n <= 6144 in whole 128-column groups, any M, row pitches honoured",
  "bf16 activations, f32 gains; E4M3 codes with an f32 scale a 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cNormJoinRoute), NOTUNE,
  nullptr, nullptr, r4d_rad_norm_add_norm_route_bf16 },

{ "join_add_scale_bf16", "join_add_scale", "norm",
  "two arms' post-norms, sum, norm, residual add, layer scalar: one block a row",
  "n <= 6144, any M, row pitches honoured",
  "bf16 activations, f32 gains, bf16 scalar",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cNormJoin), NOTUNE,
  nullptr, nullptr, r4d_rad_join_add_scale_bf16 },

{ "expert_scale_bf16", "expert_scale", "moe",
  "routing weight times its expert's scale, in place: one thread an element",
  "any M, top_k <= 64, row pitches honoured",
  "i32 ids, bf16 table and weights",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cExpertScale), NOTUNE,
  nullptr, nullptr, r4d_rad_expert_scale_bf16 },

/* Runtime LoRA, the scalar f32 rows (r4d_lora_bf16.hip). No opd_shape: libref describes both ops,
 * so rad-kbench draws the same operands for this row as for the oracle (`slot` in [0, S], S a base
 * row). Not concurrent (rad_kernel_concurrent below). Below the tiled rows, which serve every dtype
 * but f16/f32 activations (R4D_PRIO_LORA_TILED). */
{ "lora_shrink_scalar", "lora_shrink", "gemm",
  "t = x A[slot]^T per row, base rows untouched: one block a row, one wave a column, the k of a "
  "lane in order and a fixed xor butterfly across the wave",
  "any M and K, 1 <= R <= 65536, 1 <= S <= 2^20; row and column strides of a, a_scale and t "
  "honoured",
  "bf16 / f16 / f32 a, or fp8_e4m3 / fp8_e5m2 / i8 a with its f32 scale stream; bf16 lora_a; "
  "i32 slot; f32 t",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LORA_SCALAR,
  ROW(cLoraShrink), NOTUNE,
  nullptr, nullptr, r4d_rad_lora_shrink },
{ "lora_expand_scalar", "lora_expand", "gemm",
  "y += scale * (t B[slot]^T) per row, base rows neither read nor written: one thread an "
  "output, j in order",
  "any M and N, 1 <= R <= 65536, 1 <= S <= 2^20; row and column strides of t and y honoured",
  "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LORA_SCALAR,
  ROW(cLoraExpand), NOTUNE,
  nullptr, nullptr, r4d_rad_lora_expand },
/* Runtime LoRA, the tiled rows. The grid is row tiles, a tile of base rows leaves on one ballot,
 * and a mixed tile runs one pass per distinct slot; shrink's summation order is fixed by K and R
 * (sh_plan), and expand computes the scalar row's bytes. The shrink split keeps its partials in the
 * op's scratch (the hook sizes it for the band) and its arrival counters in a block keyed on `t`. */
{ "lora_shrink_wmma", "lora_shrink", "gemm",
  "t = x A[slot]^T per row, base rows untouched: WMMA bf16 over 16-row x 16-j tiles, K in 32-wide "
  "steps over 4 waves and a K split fixed by K and R, one pass a distinct slot of a tile",
  "any M and K, 1 <= R <= 65536, 1 <= S <= 2^20; row and column strides of a and t honoured, "
  "16-byte loads where the rows are aligned",
  "bf16 a; bf16 lora_a; i32 slot; f32 t",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LORA_TILED,
  ROW(cLoraShrinkWmma), NOTUNE,
  nullptr, nullptr, r4d_rad_lora_shrink_wmma, r4d_rad_lora_shrink_wmma_scratch },
{ "lora_shrink_wmma_q", "lora_shrink", "gemm",
  "lora_shrink_wmma over fp8 / i8 codes: each code is exact in bf16, and each 32-wide step's "
  "partial is scaled by its group's f32 scale",
  "any M and K, a scale group that is a multiple of 32 (or one group across K), 1 <= R <= 65536, "
  "1 <= S <= 2^20; strides of a, a_scale and t honoured",
  "fp8_e4m3 / fp8_e5m2 / i8 a with its f32 scale stream; bf16 lora_a; i32 slot; f32 t",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LORA_TILED,
  ROW(cLoraShrinkWmmaQ), NOTUNE,
  nullptr, nullptr, r4d_rad_lora_shrink_wmma, r4d_rad_lora_shrink_wmma_scratch },
{ "lora_expand_tiled", "lora_expand", "gemm",
  "y += scale * (t B[slot]^T) per row, base rows neither read nor written: 32-row x 128-column "
  "tiles, B[slot]'s slab staged in LDS once a tile, the scalar row's j-ascending order an output",
  "any M and N, 1 <= R <= 65536, 1 <= S <= 2^20; row and column strides of t and y honoured",
  "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LORA_TILED,
  ROW(cLoraExpand), NOTUNE,
  nullptr, nullptr, r4d_rad_lora_expand_tiled },
/* The generators' expand (lora_expand_mm): the same delta on the WMMA, for passes of many rows of
 * one adapter, where the exact row's scalar sum is most of the adapter's cost. */
{ "lora_expand_mm_wmma", "lora_expand_mm", "gemm",
  "y += scale * (bf16(t) B[slot]^T) per row, base rows neither read nor written: WMMA bf16 over "
  "64-row x 64-column tiles, four waves of 2 x 2 16 x 16 tiles, one pass a distinct slot of a tile",
  "any M and N, 1 <= R <= 65536, 1 <= S <= 2^20; row and column strides of t and y honoured, "
  "16-byte pool loads where B's rows are aligned",
  "f32 t and scale; bf16 lora_b; i32 slot; bf16 / f16 / f32 y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_LORA_TILED,
  ROW(cLoraExpand), NOTUNE,
  nullptr, nullptr, r4d_rad_lora_expand_mm },
{ "attn_prefix_self_h128_bf16", "attn_prefix_self", "attn",
  "attention over a paged prefix and the pass's own rows, causal over the own rows or not",
  "head_dim 128, 1/2/4/8 query heads a KV head, a flat grid over the sequences' query blocks; "
  "the int8 Q.K form (k_i8 / k_scale) at 1 only; the fp8 P.V form (v_scale) at 1 without a "
  "prefix",
  "bf16 q/k/v/out and cache, bf16 operands, fp32 accumulate; or i8 Q.K codes, i32 accumulate; "
  "or E4M3 V codes and P, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnPrefixSelf), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefix_self },

{ "attn_dense_h128_bf16", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens, causal or bidirectional",
  "head_dim 128, 1/2/4/8 query heads a KV head",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH128), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense_h128 },

{ "attn_cross_h128_bf16", "attn_cross", "attn",
  "dense varlen attention with the keys bounded apart from the queries (cross attention)",
  "head_dim 128, 1/2/4/8 query heads a KV head; attn_dense_h128's kernel",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH128), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_cross },

{ "attn_cross_h64_bf16", "attn_cross", "attn",
  "dense varlen attention with the keys bounded apart from the queries (cross attention)",
  "head_dim 64, 1/2/4/8 query heads a KV head; attn_dense_h128's kernel at 64",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnCrossH64), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_cross },

{ "attn_na3d_h64", "attn_na3d", "attn",
  "3-D neighborhood attention: a thread a (query, head), the window's keys streamed through the "
  "caches, libref's two passes",
  "head_dim 64, any window, row pitches honoured", "bf16 or f32 q / k / v / out, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnNa3d), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_na3d },

{ "attn_dense_h384_bf16", "attn_dense", "attn",
  "one 384-wide head over every row of a segment: an image encoder's mid-block attention",
  "head_dim 384 split across eight waves, 16 queries a workgroup",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH384), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense_wide },

{ "attn_dense_h512_bf16", "attn_dense", "attn",
  "one 512-wide head over every row of a segment: an image encoder's mid-block attention",
  "head_dim 512 split across eight waves, 16 queries a workgroup",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH512), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense_wide },

{ "attn_dense_h768_bf16", "attn_dense", "attn",
  "one 768-wide head over every row of a segment: an image encoder's mid-block attention",
  "head_dim 768 split across eight waves, 16 queries a workgroup",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH768), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense_wide },

{ "attn_dense_h1152_bf16", "attn_dense", "attn",
  "one 1152-wide head over every row of a segment: an image decoder's mid-block attention",
  "head_dim 1152 split across eight waves, 16 queries a workgroup",
  "bf16 query / key / value / out, bf16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseH1152), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense_wide },

{ "ln_mod_quant_fp8_bf16", "ln_mod_quant_fp8", "norm",
  "gate_add, ln_mod and quant_act_fp8 in one pass: a workgroup a row",
  "any M, n a multiple of 128 up to 8192, group 128",
  "bf16 in, E4M3 codes with f32 scales out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLnModQ), NOTUNE,
  nullptr, nullptr, r4d_rad_ln_mod_quant_fp8 },
{ "ln_mod_bf16", "ln_mod", "norm",
  "gainless layer norm times (1 + a sequence's scale row): a workgroup a row",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_ln_mod },

{ "table_add_bf16", "table_add", "elem",
  "a shared modulation row plus each block's learned table row, every block in one launch",
  "any M, row pitches honoured; n a multiple of 8",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_table_add },

{ "chan_stats_bf16", "chan_stats", "elem",
  "a VAE latent through its per-channel statistics (running mean and variance, or mean and std)",
  "any M, row pitches honoured; n a multiple of 8",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_chan_stats },

{ "chan_stats_f32", "chan_stats", "elem",
  "a latent kept in f32 through its VAE's per-channel statistics",
  "any M, row pitches honoured",
  "f32 x and y, bf16 or f32 statistics, every step rounded to f32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_chan_stats },

{ "mix_rows_bf16", "mix_rows", "elem",
  "a learned weighted sum of k consecutive rows (a Linear(k, 1) over a stacked axis)",
  "any M, row pitches honoured; n a multiple of 8",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_mix_rows },

{ "gate_add_bf16", "gate_add", "elem",
  "x += tanh(gate row) * y, the reference's three roundings; with `act` none the gate as it is",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_gate_add },

{ "timestep_embed_bf16", "timestep_embed", "elem",
  "sinusoidal timestep embedding, f32 angles",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenTimestep), NOTUNE,
  nullptr, nullptr, r4d_rad_timestep_embed },

{ "timestep_embed_f32", "timestep_embed", "elem",
  "sinusoidal timestep embedding into f32 rows (a timestep MLP kept in f32)",
  "any M, row pitches honoured",
  "f32 t and out, f32 angles",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenTimestepF32), NOTUNE,
  nullptr, nullptr, r4d_rad_timestep_embed },

{ "gather_rows2_bf16", "gather_rows2", "elem",
  "rows from two sources by a signed index",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_gather_rows2 },

{ "lane_gather_w32", "lane_gather", "elem",
  "each entry's rows out of its lane",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenLanes), NOTUNE,
  nullptr, nullptr, r4d_rad_lane_gather },

{ "lane_store_w32", "lane_store", "elem",
  "one source's rows into several lanes",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenLanes), NOTUNE,
  nullptr, nullptr, r4d_rad_lane_store },

{ "randn_lanes_bf16", "randn_lanes", "elem",
  "a lane's rows from its seed's Philox normal stream",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_randn_lanes },

{ "randn_lanes_f32", "randn_lanes", "elem",
  "a lane's rows from its seed's Philox normal stream, kept in f32 (a sampler's f32 latents)",
  "any M, row pitches honoured",
  "f32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_randn_lanes },

{ "flow_euler_bf16", "flow_euler", "elem",
  "a flow-matching Euler step with guidance, preview and repaint",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_flow_euler },

{ "kd_input_bf16", "kd_input", "elem",
  "a k-diffusion sampler's model input: a first step's noise into the f32 lane, c_in x out",
  "any M, row pitches honoured; a thread an element of an entry",
  "f32 lanes, bf16 y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_kd_input },

{ "kd_step_bf16", "kd_step", "elem",
  "a k-diffusion sampler step with guidance and preview: Euler, Euler ancestral or DPM++ 2M",
  "any M, row pitches honoured; a thread an element of an entry",
  "bf16 model output, f32 lanes, history and preview, f32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_kd_step },

{ "chan_affine_bf16", "chan_affine", "elem",
  "a per-channel affine from parameter lists; `round` 0 keeps the lists and arithmetic f32",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAffine), NOTUNE,
  nullptr, nullptr, r4d_rad_chan_affine },

{ "chan_affine_f32", "chan_affine", "elem",
  "a per-channel affine from parameter lists over f32 rows (an f32 latent denormalised)",
  "any M, row pitches honoured",
  "f32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAffineF32), NOTUNE,
  nullptr, nullptr, r4d_rad_chan_affine },

{ "l2norm_chan_bf16", "l2norm_chan", "norm",
  "unit-L2 channel norm, scaled, optional silu: a wave a pixel",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_l2norm_chan },

{ "d2s_add_bf16", "d2s_add", "elem",
  "an upsampling shortcut added in",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_d2s_add },

{ "s2d_avg_add_bf16", "s2d_avg_add", "elem",
  "a downsampling shortcut added in",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_s2d_avg_add },

{ "to_rgba8_bf16", "to_rgba8", "elem",
  "decoded pixels to bytes",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_to_rgba8 },

{ "gelu_tanh_bf16", "gelu_tanh", "elem",
  "tanh gelu",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_gelu_tanh },

{ "gelu_quick_bf16", "gelu_quick", "elem",
  "CLIP's quick_gelu, libref's three roundings",
  "any M and n, row pitches honoured",
  "bf16 or f32 in and out (one dtype), f32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGeluQuick), NOTUNE,
  nullptr, nullptr, r4d_rad_gelu_quick },

{ "scatter_add_rows_bf16", "scatter_add_rows", "elem",
  "x[idx[i]] += v[i]",
  "any M, row pitches honoured",
  "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenRows8), NOTUNE,
  nullptr, nullptr, r4d_rad_scatter_add_rows },

{ "conv2d_nhwc_bf16", "conv2d_nhwc", "conv",
  "an image convolution as an implicit GEMM over (tap, cin), the weight stored tap-major",
  "128 pixels x 64 or 128 output channels a block, 64 K a stage; cin % 8 != 0 takes a direct kernel",
  "bf16 image, weight and bias, fp32 accumulate, one rounding",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_conv2d_nhwc, nullptr, r4d_rad_layout_conv_tap, r4d_rad_relayout_conv_tap,
  nullptr, r4d_rad_unrelayout_conv_tap },

{ "im2col_nhwc_bf16", "im2col_nhwc", "elem",
  "a convolution's receptive fields as GEMM rows in the weight's (cin, kh, kw) order: 32 output "
  "pixels by 64 channels a block, the input rows staged in LDS, coalesced loads and stores",
  "k <= 7, stride and up 1 or 2; x rows at their own pitch; y exactly Ho * Wo rows at its pitch",
  "bf16 in and out, copied bit for bit",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_im2col_nhwc },

/* -------------------------------------------------------------------------- audio generation */
{ "conv1d_bf16", "conv1d", "conv",
  "a 1-D convolution over sequences as an implicit GEMM over (tap, cin), the weight tap-major",
  "128 rows x 64 or 128 output channels a block, 64 K a stage; cin % 8 != 0 or cout < 16 takes a "
  "direct kernel",
  "bf16 rows, weight, bias and residual, fp32 accumulate, one rounding; act none, tanh or clamp",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_conv1d, nullptr, r4d_rad_layout_conv1d_tap,
  r4d_rad_relayout_conv1d_tap, nullptr, r4d_rad_unrelayout_conv1d_tap },

{ "conv1d_f32", "conv1d", "conv",
  "a 1-D convolution over sequences in f32 (a vocoder kept in f32): an implicit GEMM over "
  "(tap, cin) on the f32 FMA tile, the weight tap-major",
  "128 rows x 128, 64 or 32 output channels a block, 16 K a slab; fewer than 16 output channels "
  "take a direct kernel; `res` may be the output itself",
  "f32 rows, weight, bias and residual, f32 accumulate; act none, tanh or clamp",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_conv1d_f32, nullptr, r4d_rad_layout_conv1d_tap_f32,
  r4d_rad_relayout_conv1d_tap_f32, nullptr, r4d_rad_unrelayout_conv1d_tap_f32 },

{ "conv_t1d_bf16", "conv_t1d", "conv",
  "a transposed 1-D convolution as one implicit GEMM a phase over its one or two taps, the "
  "weight phase-major",
  "128 input rows x 64 or 128 output channels a block, a grid slice a phase; cin % 8 == 0",
  "bf16 rows, weight and bias, fp32 accumulate, one rounding",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_conv_t1d, nullptr, r4d_rad_layout_convt_phase,
  r4d_rad_relayout_convt_phase, nullptr, r4d_rad_unrelayout_convt_phase },

{ "conv_t1d_f32", "conv_t1d", "conv",
  "a transposed 1-D convolution in f32 as one implicit GEMM a phase over ceil(k / stride) taps, "
  "the weight phase-major with the taps past k zero",
  "any k of at least a stride and any padding; 128 input rows x 128, 64 or 32 output channels a "
  "block, a grid slice a phase",
  "f32 rows, weight and bias, f32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r4d_rad_conv_t1d_f32, nullptr, r4d_rad_layout_convt_phase_f32,
  r4d_rad_relayout_convt_phase_f32, nullptr, r4d_rad_unrelayout_convt_phase_f32 },

{ "snake_bf16", "snake", "elem",
  "the snake activation per channel", "any M, row pitches honoured", "bf16 rows, f32 alpha",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_snake },

{ "cfg_topk_sample", "cfg_topk_sample", "sample",
  "a guided top-k pick a lane: radix selects for the conditional and the guided thresholds, ties "
  "to the lower index, inverse CDF over the kept entries",
  "a workgroup a lane, k <= 256, rep 1 or 2", "f32 or bf16 logits, i32 picks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAudioAny), NOTUNE,
  nullptr, nullptr, r4d_rad_cfg_topk_sample },

{ "rvq_embed_bf16", "rvq_embed", "elem",
  "a frame's embedding from its residual codes", "a workgroup a row", "bf16 tables and rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_rvq_embed },

{ "frame_mix_store_bf16", "frame_mix_store", "elem",
  "a frame's conditioning row into its lane", "a workgroup a lane, L <= 16",
  "bf16 planes and rows, f32 logits and scale",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_frame_mix_store },

{ "end_status", "end_status", "elem",
  "a frame's status from its first pick", "a thread a lane", "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_end_status },

{ "frame_store", "frame_store", "elem",
  "a frame's codes into its lane's history", "a thread a (lane, code); dst contiguous", "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_frame_store },

{ "resample_nearest_bf16", "resample_nearest", "elem",
  "rows resampled nearest-neighbour", "a workgroup an output row", "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_resample_nearest },

{ "fourier_embed_bf16", "fourier_embed", "elem",
  "random Fourier features of a time", "a workgroup a row", "f32 time and w, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_fourier_embed },

{ "win_input_bf16", "win_input", "elem",
  "a denoiser's input rows from the window states", "a workgroup a row", "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_win_input },

{ "win_prep_bf16", "win_prep", "elem",
  "a window step's noise and overlap binding", "a grid slice an entry", "bf16 states, f32 t",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_win_prep },

{ "win_finish_bf16", "win_finish", "elem",
  "a window's end and the overlap it hands on", "two kernels: the hand-on, then the join",
  "bf16 states",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_win_finish },

{ "chan_split_bf16", "chan_split", "elem",
  "channel groups to sequences", "any M, row pitches honoured", "bf16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_chan_split },

{ "interleave2_bf16", "interleave2", "elem",
  "two streams interleaved into stereo samples", "contiguous", "bf16 or f32 in, f32 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAudioAny), NOTUNE,
  nullptr, nullptr, r4d_rad_interleave2 },

{ "iota", "iota", "elem",
  "an arithmetic index sequence", "contiguous i32", "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_iota },

{ "cu_repeat", "cu_repeat", "elem",
  "bounds of equal sequences from a device length", "contiguous i32", "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_cu_repeat },

{ "int_affine", "int_affine", "elem",
  "integers derived from device geometry: one wave, the lists by value", "up to 16 values",
  "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_int_affine },

{ "attn_dense_small_h256_bf16", "attn_dense", "attn",
  "dense attention over short sequences, causal or not: a depth decoder's frame positions",
  "head_dim 256, one query head a key head, sequences of at most 32 rows, a workgroup a (sequence, "
  "head) with its keys and values in LDS",
  "bf16 query / key / value / out, fp32 scores and accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnDenseSmallH256), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_dense_small_h256 },

{ "quant_cols_fp8_bf16", "quant_cols_fp8", "quant",
  "E4M3 codes with one scale per column over every row: two passes, column maxima over 32 row "
  "chunks then the codes",
  "n a multiple of 8, a thread eight columns; the chunks' maxima in scratch",
  "bf16 in, E4M3 out at its own pitch, f32 scale per column",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQuantCols), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_cols_fp8, r4d_rad_quant_cols_fp8_scratch },

{ "vsa_select_h128_bf16", "vsa_select", "attn",
  "video sparse attention's tile selection: f32 tile means of q, k (and v), f32 tile scores, the "
  "top k video tiles of each video query tile ascending, and the coarse softmax over the tiles "
  "times the v means",
  "head_dim 128, 4 x 4 x 4 tiles, at most 3072 tiles; the k and v means of a group of heads in "
  "scratch, eight (four above 1536 tiles) query tiles a workgroup",
  "bf16 q / k / v / coarse, i32 segs / geom / sel; f32 means, scores and softmax",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cVsa), NOTUNE,
  nullptr, nullptr, r4d_rad_vsa_select, r4d_rad_vsa_select_scratch },

{ "attn_vsa_h128_bf16", "attn_vsa", "attn",
  "block-sparse attention over the selected 64-row tiles, with the gated coarse term",
  "head_dim 128, 4 x 4 x 4 tiles, a workgroup two query tiles of one head walking the merge of "
  "their key tiles, each tile's rows computed from the geometry; the int8 Q.K form (k_i8 / "
  "k_scale)",
  "bf16 q / k / v / coarse / gate / out, i32 segs / geom / sel; bf16 operands, fp32 accumulate; "
  "or i8 Q.K codes, i32 accumulate; or E4M3 V codes and P, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cVsa), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_vsa },

{ "rms_mod_bf16", "rms_mod", "norm",
  "gated residual and modulated RMS norm in one pass: a workgroup a row",
  "n % 8 == 0, n <= 8192; s a column slice at its own pitch",
  "bf16 rows, gain and table, fp32 sum of squares",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRmsMod), NOTUNE,
  nullptr, nullptr, r4d_rad_rms_mod },

{ "rms_mod_quant_fp8_bf16", "rms_mod_quant_fp8", "norm",
  "gated residual, modulated RMS norm and quant_act_fp8 in one pass: a workgroup a row",
  "n % 128 == 0, n <= 8192, group 128; s a column slice at its own pitch",
  "bf16 rows, gain and table, E4M3 codes and f32 scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRmsModQ), NOTUNE,
  nullptr, nullptr, r4d_rad_rms_mod_quant_fp8 },

{ "flow_x0_step_f32", "flow_x0_step", "elem",
  "a sampler step toward the clean estimate", "any M, row pitches honoured",
  "f32 state, f32 or bf16 velocity",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenF32), NOTUNE,
  nullptr, nullptr, r4d_rad_flow_x0_step },

{ "patch_unfold_f32", "patch_unfold", "elem",
  "patch rows to denormalised voxel rows, the geometry read on the device",
  "C <= 64, row pitches honoured", "f32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPatchUnfold), NOTUNE,
  nullptr, nullptr, r4d_rad_patch_unfold },

{ "vtile_gather", "vtile_gather", "elem",
  "a video decoder's tiles out of the latent, the tile layout from the device geometry: a "
  "workgroup a tile row",
  "row pitches honoured", "bf16 or f32 rows and registers",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_vtile_gather },

{ "vtile_stitch", "vtile_stitch", "elem",
  "decoded tiles blended into frames and the chunk overlap: a thread a (pixel, colour) over the "
  "chunk's frames",
  "row pitches honoured", "bf16 or f32 tiles and overlap, u8 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_vtile_stitch },

{ "grid_pos3", "grid_pos3", "elem",
  "a video decoder's rotary positions over its tiles", "a [3, M] view at its own pitch", "f32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_grid_pos3 },

{ "snake_aa_f32", "snake_aa", "elem",
  "the alias-free SnakeBeta: a thread an output, its activations recomputed per output",
  "k <= 32, row pitches honoured", "f32 rows, filters, alpha and beta",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cSnakeAa), NOTUNE,
  nullptr, nullptr, r4d_rad_snake_aa },

/* ------------------------------------------------------------------- a video latent upscaler */
{ "conv3d_ndhwc_bf16", "conv3d_ndhwc", "conv",
  "a 3 x 3 x 3 video convolution as an implicit GEMM over (tap, cin): the weight stored tap-major "
  "and read into registers, the voxels' tap neighbours staged through LDS",
  "128 voxels x 128 output channels a block, 64 K a stage; k = kt = 3, cin % 8 == 0 (a multiple "
  "of 64 decodes the tap once per cin / 64 stages); row pitches honoured, res may alias y",
  "bf16 voxels, weight and residual, bf16 or f32 bias, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cConv3d), NOTUNE,
  nullptr, nullptr, r4d_rad_conv3d_ndhwc, nullptr, r4d_rad_layout_conv3d_tap,
  r4d_rad_relayout_conv3d_tap, nullptr, r4d_rad_unrelayout_conv3d_tap },

{ "group_norm", "group_norm", "norm",
  "GroupNorm over all rows in three launches: per-part (count, mean, M2), their merge to mean and "
  "rstd, then the affine, modulation and SiLU a thread an 8-, 4- or 2-channel chunk",
  "n % 8 == 0 up to 8192, an even n / groups, groups <= 256, M n / groups < 2^31; row pitches "
  "honoured, y may alias x",
  "bf16 or f32 rows, bf16 or f32 gain, bias and modulation, f32 statistics",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGroupNorm), NOTUNE,
  nullptr, nullptr, r4d_rad_group_norm },

{ "tconv_dw_bf16", "tconv_dw", "conv",
  "a depthwise convolution along the frames: a thread an 8-channel chunk of a frame position, the "
  "5-frame window in registers over every frame",
  "k = 5, n % 8 == 0, row pitches honoured", "bf16 voxels, weight and bias, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cTconvDw), NOTUNE,
  nullptr, nullptr, r4d_rad_tconv_dw },

{ "resize_bilinear", "resize_bilinear", "elem",
  "torch's per-frame bilinear resize at a size: a thread an 8-channel chunk of an output voxel",
  "n % 8 == 0, any sizes, row pitches honoured", "bf16 or f32 voxels, f32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cResizeBil), NOTUNE,
  nullptr, nullptr, r4d_rad_resize_bilinear },

{ "patch_fold_f32", "patch_fold", "elem",
  "voxel rows folded to normalised patch rows, the geometry read on the device",
  "C <= 64, row pitches honoured", "bf16 or f32 voxels, f32 patch rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPatchFold), NOTUNE,
  nullptr, nullptr, r4d_rad_patch_fold },

{ "flow_renoise_f32", "flow_renoise", "elem",
  "a clean latent re-noised to sigma from a fresh normal stream: a thread a value",
  "any M and n, row pitches honoured", "f32 latent and sigma, u32 seed words",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenF32), NOTUNE,
  nullptr, nullptr, r4d_rad_flow_renoise },

{ "rope_split", "rope_split", "elem",
  "LTX-2's rotary: a workgroup a (head, 64 rows), the head's ladder in LDS, sincosf a pair",
  "any M, row pitches honoured; head_dim / 2 dividing 256; up to four axes",
  "bf16 or f32 x, f32 or i32 positions",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_rope_split },

{ "head_gate", "head_gate", "elem",
  "each head times 2 sigmoid of its logit: a thread a value",
  "any M, row pitches honoured", "bf16 logits, bf16 or f32 x",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_head_gate },

{ "ltx_guide", "ltx_guide", "elem",
  "LTX-2's guided step: f64 deviation sums in up to 1024 parts, then the step in place",
  "any M and n, row pitches honoured", "f32 latent, bf16 or f32 velocities",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_ltx_guide },

{ "pixel_norm", "pixel_norm", "norm",
  "an RMS norm as bf16 tensors compute it: a wave a row, its sum of squares a wave reduction",
  "any M and n, row pitches honoured", "bf16 or f32 x and y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_pixel_norm },

{ "d2s3d", "d2s3d", "elem",
  "depth to space over voxel rows: a thread a value, the volume off the device",
  "any C and factors, row pitches honoured", "bf16 or f32 x and y, i32 geom",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_d2s3d },

{ "s2d3d", "s2d3d", "elem",
  "space to depth over voxel rows, copy or group mean, optionally added: a thread a value",
  "any C and factors, row pitches honoured", "bf16 or f32 x and y, i32 geom",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_s2d3d },

{ "pix_patch", "pix_patch", "elem",
  "an image to patch rows: a thread a value",
  "dense 2-D operands, row pitches honoured", "bf16 or f32 x and y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_pix_patch },

{ "patch_pix8", "patch_pix8", "elem",
  "patch rows to RGB bytes: a thread a byte",
  "dense 2-D operands, row pitches honoured", "bf16 or f32 x, u8 y",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_patch_pix8 },

{ "vol_index", "vol_index", "elem",
  "a volume's voxel coordinates: a thread a voxel", "i32, a dense [3, n]", "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_vol_index },

{ "tile_grid", "tile_grid", "elem",
  "a tiled decode's boxes off a device volume: a thread a row", "contiguous i32", "i32",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  nullptr, 0, NOTUNE,
  nullptr, nullptr, r4d_rad_tile_grid },

{ "tile_crop", "tile_crop", "elem",
  "a tile's voxels out of a volume: a thread a value, the box read on the device",
  "row pitches honoured", "bf16 or f32 x and y, i32 geom and box",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_tile_crop },

{ "tile_stitch", "tile_stitch", "elem",
  "decoded tiles blended to RGB bytes: a thread a byte, up to four tiles read",
  "dense 2-D operands, row pitches honoured", "bf16 or f32 x, u8 y, i32 geom and first",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_tile_stitch },

{ "stft_logmel", "stft_logmel", "elem",
  "a causal log-mel spectrogram: a workgroup a (frame, channel), the frame in LDS",
  "nfft up to 4096, row pitches honoured", "bf16 or f32 operands",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_stft_logmel },

{ "bwe_up", "bwe_up", "elem",
  "a sinc-resampled waveform plus a residual, clamped: a thread a value, the taps by value",
  "ratio up to 9, row pitches honoured", "bf16 or f32 operands",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGenAnyFloat), NOTUNE,
  nullptr, nullptr, r4d_rad_bwe_up },
};

/* ================================================================================== the exports */

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libr4d",
    "0.5.0",
    "the RDNA4 (gfx1201) kernel library: paged and dense attention, gated delta net, "
    "all-reduce over P2P or a host-memory mailbox, skinny and tiled quantised GEMM, and the "
    "rotate-and-quantise fusions",
    "gfx1201",
};

/* r4d_record_dispatches' log, read by r4d_emit (r4d_emit.h). */
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n   = 0;

extern "C" int r4d_record_dispatches(const void** fns, int cap) {
    const int n = r4d_dispatch_n;
    r4d_dispatch_log = fns;
    r4d_dispatch_cap = fns ? cap : 0;
    r4d_dispatch_n   = 0;
    return n;
}

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

extern "C" int rad_kernel_schema_count(void) {
    return (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    return &kSchemas[i];
}
/* ================================================================================ the host twins
 *
 * Without peer access, the same collectives run with their mailbox in host memory. Every row using
 * one of the three all-reduce inits gets a twin: same op, kernel, launch, fini, priority and tuning
 * axes, `_host` appended to its name, device.p2p == 1 turned into device.p2p == 0, and the init
 * that puts every rank's exchange buffers in pinned host memory (r4d_ar_entry.hip, build). Exactly
 * one of a pair matches any machine.
 *
 * Twins are derived, so they cannot drift from the table. A collective row that does not require
 * device.p2p == 1 gets no twin, and r4d_selftest fails it. */
namespace {

struct HostTwin {
    RadKernelInfo              info;
    std::string                name, computes, shape;
    std::vector<RadConstraint> c;
};

RadInitFn host_init_for(RadInitFn f) {
    if (f == r4d_rad_ar_init)         return r4d_rad_ar_host_init;
    if (f == r4d_rad_ar_ln_init)      return r4d_rad_ar_ln_host_init;
    if (f == r4d_rad_ar_hc_wide_init) return r4d_rad_ar_hc_wide_host_init;
    return nullptr;
}

std::string replaced(std::string s, const char* from, const char* to) {
    const size_t at = s.find(from);
    if (at != std::string::npos) s.replace(at, std::strlen(from), to);
    return s;
}

const std::vector<HostTwin>& host_twins() {
    static const std::vector<HostTwin> v = [] {
        std::vector<HostTwin> t;
        for (const RadKernelInfo& k : kKernels) {
            const RadInitFn init = host_init_for(k.init);
            if (!init) continue;
            HostTwin h;
            h.c.assign(k.constraints, k.constraints + k.n_constraints);
            int p2p = 0;
            for (RadConstraint& c : h.c)
                if (!std::strcmp(c.key, RAD_FACT_P2P) && c.op == RAD_C_EQ && c.ival == 1) {
                    c.ival = 0;
                    ++p2p;
                }
            if (p2p != 1) continue;
            h.info = k;
            h.info.init = init;
            h.name = std::string(k.name) + "_host";
            h.computes = k.computes;
            if (h.computes.find("over P2P IPC scratch") != std::string::npos)
                h.computes = replaced(h.computes, "over P2P IPC scratch",
                                      "through a pinned host-memory mailbox");
            else
                h.computes += ", exchanged through a pinned host-memory mailbox";
            h.shape = replaced(replaced(k.shape, "all peer-accessible", "no peer access needed"),
                               "peer-accessible", "no peer access needed");
            t.push_back(std::move(h));
        }
        /* The strings are pointed into only once the vector has stopped growing. */
        for (HostTwin& h : t) {
            h.info.name          = h.name.c_str();
            h.info.computes      = h.computes.c_str();
            h.info.shape         = h.shape.c_str();
            h.info.constraints   = h.c.data();
            h.info.n_constraints = (int)h.c.size();
        }
        return t;
    }();
    return v;
}

}  /* namespace */

extern "C" int rad_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0])) + (int)host_twins().size();
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    const int base = (int)(sizeof(kKernels) / sizeof(kKernels[0]));
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    if (i < base) return &kKernels[i];
    return &host_twins()[(size_t)(i - base)].info;
}

/* Which rows touch nothing but their operands and scratch (rad_abi.h). The exceptions keep state
 * between launches keyed on the stream (arrival counters, split workspaces allocated once), so only
 * the stream's ordering keeps two launches from sharing it:
 *
 *   hc_* and mtp_enter  r4d_hc_bf16.hip's mix counters, one block a stream
 *   gemm_w4a8_tiled,    the split tail's workspace and counters, one a (device, stream)
 *   gemm_w8a8_tiled
 *   every collective    the peer flags and exchange buffers the all-reduces own
 *
 * gemm_fp8a8_nt_m16's finisher counters are keyed on the stream and the output;
 * gemm_mxfp4a8_decode's split partial is the op's scratch and its arrival counters are keyed on the
 * output alone; router_gemm_topk_scatter_bf16's arrival counter belongs to its op instance (taken
 * at init, one per declared op and band). Concurrent launches never share these, so those three
 * rows are concurrent. A new row with stream-ordered state belongs in the list above.
 *
 *   lora_*              ordered on purpose: the core appends them behind the GEMM whose `y` they
 *                       update and refills the pools they read between steps with its own copies,
 *                       so they stay out of the replay's overlap analysis. */
extern "C" int rad_kernel_concurrent(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k) return 0;
    if (k->family && std::strcmp(k->family, "ar") == 0) return 0;
    static const char* const kStateful[] = {
        "hc_", "mtp_enter", "ar_", "gemm_w4a8_tiled", "gemm_w8a8_tiled", "lora_",
    };
    for (const char* p : kStateful)
        if (std::strncmp(k->name, p, std::strlen(p)) == 0) return 0;
    return 1;
}

/* Which rows are collectives (rad_abi.h): every row whose init builds the all-reduce family's
 * exchange buffers, peer flags and handshake (standalone all-reduces and gathers, and norms and hc
 * writes with one folded in), plus each `_host` twin. Keyed on the init, since a cross-rank kernel
 * cannot run without the instance only these build. */
extern "C" int rad_kernel_collective(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k || !k->init) return 0;
    const bool coll = host_init_for(k->init) != nullptr || k->init == r4d_rad_ar_host_init ||
                      k->init == r4d_rad_ar_ln_host_init || k->init == r4d_rad_ar_hc_wide_host_init;
    if (!coll) return 0;
    /* The rows that end the engine's link hold themselves, through `link_gate`, as soon as the
     * peers' pushes have landed over P2P (RAD_LINK_ENDED); through the host mailbox they leave it
     * to the engine (`in->host` keeps `early` 0, so the launch returns RAD_OK and the barrier
     * release stands).
     *
     * Invariant: a launch joins this list only if its device kernel calls r4d_ar_link_landed on
     * every path after its last handshake. A row answering 2 whose kernel does not decrement
     * leaves the gate odd, and every yielding DMA list on that card waits forever. The one-shot
     * family releases after its single handshake, the two-shot family after its second (its
     * reduce is fused with the phase-2 push); the 3- and 8-rank entries pass no gate and answer
     * 1. */
    static const RadLaunchFn kReleasing[] = {
        r4d_rad_ar_hc_write, r4d_rad_ar_gather_hc_write,
        r4d_rad_ar_oneshot_4rank_exact, r4d_rad_ar_twoshot_4rank_exact,
        r4d_rad_ar_twoshot_4rank_ti8, r4d_rad_ar_gather_4rank,
        r4d_rad_ar_hc_write_4rank, r4d_rad_ar_gather_hc_write_4rank,
        r4d_rad_ar_rmsnorm_quant_i8_4rank,
    };
    for (RadLaunchFn f : kReleasing)
        if (k->launch == f) return 2;
    return 1;
}

/* ------------------------------------------------------------------------ tuned config access
 *
 * The chosen point arrives as ordinary parameters: the core picks a point in the space above (from
 * the tune cache, RADIANCE_TUNE, or each axis's default) and appends it to RadArgs::p, so a launch
 * reads `sk` as it reads `M`.
 *
 * The fallback comes from the tables above, so each default is defined once, and a caller that
 * supplies no tuned key (rad-kbench, a selftest) runs what an untuned install runs. */
static int64_t tune_i(const RadArgs* a, const RadTunable* tab, int n, const char* key) {
    long long v = 0;
    if (rad_args_geti(a, key, &v)) return (int64_t)v;
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(tab[i].key, key)) return tab[i].deflt;
    return 0;
}
#define TUNE(a, tab, key) ((int)tune_i((a), (tab), NELEM(tab), (key)))

extern "C" void r4d_cfg_gemm(const RadArgs* a, R4dGemmCfg* out) {
    out->wv  = TUNE(a, tGemmSkinny, "wv");
    out->sk  = TUNE(a, tGemmSkinny, "sk");
    out->mb  = TUNE(a, tGemmSkinny, "mb");
    out->npw = TUNE(a, tGemmSkinny, "npw");
    out->nt  = TUNE(a, tGemmSkinny, "nt");
}

/* Maps a compiled tile arm to its (BM, BN) (see tGemmTiled); the only copy of this map. A call
 * checks `N % BN`, so a wrong BN here would silently compute a wrong answer. */
static void tile_bmbn(int arm, int prefill, R4dTileCfg* out) {
    struct Arm { int v, bm, bn; };
    static const Arm tiled[] = {
        { 0, 128, 256 }, { 6,  64, 256 }, { 10,  96, 256 }, { 12,  96, 128 }, { 57, 160, 128 },
        { 59, 128, 128 }, { 114, 80, 128 }, { 124, 64, 128 }, { 125, 112, 128 },
        { 126, 160, 128 }, { 128, 32, 128 }, { 129, 48, 128 }, { 138, 48, 256 },
        { 139, 112, 256 }, { 146, 144, 128 },
    };
    /* Every prefill arm listed is a shape ablation of BM 128 x BN 256 except the four that say
     * otherwise in their own name. */
    static const Arm pf[] = {
        { 0, 128, 256 }, { 2, 128, 256 }, { 3, 128, 256 }, { 5, 128, 256 }, { 6, 128, 256 },
        { 8,  64, 256 }, { 9,  96, 256 }, { 10, 128, 128 }, { 13, 64, 256 }, { 14, 256, 128 },
        { 27, 128, 256 }, { 28, 128, 256 }, { 29, 128, 256 }, { 30, 128, 256 },
    };
    const Arm* t = prefill ? pf : tiled;
    const int  n = prefill ? NELEM(pf) : NELEM(tiled);
    for (int i = 0; i < n; ++i)
        if (t[i].v == arm) { out->variant = arm; out->bm = t[i].bm; out->bn = t[i].bn; return; }
    /* An unknown arm falls back to the default rather than guessing a tile width. The loader
     * refuses a default outside its value list, so arm 0 is always reachable. */
    out->variant = 0; out->bm = 128; out->bn = 256;
}

extern "C" void r4d_cfg_tiled(const RadArgs* a, R4dTileCfg* out) {
    tile_bmbn(TUNE(a, tGemmTiled, "tile"), 0, out);
}
extern "C" void r4d_cfg_prefill(const RadArgs* a, R4dTileCfg* out) {
    tile_bmbn(TUNE(a, tGemmPrefill, "tile"), 1, out);
}
extern "C" void r4d_cfg_mxfp4_decode(const RadArgs* a, R4dMxDecCfg* out) {
    out->ks = TUNE(a, tGemmMxfp4Decode, "ks");
    out->bk = TUNE(a, tGemmMxfp4Decode, "bk");
    out->nt = TUNE(a, tGemmMxfp4Decode, "nt");
}
extern "C" void r4d_cfg_mxfp4_tiled(const RadArgs* a, R4dMxTileCfg* out) {
    out->tn = TUNE(a, tGemmMxfp4Tiled, "tn");
}
extern "C" void r4d_cfg_split(const RadArgs* a, R4dSplitCfg* out) {
    out->splits = TUNE(a, tAttnSplit, "splits");
}
extern "C" void r4d_cfg_ar(const RadArgs* a, R4dArCfg* out) {
    out->nblocks  = TUNE(a, tAr, "nblocks");
    out->nthreads = TUNE(a, tAr, "nthreads");
    out->drain    = TUNE(a, tAr, "drain");
    out->acq      = TUNE(a, tAr, "acq");
    /* An axis of the four-rank junction only (tArQ84); 1 for every other row, as before. */
    out->pub      = TUNE(a, tArQ84, "pub");
}
extern "C" int r4d_cfg_ar_tail(const RadArgs* a) {
    return TUNE(a, tArQ84, "tail");
}
extern "C" long r4d_cfg_ar_ts_min(const RadArgs* a) {
    return (long)tune_i(a, tAr, NELEM(tAr), "ts_min");
}
extern "C" void r4d_cfg_topk(const RadArgs* a, R4dTopkCfg* out) {
    out->rc  = 0;                      /* derived from pt1 at issue; see the note at tTopk */
    out->nch = TUNE(a, tTopk, "nch");
    out->pt1 = TUNE(a, tTopk, "pt1");
}
extern "C" void r4d_cfg_mv(const RadArgs* a, R4dMvCfg* out) {
    out->rb = TUNE(a, tMv, "rb");
}
extern "C" int r4d_cfg_rmsq8_reg(const RadArgs* a) {
    return TUNE(a, tRmsQ8, "reg");
}
