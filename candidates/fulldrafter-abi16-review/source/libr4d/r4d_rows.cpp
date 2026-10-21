/* r4d_rows.cpp -- the libr4d row table, and the op schemas that go with it.
 *
 * THIS IS THE TABLE -- the single declaration of what libr4d can serve. A KERNEL MISSING FROM THIS
 * FILE IS INVISIBLE to every caller that asks rather than remembers, and a second table beside it
 * drifts from it, so there is exactly one.
 *
 * A row's constraints are the ENTRY POINT'S rules, restated here; each row says where its band
 * came from, which is the question a reader of one of these rows actually has. Three things do not
 * line up one-to-one with the kernel side's own spelling, and each is marked at the row that does
 * it:
 *
 *   1. KEY NAMES. The kernel library's select() has its own parameter vocabulary; radiance's op
 *      schemas are docs/OPS.md's, because that is what an architecture plugin writes and what
 *      `libref` has to implement. Where the two differ the PREDICATE and its VALUE are the
 *      kernel's exactly and only the key is renamed, with the kernel-side key named in a comment.
 *      `K` -> `n` in the quantiser family, `had` -> `group` for the rotation width, and
 *      `q_dtype`/`kv_dtype` -> `dtype` for the dense vision kernel are the whole list.
 *
 *   2. ORDER IS `priority`, not position. Preference between two rows that both match is expressed
 *      by the R4D_PRIO_* constants, each of which names the kernel-side ordering it preserves
 *      (r4d_plugin.h).
 *
 *   3. ROWS WHOSE BAND COMES FROM THE ENTRY POINT ITSELF -- the tiled and prefill W4A8/W8A8 GEMMs,
 *      the asymmetric grids, `quant_act_i8_asum`, `gemm_w8a8_nt_m64`. Their constraints are read
 *      off the entry point's own rejection tests and its header prose rather than off a
 *      kernel-side registry row, and every one of them says so.
 *
 * Every row is RAD_DOMAIN_DEVICE. libr4d is a gfx1201 kernel library; there is no host arm of any
 * of this, and claiming one would tell the placement planner that a host execution site exists
 * where it does not.
 */

#include "r4d_plugin.h"

#include <cstdlib>
#include <cstring>

#define ROW(a) (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Parameter kinds, spelled once. A real-valued parameter is RAD_P_F64 (docs/OPS.md, "The four
 * rules"): eps, scale, theta, softplus_thr, l2_eps and wadd are all real, and round-tripping one
 * through a "%.9g" string is exact for an IEEE single but unreadable in a graph dump. RAD_DERIVED
 * is a value the KERNEL supplies rather than the caller -- `block_size` and `chunk` -- and a
 * constraint on one is skipped during matching and read back off the winner (spec §7.2). */
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

/* Spelled the way the kernel side spells its own constraints, so a row can be compared against the
 * entry point line for line. */
#define C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }

/* EVERY KERNEL THAT TOUCHES A PEER'S MEMORY: the collectives push into the other ranks' scratch and
 * raise their flags directly, so a machine whose cards cannot reach each other must not select
 * them. The engine measures this at launch (rad_types.h, device.p2p). */
#define C_P2P       C_EQ(RAD_FACT_P2P, 1)

/* libr4d's R4D_AR_TWOSHOT_WIDE_MAX_ELEMS, restated because this file must not include r4d.h.
 * r4d_ar_entry.hip static_asserts the two against each other. */
#define R4D_AR_TWOSHOT_WIDE_MAX_ELEMS 20971520

/* The largest message, in elements, a 4- or 8-rank exact all-reduce keeps on the one-shot; above it
 * the two-shot rows take it. 1 MiB of a 16-bit payload. The one-shot sends (ws-1)*N bytes a rank
 * in one hop and the two-shot 2*(ws-1)/ws*N in two, so the one-shot wins while the second hop's
 * latency outweighs the bytes it saves. The value is a separation, not a measured crossover: on
 * Qwen3.8-Flash-Next (2560 wide, 5 KiB a row) the largest decode message is 640 KiB, at 32
 * sequences each verifying four tokens, and a 2048-token prefill chunk is 10 MiB, so any bound
 * between the two sends decode one-shot and prefill two-shot. It is a multiple of 128 for the
 * reason pAllReduce's `numel` gives. */
#define R4D_AR_WIDE_ONESHOT_MAX_ELEMS 524288
#define R4D_XSTR(v) R4D_STR(v)
#define R4D_STR(v)  #v

/* ==================================================================================== schemas
 *
 * The first plugin in hierarchy order to declare an op fixes its schema (spec §2.3), and this
 * plugin sits ahead of `libref`. So where an op below carries operands or parameters that
 * docs/OPS.md does not list, THAT IS A CHANGE TO THE VOCABULARY and it is called out at the
 * schema, not buried: ref has to grow the same operand or the two plugins are refused against
 * each other at load, which is the correct outcome and the reason the check exists.
 */

/* -------------------------------------------------------------------------------- attention */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"),
    P_INT("head_dim"),
    P_INT("gqa"),
    /* SUPPLIED BY THE KERNEL, not the caller (spec §7.2): the block manager reads the paged block
     * size OFF the resolved attention kernel, so a caller cannot state it in the query that
     * resolves it. libr4d's rows answer 16. Declared RAD_REQUIRED it would make those rows
     * unreachable by the only query anyone can ask. */
    P_INTD("block_size"),
    P_INT("causal"),
    P_INT("window"),
    P_STR("q_dtype"),
    P_STR("kv_dtype"),
    /* The softmax scale is a model constant and libr4d takes it per launch. Absent, the shim uses
     * 1/sqrt(head_dim), which is right for every model in scope and wrong for anything that
     * scales its query differently -- so it is a parameter and not a hardcode. */
    P_F64O("scale"),
    /* NOT IN docs/OPS.md, and needed at DECLARE: the split-KV scratch hook has to size the arena
     * before there is a batch, and libr4d's split law reads the host-side context bound. Absent,
     * the shim derives it from the block table's width, which is an upper bound and therefore
     * safe. */
    P_INTO("max_ctx"),
    /* The other two the scratch hook needs, and for the same reason: RadScratchFn is called at
     * DECLARE, when the op has parameters and no tensors, and libr4d's split-KV partial buffer is
     * num_seqs * q_len * q_heads * splits rows. q_heads and num_seqs are not derivable from
     * anything else in this list -- gqa gives the ratio, not the count.
     *
     * `max_seqs` is the BOUND, not the step's sequence count, and that is correct rather than
     * conservative: the scratch region is arena storage sized once for the worst step, so sizing
     * it for max_seqs is what makes every later step fit. Absent, the shim falls back to the
     * tensors, which exist at issue and not at declare. */
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
    { "tree_ancestors", RAD_OPD_IN, 1 },  /* DARTree: 2D causal mask bitmasks */
};

/* attn_paged with gate_quant_fp8 folded into its merge: attn_paged's parameters and the two of
 * gate_quant_fp8's that are not implied -- its `n` is `head_dim` and its rows are attn_paged's
 * (token, head) pairs -- and attn_paged's operands, then the gate, the codes and the scales. `out`
 * holds the gated bf16 product, which is where the in-place pair leaves it. */
static const RadParamSpec pAttnPagedGq[] = {
    P_INT("q_len"),
    P_INT("head_dim"),
    P_INT("gqa"),
    /* REQUIRED here where attn_paged derives it: the cache's page size is read off the resolved
     * attn_paged kernel, and the caller passes that value on rather than this op answering it a
     * second time. */
    P_INT("block_size"),
    P_INT("causal"),
    P_INT("window"),
    P_STR("q_dtype"),
    P_STR("kv_dtype"),
    P_F64O("scale"),
    P_INTO("max_ctx"),
    P_INTO("n_head"),
    /* A capacity: the row launches by the block table's rows and checks its scratch against
     * them, so a declare at a smaller step may bound it lower -- which the gated attention block
     * does, because it sizes this op at the step (one launch a step on the sparse path). */
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
    /* libr4d's `max_seqlen` selects the query-block height and is a HOST-side bound on the
     * cu_seqlens segments. Absent, the shim passes the total token count, which is a valid bound
     * and picks the tall arm -- a speed choice, never a correctness one. */
    P_INTO("max_seqlen"),
};
static const RadOperandSpec oAttnDense[] = {
    { "q",          RAD_OPD_IN,  0 },
    { "k",          RAD_OPD_IN,  0 },
    { "v",          RAD_OPD_IN,  0 },
    { "cu_seqlens", RAD_OPD_IN,  0 },
    { "out",        RAD_OPD_OUT, 0 },
};

/* docs/OPS.md has `qk_norm_rope` (qkv in place, `theta`). libr4d's kernel is a DIFFERENT op: it
 * splits a [q|gate] packed projection per head, writes q, k and the gate to three separate
 * destinations, and takes a PRECOMPUTED cos/sin table rather than theta. One operand list cannot
 * be both, so this is its own op name and `qk_norm_rope` stays available for a kernel that
 * matches OPS.md. */
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
/* THE q HALF IS OPTIONAL, WHICH IS THE K-ONLY FORM. `n_head` 0 passes no `q_gate`, no `q_w` and
 * no `q_out`, and every head the op serves is then read from `k` at its own row pitch and written
 * contiguously to `k_out`. That is not a special case bolted on: the k side IS the plain
 * [tokens, heads, head_dim] reader, and the only thing the q side adds is the [q|gate] split. An
 * input with no gate to split is a k. */
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
/* The prologue with kv_store folded in: its parameters and kv_store's, its operands and then
 * kv_store's `v`, `slot_mapping` and `kv_cache` -- the `k` kv_store reads is `k_out`. */
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
/* The table that operand reads, filled at each position's OWN row of a [max_ctx, rot] plane. It is
 * the reason the fused prologue is reachable at all: nothing in the tree built such a table, and
 * `theta` is not one of the fused op's parameters. `rot` and the two f64s are `rope`'s own, spelled
 * the same way, because the two have to compute the same angles or the fusion is a different
 * model. */
static const RadParamSpec pRopeTable[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "rot",   RAD_P_INT, RAD_REQUIRED },
    P_F64("theta"),
    P_F64("scale"),
    { "dtype", RAD_P_STR, RAD_REQUIRED },
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
    /* The chunk length the gate cumsum is cut at is the same one gdn_chunk_scan tiles on, and that
     * one belongs to the kernel. Declaring it here as anything but DERIVED gives the same number
     * two sources, and a disagreement is a decay summed over the wrong span rather than a
     * diagnostic. libr4d answers 64. */
    P_CHUNK("chunk"),
    P_INT("conv_width"),
    /* The l2 norm's epsilon. libr4d has no argument for it -- it is 1e-6f in the kernel body, FLA's
     * L2NORM_EPS -- so the shim accepts the key only at that value and refuses anything else by
     * name rather than silently normalising at a different epsilon. */
    P_F64O("l2_eps"),
    P_F64O("softplus_thr"),
};
/* THE INPUTS COME FIRST AND THE OUTPUTS LAST. `a` and `b_gate` are the gate and beta projections,
 * which the layer hands separately from x (x is the qkv slice); `cache_idx` is the conv-state slot
 * each sequence owns and `has_init` says whether that slot holds a history. All four are optional
 * in the SCHEMA -- libref reads a and b_gate out of the fused x row when they are absent --
 * and the first three are required by THIS shim, which dereferences them unconditionally and so
 * returns RAD_E_INVAL naming them rather than reading a null. */
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
    /* REQUIRED. spec §10 makes speculative rollback part of this op's schema rather than an
     * assumption: the conv window has already absorbed the rejected tokens and a rejection is a
     * change of read offset. An op that lets the caller omit num_accepted is an op whose rollback
     * contract is optional. A non-speculative decode passes all ones. */
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
    /* An fp16 SSM state costs less step time than an fp32 one and frees KV capacity, so which
     * one a deployment runs is a declared choice. */
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
    /* The state slot each batch row owns. Optional: without it a row owns slot n, which is what
     * a caller that hands slots out in batch order means. */
    { "state_idx", RAD_OPD_IN, 1 },
};

/* `norm_eps` and `norm_act` are libr4d's argument names; the vocabulary calls the folded gated
 * norm's epsilon `eps` and its activation `act`, because that is what the unfused gdn_gated_rmsnorm
 * calls them and one op composing another's arithmetic cannot rename its parameters. `act` is a
 * NAME -- "silu" or "sigmoid" -- and not an integer: a graph dump reading `act=1` is precisely the
 * "unclear what actually runs" complaint this engine exists to answer, and the shim maps the name
 * to the integer libr4d's entry point wants. */
static const RadParamSpec pGdnRecurrentUpdate[] = {
    P_INT("q_len"),
    P_INT("head_k"),
    P_INT("head_v"),
    /* Stated rather than left to the value tensor's rank: 48 value heads against 16 key heads is
     * the one geometry where guessing it from a flattened view gives a plausible wrong answer. */
    P_INTO("n_head_v"),
    P_F64O("scale"),
    P_F64O("eps"),
    /* The qk l2 norm's epsilon, 1e-6f in libr4d's kernel body and not an argument. The shim
     * accepts the key at that value and refuses any other by name. */
    P_F64O("l2_eps"),
    P_F64O("softplus_thr"),
    P_STRO("act"),
    P_INTO("state_fp16"),
    /* WHAT THE STATE CACHE MEANS, and the two forms produce the same `o` from a completely
     * different `state`: "per_candidate" writes one full state per candidate token and needs
     * 1 + n_spec distinct slots, "anchor" keeps one committed state a sequence and replays the
     * accepted prefix onto it out of a scratch slot. Optional because the out-of-tree caller that
     * shares this kernel has no RadArgs to declare it in and selects the arm by environment; a
     * caller that DOES declare it is naming the shape its pool was sized for, and the entry point
     * refuses the arm that disagrees rather than writing a state the pool cannot hold. */
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
    /* REQUIRED, for the same reason as on gdn_conv_update: the recurrent state has already
     * absorbed the rejected tokens, so rollback is part of the schema and not an assumption
     * (spec §10). A non-speculative decode passes all ones. */
    { "num_accepted", RAD_OPD_IN,     0 },
    /* cu IS AN OPERAND, in the same slot the sibling gdn_conv_update puts it. The decode kernel
     * opens with `bos = cu[n], T = cu[n+1] - bos` and dereferences it before any null test, so a
     * schema that omits it does not make it optional -- it hands the kernel a null and the launch
     * faults. The two decode kernels run off one cu on one step, and the two schemas say so in one
     * shape. */
    { "cu",           RAD_OPD_IN,     0 },
    { "z",            RAD_OPD_IN,     1 },
    { "norm_w",       RAD_OPD_WEIGHT, 1 },
    { "o",            RAD_OPD_OUT,    0 },
    /* THE OUT PROJECTION'S QUANTISER, OPTIONAL AND FOLDED IN. `o` is read by exactly one thing,
     * the fp8 out projection, and the standalone quant_act_fp8 between them costs a launch a layer
     * for less work than the gap in front of it. The fold is only expressible because the group
     * IS the row -- head_v is 128 and so is the fp8 block -- so the scale plane is one f32 per
     * (token, value head) and the kernel's own workgroup already owns exactly that. Absent, the
     * kernel writes bf16 only and the caller quantises separately; the prefill path does. */
    { "o_q",          RAD_OPD_OUT,    1 },
    { "o_scale",      RAD_OPD_OUT,    1 },
};

/* THE DECODE CONVOLUTION FOLDED INTO THE RECURRENT UPDATE. gdn_conv_update's only reader at
 * decode is gdn_recurrent_update, so the pair's q, k and v are an intermediate: this op takes the
 * convolution's inputs and the recurrent update's minus those three. The parameters are the
 * union of the two ops' under their own names. `z` and `norm_w` are REQUIRED here: the fold is
 * the decode shape, one workgroup per (sequence, value head), which is the shape the fused row
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

/* The GATED RESIDUAL. `hc_enter` and `hc_write` share a parameter list; `hc_read` adds the two
 * switches the operand list is banded on -- `inject` 0 is the final mixer and `group` 0 is a read
 * whose consumer wants bf16. Both DEFAULT ON, because every trunk connection wants both and the
 * exceptions are two sites in a 48-layer model. */
static const RadParamSpec pHcPlain[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype"),
};
static const RadOperandSpec oHcEnter[] = {
    { "x", RAD_OPD_IN, 0 }, { "h", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oHcWrite[] = {
    { "y", RAD_OPD_IN, 0 }, { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 },
};
/* hc_write with the preceding all-reduce folded in: `y` is the message too, so it is INOUT --
 * the reduced sum lands in it exactly as the standalone collective leaves it. */
static const RadParamSpec pArHcWrite[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("world_size"), P_STR("dtype"),
    P_INTO("wire"),   /* 0 = exact bf16, 6 = rotated 6-bit */
};
static const RadOperandSpec oArHcWrite[] = {
    { "y", RAD_OPD_INOUT, 0 }, { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 },
};
/* moe_gather's parameters and operands, then ar_hc_write's: the gather's output IS the reduction's
 * message, so it is one operand -- written by the gather half and reduced in place by the rest. */
static const RadParamSpec pArGatherHcWrite[] = {
    P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("top_k"), P_INT("world_size"), P_STR("dtype"),
    P_STRO("act"), P_INTO("wire"),
};
static const RadOperandSpec oArGatherHcWrite[] = {
    { "y_expert", RAD_OPD_IN, 0 }, { "expert_w", RAD_OPD_IN, 0 }, { "sorted_tok", RAD_OPD_IN, 0 },
    { "shared", RAD_OPD_IN, 1 },   { "shared_gate", RAD_OPD_IN, 1 },
    { "y", RAD_OPD_INOUT, 0 },     { "inj", RAD_OPD_IN, 0 },       { "h", RAD_OPD_INOUT, 0 },
};

/* The MTP head's entry, for a model whose residual stream is `hc` wide. `ngroup` is the one thing
 * the checkpoint cannot say -- hc rsqrts, one per sub-stream, or a single one over the whole wide
 * row -- so it is a parameter and defaults to the grouped form every gated residual here uses. */
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

/* Not in docs/OPS.md at all: the DFlash2 drafter's grouped dynamic depthwise convolution. It is a
 * drafter op, so it has no unfused equivalent in the vocabulary and no ref implementation --
 * which means the drafter has no fallback for it. Stated here so that is a decision on the record
 * (spec §17) and not a discovery. */
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
    /* A WEIGHT, not an input: `base_kernel` is a checkpoint parameter and the role is what puts
     * it in the op's weight list -- which is what gives it a prefetch interval and a residency
     * class. Declared as an input it would be a weight no op admits to using. */
    { "base",  RAD_OPD_WEIGHT, 0 },
    { "out",   RAD_OPD_OUT, 0 },
};

/* The other half of the drafter that has no equivalent anywhere else in the vocabulary: DFlash2's
 * candidate selector. A block-diffusion drafter computes every position of a block in one pass, so
 * position l+1 cannot be conditioned on the token chosen at position l -- the selector is what puts
 * that dependence back, as a low-rank bilinear score on the EDGE between consecutive positions plus
 * a greedy walk over those scores.
 *
 * `M` is ranged and is the SEQUENCE count; `steps` is the draft depth and is not, because the block
 * length is a property of the checkpoint (it was trained at one) and not of the step. Like
 * dflash_conv this has no ref implementation, so the drafter has no host fallback -- on the record
 * (spec §17), and the same decision for the same reason. */
static const RadParamSpec pDflashSelect[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "steps",   RAD_P_INT, RAD_REQUIRED },
    { "top_k",   RAD_P_INT, RAD_REQUIRED },
    { "rank",    RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    /* The stride between one sequence's anchor and the next INSIDE the `anchor` operand. It exists
     * so the operand can be the step's own token_ids: a draft pass carries `block` ids a sequence,
     * the anchor is the first of them, and the alternative is a second batch field holding a copy
     * of every `block`-th entry. Absent means a packed [M] array. */
    { "anchor_stride", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oDflashSelect[] = {
    { "cand",   RAD_OPD_IN,     0 },
    { "unary",  RAD_OPD_IN,     0 },
    /* OPTIONAL, and absent it is ones: the edge score drops from trilinear to a plain bigram
     * `<pred[prev], succ[cand]>`, which is DSpark's Markov head. r4d_dflash_select_bf16.hip
     * argues why that belongs in this walk and not in a head of its own. */
    { "hp",     RAD_OPD_IN,     1 },
    { "anchor", RAD_OPD_IN,     0 },
    { "pred",   RAD_OPD_WEIGHT, 0 },
    { "succ",   RAD_OPD_WEIGHT, 0 },
    { "tokens", RAD_OPD_OUT,    0 },
};

static const RadParamSpec pFulldrafterTwoStage[] = {
    { "M",            RAD_P_INT, RAD_REQUIRED },
    { "block_stride", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",        RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oFulldrafterTwoStage[] = {
    { "ktap0",       RAD_OPD_IN,  0 },
    { "ktap1",       RAD_OPD_IN,  0 },
    { "ktap2",       RAD_OPD_IN,  0 },
    { "ktap_anc",    RAD_OPD_IN,  0 },
    { "tokens_in",   RAD_OPD_IN,  0 },
    { "gpu_params",  RAD_OPD_IN,  0 },
    { "tokens_out",  RAD_OPD_OUT, 0 },
};

/* ------------------------------------------------------------------------------ collectives */
static const RadParamSpec pAllReduce[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    /* The message, in elements. Every block declares it as a RANGE -- a step carries its row
     * count times the width -- so the selector cuts it into bands at the bounds the rows below
     * place on it and resolves each band at its upper bound: init() sizes the peer scratch for
     * that bound once, and the launch reads the real count off the tensor. A row's RAD_C_DIV on
     * `numel` is therefore checked at a band's upper bound, so every bound a row here places on
     * it is a multiple of 128, the largest DIV among them; a bound that was not would leave a
     * band no DIV row could serve. */
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "exact",      RAD_P_INT, RAD_REQUIRED },
    /* The byte floor under which a LOSSY row must serve the message exactly anyway. Absent means
     * the shim's own measured default. It is a parameter and not a constant because it is a
     * property of the LINK, not of the kernel: the crossover sits where the rotate/pack/unpack
     * stops costing more than the bytes it removes, and that is a bandwidth the deployment knows
     * and the library does not. An exact row ignores it -- there is nothing to fall back to. */
    { "min_bytes",  RAD_P_INT, RAD_OPTIONAL },
    /* A caller's request for a one-shot (1) or two-shot (2). docs/OPS.md's schema is the union of
     * every plugin's parameters and libref accepts it, so it stays declared here; no row of this
     * plugin reads it, because the message size picks the two-shot (the all-reduce constraints
     * below). */
    { "hops",       RAD_P_INT, RAD_OPTIONAL },
};
/* docs/OPS.md has `x` -> `x` (inout). Every kernel here is safe in place -- the thread that
 * overwrites a word of the input is the one that read it (r4d_ar_entry.hip says so per kernel) --
 * and every block issues it that way. The output is an optional SECOND operand for a caller that
 * wants the sum somewhere else: absent means in place. */
static const RadOperandSpec oAllReduce[] = {
    { "x", RAD_OPD_INOUT, 0 },
    { "y", RAD_OPD_OUT,   1 },
};

/* No `exact` and no `hops`. A gather moves its payload, so there is no lossy wire to be told
 * apart from and no second topology at two ranks. `numel` is this rank's CONTRIBUTION -- the
 * output carries world_size times it -- and, like the all-reduce's, it is the bound init sizes
 * the peer scratch from once rather than the count a given step passes. */
static const RadParamSpec pAllGather[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    /* ONE ROW of this rank's contribution, in elements. Absent or 0 means the whole message is
     * one row and `y` is the plain concatenation: rank r's slice at r * numel. Present, and the
     * placement is ROW-INTERLEAVED -- x is [rows][row] and y is [rows][world_size][row] -- which
     * is what the vocab-parallel candidate merge needs, because it reads one row of gathered
     * candidates per sampled position at a single row stride and the concatenated form puts rank
     * 1's row a whole message away from rank 0's. */
    { "row",        RAD_P_INT, RAD_OPTIONAL },
};
/* Both REQUIRED (the third field is `optional`, not an index): unlike the all-reduce there is no
 * in-place form, because the output is world_size times the input and cannot alias it. */
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

static const RadParamSpec pGemmNtQ[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
/* docs/OPS.md's `gemm_nt_q` is `a`, `a_scale`, `b`w, `b_scale`w -> `y`. Two of libr4d's six
 * quantised GEMMs need a seventh operand each and neither is expressible as a weight aux stream,
 * because RadLayout can DECLARE aux streams beside a weight but RadArgs has no way to DELIVER
 * them to a launch:
 *
 *   `a_sum`  the per-row per-group sum of the int8 activation codes, which the ASYMMETRIC 4-bit
 *            and the 2-bit grids subtract to remove the stored zero point. It is produced by
 *            `quant_act_i8`'s third output, so it is activation-side and could never have been a
 *            weight aux in the first place.
 *   `b_ref`  MXFP4's per-output-row reference exponent, against which the E8M0 block scale is
 *            folded so the inner loop has no rescale. This one IS a weight aux and is exactly the
 *            case RadLayout.aux_tag was written for -- it just cannot be passed.
 *
 * Both are appended after `y` so OPS.md's list stays a positional prefix and ref is unaffected
 * until it implements a grid that needs them. That RadArgs cannot deliver a declared aux stream is
 * a genuine gap in the ABI, not a shortcut taken here. */
static const RadOperandSpec oGemmNtQ[] = {
    { "a",       RAD_OPD_IN,     0 },
    { "a_scale", RAD_OPD_IN,     1 },   /* absent for the f16-activation grids (w4a16, w8a16) */
    { "b",       RAD_OPD_WEIGHT, 0 },
    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "y",       RAD_OPD_OUT,    0 },
    { "a_sum",   RAD_OPD_IN,     1 },
    { "b_ref",   RAD_OPD_WEIGHT, 1 },
};

/* docs/OPS.md's `dequant` is activation-shaped (`M`(range) `n` `group` `dtype`). libr4d's
 * dequant_w4_bf16 unpacks a WEIGHT out of WMMA fragment order back to row-major bf16, for the
 * prefill path where W4A8 wins nothing. Its own op, with libr4d's own N/K keys, because calling a
 * weight [N,K] an activation [M,n] would put the fragment-order inverse under a name that a
 * sensible ref implementation would answer with a plain elementwise dequantiser.
 *
 * NOT ONE OF THE SIXTEEN SHARED OPS, whatever docs/OPS.md's resolved table says: `libref`
 * declares `dequant`, not `dequant_w4`, so nothing here is negotiated with anything and the union
 * rule leaves this schema as libr4d's own. The table's row also drops `twos`, which is load-
 * bearing -- the symmetric grid's two's-complement nibble and the asymmetric grid's unsigned
 * nibble minus a per-group zero differ by bit 3, and the wrong one is silently wrong -- and marks
 * the two weight operands as plain inputs, which would stop the core resolving their handles. The
 * schema here is the one the kernel needs; that table's row is not applied. */
static const RadParamSpec pDequantW4[] = {
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    /* 1 = the symmetric grid's two's-complement nibble, 0 = the asymmetric grid's unsigned nibble
     * minus a per-group zero. libr4d: "they differ by bit 3 and the wrong one is silently wrong". */
    { "twos",  RAD_P_INT, RAD_REQUIRED },
};
static const RadOperandSpec oDequantW4[] = {
    { "wq", RAD_OPD_WEIGHT, 0 },
    { "ws", RAD_OPD_WEIGHT, 0 },
    { "y",  RAD_OPD_OUT,    0 },
};

/* `vocab_off` and `pairs` are the VOCAB-PARALLEL form and both are absent on a single-rank graph,
 * which is what keeps that graph byte-identical to the unsplit form. With them, `x` is one rank's
 * SLICE of a wider plane, `idx` carries GLOBAL columns, and `pairs` is an [M, R, 2] plane of (int32
 * id, float value) -- the plane `core/sample/sampler.h` holds as `buf_pairs_` -- which an
 * all_gather moves so the ranks can agree on a global top-R without ever gathering the plane. */
/* ---- the routed mixture of experts ------------------------------------------------------------
 *
 * `moe_gemm_q` is a SEPARATE OP from docs/OPS.md's `moe_gemm`, for the same reason `gemm_nt_q` is
 * separate from `gemm_nt`: the operand list differs -- an activation scale stream and a weight
 * scale plane per expert -- and an op whose operand COUNT depends on a dtype string is an op whose
 * positional arguments mean different things to different kernels. libref declares the same two.
 *
 * Both weight operands are RAD_OPD_WTAB and both carry one entry per expert, which is what lets
 * the declaration divide evenly between them (core/build/rad_builder.cpp states that rule). */
static const RadParamSpec pScaleRows[] = {
    P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype"),
};
static const RadOperandSpec oScaleRows[] = {
    { "x", RAD_OPD_IN,  0 },
    { "s", RAD_OPD_IN,  0 },
    /* Present = the fused gate-and-add form, `y = bf16(x * act(s)) + add`. It exists because the
     * two ops are adjacent everywhere this is used -- a gated shared expert scales its output and
     * then adds the routed arm to it -- and on the MoE block that pair is two dispatches a layer
     * on ONE rank, which the other rank waits out inside the block's all-reduce. */
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
    /* THIS RANK'S FIRST EXPERT ID -- what makes expert parallelism a parameter and not a second
     * op. Absent or 0 is the whole-model case. */
    P_INTO("expert_base"),
    /* THE EXPERTS SORTED LAST: a comma list of local indices, ascending. A layer that keeps a few
     * experts at a higher precision serves them with a second grouped GEMM, and that GEMM takes a
     * contiguous run of the offsets -- so their rows go after every other expert's, and the
     * offsets and counts are in sort order. Absent is none. */
    P_STRO("protect"),
};
/* THE PAIR AS ONE OP. Its parameters are the union of the two -- nothing new is computed, so
 * nothing new is declared -- and its operands are their operands with the shared `expert_ids`
 * named once, as an OUTPUT, because the fused form produces it rather than being handed it. */
static const RadParamSpec pRouterTopkScatter[] = {
    P_INT("M"),
    /* THE ROUTER'S COUNT, which is every expert in the model: the router is replicated and each
     * rank picks from all of them. */
    P_INT("n_expert"),
    /* THIS RANK'S SLICE, which the SORT runs over. Absent means the slice is the router, which is
     * every deployment without expert parallelism. Under it the two counts differ, and one number
     * serving both would either rank half the experts or overrun the histogram. */
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
static const RadOperandSpec oMoeScatter[] = {
    { "expert_ids",    RAD_OPD_IN,  0 },
    { "sorted_tok",    RAD_OPD_OUT, 0 },
    { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count",  RAD_OPD_OUT, 0 },
};
/* `a_order` is "token" or "sorted" and says which rows the ACTIVATION has: one per token, to be
 * gathered through sorted_tok, or one per (token, slot) already in the scatter's order. The two
 * grouped GEMMs of one routed block disagree about it -- gate_up reads the block input, down reads
 * gate_up's output -- and it is a parameter rather than a row count because an operand whose
 * meaning changes with its extent gives no way to tell the two readings apart. */
/* THE EXPERTS ARE TWO TABLES, BY PARITY: `w`/`w_scale` hold the even-numbered experts and
 * `w_odd`/`w_odd_scale` the odd ones, expert e being entry e / 2 of its class. A table's entries
 * share one geometry, and a rank that holds an uneven slice of every expert's width -- three
 * 128-column blocks of one parity's experts and two of the other's, which is how a five-block
 * expert splits across two ranks evenly on average -- has two geometries. `N`/`K` are the even
 * class's, `N_odd`/`K_odd` the odd class's and default to them.
 *
 * `parts` is how many stacked parts the weight rows and the output columns both hold -- two for a
 * fused gate/up. The output is as wide as the wider class, and a narrower class fills the leading
 * columns of each output part and writes zeros after them. */
/* `moe_gemm`, the unquantised op, is the vocabulary's and libref fixes its schema; this is the same
 * schema, which the registry requires of any plugin that serves the op. ONE weight table: every
 * entry has one geometry, and nothing about an unscaled plane forces the parity split above. */
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
/* THE SAME GEMM OVER FOUR CLASSES, BY THE EXPERT'S NUMBER MOD 4: `w`/`w_scale` hold class 0 at
 * [N, K] and `w_q`/`w_q_scale` class q at [N_q, K_q], expert e being entry e / 4 of class e % 4.
 * Four ranks splitting a five-block expert cannot share the extra block evenly over two classes --
 * whichever two ranks take it carry three blocks of every even/odd pair against the others' two --
 * and over four they can: rank r takes two blocks of class r and one of each other, five of every
 * four experts. A separate op and not a parameter on `moe_gemm_q`, because the declaration divides
 * a table op's weights over its table operands by count alone (core/build/rad_builder.cpp): the
 * operand list is the class count. `y` follows the last class. */
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
    /* The shared arm's gate, and it is present exactly when `shared` and `shared_gate` are.
     * "sigmoid" or "none", applying to the SCALAR -- the same spelling and the same meaning
     * `scale_rows` gives it, because this is that op folded in. */
    P_STRO("act"),
};
static const RadOperandSpec oMoeGather[] = {
    { "y_expert",    RAD_OPD_IN,  0 },
    { "expert_w",    RAD_OPD_IN,  0 },
    { "sorted_tok",  RAD_OPD_IN,  0 },
    /* PRESENT = THE SHARED ARM FOLDED IN, `y = bf16(shared * act(gate)) + gathered`. It is the
     * `scale_rows` this op's output would otherwise be handed to, and the two narrows are
     * load-bearing: the unfused pair writes the gather to a bf16 buffer and the product to another
     * one before adding, so BOTH are rounded before the sum sees them.
     *
     * WHY IT IS WORTH A PAIR OF OPERANDS: the shared arm runs on ONE RANK, so the scale_rows it
     * feeds is a dispatch a layer that the other rank waits out inside the block's all-reduce --
     * and with it gone the routed arm's `moe` buffer is never written either. */
    { "shared",      RAD_OPD_IN,  1 },
    { "shared_gate", RAD_OPD_IN,  1 },
    { "y",           RAD_OPD_OUT, 0 },
};

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
/* The other half of that form: the k largest of the all-gathered per-rank candidate sets, in the
 * same (idx, val) shape the single-rank top-R produces, so `dflash_select` downstream cannot tell
 * which path it was handed. `R` is the width OUT; the input holds world_size * R of them. */
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
/* docs/OPS.md gives `x` -> `q`, `scale`. The third output selects libr4d's `_asum` form, which
 * additionally emits the [K/group][Mpad] group-major row sums the ASYMMETRIC weight grids
 * subtract. One row, not two: whether the caller wants the sums is a property of the call and not
 * of the model, so it belongs in the entry point exactly where spec §2.1 puts strides and
 * alignment. */
static const RadOperandSpec oQuantAct[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
    { "asum",  RAD_OPD_OUT, 1 },
};

/* `quant_act_fp8` is a NEW OP and this plugin fixes its schema. Shaped exactly on docs/OPS.md's
 * `quant_act_i8` -- same params, same first three operands -- because it is the same op at a
 * different width, and an architecture plugin that switches activation format should change one op
 * name and nothing else. There is no `asum`: that output exists for the asymmetric integer grids,
 * and an fp8 grid is symmetric by construction.
 *
 * libref implements it (ref_quant.cpp), so the whole block-scaled fp8 family has an oracle:
 * without a reference for this op the GEMM that consumes its output could only ever be checked
 * against activations libref quantised differently (spec §17). */
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

/* `gram_accum` is a NEW OP and this plugin fixes its schema. It is the calibration half of GPTQ:
 * `h += x^T x` over an E4M3 activation, accumulated across every launch of a corpus.
 *
 * `h` IS INOUT AND THAT IS THE WHOLE OP. A caller zeroes it once and then drives a corpus through
 * the graph; each launch adds its own block. Declaring it OUT would say the op overwrites, which
 * is the one reading that makes the result wrong without making it fail.
 *
 * The operand names are `a` and `a_scale` rather than `x` and `scale` because they are the same
 * pair `gemm_nt_q` takes in that order, and this op exists to characterise exactly the operand
 * that GEMM contracts.
 *
 * `a_scale` IS OPTIONAL because an unquantised GEMM contracts a plain activation: dtype "bf16"
 * reads `a` as it stands and takes no scale, which is the tap of a model served at bf16. */
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

/* `qsa_block_key` is a NEW OP and this plugin fixes its schema. It builds the QSA indexer's
 * compressed key: one vector per `ratio` tokens of key history, which is what the indexer's
 * scores are taken against.
 *
 * `M` IS BLOCKS AND `k` IS `ratio` TIMES TALLER. Every other op in this vocabulary has one output
 * row per input row; this one has one per four, and a caller that sized `k` at the block count
 * would have every block pool three rows of the next one -- plausible numbers, different model.
 *
 * `pos0` IS A PARAMETER AND NOT AN OPERAND because blocks TILE the token stream: block i's first
 * token is at `pos0 + i * ratio`, always. A caller with a discontiguous run issues it in runs,
 * which is cheaper than a position operand every launch would have to build. */
static const RadParamSpec pQsaBlockKey[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    { "ratio",      RAD_P_INT, RAD_REQUIRED },
    { "rotary_dim", RAD_P_INT, RAD_REQUIRED },
    { "pos0",       RAD_P_INT, RAD_OPTIONAL },   /* the tiling; a `pos` operand overrides it */
    /* THE SAME NUMBER AS `ratio`, AND IT IS HERE TO SIZE A KV GROUP. `Builder::find_kv_consumer`
     * binds a group to the first op declared after it that carries the literal key `block_size`,
     * so an op that wants to decide a group's page size has to spell it that way -- and the
     * block-key store's page IS the compress block. The shim refuses a disagreement rather than
     * picking one. */
    { "block_size", RAD_P_INT, RAD_OPTIONAL },
    P_F64("eps"),
    P_F64("theta"),
    P_F64O("scale"),
    P_F64O("wadd"),
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    /* THE LAYER'S ROTARY, when it has several position components: `mode` mrope or imrope and
     * its `sections`, read against a [3, M] component-major `pos` list. Absent is NeoX. */
    { "mode",       RAD_P_STR, RAD_OPTIONAL },
    { "sections",   RAD_P_STR, RAD_OPTIONAL },
};
static const RadOperandSpec oQsaBlockKey[] = {
    { "k",    RAD_OPD_IN,     0 },
    { "w",    RAD_OPD_WEIGHT, 0 },
    /* OPTIONAL, and its presence changes where the answer goes rather than what it is. The
     * block-key store is the PAGED KV group of docs/QSA.md, so block b of a sequence lives at
     * whatever page the manager gave it and consecutive blocks are not consecutive pages. Absent,
     * the write is contiguous -- a caller with its own scratch, or a test. Present, `M` comes off
     * THIS operand and not off `out`, because `out` is then as tall as the pool. */
    { "page", RAD_OPD_IN,     1 },
    /* OPTIONAL, and it is what turns a launch from a RANGE into a LIST. `pos0 + b*ratio` is the
     * tiling and serves a prefill chunk, where the completing blocks are a contiguous run of one
     * sequence. It does not serve a decode step: the sequences in a batch are at different
     * positions, so the blocks completing in one launch belong to different sequences and are not
     * a run. With this, one launch a layer a step serves any mix. */
    { "pos",  RAD_OPD_IN,     1 },
    { "out",  RAD_OPD_OUT,    0 },
};

/* `qsa_score` is a NEW OP and this plugin fixes its schema. It is the indexer's vote: how much
 * one query wants each compressed block of key history.
 *
 * `bt` IS A LIST OF PAGE IDS AND NOT BLOCK INDICES. The attention KV group is paged at the
 * compress ratio, so a block and a page are the same thing and the sequence's existing block
 * table is already the list this scores -- and the top-k over the output is already the block
 * table the attention reads. That identity is the design (docs/QSA.md); an op that took block
 * indices would put a translation on both sides of it. */
/* `qsa_select` is a NEW OP and this plugin fixes its schema. It turns the scorer's numbers into
 * the page-id table the attention reads.
 *
 * `nc` IS THE COMPLETE-BLOCK COUNT AND IS A SEPARATE OPERAND from the table's width, because the
 * two differ by exactly the thing the op has to treat specially: `bt[nc]`, when it is not -1, is
 * the PARTIAL page -- the tokens after the last complete block, which the reference attends
 * unconditionally and which the query itself lives in. It is appended after the selection and is
 * never a candidate for it.
 *
 * `sel` IS `topk + 1` WIDE for that reason and the last slot is not spare. */
/* `qsa_tail_store` is a NEW OP and this plugin fixes its schema. It parks the raw index keys of
 * the tokens that have not completed a block yet, so the next step can finish one.
 *
 * `cu` IS HOW A TOKEN FINDS ITS SEQUENCE, and it is the only operand in the batch that says so --
 * `slot_mapping` is a flat KV slot and carries no sequence. It is also where the sequence COUNT
 * comes from at run time, rather than from any declared height, which is max_seqs.
 *
 * WHICH SEQUENCE'S STATE IS `sidx`, AND THAT IS A DIFFERENT QUESTION. `cu` says which ROW of this
 * step a token is in; the row is step-local, so the keys have to be kept somewhere the engine
 * keeps per SEQUENCE. That is a linear KV group, and `state` is its cache for this layer. */
/* `qsa_work` is a NEW OP and this plugin fixes its schema. It is the gather that turns a batch
 * into the flat block list `qsa_block_key` reads.
 *
 * `M` IS TOKENS AND `work` IS BLOCKS, and they are different numbers: one block a `ratio` tokens
 * plus at most one straddling block a sequence. `work` is a HOST bound on how many blocks can
 * complete, not a count of how many did -- the count is device data, and slots past it get
 * `page = -1`, which is the sentinel `qsa_block_key` already skips. */
static const RadParamSpec pQsaWork[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "ratio", RAD_P_INT, RAD_REQUIRED },
    { "ring",  RAD_P_INT, RAD_REQUIRED },
    { "seqs",  RAD_P_INT, RAD_REQUIRED },
    /* A capacity: the launch is as wide as `page`, whatever this says (r4d_qsa_work_bf16.hip). */
    { "work",  RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQsaWork[] = {
    { "k",     RAD_OPD_IN,  0 },
    { "pos",   RAD_OPD_IN,  0 },
    { "cu",    RAD_OPD_IN,  0 },
    /* THE TAIL IS THE STATE GROUP'S CACHE FOR THIS LAYER and `sidx` names the sequence's slot
     * in it. Indexing it by the batch row instead would not survive a step: the row is
     * step-local, the sequence is not. */
    { "state", RAD_OPD_IN,  0 },
    { "sidx",  RAD_OPD_IN,  0 },
    { "bt",    RAD_OPD_IN,  0 },
    /* OPTIONAL: the step's rotary components, [3, T] (RadBatch::rope_pos). Present, `bpos` holds
     * each block's first-token components as [3, work] -- what an M-RoPE layer's block key
     * rotates at -- instead of its index. */
    { "rope",  RAD_OPD_IN,  1 },
    { "stage", RAD_OPD_OUT, 0 },
    { "page",  RAD_OPD_OUT, 0 },
    { "bpos",  RAD_OPD_OUT, 0 },
    /* COMPLETE BLOCKS A SEQUENCE, which the scorer and the selection both need and which this op
     * has already computed on its way to the work list. A separate op for one division a sequence
     * would be a launch a layer a step for nothing. */
    { "nc",    RAD_OPD_OUT, 0 },
};

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

static const RadParamSpec pQsaSelect[] = {
    { "M",            RAD_P_INT, RAD_REQUIRED },
    { "blocks",       RAD_P_INT, RAD_REQUIRED },
    { "topk",         RAD_P_INT, RAD_REQUIRED },
    { "ratio",        RAD_P_INT, RAD_REQUIRED },
    { "seqs",         RAD_P_INT, RAD_REQUIRED },
    { "tp_partition", RAD_P_INT, RAD_OPTIONAL },
    { "m_offset",     RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oQsaSelect[] = {
    { "score",   RAD_OPD_IN,  0 },
    { "bt",      RAD_OPD_IN,  0 },
    { "nc",      RAD_OPD_IN,  0 },
    { "pos",     RAD_OPD_IN,  0 },
    /* `M` IS QUERIES AND `bt`/`nc` ARE PER SEQUENCE, so something has to join them: at decode one
     * query a sequence made the two the same list, and at prefill a chunk's queries share one
     * history and each selects its own set out of it (docs/QSA.md, Stage B). `cu` is the only
     * operand in the batch that says which sequence a token belongs to. */
    { "cu",      RAD_OPD_IN,  0 },
    { "sel",     RAD_OPD_OUT, 0 },
    { "nsel",    RAD_OPD_OUT, 0 },
    /* `seqused` IS WHY QSA NEEDS NO NEW ATTENTION KERNEL. `attn_paged` takes its block table and
     * its per-sequence length as ORDINARY OPERANDS -- the architecture reads them off the KV group
     * batch and passes them -- so the sparse attention is the same op issued with `sel` in place
     * of the table and this in place of the length. Emitted here rather than by an op of its own
     * because it is one store off a number this kernel already has. */
    { "seqused", RAD_OPD_OUT, 0 },
};

static const RadParamSpec pQsaScore[] = {
    { "M",            RAD_P_INT, RAD_REQUIRED },
    { "n",            RAD_P_INT, RAD_REQUIRED },
    { "heads",        RAD_P_INT, RAD_REQUIRED },
    { "blocks",       RAD_P_INT, RAD_REQUIRED },
    { "seqs",         RAD_P_INT, RAD_REQUIRED },
    { "dtype",        RAD_P_STR, RAD_REQUIRED },
    { "tp_partition", RAD_P_INT, RAD_OPTIONAL },
    { "m_offset",     RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oQsaScore[] = {
    { "q",     RAD_OPD_IN,  0 },
    { "bk",    RAD_OPD_IN,  0 },
    { "bt",    RAD_OPD_IN,  0 },
    /* See oQsaSelect's note: `M` is QUERIES and the table is the SEQUENCE's. */
    { "cu",    RAD_OPD_IN,  0 },
    /* THE BLOCK LOOP'S BOUND, and it is required rather than derived. `blocks` is a DECLARED
     * width sized for max_ctx, so a scorer that trusted it would read 65536 padding entries a
     * query on a 200-token prompt. `nc` is the sequence's live count, which is an upper bound for
     * every query in the chunk; `qsa_select` narrows it to each query's own. */
    { "nc",    RAD_OPD_IN,  0 },
    { "score", RAD_OPD_OUT, 0 },
};

/* `rmsnorm_quant_fp8` and `gated_quant_fp8` are NEW OPS and this plugin fixes their schemas. Each
 * is its producer's operand list with the quantiser's two outputs appended and the producer's own
 * bf16 output demoted to OPTIONAL -- so an architecture that fuses changes an op name, keeps the
 * operands it already passed in the order it already passed them, and adds `q` and `scale`.
 *
 * THE RESIDUAL IS SECOND, which is where `rmsnorm_had_quant_i8` puts it, so the two ops differ by
 * the rotation and nothing else. It is what turns `add` + `rmsnorm` into one launch: a block adds
 * the PREVIOUS block's delta into the shared residual stream at the front of its own norm rather
 * than at the end of the block before it. Unlike the int8 sibling this one rounds the sum to bf16
 * and stores it BEFORE normalising, which is what the pair it replaces does and is what keeps the
 * output identical to the byte. */
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

/* The same op again with the all-reduce in front of it, which is `ar_ln_had_quant_i8`'s bargain
 * without the rotation. Same operands in the same order, plus `world_size` -- and no `wire`, which
 * is the one place this differs from the int8 sibling's schema: that kernel carries the rotated
 * 6-bit payload as well and this one serves the exact wire only, so a caller above
 * --tp-wire-min-kb emits `all_reduce` then `rmsnorm_quant_fp8` and gets the compressed message
 * from the standalone kernel. A parameter for a wire there is no code for would be a lie the
 * matcher could not catch. */
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
    { "x",        RAD_OPD_INOUT,  0 },   /* the message AND the input; the reduced sum lands here */
    { "residual", RAD_OPD_INOUT,  1 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
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
/* THE GATE IN A SEPARATE OPERAND, which is the shape a gated attention has: the gate is a column
 * slice of the [q|gate] projection and `x` is the attention output, two buffers with different
 * pitches. `M` is a (token, head) pair rather than a token -- see the kernel -- and that is what
 * makes both sources a fixed stride apart and leaves the scale plane exactly where the out
 * projection reads it. */
/* THE gate_up GEMM WITH gated_quant_fp8'S PASS IN ITS EPILOGUE, and a schema of its own rather
 * than a flag on `gemm_nt_q`, because it does not write a `y`: its outputs are the quantiser's
 * and an architecture that resolved it into a `gemm_nt_q` operand list would hand the GEMM a
 * buffer of the wrong width. `N` is the WEIGHT's rows -- 2 * the fused width, gate then up, the
 * same plane the unfused GEMM reads -- so a caller that has both declared spells the geometry
 * once. `act` is `gated_quant_fp8`'s and is REQUIRED here: the fold is compiled for silu only,
 * and an op that took the parameter optionally would resolve for a sigmoid it cannot serve. */
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

/* `sample_chain` is a NEW OP, and it is NOT docs/OPS.md's eight-op sampler chain
 * (`sample_penalties`, `sample_temp`, `sample_topk`, `sample_topp`, `sample_minp`, `sample_mask`,
 * `sample_pick`, `sample_argmax`). Those are eight launches over a 248K-entry row per position and
 * nothing in this project implements any of them on a device. libr4d's is ONE kernel that walks the
 * vocabulary several times inside one workgroup, which after the first pass is L2-resident.
 * Splitting it into the eight would throw that away, and fusing back is not something the core
 * does: "fusion is a selection, not a compiler pass" (spec §2.3), so the plugin declares the fused
 * op and a caller asks for it first.
 *
 * top_k, top_p, min_p and temperature are per-REQUEST and change between steps, so they are
 * parameters read at LAUNCH and never constraints -- putting a resolution boundary on a sampling
 * setting would rebuild the bucket table when a user moved a slider. The float ones are RAD_P_F64
 * per docs/OPS.md; constraints never match on a float.
 *
 * THE SPECULATIVE OPERANDS ARE WHY IT IS WORTH ONE OP RATHER THAN EIGHT: `query` is the drafted
 * token per position and `u` two uniforms per position, and the three outputs are a draw, a draw
 * with `query` excluded and renormalised, and `query`'s filtered probability -- everything a
 * rejection-sampling verify needs from one pass, so the host never comes back (spec §10).
 *
 * libref does not implement this either, and here the gap is real work rather than fifteen
 * lines. Its oracle is llama.cpp's host sampler, which spec §13 already requires rad-kbench to keep. */
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
    { "u",      RAD_OPD_IN,  0 },   /* TWO uniforms per position */
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
     * Forming that in bf16 first costs 0.3% per channel, so libr4d adds it in fp32 after the
     * load and takes the addend as an argument. 0 is a plain RMS norm. */
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

/* `act` is the vocabulary's name for the gate and it is a NAME: "silu" is down_proj's input,
 * "sigmoid" is a gated-attention o_proj's. `mode` is libr4d's integer spelling of the same choice
 * and survives as OPTIONAL because it is libr4d's, not the vocabulary's; a caller written against
 * either name works, and `act` wins where both are given. */
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

/* Not in docs/OPS.md. A fusion with no unfused equivalent in the vocabulary, exactly like
 * `rmsnorm_had_quant_i8`: the GDN gated RMS norm is per HEAD while the rotation and the int8 scale
 * are per TOKEN, so the reduction is segmented and nothing in the vocabulary composes to it. */
static const RadParamSpec pGdnGatedNormHadQuant[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED },
    { "n",        RAD_P_INT, RAD_REQUIRED },
    { "head_dim", RAD_P_INT, RAD_REQUIRED },
    { "group",    RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "dtype",    RAD_P_STR, RAD_REQUIRED },
    /* Z may be the INTERLEAVED qkvz projection rather than a compacted gate: element i of the
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

/* Not in docs/OPS.md, and spec §2.3 names it as the example of a fused op a plugin asks for
 * first. Every row-parallel linear's output is all-reduced and then immediately normalised -- 128
 * adjacent dispatches on a decode step -- and fused, the all-reduce's bf16 output never exists.
 * The unfused emission a plugin falls back to is `all_reduce` then `rmsnorm_had_quant_i8`. */
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
 * TRANSCRIBED FROM libref, EXACTLY. The first plugin in hierarchy order to declare an op
 * fixes its schema and a later plugin that disagrees is refused at load with both plugins named
 * (core/plugin/loader.cpp) -- and rightly, since arguments are positional and two schemas would
 * make the same call mean different things depending on which plugin won. libr4d loads FIRST, so
 * these are the fixed forms and ref's have to match: they agree with libref/ref_registry.cpp field
 * for field, and a change to one belongs in both files at once.
 *
 * These ops have no interesting geometry -- there is nothing to specialise an add on -- so the
 * constraint rows below say only what the kernels actually require: bf16 operands, and for
 * kv_store a block size at all. That is a NECESSARY condition and not a sufficient one: the shims
 * still check every stride, pitch and extent per call. */
static const RadParamSpec pRmsNorm[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    P_F64("eps"),
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    P_F64O("wadd"),
};
static const RadOperandSpec oRmsNorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT,    0 },
};

static const RadParamSpec pMnDtype[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oBinary[] = {
    { "a", RAD_OPD_IN,  0 },
    { "b", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oGateUp[] = {
    { "gate_up", RAD_OPD_IN,  0 },
    { "y",       RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oUnary[] = {
    { "x", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};
/* `cast`'s two dtype keys are STRINGS and not one `dtype`, because the whole point of the op is
 * that the two ends differ. ref's spelling, matched field for field so the two plugins agree. */
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
    /* THE DESCALES THE READER WILL MULTIPLY BACK BY, so the store divides. Meaningless for a bf16
     * cache and optional for that reason; an fp8 one defaults them to 1.0, which is vLLM's default
     * and what the attention kernels assume when their descale operands are absent. They are ONE
     * number per (sequence, head) on the read side, so they cannot be derived from the tokens
     * being written -- see r4d.h on r4d_kv_store_fp8. */
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
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },   /* THIS RANK'S rows when vocab_offset is given */
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
    { "vocab_offset", RAD_P_INT, RAD_OPTIONAL },  /* first GLOBAL row; vocab-parallel embedding */
    /* The scale embed_lookup_q's layout quantises with. That op shares this parameter list and
     * libref fixes it for both, so it is spelled here to match; embed_lookup reads no scale. */
    { "wscale",  RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oEmbed[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "wte",    RAD_OPD_WEIGHT, 0 },
    { "x",      RAD_OPD_OUT,    0 },
};
/* The quantised gather's: libref's list (ref_registry.cpp), which the loader holds every plugin
 * serving the op to -- a one-element `scale`, absent for a table stored plain, and `ahead`, the
 * next prompt's ids for a host reader that fetches rows from a file. */
static const RadOperandSpec oEmbedQ[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "wte",    RAD_OPD_WEIGHT, 0 },
    { "scale",  RAD_OPD_WEIGHT, 1 },
    { "x",      RAD_OPD_OUT,    0 },
    { "ahead",  RAD_OPD_IN,     1 },
};
/* PLE's hashed n-gram ids. `state` is a rolling window of committed ids -- RAD_KV_CONV's shape and
 * ple_conv's contract -- except that a cold window is EOS and not zero, because zero is a real
 * token id. */
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
/* Four optionals, and they are the union of the two contracts gdn_conv_prep and gdn_conv_update
 * split between them: `has_init` is prefill's question and `num_accepted` is decode's. */
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

/* EVERY SAMPLER SCALAR IS PER ROW and arrives in `params`, an array of RadSampleParams
 * (abi/rad_sample.h). A parameter is geometry -- the selector matches on it and freezes it
 * at declare -- so `top_p` as a parameter would make two requests with different top_p two
 * different resolutions of one op, and a deployment would have exactly one top_p. Only the two
 * extents are parameters. */
static const RadParamSpec pMVocab[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
};
static const RadParamSpec pMCand[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "n_cand", RAD_P_INT, RAD_REQUIRED },
};
/* The history stages read GLOBAL token ids and write this rank's columns, so under a split
 * vocabulary they take the rank's first row -- sample_topk's `vocab_off`, for the same crossing in
 * the other direction. Absent is 0. */
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
/* `breakers` is optional: a u8 plane the shape of `history`, nonzero where that position's token
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
/* `pairs` is the VOCAB-PARALLEL path and is optional: [M, 2 * n_cand] u32 holding this rank's
 * candidates as (u32 GLOBAL id, f32 logit) -- what sample_merge_topk reads once the all-gather has
 * put every rank's side by side. `vocab_off` is added by the kernel, so the gather never has to
 * know the shard layout. Absent, and this is the single-rank top-k. */
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

#define SCHEMA(op, p, o, doc) { (op), ROW(p), ROW(o), (doc) }

static const RadOpSchema kSchemas[] = {
    SCHEMA("rmsnorm", pRmsNorm, oRmsNorm,
           "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). `wadd` is GemmaRMSNorm's `1 +`, added in "
           "f32 after the load because forming it in bf16 first costs 0.3% per channel."),
    SCHEMA("add", pMnDtype, oBinary,
           "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
           "along rows, and nothing else broadcasts at all."),
    SCHEMA("mul", pMnDtype, oBinary, "y = a * b, with the same one broadcast rule as `add`."),
    SCHEMA("silu_mul", pMnDtype, oGateUp,
           "y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The GATE IS THE FIRST HALF, "
           "which is the order rad-convert fuses gate_proj and up_proj in."),
    SCHEMA("sigmoid", pMnDtype, oUnary, "y = 1 / (1 + exp(-x))."),
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
    SCHEMA("gemm_nt_bias", pGemmBias, oGemmBias,
           "y = res + act(a @ b^T + bias), rounded to bf16 after the biased product, after the "
           "activation and after the residual -- the unfused chain's three tensors."),
    SCHEMA("kv_store", pKvStore, oKvStore,
           "scatter k and v into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim] -- K at "
           "[.., d] and V at [.., head_dim + d]. A negative slot stores nothing."),
    SCHEMA("embed_lookup", pEmbed, oEmbed,
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
    SCHEMA("attn_paged_gate_quant", pAttnPagedGq, oAttnPagedGq,
           "attn_paged followed by gate_quant_fp8 over its output, in the split-KV merge: each "
           "merged (token, head) row is gated, rounded, written to `out` and quantised to E4M3 "
           "with one scale per 128 columns into `q8` / `q8_scale`."),
    SCHEMA("attn_dense", pAttnDense, oAttnDense,
           "dense non-causal attention over cu_seqlens segments -- the vision tower's."),
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
    SCHEMA("hc_read", pHcRead, oHcRead,
           "the read half of a gated residual: a grouped rmsnorm, a low-rank sigmoid gate, and "
           "the mean of the gated streams -- plus the per-branch write gains and, optionally, "
           "the block-scaled fp8 form of the block input and its Hadamard-rotated twin."),
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
    SCHEMA("fulldrafter_two_stage", pFulldrafterTwoStage, oFulldrafterTwoStage,
           "FullDrafter discrete two-stage block drafter (K=12, 2x6) with Markov correction and Top-16 Selector."),
    SCHEMA("all_reduce", pAllReduce, oAllReduce,
           "sum over the tensor-parallel group. `exact` separates the exact sum from the "
           "quantised wire and a caller that does not declare it gets no kernel. At 4 and 8 ranks "
           "the exact sum is one-shot up to " R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements and "
           "two-shot above, chosen by the "
           "message's band; `hops` is accepted and no row of this plugin reads it."),
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
    SCHEMA("dequant_w4", pDequantW4, oDequantW4,
           "a packed 4-bit fragment-ordered weight back to row-major bf16."),
    SCHEMA("scale_rows", pScaleRows, oScaleRows,
           "y[m, i] = x[m, i] * act(s[m]): one scalar per ROW, broadcast ACROSS the row. `mul` "
           "broadcasts the other way -- one row down the rows -- and one op with both rules would "
           "be ambiguous at rows == n. `act` is \"sigmoid\" or \"none\" and applies to the SCALAR. "
           "With `add` present the form is y = bf16(x * act(s)) + add: the product is NARROWED "
           "before the sum, which is what makes the fused form bit-identical to the two ops it "
           "stands in for rather than merely close to them."),
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
    SCHEMA("gated_quant_fp8", pGatedQuantFp8, oGatedQuantFp8,
           "silu(a)*b or sigmoid(a)*b, then the same quantiser, in one pass."),
    SCHEMA("gated_had_quant_fp8", pGatedQuantFp8, oGatedQuantFp8,
           "the same, with a `group`-wide Hadamard between the product and the absmax. `out_bf16` "
           "still receives the UNROTATED product -- it exists for a bf16 consumer of the gate, and "
           "the rotation is a property of the quantised pair alone. Pairs with a weight stored at "
           "dtype w4a8h; substituting it for gated_quant_fp8, or the reverse, produces a model "
           "that runs and is wrong."),
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
           "break to the LOWER block index through a prefix count, never through atomic order."),
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

    SCHEMA("ar_rmsnorm_quant_fp8", pArRmsNormQuantFp8, oArRmsNormQuantFp8,
           "the two-rank all-reduce and the residual add, RMS norm and block-scaled fp8 "
           "quantisation that always follow it, in one kernel. Both wires."),
    SCHEMA("ar_hc_write", pArHcWrite, oArHcWrite,
           "the all-reduce and the gated residual's write that follows it, in one kernel. Both "
           "wires at two ranks, the exact one at four."),
    SCHEMA("ar_gather_hc_write", pArGatherHcWrite, oArGatherHcWrite,
           "moe_gather, then ar_hc_write, in one kernel: the routed experts' rows summed back to "
           "token order (with the shared arm folded when present) ARE the all-reduce's message, "
           "and the gated residual's write follows the reduction. Both wires at two ranks, the "
           "exact one at four."),
};

/* ================================================================================ constraints
 *
 * Each row restates its entry point's own admissibility rules, line for line where the keys agree;
 * the three key renames are marked where they happen.
 */

/* ---- attention ------------------------------------------------------------------------------
 * `causal` is a constraint, not a note: the paged kernels are causal by construction and the
 * vision kernel is dense, and neither can serve the other's masking. */
static const RadConstraint cAttnPrefillFp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillBf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* q_len <= 10 is the q_len*gqa <= 64 of the prose, at the gqa 6 this kernel is compiled for. In
 * radiance this doubles as the bucket boundary that splits `attn_paged` into a decode band and a
 * prefill band (spec §2.2) -- the boundary is the constraint value and nobody chose a policy. */
static const RadConstraint cAttnDecodeFp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeBf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The DFlash2 drafter: head_dim 128, 4 queries per kv head, and a sliding window -- which this
 * kernel supports and the h256 pair above does not, so `window` is a constraint on both sides
 * rather than an option one of them ignores. */
static const RadConstraint cAttnDecodeH128Fp8[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH128Bf16[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* EIGHT QUERIES PER KV HEAD, at both head widths (r4d_attn_paged_gqa8.hip). q_len <= 8 is the
 * `q_len * gqa <= 64` of the prose at gqa 8, and the difference from the gqa6 leg's 10 is the whole
 * of what the ratio changes: a decode workgroup holds 64 rows and the verify width is how many of
 * them a query position claims. As on the gqa6 leg, that bound doubles as the bucket boundary that
 * splits `attn_paged` into a decode band and a prefill band (spec §2.2). */
static const RadConstraint cAttnPrefillH256G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH256G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* The same two, at the other head width and for the same reason: one kernel, one runtime field. */
static const RadConstraint cAttnDecodeH256G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH256G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* TWELVE QUERIES PER KV HEAD, head_dim 256 (r4d_attn_paged_h256_gqa12.hip) -- Qwen3.8-Flash-Next's
 * QSA layers, 24 query heads over 2. `q_len <= 5` is the same `q_len * gqa <= 64` bound the two
 * ratios above state, and it is TIGHTER than the legal speculation depths: a verify of depth 7 is
 * q_len 8, which 12 rows a position does not fit, so that band takes the PREFILL kernel -- which
 * serves any q_len and is causal, which a verify is. Nothing is missing; the bucket boundary just
 * falls in a different place at this ratio. */
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
/* SIX QUERIES PER KV HEAD, head_dim 256, paged block 4 (r4d_attn_paged_h256_gqa6_bs4.hip) -- the
 * SAME QSA layers as the gqa12 leg above, at four ranks. The model has two KV heads; a world of
 * four replicates each across a rank pair, so a rank declares 6 queries over the 1 kv head it
 * holds and `gqa` is 6 where the TP2 leg reads 12. What the ratio changes is the two bounds, as
 * ever: `q_len <= 10` is the same `q_len * gqa <= 64` at 6 queries a position rather than 12, so
 * this leg's bucket boundary sits where the plain h256/gqa6 leg's does, and a draft block of six or
 * seven positions -- which gqa12 hands to the prefill kernel -- is a decode launch here. The page
 * stays 4 because it is the model's indexer's block size, not a consequence of the parallelism.
 *
 * THE `attn_paged_gate_quant` ROWS ARE HERE TOO (see attn_decode_gq_h256_gqa6_fp8kv). Registering
 * them costs nothing the plain legs don't already pay -- the body instantiates the `_gq` entry
 * points at EVERY ratio, so this leg needs a row and not a kernel -- and NOT registering them is
 * not neutral: the block declares the fused op beside its unfused pair either way, so an
 * unregistered geometry is a device band with nothing behind it, reported as a hole a request can
 * fall into. The unfused pair still runs whenever the fused row does not resolve, unchanged. */
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
/* The gate and the quantiser as the merge's epilogue, at the four-rank ratio: the gqa12 pair's
 * constraints with the ratio and the verify width moved together (`q_len <= 10`, not `<= 5`). */
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
static const RadConstraint cAttnPrefillH128G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnPrefillH128G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* NON-CAUSAL IS SERVED AND THE WINDOW IS NOT, and the two halves of that come from different
 * places -- which is why one loosens here and the other must not.
 *
 * `causal` IS A RUNTIME FIELD of R4DArgs that the shared decode kernel already reads: at cz == 0
 * it bounds the key by ctx - 1 instead of by the row's own position, which is exactly a
 * block-diffusion drafter's own block attending to the whole context AND to itself
 * (r4d_attn_decode_h256_gqa6.hip, klimit). Nothing is compiled out and nothing is added. The gqa4
 * leg states C_GE("causal", 0) over the same kernel and these two do as well, because DSpark
 * issues a non-causal gqa8 decode -- 16 heads over 2 kv at head_dim 128, non-causal over 7 noise
 * rows.
 *
 * `window` STAYS C_EQ 0, because that one IS a template parameter: this dispatch instantiates the
 * kernel at WIN = 0 and a windowed geometry has no bound, no clamp and no second mask term in the
 * code that would run. Loosening it would resolve a kernel that silently attends to everything. */
static const RadConstraint cAttnDecodeH128G8Fp8[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH128G8Bf16[] = {
    C_EQ("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 8), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 8), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
/* KEY RENAME. libr4d writes C_IN("q_dtype","bf16") and C_IN("kv_dtype","bf16"); docs/OPS.md's
 * `attn_dense` has one `dtype` because nothing about a dense bf16 tower is mixed-precision. The
 * predicate -- "bf16, and nothing else" -- is unchanged. */
static const RadConstraint cAttnVit[] = {
    C_EQ("head_dim", 72), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16"),
};

/* `cast` at bf16 -> bf16, which is a strided ROW COPY and is why this row exists. An architecture
 * that taps an intermediate residual stream -- DFlash2 conditions its drafter on the trunk's
 * hidden states at five layers -- has to move rows out of a buffer the next layer overwrites, and
 * the vocabulary already has the op for it. What this row adds is a DEVICE arm: without one `cast`
 * resolves to ref alone, so a model using it runs the copy on the host and the reference gate
 * refuses the serve. libr4d's 2D copy is exactly the kernel, pitches and all.
 *
 * The predicate is deliberately from == to == bf16. A widening or narrowing cast is a different
 * kernel and this one would reinterpret the bytes rather than convert them. */
static const RadConstraint cCastBf16[] = {
    C_IN("from", "bf16"), C_IN("to", "bf16"),
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
/* q_len <= 8 IS THE REGISTER WINDOW: the fold preloads a sequence's tokens into registers and
 * indexes them at compile time, as the standalone update does. The entry point checks the conv
 * cache's own depth against the same bound. */
static const RadConstraint cGdnConvRecurrentUpdate[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("conv_width", 4), C_GE("q_len", 1),
    C_LE("q_len", 16),
};
static const RadConstraint cGdnGatedRmsNorm[] = {
    C_EQ("channels", 128),
};

/* ---- all-reduce ------------------------------------------------------------------------------
 * `exact` separates the two, and a caller has to say which it wants: a request that does not
 * declare it will accept a lossy wire format gets no kernel rather than the quantised one.
 * MESSAGE SIZE separates the one-shot from the two-shot at a width that has both. The two-shot
 * rows start above R4D_AR_WIDE_ONESHOT_MAX_ELEMS, which makes that bound a band edge of `numel`:
 * the band at or under it resolves only the one-shot, and in the band above both rows match and
 * R4D_PRIO_AR_TWOSHOT > R4D_PRIO_AR_ONESHOT hands it the two-shot. The GE is one past the bound so
 * the band under it ends ON the bound, a multiple of every DIV here (see pAllReduce's `numel`). The
 * choice is the selector's and so it is what the band report names; no entry point substitutes
 * one kernel for another. The ti8 wire needs no such key -- exact=0 already separates it, and it
 * serves its own small messages exactly. */
/* ---- all-gather -------------------------------------------------------------------------------
 * The vocab-parallel sampler's candidate merge and QSA's query-row shards. No
 * `exact` key: a gather MOVES its payload and cannot be lossy, so there is no second row for it
 * to be told apart from. dtype is the element SIZE here -- the pairs it carries are half u32 --
 * and BOTH SPELLINGS of each type are listed because the two callers disagree: the arch blocks
 * declare `bf16`/`fp32` and core/sample/sampler.cpp declares `f32`. r4d_ar_dtype_code already
 * accepts either, so a constraint that took only one would refuse the sampler by a synonym. */
static const RadConstraint cArGather2[] = {
    C_EQ("world_size", 2), C_IN("dtype", "bf16 fp16 f16 fp32 f32 i32 u32"),
    C_P2P,
};
/* The same gather at a width whose handshake is the wide family's tree and not one peer flag. The
 * dtype set is cArGather2's verbatim for the same reason -- the two callers' synonyms. */
static const RadConstraint cArGather4[] = {
    C_EQ("world_size", 4), C_IN("dtype", "bf16 fp16 f16 fp32 f32 i32 u32"),
    C_P2P,
};
static const RadConstraint cArGather8[] = {
    C_EQ("world_size", 8), C_IN("dtype", "bf16 fp16 f16 fp32 f32 i32 u32"),
    C_P2P,
};

static const RadConstraint cArExact[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_P2P,
};
static const RadConstraint cArGather3[] = {
    C_EQ("world_size", 3), C_IN("dtype", "bf16 fp16 f16 fp32 f32 i32 u32"),
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
static const RadConstraint cArTsExact4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R4D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R4D_AR_TWOSHOT_WIDE_MAX_ELEMS),
    C_P2P,
};
static const RadConstraint cArTsExact8[] = {
    C_EQ("world_size", 8), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R4D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R4D_AR_TWOSHOT_WIDE_MAX_ELEMS),
    C_P2P,
};
static const RadConstraint cArWht6[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
    C_P2P,
};
static const RadConstraint cArTi8Ts4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 128),
    C_P2P,
};

/* ---- GEMM ------------------------------------------------------------------------------------
 * DFlash2's grouped dynamic depthwise convolution. block_size is not listed as a predicate: the
 * body masks the position with block_size-1, so only powers of two are admissible, and that is a
 * property of the value rather than a bound a constraint can express. The entry point rejects it. */
static const RadConstraint cDflashConv[] = {
    C_EQ("taps", 2), C_EQ("group_size", 16), C_DIV("hidden_size", 64), C_IN("dtype", "bf16"),
};
static const RadConstraint cDflashSelect[] = {
    C_GE("steps", 1), C_GE("top_k", 1), C_LE("top_k", 32),
    C_DIV("rank", 32), C_LE("rank", 512), C_IN("dtype", "bf16"),
};
static const RadConstraint cFulldrafterTwoStage[] = {
    C_IN("dtype", "bf16"),
};

/* ---- the routed mixture of experts ----------------------------------------------------------
 *
 * The router's `n_expert` bound is a real limit rather than a tuning choice: the selection keeps a
 * lane's share of a row's logits in registers, and the LDS histogram the scatter's small form
 * keeps is one word an expert. Both are sized at r4d_moe.hip's kMaxExpert -- see the note on that
 * constant below -- and a wider model wants a different distribution, which would be a different
 * kernel rather than a larger constant.
 *
 * The GEMM's N%64 is the block's N tile and K%128 is the fp8 scale block. `group` has to BE that
 * block: a checkpoint quantised at any other edge is not one these kernels read, and saying so as
 * a predicate is what makes the selector decline it rather than the launch. */
static const RadConstraint cScaleRows[] = {
    C_GE("M", 1), C_GE("n", 1), C_IN("dtype", "bf16"),
};
/* 512 IS r4d_moe.hip's kMaxExpert and it is a real limit of those kernels: the router keeps a
 * lane's share of the row's logits in registers (16 a lane at 512, which is a separate
 * instantiation) and both scatter forms keep an n_expert + 1 histogram in LDS. Qwen3.8-Flash-Next
 * routes 10 of 512, which is the widest geometry the bound serves. */
/* THE GATED RESIDUAL. `n <= 7552` and `hc <= 4` together bound hc*n at 30208, inside the 30576
 * elements of bf16 that kernel A can hold the normed row in once its own 4384 bytes of static
 * planes are out of the 64 KiB (kHcLdsMaxHN) -- claiming a wider stream would be claiming a band
 * the shared allocation cannot serve, and there is no fallback under these rows to catch it
 * (libref is HOST domain, so it is the oracle and not a second implementation). 7552 is the
 * widest multiple of 128 under that bound at hc = 4. `n` divisible by 128 is the workgroup's
 * channel tile AND the fp8 scale group, which are deliberately the same number so that a workgroup
 * owns whole groups. */
static const RadConstraint cHcRead[] = {
    C_GE("M", 1), C_DIV("n", 128), C_LE("n", 7552), C_GE("hc", 2), C_LE("hc", 4),
    C_GE("lowrank", 1), C_LE("group", 128), C_LE("inject", 1), C_IN("dtype", "bf16"),
    C_IN("mix", "bf16"),
};
/* The mixing matrices as E4M3 rows (r4d_hc_fp8.h). `n` a multiple of 512 keeps hc*n a whole number
 * of the down dot's 512-element steps at any hc; `lowrank` a multiple of 64 up to 512 makes the up
 * row's quarter -- its scale group -- one to eight sixteen-code loads. */
static const RadConstraint cHcReadE4m3[] = {
    C_GE("M", 1), C_DIV("n", 512), C_LE("n", 7552), C_GE("hc", 2), C_LE("hc", 4),
    C_DIV("lowrank", 64), C_LE("lowrank", 512), C_LE("group", 128), C_LE("inject", 1),
    C_IN("dtype", "bf16"), C_IN("mix", "e4m3"),
};
static const RadConstraint cHcPlain[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};
/* `n` is bounded by LDS: nh is hc*n and ne is n more, both bf16, and both live for the whole of
 * the second phase. 28672 elements is that budget at hc 4, which 2560 sits well inside. */
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
static const RadConstraint cRouterTopK[] = {
    C_GE("M", 1), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
    C_LE("top_k", 32), C_IN("dtype", "bf16"),
};
static const RadConstraint cRouterTopkScatter[] = {
    C_GE("M", 1), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
    C_LE("top_k", 32), C_IN("dtype", "bf16"),
};
static const RadConstraint cMoeScatter[] = {
    C_GE("M", 1), C_GE("n_expert", 1), C_LE("n_expert", 512), C_GE("top_k", 1),
};
static const RadConstraint cMoeGemmQ[] = {
    C_GE("M", 1), C_DIV("N", 64), C_DIV("K", 128), C_GE("n_expert", 1), C_LE("n_expert", 65536),
    C_GE("top_k", 1), C_EQ("group", 128), C_IN("dtype", "fp8a8"),
};
/* The same op at a FOUR-BIT expert weight. Identical band, identical operands -- only the stored
 * weight plane and its scale grid differ, which is why the two rows share a schema and are told
 * apart by the dtype string alone.
 *
 * `w4a8h` IS THE SAME ROW AND THE SAME LAUNCH. It says the stored weight was rotated by a 128-wide
 * Hadamard along K before it was quantised, and the residual factor was folded into the stored
 * scale -- so the kernel still just multiplies codes by scales and there is nothing here to
 * branch on. What differs is the LAYOUT TAG the hooks return, which is what stops a container
 * converted one way from being served the other, and the activation the graph must pair with it
 * (`had_quant_act_fp8`, not `quant_act_fp8`). A second row would be the same entry point, the same
 * band and the same operands under a second name.
 *
 * `w4nla8h` IS THE ROTATED PLANE WITH ANOTHER CODE TABLE: the codes index libquant's w4nl levels
 * instead of -8..7, and the launcher hands the kernel that table (r4d_args.h). Same bytes a code,
 * same scale grid, same activation; the layout hook checks the container's table against the
 * kernel's before it serves it.
 *
 * `w4nl64a8h` IS w4nla8h WITH A SCALE A 64 OF K: one E4M3 byte under the fixed kW4G64Scale, so the
 * same bytes as a bf16 scale a 128; `w4nl32a8h` one a 32, twice those bytes. `group` stays 128 --
 * the staged K block and the rotation -- and the kernel sums each scale group of it apart.
 *
 * `w5nl64a8h` IS w4nl64a8h AT FIVE BITS: the codes index libquant's 32-level w5nl table, whose
 * high bit is the sign, so the kernel expands the low four as a w4nl code over the sixteen
 * magnitudes and sets the sign from a bit plane stored after each row's nibbles. */
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

static const RadConstraint cRowTopK[] = {
    C_DIV("N", 8), C_GE("M", 1), C_GE("R", 1), C_LE("R", 1024), C_IN("dtype", "bf16"),
};
/* R <= 32 is stage 2's PT, and the merge carries the same one for the same reason: a lane holds
 * the whole answer, so no distribution of the candidates over lanes can lose one. */
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
/* NO 64-ROW CAP: grid.y covers any row count this kernel is given, so the band is the grid's reach
 * and not a launcher guard. A cap here would force a caller with a taller projection -- the gated
 * delta net's a|b, for one -- to slice it. */
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
 * THE `group` PREDICATE IS THIS TABLE'S OWN, and the only one on this family that has no
 * kernel-side counterpart: the kernel takes no `group` argument on a GEMM -- its quantised GEMMs
 * read the group off a build flag -- but docs/OPS.md's `gemm_nt_q` schema carries one and this
 * family's block is a property of the checkpoint on disk rather than of the build. A caller
 * declaring a different group is asking for a format these kernels do not read, and the right
 * answer is no kernel rather than a silent 128.
 *
 * K's divisor differs between the two fp8a8 rows and is not cosmetic: the narrow instantiation
 * stages a whole 256-byte cacheline a row, and a K that is only a multiple of 128 makes a split
 * slice run one tile past its own end. The entry point refuses it. */
static const RadConstraint cGemmNtFp8A16[] = {
    C_LE("M", 8), C_DIV("K", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a16"),
};
/* K DIV 128, NOT 256, AND THE DECLARED NUMBER IS THE ONE THAT DECIDES. The entry serves a K that
 * is a multiple of 128 by staging a 128-byte tile instead of a 256-byte one (see narrow_fk), and
 * relaxing that runtime guard alone achieves nothing: the resolver offers a call to a kernel by
 * its DECLARED constraints, so a row claiming 256 never sees the shape at all. That is the
 * band-coverage hazard read backwards -- a kernel that does not claim a band it can serve is as
 * invisible as one that claims a band it cannot. */
static const RadConstraint cGemmNtFp8A8M16[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a8"),
};
static const RadConstraint cGemmNtFp8A8Tiled[] = {
    C_DIV("K", R4D_FP8_BLOCK), C_DIV("N", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "fp8a8"),
};
/* The int8 twins: the same bands at dtype i8a8. */
static const RadConstraint cGemmNtI8A8M16[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "i8a8"),
};
static const RadConstraint cGemmNtI8A8Tiled[] = {
    C_DIV("K", R4D_FP8_BLOCK), C_DIV("N", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK),
    C_IN("dtype", "i8a8"),
};
/* libr4d's row says `K DIV 4`; the key rename to docs/OPS.md's `n` is the same one every quantiser
 * row here makes. */
static const RadConstraint cQuantActFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* The rotated form wants n DIVISIBLE BY THE GROUP, not merely by 4. A rotation block cannot
 * straddle the end of a row: the ragged tail the unrotated kernel writes zero-padded would mix
 * real values with padding inside one butterfly, and there is no weight block on the other side
 * that was rotated the same way. */
static const RadConstraint cHadQuantActFp8[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};

/* The dtype is the INPUT's, and the input is already quantised -- so it is `fp8a8` and not `bf16`
 * the way every other row in this family reads it. A four-multiple `n` is the staging slot (one
 * dword of four codes); the 64-wide C tile needs `n` to be nothing in particular, because a ragged
 * tile masks. */
/* The head dim is bounded by the LDS row the block reduction needs, and the rotary width by the
 * neox pairing (`rot/2` apart, so it must be even and fit inside the head). 128 and 64 are what
 * this checkpoint has; the bounds are the kernel's, not the model's. */
/* A lane owns n/32 dims of both operands, so n is a wave multiple and at most eight of them. The
 * head bound is the register file: the query stays resident across the whole block loop. */
/* No dtype: every operand is f32 or i32 by the op's meaning and none of them is an activation. */
/* The sequence bound is the loop that finds which sequence owns a work item: it is recomputed by
 * every workgroup, so it is a handful of compares and not a table. */
/* NO BOUND ON `seqs` HERE. The declared value is a deployment MAXIMUM -- rad-convert declares at
 * 256 -- and a row that refused it would not resolve at convert, so the weights the op reads
 * would never be written into the container and the model would refuse to load with a missing
 * weight rather than with a band. The real count comes off `cu_seqlens` at issue, and the
 * launcher is where it is checked. */
static const RadConstraint cQsaWork[] = {
    C_GE("n", 1), C_GE("ratio", 1), C_GE("seqs", 1), C_GE("work", 1), C_IN("dtype", "bf16"),
};

static const RadConstraint cQsaTailStore[] = {
    C_GE("n", 1), C_GE("ratio", 1), C_GE("seqs", 1), C_IN("dtype", "bf16"),
};

static const RadConstraint cQsaSelect[] = { C_GE("blocks", 1), C_GE("topk", 1), C_GE("ratio", 1) };

static const RadConstraint cQsaScore[] = {
    C_DIV("n", 32), C_GE("n", 32), C_LE("n", 256), C_GE("heads", 1), C_LE("heads", 8),
    C_GE("blocks", 1), C_IN("dtype", "bf16"),
};

static const RadConstraint cQsaBlockKey[] = {
    C_GE("n", 32), C_LE("n", 256), C_DIV("n", 32), C_GE("ratio", 1),
    C_GE("rotary_dim", 0), C_IN("dtype", "bf16"),
};

static const RadConstraint cGramAccum[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "fp8a8"),
};
/* The bf16 form has no scale grid, so `group` constrains nothing; the staging slot is four elements
 * of a row, which is the only divisor left. */
static const RadConstraint cGramAccumBf16[] = {
    C_DIV("n", 4), C_IN("dtype", "bf16"),
};
/* The two fused forms carry the quantiser's own row and nothing else. That is the whole difference
 * from the int8 fusions below, whose rotation makes the width a power-of-two divisor question. */
static const RadConstraint cRmsNormQuantFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};

/* `n DIV 8`, not 4: this one is the all-reduce's message as well as the norm's input and a block
 * pushes its rows as 16B WORDS, so a row has to be a whole number of them. world_size is a hard
 * constraint for the reason cArLnHadQuantI8 gives -- the handshake is one peer flag per block. */
static const RadConstraint cArRmsNormQuantFp8[] = {
    C_DIV("n", 8), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
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
/* The same bounds at four ranks, where the handshake is the wide family's (r4d_ar_wide.h). Exact
 * wire only: the four-rank lossy wire is the ti8 two-shot and has no fused form, so a `wire` 6
 * request is a selection miss here and its caller keeps the unfused pair, where an init refusal
 * would stop declare. The caller states `wire` 0 on the exact form, because a constraint on an
 * absent key does not hold; the init and the entry still refuse any other wire. */
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
/* The same at four ranks, exact wire only and constrained so for cArHcWrite4's reason. */
static const RadConstraint cArGatherHcWrite4[] = {
    C_GE("M", 1), C_DIV("n", 8), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
    C_EQ("world_size", 4), C_EQ("wire", 0), C_GE("top_k", 1), C_LE("top_k", 64),
    C_P2P,
};
static const RadConstraint cGatedQuantFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* The rotated form wants n divisible by the GROUP and not merely by 4, for the reason
 * cHadQuantActFp8 gives: a butterfly cannot straddle the end of a row. */
static const RadConstraint cGatedHadQuantFp8[] = {
    C_DIV("n", R4D_FP8_BLOCK), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* The narrow GEMM's row plus the fold's own two. `N DIV 256` is `nout DIV 128` restated on the
 * weight's extent: a block emits exactly one 128-column scale group, so a fused width that is not
 * a multiple of 128 would put a group across two blocks with nowhere to finish it. `act IN silu`
 * because only that arm is compiled -- a sigmoid gate is a gated attention's and reaches
 * `gate_quant_fp8`, whose operands are two buffers rather than one plane.
 *
 * `N GE 2048` IS THE SPLIT-K RULE, NOT A TUNING. The fold has no partial plane and no second
 * launch, so its entry point REFUSES any shape the narrow split rule would widen, and that rule
 * splits whenever K >= 512 and the grid's wave count, N/16, is under its target of 128 -- which is
 * every N under 2048. The one shape it serves below that, K = 256, is a 256-wide residual stream
 * no model here has. Claiming the band anyway hands the op a kernel that fails at issue, where an
 * unclaimed band is reported at declare. */
static const RadConstraint cGemmNtQGatedFp8[] = {
    C_LE("M", 64), C_DIV("K", 256), C_DIV("N", 256), C_GE("N", 2048),
    C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "fp8a8"), C_IN("act", "silu"),
};
static const RadConstraint cGateQuantFp8[] = {
    C_DIV("n", 4), C_EQ("group", R4D_FP8_BLOCK), C_IN("dtype", "bf16"),
};
/* libr4d's cSampleChain, unchanged. The kernel has no geometry -- the radix passes are over the 256
 * buckets of a float's bytes, not over the vocabulary -- so a row with no predicate at all would
 * match a query that forgot to say what it wanted. */
static const RadConstraint cSampleChain[] = {
    C_GE("n_vocab", 1),
};

static const RadConstraint cQuantActI8[] = {
    C_DIV("n", 16), C_IN("dtype", "bf16"),
};
static const RadConstraint cHadQuantActI8[] = {
    C_DIV("n", 16), C_DIV("n", 128), C_LE("group", 512), C_IN("dtype", "bf16"),
};
static const RadConstraint cRmsNormHadQuantI8[] = {
    C_DIV("n", 16), C_DIV("n", 128), C_LE("group", 512), C_IN("dtype", "bf16"),
};
/* world_size is a hard constraint, not a preference: the handshake is one peer flag per block.
 * `n <= 16256` is the widest row whose fp32 staging fits LDS beside the reduction planes at the
 * widest workgroup (r4d_ar_ln_blocks_min), rounded down to the 128 above. The compressed wire also
 * stages every row a block owns, which bounds M by the flag capacity times the rows that fit; that
 * product is not a predicate, so the shim refuses it by name. */
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

/* ---- THE INT4 / INT8 / ROTATED-QUANT FAMILY IS RETAINED ON PURPOSE ---------------------------
 *
 * No architecture in this tree resolves a dense w4a8, w8a8, w4a16 or `*_had_quant_i8` row, and
 * `rad-quantize`'s w4_g128 output has no consumer in an engine run. They stay because a kernel
 * family is far cheaper to carry than to rewrite.
 *
 * So the rows below are not dead, they are UNRESOLVED, and the difference is which plugin exists
 * rather than whether the kernel works. `rad-kbench --bench` still covers them and r4d_selftest
 * still checks the fusion claims, which is what keeps "retained" from decaying into "rotted".
 */

/* ---- rows whose band is read off the entry point ---------------------------------------------
 * The constraints below come from the entry point's OWN rejection tests and its header prose
 * rather than from a kernel-side row, which is the weaker source a row table exists to replace --
 * so each is marked.
 */

/* r4d_gemm_w8a8_nt_m64. Prose: same fragment order as the 4-bit kernel at twice the radix, one
 * f16 scale per (row, group of 128) in an f16-ONLY plane. Rejection tests mirror the w4a8 kernel's
 * (N % 16, K % (SK*group)); MAX_M is 32768 and explicitly "not structural", but the kernel is a
 * skinny one and the tiled sibling covers above 64, so the band bound is transcribed from the
 * w4a8 row it is a copy of. */
static const RadConstraint cGemmNtW8A8[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w8a8"),
};
/* r4d_gemm_w4a8_tiled / _asym_tiled. Entry point: N % 256, K % group, M >= 1. */
static const RadConstraint cGemmW4A8Tiled[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 256), C_IN("dtype", "w4a8 w4a8_asym"),
};
/* r4d_gemm_w8a8_tiled. The same kernel at radix 8; the asym arm exists in the .hip but is NOT
 * declared in r4d.h, so this plugin cannot reach it and `w8a8_asym` is deliberately absent from
 * the dtype set. */
static const RadConstraint cGemmW8A8Tiled[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 256), C_IN("dtype", "w8a8"),
};
/* r4d_gemm_w4a8_prefill. Entry point: N % BN and K % group, where BN is the VARIANT's block width
 * -- 256 at the default and at every variant listed below except the two 1024-thread arms. N %
 * 256 is the necessary condition for the default; a variant whose BN does not divide N is refused
 * by the shim, which is the per-call half of spec §2.1. */
static const RadConstraint cGemmW4A8Prefill[] = {
    C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 256), C_IN("dtype", "w4a8"),
};

/* =================================================================================== tunables
 *
 * spec §15: constraint matching finds a kernel that is CORRECT for a geometry; the tile
 * configuration is a property of the shape and the machine. What follows is the SPACE each of
 * those kernels can be instantiated over, one axis at a time, plus the cross-axis legality test
 * that a product of independent lists cannot express. Every axis carries the value an untuned
 * install runs, so a fresh card is fast rather than broken.
 *
 * AXES, NOT A LIST OF PRE-NAMED COMBINATIONS, and the difference is not cosmetic. A hand-written
 * list of points has holes an axis list cannot have: a ladder that skips one value leaves every
 * shape that wanted it running on a tile sized for something else, silently. Thirty names over
 * five axes reach thirty of nine hundred points.
 *
 * Naming points also keeps the tuning LAW -- ladders, caps, pick_sk -- inside this plugin's launch
 * adapter as C++, where an out-of-tree author cannot write the same thing. Declaring the space is
 * what lets the core tune a library it was not compiled against.
 */

/* ------------------------------------------------------------------- the skinny GEMM's axes
 *
 * (WV, SK, MB, NPW, NT). Zero means "decide at issue" on the two axes that have a law -- SK is
 * then the split count the shape's own ladder picks and MB covers every row tile in one block --
 * and those laws stay in r4d_gemm_args.h because they read M and the N tile count, which exist only
 * at issue. A TUNED value overrides the law; it never merges with it.
 *
 * A point in this space is what a name like "wv1_sk8_mb3_npw4" abbreviates: wv=1 sk=8 mb=3 npw=4
 * nt=0. The space holds nine hundred of them. */
static const int64_t kGemmWv[]  = { 1, 2, 4 };
static const int64_t kGemmSk[]  = { 0, 1, 2, 4, 5, 8, 10, 16, 20, 32 };
static const int64_t kGemmMb[]  = { 0, 1, 2, 3, 4 };
static const int64_t kGemmNpw[] = { 1, 2, 4 };
static const int64_t kGemmNt[]  = { 0, 1 };

/* THE TILE IS (BM, BN), named as one axis because each value is one compiled arm. 0 is the
 * shape's own choice; see r4d_rad_gemm_bf16_tiled. */
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

/* wv*sk*32 IS THE BLOCK'S THREAD COUNT and 1024 is the hardware's ceiling; the per-MB/NPW bound
 * is tighter still and r4d_gemm_args.h's block_bound() owns it, because it depends on the bound
 * RULE which is a property of the row rather than of the geometry. This is the part that can be
 * decided from the geometry alone, and it is here so the tuner skips ~40% of the space without
 * launching anything.
 *
 * K MUST DIVIDE INTO SK SLICES OF THE KERNEL'S OWN UNIT -- 16 for the bf16 kernels, 128 wherever
 * a quantised weight carries a scale per 128 K. `dtype` is a geometry parameter on every one of
 * these rows, so the unit is derivable here and a candidate that this K cannot split is skipped
 * rather than measured. cfg_for() refuses the same combination at issue; this is the cheap half
 * of the same rule and neither is load-bearing alone. */
static int r4d_tune_valid_gemm(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    long long wv = 1, sk = 0;
    for (int i = 0; i < n_c; ++i) {
        if (c[i].kind != RAD_P_INT) continue;
        if      (!std::strcmp(c[i].key, "wv")) wv = c[i].ival;
        else if (!std::strcmp(c[i].key, "sk")) sk = c[i].ival;
    }
    if (sk <= 0) return 1;                       /* the law decides, and it decides legally */
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
 * ONE AXIS, AND IT IS NOT A CROSS PRODUCT, because libr4d's tiled kernel is not parameterised on
 * (BM, BN): it is a switch over compiled template instantiations, and only some (BM, BN) pairs
 * have an arm. Declaring bm x bn would name combinations that do not exist -- the opposite defect
 * to the hole the skinny GEMM had, and just as wrong.
 *
 * The values are the arm ids, and r4d_cfg_tiled() maps each to the (BM, BN) the N constraint
 * needs. Only arms that compute a CORRECT result are listed: libr4d's table is mostly cost
 * ablations, which "compute wrong answers by construction" in its own words, and a tuner handed
 * those would benchmark a kernel that does not do the GEMM and then pick it. */
static const int64_t kTileArms[] = { 0, 6, 10, 12, 57, 59, 114, 124, 125, 126, 128, 129, 138, 139, 146 };
static const RadTunable tGemmTiled[] = {
  { "tile", kTileArms, NELEM(kTileArms), 0,
    "compiled tile arm; 0 is BM 128 x BN 256, libr4d's own default" },
};

/* The prefill kernel's arms, on the same terms. Ablations (20-23, 25, 26) are excluded for the
 * same reason. */
static const int64_t kPrefillArms[] = { 0, 2, 3, 5, 6, 8, 9, 10, 13, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35 };
static const RadTunable tGemmPrefill[] = {
  { "tile", kPrefillArms, NELEM(kPrefillArms), 34,
    "compiled tile arm; 34 is BM 128 x BN 256 with VOPD, ASCVEC, STFAST and Swizzle 4" },
};

/* ------------------------------------------------------------------------ split-KV attention
 * 0 is libr4d's measured split law (r4d_attn_paged_h256_gqa6.hip carries the table), which reads
 * num_seqs and max_ctx -- values that exist only at issue -- so it cannot be a constant and stays
 * the default. The fixed counts are for a tuner that has a concrete batch. RadScratchFn is handed
 * the same choice, so the arena is sized for what this picks and not for the law's answer at some
 * other shape. */
static const int64_t kSplits[] = { 0, 8, 16, 32, 64 };
static const RadTunable tAttnSplit[] = {
  { "splits", kSplits, NELEM(kSplits), 0, "split-KV segments per sequence. 0 = libr4d's split law" },
};

/* ------------------------------------------------------------------- all-reduce launch geometry
 *
 * FOUR AXES, and the cross product is real here: nblocks, nthreads, drain and acq are independent
 * properties of the handshake. `pub` is not an axis -- every arm uses 1 -- so it stays a constant
 * in r4d_ar_entry.hip rather than becoming a knob with one value.
 *
 * nblocks 0 is "scale with the message and clamp"; drain 3 is s_wait_storecnt, which is the
 * correct wait for the fine-grained uncached IPC scratch.
 *
 * RAD-TUNE CANNOT REACH THIS ROW: the 2-rank kernels need a real peer over IPC and rad-tune runs
 * one process, so it reports "no candidate served it" and a deployment runs the defaults. The only
 * harness that can compare them is a serving engine at --tp 2, and RADIANCE_TUNE is how it is told
 * which point to use -- `RADIANCE_TUNE=all_reduce:drain=1`.
 *
 * THE DEFAULTS BELOW ARE THE FASTEST POINT AT BOTH ENDS OF THE RANGE. At a decode message the axes
 * barely separate; at a prefill chunk they do, and the default still wins.
 *
 * On the fp8 fused row only drain and acq are meaningful: it forces `nthreads` to the norm's width
 * for bit-exactness, and pinning `nblocks` would undo the cap r4d_ar_entry.hip's own comment
 * argues for. None of these axes changes a byte of the result. */
static const int64_t kArNblocks[]  = { 0, 8, 16, 24 };
static const int64_t kArNthreads[] = { 256, 512, 1024 };
static const int64_t kArDrain[]    = { 1, 3 };
static const int64_t kArAcq[]      = { 0, 1 };
static const RadTunable tAr[] = {
  { "nblocks",  kArNblocks,  NELEM(kArNblocks),  0, "blocks in the grid. 0 = scale with the message and clamp" },
  { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "threads per block; the fused fp8 row overrides this for bit-exactness" },
  { "drain",    kArDrain,    NELEM(kArDrain),    3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
  { "acq",      kArAcq,      NELEM(kArAcq),      0, "acquire fence after the flag read" },
};

/* --------------------------------------------------------------------------------- row top-k
 * NCH is a REQUEST -- libr4d rounds the chunk width up to a multiple of eight columns -- so the
 * scratch hook asks r4d_rowtopk_bf16_chunks() rather than computing it. PT1 is the stage-1 partial
 * depth in waves and the entry point compiles exactly 2, 4, 8 and 16. `rc`, how many candidates a
 * chunk keeps, is bounded by pt1*32 and stays derived. */
static const int64_t kTopkNch[] = { 0, 1024, 2048, 4096 };
static const int64_t kTopkPt1[] = { 2, 4, 8, 16 };
static const RadTunable tTopk[] = {
  { "nch", kTopkNch, NELEM(kTopkNch), 0, "requested chunk width in columns. 0 = derive from R and N at issue" },
  { "pt1", kTopkPt1, NELEM(kTopkPt1), 8, "stage-1 partial depth in waves" },
};

/* ------------------------------------------------------------------------------- fp8 matvec
 * Rows a wave. 0 is libr4d's own rule, so an untuned install is fast rather than broken. 8 is
 * offered even though the rule never returns it -- `rows_dot` is templated and 8 divides 128, so
 * it is a legal instantiation, and the rule's ceiling of 4 was not derived from the shapes of the
 * models served here. */
static const int64_t kMvRb[] = { 0, 1, 2, 4, 8 };
static const RadTunable tMv[] = {
  { "rb", kMvRb, NELEM(kMvRb), 0, "rows of the weight one wave owns. 0 = libr4d's own rule" },
};

/* ==================================================================================== the rows */

/* ------------------------------------------- the small model ops and the sampler chain -------
 *
 * A constraint is a NECESSARY condition on the GEOMETRY (spec §2.1), and these ops have almost
 * none: there is nothing to specialise an elementwise add on and nothing a norm rejects. What the
 * kernels do require is the activation width -- they read through a uint16_t, and an f32 operand
 * would be read as two bf16 halves and produce plausible garbage -- so `dtype` is the row, and the
 * shims check the operands' own dtypes per call because a parameter is not a promise about a
 * tensor.
 *
 * kv_store carries `block_size >= 1` for the reason ref's does: the key stays RAD_REQUIRED there,
 * so the predicate admits every value and refuses only a query that does not name one. The paged
 * block size itself is derived from the resolved ATTENTION kernel (spec §7.2), which answers 16.
 *
 * The sampler rows constrain only the extent, and only so that a row with no predicate at all
 * cannot match a query that forgot to say what it wanted -- the same reasoning cSampleChain gives.
 * n_cand is bounded because the top-k's bitonic sort runs in LDS over a power-of-two padding. */
static const RadConstraint cBf16Row[]   = { C_IN("dtype", "bf16") };
/* bf16 rows over the byte copy, which moves four bytes a thread: an even width. */
static const RadConstraint cBf16Row2[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
/* Opaque byte rows, four at a time. The width constraint is the vectorised copy's
 * alignment requirement stated where the selector can see it, so a row that cannot be
 * served is refused at resolve rather than at launch. */
static const RadConstraint cU8Row4[]    = { C_IN("dtype", "u8"), C_DIV("n", 4) };
/* The n-gram hash. It reads no activation, so it constrains no dtype -- what it DOES constrain is
 * the shift window, which lives in a register array: `ngram` past 8 is a kernel that would spill
 * its whole history to scratch, and no model asks for one. */
static const RadConstraint cNgramIds[]  = { C_GE("ngram", 2), C_LE("ngram", 8), C_GE("heads", 1) };
/* PLE's gate. `n` is the per-stream width every reduction runs over and it lives in a workgroup, so
 * 4096 is the ceiling: 16 values a thread at 256 threads, which is what the register arrays hold. */
static const RadConstraint cPleGate[]   = { C_IN("dtype", "bf16"), C_GE("n", 1), C_LE("n", 4096),
                                            C_GE("hc", 2) };
/* PLE's convolution. The dtype is the activation's; `width` and `dilation` are capped so that
 * (width-1)*dilation -- the register window the state write-back holds -- cannot exceed 16. The
 * product is what the kernel actually bounds, and a constraint table cannot say "product"; these
 * two caps are the tightest pair that admits every combination under it. */
static const RadConstraint cPleConv[]   = { C_IN("dtype", "bf16"), C_GE("width", 1),
                                            C_LE("width", 5), C_GE("dilation", 1),
                                            C_LE("dilation", 4) };
static const RadConstraint cKvStore[]   = { C_GE("block_size", 1), C_IN("kv_dtype", "bf16") };
static const RadConstraint cKvStoreFp8[] = { C_GE("block_size", 1), C_IN("kv_dtype", "fp8_e4m3") };
static const RadConstraint cRopeRow[]   = { C_GE("head_dim", 1),
                                            C_IN("mode", "neox mrope gptj imrope axial") };
/* The vision tower's rows. The GEMM stages 16-byte chunks, so K is a multiple of 8; the norm and
 * the position embedding take any width. */
static const RadConstraint cLayernorm[] = { C_GE("n", 1), C_IN("dtype", "bf16") };
static const RadConstraint cGridEmbed[] = { C_GE("n", 1), C_GE("side", 1), C_IN("dtype", "bf16") };
static const RadConstraint cGemmBias[]  = { C_GE("M", 1), C_DIV("K", 8), C_IN("dtype", "bf16") };
static const RadConstraint cSampleVoc[] = { C_GE("n_vocab", 1) };
static const RadConstraint cSampleCand[] = { C_GE("n_cand", 1), C_LE("n_cand", 4096) };
/* The activation width the column loop vectorises over; M is unbounded -- the kernel chunks rows. */
static const RadConstraint cLogitsGemm[] = { C_DIV("n_embd", 8), C_IN("dtype", "bf16") };
/* The fp8 head. n_embd on the 128 block the scales are taken over, and NO CAP ON M -- the row
 * re-layouts the weight, so it has to cover the whole range or none of it. A cap at 32 would leave
 * the band above it to the bf16 projection, which reads the E4M3 codes as bf16 and returns empty
 * completions as soon as enough sequences batch together. The kernel chunks above 32 instead, and
 * core/build/rad_builder.cpp refuses the capped arrangement at declare time as well. */
/* 128 IS THE SCALE BLOCK AND IT IS THE WHOLE CONSTRAINT. What makes that worth explaining is its
 * interaction with the layout: r4d_rad_layout_fp8_logits stores a vocabulary row as `n_embd` codes
 * followed by its own scales, and the kernel reads sixteen codes at a time through a uint4, so the
 * row stride has to be a multiple of 16. With K = 128j the used width is 130j, and 130j % 16 == 0
 * reduces to j % 8 == 0 -- K % 1024 == 0 -- so a layout that REFUSED an unaligned stride could
 * only honestly claim 1024 here.
 *
 * A ROW'S CONSTRAINTS HAVE TO BE WHAT ITS LAYOUT CAN DO. Claiming 128 against such a layout claims
 * a band the row cannot serve, and the failure is invisible at any n_embd that happens to be a
 * multiple of 1024 (5120, 2048). At one that is not -- 2560 -- the row satisfies the constraint,
 * wins selection on priority, and then has its layout REFUSED at declare time, with the fallback
 * below it unreachable. That is the band-coverage hazard.
 *
 * SO THE LAYOUT IS THE SIDE THAT WIDENS, NOT THE CONSTRAINT. r4d_fp8_lm.h pads the stored row to
 * 16 bytes -- under 0.3% at K = 2560 and exactly nothing at every K the 1024 rule already admitted
 * -- so the hook serves every multiple of 128 and this row can say so. Which direction to move is
 * decided by what the band is WORTH: at 2560 the alternative is the bf16 head at 1.27 GB a step. */
static const RadConstraint cLogitsGemmFp8[] = { C_DIV("n_embd", 128), C_IN("dtype", "bf16") };

/* No axes: one implementation with nothing to tune. The core appends no choices and the kernel
 * reads no tuned key. */
#define NOTUNE nullptr, 0, nullptr

static const RadKernelInfo kKernels[] = {
/* ---------------------------------- the small model ops (r4d_model_bf16.hip) ----------------
 *
 * The ops between the interesting kernels. They are in this library because a step is a CHAIN: the
 * engine's buffers live in VRAM, libref is domain HOST, and `rad_issue` refuses an op with no
 * launch for the domain the placement planner put it on. An add with no device row is not a slow
 * add, it is a fatal "no device kernel for this band" on the first request.
 *
 * R4D_PRIO_ONLY throughout: nothing else in this plugin implements them, and ref is the fallback
 * only on a host domain.
 */
{ "embed_lookup_bf16", "embed_lookup", "vocab",
  "embedding gather, one workgroup a token",
  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
  "i32 tokens, bf16 table and output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup, nullptr, nullptr, nullptr,
  r4d_shape_embed_lookup, nullptr, nullptr, r4d_rad_embed_lookup_describe },

{ "embed_lookup_q_bf16", "embed_lookup_q", "vocab",
  "quantised embedding gather, four entries a thread",
  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
  "i32 tokens, an E4M3 table with a bf16 or f32 scale or a plain bf16 one, bf16 output",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup_q, nullptr, r4d_rad_layout_embed_q, nullptr,
  r4d_shape_embed_lookup_q, nullptr, nullptr, r4d_rad_embed_lookup_q_describe },

{ "ple_gate_bf16", "ple_gate", "norm",
  "PLE's dot-product gate, one workgroup a (row, stream) pair",
  "n up to 4096 -- every reduction is over one stream's n and lives in a workgroup",
  "bf16 activations, F32 norm gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPleGate), NOTUNE,
  nullptr, nullptr, r4d_rad_ple_gate, nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "ple_conv_bf16", "ple_conv", "norm",
  "PLE's dilated depthwise conv, silu and residual add; one thread a channel",
  "width up to 5 and dilation up to 4 -- (width-1)*dilation is a register window",
  "bf16 activations, weights and conv state; i32 cu, slots, has_init and num_accepted",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cPleConv), NOTUNE,
  nullptr, nullptr, r4d_rad_ple_conv, nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "ngram_ids_i32", "ngram_ids", "vocab",
  "PLE's hashed n-gram row ids, one thread a token",
  "any heads and n-gram size up to 8; heads must divide into ngram-1 equal blocks",
  "i32 tokens and state, i64 multipliers and per-head vocab sizes and offsets",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cNgramIds), NOTUNE,
  nullptr, nullptr, r4d_rad_ngram_ids, nullptr, nullptr, nullptr,
  nullptr, nullptr, nullptr, r4d_rad_ngram_ids_describe },

{ "gather_rows_bf16", "gather_rows", "elem",
  "y[i, :] = x[idx[i], :], one workgroup a row",
  "any n; source and destination row pitches are read off the operands",
  "i32 indices, bf16 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_gather_rows, nullptr, nullptr, nullptr,
  r4d_shape_gather_rows, nullptr, nullptr, r4d_rad_gather_rows_describe },

/* THE BYTE FORMS, and they are separate rows rather than a widened dtype constraint on the bf16
 * one. What moves through here is a KV fragment -- one layer's worth of one block -- and its
 * contents are whatever the attention kernel wrote. Calling that bf16 would be a lie with teeth:
 * libref is the oracle for every op in this tree and its bf16 path loads through float, which is
 * not guaranteed to return a NaN payload unchanged. At u8 every value is 0..255, float carries it
 * exactly, and oracle and kernel agree bit for bit on any payload at all. */
{ "gather_rows_u8", "gather_rows", "elem",
  "y[i, :] = x[idx[i], :] over opaque bytes, one workgroup a row, 4 bytes a thread",
  "n a multiple of 4; source and destination row pitches are read off the operands",
  "i32 indices, u8 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cU8Row4), NOTUNE,
  nullptr, nullptr, r4d_rad_gather_rows_u8, nullptr, nullptr, nullptr,
  r4d_shape_gather_rows_u8, nullptr, nullptr, r4d_rad_gather_rows_u8_describe },

{ "scatter_rows_u8", "scatter_rows", "elem",
  "x[idx[i], :] = v[i, :] over opaque bytes, one workgroup a row, 4 bytes a thread",
  "n a multiple of 4; distinct indices; the destination pitch is read off the operand",
  "i32 indices, u8 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cU8Row4), NOTUNE,
  nullptr, nullptr, r4d_rad_scatter_rows_u8, nullptr, nullptr, nullptr,
  r4d_shape_scatter_rows_u8, nullptr, nullptr, r4d_rad_scatter_rows_u8_describe },

{ "scatter_rows_bf16", "scatter_rows", "elem",
  "x[idx[i], :] = v[i, :] for bf16 rows -- the u8 row's byte copy at twice the width",
  "any n; distinct indices; the destination pitch is read off the operand",
  "i32 indices, bf16 rows",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row2), NOTUNE,
  nullptr, nullptr, r4d_rad_scatter_rows_bf16, nullptr, nullptr, nullptr,
  r4d_shape_scatter_rows_bf16, nullptr, nullptr, r4d_rad_scatter_rows_bf16_describe },

{ "cast_bf16_bf16", "cast", "elem",
  "y = x, bf16 to bf16 -- a strided row copy through libr4d's 2D copy",
  "any M and n; both row pitches are read off the operands, so a COLUMN SLICE of a wider "
  "destination is expressible and is what an intermediate-hidden-state tap writes into",
  "bf16 in, bf16 out, byte-for-byte",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cCastBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_cast, nullptr, nullptr, nullptr,
  r4d_shape_cast },

{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x / gain / y, fp32 reduction",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_rmsnorm },

{ "layernorm_bf16", "layernorm", "norm",
  "layer norm with a gain and an optional bias, two passes over the row",
  "any n; mean and variance accumulate in f32 and only the store narrows",
  "bf16 x / y, f32 or bf16 gain and bias",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLayernorm), NOTUNE,
  nullptr, nullptr, r4d_rad_layernorm, nullptr, nullptr, nullptr,
  r4d_shape_layernorm, nullptr, nullptr, r4d_rad_layernorm_describe },

{ "grid_embed_bf16", "grid_embed", "elem",
  "a learned position grid resampled bilinearly to each row's grid and added in",
  "any n and side; coord [4, M] dense; one workgroup a row",
  "bf16 x, bf16 or f32 table, i32 coord, f32 taps",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGridEmbed), NOTUNE,
  nullptr, nullptr, r4d_rad_grid_embed, nullptr, nullptr, nullptr,
  r4d_shape_grid_embed, nullptr, nullptr, r4d_rad_grid_embed_describe },

{ "gemm_bf16_nt_bias_tiled", "gemm_nt_bias", "gemm",
  "biased GEMM C = res + act(A @ W^T + bias): eight waves over an LDS-staged 64-deep K tile, "
  "the K tail staged as zeros, the activation and residual in the store",
  "any M, N; K a multiple of 8; A rows 16-byte aligned; `res` may be the output itself",
  "bf16 A / W / C / res, bf16 or f32 bias, fp32 accumulate, three RNE roundings in the store",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGemmBias), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_bias, nullptr, nullptr, nullptr,
  r4d_shape_gemm_nt_bias, nullptr, nullptr, r4d_rad_gemm_nt_bias_describe },

{ "add_bf16", "add", "elem",
  "elementwise sum, with the one row-broadcast form",
  "any n; `b` with exactly n elements broadcasts along rows and nothing else does",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
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

{ "sigmoid_bf16", "sigmoid", "elem",
  "logistic sigmoid, grid-strided over the whole extent",
  "any extent; contiguous in and out",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_sigmoid, nullptr, nullptr, nullptr,
  r4d_shape_unary },

{ "silu_mul_bf16", "silu_mul", "elem",
  "silu-gated product over a fused [M, 2n] gate_up",
  "any n; the gate is the first half",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r4d_rad_silu_mul, nullptr, nullptr, nullptr,
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

/* The write side of the fp8kv attention family: without it a cache declared fp8_e4m3 has no way to
 * be filled. Four columns a thread, so head_dim must be a multiple of 4. */
{ "kv_store_fp8", "kv_store", "attn",
  "scatter k and v into a paged E4M3 cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim]; head_dim a multiple of 4; a "
  "negative or out-of-range slot stores nothing",
  "bf16 k / v, fp8-e4m3 cache, i32 slot mapping, scalar k_scale / v_scale descales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cKvStoreFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store_fp8, nullptr, nullptr, nullptr,
  r4d_shape_kv_store, nullptr, nullptr, r4d_rad_kv_store_fp8_describe },

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
 * NOT sample_chain above, and both stay. That one fuses temperature, softmax, top_k, top_p, min_p
 * and the draw in vLLM's fixed order and answers the speculative verify's three outputs; these are
 * the eleven ops the core composes per request, and they carry the stages the fused form has no
 * room for -- the penalties, DRY, a grammar bitmask, locally-typical, XTC. "Fusion is a selection,
 * not a compiler pass" (spec §2.3): shipping only the fused form would make the choice for the
 * plugin.
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
  /* NO `opd_shape`: the gathered plane is (u32 id, f32 logit) pairs in one word-typed buffer, so
   * no fill domain can put a sensible float in the value lane, and `world_size` is not a parameter
   * of this op. r4d_shapes.cpp's header carries the argument; ref declines it for the same two
   * reasons, and a tool skips the case by name rather than comparing two orderings of denormals. */
  nullptr, nullptr, r4d_rad_sample_merge_topk, nullptr, nullptr, nullptr },


/* ---------------------------------------------------------------------------------- attention */
{ "attn_prefill_h256_gqa6_fp8kv", "attn_paged", "attn",
  "paged causal attention, query-tiled (prefill / chunked prefill)",
  "head_dim 256, 6 queries per kv head, paged block 16, varlen",
  "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL,
  ROW(cAttnPrefillFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_fp8, nullptr, nullptr, nullptr,
  r4d_shape_attn_paged },

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
  r4d_shape_attn_paged, nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_fp8_describe },

/* The same decode dispatch with gate_quant_fp8 as the merge's epilogue: two launches, where the
 * attention followed by gate_quant_fp8 is three. */
{ "attn_decode_gq_h256_gqa12_fp8kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa12_fp8kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, fp8-e4m3 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G12Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa12_fp8, r4d_rad_attn_decode_h256_gqa12_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant,
  r4d_rad_attn_decode_gq_h256_gqa12_fp8_describe },

{ "attn_decode_h256_gqa12_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 12 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G12Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_bf16, r4d_rad_attn_decode_h256_gqa12_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged, nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_bf16_describe },

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
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant,
  r4d_rad_attn_decode_gq_h256_gqa12_bf16_describe },

/* The TP4 leg of Flash-Next's sparse attention: the same kernels the gqa12 rows above resolve, at
 * the ratio a four-rank world hands a rank (see cAttnPrefillH256G6Bs4Fp8). The two are not
 * alternatives -- `gqa` is a per-rank parameter, so world 2 resolves gqa 12 and these four rows
 * never win there, and world 4 resolves gqa 6 and the gqa12 rows never win. */
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
  r4d_shape_attn_paged, nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_fp8_describe },

{ "attn_decode_h256_gqa6_bs4_bf16kv", "attn_paged", "attn",
  "paged attention, causal or not, split-KV (decode / speculative verify / draft block)",
  "head_dim 256, 6 queries per kv head, paged block 4, q_len*gqa <= 64, no window",
  "bf16 query, bf16 kv cache, bf16 out, f16 partials",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeH256G6Bs4Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_bf16, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged, nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_bf16_describe },

/* The TP4 leg's sparse decode WITH the gate and the quantiser in the merge, as the gqa12 pair
 * above. Two launches where the unfused path is three, once a layer, at the shape TP4 serves every
 * decode step of this model. */
{ "attn_decode_gq_h256_gqa6_bs4_fp8kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa6_bs4_fp8kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, fp8-e4m3 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G6Bs4Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_fp8, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant,
  r4d_rad_attn_decode_gq_h256_gqa6_fp8_describe },

{ "attn_decode_gq_h256_gqa6_bs4_bf16kv", "attn_paged_gate_quant", "attn",
  "paged split-KV decode attention with the output gate and the fp8 quantiser in the merge",
  "attn_decode_h256_gqa6_bs4_bf16kv's geometry; gate_quant_fp8 over the merged (token, head) rows: "
  "sigmoid only, group 128, the gate 16-byte aligned with a pitch of 8, the codes 8-byte aligned",
  "bf16 query, bf16 kv cache, bf16 gated out, E4M3 codes with an f32 scale per 128 columns",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE,
  ROW(cAttnDecodeGqH256G6Bs4Bf16), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_bf16, r4d_rad_attn_decode_h256_gqa6_scratch,
  nullptr, nullptr,
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant,
  r4d_rad_attn_decode_gq_h256_gqa6_bf16_describe },

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

{ "attn_vit_h72_bf16", "attn_dense", "attn",
  "dense non-causal vision-encoder attention, varlen over images / windows",
  "head_dim 72 native (no padding to 128), multi-head (no GQA), contiguous q/k/v/o, "
  "one cu_seqlens segment per image",
  "bf16 query / key / value / out, f16 operands, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cAttnVit), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_vit, nullptr, nullptr, nullptr },

/* -------------------------------------------------------------------------- gated delta net */
{ "gdn_chunk_scan_k128_v128_c64_bf16", "gdn_chunk_scan", "gdn",
  "gated delta net chunked scan: WY recompute, state recurrence and output in one kernel",
  "head_k 128, head_v 128, chunk 64, varlen, state resident in WMMA accumulators",
  "bf16 q/k/v/A, fp32 gate/beta, bf16 out; the state cache is f32 or f16",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnChunkScan), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_chunk_scan, nullptr, nullptr, nullptr },

/* UNVALIDATED. The DPP row_xmask exchanges in this translation unit (r4d_common.h r4d_xor_lane,
 * used by the conv) are compile-verified only: never measured and never through a correctness
 * gate. Anything this row produces is unverified until it passes rad-kbench. */
{ "gdn_conv_prep_w4_h128_bf16", "gdn_conv_prep", "gdn",
  "gdn prefill preamble: causal conv + qkv split + qk l2norm + gating + per-chunk cumsum "
  "[UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "conv width 4, head_k/head_v 128, chunk 64, varlen, conv state read and rewritten in place",
  "bf16 x / weights / conv state / q,k,v out, fp32 a,b and gate/beta out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnConvPrep), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_conv_prep, nullptr, nullptr, nullptr },

/* UNVALIDATED -- same translation unit and same exchanges as gdn_conv_prep above. */
{ "gdn_conv_update_w4_h128_bf16", "gdn_conv_update", "gdn",
  "gdn decode conv: rolling speculative window over the conv state cache, writes q/k/v "
  "[UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "conv width 4, head_k/head_v 128, state cache width-1 + num_spec, read at accepted offset",
  "bf16 x / weights / conv state / q,k,v out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnConvUpdate), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_conv_update, nullptr, nullptr, nullptr },

{ "gdn_kkt_solve_k128_c64_bf16", "gdn_kkt_solve", "gdn",
  "gdn chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^dg))^-1, gram never in HBM",
  "head_k 128, chunk 64, varlen; one gram per k head, applied to every v head of the group",
  "bf16 k and A out, fp32 beta / gate / gram / inverse",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnKktSolve), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_kkt_solve, nullptr, nullptr, nullptr },

/* UNVALIDATED. The DPP row_xmask exchanges here stand in for 408 ds_bpermute, several of them
 * inside the token loop on the dependency chain, and have never been measured or
 * correctness-gated. */
{ "gdn_recurrent_update_k128_v128_bf16_fp32state", "gdn_recurrent_update", "gdn",
  "gdn decode recurrent update: gating, qk l2norm, delta-rule state update, gated rms norm and "
  "output [UNVALIDATED: DPP row_xmask rewrite never correctness-gated]",
  "head_k 128, head_v 128, paged state, one state written per candidate token",
  "bf16 q/k/v and out, fp32 a,b; the state cache is f32 or f16 and the operand says which",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnRecurrentUpdate), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_recurrent_update, nullptr, nullptr, nullptr },

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
  nullptr, r4d_fuse_gdn_conv_recurrent_update, nullptr },

/* ---------------------------------------------------------------- the gated residual --------
 * r4d_hc_bf16.hip. Three rows; the read is two launches behind one entry point, because `x`
 * depends on every column of the low-rank waist and the kernel boundary is the grid-wide barrier
 * that says so. */
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
  r4d_shape_hc_write, nullptr, nullptr, r4d_rad_hc_write_describe },

{ "mtp_enter_bf16", "mtp_enter", "norm",
  "MTP head entry: two norms, a per-sub-stream fc and one broadcast fc",
  "any M, n 128..5734, hc 2..4",
  "bf16 activations and fc, f32 gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMtpEnter), NOTUNE,
  nullptr, nullptr, r4d_rad_mtp_enter, r4d_rad_mtp_enter_scratch, nullptr, nullptr,
  r4d_shape_mtp_enter, nullptr, nullptr, r4d_rad_mtp_enter_describe },

{ "mtp_enter_e4m3", "mtp_enter", "norm",
  "the MTP head's entry with fc_hidden and fc_embedding stored as E4M3 rows: half the bytes of the "
  "two square matrices every draft round reads",
  "any M, n a multiple of 128 up to 5734, hc 2..4",
  "bf16 activations, E4M3 fc rows with f32 128-column group scales in each row's tail, f32 gains",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMtpEnterE4m3), NOTUNE,
  nullptr, nullptr, r4d_rad_mtp_enter, r4d_rad_mtp_enter_scratch,
  r4d_rad_layout_mtp_e4m3, r4d_rad_relayout_mtp_e4m3,
  r4d_shape_mtp_enter, r4d_rad_unrelayout_mtp_e4m3, nullptr, r4d_rad_mtp_enter_describe },

{ "hc_read_bf16", "hc_read", "norm",
  "grouped rmsnorm, the low-rank read gate, the stream mean, the per-branch write gains and the "
  "fp8 twin of the block input -- in two launches",
  "any M; n a multiple of 128 and at most 7552; hc 2..4; any lowrank; group 0 or 128",
  "bf16 stream and mixing matrices, f32 gain, bf16 out, E4M3 + f32 scale for the optional twin",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHcRead), NOTUNE,
  nullptr, nullptr, r4d_rad_hc_read, r4d_rad_hc_read_scratch, nullptr, nullptr,
  r4d_shape_hc_read, nullptr, nullptr, r4d_rad_hc_read_describe },

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
  r4d_shape_hc_read, r4d_rad_unrelayout_hc_e4m3, nullptr, r4d_rad_hc_read_describe },

{ "gdn_gated_rmsnorm_h128_bf16", "gdn_gated_rmsnorm", "gdn",
  "gated rms norm over a row: out = rms(x) . w . act(z), for the prefill path only",
  "128 channels per row, one wave per row; silu or sigmoid gate",
  "bf16 x / z / out, fp32 weight",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cGdnGatedRmsNorm), NOTUNE,
  nullptr, nullptr, r4d_rad_gdn_gated_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_gdn_gated_rmsnorm },

/* ---------------------------------------------------------------------------- all-reduce ----
 * The two-shot rows outrank the one-shots, for the reason the constraint block above states. */
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

{ "ar_twoshot_8rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch, per-block pipelined "
  "handshake",
  "exactly 8 ranks, all peer-accessible; messages above "
  R4D_XSTR(R4D_AR_WIDE_ONESHOT_MAX_ELEMS) " elements, two hops; shares the wide family's "
  "flag/seq/scratch set; message must shard evenly over the ranks; in place; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R4D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact8), ROW(tAr), nullptr,
  r4d_rad_ar_init, r4d_rad_ar_fini, r4d_rad_ar_twoshot_8rank_exact, nullptr, nullptr, nullptr,
  r4d_shape_all_reduce },

{ "ar_gather_2rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch, per-block flag handshake",
  "exactly 2 ranks, peer-accessible; y is [2 * numel] with rank r's input at r * numel; "
  "message size must be a multiple of 4 bytes; scratch sized by the full message; cudagraph-safe",
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
  "exactly 3 ranks, all peer-accessible; one scratch slot per (parity, source rank); "
  "the only wide form at three -- the two-shot shards the message `world` ways; cudagraph-safe",
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
 * THE FORMAT THE CHECKPOINT SHIPS IN. Every other quantised GEMM here reads a weight this project's
 * own quantiser produced; these three read `<linear>.weight` as safetensors stores it -- F8_E4M3
 * [N][K] with `<linear>.weight_scale_inv` BF16 on a [ceil(N/128)][ceil(K/128)] grid. No repacking
 * and no scale conversion, so a .rad built for them costs exactly what the checkpoint does. Without
 * them the only way to serve Qwen3.8-27B-FP8 is to dequantise it to bf16 at conversion, at twice
 * the weight bytes of a 27B model across two 32 GB cards.
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
  r4d_shape_gemm_nt_q, nullptr },

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
  r4d_shape_gemm_nt_q, r4d_rad_unrelayout_fp8_block, nullptr,
  r4d_rad_gemm_fp8a8_m16_describe },

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
  /* The "byte-identical to gemm_nt_q followed by gated_quant_fp8" above, made executable: both
   * steps are rows of this plugin, so the claim is checkable libr4d-against-libr4d rather than
   * against a tolerance. r4d_fuse.cpp names the two parameters that are not copies. */
  r4d_fuse_gemm_nt_q_gated },

{ "gemm_fp8a8_tiled", "gemm_nt_q", "gemm",
  "the same GEMM for prefill row counts: 128x128 tile, eight waves, one K tile prefetched in "
  "registers, and the ACTIVATION on the A operand so the epilogue is a coalesced store",
  "any M; K and N multiples of 128 -- the fold at the end of the K loop assumes a whole scale "
  "block, and the weight's block scale is read as one scalar for the block's 128 N rows",
  "E4M3 A with an f32 scale per (row, 128 K), E4M3 W with a bf16 scale per 128x128 tile, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_TILED,
  ROW(cGemmNtFp8A8Tiled), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_fp8a8_tiled, nullptr,
  r4d_rad_layout_fp8_block, r4d_rad_relayout_fp8_block,
  r4d_shape_gemm_nt_q, r4d_rad_unrelayout_fp8_block },

{ "gemm_i8a8_nt_m16", "gemm_nt_q", "gemm",
  "gemm_fp8a8_nt_m16 at INT8 codes: the same narrow tile, staging and split-K, with the iu8 WMMA "
  "for the fp8 one and the weight's scale per row",
  "the fp8 row's: M <= 64; K a multiple of 128; N a multiple of 16",
  "i8 A with an f32 scale per (row, 128 K), i8 W with a bf16 scale per ROW per 128 K, bf16 C; "
  "int32 accumulation within a 128-K block, folded to f32 at its end",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_M16,
  ROW(cGemmNtI8A8M16), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_i8a8_m16, r4d_rad_gemm_i8a8_m16_scratch,
  r4d_rad_layout_i8_block, r4d_rad_relayout_i8_block,
  nullptr, r4d_rad_unrelayout_i8_block, nullptr,
  r4d_rad_gemm_i8a8_m16_describe },

{ "gemm_i8a8_tiled", "gemm_nt_q", "gemm",
  "gemm_fp8a8_tiled at INT8 codes: the same 128x128 prefill tile, the iu8 WMMA for the fp8 one and "
  "the weight's scale per row, one a lane a tile in the fold",
  "the fp8 row's: any M; K and N multiples of 128",
  "i8 A with an f32 scale per (row, 128 K), i8 W with a bf16 scale per ROW per 128 K, bf16 C",
  RAD_DOMAIN_DEVICE, R4D_PRIO_GEMM_FP8_TILED,
  ROW(cGemmNtI8A8Tiled), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_i8a8_tiled, nullptr,
  r4d_rad_layout_i8_block, r4d_rad_relayout_i8_block,
  nullptr, r4d_rad_unrelayout_i8_block },

/* BAND READ OFF THE ENTRY POINT'S OWN REJECTION TESTS, not off a kernel-side row. */
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

/* BAND READ OFF THE ENTRY POINT'S OWN REJECTION TESTS. */
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

/* BAND READ OFF THE ENTRY POINT'S OWN REJECTION TESTS. */
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

/* BAND READ OFF THE ENTRY POINT'S OWN REJECTION TESTS. */
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

/* THE KERNEL-SIDE ROW AND THE ENTRY POINT DISAGREE ON R. The kernel's row says R <= 1024, in the
 * predicate and in the prose; the entry point throws for R > R4D_TOPK_PT2, which is 32, and calls
 * that "a real limit, not a tuning choice". The predicate below keeps the kernel-side value -- a
 * constraint is a NECESSARY condition and this one is simply weak -- and the shim rejects R > 32
 * with RAD_E_SHAPE. Here the code is the only correct statement of the bound: both the row and its
 * prose overstate it. */
{ "scale_rows_bf16", "scale_rows", "elem",
  "one scalar a row, optionally through a sigmoid, broadcast across the row",
  "any M and n; the scalar stream is [M] or [M, 1]",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cScaleRows), NOTUNE,
  nullptr, nullptr, r4d_rad_scale_rows, nullptr, nullptr, nullptr,
  r4d_shape_scale_rows },

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
  "n_expert <= 512, top_k <= min(n_expert, 32); single-workgroup decode or fused prefill; "
  "bf16 logits and weights, i32 everything else",
  "bf16 in, bf16 weights + i32 ids, sorted slots, offsets and counts out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRouterTopkScatter), NOTUNE,
  nullptr, nullptr, r4d_rad_router_topk_scatter, r4d_rad_router_topk_scatter_scratch, nullptr, nullptr,
  r4d_shape_router_topk_scatter, nullptr, nullptr, r4d_rad_router_topk_scatter_describe },

{ "moe_scatter_i32", "moe_scatter", "moe",
  "a stable counting sort of the routing slots by expert, with the offsets and the histogram",
  "n_expert <= 256; one launch at M*top_k <= 1024 and four above it, and the four exist to keep "
  "the sort stable without an atomic deciding the order",
  "i32 throughout",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeScatter), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_scatter, r4d_rad_moe_scatter_scratch, nullptr, nullptr,
  r4d_shape_moe_scatter },

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
  /* THE ROW-MAJOR fp8 LAYOUT, WHICH IS THE CHECKPOINT'S OWN BYTES, and it is shared verbatim with
   * the fp8a16 matvec. `moe_gemm_q`'s operand positions put the weight at 2 and its scale plane at
   * 3 -- the same places `gemm_nt_q` puts them -- so the hook needs no MoE-specific arm and the
   * container stores an expert exactly as the safetensors file holds it.
   *
   * NOT the FRAGMENT order the dense fp8a8 GEMMs use. A fragment plane is a permutation for a
   * kernel that reads a WMMA fragment straight from a coalesced global load; this kernel stages
   * through LDS because its rows are GATHERED -- a sorted row's activation comes from whichever
   * token routed there -- so there is nothing for the permutation to buy, and reading it would
   * require the de-permuting staging path the dense kernel uses for its OTHER operand.
   *
   * A DECLARED-fp8 WEIGHT WITH NO LAYOUT HOOK IS REFUSED BY rad-convert, by name, with "fp8_e4m3
   * with block scales in a companion tensor, which this path cannot read". That is the right
   * refusal when the hook is missing: the converter has no other way to know that the bytes on
   * disk are already the bytes the kernel wants. */
  nullptr, nullptr, r4d_rad_moe_gemm_q, r4d_rad_moe_gemm_q_scratch,
  r4d_rad_layout_moe_fp8, nullptr,
  r4d_shape_moe_gemm_q, nullptr },

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
  /* NO SHAPE FUNCTION, for the reason r4d_shapes.cpp's header gives for every non-fp8 gemm_nt_q
   * grid: the weight is a plugin-private packed plane whose byte count r4d_layout.cpp owns, and a
   * second statement of it here would drift the first time the packing changed. The kernel's
   * decode is checked by r4d_selftest against the relayout that produced the bytes, which is the
   * comparison that can actually fail -- a kbench case would be comparing two readings of random
   * nibbles.
   *
   * WHY 4 BITS AT ALL, and it is capacity and not quality. Qwen3.8-Flash-Next is 512 experts of
   * 640 over 48 layers: 112.8 GiB of expert weight at E4M3, against two 32 GB cards. Expert
   * parallelism halves the per-rank share and the placement plan still refuses 26 GiB of it --
   * and a ROUTED weight cannot be served from the file tier, because nothing schedules a read for
   * an expert the router picks this step. At four bits the share fits VRAM plus a few GiB of
   * pinned host, which is the difference between a model that serves and one that does not.
   *
   * THERE IS AN INVERSE THOUGH, and with no shape function beside it that is not a contradiction:
   * the two hooks answer different questions. The shape function describes the plane the KERNEL is
   * handed, which here is a plugin-private packed one a caller cannot allocate meaningfully; the
   * inverse produces the canonical codes plane, from which the oracle hands a dense REFERENCE the
   * codes one per element in whatever form that reference reads. Without it this row -- the
   * largest weight a routed model touches -- would be one an oracle has to decline by name. */
  nullptr, nullptr, r4d_rad_moe_gemm_w4a8, r4d_rad_moe_gemm_w4a8_scratch,
  r4d_rad_layout_moe_w4, r4d_rad_relayout_moe_w4, nullptr, r4d_rad_unrelayout_moe_w4, nullptr,
  r4d_rad_moe_gemm_w4a8_describe },

{ "moe_gemm_fp8a8_mod4", "moe_gemm_q_mod4", "moe",
  "moe_gemm_fp8a8 over four classes of experts by number mod 4: the same kernel, each run reading "
  "its class's table",
  "as moe_gemm_fp8a8, every class",
  "as moe_gemm_fp8a8",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmQ), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_q_mod4, r4d_rad_moe_gemm_q_scratch,
  r4d_rad_layout_moe_fp8_mod4, nullptr,
  r4d_shape_moe_gemm_q_mod4, nullptr },

{ "moe_gemm_w4a8_mod4", "moe_gemm_q_mod4", "moe",
  "moe_gemm_w4a8 over four classes of experts by number mod 4: the same kernel, each run reading "
  "its class's table",
  "as moe_gemm_w4a8, every class",
  "as moe_gemm_w4a8",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmW4), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_w4a8_mod4, r4d_rad_moe_gemm_w4a8_scratch,
  r4d_rad_layout_moe_w4_mod4, r4d_rad_relayout_moe_w4_mod4, nullptr,
  r4d_rad_unrelayout_moe_w4_mod4, nullptr, r4d_rad_moe_gemm_w4a8_mod4_describe },

{ "moe_gemm_bf16", "moe_gemm", "moe",
  "the grouped GEMM with unquantised experts: one run of expert-sorted rows a block, bf16 WMMA over "
  "an LDS-staged 64-deep K pass with the next one prefetched in registers",
  "N a multiple of 64, K a multiple of 64; the 16-row decode tiles and the 32-row run-aligned "
  "prefill tiles of moe_gemm_q, by the same rule",
  "bf16 activation, bf16 weight as the checkpoint stores it ([N, K] row-major, no layout hook), "
  "f32 accumulate over the whole of K, bf16 out rounded to nearest-even",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGemmBf16), NOTUNE,
  /* NO LAYOUT HOOK AND NO SHAPE FUNCTION. The plane is the checkpoint's own bytes, so the core's
   * identity layout is the right answer and a container stores an expert exactly as the
   * safetensors file holds it -- which is what lets the file tier read one into a staging buffer
   * and hand it to this kernel unchanged. libref describes the op for a tool that needs buffers. */
  nullptr, nullptr, r4d_rad_moe_gemm_bf16, r4d_rad_moe_gemm_bf16_scratch, nullptr, nullptr,
  nullptr, nullptr, nullptr, r4d_rad_moe_gemm_bf16_describe },

{ "moe_gather_bf16", "moe_gather", "moe",
  "the weighted scatter back to token order, summing a token's slots in ascending slot order",
  "any M and n; the inverse permutation is built in scratch rather than searched",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cMoeGather), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gather, r4d_rad_moe_gather_scratch, nullptr, nullptr,
  r4d_shape_moe_gather },

{ "rowtopk_bf16", "row_topk", "select",
  "exact per-row top-R of a bf16 matrix, descending, ties to the lower column",
  "any M, N a multiple of 8, R <= 32 (the row says 1024 and the entry point says 32); "
  "two launches, caller-sized partial buffer",
  "bf16 in, f32 values + i32 columns out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRowTopK), ROW(tTopk), nullptr,
  nullptr, nullptr, r4d_rad_rowtopk, r4d_rad_rowtopk_scratch, nullptr, nullptr,
  r4d_shape_row_topk , nullptr, nullptr, r4d_rad_rowtopk_describe },

{ "logit_rerank", "logit_rerank", "select",
  "R candidates a row rescored exactly against the full-precision head",
  "any M, R <= 32, bf16 x and the E4M3 lm_head plane logits_gemm_fp8 stores; one block a row with "
  "the waves spread over the candidates, one launch, no scratch",
  "bf16 in, f32 values + i32 ids out, ties to the lower id",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLogitRerank), NOTUNE,
  nullptr, nullptr, r4d_rad_logit_rerank, nullptr,
  r4d_rad_layout_fp8_logits, r4d_rad_relayout_fp8_logits,
  /* NO opd_shape, for logits_gemm_fp8's reason and in its words: the declared shape and
   * the stored plane differ and only the container knows. A tool that allocated
   * [n_vocab][n_embd] would hand this kernel a buffer it reads 1.6% past the end of on
   * every row, because the row's own bf16 scales ride in its tail -- and the comparison
   * that follows fails on the padding rather than on the arithmetic. So it is skipped by
   * name like its neighbour until RadOpdDesc can say "declared this, stored that". */
  nullptr, r4d_rad_unrelayout_fp8_logits, nullptr, r4d_rad_logit_rerank_describe },

{ "logit_rerank_i8", "logit_rerank", "select",
  "logit_rerank over an INT8 head: the same candidates rescored against logits_gemm_i8's rows",
  "logit_rerank's: any M, R <= 32, bf16 x and the i8 lm_head plane logits_gemm_i8 stores",
  "bf16 in, f32 values + i32 ids out, ties to the lower id",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cLogitRerank), NOTUNE,
  nullptr, nullptr, r4d_rad_logit_rerank_i8, nullptr,
  r4d_rad_layout_i8_logits, r4d_rad_relayout_i8_logits,
  /* No opd_shape, for logit_rerank's reason: the declared shape and the stored row differ. */
  nullptr, r4d_rad_unrelayout_i8_logits, nullptr, r4d_rad_logit_rerank_i8_describe },

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

/* The gate on both fp8a8 GEMMs: `activation_scheme: "dynamic"` means the checkpoint ships no
 * activation scale, so one is computed every step. This is the STANDALONE form, for the launches
 * whose input is some GEMM's output and so cannot ride on a producer. Where a producer already
 * holds the row, `rmsnorm_quant_fp8` or `gated_quant_fp8` below does the same work on its pass. */
{ "quant_act_fp8", "quant_act_fp8", "quant",
  "per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation",
  "n a multiple of 4 -- a lane owns four columns, so the store is one dword and the wave's is 128 "
  "contiguous bytes; group 128, which is the WEIGHT scale's block width and therefore a property "
  "of the checkpoint rather than a knob; a ragged last group is written zero-padded",
  "bf16 in, E4M3 out in PLAIN ROW-MAJOR order with an f32 scale per (row, 128 columns)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_act_fp8, nullptr, nullptr, nullptr,
  r4d_shape_quant_act_fp8, nullptr, nullptr, r4d_rad_quant_act_fp8_describe },

/* The rotated twin. It exists for the FOUR-BIT expert weight and for nothing else: at E4M3 a
 * rotation is slightly WORSE, because four exponent bits already carry the outliers a rotation
 * removes. Four-bit weights are the opposite case -- sixteen levels spent covering one outlier
 * channel is the whole error -- so this pays the activation's small price to buy the weight's
 * large one. The trade only closes with GPTQ on top: rotation alone makes plain round-to-nearest
 * int4 substantially worse, which is why this row exists ahead of the error-feedback packer and
 * not the other way round. */
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
  r4d_shape_quant_act_fp8, nullptr, nullptr, r4d_rad_had_quant_act_fp8_describe },

{ "quant_act_i8g", "quant_act_i8g", "quant",
  "per-(row, 128-column) symmetric int8 quantisation of a bf16 activation",
  "quant_act_fp8's: n a multiple of 4, group 128, a ragged last group written zero-padded",
  "bf16 in, int8 out in PLAIN ROW-MAJOR order with an f32 scale of amax/127 per (row, 128 columns)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_act_i8g, nullptr, nullptr, nullptr,
  nullptr, nullptr, nullptr, r4d_rad_quant_act_i8g_describe },

{ "had_quant_act_i8g", "had_quant_act_i8g", "quant",
  "a 128-wide Hadamard along K, then per-(row, 128-column) symmetric int8 quantisation",
  "had_quant_act_fp8's: n a multiple of 128 and group 128",
  "bf16 in, int8 out in PLAIN ROW-MAJOR order with an f32 scale of amax/127 per (row, 128 "
  "columns); the sqrt(128) of the unnormalised transform is left to the weight's stored scale",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cHadQuantActFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_had_quant_act_i8g, nullptr, nullptr, nullptr,
  nullptr, nullptr, nullptr, r4d_rad_had_quant_act_i8g_describe },

/* THE CALIBRATION HALF OF GPTQ, and the only op in this library that no forward pass issues. It
 * runs under a calibration flag, beside the quantiser whose output it characterises, and the
 * Hessians it accumulates are read at CONVERT time by the 4-bit expert packer -- so the producer
 * and the consumer are in different processes and the channel between them is a file. That is
 * unusual enough to say out loud: a row here that the serving graph never resolves is normally a
 * defect -- a kernel built and never wired -- and this one is not. */
/* THE FIRST PIECE OF QSA, and the only one of its four that is query-independent -- which is
 * exactly why it is a kernel and not a loop inside the scorer. A block key is fixed once its four
 * tokens exist, so it is built when the fourth arrives and read by every query after it.
 * docs/QSA.md is the decomposition it belongs to. */
/* The indexer's vote. One workgroup a query, one WAVE a block: a lane owns n/32 consecutive dims
 * of the query heads AND the block key, so a head's dot product is one wave reduction and the
 * block key is read once for all four heads -- which is why the head loop is inside the block
 * loop. docs/QSA.md is what it belongs to. */
/* A four-digit radix select, one workgroup a query. Not a sort: attention is a softmax over a SET
 * so the order inside the table does not matter, and sorting 65536 scores a query a layer a step
 * is not affordable. What does matter is that the set is exact and DETERMINISTIC, so ties break
 * to the lower block index through a prefix count rather than through whichever atomic landed
 * first -- this tree compares generated text across builds byte for byte. */
/* One workgroup a token, and most of them exit on the second compare. */
/* One workgroup a potential block. */
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
  r4d_shape_qsa_work, nullptr, nullptr, r4d_rad_qsa_work_describe },

{ "qsa_tail_store_bf16", "qsa_tail_store", "attn",
  "park the raw index keys of the tokens that have not completed a block, one slot a position",
  "any token count, any width, any ratio; the sequence count comes off cu_seqlens at run time "
  "and not off the tail's declared height",
  "bf16 keys in, an i32 position and an i32 cu_seqlens; the bf16 tail written in place",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaTailStore), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_tail_store, nullptr, nullptr, nullptr,
  r4d_shape_qsa_tail_store, nullptr, nullptr, r4d_rad_qsa_tail_store_describe },

{ "qsa_select_i32", "qsa_select", "attn",
  "the topk highest-scoring blocks as a page-id table, with the partial page appended last",
  "any M, any block count, any topk; a decode-shaped launch splits the block list across the "
  "device and merges, and falls back to one workgroup a query when the scratch is absent",
  "f32 scores and an i32 page table in; an i32 [M, topk+1] table and a live count out",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaSelect), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_select, r4d_rad_qsa_select_scratch, nullptr, nullptr,
  r4d_shape_qsa_select, nullptr, nullptr, r4d_rad_qsa_select_describe },

{ "qsa_score_bf16", "qsa_score", "attn",
  "the QSA indexer's block score: relu per (block, head), summed over heads, over 1/sqrt(n)",
  "n a multiple of 32 up to 256, at most 8 heads, any block count and any M",
  "bf16 queries and block keys, an i32 page-id table, f32 scores; a negative table entry scores "
  "-inf and not zero",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQsaScore), NOTUNE,
  nullptr, nullptr, r4d_rad_qsa_score, nullptr, nullptr, nullptr,
  r4d_shape_qsa_score, nullptr, nullptr, r4d_rad_qsa_score_describe },

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
  r4d_shape_qsa_block_key, nullptr, nullptr, r4d_rad_qsa_block_key_describe },

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

/* THE TWO fp8 FUSIONS, the counterparts of the ones the int8 path carries. A producer that
 * already holds the row -- an RMS norm, a gated product -- writes the E4M3 pair on the same pass,
 * so the quantiser's launch and its second read of the row both disappear. The
 * codes are the unfused pair's exactly, because both round to bf16 before quantising; the engine
 * compares generated text across builds byte for byte and that is what makes it hold. */
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
  /* `add` + `rmsnorm` + `quant_act_fp8`, and NOT `rmsnorm_add` + `quant_act_fp8`: this kernel
   * takes the sum of squares over the BF16-ROUNDED residual and ref's `rmsnorm_add` takes it over
   * the unrounded f32 sum, so only the first spelling is the function this computes. See
   * r4d_fuse.cpp, which also states what the hook cannot see about the residual's presence. */
  r4d_fuse_rmsnorm_quant_fp8 },

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
  /* `silu_mul` + `quant_act_fp8` on the silu arm only. The sigmoid arm has no chain and the
   * reason is not the activation -- `sigmoid` and `mul` are both ops with rows here -- but that
   * `gate_up` is one packed [M, 2n] plane and the operand wiring cannot name half of an operand.
   * `silu_mul` is the only op that does that split itself. */
  r4d_fuse_gated_quant_fp8, r4d_rad_gated_quant_fp8_describe },

/* The rotated twin, for a down projection whose weight is stored at w4a8h. NO FUSION HOOK, and
 * the absence is the honest answer rather than an oversight: `silu_mul` + `had_quant_act_fp8` is
 * a pair a plugin could emit, but nothing in this tree emits it -- the rotated activation exists
 * for the routed MoE arm and the MoE block declares this op directly. A hook that matched a chain
 * no graph contains would be a claim nothing exercises. */
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
  r4d_shape_gated_quant_fp8, nullptr, nullptr, r4d_rad_gated_had_quant_fp8_describe },

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
  /* `sigmoid` + `mul` + `quant_act_fp8`. This row rounds the activation to bf16 before the
   * multiply where its `gated_quant_fp8` sibling does not, precisely because the pair IT replaces
   * is two kernels with a bf16 buffer between them -- so the sibling's expressible arm is silu and
   * this one's is sigmoid. The silu arm is declined: libr4d has no `silu` row and ref's is
   * `x * sigmoid(x)` against this kernel's `x / (1 + exp(-x))`. */
  r4d_fuse_gate_quant_fp8, r4d_rad_gate_quant_fp8_describe },

/* ---------------------------------------------------------------------------------- sampling
 *
 * spec §13 requires sampling on device, and this is the project's only device sampler: libref
 * implements docs/OPS.md's eight HOST ops and nothing else implements this one, so there is no
 * second implementation to compare it against. */
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
  r4d_shape_quant_act_i8 , nullptr, nullptr, r4d_rad_quant_act_i8_describe },

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

/* UNVALIDATED. The DPP row_xmask form of the fused all-reduce (the r4d_xor_lane exchanges in
 * r4d_ar_ln_had_quant_i8.hip) is compile-verified only: never measured and never
 * correctness-gated. */
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
  /* `all_reduce` + `rmsnorm_quant_fp8`, both wires -- the "byte-identical to the pair it replaces"
   * above, made executable. It is the one chain here that needs no assumption about which optional
   * operands were passed: its tail step carries the same two optionals in the same positions, so
   * absence propagates through the wiring. `ar_ln_had_quant_i8` gets no hook for the opposite
   * reason -- it reduces without the bf16 round its unfused pair stores. */
  r4d_fuse_ar_rmsnorm_quant_fp8 },

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
  r4d_shape_qk_norm_rope_gate, nullptr, nullptr, r4d_rad_qk_norm_rope_gate_describe },

/* The prologue and the fp8 cache store as one launch: a k head's block holds its token's rotated
 * k one element a lane already, so the store is two byte writes a lane on the same block. */
{ "qk_norm_rope_gate_kv_store_fp8", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into an E4M3 paged cache from the same launch",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] and "
  "one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, fp8-e4m3 cache, i32 slot mapping, scalar k_scale / v_scale descales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGateKv), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store,
  r4d_rad_qk_norm_rope_gate_kv_store_describe },

/* The same fold into a bf16 cache: the store is the two values the pair's kv_store_bf16 copies. */
{ "qk_norm_rope_gate_kv_store_bf16", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into a bf16 paged cache from the same launch",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] and "
  "one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, bf16 cache, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cQkNormRopeGateKvBf16), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store,
  r4d_rad_qk_norm_rope_gate_kv_store_describe },

{ "rope_table_f32", "rope_table", "attn",
  "the cos/sin plane the fused prologue reads, at each position's own row",
  "rot even and >= 2; the table is [max_ctx, rot] and only the rows `positions` names are written, "
  "so a row nothing addresses keeps whatever it held; a negative position writes nothing",
  "f32 or bf16 table, i32 or i64 positions; the angles are rope_bf16's exactly",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cRopeTable), NOTUNE,
  nullptr, nullptr, r4d_rad_rope_table, nullptr, nullptr, nullptr,
  r4d_shape_rope_table, nullptr, nullptr, r4d_rad_rope_table_describe },

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

{ "fulldrafter_two_stage_bf16", "fulldrafter_two_stage", "dflash",
  "FullDrafter discrete two-stage block drafter (K=12, 2x6) with Markov correction and Top-16 Selector",
  "M >= 1, dtype bf16",
  "bf16 activations, bf16 weights, i32 output tokens",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cFulldrafterTwoStage), NOTUNE,
  nullptr, nullptr, r4d_rad_fulldrafter_two_stage, r4d_rad_fulldrafter_two_stage_scratch,
  nullptr, nullptr, nullptr },

};

/* ================================================================================== the exports */

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libr4d",
    "0.5.0",
    "the RDNA4 (gfx1201) kernel library: paged and dense attention, gated delta net, P2P "
    "all-reduce, skinny and tiled quantised GEMM, and the rotate-and-quantise fusions",
    "gfx1201",
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

/* WHICH ROWS TOUCH NOTHING BUT THEIR OPERANDS AND SCRATCH (rad_abi.h). Nothing on an R4D launch
 * path allocates, and the exceptions are the kernels that keep something between launches on a
 * stream -- arrival counters and split workspaces a file allocates once and keys on the stream,
 * so that only the stream's own ordering keeps two launches from sharing them:
 *
 *   hc_* and mtp_enter  r4d_hc_bf16.hip's mix counters, one block a stream
 *   gemm_w4a8_tiled,    the split tail's workspace and counters, one a (device, stream)
 *   gemm_w8a8_tiled
 *   every collective    the peer flags and exchange buffers the all-reduces own
 *
 * gemm_fp8a8_nt_m16's finisher counters are keyed on the stream AND the output, so two launches
 * that could run at once never share them, and the row is concurrent. A row added with state of
 * its own that the stream's order protects belongs in this list. */
extern "C" int rad_kernel_concurrent(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k) return 0;
    if (k->family && std::strcmp(k->family, "ar") == 0) return 0;
    static const char* const kStateful[] = {
        "hc_", "mtp_enter", "ar_", "gemm_w4a8_tiled", "gemm_w8a8_tiled",
    };
    for (const char* p : kStateful)
        if (std::strncmp(k->name, p, std::strlen(p)) == 0) return 0;
    return 1;
}

/* ------------------------------------------------------------------------ tuned config access
 *
 * THE CHOSEN POINT ARRIVES AS ORDINARY PARAMETERS. The core resolved one instantiation of this
 * kernel, picked a point in the space declared above -- from the tune cache, from RADIANCE_TUNE,
 * or from each axis's own default -- and appended the choices to RadArgs::p. A launch reads `sk`
 * with the same rad_args_geti() it reads `M` with. There is no variant index anywhere.
 *
 * These read the tables above for the fallback, so the default is defined ONCE. A cold caller
 * (rad-kbench, a selftest) that supplies no tuned key gets exactly what an untuned install runs,
 * which is what makes a checker's answer mean something about a deployment. */
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

/* The tile axis names a COMPILED ARM, and (BM, BN) is a property of that arm rather than a second
 * axis -- see the note at tGemmTiled. This is the map, and it is the only place it exists: the N
 * constraint a call has to check is `N % BN`, and a wrong BN here would not fail, it would compute
 * a wrong answer in silence. */
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
        { 31, 128, 256 }, { 32, 128, 256 }, { 33, 128, 256 }, { 34, 128, 256 }, { 35, 128, 256 },
    };
    const Arm* t = prefill ? pf : tiled;
    const int  n = prefill ? NELEM(pf) : NELEM(tiled);
    for (int i = 0; i < n; ++i)
        if (t[i].v == arm) { out->variant = arm; out->bm = t[i].bm; out->bn = t[i].bn; return; }
    /* An arm the map does not know cannot be launched: fall to the default rather than guess a
     * tile width the kernel was not compiled for. The loader already refuses a default outside
     * its own value list, so arm 0 is always reachable. */
    out->variant = 0; out->bm = 128; out->bn = 256;
}

extern "C" void r4d_cfg_tiled(const RadArgs* a, R4dTileCfg* out) {
    tile_bmbn(TUNE(a, tGemmTiled, "tile"), 0, out);
}
extern "C" void r4d_cfg_prefill(const RadArgs* a, R4dTileCfg* out) {
    tile_bmbn(TUNE(a, tGemmPrefill, "tile"), 1, out);
}
extern "C" void r4d_cfg_split(const RadArgs* a, R4dSplitCfg* out) {
    out->splits = TUNE(a, tAttnSplit, "splits");
}
extern "C" void r4d_cfg_ar(const RadArgs* a, R4dArCfg* out) {
    out->nblocks  = TUNE(a, tAr, "nblocks");
    out->nthreads = TUNE(a, tAr, "nthreads");
    out->drain    = TUNE(a, tAr, "drain");
    out->acq      = TUNE(a, tAr, "acq");
    out->pub      = 1;                 /* not an axis; see the note at tAr */
}
extern "C" void r4d_cfg_topk(const RadArgs* a, R4dTopkCfg* out) {
    out->rc  = 0;                      /* derived from pt1 at issue; see the note at tTopk */
    out->nch = TUNE(a, tTopk, "nch");
    out->pt1 = TUNE(a, tTopk, "pt1");
}
extern "C" void r4d_cfg_mv(const RadArgs* a, R4dMvCfg* out) {
    out->rb = TUNE(a, tMv, "rb");
}
