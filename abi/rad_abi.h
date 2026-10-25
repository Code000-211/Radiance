/* rad_abi.h -- the plugin ABI. This file and rad_types.h are the contract between the engine and
 * an .so it loads. The interface is plain C: C++ across an .so boundary invites ABI mismatches,
 * and this is the boundary where users substitute their own builds. See spec.md §2.
 */
#ifndef RAD_ABI_H
#define RAD_ABI_H

#include "rad_types.h"
#include "rad_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever anything a plugin compiles against changes shape: this header, rad_types.h,
 * rad_builder.h, rad_runtime.h, rad_device.h, rad_encoding.h, rad_quant.h or rad_sample.h. Every
 * kind of plugin reports it, and a plugin that does not report exactly this version is refused at
 * load with both versions printed. A new optional export (e.g. rad_plugin_reference) is not a
 * change of shape: a plugin without it gets the default. */
#define RAD_ABI_VERSION 27u

/* ================================================================== plugin identity */
/* One value per plugin, not flags (QUANT is 3). RAD_PLUGIN_QUANT is a quantiser plugin:
 * rad_quant.h is its whole contract. */
enum { RAD_PLUGIN_KERNEL = 1, RAD_PLUGIN_ARCH = 2, RAD_PLUGIN_QUANT = 3 };

typedef struct RadPluginInfo {
    uint32_t    kind;        /* one of RAD_PLUGIN_KERNEL, _ARCH, _QUANT */
    const char* name;        /* "libr4d", "libref", "qwen3_dense_bf16" */
    const char* version;
    const char* description;
    /* The devices this build can run on, spelled as RadDeviceProps::arch. "host" for a library of
     * CPU kernels, which run anywhere; "" for a plugin that claims nothing. A binary built for
     * several architectures lists them separated by spaces or commas. */
    const char* build_target;
} RadPluginInfo;

/* Every plugin exports these two. */
uint32_t             rad_plugin_abi_version(void);
const RadPluginInfo* rad_plugin_info(void);

/* ================================================================== execution domain */
/* A CPU kernel is an ordinary kernel with domain RAD_DOMAIN_HOST, resolved through the same
 * hierarchy and constraints. Declare resolves both domains where both exist, so the core can
 * change execution site at runtime without re-resolving (spec §2.1). */
enum { RAD_DOMAIN_DEVICE = 0, RAD_DOMAIN_HOST = 1, RAD_N_DOMAINS = 2 };

/* ================================================================== kernel arguments */
typedef struct RadArgs {
    const RadTensor* t;             /* operands, positional, in the op schema's order */
    int              n_t;
    const RadParam*  p;             /* the resolved geometry: every param, ranges collapsed */
    int              n_p;
    void*            scratch;       /* device (or host) scratch, >= the scratch hook's answer */
    int64_t          scratch_bytes;
    void*            instance;      /* whatever init() returned for this resolved instance */
    int              rank;          /* tensor-parallel rank and world size, for collectives */
    int              world_size;
    /* The step's phase, one of RAD_PHASE_* (rad_runtime.h), copied from RadBatch::phase for this
     * launch: a kernel that measures or sizes itself per phase takes it from here rather than
     * inferring it from the row count or from which op the architecture happened to issue. */
    int              phase;
} RadArgs;

/* Read a parameter by key. Inline definitions, because a plugin links only the ABI headers and
 * nothing of the engine's. See the note in rad_types.h. */
static inline int rad_args_geti(const RadArgs* a, const char* key, long long* out) {
    if (!a || !key) return 0;
    for (int i = 0; i < a->n_p; ++i)
        if (a->p[i].kind == RAD_P_INT && strcmp(a->p[i].key, key) == 0) {
            if (out) *out = a->p[i].ival;
            return 1;
        }
    return 0;
}

/* Null if absent. */
static inline const char* rad_args_gets(const RadArgs* a, const char* key) {
    if (!a || !key) return 0;
    for (int i = 0; i < a->n_p; ++i)
        if (a->p[i].kind == RAD_P_STR && strcmp(a->p[i].key, key) == 0) return a->p[i].sval;
    return 0;
}

/* ================================================================== kernel hooks */
/* init/fini hold per-instance state that is set up once rather than per launch: libr4d's
 * all-reduce exchanges peer signal buffers and IPC handles across ranks before the first call,
 * and a kernel that JITs or loads a config table does it here, off the hot path.
 *
 * init is called once per resolved instance (one op, one band, one domain), after selection and
 * before load, with the frozen geometry. The pointer it returns (or null) is handed back in
 * RadArgs.instance. A negative status aborts startup. */
typedef int     (*RadInitFn)   (const RadParam* p, int n_p, int rank, int world_size,
                                void** out_instance);
typedef void    (*RadFiniFn)   (void* instance);

/* The launch. Must not allocate and must not synchronise. Returns RAD_OK or negative; a negative
 * return aborts the step and fails its requests, naming the kernel and the geometry. A launch
 * handed `link_gate` may return RAD_LINK_ENDED instead of RAD_OK (rad_kernel_collective). */
typedef int     (*RadLaunchFn) (const RadArgs* a, RadStream stream);

/* Bytes of scratch one launch of this shape needs. Null means none. Called at declare with the
 * band's upper bound, so the arena is sized for the worst step. */
typedef int64_t (*RadScratchFn)(const RadArgs* a);

/* ================================================================== weight layout
 *
 * What a kernel stores for a weight, given the weight's encoding. spec.md §4.1, §4.3.
 *
 * A container holds each weight's encoding (codes, scales and zero points as canonical planes,
 * rad_encoding.h) and nothing shaped for a kernel. The kernel that reads a weight describes the
 * form it wants here, and the loader converts when the model loads. The conversion is lossless: a
 * relayout moves bits and derives only what a reader of the planes could derive (a fragment order,
 * a scale repeated into a row's tail, a reference exponent that is a row's maximum). Choosing
 * numbers is the quantiser's job (rad_quant.h); a kernel doing it here would make the served
 * numbers depend on which library loaded the container. So a kernel library may change any layout
 * without containers being converted again. */
typedef struct RadLayout {
    const char* tag;          /* stable name of the arrangement, for messages and for checking
                               * that every kernel reading one weight wants the same bytes */
    uint32_t    dtype;        /* the stored dtype, possibly a plugin-private one */
    uint32_t    rank;
    int64_t     shape[RAD_MAX_RANK];   /* the stored shape, which may differ from the logical one */
    int64_t     align;        /* byte alignment the stored base must satisfy */
    int64_t     bytes;        /* total, including any padding the layout implies */
} RadLayout;

/* Describe what this kernel stores for operand `operand` of the given geometry.
 *
 * `enc` is the weight's encoding and `sel[i]` the index into `enc->plane` of planes[i], the planes
 * this operand takes, in RadWeightDecl::planes order. Each plane is this rank's slice, dense: rank
 * 2, [rows, cols] in the plane's own elements, every row starting on a byte and
 * rad_dtype_bytes(dtype, cols) long. Here the planes carry their geometry and a null `data`.
 *
 *   RAD_OK             `out` describes the stored form, and `relayout` will produce it.
 *   RAD_E_UNSUPPORTED  stored as is: the one selected plane, bytes unchanged. This is most
 *                      weights, and what a kernel with no hook gets. Only such a weight can be
 *                      read where it lies in the container (the file and mapped tiers).
 *   RAD_E_DTYPE        the kernel does not read this encoding (an fp8 GEMM handed int4, a rotated
 *                      weight to a kernel that does not rotate its activation). A declare-time
 *                      error naming the weight, the encoding and the kernel.
 *   RAD_E_SHAPE        the selected planes, or their extents, are not what the kernel reads. */
typedef int (*RadLayoutFn)(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                           const int* sel, const RadTensor* planes, int n_planes,
                           RadLayout* out);

/* Write the stored form `layout` described into `dst`, `dst_bytes` long, from the planes, which
 * now carry their bytes. Pure host code, run by the loader on a pool behind the container read,
 * so it has to keep pace with the disk. */
typedef int (*RadRelayoutFn)(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                             const int* sel, const RadTensor* planes, int n_planes, void* dst,
                             int64_t dst_bytes);

/* The inverse: recover the planes from a stored form. `planes` arrive with their geometry and a
 * buffer each.
 *
 * A tool checking a kernel against the oracle holds the stored bytes, but libref reads canonical
 * planes; given the stored bytes it answers a permuted question, and the comparison reports an
 * error the size of two unrelated tensors against a correct kernel. Only the plugin knows its
 * permutation. A kernel that relayouts without an inverse is not at fault, only not checkable
 * against a dense oracle, and the tools report it by name. */
typedef int (*RadUnrelayoutFn)(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const void* src, int64_t src_bytes,
                               const RadTensor* planes, int n_planes);

/* ================================================================== operand description */
/* How big operand i of an op is, and what values it may hold.
 *
 * A schema names operands and their roles but not their extents, which are in general not
 * derivable from the parameters (the run phase supplies them at issue). A tool that calls an op
 * cold, such as a correctness check or a benchmark, asks the kernel through this hook instead of
 * keeping its own table of ops.
 *
 * The fill domain matters as much as the shape. A gated-delta-net `g` is an intra-chunk cumulative
 * sum of log-decays: non-increasing and never positive, since each decay e^{g_i - g_j} with j <= i
 * is <= 1 only because g is monotone. Fed normals, the oracle overflows to 2e38 and the comparison
 * measures the test rather than the kernel. The same holds for a sigmoid gate, a lower-triangular
 * factor and every index operand: a KV slot indexes n_blocks * block_size positions and a
 * block-table entry n_blocks, and a draw outside that range tests out-of-range handling instead. */
enum {
    RAD_FILL_NORMAL      = 0,  /* standard normals -- the default for a float operand */
    RAD_FILL_SIGMOID     = 1,  /* (0, 1): a gate */
    RAD_FILL_GATE_CUMSUM = 2,  /* non-increasing and <= 0, reset every `fill_chunk` rows */
    RAD_FILL_TRI_LOWER   = 3,  /* zero above the diagonal within a `fill_chunk` block */
    RAD_FILL_INDEX       = 4,  /* integers; see idx_max / idx_const / the IDX flags */
    /* Unit L2 norm along the last dimension. The gated-delta-net key is normalised by qk_norm
     * before the solve, which inverts I + tril(beta k k^T, -1). With unit normals over a 128-wide
     * head the off-diagonals are ~100x too large, the matrix is ill-conditioned, and two
     * summation orders disagree by its condition number. Like SIGMOID and GATE_CUMSUM, this puts
     * the operand in the domain the op is actually given. */
    RAD_FILL_UNIT_ROW    = 5,
    /* The samplers' `params` plane: [rows, 24] 32-bit words, each row a RadSampleParams
     * (rad_sample.h) as a request sets it, its flags agreeing with its fields. Drawn as numbers,
     * the integer fields and `flags` are bit patterns (a window is all or nothing, a mask row is
     * never in range, each flag is a coin toss) and XTC's threshold is past any probability, so
     * the mask and XTC stages do nothing and the others act on a random part of the rows.
     * Row 0 switches every stage on; every other row switches each stage on or leaves it at its
     * identity at random, and a minority is greedy. `fill_chunk` is the width of the op's history
     * plane (0 when it has none: the window is then empty) and `idx_max` the row count of its
     * bitmask plane (0 when it has none: `mask_row` is then -1). A tool that does not know the
     * value draws normals. */
    RAD_FILL_SAMPLE_PARAMS = 6,
};

enum {
    /* An optional operand this geometry does not pass. One entry per schema operand is still
     * emitted, since that count catches a hook that drifted from its schema; a null
     * RadTensor.data marks the absence. */
    RAD_OPD_F_ABSENT     = 1u << 0,
    /* A scatter destination: no value may repeat. Two tokens on one slot make the store a race
     * that two implementations resolve differently. A read-side index (a gather row, a
     * block-table entry) may repeat. */
    RAD_OPD_F_IDX_UNIQUE = 1u << 1,
    /* Cumulative sequence lengths, [0, ..., T]: the batch's shape, not a draw. */
    RAD_OPD_F_IDX_CU     = 1u << 2,
    /* A routing histogram's offsets, [0, ..., idx_max - 1]: non-decreasing, first 0, last the row
     * count, each bucket a run of random length (moe_scatter's `expert_offset`). A uniform draw is
     * a malformed histogram that two correct implementations may read differently. Additive:
     * RAD_ABI_VERSION does not move, and a tool that does not know the flag draws as before. */
    RAD_OPD_F_IDX_PREFIX = 1u << 3,
    /* A token history, [rows, width] ids in [0, idx_max): each row repeats itself as generated
     * text does, drawing on a few ids spread over the range and copying runs of its own earlier
     * tokens. A uniform draw over a vocabulary never repeats, so a frequency penalty never counts
     * past one and DRY (which penalises the token that would extend a repeated suffix) never
     * finds a repeat. Additive, as IDX_PREFIX: a tool that does not know the flag draws uniform
     * ids. */
    RAD_OPD_F_IDX_HISTORY = 1u << 4,
};

typedef struct RadOpdDesc {
    uint32_t dtype;
    uint32_t rank;
    int64_t  shape[RAD_MAX_RANK];

    uint32_t fill;         /* RAD_FILL_* */
    /* The reset width for GATE_CUMSUM and TRI_LOWER; the history plane's width for
     * SAMPLE_PARAMS. */
    int64_t  fill_chunk;

    /* RAD_FILL_INDEX only, except that SAMPLE_PARAMS reads `idx_max` as its bitmask plane's row
     * count. `idx_max` is the exclusive upper bound of the draw; 0 means the leading extent of the
     * previous operand, the common case for a gather. `idx_const` >= 0 sets every element to that
     * value instead of drawing. `seqused` needs it: a random context length shorter than the query
     * makes causal attention read a prefix that does not exist. */
    int64_t  idx_max;
    int64_t  idx_const;    /* -1 to draw */

    uint32_t flags;        /* RAD_OPD_F_* */
} RadOpdDesc;

/* Fill `out` for operand `operand` of this op at this geometry. `p`/`n_p` is the resolved
 * parameter list, exactly as `launch` would receive it. Returns RAD_OK, or RAD_E_UNSUPPORTED when
 * this geometry lacks what the description needs; a caller skips and counts that instead of
 * guessing an extent.
 *
 * Pure host code, no device memory, no side effects: it is called before anything is allocated. */
typedef int (*RadShapeFn)(const RadParam* p, int n_p, int operand, RadOpdDesc* out);


/* ================================================================== tunable instantiation */
/* A kernel is a family: tile sizes, split counts, waves per block. Which member is fastest
 * depends on the shape and the machine and must be measured there (spec §15). A library declares
 * the axes here and the core takes their cross product. Axes rather than a list of named
 * combinations, because a hand-enumerated list has holes (a shape with 40 rows of work runs a
 * 64-row tile), and because it keeps the tuning policy out of the launch adapter, where an
 * out-of-tree author could not write it.
 *
 * The chosen values arrive as ordinary parameters appended to RadArgs::p, with no variant index:
 * the kernel reads `sk` with the same rad_args_geti() it reads `M` with. */
typedef struct RadTunable {
    const char*    key;       /* the kernel's own name for the axis: "sk", "bm", "npw" */
    const int64_t* values;    /* legal values, ascending; the core tries exactly these */
    int            n_values;
    /* The value used when this machine has no measurement for this instantiation (a fresh
     * install, an untuned shape, a cache from another card). There is no fallback beneath it, so
     * it must be launchable: the loader checks it appears in `values` and refuses the plugin by
     * name if not. */
    int64_t        deflt;
    const char*    doc;       /* what the axis does, for the tuner's report */
} RadTunable;

/* Whether a combination is legal at this geometry. Cross-axis rules live here because independent
 * axis lists cannot express them: libr4d's skinny GEMM needs `wv * sk * 32` inside the block bound
 * and `K % (sk * unit) == 0`.
 *
 * `p` is the resolved geometry, `choice` the candidate as key/value parameters. Pure host code,
 * called once per candidate before anything is allocated.
 *
 * Returns 1 for legal and 0 for skip, which is not an error: a kernel may serve a subset of its
 * cross product, and the tuner reports the candidate as skipped. Null means every combination is
 * legal. */
typedef int (*RadTuneValidFn)(const RadParam* p, int n_p, const RadParam* choice, int n_choice);
/* ================================================================== op schemas */
/* A parameter is supplied by the caller (REQUIRED), may be supplied (OPTIONAL), or is supplied by
 * the kernel (DERIVED).
 *
 * DERIVED breaks a circularity. The paged block size comes from the resolved attention kernel, not
 * a core constant (spec §7.2; libr4d is compiled for 16). But a constraint whose key the geometry
 * does not carry does not hold (which is also why libr4d's two-shot all-reduce rows must precede
 * the one-shot ones), so a kernel constrained to block_size == 16 could never be selected by a
 * query that omits block_size.
 *
 * So constraints on a DERIVED parameter are skipped during matching, and the winning kernel's
 * RAD_C_EQ value for that key is written into the resolved geometry. A caller that supplies one
 * pins it and it matches normally, so a forced block size is refused by name rather than silently
 * substituted. A kernel with no EQ on a derived key leaves it unset, and rad_kv_block_size()
 * reports that rather than returning zero. */
enum { RAD_OPTIONAL = 0, RAD_REQUIRED = 1, RAD_DERIVED = 2 };

/* What a parameter means to the core. The core carries almost every parameter opaquely, which is
 * what lets a third party add an op. The exception is the sequence chunk: a linear-state scan is
 * compiled for one tile length, and the scheduler must cut its steps on a multiple of it or a
 * chunk straddles a sequence boundary. The schema names that role so the core does not have to
 * find the parameter by guessing its spelling. */
/* RAD_PROLE_CAPACITY marks a count the kernel is handed as a bound and never writes past its
 * operands for: it launches, loops and writes by the extents of the tensors it is given. That lets
 * the activation arena be declared again at a smaller step (RadBuildCtx::shape_probe) while the
 * kernel keeps the geometry it was selected with, including the real declare's value here. It is
 * the kernel's claim to make; without it, a fixed parameter that differs between the two declares
 * refuses the smaller size. */
enum { RAD_PROLE_NONE = 0, RAD_PROLE_SEQ_CHUNK = 1, RAD_PROLE_CAPACITY = 2 };

typedef struct RadParamSpec {
    const char* key;
    int         type;      /* RAD_P_INT | RAD_P_STR */
    int         required;
    int         role;      /* RAD_PROLE_*; 0 for every parameter the core does not interpret */
} RadParamSpec;

/* Positional operand roles. The core needs to know only which operands are weights (issue resolves
 * a handle and waits on the mover) and which are written (the buffer planner computes liveness). */
/* RAD_OPD_WTAB is a weight operand whose issue names a run of weights (a MoE layer's experts);
 * the kernel is handed a device array of pointers. It is a separate role, not a flag on
 * RAD_OPD_WEIGHT, so a tool reading the schema knows the operand is not one plane it can describe
 * with one shape: see RAD_WTAB in rad_runtime.h. */
enum { RAD_OPD_IN = 0, RAD_OPD_OUT = 1, RAD_OPD_INOUT = 2, RAD_OPD_WEIGHT = 3,
       RAD_OPD_WTAB = 4 };

typedef struct RadOperandSpec {
    const char* name;      /* for diagnostics: "x", "w_gate", "out" */
    int         role;
    int         optional;  /* an absent optional operand is a null RadTensor.data */
} RadOperandSpec;

typedef struct RadOpSchema {
    const char*           op;
    const RadParamSpec*   params;   int n_params;
    const RadOperandSpec* operands; int n_operands;
    const char*           doc;
} RadOpSchema;


/* ================================================================== declared fusions
 *
 * Which chain of conventional ops a kernel computes.
 *
 * This is a declaration, not a rewriter. spec.md §2.3 rules out a fusion rewriter, and the
 * declare phase keeps nothing one could work on: `Program` holds no operand edges, and the Builder
 * collapses each buffer's use-set to a `first_def`/`last_use` pair with reads and writes merged.
 * "Op A's output is op B's only input" is not recoverable from that, which is also why
 * `--debug-graph` prints no edge it cannot prove. Fusion stays a selection: the architecture asks
 * for the fused op and emits the sequence when nothing serves it (rad_op_resolved in
 * rad_builder.h is the idiom for a banded or optional fusion).
 *
 * The declaration makes the equivalence machine-readable, which gives:
 *   - an oracle: a fused kernel with no libref counterpart is checked against its chain run on the
 *     same plugin's unfused rows, and the answer should be byte-identical;
 *   - an advisory in --debug-graph: these declared ops form a chain some kernel can run in one
 *     launch, and why it did not fire;
 *   - the per-library fusion-claim count in rad-kbench's report.
 *
 * It is a hook rather than a table because a step's geometry is not always derivable from the
 * fused op's: `qk_norm_rope_gate` replaces an `rmsnorm` at M = T * n_head, a product that is not
 * in its parameter list.
 *
 * Operand wiring: every operand of every step is either an operand of the fused kernel or a
 * temporary passed between steps. A temporary needs no description here; the step op's own
 * `opd_shape` sizes and types it. Often none is needed: the fp8 folds still write their
 * intermediate when `out_bf16` is passed, so the fused and unfused forms leave the same buffers. */
enum {
    RAD_FUSE_NONE = -1,          /* this step does not pass that optional operand */
};
/* Temporary `t`, for t in [0, RAD_MAX_FUSE_TMP). Encoded so that a non-negative value is always an
 * operand of the fused kernel and needs no tagging. */
#define RAD_FUSE_TMP(t) ((int16_t)(-2 - (t)))
#define RAD_FUSE_IS_TMP(v) ((v) <= -2)
#define RAD_FUSE_TMP_INDEX(v) ((int)(-2 - (v)))

enum { RAD_MAX_FUSE_OPD = 16, RAD_MAX_FUSE_PARAMS = 12, RAD_MAX_FUSE_TMP = 8 };

typedef struct RadFuseStep {
    const char* op;                            /* a conventional op name -- see docs/OPS.md */
    RadParam    p[RAD_MAX_FUSE_PARAMS];        /* this step's geometry, not the fused one's */
    int         n_p;
    /* One entry per operand of `op`, in its schema's order: an operand index of the fused kernel,
     * RAD_FUSE_TMP(t), or RAD_FUSE_NONE. The count is checked against the step op's schema, so a
     * hook that drifts from the vocabulary is a diagnosable error (as with RadShapeFn). */
    int16_t     from[RAD_MAX_FUSE_OPD];
    int         n_from;
} RadFuseStep;

/* Describe step `step` of the chain this kernel replaces, for this geometry and variant.
 *
 *   RAD_OK             `out` is filled; ask again with step + 1.
 *   RAD_E_NOTFOUND     `step` is past the end. This is how a caller learns the chain's length,
 *                      which may differ with the geometry (an `rmsnorm_quant_fp8` with a residual
 *                      replaces `rmsnorm_add`, without one `rmsnorm`).
 *   RAD_E_UNSUPPORTED  this geometry or variant has no unfused equivalent in the vocabulary.
 *
 * A chain must be exact. A kernel that is merely close to a sequence publishes no hook, because
 * callers substitute one for the other and compare the bytes. */
/* `present` has bit i set when operand i of the fused op is passed. The chain can depend on an
 * operand, not only on a parameter: `rmsnorm_quant_fp8` replaces `add` -> `rmsnorm` ->
 * `quant_act_fp8` given a residual and `rmsnorm` -> `quant_act_fp8` without. Inferring presence
 * from the plugin's own RadShapeFn is right for a checker calling the kernel cold but wrong for a
 * caller holding a real issue, such as the graph dump. `RAD_FUSE_NONE` cannot express it, since
 * the operands that vary are the step's required ones.
 *
 * RAD_FUSE_PRESENT_ALL means the caller does not know: answer for every operand your RadShapeFn
 * says exists. */
#define RAD_FUSE_PRESENT_ALL ((uint64_t)~0ull)
#define RAD_FUSE_HAS(mask, i) (((mask) >> (i)) & 1ull)

typedef int (*RadFuseFn)(const RadParam* p, int n_p, uint64_t present, int step,
                         RadFuseStep* out);

/* Optional: print this instance's own telemetry, given what the caller measured for the op it
 * implements. `phase_ms[0..n_phase)` is the caller's timing for this instance's op in a step of
 * that phase (indexed by RAD_PHASE_*), -1.0 for a phase the op never ran in. A kernel that counts
 * something prints it here, where its counters and the caller's timing meet in one line; null --
 * the common case -- keeps nothing and prints nothing. A kernel may also print from its `fini`,
 * which is the shutdown dump: it has the counters but not the timing. */
typedef void (*RadStatsDumpFn)(void* instance, const double* phase_ms, int n_phase);

/* ================================================================== kernel rows */
typedef struct RadKernelInfo {
    const char* name;       /* entry point, unique within the plugin */
    const char* op;         /* the op it implements -- the key callers select on */
    const char* family;     /* attn | gdn | ar | gemm | norm | sample | ... , for grouping only */
    const char* computes;   /* prose, for a human */
    const char* shape;      /* the geometry it is compiled for, prose */
    const char* dtypes;

    int         domain;     /* RAD_DOMAIN_DEVICE | RAD_DOMAIN_HOST */
    int         priority;   /* higher wins within this plugin; ties break on declaration order */

    const RadConstraint* constraints;  int n_constraints;

    /* The axes of this kernel's instantiation space and the cross-axis legality test. Null/0
     * means one implementation with nothing to tune (the common case): the core appends no
     * choices and the kernel reads no tuned key. See RadTunable. */
    const RadTunable*    tunables;     int n_tunables;
    RadTuneValidFn       tune_valid;

    RadInitFn      init;
    RadFiniFn      fini;
    RadLaunchFn    launch;
    RadScratchFn   scratch;
    RadLayoutFn    layout;
    RadRelayoutFn  relayout;
    /* How to size and fill each operand, for a tool that calls this kernel cold. Null means tools
     * that need a buffer for it skip it by name. libref supplies it for the whole conventional
     * vocabulary, so an op in docs/OPS.md is describable even when the kernel under test is not. */
    RadShapeFn     opd_shape;
    /* The inverse of `relayout`, for a tool holding the stored bytes rather than the planes. Null
     * means the stored form cannot be read back, so a dense oracle has nothing to compare
     * against: a named skip, not a failure. */
    RadUnrelayoutFn unrelayout;
    /* The chain of conventional ops this kernel computes in one launch, or null if it is not a
     * fusion. Non-null also counts in rad-kbench's fusion-claim column.
     *
     * Kernel tables are positional aggregate initialisers
     * (`{..., nullptr, nullptr, r4d_rad_embed_lookup, nullptr, ...}`), so a field inserted
     * anywhere else shifts every later initialiser, and since `nullptr` converts to any function
     * pointer type, part of that shift would still compile. Append new fields at the end. */
    RadFuseFn      replaces;
    RadStatsDumpFn stats_dump;   /* see RadStatsDumpFn: null when the kernel keeps no counters */
} RadKernelInfo;

/* ================================================================== kernel plugin exports */
/* Every kernel plugin exports the whole set. The first plugin in hierarchy order to declare an op
 * fixes its schema; a later plugin declaring the same op name with a different schema is refused
 * at load with both plugins named. Arguments are positional, so disagreeing schemas would make
 * the same call mean different things depending on which plugin won selection, silently
 * (spec §2.3). */
int                  rad_kernel_schema_count(void);
const RadOpSchema*   rad_kernel_schema_at(int i);
int                  rad_kernel_count(void);
const RadKernelInfo* rad_kernel_at(int i);

/* Optional. Called once after the plugin is loaded and before any selection, and once at
 * shutdown. A plugin with no global state need not export them.
 *
 * The symbol names are `rad_plugin_open` and `rad_plugin_close`; the types exist so a plugin can
 * declare them with the right signature. The loader resolves those exact names, so a plugin that
 * spells them differently is silently never initialised.
 *
 * `rad_plugin_open` returning RAD_E_UNSUPPORTED declines this machine: the plugin is unloaded and
 * the rest of the installation carries on, so one installation can ship libraries for several
 * cards. Any other negative status refuses the plugin as an error. A kernel library need not
 * decline a card its device code was not built for: the loader reads that from the code objects
 * and leaves those device kernels out itself. */
typedef int  (*RadPluginOpenFn) (void);
typedef void (*RadPluginCloseFn)(void);

/* Optional promise about kernel `i` of rad_kernel_at: nonzero means every launch reads and writes
 * only its operands and the scratch it was handed (no allocation it keeps, no counter kept between
 * launches), so two launches may run at once on one device when their operands and scratch do not
 * overlap. The engine then lets such a launch start before independent earlier launches on its
 * queue finish. Without the symbol, or with 0, every launch is ordered behind everything before
 * it, which is always correct. Symbol name: `rad_kernel_concurrent`. */
typedef int  (*RadKernelConcurrentFn)(int i);

/* Optional promise about kernel `i` of rad_kernel_at: nonzero means it is a collective. Its
 * launches exchange data with the other ranks' cards, so the interconnect carries their bytes
 * while they run and the step waits on it. The engine keeps its own bulk traffic on those links
 * (weights streamed to the card during compute) out of the way for that time. Without the symbol,
 * or with 0, only the conventional `all_reduce` op is treated as a collective. Symbol name:
 * `rad_kernel_collective`.
 *
 * 2 promises more: each launch ends the link hold itself as soon as the last byte the other ranks
 * send this card for it has landed, so a kernel that reduces after the exchange frees the link for
 * the rest of its run. The engine then passes each launch it holds the link for the parameter
 * `link_gate`, the address of a 64-bit word the kernel reaches at system scope; a launch without
 * it holds nothing. The kernel reads the word on the device as it starts (not on the host in the
 * launch call, where the hold is not yet in effect) and, once those bytes have landed, decrements
 * it with a system-scope atomic if it was odd; an even word it leaves alone. A launch that
 * decremented returns RAD_LINK_ENDED. Any other success returns RAD_OK and the engine ends the
 * hold behind the launch, so a path that ignores the word is always safe. */
typedef int  (*RadKernelCollectiveFn)(int i);
#define RAD_LINK_ENDED 1

/* Optional: nonzero declares this kernel library a reference implementation, written to be
 * plainly correct rather than fast. For any library that says so, whatever its name:
 *   - the loader ranks it below every unnamed kernel library, because a reference matches every
 *     op it declares and would otherwise win over the library serving the card (naming it in
 *     --kernels still places it where named, and a --kernels list that omits it leaves it out);
 *   - the engine refuses to serve a model with an op only a reference library implements, unless
 *     given --debug-accept-reference-kernels;
 *   - rad-kbench checks every other library against it.
 * Without the symbol, or with 0, it is an ordinary library. Symbol name: `rad_plugin_reference`.
 *
 * It is a whole-library answer, because a library is usually one implementation of one thing. A
 * plugin that ships a device row AND the host oracle beside it is not: marking the library would
 * mark its device rows too, and the engine would refuse to serve an op that has a real kernel.
 * Such a plugin exports `rad_kernel_reference` below and leaves this one 0. */
typedef int  (*RadPluginReferenceFn)(void);

/* Optional, per row: nonzero says kernel `i` of rad_kernel_at is that row's reference
 * implementation. Overrides `rad_plugin_reference` for the rows it is exported for, so one
 * library can declare a device row (0) and its host oracle (1) side by side. A plugin that
 * exports neither leaves every row classified by `rad_plugin_reference`. Symbol name:
 * `rad_kernel_reference`. */
typedef int  (*RadKernelReferenceFn)(int i);

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_ABI_H */
