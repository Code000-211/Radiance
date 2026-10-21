/* rad_abi.h -- the plugin ABI. This file and rad_types.h are the whole contract between the
 * engine and an .so it loads. Everything is C: C++ across an .so boundary is a standing
 * invitation to a mismatch that manifests as corruption three frames later, and this boundary is
 * exactly where a user is expected to substitute their own build. See spec.md §2.
 */
#ifndef RAD_ABI_H
#define RAD_ABI_H

#include "rad_types.h"
#include "rad_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever anything a plugin compiles against changes shape -- this header, rad_types.h,
 * rad_builder.h, rad_runtime.h, rad_device.h, rad_encoding.h, rad_quant.h or rad_sample.h. Every
 * kind of plugin reports it. A plugin that does not report exactly this is refused at load, by
 * name, with both versions printed. An OPTIONAL export added beside the existing ones
 * (rad_plugin_reference, say) is not a change of shape: a plugin without it is read as answering
 * the default. */
#define RAD_ABI_VERSION 16u

/* ================================================================== plugin identity */
/* RAD_PLUGIN_QUANT is a quantiser plugin: rad_quant.h is its whole contract. */
enum { RAD_PLUGIN_KERNEL = 1, RAD_PLUGIN_ARCH = 2, RAD_PLUGIN_QUANT = 3 };

typedef struct RadPluginInfo {
    uint32_t    kind;        /* RAD_PLUGIN_KERNEL | RAD_PLUGIN_ARCH | RAD_PLUGIN_QUANT */
    const char* name;        /* "libr4d", "libref", "qwen3_dense_bf16" */
    const char* version;
    const char* description;
    /* The devices this build of the plugin can run on, as the backend spells an architecture in
     * RadDeviceProps::arch. "host" for a library of CPU kernels, which run anywhere; "" for a
     * plugin that claims nothing. Several architectures may be listed, separated by spaces or
     * commas, for a binary built to cover more than one. */
    const char* build_target;
} RadPluginInfo;

/* Every plugin exports these two. */
uint32_t             rad_plugin_abi_version(void);
const RadPluginInfo* rad_plugin_info(void);

/* ================================================================== execution domain */
/* A CPU kernel is an ordinary kernel with domain RAD_DOMAIN_HOST, resolved through the same
 * hierarchy and the same constraints. Declare resolves both domains where both exist, so the
 * core can change execution site at runtime without re-resolving -- spec §5.1. */
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
} RadArgs;

/* Read a parameter by key. Definitions, not declarations: a plugin links the ABI headers and
 * nothing of the engine's, so a function declared here and defined in the core would be
 * unlinkable from the plugins that are its only callers. See the note in rad_types.h. */
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
/* init/fini exist because some kernels have per-instance state that must be established once and
 * not per launch -- libr4d's all-reduce needs its peer signal buffers and IPC handles set up
 * across ranks before the first call, and a kernel that JITs or loads a config table wants
 * somewhere to do it that is not the hot path.
 *
 * init is called once per RESOLVED INSTANCE (one op, one band, one domain), after selection and
 * before load, with the frozen geometry. It returns an opaque pointer handed back in
 * RadArgs.instance, or null. A negative status aborts startup. */
typedef int     (*RadInitFn)   (const RadParam* p, int n_p, int rank, int world_size,
                                void** out_instance);
typedef void    (*RadFiniFn)   (void* instance);

/* The launch. Must not allocate and must not synchronise. Returns RAD_OK or negative; a negative
 * return aborts the step and fails the requests in it, naming the kernel and the geometry. */
typedef int     (*RadLaunchFn) (const RadArgs* a, RadStream stream);

/* ================================================================== the described launch */
/* A KERNEL THAT DESCRIBES ITS LAUNCH INSTEAD OF PERFORMING IT, so the runtime can turn one
 * description into either a dispatch or a graph node update.
 *
 * WHY THIS EXISTS. The engine re-issues a fixed op sequence every step: the topology is identical
 * from step to step and only a few of the dispatches carry an operand that moved. The cost of
 * launching a graph is roughly flat in the number of nodes, while submitting the same dispatches
 * one at a time is linear in it, so replaying such a sequence as a graph is cheaper. What a
 * `launch` hook cannot support is the UPDATE: it performs the dispatch itself, marshalling its own
 * arguments, so nothing above it knows which argument slot holds which operand -- and a node update
 * needs exactly that slot. `describe` reports the dispatch instead of performing it.
 *
 * IT IS OPTIONAL. `launch` remains the required entry point and every kernel that only implements
 * it keeps working unchanged -- it simply never becomes a graph node, and a sequence containing
 * one is submitted the ordinary way. Implement `describe` for a kernel that is issued repeatedly
 * with a stable shape; there is nothing to gain on one that runs once.
 *
 * THE TRAP, AND THE ONE WAY TO AVOID IT. `launch` and `describe` must dispatch the SAME kernel
 * with the SAME arguments, and nothing can check that for you: a divergence is not a crash, it is
 * a wrong number produced only on the graph path, only after a capture, and it will read as a
 * numerics bug anywhere but here. Do NOT write the two bodies twice. Write the shape arithmetic
 * once and let it emit into a sink that is either a stream or a descriptor -- libr4d does this
 * with one macro, and an out-of-tree plugin should do the same.
 *
 * ALLOCATION-FREE, like `launch`. The runtime owns the RadLaunchDesc storage and keeps it alive
 * for as long as the node exists, so the shim writes its argument VALUES into `blob` and points
 * `params[i]` at them. Pointing `params[i]` at anything else is legal only if that storage
 * outlives the graph -- a string literal or the instance state does, a local does not. */

/* Fill `out[0..cap)` with the dispatches this call WOULD perform and write how many to `n_out`.
 * Most kernels emit one; a split-K kernel and its finisher emit two. Returns RAD_E_FULL without
 * writing anything if `cap` is too small, so the runtime can retry with a bigger array; returns
 * RAD_OK having written `*n_out` otherwise. Must not allocate, must not synchronise and must not
 * touch the device -- it is called at capture time, off the step path, and again whenever the
 * runtime needs to know whether an argument moved.
 *
 * IT DESCRIBES A KERNEL DISPATCH AND ONLY THAT. An op whose work is a memory copy -- libr4d's
 * `cast` is a hipMemcpy2DAsync, not a launch -- has nothing to put in a RadLaunchDesc and should
 * leave `describe` null. A graph can hold copy nodes, but expressing one would need a second
 * descriptor shape, and the ops that want it are the ops a graph saves least on.
 *
 * IT TAKES THE STREAM, because some launches depend on which one they are going onto: libr4d's
 * split-K GEMM takes its arrival counter from a per-stream pool, so a description made without
 * the stream would name a different counter than the launch would have used -- a divergence of
 * exactly the kind the paragraph above says nothing can check for you. The runtime passes the
 * stream the dispatch will actually run on, which for a graph node is the stream the graph is
 * launched on. */
typedef int     (*RadDescribeFn)(const RadArgs* a, RadStream s, RadLaunchDesc* out,
                                 int cap, int* n_out);

/* Bytes of scratch one launch of this shape needs. Null means none. Called at declare with the
 * band's upper bound, so the arena is sized for the worst step. */
typedef int64_t (*RadScratchFn)(const RadArgs* a);

/* ================================================================== weight layout
 *
 * WHAT A KERNEL STORES FOR A WEIGHT, GIVEN WHAT THE WEIGHT IS. spec.md §4.1, §4.3.
 *
 * A container holds each weight's ENCODING -- its codes, scales and zero points as canonical planes
 * (rad_encoding.h) -- and nothing shaped for a kernel. The kernel that reads a weight says here
 * what it wants instead, and the loader turns one into the other when the model loads. Lossless,
 * always: a relayout moves bits and derives nothing a reader of the planes could not derive -- a
 * fragment order, a scale repeated into a row's tail, a reference exponent that is a row's
 * maximum. Choosing numbers is a quantiser's job (rad_quant.h); a kernel that did it here would
 * make the served numbers depend on which library loaded the container. So a kernel library may
 * change every layout it has and no container is converted again. */
typedef struct RadLayout {
    const char* tag;          /* stable name of the arrangement, for messages and for checking
                               * that every kernel reading one weight wants the same bytes */
    uint32_t    dtype;        /* the stored dtype, possibly a plugin-private one */
    uint32_t    rank;
    int64_t     shape[RAD_MAX_RANK];   /* the STORED shape, which may differ from the logical one */
    int64_t     align;        /* byte alignment the stored base must satisfy */
    int64_t     bytes;        /* total, including any padding the layout implies */
} RadLayout;

/* Describe what this kernel stores for operand `operand` of the given geometry.
 *
 * `enc` is the weight's encoding and `sel[i]` the index into `enc->plane` of planes[i], the planes
 * this operand takes, in the order the declaration named them (RadWeightDecl::planes). Each plane
 * is THIS RANK'S SLICE, dense: rank 2, [rows, cols] in the plane's own elements, every row starting
 * on a byte and rad_dtype_bytes(dtype, cols) long. Here the planes carry their geometry and a null
 * `data`.
 *
 *   RAD_OK             `out` describes the stored form, and `relayout` will produce it.
 *   RAD_E_UNSUPPORTED  stored as it is: the one selected plane, its bytes unchanged -- what most
 *                      weights are, and what a kernel with no hook at all gets. Only such a weight
 *                      can be read where it lies in the container (the file and mapped tiers).
 *   RAD_E_DTYPE        THE KERNEL DOES NOT READ THIS ENCODING -- an fp8 GEMM handed int4, a
 *                      rotated weight to a kernel that does not rotate its activation. A
 *                      declare-time error that names the weight, the encoding and the kernel.
 *   RAD_E_SHAPE        the planes selected, or their extents, are not what the kernel reads. */
typedef int (*RadLayoutFn)(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                           const int* sel, const RadTensor* planes, int n_planes,
                           RadLayout* out);

/* Write the stored form `layout` described into `dst`, `dst_bytes` long, from the planes, which
 * now carry their bytes. Pure host code, run by the loader -- on a pool, behind the read of the
 * container, so it has to keep pace with the disk. */
typedef int (*RadRelayoutFn)(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                             const int* sel, const RadTensor* planes, int n_planes, void* dst,
                             int64_t dst_bytes);

/* ...AND BACK: recover the planes from a stored form. `planes` arrive with their geometry and a
 * buffer each.
 *
 * WHY THE INVERSE HAS TO EXIST, stated from the failure it prevents. A tool that checks a kernel
 * against the oracle holds the STORED bytes -- what the kernel reads -- and libref reads canonical
 * planes. Handed the stored bytes it computes a correct answer to a permuted question, and the
 * comparison is then between two unrelated tensors: a relative error of the size two independent
 * draws give, reported against a kernel that is in fact correct. Only the plugin knows its
 * permutation, so only the plugin can undo it. A kernel that relayouts and supplies no inverse is
 * not a fault -- it is simply not checkable against a dense oracle, and the tools say so BY NAME
 * rather than guessing. */
typedef int (*RadUnrelayoutFn)(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const void* src, int64_t src_bytes,
                               const RadTensor* planes, int n_planes);

/* ================================================================== operand description */
/* HOW BIG IS OPERAND i OF THIS OP, AND WHAT MAY BE IN IT.
 *
 * An op's schema names its operands and their roles but not their extents, because the extents
 * are not derivable from the parameters in general -- the run phase supplies them at issue. So a
 * tool that wants to CALL an op cold, which is what a correctness check and a benchmark both do,
 * has to size its buffers somehow. A hand-written table inside `tools/` cannot serve that: it
 * covers only the ops someone wrote a row for and skips the rest, it cannot be extended without
 * patching the engine, and it is a place where a tool knows something about an op.
 *
 * The kernel is the component that knows. This hook asks it.
 *
 * WHAT IS IN AN OPERAND MATTERS AS MUCH AS HOW BIG IT IS, and that is why this carries a fill
 * domain rather than only a shape. A gated-delta-net `g` is an intra-chunk cumulative sum of
 * log-decays: non-increasing and never positive, because every decay the scan forms is
 * e^{g_i - g_j} with j <= i and that is <= 1 only because g is monotone. Feed it normals and the
 * ORACLE overflows to 2e38 while the kernel does something else, and the comparison then measures
 * the test rather than the kernel. The same is true of a sigmoid gate, of a lower-triangular
 * factor, and of every index operand: a KV slot indexes n_blocks * block_size positions and a
 * block-table entry indexes n_blocks, and drawing either from the wrong range tests the kernel's
 * out-of-range handling and calls the result correctness. */
enum {
    RAD_FILL_NORMAL      = 0,  /* standard normals -- the default for a float operand */
    RAD_FILL_SIGMOID     = 1,  /* (0, 1): a gate */
    RAD_FILL_GATE_CUMSUM = 2,  /* non-increasing and <= 0, reset every `fill_chunk` rows */
    RAD_FILL_TRI_LOWER   = 3,  /* zero above the diagonal within a `fill_chunk` block */
    RAD_FILL_INDEX       = 4,  /* integers; see idx_max / idx_const / the IDX flags */
    /* UNIT L2 NORM ALONG THE LAST DIMENSION. The gated-delta-net key is normalised by the
     * model's own qk_norm before it reaches the solve, and the solve inverts I + tril(beta k k^T,
     * -1): drawn as unit normals over a 128-wide head that matrix has off-diagonals a hundred
     * times too large, its inverse is enormous, and two implementations summing an
     * ill-conditioned product in different orders disagree by however ill-conditioned it is. What
     * that reports is the condition number of the TEST. Same correction as SIGMOID and
     * GATE_CUMSUM above: put the operand in the domain the op is only ever handed. */
    RAD_FILL_UNIT_ROW    = 5,
};

enum {
    /* An OPTIONAL operand this geometry does not pass. One entry per schema operand is still
     * emitted -- that count is what catches a hook that has drifted from its schema -- so absence
     * needs a spelling, and a null RadTensor.data is what the ABI already means by it. */
    RAD_OPD_F_ABSENT     = 1u << 0,
    /* A SCATTER DESTINATION: no value may repeat. Two tokens drawn onto one slot make the store a
     * race, and two implementations resolve it differently -- so the comparison would measure the
     * scheduler. A read-side index (a gather row, a block-table entry) may repeat and does. */
    RAD_OPD_F_IDX_UNIQUE = 1u << 1,
    /* CUMULATIVE SEQUENCE LENGTHS, [0, ..., T]. Not a draw: it is the batch's shape. */
    RAD_OPD_F_IDX_CU     = 1u << 2,
};

typedef struct RadOpdDesc {
    uint32_t dtype;
    uint32_t rank;
    int64_t  shape[RAD_MAX_RANK];

    uint32_t fill;         /* RAD_FILL_* */
    int64_t  fill_chunk;   /* the reset width for GATE_CUMSUM and TRI_LOWER */

    /* RAD_FILL_INDEX only. `idx_max` is the EXCLUSIVE upper bound of the draw; 0 means "the
     * leading extent of the previous operand", which is the common case and saves every gather
     * from restating it. `idx_const` >= 0 makes every element that value rather than a draw --
     * `seqused` is the one that needs it, because a random context length shorter than the query
     * makes a causal attention read a prefix that does not exist, which both implementations
     * would then agree about while neither computed what the model asks for. */
    int64_t  idx_max;
    int64_t  idx_const;    /* -1 to draw */

    uint32_t flags;        /* RAD_OPD_F_* */
} RadOpdDesc;

/* Fill `out` for operand `operand` of this op at this geometry. `p`/`n_p` is the resolved
 * parameter list, exactly as `launch` would receive it. Return RAD_OK, or RAD_E_UNSUPPORTED when
 * this geometry does not carry what the description needs -- a caller SKIPS AND COUNTS that
 * rather than guessing an extent, because a tool that invents one tests the kernel's bounds
 * handling and calls the result correctness.
 *
 * Pure host code, no device memory, no side effects: it is called before anything is allocated. */
typedef int (*RadShapeFn)(const RadParam* p, int n_p, int operand, RadOpdDesc* out);


/* ================================================================== tunable instantiation */
/* A KERNEL IS A FAMILY, NOT AN IMPLEMENTATION. Tile sizes, split counts, waves per block: which
 * member of the family is fastest is a property of the shape and the machine, and the only way to
 * know is to measure it on the machine (spec §15). A library declares the AXES here and the core
 * takes the cross product.
 *
 * AXES, NOT A LIST OF PRE-NAMED COMBINATIONS. Hand-enumerating the interesting points of a
 * multi-axis space -- "wv1_sk8_mb3_npw4" and its twenty-odd siblings -- leaves HOLES, and a hole
 * costs step time silently: a ladder that skips a value runs the next tile up, so a shape with
 * forty rows of work gets a sixty-four-row tile. A cross product has no holes.
 *
 * It also keeps the tuning LAW out of the launch adapter, where it would be C++ -- ladders, caps,
 * a hand-written picker -- that an out-of-tree author cannot write.
 *
 * THE CHOSEN VALUES ARRIVE AS ORDINARY PARAMETERS. There is no variant index: once the core has
 * picked a point in the space it appends the choices to RadArgs::p, and the kernel reads `sk` with
 * the same rad_args_geti() it reads `M` with. One way in, for a value the caller supplied and a
 * value the tuner chose. */
typedef struct RadTunable {
    const char*    key;       /* the kernel's own name for the axis: "sk", "bm", "npw" */
    const int64_t* values;    /* legal values, ascending; the core tries exactly these */
    int            n_values;
    /* WHAT RUNS WHEN THIS MACHINE HAS NO MEASUREMENT FOR THIS INSTANTIATION -- a fresh install, a
     * shape nobody tuned, a cache from another card. There is no fallback beneath it, so it must
     * be a value the kernel can actually launch: the loader CHECKS that it appears in `values`
     * and refuses the plugin by name if it does not. */
    int64_t        deflt;
    const char*    doc;       /* what the axis does, for the tuner's report */
} RadTunable;

/* IS THIS COMBINATION LEGAL AT THIS GEOMETRY? Cross-axis rules live here rather than in the axis
 * lists, because they are exactly what a product of independent lists cannot say: libr4d's skinny
 * GEMM needs `wv * sk * 32` inside the block bound AND `K % (sk * unit) == 0`, and neither is a
 * property of `sk` alone.
 *
 * `p` is the resolved geometry, `choice` the candidate as key/value parameters. Pure host code,
 * called before anything is allocated, once per candidate.
 *
 * Returns 1 for legal and 0 for SKIP -- which is not an error. A kernel may serve a subset of its
 * own cross product, and the tuner reports a skipped candidate as skipped rather than as a
 * failure. Null means every combination is legal. */
typedef int (*RadTuneValidFn)(const RadParam* p, int n_p, const RadParam* choice, int n_choice);
/* ================================================================== op schemas */
/* A parameter is supplied by the caller (REQUIRED), may be (OPTIONAL), or is supplied by the
 * KERNEL (DERIVED).
 *
 * DERIVED exists because of a circularity that block_size makes concrete. Spec §7.2 says the
 * paged block size comes from the resolved attention kernel and not from a core constant --
 * libr4d is compiled for 16 and rejects anything else. But a constraint whose key the geometry
 * does not carry does NOT hold (libr4d's rule, and the reason its two-shot all-reduce rows have
 * to precede the one-shot ones), so a kernel constrained to block_size == 16 can never be
 * selected by a query that omits block_size -- and the resolution the answer was to be read off
 * never happens.
 *
 * So: constraints on a DERIVED parameter are SKIPPED during matching, and once a kernel wins,
 * its RAD_C_EQ value for that key is written into the resolved geometry. A caller that supplies
 * one anyway pins it, and it matches normally -- which is how an operator forces a block size
 * and gets a refusal by name rather than a silent substitution. A kernel with no EQ on a derived
 * key leaves it unset, and rad_kv_block_size() reports that rather than returning zero. */
enum { RAD_OPTIONAL = 0, RAD_REQUIRED = 1, RAD_DERIVED = 2 };

/* WHAT A PARAMETER MEANS TO THE CORE, when the core has to act on one it did not invent.
 *
 * Almost none of them: a parameter is a key the plugin and the kernel agree on and the core
 * carries it opaquely, which is the whole reason a third party can add an op. The exception is
 * the SEQUENCE CHUNK. A linear-state scan tiles the token stream, its kernel is compiled for one
 * tile length, and the SCHEDULER has to cut its steps on a multiple of that or a chunk straddles
 * a sequence boundary -- so the core does need to find that one number in a geometry it otherwise
 * does not read.
 *
 * The schema is where that number is named, and the alternative is worse: a core that found it by
 * SPELLING would carry a list of likely keys ("chunk", "chunk_size", "state_chunk", ...) inside
 * the scheduler, which makes the core the thing you edit to add a kernel that spells it
 * differently. Declaring the role costs a field and keeps the core free of that vocabulary. */
/* RAD_PROLE_CAPACITY is a count the kernel is handed as a BOUND and never writes past its
 * operands for: it launches, loops and writes by the extents of the tensors it is given. That is
 * what lets the activation arena be declared again at a smaller step (RadBuildCtx::shape_probe) --
 * the kernel keeps the geometry it was selected with, including the real declare's value here,
 * while its buffers are the smaller step's -- so it is the KERNEL's claim to make, and a fixed
 * parameter that differs between the two declares without it refuses the smaller size. */
enum { RAD_PROLE_NONE = 0, RAD_PROLE_SEQ_CHUNK = 1, RAD_PROLE_CAPACITY = 2 };

typedef struct RadParamSpec {
    const char* key;
    int         type;      /* RAD_P_INT | RAD_P_STR */
    int         required;
    int         role;      /* RAD_PROLE_*; 0 for every parameter the core does not interpret */
} RadParamSpec;

/* Positional operand roles. The core needs to know which operands are weights (so issue can
 * resolve a handle and wait on the mover) and which are written (so the buffer planner can
 * compute liveness). It does not need to know anything else about them. */
/* RAD_OPD_WTAB is a weight operand whose issue names a RUN of weights rather than one -- a MoE
 * layer's experts -- and the kernel is handed a device array of pointers. It is a role of its own
 * and not a flag on RAD_OPD_WEIGHT so that a tool reading the schema knows the operand is not a
 * plane it can describe with one shape: see RAD_WTAB in rad_runtime.h. */
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
 * WHAT CHAIN OF CONVENTIONAL OPS DOES THIS KERNEL COMPUTE?
 *
 * The obvious design is for a kernel library to advertise that "kernel A can be used to fuse B
 * and C", and for the engine to apply that repeatedly until the graph converges. spec.md §2.3
 * refuses the second half -- "a fusion rewriter is a compiler, and a compiler is the thing whose output nobody can
 * predict" -- and the declare phase keeps nothing a rewriter could work on in any case: `Program`
 * holds no operand edges. The Builder computes them, collapses each buffer's whole use-set to a
 * `first_def`/`last_use` pair with reads and writes MERGED, and discards the rest when it returns.
 * "Op A's output is op B's only input" is not recoverable from that, which is also why
 * `--debug-graph` declines to print an edge it cannot prove.
 *
 * So this is a DECLARATION, not a rewriter. Fusion stays a selection: the architecture asks for
 * the fused op and emits the sequence when nothing serves it (spec.md 2.3, and rad_op_resolved in
 * rad_builder.h, which is the idiom an out-of-tree plugin needs for a banded or optional fusion).
 * What the declaration adds is that the EQUIVALENCE becomes machine-readable instead of prose in a
 * kernel row -- "byte-identical to the two ops it replaces" -- and prose cannot be checked. It
 * buys three things:
 *
 *   AN ORACLE, which is the one that matters. A fused kernel with no libref counterpart can only
 *   be skipped by name, and such skips dominate what a coverage run reports on a model built out
 *   of fusions. A fused kernel that declares its chain can instead be checked against that chain
 *   run out of the SAME plugin's unfused rows -- which is precisely the claim the row makes, and
 *   a stronger test than a tolerance against ref, because the answer should be BYTE-IDENTICAL.
 *
 *   AN ADVISORY in --debug-graph: these N declared ops are a chain some kernel says it can do in
 *   one launch, and here is why it did not fire.
 *
 *   A COVERAGE METRIC, which spec.md:19 lists.
 *
 * WHY A HOOK AND NOT A TABLE. A step's geometry is not the fused op's geometry and is not always
 * derivable by a caller: `qk_norm_rope_gate` replaces an `rmsnorm` at M = T * n_head, and nothing
 * in the fused op's parameter list is that product. The kernel is the component that knows, which
 * is the same argument RadShapeFn and RadLayoutFn already make.
 *
 * OPERAND WIRING. Every operand of every step is either an operand of the FUSED kernel or a
 * TEMPORARY the chain passes between its own steps. A temporary needs no description here: it is
 * sized and typed by the step's own `opd_shape`, which every kernel that wants to be checked this
 * way already publishes. Note how often no temporary is needed at all -- the fp8 folds still WRITE
 * their intermediate when `out_bf16` is passed, precisely so the fused and unfused forms leave the
 * same buffers behind. */
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
    RadParam    p[RAD_MAX_FUSE_PARAMS];        /* this STEP's geometry, not the fused one's */
    int         n_p;
    /* One entry per operand of `op`, in its schema's order: an operand index of the fused kernel,
     * RAD_FUSE_TMP(t), or RAD_FUSE_NONE. The count is checked against the step op's schema, so a
     * hook that drifts from the vocabulary is a diagnosable error rather than a silent misfire --
     * the same rule RadShapeFn's operand count follows, and for the same reason. */
    int16_t     from[RAD_MAX_FUSE_OPD];
    int         n_from;
} RadFuseStep;

/* Describe step `step` of the chain this kernel replaces, for this geometry and variant.
 *
 *   RAD_OK             `out` is filled; ask again with step + 1.
 *   RAD_E_NOTFOUND     `step` is past the end. This is how a caller learns the chain's LENGTH,
 *                      and it is why there is no count beside the hook: the chain may differ with
 *                      the geometry (an `rmsnorm_quant_fp8` with a residual replaces `rmsnorm_add`
 *                      and without one replaces `rmsnorm`).
 *   RAD_E_UNSUPPORTED  this geometry or variant has no unfused equivalent in the vocabulary. An
 *                      honest answer, and the right one wherever a fusion computes something the
 *                      conventional ops cannot express between them.
 *
 * A chain must be EXACT, not approximate. A kernel that is merely close to the sequence it names
 * should publish no hook at all: the whole value of the declaration is that a caller may
 * substitute one for the other and compare the bytes. */
/* WHICH OPTIONAL OPERANDS THIS CALL PASSES, one bit per operand of the FUSED op, bit i set when
 * operand i is present.
 *
 * THIS IS NOT DECORATION: without it the chain is not answerable at all.
 * `rmsnorm_quant_fp8` replaces `add` -> `rmsnorm` -> `quant_act_fp8` when it is given a residual
 * and `rmsnorm` -> `quant_act_fp8` when it is not: THE CHAIN LENGTH ITSELF depends on an operand,
 * not on a parameter. Without this the hook has to infer presence, and the only honest inference
 * available is "whatever this plugin's own RadShapeFn publishes" -- which is right for a checker
 * calling the kernel cold, since that is where its buffers came from, and WRONG for a caller
 * holding a real issue, which is what the graph dump has. A fold with no residual would be
 * described as three steps and the advisory would be a lie.
 *
 * Nor could `RAD_FUSE_NONE` express it: the operands that vary are the step's REQUIRED ones.
 *
 * RAD_FUSE_PRESENT_ALL is what a caller passes when it does not know, and it means exactly "answer
 * for every operand your description says exists". That is a contract rather than a coupling: the
 * hook may then assume its own RadShapeFn, and a caller that knows better says so. */
#define RAD_FUSE_PRESENT_ALL ((uint64_t)~0ull)
#define RAD_FUSE_HAS(mask, i) (((mask) >> (i)) & 1ull)

typedef int (*RadFuseFn)(const RadParam* p, int n_p, uint64_t present, int step,
                         RadFuseStep* out);

/* ================================================================== kernel rows */
typedef struct RadKernelInfo {
    const char* name;       /* entry point, unique within the plugin */
    const char* op;         /* the op it implements -- the key callers select on */
    const char* family;     /* attn | gdn | ar | gemm | norm | sample | ... , for grouping only */
    const char* computes;   /* prose, for a human */
    const char* shape;      /* the geometry it is compiled for, prose */
    const char* dtypes;

    int         domain;     /* RAD_DOMAIN_DEVICE | RAD_DOMAIN_HOST */
    int         priority;   /* higher wins WITHIN THIS PLUGIN; ties break on declaration order */

    const RadConstraint* constraints;  int n_constraints;

    /* The axes of this kernel's instantiation space, and the cross-axis legality test. Null/0
     * means one implementation with nothing to tune, which is the common case and costs nothing:
     * the core appends no choices and the kernel reads no tuned key. See RadTunable. */
    const RadTunable*    tunables;     int n_tunables;
    RadTuneValidFn       tune_valid;

    RadInitFn      init;
    RadFiniFn      fini;
    RadLaunchFn    launch;
    RadScratchFn   scratch;
    RadLayoutFn    layout;
    RadRelayoutFn  relayout;
    /* How to size and fill each operand, for a tool that calls this kernel cold. Null means this
     * kernel cannot describe itself and every tool that needs a buffer for it skips it BY NAME
     * rather than guessing. libref supplies it for the whole conventional vocabulary, so an
     * op that stays inside docs/OPS.md is describable even when the kernel under test is not. */
    RadShapeFn     opd_shape;
    /* The inverse of `relayout`, for a tool holding the stored bytes rather than the planes.
     * Null means this kernel's stored form cannot be read back, and a dense oracle therefore has
     * nothing to compare against -- which is a skip, named, and not a failure. */
    RadUnrelayoutFn unrelayout;
    /* The chain of conventional ops this kernel computes in one launch, or null for a kernel that
     * is not a fusion. Non-null is also the fusion-coverage metric's numerator: it is a kernel
     * saying "I stand in for these, and you may check me against them". */
    RadFuseFn      replaces;
    /* Optional. The same dispatch, described rather than performed, so a repeatedly issued
     * sequence can be replayed as a graph. Null means this kernel is always submitted
     * individually, which is correct and merely slower. See RadDescribeFn for the one trap.
     *
     * LAST IN THE STRUCT ON PURPOSE. Every kernel table in this tree is a POSITIONAL aggregate
     * initialiser -- `{..., nullptr, nullptr, r4d_rad_embed_lookup, nullptr, ...}` -- so a field
     * inserted anywhere else silently shifts every initialiser after it, and `nullptr` converts
     * to any function pointer type, so part of that shift would compile. Append here. */
    RadDescribeFn  describe;
} RadKernelInfo;

/* ================================================================== kernel plugin exports */
/* Every kernel plugin exports the whole set. The schemas come first: the first plugin in
 * hierarchy order to declare an op fixes its schema, and a later plugin declaring the same op
 * name with a different schema is refused at load with both plugins named. Since arguments are
 * positional, two disagreeing schemas would make the same call mean different things depending
 * on which plugin won selection, and that is not a diagnosable failure -- it is silent
 * numerical garbage (spec §2.3). */
int                  rad_kernel_schema_count(void);
const RadOpSchema*   rad_kernel_schema_at(int i);
int                  rad_kernel_count(void);
const RadKernelInfo* rad_kernel_at(int i);

/* Optional. Called once after the plugin is loaded and before any selection, and once at
 * shutdown. A plugin with no global state need not export them.
 *
 * THE SYMBOL NAMES ARE `rad_plugin_open` and `rad_plugin_close`. The types are named here so a
 * plugin can declare them with the right signature; the loader resolves them by those exact
 * names, and a plugin that spells them differently is silently never initialised.
 *
 * `rad_plugin_open` answering RAD_E_UNSUPPORTED DECLINES this machine: the plugin is unloaded and
 * the rest of the installation carries on, which is how one installation ships libraries for
 * several cards. Any other negative status refuses the plugin as an error. A kernel library does
 * not need to decline a card its device code was not built for -- the loader reads that out of
 * the code objects it carries and leaves its device kernels out on its own. */
typedef int  (*RadPluginOpenFn) (void);
typedef void (*RadPluginCloseFn)(void);

/* Optional, and a promise about kernel `i` of rad_kernel_at: nonzero means every launch of it
 * reads and writes no memory but its operands and the scratch it was handed -- no allocation it
 * keeps for itself, no counter it keeps between launches -- so two launches may run at the same
 * time on one device whenever their operands and scratch do not overlap. The engine then lets a
 * launch start before the independent ones in front of it on its queue have finished. A plugin
 * that does not export the symbol, or answers 0, has every launch ordered behind everything
 * before it, which is always correct. The symbol name is `rad_kernel_concurrent`. */
typedef int  (*RadKernelConcurrentFn)(int i);

/* Optional: nonzero declares this kernel library a REFERENCE implementation -- written to be
 * plainly correct rather than fast. Three things follow, for any library that says so and for no
 * other, whatever it is called:
 *   - the loader ranks it below every unnamed kernel library, because a reference matches every
 *     op it declares and would otherwise answer for them ahead of the library serving the card
 *     (naming it in --kernels still puts it where it is named);
 *   - the engine refuses to serve a model with an op only a reference library implements, unless
 *     told to with --debug-accept-reference-kernels;
 *   - rad-kbench checks every other library against it.
 * A plugin that does not export the symbol, or answers 0, is an ordinary library. The symbol name
 * is `rad_plugin_reference`. */
typedef int  (*RadPluginReferenceFn)(void);

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_ABI_H */
