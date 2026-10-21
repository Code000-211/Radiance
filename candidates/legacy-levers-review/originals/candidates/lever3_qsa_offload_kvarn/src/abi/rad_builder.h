/* rad_builder.h -- the declare phase. An architecture plugin states everything it will ever do
 * before it does any of it: every weight, every buffer, every op, in order. See spec.md §3.1.
 *
 * Between declare and plan the whole model is known statically -- every op, every kernel that
 * will service it, every weight it touches, and the order. That is the property the rest of the
 * engine spends.
 */
#ifndef RAD_BUILDER_H
#define RAD_BUILDER_H

#include "rad_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RadBuilder RadBuilder;

/* ================================================================== model metadata */
/* What the loader read out of the container's header before any plugin ran. Dimensions come from
 * here, so one plugin covers every size in a family: qwen3_dense_bf16.so serves 2B, 9B and 27B. */
typedef struct RadModelMeta {
    const char* arch_id;         /* the string that selected this plugin */
    const char* name;
    const char* quant;           /* the quantisation descriptor, e.g. "w4a8_symz_g128" */

    int64_t n_layers;
    int64_t n_embd;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t head_dim;
    int64_t n_ff;
    int64_t n_vocab;
    int64_t n_ctx_train;

    int64_t n_expert;            /* 0 for a dense model */
    int64_t n_expert_used;
    int64_t n_expert_shared;

    float   rms_eps;
    float   rope_theta;
    float   rope_scale;

    /* Free-form key/value from the container header, for anything the struct above does not
     * name. A plugin reads what it needs; the core never interprets these. */
    int         n_kv;
    const char* const* kv_key;
    const char* const* kv_val;
} RadModelMeta;

long long   rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt);
double      rad_meta_getf(const RadModelMeta* m, const char* key, double dflt);
const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt);

/* ================================================================== build context */
typedef struct RadBuildCtx {
    int      rank;            /* declare runs once per rank; the plugin declares dimensions */
    int      world_size;      /* ALREADY DIVIDED -- the core inserts no collectives (spec §9) */

    int64_t  max_tok;         /* the largest token count one step may carry */
    int64_t  max_seqs;        /* the largest number of sequences one step may carry */
    int64_t  max_ctx;         /* context bound this deployment was configured for */
    int      max_spec;        /* speculative window bound, 0 if speculation is off */

    /* THE TENSOR-PARALLEL WIRE. 0 -- which is what a zero-initialised context gives, and that is
     * deliberate -- is the EXACT all-reduce. Nonzero says this deployment accepts a lossy wire,
     * which at two ranks libr4d serves as a Walsh-Hadamard-rotated 6-bit payload: 6 bits an
     * element plus a bf16 scale per group of 64, against 16 bits an element, so the message a
     * prefill chunk pushes across the link drops by 2.4x. The plugin inverts it into
     * `all_reduce`'s `exact` parameter. The polarity is this way round because the sum a rank
     * receives is then not the sum it would have received, and a field that defaults to lossy
     * when someone forgets to fill it is the failure spec §17 exists to prevent. */
    int      tp_wire_lossy;

    /* WHERE THE LOSSY WIRE STARTS PAYING, in bytes of one all-reduce message. Below it the exact
     * kernel wins and the plugin must use it even when `tp_wire_lossy` is set: compression only
     * pays once the transfer is bandwidth-bound, and a decode step's message is not -- it is a few
     * tokens of the residual stream, where the rotate/pack/unpack costs more than the bytes it
     * saves. Above it a prefill chunk is squarely on the wire and the 2.4x is real. 0 means the
     * plugin's own default, which is 128 KiB. */
    int64_t  tp_wire_min_bytes;


    /* THE KV CACHE'S WIDTH, AND IT IS A DEPLOYMENT CHOICE RATHER THAN A PROPERTY OF THE
     * CHECKPOINT. `kv_dtype` follows the CACHE, not the weights: a container whose weights are
     * fp8 may well want a bf16 cache and vice versa, and one field read as "the cache is bf16
     * because the weights are" was true of one checkpoint and never a rule.
     *
     * RAD_DT_INVALID -- which is what a zero-initialised context gives -- means the plugin's own
     * default, which every architecture in this tree takes as bf16. RAD_F8E4M3 is the other
     * value anything implements today; the fp8kv attention family is band-gated on it and the
     * store side writes E4M3 with per-(sequence, head) descales.
     *
     * IT IS AN ABI FIELD AND NOT AN ENVIRONMENT VARIABLE because it changes what the model
     * computes. A per-architecture environment switch gives "is the cache fp8" one answer per
     * architecture, each with its own default, and none of them appears in the startup log or in
     * a bug report. --kv-cache-dtype is the one answer. */
    int      kv_dtype;        /* RAD_DT_INVALID = the plugin's default (bf16); or RAD_F8E4M3 */
    int      is_draft;        /* this builder scope is a drafter (spec §10) */
    const char* scope;        /* "", "draft", "vision", "audio" -- prefixes declared names */

    /* A DECLARE THAT ONLY SIZES THE ACTIVATIONS, at a `max_tok` smaller than the deployment's.
     * The core runs one per level of the activation arena it lends out: what the plugin declares
     * says how big every buffer is when no step carries more than `max_tok` tokens, and the top of
     * the arena past that is lent to the expert slab while the steps are that small.
     *
     * THE PLUGIN MUST LEAVE ITS OWN STATE ALONE WHILE THIS IS SET. The state its step function
     * reads was written by the real declare, at the real `max_tok`, and a sizing declare that
     * overwrote it would run every later step against the smaller configuration. Declare into
     * scratch storage instead; nothing a sizing declare produces is ever issued, and its op
     * handles are the real declare's.
     *
     * Only asked of a plugin that set RadArchProbe::shape_probe_ok. Trailing and zero-defaulted,
     * so a plugin built against the older struct is unchanged. */
    int      shape_probe;

    /* THE MOST PATCHES ONE ENCODER PASS MAY CARRY, and 0 means this deployment serves no media:
     * the plugin declares no encoder, so a tower the container carries stays off the card. It is
     * the encoder's `max_tok` -- every activation it declares is sized against it -- and it bounds
     * the largest single image the server accepts, since one image is one attention segment and a
     * segment is never split across passes. */
    int64_t  max_enc_patches;

    /* THE MOST LOGITS ROWS ONE STEP MAY NAME, when that is more than the plugin's own rule of one
     * per sequence (1 + max_spec at a speculative step). 0 is the rule. The engine's KL mode sets
     * it to a whole step's tokens plus a row a sequence, because it reads a full-vocabulary row at
     * every prompt position; the logits buffer and the lm_head's row range are sized from it. The
     * engine reads the declared logits buffer back and refuses the mode when it is shorter, so a
     * plugin that ignores the field is refused rather than overrun.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int64_t  max_out_rows;

    int      qsa_offload;
    int      kvarn;
} RadBuildCtx;

/* ================================================================== weights */
/* Access class, declared per weight because it is information only the architecture plugin has.
 * It is what tells the placement planner whether prefetch can be exact (spec §5.2). */
enum {
    RAD_ACCESS_PER_TOKEN   = 0,  /* read every forward pass -- dense weights */
    RAD_ACCESS_PER_REQUEST = 1,  /* read once per request -- an encoder tower, with an image */
    RAD_ACCESS_CONDITIONAL = 2,  /* read only if selected -- routed experts */
    RAD_ACCESS_RARE        = 3,  /* may not be read at all in a given deployment */
    RAD_ACCESS_VOCAB       = 4   /* lm_head and the embedding table: read every token, enormous,
                                  * and one of them is a gather rather than a GEMM. Its own class
                                  * because neither PER_TOKEN's prefetch nor CONDITIONAL's
                                  * prediction describes it (spec §19.2). */
};

/* How a weight is split across tensor-parallel ranks. The plugin says; the core does not infer. */
enum {
    RAD_SHARD_NONE = 0,   /* replicated on every rank */
    RAD_SHARD_ROW  = 1,   /* split along dim 0 -- column-parallel in the usual naming */
    RAD_SHARD_COL  = 2    /* split along dim 1 -- row-parallel, needs a following all-reduce */
};

/* Grouping key: the unit of movement. An expert's gate/up/down share one, because the unit of
 * transfer is an expert, not a tensor (spec §4.1). Layer grouping is what makes layer offload a
 * contiguous run rather than a scatter. */
typedef struct RadWeightGroup {
    int32_t layer;    /* -1 for a model-level weight */
    int32_t expert;   /* -1 for a non-expert weight */
    int32_t slot;     /* free index within the group, e.g. gate=0 up=1 down=2 */
} RadWeightGroup;

/* NOTE: these are C compound literals, which C++ accepts only as a GNU extension. An architecture
 * plugin written in C++ -- which is the norm -- should use the grp_layer()/grp_expert()/grp_model()
 * inline functions in `<arch/rad_arch.h>` instead: the installed spelling, which the plugins in
 * radiance's own tree use as well. Same values, no extension. */
#define rad_group_layer(l)      ((RadWeightGroup){ (int32_t)(l), -1, 0 })
#define rad_group_expert(l, e)  ((RadWeightGroup){ (int32_t)(l), (int32_t)(e), 0 })
#define rad_group_model()       ((RadWeightGroup){ -1, -1, 0 })

/* The most projections one stored tensor may stack along dim 0. Five is what the bf16 delta net
 * needs -- q, k, v, a, b in one row -- and eight leaves room without making the decl big. */
enum { RAD_MAX_ROW_PARTS = 8 };

typedef struct RadWeightDecl {
    uint32_t       dtype;       /* what the kernel reads; the stored layout comes from the kernel */
    uint32_t       rank;
    int64_t        shape[RAD_MAX_RANK];   /* THIS RANK's extent of what `planes` select (below) */
    int            access;      /* RAD_ACCESS_* */
    int            shard;       /* RAD_SHARD_* */
    RadWeightGroup group;
    int            optional;    /* absent in the container is not an error */
    /* WHAT IS STACKED IN DIM 0, for a RAD_SHARD_ROW weight whose stored tensor is several
     * projections concatenated -- [gate|up], or the delta net's [q|k|v]. Rows of THIS RANK, in
     * stored order, summing to shape[0]; 0 parts means one part and dim 0 is one projection.
     *
     * It exists because a tensor-parallel rank owns a slice of EACH stacked projection and not a
     * contiguous slice of the stack. Taking rows [rank*D, (rank+1)*D) of a fused gate_up hands
     * rank 0 the whole gate and rank 1 the whole up; both ranks then compute a perfectly
     * well-formed wrong thing and the model produces fluent wrong text, which is the failure
     * class spec §17 exists to prevent. The parts are NOT inferable from the container: it holds
     * the fused result and not the shapes it was fused from, and a single checkpoint tensor may
     * already be a fusion nothing in the name map mentions (in_proj_qkv is three head groups in
     * one [10240, 5120]). The plugin is the only component that knows, so the plugin says. */
    int            n_row_parts;
    int64_t        row_parts[RAD_MAX_ROW_PARTS];

    /* WHICH STORED WEIGHT THIS IS A VIEW OF, AND WHICH OF ITS PLANES. spec.md §4.3.
     *
     * A container entry, or a checkpoint tensor, is a LOGICAL weight: the planes of its encoding
     * (rad_encoding.h) -- a block-fp8 linear's codes and its scale plane are one weight, not two.
     * A kernel takes them as whatever operands its schema has: gemm_nt_q takes the codes and the
     * scales as two, the fp8 lm_head one operand holding both. So a declared weight names its
     * `source` -- null for the logical weight of its own name -- and the `planes` it takes, by
     * role, comma-separated: "codes", "scale", "codes,scale". Null takes every plane, which for a
     * plain weight is its one.
     *
     * One declared weight is one stored buffer. Its dtype and shape describe what is selected: a
     * single plane's own dtype and extents, or, for several, the codes' dtype and the logical
     * extents. A dtype other than the plane's is legal only as an exact widening -- a bf16 norm
     * read as f32 -- which the loader performs; anything lossy is a quantiser's job and happens
     * before a container is written. */
    const char*    source;
    const char*    planes;
} RadWeightDecl;

rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d);

/* WHAT A STORED WEIGHT IS: its encoding, and its logical extents in `shape` (RAD_MAX_RANK
 * entries, may be null) with the rank returned in `rank` (may be null). RAD_OK, or RAD_E_NOTFOUND
 * when the model has no such weight.
 *
 * The answer comes from wherever the model is being loaded from -- a container's entry, a
 * checkpoint's tensor, or, in rad-convert, the recipe that will quantise it -- so a plugin learns
 * the same thing the same way at convert and at load, and chooses its ops by it: an fp8 GEMM for
 * a block-fp8 weight, the rotated expert kernels for `i4*bf16[1x128]/fwht128`. Ask after the
 * weight's name map entry (rad_decl_name_map), which is how a checkpoint is searched for it. A
 * builder with no model behind it -- a test, a synthetic tool -- answers RAD_E_NOTFOUND, and a
 * weight declared there is read as plain in its declared dtype. */
int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t* shape,
                        uint32_t* rank);

/* ================================================================== buffers */
/* Activations are declared too, or the buffer plan has nothing to analyse. Liveness is computed
 * over the declared op list, so scratch and intermediates come from one arena sized once instead
 * of an allocator called per step (spec §3.1). */
enum {
    RAD_BUF_TRANSIENT = 0,  /* liveness-analysed, may share storage with another transient */
    RAD_BUF_PERSIST   = 1,  /* survives across steps: a KV-adjacent scratch, a routing histogram */
    RAD_BUF_DERIVED   = 2   /* computed once per step by the core from the batch (spec §8) */
};

typedef struct RadBufDecl {
    uint32_t    dtype;
    uint32_t    rank;
    int64_t     shape[RAD_MAX_RANK];   /* sized at max_tok, which is the worst step */
    int         kind;                  /* RAD_BUF_* */
    int         domain;                /* RAD_DOMAIN_DEVICE or RAD_DOMAIN_HOST */
    const char* derive;                /* RAD_BUF_DERIVED: the derivation the core runs */
    /* WHICH KV GROUP A DERIVATION IS ABOUT, or 0 for "the only one of its kind".
     *
     * `conv_state_index` computes a read cursor inside a rolling window, which means it needs THAT
     * GROUP's slot per sequence and THAT GROUP's window depth -- and a model may declare several.
     * Qwen4-Exp's PLE layer has two conv states of different depths and dtypes beside the gated
     * delta net's third. Leaving this 0 with more than one group of the kind a derivation reads is
     * refused rather than guessed at: the wrong window's depth produces a plausible cursor and a
     * convolution that reads history it never wrote.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    rad_kvgroup kv;
} RadBufDecl;

rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d);

/* ================================================================== KV groups */
/* Each distinct state kind is a KV group with its own block geometry, its own pool, and its own
 * slot mapping in the step batch. Linear-attention state is a KV group too: it is not paged the
 * same way, but it is allocated, mapped and freed by the same manager, because otherwise two
 * allocators disagree about what a sequence owns (spec §7.2). */
enum {
    RAD_KV_FULL    = 0,   /* paged, per-token, growing -- full attention */
    RAD_KV_WINDOW  = 1,   /* paged, per-token, bounded by the window */
    RAD_KV_LINEAR  = 2,   /* per-sequence, fixed size -- GDN recurrent state */
    RAD_KV_CONV    = 3    /* per-sequence, a rolling window of width-1 + num_spec entries */
};

typedef struct RadKVGroupDecl {
    int      kind;            /* RAD_KV_* */
    uint32_t dtype;
    int64_t  window;          /* RAD_KV_WINDOW: the window; 0 otherwise */
    int64_t  n_head_kv;
    int64_t  head_dim;
    int64_t  state_dim[2];    /* RAD_KV_LINEAR: the per-head state geometry, e.g. {128,128} */
    int64_t  conv_width;      /* RAD_KV_CONV */
    /* Block size is NOT declared. It is taken off the resolved attention kernel's constraints at
     * declare, because a core that picked its own would be a core that has to be edited when a
     * kernel changes (spec §7.2). Read it back with rad_kv_block_size() after declaring the
     * attention op that consumes this group. */
} RadKVGroupDecl;

/* `rad_kvgroup` is in rad_types.h with the other handles -- RadBufDecl above carries one. */
rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name, const RadKVGroupDecl* d);
/* Valid only after the op that reads this group has been declared. 0 before that. */
int64_t     rad_kv_block_size(RadBuilder* b, rad_kvgroup g);

/* Bind a layer to a group, so the block manager knows how many state instances a sequence owns. */
int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g);

/* ================================================================== ops */
/* Declared with their parameters and their weight operands. The core resolves each against the
 * kernel hierarchy, per band, per domain. This phase ALWAYS completes; if anything failed to
 * resolve, the full list is reported at the end rather than the first failure at the top. */
rad_op rad_decl_op(RadBuilder* b, const char* op,
                   const RadParam* params, int n_params,
                   const rad_weight* weights, int n_weights);

/* Convenience wrappers so a declare function reads like a list of what the model does.
 *   rad_op h = RAD_OP(b, "gemm_w4a8_nt",
 *                     RAD_PARAMS(RAD_RANGE("M",1,max_tok), RAD_INT("N",8192), RAD_INT("K",5120)),
 *                     RAD_WEIGHTS(w));
 */
#ifdef __cplusplus
} /* extern "C" */
#include <initializer_list>

inline rad_op rad_decl_op(RadBuilder* b, const char* op,
                          std::initializer_list<RadParam> params,
                          std::initializer_list<rad_weight> weights) {
    return rad_decl_op(b, op,
                       params.size() ? params.begin() : nullptr, (int)params.size(),
                       weights.size() ? weights.begin() : nullptr, (int)weights.size());
}
inline rad_op rad_decl_op(RadBuilder* b, const char* op,
                          std::initializer_list<RadParam> params,
                          const rad_weight* weights, int n_weights) {
    return rad_decl_op(b, op,
                       params.size() ? params.begin() : nullptr, (int)params.size(),
                       weights, n_weights);
}
inline rad_op rad_decl_op(RadBuilder* b, const char* op,
                          const RadParam* params, int n_params,
                          std::initializer_list<rad_weight> weights) {
    return rad_decl_op(b, op,
                       params, n_params,
                       weights.size() ? weights.begin() : nullptr, (int)weights.size());
}

#define RAD_PARAMS(...)   std::initializer_list<RadParam>{ __VA_ARGS__ }
#define RAD_NOPARAMS      ((const RadParam*)0), 0
#define RAD_WEIGHTS(...)  std::initializer_list<rad_weight>{ __VA_ARGS__ }
#define RAD_NOWEIGHTS     ((const rad_weight*)0), 0
#define RAD_OP(b, op, ...) rad_decl_op((b), (op), __VA_ARGS__)
extern "C" {
#else
#define RAD_PARAMS(...)   ((const RadParam[]){ __VA_ARGS__ }), \
                          (int)(sizeof((const RadParam[]){ __VA_ARGS__ }) / sizeof(RadParam))
#define RAD_NOPARAMS      ((const RadParam*)0), 0
#define RAD_WEIGHTS(...)  ((const rad_weight[]){ __VA_ARGS__ }), \
                          (int)(sizeof((const rad_weight[]){ __VA_ARGS__ }) / sizeof(rad_weight))
#define RAD_NOWEIGHTS     ((const rad_weight*)0), 0
#define RAD_OP(b, op, ...) rad_decl_op((b), (op), __VA_ARGS__)
#endif

/* An op's BUFFER operands, which rad_decl_op does not name -- it names weights, because those are
 * what placement moves. But the buffer planner needs liveness, and liveness is exactly which ops
 * read and write which buffers. Without these a transient has no declared use and must be assumed
 * live for the whole program, which is sound (a plugin may issue any handle in any order) and as
 * wasteful as it sounds; --debug-graph prints how many buffers fell into that case so the cost is
 * never invisible.
 *
 * Separate from rad_decl_op rather than more arguments to it, because the common case is one call
 * with one read set and one write set and the alternative is four more parameters on every
 * declaration. */
int rad_op_reads (RadBuilder* b, rad_op h, const rad_buf* bufs, int n);
int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n);

#ifdef __cplusplus
} /* extern "C" */
inline int rad_op_reads (RadBuilder* b, rad_op h, std::initializer_list<rad_buf> bufs) {
    return rad_op_reads(b, h, bufs.size() ? bufs.begin() : nullptr, (int)bufs.size());
}
inline int rad_op_writes(RadBuilder* b, rad_op h, std::initializer_list<rad_buf> bufs) {
    return rad_op_writes(b, h, bufs.size() ? bufs.begin() : nullptr, (int)bufs.size());
}
extern "C" {
#endif

/* A BUFFER THE ARCHITECTURE TOUCHES FROM THE SECOND LANE, which is the one case where the op
 * range a buffer's declarations imply is NOT a bound on when its bytes are in use.
 *
 * The planner packs transients from ONE LINEAR ORDER (spec §3.1): two whose declared lives do not
 * overlap share arena bytes. `rad_lane` breaks that assumption rather than the declarations --
 * lane 1's ops are issued at op indices that sit inside lane 0's, so a shared-arm scratch whose
 * life is "ops 41..42" can be laid over a routed-arm scratch whose life is "ops 37..38", and on
 * two streams those two runs OVERLAP IN TIME. Nothing downstream can tell: it reads as data.
 *
 * So a buffer named here is given the WHOLE PROGRAM, which is what a buffer with no declared use
 * already gets and is sound for the same reason. It is deliberately per-buffer and not a mode:
 * unsharing every buffer would cost ~220 MiB on a model this size, where the lane-1 transients
 * of one block are a few MiB. Call it at declare, for every buffer the
 * second lane reads or writes -- the ones lane 0 touches in the same window are covered by
 * symmetry, since sharing needs both sides.
 *
 * Calling it for a PERSIST or DERIVED buffer is a no-op: those are live everywhere already. */
int rad_buf_concurrent(RadBuilder* b, rad_buf buf);

/* AN UNEVEN SHARD: this rank's slice of a RAD_SHARD_ROW or RAD_SHARD_COL weight's split dimension
 * is [lo, hi) and not the even 1/tp of it -- rows of EACH stacked row part for ROW, columns for
 * COL, in the container's (unsharded) units. The declared extent must already be `hi - lo` (each
 * row part's, for ROW).
 *
 * For a width that does not divide into equal whole blocks across the ranks: a 640-column
 * projection with 128-column scale blocks is five blocks, which two ranks can split three and two
 * but not in half. The stacked parts of a ROW weight must be equal in the container, since only
 * the rank's own part sizes are declared. Refused by a container converted for more than one
 * rank, whose entries are already even shares, and by a stored layout whose row runs or column
 * groups the span would cut. */
int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi);

#ifdef __cplusplus
#define RAD_BUFS(...)  std::initializer_list<rad_buf>{ __VA_ARGS__ }
#else
#define RAD_BUFS(...)  ((const rad_buf[]){ __VA_ARGS__ }), \
                       (int)(sizeof((const rad_buf[]){ __VA_ARGS__ }) / sizeof(rad_buf))
#endif

/* Did the last rad_decl_op resolve? A plugin asks for the fused op first and emits the unfused
 * sequence when it does not, which is what makes fusion a selection rather than a compiler pass
 * (spec §2.3). A handle that did not resolve is RAD_NULL_HANDLE and is NOT an error yet -- it
 * becomes one only if the plugin issues it. */
int rad_op_resolved(RadBuilder* b, rad_op h);

/* ================================================================== checkpoint name map */
/* The plugin owns the map from checkpoint tensor names to declared weight names, including
 * convert-time fusions -- q/k/v into a single qkv, gate and up into gate_up. It is the only
 * component that knows both sides, and rad-convert reads the map from it (spec §2.4). */
enum { RAD_MAP_COPY = 0, RAD_MAP_CONCAT = 1 };

typedef struct RadNameMap {
    const char* declared;        /* the name given to rad_decl_weight */
    int         mode;            /* RAD_MAP_COPY | RAD_MAP_CONCAT */
    int         n_src;
    const char* src[8];          /* checkpoint tensor names */
    int         concat_dim;

    /* A sub-tensor index within src[i], or -1 for the whole tensor. GGUF stores a MoE layer's
     * experts stacked as one ffn_gate_exps of shape [.., n_expert], while placement needs one
     * declared weight per expert -- the movement unit is an expert, not a tensor. Without this
     * the expert index survives only inside the declared NAME, which is an unwritten convention
     * rad-convert would have to parse. */
    int         src_index[8];
} RadNameMap;

/* RAD_MAP_COPY with n_src > 1 means FIRST SOURCE PRESENT IN THE CONTAINER WINS. That is how a
 * tied lm_head is expressed -- output.weight if the checkpoint has one, else token_embd.weight --
 * and without it a model family whose members differ only in tying needs two plugins.
 * RAD_MAP_CONCAT with n_src > 1 concatenates all of them along concat_dim and requires every one. */

int rad_decl_name_map(RadBuilder* b, const RadNameMap* m);


/* ================================================================== the step's output */
/* WHICH BUFFER HOLDS THIS STEP'S LOGITS. The core samples from it -- the sampler chain is core
 * logic and its ops resolve through the same hierarchy at the same time as the plugin's (spec
 * §13) -- so the core has to be told which buffer that is, and the plugin is the only one who
 * knows.
 *
 * A name convention would do it and is wrong: a drafter or an encoder tower declares into a SCOPE
 * (RadBuildCtx::scope, spec §10, §11), so "the buffer called logits" is two different buffers in
 * one Program and matching on the string would sample the draft model's distribution into the
 * target's tokens. It is one call and it removes a string match from startup.
 *
 * Its extent along dim 0 bounds RadBatch::n_out: the buffer holds one row per sampled position,
 * not one per token of the step. Declaring it [max_tok, n_vocab] is legal and, at a 248K
 * vocabulary, 8 GiB.
 *
 * A Program with no logits buffer cannot serve, and the engine says so at startup rather than at
 * the first step. An embedding-only or encoder-only plugin declares none, deliberately. */
int rad_declare_logits(RadBuilder* b, rad_buf logits);


/* ================================================================== the drafter */
/* WHAT THE PLUGIN'S DRAFT HEAD IS, declared the same way and for the same reason the logits
 * buffer above is: the core drives the extra passes, so it has to be told what it is driving,
 * and the plugin is the only one who knows.
 *
 * IT REPLACES A WEIGHT-NAME MATCH. The alternative is for the engine to decide whether a
 * container drafts by scanning the declared weights for names like `mtp.fc.weight` or
 * `dflash.fc.weight`, and then to re-read the drafter's own configuration keys -- block size,
 * mask token, sliding window, predicted-layer count -- every one of them a key the plugin has
 * already read to build the thing. Two readers of one fact is two places for it to disagree, and
 * the second reader would be the core, which is the one that must not know.
 *
 * KIND IS A SHAPE, NOT A MODEL. There are two ways a draft head can be driven and the difference
 * is visible from outside it: SERIAL runs `depth` dependent passes, each embedding what the last
 * one proposed, so the tokens arrive one at a time and a pass that declines ends the chain.
 * BLOCK runs a context pass and then ONE pass that fills every drafted position at once. MTP,
 * EAGLE and Medusa-serial are all SERIAL; DFlash2 and any other block-diffusion drafter is
 * BLOCK. A plugin picks the shape that describes it and needs no core change to do it.
 *
 * `proposal` is the buffer a pass writes its token ids into, and 0 means "the sampler's token
 * buffer" -- which is what a head whose logits go through the ordinary sampler chain wants. A
 * by-name lookup cannot supply this field: it would have to suffix-match a name that is not
 * unique in the Program, in exactly the way the comment above rad_declare_logits explains is
 * wrong -- a drafter declares into its own scope.
 *
 * A plugin that declares no drafter is not an error and is the normal case; the scheduler's
 * prompt-lookup drafter still runs. */
typedef enum {
    RAD_DRAFT_NONE   = 0,
    RAD_DRAFT_SERIAL = 1,   /* `depth` dependent passes, one proposal each */
    RAD_DRAFT_BLOCK  = 2,   /* a context pass, then one pass that fills the whole block */
} RadDraftKind;

typedef struct RadDrafterDecl {
    const char* name;            /* "mtp", "dflash2" -- for messages and the graph dump */
    int         kind;            /* RAD_DRAFT_* */
    int         depth;           /* tokens proposed a step; must equal RadBuildCtx::max_spec */

    /* Where a pass leaves its proposal. 0 = the sampler's token buffer. */
    rad_buf     proposal;
    int64_t     proposal_pitch;  /* elements a row in `proposal`; 0 means 1 */

    /* BLOCK only. `mask_token` is the id every unfilled position of the block embeds, and a
     * block drafter without one has nothing to run. `window` is the drafter's OWN sliding
     * window over its OWN cache, 0 for unbounded -- the core stages the block pass against it,
     * so it must be the same number the plugin gated its attention with. */
    int32_t     mask_token;
    int64_t     window;

    /* BLOCK only: HOW MANY ROWS THE QUERY PASS RUNS OVER, one sequence's worth. 0 means
     * `depth + 1`.
     *
     * IT IS NOT ALWAYS depth + 1, AND THE DIFFERENCE IS A ROW ALIGNMENT. The core stages the
     * block as an anchor -- the token the sampler just chose -- followed by `block - 1` copies of
     * `mask_token`, and THE HEAD RUNS OVER THE LAST `depth` ROWS. That one rule covers both
     * conventions a block-diffusion drafter can be trained in:
     *
     *   block = depth + 1    the anchor row predicts NOTHING and each mask row predicts the token
     *                        at ITS OWN position. The head sees rows 1..depth. DFlash2.
     *   block = depth        every row predicts the token at the position AFTER it, the anchor
     *                        row included -- the ordinary next-token relation, so the anchor is
     *                        not a wasted row. The head sees rows 0..depth-1. DSpark, whose
     *                        reference takes `hidden[:, -block_size:, :]` over exactly this.
     *
     * GETTING IT WRONG IS INVISIBLE FROM THE OUTSIDE. Both values run, both produce `depth`
     * tokens, and the model's own output stays exactly correct because every draft is verified;
     * what happens instead is that the proposals are shifted by one position and acceptance
     * collapses to roughly the rate of chance. That is why this is declared by the plugin that
     * knows how its checkpoint was trained, and not inferred here. */
    int64_t     block;
} RadDrafterDecl;

int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d);

/* ================================================================== the encoder */
/* A MEDIA ENCODER THE PLUGIN RUNS ON AN ENCODER PASS (RadBatch::enc), declared for the reason the
 * logits buffer and the drafter are: the core drives the pass and reads its result, so it has to
 * be told what it is driving, and only the plugin knows. A plugin that declares none serves text,
 * and the server refuses a media part by name.
 *
 * WHAT IS HERE IS THE SHAPE OF THE INTERFACE, NOT THE MODEL. The core stages patches of
 * `patch_dim` elements and reads back one `n_embd`-wide row for every `merge` of them; what turns
 * the one into the other -- the tower's depth, its attention, its position scheme -- is the
 * plugin's business and appears nowhere in the core. How an image or a video becomes patches is
 * the processor's (core/mm), which reads its geometry from the container: the two agree through
 * `patch_dim` and `merge`, and the core refuses a program whose encoder disagrees with the
 * processor about either. */
enum {
    RAD_MM_IMAGE = 1u << 0,
    RAD_MM_VIDEO = 1u << 1,
};

typedef struct RadEncoderDecl {
    const char* name;            /* "vision" -- for messages */
    uint32_t    modalities;      /* RAD_MM_* this encoder serves */
    int64_t     patch_dim;       /* elements in one input patch row */
    int64_t     merge;           /* input patches per output row */
    int64_t     max_patches;     /* must equal RadBuildCtx::max_enc_patches */
    int64_t     n_embd;          /* the width of an output row, the model's embedding width */
    /* [max_patches / merge, n_embd] bf16: where a pass leaves its rows. The core reads it after
     * the pass and before any other pass runs, so the planner may share its bytes with buffers
     * no encoder pass touches. */
    rad_buf     out;
} RadEncoderDecl;

int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d);

/* ================================================================== diagnostics */
/* A plugin may annotate the graph dump. Free-form, printed under --debug-graph. */
void rad_note(RadBuilder* b, const char* fmt, ...);

/* ================================================================== the arch plugin exports */
const char* rad_arch_id(void);

/* THE QUANTISATION THIS PLUGIN SERVES, and the second half of the key that selects it.
 *
 * spec §2.4 is one plugin per (architecture, quantisation): a w4a8 model wants
 * rmsnorm_had_quant_i8 where a bf16 model wants a plain rmsnorm, and a plugin full of branches on
 * a quant descriptor is how you get back to "unclear what actually runs". So `qwen35_bf16` and
 * `qwen35_fp8` both answer "qwen35" to rad_arch_id and are told apart HERE, by the string the
 * container already carries in RadModelMeta::quant.
 *
 * The match is EXACT and there is no fallback. An unquantised container and an unquantised plugin
 * both say "" -- the symbol is optional and a plugin that does not export it is read as "".
 * Falling back from a quantised container to a plugin that declares dense weights would fail
 * later, at the name map, with a worse message than "no plugin claims qwen35 at fp8_e4m3; qwen35
 * is served at: (unquantised)". */
const char* rad_arch_quant(void);

/* WHAT THE PLUGIN CAN SAY BEFORE IT DECLARES ANYTHING. OPTIONAL: a plugin that does not export
 * it answers every question with the zero, which is each field's default.
 *
 * It exists for the ONE fact that is an INPUT to declaration and a property of the model:
 * the draft depth. `RadBuildCtx::max_spec` sizes buffers, bands kernels and sets the scheduler's
 * window, so it must be fixed before rad_arch_declare runs -- but "how deep does this container's
 * drafter want to go" is answerable only by reading the drafter's own configuration, which is the
 * plugin's job. The alternative is a core that reads those keys itself, by name, with an
 * architecture's own prefix in the string.
 *
 * A serial head names a THROUGHPUT TRADE-OFF: more rounds cost a pass each and acceptance decays,
 * so the number is measured, not derived. A block drafter names the block length ITS CHECKPOINT
 * WAS TRAINED AT, and asking for anything else is a silently worse draft. Both are the plugin's
 * to know. 0 means "this container carries no drafter", which is the right answer to "speculate
 * as deeply as this model can" on a model that cannot.
 *
 * --num-speculative-tokens overrides whatever comes back, and the plugin sees the result in
 * RadBuildCtx::max_spec; a plugin whose checkpoint cannot serve that depth refuses in declare,
 * where it has the whole configuration in front of it. */
typedef struct RadArchProbe {
    int draft_depth;

    /* WHETHER THAT DEPTH IS THE ONLY ONE THE DRAFTER SERVES, and the two callers of this need
     * different answers. A SERIAL head names a throughput trade-off: more passes cost one each
     * and acceptance decays, so the depth it reports is an operating point and one more pass
     * still runs. A BLOCK drafter
     * names the block length its checkpoint was TRAINED at -- its convolution masks the position
     * within that block -- so any other depth is not a shorter proposal, it is a refusal.
     *
     * `--num-speculative-tokens auto` takes `draft_depth` either way; it is choosing an operating
     * point. rad-convert must not: a CONTAINER carries what the model can serve, so narrowing it
     * to a serial head's preferred depth would mean a deployment asking for one more had to
     * convert again. It widens to `draft_depth` only when this says the drafter has no choice. */
    int draft_depth_fixed;

    /* THE PLUGIN HONOURS RadBuildCtx::shape_probe: a sizing declare leaves the state its step
     * function reads untouched. Zero -- what a plugin that never heard of it leaves -- means the
     * core does not ask, and the activation arena is not lent. */
    int shape_probe_ok;
} RadArchProbe;

int rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out);

/* HOW THIS MODEL SPELLS A TURN, AS DATA. OPTIONAL: a plugin that does not export it leaves the
 * engine to derive the format by analysing the chat template.
 *
 * A reply is read back into reasoning, content and tool calls by one streaming parser driven by
 * this description, and the lazy grammar that constrains a call and the literal words that
 * trigger it are generated from it too -- so a model whose format the template analysis cannot
 * find, or finds wrongly, is supported by stating the format here once. What it describes is the
 * tagged family:
 *
 *   [reasoning_open] reasoning [reasoning_close]  content
 *   [call_open] name_prefix NAME name_suffix
 *       arg_prefix ARG arg_name_suffix arg_value_prefix VALUE arg_value_suffix  ...
 *   call_end [call_close]
 *
 * Qwen3.8, for example, is "<think>\n" / "\n</think>\n\n", calls "<tool_call>\n" / "</tool_call>",
 * "<function=" / ">\n" / "</function>\n", arguments "<parameter=" / ">\n" / "" / "</parameter>\n".
 * Whitespace at the ends of a tag is optional when matching. Every string is NUL-terminated
 * UTF-8; NULL reads as "". The engine copies what it needs while loading, so static storage in
 * the plugin is enough.
 *
 * THE STRUCT GROWS AT ITS END AND NEVER CHANGES IN THE MIDDLE. `struct_size` is
 * sizeof(RadChatFormat) as the plugin was compiled; the engine reads only the fields that size
 * covers and treats the rest as zero, so a plugin built against an older header keeps loading
 * and a field added later is appended with a zero that means "as before". That is why this
 * export does not move RAD_ABI_VERSION: it is additive, and a plugin that lacks it or predates a
 * field is read correctly. A change that is NOT additive -- a field reordered, retyped or
 * removed -- is a layout change and bumps RAD_ABI_VERSION like any other. */
enum {
    RAD_CHAT_VALUE_RAW   = 0,  /* the text between the delimiters, verbatim */
    RAD_CHAT_VALUE_CDATA = 1,  /* verbatim, unless wrapped in <![CDATA[ ... ]]>, which is removed */
    RAD_CHAT_VALUE_JSON  = 2   /* a JSON value; text that does not parse as JSON is taken raw */
};

enum {
    /* The generation prompt shows whether the reply starts inside a reasoning block: whatever
     * follows the first `reasoning_open` in it is the start of the reply's own text. */
    RAD_CHAT_THINK_FROM_PROMPT = 0,
    RAD_CHAT_THINK_ALWAYS      = 1,  /* the reply is inside a reasoning block from its first byte */
    RAD_CHAT_THINK_NEVER       = 2   /* the model opens its own block, if any */
};

typedef struct RadChatFormat {
    uint32_t    struct_size;
    const char* name;                /* for the startup log; NULL is the plugin's name */

    const char* reasoning_open;      /* "" when the model does not reason */
    const char* reasoning_close;
    /* Literals that end a reasoning block the model forgot to close, when a call starts there.
     * Each must be the call opener: `call_open`, or `name_prefix` when there is no call_open. */
    const char* const* reasoning_breaks;
    uint32_t    n_reasoning_breaks;
    uint32_t    reasoning_in_prompt; /* RAD_CHAT_THINK_* */

    /* Around ALL of a turn's calls, and between two calls beyond whitespace. The parser does not
     * read either yet, so a format that needs one is refused at startup by name; they are fields
     * so that supporting them does not change the struct. */
    const char* section_open;
    const char* section_close;
    const char* call_separator;

    const char* call_open;           /* around each call; "" when calls are not wrapped */
    const char* call_close;
    const char* name_prefix;         /* "" when the model has no tool calls */
    const char* name_suffix;
    const char* call_end;
    const char* arg_prefix;
    const char* arg_name_suffix;     /* "" when arg_value_prefix alone ends the name */
    const char* arg_value_prefix;
    const char* arg_value_suffix;

    uint32_t    string_value;        /* RAD_CHAT_VALUE_RAW or _CDATA: a string-typed argument */
    uint32_t    other_value;         /* RAD_CHAT_VALUE_JSON: any other type (or _RAW, as text) */
    /* Non-zero: one newline at each end of a value is punctuation, not part of the value. */
    uint32_t    value_newlines;

    /* The literal words that start constrained decoding of a call. None: no grammar, and the
     * reply is read unconstrained. */
    const char* const* triggers;
    uint32_t    n_triggers;
} RadChatFormat;

/* Return the format, or NULL for "derive it from the chat template". Called once at startup. */
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta);

int         rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
/* Forward declaration; the signature is in rad_runtime.h. */
struct RadCtx;
struct RadBatch;
void        rad_arch_step(struct RadCtx* c, const struct RadBatch* batch);

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_BUILDER_H */
