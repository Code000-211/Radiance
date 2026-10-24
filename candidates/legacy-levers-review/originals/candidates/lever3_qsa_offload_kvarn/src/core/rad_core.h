/* rad_core.h -- the data model shared BETWEEN core components. Each component owns its own
 * header for its own methods; this file owns the structs that cross a component boundary, so
 * two components cannot disagree about what a weight or a band or a request is.
 *
 * If you need to add a field here, add it. If you need to CHANGE one, say so loudly -- somebody
 * else is reading it.
 */
#pragma once
#include "rad_internal.h"

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

namespace mm { struct Item; struct PromptMedia; }

class Registry;
class Builder;
class Program;
class Ctx;
class Arena;
class Pools;
class KVManager;
class PrefixCache;
class Scheduler;
class Planner;
class Mover;
class HeatEngine;
class Sampler;
struct GbnfProgram;
class Tokenizer;
class ChatTemplate;
class RadFile;
class Engine;

/* ================================================================== plugin layer */

/* One kernel row plus where it came from. The hierarchy is absolute: the first plugin with any
 * matching kernel supplies it, and a lower-priority plugin never outbids a higher one no matter
 * how specialised its kernel (spec §2.1). `plugin_order` is that hierarchy position. */
struct KernelRow {
    const RadKernelInfo* info = nullptr;
    std::string          plugin;
    int                  plugin_order = 0;   /* 0 is highest priority */
    int                  index = 0;          /* declaration order within the plugin */
    /* The plugin's rad_kernel_concurrent answer for this row: its launches touch nothing but
     * their operands and their scratch, so one may run beside another that shares none of those
     * bytes. False when the plugin does not say. */
    bool                 concurrent = false;
    /* The row's plugin declared itself a reference implementation (RadPluginReferenceFn). */
    bool                 reference = false;
};

/* A kernel bound to one geometry, one domain and one variant, with init() already run. */
struct Resolved {
    const KernelRow* row = nullptr;
    void*            instance = nullptr;
    /* The frozen geometry, INCLUDING the tuned axes: once selection has picked a point in the
     * kernel's RadTunable space it is set here, so RadArgs.p carries it and the kernel reads `sk`
     * with the same rad_args_geti() it reads `M` with. There is no variant index anywhere. */
    Geometry         geom;
    int64_t          scratch_bytes = 0;
    /* The geometry WITHOUT the tuned axes, in tunecache.h's canonical spelling. The cache keys on
     * it, so it has to be captured BEFORE the choices are folded into `geom` -- otherwise the key
     * would depend on the answer it is used to look up. */
    std::string      geom_key;
    /* The chosen axes, `key=value` comma separated and sorted -- the cache's `choice` column --
     * and where they came from: "axis defaults", "tuned <timestamp>" or "RADIANCE_TUNE" (spec §15).
     * Both empty for a kernel with nothing to tune. */
    std::string      choice;
    std::string      tune_source;

    explicit operator bool() const { return row != nullptr; }
};

/* One row of a bucket table: a band of the ranged parameter, resolved per domain.
 * The bucket boundaries are the constraint values themselves -- take the union of the LE, GE and
 * EQ values that candidate kernels place on the ranged parameter, and those are the bands. No
 * policy was chosen by anyone (spec §2.2). */
struct Band {
    int64_t  hi = 0;                          /* inclusive upper bound */
    Resolved dom[RAD_N_DOMAINS];              /* [RAD_DOMAIN_DEVICE], [RAD_DOMAIN_HOST] */
    std::string miss[RAD_N_DOMAINS];          /* why nothing resolved, for the report */

    /* The band as a reader sees it -- "M <= 64", "M in (64, 512]", "any shape". Carried on
     * the band rather than recomputed at each reporting site, because the exclusive lower
     * bound is the previous band's hi and only the table's builder has that. */
    std::string span;
};

/* ================================================================== the declared program */

/* MAPPED IS NOT "HOST MEMORY", IT IS THE CONTAINER'S OWN MAPPING, and that is the whole of the
 * difference. The other three off-device tiers all COPY: pinned and pageable allocate an arena and
 * memcpy the weight into it at load, and SSD preads it a slot at a time. A mapped weight is never
 * copied and occupies no pool -- the loader publishes a pointer into the mmap that
 * core/format/radfile.cpp already holds, and the OS page cache decides what is resident.
 *
 * It exists for a weight that is tens of gigabytes and read A HANDFUL OF ROWS A TOKEN -- an
 * n-gram embedding table is the shape. Every copying tier is wrong for that: staging the unit
 * moves the whole table to use a few kilobytes of it, pinning it forces all of it resident on a
 * host whose RAM is the same order as the table, and an arena copy wants a second copy of it
 * beside the mapping that already holds one. A sparse Zipfian gather of a dozen rows out of
 * hundreds of millions is exactly what a page cache is good at.
 *
 * Only a HOST kernel can read the mapping -- the bytes are pageable and at no device address -- so
 * the planner assigns this tier where no device kernel for the op exists at all, and where
 * --ngram-placement keeps an n-gram table off the card (embed_lookup_q, libavx/avx_ngram.cpp).
 * Under `ram` the load phase publishes a copy in place of the mapping: locked memory for the host
 * gather, or pinned memory a device kernel reads over the link (the zero-copy site). */
enum class Tier { VRAM = 0, HostPinned = 1, HostPageable = 2, SSD = 3, Mapped = 4 };

/* WHERE A LOOKUP TABLE LIVES: the token embedding (--embedding-placement) and an n-gram embedding
 * (--ngram-placement), each a model-level weight that only a row lookup reads, so a step touches M
 * of its rows and never the whole table. That is what makes every tier below the card usable for
 * it, and why the operator gets to choose.
 *
 *   Vram   resident on the card, a weight like any other.
 *   Ram    in host memory the card or a host kernel reads in place: the embedding in the pinned
 *          host pool, read over the link; an n-gram table as a locked copy made at load.
 *   Disk   read from the container on demand.
 *   Auto   the fastest of the three that has room once every other weight is placed; Engine::plan
 *          decides it and writes the answer into Config, so every rank plans from one choice. */
enum class TablePlacement { Auto = 0, Vram = 1, Ram = 2, Disk = 3 };
const char* table_placement_name(TablePlacement p);
enum class Site { Device = 0, DeviceStaged = 1, DeviceZeroCopy = 2, Host = 3 };

/* Tier-indexed arrays are sized by this rather than by a literal: the report, the live view and
 * the planner's totals all index by (int)Tier and a fifth tier that fits in four of them is an
 * out-of-bounds write that costs nothing until it costs everything. */
constexpr int kTierCount = 5;

const char* tier_name(Tier t);
const char* site_name(Site s);

struct WeightInfo {
    std::string    name;
    RadWeightDecl  decl{};
    int64_t        logical_bytes = 0;   /* from the declared dtype and shape */
    /* An uneven shard's slice of the split dimension (rad_weight_shard_span); hi 0 is the even
     * split. */
    int64_t        shard_lo = 0, shard_hi = 0;

    /* WHAT IT IS A VIEW OF (RadWeightDecl::source and ::planes, spec §4.3). `enc` is the source's
     * encoding as the model holds it -- or, with nothing behind the builder, plain in the declared
     * dtype -- and `sel` the planes this declaration takes, in its order. `sel_rows`/`sel_cols`
     * are each plane's extents for THIS RANK, which is what a layout hook is shown. */
    std::string    source;
    RadEncoding    enc{};
    bool           enc_known = false;
    int32_t        n_sel = 0;
    int32_t        sel[RAD_ENC_MAX_PLANES] = {0};
    int64_t        sel_rows[RAD_ENC_MAX_PLANES] = {0};
    int64_t        sel_cols[RAD_ENC_MAX_PLANES] = {0};

    /* Filled by the layout pass: what the resolved kernel stores for it. `identity` is the
     * selected plane's bytes unchanged -- the only form the file and mapped tiers can serve,
     * since they read the container where it lies; `widen` is identity but for an exact widening
     * (a bf16 norm read as f32) the loader performs. Otherwise the kernel at (lay_op, lay_band,
     * lay_dom) relayouts it at load, as operand `lay_operand`. */
    std::string    layout_tag;
    bool           identity = true;
    bool           widen = false;
    int32_t        lay_op = -1, lay_band = -1, lay_dom = -1, lay_operand = -1;
    uint32_t       stored_dtype = RAD_DT_INVALID;
    uint32_t       stored_rank = 0;
    int64_t        stored_shape[RAD_MAX_RANK] = {0};   /* what the kernel will receive. A layout
                                       * hook may reshape -- WMMA fragment order is not the
                                       * declared shape -- and the kernel has to be handed the
                                       * shape it actually gets, not the logical one. */
    int64_t        stored_bytes = 0;
    int64_t        align = RAD_ALIGN_UNIT;

    /* Filled by the planner (spec §5.3). */
    Tier           tier = Tier::VRAM;
    Site           site = Site::Device;
    int32_t        slab_slot = -1;      /* -1 = not slab-resident */

    /* Filled at load. Under TP this is THIS RANK's shard, not the whole weight. */
    void*          ptr = nullptr;
    void*          host_ptr = nullptr;
    uint64_t       file_offset = 0;
    std::atomic<uint32_t> generation{0}; /* bumped when the mover relocates it */

    /* Ordering, so layer offload is a scheduling problem and not a prediction one: this is the
     * first op index that reads the weight, and prefetch runs off it (spec §5). */
    int32_t        first_use_op = -1;
    int32_t        last_use_op  = -1;

    /* The atomic makes the copy hand-written, so a field added above has to be added here too.
     * The weight list reallocates through this copy as the plugin declares, so a field left out
     * is reset to its default on every weight declared before the last growth. */
    WeightInfo() = default;
    WeightInfo(const WeightInfo& o) { *this = o; }
    WeightInfo& operator=(const WeightInfo& o) {
        name = o.name; decl = o.decl; logical_bytes = o.logical_bytes;
        shard_lo = o.shard_lo; shard_hi = o.shard_hi;
        source = o.source; enc = o.enc; enc_known = o.enc_known; n_sel = o.n_sel;
        for (int i = 0; i < RAD_ENC_MAX_PLANES; ++i) {
            sel[i] = o.sel[i]; sel_rows[i] = o.sel_rows[i]; sel_cols[i] = o.sel_cols[i];
        }
        layout_tag = o.layout_tag; identity = o.identity; widen = o.widen;
        lay_op = o.lay_op; lay_band = o.lay_band; lay_dom = o.lay_dom; lay_operand = o.lay_operand;
        stored_dtype = o.stored_dtype;
        stored_bytes = o.stored_bytes; align = o.align;
        stored_rank = o.stored_rank;
        for (int i = 0; i < RAD_MAX_RANK; ++i) stored_shape[i] = o.stored_shape[i];
        tier = o.tier; site = o.site; slab_slot = o.slab_slot;
        ptr = o.ptr; host_ptr = o.host_ptr; file_offset = o.file_offset;
        generation.store(o.generation.load());
        first_use_op = o.first_use_op; last_use_op = o.last_use_op;
        return *this;
    }
};

/* THIS RANK'S SELECTED PLANES AS A LAYOUT HOOK SEES THEM: dense rank-2 geometry, `data` null
 * unless the caller fills it. One definition, because the builder asks the hook with it at
 * declare and the runtime asks again before moving an op to another domain, and the two
 * questions have to be the same question. */
inline void weight_planes(const WeightInfo& w, RadTensor* t) {
    for (int k = 0; k < w.n_sel; ++k) {
        t[k] = RadTensor{};
        t[k].dtype = w.enc.plane[w.sel[k]].dtype;
        t[k].rank = 2;
        t[k].shape[0] = w.sel_rows[k];
        t[k].shape[1] = w.sel_cols[k];
        t[k].stride[0] = w.sel_cols[k];
        t[k].stride[1] = 1;
    }
}

/* CAN THE BYTES BE READ WHERE THEY LIE IN THE CONTAINER -- the file tier's O_DIRECT stage, the
 * mapped tier's pointer? Only when they are the selected plane's bytes unchanged and this rank
 * holds all of it: a relayout has no place in the file that already looks the way the kernel
 * reads it, and a column shard is not one range of it. */
inline bool weight_file_direct(const WeightInfo& w, int world_size) {
    return w.identity && !w.widen && (w.decl.shard == RAD_SHARD_NONE || world_size <= 1);
}

struct BufferInfo {
    std::string name;
    RadBufDecl  decl{};
    int64_t     bytes = 0;

    /* From the buffer plan: liveness over the declared op list, and the arena offset the plan
     * assigned. Two transients whose lives do not overlap share storage (spec §3.1). */
    int32_t     first_def = -1;
    int32_t     last_use  = -1;
    int64_t     arena_offset = -1;
    void*       ptr = nullptr;
};

struct OpInfo {
    std::string           op;
    Geometry              base;         /* the declared params, ranges NOT collapsed */
    std::string           ranged_key;   /* "" if none */
    int64_t               range_lo = 0, range_hi = 0;
    std::vector<Band>     bands;
    std::vector<rad_weight> weights;
    /* Which SCHEMA OPERAND POSITION each declared weight belongs to, parallel to `weights`.
     *
     * It is NOT the index. The n-th declared weight is not the n-th operand whose role is
     * RAD_OPD_WEIGHT, because RAD_OPD_WTAB lets one operand position consume a whole RUN of
     * declared weights (a MoE layer's experts), so the mapping is many-to-one. Recorded once at
     * declare rather than recomputed at each site that needs it: the recomputation is easy to get
     * wrong, and one recorded answer is one the sites cannot disagree about. */
    std::vector<int32_t>  weight_opd;
    const RadOpSchema*    schema = nullptr;
    int32_t               index = 0;    /* position in the declared order */
    bool                  issued = false;   /* did the run phase ever issue it */
};

struct KVGroupInfo {
    std::string     name;
    RadKVGroupDecl  decl{};
    int64_t         block_size = 0;     /* taken off the resolved attention kernel, not chosen */
    int64_t         bytes_per_block = 0;
    int64_t         bytes_per_state = 0;   /* linear/conv: per sequence */
    std::vector<int> layers;
};

/* A RAD_KV_CONV GROUP'S WINDOW, IN ENTRIES: conv_width - 1 of history and one more for every
 * speculative position. It is the last extent of the state tensor the kernels index, and the
 * kernels take the layout from that extent -- a slot is `entries` consecutive entries a channel,
 * and libr4d's conv update derives the verify step's query length from it as
 * entries - (conv_width - 2). So the allocation's slot, the KV manager's view of a slot (its clear
 * when one is handed out, a checkpoint's save and restore) and the cursor conv_state_index derives
 * all take this one number. A slot sized larger on the manager's side than in the tensor puts
 * each of those views across the start of the next sequence's window. */
inline int64_t conv_window_entries(const RadKVGroupDecl& d, int64_t n_spec) {
    return (d.conv_width > 0 ? d.conv_width - 1 : 0) + (n_spec > 0 ? n_spec : 0);
}

/* Everything declare produced. The core owns it; the plugin only ever sees handles into it. */
class Program {
public:
    std::vector<WeightInfo>  weights;    /* index 0 is a sentinel: handles are 1-based */
    std::vector<BufferInfo>  buffers;
    std::vector<OpInfo>      ops;
    std::vector<KVGroupInfo> kv_groups;
    std::vector<RadNameMap>  name_map;
    std::vector<std::string> notes;

    RadBuildCtx  ctx{};
    RadModelMeta meta{};

    /* The buffer the plugin named as this step's logits (rad_declare_logits). Zero when the
     * plugin declared none, which is what an encoder-only or embedding-only plugin does on
     * purpose -- the engine refuses to SERVE such a program and says so at startup, rather
     * than discovering at the first step that there is nothing to sample. */
    rad_buf logits_buf = 0;

    /* THE DRAFT HEAD THIS PLUGIN DECLARED (rad_declare_drafter). `kind == RAD_DRAFT_NONE` when
     * it declared none, which is the normal case -- the engine then speculates with the
     * scheduler's prompt-lookup drafter or not at all. `name` is kept beside the decl because
     * RadDrafterDecl::name points into the plugin's own storage. */
    RadDrafterDecl drafter{};
    std::string    drafter_name;

    /* THE MEDIA ENCODER THIS PLUGIN DECLARED (rad_declare_encoder). `modalities == 0` when it
     * declared none, which is every text-only program and every deployment that asked for no
     * media (RadBuildCtx::max_enc_patches 0). `encoder_name` for the reason `drafter_name` is. */
    RadEncoderDecl encoder{};
    std::string    encoder_name;

    /* Two arenas, because a host-site activation cannot index device memory. The buffer plan
     * colours the domains SEPARATELY -- a transient may share storage only with a transient in the
     * same domain -- so each of these is a high-water mark in its own offset space and
     * BufferInfo::arena_offset is relative to the arena of that buffer's declared domain. Both are
     * sized for max_tok, which is the worst step (spec §3.1). */
    int64_t arena_bytes = 0;         /* RAD_DOMAIN_DEVICE */
    /* Where the device arena's TRANSIENTS start: everything below is PERSIST and DERIVED, at the
     * same offset at every size a step can be, and everything above is the part a smaller step
     * packs smaller (core/build/rad_bufplan.cpp, plan_level). */
    int64_t arena_fixed_bytes = 0;
    /* THE PREFILL STAGING REGION, above every buffer the plan placed: where a step at the full
     * plan copies a routed layer's non-resident experts ahead of the layer (core/place/stager.h).
     * Sized by the VRAM budget, appended by the engine after it, and part of the top of the arena
     * that the smaller levels lend to the expert slab. -1 / 0 where there is none. */
    int64_t arena_stage_off = -1;
    int64_t arena_stage_bytes = 0;
    int64_t host_arena_bytes = 0;    /* RAD_DOMAIN_HOST */
    int64_t scratch_bytes = 0;       /* the largest single kernel scratch requirement */
    /* How many scratch regions of that size the arena holds: one per lane the program runs.
     * Every kernel on a stream is handed the same region, which its in-order launches make safe;
     * a second lane is a second stream, and a region both lanes wrote would take two kernels'
     * partials at once. Two when the architecture declared a buffer rad_buf_concurrent. */
    int32_t scratch_regions = 1;

    /* Everything that resolved to nothing, reported whole rather than the first failure at the
     * top (spec §3.1).
     *
     * STRUCTURED, not a pre-formatted sentence, because the two domains want two different
     * reports and one flat list of strings cannot give them. A DEVICE miss is a hole a request
     * will fall into and deserves its own line; a HOST miss removes one placement option for one
     * op and deserves a line per OP, not per band. A flat list of strings leaves the reporting
     * site able only to print all of them, and a deep model at tp2 then opens with thousands of
     * lines of `E` -- every one of them expected, and a real device hole invisible among them. */
    struct Miss {
        std::string op;
        std::string geom;         /* the band's display geometry */
        std::string span;         /* the band, as Band::span */
        std::string why;
        int         domain = 0;   /* RAD_DOMAIN_DEVICE | RAD_DOMAIN_HOST */
        std::string line() const; /* "op  geometry  [domain] -> why", the full-detail rendering */
    };
    std::vector<Miss> misses;

    /* Every (kernel, shape) pair rad-tune would need to benchmark (spec §15). */
    /* `info` is the kernel's own row, so rad-tune can read its RadTunable axes and expand the
     * cross product itself. A count would not do: the tuner has to know the KEYS and their legal
     * values to build a candidate, and the validity predicate to discard the ones this geometry
     * cannot run. Plugins outlive the Program, so the pointer is good for the process. */
    /* ONE PER KERNEL INSTANTIATION: the kernel, the plugin it came from, and the input parameters
     * it was resolved at. `geom` is what a launch is handed and therefore CARRIES the axis
     * defaults the selector folded in -- a tuner overrides them per candidate with set_i.
     * `geom_key` is the instantiation WITHOUT them, in tunecache.h's canonical spelling, and it is
     * the cache key: the selector looks an instantiation up before it knows what to choose, so a
     * key that included the choice could never be found. */
    struct TuneRequest {
        std::string          kernel, plugin;
        Geometry             geom;
        std::string          geom_key;
        const RadKernelInfo* info = nullptr;
    };
    std::vector<TuneRequest> tune_requests;

    WeightInfo* weight(rad_weight h) { return h && h < weights.size() ? &weights[h] : nullptr; }
    BufferInfo* buffer(rad_buf h)    { return h && h < buffers.size() ? &buffers[h] : nullptr; }
    OpInfo*     op(rad_op h)         { return h && h < ops.size()     ? &ops[h] : nullptr; }

};

/* ================================================================== requests */

enum class ReqState { Waiting, Running, Preempted, Finished, Failed, Cancelled };

/* A CHAT REPLY'S REASONING BLOCK, AS TOKENS. The reply parser reads the block in text
 * (core/text/chatparse.h); this is the same block named by token id, so the engine can count it
 * and close it without detokenising. It exists only where both markers the reply needs are
 * single tokens -- `<think>` and `</think>` are, on every model this engine serves -- and
 * `close < 0` says it does not: such a reply's reasoning is still split out as text, but it is
 * not counted and no budget can be held on it.
 *
 * The block is read where the parser reads it: at the start of the reply. It is open from the
 * first token when the generation prompt opened it (`starts_open`), or opens on `open` as the
 * reply's first token, and the first `close` ends it. A reply that starts any other way has no
 * block. */
struct ReasoningSpan {
    int32_t open = -1;            /* the token a reply opens its own block with; -1 for none */
    int32_t close = -1;           /* the token that ends the block; -1: no countable block */
    bool    starts_open = false;  /* the generation prompt left the reply inside the block */
    bool known() const { return close >= 0; }
};

/* A reply's position relative to its ReasoningSpan, advanced one emitted token at a time. The
 * server counts with it and the sampler enforces a budget with it, and both must agree on where
 * the block ends, so there is one definition. */
struct ReasoningCount {
    enum State : uint8_t { Lead, Think, Done };
    State   st = Lead;
    int64_t tokens = 0;           /* reasoning tokens: the markers the reply wrote, and the body */
    int64_t body = 0;             /* the tokens between the markers, which a budget limits */

    void start(const ReasoningSpan& s) {
        st = !s.known() ? Done : s.starts_open ? Think : Lead;
        tokens = body = 0;
    }
    void push(const ReasoningSpan& s, int32_t tok) {
        if (st == Lead) {
            if (s.open >= 0 && tok == s.open) { st = Think; ++tokens; }
            else st = Done;
        } else if (st == Think) {
            ++tokens;
            if (tok == s.close) st = Done;
            else ++body;
        }
    }
    bool inside() const { return st == Think; }
};

struct SamplingParams {
    float  temp = 1.0f;
    int    top_k = 0;                /* 0 = off */
    float  top_p = 1.0f;
    float  min_p = 0.0f;
    float  typical_p = 1.0f;
    float  rep_penalty = 1.0f;
    float  freq_penalty = 0.0f;
    float  pres_penalty = 0.0f;
    int    penalty_last_n = 64;
    float  dry_multiplier = 0.0f;
    float  dry_base = 1.75f;
    int    dry_allowed_length = 2;
    int    dry_penalty_last_n = -1;
    float  xtc_probability = 0.0f;
    float  xtc_threshold = 0.1f;
    uint64_t seed = 0;
    bool   greedy() const { return top_k == 1 || temp <= 0.0f; }
    std::string grammar;             /* GBNF, or "" */
    std::vector<std::string> dry_seq_breakers;

    /* THE REPLY'S REASONING BLOCK, and how many tokens its body may run to. -1 is no limit. At
     * the limit the sampler masks every position still inside the block to `reasoning.close`, so
     * the block ends there and the answer follows -- vLLM's `thinking_token_budget`. A budget
     * needs a known span; the server refuses one it cannot hold. */
    ReasoningSpan reasoning;
    int32_t       thinking_budget = -1;

    /* A LAZY GRAMMAR CONSTRAINS NOTHING UNTIL A TRIGGER FIRES, and that is the only shape in
     * which a tool-calling chat template can be enforced: the reply is free prose until the
     * model starts writing a call, and grammar-shaped from there. Without this a request that
     * carries `tools` would be constrained to emit a tool call and nothing else, so a lazy
     * grammar arriving as a non-lazy one is not a small difference -- it is a model that can no
     * longer answer in words. The trigger types are common_grammar_trigger_type's, kept as ints
     * because rad_core.h does not include the lift: 0 token, 1 word, 2 pattern, 3 pattern-full.
     * Resolving a word or a token to vocabulary ids happens in the sampler, which is the layer
     * that has the vocabulary. */
    struct GrammarTrigger { int type = 0; std::string value; };
    bool                        grammar_lazy = false;
    std::vector<GrammarTrigger> grammar_triggers;

    /* Text the grammar has already seen: a chat template's grammar describes the whole assistant
     * turn, generation prompt included, but those bytes are in the prompt rather than in what is
     * about to be sampled. See GrammarSpec::prefix in core/sample/grammar.h. */
    std::string                 grammar_prefix;

    /* `grammar` COMPILED, by the thread that admitted the request (core/sample/gbnf.h,
     * gbnf_compile). Compiling a grammar builds its mask plan, which is host work in the tens to
     * hundreds of milliseconds, and the sampler starts a request on the scheduler thread with
     * every other request's step waiting on it. Holding the program here keeps it findable by
     * the sampler for as long as the request lives, so starting the request there is a lookup.
     * Null for a request that was not admitted through the server; the sampler then compiles. */
    std::shared_ptr<const GbnfProgram> grammar_program;
};

struct Request {
    uint64_t             id = 0;
    std::vector<int32_t> prompt;
    std::vector<int32_t> output;
    SamplingParams       sp;
    int32_t              max_tokens = 0;
    std::vector<std::string> stop;
    /* GENERATE PAST THE END-OF-TEXT TOKEN, to max_tokens. It exists so a caller can ask for a
     * request whose LENGTH IS THE ONE IT ASKED FOR. Without it the length is the model's opinion
     * and varies from run to run on the same prompt, so a steady-state decode measurement over
     * such a request may be measuring a mostly idle engine, and its variance reads as a
     * difference between whatever was being compared.
     *
     * It ignores the EOS TOKEN and nothing else: `stop` strings and max_tokens still end a
     * request, which is vLLM's meaning of the field and the one a client already expects. */
    bool                 ignore_eos = false;
    int                  priority = 0;
    bool                 stream = false;

    ReqState             state = ReqState::Waiting;
    int32_t              n_computed = 0;      /* prompt tokens already in the KV */
    int32_t              n_cached = 0;        /* of those, how many came from the prefix cache */
    std::vector<int32_t> blocks;              /* per KV group, flattened by group offset */
    int32_t              state_slot = -1;     /* linear/conv state slot */

    /* THE MEDIA THIS PROMPT CARRIES, null for a text prompt: each item's patches, where its
     * token run sits in `prompt`, the rotary layout those runs give the prompt, and -- once the
     * encoder has run -- the rows that replace the placeholders' embeddings (core/mm). */
    std::shared_ptr<mm::PromptMedia> media;

    int32_t              n_draft = 0;         /* speculative window this step */
    /* AND THE PROPOSAL ITSELF, because masking a speculative window needs the tokens and not
     * just the count: Sampler::build_step walks a constraining grammar over these to mask the
     * position behind each one, then rewinds. Points into the scheduler's own array for this
     * request -- SchedReq owns this Request, so it outlives it -- and is written beside n_draft,
     * in the same two places, for the reason the comment there gives. Null when n_draft is 0. */
    const int32_t*       draft = nullptr;
    int32_t              n_accepted = 0;      /* from the previous verify */

    /* MAY THIS REQUEST BE SPECULATED ON AT ALL. Written by the engine after every step, read by
     * the scheduler when it sizes the next one. One thing clears it, and it is a property the
     * scheduler cannot see for itself: its grammar is constraining right now. The mask for draft
     * position i depends on which of 0..i-1 were accepted, and the host automaton cannot know
     * that before the verify runs. A LAZY grammar clears this only once its trigger fires, so a
     * tool-calling request speculates through its prose and stops the moment it starts writing
     * the call.
     *
     * TEMPERATURE DOES NOT CLEAR IT. A sampled target does not need rejection sampling to stay
     * exact: acceptance compares against the token the target itself drew, and the sampler gives
     * every verified position its own RNG coordinate. See Engine::collect_step.
     *
     * It is a flag rather than a call because there are two drafters and an n-gram fallback inside
     * step() itself, and every one of them has to obey: the fallback fills any proposal the others
     * left empty, so guarding only the drafters guards nothing. */
    bool                 may_speculate = true;
    /* AND WHETHER THAT ANSWER CAN CHANGE AT A COMMIT, which only a grammar that cannot rewind
     * makes it do. A step issued ahead is sized before the commit in front of it, so the scheduler
     * issues ahead for a request only when this is set. Written by the engine when the sampler
     * starts the request (Sampler::speculation_fixed); false until then. */
    bool                 speculation_fixed = false;

    /* A LOGITS ROW AT EVERY PROMPT POSITION, not only at the last: each trunk step that carries
     * this request's prompt names all of its tokens in `out_ids`, after the rows the sampler
     * reads, and the engine hands them to the KL scorer (core/kld.h). Only the --kld mode sets
     * it; the logits buffer is sized for it there and nowhere else. */
    bool                 score = false;

    std::string          finish_reason;
    void*                sink = nullptr;      /* the server's per-request stream sink */
};

/* ================================================================== engine config */

struct Config {
    std::string model;                 /* path to the .rad */
    std::string radiance_home;
    std::vector<std::string> kernel_hierarchy;   /* ordered; first wins */

    int   tp = 1;
    /* THE TENSOR-PARALLEL WIRE, and it is a deployment choice rather than a tuning one. 1 is the
     * exact bf16 all-reduce. 0 accepts a LOSSY wire: at two ranks libr4d serves a
     * Walsh-Hadamard-rotated 6-bit payload -- 6 bits an element plus a bf16 scale per group of
     * 64, against 16 bits an element -- so the message a prefill chunk pushes across PCIe drops
     * by 2.4x. The sum a rank receives is then not the sum it would have received and no check
     * downstream can tell, which is why it is a named flag and not a heuristic. */
    int   tp_wire_exact = 1;          /* --tp-wire exact|wht6 */
    /* WHERE THE LOSSY WIRE STARTS PAYING, in KiB of one all-reduce message. Compression only wins
     * once the transfer is bandwidth-bound: a decode step all-reduces a handful of residual rows
     * and the exact kernel already moves that at most of what the link delivers, so rotating it
     * costs more than the bytes it saves. A prefill chunk's message is orders of magnitude larger
     * and is squarely on the wire. Below this floor `--tp-wire wht6` therefore serves the message
     * EXACTLY -- which also means a decode-only serve sees no quality change from the flag at all.
     *
     * 128 KiB IS THE KNEE. A decode step's message scales with the number of sequences in the
     * batch, so concurrency decides which side of the floor a step lands on. Compressing pays for
     * every message above the floor and costs on the smallest ones, and 128 KiB is the smallest
     * floor that compresses everything which benefits while leaving single-sequence decode
     * exact. */
    int64_t tp_wire_min_kb = 128;     /* --tp-wire-min-kb */
    int64_t max_tok = 8192;            /* --max-num-batched-tokens */
    int64_t max_seqs = 256;
    int64_t max_ctx = 0;               /* 0 = the model's training context */

    /* THE OPERATOR STATES A BUDGET OR THE ENGINE DERIVES ONE FROM THE CARD, AND EITHER WAY IT IS
     * STATED IN FULL BEFORE ANYTHING IS ALLOCATED.
     *
     * Spec §6 forbids a derived budget because a budget derived from the FREE VRAM at startup is
     * a benchmark that moves when a desktop opens a window. What it does not forbid is arithmetic
     * on numbers that are all known and all printed: the card total is a fixed property, every
     * non-pool allocation this engine makes is computed from the program before it is made, and
     * what the driver already holds is measured rather than assumed. core/mem/vram_budget.cpp does
     * that sum and writes the answer back into the two fields below, so every consumer downstream
     * -- the planner, the KV manager, the pool report -- reads one resolved number and there is no
     * second place for them to disagree.
     *
     * Zero means "derive"; a stated --vram-weights-mib / --vram-kv-mib still wins outright. */
    int64_t vram_weights_mib = 0;
    int64_t vram_kv_mib = 0;
    int64_t gpu_headroom_mib = 96;     /* --gpu-headroom, left unclaimed on every card */
    int64_t prefill_stage_mib = 0;     /* --prefill-stage-mib, buffer size for each prefill staging buffer */
    /* --kv-cache-dtype bf16|fp8. fp8 is an E4M3 cache with per-(sequence, head) descales, and it
     * halves what a token of context costs -- which at a production context is most of the KV
     * pool. It is a real quantisation and the model computes a slightly different function with
     * it, so it is stated rather than inferred, and bf16 is the default.
     *
     * TWO VALUES AND NOT THREE. An "auto" that resolved to bf16 everywhere in this tree would be
     * a third name for the second one, and a flag whose values are not all distinct is a flag
     * whose log line cannot be read back. */
    std::string kv_cache_dtype = "bf16";
    bool    qsa_offload = false;       /* --qsa-offload: QSA Cross-Layer Host-Offload (PCIe prefetch) */
    bool    kvarn = false;             /* --kvarn: KVarN 1:1 INT4 KV compression with FWHT-256 */
    double  expert_cache_ratio = 0.75; /* --expert-vs-cache-ratio, weights vs KV of the remainder */
    /* SET BY THE RESOLVER, NOT BY A FLAG. vram_budget_resolve writes the two budgets above into
     * this config, which erases the only evidence of whether they were stated or derived -- and
     * Pools needs that to know whether unspent VRAM is a mis-stated budget worth warning about or
     * a split it made itself and already printed a receipt for. */
    bool    vram_budget_derived = false;
    int64_t host_pool_mib = 0;
    /* THE FILE TIER FOR WEIGHTS: what fits neither the VRAM slab nor the pinned host pool is read
     * from the container itself, with O_DIRECT, ahead of the ops that read it -- a scheduled unit by
     * the exact prefetch, a routed expert a routed layer at a time. Off unless asked for, because a
     * weight served from disk is a step paced by the drive: a budget that spills there by accident
     * is a server that is correct and several orders of magnitude slow. The tier needs no directory
     * of its own; the container is the file. */
    bool    weights_disk_tier = false;

    /* PLE table backend: "device" (GPU zero-copy from pinned host RAM / SHM, default)
     * or "host" (original CPU AVX sparse O_DIRECT reads from NVMe, 0 B pinned RAM).
     * Flag: --ple-backend device|host (aliases: --ple-storage, --ple-mode).
     * Env: RADIANCE_PLE_BACKEND=device|host. */
    std::string ple_backend = "device";

    std::string placement = "auto";    /* all_vram | layer_offload | expert_tiered | auto */
    /* --embedding-placement and --ngram-placement, as asked (TablePlacement says what each tier
     * is), and what Engine::plan resolved them to before any rank planned: never Auto. */
    TablePlacement embedding_placement = TablePlacement::Auto;
    TablePlacement ngram_placement     = TablePlacement::Auto;
    TablePlacement embedding_resolved  = TablePlacement::Vram;
    TablePlacement ngram_resolved      = TablePlacement::Disk;
    bool    deterministic = false;
    int64_t checkpoint_interval = 2048;   /* linear-state checkpoints (spec §7.3) */
    /* HOW MANY LINEAR-STATE SNAPSHOTS THE SERVER MAY HOLD AT ONCE -- a CAP and not a charge.
     * The slots are address space in the KV pool and are backed as they are handed out, so a
     * run that writes no snapshot pays nothing for them and one holding two pays for two. What
     * it still bounds is the ADDRESS SPACE: a slot is a sizeable fraction of a gigabyte on a
     * linear-attention model, and the range it reserves is range the paged groups do not get,
     * so raising this trades addressable context rather than VRAM (spec 7.3, 19.1). */
    int64_t checkpoint_slots = 8;
    bool    prefix_cache = true;
    bool    prefix_aware_prefetch = true;  /* --prefix-aware-prefetch / --no-prefix-aware-prefetch */
    bool    ced_prefetch = true;           /* --ced-prefetch / --no-ced-prefetch */
    int     ced_lookahead = 4;             /* --ced-lookahead ΔL (default 4 layers ahead) */
    float   ced_threshold = 0.15f;         /* --ced-threshold probability threshold tau */
    int     ced_top_k = 2;                 /* --ced-topk candidates per lookahead */
    bool    ced_reuse_ridgefill = false;   /* --ced-reuse-ridgefill */
    /* geometric-backoff | tip-only | keep-all. §19.1 records the default as a GUESS until there is
     * a workload to measure, and core/mem/prefix.h ships the two ends of the trade beside it for
     * exactly that measurement -- which needs a way to select them. */
    std::string checkpoint_policy = "geometric-backoff";

    /* ---------------- IDLE SESSION TIERS (spec §7.3) ----------------
     *
     * A finished turn leaves its prefix in the cache holding KV blocks, which is what makes the
     * next turn cheap. Those blocks are the most valuable thing in the pool while a session is
     * live and the least valuable thing in it once the session stops. With a host tier, a
     * finished conversation is COPIED to host memory at once and its VRAM given up the moment
     * something wants it -- the expert slab's loan, or a request -- so no threshold decides when
     * it leaves the card: the copy makes leaving free. With a disk tier every host copy is copied
     * on to disk too, so a full host tier gives the oldest slots up for nothing in the same way.
     *
     * Moving is worth it because the alternative is a re-prefill. Recovering a 200K-token turn
     * costs a copy over PCIe; recomputing it costs minutes. The two are not the same order.
     *
     * CAPACITY IS WHAT TURNS THEM ON. Nothing is copied until the host tier has a size: a tier of
     * zero bytes holds nothing, so the feature is off until an operator says how much memory and
     * how much disk it may have. */
    int64_t prefix_cache_host_mib = 0;        /* the host tier's size. 0 is off */
    int64_t prefix_cache_disk_mib = 0;        /* the disk tier's size. 0 is off */
    /* WHERE THE DISK TIER LIVES, and a flag of its own rather than a second meaning for
     * --weights-disk-tier. That one is the WEIGHT planner's permission to place weights on the file
     * tier -- it reads the container, needs pinned staging, and refuses startup without it. The
     * two have nothing in common but the word "disk", and a shared flag would make asking for a KV
     * cache also ask for weight offload, which fails for want of a host pool nobody wanted. */
    std::string prefix_cache_dir;

    /* THE CACHE ALWAYS LENDS AND THERE IS NO FLAG FOR IT. It keeps the blocks it is using plus a
     * buffer it derives, and lends the rest to the expert plane at the addresses it already owns;
     * see core/mem/kv.h. The buffer's right size is a property of the workload (a prefill chunk,
     * or the largest recent shortfall) that nobody can state in advance, which is why it is
     * derived rather than a number to set.
     *
     * The loan allocates nothing, so it does not touch --gpu-headroom-mib: that reserve stays what
     * a card keeps for the things this engine does not allocate -- chief among them a kernel's
     * code object, which HIP loads on the kernel's FIRST LAUNCH and not at startup, and which
     * faults inside the HIP runtime rather than returning an error when there is no room. */

    /* The speculative window, and the whole of the draft policy: the depth is constant and this
     * is it. See core/sched/draft.h. */
    /* -1 is `auto`: engine_bringup asks rad_arch_probe for the depth THIS container's drafter was
     * trained at, and a container with no draft head resolves to 0 with a line saying so.
     *
     * IT DEFAULTS TO AUTO BECAUSE MERGING A DRAFTER IS THE STATEMENT OF INTENT. rad-kbench and
     * rad-tune read the container's own operating point; a serving engine that defaulted to off
     * would leave a deployment that forgot the flag paying for a draft head at convert time,
     * paying for it again in VRAM, and then running at one token a step -- a small fraction of
     * the throughput the same container gives at its declared depth. */
    int     n_spec = -1;                /* --num-speculative-tokens; -1 = auto, 0 = off */
    bool    adaptive_spec = false;     /* --adaptive-spec: enable dynamic speculative depth controller */

    /* RidgeFill cold prefill projector acceleration */
    std::string ridgefill;              /* --ridgefill off|speed|quality|plumb */
    std::string ridgefill_projector;    /* --ridgefill-projector DIR */

    /* ---------------- KL DIVERGENCE AGAINST A REFERENCE (core/kld.h) ----------------
     *
     * A MODE, LIKE --debug-graph: the engine starts as it would to serve, then runs a fixed
     * corpus through prefill instead of listening, with a full-vocabulary logits row at every
     * scored position. `--kld-record DIR` makes THIS model the yardstick and writes its
     * log-probabilities to DIR once; `--kld-ref DIR` scores this model against what is there, and
     * only the candidate's own logits are computed. Exactly one of the two. */
    std::string kld_record;             /* --kld-record DIR */
    std::string kld_ref;                /* --kld-ref DIR */
    std::string kld_corpus;             /* --kld-corpus FILE, read by --kld-record only */
    std::string kld_out;                /* --kld-out FILE: the report as JSON, rows beside it */
    bool kld() const { return !kld_record.empty() || !kld_ref.empty(); }

    /* --mm-max-patches: the most patches one encoder pass carries, which bounds the largest image
     * (one image is one pass) and sizes the encoder's activations. -1 is `auto`: the default below
     * when the container carries a vision tower and 0 when it does not. 0 serves text only and
     * keeps a tower the container carries off the card. */
    int64_t mm_max_patches = -1;

    std::string host = "0.0.0.0";
    /* The chat endpoint's default `reasoning_effort`, empty for the template's own default.
     * Passed through verbatim: the legal set belongs to the template, not to us. */
    std::string reasoning_effort;
    /* --thinking-token-budget: the chat endpoint's default `thinking_token_budget`, -1 for none. */
    int64_t     thinking_token_budget = -1;
    int         port = 8000;
    /* --api-key: the bearer key every request but the probes and the dashboard page must carry.
     * Empty serves without one. */
    std::string api_key;
    /* --served-model-name: the id /v1/models lists, a response names when its request named none,
     * and the metrics carry. Empty takes the container's own name. A request may still name any
     * model: one model is loaded and every request is served by it. */
    std::string served_model_name;

    /* WHAT A REQUEST THAT SENDS NO SAMPLER SETTINGS IS SERVED WITH.
     *
     * A checkpoint states the sampling it was tuned for, and serving it at anything else is a
     * different model. Left alone these are resolved from the container and from the
     * `generation_config.json` beside it (resolve_sampling_defaults in engine.cpp); set, they win over
     * both, which is what makes a deployment able to pin a sampler a container gets wrong.
     *
     * -1 IS "NOT SET HERE" RATHER THAN A VALUE, because every one of these has a legal setting
     * that would otherwise be indistinguishable from silence: 0 is a real temperature (greedy),
     * 0 is a real top_k (off) and 0 is a real min_p. A flag that cannot say "unset" cannot have
     * anything resolved underneath it. */
    float   sample_temp  = -1.0f;
    int     sample_top_k = -1;
    float   sample_top_p = -1.0f;
    float   sample_min_p = -1.0f;
    /* THE REST OF THE SAMPLER, each one the default for a request that does not name the field.
     * Optional rather than negative-for-unset: a presence penalty of -1 and a DRY multiplier of
     * -1 are settings a request can send, so no value of the field itself can mean "not set".
     * Unset, a request that sends nothing gets SamplingParams' own value, exactly as before the
     * flags existed. */
    std::optional<float> sample_typical_p, sample_rep_penalty, sample_pres_penalty,
                         sample_freq_penalty, sample_dry_multiplier, sample_dry_base,
                         sample_xtc_probability, sample_xtc_threshold;
    std::optional<int>   sample_penalty_last_n, sample_dry_allowed_length,
                         sample_dry_penalty_last_n;
    std::optional<std::vector<std::string>> sample_dry_seq_breakers;

    /* --default-max-tokens: what a request that names no max_tokens may generate. -1 is unset,
     * which is the server's default, `auto`; 0 is `auto` stated (whatever the context leaves after
     * the prompt). */
    int64_t default_max_tokens = -1;
    /* --max-tokens-cap: the most any request may generate. A larger max_tokens is clamped to it,
     * the way the context clamps one, and the reply ends with finish_reason "length". 0 is none. */
    int64_t max_tokens_cap = 0;
    /* --max-n: the largest `n` a request may ask for. 0 is the larger of 8 and --max-num-seqs. */
    int64_t max_n = 0;
    /* --max-queued-requests: requests admitted and unfinished at once, beyond which the answer is
     * 429. 0 is 8 x --max-num-seqs. */
    int64_t max_queued_requests = 0;
    /* --max-stop-strings / --max-stop-bytes: how many stop strings a request may send, and how
     * long each may be. Every stop string is searched for in every stretch of generated text, so
     * these bound what one request can make that scan cost. */
    int64_t max_stop_strings = 64;
    int64_t max_stop_bytes = 4096;

    /* --chat-template-kwargs: variables handed to the chat template for every chat request, as
     * key -> JSON text (a string arrives still quoted, as the template context wants it). A key
     * the request's own chat_template_kwargs names is the request's. */
    std::map<std::string, std::string> chat_template_kwargs;
    /* --reasoning-format auto|none: `auto` returns a reasoning block as reasoning_content, `none`
     * leaves it inline in content for a client that reads it there. */
    std::string reasoning_format = "auto";

    /* ---------------- THE HTTP TRANSPORT ----------------
     * The values the transport has always used, settable. See TransportOptions (server/http.h). */
    int     http_threads = 0;                /* 0: one per core, at least 4 */
    int     http_read_timeout_s = 30;
    int     http_write_timeout_s = 600;
    int     http_keep_alive_timeout_s = 5;
    int64_t http_max_body_mib = 512;
    int     retry_after_s = 1;
    bool    cors = true;

    /* ---------------- MEDIA PREPROCESSING ----------------
     * Each one replaces what the container's preprocessor configuration states (mm::VisionConfig).
     * Unset, the container's value is served. */
    std::optional<int64_t> image_min_pixels, image_max_pixels, video_min_pixels, video_max_pixels;
    std::optional<double>  video_fps;
    std::optional<int32_t> video_min_frames, video_max_frames, video_max_frame_tokens;
    std::optional<int64_t> max_source_pixels;
    bool media_flags_set() const {
        return image_min_pixels || image_max_pixels || video_min_pixels || video_max_pixels ||
               video_fps || video_min_frames || video_max_frames || video_max_frame_tokens ||
               max_source_pixels;
    }

    /* An operator-supplied generation_config.json, overriding the search described above. */
    std::string generation_config;
    /* --override-chat-template: a Jinja file whose text replaces the chat template the container
     * (or the checkpoint directory) carries. Everything that reads the template reads this one:
     * the chat endpoint's rendering, and the reply format derived from it -- reasoning markers and
     * tool calls. Empty serves the container's. */
    std::string override_chat_template;

    bool    debug_graph = false;
    bool    debug_selection = false;
    bool    debug_placement = false;
    bool    profile_ops = false;
    /* --live: the live view (spec §16). Off by default like everything else in that section. */
    bool    live_view = false;

    /* libref is the ORACLE and the last resort, not a deployment. An op silently falling
     * back to it is a model running orders of magnitude below the machine with nothing in the
     * output to say so, so it is refused unless this says otherwise -- deliberately, and by a
     * flag with `debug` in its name (startup.h, rad_gate_reference_kernels). */
    bool    accept_reference_kernels = false;
};

}  /* namespace rad */

/* Declared here because config.cpp is owned by the engine, and every tool parses the same flags
 * so that `rad-info --model X` and `radiance --model X` cannot disagree about what a flag means. */
namespace rad {
int  config_parse(int argc, char** argv, Config* c);
void config_usage(const char* argv0);
}
