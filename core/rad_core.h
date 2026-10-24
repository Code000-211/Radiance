/* rad_core.h -- the data model shared between core components. Each component owns a header for
 * its own methods; this file owns the structs that cross a component boundary (weights, bands,
 * requests, config), so components cannot disagree about them.
 *
 * Adding a field is fine. Changing one affects other readers; flag it in review.
 */
#pragma once
#include "rad_internal.h"
#include "lora/config.h"

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

/* One kernel row plus its origin. The hierarchy is absolute: the first plugin with any matching
 * kernel supplies it, and a lower-priority plugin never outbids a higher one, however specialised
 * (spec §2.1). `plugin_order` is the hierarchy position. */
struct KernelRow {
    const RadKernelInfo* info = nullptr;
    std::string          plugin;
    int                  plugin_order = 0;   /* 0 is highest priority */
    int                  index = 0;          /* declaration order within the plugin */
    /* rad_kernel_concurrent: launches touch only their operands and scratch, so one may run beside
     * another that shares none of those bytes. False when the plugin does not say. */
    bool                 concurrent = false;
    /* rad_kernel_collective: launches exchange data with other ranks' cards; `releases_link` means
     * they end the engine's hold on the link themselves, through `link_gate`. False when unsaid. */
    bool                 collective = false;
    bool                 releases_link = false;
    /* The row's plugin declared itself a reference implementation (RadPluginReferenceFn). */
    bool                 reference = false;
};

/* A kernel bound to one geometry, one domain and one variant, with init() already run. */
struct Resolved {
    const KernelRow* row = nullptr;
    void*            instance = nullptr;
    /* The frozen geometry, including the tuned axes, so RadArgs.p carries them and the kernel
     * reads `sk` with rad_args_geti() like `M`. There is no variant index. */
    Geometry         geom;
    int64_t          scratch_bytes = 0;
    /* The geometry without the tuned axes, in tunecache.h's canonical spelling. It is the cache
     * key, so it is captured before the choices are folded into `geom`. */
    std::string      geom_key;
    /* The chosen axes as sorted `key=value,...` (the cache's `choice` column), and their source:
     * "axis defaults", "tuned <timestamp>" or "RADIANCE_TUNE" (spec §15). Both empty for a kernel
     * with nothing to tune. */
    std::string      choice;
    std::string      tune_source;
    /* The row's DIV on the op's ranged parameter (the least common multiple of several), 0 when
     * it has none. Selection checked it at the band's upper bound only, so the issue path refuses
     * a value in the band it does not divide (Registry::build_bands). */
    int64_t          ranged_div = 0;

    explicit operator bool() const { return row != nullptr; }
};

/* One row of a bucket table: a band of the ranged parameter, resolved per domain. The band
 * boundaries are the union of the LE, GE and EQ values candidate kernels place on the ranged
 * parameter; there is no chosen policy (spec §2.2). */
struct Band {
    int64_t  hi = 0;                          /* inclusive upper bound */
    Resolved dom[RAD_N_DOMAINS];              /* [RAD_DOMAIN_DEVICE], [RAD_DOMAIN_HOST] */
    std::string miss[RAD_N_DOMAINS];          /* why nothing resolved, for the report */

    /* Display form: "M <= 64", "M in (64, 512]", "any shape". Stored here because the exclusive
     * lower bound is the previous band's hi, which only the table's builder has. */
    std::string span;
};

/* ================================================================== the declared program */

/* Weight tiers. HostPinned and HostPageable copy the weight into an arena at load, and SSD preads
 * it a slot at a time. Mapped is not host memory but the container's own mmap
 * (core/format/radfile.cpp): never copied, in no pool, and the OS page cache decides residency.
 *
 * Mapped is for tables of tens of gigabytes read a few rows a token, such as an n-gram embedding.
 * A copying tier would stage the whole table for a few kilobytes, pin it on a host with RAM of
 * the same order, or keep a second copy beside the mapping; a sparse Zipfian gather suits a page
 * cache.
 *
 * Only a host kernel can read the mapping (pageable, no device address), so the planner assigns
 * it where no device kernel exists for the op, and where --ngram-placement keeps an n-gram table
 * off the card (embed_lookup_q, libavx/avx_ngram.cpp). Under `ram` the load phase publishes a
 * copy instead: locked memory for the host gather, or pinned memory a device kernel reads over
 * the link (the zero-copy site). */
enum class Tier { VRAM = 0, HostPinned = 1, HostPageable = 2, SSD = 3, Mapped = 4 };

/* Where a lookup table lives: the token embedding (--embedding-placement) or an n-gram embedding
 * (--ngram-placement). Only a row lookup reads them, so a step touches M rows, never the whole
 * table, which makes every tier below the card usable.
 *
 *   Vram   resident on the card, like any other weight.
 *   Ram    host memory the card or a host kernel reads in place: the embedding pinned beside the
 *          host pool and read over the link; an n-gram table as a copy made at load.
 *   Disk   read from the container on demand; n-gram table only.
 *   Auto   the fastest of the three with room once every other weight is placed; n-gram table
 *          only. Engine::plan resolves it into Config so every rank plans from one choice. The
 *          embedding takes Ram or Vram. */
enum class TablePlacement { Auto = 0, Vram = 1, Ram = 2, Disk = 3 };
const char* table_placement_name(TablePlacement p);
/* --draft-window: how many drafts a step verifies per sequence (core/sched/draft.h).
 *   Fixed       every sequence drafts --num-speculative-tokens.
 *   Adaptive    that is the ceiling; head depth per step and each sequence's stop follow measured
 *               acceptance and step cost. Serial drafters only.
 *   Confidence  that is the ceiling; a sequence's next window follows the head's own q(x) on the
 *               window the step in flight verifies (--draft-confidence-rule), at one decoding
 *               row. A serial head under RADIANCE_MTP_COUPLED; a batched cycle and a
 *               publication without q draft --draft-confidence-fallback.
 * The default is kDraftWindowDefault; the help text and startup line read it from there.
 *
 * Fixed is the default because adaptive loses on a busy server. Qwen3.8-Flash-Next, TP2 on two
 * R9700, distinct prompts, throughput against fixed depth 3:
 *
 *      concurrent sequences        1        2        4        8
 *      adaptive, ceiling 3       +0.4%    -0.1%    +1.7%    -0.3%
 *      adaptive, ceiling 5       +7.1%    -2.3%    +7.4%   -11.9%
 *
 * At eight sequences depth 5 costs 1.18x the step time (43.0 vs 36.3 ms) for 1.04x the tokens
 * (23.3 vs 22.3). The cost model's prior puts depth 5 at 1.10x depth 3, and a depth not yet timed
 * at a load is priced on the prior. */
enum class DraftWindow { Fixed = 0, Adaptive = 1, Confidence = 2 };
constexpr DraftWindow kDraftWindowDefault = DraftWindow::Fixed;
inline const char* draft_window_name(DraftWindow w) {
    return w == DraftWindow::Adaptive ? "adaptive" : w == DraftWindow::Confidence ? "confidence"
                                                                                : "fixed";
}

/* --draft-policy: what decides each cycle's draft window.
 *   Legacy  the fixed depth above, or --draft-window adaptive's per-sequence stop.
 *   Cost    the adaptive-MTP cost policy (core/sched/draft_policy.h): B's predictor supplies
 *           each request's per-position acceptance, A's cost model prices the batch shape, and
 *           the plan it chooses is what every proposal site reads for the cycle. Without a
 *           measured profile the model has no anchor, so the policy issues its named fallback at
 *           the configured depth rather than inventing a price.
 * The default is kDraftPolicyDefault, read by the help text and the startup line.
 * `Kind`, not `DraftPolicy`: that name is the seam's choose hook (core/sched/draft_types.h). */
enum class DraftPolicyKind { Legacy = 0, Cost = 1 };
constexpr DraftPolicyKind kDraftPolicyDefault = DraftPolicyKind::Legacy;
inline const char* draft_policy_name(DraftPolicyKind p) {
    return p == DraftPolicyKind::Cost ? "cost" : "legacy";
}
/* The most K values --draft-depth-tiers carries; sizes Config::draft_tiers and the planner's
 * ladder (RAD_DRAFT_MAX_TIERS, core/sched/draft_policy.h). */
constexpr int kDraftMaxTiers = 8;
enum class Site { Device = 0, DeviceStaged = 1, DeviceZeroCopy = 2, Host = 3 };

/* Size of every Tier-indexed array (report, live view, planner totals). Use it rather than a
 * literal, so a new tier cannot silently write out of bounds. */
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
    /* rad_weight_shard_span_parts: a span per stacked part and each part's rows in the
     * container; n_part_spans 0 is shard_lo/hi (one span for every part) or the even share. */
    int            n_part_spans = 0;
    int64_t        part_lo[RAD_MAX_ROW_PARTS] = {0}, part_hi[RAD_MAX_ROW_PARTS] = {0};
    int64_t        part_total[RAD_MAX_ROW_PARTS] = {0};

    /* The view's source (RadWeightDecl::source and ::planes, spec §4.3). `enc` is the source's
     * encoding as the model holds it, or plain in the declared dtype when nothing is behind the
     * builder; `sel` lists the planes this declaration takes, in order. `sel_rows`/`sel_cols` are
     * each plane's extents for this rank, which is what a layout hook is shown. */
    std::string    source;
    RadEncoding    enc{};
    bool           enc_known = false;
    int32_t        n_sel = 0;
    int32_t        sel[RAD_ENC_MAX_PLANES] = {0};
    int64_t        sel_rows[RAD_ENC_MAX_PLANES] = {0};
    int64_t        sel_cols[RAD_ENC_MAX_PLANES] = {0};

    /* Filled by the layout pass: what the resolved kernel stores. `identity` is the selected
     * plane's bytes unchanged, the only form the file and mapped tiers can serve since they read
     * the container in place; `widen` is identity plus an exact widening (a bf16 norm read as
     * f32) done by the loader. Otherwise the kernel at (lay_op, lay_band, lay_dom) relayouts it at
     * load, as operand `lay_operand`. */
    std::string    layout_tag;
    bool           identity = true;
    bool           widen = false;
    int32_t        lay_op = -1, lay_band = -1, lay_dom = -1, lay_operand = -1;
    uint32_t       stored_dtype = RAD_DT_INVALID;
    uint32_t       stored_rank = 0;
    int64_t        stored_shape[RAD_MAX_RANK] = {0};   /* the shape the kernel receives, which a
                                       * layout hook may change (WMMA fragment order is not the
                                       * declared shape) */
    int64_t        stored_bytes = 0;
    int64_t        align = RAD_ALIGN_UNIT;

    /* Filled by the planner (spec §5.3). */
    Tier           tier = Tier::VRAM;
    Site           site = Site::Device;
    int32_t        slab_slot = -1;      /* -1 = not slab-resident */

    /* Filled at load. Under TP this is this rank's shard, not the whole weight. */
    void*          ptr = nullptr;
    void*          host_ptr = nullptr;
    uint64_t       file_offset = 0;

    /* First and last op index that read the weight. Prefetch runs off these, so layer offload is
     * scheduling rather than prediction (spec §5). */
    int32_t        first_use_op = -1;
    int32_t        last_use_op  = -1;
};

/* This rank's selected planes as a layout hook sees them: dense rank-2 geometry, `data` null
 * unless the caller fills it. Shared by the builder at declare and the runtime before moving an
 * op to another domain, so both ask the hook the same question. */
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

/* Whether the bytes can be read where they lie in the container (the file tier's O_DIRECT stage,
 * the mapped tier's pointer): only when they are the selected plane unchanged and this rank holds
 * all of it. A relayout is not in the file, and a column shard is not one contiguous range. */
inline bool weight_file_direct(const WeightInfo& w, int world_size) {
    return w.identity && !w.widen && (w.decl.shard == RAD_SHARD_NONE || world_size <= 1);
}

struct BufferInfo {
    std::string name;
    RadBufDecl  decl{};
    int64_t     bytes = 0;

    /* From the buffer plan: liveness over the declared op list, and the assigned arena offset.
     * Two transients whose lives do not overlap share storage (spec §3.1). */
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
    /* The schema operand position each declared weight belongs to, parallel to `weights`. Not the
     * index: RAD_OPD_WTAB lets one operand consume a run of declared weights (a MoE layer's
     * experts), so the mapping is many-to-one. Recorded once at declare so call sites cannot
     * disagree. */
    std::vector<int32_t>  weight_opd;
    const RadOpSchema*    schema = nullptr;
    int32_t               index = 0;    /* position in the declared order */
    bool                  issued = false;   /* did the run phase ever issue it */
    /* Runtime LoRA (core/lora/). `anchor`: the op a core-appended op is issued inside of; its
     * buffers are that op's, so their liveness is judged at the anchor's position rather than this
     * op's own (past every plugin op). -1 for a plugin op. `lora_site`: a targeted gemm's row in
     * Program::lora_sites, -1 otherwise. */
    int32_t               anchor = -1;
    int32_t               lora_site = -1;
    /* Declined at declare because runtime LoRA targets its weight (a fused GEMM no delta can
     * follow): recorded unresolved and never handed to the plugin, so it is never issued and the
     * reference-kernel gate does not count it. */
    bool                  lora_declined = false;
};

struct KVGroupInfo {
    std::string     name;
    RadKVGroupDecl  decl{};
    int64_t         block_size = 0;     /* taken off the resolved attention kernel, not chosen */
    int64_t         bytes_per_block = 0;
    int64_t         bytes_per_state = 0;   /* linear/conv: per sequence */
    std::vector<int> layers;
    bool            zero = false;       /* rad_kv_group_zero: declared for position, nothing carved */
    /* Bound only at meta.n_layers and above: the drafter's own cache, which no trunk layer reads.
     * Set at declare (Builder::finish_kv_groups). */
    bool            drafter = false;
};

/* A RAD_KV_CONV group's window in entries: conv_width - 1 of history plus one per speculative
 * position. It is the last extent of the state tensor, and kernels derive the layout from it: a
 * slot is `entries` consecutive entries per channel, and libr4d's conv update derives the verify
 * query length as entries - (conv_width - 2). The allocation, the KV manager's slot view (clear on
 * hand-out, checkpoint save and restore) and conv_state_index's cursor must all use this number;
 * a larger manager-side slot overlaps the next sequence's window. */
inline int64_t conv_window_entries(const RadKVGroupDecl& d, int64_t n_spec) {
    return (d.conv_width > 0 ? d.conv_width - 1 : 0) + (n_spec > 0 ? n_spec : 0);
}

/* One targeted GEMM of a runtime-LoRA program (core/lora/), as the issue path needs it: the two ops
 * the core appended for it, where the gemm's activation and output are in its operand list, and
 * the buffers the appended ops read. Built once at declare. */
struct LoraIssueSite {
    uint32_t gemm = 0, shrink = 0, expand = 0;     /* op handles */
    int8_t   opd_a = -1, opd_a_scale = -1, opd_y = -1;   /* positions in the GEMM's schema */
    uint32_t a_dtype = 0, y_dtype = 0;             /* what the appended ops were declared for */
    int32_t  pool = -1;                            /* index into Program::lora_pools */
    int32_t  t = -1;                               /* index into Program::lora_t */
    int64_t  R = 0, K = 0, N = 0;
    /* A generator's narrower pairs: the same pools read to rank R_v[i] < R, the pass taking the
     * narrowest that holds its slot's ranks (Program::lora_rused). */
    uint32_t shrink_v[2] = { 0, 0 }, expand_v[2] = { 0, 0 };
    int64_t  R_v[2] = { 0, 0 };
    int8_t   n_v = 0;
};
/* One targeted weight's adapter slots: A [S, R, K], B [S, N, R] bf16 and the per-column scale
 * [S, N] f32, persistent buffers, so the arena plan (and with it the KV budget) holds them. */
struct LoraPool {
    std::string weight;
    rad_buf     a = 0, b = 0, scale = 0;
    int64_t     R = 0, K = 0, N = 0, S = 0;
};
/* The f32 intermediate t = A.x of one rank capacity R, one per lane (lane 1 runs beside lane 0). */
struct LoraScratch {
    int64_t R = 0;
    rad_buf lane[2] = { 0, 0 };
};
/* A generator's adapter key names (rad_lora_root, rad_lora_alias in rad_builder.h). */
struct LoraRoot {
    std::string              native;      /* checkpoint prefix of the network's tensors */
    std::vector<std::string> spellings;   /* adapter-side prefixes; "" is a bare key */
};
struct LoraAlias {
    std::string root;                     /* LoraRoot::native it is relative to */
    std::string from, to;                 /* module paths, '#' a decimal index */
    int64_t     from_lo = 0, from_hi = 0, to_lo = 0;
};

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

    /* The step's logits buffer (rad_declare_logits). Zero when the plugin declared none, as an
     * encoder-only or embedding-only plugin does; the engine refuses to serve such a program at
     * startup. */
    rad_buf logits_buf = 0;
    /* The step's pooled vectors (rad_declare_pooled): set only by an embedding program, which has
     * no logits buffer. */
    rad_buf pooled_buf = 0;

    /* Runtime LoRA (core/lora/), empty unless the engine was started with adapters enabled. */
    std::vector<LoraIssueSite> lora_sites;
    std::vector<LoraPool>      lora_pools;
    std::vector<LoraScratch>   lora_t;
    /* A generator's adapter key names: declared by the plugin whether or not adapters are on. A
     * generator that declares no root carries no adapters. */
    std::vector<LoraRoot>      lora_roots;
    std::vector<LoraAlias>     lora_aliases;
    /* A generator's pass runs with one adapter in every row: `lora_rows` is [S * lora_rows_n]
     * i32, run s all s, so a pass's slot operand is run s of it. 0 on a token-step program. */
    rad_buf                    lora_rows = 0;
    int64_t                    lora_rows_n = 0;
    /* A generator's slots pack each combination's ranks from j = 0: the ranks pool p takes in
     * slot s at [p * S + s], 0 for a pool the combination does not reach (no adapter op runs). */
    std::vector<int32_t>       lora_rused;

    /* The declared draft head (rad_declare_drafter). `kind == RAD_DRAFT_NONE` when none, the
     * normal case: the engine then speculates with the scheduler's prompt-lookup drafter or not at
     * all. `drafter_name` is a copy because RadDrafterDecl::name points into plugin storage. */
    RadDrafterDecl drafter{};
    std::string    drafter_name;
    /* A serial head with no cache of its own (rad_drafter_shared_kv): it reads the trunk's K and V
     * and writes none, so the driver owes it no history and runs every round at the committed
     * position. `drafter_pos_offset` (0 or 1) is where those rows sit for the rotary. False and 0
     * for drafters that did not ask. */
    bool           drafter_shared_kv = false;
    int32_t        drafter_pos_offset = 0;

    /* The declared media encoder (rad_declare_encoder). `modalities == 0` when none: every
     * text-only program, and every deployment with RadBuildCtx::max_enc_patches 0.
     * `encoder_name` is a copy, as `drafter_name` is. */
    RadEncoderDecl encoder{};
    std::string    encoder_name;
    /* The audio encoder, the second a program may declare (RAD_MM_AUDIO). */
    RadEncoderDecl audio_encoder{};
    std::string    audio_encoder_name;
    /* The declared image generator (rad_declare_generator); `layout` 0 when none. Its strings and
     * fixed schedule point at the copies in `gen_str`, not plugin storage. A generator program has
     * no logits and no pooled vectors. */
    RadGeneratorDecl generator{};
    struct GenStrings {
        std::string name, tmpl_text, tmpl_edit, tmpl_image, tmpl_image_sep, drop_prefix,
                    tmpl_transparent, default_negative;
        bool        has_default_negative = false;
        std::vector<double> sigmas;
    } gen_str;
    bool is_generator() const { return generator.layout != 0; }
    /* The declared audio generator (rad_declare_audio_generator); `max_lanes` 0 when none. Its
     * strings point at `agen_str`, as the image generator's do at `gen_str`. */
    RadAudioGenDecl audio_gen{};
    struct AudioGenStrings {
        std::string name, tmpl, tmpl_plain, neg_tmpl;   /* "" for an absent template */
    } agen_str;
    bool is_audio_generator() const { return audio_gen.max_lanes != 0; }
    /* The declared video generator (rad_declare_video_generator); `max_lanes` 0 when none. */
    RadVideoGenDecl video_gen{};
    struct VideoGenStrings {
        std::string name, tmpl;
        std::vector<double> rungs;   /* RadVideoGenDecl::rungs; the declaration's pointer is null */
        std::vector<double> refine_rungs;
        std::string default_negative;
        /* RadVideoGenDecl::decoders, one name each, and their largest frames (0: max_area). */
        std::vector<std::string> decoders;
        std::vector<int64_t>     decoder_max_area;
    } vgen_str;
    bool is_video_generator() const { return video_gen.max_lanes != 0; }
    /* Driven by passes, not token steps: any generator. */
    bool is_pass_driven() const {
        return is_generator() || is_audio_generator() || is_video_generator();
    }

    /* Two arenas, because a host-site activation cannot index device memory. The buffer plan
     * colours each domain separately (transients share storage only within a domain), so each is
     * a high-water mark in its own offset space and BufferInfo::arena_offset is relative to the
     * arena of the buffer's declared domain. Both are sized for max_tok, the worst step (spec
     * §3.1). */
    int64_t arena_bytes = 0;         /* RAD_DOMAIN_DEVICE */
    /* Start of the device arena's transients. PERSIST and DERIVED buffers sit below, at the same
     * offset at every step size; above is what a smaller step packs smaller
     * (core/build/rad_bufplan.cpp, plan_level). */
    int64_t arena_fixed_bytes = 0;
    /* The prefill staging region, above every planned buffer: where a full-size step copies a
     * routed layer's non-resident experts ahead of the layer (core/place/stager.h). Sized by the
     * VRAM budget and appended by the engine; part of the arena top that smaller levels lend to
     * the expert slab. -1 / 0 when absent. */
    int64_t arena_stage_off = -1;
    int64_t arena_stage_bytes = 0;
    int64_t host_arena_bytes = 0;    /* RAD_DOMAIN_HOST */
    int64_t scratch_bytes = 0;       /* the largest single kernel scratch requirement */
    /* Scratch regions of that size in the arena, one per lane. Kernels on one stream share a
     * region safely through in-order launch; a second lane is a second stream and needs its own.
     * Two when the architecture declared a buffer rad_buf_concurrent. */
    int32_t scratch_regions = 1;

    /* Everything that resolved to nothing, reported whole rather than stopping at the first
     * failure (spec §3.1). Structured because the domains are reported differently: a device miss
     * is a hole a request can hit and gets its own line; a host miss only removes a placement
     * option and gets one line per op, not per band. Otherwise a deep model at tp2 prints
     * thousands of expected lines that hide a real device hole. */
    struct Miss {
        std::string op;
        std::string geom;         /* the band's display geometry */
        std::string span;         /* the band, as Band::span */
        std::string why;
        int         domain = 0;   /* RAD_DOMAIN_DEVICE | RAD_DOMAIN_HOST */
        int32_t     index = 0;    /* the instance that missed (OpInfo::index): bands are its own */
        std::string line() const; /* "op  geometry  [domain] -> why", the full-detail rendering */
    };
    std::vector<Miss> misses;
    /* The device misses a request can hit, in `misses` order: device bands no kernel serves and no
     * host kernel stands in for. On the host backend a host kernel for the band stands in, since
     * every op without a device row runs on the host (Ctx::bind_ops). On a card it stands in only
     * for an op with weights, because the planner moves an op to the host by placing its weights
     * there; attention over a VRAM cache is the case that matters. Shared by the engine's startup
     * refusal and the tools that count holes. */
    std::vector<const Miss*> device_holes(bool host_backend) const;

    /* One per kernel instantiation that rad-tune would benchmark (spec §15). `info` is the
     * kernel's row, so the tuner can read the RadTunable axes, their legal values and the validity
     * predicate; plugins outlive the Program, so the pointer stays valid. `geom` is what a launch
     * receives and carries the selector's axis defaults, which the tuner overrides per candidate
     * with set_i. `geom_key` is the instantiation without them (tunecache.h's spelling) and is the
     * cache key, because the selector looks up before it knows the choice. */
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

/* A chat reply's reasoning block by token id, so the engine can count and close it without
 * detokenising; the reply parser reads the same block as text (core/text/chatparse.h). It exists
 * only when both markers are single tokens (`<think>`/`</think>` are on every served model).
 * `close < 0` means no countable block: the reasoning is still split out as text, but is not
 * counted and cannot carry a budget.
 *
 * The block starts the reply: open from the first token when the generation prompt opened it
 * (`starts_open`), or opened by `open` as the first token; the first `close` ends it. A reply that
 * starts any other way has no block. */
struct ReasoningSpan {
    int32_t open = -1;            /* the token a reply opens its own block with; -1 for none */
    int32_t close = -1;           /* the token that ends the block; -1: no countable block */
    bool    starts_open = false;  /* the generation prompt left the reply inside the block */
    bool known() const { return close >= 0; }
};

/* A reply's position relative to its ReasoningSpan, advanced one emitted token at a time. Shared
 * by the server's count and the sampler's budget so both agree where the block ends. */
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

/* Most alternatives one token's logprobs may carry: OpenAI's top_logprobs bound; larger requests
 * are refused. The sampler's `sample_logprobs` is one wider (Sampler::kLpK) for the
 * split-vocabulary gather; the extra pair is never reported. */
static constexpr int RAD_MAX_LOGPROBS = 20;

/* One emitted token's log-probability under the raw distribution (logits before any penalty,
 * temperature, mask or cut), and its `n_top` most likely alternatives in descending order, ties to
 * the lower id. This is OpenAI's `logprobs` and llama-server's `n_probs`. `token` -1 means the
 * logprobs could not be computed (no finite logit in the row); the scheduler fails the request. */
struct TokenLogprobs {
    int32_t token = -1;
    float   logprob = 0.0f;
    int32_t n_top = 0;
    int32_t top_id[RAD_MAX_LOGPROBS] = {};
    float   top_lp[RAD_MAX_LOGPROBS] = {};
};

struct SamplingParams {
    float  temp = 1.0f;
    int    top_k = 0;                /* 0 = the 1024-candidate width */
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
    /* Temperature after the cuts: top-p, min-p, typical and XTC read the distribution at
     * temperature 1 and the pick draws at `temp`. This is llama.cpp's order; the default is
     * vLLM's (temperature first). A request selects it with `samplers`. */
    bool   temp_last = false;
    /* Logprobs: -1 is none; 0..RAD_MAX_LOGPROBS returns every emitted token's logprob and that
     * many alternatives. Works under speculation: each verified position is its own row, and an
     * accepted draft is the token the target drew there. */
    int    n_probs = -1;
    uint64_t seed = 0;
    bool   greedy() const { return top_k == 1 || temp <= 0.0f; }
    std::string grammar;             /* GBNF, or "" */
    std::vector<std::string> dry_seq_breakers;

    /* The reasoning block and its body's token limit (-1 = none). At the limit the sampler masks
     * every position still inside the block to `reasoning.close`, so the answer follows (vLLM's
     * `thinking_token_budget`). A budget needs a known span; the server refuses one otherwise. */
    ReasoningSpan reasoning;
    int32_t       thinking_budget = -1;

    /* A lazy grammar constrains nothing until a trigger fires. That is the only way to enforce a
     * tool-calling template: free prose until the model starts a call, grammar-shaped after. Treat
     * a lazy grammar as non-lazy and a request with `tools` can only emit a tool call. Trigger
     * types are common_grammar_trigger_type's, as ints because this header does not include the
     * lift: 0 token, 1 word, 2 pattern, 3 pattern-full. The sampler resolves words and tokens to
     * vocabulary ids. */
    struct GrammarTrigger { int type = 0; std::string value; };
    bool                        grammar_lazy = false;
    std::vector<GrammarTrigger> grammar_triggers;

    /* Text the grammar has already seen: a chat template's grammar describes the whole assistant
     * turn, generation prompt included, but those bytes are in the prompt rather than in what is
     * about to be sampled. See GrammarSpec::prefix in core/sample/grammar.h. */
    std::string                 grammar_prefix;

    /* `grammar` compiled by the admitting thread (core/sample/gbnf.h, gbnf_compile). Building the
     * mask plan takes tens to hundreds of milliseconds, and the sampler starts requests on the
     * scheduler thread where every other request's step waits; held here, starting is a lookup.
     * Null for a request not admitted through the server; the sampler then compiles. */
    std::shared_ptr<const GbnfProgram> grammar_program;

    /* The control vector (core/cvec.h): whether this request's steps project the vector found for
     * this deployment out of the residual stream. The server sets it from the request's
     * `control_vector` field (`experimental_speed_projection` is Strata's spelling of the same
     * switch), or from the deployment's runtime default when the request names none; it needs no
     * reload and leaves the graph untouched. A deployment with no vector refuses `true` at
     * admission rather than serving the stock model. */
    bool   use_control_vector = false;
};

/* Raw logits at chosen vocabulary ids at the prompt's last position, for a caller that answers
 * from a distribution over a few tokens (/v1/systemone). The caller fills `tokens` before
 * submission; the engine writes `logits` in the same order, from the row before the sampler
 * touches it, before the sink finishes. Read only after a successful finish. Shared so the result
 * outlives a caller that left mid-step. */
struct TokenScores {
    std::vector<int32_t> tokens;
    std::vector<float>   logits;
};

struct Request {
    uint64_t             id = 0;
    /* The runtime LoRA adapter this request runs with (core/lora/): its slot in the adapter pools,
     * -1 for the base model, and the uid of the load that slot held at admission. The prefix cache
     * keys on the uid, so a reload under the same name starts cold. */
    int32_t              lora_slot = -1;
    uint64_t             lora_uid = 0;
    std::vector<int32_t> prompt;
    std::vector<int32_t> output;
    SamplingParams       sp;
    int32_t              max_tokens = 0;
    std::vector<std::string> stop;
    /* Generate past the end-of-text token, up to max_tokens, so the output length is fixed (for
     * example in a steady-state decode measurement). Only the EOS token is ignored: `stop` strings
     * and max_tokens still end the request, as in vLLM. */
    bool                 ignore_eos = false;
    int                  priority = 0;
    bool                 stream = false;

    ReqState             state = ReqState::Waiting;
    int32_t              n_computed = 0;      /* prompt tokens already in the KV */
    int32_t              n_cached = 0;        /* of those, how many came from the prefix cache */
    std::vector<int32_t> blocks;              /* per KV group, flattened by group offset */
    int32_t              state_slot = -1;     /* linear/conv state slot */

    /* The prompt's media, null for text (core/mm): each item's patches, where its token run sits
     * in `prompt`, the rotary layout those runs give, and after the encoder runs, the rows that
     * replace the placeholders' embeddings. */
    std::shared_ptr<mm::PromptMedia> media;

    int32_t              n_draft = 0;         /* speculative window this step */
    /* The proposal itself: Sampler::build_step walks a constraining grammar over these tokens to
     * mask each draft position, then rewinds. Points into the scheduler's array for this request
     * (SchedReq owns this Request, so it outlives it) and is written alongside n_draft. Null when
     * n_draft is 0. */
    const int32_t*       draft = nullptr;
    int32_t              n_accepted = 0;      /* from the previous verify */

    /* Whether this request may be speculated on. Written by the engine after every step, read by
     * the scheduler when sizing the next. Only a currently constraining grammar clears it: the
     * mask at draft position i depends on which of 0..i-1 were accepted, which the host automaton
     * cannot know before the verify. A lazy grammar clears it only once its trigger fires.
     *
     * Temperature does not clear it: acceptance compares against the token the target itself
     * drew, and each verified position has its own RNG coordinate (Engine::collect_step).
     *
     * A flag rather than a call because two drafters and an n-gram fallback inside step() must all
     * obey it; the fallback fills any proposal the others left empty. */
    bool                 may_speculate = true;
    /* Whether may_speculate can change at a commit, which only a non-rewindable grammar causes. A
     * step issued ahead is sized before the preceding commit, so the scheduler issues ahead only
     * when this is set. Written when the sampler starts the request
     * (Sampler::speculation_fixed); false until then. */
    bool                 speculation_fixed = false;

    /* A logits row at every prompt position, not only the last: each trunk step carrying this
     * prompt names all its tokens in `out_ids` after the sampler's rows, and the engine hands them
     * to the KL scorer (core/kld.h). Set only in --kld mode, which sizes the logits buffer for
     * it. */
    bool                 score = false;

    /* End of a prefix other requests will start from, in prompt tokens; 0 for none. With linear
     * state, a prefix hit resumes from a state checkpoint, written only on the checkpoint interval
     * and at a long prompt's tail. The scheduler ends a prefill chunk here (rounded down to its
     * quantum) and writes a checkpoint (Scheduler::plan_chunk), so following requests reuse the
     * whole head. /v1/systemone sets it on the first of several questions over one state. */
    int64_t              checkpoint_at = 0;

    /* Non-null: read the raw logits at these ids at the prompt's last position (TokenScores).
     * The request still samples the one token max_tokens allows; the caller discards it. */
    std::shared_ptr<TokenScores> token_scores;

    /* An embedding request: the whole prompt in one step, never split (a bidirectional encoder
     * cannot take half its input), finished at that step with its pooled vector in the sink
     * (Sink::embedding). Nothing is generated. Only an embedding program (rad_declare_pooled)
     * serves these, and it serves nothing else. */
    bool                 embed = false;

    /* The client's conversation id (OpenRouter's `session_id`), "" when absent. A label for the
     * prefix cache's session on /sessions and the dashboard; it does not affect serving. */
    std::string          session_label;

    std::string          finish_reason;
    void*                sink = nullptr;      /* the server's per-request stream sink */
};

/* ================================================================== engine config */

/* --p2p: whether the collectives use the cards' peer access. */
enum { RAD_P2P_AUTO = 0, RAD_P2P_ON = 1, RAD_P2P_OFF = 2 };

struct Config {
    std::string model;                 /* path to the .rad, or a Hugging Face repo id (hf_model.h) */
    LoraConfig  lora;                  /* --lora, --max-loras, --max-lora-rank, --lora-target-modules */
    std::string download_dir;          /* where a repo id is fetched to; empty = the XDG cache */
    std::string radiance_home;
    std::vector<std::string> kernel_hierarchy;   /* ordered; first wins */

    int   tp = 1;
    /* The tensor-parallel wire, a deployment choice. 1 is the exact bf16 all-reduce. 0 accepts a
     * lossy wire: at two ranks libr4d sends a Walsh-Hadamard-rotated 6-bit payload (6 bits per
     * element plus a bf16 scale per 64) instead of 16 bits per element, 2.4x fewer bytes over
     * PCIe. The received sum then differs undetectably, hence a named flag, not a heuristic. */
    int   tp_wire_exact = 1;          /* --tp-wire exact|wht6 */
    /* Smallest all-reduce message, in KiB, that the lossy wire compresses; smaller ones are sent
     * exact. Compression pays only when the transfer is bandwidth-bound: a decode step's handful
     * of residual rows costs more to rotate than it saves, while a prefill chunk is far larger.
     * Decode message size scales with batch size; 128 KiB is the smallest floor that compresses
     * every message that benefits while keeping single-sequence decode exact. */
    int64_t tp_wire_min_kb = 128;     /* --tp-wire-min-kb */
    /* How the collectives reach the other cards. Auto takes the P2P kernels only if the driver
     * reports peer access for every ordered pair of the ranks' cards. `off` forces the
     * host-mailbox collectives, for boards where the driver says yes but peer traffic is broken or
     * slower. `on` forces P2P and stops startup naming the refused pairs instead of falling
     * back. */
    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */
    int64_t max_tok = 2048;            /* --max-num-batched-tokens */
    /* --busy-prefill-chunk: how much of a step a prefill chunk may take when other work is
     * pending in it (decode rows, another prefill, a request waiting for admission). -1 = auto:
     * leave those their rows plus a share for admission, but keep at least the smallest pass the
     * prefill stager arms for. 0 = off: the chunk takes every token the step has left. N > 0: cap
     * the chunk at N tokens on such a step. See Scheduler::step, docs/GUIDE.md. */
    int64_t busy_prefill_chunk = -1;
    int64_t prefill_interleave = 0;    /* --prefill-interleave: decode-only steps between chunks */
    int64_t max_seqs = 256;
    int64_t max_ctx = 0;               /* 0 = the model's training context */
    /* --rope-scaling: "yarn:FACTOR[:ORIG]" writes YaRN over the checkpoint's own rotary keys (the
     * model card's extension past its trained context, no reconvert); "none" takes a checkpoint's
     * YaRN off; empty serves what the checkpoint says. Engine::apply_rope_scaling. */
    std::string rope_scaling;

    /* VRAM budget, derived from the card and printed in full before anything is allocated.
     * Spec §6 forbids a budget derived from free VRAM at startup, which moves with other
     * processes. This one is arithmetic on known numbers: the fixed card total, every non-pool
     * allocation computed from the program beforehand, and what the driver already holds,
     * measured. core/mem/vram_budget.cpp writes the result into the two fields below, so the
     * planner, KV manager and pool report read one number. Engine::plan derives each rank's copy
     * (rank_cfg_, docs/TP3.md §1), which that rank's pools and planner both read.
     *
     * No flag sets them; zero until the resolver runs. */
    int64_t vram_weights_mib = 0;
    int64_t vram_kv_mib = 0;
    int64_t gpu_headroom_mib = 96;     /* --gpu-headroom-mib, left unclaimed on every card */
    /* --kv-cache-dtype bf16|fp8|fp8_v4|k6_v4. fp8 is an E4M3 cache with per-(sequence, head)
     * descales; it halves the per-token cost, most of the KV pool at production context. fp8_v4
     * keeps K as E4M3 and V as int4 in groups of 32, k6_v4 also K as int6 (RadBuildCtx::kv_quant),
     * served only where the plugin says so. Each changes the model's output slightly, so it is
     * explicit and bf16 is the default. No "auto": it would always resolve to bf16 and only add
     * another name. */
    std::string kv_cache_dtype = "bf16";
    int64_t host_pool_mib = 0;
    int64_t host_headroom_mib = 4096; /* --host-headroom-mib: RAM left free for the system at startup */
    /* File tier for weights: what fits in neither the VRAM slab nor the pinned host pool is read
     * from the container with O_DIRECT ahead of the ops that read it (a scheduled unit by the
     * exact prefetch, a routed expert a routed layer at a time). Off by default because disk
     * paces the step, so an accidental spill makes the server orders of magnitude slower. */
    bool    weights_disk_tier = false;

    std::string placement = "auto";    /* all_vram | layer_offload | expert_tiered | auto */
    /* --embedding-placement and --ngram-placement as given (see TablePlacement), and their
     * resolved values from Engine::plan before any rank plans (never Auto). The embedding defaults
     * to Ram; `embedding_placement_given` records whether the operator named it, because a named
     * tier that cannot be honoured stops the server, while the default falls back to the card with
     * the reason. */
    TablePlacement embedding_placement = TablePlacement::Ram;
    bool           embedding_placement_given = false;
    TablePlacement ngram_placement     = TablePlacement::Auto;
    TablePlacement embedding_resolved  = TablePlacement::Vram;
    TablePlacement ngram_resolved      = TablePlacement::Disk;
    /* Set by the resolver: pinned host MiB per rank for the token embedding when it resolves to
     * Ram, on top of host_pool_mib, so it never takes the experts' share and a dense model with no
     * pool still has room for it. */
    int64_t embedding_host_mib = 0;
    /* --vision-tower-placement: where encoder towers live. A tower is every weight declared
     * RAD_ACCESS_PER_REQUEST (the vision tower, and an audio encoder where present). Vram, the
     * default, keeps it on the card. Ram pins it in its own host memory, `tower_host_mib` per rank
     * on top of host_pool_mib like the embedding; the encoder reads it over the link once per
     * image, and its VRAM goes to the KV cache. */
    TablePlacement tower_placement = TablePlacement::Vram;
    TablePlacement tower_resolved  = TablePlacement::Vram;
    int64_t        tower_host_mib  = 0;
    /* The pinned host pool one rank allocates: --host-pool-mib and the tables placed beside it. */
    int64_t host_pool_total_mib() const { return host_pool_mib + embedding_host_mib + tower_host_mib; }
    bool    deterministic = false;
    /* --control-vector off|on. Without the flag (`disabled`) no vector is loaded: the program
     * declares no projection op, an off request costs the stock model's launches, and a request
     * that asks for one is refused by name. `off` loads the vector found for this container -- the
     * op is declared for its layers -- and starts requests off; `on` does the same with the request
     * default on. The graph is declared once, so this flag is the deployment's only decision about
     * whether the op exists at all; the value names the startup switch. */
    std::string control_vector = "disabled";
    int64_t checkpoint_interval = 2048;   /* linear-state checkpoints (spec §7.3) */
    /* Checkpoint intervals one step may carry of a prefill alone in its step, from an interval
     * boundary past the first (--prefill-span; 1 splits at every interval). The step's rows stay
     * on the interval-sized pieces a span of 1 cuts, so an architecture that runs row-local work a
     * piece at a time computes them identically and can overlap one piece's collectives with the
     * next piece's work. Every other step is planned at ChunkGeometry::shared_tok. Checkpoints
     * inside a span are not written; those at its end and at a prompt's tail are, where a span of
     * 1 writes them. */
    int64_t prefill_span = 1;
    /* Cap on linear-state (or sliding-window) snapshots held at once, for the whole server. The
     * slots are pinned host memory, all allocated at startup and never on the card (spec §7.3).
     * A slot is a sizeable fraction of a gigabyte on a linear-attention model, so raising the cap
     * trades host RAM for retained checkpoints. -1 is `auto`, resolved by the engine when the
     * store is opened (checkpoint_slots_auto in core/mem/prefix.h). */
    int64_t checkpoint_slots = -1;
    bool    prefix_cache = true;
    /* geometric-backoff | tip-only | keep-all. §19.1 records the default as unmeasured;
     * core/mem/prefix.h ships the two extremes so they can be compared. */
    std::string checkpoint_policy = "geometric-backoff";

    /* ---------------- IDLE SESSION TIERS (spec §7.3) ----------------
     *
     * A finished turn's prefix holds KV blocks in the cache, making the next turn cheap. With a
     * host tier, a finished conversation is copied to host memory at once and its VRAM released
     * as soon as something needs it (the expert slab's loan, or a request), so no threshold
     * decides eviction. A full host tier drops its least recently used entries. Nothing is
     * written to disk. Restoring a 200K-token turn costs a PCIe copy; recomputing it takes
     * minutes.
     *
     * The tier is off until it has a size. */
    int64_t prefix_cache_mib = 0;             /* the host tier's size. 0 is off */
    /* Seconds an unused conversation is kept in any tier (VRAM, host memory, its snapshots). 0
     * keeps everything until the memory is needed. */
    int64_t prefix_cache_ttl = 3600;

    /* The cache always lends; there is no flag. It keeps the blocks in use plus a derived buffer
     * and lends the rest to the expert plane at addresses it already owns (core/mem/kv.h). The
     * buffer's size depends on the workload (a prefill chunk, the largest recent shortfall), so it
     * is derived.
     *
     * The loan allocates nothing, so --gpu-headroom-mib is unaffected. That reserve covers what
     * this engine does not allocate, mainly kernel code objects: HIP loads one on the kernel's
     * first launch, not at startup, and faults inside the runtime rather than returning an error
     * when there is no room. */

    /* The speculative window: the depth, or under --draft-window adaptive its ceiling
     * (core/sched/draft.h). -1 is auto: engine_bringup asks rad_arch_probe for the depth this
     * container's drafter was trained at; a container without a draft head resolves to 0 with a
     * log line. Auto is the default because a merged drafter states the intent, and rad-kbench
     * and rad-tune read the same operating point; defaulting to off would pay for the head in
     * VRAM and still decode one token a step. */
    int     n_spec = -1;                /* --num-speculative-tokens; -1 = auto, 0 = off */
    DraftWindow draft_window = kDraftWindowDefault;   /* --draft-window */
    /* --draft-confidence-rule SCALE,OFFSET,FLOOR: under --draft-window confidence a sequence's
     * next window is ceil(SCALE * E + OFFSET) clamped to FLOOR..n_spec, E = sum over j of
     * q1*...*qj on the window in flight (DraftController::confidence_window). The default is the
     * rule cross-validated on two legacy K6 BetterBench runs (Qwen3.8-Flash-Next, TP2 wht6). Set
     * explicitly outside that window it is refused (Scheduler::init). */
    float   draft_conf_scale  = 1.25f;
    float   draft_conf_offset = 0.0f;
    int     draft_conf_floor  = 2;
    bool    draft_conf_rule_set = false;
    /* --draft-confidence-fallback N: the window under --draft-window confidence where the rule has
     * no say -- a cycle with more than one decoding row (the rule was fitted on single-row
     * windows, and a batched head runs as many rounds as its deepest window) and a publication
     * without q. 0 is n_spec. Refused outside that window, and above n_spec (Scheduler::init). */
    int     draft_conf_fallback = 3;
    bool    draft_conf_fallback_set = false;
    DraftPolicyKind draft_policy = kDraftPolicyDefault;   /* --draft-policy */
    /* --draft-depth-tiers: the K values the cost policy chooses between, ascending and unique.
     * The default is the ladder the policy was tuned with. An explicit tier above the executable
     * head depth is refused by Scheduler::init, which is where that depth is known (auto resolves
     * it). A top below it is left as given: a profile that prices that top is a valid reload, so
     * the list is not refused at init. Only a cycle whose top is unpriced while the named
     * baseline would run deeper is refused, by the policy that can see the shape; nothing is
     * learnt into a depth the ladder never offers (docs/ADAPTIVE-MTP.md). The default ladder is
     * trimmed to the depth instead, so a valid deployment is not refused for its deeper end. */
    int     draft_tiers[kDraftMaxTiers] = { 0, 2, 3, 4, 6 };
    int     draft_n_tiers = 5;
    bool    draft_tiers_set = false;    /* --draft-depth-tiers was given */
    /* --draft-hysteresis: `raise` (false, the default) holds only a raise for the cost policy's
     * hold and margin, a reduction is immediate; `both` (true) holds a reduction the same way
     * (DraftPolicyConfig::hold_reductions). Refused by Scheduler::init without --draft-policy
     * cost, where it would change nothing. */
    bool    draft_hold_reductions = false;
    /* --draft-calib-k: the calibration's fixed depth sequence, in the given order, duplicates
     * allowed, 2..kDraftMaxTiers values, e.g. 3,2,2,3. Not a ladder: every value must be one of
     * the effective --draft-depth-tiers, checked at startup against the list the scheduler
     * really offers. Only a calibration may set it (--draft-profile-out, --draft-policy cost):
     * with --draft-reference the no-loan reference, without it a live diagnostic whose
     * placement-affected buckets the export keeps flagged (never priced). A serving run must let
     * the cost policy decide, and Engine::begin_draft_calibration refuses the flag without them.
     * Empty: off. */
    int     draft_calib_k[kDraftMaxTiers] = {};
    int     draft_calib_n = 0;
    /* --draft-fallback-k: the depth the cost policy runs when it can price no candidate at all
     * (`DraftFallback::UnknownShape`). -1 is the policy's own default, the ladder's middle tier: a
     * profile measured at one row count (the shipped file's 47 buckets are all n_rows = 1) can
     * never price a wider batch, and the ceiling is then the wrong depth for every cycle of it --
     * one step off the top is the depth such a batch runs instead of the maximum. Deeper than the
     * executable head depth is refused at startup, and a recording run
     * (--draft-profile-out) keeps the executable depth, because the baseline is the depth its
     * export is priced at. Refused without --draft-policy cost, where it would do nothing. */
    int     draft_fallback_k = -1;
    /* --draft-reference: this run calibrates the no-loan reference. Both lenders stay off for the
     * whole run, so the buckets are the plan's start placement without borrowed copies; the file
     * is stamped with that basis and a serving run may load it while its own loans stay dynamic.
     * Not a switch a serving run sets, and not a bound: see docs/ADAPTIVE-MTP.md. */
    bool    draft_reference = false;
    /* --draft-profile: a rad-draft-profile/4 file measured on this deployment. Read once before
     * the scheduler initialises; a fingerprint or execution mismatch leaves the cost table
     * inactive and names the difference, and Cost is then refused by Scheduler::init. An older
     * format is refused by name: it cannot be bound to this deployment. */
    std::string draft_profile;
    /* --draft-profile-out: where a calibration run writes the cost profile it measured, with the
     * same fingerprint and execution the reader derives at startup. The observation ring records
     * under legacy/adaptive; written once at the end of serve(), after the ring is flushed, and
     * only when the run measured at least one bucket; never over an existing file. Under cost the
     * table is record-only: the run issues the admitted fallback at its ceiling -- the start cycle
     * included, which legacy drafting never executes -- and prices nothing from what it records. */
    std::string draft_profile_out;

    /* ---------------- KL DIVERGENCE AGAINST A REFERENCE (core/kld.h) ----------------
     *
     * A mode, like --debug-graph: the engine starts as for serving, then runs a fixed corpus
     * through prefill with a full-vocabulary logits row at every scored position instead of
     * listening. `--kld-record DIR` writes this model's log-probabilities to DIR as the reference;
     * `--kld-ref DIR` scores this model against them, computing only its own logits. Exactly one
     * of the two. */
    std::string kld_record;             /* --kld-record DIR */
    std::string kld_ref;                /* --kld-ref DIR */
    std::string kld_lora;               /* --kld-lora NAME: score through this --lora adapter */
    std::string kld_lora_sources;       /* --kld-lora-sources PREFIX: only documents whose source starts so */
    std::string kld_corpus;             /* --kld-corpus FILE, read by --kld-record only */
    std::string kld_out;                /* --kld-out FILE: the report as JSON, rows beside it */
    bool kld() const { return !kld_record.empty() || !kld_ref.empty(); }

    /* --mm-max-patches: the most patches one encoder pass carries, which bounds the largest image
     * (one image is one pass) and sizes the encoder's activations. -1 is `auto`: the default below
     * when the container carries a vision tower and 0 when it does not. 0 serves text only and
     * keeps a tower the container carries off the card. */
    int64_t mm_max_patches = -1;
    /* The audio encoder's pass in frames, resolved at bring-up from the plugin's probe when the
     * container carries an audio encoder; --mm-max-patches 0 (text only) turns it off too. */
    int64_t mm_audio_frames = -1;

    std::string host = "0.0.0.0";
    /* The chat endpoint's default `reasoning_effort`, empty for the template's own default.
     * Passed through verbatim: the legal set belongs to the template. */
    std::string reasoning_effort;
    /* --reasoning-effort-map FROM=TO,...: an effort a request sends, replaced before the template
     * sees it. */
    std::map<std::string, std::string> reasoning_effort_map;
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

    /* Sampler defaults for a request that sends none. Unset, they resolve from the container and
     * the `generation_config.json` beside it (resolve_sampling_defaults in engine.cpp), since a
     * checkpoint is tuned for its stated sampling; set, they override both. -1 means unset,
     * because 0 is a legal value for each (greedy temperature, top_k at the candidate width, min_p
     * off). */
    float   sample_temp  = -1.0f;
    int     sample_top_k = -1;
    float   sample_top_p = -1.0f;
    float   sample_min_p = -1.0f;
    /* The remaining sampler defaults for requests that omit the field. Optional, because -1 is a
     * legal value for some (presence penalty, DRY multiplier). Unset, SamplingParams' own value
     * applies. */
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
    /* --max-stop-strings / --max-stop-bytes: count and length limits on a request's stop strings.
     * Every stop string is searched in all generated text, so these bound that scan's cost. */
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
     * See TransportOptions (server/http.h). */
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
    /* A token-budget processor's (Gemma 4's) soft tokens an image. */
    std::optional<int32_t> image_tokens;
    bool media_flags_set() const {
        return image_min_pixels || image_max_pixels || video_min_pixels || video_max_pixels ||
               video_fps || video_min_frames || video_max_frames || video_max_frame_tokens ||
               max_source_pixels || image_tokens;
    }

    /* An operator-supplied generation_config.json, overriding the search described above. */
    std::string generation_config;
    /* --override-chat-template: a Jinja file replacing the template the container (or checkpoint
     * directory) carries, for chat rendering and for the reply format derived from it (reasoning
     * markers, tool calls). Empty serves the container's. */
    std::string override_chat_template;

    bool    debug_graph = false;
    bool    debug_selection = false;
    bool    debug_placement = false;
    bool    profile_ops = false;
    /* --live: the live view (spec §16). Off by default like everything else in that section. */
    bool    live_view = false;

    /* libref is the oracle and last resort, not a deployment: an op falling back to it runs orders
     * of magnitude slower with no visible sign. It is refused unless this debug flag is set
     * (startup.h, rad_gate_reference_kernels). */
    bool    accept_reference_kernels = false;

    /* Where each operator-set option came from, keyed by the option table's spelling: "command
     * line" or the supplying environment variable. Absent means default or resolved at startup;
     * the startup report (core/diag/diag.h) prints this beside every value. */
    std::map<std::string, std::string> given;
};

}  /* namespace rad */

/* Declared here because config.cpp is owned by the engine, and every tool parses the same flags,
 * so `rad-info --model X` and `radiance --model X` agree on what a flag means. */
namespace rad {
int  config_parse(int argc, char** argv, Config* c);
void config_usage(const char* argv0);
}
