/* rad_runtime.h -- the run phase. Ordinary C++ with one rule: it may only issue handles obtained
 * during declare. Nothing else is constrained; in particular there is no capture discipline on
 * this code, because a graph is built out of the kernels' own launch DESCRIPTIONS (RadDescribeFn
 * in rad_abi.h) rather than by capturing the step. See spec.md §3.2.
 */
#ifndef RAD_RUNTIME_H
#define RAD_RUNTIME_H

#include "rad_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RadCtx RadCtx;

/* ================================================================== the step batch */
/* Built ONCE PER STEP BY THE CORE and shared by every layer and every KV group. Rebuilding it per
 * KV-cache group and per layer instead turns a substantial share of a decode step into host work
 * with the device idle, which is the cost spec §8 exists to avoid.
 *
 * Every pointer here is device memory unless the field name says _h. The host mirrors exist
 * because the scheduler and the run phase both need the bounds without a device read. */
enum { RAD_PHASE_PREFILL = 0, RAD_PHASE_DECODE = 1, RAD_PHASE_MIXED = 2 };

typedef struct RadKVGroupBatch {
    rad_kvgroup group;
    int32_t*    slot_mapping;    /* [n_tok] flat slot index per token, device */
    int32_t*    block_table;     /* [n_seq, max_blocks] device */
    int64_t     block_table_pitch;
    int32_t*    seqused;         /* [n_seq] context length in this group, device */
    int32_t*    state_index;     /* [n_seq, state_index_pitch] state slots per sequence, device */
    /* HOW MANY SLOTS A SEQUENCE NAMES, and it is 1 unless the group's kernel asked for scratch
     * beside the state. A linear kernel serving speculation does: it cannot commit the state
     * until acceptance is known, so it keeps the committed one in column 0 and its own replay
     * scratch in the columns after it. The engine zeroes every one of them when the sequence is
     * admitted and otherwise does not look inside. Zero means 1, for a caller that predates it. */
    int64_t     state_index_pitch;
    int64_t     block_size;
    int64_t     max_blocks;
} RadKVGroupBatch;

typedef struct RadBatch {
    int      phase;              /* RAD_PHASE_* -- the run phase uses it to index bucket tables */
    int      step;               /* monotonic, for the heat engine and for logs */

    int64_t  n_tok;              /* total tokens across all sequences in this step */
    int64_t  n_seq;

    const int32_t* token_ids;    /* [n_tok] device */
    const int32_t* positions;    /* [n_tok] device */
    const int32_t* cu_seqlens;   /* [n_seq+1] device: sequence boundaries within this step */
    const int32_t* seq_ids;      /* [n_seq] device: engine-side sequence identity */

    /* WHICH TOKENS THE SAMPLER WANTS LOGITS FOR: indices into THIS STEP's tokens, [n_out].
     *
     * A prefill chunk computes hidden states for every token in it and the sampler wants one row
     * per sequence -- the last. Without this array the architecture plugin has to run the lm_head
     * GEMM over all n_tok rows, which is n_tok/n_seq times the work and, at max_tok 8192 against
     * a 248K vocab, an 8 GiB logits buffer instead of a 63 MiB one. The plugin issues gather_rows
     * against it and then multiplies only what came back (docs/OPS.md, "Vocabulary edges").
     *
     * n_out is 0 for a chunk that finishes no sequence, which is the ordinary middle of a long
     * prefill: no logits are produced at all and the lm_head is not touched. */
    int64_t        n_out;
    const int32_t* out_ids;            /* [n_out] device */

    const int32_t* q_lens;       /* [n_seq] device: query length this step */
    const int32_t* ctx_lens;     /* [n_seq] device: context length BEFORE this step */
    int32_t        max_q_len;    /* host-side bounds -- these select kernel bands */
    int32_t        max_ctx_len;

    int              n_kv_groups;
    const RadKVGroupBatch* kv;   /* [n_kv_groups] */

    /* Speculative window (spec §10). n_spec == 0 for an ordinary decode step. */
    int            n_spec;             /* draft tokens per sequence this step */
    const int32_t* num_accepted;       /* [n_seq] device, from the PREVIOUS verify */
    const int32_t* spec_parent;        /* [n_tok] device, tree layout; null for a linear chain */
    const uint16_t* tree_ancestors;    /* legacy [n_seq,16] device; null for wide trees too.
                                      * Wide tree attention must use spec_parent, not causal masking. */

    /* THE DRAFT PASS. 0 is an ordinary step and the plugin runs the trunk; a non-zero value runs
     * ONLY the draft head, and the SIGN says which kind of pass:
     *
     *    r < 0   a HISTORY pass: one row a token, gathered from the trunk step's rows
     *            (out_ids). The head attends to its own K and V, and nothing else in the engine
     *            writes those -- a prefill fills the trunk's layers and leaves the head's
     *            untouched. This fills them for the positions the step just committed. When
     *            `n_draft_out` is non-zero the pass is ALSO draft round 1: it ends in the lm_head
     *            over the rows `draft_out_ids` names, one a drafting sequence (see there).
     *    r > 1   draft round r, ONE-BASED. One row a sequence, reading the head's own output
     *            from the round before, ending in the lm_head; the token it samples is the r-th
     *            draft. There is no pass with r == 1: the history pass carries that round.
     *
     * Zero is the ordinary step so that a zeroed RadBatch runs the model, the same reason
     * `num_accepted` is one-based (docs/OPS.md).
     *
     * On any such pass `token_ids` is already shifted: the head at index i reads the token at
     * i + 1, and the batch builder applies that shift so no kernel and no plugin has to know it.
     * Everything else in this struct means what it means on any other step. */
    int32_t        draft_pass;

    /* Linear-state checkpointing (spec §7.3): tokens at which the scheduler wants a checkpoint
     * written, already aligned to the checkpoint interval by the chunk split. */
    int            n_checkpoints;
    const int32_t* checkpoint_tok;     /* [n_checkpoints] device: index into this step's tokens */
    const int32_t* checkpoint_slot;    /* [n_checkpoints] device: destination snapshot slot */

    /* THE DECODE / PREFILL SPLIT WITHIN A MIXED STEP.
     *
     * Sequences [0, n_seq_decode) carry decode rows and the rest carry prefill chunks. The
     * scheduler orders every step that way, so a block that has to treat the two differently can
     * issue over each as a CONTIGUOUS sub-range of the per-sequence arrays -- `cu_seqlens +
     * n_seq_decode`, `state_index + n_seq_decode * pitch`, and so on -- while leaving the token
     * operands at their base. That works because cu_seqlens holds ABSOLUTE row offsets: a kernel
     * whose workgroup n reads cu[n] and cu[n+1] finds the right rows whichever sub-array it was
     * handed, and no kernel changes to allow it.
     *
     * It exists because without the split a mixed step is prefill-shaped for every sequence in it,
     * so the decode rows travelling with a prefill chunk cannot verify a draft and produce one
     * token where they could have produced several. On an interactive load a large fraction of
     * steps are mixed, so that is a large fraction of all decode rows.
     *
     * n_seq_decode == n_seq on a pure decode step and 0 on a pure prefill one, so a block that
     * does not care can ignore all three fields, and one that does needs no phase test. */
    int64_t n_seq_decode;
    /* AND WHERE THEIR TOKENS END. The decode rows are tokens [0, n_tok_decode) and the prefill
     * chunks are the rest, which a per-TOKEN op needs and cannot derive: each decode row is
     * 1 + its own draft count, and those differ between sequences. */
    int64_t n_tok_decode;
    /* The longest query in each half, because max_q_len over a mixed step is the prefill chunk and
     * banding the decode half with it would hand a thousand-row GEMM's band to eight rows.
     *
     * ONLY THE DECODE ONE IS READ TODAY, by the GDN blocks. The prefill one is here because the
     * pair is what makes the split legible, and because a block whose prefill path bands on a
     * per-sequence query length will want it -- the GDN prefill ops band on the step's TOTAL
     * tokens instead, which they take from their operands, so they never ask. Said plainly rather
     * than left to be discovered: an unread field that looks consumed is worse than one that
     * says it is not. */
    int32_t max_q_len_decode;
    int32_t max_q_len_prefill;

    /* DRAFT ROUND 1, CARRIED BY THE HISTORY PASS: [n_draft_out] device, indices into THIS pass's
     * rows, one a sequence that drafts, in the order the later rounds' rows take.
     *
     * Round 1 continues from the last position the verify step committed, with the token the
     * trunk chose after it -- and the history pass already has a row with exactly those inputs,
     * because it covers every position the verify step could have committed and repeats the last
     * committed one in the rows past it (core/sched/advance.h, ADV_HIST). A separate round-1 pass
     * would run the head's whole layer again over a row that pass has just computed. So each
     * drafting sequence's LAST history row is its round-1 row, and the pass ends in the lm_head
     * over those rows; the head's wide state at them is what round 2 continues from.
     *
     * ON A BLOCK DRAFTER'S QUERY PASS (RAD_DRAFT_BLOCK, draft_pass > 0) it is one a sequence of
     * the pass, in its order, and indexes the TRUNK STEP's rows instead: the out_ids row of the
     * last position that step committed -- first sampled row + accepted drafts -- which is where
     * a buffer the trunk filled through `out_ids` holds the hidden state the anchor was sampled
     * from. A drafter gathers through it rather than taking the host's row numbers, so its pass
     * issues the same operands every step and replays.
     *
     * 0 on a history pass that drafts nothing, and on every other pass. */
    int64_t        n_draft_out;
    const int32_t* draft_out_ids;

    /* THE PROMPT THAT FOLLOWS THIS STEP: `n_ahead` more tokens of the LAST sequence, stored in
     * `token_ids` right after its n_tok -- the next chunk of a prefill with more prompt to run, and
     * 0 on every other step (a decode step, a draft pass, a prefill's last chunk).
     *
     * Nothing is computed on them and no output depends on them. They exist so an op that fetches
     * per-token data from a slow tier can start the NEXT step's reads while this step runs: the
     * PLE gather reads its rows from the container file, a few tens of thousands of random reads
     * a chunk, and the chunk after this one is already known (docs/OPS.md, ngram_ids' `ahead_ids`
     * and embed_lookup_q's `ahead`). */
    int64_t        n_ahead;

    /* ================================================ multimodal (spec §11) */

    /* THE ROTARY POSITION OF EVERY TOKEN, AND IT IS NOT `positions`. `positions` is a token's
     * index in its sequence -- the KV slot, the QSA block and every other piece of bookkeeping
     * count in it -- and a rotary embedding over several position components does not. Qwen-VL
     * style M-RoPE gives each token three components (temporal, height, width): a text token has
     * all three equal, an image token has the image's start in the first and its row and column
     * offsets in the other two, and the text after an image continues from the image's largest
     * component, so from the first image on the rotary position runs BEHIND the index.
     *
     * [3, n_tok] device, COMPONENT-MAJOR: plane c is rope_pos + c * n_tok, each n_tok long. Plane
     * 0 alone is a valid one-component position for every token whose three components agree,
     * which is every token on a pass where `rope_mixed` is 0 -- so a kernel that reads a position
     * per token (a cos/sin table row, say) takes plane 0 on such a pass and needs no change.
     *
     * Null when the program declares no encoder: nothing can move a rotary position away from
     * the index then, and `positions` is the rotary position. */
    const int32_t* rope_pos;
    /* Host-side: non-zero when some token on this pass has components that differ -- an image or
     * video span is in the rows. A block whose fast path reads one component a token takes the
     * multi-component path on such a pass. */
    int32_t        rope_mixed;

    /* ROWS THAT TAKE AN ENCODER'S OUTPUT IN PLACE OF A TOKEN EMBEDDING. The placeholder ids a
     * media span holds in `token_ids` are real ids -- a hashed n-gram embedding reads them -- but
     * the embedding the model sees at those rows is the encoder's, and this is it: row i of
     * `mm_embd` replaces the embedding of this pass's token `mm_rows[i]`.
     *
     * RANK 0 HOLDS THE ROWS AND EVERY OTHER RANK HOLDS ZEROS, which is the split a vocabulary-
     * sharded embedding lookup produces: a token's embedding comes from the one rank that owns
     * its id and the rest contribute zero to the reduction that follows. So a plugin overwrites
     * its lookup's output with these rows BEFORE that reduction, and every row arrives once --
     * whether the reduction is its own all-reduce or folded into the first layer's norm. At one
     * rank there is no difference.
     *
     * On a pass that shifts its tokens (a draft head's history pass) the rows are shifted with
     * them. 0 rows on every pass with no media in it. */
    int64_t        n_mm_rows;
    const int32_t* mm_rows;            /* [n_mm_rows] device: index into this pass's tokens */
    const void*    mm_embd;            /* [n_mm_rows, n_embd] bf16 device */

    /* THE ENCODER PASS. Non-zero `enc` runs the plugin's declared encoder (rad_declare_encoder)
     * and nothing else: no token rows, no KV, no logits. Its input is a list of patches, already
     * resized, normalised and cut by the processor, and its output is the encoder's declared
     * `out` buffer, one row per `merge` patches, which the core reads back after the pass.
     *
     * Patches are grouped into SEGMENTS -- one per image, and one per temporal group of a video
     * -- and attention runs within a segment. `enc_coord` gives each patch its place in its
     * segment, which is everything a learned position table and a 2-D rotary embedding need. */
    int32_t        enc;
    int64_t        enc_n_patch;
    const void*    enc_pixels;         /* [enc_n_patch, patch_dim] bf16 device */
    /* [4, enc_n_patch] device, COMPONENT-MAJOR like rope_pos: row, column, the segment's height
     * and its width, all in patches. Planes 0 and 1 together are a [2, enc_n_patch] position. */
    const int32_t* enc_coord;
    int64_t        enc_n_seg;
    const int32_t* enc_cu;             /* [enc_n_seg + 1] device: segment boundaries in patches */
    int32_t        enc_max_seg;        /* host: the longest segment, which bands the attention */
} RadBatch;

/* ================================================================== operands */
/* RAD_OPK_KV names a KV group's pool rather than a buffer: attention and kv_store take the cache
 * as an operand, and RadKVGroupBatch carries the slot mapping and the block table but not the
 * pool base -- the pool belongs to the block manager, not to the batch. handle is the
 * rad_kvgroup and offset is the LAYER, because a group's pool holds one cache per bound layer. */
/* RAD_OPK_WTAB names a RUN of consecutively declared weights rather than one weight, and it
 * exists because a routed mixture of experts has no single base to stride from. A layer's experts
 * are separate movement units (spec §4.1) -- after placement one of them is in VRAM, the next is
 * streamed over the link and the third is still on disk -- so "expert e is base + e*stride" is
 * true only of the all-in-VRAM case and silently wrong of every other. The core therefore
 * resolves each handle in the run the way it resolves any weight (residency table, mover event,
 * use note) and hands the kernel a DEVICE ARRAY OF POINTERS, one per expert, in declaration
 * order. See the RadTensor shape rule at RAD_WTAB below. */
enum { RAD_OPK_BUF = 0, RAD_OPK_WEIGHT = 1, RAD_OPK_RAW = 2, RAD_OPK_NONE = 3, RAD_OPK_KV = 4,
       RAD_OPK_WTAB = 5 };

typedef struct RadOperand {
    uint8_t  kind;
    uint32_t dtype;      /* RAD_OPK_RAW only: what the pointer points at. A buffer and a weight
                          * carry their dtype in the declaration; a raw pointer is a batch field
                          * the core did not allocate and has nowhere else to say. RAD_DT_INVALID
                          * means "the kernel knows from the op schema position", which is true
                          * for a specialised kernel and false for libref. */
    uint32_t handle;     /* rad_buf or rad_weight */
    void*    raw;        /* RAD_OPK_RAW: a pointer the core did not allocate (a batch field) */
    int64_t  offset;     /* element offset into the buffer or weight */
    int64_t  rows;       /* 0 = the declared extent; >0 = only this many rows of dim 0 */
    int64_t  cols;       /* 0 = the declared extent; >0 = a COLUMN SLICE of that width, with the
                          * declared width kept as the stride. This is what makes a fused qkv
                          * projection usable: q, k and v are views of one buffer, and without it
                          * the architecture plugin has to declare three buffers and copy. */
} RadOperand;

#define RAD_B(h)            ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, 0, 0, 0 })
#define RAD_B_AT(h, o)      ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, (o), 0, 0 })
#define RAD_B_N(h, o, n)    ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, (o), (n), 0 })
/* A column slice: `w` columns starting at column `c`, keeping the buffer's declared width as the
 * stride. RAD_B_COL(qkv, 0, n_q) is the q view of a fused projection. */
#define RAD_B_COL(h, c, w)  ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, (c), 0, (w) })
#define RAD_W(h)            ((RadOperand){ RAD_OPK_WEIGHT, RAD_DT_INVALID, (h), 0, 0, 0, 0 })
/* A WEIGHT TABLE: the `n` weights declared at handles h, h+1, ... h+n-1 -- which is what a
 * declaration loop over experts produces, because rad_decl_weight hands out consecutive indices.
 *
 * WHAT THE KERNEL RECEIVES. `RadTensor::data` is a device array of `n` `void*`, entry e being
 * expert e's first byte. `shape[0]` is n and `stride[0]` is 0, which is the marker: a zero
 * leading stride says the axis is not addressable by arithmetic and the table must be indexed.
 * `dtype` and `shape[1..]` / `stride[1..]` are ONE entry's -- every weight in a run must agree
 * about those, and the core refuses the issue by name if they do not, because a table whose
 * entries have different shapes is a table the kernel cannot walk.
 *
 * The table is rebuilt only when a pointer in it actually changes, so a resident model pays one
 * comparison per expert per issue and no copy at all. */
#define RAD_WTAB(h, n)      ((RadOperand){ RAD_OPK_WTAB,   RAD_DT_INVALID, (h), 0, 0, (n), 0 })
#define RAD_KV(g, layer)    ((RadOperand){ RAD_OPK_KV,     RAD_DT_INVALID, (g), 0, (layer), 0, 0 })
#define RAD_P(ptr)          ((RadOperand){ RAD_OPK_RAW,    RAD_DT_INVALID, 0, (void*)(ptr), 0, 0, 0 })
#define RAD_P_T(ptr, dt)    ((RadOperand){ RAD_OPK_RAW,    (dt), 0, (void*)(ptr), 0, 0, 0 })
/* A raw pointer to a DENSE 2-D array: [rows, cols], packed. `cols` on a raw operand is not the
 * column SLICE it is on a buffer -- there is no declared width to slice out of -- it is the second
 * extent, and giving it is the only way a batch field that is genuinely two-dimensional can say so.
 * The block table is the case that forced it: [n_seq, max_blocks], and passed as a rank-1 array of
 * n_seq the width is simply gone. Every kernel that reads it then has to guess, and the only two
 * guesses available are to refuse the operand for not being rank 2 or to read a width of one. */
#define RAD_P_T2(ptr, dt, r, c) \
    ((RadOperand){ RAD_OPK_RAW, (dt), 0, (void*)(ptr), 0, (r), (c) })
#define RAD_NONE            ((RadOperand){ RAD_OPK_NONE,   RAD_DT_INVALID, 0, 0, 0, 0, 0 })

#ifdef __cplusplus
#define RAD_OPD(...)  std::initializer_list<RadOperand>{ __VA_ARGS__ }
#else
#define RAD_OPD(...)  ((const RadOperand[]){ __VA_ARGS__ }), \
                      (int)(sizeof((const RadOperand[]){ __VA_ARGS__ }) / sizeof(RadOperand))
#endif

/* ================================================================== issue */
/* The core resolves each weight handle to a pointer, waits on the mover's event if the weight is
 * in flight, indexes the op's bucket table with `n`, and dispatches to the device or host
 * implementation according to the current execution site.
 *
 * `n` is the value of the op's ranged parameter -- at most one per op, which is what keeps this
 * a single integer on the hot path. RAD_N_BATCH means "the batch's token count", which is what
 * almost every op wants. An op with no ranged parameter ignores it.
 *
 * If an op received a void*, the weight's location would be fixed at declare and placement could
 * not be a core feature at all. This is the single ABI decision the whole placement design rests
 * on (spec §3.2). */
#define RAD_N_BATCH ((int64_t)-1)

int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n);

#ifdef __cplusplus
} /* extern "C" */
#include <initializer_list>
inline int rad_issue(RadCtx* c, rad_op op, std::initializer_list<RadOperand> opd, int64_t n) {
    return rad_issue(c, op, opd.size() ? opd.begin() : nullptr, (int)opd.size(), n);
}
extern "C" {
#endif

/* THE SECOND LANE, for overlapping a collective -- or an arm no shard divides -- with compute.
 * `rad_lane` selects which stream subsequent issues launch on (0 or 1); `rad_lane_join(c, from,
 * to)` orders `to` behind everything already issued on `from` without blocking the host or
 * draining either. The runtime joins lane 1 back into lane 0 at the end of every step, so a
 * missing join costs speed and not correctness.
 *
 * THREE RULES, and Ctx::lane1_ in core carries the reasons: every collective must ride the SAME
 * lane (the all-reduce numbers its scratch slots per launch and the two ranks must agree); a
 * kernel that carries a device-wide arrival counter must key it on the STREAM, which libr4d's
 * split-K GEMM does and a plugin that does not may not run here; and the buffer planner packs
 * transients from a SINGLE LINEAR ORDER, so every buffer the second lane touches has to be named
 * to rad_buf_concurrent, which gives up a few buffers' worth of the arena's sharing. */
int rad_lane(RadCtx* c, int lane);
int rad_lane_join(RadCtx* c, int from, int to);

/* Same, with the ranged value taken from the batch. */
#define RAD_ISSUE(c, op, ...) rad_issue((c), (op), RAD_OPD(__VA_ARGS__), RAD_N_BATCH)
#define RAD_ISSUE_N(c, op, n, ...) rad_issue((c), (op), RAD_OPD(__VA_ARGS__), (n))

/* FAIL THE STEP WITH A REASON. An architecture block's step() returns void, so without this the
 * only way to refuse a batch it cannot serve is to issue something invalid and let the operand
 * check reject it -- which reports the wrong cause. A block that knows it does not serve
 * a batch shape says so here, and the engine stops with `what` in the message rather than
 * producing numbers nobody should trust. Returns the status it set, so `return rad_step_fail(...)`
 * reads naturally in a helper. */
int rad_step_fail(RadCtx* c, const char* what);

/* ================================================================== run-phase queries */
const RadBatch* rad_batch(RadCtx* c);
int             rad_rank(RadCtx* c);
int             rad_world_size(RadCtx* c);
RadStream       rad_stream(RadCtx* c);

/* The routing result for a MoE layer, resolved on device against the current residency table.
 * Routing NEVER blocks the compute stream: the histogram is copied back asynchronously and the
 * heat engine consumes it on the next step (spec §5.5). This returns the device-side buffers,
 * not host-visible expert ids -- the plugin issues its expert ops against them. */
typedef struct RadRouting {
    const int32_t* expert_ids;    /* [n_tok, top_k] device */
    const float*   expert_w;      /* [n_tok, top_k] device */
    const int32_t* expert_count;  /* [n_expert] device: tokens routed to each */
    const int32_t* sorted_tok;    /* [n_tok * top_k] device: tokens grouped by expert */
    const int32_t* expert_offset; /* [n_expert+1] device */
    int64_t        top_k;
    int64_t        n_expert;
} RadRouting;

/* Record that this layer's routing lives in these buffers, so the heat engine can read the
 * histogram back and the mover can be told what was wanted. The plugin computed them with
 * ordinary declared ops; this is the handoff, not a computation. */
int rad_route_report(RadCtx* c, int layer, const RadRouting* r);

/* WHERE A LAYER'S EXPERT COUNTS CAN BE WRITTEN IN PLACE: a device row of at least `n_expert` int32
 * that the core owns and carries to the host itself, once, at the end of the pass. A router that
 * writes its counts here and reports this pointer as `expert_count` hands them over for nothing;
 * any other pointer is copied into this row, which is a dispatch a routed layer on the compute
 * stream. Null when the core keeps no histogram for `layer` or its row is narrower than
 * `n_expert` -- the router then writes its own buffer and reports that. */
int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert);

/* A weight's pointer, for the rare op that genuinely needs one (a plugin-private kernel invoked
 * outside the op vocabulary). Waits on the mover. Using this forfeits nothing except the ability
 * to move that weight while the pointer is live, so it is legal only within one issue. */
void* rad_weight_ptr(RadCtx* c, rad_weight w);
void* rad_buf_ptr(RadCtx* c, rad_buf b);

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_RUNTIME_H */
