/* batch.h -- the step batch, built ONCE PER STEP BY THE CORE and shared by every layer and every
 * KV group (spec §8).
 *
 * vLLM builds this per KV-cache group, per layer, in Python, and that shape is what this file
 * exists to avoid: a model whose linear-attention layers fall into several KV-cache groups runs
 * the metadata builder once per group, and every run after the first produces values identical to
 * it -- dozens of tiny dispatches a step for numbers already computed, and host time on the step
 * path is time the GPU spends idle.
 *
 * So: one build, one owner, every consumer reads the same buffers.
 *
 * Everything here comes out of pools sized at max_tok / max_seqs at init. Nothing on this path
 * allocates. Staging is PINNED HOST memory crossing in one async copy a rank, out of a RING so
 * that a build is never written while an earlier build's copy is still reading it -- the cost of
 * getting that wrong is a torn block table, which shows up as a sequence reading another
 * sequence's KV and reads as a model quality problem -- and so that the host can stage several
 * passes ahead of the card without waiting for any of them.
 */
#pragma once
#include "advance.h"
#include "device/device.h"
#include "geometry.h"
#include "mem_iface.h"
#include "metrics.h"

namespace rad {

/* One sequence's participation in one step. The scheduler fills these; the builder turns them
 * into device arrays and touches nothing else. */
struct StepEntry {
    const Request* req = nullptr;
    uint64_t seq = 0;             /* the KV manager's key -- the request id */
    int32_t slot = -1;            /* scheduler slot: where the scheduler keeps the request */
    int32_t card = -1;            /* the card's StepSlot index, [0, max_seqs): the sequence
                                   * identity in this batch (SchedReq::card) */
    int32_t n_tokens = 0;         /* query length this step */
    int32_t ctx_len = 0;          /* computed tokens BEFORE this step -- the first token's position */
    int32_t n_spec = 0;           /* draft tokens included in n_tokens (decode only) */
    /* HOW MANY TOKENS THE PREVIOUS STEP COMMITTED, and it is ONE-BASED: the token that step was
     * given plus however many of its drafts survived. Both linear kernels read it that way --
     * gdn_conv_update reads its rolling window at `num_accepted - 1` and gdn_recurrent_update
     * takes the state the (num_accepted)th token left -- so 1 is the ordinary decode step that
     * drafted nothing, which is exactly what they assume when the operand is absent. Zero is not
     * a value either of them can read; the scheduler's own `SchedReq::n_accepted` counts DRAFTS,
     * which is one less. */
    int32_t n_accepted = 1;
    int32_t conv_cursor = 0;      /* read offset inside the rolling conv window (§10) */
    int32_t ckpt_tok = -1;        /* token index WITHIN this entry to snapshot at, or -1 */
    int32_t ckpt_slot = -1;
    /* THE SNAPSHOT THIS ENTRY RESUMES FROM, or -1. Set on the first step after a prefix hit and
     * never again: the recurrent half of a hit is carried by this and by nothing else, because
     * the adopted blocks describe the attention layers only. */
    int32_t ckpt_restore = -1;
    const int32_t* draft = nullptr;   /* the drafter's proposal, appended after the committed tail */
    int32_t n_draft = 0;
    /* WHICH LOGITS ROW OF THE STEP JUST RUN holds the hidden state this entry continues from,
     * or -1 for the ordinary rule. It exists for the MTP drafter's history pass: the trunk step
     * produced 1 + n_spec rows for a verifying sequence, and the head gathers the run of them
     * starting here. Filled into `out_ids`, so the head's gather picks them out of the same
     * buffer.
     *
     * A pass that gathers its OWN output names rows of itself through the same field -- the
     * DFlash2 query pass does, because the rows it wants logits for are the mask positions of the
     * block it just ran. Which buffer out_ids indexes is the architecture block's business; this
     * is only the list. */
    int32_t hidden_row = -1;
    /* HOW MANY ROWS FROM `hidden_row`, or 0 for the rule that predates it: one a token. The MTP
     * history pass wants exactly its n_tokens and says 0; the DFlash2 query pass runs `block`
     * tokens and wants the block-1 mask rows, so it says block-1 and starts one row in. */
    int32_t hidden_n = 0;
    /* THIS ENTRY'S LAST ROW IS DRAFT ROUND 1 (RadBatch::draft_out_ids): set on a history-pass
     * entry whose sequence drafts, whose rows end with the position round 1 continues from. */
    bool    draft_out = false;
    /* A BLOCK QUERY PASS's entry: WHICH ROW OF THE TRUNK STEP'S out_ids its anchor was sampled
     * at, or -1. Staged into RadBatch::draft_out_ids, one a drafting sequence, so a drafter that
     * reads the trunk's hidden states at the last accepted position gathers them through a device
     * index -- the same pointer every step, which is what lets the pass be replayed. */
    int32_t src_row = -1;
    bool    is_prefill = false;
    bool    produces_token = false;   /* this entry's last token is the sequence's last */
    /* AN EXPLICIT ROW LIST, ONE ENTRY A ROW OF THIS PASS, or empty for the positional rules
     * below. It is the only form that can express a tree's accepted path: after a verify step the
     * rows to read (a context pass) or to write (a history pass) are the accepted NODES, in
     * emission order, and those are not a contiguous run -- a sibling the target rejected can sit
     * BETWEEN two accepted ones.
     *
     * When it is set it WINS over `hidden_row`/`hidden_n` and over the produces_token rule, and
     * its length is the pass's row count: the consumer indexes it by row, so a shorter list would
     * be read past its end. Empty is the identity, which is every chain and every prefill. */
    const int32_t* stage_out = nullptr;
    int32_t n_stage_out = 0;
    /* HOW MANY OF THIS ENTRY'S ROWS MAY WRITE KV, or 0 for all of them.
     *
     * A DRAFT PASS THAT MIRRORS A VERIFY STEP RUNS OVER ROWS THE TARGET MAY REJECT, and those rows
     * must never reach a cache that outlives the step. A paged draft cache is addressed by position,
     * so a rejected row's slot is a position the sequence never occupied -- writing it puts KV
     * computed from text the model never emitted where a later pass can find it. The slot is
     * therefore staged -1 past this count, which every kernel already reads as "no slot"
     * (libref/ref_attn.cpp kv_store skips it, and so does libr4d).
     *
     * It applies to PAGED groups only. A state group's row maps to one state a sequence, not to a
     * position, so masking there would drop a sequence's state write entirely.
     *
     * 0 means "every row writes", which is the rule for a trunk step and for a prefill chunk: those
     * rows are committed by definition. */
    int32_t n_write = 0;
    /* A DEVICE ROW: the card fills its tokens, positions, rollback count and slots from the slot's
     * StepSlot before the pass runs (advance.h), and what the host staged for them is replaced.
     * `ctx_len` is then an upper bound, which is what the host-side ceilings read. The other three
     * are advance.h's row descriptor: the verify step's token row of the row's first verified
     * position, its first sampled row (-1 for none), and the mode's own number. A verify step's
     * entries always carry them, for its acceptance. */
    bool    dev = false;
    int32_t dev_tok = 0;
    int32_t dev_out = -1;
    int32_t dev_aux = 0;
};

struct StepPlan {
    std::vector<StepEntry> e;
    int64_t n_tok = 0;
    int     phase = RAD_PHASE_DECODE;
    /* THE DRAFT PASS, straight through to RadBatch::draft_pass. 0 is a trunk step; r > 0 runs
     * only the draft head, r < 0 is a history pass. */
    int     draft_pass = 0;
    /* MTP frontier rows are siblings at one position, never a committed token run. */
    bool    mtp_frontier = false;
    /* HOW FAR AHEAD OF ITS POSITION A ROW READS ITS TOKEN. The MTP head pairs the trunk's hidden
     * state at index i with the embedding of the token at i + 1 (rad_block_mtp_fp8.h), so its
     * passes say 1 and the shift is applied once, here, rather than in a kernel. It is NOT a
     * property of "being a draft pass": DFlash2's query block embeds the token at its own
     * position -- the anchor at the row that holds it and a mask everywhere after -- and says 0.
     * Deriving it from `draft_pass` instead feeds such a drafter the token one past its anchor. */
    int     tok_shift = 0;
    int     n_spec = 0;               /* the batch-wide speculative window */
    bool    spec_is_tree = false;     /* a tree drafter fills spec_parent; a chain leaves it null */
    /* THE SPLIT, for a block that issues over each half separately. Sorted above, so the decode
     * rows are sequences [0, n_seq_decode) and the prefill chunks are the rest. */
    int64_t n_seq_decode = 0;
    int64_t n_tok_decode = 0;
    int32_t max_q_len = 0;
    int32_t max_ctx_len = 0;

    void reset() {
        e.clear();
        n_tok = 0; phase = RAD_PHASE_DECODE; draft_pass = 0; tok_shift = 0; n_spec = 0;
        mtp_frontier = false;
        spec_is_tree = false; max_q_len = 0; max_ctx_len = 0; n_seq_decode = 0; n_tok_decode = 0;
    }
};

/* ------------------------------------------------------------------ the builder */
class BatchBuilder {
public:
    /* `prog` is non-const because derived buffers are written into the arena, and the arena
     * pointer lands in BufferInfo::ptr after the buffer plan runs. */
    /* What the builder needs about one rank: which card its slab lives on, which stream its
     * upload rides, and which Program holds the arena a derived buffer is copied into. */
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
        Program*  prog   = nullptr;
    };

    int  init(Program& prog, const Config& cfg, const ChunkGeometry& geo,
              const KVGeom& kvg, const std::vector<RankIO>& ranks);
    void fini();

    /* The bytes init() allocates on each rank's card -- the step slab and, with an encoder, the
     * media staging -- computed without allocating, for the VRAM budget. -1 when the program is
     * one init() would refuse. */
    static int64_t device_bytes(Program& prog, const Config& cfg, const KVGeom& kvg);

    /* Build the batch for this step. The returned pointer is stable until the kOut-th next call
     * and is the pointer the rank threads share -- because the ranks share an address space, the
     * batch is shared by pointer rather than broadcast (spec §1). Null on a device failure, which
     * is fatal for the step. */
    const RadBatch* build(const StepPlan& plan, const IKVManager& kv, int step);

    /* The batch the LAST build produced for one rank. Same contents as build()'s return
     * except for the device pointers, which name that rank's staging slab. */
    const RadBatch* batch_for(int rank) const;

    /* THE ADVANCE'S VIEW OF THE LAST BUILD on one rank: the slab and every offset in it. The mode,
     * the round, the slot state and the inputs are the caller's (advance.h). */
    AdvanceArgs advance_args(int rank) const;

    /* THE LONGEST SEQUENCE A BATCH CAN DESCRIBE, in tokens: what the block-table rows were sized
     * for at init. A position at or past it has no column to be addressed through. */
    int64_t max_ctx() const { return max_ctx_; }
    bool qsa_dense_for(const StepPlan& plan, int64_t exact_to) const;
    /* Winner metadata is mapped/pinned and reused only after the next trunk's
     * sampling fence, which also retires the previous winner commit. For a step of S sequences
     * and P = tree_path_pitch(): node paths [S,P], cu {0,1}, lengths [S], starts [S],
     * trunk rows [S,P], resets [S]. The commit release reads paths as draft_out_ids with
     * n_draft_out = S*P, and lengths as num_accepted = paths + S*P + 2. */
    int32_t* tree_paths() const { return tree_paths_; }
    int64_t tree_path_pitch() const { return tree_path_pitch_; }
    /* WHETHER THE LAST build VERIFIED A TREE. The advance that walks a tree's acceptance runs
     * AFTER the build and must read the topology the attention kernel masked with, so it asks the
     * build rather than re-deriving the answer from the scheduler's proposal -- which is where a
     * mismatch would surface as a wrong accepted position instead of as a refused tree. */
    bool last_spec_is_tree() const { return last_spec_is_tree_; }

    /* THE DLOOP CONTEXT SPAN, in the tail of the same mapped region: `[start,count]` a sequence,
     * laid out immediately after the node-path/lengths/starts/absolute/resets block that spans
     * [0, max_seqs*(2*tree_path_pitch()+3)+2). Zero when the deployed backend declares no span buffer.
     *
     * IT LIVES HERE RATHER THAN IN A StepOut VECTOR, and that is a lifetime rule and not a
     * preference: this region is allocated ONCE at init and reused every step, so staging a span
     * costs no per-step allocation and the upload that carries it is on the same stream, in the
     * same order, as the path and reset flags it is read with -- no separate vector to keep alive
     * across the barrier and no allocator traffic on the step path. */
    int32_t* dloop_span() const { return tree_paths_ ? tree_paths_ + span_at_ : nullptr; }
    int64_t   dloop_span_at() const { return span_at_; }
    /* Whether the program's drafter is the full-context one, so the step loop knows to stage a
     * span at all. Asked of the declared ops, which is what a tree-state program is asked too. */
    bool dloop_full_state() const { return dloop_full_; }

    /* AN ENCODER PASS (RadBatch::enc): the patches of whole segments of one or more media items,
     * staged into every rank's encoder input -- each rank runs the tower on its own card, so none
     * waits on another's output. Returns rank 0's batch like build(); batch_for() serves the rest.
     * The caller synchronises every rank's stream before the next encoder build, which it has to
     * anyway to read the pass's rows back. */
    struct EncoderPart {
        const mm::Item* item = nullptr;
        int32_t         seg0 = 0, n_seg = 0;
    };
    const RadBatch* build_encoder(const std::vector<EncoderPart>& parts, int step);
    bool    has_encoder() const { return mm_on_; }
    int64_t encoder_patches() const { return enc_patches_; }


private:
    struct Group {
        rad_kvgroup g = 0;
        int64_t     block_size = 0;
        int64_t     max_blocks = 0;
        int32_t*    h_slot = nullptr;  int32_t* d_slot = nullptr;
        /* The group's block tables on the card: a fixed place in the table region, `bt_fix`
         * int32 past its start, max_seqs rows of this build's pitch. They have no host twin -- a
         * build stages only what changed. */
        int64_t     bt_fix = 0;
        int32_t*    d_bt   = nullptr;
        int32_t*    h_used = nullptr;  int32_t* d_used = nullptr;
        int32_t*    h_sidx = nullptr;  int32_t* d_sidx = nullptr;
        int64_t     sidx_pitch = 1;    /* columns per sequence in h_sidx/d_sidx */
    };

    /* THE LAYOUT OF A BUILD: every array's place on the card and in host staging, at the same
     * offset in both, so that one copy moves a whole build.
     *
     * THE DEVICE SLAB IS ONE PER RANK AND IS NEVER DOUBLE BUFFERED. A build's upload and every
     * kernel that reads it are on that rank's compute stream, so the next build's upload is
     * ordered behind the kernels of the last one by the queue itself: nothing on the card can
     * read a slab another build is writing. What does need more than one copy is the HOST side,
     * which the upload reads when the stream reaches it and not when it is issued -- and the host
     * may be several passes ahead of the card. That staging is carved out of a ring (see Ring),
     * so any number of builds can be in flight and a build only waits when the ring is full.
     *
     * `h_*` are re-carved for every build at its place in the ring; `d_*` are rank 0's slab and
     * never move. */
    struct Set {
        int64_t host_bytes = 0, dev_bytes = 0;
        /* THE BLOCK TABLES ARE NOT PART OF THE FIXED LAYOUT'S COPY. `bt_base` is where they
         * start, and everything below it crosses whole every build. On the card the region past
         * it is every group's tables at a fixed place; in the host span it is the build's
         * PAYLOAD -- the table entries that changed, packed -- followed by the list of ranges
         * saying where each run of it lands (see BtMirror). `dev_bytes` is the ALLOCATION, which
         * holds every group's worst table. */
        int32_t* bt_h0 = nullptr;  int32_t* bt_d0 = nullptr;
        int64_t  bt_base = 0;
        /* ONE DEVICE SLAB PER RANK, and this is not an optimisation.
         *
         * The batch is built once and shared BY POINTER (spec §1), which is right for the host
         * side and wrong for the device side. A single slab on rank 0's card, filled by async
         * copies on rank 0's stream and read by every rank's kernels off its own stream, has
         * nothing ordering the other ranks' reads after rank 0's copies. A slab per rank makes
         * each rank's upload an ordinary same-stream dependency and puts the metadata on the card
         * that reads it, instead of behind PCIe. The carve is identical on every rank, so d_*
         * below are rank 0's pointers AND the offset basis for the rest. */
        std::vector<void*> dev;

        int32_t *h_tok = nullptr, *h_pos = nullptr, *h_cu = nullptr, *h_sid = nullptr;
        int32_t *h_q = nullptr, *h_ctx = nullptr, *h_acc = nullptr, *h_parent = nullptr;
        uint16_t *h_tree_ancestors = nullptr, *d_tree_ancestors = nullptr;
        int32_t *h_cktok = nullptr, *h_ckslot = nullptr;
        int32_t *h_out = nullptr;      /* [n_out] which of this step's tokens want logits */
        int32_t *h_dout = nullptr;     /* [n_draft_out] round-1 rows, or a query pass's src_row */
        int32_t *d_tok = nullptr, *d_pos = nullptr, *d_cu = nullptr, *d_sid = nullptr;
        int32_t *d_q = nullptr, *d_ctx = nullptr, *d_acc = nullptr, *d_parent = nullptr;
        int32_t *d_cktok = nullptr, *d_ckslot = nullptr;
        int32_t *d_out = nullptr;
        int32_t *d_dout = nullptr;
        /* The advance's inputs (advance.h): [4][max_seqs] row descriptors, [groups][max_seqs]
         * first positions, and one AdvGroup a group. */
        int32_t  *h_rows = nullptr, *d_rows = nullptr;
        int32_t  *h_first = nullptr, *d_first = nullptr;
        AdvGroup *h_grp = nullptr, *d_grp = nullptr;

        char*      h_derived = nullptr;   /* derivation staging, byte-addressed */
        /* [3][n_tok] rotary positions and [n_tok] encoder-row targets (RadBatch::rope_pos,
         * mm_rows): carved only when the program declared an encoder, empty otherwise. */
        int32_t   *h_rope = nullptr, *d_rope = nullptr;
        int32_t   *h_mmrow = nullptr, *d_mmrow = nullptr;
        /* [n_kv_groups][max_seqs]: EVERY conv group's state slot per sequence, not just one, since
         * a model may have more than one conv group -- see init(). */
        int32_t*   h_convslot = nullptr;

        std::vector<Group>           groups;
    };

    /* THE HOST STAGING RING. A build takes the bytes it will upload -- the fixed prefix and the
     * block-table rows it fills -- as one contiguous span, fills it, and records an event per rank
     * behind the upload. A span is reusable once every rank's upload of it has run, which is what
     * `reserve` waits for when the ring is full; at steady state it never is, because a decode
     * build's span is a small fraction of the ring. */
    struct Flight {
        int64_t begin = 0, end = 0;
        int     ev = -1;               /* row of ev_ holding one event per rank */
    };
    static constexpr int kFlights = 64;
    struct Ring {
        char*   host = nullptr;
        int64_t bytes = 0;
        int64_t head = 0;              /* where the next span starts */
        Flight  fifo[kFlights];        /* live flights, oldest at `first`, `live` of them */
        int     first = 0, live = 0;
    };

    /* What a build hands the ranks: the ABI views and the batch, ONE PER RANK, since they differ
     * only in the device pointers. kOut of them so a batch stays valid while later ones are
     * built: a step's own batch is read after its draft passes and the next step's batch are. */
    static constexpr int kOut = RAD_SCHED_MAX_SPEC + 4;
    struct Out {
        std::vector<std::vector<RadKVGroupBatch>> kv;
        std::vector<RadBatch>                     batch;
    };

    struct Derived {
        rad_buf     handle = 0;
        /* WHICH conv group this derivation is about, or -1 for none. From RadBufDecl::kv, or the
         * only conv group when the plugin left that 0 and there IS only one. */
        int32_t     conv_grp = -1;
        const char* name = nullptr;
        int         fn_index = -1;
        int64_t     off = 0;            /* byte offset into a set's derived region */
        int64_t     capacity = 0;       /* elements */
        int64_t     produced = 0;
        BufferInfo* buf = nullptr;
    };

    Program*      prog_ = nullptr;
    Config        cfg_{};
    ChunkGeometry geo_{};
    KVGeom        kvgeom_{};
    std::vector<RankIO> ranks_;

    int64_t max_tok_ = 0, max_seqs_ = 0, max_ctx_ = 0;
    int32_t* tree_paths_ = nullptr;  /* [span_at_ + max_seqs*2], mapped host winner staging */
    int64_t tree_path_pitch_ = 16;  /* fixed by configured verification capacity, never acceptance */
    /* WHETHER THE LAST build VERIFIED A TREE (batch.h at last_spec_is_tree). Set by build from the
     * plan it was handed, so the advance that follows reads the same answer the attention kernel
     * was given. */
    bool     last_spec_is_tree_ = false;
    /* WHERE THE DLOOP CONTEXT SPAN STARTS inside `tree_paths_`, and whether this program's
     * backend wants one. Both fixed at init: `span_at_` is the region's own size and never moves,
     * and `dloop_full_` is asked of the declared ops once rather than re-scanned per step. */
    int64_t  span_at_ = 0;
    bool     dloop_full_ = false;
    /* Per KV GROUP, indexed by group, and 0 for a group that is not RAD_KV_CONV.
     *
     * THIS AND `h_convslot` FEED `DeriveInput` AND NOTHING ELSE. A conv KERNEL reads its slot from
     * RadKVGroupBatch::state_index, which is per group. What reads these two is the
     * `conv_state_index` DERIVATION, which no architecture plugin currently calls --
     * rad_block_gdn_fp8.h takes `cv->state_index` straight off the group batch. Keeping them per
     * group is what makes the derivation correct, and usable, for a model with a second conv
     * group. */
    std::vector<int32_t> conv_window_;
    int32_t conv_group_ = -1;    /* the ONLY conv group, when there is exactly one; else -1 */
    int32_t n_conv_ = 0;

    Set  set_;
    Ring ring_;
    /* [flight][rank]: the events a flight records behind its uploads, reused as flights retire. */
    std::vector<std::vector<RadEvent>> ev_;
    std::vector<int>                   ev_free_;
    Out  out_[kOut];
    int  out_next_ = 0;
    int  out_last_ = -1;    /* the Out build() just filled */
    /* Per group, this build's block-table pitch: sized before the span is reserved. */
    std::vector<int64_t> pitch_;

    /* WHAT EVERY RANK'S SLAB HOLDS IN ITS BLOCK-TABLE REGION, so a build stages only what changed.
     *
     * A sequence's table is its whole context in blocks, thousands of entries a group at long
     * context, and restaging every row whole every build cost the scheduler time in proportion
     * to context times sequences times the several builds of a speculative step -- and the card
     * as much in upload. But the slab is one per rank and outlives the build, the uploads reach it
     * in the order the builds were made, and a group's tables have a fixed place in it, so it
     * already holds the last build's rows. A row that still holds the same sequence at the same
     * pitch is restaged from the lowest entry the manager changed (take_table_changes) to the end
     * of what it or the card holds; any other row is restaged whole.
     *
     * `len[i]` is how many entries of row i are the table of `seq[i]`, and the rest of the row is
     * -1, which is what a whole restage leaves; -1 in `len` is "unknown". A sequence is current in
     * at most one row: its change marks are consumed where it is written, so a copy of it left
     * in another row goes unknown. A layout change -- the group's pitch -- and a build that fails
     * after taking marks make every row unknown. */
    struct BtMirror {
        int64_t pitch = -1;
        std::vector<uint64_t> seq;
        std::vector<int64_t>  len;
    };
    std::vector<BtMirror> bt_mirror_;
    /* This build's upload ranges, host-side, until build() writes them behind the payload. */
    std::vector<RadCopyRange> ranges_;
    int64_t pay_words_ = 0;   /* the payload build_groups wrote, in int32 */
    void forget_tables();
    int64_t last_n_seq_ = 0;

    int64_t               derived_bytes_ = 0;
    std::vector<Derived>  derived_;

    /* THE MEDIA STAGING, allocated only when the program declared an encoder (mm_on_).
     *
     * The encoder rows a pass overwrites its embeddings with are NOT in the slab: the slab's
     * prefix crosses whole on every build, and max_tok rows of an embedding are megabytes a decode
     * step would copy for nothing. They have a device region of their own on every rank, filled
     * only by a build that carries them -- rank 0 from pinned staging, every other rank with zeros
     * (RadBatch::mm_embd says why). The staging is refilled only once the last upload out of it
     * has run, which `mm_ev_` records. */
    bool                   mm_on_ = false;
    int64_t                n_embd_ = 0;
    std::vector<void*>     mm_dev_;
    uint16_t*              mm_host_ = nullptr;
    RadEvent               mm_ev_ = nullptr;
    bool                   mm_ev_live_ = false;
    /* ...and the encoder's inputs: patches, their coordinates and the segment boundaries, the
     * largest pass the plugin declared. One copy a rank, since each runs the tower itself. */
    int64_t                enc_patches_ = 0, enc_dim_ = 0, enc_segs_ = 0;
    std::vector<void*>     enc_pix_dev_, enc_coord_dev_, enc_cu_dev_;
    uint16_t*              enc_pix_host_ = nullptr;
    int32_t*               enc_coord_host_ = nullptr;
    int32_t*               enc_cu_host_ = nullptr;
    int64_t                last_n_tok_ = 0;
    struct MediaBytes { int64_t mm = 0, pix = 0, crd = 0, cu = 0; };
    MediaBytes media_bytes();   /* the four device regions alloc_media reserves, per rank */
    int  alloc_media();
    void free_media();
    /* The encoder rows of one entry: the media tokens among its positions, their rows copied into
     * the staging from `n_rows` on. Returns how many it added, or negative. */
    int64_t stage_mm_rows(const StepEntry& e, int64_t k, int64_t shift, int64_t n_rows,
                          int32_t* rows);

    /* The members init() derives from the program and the config, allocating nothing. */
    int  layout(Program& prog, const Config& cfg);
    /* The set's groups and its carve with no base: dev_bytes and host_bytes, nothing reserved. */
    void size_set(Set& s, const std::vector<KVGroupInfo>& groups);
    int  alloc_set(Set& s, const std::vector<KVGroupInfo>& groups);
    int free_set(Set& s);
    /* Point every h_* at host staging based at `base`, and every d_* at rank 0's slab. */
    void carve(Set& s, char* base);
    /* A span of `need` bytes of the ring, waiting for old uploads when it is full. */
    int  reserve(int64_t need, int64_t* off);
    int  run_derivations(const StepPlan& plan, Set& s);
    int  build_groups(const StepPlan& plan, const IKVManager& kv, Set& s, Out& o);
};

}  /* namespace rad */
