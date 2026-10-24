/* batch.h -- the step batch, built once per step by the core and shared by every layer and every
 * KV group (spec §8). Unlike vLLM, which rebuilds the metadata per KV-cache group, there is one
 * build and every consumer reads the same buffers.
 *
 * Everything comes from pools sized at max_tok / max_seqs at init; nothing on this path
 * allocates. Staging is pinned host memory, uploaded in one async copy per rank, carved from a
 * ring so that a build never overwrites staging an earlier upload is still reading (that would
 * tear a block table) and the host can stage several passes ahead of the card.
 */
#pragma once
#include "advance.h"
#include "device/device.h"
#include "geometry.h"
#include "mem_iface.h"
#include "metrics.h"

namespace rad {

/* One sequence's participation in one step. The scheduler fills these; the builder turns them
 * into device arrays. */
struct StepEntry {
    const Request* req = nullptr;
    uint64_t seq = 0;             /* the KV manager's key (the request id) */
    int32_t slot = -1;            /* the scheduler's slot for the request */
    int32_t card = -1;            /* the card's StepSlot index, [0, max_seqs): the sequence
                                   * identity in this batch (SchedReq::card) */
    int32_t n_tokens = 0;         /* query length this step */
    int32_t ctx_len = 0;          /* tokens computed before this step: the first token's position */
    int32_t n_spec = 0;           /* draft tokens included in n_tokens (decode only) */
    /* Tokens the previous step committed, one-based: its input token plus its accepted drafts.
     * gdn_conv_update reads its rolling window at `num_accepted - 1` and gdn_recurrent_update
     * takes the state the (num_accepted)th token left, so 1 is a plain decode step (also their
     * default when the operand is absent) and 0 is invalid. SchedReq::n_accepted counts drafts
     * only, one less. */
    int32_t n_accepted = 1;
    int32_t conv_cursor = 0;      /* read offset inside the rolling conv window (§10) */
    int32_t ckpt_tok = -1;        /* token index WITHIN this entry to snapshot at, or -1 */
    int32_t ckpt_slot = -1;
    /* Snapshot this entry resumes from, or -1. Set only on the first step after a prefix hit; the
     * adopted blocks cover attention only, so this is the sole carrier of the recurrent state. */
    int32_t ckpt_restore = -1;
    const int32_t* draft = nullptr;   /* the drafter's proposal, appended after the committed tail */
    int32_t n_draft = 0;
    /* First logits row, written into `out_ids`, that this entry reads, or -1 for the default
     * rule. The MTP history pass uses it to gather the 1 + n_spec trunk rows of a verifying
     * sequence; the DFlash2 query pass uses it to name the mask rows of its own block. Which
     * buffer out_ids indexes is up to the architecture block. */
    int32_t hidden_row = -1;
    /* Rows from `hidden_row`, or 0 for one per token. The MTP history pass uses 0; the DFlash2
     * query pass runs `block` tokens and uses block-1, starting one row in. */
    int32_t hidden_n = 0;
    /* This entry's last row is where draft round 1 continues from (RadBatch::draft_out_ids).
     * Set on a history-pass entry whose sequence drafts. */
    bool    draft_out = false;
    bool    is_prefill = false;
    bool    produces_token = false;   /* this entry's last token is the sequence's last */
    /* The drafts came from prompt lookup, not the declared drafter. They count toward reported
     * acceptance but not the adaptive window's estimate of the draft head (DraftController,
     * draft.h). */
    bool    ngram = false;
    /* Device row: the card overwrites its tokens, positions, rollback count and slots from the
     * StepSlot before the pass runs (advance.h); `ctx_len` is then only an upper bound for the
     * host-side ceilings. dev_tok/dev_out/dev_aux are advance.h's row descriptor: the verify
     * step's token row of the first verified position, the first sampled row (-1 for none), and
     * the mode's own number. A verify step's entries always carry them. */
    bool    dev = false;
    int32_t dev_tok = 0;
    int32_t dev_out = -1;
    int32_t dev_aux = 0;
};

struct StepPlan {
    std::vector<StepEntry> e;
    int64_t n_tok = 0;
    int     phase = RAD_PHASE_DECODE;
    /* Passed to RadBatch::draft_pass. 0 is a trunk step, r > 0 runs only the draft head, r < 0
     * is a history pass. */
    int     draft_pass = 0;
    /* How far past its position a row reads its token. The MTP head pairs hidden state i with
     * the token at i + 1 (rad_block_mtp_fp8.h), so its passes use 1. Do not derive this from
     * `draft_pass`: DFlash2's query block embeds the token at its own position and uses 0. */
    int     tok_shift = 0;
    int     n_spec = 0;               /* the batch-wide speculative window */
    bool    spec_is_tree = false;     /* a tree drafter fills spec_parent; a chain leaves it null */
    /* Decode/prefill split, for a block that issues each half separately. Entries are sorted so
     * decode rows are sequences [0, n_seq_decode) and prefill chunks follow. */
    int64_t n_seq_decode = 0;
    int64_t n_tok_decode = 0;
    int32_t max_q_len = 0;
    int32_t max_ctx_len = 0;

    /* Whether the entries agree with the split: sequences [0, n_seq_decode) are the decode rows
     * and the rest prefill chunks.
     *
     * This is the ordering that makes a mixed step's recurrent state correct rather than merely
     * concurrent. Every block that treats the two kinds differently issues over each half as its
     * own contiguous run (RadBatch::n_seq_decode), and the half a sequence lands in is what writes
     * its state: the recurrent path, which rolls a rejected draft back, or the chunked scan, which
     * has no rollback operand at all. A decode row left in the prefill half keeps its rejected
     * drafts folded into the state and the conv window and verifies one token instead of several;
     * a chunk moved into the decode half is stepped as a single token. Neither shows in the
     * batch's shape afterwards.
     *
     * Scheduler::step sorts every plan this way (its `part_` partition). This is the same rule
     * read back, which is what BatchBuilder::build checks before it turns a plan into device
     * arrays. Only a MIXED plan carries a split at all: for a pure plan the phase decides which
     * half every sequence is in (rad::batch_split), so a draft pass -- all decode rows, and
     * n_seq_decode left zero by reset() -- answers true.
     *
     * `bad`, when given, gets the first entry that disagrees with the split, which is what the
     * builder's refusal names. */
    bool split_ordered(int64_t* bad = nullptr) const {
        if (phase != RAD_PHASE_MIXED) return true;
        for (size_t i = 0; i < e.size(); ++i)
            if ((int64_t)i < n_seq_decode ? e[i].is_prefill : !e[i].is_prefill) {
                if (bad) *bad = (int64_t)i;
                return false;
            }
        return true;
    }

    void reset() {
        e.clear();
        n_tok = 0; phase = RAD_PHASE_DECODE; draft_pass = 0; tok_shift = 0; n_spec = 0;
        spec_is_tree = false; max_q_len = 0; max_ctx_len = 0; n_seq_decode = 0; n_tok_decode = 0;
    }
};

/* ------------------------------------------------------------------ the builder */
class BatchBuilder {
public:
    /* One rank: the card its slab lives on, the stream its upload runs on, and the Program whose
     * arena derived buffers are copied into. */
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
        Program*  prog   = nullptr;
    };

    /* `prog` is non-const because derived buffers are written into its arena, whose pointer lands
     * in BufferInfo::ptr after the buffer plan runs. */
    int  init(Program& prog, const Config& cfg, const ChunkGeometry& geo,
              const KVGeom& kvg, const std::vector<RankIO>& ranks);
    void fini();

    /* Bytes init() allocates on each rank's card (the step slab and, with an encoder, the media
     * regions), computed without allocating, for the VRAM budget. -1 if init() would refuse. */
    static int64_t device_bytes(Program& prog, const Config& cfg, const KVGeom& kvg);

    /* Build the batch for this step. The returned pointer stays valid for the next kOut - 1
     * calls and is shared by pointer across rank threads (spec §1). Null on a device failure,
     * which is fatal for the step. */
    const RadBatch* build(const StepPlan& plan, const IKVManager& kv, int step);

    /* The last build's batch for one rank: build()'s result with that rank's slab pointers. */
    const RadBatch* batch_for(int rank) const;

    /* The last build's slab and offsets on one rank, for the advance. The caller supplies mode,
     * round, slot state and inputs (advance.h). */
    AdvanceArgs advance_args(int rank) const;

    /* Longest sequence a batch can describe, in tokens, as sized into the block-table rows at
     * init. A position at or past it cannot be addressed. */
    int64_t max_ctx() const { return max_ctx_; }

    /* An encoder pass (RadBatch::enc): patches of whole segments of one or more media items,
     * staged into every rank's encoder input, since each rank runs the tower itself. Returns rank
     * 0's batch like build(); batch_for() serves the rest. The caller must synchronise every
     * rank's stream before the next encoder build. */
    struct EncoderPart {
        const mm::Item* item = nullptr;
        int32_t         seg0 = 0, n_seg = 0;
    };
    /* `enc` is the encoder the pass runs: its patch width, pass size and merge bound the parts;
     * its modalities become RadBatch::enc. */
    const RadBatch* build_encoder(const std::vector<EncoderPart>& parts, int step,
                                  const RadEncoderDecl& enc);

private:
    struct Group {
        rad_kvgroup g = 0;
        int64_t     block_size = 0;
        int64_t     max_blocks = 0;
        int32_t*    h_slot = nullptr;  int32_t* d_slot = nullptr;
        /* The group's block tables on the card: max_seqs rows of this build's pitch, `bt_fix`
         * int32 into the table region. No host twin; a build stages only what changed. */
        int64_t     bt_fix = 0;
        int32_t*    d_bt   = nullptr;
        int32_t*    h_used = nullptr;  int32_t* d_used = nullptr;
        int32_t*    h_sidx = nullptr;  int32_t* d_sidx = nullptr;
        int64_t     sidx_pitch = 1;    /* columns per sequence in h_sidx/d_sidx */
    };

    /* Layout of a build: every array at the same offset on the card and in host staging, so one
     * copy moves a whole build.
     *
     * The device slab is one per rank and never double buffered: uploads and the kernels reading
     * them share the rank's compute stream, so stream order protects it. The host side is read
     * when the stream reaches the upload, possibly several passes later, so it comes from a ring
     * (see Ring). `h_*` are re-carved for every build at its ring position; `d_*` are rank 0's
     * slab and never move. */
    struct Set {
        int64_t host_bytes = 0, dev_bytes = 0;
        /* Everything below `bt_base` is uploaded whole every build. Past it, the card holds every
         * group's block tables at fixed places, while the host span holds the payload (changed
         * table entries, packed) followed by the ranges saying where each run lands (see
         * BtMirror). `dev_bytes` is the allocation, sized for every group's largest table. */
        int32_t* bt_h0 = nullptr;  int32_t* bt_d0 = nullptr;
        int64_t  bt_base = 0;
        /* One device slab per rank, required for correctness: a single slab on rank 0 would leave
         * other ranks' reads unordered against rank 0's copies. Per rank, each upload is a
         * same-stream dependency and the metadata sits on the card that reads it. The carve is
         * identical on every rank, so d_* below are rank 0's pointers and the offset basis for
         * the rest. */
        std::vector<void*> dev;

        int32_t *h_tok = nullptr, *h_pos = nullptr, *h_cu = nullptr, *h_sid = nullptr;
        int32_t *h_q = nullptr, *h_ctx = nullptr, *h_acc = nullptr, *h_parent = nullptr;
        int32_t *h_cktok = nullptr, *h_ckslot = nullptr;
        int32_t *h_out = nullptr;      /* [n_out] which of this step's tokens want logits */
        int32_t *h_dout = nullptr;     /* [n_draft_out] a history pass's round-1 rows */
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
        /* [n_tok] each row's runtime LoRA adapter slot (RadBatch::lora_slot): carved only when the
         * engine serves adapters, empty otherwise. */
        int32_t   *h_lora = nullptr, *d_lora = nullptr;
        /* [n_kv_groups][max_seqs]: every conv group's state slot per sequence (see init()). */
        int32_t*   h_convslot = nullptr;

        std::vector<Group>           groups;
    };

    /* Host staging ring. A build takes one contiguous span for what it uploads (the fixed prefix
     * and its block-table payload), fills it, and records an event per rank behind the upload. A
     * span is reusable once every rank's upload has run; `reserve` waits for that only when the
     * ring is full. */
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

    /* A build's ABI views and batch, one per rank (they differ only in device pointers). kOut of
     * them, because a step's batch is still read after its draft passes and the next step have
     * been built. */
    static constexpr int kOut = RAD_SCHED_MAX_SPEC + 4;
    struct Out {
        std::vector<std::vector<RadKVGroupBatch>> kv;
        std::vector<RadBatch>                     batch;
    };

    struct Derived {
        rad_buf     handle = 0;
        /* Conv group this derivation is about, or -1. From RadBufDecl::kv, or the only conv
         * group when the plugin left it 0 and there is exactly one. */
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
    /* Per KV group, 0 for a non-conv group. This and `h_convslot` only feed DeriveInput for the
     * `conv_state_index` derivation, which no plugin calls at present; conv kernels read
     * RadKVGroupBatch::state_index instead. Kept per group so the derivation is correct with more
     * than one conv group. */
    std::vector<int32_t> conv_window_;
    int32_t conv_group_ = -1;    /* the conv group when there is exactly one, else -1 */
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

    /* What every rank's slab holds in its block-table region, so a build stages only what changed.
     * At long context a table is thousands of entries per group, too many to restage every build.
     * The slab outlives the build and uploads land in build order, so it holds the last build's
     * rows. A row still holding the same sequence at the same pitch is restaged from the first
     * entry the manager changed (take_table_changes) to the end of what it or the card holds; any
     * other row is restaged whole.
     *
     * `len[i]` is how many entries of row i are the table of `seq[i]`; the rest of the row is -1.
     * -1 in `len` means unknown. A sequence is current in at most one row, because its change marks
     * are consumed where it is written; a copy left in another row becomes unknown. A pitch change,
     * or a build that fails after taking marks, makes every row unknown. */
    struct BtMirror {
        int64_t pitch = -1;
        std::vector<uint64_t> seq;
        std::vector<int64_t>  len;
    };
    std::vector<BtMirror> bt_mirror_;
    /* This build's upload ranges, host-side, until build() writes them behind the payload. */
    std::vector<RadCopyRange> ranges_;
    int64_t pay_words_ = 0;   /* payload build_groups wrote, in int32 words */
    void forget_tables();
    int64_t last_n_seq_ = 0;

    int64_t               derived_bytes_ = 0;
    std::vector<Derived>  derived_;

    /* Media staging, allocated only when the program declares an encoder (mm_on_).
     *
     * Encoder rows are kept out of the slab, whose prefix is uploaded every build; max_tok
     * embedding rows would be megabytes per decode step. Each rank has its own device region,
     * filled only by a build that carries rows: rank 0 from pinned staging, the others with zeros
     * (see RadBatch::mm_embd). The staging is refilled only after its last upload has run
     * (`mm_ev_`). */
    bool                   mm_on_ = false;
    /* The engine serves runtime LoRA adapters (Config::lora): the batch carries each row's slot. */
    bool                   lora_on_ = false;
    int64_t                n_embd_ = 0;
    std::vector<void*>     mm_dev_;
    uint16_t*              mm_host_ = nullptr;
    RadEvent               mm_ev_ = nullptr;
    bool                   mm_ev_live_ = false;
    /* Encoder inputs: patches, their coordinates and segment boundaries, sized for the largest
     * declared pass (encoder passes run one at a time and share it). One copy per rank. */
    int64_t                enc_patches_ = 0, enc_pix_elems_ = 0, enc_segs_ = 0;
    std::vector<void*>     enc_pix_dev_, enc_coord_dev_, enc_cu_dev_;
    uint16_t*              enc_pix_host_ = nullptr;
    int32_t*               enc_coord_host_ = nullptr;
    int32_t*               enc_cu_host_ = nullptr;
    std::vector<int32_t>   enc_coord_tmp_;   /* one item's coordinates; kept for its capacity */
    int64_t                last_n_tok_ = 0;
    struct MediaBytes { int64_t mm = 0, pix = 0, crd = 0, cu = 0; };
    MediaBytes media_bytes();   /* the four per-rank device regions alloc_media reserves */
    int  alloc_media();
    void free_media();
    /* Stage the encoder rows of one entry's media tokens, starting at row `n_rows`. Returns how
     * many it added, or negative on error. */
    int64_t stage_mm_rows(const StepEntry& e, int64_t k, int64_t shift, int64_t n_rows,
                          int32_t* rows);

    /* Derive the members init() needs from the program and config, without allocating. */
    int  layout(Program& prog, const Config& cfg);
    /* Fill the set's groups and sizes (dev_bytes, host_bytes) without reserving anything. */
    void size_set(Set& s, const std::vector<KVGroupInfo>& groups);
    int  alloc_set(Set& s, const std::vector<KVGroupInfo>& groups);
    void free_set(Set& s);
    /* Point every h_* at host staging based at `base`, and every d_* at rank 0's slab. */
    void carve(Set& s, char* base);
    /* Reserve `need` bytes of the ring, waiting for old uploads when it is full. */
    int  reserve(int64_t need, int64_t* off);
    int  run_derivations(const StepPlan& plan, Set& s);
    int  build_groups(const StepPlan& plan, const IKVManager& kv, Set& s, Out& o);
};

}  /* namespace rad */
