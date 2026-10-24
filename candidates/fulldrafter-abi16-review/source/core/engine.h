/* engine.h -- the startup sequence and the step loop. This is the file that says, in order, what
 * happens between a command line and a served token, and it is deliberately the thinnest thing in
 * the project: every step below is one component doing its own job.
 *
 * Startup, per spec §1:
 *
 *   1. Load kernel plugins from $RADIANCE_HOME/kernels/, in the configured hierarchy order.
 *   2. Load the model's metadata ONLY -- architecture id, dimensions, quantisation, vocab.
 *   3. Look up the architecture id in $RADIANCE_HOME/architectures/. No match, no model.
 *   4. DECLARE. The architecture plugin enumerates every weight, buffer and op it will ever
 *      issue. The core resolves each op against the kernel hierarchy. This phase ALWAYS
 *      completes; if anything failed to resolve, the full list is reported at the end rather
 *      than the first failure at the top.
 *   5. PLAN. The placement planner assigns every weight a tier and an execution site, against
 *      explicit pool budgets.
 *   6. LOAD. Weights are mapped and staged according to the plan.
 *   7. Serve.
 *
 * Between 4 and 5 the whole model is known statically: every op, every kernel that will service
 * it, every weight it touches, and the order. That is the property the rest of the engine spends.
 */
#pragma once
#include "rad_core.h"
#include "build/startup.h"
#include "sched/mem_iface.h"
#include "sched/scheduler.h"
#include "sample/vocab_view.h"
#include "runtime/rank_barrier.h"
#include "text/tokenizer.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "rad_spechub.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace rad {

class SSDTier;
class Kld;
struct TierPump;
struct TierJob;
struct StepSlot;

struct VramFacts;   /* mem/vram_budget.h */

class Engine {
public:
    explicit Engine(Config cfg);
    ~Engine();

    /* Steps 1..6. Returns negative and has already printed WHAT failed and WHY. A missing kernel
     * is reported as the whole list, not the first one. */
    int start();

    /* Step 7. Blocks. Returns when the server is asked to stop or a device fault ends the
     * process -- a device fault is not recoverable in-process and we exit non-zero rather than
     * serving from a wedged queue (spec §17). */
    int serve();

    /* --debug-graph and friends: everything declare and plan learned, printed, without serving.
     * This is the answer to "it is unclear what even runs", so it is a first-class mode and not
     * a side effect of a verbose flag. */
    int dump();

    const Program& program() const;
    const Config&  config()  const { return cfg_; }

private:
    /* One thread per tensor-parallel rank, each owning one device. No IPC, no shared-memory
     * handshake, no serialisation between ranks -- which is most of what a Python engine spends
     * its multiprocessing budget on. Because the ranks share an address space, the step batch is
     * shared BY POINTER rather than broadcast (spec §1). */
    struct Rank;
    std::vector<std::unique_ptr<Rank>> ranks_;
    /* THE DISK HALF OF THE IDLE TIERS. One store for the engine, not one per rank: it is
     * keyed by a chained hash of TOKENS, which every rank computes identically, and a
     * per-rank store would write the same key several times over. Off unless
     * --prefix-cache-disk-mib and a directory are both given; it holds conversation content
     * unencrypted and says so at startup.
     *
     * unique_ptr because engine.h is included by main.cpp and the tools, and SSDTier
     * drags in the O_DIRECT device layer that none of them want to compile against. */
    std::unique_ptr<SSDTier> kv_disk_;
    /* THE TIERS' WORK IN FLIGHT: the copy to host memory the transfer streams are running and the
     * disk batch the IO thread is writing. Declared after kv_disk_ so it goes first -- the thread
     * writes into that store. */
    std::unique_ptr<TierPump> pump_;
    std::vector<std::thread>           rank_threads_;

    Config cfg_;

    /* The model's source, one of the two. A container stays mapped for the life of the engine:
     * the mapped and file tiers read it where it lies (spec §4.2). A checkpoint served directly
     * stays mapped too, and is read once, at load (spec §4.3). */
    std::unique_ptr<RadFile> file_;
    std::unique_ptr<Checkpoint> ckpt_;
    std::unique_ptr<CheckpointSources> ckpt_src_;
    /* What rad_weight_encoding is answered from: the container's entries or the checkpoint's
     * tensors. */
    WeightSourceFn sources() const;

    RadModelMeta   meta_{};
    LoadedPlugins  plugins_;

    /* RadModelMeta holds bare char*, so the strings behind its free-form key/value table have to
     * outlive every declare. */
    std::vector<std::string> meta_kv_k_, meta_kv_v_;
    std::vector<const char*> meta_kv_kp_, meta_kv_vp_;

    void (*arch_step_)(RadCtx*, const RadBatch*) = nullptr;

    int load_plugins();
    int read_metadata();
    int read_checkpoint_metadata();
    int declare();
    int plan();
    /* --embedding-placement and --ngram-placement, resolved into Config before the card is split
     * (engine_bringup.cpp says how auto chooses). */
    int resolve_table_placement(const std::vector<VramFacts>& facts,
                                const std::vector<int64_t>& emb_bytes, int64_t kv_floor,
                                int64_t ngram_bytes, bool ngram_device, bool ngram_host);
    /* --ngram-placement ram: the copy of the container's [src, src + n), file offset `foff`, made
     * once and shared by every rank -- locked for the host gather, or pinned for a device kernel
     * to read over the link (`card`). */
    int copy_table(const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                   std::string* why);
    /* An auto-chosen pinned copy that could not be made: every rank's n-gram gather to the host. */
    bool ngram_gather_to_host();
    /* The copies made: the locked ones, each [base, base + bytes); the pinned ones, rad_dev_alloc
     * RAD_MEM_HOST_MAPPED ranges; and the container address each one copies. Freed in ~Engine
     * after the ranks that read them are gone. */
    std::vector<TableCopy> table_copies_;
    std::vector<void*> card_copies_;
    struct ShmCopy { void* ptr; size_t bytes; };
    std::vector<ShmCopy> shm_copies_;
    std::vector<std::pair<const void*, void*>> table_copy_of_;
    int load_weights();
    int configure_kv();
    /* One sampled row's worth of bookkeeping, so the step loop can map a logits row back to the
     * sequence that wanted it. It is NOT the identity: a prefill chunk in the middle of a long
     * prompt produces no row at all, and a speculative verify produces 1 + n_spec of them. */
    struct StepOut {
        std::vector<Request*> reqs;       /* the requests that wanted a row, in row order */
        std::vector<int32_t>  first_row;  /* [n_seq] that entry's first row, or -1 */
        /* HOW MANY OF THE STEP'S LOGITS ROWS THE SAMPLER READS: the first ones. RadBatch::n_out
         * counts the KL mode's scored rows after them too (Request::score). */
        int32_t               n_rows = 0;
        std::vector<int32_t>  rows;       /* [n_out] the device's answer, copied back */
        int32_t*              rb = nullptr;   /* the pinned readback collect_issue filled */
        std::vector<float>    lg;         /* the first row's logits, at -vv */
        std::vector<int32_t>  token;      /* [n_seq] */
        std::vector<int32_t>  accepted;   /* [n_seq] */
        std::vector<uint8_t>  eos;        /* [n_seq] */
    };

    /* The text frontend, loaded out of the container. `vocab_view_` is what the sampler's
     * grammars walk; it holds a reference to `vocab_` and must not outlive it. */
    std::shared_ptr<const Vocab> vocab_;
    std::unique_ptr<VocabView>   vocab_view_;

    /* The chunk geometry, resolved ONCE here and handed to both consumers. The prefix cache needs
     * the checkpoint interval at configure time and the scheduler needs the whole geometry at
     * init; resolving it independently in each is how the cache ends up testing the operator's raw
     * interval against positions the scheduler chose with the rounded one. */
    ChunkGeometry                 geo_;

    /* One scheduler for every rank: one decision per step, one batch, shared by pointer. */
    Scheduler                     sched_;
    std::unique_ptr<IKVManager>   kv_iface_;
    std::unique_ptr<IPrefixCache> pc_iface_;

    /* The release point. Width is tp + 1 -- the scheduler thread is on both sides of it. */
    std::unique_ptr<RankBarrier>  barrier_;
    std::atomic<const RadBatch*>  step_batch_{nullptr};
    /* WHAT A RELEASE ASKS THE RANKS TO ISSUE: the forward pass and its sampler (kIssueAll), the
     * forward pass alone (kIssueModel) -- a block drafter, or the next step issued ahead --
     * or that pass's sampler alone (kIssueSampler), once the host has committed what the sampler
     * stages from. Every release sets it: one left over from the release before tells a draft
     * pass to issue nothing but its sampler. */
    enum { kIssueAll = 0, kIssueModel = 1, kIssueSampler = 2 };
    std::atomic<int>              issue_what_{kIssueAll};
    std::atomic<int>              step_status_{RAD_OK};
    std::atomic<bool>             stopping_{false};
    /* WHY THE STEP LOOP ENDED, when it ended on an error; RAD_OK while it runs or after a clean
     * stop. Written once, by fail_loop(); read by the server watcher to take the HTTP side down
     * and by serve() to return it. */
    std::atomic<int>              loop_status_{RAD_OK};
    int  fail_loop(int st);

    /* THE SAMPLER'S HOST WORK RUNS UNDER THE MODEL, NOT IN FRONT OF IT.
     *
     * Staging a step's sampler parameters -- and with a grammar, walking a whole-vocabulary mask
     * for every position in the speculative window -- is pure host work: every upload in the chain
     * is issued by Sampler::run on the rank thread, so build_step touches no stream and nothing
     * the ranks read until they are told to sample. Run BEFORE the release it sits in front of a
     * device with nothing to do, and every microsecond of it is added to the step.
     *
     * So the ranks are released to enqueue the model first, and the scheduler stages the sampler
     * while they do. The ranks then wait here for it before enqueueing the sampler chain -- which
     * the device cannot reach until the whole forward pass has run, and that pass is the time the
     * walk hides inside. This is a one-shot gate per step, reset while the ranks are parked at the
     * release, so there is no window in which a rank can read a stale ready flag. */
    std::mutex                    build_mu_;
    std::condition_variable       build_cv_;
    bool                          build_ready_  = false;
    int                           build_status_ = RAD_OK;

    /* Scratch for the checkpoint copies: the state slots of one sequence, reused so the step path
     * does not allocate. See state_slots_of. */
    std::vector<int32_t>          ckpt_slots_;
    bool state_slots_of(uint64_t seq, std::vector<int32_t>* out) const;

    int  load_vocab();
    int  declare_sampler(int rank, RadBuilder* b, int64_t logits_width, bool sizing = false);
    /* The build context every declare of rank `rank` gets, at `max_tok` -- the real one and the
     * activation arena's sizing declares differ in nothing else. */
    RadBuildCtx build_ctx(int rank, int64_t max_tok) const;
    void probe_arena_levels();
    int  step_loop();
    /* The step's sampled ids on their way to the host, and then the wait for them and what they
     * mean. The copy and trunk_end_ precede draft passes, so acceptance and commit can overlap
     * them without waiting for step_end_. */
    int  collect_issue(const RadBatch* b, StepOut& so);
    int  collect_step(const RadBatch* b, StepOut& so);
    /* The sampled-token readback completes here, before any drafter work. collect_step waits
     * only for this event; it must not wait for the proposal readback at step_end_. */
    int  mark_trunk_end(const RadBatch* b, StepOut& so);
    /* Fence draft passes and their readbacks before publication. Never issue a sampled-token
     * copy here: post-commit callers have consumed it and collect_step has cleared so.rb. */
    int  mark_step_end();
    RadEvent trunk_end_ = nullptr;
    RadEvent step_end_ = nullptr;

    /* THE DRAFT HEAD, exactly as the plugin declared it (rad_declare_drafter), and zeroed when
     * the deployment asked for no speculative window. `kind` selects which of the two drivers
     * below runs; nothing else in the engine knows what model this is.
     *
     * `drafter_name_` owns the string, because RadDrafterDecl::name points into the Program. */
    RadDrafterDecl       drafter_{};
    std::string          drafter_name_;

    /* THE SERIAL DRIVER (spec §10): `depth` dependent passes after the step commits, each
     * embedding what the last one proposed. MTP, EAGLE and Medusa-serial are all this shape. */
    std::vector<int32_t> draft_out_;   /* [rows][depth], what each pass proposed */
    /* Serial tree producers leave root-inclusive packed rows on device. The driver reads them
     * once after the last round; no candidate planes or topology construction cross to the host. */
    bool                 serial_tree_ = false;
    bool                 serial_pruned_ = false;
    int                  pruned_tile_ = 0;

    /* Pinned readback keeps copies asynchronous until the step's event wait.
     * Serial packed trees reserve their maximum extent at initialization. */
    struct Readback {
        int32_t* p = nullptr;
        int64_t  n = 0;
        int32_t* reserve(int64_t count);
        ~Readback();
    };
    Readback             readback_;
    Readback             sample_readback_; /* trunk ids must survive the preissued block readback */
    bool serial_drafting() const;
    int  advance_all(int mode, int round, bool from_proposal, int rows = -1);
    int  issue_serial_drafter(int step, const RadBatch* b, StepOut& so);
    /* The next step's forward pass, issued before this one is read back; null when the scheduler
     * wants this step committed first. See Scheduler::step_next. */
    int  issue_ahead(const RadBatch** out);
    /* The arena level and the followers' pools, before any release that runs a forward pass. */
    int  prepare_release(const RadBatch* b);
    /* Whether a step may be issued ahead at all: the drafter, if there is one, runs on the card
     * (a block drafter and the prompt-lookup drafter both read the committed sequence). */
    bool pipelining_ = false;
    int  finish_serial_drafter(const StepOut& so);
    /* What issue_serial_drafter left for finish_serial_drafter: the rows that draft, the rounds
     * asked for, and the rounds that ran. */
    std::vector<Scheduler::SerialRow> serial_row_;
    int  serial_rounds_ = 0, serial_rows_ = 0, serial_done_ = 0;
    /* The StepSlots and end-of-generation ids on every rank, and the pinned host copy of rank 0's
     * slots the step reads back (sched/advance.h). */
    int  step_state_init();
    void step_state_fini();
    StepSlot*            slots_host_ = nullptr;
    /* THE DLoop CONTEXT SPAN'S BUFFER HANDLE, found once at startup by name and zero when the
     * deployed backend declares none. The ring drafter has no persistent cache and so has no span;
     * a full-context one cannot append without it. Kept here rather than re-searched per step,
     * because the step path allocates nothing and looks nothing up. */
    rad_buf              span_buf_ = 0;

    /* STEP-TIME ATTRIBUTION, Debug only (see the barrier in step_loop): which of the scheduler's
     * own jobs the time went to. A two-way host/ranks split says which side of the barrier a step
     * is slow on and nothing more: it does not separate the plan from the grammar from the staging,
     * and the whole region after the join -- the readback, the commit, the drafter -- falls in
     * neither bucket, leaving most of the step unattributed. These nine sum to the step. */
    enum Ph { PH_PLAN, PH_BEGIN, PH_BUILD, PH_PRE, PH_RANKS, PH_SAVE, PH_COLLECT, PH_COMMIT,
              PH_DRAFT, PH_N };
    double phase_us_[PH_N] = {};
    int    phase_n_ = 0;
    std::chrono::steady_clock::time_point phase_t_{};
    void   phase_mark(int p);

    /* THE BLOCK DRIVER (spec §10): a context pass that mirrors the step just run, then ONE pass
     * that fills every drafted position at once. DFlash2 is this shape. */
    rad_buf              raw_buf_ = 0;  /* RADIANCE_DRAFT_RAW's bisect; see run_block_drafter */
    int  run_block_drafter(int step);
    int  issue_block_drafter(int step, const RadBatch* b, StepOut& so);
    int  finish_block_drafter(const StepOut& so);
    int  run_block_pass(const RadBatch* b, int step, const char* what);
    int  publish_block_drafts(bool committed);
    std::vector<Scheduler::BlockRow> block_rows_;
    bool block_preissued_ = false;
    int  block_rows_count_ = 0;
    std::vector<int32_t> block_tree_out_;  /* declared root-inclusive packed tree rows */
    std::vector<uint64_t> tree_chain_notice_; /* sequence last logged at each scheduler card */
    int64_t tree_exact_to_ = 0;
    bool spechub_ = false;
    rad_buf spechub_draw_buf_ = 0, spechub_state_buf_ = 0;
    RadSpechubDraw* spechub_draw_stage_ = nullptr;
    RadSpechubState* spechub_readback_ = nullptr;
    std::vector<RadSpechubDraw> spechub_by_card_;
    /* RADIANCE_MTP_COUPLED (engine_draft.cpp): draws staged per drafting row before the serial
     * passes, the [rounds, rows] state plane read back beside the slots, bound after commit. */
    bool mtp_couple_ = false;
    rad_buf couple_draw_buf_ = 0, couple_state_buf_ = 0;
    int64_t couple_rows_ = 0;                         /* the state plane's rows per round */
    RadMtpCoupleDraw* couple_draw_stage_ = nullptr;   /* pinned [max_seqs] */
    RadMtpCoupleState* couple_readback_ = nullptr;    /* pinned [rounds, couple_rows_] */
    std::vector<RadMtpCoupleDraw> couple_by_card_;
    RadMtpCoupleState couple_bound_[RAD_MTP_COUPLE_MAX_ROUNDS] = {};
    int  stage_couple_draws(int n);
    int  bind_couple(int row, int done, const StepSlot& S);
    /* THE ENCODER DRIVER (spec §11): every media item the scheduler is waiting on, encoded in
     * passes of whole segments and read back into the item, between steps and never inside one.
     * core/engine_mm.cpp. */
    int  run_encodes();
    int32_t enc_step_ = 0;
    void drain_completed();
    /* The terminal ids drain_completed() took, reused so a steady state allocates nothing. */
    std::vector<uint64_t> done_;
    /* THE TIERS' SIDE OF THE LOOP, every iteration: collect a copy that has landed, hand the next
     * one out, collect and hand out the disk work. Nothing in it waits on the device or on a file,
     * and the scheduler's lock is held only around the bookkeeping. */
    void tier_tick();
    bool submit_fetch(uint64_t tag, std::vector<TierJob>&& jobs);
    uint64_t tier_last_ns_ = 0;
    /* THE LOAN: what the KV cache is not using, lent to every rank's expert slab at the addresses
     * the cache already owns (KVManager, the loan block). Here rather than inside the scheduler
     * because it spans both sides -- rank 0's cache decides how much, and every rank's mover
     * fills or empties its slots -- and because a mover may only be driven while its rank thread
     * is parked, which this thread is the one to know. `idle` also runs each mover's own fill,
     * which a busy engine gets from the step. */
    void loan_tick(bool idle);
    bool     lending_ = false;
    uint64_t loan_last_ns_ = 0;
    void on_prefix_hit(const Request& r, const std::vector<int32_t>& units);
    /* THE ACTIVATION ARENA'S LOAN (Engine::arena_step). `arena_rows_[k]` is the largest step Ctx
     * level k serves -- level 0 is the full plan -- and empty when the arena lends nothing. The
     * floor is the largest step seen lately: the loan grows past what it allows only once no step
     * that large has run for a while, so a turn of prefill chunks does not refill and recall the
     * slab between every chunk. */
    int  arena_step(const RadBatch* b);
    void arena_tick(uint64_t now_ns);
    std::vector<int64_t> arena_rows_;
    int64_t  arena_floor_ = 0;
    int64_t  arena_last_ = 0;
    uint64_t arena_floor_ns_ = 0;
    /* WHETHER THIS DRIVER RETURNS DECOMMITTED VRAM TO THE CARD, for the checkpoint slots -- the
     * one part of the KV pool still backed on demand. Measured once at the end of bringup; it is
     * not something the API promises. */
    bool     release_frees_card_ = true;
    /* WHAT THE VRAM BUDGET KEPT FOR ALLOCATIONS MADE AFTER start(), on top of --gpu-headroom-mib:
     * the step batch's staging, which serve() allocates, and what is built on first use, if
     * ever -- the KV tiers' copy staging and the second compute lane's queue. The checkpoint
     * slots backed at the end of bringup leave both free, and the bringup's closing reading is
     * checked against both. */
    int64_t  late_step_bytes_ = 0;
    int64_t  late_lazy_bytes_ = 0;
    /* Every rank's checkpoint floor: a slot is backed only while the card keeps `keep` free. */
    void set_checkpoint_floors(int64_t keep);
    int  run_server();
    /* THE KL MODE (core/kld.h), in place of run_server: the corpus's documents submitted as
     * scoring requests, and every step's scored rows copied to the host and handed to `kld_`.
     * One pinned block a rank, a step's tokens of this rank's logits columns. */
    int  run_kld();
    int  run_diagnose();   /* --diagnose: core/engine_diagnose.cpp */
    int  score_rows(const RadBatch* b, const StepOut& so);
    std::unique_ptr<Kld> kld_;
    std::vector<float*>  kld_host_;
    std::vector<int64_t> kld_vocab0_, kld_width_;   /* per rank read: its logits columns */
    /* --kld-taps: rank 0's tap buffers (the stream is replicated, so one rank holds all of it),
     * a pinned block each for a step's rows, and their common row width. */
    std::vector<rad_buf>   kld_tap_buf_;
    std::vector<uint16_t*> kld_tap_host_;
    int64_t                kld_tap_pw_ = 0;
    rad_buf                kld_anchor_buf_ = RAD_NULL_HANDLE;
    uint16_t*              kld_anchor_host_ = nullptr;
    int64_t                kld_anchor_w_ = 0;
    /* The engine half of /server_info, as a JSON object. See the definition in engine.cpp. */
    std::string engine_info() const;

    /* One dispatch of the tiering seam: retire what landed, credit last step's routing and issue
     * what the heat engine proposes. A member because Rank is private, and a no-op on a static
     * plan. See the long note at the definition for why a step is the unit. */
    void tiering_dispatch(Rank& r, const RadBatch* b);
    void rank_loop(int rank);
};

}  /* namespace rad */
