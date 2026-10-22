/* engine.h -- the startup sequence and the step loop: what happens, in order, between a command
 * line and a served token. Kept thin; each step is one component doing its own job.
 *
 * Startup, per spec §1:
 *
 *   1. Load kernel plugins from $RADIANCE_HOME/kernels/, in the configured hierarchy order.
 *   2. Load the model's metadata only: architecture id, dimensions, quantisation, vocab.
 *   3. Look up the architecture id in $RADIANCE_HOME/architectures/. No match, no model.
 *   4. Declare. The architecture plugin enumerates every weight, buffer and op it will ever
 *      issue, and the core resolves each op against the kernel hierarchy. This phase always
 *      completes and reports every resolution failure, not just the first.
 *   5. Plan. The placement planner assigns every weight a tier and an execution site against
 *      explicit pool budgets.
 *   6. Load. Weights are mapped and staged according to the plan.
 *   7. Serve.
 *
 * After step 4 the whole model is known statically: every op, its kernel, the weights it touches,
 * and the order. The rest of the engine relies on that.
 */
#pragma once
#include "rad_core.h"
#include "cvec.h"
#include <map>
#include <set>
#include "build/startup.h"
#include "diag/diag.h"
#include "mem/hostarena.h"
#include "sched/mem_iface.h"
#include "sched/scheduler.h"
#include "sample/vocab_view.h"
#include "runtime/rank_barrier.h"
#include "text/tokenizer.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "lora/adapter.h"
#include "lora/slots.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace rad {

class Kld;
class StallWatch;
namespace gen { class Driver; class AudioDriver; class VideoDriver; struct IPassDriver; }
struct TierPump;
struct TierJob;
struct StepSlot;

struct VramFacts;   /* mem/vram_budget.h */
struct WeightFootprint;   /* place/planner.h */

class Engine {
public:
    explicit Engine(Config cfg);
    ~Engine();

    /* Steps 1..6. Returns negative after printing what failed and why; missing kernels are
     * reported as a full list. */
    int start();

    /* Step 7. Blocks until the server is asked to stop. A device fault is not recoverable
     * in-process, so it exits non-zero rather than serving from a wedged queue (spec §17).
     * `watch`, when given, is the startup's stall watch (util/stallwatch.h): held until the
     * server listens, or the KL mode starts scoring, so the stretch from start() to there
     * reports where it waits like start() itself. */
    int serve(std::unique_ptr<StallWatch> watch);

    /* --debug-graph and friends: print everything declare and plan learned, without serving. A
     * first-class mode rather than a verbose-flag side effect, so what runs is always visible. */
    int dump();

    const Program& program() const;
    const Config&  config()  const { return cfg_; }

    /* The startup report (core/diag/diag.h): every option with its value and source, the
     * machine, and what on it costs speed or correctness. serve() prints it with RAD_OK before
     * listening; main prints it with the error start() or serve() failed with, so a server that
     * never came up still logs its setup. The report then repeats the error on an E line, so the
     * log's tail shows it. */
    void print_startup_report(int status);
    /* The report as JSON, with the hardware re-read, since link and clock state only mean
     * something under load (/diagnostics). The reading is reused for a second and covers only the
     * serving cards' power-management files (hardware_facts). */
    std::string diagnostics_json() const;
    bool startup_report_printed() const { return report_printed_; }

    /* ---- the control vector's public surface (core/cvec.h).
     *
     * `cvec_loaded` is whether a vector was found and validated for this container; a request
     * asking for the feature on a deployment without one is refused by name rather than silently
     * served unprojected. `cvec_default_on` is the runtime default for requests that name no
     * choice, and the setter is what POST /admin/control-vector calls: it flips a flag, builds
     * nothing, reloads nothing, and leaves every running request alone. */
    bool cvec_loaded() const { return cvec_.loaded(); }
    bool cvec_default_on() const { return cvec_default_.load(); }
    void set_cvec_default(bool on) { cvec_default_.store(on); }
    /* {available, default, mode, first, last, n_embd, n_layers, scale, file} for /stats, the
     * admin route and the dashboard. */
    std::string cvec_json() const;
    /* The value the block cache is keyed by for a step with this mode: the vector's identity when
     * on, 0 when off. Zero with no vector loaded, so a deployment without the feature hashes
     * exactly as before. */
    uint64_t cvec_salt_for(bool on) const { return on ? cvec_salt_ : 0; }

private:
    /* One thread per tensor-parallel rank, each owning one device. Ranks share the address
     * space, so there is no IPC or serialisation and the step batch is shared by pointer rather
     * than broadcast (spec §1). */
    struct Rank;
    std::vector<std::unique_ptr<Rank>> ranks_;
    /* The tiers' work in flight: the host-memory copies the transfer streams are running. */
    std::unique_ptr<TierPump> pump_;
    /* serve()'s startup stall watch, released when the server listens (run_server) or the KL
     * mode starts (run_kld). */
    std::unique_ptr<StallWatch> startup_watch_;
    /* rad_mono_ns at the top of start(), so serve() can report how long bringup took. The same
     * span the startup watch covers (main.cpp creates it here and ends it at the listen). */
    uint64_t              startup_begin_ns_ = 0;
    std::vector<std::thread>           rank_threads_;

    Config cfg_;
    /* The VRAM split each rank actually takes, filled by Engine::plan. Equal to cfg_ on every
     * rank whose cache costs as much per token as the dearest rank's. A cheaper rank (an
     * attention-zero rank) gets a KV pool sized for the common token count and the difference in
     * its weights pool. Pools::configure and the planner must read the rank's entry, never cfg_;
     * plan() refuses a rank whose plan and weights pool disagree. */
    std::vector<Config> rank_cfg_;
    int64_t kv_planned_tokens_ = 0;   /* the common token count plan() sized the re-split pools for */

    /* The model's source, one of the two. A container stays mapped for the engine's life because
     * the mapped and file tiers read it in place (spec §4.3). A checkpoint served directly also
     * stays mapped and is read once, at load (spec §4.3). */
    std::unique_ptr<RadFile> file_;
    std::unique_ptr<Checkpoint> ckpt_;
    std::unique_ptr<CheckpointSources> ckpt_src_;
    /* The source rad_weight_encoding answers from: container entries or checkpoint tensors. */
    WeightSourceFn sources() const;

    RadModelMeta   meta_{};
    LoadedPlugins  plugins_;

    /* RadModelMeta holds bare char*, so the strings behind its key/value table must outlive
     * every declare. */
    std::vector<std::string> meta_kv_k_, meta_kv_v_;
    std::vector<const char*> meta_kv_kp_, meta_kv_vp_;

    void (*arch_step_)(RadCtx*, const RadBatch*) = nullptr;

    int load_plugins();
    int read_metadata();
    int read_checkpoint_metadata();
    /* --rope-scaling over the free-form keys, then the startup line and the context checks;
     * runs on meta_kv_k_/meta_kv_v_ before their pointer arrays are built. */
    int apply_rope_scaling();
    int declare();
    int plan();
    /* Runtime LoRA (core/engine_lora.cpp). Targets are fixed before declare; adapters are loaded
     * into slots after every rank's Ctx is bound, and a request's `model` names one. */
    int lora_targets();
    int lora_start();
public:
    int lora_load(const std::string& name, const std::string& dir, std::string* err);
    /* Unpublish an adapter (lora::SlotTable::unload). Idle, its slot is freed at once; in use, it
     * drains: *in_flight says how many requests still hold it, no new request may name it, and the
     * last of them frees the slot, which cools before reuse (lora/slots.h). */
    int lora_unload(const std::string& name, std::string* err, int* in_flight);
    /* RAD_OK with the slot, the uid of its load and a hold: the adapter is not freed while the
     * hold lives (the server keeps it in the request's Generation). RAD_E_NOTFOUND for a name no
     * slot holds, or one that is draining. */
    int lora_acquire(const std::string& name, int32_t* slot, uint64_t* uid,
                     std::shared_ptr<void>* hold);
    std::vector<std::string> lora_names() const;
    /* A generator's adapters: a request names one or more adapter files at their strengths, and
     * gets the slot holding exactly that combination, composed into a free (or the least recently
     * used idle) slot when none holds it yet. `note` says what the composition left out or
     * approximated, for the reply. RAD_E_NOTFOUND names an adapter neither --lora nor --lora-dir
     * provides; RAD_E_FULL means every slot is held by running requests. */
    struct LoraUse { std::string name; double weight = 1.0; };
    int gen_lora_acquire(const std::vector<LoraUse>& uses, int32_t* slot,
                         std::shared_ptr<void>* hold, std::string* note, std::string* err);
    /* The adapter files a generator request may name: --lora's and those under --lora-dir. */
    std::vector<std::string> gen_lora_names() const;
    bool gen_lora_enabled() const { return lora_gen_; }
private:
    /* A generator's adapter files by name (--lora, POST /v1/load_lora_adapter, --lora-dir), opened
     * once and kept mapped; compositions read them when a combination is put in a slot. */
    int gen_lora_open(const std::string& name, std::shared_ptr<const lora::AdapterFile>* out,
                      std::string* err);
    int gen_lora_start();
    /* Every rank's packed bytes into `slot` of its pools, each rank on a stream of its own. */
    int lora_copy(int slot, const std::vector<std::vector<lora::PackedSite>>& packed,
                  std::string* err);
    bool                        lora_gen_ = false;
    std::unique_ptr<lora::KeyMap> lora_keys_;
    mutable std::mutex          lora_lib_mu_;
    std::map<std::string, std::pair<std::string, std::shared_ptr<const lora::AdapterFile>>> lora_lib_;
private:
    std::atomic<uint64_t>      lora_epoch_{0};   /* step-loop iterations, for slot reuse */
    lora::SlotTable            lora_table_;      /* reads lora_epoch_, so declared after it */
    /* Serialises loads: a slot is reserved, filled on every rank and published by one load at a
     * time. Requests and unloads go to the table alone and never wait for a load's copy. */
    std::mutex                 lora_mu_;
    std::set<std::string>      lora_targets_;
    /* --host-headroom-mib in bytes: the host RAM startup leaves free for the system when it
     * pins host memory (engine_bringup.cpp). 4 GiB suits a busy server; a small host wants to
     * leave less to make room for its tiers, which is why it is a flag and not a constant. */
    int64_t host_headroom_bytes() const { return cfg_.host_headroom_mib * ((int64_t)1 << 20); }
    /* Resolves --embedding-placement, --vision-tower-placement and --ngram-placement into Config
     * before the card is split (engine_bringup.cpp describes how auto chooses). `wfs` is every
     * rank's footprint, read for its towers. `host_late` is the pinned host memory startup
     * allocates after the tables and outside every pool (the checkpoint store, the host
     * activation arenas), which a table pinned in host memory must leave room for. */
    int resolve_table_placement(const std::vector<VramFacts>& facts,
                                const std::vector<int64_t>& emb_bytes,
                                const std::vector<WeightFootprint>& wfs, int64_t kv_floor,
                                int64_t ngram_bytes, bool ngram_device, bool ngram_host,
                                int64_t host_late);
    /* --ngram-placement ram: one copy of the container's [src, src + n) at file offset `foff`,
     * shared by every rank. Locked for the host gather, or pinned for a device kernel to read
     * over the link (`card`). */
    int copy_table(const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                   std::string* why);
    /* An auto-chosen pinned copy failed: move every rank's n-gram gather to the host. */
    bool ngram_gather_to_host();
    /* The table copies: locked ones as [base, base + bytes), pinned ones as rad_dev_alloc
     * RAD_MEM_HOST_MAPPED ranges, and the container address each copies. Freed in ~Engine after
     * the ranks that read them are gone. */
    std::vector<TableCopy> table_copies_;
    std::vector<void*> card_copies_;
    std::vector<std::pair<const void*, void*>> table_copy_of_;
    int load_weights();
    int configure_kv();
    /* Resolves --checkpoint-slots auto for snapshots of `slot` bytes (`what` names their kind). */
    int size_checkpoint_store(int64_t slot, const char* what);
    /* Maps the step's logits rows back to the sequences that wanted them. Not the identity: a
     * mid-prompt prefill chunk produces no row, and a speculative verify produces 1 + n_spec. */
    struct StepOut {
        std::vector<Request*> reqs;       /* the requests that wanted a row, in row order */
        std::vector<int32_t>  first_row;  /* [n_seq] that entry's first row, or -1 */
        /* Leading logits rows the sampler reads. RadBatch::n_out also counts the KL mode's scored
         * rows after them (Request::score). */
        int32_t               n_rows = 0;
        std::vector<int32_t>  rows;       /* [n_out] the device's answer, copied back */
        int32_t*              rb = nullptr;   /* the pinned readback collect_issue filled */
        std::vector<float>    lg;         /* the first row's logits, at -vv */
        std::vector<int32_t>  token;      /* [n_seq] */
        std::vector<int32_t>  accepted;   /* [n_seq] */
        std::vector<uint8_t>  eos;        /* [n_seq] */
        /* [n_rows] each sampled row's logprobs, for the rows of requests that asked for them;
         * empty on a step with none. */
        std::vector<TokenLogprobs> lp;
    };

    /* The text frontend, loaded from the container. The sampler's grammars walk `vocab_view_`,
     * which references `vocab_` and must not outlive it. */
    std::shared_ptr<const Vocab> vocab_;
    std::unique_ptr<VocabView>   vocab_view_;
    /* The container's second vocabulary (rad-convert --tokenizer2), which an image generator that
     * declares `aux_tokens` reads its prompt through; null without one. */
    std::shared_ptr<const Vocab> aux_vocab_;

    /* The chunk geometry, resolved once and shared: the prefix cache needs the checkpoint interval
     * at configure time and the scheduler the whole geometry at init. Resolving it separately
     * would let the cache use the raw interval while the scheduler uses the rounded one. */
    ChunkGeometry                 geo_;

    /* One scheduler for every rank: one decision per step, one batch, shared by pointer. */
    Scheduler                     sched_;
    std::unique_ptr<IKVManager>   kv_iface_;
    std::unique_ptr<IPrefixCache> pc_iface_;

    /* The release point. Width is tp + 1: the scheduler thread also takes part. */
    std::unique_ptr<RankBarrier>  barrier_;
    std::atomic<const RadBatch*>  step_batch_{nullptr};
    /* What a release asks the ranks to issue: the forward pass and its sampler (kIssueAll), the
     * forward pass alone (kIssueModel: the next step's, issued before this one is read back), or
     * that pass's sampler alone (kIssueSampler, once the host has committed its inputs). Every
     * release must set it; a stale value would make a draft pass issue only its sampler. */
    enum { kIssueAll = 0, kIssueModel = 1, kIssueSampler = 2 };
    std::atomic<int>              issue_what_{kIssueAll};
    std::atomic<int>              step_status_{RAD_OK};
    std::atomic<bool>             stopping_{false};
    /* The error the step loop ended on; RAD_OK while running or after a clean stop. Written once
     * by fail_loop(); read by the server watcher to stop HTTP and by serve() to return it. */
    std::atomic<int>              loop_status_{RAD_OK};
    int  fail_loop(int st);

    /* This step's scored rows (Request::token_scores): a logits row and the result it fills.
     * Written by the scheduler thread while the ranks are parked, read by them during the step,
     * consumed by collect_step. */
    std::vector<std::pair<int32_t, TokenScores*>> score_rows_;
    int  stage_scored_rows(const StepPlan& plan, const StepOut& so);
    int  issue_scored_rows(Rank& r);
    int  collect_scored_rows();

    /* Gate for the sampler's host work, which runs while the device executes the model.
     * Staging sampler parameters (and, with a grammar, a whole-vocabulary mask per draft
     * position) touches no stream: Sampler::run issues every upload on the rank thread. So the
     * ranks enqueue the model first, the scheduler runs build_step meanwhile, and the ranks wait
     * here before enqueueing the sampler chain. One-shot per step, reset while the ranks are
     * parked at the release, so no rank can see a stale ready flag. */
    std::mutex                    build_mu_;
    std::condition_variable       build_cv_;
    bool                          build_ready_  = false;
    int                           build_status_ = RAD_OK;

    /* Scratch for the checkpoint copies: one sequence's state slots, reused so the step path
     * does not allocate. See state_slots_of. */
    std::vector<int32_t>          ckpt_slots_;
    std::vector<int32_t>          ckpt_win_;    /* KVManager::window_rows, same reuse */
    bool state_slots_of(uint64_t seq, std::vector<int32_t>* out) const;

    int  load_vocab();
    int  declare_sampler(int rank, RadBuilder* b, int64_t logits_width, bool sizing = false);
    /* The build context for every declare of rank `rank` at `max_tok`; the real declare and the
     * arena's sizing declares differ only in max_tok. */
    RadBuildCtx build_ctx(int rank, int64_t max_tok) const;
    void probe_arena_levels();
    int  step_loop();
    /* Issue the copy of the step's sampled ids to the host, then wait for and interpret them.
     * Split so the copy is issued before draft passes overwrite the ids, and the wait comes after
     * them. */
    int  collect_issue(StepOut& so);
    int  collect_step(const RadBatch* b, StepOut& so);
    int  row_logprobs(StepOut& so, int32_t row, int32_t token);
    /* The step's end on the card: after its last draft pass and the host readbacks, before the
     * step issued ahead. collect_step waits on this event, not the stream, so the card runs the
     * next step while the host commits this one. */
    int  mark_step_end(const RadBatch* b, StepOut& so);
    RadEvent step_end_ = nullptr;

    /* The draft head as declared (rad_declare_drafter), zeroed when no speculative window was
     * asked for. `kind` selects which of the two drivers below runs. `drafter_name_` owns the
     * string because RadDrafterDecl::name points into the Program. */
    RadDrafterDecl       drafter_{};
    std::string          drafter_name_;

    /* Serial driver (spec §10): `depth` dependent passes after the step commits, each embedding
     * the previous proposal. MTP, EAGLE and Medusa-serial have this shape. */
    std::vector<int32_t> draft_out_;   /* [rows][depth], what each pass proposed */

    /* Host side of every step-path readback (sampled tokens, draft proposals). Pinned, because a
     * device-to-host copy into pageable memory bounces synchronously through a staging buffer and
     * blocks on the whole stream; into pinned memory it is one more stream packet and the step's
     * own wait is the only wait. Grown on first use to the largest readback requested. */
    struct Readback {
        int32_t* p = nullptr;
        int64_t  n = 0;
        int32_t* reserve(int64_t count);
        ~Readback();
    };
    Readback             readback_;
    /* Embedding program (rad_declare_pooled): no sampler; a step's answer is the pooled buffer's
     * first n_seq rows. `pooled_host_` is their pinned landing area, [max_seqs, pooled_dim_] f32;
     * collect_pooled copies them behind the step and hands each row to its sink after the
     * step's one wait. */
    bool                 embedding_ = false;
    int64_t              pooled_dim_ = 0;
    float*               pooled_host_ = nullptr;
    int  collect_pooled_issue(const RadBatch* b, StepOut& so);
    int  collect_pooled(const RadBatch* b, StepOut& so);
    /* A generator (rad_declare_generator, rad_declare_audio_generator,
     * rad_declare_video_generator): the driver that turns requests into passes and the loop that
     * runs them, replacing the scheduler and step loop (engine_gen.cpp). Null for every other
     * program. `img_gen_`, `aud_gen_` and `vid_gen_` are the same driver by its kind, for the
     * endpoints; one of them is set when `gen_` is. */
    std::unique_ptr<gen::IPassDriver> gen_;
    gen::Driver*                      img_gen_ = nullptr;
    gen::AudioDriver*                 aud_gen_ = nullptr;
    gen::VideoDriver*                 vid_gen_ = nullptr;
    int  gen_loop();
    bool serial_drafting() const;
    int  advance_all(int mode, int round, bool from_proposal, int rows = -1);
    int  issue_serial_drafter(int step, StepOut& so);
    int  publish_draft(uint64_t seq, const int32_t* tok, int n);
    /* The next step's forward pass, issued before this one is read back; null when the scheduler
     * wants this step committed first. See Scheduler::step_next. */
    int  issue_ahead(const RadBatch** out);
    /* The arena level and the followers' pools, before any release that runs a forward pass. */
    int  prepare_release(const RadBatch* b);
    /* Whether a step may be issued ahead: requires that the drafter, if any, runs on the card (the
     * prompt-lookup drafter and the host's block driver both read the committed sequence). */
    bool pipelining_ = false;
    int  finish_serial_drafter(const StepOut& so);
    /* Handed from issue_serial_drafter to finish_serial_drafter: the drafting rows, the rounds
     * asked for, and the rounds that ran. `serial_k_` is the frozen per-row depth: the last round
     * each row was still in the batch, read off the passes that were actually issued, so the
     * finish can compare the card's StepSlot::gen against the plan rather than the cycle's depth. */
    std::vector<Scheduler::SerialRow> serial_row_;
    std::vector<int32_t> serial_k_;
    int  serial_rounds_ = 0, serial_rows_ = 0, serial_done_ = 0;
    /* Whether round 0 -- the history pass -- is confirmed to have run on every participating rank,
     * which is what serial_end's `history_ran` asks for. Only rank 0's stream can be fenced here,
     * so a multi-rank engine reports false and pays a repeated history pass rather than moving a
     * watermark nobody executed. On a single rank the word also needs the step's wait to have
     * covered the passes: collect_step waits only on a step with logit rows, so
     * finish_serial_drafter takes that wait itself when the step has none. */
    bool serial_hist_ran_ = false;
    /* The StepSlots and end-of-generation ids on every rank, and the pinned host copy of rank 0's
     * slots the step reads back (sched/advance.h). */
    int  step_state_init();
    void step_state_fini();
    StepSlot*            slots_host_ = nullptr;

    /* Step-time attribution, Debug only (see the barrier in step_loop): time per scheduler phase.
     * The nine phases sum to the whole step, including the readback, commit and drafter after the
     * join. */
    enum Ph { PH_PLAN, PH_BEGIN, PH_BUILD, PH_PRE, PH_RANKS, PH_SAVE, PH_COLLECT, PH_COMMIT,
              PH_DRAFT, PH_N };
    double phase_us_[PH_N] = {};
    int    phase_n_ = 0;
    std::chrono::steady_clock::time_point phase_t_{};
    void   phase_mark(int p);

    /* Block driver (spec §10): a context pass mirroring the step just run, then one pass that
     * fills every drafted position at once. DFlash2 has this shape. run_block_drafter stages both
     * after the commit; on the card (`block_card_`, the default), issue_block_drafter queues them
     * behind the step before its readback and finish_block_drafter publishes the proposals after
     * the commit, so the step can be issued ahead. RADIANCE_DRAFT_BLOCK_HOST=1 keeps the host's. */
    int  run_block_drafter(int step);
    int  block_pass(const RadBatch* db, const char* what, int step);
    bool block_card_ = false;
    int  issue_block_drafter(int step, StepOut& so);
    int  finish_block_drafter(const StepOut& so);
    /* Handed from issue_block_drafter to finish_block_drafter and issue_ahead: the drafting rows,
     * in the query pass's order, and how many there are (0 when no query pass ran). */
    std::vector<Scheduler::BlockRow> block_row_;
    int  block_rows_ = 0;
    /* Encoder driver (spec §11, core/engine_mm.cpp): encodes every media item the scheduler waits
     * on, in passes of whole segments, and reads it back into the item. Runs between steps, never
     * inside one. */
    int  run_encodes();
    int32_t enc_step_ = 0;
    void drain_completed();
    /* Terminal ids drain_completed() took, reused so steady state allocates nothing. */
    std::vector<uint64_t> done_;
    /* The tiers' per-iteration work: collect a landed copy, start the next, make room in a full
     * arena. Never waits on the device; the scheduler's lock covers only the bookkeeping. */
    void tier_tick();
    /* --prefix-cache-ttl, once a second, with or without the tiers (PrefixCache::expire). */
    void expire_tick(uint64_t now_ns);
    uint64_t expire_last_ns_ = 0;
    /* Issue and land every waiting host copy now. Called under the scheduler's lock by an
     * eviction that would otherwise erase what the copies cover. True when any landed. */
    bool flush_copies();
    /* The KV loan: what the cache is not using, lent to every rank's expert slab at addresses the
     * cache already owns (KVManager, the loan block). Lives here, not in the scheduler, because
     * rank 0's cache sets the amount while every rank's mover fills or empties slots, and a mover
     * may only be driven while its rank thread is parked, which this thread knows. `idle` also
     * runs each mover's own fill, which a busy engine gets from the step. */
    void loan_tick(bool idle);
    bool     lending_ = false;
    uint64_t loan_last_ns_ = 0;
    /* The activation arena's loan (Engine::arena_step). `arena_rows_[k]` is the largest step Ctx
     * level k serves (level 0 is the full plan); empty when the arena lends nothing. The floor is
     * the largest recent step: the loan grows past it only after no step that large has run for a
     * while, so a run of prefill chunks does not refill and recall the slab between chunks. */
    int  arena_step(const RadBatch* b);
    void arena_tick(uint64_t now_ns);
    int  arena_level(int64_t rows) const;
    std::vector<int64_t> arena_rows_;
    /* The span's shared level (Engine::probe_arena_levels): the rows of the level a step of one
     * interval runs at under --prefill-span, where the prefill stager's ring is the level's own
     * (stager.h); 0 without one. */
    int64_t  arena_span_rows_ = 0;
    int64_t  arena_floor_ = 0;
    int64_t  arena_last_ = 0;
    uint64_t arena_floor_ns_ = 0;
    /* VRAM the budget reserves for allocations after start(), on top of --gpu-headroom-mib: the
     * step batch's staging (allocated by serve()) and what is built on first use, if ever (the KV
     * tiers' copy staging, the second compute lane's queue). Bring-up's final VRAM reading is
     * checked against both. */
    int64_t  late_step_bytes_ = 0;
    int64_t  late_lazy_bytes_ = 0;
    /* Linear-state snapshot store: pinned host memory, --checkpoint-slots slots, each holding every
     * rank's shard of one snapshot (KVManager::set_checkpoint_store). Off the card because no
     * kernel reads a snapshot; it is only copied back into a sequence's state. */
    HostArena ck_store_;
    int  run_server();
    /* KL mode (core/kld.h), replacing run_server: submits the corpus documents as scoring
     * requests and hands every step's scored rows, copied to the host, to `kld_`. One pinned
     * block per rank, holding a step's tokens of that rank's logits columns. */
    int  run_kld();
    int  score_rows(const RadBatch* b, const StepOut& so);
    std::unique_ptr<Kld> kld_;
    std::vector<float*>  kld_host_;
    std::vector<int64_t> kld_vocab0_, kld_width_;   /* per rank read: its logits columns */
    /* The engine half of /server_info, as a JSON object. See the definition in engine.cpp. */
    std::string engine_info() const;

    /* ---- startup report inputs (engine_diag.cpp). `decisions_` holds each value startup resolved
     * and why, recorded where it was resolved; the rest are facts only startup saw. */
    std::vector<diag::Decision> decisions_;
    void decide(const std::string& key, const std::string& value, const std::string& why) {
        diag::decide(&decisions_, key, value, why);
    }
    std::string devnote_;            /* the devices usable_devices left out, and why */
    int         p2p_measured_ = -1;  /* 1 peer to peer, 0 through host memory, -1 not asked */
    std::string p2p_blocked_;
    bool        has_ngram_ = false;  /* the model carries an n-gram table */
    std::string prefix_off_why_;     /* why the engine itself turned prefix caching off, or "" */

    /* ---- control vectors (core/cvec.h). `cvec_` is the host table, loaded
     * once at bringup and empty when the feature is off; `cvec_views_` is, per rank, that table
     * uploaded to the rank's card and the ABI view build_ctx hands to a declaring plugin (the
     * view's `dir` is the allocation, so one vector holds both). `cvec_salt_` partitions the
     * prefix cache between the mode on and off, so a conversation's KV and linear state are never
     * reused by a step that would compute different hidden states from the same tokens. */
    ControlVector                  cvec_;
    std::vector<RadControlVector>  cvec_views_;
    uint64_t                       cvec_salt_ = 0;
    /* Runtime default for requests that name no choice (flippable through
     * POST /admin/control-vector). Atomic: the server's admin thread writes it while
     * request threads read it; nothing about the graph changes with it. */
    std::atomic<bool>              cvec_default_{false};
    std::vector<std::string> env_set_;   /* the RADIANCE_* names set when the engine was made */
    bool        report_printed_ = false;
    /* Runtime device identity (PCI address, name, integrated flag, runtime version), read once in
     * start() beside usable_devices. Later reports make no runtime call: after a failed start
     * there may have been a device fault, and on an HTTP thread (/diagnostics) a runtime call can
     * wait on a lock the launch path holds (see DeviceBackend::vram_used). */
    struct DevId { int device = -1; std::string bdf, name; bool integrated = false; };
    std::vector<DevId> dev_ids_;
    std::string        runtime_version_;
    void identify_devices(const std::vector<std::string>& names);
    /* Syscall-only facts (diag::probe_process), read once at the first report: the memlock limit
     * cannot change under a running process, and the io_uring probe need not repeat per
     * /diagnostics request. */
    bool        probed_ = false;
    int64_t     memlock_soft_ = -1;
    std::string io_uring_;
    /* The last /diagnostics hardware reading, reused for kDiagReuseNs. The SMU answers the card
     * files under the driver's power-management lock, so a polling client must not become a
     * stream of SMU requests beside the server. */
    mutable std::mutex  diag_mu_;
    mutable diag::Facts diag_facts_;
    mutable uint64_t    diag_read_ns_ = 0;
    /* `serving`: read power-management files only for the cards the ranks serve on
     * (diag::collect's `managed`). */
    diag::Facts    hardware_facts(bool serving) const;
    diag::RunFacts run_facts(bool failed) const;

    /* One dispatch of the tiering seam: retire what landed, credit last step's routing, issue
     * what the heat engine proposes. A member because Rank is private; a no-op on a static plan.
     * The definition explains why a step is the unit. */
    void tiering_dispatch(Rank& r, int step);
    void rank_loop(int rank);
};

}  /* namespace rad */
