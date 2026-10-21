/* sampler.h -- the sampler driver.
 *
 * SAMPLING RUNS ON DEVICE (spec §13). At a 151K vocab and batch 256 the logits are ~77 MiB a
 * step; moving them across the link to run a host chain costs both the bandwidth and a
 * synchronisation, on the step path, at exactly the concurrency the engine exists to serve. So
 * every sampler is a kernel, not a lift -- and this file does not implement one. It ISSUES the
 * declared sampler ops in order, exactly as an architecture plugin issues a GEMM: declared
 * through the ordinary builder, resolved through the ordinary hierarchy, serviced by
 * `libref` on the host domain or by a device plugin on the fast one.
 *
 * THE ORDER, and it is not a detail -- reordering changes which tokens survive:
 *
 *     penalties (repetition / frequency / presence, then DRY)
 *     temperature
 *     the grammar mask
 *     top-k
 *     top-p, min-p, typical
 *     XTC
 *     the pick
 *
 * A GREEDY SEQUENCE SKIPS ALL OF IT and takes `sample_argmax` -- after penalties and the mask,
 * because temperature 0 collapses every other logit to -inf and a mask applied afterwards would
 * leave nothing at all. host_ref.cpp makes the same choice for the same reason and the two must
 * agree, or a temperature-0 request stops matching a greedy run.
 *
 * PARAMETERS ARE DATA, NOT BRANCHES. Sequences in one batch have different sampling parameters,
 * so the parameters are a device-side per-sequence array built once per step from a pinned
 * staging buffer -- the same discipline the step batch uses (spec §8). A kernel that branched on
 * a host-side parameter would need one launch per distinct parameter set, which at batch 256 is
 * up to 256 launches for what is one kernel's worth of work.
 *
 * THE SCHEMA THIS DRIVER ISSUES AGAINST (docs/OPS.md, "Sampling"). No per-sequence value is an
 * op PARAMETER, because parameters are geometry -- the thing the selector matches constraints
 * against and freezes at declare. A per-sequence value cannot live there: two requests in one
 * batch with different top_p would be two different geometries and therefore two different
 * resolutions of the same op. The seed and output position `sample_pick` draws from are in the
 * parameter row too. So every sampler op takes the row array as an operand:
 *
 *   sample_penalties   M(range) n_vocab vocab_off?   logits, params, history  -> logits (inout)
 *   sample_dry         M(range) n_vocab vocab_off?   logits, params, history, breakers?
 *                                                                          -> logits (inout)
 *   sample_temp        M(range) n_vocab   logits, params                  -> logits (inout)
 *   sample_mask        M(range) n_vocab   logits, params, bitmask         -> logits (inout)
 *   sample_topk        M(range) n_vocab vocab_off?   logits, params       -> cand_idx, cand_val,
 *                                                                             pairs?
 *   sample_argmax      M(range) n_vocab   logits, params                  -> token
 *   sample_topp        M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_minp        M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_typical     M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_xtc         M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_pick        M(range) n_cand    cand_idx, cand_val, params      -> token
 *   sample_merge_topk  M(range) n_cand    gathered, params                -> cand_idx, cand_val
 *
 * `M` is the number of SAMPLED POSITIONS, not tokens: one per sequence at decode, n_spec+1 per
 * sequence at a verify. `params` is [M] rows of DeviceSampleParams below; `history` is
 * [M, max_hist] int32 of GLOBAL token ids, row m addressed by row m's hist_off/hist_len;
 * `breakers` is [M, max_hist] uint8 beside it, nonzero where that position's token is a whole DRY
 * sequence breaker; `bitmask` is [M, mask_words] uint32 addressed by the row's mask_row.
 * `vocab_off` is this rank's first vocabulary row and is declared only when the vocabulary is
 * split: the three ops that turn a token id into a column of this rank's logits subtract it.
 * Every one of these has a libref implementation: a sampler with no ref implementation has no
 * fallback and no oracle (spec §17).
 */
#pragma once
#include "rad_sample.h"
#include "rad_spechub.h"

#include "../rad_core.h"
#include "grammar.h"
#include "host_ref.h"
#include "vocab_view.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ================================================================== the per-sequence row */

/* THE ROW AND ITS FLAGS LIVE IN abi/rad_sample.h, because they are the wire format of
 * an OPERAND and therefore part of the ABI: libref reads these rows and a kernel plugin
 * cannot include a core header. The alias keeps this file's spelling. */
using DeviceSampleParams = RadSampleParams;

/* ================================================================== configuration */

struct SamplerConfig {
    int64_t n_vocab       = 0;    /* the WHOLE vocabulary */
    int64_t n_vocab_local = 0;    /* THIS RANK's shard of it; == n_vocab when world_size == 1 */
    int64_t max_seqs      = 0;    /* the largest number of sequences one step may carry */
    int64_t max_hist      = 2048; /* the penalty window bound; a longer request is truncated */
    int     max_spec      = 0;
    int     rank          = 0;
    int     world_size    = 1;
    /* This rank's first vocabulary row, which sample_topk adds to every candidate id it packs
     * into the pair plane so the merge sees GLOBAL ids, and which sample_penalties and sample_dry
     * subtract from the GLOBAL ids in the history to find this rank's column. Zero unless lm_head
     * is row-sharded. */
    int64_t vocab_off     = 0;

    /* The candidate-set width the cuts work over. Every sampler after top-k operates on
     * candidates rather than the vocabulary, so this bounds the buffers they read and write. It
     * is a bound, not a policy: a request asking for a larger top-k than this is refused at
     * admission, by name, rather than silently narrowed. */
    int64_t max_candidates = 1024;
};

/* ================================================================== the narrow candidate width */

/* RADIANCE_SAMPLE_CANDIDATES64=1 declares a SECOND candidate plan 64 wide beside the wide one and
 * runs a batch on it when every row's top_k fits. Off, the driver declares the wide plan only and
 * this header's behaviour is the one it always had.
 *
 * IT IS NOT A SMALLER max_candidates. `max_candidates` is the admission bound and stays as the
 * deployment configured it: a request asking for top_k 70 is still admitted, and still served --
 * by the wide plan, for the whole batch it lands in. What the narrow plan buys is that a batch of
 * ordinary requests (top_k 20, say) stops paying for a 1024-wide candidate plane in the narrowing
 * stages, the pick, and -- under tensor parallelism -- the pair gather and the merge.
 *
 * The width is 64 because that is the width this engine's own sampler rows are written against
 * (`REF_SAMPLE_CAND` in libref/ref_shapes.cpp, `n_cand` in libr4d/r4d_shapes.cpp), so a 64-wide
 * candidate plane is the one every resolved sampler already has a kernel for. */
enum { kSampleNarrowCand = 64 };

/* Exactly "1", like every other RADIANCE_* switch here: a typo is off, never something else. */
inline bool sample_candidates64_enabled() {
    const char* v = std::getenv("RADIANCE_SAMPLE_CANDIDATES64");
    return v && !std::strcmp(v, "1");
}


/* ================================================================== the driver */

class Sampler {
public:
    Sampler() = default;
    ~Sampler();

    /* ---------------------------------------------------------- declare phase */

    /* Declare every sampler op and every buffer the chain needs, and allocate the pinned staging.
     * Runs once, alongside the architecture plugin's declare, so the sampler's kernels resolve
     * through the same hierarchy at the same time as everything else and a missing one is in the
     * same missing-list (spec §3.1).
     *
     * A sampler op that does not resolve is NOT a declare failure: a deployment that never sends
     * `typical_p` does not need a typical kernel. It becomes a failure when a request asks for
     * it, which `check_request` reports by name. */
    int declare(RadBuilder* b, const SamplerConfig& cfg);

    /* The vocabulary view the grammar backends walk. Set after the tokeniser loads, before the
     * first request. */
    void attach_vocab(const VocabView* v) { vocab_ = v; }

    /* ---------------------------------------------------------- admission */

    /* Can this request's sampling parameters be served? Returns RAD_OK, or RAD_E_NOKERNEL naming
     * the sampler that did not resolve, or RAD_E_INVAL naming the parameter that is out of range.
     * Called on admission, so a request fails at the door rather than three steps in. */
    int check_request(const SamplingParams& sp, std::string* why) const;

    /* Build this request's grammar state, if it has a grammar. Runs on the scheduler thread, so
     * it must not compile: a request admitted through the server carries its grammar compiled
     * (SamplingParams::grammar_program), and this finds that program and positions a grammar in
     * it. A request that arrives without one is compiled here, which costs its mask plan. */
    int begin_request(uint64_t req_id, const SamplingParams& sp, std::string* err);
    void end_request(uint64_t req_id);
    /* Does this request already have state here? The step loop reaches a request on every step it
     * is scheduled in, and begin_request is not idempotent -- calling it twice would rebuild the
     * automaton and throw away everything it had accepted, which silently unconstrains a
     * generation from its second token onward. */
    bool has_request(uint64_t req_id) const;

    /* Requests build_step could not serve: a grammar that reached a state admitting no token.
     * Returned and cleared; the caller cancels them. Empty on almost every step. */
    std::vector<uint64_t> take_failed();

    /* Advance a request's grammar over the token it just emitted. */
    int accept_token(uint64_t req_id, int32_t token);

    /* May the scheduler speculate on this request? Yes, unless it carries a constraining grammar
     * whose backend cannot snapshot its state. The mask for draft position i depends on which of
     * positions 0..i-1 were accepted, so masking a window means walking the grammar forward over
     * the proposals and rewinding afterwards; a backend without push_state/rollback cannot, and
     * such a request decodes one token at a time rather than silently emitting drafts the grammar
     * forbids. */
    bool allows_speculation(uint64_t req_id) const;
    /* Is allows_speculation's answer fixed for this request? It changes only when a grammar whose
     * backend cannot rewind starts to constrain, which is a commit's doing; a request with no
     * grammar, or one whose backend rewinds, answers the same at every step. */
    bool speculation_fixed(uint64_t req_id) const;
    /* May a draft for this request branch? Not while it carries a grammar: the mask walk conditions
     * every window row on one linear history, so build_step refuses a branched window under a
     * grammar. A tree drafter publishes a chain for such a request instead. */
    bool allows_branching(uint64_t req_id) const;
    /* Bind the GPU proposal to its request and committed output coordinate. */
    int set_spechub(uint64_t req_id, const RadSpechubState& state);
    /* RADIANCE_MTP_COUPLED: the bound per-round states of the linear draft just published, n of
     * them (0 clears). RAD_E_NOTFOUND for a request already ended here. */
    int set_mtp_couple(uint64_t req_id, const RadMtpCoupleState* states, int n);

    /* ---------------------------------------------------------- step path */

    /* Build the per-sequence parameter array, the packed penalty history and the grammar
     * bitmasks for this step, into pinned staging. Scheduler-thread work: it touches request
     * state and host grammar automata, and it must be done before the rank threads are released
     * at the barrier.
     *
     * Allocates nothing -- every buffer was sized at declare against max_seqs (spec §3.1). */
    int build_step(const std::vector<Request*>& reqs);

    /* THE SAME STAGING, FILLED FOR A GREEDY DRAFTER. `n` rows of argmax: no penalties, no
     * grammar mask, no history, and the RNG untouched -- so the chain resolves to op_argmax and
     * nothing else runs.
     *
     * It exists because a DRAFT round is not a request's own sampling. Its output is a
     * proposal the target then verifies, and the verifier is greedy (core/sample/accept.h): a
     * drafter that sampled at the request's temperature would spend the request's RNG stream
     * on tokens that may never be emitted, and would lower acceptance for nothing. Every MTP
     * head that ships inside a target checkpoint is greedy for the same reason. */
    int build_greedy(int64_t n);

    /* Upload the staging and issue the chain. Rank-thread work, on the compute stream, with no
     * host synchronisation in it.
     *
     * `logits` holds [n_seq, n_vocab_local] and `out_tokens` receives [n_seq] token ids. Under
     * tensor parallel the candidate merge (`sample_merge_topk`, or `row_topk_merge` fused with the
     * top-k) happens between the top-k and the cuts; full logits are never gathered. */
    int run(RadCtx* c, rad_buf logits, rad_buf out_tokens);

    /* The parameter rows the last build_step or build_greedy staged, and how many. What `run`
     * uploads, readable so the staging can be checked without a device behind it. */
    const DeviceSampleParams* staged_params(int64_t* n) const {
        if (n) *n = n_seq_;
        return cur_stage_;
    }
    /* The coupling rows the last build_step staged (status 0 = ordinary pick), row-aligned with
     * staged_params; null when the flag is off. */
    const RadMtpCoupleState* staged_couple() const { return couple_ ? couple_stage_ : nullptr; }

private:
    /* IS THE VOCABULARY ACTUALLY SPLIT? Not the same question as "is there more than one rank".
     * A tensor-parallel deployment whose plugin declares `output.weight` replicated gives every
     * rank the whole vocabulary, and then the candidate merge is not just unnecessary but wrong:
     * every rank would contribute the SAME k candidates and the gather would count them once per
     * rank. The declared width of this rank's logits is the fact, so that is what this asks. */
    bool vocab_parallel() const {
        return cfg_.world_size > 1 && cfg_.n_vocab_local < cfg_.n_vocab;
    }

    /* `dry` holds the request's resolved sequence breakers, empty when it sent none -- which is
     * not "DRY off": DRY runs whenever its multiplier is set, with or without breakers. */
    struct SeqState {
        std::unique_ptr<GrammarBackend> grammar;
        DryBreakers                     dry;
        /* Where the reply stands against its reasoning block, over `output[0, think_seen)`.
         * Advanced by build_step over what each commit appended, so a step costs the tokens it
         * added and not the reply's length. Read only when the request carries a budget. */
        ReasoningCount                  think;
        size_t                          think_seen = 0;
        RadSpechubState spechub{};
        RadMtpCoupleState couple[RAD_MTP_COUPLE_MAX_ROUNDS]{};
        int couple_n = 0;
    };

    /* Mask staged row `row` to the reasoning block's close token: the budget is spent and the
     * sampled position is still inside the block. A grammar already masking the row keeps the
     * last word -- the close is forced only where the grammar admits it, and a row whose grammar
     * forbids it is left to the grammar and forced at a later position. */
    void force_close(DeviceSampleParams& p, int64_t row, int32_t close);

    SamplerConfig    cfg_{};
    const VocabView* vocab_ = nullptr;
    bool             declared_ = false;

    /* Declared op handles. RAD_NULL_HANDLE means the op did not resolve; the admission check
     * turns that into a named refusal rather than a silent skip. */
    rad_op op_penalties_ = RAD_NULL_HANDLE;
    rad_op op_dry_       = RAD_NULL_HANDLE;
    rad_op op_temp_      = RAD_NULL_HANDLE;
    rad_op op_mask_      = RAD_NULL_HANDLE;
    rad_op op_argmax_    = RAD_NULL_HANDLE;

    /* ONE CANDIDATE PLAN: the width its planes are cut to, the ops declared over that width, and
     * the planes themselves.
     *
     * THE SAME PLAN TYPE DECLARED TWICE is what makes RADIANCE_SAMPLE_CANDIDATES64 an A/B instead
     * of a second code path: `wide_` is the deployment's `max_candidates` and is what every batch
     * ran on before the flag existed; `narrow_` is the same ops, the same buffers and the same
     * issue order, cut to kSampleNarrowCand. A batch runs on exactly one of them, whole, and the
     * ops after top-k read their geometry off the plan rather than off a member that could
     * disagree with the buffers it was handed. */
    struct Plan {
        int64_t  n_cand    = 0;
        rad_op   topk      = RAD_NULL_HANDLE;
        rad_op   topp      = RAD_NULL_HANDLE;
        rad_op   minp      = RAD_NULL_HANDLE;
        rad_op   typical   = RAD_NULL_HANDLE;
        rad_op   xtc       = RAD_NULL_HANDLE;
        rad_op   pick      = RAD_NULL_HANDLE;
        rad_op   allgather = RAD_NULL_HANDLE;   /* tensor parallel only */
        rad_op   merge     = RAD_NULL_HANDLE;   /* and the merge beside it */
        rad_op   spechub   = RAD_NULL_HANDLE;   /* RADIANCE_SPECHUB only */
        rad_op   couple    = RAD_NULL_HANDLE;   /* RADIANCE_MTP_COUPLED only */
        rad_buf  cand_i    = RAD_NULL_HANDLE;
        rad_buf  cand_v    = RAD_NULL_HANDLE;
        rad_buf  pairs     = RAD_NULL_HANDLE;   /* this rank's candidates as (global id, logit) */
        rad_buf  gather    = RAD_NULL_HANDLE;
    };
    Plan wide_{};
    Plan narrow_{};
    /* Was `narrow_` declared at all? False without the flag, and false when the deployment's own
     * width is already at or below the narrow one -- declaring a plan identical to `wide_` would
     * be one more handle for no candidate byte saved. */
    bool narrow_declared_ = false;

    /* Device-side buffers, from the ordinary buffer plan. */
    rad_buf buf_params_  = RAD_NULL_HANDLE;
    rad_buf buf_hist_    = RAD_NULL_HANDLE;
    rad_buf buf_brk_     = RAD_NULL_HANDLE;   /* DRY's single-token breaker marks, beside history */
    rad_buf buf_mask_    = RAD_NULL_HANDLE;
    rad_buf buf_spechub_ = RAD_NULL_HANDLE;
    RadSpechubState* spechub_stage_ = nullptr;
    bool spechub_ = false;
    bool any_spechub_ = false;
    rad_buf buf_couple_ = RAD_NULL_HANDLE;
    RadMtpCoupleState* couple_stage_ = nullptr;
    bool couple_ = false;
    bool any_couple_ = false;

    /* Pinned staging. One allocation each, at declare, never on the step path. */
    DeviceSampleParams* params_stage_ = nullptr;
    /* THE GREEDY ROWS HAVE STAGING OF THEIR OWN, written once. A draft pass is staged while the
     * step before it -- whose sampler uploads `params_stage_` when the card reaches it -- may not
     * have run yet; rows written into the same staging would reach that step's sampler instead.
     * Every greedy row is the same row, so this one is filled on first use and never again, and
     * no upload of it can ever see a later build. */
    DeviceSampleParams* greedy_stage_ = nullptr;
    bool                greedy_ready_ = false;
    const DeviceSampleParams* cur_stage_ = nullptr;   /* what the last build staged */
    int32_t*            hist_stage_   = nullptr;
    uint8_t*            brk_stage_    = nullptr;
    uint32_t*           mask_stage_   = nullptr;

    int64_t mask_words_ = 0;      /* words for THIS RANK's shard; what the mask op indexes */
    int64_t n_seq_      = 0;
    /* THE PLAN THIS STEP RUNS ON, chosen in build_step and read by run. Never null once declared:
     * a batch with any row the narrow plan cannot serve keeps `wide_`, whole. */
    const Plan* plan_   = &wide_;
    /* Does any DRY row this step carry breakers? Only then is the breaker plane uploaded and
     * handed to sample_dry; a batch whose DRY requests sent none issues it without the operand. */
    bool    any_brk_    = false;

    /* The whole-vocabulary mask the grammar fills, from which the staged shard is cut. Only used
     * when the head is sharded; see the note in declare(). */
    std::vector<uint32_t> mask_full_;

    /* The grammar mask's cost, accumulated for the Debug line in build_step. Not state the step
       path reads -- a diagnostic, and one worth carrying: the mask walk is scheduler-thread work
       large enough to decide the step time of a constrained request, so it has to be visible
       rather than assumed. */
    double  mask_us_       = 0.0;
    double  mask_worst_us_ = 0.0;
    int64_t mask_us_n_     = 0;
    int64_t mask_fast_     = 0;   /* memo hits */
    int64_t mask_slow_     = 0;   /* walks    */
    double  mask_walk_us_  = 0.0; /* and what only the walks cost */
    double  build_us_      = 0.0;
    int64_t build_n_       = 0;
    double  accept_us_     = 0.0;
    int64_t accept_n_      = 0;
    void shard_mask(uint32_t* dst) const;

    /* Declare one candidate plan at width `n_cand`: its two planes, its pair and gathered planes
     * under tensor parallel, and the ops that read them. `tag` distinguishes the two plans' buffer
     * names, because a builder hands out one handle per name. Returns RAD_OK, or the first status
     * the liveness declarations refused. */
    int declare_plan(RadBuilder* b, Plan* p, int64_t rows, int64_t n_cand, const char* tag);

    /* May this step run on the narrow plan? EVERY active row must be one the narrow width serves
     * with the same answer the wide width gives, and the test is per BATCH: one row that fails it
     * puts the whole batch back on `wide_`, because a plan is a set of buffers and a stage cannot
     * read half a row from one width and half from another. */
    bool narrow_suits(const DeviceSampleParams* rows, int64_t n) const;

    /* Requests whose grammar died this step. Drained by the engine, which cancels them -- one bad
     * request must not stop the step loop for everyone else. */
    std::vector<uint64_t> failed_;

    /* Per-request grammar and DRY state, keyed by request id. Held here rather than in Request
     * because Request is a frozen struct shared across components and a unique_ptr in it would
     * make the scheduler's copies a compile error. */
    std::unordered_map<uint64_t, SeqState> seqs_;

    int stage_alloc();
    void stage_free();
};

}  /* namespace rad */
