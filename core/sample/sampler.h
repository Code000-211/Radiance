/* sampler.h -- the sampler driver.
 *
 * Sampling runs on device (spec §13): at a 151K vocab and batch 256 the f32 logits are ~148 MiB a
 * step, too much to move to the host and synchronise on the step path. This file implements no
 * sampler. It issues the declared sampler ops in order, as an architecture plugin issues a GEMM:
 * declared through the ordinary builder and resolved through the ordinary hierarchy to `libref`
 * on the host domain or to a device plugin.
 *
 * Chain order (reordering changes which tokens survive):
 *
 *     penalties (repetition / frequency / presence, then DRY)
 *     temperature
 *     the grammar mask
 *     top-k
 *     top-p, min-p, typical
 *     XTC
 *     the pick
 *
 * Temperature last: a request may ask for llama.cpp's order with `samplers`
 * (SamplingParams::temp_last). Such a row is staged with temperature 1 for the stage above, which
 * then skips it, and with its own value for a second `sample_temp` over the candidate plane after
 * XTC (the same kernel at the candidate width). Every other row is 1 there.
 *
 * Logprobs: `sample_logprobs` summarises the raw logits, before the first stage, for each row
 * flagged RAD_SP_LOGPROBS: its maximum, its normaliser and its kLpK largest (one more than
 * RAD_MAX_LOGPROBS, see kLpK). Rank 0 copies the summary back behind the pick. The emitted
 * token's logprob is its raw logit minus the log normaliser. When the token is not among the
 * largest, its raw logit is gone (every stage writes the logits in place), so logprobs_for
 * recovers it from the candidate-plane value the pick drew it at, undoing temperature, DRY and
 * the penalties on the host. Such a row is therefore never greedy-flagged: a greedy row asking
 * for logprobs is staged as a top-1 chain row.
 *
 * Greedy: a greedy sequence skips the chain and takes `sample_argmax`, after penalties and the
 * mask. Temperature 0 collapses every other logit to -inf, so a mask applied afterwards would
 * leave nothing. host_ref.cpp makes the same choice and the two must agree, or a temperature-0
 * request stops matching a greedy run.
 *
 * Parameters are data, not branches. They are a per-sequence device array built once per step
 * from pinned staging, as the step batch is (spec §8). Branching on host-side parameters would
 * need one launch per distinct parameter set, up to 256 at batch 256.
 *
 * Schema (docs/OPS.md, "Sampling"). Op parameters are geometry, matched and frozen at declare,
 * so no per-sequence value can be one: two requests with different top_p would be two
 * resolutions of the same op. The seed and output position `sample_pick` draws from are in the
 * row too. Every sampler op therefore takes the row array as an operand:
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
 *   sample_logprobs    M(range) n_vocab k vocab_off?   logits, params     -> stats
 *
 * `M` counts sampled positions, not tokens: one per sequence at decode, n_spec+1 per sequence at
 * a verify. `params` is [M] DeviceSampleParams rows. `history` is [M, max_hist] int32 global
 * token ids, row m addressed by its hist_off/hist_len. `breakers` is [M, max_hist] uint8 beside
 * it, nonzero where that token is a whole DRY sequence breaker. `bitmask` is [M, mask_words]
 * uint32, addressed by the row's mask_row. `vocab_off` is this rank's first vocabulary row,
 * declared only when the vocabulary is split; the three ops that map between token ids and this
 * rank's logit columns use it. Every op has a libref implementation, which is both its fallback
 * and its oracle (spec §17).
 */
#pragma once
#include "rad_sample.h"
#include "rad_spechub.h"

#include "../rad_core.h"
#include "grammar.h"
#include "host_ref.h"
#include "vocab_view.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ================================================================== the per-sequence row */

/* The row and its flags live in abi/rad_sample.h: they are an operand's wire format, read by
 * libref and by kernel plugins, which cannot include a core header. */
using DeviceSampleParams = RadSampleParams;

/* ================================================================== configuration */

struct SamplerConfig {
    int64_t n_vocab       = 0;    /* the whole vocabulary */
    int64_t n_vocab_local = 0;    /* this rank's shard of it; == n_vocab when world_size == 1 */
    int64_t max_seqs      = 0;    /* the largest number of sequences one step may carry */
    int64_t max_hist      = 2048; /* the penalty window bound; a longer request is truncated */
    int     max_spec      = 0;
    int     rank          = 0;
    int     world_size    = 1;
    /* This rank's first vocabulary row. sample_topk adds it to the candidate ids it packs into
     * the pair plane, so the merge sees global ids; sample_penalties and sample_dry subtract it
     * from the history's global ids. Zero unless lm_head is row-sharded. */
    int64_t vocab_off     = 0;

    /* Candidate-set width for every stage after top-k; sizes their buffers. top_k 0 samples
     * from this many of the most likely tokens, not the whole vocabulary. A request asking for a
     * larger top-k is refused at admission, not silently narrowed. */
    int64_t max_candidates = kSampleCandidates;
};

/* ================================================================== the driver */

class Sampler {
public:
    Sampler() = default;
    ~Sampler();

    /* ---------------------------------------------------------- declare phase */

    /* Declare every sampler op and buffer the chain needs and allocate the pinned staging. Runs
     * alongside the architecture plugin's declare, so a missing kernel lands in the same
     * missing-list (spec §3.1). An unresolved op is not a declare failure; check_request refuses
     * the requests that need it, by name. */
    int declare(RadBuilder* b, const SamplerConfig& cfg);

    /* The vocabulary the grammar backends walk. Set after the tokeniser loads, before the first
     * request. */
    void attach_vocab(const VocabView* v) { vocab_ = v; }

    /* ---------------------------------------------------------- admission */

    /* Admission check. RAD_OK, RAD_E_NOKERNEL naming the sampler that did not resolve, or
     * RAD_E_INVAL naming the out-of-range parameter. */
    int check_request(const SamplingParams& sp, std::string* why) const;

    /* Build the request's grammar state, if it has a grammar. Runs on the scheduler thread, so it
     * should not compile: a server request carries its grammar compiled
     * (SamplingParams::grammar_program) and this only positions a grammar in it. A request
     * without one is compiled here, at the cost of its mask plan. */
    int begin_request(uint64_t req_id, const SamplingParams& sp, std::string* err);
    void end_request(uint64_t req_id);
    /* begin_request is not idempotent: a second call rebuilds the automaton and discards what it
     * accepted, unconstraining the rest of the generation. The step loop checks this first. */
    bool has_request(uint64_t req_id) const;

    /* Requests whose grammar reached a state admitting no token. Returned and cleared; the
     * caller cancels them. */
    std::vector<uint64_t> take_failed();

    /* Advance a request's grammar over the token it just emitted. */
    int accept_token(uint64_t req_id, int32_t token);

    /* False only for a constraining grammar whose backend cannot snapshot its state. Masking a
     * draft window walks the grammar over the drafts and rewinds, which needs
     * push_state/rollback; without it the request decodes one token at a time. */
    bool allows_speculation(uint64_t req_id) const;
    /* Whether allows_speculation's answer can no longer change. It changes only when a grammar
     * whose backend cannot rewind starts to constrain (on a commit). */
    bool speculation_fixed(uint64_t req_id) const;
    /* RADIANCE_MTP_COUPLED: bind the per-round states of the linear draft just published, `n` of
     * them (0 clears). Each must name this request, the round it belongs to, and a `draw.mode` the
     * row actually ran. RAD_E_NOTFOUND for a request already ended here. */
    int set_mtp_couple(uint64_t req_id, const RadMtpCoupleState* states, int n);

    /* ---------------------------------------------------------- step path */

    /* Stage this step's parameter rows, penalty history and grammar bitmasks into pinned memory.
     * Scheduler thread, before the rank threads are released at the barrier. Allocates nothing;
     * every buffer was sized at declare against max_seqs (spec §3.1). */
    int build_step(const std::vector<Request*>& reqs);

    /* Stage `n` greedy rows for a drafter: no penalties, mask or history, and no RNG use, so
     * only the argmax runs. Draft output is a proposal checked by a greedy verifier
     * (core/sample/accept.h); sampling it at the request's temperature would spend the request's
     * RNG stream on tokens that may never be emitted and lower acceptance. */
    int build_greedy(int64_t n);

    /* Upload the staging and issue the chain. Rank thread, compute stream, no host sync.
     * `logits` is [n_seq, n_vocab_local]; `out_tokens` receives [n_seq] token ids. Under tensor
     * parallel the candidate merge (`sample_merge_topk`, or `row_topk_merge` fused with the
     * top-k) runs between the top-k and the cuts; full logits are never gathered. */
    int run(RadCtx* c, rad_buf logits, rad_buf out_tokens);

    /* The emitted token's logprobs, once the step the last build_step staged has completed.
     * `row` must have been staged with SamplingParams::n_probs >= 0 and `token` is what was
     * sampled there. Rank 0 only, since it holds the copy. RAD_E_STATE for a row not staged for
     * logprobs; RAD_E_INVAL with `out->token` = -1 when the numbers are not finite (no finite
     * logit in the row, or none for the token). */
    int logprobs_for(int64_t row, int32_t token, TokenLogprobs* out) const;
    /* Whether `sample_logprobs` resolved; what the server advertises as logprobs support. */
    bool has_logprobs() const { return op_logprobs_ != RAD_NULL_HANDLE; }
    const SamplerConfig& config() const { return cfg_; }

    /* The rows the last build_step or build_greedy staged, as `run` uploads them; lets the
     * staging be checked without a device. */
    const DeviceSampleParams* staged_params(int64_t* n) const {
        if (n) *n = n_seq_;
        return cur_stage_;
    }
    /* Staged row `row`'s bitmask as `run` uploads it: this rank's shard, `mask_words()` words,
     * meaningful only where the row carries RAD_SP_MASK. */
    const uint32_t* staged_mask(int64_t row) const { return mask_stage_ + row * mask_words_; }
    int64_t         mask_words() const { return mask_words_; }
    /* Staged row `row`'s bound coupled state, as `run` uploads it: all-zero where the row carries
     * none (bonus row, greedy row, uncoupled step). Lets a build be checked without a device. */
    const RadMtpCoupleState& staged_couple(int64_t row) const { return couple_stage_[row]; }
    /* Whether any row of the last build carries a bound state, i.e. whether `run` issues the
     * verify op at all. */
    bool staged_any_couple() const { return any_couple_; }
    /* The verify op the last build resolved, for naming its kernel in a log line. */
    rad_op staged_couple_op() const { return op_couple_; }

private:
    /* Whether the vocabulary is split, judged by this rank's logit width rather than the rank
     * count: with `output.weight` replicated every rank holds the whole vocabulary, and a merge
     * would count each candidate once per rank. */
    bool vocab_parallel() const {
        return cfg_.world_size > 1 && cfg_.n_vocab_local < cfg_.n_vocab;
    }

    /* `dry` holds the resolved sequence breakers, empty when the request sent none. Empty does
     * not mean DRY is off; DRY runs whenever its multiplier is set. */
    struct SeqState {
        std::unique_ptr<GrammarBackend> grammar;
        DryBreakers                     dry;
        /* Reasoning-block state over `output[0, think_seen)`. build_step advances it over the
         * newly committed tokens only. Read only when the request carries a thinking budget. */
        ReasoningCount                  think;
        size_t                          think_seen = 0;
        /* RADIANCE_MTP_COUPLED: the linear draft's per-round states for the window the scheduler
         * published, bound by set_mtp_couple and consumed by build_step. `couple_n` is how many of
         * `couple` are live; 0 for a request whose drafts carry no coupled proposal. */
        RadMtpCoupleState couple[RAD_MTP_COUPLE_MAX_ROUNDS]{};
        int               couple_n = 0;
    };

    /* Mask staged row `row` to the reasoning block's close token, once the budget is spent
     * inside the block. A grammar mask on the row wins: the close is forced only where the
     * grammar admits it, otherwise at a later position. */
    void force_close(DeviceSampleParams& p, int64_t row, int32_t close);

    SamplerConfig    cfg_{};
    const VocabView* vocab_ = nullptr;
    bool             declared_ = false;

    /* Declared op handles; RAD_NULL_HANDLE means unresolved, which admission refuses by name. */
    rad_op op_penalties_ = RAD_NULL_HANDLE;
    rad_op op_dry_       = RAD_NULL_HANDLE;
    rad_op op_temp_      = RAD_NULL_HANDLE;
    rad_op op_temp_late_ = RAD_NULL_HANDLE;   /* sample_temp over the candidates (temp_last) */
    rad_op op_mask_      = RAD_NULL_HANDLE;
    rad_op op_topk_      = RAD_NULL_HANDLE;
    rad_op op_topp_      = RAD_NULL_HANDLE;
    rad_op op_minp_      = RAD_NULL_HANDLE;
    rad_op op_typical_   = RAD_NULL_HANDLE;
    rad_op op_xtc_       = RAD_NULL_HANDLE;
    rad_op op_pick_      = RAD_NULL_HANDLE;
    rad_op op_argmax_    = RAD_NULL_HANDLE;
    rad_op op_allgather_ = RAD_NULL_HANDLE;
    rad_op op_merge_     = RAD_NULL_HANDLE;   /* the candidate merge, under tensor parallel */
    rad_op op_couple_    = RAD_NULL_HANDLE;   /* RADIANCE_MTP_COUPLED only */
    rad_op op_logprobs_  = RAD_NULL_HANDLE;   /* summary of the raw distribution */
    rad_op op_lp_gather_ = RAD_NULL_HANDLE;   /* every rank's summary, under a split vocabulary */

    /* Device-side buffers, from the ordinary buffer plan. */
    rad_buf buf_params_  = RAD_NULL_HANDLE;
    rad_buf buf_params_late_ = RAD_NULL_HANDLE;   /* the late temperature's rows */
    rad_buf buf_hist_    = RAD_NULL_HANDLE;
    rad_buf buf_brk_     = RAD_NULL_HANDLE;   /* DRY's single-token breaker marks, beside history */
    rad_buf buf_mask_    = RAD_NULL_HANDLE;
    rad_buf buf_cand_i_  = RAD_NULL_HANDLE;
    rad_buf buf_cand_v_  = RAD_NULL_HANDLE;
    rad_buf buf_gather_  = RAD_NULL_HANDLE;
    rad_buf buf_pairs_   = RAD_NULL_HANDLE;   /* this rank's candidates as (global id, logit) */
    rad_buf buf_lp_      = RAD_NULL_HANDLE;   /* sample_logprobs' stats, [rows, lp_width()] */
    rad_buf buf_lp_g_    = RAD_NULL_HANDLE;   /* and every rank's side by side */
    rad_buf buf_couple_  = RAD_NULL_HANDLE;   /* [rows, RAD_MTP_COUPLE_STATE_WORDS], flag only */
    RadMtpCoupleState* couple_stage_ = nullptr;   /* pinned, one row per sampled position */
    bool    couple_      = false;   /* RADIANCE_MTP_COUPLED */
    bool    any_couple_  = false;   /* does any row this step carry a bound state? */

    /* Pinned staging, allocated once at declare. */
    DeviceSampleParams* params_stage_ = nullptr;
    /* Copies of params_stage_'s rows with temperature 1 unless the row runs temperature last.
     * Uploaded only when some row does (any_late_). */
    DeviceSampleParams* late_stage_   = nullptr;
    bool                any_late_     = false;
    /* Greedy rows have their own staging, filled once on first use. A draft pass is staged
     * while the previous step's upload of params_stage_ may still be pending, so writing there
     * would change that step's rows. Every greedy row is identical, so no upload of this buffer
     * can observe a later build. */
    DeviceSampleParams* greedy_stage_ = nullptr;
    bool                greedy_ready_ = false;
    const DeviceSampleParams* cur_stage_ = nullptr;   /* what the last build staged */
    int32_t*            hist_stage_   = nullptr;
    uint8_t*            brk_stage_    = nullptr;
    uint32_t*           mask_stage_   = nullptr;

    int64_t mask_words_ = 0;      /* words for this rank's shard; what the mask op indexes */
    int64_t n_seq_      = 0;
    /* Whether any DRY row this step has breakers; only then is the breaker plane uploaded and
     * passed to sample_dry. */
    bool    any_brk_    = false;

    /* The whole-vocabulary mask the grammar fills, from which this rank's shard is cut. Used
     * only when the head is sharded; see declare(). */
    std::vector<uint32_t> mask_full_;

    /* Debug-level timing of the grammar mask, build_step and accept_token. The mask walk is
       scheduler-thread work that can set the step time of a constrained request. */
    double  mask_us_       = 0.0;
    double  mask_worst_us_ = 0.0;
    int64_t mask_us_n_     = 0;
    int64_t mask_fast_     = 0;   /* memo hits */
    int64_t mask_slow_     = 0;   /* walks    */
    double  mask_walk_us_  = 0.0; /* time spent in walks only */
    double  build_us_      = 0.0;
    int64_t build_n_       = 0;
    double  accept_us_     = 0.0;
    int64_t accept_n_      = 0;
    void shard_mask(uint32_t* dst) const;

    /* Requests whose grammar died this step. The engine drains and cancels them, so one bad
     * request does not stop the step loop. */
    std::vector<uint64_t> failed_;

    /* ---------------------------------------------------------- logprobs */
    /* RAD_MAX_LOGPROBS + 1 pairs, so a summary row is 44 floats: whole 16-byte words, which
     * libr4d's all-gather of a split vocabulary requires. The extra pair is never reported. */
    static constexpr int kLpK = RAD_MAX_LOGPROBS + 1;
    static int64_t lp_width() { return 2 + 2 * (int64_t)kLpK; }
    /* What raw_logit needs to undo a RAD_SP_LOGPROBS row's chain, saved by build_step: the
     * staged row, the temperature the candidate value carries, the history window and the DRY
     * breakers. Copied, because the next step rebuilds the staging. */
    struct LpRow {
        int64_t            row = -1;
        int32_t            n_top = 0;
        float              temp = 1.0f;          /* the candidate value is l / temp */
        DeviceSampleParams p{};
        std::vector<int32_t> hist;               /* the staged history row, [0, hist_len) */
        const DryBreakers* dry = nullptr;        /* lives in seqs_ for the request's lifetime */
    };
    std::vector<LpRow> lp_rows_;
    /* Rank 0's copies, written behind the pick: the summaries ([rows, W * lp_width()], W ranks
     * side by side) and the candidate planes the pick drew from. */
    float*   lp_host_    = nullptr;
    int32_t* lp_ci_host_ = nullptr;
    float*   lp_cv_host_ = nullptr;
    int64_t  lp_lo_ = 0, lp_hi_ = 0;             /* the rows copied, [lo, hi) */
    /* A token's raw logit from the value the pick drew it at: temperature, DRY and the penalties
     * undone in reverse order. */
    float raw_logit(const LpRow& L, int32_t token, float v) const;

    /* Per-request grammar and DRY state. Not in Request, a frozen struct the scheduler copies,
     * which a unique_ptr member would make non-copyable. */
    std::unordered_map<uint64_t, SeqState> seqs_;

    int stage_alloc();
    void stage_free();
};

/* Merge `W` ranks' `sample_logprobs` rows of `k` pairs each, side by side in `st`
 * (W * (2 + 2k) floats). Sets the combined log normaliser and fills up to `cap` of the largest
 * (id, logit) pairs, descending, ties to the lower id; returns how many. Ranks with a non-finite
 * maximum or normaliser and pairs with a non-finite logit are skipped; `*log_z` is NaN when no
 * rank contributes. W = 1 is valid. Used by Sampler::logprobs_for. */
int merge_logprob_rows(const float* st, int W, int k, double* log_z, int32_t* ids, float* vals,
                       int cap);

}  /* namespace rad */
