/* sampler.cpp -- the sampler driver: declare the sampler ops, build the per-sequence parameter
 * array once per step, issue the chain. No sampler is implemented here; see sampler.h.
 */
#include "sampler.h"
#include "gbnf.h"

#include "../../arch/common/rad_spechub_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rad {

/* ================================================================== grammar triggers
 *
 * Translates a chat template's triggers into the two forms GbnfGrammar checks as tokens arrive:
 * token ids compared against the emitted id, and literal words searched in the emitted bytes.
 * This needs the vocabulary, so it happens here rather than in the server.
 *
 * A word is tokenised with special tokens recognised, since template markers such as
 * `<tool_call>` are usually special tokens. If it is one token it becomes an id trigger;
 * otherwise it is matched as bytes, because the tokeniser may split it differently depending on
 * context. A wrong translation fails silently: the trigger never fires and the call comes out
 * unconstrained.
 *
 * Regular-expression triggers are refused at admission, naming the pattern; their matcher would
 * run over the reply once per token on the scheduler thread. */
static int resolve_triggers(const VocabView& vocab,
                            const std::vector<SamplingParams::GrammarTrigger>& triggers,
                            std::vector<std::string>* words, std::vector<int32_t>* tokens,
                            std::string* err) {
    words->clear();
    tokens->clear();
    for (const auto& t : triggers) {
        if (t.value.empty()) continue;
        std::vector<int32_t> ids;
        switch (t.type) {
            case 0:      /* token: the spelling of exactly one token */
            case 1: {    /* word */
                const int st = vocab.tokenize_special(t.value, &ids);
                if (st < 0) {
                    if (err) *err = "grammar trigger '" + t.value + "' could not be tokenised";
                    return st;
                }
                if (ids.size() == 1) { tokens->push_back(ids[0]); break; }
                if (t.type == 0) {
                    if (err) *err = "grammar trigger token '" + t.value +
                                    "' is not a single token in this vocabulary";
                    return RAD_E_INVAL;
                }
                words->push_back(t.value);
                break;
            }
            case 2:      /* pattern */
            case 3:      /* pattern_full */
                if (err) *err = "this chat template triggers its tool-call grammar on the "
                                "regular expression '" + t.value + "'; only word and token "
                                "triggers are supported";
                return RAD_E_UNSUPPORTED;
            default:
                if (err) *err = "unknown grammar trigger type " + std::to_string(t.type);
                return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* A lazy grammar waits for a trigger before it constrains anything, so one without a trigger
 * would leave the whole reply unconstrained. */
static const char* const kLazyNoTrigger =
    "a lazy grammar needs at least one word or token trigger; without one it never constrains";

/* ================================================================== small helpers */

/* Copy this rank's window out of a whole-vocabulary bitmask. The engine splits the head on a
 * multiple of 32, so `vocab_off` is normally word-aligned; the bitwise path covers any other
 * shard boundary, where a misaligned copy would admit the wrong tokens. */
void Sampler::shard_mask(uint32_t* dst) const {
    const int64_t full = (int64_t)mask_full_.size();
    std::memset(dst, 0, (size_t)mask_words_ * 4);
    if ((cfg_.vocab_off & 31) == 0) {
        const int64_t w0 = cfg_.vocab_off >> 5;
        const int64_t n  = std::min(mask_words_, full - w0);
        if (n > 0) std::memcpy(dst, mask_full_.data() + w0, (size_t)n * 4);
        return;
    }
    for (int64_t i = 0; i < cfg_.n_vocab_local; ++i) {
        const int64_t g = cfg_.vocab_off + i;
        if (g >= cfg_.n_vocab) break;
        if (mask_full_[(size_t)(g >> 5)] & (1u << (uint32_t)(g & 31)))
            dst[i >> 5] |= 1u << (uint32_t)(i & 31);
    }
}

/* rad_runtime.h's RAD_B / RAD_W macros are C compound literals, legal C++ only as an extension,
 * and core code builds -Wall -Wextra clean, so operands are built the ordinary way. */
/* Operands are always narrowed to this step's rows. `M` is only the band's upper bound, so
 * kernels take the row count from the operand (arch/common/rad_arch.h); at the declared extent,
 * max_seqs * (max_spec + 1), every stage would do a vocabulary-wide reduction for each empty row.
 *
 * Every buffer here has sampled positions as dim 0, and its strides are products of the trailing
 * dimensions, so shortening dim 0 leaves them correct. */
static RadOperand opd_rows(rad_buf h, int64_t rows) {
    RadOperand o{};
    o.kind = RAD_OPK_BUF;
    o.handle = h;
    o.rows = rows;
    return o;
}

/* An absent optional operand. Operands are positional and the runtime checks an issue against
 * the schema's operand count, so a missing trailing optional must be passed as this. */
static RadOperand opd_none() {
    RadOperand o{};
    o.kind = RAD_OPK_NONE;
    return o;
}

static int64_t rows_for(const SamplerConfig& c) {
    /* One row per sampled position: a verify samples n_spec+1 positions per sequence, each with
     * its own RNG coordinate. Sized for the worst step, as the arena is (spec §3.1). */
    return c.max_seqs * (int64_t)(c.max_spec + 1);
}

/* ================================================================== declare */

int Sampler::declare(RadBuilder* b, const SamplerConfig& cfg) {
    if (!b) return RAD_E_INVAL;
    if (cfg.n_vocab <= 0 || cfg.max_seqs <= 0) {
        RAD_ERR("sampler: n_vocab=%lld max_seqs=%lld is not a configuration",
                (long long)cfg.n_vocab, (long long)cfg.max_seqs);
        return RAD_E_INVAL;
    }

    cfg_ = cfg;
    /* A value other than unset/0/1 is refused, not read as off: a deployment that asked for the
     * coupling and got a typo must not be served the uncoupled path while believing otherwise. */
    if (!mtp_coupled_config_valid()) {
        RAD_ERR("RADIANCE_MTP_COUPLED accepts 0|1 (or unset); refusing the start rather than "
                "serving the uncoupled path");
        return RAD_E_INVAL;
    }
    couple_ = mtp_coupled_enabled();
    if (cfg_.n_vocab_local <= 0) cfg_.n_vocab_local = cfg_.n_vocab / cfg_.world_size;

    /* The staged mask covers this rank's shard; the grammar's covers the whole vocabulary.
     * `sample_mask` indexes its bitmask from bit zero over `n_vocab_local`, so handing rank 1 the
     * full mask would apply rank 0's token admissibility to it. The grammar therefore fills
     * `mask_full_` (where the "nothing admissible anywhere" check belongs) and only this rank's
     * window is staged. */
    mask_words_ = grammar_mask_words(cfg_.n_vocab_local);
    mask_full_.assign((size_t)grammar_mask_words(cfg_.n_vocab), 0u);

    const int64_t rows = rows_for(cfg_);
    const int64_t nv   = cfg_.n_vocab_local;
    if (vocab_parallel() && cfg_.max_candidates * cfg_.world_size > kSampleMergeCandidates) {
        const int64_t cap = kSampleMergeCandidates / cfg_.world_size;
        RAD_INFO("sampler: candidate width %lld a rank -> %lld: the merge of %d ranks' candidate "
                 "sets holds %lld pairs at most (top_k above %lld is refused per request)",
                 (long long)cfg_.max_candidates, (long long)cap, cfg_.world_size,
                 (long long)kSampleMergeCandidates, (long long)cap);
        cfg_.max_candidates = cap;
    }
    const int64_t ncand = cfg_.max_candidates;

    /* ---------------------------------------------------------- buffers */
    /* The staging planes are PERSIST: they are filled by an upload before any op is issued, which
     * the planner's liveness over the op list cannot see, so as TRANSIENT they would be
     * overwritten. */
    auto decl = [&](const char* name, uint32_t dt, int64_t d0, int64_t d1, int kind) -> rad_buf {
        RadBufDecl d{};
        d.dtype = dt;
        d.rank  = 2;
        d.shape[0] = d0;
        d.shape[1] = d1;
        d.kind = kind;
        d.domain = RAD_DOMAIN_DEVICE;
        return rad_decl_buffer(b, name, &d);
    };

    buf_params_ = decl("sampler.params", RAD_U32, rows,
                       (int64_t)(sizeof(DeviceSampleParams) / 4), RAD_BUF_PERSIST);
    buf_params_late_ = decl("sampler.params_late", RAD_U32, rows,
                            (int64_t)(sizeof(DeviceSampleParams) / 4), RAD_BUF_PERSIST);
    buf_hist_   = decl("sampler.history", RAD_I32, rows, cfg_.max_hist, RAD_BUF_PERSIST);
    /* One byte per history token, nonzero where the token is a whole DRY sequence breaker (never
     * penalised). Resolved on the host, which has the vocabulary. */
    buf_brk_    = decl("sampler.dry_breakers", RAD_U8, rows, cfg_.max_hist, RAD_BUF_PERSIST);
    buf_mask_   = decl("sampler.mask", RAD_U32, rows, mask_words_, RAD_BUF_PERSIST);
    /* THE COUPLED STATE PLANE, one row per sampled position PERSIST because it is filled by an upload
     * before the run phase issues anything. A draft round's state is bound to the verify row that
     * follows it, not to the request, which is why it rides the sampler's own row array. */
    if (couple_) {
        buf_couple_ = decl("sampler.mtp_couple", RAD_U32, rows, RAD_MTP_COUPLE_STATE_WORDS,
                           RAD_BUF_PERSIST);
        if (!buf_couple_) return RAD_E_NOMEM;
    }
    buf_cand_i_ = decl("sampler.cand_idx", RAD_I32, rows, ncand, RAD_BUF_TRANSIENT);
    buf_cand_v_ = decl("sampler.cand_val", RAD_F32, rows, ncand, RAD_BUF_TRANSIENT);
    if (vocab_parallel()) {
        /* sample_topk packs this rank's candidates into `pairs` as (u32 global id, f32 logit);
         * the all-gather lays every rank's pairs for a row side by side, so the merge reads one
         * row at a single stride. */
        buf_pairs_  = decl("sampler.cand_pairs", RAD_U32, rows, ncand * 2, RAD_BUF_TRANSIENT);
        buf_gather_ = decl("sampler.cand_gathered", RAD_U32, rows,
                           ncand * cfg_.world_size * 2, RAD_BUF_TRANSIENT);
    }

    /* The logprobs summary is PERSIST too: no declared op reads it (rank 0 copies it to the host
     * behind the pick), so liveness would let the candidate planes reuse its bytes. */
    buf_lp_ = decl("sampler.logprobs", RAD_F32, rows, lp_width(), RAD_BUF_PERSIST);
    if (vocab_parallel())
        buf_lp_g_ = decl("sampler.logprobs_gathered", RAD_F32, rows,
                         lp_width() * cfg_.world_size, RAD_BUF_PERSIST);

    if (!buf_params_ || !buf_params_late_ || !buf_hist_ || !buf_brk_ || !buf_mask_ ||
        !buf_cand_i_ || !buf_cand_v_ || !buf_lp_ ||
        (vocab_parallel() && (!buf_pairs_ || !buf_gather_ || !buf_lp_g_))) {
        RAD_ERR("sampler: a staging buffer failed to declare");
        return RAD_E_NOMEM;
    }

    /* ---------------------------------------------------------- ops */
    /* Every op is declared unconditionally and may fail to resolve; check_request refuses the
     * requests that need a missing one. `M` is ranged over sampled positions, so a kernel may
     * specialise for one row versus a batch of 256 (spec §2.2). */
    auto op_vocab = [&](const char* name) -> rad_op {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 2, nullptr, 0);
    };
    /* `vocab_off` for the three stages that map between token ids and this rank's columns:
     * top-k (column to id), penalties and DRY (history id to column). Declared only when the
     * vocabulary is sharded. */
    auto op_vocab_off = [&](const char* name) -> rad_op {
        if (!vocab_parallel())
            return op_vocab(name);
        RadParam p[3] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
            RadParam{ "vocab_off", RAD_P_INT, (long long)cfg_.vocab_off, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 3, nullptr, 0);
    };
    /* Stages after top-k work on the candidate set, so their geometry is the candidate width. */
    auto op_cand = [&](const char* name) -> rad_op {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_cand", RAD_P_INT, (long long)ncand, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 2, nullptr, 0);
    };

    op_penalties_ = op_vocab_off("sample_penalties");
    op_dry_       = op_vocab_off("sample_dry");
    op_temp_      = op_vocab("sample_temp");
    op_mask_      = op_vocab("sample_mask");
    op_topk_      = op_vocab_off("sample_topk");
    op_argmax_    = op_vocab("sample_argmax");

    op_topp_      = op_cand("sample_topp");
    op_minp_      = op_cand("sample_minp");
    op_typical_   = op_cand("sample_typical");
    op_xtc_       = op_cand("sample_xtc");
    op_pick_      = op_cand("sample_pick");
    /* THE COUPLING'S VERIFY, over the SAME candidate plane the pick just read. A row whose state is
     * not bound keeps the pick's token, so an uncoupled batch issues nothing and pays nothing; a
     * coupled one pays one more launch over its candidate width, not its vocabulary. */
    if (couple_) {
        RadParam pc[3] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_cand", RAD_P_INT, (long long)ncand, 0, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)cfg_.n_vocab, 0, nullptr },
        };
        op_couple_ = rad_decl_op(b, "mtp_couple_verify", pc, 3, nullptr, 0);
        if (!op_couple_) return RAD_E_NOKERNEL;
        /* No rad_op_reads here: the sampler's buffers are shared by ops that declare no lists
         * (topk, topp, pick), so declaring only this one would narrow the candidate planes to
         * its own op and the first sample_topk issue would be refused as outside their
         * lifetime. They stay live for the whole program, as before the coupling. */
    }
    /* The temperature kernel again, over the candidate plane: `n_vocab` is only a row length to
     * it, so the candidate width works. */
    {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)ncand, 0, nullptr },
        };
        op_temp_late_ = rad_decl_op(b, "sample_temp", p, 2, nullptr, 0);
    }

    if (vocab_parallel()) {
        /* Each rank reduces its own shard and only the candidate sets are gathered: 2 KiB at
         * k=64 and world_size=4, versus ~148 MiB for full logits (spec §9). */
        RadParam ag[4] = {
            RadParam{ "world_size", RAD_P_INT, (long long)cfg_.world_size, 0, nullptr },
            RadParam{ "numel", RAD_P_INT, (long long)(rows * ncand * 2), 0, nullptr },
            RadParam{ "dtype", RAD_P_STR, 0, 0, "f32" },
            /* Row-interleaved, so each position's pairs from every rank are adjacent. */
            RadParam{ "row", RAD_P_INT, (long long)(ncand * 2), 0, nullptr },
        };
        op_allgather_ = rad_decl_op(b, "all_gather", ag, 4, nullptr, 0);
        op_merge_     = op_cand("sample_merge_topk");
    }

    /* The raw-distribution summary, at a width that fits every request's top_logprobs. If it
     * does not resolve, check_request refuses only requests that ask for logprobs. */
    {
        RadParam p[4] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
            RadParam{ "k", RAD_P_INT, (long long)kLpK, 0, nullptr },
            RadParam{ "vocab_off", RAD_P_INT, (long long)cfg_.vocab_off, 0, nullptr },
        };
        op_logprobs_ = rad_decl_op(b, "sample_logprobs", p, vocab_parallel() ? 4 : 3, nullptr, 0);
    }
    if (vocab_parallel()) {
        /* Each rank summarises its own shard; rank 0 merges them on the host. Laid out side by
         * side, like the candidate pairs. */
        RadParam ag[4] = {
            RadParam{ "world_size", RAD_P_INT, (long long)cfg_.world_size, 0, nullptr },
            RadParam{ "numel", RAD_P_INT, (long long)(rows * lp_width()), 0, nullptr },
            RadParam{ "dtype", RAD_P_STR, 0, 0, "f32" },
            RadParam{ "row", RAD_P_INT, (long long)lp_width(), 0, nullptr },
        };
        op_lp_gather_ = rad_decl_op(b, "all_gather", ag, 4, nullptr, 0);
        if (!op_lp_gather_) op_logprobs_ = RAD_NULL_HANDLE;
    }

    RAD_TRY(stage_alloc());
    declared_ = true;

    RAD_INFO("sampler: %lld-token vocabulary (%lld this rank), %lld rows, %lld mask words "
             "(%s per row)", (long long)cfg_.n_vocab, (long long)nv, (long long)rows,
             (long long)mask_words_, humanb(mask_words_ * 4).c_str());
    return RAD_OK;
}

int Sampler::stage_alloc() {
    const int64_t rows = rows_for(cfg_);

    /* Pinned: uploaded every step, and a pageable source turns the async copy into a staged one
     * with a host bounce (spec §8). */
    params_stage_ = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    greedy_stage_ = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    late_stage_   = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    greedy_ready_ = false;
    hist_stage_   = (int32_t*)rad_dev_alloc(rows * cfg_.max_hist * 4, RAD_MEM_HOST_PINNED);
    brk_stage_    = (uint8_t*)rad_dev_alloc(rows * cfg_.max_hist, RAD_MEM_HOST_PINNED);
    mask_stage_   = (uint32_t*)rad_dev_alloc(rows * mask_words_ * 4, RAD_MEM_HOST_PINNED);
    /* The coupled rows, pinned: uploaded every coupled step, and a pageable source would turn the
     * async copy into a staged one with a host bounce (spec §8). */
    if (couple_)
        couple_stage_ = (RadMtpCoupleState*)rad_dev_alloc(
            rows * (int64_t)sizeof(RadMtpCoupleState), RAD_MEM_HOST_PINNED);

    /* Logprobs readback; only rank 0 reads it. */
    if (cfg_.rank == 0) {
        const int64_t w = lp_width() * (vocab_parallel() ? cfg_.world_size : 1);
        lp_host_    = (float*)rad_dev_alloc(rows * w * 4, RAD_MEM_HOST_PINNED);
        lp_ci_host_ = (int32_t*)rad_dev_alloc(rows * cfg_.max_candidates * 4, RAD_MEM_HOST_PINNED);
        lp_cv_host_ = (float*)rad_dev_alloc(rows * cfg_.max_candidates * 4, RAD_MEM_HOST_PINNED);
        if (!lp_host_ || !lp_ci_host_ || !lp_cv_host_) {
            RAD_ERR("sampler: could not pin %s for the logprobs readback",
                    humanb(rows * (w + 2 * cfg_.max_candidates) * 4).c_str());
            stage_free();
            return RAD_E_NOMEM;
        }
    }

    if (!params_stage_ || !greedy_stage_ || !late_stage_ || !hist_stage_ || !brk_stage_ ||
        !mask_stage_ || (couple_ && !couple_stage_)) {
        RAD_ERR("sampler: could not pin %s of staging",
                humanb(rows * (3 * (int64_t)sizeof(DeviceSampleParams) + cfg_.max_hist * 5 +
                               mask_words_ * 4)).c_str());
        stage_free();
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

void Sampler::stage_free() {
    if (params_stage_) rad_dev_free(params_stage_, RAD_MEM_HOST_PINNED);
    if (greedy_stage_) rad_dev_free(greedy_stage_, RAD_MEM_HOST_PINNED);
    if (late_stage_)   rad_dev_free(late_stage_,   RAD_MEM_HOST_PINNED);
    if (hist_stage_)   rad_dev_free(hist_stage_,   RAD_MEM_HOST_PINNED);
    if (brk_stage_)    rad_dev_free(brk_stage_,    RAD_MEM_HOST_PINNED);
    if (mask_stage_)   rad_dev_free(mask_stage_,   RAD_MEM_HOST_PINNED);
    if (couple_stage_) rad_dev_free(couple_stage_, RAD_MEM_HOST_PINNED);
    couple_stage_ = nullptr;
    if (lp_host_)      rad_dev_free(lp_host_,      RAD_MEM_HOST_PINNED);
    if (lp_ci_host_)   rad_dev_free(lp_ci_host_,   RAD_MEM_HOST_PINNED);
    if (lp_cv_host_)   rad_dev_free(lp_cv_host_,   RAD_MEM_HOST_PINNED);
    lp_host_      = nullptr;
    lp_ci_host_   = nullptr;
    lp_cv_host_   = nullptr;
    params_stage_ = nullptr;
    greedy_stage_ = nullptr;
    late_stage_   = nullptr;
    cur_stage_    = nullptr;
    hist_stage_   = nullptr;
    brk_stage_    = nullptr;
    mask_stage_   = nullptr;
}

Sampler::~Sampler() { stage_free(); }

/* ================================================================== admission */

/* Whether the two history stages run, shared by admission, request setup and step staging so
 * they agree. As in llama.cpp, a window of 0 turns a stage off and DRY needs a base >= 1. DRY
 * also requires a positive multiplier; a negative one would reward repetition. */
static bool penalties_on(const SamplingParams& sp) {
    return sp.penalty_last_n != 0 &&
           (sp.rep_penalty != 1.0f || sp.freq_penalty != 0.0f || sp.pres_penalty != 0.0f);
}

static bool dry_on(const SamplingParams& sp) {
    return sp.dry_multiplier > 0.0f && sp.dry_base >= 1.0f && sp.dry_penalty_last_n != 0;
}

int Sampler::check_request(const SamplingParams& sp, std::string* why) const {
    auto refuse = [&](const char* what, int st) {
        if (why) *why = what;
        return st;
    };

    /* Each range is tested as "inside", so a NaN fails it. top_p and typical_p are (0, 1], as
     * the server takes them: the chain reads 0 as off, where llama.cpp keeps one candidate, so
     * 0 has no single meaning to honour. */
    if (sp.top_k < 0)                       return refuse("top_k is negative", RAD_E_INVAL);
    if (!(sp.top_p > 0.0f && sp.top_p <= 1.0f))
        return refuse("top_p is outside (0,1]", RAD_E_INVAL);
    if (!(sp.min_p >= 0.0f && sp.min_p <= 1.0f))
        return refuse("min_p is outside [0,1]", RAD_E_INVAL);
    if (!(sp.typical_p > 0.0f && sp.typical_p <= 1.0f))
        return refuse("typical_p is outside (0,1]", RAD_E_INVAL);
    if (!(sp.temp >= 0.0f))                 return refuse("temperature must be >= 0", RAD_E_INVAL);
    if (!(sp.xtc_probability >= 0.0f && sp.xtc_probability <= 1.0f))
        return refuse("xtc_probability is outside [0,1]", RAD_E_INVAL);
    if (!(sp.xtc_threshold >= 0.0f && sp.xtc_threshold <= 1.0f))
        return refuse("xtc_threshold is outside [0,1]", RAD_E_INVAL);
    if (sp.top_k > cfg_.max_candidates) {
        if (why)
            *why = "top_k " + std::to_string(sp.top_k) + " is wider than the sampler's candidate "
                   "width of " + std::to_string(cfg_.max_candidates) + " tokens";
        return RAD_E_INVAL;
    }
    /* The penalty divides a positive logit by rep_penalty, so 0 would make it infinite. */
    if (penalties_on(sp) && !(sp.rep_penalty > 0.0f))
        return refuse("rep_penalty must be > 0", RAD_E_INVAL);
    /* The DRY kernels read an allowed length below 1 as 2; refused rather than rewritten. */
    if (dry_on(sp) && sp.dry_allowed_length < 1)
        return refuse("dry_allowed_length must be >= 1", RAD_E_INVAL);
    if (sp.n_probs > RAD_MAX_LOGPROBS) {
        if (why) *why = "at most " + std::to_string(RAD_MAX_LOGPROBS) +
                        " logprobs alternatives may be asked for";
        return RAD_E_INVAL;
    }
    /* Each breaker is a full-vocabulary scan on the scheduler thread (see kDryMaxBreakers).
     * Refused rather than truncated, which would silently change the output. */
    if (dry_on(sp) && sp.dry_seq_breakers.size() > kDryMaxBreakers) {
        if (why)
            *why = "dry_sequence_breakers has " + std::to_string(sp.dry_seq_breakers.size()) +
                   " entries; at most " + std::to_string(kDryMaxBreakers) + " are accepted";
        return RAD_E_INVAL;
    }

    /* An unresolved sampler is a named refusal, never a silent skip. */
    auto need = [&](bool wanted, rad_op h, const char* n) -> const char* {
        return (wanted && !h) ? n : nullptr;
    };
    /* Under a split vocabulary, or when it asks for logprobs, a greedy request is staged as a
     * top-1 chain row (see build_step), so it needs the chain ops (temperature, top-k, pick)
     * rather than the argmax. run() issues the temperature stage only when some chain row has a
     * temperature other than 0 or 1, the two the stage leaves as they are. */
    const bool lp = sp.n_probs >= 0;
    const bool greedy_chain = sp.greedy() && (vocab_parallel() || lp);
    const char* missing = nullptr;
    if (!missing) missing = need(lp, op_logprobs_, "sample_logprobs");
    if (!missing) missing = need(sp.greedy() && !greedy_chain, op_argmax_, "sample_argmax");
    if (!missing) missing = need(!sp.greedy() || greedy_chain, op_temp_, "sample_temp");
    if (!missing) missing = need(!sp.greedy() || greedy_chain, op_pick_, "sample_pick");
    /* Every chain row needs top-k, whatever its top_k: it writes the candidate set the later
     * stages read. top_k = 0 means the whole candidate width (max_candidates). */
    if (!missing) missing = need(!sp.greedy() || greedy_chain, op_topk_, "sample_topk");
    if (!missing) missing = need(sp.top_p < 1.0f, op_topp_, "sample_topp");
    if (!missing) missing = need(sp.min_p > 0.0f, op_minp_, "sample_minp");
    if (!missing) missing = need(sp.typical_p < 1.0f, op_typical_, "sample_typical");
    if (!missing) missing = need(sp.xtc_probability > 0.0f, op_xtc_, "sample_xtc");
    if (!missing) missing = need(!sp.greedy() && sp.temp_last, op_temp_late_,
                                 "sample_temp (over the candidates, for temperature last)");
    if (!missing) missing = need(dry_on(sp), op_dry_, "sample_dry");
    if (!missing) missing = need(!sp.grammar.empty() || sp.thinking_budget >= 0, op_mask_,
                                 "sample_mask");
    if (!missing) missing = need(penalties_on(sp), op_penalties_, "sample_penalties");
    if (missing) {
        if (why) *why = std::string("no kernel resolved for '") + missing + "'";
        return RAD_E_NOKERNEL;
    }

    if (!sp.grammar.empty() && !vocab_)
        return refuse("a grammar was requested before the vocabulary was attached", RAD_E_STATE);

    /* Resolved here as well as in begin_request, so unsupported triggers are refused at
     * admission rather than cancelling the request on the scheduler thread. */
    if (!sp.grammar.empty() && (sp.grammar_lazy || !sp.grammar_triggers.empty())) {
        std::vector<std::string> words;
        std::vector<int32_t>     tokens;
        std::string              e;
        const int st = resolve_triggers(*vocab_, sp.grammar_triggers, &words, &tokens, &e);
        if (st < 0) {
            if (why) *why = e;
            return st;
        }
        if (sp.grammar_lazy && words.empty() && tokens.empty())
            return refuse(kLazyNoTrigger, RAD_E_INVAL);
    }

    return RAD_OK;
}

int Sampler::begin_request(uint64_t req_id, const SamplingParams& sp, std::string* err) {
    SeqState st;

    if (!sp.grammar.empty()) {
        if (!vocab_) {
            if (err) *err = "no vocabulary attached";
            return RAD_E_STATE;
        }
        GrammarSpec spec;
        spec.kind = GrammarKind::Gbnf;
        spec.text = sp.grammar;
        spec.root = "root";
        spec.lazy = sp.grammar_lazy;
        /* Only an eager grammar replays the generation prompt prefix: its root describes the
         * whole assistant turn, which opens with text already in the prompt. A lazy grammar's
         * root starts at its trigger (e.g. `<tool_call>`) and would fail on that prefix. */
        spec.prefix = sp.grammar_lazy ? std::string() : sp.grammar_prefix;
        RAD_TRY(resolve_triggers(*vocab_, sp.grammar_triggers, &spec.trigger_words,
                                 &spec.trigger_tokens, err));
        RAD_TRY(grammar_create(spec, *vocab_, &st.grammar, err));
    }

    /* Breakers are optional; DRY runs without them, as llama.cpp's does. They only cap how far
     * back a repeat is counted and exempt whole-breaker tokens. */
    if (dry_on(sp) && !sp.dry_seq_breakers.empty()) {
        if (!vocab_) {
            if (err) *err = "DRY needs a vocabulary to resolve its sequence breakers";
            return RAD_E_STATE;
        }
        /* Once per request: a full vocabulary scan per breaker costs milliseconds at a 151K
         * vocab. */
        const int s = dry_prepare_breakers(*vocab_, sp.dry_seq_breakers, &st.dry,
                                           kDryBreakerMaxTail);
        if (s < 0) {
            if (err) *err = "a DRY sequence breaker could not be resolved against the vocabulary";
            return s;
        }
    }

    st.think.start(sp.reasoning);
    seqs_[req_id] = std::move(st);
    return RAD_OK;
}

void Sampler::end_request(uint64_t req_id) { seqs_.erase(req_id); }
bool Sampler::has_request(uint64_t req_id) const { return seqs_.find(req_id) != seqs_.end(); }

std::vector<uint64_t> Sampler::take_failed() {
    std::vector<uint64_t> v;
    v.swap(failed_);
    return v;
}

int Sampler::accept_token(uint64_t req_id, int32_t token) {
    auto it = seqs_.find(req_id);
    if (it == seqs_.end()) return RAD_OK;      /* no grammar, nothing to advance */
    if (!it->second.grammar) return RAD_OK;
    /* Timed at Debug, like build_step: runs once per accepted token per rank on the step thread,
     * so it is wall time on every constrained token. */
    if ((int)log_level() < (int)Log::Debug) return it->second.grammar->accept_token(token);
    const auto t0 = std::chrono::steady_clock::now();
    const int s = it->second.grammar->accept_token(token);
    accept_us_ += std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0).count();
    if (++accept_n_ == 256) {
        RAD_DEBUG("sampler: grammar accept_token %.3f ms mean over 256", accept_us_ / 256000.0);
        accept_us_ = 0; accept_n_ = 0;
    }
    return s;
}

/* A constraining grammar does not rule out speculation: build_step masks draft position i
 * against the state after drafts 0..i-1 and then rewinds. Only a backend that cannot rewind is
 * refused. */
bool Sampler::allows_speculation(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    if (it == seqs_.end() || !it->second.grammar) return true;
    return !it->second.grammar->constrains() || it->second.grammar->can_speculate();
}

bool Sampler::speculation_fixed(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    return it != seqs_.end() && (!it->second.grammar || it->second.grammar->can_speculate());
}

/* ================================================================== the step path */

/* BIND THE DRAFT'S OWN STATES. Every row must name this request and the round it belongs to: a state
 * that arrived under another request's id, or at another round, would be coupled against a law this
 * step never drew, and the verify would silently apply the wrong Q. */
int Sampler::set_mtp_couple(uint64_t req_id, const RadMtpCoupleState* states, int n) {
    if (!couple_ || n < 0 || n > RAD_MTP_COUPLE_MAX_ROUNDS || (n && !states)) return RAD_E_STATE;
    const auto it = seqs_.find(req_id);
    if (it == seqs_.end()) return RAD_E_NOTFOUND;
    for (int i = 0; i < n; ++i)
        if (states[i].draw.seq != req_id || states[i].round != i + 1) return RAD_E_STATE;
    SeqState& s = it->second;
    for (int i = 0; i < n; ++i) s.couple[i] = states[i];
    s.couple_n = n;
    return RAD_OK;
}

int Sampler::build_step(const std::vector<Request*>& reqs) {
    if (!declared_) return RAD_E_STATE;

    const auto t_build = std::chrono::steady_clock::now();
    const int64_t cap = rows_for(cfg_);
    n_seq_ = 0;
    any_brk_ = false;
    any_late_ = false;
    any_couple_ = false;
    cur_stage_ = params_stage_;
    lp_rows_.clear();

    /* DRY's host half for one row whose history `h[0, take)` is staged. Over the window the
     * kernel uses (the tail of the row, `dry_penalty_last_n` long when positive) it computes the
     * repeat limit from the most recent breaker (llama.cpp's step 1) and marks whole-breaker
     * tokens, which are never penalised.
     *
     * Returns false when a breaker closer to the end than `allowed_length` rules DRY out for the
     * row. A row without breakers still gets zero marks, because the plane is shared by the
     * batch. */
    auto stage_dry = [&](DeviceSampleParams& p, const int32_t* h, int64_t take,
                         const DryBreakers* brk, int64_t row) -> bool {
        const int64_t last_n  = p.dry_penalty_last_n;
        const int64_t win     = last_n > 0 ? std::min<int64_t>(take, last_n) : take;
        const int64_t w0      = take - win;
        const int32_t allowed = p.dry_allowed_length > 0 ? p.dry_allowed_length : 2;
        uint8_t* f = brk_stage_ + row * cfg_.max_hist + w0;
        if (!brk) {
            std::memset(f, 0, (size_t)win);
            return true;
        }
        if (win <= allowed) return true;     /* the kernel takes its own early out */
        const int rep = dry_rep_limit(*brk, h + w0, (int)win);
        if (rep < allowed) return false;
        p.dry_rep_limit = rep < win ? rep : 0;
        for (int64_t i = 0; i < win; ++i) f[i] = brk->is_single(h[w0 + i]) ? 1 : 0;
        any_brk_ = true;
        return true;
    };

    for (Request* r : reqs) {
        if (!r) continue;
        const SamplingParams& sp = r->sp;
        const int n_pos = r->n_draft + 1;

        if (n_pos > 1 && !allows_speculation(r->id)) {
            RAD_ERR("sampler: request %llu has %d draft tokens and a constraining grammar that "
                    "cannot snapshot its state; the scheduler must decode it a token at a time "
                    "(allows_speculation)", (unsigned long long)r->id, r->n_draft);
            return RAD_E_UNSUPPORTED;
        }

        if (n_seq_ + n_pos > cap) {
            RAD_ERR("sampler: %lld sampled positions exceeds the %lld the arena was sized for",
                    (long long)(n_seq_ + n_pos), (long long)cap);
            return RAD_E_FULL;
        }

        const int64_t n_prompt = (int64_t)r->prompt.size();
        const int64_t n_out    = (int64_t)r->output.size();
        const int64_t n_total  = n_prompt + n_out;
        const bool    pen      = penalties_on(sp);
        const bool    dry      = dry_on(sp);

        auto it = seqs_.find(r->id);
        SeqState* st = it == seqs_.end() ? nullptr : &it->second;
        const DryBreakers* brk = st && !st->dry.empty() ? &st->dry : nullptr;

        /* The window is walked, then rewound: position k is masked against the state after
         * accepting drafts 0..k-1. The push happens even when the grammar is not yet
         * constraining, because a draft can fire a lazy grammar's trigger mid-window.
         *
         * `dead` is not a failure: a draft the grammar forbids will be rejected at verification
         * and every later position discarded, so masking stops for the rest of the window. */
        GrammarBackend* gr = st && st->grammar ? st->grammar.get() : nullptr;
        const bool walk = gr && n_pos > 1;
        if (walk) gr->push_state();
        bool dead = false;

        /* The reasoning budget is walked over the window like the grammar: position k is
         * judged by the committed reply plus drafts 0..k-1. */
        const bool budget = st && sp.thinking_budget >= 0 && sp.reasoning.known();
        ReasoningCount think;
        if (budget) {
            if (st->think_seen > (size_t)n_out) {
                st->think.start(sp.reasoning);
                st->think_seen = 0;
            }
            for (; st->think_seen < (size_t)n_out; ++st->think_seen)
                st->think.push(sp.reasoning, r->output[st->think_seen]);
            think = st->think;
        }

        for (int k = 0; k < n_pos; ++k) {
            if (walk && k > 0 && !dead && gr->accept_token(r->draft[k - 1]) < 0) dead = true;
            if (budget && k > 0) think.push(sp.reasoning, r->draft[k - 1]);
            const int64_t row = n_seq_++;
            /* THE COUPLED ROW. `k` is this sampled position's index in the window, so draft row k
             * verifies against the state the draft that produced it left behind. A row with no
             * bound state -- the bonus row, a greedy row, an uncoupled step -- gets a zeroed one,
             * and the verify leaves the pick's token alone. */
            if (couple_) {
                couple_stage_[row] = RadMtpCoupleState{};
                const auto sit = seqs_.find(r->id);
                if (sit != seqs_.end() && !sp.greedy() && k + 1 < n_pos &&
                    k < sit->second.couple_n) {
                    const RadMtpCoupleState& c = sit->second.couple[k];
                    /* BIND TO THIS ROW: same request and seed, the round this row's draft came
                     * from, the position it samples, and the very token it proposes. Anything else
                     * is a stale state from another step, and coupling against it would apply a law
                     * this step never drew. The rule is rad_mtp_couple_binds (abi/rad_spechub.h),
                     * the same one Engine::bind_couple writes the position by. */
                    if (rad_mtp_couple_binds(c, r->id, sp.seed, (uint64_t)n_out, k,
                                             r->draft ? r->draft[k] : -1, cfg_.n_vocab)) {
                        couple_stage_[row] = c;
                        any_couple_ = true;
                    }
                }
            }

            DeviceSampleParams& p = params_stage_[row];
            std::memset(&p, 0, sizeof p);

            p.seed = sp.seed;
            /* Output position: draft position k is token n_out + k, so its draw is the same with
             * or without speculation. */
            p.pos = (uint64_t)(n_out + k);

            p.temp        = sp.temp;
            p.top_p       = sp.top_p;
            p.min_p       = sp.min_p;
            p.typical_p   = sp.typical_p;
            p.rep_penalty = sp.rep_penalty;
            p.freq_penalty = sp.freq_penalty;
            p.pres_penalty = sp.pres_penalty;
            p.xtc_probability = sp.xtc_probability;
            p.xtc_threshold   = sp.xtc_threshold;
            p.dry_multiplier  = sp.dry_multiplier;
            p.dry_base        = sp.dry_base;
            p.top_k              = sp.top_k;
            p.penalty_last_n     = sp.penalty_last_n;
            p.dry_allowed_length = sp.dry_allowed_length;
            p.dry_penalty_last_n = sp.dry_penalty_last_n;
            p.mask_row = -1;

            /* Greedy over a split vocabulary is a top-1 chain row, not an argmax: sample_argmax
             * writes a column of this rank's shard, not a token id. Top-k plus the merge produce
             * global ids, and the pick over one candidate is deterministic. A greedy row asking
             * for logprobs is staged the same way, because logprobs_for reads the drawn value
             * from the candidate plane; the argmax and top-k share a tie rule, so the token is
             * unchanged.
             *
             * Such a row does not carry RAD_SP_GREEDY: candidate stages skip flagged rows, so the
             * pick does not overwrite the argmax's token. */
            const bool want_lp = sp.n_probs >= 0;
            if (sp.greedy()) {
                if (vocab_parallel() || want_lp) p.top_k = 1;
                else                             p.flags |= RAD_SP_GREEDY;
            }
            if (want_lp) p.flags |= RAD_SP_LOGPROBS;
            if (sp.xtc_probability > 0.0f) p.flags |= RAD_SP_XTC;
            /* Temperature last: the row's temperature goes in the late row and 1 (skipped) in the
             * early one; otherwise the reverse. Every row gets a late row, because the late stage
             * is issued over the whole batch. */
            const bool late = sp.temp_last && !sp.greedy() && sp.temp != 1.0f;
            DeviceSampleParams& lp = late_stage_[row];
            lp = p;
            lp.temp = late ? sp.temp : 1.0f;
            if (late) {
                p.temp = 1.0f;
                any_late_ = true;
            }

            /* History for the penalties and DRY, staged only for rows that run one of them:
             * prompt, output so far and, at draft position k, the k drafts before it (they are
             * emitted only if accepted, as one token at a time). Oldest first, truncated to
             * max_hist; penalty_last_n still decides how much counts. Each row owns its row of the
             * plane, so hist_off stays 0. */
            if (pen || dry) {
                const int64_t len  = n_total + k;
                const int64_t take = std::min<int64_t>(len, cfg_.max_hist);
                int32_t* h = hist_stage_ + row * cfg_.max_hist;
                for (int64_t i = 0; i < take; ++i) {
                    const int64_t src = len - take + i;
                    h[i] = src < n_prompt ? r->prompt[(size_t)src]
                         : src < n_total  ? r->output[(size_t)(src - n_prompt)]
                                          : r->draft[src - n_total];
                }
                p.hist_len = (int32_t)take;
                if (pen) p.flags |= RAD_SP_PENALTY;
                if (dry && stage_dry(p, h, take, brk, row)) p.flags |= RAD_SP_DRY;
            }

            if (gr && !dead && gr->constrains()) {
                uint32_t* dst = mask_stage_ + row * mask_words_;
                int s;
                /* Mask cost, reported at Debug per 256 masks (mean and worst). It runs on the
                 * scheduler thread while the rank threads wait at the barrier. */
                const bool   inst = (int)log_level() >= (int)Log::Debug;
                const auto   t0   = std::chrono::steady_clock::now();
                /* Memo hits and computed masks are told apart by the grammar's walk counter, not
                 * by duration: a computed mask can be as fast as a memo hit on a large
                 * vocabulary. */
                const uint64_t w0 = inst ? st->grammar->masks_walked() : 0;
                if (!vocab_parallel()) {
                    s = st->grammar->fill_mask(dst, mask_words_);
                } else {
                    s = st->grammar->fill_mask(mask_full_.data(), (int64_t)mask_full_.size());
                    if (s >= 0) shard_mask(dst);
                }
                if (inst) {
                    const double us = std::chrono::duration<double, std::micro>(
                                          std::chrono::steady_clock::now() - t0).count();
                    mask_us_ += us;
                    mask_worst_us_ = std::max(mask_worst_us_, us);
                    if (st->grammar->masks_walked() != w0) { ++mask_slow_; mask_walk_us_ += us; }
                    else                                     ++mask_fast_;
                    if (++mask_us_n_ == 256) {
                        /* Mean per computed mask is what a new state costs; the overall mean
                         * is dominated by memo hits. */
                        RAD_DEBUG("sampler: grammar mask %.2f ms mean, %.2f ms worst over 256; "
                                  "%lld memo hits, %lld computed at %.2f ms mean (%lld vocab, "
                                  "%lld words)",
                                  mask_us_ / 256000.0, mask_worst_us_ / 1000.0,
                                  (long long)mask_fast_, (long long)mask_slow_,
                                  mask_slow_ ? mask_walk_us_ / (1000.0 * (double)mask_slow_) : 0.0,
                                  (long long)cfg_.n_vocab, (long long)mask_words_);
                        mask_us_ = 0; mask_us_n_ = 0; mask_worst_us_ = 0;
                        mask_fast_ = 0; mask_slow_ = 0; mask_walk_us_ = 0.0;
                    }
                }
                if (s < 0 && k > 0) {
                    /* At a draft position this only means a draft is wrong; verification
                     * rejects at or before here, so stop masking the rest of the window. */
                    dead = true;
                } else if (s < 0) {
                    /* A dead grammar fails its request, not the step: an error status would stop
                     * the step loop for every client. The row stays unmasked and the engine
                     * cancels the request. */
                    RAD_WARN("sampler: request %llu has no admissible token under its grammar; "
                             "failing that request", (unsigned long long)r->id);
                    failed_.push_back(r->id);
                } else {
                    p.mask_row = (int32_t)row;
                    p.flags |= RAD_SP_MASK;
                    if ((int)log_level() >= (int)Log::Trace) {
                        /* Tokens the grammar allows on this rank. Zero is legal under sharding,
                         * when the admissible set lives on another rank. */
                        int64_t bits = 0;
                        for (int64_t w = 0; w < mask_words_; ++w)
                            bits += __builtin_popcount(dst[w]);
                        RAD_TRACE("sampler: req %llu row %lld mask allows %lld of %lld local "
                                  "tokens (vocab_off %lld)", (unsigned long long)r->id,
                                  (long long)row, (long long)bits,
                                  (long long)cfg_.n_vocab_local, (long long)cfg_.vocab_off);
                    }
                }
            }

            if (budget && think.inside() && think.body >= sp.thinking_budget)
                force_close(p, row, sp.reasoning.close);

            if (want_lp) {
                /* Temperature the candidate value carries: early and late, each counted only
                 * where its kernel applies it (temp > 0 and not 1). */
                auto t_of = [](float t) { return t > 0.0f && t != 1.0f ? t : 1.0f; };
                LpRow L;
                L.row   = row;
                L.n_top = sp.n_probs;
                L.temp  = t_of(p.temp) * t_of(late_stage_[row].temp);
                L.p     = p;
                if (p.hist_len > 0) {
                    const int32_t* h = hist_stage_ + row * cfg_.max_hist;
                    L.hist.assign(h, h + p.hist_len);
                }
                L.dry = brk;
                lp_rows_.push_back(std::move(L));
            }
        }

        /* Rewind the walk; accepted tokens are applied by accept_token after verification. */
        if (walk) RAD_TRY(gr->rollback());
    }

    /* Whole build time, which is what the rank threads wait for at the barrier. */
    if ((int)log_level() >= (int)Log::Debug) {
        build_us_ += std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - t_build).count();
        if (++build_n_ == 256) {
            RAD_DEBUG("sampler: build_step %.3f ms mean over 256 (%lld rows)",
                      build_us_ / 256000.0, (long long)n_seq_);
            build_us_ = 0; build_n_ = 0;
        }
    }
    return RAD_OK;
}

void Sampler::force_close(DeviceSampleParams& p, int64_t row, int32_t close) {
    uint32_t* dst = mask_stage_ + row * mask_words_;
    const int64_t at = (int64_t)close - cfg_.vocab_off;
    const bool local = at >= 0 && at < cfg_.n_vocab_local;
    if (p.flags & RAD_SP_MASK) {
        /* Check the grammar's mask. Under a split vocabulary use mask_full_, since the close may
         * be another rank's column. */
        const bool admitted =
            vocab_parallel()
                ? close < (int64_t)mask_full_.size() * 32 &&
                      (mask_full_[(size_t)(close >> 5)] >> (uint32_t)(close & 31)) & 1u
                : local && (dst[at >> 5] >> (uint32_t)(at & 31)) & 1u;
        if (!admitted) return;
    }
    /* One admissible token, on the rank that holds its column. Other ranks mask the whole row,
     * which the candidate merge reads as no candidates from them. */
    std::memset(dst, 0, (size_t)mask_words_ * 4);
    if (local) dst[at >> 5] |= 1u << (uint32_t)(at & 31);
    p.mask_row = (int32_t)row;
    p.flags |= RAD_SP_MASK;
}

int Sampler::build_greedy(int64_t n) {
    if (!declared_) return RAD_E_STATE;
    if (n < 0) return RAD_E_INVAL;
    const int64_t cap = rows_for(cfg_);
    if (n > cap) {
        RAD_ERR("sampler: %lld greedy rows exceeds the %lld the arena was sized for",
                (long long)n, (long long)cap);
        return RAD_E_FULL;
    }
    n_seq_ = n;
    any_brk_ = false;
    any_late_ = false;
    any_couple_ = false;
    cur_stage_ = greedy_stage_;
    if (greedy_ready_) return RAD_OK;
    for (int64_t i = 0; i < cap; ++i) {
        DeviceSampleParams& p = greedy_stage_[i];
        std::memset(&p, 0, sizeof p);
        p.top_k    = 1;
        p.mask_row = -1;
        /* Over a split vocabulary a greedy row is an unflagged top-1 chain row (see
         * build_step). */
        p.flags    = vocab_parallel() ? 0 : RAD_SP_GREEDY;
    }
    greedy_ready_ = true;
    return RAD_OK;
}

int Sampler::run(RadCtx* c, rad_buf logits, rad_buf out_tokens) {
    if (!declared_) return RAD_E_STATE;
    if (!c || !logits || !out_tokens) return RAD_E_INVAL;
    if (n_seq_ == 0) return RAD_OK;

    const RadStream s = rad_stream(c);

    bool any_mask = false, any_greedy = false, any_chain = false;
    bool any_penalty = false, any_dry = false, any_xtc = false;
    bool any_topp = false, any_minp = false, any_typ = false, any_temp = false;
    for (int64_t i = 0; i < n_seq_; ++i) {
        const DeviceSampleParams& p = cur_stage_[i];
        any_mask    |= (p.flags & RAD_SP_MASK)    != 0;
        any_penalty |= (p.flags & RAD_SP_PENALTY) != 0;
        any_dry     |= (p.flags & RAD_SP_DRY)     != 0;
        any_xtc     |= (p.flags & RAD_SP_XTC)     != 0;
        if (p.flags & RAD_SP_GREEDY) any_greedy = true;
        else {
            any_chain = true;
            /* sample_temp leaves a row at temperature <= 0 or exactly 1 as it is */
            any_temp |= p.temp > 0.0f && p.temp != 1.0f;
            any_topp |= p.top_p < 1.0f;
            any_minp |= p.min_p > 0.0f;
            any_typ  |= p.typical_p < 1.0f;
        }
    }

    /* A greedy-flagged row under a split vocabulary is a staging bug (the argmax would return a
     * shard column), so it is refused. */
    if (any_greedy && vocab_parallel()) {
        RAD_ERR("sampler: a greedy-flagged row reached a split vocabulary, where the argmax "
                "returns a column of this rank's shard rather than a token id");
        return RAD_E_STATE;
    }

    /* Async copies on the compute stream, only of what some row reads. The mask is 19 KiB per
     * constrained sequence at a 151K vocab, which keeps a host-side grammar affordable
     * (spec §13). */
    RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_params_), cur_stage_,
                             n_seq_ * (int64_t)sizeof(DeviceSampleParams), s));
    /* Late rows only for a build_step build; build_greedy clears any_late_. */
    const bool late = any_late_;
    if (late)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_params_late_), late_stage_,
                                 n_seq_ * (int64_t)sizeof(DeviceSampleParams), s));
    if (any_penalty || any_dry)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_hist_), hist_stage_,
                                 n_seq_ * cfg_.max_hist * 4, s));
    const bool pass_brk = any_dry && any_brk_;
    if (pass_brk)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_brk_), brk_stage_,
                                 n_seq_ * cfg_.max_hist, s));
    if (any_mask)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_mask_), mask_stage_,
                                 n_seq_ * mask_words_ * 4, s));
    if (any_couple_)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_couple_), couple_stage_,
                                 n_seq_ * (int64_t)sizeof(RadMtpCoupleState), s));

    const RadOperand o_logits = opd_rows(logits, n_seq_);
    const RadOperand o_params = opd_rows(buf_params_, n_seq_);
    const RadOperand o_hist = opd_rows(buf_hist_, n_seq_);
    const RadOperand o_brk = opd_rows(buf_brk_, n_seq_);
    const RadOperand o_mask = opd_rows(buf_mask_, n_seq_);
    const RadOperand o_ci = opd_rows(buf_cand_i_, n_seq_);
    const RadOperand o_cv = opd_rows(buf_cand_v_, n_seq_);
    const RadOperand o_tok = opd_rows(out_tokens, n_seq_);

    auto issue = [&](rad_op op, std::initializer_list<RadOperand> ops) -> int {
        std::vector<RadOperand> v(ops);
        return rad_issue(c, op, v.data(), (int)v.size(), n_seq_);
    };

    /* The logprobs summary reads the raw logits, before any stage writes them in place. Only for
     * a build_step build with logprobs rows: a draft pass must not overwrite the summary rank 0
     * is about to copy back. */
    const bool lp = !lp_rows_.empty() && cur_stage_ == params_stage_ && op_logprobs_;
    const RadOperand o_lp = opd_rows(buf_lp_, n_seq_);
    if (lp) {
        RAD_TRY(issue(op_logprobs_, { o_logits, o_params, o_lp }));
        if (vocab_parallel())
            RAD_TRY(issue(op_lp_gather_, { o_lp, opd_rows(buf_lp_g_, n_seq_) }));
    }

    /* Penalties act on raw logits. DRY gets the breaker marks only when some row has breakers;
     * absent means no token is exempt. */
    if (any_penalty) RAD_TRY(issue(op_penalties_, { o_logits, o_params, o_hist }));
    if (any_dry)
        RAD_TRY(issue(op_dry_, { o_logits, o_params, o_hist, pass_brk ? o_brk : opd_none() }));

    /* Greedy rows take the argmax after penalties and the mask. In a mixed batch both the argmax
     * and the chain run over the whole batch: the argmax writes every row, then the candidate
     * stages skip RAD_SP_GREEDY rows so the pick overwrites only chain rows. One extra logits
     * pass (~148 MiB at batch 256) beats up to 256 launches, one per parameter set. Under vocab
     * parallelism no row is flagged greedy (see build_step). */
    if (any_greedy) {
        if (any_mask) RAD_TRY(issue(op_mask_, { o_logits, o_params, o_mask }));
        RAD_TRY(issue(op_argmax_, { o_logits, o_params, o_tok }));
    }

    if (!any_chain) return RAD_OK;

    /* Only where some row is scaled: a greedy request under a split vocabulary (temperature 0)
     * and a plain T=1 one are left alone, and the launch over the whole vocabulary still costs
     * ~10 us a step (4x R9700). */
    if (any_temp) RAD_TRY(issue(op_temp_, { o_logits, o_params }));
    if (any_mask && !any_greedy)
        RAD_TRY(issue(op_mask_, { o_logits, o_params, o_mask }));

    /* Top-k reduces the vocabulary to the candidate set the rest works on; under tensor parallel
     * it is per shard and the sets are merged below (spec §9). Always issued when the chain runs:
     * top_k = 0 means the full candidate width, and the transient candidate planes hold stale
     * bytes otherwise. */
    if (vocab_parallel()) {
        /* The fifth operand is the pair plane (global ids), which is what is gathered.
         * cand_idx/cand_val come back local and the merge overwrites both. */
        const RadOperand o_p = opd_rows(buf_pairs_, n_seq_);
        const RadOperand o_g = opd_rows(buf_gather_, n_seq_);
        RAD_TRY(issue(op_topk_,      { o_logits, o_params, o_ci, o_cv, o_p }));
        RAD_TRY(issue(op_allgather_, { o_p, o_g }));
        RAD_TRY(issue(op_merge_,     { o_g, o_params, o_ci, o_cv }));
    } else {
        /* No pair plane; passed as absent because the schema has five operands. */
        RAD_TRY(issue(op_topk_, { o_logits, o_params, o_ci, o_cv, opd_none() }));
    }

    if (any_topp) RAD_TRY(issue(op_topp_,    { o_ci, o_cv, o_params }));
    if (any_minp) RAD_TRY(issue(op_minp_,    { o_ci, o_cv, o_params }));
    if (any_typ)  RAD_TRY(issue(op_typical_, { o_ci, o_cv, o_params }));
    if (any_xtc)  RAD_TRY(issue(op_xtc_,     { o_ci, o_cv, o_params }));
    if (late)
        RAD_TRY(issue(op_temp_late_, { o_cv, opd_rows(buf_params_late_, n_seq_) }));

    RAD_TRY(issue(op_pick_, { o_ci, o_cv, o_params, o_tok }));
    /* AFTER the pick, over the same plane: a coupled row replaces the pick's token with the
     * coupling's, and an unbound row leaves it. */
    if (any_couple_)
        RAD_TRY(issue(op_couple_, { o_ci, o_cv, o_params, opd_rows(buf_couple_, n_seq_), o_tok }));

    /* Copy the logprobs rows to the host behind the pick, on rank 0 only (collect_step asks
     * it): the summary and the candidate planes, which logprobs_for reads when the summary lacks
     * the token. Only rows from the first to the last that asked. */
    if (lp && cfg_.rank == 0) {
        lp_lo_ = lp_rows_.front().row;
        lp_hi_ = lp_rows_.back().row + 1;
        const int64_t w = lp_width() * (vocab_parallel() ? cfg_.world_size : 1);
        const int64_t nc = cfg_.max_candidates, n = lp_hi_ - lp_lo_;
        const char* sp = (const char*)rad_buf_ptr(c, vocab_parallel() ? buf_lp_g_ : buf_lp_);
        const char* ci = (const char*)rad_buf_ptr(c, buf_cand_i_);
        const char* cv = (const char*)rad_buf_ptr(c, buf_cand_v_);
        if (!sp || !ci || !cv) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(lp_host_ + lp_lo_ * w, sp + lp_lo_ * w * 4, n * w * 4, s));
        RAD_TRY(rad_memcpy_async(lp_ci_host_ + lp_lo_ * nc, ci + lp_lo_ * nc * 4, n * nc * 4, s));
        RAD_TRY(rad_memcpy_async(lp_cv_host_ + lp_lo_ * nc, cv + lp_lo_ * nc * 4, n * nc * 4, s));
    }
    return RAD_OK;
}

/* ================================================================== logprobs */

float Sampler::raw_logit(const LpRow& L, int32_t token, float v) const {
    const DeviceSampleParams& p = L.p;
    double y = (double)v * (double)L.temp;
    /* DRY's penalty depends only on the token and the window, so the host reference recomputes
     * it on a single candidate at 0, which comes back as minus the penalty. */
    if (p.flags & RAD_SP_DRY) {
        static const DryBreakers kNone;
        CandidateSet s;
        s.c.push_back(Candidate{ token, 0.0f, 0.0f });
        DryParams dp;
        dp.multiplier     = p.dry_multiplier;
        dp.base           = p.dry_base;
        dp.allowed_length = p.dry_allowed_length > 0 ? p.dry_allowed_length : 2;
        dp.penalty_last_n = p.dry_penalty_last_n < 0 ? -1 : p.dry_penalty_last_n;
        host_dry(s, L.hist.data(), (int64_t)L.hist.size(), (int64_t)L.hist.size(), dp,
                 L.dry ? *L.dry : kNone);
        y -= (double)s.c[0].logit;
    }
    /* Penalties inverted: presence and frequency added back, then repetition undone on the
     * logit's side of zero (it never changes sign). Window: the tail of the staged history,
     * penalty_last_n long when positive. */
    if (p.flags & RAD_SP_PENALTY) {
        int64_t n = (int64_t)L.hist.size(), o = 0;
        if (p.penalty_last_n > 0 && n > p.penalty_last_n) { o = n - p.penalty_last_n; n = p.penalty_last_n; }
        int64_t count = 0;
        for (int64_t i = 0; i < n; ++i) count += L.hist[(size_t)(o + i)] == token;
        if (count > 0) {
            y += (double)count * p.freq_penalty + p.pres_penalty;
            y = y > 0.0 ? y * p.rep_penalty : y / p.rep_penalty;
        }
    }
    return (float)y;
}

int merge_logprob_rows(const float* st, int W, int k, double* log_z, int32_t* ids, float* vals,
                       int cap) {
    const int64_t w = 2 + 2 * (int64_t)k;
    auto live = [&](int r) {
        const float m = st[r * w], z = st[r * w + 1];
        return std::isfinite(m) && std::isfinite(z) && z > 0.0f;
    };
    double mx = -INFINITY;
    for (int r = 0; r < W; ++r)
        if (live(r)) mx = std::max(mx, (double)st[r * w]);
    double sum = 0.0;
    for (int r = 0; r < W; ++r)
        if (live(r)) sum += (double)st[r * w + 1] * std::exp((double)st[r * w] - mx);
    *log_z = sum > 0.0 ? mx + std::log(sum) : NAN;

    /* Stack buffer covers up to eight ranks of 21 pairs (168); more falls back to the heap.
     * Runs once per emitted token on collect_step's path. */
    struct Pair { float v; int32_t id; };
    Pair buf[256];
    std::vector<Pair> heap;
    Pair* top = buf;
    if ((int64_t)W * k > (int64_t)(sizeof buf / sizeof buf[0])) {
        heap.resize((size_t)W * (size_t)k);
        top = heap.data();
    }
    int m = 0;
    for (int r = 0; r < W; ++r) {
        if (!live(r)) continue;
        for (int j = 0; j < k; ++j) {
            const int32_t id = (int32_t)st[r * w + 2 + 2 * j];
            const float   v  = st[r * w + 3 + 2 * j];
            if (id >= 0 && std::isfinite(v)) top[m++] = Pair{ v, id };
        }
    }
    std::sort(top, top + m, [](const Pair& a, const Pair& b) {
        return a.v > b.v || (a.v == b.v && a.id < b.id);
    });
    const int n = std::min(cap, m);
    for (int j = 0; j < n; ++j) {
        ids[j]  = top[j].id;
        vals[j] = top[j].v;
    }
    return n;
}

int Sampler::logprobs_for(int64_t row, int32_t token, TokenLogprobs* out) const {
    if (!out || !lp_host_ || row < lp_lo_ || row >= lp_hi_) return RAD_E_STATE;
    const LpRow* L = nullptr;
    for (const LpRow& x : lp_rows_)
        if (x.row == row) { L = &x; break; }
    if (!L) return RAD_E_STATE;

    /* Merge every rank's shard summary. */
    const int W = vocab_parallel() ? cfg_.world_size : 1;
    double log_z = 0.0;
    int32_t ids[kLpK * 8];
    float   vals[kLpK * 8];
    const int n = merge_logprob_rows(lp_host_ + row * W * lp_width(), W, kLpK, &log_z, ids, vals,
                                     (int)(sizeof ids / sizeof ids[0]));
    /* No finite logit (NaN from the model, or all -inf) means no distribution to report. */
    if (!std::isfinite(log_z)) {
        RAD_WARN("sampler: row %lld has no finite logit; its logprobs cannot be computed",
                 (long long)row);
        out->token = -1;
        return RAD_E_INVAL;
    }

    out->token = token;
    out->n_top = std::min(L->n_top, n);
    for (int j = 0; j < out->n_top; ++j) {
        out->top_id[j] = ids[j];
        out->top_lp[j] = (float)((double)vals[j] - log_z);
    }

    /* The token's raw logit: exact if among the largest, else recovered from the value the pick
     * drew it at. */
    double own = NAN;
    for (int j = 0; j < n && std::isnan(own); ++j)
        if (ids[j] == token) own = (double)vals[j];
    if (std::isnan(own)) {
        const int64_t nc = cfg_.max_candidates;
        const int32_t* ci = lp_ci_host_ + row * nc;
        const float*   cv = lp_cv_host_ + row * nc;
        for (int64_t c = 0; c < nc && std::isnan(own); ++c)
            if (ci[c] == token) own = (double)raw_logit(*L, token, cv[c]);
    }
    if (!std::isfinite(own)) {
        RAD_WARN("sampler: row %lld sampled token %d, which has no finite logit on the candidate "
                 "plane it was drawn from; its logprobs cannot be computed", (long long)row, token);
        out->token = -1;
        return RAD_E_INVAL;
    }
    out->logprob = (float)(own - log_z);
    return RAD_OK;
}

}  /* namespace rad */
