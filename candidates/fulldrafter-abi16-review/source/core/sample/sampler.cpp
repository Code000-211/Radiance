/* sampler.cpp -- the sampler driver: declare the sampler ops, build the per-sequence parameter
 * array once per step, issue the chain. No sampler is implemented here; see sampler.h for why.
 */
#include "sampler.h"
#include "gbnf.h"
#include "../../arch/common/rad_spechub_config.h"
#include "../../mine/spechub.h"   /* rad_mtp_couple_state_valid: the kernels' own validator */

#if __has_include("../../arch/common/rad_tree_candidate.h")
#include "../../arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#elif __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rad {

/* ================================================================== grammar triggers
 *
 * A chat template names its triggers the way a person would -- "the call starts at the word
 * <tool_call>" -- while GbnfGrammar wants them in the two forms it checks as tokens arrive: token
 * ids to compare against the id just emitted, and literal words to find in the emitted bytes.
 * The translation needs a vocabulary, so it happens here and not in the server.
 *
 * A WORD IS A TOKEN WHEN THE VOCABULARY SAYS SO. The word is tokenised with special tokens
 * recognised, since a template's markers are usually special tokens and a split of `<tool_call>`
 * into `<`, `tool`, `_call`, `>` is not what the model emits. One token makes an id trigger, which
 * costs a comparison; anything longer is scanned for as bytes, because the tokeniser may split it
 * differently depending on what came before it. Getting this wrong does not fail loudly -- it
 * yields a trigger that never fires, so the call comes out unconstrained, which reads as a model
 * that writes malformed calls.
 *
 * A REGULAR-EXPRESSION TRIGGER IS REFUSED. Its matcher would run over the reply once per token
 * on the scheduler thread; the templates in service all trigger on words. A request whose
 * template needs one is refused at admission, naming the pattern. */
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

/* ================================================================== small helpers */

/* Copy this rank's window out of a whole-vocabulary bitmask. `vocab_off` is word-aligned for
 * every shard this engine builds (the head is split on a multiple of 32), so the aligned arm is
 * the one that runs; the bit-shifting arm exists because a shard boundary is a placement decision
 * and not a promise, and a mask that is silently off by a few bits admits the wrong tokens. */
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

/* ================================================================== the narrow plan's admission
 *
 * WHICH BATCHES THE 64-WIDE PLAN CAN SERVE WITHOUT CHANGING ONE TOKEN. The narrow plan is the same
 * ops over the same arithmetic at a smaller candidate width, so the question is exactly: for which
 * rows does every candidate-plane stage produce bit-identical output when the tail holes above the
 * row's live candidates are removed?
 *
 * THE PROOF, PER STAGE, and it rests on one fact about the layout rather than on the values:
 *
 *   sample_topk        writes `got = min(top_k, live)` live entries in slots [0, got) in descending
 *                      (logit, ~id) order and holes above. It sorts `sn = next_pow2(got)` words and
 *                      reads `n_cand` only as an upper bound on `k`, so a smaller `n_cand` that is
 *                      still >= got produces the same first `got` entries. Requiring top_k <= 64
 *                      per row and 64 >= 64 makes `got <= 64` hold at either width.
 *
 *   the cuts           read every slot, skip `id < 0`, and reduce with block_max / block_sum. Those
 *                      are tree reductions over a fixed 256-lane block with kThreads = 256, so
 *                      EVERY slot is held by its OWN lane at index < 256 and slot i lands in the same
 *                      lane at either width. Adding holes above 64 contributes nothing to a max and
 *                      adds exp(-inf - mx) = 0.0f to a sum -- and 0.0f is the exact additive
 *                      identity, so no lane's partial changes and the tree sees the same input at
 *                      both widths. The three narrowing stages are per-candidate predicates over
 *                      live slots, and their `mass`/`cum` walks are ordered by candidate index, so
 *                      the removed tail comes strictly after every live index and is never
 *                      visited: those walks are identical by prefix.
 *
 *   sample_pick        the same tree sum, then a SEQUENTIAL inverse-CDF walk in candidate order --
 *                      a prefix, again identical up to the last live slot, and a live -infinity
 *                      candidate adds 0.0f and cannot be chosen. Its `last` fallback only reads
 *                      live slots.
 *
 *   the merge          (tensor parallel) selection-sorts `n_in = world_size * n_cand` gathered
 *                      pairs into `n_cand` outputs, and the row's own top_k -- not the buffer width
 *                      -- bounds `keep`. Fewer per-rank slots means fewer gathered pairs, but every
 *                      one of them is a candidate some rank actually returned; the pairs that
 *                      disappear are the holes each rank wrote above its own `got`. A hole is
 *                      skipped by `id < 0`, so the selection order over the remaining pairs is the
 *                      same set in the same order, and `keep <= 64` at either width.
 *
 * AND THE ROWS THAT FAIL IT KEEP THE WHOLE BATCH WIDE, because a plan is a pair of buffers and a
 * set of op handles: one row cannot read half its candidates from a 64-wide plane and half from a
 * 1024-wide one. `top_k == 0` is the case that makes this mandatory rather than cautious -- it
 * means "the whole candidate width", so its SUPPORT IS `n_cand` and the two widths would sample
 * from different supports. That is a different answer, not a rounding difference, and it is why
 * the narrow plan is a second plan and not a smaller admission bound.
 *
 * The width comparison below is over the STAGED rows, after build_step has rewritten a greedy row
 * to top_k = 1 over a split vocabulary; that is the value the kernel reads. */
bool Sampler::narrow_suits(const DeviceSampleParams* rows, int64_t n) const {
    if (!narrow_declared_ || n <= 0) return false;
    for (int64_t i = 0; i < n; ++i) {
        /* A greedy-flagged row is an ARGMAX row in an unsplit batch: every candidate stage skips
         * it and its token never touches a candidate plane, so it constrains nothing. */
        if (rows[i].flags & RAD_SP_GREEDY) continue;
        /* 0 is the whole candidate width, which the two plans give different answers to. Negative
         * is refused at admission and never staged; treated the same way here rather than trusted. */
        if (rows[i].top_k <= 0) return false;
        if (rows[i].top_k > (int32_t)kSampleNarrowCand) return false;
    }
    return true;
}

/* rad_runtime.h's RAD_B / RAD_W macros are C compound literals. They are legal C++ only as a
 * compiler extension, and this file is core code that has to build with -Wall -Wextra clean, so
 * the operands are built the ordinary way. */
/* NARROWED TO THIS STEP'S ROWS, always.  arch/common/rad_arch.h says why for the model's own
 * buffers and the reason is the sampler's too: a kernel cannot read the row count out of `M`,
 * because `M` is the band's upper bound, so every shim takes it off the operand.  Handed over at
 * its DECLARED extent -- max_seqs * (max_spec + 1) rows -- an operand makes every stage compute a
 * full candidate set for rows that have no request behind them, over params the upload never
 * wrote.  The extra rows are discarded, so nothing downstream is wrong and no bench can see it,
 * but each phantom row costs a vocabulary-wide reduction, and at one sequence every row but one
 * is a phantom.
 *
 * `rows` is dim 0 for every buffer here -- logits, params, history, mask, candidates, pairs, the
 * gather and the token output all count sampled POSITIONS first -- and the strides below it are
 * products of the trailing dimensions, so shortening the outermost one leaves them correct. */
static RadOperand opd_rows(rad_buf h, int64_t rows) {
    RadOperand o{};
    o.kind = RAD_OPK_BUF;
    o.handle = h;
    o.rows = rows;
    return o;
}

/* An optional operand this issue does not pass. The runtime holds every issue to its schema's
 * operand COUNT -- operands are positional -- so an absent trailing optional is spelled, not
 * dropped. */
static RadOperand opd_none() {
    RadOperand o{};
    o.kind = RAD_OPK_NONE;
    return o;
}

static int64_t rows_for(const SamplerConfig& c) {
    /* One row per SAMPLED POSITION, not per sequence: a speculative verify samples n_spec+1
     * positions for a sequence and each one has its own output position, hence its own RNG
     * coordinate. Sizing for the worst step is the same rule the arena follows (spec §3.1). */
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
    if (!spechub_config_valid() || !mtp_coupled_config_valid()) return RAD_E_INVAL;
    spechub_ = spechub_enabled();
    couple_ = mtp_coupled_enabled();
    if (cfg_.n_vocab_local <= 0) cfg_.n_vocab_local = cfg_.n_vocab / cfg_.world_size;

    /* THE STAGED MASK IS THIS RANK'S SHARD, AND THE GRAMMAR'S IS THE WHOLE VOCABULARY. Those are
     * different lengths the moment the head is sharded, and conflating them is silent: every
     * vocab-scoped sampler op is declared over `n_vocab_local` (op_vocab, below) and `sample_mask`
     * indexes its bitmask from bit zero, so a full-vocabulary mask handed to rank 1 masks its
     * logits with the admissibility of tokens 0..n_local -- ids that belong to rank 0. The result
     * is not a diagnosable error, it is a rank sampling from the wrong permission set.
     *
     * So the grammar fills `mask_full_` over the whole vocabulary, which is also where the "no
     * token is admissible anywhere" check belongs, and only this rank's window of it is staged. */
    mask_words_ = grammar_mask_words(cfg_.n_vocab_local);
    mask_full_.assign((size_t)grammar_mask_words(cfg_.n_vocab), 0u);

    const int64_t rows = rows_for(cfg_);
    const int64_t nv   = cfg_.n_vocab_local;
    const int64_t ncand = cfg_.max_candidates;

    /* A second declare on the same object starts from no plans at all: `wide_` and `narrow_` hold
     * handles from the builder that is being replaced, and a step that kept one would issue ops
     * the new program never declared. */
    wide_ = Plan{};
    narrow_ = Plan{};
    narrow_declared_ = false;
    plan_ = &wide_;

    /* ---------------------------------------------------------- buffers */
    /* PERSIST, not TRANSIENT: the parameter, history and mask planes are written by an upload
     * that happens BEFORE the run phase issues anything, so liveness over the declared op list
     * would conclude they are dead where they are in fact being filled. A transient whose first
     * definition is invisible to the planner is a transient the planner will happily overwrite. */
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
    buf_hist_   = decl("sampler.history", RAD_I32, rows, cfg_.max_hist, RAD_BUF_PERSIST);
    /* One byte beside every history token: nonzero where that token is a whole DRY sequence
     * breaker, which DRY never penalises. Resolving a token against the breaker strings needs the
     * vocabulary, so it happens here on the host and the kernel reads the answer. */
    buf_brk_    = decl("sampler.dry_breakers", RAD_U8, rows, cfg_.max_hist, RAD_BUF_PERSIST);
    buf_mask_   = decl("sampler.mask", RAD_U32, rows, mask_words_, RAD_BUF_PERSIST);
    if (spechub_) {
        buf_spechub_ = decl("sampler.spechub", RAD_U32, rows, RAD_SPECHUB_STATE_WORDS,
                            RAD_BUF_PERSIST);
        if (!buf_spechub_) return RAD_E_NOMEM;
    }
    if (couple_) {
        buf_couple_ = decl("sampler.mtp_couple", RAD_U32, rows, RAD_MTP_COUPLE_STATE_WORDS,
                           RAD_BUF_PERSIST);
        if (!buf_couple_) return RAD_E_NOMEM;
    }

    if (!buf_params_ || !buf_hist_ || !buf_brk_ || !buf_mask_) {
        RAD_ERR("sampler: a staging buffer failed to declare");
        return RAD_E_NOMEM;
    }

    /* ---------------------------------------------------------- ops */
    /* Every one of these is declared unconditionally and may fail to resolve. That is not an
     * error here: a deployment that never sends `typical_p` has no use for a typical kernel, and
     * failing declare over it would make an absent sampler an absent model. check_request turns
     * an unresolved sampler into a named refusal at admission instead. */
    /* `M` is the ranged parameter and it is the number of SAMPLED POSITIONS, not tokens: one row
     * at decode, n_spec+1 per sequence at a verify, and a kernel written for a single row is a
     * different kernel from one written for a batch of 256 (spec §2.2). */
    auto op_vocab = [&](const char* name) -> rad_op {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 2, nullptr, 0);
    };
    /* THE VOCABULARY OFFSET, on the three stages that cross between a token id and a column of
     * this rank's logits row: top-k turns a column into a token id, and the penalties and DRY turn
     * the history's token ids into columns. Without it a rank past the first penalises the
     * columns of whatever ids happen to be small, which are other tokens entirely. Declared only
     * when the vocabulary is sharded, so a single-rank graph carries no parameter it has no use
     * for. */
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

    op_penalties_ = op_vocab_off("sample_penalties");
    op_dry_       = op_vocab_off("sample_dry");
    op_temp_      = op_vocab("sample_temp");
    op_mask_      = op_vocab("sample_mask");
    op_argmax_    = op_vocab("sample_argmax");

    /* THE TWO CANDIDATE PLANS. `wide_` keeps the names, the width and the handles it has always
     * had; `narrow_` is the same plan at kSampleNarrowCand, and only exists when the flag asks for
     * it and the deployment's own width is wider than that. A width at or below the narrow one
     * would make the second plan a duplicate of the first, so it is not declared and every step
     * stays on `wide_` -- the flag then costs one branch and no bytes. */
    RAD_TRY(declare_plan(b, &wide_, rows, ncand, ""));
    if (sample_candidates64_enabled() && ncand > (int64_t)kSampleNarrowCand) {
        /* FAIL-CLOSED. A narrow plan that does not resolve is a BROKEN INITIALIZATION, not a
         * request this deployment cannot serve, so the flag cannot be left silently inert and the
         * declaration cannot be half-made. Every op here is the same op at a smaller `n_cand`, so a
         * refusal means a kernel's band pins the wide width -- which is a fact about the build, and
         * serving anyway would report the flag as honoured when it is not.
         *
         * AND IT CANNOT BE ABANDONED PART-WAY EITHER: the builder latches the first error and
         * `finish()` propagates it, so buffers and ops already declared for this plan stay in the
         * program. Returning the status here is what keeps that visible instead of letting a later
         * step discover it. */
        RAD_TRY(declare_plan(b, &narrow_, rows, (int64_t)kSampleNarrowCand, "64"));
        narrow_declared_ = true;
    }

    RAD_TRY(stage_alloc());
    declared_ = true;

    RAD_INFO("sampler: %lld-token vocabulary (%lld this rank), %lld rows, %lld mask words "
             "(%s per row)%s", (long long)cfg_.n_vocab, (long long)nv, (long long)rows,
             (long long)mask_words_, humanb(mask_words_ * 4).c_str(),
             narrow_declared_ ? ", narrow 64-candidate plan beside the wide one" : "");
    return RAD_OK;
}

/* ONE CANDIDATE PLAN, cut to `n_cand`.
 *
 * The buffer names carry `tag` because a builder hands out one handle per name and two plans are
 * two sets of planes. Under tensor parallel the pair plane and the gathered plane follow the same
 * width: sample_topk packs this rank's candidates into `pairs` as (u32 global id, f32 logit), and
 * the all-gather places every rank's pairs for a row SIDE BY SIDE, which is the one layout the
 * merge reads at a single row stride.
 *
 * THE MERGE READS `wide_.n_cand` CANDIDATES OUT OF A GATHERED PLANE THAT HOLDS world_size * n_cand
 * PAIRS. A narrow plan's merge is declared at width 64 and its gathered plane at 64 * world_size
 * pairs, so the two agree by construction rather than by the caller remembering to pass both. */
int Sampler::declare_plan(RadBuilder* b, Plan* p, int64_t rows, int64_t n_cand, const char* tag) {
    /* The buffer names carry `tag` because a builder hands out one handle per name and two plans
     * are two sets of planes. The name is built in a local string and handed to the builder, which
     * copies it; nothing keeps a pointer to it. */
    std::string name;
    auto decl = [&](const char* base, uint32_t dt, int64_t d1, int kind) -> rad_buf {
        name.assign(base);
        if (*tag) { name += '.'; name += tag; }
        RadBufDecl d{};
        d.dtype = dt;
        d.rank  = 2;
        d.shape[0] = rows;
        d.shape[1] = d1;
        d.kind = kind;
        d.domain = RAD_DOMAIN_DEVICE;
        return rad_decl_buffer(b, name.c_str(), &d);
    };

    p->n_cand = n_cand;
    p->cand_i = decl("sampler.cand_idx", RAD_I32, n_cand, RAD_BUF_TRANSIENT);
    p->cand_v = decl("sampler.cand_val", RAD_F32, n_cand, RAD_BUF_TRANSIENT);
    if (!p->cand_i || !p->cand_v) return RAD_E_NOMEM;
    if (vocab_parallel()) {
        p->pairs  = decl("sampler.cand_pairs", RAD_U32, n_cand * 2, RAD_BUF_TRANSIENT);
        p->gather = decl("sampler.cand_gathered", RAD_U32, n_cand * cfg_.world_size * 2,
                         RAD_BUF_TRANSIENT);
        if (!p->pairs || !p->gather) return RAD_E_NOMEM;
    }

    /* `M` is the ranged parameter and it is the number of SAMPLED POSITIONS, not tokens: one row
     * at decode, n_spec+1 per sequence at a verify, and a kernel written for a single row is a
     * different kernel from one written for a batch of 256 (spec §2.2). The geometry after top-k
     * is the candidate width, which is what makes those stages cheap. */
    RadParam pc[2] = {
        RadParam{ "M", RAD_P_RANGE, 1, rows_for(cfg_), nullptr },
        RadParam{ "n_cand", RAD_P_INT, (long long)n_cand, 0, nullptr },
    };
    auto op_cand = [&](const char* opname) -> rad_op {
        return rad_decl_op(b, opname, pc, 2, nullptr, 0);
    };

    /* sample_topk's parameters are the VOCABULARY's, not the candidate width's: it reads a whole
     * logits row and writes a candidate set, so its geometry is (M, n_vocab[, vocab_off]). Its
     * output width is the candidate plane's extent, which is why the same op serves both plans. */
    {
        RadParam pt[3] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows_for(cfg_), nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)cfg_.n_vocab_local, 0, nullptr },
            RadParam{ "vocab_off", RAD_P_INT, (long long)cfg_.vocab_off, 0, nullptr },
        };
        const int npt = vocab_parallel() ? 3 : 2;
        p->topk = rad_decl_op(b, "sample_topk", pt, npt, nullptr, 0);
    }
    p->topp    = op_cand("sample_topp");
    p->minp    = op_cand("sample_minp");
    p->typical = op_cand("sample_typical");
    p->xtc     = op_cand("sample_xtc");
    p->pick    = op_cand("sample_pick");

    /* Every candidate producer and consumer must participate in liveness. Declaring only a later
     * stage's read would make the planes live at that op alone, excluding top-k's write and every
     * intervening cut and pick. The uploaded planes remain PERSIST. */
    const rad_buf cand[] = {p->cand_i, p->cand_v};
    RAD_TRY(rad_op_writes(b, p->topk, cand, 2));
    for (rad_op op : {p->topp, p->minp, p->typical, p->xtc}) {
        RAD_TRY(rad_op_reads(b, op, cand, 2));
        RAD_TRY(rad_op_writes(b, op, cand, 2));
    }
    RAD_TRY(rad_op_reads(b, p->pick, cand, 2));

    if (vocab_parallel()) {
        /* lm_head is vocab-sharded, so each rank reduces ITS OWN shard and only the candidate
         * sets are combined -- k * world_size pairs, 2 KiB at k=64 and world_size=4, against the
         * ~77 MiB a full-logits gather would move (spec §9). */
        RadParam ag[4] = {
            RadParam{ "world_size", RAD_P_INT, (long long)cfg_.world_size, 0, nullptr },
            RadParam{ "numel", RAD_P_INT, (long long)(rows * n_cand * 2), 0, nullptr },
            RadParam{ "dtype", RAD_P_STR, 0, 0, "f32" },
            /* ROW-INTERLEAVED. One sampled position's pairs from every rank have to land side by
             * side, because sample_merge_topk reads one row at one row stride. */
            RadParam{ "row", RAD_P_INT, (long long)(n_cand * 2), 0, nullptr },
        };
        p->allgather = rad_decl_op(b, "all_gather", ag, 4, nullptr, 0);
        p->merge     = op_cand("sample_merge_topk");
        RAD_TRY(rad_op_writes(b, p->merge, cand, 2));
    }
    /* After every target filter and the ordinary pick: the coupling reads the FINAL candidates.
     * Declared per plan, because "the final candidates" is that plan's plane and width -- a
     * narrow step handed the wide coupler would read 1024 entries out of a 64-wide row. */
    if (couple_) {
        RadParam pcp[3] = {
            RadParam{"M", RAD_P_RANGE, 1, rows_for(cfg_), nullptr},
            RadParam{"n_cand", RAD_P_INT, (long long)n_cand, 0, nullptr},
            RadParam{"n_vocab", RAD_P_INT, cfg_.n_vocab, 0, nullptr}
        };
        p->couple = rad_decl_op(b, "mtp_couple_verify", pcp, 3, nullptr, 0);
        if (!p->couple) return RAD_E_NOKERNEL;
        const rad_buf reads[] = {p->cand_i, p->cand_v, buf_params_, buf_couple_};
        RAD_TRY(rad_op_reads(b, p->couple, reads, 4));
    }
    /* SpecHub's verifier reads the same final set, and for the same reason it is per plan. */
    if (spechub_) {
        RadParam psp[4] = {
            RadParam{"M", RAD_P_RANGE, 1, rows_for(cfg_), nullptr},
            RadParam{"n_cand", RAD_P_INT, (long long)n_cand, 0, nullptr},
            RadParam{"n_vocab", RAD_P_INT, cfg_.n_vocab, 0, nullptr},
            RadParam{"transport", RAD_P_INT, spechub_transport() ? 1 : 0, 0, nullptr}
        };
        p->spechub = rad_decl_op(b, "spechub_verify", psp, 4, nullptr, 0);
        if (!p->spechub) return RAD_E_NOKERNEL;
        const rad_buf reads[] = {p->cand_i, p->cand_v, buf_params_, buf_spechub_};
        RAD_TRY(rad_op_reads(b, p->spechub, reads, 4));
    }

    /* WHAT THE NARROW PLAN MUST RESOLVE: exactly what the wide one did.
     *
     * The wide plan is allowed to leave a stage unresolved -- a deployment that never sends
     * `typical_p` has no use for a typical kernel, and check_request turns that into a named
     * refusal if a request ever asks. A plan that is only ever chosen for batches the wide plan
     * would also have served has to meet the same bar and no higher: an op the wide plan resolved
     * and this one did not would be a stage a step silently ran without, which is the failure the
     * whole named-refusal design exists to prevent. A plan that resolves MORE than the wide one is
     * fine and is simply never issued.
     *
     * This runs at declare, where a broken initialization belongs. It is not a reason to fall back
     * to the wide plan: falling back is for request data the narrow width cannot serve, and a
     * kernel that will not take `n_cand = 64` is a fact about the build that the flag cannot be
     * quietly reconciled with. Checked only when this plan is the narrow one, because the wide
     * plan is the reference it is checked against. */
    if (p != &wide_) {
        struct Stage { rad_op mine; rad_op theirs; const char* name; };
        const Stage stages[] = {
            { p->topk,      wide_.topk,      "sample_topk"       },
            { p->topp,      wide_.topp,      "sample_topp"       },
            { p->minp,      wide_.minp,      "sample_minp"       },
            { p->typical,   wide_.typical,   "sample_typical"    },
            { p->xtc,       wide_.xtc,       "sample_xtc"        },
            { p->pick,      wide_.pick,      "sample_pick"       },
            { p->allgather, wide_.allgather, "all_gather"        },
            { p->merge,     wide_.merge,     "sample_merge_topk" },
        };
        for (const Stage& st : stages) {
            if (st.theirs && !st.mine) {
                RAD_ERR("sampler: RADIANCE_SAMPLE_CANDIDATES64 declared a %lld-wide plan whose "
                        "'%s' did not resolve, while the %lld-wide plan's did. The flag is "
                        "fail-closed: a plan that would silently run a step without a stage the "
                        "wide plan applies is not an acceptable fallback",
                        (long long)n_cand, st.name, (long long)wide_.n_cand);
                return RAD_E_NOKERNEL;
            }
        }
    }
}

int Sampler::stage_alloc() {
    const int64_t rows = rows_for(cfg_);

    /* Pinned, because these are uploaded every step and a pageable source turns an async copy
     * into a staged one with a host bounce -- which is exactly the per-step cost the step batch
     * exists to remove (spec §8). */
    params_stage_ = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    greedy_stage_ = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    greedy_ready_ = false;
    hist_stage_   = (int32_t*)rad_dev_alloc(rows * cfg_.max_hist * 4, RAD_MEM_HOST_PINNED);
    brk_stage_    = (uint8_t*)rad_dev_alloc(rows * cfg_.max_hist, RAD_MEM_HOST_PINNED);
    mask_stage_   = (uint32_t*)rad_dev_alloc(rows * mask_words_ * 4, RAD_MEM_HOST_PINNED);
    if (spechub_)
        spechub_stage_ = static_cast<RadSpechubState*>(
            rad_dev_alloc(rows * sizeof(RadSpechubState), RAD_MEM_HOST_PINNED));
    if (couple_)
        couple_stage_ = static_cast<RadMtpCoupleState*>(
            rad_dev_alloc(rows * sizeof(RadMtpCoupleState), RAD_MEM_HOST_PINNED));

    if (!params_stage_ || !greedy_stage_ || !hist_stage_ || !brk_stage_ || !mask_stage_ ||
        (spechub_ && !spechub_stage_) || (couple_ && !couple_stage_)) {
        RAD_ERR("sampler: could not pin %s of staging",
                humanb(rows * ((int64_t)sizeof(DeviceSampleParams) + cfg_.max_hist * 5 +
                               mask_words_ * 4)).c_str());
        stage_free();
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

void Sampler::stage_free() {
    if (params_stage_) rad_dev_free(params_stage_, RAD_MEM_HOST_PINNED);
    if (greedy_stage_) rad_dev_free(greedy_stage_, RAD_MEM_HOST_PINNED);
    if (hist_stage_)   rad_dev_free(hist_stage_,   RAD_MEM_HOST_PINNED);
    if (brk_stage_)    rad_dev_free(brk_stage_,    RAD_MEM_HOST_PINNED);
    if (mask_stage_)   rad_dev_free(mask_stage_,   RAD_MEM_HOST_PINNED);
    if (spechub_stage_) rad_dev_free(spechub_stage_, RAD_MEM_HOST_PINNED);
    if (couple_stage_) rad_dev_free(couple_stage_, RAD_MEM_HOST_PINNED);
    spechub_stage_ = nullptr;
    couple_stage_ = nullptr;
    params_stage_ = nullptr;
    greedy_stage_ = nullptr;
    cur_stage_    = nullptr;
    hist_stage_   = nullptr;
    brk_stage_    = nullptr;
    mask_stage_   = nullptr;
    /* Nothing is staged to run on, so `run` must not find a plan that claims otherwise. */
    plan_ = &wide_;
}

Sampler::~Sampler() { stage_free(); }

/* ================================================================== admission */

/* WHEN THE TWO HISTORY STAGES RUN, in one place, because admission, request setup and the step
 * staging each ask and must not disagree. llama.cpp's meaning throughout: a window of 0 turns the
 * stage off, and DRY also needs a base of at least 1. The multiplier is taken as positive because
 * a negative one rewards repetition, which is not a request anyone makes on purpose. */
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

    if (spechub_ && (!(sp.temp > 0.0f) || sp.top_k <= 0 ||
        !sp.grammar.empty() || sp.xtc_probability != 0.0f || sp.thinking_budget >= 0))
        return refuse("SpecHub requires positive temperature, explicit top_k in 1..max_candidates, "
                      "no grammar, no XTC and no thinking budget", RAD_E_UNSUPPORTED);
    if (sp.top_k < 0)                       return refuse("top_k is negative", RAD_E_INVAL);
    if (sp.top_p < 0.0f || sp.top_p > 1.0f) return refuse("top_p is outside [0,1]", RAD_E_INVAL);
    if (sp.min_p < 0.0f || sp.min_p > 1.0f) return refuse("min_p is outside [0,1]", RAD_E_INVAL);
    if (sp.typical_p < 0.0f || sp.typical_p > 1.0f)
        return refuse("typical_p is outside [0,1]", RAD_E_INVAL);
    if (sp.temp < 0.0f)                     return refuse("temperature is negative", RAD_E_INVAL);
    if (sp.top_k > cfg_.max_candidates)
        return refuse("top_k exceeds the sampler's candidate width (--max-candidates)",
                      RAD_E_INVAL);
    /* Every breaker is a pass over the whole vocabulary on the scheduler thread, and every other
     * sequence on the server waits for it; see kDryMaxBreakers. Refused here, before any of that
     * work is queued, rather than truncated -- a request whose breakers quietly went missing is a
     * request whose output changed. */
    if (dry_on(sp) && sp.dry_seq_breakers.size() > kDryMaxBreakers) {
        if (why)
            *why = "dry_sequence_breakers has " + std::to_string(sp.dry_seq_breakers.size()) +
                   " entries; at most " + std::to_string(kDryMaxBreakers) + " are accepted";
        return RAD_E_INVAL;
    }

    /* An unresolved sampler is a named refusal, not a silent skip. A request whose min_p quietly
     * stopped being applied is a request whose output changed and nothing said so. */
    auto need = [&](bool wanted, rad_op h, const char* n) -> const char* {
        return (wanted && !h) ? n : nullptr;
    };
    /* A SPLIT VOCABULARY MOVES GREEDY ONTO THE CHAIN, so what a greedy request needs moves with
     * it: not the argmax, but the top-k, the merge and the pick that build_step's top_k = 1 sends
     * it through. Admission has to check what will actually be issued.
     *
     * ADMISSION CHECKS THE WIDE PLAN, and that is the whole point of the narrow one being a
     * second plan rather than a smaller bound: every request a deployment admits has to be
     * servable, and the wide plan is the one that serves all of them. The narrow plan only ever
     * carries batches whose rows are already inside its width, so a resolved wide stage is
     * necessary and sufficient for admission. */
    const bool greedy_chain = sp.greedy() && vocab_parallel();
    const char* missing = nullptr;
    if (!missing) missing = need(sp.greedy() && !greedy_chain, op_argmax_, "sample_argmax");
    if (!missing) missing = need(!sp.greedy(), op_temp_, "sample_temp");
    if (!missing) missing = need(!sp.greedy() || greedy_chain, wide_.pick, "sample_pick");
    /* EVERY CHAIN ROW NEEDS THE TOP-K, whatever its own top_k says: it is the stage that writes
     * the candidate set every later stage reads. top_k = 0 asks for the whole candidate width,
     * not for no candidates. */
    if (!missing) missing = need(!sp.greedy() || greedy_chain, wide_.topk, "sample_topk");
    if (!missing) missing = need(sp.top_p < 1.0f, wide_.topp, "sample_topp");
    if (!missing) missing = need(sp.min_p > 0.0f, wide_.minp, "sample_minp");
    if (!missing) missing = need(sp.typical_p < 1.0f, wide_.typical, "sample_typical");
    if (!missing) missing = need(sp.xtc_probability > 0.0f, wide_.xtc, "sample_xtc");
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

    /* The triggers are resolved here as well as at begin_request, so a template whose triggers
     * this machine cannot run is a refusal at admission rather than a request cancelled on the
     * scheduler thread. */
    if (!sp.grammar.empty() && !sp.grammar_triggers.empty()) {
        std::vector<std::string> words;
        std::vector<int32_t>     tokens;
        std::string              e;
        const int st = resolve_triggers(*vocab_, sp.grammar_triggers, &words, &tokens, &e);
        if (st < 0) {
            if (why) *why = e;
            return st;
        }
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
        /* ONLY A NON-LAZY GRAMMAR IS POSITIONED AT THE START OF THE TURN. An eager grammar's root
         * describes the whole assistant turn and therefore opens with the generation prompt, which
         * is already in the prompt -- so it has to be replayed or the model is made to write it
         * out again. A LAZY one describes only the region after its trigger: its root begins at
         * `<tool_call>`, and feeding it a prefix it never claimed to match fails outright, taking
         * every tool-calling request with it. */
        spec.prefix = sp.grammar_lazy ? std::string() : sp.grammar_prefix;
        RAD_TRY(resolve_triggers(*vocab_, sp.grammar_triggers, &spec.trigger_words,
                                 &spec.trigger_tokens, err));
        RAD_TRY(grammar_create(spec, *vocab_, &st.grammar, err));
    }

    /* BREAKERS ARE OPTIONAL. DRY runs whenever the request turns it on, as llama.cpp's does with an
     * empty breaker list; the breakers only limit how far back a repeat may be counted and exempt
     * the tokens that are breakers themselves. */
    if (dry_on(sp) && !sp.dry_seq_breakers.empty()) {
        if (!vocab_) {
            if (err) *err = "DRY needs a vocabulary to resolve its sequence breakers";
            return RAD_E_STATE;
        }
        if (sp.dry_seq_breakers.size() > kDryMaxBreakers) {
            if (err) *err = "too many DRY sequence breakers";
            return RAD_E_INVAL;
        }
        /* Once per request, never per step: a full vocabulary scan per breaker string is
         * milliseconds at a 151K vocab and has no business on the step path. */
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
    /* TIMED for the same reason build_step is: this runs once per ACCEPTED token per rank, on the
     * step thread, so it is wall time on every constrained token. */
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

/* A CONSTRAINING GRAMMAR DOES NOT MEAN NO SPECULATION. build_step masks draft position i
 * against the state reached by accepting drafts 0..i-1 and then rewinds, so a backend that can
 * push and pop its state speculates like any other request -- which is worth several times the
 * per-token cost on a constrained request. What IS refused is a backend that cannot rewind:
 * masking its window would mean masking against a state the sequence is not in. */
bool Sampler::allows_speculation(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    if (it == seqs_.end() || !it->second.grammar) return true;
    return !it->second.grammar->constrains() || it->second.grammar->can_speculate();
}

bool Sampler::speculation_fixed(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    return it != seqs_.end() && (!it->second.grammar || it->second.grammar->can_speculate());
}

bool Sampler::allows_branching(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    return it == seqs_.end() || !it->second.grammar;
}

int Sampler::set_spechub(uint64_t req_id, const RadSpechubState& state) {
    const auto it = seqs_.find(req_id);
    if (!spechub_ || it == seqs_.end() || state.status != 1 || state.draw.seq != req_id)
        return RAD_E_STATE;
    it->second.spechub = state;
    return RAD_OK;
}

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

/* Is `c` the state the engine bound to draft row k of THIS window: same request and seed, the
 * round of that row, the output position it samples, and the very token it verifies? */
static bool couple_bound(const RadMtpCoupleState& c, const Request& r, int k, int64_t vocab) {
    return c.status == 1 && c.draw.seq == r.id && c.draw.seed == r.sp.seed && c.round == k + 1 &&
           c.pos == (uint64_t)r.output.size() + (uint64_t)k && r.draft && c.x == r.draft[k] &&
           rad_mtp_couple_state_valid(c, vocab);
}

/* ================================================================== the step path */

int Sampler::build_step(const std::vector<Request*>& reqs) {
    if (!declared_) return RAD_E_STATE;

    const auto t_build = ((int)log_level() >= (int)Log::Debug)
                             ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    const int64_t cap = rows_for(cfg_);
    n_seq_ = 0;
    any_brk_ = false;
    any_spechub_ = false;
    any_couple_ = false;
    cur_stage_ = params_stage_;

    /* DRY'S HOST HALF, for one row whose history `h[0, take)` is already staged. The breakers are
     * strings resolved against the vocabulary, so what they mean for this window is worked out
     * here and handed to the kernel as data: the repeat limit the most recent breaker imposes
     * (llama.cpp's step 1), and one byte per position marking the tokens that are whole breakers
     * and therefore never penalised. The window is the one the kernel will take -- the tail of
     * the staged row, `dry_penalty_last_n` long when that is positive.
     *
     * Returns false when the breakers rule DRY out for the row altogether: a breaker closer to the
     * end than `allowed_length` leaves no repeat long enough to count. A row with no breakers
     * gets zero marks, because the plane is shared by the whole batch and another row may have
     * some. */
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

    /* Validate the batch transactionally before any row is staged or any automaton pushed. */
    int64_t total_pos = 0;
    for (Request* r : reqs) {
        if (!r) continue;
        const int n_pos = r->n_draft + 1;
        total_pos += n_pos;

        const bool is_tree = r->tree_parent && r->tree_depth;
        bool tree_branched = false;
        if (is_tree) {
            const char* bad = nullptr;
            if (n_pos > (int)radiance::RAD_TREE_MAX_NODES)
                bad = "the window exceeds the RAD_TREE_MAX_NODES tree budget";
            else if (r->tree_parent[0] != -1 || r->tree_depth[0] != 0)
                bad = "the root is not parent -1 at depth 0";
            else if (n_pos > 1 && !r->draft)
                bad = "the node tokens are absent";
            for (int j = 1; !bad && j < n_pos; ++j) {
                const int32_t pj = r->tree_parent[j];
                if (pj < 0 || pj >= j || r->tree_depth[j] != r->tree_depth[pj] + 1)
                    bad = "a node's parent is not an earlier node one depth above it";
                else {
                    if (pj != j - 1) tree_branched = true;
                    for (int i = 1; i < j; ++i)
                        if (r->tree_parent[i] == pj && r->draft[j - 1] == r->draft[i - 1]) {
                            bad = "two siblings propose the same token";
                            break;
                        }
                }
            }
            if (bad) {
                RAD_ERR("sampler: request %llu's tree window is malformed: %s",
                        (unsigned long long)r->id, bad);
                return RAD_E_INVAL;
            }
        }
        if (spechub_ && n_pos > 1) {
            const auto it = seqs_.find(r->id);
            const RadSpechubState* s = it == seqs_.end() ? nullptr : &it->second.spechub;
            const int32_t root = !r->output.empty() ? r->output.back()
                               : !r->prompt.empty() ? r->prompt.back() : -1;
            bool valid = s && s->status == 1 && s->draw.seq == r->id &&
                s->draw.seed == r->sp.seed && s->draw.pos == r->output.size() &&
                s->root == root && n_pos <= 3 && r->draft;
            for (int j = 1; valid && j < n_pos; ++j)
                valid = r->draft[j - 1] == s->pair[j - 1] &&
                    (!is_tree || (r->tree_parent[j] == 0 && r->tree_depth[j] == 1));
            if (!valid) {
                RAD_ERR("sampler: SpecHub request %llu has stale coordinate/root/pair metadata",
                        (unsigned long long)r->id);
                return RAD_E_STATE;
            }
        }
        /* STRICT: every draft row of a running stochastic request verifies its own bound Q. Drafts
         * reach a coupled deployment only through MTP's set_draft, so a missing or foreign state
         * is a broken path, never a reason to verify a sampled x as if it were independent. */
        if (couple_ && n_pos > 1 && !r->sp.greedy() && r->state == ReqState::Running) {
            const auto it = seqs_.find(r->id);
            const SeqState* s = it == seqs_.end() ? nullptr : &it->second;
            for (int k = 0; k + 1 < n_pos; ++k)
                if (is_tree || !s || k >= s->couple_n ||
                    !couple_bound(s->couple[k], *r, k, cfg_.n_vocab)) {
                    RAD_ERR("sampler: MTP coupled request %llu draft row %d (output position "
                            "%llu) has no bound proposal state", (unsigned long long)r->id, k,
                            (unsigned long long)(r->output.size() + (size_t)k));
                    return RAD_E_STATE;
                }
        }

        if (n_pos > 1 && !allows_speculation(r->id)) {
            RAD_ERR("sampler: request %llu carries a constraining grammar its backend cannot "
                    "snapshot, and %d draft tokens. The mask for draft position i is the mask of "
                    "the state reached by accepting drafts 0..i-1, so masking the window needs "
                    "push_state/rollback. The scheduler must consult allows_speculation() and "
                    "decode this one a token at a time.",
                    (unsigned long long)r->id, r->n_draft);
            return RAD_E_UNSUPPORTED;
        }

        /* A BRANCHED TREE HAS NO GRAMMAR YET. The walk masks draft position k against the
         * state drafts 0..k-1 leave the automaton in -- a LINEAR conditioning, which is exactly
         * what a chain's path is. Two nodes that share a parent are conditioned on different
         * histories, and serving either with the other's mask constrains a branch against a
         * state its sequence is not in; a correct per-path fork state is separate work, and this
         * refuses until it exists rather than quietly unconstraining the branch. A LINEAR
         * topology under a grammar is not refused: its paths are the flat draft, and the walk is
         * the right one. */
        if (is_tree && tree_branched && !allows_branching(r->id)) {
            RAD_ERR("sampler: request %llu carries a constraining grammar over a branched "
                    "tree window. The grammar's mask for one node is the state its own "
                    "ancestor path leaves the automaton in, and no per-path fork state "
                    "exists; refusing the step rather than masking siblings against one "
                    "branch's state", (unsigned long long)r->id);
            return RAD_E_UNSUPPORTED;
        }
    }

    if (total_pos > cap) {
        RAD_ERR("sampler: %lld sampled positions exceeds the %lld the arena was sized for",
                (long long)total_pos, (long long)cap);
        return RAD_E_FULL;
    }

    for (Request* r : reqs) {
        if (!r) continue;
        const SamplingParams& sp = r->sp;
        const int n_pos = r->n_draft + 1;
        const bool is_tree = r->tree_parent && r->tree_depth;

        const int64_t n_prompt = (int64_t)r->prompt.size();
        const int64_t n_out    = (int64_t)r->output.size();
        const int64_t n_total  = n_prompt + n_out;
        const bool    pen      = penalties_on(sp);
        const bool    dry      = dry_on(sp);

        auto it = seqs_.find(r->id);
        SeqState* st = it == seqs_.end() ? nullptr : &it->second;
        const DryBreakers* brk = st && !st->dry.empty() ? &st->dry : nullptr;

        /* THE WINDOW IS WALKED, THEN REWOUND. Position 0 is masked against the state the sequence
         * is really in; position k against the state reached by accepting drafts 0..k-1. The push
         * happens whatever the grammar's current mode, because a draft token can fire a lazy
         * grammar's trigger mid-window -- at which point the positions behind it DO need a mask,
         * and without the push they would be sampled unconstrained.
         *
         * `dead` is not a failure. A draft the grammar forbids simply cannot be accepted: the
         * target's own masked pick at that position will differ from it, so verification rejects
         * there and every position behind it is discarded. Masking stops and the rest of the
         * window is left unconstrained, which is what "discarded" needs it to be. */
        GrammarBackend* gr = st && st->grammar ? st->grammar.get() : nullptr;
        const bool walk = gr && n_pos > 1;
        if (walk) gr->push_state();
        bool dead = false;

        /* THE REASONING BUDGET, walked over the window as the grammar is. Position k is inside the
         * block or not according to the committed reply and drafts 0..k-1, which is the only
         * history under which its token is ever emitted; a draft that ends the block early is
         * accepted or rejected like any other. */
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
            /* THE TOKENS IN FRONT OF THE COMMITTED OUTPUT at this row's position: drafts 0..k-1
             * for a chain, or -- when the window is a tree -- the ANCESTOR PATH of node k,
             * root's child down to k itself, which is what the token sampled at this row is
             * conditioned on. Row k IS node k; once two nodes share a parent the physical row
             * index names another branch's history, so everything below reads `pre` and not
             * r->draft. The path lands in caller stack storage -- fixed-size, nothing on this
             * path allocates (spec §3.1) -- and the walk parents down from k, each step to a
             * strictly smaller node id, so it is bounded by the depth the validation capped. */
            int32_t        tree_path[radiance::RAD_TREE_MAX_NODES];
            const int32_t* pre   = r->draft;
            int64_t        n_pre = k;
            if (is_tree) {
                int32_t j = (int32_t)k;
                int32_t d = r->tree_depth[k];
                while (d > 0) {
                    tree_path[d - 1] = r->draft[j - 1];
                    j = r->tree_parent[j];
                    --d;
                }
                pre   = tree_path;
                n_pre = r->tree_depth[k];
            }
            /* THE REASONING BUDGET AT THIS ROW'S POSITION: the committed reply plus the path in
             * front of it, recomputed from the committed state per row because a tree's rows are
             * not one growing prefix. For a chain this is the same sequence of pushes the
             * incremental walk made -- ReasoningCount is a pure state machine over the tokens
             * pushed, and drafts 0..k-1 in order is what it saw. */
            if (budget) {
                think = st->think;
                for (int64_t i = 0; i < n_pre; ++i) think.push(sp.reasoning, pre[i]);
            }
            const int64_t row = n_seq_++;
            if (spechub_) {
                spechub_stage_[row] = {};
                /* A capacity-truncated tree retains the COMPLETE sampled pair and transport.
                 * Missing siblings cannot be accepted; root y still has full target law p.
                 * No draft at all takes the explicit ordinary target path instead. */
                if (k == 0 && n_pos > 1) {
                    spechub_stage_[row] = st->spechub;
                    any_spechub_ = true;
                }
            }
            if (couple_) {
                /* Draft rows only; the bonus row and every unbound row keep the ordinary pick. */
                couple_stage_[row] = RadMtpCoupleState{};
                if (st && !is_tree && !sp.greedy() && k + 1 < n_pos && k < st->couple_n &&
                    couple_bound(st->couple[k], *r, k, cfg_.n_vocab)) {
                    couple_stage_[row] = st->couple[k];
                    any_couple_ = true;
                    RAD_TRACE("sampler: MTP coupled row req=%llu round=%d pos=%llu x=%d support=%d",
                              (unsigned long long)r->id, k + 1,
                              (unsigned long long)couple_stage_[row].pos, couple_stage_[row].x,
                              couple_stage_[row].count);
                }
            }

            DeviceSampleParams& p = params_stage_[row];
            std::memset(&p, 0, sizeof p);

            p.seed = sp.seed;
            /* The output position, advanced across the speculative window: draft position k is
             * the (n_out + k)-th token this request will emit, so its randomness is the same
             * whether it arrived through speculation or one token at a time. A tree row takes its
             * NODE'S DEPTH as that advance -- the position the token it verifies would occupy on
             * its own branch -- and never the physical row index, which for a sibling is a
             * position on the other branch. */
            p.pos = (uint64_t)(n_out + n_pre);

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

            /* GREEDY OVER A SPLIT VOCABULARY IS A TOP-1 CHAIN, not an argmax. sample_argmax reads a
             * whole logits row and writes the winning COLUMN, which under vocab parallelism is
             * this rank's local index into its own shard -- a fluent answer built from the wrong
             * token ids. Staging top_k = 1 sends the row through top-k, the pair plane, the
             * gather and the merge instead, which is where a local index becomes a global one;
             * sample_pick over a single live candidate returns it deterministically, so the
             * result is the argmax it was always meant to be.
             *
             * Such a row is a CHAIN row and does not carry RAD_SP_GREEDY: every candidate stage
             * skips a row that does, because in an unsplit batch that flag is what keeps the pick
             * from overwriting the argmax's answer with a draw. */
            if (sp.greedy()) {
                if (vocab_parallel()) p.top_k = 1;
                else                  p.flags |= RAD_SP_GREEDY;
            }
            if (sp.xtc_probability > 0.0f) p.flags |= RAD_SP_XTC;

            /* THE HISTORY THE PENALTIES AND DRY READ, staged only for a row that runs one of them
             * -- the copy is scheduler-thread work and the upload is step-path bytes, and neither
             * buys anything for a row that reads none of it. It is the prompt, everything
             * generated so far and the tokens in front of that position (`pre`): the chain's
             * k drafted tokens, or the tree row's ancestor path. The token sampled there is
             * emitted only if that path is, so it is part of the output it is penalised against,
             * exactly as it would be one token at a time. Oldest first, truncated to the declared
             * window -- a bound on the buffer, not a policy on the sampler; penalty_last_n still
             * decides how much of it counts. The row owns its own row of the plane, so the window
             * starts at offset 0 within it. */
            p.hist_off = 0;
            p.hist_len = 0;
            if (pen || dry) {
                const int64_t len  = n_total + n_pre;
                const int64_t take = std::min<int64_t>(len, cfg_.max_hist);
                int32_t* h = hist_stage_ + row * cfg_.max_hist;
                for (int64_t i = 0; i < take; ++i) {
                    const int64_t src = len - take + i;
                    h[i] = src < n_prompt ? r->prompt[(size_t)src]
                         : src < n_total  ? r->output[(size_t)(src - n_prompt)]
                                          : pre[src - n_total];
                }
                p.hist_len = (int32_t)take;
                if (pen) p.flags |= RAD_SP_PENALTY;
                if (dry && stage_dry(p, h, take, brk, row)) p.flags |= RAD_SP_DRY;
            }

            if (gr && !dead && gr->constrains()) {
                uint32_t* dst = mask_stage_ + row * mask_words_;
                int s;
                /* WHAT THE MASK COSTS, MEASURED RATHER THAN ASSUMED. The mask is scheduler-thread
                 * work with the rank threads waiting at the barrier, so it is wall time on every
                 * constrained token and would decide the step if it were slow. Reported at Debug,
                 * per 256 masks, mean and worst. */
                const bool   inst = (int)log_level() >= (int)Log::Debug;
                const auto   t0   = inst ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
                /* ASK THE GRAMMAR WHETHER IT COMPUTED THE MASK, rather than inferring it from how
                 * long the call took. No duration threshold separates the two: a mask computed
                 * from the plan can finish inside the time a memo hit costs on a large
                 * vocabulary, so a timing test counts computed masks as hits and any per-mask
                 * cost derived from the split is wrong by that factor. The counter is exact and
                 * costs a relaxed load. */
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
                    mask_us_ += us; mask_worst_us_ = mask_us_n_ && us < mask_worst_us_
                                                   ? mask_worst_us_ : us;
                    if (st->grammar->masks_walked() != w0) { ++mask_slow_; mask_walk_us_ += us; }
                    else                                     ++mask_fast_;
                    if (++mask_us_n_ == 256) {
                        /* Mean per MASK and mean per COMPUTED mask are different questions and
                         * the second is the one that says what a new state costs: a window's
                         * masks are mostly memo hits, so the first tracks how many positions
                         * asked. */
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
                    /* A SPECULATIVE POSITION IS NOT A DEAD REQUEST. This is the state reached by
                     * accepting drafts the sequence has not committed to, so "nothing is
                     * admissible here" says the draft is wrong, not that the request is stuck.
                     * Verification will reject at or before this position; stop masking and leave
                     * the rest of the window unconstrained, since it is about to be discarded. */
                    dead = true;
                } else if (s < 0) {
                    /* A DEAD GRAMMAR FAILS ITS REQUEST, NOT THE STEP. Returning a status here
                     * would reach the step loop as an engine fault and stop it stepping, so every
                     * other client on the server would stop receiving tokens because one of them
                     * sent a schema the model painted itself out of. The row is left unmasked for
                     * this step -- it is about to be cancelled -- and the id goes to the engine
                     * to close. */
                    RAD_WARN("sampler: request %llu has no admissible token under its grammar; "
                             "failing that request", (unsigned long long)r->id);
                    failed_.push_back(r->id);
                } else {
                    p.mask_row = (int32_t)row;
                    p.flags |= RAD_SP_MASK;
                    if ((int)log_level() >= (int)Log::Trace) {
                        /* How many tokens the grammar allows THIS RANK to pick. Zero is legal
                         * under sharding -- the admissible set can live entirely on another rank
                         * -- and is the first thing to check when a grammar-constrained answer
                         * comes out unconstrained. */
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
        }

        /* Back to where the sequence actually is. The tokens it really accepted are applied by
         * Sampler::accept_token after the step verifies, so anything the walk did must be gone
         * before then or the grammar would be ahead of the sequence by the whole window. */
        if (walk) RAD_TRY(gr->rollback());
    }

    /* WHICH CANDIDATE PLAN THIS STEP RUNS ON. Decided from the staged rows and not from the
     * requests, because the staged rows are what `run` will issue against: a row's own top_k has
     * already been rewritten by the greedy rule above, and it is that rewritten value the kernel
     * reads. A batch with no row to serve keeps `wide_`. */
    plan_ = narrow_suits(params_stage_, n_seq_) ? &narrow_ : &wide_;

    /* THE WHOLE OF IT, not just the mask, because that is what the rank threads wait for. Without
     * this the mask's own number could fall to nothing and the step stay slow with no way to see
     * which side of the barrier the time was on. */
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
        /* The grammar's mask for this row. Under a split vocabulary the whole one is still in
         * mask_full_, and the close may be another rank's column. */
        const bool admitted =
            vocab_parallel()
                ? close < (int64_t)mask_full_.size() * 32 &&
                      (mask_full_[(size_t)(close >> 5)] >> (uint32_t)(close & 31)) & 1u
                : local && (dst[at >> 5] >> (uint32_t)(at & 31)) & 1u;
        if (!admitted) return;
    }
    /* ONE ADMISSIBLE TOKEN, on whichever rank holds its column. The others mask the whole row
     * out, which the candidate merge reads as no candidate from them (the grammar's own masks do
     * the same when the admissible set lives on one shard). */
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
    any_spechub_ = false;
    any_couple_ = false;
    cur_stage_ = greedy_stage_;
    if (!greedy_ready_) {
        for (int64_t i = 0; i < cap; ++i) {
            DeviceSampleParams& p = greedy_stage_[i];
            std::memset(&p, 0, sizeof p);
            p.temp     = 0.0f;
            p.top_k    = 1;
            p.hist_off = 0;
            p.hist_len = 0;
            p.mask_row = -1;
            /* Over a split vocabulary a greedy row is a top-1 CHAIN row, for the reason
             * build_step gives, and so it does not carry the flag every candidate stage skips. */
            p.flags    = vocab_parallel() ? 0 : RAD_SP_GREEDY;
        }
        greedy_ready_ = true;
    }
    /* A greedy pass is staged at top_k = 1 for every row. Over a split vocabulary those rows carry
     * no greedy flag and take the top-k chain, so the narrow plan serves them exactly; over an
     * unsplit vocabulary every row is an ARGMAX row and touches no candidate plane at all, so
     * either plan is correct and `wide_` is kept -- a draft pass is not where candidate bytes
     * matter. */
    plan_ = (vocab_parallel() && narrow_suits(greedy_stage_, n)) ? &narrow_ : &wide_;
    return RAD_OK;
}

int Sampler::run(RadCtx* c, rad_buf logits, rad_buf out_tokens) {
    if (!declared_) return RAD_E_STATE;
    if (!c || !logits || !out_tokens) return RAD_E_INVAL;
    if (n_seq_ == 0) return RAD_OK;

    const RadStream s = rad_stream(c);

    bool any_mask = false, any_greedy = false, any_chain = false;
    bool any_penalty = false, any_dry = false, any_xtc = false;
    bool any_topp = false, any_minp = false, any_typ = false;
    bool any_temp = false;
    for (int64_t i = 0; i < n_seq_; ++i) {
        const DeviceSampleParams& p = cur_stage_[i];
        any_mask    |= (p.flags & RAD_SP_MASK)    != 0;
        any_penalty |= (p.flags & RAD_SP_PENALTY) != 0;
        any_dry     |= (p.flags & RAD_SP_DRY)     != 0;
        any_xtc     |= (p.flags & RAD_SP_XTC)     != 0;
        if (p.flags & RAD_SP_GREEDY) any_greedy = true;
        else {
            any_chain = true;
            any_temp |= (p.temp > 0.0f && p.temp != 1.0f);
            any_topp |= p.top_p < 1.0f;
            any_minp |= p.min_p > 0.0f;
            any_typ  |= p.typical_p < 1.0f;
        }
    }

    /* A GREEDY ROW OVER A SPLIT VOCABULARY is staged as a top-1 chain row, never with the flag:
     * the argmax would return this rank's column for a token id. build_step and build_greedy both
     * hold that, so a flagged row here is a staging bug, and it is refused rather than served. */
    if (any_greedy && vocab_parallel()) {
        RAD_ERR("sampler: a greedy-flagged row reached a split vocabulary, where the argmax "
                "returns a column of this rank's shard rather than a token id");
        return RAD_E_STATE;
    }

    /* Async copies on the compute stream and no host synchronisation, and only of what some row
     * will read: the history and the breaker marks for the penalties and DRY, the mask for a
     * grammar. The mask upload is 19 KiB a constrained sequence at a 151K vocab, which is the
     * number that makes a host-side grammar affordable at all (spec §13). */
    RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_params_), cur_stage_,
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
    if (any_spechub_)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_spechub_), spechub_stage_,
                                 n_seq_ * sizeof(RadSpechubState), s));
    if (any_couple_)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_couple_), couple_stage_,
                                 n_seq_ * (int64_t)sizeof(RadMtpCoupleState), s));

    const RadOperand o_logits = opd_rows(logits, n_seq_);
    const RadOperand o_params = opd_rows(buf_params_, n_seq_);
    const RadOperand o_hist = opd_rows(buf_hist_, n_seq_);
    const RadOperand o_brk = opd_rows(buf_brk_, n_seq_);
    const RadOperand o_mask = opd_rows(buf_mask_, n_seq_);
    const RadOperand o_ci = opd_rows(plan_->cand_i, n_seq_);
    const RadOperand o_cv = opd_rows(plan_->cand_v, n_seq_);
    const RadOperand o_tok = opd_rows(out_tokens, n_seq_);

    auto issue = [&](rad_op op, std::initializer_list<RadOperand> ops) -> int {
        return rad_issue(c, op, ops.begin(), (int)ops.size(), n_seq_);
    };

    /* Penalties act on raw logits, before anything derives a probability from them. DRY takes the
     * breaker marks only when some row in the batch has breakers; its fourth operand is optional
     * and absent means no token is exempt. */
    if (any_penalty) RAD_TRY(issue(op_penalties_, { o_logits, o_params, o_hist }));
    if (any_dry)
        RAD_TRY(issue(op_dry_, { o_logits, o_params, o_hist, pass_brk ? o_brk : opd_none() }));

    /* The greedy rows take the argmax after penalties and the mask and never see temperature.
     * When the batch is mixed, BOTH the argmax and the chain are issued over the whole batch: the
     * argmax writes every row's token, and then every candidate stage skips the greedy rows
     * (RAD_SP_GREEDY) so the pick overwrites only the rows the chain owns. The cost is one extra
     * pass over the logits for the rows the other op owns -- ~77 MiB read at batch 256 -- against
     * one kernel launch per distinct parameter set, which at batch 256 is up to 256 launches. The
     * pass wins.
     *
     * Under vocab parallelism there is no greedy shortcut at all: build_step staged those rows at
     * top_k = 1, without the flag, so they take the chain, where the merge turns a local candidate
     * index into a global token id. */
    if (any_greedy) {
        if (any_mask) RAD_TRY(issue(op_mask_, { o_logits, o_params, o_mask }));
        RAD_TRY(issue(op_argmax_, { o_logits, o_params, o_tok }));
    }

    if (!any_chain) return RAD_OK;

    if (any_temp) RAD_TRY(issue(op_temp_, { o_logits, o_params }));
    if (any_mask && !any_greedy)
        RAD_TRY(issue(op_mask_, { o_logits, o_params, o_mask }));

    /* top-k reduces the vocabulary to a candidate set, and everything after it works on that set.
     * Under tensor parallel this is the LOCAL top-k over this rank's shard; the candidate sets
     * are combined below and full logits are never gathered (spec §9).
     *
     * IT IS ISSUED WHENEVER THE CHAIN RUNS, not only when some row sets top_k. A row at
     * top_k = 0 asks for the whole candidate width, and the candidate planes are transient: a
     * step that skipped this would hand the pick whatever the planes last held -- the previous
     * step's candidates, or another op's bytes. */
    if (vocab_parallel()) {
        /* The fifth operand is the pair plane: this rank's candidates with GLOBAL ids, which is
         * what crosses the wire. cand_idx and cand_val are omitted (pair-only) because the merge
         * overwrites both, so local stores are dead in TP2. */
        const RadOperand o_p = opd_rows(plan_->pairs, n_seq_);
        const RadOperand o_g = opd_rows(plan_->gather, n_seq_);
        RAD_TRY(issue(plan_->topk,      { o_logits, o_params, o_ci, o_cv, o_p }));
        RAD_TRY(issue(plan_->allgather, { o_p, o_g }));
        RAD_TRY(issue(plan_->merge,     { o_g, o_params, o_ci, o_cv }));
    } else {
        /* No pair plane on a whole vocabulary, and it is passed as absent rather than left off:
         * the schema has five operands and an issue of four is refused. */
        RAD_TRY(issue(plan_->topk, { o_logits, o_params, o_ci, o_cv, opd_none() }));
    }

    if (any_topp) RAD_TRY(issue(plan_->topp,    { o_ci, o_cv, o_params }));
    if (any_minp) RAD_TRY(issue(plan_->minp,    { o_ci, o_cv, o_params }));
    if (any_typ)  RAD_TRY(issue(plan_->typical, { o_ci, o_cv, o_params }));
    if (any_xtc)  RAD_TRY(issue(plan_->xtc,     { o_ci, o_cv, o_params }));

    RAD_TRY(issue(plan_->pick, { o_ci, o_cv, o_params, o_tok }));
    if (any_spechub_)
        RAD_TRY(issue(plan_->spechub, {o_ci, o_cv, o_params,
                                       opd_rows(buf_spechub_, n_seq_), o_tok}));
    if (any_couple_)
        RAD_TRY(issue(plan_->couple, {o_ci, o_cv, o_params, opd_rows(buf_couple_, n_seq_), o_tok}));
    return RAD_OK;
}

}  /* namespace rad */
