/* draft_types.h -- the adaptive draft policy's internal contracts (P00).
 *
 * This is the seam the six strands build against (docs/ADAPTIVE-MTP.md):
 *
 *   A  core/sched/draft_cost.{h,cpp}       estimate(shape), the measured profile
 *   B  core/sched/draft_predictor.{h,cpp}  predict(view, predictions), per-position acceptance
 *   C  core/sched/draft_policy.{h,cpp}     choose(view, predictions, costs, plan)
 *   E  core/engine_draft.cpp               executes a plan's rounds and publishes observations
 *
 * Nothing here is public ABI; a plugin never sees this header. Nothing here allocates: every
 * array is caller-provided and sized at max_seqs (requests) or RAD_SCHED_MAX_SPEC (draft
 * positions). The structs hold pointers into that storage, so copying one copies the handle, not
 * the data.
 *
 * A plan is written once, by choose(), and is immutable from issue until every pass it names has
 * been consumed. No hook writes into a plan it did not just fill; observe() takes only const
 * inputs and an observation names its plan rather than editing it. A stale plan is recognised by
 * (cycle_id, policy_version, req_id, slot, slot_gen): none of those may be inferred from an index
 * alone, because slots are recycled.
 *
 * Units: milliseconds wherever a duration appears. A value that is not a duration is never
 * presented as one -- DraftCostSource says where an estimate came from, and an unmeasured shape is
 * an invalid estimate or a named conservative fallback, not a unitless prior labelled ms.
 *
 * Labels: only checked positions are labels (draft_label). The accepted prefix is 1, the first
 * real rejection is 0, and positions past it were never measured. EOG, max_tokens, cancellation
 * and the context end censor a cycle; they are recorded as the reason and do not turn into
 * labels.
 *
 * K counts drafts after the committed token. The first target is K = 0..6; the arrays are sized
 * for RAD_SCHED_MAX_SPEC so a wider window the kernel band allows still has storage.
 */
#pragma once
#include "rad_core.h"

#include <cstdint>

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows, the verify-step band (spec §10), so 64 is the
 * largest window any batch can carry. Every per-position array is bounded by it. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* The draft profile's on-disk format (docs/ADAPTIVE-MTP.md). A file whose header differs is
 * refused by name, never reinterpreted under the current keys. */
enum { RAD_DRAFT_PROFILE_VERSION = 1 };

/* Context-length buckets of the profile key. A bucket is a property of the text, not of the
 * deployment: profiles only carry the buckets the configured capacity can reach. */
enum { RAD_DRAFT_CTX_CLASSES = 4 };
inline int draft_ctx_class(int32_t ctx_len) {
    if (ctx_len < 4096)   return 0;
    if (ctx_len < 32768)  return 1;
    if (ctx_len < 131072) return 2;
    return 3;
}

/* ------------------------------------------------------------------ small vocabularies */

/* How a verified position is accepted. At T > 0 the target's own draw, never its argmax: the rule
 * is exact only against the token the target produced (core/sample/accept.h). */
enum class DraftSampling : uint8_t { Greedy = 0, Sampled = 1 };

/* Why a request does not draft in this cycle. `Eligible` is the normal case. */
enum class DraftEligibility : uint8_t {
    Eligible = 0,
    Grammar = 1,          /* a currently constraining grammar: the mask at i needs 0..i-1 decided */
    OutputExhausted = 2,  /* remaining output is used up */
    Cancelled = 3,
    Preempted = 4,
    NoDrafter = 5,        /* no head and no prompt lookup (the n-gram gate) */
    Budget = 6,           /* token or KV budget for another draft is gone */
};

/* Why a plan's K differs from what the policy would otherwise choose. Kept per request as well as
 * per plan: one sequence can be capped by its budget while the batch runs at the cost depth. */
enum class DraftFallback : uint8_t {
    None = 0,
    NoProfile = 1,    /* no valid profile for this fingerprint */
    UnknownShape = 2, /* no measured or interpolable cost for this shape */
    Resources = 3,    /* token/KV/latency budget cut the plan */
    Unsupported = 4,  /* a mode this runtime does not implement; never a silent substitute */
};

/* Who may fill a request's draft budget. `k[]` is the budget for every drafter, not for the head
 * alone: a request the plan puts at K = 0 must stay empty, so the n-gram fallback may only fill a
 * request the plan assigned to it. The view's batch-wide `ngram_allowed` is the gate the plan is
 * chosen under; once a plan exists, its per-request answer is the one the scheduler follows. */
enum class DraftDrafter : uint8_t {
    None = 0,   /* no drafter: k[] must be 0, and no fallback may fill it */
    Serial = 1, /* the model's draft head fills k[], capped by k[] */
    Ngram = 2,  /* the prompt-lookup fallback fills k[], capped by k[] */
};

/* Where a cycle-level plan came from. `Fallback` is a fixed baseline chosen because the cost
 * path could not price the shape (see DraftFallback for why); it is a visible answer, not a
 * pretend-selection. */
enum class DraftPlanSource : uint8_t { Legacy = 0, Cost = 1, Fallback = 2, Probe = 3 };

/* An exploratory probe's reason; probes are bounded and rate-limited by B. */
enum class DraftProbe : uint8_t { None = 0, Recovery = 1, Uncertainty = 2, Age = 3 };

/* Why a draft prefix ended before its last position was checked. Every value except None censors
 * the labels past the cut. */
enum class DraftCensor : uint8_t { None = 0, Eog = 1, MaxTokens = 2, Cancelled = 3, ContextEnd = 4 };

/* Where a cost estimate came from. A prior that is not a duration is ConservativeFallback (with
 * invalid set when it cannot even bound the cost), never Measured. */
enum class DraftCostSource : uint8_t { Measured = 0, Interpolated = 1, ConservativeFallback = 2 };

/* The runtime's replay state for a pass, so a sample taken while recording or auditing never
 * updates the profile that prices normal operation. */
enum class DraftReplay : uint8_t { None = 0, Play = 1, Record = 2, Audit = 3 };

/* A cost sample's attribution. Only a sample with kDraftSampleValid and none of the disturbing
 * flags may update a measured bucket (draft_sample_attributable). */
enum DraftSampleFlag : uint16_t {
    kDraftSampleValid     = 1u << 0,  /* the bracket times this shape, start to end */
    kDraftSampleAhead     = 1u << 1,  /* issued ahead: under `issue_ahead` the bracket may hold
                                       * commit/staging only; valid only when the caller checked */
    kDraftSampleWarmup    = 1u << 2,  /* isolated warmup or first recording of the shape */
    kDraftSampleRecord    = 1u << 3,  /* replay recording */
    kDraftSampleAudit     = 1u << 4,  /* replay audit */
    kDraftSampleAdmission = 1u << 5,  /* an admission or compaction ran inside the bracket */
    kDraftSamplePlacement = 1u << 6,  /* a placement/DMA move ran inside the bracket */
};

/* ------------------------------------------------------------------ the cycle's inputs */

/* One candidate request. The identity triple is what a plan, an observation and a sample agree
 * on: a recycled slot with another generation is another request, and its history is not this
 * plan's. */
struct DraftSeqView {
    uint64_t req_id = 0;
    int32_t  slot = -1;
    uint32_t slot_gen = 0;
    int32_t  ctx_len = 0;    /* tokens computed before the cycle: the context the draft reads */
    int32_t  ctx_class = 0;  /* draft_ctx_class(ctx_len) */
    int32_t  remaining = 0;  /* output tokens still allowed; a draft past this is wasted */
    DraftSampling    sampling = DraftSampling::Greedy;
    DraftEligibility eligibility = DraftEligibility::Eligible;
    bool prefill = false;    /* this request still carries prefill work in the same step */
};

/* The cycle's batch, read-only for every hook. The caller fills it; nothing here is cached
 * across cycles. */
struct DraftBatchView {
    uint64_t cycle_id = 0;        /* unique and monotonic across the pipeline */
    uint32_t policy_version = 0;  /* the hooks' version, copied into the plan */
    int32_t  n_seq = 0;           /* candidate requests, [0, cap) */
    int32_t  cap = 0;             /* what seq[] holds: max_seqs at init */
    const DraftSeqView* seq = nullptr;

    int32_t k_ceiling = 0;        /* the declared maximum (--num-speculative-tokens); 0 = no
                                   * drafter loaded, which is a process, not a temporary K0 */
    int32_t token_budget = 0;     /* draft/verify tokens this cycle may add */
    int32_t kv_budget = 0;        /* KV tokens available for draft lookahead */
    int32_t prefill_tokens = 0;   /* prefill tokens sharing the step; 0 decode-only */
    bool    ngram_allowed = true; /* false: the head is the only drafter (the n-gram gate) */
};

/* ------------------------------------------------------------------ A: the cost key */

/* What a verify pass computes, separated from what the draft rounds actually ran. `rows[j]` is
 * N_j, the number of requests whose K reaches position j: a rectangular window has all N_j equal,
 * E2's shrinking rounds do not. The profile versions the two executions apart, so a shape keyed
 * only by (rows, max K) would mix them. */
struct DraftVerifyShape {
    int32_t n_rows = 0;         /* decode sequences in the batch */
    int32_t verify_k = 0;       /* deepest window verified; 0 is a plain decode step */
    int32_t verify_tokens = 0;  /* tokens the pass computes: sum_i (1 + k_i) over the rows */
    int32_t ctx_class = 0;      /* draft_ctx_class over the rows (the deepest one) */
    int32_t history_tokens = 0; /* MTP history/QSA rows carried in the same cycle */
    int32_t prefill_tokens = 0; /* prefill tokens sharing the step; 0 decode-only */
    /* N_j for j = 1..verify_k; index 0 unused. */
    int32_t rows[RAD_SCHED_MAX_SPEC + 1] = {};
    DraftSampling sampling = DraftSampling::Greedy;
    DraftReplay   replay = DraftReplay::None;
};

/* One completed cycle's timing, tied to the plan that shaped it. Verify shape and draft extent
 * are separate because their costs scale with different counts (rows vs rounds). */
struct DraftCostSample {
    uint64_t cycle_id = 0;      /* the plan this sample belongs to; 0 = unattributed */
    uint32_t policy_version = 0;
    DraftVerifyShape shape;
    int32_t rounds = 0;         /* deepest draft round the head actually ran */
    int32_t draft_rows = 0;     /* sum_j N_j actually executed */
    int32_t drafted = 0;        /* draft tokens produced */
    double  total_ms = 0.0;     /* the whole cycle: trunk step, draft rounds, commit */
    /* Split timings, -1 when the phase was not timed. Under overlap they do not add up to
     * total_ms; use them to calibrate components, the total to compare arms. */
    double  target_ms = -1.0;
    double  draft_ms = -1.0;
    double  host_ms = -1.0;
    uint16_t flags = 0;         /* DraftSampleFlag */
};

/* A price for one shape. `valid` false means nothing usable; the caller then uses its named
 * baseline, never a fabricated number. */
struct DraftCostEstimate {
    double  ms = 0.0;
    double  unc_ms = 0.0;   /* bound: the cost is expected within [ms - unc, ms + unc] */
    int32_t coverage = 0;   /* samples behind it, 0 for a conservative fallback */
    DraftCostSource source = DraftCostSource::ConservativeFallback;
    bool    valid = false;
};

/* The model A owns, handed to C so it can price each candidate plan it considers. */
using DraftEstimateFn = int (*)(void* ctx, const DraftVerifyShape& shape, DraftCostEstimate* out);
struct DraftCostModel {
    void*           ctx = nullptr;
    DraftEstimateFn estimate = nullptr;
};

/* ------------------------------------------------------------------ B: the prediction */

/* Per-position acceptance for one request. `a[j]` is the chance position j+1 is accepted given
 * that 1..j were; `n[j]` counts real observations and never includes an extrapolation; `valid`'s
 * bit j says a[j] was learned from data rather than filled from a prior. */
struct DraftPrediction {
    float    a  [RAD_SCHED_MAX_SPEC] = {};
    float    unc[RAD_SCHED_MAX_SPEC] = {};  /* half-width of a[j]'s uncertainty */
    uint16_t n  [RAD_SCHED_MAX_SPEC] = {};
    uint64_t valid = 0;                     /* bit j: a[j] is a prediction, not only a prior */
};

struct DraftPredictions {
    int32_t n_seq = 0;
    int32_t cap = 0;                   /* requests the caller's arrays hold (max_seqs) */
    uint64_t* req_id = nullptr;        /* [cap], same order as the view */
    DraftPrediction* seq = nullptr;    /* [cap] */
    uint32_t version = 0;
};

/* ------------------------------------------------------------------ the plan */

/* The cycle's frozen plan. choose() fills it once; every consumer reads the same pointers. `k[i]`
 * is request i's drafts, and `rows[j]` the number of requests that reach round j. It is a budget
 * for every drafter, so `fill[i]` names who may spend it; where k is 0 that is None. `verify_k`
 * (the window the paired step verifies, chosen by the previous cycle) and `max_k` (the deepest
 * window this cycle drafts) are separate numbers: reading one for the other is how a proposal from
 * another generation leaks into a step. */
struct DraftPlan {
    uint64_t cycle_id = 0;
    uint32_t policy_version = 0;
    int32_t  n_seq = 0;
    int32_t  cap = 0;
    int32_t  k_ceiling = 0;   /* copied from the view so telemetry needs no view */
    int32_t  max_k = 0;       /* deepest k[]; 0 = no drafts this cycle */
    int32_t  verify_k = 0;    /* deepest window under verification in the paired step */
    DraftPlanSource source = DraftPlanSource::Legacy;
    DraftFallback   fallback = DraftFallback::None;  /* set when source is Fallback */
    DraftProbe      probe = DraftProbe::None;

    /* Caller-provided. Request arrays are sized at cap; `rows` at RAD_SCHED_MAX_SPEC + 1. */
    uint64_t*       req_id = nullptr;   /* [cap] */
    int32_t*        slot = nullptr;     /* [cap] */
    uint32_t*       slot_gen = nullptr; /* [cap] */
    int32_t*        k = nullptr;        /* [cap] drafts per request, 0..max_k */
    DraftDrafter*   fill = nullptr;    /* [cap] who may fill k[]; None exactly where k[] is 0 */
    DraftFallback*  why = nullptr;      /* [cap] the request's own downgrade reason, else None */
    int32_t*        rows = nullptr;     /* [RAD_SCHED_MAX_SPEC + 1] N_j; index 0 unused */
};

/* ------------------------------------------------------------------ observations */

enum { RAD_DRAFT_MAX_FEATURES = 4 };

/* One request's verify outcome. The counts are separate on purpose: a draft can be generated and
 * not published, published and not verified, verified and not checked (past the first rejection),
 * and checked and still censored before it is committed. */
struct DraftSeqObservation {
    uint64_t raw_match = 0;  /* consecutive drafts that matched the target's own draw, uncensored */
    uint8_t  generated = 0;  /* draft tokens the head produced */
    uint8_t  published = 0;  /* tokens actually published to the proposal buffer */
    uint8_t  verified = 0;   /* positions the target computed */
    uint8_t  checked = 0;    /* positions with a label: accepted prefix plus the first rejection */
    uint8_t  accepted = 0;   /* drafts committed (the emitted prefix) */
    DraftCensor censor = DraftCensor::None;
    float   feat[RAD_DRAFT_MAX_FEATURES] = {};  /* optional head features (margin, ...) */
    uint8_t feat_valid = 0;                     /* bit i: feat[i] is meaningful */
};

/* A completed verify, tied to the plan that produced it. `complete` is the caller's statement that
 * every row finished safely (no cancelled row still in flight); observe() refuses the rest. The
 * plan pointer must stay valid and unmodified for the observation's lifetime -- the scheduler's
 * ring keeps it alive while issue_ahead() may still name it. */
struct DraftObservation {
    uint64_t cycle_id = 0;
    uint32_t policy_version = 0;
    int32_t  n_seq = 0;
    int32_t  cap = 0;
    const DraftPlan* plan = nullptr;          /* the plan these rows came from, or null */
    const uint64_t*  req_id = nullptr;        /* [cap] */
    const int32_t*   slot = nullptr;          /* [cap] */
    const uint32_t*  slot_gen = nullptr;      /* [cap] */
    const DraftSeqObservation* seq = nullptr; /* [cap] */
    bool complete = false;
};

/* ------------------------------------------------------------------ the hooks */

/* Each hook returns RAD_OK, or a negative RAD_E_* (RAD_E_INVAL for a malformed argument,
 * RAD_E_UNSUPPORTED for a mode that is not implemented yet). A hook never fabricates a plan or an
 * estimate to look successful; an unpriced shape is an invalid estimate with a named fallback. */
using DraftPredictFn = int (*)(void* ctx, const DraftBatchView& view, DraftPredictions* out);
using DraftChooseFn  = int (*)(void* ctx, const DraftBatchView& view,
                               const DraftPredictions& predictions, const DraftCostModel& costs,
                               DraftPlan* out);
using DraftObserveFn = int (*)(void* ctx, const DraftObservation& observation,
                               const DraftCostSample& sample);

struct DraftPredictor { void* ctx = nullptr; DraftPredictFn predict = nullptr; };  /* B */
struct DraftPolicy    { void* ctx = nullptr; DraftChooseFn  choose  = nullptr; };  /* C */
/* A and B both learn from a completed cycle; whoever wires the pipeline composes their updates
 * behind this one hook. It is optional: a deployment without online learning leaves it null. */
struct DraftObserver  { void* ctx = nullptr; DraftObserveFn observe = nullptr; };

/* What the integrator wires into the scheduler. `version` is stamped into every plan, prediction
 * and sample, so a plan issued under an older policy is recognisable and never read as current. */
struct DraftHooks {
    DraftPredictor predictor;
    DraftCostModel costs;
    DraftPolicy    policy;
    DraftObserver  observer;
    uint32_t       version = 1;
};

/* ------------------------------------------------------------------ contract checks */

/* Fill rows[1..max_k] with N_j for the plan's k[]. Requires max_k == the deepest k and every k in
 * 0..max_k; index 0 and the rest of the array are zeroed. RAD_OK, or RAD_E_INVAL. */
inline int draft_plan_rows(DraftPlan& p) {
    if (!p.k || !p.rows) return RAD_E_INVAL;
    if (p.max_k < 0 || p.max_k > RAD_SCHED_MAX_SPEC) return RAD_E_INVAL;
    for (int j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) p.rows[j] = 0;
    int32_t top = 0;
    for (int32_t i = 0; i < p.n_seq; ++i) {
        if (p.k[i] < 0 || p.k[i] > p.max_k) {
            RAD_ERR("draft plan: request %d drafts %d, outside 0..%d", (int)i, (int)p.k[i],
                    (int)p.max_k);
            return RAD_E_INVAL;
        }
        for (int j = 1; j <= p.k[i]; ++j) ++p.rows[j];
        if (p.k[i] > top) top = p.k[i];
    }
    if (top != p.max_k) {
        RAD_ERR("draft plan: max_k is %d but the deepest k is %d", (int)p.max_k, (int)top);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* The gate before a plan is issued. Checks the identity arrays are present, every k is within the
 * declared ceiling, and rows[j] equals the count k[] implies. RAD_OK, or RAD_E_INVAL. */
inline int draft_plan_valid(const DraftPlan& p) {
    if (p.n_seq < 0 || p.cap < 0 || p.n_seq > p.cap) return RAD_E_INVAL;
    if (p.k_ceiling < 0 || p.k_ceiling > RAD_SCHED_MAX_SPEC) return RAD_E_INVAL;
    if (p.max_k < 0 || p.max_k > p.k_ceiling) {
        RAD_ERR("draft plan: max_k %d is outside 0..ceiling %d", (int)p.max_k, (int)p.k_ceiling);
        return RAD_E_INVAL;
    }
    if (p.verify_k < 0 || p.verify_k > RAD_SCHED_MAX_SPEC) return RAD_E_INVAL;
    if (!p.k || !p.rows || !p.fill) return RAD_E_INVAL;
    if (p.n_seq > 0 && (!p.req_id || !p.slot || !p.slot_gen)) return RAD_E_INVAL;
    int32_t count[RAD_SCHED_MAX_SPEC + 1] = {};
    for (int32_t i = 0; i < p.n_seq; ++i) {
        if (p.k[i] < 0 || p.k[i] > p.max_k) {
            RAD_ERR("draft plan: request %d drafts %d, outside 0..%d", (int)i, (int)p.k[i],
                    (int)p.max_k);
            return RAD_E_INVAL;
        }
        if ((p.k[i] == 0) != (p.fill[i] == DraftDrafter::None)) {
            RAD_ERR("draft plan: request %d has k %d and fill %d", (int)i, (int)p.k[i],
                    (int)p.fill[i]);
            return RAD_E_INVAL;
        }
        for (int j = 1; j <= p.k[i]; ++j) ++count[j];
    }
    for (int j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) {
        if (p.rows[j] != count[j]) {
            RAD_ERR("draft plan: rows[%d] is %d, k[] implies %d", j, (int)p.rows[j],
                    (int)count[j]);
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* True when request i may not be filled by any drafter this cycle. The scheduler's n-gram
 * fallback is a drafter too and must ask this before it fills an empty proposal: a K0 plan is a
 * plan, not an absence of one, and `k[]` is the budget for every drafter. */
inline bool draft_plan_bars_fill(const DraftPlan& p, int32_t i) {
    return p.k && p.fill && i >= 0 && i < p.n_seq && p.k[i] == 0 && p.fill[i] == DraftDrafter::None;
}

/* The label for draft position j (0-based), if this cycle measured one: 1 when the position was
 * accepted, 0 at the first real rejection. False past `checked` -- an unreached position has no
 * label, and neither does one past a censor. */
inline bool draft_label(const DraftSeqObservation& o, int j, float* label) {
    if (j < 0 || j >= (int)o.checked) return false;
    if (label) *label = j < (int)o.accepted ? 1.0f : 0.0f;
    return true;
}

/* Only a sample that timed its own shape and carried no recording, audit, admission, placement or
 * warmup inside the bracket may update a measured bucket. An ahead sample may, if the caller
 * checked the bracket; it stays flagged so the model can weigh it. Real DMA cost is not an
 * outlier: the placement flag routes it away from the decode bucket, it is not dropped. */
inline bool draft_sample_attributable(const DraftCostSample& s) {
    const uint16_t disturbing = kDraftSampleWarmup | kDraftSampleRecord | kDraftSampleAudit |
                                kDraftSampleAdmission | kDraftSamplePlacement;
    return (s.flags & kDraftSampleValid) && !(s.flags & disturbing) && s.total_ms > 0.0;
}

}  /* namespace rad */
