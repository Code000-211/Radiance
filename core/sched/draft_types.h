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
 * Lifetime: a plan and its observation live in the cycle ring, keyed by that identity, until the
 * observation is delivered or abandoned. Preemption stops the work but does not discard a cycle
 * that already ran; recycling a slot resets slot state but must not free or rewrite a plan or
 * observation still in flight; resume starts a new cycle with a new plan, and the old observation
 * never merges into it. Delivery never consults the live slot, so a recycled generation cannot
 * make a measured row unobservable (draft_observation_deliverable).
 *
 * Units: milliseconds wherever a duration appears. A value that is not a duration is never
 * presented as one -- DraftCostSource says where an estimate came from, and an unmeasured shape is
 * an invalid estimate or a named conservative fallback, not a unitless prior labelled ms.
 *
 * Labels: only checked positions are labels (draft_label). The accepted prefix is 1 and the first
 * real rejection is 0. A cycle that stopped at a token the draft itself proposed (EOG,
 * max_tokens) has that position compared but not committed: `end` says so, and the position gets
 * no label at all -- never a fabricated reject. EOG, max_tokens, cancellation and the context end
 * remain the censor reason; a censor does not delete a rejection measured before it.
 *
 * K counts drafts after the committed token. The first target is K = 0..6; the arrays are sized
 * for RAD_SCHED_MAX_SPEC so a wider window the kernel band allows still has storage.
 */
#pragma once
#include "rad_core.h"

#include <cmath>
#include <cstdint>

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows, the verify-step band (spec §10), so 64 is the
 * largest window any batch can carry. Every per-position array is bounded by it. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* The draft profile's on-disk format (docs/ADAPTIVE-MTP.md). A file whose header differs is
 * refused by name, never reinterpreted under the current keys. Version 3 is the first whose
 * fingerprint carries the resolved placement (strategy, budgets, residency digest) and the device
 * equipment; a version-2 file was written before those were required, so it cannot be bound to a
 * deployment and is regenerated rather than rescaled. */
enum { RAD_DRAFT_PROFILE_VERSION = 3 };

/* Context-length buckets of the profile key. A bucket is a property of the text, not of the
 * deployment: profiles only carry the buckets the configured capacity can reach. */
enum { RAD_DRAFT_CTX_CLASSES = 4 };
inline int draft_ctx_class(int32_t ctx_len) {
    if (ctx_len < 4096)   return 0;
    if (ctx_len < 32768)  return 1;
    if (ctx_len < 131072) return 2;
    return 3;
}

/* Drafts a request may run for `want` given its output budget. The verify step emits the bonus
 * token -- the target's own draw after the accepted prefix -- whether or not a draft was accepted,
 * so the drafts are bounded by remaining - 1, never by remaining. remaining <= 1 leaves room for
 * the bonus token alone. */
inline int32_t draft_budget_k(int32_t remaining, int32_t want) {
    if (remaining <= 1 || want <= 0) return 0;
    return want < remaining - 1 ? want : remaining - 1;
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
 * the labels past the cut. A censor is the reason, not a label: a real rejection measured just
 * before the cut keeps its 0 (DraftSeqEnd::Rejected). */
enum class DraftCensor : uint8_t { None = 0, Eog = 1, MaxTokens = 2, Cancelled = 3, ContextEnd = 4 };

/* How the labelled span ends. The three cases are different measurements and one label mask cannot
 * guess between them: `Rejected` is a real 0 at `accepted` (the target drew another token, even
 * when the cycle then censors); `Stopped` is the target and the draft agreeing on a stopping token
 * that is not committed -- compared, not a reject, so no label there; `AllAccepted` ends because
 * no position was left to check. `Unset` is refused: a caller that did not classify the cut gets
 * no label, never a guessed reject. */
enum class DraftSeqEnd : uint8_t {
    Unset = 0,
    AllAccepted = 1,
    Rejected = 2,
    Stopped = 3,
};

/* Who produced a request's proposals, per request. The fact, where DraftPlan::fill[] is the
 * permission: the plan says who may spend k[], the observation says who did. `Head` maps to
 * fill Serial and `Lookup` to fill Ngram; `None` means nothing was produced (K 0, or a preempted
 * row) and requires generated == 0. Never an implicit default -- a row whose origin is unknown is
 * not priced as head. */
enum class DraftOrigin : uint8_t { None = 0, Head = 1, Lookup = 2 };

/* Where a cost estimate came from. A prior that is not a duration is ConservativeFallback (with
 * invalid set when it cannot even bound the cost), never Measured. */
enum class DraftCostSource : uint8_t { Measured = 0, Interpolated = 1, ConservativeFallback = 2 };

/* The cycle's replay state as the executing path saw it, so a sample taken while recording or
 * auditing never updates the profile that prices normal operation. `None` is a claim ("provably
 * no replay" -- a plain run), so it may only be written from evidence; a caller that cannot tell
 * leaves `Unknown`, which is never read as a measurement. `plan.replay` is what was decided at
 * issue time and is not proof of what ran.
 *
 * The profile file stores these states by name, never by number: a reader that casts the file's
 * index would read a measured `none` as `Unknown` now that `Unknown` sits at 0. A profile is a
 * measurement, so `unknown` is not a file value at all. */
enum class DraftReplay : uint8_t { Unknown = 0, None = 1, Play = 2, Record = 3, Audit = 4 };

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
    kDraftSampleShortened = 1u << 7,  /* the cycle ended before its planned extent (rounds <
                                       * plan.max_k, or a cancelled row): its bracket times less
                                       * work than the shape it would be filed under */
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
    int32_t  remaining = 0;  /* output tokens still allowed, including the bonus token the verify
                              * step will emit; drafts are bounded by draft_budget_k() */
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
    int32_t pipeline_depth = 0;  /* cycles this plan is issued ahead of its verification; 0 = the
                                   * paired step verifies exactly what this cycle drafts */
    int32_t token_budget = 0;     /* draft/verify tokens this cycle may add */
    int32_t kv_budget = 0;        /* KV tokens available for draft lookahead */
    int32_t prefill_tokens = 0;   /* prefill tokens sharing the step; 0 decode-only */
    bool    ngram_allowed = true; /* false: the head is the only drafter (the n-gram gate) */
};

/* ------------------------------------------------------------------ A: the cost key */

/* What a verify pass computes, separated from what the draft rounds actually ran. `rows[j]` is
 * N_j, the number of requests whose verify reaches position j: a rectangular window has all N_j
 * equal, a cycle of a shrinking backend does not. `draft_rows[j]` is the plan's own N_j for the
 * drafting rounds, so the shape can be checked against the cycle that asked for it instead of
 * against a total. This shape is the bucket key, and it already separates a rectangular cycle
 * from a shrinking one; the profile's `execution` names the backend's round model instead, so a
 * cycle's distribution never changes the profile identity.
 *
 * `unknown` exists because a missing measurement is not a zero: a caller that lacks history,
 * prefill, replay or the plan's draft distribution sets the flag, and the model refuses the shape
 * (draft_shape_priceable) rather than pricing the zeros as measured. */
enum DraftShapeUnknown : uint32_t {
    kDraftShapeHistoryUnknown = 1u << 0,
    kDraftShapePrefillUnknown = 1u << 1,
    kDraftShapeReplayUnknown  = 1u << 2,
    kDraftShapeDraftUnknown   = 1u << 3,  /* no cycle-bound draft distribution, not "no drafts" */
};

struct DraftVerifyShape {
    int32_t n_rows = 0;         /* decode sequences in the batch */
    int32_t verify_k = 0;       /* deepest window verified; 0 is a plain decode step */
    int32_t verify_tokens = 0;  /* tokens the pass computes: sum_i (1 + k_i) over the rows */
    int32_t ctx_class = 0;      /* draft_ctx_class over the rows (the deepest one) */
    int32_t history_tokens = 0; /* MTP history/QSA rows carried in the same cycle */
    int32_t prefill_tokens = 0; /* prefill tokens sharing the step; 0 decode-only */
    /* N_j for j = 1..verify_k of the verify pass; index 0 unused. */
    int32_t rows[RAD_SCHED_MAX_SPEC + 1] = {};
    /* The plan's draft distribution, separate from the verify distribution above: N_j for
     * j = 1..draft_k, index 0 unused. Under pipeline_depth > 0 the paired step verifies a window
     * the previous cycle drafted, so the two distributions are not interchangeable. */
    int32_t draft_k = 0;
    int32_t draft_rows[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t pipeline_depth = 0; /* cycles between drafting this plan and verifying it */
    DraftSampling sampling = DraftSampling::Greedy;
    DraftReplay   replay = DraftReplay::None;
    /* The drafter loaded for the pass, not who filled k[]: K0 with the head loaded (its history
     * and KV still maintained) and K0 with no drafter process are different buckets. */
    DraftDrafter  drafter = DraftDrafter::None;
    uint32_t      unknown = 0;  /* DraftShapeUnknown bits; nonzero = not a measured key */
};

/* One completed cycle's timing, tied to the plan that shaped it. Verify shape and draft extent
 * are separate because their costs scale with different counts (rows vs rounds). The executed
 * draft distribution is kept per round: two cycles with the same `rounds` and total can shrink
 * differently and cost differently. */
struct DraftCostSample {
    uint64_t cycle_id = 0;      /* the plan this sample belongs to; 0 = unattributed */
    uint32_t policy_version = 0;
    DraftVerifyShape shape;
    DraftReplay replay = DraftReplay::Unknown;  /* what the executing path did, not what the plan
                                                 * declared; Unknown when it cannot be evidenced */
    int32_t rounds = 0;         /* deepest draft round the head actually ran */
    int32_t draft_round_rows[RAD_SCHED_MAX_SPEC + 1] = {}; /* actual N_j per round, index 0 unused */
    int32_t draft_rows_total = 0; /* sum_j draft_round_rows[j] actually executed */
    int32_t drafted = 0;        /* draft tokens produced */
    double  total_ms = 0.0;     /* the whole cycle: trunk step, draft rounds, commit */
    /* Split timings, -1 when the phase was not timed. Under overlap they do not add up to
     * total_ms; use them to calibrate components, the total to compare arms. */
    double  target_ms = -1.0;
    double  draft_ms = -1.0;
    double  host_ms = -1.0;
    uint16_t flags = 0;         /* DraftSampleFlag */
    /* The per-row provenance of the verify window the shape describes (index i = row i), from
     * DraftSeqObservation::label_origin. `origins_valid` is the caller's statement that the
     * executing path captured it; a profile writer refuses a bucket without it rather than
     * defaulting rows to None. Not part of any key -- see draft_cost.h's note. */
    DraftOrigin origins[RAD_SCHED_MAX_SPEC] = {};
    uint8_t     origins_valid = 0;
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
    uint64_t cycle_id = 0;             /* the view's cycle this set was computed for */
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
    int32_t  pipeline_depth = 0; /* cycles this plan is issued ahead; 0 = verified by the next
                                  * step. Part of the plan's immutable record, not a live query. */
    DraftPlanSource source = DraftPlanSource::Legacy;
    DraftFallback   fallback = DraftFallback::None;  /* set when source is Fallback */
    DraftProbe      probe = DraftProbe::None;
    DraftReplay     replay = DraftReplay::Unknown;   /* the cycle's replay state, part of its shape;
                                                      * None only when proven absent */
    bool            timing_valid = false;            /* total_ms is a real bracket for this plan */
    int32_t         prefill_tokens = 0;              /* prefill sharing the step; 0 decode-only */

    /* The shape the plan was priced on, per request where it varies: a historical plan read under
     * issue_ahead() must still say it, and a K0 plan with history_tokens > 0 is "policy K0 with
     * history", not "no work". */
    int32_t*        ctx_class = nullptr;      /* [cap] draft_ctx_class of the request's context */
    int32_t*        history_tokens = nullptr; /* [cap] MTP history/QSA rows the request carried */

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
 * and checked and still censored before it is committed. `end` classifies the cut; `origin` says
 * who produced this row (the plan's fill[] is only the permission). Both are part of the
 * observation, so no caller keeps a side list that can drift from the cycle.
 *
 * The counts have two origins and neither orders the other: `generated` and `published` are this
 * plan's drafting; `verified`, `checked`, `accepted` and `raw_match` are the paired verify, which
 * under issue-ahead verifies a window a previous cycle published. See
 * draft_seq_observation_valid for what is ordered inside each. */
struct DraftSeqObservation {
    uint64_t raw_match = 0;  /* consecutive drafts that matched the target's own draw, uncensored:
                              * accept_greedy's count, before the end-of-generation cut */
    uint8_t  generated = 0;  /* draft tokens the head produced */
    uint8_t  published = 0;  /* tokens actually published to the proposal buffer */
    uint8_t  verified = 0;   /* draft positions the target computed (never the bonus token) */
    uint8_t  checked = 0;    /* positions with a label: accepted prefix plus the first rejection */
    /* Drafts the step committed, i.e. the drafts whose token the target accepted. Not the number
     * of tokens the step emitted: the emitted prefix is those drafts plus the target's own token at
     * the next position, which is a draft too when `accepted == verified` and the end-of-generation
     * cut otherwise. So a step emits `accepted + 1` tokens, and reading `accepted` as a token count
     * understates every step by one. */
    uint8_t  accepted = 0;
    DraftSeqEnd end = DraftSeqEnd::Unset;
    DraftOrigin origin = DraftOrigin::None;
    /* The provenance of the window the verify actually consumed. With pipeline_depth > 0 that
     * window was published by the previous cycle's plan, so `origin`/`generated` (what THIS plan
     * drafted) and `label_origin`/`label_generated` (what the verify measured) name different
     * plans. Only `label_*` says whose labels `checked`/`verified` carry. */
    DraftOrigin label_origin = DraftOrigin::None;
    uint8_t     label_generated = 0;
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

/* Whether `o` observes the plan it names: the same memory set, the same cycle and policy version,
 * and per row the same request identity. A ring slot reused by a newer cycle changes req_id or
 * slot_gen, so this is the check that stops one cycle's rows from teaching another cycle's plan.
 * One memory set per live cycle is the caller's rule: a plan must not be rewritten while an
 * observation or an issued-ahead pass still names it. */
inline bool draft_plan_identifies(const DraftPlan& p, const DraftObservation& o) {
    if (o.plan != &p) return false;
    if (p.cycle_id == 0 || p.cycle_id != o.cycle_id) return false;
    if (p.policy_version != o.policy_version) return false;
    /* Both capacities, before the identity loop: a plan that claims more rows than the storage
     * it was sized for would make this check the out-of-bounds read it is meant to prevent. */
    if (p.n_seq != o.n_seq || p.n_seq < 0 || p.n_seq > p.cap || o.n_seq > o.cap) return false;
    if (p.n_seq == 0) return true;
    if (!p.req_id || !p.slot || !p.slot_gen || !o.req_id || !o.slot || !o.slot_gen) return false;
    for (int32_t i = 0; i < p.n_seq; ++i) {
        if (p.req_id[i] != o.req_id[i] || p.slot[i] != o.slot[i] || p.slot_gen[i] != o.slot_gen[i])
            return false;
    }
    return true;
}

/* Fill rows[1..max_k] with N_j for the plan's k[]. Requires max_k == the deepest k and every k in
 * 0..max_k; index 0 and the rest of the array are zeroed. RAD_OK, or RAD_E_INVAL. */
inline int draft_plan_rows(DraftPlan& p) {
    if (!p.k || !p.rows) return RAD_E_INVAL;
    if (p.n_seq < 0 || p.n_seq > p.cap) {
        RAD_ERR("draft plan: n_seq %d exceeds the caller's %d slots", (int)p.n_seq, (int)p.cap);
        return RAD_E_INVAL;
    }
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
    if (!p.k || !p.rows || !p.fill || !p.ctx_class || !p.history_tokens) return RAD_E_INVAL;
    if (p.prefill_tokens < 0 || p.pipeline_depth < 0) return RAD_E_INVAL;
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
        if (p.ctx_class[i] < 0 || p.ctx_class[i] >= RAD_DRAFT_CTX_CLASSES ||
            p.history_tokens[i] < 0) {
            RAD_ERR("draft plan: request %d has ctx class %d and %d history rows", (int)i,
                    (int)p.ctx_class[i], (int)p.history_tokens[i]);
            return RAD_E_INVAL;
        }
        for (int j = 1; j <= p.k[i]; ++j) ++count[j];
    }
    if (p.max_k > 0 && count[p.max_k] == 0) {
        RAD_ERR("draft plan: max_k is %d but no request drafts that far", (int)p.max_k);
        return RAD_E_INVAL;
    }
    for (int j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) {
        if (p.rows[j] != count[j]) {
            RAD_ERR("draft plan: rows[%d] is %d, k[] implies %d", j, (int)p.rows[j],
                    (int)count[j]);
            return RAD_E_INVAL;
        }
    }
    /* Not issued ahead: the plan drafts and the paired verify pass runs on the same window, so
     * the verify-K it names is its own max_k. Under pipeline_depth > 0 the verify pass belongs to
     * the cycle the paired step verifies, and a different K is the point of issuing ahead. Last,
     * so the structural checks above keep their own diagnostics. */
    if (p.pipeline_depth == 0 && p.verify_k != p.max_k) {
        RAD_ERR("draft plan: verifies %d but drafts %d at pipeline depth 0", (int)p.verify_k,
                (int)p.max_k);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* True when request i may not be filled by any drafter this cycle. The scheduler's n-gram
 * fallback is a drafter too and must ask this before it fills an empty proposal: a K0 plan is a
 * plan, not an absence of one, and `k[]` is the budget for every drafter. */
inline bool draft_plan_bars_fill(const DraftPlan& p, int32_t i) {
    return p.k && p.fill && i >= 0 && i < p.n_seq && p.k[i] == 0 && p.fill[i] == DraftDrafter::None;
}

/* Two version checks, deliberately different. A prediction set describes the view it was computed
 * for, so its cycle, version and rows must be the view's; reading the hook's current version
 * instead would hide a stale set. The cycle is part of that identity because a set computed for an
 * earlier cycle carries that cycle's plan parameters even when every `req_id` still matches. The
 * *history* a request owns is not this struct -- it outlives cycles and is keyed by `req_id` alone.
 * An observation describes the plan it names, so its version must be that plan's
 * (draft_plan_identifies), not the version the hooks carry now: a plan issued under an older policy
 * stays observable while it is alive. Neither check is the other's guard. */
inline bool draft_predictions_current(const DraftBatchView& view, const DraftPredictions& p) {
    if (p.cycle_id != view.cycle_id) return false;
    if (p.version != view.policy_version) return false;
    if (p.n_seq != view.n_seq || p.n_seq < 0 || p.n_seq > p.cap) return false;
    if (p.n_seq == 0) return true;
    if (!p.req_id || !p.seq || !view.seq) return false;
    for (int32_t i = 0; i < p.n_seq; ++i) {
        if (p.req_id[i] != view.seq[i].req_id) return false;
    }
    return true;
}

/* A row that may be learned from: the counts are ordered *within each of the two windows they come
 * from*, `raw_match` counts matches among what the paired verify computed and covers the committed
 * prefix, and `end` classifies the cut. An unclassified row is refused instead of guessed.
 *
 * The two windows are not interchangeable, and the row carries both:
 *   - this plan's drafting -- `generated >= published`;
 *   - the paired verify -- `verified >= checked >= accepted`, and the tally is a comparison, so
 *     `raw_match <= verified` and `raw_match >= accepted`.
 * `accepted` belongs to the verify cycle, not to this plan's draft output. A K0 plan issued ahead
 * verifies the window a previously published plan produced, and commits drafts this row never
 * published: `accepted > published` is a measured cycle, not a contradiction. Chaining the two
 * would refuse it and lose its labels; writing a synthetic zero to fit the chain would lose the
 * acceptance itself. `verified` is likewise *not* chained to `published`: a plan issued ahead drafts
 * and verifies different windows, so a verify window above the draft window makes
 * `verified > published` and one below makes `verified < published`, both legitimate. Binding them
 * is the confusion `verify_k` exists to avoid.
 *
 * `raw_match` is accept_greedy's uncensored tally, taken before `accept_cut_at_eog`: the cut
 * shortens what is committed, not what was compared. Its window is the paired verify's, which is
 * why `raw_match <= verified` and not `<= generated`: a match is a comparison the target actually
 * made, and reading `k[i]` for the tally would bound it by a count from the other window.
 *
 * `raw_match` is bounded further where it protects a label and nowhere else. `Rejected` writes a 0
 * at position `accepted` (draft_label), so the tally must agree that this position did not match:
 * `raw_match == accepted`. `AllAccepted` claims everything it checked was committed, so a taller
 * tally contradicts it. `Stopped` carries no label at `accepted` at all, and the censor -- not the
 * tally -- decides where commitment stopped: `DraftController::observe` takes the committed count
 * from the card, so a budget or cancellation censor can stop it below the compared span. An upper
 * bound there would refuse a legitimate row and lose every label in that cycle, so `Stopped` keeps
 * only the general `raw_match >= accepted`. */
inline bool draft_seq_observation_valid(const DraftSeqObservation& o) {
    const int a = (int)o.accepted, c = (int)o.checked, v = (int)o.verified;
    if (o.generated > RAD_SCHED_MAX_SPEC || o.published > RAD_SCHED_MAX_SPEC) return false;
    if (a > RAD_SCHED_MAX_SPEC || c > RAD_SCHED_MAX_SPEC || v > RAD_SCHED_MAX_SPEC) return false;
    /* The label provenance is a state and an amount that must agree: a row the verified window
     * did not draft carries no labels from it, whatever this plan drafted. */
    if (o.label_generated > RAD_SCHED_MAX_SPEC) return false;
    if ((o.label_origin == DraftOrigin::None) != (o.label_generated == 0)) return false;
    /* This plan's drafting. `a > published` is not a drafting error: `accepted` counts the paired
     * verify's commits, which can be drafts another plan published. See the header comment. */
    if (o.published > o.generated) return false;
    /* The paired verify's window. `v > published` is not an error either. */
    if (a > c || c > v) return false;
    if (o.raw_match > (uint64_t)v) return false;
    if (o.raw_match < (uint64_t)a) return false;
    switch (o.end) {
        case DraftSeqEnd::AllAccepted: return c == a && o.raw_match == (uint64_t)a;
        case DraftSeqEnd::Rejected:    return c == a + 1 && o.raw_match == (uint64_t)a;
        case DraftSeqEnd::Stopped:     return c == a;   /* no label at `accepted`, so no tally bound */
        case DraftSeqEnd::Unset:       return false;
    }
    return false;
}

/* The actual origin must fit the plan's permission: a row nobody was allowed to fill may not claim
 * proposals (that is the K0/fallback leak), the head and the lookup fallback cannot stand in for
 * one another, and the amount is bounded the same way -- `k[i]` is the budget for every drafter, so
 * a row that reports more proposals than that budget, or claims proposals while reporting none, is
 * not this plan's work. A row that produced nothing is None, whatever its budget was. */
inline bool draft_origin_fits_plan(const DraftPlan& p, int32_t i, const DraftSeqObservation& o) {
    if (i < 0 || i >= p.n_seq || !p.k || !p.fill) return false;
    switch (o.origin) {
        case DraftOrigin::None:   return o.generated == 0;
        case DraftOrigin::Head:
            return p.k[i] > 0 && p.fill[i] == DraftDrafter::Serial && o.generated > 0 &&
                   o.generated <= p.k[i];
        case DraftOrigin::Lookup:
            return p.k[i] > 0 && p.fill[i] == DraftDrafter::Ngram && o.generated > 0 &&
                   o.generated <= p.k[i];
    }
    return false;
}

/* The lifetime gate for a completed cycle, and the only one. It never consults the live slot: a
 * request preempted after its pass ran, a slot recycled for a new generation, and a resumed
 * request all leave this observation deliverable, because its rows are the plan's and both
 * identities travel with it. Refusing on "the slot now holds another generation" would drop a
 * measured cycle; accepting a row that does not fit its plan's permission would teach another
 * cycle's work. Identity is not enough either: the plan it names must be a plan at all, or a broken
 * choice would be delivered as a measured cycle with rows checked against a shape nobody issued. */
inline bool draft_observation_deliverable(const DraftObservation& o) {
    if (!o.complete || !o.plan) return false;
    if (!draft_plan_identifies(*o.plan, o)) return false;
    if (draft_plan_valid(*o.plan) != RAD_OK) return false;
    if (o.n_seq > 0 && !o.seq) return false;
    const DraftPlan& p = *o.plan;
    for (int32_t i = 0; i < o.n_seq; ++i) {
        const DraftSeqObservation& s = o.seq[i];
        if (!draft_seq_observation_valid(s)) return false;
        if (!draft_origin_fits_plan(p, i, s)) return false;
        /* The two windows are per-row quantities, not a single K: this row's drafter may only
         * produce what its own k[i] paid for, and it may only claim positions its own provenance's
         * verify step actually ran. A union of the two candidate windows is not that check: it
         * admits a head row at the deepest row's K and a lookup row at the frozen plan's window.
         * The head publishes ahead, so with pipeline_depth > 0 its labels were verified by the
         * paired step over the frozen `verify_k`, and at depth 0 by the same pass over this row's
         * k[i] (draft_plan_valid binds verify_k == max_k there). The lookup fills in the step that
         * verifies it, so its window is k[i]. A None row consumed no window at all. */
        int32_t window;
        switch (s.label_origin) {
            case DraftOrigin::None:   window = 0; break;
            case DraftOrigin::Head:   window = p.pipeline_depth > 0 ? p.verify_k : p.k[i]; break;
            case DraftOrigin::Lookup: window = p.k[i]; break;
            default: return false;
        }
        if ((int32_t)s.generated > p.k[i]) return false;
        if ((int32_t)s.verified > window) return false;
        /* The labels are what that same window produced, so a row cannot claim a measurement
         * wider than the window it says it consumed. */
        if ((int32_t)s.label_generated > window) return false;
        /* One row per request per cycle: two rows for the same request would apply one cycle's
         * gradient to the same state twice. `draft_plan_identifies` already ties row i to row i
         * of the plan, so this only has to catch a plan/view that names one request twice. */
        for (int32_t j = 0; j < i; ++j) {
            if (o.req_id[j] == o.req_id[i]) return false;
        }
    }
    return true;
}

/* Whether the shape is a key the profile may be asked for. `unknown` is the explicit "not
 * measured" marker: a missing history, prefill, replay or draft distribution is refused, never
 * priced as a zero. Both distributions must be monotone and complete, verify_tokens must equal
 * n_rows + sum_j rows[j], and no stale entry may survive past the deepest round of either. A round
 * with no rows at all is not a measured zero: `verify_k > 0` with `rows[verify_k] == 0` is the same
 * missing shape as `unknown`, and the mirror of the draft rule below. */
inline bool draft_shape_priceable(const DraftVerifyShape& s) {
    if (s.unknown != 0) return false;
    if (s.n_rows < 0 || s.n_rows > RAD_SCHED_MAX_SPEC) return false;
    if (s.verify_k < 0 || s.verify_k > RAD_SCHED_MAX_SPEC) return false;
    if (s.draft_k < 0 || s.draft_k > RAD_SCHED_MAX_SPEC) return false;
    if (s.ctx_class < 0 || s.ctx_class >= RAD_DRAFT_CTX_CLASSES) return false;
    if (s.history_tokens < 0 || s.prefill_tokens < 0 || s.pipeline_depth < 0) return false;
    int32_t tokens = s.n_rows, prev = s.n_rows;
    for (int j = 1; j <= s.verify_k; ++j) {
        if (s.rows[j] < 0 || s.rows[j] > prev) return false;
        prev = s.rows[j];
        tokens += prev;
    }
    if (s.verify_tokens != tokens) return false;
    prev = s.n_rows;
    for (int j = 1; j <= s.draft_k; ++j) {
        if (s.draft_rows[j] < 0 || s.draft_rows[j] > prev) return false;
        prev = s.draft_rows[j];
    }
    if (s.verify_k > 0 && s.rows[s.verify_k] == 0) return false;
    if (s.draft_k > 0 && s.draft_rows[s.draft_k] == 0) return false;
    for (int j = s.verify_k + 1; j <= RAD_SCHED_MAX_SPEC; ++j) {
        if (s.rows[j] != 0) return false;
    }
    for (int j = s.draft_k + 1; j <= RAD_SCHED_MAX_SPEC; ++j) {
        if (s.draft_rows[j] != 0) return false;
    }
    return true;
}

/* The planned draft distribution must be the cycle's own: same request count, same deepest round,
 * same N_j as the plan's k[], and the same pipeline depth. Under pipeline_depth > 0 the verify
 * distribution belongs to the cycle the paired step verifies, so a different K is legitimate and
 * only draft_shape_priceable bounds it. At pipeline_depth == 0 the plan drafts and verifies the
 * same window (draft_plan_valid binds verify_k to max_k), so the verify distribution must be the
 * plan's too -- otherwise a shape priced for another K would be stored under this plan's identity. */
inline bool draft_shape_matches_plan(const DraftVerifyShape& s, const DraftPlan& p) {
    if (s.n_rows != p.n_seq || s.draft_k != p.max_k) return false;
    if (s.pipeline_depth != p.pipeline_depth) return false;
    if (!p.rows) return false;
    for (int j = 1; j <= p.max_k; ++j) {
        if (s.draft_rows[j] != p.rows[j]) return false;
    }
    if (p.pipeline_depth == 0) {
        if (s.verify_k != p.verify_k) return false;
        for (int j = 1; j <= p.verify_k; ++j) {
            if (s.rows[j] != p.rows[j]) return false;
        }
    }
    return true;
}

/* The label for draft position j (0-based), if this cycle measured one: 1 on the accepted prefix,
 * 0 at a real rejection, and nothing else. The position at `accepted` is labelled only when
 * `end == Rejected` -- a matched stopping token (Stopped) is compared and not committed, so giving
 * it a 0 would teach the head that a correct EOG was wrong. `checked` ends the labelled span, so a
 * censor past a rejection does not erase it. */
inline bool draft_label(const DraftSeqObservation& o, int j, float* label) {
    if (j < 0 || j >= (int)o.checked) return false;
    if (j < (int)o.accepted) {
        if (label) *label = 1.0f;
        return true;
    }
    if (o.end == DraftSeqEnd::Rejected && j == (int)o.accepted) {
        if (label) *label = 0.0f;
        return true;
    }
    return false;
}

/* A bracket that may be stored at all: marked valid, finite and positive. Storage and attribution
 * are different questions -- a record or audit bracket is stored (storable) in its own bucket but
 * never updates a play bucket (not attributable). */
inline bool draft_sample_storable(const DraftCostSample& s) {
    return (s.flags & kDraftSampleValid) && std::isfinite(s.total_ms) && s.total_ms > 0.0;
}

/* Only a sample that names the cycle it timed, is storable, and carried no recording, audit,
 * admission, placement or warmup inside the bracket may update a measured play bucket. An ahead
 * sample may, if the caller checked the bracket; it stays flagged so the model can weigh it. Real
 * DMA cost is not an outlier: the placement flag routes it to its own bucket, it is not dropped.
 * Record and audit samples have no path here at all; they are stored through
 * draft_sample_storable and never folded into a play bucket.
 *
 * The bucket is keyed by replay state, so that state has to be the run's, twice: the executing path
 * must have established one (`Unknown` fills no bucket), and the key the sample is filed under must
 * be that same state. A plan's declared `None` is a declaration, not evidence that no replay ran;
 * letting it key a sample whose replay state the run could not establish is what prices a shape
 * with a fact nobody measured. */
inline bool draft_sample_attributable(const DraftCostSample& s) {
    const uint16_t disturbing = kDraftSampleWarmup | kDraftSampleRecord | kDraftSampleAudit |
                                kDraftSampleAdmission | kDraftSamplePlacement |
                                kDraftSampleShortened;
    if (s.cycle_id == 0) return false;   /* nothing to attribute it to */
    if (s.replay == DraftReplay::Unknown) return false;
    if (s.shape.replay != s.replay) return false;
    return draft_sample_storable(s) && !(s.flags & disturbing);
}

/* The sample must also describe the cycle the observation names: a sample from another cycle or
 * policy version is not this observation's timing, however clean its bracket. Only a completed
 * cycle over a valid plan is attributable at all -- an unfinished cycle has no bracket to fold in,
 * and a plan that fails its own gate names no shape the sample could have been measured under. */
inline bool draft_sample_attributable(const DraftObservation& o, const DraftCostSample& s) {
    if (!o.complete || !o.plan) return false;
    if (!draft_plan_identifies(*o.plan, o)) return false;
    if (draft_plan_valid(*o.plan) != RAD_OK) return false;
    return s.cycle_id != 0 && s.cycle_id == o.cycle_id && s.policy_version == o.policy_version &&
           draft_sample_attributable(s);
}

}  /* namespace rad */
