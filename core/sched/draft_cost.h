/* draft_cost.h -- strand A: the measured cost profile and the model that prices a verify shape.
 *
 * The policy (C) compares candidate plans by their expected batch time, so it asks the
 * DraftCostModel seam for a price in milliseconds for a DraftVerifyShape. This is that price: a
 * versioned profile of measured buckets (`rad-draft-profile/4`, docs/ADAPTIVE-MTP.md), keyed by
 * the verify shape -- verify tokens and N_j, pipeline depth, context class, history and prefill
 * tokens, sampling, replay state and the loaded drafter -- paired with the draft distribution that
 * actually ran, never by (rows, max K) alone.
 *
 * The price is the cost of the executed distribution. `shape.rows[]` is the verify window the cycle
 * paid for; the draft work it paid for is `rounds` and `DraftCostBucket::draft_round_rows[]`, what
 * actually ran. A cycle may legally stop before its plan's deepest round, so its bracket times that
 * shorter execution and not the plan it took. The lookup key is therefore the sample's shape with
 * the executed distribution in place of the plan's: a query names in `shape.draft_k` /
 * `draft_rows[]` the distribution it plans to execute, and that distribution is compared against
 * the executed one. The plan's own distribution stays in the bucket's shape -- it is what the sample
 * was attributed against (`draft_shape_matches_plan`, `executed_within_plan`) and the upper bound of
 * the execution, so a bucket measured at a shorter run never matches a query that plans the deeper
 * one. The bucket's `shape.draft_rows[]` is therefore provenance, not key: it is what the sample
 * was attributed against, and no query is matched on it.
 *
 * A missing measurement is not a zero. A shape carrying an unknown dimension is refused by
 * draft_shape_priceable() and by the reader, never priced as if its zeros had been measured.
 *
 * One axis this reader still cannot discriminate is the per-row origin mix: the file carries it
 * and it is validated against the loaded drafter, but DraftVerifyShape has no origins field, so a
 * query can only express the drafter, not the mix (interface request against the shared header,
 * not an approximation made here).
 *
 * The reader runs off the step path: read()/configure() happen at startup. estimate() and
 * observe() are allocation-free. The bucket vector is reserved at configure; an observation that
 * would need a bucket past that capacity is counted in dropped() instead of growing the vector,
 * so telemetry overflow never blocks the card.
 *
 * Units are milliseconds or nothing. A shape with no bucket is priced by one documented
 * interpolation step (the neighbouring context class) or by a heuristic fallback: a measured
 * anchor that ran the same executed draft distribution and the same history and prefill work,
 * scaled by the verify-token ratio plus a fixed margin. History and prefill are part of the class
 * because no measured scaling exists for them -- borrowing across a different amount of either
 * would be a bound on another measurement. The fallback scales verify work only -- it never prices
 * draft work it has not measured, so a query planning more draft rounds than any bucket ran gets no
 * number and the caller's named baseline instead of a shallower run's cost. That margin is a rule
 * of thumb for preferring a bound over nothing, not a proven upper bound, and the estimate carries
 * it as uncertainty. With no anchor at all the estimate is invalid and the caller uses its named
 * fixed baseline. A unitless prior is never reported as milliseconds.
 *
 * Only a sample draft_sample_attributable() accepts updates a measured bucket, and only when its
 * identity is known: a sample whose cycle_id or policy_version is 0 (unattributed) is refused, and
 * so is one that disagrees with its observation or with the plan the observation names. Recording,
 * audit, warmup, admission and placement brackets are kept as diagnostic records with their flags
 * and are never returned by estimate(). An ahead sample may update (the caller checked its
 * bracket) and keeps the ahead flag.
 */
#pragma once
#include "rad_core.h"
#include "sched/draft_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* The profile's identity. A profile is used only when every field matches the running
 * deployment; a mismatch disables it (the caller logs the first difference and names
 * DraftFallback::NoProfile) rather than rescaling a file measured on other hardware.
 *
 * `placement` is the resolved plan, not the flag: `auto` is resolved against a budget that depends
 * on the free VRAM at start, so the flag value would let two differently placed deployments share
 * a profile. It carries, per rank in rank order, the strategy, the resolved budgets and a digest
 * of the plan's tier/site per weight -- the expert residency a strategy name alone does not name.
 * The engine refuses to derive a runtime from a plan that is still `auto`.
 *
 * `basis` says how the numbers were measured and selects the comparison rule: a `NoLoan`
 * reference compares the plan, the budgets and the arena levels and ignores `live_loan`; a `Live`
 * profile compares the loan as well. `arena` is the level-row identity from the start's arena
 * plan (`Engine::arena_rows_`): the level a step runs at changes the activation geometry, and two
 * ceilings that resolve the same level set execute the same geometry while two different sets do
 * not. `live_loan` is the per-rank `flex=`/`lend=` the start loan covers -- the old part of
 * `placement`, now a field of its own so a no-loan reference does not have to name a loan of
 * zero. */
struct DraftCostFingerprint {
    std::string model;
    std::string quant;
    std::string tp_wire;
    std::string gpus;
    std::string kernels;
    std::string kv_format;
    std::string placement;
    DraftCostBasis basis = DraftCostBasis::Live;
    std::string arena;
    std::string live_loan;
    int32_t     tp = 0;
};

/* Compares two fingerprints; `first` names the first differing field, for the log and the plan. */
bool draft_cost_fingerprint_same(const DraftCostFingerprint& a, const DraftCostFingerprint& b,
                                 std::string* first);

/* The backend's round model -- how the active drafter issues draft work, not how one cycle's K
 * happened to fall. Shrinking is a serial head that runs one round per row, so N_j falls with j
 * whenever the row depths differ -- and stays shrinking in a cycle whose K are all equal; the
 * distribution is the bucket key, not the profile identity. Rectangular is a backend without
 * per-row rounds (a block drafter in one pass over `block` positions), a disabled drafter, or
 * drafting switched off. The two run different kernels, so they are different profiles, and a
 * profile for the other execution is refused, not rescaled. */
enum class DraftCostExecution : uint8_t { Rectangular = 0, Shrinking = 1 };

/* The backend's round model from the drafter that is actually active: `serial_head` is a serial
 * drafter with a positive depth, and `n_spec` is its resolved depth (0 when drafting is off, and
 * already resolved from auto before the scheduler starts). The reader and the writer both derive
 * the execution here, so a record cannot be stamped with anything but what the engine is. */
inline DraftCostExecution draft_cost_execution(bool serial_head, int32_t n_spec) {
    return serial_head && n_spec > 0 ? DraftCostExecution::Shrinking
                                     : DraftCostExecution::Rectangular;
}

/* What the running deployment is: the profile is accepted only against this. */
struct DraftCostRuntime {
    DraftCostFingerprint fingerprint;
    DraftCostExecution   execution = DraftCostExecution::Rectangular;
    /* The table is a calibration's recording target, not a loaded reference. Both can carry a
     * no-loan fingerprint; only the loaded one must stay immutable. */
    bool recording = false;
};

/* One measured bucket. Its price is the cost of the executed distribution: the lookup key is the
 * sample's shape -- verify window, pipeline depth, context class, history and prefill tokens,
 * sampling, replay and the loaded drafter -- paired with `rounds` and `draft_round_rows[]`, what
 * actually ran. Two cycles that verified the same window but ran a different draft distribution are
 * different kernels and stay separate: the same round count and the same total rows are not enough,
 * [3,1,1] and [2,2,1] are two measurements.
 *
 * `shape.draft_k` / `shape.draft_rows[]` are not part of that key. They keep the plan the samples
 * took, which the execution could not exceed and which therefore bounds it: the bucket's provenance
 * and its upper bound, so a query planning a deeper distribution does not match.
 *
 * The origin set (`shape.origins` in the file) is validated against the bucket's own drafter and
 * kept as provenance, but it is not part of the key: DraftVerifyShape has no origins field, so a
 * query could not discriminate on it anyway. See the note on that interface request above.
 *
 * `flags` are the attribution flags of the samples it was built from; a bucket carrying a disturbing
 * flag -- or the online-only kDraftBucketUnpriced -- is a diagnostic record and is never used to
 * price a shape. */
struct DraftCostBucket {
    DraftVerifyShape shape;
    uint16_t flags = 0;
    /* The per-row provenance of the verify window the shape describes, captured from the samples
     * that built it (DraftCostSample::origins). Provenance only, never a key. Each row holds the
     * *set* of origins its window came from -- bit (int)DraftOrigin -- because the cycles behind
     * one bucket need not agree: the head drops a row in one cycle and the lookup fallback fills
     * it in the next. A single value per row would either pick one sample's mix or throw the whole
     * export away over a difference the key does not even see, so the file carries the set
     * ("lookup", "head+lookup"). A row with no bit at all came from no sample that could name it;
     * the writer refuses that bucket rather than inventing one. */
    uint8_t origin_set[RAD_SCHED_MAX_SPEC] = {};
    /* The executed extent. rounds = -1 means "unknown", which matches no query. */
    int32_t  rounds = -1;       /* deepest draft round that actually ran */
    int32_t  draft_round_rows[RAD_SCHED_MAX_SPEC + 1] = {};  /* actual N_j, index 0 unused */
    int32_t  draft_rows_total = -1;                          /* sum_j draft_round_rows[j] */
    double   mean_ms = 0.0;     /* the amortised batch time at the same pipeline depth */
    double   unc_ms = 0.0;      /* half-width of mean_ms; a heuristic, not a bound on a future run */
    int32_t  n = 0;             /* samples behind the mean */
    int32_t  n_ahead = 0;       /* of them issued ahead; online only -- a profile keeps the flag, not the count */
    /* Split timings, -1 when not timed. Under overlap they do not add up to mean_ms. */
    double   target_ms = -1.0;
    double   draft_ms = -1.0;
    double   host_ms = -1.0;
};

struct DraftCostProfile {
    DraftCostFingerprint fingerprint;
    DraftCostExecution   execution = DraftCostExecution::Rectangular;
    std::vector<DraftCostBucket> buckets;
};

/* Whether a bucket may price normal operation at all: measured cleanly (no disturbing flag), at
 * least one sample, a finite positive mean. DraftCostTable::estimate()/has_usable_bucket() and the
 * calibration export use this one rule -- an export that kept only diagnostic buckets would
 * otherwise report a profile that can never price as a success. */
bool draft_bucket_usable(const DraftCostBucket& b);

/* Parses and validates one profile. RAD_OK, or a negative RAD_E_* (RAD_E_UNSUPPORTED for a
 * format id this reader does not know -- rad-draft-profile/1 has no draft distribution, pipeline
 * depth or origin mix, and rad-draft-profile/2 has no resolved placement or device equipment in
 * its fingerprint, so neither can be bound to a deployment; RAD_E_INVAL for a malformed or
 * inconsistent file) with `err` naming the problem. */
int draft_cost_profile_parse(const std::string& text, DraftCostProfile* out, std::string* err);
/* Reads `path` and parses it. The file is not on the step path. */
int draft_cost_profile_read(const std::string& path, DraftCostProfile* out, std::string* err);

/* The inverse reader: `p` as a rad-draft-profile/4 file, in `*out`. Refuses (RAD_E_INVAL with
 * `err` naming the bucket) rather than inventing what it cannot state: a bucket whose origin mix
 * was not captured, or whose origins do not fit its drafter, is not written. Buckets carrying a
 * flag the file cannot name -- the runtime-only kDraftSampleShortened / kDraftBucketUnpriced -- are
 * diagnostics, are left out, and are counted in `*skipped` (optional); the caller reports them.
 * A profile that would end up with no bucket at all is refused: an empty file is not a
 * calibration, and a reader must never take one for a baseline. The text parses back with
 * draft_cost_profile_parse. */
int draft_cost_profile_write(const DraftCostProfile& p, std::string* out, int32_t* skipped,
                             std::string* err);
/* Writes `p` to `path` atomically and never over an existing file: the text is built and parsed
 * back first, written and fsynced as an O_EXCL temporary in the same directory, then published
 * with link(), which fails with EEXIST instead of replacing, and the directory is synced before
 * the write is reported as done. Every failure is RAD_E_IO (or the writer's own refusal) with
 * `err` naming the path and the reason -- an export that did not happen is never silent. */
int draft_cost_profile_write_file(const std::string& path, const DraftCostProfile& p,
                                  int32_t* skipped, std::string* err);

/* The `gpus` fingerprint field from the ranks' device names, in rank order: identical names
 * collapse to "Nx NAME", mixed ones stay comma-separated, an empty list is "none". */
std::string draft_cost_gpus(const std::vector<std::string>& names);

/* How many buckets a table reserves by default: 2048 covers the wave-1 matrix (4 context classes
 * x 7 depths x the documented N_j windows x sampling x replay) with headroom. */
enum { kDraftCostDefaultBuckets = 2048 };

/* The per-context-class interpolation allowance: a bucket borrowed from the neighbouring class
 * is widened by this fraction of its mean, propagated into the estimate's uncertainty. */
constexpr double kDraftCostCtxStep = 0.10;
/* The heuristic fallback's margin over the anchor's verify-token scaling: a rule of thumb for
 * preferring a bound over nothing, not a proven upper bound. It is reported through
 * DraftCostEstimate::unc_ms, so the caller can see how much of the number is the rule. */
constexpr double kDraftCostFallbackMargin = 0.25;

/* Strand-A bucket flag, created online only: the sample was a truncation -- the plan's deepest
 * round never ran -- so its bracket carries the cost of stopping early and prices no plan.
 * observe() keeps it for diagnosis and never returns it by estimate(); a profile file has no name
 * for it and it is never written. */
enum : uint16_t { kDraftBucketUnpriced = 1u << 15 };

/* The profile in memory, plus the online updates from completed cycles. configure() copies the
 * profile; the caller's DraftCostProfile and its storage are not referenced afterwards. */
class DraftCostTable {
public:
    DraftCostTable() = default;
    DraftCostTable(const DraftCostTable&) = delete;
    DraftCostTable& operator=(const DraftCostTable&) = delete;

    /* Installs a validated profile for this deployment. Returns RAD_OK even when the fingerprint
     * or the execution does not match: the table then stays empty and reason() names the
     * mismatch, so a refused profile is a visible fallback, not a parse error. RAD_E_INVAL for a
     * malformed profile (no fingerprint field, a bucket with an invalid shape). A `NoLoan`
     * profile becomes an immutable reference unless this table is a calibration's recording
     * target (`runtime.recording`). */
    int configure(const DraftCostProfile& profile, const DraftCostRuntime& runtime,
                  int32_t capacity = kDraftCostDefaultBuckets);

    /* No profile: every estimate is invalid. */
    void clear();

    /* The deployment's explicit admission of an empty baseline: it accepts that Cost runs on the
     * policy's named fixed baseline until a cycle has been measured. `active()` alone is not that
     * admission -- a matching profile with no bucket proves the reader ran, not that anyone
     * accepted pricing without measurements (docs/ADAPTIVE-MTP.md). Cleared by clear(), so a
     * reset table cannot inherit an earlier deployment's admission. */
    void admit_fallback() { fallback_admitted_ = true; }
    bool fallback_admitted() const { return fallback_admitted_; }

    /* A calibration measures, it never prices: observe() files the cycles the run executes, while
     * estimate() reports no price at all. A cost run measuring its own profile would otherwise
     * decide with the buckets it is still filling -- the first cycle's measurement feeding the
     * second cycle's plan -- and the exported numbers would describe a run that was steered by
     * them. The fallback baseline is what such a run executes, and exactly what it measures. */
    void record_only() { record_only_ = true; }
    bool records_only() const { return record_only_; }

    /* The loaded profile is a no-loan reference (DraftCostBasis::NoLoan): it is an optimisation
     * basis, not a bound, and it is immutable. estimate() answers only from exact Measured
     * buckets -- no interpolation, no conservative fallback -- and observe() counts the cycle
     * (`reference_skipped()`) without touching a bucket. Configured from the file's basis; a
     * calibration's recording table (`DraftCostRuntime::recording`) is never a reference. */
    bool reference() const { return reference_; }
    int64_t reference_skipped() const { return reference_skipped_; }

    /* The start placement is not materialised yet (Engine::start_placement_settled): a profile
     * measured on it would price a residency the run has not reached, so estimate() answers
     * nothing until the fill has settled and the policy runs its named baseline. Not
     * deactivation -- the profile is still the one this deployment will run on. */
    void hold_prices(bool hold) { hold_prices_ = hold; }
    bool prices_held() const { return hold_prices_; }

    bool active() const { return active_; }
    const std::string& reason() const { return reason_; }
    /* Stops pricing and learning, naming why, while keeping the loaded buckets visible: the run's
     * residency diverged from the fingerprint's start placement, so the profile is no longer a
     * measurement of this deployment. Not admit_fallback(): drift is not an admission to price
     * without measurements. */
    void deactivate(const std::string& why) { active_ = false; reason_ = why; }
    const std::vector<DraftCostBucket>& buckets() const { return buckets_; }
    /* Whether any bucket may price a shape at all -- the same rule estimate() applies
     * (draft_bucket_usable): no disturbing flag, n > 0, a finite positive mean. The scheduler's baseline
     * lock reads this: an unexamined buckets() list is not a measured baseline, since a profile of
     * nothing but diagnostic buckets passes it and prices nothing. */
    bool has_usable_bucket() const;
    int64_t dropped() const { return dropped_; }   /* buckets dropped at capacity */
    int64_t ignored() const { return ignored_; }   /* observations with no timing to learn from */
    /* A valid observation the table had no profile to learn into. Separate from ignored(): this
     * sample was not refused and carried no disturbing flag, it simply met an inactive table, so a
     * deployment that only observes (policy Cost with no admissible baseline) reports cycles here
     * and adds nothing to buckets(). */
    int64_t inactive() const { return inactive_; }

    /* Prices one shape. The key is the shape -- verify window, pipeline depth and the loaded
     * drafter included -- against the executed draft distribution: the query's
     * `draft_k`/`draft_rows[]` is the distribution it plans to execute, and only a bucket measured
     * at exactly that distribution prices it. A query that plans more than was ever measured gets
     * no measurement, however close the bucket's plan was; so does one with no anchor that ran its
     * own distribution with the same history and prefill. Measured from an exact bucket,
     * Interpolated from the neighbouring context class, or a heuristic fallback scaled from the
     * nearest anchor that ran the same distribution, history and prefill. A reference table
     * returns only the exact Measured bucket; every other query is invalid. valid = false when
     * there is no such anchor at all. Allocation-free. */
    int estimate(const DraftVerifyShape& shape, DraftCostEstimate* out) const;

    /* Learns from one completed cycle. Refuses an incomplete observation, one that does not
     * identify the plan it names, a sample whose identity disagrees with the observation, a plan
     * that says its bracket is not valid (DraftPlan::timing_valid), a shape that is not the plan's
     * own distribution or that is not priceable (an unknown dimension is refused, never priced as
     * zero), a replay state that contradicts the shape it would be filed under, an executed
     * distribution that could not have run or that exceeds the plan it took, and a non-finite or
     * non-positive time. A sample whose replay state the run could not establish is not an error:
     * it is a bracket with no evidence and is counted in ignored(). Only an attributable sample
     * updates a measured bucket; the rest are kept as diagnostics, and a truncation -- the plan's
     * deepest round never ran, so the bracket carries abort cost -- is stored with
     * kDraftBucketUnpriced instead of seeding a measured bucket. A reference table records
     * nothing: a valid observation is only counted (`reference_skipped()`), because the loaded
     * reference may not be overwritten by the run it prices. */
    int observe(const DraftObservation& observation, const DraftCostSample& sample);

    /* The P00 handle the pipeline wires into DraftHooks::costs. */
    DraftCostModel handle() { return DraftCostModel{this, &estimate_thunk}; }
    static int estimate_thunk(void* ctx, const DraftVerifyShape& shape, DraftCostEstimate* out);

private:
    std::vector<DraftCostBucket> buckets_;
    std::string reason_;
    int32_t capacity_ = 0;
    int64_t dropped_ = 0;
    int64_t ignored_ = 0;
    int64_t inactive_ = 0;
    int64_t reference_skipped_ = 0;
    bool active_ = false;
    bool fallback_admitted_ = false;
    bool record_only_ = false;
    bool hold_prices_ = false;
    bool reference_ = false;

    DraftCostBucket* find_mutable(const DraftVerifyShape& shape, uint16_t flags, int32_t rounds,
                                  const int32_t* round_rows);
    void record(const DraftCostSample& sample, uint16_t flags);
};

}  /* namespace rad */
