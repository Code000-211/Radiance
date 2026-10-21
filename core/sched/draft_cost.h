/* draft_cost.h -- strand A: the measured cost profile and the model that prices a verify shape.
 *
 * The policy (C) compares candidate plans by their expected batch time, so it asks the
 * DraftCostModel seam for a price in milliseconds for a DraftVerifyShape. This is that price: a
 * versioned profile of measured buckets (`rad-draft-profile/1`, docs/ADAPTIVE-MTP.md), keyed by
 * the whole shape -- verify tokens and N_j, context class, history and prefill tokens, sampling
 * and replay state -- never by (rows, max K) alone.
 *
 * `shape` is the verify window of the associated cycle, not its draft window: verify_k is the
 * deepest position the paired step verified and rows[j] is the N_j of that verify step. Under
 * issue_ahead the cycle drafts a different window than it verifies, so the executed draft extent
 * (rounds, draft_rows) is a second part of the key: the same verify shape at another pipeline
 * depth is a different bucket, never averaged in.
 *
 * Two axes this reader cannot key yet are interface requests against the shared header, not
 * approximations made here: (1) the planned draft extent in the lookup key -- DraftVerifyShape
 * carries the verify window only, so estimate() derives the query's extent from the shape itself
 * (rounds = verify_k, draft_rows = verify_tokens - n_rows) until the P00 shape carries the plan's
 * planned distribution; (2) the drafter-loaded/k_ceiling state, which the fingerprint does not
 * carry. Until it does, the conservative fallback refuses to borrow across a change of history
 * presence as a stopgap -- that is a bound, not a measurement of the other state.
 *
 * The reader runs off the step path: read()/configure() happen at startup. estimate() and
 * observe() are allocation-free. The bucket vector is reserved at configure; an observation that
 * would need a bucket past that capacity is counted in dropped() instead of growing the vector,
 * so telemetry overflow never blocks the card.
 *
 * Units are milliseconds or nothing. A shape with no bucket is priced by one documented
 * interpolation step (the neighbouring context class) or by a conservative bound scaled from the
 * nearest measured anchor; with no anchor at all the estimate is invalid and the caller uses its
 * named fixed baseline. A unitless prior is never reported as milliseconds.
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
 * DraftFallback::NoProfile) rather than rescaling a file measured on other hardware. */
struct DraftCostFingerprint {
    std::string model;
    std::string quant;
    std::string tp_wire;
    std::string gpus;
    std::string kernels;
    std::string kv_format;
    std::string placement;
    int32_t     tp = 0;
};

/* Compares two fingerprints; `first` names the first differing field, for the log and the plan. */
bool draft_cost_fingerprint_same(const DraftCostFingerprint& a, const DraftCostFingerprint& b,
                                 std::string* first);

/* Rectangular and shrinking (E2) executions run different kernels, so they are different
 * profiles. A profile for the other execution is refused, not rescaled. */
enum class DraftCostExecution : uint8_t { Rectangular = 0, Shrinking = 1 };

/* What the running deployment is: the profile is accepted only against this. */
struct DraftCostRuntime {
    DraftCostFingerprint fingerprint;
    DraftCostExecution   execution = DraftCostExecution::Rectangular;
};

/* One measured bucket. `shape` plus the executed extent `rounds`/`draft_rows` is the lookup key:
 * two cycles that verified the same window but ran a different draft extent are different
 * pipeline depths and stay separate. `flags` are the attribution flags of the samples it was
 * built from. A bucket carrying a disturbing flag is a diagnostic record: it is never used to
 * price a shape. */
struct DraftCostBucket {
    DraftVerifyShape shape;
    uint16_t flags = 0;
    int32_t  rounds = -1;       /* key: deepest draft round executed, -1 unknown (matches no query) */
    int32_t  draft_rows = -1;   /* key: sum_j N_j actually executed, -1 unknown (matches no query) */
    double   mean_ms = 0.0;     /* the amortised batch time at the same pipeline depth */
    double   unc_ms = 0.0;      /* half-width of mean_ms */
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

/* Parses and validates one profile. RAD_OK, or a negative RAD_E_* (RAD_E_UNSUPPORTED for a
 * format id this reader does not know, RAD_E_INVAL for a malformed or inconsistent file) with
 * `err` naming the problem. */
int draft_cost_profile_parse(const std::string& text, DraftCostProfile* out, std::string* err);
/* Reads `path` and parses it. The file is not on the step path. */
int draft_cost_profile_read(const std::string& path, DraftCostProfile* out, std::string* err);

/* How many buckets a table reserves by default: 2048 covers the wave-1 matrix (4 context classes
 * x 7 depths x the documented N_j windows x sampling x replay) with headroom. */
enum { kDraftCostDefaultBuckets = 2048 };

/* The per-context-class interpolation allowance: a bucket borrowed from the neighbouring class
 * is widened by this fraction of its mean, propagated into the estimate's uncertainty. */
constexpr double kDraftCostCtxStep = 0.10;
/* The conservative bound's margin over the anchor's verify-token scaling. */
constexpr double kDraftCostFallbackMargin = 0.25;

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
     * malformed profile (no fingerprint field, a bucket with an invalid shape). */
    int configure(const DraftCostProfile& profile, const DraftCostRuntime& runtime,
                  int32_t capacity = kDraftCostDefaultBuckets);

    /* No profile: every estimate is invalid. */
    void clear();

    bool active() const { return active_; }
    const std::string& reason() const { return reason_; }
    const std::vector<DraftCostBucket>& buckets() const { return buckets_; }
    int64_t dropped() const { return dropped_; }   /* buckets dropped at capacity */
    int64_t ignored() const { return ignored_; }   /* observations with no timing to learn from */

    /* Prices one shape. The query's draft extent follows from the shape (rounds = verify_k,
     * draft_rows = verify_tokens - n_rows); a bucket measured at another extent is a different
     * pipeline depth and does not price it. Measured from an exact bucket, Interpolated from the
     * neighbouring context class, or a ConservativeFallback scaled from the nearest anchor.
     * valid = false when there is no anchor at all. Allocation-free. */
    int estimate(const DraftVerifyShape& shape, DraftCostEstimate* out) const;

    /* Learns from one completed cycle. Refuses an incomplete observation, an unattributed one
     * (cycle_id or policy_version 0), a sample whose identity disagrees with the observation or
     * with the plan it names, and a non-finite or non-positive time. The sample's verify window is
     * checked against that plan (plan->verify_k, plan->n_seq), so a shape filled from the current
     * plan under issue_ahead -- verify_k != max_k -- is refused rather than learned. Only an
     * attributable sample updates a measured bucket; the rest are kept as diagnostics. */
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
    bool active_ = false;

    DraftCostBucket* find_mutable(const DraftVerifyShape& shape, uint16_t flags, int32_t rounds,
                                  int32_t draft_rows);
    void record(const DraftCostSample& sample, uint16_t flags);
};

}  /* namespace rad */
