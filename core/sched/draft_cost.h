/* draft_cost.h -- strand A: the measured cost profile and the model that prices a verify shape.
 *
 * The policy (C) compares candidate plans by their expected batch time, so it asks the
 * DraftCostModel seam for a price in milliseconds for a DraftVerifyShape. This is that price: a
 * versioned profile of measured buckets (`rad-draft-profile/1`, docs/ADAPTIVE-MTP.md), keyed by
 * the whole shape -- verify tokens and N_j, context class, history and prefill tokens, sampling
 * and replay state -- never by (rows, max K) alone.
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
 * Only a sample draft_sample_attributable() accepts updates a measured bucket. Recording, audit,
 * warmup, admission and placement brackets are kept as diagnostic records with their flags and
 * are never returned by estimate(). An ahead sample may update (the caller checked its bracket)
 * and stays flagged through n_ahead so the model can weigh it.
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

/* One measured bucket. `shape` is the lookup key; `flags` are the attribution flags of the
 * samples it was built from. A bucket carrying a disturbing flag is a diagnostic record: it is
 * never used to price a shape. */
struct DraftCostBucket {
    DraftVerifyShape shape;
    uint16_t flags = 0;
    int32_t  rounds = -1;       /* deepest draft round executed, -1 unknown */
    int32_t  draft_rows = -1;   /* sum_j N_j actually executed, -1 unknown */
    double   mean_ms = 0.0;     /* the amortised batch time at the same pipeline depth */
    double   unc_ms = 0.0;      /* half-width of mean_ms */
    int32_t  n = 0;             /* samples behind the mean */
    int32_t  n_ahead = 0;       /* of them issued ahead (kDraftSampleAhead) */
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

    /* Prices one shape: Measured from an exact bucket, Interpolated from the neighbouring context
     * class, or a ConservativeFallback scaled from the nearest anchor. valid = false when there
     * is no anchor at all. Allocation-free. */
    int estimate(const DraftVerifyShape& shape, DraftCostEstimate* out) const;

    /* Learns from one completed cycle. Refuses an incomplete observation, a cycle or policy
     * version that disagrees with the sample, and a non-finite or non-positive time. Only an
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

    DraftCostBucket* find_mutable(const DraftVerifyShape& shape, uint16_t flags);
    void record(const DraftCostSample& sample, uint16_t flags);
};

}  /* namespace rad */
