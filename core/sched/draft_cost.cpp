/* draft_cost.cpp -- strand A: the profile reader and the cost table. See draft_cost.h.
 *
 * The reader validates every field it will later rely on and refuses a file it cannot interpret
 * instead of guessing: the format id, the full fingerprint, the execution, and each bucket's
 * shape, extent and timing. Nothing here runs on the step path.
 *
 * The table's lookup is a linear scan over the reserved buckets. A shape key is compared cheap
 * fields first (depths, counts, classes), then N_j; the scan is over a few hundred buckets, not
 * thousands, and it runs once per priced candidate plan, not per draft token.
 */
#include "draft_cost.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace rad {
namespace {

using json = nlohmann::json;

/* The flags that make a bucket a diagnostic record rather than a measurement of normal
 * operation. kDraftSampleAhead is not here: an ahead sample is usable when the caller checked
 * its bracket, it only stays visible through n_ahead. */
constexpr uint16_t kDisturbing = kDraftSampleWarmup | kDraftSampleRecord | kDraftSampleAudit |
                                 kDraftSampleAdmission | kDraftSamplePlacement;

int refuse(std::string* err, int code, const std::string& what) {
    if (err) *err = what;
    return code;
}

bool get_string(const json& j, const char* key, std::string* out) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_string()) return false;
    *out = it->get<std::string>();
    return !out->empty();
}

bool get_int(const json& j, const char* key, int32_t* out) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_number_integer()) return false;
    const int64_t v = it->get<int64_t>();
    if (v < INT32_MIN || v > INT32_MAX) return false;
    *out = (int32_t)v;
    return true;
}

bool get_number(const json& j, const char* key, double* out) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_number()) return false;
    *out = it->get<double>();
    return true;
}

bool get_enum(const json& j, const char* key, const char* const* names, int n, int* out) {
    std::string s;
    if (!get_string(j, key, &s)) return false;
    for (int i = 0; i < n; ++i)
        if (s == names[i]) { *out = i; return true; }
    return false;
}

/* Cheap half of the shape key; the N_j comparison only runs when this matches. */
bool shape_head_same(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    return a.n_rows == b.n_rows && a.verify_k == b.verify_k &&
           a.verify_tokens == b.verify_tokens && a.ctx_class == b.ctx_class &&
           a.history_tokens == b.history_tokens && a.prefill_tokens == b.prefill_tokens &&
           a.sampling == b.sampling && a.replay == b.replay;
}

bool shape_same(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    if (!shape_head_same(a, b)) return false;
    for (int j = 0; j <= a.verify_k; ++j)
        if (a.rows[j] != b.rows[j]) return false;
    return true;
}

/* Everything the shape has except the context class: the one documented interpolation step. */
bool shape_same_except_ctx(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    if (a.ctx_class == b.ctx_class) return false;
    return a.n_rows == b.n_rows && a.verify_k == b.verify_k &&
           a.verify_tokens == b.verify_tokens && a.history_tokens == b.history_tokens &&
           a.prefill_tokens == b.prefill_tokens && a.sampling == b.sampling &&
           a.replay == b.replay;
}

/* Whether a bucket may price normal operation at all: it was measured cleanly. */
bool bucket_usable(const DraftCostBucket& b) {
    return (b.flags & kDisturbing) == 0 && b.n > 0 && std::isfinite(b.mean_ms) && b.mean_ms > 0.0;
}

/* What separates two buckets of the same shape: the disturbing flags. The ahead bit does not --
 * an ahead sample belongs to the measured bucket, it only stays visible through n_ahead. */
uint16_t bucket_class(uint16_t flags) { return (uint16_t)(flags & kDisturbing); }

/* The conservative fallback may only borrow from an anchor of the same context class, sampling
 * and replay, and the same history/prefill presence: K0 with the drafter loaded carries history,
 * K0 in a process without one does not, and their times are different measurements. */
bool anchor_class_matches(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    return a.ctx_class == b.ctx_class && a.sampling == b.sampling && a.replay == b.replay &&
           (a.history_tokens > 0) == (b.history_tokens > 0) &&
           (a.prefill_tokens > 0) == (b.prefill_tokens > 0);
}

bool finite(double x) { return std::isfinite(x); }

int parse_fingerprint(const json& j, DraftCostFingerprint* f, std::string* err) {
    if (!j.is_object()) return refuse(err, RAD_E_INVAL, "fingerprint: not an object");
    const char* keys[] = { "model", "quant", "tp_wire", "gpus", "kernels", "kv_format", "placement" };
    std::string* dst[] = { &f->model, &f->quant, &f->tp_wire, &f->gpus,
                           &f->kernels, &f->kv_format, &f->placement };
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i)
        if (!get_string(j, keys[i], dst[i]))
            return refuse(err, RAD_E_INVAL,
                          std::string("fingerprint.") + keys[i] + ": missing or not a non-empty string");
    if (!get_int(j, "tp", &f->tp) || f->tp < 1)
        return refuse(err, RAD_E_INVAL, "fingerprint.tp: missing or not a positive integer");
    return RAD_OK;
}

int parse_shape(const json& b, int32_t idx, DraftVerifyShape* s, std::string* err) {
    const std::string where = "buckets[" + std::to_string(idx) + "]: ";
    int32_t ctx = 0;
    if (!get_int(b, "ctx_class", &ctx) || ctx < 0 || ctx >= RAD_DRAFT_CTX_CLASSES)
        return refuse(err, RAD_E_INVAL, where + "ctx_class: missing or outside 0.." +
                                            std::to_string(RAD_DRAFT_CTX_CLASSES - 1));
    const auto sh_it = b.find("shape");
    if (sh_it == b.end() || !sh_it->is_object())
        return refuse(err, RAD_E_INVAL, where + "shape: missing or not an object");
    const json& sh = *sh_it;

    int32_t verify_k = 0;
    if (!get_int(sh, "verify_k", &verify_k) || verify_k < 0 || verify_k > RAD_SCHED_MAX_SPEC)
        return refuse(err, RAD_E_INVAL, where + "shape.verify_k: missing or outside 0.." +
                                            std::to_string(RAD_SCHED_MAX_SPEC));
    s->verify_k = verify_k;
    if (!get_int(sh, "n_rows", &s->n_rows) || s->n_rows < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.n_rows: missing or negative");
    if (!get_int(sh, "verify_tokens", &s->verify_tokens) || s->verify_tokens < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.verify_tokens: missing or negative");
    if (!get_int(sh, "history_tokens", &s->history_tokens) || s->history_tokens < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.history_tokens: missing or negative");
    if (!get_int(sh, "prefill_tokens", &s->prefill_tokens) || s->prefill_tokens < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.prefill_tokens: missing or negative");

    static const char* const kSampling[] = { "greedy", "sampled" };
    int sampling = 0;
    if (!get_enum(sh, "sampling", kSampling, 2, &sampling))
        return refuse(err, RAD_E_INVAL, where + "shape.sampling: missing or unknown");
    s->sampling = (DraftSampling)sampling;

    static const char* const kReplay[] = { "none", "play", "record", "audit" };
    int replay = 0;
    if (!get_enum(sh, "replay", kReplay, 4, &replay))
        return refuse(err, RAD_E_INVAL, where + "shape.replay: missing or unknown");
    s->replay = (DraftReplay)replay;

    const auto rows_it = sh.find("rows");
    if (rows_it == sh.end() || !rows_it->is_array() || (int32_t)rows_it->size() != verify_k + 1)
        return refuse(err, RAD_E_INVAL, where + "shape.rows: missing or not exactly verify_k + 1 long");
    s->ctx_class = ctx;
    for (int32_t j = 0; j <= verify_k; ++j) {
        const json& r = (*rows_it)[(size_t)j];
        if (!r.is_number_integer())
            return refuse(err, RAD_E_INVAL, where + "shape.rows[" + std::to_string(j) + "]: not an integer");
        const int64_t v = r.get<int64_t>();
        if (v < 0 || v > s->n_rows)
            return refuse(err, RAD_E_INVAL, where + "shape.rows[" + std::to_string(j) +
                                                "]: outside 0..n_rows");
        s->rows[j] = (int32_t)v;
    }
    if (s->rows[0] != 0)
        return refuse(err, RAD_E_INVAL, where + "shape.rows[0]: must be 0 (index 0 is unused)");
    for (int32_t j = 1; j < verify_k; ++j)
        if (s->rows[j] < s->rows[j + 1])
            return refuse(err, RAD_E_INVAL, where + "shape.rows: must not rise with the round");
    /* verify_tokens is sum_i (1 + k_i) and sum_i k_i is sum_j N_j, so the two counts must agree;
     * a file where they do not was not produced by the measurement and is refused. */
    int64_t implied = s->n_rows;
    for (int32_t j = 1; j <= verify_k; ++j) implied += s->rows[j];
    if ((int64_t)s->verify_tokens != implied)
        return refuse(err, RAD_E_INVAL, where + "shape.verify_tokens: is " +
                                            std::to_string(s->verify_tokens) + ", n_rows + sum(rows) is " +
                                            std::to_string(implied));
    return RAD_OK;
}

int parse_flags(const json& b, int32_t idx, uint16_t* flags, std::string* err) {
    *flags = 0;
    const auto it = b.find("flags");
    if (it == b.end()) return RAD_OK;
    if (!it->is_array()) return refuse(err, RAD_E_INVAL,
                                       "buckets[" + std::to_string(idx) + "]: flags: not an array");
    static const struct { const char* name; uint16_t bit; } kNames[] = {
        { "ahead",     kDraftSampleAhead },
        { "warmup",    kDraftSampleWarmup },
        { "record",    kDraftSampleRecord },
        { "audit",     kDraftSampleAudit },
        { "admission", kDraftSampleAdmission },
        { "placement", kDraftSamplePlacement },
    };
    for (const json& f : *it) {
        if (!f.is_string())
            return refuse(err, RAD_E_INVAL, "buckets[" + std::to_string(idx) + "]: flags: not a string");
        const std::string name = f.get<std::string>();
        bool known = false;
        for (const auto& k : kNames)
            if (name == k.name) { *flags |= k.bit; known = true; break; }
        if (!known)
            return refuse(err, RAD_E_INVAL,
                          "buckets[" + std::to_string(idx) + "]: unknown flag \"" + name + "\"");
    }
    return RAD_OK;
}

int parse_bucket(const json& b, int32_t idx, DraftCostBucket* out, std::string* err) {
    const std::string where = "buckets[" + std::to_string(idx) + "]: ";
    if (!b.is_object()) return refuse(err, RAD_E_INVAL, where + "not an object");

    int rc = parse_shape(b, idx, &out->shape, err);
    if (rc < 0) return rc;
    rc = parse_flags(b, idx, &out->flags, err);
    if (rc < 0) return rc;

    const auto ex = b.find("extent");
    if (ex == b.end() || !ex->is_object())
        return refuse(err, RAD_E_INVAL, where + "extent: missing or not an object");
    if (!get_int(*ex, "rounds", &out->rounds) || out->rounds < 0 || out->rounds > RAD_SCHED_MAX_SPEC)
        return refuse(err, RAD_E_INVAL, where + "extent.rounds: missing or outside 0.." +
                                            std::to_string(RAD_SCHED_MAX_SPEC));
    if (!get_int(*ex, "draft_rows", &out->draft_rows) || out->draft_rows < 0)
        return refuse(err, RAD_E_INVAL, where + "extent.draft_rows: missing or negative");

    const auto tm = b.find("time_ms");
    if (tm == b.end() || !tm->is_object())
        return refuse(err, RAD_E_INVAL, where + "time_ms: missing or not an object");
    if (!get_number(*tm, "mean", &out->mean_ms) || !finite(out->mean_ms) || !(out->mean_ms > 0.0))
        return refuse(err, RAD_E_INVAL, where + "time_ms.mean: missing, not finite, or not positive");
    if (!get_number(*tm, "unc", &out->unc_ms) || !finite(out->unc_ms) || out->unc_ms < 0.0)
        return refuse(err, RAD_E_INVAL, where + "time_ms.unc: missing, not finite, or negative");
    if (!get_int(*tm, "n", &out->n) || out->n < 1)
        return refuse(err, RAD_E_INVAL, where + "time_ms.n: missing or below 1");

    const auto cm = b.find("components_ms");
    if (cm != b.end()) {
        if (!cm->is_object()) return refuse(err, RAD_E_INVAL, where + "components_ms: not an object");
        bool valid = false;
        if (const auto v = cm->find("valid"); v != cm->end() && v->is_boolean()) valid = v->get<bool>();
        if (valid) {
            if (!get_number(*cm, "target", &out->target_ms) || !finite(out->target_ms) ||
                !get_number(*cm, "draft", &out->draft_ms) || !finite(out->draft_ms) ||
                !get_number(*cm, "host", &out->host_ms) || !finite(out->host_ms))
                return refuse(err, RAD_E_INVAL, where + "components_ms: valid is true but a component is missing");
        }
    }
    return RAD_OK;
}

}  /* namespace */

bool draft_cost_fingerprint_same(const DraftCostFingerprint& a, const DraftCostFingerprint& b,
                                 std::string* first) {
    auto str = [&](const char* name, const std::string& x, const std::string& y) {
        if (x == y) return true;
        if (first) *first = std::string(name) + " profile=\"" + x + "\" runtime=\"" + y + "\"";
        return false;
    };
    auto num = [&](const char* name, int32_t x, int32_t y) {
        if (x == y) return true;
        if (first) *first = std::string(name) + " profile=" + std::to_string(x) +
                            " runtime=" + std::to_string(y);
        return false;
    };
    return str("model", a.model, b.model) && str("quant", a.quant, b.quant) &&
           num("tp", a.tp, b.tp) && str("tp_wire", a.tp_wire, b.tp_wire) &&
           str("gpus", a.gpus, b.gpus) && str("kernels", a.kernels, b.kernels) &&
           str("kv_format", a.kv_format, b.kv_format) && str("placement", a.placement, b.placement);
}

int draft_cost_profile_parse(const std::string& text, DraftCostProfile* out, std::string* err) {
    if (!out) return RAD_E_INVAL;
    *out = DraftCostProfile{};

    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded()) return refuse(err, RAD_E_INVAL, "profile: not valid JSON");

    std::string format;
    if (!get_string(j, "format", &format))
        return refuse(err, RAD_E_INVAL, "format: missing");
    if (format != "rad-draft-profile/1")
        return refuse(err, RAD_E_UNSUPPORTED,
                      "format: \"" + format + "\" is not rad-draft-profile/1");

    const auto fp = j.find("fingerprint");
    if (fp == j.end()) return refuse(err, RAD_E_INVAL, "fingerprint: missing");
    int rc = parse_fingerprint(*fp, &out->fingerprint, err);
    if (rc < 0) return rc;

    static const char* const kExec[] = { "rectangular", "shrinking" };
    int exec = 0;
    if (!get_enum(j, "execution", kExec, 2, &exec))
        return refuse(err, RAD_E_INVAL, "execution: missing, unknown, or not a string");
    out->execution = (DraftCostExecution)exec;

    const auto buckets = j.find("buckets");
    if (buckets == j.end() || !buckets->is_array())
        return refuse(err, RAD_E_INVAL, "buckets: missing or not an array");
    if (buckets->size() > (size_t)kDraftCostDefaultBuckets)
        return refuse(err, RAD_E_INVAL, "buckets: more than " +
                                            std::to_string(kDraftCostDefaultBuckets));

    out->buckets.reserve(buckets->size());
    for (size_t i = 0; i < buckets->size(); ++i) {
        DraftCostBucket b;
        rc = parse_bucket((*buckets)[i], (int32_t)i, &b, err);
        if (rc < 0) return rc;
        for (const DraftCostBucket& prev : out->buckets)
            if (bucket_class(prev.flags) == bucket_class(b.flags) && shape_same(prev.shape, b.shape))
                return refuse(err, RAD_E_INVAL,
                              "buckets[" + std::to_string(i) + "]: duplicates an earlier bucket");
        out->buckets.push_back(b);
    }
    return RAD_OK;
}

int draft_cost_profile_read(const std::string& path, DraftCostProfile* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return refuse(err, RAD_E_INVAL, "profile: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    if (!f.good() && !f.eof()) return refuse(err, RAD_E_INVAL, "profile: cannot read " + path);
    return draft_cost_profile_parse(ss.str(), out, err);
}

/* ------------------------------------------------------------------ the table */

void DraftCostTable::clear() {
    buckets_.clear();
    buckets_.shrink_to_fit();
    reason_.clear();
    dropped_ = 0;
    ignored_ = 0;
    capacity_ = 0;
    active_ = false;
}

int DraftCostTable::configure(const DraftCostProfile& profile, const DraftCostRuntime& runtime,
                              int32_t capacity) {
    clear();
    if (capacity <= 0) return RAD_E_INVAL;

    std::string diff;
    if (!draft_cost_fingerprint_same(profile.fingerprint, runtime.fingerprint, &diff)) {
        reason_ = "profile fingerprint differs: " + diff;
        return RAD_OK;   /* a visible fallback, not a parse error */
    }
    if (profile.execution != runtime.execution) {
        reason_ = profile.execution == DraftCostExecution::Shrinking
                      ? "profile is for shrinking rounds, this runtime runs rectangular"
                      : "profile is for rectangular rounds, this runtime runs shrinking";
        return RAD_OK;
    }
    if ((int32_t)profile.buckets.size() > capacity)
        return refuse(&reason_, RAD_E_INVAL, "profile has more buckets than the table holds");

    buckets_ = profile.buckets;
    capacity_ = capacity;
    buckets_.reserve((size_t)capacity);
    active_ = true;
    return RAD_OK;
}

int DraftCostTable::estimate(const DraftVerifyShape& s, DraftCostEstimate* out) const {
    if (!out) return RAD_E_INVAL;
    *out = DraftCostEstimate{};
    if (s.verify_k < 0 || s.verify_k > RAD_SCHED_MAX_SPEC || s.n_rows < 0 ||
        s.ctx_class < 0 || s.ctx_class >= RAD_DRAFT_CTX_CLASSES || s.verify_tokens < 0)
        return RAD_E_INVAL;
    if (!active_) return RAD_OK;   /* no profile: invalid estimate, the caller's named baseline */

    /* 1. The shape itself. */
    for (const DraftCostBucket& b : buckets_) {
        if (!bucket_usable(b) || !shape_same(b.shape, s)) continue;
        out->ms = b.mean_ms;
        out->unc_ms = b.unc_ms;
        out->coverage = b.n;
        out->source = DraftCostSource::Measured;
        out->valid = true;
        return RAD_OK;
    }

    /* 2. The one documented interpolation step: the neighbouring context class, same shape
     * otherwise. The result is a real measurement; only its uncertainty is widened, and ties go
     * to the larger class (more context, higher cost). */
    const DraftCostBucket* near = nullptr;
    for (const DraftCostBucket& b : buckets_) {
        if (!bucket_usable(b) || !shape_same_except_ctx(b.shape, s)) continue;
        if (std::abs(b.shape.ctx_class - s.ctx_class) != 1) continue;
        if (!near || b.shape.ctx_class > near->shape.ctx_class) near = &b;
    }
    if (near) {
        const double step = kDraftCostCtxStep * near->mean_ms;
        out->ms = near->mean_ms;
        out->unc_ms = std::sqrt(near->unc_ms * near->unc_ms + step * step);
        out->coverage = near->n;
        out->source = DraftCostSource::Interpolated;
        out->valid = true;
        return RAD_OK;
    }

    /* 3. No bucket at all for this shape: a bound scaled from the nearest measured anchor of the
     * same class. The ratio is never below 1, so the bound never prices a shape under a measured
     * one, and the margin is reported as uncertainty rather than hidden. */
    const DraftCostBucket* anchor = nullptr;
    for (const DraftCostBucket& b : buckets_) {
        if (!bucket_usable(b) || !anchor_class_matches(b.shape, s)) continue;
        if (b.shape.verify_tokens <= 0) continue;
        if (!anchor) { anchor = &b; continue; }
        const int d_new  = std::abs(b.shape.verify_k - s.verify_k);
        const int d_old  = std::abs(anchor->shape.verify_k - s.verify_k);
        if (d_new < d_old || (d_new == d_old && b.shape.verify_k > anchor->shape.verify_k))
            anchor = &b;
    }
    if (anchor && s.verify_tokens > 0) {
        const double ratio = std::max(1.0, (double)s.verify_tokens / (double)anchor->shape.verify_tokens);
        const double scaled = anchor->mean_ms * ratio;
        const double margin = scaled * kDraftCostFallbackMargin;
        out->ms = scaled + margin;
        out->unc_ms = std::sqrt(anchor->unc_ms * anchor->unc_ms * ratio * ratio + margin * margin);
        out->coverage = 0;
        out->source = DraftCostSource::ConservativeFallback;
        out->valid = true;
    }
    return RAD_OK;
}

int DraftCostTable::estimate_thunk(void* ctx, const DraftVerifyShape& shape, DraftCostEstimate* out) {
    return static_cast<DraftCostTable*>(ctx)->estimate(shape, out);
}

DraftCostBucket* DraftCostTable::find_mutable(const DraftVerifyShape& shape, uint16_t flags) {
    for (DraftCostBucket& b : buckets_)
        if (bucket_class(b.flags) == bucket_class(flags) && shape_same(b.shape, shape)) return &b;
    if ((int32_t)buckets_.size() >= capacity_) { ++dropped_; return nullptr; }
    DraftCostBucket b;
    b.shape = shape;
    b.flags = flags;
    buckets_.push_back(b);
    return &buckets_.back();
}

void DraftCostTable::record(const DraftCostSample& s, uint16_t flags) {
    DraftCostBucket* b = find_mutable(s.shape, flags);
    if (!b) return;
    const double x = s.total_ms;
    if (b->n == 0) {
        b->mean_ms = x;
        b->unc_ms = 0.0;
    } else {
        const double d = x - b->mean_ms;
        b->mean_ms += d / (double)(b->n + 1);
        b->unc_ms = std::max(b->unc_ms, std::fabs(d));
    }
    ++b->n;
    if (flags & kDraftSampleAhead) ++b->n_ahead;
    b->rounds = s.rounds;
    b->draft_rows = s.draft_rows;
    if (s.target_ms >= 0.0 && finite(s.target_ms)) b->target_ms = s.target_ms;
    if (s.draft_ms >= 0.0 && finite(s.draft_ms)) b->draft_ms = s.draft_ms;
    if (s.host_ms >= 0.0 && finite(s.host_ms)) b->host_ms = s.host_ms;
}

int DraftCostTable::observe(const DraftObservation& observation, const DraftCostSample& sample) {
    if (!observation.complete) return RAD_E_INVAL;
    if (observation.cycle_id != 0 && sample.cycle_id != 0 &&
        observation.cycle_id != sample.cycle_id)
        return RAD_E_INVAL;
    if (observation.policy_version != 0 && sample.policy_version != 0 &&
        observation.policy_version != sample.policy_version)
        return RAD_E_INVAL;
    if (!finite(sample.total_ms) || !(sample.total_ms > 0.0)) return RAD_E_INVAL;
    if (sample.shape.verify_k < 0 || sample.shape.verify_k > RAD_SCHED_MAX_SPEC ||
        sample.shape.n_rows < 0 || sample.shape.ctx_class < 0 ||
        sample.shape.ctx_class >= RAD_DRAFT_CTX_CLASSES || sample.shape.verify_tokens < 0)
        return RAD_E_INVAL;
    if (!active_) return RAD_OK;   /* no usable profile: nothing to learn into */

    if (draft_sample_attributable(sample)) {
        record(sample, (uint16_t)(sample.flags & kDraftSampleAhead));
        return RAD_OK;
    }
    if (sample.flags & kDisturbing) {
        /* Stored apart with its flags: a recording, audit, warmup, admission or placement bracket
         * is not a price of normal operation, but it is not an outlier to be discarded either. */
        record(sample, (uint16_t)(sample.flags & (kDisturbing | kDraftSampleAhead)));
        return RAD_OK;
    }
    /* Not valid and not flagged: the bracket did not time this shape. Nothing to learn. */
    ++ignored_;
    return RAD_OK;
}

}  /* namespace rad */
