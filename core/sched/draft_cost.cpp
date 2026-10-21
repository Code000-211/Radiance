/* draft_cost.cpp -- strand A: the profile reader and the cost table. See draft_cost.h.
 *
 * The reader validates every field it will later rely on and refuses a file it cannot interpret
 * instead of guessing: the format id, the full fingerprint, the execution, and each bucket's two
 * distributions, executed extent, origin mix and timing. Nothing here runs on the step path.
 *
 * The table's lookup is a linear scan over the reserved buckets. A key is the verify shape plus
 * the executed draft extent; the query's extent is the plan's own distribution, which the shape
 * carries. A shape key is compared cheap fields first (depths, counts, classes), then the two N_j
 * arrays; the scan is over a few hundred buckets, not thousands, and it runs once per priced
 * candidate plan, not per draft token.
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

/* Reads an N_j array of exactly `len` entries into dst[0..len-1], checking each against `cap`. */
bool get_rows(const json& j, const char* key, int32_t len, int32_t cap, int32_t* dst) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_array() || (int32_t)it->size() != len) return false;
    for (int32_t j2 = 0; j2 < len; ++j2) {
        const json& r = (*it)[(size_t)j2];
        if (!r.is_number_integer()) return false;
        const int64_t v = r.get<int64_t>();
        if (v < 0 || v > cap) return false;
        dst[j2] = (int32_t)v;
    }
    return true;
}

/* Cheap half of the shape key; the N_j arrays are compared only when this matches. */
bool shape_head_same(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    return a.n_rows == b.n_rows && a.verify_k == b.verify_k &&
           a.verify_tokens == b.verify_tokens && a.ctx_class == b.ctx_class &&
           a.history_tokens == b.history_tokens && a.prefill_tokens == b.prefill_tokens &&
           a.sampling == b.sampling && a.replay == b.replay && a.drafter == b.drafter &&
           a.pipeline_depth == b.pipeline_depth && a.draft_k == b.draft_k;
}

/* Both distributions are part of the key: the verify N_j and the draft N_j the plan asked for. */
bool shape_same(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    if (!shape_head_same(a, b)) return false;
    for (int j = 0; j <= a.verify_k; ++j)
        if (a.rows[j] != b.rows[j]) return false;
    for (int j = 0; j <= a.draft_k; ++j)
        if (a.draft_rows[j] != b.draft_rows[j]) return false;
    return true;
}

/* Everything the shape has except the context class: the one documented interpolation step. Both
 * distributions, the pipeline depth and the drafter must match -- the same depths with a
 * different length distribution are another kernel. */
bool shape_same_except_ctx(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    if (a.ctx_class == b.ctx_class) return false;
    DraftVerifyShape a2 = a, b2 = b;
    a2.ctx_class = b2.ctx_class = 0;
    return shape_same(a2, b2);
}

/* The executed extent a bucket was measured at. An unknown extent (-1) matches nothing. */
bool extent_same(const DraftCostBucket& b, int32_t rounds, const int32_t* round_rows) {
    if (b.rounds != rounds || b.rounds < 0) return false;
    for (int j = 0; j <= rounds; ++j)
        if (b.draft_round_rows[j] != round_rows[j]) return false;
    return true;
}

/* An executed distribution that can have run: N_j never rises with the round, the deepest round
 * drafted at least one row, nothing survives past it, and the total is exactly their sum. It says
 * nothing about the verify window -- under pipeline_depth > 0 the two belong to different cycles. */
bool executed_consistent(int32_t rounds, const int32_t* round_rows, int32_t total, int32_t cap) {
    if (rounds < 0 || rounds > RAD_SCHED_MAX_SPEC || total < 0) return false;
    int64_t sum = 0;
    int32_t prev = cap;
    for (int j = 1; j <= rounds; ++j) {
        if (round_rows[j] < 0 || round_rows[j] > prev) return false;
        prev = round_rows[j];
        sum += round_rows[j];
    }
    if (rounds > 0 && prev == 0) return false;
    if (rounds == 0 && round_rows[0] != 0) return false;
    for (int j = rounds + 1; j <= RAD_SCHED_MAX_SPEC; ++j)
        if (round_rows[j] != 0) return false;
    return (int64_t)total == sum;
}

/* What actually ran cannot exceed what the plan asked for, round by round. */
bool executed_within_plan(const DraftCostSample& s, const DraftPlan& p) {
    if (s.rounds > p.max_k || !p.rows) return false;
    for (int j = 1; j <= s.rounds; ++j)
        if (s.draft_round_rows[j] > p.rows[j]) return false;
    return true;
}

/* The origin mix the file records must fit the drafter the same bucket names: the head and the
 * lookup fallback cannot stand in for one another, and a row that produced nothing is None. */
bool origins_fit_drafter(const DraftOrigin* origins, int32_t n_rows, DraftDrafter drafter) {
    for (int32_t i = 0; i < n_rows; ++i) {
        switch (origins[i]) {
            case DraftOrigin::None:   break;
            case DraftOrigin::Head:   if (drafter != DraftDrafter::Serial) return false; break;
            case DraftOrigin::Lookup: if (drafter != DraftDrafter::Ngram) return false; break;
        }
    }
    return true;
}

/* At pipeline_depth 0 the plan's draft distribution and the verify window are one execution, so a
 * file that lets them disagree was not produced by the measurement. */
bool distributions_agree(const DraftVerifyShape& s) {
    if (s.pipeline_depth != 0) return true;
    if (s.draft_k != s.verify_k) return false;
    for (int j = 0; j <= s.verify_k; ++j)
        if (s.draft_rows[j] != s.rows[j]) return false;
    return true;
}

/* Whether a bucket may price normal operation at all: it was measured cleanly. */
bool bucket_usable(const DraftCostBucket& b) {
    return (b.flags & kDisturbing) == 0 && b.n > 0 && std::isfinite(b.mean_ms) && b.mean_ms > 0.0;
}

/* What separates two buckets of the same shape: the disturbing flags. The ahead bit does not --
 * an ahead sample belongs to the measured bucket, it only stays visible through n_ahead. */
uint16_t bucket_class(uint16_t flags) { return (uint16_t)(flags & kDisturbing); }

/* The conservative fallback may only borrow from an anchor of the same context class, sampling,
 * replay, loaded drafter and pipeline depth: a bound scaled across a different drafter process or
 * another overlap structure would be a bound on another measurement. */
bool anchor_class_matches(const DraftVerifyShape& a, const DraftVerifyShape& b) {
    return a.ctx_class == b.ctx_class && a.sampling == b.sampling && a.replay == b.replay &&
           a.drafter == b.drafter && a.pipeline_depth == b.pipeline_depth;
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
    if (!get_int(sh, "n_rows", &s->n_rows) || s->n_rows < 0 || s->n_rows > RAD_SCHED_MAX_SPEC)
        return refuse(err, RAD_E_INVAL, where + "shape.n_rows: missing or outside 0.." +
                                            std::to_string(RAD_SCHED_MAX_SPEC));
    if (!get_int(sh, "verify_tokens", &s->verify_tokens) || s->verify_tokens < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.verify_tokens: missing or negative");
    if (!get_int(sh, "history_tokens", &s->history_tokens) || s->history_tokens < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.history_tokens: missing or negative");
    if (!get_int(sh, "prefill_tokens", &s->prefill_tokens) || s->prefill_tokens < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.prefill_tokens: missing or negative");
    if (!get_int(sh, "pipeline_depth", &s->pipeline_depth) || s->pipeline_depth < 0)
        return refuse(err, RAD_E_INVAL, where + "shape.pipeline_depth: missing or negative");
    if (!get_int(sh, "draft_k", &s->draft_k) || s->draft_k < 0 || s->draft_k > RAD_SCHED_MAX_SPEC)
        return refuse(err, RAD_E_INVAL, where + "shape.draft_k: missing or outside 0.." +
                                            std::to_string(RAD_SCHED_MAX_SPEC));

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

    static const char* const kDrafter[] = { "none", "serial", "ngram" };
    int drafter = 0;
    if (!get_enum(sh, "drafter", kDrafter, 3, &drafter))
        return refuse(err, RAD_E_INVAL, where + "shape.drafter: missing or unknown");
    s->drafter = (DraftDrafter)drafter;

    s->ctx_class = ctx;
    if (!get_rows(sh, "rows", verify_k + 1, s->n_rows, s->rows))
        return refuse(err, RAD_E_INVAL, where + "shape.rows: missing, not exactly verify_k + 1 "
                                            "long, or outside 0..n_rows");
    if (!get_rows(sh, "draft_rows", s->draft_k + 1, s->n_rows, s->draft_rows))
        return refuse(err, RAD_E_INVAL, where + "shape.draft_rows: missing, not exactly draft_k + 1 "
                                            "long, or outside 0..n_rows");
    /* draft_shape_priceable covers the monotonicity, the token sum, the deepest-round rule and the
     * stale tails of both arrays. */
    if (!draft_shape_priceable(*s))
        return refuse(err, RAD_E_INVAL, where + "shape: not a priceable key (both distributions "
                                            "must be monotone and complete, verify_tokens must be "
                                            "n_rows + sum(rows))");
    if (!distributions_agree(*s))
        return refuse(err, RAD_E_INVAL, where + "shape.draft_rows: disagrees with shape.rows at "
                                            "pipeline_depth 0, where they are one execution");

    const auto orig = sh.find("origins");
    if (orig == sh.end() || !orig->is_array() || (int32_t)orig->size() != s->n_rows)
        return refuse(err, RAD_E_INVAL, where + "shape.origins: missing or not exactly n_rows long");
    static const char* const kOrigin[] = { "none", "head", "lookup" };
    DraftOrigin origins[RAD_SCHED_MAX_SPEC] = {};
    for (int32_t i = 0; i < s->n_rows; ++i) {
        const json& o = (*orig)[(size_t)i];
        if (!o.is_string()) return refuse(err, RAD_E_INVAL, where + "shape.origins: not a string");
        const std::string name = o.get<std::string>();
        int v = -1;
        for (int k = 0; k < 3; ++k)
            if (name == kOrigin[k]) { v = k; break; }
        if (v < 0) return refuse(err, RAD_E_INVAL, where + "shape.origins: unknown origin \"" + name + "\"");
        origins[i] = (DraftOrigin)v;
    }
    if (!origins_fit_drafter(origins, s->n_rows, s->drafter))
        return refuse(err, RAD_E_INVAL, where + "shape.origins: does not fit drafter \"" +
                                            kDrafter[drafter] + "\"");
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
    if (!get_int(*ex, "draft_rows_total", &out->draft_rows_total))
        return refuse(err, RAD_E_INVAL, where + "extent.draft_rows_total: missing");
    if (!get_rows(*ex, "draft_round_rows", out->rounds + 1, out->shape.n_rows,
                  out->draft_round_rows))
        return refuse(err, RAD_E_INVAL, where + "extent.draft_round_rows: missing, not exactly "
                                            "rounds + 1 long, or outside 0..n_rows");
    if (!executed_consistent(out->rounds, out->draft_round_rows, out->draft_rows_total,
                             out->shape.n_rows))
        return refuse(err, RAD_E_INVAL, where + "extent: the executed distribution cannot have "
                                            "run (monotone, deepest round non-empty, total is "
                                            "the sum of the rounds)");
    /* What ran cannot exceed what the plan asked for: not more rounds than planned, and not more
     * rows in any round. */
    if (out->rounds > out->shape.draft_k)
        return refuse(err, RAD_E_INVAL, where + "extent.rounds: exceeds the planned distribution");
    for (int j = 1; j <= out->rounds; ++j)
        if (out->draft_round_rows[j] > out->shape.draft_rows[j])
            return refuse(err, RAD_E_INVAL, where + "extent.draft_round_rows[" +
                                                std::to_string(j) + "]: exceeds the planned "
                                                "distribution");

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

    const std::string expected = "rad-draft-profile/" + std::to_string(RAD_DRAFT_PROFILE_VERSION);
    std::string format;
    if (!get_string(j, "format", &format))
        return refuse(err, RAD_E_INVAL, "format: missing");
    if (format != expected)
        return refuse(err, RAD_E_UNSUPPORTED,
                      "format: \"" + format + "\" is not " + expected +
                          " (version 1 has no draft distribution, pipeline depth or origin mix; "
                          "a profile is regenerated, not rescaled)");

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
            if (bucket_class(prev.flags) == bucket_class(b.flags) && shape_same(prev.shape, b.shape) &&
                extent_same(prev, b.rounds, b.draft_round_rows))
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
    if (!draft_shape_priceable(s)) return RAD_E_INVAL;
    if (!active_) return RAD_OK;   /* no profile: invalid estimate, the caller's named baseline */

    /* The extent this shape would be executed at is its own draft distribution: a bucket measured
     * with another distribution ran another kernel, whatever its round count and total.
     * draft_shape_priceable already guarantees that distribution is complete and monotone. */

    /* 1. The shape itself. */
    for (const DraftCostBucket& b : buckets_) {
        if (!bucket_usable(b) || !shape_same(b.shape, s) ||
            !extent_same(b, s.draft_k, s.draft_rows))
            continue;
        out->ms = b.mean_ms;
        out->unc_ms = b.unc_ms;
        out->coverage = b.n;
        out->source = DraftCostSource::Measured;
        out->valid = true;
        return RAD_OK;
    }

    /* 2. The one documented interpolation step: the neighbouring context class, same shape
     * otherwise -- both distributions, the pipeline depth and the drafter included. The result is
     * a real measurement; only its uncertainty is widened, and ties go to the larger class (more
     * context, higher cost). */
    const DraftCostBucket* near = nullptr;
    for (const DraftCostBucket& b : buckets_) {
        if (!bucket_usable(b) || !shape_same_except_ctx(b.shape, s)) continue;
    if (!extent_same(b, s.draft_k, s.draft_rows)) continue;
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
     * same class, preferring an anchor at the same pipeline depth. The ratio is never below 1, so
     * the bound never prices a shape under a measured one, and the margin is reported as
     * uncertainty rather than hidden. */
    const DraftCostBucket* anchor = nullptr;
    int32_t anchor_rank = INT32_MAX;
    for (const DraftCostBucket& b : buckets_) {
        if (!bucket_usable(b) || !anchor_class_matches(b.shape, s)) continue;
        if (b.shape.verify_tokens <= 0) continue;
        const int32_t rank = (extent_same(b, s.draft_k, s.draft_rows) ? 0 : 1000) +
                             std::abs(b.shape.verify_k - s.verify_k);
        if (rank < anchor_rank) {
            anchor = &b;
            anchor_rank = rank;
        }
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

DraftCostBucket* DraftCostTable::find_mutable(const DraftVerifyShape& shape, uint16_t flags,
                                              int32_t rounds, const int32_t* round_rows) {
    for (DraftCostBucket& b : buckets_)
        if (bucket_class(b.flags) == bucket_class(flags) && shape_same(b.shape, shape) &&
            extent_same(b, rounds, round_rows))
            return &b;
    if ((int32_t)buckets_.size() >= capacity_) { ++dropped_; return nullptr; }
    DraftCostBucket b;
    b.shape = shape;
    b.flags = flags;
    b.rounds = rounds;
    b.draft_rows_total = 0;
    for (int j = 0; j <= rounds; ++j) {
        b.draft_round_rows[j] = round_rows[j];
        b.draft_rows_total += round_rows[j];
    }
    buckets_.push_back(b);
    return &buckets_.back();
}

void DraftCostTable::record(const DraftCostSample& s, uint16_t flags) {
    DraftCostBucket* b = find_mutable(s.shape, flags, s.rounds, s.draft_round_rows);
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
    if (s.target_ms >= 0.0 && finite(s.target_ms)) b->target_ms = s.target_ms;
    if (s.draft_ms >= 0.0 && finite(s.draft_ms)) b->draft_ms = s.draft_ms;
    if (s.host_ms >= 0.0 && finite(s.host_ms)) b->host_ms = s.host_ms;
}

int DraftCostTable::observe(const DraftObservation& observation, const DraftCostSample& sample) {
    if (!observation.complete) return RAD_E_INVAL;

    /* Identity first: an unattributed sample (0 = unattributed, draft_types.h) cannot be traced to
     * a cycle or a policy, so it teaches nothing and must not be folded into a bucket. The
     * observation must identify the plan it names, and the sample must be that cycle's -- the
     * shared predicates, not a second policy here. */
    const DraftPlan* plan = observation.plan;
    if (!plan) return RAD_E_INVAL;
    if (!draft_plan_identifies(*plan, observation)) return RAD_E_INVAL;
    if (sample.cycle_id == 0 || sample.cycle_id != observation.cycle_id) return RAD_E_INVAL;
    if (sample.policy_version != observation.policy_version) return RAD_E_INVAL;

    /* The shape must be the plan's own: both distributions, the depth and the row count. Under
     * pipeline_depth > 0 the verify window belongs to the paired step, so a shape filled from the
     * current plan would be self-consistent and false. A shape carrying an unknown dimension is
     * not priced as zero, and one that could not be measured is not learned from. */
    if (!draft_shape_matches_plan(sample.shape, *plan)) return RAD_E_INVAL;
    if (!draft_shape_priceable(sample.shape)) return RAD_E_INVAL;
    if (!executed_consistent(sample.rounds, sample.draft_round_rows, sample.draft_rows_total,
                             sample.shape.n_rows))
        return RAD_E_INVAL;
    if (!executed_within_plan(sample, *plan)) return RAD_E_INVAL;

    if (!finite(sample.total_ms) || !(sample.total_ms > 0.0)) return RAD_E_INVAL;
    if (!active_) return RAD_OK;   /* no usable profile: nothing to learn into */

    if (draft_sample_attributable(observation, sample)) {
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
