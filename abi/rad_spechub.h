/* rad_spechub.h -- the linear-MTP coupling wire row.
 *
 * These rows are what a kernel plugin reads and the core writes, so their layout is part of the ABI
 * and belongs here rather than in a plugin header: the core may include an abi/ header and a plugin
 * may not include a core one. Everything in this file is header-only and total -- nothing
 * allocates, holds state or calls back into the core -- so the host reference, the GPU kernel and
 * the sampler read one definition of the wire AND of the arithmetic that turns it into a token.
 *
 * The row is a U32 plane with an eight-byte aligned base and row strides, which is what makes the
 * state wire safe to load whole on the device.
 */
#ifndef RAD_SPECHUB_H
#define RAD_SPECHUB_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* The verify reads a parameter row, so this header carries the row's definition: a plugin that had
 * to include two abi/ headers, in the right order, to read one row would get it wrong eventually. */
#include "rad_sample.h"

/* Linear MTP coupling (RADIANCE_MTP_COUPLED=1). One draw per drafting row, staged before the
 * serial passes; one state per (temporal round, row). `base` is the request's PRE-commit output
 * size, so (base, round) names one draft stream that no later step reuses. `card` is the step slot
 * the advance writes, checked against RadBatch::seq_ids. `mode` 1 samples x~Q; 0 keeps the greedy
 * proposal (temperature 0 or top_k 1). */
typedef struct RadMtpCoupleDraw {
    uint64_t seed, base, seq;
    int32_t  card;
    uint32_t mode;
} RadMtpCoupleDraw;
/* status 1: x sampled from the exact q[] stored here; 0: greedy proposal kept; -1: malformed
 * shortlist or row identity, greedy proposal kept. `pos` is the verify row's output position, bound
 * by the host after commit; the device writes 0. ids[count..7] = -1, q[count..7] = 0. */
typedef struct RadMtpCoupleState {
    RadMtpCoupleDraw draw;
    uint64_t pos;
    int32_t  status, round, x, count;
    int32_t  ids[8];
    double   q[8];
} RadMtpCoupleState;
#define RAD_MTP_COUPLE_DRAW_WORDS  (sizeof(RadMtpCoupleDraw) / 4)
#define RAD_MTP_COUPLE_STATE_WORDS (sizeof(RadMtpCoupleState) / 4)
#define RAD_MTP_COUPLE_MAX_ROUNDS 8
/* Draft stream of round r: seed ^ (KEY_DRAFT + r) at coordinate `base`. Acceptance and the
 * residual draw use RAD_SAMPLE_KEY_ACCEPT / RAD_SAMPLE_KEY_RESID at the verify position. */
#define RAD_MTP_COUPLE_KEY_DRAFT 0x4d5450434f55504cull
#ifdef __cplusplus
static_assert(sizeof(RadMtpCoupleDraw) == 32, "MTP coupling draw wire");
static_assert(sizeof(RadMtpCoupleState) == 152, "MTP coupling state wire");
static_assert(offsetof(RadMtpCoupleState, q) % 8 == 0, "MTP coupling q alignment");
#endif

/* ================================================================== the shared arithmetic
 *
 * NUMERICS ARE THE CONTRACT. Q is the FP64 softmax of an FP32 shortlist at temperature 1; P is the
 * FP64 softmax of the sampler's FP32 candidate logits. Both are same-or-higher precision than the
 * ordinary FP32/24-bit sample_pick, so a coupled row realizes the ideal filtered target law to FP64
 * rounding, NOT the FP32 pick's finite law, and is NOT seed-identical to it. Uniforms are 53-bit.
 */
#ifdef __cplusplus

#if defined(__HIPCC__)
#define RAD_SPECHUB_HD __host__ __device__
#else
#define RAD_SPECHUB_HD
#endif

static inline RAD_SPECHUB_HD double rad_spechub_positive(double a) { return a > 0 ? a : 0; }
static inline RAD_SPECHUB_HD bool rad_spechub_finite(double x) {
    return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308;
}

/* THE 53-BIT UNIFORM IN [0, 1): splitmix64 over (seed ^ key, pos), then the top 53 bits scaled by
 * 2^-53. A seed and a position name one number whoever computes it. */
static inline RAD_SPECHUB_HD double rad_spechub_uniform(uint64_t seed, uint64_t pos, uint64_t key) {
    uint64_t z = (seed ^ key) * 0x9e3779b97f4a7c15ull + pos + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1p-53;
}

/* Is `s` a state the coupling's propose kernel could have written for a sampled row? The verifier
 * and the engine's bind check refuse a row by this rule, on the device and on the host. */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_state_valid(const RadMtpCoupleState& s,
                                                             int64_t vocab) {
    if (s.status != 1 || s.draw.mode != 1 || s.count < 1 || s.count > 8 || s.round < 1 ||
        s.round > RAD_MTP_COUPLE_MAX_ROUNDS)
        return false;
    double sum = 0;
    bool has_x = false;
    for (int i = 0; i < 8; ++i) {
        if (i >= s.count) {
            if (s.ids[i] != -1 || s.q[i] != 0) return false;
            continue;
        }
        if (s.ids[i] < 0 || s.ids[i] >= vocab || !rad_spechub_finite(s.q[i]) || !(s.q[i] > 0) ||
            s.q[i] > 1)
            return false;
        for (int j = 0; j < i; ++j)
            if (s.ids[j] == s.ids[i]) return false;
        sum += s.q[i];
        has_x |= s.ids[i] == s.x;
    }
    return has_x && (sum - 1) <= 1e-12 && (1 - sum) <= 1e-12;
}

/* Build the stored Q from a shortlist of eight (id, score) slots.
 *
 * Holes (id < 0) and -inf scores carry no mass. An entry whose FP64 weight or normalized q
 * underflows to zero is DROPPED, so the stored q[] is exactly the law the draw walks. False on an
 * out-of-range id, NaN/+inf score, duplicate id or empty support (s->count = 0). */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_build_q(const int32_t* ids, const float* val,
                                                         int64_t vocab, RadMtpCoupleState* s) {
    int32_t kid[8];
    float   kv[8];
    int     n = 0;
    s->count = 0;
    for (int i = 0; i < 8; ++i) { s->ids[i] = -1; s->q[i] = 0; }
    for (int i = 0; i < 8; ++i) {
        if (ids[i] < 0) continue;
        if (ids[i] >= vocab || val[i] != val[i] || val[i] == INFINITY) return false;
        if (val[i] == -INFINITY) continue;
        for (int j = 0; j < n; ++j)
            if (kid[j] == ids[i]) return false;
        kid[n] = ids[i];
        kv[n++] = val[i];
    }
    if (!n) return false;
    float top = kv[0];
    for (int j = 1; j < n; ++j)
        if (kv[j] > top) top = kv[j];
    double w[8], sum = 0;
    for (int j = 0; j < n; ++j) { w[j] = ::exp((double)kv[j] - top); sum += w[j]; }
    for (int j = 0; j < n; ++j) {
        const double q = w[j] / sum;
        if (q > 0) { s->ids[s->count] = kid[j]; s->q[s->count++] = q; }
    }
    return s->count > 0;
}

/* Inverse CDF over the stored q with a uniform in [0, 1); index into ids. */
static inline RAD_SPECHUB_HD int rad_mtp_couple_pick(const RadMtpCoupleState& s, double u) {
    double total = 0;
    for (int i = 0; i < s.count; ++i) total += s.q[i];
    double c = u * total;
    for (int i = 0; i < s.count; ++i) { c -= s.q[i]; if (c < 0) return i; }
    return s.count - 1;
}

/* Residual mass r(y) = max(0, P(y) - Q(y)) of one candidate. Q tokens use their P aggregated over
 * duplicate candidates, counted once at the first occurrence (`seen`). */
static inline RAD_SPECHUB_HD double rad_mtp_couple_mass(const RadMtpCoupleState& s, const double* pq,
                                                        int32_t id, float v, float top, double sum,
                                                        unsigned* seen) {
    if (id < 0 || v == -INFINITY) return 0;
    for (int k = 0; k < s.count; ++k)
        if (s.ids[k] == id) {
            if (*seen & (1u << k)) return 0;
            *seen |= 1u << k;
            return rad_spechub_positive(pq[k] - s.q[k]);
        }
    return ::exp((double)v - top) / sum;
}

/* THE MAXIMAL COUPLING. Emits into *out and returns true, or returns false for an invalid/empty
 * target. Needs u_accept, u_resid in [0, 1).
 *
 * A DRAFT TOKEN IS ACCEPTED WHEN IT EQUALS THE TOKEN THE TARGET ITSELF DREW AT THAT POSITION: the
 * target's own law over the surviving candidates is P, the draft's is Q, x is the draft, and the
 * rule is `u * Q(x) < P(x)`, after the case where P dominates Q everywhere. On rejection the
 * emitted token is a draw from the residual (P - Q)+, which is what makes the composed law the
 * target's and nothing else.
 *
 * If FP64 cancellation leaves R = 0 while some Q token has P < Q, sample P directly in FP64 with
 * the independent residual uniform. The fallback decision reads (P, Q) only -- never x or either
 * uniform -- and must not restore sample_pick's FP32/24-bit law.
 *
 * R sums r over every candidate INCLUDING x. The residual walk runs only when P(x) < Q(x), where
 * r(x) = max(0, P(x) - Q(x)) is exactly +0 (distinct doubles never subtract to 0), so the walk --
 * which skips x explicitly, residual at x exactly zero -- has total R bit for bit. */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_decide(const RadMtpCoupleState& s,
                                                        const int32_t* ids, const float* val,
                                                        int64_t n, int64_t vocab, double u_accept,
                                                        double u_resid, int32_t* out) {
    float top = -INFINITY;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0) continue;
        if (ids[j] >= vocab || val[j] != val[j] || val[j] == INFINITY) return false;
        if (val[j] > top) top = val[j];
    }
    if (top == -INFINITY) return false;
    double sum = 0, pq[8] = {};
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        const double w = ::exp((double)val[j] - top);
        sum += w;
        for (int k = 0; k < s.count; ++k)
            if (ids[j] == s.ids[k]) { pq[k] += w; break; }
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    int  ix = -1;
    bool dominated = true;
    for (int k = 0; k < s.count; ++k) {
        pq[k] /= sum;
        dominated = dominated && pq[k] >= s.q[k];
        if (s.ids[k] == s.x) ix = k;
    }
    if (ix < 0) return false;   /* unreachable for a rad_mtp_couple_state_valid state */
    unsigned seen = 0;
    double   resid = 0;
    for (int64_t j = 0; j < n; ++j)
        resid += rad_mtp_couple_mass(s, pq, ids[j], val[j], top, sum, &seen);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double  c = u_resid * sum;
        int32_t last = -1;
        for (int64_t j = 0; j < n; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = ::exp((double)val[j] - top);
                if (w > 0) { last = ids[j]; c -= w; if (c < 0) { *out = last; return true; } }
            }
        *out = last;
        return true;
    }
    if (pq[ix] >= s.q[ix] || u_accept * s.q[ix] < pq[ix]) { *out = s.x; return true; }
    /* Rejected, so P(x) < Q(x): not dominated, resid > 0, and some y != x has mass > 0 below. */
    double  c = u_resid * resid;
    int32_t last = -1;
    seen = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] == s.x) continue;
        const double m = rad_mtp_couple_mass(s, pq, ids[j], val[j], top, sum, &seen);
        if (m > 0) { last = ids[j]; c -= m; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return true;
}

#undef RAD_SPECHUB_HD
#endif  /* __cplusplus */
#endif  /* RAD_SPECHUB_H */
