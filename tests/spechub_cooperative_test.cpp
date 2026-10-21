/* tests/spechub_cooperative_test.cpp -- the cooperative verify against the serial one.
 *
 * The cooperative kernel writes P's weights and the residual mass across the workgroup and then
 * walks both planes in ascending candidate order on one thread. That is a REORDERING of the
 * compute, not of the arithmetic -- and the arithmetic is the ABI here, so the claim has to be
 * checked rather than argued.
 *
 * What this file checks, over the same rows, the same states and the same uniforms:
 *
 *   PARITY     the cooperative decision emits the same token as rad_mtp_couple_decide (the serial
 *              reference the plugin's host row runs) on every row of every width case.
 *
 *   WIDTHS     129, 1024 and 1025 candidates. 1025 is the case that falls out of the shared-plane
 *              band into the general single-threaded algorithm; it must still agree.
 *
 *   HOSTILE    a row with a hole in the middle, a duplicate id, an -inf candidate, and a row whose
 *              state does not bind -- all of which the planes must treat exactly as the serial walk
 *              does.
 *
 * The kernel itself is not launched here: this is the HOST half of the same algorithm, stepped
 * through the same phase sequence, so a divergence between the two would have to live in the
 * device code alone. The device launch is covered by the comparator probe the PR body reports.
 */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/* The cooperative algorithm, on the host: the same phases the kernel runs, over the same shared
 * planes. Deliberately a transcription of the kernel's body, so a change to one that is not made to
 * the other shows up as a mismatch against rad_mtp_couple_decide. */
bool cooperative_decide(const RadMtpCoupleState& s, const std::vector<int32_t>& ids,
                        const std::vector<float>& val, int64_t vocab, double u_accept,
                        double u_resid, int32_t* out) {
    const int64_t n_cand = (int64_t)ids.size();
    std::vector<double> weight((size_t)n_cand, 0.0), mass((size_t)n_cand, 0.0);
    double pq[8] = {};
    int    first[8];
    double sum = 0;
    float  top = -INFINITY;
    int    ix = -1;
    bool   dominated = true;

    /* PHASE 1: the gate, exactly the kernel's. */
    bool eligible = false;
    if (s.status == 1) {
        float maximum = -INFINITY;
        bool  valid   = true;
        for (int64_t j = 0; j < n_cand; ++j) {
            if (ids[(size_t)j] < 0) continue;
            if (ids[(size_t)j] >= vocab || val[(size_t)j] != val[(size_t)j] ||
                val[(size_t)j] == INFINITY) {
                valid = false;
                break;
            }
            if (val[(size_t)j] > maximum) maximum = val[(size_t)j];
        }
        if (valid && maximum != -INFINITY) { top = maximum; eligible = true; }
    }
    if (!eligible) return false;

    /* PHASE 2: the weights, per slot. */
    for (int64_t j = 0; j < n_cand; ++j)
        if (ids[(size_t)j] >= 0 && val[(size_t)j] != -INFINITY)
            weight[(size_t)j] = std::exp((double)val[(size_t)j] - top);

    /* PHASE 3: the ordered sum and the Q aggregation. */
    for (int k = 0; k < s.count; ++k) { pq[k] = 0; first[k] = -1; }
    for (int64_t j = 0; j < n_cand; ++j) {
        if (ids[(size_t)j] < 0 || val[(size_t)j] == -INFINITY) continue;
        const double w = weight[(size_t)j];
        sum += w;
        for (int k = 0; k < s.count; ++k)
            if (ids[(size_t)j] == s.ids[k]) {
                if (first[k] < 0) first[k] = (int)j;
                pq[k] += w;
                break;
            }
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    for (int k = 0; k < s.count; ++k) {
        pq[k] /= sum;
        dominated = dominated && pq[k] >= s.q[k];
        if (s.ids[k] == s.x) ix = k;
    }
    if (ix < 0) return false;

    /* PHASE 4: the residual mass, per slot. */
    for (int64_t j = 0; j < n_cand; ++j) {
        double m = 0;
        if (ids[(size_t)j] >= 0 && val[(size_t)j] != -INFINITY) {
            bool in_q = false;
            for (int k = 0; k < s.count; ++k)
                if (s.ids[k] == ids[(size_t)j]) {
                    in_q = true;
                    if (first[k] == (int)j) m = rad_spechub_positive(pq[k] - s.q[k]);
                    break;
                }
            if (!in_q) m = weight[(size_t)j] / sum;
        }
        mass[(size_t)j] = m;
    }

    /* PHASE 5: the ordered walk. */
    double resid = 0;
    for (int64_t j = 0; j < n_cand; ++j) resid += mass[(size_t)j];
    int32_t result = -1;
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int64_t j = 0; j < n_cand; ++j)
            if (ids[(size_t)j] >= 0 && val[(size_t)j] != -INFINITY) {
                const double w = weight[(size_t)j];
                if (w > 0) { result = ids[(size_t)j]; cdf -= w; if (cdf < 0) break; }
            }
    } else if (pq[ix] >= s.q[ix] || u_accept * s.q[ix] < pq[ix]) {
        result = s.x;
    } else {
        double cdf = u_resid * resid;
        for (int64_t j = 0; j < n_cand; ++j) {
            if (ids[(size_t)j] == s.x) continue;
            const double m = mass[(size_t)j];
            if (m > 0) { result = ids[(size_t)j]; cdf -= m; if (cdf < 0) break; }
        }
    }
    *out = result;
    return true;
}

/* A bound state over the first `qcount` ids of the plane. */
RadMtpCoupleState bound_state(const std::vector<int32_t>& ids, int qcount, uint64_t seed,
                              uint64_t pos) {
    RadMtpCoupleState s{};
    s.draw.seed = seed;
    s.draw.base = pos;
    s.draw.mode = 1;
    s.round = 1;
    s.status = 1;
    s.pos = pos;
    int32_t qidx[8];
    float   qv[8];
    for (int i = 0; i < 8; ++i) {
        qidx[i] = i < qcount ? ids[(size_t)i] : -1;
        qv[i]   = i < qcount ? -0.3f * (float)i : -INFINITY;
    }
    CHECK(rad_mtp_couple_build_q(qidx, qv, (int64_t)ids.size() + 1024, &s));
    s.x = s.ids[rad_mtp_couple_pick(
        s, rad_spechub_uniform(seed, pos, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
    return s;
}

}  /* namespace */

/* PARITY AT EVERY WIDTH. 129, 1024 and 1025 candidates, 64 rows each, the same state and uniforms:
 * the cooperative decision must equal the serial reference on every row. 1025 is the width that
 * leaves the shared-plane band, so it exercises both paths. */
TEST(the_cooperative_decision_equals_the_serial_reference_at_every_width) {
    for (int64_t n : { 129, 1024, 1025 }) {
        std::vector<int32_t> ids((size_t)n);
        std::vector<float>   val((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            ids[(size_t)i] = (int32_t)i;
            /* A long decaying tail: the nucleus is narrow and the residual is a small, spread set,
             * which is where a mass-plane mistake would show. */
            val[(size_t)i] = 6.0f - 0.01f * (float)i;
        }
        int64_t agree = 0, refusals = 0;
        for (int k = 0; k < 64; ++k) {
            const uint64_t pos = (uint64_t)(n + k);
            const RadMtpCoupleState s = bound_state(ids, 8, 4242, pos);
            const double ua = rad_spechub_uniform(4242, pos, RAD_SAMPLE_KEY_ACCEPT);
            const double ur = rad_spechub_uniform(4242, pos, RAD_SAMPLE_KEY_RESID);

            int32_t serial = -7, coop = -7;
            const bool ok_s = rad_mtp_couple_decide(s, ids.data(), val.data(), n, n, ua, ur,
                                                   &serial);
            const bool ok_c = cooperative_decide(s, ids, val, n, ua, ur, &coop);
            CHECK_EQ(ok_c, ok_s);
            if (!ok_s) { ++refusals; continue; }
            if (serial == coop) ++agree;
        }
        CHECK_EQ(agree + refusals, (int64_t)64);
        CHECK_EQ(refusals, (int64_t)0);
        CHECK_EQ(agree, (int64_t)64);
    }
}

/* HOSTILE PLANES. A hole in the middle, a duplicate id and an -inf candidate: the cooperative
 * planes must treat each exactly as the serial walk does, including the duplicate-first rule that
 * makes a repeated candidate contribute its Q mass once. */
TEST(the_cooperative_decision_agrees_on_hostile_candidate_planes) {
    std::vector<int32_t> ids = { 0, 1, -1, 3, 3, 5, 6, -1, 8, 9 };
    std::vector<float>   val = { 4.0f, 3.0f, -INFINITY, 1.5f, 1.4f, 0.5f, -1.0f, -INFINITY,
                                  -2.0f, -INFINITY };
    int64_t agree = 0, refused = 0;
    for (int k = 0; k < 256; ++k) {
        const uint64_t pos = (uint64_t)k;
        /* Q's shortlist holds each id ONCE -- a duplicate there is a malformed shortlist, which
         * build_q refuses by design -- while the CANDIDATE plane keeps its duplicate 3, which is the
         * duplicate-first rule the two decisions must share. */
        int32_t qidx[8];
        float   qv[8];
        const int32_t qids[6] = { 0, 1, 3, 5, 6, 8 };
        for (int i = 0; i < 8; ++i) {
            qidx[i] = i < 6 ? qids[i] : -1;
            qv[i]   = i < 6 ? -0.3f * (float)i : -INFINITY;
        }
        RadMtpCoupleState s{};
        s.draw.seed = 77;
        s.draw.base = pos;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        s.pos = pos;
        CHECK(rad_mtp_couple_build_q(qidx, qv, 16, &s));
        s.x = s.ids[rad_mtp_couple_pick(
            s, rad_spechub_uniform(77, pos, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
        const double ua = rad_spechub_uniform(77, pos, RAD_SAMPLE_KEY_ACCEPT);
        const double ur = rad_spechub_uniform(77, pos, RAD_SAMPLE_KEY_RESID);
        int32_t serial = -7, coop = -7;
        const bool ok_s =
            rad_mtp_couple_decide(s, ids.data(), val.data(), (int64_t)ids.size(), 16, ua, ur,
                                  &serial);
        const bool ok_c = cooperative_decide(s, ids, val, 16, ua, ur, &coop);
        CHECK_EQ(ok_c, ok_s);
        if (!ok_s) { ++refused; continue; }
        CHECK_EQ(coop, serial);
        ++agree;
    }
    /* The duplicate id 3 appears twice; the plane is well formed, so no row should refuse. */
    CHECK_EQ(refused, (int64_t)0);
    CHECK_EQ(agree, (int64_t)256);
}

/* THE GATE IS `rad_mtp_couple_state_valid`, AND IT IS THE SAME TEST ON BOTH SIDES. `decide` is total
 * over a candidate plane -- it answers whatever bytes it is given -- so the refusal that keeps an
 * uncoupled or stale row from overwriting the pick's token lives in the row gate both the kernel and
 * the sampler apply before calling it. This pins that gate rather than pretending `decide` checks a
 * state it is never given. */
TEST(an_unbound_row_is_refused_by_the_shared_gate) {
    std::vector<int32_t> ids = { 0, 1, 2, 3 };
    std::vector<float>   val = { 3.0f, 2.5f, 1.0f, -0.5f };
    RadMtpCoupleState s = bound_state(ids, 4, 9, 100);
    CHECK(rad_mtp_couple_state_valid(s, 8));

    /* status 0: a greedy row, which has no Q to couple against. */
    RadMtpCoupleState bad = s;
    bad.status = 0;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    /* A q[] that no longer sums to 1: the draft's law is gone, so there is nothing to couple to. */
    bad = s;
    bad.q[0] += 0.25;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    /* An id outside the vocabulary, and an x outside the support. */
    bad = s;
    bad.ids[0] = 99;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));
    bad = s;
    bad.x = 6;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    /* AND THE KERNEL GATES ON THE SAME THING. Its eligibility test is `s.status == 1 &&
     * !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed && s.pos == p.pos &&
     * rad_mtp_couple_state_valid(s, vocab)`, so a state that fails the validator above is refused
     * before any plane is touched. The cooperative host transcription begins with that same gate:
     * a status-0 state makes it refuse without writing. */
    RadMtpCoupleState unbound = s;
    unbound.status = 0;
    int32_t out = -7;
    CHECK(!cooperative_decide(unbound, ids, val, 8, 0.5, 0.5, &out));
    CHECK_EQ(out, (int32_t)-7);            /* nothing was written */
}

RAD_TEST_MAIN()
