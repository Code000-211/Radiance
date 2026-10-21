/* tests/spechub_accept_test.cpp -- the coupling's ACCEPTANCE, measured on a real candidate stream.
 *
 * spechub_couple_test pins the law. This file measures the number the flag exists for: how many
 * draft tokens survive verification, compared against the same drafts WITHOUT the coupling's
 * acceptance rule.
 *
 * The comparator is honest about what it is. It does not run a card, a checkpoint or libr4d; it
 * runs the two decisions over the SAME candidate stream, the same drafts and the same uniforms:
 *
 *   ORDINARY   sample_pick's rule: the draft is a proposal the target checks for equality, so a
 *               draft survives only when it IS the target's own pick. That is the equality
 *               acceptance the plain MTP path applies (core/sample/accept.h).
 *
 *   COUPLED     rad_mtp_couple_decide: accept x with probability min(1, P(x)/Q(x)), and on refusal
 *               emit a draw from the residual (P-Q)+. The COMPOSED law is P (spechub_couple_test
 *               proves it), so the emitted token agrees with the target's own draw more often.
 *
 * Both are measured as "the emitted token equals the token the target itself drew" over the same
 * trials, which is the quantity acceptance length is built from. The report prints the two rates.
 *
 * What this is NOT: a serving measurement. The source's own ABI-16 benchmark is the only end-to-end
 * evidence and it is cited as such in the PR body.
 */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

/* Softmax over a shortlist in the verify's own FP64 formula, and the index it picks for a uniform. */
int32_t pick_from(const std::vector<float>& v, double u) {
    float top = -INFINITY;
    for (float x : v) top = std::max(top, x);
    double sum = 0, w[16] = {};
    for (size_t i = 0; i < v.size(); ++i) { w[i] = std::exp((double)v[i] - top); sum += w[i]; }
    double c = u * sum;
    for (size_t i = 0; i < v.size(); ++i) { c -= w[i]; if (c < 0) return (int32_t)i; }
    return (int32_t)v.size() - 1;
}

}  /* namespace */

/* THE ACCEPTANCE COMPARATOR. Over 20000 trials on one shortlist: the target's own draw t, a draft
 * x drawn from the head's Q, and the two decisions. Prints both rates and requires the coupled
 * rate to be at least the ordinary one -- never below, which is the whole claim. */
TEST(the_coupled_acceptance_beats_equality_on_the_same_stream) {
    /* The target's candidate plane: 8 live candidates, a plausible decode shape. */
    const std::vector<float> pval = { 3.10f, 2.95f, 2.40f, 1.80f, 1.10f, 0.40f, -0.60f, -2.00f };
    /* The head's own shortlist: the same 8, a flatter ranking, which is what a 2-bit head gives. */
    const std::vector<float> qval = { 2.60f, 2.55f, 2.30f, 2.00f, 1.60f, 1.20f, 0.60f, -0.40f };
    std::vector<int32_t> ids;
    for (int i = 0; i < 8; ++i) ids.push_back(i);

    const uint64_t seed = 90210;
    const int trials = 20000;
    int64_t ordinary_ok = 0, coupled_ok = 0, drafts = 0;

    for (int k = 0; k < trials; ++k) {
        const uint64_t pos = (uint64_t)k;
        /* The target's own draw at this position: what acceptance is measured against. */
        const int32_t target =
            pick_from(pval, rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_PICK));

        /* The head's draft, from its own Q, with the coupling's own draw key. */
        RadMtpCoupleState s{};
        s.draw.seed = seed;
        s.draw.base = pos;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        int32_t qidx[8];
        float   qv[8];
        for (int i = 0; i < 8; ++i) { qidx[i] = i; qv[i] = qval[(size_t)i]; }
        CHECK(rad_mtp_couple_build_q(qidx, qv, 8, &s));
        s.x = s.ids[rad_mtp_couple_pick(
            s, rad_spechub_uniform(seed, pos, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
        ++drafts;

        /* ORDINARY: equality, the plain MTP verifier's rule. */
        if (s.x == target) ++ordinary_ok;

        /* COUPLED: the maximal coupling, then the emitted token against the same target draw. */
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(s, ids.data(), pval.data(), 8, 8,
                                    rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID), &out));
        if (out == target) ++coupled_ok;
    }

    const double ordinary = (double)ordinary_ok / (double)drafts;
    const double coupled  = (double)coupled_ok / (double)drafts;
    std::fprintf(stderr,
                 "  acceptance over %d coupled draws: equality %.6f, coupling %.6f\n",
                 trials, ordinary, coupled);

    /* THE TWO RATES IN CLOSED FORM. The target's draw and the draft are independent, so:
     *   equality acceptance  = sum_i P_i Q_i   (the draft must equal the target's draw)
     *   the coupling         = sum_i P_i^2     (its output is distributed as P, and the target's own
     *                                          draw is an independent P draw)
     * Checking against these turns the measurement into an oracle rather than a threshold: a ported
     * rule that drifted would miss its own closed form, not a number somebody picked. */
    auto softmax = [](const std::vector<float>& v, double* out) {
        float top = -INFINITY;
        for (float x : v) top = std::max(top, x);
        double sum = 0;
        for (size_t i = 0; i < v.size(); ++i) { out[i] = std::exp((double)v[i] - top); sum += out[i]; }
        for (size_t i = 0; i < v.size(); ++i) out[i] /= sum;
    };
    const size_t n = pval.size();
    std::vector<double> P(n), Q(n);
    softmax(pval, P.data());
    softmax(qval, Q.data());
    double want_ordinary = 0.0, want_coupled = 0.0;
    for (size_t i = 0; i < n; ++i) {
        want_ordinary += P[i] * Q[i];
        want_coupled  += P[i] * P[i];
    }
    const double sig = std::sqrt(0.25 / (double)drafts);   /* a binomial's widest sigma */
    CHECK_NEAR(ordinary, want_ordinary, 5.0 * sig);
    CHECK_NEAR(coupled, want_coupled, 5.0 * sig);
    /* AND THE CLAIM, which follows from the two closed forms: sigma(P^2) >= sigma(P Q) because P is
     * the draft's own law. Strictly here, since P != Q on this stream. */
    CHECK(want_coupled > want_ordinary);
    CHECK(coupled > ordinary);
}

/* AND THE DRAFT ITSELF IS STILL SAMPLED FROM Q. The coupling can only pay if x~Q, so this pins that
 * the propose row's draw is the FP64 softmax of the shortlist and not the argmax: over many rounds
 * every candidate with mass appears, and the frequencies match Q. */
TEST(the_drafted_token_is_sampled_from_the_heads_own_law) {
    const std::vector<float> qval = { 1.0f, 0.4f, -0.3f, -1.2f };
    int32_t qidx[8];
    float   qv[8];
    for (int i = 0; i < 8; ++i) { qidx[i] = i < 4 ? i : -1; qv[i] = i < 4 ? qval[(size_t)i] : -INFINITY; }

    double sum = 0, Q[4] = {};
    for (int i = 0; i < 4; ++i) { Q[i] = std::exp((double)qval[(size_t)i] - 1.0); sum += Q[i]; }
    for (int i = 0; i < 4; ++i) Q[i] /= sum;

    const int trials = 20000;
    int64_t seen[4] = {};
    for (int k = 0; k < trials; ++k) {
        RadMtpCoupleState s{};
        s.draw.seed = 7;
        s.draw.base = (uint64_t)k;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        CHECK(rad_mtp_couple_build_q(qidx, qv, 8, &s));
        const int32_t x = s.ids[rad_mtp_couple_pick(
            s, rad_spechub_uniform(7, (uint64_t)k, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
        CHECK(x >= 0 && x < 4);
        ++seen[x];
    }
    for (int i = 0; i < 4; ++i) {
        const double got = (double)seen[i] / (double)trials;
        const double sigma = std::sqrt(Q[i] * (1.0 - Q[i]) / (double)trials);
        CHECK_NEAR(got, Q[i], 5.0 * sigma + 1e-9);
    }
}

RAD_TEST_MAIN()
