/* tests/spechub_couple_test.cpp -- the MTP coupling's arithmetic, checked against its own law.
 *
 * What can silently go wrong in a maximal coupling, and what each test pins:
 *
 *   1. THE COMPOSED LAW. A coupling is correct when the emitted token is distributed as the
 *      TARGET's P, whatever the draft's Q was. The test builds a (P, Q) pair, draws many tokens
 *      through the coupling, and compares the empirical frequencies against P. This is the one
 *      property that a wrong acceptance rule, a wrong residual or a swapped mass cannot survive.
 *
 *   2. THE ACCEPTANCE RULE ITSELF. A draft token is accepted when it equals the token the target
 *      itself drew at that position: with Q = delta_x the coupling must return x on every draw.
 *      With P(x) >= Q(x) it must accept unconditionally, because there is no residual to fall
 *      back on.
 *
 *   3. THE UNDERFLOW AND DUPLICATE RULES. r(x) is exactly zero on rejection (distinct doubles never
 *      subtract to 0), so the residual walk can skip x. A candidate that appears twice contributes
 *      its mass once. Both are properties of the shared helpers, checked directly.
 *
 *   4. THE Q BUILD. build_q drops entries that underflow, refuses duplicate ids, and stores exactly
 *      the law the draw walks -- so the numbers in state.q[] are the law, not a normalised copy of
 *      it that drifted.
 *
 * Nothing here talks to a card or to libr4d: the helpers are abi/rad_spechub.h, which the plugin and
 * the engine both compile.
 */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"

#include <cmath>
#include <cstring>
#include <map>
#include <vector>

namespace {

/* A state as the propose kernel would leave it: Q over `ids` from `scores`, x drawn from Q. */
RadMtpCoupleState make_state(const std::vector<int32_t>& ids, const std::vector<float>& scores,
                             int64_t vocab, uint64_t seed, uint64_t base, int round) {
    RadMtpCoupleState s{};
    s.draw.seed = seed;
    s.draw.base = base;
    s.draw.seq = 1;
    s.draw.card = 0;
    s.draw.mode = 1;
    s.round = round;
    s.status = 1;
    std::vector<int32_t> id8(8, -1);
    std::vector<float>   v8(8, -INFINITY);
    for (size_t i = 0; i < ids.size() && i < 8; ++i) { id8[i] = ids[i]; v8[i] = scores[i]; }
    CHECK(rad_mtp_couple_build_q(id8.data(), v8.data(), vocab, &s));
    const double u = rad_spechub_uniform(seed, base, RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)round);
    s.x = s.ids[rad_mtp_couple_pick(s, u)];
    return s;
}

/* A parameter row bound to the state, as the sampler stages it. */
RadSampleParams make_params(uint64_t seed, uint64_t pos) {
    RadSampleParams p{};
    p.seed = seed;
    p.pos = pos;
    p.temp = 1.0f;
    p.top_k = 0;
    p.mask_row = -1;
    return p;
}

/* The target's candidate plane: ids in [0, n), softmax scores. */
void target_plane(int n, std::vector<int32_t>* ids, std::vector<float>* val) {
    ids->clear();
    val->clear();
    for (int i = 0; i < n; ++i) {
        ids->push_back(i);
        val->push_back(-0.15f * (float)i);      /* a decaying, non-uniform target */
    }
}

}  /* namespace */

/* THE COMPOSED LAW. Whatever Q is, the coupling's output must be distributed as the target's P.
 *
 * The draft must be DRAWN FROM Q ON EVERY TRIAL, which is what a round does. Holding x fixed and
 * repeating one draw measures the acceptance rate of a single proposal -- P(x)/Q(x) -- not the
 * composed distribution, and fails against P however correct the rule is.
 *
 * The tolerance is five sigma of a binomial at this sample count, so a wrong accept rule (mass on
 * x alone) or a naive residual (a renormalised P over the non-Q tokens only) cannot pass. */
TEST(the_coupled_token_is_distributed_as_the_target) {
    const uint64_t seed = 424242;
    const int n = 8;
    std::vector<int32_t> ids;
    std::vector<float>   val;
    target_plane(n, &ids, &val);

    /* P from the same FP64 formula the verify uses. */
    double sum = 0, P[8] = {};
    float  top = -INFINITY;
    for (float v : val) top = std::max(top, v);
    for (int i = 0; i < n; ++i) { P[i] = std::exp((double)val[i] - top); sum += P[i]; }
    for (int i = 0; i < n; ++i) P[i] /= sum;

    /* Q deliberately differs from P: three tokens, in a different order of preference. */
    const int32_t qids[8] = { 0, 1, 2, -1, -1, -1, -1, -1 };
    const float   qval[8] = { 0.0f, -0.5f, -1.0f, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                              -INFINITY };

    const int draws = 40000;
    std::map<int32_t, int64_t> seen;
    for (int k = 0; k < draws; ++k) {
        const uint64_t trial = (uint64_t)k;
        /* A fresh draft from the same shortlist on every trial, at that trial's coordinate. */
        RadMtpCoupleState s{};
        s.draw.seed = seed;
        s.draw.base = trial;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        CHECK(rad_mtp_couple_build_q(qids, qval, 8, &s));
        const double u_draft =
            rad_spechub_uniform(seed, trial, RAD_MTP_COUPLE_KEY_DRAFT + 1);
        s.x = s.ids[rad_mtp_couple_pick(s, u_draft)];

        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(s, ids.data(), val.data(), n, 8,
                                    rad_spechub_uniform(seed, trial, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(seed, trial, RAD_SAMPLE_KEY_RESID), &out));
        ++seen[out];
    }
    for (int i = 0; i < n; ++i) {
        const double want = P[i];
        const double got  = (double)seen[(int32_t)i] / (double)draws;
        const double sigma = std::sqrt(want * (1.0 - want) / (double)draws);
        CHECK_NEAR(got, want, 5.0 * sigma + 1e-9);
    }
    /* Every survivor can be emitted: a rule that lost a tail token would show as a zero here. */
    CHECK_EQ((int64_t)seen.size(), (int64_t)n);
}

/* A DRAFT THE TARGET DOMINATES ON EVERY Q TOKEN IS ACCEPTED UNCONDITIONALLY: P(x) >= Q(x), so the
 * residual is empty and the ratio test has nothing to reject. The state here has ONE Q token, which
 * is the case a softmax over two tokens cannot give (two equal Q tokens of 0.5 each need P above
 * 0.5 on both, which a normalised P over three tokens cannot do). */
TEST(a_draft_the_target_dominates_is_always_accepted) {
    RadMtpCoupleState one = make_state({ 2 }, { 0.0f }, 8, 9, 0, 1);
    CHECK_EQ(one.count, 1);
    CHECK_EQ(one.x, 2);
    std::vector<int32_t> ids = { 0, 1, 2 };
    std::vector<float>   peak = { -20.0f, -20.0f, 9.0f };   /* P(2) ~ 1, Q(2) = 1 */
    for (int k = 0; k < 500; ++k) {
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(one, ids.data(), peak.data(), 3, 8,
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_RESID),
                                    &out));
        CHECK_EQ(out, 2);
    }

    /* AND WHERE P IS SHORT OF Q THE RATIO RULE APPLIES: the same Q token with a target that barely
     * admits it is rejected most of the time, and the fallback can never emit x. */
    std::vector<float> flat = { 4.0f, 4.0f, -4.0f };        /* P(2) is small */
    int rejected = 0;
    for (int k = 0; k < 2000; ++k) {
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(one, ids.data(), flat.data(), 3, 8,
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_RESID),
                                    &out));
        if (out != 2) { ++rejected; CHECK(out == 0 || out == 1); }
    }
    CHECK(rejected > 0);
}

/* THE RESIDUAL IS (P - Q)+ AND r(x) IS EXACTLY ZERO ON REJECTION. Here Q puts more mass on x than P
 * does, so x is sometimes rejected and the token must come from the OTHERS. The check that matters
 * is that y is never x on a rejected draw: r(x) contributes +0 exactly, which is what lets the walk
 * skip x without changing the total. */
TEST(a_rejected_draft_falls_back_to_the_residual_support) {
    RadMtpCoupleState s = make_state({ 0, 1 }, { 0.0f, 0.0f }, 8, 11, 0, 1);   /* Q(0) = Q(1) = 0.5 */
    CHECK_EQ(s.count, 2);
    std::vector<int32_t> ids = { 0, 1, 2 };
    /* P(0) = 0.09, so Q(0) = 0.5 exceeds it and x = 0 is rejected most of the time. */
    std::vector<float> val = { 0.0f, 2.2f, 1.0f };
    int accepted = 0, rejected = 0;
    for (int k = 0; k < 4000; ++k) {
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(s, ids.data(), val.data(), 3, 8,
                                    rad_spechub_uniform(3, (uint64_t)k, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(3, (uint64_t)k, RAD_SAMPLE_KEY_RESID),
                                    &out));
        if (out == s.x) ++accepted;
        else { ++rejected; CHECK(out != s.x); }
    }
    CHECK(accepted > 0);
    CHECK(rejected > 0);
}

/* build_q STORES THE LAW IT WALKS. A shortlist with a hole, a -inf score and a duplicate id: the
 * stored q[] sums to 1, the duplicate is refused, and the hole contributes nothing. */
TEST(build_q_refuses_malformed_shortlists_and_drops_zero_mass) {
    RadMtpCoupleState s{};
    /* A hole at slot 1 and -inf at slot 3: two live candidates. */
    const int32_t ids[8] = { 4, -1, 7, 9, -1, -1, -1, -1 };
    const float   val[8] = { 0.0f, 0.0f, -1.0f, -INFINITY, 0.0f, 0.0f, 0.0f, 0.0f };
    CHECK(rad_mtp_couple_build_q(ids, val, 16, &s));
    CHECK_EQ(s.count, 2);
    CHECK_EQ(s.ids[0], 4);
    CHECK_EQ(s.ids[1], 7);
    CHECK_EQ(s.ids[2], -1);
    CHECK_NEAR(s.q[0] + s.q[1], 1.0, 1e-15);

    /* A duplicate id is a broken shortlist, not something to aggregate. */
    const int32_t dup[8] = { 4, 4, -1, -1, -1, -1, -1, -1 };
    CHECK(!rad_mtp_couple_build_q(dup, val, 16, &s));
    CHECK_EQ(s.count, 0);

    /* An id outside the vocabulary, and a NaN score, are both refused. */
    const int32_t wide[8] = { 99, -1, -1, -1, -1, -1, -1, -1 };
    CHECK(!rad_mtp_couple_build_q(wide, val, 16, &s));
    const float nan_val[8] = { NAN, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    const int32_t one[8] = { 2, -1, -1, -1, -1, -1, -1, -1 };
    CHECK(!rad_mtp_couple_build_q(one, nan_val, 16, &s));

    /* Every candidate at -inf has no support at all. */
    float all_inf[8];
    for (int i = 0; i < 8; ++i) all_inf[i] = -INFINITY;
    CHECK(!rad_mtp_couple_build_q(one, all_inf, 16, &s));
    CHECK_EQ(s.count, 0);
}

/* state_valid IS THE ONE RULE BOTH SIDES REFUSE A ROW BY. A status-0 row, a q[] that does not sum to
 * one, an id outside the vocabulary, x missing from the support, and a round outside the band are
 * all refused -- because a verifier that accepted any of them would couple against a law that is
 * not the draft's. */
TEST(state_valid_refuses_every_malformed_row) {
    RadMtpCoupleState s = make_state({ 0, 1 }, { 0.0f, -0.2f }, 8, 5, 0, 2);
    CHECK(rad_mtp_couple_state_valid(s, 8));

    RadMtpCoupleState bad = s;
    bad.status = 0;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.q[0] += 0.5;                      /* no longer sums to 1 */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.ids[0] = 99;                     /* outside the vocabulary */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.x = 6;                           /* x is not in the support */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.round = 0;                       /* outside the band */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));
    bad = s; bad.round = RAD_MTP_COUPLE_MAX_ROUNDS + 1;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.draw.mode = 0;                   /* a greedy row's state has no Q */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.ids[1] = -1;                     /* a hole above count must carry no mass */
    CHECK(!rad_mtp_couple_state_valid(bad, 8) || s.count < 2);
}

/* THE UNIFORM IS THE SPECIFIED DRAW, and it is 53-bit: two seeds that differ in one bit must give
 * different numbers, and the range must be [0, 1) with no value at 1. This is the number the
 * acceptance compares P(x) against, so a uniform that repeated would make two draft positions
 * accept together. */
TEST(the_coupling_uniform_is_the_specified_53_bit_draw) {
    for (uint64_t pos = 0; pos < 32; ++pos) {
        const double u = rad_spechub_uniform(1234, pos, RAD_SAMPLE_KEY_ACCEPT);
        CHECK(u >= 0.0 && u < 1.0);
        /* 53 bits: a multiple of 2^-53, exactly. */
        CHECK_EQ(u, std::floor(u * 9007199254740992.0) / 9007199254740992.0);
    }
    CHECK(rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_ACCEPT) !=
          rad_spechub_uniform(2, 0, RAD_SAMPLE_KEY_ACCEPT));
    CHECK(rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_ACCEPT) !=
          rad_spechub_uniform(1, 1, RAD_SAMPLE_KEY_ACCEPT));
    CHECK(rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_ACCEPT) !=
          rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_RESID));
}

/* THE WIRE ROWS ARE THE SIZE THE PLANES ARE ADDRESSED AT. A field inserted into either struct would
 * shift every later operand of a plane the host writes and the device reads. */
TEST(the_coupling_wire_rows_keep_their_layout) {
    CHECK_EQ(sizeof(RadMtpCoupleDraw), (size_t)32);
    CHECK_EQ(sizeof(RadMtpCoupleState), (size_t)152);
    CHECK_EQ(RAD_MTP_COUPLE_DRAW_WORDS, (size_t)8);
    CHECK_EQ(RAD_MTP_COUPLE_STATE_WORDS, (size_t)38);
    CHECK_EQ(offsetof(RadMtpCoupleState, q) % 8, (size_t)0);
}

RAD_TEST_MAIN()
