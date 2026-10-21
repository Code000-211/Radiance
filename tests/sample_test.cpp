/* sample_test.cpp -- the sampler component's tests.
 *
 * Each host reference sampler is checked against a hand-computed answer on a tiny vocabulary,
 * since a self-checked reference agrees with its own bugs. rad-kbench holds the device samplers to
 * these references (spec §17), so an error here propagates upward.
 */
#include "rad_test.h"

#include "sample/accept.h"
#include "sample/gbnf.h"
#include "gbnf_oracle.h"
#include "sample/grammar.h"
#include "sample/host_ref.h"
#include "sample/sampler.h"
#include "sample/vocab_view.h"
#include "text/tokenizer.h"

#include "build/rad_build.h"
#include "plugin/registry.h"
#include "runtime/ctx.h"
#include "rad_sample.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace rad;

/* ------------------------------------------------------------------ helpers */

static CandidateSet set_of(const std::vector<float>& logits) {
    CandidateSet s;
    s.from_logits(logits.data(), (int64_t)logits.size());
    return s;
}

/* Logits whose softmax is exactly the given probability vector, so a test states probabilities
 * directly. */
static std::vector<float> logits_for(const std::vector<double>& p) {
    std::vector<float> l;
    l.reserve(p.size());
    for (double x : p) l.push_back((float)std::log(x));
    return l;
}

static std::vector<int32_t> ids_of(const CandidateSet& s) {
    std::vector<int32_t> v;
    for (const auto& c : s.c) v.push_back(c.id);
    return v;
}

/* ================================================================== temperature */

TEST(temp_divides_logits) {
    auto s = set_of({ 1.0f, 2.0f, 3.0f });
    host_temp(s, 2.0f);
    CHECK_NEAR(s.c[0].logit, 0.5f, 1e-6);
    CHECK_NEAR(s.c[1].logit, 1.0f, 1e-6);
    CHECK_NEAR(s.c[2].logit, 1.5f, 1e-6);
}

TEST(temp_zero_collapses_to_argmax) {
    /* Temperature 0 must leave exactly the argmax finite and pick the same token host_argmax
     * would; otherwise a temperature-0 request would silently stop matching a greedy run. */
    auto s = set_of({ 1.0f, 3.0f, 3.0f, 0.0f });
    host_temp(s, 0.0f);
    CHECK(std::isinf(s.c[0].logit) && s.c[0].logit < 0);
    CHECK_EQ(s.c[1].logit, 3.0f);
    CHECK(std::isinf(s.c[2].logit) && s.c[2].logit < 0);
    CHECK(std::isinf(s.c[3].logit) && s.c[3].logit < 0);

    const float raw[] = { 1.0f, 3.0f, 3.0f, 0.0f };
    CHECK_EQ(host_argmax(raw, 4), 1);      /* the tie goes to the lowest id */
}

TEST(temp_zero_breaks_a_tie_by_id_in_any_set_order) {
    /* Typical sampling leaves the set in surprisal order, so position is not id order. */
    auto s = set_of({ 3.0f, 3.0f, 1.0f });
    std::swap(s.c[0], s.c[1]);
    s.sorted = false;
    host_temp(s, 0.0f);
    CHECK_EQ(s.c[0].id, 1);
    CHECK(std::isinf(s.c[0].logit) && s.c[0].logit < 0);
    CHECK_EQ(s.c[1].id, 0);
    CHECK_EQ(s.c[1].logit, 3.0f);
}

/* ================================================================== softmax */

TEST(softmax_normalises) {
    auto s = set_of({ 0.0f, (float)std::log(2.0), (float)std::log(3.0) });
    host_softmax(s, false);
    CHECK_NEAR(s.c[0].p, 1.0 / 6.0, 1e-6);
    CHECK_NEAR(s.c[1].p, 2.0 / 6.0, 1e-6);
    CHECK_NEAR(s.c[2].p, 3.0 / 6.0, 1e-6);
}

/* ================================================================== top-k */

TEST(top_k_keeps_k_highest) {
    auto s = set_of({ 5.0f, 1.0f, 4.0f, 2.0f, 3.0f });
    host_top_k(s, 2);
    CHECK_EQ(s.c.size(), (size_t)2);
    CHECK_EQ(s.c[0].id, 0);
    CHECK_EQ(s.c[1].id, 2);
}

TEST(top_k_zero_is_off) {
    auto s = set_of({ 5.0f, 1.0f, 4.0f });
    host_top_k(s, 0);
    CHECK_EQ(s.c.size(), (size_t)3);
}

TEST(top_k_ties_break_on_lowest_id) {
    /* Three equal logits and k=2: the answer must depend on the ids, not on how the sort landed,
     * or the device kernel cannot reproduce it. */
    auto s = set_of({ 2.0f, 2.0f, 2.0f, 1.0f });
    host_top_k(s, 2);
    CHECK_EQ(s.c.size(), (size_t)2);
    CHECK_EQ(s.c[0].id, 0);
    CHECK_EQ(s.c[1].id, 1);
}

/* ================================================================== top-p */

TEST(top_p_cuts_at_the_nucleus) {
    /* probs 0.5 0.3 0.15 0.05: cumulative 0.5, 0.8, 0.95, 1.0. The thresholds sit clear of those
     * cumulants; at exactly p = 0.8 the answer depends on float addition (see
     * top_p_boundary_is_the_float_sum). */
    auto l = logits_for({ 0.5, 0.3, 0.15, 0.05 });

    { auto s = set_of(l); host_top_p(s, 0.4f,  1); CHECK_EQ(s.c.size(), (size_t)1); }
    { auto s = set_of(l); host_top_p(s, 0.7f,  1); CHECK_EQ(s.c.size(), (size_t)2); }
    { auto s = set_of(l); host_top_p(s, 0.85f, 1); CHECK_EQ(s.c.size(), (size_t)3); }
    { auto s = set_of(l); host_top_p(s, 0.99f, 1); CHECK_EQ(s.c.size(), (size_t)4); }
    /* p >= 1 is off entirely. */
    { auto s = set_of(l); host_top_p(s, 1.0f,  1); CHECK_EQ(s.c.size(), (size_t)4); }
}

TEST(top_p_boundary_is_the_float_sum) {
    /* The cut is where the running float sum reaches p, not the exact sum. With probs
     * 0.5 0.3 0.15 0.05 the sum after two terms is just under 0.8, so p = 0.8 keeps three.
     * The device kernel accumulates the same way (as does llama.cpp); a reference that rounded
     * here would flag every near-boundary request as a kernel mismatch. */
    auto l = logits_for({ 0.5, 0.3, 0.15, 0.05 });
    auto s = set_of(l);
    host_softmax(s, true);
    CHECK(s.c[0].p + s.c[1].p < 0.8f);

    auto t = set_of(l);
    host_top_p(t, 0.8f, 1);
    CHECK_EQ(t.c.size(), (size_t)3);

    /* Where the sum is exact (dyadic probabilities) the boundary follows arithmetic: cumulative
     * 0.5 then 0.75, and p = 0.75 keeps exactly two. */
    auto d = set_of(logits_for({ 0.5, 0.25, 0.125, 0.125 }));
    host_softmax(d, true);
    CHECK_NEAR(d.c[0].p + d.c[1].p, 0.75, 1e-6);
}

TEST(top_p_single_dominant_token) {
    auto s = set_of(logits_for({ 0.99, 0.005, 0.005 }));
    host_top_p(s, 0.9f, 1);
    CHECK_EQ(s.c.size(), (size_t)1);
    CHECK_EQ(s.c[0].id, 0);
}

TEST(top_p_flat_distribution_and_ties) {
    /* Four identical probabilities. The cut lands exactly between the second and the third, so
     * the answer is the two lowest ids, the only answer a device kernel can also produce. */
    auto s = set_of({ 1.0f, 1.0f, 1.0f, 1.0f });
    host_top_p(s, 0.5f, 1);
    CHECK_EQ(s.c.size(), (size_t)2);
    CHECK_EQ(s.c[0].id, 0);
    CHECK_EQ(s.c[1].id, 1);
}

TEST(top_p_min_keep_holds_the_floor) {
    auto s = set_of(logits_for({ 0.99, 0.005, 0.005 }));
    host_top_p(s, 0.5f, 3);
    CHECK_EQ(s.c.size(), (size_t)3);
}

/* ================================================================== min-p */

TEST(min_p_is_a_floor_relative_to_the_mode) {
    auto l = logits_for({ 0.5, 0.3, 0.15, 0.05 });

    /* min_p 0.4 -> threshold 0.2: keeps 0.5 and 0.3. */
    { auto s = set_of(l); host_min_p(s, 0.4f, 1); CHECK_EQ(s.c.size(), (size_t)2); }
    /* min_p 0.2 -> threshold 0.1: keeps 0.5, 0.3, 0.15. */
    { auto s = set_of(l); host_min_p(s, 0.2f, 1); CHECK_EQ(s.c.size(), (size_t)3); }
    /* min_p 0 is off. */
    { auto s = set_of(l); host_min_p(s, 0.0f, 1); CHECK_EQ(s.c.size(), (size_t)4); }
}

TEST(min_p_single_dominant_token) {
    /* The mode always survives, even when the floor would exclude everything. */
    auto s = set_of(logits_for({ 0.99, 0.005, 0.005 }));
    host_min_p(s, 0.5f, 1);
    CHECK_EQ(s.c.size(), (size_t)1);
    CHECK_EQ(s.c[0].id, 0);
}

TEST(min_p_flat_distribution_keeps_everything) {
    auto s = set_of({ 1.0f, 1.0f, 1.0f, 1.0f });
    host_min_p(s, 0.5f, 1);
    CHECK_EQ(s.c.size(), (size_t)4);
}

/* ================================================================== typical */

TEST(typical_on_a_uniform_distribution) {
    /* Every candidate's surprisal equals the entropy, so all four sit at distance zero, the
     * smallest. The cut keeps that whole tie whatever p asks, as sample_typical does, instead of
     * choosing among equal candidates by id. */
    auto s = set_of({ 1.0f, 1.0f, 1.0f, 1.0f });
    host_typical(s, 0.5f, 1);
    CHECK_EQ(s.c.size(), (size_t)4);
}

TEST(typical_keeps_every_candidate_tied_at_its_cut) {
    /* probs 0.4 0.2 0.2 0.2. H = 1.3322 nats; surprisals 0.916 and 1.609, so the distances are
     * 0.416 and 0.277 three times. p = 0.3 is reached inside the tie: the three 0.2s stay (mass
     * 0.6) and the 0.4 goes. A prefix cut, upstream's, would keep two of the three by id. */
    auto s = set_of(logits_for({ 0.4, 0.2, 0.2, 0.2 }));
    host_typical(s, 0.3f, 1);
    CHECK(ids_of(s) == (std::vector<int32_t>{ 1, 2, 3 }));
}

TEST(typical_min_keep_holds_the_floor) {
    /* probs 0.6 0.2 0.15 0.05 at p = 0.1 keep only the most typical candidate (see below);
     * min_keep 3 adds the next two by distance, 0 and 2, and the set comes back in logit order. */
    auto s = set_of(logits_for({ 0.6, 0.2, 0.15, 0.05 }));
    host_typical(s, 0.1f, 3);
    CHECK(ids_of(s) == (std::vector<int32_t>{ 0, 1, 2 }));
}

/* top_p, typical_p and xtc_threshold at 0 turn the stage off, as the chain's ops read them; the
 * host reference must not read 0 as llama.cpp does (keep one candidate, or keep only the least
 * probable). */
TEST(a_zero_top_p_typical_p_or_xtc_threshold_is_off) {
    const auto l = logits_for({ 0.5, 0.3, 0.15, 0.05 });
    { auto s = set_of(l); host_top_p(s, 0.0f, 1);   CHECK_EQ(s.c.size(), (size_t)4); }
    { auto s = set_of(l); host_typical(s, 0.0f, 1); CHECK_EQ(s.c.size(), (size_t)4); }
    { auto s = set_of(l); host_xtc(s, 1.0f, 0.0f, 1, 7, 0); CHECK_EQ(s.c.size(), (size_t)4); }
    /* Above 0.5 at most one candidate can reach the threshold, so XTC has nothing to drop. */
    { auto s = set_of(l); host_xtc(s, 1.0f, 0.6f, 1, 7, 0); CHECK_EQ(s.c.size(), (size_t)4); }
}

TEST(typical_prefers_the_expected_surprisal) {
    /* probs 0.6 0.2 0.15 0.05. Entropy H = 1.1055 nats; surprisals are 0.511, 1.609, 1.897,
     * 2.996, so the distances to H are 0.595, 0.504, 0.792, 1.890: the second candidate is the
     * most typical, not the first. This is what separates typical sampling from top-p. */
    auto s = set_of(logits_for({ 0.6, 0.2, 0.15, 0.05 }));
    host_typical(s, 0.1f, 1);
    CHECK_EQ(s.c.size(), (size_t)1);
    CHECK_EQ(s.c[0].id, 1);
}

/* ================================================================== penalties */

TEST(penalties_over_a_history_window) {
    /* rep 2.0, freq 0.5, pres 0.25 over history {0,0,1}, last_n = 3.
     *   token 0: count 2, logit  1.0 > 0 -> /2 = 0.5, then -(2*0.5 + 0.25) = -0.75
     *   token 1: count 1, logit -1.0 <=0 -> *2 = -2.0, then -(1*0.5 + 0.25) = -2.75
     *   token 2: absent, untouched. */
    const int32_t hist[] = { 0, 0, 1 };
    auto s = set_of({ 1.0f, -1.0f, 0.0f });
    host_penalties(s, hist, 3, 3, 2.0f, 0.5f, 0.25f);
    CHECK_NEAR(s.c[0].logit, -0.75f, 1e-6);
    CHECK_NEAR(s.c[1].logit, -2.75f, 1e-6);
    CHECK_NEAR(s.c[2].logit,  0.0f,  1e-6);
}

TEST(penalties_respect_the_window_length) {
    /* The same history with last_n = 2 counts only {0,1}, so token 0's count drops from 2 to 1. */
    const int32_t hist[] = { 0, 0, 1 };
    auto s = set_of({ 1.0f, -1.0f, 0.0f });
    host_penalties(s, hist, 3, 2, 2.0f, 0.5f, 0.25f);
    CHECK_NEAR(s.c[0].logit, -0.25f, 1e-6);
}

/* A window of -1 counts the whole history; a positive one only its tail. Token 3 appears only at
 * the start of a 100-token history. */
TEST(penalties_window_of_minus_one_counts_the_whole_history) {
    std::vector<int32_t> hist(100, 0);
    hist[0] = 3;
    { auto s = set_of({ 1.0f, 1.0f, 1.0f, 2.0f });
      host_penalties(s, hist.data(), 100, -1, 2.0f, 0.0f, 0.0f);
      CHECK_EQ(s.c[3].logit, 1.0f); }
    { auto s = set_of({ 1.0f, 1.0f, 1.0f, 2.0f });
      host_penalties(s, hist.data(), 100, 50, 2.0f, 0.0f, 0.0f);
      CHECK_EQ(s.c[3].logit, 2.0f); }
}

TEST(penalties_identity_is_a_no_op) {
    const int32_t hist[] = { 0, 0, 1 };
    auto s = set_of({ 1.0f, -1.0f, 0.0f });
    host_penalties(s, hist, 3, 3, 1.0f, 0.0f, 0.0f);
    CHECK_EQ(s.c[0].logit, 1.0f);
    CHECK_EQ(s.c[1].logit, -1.0f);
}

/* ================================================================== DRY */

TEST(dry_penalises_the_token_that_extends_a_repeat) {
    /* Upstream's own worked example, tokens a=0 b=1 c=2 y=3:
     *     last tokens:  a b c c b c y a b c
     *     repeat counts 0 0 3 1 0 2 0 0 0 0
     * With allowed_length 2, only counts >= 2 matter, and each names the token that would extend
     * that repeat: c (reaching 3) and y (reaching 2). At base 2 and multiplier 1 the penalties
     * are 2^(3-2) = 2 and 2^(2-2) = 1. */
    const int32_t hist[] = { 0, 1, 2, 2, 1, 2, 3, 0, 1, 2 };

    DryParams dp;
    dp.multiplier     = 1.0f;
    dp.base           = 2.0f;
    dp.allowed_length = 2;
    dp.penalty_last_n = -1;

    DryBreakers none;
    auto s = set_of({ 0.0f, 0.0f, 0.0f, 0.0f });
    host_dry(s, hist, 10, 10, dp, none);

    CHECK_NEAR(s.c[0].logit,  0.0f, 1e-6);   /* a: never extends a repeat */
    CHECK_NEAR(s.c[1].logit,  0.0f, 1e-6);   /* b: its repeat is only 1 long */
    CHECK_NEAR(s.c[2].logit, -2.0f, 1e-6);   /* c */
    CHECK_NEAR(s.c[3].logit, -1.0f, 1e-6);   /* y */
}

TEST(dry_runs_in_the_chain_without_breakers) {
    /* The history of the worked example above: c (id 2) extends a repeat of 3 and is penalised
     * by 2, which takes it below b. The device chain runs DRY with no breakers too. */
    const int32_t hist[] = { 0, 1, 2, 2, 1, 2, 3, 0, 1, 2 };
    const float logits[] = { 0.0f, 1.0f, 2.5f, -5.0f };
    SamplingParams sp;
    sp.top_k              = 1;
    sp.dry_multiplier     = 1.0f;
    sp.dry_base           = 2.0f;
    sp.dry_allowed_length = 2;
    sp.dry_penalty_last_n = -1;
    HostRefInput in;
    in.history   = hist;
    in.n_history = 10;
    CHECK_EQ(host_sample(logits, 4, sp, in), 1);
}

TEST(dry_off_when_multiplier_is_zero) {
    const int32_t hist[] = { 0, 1, 2, 2, 1, 2, 3, 0, 1, 2 };
    DryParams dp;
    dp.multiplier = 0.0f;
    DryBreakers none;
    auto s = set_of({ 0.0f, 0.0f, 0.0f, 0.0f });
    host_dry(s, hist, 10, 10, dp, none);
    CHECK_EQ(s.c[2].logit, 0.0f);
}

/* ================================================================== XTC */

TEST(xtc_drops_the_obvious_continuations) {
    /* probs 0.5 0.3 0.15 0.05, threshold 0.1: the first three are above it, so all above the
     * threshold except the least probable of them are removed. Probability 1 takes the coin out
     * of the test. */
    auto s = set_of(logits_for({ 0.5, 0.3, 0.15, 0.05 }));
    host_xtc(s, 1.0f, 0.1f, 1, /*seed*/ 7, /*pos*/ 0);
    CHECK_EQ(s.c.size(), (size_t)2);
    CHECK_EQ(s.c[0].id, 2);
    CHECK_EQ(s.c[1].id, 3);
}

/* DRY with a sequence breaker, by hand. Vocabulary b \n c d y, breaker "\n", multiplier 1, base
 * 2, allowed length 2.
 *
 * History "b \n c d y b \n c d": the suffix "b \n c d" occurred before, followed by y, so y would
 * extend a repeat of 4: penalty 2^(4-2) = 4. The breaker 2 tokens from the end caps the countable
 * repeat at 2: penalty 2^0 = 1.
 *
 * History "c d \n c d": "c d" was followed by "\n", a whole breaker, which is never penalised. */
TEST(dry_breakers_cap_the_repeat_and_exempt_themselves_in_the_reference) {
    SimpleVocab v({ "b", "\n", "c", "d", "y" });
    DryBreakers brk, none;
    CHECK_OK(dry_prepare_breakers(v, { "\n" }, &brk));
    CHECK(brk.is_single(1));
    DryParams dp;
    dp.multiplier = 1.0f;
    dp.base = 2.0f;
    dp.allowed_length = 2;

    const int32_t h1[] = { 0, 1, 2, 3, 4, 0, 1, 2, 3 };
    { auto s = set_of({ 0, 0, 0, 0, 0 }); host_dry(s, h1, 9, 9, dp, none);
      CHECK(s.c[4].logit == -4.0f && s.c[0].logit == 0.0f && s.c[1].logit == 0.0f); }
    { auto s = set_of({ 0, 0, 0, 0, 0 }); host_dry(s, h1, 9, 9, dp, brk);
      CHECK_EQ(s.c[4].logit, -1.0f); }
    CHECK_EQ(dry_rep_limit(brk, h1, 9), 2);

    const int32_t h2[] = { 2, 3, 1, 2, 3 };
    { auto s = set_of({ 0, 0, 0, 0, 0 }); host_dry(s, h2, 5, 5, dp, none);
      CHECK_EQ(s.c[1].logit, -1.0f); }
    { auto s = set_of({ 0, 0, 0, 0, 0 }); host_dry(s, h2, 5, 5, dp, brk);
      CHECK_EQ(s.c[1].logit, 0.0f); }
}

/* The penalty's exponent is clamped so pow() stays finite: a 300-token run at base 3 would be
 * 3^297, and the clamp is floor(log(FLT_MAX) / ln 3) = 80. An infinite penalty would make the
 * recovered logprob of such a token NaN. */
TEST(dry_penalty_exponent_is_clamped_below_overflow) {
    std::vector<int32_t> hist(300, 0);
    DryParams dp;
    dp.multiplier = 1.0f;
    dp.base = 3.0f;
    dp.allowed_length = 2;
    auto s = set_of({ 0.0f, 0.0f });
    host_dry(s, hist.data(), 300, 300, dp, DryBreakers{});
    CHECK(std::isfinite(s.c[0].logit));
    CHECK_EQ(s.c[0].logit, -std::pow(3.0f, 80.0f));
    CHECK_EQ(s.c[1].logit, 0.0f);
}

TEST(xtc_zero_probability_never_fires) {
    auto s = set_of(logits_for({ 0.5, 0.3, 0.15, 0.05 }));
    host_xtc(s, 0.0f, 0.1f, 1, 7, 0);
    CHECK_EQ(s.c.size(), (size_t)4);
}

/* ================================================================== the mask */

TEST(mask_takes_forbidden_logits_to_negative_infinity) {
    uint32_t bits = 0;
    bits |= 1u << 1;
    bits |= 1u << 3;
    auto s = set_of({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    host_mask(s, &bits, 1);
    CHECK(std::isinf(s.c[0].logit) && s.c[0].logit < 0);
    CHECK_EQ(s.c[1].logit, 2.0f);
    CHECK(std::isinf(s.c[2].logit) && s.c[2].logit < 0);
    CHECK_EQ(s.c[3].logit, 4.0f);
    CHECK(std::isinf(s.c[4].logit) && s.c[4].logit < 0);
}

/* ================================================================== the RNG and the pick */

TEST(rng_is_a_pure_function_of_its_coordinates) {
    for (uint64_t pos = 0; pos < 64; ++pos) {
        const float a = rng_uniform(12345, pos, RAD_RNG_PICK);
        const float b = rng_uniform(12345, pos, RAD_RNG_PICK);
        CHECK_EQ(a, b);
        CHECK(a >= 0.0f && a < 1.0f);
    }
    /* Different streams at the same position must give different draws, or XTC's coin and the
     * pick would be correlated. */
    int same = 0;
    for (uint64_t pos = 0; pos < 64; ++pos)
        if (rng_uniform(1, pos, RAD_RNG_PICK) == rng_uniform(1, pos, RAD_RNG_XTC)) ++same;
    CHECK_EQ(same, 0);
}

TEST(pick_is_reproducible_from_seed_and_position) {
    auto l = logits_for({ 0.4, 0.3, 0.2, 0.1 });

    for (uint64_t pos = 0; pos < 32; ++pos) {
        auto a = set_of(l);
        auto b = set_of(l);
        const int32_t ta = host_pick(a, 99, pos);
        const int32_t tb = host_pick(b, 99, pos);
        CHECK_EQ(ta, tb);
        CHECK(ta >= 0 && ta < 4);
    }

    /* The set's order must not change the answer: the pick walks a total order (descending
     * logit, ties to the lower id) so a permuted candidate set draws the same token. */
    auto asc = set_of(l);
    auto srt = set_of(l);
    srt.sort_desc();
    CHECK_EQ(host_pick(asc, 99, 5), host_pick(srt, 99, 5));
}

TEST(pick_returns_the_only_token_when_one_dominates) {
    auto s = set_of({ 100.0f, 0.0f, 0.0f });
    for (uint64_t pos = 0; pos < 32; ++pos) {
        auto t = s;
        CHECK_EQ(host_pick(t, 3, pos), 0);
    }
}

TEST(pick_visits_both_halves_of_a_fair_coin) {
    /* A seeded pick is deterministic, so this is a fixed answer, not a statistical claim; it
     * catches an inverse CDF that always returns the first candidate. */
    int n0 = 0, n1 = 0;
    for (uint64_t pos = 0; pos < 1000; ++pos) {
        auto s = set_of({ 0.0f, 0.0f });
        const int32_t t = host_pick(s, 4242, pos);
        if (t == 0) ++n0; else ++n1;
    }
    CHECK_EQ(n0 + n1, 1000);
    CHECK(n0 > 400 && n0 < 600);
}

TEST(pick_refuses_a_fully_masked_distribution) {
    auto s = set_of({ 1.0f, 2.0f });
    uint32_t none = 0;
    host_mask(s, &none, 1);
    CHECK_EQ(host_pick(s, 1, 0), -1);
}

/* ================================================================== the whole chain */

TEST(greedy_chain_masks_before_it_picks) {
    /* Temperature 0 collapses every other logit to -inf, so a mask applied after it would leave
     * nothing. The greedy path masks first and takes the argmax of what survives. */
    SamplingParams sp;
    sp.temp = 0.0f;
    CHECK(sp.greedy());

    uint32_t bits = (1u << 1) | (1u << 2);
    const float logits[] = { 10.0f, 3.0f, 5.0f, 9.0f };

    HostRefInput in;
    in.bitmask = &bits;
    in.n_words = 1;

    CHECK_EQ(host_sample(logits, 4, sp, in), 2);
}

TEST(chain_with_top_k_one_is_greedy) {
    SamplingParams sp;
    sp.top_k = 1;
    CHECK(sp.greedy());
    const float logits[] = { 1.0f, 7.0f, 2.0f };
    HostRefInput in;
    CHECK_EQ(host_sample(logits, 3, sp, in), 1);
}

TEST(chain_order_temperature_before_the_cuts_is_observable) {
    /* Radiance applies temperature before top-p; llama.cpp after. With logits 3/2/1 and a high
     * temperature the nucleus of 0.5 keeps more candidates. If the orders ever agree on these
     * numbers, the deviation documented in host_ref.h is gone and rad-kbench compares the wrong
     * thing. */
    auto l = std::vector<float>{ 3.0f, 2.0f, 1.0f };

    auto a = set_of(l);
    host_temp(a, 5.0f);
    host_top_p(a, 0.5f, 1);

    auto b = set_of(l);
    host_top_p(b, 0.5f, 1);
    host_temp(b, 5.0f);

    CHECK(a.c.size() != b.c.size());
}

TEST(temp_last_draws_only_from_the_untempered_nucleus) {
    /* The same numbers through the whole chain. Temperature last is llama.cpp's order, requested
     * with `samplers`: the nucleus is cut on the untempered distribution, so a hot temperature
     * flattens only what survived and never revives a removed token. Temperature first cuts on
     * the flattened distribution and keeps more. */
    const float logits[3] = { 3.0f, 2.0f, 1.0f };
    auto b = set_of({ 3.0f, 2.0f, 1.0f });
    host_top_p(b, 0.5f, 1);
    std::set<int32_t> nucleus;
    for (const auto& e : b.c) nucleus.insert(e.id);

    SamplingParams sp;
    sp.temp = 5.0f;
    sp.top_p = 0.5f;
    std::set<int32_t> first, last;
    for (uint64_t seed = 0; seed < 256; ++seed) {
        sp.seed = seed;
        HostRefInput in;
        sp.temp_last = false;
        first.insert(host_sample(logits, 3, sp, in));
        sp.temp_last = true;
        last.insert(host_sample(logits, 3, sp, in));
    }
    for (int32_t t : last) CHECK(nucleus.count(t) == 1);
    CHECK(first.size() > last.size());
}

/* ================================================================== GBNF */

namespace {

/* a b c ab d and an end-of-generation token: small enough that the admissible set at every
 * position can be worked out by hand. */
SimpleVocab tiny_vocab() {
    SimpleVocab v;
    v.add("a");        /* 0 */
    v.add("b");        /* 1 */
    v.add("c");        /* 2 */
    v.add("ab");       /* 3 */
    v.add("d");        /* 4 */
    v.add("", true);   /* 5: end of generation, no text */
    return v;
}

/* The ids of `mask`'s set bits below n_tok, except `skip`. */
std::vector<int32_t> set_ids(const std::vector<uint32_t>& mask, int64_t n_tok, int32_t skip = -1) {
    std::vector<int32_t> ids;
    for (int32_t i = 0; i < (int32_t)n_tok; ++i)
        if (i != skip && (mask[(size_t)(i >> 5)] & (1u << ((uint32_t)i & 31u)))) ids.push_back(i);
    return ids;
}

std::vector<int32_t> admitted(const GbnfGrammar& g, int64_t n_tok) {
    std::vector<uint32_t> mask((size_t)bitmask_words(n_tok), 0u);
    if (g.fill_mask(mask.data(), (int64_t)mask.size()) < 0) return {};
    return set_ids(mask, n_tok);
}

}  /* namespace */

TEST(gbnf_admits_exactly_the_right_tokens_at_each_position) {
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_OK(GbnfGrammar::create("root ::= \"ab\" | \"c\"\n", "root", &v, false, {}, {}, &g, &err));

    /* Position 0: "a" (a prefix of ab), "ab" (the whole alternative) and "c". Not "b", not "d",
     * and not end-of-generation, since no parse has completed. */
    auto p0 = admitted(*g, v.n_tokens());
    CHECK_EQ(p0.size(), (size_t)3);
    CHECK_EQ(p0[0], 0);
    CHECK_EQ(p0[1], 2);
    CHECK_EQ(p0[2], 3);

    /* After "a" only "b" continues the parse. */
    CHECK_OK(g->accept_token(0));
    auto p1 = admitted(*g, v.n_tokens());
    CHECK_EQ(p1.size(), (size_t)1);
    CHECK_EQ(p1[0], 1);

    /* After "b" the parse is complete, so only end-of-generation is admissible; admitting text
     * here would produce output past the grammar. */
    CHECK_OK(g->accept_token(1));
    CHECK(g->complete());
    auto p2 = admitted(*g, v.n_tokens());
    CHECK_EQ(p2.size(), (size_t)1);
    CHECK_EQ(p2[0], 5);
}

TEST(gbnf_multi_character_token_is_admitted_whole) {
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_OK(GbnfGrammar::create("root ::= \"ab\" | \"c\"\n", "root", &v, false, {}, {}, &g, &err));

    CHECK_OK(g->accept_token(3));       /* "ab" in one token */
    CHECK(g->complete());
    auto p = admitted(*g, v.n_tokens());
    CHECK_EQ(p.size(), (size_t)1);
    CHECK_EQ(p[0], 5);
}

TEST(gbnf_char_class_and_repetition) {
    SimpleVocab v;
    v.add("0"); v.add("1"); v.add("x"); v.add("01"); v.add("", true);
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_OK(GbnfGrammar::create("root ::= [01]+\n", "root", &v, false, {}, {}, &g, &err));

    auto p0 = admitted(*g, v.n_tokens());
    CHECK_EQ(p0.size(), (size_t)3);      /* "0", "1", "01" -- not "x", not eog */
    CHECK_EQ(p0[0], 0);
    CHECK_EQ(p0[1], 1);
    CHECK_EQ(p0[2], 3);

    CHECK_OK(g->accept_token(0));
    /* One digit is a complete match and more digits are still admissible, so eog joins the set
     * rather than replacing it. */
    auto p1 = admitted(*g, v.n_tokens());
    CHECK_EQ(p1.size(), (size_t)4);
    CHECK_EQ(p1[3], 4);
}

/* The mask engine against upstream's walk, bit for bit, over every state a JSON tool call
 * reaches: fill_mask uses a per-grammar precomputed plan, the oracle rescans every candidate at
 * every depth for every stack. The vocabulary is shaped like byte-level BPE (long pieces, shared
 * prefixes, pieces ending mid-character) to reach the trie's depth and the partial-UTF-8 paths.
 * gbnf_test and rad-gbnf-check cover far more; this is the fast check that runs anywhere. */
TEST(gbnf_mask_agrees_with_the_oracle_on_a_json_tool_call) {
    SimpleVocab v;
    for (int c = 0x20; c < 0x7f; ++c) v.add(std::string(1, (char)c));
    const char* frag[] = {
        "{\"", "\":", "\":\"", "\",\"", "\"}", "}}", "{}", "[]", "[{", "}]", "true", "false",
        "null", "name", "value", "args", "query", "path", "city", "unit", "count", "id", "type",
        "get_", "set_", "_weather", "_time", "0.0", "1.5", "-1", "12", "345", "6789", "  ",
        "    ", "\\n", "\\\"", "\\u00e9", "e+", "E-", "the", " the", " a", "ing", "tion", "s\"",
        "\"\"", ": ", ", ", "\": \"", "\", \"", "abc", "xyz",
    };
    for (const char* f : frag) v.add(f);
    /* Multi-byte characters whole, then the same bytes split so a piece leaves an incomplete
     * sequence and its neighbour resumes it, as byte-level BPE often does. */
    const char* utf8[] = { "\xc3\xa9", "\xc3\xbc", "\xe2\x82\xac", "\xe6\x97\xa5",
                           "\xf0\x9f\x98\x80" };
    for (const char* u : utf8) {
        v.add(u);
        const std::string s(u);
        v.add(s.substr(0, 1));                      /* leaves n_remain > 0 */
        v.add(s.substr(1));                         /* pure continuation bytes */
        if (s.size() > 2) v.add(s.substr(0, 2));
    }
    /* A prefix-heavy family: what a trie is for and a flat scan pays for. */
    for (char a = 'a'; a <= 'p'; ++a)
        for (char b = 'a'; b <= 'p'; ++b) {
            v.add(std::string("\"") + a + b);
            v.add(std::string() + a + b + "\":");
        }
    const int32_t eog = v.add("", true);

    std::string schema =
        "{\"type\":\"object\",\"properties\":{"
        "\"name\":{\"type\":\"string\"},"
        "\"arguments\":{\"type\":\"object\",\"properties\":{"
        "  \"city\":{\"type\":\"string\"},"
        "  \"days\":{\"type\":\"integer\"},"
        "  \"flags\":{\"type\":\"array\",\"items\":{\"type\":\"boolean\"}},"
        "  \"unit\":{\"type\":\"string\",\"enum\":[\"c\",\"f\"]}},"
        "  \"required\":[\"city\"]}},"
        "\"required\":[\"name\",\"arguments\"]}";
    std::string gbnf, err;
    CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));

    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<uint32_t> fast((size_t)n_words), ref((size_t)n_words);

    /* Several independent paths through the grammar, each choosing among admitted tokens, so the
     * states visited are reachable ones. */
    int64_t masks = 0, states_deep = 0;
    for (uint64_t seed = 1; seed <= 24; ++seed) {
        std::unique_ptr<GbnfGrammar> g;
        CHECK_OK(GbnfGrammar::create(gbnf, "root", &v, false, {}, {}, &g, &err));

        uint64_t rng = seed * 0x9e3779b97f4a7c15ull;
        for (int step = 0; step < 120; ++step) {
            const int fs = g->fill_mask(fast.data(), n_words);
            const int rs = gbnf_oracle_mask(*g, ref.data(), n_words);
            CHECK_EQ(fs, rs);
            ++masks;
            if (fs < 0) break;
            CHECK_EQ(std::memcmp(fast.data(), ref.data(), (size_t)n_words * 4), 0);

            /* Advance on an admitted token, preferring text over end-of-generation so the path
             * continues; take eog only when nothing else is admitted. */
            const std::vector<int32_t> ids = set_ids(fast, v.n_tokens(), eog);
            if (ids.empty()) break;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            CHECK_OK(g->accept_token(ids[(size_t)(rng % ids.size())]));
            if (step > 20) ++states_deep;
        }
    }
    /* Paths that died immediately would prove nothing, so check how much was walked: a grammar
     * dead at step one must fail here. */
    CHECK(masks > 900);
    CHECK(states_deep > 400);
}

TEST(gbnf_rejects_a_malformed_grammar) {
    SimpleVocab v = tiny_vocab();
    std::string err;

    /* An unterminated string literal. The parser throws internally, as upstream's does, and the
     * failure arrives as a status and a message; nothing escapes across the boundary. */
    {
        std::unique_ptr<GbnfGrammar> g;
        CHECK_EQ(GbnfGrammar::create("root ::= \"abc\n", "root", &v, false, {}, {}, &g, &err),
                 RAD_E_FORMAT);
        CHECK(!err.empty());
    }
    /* A reference to an undefined rule, reported by name rather than by index. */
    {
        std::unique_ptr<GbnfGrammar> g;
        err.clear();
        CHECK_EQ(GbnfGrammar::create("root ::= missing\n", "root", &v, false, {}, {}, &g, &err),
                 RAD_E_FORMAT);
        CHECK(err.find("missing") != std::string::npos);
    }
    /* An unclosed group. */
    {
        std::unique_ptr<GbnfGrammar> g;
        CHECK(GbnfGrammar::create("root ::= (\"a\"\n", "root", &v, false, {}, {}, &g, &err) < 0);
    }
}

TEST(gbnf_rejects_left_recursion_by_name) {
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    /* A pushdown automaton cannot run this, and refusing is better than hanging. */
    const int st = GbnfGrammar::create("root ::= root \"a\" | \"a\"\n", "root", &v,
                                       false, {}, {}, &g, &err);
    CHECK_EQ(st, RAD_E_UNSUPPORTED);
}

TEST(gbnf_missing_root_is_named) {
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(GbnfGrammar::create("other ::= \"a\"\n", "root", &v, false, {}, {}, &g, &err),
             RAD_E_NOTFOUND);
}

/* ================================================================== GBNF: what a grammar may cost
 *
 * A /v1/completions grammar is client text, compiled on the admission thread and walked on the
 * scheduler thread. Each case is a small grammar that, unbounded, would exhaust memory, time or
 * stack. A regression shows as a killed or hung test rather than a failed CHECK. */

namespace {

int gbnf_create(const std::string& text, const VocabView& v, std::unique_ptr<GbnfGrammar>* g,
                std::string* err) {
    return GbnfGrammar::create(text, "root", &v, false, {}, {}, g, err);
}

}  /* namespace */

TEST(gbnf_refuses_a_repetition_whose_minimum_is_above_its_maximum) {
    /* max - min would wrap to about 1.8e19 optional rules for the parser to build. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(gbnf_create("root ::= \"a\"{5,3}\n", v, &g, &err), RAD_E_FORMAT);
    CHECK(err.find("minimum") != std::string::npos);
    /* The ordinary ranges around it still compile. */
    CHECK_OK(gbnf_create("root ::= \"a\"{3,5}\n", v, &g, &err));
    CHECK_OK(gbnf_create("root ::= \"a\"{3,3}\n", v, &g, &err));
}

TEST(gbnf_refuses_a_repetition_that_expands_past_its_budget) {
    /* 100 KB of literal repeated 1999 times is 1.6 GB of rule, under the per-repetition count
     * limit. It is refused on its cost, before any of it is allocated. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    const std::string big = "root ::= \"" + std::string(100000, 'a') + "\"{1999}\n";
    CHECK_EQ(gbnf_create(big, v, &g, &err), RAD_E_FORMAT);
    CHECK(err.find("MiB") != std::string::npos);
    /* The same literal once is fine: the text size itself is not bounded. */
    CHECK_OK(gbnf_create("root ::= \"" + std::string(100000, 'a') + "\"\n", v, &g, &err));
}

TEST(gbnf_refuses_parentheses_nested_past_the_limit) {
    /* 20,000 levels is a stack overflow in a recursive-descent parser. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    const std::string deep = "root ::= " + std::string(20000, '(') + "\"a\"" +
                             std::string(20000, ')') + "\n";
    CHECK_EQ(gbnf_create(deep, v, &g, &err), RAD_E_FORMAT);
    CHECK(err.find("nest") != std::string::npos);

    /* A depth a hand-written grammar could plausibly use still compiles and still runs. */
    const std::string ok = "root ::= " + std::string(200, '(') + "\"a\"" +
                           std::string(200, ')') + "\n";
    CHECK_OK(gbnf_create(ok, v, &g, &err));
    const auto ids = admitted(*g, v.n_tokens());
    CHECK_EQ(ids.size(), (size_t)1);
    CHECK_EQ(ids[0], 0);
}

TEST(gbnf_left_recursion_check_does_not_recurse_per_rule) {
    /* A 100,000-rule chain has no left recursion, but a recursive check would need a stack frame
     * per rule to find that out. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    std::string chain = "root ::= r0\n";
    for (int i = 0; i < 100000; ++i)
        chain += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + "\n";
    chain += "r100000 ::= \"a\"\n";
    CHECK_OK(gbnf_create(chain, v, &g, &err));
    const auto ids = admitted(*g, v.n_tokens());
    CHECK_EQ(ids.size(), (size_t)1);
    CHECK_EQ(ids[0], 0);

    /* Every rule reaches the next twice, so a check that revisits cleared rules takes 2^60 steps.
     * The closure itself de-duplicates, so the grammar is cheap to run. */
    std::string diamond = "root ::= r0\n";
    for (int i = 0; i < 60; ++i)
        diamond += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + " | r" +
                   std::to_string(i + 1) + "\n";
    diamond += "r60 ::= \"a\"\n";
    CHECK_OK(gbnf_create(diamond, v, &g, &err));
    CHECK_EQ(admitted(*g, v.n_tokens()).size(), (size_t)1);
}

TEST(gbnf_refuses_left_recursion_through_an_indirectly_nullable_rule) {
    /* `("b"?)` can match nothing, so `root` can start with `root`, but only through a rule that
     * is nullable because another rule is. A check that sees only empty alternatives misses this,
     * and the closure then grows a stack of "c"s until the process dies. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(gbnf_create("root ::= (\"b\"?) root \"c\" | \"a\"\n", v, &g, &err),
             RAD_E_UNSUPPORTED);
    CHECK(err.find("'root'") != std::string::npos);

    err.clear();
    CHECK_EQ(gbnf_create("root ::= e root \"c\" | \"a\"\ne ::= f\nf ::= \"b\" |\n", v, &g, &err),
             RAD_E_UNSUPPORTED);
    CHECK(err.find("left recursion") != std::string::npos);
}

TEST(gbnf_a_cycle_in_tail_position_is_not_left_recursion) {
    /* A star over something nullable is a cycle of rules that can come first, but each is the
     * last thing in its alternative, so the stack never grows: the closure meets a stack it has
     * seen and stops. It must compile and run. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_OK(gbnf_create("root ::= (\"a\"?)*\n", v, &g, &err));
    for (int i = 0; i < 4; ++i) {
        const auto ids = admitted(*g, v.n_tokens());
        CHECK_EQ(ids.size(), (size_t)2);     /* "a", and the end: the star is always complete */
        CHECK_EQ(ids[0], 0);
        CHECK_EQ(ids[1], 5);
        CHECK_OK(g->accept_token(0));
    }
}

TEST(gbnf_a_closure_whose_paths_multiply_is_refused) {
    /* No left recursion anywhere, but each level offers two continuations, so the start state
     * alone is 2^40 distinct stacks. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    std::string text = "root ::= r0\n";
    for (int i = 0; i < 40; ++i)
        text += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + " \"b\" | r" +
                std::to_string(i + 1) + " \"c\"\n";
    text += "r40 ::= \"a\"\n";
    CHECK_EQ(gbnf_create(text, v, &g, &err), RAD_E_UNSUPPORTED);
    CHECK(err.find("bounds") != std::string::npos);
}

TEST(gbnf_a_parse_that_doubles_per_token_fails_its_request) {
    /* Every "a" is ambiguous between two continuations, so the stack set doubles per token. The
     * grammar is legal but the request cannot continue past a point; it must end as a status on
     * the request, with the grammar left dead, not by exhausting the scheduler thread's memory. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_OK(gbnf_create("root ::= \"a\" root \"b\" | \"a\" root \"c\" | \"d\"\n", v, &g, &err));
    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<uint32_t> mask((size_t)n_words);
    int failed_at = -1, status = RAD_OK;
    for (int i = 0; i < 40 && failed_at < 0; ++i) {
        status = g->fill_mask(mask.data(), n_words);
        if (status >= 0) status = g->accept_token(0);
        if (status < 0) failed_at = i;
    }
    CHECK(failed_at > 8);                    /* it ran for a while first */
    CHECK_EQ(status, RAD_E_FULL);
    CHECK(g->fill_mask(mask.data(), n_words) < 0);
}

TEST(gbnf_stacks_that_converge_are_expanded_once) {
    /* A star over 16,000 alternatives that all match "a". After each "a" every one returns to the
     * same continuation, which re-expands into all 16,000; expanding per returning stack would be
     * 256 million stacks per token. It must be cheap, not refused. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    std::string text = "root ::= t*\nt ::= \"a\"";
    for (int i = 1; i < 16000; ++i) text += " | \"a\"";
    text += "\n";
    CHECK_OK(gbnf_create(text, v, &g, &err));
    for (int i = 0; i < 3; ++i) {
        const auto ids = admitted(*g, v.n_tokens());
        CHECK_EQ(ids.size(), (size_t)2);     /* "a", and the end */
        CHECK_OK(g->accept_token(0));
    }
}

/* A schema nested thousands of levels deep is a few tens of kilobytes of request. The converter
 * recurses once per level and its output grows with the square of the depth, exhausting the
 * admitting thread's memory or stack, so the schema is refused on depth first. */
TEST(json_schema_refuses_a_schema_nested_past_its_limit) {
    auto nested = [](int depth) {
        std::string s;
        for (int i = 0; i < depth; ++i) s += R"({"type":"array","items":)";
        s += R"({"type":"integer"})";
        for (int i = 0; i < depth; ++i) s += "}";
        return s;
    };
    std::string gbnf, err;
    CHECK_EQ(grammar_from_json_schema(nested(20000), "root", &gbnf, &err), RAD_E_INVAL);
    CHECK(err.find("levels deep") != std::string::npos);
    err.clear();
    CHECK_OK(grammar_from_json_schema(nested(100), "root", &gbnf, &err));
}

/* The bounded machinery against upstream's walk, over the shapes the bounds were built for: a star
 * over something nullable, a wide converging rule (past the 64 stacks a firing pattern is
 * memoised for and the 16 past which a set is indexed), an ambiguous rule whose stack set grows,
 * nesting, and a char class under a repetition. Shared exploration, joint advancing and merged
 * successors must each give upstream's set. Shapes stay small because the oracle re-expands a
 * converging state once per stack per code point. */
TEST(gbnf_bounded_shapes_agree_with_the_oracle) {
    SimpleVocab v;
    for (const char* p : { "a", "b", "c", "d", "x", "ab", "ba", "aa", "bx", "(", ")", "()",
                           "((", "))", "a)", "(a", " ", "  ", "a ", "0", "1", "01", "10" })
        v.add(p);
    const int32_t eog = v.add("", true);

    std::string wide = "root ::= t* \"x\"\nt ::= \"a\"";
    for (int i = 1; i < 66; ++i)
        wide += (i % 3 == 0) ? " | \"ab\"" : (i % 3 == 1) ? " | \"a\"" : " | [ab]";
    wide += "\n";
    const std::string grammars[] = {
        "root ::= (\"a\"?)* \"x\"\n",
        wide,
        "root ::= \"a\" root \"b\" | \"a\" root \"c\" | \"d\"\n",
        "root ::= \"(\" root \")\" root | \"a\" | \" \"?\n",
        "root ::= ([01]{1,4} \" \"?)+ \"x\"\n",
        "root ::= a b | a c\na ::= \"a\"*\nb ::= \"b\" a | \"x\"\nc ::= (\"c\" | \"a\")+\n",
    };

    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<uint32_t> fast((size_t)n_words), ref((size_t)n_words);
    int64_t masks = 0;
    for (const std::string& text : grammars) {
        for (uint64_t seed = 1; seed <= 3; ++seed) {
            /* `h` takes the same path through accept_str, which advances the whole stack set one
             * code point at a time as upstream does, so matching masks hold accept_token's path
             * to the same answer. */
            std::unique_ptr<GbnfGrammar> g, h;
            std::string err;
            CHECK_OK(gbnf_create(text, v, &g, &err));
            CHECK_OK(gbnf_create(text, v, &h, &err));
            std::vector<uint32_t> twin((size_t)n_words);
            uint64_t rng = seed * 0x9e3779b97f4a7c15ull;
            for (int step = 0; step < 8; ++step) {
                const int fs = g->fill_mask(fast.data(), n_words);
                const int rs = gbnf_oracle_mask(*g, ref.data(), n_words);
                CHECK_EQ(fs, rs);
                CHECK_EQ(h->fill_mask(twin.data(), n_words), fs);
                if (fs < 0) break;
                ++masks;
                CHECK_EQ(std::memcmp(fast.data(), ref.data(), (size_t)n_words * 4), 0);
                CHECK_EQ(std::memcmp(fast.data(), twin.data(), (size_t)n_words * 4), 0);
                const std::vector<int32_t> ids = set_ids(fast, v.n_tokens(), eog);
                if (ids.empty()) break;
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                const int32_t pick = ids[(size_t)(rng % ids.size())];
                if (g->accept_token(pick) < 0) break;
                CHECK_OK(h->accept_str(v.token_piece(pick)));
            }
        }
    }
    CHECK(masks > 100);
}

/* ================================================================== the backend interface */

TEST(grammar_create_builds_a_constraining_backend) {
    SimpleVocab v = tiny_vocab();

    GrammarSpec spec;
    spec.text = "root ::= \"c\"\n";
    std::unique_ptr<GrammarBackend> b;
    std::string err;
    CHECK_OK(grammar_create(spec, v, &b, &err));
    CHECK(b->constrains());
}

TEST(json_schema_compiles_to_a_usable_grammar) {
    std::string gbnf, err;
    const int st = grammar_from_json_schema(
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},"
        "\"required\":[\"a\"]}", "root", &gbnf, &err);
    CHECK_OK(st);
    CHECK(gbnf.find("root ::=") != std::string::npos);

    /* The compiled grammar must parse; a converter whose GBNF does not load would go unnoticed
     * until a request arrived. */
    SimpleVocab v;
    v.add("{"); v.add("\""); v.add("a"); v.add("}"); v.add(" "); v.add("", true);
    GbnfParser p;
    std::string perr;
    CHECK_OK(p.parse(gbnf.c_str(), &perr));
}

TEST(json_schema_rejects_malformed_input) {
    std::string gbnf, err;
    CHECK(grammar_from_json_schema("{\"type\":", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());
}

/* The refusals vendor/patches/0004 adds, and the hang 0003 closes. Upstream accepts each input and
 * then drops the constraint, reaches the network, emits an unloadable grammar, or spins. `pattern`
 * and `$ref` are client data on both completion endpoints, so each is a request shape. */
TEST(json_schema_refuses_what_it_cannot_actually_enforce) {
    std::string gbnf, err;

    /* "not" has no case in the converter, so falling through would accept any value: a
     * constraint the caller asked for and silently did not get. */
    CHECK(grammar_from_json_schema("{\"not\":{\"type\":\"string\"}}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* A sampler that reaches the network to decide what tokens are legal is not one we ship. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"$ref\":\"https://example.com/s.json#/x\"}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* GBNF cannot spell \b. Copying it through would succeed here and produce a grammar that
     * fails to load, an internal error on a schema we accepted. (\d and its kin are written out
     * as classes since vendor patch 0018.) */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"^a\\\\bb$\"}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* A stray ']' is in NON_LITERAL_SET but claimed by no arm of the pattern scanner, so the
     * literal arm breaks without consuming it and the loop reconsiders the same byte forever. A
     * regression shows as a hung test, the same hang a server thread would take on the request. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"^a]$\"}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* An empty pattern would run front()/back() on an empty string. Unanchored, as JSON Schema
     * reads it, it now matches every string and never reaches them (vendor patch 0017). */
    err.clear();
    CHECK_OK(grammar_from_json_schema(
                 "{\"type\":\"string\",\"pattern\":\"\"}", "root", &gbnf, &err));

    /* A leading quantifier would run back() on an empty vector. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"^*$\"}", "root", &gbnf, &err) < 0);

    /* The ordinary case still compiles, so the above is not refusing everything. */
    err.clear();
    CHECK_OK(grammar_from_json_schema(
                 "{\"type\":\"string\",\"pattern\":\"^[a-z]+$\"}", "root", &gbnf, &err));
    CHECK(gbnf.find("root ::=") != std::string::npos);
}

/* ================================================================== speculative acceptance */

TEST(greedy_acceptance_stops_at_the_first_disagreement) {
    /* draft   5 7 9
     * The first two agree and the third does not, so two are accepted and the emitted run is the
     * two accepted tokens plus the target's own answer at the rejected position. */
    const int32_t draft[]  = { 5, 7, 9 };
    const int32_t target[] = { 5, 7, 3, 11 };
    int32_t out[4] = { -1, -1, -1, -1 };
    int32_t n = -1;

    CHECK_OK(accept_greedy(draft, 3, target, out, &n));
    CHECK_EQ(n, 2);
    CHECK_EQ(out[0], 5);
    CHECK_EQ(out[1], 7);
    CHECK_EQ(out[2], 3);
}

TEST(greedy_acceptance_takes_the_bonus_when_everything_agrees) {
    const int32_t draft[]  = { 5, 7 };
    const int32_t target[] = { 5, 7, 11 };
    int32_t out[3] = { -1, -1, -1 };
    int32_t n = -1;

    CHECK_OK(accept_greedy(draft, 2, target, out, &n));
    CHECK_EQ(n, 2);
    CHECK_EQ(out[2], 11);      /* the free extra token speculation buys */
}

TEST(greedy_acceptance_with_no_draft_is_an_ordinary_step) {
    const int32_t target[] = { 42 };
    int32_t out[1] = { -1 };
    int32_t n = -1;
    CHECK_OK(accept_greedy(nullptr, 0, target, out, &n));
    CHECK_EQ(n, 0);
    CHECK_EQ(out[0], 42);
}

/* ================================================================== the sampler's front door
 *
 * Admission and per-request state, all host work on the scheduler thread: check_request fails a bad
 * request at admission, and begin_request/end_request bound a grammar's life to its request.
 * These tests call the class without a builder or a card. An undeclared Sampler resolves no
 * kernels, so only check_request's parameter-range half (which runs first) is reachable.
 */
namespace {

/* Everything a request needs and nothing a kernel does. */
SamplingParams plain_params() {
    SamplingParams sp;
    sp.temp = 0.8f;
    sp.top_k = 40;
    return sp;
}

}  /* namespace */

TEST(admission_refuses_a_sampling_parameter_outside_its_range) {
    Sampler s;
    std::string why;

    SamplingParams sp = plain_params();
    sp.top_k = -1;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("top_k") != std::string::npos);

    sp = plain_params(); sp.top_p = 1.5f;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("top_p") != std::string::npos);

    sp = plain_params(); sp.top_p = -0.1f;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);

    sp = plain_params(); sp.min_p = 2.0f;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("min_p") != std::string::npos);

    sp = plain_params(); sp.typical_p = -1.0f;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("typical_p") != std::string::npos);

    sp = plain_params(); sp.temp = -0.5f;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("temperature") != std::string::npos);
}

/* The candidate width bounds every stage after top-k, so a request asking for more candidates
 * than the buffers hold is refused by name at admission, not silently narrowed: a top_k that
 * quietly became something else changes the output with nothing to say so. */
TEST(admission_refuses_a_top_k_wider_than_the_candidate_buffers) {
    Sampler s;
    std::string why;
    SamplingParams sp = plain_params();
    sp.top_k = 1 << 20;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("candidate width of 1024") != std::string::npos);
    CHECK(why.find("1048576") != std::string::npos);
    sp.top_k = (int)kSampleCandidates;
    why.clear();
    CHECK(s.check_request(sp, &why) != RAD_E_INVAL);
}

/* Values the chain cannot honour as a client means them are refused rather than reinterpreted:
 * top_p and typical_p 0 (off in the chain, one candidate in llama.cpp), a repetition penalty of 0
 * (a division by zero), a DRY allowed length below 1 (read as 2), and NaN anywhere, which an
 * "outside the range" comparison lets through. */
TEST(admission_refuses_values_the_chain_would_reinterpret) {
    Sampler s;
    std::string why;
    auto refused = [&](const SamplingParams& sp, const char* field) {
        why.clear();
        return s.check_request(sp, &why) == RAD_E_INVAL && why.find(field) != std::string::npos;
    };
    SamplingParams sp = plain_params(); sp.top_p = 0.0f;
    CHECK(refused(sp, "top_p"));
    sp = plain_params(); sp.typical_p = 0.0f;
    CHECK(refused(sp, "typical_p"));
    sp = plain_params(); sp.top_p = NAN;
    CHECK(refused(sp, "top_p"));
    sp = plain_params(); sp.temp = NAN;
    CHECK(refused(sp, "temperature"));
    sp = plain_params(); sp.min_p = NAN;
    CHECK(refused(sp, "min_p"));
    sp = plain_params(); sp.xtc_threshold = 1.5f;
    CHECK(refused(sp, "xtc_threshold"));
    sp = plain_params(); sp.xtc_probability = -0.5f;
    CHECK(refused(sp, "xtc_probability"));
    sp = plain_params(); sp.rep_penalty = 0.0f;
    CHECK(refused(sp, "rep_penalty"));
    sp = plain_params(); sp.dry_multiplier = 0.8f; sp.dry_allowed_length = 0;
    CHECK(refused(sp, "dry_allowed_length"));

    /* A stage that is off never reads its value, so the value is no reason to refuse. (This
     * undeclared sampler still refuses, for having no kernels.) */
    sp = plain_params(); sp.dry_multiplier = 0.0f; sp.dry_allowed_length = 0;
    why.clear();
    CHECK(s.check_request(sp, &why) != RAD_E_INVAL);
    sp = plain_params(); sp.penalty_last_n = 0; sp.rep_penalty = 0.0f;
    why.clear();
    CHECK(s.check_request(sp, &why) != RAD_E_INVAL);
}

/* An unresolved sampler kernel is a named refusal, not a silent skip, for the same reason: a
 * request whose min_p stopped being applied has different output. An undeclared Sampler has
 * resolved nothing, so every request needing a stage is refused and the message names it. */
TEST(admission_names_the_sampler_that_has_no_kernel) {
    Sampler s;
    std::string why;

    SamplingParams sp = plain_params();
    sp.top_p = 0.9f;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_NOKERNEL);
    CHECK(why.find("sample_") != std::string::npos);

    sp = plain_params();
    sp.temp = 0.0f;                       /* greedy: the argmax, not the chain */
    CHECK_EQ(s.check_request(sp, &why), RAD_E_NOKERNEL);
    CHECK(why.find("sample_argmax") != std::string::npos);
}

/* A grammar needs the vocabulary it walks, which is attached after the tokeniser loads. Asked
 * for one before that, the answer is a named state error, not a null dereference on the step
 * path. */
TEST(a_grammar_before_the_vocabulary_is_a_named_refusal) {
    Sampler s;
    std::string err;
    SamplingParams sp = plain_params();
    sp.grammar = "root ::= \"a\"";
    /* Admission reaches the vocabulary check only once every stage has a kernel, which an
     * undeclared sampler lacks, so the refusal here is begin_request's, the call that would
     * dereference it. */
    CHECK_EQ(s.begin_request(1, sp, &err), RAD_E_STATE);
    CHECK(!err.empty());
    CHECK(!s.has_request(1));
}

TEST(dry_sequence_breakers_need_a_vocabulary_to_resolve_against) {
    Sampler s;
    std::string err;
    SamplingParams sp = plain_params();
    sp.dry_multiplier = 0.8f;
    sp.dry_seq_breakers = { "\n" };
    CHECK_EQ(s.begin_request(2, sp, &err), RAD_E_STATE);
    CHECK(!s.has_request(2));
}

/* Request state lives as long as the request. begin_request is not idempotent: calling it twice
 * rebuilds the automaton and discards everything accepted, unconstraining the generation from its
 * second token. So has_request, which the step loop asks first, must be right both ways. */
TEST(request_state_begins_and_ends_with_the_request) {
    Sampler s;
    SimpleVocab v = tiny_vocab();
    s.attach_vocab(&v);
    std::string err;

    CHECK(!s.has_request(7));
    CHECK_OK(s.begin_request(7, plain_params(), &err));
    CHECK(s.has_request(7));
    /* A second request is its own state and does not disturb the first. */
    CHECK_OK(s.begin_request(8, plain_params(), &err));
    CHECK(s.has_request(7));
    CHECK(s.has_request(8));

    s.end_request(7);
    CHECK(!s.has_request(7));
    CHECK(s.has_request(8));
    /* Ending a request that is not there is not an error: the step loop reaches a request on
     * every step it is scheduled in, and cancellation can get there first. */
    s.end_request(7);
    s.end_request(999);
    CHECK(s.has_request(8));
}

/* A request with no grammar has nothing to advance; accept_token must succeed rather than fail,
 * since it runs on every emitted token of every request. */
TEST(accepting_a_token_against_no_grammar_is_not_an_error) {
    Sampler s;
    SimpleVocab v = tiny_vocab();
    s.attach_vocab(&v);
    std::string err;
    CHECK_OK(s.begin_request(3, plain_params(), &err));
    CHECK_OK(s.accept_token(3, 0));
    /* A request this sampler has never seen gets the same answer, for the same reason. */
    CHECK_OK(s.accept_token(404, 0));
}

/* Speculation is allowed unless a grammar cannot rewind: masking draft position i walks the
 * grammar over 0..i-1 and rewinds, so a non-rewinding backend decodes one token at a time. An
 * unconstrained request must not be slowed that way. */
TEST(speculation_is_allowed_for_a_request_with_no_grammar) {
    Sampler s;
    SimpleVocab v = tiny_vocab();
    s.attach_vocab(&v);
    std::string err;
    CHECK_OK(s.begin_request(4, plain_params(), &err));
    CHECK(s.allows_speculation(4));
    /* The same for a request with no sampler state at all, as one that never needed any has. */
    CHECK(s.allows_speculation(405));
}

/* A grammar request gets state, advances, and still speculates. The GBNF backend snapshots, so
 * a constrained request is not silently demoted to one token per step, which would cost several
 * times the per-token cost of the mask. */
TEST(a_grammar_request_carries_state_that_advances_and_can_rewind) {
    Sampler s;
    SimpleVocab v = tiny_vocab();
    s.attach_vocab(&v);
    std::string err;

    SamplingParams sp = plain_params();
    sp.grammar = "root ::= \"ab\"";
    CHECK_OK(s.begin_request(5, sp, &err));
    CHECK(s.has_request(5));
    CHECK(s.allows_speculation(5));
    CHECK_OK(s.accept_token(5, 3));        /* the "ab" piece */

    s.end_request(5);
    CHECK(!s.has_request(5));
}

/* Requests build_step could not serve are handed back once and cleared, because the caller
 * cancels them; reporting the same failure next step would cancel a request that is gone. */
TEST(the_failed_list_is_taken_once_and_is_empty_on_a_quiet_step) {
    Sampler s;
    CHECK(s.take_failed().empty());
    CHECK(s.take_failed().empty());
}

/* ================================================================== the RNG is the device's */

/* The host reference draws abi/rad_sample.h's uniform bit for bit: the pick on the seed itself and
 * XTC's coin on its key, exactly what sample_pick and sample_xtc draw. A reference with its own
 * generator would agree with the device on no seeded draw. */
TEST(host_rng_is_the_specified_draw) {
    for (uint64_t seed : { 0ull, 1ull, 12345ull, 0xFFFFFFFFFFFFFFFFull }) {
        for (uint64_t pos = 0; pos < 64; ++pos) {
            CHECK_EQ(rng_uniform(seed, pos, RAD_RNG_PICK), rad_sample_uniform01(seed, pos));
            CHECK_EQ(rng_uniform(seed, pos, RAD_RNG_XTC),
                     rad_sample_uniform01(seed ^ RAD_SAMPLE_KEY_XTC, pos));
        }
    }
}

/* ================================================================== DRY's breakers, prepared */

/* The scan is bounded as upstream bounds it, plus a count: a breaker longer than 40 bytes is cut
 * (at a character boundary), a tail longer than 20 tokens is cut, and more than kDryMaxBreakers
 * breakers are refused, since each is a pass over the whole vocabulary on the scheduler thread
 * while every other sequence waits. */
TEST(dry_breakers_are_bounded_the_way_upstream_bounds_them) {
    SimpleVocab v;
    for (char c = 'a'; c <= 'z'; ++c) v.add(std::string(1, c));   /* 0..25 */

    /* 50 bytes. The token "x" covers the breaker's first byte and the rest tokenises letter by
     * letter: 39 tokens after the 40-byte cut, 20 after the tail cut. */
    std::string long_one = "x";
    for (int i = 0; i < 49; ++i) long_one.push_back((char)('a' + i % 26));
    DryBreakers b;
    CHECK_OK(dry_prepare_breakers(v, { long_one }, &b));
    size_t longest = 0;
    for (const auto& e : b.seqs) longest = std::max(longest, e.second.size());
    CHECK_EQ(longest, (size_t)kDryBreakerMaxTail);
    CHECK(b.is_head('x' - 'a'));
    CHECK(!b.is_single('x' - 'a'));

    /* A two-byte character straddling the 40-byte cut is dropped whole, not split: the remaining
     * prefix is 39 bytes of ASCII, which still tokenises. */
    std::string straddle(39, 'a');
    straddle += "\xC3\xA9";                                         /* U+00E9, bytes 40 and 41 */
    DryBreakers s;
    CHECK_OK(dry_prepare_breakers(v, { straddle }, &s));

    std::vector<std::string> many(kDryMaxBreakers + 1, "q");
    DryBreakers m;
    CHECK_EQ(dry_prepare_breakers(v, many, &m), RAD_E_INVAL);
    CHECK(m.empty());
}

TEST(admission_refuses_more_dry_breakers_than_the_scan_takes) {
    Sampler s;
    std::string why;
    SamplingParams sp = plain_params();
    sp.dry_multiplier = 0.8f;
    sp.dry_seq_breakers.assign(kDryMaxBreakers + 1, "\n");
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("dry_sequence_breakers") != std::string::npos);

    /* With DRY off the list is never scanned, so it is no reason to refuse. (This unconfigured
     * sampler still refuses, for having no kernels.) */
    sp.dry_multiplier = 0.0f;
    why.clear();
    CHECK(s.check_request(sp, &why) != RAD_E_INVAL);
    CHECK(why.find("dry_sequence_breakers") == std::string::npos);
}

/* ================================================================== the chain, end to end
 *
 * The real sampler, declared through the real builder and run by libref on the host backend.
 * Each op can match its schema while the driver stages or issues the wrong thing (a row-relative
 * history offset, a top-k never issued, a pick overwriting a greedy argmax, DRY never on); only
 * staging and issue order together decide the token. Every op is moved to the host domain after
 * the context is built, and device buffers are host memory, so logits and tokens are read
 * directly. */
namespace {

struct HostChain {
    Registry reg;
    Program  prog;
    Ctx      ctx;
    Sampler  s;
    rad_buf  logits = RAD_NULL_HANDLE, tokens = RAD_NULL_HANDLE;
    int64_t  nv = 0;
    bool     ok = false;
    int      st = RAD_OK;

    static inline HostChain* cur = nullptr;
    static void trampoline(RadCtx* c, const RadBatch*) {
        if (cur) cur->st = cur->s.run(c, cur->logits, cur->tokens);
    }

    HostChain(int64_t n_vocab, int64_t max_seqs, int max_spec, const VocabView* vocab = nullptr,
              int64_t max_candidates = 8)
        : nv(n_vocab) {
        /* libref alone: it is the implementation under test, and a directory scan would also
         * pick up whatever else was built beside it. */
        if (reg.load_plugin(home() + "/kernels/libref.so", 0) < 0) return;
        RadModelMeta meta{};
        meta.arch_id = "sampler_chain";
        meta.name    = "sampler_chain";
        meta.n_vocab = n_vocab;
        RadBuildCtx bc{};
        bc.rank = 0;
        bc.world_size = 1;
        bc.max_seqs = max_seqs;
        bc.max_tok = max_seqs * (max_spec + 1);
        bc.max_ctx = 64;
        bc.max_spec = max_spec;
        bc.scope = "";
        Builder b(reg, meta, bc);

        const int64_t rows = max_seqs * (max_spec + 1);
        RadBufDecl d{};
        d.dtype = RAD_F32;
        d.rank = 2;
        d.shape[0] = rows;
        d.shape[1] = n_vocab;
        d.kind = RAD_BUF_PERSIST;
        d.domain = RAD_DOMAIN_DEVICE;
        logits = rad_decl_buffer(&b, "logits", &d);
        d.dtype = RAD_I32;
        d.shape[1] = 1;
        tokens = rad_decl_buffer(&b, "tokens", &d);

        SamplerConfig cfg;
        cfg.n_vocab = n_vocab;
        cfg.n_vocab_local = n_vocab;
        cfg.max_seqs = max_seqs;
        cfg.max_spec = max_spec;
        cfg.max_hist = 64;
        cfg.max_candidates = max_candidates;
        if (s.declare(&b, cfg) < 0) return;
        if (b.finish() < 0) return;
        prog = std::move(b.program());
        if (vocab) s.attach_vocab(vocab);

        CtxDesc cd;
        cd.program = &prog;
        cd.arch_step = &HostChain::trampoline;
        /* The pass body is the sampler chain, whose host ops carry each request's sampling state,
         * so two steps with one batch shape are not one pass. */
        cd.record_passes = false;
        if (ctx.init(cd) < 0) return;
        for (size_t i = 1; i < prog.ops.size(); ++i)
            if (ctx.set_op_domain((rad_op)i, RAD_DOMAIN_HOST) < 0) return;
        ok = true;
    }

    /* One step: begin every request not yet seen, stage the step, write `lg` (n_vocab floats per
     * sampled position, in staging order) into the logits plane, run the chain and read the
     * tokens back. */
    int step(const std::vector<Request*>& reqs, const std::vector<float>& lg,
             std::vector<int32_t>* out) {
        for (Request* r : reqs) {
            if (s.has_request(r->id)) continue;
            std::string err;
            const int b = s.begin_request(r->id, r->sp, &err);
            if (b < 0) return b;
        }
        const int bs = s.build_step(reqs);
        if (bs < 0) return bs;
        int64_t n = 0;
        s.staged_params(&n);
        if ((int64_t)lg.size() != n * nv) return RAD_E_SHAPE;
        std::memcpy(ctx.buf_ptr(logits), lg.data(), lg.size() * sizeof(float));
        int32_t* t = (int32_t*)ctx.buf_ptr(tokens);
        for (int64_t i = 0; i < n; ++i) t[i] = -7;

        RadBatch batch{};
        batch.phase = RAD_PHASE_DECODE;
        batch.n_tok = n;
        batch.n_seq = n;
        cur = this;
        st = RAD_OK;
        const int rs = ctx.run_step(&batch);
        cur = nullptr;
        if (rs < 0) return rs;
        if (st < 0) return st;
        rad_stream_sync(ctx.stream());
        out->assign(t, t + n);
        return RAD_OK;
    }
};

#define NEED_CHAIN(ch)                                                                \
    if (!(ch).ok) {                                                                   \
        fprintf(stderr, "  SKIP %s: libref did not load from %s/kernels\n",           \
                ::radtest::current(), home().c_str());                          \
        return;                                                                       \
    }

Request make_req(uint64_t id, std::vector<int32_t> prompt, const SamplingParams& sp) {
    Request r;
    r.id = id;
    r.prompt = std::move(prompt);
    r.sp = sp;
    return r;
}

SamplingParams greedy_params() {
    SamplingParams sp;
    sp.temp = 0.0f;
    return sp;
}

double log_sum_exp(const std::vector<float>& lg) {
    double z = 0.0;
    for (float v : lg) z += std::exp((double)v);
    return std::log(z);
}

}  /* namespace */

/* A second sampled row is penalised too. Two greedy sequences whose histories both hold token 0,
 * with a repetition penalty that takes token 0 below token 1: each row must pick 1. Each row's
 * history offset is within its own row, so row 1 reads row 1. */
TEST(every_sampled_row_is_penalised_against_its_own_history) {
    HostChain ch(4, 2, 0);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.rep_penalty = 10.0f;
    Request a = make_req(1, { 0, 2 }, sp);
    Request b = make_req(2, { 3, 0 }, sp);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a, &b }, { 3.0f, 2.9f, -5.0f, -5.0f,  3.0f, 2.9f, -5.0f, -5.0f }, &tok));
    CHECK_EQ(tok[0], 1);
    CHECK_EQ(tok[1], 1);

    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    CHECK_EQ(n, (int64_t)2);
    CHECK_EQ(p[1].hist_off, 0);
    CHECK_EQ(p[1].hist_len, 2);
}

/* A draft position is penalised against the drafts before it. The token at draft position k is
 * emitted only if drafts 0..k-1 are, so they are part of the output it is judged against, as one
 * token at a time would be. */
TEST(a_draft_position_is_penalised_against_the_drafts_before_it) {
    HostChain ch(4, 1, 1);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.rep_penalty = 10.0f;
    Request a = make_req(1, { 0 }, sp);
    const int32_t draft[1] = { 1 };
    a.n_draft = 1;
    a.draft = draft;
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, { 2.0f, 3.0f, 0.0f, 0.0f,  0.0f, 3.0f, 2.0f, 0.0f }, &tok));
    CHECK_EQ(tok[0], 1);                  /* history {0}: token 1 stands */
    CHECK_EQ(tok[1], 2);                  /* history {0, 1}: token 1 is penalised below 2 */

    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    CHECK_EQ(p[0].hist_len, 1);
    CHECK_EQ(p[1].hist_len, 2);
}

/* A greedy row keeps its argmax beside a sampled row. The batch mixes both paths, so the argmax
 * and the candidate chain each run over every row; the greedy row's logits are nearly flat, so a
 * draw would land on its argmax only about a quarter of the time. */
TEST(a_greedy_row_keeps_its_argmax_beside_a_sampled_row) {
    HostChain ch(4, 2, 0);
    NEED_CHAIN(ch)
    SamplingParams sampled;
    sampled.temp = 1.0f;
    sampled.top_k = 2;
    for (uint64_t seed = 1; seed <= 16; ++seed) {
        SamplingParams g = greedy_params();
        g.seed = seed;
        sampled.seed = seed;
        Request a = make_req(10 + seed, { 1 }, g);
        Request b = make_req(100 + seed, { 1 }, sampled);
        std::vector<int32_t> tok;
        CHECK_OK(ch.step({ &a, &b }, { 1.0f, 1.1f, 1.0f, 1.0f,  -80.0f, -80.0f, 0.0f, -80.0f },
                         &tok));
        CHECK_EQ(tok[0], 1);
        CHECK_EQ(tok[1], 2);
        ch.s.end_request(a.id);
        ch.s.end_request(b.id);
    }
}

/* top_k = 0 means the whole candidate width, not no candidates. The candidate planes are
 * transient, so a step that skipped top-k would hand the pick whatever they last held. Two steps
 * with different winners: each must sample from its own logits. */
TEST(a_top_k_of_zero_samples_this_steps_candidates) {
    HostChain ch(4, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 0;
    sp.seed = 5;
    Request a = make_req(1, { 0 }, sp);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, { -80.0f, -80.0f, 0.0f, -80.0f }, &tok));
    CHECK_EQ(tok[0], 2);
    CHECK_OK(ch.step({ &a }, { -80.0f, -80.0f, -80.0f, 0.0f }, &tok));
    CHECK_EQ(tok[0], 3);
}

/* Logprobs are the raw distribution's, read before any stage. A greedy row asking for them is
 * staged as a top-1 chain row, and its token and alternatives come back against the normaliser of
 * the logits the model wrote. */
TEST(logprobs_are_the_raw_distributions) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.n_probs = 3;
    Request a = make_req(1, { 30 }, sp);
    std::vector<float> lg(32, 0.0f);
    lg[7] = 4.0f; lg[3] = 4.0f; lg[9] = 2.0f;
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 3);                           /* the tie goes to the lower id */

    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    CHECK((p[0].flags & RAD_SP_LOGPROBS) != 0);
    CHECK((p[0].flags & RAD_SP_GREEDY) == 0);
    CHECK_EQ(p[0].top_k, 1);

    const double lz = log_sum_exp(lg);
    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, 4.0 - lz, 1e-5);
    CHECK_EQ(lp.n_top, 3);
    CHECK_EQ(lp.top_id[0], 3);
    CHECK_EQ(lp.top_id[1], 7);
    CHECK_EQ(lp.top_id[2], 9);
    CHECK_NEAR(lp.top_lp[2], 2.0 - lz, 1e-5);
    /* A row that was not staged for logprobs has none, and asking is the caller's error. */
    CHECK_EQ(ch.s.logprobs_for(1, 0, &lp), RAD_E_STATE);
}

namespace {

/* The summary holds 21 pairs (RAD_MAX_LOGPROBS + 1), so a token is outside it only behind 21
 * larger or tied ones: tokens 0..20 at `hi` fill it, and token 21 at `at` is the first outside (at
 * `hi` too, it loses the tie to every lower id). The rest of a 32-token vocabulary sits at
 * `rest`. The prompt is `head`, 0..20 twice, then `tail`, so the penalties count each of 0..20
 * twice. */
std::vector<float> outside_logits(float hi, float at, float rest) {
    std::vector<float> lg(32, rest);
    for (int t = 0; t <= 20; ++t) lg[(size_t)t] = hi;
    lg[21] = at;
    return lg;
}
std::vector<int32_t> outside_prompt(std::vector<int32_t> head, const std::vector<int32_t>& tail) {
    for (int k = 0; k < 2; ++k)
        for (int32_t t = 0; t <= 20; ++t) head.push_back(t);
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
}

}  /* namespace */

/* A token outside the raw summary is recovered from the value it was drawn at. Tokens 0..21 tie at
 * 5, so the summary holds 0..20; the penalties push those to 5/1.25 - 2 = 2 and token 21 to
 * 5/1.25 - 1 = 3, which is chosen, and its raw logit is the candidate value with temperature,
 * frequency and repetition undone. */
TEST(a_chosen_token_outside_the_summary_has_its_raw_logprob) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.top_k = 1;
    sp.temp = 0.5f;
    sp.rep_penalty = 1.25f;
    sp.freq_penalty = 1.0f;
    sp.n_probs = 0;
    Request a = make_req(1, outside_prompt({}, { 21 }), sp);
    const std::vector<float> lg = outside_logits(5.0f, 5.0f, 0.0f);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 21);

    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, 5.0 - log_sum_exp(lg), 1e-5);
    CHECK_EQ(lp.n_top, 0);
}

/* A negative logit's repetition penalty is undone on its own side of zero, and the presence
 * penalty is added back with the frequency one. Token 21's raw logit is -1: repetition makes it
 * -1.25, frequency (once) and presence take it to -13.25, and tokens 0..20 (5/1.25 - 20 - 2 = -18)
 * and the rest (-30) fall below it. Inverting: -13.25 + 10 + 2 = -1.25, divided by 1.25 is -1. */
TEST(a_negative_penalised_logit_has_its_raw_logprob) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.top_k = 1;
    sp.temp = 0.5f;
    sp.rep_penalty = 1.25f;
    sp.freq_penalty = 10.0f;
    sp.pres_penalty = 2.0f;
    sp.n_probs = 2;
    Request a = make_req(1, outside_prompt({}, { 21 }), sp);
    const std::vector<float> lg = outside_logits(5.0f, -1.0f, -30.0f);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 21);

    const double lz = log_sum_exp(lg);
    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, -1.0 - lz, 1e-5);
    CHECK_EQ(lp.n_top, 2);
    CHECK_EQ(lp.top_id[0], 0);
    CHECK_NEAR(lp.top_lp[0], 5.0 - lz, 1e-5);
}

/* Only the penalty window is counted back. Token 21 opens the history three times, but the last 42
 * tokens (penalty_last_n) are 0..20 twice, so the stage never penalised it: 0..20 drop to 2 and 21
 * stays at 5. Counting the whole staged history would add 3 back and multiply by 1.25. */
TEST(a_logprob_counts_only_the_penalty_window_back) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.top_k = 1;
    sp.temp = 0.5f;
    sp.rep_penalty = 1.25f;
    sp.freq_penalty = 1.0f;
    sp.penalty_last_n = 42;
    sp.n_probs = 0;
    Request a = make_req(1, outside_prompt({ 21, 21, 21 }, {}), sp);
    const std::vector<float> lg = outside_logits(5.0f, 5.0f, 0.0f);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 21);
    int64_t n = 0;
    CHECK_EQ(ch.s.staged_params(&n)[0].hist_len, 45);

    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, 5.0 - log_sum_exp(lg), 1e-5);
}

/* Temperature last is undone from the late stage's row. A sampled row (top_k 2, since top_k 1 is
 * greedy and has no late stage), so the early row runs at 1 and the late one at 0.5 over the two
 * candidates: token 21 at (5/1.25 - 20) / 0.5 = -32 against -72 for the runner-up, a draw that
 * cannot go the other way. Only the late temperature takes -32 back to -16 before the penalties
 * are undone. */
TEST(a_temperature_last_row_has_its_raw_logprob) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.top_k = 2;
    sp.temp = 0.5f;
    sp.temp_last = true;
    sp.rep_penalty = 1.25f;
    sp.freq_penalty = 20.0f;
    sp.n_probs = 0;
    Request a = make_req(1, outside_prompt({}, { 21 }), sp);
    const std::vector<float> lg = outside_logits(5.0f, 5.0f, -100.0f);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 21);
    int64_t n = 0;
    CHECK_EQ(ch.s.staged_params(&n)[0].temp, 1.0f);

    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, 5.0 - log_sum_exp(lg), 1e-5);
}

/* A row with no finite logit has no logprobs and says so, rather than reporting -inf or NaN: the
 * step still runs, and the engine fails the one request (Engine::row_logprobs). */
TEST(a_row_with_no_finite_logit_has_no_logprobs) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.n_probs = 3;
    Request a = make_req(1, { 1 }, sp);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, std::vector<float>(32, -INFINITY), &tok));
    TokenLogprobs lp;
    lp.token = 7;
    CHECK_EQ(ch.s.logprobs_for(0, tok[0], &lp), RAD_E_INVAL);
    CHECK_EQ(lp.token, -1);
}

/* The ranks' summaries merge by rescaling to the larger maximum. Rank 0's shard has maximum 2 and
 * normaliser 3, rank 1's maximum 5 and normaliser 1.5; their pairs interleave by value, a tie
 * across ranks going to the lower id. A rank whose normaliser is not finite contributes nothing,
 * and with no rank left there is no normaliser. */
TEST(split_vocabulary_logprob_summaries_merge_across_ranks) {
    const float rows[2][8] = {
        { 2.0f, 3.0f,   5.0f, 2.0f,   1.0f, 1.0f,  -1.0f, -INFINITY },
        { 5.0f, 1.5f,  40.0f, 5.0f,  41.0f, 2.0f,  42.0f, 0.5f },
    };
    double lz = 0.0;
    int32_t id[8];
    float   v[8];
    CHECK_EQ(merge_logprob_rows(&rows[0][0], 2, 3, &lz, id, v, 8), 5);
    CHECK_NEAR(lz, std::log(3.0 * std::exp(2.0) + 1.5 * std::exp(5.0)), 1e-6);
    const int32_t want[5] = { 40, 5, 41, 1, 42 };
    for (int j = 0; j < 5; ++j) CHECK_EQ(id[j], want[j]);
    CHECK_EQ(v[1], 2.0f);
    CHECK_EQ(merge_logprob_rows(&rows[0][0], 2, 3, &lz, id, v, 2), 2);

    float dead[2][8];
    std::memcpy(dead, rows, sizeof dead);
    dead[1][1] = NAN;
    CHECK_EQ(merge_logprob_rows(&dead[0][0], 2, 3, &lz, id, v, 8), 2);
    CHECK_NEAR(lz, 2.0 + std::log(3.0), 1e-6);
    CHECK_EQ(id[0], 5);
    dead[0][1] = 0.0f;
    CHECK_EQ(merge_logprob_rows(&dead[0][0], 2, 3, &lz, id, v, 8), 0);
    CHECK(std::isnan(lz));
}

/* DRY is undone too. "22 23" ended the history once before, followed by 21, so DRY takes its
 * multiplier off token 21 (5 - 1 - 0.5 = 3.5); tokens 0..20 appear twice and lose more to the
 * frequency penalty (5 - 2 = 3), so 21 is chosen from outside the summary and its raw logit needs
 * DRY, the penalties and temperature all undone. */
TEST(a_dry_penalised_token_has_its_raw_logprob) {
    HostChain ch(32, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.top_k = 1;
    sp.temp = 0.8f;
    sp.freq_penalty = 1.0f;
    sp.dry_multiplier = 0.5f;
    sp.dry_base = 1.75f;
    sp.dry_allowed_length = 2;
    sp.n_probs = 0;
    Request a = make_req(1, outside_prompt({ 22, 23, 21 }, { 22, 23 }), sp);
    const std::vector<float> lg = outside_logits(5.0f, 5.0f, 0.0f);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 21);
    int64_t n = 0;
    CHECK((ch.s.staged_params(&n)[0].flags & RAD_SP_DRY) != 0);

    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, 5.0 - log_sum_exp(lg), 1e-5);
}

TEST(admission_refuses_more_than_twenty_logprobs) {
    Sampler s;
    SamplingParams sp;
    sp.n_probs = RAD_MAX_LOGPROBS + 1;
    std::string why;
    CHECK(s.check_request(sp, &why) < 0);
    CHECK(why.find("logprobs") != std::string::npos);
}

/* repeat_last_n = 0 turns the penalties off, as in llama.cpp; it does not mean "the whole
 * history". */
TEST(a_penalty_window_of_zero_leaves_the_logits_alone) {
    HostChain ch(4, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.rep_penalty = 10.0f;
    sp.penalty_last_n = 0;
    Request a = make_req(1, { 0 }, sp);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, { 3.0f, 2.9f, -5.0f, -5.0f }, &tok));
    CHECK_EQ(tok[0], 0);
    int64_t n = 0;
    CHECK_EQ(ch.s.staged_params(&n)[0].flags & RAD_SP_PENALTY, 0);
}

/* DRY runs with no breakers, as llama.cpp's does. History "a b c a b": c would extend the repeat
 * "a b" and is penalised by 1 * 2^0, which takes it from the top logit to below the rest. */
TEST(dry_runs_without_sequence_breakers) {
    HostChain ch(4, 1, 0);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.dry_multiplier = 1.0f;
    sp.dry_base = 2.0f;
    sp.dry_allowed_length = 2;
    Request a = make_req(1, { 0, 1, 2, 0, 1 }, sp);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, { 0.0f, 0.0f, 0.5f, 0.0f }, &tok));
    CHECK_EQ(tok[0], 0);
}

/* The breakers reach the kernel, both halves. Vocabulary b, \n, c, d, y; "\n" is the breaker.
 *
 * History "b \n c d y b \n c d": the suffix "b \n c d" repeats and y followed it, a repeat of 4
 * and a penalty of 2^(4-2) = 4 with no breakers. The \n caps the countable repeat at 2, so with
 * the breaker the penalty is 2^0 = 1.
 *
 * History "c d \n c d": the continuation of "c d" is "\n", a breaker, never penalised. */
TEST(dry_sequence_breakers_cap_the_repeat_and_exempt_themselves) {
    SimpleVocab v({ "b", "\n", "c", "d", "y" });
    HostChain ch(5, 1, 0, &v);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.dry_multiplier = 1.0f;
    sp.dry_base = 2.0f;
    sp.dry_allowed_length = 2;

    const std::vector<float> lg = { 1.0f, 1.0f, 1.0f, 1.0f, 2.5f };
    std::vector<int32_t> tok;
    Request plain = make_req(1, { 0, 1, 2, 3, 4, 0, 1, 2, 3 }, sp);
    CHECK_OK(ch.step({ &plain }, lg, &tok));
    CHECK_EQ(tok[0], 0);                  /* y at 2.5 - 4: below everything */

    SamplingParams with = sp;
    with.dry_seq_breakers = { "\n" };
    Request capped = make_req(2, { 0, 1, 2, 3, 4, 0, 1, 2, 3 }, with);
    CHECK_OK(ch.step({ &capped }, lg, &tok));
    CHECK_EQ(tok[0], 4);                  /* y at 2.5 - 1: still the top */
    int64_t n = 0;
    CHECK_EQ(ch.s.staged_params(&n)[0].dry_rep_limit, 2);

    const std::vector<float> nl = { 1.0f, 1.5f, 1.0f, 1.0f, 1.0f };
    Request exempt = make_req(3, { 2, 3, 1, 2, 3 }, with);
    CHECK_OK(ch.step({ &exempt }, nl, &tok));
    CHECK_EQ(tok[0], 1);                  /* \n is not penalised */
    Request not_exempt = make_req(4, { 2, 3, 1, 2, 3 }, sp);
    CHECK_OK(ch.step({ &not_exempt }, nl, &tok));
    CHECK_EQ(tok[0], 0);                  /* without the breaker it is */
}

/* The host reference and the chain draw the same token: same uniform, same walk order (the
 * candidate set in descending logit order), so over a distribution with no near-ties at the draw
 * they agree token for token. */
TEST(the_host_reference_and_the_chain_draw_the_same_token) {
    HostChain ch(4, 1, 0);
    NEED_CHAIN(ch)
    const std::vector<float> lg = { 0.1f, 0.9f, 0.4f, -0.3f };
    int agree = 0, total = 0;
    for (uint64_t pos = 0; pos < 48; ++pos) {
        SamplingParams sp;
        sp.temp = 1.0f;
        sp.seed = 77;
        Request a = make_req(1000 + pos, {}, sp);
        std::vector<int32_t> out(pos, 0);
        a.output = out;                                   /* the draw's position is n_out */
        std::vector<int32_t> tok;
        CHECK_OK(ch.step({ &a }, lg, &tok));
        HostRefInput in;
        in.pos = pos;
        const int32_t want = host_sample(lg.data(), 4, sp, in);
        ++total;
        if (tok[0] == want) ++agree;
        ch.s.end_request(a.id);
    }
    CHECK_EQ(agree, total);
}

/* Over a split vocabulary a greedy row is a top-1 chain row, staged without the greedy flag: every
 * candidate stage skips a flagged row, and this one must pass through top-k, the gather and the
 * merge to become a global token id. */
TEST(a_greedy_row_over_a_split_vocabulary_is_staged_as_a_chain_row) {
    /* Staging only: nothing runs, so the ops need not resolve, but with libref loaded they do and
     * declare has nothing to report. */
    Registry reg;
    (void)reg.load_plugin(home() + "/kernels/libref.so", 0);
    RadModelMeta meta{};
    meta.arch_id = "sampler_split";
    meta.name = "sampler_split";
    RadBuildCtx bc{};
    bc.world_size = 2;
    bc.max_seqs = 2;
    bc.max_tok = 2;
    bc.max_ctx = 64;
    bc.scope = "";
    Builder b(reg, meta, bc);
    Sampler s;
    SamplerConfig cfg;
    cfg.n_vocab = 64;
    cfg.n_vocab_local = 32;
    cfg.max_seqs = 2;
    cfg.world_size = 2;
    cfg.rank = 1;
    cfg.vocab_off = 32;
    cfg.max_hist = 16;
    cfg.max_candidates = 8;
    CHECK_OK(s.declare(&b, cfg));

    Request a = make_req(1, { 3 }, greedy_params());
    CHECK_OK(s.build_step({ &a }));
    int64_t n = 0;
    const DeviceSampleParams* p = s.staged_params(&n);
    CHECK_EQ(n, (int64_t)1);
    CHECK_EQ(p[0].top_k, 1);
    CHECK_EQ(p[0].flags & RAD_SP_GREEDY, 0);

    CHECK_OK(s.build_greedy(2));
    p = s.staged_params(&n);
    CHECK_EQ(p[1].flags & RAD_SP_GREEDY, 0);
    CHECK_EQ(p[1].top_k, 1);
}

/* Over a split vocabulary the merge holds every rank's candidates in one workgroup, 4096 pairs at
 * most (kSampleMergeCandidates, libr4d's sample_merge_topk). At two and four ranks the default
 * width of 1024 a rank fits; at eight it would ask 8192 and the step's merge would find no kernel,
 * so declare narrows it to 512 and admission refuses a wider top_k instead. */
TEST(the_candidate_width_over_a_split_vocabulary_is_what_the_merge_holds) {
    Registry reg;
    (void)reg.load_plugin(home() + "/kernels/libref.so", 0);
    for (int world : { 2, 4, 8 }) {
        RadModelMeta meta{};
        meta.arch_id = "sampler_split";
        meta.name = "sampler_split";
        RadBuildCtx bc{};
        bc.world_size = world;
        bc.max_seqs = 2;
        bc.max_tok = 2;
        bc.max_ctx = 64;
        bc.scope = "";
        Builder b(reg, meta, bc);
        Sampler s;
        SamplerConfig cfg;
        cfg.n_vocab = 2048 * (int64_t)world;
        cfg.n_vocab_local = 2048;
        cfg.max_seqs = 2;
        cfg.world_size = world;
        cfg.rank = world - 1;
        cfg.vocab_off = 2048 * (int64_t)(world - 1);
        cfg.max_hist = 16;
        CHECK_OK(s.declare(&b, cfg));

        const int64_t width = world == 8 ? 512 : kSampleCandidates;
        SamplingParams sp = greedy_params();
        std::string why;
        sp.top_k = (int)width;
        CHECK_OK(s.check_request(sp, &why));
        sp.top_k = (int)width + 1;
        CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
        CHECK(why.find("candidate width of " + std::to_string(width)) != std::string::npos);
    }
}

/* Triggers are resolved at admission, with special tokens recognised: a one-token word becomes an
 * id, a longer word is scanned for as bytes, a token trigger that is not one token is a bad
 * request, and a regular expression is refused by name (it would be matched over the reply once
 * per token on the scheduler thread). */
TEST(admission_resolves_grammar_triggers_and_refuses_a_regular_expression) {
    SimpleVocab v({ "a", "b", "<call>", "\n" });
    HostChain ch(4, 1, 0, &v);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.grammar = "root ::= \"<call>\\n\" \"a\"+";
    sp.grammar_lazy = true;
    std::string why;

    sp.grammar_triggers = { { 1, "<call>\n" } };
    CHECK_OK(ch.s.check_request(sp, &why));
    sp.grammar_triggers = { { 1, "<call>" } };
    CHECK_OK(ch.s.check_request(sp, &why));
    sp.grammar_triggers = { { 0, "<call>\n" } };
    CHECK_EQ(ch.s.check_request(sp, &why), RAD_E_INVAL);
    sp.grammar_triggers = { { 2, "<call>\\s*" } };
    why.clear();
    CHECK_EQ(ch.s.check_request(sp, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("regular expression") != std::string::npos);
    sp.grammar_triggers = { { 3, "^(<call>)[\\s\\S]*" } };
    CHECK_EQ(ch.s.check_request(sp, &why), RAD_E_UNSUPPORTED);
}

/* A thinking budget closes the reasoning block where the body reaches it, walked over the
 * speculative window as a grammar is: position k counts the committed reply and drafts 0..k-1, so
 * a window crossing the budget is forced from the first position past it, and a draft that closes
 * the block itself frees every position behind it. Token 1 wins every row's logits, so a row
 * returning 5 was forced. */
TEST(a_thinking_budget_forces_the_close_from_the_position_that_spends_it) {
    HostChain ch(8, 1, 2);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.reasoning.close = 5;
    sp.reasoning.starts_open = true;
    sp.thinking_budget = 2;
    std::vector<float> lg(3 * 8, 0.0f);
    for (int r = 0; r < 3; ++r) lg[(size_t)r * 8 + 1] = 5.0f;
    std::vector<int32_t> tok;

    Request a = make_req(1, { 3 }, sp);
    a.output = { 1 };
    int32_t da[2] = { 2, 2 };
    a.n_draft = 2;
    a.draft = da;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 1);                   /* one body token: within the budget */
    CHECK_EQ(tok[1], 5);                   /* behind a draft that makes it two */
    CHECK_EQ(tok[2], 5);

    Request b = make_req(2, { 3 }, sp);
    b.output = { 1 };
    int32_t db[2] = { 5, 2 };
    b.n_draft = 2;
    b.draft = db;
    CHECK_OK(ch.step({ &b }, lg, &tok));
    CHECK_EQ(tok[0], 1);
    CHECK_EQ(tok[1], 1);                   /* the draft closed the block: nothing left to force */
    CHECK_EQ(tok[2], 1);
}

/* The committed reply is counted across steps, and only from where the block opens: a reply that
 * opens its own block counts from its opener, and one that starts any other way has no block to
 * budget. Sampled, not greedy, so the mask decides rather than an argmax that happens to agree. */
TEST(a_thinking_budget_counts_the_reply_across_steps_and_only_inside_the_block) {
    HostChain ch(8, 2, 0);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.reasoning.open = 4;
    sp.reasoning.close = 5;
    sp.thinking_budget = 1;
    std::vector<float> lg(2 * 8, 0.0f);
    lg[1] = 40.0f;
    lg[8 + 1] = 40.0f;
    std::vector<int32_t> tok;

    Request opens = make_req(1, { 3 }, sp);
    Request plain = make_req(2, { 3 }, sp);
    opens.output = { 4 };                  /* the opener: the block starts, nothing in it yet */
    plain.output = { 2 };                  /* no opener first: there is no block */
    CHECK_OK(ch.step({ &opens, &plain }, lg, &tok));
    CHECK_EQ(tok[0], 1);
    CHECK_EQ(tok[1], 1);

    opens.output.push_back(1);             /* one body token: the budget is spent */
    plain.output.push_back(1);
    CHECK_OK(ch.step({ &opens, &plain }, lg, &tok));
    CHECK_EQ(tok[0], 5);
    CHECK_EQ(tok[1], 1);

    opens.output.push_back(5);             /* closed: the answer is free again */
    CHECK_OK(ch.step({ &opens, &plain }, lg, &tok));
    CHECK_EQ(tok[0], 1);
}

/* A grammar keeps the last word. The close is forced only where the grammar admits it; elsewhere
 * the row follows the grammar, and the budget waits for a position the grammar allows. */
TEST(a_thinking_budget_forces_the_close_only_where_the_grammar_admits_it) {
    SimpleVocab v({ "a", "b", "c", "</t>" });
    HostChain ch(4, 1, 0, &v);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.reasoning.close = 3;
    sp.reasoning.starts_open = true;
    sp.thinking_budget = 0;
    std::vector<float> lg = { 9.0f, 0.0f, 0.0f, 0.0f };
    std::vector<int32_t> tok;

    sp.grammar = "root ::= (\"a\" | \"</t>\")+";
    Request admits = make_req(1, { 1 }, sp);
    CHECK_OK(ch.step({ &admits }, lg, &tok));
    CHECK_EQ(tok[0], 3);

    sp.grammar = "root ::= \"a\"+";
    Request forbids = make_req(2, { 1 }, sp);
    CHECK_OK(ch.step({ &forbids }, lg, &tok));
    CHECK_EQ(tok[0], 0);
}

/* The count a budget is held against and the count usage reports are one definition. */
TEST(reasoning_count_follows_the_parser_s_reading_of_the_block) {
    ReasoningSpan s;
    s.open = 4;
    s.close = 5;
    ReasoningCount c;
    c.start(s);
    for (int32_t t : { 4, 1, 1, 5, 1, 4, 1 }) c.push(s, t);
    CHECK_EQ(c.tokens, (int64_t)4);        /* the opener, two body tokens, the close */
    CHECK_EQ(c.body, (int64_t)2);
    CHECK(!c.inside());

    s.starts_open = true;
    c.start(s);
    for (int32_t t : { 1, 1, 1 }) c.push(s, t);
    CHECK(c.inside());
    CHECK_EQ(c.tokens, (int64_t)3);

    s.starts_open = false;
    c.start(s);
    for (int32_t t : { 1, 4, 1 }) c.push(s, t);
    CHECK_EQ(c.tokens, (int64_t)0);        /* a block opened mid-reply is not one */

    c.start(ReasoningSpan{});
    c.push(ReasoningSpan{}, 1);
    CHECK_EQ(c.tokens, (int64_t)0);
}

/* ================================================================== the mask's width and owner */

/* The mask is one bit a token, and its width is the one number the grammar backend, the
 * sampler's staging buffer and the device kernel must agree on. A width short by one word leaves
 * the top 32 tokens unmasked, admissible whatever the grammar says; one word long has the kernel
 * read past the buffer. Neither shows up as an error. */
TEST(the_mask_width_covers_the_whole_vocabulary_and_no_more) {
    CHECK_EQ(grammar_mask_words(0), 0ll);
    CHECK_EQ(grammar_mask_words(1), 1ll);
    CHECK_EQ(grammar_mask_words(31), 1ll);
    CHECK_EQ(grammar_mask_words(32), 1ll);
    CHECK_EQ(grammar_mask_words(33), 2ll);
    CHECK_EQ(grammar_mask_words(64), 2ll);
    /* 151936 is a multiple of 32 and 248046 is not, the pair that separates a correct ceiling
     * from a floor plus one. */
    CHECK_EQ(grammar_mask_words(151936), 151936ll / 32);
    CHECK_EQ(grammar_mask_words(248046), (248046ll + 31) / 32);
    for (int64_t v = 1; v <= 200; ++v) {
        const int64_t w = grammar_mask_words(v);
        CHECK(w * 32 >= v);
        CHECK((w - 1) * 32 < v);
    }
}

/* What GBNF derives from a vocabulary (the decoded pieces, the trie, the compiled programs) is
 * cached with the vocabulary itself. Keyed on the view's address instead, a view built in the
 * storage of a destroyed one is served the old one's pieces: the grammar compiles, fill_mask
 * succeeds, and the mask is over another vocabulary's tokens. The two vocabularies share no piece,
 * so a stale entry cannot answer correctly by accident, and each is built in the same lambda frame
 * so the second very likely lands on the first one's storage. */
TEST(a_second_vocabulary_is_not_served_the_first_ones_decoding) {
    auto ids_for = [](const char* a, const char* b, const char* c, const char* d,
                      const char* grammar) {
        SimpleVocab v;
        v.add(a); v.add(b); v.add(c); v.add(d); v.add("", true);
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        std::vector<int32_t> out;
        if (GbnfGrammar::create(grammar, "root", &v, false, {}, {}, &g, &err) < 0) return out;
        return admitted(*g, v.n_tokens());
    };

    const auto letters = ids_for("a", "b", "c", "ab", "root ::= [ab]+\n");
    CHECK(letters == (std::vector<int32_t>{ 0, 1, 3 }));
    const auto digits = ids_for("0", "1", "x", "01", "root ::= [01]+\n");
    CHECK(digits == (std::vector<int32_t>{ 0, 1, 3 }));
    /* And back, so the case fails whichever of the two a stale entry held. */
    CHECK(ids_for("a", "b", "c", "ab", "root ::= [ab]+\n") == letters);
    CHECK(ids_for("x", "1", "0", "10", "root ::= [01]+\n") == (std::vector<int32_t>{ 1, 2, 3 }));
}

/* Two vocabularies alive at once each get their own answers, in either order of first use: the
 * case an address key got right and a content fingerprint would not. */
TEST(two_live_vocabularies_each_get_their_own_mask) {
    SimpleVocab a, b;
    a.add("a"); a.add("b"); a.add("c"); a.add("", true);
    b.add("0"); b.add("1"); b.add("x"); b.add("", true);

    std::unique_ptr<GbnfGrammar> ga, gb;
    std::string err;
    CHECK_OK(GbnfGrammar::create("root ::= [ab]+\n", "root", &a, false, {}, {}, &ga, &err));
    CHECK_OK(GbnfGrammar::create("root ::= [01]+\n", "root", &b, false, {}, {}, &gb, &err));

    CHECK(admitted(*gb, b.n_tokens()) == (std::vector<int32_t>{ 0, 1 }));
    CHECK(admitted(*ga, a.n_tokens()) == (std::vector<int32_t>{ 0, 1 }));
    CHECK_OK(ga->accept_token(0));
    CHECK_OK(gb->accept_token(1));
    CHECK(admitted(*ga, a.n_tokens()) == (std::vector<int32_t>{ 0, 1, 3 }));   /* and now eog */
    CHECK(admitted(*gb, b.n_tokens()) == (std::vector<int32_t>{ 0, 1, 3 }));
}

/* The same grammar text against two vocabularies is two programs: a program holds its
 * vocabulary's token sets, so sharing one across vocabularies would mask with the wrong ids. */
TEST(one_grammar_text_compiles_to_a_program_per_vocabulary) {
    SimpleVocab a, b;
    a.add("a"); a.add("b"); a.add("", true);
    b.add("b"); b.add("a"); b.add("", true);
    std::unique_ptr<GbnfGrammar> ga, gb, ga2;
    std::string err;
    CHECK_OK(GbnfGrammar::create("root ::= \"a\"\n", "root", &a, false, {}, {}, &ga, &err));
    CHECK_OK(GbnfGrammar::create("root ::= \"a\"\n", "root", &b, false, {}, {}, &gb, &err));
    CHECK_OK(GbnfGrammar::create("root ::= \"a\"\n", "root", &a, false, {}, {}, &ga2, &err));
    CHECK(&ga->program() != &gb->program());
    CHECK(&ga->program() == &ga2->program());
    CHECK(admitted(*ga, a.n_tokens()) == (std::vector<int32_t>{ 0 }));
    CHECK(admitted(*gb, b.n_tokens()) == (std::vector<int32_t>{ 1 }));
}

/* ================================================================== GBNF error text */

/* A parse error quotes at most 40 bytes of the grammar, cut on a character boundary. The message
 * goes into a JSON error body; a cut through a multi-byte character makes it invalid UTF-8, which
 * the strict JSON encoder refuses, turning a 400 into a 500. */
TEST(a_gbnf_parse_error_quotes_a_short_well_formed_excerpt) {
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\" ) " + std::string(1 << 20, 'x'), "root", &v,
                                 false, {}, {}, &g, &err), RAD_E_FORMAT);
    CHECK(err.size() < 120);
    CHECK(err.size() >= 3 && err.compare(err.size() - 3, 3, "...") == 0);

    std::string cjk;
    for (int i = 0; i < 20; ++i) cjk += "\xe6\x97\xa5\xe6\x9c\xac";   /* 日本, 3 bytes each */
    /* The excerpt starts at ')': ") " then 38 bytes, which end inside the 13th character. */
    err.clear();
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\" ) " + cjk, "root", &v, false, {}, {}, &g, &err),
             RAD_E_FORMAT);
    CHECK(!err.empty());
    CHECK_EQ(utf8_valid_prefix(err), err.size());
    CHECK(err.find(") \xe6\x97\xa5") != std::string::npos);
}

/* ================================================================== lazy grammars */

/* A lazy grammar constrains nothing until a trigger fires, so one with no trigger would leave the
 * whole reply unconstrained with nothing to say so. Refused at compile and at admission. */
TEST(a_lazy_grammar_without_a_trigger_is_refused) {
    SimpleVocab v({ "a", "b", "<call>" });
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\"", "root", &v, true, {}, {}, &g, &err),
             RAD_E_INVAL);
    CHECK(err.find("trigger") != std::string::npos);
    CHECK_OK(GbnfGrammar::create("root ::= \"a\"", "root", &v, true, {}, { 2 }, &g, &err));
    CHECK_OK(GbnfGrammar::create("root ::= \"a\"", "root", &v, false, {}, {}, &g, &err));

    HostChain ch(3, 1, 0, &v);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.grammar = "root ::= \"<call>\" \"a\"";
    sp.grammar_lazy = true;
    std::string why;
    CHECK_EQ(ch.s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("trigger") != std::string::npos);
    /* A trigger whose value is empty is skipped, so it does not count. */
    sp.grammar_triggers = { { 1, "" } };
    CHECK_EQ(ch.s.check_request(sp, &why), RAD_E_INVAL);
    sp.grammar_triggers = { { 1, "<call>" } };
    CHECK_OK(ch.s.check_request(sp, &why));
    sp.grammar_lazy = false;
    sp.grammar_triggers.clear();
    CHECK_OK(ch.s.check_request(sp, &why));
}

/* A template's eager grammar describes the whole assistant turn, generation prompt included, but
 * that prompt is already in the context: the sampler replays it into the grammar, or the mask
 * would make the model write it again. A lazy grammar starts at its trigger and must not replay
 * it; a prompt the grammar refuses fails the request rather than mask from a wrong state. */
TEST(an_eager_grammar_replays_the_generation_prompt_and_a_lazy_one_does_not) {
    SimpleVocab v({ "<p>", "a", "b" });
    HostChain ch(3, 1, 0, &v);
    NEED_CHAIN(ch)
    const std::vector<float> lg = { 5.0f, 1.0f, 0.0f };      /* "<p>" first, then "a" */
    std::vector<int32_t> tok;

    SamplingParams sp = greedy_params();
    sp.grammar = "root ::= \"<p>\" \"a\"+";
    sp.grammar_prefix = "<p>";
    Request eager = make_req(1, { 2 }, sp);
    CHECK_OK(ch.step({ &eager }, lg, &tok));
    CHECK_EQ(tok[0], 1);

    sp.grammar_lazy = true;
    sp.grammar_triggers = { { 1, "<p>" } };
    Request lazy = make_req(2, { 2 }, sp);
    CHECK_OK(ch.step({ &lazy }, lg, &tok));
    CHECK_EQ(tok[0], 0);
    int64_t n = 0;
    CHECK_EQ(ch.s.staged_params(&n)[0].flags & RAD_SP_MASK, 0);

    SamplingParams bad = greedy_params();
    bad.grammar = "root ::= \"<p>\" \"a\"+";
    bad.grammar_prefix = "<q>";
    std::string err;
    CHECK(ch.s.begin_request(3, bad, &err) < 0);
    CHECK(err.find("generation prompt") != std::string::npos);
    CHECK(!ch.s.has_request(3));
}

/* ================================================================== the chain against host_ref */

namespace {

/* splitmix64 as a stream, for the randomised cases. */
struct Mix {
    uint64_t s;
    explicit Mix(uint64_t seed) : s(seed) {}
    uint64_t next() { return rad_sample_splitmix64(s += 0x9E3779B97F4A7C15ull); }
    double unit() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    int below(int n) { return (int)(next() % (uint64_t)n); }
    bool chance(int pct) { return below(100) < pct; }
    float in(double lo, double hi) { return (float)(lo + (hi - lo) * unit()); }
};

std::vector<double> probs_of(const CandidateSet& s) {
    double mx = -INFINITY;
    for (const auto& e : s.c) mx = std::max(mx, (double)e.logit);
    std::vector<double> p;
    double z = 0.0;
    for (const auto& e : s.c) {
        p.push_back(std::exp((double)e.logit - mx));
        z += p.back();
    }
    for (double& x : p) x /= z;
    return p;
}

/* How close the row's outcome comes to a rounding decision: the smallest distance, over every cut
 * and the draw, between a compared quantity and its threshold, and between two candidates that
 * differ by less than rounding. Where it is tiny two correct implementations that sum in
 * different orders may disagree, so such a row proves nothing either way. Recomputed in double
 * from host_ref's stages, in the order host_sample runs them. */
double rounding_margin(const std::vector<float>& lg, const SamplingParams& sp,
                       const HostRefInput& in) {
    CandidateSet s;
    s.from_logits(lg.data(), (int64_t)lg.size());
    host_penalties(s, in.history, in.n_history, sp.penalty_last_n, sp.rep_penalty,
                   sp.freq_penalty, sp.pres_penalty);
    if (sp.dry_multiplier > 0.0f) {
        DryParams dp;
        dp.multiplier = sp.dry_multiplier;
        dp.base = sp.dry_base;
        dp.allowed_length = sp.dry_allowed_length;
        dp.penalty_last_n = sp.dry_penalty_last_n;
        host_dry(s, in.history, in.n_history, in.total_context, dp,
                 in.dry_breakers ? *in.dry_breakers : DryBreakers{});
    }
    double m = INFINITY;
    auto near_ties = [&](const CandidateSet& c) {
        CandidateSet t = c;
        t.sorted = false;
        t.sort_desc();
        for (size_t i = 0; i + 1 < t.c.size(); ++i) {
            const double d = (double)t.c[i].logit - (double)t.c[i + 1].logit;
            if (d > 0.0) m = std::min(m, d);
        }
    };
    if (sp.greedy()) {
        near_ties(s);
        return m;
    }
    if (!sp.temp_last) host_temp(s, sp.temp);
    const int64_t width = in.max_candidates > 0 ? in.max_candidates : (int64_t)s.c.size();
    host_top_k(s, (int32_t)(sp.top_k > 0 ? std::min<int64_t>(sp.top_k, width) : width));
    near_ties(s);
    if (sp.top_p < 1.0f) {
        s.sort_desc();
        double cum = 0.0;
        for (double x : probs_of(s)) { cum += x; m = std::min(m, std::fabs(cum - sp.top_p)); }
    }
    host_top_p(s, sp.top_p, 1);
    if (sp.min_p > 0.0f) {
        double mx = -INFINITY;
        for (const auto& e : s.c) mx = std::max(mx, (double)e.logit);
        for (const auto& e : s.c)
            m = std::min(m, std::fabs(std::exp((double)e.logit - mx) - sp.min_p));
    }
    host_min_p(s, sp.min_p, 1);
    if (sp.typical_p < 1.0f) {
        const std::vector<double> p = probs_of(s);
        double h = 0.0;
        for (double x : p) if (x > 0.0) h -= x * std::log(x);
        std::vector<double> d;
        for (double x : p) d.push_back(x > 0.0 ? std::fabs(-std::log(x) - h) : INFINITY);
        for (size_t i = 0; i < d.size(); ++i) {
            double mass = 0.0;
            for (size_t j = 0; j < d.size(); ++j) {
                if (d[j] <= d[i]) mass += p[j];
                if (d[j] != d[i]) m = std::min(m, std::fabs(d[j] - d[i]));
            }
            m = std::min(m, std::fabs(mass - sp.typical_p));
        }
    }
    host_typical(s, sp.typical_p, 1);
    if (sp.xtc_probability > 0.0f && sp.xtc_threshold > 0.0f &&
        rng_uniform(sp.seed, in.pos, RAD_RNG_XTC) < sp.xtc_probability)
        for (double x : probs_of(s)) m = std::min(m, std::fabs(x - sp.xtc_threshold));
    host_xtc(s, sp.xtc_probability, sp.xtc_threshold, 1, sp.seed, in.pos);
    if (sp.temp_last) host_temp(s, sp.temp);
    s.sorted = false;
    s.sort_desc();
    const double u = rng_uniform(sp.seed, in.pos, RAD_RNG_PICK);
    double cum = 0.0;
    for (double x : probs_of(s)) { cum += x; m = std::min(m, std::fabs(cum - u)); }
    return m;
}

/* A random request: each stage on or off at random, values across their ranges, including the
 * boundary values the chain reads specially (top_k 0, xtc_threshold 0, temperature 0). */
SamplingParams random_params(Mix& r) {
    SamplingParams sp;
    static const float temps[] = { 0.0f, 0.3f, 0.7f, 1.0f, 1.6f };
    sp.temp = temps[r.below(5)];
    sp.top_k = r.chance(40) ? 0 : 1 + r.below(24);
    sp.top_p = r.chance(50) ? 1.0f : r.in(0.05, 1.0);
    sp.min_p = r.chance(60) ? 0.0f : r.in(0.0, 0.5);
    sp.typical_p = r.chance(50) ? 1.0f : r.in(0.05, 1.0);
    if (r.chance(50)) {
        sp.rep_penalty = r.in(0.5, 2.0);
        sp.freq_penalty = r.in(-1.0, 1.0);
        sp.pres_penalty = r.in(-1.0, 1.0);
    }
    static const int windows[] = { -1, 0, 4, 16, 64 };
    sp.penalty_last_n = windows[r.below(5)];
    if (r.chance(50)) {
        static const float mults[] = { 0.5f, 0.8f, 1.0f, 1.25f };
        static const float bases[] = { 1.0f, 1.5f, 1.75f, 2.5f };
        static const int dwin[] = { -1, 0, 8, 32 };
        sp.dry_multiplier = mults[r.below(4)];
        sp.dry_base = bases[r.below(4)];
        sp.dry_allowed_length = 1 + r.below(3);
        sp.dry_penalty_last_n = dwin[r.below(4)];
        if (r.chance(50)) sp.dry_seq_breakers = { "\n" };
    }
    if (r.chance(40)) {
        sp.xtc_probability = r.in(0.2, 1.0);
        sp.xtc_threshold = r.chance(25) ? 0.0f : r.in(0.01, 0.5);
    }
    sp.temp_last = r.chance(30);
    sp.seed = r.next();
    return sp;
}

/* A history that repeats itself, so DRY and the penalties have something to find. */
std::vector<int32_t> random_history(Mix& r, int n_vocab, int len) {
    std::vector<int32_t> motif;
    for (int i = 0, n = 2 + r.below(4); i < n; ++i) motif.push_back(r.below(n_vocab));
    std::vector<int32_t> h;
    while ((int)h.size() < len) {
        if (r.chance(60)) h.insert(h.end(), motif.begin(), motif.end());
        else h.push_back(r.below(n_vocab));
    }
    h.resize((size_t)len);
    return h;
}

}  /* namespace */

/* The chain that runs (Sampler staging libref's ops) and the host reference draw the same token
 * for every request shape: every stage on or off, its values across their ranges, histories with
 * repeats and a DRY breaker, flat and tied logits, temperature first and last. Each formulation
 * is independent, so a wrong stage in either (a penalty's sign, DRY's window, a cut's threshold,
 * XTC's coin, where temperature runs) shows as a disagreement. Rows whose outcome turns on a
 * rounding (rounding_margin) are skipped and counted. */
TEST(the_chain_matches_the_host_reference_on_random_rows) {
    constexpr int kVocab = 24, kSeqs = 16, kSteps = 40;
    SimpleVocab v;
    for (int i = 0; i < kVocab - 1; ++i) v.add(std::string(1, (char)('a' + i)));
    const int32_t nl = v.add("\n");
    DryBreakers brk;
    REQUIRE(dry_prepare_breakers(v, { "\n" }, &brk) >= 0);
    REQUIRE(brk.is_single(nl));

    HostChain ch(kVocab, kSeqs, 0, &v, /*max_candidates*/ 32);
    NEED_CHAIN(ch)
    const uint64_t seed = 0x5eed0001;
    Mix r(seed);
    int compared = 0, skipped = 0, typical_rows = 0, xtc_off_rows = 0, dry_rows = 0;
    uint64_t next_id = 1;
    for (int step = 0; step < kSteps; ++step) {
        std::vector<Request> rq(kSeqs);
        std::vector<Request*> ptr;
        std::vector<float> lg((size_t)kSeqs * kVocab);
        for (int i = 0; i < kSeqs; ++i) {
            const SamplingParams sp = random_params(r);
            rq[(size_t)i] = make_req(next_id++, random_history(r, kVocab, r.below(40)), sp);
            rq[(size_t)i].output = random_history(r, kVocab, r.below(30));
            ptr.push_back(&rq[(size_t)i]);
            const bool tied = r.chance(50);
            for (int t = 0; t < kVocab; ++t)
                lg[(size_t)i * kVocab + t] = tied ? (float)(r.below(25) - 12) * 0.25f
                                                  : r.in(-4.0, 4.0);
        }
        std::vector<int32_t> tok;
        REQUIRE(ch.step(ptr, lg, &tok) >= 0);
        for (int i = 0; i < kSeqs; ++i) {
            const Request& q = rq[(size_t)i];
            /* The history the chain staged: prompt then output, its last max_hist (64) tokens. */
            std::vector<int32_t> hist = q.prompt;
            hist.insert(hist.end(), q.output.begin(), q.output.end());
            if (hist.size() > 64) hist.erase(hist.begin(), hist.end() - 64);
            HostRefInput in;
            in.history = hist.data();
            in.n_history = (int64_t)hist.size();
            in.pos = q.output.size();
            in.dry_breakers = q.sp.dry_seq_breakers.empty() ? nullptr : &brk;
            in.max_candidates = 32;
            const std::vector<float> row(lg.begin() + (ptrdiff_t)i * kVocab,
                                         lg.begin() + (ptrdiff_t)(i + 1) * kVocab);
            if (rounding_margin(row, q.sp, in) < 1e-5) { ++skipped; continue; }
            ++compared;
            typical_rows += q.sp.typical_p < 1.0f && !q.sp.greedy();
            xtc_off_rows += q.sp.xtc_probability > 0.0f && q.sp.xtc_threshold == 0.0f;
            dry_rows += q.sp.dry_multiplier > 0.0f;
            const int32_t want = host_sample(row.data(), kVocab, q.sp, in);
            if (tok[(size_t)i] != want) {
                CHECK_EQ(tok[(size_t)i], want);
                std::fprintf(stderr, "    seed %llx step %d row %d: temp %g top_k %d top_p %g "
                             "min_p %g typical %g rep %g/%g/%g last_n %d dry %g/%g/%d/%d brk %d "
                             "xtc %g/%g temp_last %d\n", (unsigned long long)seed, step, i,
                             q.sp.temp, q.sp.top_k, q.sp.top_p, q.sp.min_p, q.sp.typical_p,
                             q.sp.rep_penalty, q.sp.freq_penalty, q.sp.pres_penalty,
                             q.sp.penalty_last_n, q.sp.dry_multiplier, q.sp.dry_base,
                             q.sp.dry_allowed_length, q.sp.dry_penalty_last_n,
                             (int)!q.sp.dry_seq_breakers.empty(), q.sp.xtc_probability,
                             q.sp.xtc_threshold, (int)q.sp.temp_last);
            }
        }
        for (const Request& q : rq) ch.s.end_request(q.id);
    }
    std::fprintf(stderr, "    %d rows compared, %d skipped at a rounding boundary\n", compared,
                 skipped);
    CHECK(compared >= kSteps * kSeqs * 9 / 10);
    CHECK(typical_rows > 40);
    CHECK(xtc_off_rows > 10);
    CHECK(dry_rows > 100);
}

/* top_k 0 is the candidate width, not the whole vocabulary: every chain row runs top-k at
 * max_candidates, so on flat logits a 32-token vocabulary with 8 candidates never draws past the
 * 8 lowest ids. host_ref given the same width draws the same token at every position, and the
 * whole vocabulary would not. */
TEST(top_k_zero_samples_the_candidate_width) {
    HostChain ch(32, 1, 0, nullptr, /*max_candidates*/ 8);
    NEED_CHAIN(ch)
    const std::vector<float> lg(32, 0.0f);
    int agree = 0, past_width = 0, whole_past = 0;
    for (uint64_t pos = 0; pos < 200; ++pos) {
        SamplingParams sp;
        sp.temp = 2.0f;
        sp.top_k = 0;
        sp.seed = 99;
        Request a = make_req(5000 + pos, {}, sp);
        a.output.assign(pos, 0);
        std::vector<int32_t> tok;
        REQUIRE(ch.step({ &a }, lg, &tok) >= 0);
        ch.s.end_request(a.id);
        past_width += tok[0] >= 8;
        HostRefInput in;
        in.pos = pos;
        in.max_candidates = 8;
        agree += tok[0] == host_sample(lg.data(), 32, sp, in);
        in.max_candidates = 0;
        whole_past += host_sample(lg.data(), 32, sp, in) >= 8;
    }
    CHECK_EQ(past_width, 0);
    CHECK_EQ(agree, 200);
    CHECK(whole_past > 100);

    std::string why;
    SamplingParams wide;
    wide.top_k = 9;
    CHECK_EQ(ch.s.check_request(wide, &why), RAD_E_INVAL);
    CHECK(why.find("candidate width of 8") != std::string::npos);
}

/* ================================================================== speculation */

namespace {

/* A deterministic toy model: the logits after a prefix are a hash of it. A speculative run and a
 * token-at-a-time run see the same distribution at the same prefix, as the engine's verify and
 * decode passes do when they compute the same logits. */
std::vector<float> toy_logits(const std::vector<int32_t>& prefix, int64_t nv) {
    uint64_t h = 0x243F6A8885A308D3ull;
    for (int32_t t : prefix) h = rad_sample_splitmix64(h ^ (uint64_t)(t + 1));
    std::vector<float> lg((size_t)nv);
    for (int64_t i = 0; i < nv; ++i) {
        h = rad_sample_splitmix64(h);
        lg[(size_t)i] = (float)((int64_t)(h >> 40) % 2001 - 1000) / 400.0f;
    }
    return lg;
}

/* Sixteen pieces: letters, a separator, a newline (a DRY breaker), a multi-letter piece and a
 * reasoning block's markers. */
SimpleVocab toy_vocab() {
    return SimpleVocab({ "a", "b", "c", "x", "y", " ", "\n", "ab", "d", "e", "f", "g", "h", "i",
                         "<t>", "</t>" });
}

/* Eight requests that between them use every stage the step path stages per position: penalties,
 * DRY with a breaker, XTC, min-p, top-p, typical, temperature last, a grammar and a thinking
 * budget, plus one greedy. */
std::vector<SamplingParams> toy_params() {
    std::vector<SamplingParams> v(8);
    v[0].temp = 0.9f; v[0].top_p = 0.9f;
    v[1].temp = 0.7f; v[1].rep_penalty = 1.3f; v[1].freq_penalty = 0.2f; v[1].pres_penalty = 0.1f;
    v[1].penalty_last_n = 16;
    v[2].temp = 1.0f; v[2].dry_multiplier = 0.8f; v[2].dry_allowed_length = 2;
    v[2].dry_seq_breakers = { "\n" };
    v[3].temp = 1.2f; v[3].xtc_probability = 0.5f; v[3].xtc_threshold = 0.1f; v[3].min_p = 0.05f;
    v[4].temp = 0.8f; v[4].temp_last = true; v[4].top_k = 6; v[4].typical_p = 0.9f;
    v[5].temp = 0.9f; v[5].grammar = "root ::= (\"x\" [abc]+ \"y\" \" \"?)+";
    v[6].temp = 1.0f; v[6].reasoning.open = 14; v[6].reasoning.close = 15;
    v[6].reasoning.starts_open = true; v[6].thinking_budget = 5;
    v[7].temp = 0.0f; v[7].rep_penalty = 1.5f;
    for (size_t i = 0; i < v.size(); ++i) v[i].seed = 1000 + i;
    return v;
}

}  /* namespace */

/* Speculation emits exactly the tokens a token-at-a-time run emits, at any temperature, given the
 * same logits: each verified position draws at its own output position (n_out + k), against the
 * drafts before it in the penalty and DRY history, under the grammar state and thinking budget
 * after them. A regression in any of these changes sampled text only when speculation is on. Drafts
 * mix the reference continuation, so some are accepted, with random tokens, so some are not. */
TEST(speculation_emits_the_tokens_a_token_at_a_time_run_emits) {
    constexpr int kLen = 48, kSpec = 3;
    const SimpleVocab v = toy_vocab();
    const int64_t nv = v.n_tokens();
    HostChain ch(nv, 8, kSpec, &v, 16);
    NEED_CHAIN(ch)
    const std::vector<SamplingParams> params = toy_params();
    const std::vector<int32_t> prompt = { 5, 1, 2, 13 };

    std::vector<Request> ref(8);
    for (size_t i = 0; i < 8; ++i) ref[i] = make_req(1 + i, prompt, params[i]);
    for (int t = 0; t < kLen; ++t) {
        std::vector<Request*> ptr;
        std::vector<float> lg;
        for (Request& q : ref) {
            ptr.push_back(&q);
            std::vector<int32_t> pre = q.prompt;
            pre.insert(pre.end(), q.output.begin(), q.output.end());
            const auto row = toy_logits(pre, nv);
            lg.insert(lg.end(), row.begin(), row.end());
        }
        std::vector<int32_t> tok;
        REQUIRE(ch.step(ptr, lg, &tok) >= 0);
        REQUIRE(ch.s.take_failed().empty());
        for (size_t i = 0; i < 8; ++i) {
            REQUIRE(ch.s.accept_token(ref[i].id, tok[i]) >= 0);
            ref[i].output.push_back(tok[i]);
        }
    }

    std::vector<Request> spec(8);
    std::vector<std::vector<int32_t>> drafts(8, std::vector<int32_t>(kSpec));
    for (size_t i = 0; i < 8; ++i) spec[i] = make_req(101 + i, prompt, params[i]);
    Mix r(77);
    int accepted = 0, steps = 0;
    auto done = [&] {
        for (const Request& q : spec) if ((int)q.output.size() < kLen) return false;
        return true;
    };
    while (!done() && steps < 200) {
        ++steps;
        std::vector<Request*> ptr;
        std::vector<float> lg;
        for (size_t i = 0; i < 8; ++i) {
            Request& q = spec[i];
            for (int k = 0; k < kSpec; ++k) {
                const size_t at = q.output.size() + (size_t)k;
                drafts[i][(size_t)k] = r.chance(75) && at < ref[i].output.size()
                                           ? ref[i].output[at] : r.below((int)nv);
            }
            q.n_draft = kSpec;
            q.draft = drafts[i].data();
            ptr.push_back(&q);
            std::vector<int32_t> pre = q.prompt;
            pre.insert(pre.end(), q.output.begin(), q.output.end());
            for (int k = 0; k <= kSpec; ++k) {
                if (k > 0) pre.push_back(drafts[i][(size_t)k - 1]);
                const auto row = toy_logits(pre, nv);
                lg.insert(lg.end(), row.begin(), row.end());
            }
        }
        std::vector<int32_t> tok;
        REQUIRE(ch.step(ptr, lg, &tok) >= 0);
        REQUIRE(ch.s.take_failed().empty());
        for (size_t i = 0; i < 8; ++i) {
            int32_t out[kSpec + 1];
            int32_t n_acc = 0;
            REQUIRE(accept_greedy(drafts[i].data(), kSpec, &tok[i * (kSpec + 1)], out, &n_acc) >= 0);
            accepted += n_acc;
            for (int k = 0; k <= n_acc; ++k) {
                REQUIRE(ch.s.accept_token(spec[i].id, out[k]) >= 0);
                spec[i].output.push_back(out[k]);
            }
        }
    }
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(spec[i].output.size() >= (size_t)kLen);
        const std::vector<int32_t> head(spec[i].output.begin(), spec[i].output.begin() + kLen);
        if (head != ref[i].output) {
            int at = 0;
            while (head[(size_t)at] == ref[i].output[(size_t)at]) ++at;
            std::fprintf(stderr, "    request %zu diverges at token %d\n", i, at);
        }
        CHECK(head == ref[i].output);
    }
    /* The drafts were accepted often enough for the windows to matter. */
    CHECK(accepted > 150);
    std::fprintf(stderr, "    %d steps, %d drafts accepted\n", steps, accepted);
}

/* A grammar over a speculative window. Greedy rows; each row's logits favour a token the grammar
 * forbids there, so the output shows which mask applied. Position k is masked against the state
 * after drafts 0..k-1; masking stops at a refused draft, without failing the request, since
 * verification rejects there; and the window is rewound afterwards. */
TEST(a_grammar_masks_each_draft_position_against_the_drafts_before_it) {
    SimpleVocab v({ "a", "b", "c", "x", "<call>", "z" });
    HostChain ch(6, 1, 3, &v);
    NEED_CHAIN(ch)
    auto favour = [](std::initializer_list<int> best) {
        std::vector<float> lg;
        for (int b : best) {
            std::vector<float> row(6, 0.0f);
            row[(size_t)b] = 5.0f;
            lg.insert(lg.end(), row.begin(), row.end());
        }
        return lg;
    };
    std::vector<int32_t> tok;
    int64_t n = 0;

    SamplingParams sp = greedy_params();
    sp.grammar = "root ::= \"a\" \"b\" \"c\"";
    Request a = make_req(1, { 5 }, sp);
    const int32_t da[3] = { 0, 1, 3 };                 /* a, b, then x: refused */
    a.n_draft = 3;
    a.draft = da;
    CHECK_OK(ch.step({ &a }, favour({ 1, 0, 0, 0 }), &tok));
    CHECK(tok == (std::vector<int32_t>{ 0, 1, 2, 0 }));
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    CHECK((p[2].flags & RAD_SP_MASK) != 0);
    CHECK_EQ(p[3].flags & RAD_SP_MASK, 0);
    CHECK(ch.s.take_failed().empty());
    CHECK(ch.s.allows_speculation(1));
    /* Rewound: the grammar accepts the sequence from its start. */
    CHECK_OK(ch.s.accept_token(1, 0));
    CHECK_OK(ch.s.accept_token(1, 1));
    CHECK_OK(ch.s.accept_token(1, 2));

    /* A lazy grammar whose trigger is a draft: the first position is free, the ones after it
     * constrained, so a call that starts mid-window is masked from its first token on. */
    SamplingParams lz = greedy_params();
    lz.grammar = "root ::= \"<call>\" \"a\" \"b\"";
    lz.grammar_lazy = true;
    lz.grammar_triggers = { { 1, "<call>" } };
    Request b = make_req(2, { 5 }, lz);
    const int32_t db[3] = { 4, 0, 5 };                 /* <call>, a, then z: refused */
    b.n_draft = 3;
    b.draft = db;
    CHECK_OK(ch.step({ &b }, favour({ 5, 5, 5, 5 }), &tok));
    CHECK(tok == (std::vector<int32_t>{ 5, 0, 1, 5 }));
    p = ch.s.staged_params(&n);
    CHECK_EQ(p[0].flags & RAD_SP_MASK, 0);
    CHECK((p[1].flags & RAD_SP_MASK) != 0);
    CHECK_EQ(p[3].flags & RAD_SP_MASK, 0);
}

/* A grammar that admits nothing (complete, and no end-of-generation token in the vocabulary)
 * fails its request, not the step: the step returns success, the request is reported once, and
 * the next take is empty. */
TEST(a_dead_grammar_fails_its_request_once_and_not_the_step) {
    SimpleVocab v({ "a", "b" });
    HostChain ch(2, 1, 0, &v);
    NEED_CHAIN(ch)
    SamplingParams sp = greedy_params();
    sp.grammar = "root ::= \"a\"";
    Request a = make_req(9, { 1 }, sp);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, { 0.0f, 5.0f }, &tok));
    CHECK_EQ(tok[0], 0);
    CHECK_OK(ch.s.accept_token(9, 0));
    a.output.push_back(0);
    CHECK_OK(ch.step({ &a }, { 0.0f, 5.0f }, &tok));
    CHECK(ch.s.take_failed() == (std::vector<uint64_t>{ 9 }));
    CHECK(ch.s.take_failed().empty());
}

/* ================================================================== a split vocabulary */

namespace {

/* One rank's sampler over an 80-token vocabulary split at `off`. Pieces "t00".."t79", so a
 * grammar's admissible set is stated by number. */
struct SplitRank {
    Registry reg;
    std::unique_ptr<Builder> b;
    Sampler s;
    SimpleVocab v;
    int st = RAD_OK;
    SplitRank(int rank, int64_t off, int64_t local) {
        for (int i = 0; i < 80; ++i) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "t%02d", i);
            v.add(buf);
        }
        (void)reg.load_plugin(home() + "/kernels/libref.so", 0);
        RadModelMeta meta{};
        meta.arch_id = "sampler_split_mask";
        meta.name = "sampler_split_mask";
        RadBuildCtx bc{};
        bc.rank = rank;
        bc.world_size = 2;
        bc.max_seqs = 1;
        bc.max_tok = 1;
        bc.max_ctx = 64;
        bc.scope = "";
        b = std::make_unique<Builder>(reg, meta, bc);
        SamplerConfig cfg;
        cfg.n_vocab = 80;
        cfg.n_vocab_local = local;
        cfg.max_seqs = 1;
        cfg.world_size = 2;
        cfg.rank = rank;
        cfg.vocab_off = off;
        cfg.max_hist = 16;
        cfg.max_candidates = 8;
        st = s.declare(b.get(), cfg);
        s.attach_vocab(&v);
    }
    /* Global ids whose bit is set in staged row 0, below this rank's width. */
    std::vector<int32_t> staged(int64_t off, int64_t local) const {
        std::vector<int32_t> ids;
        const uint32_t* m = s.staged_mask(0);
        for (int64_t i = 0; i < local; ++i)
            if ((m[i >> 5] >> (i & 31)) & 1u) ids.push_back((int32_t)(off + i));
        return ids;
    }
};

}  /* namespace */

/* Under a split vocabulary the grammar fills a whole-vocabulary mask and each rank stages its own
 * window of it, at a word-aligned offset by copy and at any other by bits. An off-by-one there
 * admits another rank's tokens. The forced reasoning close checks the whole mask, sets only the
 * close's bit on the rank holding it, and leaves the other rank with nothing. */
TEST(each_rank_stages_its_window_of_the_grammar_mask) {
    const int64_t splits[][2] = { { 32, 48 }, { 37, 43 } };
    for (const auto& sp2 : splits) {
        const int64_t off = sp2[0];
        SplitRank r0(0, 0, off), r1(1, off, sp2[1]);
        REQUIRE(r0.st >= 0 && r1.st >= 0);

        SamplingParams sp = greedy_params();
        sp.grammar = "root ::= \"t3\" [0-9]";               /* t30..t39, across both splits */
        Request q = make_req(1, { 70 }, sp);
        std::vector<int32_t> want0, want1;
        for (int32_t id = 30; id <= 39; ++id) (id < off ? want0 : want1).push_back(id);
        for (SplitRank* r : { &r0, &r1 }) {
            std::string err;
            REQUIRE(r->s.begin_request(1, sp, &err) >= 0);
            REQUIRE(r->s.build_step({ &q }) >= 0);
            int64_t n = 0;
            CHECK((r->s.staged_params(&n)[0].flags & RAD_SP_MASK) != 0);
        }
        CHECK(r0.staged(0, off) == want0);
        CHECK(r1.staged(off, sp2[1]) == want1);

        /* A spent thinking budget, the close on rank 1 and admitted by the grammar: only its
         * bit, on rank 1. */
        SamplingParams th = greedy_params();
        th.grammar = "root ::= \"t3\" [0-9] | \"t50\"";
        th.reasoning.close = 50;
        th.reasoning.starts_open = true;
        th.thinking_budget = 0;
        Request c = make_req(2, { 70 }, th);
        for (SplitRank* r : { &r0, &r1 }) {
            std::string err;
            REQUIRE(r->s.begin_request(2, th, &err) >= 0);
            REQUIRE(r->s.build_step({ &c }) >= 0);
            int64_t n = 0;
            CHECK((r->s.staged_params(&n)[0].flags & RAD_SP_MASK) != 0);
        }
        CHECK(r0.staged(0, off).empty());
        CHECK(r1.staged(off, sp2[1]) == (std::vector<int32_t>{ 50 }));

        /* The grammar does not admit the close: the grammar's own masks stand. */
        th.grammar = "root ::= \"t3\" [0-9]";
        Request d = make_req(3, { 70 }, th);
        for (SplitRank* r : { &r0, &r1 }) {
            std::string err;
            REQUIRE(r->s.begin_request(3, th, &err) >= 0);
            REQUIRE(r->s.build_step({ &d }) >= 0);
        }
        CHECK(r0.staged(0, off) == want0);
        CHECK(r1.staged(off, sp2[1]) == want1);
    }
}

/* ================================================================== the vocabulary-parallel merge */

namespace {

/* One libref op called as the engine calls it, through its kernel row. */
int call_op(const Registry& reg, const char* op, std::vector<RadTensor> t,
            std::vector<RadParam> p) {
    const auto* rows = reg.rows_for(op);
    if (!rows || rows->empty()) return RAD_E_NOKERNEL;
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    return (*rows)[0]->info->launch(&a, nullptr);
}

RadTensor tensor2(void* data, uint32_t dtype, int64_t rows, int64_t cols) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = cols;
    t.stride[1] = 1;
    return t;
}

RadParam iparam(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}

}  /* namespace */

/* Under a split vocabulary every sampled token goes top-k per shard, an all-gather, and the merge.
 * That equals top-k over the whole row only if both stages break ties on the lower global id and
 * the merge reads an empty slot (id 0xffffffff) as a hole, not as token 4294967295. Logits from a
 * five-value alphabet make ties everywhere, and a -inf tail makes ties at -inf; rows ask for top_k
 * 0, 1, 3 and 8 and one is greedy (no candidates). The splits are even, uneven, and narrower than
 * the candidate width, which leaves holes the merge must step over. */
TEST(the_split_vocabulary_merge_equals_top_k_over_the_whole_row) {
    Registry reg;
    if (reg.load_plugin(home() + "/kernels/libref.so", 0) < 0) {
        fprintf(stderr, "  SKIP: libref did not load from %s/kernels\n", home().c_str());
        return;
    }
    struct Shape { int64_t V, C; std::vector<int64_t> splits; };
    const Shape shapes[] = { { 37, 8, { 16, 20 } }, { 10, 16, { 4 } } };
    constexpr int64_t M = 6;
    const int64_t pw = (int64_t)(sizeof(RadSampleParams) / 4);
    std::vector<RadSampleParams> prm((size_t)M);
    const int32_t ks[M] = { 0, 1, 3, 8, 0, 0 };
    for (int64_t m = 0; m < M; ++m) {
        std::memset(&prm[(size_t)m], 0, sizeof(RadSampleParams));
        prm[(size_t)m].top_k = ks[m];
        prm[(size_t)m].mask_row = -1;
    }
    prm[4].flags = RAD_SP_GREEDY;
    Mix r(31);
    int holes = 0;
    for (const Shape& sh : shapes) {
        const int64_t V = sh.V, C = sh.C;
        std::vector<float> lg((size_t)(M * V));
        static const float alphabet[] = { -1.0f, 0.0f, 0.5f, 2.0f, 3.0f };
        for (float& x : lg) x = alphabet[r.below(5)];
        for (int64_t i = 4; i < V; ++i) lg[(size_t)(5 * V + i)] = -INFINITY;

        std::vector<int32_t> wi((size_t)(M * C));
        std::vector<float>   wv((size_t)(M * C));
        std::vector<float>   lgc = lg;
        REQUIRE_EQ(call_op(reg, "sample_topk",
                           { tensor2(lgc.data(), RAD_F32, M, V),
                             tensor2(prm.data(), RAD_U32, M, pw),
                             tensor2(wi.data(), RAD_I32, M, C), tensor2(wv.data(), RAD_F32, M, C) },
                           { iparam("M", M), iparam("n_vocab", V) }), RAD_OK);
        CHECK_EQ(wi[(size_t)(4 * C)], -1);                      /* the greedy row: none */
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < (ks[m] > 0 ? std::min<int64_t>(ks[m], C) : C); ++k)
                holes += m != 4 && wi[(size_t)(m * C + k)] < 0;

        for (int64_t split : sh.splits) {
            const int64_t widths[2] = { split, V - split };
            std::vector<uint32_t> gathered((size_t)(M * 2 * C * 2));
            for (int rank = 0; rank < 2; ++rank) {
                const int64_t off = rank ? split : 0, w = widths[rank];
                std::vector<float> shard((size_t)(M * w));
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t i = 0; i < w; ++i)
                        shard[(size_t)(m * w + i)] = lg[(size_t)(m * V + off + i)];
                std::vector<int32_t>  ci((size_t)(M * C));
                std::vector<float>    cv((size_t)(M * C));
                std::vector<uint32_t> pairs((size_t)(M * C * 2));
                REQUIRE_EQ(call_op(reg, "sample_topk",
                                   { tensor2(shard.data(), RAD_F32, M, w),
                                     tensor2(prm.data(), RAD_U32, M, pw),
                                     tensor2(ci.data(), RAD_I32, M, C),
                                     tensor2(cv.data(), RAD_F32, M, C),
                                     tensor2(pairs.data(), RAD_U32, M, 2 * C) },
                                   { iparam("M", M), iparam("n_vocab", w),
                                     iparam("vocab_off", off) }), RAD_OK);
                /* all_gather with row = 2 * n_cand: each position's pairs from both ranks side
                 * by side. */
                for (int64_t m = 0; m < M; ++m)
                    std::memcpy(&gathered[(size_t)(m * 4 * C + rank * 2 * C)],
                                &pairs[(size_t)(m * 2 * C)], (size_t)(2 * C) * 4);
            }
            std::vector<int32_t> mi((size_t)(M * C), 7);
            std::vector<float>   mv((size_t)(M * C), 7.0f);
            REQUIRE_EQ(call_op(reg, "sample_merge_topk",
                               { tensor2(gathered.data(), RAD_U32, M, 4 * C),
                                 tensor2(prm.data(), RAD_U32, M, pw),
                                 tensor2(mi.data(), RAD_I32, M, C),
                                 tensor2(mv.data(), RAD_F32, M, C) },
                               { iparam("M", M), iparam("n_cand", C) }), RAD_OK);
            CHECK(mi == wi);
            CHECK(std::memcmp(mv.data(), wv.data(), mv.size() * 4) == 0);

            /* The same merge into 64-bit ids: a hole is -1, not 4294967295. */
            std::vector<int64_t> mi64((size_t)(M * C), 7);
            REQUIRE_EQ(call_op(reg, "sample_merge_topk",
                               { tensor2(gathered.data(), RAD_U32, M, 4 * C),
                                 tensor2(prm.data(), RAD_U32, M, pw),
                                 tensor2(mi64.data(), RAD_I64, M, C),
                                 tensor2(mv.data(), RAD_F32, M, C) },
                               { iparam("M", M), iparam("n_cand", C) }), RAD_OK);
            bool same = true;
            for (size_t k = 0; k < mi64.size(); ++k) same = same && mi64[k] == (int64_t)wi[k];
            CHECK(same);
        }
    }
    /* Holes inside a row's top_k, past every live candidate, were reached. */
    CHECK(holes > 0);
}

/* ================================================================== logprobs with DRY breakers */

/* A DRY row with a sequence breaker has its raw logprob recovered with the breaker's cap, as the
 * kernel applied it. "\n 22 23" repeats with 21 after it, a repeat of 3 (penalty 0.5 * 1.75); the
 * "\n" two tokens from the end caps it at 2 (penalty 0.5). Tokens 0..20 appear twice, so the
 * frequency penalty puts them at 3 against 21's 5 - 1 - 0.5 = 3.5: 21 is chosen from outside the
 * summary, and undoing the uncapped penalty would be off by 0.375. */
TEST(a_dry_row_with_breakers_has_its_raw_logprob) {
    std::vector<std::string> pieces;
    for (int i = 0; i < 32; ++i) pieces.push_back("t" + std::to_string(i) + ".");
    pieces[24] = "\n";
    SimpleVocab v(pieces);
    HostChain ch(32, 1, 0, &v);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.top_k = 1;
    sp.temp = 0.8f;
    sp.freq_penalty = 1.0f;
    sp.dry_multiplier = 0.5f;
    sp.dry_base = 1.75f;
    sp.dry_allowed_length = 2;
    sp.dry_seq_breakers = { "\n" };
    sp.n_probs = 0;
    Request a = make_req(1, outside_prompt({ 24, 22, 23, 21 }, { 24, 22, 23 }), sp);
    const std::vector<float> lg = outside_logits(5.0f, 5.0f, 0.0f);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    CHECK_EQ(tok[0], 21);
    int64_t n = 0;
    CHECK_EQ(ch.s.staged_params(&n)[0].dry_rep_limit, 2);

    TokenLogprobs lp;
    CHECK_OK(ch.s.logprobs_for(0, tok[0], &lp));
    CHECK_NEAR(lp.logprob, 5.0 - log_sum_exp(lg), 1e-5);
}

/* ================================================================== the narrow candidate plan
 *
 * RADIANCE_SAMPLE_CANDIDATES64 selects a 64-wide candidate plan BESIDE the deployment's own. What
 * these tests pin is the two things that can go wrong and that a "it ran" check would miss:
 *
 *   1. WHICH batches it can serve. A batch with one row the narrow width cannot serve has to keep
 *      the WHOLE batch on the wide plan -- a plan is one set of buffers, so half a batch cannot read
 *      from one width and half from another. top_k = 0 is the row that makes this a correctness
 *      requirement and not an optimisation detail: it means "the whole candidate width", so it
 *      samples from a DIFFERENT SUPPORT at either width.
 *
 *   2. THAT THE TOKEN IS THE SAME. Accepted batches are compared against the wide plan token for
 *      token, on the cases where the width could plausibly matter: ties at the cut, logits so low
 *      that the f32 sum underflows, a grammar mask, and the filter stages that walk the candidate
 *      list in order.
 *
 * Each test declares TWO chains -- one with the flag unset, one with it set -- and compares their
 * outputs. Nothing here observes which internal plan was chosen; the token is the observable, and a
 * width that changed an answer shows up as a token mismatch.
 *
 * The flag is read at declare, so each test sets it around its chain and restores it -- a test that
 * left it set would change what every later test in this file declares. */
namespace {

struct NarrowFlag {
    bool had = false;
    std::string prev;
    explicit NarrowFlag(bool on) {
        if (const char* v = std::getenv("RADIANCE_SAMPLE_CANDIDATES64")) { had = true; prev = v; }
        if (on) setenv("RADIANCE_SAMPLE_CANDIDATES64", "1", 1);
        else    unsetenv("RADIANCE_SAMPLE_CANDIDATES64");
    }
    ~NarrowFlag() {
        if (had) setenv("RADIANCE_SAMPLE_CANDIDATES64", prev.c_str(), 1);
        else     unsetenv("RADIANCE_SAMPLE_CANDIDATES64");
    }
};

/* One step for one request, returning its token, or INT32_MIN if the step failed. */
int32_t one_token(HostChain& ch, Request& r, const std::vector<float>& lg) {
    std::vector<int32_t> tok;
    const int st = ch.step({ &r }, lg, &tok);
    ch.s.end_request(r.id);
    return st == RAD_OK && !tok.empty() ? tok[0] : INT32_MIN;
}

}  /* namespace */

/* top_k 1, 20 AND 64 -- the ends of the range the narrow plan accepts, and the default in the
 * middle -- pick the same token at both widths. The logits are shaped so the answer depends on the
 * cut and not on the argmax: a dominant token is not the point, a tie at rank k is. */
TEST(a_batch_inside_the_narrow_width_picks_the_same_token_as_the_wide_plan) {
    for (int k : { 1, 20, 64 }) {
        SamplingParams sp;
        sp.temp = 1.0f;
        sp.top_k = k;
        sp.seed = 12345;
        /* A strict descending tail except for a TIE at the k-th and (k+1)-th place, which is the
         * only place a width change could reorder anything: ties break on the lower id, so the
         * narrow plan must keep the lower id too. */
        std::vector<float> lg(256, -40.0f);
        for (int i = 0; i < k; ++i) lg[(size_t)i] = 8.0f - 0.01f * (float)i;
        lg[(size_t)k] = lg[(size_t)(k - 1)];

        NarrowFlag off(false);
        HostChain wide(256, 1, 0, nullptr, 1024);
        NEED_CHAIN(wide)
        Request a = make_req(1, { 0 }, sp);
        const int32_t tw = one_token(wide, a, lg);

        NarrowFlag on(true);
        HostChain narrow(256, 1, 0, nullptr, 1024);
        NEED_CHAIN(narrow)
        Request b = make_req(1, { 0 }, sp);
        const int32_t tn = one_token(narrow, b, lg);
        CHECK_EQ(tn, tw);
        CHECK(tn != INT32_MIN);
    }
}

/* FINITE UNDERFLOW: every candidate is so far below the maximum that exp(v - max) underflows to
 * zero in f32. The wide plan sums a thousand of those zeroes and the narrow one sixty-four, and the
 * token must not change -- 0.0f is the exact additive identity, so the sum, and therefore the CDF,
 * is the same value at either width. The pick then falls through every comparison to its last live
 * candidate, at both widths. */
TEST(logits_that_underflow_the_f32_sum_pick_the_same_token_at_either_width) {
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 64;
    sp.seed = 7;
    std::vector<float> lg(256, -3.0e38f);
    lg[0]  = -1.0e38f;               /* the maximum; v - max is a huge negative for the rest */
    lg[63] = -1.0e38f;

    NarrowFlag off(false);
    HostChain wide(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(wide)
    Request a = make_req(1, { 0 }, sp);
    const int32_t tw = one_token(wide, a, lg);

    NarrowFlag on(true);
    HostChain narrow(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(narrow)
    Request b = make_req(1, { 0 }, sp);
    const int32_t tn = one_token(narrow, b, lg);
    CHECK_EQ(tn, tw);
    CHECK(tn >= 0);
}

/* top_k 0 IS THE WHOLE CANDIDATE WIDTH: at 64 it would sample from 64 candidates instead of 1024,
 * which is a different support and a different answer. The token it must return is one only the
 * wider support contains. */
TEST(a_row_at_top_k_zero_can_reach_a_token_beyond_the_narrow_width) {
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 0;
    sp.seed = 3;
    std::vector<float> lg(256, -60.0f);
    for (int i = 0; i < 200; ++i) lg[(size_t)i] = -1.0f - 0.001f * (float)i;
    lg[200] = 20.0f;                /* outside a 64-wide support, inside the wide one, and so far
                                     * above the tail that every other candidate's exp(v - max)
                                     * underflows: the draw is not a matter of luck. */

    NarrowFlag on(true);
    HostChain ch(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(ch)
    Request a = make_req(1, { 0 }, sp);
    /* The winning token is #200 of 256. A step that ran this row on a 64-candidate support could
     * not return it at any seed, so the token itself is the proof that top_k = 0 kept the whole
     * candidate width -- no interior state has to be inspected. */
    CHECK_EQ(one_token(ch, a, lg), 200);
}

/* MORE THAN 64 IS ADMITTED -- the admission bound is max_candidates, not the narrow width -- and
 * the token it returns is one only the wider support contains. Without this the flag would be a
 * silent admission change, which is the one thing it must never be. */
TEST(a_row_above_the_narrow_width_is_admitted_and_reaches_its_own_candidates) {
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 200;
    sp.seed = 9;
    std::vector<float> lg(256, -60.0f);
    for (int i = 0; i < 200; ++i) lg[(size_t)i] = -1.0f - 0.001f * (float)i;
    lg[150] = 20.0f;                /* rank 0, and candidate slot 0 of a 200-wide top-k; dominant
                                     * enough that the draw is deterministic. */

    NarrowFlag on(true);
    HostChain ch(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(ch)
    std::string why;
    CHECK_OK(ch.s.check_request(sp, &why));      /* admitted at all */
    Request a = make_req(1, { 0 }, sp);
    CHECK_EQ(one_token(ch, a, lg), 150);
}

/* MIXED ELIGIBLE/INELIGIBLE: one top_k 8 row and one top_k 0 row in the same batch. The top_k 0
 * row's only viable candidate is token 200, which a 64-wide support cannot hold -- so its token
 * proves the WHOLE batch stayed wide, the eligible row included. A per-row choice would have
 * starved row 1 and returned something else (or -1) while row 0 still looked right. */
TEST(one_ineligible_row_keeps_the_whole_batch_at_the_full_width) {
    SamplingParams tiny;
    tiny.temp = 1.0f;
    tiny.top_k = 8;
    tiny.seed = 11;
    SamplingParams whole;
    whole.temp = 1.0f;
    whole.top_k = 0;
    whole.seed = 12;

    std::vector<float> lg(2 * 256, -60.0f);
    lg[3]  = 20.0f;                   /* row 0: a clear winner inside 8 */
    lg[256 + 200] = 20.0f;            /* row 1: only reachable with the whole width */

    NarrowFlag on(true);
    HostChain ch(256, 2, 0, nullptr, 1024);
    NEED_CHAIN(ch)
    Request a = make_req(1, { 0 }, tiny);
    Request b = make_req(2, { 0 }, whole);
    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a, &b }, lg, &tok));
    CHECK_EQ(tok[0], 3);
    CHECK_EQ(tok[1], 200);
    ch.s.end_request(a.id);
    ch.s.end_request(b.id);
}

/* GREEDY BESIDE STOCHASTIC ON THE NARROW PLAN: a greedy row carries no candidates, so it cannot
 * constrain the width, and the sampled row's token must match the wide plan's for the same seed.
 * This is the mixed-filter case too -- the sampled row runs temperature, top-k, top-p and min-p,
 * all of which walk the candidate list in order. */
TEST(a_greedy_row_beside_a_filtered_sampled_row_matches_the_wide_plan) {
    for (uint64_t seed = 1; seed <= 8; ++seed) {
        SamplingParams g = greedy_params();
        g.seed = seed;
        SamplingParams s;
        s.temp = 0.9f;
        s.top_k = 32;
        s.top_p = 0.9f;
        s.min_p = 0.01f;
        s.seed = seed;

        std::vector<float> lg(2 * 256, -30.0f);
        for (int i = 0; i < 40; ++i) lg[(size_t)i] = 2.0f - 0.05f * (float)i;   /* row 0 */
        for (int i = 0; i < 40; ++i) lg[256 + (size_t)i] = 1.5f - 0.03f * (float)i; /* row 1 */

        NarrowFlag off(false);
        HostChain wide(256, 2, 0, nullptr, 1024);
        NEED_CHAIN(wide)
        Request a = make_req(10 + seed, { 0 }, g);
        Request b = make_req(20 + seed, { 0 }, s);
        std::vector<int32_t> wt;
        CHECK_OK(wide.step({ &a, &b }, lg, &wt));
        wide.s.end_request(a.id);
        wide.s.end_request(b.id);

        NarrowFlag on(true);
        HostChain narrow(256, 2, 0, nullptr, 1024);
        NEED_CHAIN(narrow)
        Request c = make_req(10 + seed, { 0 }, g);
        Request d = make_req(20 + seed, { 0 }, s);
        std::vector<int32_t> nt;
        CHECK_OK(narrow.step({ &c, &d }, lg, &nt));
        CHECK_EQ(nt[0], wt[0]);
        CHECK_EQ(nt[1], wt[1]);
        narrow.s.end_request(c.id);
        narrow.s.end_request(d.id);
    }
}

/* A GRAMMAR MASK AND ITS HIGH ADMISSIBLE ID. The mask is this rank's shard of the vocabulary, and a
 * top-k over a masked row fills its width with the forbidden tail at -infinity. The narrow plan
 * must still return the one admissible token, which sits at id 31 of 32 -- inside the shard and
 * inside the first 64 candidates -- and the wide plan must agree. */
TEST(a_grammar_masked_row_on_the_narrow_plan_keeps_its_only_admissible_token) {
    static SimpleVocab vocab({ "a", "b", "c", "d" });
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 64;
    sp.seed = 21;
    sp.grammar = "root ::= \"d\"";
    std::vector<float> lg(32, -20.0f);
    lg[3] = 1.0f;                    /* the grammar admits "d" = id 3 and nothing else */

    NarrowFlag off(false);
    HostChain wide(32, 1, 0, &vocab, 1024);
    NEED_CHAIN(wide)
    Request a = make_req(1, { 0 }, sp);
    const int32_t tw = one_token(wide, a, lg);

    NarrowFlag on(true);
    HostChain narrow(32, 1, 0, &vocab, 1024);
    NEED_CHAIN(narrow)
    Request b = make_req(1, { 0 }, sp);
    const int32_t tn = one_token(narrow, b, lg);
    CHECK_EQ(tn, 3);
    CHECK_EQ(tn, tw);
}

/* A HIGH TOKEN ID IS NOT A HIGH CANDIDATE SLOT. A grammar admits only token 200 of 256, which is
 * beyond the narrow plan's 64 slots if you confuse the token id with the candidate index, but
 * sample_topk ranks by logit and the masked row has one live candidate, so the admitted token lands
 * in slot 0 and a 64-wide plane holds it. Both widths therefore agree, and this pins that they do:
 * the width bounds SLOTS, and the token ids in them stay global. */
TEST(a_high_token_id_lands_in_a_low_candidate_slot_and_both_widths_agree) {
    /* Every piece is the same length, so no piece is a prefix of another and the grammar's literal
     * names exactly ONE token. A spelling like "t200" beside "t2" would admit the shorter word as a
     * prefix of it, and the mask would stop being a singleton. */
    static SimpleVocab vocab = [] {
        std::vector<std::string> w;
        for (int i = 0; i < 256; ++i) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "<%03d>", i);
            w.push_back(buf);
        }
        return SimpleVocab(std::move(w));
    }();

    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 64;
    sp.seed = 31;
    sp.grammar = "root ::= \"<200>\"";

    std::vector<float> lg(256, 10.0f);   /* the high id is not the largest logit at all */
    lg[0] = 20.0f;

    NarrowFlag off(false);
    HostChain wide(256, 1, 0, &vocab, 1024);
    NEED_CHAIN(wide)
    Request a = make_req(1, { 0 }, sp);
    const int32_t tw = one_token(wide, a, lg);

    NarrowFlag on(true);
    HostChain narrow(256, 1, 0, &vocab, 1024);
    NEED_CHAIN(narrow)
    Request b = make_req(1, { 0 }, sp);
    const int32_t tn = one_token(narrow, b, lg);
    CHECK_EQ(tn, 200);
    CHECK_EQ(tn, tw);
}

/* THE FLAG IS EXACTLY "1". Anything else -- a typo, a word, an empty string -- is OFF, so a
 * deployment that mis-spells it serves the plan it always served rather than half-declaring a
 * second one. The check is the observable: a top_k 0 row can reach token 200 only on the wide
 * plan, and it does at every spelling here. */
TEST(the_narrow_flag_needs_exactly_one) {
    for (const char* v : { "0", "2", "true", "on", "", " 1" }) {
        setenv("RADIANCE_SAMPLE_CANDIDATES64", v, 1);
        SamplingParams sp;
        sp.temp = 1.0f;
        sp.top_k = 0;
        sp.seed = 5;
        std::vector<float> lg(256, -60.0f);
        for (int i = 0; i < 200; ++i) lg[(size_t)i] = -1.0f - 0.001f * (float)i;
        lg[200] = 20.0f;
        HostChain ch(256, 1, 0, nullptr, 1024);
        if (ch.ok) {
            Request a = make_req(1, { 0 }, sp);
            CHECK_EQ(one_token(ch, a, lg), 200);
        }
    }
    unsetenv("RADIANCE_SAMPLE_CANDIDATES64");
}

/* A LOGPROBS ROW KEEPS THE BATCH ON THE WIDE PLAN, and its logprob is the one the wide plan
 * reports. logprobs_for recovers the token's raw logit from the candidate plane the pick drew it
 * at, at the pinned buffer's max_candidates stride; a 64-wide plane would make that read wrong.
 * The token and the logprob are both compared against the flag-off chain. */
TEST(a_logprobs_row_keeps_the_wide_plan_and_reports_the_same_logprob) {
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 20;
    sp.seed = 41;
    sp.n_probs = 5;
    std::vector<float> lg(256, -30.0f);
    for (int i = 0; i < 20; ++i) lg[(size_t)i] = 4.0f - 0.05f * (float)i;
    const double z = log_sum_exp(lg);

    NarrowFlag off(false);
    HostChain wide(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(wide)
    Request a = make_req(1, { 0 }, sp);
    std::vector<int32_t> wt;
    CHECK_OK(wide.step({ &a }, lg, &wt));
    int64_t wn = 0;
    wide.s.staged_params(&wn);
    TokenLogprobs wlp;
    CHECK_OK(wide.s.logprobs_for(0, wt[0], &wlp));
    wide.s.end_request(a.id);

    NarrowFlag on(true);
    HostChain narrow(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(narrow)
    Request b = make_req(1, { 0 }, sp);
    std::vector<int32_t> nt;
    CHECK_OK(narrow.step({ &b }, lg, &nt));
    int64_t nn = 0;
    narrow.s.staged_params(&nn);
    TokenLogprobs nlp;
    CHECK_OK(narrow.s.logprobs_for(0, nt[0], &nlp));
    narrow.s.end_request(b.id);

    CHECK_EQ(nt[0], wt[0]);
    CHECK_NEAR(nlp.logprob, wlp.logprob, 0.0f);
    CHECK_NEAR(nlp.logprob, (float)((double)lg[(size_t)nt[0]] - z), 1e-5);
}

RAD_TEST_MAIN()
