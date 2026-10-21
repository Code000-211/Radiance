/* sample_test.cpp -- the sampler component's tests.
 *
 * Every host reference sampler is checked against a hand-computed answer on a vocabulary small
 * enough to work out on paper, because a reference checked only against itself is a reference
 * that agrees with its own bugs. The device samplers are then checked against THESE by rad-kbench
 * on real hardware (spec §17); this file is the bottom of that stack, so an error here is an
 * error everywhere above it.
 */
#include "rad_test.h"

#include "sample/accept.h"
#include "sample/gbnf.h"
#include "gbnf_oracle.h"
#include "sample/grammar.h"
#include "sample/host_ref.h"
#include "sample/sampler.h"
#include "sample/vocab_view.h"

#include "build/rad_build.h"
#include "plugin/registry.h"
#include "runtime/ctx.h"
#include "rad_sample.h"
#include "../mine/spechub.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;

/* ------------------------------------------------------------------ helpers */

static CandidateSet set_of(const std::vector<float>& logits) {
    CandidateSet s;
    s.from_logits(logits.data(), (int64_t)logits.size());
    return s;
}

/* Logits whose softmax is exactly the given probability vector, so a test can say what it means
 * instead of saying exp() of what it means. */
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
    /* temperature 0 must leave exactly the argmax finite, and it must pick the SAME token
     * host_argmax would -- a temperature-0 request that stopped matching a greedy run would be a
     * silent behaviour change nobody could bisect. */
    auto s = set_of({ 1.0f, 3.0f, 3.0f, 0.0f });
    host_temp(s, 0.0f);
    CHECK(std::isinf(s.c[0].logit) && s.c[0].logit < 0);
    CHECK_EQ(s.c[1].logit, 3.0f);
    CHECK(std::isinf(s.c[2].logit) && s.c[2].logit < 0);
    CHECK(std::isinf(s.c[3].logit) && s.c[3].logit < 0);

    const float raw[] = { 1.0f, 3.0f, 3.0f, 0.0f };
    CHECK_EQ(host_argmax(raw, 4), 1);      /* the tie goes to the lowest id */
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
    /* Three equal logits and k=2: the answer has to be a property of the ids, not of how the
     * sort happened to land, or the device kernel cannot reproduce it. */
    auto s = set_of({ 2.0f, 2.0f, 2.0f, 1.0f });
    host_top_k(s, 2);
    CHECK_EQ(s.c.size(), (size_t)2);
    CHECK_EQ(s.c[0].id, 0);
    CHECK_EQ(s.c[1].id, 1);
}

/* ================================================================== top-p */

TEST(top_p_cuts_at_the_nucleus) {
    /* probs 0.5 0.3 0.15 0.05: cumulative 0.5, 0.8, 0.95, 1.0. The thresholds below sit clear of
     * those cumulants -- see top_p_boundary_is_the_float_sum for why asking about p = 0.8 exactly
     * is asking a question about float addition rather than about the sampler. */
    auto l = logits_for({ 0.5, 0.3, 0.15, 0.05 });

    { auto s = set_of(l); host_top_p(s, 0.4f,  1); CHECK_EQ(s.c.size(), (size_t)1); }
    { auto s = set_of(l); host_top_p(s, 0.7f,  1); CHECK_EQ(s.c.size(), (size_t)2); }
    { auto s = set_of(l); host_top_p(s, 0.85f, 1); CHECK_EQ(s.c.size(), (size_t)3); }
    { auto s = set_of(l); host_top_p(s, 0.99f, 1); CHECK_EQ(s.c.size(), (size_t)4); }
    /* p >= 1 is off entirely. */
    { auto s = set_of(l); host_top_p(s, 1.0f,  1); CHECK_EQ(s.c.size(), (size_t)4); }
}

TEST(top_p_boundary_is_the_float_sum) {
    /* The cut is "the first index at which the RUNNING FLOAT SUM reaches p", and that is a
     * different question from "the first index at which the exact sum reaches p". With probs
     * 0.5 0.3 0.15 0.05 the running sum after two terms is a hair under 0.8, so p = 0.8 keeps
     * THREE candidates, not two.
     *
     * This is pinned rather than papered over because the device kernel accumulates the same way
     * and has to land in the same place: a reference that quietly rounded here would report a
     * mismatch on every near-boundary request and there would be no way to tell that from a real
     * kernel bug. llama.cpp has the same property for the same reason. */
    auto l = logits_for({ 0.5, 0.3, 0.15, 0.05 });
    auto s = set_of(l);
    host_softmax(s, true);
    CHECK(s.c[0].p + s.c[1].p < 0.8f);

    auto t = set_of(l);
    host_top_p(t, 0.8f, 1);
    CHECK_EQ(t.c.size(), (size_t)3);

    /* Where the sum IS exact -- dyadic probabilities -- the boundary behaves as arithmetic says:
     * cumulative 0.5 then 0.75, and p = 0.75 keeps exactly two. */
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
    /* Four identical probabilities. Ties AT the threshold are all kept in principle, but here
     * the cut lands exactly between the second and the third, so the answer is the two lowest
     * ids -- which is the only answer a device kernel can also produce. */
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
    /* Every candidate has surprisal exactly equal to the entropy, so the shifted scores are all
     * zero and the order falls back to the id tie-break. p = 0.5: cumulative 0.25, 0.5, 0.75 --
     * the cut is strictly greater than p, so it takes three. */
    auto s = set_of({ 1.0f, 1.0f, 1.0f, 1.0f });
    host_typical(s, 0.5f, 1);
    CHECK_EQ(s.c.size(), (size_t)3);
    auto ids = ids_of(s);
    CHECK_EQ(ids[0], 0);
    CHECK_EQ(ids[1], 1);
    CHECK_EQ(ids[2], 2);
}

TEST(typical_prefers_the_expected_surprisal) {
    /* probs 0.6 0.2 0.15 0.05. Entropy H = 1.1055 nats; surprisals are 0.511, 1.609, 1.897,
     * 2.996, so the distances to H are 0.595, 0.504, 0.792, 1.890 -- the SECOND candidate is the
     * most typical one, not the first. That is the whole point of the sampler and it is what
     * separates it from top-p. */
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
     * With allowed_length 2, only the two counts >= 2 matter, and each names the token that
     * WOULD extend that repeat: c (reaching 3) and y (reaching 2). At base 2 and multiplier 1
     * the penalties are 2^(3-2) = 2 and 2^(2-2) = 1. */
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
    /* probs 0.5 0.3 0.15 0.05, threshold 0.1: the first three are above it, so everything above
     * the threshold EXCEPT the least probable of them is removed. Probability 1 so the coin is
     * not part of what is being tested here. */
    auto s = set_of(logits_for({ 0.5, 0.3, 0.15, 0.05 }));
    host_xtc(s, 1.0f, 0.1f, 1, /*seed*/ 7, /*pos*/ 0);
    CHECK_EQ(s.c.size(), (size_t)2);
    CHECK_EQ(s.c[0].id, 2);
    CHECK_EQ(s.c[1].id, 3);
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
    /* Different streams at the same position must not be the same draw, or XTC's coin and the
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

    /* The set's ORDER must not change the answer: the pick walks a total order -- descending
     * logit, ties to the lower id -- precisely so that a permuted candidate set draws the same
     * token. */
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
    /* A seeded pick is deterministic, so this is a fixed answer, not a statistical claim -- but
     * an inverse CDF that always returned the first candidate would also be deterministic, and
     * that is the bug this catches. */
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
    /* Temperature 0 collapses every other logit to -inf, so a mask applied AFTER it would leave
     * nothing. The greedy path therefore masks first and takes the argmax of what survives. */
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
    /* Radiance applies temperature BEFORE top-p; llama.cpp applies it after. With logits
     * 3/2/1 and a high temperature the distribution flattens, so a nucleus of 0.5 keeps more
     * candidates than it would on the untempered probabilities. If the two orders ever produce
     * the same set for these numbers, the deviation documented in host_ref.h has quietly
     * stopped being real and rad-kbench would be comparing the wrong thing. */
    auto l = std::vector<float>{ 3.0f, 2.0f, 1.0f };

    auto a = set_of(l);
    host_temp(a, 5.0f);
    host_top_p(a, 0.5f, 1);

    auto b = set_of(l);
    host_top_p(b, 0.5f, 1);
    host_temp(b, 5.0f);

    CHECK(a.c.size() != b.c.size());
}


/* ================================================================== GBNF */

namespace {

/* a b c ab d and an end-of-generation token. Small enough that the admissible set at every
 * position can be worked out by hand, which is the only way this test means anything. */
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

std::vector<int32_t> admitted(const GbnfGrammar& g, int64_t n_tok) {
    std::vector<uint32_t> mask((size_t)bitmask_words(n_tok), 0u);
    std::vector<int32_t> ids;
    if (g.fill_mask(mask.data(), (int64_t)mask.size()) < 0) return ids;
    for (int32_t i = 0; i < (int32_t)n_tok; ++i)
        if (mask[(size_t)(i >> 5)] & (1u << ((uint32_t)i & 31u))) ids.push_back(i);
    return ids;
}

}  /* namespace */

TEST(gbnf_admits_exactly_the_right_tokens_at_each_position) {
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_OK(GbnfGrammar::create("root ::= \"ab\" | \"c\"\n", "root", &v, false, {}, {}, &g, &err));

    /* Position 0: "a" (a prefix of ab), "ab" (the whole alternative) and "c". Not "b", not "d",
     * and not the end-of-generation token, because no parse has completed. */
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

    /* After "b" the parse is complete, so the only admissible token is the end-of-generation
     * one -- a grammar that still admitted text here would produce output past its own grammar. */
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
    /* One digit is a complete match, and more digits are still admissible, so the eog token
     * joins the set rather than replacing it. */
    auto p1 = admitted(*g, v.n_tokens());
    CHECK_EQ(p1.size(), (size_t)4);
    CHECK_EQ(p1[3], 4);
}

/* THE MASK ENGINE AGAINST UPSTREAM'S WALK, bit for bit, over every state a JSON tool call
 * reaches. fill_mask assembles a mask from a plan precomputed per grammar; the oracle rescans every
 * candidate at every depth for every stack, which is what upstream does. Two implementations of
 * one pure function, so the test is equality and nothing else.
 *
 * The vocabulary is shaped like a byte-level BPE on purpose -- long pieces, shared prefixes,
 * multi-byte characters, and pieces that END MID-CHARACTER so the carried-partial paths are
 * reached -- because a vocabulary of single letters exercises neither the trie's depth nor the
 * partial-UTF-8 test at a terminal node. It is still not evidence on its own: gbnf_test holds the
 * engine to the oracle over far more grammars and states, and rad-gbnf-check does the same
 * against a real quarter-million-piece vocabulary. This test is the one that fails in seconds,
 * anywhere. */
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
     * sequence behind and its neighbour resumes it. Byte-level BPE does this constantly. */
    const char* utf8[] = { "\xc3\xa9", "\xc3\xbc", "\xe2\x82\xac", "\xe6\x97\xa5",
                           "\xf0\x9f\x98\x80" };
    for (const char* u : utf8) {
        v.add(u);
        const std::string s(u);
        v.add(s.substr(0, 1));                      /* leaves n_remain > 0 */
        v.add(s.substr(1));                         /* pure continuation bytes */
        if (s.size() > 2) v.add(s.substr(0, 2));
    }
    /* A prefix-heavy family, which is what a trie is for and what a flat scan pays for. */
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

    /* Several independent paths through the grammar, each choosing among the tokens the mask
     * admits, so the states visited are states the grammar can actually be in. */
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
             * keeps going; take eog only when the grammar admits nothing else. */
            std::vector<int32_t> ids;
            for (int32_t i = 0; i < v.n_tokens(); ++i)
                if (i != eog && (fast[(size_t)(i >> 5)] & (1u << ((uint32_t)i & 31u))))
                    ids.push_back(i);
            if (ids.empty()) break;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            CHECK_OK(g->accept_token(ids[(size_t)(rng % ids.size())]));
            if (step > 20) ++states_deep;
        }
    }
    /* If the paths died immediately the comparison above proved nothing, so say what was walked.
     * A regression that makes the grammar dead at step one must fail here, not pass quietly. */
    CHECK(masks > 900);
    CHECK(states_deep > 400);
}

TEST(gbnf_rejects_a_malformed_grammar) {
    SimpleVocab v = tiny_vocab();
    std::string err;

    /* An unterminated string literal. The parser throws internally, as upstream's does, and the
     * failure arrives as a status and a message -- nothing escapes across the boundary. */
    {
        std::unique_ptr<GbnfGrammar> g;
        CHECK_EQ(GbnfGrammar::create("root ::= \"abc\n", "root", &v, false, {}, {}, &g, &err),
                 RAD_E_FORMAT);
        CHECK(!err.empty());
    }
    /* A reference to a rule nobody defined, reported by NAME rather than by index. */
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
    /* A pushdown automaton cannot run this, and hanging on it would be worse than refusing. */
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
 * A grammar on /v1/completions is the client's own text, compiled on the admission thread and
 * walked on the scheduler thread. Each case below is a few bytes to a few hundred kilobytes of
 * grammar that, left unbounded, takes memory, time or stack without limit -- and the whole
 * server with it. If one regresses, the failure is a test that is killed or never finishes, not
 * a CHECK. */

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
     * limit. It is refused on what it would cost, before any of it is allocated. */
    SimpleVocab v = tiny_vocab();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    const std::string big = "root ::= \"" + std::string(100000, 'a') + "\"{1999}\n";
    CHECK_EQ(gbnf_create(big, v, &g, &err), RAD_E_FORMAT);
    CHECK(err.find("MiB") != std::string::npos);
    /* The same literal once is fine: the text itself is not what is bounded. */
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
    /* A 100,000-rule chain has no left recursion, and a recursive check needs a stack frame per
     * rule to find that out. */
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

    /* Every rule reaches the next twice, so a check that revisits what it has already cleared
     * takes 2^60 steps. The closure itself de-duplicates, so the grammar is cheap to run. */
    std::string diamond = "root ::= r0\n";
    for (int i = 0; i < 60; ++i)
        diamond += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + " | r" +
                   std::to_string(i + 1) + "\n";
    diamond += "r60 ::= \"a\"\n";
    CHECK_OK(gbnf_create(diamond, v, &g, &err));
    CHECK_EQ(admitted(*g, v.n_tokens()).size(), (size_t)1);
}

TEST(gbnf_refuses_left_recursion_through_an_indirectly_nullable_rule) {
    /* `("b"?)` can match nothing, so `root` can start with `root` -- but only through a rule
     * that is nullable because ANOTHER rule is. A check that sees only empty alternatives lets
     * this through, and the closure then grows a stack of "c"s until the process dies. */
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
    /* A star over something that can match nothing is a cycle of rules that can come first, but
     * each one is the LAST thing in its alternative, so the stack never grows: the closure meets
     * a stack it has already seen and stops. It must compile and run, not be refused. */
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
    /* Every "a" is ambiguous between two continuations, so the stack set doubles per token. That
     * is a legal grammar and a request that cannot go on past a point; it must end as a status
     * on the request, with the grammar left dead, not as the scheduler thread's memory. */
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
    /* A star over 16,000 alternatives that all match "a". After each "a" every one of them
     * returns to the same continuation, which re-expands into all 16,000 again; expanded once per
     * returning stack that is 256 million stacks per token. It has to be cheap, not refused. */
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
 * recurses once per level and grows its output with the square of the depth, so such a schema
 * exhausts the memory or the stack of the thread admitting it; it is refused on its depth first. */
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

/* THE BOUNDED MACHINERY AGAINST UPSTREAM'S WALK, over the grammar shapes the bounds were built
 * around: a star over something nullable, a wide rule whose alternatives converge (wider than the
 * 64 stacks the automaton memoises a firing pattern for, and the 16 past which a set is indexed),
 * an ambiguous rule whose stack set grows per token, nesting, and a char class under a
 * repetition. Sharing explored stacks across closures, advancing matching stacks together and
 * merging successor states are each claimed to give the same set as upstream's walk, and this is
 * where that is held to it -- the oracle takes none of those paths through the automaton. The
 * shapes are kept small because the oracle re-expands a converging state once per stack per code
 * point. */
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
            /* `h` follows the same path through accept_str, which advances the whole stack set
             * one code point at a time as upstream's walk does, so the two masks agreeing
             * holds accept_token's own path to the same answer. */
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
                std::vector<int32_t> ids;
                for (int32_t i = 0; i < v.n_tokens(); ++i)
                    if (i != eog && (fast[(size_t)(i >> 5)] & (1u << ((uint32_t)i & 31u))))
                        ids.push_back(i);
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

TEST(grammar_backend_registry_ships_gbnf_and_refuses_unknown_names) {
    SimpleVocab v = tiny_vocab();

    GrammarSpec spec;
    spec.text = "root ::= \"c\"\n";
    std::unique_ptr<GrammarBackend> b;
    std::string err;
    CHECK_OK(grammar_create(spec, v, &b, &err));
    CHECK_EQ(std::string(b->name()), std::string("gbnf"));
    CHECK(b->constrains());
}

TEST(json_schema_compiles_to_a_usable_grammar) {
    std::string gbnf, err;
    const int st = grammar_from_json_schema(
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},"
        "\"required\":[\"a\"]}", "root", &gbnf, &err);
    CHECK_OK(st);
    CHECK(gbnf.find("root ::=") != std::string::npos);

    /* The compiled grammar has to actually parse: a schema converter whose output GBNF does not
     * load is a converter nobody would notice was broken until a request arrived. */
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

/* The refusals vendor/patches/0004 adds, and the hang 0003 closes. Upstream accepts every one of
 * these inputs and then gets them wrong -- silently dropping the constraint, reaching the network,
 * emitting a grammar that will not load, or spinning forever. A `pattern` and a `$ref` are client
 * data on /v1/chat/completions and /v1/completions alike, so each of these is a request shape. */
TEST(json_schema_refuses_what_it_cannot_actually_enforce) {
    std::string gbnf, err;

    /* "not" has no case in the converter, so falling through leaves it as "accepts any value" --
     * a constraint the caller asked for, did not get, and was told nothing about. */
    CHECK(grammar_from_json_schema("{\"not\":{\"type\":\"string\"}}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* A sampler that reaches the network to decide what tokens are legal is not one we ship. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"$ref\":\"https://example.com/s.json#/x\"}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* GBNF cannot spell \d. Copying it through returns success here and then produces a grammar
     * that fails to LOAD, which the caller sees as an internal error on a schema we accepted. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"^\\\\d+$\"}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* A stray ']' is in NON_LITERAL_SET but claimed by no arm of the pattern scanner, so the
     * literal arm breaks without consuming it and the loop reconsiders the same byte forever.
     * If this regresses the failure mode is a hung test rather than a failing one -- the same
     * hang a server thread takes on the request that carries the pattern. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"^a]$\"}", "root", &gbnf, &err) < 0);
    CHECK(!err.empty());

    /* An empty pattern would run front()/back() on an empty string. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"\"}", "root", &gbnf, &err) < 0);

    /* A leading quantifier would run back() on an empty vector. */
    err.clear();
    CHECK(grammar_from_json_schema(
              "{\"type\":\"string\",\"pattern\":\"^*$\"}", "root", &gbnf, &err) < 0);

    /* And the ordinary case still compiles, so none of the above is refusing everything. */
    err.clear();
    CHECK_OK(grammar_from_json_schema(
                 "{\"type\":\"string\",\"pattern\":\"^[a-z]+$\"}", "root", &gbnf, &err));
    CHECK(gbnf.find("root ::=") != std::string::npos);
}

/* ================================================================== speculative acceptance */

TEST(greedy_acceptance_stops_at_the_first_disagreement) {
    /* draft   5 7 9
     * target  5 7 3 11
     * The first two agree, the third does not, so two are accepted and the emitted run is the
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

TEST(stochastic_acceptance_always_takes_a_token_the_target_is_certain_of) {
    /* p(x) = 1 for the drafted token, so min(1, p/q) = 1 and no uniform can reject it. */
    const int64_t V = 4;
    const float target[] = { 0.0f, 1.0f, 0.0f, 0.0f,      /* position 0 */
                             0.0f, 0.0f, 1.0f, 0.0f };    /* the bonus position */
    const int32_t draft[] = { 1 };

    SpecAcceptParams p;
    p.n_draft = 1;
    p.n_vocab = V;
    p.seed    = 777;

    for (uint64_t pos = 0; pos < 16; ++pos) {
        p.pos = pos;
        int32_t out[2] = { -1, -1 };
        int32_t n = -1;
        CHECK_OK(accept_stochastic(p, draft, target, nullptr, out, &n));
        CHECK_EQ(n, 1);
        CHECK_EQ(out[0], 1);
        CHECK_EQ(out[1], 2);
    }
}

TEST(stochastic_acceptance_matches_the_hand_worked_rule) {
    /* Target at position 0: 0.1 0.6 0.2 0.1. Greedy drafter proposing token 1, so q is a point
     * mass and the accept probability is p(1) = 0.6.
     *
     * On acceptance the bonus comes from position 1, which is a point mass on token 2.
     * On rejection the residual is max(0, p - q) = 0.1 0 0.2 0.1, total 0.4, walked in ascending
     * token id: the draw lands on 0 below 0.1, on 2 below 0.3, on 3 otherwise.
     *
     * The expected answer is computed here from those numbers rather than from the code, which is
     * the only way this test can disagree with the implementation. */
    const int64_t V = 4;
    const float target[] = { 0.1f, 0.6f, 0.2f, 0.1f,
                             0.0f, 0.0f, 1.0f, 0.0f };
    const int32_t draft[] = { 1 };

    SpecAcceptParams p;
    p.n_draft = 1;
    p.n_vocab = V;
    p.seed    = 20260904;

    int n_acc = 0, n_rej = 0;
    for (uint64_t pos = 0; pos < 64; ++pos) {
        p.pos = pos;
        int32_t out[2] = { -1, -1 };
        int32_t n = -1;
        CHECK_OK(accept_stochastic(p, draft, target, nullptr, out, &n));

        const float u = rng_uniform(p.seed, pos, RAD_RNG_ACCEPT);
        if (u < 0.6f) {
            ++n_acc;
            CHECK_EQ(n, 1);
            CHECK_EQ(out[0], 1);
            CHECK_EQ(out[1], 2);
        } else {
            ++n_rej;
            CHECK_EQ(n, 0);
            const double want = (double)rng_uniform(p.seed, pos, RAD_RNG_RESID) * 0.4;
            const int32_t expect = want <= 0.1 ? 0 : (want <= 0.30000001 ? 2 : 3);
            CHECK_EQ(out[0], expect);
        }
    }
    /* Both branches must actually be exercised, or the loop above proved nothing about one. */
    CHECK(n_acc > 0);
    CHECK(n_rej > 0);
}

TEST(stochastic_acceptance_with_an_explicit_draft_distribution) {
    /* q(1) = 0.5, p(1) = 0.25, so the accept probability is p/q = 0.5 rather than p. Getting
     * this ratio backwards is the classic speculative-decoding bug and it shows up as an
     * acceptance rate that looks fine and an output distribution that is not the target's. */
    const int64_t V = 2;
    const float target[] = { 0.75f, 0.25f,
                             1.0f,  0.0f };
    const float dprobs[] = { 0.5f,  0.5f };
    const int32_t draft[] = { 1 };

    SpecAcceptParams p;
    p.n_draft = 1;
    p.n_vocab = V;
    p.seed    = 5150;

    for (uint64_t pos = 0; pos < 64; ++pos) {
        p.pos = pos;
        int32_t out[2] = { -1, -1 };
        int32_t n = -1;
        CHECK_OK(accept_stochastic(p, draft, target, dprobs, out, &n));

        const float u = rng_uniform(p.seed, pos, RAD_RNG_ACCEPT);
        if ((double)u < 0.25 / 0.5) {
            CHECK_EQ(n, 1);
            CHECK_EQ(out[0], 1);
            CHECK_EQ(out[1], 0);
        } else {
            CHECK_EQ(n, 0);
            CHECK_EQ(out[0], 0);       /* residual max(0, 0.75-0.5) is all on token 0 */
        }
    }
}

TEST(accept_sequence_refuses_the_wrong_inputs_for_the_mode) {
    SpecAcceptParams p;
    p.n_draft = 1;
    p.n_vocab = 2;
    p.greedy  = true;

    const int32_t draft[] = { 0 };
    int32_t out[2];
    int32_t n = 0;
    /* Greedy acceptance without the target's argmax must NOT quietly fall back to rejection
     * sampling: that would change the emitted distribution with nothing saying so. */
    CHECK_EQ(accept_sequence(p, draft, nullptr, nullptr, nullptr, out, &n), RAD_E_INVAL);

    p.greedy = false;
    const int32_t argmax[] = { 0, 0 };
    CHECK_EQ(accept_sequence(p, draft, argmax, nullptr, nullptr, out, &n), RAD_E_INVAL);
}

/* ================================================================== the sampler's front door
 *
 * ADMISSION AND PER-REQUEST STATE, which is everything the Sampler does before a device is
 * involved. check_request is what makes a bad request fail AT THE DOOR rather than three steps
 * in, and begin_request/end_request is what makes a grammar exist for exactly as long as the
 * request does. Both run on the scheduler thread and both are pure host work; everywhere else the
 * class is reached only through the declare phase, so these tests are the ones that call it
 * without a builder and a card.
 *
 * An undeclared Sampler resolves no kernels, so the parameter-range half of check_request is
 * exactly what is reachable here -- and it is the half that runs first and refuses first.
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
 * than the buffers hold is refused BY NAME at admission rather than silently narrowed -- a
 * request whose top_k quietly became something else is a request whose output changed with
 * nothing to say so. */
TEST(admission_refuses_a_top_k_wider_than_the_candidate_buffers) {
    Sampler s;
    std::string why;
    SamplingParams sp = plain_params();
    sp.top_k = 1 << 20;
    CHECK_EQ(s.check_request(sp, &why), RAD_E_INVAL);
    CHECK(why.find("--max-candidates") != std::string::npos);
}

/* An unresolved sampler kernel is a NAMED refusal and not a silent skip, for the same reason: a
 * request whose min_p stopped being applied is a request whose output changed. An undeclared
 * Sampler has resolved nothing, so every request that needs a stage is refused and the message
 * says which one. */
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

/* A grammar needs the vocabulary the grammar walks, and the vocabulary is attached after the
 * tokeniser loads. Asked for one before that, the answer is a named state error rather than a
 * null dereference on the step path. */
TEST(a_grammar_before_the_vocabulary_is_a_named_refusal) {
    Sampler s;
    std::string err;
    SamplingParams sp = plain_params();
    sp.grammar = "root ::= \"a\"";
    /* Admission reaches the vocabulary check only once every stage it needs has a kernel, which
     * an undeclared sampler cannot give it -- so the refusal here is begin_request's, which is
     * the call that would actually dereference it. */
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

/* THE LIFETIME IS THE REQUEST'S. begin_request is not idempotent -- calling it twice rebuilds
 * the automaton and throws away everything it had accepted, which unconstrains a generation from
 * its second token onward -- so has_request is what the step loop asks before calling it, and it
 * has to be right in both directions. */
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
    /* Ending a request that is not there is not an error -- the step loop reaches a request on
     * every step it is scheduled in and cancellation can have got there first. */
    s.end_request(7);
    s.end_request(999);
    CHECK(s.has_request(8));
}

/* A request with no grammar has nothing to advance, and accept_token must say so rather than
 * fail: it is called on every emitted token of every request. */
TEST(accepting_a_token_against_no_grammar_is_not_an_error) {
    Sampler s;
    SimpleVocab v = tiny_vocab();
    s.attach_vocab(&v);
    std::string err;
    CHECK_OK(s.begin_request(3, plain_params(), &err));
    CHECK_OK(s.accept_token(3, 0));
    /* And a request this sampler has never seen is the same answer, for the same reason. */
    CHECK_OK(s.accept_token(404, 0));
}

/* SPECULATION IS ALLOWED UNLESS A GRAMMAR CANNOT REWIND. Masking draft position i means walking
 * the grammar over positions 0..i-1 and rewinding afterwards, so a backend without that decodes
 * one token at a time -- rather than emitting drafts the grammar forbids. An unconstrained
 * request has no such problem and must not be slowed to a single token a step. */
TEST(speculation_is_allowed_for_a_request_with_no_grammar) {
    Sampler s;
    SimpleVocab v = tiny_vocab();
    s.attach_vocab(&v);
    std::string err;
    CHECK_OK(s.begin_request(4, plain_params(), &err));
    CHECK(s.allows_speculation(4));
    /* And for a request this sampler has no state for at all, which is what a request that never
     * needed any looks like. */
    CHECK(s.allows_speculation(405));
}

/* A GRAMMAR REQUEST GETS STATE, ADVANCES, AND STILL SPECULATES. The GBNF backend snapshots, so a
 * constrained request is not silently demoted to one token a step -- which is worth several times
 * the per-token cost of the mask. */
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
 * cancels them: reporting the same failure on the next step would cancel a request that is no
 * longer there. */
TEST(the_failed_list_is_taken_once_and_is_empty_on_a_quiet_step) {
    Sampler s;
    CHECK(s.take_failed().empty());
    CHECK(s.take_failed().empty());
}

/* ================================================================== the RNG is the device's */

/* THE HOST REFERENCE DRAWS abi/rad_sample.h's UNIFORM, bit for bit: the pick on the seed itself and
 * XTC's coin on its key -- exactly what sample_pick and sample_xtc draw. A reference with its own
 * generator agrees with the device on no seeded draw at all. */
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

/* THE SCAN IS BOUNDED THE WAY UPSTREAM BOUNDS IT, plus a count: a breaker longer than 40 bytes is
 * cut (at a character boundary), a tail longer than 20 tokens is cut, and more than
 * kDryMaxBreakers breakers are refused -- each one is a pass over the whole vocabulary on the
 * scheduler thread, where every other sequence waits for it. */
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

    /* A two-byte character straddling the 40-byte cut is dropped whole, not split: the prefix
     * that remains is 39 bytes of ASCII, which still tokenises. */
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

    /* DRY off, the list is never scanned and costs nothing, so it is not a reason to refuse. */
    sp.dry_multiplier = 0.0f;
    why.clear();
    const int st = s.check_request(sp, &why);
    CHECK(why.find("dry_sequence_breakers") == std::string::npos);
    (void)st;
}

/* ================================================================== the chain, end to end
 *
 * THE REAL SAMPLER, DECLARED THROUGH THE REAL BUILDER AND RUN ON THE HOST BACKEND BY libref. Every
 * defect this block pins is invisible to the per-op tests, because each op can do what its schema
 * says while the driver stages, or issues, something else: a history offset the ops read as
 * row-relative, a top-k that is never issued, a pick that overwrites a greedy row's argmax, DRY
 * that never switches on. Only the staging and the issue order together decide the token.
 *
 * libref serves the HOST domain, so every op is moved there after the context is built; on this
 * backend a device buffer is host memory, which is what lets the test lay the logits in and read
 * the tokens back directly. */
namespace {

std::string chain_home() {
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "radiance_home";
}

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
        if (reg.load_plugin(chain_home() + "/kernels/libref.so", 0) < 0) return;
        if ((std::getenv("RADIANCE_SPECHUB") || std::getenv("RADIANCE_MTP_COUPLED")) &&
            reg.load_plugin(chain_home() + "/kernels/mine_ref.so", 0) < 0) return;
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
        /* The pass body is the sampler chain, whose host ops carry each request's sampling state:
         * two steps with one batch shape are not one pass. */
        cd.record_passes = false;
        if (ctx.init(cd) < 0) return;
        for (size_t i = 1; i < prog.ops.size(); ++i)
            if (ctx.set_op_domain((rad_op)i, RAD_DOMAIN_HOST) < 0) return;
        ok = true;
    }

    /* One step: begin every request not yet seen, stage the step, lay `lg` -- n_vocab floats per
     * sampled position, in staging order -- into the logits plane, run the chain and read the
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
                ::radtest::current(), chain_home().c_str());                          \
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

}  /* namespace */

/* A SECOND SAMPLED ROW IS PENALISED TOO. Two greedy sequences whose histories both hold token 0,
 * with a repetition penalty that takes token 0 below token 1: each row must pick 1. The history
 * offset each row is staged with is an offset within its own row, so row 1 reads row 1. */
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

/* A DRAFT POSITION IS PENALISED AGAINST THE DRAFTS IN FRONT OF IT. The token sampled at draft
 * position k is emitted only if drafts 0..k-1 are, so they are part of the output it is judged
 * against -- exactly as they would be one token at a time. */
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

/* A GREEDY ROW KEEPS ITS ARGMAX BESIDE A SAMPLED ROW. The batch mixes both paths, so the argmax and
 * the candidate chain each run over every row; the greedy row's logits are nearly flat, so a draw
 * over them would land on its argmax only about a quarter of the time. */
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

/* top_k = 0 IS THE WHOLE CANDIDATE WIDTH, NOT NO CANDIDATES. The candidate planes are transient,
 * so a step that skipped the top-k would hand the pick whatever they last held. Two steps with
 * different winners: each must be sampled from its own logits. */
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

/* ================================================================== the narrow candidate plan
 *
 * RADIANCE_SAMPLE_CANDIDATES64 selects a 64-wide candidate plan BESIDE the deployment's own. What
 * these tests pin is the two things that can go wrong and that a "it ran" check would miss:
 *
 *   1. WHICH batches it can serve. A batch with one row the narrow width cannot serve has to keep
 *      the WHOLE batch on the wide plan -- a plan is one set of buffers, so half a batch cannot
 *      read from one width and half from another. top_k = 0 is the row that makes this a
 *      correctness requirement and not an optimisation detail: it means "the whole candidate
 *      width", so it samples from a DIFFERENT SUPPORT at either width.
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
 * is the same value at either width. The pick then falls through every comparison to its last
 * live candidate, at both widths. */
TEST(logits_that_underflow_the_f32_sum_pick_the_same_token_at_either_width) {
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 0;                     /* rewritten below: this test needs a bounded k */
    sp.top_k = 64;
    sp.seed = 7;
    std::vector<float> lg(256, -3.0e38f);
    lg[0] = -1.0e38f;                 /* the maximum; v - max is a huge negative for the rest */
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
    std::vector<float> lg(256, -40.0f);
    for (int i = 0; i < 200; ++i) lg[(size_t)i] = -1.0f - 0.001f * (float)i;
    lg[200] = 5.0f;                  /* outside a 64-wide support, inside the wide one */

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
    std::vector<float> lg(256, -40.0f);
    for (int i = 0; i < 200; ++i) lg[(size_t)i] = -1.0f - 0.001f * (float)i;
    lg[150] = 5.0f;                 /* rank 0, and candidate slot 0 of a 200-wide top-k */

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

    std::vector<float> lg(2 * 256, -40.0f);
    lg[3]  = 5.0f;                    /* row 0: a clear winner inside 8 */
    lg[256 + 200] = 5.0f;             /* row 1: only reachable with the whole width */

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

/* A HIGH TOKEN ID IS NOT A HIGH CANDIDATE SLOT. The grammar admits only token 200 of 256, which is
 * beyond the narrow plan's 64 slots if you confuse the token id with the candidate index -- but
 * sample_topk ranks by logit and the masked row has one live candidate, so the admitted token lands
 * in slot 0 and a 64-wide plane holds it. The wide plan and the narrow plan therefore agree, and
 * this test pins that they do: the width bounds SLOTS, and the token ids in them stay global. */
TEST(a_high_token_id_lands_in_a_low_candidate_slot_and_both_widths_agree) {
    /* A vocabulary whose only token matching the grammar is a high id. SimpleVocab is a list of
     * strings, so the grammar names the 201st. */
    std::vector<std::string> words;
    for (int i = 0; i < 256; ++i) words.push_back("t" + std::to_string(i));
    static SimpleVocab vocab(words);

    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 64;
    sp.seed = 31;
    sp.grammar = "root ::= \"t200\"";

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

/* A STEP THAT COULD FIT THE NARROW WIDTH MUST NOT CONSTRAIN THE NEXT ONE THAT CANNOT. Alternating a
 * top_k 4 step with a top_k 0 step: the top_k 0 step's winning token is #200 of 256, which only the
 * full width can reach. A plan cached in a member that build_step never revisits would return a
 * different token on that round -- so the token, not any interior flag, is what this checks. */
TEST(the_candidate_width_a_step_used_does_not_carry_into_the_next) {
    SamplingParams bounded;
    bounded.temp = 1.0f;
    bounded.top_k = 4;
    bounded.seed = 5;
    SamplingParams whole;
    whole.temp = 1.0f;
    whole.top_k = 0;
    whole.seed = 6;

    NarrowFlag on(true);
    HostChain ch(256, 1, 0, nullptr, 1024);
    NEED_CHAIN(ch)
    std::vector<int32_t> lg(256, -40.0f);
    lg[1] = 4.0f;
    lg[200] = 7.0f;               /* reachable only at full width */
    std::vector<int32_t> tok;

    for (int round = 0; round < 3; ++round) {
        Request a = make_req(100 + round, { 0 }, bounded);
        CHECK_OK(ch.step({ &a }, lg, &tok));
        CHECK_EQ(tok[0], 1);      /* top_k 4 keeps id 1; id 200 is outside it */
        ch.s.end_request(a.id);

        Request b = make_req(200 + round, { 0 }, whole);
        CHECK_OK(ch.step({ &b }, lg, &tok));
        CHECK_EQ(tok[0], 200);    /* the whole width reaches id 200 */
        ch.s.end_request(b.id);
    }
}

/* repeat_last_n = 0 TURNS THE PENALTIES OFF, as it does in llama.cpp -- it is not "the whole
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

/* DRY RUNS WITH NO BREAKERS, as llama.cpp's does. History "a b c a b": c would extend the repeat
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

/* THE BREAKERS REACH THE KERNEL, both halves of them. The vocabulary is b, \n, c, d, y, and "\n" is
 * the breaker.
 *
 * History "b \n c d y b \n c d": the suffix "b \n c d" repeats, and y followed it, so y would
 * extend a repeat of 4 -- a penalty of 2^(4-2) = 4 with no breakers. The \n inside the suffix caps
 * the repeat DRY may count at 2, so with the breaker the penalty is 2^0 = 1.
 *
 * History "c d \n c d": the continuation of "c d" is "\n" itself, which is a breaker and is never
 * penalised. */
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
    CHECK_EQ(tok[0], 0);                  /* without the breaker it would have been */
}

/* THE HOST REFERENCE AND THE CHAIN DRAW THE SAME TOKEN. Same uniform, same walk order -- the
 * candidate set in descending logit order -- so over a distribution with no near-ties at the draw
 * the two agree token for token. */
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

/* OVER A SPLIT VOCABULARY A GREEDY ROW IS A TOP-1 CHAIN ROW, and it is staged without the greedy
 * flag: every candidate stage skips a flagged row, and this one has to pass through top-k, the
 * gather and the merge to become a global token id. */
TEST(a_greedy_row_over_a_split_vocabulary_is_staged_as_a_chain_row) {
    /* Staging only: nothing runs, so the ops need not resolve, but with libref loaded they do and
     * declare has nothing to report. */
    Registry reg;
    (void)reg.load_plugin(chain_home() + "/kernels/libref.so", 0);
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

/* TRIGGERS ARE RESOLVED AT ADMISSION, with special tokens recognised: a word that is one token
 * becomes an id, a longer word is scanned for as bytes, a token trigger that is not one token is a
 * bad request, and a regular expression is refused by name -- it would be matched over the reply
 * once per token on the scheduler thread. */
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

/* A THINKING BUDGET CLOSES THE REASONING BLOCK WHERE THE BODY REACHES IT, and it is walked over the
 * speculative window as a grammar is: position k counts the committed reply and drafts 0..k-1, so
 * a window that crosses the budget is forced from the first position past it, and a draft that
 * closes the block itself frees every position behind it. Token 1 wins every row's logits, so a
 * row that comes back 5 was forced. */
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

/* THE COMMITTED REPLY IS COUNTED ACROSS STEPS, and only from where the block opens: a reply that
 * opens its own block counts from its opener, and one that starts any other way has no block to
 * budget. Sampled, not greedy, so the mask is what decides and not an argmax that happens to
 * agree. */
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

/* A GRAMMAR KEEPS THE LAST WORD. The close is forced only where the grammar admits it; where it
 * does not, the row is the grammar's, and the budget waits for a position the grammar allows. */
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

namespace {
struct ScopedSpecHub {
    const bool had_enable = std::getenv("RADIANCE_SPECHUB") != nullptr;
    const bool had_verify = std::getenv("RADIANCE_SPECHUB_VERIFY") != nullptr;
    const std::string enable = had_enable ? std::getenv("RADIANCE_SPECHUB") : "";
    const std::string verify = had_verify ? std::getenv("RADIANCE_SPECHUB_VERIFY") : "";
    ScopedSpecHub() {
        setenv("RADIANCE_SPECHUB", "1", 1);
        setenv("RADIANCE_SPECHUB_VERIFY", "transport", 1);
    }
    ~ScopedSpecHub() {
        if (had_enable) setenv("RADIANCE_SPECHUB", enable.c_str(), 1);
        else unsetenv("RADIANCE_SPECHUB");
        if (had_verify) setenv("RADIANCE_SPECHUB_VERIFY", verify.c_str(), 1);
        else unsetenv("RADIANCE_SPECHUB_VERIFY");
    }
};
RadSpechubState hub_state(const Request& r) {
    RadSpechubState s{};
    s.draw = {r.sp.seed, r.output.size(), r.id};
    s.status = 1; s.root = r.output.back(); s.count = 3;
    s.pair[0] = 0; s.pair[1] = 1;
    s.ids[0] = 0; s.ids[1] = 1; s.ids[2] = 2;
    s.q[0] = 0.5; s.q[1] = 0.3; s.q[2] = 0.2;
    return s;
}
std::vector<float> point_rows(std::initializer_list<int> tokens) {
    std::vector<float> logits(tokens.size() * 8, -INFINITY);
    size_t row = 0;
    for (int token : tokens) logits[row++ * 8 + static_cast<size_t>(token)] = 0.0f;
    return logits;
}
} // namespace

TEST(spechub_explicit_zero_runs_ordinary_sampler) {
    ScopedSpecHub env;
    setenv("RADIANCE_SPECHUB", "0", 1);
    unsetenv("RADIANCE_SPECHUB_VERIFY");
    HostChain ch(8, 1, 0);
    REQUIRE(ch.ok);
    Request a = make_req(8001, {7}, greedy_params());
    std::vector<int32_t> tokens;
    REQUIRE_OK(ch.step({&a}, point_rows({6}), &tokens));
    REQUIRE_EQ(tokens.size(), (size_t)1);
    CHECK_EQ(tokens[0], 6);
}

TEST(spechub_admission_refuses_unsupported_target_settings) {
    ScopedSpecHub enabled;
    HostChain ch(8, 1, 2);
    REQUIRE(ch.ok); // Missing SpecHub oracle is a failure, never a skip.
    SamplingParams sp;
    sp.temp = 1.0f; sp.top_k = 8;
    std::string why;
    CHECK_OK(ch.s.check_request(sp, &why));
    for (int which = 0; which < 5; ++which) {
        auto unsupported = sp;
        if (which == 0) unsupported.temp = 0.0f;
        if (which == 1) unsupported.top_k = 0;
        if (which == 2) unsupported.grammar = "root ::= \"a\"";
        if (which == 3) unsupported.xtc_probability = 0.5f;
        if (which == 4) unsupported.thinking_budget = 0;
        CHECK_EQ(ch.s.check_request(unsupported, &why), RAD_E_UNSUPPORTED);
        CHECK(why.find("SpecHub") != std::string::npos);
    }
}

TEST(spechub_live_sampler_preserves_full_target_and_sequence_coordinates) {
    ScopedSpecHub enabled;
    HostChain ch(8, 2, 2);
    REQUIRE(ch.ok);
    SamplingParams sp;
    sp.temp = 1.0f; sp.top_k = 8; sp.seed = 37;
    Request a = make_req(8101, {7}, sp), b = make_req(8102, {6}, sp);
    a.output = {4, 2}; b.output = {2};
    const int32_t draft[] = {0, 1}, parent[] = {-1, 0, 0}, depth[] = {0, 1, 1};
    for (Request* r : {&a, &b}) {
        r->n_draft = 2; r->draft = draft; r->tree_parent = parent; r->tree_depth = depth;
        REQUIRE_OK(ch.s.begin_request(r->id, r->sp, nullptr));
        REQUIRE_OK(ch.s.set_spechub(r->id, hub_state(*r)));
    }
    std::vector<int32_t> tokens;
    // Both root target supports lie COMPLETELY outside q; neither residual may be narrowed to q.
    REQUIRE_OK(ch.step({&a, &b}, point_rows({3, 4, 5, 6, 7, 4}), &tokens));
    REQUIRE_EQ(tokens.size(), (size_t)6);
    CHECK_EQ(tokens[0], 3); CHECK_EQ(tokens[3], 6);
    int64_t count = 0;
    const auto* params = ch.s.staged_params(&count);
    REQUIRE_EQ(count, (int64_t)6);
    REQUIRE(params);
    CHECK_EQ(params[0].pos, (uint64_t)2);
    CHECK_EQ(params[1].pos, (uint64_t)3); CHECK_EQ(params[2].pos, (uint64_t)3);
    CHECK_EQ(params[3].pos, (uint64_t)1);
    CHECK_EQ(params[4].pos, (uint64_t)2); CHECK_EQ(params[5].pos, (uint64_t)2);
    auto state = hub_state(a);
    for (int which = 0; which < 4; ++which) {
        auto bad = state;
        if (which == 0) ++bad.draw.pos;
        if (which == 1) ++bad.root;
        if (which == 2) bad.pair[0] = 2;
        if (which == 3) ++bad.draw.seed;
        REQUIRE_OK(ch.s.set_spechub(a.id, bad));
        CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE);
    }
    auto wrong_request = state;
    wrong_request.draw.seq = b.id;
    CHECK_EQ(ch.s.set_spechub(a.id, wrong_request), RAD_E_STATE);
    ch.s.end_request(a.id);
    REQUIRE_OK(ch.s.begin_request(a.id, a.sp, nullptr));
    CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE); // Recycled identity inherits no proposal.
    REQUIRE_OK(ch.s.set_spechub(a.id, state));
    CHECK_OK(ch.s.build_step({&a}));
}

TEST(spechub_truncated_pair_keeps_root_law_and_zero_draft_uses_ordinary_target) {
    ScopedSpecHub enabled;
    HostChain ch(8, 1, 2);
    REQUIRE(ch.ok);
    SamplingParams sp;
    sp.temp = 1.0f; sp.top_k = 8; sp.seed = 53;
    Request a = make_req(8201, {7}, sp);
    a.output = {2};
    const int32_t draft[] = {0};
    a.n_draft = 1; a.draft = draft; // Scheduler retained only the first sibling.
    REQUIRE_OK(ch.s.begin_request(a.id, a.sp, nullptr));
    REQUIRE_OK(ch.s.set_spechub(a.id, hub_state(a))); // Complete (0,1) Q metadata survives.
    std::vector<int32_t> tokens;
    REQUIRE_OK(ch.step({&a}, point_rows({1, 4}), &tokens));
    REQUIRE_EQ(tokens.size(), (size_t)2);
    CHECK_EQ(tokens[0], 1); // Missing sibling token remains in the true root target law.
    const int32_t retained[] = {0};
    int32_t out[2]{}, accepted = -1;
    REQUIRE_OK(accept_greedy(retained, 1, tokens.data(), out, &accepted));
    CHECK_EQ(accepted, 0); CHECK_EQ(out[0], 1); // Never commits an unverified slot.
    a.n_draft = 0; a.draft = nullptr;
    a.output.push_back(4); // Old metadata is now stale, but final-token ordinary mode ignores it.
    REQUIRE_OK(ch.step({&a}, point_rows({6}), &tokens));
    REQUIRE_EQ(tokens.size(), (size_t)1);
    CHECK_EQ(tokens[0], 6);
}

namespace {
struct ScopedCoupled {
    const bool had = std::getenv("RADIANCE_MTP_COUPLED") != nullptr;
    const std::string value = had ? std::getenv("RADIANCE_MTP_COUPLED") : "";
    explicit ScopedCoupled(const char* v) { setenv("RADIANCE_MTP_COUPLED", v, 1); }
    ~ScopedCoupled() {
        if (had) setenv("RADIANCE_MTP_COUPLED", value.c_str(), 1);
        else unsetenv("RADIANCE_MTP_COUPLED");
    }
};
/* One target row over the 8-token vocabulary; distinct logits fix top-k's order. */
const float kCoupleRow[8] = {0.4f, -0.3f, 1.1f, -2.0f, -1.0f, 0.2f, -0.7f, 0.9f};
std::vector<float> couple_rows(int n) {
    std::vector<float> lg;
    for (int i = 0; i < n; ++i) lg.insert(lg.end(), kCoupleRow, kCoupleRow + 8);
    return lg;
}
/* What the engine binds for draft row k: Q over the row's logits in top-k order -- so the
 * sampler's FP64 P equals Q bit for bit -- x drawn at uniform u, at output position n_out + k. */
RadMtpCoupleState couple_state(const Request& r, int k, double u) {
    int32_t ids[8];
    float val[8];
    for (int i = 0; i < 8; ++i) ids[i] = i;
    std::sort(ids, ids + 8, [](int32_t a, int32_t b) { return kCoupleRow[a] > kCoupleRow[b]; });
    for (int i = 0; i < 8; ++i) val[i] = kCoupleRow[ids[i]];
    RadMtpCoupleState s{};
    s.draw = {r.sp.seed, 0, r.id, 0, 1};
    s.status = 1;
    s.round = k + 1;
    if (!rad_mtp_couple_build_q(ids, val, 8, &s)) s.status = -1;
    s.x = s.ids[rad_mtp_couple_pick(s, u)];
    s.pos = (uint64_t)r.output.size() + (uint64_t)k;
    return s;
}
} // namespace

TEST(mtp_coupled_explicit_zero_and_invalid_flags) {
    {
        ScopedCoupled off("0");
        HostChain ch(8, 1, 0);
        REQUIRE(ch.ok);
        CHECK(ch.s.staged_couple() == nullptr);
        Request a = make_req(9301, {7}, greedy_params());
        std::vector<int32_t> tokens;
        REQUIRE_OK(ch.step({&a}, point_rows({6}), &tokens));
        CHECK_EQ(tokens[0], 6);
    }
    {
        ScopedCoupled bad("2");
        HostChain ch(8, 1, 2);
        CHECK(!ch.ok);
    }
    {
        ScopedCoupled on("1");
        ScopedSpecHub hub;
        HostChain ch(8, 1, 2);
        CHECK(!ch.ok);   // SpecHub owns the one-position tree experiment; never both
    }
    {
        const char* previous = std::getenv("RADIANCE_NO_DRAFT");
        const bool had = previous != nullptr;
        const std::string saved = previous ? previous : "";
        for (const char* value : {"", "0", "1"}) {
            setenv("RADIANCE_NO_DRAFT", value, 1); // The engine disables drafting by presence.
            ScopedCoupled on("1");
            HostChain ch(8, 1, 2);
            CHECK(!ch.ok);
        }
        {
            ScopedCoupled off("0");
            HostChain ch(8, 1, 2);
            CHECK(ch.ok); // Ordinary sampling still permits explicitly disabled drafting.
        }
        if (had) setenv("RADIANCE_NO_DRAFT", saved.c_str(), 1);
        else unsetenv("RADIANCE_NO_DRAFT");
    }
}

TEST(mtp_coupled_rows_emit_their_bound_draft_and_the_bonus_stays_ordinary) {
    ScopedCoupled on("1");
    HostChain ch(8, 1, 2);
    REQUIRE(ch.ok); // A missing coupling kernel is a failure, never a skip.
    ScopedCoupled off("0");
    HostChain plain(8, 1, 2);
    REQUIRE(plain.ok);
    SamplingParams sp;
    sp.temp = 1.0f; sp.top_k = 8;
    int differs = 0;
    for (int t = 0; t < 64; ++t) {
        sp.seed = 500 + (uint64_t)t;
        Request a = make_req(9001 + (uint64_t)t, {7}, sp);
        a.output = {3, 5};
        a.state = ReqState::Running;
        const RadMtpCoupleState st[2] = {couple_state(a, 0, (t + 0.5) / 64),
                                         couple_state(a, 1, (63 - t + 0.5) / 64)};
        const int32_t draft[2] = {st[0].x, st[1].x};
        a.n_draft = 2; a.draft = draft;
        REQUIRE_OK(ch.s.begin_request(a.id, a.sp, nullptr));
        REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 2));
        std::vector<int32_t> coupled, ordinary;
        REQUIRE_OK(ch.step({&a}, couple_rows(3), &coupled));
        const RadMtpCoupleState* staged = ch.s.staged_couple();
        REQUIRE(staged);
        CHECK_EQ(staged[0].status, 1); CHECK_EQ(staged[1].status, 1); CHECK_EQ(staged[2].status, 0);
        REQUIRE_OK(plain.step({&a}, couple_rows(3), &ordinary));
        REQUIRE_EQ(coupled.size(), (size_t)3);
        REQUIRE_EQ(ordinary.size(), (size_t)3);
        CHECK_EQ(coupled[0], draft[0]);       // P == Q: the coupled row always accepts x
        CHECK_EQ(coupled[1], draft[1]);
        CHECK_EQ(coupled[2], ordinary[2]);    // the bonus row is the ordinary pick
        differs += ordinary[0] != draft[0];
    }
    CHECK(differs > 16);   // the independent ordinary sample would often have rejected x
}

TEST(mtp_coupled_sampler_binds_each_draft_row_and_refuses_a_missing_or_foreign_state) {
    ScopedCoupled on("1");
    HostChain ch(8, 2, 2);
    REQUIRE(ch.ok);
    SamplingParams sp;
    sp.temp = 1.0f; sp.top_k = 8; sp.seed = 41;
    Request a = make_req(9201, {7}, sp);
    a.output = {3, 5};
    a.state = ReqState::Running;
    RadMtpCoupleState st[2] = {couple_state(a, 0, 0.3), couple_state(a, 1, 0.8)};
    int32_t draft[2] = {st[0].x, st[1].x};
    a.n_draft = 2; a.draft = draft;
    CHECK_EQ(ch.s.set_mtp_couple(a.id, st, 2), RAD_E_NOTFOUND);
    REQUIRE_OK(ch.s.begin_request(a.id, a.sp, nullptr));
    CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE);   // a running stochastic draft without state
    REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 2));
    REQUIRE_OK(ch.s.build_step({&a}));
    for (int which = 0; which < 4; ++which) {
        RadMtpCoupleState bad[2] = {st[0], st[1]};
        if (which == 0) bad[1].x = st[1].x == 0 ? 1 : 0;   // valid Q, but not the verified token
        if (which == 1) ++bad[0].pos;
        if (which == 2) ++bad[1].draw.seed;
        if (which == 3) bad[0].ids[1] = bad[0].ids[0];
        REQUIRE_OK(ch.s.set_mtp_couple(a.id, bad, 2));
        CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE);
    }
    RadMtpCoupleState foreign[2] = {st[0], st[1]};
    foreign[0].draw.seq = 9999;
    CHECK_EQ(ch.s.set_mtp_couple(a.id, foreign, 2), RAD_E_STATE);
    const RadMtpCoupleState swapped[2] = {st[1], st[0]};
    CHECK_EQ(ch.s.set_mtp_couple(a.id, swapped, 2), RAD_E_STATE);

    // A truncated publication keeps its prefix: one coupled draft row, an ordinary bonus.
    REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 2));
    a.n_draft = 1;
    REQUIRE_OK(ch.s.build_step({&a}));
    CHECK_EQ(ch.s.staged_couple()[0].status, 1); CHECK_EQ(ch.s.staged_couple()[1].status, 0);

    // Repeated steps: once committed, the old window is stale for a running request...
    a.n_draft = 2;
    a.output.push_back(draft[0]);
    CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE);
    // ...a finished request still carried by an ahead step keeps the ordinary pick...
    a.state = ReqState::Finished;
    REQUIRE_OK(ch.s.build_step({&a}));
    CHECK_EQ(ch.s.staged_couple()[0].status, 0);
    // ...and the next window binds afresh at the new output position.
    a.state = ReqState::Running;
    st[0] = couple_state(a, 0, 0.6); st[1] = couple_state(a, 1, 0.1);
    draft[0] = st[0].x; draft[1] = st[1].x;
    REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 2));
    REQUIRE_OK(ch.s.build_step({&a}));
    CHECK_EQ(ch.s.staged_couple()[1].pos, (uint64_t)a.output.size() + 1);

    // Cancellation and id reuse inherit nothing; a cleared request has no state either.
    ch.s.end_request(a.id);
    CHECK_EQ(ch.s.set_mtp_couple(a.id, st, 2), RAD_E_NOTFOUND);
    REQUIRE_OK(ch.s.begin_request(a.id, a.sp, nullptr));
    CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE);
    REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 2));
    REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 0));
    CHECK_EQ(ch.s.build_step({&a}), RAD_E_STATE);

    // Mixed batch: a greedy request's drafts need no state and keep the ordinary path.
    REQUIRE_OK(ch.s.set_mtp_couple(a.id, st, 2));
    SamplingParams gp = sp;
    gp.temp = 0.0f;
    Request g = make_req(9202, {6}, gp);
    g.output = {1};
    g.state = ReqState::Running;
    g.n_draft = 2; g.draft = draft;
    REQUIRE_OK(ch.s.begin_request(g.id, g.sp, nullptr));
    REQUIRE_OK(ch.s.build_step({&a, &g}));
    const RadMtpCoupleState* staged = ch.s.staged_couple();
    CHECK_EQ(staged[0].status, 1); CHECK_EQ(staged[1].status, 1); CHECK_EQ(staged[2].status, 0);
    for (int i = 3; i < 6; ++i) CHECK_EQ(staged[i].status, 0);
}

RAD_TEST_MAIN()
