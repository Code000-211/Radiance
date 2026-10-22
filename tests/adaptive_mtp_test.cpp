/* adaptive_mtp_test.cpp -- F: the deterministic correctness fixtures for adaptive MTP.
 *
 * A scripted target/drafter pair, nothing else: each case states what the head proposes and what
 * the target itself produced at those positions, and the fixture turns the two into the
 * observation a cycle publishes. The acceptance arithmetic is the engine's own
 * (sample/accept.h), not a second implementation, so a case that passes here says something about
 * the rule E has to implement and not only about this file.
 *
 * What is pinned:
 *   - the label rule. The accepted prefix is 1, the first real rejection is 0, and nothing past it
 *     -- or past a censor -- carries a label.
 *   - the end classification. A draft that matched the target's stopping token is compared but not
 *     committed: `end == Stopped`, `checked == accepted`, and the stop position itself gets no
 *     label. Giving it a 0 would teach the head that a correct stop was wrong. A stop the head
 *     predicted and the target did not draw is the mirror image: a real 0 at that position. The
 *     target drawing the stop token instead of a draft is a real 0 too, even though the cycle then
 *     censors. A stop at the bonus position cuts no draft prefix, so the censor stays None. A row
 *     that reached verification with `end == Unset` is refused, not guessed.
 *   - the bonus token. The verify step emits the target's own token after the accepted prefix
 *     whether or not a draft was accepted, so the drafts are bounded by `remaining - 1`; a cycle
 *     never emits past `remaining`, and `remaining == 0` emits nothing at all.
 *   - raw match, commit length and the censor reason are three separate numbers.
 *   - above temperature 0 acceptance is equality with the target's own draw. A draft that is the
 *     mode is rejected when the target drew something else, and a draft that is not the mode is
 *     accepted when the target drew it; that is neither argmax acceptance nor min(1, p/q).
 *   - a K0 row in a mixed batch publishes no label but a valid empty observation (nothing
 *     generated, nothing rejected), and a plan from an earlier cycle still holds its own values
 *     after the next cycle is chosen (one storage set per live cycle).
 *
 * What is deliberately not here: state parity (KV bounds, GDN anchors, conv cursor, QSA tails,
 * MTP-history watermark) and the concurrency matrix. Both need a running engine, and a text
 * comparison would not stand in for them.
 */
#include "rad_test.h"

#include "draft_fakes.h"
#include "sample/accept.h"
#include "sample/host_ref.h"

#include <cmath>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* Token 0 is the scripted end of generation; drafts live at 1 and up. */
constexpr int32_t kEog = 0;

/* One request's scripted cycle. `draft` is the head's proposal, `target` what the target itself
 * drew at those positions plus the bonus position, so target.size() == draft.size() + 1. */
struct Seq {
    uint64_t         req_id = 0;
    int32_t          remaining = 4096;
    DraftEligibility eligibility = DraftEligibility::Eligible;
    std::vector<int32_t> draft;
    std::vector<int32_t> target;
};

/* What a cycle would publish for one request, derived with the engine's acceptance rule. */
struct Row {
    DraftSeqObservation o;
    int32_t committed = 0;   /* drafts committed; < o.raw_match when a censor cut the run */
    int32_t emitted = 0;     /* tokens the commit emits */
};

bool is_eog(int32_t t) { return t == kEog; }

/* The cycle's row. A request that never reached verification (cancelled, preempted, no drafter,
 * output exhausted, grammar-blocked) publishes no counts at all: an ineligible request produced
 * nothing, and its missing label must not read as a rejection. */
Row run_one(const Seq& s) {
    Row r;
    if (s.eligibility != DraftEligibility::Eligible) return r;
    if (s.draft.empty()) {
        /* K0: a plain decode row. It emits the target's own token, bounded by what is left, and
         * publishes a valid empty observation: nothing generated, nothing rejected. */
        r.o.end = DraftSeqEnd::AllAccepted;
        r.emitted = s.remaining > 0 ? 1 : 0;
        return r;
    }

    const int32_t k = (int32_t)s.draft.size();
    r.o.generated = (uint8_t)k;
    r.o.published = (uint8_t)k;   /* every generated draft was published */
    r.o.verified = (uint8_t)k;    /* the target scored each draft position */
    r.o.origin = DraftOrigin::Head;

    int32_t out[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t acc = 0;
    CHECK_OK(accept_greedy(s.draft.data(), k, s.target.data(), out, &acc));

    /* The engine's own cut (sample/accept.h): a stopping token ends the emission wherever it
     * appears among the accepted prefix and the token past it. `acc` stays the uncensored match
     * length, which is what the cost model is trained on. */
    const int32_t cut = accept_cut_at_eog(out, acc, is_eog);

    r.o.raw_match = (uint64_t)acc;
    r.o.accepted = (uint8_t)cut;
    r.committed = cut;
    r.emitted = cut + 1;                                    /* the bonus token is always emitted */

    if (cut < acc) {
        /* The draft agreed on the stop token. It is emitted but not committed, so `checked` ends
         * at `accepted` and that position carries no label. */
        r.o.end = DraftSeqEnd::Stopped;
        r.o.checked = (uint8_t)cut;
        r.o.censor = DraftCensor::Eog;
    } else {
        r.o.end = acc < k ? DraftSeqEnd::Rejected : DraftSeqEnd::AllAccepted;
        r.o.checked = (uint8_t)(r.o.end == DraftSeqEnd::Rejected ? acc + 1 : acc);
        /* The censor says which *draft prefix* was cut. A stop token the target drew at a
         * rejected draft position is the correction there, so the cycle censors. A stop token at
         * the bonus position cut no draft at all: the run ends, the censor stays None. */
        if (acc < k && is_eog(out[acc])) r.o.censor = DraftCensor::Eog;
        else if (s.remaining > 0 && r.emitted >= s.remaining) r.o.censor = DraftCensor::MaxTokens;
    }
    return r;
}

/* The label sequence as a compact string: '1' accepted, '0' rejected, '.' no label. */
std::string labels(const DraftSeqObservation& o, int n = 8) {
    std::string s;
    for (int j = 0; j < n; ++j) {
        float v = -1.0f;
        s += draft_label(o, j, &v) ? (v > 0.5f ? '1' : '0') : '.';
    }
    return s;
}

/* A draft of k tokens the target agrees with everywhere, plus a distinct bonus token. */
Seq agreeing(uint64_t id, int k, int32_t first = 1) {
    Seq s;
    s.req_id = id;
    for (int j = 0; j < k; ++j) {
        s.draft.push_back(first + j);
        s.target.push_back(first + j);
    }
    s.target.push_back(first + k);   /* the bonus token, always right by construction */
    return s;
}

/* As agreeing(), but the target's draw at `at` differs from the draft. */
Seq rejecting_at(uint64_t id, int k, int at) {
    Seq s = agreeing(id, k);
    s.target[(size_t)at] = 90 + at;   /* a token the head did not propose */
    return s;
}

}  /* namespace */

/* ================================================================== all-accept, all-reject */

TEST(all_accepted_drafts_are_labelled_one_and_the_bonus_adds_no_label) {
    const Row r = run_one(agreeing(1, 6));
    CHECK_EQ(r.o.raw_match, 6u);
    CHECK_EQ(r.o.accepted, 6);
    CHECK_EQ(r.o.checked, 6);      /* the bonus position was never a draft position */
    CHECK_EQ(r.emitted, 7);        /* six drafts and the target's own token */
    CHECK(r.o.censor == DraftCensor::None);
    CHECK_EQ(labels(r.o), std::string("111111.."));
}

TEST(every_rejection_position_labels_a_prefix_of_ones_and_one_zero) {
    /* k = 6, the rejection walks from the first draft position to the last. */
    for (int at = 0; at < 6; ++at) {
        const Row r = run_one(rejecting_at(1, 6, at));
        CHECK_EQ((int)r.o.raw_match, at);
        CHECK_EQ((int)r.o.accepted, at);
        CHECK_EQ((int)r.o.checked, at + 1);   /* the accepted prefix plus the rejection */
        CHECK_EQ(r.emitted, at + 1);
        CHECK(r.o.censor == DraftCensor::None);

        std::string want((size_t)at, '1');
        want += '0';
        want.append((size_t)(8 - (at + 1)), '.');
        CHECK_EQ(labels(r.o), want);
    }
}

/* ================================================================== censoring */

TEST(a_predicted_end_of_generation_matches_but_is_not_committed) {
    /* The head predicts the stop token at every position; each is a real match, so raw_match
     * counts them, while the commit stops at the first one. This is the raw-match / commit-length
     * split the cost model depends on.
     *
     * The stop token's own position is compared but not committed, so it is Stopped: checked ends
     * at accepted and the position gets no label. A 0 there would teach the head that a correct
     * stop was wrong. */
    for (int e = 0; e < 6; ++e) {
        Seq s = agreeing(1, 6);
        s.target[(size_t)e] = kEog;
        s.draft[(size_t)e] = kEog;     /* the head predicted it, so it agrees */

        const Row r = run_one(s);
        CHECK_EQ((int)r.o.raw_match, 6);        /* every position matched */
        CHECK_EQ((int)r.o.accepted, e);         /* nothing from the stop token on is committed */
        CHECK_EQ((int)r.o.checked, e);          /* and nothing from it on is labelled */
        CHECK_EQ(r.emitted, e + 1);
        CHECK(r.o.end == DraftSeqEnd::Stopped);
        CHECK(r.o.censor == DraftCensor::Eog);
        CHECK(draft_seq_observation_valid(r.o));

        std::string want((size_t)e, '1');
        want.append((size_t)(8 - e), '.');
        CHECK_EQ(labels(r.o), want);
    }
}

TEST(a_stop_token_the_target_chooses_after_a_rejection_is_a_real_zero) {
    /* The correction at the rejected position is the stop token: the run ends there, but the
     * target drew another token than the head proposed, so that position is a measured rejection
     * and keeps its 0. The censor explains why nothing follows; it does not erase the 0. */
    Seq s = rejecting_at(1, 6, 3);
    s.target[3] = kEog;

    const Row r = run_one(s);
    CHECK_EQ((int)r.o.raw_match, 3);
    CHECK_EQ((int)r.o.accepted, 3);
    CHECK_EQ((int)r.o.checked, 4);
    CHECK_EQ(r.emitted, 4);
    CHECK(r.o.end == DraftSeqEnd::Rejected);
    CHECK(r.o.censor == DraftCensor::Eog);
    CHECK_EQ(labels(r.o), std::string("1110...."));
}

TEST(a_stop_token_at_the_bonus_position_cuts_no_draft_so_it_censors_nothing) {
    /* Every draft matched and the target's own token past the prefix is the stop. The run ends
     * there, but no draft position was cut: `end` stays AllAccepted and the censor stays None. */
    Seq s = agreeing(1, 3);
    s.target[3] = kEog;                    /* the bonus position, past the last draft */

    const Row r = run_one(s);
    CHECK_EQ((int)r.o.raw_match, 3);
    CHECK_EQ((int)r.o.accepted, 3);
    CHECK_EQ((int)r.o.checked, 3);
    CHECK_EQ(r.emitted, 4);
    CHECK(r.o.end == DraftSeqEnd::AllAccepted);
    CHECK(r.o.censor == DraftCensor::None);
    CHECK(draft_seq_observation_valid(r.o));
    CHECK_EQ(labels(r.o), std::string("111....."));
}

TEST(a_stop_the_head_predicted_but_the_target_did_not_is_a_real_reject) {
    /* The mirror of the stopped case: the head proposes the stop token and the target draws
     * something else at that position. The draft was wrong, so the position is a measured 0 --
     * not a Stopped row with no label. The target's token is not the stop, so nothing censors. */
    Seq s;
    s.req_id = 1;
    s.draft  = { 1, kEog, 3 };
    s.target = { 1, 9, 3, 5 };

    const Row r = run_one(s);
    CHECK_EQ((int)r.o.raw_match, 1);
    CHECK_EQ((int)r.o.accepted, 1);
    CHECK_EQ((int)r.o.checked, 2);
    CHECK_EQ(r.emitted, 2);
    CHECK(r.o.end == DraftSeqEnd::Rejected);
    CHECK(r.o.censor == DraftCensor::None);
    CHECK(draft_seq_observation_valid(r.o));
    CHECK_EQ(labels(r.o), std::string("10......"));
}

TEST(output_budget_censors_the_cycle_and_exhaustion_is_not_a_rejection) {
    /* remaining = 4 with k = 3: the commit emits four tokens and the run ends. Positions the
     * budget already used up carry no label, and the end is still AllAccepted -- the budget ran
     * out, the head was not wrong. */
    Seq s = agreeing(1, 3);
    s.remaining = 4;
    const Row r = run_one(s);
    CHECK_EQ(r.emitted, 4);
    CHECK(r.o.end == DraftSeqEnd::AllAccepted);
    CHECK(r.o.censor == DraftCensor::MaxTokens);
    CHECK_EQ(labels(r.o), std::string("111....."));

    /* A request with nothing left to emit drafts nothing. It is a plain decode row. */
    Seq done = agreeing(2, 3);
    done.remaining = 0;
    done.eligibility = DraftEligibility::OutputExhausted;
    const Row d = run_one(done);
    CHECK_EQ((int)d.o.generated, 0);
    CHECK_EQ((int)d.o.checked, 0);
    CHECK_EQ(labels(d.o), std::string("........"));
}

TEST(the_bonus_token_is_never_spent_on_a_draft) {
    /* The verify step emits the target's own token after the accepted prefix whether or not a
     * draft was accepted, so the drafts are bounded by remaining - 1. remaining <= 1 leaves room
     * for that token alone: K 0, not a draft of one. */
    CHECK_EQ(draft_budget_k(0, 6), 0);
    CHECK_EQ(draft_budget_k(1, 6), 0);
    CHECK_EQ(draft_budget_k(2, 6), 1);
    CHECK_EQ(draft_budget_k(3, 6), 2);
    CHECK_EQ(draft_budget_k(7, 6), 6);     /* the window is the cap, not the budget */
    CHECK_EQ(draft_budget_k(64, 6), 6);
    CHECK_EQ(draft_budget_k(64, 0), 0);   /* no drafter to spend it on */

    /* End to end: a cycle planned at the exact boundary emits exactly the budget and no more. */
    Seq s = agreeing(1, 2);
    s.remaining = 3;                      /* draft_budget_k(3, 6) == 2 */
    const Row r = run_one(s);
    CHECK_EQ(r.emitted, 3);
    CHECK_EQ(r.emitted, s.remaining);     /* not one token past it */
    CHECK(r.o.censor == DraftCensor::MaxTokens);
    CHECK_EQ(labels(r.o), std::string("11......"));

    /* And one token tighter drafts one less rather than overrunning. */
    Seq tight = agreeing(2, 1);
    tight.remaining = 2;
    const Row t = run_one(tight);
    CHECK_EQ(t.emitted, 2);
    CHECK_EQ(t.emitted, tight.remaining);

    /* The plan the scheduler publishes spends the same budget: four requests with 64, 3, 2 and 1
     * tokens left hold 6, 2, 1 and 0 drafts. A budget of one token is K0, not a draft of zero
     * width. */
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 61, 64);
    f.add(1, 62, 3);
    f.add(2, 63, 2);
    f.add(3, 64, 1);
    draft_fake::Policy policy;
    policy.legacy_k = 6;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    const int32_t want_k[4] = { 6, 2, 1, 0 };
    for (int i = 0; i < 4; ++i) CHECK_EQ(f.plan.k[i], want_k[i]);
    CHECK(f.plan.fill[3] == DraftDrafter::None);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(the_last_token_of_the_budget_is_emitted_once_and_zero_emits_nothing) {
    /* remaining 1: the verify step's own token is all that is left, so K0 and exactly one emitted
     * token. remaining 0: nothing is left and nothing is emitted, not even the bonus token. */
    CHECK_EQ(draft_budget_k(1, 6), 0);   /* one token left is K0, not a draft of zero width */
    CHECK_EQ(draft_budget_k(0, 6), 0);

    Seq one;
    one.req_id = 1;
    one.remaining = 1;
    const Row r = run_one(one);
    CHECK_EQ(r.emitted, 1);
    CHECK_EQ(r.emitted, one.remaining);
    CHECK_EQ((int)r.o.accepted, 0);
    CHECK(r.o.end == DraftSeqEnd::AllAccepted);
    CHECK(draft_seq_observation_valid(r.o));

    Seq zero;
    zero.req_id = 2;
    zero.remaining = 0;
    const Row z = run_one(zero);
    CHECK_EQ(z.emitted, 0);
    CHECK_EQ(z.emitted, zero.remaining);
    CHECK_EQ((int)z.o.accepted, 0);
    CHECK(z.o.end == DraftSeqEnd::AllAccepted);
    CHECK(draft_seq_observation_valid(z.o));
}

TEST(cancelled_preempted_and_grammar_capped_requests_publish_no_labels) {
    for (DraftEligibility e : { DraftEligibility::Cancelled, DraftEligibility::Preempted,
                                DraftEligibility::Grammar, DraftEligibility::NoDrafter,
                                DraftEligibility::Budget }) {
        Seq s = agreeing(1, 6);
        s.eligibility = e;
        const Row r = run_one(s);
        CHECK_EQ((int)r.o.generated, 0);
        CHECK_EQ((int)r.o.checked, 0);
        CHECK_EQ(labels(r.o), std::string("........"));
    }
}

TEST(a_half_stated_observation_is_refused_before_anything_is_learned) {
    /* A cancelled row still in flight is not a label. The observer refuses a cycle the caller has
     * not declared complete, so no predictor sees a half-finished verify. */
    draft_fake::Observer obs;
    DraftObservation o;
    o.complete = false;
    CHECK_EQ(draft_fake::observe(&obs, o, DraftCostSample{}), RAD_E_INVAL);
    CHECK_EQ(obs.calls, 0);

    o.complete = true;
    CHECK_EQ(draft_fake::observe(&obs, o, DraftCostSample{}), RAD_E_INVAL);   /* still no plan */
    CHECK_EQ(obs.calls, 0);

    /* Complete but without the plan it came from is equally unreadable: a row's identity is
     * (req_id, slot, slot_gen) against its plan, not the counts alone. */
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 71, 64);
    draft_fake::Policy policy;
    policy.legacy_k = 2;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    /* An unset end is refused, never guessed: a row that reached verification must say how the
     * run ended, because AllAccepted and Stopped both have checked == accepted. */
    DraftSeqObservation row;
    row.generated = 2;
    row.published = 2;
    row.verified = 2;
    row.checked = 2;
    row.accepted = 2;
    row.raw_match = 2;
    row.end = DraftSeqEnd::AllAccepted;
    row.origin = DraftOrigin::Head;

    uint64_t rid[1] = { 71 };
    int32_t  slot[1] = { 0 };
    uint32_t gen[1] = { 1 };
    DraftObservation d;
    d.cycle_id = f.plan.cycle_id;
    d.policy_version = f.plan.policy_version;
    d.n_seq = 1;
    d.cap = 1;
    d.plan = &f.plan;
    d.req_id = rid;
    d.slot = slot;
    d.slot_gen = gen;
    d.seq = &row;
    d.complete = true;

    CHECK_EQ(draft_fake::observe(&obs, d, DraftCostSample{}), RAD_OK);
    CHECK_EQ(obs.calls, 1);

    row.end = DraftSeqEnd::Unset;
    CHECK(!draft_seq_observation_valid(row));
    CHECK_EQ(draft_fake::observe(&obs, d, DraftCostSample{}), RAD_E_INVAL);
    CHECK_EQ(obs.calls, 1);   /* the refused cycle taught nothing */
}

/* ================================================================== sampling above T = 0 */

TEST(sampled_acceptance_is_the_targets_own_draw_not_the_mode) {
    /* Two candidates, 0.62 / 0.38. The target's draw comes from the sampler's own seeded pick, so
     * the case states an exact, reproducible outcome rather than a distribution. */
    const float logits[2] = { (float)std::log(0.62), (float)std::log(0.38) };
    constexpr uint64_t seed = 77;
    CHECK_EQ(host_argmax(logits, 2), 0);   /* 0 is the mode at every position */

    int pos_non_mode = -1;
    for (uint64_t pos = 0; pos < 64; ++pos) {
        CandidateSet cs;
        cs.from_logits(logits, 2);
        if (host_pick(cs, seed, pos) == 1) { pos_non_mode = (int32_t)pos; break; }
    }
    CHECK(pos_non_mode >= 0);

    /* Draft = the mode, target drew the other token. Rejection sampling with min(1, p/q) would
     * accept this with probability 1; agreeing with the target's own draw rejects it. */
    {
        Seq s;
        s.draft  = { 0 };
        s.target = { 1, 5 };
        const Row r = run_one(s);
        CHECK_EQ((int)r.o.accepted, 0);
        CHECK_EQ((int)r.o.checked, 1);
        CHECK_EQ(labels(r.o), std::string("0......."));
    }
    /* Draft = the token the target drew, though it is not the mode. Accepted: the rule is
     * equality with the draw, not argmax agreement. */
    {
        Seq s;
        s.draft  = { 1 };
        s.target = { 1, 5 };
        const Row r = run_one(s);
        CHECK_EQ((int)r.o.accepted, 1);
        CHECK_EQ(labels(r.o), std::string("1......."));
    }
}

/* ================================================================== mixed K and plan lifetime */

TEST(a_mixed_batch_counts_the_rows_that_reach_each_round) {
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 21, 64);
    f.add(1, 22, 64);
    f.add(2, 23, 64);
    f.add(3, 24, 2);   /* two tokens left: one draft, because the second is the bonus */
    f.add(4, 25, 1);   /* one token left: no draft at all */

    draft_fake::Policy policy;
    policy.legacy_k = 6;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 6);
    CHECK_EQ(f.plan.k[3], 1);
    CHECK_EQ(f.plan.k[4], 0);
    CHECK(f.plan.why[3] == DraftFallback::Resources);
    CHECK(f.plan.fill[4] == DraftDrafter::None);
    /* N_j: four rows reach round 1, the three uncapped rows reach 2..6. */
    const int32_t want[7] = { 0, 4, 3, 3, 3, 3, 3 };
    for (int j = 0; j <= 6; ++j) CHECK_EQ(f.plan.rows[j], want[j]);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    /* The K0 row publishes no label -- the empty sequence is not a run of rejections -- and its
     * observation is still valid: nothing generated, nothing rejected. */
    Seq k0;
    k0.req_id = 25;
    const Row k0r = run_one(k0);
    CHECK_EQ(labels(k0r.o), std::string("........"));
    CHECK_EQ((int)k0r.o.generated, 0);
    CHECK_EQ((int)k0r.o.accepted, 0);
    CHECK(k0r.o.end == DraftSeqEnd::AllAccepted);
    CHECK(k0r.o.origin == DraftOrigin::None);   /* a row that produced nothing is None */
    CHECK(draft_seq_observation_valid(k0r.o));
}

TEST(a_k0_row_in_a_batch_still_has_a_plan_of_its_own) {
    /* K0 is a plan, not the absence of one: the request keeps its identity and the plan says 0.
     * The distinction from "no drafter loaded" is k_ceiling, which the view declares. */
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;   /* the drafter is loaded; this cycle just drafts nothing */
    f.add(0, 31);
    draft_fake::Policy policy;
    policy.legacy_k = 0;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 0);
    CHECK_EQ(f.plan.k[0], 0);
    CHECK_EQ(f.plan.k_ceiling, 6);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    /* A process with no drafter says so in the ceiling, which is not a temporary K0. */
    f.view.k_ceiling = 0;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.k_ceiling, 0);
}

TEST(a_plan_from_an_earlier_cycle_survives_the_next_one_unchanged) {
    /* One storage set per live cycle. The first cycle's plan is still named by issue_ahead() when
     * the second is chosen, so choosing into another set must leave the first intact. */
    draft_fake::Fixture c1, c2;
    c1.view.k_ceiling = 6;
    c2.view.k_ceiling = 6;
    c1.add(0, 41, 64);
    c2.add(0, 41, 64);

    draft_fake::Policy p1, p2;
    p1.legacy_k = 6;
    p2.legacy_k = 2;
    CHECK_EQ(draft_fake::choose(&p1, c1.view, c1.pred, DraftCostModel{}, &c1.plan), RAD_OK);
    c2.view.cycle_id = 2;
    CHECK_EQ(draft_fake::choose(&p2, c2.view, c2.pred, DraftCostModel{}, &c2.plan), RAD_OK);

    CHECK_EQ(c1.plan.cycle_id, 1u);
    CHECK_EQ(c1.plan.max_k, 6);
    CHECK_EQ(c1.plan.k[0], 6);
    CHECK_EQ(c1.plan.rows[6], 1);
    CHECK_EQ(c2.plan.cycle_id, 2u);
    CHECK_EQ(c2.plan.max_k, 2);
    CHECK_EQ(c2.plan.k[0], 2);
    CHECK_EQ(c2.plan.rows[6], 0);
}

TEST(a_recycled_slot_is_a_different_request_even_at_the_same_index) {
    /* A preemption keeps a request's estimate; a recycled slot resets it. The identity a plan and
     * an observation agree on is (req_id, slot, slot_gen), not the index. */
    draft_fake::Fixture f;
    f.add(0, 51);
    draft_fake::Policy policy;
    policy.legacy_k = 3;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    const uint64_t old_id = f.plan.req_id[0];
    const uint32_t old_gen = f.plan.slot_gen[0];

    /* The slot is recycled for another request, which gets its own generation. */
    draft_fake::Fixture g;
    DraftSeqView& v = g.add(0, 52);
    v.slot_gen = old_gen + 1;
    g.view.cycle_id = f.view.cycle_id + 1;
    CHECK_EQ(draft_fake::choose(&policy, g.view, g.pred, DraftCostModel{}, &g.plan), RAD_OK);

    CHECK_EQ(f.plan.slot[0], g.plan.slot[0]);   /* the index alone says nothing */
    CHECK(old_id != g.plan.req_id[0]);
    CHECK(f.plan.slot_gen[0] != g.plan.slot_gen[0]);

    /* And the earlier request's plan still describes the earlier request. */
    CHECK_EQ(f.plan.req_id[0], 51u);
    CHECK_EQ(f.plan.k[0], 3);
}

RAD_TEST_MAIN()
