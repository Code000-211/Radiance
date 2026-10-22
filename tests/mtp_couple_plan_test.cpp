/* mtp_couple_plan_test.cpp -- the draw the engine stages for a serial drafting row, resolved
 * against a step plan (core/sched/mtp_couple_plan.h, called by Engine::stage_couple_draws).
 *
 * The coupling failed twice before its first draft, and both failures were here rather than in the
 * ABI:
 *
 *   - The draw plane was sized from nothing, so every row lookup missed and the first step that
 *     drafted refused with "has no plan entry on its card" -- the engine started and then could
 *     not take one coupled step. The plane is now sized from cfg_.max_seqs, and the rule refuses a
 *     plane that is too small instead of silently missing.
 *   - A row whose card carries another sequence's draw is stale: the plugin refuses the staged
 *     draw, so the row is refused here, named, before any pass reads it.
 *
 * The plan is the real StepPlan the scheduler produces; only its entries are built by hand, since
 * the rule reads nothing else off it.
 */
#include "rad_test.h"

#include "sched/mtp_couple_bind.h"
#include "sched/mtp_couple_plan.h"

#include <cstring>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* One plan entry for a sequence on a card, with the pre-commit output size the draw coordinates
 * on. `greedy` is the request's own choice (top_k 1). */
StepEntry entry(Request* r, uint64_t seq, int32_t card, size_t output, bool greedy) {
    r->sp = SamplingParams{};
    r->sp.top_k = greedy ? 1 : 0;
    r->sp.temp  = 1.0f;
    r->output.assign(output, 0);
    StepEntry e;
    e.req  = r;
    e.seq  = seq;
    e.card = card;
    return e;
}

Scheduler::SerialRow row(uint64_t seq, int32_t card) {
    Scheduler::SerialRow r;
    r.seq  = seq;
    r.card = card;
    return r;
}

}  /* namespace */

/* The staged draw names the plan entry on the row's card: sequence, card, the PRE-commit output
 * size as the coordinate, the request's seed and whether it samples. Cards the plan does not use
 * stay zero, so a row landing on one of them is refused rather than served a stale draw. */
TEST(the_staged_draw_names_the_plan_entry_on_the_rows_card) {
    Request a, b, c;
    StepPlan plan;
    plan.e.push_back(entry(&a, 11, 3, 7, false));
    plan.e.push_back(entry(&b, 22, 5, 9, true));
    const Scheduler::SerialRow rows[] = { row(11, 3), row(22, 5) };

    std::vector<RadMtpCoupleDraw> by_card(8);
    RadMtpCoupleDraw stage[2];
    std::string why;
    REQUIRE_EQ(mtp_couple_plan_stage(plan, 8, rows, 2, &by_card, stage, &why), RAD_OK);

    CHECK_EQ(stage[0].seq, (uint64_t)11);
    CHECK_EQ(stage[0].card, 3);
    CHECK_EQ(stage[0].base, (uint64_t)7);
    CHECK_EQ(stage[0].seed, a.sp.seed);
    CHECK_EQ(stage[0].mode, 1u);
    /* A greedy request keeps the head's proposal instead of sampling x~Q. */
    CHECK_EQ(stage[1].seq, (uint64_t)22);
    CHECK_EQ(stage[1].card, 5);
    CHECK_EQ(stage[1].mode, 0u);
    /* Card 0 carried no entry: zeroed, so its seq names nothing a row could match. */
    CHECK_EQ(by_card[0].seq, (uint64_t)0);
    CHECK_EQ(by_card[7].seq, (uint64_t)0);
    (void)c;
}

/* THE FIRST-DRAFT REGRESSION: a plane sized for fewer slots than the step can use would make every
 * row miss. It is refused by name -- never a silent miss, and never a start that cannot draft. */
TEST(a_draw_plane_sized_for_fewer_slots_than_the_step_uses_is_refused) {
    Request a;
    StepPlan plan;
    plan.e.push_back(entry(&a, 11, 30, 7, false));
    const Scheduler::SerialRow rows[] = { row(11, 30) };

    std::vector<RadMtpCoupleDraw> empty;
    RadMtpCoupleDraw stage[1];
    std::string why;
    const int st = mtp_couple_plan_stage(plan, 32, rows, 1, &empty, stage, &why);
    if (st == RAD_OK || why.find("holds 0 card slot(s)") == std::string::npos)
        fprintf(stderr, "    status %d, why '%s'\n", st, why.c_str());
    CHECK_EQ(st, RAD_E_STATE);
    CHECK(why.find("holds 0 card slot(s), the step can use 32") != std::string::npos);
}

/* A drafting row whose card names another sequence is stale: the staged draw would name a card the
 * advance writes for someone else, which the plugin refuses. Refused here, naming the row. */
TEST(a_row_whose_card_names_another_sequence_is_refused) {
    Request a;
    StepPlan plan;
    plan.e.push_back(entry(&a, 11, 3, 7, false));
    const Scheduler::SerialRow rows[] = { row(11, 3), row(33, 3) };

    std::vector<RadMtpCoupleDraw> by_card(8);
    RadMtpCoupleDraw stage[2];
    std::string why;
    const int st = mtp_couple_plan_stage(plan, 8, rows, 2, &by_card, stage, &why);
    if (st == RAD_OK || why.find("drafting row 1 (sequence 33, card 3)") == std::string::npos)
        fprintf(stderr, "    status %d, why '%s'\n", st, why.c_str());
    CHECK_EQ(st, RAD_E_STATE);
    CHECK(why.find("drafting row 1 (sequence 33, card 3) has no plan entry on its card") !=
          std::string::npos);
}

/* ================================================== THE VERIFY POSITION CONTRACT
 *
 * The sampler stages row k of a request at `n_out + k`, `n_out` being that request's committed
 * output length when the step is built. Round rd is verified at row k = rd - 1, so the state bound
 * after the drafting step must carry `n_out + (rd - 1)` for the `n_out` the NEXT step will see.
 *
 * These tests walk successive steps through the REAL staging function, the REAL bind rule
 * (core/sched/mtp_couple_bind.h) and the REAL row check (rad_mtp_couple_binds), with the request's
 * output growing the way a commit grows it.
 *
 * THE ENGINE USED TO REBUILD the coordinate as `pre-commit base + slot n_acc`. For a commit that
 * emitted exactly what the card accepted -- every commit of a request that is still running -- that
 * is the same number. It differs only when the commit emits fewer, and such a commit closes the
 * request (a max_tokens or context cut): the row is then terminal and will never sample again. Work
 * bound for it is carried terminal work, and a coupled verify there reads as a live round in any
 * log without being one. So the bind skips a request that is not Running, and the old formula's
 * one-ahead position for it is checked not to satisfy rad_mtp_couple_binds. */

namespace {

/* A sampled state as the propose kernel would leave it for `draw`, at round `rd` and position
 * `pos`: one Q token, x drawn from it, status 1. */
RadMtpCoupleState sampled(const RadMtpCoupleDraw& draw, int rd, uint64_t pos) {
    RadMtpCoupleState s{};
    s.draw   = draw;
    s.round  = rd;
    s.status = 1;
    s.count  = 1;
    s.x      = 100 + rd;
    for (int i = 0; i < 8; ++i) s.ids[i] = -1;   /* the proposer's padding */
    s.ids[0] = s.x;
    s.q[0]   = 1.0;
    s.pos    = pos;
    return s;
}

/* One step's staging through mtp_couple_plan_stage, for request `r` on `card`. */
RadMtpCoupleDraw stage_one(Request* r, uint64_t seq, int32_t card, std::string* why) {
    StepPlan plan;
    plan.e.push_back(entry(r, seq, card, r->output.size(), false));
    const Scheduler::SerialRow rows[] = { row(seq, card) };
    std::vector<RadMtpCoupleDraw> by_card(8);
    RadMtpCoupleDraw stage[1];
    if (mtp_couple_plan_stage(plan, 8, rows, 1, &by_card, stage, why) != RAD_OK) return {};
    return stage[0];
}

/* The card's slot after a step that accepted `n_acc - 1` drafts: the bind checks each round's
 * state against the token the card holds for it. */
StepSlot slot_for(const int32_t* tok, int rounds, int n_acc) {
    StepSlot S{};
    S.acc   = n_acc - 1;
    S.n_acc = n_acc;
    for (int rd = 1; rd <= rounds; ++rd) S.tok[rd] = tok[rd];
    return S;
}

}  /* namespace */

/* VARYING n_acc, SEVERAL ROUNDS, A CUT COMMIT. Every round of a live request binds at the length
 * its commit left, which is the position the next build samples it at. The cut commit closes the
 * request; its rows bind nothing, and the old formula's position for them is provably not a row the
 * sampler would stage. */
TEST(the_bound_position_is_the_verify_rows_sampling_position) {
    const int kRounds = 3;
    /* The card's one-based acceptance (S.n_acc) and what the commit actually appended, per step.
     * emitted == n_acc while the request runs; the last step is the cut: the card accepted all
     * three drafts (n_acc 4), the token budget let two through, and the commit closed the request
     * with "length". A step after it does not exist for this request. */
    const int n_acc[]   = { 1, 4, 3, 4 };
    const int emitted[] = { 1, 4, 3, 2 };
    const int steps = (int)(sizeof n_acc / sizeof n_acc[0]);

    Request r;
    r.id = 7;
    r.sp = SamplingParams{};
    r.sp.temp = 1.0f;
    r.state = ReqState::Running;

    std::vector<RadMtpCoupleState> bound;   /* the states bound after the previous step */
    uint64_t prev_base = 0;                 /* the previous draw's pre-commit coordinate */
    int prev_n_acc = 0;                     /* and the card's one-based acceptance of it */
    for (int step = 0; step < steps; ++step) {
        std::string why;
        const RadMtpCoupleDraw draw = stage_one(&r, r.id, 2, &why);
        REQUIRE_EQ(why, std::string());
        /* The draw's base is the PRE-commit length: the draft's own coordinate, not the verify's. */
        CHECK_EQ(draw.base, (uint64_t)r.output.size());

        /* The build of THIS step runs over the states bound at the previous one, at the length
         * that commit left. Every round lands on its row. */
        if (!bound.empty()) {
            for (int k = 0; k < kRounds; ++k) {
                const uint64_t n_out = (uint64_t)r.output.size();
                CHECK(rad_mtp_couple_binds(bound[(size_t)k], r.id, draw.seed, n_out, k,
                                           bound[(size_t)k].x, 1 << 20));
                /* The old formula on the same states: identical while the previous commit emitted
                 * its whole acceptance, one ahead per missing token otherwise. */
                const uint64_t old_pos = prev_base + (uint64_t)prev_n_acc + (uint64_t)k;
                CHECK_EQ(old_pos == n_out + (uint64_t)k,
                         emitted[step - 1] == n_acc[step - 1]);
            }
        }

        /* The commit, and the request's state afterwards: a cut closes it. */
        for (int j = 0; j < emitted[step]; ++j) r.output.push_back(0);
        const bool live = emitted[step] == n_acc[step];
        r.state = live ? ReqState::Running : ReqState::Finished;

        /* The bind, after the commit, through the production rule. The point is what the
         * scheduler's commit snapshot would carry: this fixture owns the request, so it is built
         * here directly. */
        MtpCoupleBindPoint bp;
        bp.found = true;
        bp.live  = r.state == ReqState::Running;
        bp.n_out = (uint64_t)r.output.size();
        CHECK_EQ(bp.live, live);

        const int32_t tok[] = { 11, 21, 22, 23 };
        const StepSlot S = slot_for(tok, kRounds, n_acc[step]);
        RadMtpCoupleState rb[(size_t)kRounds * 3]{};
        for (int rd = 1; rd <= kRounds; ++rd) {
            rb[(size_t)(rd - 1) * 3] = sampled(draw, rd, 0);
            rb[(size_t)(rd - 1) * 3].x = tok[rd];          /* the card's proposal */
            rb[(size_t)(rd - 1) * 3].ids[0] = tok[rd];
        }
        RadMtpCoupleState out[RAD_MTP_COUPLE_MAX_ROUNDS]{};
        int n_bound = -1;
        std::string bwhy;
        CHECK_EQ(mtp_couple_bind_rounds(bp, draw, rb, 3, kRounds, S, 1 << 20, out, &n_bound,
                                        &bwhy), RAD_OK);
        /* NOT LIVE, NOT BOUND: a terminal row must not read as a coupled verify. */
        CHECK_EQ(n_bound, live ? kRounds : 0);
        for (int rd = 1; rd <= (live ? kRounds : 0); ++rd) {
            const uint64_t want = (uint64_t)r.output.size() + (uint64_t)(rd - 1);
            CHECK_EQ(out[rd - 1].pos, want);
            CHECK(rad_mtp_couple_binds(out[rd - 1], r.id, draw.seed,
                                       (uint64_t)r.output.size(), rd - 1, tok[rd], 1 << 20));
        }

        /* THE REGRESSION: where the old formula would have bound this step's rounds, the row
         * check fails as soon as the commit emitted fewer than the card accepted. */
        if (!live) {
            for (int rd = 1; rd <= kRounds; ++rd) {
                const uint64_t old_pos = draw.base + (uint64_t)n_acc[step] + (uint64_t)(rd - 1);
                CHECK(old_pos != (uint64_t)r.output.size() + (uint64_t)(rd - 1));
                CHECK(!rad_mtp_couple_binds(sampled(draw, rd, old_pos), r.id, draw.seed,
                                            (uint64_t)r.output.size(), rd - 1, tok[rd], 1 << 20));
            }
        }

        prev_base = draw.base;
        prev_n_acc = n_acc[step];
        bound.assign(out, out + (live ? kRounds : 0));
        if (!live) break;   /* the request is closed: no further step drafts it */
    }
}

/* STALE STATES ARE REJECTED. A round from another step, a round whose position is off by one, and
 * a token the row does not propose all fail the rule -- the coupling never applies a law this step
 * did not draw. */
TEST(a_state_from_another_step_is_refused) {
    Request r;
    r.id = 11;
    r.sp = SamplingParams{};
    r.sp.temp = 1.0f;
    std::string why;
    const RadMtpCoupleDraw d = stage_one(&r, r.id, 3, &why);
    REQUIRE_EQ(why, std::string());

    const uint64_t n_out = 5;
    const RadMtpCoupleState good = sampled(d, 1, rad_mtp_couple_verify_pos(n_out, 1));
    CHECK(rad_mtp_couple_binds(good, r.id, d.seed, n_out, 0, good.x, 1 << 20));

    /* One position ahead: the old arithmetic's answer. */
    CHECK(!rad_mtp_couple_binds(sampled(d, 1, rad_mtp_couple_verify_pos(n_out, 1) + 1), r.id,
                                d.seed, n_out, 0, good.x, 1 << 20));
    /* Another step's round. */
    CHECK(!rad_mtp_couple_binds(sampled(d, 2, rad_mtp_couple_verify_pos(n_out, 2)), r.id,
                                d.seed, n_out, 0, good.x, 1 << 20));
    /* Another seed, another sequence. */
    CHECK(!rad_mtp_couple_binds(good, r.id, d.seed + 1, n_out, 0, good.x, 1 << 20));
    CHECK(!rad_mtp_couple_binds(good, r.id + 1, d.seed, n_out, 0, good.x, 1 << 20));
    /* The row proposes a token this state did not draw, and a state with no Q at all. */
    CHECK(!rad_mtp_couple_binds(good, r.id, d.seed, n_out, 0, good.x + 1, 1 << 20));
    RadMtpCoupleState greedy = good;
    greedy.status = 0;
    greedy.draw.mode = 0;
    CHECK(!rad_mtp_couple_binds(greedy, r.id, d.seed, n_out, 0, good.x, 1 << 20));
}

/* REQUEST AND SLOT REUSE. Two requests take turns on the same card, so each step's staged draw
 * names the sequence currently on it. The previous request's rounds are not accepted by the new
 * one's rows, whichever position they carry. */
TEST(a_reused_card_serves_the_sequence_that_is_on_it) {
    Request a, b;
    a.id = 21;
    b.id = 22;
    a.sp = SamplingParams{};
    b.sp = SamplingParams{};
    a.output.assign(4, 0);
    b.output.assign(9, 0);

    std::string why;
    const RadMtpCoupleDraw da = stage_one(&a, a.id, 5, &why);
    REQUIRE_EQ(why, std::string());
    const RadMtpCoupleDraw db = stage_one(&b, b.id, 5, &why);
    REQUIRE_EQ(why, std::string());
    CHECK_EQ(da.seq, (uint64_t)21);
    CHECK_EQ(db.seq, (uint64_t)22);

    /* a's rounds, at a's length, do not bind a row of b -- not even at b's position. */
    const RadMtpCoupleState sa = sampled(da, 1, rad_mtp_couple_verify_pos((uint64_t)b.output.size(), 1));
    CHECK(!rad_mtp_couple_binds(sa, b.id, db.seed, (uint64_t)b.output.size(), 0, sa.x, 1 << 20));
    /* b's own state does. */
    const RadMtpCoupleState sb = sampled(db, 1, rad_mtp_couple_verify_pos((uint64_t)b.output.size(), 1));
    CHECK(rad_mtp_couple_binds(sb, b.id, db.seed, (uint64_t)b.output.size(), 0, sb.x, 1 << 20));
}

RAD_TEST_MAIN()
