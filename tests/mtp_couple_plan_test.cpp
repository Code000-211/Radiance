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

RAD_TEST_MAIN()
