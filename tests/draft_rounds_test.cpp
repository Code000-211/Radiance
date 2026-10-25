/* draft_rounds_test.cpp -- a draft cycle whose rounds shrink to the active prefix
 * (core/sched/advance.h, advance_impl.h).
 *
 * A plan gives each request its own K, so round j runs only for the requests with k >= j: the
 * deepest request's first round covers every drafting row, the next one only the rows that reach
 * it. What the host reads back after the cycle is therefore per row, not per cycle: StepSlot::gen
 * is the rounds that row finished, and its proposals sit at tok[1..gen] no matter how deep the
 * cycle ran for anyone else. Publishing the cycle's deepest round for every row reads another
 * round's proposal (or a stale cycle's), which the card cannot repair and acceptance then drops
 * silently.
 *
 * These are the card's own rows, run on host arrays through the file the kernel shares, with a
 * hand-built plan and hand-written proposals. No scheduler and no engine: the scheduler's half of
 * E2 (sorting the requests and emitting a round's active prefix) is another strand's.
 */
#include "rad_test.h"

#include "sched/advance_impl.h"
#include "sched/draft_types.h"

#include <cstring>

using namespace rad;

namespace {

/* The slab words the draft passes touch, carved the way BatchBuilder carves a build: one array per
 * field and a row descriptor block. Nothing else: these rows have no KV groups, no rotary planes
 * and no end-of-generation set. */
struct Slab {
    static constexpr int kSeq = 8, kTok = 32;
    enum : int32_t {
        ROWS = 0,                          /* [4][kSeq] row descriptors */
        TOK  = ROWS + 4 * kSeq,
        POS  = TOK + kTok,
        CTX  = POS + kTok,
        ACC  = CTX + kSeq,
        OUT  = ACC + kSeq,
        SID  = OUT + kTok,
        CU   = SID + kSeq,
        Q    = CU + kSeq + 8,
        END  = Q + kSeq
    };
    alignas(16) int32_t w[END] = {};
    StepSlot st[kSeq];
    int32_t  src[kTok] = {};

    Slab() {
        std::memset(st, 0x7f, sizeof st);   /* untouched entries stay recognisable */
        a.slab = reinterpret_cast<char*>(w);
        a.row_off = ROWS * 4;  a.row_pitch = kSeq;
        a.tok_off = TOK * 4;   a.pos_off = POS * 4; a.ctx_off = CTX * 4; a.acc_off = ACC * 4;
        a.out_off = OUT * 4;   a.sid_off = SID * 4; a.cu_off = CU * 4;   a.q_off = Q * 4;
        a.st  = st;
        a.src = src;
    }

    /* Row i: its one verified position (the row's tokens [k, k + 1)), its context, the slot it
     * names, and the advance's row descriptor (first verified token row, first sampled row, and
     * the number the mode reads, 0 accepted drafts here). */
    void row(int i, int k, int ctx, int card, int rout) {
        w[ROWS + i] = 1;
        w[ROWS + kSeq + i] = k;
        w[ROWS + 2 * kSeq + i] = rout;
        w[ROWS + 3 * kSeq + i] = 0;
        w[CU + i] = k;
        w[CU + i + 1] = k + 1;
        w[Q + i] = 1;
        w[CTX + i] = ctx;
        w[SID + i] = card;
    }

    void run(int mode, int round, int n) {
        a.mode = mode;
        a.round = round;
        a.n_seq = n;
        for (int i = 0; i < n; ++i) adv_row(a, i);
    }

    AdvanceArgs a;
    Slab(const Slab&) = delete;   /* `a` points into the object */
};

static_assert(sizeof(StepSlot) == 288, "StepSlot's layout is part of the card's slab");

}  /* namespace */

/* The plan's rows are the requests that reach each round. A cycle over k = {3, 2, 1, 0} runs round
 * 1 over three rows, round 2 over two, round 3 over one -- and each row carries exactly its own
 * proposals afterwards. */
TEST(a_rows_last_proposal_survives_the_rounds_that_no_longer_name_it) {
    Slab s;
    int32_t k[4] = { 3, 2, 1, 0 };
    int32_t rows[RAD_SCHED_MAX_SPEC + 1] = {};
    DraftPlan plan;
    plan.n_seq = 4; plan.cap = 4; plan.k_ceiling = 3; plan.max_k = 3;
    plan.k = k; plan.rows = rows;
    REQUIRE_EQ(draft_plan_rows(plan), RAD_OK);
    CHECK_EQ(plan.rows[1], 3);
    CHECK_EQ(plan.rows[2], 2);
    CHECK_EQ(plan.rows[3], 1);

    /* The acceptance that opens the cycle: no draft accepted, so every row's head is its context
     * and the cycle starts with nothing proposed. */
    for (int i = 0; i < 4; ++i) {
        s.row(i, /*k=*/0, /*ctx=*/100 + 10 * i, /*card=*/i, /*rout=*/i);
        s.src[i] = 9000 + i;
    }
    s.run(ADV_ACCEPT, 0, 4);
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(s.st[i].gen, 0);
        CHECK_EQ(s.st[i].head, 100 + 10 * i);
        CHECK_EQ(s.st[i].tok[0], 9000 + i);
    }

    /* One pass per round over the active prefix, each followed by its own take. Round 1 has no
     * pass of its own: the history pass carries it, and its take is the only one a row with k = 1
     * ever gets. Round r's pass embeds round r-1's proposal; the take behind it publishes its own,
     * P(r, i) = 1000 * r + i. */
    for (int rd = 1; rd <= 3; ++rd) {
        const int n = plan.rows[rd];
        if (rd > 1) {
            for (int i = 0; i < n; ++i) s.src[i] = 1000 * (rd - 1) + i;
            s.run(ADV_ROUND, rd, n);
        }
        for (int i = 0; i < n; ++i) s.src[i] = 1000 * rd + i;
        s.run(ADV_TAKE, rd, n);
    }

    for (int i = 0; i < 4; ++i) {
        const StepSlot& S = s.st[i];
        CHECK_EQ(S.gen, k[i]);
        for (int rd = 1; rd <= k[i]; ++rd) CHECK_EQ(S.tok[rd], 1000 * rd + i);
        /* Nothing beyond the row's own last round was written by the rounds that ran without it. */
        CHECK_EQ(S.tok[k[i] + 1], 0x7f7f7f7f);
    }
}

/* A new cycle publishes nothing from the last one: the acceptance resets the count, so a row that
 * proposes nothing this time is not handed the proposals the slots still hold. */
TEST(a_new_cycle_clears_the_rounds_it_did_not_repeat) {
    Slab s;
    for (int i = 0; i < 3; ++i) s.row(i, 0, 100 + 10 * i, i, i);
    s.run(ADV_ACCEPT, 0, 3);
    for (int rd = 1; rd <= 3; ++rd) {
        if (rd > 1) {
            for (int i = 0; i < 3; ++i) s.src[i] = 1000 * (rd - 1) + i;
            s.run(ADV_ROUND, rd, 3);
        }
        for (int i = 0; i < 3; ++i) s.src[i] = 1000 * rd + i;
        s.run(ADV_TAKE, rd, 3);
    }
    REQUIRE_EQ(s.st[0].gen, 3);

    /* The next step verifies again; this cycle's rows propose nothing and no round runs. */
    s.run(ADV_ACCEPT, 0, 3);
    for (int i = 0; i < 3; ++i) CHECK_EQ(s.st[i].gen, 0);

    /* Only round 1 runs, for one row: the others keep the old proposals in their slots, and their
     * count is what keeps them out of the host's publish. */
    s.src[0] = 2001;
    s.run(ADV_TAKE, 1, 1);
    CHECK_EQ(s.st[0].gen, 1);
    CHECK_EQ(s.st[0].tok[1], 2001);
    CHECK_EQ(s.st[0].tok[2], 1000 * 2 + 0);   /* still there, past the row's count */
    for (int i = 1; i < 3; ++i) CHECK_EQ(s.st[i].gen, 0);
}

/* A take the card swallowed: the round's pass ran, its ADV_TAKE did not, so the slot keeps the
 * count of the round before and never wrote that round's proposal. The host's expected depth is
 * the rounds it issued, so `gen` below it is exactly the mismatch the finish refuses -- and the
 * rejected round left nothing behind that could be published in its place. The next acceptance
 * then clears the count, so the proposals the slots still hold are out of reach of the step that
 * follows.
 *
 * This is a simulated card fault, not the scheduler's rejection path: a `set_draft` refusal is a
 * count mismatch in the scheduler, which no row-level test can reach. */
TEST(a_swallowed_take_is_short_of_the_plan_and_never_reaches_the_next_step) {
    Slab s;
    for (int i = 0; i < 3; ++i) s.row(i, 0, 100 + 10 * i, i, i);
    s.run(ADV_ACCEPT, 0, 3);

    /* Rounds 1 and 2 publish; round 3's advance is rejected, so its take never runs. */
    for (int rd = 1; rd <= 2; ++rd) {
        for (int i = 0; i < 3; ++i) s.src[i] = 1000 * rd + i;
        s.run(ADV_TAKE, rd, 3);
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(s.st[i].gen, 2);                      /* the plan issued 3: a refusal, not a publish */
        CHECK_EQ(s.st[i].tok[2], 1000 * 2 + i);
        CHECK_EQ(s.st[i].tok[3], 0x7f7f7f7f);          /* the rejected round wrote nothing */
    }

    /* The next cycle verifies again: every row that drafts sampled, so the acceptance resets its
     * count and the old proposals cannot be mistaken for this cycle's. */
    s.run(ADV_ACCEPT, 0, 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(s.st[i].gen, 0);
        CHECK_EQ(s.st[i].tok[2], 1000 * 2 + i);        /* still there, past the row's count */
    }
}

RAD_TEST_MAIN()
