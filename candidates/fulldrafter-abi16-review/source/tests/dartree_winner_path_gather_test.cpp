/* dartree_winner_path_gather_test.cpp -- the ACCEPTED-PATH walk and the rows it names.
 *
 * The central changed branch of the DarTree work is not "is a tree built" -- it is WHICH ROWS the
 * passes that follow a verify read, and WHERE the sequence continues. A tree's winner is a set of
 * NODE ids, and the sampled row for the accepted token at depth j is the row of the NODE at depth
 * j, which may be any index. A winner of nodes {0, 2, 4, 5} occupies rows 0, 2, 4 and 5, with
 * REJECTED siblings on rows 1 and 3; the drafter's cache outlives the step, so reading the row run
 * 0..3 instead puts a rejected sibling's hidden state in the head's own cache and the model is then
 * conditioned on text it never emitted. Nothing in the step's output shows it.
 *
 * This file drives the decision where it is made -- `adv_row` in core/sched/advance_impl.h, which
 * `step_advance` runs directly on the host backend and the card runs as the same function -- and
 * asserts on what came out.
 *
 * ACCEPTANCE'S RULE, restated because the fixtures depend on it: at each depth the walk takes the
 * child of the node just accepted whose token equals the target's own pick AT THAT NODE. So a
 * fixture only accepts node j when `tok[j] == pick[parent(j)]`, transitively from the anchor.
 */
#include "rad_test.h"

#include "sched/advance.h"
#include "sched/advance_impl.h"

#include <vector>

using namespace rad;

namespace {

/* ONE ARENA, carved the way the step batch carves its slab: `AdvanceArgs` addresses everything by a
 * byte offset from the slab's base, so the arrays have to be inside one object for those offsets to
 * mean anything. */
struct Slab {
    std::vector<char> arena;
    int32_t *tok, *pos, *ctx, *acc, *sid, *cu, *q, *rows4, *out, *dout, *src;
    std::vector<StepSlot> st;
    AdvanceArgs a{};

    Slab(int n_seq, int n_tok) {
        const int64_t n_i32 = (int64_t)(n_tok + 2) + n_tok + (int64_t)n_seq * 5 + (n_seq + 1) +
                              (int64_t)n_seq * 4 + n_tok + n_seq + (int64_t)n_seq * 16;
        arena.assign((size_t)n_i32 * sizeof(int32_t) + 128, 0);
        char* p = arena.data();
        auto take = [&](int64_t n) {
            int32_t* r = (int32_t*)p;
            p += n * (int64_t)sizeof(int32_t);
            return r;
        };
        tok   = take(n_tok + 2);
        pos   = take(n_tok);
        ctx   = take(n_seq);
        acc   = take(n_seq);
        sid   = take(n_seq);
        cu    = take(n_seq + 1);
        q     = take(n_seq);
        rows4 = take((int64_t)n_seq * 4);
        out   = take(n_tok);
        dout  = take(n_seq);
        src   = take((int64_t)n_seq * 16);
        st.assign((size_t)n_seq, StepSlot{});
        a.n_seq = n_seq;
        a.slab = arena.data();
        a.tok_off = off(tok);
        a.pos_off = off(pos);
        a.ctx_off = off(ctx);
        a.acc_off = off(acc);
        a.out_off = off(out);
        a.dout_off = off(dout);
        a.sid_off = off(sid);
        a.cu_off = off(cu);
        a.q_off = off(q);
        a.row_off = off(rows4);
        a.row_pitch = n_seq;
        a.grp_off = off(tok);       /* unused: n_groups is 0 */
        a.n_groups = 0;             /* no KV groups: this file is about the walk, not the store */
        a.rope_off = -1;
        a.st = st.data();
        a.src = src;
        a.src_pitch = 16;
        a.n_eog = 0;
        a.eog = nullptr;
        a.spec_parent = nullptr;
    }
    int64_t off(const void* q_) const {
        return (int64_t)((const char*)q_ - (const char*)arena.data());
    }
    void set_row(int i, int32_t dev_tok, int32_t dev_out, int32_t aux, bool dev) {
        rows4[(size_t)i] = dev ? 1 : 0;
        rows4[(size_t)a.row_pitch + i] = dev_tok;
        rows4[(size_t)2 * a.row_pitch + i] = dev_out;
        rows4[(size_t)3 * a.row_pitch + i] = aux;
    }
};

/* A tree shaped like the one DFlash2/MTP publish, as a parent array over node ids:
 *
 *     node 0  "A"       parent -1   depth 0   the anchor, a token the target already chose
 *     node 1  "A1"      parent  0   depth 1   REJECTED sibling
 *     node 2  "A2"      parent  0   depth 1   ACCEPTED
 *     node 3  "A2B"     parent  2   depth 2   REJECTED sibling
 *     node 4  "A2C"     parent  2   depth 2   ACCEPTED
 *     node 5  "A2CQ"    parent  4   depth 3   ACCEPTED
 *
 * `pick` is the target's own choice at each node's row, chosen so the walk takes 0 -> 2 -> 4 -> 5
 * and then stops. The accepted path is non-identity from the first step, with a rejected sibling
 * BETWEEN two accepted nodes -- which is the case a start-and-count gather cannot represent.
 */
struct Tree {
    int32_t parent[6] = { -1, 0, 0, 2, 2, 4 };
    int32_t tok[6]    = { 0, 11, 21, 12, 22, 31 };
    int32_t pick[6]   = { 21, 90, 22, 91, 31, 77 };
};

}  /* namespace */

/* 1. THE WALK. A non-identity winner, driven through the real host advance row. */
TEST(accept_walk_follows_a_nonidentity_winner) {
    const int n_tok = 6, n_aux = 5;
    Slab s(1, n_tok);
    Tree t;
    s.tok[0] = 100;
    for (int j = 1; j <= n_aux; ++j) s.tok[j] = t.tok[j];
    for (int j = 0; j <= n_aux; ++j) s.src[j] = t.pick[j];

    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = t.parent;
    s.set_row(0, /*dev_tok=*/0, /*dev_out=*/0, n_aux, /*dev=*/true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 50;                    /* committed end before this step */
    s.st[0].ctx = 50;

    adv_row(s.a, 0);

    /* THE PATH: nodes 0, 2, 4, 5 -- node 1 rejected at depth 1, node 3 rejected at depth 2. */
    CHECK_EQ(s.st[0].acc, 3);
    CHECK_EQ(s.st[0].path[0], 0);
    CHECK_EQ(s.st[0].path[1], 2);
    CHECK_EQ(s.st[0].path[2], 4);
    CHECK_EQ(s.st[0].path[3], 5);

    /* THE DEPTH IS WHAT THE SEQUENCE REACHES, and it is not the node index: the walk ended on
     * node 5, which sits at depth 3 -- not at index 5, and not at `acc` = 3 by coincidence, which
     * is why the node index in the fixture makes the two disagree in the interesting direction. */
    CHECK_EQ(s.st[0].head, 50 + 3);
    CHECK_EQ(s.st[0].n_acc, 4);

    /* THE BONUS IS THE TARGET'S PICK AT THE LAST ACCEPTED NODE'S ROW, not at index `acc`. */
    CHECK_EQ(s.st[0].tok[0], t.pick[5]);
    CHECK(s.st[0].tok[0] != t.pick[3]);
}

/* 2. THE PASS THAT FILLS THE HEAD'S OWN CACHE READS THE ACCEPTED PATH'S ROWS. This is the consumer
 * `StepSlot::path` exists for. The history pass commits the tokens the verify step actually took,
 * one a row, and the token committed at depth j is the pick at node `path[j]`. */
TEST(history_pass_reads_the_accepted_path_rows) {
    const int n_tok = 6, n_aux = 5;
    Slab s(1, n_tok);
    Tree t;
    s.tok[0] = 100;
    for (int j = 1; j <= n_aux; ++j) s.tok[j] = t.tok[j];
    for (int j = 0; j <= n_aux; ++j) s.src[j] = t.pick[j];

    /* the verify step's acceptance: path 0 -> 2 -> 4 -> 5 */
    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = t.parent;
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 50;
    s.st[0].ctx = 50;
    adv_row(s.a, 0);
    CHECK_EQ(s.st[0].acc, 3);

    /* ...then the history pass over the four committed positions, nothing host-staged before them */
    s.a.mode = ADV_HIST;
    s.a.spec_parent = nullptr;
    s.set_row(0, 0, 0, /*aux=*/0, true);
    s.cu[0] = 0;
    s.q[0] = 4;
    adv_row(s.a, 0);

    /* THE TOKENS ARE THE PICKS AT NODES 0, 2, 4 AND 5. */
    CHECK_EQ(s.tok[0], t.pick[0]);
    CHECK_EQ(s.tok[1], t.pick[2]);
    CHECK_EQ(s.tok[2], t.pick[4]);
    CHECK_EQ(s.tok[3], t.pick[5]);

    /* AND NOT THE ROW RUN: a depth-indexed read would take node 1's pick at position 1 and node 3's
     * at position 2 -- the rejected siblings -- into the head's cache. */
    CHECK(s.tok[1] != t.pick[1]);
    CHECK(s.tok[2] != t.pick[3]);
}

/* 3. THE NEXT VERIFY STEP'S POSITIONS ARE CONTIGUOUS even though the rows were not, because the
 * text is a chain. The two are different coordinates and a change that conflates them advances the
 * model past a token it never emitted. */
TEST(positions_are_contiguous_while_rows_are_not) {
    const int n_tok = 6, n_aux = 5;
    Slab s(1, n_tok);
    Tree t;
    s.tok[0] = 100;
    for (int j = 1; j <= n_aux; ++j) s.tok[j] = t.tok[j];
    for (int j = 0; j <= n_aux; ++j) s.src[j] = t.pick[j];

    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = t.parent;
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 50;
    s.st[0].ctx = 50;
    adv_row(s.a, 0);

    const int32_t ctx_after = s.st[0].ctx;
    const int32_t n_acc = s.st[0].n_acc;
    CHECK_EQ(n_acc, 4);

    /* the trunk step that verifies against it: one row a committed position */
    s.a.mode = ADV_TRUNK;
    s.set_row(0, 0, 0, 0, true);
    s.q[0] = n_acc;
    adv_row(s.a, 0);

    for (int j = 0; j < n_acc; ++j) CHECK_EQ(s.pos[(size_t)j], ctx_after + j);
    CHECK_EQ(s.acc[0], n_acc);
    CHECK_EQ(s.ctx[0], ctx_after);
}

/* 4. A CHAIN IS UNCHANGED. Every program that does not declare a tree walks this path, so the same
 * walk with no parent array has to reproduce the old rule exactly. */
TEST(chain_walk_is_unchanged) {
    const int n_tok = 4, n_aux = 3;
    Slab s(1, n_tok);
    s.tok[0] = 100; s.tok[1] = 11; s.tok[2] = 22; s.tok[3] = 33;
    s.src[0] = 11; s.src[1] = 22; s.src[2] = 44; s.src[3] = 55;
    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = nullptr;                 /* a chain */
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 10;
    s.st[0].ctx = 10;
    adv_row(s.a, 0);

    CHECK_EQ(s.st[0].acc, 2);                  /* drafts 11 and 22 matched; 33 did not */
    CHECK_EQ(s.st[0].head, 10 + 2);
    CHECK_EQ(s.st[0].tok[0], 44);              /* the bonus is the target's pick at index acc */
    CHECK_EQ(s.st[0].path[1], 1);              /* and the path is the identity */
    CHECK_EQ(s.st[0].path[2], 2);
}

/* 5. AN END-OF-GENERATION TOKEN ENDS THE RUN WHERE IT IS PUSHED, and the position stops there --
 * counting it as an accepted draft would advance the next step one position past the token that
 * ended the turn. */
TEST(chain_walk_stops_at_end_of_generation) {
    const int n_tok = 4, n_aux = 3;
    Slab s(1, n_tok);
    s.tok[0] = 100; s.tok[1] = 11; s.tok[2] = 7; s.tok[3] = 33;
    s.src[0] = 11; s.src[1] = 7; s.src[2] = 55; s.src[3] = 66;   /* 7 is the stop id below */
    int32_t eog[1] = { 7 };
    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = nullptr;
    s.a.eog = eog;
    s.a.n_eog = 1;
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 10;
    s.st[0].ctx = 10;
    adv_row(s.a, 0);

    /* Draft 11 matched and is taken. The next position's draft IS the stop token, which ends the
     * run: one accepted draft, and the stop token is the step's committed token. */
    CHECK_EQ(s.st[0].acc, 1);
    CHECK_EQ(s.st[0].head, 11);
    CHECK_EQ(s.st[0].tok[0], 7);
}

/* 6. A TREE THAT ENDS AT A STOP TOKEN STOPS ON A NODE, NOT ON AN INDEX. Same non-identity winner,
 * with the token at depth 2 being the stop id: the walk takes nodes 0 and 2, stops, and the
 * position is DEPTH 1 -- not the node index 2, and not depth 2. */
TEST(tree_walk_stops_at_end_of_generation_on_a_node) {
    const int n_tok = 6, n_aux = 5;
    Slab s(1, n_tok);
    Tree t;
    s.tok[0] = 100;
    for (int j = 1; j <= n_aux; ++j) s.tok[j] = t.tok[j];
    for (int j = 0; j <= n_aux; ++j) s.src[j] = t.pick[j];
    /* node 4 proposes the stop id, and the target agrees at node 2's row -- acceptance is "the
     * node's token equals the target's pick there", so a stop token is taken only when both say so */
    s.tok[4] = 7;
    s.src[2] = 7;
    s.src[4] = 7;
    int32_t eog[1] = { 7 };
    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = t.parent;
    s.a.eog = eog;
    s.a.n_eog = 1;
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 50;
    s.st[0].ctx = 50;
    adv_row(s.a, 0);

    CHECK_EQ(s.st[0].acc, 1);              /* node 2 taken; the stop token is NOT counted */
    CHECK_EQ(s.st[0].head, 50 + 1);        /* depth 1, not node index 2 */
    CHECK_EQ(s.st[0].path[1], 2);
    CHECK_EQ(s.st[0].tok[0], 7);           /* the stop token IS the step's committed token */
}

/* 7. THE CARD'S ROW LIST IS THE PATH TOO. `out_ids` is the pass's tap-row list: the host stages the
 * rows the pass must read and ADV_HIST writes back the ones it actually used. Writing the flat
 * depth there instead of the node leaves the drafter reading whichever sibling sits at each depth
 * -- and the tokens are right, so nothing in the output shows it. This is the case the token-only
 * assertion above cannot catch. */
TEST(history_pass_writes_the_path_into_the_row_list) {
    const int n_tok = 6, n_aux = 5;
    Slab s(1, n_tok);
    Tree t;
    s.tok[0] = 100;
    for (int j = 1; j <= n_aux; ++j) s.tok[j] = t.tok[j];
    for (int j = 0; j <= n_aux; ++j) s.src[j] = t.pick[j];

    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = t.parent;
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 50;
    s.st[0].ctx = 50;
    adv_row(s.a, 0);

    /* the history pass, with the host having staged the path's rows at a non-zero base */
    const int32_t row0 = 7;
    s.a.mode = ADV_HIST;
    s.a.spec_parent = nullptr;
    s.set_row(0, 0, 0, /*aux=*/0, true);
    s.cu[0] = 0;
    s.q[0] = 4;
    s.out[0] = row0;
    adv_row(s.a, 0);

    /* ROWS 7, 9, 11, 12 -- the accepted nodes at that base, NOT 7, 8, 9, 10. */
    CHECK_EQ(s.out[0], row0 + 0);
    CHECK_EQ(s.out[1], row0 + 2);
    CHECK_EQ(s.out[2], row0 + 4);
    CHECK_EQ(s.out[3], row0 + 5);

    /* and the run is exactly what a depth-indexed write would have produced */
    CHECK(s.out[1] != row0 + 1);
    CHECK(s.out[2] != row0 + 2);
    CHECK(s.out[3] != row0 + 3);
}

/* 8. WITH A LEAD IN FRONT OF THE VERIFIED ROWS, THE TWO HAVE DIFFERENT BASES. The lead starts at
 * the host's first staged row; the verified rows start at the staged row of the first ACCEPTED
 * NODE, which is a different row of the tap plane. A patch that uses one base for both drops the
 * prefix offset on every verified row -- and the tokens stay right, so only the row list shows it.
 *
 * Shape: `aux = 2` lead positions (trunk rows 10, 11), then the accepted path {0, 2, 4, 5} of a
 * verify step whose first verified row is 10 + 1 + n_spec... here simply staged at 40. */
TEST(history_pass_keeps_the_lead_and_verified_bases_apart) {
    const int n_tok = 6, n_aux = 5;
    Slab s(1, n_tok);
    Tree t;
    s.tok[0] = 100;
    for (int j = 1; j <= n_aux; ++j) s.tok[j] = t.tok[j];
    for (int j = 0; j <= n_aux; ++j) s.src[j] = t.pick[j];

    s.a.mode = ADV_ACCEPT;
    s.a.spec_parent = t.parent;
    s.set_row(0, 0, 0, n_aux, true);
    s.cu[1] = n_tok;
    s.q[0] = n_tok;
    s.ctx[0] = 50;
    s.st[0].ctx = 50;
    adv_row(s.a, 0);
    CHECK_EQ(s.st[0].acc, 3);

    /* the history pass: 2 lead positions then 4 verified ones */
    const int32_t lead_base = 10;      /* where the lead's trunk rows begin */
    const int32_t ver_base  = 40;      /* where the first accepted node's row sits, elsewhere */
    s.a.mode = ADV_HIST;
    s.a.spec_parent = nullptr;
    s.set_row(0, 0, 0, /*aux=*/2, true);
    s.cu[0] = 0;
    s.q[0] = 6;
    s.out[0] = lead_base;
    s.out[1] = lead_base + 1;
    s.out[2] = ver_base;               /* the staged row of node 0 -- the verified base */
    adv_row(s.a, 0);

    /* THE LEAD IS ITS OWN RUN. */
    CHECK_EQ(s.out[0], lead_base + 0);
    CHECK_EQ(s.out[1], lead_base + 1);

    /* THE VERIFIED ROWS ARE THE PATH AT THE VERIFIED BASE, not the lead base. */
    CHECK_EQ(s.out[2], ver_base + 0);
    CHECK_EQ(s.out[3], ver_base + 2);
    CHECK_EQ(s.out[4], ver_base + 4);
    CHECK_EQ(s.out[5], ver_base + 5);

    /* THE PREFIX OFFSET IS NOT DROPPED: a single-base patch writes node 2 as lead_base + 2 here. */
    CHECK(s.out[3] != lead_base + 2);
    CHECK(s.out[4] != lead_base + 4);
}

RAD_TEST_MAIN()
