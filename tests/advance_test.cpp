/* advance_test.cpp -- the card's bookkeeping between passes (core/sched/advance_impl.h), run on
 * host arrays, against the host's own acceptance rule and hand-computed rows.
 *
 * advance_impl.h is shared by the kernel and the host backend, so these rows are the arithmetic
 * the card runs. The slab below is carved the way BatchBuilder carves a build: one int32 array
 * per field, a row descriptor block, AdvGroup records and per-group tables.
 */
#include "rad_test.h"

#include "sample/accept.h"
#include "sched/advance_impl.h"
#include "sched/tree_dispatch.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace rad;

namespace {

static_assert(sizeof(AdvGroup) == 48, "AdvGroup's layout is part of the card's slab");

/* Word offsets of every array the advance reads or writes. Up to kSeq rows, kTok tokens and
 * kGroups KV groups, each group with a kPitch-entry block table per row. */
struct Slab {
    static constexpr int kSeq = 8, kTok = 64, kPitch = 16, kGroups = 2;
    enum : int32_t {
        ROWS  = 0,                              /* [4][kSeq] row descriptors */
        TOK   = ROWS + 4 * kSeq,
        POS   = TOK + kTok,
        CTX   = POS + kTok,
        ACC   = CTX + kSeq,
        OUT   = ACC + kSeq,
        SID   = OUT + kTok,
        CU    = SID + kSeq,
        Q     = CU + kSeq + 8,
        ROPE  = Q + kSeq,                       /* [3][kTok] */
        GRP   = ROPE + 3 * kTok,                /* [kGroups] AdvGroup, 12 words each */
        SLOT  = GRP + 12 * kGroups,             /* [kGroups][kTok] */
        USED  = SLOT + kGroups * kTok,          /* [kGroups][kSeq] */
        BT    = USED + kGroups * kSeq,          /* [kGroups][kSeq][kPitch] */
        FIRST = BT + kGroups * kSeq * kPitch,   /* [kGroups][kSeq] */
        PAR   = FIRST + kGroups * kSeq,         /* [kTok] tree parent ids, ADV_TREE_ACCEPT */
        END   = PAR + kTok
    };
    alignas(16) int32_t w[END] = {};
    StepSlot st[kSeq];
    int32_t  src[3 * kTok] = {};
    int32_t  eog[8] = {};
    AdvanceArgs a;

    /* `n_groups` groups: group 0 paged with `block` tokens a block, group 1 (when present) a
     * state group. Every block table starts as block b at entry b, so slot == position. */
    Slab(const Slab&) = delete;   /* `a` points into the object */
    explicit Slab(int mode, int n_groups = 1, int block = 16) {
        std::memset(st, 0x7f, sizeof st);
        a.mode = mode;
        a.n_groups = n_groups;
        a.slab = reinterpret_cast<char*>(w);
        a.row_off = ROWS * 4;  a.row_pitch = kSeq;
        a.tok_off = TOK * 4;   a.pos_off = POS * 4; a.ctx_off = CTX * 4; a.acc_off = ACC * 4;
        a.out_off = OUT * 4;   a.sid_off = SID * 4; a.cu_off = CU * 4;   a.q_off = Q * 4;
        a.grp_off = GRP * 4;   a.parent_off = PAR * 4;
        a.st = st;
        a.src = src;
        a.eog = eog;
        for (int g = 0; g < n_groups; ++g) {
            AdvGroup G{};
            G.block_size = g == 0 ? block : 1;
            G.paged = g == 0 ? 1 : 0;
            G.pitch = g == 0 ? kPitch : 1;
            G.slot_off  = (int64_t)(SLOT + g * kTok) * 4;
            G.used_off  = (int64_t)(USED + g * kSeq) * 4;
            G.bt_off    = (int64_t)(BT + g * kSeq * kPitch) * 4;
            G.first_off = (int64_t)(FIRST + g * kSeq) * 4;
            std::memcpy(reinterpret_cast<char*>(w) + GRP * 4 + g * (int)sizeof(AdvGroup), &G,
                        sizeof G);
            for (int i = 0; i < kSeq; ++i)
                for (int b = 0; b < kPitch; ++b) bt(g, i)[b] = b;
        }
    }

    int32_t* bt(int g, int i) { return &w[BT + (g * kSeq + i) * kPitch]; }
    int32_t& first(int g, int i) { return w[FIRST + g * kSeq + i]; }
    int32_t& used(int g, int i) { return w[USED + g * kSeq + i]; }
    int32_t& slot(int g, int t) { return w[SLOT + g * kTok + t]; }

    /* Row i: a device row or not, its tokens [k, k + n), its context, the StepSlot it names, and
     * the advance's row descriptor (first verified token row, first sampled row, mode number). */
    void row(int i, bool dev, int k, int n, int ctx, int card, int rtok, int rout, int aux) {
        w[ROWS + i] = dev ? 1 : 0;
        w[ROWS + kSeq + i] = rtok;
        w[ROWS + 2 * kSeq + i] = rout;
        w[ROWS + 3 * kSeq + i] = aux;
        w[CU + i] = k;
        w[CU + i + 1] = k + n;
        w[Q + i] = n;
        w[CTX + i] = ctx;
        w[SID + i] = card;
        if (i + 1 > a.n_seq) a.n_seq = i + 1;
    }
    void set_eog(std::initializer_list<int32_t> ids) {
        int n = 0;
        for (int32_t t : ids) eog[n++] = t;
        std::sort(eog, eog + n);
        a.n_eog = n;
    }
    void run() { for (int i = 0; i < a.n_seq; ++i) adv_row(a, i); }
};

uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }

}  /* namespace */

/* ================================================================== the end-of-generation cut */

/* The host's cut: the first end-of-generation token among the accepted drafts and the target's
 * own token is emitted and ends the run; a run without one is kept whole. */
TEST(the_end_of_generation_cut_keeps_the_stop_token_and_drops_what_follows) {
    const int32_t out[3] = { 5, 9, 7 };            /* two accepted drafts, then the target's */
    auto in = [](std::initializer_list<int32_t> s) {
        return [s](int32_t t) { return std::find(s.begin(), s.end(), t) != s.end(); };
    };
    CHECK_EQ(accept_cut_at_eog(out, 2, in({})), 2);
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 5 })), 0);      /* the first draft ends it */
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 9 })), 1);
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 7 })), 2);      /* the target's own token */
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 9, 5 })), 0);   /* the earliest wins */
    CHECK_EQ(accept_cut_at_eog(out, 0, in({ 9 })), 0);      /* nothing accepted past out[0] */
}

/* ================================================================== ADV_ACCEPT */

/* The card's acceptance is the host's: for any drafts, targets and end-of-generation set, a row's
 * StepSlot holds the drafts accept_greedy accepts, cut at the first end-of-generation token
 * (accept_cut_at_eog), and the token after them. The engine refuses a step where the two
 * disagree, so a drift between the copies stops serving; and the card's count positions every
 * later pass, so a wrong one writes K/V at the wrong positions. Rows that sampled nothing keep
 * their StepSlot. Seeded; a failure prints the case. */
TEST(the_cards_acceptance_is_accept_greedy_and_the_end_of_generation_cut) {
    const uint32_t seed = 0x5eed2a2u;
    uint32_t rng = seed;
    int bad = 0, cut_cases = 0, partial = 0;
    for (int c = 0; c < 10000 && !bad; ++c) {
        Slab s(ADV_ACCEPT);
        /* Tokens from {1, 2, 3}, so drafts often agree; stop tokens a subset of {2, 3, 8}. */
        int n_eog = 0;
        int32_t eog_ids[3];
        for (int32_t t : { 2, 3, 8 }) if (lcg(rng) % 3 == 0) eog_ids[n_eog++] = t;
        for (int j = 0; j < n_eog; ++j) s.eog[j] = eog_ids[j];
        s.a.n_eog = n_eog;
        auto is_eog = [&](int32_t t) {
            return std::find(eog_ids, eog_ids + n_eog, t) != eog_ids + n_eog;
        };

        const int n_rows = 1 + (int)(lcg(rng) % 4);
        int k = 0, out_row = 0;
        int32_t cards[Slab::kSeq];
        for (int i = 0; i < Slab::kSeq; ++i) cards[i] = i;
        for (int i = Slab::kSeq - 1; i > 0; --i) std::swap(cards[i], cards[lcg(rng) % (i + 1)]);
        struct Want { int card, rout, acc, tok, head; } want[4];
        for (int i = 0; i < n_rows; ++i) {
            const int kind = (int)(lcg(rng) % 4);      /* 0: mid-prefill, 1: last chunk, else decode */
            const int n_spec = kind >= 2 ? (int)(lcg(rng) % 9) : 0;
            const int n_tok = kind >= 2 ? 1 + n_spec : 2 + (int)(lcg(rng) % 5);
            const int ctx = (int)(lcg(rng) % 1000);
            const int rtok = k + n_tok - 1 - n_spec;
            const int rout = kind == 0 ? -1 : out_row;
            s.row(i, kind >= 2, k, n_tok, ctx, cards[i], rtok, rout, n_spec);
            int32_t draft[8] = {}, target[9] = {};
            for (int j = 0; j < n_tok; ++j) s.w[Slab::TOK + k + j] = 50 + j;
            for (int j = 0; j < n_spec; ++j) {
                draft[j] = 1 + (int32_t)(lcg(rng) % 3);
                s.w[Slab::TOK + rtok + 1 + j] = draft[j];
            }
            for (int j = 0; j <= n_spec; ++j) {
                target[j] = (j < n_spec && lcg(rng) % 10 < 7) ? draft[j]
                                                              : 1 + (int32_t)(lcg(rng) % 3);
                if (rout >= 0) s.src[rout + j] = target[j];
            }
            want[i] = { cards[i], rout, -1, 0, 0 };
            if (rout >= 0) {
                int32_t out[9] = {}, acc = -1;
                CHECK_OK(accept_greedy(draft, n_spec, target, out, &acc));
                const int32_t cut = accept_cut_at_eog(out, acc, is_eog);
                cut_cases += cut < acc;
                partial += cut > 0 && cut < n_spec;
                want[i].acc = cut;
                want[i].tok = out[cut];
                want[i].head = ctx + (rtok - k) + cut;
                out_row += 1 + n_spec;
            }
            k += n_tok;
        }
        s.run();
        for (int i = 0; i < n_rows && !bad; ++i) {
            const StepSlot& S = s.st[want[i].card];
            bool ok;
            if (want[i].rout < 0) {
                StepSlot untouched;
                std::memset(&untouched, 0x7f, sizeof untouched);
                ok = std::memcmp(&S, &untouched, sizeof S) == 0;
            } else {
                ok = S.acc == want[i].acc && S.tok[0] == want[i].tok && S.n_acc == want[i].acc + 1 &&
                     S.head == want[i].head && S.ctx == want[i].head + 1;
            }
            if (!ok) {
                bad = 1;
                CHECK(ok);
                std::fprintf(stderr, "    seed 0x%x, case %d, row %d: card acc %d tok %d head %d "
                             "ctx %d n_acc %d; host acc %d tok %d head %d (sampled row %d)\n",
                             seed, c, i, S.acc, S.tok[0], S.head, S.ctx, S.n_acc, want[i].acc,
                             want[i].tok, want[i].head, want[i].rout);
            }
        }
    }
    CHECK_EQ(bad, 0);
    /* The draw reaches the interesting branches, not only full acceptance or none. */
    CHECK(cut_cases > 100);
    CHECK(partial > 100);
}

/* ================================================================== ADV_TRUNK */

/* A verify step issued ahead takes its rows from the StepSlots: the bonus token and the head's
 * proposals as tokens, positions from the committed context, the rollback count, each group's
 * length and every slot through the row's own block table. Rows differ in offset, length,
 * context and StepSlot; a host row keeps what the host staged; a position past the table's pitch
 * or on an unassigned block gets no slot. A wrong row here writes K/V into another sequence's
 * blocks or reads the wrong token, with no error anywhere. */
TEST(a_step_issued_ahead_takes_every_rows_tokens_positions_and_slots_from_its_step_slot) {
    Slab s(ADV_TRUNK, /*n_groups=*/2, /*block=*/4);
    /* Row 0: three positions (two drafts) for card 2, block b of its table at 100 + 3b. */
    s.row(0, true, 0, 3, /*ctx bound=*/45, /*card=*/2, 0, 0, 2);
    for (int b = 0; b < Slab::kPitch; ++b) s.bt(0, 0)[b] = 100 + 3 * b;
    s.st[2].ctx = 37; s.st[2].n_acc = 2;
    s.st[2].tok[0] = 501; s.st[2].tok[1] = 502; s.st[2].tok[2] = 503;
    /* Row 1: a host-staged prefill row, which no advance may touch. */
    s.row(1, false, 3, 2, 70, 4, 4, -1, 0);
    s.w[Slab::TOK + 3] = 900; s.w[Slab::TOK + 4] = 901;
    s.w[Slab::POS + 3] = 70;  s.w[Slab::POS + 4] = 71;
    s.slot(0, 3) = 1234;      s.slot(0, 4) = 1235;
    s.w[Slab::ACC + 1] = 1;
    s.used(0, 1) = 72;
    /* Row 2: one position for card 0, whose second block was never assigned. */
    s.row(2, true, 5, 1, 64, 0, 5, 3, 0);
    s.bt(0, 2)[1] = -1;
    s.st[0].ctx = 6; s.st[0].n_acc = 1; s.st[0].tok[0] = 777;
    /* Row 3: three positions for card 5 in a table rebased at 8; the last falls past the
     * pitch (position 72 is block (72 - 8) / 4 = 16). */
    s.row(3, true, 6, 3, 80, 5, 6, 4, 2);
    s.first(0, 3) = 8;
    s.st[5].ctx = 70; s.st[5].n_acc = 3;
    s.st[5].tok[0] = 601; s.st[5].tok[1] = 602; s.st[5].tok[2] = 603;
    s.run();

    const int32_t tok[9]  = { 501, 502, 503, 900, 901, 777, 601, 602, 603 };
    const int32_t pos[9]  = { 37, 38, 39, 70, 71, 6, 70, 71, 72 };
    /* Paged slot = table[(p - first) / 4] * 4 + p % 4. */
    const int32_t slot[9] = { (100 + 3 * 9) * 4 + 1, (100 + 3 * 9) * 4 + 2, (100 + 3 * 9) * 4 + 3,
                              1234, 1235, -1, 15 * 4 + 2, 15 * 4 + 3, -1 };
    for (int t = 0; t < 9; ++t) {
        CHECK_EQ(s.w[Slab::TOK + t], tok[t]);
        CHECK_EQ(s.w[Slab::POS + t], pos[t]);
        CHECK_EQ(s.slot(0, t), slot[t]);
    }
    const int32_t ctx[4] = { 37, 70, 6, 70 }, acc[4] = { 2, 1, 1, 3 };
    const int32_t used0[4] = { 40, 72, 7, 65 }, used1[4] = { 40, 0, 7, 73 };
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(s.w[Slab::CTX + i], ctx[i]);
        CHECK_EQ(s.w[Slab::ACC + i], acc[i]);
        CHECK_EQ(s.used(0, i), used0[i]);   /* from the table's first position */
        CHECK_EQ(s.used(1, i), used1[i]);   /* a state group counts from 0 and has no slots */
    }
    for (int t = 0; t < 9; ++t) CHECK_EQ(s.slot(1, t), 0);
}

/* A row the card moves keeps the host's offset between its position and its rotary position on
 * all three planes (a text row after an image rotates at its index plus a shift); a row whose
 * position does not move keeps the host's three components, which differ for a media token. A
 * wrong shift misplaces every M-RoPE text row after a picture during speculation. */
TEST(a_moved_row_keeps_its_rotary_shift_and_an_unmoved_row_keeps_its_components) {
    Slab s(ADV_TRUNK);
    s.a.rope_off = Slab::ROPE * 4;
    s.a.rope_pitch = 3;
    int32_t* r = &s.w[Slab::ROPE];
    /* Row 0 (one position) moves from the bound 120 to 100; its rotary index was 120 + 7. */
    s.row(0, true, 0, 1, 120, 0, 0, 0, 0);
    s.w[Slab::POS + 0] = 120;
    r[0] = 127; r[3 + 0] = 127; r[6 + 0] = 127;
    s.st[0].ctx = 100; s.st[0].n_acc = 1; s.st[0].tok[0] = 11;
    /* Row 1 (two positions) is already where the card puts it: a media row's three components
     * stay as the host staged them. */
    s.row(1, true, 1, 2, 50, 1, 1, 1, 1);
    s.w[Slab::POS + 1] = 50; s.w[Slab::POS + 2] = 51;
    r[1] = 10; r[3 + 1] = 20; r[6 + 1] = 30;
    r[2] = 31; r[3 + 2] = 21; r[6 + 2] = 11;
    s.st[1].ctx = 50; s.st[1].n_acc = 1; s.st[1].tok[0] = 12; s.st[1].tok[1] = 13;
    s.run();
    CHECK_EQ(s.w[Slab::POS + 0], 100);
    for (int p = 0; p < 3; ++p) CHECK_EQ(r[3 * p + 0], 107);
    CHECK_EQ(r[1], 10); CHECK_EQ(r[3 + 1], 20); CHECK_EQ(r[6 + 1], 30);
    CHECK_EQ(r[2], 31); CHECK_EQ(r[3 + 2], 21); CHECK_EQ(r[6 + 2], 11);
}

/* ================================================================== the draft passes */

/* Round r reads row i's proposal at i * src_pitch: a head with a proposal buffer lays rows out at
 * its own pitch. Reading at i instead gives sequence i the proposal of sequence i / pitch, and
 * acceptance collapses with no error. Both the round's token and the final take follow it. */
TEST(draft_rounds_read_each_rows_proposal_at_the_buffers_pitch) {
    for (int mode : { (int)ADV_ROUND, (int)ADV_TAKE }) {
        Slab s(mode);
        s.a.round = 3;
        s.a.src_pitch = 3;
        for (int i = 0; i < 3; ++i) {
            s.row(i, true, i, 1, 0, /*card=*/2 - i, i, i, 0);
            s.st[2 - i].head = 100 * (i + 1);
            for (int c = 0; c < 3; ++c) s.src[i * 3 + c] = 1000 * (i + 1) + c;
        }
        s.run();
        for (int i = 0; i < 3; ++i) {
            const StepSlot& S = s.st[2 - i];
            if (mode == ADV_ROUND) {
                CHECK_EQ(S.tok[2], 1000 * (i + 1));
                CHECK_EQ(s.w[Slab::TOK + i], 1000 * (i + 1));
                CHECK_EQ(s.w[Slab::POS + i], 100 * (i + 1) + 2);   /* head + round - 1 */
                CHECK_EQ(s.w[Slab::CTX + i], 100 * (i + 1) + 2);
            } else {
                CHECK_EQ(S.tok[3], 1000 * (i + 1));
            }
        }
    }
}

/* A history pass with `aux` host-staged positions before the verified ones: those keep the host's
 * tokens, the verified positions up to the last committed one take the sampled tokens, and rows
 * past it repeat it, so no rejected draft's hidden state reaches the head's cache. */
TEST(a_history_pass_keeps_the_hosts_rows_and_repeats_the_last_committed_one) {
    Slab s(ADV_HIST);
    s.row(0, true, 0, 5, 0, 1, 0, 6, /*aux=*/2);
    s.w[Slab::TOK + 0] = 41; s.w[Slab::TOK + 1] = 42;
    s.w[Slab::OUT + 0] = 20;
    s.st[1].head = 103; s.st[1].acc = 1;
    s.src[6] = 7001; s.src[7] = 7002; s.src[8] = 7003;
    s.run();
    /* base = head - acc - aux = 100; rows reach index aux + acc = 3. */
    const int32_t pos[5] = { 100, 101, 102, 103, 103 }, tok[5] = { 41, 42, 7001, 7002, 7002 };
    for (int j = 0; j < 5; ++j) {
        CHECK_EQ(s.w[Slab::POS + j], pos[j]);
        CHECK_EQ(s.w[Slab::TOK + j], tok[j]);
        CHECK_EQ(s.w[Slab::OUT + j], 20 + (j < 3 ? j : 3));
        CHECK_EQ(s.slot(0, j), pos[j]);
    }
    CHECK_EQ(s.w[Slab::CTX], 100);
    CHECK_EQ(s.used(0, 0), 105);
}

/* What the card writes into a draft pass, for both head shapes. A verify at positions 101..105
 * accepted two drafts: committed position 103, trunk token 7003; slot p is position p. A head
 * with its own cache gets the verified run, tail repeated, context to its end; a cacheless head
 * gets the committed position and a context ending there, every round. */
TEST(the_card_stages_a_cacheless_head_at_the_committed_position) {
    auto slab = [](Slab& s, int n_rows, int round, bool shared, int shift) {
        s.a.round = round;
        s.a.shared_kv = shared ? 1 : 0;
        s.a.rot_shift = shift;
        s.row(0, true, 0, n_rows, 0, 0, 0, 0, 0);
        s.w[Slab::OUT] = 40;   /* the host staged the run's first trunk row */
        s.src[0] = 7001; s.src[1] = 7002; s.src[2] = 7003; s.src[3] = 7004; s.src[4] = 7005;
        s.st[0].head = 103; s.st[0].acc = 2; s.st[0].ctx = 104; s.st[0].n_acc = 3;
        s.st[0].tok[0] = 7003;
    };
    /* The history pass. */
    {
        Slab o(ADV_HIST);
        slab(o, 5, 0, false, 0);
        o.run();
        const int32_t pos[5] = { 101, 102, 103, 103, 103 }, tok[5] = { 7001, 7002, 7003, 7003, 7003 };
        for (int j = 0; j < 5; ++j) {
            CHECK_EQ(o.w[Slab::POS + j], pos[j]);
            CHECK_EQ(o.w[Slab::TOK + j], tok[j]);
            CHECK_EQ(o.w[Slab::OUT + j], 40 + (j < 2 ? j : 2));
            CHECK_EQ(o.slot(0, j), pos[j]);
        }
        CHECK_EQ(o.used(0, 0), 106);        /* the verify step's whole run */
        CHECK_EQ(o.w[Slab::CTX], 101);
    }
    for (int shift = 0; shift < 2; ++shift) {
        Slab s(ADV_HIST);
        slab(s, 1, 0, true, shift);
        s.run();
        CHECK_EQ(s.w[Slab::POS], 103 + shift);
        CHECK_EQ(s.w[Slab::TOK], 7003);
        CHECK_EQ(s.w[Slab::OUT], 42);
        CHECK_EQ(s.used(0, 0), 104);        /* the committed positions and no more */
        CHECK_EQ(s.w[Slab::CTX], 103);
    }
    /* Round 3: the old head two positions on, the cacheless one where it was. */
    {
        Slab o(ADV_ROUND);
        slab(o, 1, 3, false, 0);
        o.src[0] = 9001;
        o.run();
        CHECK_EQ(o.w[Slab::POS], 105);
        CHECK_EQ(o.used(0, 0), 106);
        CHECK_EQ(o.w[Slab::TOK], 9001);
        CHECK_EQ(o.st[0].tok[2], 9001);
    }
    for (int shift = 0; shift < 2; ++shift) {
        Slab s(ADV_ROUND);
        slab(s, 1, 3, true, shift);
        s.src[0] = 9001;
        s.run();
        CHECK_EQ(s.w[Slab::POS], 103 + shift);
        CHECK_EQ(s.used(0, 0), 104);
        CHECK_EQ(s.w[Slab::CTX], 103);
        CHECK_EQ(s.w[Slab::TOK], 9001);
        CHECK_EQ(s.st[0].tok[2], 9001);
    }
}

/* ================================================================== the block drafter */

/* A block drafter's query row on the card: the anchor (the verify's chosen token) at the
 * committed end, the row's mask id (`aux`) at every position after it, slots through the row's
 * own table and the group's length to the block's end. The host staged the furthest the anchor
 * could be; a host row is left as staged. A wrong base drafts from the wrong position and
 * acceptance collapses with no error; a wrong slot writes the drafter's K/V into another block. */
TEST(a_block_query_row_takes_its_anchor_and_position_from_the_step_slot) {
    Slab s(ADV_BLOCK, /*n_groups=*/1, /*block=*/4);
    /* Row 0: a block of four for card 3, staged at 60; the card has it at 50. */
    s.row(0, true, 0, 4, 60, 3, 0, -1, /*mask=*/4);
    for (int b = 0; b < Slab::kPitch; ++b) s.bt(0, 0)[b] = 100 + b;
    s.st[3].ctx = 50; s.st[3].tok[0] = 777;
    /* Row 1: host-staged, untouched. */
    s.row(1, false, 4, 2, 30, 4, 0, -1, 9);
    s.w[Slab::TOK + 4] = 900; s.w[Slab::TOK + 5] = 901;
    s.w[Slab::POS + 4] = 30;  s.w[Slab::POS + 5] = 31;
    /* Row 2: card 1 at 7 with mask 9; its block 1 (positions 4..7) was never assigned. */
    s.row(2, true, 6, 4, 20, 1, 0, -1, 9);
    s.bt(0, 2)[1] = -1;
    s.st[1].ctx = 7; s.st[1].tok[0] = 555;
    s.run();

    const int32_t tok[10] = { 777, 4, 4, 4, 900, 901, 555, 9, 9, 9 };
    const int32_t pos[10] = { 50, 51, 52, 53, 30, 31, 7, 8, 9, 10 };
    const int32_t slot[10] = { 112 * 4 + 2, 112 * 4 + 3, 113 * 4 + 0, 113 * 4 + 1, 0, 0,
                               -1, 8, 9, 10 };
    for (int t = 0; t < 10; ++t) {
        CHECK_EQ(s.w[Slab::TOK + t], tok[t]);
        CHECK_EQ(s.w[Slab::POS + t], pos[t]);
        CHECK_EQ(s.slot(0, t), slot[t]);
    }
    CHECK_EQ(s.w[Slab::CTX + 0], 50);
    CHECK_EQ(s.w[Slab::CTX + 1], 30);
    CHECK_EQ(s.w[Slab::CTX + 2], 7);
    CHECK_EQ(s.used(0, 0), 54);
    CHECK_EQ(s.used(0, 2), 11);
    CHECK_EQ(s.st[3].tok[0], 777);   /* the anchor stays the next verify's first token */
}

/* The block's proposals go into the slot behind the anchor, row i's from i * src_pitch: exactly
 * the tokens the next verify reads after its first. The anchor and host rows are untouched. */
TEST(a_block_take_puts_each_rows_proposals_behind_its_anchor) {
    Slab s(ADV_BTAKE);
    s.a.round = 3;
    s.a.src_pitch = 3;
    for (int i = 0; i < 3; ++i) {
        s.row(i, true, 4 * i, 4, 0, /*card=*/2 - i, 0, -1, 4);
        s.st[2 - i].tok[0] = 50 + i;
        for (int c = 0; c < 3; ++c) s.src[i * 3 + c] = 1000 * (i + 1) + c;
    }
    s.row(3, false, 12, 4, 0, 5, 0, -1, 4);
    s.src[9] = 4242;
    s.run();
    for (int i = 0; i < 3; ++i) {
        const StepSlot& S = s.st[2 - i];
        CHECK_EQ(S.tok[0], 50 + i);
        for (int c = 0; c < 3; ++c) CHECK_EQ(S.tok[1 + c], 1000 * (i + 1) + c);
    }
    CHECK(s.st[5].tok[1] != 4242);
}

/* The chain a pipelined block-drafter step runs, row for row: the verify's acceptance, the query
 * block at the committed end, the take, then the step issued ahead. A verify at 101..105 (four
 * drafts) accepts two, so the anchor is the target's third pick at 104, and the next verify
 * starts there with the anchor and the block's proposals. */
TEST(a_block_drafter_step_on_the_card_continues_where_the_verify_left_off) {
    StepSlot st;
    /* The verify: 7000 then drafts 7101..7104; the target picks 7101, 7102, 9999, ... */
    {
        Slab v(ADV_ACCEPT);
        v.row(0, false, 0, 5, 101, 1, 0, 0, 4);
        const int32_t t[5] = { 7000, 7101, 7102, 7103, 7104 };
        const int32_t y[5] = { 7101, 7102, 9999, 1, 2 };
        for (int j = 0; j < 5; ++j) { v.w[Slab::TOK + j] = t[j]; v.src[j] = y[j]; }
        v.run();
        st = v.st[1];
    }
    CHECK_EQ(st.acc, 2);
    CHECK_EQ(st.ctx, 104);
    CHECK_EQ(st.tok[0], 9999);
    /* The query block of four, staged at the furthest anchor, 106. */
    {
        Slab q(ADV_BLOCK);
        q.st[1] = st;
        q.row(0, true, 0, 4, 106, 1, 0, -1, 4);
        q.run();
        const int32_t tok[4] = { 9999, 4, 4, 4 };
        for (int j = 0; j < 4; ++j) {
            CHECK_EQ(q.w[Slab::TOK + j], tok[j]);
            CHECK_EQ(q.w[Slab::POS + j], 104 + j);
        }
        st = q.st[1];
    }
    /* Its three proposals, then the next verify of 1 + 3. */
    {
        Slab t(ADV_BTAKE);
        t.st[1] = st;
        t.a.round = 3;
        t.a.src_pitch = 3;
        t.row(0, true, 0, 4, 106, 1, 0, -1, 4);
        t.src[0] = 8001; t.src[1] = 8002; t.src[2] = 8003;
        t.run();
        st = t.st[1];
    }
    {
        Slab n(ADV_TRUNK);
        n.st[1] = st;
        n.row(0, true, 0, 4, 106, 1, 0, 0, 3);
        n.run();
        const int32_t tok[4] = { 9999, 8001, 8002, 8003 };
        for (int j = 0; j < 4; ++j) {
            CHECK_EQ(n.w[Slab::TOK + j], tok[j]);
            CHECK_EQ(n.w[Slab::POS + j], 104 + j);
        }
        CHECK_EQ(n.w[Slab::CTX], 104);
        CHECK_EQ(n.w[Slab::ACC], 3);
    }
}

RAD_TEST_MAIN()

/* ================================================================== ADV_TREE_ACCEPT */

/* The card walks a tree verify's nodes exactly as the host does (tree_dispatch.h's
 * tree_dispatch_commit, on verify_tree_greedy): the accepted children's tokens, the pick at the
 * node the walk stops at, and the end-of-generation cut. A drift here emits another branch's
 * tokens than the host streams, or leaves the draft head at a position the winner never reached.
 * Seeded over random topologies, tokens, picks and stop sets. */
TEST(the_cards_tree_walk_commits_the_hosts_winner_and_cuts_at_the_stop_token) {
    CHECK_EQ(kAdvTreeMaxNodes, kTreeMaxNodes);
    const uint32_t seed = 0x7ee1234u;
    uint32_t rng = seed;
    int bad = 0, deep = 0, cut_cases = 0, none = 0;
    for (int c = 0; c < 4000 && !bad; ++c) {
        Slab s(ADV_TREE_ACCEPT);
        CandidateTree t;
        CHECK(t.add(-1, 1 + lcg(rng) % 3));
        const int n = 1 + (int)(lcg(rng) % 8);
        for (int j = 1; j < n; ++j)
            CHECK(t.add((int)(lcg(rng) % j), 1 + lcg(rng) % 3));

        int32_t eog_ids[3];
        int n_eog = 0;
        for (int32_t tok : { 2, 3, 8 }) if (lcg(rng) % 3 == 0) eog_ids[n_eog++] = tok;
        for (int j = 0; j < n_eog; ++j) s.eog[j] = eog_ids[j];
        s.a.n_eog = n_eog;
        auto is_eog = [&](int32_t x) {
            return std::find(eog_ids, eog_ids + n_eog, x) != eog_ids + n_eog;
        };

        int32_t pick[kTreeMaxNodes];
        for (int j = 0; j < n; ++j) {
            s.w[Slab::TOK + j] = (int32_t)t[j].token;
            s.w[Slab::PAR + j] = t[j].parent;
            /* Mostly the parent's own pick, sometimes another token: both branches. */
            pick[j] = (j > 0 && lcg(rng) % 10 < 7) ? (int32_t)t[j].token : 1 + (int32_t)(lcg(rng) % 3);
            s.src[j] = pick[j];
        }
        const int ctx = 1 + (int)(lcg(rng) % 1000);
        s.row(0, true, 0, n, ctx, 0, 0, 0, n);
        s.run();

        const StepSlot& S = s.st[0];
        int32_t out[kTreeMaxNodes] = {};
        int32_t path[kTreeMaxNodes] = {};
        int32_t acc = 0;
        CHECK_OK(accept_tree_greedy(t, pick, n, out, &acc, path));
        const int32_t cut = accept_cut_at_eog(out, acc, is_eog);
        cut_cases += cut < acc;
        none += cut == 0;
        deep += cut >= 2;
        const bool ok = S.acc == cut && S.tok[0] == out[cut] && S.n_acc == cut + 1 &&
                        S.head == ctx + cut && S.ctx == S.head + 1;
        if (!ok) {
            bad = 1;
            CHECK(ok);
            std::fprintf(stderr, "    seed 0x%x, case %d, %d nodes: card acc %d tok %d head %d "
                         "ctx %d n_acc %d; host acc %d tok %d (cut %d)\n",
                         seed, c, n, S.acc, S.tok[0], S.head, S.ctx, S.n_acc, cut, out[cut], cut);
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(deep > 100);
    CHECK(cut_cases > 100);
    CHECK(none > 100);

    /* A row the host did not stage as a tree (no pick, or a node past the capacity) is refused by
     * the host's cross-check instead of being committed. */
    {
        Slab s(ADV_TREE_ACCEPT);
        s.w[Slab::TOK + 0] = 1;
        s.w[Slab::TOK + 1] = 2;
        s.w[Slab::PAR + 0] = -1;
        s.w[Slab::PAR + 1] = 0;
        s.row(0, true, 0, 2, 10, 0, 0, 0, 2);
        s.run();
        CHECK_EQ(s.st[0].acc, 0);
        std::memset(&s.st[0], 0, sizeof s.st[0]);
        s.w[Slab::Q + 0] = kTreeMaxNodes + 1;   /* a tree the row cannot hold */
        s.run();
        CHECK_EQ(s.st[0].acc, -1);   /* what the host refuses the step on */
        std::memset(&s.st[0], 0, sizeof s.st[0]);
        s.w[Slab::Q + 0] = 2;
        s.src[0] = -1;   /* a node without a pick */
        s.run();
        CHECK_EQ(s.st[0].acc, -1);
    }
}
