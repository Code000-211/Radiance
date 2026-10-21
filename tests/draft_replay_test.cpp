/* draft_replay_test.cpp -- strand D on the host: a draft cycle's replay keys, and the pass
 * accounting that names the round a pass came from.
 *
 * A draft cycle runs one trunk pass (draft_pass 0), one history pass (draft_pass < 0, which also
 * carries round 1 when n_draft_out is non-zero) and rounds 2..K (draft_pass > 0). Each of those has
 * its own shapes -- rows, window, KV geometry -- and must own a tape of its own, or a round plays
 * another round's packets.
 *
 * What the host cannot check is the tape itself: the host backend records no passes
 * (tape_supported() is false), so played/recorded/audited, eviction and re-recording are exercised
 * only by a run on the card. The key and the classification are the two things a wrong draft cycle
 * breaks first, and both are ordinary C++ here.
 */
#include "rad_test.h"

#include "runtime/ctx.h"

#include <cstring>

using rad::Ctx;
using rad::CtxDesc;
using rad::Program;

static void no_arch_step(RadCtx*, const RadBatch*) {}

struct ReplayFixture {
    Program  P;
    Ctx      ctx;
    RadBatch b{};

    ReplayFixture() {
        P.meta.n_layers = 2;
        P.meta.n_expert = 4;
        P.arena_bytes   = 8192;
        P.scratch_bytes = 256;
        P.weights.resize(1);   /* handles are 1-based; index 0 is the sentinel */
        P.buffers.resize(1);
        P.ops.resize(1);
    }

    int start() {
        CtxDesc d;
        d.program        = &P;
        d.arch_step      = no_arch_step;
        d.record_passes  = true;   /* the backend decides: the host backend cannot record */
        return ctx.init(d);
    }

    /* The batch of one pass of a draft cycle: `draft_pass` selects the kind, `rows` the active
     * rows (E2 shrinks these round by round; padding is arena space and is never a row here),
     * `window` the verify/draft K, and `draft_out` the history pass's round-1 rows. */
    const RadBatch& pass(int draft_pass, int64_t rows, int window, int draft_out, int step_no,
                         int64_t ahead = 0) {
        std::memset(&b, 0, sizeof b);
        std::memset(&kv, 0, sizeof kv);
        b.phase     = RAD_PHASE_DECODE;
        b.step      = step_no;
        b.n_tok     = rows;
        b.n_seq     = 1;
        b.n_out     = 1;
        b.max_q_len = 1;
        b.max_ctx_len     = 4096;
        b.n_kv_groups     = 1;
        b.kv              = &kv;
        b.n_spec          = window;
        b.draft_pass      = draft_pass;
        b.n_draft_out     = draft_out;
        b.n_ahead         = ahead;
        b.n_seq_decode    = 1;
        b.n_tok_decode    = rows;
        b.max_q_len_decode = 1;
        b.token_ids = tok;
        b.positions = tok;
        b.cu_seqlens = cu;
        b.seq_ids = ids;
        b.out_ids = ids;
        b.q_lens = ids;
        b.ctx_lens = ids;
        b.num_accepted = ids;
        b.draft_out_ids = ids;
        kv.group            = 0;
        kv.slot_mapping     = ids;
        kv.block_table      = ids;
        kv.block_table_pitch = 64;
        kv.seqused          = ids;
        kv.state_index      = ids;
        kv.state_index_pitch = 1;
        kv.block_size       = 16;
        kv.max_blocks       = 64;
        return b;
    }

    static int32_t tok[64], cu[8], ids[8];
    RadKVGroupBatch kv{};
};

int32_t ReplayFixture::tok[64];
int32_t ReplayFixture::cu[8];
int32_t ReplayFixture::ids[8];

/* Every shape a cycle runs is its own tape, and the step number alone is not a shape. */
TEST(a_draft_cycle_keys_every_shape_it_runs) {
    ReplayFixture f;
    CHECK_OK(f.start());

    /* The trunk verifies 1 + K tokens; the history pass carries the same rows, minus the draft
     * tail, and carries round 1 only when it drafts. */
    const uint64_t trunk = f.ctx.pass_key(&f.pass(0, 7, 6, 0, 100));
    const uint64_t hist  = f.ctx.pass_key(&f.pass(-1, 7, 0, 2, 100));
    const uint64_t hist0 = f.ctx.pass_key(&f.pass(-1, 7, 0, 0, 100));
    CHECK(trunk != hist);
    CHECK(hist != hist0);   /* a history pass that drafts is a different pass */

    /* A repeat of the same pass at a later step must hit the same tape. */
    CHECK_EQ(f.ctx.pass_key(&f.pass(0, 7, 6, 0, 101)), trunk);

    /* Round r is a different pass from round r', even at the same row count: the head reads the
     * round-before's output. */
    uint64_t round[7] = {};
    for (int r = 2; r <= 6; ++r) {
        round[r] = f.ctx.pass_key(&f.pass(r, 1, 6, 0, 100));
        for (int q = 2; q < r; ++q) CHECK(round[r] != round[q]);
        CHECK(round[r] != hist);
        CHECK(round[r] != trunk);
    }

    /* E2 shrinks the active rows. Two rounds of the same window with different active rows are
     * different passes, and so are a shrinking and a re-expanded round: the key reads the row
     * count, not the depth alone. */
    const uint64_t r3_two = f.ctx.pass_key(&f.pass(3, 2, 6, 0, 100));
    const uint64_t r3_one = f.ctx.pass_key(&f.pass(3, 1, 6, 0, 100));
    CHECK(r3_two != r3_one);
    CHECK(r3_two != round[3]);

    /* K is part of the shape: speculation off (K=0) must not replay the K=6 trunk. */
    CHECK(f.ctx.pass_key(&f.pass(0, 1, 0, 0, 100)) != trunk);

    /* A step that carries prompt tokens ahead is a different pass (the lookahead arrays are read
     * by the op path). */
    CHECK(f.ctx.pass_key(&f.pass(0, 7, 6, 0, 100, 4)) != trunk);
}

/* The counters attribute each pass to its pass kind. On the host backend every pass walks the
 * issue path, which is what the `issued` column counts; the tape outcomes stay zero because no
 * backend here can record. The block drafter's query pass (`draft_pass == 1`, a value the MTP
 * contract never uses) is its own kind: it must not be counted as a serial round. */
TEST(the_pass_counters_name_the_kind_of_each_pass) {
    ReplayFixture f;
    CHECK_OK(f.start());
    CHECK_EQ(f.ctx.records_passes(), rad::tape_supported());

    /* trunk, history, block query, two serial rounds */
    const int kinds[5] = { 0, -1, 1, 2, 4 };
    for (int i = 0; i < 5; ++i) {
        const RadBatch& b = f.pass(kinds[i], kinds[i] > 1 ? 1 : 7, kinds[i] == 0 ? 6 : 0, 0, 10 + i);
        CHECK_OK(f.ctx.run_step(&b));
    }

    const Ctx::PassCounts c = f.ctx.pass_counts();
    CHECK_EQ(c.kind[Ctx::kPassTrunk].issued, 1ll);
    CHECK_EQ(c.kind[Ctx::kPassContext].issued, 1ll);
    CHECK_EQ(c.kind[Ctx::kPassBlockQuery].issued, 1ll);
    CHECK_EQ(c.kind[Ctx::kPassDraft].issued, 2ll);
    CHECK_EQ(c.kind[Ctx::kPassTrunk].played, 0ll);
    CHECK_EQ(c.kind[Ctx::kPassContext].played, 0ll);
    CHECK_EQ(c.kind[Ctx::kPassBlockQuery].played, 0ll);
    CHECK_EQ(c.kind[Ctx::kPassDraft].played, 0ll);
    CHECK_EQ(c.played, 0ll);

    /* The whole set walked the issue path: the host backend holds no tape to play. */
    CHECK_EQ(c.issued_unkeyed, 5ll);
    CHECK_EQ(c.issued_unkeyed, c.kind[Ctx::kPassTrunk].issued + c.kind[Ctx::kPassContext].issued +
                              c.kind[Ctx::kPassBlockQuery].issued + c.kind[Ctx::kPassDraft].issued);

    /* Coverage of a cache that never held a tape: a real run fills these. `rerecorded` is an
     * observation that under-counts past 32 evictions, not a verdict on the cache size. */
    CHECK_EQ(c.keys_live, 0ll);
    CHECK_EQ(c.keys_peak, 0ll);
    CHECK_EQ(c.evicted, 0ll);
    CHECK_EQ(c.rerecorded, 0ll);

    /* `recorded` counts attempts, `recorded_ok` the attempts that produced a tape. A host backend
     * records nothing, so neither moves -- and no success can outnumber an attempt. */
    CHECK_EQ(c.recorded, 0ll);
    CHECK_EQ(c.recorded_ok, 0ll);
    for (int k = 0; k < Ctx::kPassKinds; ++k) {
        CHECK(c.kind[k].recorded_ok <= c.kind[k].recorded);
        CHECK_EQ(c.kind[k].recorded, 0ll);
    }
}

/* The evicted-key ring remembers the last 32 evictions. A key evicted earlier is forgotten, so
 * `rerecorded` under-counts instead of inventing evidence; a key it does remember is consumed by
 * the first re-record, so one eviction is counted once. */
TEST(the_evicted_ring_is_bounded_and_counts_a_re_record_once) {
    Ctx::EvictedRing ring;

    /* Forty evictions into 32 slots: the first eight are gone, the rest are recognised. */
    for (uint64_t k = 1; k <= 40; ++k) ring.note(k);
    for (uint64_t k = 1; k <= 8; ++k) CHECK(!ring.take(k));
    for (uint64_t k = 9; k <= 40; ++k) CHECK(ring.take(k));

    /* Consumed on the first take: a second record of the same key is not a second re-record. */
    CHECK(!ring.take(9));
    CHECK(!ring.take(40));

    ring.note(7);
    CHECK(ring.take(7));
    CHECK(!ring.take(7));

    /* A key that was never evicted is never a re-record. */
    CHECK(!ring.take(12345));
}

RAD_TEST_MAIN()
