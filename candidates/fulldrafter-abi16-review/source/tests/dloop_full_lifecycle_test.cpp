/* dloop_full_lifecycle_test -- THE ENGINE'S METADATA, ON THE PATH THE PERSISTENT CACHE IS BUILT
 * FROM. CPU only: no device, no engine object, no model file, no GPU.
 *
 * WHAT THIS FILE IS FOR. The full-context DLoop drafter's correctness rests on one thing the
 * kernels cannot check about themselves: that the engine stages, for every step, the metadata the
 * append op needs -- and that the metadata names THE ROWS THE VERIFIED PREFIX ACTUALLY OCCUPIES.
 * Get that wrong and every kernel still agrees with its oracle on the operands it was handed,
 * while the cache fills with the wrong positions: the failure mode the plan calls out by name.
 *
 * So these cases drive Engine::collect_step -- the real function, on a real StepPlan, with a real
 * BatchBuilder's tree staging -- and assert on the buffers it produces:
 *
 *   MULTI-CHUNK PREFILL > 32. Four chunks of a 48-token prompt, the first three of which sample
 *   nothing. `n_out == 0` used to mean "no tap rows at all", and the whole point of the
 *   full-context contract is that those chunks are appended like any other: a span whose count is
 *   the chunk length, a reset only on the first, and a running row start that ends at the sum.
 *
 *   A NON-LAST WINNING PATH. A verify step whose accepted node is NOT the first child -- the
 *   target's draw matches the root's SECOND child, so the winning path is 0,1 rather than 0,2.
 *   `absolute` must name TRUNK rows (`trunk_start + node`) and the span must select the
 *   -1-terminated path, because a winning path is a parent chain and not a contiguous run.
 *
 *   SLOT SCOPE. The reset flag belongs to an OCCUPATION and not to a slot index, and a step's
 *   spans are a RUNNING row counter over the whole batch -- not a per-sequence one, which a
 *   single-sequence test cannot tell apart.
 *
 * These check the ENGINE's output, not a copy of its logic: the assertions are on the arrays
 * collect_step wrote. A test that re-implemented the mapping and compared it with itself would
 * pass through exactly the bug this exists to catch.
 *
 * The plan document for this work is docs/DLOOP_FULL_CONTEXT_PLAN.md. */
#include "rad_test.h"
#include "vendor/nlohmann/json.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* The candidate tree and its registry, the same two-spelling include the other tests in this tree
 * use: the header is installed under `arch/` by the build, and sits at `arch/common/` in a source
 * checkout. */
#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

#define private public
#include "sched/scheduler.h"
#include "engine.h"
#undef private

#include "rad_core.h"
#include "sched/batch.h"

using rad::Config;
using rad::Engine;
using rad::Request;
using rad::ReqState;
using rad::Scheduler;
using rad::StepEntry;
using rad::StepPlan;

namespace {

/* THE TREE STAGING REGION, at the layout core/engine.cpp and core/sched/batch.cpp agree on:
 *   [0,      16S)  the node path of each accepted winner     (tree_paths)
 *   [16S,  16S+2)  the constant cu_seqlens {0,1} the QSA commit reads
 *   [16S+2, 18S+2) lengths  -- accepted path length a slot
 *   [18S+2, 19S+2) starts   -- each entry's first TRUNK row
 *   [19S+2, 35S+2) absolute -- the trunk row of each node
 *   [35S+2, 36S+2) resets   -- 1 at a sequence start
 * plus `dloop_context_span` as its own buffer, which is what the append actually reads. */
struct Staged {
    std::vector<int32_t> paths;      /* [S,16] node ids, -1 padded */
    std::vector<int32_t> lengths;    /* [S] */
    std::vector<int32_t> starts;     /* [S] trunk row of each entry */
    std::vector<int32_t> absolute;   /* [S,16] trunk rows of each node */
    std::vector<int32_t> resets;     /* [S] */
    std::vector<int32_t> span;       /* [S,2] the append's own operand */
};

/* Pull the staging region out of the mapped buffer collect_step filled. `S` is the batch's
 * n_seq, which is what the layout is expressed in, and the span lives in the region's own tail --
 * the same place the engine writes it, so the test reads what the device would. */
Staged read_staged(const int32_t* p, int64_t S, int64_t span_at) {
    Staged s;
    if (!p) return s;
    const int32_t* paths    = p;
    const int32_t* lengths  = p + S * 16 + 2;
    const int32_t* starts   = lengths + S;
    const int32_t* absolute = starts + S;
    const int32_t* resets   = absolute + S * 16;
    s.paths.assign(paths, paths + S * 16);
    s.lengths.assign(lengths, lengths + S);
    s.starts.assign(starts, starts + S);
    s.absolute.assign(absolute, absolute + S * 16);
    s.resets.assign(resets, resets + S);
    s.span.assign(p + span_at, p + span_at + S * 2);
    return s;
}

/* THE STAGING REGION AS THE BUILDER ALLOCATES IT, `max_seqs * 35 + 2` ints plus the span tail.
 * Host-mapped memory is ordinary memory to the host, and the exact size and tail offset are what
 * make the pointer arithmetic below a real bound rather than a convention. */
int32_t* alloc_staging(int64_t max_seqs) {
    return static_cast<int32_t*>(
        std::calloc(static_cast<size_t>(max_seqs * 35 + 2 + max_seqs * 2), sizeof(int32_t)));
}
constexpr int64_t staging_span_at(int64_t max_seqs) { return max_seqs * 35 + 2; }

/* A DLOOP BACKEND DECLARES A SPAN BUFFER AND THE RING ONE DOES NOT, and collect_step stages the
 * pairs only in the first case. The tests below are all about the span, so they stand the engine up
 * exactly as a full-context deployment would: the handle is found by name at startup, and a
 * non-zero stand-in is enough for the host half -- collect_step only asks whether one exists. */
void declare_span(Engine& e, int64_t max_seqs) {
    e.span_buf_ = 1;
    e.sched_.builder_.span_at_ = staging_span_at(max_seqs);
    e.sched_.builder_.dloop_full_ = true;
}

/* A batch with nothing but the fields collect_step reads. Every pointer is null: this case drives
 * the host half, which never dereferences a device pointer on the path under test. */
RadBatch make_batch(int64_t n_seq, int64_t n_tok, int step) {
    RadBatch b{};
    b.step = step;
    b.n_seq = n_seq;
    b.n_tok = n_tok;
    b.phase = RAD_PHASE_DECODE;
    return b;
}

/* ONE ENTRY, and only the fields the staging reads are meaningful. `produces_token` is the
 * scheduler's answer to "does this entry's last token end the sequence", and it is what decides
 * whether the entry contributes a logits row at all. `ctx_len` is the tokens computed BEFORE this
 * step, which is what makes an entry the START of its sequence's run through this drafter. */
StepEntry entry(uint64_t seq, int32_t card, int32_t n_tokens, bool prefill, bool produces,
                int32_t ctx_len = 0, int32_t n_spec = 0) {
    StepEntry e{};
    e.seq = seq;
    e.slot = static_cast<int32_t>(seq);
    e.card = card;
    e.n_tokens = n_tokens;
    e.ctx_len = ctx_len;
    e.n_spec = n_spec;
    e.is_prefill = prefill;
    e.produces_token = produces;
    return e;
}

}  /* namespace */

/* ============================================================================
 * A MULTI-CHUNK PREFILL: EVERY CHUNK IS APPENDED, INCLUDING THE SILENT ONES
 * ============================================================================
 *
 * 48 prompt tokens in four chunks of 16/16/8/8, so the first three sample nothing and only
 * the last does. Each step reuses the trunk tap planes from row zero; the cache append position
 * instead comes from its persistent length. Reset is set only on the first chunk.
 *
 * The alternative this rules out is the old behaviour, where a step with n_out == 0 recorded no
 * tap rows and the drafter's context silently lost the middle of every long prompt. */
TEST(multi_chunk_prefill_stages_every_chunk_including_silent_ones) {
    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    declare_span(engine, 8);

    const int64_t S = 1;
    sched.builder_.max_seqs_ = 8;
    sched.builder_.tree_paths_ = alloc_staging(8);
    REQUIRE(sched.builder_.tree_paths_ != nullptr);

    const int32_t chunk[4] = {16, 16, 8, 8};
    int32_t staged_rows = 0;
    int32_t expect_ctx = 0;
    for (int k = 0; k < 4; ++k) {
        const bool last = (k == 3);
        sched.plan_.e.clear();
        /* ctx_len is the tokens BEFORE this step, so it is where the previous chunk left off --
         * which is what makes the second, third and fourth chunks NOT sequence starts. */
        sched.plan_.e.push_back(entry(7001, 0, chunk[k], /*prefill=*/true, /*produces=*/last,
                                      /*ctx_len=*/expect_ctx));
        sched.plan_.n_tok = chunk[k];

        RadBatch b = make_batch(S, chunk[k], /*step=*/k + 1);
        Engine::StepOut so;
        so.first_row.assign(1, last ? 0 : -1);
        so.n_rows = last ? 1 : 0;
        /* THE READBACK IS ITS OWN BUFFER in production -- collect_step memcpy's it into so.rows --
         * so it is its own buffer here too, rather than an alias of the vector it is copied into. */
        int32_t sample[1] = {42};
        if (last) so.rb = sample;
        REQUIRE_OK(engine.collect_step(&b, so));

        Staged st = read_staged(sched.builder_.tree_paths_, S, sched.builder_.span_at_);
        /* These are row coordinates inside THIS step's tap planes, not absolute token positions. */
        CHECK_EQ(st.starts[0], 0);
        CHECK_EQ(st.span[0], 0);
        /* count >= 0 names the run, and the run is the whole chunk -- sampled or not. */
        CHECK_EQ(st.span[1], chunk[k]);
        /* RESET ONLY ON THE FIRST CHUNK. Every later chunk must append, and this assertion is the
         * one that fails if the flag goes back to "is this a prefill entry": a second chunk
         * carrying 1 would CLEAR the K/V the first chunk just wrote, and a long prompt would end
         * with the drafter believing the sequence began at its last chunk. */
        CHECK_EQ(st.resets[0], k == 0 ? 1 : 0);
        staged_rows += st.span[1];
        expect_ctx += chunk[k];
    }
    /* Silent chunks must contribute their rows just like the final sampled chunk. */
    CHECK_EQ(staged_rows, 48);

    std::free(sched.builder_.tree_paths_);
    sched.builder_.tree_paths_ = nullptr;
}

/* ============================================================================
 * A NON-LAST WINNING PATH ON A DECODE STEP
 * ============================================================================
 *
 * A tree whose accepted leaf is node 2 -- NOT the first child, so the winning path leaves the
 * trunk's first row and the mapping from node index to tap row is not the identity. `absolute`
 * must carry TRUNK rows, and every one of them is a row of the tap planes.
 *
 * The span must select the PATH (count == -1) rather than a run: a winning path is a set of nodes
 * with a parent chain, and a contiguous [start,count] would name the siblings the target REJECTED.
 *
 * This is the case the plan's acceptance criterion names, and it is the one that catches the
 * regression an out-row mapping produces: `first_row[i] + node` and `trunk_start[i] + node` are the
 * same number only on a step where every sequence is one row and no sequence precedes this one. */
TEST(nonlast_winner_path_stages_trunk_rows_and_selects_the_path) {
    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    declare_span(engine, 8);

    const int64_t S = 1;
    sched.builder_.max_seqs_ = 8;
    sched.builder_.tree_paths_ = alloc_staging(8);
    REQUIRE(sched.builder_.tree_paths_ != nullptr);

    sched.plan_.e.clear();
    /* ONE decode entry with a speculative window, so collect_step runs the accept path. */
    sched.plan_.e.push_back(entry(7101, 0, 4, /*prefill=*/false, /*produces=*/true,
                                  /*ctx_len=*/0, /*n_spec=*/3));
    sched.plan_.n_tok = 4;

    RadBatch b = make_batch(S, 4, /*step=*/9);
    Engine::StepOut so;
    so.first_row.assign(1, 0);
    so.n_rows = 4;
    /* The target's answer at each verified row, in its own buffer: collect_step copies the readback
     * into `so.rows`, so `rb` must not alias it. */
    int32_t target[4] = {0, 0, 0, 0};
    so.rb = target;

    /* A DRAFT TREE WHOSE WINNER IS NOT THE FIRST CHILD. Root 11 has two children in topological
     * order: node 1 proposes 12 and node 2 proposes 90. The target's draw at node 0's row is 90 --
     * the SECOND root child's token -- so the accepted node is 2 and the winning path is 0,2 rather
     * than 0,1. That is what makes the mapping from node to tap row non-identity at the FIRST
     * accepted position, and it is the case a tree whose winner is always node 0 cannot see.
     *
     * `e.draft` carries the proposed tokens and `e.req` must exist for the acceptance path to run
     * at all. */
    int32_t draft[3] = {11, 12, 90};
    Request req{};
    req.id = 7101;
    req.state = ReqState::Running;
    sched.plan_.e[0].draft = draft;
    sched.plan_.e[0].n_draft = 3;
    sched.plan_.e[0].req = &req;
    /* THE TARGET'S DRAWS, one row a node, written into the readback the engine will copy in. Row 0
     * is the root's own position and the answer there is 90 -- the SECOND root child -- which is
     * what makes the accepted node 2 rather than 1. */
    target[0] = 90;
    target[1] = 77;   /* diverges at the next position, so the path stops there */

    radiance::CandidateTree tree;
    REQUIRE(tree.add_node(0, -1, 11u, 0));
    REQUIRE(tree.add_node(1, 0, 12u, 1));
    REQUIRE(tree.add_node(2, 0, 90u, 1));
    radiance::TreeRegistry::instance().clear_all();
    radiance::TreeRegistry::instance().set(7101, tree, 0);

    REQUIRE_OK(engine.collect_step(&b, so));

    Staged st = read_staged(sched.builder_.tree_paths_, S, sched.builder_.span_at_);
    /* THE WINNER IS NODE 2 AND NOT NODE 1, which is the whole point of the case. */
    CHECK_EQ(st.paths[0], 0);
    CHECK_EQ(st.paths[1], 2);
    /* THE TRUNK-ROW MAPPING. The node's tap row is its TRUNK row, `trunk_start + node`, and NOT a
     * row of the logits buffer -- the two differ by exactly the rows the entries before it occupy,
     * which on this single-sequence step is the node index itself. */
    CHECK_EQ(st.absolute[0], st.starts[0] + 0);
    CHECK_EQ(st.absolute[1], st.starts[0] + 2);
    /* THE SPAN SELECTS THE PATH, not a run: a winning path is a parent chain, and a contiguous
     * [start,count] would name the siblings the target rejected. */
    CHECK_EQ(st.span[0], 0);
    CHECK_EQ(st.span[1], -1);
    /* The reset array must not have been turned into the path lengths: they are different arrays
     * with different meanings, and the QSA commit reads the lengths from `num_accepted`. This entry
     * is at ctx_len 0, so it IS a sequence start and the flag is 1 -- what must NOT be here is the
     * number of accepted nodes. */
    CHECK_EQ(st.resets[0], 1);
    CHECK_EQ(st.lengths[0], 2);   /* accepted(1) + 1, the path length QSA replays */

    radiance::TreeRegistry::instance().clear_all();
    std::free(sched.builder_.tree_paths_);
    sched.builder_.tree_paths_ = nullptr;
}

/* ============================================================================
 * SLOT REUSE: RESET IS A PROPERTY OF THE OCCUPATION, NOT OF THE SLOT INDEX
 * ============================================================================
 *
 * One step, two prefill entries: sequence A is already mid-prompt (ctx_len 48, so it is a LATER
 * chunk of a run the drafter has been appending) and sequence B was just admitted (ctx_len 0, its
 * first chunk). The two must carry DIFFERENT reset flags even though both are prefill entries and
 * both occupy a slot -- which is what a rule keyed on `is_prefill` gets wrong.
 *
 * A rule keyed on the slot index gets it wrong the other way: B may well be on a slot A used
 * before, and "the slot is not new" is not a reason to keep A's length. The flag says "start the
 * cache here", and only ctx_len == 0 does.
 *
 * The two entries also carry DIFFERENT spans, in a batch whose second entry is not a continuation
 * of the first: the running row counter is the batch's, and nothing about the flag is. */
TEST(slot_reuse_resets_while_a_later_chunk_of_the_same_occupation_does_not) {
    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    declare_span(engine, 8);

    const int64_t S = 2;
    sched.builder_.max_seqs_ = 8;
    sched.builder_.tree_paths_ = alloc_staging(8);
    REQUIRE(sched.builder_.tree_paths_ != nullptr);

    sched.plan_.e.clear();
    /* A is mid-prompt: 48 tokens are already computed, so its cache must KEEP growing.
     * B was just admitted: ctx_len 0, so its cache must START here. */
    sched.plan_.e.push_back(entry(8001, 3, 16, /*prefill=*/true, /*produces=*/false,
                                  /*ctx_len=*/48));
    sched.plan_.e.push_back(entry(8002, 5, 16, /*prefill=*/true, /*produces=*/false,
                                  /*ctx_len=*/0));
    sched.plan_.n_tok = 32;

    RadBatch b = make_batch(S, 32, /*step=*/12);
    Engine::StepOut so;
    so.first_row.assign(2, -1);
    so.n_rows = 0;
    REQUIRE_OK(engine.collect_step(&b, so));

    Staged st = read_staged(sched.builder_.tree_paths_, S, sched.builder_.span_at_);
    /* The SECOND entry's rows start after the first's: the span is a running counter over the
     * STEP, so two entries never claim the same tap rows. */
    CHECK_EQ(st.starts[0], 0);
    CHECK_EQ(st.starts[1], 16);
    CHECK_EQ(st.span[0], 0);
    CHECK_EQ(st.span[1], 16);
    CHECK_EQ(st.span[2], 16);
    CHECK_EQ(st.span[3], 16);
    /* A CONTINUES (no reset) AND B STARTS (reset): the two flags differ in one step, which no
     * rule keyed on `is_prefill` alone can produce. */
    CHECK_EQ(st.resets[0], 0);
    CHECK_EQ(st.resets[1], 1);

    std::free(sched.builder_.tree_paths_);
    sched.builder_.tree_paths_ = nullptr;
}

/* ============================================================================
 * THE SPAN RUNS ACROSS THE WHOLE BATCH, NOT PER SEQUENCE
 * ============================================================================
 *
 * Three sequences at a time in one step: their spans must be a RUNNING row counter over the step's
 * tap planes. A per-sequence counter would work on a single-sequence step and produce three
 * overlapping appends on the first batched prefill -- which is the ordinary case for a served
 * engine and the one a single-sequence test cannot see. */
TEST(span_staging_is_a_running_row_counter_over_the_whole_batch) {
    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    declare_span(engine, 4);

    const int64_t S = 3;
    sched.builder_.max_seqs_ = 4;
    sched.builder_.tree_paths_ = alloc_staging(4);
    REQUIRE(sched.builder_.tree_paths_ != nullptr);

    sched.plan_.e.clear();
    for (int i = 0; i < 3; ++i)
        sched.plan_.e.push_back(entry(9000 + (uint64_t)i, i, 8, /*prefill=*/true, /*produces=*/false));
    sched.plan_.n_tok = 24;

    RadBatch b = make_batch(S, 24, /*step=*/20);
    Engine::StepOut so;
    so.first_row.assign(3, -1);
    so.n_rows = 0;
    REQUIRE_OK(engine.collect_step(&b, so));

    Staged st = read_staged(sched.builder_.tree_paths_, S, sched.builder_.span_at_);
    CHECK_EQ(st.starts[0], 0);
    CHECK_EQ(st.starts[1], 8);
    CHECK_EQ(st.starts[2], 16);
    CHECK_EQ(st.span[4], 16);
    CHECK_EQ(st.span[5], 8);

    std::free(sched.builder_.tree_paths_);
    sched.builder_.tree_paths_ = nullptr;
}

RAD_TEST_MAIN()