/* sched_draft_test.cpp -- what the scheduler does with a cycle's cost plan (strand C).
 *
 * draft_policy_test.cpp covers the planner's choice in isolation. This file covers the other
 * half: one plan per cycle, frozen, read row by row by every site that caps a proposal, and the
 * places where the scheduler's own limits override it. It runs against the shared fakes
 * (sched_fakes.h) and declares its own program, because sched_test.cpp's fixture is local to that
 * binary; if a third file needs it, extract it into a header with its owner rather than copy it
 * again.
 *
 * The plan is enabled through the public seam the engine will use (DraftController::Params), so
 * these tests need no config wiring: the cost policy is still not reachable from Config, and
 * activation is the integrator's step.
 */
#include "rad_test.h"

#include "sched/scheduler.h"
#include "sched/draft_cost.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sched_fakes.h"

using namespace rad;
using namespace sched_fake;

namespace {

/* ------------------------------------------------------------------ a declared program */
/* The same minimal deployment sched_test.cpp declares: a paged full group beside a linear one,
 * the op whose schema names the chunk parameter, and the derived buffers each group needs.
 * Everything below the scheduler is a fake. */
static const RadParamSpec g_gdn_ps[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    { "chunk", RAD_P_INT, RAD_DERIVED,  RAD_PROLE_SEQ_CHUNK },
};
static const RadOpSchema g_gdn_schema = { "gdn_chunk_scan", g_gdn_ps, 2, nullptr, 0, "test" };
static RadConstraint g_gdn_c[] = { RAD_CDIV("chunk", 64) };
static RadKernelInfo g_gdn_ki;
static KernelRow     g_gdn_row;

struct Fixture {
    Program prog;
    Config  cfg;
    FakeKV  kv;
    FakePC  pc;
    Scheduler sch;
    std::vector<void*> owned;

    ~Fixture() { sch.fini(); for (void* p : owned) std::free(p); }

    void build(int64_t blocks = 4096, int64_t states = 64, int64_t max_tok = 512,
               int64_t max_ctx = 4096) {
        g_gdn_ki = RadKernelInfo{};
        g_gdn_ki.name = "gdn_fake";
        g_gdn_ki.op = "gdn_chunked";
        g_gdn_ki.constraints = g_gdn_c;
        g_gdn_ki.n_constraints = 1;
        g_gdn_row = KernelRow{ &g_gdn_ki, "fake", 0, 0 };

        prog.meta.n_ctx_train = 4096;
        prog.meta.n_vocab = 512;

        /* index 0 is the sentinel every handle table in Program carries */
        prog.ops.emplace_back();
        prog.buffers.emplace_back();

        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        prog.kv_groups.push_back(attn);

        KVGroupInfo lin;
        lin.name = "gdn";
        lin.decl.kind = RAD_KV_LINEAR;
        lin.decl.state_dim[0] = 128;
        lin.decl.state_dim[1] = 128;
        prog.kv_groups.push_back(lin);

        KVGroupInfo conv;
        conv.name = "conv";
        conv.decl.kind = RAD_KV_CONV;
        conv.decl.conv_width = 4;
        prog.kv_groups.push_back(conv);

        OpInfo op;
        op.op = "gdn_chunk_scan";
        op.schema = &g_gdn_schema;
        Band b;
        b.hi = 4096;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_gdn_row;
        b.dom[RAD_DOMAIN_DEVICE].geom.set_i("chunk", 64);
        op.bands.push_back(b);
        prog.ops.push_back(op);

        add_derived("seq_start", "seq_start_index", 257);
        add_derived("chunk_bounds", "state_chunk_bounds", 4097);
        add_derived("conv_idx", "conv_state_index", 256);

        cfg.max_tok = max_tok;
        cfg.max_seqs = 32;
        cfg.max_ctx = max_ctx;
        cfg.checkpoint_interval = 256;
        cfg.prefix_cache = true;
        /* Config::n_spec defaults to -1 (`auto`), which engine_bringup resolves; this fixture
         * skips that, and DraftController::configure refuses a negative depth. */
        cfg.n_spec = 0;

        kv.setup(prog.kv_groups, cfg.max_ctx, blocks, states);
        pc.kv = &kv;
        pc.groups.clear();
        for (size_t i = 0; i < prog.kv_groups.size(); ++i)
            if (kv.geom.g[i].paged) pc.groups.push_back((int32_t)i);
    }

    void add_derived(const char* name, const char* derive, int64_t n) {
        BufferInfo b;
        b.name = name;
        b.decl.dtype = RAD_I32;
        b.decl.rank = 1;
        b.decl.shape[0] = n;
        b.decl.kind = RAD_BUF_DERIVED;
        b.decl.derive = derive;
        b.bytes = n * 4;
        void* p = std::calloc(1, (size_t)b.bytes);
        owned.push_back(p);
        b.ptr = p;
        prog.buffers.push_back(b);
    }

    int start() {
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        return sch.init(prog, cfg, &kv, &pc, rio);
    }

    uint64_t next_id = 1;
    uint64_t add_prompt(std::vector<int32_t> tokens, int max_tokens = 64) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->max_tokens = max_tokens;
        r->prompt = std::move(tokens);
        const uint64_t id = r->id;
        return sch.add(std::move(r)) < 0 ? 0 : id;
    }
    uint64_t submit(int n_prompt, int max_tokens = 64) {
        std::vector<int32_t> t((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) t[(size_t)i] = 1000 + i;
        return add_prompt(std::move(t), max_tokens);
    }
    /* A prompt whose tail repeats its head, so prompt lookup has something to extend. */
    uint64_t submit_pattern(int repeats = 2) {
        std::vector<int32_t> t;
        for (int r = 0; r < repeats; ++r)
            for (int32_t v = 1; v <= 4; ++v) t.push_back(v);
        return add_prompt(std::move(t));
    }

    /* Commit the planned step: every entry sampled `token`, ended when `eos`, and accepted
     * `acc(entry)` drafts (none when `acc` is empty). */
    int commit(int32_t token, uint8_t eos = 0,
               const std::function<int32_t(const StepEntry&)>& acc = nullptr) {
        const StepPlan& p = sch.plan();
        std::vector<int32_t> a(p.e.size(), 0), t(p.e.size(), token);
        std::vector<uint8_t> e(p.e.size(), eos);
        if (acc)
            for (size_t i = 0; i < p.e.size(); ++i) a[i] = acc(p.e[i]);
        Scheduler::StepResult r;
        r.n_seq = (int)p.e.size();
        r.n_accepted = a.data();
        r.token = t.data();
        r.eos = e.data();
        return sch.commit(r);
    }
    void commit_plain(int32_t token = 7) { CHECK_OK(commit(token)); }

    /* Run prefill chunks until the request decodes. `token` is what the last chunk emits, so a
     * repeating prompt keeps repeating. */
    void prefill(uint64_t id, int32_t token = 11) {
        for (int guard = 0; guard < 32; ++guard) {
            Request* r = sch.find(id);
            if (!r || r->n_computed >= (int32_t)r->prompt.size()) return;
            CHECK(sch.step() != nullptr);
            commit_plain(token);
        }
    }
};

/* ------------------------------------------------------------------ the policy's hooks */
/* Per request, the conditional acceptance at each draft position. Under the additive price below,
 * a row's optimum is the deepest draft whose survival probability still pays 0.05. */
struct Predictions {
    struct Row {
        uint64_t id = 0;
        float    a[RAD_SCHED_MAX_SPEC] = {};
    };
    std::vector<Row> rows;

    void set(uint64_t id, std::initializer_list<float> a) {
        Row* row = nullptr;
        for (Row& r : rows)
            if (r.id == id) row = &r;
        if (!row) { rows.emplace_back(); row = &rows.back(); }
        *row = Row{};
        row->id = id;
        int j = 0;
        for (float v : a) if (j < RAD_SCHED_MAX_SPEC) row->a[j++] = v;
    }
    const float* find(uint64_t id) const {
        for (const Row& r : rows) if (r.id == id) return r.a;
        return nullptr;
    }
};

int predict(void* ctx, const DraftBatchView& view, DraftPredictions* out) {
    const Predictions& t = *static_cast<Predictions*>(ctx);
    out->n_seq = view.n_seq;
    out->version = view.policy_version;   /* this view's, not a later one */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        out->req_id[i] = view.seq[i].req_id;
        out->seq[i] = DraftPrediction{};
        const float* a = t.find(view.seq[i].req_id);
        if (!a) continue;
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) out->seq[i].a[j] = a[j];
    }
    return RAD_OK;
}

/* Additive in the rows this cycle runs: 1.0 ms of verify plus 0.05 ms per draft round. Batching is
 * already in the constant, so a draft pays exactly when its survival probability exceeds 0.05.
 * The draft side is what the candidate chooses; the verify side is not (it is the frozen window
 * every candidate shares), so the draft rounds must carry the difference.
 * A shape it cannot price is returned unpriced -- never a zero posing as a measurement. */
int price(void*, const DraftVerifyShape& s, DraftCostEstimate* out) {
    *out = DraftCostEstimate{};
    if (s.n_rows < 1) return RAD_OK;
    int32_t drafts = 0;
    for (int32_t j = 1; j <= s.draft_k; ++j) drafts += s.draft_rows[j];
    if (drafts < 0) return RAD_OK;
    out->ms       = 1.0 + 0.05 * (double)(s.verify_tokens + drafts);
    out->coverage = 4;
    out->source   = DraftCostSource::Measured;
    out->valid    = true;
    return RAD_OK;
}

/* Turn the cost policy on through the seam the engine will use: the scheduler passes its own
 * params, the test only adds what Config cannot express yet. A test that needs depths the default
 * tiers cannot name passes its own. */
void use_cost_policy(Fixture& f, Predictions& pred, const int32_t* tiers = nullptr,
                     int n_tiers = 0) {
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 6;
    dp.policy.n_tiers  = 3;
    if (tiers) {
        for (int i = 0; i < n_tiers; ++i) dp.policy.tiers[i] = tiers[i];
        dp.policy.n_tiers = n_tiers;
    }
    dp.policy.hold     = 0;      /* a single-cycle test must not wait 8 cycles for a move */
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = DraftCostModel{ nullptr, price };
    CHECK_OK(f.sch.draft().configure(dp));
}

const StepEntry* entry_of(Scheduler& s, uint64_t id) {
    for (const StepEntry& e : s.plan().e)
        if (e.seq == id) return &e;
    return nullptr;
}

/* The plan's K for a request, or -1 when the plan does not name it. */
int planned(Scheduler& s, uint64_t id) {
    const DraftPlan* p = s.draft().plan();
    if (!p) return -1;
    for (int32_t i = 0; i < p->n_seq; ++i)
        if (p->req_id[i] == id) return p->k[i];
    return -1;
}

}  /* namespace */

/* ================================================================== the frozen plan, row by row */

/* A cycle's plan reaches the step as one per-row cap. The three rows are chosen so the only
 * optimum under the additive price is 0, 2 and 6: the first row's drafts survive 1% of the time
 * (below the 5% a draft costs), the second's two survive and its third does not, the third row's
 * six do. Nothing else in the scheduler may flatten them to one window.
 *
 * The plan caps a proposal; it does not write one. The first step opens the cycle, the engine
 * publishes the head's drafts under it, and the second step verifies them -- one cycle later, as
 * on the real loop. */
TEST(a_cycle_plan_of_zero_two_six_sets_each_rows_verify_depth) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t a = f.submit(160), b = f.submit(170), c = f.submit(180);
    pred.set(a, { 0.01f });
    pred.set(b, { 0.9f, 0.9f, 0.01f });
    pred.set(c, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);

    f.prefill(a);
    f.prefill(b);
    f.prefill(c);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, a), 0);
    CHECK_EQ(planned(f.sch, b), 2);
    CHECK_EQ(planned(f.sch, c), 6);
    f.commit_plain(11);

    /* The head's proposals, published under this cycle's plan. A K=0 row gets none. */
    int32_t two[2] = { 31, 32 };
    int32_t six[6] = { 41, 42, 43, 44, 45, 46 };
    CHECK_OK(f.sch.set_draft(b, two, 2));
    CHECK_OK(f.sch.set_draft(c, six, 6));

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, a), 0);
    CHECK_EQ(planned(f.sch, b), 2);
    CHECK_EQ(planned(f.sch, c), 6);

    const StepEntry* ea = entry_of(f.sch, a);
    const StepEntry* eb = entry_of(f.sch, b);
    const StepEntry* ec = entry_of(f.sch, c);
    REQUIRE(ea && eb && ec);
    CHECK_EQ(ea->n_spec, 0);
    CHECK_EQ(ea->n_tokens, 1);
    CHECK_EQ(eb->n_spec, 2);
    CHECK_EQ(eb->n_tokens, 3);
    CHECK_EQ(ec->n_spec, 6);
    CHECK_EQ(ec->n_tokens, 7);
    /* The verify half of the cycle is the deepest window, as the batch carries it. */
    CHECK_EQ(f.sch.plan().n_spec, 6);
    CHECK_EQ(f.sch.plan().n_tok, 11);
}

/* A planned K=0 is a decision, not a vacancy: prompt lookup must not fill the row. Its neighbour,
 * which the plan lets draft, keeps the fallback. Both prompts repeat, so only the gate
 * distinguishes them. */
TEST(a_planned_zero_is_not_filled_by_prompt_lookup) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t off = f.submit_pattern();
    const uint64_t on  = f.submit_pattern();
    pred.set(off, { 0.01f });
    pred.set(on,  { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);

    f.prefill(off, /*token=*/1);      /* the pattern goes on, so the lookup has a match */
    f.prefill(on,  1);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, off), 0);
    const StepEntry* eo = entry_of(f.sch, off);
    const StepEntry* en = entry_of(f.sch, on);
    REQUIRE(eo && en);
    CHECK_EQ(eo->n_spec, 0);
    CHECK_EQ(eo->n_tokens, 1);
    CHECK(!eo->ngram);                 /* the gate: no fallback on a K0 row */
    CHECK(en->n_spec > 0);             /* the plan leaves this row drafting ... */
    CHECK(en->ngram);                  /* ... and prompt lookup is what filled it */
}

/* The fallback is held to the row's own planned depth, not to the cycle's deepest one. A plan of
 * K=2 beside a row at K=6 must not verify the four drafts the pattern offers. */
TEST(prompt_lookup_is_held_to_the_rows_planned_depth) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t shallow = f.submit_pattern();
    const uint64_t deep    = f.submit_pattern();
    pred.set(shallow, { 0.9f, 0.9f, 0.01f });                          /* optimum 2 */
    pred.set(deep,    { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f }); /* optimum 6 */
    use_cost_policy(f, pred);

    f.prefill(shallow, 1);
    f.prefill(deep, 1);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, shallow), 2);
    const StepEntry* es = entry_of(f.sch, shallow);
    REQUIRE(es != nullptr);
    CHECK(es->ngram);
    CHECK_EQ(es->n_spec, 2);           /* four were offered; the plan allows two */
    CHECK_EQ(es->n_tokens, 3);
}

/* ================================================================== history across a K0 cycle */

/* 6 -> 0 -> 6 on one sequence with a serial head. The K0 cycle drafts nothing but still owes the
 * head its history for the positions it verified, so the return to 6 starts from the watermark
 * that cycle moved and not from a gap.
 *
 * Four cycles: the plan is chosen when a step opens, so what a cycle drafts is published under it
 * and verified one cycle later. C is the K=0 cycle; D is opened at K=6 while nothing has been
 * published yet, and its one-row history starting exactly at C's committed end is what proves the
 * watermark moved; E is the first cycle that verifies six again. */
TEST(a_row_planned_off_still_owes_its_history_and_resumes_at_six) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);

    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };
    int rounds = 6;
    Scheduler::SerialRow rows[6];

    /* Cycle A: the plan opens at six, the engine publishes six. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));

    /* Cycle B: verified at six. The history pass owes all 1 + 6 positions and carries round 1. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 6);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 1);
    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK_EQ(hb->n_tok, 7);
    CHECK_EQ(hb->n_draft_out, 1);                /* round 1 rides on the history pass */
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    f.sch.serial_end(6, true);

    const int64_t after6 = f.sch.find(id)->n_computed;
    CHECK_EQ(after6, 161 + 7);

    /* Cycle C: the plan turns the row off. Its history pass still runs for the one verified
     * position, starting exactly where B's commit left the sequence. */
    pred.set(id, { 0.01f });
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 0);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 0);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);   /* nobody drafts */
    const RadBatch* h0 = f.sch.serial_batch(0, 2);
    REQUIRE(h0 != nullptr);
    CHECK_EQ(h0->n_tok, 1);                     /* the history it still owes */
    CHECK_EQ(h0->n_draft_out, 0);
    CHECK_EQ(h0->ctx_lens[0], (int32_t)after6); /* no redo of what B already wrote */
    f.commit_plain(12);
    f.sch.serial_end(0, true);

    const int64_t after0 = f.sch.find(id)->n_computed;
    CHECK_EQ(after0, after6 + 1);

    /* Cycle D: six again. The row may draft once more, but nothing has been published under the
     * new plan, so this step still verifies one token; the head's history pass starts at the
     * watermark the K0 cycle moved, not at B's -- no gap, no redo. */
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 0);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 1);   /* the plan restores drafting ... */
    CHECK_EQ(rounds, 6);                                 /* ... at the full depth */
    const RadBatch* hd = f.sch.serial_batch(0, 3);
    REQUIRE(hd != nullptr);
    CHECK_EQ(hd->n_tok, 1);
    CHECK_EQ(hd->n_draft_out, 1);
    CHECK_EQ(hd->ctx_lens[0], (int32_t)after0);
    f.commit_plain(13);
    f.sch.serial_end(6, true);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));

    /* Cycle E: six verified again, history from the committed end with no gap. */
    const int64_t before_e = f.sch.find(id)->n_computed;
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 6);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 1);
    CHECK_EQ(rounds, 6);
    const RadBatch* h6 = f.sch.serial_batch(0, 4);
    REQUIRE(h6 != nullptr);
    CHECK_EQ(h6->n_tok, 7);
    CHECK_EQ(h6->ctx_lens[0], (int32_t)before_e);
}

/* ================================================================== the history watermark */
/* serial_end() moves a sequence's watermark only for a pass the driver reports as run. A drafting
 * row needs a draft round on top of the round-0 history pass; a history-only row owes nothing but
 * that pass. Built is not run: the counter-probes below build the pass on the scheduler's side and
 * then report it as not executed (the engine rejected the cycle, or its host pass failed).
 * Each call gets a freshly opened snapshot: serial_end() ends the one it reads, so a second call
 * would check an empty one. */
TEST(a_draft_rows_watermark_needs_a_round_and_a_history_pass) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    use_cost_policy(f, pred);
    f.prefill(id);

    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };
    int rounds = 6;
    Scheduler::SerialRow rows[6];

    /* Cycle A: six planned, nothing published yet, so one token is verified. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));

    /* Cycle B: six verified. A fresh snapshot, its history pass built, four rounds ran. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 6);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 1);
    REQUIRE(f.sch.serial_batch(0, 2) != nullptr);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    const int64_t after_rounds = f.sch.find(id)->n_computed;
    f.sch.serial_end(4, true);
    CHECK_EQ(f.sch.draft_watermark(id), after_rounds);

    /* Cycle C: another fresh snapshot with its history pass built, but zero rounds. The verified
     * positions were never written, so the watermark stays where cycle B left it. */
    REQUIRE(f.sch.step() != nullptr);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 1);
    REQUIRE(f.sch.serial_batch(0, 3) != nullptr);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    const int64_t after_abort = f.sch.find(id)->n_computed;
    CHECK(after_abort > after_rounds);
    f.sch.serial_end(0, true);
    CHECK_EQ(f.sch.draft_watermark(id), after_rounds);

    /* Cycle D: the build came back, the pass did not run. The driver reports that, and the
     * watermark stays put although this scheduler built a batch and four rounds were asked for. */
    REQUIRE(f.sch.step() != nullptr);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 1);
    REQUIRE(f.sch.serial_batch(0, 4) != nullptr);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    CHECK(f.sch.find(id)->n_computed > after_abort);
    f.sch.serial_end(4, false);
    CHECK_EQ(f.sch.draft_watermark(id), after_rounds);
}

TEST(a_history_only_rows_watermark_needs_its_pass) {
    Fixture f;
    f.build();
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    /* Longer than one chunk: the first step is mid-prompt, which owes history and no token. */
    const uint64_t id = f.submit(700);
    REQUIRE(f.sch.step() != nullptr);
    const StepEntry* e = entry_of(f.sch, id);
    REQUIRE(e != nullptr);
    CHECK(e->is_prefill);
    CHECK(!e->produces_token);
    const int64_t hist_end = e->ctx_len + e->n_tokens;

    int rounds = 6;
    Scheduler::SerialRow rows[6];

    /* A fresh snapshot whose history pass is built and runs, with no draft round at all. */
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);   /* nobody drafts */
    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK_EQ(hb->n_draft_out, 0);
    f.commit_plain(11);
    f.sch.serial_end(0, true);
    CHECK_EQ(f.sch.draft_watermark(id), hist_end);

    /* The next chunk, same build, but the driver reports the pass as not executed: the step
     * commits and nothing wrote the head's history, so the watermark must not move. */
    REQUIRE(f.sch.step() != nullptr);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);
    REQUIRE(f.sch.serial_batch(0, 2) != nullptr);   /* built ... */
    f.commit_plain(11);
    CHECK(f.sch.find(id)->n_computed > hist_end);
    f.sch.serial_end(0, false);                     /* ... and did not run */
    CHECK_EQ(f.sch.draft_watermark(id), hist_end);

    /* The next chunk, on a fresh snapshot whose history pass is never built: same outcome, and
     * the default argument states it without the driver having to say so. */
    REQUIRE(f.sch.step() != nullptr);
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);
    f.commit_plain(11);
    CHECK(f.sch.find(id)->n_computed > hist_end);
    f.sch.serial_end(0);
    CHECK_EQ(f.sch.draft_watermark(id), hist_end);
}

/* ================================================================== the rounds thin by depth */

/* M2: a round runs only the rows that still have a draft at its depth, and the snapshot is
 * ordered deepest first, so every round's rows are a prefix of the one before. The plan here is
 * issued in an order the depths contradict (2, 1, 3 and one K0 row), so plan order alone would
 * put the shallowest row in every round.
 *
 * Round 0 is the history pass and carries round 1 for the drafting rows (RadBatch::draft_out_ids);
 * round 1 has no pass of its own. Rounds 2 and 3 are the driver's own count, not `round == K`. */
TEST(a_round_runs_only_the_rows_that_still_have_a_draft) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t a = f.submit(160), b = f.submit(170), c = f.submit(180), d = f.submit(190);
    pred.set(a, { 0.9f, 0.9f, 0.01f });                    /* two drafts pay */
    pred.set(b, { 0.9f, 0.01f });                          /* one */
    pred.set(c, { 0.9f, 0.9f, 0.9f, 0.01f });              /* three */
    pred.set(d, { 0.01f });                               /* none: history only */
    const int32_t tiers[4] = { 0, 1, 2, 3 };
    use_cost_policy(f, pred, tiers, 4);

    f.prefill(a);
    f.prefill(b);
    f.prefill(c);
    f.prefill(d);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, a), 2);
    CHECK_EQ(planned(f.sch, b), 1);
    CHECK_EQ(planned(f.sch, c), 3);
    CHECK_EQ(planned(f.sch, d), 0);
    const StepEntry* ea = entry_of(f.sch, a);
    const StepEntry* eb = entry_of(f.sch, b);
    const StepEntry* ec = entry_of(f.sch, c);
    const StepEntry* ed = entry_of(f.sch, d);
    REQUIRE(ea && eb && ec && ed);

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    const int n = f.sch.serial_begin(&rounds, rows, 6);
    CHECK_EQ(rounds, 3);                  /* the deepest row's depth, not the ceiling */
    REQUIRE(n == 3);                      /* the K0 row drafts nothing */
    CHECK_EQ(rows[0].seq, c);
    CHECK_EQ(rows[1].seq, a);
    CHECK_EQ(rows[2].seq, b);
    for (int i = 0; i < n; ++i) {
        const StepEntry* e = entry_of(f.sch, rows[i].seq);
        REQUIRE(e != nullptr);
        CHECK_EQ(rows[i].slot, e->slot);
    }

    /* Round 0: the three drafting rows in that order, then the K0 row's history. Each row's
     * history starts at its own committed end, which identifies it. */
    const RadBatch* r0 = f.sch.serial_batch(0, 1);
    REQUIRE(r0 != nullptr);
    CHECK_EQ(r0->n_seq, 4);
    CHECK_EQ(r0->n_draft_out, 3);          /* round 1 for each drafting row */
    CHECK_EQ(r0->ctx_lens[0], ec->ctx_len);
    CHECK_EQ(r0->ctx_lens[1], ea->ctx_len);
    CHECK_EQ(r0->ctx_lens[2], eb->ctx_len);
    CHECK_EQ(r0->ctx_lens[3], ed->ctx_len);
    CHECK(f.sch.serial_batch(1, 2) == nullptr);   /* carried by round 0 */

    /* Round 2: the rows at K >= 2, still deepest first. */
    const RadBatch* r2 = f.sch.serial_batch(2, 3);
    REQUIRE(r2 != nullptr);
    CHECK_EQ(r2->n_seq, 2);
    CHECK_EQ(r2->ctx_lens[0], ec->ctx_len + 1);
    CHECK_EQ(r2->ctx_lens[1], ea->ctx_len + 1);

    /* Round 3: the single K=3 row. */
    const RadBatch* r3 = f.sch.serial_batch(3, 4);
    REQUIRE(r3 != nullptr);
    CHECK_EQ(r3->n_seq, 1);
    CHECK_EQ(r3->ctx_lens[0], ec->ctx_len + 2);

    /* No row has a fourth draft, so round 4 builds nothing. */
    CHECK(f.sch.serial_batch(4, 5) == nullptr);

    f.commit_plain(11);
    f.sch.serial_end(2, true);           /* rounds 2 and 3 ran, over their history pass */
    CHECK_EQ(f.sch.draft_watermark(c), f.sch.find(c)->n_computed);
    CHECK_EQ(f.sch.draft_watermark(d), f.sch.find(d)->n_computed);   /* its history pass ran */
}

/* ================================================================== the real limits win */

/* A plan is not the only cap. At the context end the scheduler keeps the sequence room for the
 * one token it must commit, whatever the cycle planned. */
TEST(a_planned_depth_is_clamped_by_the_contexts_remaining_room) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/512);
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t roomy = f.submit(8);
    pred.set(roomy, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(roomy);

    /* 510 prompt tokens plus the token the last prefill chunk emits fill 511 of a 512-token
     * context: one position left, and the verified token takes it. The row's acceptance is the
     * same as the neighbour's, so only the room can make its K zero -- and the neighbour in the
     * same cycle keeps six, which is what "per row" means. */
    const uint64_t tight = f.submit(510);
    pred.set(tight, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    f.prefill(tight);
    REQUIRE(f.sch.find(tight)->n_computed == 510);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, tight), 0);
    CHECK_EQ(planned(f.sch, roomy), 6);
    const StepEntry* e = entry_of(f.sch, tight);
    REQUIRE(e != nullptr);
    CHECK_EQ(e->n_spec, 0);
    CHECK_EQ(e->n_tokens, 1);
}

/* A row the lookahead no longer fits loses its draft here, after the rounds were chosen, and keeps
 * its history. The rounds must follow it: with the deepest row gone, a round past the shallower
 * row's depth has no drafting row and no history row, so serial_batch() would build nothing while
 * the driver still counted a pass.
 *
 * The deep row sits one context end away -- its six published drafts plus six more rounds would
 * cross max_ctx -- and the shallower row is nowhere near it, so only the deep one falls. */
TEST(the_rounds_stop_at_the_deepest_row_the_context_still_has_room_for) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/512);
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t deep = f.submit(501);
    pred.set(deep, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });   /* six drafts pay */
    use_cost_policy(f, pred);
    f.prefill(deep);

    const uint64_t shallow = f.submit(8);
    pred.set(shallow, { 0.9f, 0.9f, 0.01f });                         /* two pay */
    f.prefill(shallow);

    /* Cycle A opens both rows at their own depth; nothing is published under it yet. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, deep), 6);
    CHECK_EQ(planned(f.sch, shallow), 2);
    f.commit_plain(11);
    int32_t six[6] = { 41, 42, 43, 44, 45, 46 };
    int32_t two[2] = { 51, 52 };
    CHECK_OK(f.sch.set_draft(deep, six, 6));
    CHECK_OK(f.sch.set_draft(shallow, two, 2));

    /* Cycle B verifies both. Six more rounds at the deep row's head would pass the context end;
     * the shallower row's would not. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, deep), 6);
    CHECK_EQ(planned(f.sch, shallow), 2);
    const StepEntry* ed = entry_of(f.sch, deep);
    const StepEntry* es = entry_of(f.sch, shallow);
    REQUIRE(ed && es);
    CHECK_EQ(ed->n_spec, 6);
    CHECK_EQ(es->n_spec, 2);
    CHECK(ed->ctx_len + ed->n_spec + 6 > 512);     /* the premise: six rounds no longer fit */
    CHECK(es->ctx_len + es->n_spec + 6 <= 512);

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    const int n = f.sch.serial_begin(&rounds, rows, 6);
    CHECK_EQ(n, 1);                      /* the deep row owes its history and drafts nothing */
    CHECK_EQ(rows[0].seq, shallow);
    CHECK_EQ(rounds, 2);                 /* the deepest survivor, not the cycle's six */

    /* Every pass this bound leaves is asked for and builds; the round past it is never asked. */
    const RadBatch* h = f.sch.serial_batch(0, 1);
    REQUIRE(h != nullptr);
    CHECK_EQ(h->n_seq, 2);
    CHECK_EQ(h->n_draft_out, 1);
    const RadBatch* r2 = f.sch.serial_batch(2, 2);
    REQUIRE(r2 != nullptr);
    CHECK_EQ(r2->n_seq, 1);
    CHECK(f.sch.serial_batch(3, 3) == nullptr);   /* past the prefix; `rounds` stops at 2 */
}

/* The same rejection with no shallower row: every draft row falls away and the history pass is all
 * that is left. The bound then asks for exactly that one pass, and the history-only watermark rule
 * still holds for it. */
TEST(all_draft_rows_can_fall_away_and_the_history_pass_is_still_asked_for) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/512);
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(501);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    f.commit_plain(11);
    int32_t drafts[6] = { 41, 42, 43, 44, 45, 46 };
    CHECK_OK(f.sch.set_draft(id, drafts, 6));

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    const StepEntry* e = entry_of(f.sch, id);
    REQUIRE(e != nullptr);
    CHECK_EQ(e->n_spec, 6);
    CHECK(e->ctx_len + e->n_spec + 6 > 512);

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);   /* nobody drafts */
    CHECK_EQ(rounds, 1);                                 /* one pass: the history */

    const RadBatch* h = f.sch.serial_batch(0, 2);
    REQUIRE(h != nullptr);
    CHECK_EQ(h->n_seq, 1);
    CHECK_EQ(h->n_tok, 7);                /* the 1 + 6 positions it still owes */
    CHECK_EQ(h->n_draft_out, 0);
    CHECK(f.sch.serial_batch(2, 3) == nullptr);   /* nothing past the history; rounds stops at 1 */

    /* The pass ran: the watermark moves to what it wrote, and no draft round was needed for it. */
    CHECK_OK(f.commit(11, 0, [](const StepEntry& x) { return x.n_spec; }));
    f.sch.serial_end(0, true);
    CHECK_EQ(f.sch.draft_watermark(id), f.sch.find(id)->n_computed);
}

/* ================================================================== identity across admission */

/* A cancelled request's depth must not reach the next request on its slot. The slot is recycled
 * with a new generation, and the next cycle names the new request only. */
TEST(a_cancelled_rows_depth_never_reaches_the_next_request_on_its_slot) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t gone = f.submit(160);
    const uint64_t stay = f.submit(170);
    pred.set(gone, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });  /* 6 */
    pred.set(stay, { 0.01f });                                      /* 0 */
    use_cost_policy(f, pred);

    f.prefill(gone);
    f.prefill(stay);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, gone), 6);
    CHECK_EQ(planned(f.sch, stay), 0);

    /* The slot's generation as the plan recorded it, before the plan is replaced. */
    const DraftPlan* p0 = f.sch.draft().plan();
    REQUIRE(p0 != nullptr);
    int32_t slot = -1;
    uint32_t gen = 0;
    for (int32_t i = 0; i < p0->n_seq; ++i)
        if (p0->req_id[i] == gone) { slot = p0->slot[i]; gen = p0->slot_gen[i]; }
    REQUIRE(slot >= 0);

    f.commit_plain(11);

    /* Cancel and reap, as the server would, then admit a request that takes the slot. */
    CHECK_OK(f.sch.cancel(gone));
    CHECK_OK(f.sch.reap(gone));
    const uint64_t next = f.submit(180);
    pred.set(next, { 0.01f });                  /* its own decision: no drafts */
    f.prefill(next);                           /* decode rows are what a cycle prices */

    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* p1 = f.sch.draft().plan();
    REQUIRE(p1 != nullptr);
    CHECK_EQ(planned(f.sch, gone), -1);         /* the plan no longer names it */
    CHECK_EQ(planned(f.sch, next), 0);
    int32_t nslot = -1;
    uint32_t ngen = 0;
    for (int32_t i = 0; i < p1->n_seq; ++i)
        if (p1->req_id[i] == next) { nslot = p1->slot[i]; ngen = p1->slot_gen[i]; }
    CHECK_EQ(nslot, slot);                      /* the slot was recycled ... */
    CHECK(ngen != gen);                         /* ... and is another request */
    CHECK_EQ(entry_of(f.sch, next)->n_spec, 0); /* not the cancelled row's six */
}

/* ================================================================== a step issued ahead */

/* The plan is one object per cycle and the next cycle overwrites it. A step already issued ahead
 * keeps the shape it was built with; only the cycle after it reads the new plan. */
TEST(a_step_issued_ahead_keeps_its_frozen_depth_when_the_next_cycle_changes_the_plan) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    /* Cycle A opens the plan at six; nothing has been published under it yet. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 0);
    f.commit_plain(11);

    /* Cycle B verifies the six published under A, and while it is open the engine publishes the
     * next proposal and issues the step after it -- still under B's plan. */
    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 6);
    f.sch.expect_draft(id, 6);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    const RadBatch* ahead = f.sch.step_next();
    REQUIRE(ahead != nullptr);
    CHECK_EQ(ahead->n_spec, 6);

    /* The cycle that runs during the commit decides differently. */
    pred.set(id, { 0.01f });
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    f.sch.promote_next();

    /* The promoted step is the one built at six, and the new plan already says zero. */
    REQUIRE(f.sch.plan().e.size() == 1);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 6);
    CHECK_EQ(planned(f.sch, id), 0);

    f.commit_plain(12);
    /* The step after it is planned from the new cycle. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 0);
}

/* An admission is a decision about the pool and the queue, so no step is issued ahead while one
 * is pending: the request would be sized against a running set it is not in. */
TEST(an_admission_in_flight_keeps_the_next_step_from_being_issued_ahead) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    REQUIRE(f.sch.step() != nullptr);
    /* A request arrives while this step is open. */
    const uint64_t late = f.submit(180);
    f.sch.find(late)->speculation_fixed = true;
    CHECK(f.sch.step_next() == nullptr);

    f.commit_plain(11);
    /* It is admitted by the next full step, which is the one that may decide its place. */
    REQUIRE(f.sch.step() != nullptr);
    const StepEntry* el = entry_of(f.sch, late);
    REQUIRE(el != nullptr);
    CHECK(el->is_prefill);
    CHECK_EQ(el->n_spec, 0);
}
/* The priced shape is written down, not left at its defaults. A K0 plan with history is "policy
 * K0 with history", so every row of a cycle with the head loaded carries the verified token's
 * history row (frozen K 0 here), the prefill sharing the step is named, the loaded drafter is
 * Serial, and the replay dimension is flagged Unknown because this cycle did not observe one. The
 * plan says the same per row. */
struct ShapeLog {
    DraftVerifyShape last;
    int calls = 0;
};

/* A's real table behind the seam, recording the shape it was asked to price. */
struct RealPrice {
    DraftCostTable*  table = nullptr;
    DraftVerifyShape* last = nullptr;
};

int real_price(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
    RealPrice* r = static_cast<RealPrice*>(ctx);
    if (r->last) *r->last = s;
    return r->table->estimate(s, out);
}

int price_and_record(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
    ShapeLog* log = static_cast<ShapeLog*>(ctx);
    log->last = s;
    ++log->calls;
    *out = DraftCostEstimate{};
    if (s.n_rows < 1) return RAD_OK;
    out->ms = 1.0 + 0.05 * (double)(s.verify_tokens - s.n_rows);
    out->coverage = 4;
    out->source = DraftCostSource::Measured;
    out->valid = true;
    return RAD_OK;
}

TEST(the_priced_shape_carries_history_prefill_and_the_loaded_drafter) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    ShapeLog log;
    const uint64_t id = f.submit(8);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 6;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = DraftCostModel{ &log, price_and_record };
    CHECK_OK(f.sch.draft().configure(dp));

    f.prefill(id);
    const uint64_t late = f.submit(600);
    f.sch.find(id)->sp.temp = 0.0f;   /* greedy: the profile below keys on that */
    REQUIRE(f.sch.step() != nullptr);
    /* Cycle 2: one decoding row, nothing verifies yet, the long prompt is still queued. */
    CHECK(log.calls > 0);
    CHECK_EQ(log.last.n_rows, 1);
    CHECK_EQ(log.last.history_tokens, 1);   /* the verified token's own history row */
    CHECK_EQ(log.last.prefill_tokens, 0);   /* nothing running prefills yet */
    CHECK(log.last.drafter == DraftDrafter::Serial);
    /* No replay state was observed this cycle, so the shape marks the dimension Unknown instead of
     * claiming `None` and letting a model key it as measured. */
    CHECK_EQ(log.last.unknown, (uint32_t)kDraftShapeReplayUnknown);

    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    CHECK_EQ(p->history_tokens[0], 1);
    CHECK_EQ(p->ctx_class[0], draft_ctx_class(8));
    CHECK(p->fill[0] == (p->k[0] > 0 ? DraftDrafter::Serial : DraftDrafter::None));

    /* Cycle 3: the long prompt is running mid-prefill, so the step the plan pairs with shares
     * prefill work, and the row now verifies the previous plan's window (its history rows). */
    const int prev_k = p->k[0];
    f.commit_plain(11);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(log.last.prefill_tokens, 600 - 256);   /* one chunk admitted, the rest pending */
    CHECK_EQ(log.last.history_tokens, 1 + prev_k);
    const StepEntry* el = entry_of(f.sch, late);
    REQUIRE(el != nullptr);
    CHECK(el->is_prefill);
}

/* A plan is written once and read for the cycle it names; the next cycle writes the other set, so
 * a pointer handed out under issue_ahead() still says what it priced after the next open. */
TEST(an_older_plan_keeps_its_rows_when_the_next_cycle_opens) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t a = f.submit(160);
    const uint64_t b = f.submit(180);
    pred.set(a, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    pred.set(b, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    use_cost_policy(f, pred);
    f.prefill(a);
    f.prefill(b);
    f.sch.find(a)->speculation_fixed = true;
    f.sch.find(b)->speculation_fixed = true;

    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* first = f.sch.draft().plan();
    REQUIRE(first != nullptr);
    const uint64_t first_cycle = first->cycle_id;
    const int32_t  first_k     = first->k[0];
    const int32_t  first_hist  = first->history_tokens[0];

    f.commit_plain(11);
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* next = f.sch.draft().plan();
    REQUIRE(next != nullptr);
    CHECK(next != first);
    CHECK_EQ(first->cycle_id, first_cycle);
    CHECK_EQ(first->k[0], first_k);
    CHECK_EQ(first->history_tokens[0], first_hist);
}

/* The A/C seam against the real model: a version-2 profile measured for this deployment, read by
 * A's table, sees the shape the planner prices. The cycle does not observe its own replay state
 * -- `DraftBatchView` carries none -- so the planner marks that dimension Unknown and the shape is
 * not a measured key. A's table therefore refuses it, and a cycle it cannot price becomes the
 * named baseline: a fallback, never an invented number.
 *
 * The last block shows what the same profile does once that dimension is supplied, which is the
 * piece the interface is missing (see the request in the issue).
 *
 * The buckets are keyed the way this runtime runs: publish-then-verify at pipeline depth 1, so a
 * bucket's verify window is the previous cycle's distribution (nothing yet, on the first cycle)
 * and its draft distribution is the cycle's own. */
TEST(the_real_profile_needs_the_replay_dimension_the_planner_cannot_observe) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(8);
    pred.set(id, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

    const int cls = draft_ctx_class(8);
    auto bucket = [&](int k, double ms) {
        std::string draft = "0";
        for (int j = 1; j <= k; ++j) draft += ",1";
        return "{\"ctx_class\": " + std::to_string(cls) +
               ", \"shape\": {\"n_rows\": 1, \"verify_k\": 0, \"verify_tokens\": 1"
               ", \"history_tokens\": 1, \"prefill_tokens\": 0, \"pipeline_depth\": 1"
               ", \"rows\": [0], \"draft_k\": " + std::to_string(k) +
               ", \"draft_rows\": [" + draft + "]"
               ", \"origins\": [\"head\"], \"sampling\": \"greedy\", \"replay\": \"none\""
               ", \"drafter\": \"serial\"},"
               " \"extent\": {\"rounds\": " + std::to_string(k) +
               ", \"draft_rows_total\": " + std::to_string(k) +
               ", \"draft_round_rows\": [" + draft + "]},"
               " \"time_ms\": {\"mean\": " + std::to_string(ms) + ", \"unc\": 0.01, \"n\": 9}}";
    };
    /* 1 token / 2.0 ms, 3 / 1.3, 7 / 5.0: K=2 pays, K=6 does not. */
    const std::string profile =
        "{\"format\": \"rad-draft-profile/2\", \"fingerprint\": {\"model\": \"m\", \"quant\": \"q\", "
        "\"tp\": 1, \"tp_wire\": \"w\", \"gpus\": \"g\", \"kernels\": \"k\", \"kv_format\": \"bf16\", "
        "\"placement\": \"vram\"}, \"execution\": \"rectangular\", \"buckets\": [" +
        bucket(0, 2.0) + "," + bucket(2, 1.3) + "," + bucket(6, 5.0) + "]}";

    DraftCostProfile parsed;
    std::string err;
    CHECK_OK(draft_cost_profile_parse(profile, &parsed, &err));
    DraftCostRuntime rt;
    rt.fingerprint = parsed.fingerprint;
    rt.execution   = DraftCostExecution::Rectangular;
    DraftCostTable table;
    CHECK_OK(table.configure(parsed, rt));
    CHECK(table.active());

    /* A's table behind a recorder, so the shape the planner prices is visible. */
    DraftVerifyShape last_shape;
    RealPrice real{ &table, &last_shape };

    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 6;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = DraftCostModel{ &real, real_price };
    CHECK_OK(f.sch.draft().configure(dp));

    f.prefill(id);
    f.sch.find(id)->sp.temp = 0.0f;   /* greedy, as the profile's buckets are keyed */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    CHECK_EQ(last_shape.n_rows, 1);
    CHECK_EQ(last_shape.history_tokens, 1);
    CHECK_EQ(last_shape.pipeline_depth, 1);
    CHECK_EQ(last_shape.verify_k, 0);      /* nothing was published before this cycle */
    CHECK_EQ(last_shape.verify_tokens, 1);
    /* The cycle saw no replay state, so the shape says so rather than asserting `None`. */
    CHECK_EQ(last_shape.unknown, (uint32_t)kDraftShapeReplayUnknown);
    CHECK(!draft_shape_priceable(last_shape));
    DraftCostEstimate refused;
    CHECK_EQ(table.estimate(last_shape, &refused), RAD_E_INVAL);
    /* Unpriceable is a fallback with a name, not a failed step and not a made-up price. */
    CHECK(p->source == DraftPlanSource::Fallback);
    CHECK(p->fallback == DraftFallback::UnknownShape);
    CHECK_EQ(p->k[0], dp.policy.fallback_k);
    CHECK_EQ(p->pipeline_depth, 1);
    CHECK_EQ(draft_plan_valid(*p), RAD_OK);

    /* The rest of the seam already fits: this profile prices the same cycle as measured once the
     * replay dimension is known. Supplying it is the missing interface piece. */
    DraftVerifyShape shape;
    shape.n_rows = 1;
    shape.verify_k = 0;
    shape.verify_tokens = 1;
    shape.history_tokens = 1;
    shape.ctx_class = cls;
    shape.pipeline_depth = 1;
    shape.draft_k = 2;
    shape.draft_rows[1] = 1;
    shape.draft_rows[2] = 1;
    shape.drafter = DraftDrafter::Serial;
    shape.replay = DraftReplay::None;   /* what the driver would report for a live pass */
    CHECK(draft_shape_priceable(shape));
    CHECK_EQ(shape.verify_k, p->verify_k);
    CHECK_EQ(shape.pipeline_depth, p->pipeline_depth);
    DraftCostEstimate est;
    CHECK_OK(table.estimate(shape, &est));
    CHECK(est.valid);
    CHECK(est.source == DraftCostSource::Measured);
}
RAD_TEST_MAIN()
