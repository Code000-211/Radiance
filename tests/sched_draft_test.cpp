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
 * these tests need no file-based config: `--draft-policy cost` reaches it through Config, but a
 * profile still has to be configured for the runtime or Scheduler::init refuses the policy by
 * name (the baseline lock below). The fixtures install one explicitly (activate_costs).
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
#include "sched_program.h"

using namespace rad;
using namespace sched_fake;

namespace {

/* ------------------------------------------------------------------ a declared program */
/* The program is sched_program.h's: engine_history_test.cpp hands the same shape to a real
 * Engine's scheduler, and one declaration serves both. */
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
        build_program(prog, cfg, kv, pc, owned, blocks, states, max_tok, max_ctx);
    }

    /* `pass_replay` is what a real runtime evidences about its tape (Engine passes `None` for a
     * backend that cannot record); a fixture that says nothing leaves the cycles unpriced. */
    int start(DraftReplay pass_replay = DraftReplay::Unknown) {
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        return sch.init(prog, cfg, &kv, &pc, rio, nullptr, pass_replay);
    }

    uint64_t next_id = 1;
    uint64_t add_prompt(std::vector<int32_t> tokens, int max_tokens = 64, int priority = 0) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->max_tokens = max_tokens;
        r->priority = priority;
        r->prompt = std::move(tokens);
        const uint64_t id = r->id;
        return sch.add(std::move(r)) < 0 ? 0 : id;
    }
    uint64_t submit(int n_prompt, int max_tokens = 64, int priority = 0) {
        std::vector<int32_t> t((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) t[(size_t)i] = 1000 + i;
        return add_prompt(std::move(t), max_tokens, priority);
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

/* ================================================================== A's production sample */

/* Install a profile for this runtime so the table is active. Empty by default (a configured
 * deployment with no measurements yet), whose samples land in `buckets()`, which is what a host
 * test can read without a profile file. `priced` installs one usable bucket instead, so a Cost
 * deployment is admitted on a measurement rather than on the named fallback, and
 * fallback_admitted() stays false for the test to check. `basis` NoLoan makes the loaded table
 * the immutable reference (nothing a cycle observes is folded into it), Live the learning table. */
void activate_costs(Fixture& f, bool priced = false,
                    DraftCostBasis basis = DraftCostBasis::Live) {
    DraftCostProfile p;
    p.fingerprint.model     = "test";
    p.fingerprint.quant     = "q";
    p.fingerprint.tp_wire   = "w";
    p.fingerprint.gpus      = "g";
    p.fingerprint.kernels   = "k";
    p.fingerprint.kv_format = "bf16";
    p.fingerprint.placement = "all_vram";
    p.fingerprint.tp        = 1;
    p.fingerprint.basis     = basis;
    p.fingerprint.arena     = "none";
    p.fingerprint.live_loan = "flex=0 lend=0";
    if (priced) {
        DraftCostBucket b;
        b.shape.n_rows        = 1;
        b.shape.verify_k      = 1;
        b.shape.verify_tokens = 2;
        b.shape.rows[1]       = 1;
        b.shape.draft_k       = 1;
        b.shape.draft_rows[1] = 1;
        b.shape.drafter       = DraftDrafter::Serial;
        b.rounds              = 1;
        b.draft_round_rows[1] = 1;
        b.draft_rows_total    = 1;
        b.mean_ms             = 12.0;
        b.unc_ms              = 1.0;
        b.n                   = 3;
        p.buckets.assign(1, b);
    }
    DraftCostRuntime rt;
    rt.fingerprint = p.fingerprint;
    CHECK_OK(f.sch.draft_costs().configure(p, rt));
    REQUIRE(f.sch.draft_costs().active());
    /* Matching is not admission: an empty profile has no price for any shape, so the deployment
     * says in so many words that Cost may run on the policy's named baseline (scheduler.cpp). */
    if (!priced) f.sch.draft_costs().admit_fallback();
}

/* The plan's scalars for one cycle, copied: the controller's storage moves with the next open. */
struct PlanCopy {
    int32_t max_k = 0;
    int32_t verify_k = 0;
    int32_t pipeline_depth = 0;
    int32_t n_seq = 0;
    int32_t rows[RAD_SCHED_MAX_SPEC + 1] = {};
    std::vector<int32_t> k;
    std::vector<int32_t> ctx_class;
    std::vector<uint64_t> id;
    int at(uint64_t req) const {
        for (size_t i = 0; i < id.size(); ++i)
            if (id[(size_t)i] == req) return k[i];
        return -1;
    }
    int32_t class_at(uint64_t req) const {
        for (size_t i = 0; i < id.size(); ++i)
            if (id[(size_t)i] == req) return ctx_class[i];
        return -1;
    }
};
PlanCopy copy_plan(Scheduler& s) {
    PlanCopy c;
    const DraftPlan* p = s.draft().plan();
    if (!p) return c;
    c.max_k = p->max_k;
    c.verify_k = p->verify_k;
    c.pipeline_depth = p->pipeline_depth;
    c.n_seq = p->n_seq;
    for (int32_t j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) c.rows[j] = p->rows[j];
    for (int32_t i = 0; i < p->n_seq; ++i) {
        c.k.push_back(p->k[i]);
        c.ctx_class.push_back(p->ctx_class[i]);
        c.id.push_back(p->req_id[i]);
    }
    return c;
}

/* A's table is fed by the cycle the scheduler itself ran, not by a caller: the sample carries the
 * executed distribution (what each row published, not what the plan allowed), the verify window
 * that actually ran, and the step bracket. A cycle state nobody established reports Unknown and is
 * never filed as a measured None. */
TEST(a_cycle_measures_the_executed_distribution_and_refuses_unknown_evidence) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160), b = f.submit(170);
    f.prefill(a);
    f.prefill(b);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };

    /* Cycle 1: nothing can be priced without a profile, so the fallback sets both rows to the
     * configured depth. Its plan is the one the sample is filed under. */
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy p1 = copy_plan(f.sch);
    REQUIRE(p1.n_seq == 2);
    CHECK_EQ(p1.max_k, 6);
    f.commit(11, 0, [](const StepEntry&) { return 0; });
    for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, p1.at(id)));

    /* The engine's evidence: a live pass with no tape recording (None), nothing staged or moved. */
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    f.sch.step_finished(live);

    /* Cycle 2: opening it closes cycle 1, which is now a measurement. */
    REQUIRE(f.sch.step() != nullptr);
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    REQUIRE(bs.size() == 1);
    const DraftCostBucket& b1 = bs[0];
    /* A bucket keeps the attribution flags only, so a clean measurement has none -- not even
     * Valid: `record()` strips it, the price itself is the flag. */
    CHECK_EQ(b1.flags, 0);
    CHECK_EQ(b1.n, 1);
    CHECK(b1.mean_ms > 0.0);
    /* The executed distribution is the plan's: two rows each published their full window. */
    CHECK_EQ(b1.rounds, p1.max_k);
    for (int32_t j = 1; j <= p1.max_k; ++j) {
        CHECK_EQ(b1.draft_round_rows[j], p1.rows[j]);
        CHECK_EQ(b1.shape.draft_rows[j], p1.rows[j]);
    }
    CHECK_EQ(b1.shape.draft_k, p1.max_k);
    CHECK_EQ(b1.draft_rows_total, 2 * p1.max_k);
    /* The verify half: cycle 1's step verified nothing, and n_rows is the plan's row count. */
    CHECK_EQ(b1.shape.n_rows, p1.n_seq);
    CHECK_EQ(b1.shape.verify_k, 0);
    CHECK_EQ(b1.shape.verify_tokens, 2 * (1 + 0));
    CHECK_EQ(b1.shape.history_tokens, b1.shape.verify_tokens);
    CHECK_EQ(b1.shape.pipeline_depth, 1);
    CHECK_EQ(b1.shape.prefill_tokens, 0);
    CHECK(b1.shape.sampling == DraftSampling::Sampled);   /* the fixture's rows sample (temp 1.0) */
    CHECK(b1.shape.replay == DraftReplay::None);
    CHECK(b1.shape.drafter == DraftDrafter::Serial);

    /* Cycle 2 does not establish its replay state (the harness has no tape): the default evidence
     * must not be read as a measured None, so no bucket is filed and the refusal is counted. */
    const size_t buckets_before = bs.size();
    const int64_t ignored_before = f.sch.draft_costs().ignored();
    f.commit(11, 0, [](const StepEntry&) { return 2; });
    for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, 6));
    f.sch.step_finished();                       /* no evidence at all */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_costs().buckets().size(), buckets_before);
    CHECK_EQ(f.sch.draft_costs().ignored(), ignored_before + 1);
    CHECK_EQ(f.sch.draft_cost_refused(), 0);     /* not an error: an unattributed sample */

    /* Cycle 3 with one row leaving early. `a` is cancelled after the plan named it and before the
     * step's commit; `b` still runs its full window, so the global round count reaches max_k --
     * the sample must still be shortened, because the bracket timed one row less than the shape
     * it would be filed under. */
    const PlanCopy p3 = copy_plan(f.sch);
    REQUIRE(p3.n_seq == 2);
    CHECK(f.sch.cancel(a) >= 0);
    f.commit(11, 0, [](const StepEntry&) { return 2; });
    CHECK_OK(f.sch.set_draft(b, six, p3.at(b)));
    f.sch.step_finished(live);

    const int64_t dropped_before = f.sch.draft_cycles_dropped();
    REQUIRE(f.sch.step() != nullptr);            /* cycle 4 opens: cycle 3 closes */
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped_before + 1);   /* B never got a hole */
    REQUIRE(f.sch.draft_costs().buckets().size() == buckets_before + 1);
    const DraftCostBucket& b3 = f.sch.draft_costs().buckets()[buckets_before];
    CHECK(b3.flags & kDraftSampleShortened);
    CHECK_EQ(b3.rounds, p3.max_k);               /* the global count alone says "complete" */
    CHECK_EQ(b3.draft_round_rows[1], 1);         /* one row per round actually ran */
    CHECK_EQ(b3.shape.n_rows, 2);                /* the plan's count, unchanged */
    /* A shortened bucket is a diagnostic, never a price. */
    const DraftVerifyShape query = b3.shape;
    DraftCostEstimate est;
    CHECK_OK(f.sch.draft_costs().estimate(query, &est));
    CHECK(!(est.valid && est.source == DraftCostSource::Measured));
}

/* A cycle whose step was never bracketed has no time and is refused, not filed at zero cost. */
TEST(an_untimed_cycle_is_refused_by_a_rather_than_priced_at_zero) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160);
    f.prefill(a);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    REQUIRE(f.sch.step() != nullptr);
    const int32_t k = copy_plan(f.sch).at(a);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, six, k));
    /* Deliberately no step_finished(): the cycle has no bracket. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK(f.sch.draft_costs().buckets().empty());
    CHECK_EQ(f.sch.draft_cost_refused(), 1);
    CHECK(f.sch.draft_cycles_done() >= 1);       /* B's path is unaffected by A's refusal */
}

/* A cycle is filed under the context class its plan was priced at: the deepest row's, as
 * DraftPlanner::shape_of computes it. A class-0 constant would file a long-context measurement in
 * a bucket no class-1 estimate can ever reach, and the planner's own query would miss it. */
TEST(a_cycles_context_class_is_the_deepest_frozen_rows_like_the_planners) {
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    /* One row past the 4096-token boundary (class 1), one far below it (class 0): the plan's own
     * shape is the max over the rows, and the sample must agree with it. */
    const uint64_t deep  = f.submit(4200);
    const uint64_t shortq = f.submit(160);
    f.prefill(deep);
    f.prefill(shortq);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy p1 = copy_plan(f.sch);
    REQUIRE(p1.n_seq == 2);
    CHECK_EQ(p1.class_at(deep), 1);
    CHECK_EQ(p1.class_at(shortq), 0);
    f.commit(11, 0, [](const StepEntry&) { return 0; });
    for (uint64_t id : { deep, shortq }) CHECK_OK(f.sch.set_draft(id, six, p1.at(id)));

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    f.sch.step_finished(live);

    REQUIRE(f.sch.step() != nullptr);   /* opens the next cycle: the first one is priced now */
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    REQUIRE(bs.size() == 1);
    CHECK_EQ(bs[0].shape.ctx_class, 1);

    /* The planner's query for this shape at class 1 finds that measurement; at class 0 it can only
     * interpolate. Filed at 0, both answers would be wrong. */
    DraftVerifyShape query = bs[0].shape;
    DraftCostEstimate est;
    CHECK_OK(f.sch.draft_costs().estimate(query, &est));
    CHECK(est.valid);
    CHECK(est.source == DraftCostSource::Measured);
    query.ctx_class = 0;
    CHECK_OK(f.sch.draft_costs().estimate(query, &est));
    CHECK(est.source != DraftCostSource::Measured);
}

/* A deployment that observes without an admissible baseline still builds and validates every
 * sample; the table simply has nowhere to put it. Counted apart from `ignored()` (a sample that
 * proved nothing) and from a refusal (a sample that was malformed), so "nothing was learned" is
 * never mistaken for "nothing was offered". */
TEST(an_observe_against_an_inactive_table_is_counted_and_never_absorbed) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    /* Configured for start(), which refuses Cost without an active table, then dropped: the run
     * observes, and the table is inactive from here. */
    activate_costs(f);
    CHECK_OK(f.start());
    f.sch.draft_costs().clear();
    REQUIRE(!f.sch.draft_costs().active());

    const uint64_t a = f.submit(160);
    f.prefill(a);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy p1 = copy_plan(f.sch);
    f.commit(11, 0, [](const StepEntry&) { return 0; });
    CHECK_OK(f.sch.set_draft(a, six, p1.at(a)));
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    f.sch.step_finished(live);

    const int64_t inactive_before = f.sch.draft_costs().inactive();
    const int64_t ignored_before  = f.sch.draft_costs().ignored();
    REQUIRE(f.sch.step() != nullptr);            /* prices cycle 1 into an inactive table */
    CHECK_EQ(f.sch.draft_costs().inactive(), inactive_before + 1);
    CHECK_EQ(f.sch.draft_costs().ignored(), ignored_before);   /* the sample was not evidence-less */
    CHECK(f.sch.draft_costs().buckets().empty());
    CHECK_EQ(f.sch.draft_cost_refused(), 0);     /* neither a refusal */
    CHECK_EQ(f.sch.metrics().draft_obs_inactive, inactive_before + 1);
}

/* ------------------------------------------------------------------ the policy's hooks */
/* Per request, the conditional acceptance at each draft position. Under the additive price below,
 * a row's optimum is the deepest draft whose survival probability still pays 0.05. */
struct Predictions {
    struct Row {
        uint64_t id = 0;
        float    a[RAD_SCHED_MAX_SPEC] = {};
    };
    std::vector<Row> rows;
    /* The next this many predicts fail: a one-time model error, which leaves the cycle without a
     * plan. Any negative error propagates -- RAD_E_INVAL would be read as "not priced". */
    int fail_next = 0;

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
    Predictions& t = *static_cast<Predictions*>(ctx);
    if (t.fail_next > 0) { --t.fail_next; return RAD_E_STATE; }
    out->n_seq = view.n_seq;
    out->cycle_id = view.cycle_id;        /* the set describes this cycle, as the contract says */
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
    CHECK(!f.sch.serial_batch_failed());  /* the owed history pass was built */
    CHECK(f.sch.serial_batch(2, 3) == nullptr);   /* nothing past the history; rounds stops at 1 */
    /* No drafting row: the null above is a round with nothing to do, not a pass that failed. */
    CHECK(!f.sch.serial_batch_failed());

    /* The pass ran: the watermark moves to what it wrote, and no draft round was needed for it. */
    CHECK_OK(f.commit(11, 0, [](const StepEntry& x) { return x.n_spec; }));
    f.sch.serial_end(0, true);
    CHECK_EQ(f.sch.draft_watermark(id), f.sch.find(id)->n_computed);
}

/* Counter-case to the one above: a snapshot with no row at all. Nothing is owed, so the null batch
 * is idleness -- and only serial_batch_failed() can say that, because `n == 0` looks the same as a
 * K0-only snapshot whose history pass could not be built. */
TEST(an_empty_snapshot_owes_no_pass_and_reports_no_failure) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/512);
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);   /* no request, no row */
    CHECK_EQ(rounds, 1);                                 /* only the history round is asked */

    CHECK(f.sch.serial_batch(0, 1) == nullptr);
    CHECK(!f.sch.serial_batch_failed());
}

/* ================================================================== a KV refusal keeps the history */

/* Counter-case 4: the KV reservation, not the context, takes the draft right. The pool has exactly
 * the blocks the next trunk step needs, so the lookahead reservation (head_max + rounds, `rounds`
 * more than the verify writes) is refused. The row keeps its history, which it already has blocks
 * for -- so `n == 0` and a round 0 that must still be built. That is why the failure status hangs
 * on the planned work and on neither `rounds` nor a row count. */
TEST(a_kv_refusal_takes_the_draft_and_leaves_the_history_pass_owed) {
    Fixture f;
    /* 16 tokens a block, 2 blocks: the verify step's 22 + 7 fits, its 22 + 7 + 6 rounds do not. */
    f.build(/*blocks=*/2, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/512);
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(21);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);

    /* Publish-then-verify: the first cycle verifies what the previous one published, and after the
     * prefill nothing is published. It runs its one token and publishes the plan's six. */
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
    CHECK(e->ctx_len + e->n_spec + 6 <= 512);   /* the context would allow it */

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);   /* the pool, not the context, refused */
    CHECK_EQ(rounds, 1);

    const RadBatch* h = f.sch.serial_batch(0, 2);
    REQUIRE(h != nullptr);                        /* buildable although no row drafts */
    CHECK_EQ(h->n_seq, 1);
    CHECK_EQ(h->n_tok, 7);
    CHECK_EQ(h->n_draft_out, 0);
    CHECK(!f.sch.serial_batch_failed());
}

/* ================================================================== a pass that cannot be built */

/* Counter-case 3 to `all_draft_rows_can_fall_away_and_the_history_pass_is_still_asked_for`: the
 * same K0-only snapshot, but the owed history pass itself cannot be built. The null is then a pass
 * that was owed and lost -- the one reading a driver must not take for idleness, and the one a row
 * count cannot reach, because that snapshot's only row drafts nothing.
 *
 * `sched_fail_next_build()` is the test build's one-shot refusal of the next build that has work to
 * do (tests/sched_fakes.h, behind RAD_TEST_HOOKS). It reaches the builder's own null path, which is
 * what a real refusal looks like to serial_batch(); nothing about it exists in a build without
 * tests. */
TEST(a_k0_history_pass_that_cannot_be_built_reports_a_failure) {
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
    const StepEntry* e = entry_of(f.sch, id);
    REQUIRE(e != nullptr);
    CHECK(e->ctx_len + e->n_spec + 6 > 512);   /* the premise: no room left for a draft round */

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);   /* nobody drafts */
    CHECK_EQ(rounds, 1);                                 /* one pass: the history */

    sched_fail_next_build();
    CHECK(f.sch.serial_batch(0, 2) == nullptr);
    CHECK(f.sch.serial_batch_failed());   /* owed, and not built: not idleness */
}

/* The counter-case for the status itself: it speaks for one call and for one snapshot. A pass that
 * did build must not leave the way open for a later failure to go unreported; a failure must not
 * outlive the call that produced it, nor the snapshot it belonged to. */
TEST(a_failed_pass_is_not_masked_by_an_earlier_success_or_a_new_snapshot) {
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
    const StepEntry* e = entry_of(f.sch, id);
    REQUIRE(e != nullptr);
    CHECK(e->ctx_len + e->n_spec + 6 > 512);

    int rounds = 6;
    Scheduler::SerialRow rows[6];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);
    CHECK_EQ(rounds, 1);

    /* A built pass reports no failure ... */
    REQUIRE(f.sch.serial_batch(0, 2) != nullptr);
    CHECK(!f.sch.serial_batch_failed());

    /* ... and does not hide the next one: the same owed pass, refused once. */
    sched_fail_next_build();
    CHECK(f.sch.serial_batch(0, 3) == nullptr);
    CHECK(f.sch.serial_batch_failed());

    /* The next call answers for itself: past the history nothing is owed, so this null is idle and
     * the failure above is not carried into it. */
    CHECK(f.sch.serial_batch(2, 4) == nullptr);
    CHECK(!f.sch.serial_batch_failed());

    /* Nor does ending the snapshot rewrite the last answer ... */
    sched_fail_next_build();
    CHECK(f.sch.serial_batch(0, 5) == nullptr);
    CHECK(f.sch.serial_batch_failed());
    f.sch.serial_end(0, false);
    CHECK(f.sch.serial_batch_failed());

    /* ... only the next snapshot does, before any pass is asked of it. */
    rounds = 6;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 6), 0);
    CHECK(!f.sch.serial_batch_failed());
    REQUIRE(f.sch.serial_batch(0, 6) != nullptr);   /* one shot, not a broken builder */
    CHECK(!f.sch.serial_batch_failed());
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

/* ================================================================== the key, ahead and direct */

namespace {

/* The run's own table behind a log: the price comes from the table the run also records into, and
 * the last key the planner asked for is kept, so the query is asserted and not inferred. */
struct QueryLog {
    DraftCostTable* table = nullptr;
    DraftVerifyShape last;
    int calls = 0;
    DraftCostModel model() { return DraftCostModel{ this, &priced }; }
    static int priced(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
        QueryLog* self = static_cast<QueryLog*>(ctx);
        self->last = s;
        ++self->calls;
        return self->table->estimate(s, out);
    }
};

/* One row's key as this fixture runs it: the request samples, the serial head is loaded, the
 * cycle publishes before the next step verifies, and the 160-token prompt is context class 0. */
DraftVerifyShape row_key(int32_t draft_k, int32_t verify_k) {
    DraftVerifyShape s;
    s.n_rows         = 1;
    s.ctx_class      = draft_ctx_class(160);
    s.sampling       = DraftSampling::Sampled;
    s.replay         = DraftReplay::None;
    s.drafter        = DraftDrafter::Serial;
    s.pipeline_depth = 1;
    s.draft_k        = draft_k;
    s.verify_k       = verify_k;
    s.verify_tokens  = 1;
    for (int32_t j = 1; j <= draft_k; ++j) s.draft_rows[j] = 1;
    for (int32_t j = 1; j <= verify_k; ++j) { s.rows[j] = 1; ++s.verify_tokens; }
    s.history_tokens = s.verify_tokens;
    return s;
}

/* The same key for `n` identical rows: the batch the budget-cut run plans over. */
DraftVerifyShape rows_key(int32_t n, int32_t draft_k, int32_t verify_k) {
    DraftVerifyShape s = row_key(draft_k, verify_k);
    s.n_rows        = n;
    s.verify_tokens = 0;
    for (int32_t j = 1; j <= draft_k; ++j) s.draft_rows[j] = n;
    for (int32_t j = 1; j <= verify_k; ++j) s.rows[j] = n;
    s.verify_tokens = n * (1 + verify_k);
    s.history_tokens = s.verify_tokens;
    return s;
}

void add_bucket(DraftCostProfile* p, const DraftVerifyShape& key, double ms) {
    DraftCostBucket b;
    b.shape  = key;
    b.rounds = key.draft_k;
    for (int32_t j = 1; j <= key.draft_k; ++j) {
        b.draft_round_rows[j] = key.draft_rows[j];
        b.draft_rows_total += key.draft_rows[j];
    }
    b.mean_ms = ms;
    b.unc_ms  = 0.2;
    b.n       = 3;
    p->buckets.push_back(b);
}

/* A no-loan profile installed before init() as this run's own recording target: the match is what
 * lets Config admit Cost (scheduler.cpp), `recording` keeps it writable, and the four buckets price
 * both alternatives of the cold cycle and of the cycle after a K3 plan. The deep verify window is
 * dearer per token, so the table -- not a predictor change -- decides the fall. */
/* The identity every table here is filed under: a no-loan reference for this fixture's
 * fingerprint, so the run prices from the buckets and absorbs none of its own cycles. */
DraftCostProfile cost_profile() {
    DraftCostProfile p;
    p.fingerprint.model     = "test";
    p.fingerprint.quant     = "q";
    p.fingerprint.tp_wire   = "w";
    p.fingerprint.gpus      = "g";
    p.fingerprint.kernels   = "k";
    p.fingerprint.kv_format = "bf16";
    p.fingerprint.placement = "all_vram";
    p.fingerprint.tp        = 1;
    p.fingerprint.basis     = DraftCostBasis::NoLoan;
    p.fingerprint.arena     = "none";
    p.fingerprint.live_loan = "flex=0 lend=0";
    p.execution = DraftCostExecution::Rectangular;
    return p;
}

void install_table(Fixture& f, const DraftCostProfile& p) {
    DraftCostRuntime rt;
    rt.fingerprint = p.fingerprint;
    rt.execution   = p.execution;
    rt.recording   = true;
    CHECK_EQ(f.sch.draft_costs().configure(p, rt), RAD_OK);
    REQUIRE(f.sch.draft_costs().active());
    CHECK(!f.sch.draft_costs().reference());
}

/* The fall's table: K3 wins the cold cycle, K2 the cycle after it -- on the ahead path from the
 * declared (2,3), on the direct path from the capped (2,2) that cycle executes. */
void install_cost_table(Fixture& f) {
    DraftCostProfile p = cost_profile();
    add_bucket(&p, row_key(3, 0),  6.0);   /* cold cycle: 3.439 tokens / 6.0 ms beat K2 */
    add_bucket(&p, row_key(2, 0),  5.0);
    add_bucket(&p, row_key(3, 3), 12.0);   /* after K3: 3.439 / 12.0 ... */
    add_bucket(&p, row_key(2, 3),  9.0);   /* ... loses to 2.71 / 9.0 ahead ... */
    add_bucket(&p, row_key(2, 2),  8.0);   /* ... and to 2.71 / 8.0 on the direct path */
    install_table(f, p);
}

/* The upshift's table: K2 wins the cold cycle and K3 the cycle after it (after K2 both paths
 * verify two, so the two price the same shape). */
void install_upshift_table(Fixture& f) {
    DraftCostProfile p = cost_profile();
    add_bucket(&p, row_key(2, 0),  4.0);   /* cold: 2.71 / 4.0 beats 3.439 / 20.0 */
    add_bucket(&p, row_key(3, 0), 20.0);
    add_bucket(&p, row_key(2, 2),  8.0);
    add_bucket(&p, row_key(3, 2),  6.0);   /* after K2: 3.439 / 6.0 beats 2.71 / 8.0 */
    install_table(f, p);
}

/* The budget cut's table, for the six-row run: K3 wins the cold cycle and the uncut search after
 * it, so the plan the policy would issue is K3 there; what actually runs is the cut K2, re-priced
 * on the window its step verifies -- (2,3) ahead, (2,2) on the direct path. */
void install_budget_cut_table(Fixture& f) {
    DraftCostProfile p = cost_profile();
    add_bucket(&p, rows_key(6, 3, 0),  6.0);   /* cold: 3.439 / 6.0 beats 2.71 / 5.0 */
    add_bucket(&p, rows_key(6, 2, 0),  5.0);
    add_bucket(&p, rows_key(6, 3, 3),  8.0);   /* after K3: the uncut winner at 3.439 / 8.0 ... */
    add_bucket(&p, rows_key(6, 2, 3),  9.0);   /* ... and the cut re-price ahead ... */
    add_bucket(&p, rows_key(6, 2, 2),  8.0);   /* ... and on the direct path */
    install_table(f, p);
}

/* What the two cycles saw, in the three places the contract has to agree: the last price the
 * planner asked for (the issued plan's own estimate), the verify batch the engine was handed, and
 * the key of the bucket the cycle's sample was actually filed under. */
struct Coupled {
    /* Owned here, not in the helper: the planner keeps the model's and the predictor's `self`
     * pointers, so both must outlive the cycles the caller runs after the helper returned. */
    QueryLog log;
    Predictions pred;
    uint64_t id = 0;
    int32_t plan_k = 0;
    DraftPlanSource source = DraftPlanSource::Fallback;
    DraftVerifyShape query;
    DraftTierPrice issued;
    int32_t batch_spec = 0;
    int32_t entry_spec = 0;
    int32_t tiers_priced = 0;
    int32_t uncut_k = 0;
    double  uncut_ms = 0.0;
    int32_t cut_rows = 0;   /* rows the plan says the budget scaled */
    DraftCostObserve observe;
    DraftVerifyShape recorded;
    int32_t recorded_rounds = 0;
};

/* Two cycles of the real pipeline: A at K3 (nothing frozen, the cold bucket decides), then B after
 * A published three, where the table prices K2 as the cheaper plan. `ahead` runs the engine's other
 * order, in which the step is issued before the next plan opens. `max_tokens` bounds the row's
 * output, and with it the drafts: a value just above what the two cycles emit makes the second
 * plan resource-bound rather than price-bound. */
void downshift(Fixture& f, bool ahead, Coupled& c, int max_tokens = 64) {
    f.build();
    f.cfg.n_spec = 3;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 3;
    install_cost_table(f);
    CHECK_OK(f.start(DraftReplay::None));

    Predictions& pred = c.pred;
    const uint64_t id = f.submit(160, max_tokens);
    c.id = id;
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });

    QueryLog& log = c.log;
    log.table = &f.sch.draft_costs();
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 3;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = log.model();
    CHECK_OK(f.sch.draft().configure(dp));

    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    /* The evidence a backend without a tape reports: a live pass, unkeyed. Without it no cycle's
     * bracket is attributable and no sample is filed. */
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    const int32_t three[3] = { 21, 22, 23 };
    const int32_t two[2]   = { 21, 22 };

    /* Cycle A: the cold cycle. The deepest offered tier is priced and wins, the step verifies
     * nothing, and A publishes its three drafts after it. */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* a = f.sch.draft().plan();
    REQUIRE(a != nullptr);
    REQUIRE_EQ(a->n_seq, 1);
    REQUIRE_EQ(a->k[0], 3);

    if (ahead) {
        /* The engine's order: expect_draft and the step issued ahead happen under A, before its
         * commit; promote_next() is what opens B's plan afterwards. */
        f.sch.expect_draft(id, 3);
        const RadBatch* built = f.sch.step_next();
        REQUIRE(built != nullptr);
        c.batch_spec = built->n_spec;
        f.commit_plain(11);
    } else {
        f.commit_plain(11);
    }
    CHECK_OK(f.sch.set_draft(id, three, 3));
    f.sch.step_finished(live);
    if (ahead) {
        f.sch.promote_next();
    } else {
        /* The direct order: step() opens B's plan and only then builds the step that verifies
         * A's three. */
        const RadBatch* built = f.sch.step();
        REQUIRE(built != nullptr);
        c.batch_spec = built->n_spec;
    }

    const DraftPlan* b = f.sch.draft().plan();
    REQUIRE(b != nullptr);
    c.plan_k = b->k[0];
    c.source = b->source;
    c.query  = log.last;   /* the issued plan's own estimate, the cycle's last estimate call */
    c.issued = f.sch.draft().planner().last_price();
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)1);
    c.entry_spec = f.sch.plan().e[0].n_spec;

    /* B's step commits, the head publishes under B's own K, and the bracket closes the cycle. */
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, two, 2));
    f.sch.step_finished(live);
    f.sch.flush_draft_observations();

    const DraftCostTable& t = f.sch.draft_costs();
    c.observe = t.last_observe();
    if (c.observe.what == DraftCostDisposition::Recorded && c.observe.bucket >= 0 &&
        (size_t)c.observe.bucket < t.buckets().size()) {
        const DraftCostBucket& bkt = t.buckets()[(size_t)c.observe.bucket];
        c.recorded        = bkt.shape;
        c.recorded_rounds = bkt.rounds;
    }
}

/* The other direction on the same pipeline: A at K2 (the cold bucket keeps it there), B at K3
 * after A published two. Both paths verify the frozen two, so both price (3,2). */
void upshift(Fixture& f, bool ahead, Coupled& c) {
    f.build();
    f.cfg.n_spec = 3;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 3;
    install_upshift_table(f);
    CHECK_OK(f.start(DraftReplay::None));

    Predictions& pred = c.pred;
    const uint64_t id = f.submit(160);
    c.id = id;
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });

    QueryLog& log = c.log;
    log.table = &f.sch.draft_costs();
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 3;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = log.model();
    CHECK_OK(f.sch.draft().configure(dp));

    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    const int32_t two[2]   = { 21, 22 };
    const int32_t three[3] = { 21, 22, 23 };

    /* Cycle A: K2, the cold price's choice. It publishes two after its step. */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* a = f.sch.draft().plan();
    REQUIRE(a != nullptr);
    REQUIRE_EQ(a->n_seq, 1);
    REQUIRE_EQ(a->k[0], 2);

    if (ahead) {
        f.sch.expect_draft(id, 2);
        const RadBatch* built = f.sch.step_next();
        REQUIRE(built != nullptr);
        c.batch_spec = built->n_spec;
        f.commit_plain(11);
    } else {
        f.commit_plain(11);
    }
    CHECK_OK(f.sch.set_draft(id, two, 2));
    f.sch.step_finished(live);
    if (ahead) {
        f.sch.promote_next();
    } else {
        const RadBatch* built = f.sch.step();
        REQUIRE(built != nullptr);
        c.batch_spec = built->n_spec;
    }

    const DraftPlan* b = f.sch.draft().plan();
    REQUIRE(b != nullptr);
    c.plan_k = b->k[0];
    c.source = b->source;
    c.query  = log.last;
    c.issued = f.sch.draft().planner().last_price();
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)1);
    c.entry_spec = f.sch.plan().e[0].n_spec;

    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, three, 3));
    f.sch.step_finished(live);
    f.sch.flush_draft_observations();

    const DraftCostTable& t = f.sch.draft_costs();
    c.observe = t.last_observe();
    if (c.observe.what == DraftCostDisposition::Recorded && c.observe.bucket >= 0 &&
        (size_t)c.observe.bucket < t.buckets().size()) {
        const DraftCostBucket& bkt = t.buckets()[(size_t)c.observe.bucket];
        c.recorded        = bkt.shape;
        c.recorded_rounds = bkt.rounds;
    }
}

/* A genuine KV budget cut with its re-price, on the real pipeline. Six rows run cycle A uncut at
 * K3; before B opens the pool is left with one free block -- sixteen tokens against the uncut six
 * times three -- so `apply_draft_budget` scales every K to two and the plan says why. The search
 * still weighs the uncut K3 (and would issue it); the cut plan is re-priced on the window its
 * step really runs, and that is the shape the batch and the sample carry. `ahead` runs the
 * engine's other order: the step built under A is the one B's cycle measures. */
void budget_cut(Fixture& f, bool ahead, Coupled& c) {
    /* The six 160-token prompts must prefill in one step: a split would open row-bearing cycles of
     * its own before the two this test reads. */
    f.build(4096, 64, /*max_tok=*/1024);
    f.cfg.n_spec = 3;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 3;
    install_budget_cut_table(f);
    CHECK_OK(f.start(DraftReplay::None));

    Predictions& pred = c.pred;
    uint64_t ids[6];
    for (int i = 0; i < 6; ++i) ids[i] = f.submit(160);
    for (uint64_t id : ids) pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });

    QueryLog& log = c.log;
    log.table = &f.sch.draft_costs();
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 3;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = log.model();
    CHECK_OK(f.sch.draft().configure(dp));

    for (uint64_t id : ids) f.prefill(id);
    for (uint64_t id : ids) f.sch.find(id)->speculation_fixed = true;

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    const int32_t three[3] = { 21, 22, 23 };
    const int32_t two[2]   = { 21, 22 };

    /* Cycle A: uncut. The cold bucket prices K3 and issues it; the pool is still wide. */
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy a = copy_plan(f.sch);
    REQUIRE_EQ(a.n_seq, 6);
    CHECK_EQ(a.max_k, 3);
    CHECK_EQ(a.verify_k, 0);
    for (uint64_t id : ids) CHECK_EQ(a.at(id), 3);

    if (ahead) {
        for (uint64_t id : ids) f.sch.expect_draft(id, 3);
        const RadBatch* built = f.sch.step_next();
        REQUIRE(built != nullptr);
        CHECK_EQ(built->n_spec, 3);
        /* This is the step that runs once promote_next() makes it current: B's batch. */
        c.batch_spec = built->n_spec;
        f.commit_plain(11);
    } else {
        f.commit_plain(11);
    }
    for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, three, 3));
    f.sch.step_finished(live);

    /* One free block is sixteen tokens; six rows at K3 want eighteen. */
    REQUIRE(f.kv.free_blocks(0) >= 1);
    f.kv.unbacked = f.kv.free_blocks(0) - 1;

    if (ahead) {
        f.sch.promote_next();
    } else {
        const RadBatch* built = f.sch.step();
        REQUIRE(built != nullptr);
        c.batch_spec = built->n_spec;
    }

    const DraftPlan* b = f.sch.draft().plan();
    REQUIRE(b != nullptr);
    REQUIRE_EQ(b->n_seq, 6);
    c.plan_k = b->max_k;
    c.source = b->source;
    c.tiers_priced = f.sch.draft().planner().last_tiers_priced();
    const DraftTierPrice* tp = f.sch.draft().planner().last_tier_prices();
    const int32_t ntp = f.sch.draft().planner().last_n_tier_prices();
    if (ntp > 0) { c.uncut_k = tp[ntp - 1].k; c.uncut_ms = tp[ntp - 1].ms; }
    for (int32_t i = 0; i < b->n_seq; ++i)
        if (b->why[i] == DraftFallback::Resources) ++c.cut_rows;
    c.query  = log.last;   /* the issued plan's own estimate, the cycle's last estimate call */
    c.issued = f.sch.draft().planner().last_price();
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)6);
    c.entry_spec = f.sch.plan().e[0].n_spec;

    f.commit_plain(11);
    for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, two, 2));
    f.sch.step_finished(live);
    f.sch.flush_draft_observations();

    const DraftCostTable& t = f.sch.draft_costs();
    c.observe = t.last_observe();
    if (c.observe.what == DraftCostDisposition::Recorded && c.observe.bucket >= 0 &&
        (size_t)c.observe.bucket < t.buckets().size()) {
        const DraftCostBucket& bkt = t.buckets()[(size_t)c.observe.bucket];
        c.recorded        = bkt.shape;
        c.recorded_rounds = bkt.rounds;
    }
}

/* The failed open's table: K2 wins the cold cycle (2.71 tokens / 5.0 ms against 3.439 / 7.0), and
 * the cycle after the legacy K3 must price K3 on that window (3.439 / 8.0 against 2.71 / 9.0). The
 * (3,2) bucket is what a plan still naming the pre-failure window would price instead, so the
 * issued depth alone does not tell the two apart. (2,2) prices the incumbent the direct path's
 * raise check re-caps to the previous K; dear enough that K3's raise still clears it. */
void install_no_plan_table(Fixture& f) {
    DraftCostProfile p = cost_profile();
    add_bucket(&p, row_key(2, 0), 5.0);
    add_bucket(&p, row_key(3, 0), 7.0);
    add_bucket(&p, row_key(3, 2), 4.0);
    add_bucket(&p, row_key(3, 3), 8.0);
    add_bucket(&p, row_key(2, 2), 7.0);
    add_bucket(&p, row_key(2, 3), 9.0);
    install_table(f, p);
}

/* Engine::issue_serial_drafter's host half, for the step that was just built: what the driver asks
 * for, the passes the card builds, and the deepest round any row was still in -- which is what
 * issue_ahead() hands expect_draft(), per drafting row, before the step after this one is built.
 * Every gate is the scheduler's own: serial_begin() decides the rounds and each row's depth,
 * serial_batch() decides which passes exist. Nothing here names a depth the run would not take. */
int32_t serial_ran(Fixture& f, int step) {
    int rounds = f.prog.drafter.depth;
    Scheduler::SerialRow rows[RAD_SCHED_MAX_SPEC];
    const int n = f.sch.serial_begin(&rounds, rows, RAD_SCHED_MAX_SPEC);
    if (rounds <= 0) return 0;
    int32_t done = 0;
    for (int rd = 0; rd <= rounds; rd += rd == 0 ? 2 : 1) {
        if (rd > 0 && n <= 0) break;
        const RadBatch* db = f.sch.serial_batch(rd, step);
        if (!db) continue;
        const int drafted = rd == 0 ? (int)db->n_draft_out : (int)db->n_seq;
        if (drafted > 0) done = rd == 0 ? 1 : rd;
    }
    for (int i = 0; i < n; ++i) f.sch.expect_draft(rows[i].seq, done);
    return done;
}

/* A planned cycle, `fails` refused opens, and the plan after them. Cycle A takes K2 from the cold
 * bucket. Each refused open runs legacy -- no ring record, no sample, no price -- and the plan after
 * them must name the window those legacy steps really verify. The table prices K3 only on that
 * window, so a plan still naming the pre-failure K2 would ask for (3,2) instead.
 *
 * `ineligible` turns the row into one that may not speculate just before the refused opens: the
 * engine publishes nothing for it (finish_serial_drafter's allows_speculation gate), while the
 * serial drafter still proposes for it, so only the built window says what its step verifies.
 * `max_tokens` bounds the row's output, and with it the window cap_of() would grant it. */
struct FailOpts {
    bool ahead = false;
    int  fails = 1;
    bool ineligible = false;
    int  max_tokens = 64;
    int  head_depth = 3;   /* the drafter's own depth; below cfg.n_spec it binds the rounds */
};

void failed_opens(Fixture& f, Coupled& c, const FailOpts& o) {
    f.build();
    f.cfg.n_spec = 3;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = o.head_depth;
    install_no_plan_table(f);
    CHECK_OK(f.start(DraftReplay::None));

    Predictions& pred = c.pred;
    const uint64_t id = f.submit(160, o.max_tokens);
    c.id = id;
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });

    QueryLog& log = c.log;
    log.table = &f.sch.draft_costs();
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 3;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = log.model();
    CHECK_OK(f.sch.draft().configure(dp));

    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    const int32_t two[2]   = { 21, 22 };
    const int32_t three[3] = { 21, 22, 23 };

    /* Cycle A: the cold bucket's K2, and its two drafts after the step. */
    REQUIRE(f.sch.step() != nullptr);
    {
        const PlanCopy a = copy_plan(f.sch);
        REQUIRE_EQ(a.n_seq, 1);
        CHECK_EQ(a.k[0], 2);
        CHECK_EQ(a.verify_k, 0);
    }
    if (o.ahead) {
        const int32_t ran = serial_ran(f, 1);
        CHECK_EQ(ran, 2);                 /* the plan's own window, not the head's three */
        const RadBatch* built = f.sch.step_next();
        REQUIRE(built != nullptr);
        CHECK_EQ(built->n_spec, 2);
        f.commit_plain(11);
    } else {
        f.commit_plain(11);
    }
    CHECK_OK(f.sch.set_draft(id, two, 2));
    f.sch.step_finished(live);

    if (o.ineligible) f.sch.find(id)->may_speculate = false;

    const int64_t calls_before = log.calls;
    int32_t ran = 0;
    for (int k = 0; k < o.fails; ++k) {
        pred.fail_next = 1;   /* the one-time model error */
        if (o.ahead) {
            /* The step this cycle runs was built before its open, under the plan before it: that
             * is the window it verifies. The step after it is built now, under no plan, from the
             * rounds the serial drafter really ran. */
            f.sch.promote_next();
            ran = serial_ran(f, 2 + k);
            const RadBatch* built = f.sch.step_next();
            REQUIRE(built != nullptr);
            CHECK_EQ(built->n_spec, ran);
            c.batch_spec = built->n_spec;
        } else {
            /* step() opens (and refuses) the plan, then builds the step after it: A's two first,
             * this cycle's publication from the next refused cycle on. */
            const RadBatch* built = f.sch.step();
            REQUIRE(built != nullptr);
            /* A row that may not speculate is refused by the build itself, whatever it published. */
            const int32_t want = o.ineligible ? 0 : (k == 0 ? 2 : f.sch.find(id)->n_draft);
            CHECK_EQ(built->n_spec, want);
            c.batch_spec = built->n_spec;
            ran = serial_ran(f, 2 + k);
        }
        f.commit_plain(11);
        /* The engine publishes a row the sampler will speculate on and skips the one it will not
         * (finish_serial_drafter); the window the step verifies is the same either way. */
        if (!o.ineligible) CHECK_OK(f.sch.set_draft(id, three, ran));
        f.sch.serial_end(ran, true);
        f.sch.step_finished(live);
    }

    /* Counted apart, and none of them opened a record, asked a price or failed a sample. */
    CHECK_EQ(f.sch.metrics().draft_cost_no_plan, o.fails);
    CHECK_EQ(f.sch.draft_cycles_done(), 1);   /* A's record, closed at the first refused open */
    CHECK_EQ(log.calls, calls_before);
    CHECK_EQ(f.sch.draft_cost_refused(), 0);

    /* The plan after the refused opens: priced on the window the legacy steps really ran. */
    if (o.ahead) {
        f.sch.promote_next();
        c.batch_spec = f.sch.plan().e.empty() ? 0 : f.sch.plan().e[0].n_spec;
    } else {
        const RadBatch* built = f.sch.step();
        REQUIRE(built != nullptr);
        c.batch_spec = built->n_spec;
    }

    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    c.plan_k = p->max_k;
    c.source = p->source;
    c.query  = log.last;
    c.issued = f.sch.draft().planner().last_price();
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)1);
    c.entry_spec = f.sch.plan().e[0].n_spec;

    /* The cycle's own sample, filed under the same window. */
    f.commit_plain(11);
    if (!o.ineligible) CHECK_OK(f.sch.set_draft(id, three, ran));
    f.sch.step_finished(live);
    f.sch.flush_draft_observations();
    CHECK_EQ(f.sch.draft_cycles_done(), 2);

    const DraftCostTable& t = f.sch.draft_costs();
    c.observe = t.last_observe();
    if (c.observe.what == DraftCostDisposition::Recorded && c.observe.bucket >= 0 &&
        (size_t)c.observe.bucket < t.buckets().size()) {
        const DraftCostBucket& bkt = t.buckets()[(size_t)c.observe.bucket];
        c.recorded        = bkt.shape;
        c.recorded_rounds = bkt.rounds;
    }
}

}  /* namespace */

/* The confirmed no_plan residual on the direct path: after a refused open the frozen state kept
 * the plan before it, so the next plan asked for K3 on the K2 verify window while its step ran
 * three. The window the legacy cycle really publishes is now frozen, and query, batch and sample
 * name it. */
TEST(a_cycle_with_no_plan_freezes_the_window_its_legacy_step_verifies) {
    Coupled c;   /* destroyed after `f`: the planner and the predictor point into `c` until fini() */
    Fixture f;
    failed_opens(f, c, FailOpts{});

    CHECK_EQ(c.plan_k, 3);   /* K3 priced on (3,3) beats K2 on (2,3) */
    CHECK(c.source == DraftPlanSource::Cost);
    CHECK_EQ(c.query.draft_k, 3);
    CHECK_EQ(c.query.verify_k, 3);   /* not the pre-failure two */
    CHECK_EQ(c.query.verify_tokens, 4);
    CHECK_EQ(c.query.history_tokens, 4);
    CHECK(c.issued.valid);
    CHECK_EQ(c.issued.k, 3);
    CHECK_NEAR(c.issued.ms, 8.0, 1e-9);
    CHECK_EQ(c.batch_spec, 3);
    CHECK_EQ(c.entry_spec, 3);
    CHECK(c.observe.what == DraftCostDisposition::Recorded);
    CHECK_EQ(c.recorded.draft_k, 3);
    CHECK_EQ(c.recorded.verify_k, 3);
    CHECK_EQ(c.recorded.verify_tokens, 4);
    CHECK_EQ(c.recorded.history_tokens, 4);
    CHECK_EQ(c.recorded_rounds, 3);
    CHECK_EQ(c.query.verify_k, c.entry_spec);
    CHECK_EQ(c.recorded.verify_k, c.entry_spec);
}

/* The same failure with the step issued ahead: the legacy step's window is the one that step
 * really runs, and the plan after it declares exactly that. */
TEST(a_cycle_with_no_plan_freezes_the_window_its_legacy_step_verifies_ahead) {
    Coupled c;   /* destroyed after `f`: the planner and the predictor point into `c` until fini() */
    Fixture f;
    failed_opens(f, c, FailOpts{ .ahead = true });

    CHECK_EQ(c.plan_k, 3);
    CHECK(c.source == DraftPlanSource::Cost);
    CHECK_EQ(c.query.draft_k, 3);
    CHECK_EQ(c.query.verify_k, 3);
    CHECK_EQ(c.query.verify_tokens, 4);
    CHECK(c.issued.valid);
    CHECK_EQ(c.issued.k, 3);
    CHECK_NEAR(c.issued.ms, 8.0, 1e-9);
    CHECK_EQ(c.batch_spec, 3);
    CHECK_EQ(c.entry_spec, 3);
    CHECK(c.observe.what == DraftCostDisposition::Recorded);
    CHECK_EQ(c.recorded.draft_k, 3);
    CHECK_EQ(c.recorded.verify_k, 3);
    CHECK_EQ(c.recorded.verify_tokens, 4);
    CHECK_EQ(c.recorded_rounds, 3);
    CHECK_EQ(c.query.verify_k, c.entry_spec);
    CHECK_EQ(c.recorded.verify_k, c.entry_spec);
}

/* Repeated refusals: the frozen window is rewritten to the same legacy depth every time, and the
 * plan after them still names it. */
TEST(repeated_cycles_with_no_plan_keep_the_synced_legacy_window_on_both_paths) {
    {
        Coupled c;
        Fixture f;
        failed_opens(f, c, FailOpts{ .ahead = false, .fails = 2 });

        CHECK_EQ(c.plan_k, 3);
        CHECK(c.source == DraftPlanSource::Cost);
        CHECK_EQ(c.query.verify_k, 3);
        CHECK_EQ(c.batch_spec, 3);
        CHECK_EQ(c.entry_spec, 3);
        CHECK_EQ(c.recorded.verify_k, 3);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
    }
    {
        Coupled c;
        Fixture f;
        failed_opens(f, c, FailOpts{ .ahead = true, .fails = 2 });

        CHECK_EQ(c.plan_k, 3);
        CHECK(c.source == DraftPlanSource::Cost);
        CHECK_EQ(c.query.verify_k, 3);
        CHECK_EQ(c.batch_spec, 3);
        CHECK_EQ(c.entry_spec, 3);
        CHECK_EQ(c.recorded.verify_k, 3);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
    }
}

/* A row that may not speculate publishes nothing in a legacy cycle -- finish_serial_drafter skips
 * it, and the direct path's set_draft takes nothing -- but that is not the window its step
 * verifies. The serial drafter proposes for it like any other row (only the block drafter and the
 * step's own speculation gate refuse it), so the step issued ahead verifies the rounds the head
 * really ran, and the plan after the refusal must name them. The direct path is the other half:
 * there the step itself refuses the row, so its window is zero. */
TEST(a_row_that_may_not_speculate_freezes_the_window_its_step_verifies) {
    {
        Coupled c;   /* destroyed after `f`: the planner points into `c` until fini() */
        Fixture f;
        failed_opens(f, c, FailOpts{ .ahead = true, .ineligible = true });

        CHECK_EQ(c.batch_spec, 3);       /* the head ran its three rounds for the row ... */
        CHECK_EQ(c.query.verify_k, 3);   /* ... and the plan after it names them, not the zero a
                                          * publication-based window would */
        CHECK_EQ(c.query.verify_tokens, 4);
        CHECK_EQ(c.plan_k, 0);           /* the plan still drafts nothing for it */
        CHECK_EQ(c.entry_spec, 3);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
    }
    {
        Coupled c;
        Fixture f;
        failed_opens(f, c, FailOpts{ .ahead = false, .ineligible = true });

        CHECK_EQ(c.batch_spec, 0);       /* the direct step refuses the row */
        CHECK_EQ(c.query.verify_k, 0);
        CHECK_EQ(c.entry_spec, 0);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
    }
}

/* The output budget is not the verify window. A row with two tokens left may draft nothing under
 * the plan's own budget (cap_of), while the legacy cycle's rounds are bounded only by the head's
 * depth: the step issued ahead verifies all three, and the plan after the refusal must say three.
 * `drafter.depth < n_spec` is the other bound: the head's own depth is what runs, so the window
 * there is the head's two and not the ceiling's three. */
TEST(the_frozen_window_is_the_one_the_legacy_step_was_built_with) {
    {
        Coupled c;
        Fixture f;
        /* Four tokens out of five are committed by the time the plan after the refusal opens: the
         * row's cap_of() is one, and the step it is paired with verifies three. */
        failed_opens(f, c, FailOpts{ .ahead = true, .max_tokens = 5 });

        CHECK_EQ(c.batch_spec, 3);
        CHECK_EQ(c.query.draft_k, 1);
        CHECK_EQ(c.query.verify_k, 3);   /* not the two the row's remaining output would cap */
        CHECK_EQ(c.entry_spec, 3);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
    }
    {
        Coupled c;
        Fixture f;
        /* A head two rounds deep under a ceiling of three runs two, on both paths. */
        failed_opens(f, c, FailOpts{ .ahead = true, .head_depth = 2 });

        CHECK_EQ(c.batch_spec, 2);
        CHECK_EQ(c.query.verify_k, 2);   /* the head's depth, not the ceiling's three */
        CHECK_EQ(c.entry_spec, 2);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
    }
}

/* A frozen window belongs to the slot's generation, not to the slot. A new request admitted into a
 * recycled slot verifies its own publication (none yet), not the window the request before it left
 * in that slot. The frozen list is also re-keyed at every plan open, so an entry can never outlive
 * the cycle that wrote it by more than one open; the generation keeps the last window from naming a
 * different request's rows inside that one. */
TEST(a_recycled_slot_does_not_inherit_the_frozen_window) {
    Coupled c;
    Fixture f;
    f.build();
    f.cfg.n_spec = 3;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 3;
    install_no_plan_table(f);
    CHECK_OK(f.start(DraftReplay::None));

    QueryLog& log = c.log;
    log.table = &f.sch.draft_costs();
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 3;
    dp.policy.n_tiers  = 3;
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &c.pred, predict };
    dp.costs     = log.model();
    CHECK_OK(f.sch.draft().configure(dp));

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    const int32_t three[3] = { 21, 22, 23 };

    const uint64_t a = f.submit(160);
    c.pred.set(a, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    f.prefill(a);
    REQUIRE(f.sch.step() != nullptr);
    const int32_t a_slot = f.sch.draft().plan()->slot[0];
    const uint32_t a_gen = f.sch.draft().plan()->slot_gen[0];
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, three, 3));

    /* A refused cycle publishes the legacy three, which is what the next request must not see. */
    c.pred.fail_next = 1;
    const RadBatch* built = f.sch.step();
    REQUIRE(built != nullptr);
    CHECK_EQ(built->n_spec, 2);   /* A's own publication, capped by its plan */
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, three, 3));
    f.sch.step_finished(live);

    /* The first request ends and its slot goes back to the pool. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.commit(99, /*eos=*/1), RAD_OK);
    CHECK_OK(f.sch.reap(a));
    CHECK(f.sch.find(a) == nullptr);

    /* A new request lands in that slot with a new generation, and verifies nothing. */
    const uint64_t b = f.submit(160);
    c.pred.set(b, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    f.prefill(b);
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    REQUIRE_EQ(p->n_seq, 1);
    CHECK_EQ(p->req_id[0], b);
    CHECK_EQ(p->slot[0], a_slot);      /* the slot really is the one the request before it used */
    CHECK(p->slot_gen[0] != a_gen);
    c.entry_spec = f.sch.plan().e[0].n_spec;
    CHECK_EQ(c.entry_spec, 0);
    CHECK_EQ(p->verify_k, 0);   /* not the three the previous request in this slot left */
    CHECK_EQ(c.log.last.verify_k, 0);
}

/* A refused open takes no calibration position. The sequence is [2, 3]: the first row-bearing cycle
 * takes K2, the refused cycle after it takes nothing (choose() never runs for it), and the next
 * cycle must take K3 -- the position after K2 -- on the legacy window the refused cycle published.
 * Had the refusal consumed a step of the rotation, that cycle would run K2 again; had it not synced
 * the frozen window, the plan would verify the K2 the plan before the failure named instead of the
 * three the legacy step ran. */
TEST(a_cycle_with_no_plan_does_not_advance_the_calibration_sequence) {
    Coupled c;
    Fixture f;
    f.build();
    f.cfg.n_spec = 3;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 3;
    install_no_plan_table(f);
    CHECK_OK(f.start(DraftReplay::None));

    Predictions& pred = c.pred;
    QueryLog& log = c.log;
    log.table = &f.sch.draft_costs();
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode       = DraftPolicyMode::Cost;
    dp.policy.tiers[0]   = 0;
    dp.policy.tiers[1]   = 2;
    dp.policy.tiers[2]   = 3;
    dp.policy.n_tiers    = 3;
    dp.policy.calib_k[0] = 2;
    dp.policy.calib_k[1] = 3;
    dp.policy.n_calib_k  = 2;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = log.model();
    CHECK_OK(f.sch.draft().configure(dp));

    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f });
    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    const int32_t three[3] = { 21, 22, 23 };

    /* The first row-bearing cycle: the sequence's K2, and no price is asked for a forced plan. */
    REQUIRE(f.sch.step() != nullptr);
    {
        const PlanCopy a = copy_plan(f.sch);
        REQUIRE_EQ(a.n_seq, 1);
        CHECK_EQ(a.k[0], 2);
        CHECK(f.sch.draft().plan()->source == DraftPlanSource::Calibration);
    }
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, three, 2));
    f.sch.step_finished(live);
    CHECK_EQ(log.calls, 0);

    /* The one-time model error: no plan, the legacy depth, and the sequence stays where it is. */
    pred.fail_next = 1;
    const RadBatch* built = f.sch.step();
    REQUIRE(built != nullptr);
    CHECK_EQ(built->n_spec, 2);   /* the step built after the refused open: A's two, not the ceiling */
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, three, 3));
    f.sch.step_finished(live);
    CHECK_EQ(f.sch.metrics().draft_cost_no_plan, 1);
    CHECK_EQ(f.sch.draft_cycles_done(), 1);
    CHECK_EQ(log.calls, 0);

    /* The next cycle takes the sequence's K3, and names the three the legacy step published. */
    REQUIRE(f.sch.step() != nullptr);
    {
        const PlanCopy p = copy_plan(f.sch);
        REQUIRE_EQ(p.n_seq, 1);
        CHECK_EQ(p.k[0], 3);
        CHECK(f.sch.draft().plan()->source == DraftPlanSource::Calibration);
        CHECK_EQ(p.verify_k, 3);   /* not the two the plan before the failure named */
        CHECK_EQ(p.pipeline_depth, 1);
    }
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, three, 3));
    f.sch.step_finished(live);
    f.sch.flush_draft_observations();
    CHECK_EQ(f.sch.draft_cycles_done(), 2);
    CHECK_EQ(log.calls, 0);   /* a calibration prices nothing, with or without a refusal before it */

    const DraftCostTable& t = f.sch.draft_costs();
    const DraftCostObserve o = t.last_observe();
    REQUIRE(o.what == DraftCostDisposition::Recorded);
    REQUIRE(o.bucket >= 0);
    const DraftCostBucket& b = t.buckets()[(size_t)o.bucket];
    CHECK_EQ(b.shape.draft_k, 3);
    CHECK_EQ(b.shape.verify_k, 3);
    CHECK_EQ(b.rounds, 3);
    CHECK_EQ(b.shape.verify_k, f.sch.plan().e[0].n_spec);
}

/* The coupled proof, ahead path. B's plan is opened after the step that verifies A's three drafts
 * was built, so the step really verifies three. The lookup asks for exactly that window, the batch
 * carries it, and the sample is filed under the same key: query, execution and measurement agree. */
TEST(a_fall_issued_ahead_prices_the_window_the_ahead_step_verifies) {
    Coupled c;   /* destroyed after `f`: the planner's model points into `c.log` until fini() */
    Fixture f;
    downshift(f, /*ahead=*/true, c);

    CHECK_EQ(c.plan_k, 2);   /* the table's price decided the fall */
    CHECK(c.source == DraftPlanSource::Cost);
    /* The lookup named the window the ahead step verifies: a K2 plan that verifies the three
     * drafts the K3 cycle published before it existed -- four verify and four history tokens. */
    CHECK_EQ(c.query.draft_k, 2);
    CHECK_EQ(c.query.verify_k, 3);
    CHECK_EQ(c.query.verify_tokens, 4);
    CHECK_EQ(c.query.history_tokens, 4);
    CHECK(c.issued.valid);
    CHECK_EQ(c.issued.k, 2);
    CHECK_NEAR(c.issued.ms, 9.0, 1e-9);
    /* The ahead step was built before the plan and verifies 3; so does the step that runs. */
    CHECK_EQ(c.batch_spec, 3);
    CHECK_EQ(c.entry_spec, 3);
    /* And the cycle's measurement is filed under the same key. */
    CHECK(c.observe.what == DraftCostDisposition::Recorded);
    CHECK_EQ(c.recorded.draft_k, 2);
    CHECK_EQ(c.recorded.verify_k, 3);
    CHECK_EQ(c.recorded.verify_tokens, 4);
    CHECK_EQ(c.recorded.history_tokens, 4);
    CHECK_EQ(c.recorded_rounds, 2);
    CHECK_EQ(c.query.verify_k, c.entry_spec);
    CHECK_EQ(c.recorded.verify_k, c.entry_spec);
}

/* The same two cycles on the direct path, where step() opens B's plan before building the step
 * and draft_cap re-caps the published three to K2. The cycle view names that path, so the lookup
 * asks for the capped window and query, batch and sample name it too: (2,2), 3 verify and 3
 * history tokens. */
TEST(the_step_built_after_the_plan_prices_the_window_it_verifies) {
    Coupled c;   /* destroyed after `f`: the planner's model points into `c.log` until fini() */
    Fixture f;
    downshift(f, /*ahead=*/false, c);

    CHECK_EQ(c.plan_k, 2);
    CHECK(c.source == DraftPlanSource::Cost);
    /* The capped window, priced from its own bucket ... */
    CHECK_EQ(c.query.draft_k, 2);
    CHECK_EQ(c.query.verify_k, 2);
    CHECK_EQ(c.query.verify_tokens, 3);
    CHECK_EQ(c.query.history_tokens, 3);
    CHECK(c.issued.valid);
    CHECK_EQ(c.issued.k, 2);
    CHECK_NEAR(c.issued.ms, 8.0, 1e-9);
    /* ... against a step that runs exactly that. */
    CHECK_EQ(c.batch_spec, 2);
    CHECK_EQ(c.entry_spec, 2);
    CHECK_EQ(c.query.verify_k, c.entry_spec);
    /* And the sample is filed under the same key. */
    CHECK(c.observe.what == DraftCostDisposition::Recorded);
    CHECK_EQ(c.recorded.draft_k, 2);
    CHECK_EQ(c.recorded.verify_k, 2);
    CHECK_EQ(c.recorded.verify_tokens, 3);
    CHECK_EQ(c.recorded.history_tokens, 3);
    CHECK_EQ(c.recorded_rounds, 2);
    CHECK_EQ(c.recorded.verify_k, c.entry_spec);
}

/* An upshift: K2's two drafts are verified by the K3 cycle after it. The frozen window is two on
 * both paths (the cap never raises it), and the lookup, the batch and the sample name (3,2). */
TEST(an_upshift_prices_the_frozen_window_on_the_ahead_path) {
    Coupled c;   /* destroyed after `f`: the planner and the predictor point into `c` until fini() */
    Fixture f;
    upshift(f, /*ahead=*/true, c);

    CHECK_EQ(c.plan_k, 3);
    CHECK(c.source == DraftPlanSource::Cost);
    CHECK_EQ(c.query.draft_k, 3);
    CHECK_EQ(c.query.verify_k, 2);
    CHECK_EQ(c.query.verify_tokens, 3);
    CHECK_EQ(c.query.history_tokens, 3);
    CHECK(c.issued.valid);
    CHECK_EQ(c.issued.k, 3);
    CHECK_NEAR(c.issued.ms, 6.0, 1e-9);
    CHECK_EQ(c.batch_spec, 2);
    CHECK_EQ(c.entry_spec, 2);
    CHECK_EQ(c.query.verify_k, c.entry_spec);
    CHECK(c.observe.what == DraftCostDisposition::Recorded);
    CHECK_EQ(c.recorded.draft_k, 3);
    CHECK_EQ(c.recorded.verify_k, 2);
    CHECK_EQ(c.recorded.verify_tokens, 3);
    CHECK_EQ(c.recorded.history_tokens, 3);
    CHECK_EQ(c.recorded_rounds, 3);
    CHECK_EQ(c.recorded.verify_k, c.entry_spec);
}

TEST(an_upshift_prices_the_frozen_window_on_the_direct_path) {
    Coupled c;   /* destroyed after `f`: the planner and the predictor point into `c` until fini() */
    Fixture f;
    upshift(f, /*ahead=*/false, c);

    CHECK_EQ(c.plan_k, 3);
    CHECK(c.source == DraftPlanSource::Cost);
    CHECK_EQ(c.query.draft_k, 3);
    CHECK_EQ(c.query.verify_k, 2);
    CHECK_EQ(c.query.verify_tokens, 3);
    CHECK_EQ(c.query.history_tokens, 3);
    CHECK(c.issued.valid);
    CHECK_EQ(c.issued.k, 3);
    CHECK_NEAR(c.issued.ms, 6.0, 1e-9);
    CHECK_EQ(c.batch_spec, 2);
    CHECK_EQ(c.entry_spec, 2);
    CHECK_EQ(c.query.verify_k, c.entry_spec);
    CHECK(c.observe.what == DraftCostDisposition::Recorded);
    CHECK_EQ(c.recorded.draft_k, 3);
    CHECK_EQ(c.recorded.verify_k, 2);
    CHECK_EQ(c.recorded.verify_tokens, 3);
    CHECK_EQ(c.recorded.history_tokens, 3);
    CHECK_EQ(c.recorded_rounds, 3);
    CHECK_EQ(c.recorded.verify_k, c.entry_spec);
}

/* The same fall with the row's remaining output, not its price, capping the second plan: the
 * first cycle emits a token, so the second may draft one fewer than the frozen three. A resource
 * clamp is not a price choice, but it goes through the same key: query, batch and sample agree. */
TEST(a_remaining_output_bound_caps_the_same_window_on_both_paths) {
    {
        Coupled c;
        Fixture f;
        downshift(f, /*ahead=*/true, c, /*max_tokens=*/5);

        CHECK_EQ(c.plan_k, 2);
        CHECK(c.source == DraftPlanSource::Cost);
        CHECK_EQ(c.query.draft_k, 2);
        CHECK_EQ(c.query.verify_k, 3);
        CHECK_EQ(c.query.verify_tokens, 4);
        CHECK_EQ(c.issued.k, 2);
        CHECK_EQ(c.batch_spec, 3);
        CHECK_EQ(c.entry_spec, 3);
        CHECK(c.observe.what == DraftCostDisposition::Recorded);
        CHECK_EQ(c.recorded.draft_k, 2);
        CHECK_EQ(c.recorded.verify_k, 3);
        CHECK_EQ(c.recorded.verify_tokens, 4);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
        CHECK_EQ(c.recorded.verify_k, c.entry_spec);
    }
    {
        Coupled c;
        Fixture f;
        downshift(f, /*ahead=*/false, c, /*max_tokens=*/5);

        CHECK_EQ(c.plan_k, 2);
        CHECK(c.source == DraftPlanSource::Cost);
        CHECK_EQ(c.query.draft_k, 2);
        CHECK_EQ(c.query.verify_k, 2);
        CHECK_EQ(c.query.verify_tokens, 3);
        CHECK_EQ(c.issued.k, 2);
        CHECK_EQ(c.batch_spec, 2);
        CHECK_EQ(c.entry_spec, 2);
        CHECK(c.observe.what == DraftCostDisposition::Recorded);
        CHECK_EQ(c.recorded.draft_k, 2);
        CHECK_EQ(c.recorded.verify_k, 2);
        CHECK_EQ(c.recorded.verify_tokens, 3);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
        CHECK_EQ(c.recorded.verify_k, c.entry_spec);
    }
}

/* A real token/KV budget cut, not a remaining-output clamp: the pool is left with a single free
 * block, so `apply_draft_budget` scales the uncut six-times-three to two per row. The plan the
 * search would have issued (K3) is still weighed, the issued plan is the cut and says why, and the
 * cut plan's own re-price, the batch built for it and the recorded sample name the window its step
 * runs -- (2,3) ahead, (2,2) on the direct path. */
TEST(a_cost_budget_cut_prices_the_window_its_step_runs_on_both_paths) {
    for (bool ahead : { false, true }) {
        Coupled c;
        Fixture f;
        budget_cut(f, ahead, c);

        /* The uncut search weighed both depths and would have issued K3. */
        CHECK_EQ(c.uncut_k, 3);
        CHECK_NEAR(c.uncut_ms, 8.0, 1e-9);
        CHECK_EQ(c.tiers_priced, 2);
        /* The issued plan is the cut, and it says why on every row: the search's K2 candidate lost
         * on its own price, so only the budget can have lowered the assignment. */
        CHECK_EQ(c.plan_k, 2);
        CHECK(c.source == DraftPlanSource::Cost);
        CHECK_EQ(c.cut_rows, 6);
        /* Its own price is the cut window's, not the uncut plan's. */
        CHECK_EQ(c.query.draft_k, 2);
        CHECK_EQ(c.query.verify_k, ahead ? 3 : 2);
        CHECK_EQ(c.query.verify_tokens, ahead ? 24 : 18);
        CHECK(c.issued.valid);
        CHECK_EQ(c.issued.k, 2);
        CHECK_NEAR(c.issued.ms, ahead ? 9.0 : 8.0, 1e-9);
        /* The step that runs spends exactly that plan. The ahead step was built under the uncut
         * plan before the cycle opened; the direct one is built after it and re-capped. */
        CHECK_EQ(c.batch_spec, ahead ? 3 : 2);
        CHECK_EQ(c.entry_spec, c.query.verify_k);
        CHECK(c.observe.what == DraftCostDisposition::Recorded);
        CHECK_EQ(c.recorded.draft_k, 2);
        CHECK_EQ(c.recorded.verify_k, ahead ? 3 : 2);
        CHECK_EQ(c.recorded.verify_tokens, ahead ? 24 : 18);
        CHECK_EQ(c.recorded_rounds, 2);
        CHECK_EQ(c.query.verify_k, c.entry_spec);
        CHECK_EQ(c.recorded.verify_k, c.entry_spec);
    }
}

/* The ledger an empty planning point must leave alone. After a cycle the policy priced, an idle
 * step has no request rows to plan over; it is counted as `empty` and nothing else, so a snapshot
 * taken during an idle gap still shows what the last plan compared and asked for. Blanking the
 * priced-tier count there would make "the last priced plan weighed two depths" unreadable exactly
 * when the engine is between requests. */
TEST(an_empty_planning_point_leaves_the_priced_ledger_alone) {
    Coupled c;   /* destroyed after `f`: the planner's model points into `c.log` until fini() */
    Fixture f;
    downshift(f, /*ahead=*/true, c);

    const SchedMetrics before = f.sch.metrics();
    REQUIRE(before.draft_cost_tiers_priced > 0);
    CHECK_OK(f.sch.cancel(c.id));

    CHECK(f.sch.step() == nullptr);   /* nothing to plan over */
    const SchedMetrics after = f.sch.metrics();
    CHECK_EQ(after.draft_cost_cycles, before.draft_cost_cycles + 1);
    CHECK_EQ(after.draft_cost_empty, before.draft_cost_empty + 1);
    CHECK_EQ(after.draft_cost_priced, before.draft_cost_priced);
    CHECK_EQ(after.draft_cost_tiers_priced, before.draft_cost_tiers_priced);
    CHECK_EQ(after.draft_cost_last_k, before.draft_cost_last_k);
    CHECK_EQ(after.draft_cost_last_rows, before.draft_cost_last_rows);
    CHECK_EQ(after.draft_cost_last_source, before.draft_cost_last_source);
    CHECK_EQ(after.draft_cost_last_fallback, before.draft_cost_last_fallback);
    CHECK_EQ(after.draft_cost_last_replay, before.draft_cost_last_replay);
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

/* One bucket keyed the way this fixture runs: publish-then-verify at pipeline depth 1, one row,
 * the previous cycle's window still empty on the first cycle. */
std::string seam_bucket(int cls, int k, double ms) {
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
}

/* The A/C seam against the real model: a version-3 profile measured for this deployment, read by
 * A's table, sees the shape the planner prices.
 *
 * The cycle's replay state enters the view only as evidence (`DraftBatchView::replay`): a caller
 * that cannot establish it leaves the dimension Unknown, A's table refuses the shape, and the
 * cycle becomes the named baseline -- a fallback, never an invented number. A runtime that cannot
 * record a pass establishes `None`, which is what its profile was measured under, and the same
 * cycle is priced from the loaded buckets (the second fixture).
 *
 * The buckets are keyed the way this runtime runs: publish-then-verify at pipeline depth 1, so a
 * bucket's verify window is the previous cycle's distribution (nothing yet, on the first cycle)
 * and its draft distribution is the cycle's own. */
TEST(the_real_profile_prices_the_cycle_the_runtime_evidences_its_replay) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(8);
    pred.set(id, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

    const int cls = draft_ctx_class(8);
    auto bucket = [&](int k, double ms) { return seam_bucket(cls, k, ms); };
    /* 1 token / 2.0 ms, 3 / 1.3, 7 / 5.0: K=2 pays, K=6 does not. */
    const std::string profile =
        "{\"format\": \"rad-draft-profile/4\", \"fingerprint\": {\"model\": \"m\", \"quant\": \"q\", "
        "\"tp\": 1, \"tp_wire\": \"w\", \"gpus\": \"g\", \"kernels\": \"k\", \"kv_format\": \"bf16\", "
        "\"placement\": \"vram\", \"basis\": \"live\", \"arena\": \"none\", "
        "\"live_loan\": \"flex=0 lend=0\"}, \"execution\": \"rectangular\", \"buckets\": [" +
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

    /* The same profile and the same cycle, once the deployment evidences the state its passes run
     * under: a backend that cannot record runs every pass live, so the view names `None` and the
     * shape becomes the measured key the buckets hold. */
    Fixture g;
    g.build();
    g.cfg.n_spec = 6;
    CHECK_OK(g.start(DraftReplay::None));

    DraftVerifyShape none_shape;
    RealPrice none_real{ &table, &none_shape };
    DraftController::Params gp = g.sch.draft().params();
    gp.policy    = dp.policy;
    gp.predictor = DraftPredictor{ &pred, predict };
    gp.costs     = DraftCostModel{ &none_real, real_price };
    CHECK_OK(g.sch.draft().configure(gp));

    const uint64_t id2 = g.submit(8);
    pred.set(id2, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    g.prefill(id2);
    g.sch.find(id2)->sp.temp = 0.0f;
    REQUIRE(g.sch.step() != nullptr);
    const DraftPlan* pg = g.sch.draft().plan();
    REQUIRE(pg != nullptr);
    CHECK_EQ(none_shape.unknown, 0u);
    CHECK(none_shape.replay == DraftReplay::None);
    CHECK(draft_shape_priceable(none_shape));
    CHECK(pg->source == DraftPlanSource::Cost);
    CHECK(pg->fallback == DraftFallback::None);
    CHECK(pg->replay == DraftReplay::None);
    /* The ladder is 0/2/6 at one token per 2.0/1.3/5.0 ms: K=2 pays, so the priced plan drafts it
     * rather than the fallback depth. */
    CHECK_EQ(pg->k[0], 2);
    CHECK_EQ(pg->verify_k, 0);   /* still the first cycle: nothing published before it */
    CHECK_EQ(draft_plan_valid(*pg), RAD_OK);
}

/* The deepest offered plan is the coverage gate. A batch whose ladder is 0/2/6 against a profile
 * that priced only 0 and 2 does not take K2 because it happens to be priced: the deeper
 * alternative was not measured, and a shallower price is not evidence about it. The plan is the
 * named baseline, at the baseline depth -- not a cost decision. */
TEST(an_unpriced_deepest_tier_falls_back_instead_of_choosing_a_shallower_price) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    CHECK_OK(f.start(DraftReplay::None));

    Predictions pred;
    const uint64_t id = f.submit(8);
    pred.set(id, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

    const int cls = draft_ctx_class(8);
    const std::string profile =
        "{\"format\": \"rad-draft-profile/4\", \"fingerprint\": {\"model\": \"m\", \"quant\": \"q\", "
        "\"tp\": 1, \"tp_wire\": \"w\", \"gpus\": \"g\", \"kernels\": \"k\", \"kv_format\": \"bf16\", "
        "\"placement\": \"vram\", \"basis\": \"live\", \"arena\": \"none\", "
        "\"live_loan\": \"flex=0 lend=0\"}, \"execution\": \"rectangular\", \"buckets\": [" +
        seam_bucket(cls, 0, 2.0) + "," + seam_bucket(cls, 2, 1.3) + "]}";
    DraftCostProfile parsed;
    std::string err;
    CHECK_OK(draft_cost_profile_parse(profile, &parsed, &err));
    DraftCostRuntime rt;
    rt.fingerprint = parsed.fingerprint;
    rt.execution   = DraftCostExecution::Rectangular;
    DraftCostTable table;
    CHECK_OK(table.configure(parsed, rt));
    CHECK(table.active());

    RealPrice real{ &table, nullptr };
    DraftController::Params dp = f.sch.draft().params();
    dp.policy.mode     = DraftPolicyMode::Cost;
    dp.policy.tiers[0] = 0;
    dp.policy.tiers[1] = 2;
    dp.policy.tiers[2] = 6;
    dp.policy.n_tiers  = 3;
    dp.policy.fallback_k = 5;   /* distinguishable from every tier */
    dp.policy.hold     = 0;
    dp.policy.hyst     = 0.0;
    dp.predictor = DraftPredictor{ &pred, predict };
    dp.costs     = DraftCostModel{ &real, real_price };
    CHECK_OK(f.sch.draft().configure(dp));

    f.prefill(id);
    f.sch.find(id)->sp.temp = 0.0f;
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    CHECK(p->source == DraftPlanSource::Fallback);
    CHECK(p->fallback == DraftFallback::UnknownShape);
    CHECK_EQ(p->k[0], 5);
    CHECK_EQ(draft_plan_valid(*p), RAD_OK);
}

/* ================================================================== the production wiring */
/* The policy is turned on by Config alone -- no seam call -- so this is the path a deployment
 * takes: --draft-policy cost wires B's predictor and A's (empty) table into the controller, and
 * the cycle the scheduler issues is the plan the step reads.
 *
 * The observation is applied exactly once per completed cycle, and it carries the two windows the
 * contract names: the verify of the step paired with the plan (the frozen previous window) and the
 * drafting published under the plan (which the next step verifies). */
TEST(the_cost_policy_is_wired_through_config_and_observes_each_cycle_exactly_once) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);          /* the baseline a Cost deployment is admitted with */
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);

    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };

    /* Cycle A: nothing can be priced without a profile, so the named fallback is issued at the
     * configured depth -- and the predictor ran to price it. */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* pa = f.sch.draft().plan();
    REQUIRE(pa != nullptr);
    CHECK(pa->source == DraftPlanSource::Fallback);
    CHECK_EQ(planned(f.sch, id), 6);
    const int32_t slot = entry_of(f.sch, id)->slot;
    const DraftPredictState* st = f.sch.draft_predict_state(slot);
    REQUIRE(st != nullptr);
    CHECK_EQ(st->req_id, id);          /* predict() started this slot for this request */
    CHECK_EQ(st->n[0], 0);             /* predicted, nothing learned yet */

    /* The step paired with A verifies nothing (A is the first cycle), and its commit is A's verify
     * half. The publication that follows is A's drafting half. */
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    CHECK_EQ(f.sch.draft_cycles_done(), 0);   /* published is not yet a completed cycle */

    /* Cycle B: opening it delivers A -- once -- and the step verifies the six A published. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 1);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 0);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 6);
    CHECK_EQ(f.sch.draft_predict_state(slot)->n[0], 0);  /* A measured no positions */
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return 2; }));   /* 2 of 6 accepted */
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    CHECK_EQ(f.sch.draft_cycles_done(), 1);   /* B's verify alone is not a completed cycle */

    /* Cycle C: opening it delivers B, whose labels are the three positions the verify produced
     * (two accepted, then the first rejection). */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 2);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 0);
    st = f.sch.draft_predict_state(slot);
    REQUIRE(st != nullptr);
    CHECK_EQ(st->n[0], 1);
    CHECK_EQ(st->n[1], 1);
    CHECK_EQ(st->n[2], 1);
    CHECK_EQ(st->n[3], 0);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return 2; }));
    CHECK_OK(f.sch.set_draft(id, drafts, 6));

    /* Cycle D: one more delivery, one more label per position -- never two for one cycle. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 3);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 0);
    st = f.sch.draft_predict_state(slot);
    REQUIRE(st != nullptr);
    CHECK_EQ(st->n[0], 2);
    CHECK_EQ(st->n[1], 2);
    CHECK_EQ(st->n[2], 2);
    CHECK_EQ(st->n[3], 0);
}

/* An idle step loop opens no next cycle, but it still calls step(), and draft_cost_cycle() closes
 * the open record at the start of every step -- before the loop knows there is no batch, and before
 * drain_completed() recycles slots. This drives that real flow: a fully measured cycle, the request
 * leaving, the idle step() delivering it, then the slot being reaped. A second idle step must not
 * count it again. */
TEST(an_idle_engine_delivers_its_last_measured_cycle_before_the_slot_is_recycled) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);
    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };

    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    CHECK_EQ(f.sch.draft_cycles_done(), 0);   /* measured; no step has closed it yet */
    const int64_t dropped0 = f.sch.draft_cycles_dropped();

    /* The request leaves with the cycle still open, then the engine's idle step runs: its own
     * draft_cost_cycle() retires the record, before the slot is reaped. */
    CHECK_OK(f.sch.cancel(id));
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 1);
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0);

    CHECK_OK(f.sch.reap(id));                 /* the slot recycle, after the delivery */
    CHECK_EQ(f.sch.draft_cycles_done(), 1);
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0);

    CHECK(f.sch.step() == nullptr);           /* retired: a later idle step adds nothing */
    CHECK_EQ(f.sch.draft_cycles_done(), 1);
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0);
}

TEST(an_idle_engine_counts_a_cycle_no_later_open_can_finish_as_discarded) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);
    const int64_t dropped0 = f.sch.draft_cycles_dropped();

    /* Opened, then the request leaves before the verify commits: no later open can finish it. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_OK(f.sch.cancel(id));
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 0);
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0 + 1);

    CHECK(f.sch.step() == nullptr);           /* retired: not counted twice */
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0 + 1);
}

/* The other place a cycle opens: with a step issued ahead, promote_next() makes it current and its
 * draft_cycle() is the head that closes the record the previous cycle measured. This is the
 * pipelined path, where most cycles of a serving run open -- the idle tests above cover the step()
 * head only. One delivery per record: issuing the next step ahead opens nothing, a second
 * promote_next() with nothing ahead returns at its guard, and the promoted step's own record is
 * left open for the next step() head rather than counted at the promote. */
TEST(a_promoted_step_delivers_the_measured_cycle_once_and_only_once) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;
    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };

    /* Cycle A opens at step() and is measured by its commit and its published proposals. */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    CHECK_EQ(f.sch.draft_cycles_done(), 0);   /* measured; no head has closed it yet */
    const int64_t dropped0 = f.sch.draft_cycles_dropped();

    /* Cycle B opens at step() and delivers A on the way in -- the step() head, already covered. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 1);

    /* B is measured in turn: its proposals are published under it, then the next step is issued
     * ahead. Building that step opens no cycle, so B stays open and undelivered. */
    f.sch.expect_draft(id, 6);
    CHECK_OK(f.sch.set_draft(id, drafts, 6));
    REQUIRE(f.sch.step_next() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 1);

    /* Its verify commits, and promote_next() makes the issued step current: B is delivered here. */
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    f.sch.promote_next();
    CHECK_EQ(f.sch.draft_cycles_done(), 2);
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0);

    /* Idempotent: nothing is open ahead now, so the guard returns before any close. */
    f.sch.promote_next();
    CHECK_EQ(f.sch.draft_cycles_done(), 2);

    /* The promoted step ran B's drafts, so its own cycle -- opened at the promote head -- is a
     * measurement too: its verify commit fills the row, and the next step() head delivers it once. */
    f.commit_plain(11);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 3);
    CHECK_EQ(f.sch.draft_cycles_dropped(), dropped0);
}

/* Preemption by recompute keeps the slot and its generation (only add() bumps it), so B's per-slot
 * estimate is the same request's after the resume. The cycle whose row left before its verify,
 * though, never completes: it is counted, not delivered with a hole. */
TEST(a_preempted_slot_keeps_its_history_and_its_cycle_is_dropped) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t a = f.submit(64, 512, /*priority=*/0);
    const uint64_t b = f.submit(64, 512, /*priority=*/5);
    REQUIRE(a && b);

    int32_t drafts[6] = { 21, 22, 23, 24, 25, 26 };
    int32_t slot = -1;
    uint32_t slot_gen = 0;
    uint16_t n_before[RAD_SCHED_MAX_SPEC] = {};
    int64_t dropped_before = 0;
    bool preempted = false;

    /* Decode until the pool is exhausted. Every step accepts the whole window, so the estimate
     * accumulates one label per position per delivered cycle -- and every few steps a new block
     * is needed, until `a` (the lower priority) is the victim. */
    for (int i = 0; i < 200 && !preempted; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        const StepEntry* ea = entry_of(f.sch, a);
        if (ea) slot = ea->slot;
        const DraftPredictState* st =
            slot < 0 ? nullptr : f.sch.draft_predict_state(slot);
        /* The slot is started by the first cycle that has this row as a decode row; a prefill
         * step plans no row and predicts nothing. */
        if (st && st->req_id == a) {
            if (slot_gen == 0) slot_gen = st->slot_gen;
            CHECK_EQ(st->slot_gen, slot_gen);
            for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) n_before[j] = st->n[j];
        }
        dropped_before = f.sch.draft_cycles_dropped();

        CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
        for (uint64_t id : { a, b })
            if (f.sch.find(id)->state == ReqState::Running)
                CHECK_OK(f.sch.set_draft(id, drafts, 6));

        preempted = f.sch.metrics().preemptions > 0;
        if (preempted) {
            const DraftPredictState* after =
                slot < 0 ? nullptr : f.sch.draft_predict_state(slot);
            REQUIRE(after != nullptr);
            CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
            CHECK_EQ(after->req_id, a);
            CHECK_EQ(after->slot_gen, slot_gen);
            bool learned = false;
            for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
                CHECK_EQ(after->n[j], n_before[j]);
                learned = learned || n_before[j] > 0;
            }
            CHECK(learned);   /* the history that survived is a real one, not an empty table */

            /* The interrupted cycle is retired at the next open as incomplete. */
            REQUIRE(f.sch.step() != nullptr);
            CHECK_EQ(f.sch.draft_cycles_dropped(), dropped_before + 1);
        }
    }
    CHECK(preempted);
}

/* Review 2, finding 2: the labels a cycle's verify reports measure the window it consumed -- the
 * previous plan's publication or this step's own lookup fallback -- never the plan the row is
 * named after. Both directions are exercised in one run: cycle A's verify is a lookup while A
 * publishes the head's proposals; cycle B's verify measures A's head window while B itself
 * publishes nothing (its lookup fills). */
TEST(the_labels_carry_the_provenance_of_the_verified_window_not_the_plan) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit_pattern(2);   /* enough repetition for prompt lookup */
    f.prefill(id, /*token=*/1);                /* the tail keeps repeating after the prompt */
    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };

    /* Cycle A: no head proposal, so the step fills from prompt lookup. */
    REQUIRE(f.sch.step() != nullptr);
    const StepEntry* ea = entry_of(f.sch, id);
    REQUIRE(ea != nullptr);
    const int32_t slot = ea->slot;
    CHECK(ea->ngram);
    const int32_t kA = ea->n_spec;
    REQUIRE(kA > 0);
    f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; });   /* all accepted */
    f.sch.step_finished();

    /* A publishes the head's proposals for the next step. When A is delivered, its own origin is
     * Head, but the labels it measured were the lookup's: B's estimate must not move. */
    CHECK_OK(f.sch.set_draft(id, six, kA));
    REQUIRE(f.sch.step() != nullptr);            /* opens B, delivers A */
    CHECK_EQ(f.sch.draft_cycles_done(), 1);
    const DraftPredictState* st = f.sch.draft_predict_state(slot);
    REQUIRE(st != nullptr);
    CHECK_EQ(st->req_id, id);
    CHECK_EQ(st->n[0], 0);
    CHECK_EQ(f.sch.metrics().draft_obs_applied, 0);
    CHECK_EQ(f.sch.metrics().draft_obs_skipped, 1);

    /* B verifies A's published head window but publishes nothing itself. Its delivery must teach:
     * the old gate read B's own (empty) origin and would have dropped the head's labels. */
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return 2; }));
    REQUIRE(f.sch.step() != nullptr);            /* opens C, delivers B */
    CHECK_EQ(f.sch.draft_cycles_done(), 2);
    st = f.sch.draft_predict_state(slot);
    REQUIRE(st != nullptr);
    CHECK_EQ(st->n[0], 1);
    CHECK_EQ(st->n[1], 1);
    CHECK_EQ(st->n[2], 1);      /* accepted 2, then the first rejection */
    CHECK_EQ(st->n[3], 0);
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_obs_attempted, 2);
    CHECK_EQ(m.draft_obs_applied, 1);
    CHECK_EQ(m.draft_obs_skipped, 1);
    CHECK_EQ(m.draft_obs_discarded, 0);
}

/* Review 2, finding 4: the last cycle has no next open to be delivered at. Shutdown must deliver
 * what is already complete, before the request identities are gone, and only once. */
TEST(shutdown_delivers_the_last_complete_cycle_and_not_twice) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);
    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };

    REQUIRE(f.sch.step() != nullptr);            /* cycle A */
    f.commit_plain(11);                          /* A's verify: nothing drafted */
    f.sch.step_finished();
    CHECK_OK(f.sch.set_draft(id, six, 6));       /* A's publication: six, verified by B */
    const int64_t done0 = f.sch.draft_cycles_done();
    REQUIRE(f.sch.step() != nullptr);            /* opens B, delivers A */
    CHECK_EQ(f.sch.draft_cycles_done(), done0 + 1);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return 2; }));   /* B's verify: reported */
    f.sch.step_finished();

    /* No further open. B is complete and must reach the predictor now. */
    f.sch.shutdown("test done", 0);
    CHECK_EQ(f.sch.draft_cycles_done(), done0 + 2);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 0);
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_obs_attempted, 2);
    CHECK_EQ(m.draft_obs_applied, 1);   /* B measured A's head window */
    CHECK_EQ(m.draft_obs_skipped, 1);   /* A measured nothing */
    CHECK_EQ(m.draft_obs_discarded, 0);

    /* A second shutdown (or fini) must not apply B again. */
    f.sch.shutdown("again", 0);
    CHECK_EQ(f.sch.draft_cycles_done(), done0 + 2);
    CHECK_EQ(f.sch.metrics().draft_obs_attempted, 2);
}

/* The other half of the same rule: a cycle whose row never reported is not a smaller observation.
 * It is dropped, visibly, and never delivered with a hole. */
TEST(shutdown_counts_an_incomplete_last_cycle_as_discarded) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);

    REQUIRE(f.sch.step() != nullptr);   /* cycle A opens with this decode row pending */
    f.sch.shutdown("test stop", 0);     /* the row will never report */
    CHECK_EQ(f.sch.draft_cycles_done(), 0);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 1);
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_obs_attempted, 0);
    CHECK_EQ(m.draft_obs_discarded, 1);
}

/* docs/ADAPTIVE-MTP.md: cost is serial-only and needs a ceiling. The refusal is by name, at init,
 * where the container's drafter is known -- never a silent fall back to legacy. */
TEST(the_cost_policy_is_refused_by_name_on_a_deployment_it_cannot_price) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_BLOCK;   /* a block drafter proposes in one pass */
    f.prog.drafter.depth = 6;
    CHECK_EQ(f.start(), RAD_E_UNSUPPORTED);

    f.prog.drafter.kind = RAD_DRAFT_SERIAL;
    f.cfg.n_spec = 0;                          /* no speculation, nothing to choose */
    CHECK_EQ(f.start(), RAD_E_UNSUPPORTED);

    f.cfg.n_spec = 6;
    /* A serial head and a ceiling are the shape, not a baseline: without a profile admitted for
     * this runtime Cost is still refused, and the ceiling alone is not one. */
    const std::string refused =
        stderr_of([&] { CHECK_EQ(f.start(), RAD_E_UNSUPPORTED); });
    CHECK(has(refused, "no admissible cost baseline"));

    /* A matching profile with no measured bucket is the reader's proof that it found the runtime,
     * not the deployment's admission that Cost may price nothing: the profile alone is refused by
     * name, so a forgotten admission cannot look like a loaded baseline. */
    {
        DraftCostProfile p;
        p.fingerprint.model     = "test";
        p.fingerprint.quant     = "q";
        p.fingerprint.tp_wire   = "w";
        p.fingerprint.gpus      = "g";
        p.fingerprint.kernels   = "k";
        p.fingerprint.kv_format = "bf16";
        p.fingerprint.placement = "all_vram";
        p.fingerprint.tp        = 1;
        DraftCostRuntime rt;
        rt.fingerprint = p.fingerprint;
        CHECK_OK(f.sch.draft_costs().configure(p, rt));
        REQUIRE(f.sch.draft_costs().active());
        REQUIRE(f.sch.draft_costs().buckets().empty());
        CHECK(!f.sch.draft_costs().fallback_admitted());
        const std::string empty =
            stderr_of([&] { CHECK_EQ(f.start(), RAD_E_UNSUPPORTED); });
        CHECK(has(empty, "no measured bucket"));

        /* A non-empty bucket list is not a baseline either: the gate asks the same question
         * estimate() does (draft_bucket_usable), so a profile of diagnostic buckets and one
         * carrying no positive mean are still "no measured bucket". */
        DraftCostBucket b;
        b.shape.n_rows        = 1;
        b.shape.verify_k      = 1;
        b.shape.verify_tokens = 2;
        b.shape.rows[1]       = 1;
        b.shape.draft_k       = 1;
        b.shape.draft_rows[1] = 1;
        b.shape.drafter       = DraftDrafter::Serial;
        b.flags               = kDraftSampleWarmup;
        b.rounds              = 1;
        b.draft_round_rows[1] = 1;
        b.draft_rows_total    = 1;
        b.mean_ms             = 12.0;
        b.unc_ms              = 1.0;
        b.n                   = 3;
        p.buckets.assign(1, b);
        CHECK_OK(f.sch.draft_costs().configure(p, rt));
        REQUIRE(f.sch.draft_costs().buckets().size() == 1);
        CHECK(!f.sch.draft_costs().has_usable_bucket());
        const std::string warm =
            stderr_of([&] { CHECK_EQ(f.start(), RAD_E_UNSUPPORTED); });
        CHECK(has(warm, "no measured bucket"));

        p.buckets[0].flags = 0;
        p.buckets[0].n = 0;                   /* no samples behind the mean */
        CHECK_OK(f.sch.draft_costs().configure(p, rt));
        CHECK(!f.sch.draft_costs().has_usable_bucket());
        CHECK_EQ(f.start(), RAD_E_UNSUPPORTED);

        p.buckets[0].n = 3;
        p.buckets[0].mean_ms = 0.0;           /* a missing timing, not a zero-cost bucket */
        CHECK_OK(f.sch.draft_costs().configure(p, rt));
        CHECK(!f.sch.draft_costs().has_usable_bucket());
        CHECK_EQ(f.start(), RAD_E_UNSUPPORTED);

        /* clear() resets the admission with the rest of the table, so an earlier deployment's
         * admission cannot reappear under a new profile. */
        f.sch.draft_costs().admit_fallback();
        CHECK(f.sch.draft_costs().fallback_admitted());
        f.sch.draft_costs().clear();
        CHECK(!f.sch.draft_costs().fallback_admitted());
    }

    activate_costs(f);
    CHECK_OK(f.start());
}

/* An explicit ladder below the executable depth is a valid reload when the profile prices its
 * top; only an unpriced top is a coverage gap. Then the named baseline would run at exec_depth --
 * a depth the ladder never offers -- and its bucket could never be queried by this ladder, so the
 * run would stay in the fallback. Init accepts the ladder; the cycle is refused, and the run stops
 * on the existing fail-closed latch (draft_cost_aborted_ -> Engine::fail_loop) instead of drafting
 * at the ceiling without a plan. Nothing is learnt into the ladder, and the list and the baseline
 * are not changed behind the operator's back. */
TEST(an_unpriced_explicit_ladder_below_the_executable_depth_stops_the_run) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    f.cfg.draft_tiers_set = true;
    f.cfg.draft_n_tiers   = 2;
    f.cfg.draft_tiers[0]  = 0;
    f.cfg.draft_tiers[1]  = 2;
    activate_costs(f);                    /* live, empty, fallback admitted */

    CHECK_OK(f.start(DraftReplay::None));  /* init accepts it; the cycle diagnoses the gap */
    const uint64_t id = f.submit(160);    /* 64 tokens of room: this row would run at K6 */
    f.prefill(id);
    const Request* r = f.sch.find(id);
    REQUIRE(r != nullptr);
    CHECK_EQ(r->n_computed, (int64_t)r->prompt.size());   /* decoding, so the cycle sees it */
    const SchedMetrics before = f.sch.metrics();

    const std::string gap = stderr_of([&] { CHECK(f.sch.step() == nullptr); });
    CHECK(has(gap, "deepest offered tier K2"));
    CHECK(has(gap, "named baseline would run at K6"));
    CHECK(f.sch.draft_cost_aborted());                   /* the run is on the fail-closed path */
    CHECK(f.sch.draft().plan() == nullptr);              /* no plan: not a ceiling fallback */
    CHECK_EQ(f.sch.draft_cycles_done(), 0);              /* no ring record was completed */
    CHECK(f.sch.draft_costs().buckets().empty());        /* nothing learnt at K6 */

    /* The refused cycle is in the ledger, as a cycle without a plan -- not as a fallback, and not
     * as a plan at K0. The cycle before it (the prefill's own) is untouched by the refusal. */
    const SchedMetrics ledger = f.sch.metrics();
    CHECK_EQ(ledger.draft_cost_cycles, before.draft_cost_cycles + 1);
    CHECK_EQ(ledger.draft_cost_no_plan, before.draft_cost_no_plan + 1);
    CHECK_EQ(ledger.draft_cost_priced, before.draft_cost_priced);
    CHECK_EQ(ledger.draft_cost_fallback, before.draft_cost_fallback);
    CHECK_EQ(ledger.draft_cost_last_k, before.draft_cost_last_k);
    CHECK_EQ(ledger.draft_cost_last_source, before.draft_cost_last_source);
    CHECK_EQ(ledger.draft_cost_last_fallback, before.draft_cost_last_fallback);

    /* The latch holds: no later step plans a batch, so nothing drafts at the ceiling. The ledger
     * keeps counting what the planning site keeps opening -- a refused cycle is an opened cycle --
     * and every one of them ends in `no_plan`. */
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 0);
    const SchedMetrics again = f.sch.metrics();
    CHECK_EQ(again.draft_cost_cycles, ledger.draft_cost_cycles + 1);
    CHECK_EQ(again.draft_cost_no_plan, ledger.draft_cost_no_plan + 1);
}

/* The gap belongs to the batch, not to one row: if any row's own remaining budget would run past
 * the ladder's top, the deepest offered plan cannot be priced for the cycle's shape and the whole
 * cycle is refused -- including a row whose own cap is under the top. Drafting the clamped row
 * under a plan-less cycle would be the silent ceiling fallback the admission rule refuses. */
TEST(a_mixed_batch_with_one_row_past_the_ladder_is_refused_whole) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    f.cfg.draft_tiers_set = true;
    f.cfg.draft_n_tiers   = 2;
    f.cfg.draft_tiers[0]  = 0;
    f.cfg.draft_tiers[1]  = 2;
    activate_costs(f);
    CHECK_OK(f.start(DraftReplay::None));

    const uint64_t deep    = f.submit(160);        /* 63 tokens of room: cap K6 */
    const uint64_t clamped = f.submit(170, 3);     /* two left after its first token: cap K1 */
    f.prefill(deep);
    f.prefill(clamped);
    const Request* rd = f.sch.find(deep);
    const Request* rc = f.sch.find(clamped);
    REQUIRE(rd != nullptr);
    REQUIRE(rc != nullptr);
    CHECK_EQ(rd->n_computed, (int64_t)rd->prompt.size());
    CHECK_EQ(rc->n_computed, (int64_t)rc->prompt.size());

    const std::string gap = stderr_of([&] { CHECK(f.sch.step() == nullptr); });
    CHECK(has(gap, "deepest offered tier K2"));
    CHECK(has(gap, "named baseline would run at K6"));
    CHECK(f.sch.draft_cost_aborted());
    CHECK(f.sch.draft().plan() == nullptr);
    CHECK_EQ(f.sch.draft_cycles_done(), 0);
    CHECK(f.sch.draft_costs().buckets().empty());
}

/* Rows whose own budgets clamp both the offered top and the named baseline to the same k are not
 * a gap: the fallback runs the shape the ladder offered, so its bucket is queryable by it. The
 * plan is issued at that clamped k and the step drafts -- the ladder below exec_depth is only
 * refused where it would have run deeper than it offers. */
TEST(identically_clamped_rows_take_the_named_baseline_and_still_draft) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    f.cfg.draft_tiers_set = true;
    f.cfg.draft_n_tiers   = 2;
    f.cfg.draft_tiers[0]  = 0;
    f.cfg.draft_tiers[1]  = 2;
    activate_costs(f);
    CHECK_OK(f.start(DraftReplay::None));

    /* Two tokens of room each after the prefill's own token: min(ceiling, room - 1) is K1 for
     * both, under the ladder's top. */
    const uint64_t a = f.submit(160, 3);
    const uint64_t b = f.submit(170, 3);
    f.prefill(a);
    f.prefill(b);

    REQUIRE(f.sch.step() != nullptr);
    CHECK(!f.sch.draft_cost_aborted());
    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    CHECK(p->source == DraftPlanSource::Fallback);   /* the named baseline, at the clamped k */
    CHECK(p->fallback == DraftFallback::UnknownShape);
    CHECK_EQ(p->max_k, 1);                           /* not the ceiling, not the ladder's top */
    CHECK_EQ(planned(f.sch, a), 1);
    CHECK_EQ(planned(f.sch, b), 1);
    /* The width the scheduler drafts under the plan: a two-token proposal is cut to the plan's
     * K1, not to the ceiling. */
    int32_t two[2] = { 31, 32 };
    CHECK_OK(f.sch.set_draft(a, two, 2));
    CHECK_OK(f.sch.set_draft(b, two, 2));
    CHECK_EQ(f.sch.find(a)->n_draft, 1);
    CHECK_EQ(f.sch.find(b)->n_draft, 1);
    f.commit_plain(11);
    CHECK(!f.sch.draft_cost_aborted());
}

/* The same ladder is admitted when the executable depth really is 2: the ceiling may be deeper,
 * the drafter caps it, and the deepest offered plan is the depth the baseline measures. */
TEST(an_explicit_ladder_at_the_executable_depth_is_admitted) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;                     /* the ceiling alone does not decide the depth */
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 2;             /* the head can publish two rounds */
    f.cfg.draft_tiers_set = true;
    f.cfg.draft_n_tiers   = 2;
    f.cfg.draft_tiers[0]  = 0;
    f.cfg.draft_tiers[1]  = 2;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    CHECK(p->source == DraftPlanSource::Fallback);   /* the empty baseline is the named fallback */
    CHECK_EQ(p->k[0], 2);                            /* at the ladder's top / executable depth */
    CHECK_EQ(draft_plan_valid(*p), RAD_OK);
}

/* The feedback R depends on: the named baseline's measurement is a price for the deepest offered
 * plan, so a later cycle can choose it. A live table starts with no bucket. Cycle A verifies
 * nothing and falls back at six; its bucket is a first-cycle shape (verify_k 0). Cycle B prices
 * A's published window, does not find that shape either, falls back once more and measures the
 * exact shape it executed; cycle C prices it and issues a cost plan at the deepest tier. */
TEST(a_fallback_cycle_measures_the_deepest_offered_plan_for_the_next_cycle) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);                    /* live, empty, fallback admitted */
    CHECK_OK(f.start(DraftReplay::None));

    const uint64_t id = f.submit(160);
    f.prefill(id);
    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };

    /* Cycle A: nothing priced, so the fallback runs at six; the step verifies nothing. */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* a = f.sch.draft().plan();
    REQUIRE(a != nullptr);
    CHECK(a->source == DraftPlanSource::Fallback);
    CHECK_EQ(a->k[0], 6);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, six, 6));
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    f.sch.step_finished(live);

    /* Cycle B: A is delivered first. Its bucket verified nothing; this cycle prices a shape that
     * verifies A's six, so it is not that bucket and the named baseline runs again. */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* b = f.sch.draft().plan();
    REQUIRE(b != nullptr);
    CHECK(b->source == DraftPlanSource::Fallback);
    CHECK_EQ(b->k[0], 6);
    CHECK_EQ(b->verify_k, 6);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    CHECK_OK(f.sch.set_draft(id, six, 6));
    f.sch.step_finished(live);

    /* Cycle C: B's observation closed on this open and priced the shape the deepest tier asks
     * for -- the plan is now a cost decision at six, not the baseline. */
    REQUIRE(f.sch.step() != nullptr);
    const DraftPlan* c = f.sch.draft().plan();
    REQUIRE(c != nullptr);
    CHECK(c->source == DraftPlanSource::Cost);
    CHECK(c->fallback == DraftFallback::None);
    CHECK_EQ(c->k[0], 6);
    CHECK_EQ(c->verify_k, 6);
}

/* The replay a cycle is priced under is the last bracket's *observation*, not a capability claim,
 * and even then it is only a prognosis for the next key. A cold recording deployment cannot know
 * what its first pass will do, so the first cycle is `Unknown`; every later one declares what the
 * previous bracket's ranks were seen to run (`Engine::step_evidence`, merged Record > Audit >
 * Play > None). A key or level change re-records the pass, a bracket no rank could evidence
 * leaves the candidate alone, and a non-recording deployment's cold answer is already its steady
 * one. The seam model here prices every shape, which is what makes the declared replay visible in
 * the shape A is asked about; the real table's bucket rules are draft_cost_test.cpp's. */
TEST(the_next_cycle_declares_the_last_brackets_observed_replay) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    CHECK_OK(f.start());                    /* the harness makes no claim about its tape */

    Predictions pred;
    const uint64_t id = f.submit(8);
    pred.set(id, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    ShapeLog log;
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
    const int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    const SchedMetrics base = f.sch.metrics();   /* the prefill's own plan opened a cycle */

    /* Read what the plan declares, before running anything under it. */
    int k = 0;
    auto open = [&]() -> DraftReplay {
        CHECK(f.sch.step() != nullptr);
        const DraftPlan* p = f.sch.draft().plan();
        if (!p) return DraftReplay::Unknown;
        CHECK_EQ(draft_plan_valid(*p), RAD_OK);
        k = p->k[0];
        return p->replay;
    };
    /* Run the bracket it names and report what the ranks saw in it. */
    auto finish = [&](DraftReplay seen, bool known = true) {
        f.commit_plain(11);
        CHECK_OK(f.sch.set_draft(id, six, k));
        DraftStepEvidence ev;
        ev.replay_known = known;
        ev.replay       = seen;
        f.sch.step_finished(ev);
    };

    /* Cold: nothing has run, so the plan declares the capability answer and the shape A is asked
     * about carries the unknown dimension rather than a replay nobody established. */
    CHECK(open() == DraftReplay::Unknown);
    CHECK_EQ(log.last.unknown, (uint32_t)kDraftShapeReplayUnknown);
    finish(DraftReplay::None);

    /* The first bracket reported an unkeyed pass; the next plan declares it as its candidate. */
    CHECK(open() == DraftReplay::None);
    CHECK_EQ(log.last.unknown, 0u);
    CHECK(log.last.replay == DraftReplay::None);
    finish(DraftReplay::Record);

    /* A key or level change re-records: the pass this bracket ran is not the one the last did. */
    CHECK(open() == DraftReplay::Record);
    CHECK(log.last.replay == DraftReplay::Record);
    finish(DraftReplay::Play);

    /* The tape has warmed up and the pass plays. */
    CHECK(open() == DraftReplay::Play);
    CHECK(log.last.replay == DraftReplay::Play);
    finish(DraftReplay::Audit);

    /* A bracket no rank evidenced (this rank never ran the step) says nothing about the next
     * key: the candidate keeps the last observation instead of inventing `Unknown`. */
    CHECK(open() == DraftReplay::Audit);
    finish(DraftReplay::Unknown, false);
    CHECK(open() == DraftReplay::Audit);

    /* The plan ledger tells "priced" from "fell back" and names the candidate it used. */
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_cycles, base.draft_cost_cycles + 6);
    CHECK_EQ(m.draft_cost_priced, base.draft_cost_priced + 6);   /* this model prices every shape */
    CHECK_EQ(m.draft_cost_fallback, base.draft_cost_fallback);
    CHECK_EQ(m.draft_cost_no_plan, base.draft_cost_no_plan);
    CHECK_EQ(m.draft_cost_last_k, k);
    CHECK_EQ(m.draft_cost_last_rows, 1);
    CHECK_EQ(m.draft_cost_last_replay, (int64_t)DraftReplay::Audit);
    CHECK_EQ(m.draft_cost_last_source, (int64_t)DraftPlanSource::Cost);
    CHECK_EQ(m.draft_cost_last_fallback, (int64_t)DraftFallback::None);
    /* Every opened cycle has exactly one named end, so none of them can go missing. */
    CHECK_EQ(m.draft_cost_priced + m.draft_cost_fallback + m.draft_calib_forced +
                 m.draft_cost_empty + m.draft_cost_no_plan,
             m.draft_cost_cycles);
    CHECK_EQ(m.draft_ref_skipped, (int64_t)0);   /* a live table learns; nothing is refused */
}

/* A reference table is the run's yardstick, not its notebook: the cycles the new candidate prices
 * are still only compared against it. The observed replay reaches the plan -- and through it the
 * shape A is asked about -- and never observe(): the bucket is exactly what was loaded, and each
 * valid observation is counted apart instead of folded in. */
TEST(an_observed_replay_never_writes_into_a_reference_table) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f, /*priced=*/true, DraftCostBasis::NoLoan);
    CHECK_OK(f.start(DraftReplay::None));
    REQUIRE(f.sch.draft_costs().reference());

    const DraftCostBucket before = f.sch.draft_costs().buckets()[0];

    const uint64_t id = f.submit(160);
    f.prefill(id);
    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    for (DraftReplay seen : { DraftReplay::None, DraftReplay::Play, DraftReplay::Audit }) {
        REQUIRE(f.sch.step() != nullptr);
        const DraftPlan* p = f.sch.draft().plan();
        REQUIRE(p != nullptr);
        f.commit_plain(11);
        CHECK_OK(f.sch.set_draft(id, six, p->k[0]));
        DraftStepEvidence ev;
        ev.replay_known = true;
        ev.replay       = seen;
        f.sch.step_finished(ev);
    }

    /* The last bracket is delivered when the next cycle opens, so open one more. */
    REQUIRE(f.sch.step() != nullptr);

    /* The file's bucket is what it measured: same mean, same width, same count. */
    const std::vector<DraftCostBucket>& after = f.sch.draft_costs().buckets();
    REQUIRE_EQ(after.size(), (size_t)1);
    CHECK(after[0].mean_ms == before.mean_ms);
    CHECK(after[0].unc_ms == before.unc_ms);
    CHECK_EQ(after[0].n, before.n);
    CHECK_EQ(f.sch.draft_costs().reference_skipped(), 3);
    /* The same count is what /stats publishes: the refusal is visible, not internal. */
    CHECK_EQ(f.sch.metrics().draft_ref_skipped, (int64_t)3);
    /* And the last sample's own outcome is the per-cycle record of that discard, with no bucket
     * named: a cumulative count cannot say which cycle the reference refused. */
    const DraftCostObserve& ob = f.sch.draft_costs().last_observe();
    CHECK(ob.what == DraftCostDisposition::ReferenceSkipped);
    CHECK_EQ(ob.bucket, -1);
}

/* The other side of the same gate: one bucket estimate() would price from (no disturbing flag, a
 * positive sample count, a finite positive mean) is a measured baseline on its own -- Cost starts
 * with no fallback admission, and the table's own predicate says so. */
TEST(a_usable_bucket_admits_the_cost_policy_without_a_fallback_admission) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;

    DraftCostProfile p;
    p.fingerprint.model     = "test";
    p.fingerprint.quant     = "q";
    p.fingerprint.tp_wire   = "w";
    p.fingerprint.gpus      = "g";
    p.fingerprint.kernels   = "k";
    p.fingerprint.kv_format = "bf16";
    p.fingerprint.placement = "all_vram";
    p.fingerprint.tp        = 1;
    p.fingerprint.basis     = DraftCostBasis::Live;
    p.fingerprint.arena     = "none";
    p.fingerprint.live_loan = "flex=0 lend=0";
    DraftCostBucket b;
    b.shape.n_rows        = 1;
    b.shape.verify_k      = 1;
    b.shape.verify_tokens = 2;
    b.shape.rows[1]       = 1;
    b.shape.draft_k       = 1;
    b.shape.draft_rows[1] = 1;
    b.shape.drafter       = DraftDrafter::Serial;
    b.rounds              = 1;
    b.draft_round_rows[1] = 1;
    b.draft_rows_total    = 1;
    b.mean_ms             = 12.0;
    b.unc_ms              = 1.0;
    b.n                   = 3;
    p.buckets.assign(1, b);

    DraftCostRuntime rt;
    rt.fingerprint = p.fingerprint;
    CHECK_OK(f.sch.draft_costs().configure(p, rt));
    REQUIRE(f.sch.draft_costs().active());
    CHECK(f.sch.draft_costs().has_usable_bucket());
    CHECK(!f.sch.draft_costs().fallback_admitted());
    CHECK_OK(f.start());
}
/* --draft-profile-out at its production seam: a legacy run records the cycles it actually ran,
 * and issuing that record must not turn the Cost policy on. The controller keeps no plan, so
 * draft_cap() still reads next_depth() and the run drafts exactly what it would without the flag;
 * what reaches the table is the executed distribution, not the ceiling. */
TEST(a_legacy_run_with_a_profile_out_records_the_cycles_it_ran) {
    Fixture f;
    f.build();
    f.cfg.n_spec            = 6;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_record_cycle.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);   /* the table Engine::begin_draft_calibration configures before init */
    CHECK_OK(f.start());

    /* Recording is not Cost: no controller plan, before or after the first cycle. */
    CHECK(!f.sch.draft().cost_policy());
    CHECK(f.sch.draft().plan() == nullptr);

    const uint64_t a = f.submit(160), b = f.submit(170);
    f.prefill(a);
    f.prefill(b);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    REQUIRE(f.sch.step() != nullptr);
    CHECK(f.sch.draft().plan() == nullptr);
    f.commit(11, 0, [](const StepEntry&) { return 0; });
    for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, 6));

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    f.sch.step_finished(live);

    /* The next open closes the record; its bucket is the cycle's executed distribution. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK(f.sch.draft().plan() == nullptr);
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    REQUIRE(bs.size() == 1);
    const DraftCostBucket& b1 = bs[0];
    CHECK_EQ(b1.flags, 0);
    CHECK_EQ(b1.n, 1);
    CHECK(b1.mean_ms > 0.0);
    CHECK_EQ(b1.rounds, 6);
    for (int32_t j = 1; j <= 6; ++j) {
        CHECK_EQ(b1.draft_round_rows[j], 2);
        CHECK_EQ(b1.shape.draft_rows[j], 2);
    }
    CHECK_EQ(b1.draft_rows_total, 12);
    CHECK_EQ(b1.shape.draft_k, 6);
    CHECK_EQ(b1.shape.pipeline_depth, 1);
    CHECK_EQ(b1.shape.n_rows, 2);
    /* The first cycle verifies the window the cycle before it drafted, and there was none. */
    CHECK_EQ(b1.shape.verify_k, 0);
    CHECK_EQ(b1.shape.verify_tokens, 2);
    CHECK_EQ(b1.shape.history_tokens, 2);
    CHECK(b1.shape.drafter == DraftDrafter::Serial);
}

/* The same recording under --draft-window adaptive, through the real driver: serial_begin() gives
 * each row the rounds its own window allows, serial_batch() runs each round over the rows that
 * still have one, and the bucket must price the distribution that ran. Two sequences are driven
 * until one stops at 6 and the other at 3; the cycle then executes 6/3, and both the batches and
 * the recorded bucket have to say so. Recording the cycle's deepest window for every row would
 * time four rounds for the shallow row that never ran them (the review's high finding: the
 * published extent, not the executed one). */
TEST(an_adaptive_cycle_runs_and_records_the_rounds_each_row_actually_ran) {
    Fixture f;
    f.build();
    f.cfg.n_spec            = 6;
    f.cfg.draft_window      = DraftWindow::Adaptive;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_record_adaptive_split.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());
    CHECK(!f.sch.draft().cost_policy());

    const uint64_t a = f.submit(160, 2048), b = f.submit(170, 2048);
    f.prefill(a);
    f.prefill(b);
    f.sch.find(a)->speculation_fixed = true;
    f.sch.find(b)->speculation_fixed = true;

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    Scheduler::SerialRow rows[8];
    /* The engine's loop: open a cycle, run the serial passes, issue the next step ahead, commit,
     * publish what ran under the cycle's plan, then promote the step issued ahead. The acceptance
     * drive: A takes its whole window, so its stop stays at the head's depth; B takes three of
     * four or more and two of three, so its estimate falls until its stop is 3. */
    auto acc = [&](const StepEntry& e) {
        if (e.seq == a) return e.n_spec;
        if (e.n_spec >= 4) return 3;
        if (e.n_spec == 3) return 2;
        return e.n_spec;
    };
    bool found = false;
    int  split_k[2] = { 0, 0 };
    REQUIRE(f.sch.step() != nullptr);
    for (int cycle = 0; cycle < 80 && !found; ++cycle) {
        DraftStepEvidence live;
        live.replay_known = true;
        live.replay = DraftReplay::None;

        int rounds = 6;
        REQUIRE_EQ(f.sch.serial_begin(&rounds, rows, 8), 2);
        const RadBatch* hist = f.sch.serial_batch(0, cycle + 1);
        REQUIRE(hist != nullptr);
        CHECK_EQ((int)hist->n_draft_out, 2);   /* both rows draft, so round 0 carries round 1 */
        /* Rows are ordered deepest first, and each round is a prefix of the one before, so the
         * per-round row count is each row's executed depth. */
        int k[2] = { 1, 1 };
        int n_round[7] = {};
        n_round[1] = 2;
        int done = 1;
        for (int rd = 2; rd <= rounds; ++rd) {
            const RadBatch* db = f.sch.serial_batch(rd, cycle + 1);
            REQUIRE(db != nullptr);
            CHECK(!f.sch.serial_batch_failed());
            n_round[rd] = (int)db->n_seq;
            for (int i = 0; i < n_round[rd] && i < 2; ++i) k[i] = rd;
            done = rd;
        }
        if (rounds == 6 && n_round[2] == 2 && n_round[3] == 2 && n_round[4] == 1 &&
            n_round[5] == 1 && n_round[6] == 1) {
            found = true;
            split_k[0] = k[0];
            split_k[1] = k[1];
        }

        for (int i = 0; i < 2; ++i) f.sch.expect_draft(rows[i].seq, done);
        REQUIRE(f.sch.step_next() != nullptr);
        CHECK_OK(f.commit(11, 0, acc));
        /* Published after the commit, as the engine does: `serial_k_[i]` is the depth the row ran,
         * which the step issued ahead was planned to verify. */
        for (int i = 0; i < 2; ++i) CHECK_OK(f.sch.set_draft(rows[i].seq, six, k[i]));
        if (found) {
            CHECK_EQ(f.sch.find(rows[0].seq)->n_draft, 6);
            CHECK_EQ(f.sch.find(rows[1].seq)->n_draft, 3);
        }
        f.sch.serial_end(done, true);
        f.sch.step_finished(live);
        if (found) break;
        f.sch.promote_next();
        REQUIRE_EQ((int)f.sch.plan().e.size(), 2);
    }
    REQUIRE(found);
    CHECK_EQ(split_k[0], 6);
    CHECK_EQ(split_k[1], 3);

    /* The next open closes the record: its bucket is the cycle's executed distribution. */
    f.sch.promote_next();
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    const DraftCostBucket* hit = nullptr;
    for (const DraftCostBucket& x : bs)
        if (x.rounds == 6 && x.draft_round_rows[3] == 2 && x.draft_round_rows[4] == 1) hit = &x;
    REQUIRE(hit != nullptr);
    CHECK_EQ(hit->flags, 0);
    CHECK_EQ(hit->n, 1);
    CHECK_EQ(hit->shape.draft_k, 6);            /* the plan's deepest row */
    CHECK_EQ(hit->shape.n_rows, 2);
    CHECK_EQ(hit->shape.pipeline_depth, 1);
    CHECK(hit->shape.drafter == DraftDrafter::Serial);
    /* N_j of what ran: both rows through round 3, only the deep one to 6. */
    CHECK_EQ(hit->rounds, 6);
    CHECK_EQ(hit->draft_round_rows[1], 2);
    CHECK_EQ(hit->draft_round_rows[2], 2);
    CHECK_EQ(hit->draft_round_rows[3], 2);
    CHECK_EQ(hit->draft_round_rows[4], 1);
    CHECK_EQ(hit->draft_round_rows[5], 1);
    CHECK_EQ(hit->draft_round_rows[6], 1);
    CHECK_EQ(hit->draft_rows_total, 9);
    /* The plan's own distribution, which the executed one is checked against. */
    CHECK_EQ(hit->shape.draft_rows[1], 2);
    CHECK_EQ(hit->shape.draft_rows[3], 2);
    CHECK_EQ(hit->shape.draft_rows[4], 1);
    CHECK_EQ(hit->shape.draft_rows[6], 1);
}

/* A cycle wider than the profile format can name (RAD_SCHED_MAX_SPEC). Recording a prefix of it
 * would file a shape the step never ran -- the bracket timed all of the rows -- so the cycle is
 * dropped whole and counted, and the table learns nothing from it. */
TEST(a_cycle_over_the_row_limit_is_dropped_whole_not_truncated) {
    Fixture f;
    f.build(8192, 128, 512, 4096);
    f.cfg.max_seqs          = 72;            /* a view wider than the 64 the format names */
    f.cfg.n_spec            = 6;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_record_over_rows.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);
    CHECK_OK(f.start());
    CHECK(f.sch.draft_costs().buckets().empty());

    /* One-step prompts: the first step admits and prefills all seventy, so the second opens a
     * cycle with seventy decode rows. */
    for (int i = 0; i < 70; ++i) f.submit(1, 64);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 70);
    f.commit_plain(11);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 1);
    CHECK(f.sch.draft_costs().buckets().empty());
    f.commit_plain(12);
}

/* The residency the profile was measured on. A move inside a step's bracket is reported by the
 * rank; a move the scheduler's own ticks make between two brackets is not, so the engine reports
 * the cumulative count since the profile was bound. Once it diverges, every later cycle is marked
 * -- including one that saw no move of its own: the run cannot move back, and a later clean-looking
 * bracket still does not describe the start placement. */
TEST(a_drifted_residency_marks_every_later_record_until_the_run_ends) {
    Fixture f;
    f.build();
    f.cfg.n_spec            = 6;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_record_drift.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160), b = f.submit(170);
    f.prefill(a);
    f.prefill(b);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    auto cycle = [&](int64_t moves_since_bind) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, 6));
        DraftStepEvidence live;
        live.replay_known = true;
        live.replay = DraftReplay::None;
        live.moves_since_bind = moves_since_bind;
        f.sch.step_finished(live);
    };
    cycle(0);   /* measured on the start placement */
    cycle(1);   /* the scheduler's ticks moved a unit between the two brackets */
    cycle(0);   /* no move of its own -- and no longer on the start placement either */

    /* The next open closes the last record. */
    REQUIRE(f.sch.step() != nullptr);
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    const DraftCostBucket* clean = nullptr;
    const DraftCostBucket* drifted = nullptr;
    for (const DraftCostBucket& x : bs) {
        if (x.flags & kDraftSamplePlacement) drifted = &x;
        else clean = &x;
    }
    REQUIRE(clean != nullptr);
    REQUIRE(drifted != nullptr);
    /* The one before the move stays: it measured the start plan. */
    CHECK_EQ(clean->n, 1);
    CHECK_EQ(clean->shape.verify_k, 0);
    /* The move's cycle and the clean-looking one after it are both excluded. */
    CHECK_EQ(drifted->n, 2);
    CHECK_EQ(drifted->shape.verify_k, 6);
}

/* A cost calibration is record-only: it prices nothing, so a moved residency cannot turn its
 * records into prices for a placement the run has left. It must therefore not be swept up by the
 * online fail-closed rule -- it keeps measuring the named baseline, marks every cycle after the
 * move, and the export leaves those out. The one-way latch belongs to an online table. */
TEST(a_record_only_calibration_keeps_recording_after_a_move) {
    Fixture f;
    f.build();
    f.cfg.n_spec            = 6;
    f.cfg.draft_policy      = DraftPolicyKind::Cost;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_record_only_drift.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);                        /* empty profile, fallback admitted */
    f.sch.draft_costs().record_only();        /* --draft-profile-out under cost */
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft_costs().records_only());

    const uint64_t a = f.submit(160), b = f.submit(170);
    f.prefill(a);
    f.prefill(b);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    auto cycle = [&](int64_t moves_since_bind) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, 6));
        DraftStepEvidence live;
        live.replay_known = true;
        live.replay = DraftReplay::None;
        live.moves_since_bind = moves_since_bind;
        f.sch.step_finished(live);
    };
    cycle(0);
    cycle(1);   /* the scheduler's ticks moved a unit between the two brackets */
    cycle(0);   /* no move of its own -- and no longer on the start placement either */

    /* Not deactivated and not latched: the calibration goes on measuring. */
    CHECK(f.sch.draft_costs().active());
    CHECK(f.sch.draft_costs().records_only());
    CHECK(!f.sch.draft_cost_aborted());

    /* The next planning opens a cycle, which the online rule would have refused, and the plan it
     * opens is still the named baseline, never a price. */
    const DraftPlan* before = f.sch.draft().plan();
    REQUIRE(before != nullptr);
    const uint64_t cycle_before = before->cycle_id;
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.draft().plan()->cycle_id, cycle_before + 1);
    CHECK_EQ(f.sch.draft().plan()->source, DraftPlanSource::Fallback);
    f.commit_plain(11);

    /* Only the start cycle is unmarked; every cycle after the move is filed under the placement
     * flag, which the export leaves out. */
    int64_t clean = 0, drifted = 0;
    for (const DraftCostBucket& x : f.sch.draft_costs().buckets()) {
        if (x.flags & kDraftSamplePlacement) drifted += x.n;
        else clean += x.n;
    }
    CHECK_EQ(clean, 1);
    CHECK_EQ(drifted, 2);
}

/* The start placement is the plan plus the fill, and the fill retires over several dispatches.
 * A bracket that ran while it was still landing measured a residency the fingerprint does not
 * name -- even when nothing retired inside it -- so the engine reports the state with the bracket
 * (DraftStepEvidence::placement_unresolved) and every such record is marked. The mark is per
 * bracket, not sticky: once the fill has landed, a clean cycle is a measurement of the start
 * placement and may be exported and priced. */
TEST(cycles_measured_during_the_initial_fill_are_marked_and_a_later_clean_one_is_not) {
    Fixture f;
    f.build();
    f.cfg.n_spec            = 6;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_unresolved_fill.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);
    f.sch.draft_costs().record_only();   /* --draft-profile-out under cost */
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160), b = f.submit(170);
    f.prefill(a);
    f.prefill(b);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    auto cycle = [&](bool unresolved, int64_t moves_since_bind) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, 6));
        DraftStepEvidence live;
        live.replay_known        = true;
        live.replay              = DraftReplay::None;
        live.moves_since_bind    = moves_since_bind;
        live.placement_unresolved = unresolved;
        f.sch.step_finished(live);
    };
    cycle(/*unresolved=*/true, 0);    /* the fill's copies are still landing */
    cycle(/*unresolved=*/true, 0);    /* still nothing retired in this bracket, still unresolved */
    cycle(/*unresolved=*/false, 0);   /* the fill settled between the brackets */

    REQUIRE(f.sch.step() != nullptr);   /* closes the last record */
    int64_t clean = 0, pending = 0;
    const DraftCostBucket* first_clean = nullptr;
    bool saw_pending = false;
    for (const DraftCostBucket& x : f.sch.draft_costs().buckets()) {
        if (x.flags & kDraftSamplePlacement) { pending += x.n; saw_pending = true; continue; }
        clean += x.n;
        if (!first_clean) first_clean = &x;
    }
    CHECK(saw_pending);
    CHECK_EQ(pending, 2);            /* both pre-resolve cycles, though neither moved a unit */
    CHECK_EQ(clean, 1);              /* the first bracket on the materialised placement */
    /* This is not drift: the table still prices, and a calibration keeps measuring. */
    CHECK(f.sch.draft_costs().active());
    CHECK(f.sch.draft_costs().records_only());
    CHECK(!f.sch.draft_cost_aborted());
    /* And the clean cycle is a bucket the export may write and the policy may price from. */
    REQUIRE(first_clean != nullptr);
    CHECK(draft_bucket_usable(*first_clean));
}

/* A move during the fill is sticky even after the placement resolves: the run cannot move back to
 * the start placement, and a clean-looking bracket after it is still not a measurement of it. The
 * scheduler must not re-baseline on the resolve -- the resolve is not a reset. */
TEST(a_move_during_the_fill_stays_marked_after_the_placement_resolves) {
    Fixture f;
    f.build();
    f.cfg.n_spec            = 6;
    f.cfg.draft_profile_out = radtest::tmp_path("sched_move_during_fill.json");
    f.prog.drafter.kind     = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth    = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160), b = f.submit(170);
    f.prefill(a);
    f.prefill(b);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    auto cycle = [&](bool unresolved, bool moved, int64_t moves_since_bind) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        for (uint64_t id : { a, b }) CHECK_OK(f.sch.set_draft(id, six, 6));
        DraftStepEvidence live;
        live.replay_known         = true;
        live.replay               = DraftReplay::None;
        live.moved                = moved;
        live.moves_since_bind     = moves_since_bind;
        live.placement_unresolved = unresolved;
        f.sch.step_finished(live);
    };
    cycle(/*unresolved=*/true, /*moved=*/true, 1);    /* a swap published during the fill */
    cycle(/*unresolved=*/false, /*moved=*/false, 0);  /* resolved and clean by its own evidence */

    REQUIRE(f.sch.step() != nullptr);
    int64_t clean = 0, drifted = 0;
    for (const DraftCostBucket& x : f.sch.draft_costs().buckets()) {
        if (x.flags & kDraftSamplePlacement) drifted += x.n;
        else clean += x.n;
    }
    CHECK_EQ(clean, 0);
    CHECK_EQ(drifted, 2);
    /* A calibration keeps recording and is not aborted; the export leaves all of it out. */
    CHECK(f.sch.draft_costs().active());
    CHECK(!f.sch.draft_cost_aborted());
}

TEST(a_drifted_residency_stops_the_cost_policy_from_pricing_the_loaded_profile) {
    Fixture f;
    f.build();
    f.cfg.n_spec        = 6;
    f.cfg.draft_policy  = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f, /*priced=*/true);
    CHECK(!f.sch.draft_costs().fallback_admitted());
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);

    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    REQUIRE(f.sch.draft_costs().active());

    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    live.moves_since_bind = 1;
    f.sch.step_finished(live);
    CHECK(!f.sch.draft_costs().active());
    CHECK_EQ(f.sch.draft_costs().reason(),
             std::string("residency drifted from the fingerprint's start placement"));
    CHECK(!f.sch.draft_costs().fallback_admitted());

    /* The next planning site must open no cycle at all. With the table inactive every price is
     * "not priced", the planner would issue its named fallback at the ceiling, and the run would
     * draft at a depth the profile never measured -- the implicit baseline the admission rule
     * refuses. The open is refused instead, and the latch is what the engine ends the run on. */
    CHECK(!f.sch.draft_cost_aborted());
    const DraftPlan* before = f.sch.draft().plan();
    REQUIRE(before != nullptr);
    const uint64_t cycle_before = before->cycle_id;
    CHECK(f.sch.step() == nullptr);
    CHECK(f.sch.draft_cost_aborted());
    /* No new plan was priced: the one in force is still the last cycle opened before the drift. */
    REQUIRE(f.sch.draft().plan() != nullptr);
    CHECK_EQ(f.sch.draft().plan()->cycle_id, cycle_before);
    CHECK(f.sch.draft().plan()->source != DraftPlanSource::Fallback);
    CHECK(!f.sch.draft_costs().fallback_admitted());
    /* One-way: the next planning site is refused too, without a second cycle. */
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ(f.sch.draft().plan()->cycle_id, cycle_before);
}

/* B1's other half: the move is published between two brackets -- the scheduler's idle ticks run
 * after step_finished -- so no bracket of the closing cycle carries it. The engine hands the
 * cumulative count in before it opens the next cycle (Engine::step_loop's head, draft_residency),
 * and that open is refused on the same sticky decision: no cycle priced on the start placement,
 * and no fallback at the ceiling either. */
TEST(a_cycle_opened_after_a_between_brackets_move_is_refused) {
    Fixture f;
    f.build();
    f.cfg.n_spec        = 6;
    f.cfg.draft_policy  = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f, /*priced=*/true);
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, six, 6));
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    f.sch.step_finished(live);                 /* the bracket's own evidence: no move */
    REQUIRE(f.sch.draft_costs().active());

    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    const uint64_t cycle = p->cycle_id;

    /* The idle tick's move, reported where the engine reports it: before the next planning. */
    f.sch.draft_residency(1);
    CHECK(!f.sch.draft_costs().active());
    CHECK_EQ(f.sch.draft_costs().reason(),
             std::string("residency drifted from the fingerprint's start placement"));
    CHECK(!f.sch.draft_cost_aborted());        /* noted, but no planning site has run yet */

    CHECK(f.sch.step() == nullptr);
    CHECK(f.sch.draft_cost_aborted());
    CHECK(!f.sch.draft_costs().fallback_admitted());
    REQUIRE(f.sch.draft().plan() != nullptr);
    CHECK_EQ(f.sch.draft().plan()->cycle_id, cycle);   /* the refused open priced nothing */
    CHECK(f.sch.draft().plan()->source != DraftPlanSource::Fallback);
}

/* The same refusal at the other planning site: a step issued ahead becomes current in
 * promote_next(), which opens its cycle there. A drifted run must not execute that step under the
 * old plan, and the ahead plan is left where it is. */
TEST(a_promoted_step_after_a_between_brackets_move_is_refused) {
    Fixture f;
    f.build();
    f.cfg.n_spec        = 6;
    f.cfg.draft_policy  = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f, /*priced=*/true);
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;

    /* The engine's order: this step is built, the next one is issued ahead while it is still open,
     * it commits, and only then does the promote make the ahead step current. */
    REQUIRE(f.sch.step() != nullptr);
    f.sch.expect_draft(id, 6);
    REQUIRE(f.sch.step_next() != nullptr);    /* the step issued ahead, with its own plan */
    f.commit_plain(11);

    const DraftPlan* p = f.sch.draft().plan();
    REQUIRE(p != nullptr);
    const uint64_t cycle = p->cycle_id;

    f.sch.draft_residency(1);
    f.sch.promote_next();
    CHECK(f.sch.draft_cost_aborted());
    CHECK(!f.sch.draft_costs().fallback_admitted());
    /* The ahead plan was not made current, and nothing was priced after the drift. */
    REQUIRE(f.sch.draft().plan() != nullptr);
    CHECK_EQ(f.sch.draft().plan()->cycle_id, cycle);
    REQUIRE_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 0);   /* the ahead step (6) never became current */
}

/* ================================================================== the plan's edge cases */

/* A row admitted after the cycle's view was built is not priced by that plan. The plan is the
 * budget: a row it does not name drafts nothing this cycle, rather than falling back to the
 * configured ceiling -- that would be a draft the cycle never priced. */
TEST(a_row_admitted_after_the_cycle_opened_drafts_nothing_under_its_plan) {
    Fixture f;
    f.build();
    f.cfg.n_spec        = 6;
    f.cfg.draft_policy  = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f, /*priced=*/true);
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t a = f.submit(160);
    pred.set(a, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(a);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);

    /* B arrives while A decodes: the next cycle's view is A only, and B is admitted afterwards,
     * in the same step -- so it is in the step's plan and not in the cycle's. */
    const uint64_t b = f.submit(160);
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy p = copy_plan(f.sch);
    CHECK_EQ(p.at(a), 6);
    CHECK_EQ(p.at(b), -1);                      /* the plan does not name B */
    REQUIRE(entry_of(f.sch, b) != nullptr);     /* the step does carry it */
    f.commit_plain(11);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    CHECK_OK(f.sch.set_draft(b, six, 6));
    CHECK_EQ(f.sch.find(b)->n_draft, 0);
}

/* The ceiling is the run's, not the head's: a draft head declared deeper than
 * --num-speculative-tokens can run must not be asked for rounds the step cannot verify. */
TEST(a_head_deeper_than_the_ceiling_drafts_only_the_ceiling) {
    Fixture f;
    f.build();
    f.cfg.n_spec        = 2;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    CHECK_OK(f.start());

    const uint64_t id = f.submit(160);
    f.prefill(id);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);

    int32_t two[2] = { 31, 32 };
    CHECK_OK(f.sch.set_draft(id, two, 2));
    REQUIRE(f.sch.step() != nullptr);
    const StepEntry* e = entry_of(f.sch, id);
    REQUIRE(e != nullptr);
    CHECK_EQ(e->n_spec, 2);
    f.commit_plain(11);

    int rounds = 4;                              /* what the head could run */
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 1);
    CHECK_EQ(rounds, 2);                         /* the ceiling clamps the rounds */
    CHECK_EQ(rows[0].seq, id);
}

/* The published extent may be shorter than the executed one: the step issued ahead was planned to
 * verify three and the head still ran six rounds, so set_draft() cuts the publication to three and
 * keeps what ran. The cycle keeps both -- its bucket prices six rounds and the verify it carried. */
TEST(a_publication_cut_below_the_executed_rounds_keeps_the_executed_extent) {
    Fixture f;
    f.build();
    f.cfg.n_spec        = 6;
    f.cfg.draft_policy  = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start());

    Predictions pred;
    const uint64_t id = f.submit(160);
    pred.set(id, { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.01f });
    use_cost_policy(f, pred);
    f.prefill(id);
    f.sch.find(id)->speculation_fixed = true;    /* what step_next() requires of a decode row */

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    /* Cycle A: the plan opens at six; its step verifies nothing (the prefill published nothing). */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(planned(f.sch, id), 6);

    /* The step issued ahead is planned to verify three, while the current step is still open; the
     * head then executes six. The publication is cut to the planned count, and the six rounds it
     * ran stay with the cycle. */
    f.sch.expect_draft(id, 3);
    REQUIRE(f.sch.step_next() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(id, six, 6));
    CHECK_EQ(f.sch.find(id)->n_draft, 3);

    /* A closes at the promote; the step now current verifies the three that were published. */
    f.sch.promote_next();
    REQUIRE(entry_of(f.sch, id) != nullptr);
    CHECK_EQ(entry_of(f.sch, id)->n_spec, 3);
    CHECK_OK(f.commit(11, 0, [](const StepEntry& e) { return e.n_spec; }));
    CHECK_OK(f.sch.set_draft(id, six, 6));       /* six published under B ... */
    f.sch.step_finished(live);

    /* ... and the next open closes B: six rounds ran, three positions were verified. */
    REQUIRE(f.sch.step() != nullptr);
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    const DraftCostBucket* hit = nullptr;
    for (const DraftCostBucket& x : bs)
        if (x.rounds == 6 && x.shape.verify_k == 3) hit = &x;
    REQUIRE(hit != nullptr);
    CHECK_EQ(hit->flags, 0);                     /* a complete measurement, not shortened */
    CHECK_EQ(hit->draft_round_rows[6], 1);
    CHECK_EQ(hit->draft_rows_total, 6);
    CHECK_EQ(hit->shape.draft_k, 6);
    CHECK_EQ(hit->shape.rows[1], 1);
    CHECK_EQ(hit->shape.rows[3], 1);
    CHECK_EQ(hit->shape.rows[4], 0);
    CHECK_EQ(hit->shape.verify_tokens, 4);       /* one row, 1 + three verified positions */
}

/* The forced sequence through the real cycle path. A prefill step opens a zero-row cycle first --
 * no row is decoding yet, so the sequence must not spend a depth on it -- and the decode cycles
 * that follow take 3, 2, 2, 3 in that order, the last wrapping to 3. Each plan names the window the
 * cycle before it drafted (`verify_k`); the step verifies what its own plan's per-row cap allows --
 * the previous publication on a rise, and on a fall capped to this plan's own K. Nothing is priced:
 * every plan is `Calibration`, and the samples are the ordinary ones, one clean bucket per shape
 * that ran. */
TEST(a_calibration_sequence_runs_through_the_real_cycle_after_an_empty_one) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.cfg.draft_calib_k[0] = 3;
    f.cfg.draft_calib_k[1] = 2;
    f.cfg.draft_calib_k[2] = 2;
    f.cfg.draft_calib_k[3] = 3;
    f.cfg.draft_calib_n    = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start(DraftReplay::None));

    uint64_t ids[4];
    for (int i = 0; i < 4; ++i) ids[i] = f.submit(32);
    for (int i = 0; i < 4; ++i) f.prefill(ids[i]);

    /* The prefill step(s) opened a forced plan over no rows, and the sequence did not move: the
     * next row-bearing plan still takes its first element. In the ledger such a cycle is an empty
     * planning point, not a forced measurement -- it has no depth to measure -- and it leaves the
     * last plan with rows alone (there is none yet). */
    const DraftPlan* p0 = f.sch.draft().plan();
    REQUIRE(p0 != nullptr);
    CHECK_EQ(p0->n_seq, 0);
    CHECK(p0->source == DraftPlanSource::Calibration);
    CHECK_EQ(p0->max_k, 0);
    const SchedMetrics prefilled = f.sch.metrics();
    CHECK(prefilled.draft_cost_cycles >= 1);
    CHECK_EQ(prefilled.draft_cost_empty, prefilled.draft_cost_cycles);
    CHECK_EQ(prefilled.draft_calib_forced, 0);
    CHECK_EQ(prefilled.draft_cost_last_rows, 0);
    CHECK_EQ(prefilled.draft_cost_last_k, 0);

    const int32_t forced[5] = { 3, 2, 2, 3, 3 };
    /* What the plan names and what the step verifies: the previous cycle's drafted depth, capped
     * to this plan's own K -- every cycle here is opened by step(), the direct path. The first
     * row-bearing cycle has no predecessor. */
    const int32_t verifies[5] = { 0, 2, 2, 2, 3 };
    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    for (int c = 0; c < 5; ++c) {
        REQUIRE(f.sch.step() != nullptr);
        const PlanCopy p = copy_plan(f.sch);
        REQUIRE(p.n_seq == 4);
        CHECK_EQ(p.max_k, forced[c]);
        CHECK_EQ(p.verify_k, verifies[c]);
        CHECK_EQ(p.pipeline_depth, 1);
        for (uint64_t id : ids) CHECK_EQ(p.at(id), forced[c]);
        f.commit(11, 0, [&](const StepEntry&) { return verifies[c]; });
        for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, six, forced[c]));
        f.sch.step_finished(live);
    }

    /* One more open retires the fifth cycle; the period is four, so the sixth plan takes the
     * second element again. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(copy_plan(f.sch).max_k, 2);

    /* The shapes that ran, in order: the v0 cycle (verifies nothing), the two K2 cycles, the
     * K3 cycle after a K2 one (its step verifies the two the fall published), and the K3 steady
     * cycle. The two K2 cycles are one bucket. */
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    REQUIRE(bs.size() == 4);
    const int32_t ran_k[4]  = { 3, 2, 3, 3 };
    const int32_t ran_v[4]  = { 0, 2, 2, 3 };
    const int64_t ran_n[4]  = { 1, 2, 1, 1 };
    for (int c = 0; c < 4; ++c) {
        const DraftCostBucket& b = bs[(size_t)c];
        CHECK_EQ(b.flags, 0);                  /* a clean measurement, whatever was forced */
        CHECK_EQ(b.n, ran_n[c]);
        CHECK_EQ(b.rounds, ran_k[c]);
        CHECK_EQ(b.shape.draft_k, ran_k[c]);
        CHECK_EQ(b.shape.verify_k, ran_v[c]);
        CHECK_EQ(b.shape.n_rows, 4);
        CHECK_EQ(b.draft_rows_total, 4 * ran_k[c]);
        for (int32_t j = 1; j <= ran_k[c]; ++j) CHECK_EQ(b.draft_round_rows[j], 4);
    }
    CHECK_EQ(f.sch.draft_cost_refused(), 0);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 0);
    CHECK_EQ(f.sch.draft_costs().ignored(), 0);

    /* The ledger: every opened cycle was a forced calibration plan with rows or an empty planning
     * point, none a decision, and the five ends still account for every cycle that opened. */
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_priced, 0);
    CHECK_EQ(m.draft_cost_fallback, 0);
    CHECK_EQ(m.draft_cost_no_plan, 0);
    CHECK_EQ(m.draft_cost_empty, prefilled.draft_cost_cycles);
    CHECK_EQ(m.draft_calib_forced, 6);   /* the five loop cycles and the retiring open */
    CHECK_EQ(m.draft_cost_tiers_priced, 0);
    CHECK_EQ(m.draft_calib_cut, 0);
    CHECK_EQ(m.draft_cost_priced + m.draft_cost_fallback + m.draft_calib_forced +
                 m.draft_cost_empty + m.draft_cost_no_plan,
             m.draft_cost_cycles);
    CHECK_EQ(m.draft_cost_last_source, (int64_t)DraftPlanSource::Calibration);
    CHECK_EQ(m.draft_cost_last_rows, 4);
    CHECK_EQ(m.draft_cost_last_k, 2);
}

/* The forced sequence down the ahead path, across the period boundary. From the second row-bearing
 * cycle on, every plan is opened by promote_next(), after the step that verifies the previous plan's
 * drafts was built; that step is the one the cycle runs, so each plan's `verify_k` is the frozen
 * window forced[c-1], not its own K. The period still turns: the fifth row-bearing plan takes the
 * first element again and the sixth the second. Because the drafted and verified windows differ,
 * every cycle is its own bucket -- unlike the direct run, where the two K2 cycles share one. */
TEST(a_calibration_sequence_issued_ahead_runs_across_the_period_boundary) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.cfg.draft_calib_k[0] = 3;
    f.cfg.draft_calib_k[1] = 2;
    f.cfg.draft_calib_k[2] = 2;
    f.cfg.draft_calib_k[3] = 3;
    f.cfg.draft_calib_n    = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start(DraftReplay::None));

    uint64_t ids[4];
    for (int i = 0; i < 4; ++i) ids[i] = f.submit(32);
    for (int i = 0; i < 4; ++i) f.prefill(ids[i]);
    for (int i = 0; i < 4; ++i) f.sch.find(ids[i])->speculation_fixed = true;

    /* The prefill planned over no rows; the sequence did not move. */
    const DraftPlan* p0 = f.sch.draft().plan();
    REQUIRE(p0 != nullptr);
    CHECK_EQ(p0->n_seq, 0);
    CHECK(p0->source == DraftPlanSource::Calibration);
    const SchedMetrics prefilled = f.sch.metrics();
    CHECK_EQ(prefilled.draft_cost_empty, prefilled.draft_cost_cycles);
    CHECK_EQ(prefilled.draft_calib_forced, 0);

    const int32_t forced[5] = { 3, 2, 2, 3, 3 };
    /* What the step of a cycle runs: the plan before it drafted at this depth. The first
     * row-bearing cycle has no predecessor. */
    const int32_t verifies[5] = { 0, 3, 2, 2, 3 };
    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    /* Cycle 0 is the one open that cannot be issued ahead: no step existed before it. */
    REQUIRE(f.sch.step() != nullptr);
    {
        const PlanCopy p = copy_plan(f.sch);
        REQUIRE(p.n_seq == 4);
        CHECK_EQ(p.max_k, 3);
        CHECK_EQ(p.verify_k, 0);
        CHECK_EQ(p.pipeline_depth, 1);
        for (uint64_t id : ids) CHECK_EQ(p.at(id), 3);
    }

    for (int c = 0; c < 5; ++c) {
        if (c > 0) {
            const PlanCopy p = copy_plan(f.sch);
            REQUIRE(p.n_seq == 4);
            CHECK_EQ(p.max_k, forced[c]);
            CHECK_EQ(p.verify_k, verifies[c]);
            CHECK_EQ(p.pipeline_depth, 1);
            for (uint64_t id : ids) CHECK_EQ(p.at(id), forced[c]);
        }
        /* The step that runs in this cycle was built under the plan before it, so it verifies
         * that plan's publication -- the window `verify_k` declares. */
        REQUIRE_EQ(f.sch.plan().e.size(), (size_t)4);
        for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, verifies[c]);
        /* The next cycle's step verifies what this plan's rounds publish. */
        for (uint64_t id : ids) f.sch.expect_draft(id, forced[c]);
        const RadBatch* built = f.sch.step_next();
        REQUIRE(built != nullptr);
        CHECK_EQ(built->n_spec, forced[c]);
        f.commit(11, 0, [&](const StepEntry&) { return verifies[c]; });
        for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, six, forced[c]));
        f.sch.step_finished(live);
        f.sch.promote_next();
    }

    /* The sixth row-bearing plan, opened ahead at the end of the fifth cycle: the period wrapped,
     * so it takes the second element, and the step that runs in it verifies the three the fifth
     * published. */
    {
        const PlanCopy p = copy_plan(f.sch);
        REQUIRE(p.n_seq == 4);
        CHECK_EQ(p.max_k, 2);
        CHECK_EQ(p.verify_k, 3);
    }

    /* Every cycle is its own bucket: (drafted, verified) never repeats, because on this path the
     * verify window lags the draft window by one plan. */
    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    REQUIRE(bs.size() == 5);
    const int32_t ran_k[5] = { 3, 2, 2, 3, 3 };
    const int32_t ran_v[5] = { 0, 3, 2, 2, 3 };
    for (int c = 0; c < 5; ++c) {
        const DraftCostBucket& b = bs[(size_t)c];
        CHECK_EQ(b.flags, 0);
        CHECK_EQ(b.n, 1);
        CHECK_EQ(b.rounds, ran_k[c]);
        CHECK_EQ(b.shape.draft_k, ran_k[c]);
        CHECK_EQ(b.shape.verify_k, ran_v[c]);
        CHECK_EQ(b.shape.n_rows, 4);
        CHECK_EQ(b.shape.verify_tokens, 4 * (1 + ran_v[c]));
        CHECK_EQ(b.shape.history_tokens, 4 * (1 + ran_v[c]));
        CHECK_EQ(b.draft_rows_total, 4 * ran_k[c]);
        for (int32_t j = 1; j <= ran_k[c]; ++j) CHECK_EQ(b.draft_round_rows[j], 4);
    }
    CHECK_EQ(f.sch.draft_cost_refused(), 0);
    CHECK_EQ(f.sch.draft_cycles_dropped(), 0);
    CHECK_EQ(f.sch.draft_costs().ignored(), 0);

    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_priced, 0);
    CHECK_EQ(m.draft_cost_fallback, 0);
    CHECK_EQ(m.draft_cost_no_plan, 0);
    CHECK_EQ(m.draft_cost_empty, prefilled.draft_cost_cycles);
    CHECK_EQ(m.draft_calib_forced, 6);   /* the five loop cycles and the sixth open */
    CHECK_EQ(m.draft_cost_tiers_priced, 0);
    CHECK_EQ(m.draft_calib_cut, 0);
    CHECK_EQ(m.draft_cost_priced + m.draft_cost_fallback + m.draft_calib_forced +
                 m.draft_cost_empty + m.draft_cost_no_plan,
             m.draft_cost_cycles);
    CHECK_EQ(m.draft_cost_last_source, (int64_t)DraftPlanSource::Calibration);
    CHECK_EQ(m.draft_cost_last_rows, 4);
    CHECK_EQ(m.draft_cost_last_k, 2);
}

/* A plan-time budget cut is not a shortened bracket: the sequence asks for K3 over twenty-four
 * rows but the step was given sixty-four tokens, so the plan itself is scaled to two per row and
 * says why. The cycle executes exactly that, files a clean bucket at the depth that ran, and the
 * next plan verifies the cut window -- while `calib_cut` (not `Shortened`) is what records the cut. */
TEST(a_budget_cut_shortens_a_forced_plan_and_the_next_cycle_verifies_it) {
    Fixture f;
    f.build(4096, 64, /*max_tok=*/64);   /* the geometry's floor: a chunk needs at least 64 */
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.cfg.draft_calib_k[0] = 3;
    f.cfg.draft_calib_k[1] = 2;
    f.cfg.draft_calib_n    = 2;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start(DraftReplay::None));

    /* 24 x 3 = 72 forced drafts against a 64-token step budget. */
    uint64_t ids[24];
    for (int i = 0; i < 24; ++i) ids[i] = f.submit(2);
    for (int i = 0; i < 24; ++i) f.prefill(ids[i]);

    int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;

    /* Cycle 1: every row scaled from 3 to 2, and the plan says why on each. */
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy cut = copy_plan(f.sch);
    REQUIRE(cut.n_seq == 24);
    CHECK_EQ(cut.max_k, 2);
    CHECK_EQ(cut.verify_k, 0);
    for (uint64_t id : ids) CHECK_EQ(cut.at(id), 2);
    const DraftPlan* pc = f.sch.draft().plan();
    REQUIRE(pc != nullptr);
    for (int32_t i = 0; i < pc->n_seq; ++i) CHECK(pc->why[i] == DraftFallback::Resources);
    CHECK_EQ(f.sch.metrics().draft_calib_cut, 1);
    CHECK_EQ(f.sch.metrics().draft_cost_tiers_priced, 0);

    f.commit(11, 0, [](const StepEntry&) { return 0; });
    for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, six, 2));
    f.sch.step_finished(live);

    /* Cycle 2: the next element, K2, fits the budget and is not cut; its paired step verifies the
     * two the cut plan drafted. The cut plan's own sample was clean. */
    REQUIRE(f.sch.step() != nullptr);
    const PlanCopy next = copy_plan(f.sch);
    REQUIRE(next.n_seq == 24);
    CHECK_EQ(next.max_k, 2);
    CHECK_EQ(next.verify_k, 2);
    CHECK_EQ(f.sch.metrics().draft_calib_cut, 1);      /* the second plan was not cut */

    const std::vector<DraftCostBucket>& bs = f.sch.draft_costs().buckets();
    REQUIRE(bs.size() == 1);
    CHECK_EQ(bs[0].flags, 0);                           /* the plan-time cut is not Shortened */
    CHECK_EQ(bs[0].shape.draft_k, 2);
    CHECK_EQ(bs[0].shape.verify_k, 0);
    CHECK_EQ(bs[0].rounds, 2);
    CHECK_EQ(bs[0].draft_rows_total, 24 * 2);
    CHECK_EQ(f.sch.draft_cost_refused(), 0);
}

/* A cycle opens at every step, and a step with no decode rows plans over an empty view: the
 * controller issues the valid K0 plan for it (draft_policy.cpp's fallback branch), which is
 * exactly the shape the old ledger counted as "the policy chose the fallback". The ledger counts
 * it apart (`empty`), keeps the `last_*` plan for the last plan **with rows**, and spends no trace
 * line. A valid all-K0 plan over real rows is still a decision: `last_k == 0` with `last_rows > 0`. */
TEST(an_idle_or_prefill_step_is_an_empty_planning_point_not_a_fallback) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);                     /* live, empty, fallback admitted */
    CHECK_OK(f.start(DraftReplay::None));

    /* Idle: no request exists, so every step plans nothing and the cycle is an empty planning
     * point. Nothing was decided, measured or drafted. */
    for (int i = 0; i < 10; ++i) CHECK(f.sch.step() == nullptr);
    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_cycles, 10);
    CHECK_EQ(m.draft_cost_empty, 10);
    CHECK_EQ(m.draft_cost_fallback, 0);
    CHECK_EQ(m.draft_cost_priced, 0);
    CHECK_EQ(m.draft_cost_no_plan, 0);
    CHECK_EQ(m.draft_cost_last_rows, 0);   /* no plan with rows yet: no last_* at all */
    CHECK_EQ(m.draft_cost_last_k, 0);
    CHECK_EQ(f.sch.draft_cycles_done(), 0);
    {
        const DraftPlan* p = f.sch.draft().plan();
        REQUIRE(p != nullptr);
        CHECK_EQ(p->n_seq, 0);
        CHECK_EQ(p->max_k, 0);
        /* The plan is the fallback branch's shape -- which is why the ledger must not count it
         * as the policy's fallback choice. */
        CHECK(p->source == DraftPlanSource::Fallback);
    }

    /* A pure prefill step is the same kind of planning point: the chunk runs, nothing decodes. */
    const uint64_t id = f.submit(64);
    REQUIRE(f.sch.step() != nullptr);
    m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_cycles, 11);
    CHECK_EQ(m.draft_cost_empty, 11);
    CHECK_EQ(m.draft_cost_fallback, 0);
    CHECK_EQ(m.draft_cost_last_rows, 0);
    f.commit_plain(11);

    /* The first decode cycle has a row and the named baseline (the table is empty): a real
     * fallback, and the first plan the `last_*` fields may describe. */
    REQUIRE(f.sch.step() != nullptr);
    m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_cycles, 12);
    CHECK_EQ(m.draft_cost_empty, 11);
    CHECK_EQ(m.draft_cost_fallback, 1);
    CHECK_EQ(m.draft_cost_last_source, (int64_t)DraftPlanSource::Fallback);
    CHECK_EQ(m.draft_cost_last_rows, 1);
    CHECK_EQ(m.draft_cost_last_k, 6);       /* the executable depth */
    f.commit_plain(11);

    /* A row that may not speculate plans K0 over a real row: no drafts by decision, not an empty
     * planning point. `last_k == 0` with `last_rows > 0` is that decision. */
    Request* r = f.sch.find(id);
    REQUIRE(r != nullptr);
    r->may_speculate = false;
    REQUIRE(f.sch.step() != nullptr);
    m = f.sch.metrics();
    CHECK_EQ(m.draft_cost_cycles, 13);
    CHECK_EQ(m.draft_cost_empty, 11);       /* the K0 plan is not one of them */
    CHECK_EQ(m.draft_cost_fallback, 2);
    CHECK_EQ(m.draft_cost_last_rows, 1);
    CHECK_EQ(m.draft_cost_last_k, 0);
    CHECK_EQ(m.draft_cost_priced + m.draft_cost_fallback + m.draft_calib_forced +
                 m.draft_cost_empty + m.draft_cost_no_plan,
             m.draft_cost_cycles);
}

/* ------------------------------------------------------- the trace window (a child process)
 *
 * `RADIANCE_DRAFT_TRACE` is read once per process, and the window is what a real run's diagnosis
 * depends on: 64 lines shared by plans, brackets and samples. A long idle phase before the
 * measurement window -- idle steps, then prefill chunks, none of them planning a decode row -- used
 * to spend one plan line each and leave nothing for the cycles that were being measured. The
 * child below runs 200 idle steps and then a handful of decode cycles; the parent runs it with the
 * trace on and checks that every request-bearing cycle is still fully traceable. */

/* The child: an idle phase longer than the whole window, then a request-bearing run. It prints its
 * own ledger so the parent can check the phase really was longer than 64 cycles. */
TEST(the_draft_trace_child_runs_a_long_idle_phase_before_the_measurement_window) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 6;
    f.cfg.draft_policy = DraftPolicyKind::Cost;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 6;
    activate_costs(f);
    CHECK_OK(f.start(DraftReplay::None));

    for (int i = 0; i < 200; ++i) CHECK(f.sch.step() == nullptr);

    const uint64_t id = f.submit(32);
    f.prefill(id);
    const int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
    DraftStepEvidence live;
    live.replay_known = true;
    live.replay = DraftReplay::None;
    for (int c = 0; c < 8; ++c) {
        REQUIRE(f.sch.step() != nullptr);
        const DraftPlan* p = f.sch.draft().plan();
        REQUIRE(p != nullptr && p->n_seq == 1);
        f.commit_plain(11);
        CHECK_OK(f.sch.set_draft(id, six, p->max_k));
        f.sch.step_finished(live);
    }
    /* One more open retires the last bracket's cycle, so its sample is delivered and traced. */
    REQUIRE(f.sch.step() != nullptr);

    const SchedMetrics m = f.sch.metrics();
    const int64_t rows = m.draft_cost_priced + m.draft_cost_fallback + m.draft_calib_forced;
    printf("TRACE-CHILD empty=%lld rows=%lld samples=%lld\n", (long long)m.draft_cost_empty,
           (long long)rows, (long long)f.sch.draft_cycles_done());
}

/* Runs this binary with one case selected and `env` set, returning its output. */
std::string run_one_case(const std::string& name, const std::string& env) {
    char exe[4096] = {};
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
    CHECK(n > 0);
    const std::string cmd = "env -u RAD_TEST_FILTER -u RAD_TEST_STRICT -u RAD_TEST_LEVEL "
                            "-u RAD_TEST_CPU_BUDGET -u RAD_TEST_CPU_BUDGET_SCALE " + env +
                            " '" + std::string(exe) + "' '" + name + "' 2>&1";
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, got);
    pclose(p);
    return out;
}

int count_lines_with(const std::string& out, const std::string& needle) {
    int n = 0;
    size_t at = 0;
    while ((at = out.find(needle, at)) != std::string::npos) { ++n; at += needle.size(); }
    return n;
}

/* The trace the diagnosis reads: an idle phase longer than the 64-line window must not spend it,
 * and plan, bracket and sample lines must stay attributable to each other. */
TEST(a_long_idle_phase_does_not_spend_the_draft_trace) {
    const std::string out =
        run_one_case("the_draft_trace_child_runs_a_long_idle_phase_before_the_measurement_window",
                     "RADIANCE_DRAFT_TRACE=1");
    REQUIRE(!out.empty());

    /* The child's own ledger: the idle phase alone opened more cycles than the whole window. */
    const size_t tag = out.find("TRACE-CHILD ");
    REQUIRE(tag != std::string::npos);
    long long empty = 0, rows = 0, samples = 0;
    REQUIRE(std::sscanf(out.c_str() + tag, "TRACE-CHILD empty=%lld rows=%lld samples=%lld",
                        &empty, &rows, &samples) == 3);
    CHECK(empty > 64);               /* without the fix the window was gone before any request */
    CHECK_EQ(rows, 9);               /* the eight loop cycles and the retiring open */
    CHECK_EQ(samples, 8);            /* every loop cycle but the last is delivered */

    /* Plans, brackets and samples: every request-bearing cycle is fully traceable ... */
    CHECK_EQ(count_lines_with(out, "draft trace: plan "), rows);
    /* The retiring open never finishes its step, so it has a plan and no bracket. */
    CHECK_EQ(count_lines_with(out, "draft trace: bracket "), rows - 1);
    CHECK_EQ(count_lines_with(out, "draft trace: sample "), samples);
    /* ... nothing was suppressed ... */
    CHECK(!has(out, "further trace lines are suppressed"));
    /* ... and no empty planning point was traced at all, which is what left the window intact. */
    CHECK(!has(out, "rows=0 "));
}

RAD_TEST_MAIN()
