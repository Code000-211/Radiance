/* scheduler.cpp -- one decision per step. See scheduler.h for the shape and spec §7.1 for why.
 */
#include "scheduler.h"
#include "mm/processor.h"

#include "server/sink.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace rad {

/* Microseconds on the monotonic clock, for the live request table only; the step path keeps its
 * own steady_clock points. */
static int64_t mono_us() {
    return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* Reclaiming KV from the prefix cache for a request: at most kEvictRounds rounds of
 * kEvictBlocks blocks each. Bounded, because an unbounded loop on the scheduler thread stalls the
 * server silently; batched, so a large shortfall does not walk the LRU list block by block. */
static constexpr int kEvictRounds = 256;
static constexpr int kEvictBlocks = 256;

/* The cost trace is a diagnostic for one run, off unless the deployment asks for it by name. Read
 * once: a run must not change what it logs halfway through. */
static bool draft_trace_on() {
    static const bool on = [] {
        const char* v = std::getenv("RADIANCE_DRAFT_TRACE");
        return v && *v && *v != '0';
    }();
    return on;
}

/* A cycle's plan as issued: what the policy chose, why, for which rows, and -- when a cost search
 * ran -- the prices it actually obtained: the issued assignment's own estimate and, per offered
 * depth, whether that alternative had a price at all. A calibration or legacy plan priced nothing:
 * `price=-` and `tiers=[]` say so, and are not `price=invalid` (a search that found none). The
 * price shape itself is logged from the sample, where it is exact. */
void Scheduler::draft_trace_plan(const DraftPlan& p) {
    if (!draft_trace_on() || draft_trace_left_ <= 0) return;
    const DraftPlanner& pl = draft_.planner();
    std::string s;
    char b[256];
    snprintf(b, sizeof b, "cycle=%llu source=%s fallback=%s rows=%d max_k=%d verify_k=%d replay=%s "
             "pipeline=%d prefill=%d price=", (unsigned long long)p.cycle_id,
             draft_plan_source_name(p.source), draft_fallback_name(p.fallback), p.n_seq, p.max_k,
             p.verify_k, draft_replay_name(p.replay), p.pipeline_depth, p.prefill_tokens);
    s += b;
    if (!pl.last_price_known()) {
        s += "-";
    } else if (pl.last_price().valid) {
        snprintf(b, sizeof b, "ok:%.3f+-%.3f", pl.last_price().ms, pl.last_price().unc_ms);
        s += b;
    } else {
        s += "invalid";
    }
    s += " tiers=[";
    const DraftTierPrice* tp = pl.last_tier_prices();
    for (int32_t t = 0; t < pl.last_n_tier_prices(); ++t) {
        if (tp[t].valid) snprintf(b, sizeof b, "%s%d:ok:%.3f+-%.3f", t ? "," : "", tp[t].k,
                                  tp[t].ms, tp[t].unc_ms);
        else            snprintf(b, sizeof b, "%s%d:invalid", t ? "," : "", tp[t].k);
        s += b;
    }
    s += "] k=[";
    const int32_t shown = p.n_seq < 16 ? p.n_seq : 16;
    for (int32_t i = 0; i < shown && p.k; ++i) {
        snprintf(b, sizeof b, "%s%d", i ? "," : "", p.k[i]);
        s += b;
    }
    if (p.n_seq > shown) s += ",+";
    s += "] why=[";
    for (int32_t i = 0; i < shown && p.why; ++i) {
        snprintf(b, sizeof b, "%s%s", i ? "," : "", draft_fallback_name(p.why[i]));
        s += b;
    }
    if (p.n_seq > shown) s += ",+";
    s += "]";
    if (p.n_seq > shown) s += " (first 16 rows)";
    RAD_INFO("draft trace: plan %s", s.c_str());
    if (--draft_trace_left_ == 0)
        RAD_INFO("draft trace: further trace lines are suppressed (64 written)");
}

/* Ctx::StepTape's numbering is not DraftReplay's -- None is 0 there and Unknown is 0 here -- so a
 * cast would rename every class by one. The scheduler has no runtime dependency to import, hence
 * the values by hand; DraftStepEvidence::tape says what they are. */
static const char* step_tape_name(uint8_t t) {
    switch (t) {
        case 1:  return "play";
        case 2:  return "record";
        case 3:  return "audit";
        default: return "none";
    }
}

/* The bracket's own evidence, rank by rank: the merge rule the candidate lives under is only
 * checkable when the ranks that disagreed are visible. */
void Scheduler::draft_trace_bracket(const DraftStepEvidence& ev) {
    if (!draft_trace_on() || draft_trace_left_ <= 0) return;
    std::string s;
    char b[256];
    snprintf(b, sizeof b, "cycle=%llu step=%lld known=%d replay=%s ranks=%d tape=[",
             (unsigned long long)draft_ring_cur_, (long long)step_, ev.replay_known ? 1 : 0,
             draft_replay_name(ev.replay), ev.n_ranks);
    s += b;
    const int32_t n = ev.n_ranks < DraftStepEvidence::kEvidenceRanks
                          ? ev.n_ranks : DraftStepEvidence::kEvidenceRanks;
    for (int32_t i = 0; i < n; ++i) {
        const char* t = ev.tape[i] == DraftStepEvidence::kEvidenceAbsent ? "-"
                        : step_tape_name(ev.tape[i]);
        snprintf(b, sizeof b, "%s%s", i ? "," : "", t);
        s += b;
    }
    snprintf(b, sizeof b, "] staged=%d moved=%d moves_since_bind=%lld", ev.staged ? 1 : 0,
             ev.moved ? 1 : 0, (long long)ev.moves_since_bind);
    s += b;
    RAD_INFO("draft trace: bracket %s", s.c_str());
    if (--draft_trace_left_ == 0)
        RAD_INFO("draft trace: further trace lines are suppressed (64 written)");
}

/* The completed price shape a sample was filed (or refused) under, so a plan can be followed to
 * the bucket that answered it. The whole key is printed: the verify window (`verify_rows`), the
 * executed extent (`round_rows`), pipeline depth, sampling, replay and the loaded drafter beside
 * the plan's own distribution (`draft_rows`, provenance and upper bound), plus what observe() did
 * with the sample and the bucket it landed in. `disposition` is the per-cycle record of a reference
 * discard that the cumulative counter can only count. */
void Scheduler::draft_trace_sample(const DraftCostSample& s, int rc) {
    if (!draft_trace_on() || draft_trace_left_ <= 0) return;
    const DraftVerifyShape& sh = s.shape;
    const DraftCostObserve& ob = draft_costs_.last_observe();
    std::string line;
    char b[256];
    snprintf(b, sizeof b, "cycle=%llu rc=%s flags=0x%x disposition=%s replay=%s sampling=%s "
             "drafter=%s pipeline=%d n_rows=%d ctx_class=%d verify_k=%d verify_tokens=%d "
             "history=%d prefill=%d draft_k=%d rounds=%d drafted=%d plan_rows_total=%d ",
             (unsigned long long)s.cycle_id, rc == RAD_OK ? "ok" : rad_strerror(rc),
             (unsigned)s.flags, draft_cost_disposition_name(ob.what), draft_replay_name(s.replay),
             draft_sampling_name(sh.sampling), draft_drafter_name(sh.drafter), sh.pipeline_depth,
             sh.n_rows, sh.ctx_class, sh.verify_k, sh.verify_tokens, sh.history_tokens,
             sh.prefill_tokens, sh.draft_k, s.rounds, s.drafted, s.draft_rows_total);
    line += b;
    line += "verify_rows=[";
    for (int32_t j = 1; j <= sh.verify_k && j <= RAD_SCHED_MAX_SPEC; ++j) {
        snprintf(b, sizeof b, "%s%d", j > 1 ? "," : "", sh.rows[j]);
        line += b;
    }
    line += "] draft_rows=[";
    for (int32_t j = 0; j <= sh.draft_k && j <= RAD_SCHED_MAX_SPEC; ++j) {
        snprintf(b, sizeof b, "%s%d", j ? "," : "", sh.draft_rows[j]);
        line += b;
    }
    line += "] round_rows=[";
    for (int32_t j = 1; j <= s.rounds && j <= RAD_SCHED_MAX_SPEC; ++j) {
        snprintf(b, sizeof b, "%s%d", j > 1 ? "," : "", s.draft_round_rows[j]);
        line += b;
    }
    line += "]";
    if (ob.what == DraftCostDisposition::Recorded && ob.bucket >= 0) {
        snprintf(b, sizeof b, " bucket=%d n=%d mean=%.3f+-%.3f", ob.bucket, ob.n, ob.mean_ms,
                 ob.unc_ms);
    } else {
        snprintf(b, sizeof b, " bucket=-");
    }
    line += b;
    RAD_INFO("draft trace: sample %s", line.c_str());
    if (--draft_trace_left_ == 0)
        RAD_INFO("draft trace: further trace lines are suppressed (64 written)");
}

/* ------------------------------------------------------------------ ordering */
/* Priority first, then arrival. A preempted request keeps its original arrival number, so it
 * re-enters at the front of its priority class instead of the back. */
bool Scheduler::ahead_of(int32_t a, int32_t b) const {
    const SchedReq& x = *reqs_[(size_t)a];
    const SchedReq& y = *reqs_[(size_t)b];
    if (x.req->priority != y.req->priority) return x.req->priority > y.req->priority;
    return x.arrival < y.arrival;
}

void Scheduler::queue_insert(std::vector<int32_t>& q, int32_t slot) {
    auto it = std::lower_bound(q.begin(), q.end(), slot,
                               [&](int32_t a, int32_t b) { return ahead_of(a, b); });
    q.insert(it, slot);   /* capacity is reserved, so this moves and does not allocate */
}

void Scheduler::queue_erase(std::vector<int32_t>& q, int32_t slot) {
    auto it = std::find(q.begin(), q.end(), slot);
    if (it != q.end()) q.erase(it);
}

/* ------------------------------------------------------------------ init */
int Scheduler::init(Program& prog, const Config& cfg, IKVManager* kv, IPrefixCache* pc,
                    const std::vector<BatchBuilder::RankIO>& ranks, const ChunkGeometry* geo,
                    DraftReplay pass_replay) {
    if (!kv || !pc) return RAD_E_INVAL;
    prog_ = &prog;
    cfg_ = cfg;
    /* Nothing has run yet, so the candidate is the capability answer: `Unknown` on a recording
     * deployment until a bracket completes, `None` where every pass is provably unkeyed. */
    draft_cycle_replay_ = pass_replay;
    kv_ = kv;
    pc_ = pc;

    /* The engine resolves the geometry once, because the prefix cache needs its checkpoint
     * interval first; a caller without one (tests) resolves its own from prog and cfg. Both
     * must use the resolved interval, never the raw flag, so chunks split in one place. */
    if (geo) geo_ = *geo;
    else RAD_TRY(chunk_geometry_resolve(prog, cfg, &geo_));

    kvgeom_ = KVGeom::of(prog.kv_groups);
    for (size_t i = 0; i < kvgeom_.g.size(); ++i) {
        const int64_t w = kv->state_index_width((int32_t)i);
        if (w > 0) kvgeom_.g[i].sidx_width = w;
    }
    has_linear_state_ = false;
    for (const auto& g : kvgeom_.g) if (!g.paged) has_linear_state_ = true;

    /* A hit can leave a drafter's own cache short only where a paged group is not prefix-cached
     * (scheduler.h, hit_holes_draft_). With the cache off cached_groups() is empty and this is
     * true, which costs nothing: no hits are taken, and every first block pass is at ctx_len 0. */
    {
        /* Paged as the cache counts it (KVGroupPlan::paged(), via block_tokens()), not kvgeom_'s
         * `paged` flag, which is also true for an empty placeholder group and would pin this
         * predicate true. */
        size_t n_paged = 0;
        for (size_t i = 0; i < kvgeom_.g.size(); ++i)
            if (kv->block_tokens((int32_t)i) > 0) ++n_paged;
        hit_holes_draft_ = pc->cached_groups().size() < n_paged;
        RAD_INFO("draft floor: %zu paged group(s), %zu prefix-cached -> a hit %s",
                 n_paged, pc->cached_groups().size(),
                 hit_holes_draft_ ? "leaves a hole in a drafter's own KV" : "leaves no hole");
    }
    /* The trunk's share: a paged group a trunk layer reads (a bound layer below the trunk's layer
     * count; a drafter borrows from base_layer up) that the cache does not index. Recomputing the
     * tail cannot repair that hole; see scheduler.h. */
    hit_holes_trunk_ = false;
    for (size_t i = 0; i < prog.kv_groups.size(); ++i) {
        if (kv->block_tokens((int32_t)i) <= 0) continue;
        bool trunk = false;
        for (int l : prog.kv_groups[i].layers) if (l < prog.meta.n_layers) { trunk = true; break; }
        if (!trunk) continue;
        const std::vector<int32_t>& cg = pc->cached_groups();
        if (std::find(cg.begin(), cg.end(), (int32_t)i) == cg.end()) { hit_holes_trunk_ = true; break; }
    }
    if (hit_holes_trunk_ && cfg.prefix_cache)
        RAD_INFO("prefix cache: a trunk layer's windowed group is not prefix-cached -- a hit "
                 "resumes from a window snapshot (a checkpoint slot) or re-prefills");
    if (cfg.prefill_interleave > 0)
        RAD_INFO("scheduler: --prefill-interleave %lld: after a prefill chunk that leaves its "
                 "prompt unfinished while other sequences decode, the next %lld steps carry "
                 "decode rows only and admit nothing", (long long)cfg.prefill_interleave,
                 (long long)cfg.prefill_interleave);
    interleave_left_ = 0;
    held_prefill_ = 0;
    draft_refill_ = 0;
    if (cfg.n_spec > 0 && prog.drafter.kind == RAD_DRAFT_BLOCK && prog.drafter.window > 0 &&
        hit_holes_draft_) {
        draft_refill_ = prog.drafter.window;
        RAD_INFO("draft refill: a prefix hit ends %lld tokens before the prompt's end, so the "
                 "drafter's window is recomputed", (long long)draft_refill_);
    }

    serial_shared_    = cfg.n_spec > 0 && prog.drafter.kind == RAD_DRAFT_SERIAL &&
                        prog.drafter_shared_kv;
    serial_rot_shift_ = serial_shared_ ? prog.drafter_pos_offset : 0;
    if (serial_shared_)
        RAD_INFO("serial drafter: reads the trunk's cache and keeps none -- no history pass, one "
                 "row a drafting sequence, every round at the committed position%s",
                 serial_rot_shift_ ? " (rotated one past it)" : "");

    /* docs/ADAPTIVE-MTP.md: `cost` is accepted only with a serial draft head and a usable ceiling.
     * Refused by name here, where both facts are known, rather than silently mapped to legacy. */
    if (cfg.draft_policy == DraftPolicyKind::Cost &&
        (prog.drafter.kind != RAD_DRAFT_SERIAL || prog.drafter.depth <= 0 || cfg.n_spec <= 0)) {
        RAD_ERR("--draft-policy cost: needs a serial draft head and a positive ceiling (drafter "
                "kind %d, depth %d, ceiling %d)", (int)prog.drafter.kind,
                (int)prog.drafter.depth, cfg.n_spec);
        return RAD_E_UNSUPPORTED;
    }
    /* ... and a baseline admitted for THIS runtime. DraftCostTable::configure() is the only way a
     * table becomes active, and it refuses every fingerprint but its own, so `active()` is the
     * proof that a profile was loaded and matched. An empty matching profile -- or one whose
     * buckets are all diagnostically disturbed -- is not yet a price: the policy would issue its
     * named fallback for every shape, so the deployment has to say so with admit_fallback().
     * has_usable_bucket() is the same rule estimate() applies, so a profile of warmup/record/audit
     * buckets is refused here rather than admitted on a non-empty list. Without either, every
     * price would be a guess and Cost is refused by name rather than run on unknowns
     * (docs/ADAPTIVE-MTP.md). The table is configured before init; the conditional clear below
     * keeps it, since it is the baseline this deployment was admitted with. */
    if (cfg.draft_policy == DraftPolicyKind::Cost &&
        !(draft_costs_.active() &&
          (draft_costs_.has_usable_bucket() || draft_costs_.fallback_admitted()))) {
        if (!draft_costs_.active() && !draft_costs_.reason().empty())
            RAD_ERR("--draft-policy cost: no admissible cost baseline -- %s",
                    draft_costs_.reason().c_str());
        else if (draft_costs_.active())
            RAD_ERR("--draft-policy cost: the profile for this runtime has no measured bucket -- "
                    "admit the fallback baseline explicitly before init");
        else
            RAD_ERR("--draft-policy cost: no admissible cost baseline -- configure a profile for "
                    "this runtime (model, quant, tp wire, gpus, kernels, kv format, placement) "
                    "before init");
        return RAD_E_UNSUPPORTED;
    }

    /* The calibration's recording (--draft-profile-out): the same ring the cost policy uses, opened
     * under whatever policy this deployment really runs. It never sets the controller's plan, so
     * draft_cap()/bars_fill() and the Cost lock are exactly what they are without it. A cost
     * calibration runs with a record-only table (engine.cpp), so it opens the same cycles as any
     * cost deployment -- including the start cycle -- but issues the fallback baseline and prices
     * nothing. The file can name only the serial head, so a deployment without one cannot be
     * recorded (refused by name there); the depth is the deepest round the head can publish. */
    draft_record_ = !cfg.draft_profile_out.empty();
    draft_head_depth_ = 0;
    draft_record_prev_k_ = -1;
    if (draft_record_)
        draft_head_depth_ = prog.drafter.depth < cfg.n_spec ? prog.drafter.depth : cfg.n_spec;

    DraftController::Params dp;
    dp.max_depth = cfg.n_spec;
    /* The plan's caller-provided arrays are sized at max_seqs (draft_types.h). */
    dp.max_seqs = cfg.max_seqs;
    /* The cost policy's two hooks: B's predictor and A's table. Wired here, owned by the
     * scheduler, and read by DraftController::open_cycle every cycle -- not a module that exists
     * beside the step path. `max_seqs` is the slot count both the prediction table and the plan
     * arrays are sized at. */
    dp.policy.mode = cfg.draft_policy == DraftPolicyKind::Cost ? DraftPolicyMode::Cost
                                                           : DraftPolicyMode::Legacy;
    /* --draft-depth-tiers: what the cost choice weighs. The executable head depth is the bound, not
     * the ceiling: cap_of() bounds a plan only by the request's remaining output and the ceiling, so
     * a tier above the head's own depth would be planned there, execute fewer rounds, and be filed
     * as kDraftSampleShortened -- a bucket no reader may use. An explicit list above that depth is
     * refused by name; the default ladder is trimmed to it (and its deepest value replaced by the
     * depth itself), so a deployment whose head is shallower than the tuned ladder still runs the
     * tiers it can execute. */
    int32_t exec_depth = cfg.n_spec;
    if (cfg.draft_policy == DraftPolicyKind::Cost) {
        exec_depth = prog.drafter.depth < cfg.n_spec ? prog.drafter.depth : cfg.n_spec;
        if (cfg.draft_tiers_set) {
            for (int i = 0; i < cfg.draft_n_tiers; ++i) {
                if (cfg.draft_tiers[i] > exec_depth) {
                    RAD_ERR("--draft-depth-tiers: tier %d is above the executable head depth %d "
                            "(ceiling %d, drafter depth %d)", (int)cfg.draft_tiers[i],
                            (int)exec_depth, (int)cfg.n_spec, (int)prog.drafter.depth);
                    return RAD_E_UNSUPPORTED;
                }
            }
        }
    }
    /* The explicit ladder is the whole set the cost policy may offer, and the named baseline runs
     * at exec_depth. A top below exec_depth is left as given: a profile that prices that top is a
     * valid reload, and the policy refuses a cycle it cannot price when the baseline would run
     * deeper (draft_policy.cpp), rather than extending the list or moving the baseline here. */
    dp.policy.n_tiers = 0;
    for (int i = 0; i < cfg.draft_n_tiers && dp.policy.n_tiers < (int)RAD_DRAFT_MAX_TIERS; ++i) {
        int32_t t = cfg.draft_tiers[i];
        if (cfg.draft_policy == DraftPolicyKind::Cost && t > exec_depth) t = exec_depth;
        if (dp.policy.n_tiers > 0 && t <= dp.policy.tiers[dp.policy.n_tiers - 1]) continue;
        dp.policy.tiers[dp.policy.n_tiers++] = t;
    }
    if (cfg.draft_policy == DraftPolicyKind::Cost && !cfg.draft_tiers_set) {
        int32_t top = dp.policy.tiers[dp.policy.n_tiers - 1];
        if (top < exec_depth) {
            /* The ladder ended below what the head can execute: offer the executable depth as the
             * batch's deepest plan, so the coverage gate (the last tier) is the real choice. */
            if (dp.policy.n_tiers < (int)RAD_DRAFT_MAX_TIERS)
                dp.policy.tiers[dp.policy.n_tiers++] = exec_depth;
            else
                dp.policy.tiers[dp.policy.n_tiers - 1] = exec_depth;
        }
        if (dp.policy.tiers[dp.policy.n_tiers - 1] != cfg.draft_tiers[cfg.draft_n_tiers - 1])
            RAD_INFO("--draft-depth-tiers: default ladder trimmed to the executable head depth %d "
                     "(deepest tier %d)", (int)exec_depth,
                     (int)dp.policy.tiers[dp.policy.n_tiers - 1]);
    }
    /* With no profile loaded nothing can be priced, so the cycle is the named fallback plan; its
     * depth is the executable one, or a cost deployment would plan past what the head can run. */
    dp.policy.fallback_k = exec_depth;
    /* --draft-calib-k: copied as given; DraftPlanner::configure() checks every value against the
     * effective ladder above (after the trim and dedup), which is the list the serving policy
     * offers. Engine::begin_draft_calibration refuses the flag outside a reference calibration. */
    for (int i = 0; i < cfg.draft_calib_n; ++i)
        dp.policy.calib_k[i] = cfg.draft_calib_k[i];
    dp.policy.n_calib_k = cfg.draft_calib_n;
    if (cfg.max_seqs > 0) {
        draft_predict_state_.assign((size_t)cfg.max_seqs, DraftPredictState{});
        DraftPredictModel::Params pp;
        pp.max_depth = cfg.n_spec;
        pp.version   = RAD_DRAFT_POLICY_VERSION;
        RAD_TRY(draft_predict_.configure(pp, draft_predict_state_.data(),
                                         (int32_t)draft_predict_state_.size()));
        dp.predictor = DraftPredictor{ &draft_predict_, &DraftPredictModel::predict_hook };
    }
    /* No profile is loaded, so every price is invalid and the policy issues its named fallback;
     * the table is still the handle open_cycle prices through. A table configured before init is
     * the baseline a Cost deployment was admitted with -- keep it. */
    if (!draft_costs_.active()) draft_costs_.clear();
    dp.costs = draft_costs_.handle();
    /* The ring's storage: fixed at init so a cycle never allocates one. */
    for (int32_t i = 0; i < (int32_t)kDraftRing; ++i) {
        DraftCycleRecord& r = draft_ring_[i];
        r.id.assign((size_t)cfg.max_seqs, 0);
        r.slot.assign((size_t)cfg.max_seqs, -1);
        r.gen.assign((size_t)cfg.max_seqs, 0);
        r.k.assign((size_t)cfg.max_seqs, 0);
        r.fill.assign((size_t)cfg.max_seqs, DraftDrafter::None);
        r.ctx_class.assign((size_t)cfg.max_seqs, 0);
        r.history_tokens.assign((size_t)cfg.max_seqs, 0);
        r.seq.assign((size_t)cfg.max_seqs, DraftSeqObservation{});
        r.sampled.assign((size_t)cfg.max_seqs, 0);
        r.state.assign((size_t)cfg.max_seqs, 0);
        r.plan.cap      = cfg.max_seqs;
        r.plan.req_id   = r.id.data();
        r.plan.slot     = r.slot.data();
        r.plan.slot_gen = r.gen.data();
        r.plan.k        = r.k.data();
        r.plan.why      = nullptr;   /* not part of the observation's plan */
        r.plan.rows     = r.rows;
        r.plan.ctx_class      = r.ctx_class.data();
        r.plan.history_tokens = r.history_tokens.data();
        r.plan.fill           = r.fill.data();
    }
    /* The adaptive window (draft.h) prices a draft by the measured cost of the step it adds to.
     * A block drafter proposes its block in one pass, but each verified draft is still a verify
     * row, which is not free on a MoE that reads each row's routed experts. So it takes the same
     * policy with fixed rounds: each sequence verifies the first min(D, its stop) of the block.
     * Prompt lookup runs no pass and keeps the fixed window; the outcome is logged either way. */
    if (cfg.draft_window == DraftWindow::Adaptive) {
        const RadDrafterDecl& dd = prog.drafter;
        if (cfg.n_spec <= 0)
            RAD_WARN("--draft-window adaptive: this deployment does not speculate "
                     "(--num-speculative-tokens %d), so there is no window to adapt", cfg.n_spec);
        else if ((dd.kind != RAD_DRAFT_SERIAL && dd.kind != RAD_DRAFT_BLOCK) || dd.depth <= 0)
            RAD_WARN("--draft-window adaptive: the window is fixed at %d with prompt lookup only "
                     "-- the adaptive window prices draft passes and verify rows, and prompt "
                     "lookup runs no pass", cfg.n_spec);
        else {
            dp.adapt = true;
            if (dd.kind == RAD_DRAFT_BLOCK)
                RAD_INFO("--draft-window adaptive with a block drafter: the block of %d is "
                         "proposed whole, and each sequence verifies the first min(D, its stop) "
                         "of it", dd.depth);
        }
    }
    RAD_TRY(draft_.configure(dp));
    draft_est_.clear();
    draft_est_.reserve((size_t)cfg.max_seqs);
    /* The cost policy's per-cycle storage, sized here so nothing on the step path allocates. */
    draft_view_seq_.assign((size_t)cfg.max_seqs, DraftSeqView{});
    draft_pred_seq_.assign((size_t)cfg.max_seqs, DraftPrediction{});
    draft_pred_id_.assign((size_t)cfg.max_seqs, 0);
    draft_view_ = DraftBatchView{};
    draft_view_.cap = (int32_t)draft_view_seq_.size();
    draft_view_.seq = draft_view_seq_.data();
    draft_preds_ = DraftPredictions{};
    draft_preds_.cap = (int32_t)draft_pred_seq_.size();
    draft_preds_.req_id = draft_pred_id_.data();
    draft_preds_.seq = draft_pred_seq_.data();

    RAD_TRY(builder_.init(prog, cfg, geo_, kvgeom_, ranks));
    /* The builder's bound, not a second derivation: the block tables were sized for it. */
    ctx_limit_ = builder_.max_ctx();
    /* The pool's own bound on one sequence, searched once (can_ever_fit is monotonic in the
     * length): add_locked refuses a prompt past it. */
    pool_tokens_ = -1;
    pool_refusal_.clear();
    if (const int64_t top = ctx_limit_ > 0 ? ctx_limit_ : ((int64_t)1 << 40);
        !kv->can_ever_fit(top)) {
        int64_t lo = 0, hi = top;
        if (!kv->can_ever_fit(0)) hi = 0;
        while (hi - lo > 1) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (kv->can_ever_fit(mid)) lo = mid; else hi = mid;
        }
        pool_tokens_ = hi == 0 ? 0 : lo;
        /* Worded for clients that compact a conversation on a context overflow by its message. */
        pool_refusal_ = fmt("This model's maximum context length is %lld tokens, all the KV cache "
                            "holds for one sequence: the prompt exceeds the available context "
                            "size; shorten it or give the cache more room",
                            (long long)pool_tokens_);
    }
    n_deferred_ = 0;
    closed_ = false;
    closed_status_ = RAD_E_STATE;

    /* Everything the step path touches is reserved here, so a step never calls the allocator. */
    plan_.e.reserve((size_t)cfg.max_seqs);
    part_.reserve((size_t)cfg.max_seqs);
    running_.reserve((size_t)cfg.max_seqs);
    waiting_.reserve((size_t)cfg.max_seqs * 4 + 64);
    free_slots_.reserve((size_t)cfg.max_seqs);
    free_cards_.resize((size_t)cfg.max_seqs);
    for (int64_t c = 0; c < cfg.max_seqs; ++c)
        free_cards_[(size_t)c] = (int32_t)(cfg.max_seqs - 1 - c);   /* 0 is taken first */
    reqs_.reserve((size_t)cfg.max_seqs);
    completed_.reserve((size_t)cfg.max_seqs);
    committed_.reserve((size_t)cfg.max_seqs);
    by_id_.reserve((size_t)cfg.max_seqs * 2);
    hit_blocks_.resize(prog.kv_groups.size());
    for (auto& v : hit_blocks_) v.reserve(64);
    return RAD_OK;
}

void Scheduler::fini() {
    /* A scheduler torn down without shutdown() still owes its last completed cycle to the
     * predictor. No-op when shutdown() already flushed. */
    draft_ring_flush();
    for (auto& s : reqs_)
        if (s && s->req && (s->running || s->pending_free)) kv_->free_sequence(s->req->id);
    builder_.fini();
    reqs_.clear();
    free_slots_.clear();
    free_cards_.clear();
    by_id_.clear();
    running_.clear();
    waiting_.clear();
}

/* ------------------------------------------------------------------ arrival */
int Scheduler::add(std::unique_ptr<Request> r) {
    std::lock_guard<std::mutex> lk(mu_);
    return add_locked(std::move(r));
}

/* One lock for all of them, so the step that admits the first finds them all waiting. Otherwise a
 * step could start between two adds and admit a timing-dependent subset, and the batch (and its
 * tie-level rounding) would vary between identical runs. */
void Scheduler::add_all(std::unique_ptr<Request>* rs, size_t n, int* rc) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < n; ++i) rc[i] = add_locked(std::move(rs[i]));
}

int Scheduler::add_locked(std::unique_ptr<Request> r) {
    if (!r) return RAD_E_INVAL;
    if (r->prompt.empty()) return RAD_E_INVAL;
    if (by_id_.count(r->id)) return RAD_E_DUPLICATE;
    /* An embedding input runs in one step or not at all (Request::embed); one longer than a step
     * could never be admitted. */
    if (r->embed && (int64_t)r->prompt.size() > cfg_.max_tok) return RAD_E_INVAL;
    /* Nor could a prompt the KV pool cannot hold even alone: admission takes only what the pool
     * holds, so it would wait at the head of the queue, and everything behind it, forever. The
     * prompt is the least a request needs: its first token comes from the prefill's last row.
     * RAD_E_FULL with the sink failed here, so the caller answers it as a context overflow. */
    if (!kv_->can_ever_fit((int64_t)r->prompt.size())) {
        RAD_WARN("request %llu refused: its %zu-token prompt is longer than the KV cache can hold "
                 "for one sequence (%lld tokens)", (unsigned long long)r->id, r->prompt.size(),
                 (long long)pool_tokens_);
        server::sink_fail(r->sink, RAD_E_FULL,
                          pool_refusal_.empty() ? "the prompt is longer than the KV cache can "
                                                  "hold for one sequence"
                                                : pool_refusal_.c_str());
        return RAD_E_FULL;
    }
    /* A scored prompt names real vocabulary rows and needs one step's decision; drafts would be
     * proposed for a continuation nobody reads. */
    if (r->token_scores) {
        if (r->embed || r->token_scores->tokens.empty()) return RAD_E_INVAL;
        for (int32_t t : r->token_scores->tokens)
            if (t < 0 || t >= prog_->meta.n_vocab) return RAD_E_INVAL;
        r->may_speculate = false;
    }
    /* The engine has stopped. Fail the sink, which is what the caller listens to, so it is not
     * left waiting on a step that never comes. */
    if (closed_) {
        RAD_WARN("request %llu refused: %s", (unsigned long long)r->id, closed_reason_);
        server::sink_fail(r->sink, closed_status_, closed_reason_);
        return RAD_E_STATE;
    }

    int32_t slot;
    if (!free_slots_.empty()) { slot = free_slots_.back(); free_slots_.pop_back(); }
    else { slot = (int32_t)reqs_.size(); reqs_.emplace_back(); }
    if (!reqs_[(size_t)slot]) reqs_[(size_t)slot] = std::make_unique<SchedReq>();

    SchedReq& s = *reqs_[(size_t)slot];
    s.req = std::move(r);
    s.arrival = ++arrival_;
    s.gen = ++slot_gen_;
    s.arrived_us = mono_us();
    s.first_tok_us = 0;
    s.slot = slot;
    s.card = -1;
    s.running = false;
    s.drafter_reset();
    /* A new request's text, so a new estimate. */
    s.adapt.depth = 0;
    s.adapt.held  = 0;
    s.sched_tokens = 0;
    s.entry = -1;
    s.entry_next = -1;
    s.pending_free = false;
    s.reap_pending = false;
    s.ckpt_src = -1;
    s.hit_attn = 0;
    s.hit_owed = false;
    s.req->state = ReqState::Waiting;

    /* The step path appends to `output`, so its capacity is settled here, on the arrival thread.
     * Bounded by the context's remaining room: reserving a caller's max_tokens as given could ask
     * for gigabytes. */
    int64_t out_cap = s.req->max_tokens > 0 ? (int64_t)s.req->max_tokens : 1;
    const int64_t ctx_room = ctx_limit_ - (int64_t)s.req->prompt.size();
    if (ctx_limit_ > 0 && out_cap > ctx_room) out_cap = ctx_room > 1 ? ctx_room : 1;
    s.req->output.reserve((size_t)out_cap + (size_t)RAD_SCHED_MAX_SPEC);

    by_id_[s.req->id] = slot;
    /* Enough headroom that a step preempting every running request does not grow the queue under
     * the lock. */
    if (waiting_.capacity() < waiting_.size() + (size_t)cfg_.max_seqs + 1)
        waiting_.reserve(waiting_.size() + (size_t)cfg_.max_seqs + 1);
    queue_insert(waiting_, slot);
    return RAD_OK;
}

Request* Scheduler::find(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    return it == by_id_.end() ? nullptr : reqs_[(size_t)it->second]->req.get();
}

bool Scheduler::running(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return false;
    const SchedReq& s = *reqs_[(size_t)it->second];
    return s.req && s.req->state == ReqState::Running;
}

bool Scheduler::committed_seq(uint64_t seq, uint64_t* n_out, bool* live) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const CommittedSeq& c : committed_)
        if (c.seq == seq) {
            if (n_out) *n_out = c.n_out;
            if (live) *live = c.live;
            return true;
        }
    return false;
}

int Scheduler::set_draft(uint64_t id, const int32_t* tokens, int n) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return RAD_E_NOTFOUND;
    SchedReq& s = *reqs_[(size_t)it->second];
    if (n < 0) return RAD_E_INVAL;

    /* The step issued ahead already verifies these: it was planned with the drafter's count and
     * the card has the tokens. The host copy is what its commit emits accepted tokens from, so it
     * must match that count, unclamped. Under the adaptive window the rounds ran for the deepest
     * sequence and expect_draft() cut this one's count to its window, so a longer proposal is cut
     * to the planned count (the window may have moved at the commit since). */
    if (next_open_ && s.entry_next >= 0) {
        /* What the head executed, before this row's verify window cuts what may be published: the
         * rounds ran for the row (serial_begin) and only the verify is cut here. The ring keeps
         * both, so a cycle whose execution the plan cannot express is refused rather than priced
         * from the truncated publication. */
        const int32_t ran = n;
        if ((draft_.adaptive() || draft_.cost_policy()) && n > s.draft_next) n = s.draft_next;
        if (n != s.draft_next) {
            RAD_ERR("request %llu: %d draft(s) published for a step planned to verify %d",
                    (unsigned long long)id, n, s.draft_next);
            return RAD_E_STATE;
        }
        for (int i = 0; i < n; ++i) s.draft[i] = tokens[i];
        s.n_draft = n;
        s.req->n_draft = n;
        s.req->draft = n ? s.draft : nullptr;
        draft_ring_publish(s, ran, n, DraftOrigin::Head);
        return RAD_OK;
    }

    /* An ineligible request gets no proposal rather than a refusal, so it decodes one token a step
     * as without a drafter. See Request::may_speculate, and step() for the gate that also catches
     * the n-gram fallback. */
    if (!s.req->may_speculate) {
        s.n_draft = 0; s.req->n_draft = 0; s.req->draft = nullptr;
        return RAD_OK;
    }

    int cap = draft_cap(s);
    if (n > cap) n = cap;
    if (n > RAD_SCHED_MAX_SPEC) n = RAD_SCHED_MAX_SPEC;
    for (int i = 0; i < n; ++i) s.draft[i] = tokens[i];
    s.n_draft = n;
    s.req->n_draft = n;
    s.req->draft = s.draft;
    draft_ring_publish(s, n, n, DraftOrigin::Head);
    return RAD_OK;
}

/* ------------------------------------------------------------------ cancellation */
/* ------------------------------------------------------------------ the sink
 *
 * `Request::sink` is the server's per-request channel (core/server/sink.h), a `void*` so core/sched
 * does not depend on server types; null means nobody is listening. Tokens reach the client only
 * through a push, and every sink_* call is made from this file. A push returns false only on
 * overflow, after the sink has already failed the request, so false means stop generating. */
static server::Finish finish_of(ReqState st, const std::string& reason) {
    if (st == ReqState::Cancelled) return server::Finish::Cancelled;
    if (st == ReqState::Failed)    return server::Finish::Error;
    if (reason == "length")        return server::Finish::Length;
    return server::Finish::Stop;
}

/* The cache may not own the whole pool: it holds a reference on every block it indexes, and an
 * unbounded cache keeps the pool full, so admission pays eviction on every chunk and a run of
 * distinct large prompts wedges. A fraction of the pool is kept free (what it protects is the
 * ability to admit any sequence), trimmed after a publish rather than during admission.
 *
 * Live sequences count against the reserve too and can put it out of reach; evicting in LRU
 * order until it is met would drop shared entries (freeing nothing) and the conversation just
 * published. So the trim asks only for the blocks it is short, and the cache gives only what
 * frees a block without losing a conversation (PrefixCache::trim). */

/* Drop every entry, in VRAM or not (IPrefixCache::reset): an eviction does not forget an entry
 * the idle tiers hold a copy of. */
int64_t Scheduler::reset_prefix_cache() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!pc_ || !cfg_.prefix_cache) return RAD_E_UNSUPPORTED;
    const int64_t dropped = pc_->reset();
    RAD_INFO("prefix cache: reset, %lld entries dropped", (long long)dropped);
    return dropped;
}

void Scheduler::trim_prefix_cache() {
    if (!pc_ || !cfg_.prefix_cache) return;
    const std::vector<int32_t>& cg = pc_->cached_groups();
    if (cg.empty()) return;
    const int32_t g = cg[0];
    const int64_t total = kv_->total_blocks(g);
    if (total <= 0) return;
    const int64_t reserve = total / 4;              /* a quarter of the pool stays free */
    /* Free against the carve, not free_blocks(): on an elastic pool free_blocks() is bounded by
     * the live mark, so a pool that returned its unused top would read as nearly full and the
     * trim would drain the cache for room the pool can grow back into. */
    const int64_t short_by = reserve - (total - kv_->held_blocks(g));
    if (short_by > 0) (void)pc_->trim(short_by);
}

void Scheduler::close_request(SchedReq& s, ReqState st, const char* reason, bool defer_free,
                              bool publish, int status, const char* why) {
    /* Publish before freeing: this is the only moment the whole prefix is both computed and still
     * owned. insert() takes its own KV references, so the blocks survive the free below. Cancelled
     * and failed requests are published too; their prompts were computed, and a client that hangs
     * up mid-answer is likely to ask for that prefix again. */
    if (publish && cfg_.prefix_cache && pc_ && kv_->has_sequence(s.req->id) &&
        s.req->n_computed > 0) {
        const std::vector<int32_t>& cg = pc_->cached_groups();
        pub_blocks_.assign(cg.size(), {});
        bool ok = !cg.empty();
        for (size_t i = 0; i < cg.size(); ++i) {
            /* A rebased table does not describe the prefix from position 0, so it cannot be
             * published. cached_groups() already excludes windowed groups; this is a backstop. */
            if (kv_->first_block_pos(s.req->id, cg[i]) != 0) { ok = false; break; }
            pub_blocks_[i] = kv_->block_table(s.req->id, cg[i]);
        }
        if (ok) {
            const int rc = pc_->insert(*s.req, pub_blocks_, s.req->n_computed);
            if (rc < 0)
                RAD_DEBUG("prefix cache: request %llu not published: %s",
                          (unsigned long long)s.req->id, rad_strerror(rc));
            trim_prefix_cache();
        }
    }

    if (cfg_.prefix_cache && pc_) pc_->release(*s.req);
    /* Fold what the draft head did on this text into the next text's starting estimate, once:
     * marking it unstarted makes a second close add nothing. */
    if (draft_.adaptive()) {
        draft_.adapt_retire(s.adapt);
        s.adapt.depth = 0;
    }

    if (defer_free) {
        if (!s.pending_free) ++n_deferred_;
        s.pending_free = true;
    } else if (kv_->has_sequence(s.req->id)) {
        /* A preempted request has had its blocks taken, but the manager may still know the
         * sequence and a state slot; asking first avoids relying on free_sequence() being
         * idempotent. */
        kv_->free_sequence(s.req->id);
    }
    if (!defer_free) drop_card(s);
    s.running = false;
    s.req->state = st;
    /* A cycle this request was still part of can never complete: the step that would verify its
     * proposals no longer runs for it. Dropped, so the pending record is retired instead of
     * holding a slot until the ring evicts it. */
    draft_ring_drop_row(s.req->id, s.slot, s.gen);
    if (s.req->finish_reason.empty() && reason) s.req->finish_reason = reason;
    queue_erase(running_, s.slot);
    queue_erase(waiting_, s.slot);
    s.n_draft = 0;

    /* Terminal. The sink keeps the first finish, so a cancel racing a natural stop yields one
     * reason. This is the producer's last touch of the sink, which the HTTP thread waits on. */
    if (st == ReqState::Failed && status < 0) server::sink_fail(s.req->sink, status, why);
    else server::sink_finish(s.req->sink, finish_of(st, s.req->finish_reason));

    if (st == ReqState::Failed)         ++m_.failed;
    else if (st == ReqState::Cancelled) ++m_.cancelled;
    else                                ++m_.finished;
    completed_.push_back(s.req->id);
}

/* Reserve a checkpoint slot for the state at `pos`, written after the entry's row `tok`. A
 * refusal leaves the entry without one. */
void Scheduler::reserve_ckpt(StepEntry& e, const Request& r, int64_t pos, int32_t tok) {
    int32_t cs = -1;
    if (pc_->reserve_checkpoint(r, pos, &cs) < 0 || cs < 0) return;
    e.ckpt_tok = tok;
    e.ckpt_slot = cs;
    ++m_.checkpoints_written;
}

int Scheduler::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return RAD_E_NOTFOUND;
    SchedReq& s = *reqs_[(size_t)it->second];
    if (s.req->state == ReqState::Finished || s.req->state == ReqState::Cancelled)
        return RAD_OK;

    /* Blocks go back now (spec §14), unless a step in flight or a drafter snapshot still names
     * the request; then commit or sweep_deferred releases them (SchedReq::pending_free). */
    const bool in_flight = in_step(s) || block_holds(id);
    close_request(s, ReqState::Cancelled, "cancelled", in_flight);
    return RAD_OK;
}

void Scheduler::take_completed(std::vector<uint64_t>* out) {
    if (!out) return;
    out->clear();
    std::lock_guard<std::mutex> lk(mu_);
    if (!next_open_) { out->swap(completed_); return; }
    /* A request the next step still carries is held back: that step's sampler is staged for it
     * after this one commits, so its sampler state must outlive the finish by one step. */
    size_t keep = 0;
    for (uint64_t id : completed_) {
        auto it = by_id_.find(id);
        const bool held = it != by_id_.end() && reqs_[(size_t)it->second]->entry_next >= 0;
        if (held) completed_[keep++] = id;
        else out->push_back(id);
    }
    completed_.resize(keep);
}

void Scheduler::release_locked(SchedReq& s) {
    if (s.pending_free) { free_kv(s); s.pending_free = false; --n_deferred_; }
    if (s.reap_pending) { s.reap_pending = false; --n_deferred_; }
    by_id_.erase(s.req->id);
    s.req.reset();
    free_slots_.push_back(s.slot);
}

void Scheduler::free_kv(SchedReq& s) {
    kv_->free_sequence(s.req->id);
    drop_card(s);
}

void Scheduler::drop_card(SchedReq& s) {
    if (s.card < 0) return;
    free_cards_.push_back(s.card);
    s.card = -1;
}

bool Scheduler::block_holds(uint64_t id) const {
    for (const BlockSeq& d : block_) if (d.seq == id) return true;
    for (const SerialSeq& m : serial_) if (m.seq == id) return true;
    return false;
}

const Request* Scheduler::live_req(int32_t slot, uint64_t id) const {
    if (slot < 0 || (size_t)slot >= reqs_.size() || !reqs_[(size_t)slot]) return nullptr;
    const SchedReq& s = *reqs_[(size_t)slot];
    return (s.req && s.req->id == id) ? s.req.get() : nullptr;
}

void Scheduler::sweep_deferred() {
    if (n_deferred_ <= 0) return;
    for (auto& up : reqs_) {
        if (!up) continue;
        SchedReq& s = *up;
        if (!s.req || (!s.reap_pending && !s.pending_free)) continue;
        if (in_step(s)) continue;
        if (block_holds(s.req->id)) continue;
        if (s.reap_pending) { release_locked(s); continue; }
        /* A free deferred on a drafter snapshot, for a request not yet reaped: only the blocks
         * go back. */
        free_kv(s);
        s.pending_free = false;
        --n_deferred_;
    }
}

bool Scheduler::has_deferred() const {
    std::lock_guard<std::mutex> lk(mu_);
    return n_deferred_ > 0;
}

void Scheduler::release_deferred() {
    std::lock_guard<std::mutex> lk(mu_);
    sweep_deferred();
}

void Scheduler::shutdown(const char* reason, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_reason_ = reason ? reason : "the engine has stopped";
    closed_status_ = status < 0 ? status : RAD_E_STATE;
    /* Copied first: close_request erases from both queues. */
    std::vector<int32_t> live(running_);
    live.insert(live.end(), waiting_.begin(), waiting_.end());
    for (int32_t slot : live) {
        SchedReq& s = *reqs_[(size_t)slot];
        if (!s.req) continue;
        close_request(s, ReqState::Failed, closed_reason_, /*defer_free=*/false,
                      /*publish=*/false, closed_status_, closed_reason_);
    }
    /* No next open will deliver the last cycle. Its rows are final now: the ones the closes above
     * did not resolve are marked dropped, so the record either delivers (all reported) or is
     * counted as discarded -- never silently in flight. */
    draft_ring_flush();
    /* Nothing will commit again, so no entry of the last step still owns its slot. */
    step_open_ = false;
    next_open_ = false;
    for (auto& up : reqs_) if (up) { up->entry = -1; up->entry_next = -1; }
    block_.clear();
    serial_.clear();
    sweep_deferred();
}

int Scheduler::reap(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return RAD_E_NOTFOUND;
    int32_t slot = it->second;
    SchedReq& s = *reqs_[(size_t)slot];
    if (s.req->state == ReqState::Running || s.req->state == ReqState::Waiting ||
        s.req->state == ReqState::Preempted)
        return RAD_E_STATE;
    /* Not while a step in flight or a drafter snapshot still points at the slot
     * (SchedReq::reap_pending). For a drafter snapshot, keeping the Request alive rather than
     * dropping the row also preserves the row mapping the engine sized its readback from. */
    if (in_step(s) || block_holds(id)) {
        if (!s.reap_pending) ++n_deferred_;
        s.reap_pending = true;
        return RAD_OK;
    }
    release_locked(s);
    return RAD_OK;
}

int Scheduler::queue_snapshot(uint64_t* out, int max) const {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (int32_t slot : waiting_) {
        if (n >= max) break;
        out[n++] = reqs_[(size_t)slot]->req->id;
    }
    return n;
}

/* Copied out under the step lock rather than handed out as pointers; see ReqStat. */
int Scheduler::requests(ReqStat* out, int max) const {
    std::lock_guard<std::mutex> lk(mu_);
    return requests_locked(out, max);
}

int Scheduler::requests_locked(ReqStat* out, int max) const {
    const int64_t now = mono_us();
    const int32_t n_groups = prog_ ? (int32_t)prog_->kv_groups.size() : 0;

    int n = 0;
    for (const auto& p : reqs_) {
        if (n >= max) break;
        if (!p || !p->req) continue;
        const SchedReq& s = *p;
        const Request&  r = *s.req;
        ReqStat& o = out[n++];
        o.id              = r.id;
        o.state           = (int)r.state;
        o.running         = s.running;
        o.prompt_tokens   = (int64_t)r.prompt.size();
        o.computed_tokens = r.n_computed;
        o.cached_tokens   = r.n_cached;
        o.output_tokens   = (int64_t)r.output.size();
        o.max_tokens      = r.max_tokens;
        o.ctx_tokens      = s.seq_len();
        /* Blocks add up across paged groups, tokens do not: the groups share one token axis, so
         * summing tokens would count the context once per group against a pool capacity that is
         * one group's. The token figure is the largest group's, comparable with
         * SchedMetrics::kv_tokens_used. */
        o.kv_blocks = o.kv_tokens = 0;
        for (int32_t g = 0; g < n_groups; ++g) {
            const int64_t bt = kv_->block_tokens(g);
            if (bt <= 0) continue;                    /* stateful groups are not paged */
            const int64_t held = (int64_t)kv_->block_table(r.id, g).size();
            o.kv_blocks += held;
            if (held * bt > o.kv_tokens) o.kv_tokens = held * bt;
        }
        o.n_draft         = s.n_draft;
        o.slot            = s.slot;
        o.age_s           = (double)(now - s.arrived_us) / 1e6;
        o.ttft_s          = s.first_tok_us ? (double)(s.first_tok_us - s.arrived_us) / 1e6 : -1.0;
        /* Since the first token, not admission, so time spent queued does not read as slow
         * decode. */
        const double dec = s.first_tok_us ? (double)(now - s.first_tok_us) / 1e6 : 0.0;
        o.decode_tps      = dec > 1e-3 ? (double)r.output.size() / dec : 0.0;
    }
    std::sort(out, out + n, [](const ReqStat& a, const ReqStat& b) { return a.id < b.id; });
    return n;
}

/* ------------------------------------------------------------------ drafting */
/* Prompt-lookup drafting: find where the last few committed tokens last occurred in the same
 * sequence and propose what followed. On code, diffs and JSON it is right often enough to pay,
 * and a wrong guess costs only bandwidth, since the verify still emits the target's token. Built
 * in rather than a plugin because there is no model to load and no extra forward pass.
 *
 * Longest pattern and most recent occurrence first: a longer match predicts better, and falling
 * back to shorter ones only on a miss keeps common bigrams from dragging acceptance down. A
 * declared draft head calls set_draft() before step(), so this only fills an empty proposal. */
int Scheduler::draft_ngram(SchedReq& s, int depth) const {
    if (depth <= 0 || !s.req) return 0;
    const Request& r = *s.req;
    const int64_t np = (int64_t)r.prompt.size(), no = (int64_t)r.output.size();
    const int64_t n  = np + no;
    if (n < 2) return 0;
    auto at = [&](int64_t i) -> int32_t {
        return i < np ? r.prompt[(size_t)i] : r.output[(size_t)(i - np)];
    };
    for (int k = kNgramMax; k >= kNgramMin; --k) {
        if (n < (int64_t)k + 1) continue;
        for (int64_t j = n - k - 1; j >= 0; --j) {
            bool hit = true;
            for (int t = 0; t < k && hit; ++t) hit = (at(j + t) == at(n - k + t));
            if (!hit) continue;
            int d = 0;
            while (d < depth && j + k + d < n) { s.draft[d] = at(j + k + d); ++d; }
            return d;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ chunking */
/* A prefill chunk splits at the linear-state checkpoint interval so a checkpoint is always
 * written (spec §7.3; see scheduler.h).
 *
 * The checkpoint clamp comes before the quantum clamp. The interval is a multiple of the quantum,
 * so a chunk clamped to it is already aligned; in the other order a chunk past the checkpoint
 * would be rounded down and miss the boundary. */
int64_t Scheduler::plan_chunk(int64_t n_computed, int64_t n_target, int64_t budget,
                              int64_t* checkpoint_pos, int64_t keep_at) const {
    if (checkpoint_pos) *checkpoint_pos = -1;
    int64_t remaining = n_target - n_computed;
    if (remaining <= 0 || budget <= 0) return 0;

    int64_t want = remaining < budget ? remaining : budget;

    if (geo_.checkpoints()) {
        const int64_t I = geo_.checkpoint_interval;
        int64_t next = ((n_computed / I) + 1) * I;      /* strictly after n_computed */
        if (geo_.window_ckpt) next = INT64_MAX;          /* a window snapshot: the tail only */
        /* A span (ChunkGeometry::span) starts only on an interval boundary past the first
         * interval, and its chunk is the span's intervals in a row, each the piece a span of 1
         * makes a chunk of its own. Not from a prompt's start: the first interval may be served on
         * another path (QSA's dense attention below its exactness bound). Not from a tail
         * checkpoint: the pieces after it would no longer start on the interval. A chunk the
         * budget cuts short of the span's end steps back to the last boundary it crosses. */
        int64_t far = next;
        if (geo_.span > 1 && span_now_ && n_computed >= I && n_computed % I == 0)
            far = n_computed + geo_.span * I;
        if (n_computed + want > far)
            want = far - n_computed;
        else if (n_computed + want > next && n_computed + want < n_target)
            want = (n_computed + want) / I * I - n_computed;
        if (!geo_.window_ckpt && (n_computed + want) % I == 0) {
            if (checkpoint_pos) *checkpoint_pos = n_computed + want;
        } else if (n_computed + want == n_target) {
            /* Also one at the prompt's tail. The next turn resumes from the deepest snapshot
             * under its hit, which on the interval alone can be nearly an interval short of this
             * prompt's end, re-prefilled every turn. The final chunk splits at the last quantum
             * boundary, and the remainder (under a quantum) runs as the next step's chunk
             * (tail_checkpoint decides when that is worth it). */
            const int64_t tail = tail_checkpoint(n_target);
            if (tail > n_computed) {
                want = tail - n_computed;
                if (checkpoint_pos) *checkpoint_pos = tail;
            } else if (n_computed + want > next) {
                /* A spanned last chunk with no tail checkpoint still ends on the last boundary it
                 * crosses, where a span of 1 writes the prompt's deepest checkpoint, and the rest
                 * (under a quarter interval) is the next chunk; otherwise the next turn would
                 * resume up to an interval earlier. */
                want = (n_computed + want) / I * I - n_computed;
                if (checkpoint_pos) *checkpoint_pos = n_computed + want;
            }
        }
        /* And one where the request asked (Request::checkpoint_at), on its quantum: the end of a
         * prefix other requests are about to start from. */
        const int64_t keep = keep_at / geo_.quantum * geo_.quantum;
        if (keep > n_computed && keep <= n_computed + want && keep <= n_target) {
            want = keep - n_computed;
            if (checkpoint_pos) *checkpoint_pos = keep;
        }
    }

    if (want < remaining) {
        /* A non-final chunk must end on a quantum boundary measured from the sequence origin,
         * or the next chunk hands the linear kernel a partial tile and attention a partial
         * block. The final chunk may be short. */
        const int64_t Q = geo_.quantum;
        int64_t end = n_computed + want;
        int64_t aligned = end / Q * Q;
        if (aligned <= n_computed) {
            if (checkpoint_pos) *checkpoint_pos = -1;
            return 0;                     /* the budget left cannot hold one quantum */
        }
        if (aligned < end) {
            want = aligned - n_computed;
            if (checkpoint_pos) *checkpoint_pos = -1;
        }
    }
    return want;
}

/* The tail checkpoint of a `target`-token prompt, or -1: its last quantum boundary, only past the
 * first interval (a shorter prompt is cheap to redo) and only where it gains at least a quarter
 * interval over the interval checkpoint below it, which pays for the extra step. */
int64_t Scheduler::tail_checkpoint(int64_t target) const {
    const int64_t I = geo_.checkpoint_interval, Q = geo_.quantum;
    if (I <= 0 || Q <= 0) return -1;
    const int64_t tail = target / Q * Q;
    /* A window snapshot is taken at the tail of any prompt of an interval or more, a block short
     * of the end: the next turn re-renders the last few tokens (the generation prompt is replaced
     * by the reply), and a snapshot past where the two renderings diverge is unreachable. Prompts
     * that left fewer than ~8 tokens in their last block always missed. */
    if (geo_.window_ckpt) {
        const int64_t t = (target - std::max<int64_t>(Q, 16)) / Q * Q;
        return t >= I ? t : -1;
    }
    return tail > I && tail - tail / I * I >= std::max<int64_t>(Q, I / 4) ? tail : -1;
}

/* ------------------------------------------------------------------ a prefill sharing a step */
void Scheduler::set_span_min(int64_t tokens) {
    span_min_ = tokens > 0 ? tokens : 0;
    if (span_min_ > 0 && geo_.span > 1)
        RAD_INFO("scheduler: --prefill-span: a prefill spans when it is %lld tokens or longer -- "
                 "routed experts are read from host memory, and the arena a spanned step takes is "
                 "room they lose for it", (long long)span_min_);
}

void Scheduler::set_stage_rows(int64_t rows) {
    stage_rows_ = rows > 0 ? rows : 0;
    const int64_t n = cfg_.busy_prefill_chunk;
    if (n == 0) {
        RAD_INFO("scheduler: --busy-prefill-chunk off: a prefill chunk takes every token the step "
                 "has left, so decode rows behind it and requests waiting to start get none until "
                 "it ends");
        return;
    }
    const int64_t Q = std::max<int64_t>(geo_.quantum, 1);
    if (n > 0) {
        RAD_INFO("scheduler: --busy-prefill-chunk %lld: a prefill chunk sharing a step with other "
                 "work takes at most %lld tokens of it", (long long)n, (long long)(n / Q * Q));
        if (stage_rows_ > 0 && n < stage_rows_)
            RAD_WARN("scheduler: --busy-prefill-chunk %lld is under the %lld-token step the prefill "
                     "stager overlaps at: a held-back chunk alone reads every routed layer's "
                     "pooled experts synchronously, which slows a long prefill far more than the "
                     "smaller step saves", (long long)n, (long long)stage_rows_);
        return;
    }
    RAD_INFO("scheduler: --busy-prefill-chunk auto: a prefill chunk sharing a step leaves decode "
             "rows and waiting requests their room and keeps at least %lld tokens%s",
             (long long)std::max(Q, stage_rows_ / Q * Q),
             stage_rows_ > 0 ? " (the prefill stager's smallest pass)" : "");
}

/* How much of the step a prefill chunk may take when other work wants in.
 *
 * Uncapped, a prefill row takes every token the step has left, and on a prompt of a few hundred
 * thousand tokens that lasts a minute or more, during which decode rows behind it get nothing and
 * admission never runs. The cap leaves `reserve` (computed by step()) and never goes under the
 * floor: the smallest step the prefill stager overlaps, rounded down to the quantum, or one
 * quantum with no stager. A smaller step reads every routed layer's pooled experts
 * synchronously, which costs a long prefill far more than sharing saves.
 *
 * Chunks end on the checkpoint interval (plan_chunk), so a chunk held short leaves a remainder,
 * about the size of the reserve, as the next step's whole chunk. When that would be under the
 * floor, the interval is split into two chunks at or above the floor instead. */
int64_t Scheduler::busy_cap(int64_t n_computed, int64_t n_target, int64_t budget,
                            int64_t reserve, bool shared, bool rows_only) const {
    const int64_t mode = cfg_.busy_prefill_chunk;
    if (mode == 0 || budget <= 0) return budget;
    /* Auto holds a chunk back only for work that would otherwise get nothing this step: rows
     * behind it, or a request waiting to start. Rows planned ahead of it are in the step anyway.
     * An explicit N trades prefill speed for every other row's latency, so it applies whenever
     * the chunk shares the step. */
    if (reserve <= 0 && !(mode > 0 && shared)) return budget;
    reserve = std::max<int64_t>(reserve, 0);
    const int64_t Q = std::max<int64_t>(geo_.quantum, 1);

    int64_t cap = budget - reserve;
    if (mode > 0) {
        /* The operator's number, with no floor; set_stage_rows warns when it is under the
         * stager's step. */
        cap = std::min(cap, mode);
    } else {
        const int64_t floor = std::max(Q, stage_rows_ / Q * Q);
        cap = std::max(cap, floor);
        const int64_t remaining = n_target - n_computed;
        int64_t span = remaining;                       /* where this run of chunks must end */
        if (geo_.checkpoints() && !geo_.window_ckpt) {
            const int64_t I = geo_.checkpoint_interval;
            span = std::min(span, ((n_computed / I) + 1) * I - n_computed);
        }
        /* ...but not for decode rows alone: their reserve is a few tokens, so the chunk ends a
         * quantum short and the next step finishes the interval, a step pair a prefill already
         * runs when decode rows are ahead of it. Halving would hold a long prompt to
         * half-interval chunks while anything decodes behind it, which costs more. The split is
         * for a share (a waiting request, another prefill), which lasts a step or two. */
        if (!rows_only && cap < span && span - cap < floor && span - floor >= floor)
            cap = span - floor;
    }
    return std::min(std::max(cap, Q), budget);
}

/* ------------------------------------------------------------------ preemption */
void Scheduler::preempt_back() {
    int32_t slot = running_.back();
    SchedReq& s = *reqs_[(size_t)slot];
    Request& r = *s.req;

    /* Preemption by recompute: the blocks go back and the request restarts from zero. Its output
     * is the tail of its token sequence, so it simply re-prefills prompt ++ output. */
    kv_->preempt(r.id);
    drop_card(s);
    r.n_computed = 0;
    r.n_cached = 0;
    r.n_accepted = 0;
    r.state = ReqState::Preempted;
    s.drafter_reset();
    /* The slot keeps this request's identity (preemption does not bump the generation), so B's
     * per-slot estimate survives the recompute; the cycles still waiting on its drafts do not,
     * because the step that would verify them no longer includes it. */
    draft_ring_drop_row(r.id, s.slot, s.gen);
    /* Also clear the request's own window. The sampler stages 1 + Request::n_draft rows
     * (Sampler::build_step), and a drafter may have set it for the step this preemption removed;
     * left standing, the recompute's first sampled step stages rows its batch lacks and every
     * later request in the step samples another's logits row. */
    r.n_draft = 0;
    r.draft = nullptr;
    s.running = false;

    running_.pop_back();
    queue_insert(waiting_, slot);   /* original arrival -> the front of its priority class */
    ++m_.preemptions;
}

/* ------------------------------------------------------------------ admission */
/* Admission checks the KV budget before a request enters running; a request admitted without the
 * blocks to run would be preempted on its first step, a full recompute for nothing. */
int Scheduler::admit_one(SchedReq& s, int64_t budget, int64_t reserve) {
    Request& r = *s.req;

    if (!kv_->has_sequence(r.id)) RAD_TRY(kv_->add_sequence(r.id));

    /* A fresh attempt: nothing looked up or counted yet. A request that kept an earlier
     * attempt's hit (n_computed > 0) keeps what it found. */
    if (r.n_computed == 0) {
        s.ckpt_src = -1;
        s.hit_attn = 0;
        s.hit_owed = true;
        s.hit_cut = SchedReq::HitCut::None;
    }

    /* Check what no hit can change first (a free card index, a budget that fits a chunk): a
     * restore copies the whole context back, and an admission refused after it would repeat the
     * copy on the next attempt. */
    if (s.card < 0 && free_cards_.empty()) return RAD_E_FULL;
    if (budget < std::max<int64_t>(geo_.quantum, 1) && budget < s.seq_len() - r.n_computed)
        return RAD_E_FULL;

    if (r.n_computed == 0 && cfg_.prefix_cache) {
        /* Restore before the lookup, which stops at the first block the tiers hold. This is
         * where a cold session gets its context back for the cost of a copy instead of a
         * prefill. */
        pc_->restore(r);
        HitCap cap;
        if (draft_refill_ > 0) {
            cap.tokens = std::max<int64_t>(0, (int64_t)r.prompt.size() - draft_refill_);
            cap.max_give_back = 2 * draft_refill_;
        }
        const CacheHit h = pc_->lookup(r, &hit_blocks_, draft_refill_ > 0 ? &cap : nullptr);
        if (h.given_back > 0) {
            m_.draft_refill_tokens += h.given_back;
            ++m_.draft_refills;
        }
        int32_t reuse = h.n_attn_tokens;
        if (geo_.checkpoints() && h.n_linear_tokens < reuse) {
            /* A hit between checkpoints reuses the attention blocks but must replay the linear
             * layers from the last checkpoint (§7.3). RadBatch carries one query length for all
             * groups (per-group q_lens on RadKVGroupBatch would be needed), so recompute from
             * the checkpoint instead. */
            reuse = h.n_linear_tokens;
        }
        int32_t last = (int32_t)r.prompt.size() - 1;   /* one token must remain to produce logits */
        if (reuse > last) reuse = last;
        if (reuse < 0) reuse = 0;

        /* A hit on a hybrid model needs a checkpoint. The adopted blocks cover the paged
         * attention groups, but nothing covers a recurrent group, whose state lives in a
         * per-sequence slot; skipping tokens also requires restoring that state (`ckpt_src`,
         * then KVManager::checkpoint_restore). `reuse` is already clamped to h.n_linear_tokens;
         * this guarantees reuse > 0 never happens without a snapshot, which would continue every
         * recurrent layer from stale state and produce fluent but wrong text. */
        if (reuse > 0 && has_linear_state_ && h.checkpoint_slot < 0) reuse = 0;
        if (h.n_attn_tokens > 0 && reuse == 0 && geo_.checkpoints()) {
            s.hit_cut = SchedReq::HitCut::NoSnapshot;
            RAD_DEBUG("admit %llu: prefix hit of %lld tokens reuses nothing -- no snapshot at or "
                      "below it", (unsigned long long)r.id, (long long)h.n_attn_tokens);
        } else if (reuse < h.n_attn_tokens) {
            RAD_DEBUG("admit %llu: prefix hit of %lld tokens resumes at the snapshot at %lld",
                      (unsigned long long)r.id, (long long)h.n_attn_tokens, (long long)reuse);
        }
        /* A window snapshot covers the trunk's windowed groups at the checkpoint only. The lookup
         * clamped the hit to it (a window counts as state there), so a hit with a slot is exactly
         * at a snapshot; one without is no hit. */
        const bool win_hit = geo_.window_ckpt && kv_->has_window_snapshot() && reuse > 0 &&
                             h.checkpoint_slot >= 0 && reuse == h.n_linear_tokens;

        /* Windowed-group clamp: the trunk's own KV includes a paged group the cache does not
         * index. prefix.cpp skips RAD_KV_WINDOW groups (their blocks below the window are
         * recycled), so the adopted blocks cover the cached groups only, while `used = ctx_len +
         * n_tokens` (batch.cpp) makes each windowed trunk layer read positions below ctx_len this
         * sequence never wrote. grab_block's min-heap (kv.cpp) hands out the blocks the latest
         * tenant wrote, so the model would attend to the previous request's KV. Zero the reuse
         * and re-prefill unless a window snapshot covers it.
         *
         * A drafter's own group (bound at base_layer and above) is not this clamp's business:
         * draft refill caps the hit so the recompute covers the drafter's window. */
        if (reuse > 0 && hit_holes_trunk_ && !win_hit) {
            reuse = 0;
            s.hit_cut = SchedReq::HitCut::Window;
            RAD_DEBUG("admit %llu: prefix hit %lld zeroed -- a trunk paged group is not "
                      "prefix-cached (windowed) and no window snapshot covers it; re-prefilling",
                      (unsigned long long)r.id,
                      (long long)h.n_attn_tokens);
        }

        /* Claim the cached blocks; otherwise the sequence would skip those tokens and attend to
         * blocks it does not own. adopt() takes a whole block multiple, which `reuse` is: a whole
         * number of blocks, or a checkpoint position (on the interval, or on a quantum boundary at
         * a prompt's tail per plan_chunk), both block multiples by geometry.h. */
        if (reuse > 0) {
            const std::vector<int32_t>& cg = pc_->cached_groups();
            for (size_t i = 0; i < cg.size() && i < hit_blocks_.size(); ++i) {
                int64_t nb = reuse / kvgeom_.g[(size_t)cg[i]].block_size;
                if ((int64_t)hit_blocks_[i].size() > nb) hit_blocks_[i].resize((size_t)nb);
                if (kv_->adopt(r.id, cg[i], hit_blocks_[i], reuse) < 0) {
                    /* Return what earlier groups adopted, or the recompute would write its KV
                     * into the cache's own blocks. */
                    kv_->preempt(r.id);
                    reuse = 0;
                    s.hit_cut = SchedReq::HitCut::Adopt;
                    break;
                }
            }
            /* The windowed groups start at the floor of the window behind the hit, in fresh
             * blocks the entry's restore fills from the snapshot. */
            if (reuse > 0 && win_hit && kv_->seed_window(r.id, reuse) < 0) {
                kv_->preempt(r.id);
                reuse = 0;
                s.hit_cut = SchedReq::HitCut::Adopt;
            }
        }

        r.n_computed = reuse;
        r.n_cached = reuse;
        s.ckpt_src = (reuse > 0 && reuse == h.n_linear_tokens &&
                      (has_linear_state_ || win_hit)) ? h.checkpoint_slot : -1;
        s.hit_attn = h.n_attn_tokens;
    }

    int64_t target = s.seq_len();
    int64_t ckpt = -1;
    int64_t want;
    if (r.embed) {
        /* The whole input or nothing this step (Request::embed): the next step starts with a
         * whole budget, and add_locked refused an input no budget holds. Not through plan_chunk,
         * whose checkpoint splits would cut every input longer than the interval short, and an
         * input that can only run whole would then never be admitted, nor anything queued behind
         * it. An embedding input is never resumed, so it takes no checkpoint. */
        const int64_t rest = target - (int64_t)r.n_computed;
        want = rest <= budget ? rest : 0;
    } else {
        /* An admission shares the step the same way: otherwise a cold prompt with requests
         * queued behind it takes every token left. The hit is known by now, so the cap applies
         * to what really has to run. */
        const int64_t cap = busy_cap(r.n_computed, target, budget, reserve, !plan_.e.empty());
        want = plan_chunk(r.n_computed, target, cap, &ckpt, r.checkpoint_at);
        if (cap < budget && want > 0)
            admit_full_ = plan_chunk(r.n_computed, target, budget, nullptr, r.checkpoint_at);
    }
    int rc = RAD_OK;
    if (want <= 0)                                          rc = RAD_E_FULL;
    else if (s.card < 0 && free_cards_.empty())             rc = RAD_E_FULL;
    else {
        /* The cache gives way to a new request as it does to a running one; with nothing
         * running, no other path would reclaim from it and the request would wait forever. */
        int rounds = 0;
        while (kv_->ensure_from(r.id, r.n_computed + want, r.n_computed) < 0) {
            if (++rounds > kEvictRounds || !cfg_.prefix_cache || pc_->evict(kEvictBlocks) <= 0) {
                rc = RAD_E_NOMEM;
                break;
            }
        }
    }
    if (rc < 0) {
        /* A hit that resumes from a snapshot is not kept across a failed attempt. The slot is
         * named by id only, and until the first step restores from it the cache may retire it
         * and reuse it for another sequence, so a later attempt would restore someone else's
         * recurrent state. Giving the hit back costs a lookup. An attention-only hit holds only
         * refcounted blocks and is kept. */
        if (s.ckpt_src >= 0) undo_hit(s);
        /* What the restore brought back stays for the next attempt (IPrefixCache::hold). */
        if (cfg_.prefix_cache) pc_->hold(r);
        return rc;
    }

    /* Counted once per admission, here, where it is known to have happened. */
    if (s.hit_owed) {
        m_.prompt_tokens += (int64_t)r.prompt.size();
        if (cfg_.prefix_cache) {
            m_.attn_cached_tokens += s.hit_attn;
            m_.linear_cached_tokens += r.n_cached;
            if (s.hit_attn > r.n_cached) m_.hit_cut_tokens += s.hit_attn - r.n_cached;
            switch (s.hit_cut) {
                case SchedReq::HitCut::NoSnapshot: ++m_.hit_cut_no_snapshot; break;
                case SchedReq::HitCut::Window:     ++m_.hit_cut_window;      break;
                case SchedReq::HitCut::Adopt:      ++m_.hit_cut_adopt;       break;
                case SchedReq::HitCut::None:       break;
            }
        }
        s.hit_owed = false;
    }

    if (s.card < 0) { s.card = free_cards_.back(); free_cards_.pop_back(); }
    r.state = ReqState::Running;
    s.running = true;
    server::sink_admitted(r.sink);
    server::sink_prompt_progress(r.sink, r.n_computed, (int32_t)r.prompt.size(), r.n_cached);
    s.sched_tokens = (int32_t)want;
    s.pending_ckpt = ckpt;
    ++m_.admitted;
    return RAD_OK;
}

void Scheduler::undo_hit(SchedReq& s) {
    Request& r = *s.req;
    /* preempt(), not free_sequence(): the sequence stays known to the manager at zero tokens,
     * where a fresh attempt starts. */
    if (kv_->has_sequence(r.id)) kv_->preempt(r.id);
    r.n_computed = 0;
    r.n_cached = 0;
    s.ckpt_src = -1;
    s.hit_attn = 0;
}

/* ------------------------------------------------------------------ the step */
const RadBatch* Scheduler::step() {
    std::lock_guard<std::mutex> lk(mu_);
    /* A cycle opens here, so its head depth is chosen here (scheduler.h, draft_cycle). */
    draft_cycle();
    /* The cycle was refused (a drifted residency, or a shape the ladder cannot price): nothing is
     * planned, and the engine stops on the latch instead of letting the step run at the ceiling. */
    if (draft_cost_aborted_) return nullptr;

    /* The drafter's snapshots describe the previous step and nothing reads them now. block_end()
     * and serial_end() normally clear them; this covers a drafter that stopped part way, so a
     * stale snapshot cannot keep deferring a release. */
    block_.clear();
    serial_.clear();
    /* The last step's admissions have read their snapshots: its forward committed before this. */
    if (pc_) pc_->begin_step();

    plan_.reset();
    step_prefill_tok_ = 0;
    step_decode_tok_ = 0;

    int64_t budget = cfg_.max_tok;
    bool preempted_here = false;
    /* --prefill-interleave: after a step that carried a chunk beside decoding sequences, with the
     * prompt continuing, the next N steps carry decode rows only and admit nothing. With nothing
     * decoding there is nothing to give steps to, and the hold is dropped. */
    bool any_decoding = false;
    bool hold_prefill = false;
    held_prefill_ = 0;
    if (cfg_.prefill_interleave > 0) {
        for (int32_t slot : running_) {
            const SchedReq& s = *reqs_[(size_t)slot];
            if (!s.req || !s.running) continue;
            const Request& r = *s.req;
            if (r.n_computed < (int64_t)r.prompt.size() || r.n_computed + 1 < s.seq_len()) continue;
            any_decoding = true;
            break;
        }
        if (!any_decoding) interleave_left_ = 0;
        hold_prefill = interleave_left_ > 0;
    }

    /* A span is for a prefill alone in its step (ChunkGeometry::span). A spanned step takes as
     * long as the intervals it carries, so a decode row riding along would wait that long for its
     * token. The span is taken only when the step can hold nothing but the one prefill (a single
     * running request with prompt left and no admission possible, or a single request in an empty
     * engine); every other step is planned at ChunkGeometry::shared_tok as a deployment without a
     * span plans it, --busy-prefill-chunk included. Both kinds of chunk end on the same interval
     * grid, so a prompt may switch between them at any step. */
    span_now_ = true;
    struct SpanReset { bool& f; ~SpanReset() { f = true; } } span_reset{span_now_};
    /* Keyed on the flag, not on whether this model has intervals to span: the engine told the
     * plugin that a step wider than shared_tok is one sequence (RadBuildCtx::max_mixed_tok),
     * whatever the geometry turned out to be. */
    if (cfg_.prefill_span > 1) {
        bool lone = false;
        const bool admit_possible = !waiting_.empty() && !free_cards_.empty() &&
                                    (int64_t)running_.size() < cfg_.max_seqs;
        if (running_.size() == 1 && !admit_possible) {
            const SchedReq& s0 = *reqs_[(size_t)running_[0]];
            lone = s0.req->n_computed < (int64_t)s0.req->prompt.size() ||
                   s0.req->n_computed + 1 < s0.seq_len();
            /* By the whole prefill, not what is left of it, so a prompt keeps spanning through
             * its last intervals (set_span_min). */
            if (lone && span_min_ > 0)
                lone = std::max<int64_t>((int64_t)s0.req->prompt.size(), s0.seq_len()) >= span_min_;
        } else if (running_.empty() && waiting_.size() == 1) {
            const SchedReq& w = *reqs_[(size_t)waiting_[0]];
            lone = span_min_ == 0 || (w.req && (int64_t)w.req->prompt.size() >= span_min_);
        }
        span_now_ = lone;
        if (!lone) budget = geo_.shared_tok;
    }

    /* A mixed step takes drafts like any other: the linear blocks run the recurrent path over its
     * decode rows and the chunked path over its prefill rows, so a verifying row rolls back as in
     * a pure decode step. Run through the chunked path (which has no num_accepted operand), its
     * rejected tokens would stay folded into the recurrent state and conv window. */
    const int depth = draft_.next_depth();

    /* What else wants into this step, for --busy-prefill-chunk (busy_cap): each decode row still
     * to be walked needs 1 + the deepest window it can take, each prefill row a share, plus one
     * share when a request waits and a slot is free. Counted down as the loop walks, so a row's
     * reserve is what is behind it. */
    const bool share_step = cfg_.busy_prefill_chunk != 0;
    const int64_t Q = std::max<int64_t>(geo_.quantum, 1);
    /* An eighth of this step's budget, which under a span is shared_tok for a shared step. */
    const int64_t share_tok = std::max<int64_t>(Q, budget / 8 / Q * Q);
    const int64_t dec_row_tok = 1 + std::max<int64_t>(depth, draft_.step_window());
    int64_t behind_dec = 0, behind_pf = 0;
    if (share_step) {
        for (int32_t slot : running_) {
            const Request& rr = *reqs_[(size_t)slot]->req;
            const int64_t tg = reqs_[(size_t)slot]->seq_len();
            const bool pf = (rr.n_computed < (int64_t)rr.prompt.size()) || (rr.n_computed + 1 < tg);
            if (pf) ++behind_pf; else behind_dec += dec_row_tok;
        }
    }
    const bool admit_waits = share_step && !waiting_.empty() && !free_cards_.empty() &&
                             (int64_t)running_.size() < cfg_.max_seqs;
    /* The first prefill chunk held back this step (its index in plan_.e before the sort) and what
     * it would have taken: whatever the work it made room for did not use goes back to it. */
    int32_t capped_at = -1;
    int64_t capped_full = 0;

    /* ---------------------------------------------------------- running requests first */
    /* Continuous batching: requests already holding blocks are served before new ones are
     * admitted, so new work never preempts work half done. */
    for (size_t i = 0; i < running_.size(); ) {
        if ((int64_t)plan_.e.size() >= cfg_.max_seqs || budget <= 0) break;

        int32_t slot = running_[i];
        SchedReq& s = *reqs_[(size_t)slot];
        Request& r = *s.req;

        const int64_t target = s.seq_len();
        const bool prefill = (r.n_computed < (int64_t)r.prompt.size()) ||
                             (r.n_computed + 1 < target);
        if (share_step) {
            if (prefill) behind_pf = std::max<int64_t>(0, behind_pf - 1);
            else         behind_dec = std::max<int64_t>(0, behind_dec - dec_row_tok);
        }

        int64_t ckpt = -1;
        int64_t want;
        if (prefill && hold_prefill) { ++held_prefill_; ++i; continue; }
        if (prefill) {
            const int64_t shares = share_step ? behind_pf + (admit_waits ? 1 : 0) : 0;
            const int64_t reserve = share_step ? behind_dec + share_tok * shares : 0;
            const int64_t cap = busy_cap(r.n_computed, target, budget, reserve, !plan_.e.empty(),
                                         /*rows_only=*/shares == 0);
            want = plan_chunk(r.n_computed, target, cap, &ckpt, r.checkpoint_at);
            if (cap < budget && want > 0 && capped_at < 0) {
                const int64_t full = plan_chunk(r.n_computed, target, budget, nullptr,
                                                r.checkpoint_at);
                if (want < full) { capped_at = (int32_t)plan_.e.size(); capped_full = full; }
            }
            /* A prefill with no room left does not end the walk: it takes nothing this step, but
             * a decode row behind it needs only 1 + its window. With --busy-prefill-chunk off the
             * walk stops here. */
            if (want <= 0 && share_step) { ++i; continue; }
        } else {
            /* A verify step is 1 + n_draft query rows, which is why attention kernels take a
             * query length (spec §10). */
            /* The final speculation gate. It must be here, not only in set_draft, because the
             * n-gram fallback below fills any empty proposal; refusing upstream alone would just
             * swap in a different draft. See Request::may_speculate.
             *
             * The cap is the sequence's own window: `depth`, or under the adaptive policy what
             * its acceptance of the head's drafts pays for, or under a cost plan the K this
             * cycle chose for this row. That estimate says nothing about the n-gram fallback, so
             * the fallback is held to the window the cycle planned: the step's window under the
             * adaptive policy (as deep as the head drafts this step and no deeper), the row's own
             * K under a cost plan (a K=2 row beside a K=6 one must not verify four). */
            const int cap = draft_cap(s);   /* the ceiling, the adaptive window, or the plan */
            const int pk  = planned_k(s);
            bool ngram = false;
            if (!r.may_speculate) s.n_draft = 0;
            else if (s.n_draft == 0 && !bars_fill(s)) {
                /* The plan bars this row: K=0 is a speculation-off decision, not a vacancy, and a
                 * row the plan does not name has no budget either. A row the plan lets draft
                 * (K>0) but for which the head published nothing keeps the fallback, bounded by
                 * cap. Without a plan the legacy behaviour stands. */
                s.n_draft = (int32_t)draft_ngram(s, pk >= 0 ? cap : draft_.step_window());
                ngram = s.n_draft > 0;
            }
            if (!ngram && s.n_draft > cap) s.n_draft = cap;
            s.draft_ngram = ngram;
            /* A lookup fill is deliberately not published to the cycle record: the plan's fill[]
             * never names Ngram, so the row may not claim these proposals
             * (draft_origin_fits_plan). The step still verifies them and the observation carries
             * that verify with no drafting -- so B learns nothing from a lookup row. */
            /* No verified position at or past the context end: nothing past ctx_limit_ - 1 has a
             * rotary row or block-table column. commit() finishes a sequence before it reaches
             * the limit, so the one token itself always fits. */
            const int64_t spare = ctx_limit_ - 1 - (int64_t)r.n_computed;
            if (ctx_limit_ > 0 && s.n_draft > spare) s.n_draft = spare > 0 ? (int32_t)spare : 0;
            want = 1 + s.n_draft;
            if (want > budget) want = 0;
        }
        if (want <= 0) break;   /* the budget is spent; the rest of running_ waits a step */

        /* KV first; on refusal preempt from the back (lowest priority, latest arrival), the one
         * that would be re-admitted last anyway. Reclaiming from the cache is tried first, but
         * only kEvictRounds times; past that the shortfall is preemption's to fix. */
        int evict_rounds = 0;
        bool failed = false, yielded = false;
        while (kv_->ensure_from(r.id, r.n_computed + want, r.n_computed) < 0) {
            /* The cache gives blocks back before a running request does: a cached prefix saves
             * work done once, a running request is work in flight. Nothing else releases the
             * cache's references, so this also keeps publishing from leaking. */
            if (++evict_rounds <= kEvictRounds && pc_ && cfg_.prefix_cache &&
                pc_->evict(kEvictBlocks) > 0)
                continue;
            const bool last = running_.size() - 1 <= i;
            if (!kv_->can_ever_fit(r.n_computed + want) || (last && i == 0)) {
                /* §17: either no pool could serve this length, or the request runs alone and
                 * still cannot grow. Fail the request rather than deadlock. */
                RAD_WARN("request %llu failed: the KV pool cannot hold %lld tokens%s",
                         (unsigned long long)r.id, (long long)(r.n_computed + want),
                         kv_->can_ever_fit(r.n_computed + want)
                             ? " and nothing else is left to preempt"
                             : " at any occupancy");
                close_request(s, ReqState::Failed, "kv_exhausted", false, true, RAD_E_FULL,
                              "the KV pool cannot hold it");
                failed = true;
                break;
            }
            /* The back of the running set gives its blocks up. When that is this request, the
             * requests ahead of it hold the pool: it is preempted by recompute and waits for
             * them, rather than failed. */
            preempt_back();
            preempted_here = true;
            if (last) { yielded = true; break; }
        }
        /* close_request or preempt_back removed it from running_; i now indexes the next. */
        if (failed || yielded) continue;

        StepEntry e;
        e.req = &r;
        e.seq = r.id;
        e.slot = slot;
        e.card = s.card;
        e.n_tokens = (int32_t)want;
        e.ctx_len = (int32_t)r.n_computed;
        e.n_spec = prefill ? 0 : s.n_draft;
        /* Keep the request's window equal to the entry's. Sampler::build_step sizes its staging
         * from Request::n_draft (1 + n_draft rows) while the batch sizes its logits rows from
         * `n_spec`; setting only one makes the sampler stage the wrong number of rows. Set here,
         * once, where the window is decided. */
        r.n_draft = e.n_spec;
        r.draft   = e.n_spec ? s.draft : nullptr;
        /* One-based: the number of tokens the previous step committed, with the rolling windows
         * read at `n_accepted - 1`, so 0 is never valid. A prefill rolls nothing back, which is
         * 1, as the kernels assume when the operand is absent (ple_conv's contract). Every chunk
         * after the first is warm (`has_init` is the context length), and a warm read at -1 would
         * take the slot before: the previous channel's sample in ple_conv, a foreign token in
         * ngram_ids. */
        e.n_accepted = prefill ? 1 : r.n_accepted + 1;
        e.conv_cursor = s.conv_cursor;
        e.draft = s.draft;
        e.n_draft = s.n_draft;
        e.is_prefill = prefill;
        e.ngram = !prefill && s.draft_ngram && s.n_draft > 0;
        e.produces_token = (r.n_computed + want >= target);
        /* The snapshot a hit resumes from, consumed once: the state is in the slot for the whole
         * sequence after this, and restoring it again on a later chunk would rewind it. */
        e.ckpt_restore = s.ckpt_src;
        s.ckpt_src = -1;
        /* The checkpoint is the state after the chunk's last token, which the split put at the
         * interval boundary. */
        if (ckpt > 0) reserve_ckpt(e, r, ckpt, (int32_t)(want - 1));
        s.entry = (int32_t)plan_.e.size();
        s.sched_tokens = (int32_t)want;
        plan_.e.push_back(e);
        plan_.n_tok += want;
        if (!prefill && s.n_draft > plan_.n_spec) plan_.n_spec = s.n_draft;
        budget -= want;
        ++i;
    }

    /* ---------------------------------------------------------- admission */
    /* Skipped in a step that preempted: the pool just refused a request that already had blocks,
     * so it would refuse a new one, and the queue's front is now the preempted request. */
    if (!preempted_here && !hold_prefill) {
        /* `at` walks the queue and `took_` is what it admitted: a request whose media is not
         * encoded yet stays where it is and the one behind it is asked, so the queue is compacted
         * after the walk rather than cut at its front. */
        took_.clear();
        size_t at = 0;
        while (at < waiting_.size() &&
               (int64_t)plan_.e.size() < cfg_.max_seqs && budget > 0) {
            /* A step planned for one prefill holds that one (span_now_ above). */
            if (cfg_.prefill_span > 1 && span_now_ && !plan_.e.empty()) break;
            int32_t slot = waiting_[at];
            SchedReq& s = *reqs_[(size_t)slot];
            /* A sequence the context cannot hold never runs: its prompt alone reaches the last
             * addressable position, so prefilling it would write positions the block and rotary
             * tables lack. The server refuses such prompts; this is the scheduler's own bound.
             * close_request removes it from waiting_, so `at` now names the next request. */
            if (ctx_limit_ > 0 && s.seq_len() >= ctx_limit_) {
                RAD_WARN("request %llu failed: %lld tokens leave no room in a %lld-token context",
                         (unsigned long long)s.req->id, (long long)s.seq_len(),
                         (long long)ctx_limit_);
                close_request(s, ReqState::Failed, "context_length_exceeded", false, true,
                              RAD_E_FULL, "it leaves no room in the context");
                continue;
            }
            /* Media first: a request whose items are not encoded keeps its place and the next is
             * asked; the engine encodes before the next step (encodes_pending). */
            if (s.req->media && !media_ready(s)) { ++at; continue; }
            /* Leave one share for the next queued request when there is one and a slot for it
             * (busy_cap). Embed inputs are whole-or-nothing, so a share could only turn their
             * admission into a refusal. */
            const int64_t areserve =
                share_step && !s.req->embed && at + 1 < waiting_.size() &&
                (int64_t)plan_.e.size() + 1 < cfg_.max_seqs && free_cards_.size() > 1
                    ? share_tok : 0;
            admit_full_ = 0;
            const int as = admit_one(s, budget, areserve);
            if (as < 0) break;                     /* head of line: do not admit behind it */
            if (admit_full_ > s.sched_tokens && capped_at < 0) {
                capped_at = (int32_t)plan_.e.size();
                capped_full = admit_full_;
            }

            Request& r = *s.req;
            StepEntry e;
            e.req = &r;
            e.seq = r.id;
            e.slot = slot;
            e.card = s.card;
            e.n_tokens = s.sched_tokens;
            e.ctx_len = (int32_t)r.n_computed;
            e.n_spec = 0;
            /* The sampler sizes its staging off the Request's own window, and a request back from
             * preemption may still carry a drafter's proposal. */
            r.n_draft = 0;
            r.draft = nullptr;
            e.n_accepted = 1;               /* one-based; a prefill rolls nothing back */
            e.conv_cursor = s.conv_cursor;
            e.draft = s.draft;
            e.n_draft = 0;
            e.is_prefill = true;
            e.produces_token = (r.n_computed + s.sched_tokens >= s.seq_len());
            /* admit_one just set ckpt_src, and this is the entry it applies to. Consumed once:
             * restoring again on a later chunk would rewind the recurrence to the hit. */
            e.ckpt_restore = s.ckpt_src;
            s.ckpt_src = -1;
            if (s.pending_ckpt > 0) reserve_ckpt(e, r, s.pending_ckpt, s.sched_tokens - 1);
            s.entry = (int32_t)plan_.e.size();
            plan_.e.push_back(e);
            plan_.n_tok += s.sched_tokens;
            budget -= s.sched_tokens;
            took_.push_back(at);
            ++at;
        }
        if (!took_.empty()) {
            for (size_t i : took_) queue_insert(running_, waiting_[i]);
            size_t w = 0, k = 0;
            for (size_t i = 0; i < waiting_.size(); ++i) {
                if (k < took_.size() && took_[k] == i) { ++k; continue; }
                waiting_[w++] = waiting_[i];
            }
            waiting_.resize(w);
        }
    }

    /* Budget the held-back chunk reserved for others but they did not use goes back to it. The
     * reserve is a forecast: a decode row drafts less than its window, an admission is refused,
     * a step preempts and skips admission. A chunk held to the stager's floor in a step with
     * nothing else is the slow pass the floor exists to avoid, so the unused budget is returned,
     * as far as plan_chunk allows and only if the pool grants the blocks without preempting. */
    if (capped_at >= 0) {
        StepEntry& e = plan_.e[(size_t)capped_at];
        const Request& r = *e.req;
        SchedReq& s = *reqs_[(size_t)e.slot];
        const int64_t had = e.n_tokens;
        /* An explicit --busy-prefill-chunk N is a ceiling, not a forecast: the chunk grows back
         * to it at most. */
        int64_t limit = std::min(capped_full, had + budget);
        if (cfg_.busy_prefill_chunk > 0) limit = std::min(limit, std::max(had, cfg_.busy_prefill_chunk));
        int64_t ckpt = -1;
        const int64_t grown = e.ckpt_slot < 0 && limit > had
            ? plan_chunk(r.n_computed, s.seq_len(), limit, &ckpt, r.checkpoint_at) : had;
        if (grown > had && kv_->ensure_from(r.id, r.n_computed + grown, r.n_computed) >= 0) {
            e.n_tokens = (int32_t)grown;
            e.produces_token = (r.n_computed + grown >= s.seq_len());
            if (ckpt > 0) reserve_ckpt(e, r, ckpt, (int32_t)(grown - 1));
            s.sched_tokens = (int32_t)grown;
            plan_.n_tok += grown - had;
            budget -= grown - had;
        }
        if (e.n_tokens < capped_full) ++m_.prefill_chunks_capped;
    }

    if (plan_.e.empty()) { interleave_left_ = 0; held_prefill_ = 0; return nullptr; }

    /* Decode rows first, so each half is contiguous. The running loop walks `running_` in
     * priority order, interleaving the two kinds; sorted, a block can issue the recurrent path
     * over one run of sequences and the chunked path over the other. cu_seqlens holds absolute
     * row offsets, so a sub-array needs no rebasing (RadBatch::n_seq_decode).
     *
     * Stable, so each half keeps the scheduler's order. `s.entry` indexes this vector and was
     * assigned before the sort, so every one is rewritten here; a stale index makes a request
     * read another's result. */
    {
        part_.clear();
        size_t w = 0;
        for (size_t i = 0; i < plan_.e.size(); ++i) {
            if (plan_.e[i].is_prefill) part_.push_back(plan_.e[i]);
            else plan_.e[w++] = plan_.e[i];
        }
        for (const StepEntry& e : part_) plan_.e[w++] = e;
    }
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        StepEntry& e = plan_.e[i];
        if (e.slot >= 0 && (size_t)e.slot < reqs_.size()) reqs_[(size_t)e.slot]->entry = (int32_t)i;
    }

    /* Phase from what was scheduled; the run phase indexes bucket tables by it. A step with one
     * prefill chunk is MIXED, not DECODE, or it would pick the skinny-GEMM band for a large
     * GEMM. */
    bool any_pf = false; int64_t n_dec = 0;
    for (const StepEntry& e : plan_.e) { if (e.is_prefill) any_pf = true; else ++n_dec; }
    plan_.phase = any_pf && n_dec ? RAD_PHASE_MIXED
                : any_pf         ? RAD_PHASE_PREFILL
                                 : RAD_PHASE_DECODE;

    /* Step-shape counters. `steps_mixed` / `steps` shows how often a step carries both kinds of
     * row. `decode_rows_unspeculated` counts every decode row that took no draft, whatever the
     * reason (depth 0, an ineligible request, nothing to propose). */
    switch (plan_.phase) {
        case RAD_PHASE_PREFILL: ++m_.steps_prefill; break;
        case RAD_PHASE_MIXED:   ++m_.steps_mixed;   break;
        default:                ++m_.steps_decode;  break;
    }
    if (cfg_.prefill_interleave > 0) {
        /* Armed only while a prompt continues past this step: after a last chunk nothing is left
         * to interleave, and holding would only keep the next request out. */
        bool continues = false;
        for (const StepEntry& e : plan_.e)
            if (e.is_prefill && !e.produces_token) continues = true;
        if (any_pf) interleave_left_ = any_decoding && continues ? cfg_.prefill_interleave : 0;
        else if (interleave_left_ > 0) --interleave_left_;
        if (held_prefill_ > 0) ++m_.steps_held;
    }
    for (const StepEntry& e : plan_.e)
        if (!e.is_prefill && e.n_spec == 0) ++m_.decode_rows_unspeculated;
    plan_.n_seq_decode = n_dec;
    plan_.n_tok_decode = 0;
    for (int64_t i = 0; i < n_dec; ++i) plan_.n_tok_decode += plan_.e[(size_t)i].n_tokens;
    /* The rows the card's acceptance reads (advance.h), in final order: each entry's token row of
     * its first verified position, its first sampled row, and its draft count. Entries that
     * produce a token get 1 + n_spec sampled rows each, in the order the builder writes out_ids
     * and the engine reads them back. */
    {
        int64_t k = 0;
        int32_t row = 0;
        for (StepEntry& e : plan_.e) {
            e.dev_tok = (int32_t)(k + e.n_tokens - 1 - e.n_spec);
            e.dev_out = e.produces_token ? row : -1;
            e.dev_aux = e.n_spec;
            if (e.produces_token) row += 1 + e.n_spec;
            k += e.n_tokens;
        }
    }
    /* Every snapshot this step restores from must still be in the store. The prefix cache keeps a
     * handed-out snapshot's slot until the next step begins (PrefixCache::begin_step), so this
     * should always hold; it is checked because restoring from a released slot reads another
     * session's state. A step that fails it is abandoned like a failed staging (below): the
     * request gives its hit back and is prefilled again. */
    bool lost_restore = false;
    for (const StepEntry& e : plan_.e) {
        if (e.ckpt_restore < 0 || kv_->checkpoint_held(e.ckpt_restore)) continue;
        RAD_ERR("sequence %llu was to resume from checkpoint slot %d, which the store no longer "
                "holds; its hit is given back and its prompt prefilled again",
                (unsigned long long)e.seq, (int)e.ckpt_restore);
        lost_restore = true;
    }

    step_start_ = std::chrono::steady_clock::now();
    const RadBatch* b = lost_restore ? nullptr : builder_.build(plan_, *kv_, step_);
    if (!b) {
        /* Staging failed. Nothing was computed, so nothing is committed and n_computed did not
         * advance: the next step re-schedules the same chunks on the KV already allocated. The
         * step is not left open, or the engine would deadlock waiting for its commit.
         *
         * A restore cannot be re-scheduled: the entry consumed its checkpoint slot, and the next
         * step would run on as if the recurrent state were in place. Such a request gives its
         * hit back and returns to the queue for a fresh lookup. */
        for (StepEntry& e : plan_.e) {
            SchedReq& s = *reqs_[(size_t)e.slot];
            s.entry = -1;
            if (e.ckpt_restore < 0 || !s.req) continue;
            undo_hit(s);
            s.running = false;
            s.req->state = ReqState::Waiting;
            queue_erase(running_, s.slot);
            queue_insert(waiting_, s.slot);
        }
        plan_.reset();
        step_open_ = false;
        return nullptr;
    }
    ++step_;
    step_open_ = true;
    return b;
}

/* ------------------------------------------------------------------ media */
bool Scheduler::prefill_continues() const {
    std::lock_guard<std::mutex> lk(mu_);
    /* Only the step being released has a plan to read. Between a commit and the next step (an
     * encoder pass is prepared there) `plan_` is the committed step's, whose requests in_step() no
     * longer holds: a cancel and reap may have released one (#127). */
    if (stage_rows_ <= 0 || !step_open_) return false;
    /* The next step is held to decode rows (--prefill-interleave): experts staged for a chunk now
     * would be dropped by that step. */
    if (interleave_left_ > 0) return false;
    for (const StepEntry& e : plan_.e) {
        if (!e.is_prefill || e.slot < 0 || (size_t)e.slot >= reqs_.size() || !reqs_[(size_t)e.slot])
            continue;
        const int64_t left = reqs_[(size_t)e.slot]->seq_len() - ((int64_t)e.ctx_len + e.n_tokens);
        if (left >= stage_rows_) return true;
    }
    return false;
}

bool Scheduler::media_ready(SchedReq& s) {
    bool ready = true;
    for (mm::Placed& p : s.req->media->items) {
        mm::Item& it = *p.item;
        if (it.encoded) continue;
        ready = false;
        if (!it.queued) {
            it.queued = true;
            enc_want_.push_back(p.item);
        }
    }
    return ready;
}

bool Scheduler::encodes_pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !enc_want_.empty();
}

void Scheduler::take_encodes(std::vector<std::shared_ptr<mm::Item>>* out) {
    std::lock_guard<std::mutex> lk(mu_);
    out->clear();
    out->swap(enc_want_);
}

/* ------------------------------------------------------------------ the step's head depth */
void Scheduler::draft_cycle() {
    if (draft_.cost_policy()) { draft_cost_cycle(); return; }
    if (draft_.adaptive()) {
        /* The rows the opening step will decode: every running sequence past its prompt. Those
         * that may speculate choose D through their estimates; the rest emit one token each, which
         * still counts in the step's yield and cost. A sequence still prefilling is neither. */
        draft_est_.clear();
        int n_rows = 0;
        for (int32_t slot : running_) {
            SchedReq& s = *reqs_[(size_t)slot];
            if (!s.req || !s.running) continue;
            const Request& r = *s.req;
            if (r.n_computed < (int64_t)r.prompt.size() || r.n_computed + 1 < s.seq_len())
                continue;
            ++n_rows;
            if (r.may_speculate) draft_est_.push_back(&s.adapt);
        }
        (void)draft_.plan_step(draft_est_.data(), (int)draft_est_.size(), n_rows);
    }
    if (draft_record_) draft_record_cycle();
}

/* The decode rows at this cycle's open, one view each. Both the cost policy's cycle and the
 * calibration recording name the same rows, so the view is built in one place. A row still
 * prefilling is not in it: it emits nothing to speculate about. */
int32_t Scheduler::draft_fill_view() {
    int32_t n = 0;
    for (int32_t slot : running_) {
        SchedReq& s = *reqs_[(size_t)slot];
        if (!s.req || !s.running) continue;
        const Request& r = *s.req;
        if (r.n_computed < (int64_t)r.prompt.size() || r.n_computed + 1 < s.seq_len()) continue;
        if (n >= draft_view_.cap) break;

        DraftSeqView& v = draft_view_seq_[(size_t)n++];
        v = DraftSeqView{};
        v.req_id   = r.id;
        v.slot     = s.slot;
        v.slot_gen = s.gen;
        v.ctx_len  = r.n_computed > INT32_MAX ? INT32_MAX : (int32_t)r.n_computed;
        v.ctx_class = draft_ctx_class(v.ctx_len);
        int64_t room = r.max_tokens > 0 ? r.max_tokens - (int64_t)r.output.size()
                                        : (int64_t)INT32_MAX;
        if (ctx_limit_ > 0 && ctx_limit_ - s.seq_len() < room) room = ctx_limit_ - s.seq_len();
        v.remaining = room > INT32_MAX ? INT32_MAX : room > 0 ? (int32_t)room : 0;
        v.sampling  = r.sp.temp > 0.0f ? DraftSampling::Sampled : DraftSampling::Greedy;
        v.prefill   = false;
        /* A cleared may_speculate means a currently constraining grammar, unless the request is a
         * scored prompt (which never drafts either). Name the reason, not "no drafter":
         * draft_types.h keeps them apart. */
        v.eligibility = !r.may_speculate
                            ? (r.token_scores ? DraftEligibility::NoDrafter
                                              : DraftEligibility::Grammar)
                      : v.remaining <= 0 ? DraftEligibility::OutputExhausted
                                         : DraftEligibility::Eligible;
    }
    return n;
}

/* The calibration cycle. The cost policy is off, so nothing here may issue a plan to the
 * controller: draft_cap() and bars_fill() read the controller's plan, and setting one would
 * change what this run drafts. The record is the ring's, and its plan states what the
 * legacy/adaptive sites can actually do this cycle:
 *
 *   - k[i] is the executed depth, min(the step's window, the head's depth), for every row. The
 *     serial rounds are capped by next_depth() and by the deepest adaptive window, and the
 *     head's own depth can be shallower than either; nothing publishes past that, and a plan
 *     that claimed the ceiling would file every cycle as a truncated run (observe()).
 *   - fill[] is Serial where k[] is positive, because that is the drafter the format can name.
 *     The lookup fallback is deliberately not named (the n-gram site in step()); its verify
 *     still lands as label_origin, and the writer accepts that under a serial bucket.
 *   - pipeline_depth is 1: the head's proposals are verified by the paired step, exactly as
 *     under the cost policy.
 *
 * The cycle id is the same counter the cost policy uses, so the ring's records never collide. */
void Scheduler::draft_record_cycle() {
    draft_ring_close(draft_ring_cur_);
    draft_ring_cur_ = 0;
    int32_t n = draft_fill_view();
    /* A cycle with more rows than the widest verify window cannot be priced at all
     * (draft_shape_priceable), and naming a prefix of it would describe a cycle the run did not
     * execute -- shorter than the bracket step_finished times. The cycle is not recorded, and said
     * so once. */
    if (n > RAD_SCHED_MAX_SPEC) {
        ++draft_cycles_dropped_;
        if (!draft_record_over_rows_warned_) {
            draft_record_over_rows_warned_ = true;
            RAD_WARN("draft cost: %d decode rows exceed the %d the profile format can name; the "
                     "cycle is not recorded (further ones are counted only)", n,
                     (int)RAD_SCHED_MAX_SPEC);
        }
        return;
    }
    if (n <= 0) return;

    /* The plan's arrays are the stack's, unlike the cost policy's: the record copies them, and
     * nothing else reads this plan. */
    int32_t kk[RAD_SCHED_MAX_SPEC + 1] = {};
    DraftDrafter fill[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t nj[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t hist[RAD_SCHED_MAX_SPEC + 1] = {};
    const int32_t prev = draft_record_prev_k_ >= 0 ? draft_record_prev_k_ : 0;
    int32_t kmax = 0;
    for (int32_t i = 0; i < n; ++i) {
        const DraftSeqView& v = draft_view_seq_[(size_t)i];
        /* What the step will really cap this row at: the head cannot publish past its own depth,
         * and the step sites refuse a row that may not speculate. Naming the window for every row
         * would file a shape more rows drafted than the run can execute. */
        int32_t ki = 0;
        if (v.eligibility == DraftEligibility::Eligible && (size_t)v.slot < reqs_.size()) {
            ki = draft_cap(*reqs_[(size_t)v.slot]);
            if (ki > draft_head_depth_) ki = draft_head_depth_;
            if (ki < 0) ki = 0;
            if (ki > RAD_SCHED_MAX_SPEC) ki = RAD_SCHED_MAX_SPEC;
        }
        kk[i]   = ki;
        fill[i] = ki > 0 ? DraftDrafter::Serial : DraftDrafter::None;
        /* The paired step verifies the window the previous cycle drafted (the cost plan's rule);
         * with no controller plan there is nothing to freeze, so the previous record's depth
         * stands in -- legacy chooses the same window every cycle unless the window moved. */
        hist[i] = 1 + prev;
        if (ki > kmax) kmax = ki;
    }
    DraftPlan p;
    p.cycle_id       = ++draft_cycle_id_;
    p.policy_version = RAD_DRAFT_POLICY_VERSION;
    p.n_seq          = n;
    p.cap            = n;
    p.k_ceiling      = cfg_.n_spec;
    p.max_k          = kmax;
    p.verify_k       = kmax;
    p.pipeline_depth = 1;
    p.source         = DraftPlanSource::Legacy;
    p.prefill_tokens = 0;   /* step_finished stamps the step's own share on the record */
    p.k              = kk;
    p.fill           = fill;
    p.rows           = nj;
    p.history_tokens = hist;
    /* Identity and context class come from the view the cycle just built (draft_ring_open). */
    if (draft_plan_rows(p) != RAD_OK) return;
    draft_ring_open(p);
    if (draft_ring_cycle(p.cycle_id)) {
        draft_ring_cur_ = p.cycle_id;
        draft_record_prev_k_ = kmax;
    }
}

/* The cost policy opens its cycle here: one view of the decode rows, one frozen plan. The plan
 * decides each row's K for the whole cycle, K=0 included: a speculation-off decision is not a
 * vacancy the n-gram fallback may fill (docs/ADAPTIVE-MTP.md). */
void Scheduler::draft_cost_cycle() {
    if (!draft_.cost_policy()) return;
    /* After a divergence there is no cycle this profile may price: it was measured on a placement
     * the run has left. Opening the cycle anyway would leave draft_cap() at the ceiling, which is
     * the implicit fallback the admission rule refuses at init; the cycle is refused and the next
     * planning site reports it, so the engine ends the run on its error path. A record-only
     * calibration prices nothing to begin with and keeps recording (draft_drift_note). */
    if (draft_drift_ && !draft_costs_.records_only()) {
        if (!draft_cost_aborted_) {
            draft_cost_aborted_ = true;
            RAD_ERR("draft cost: %s; the cost run cannot price another cycle",
                    draft_costs_.reason().empty()
                        ? "the residency no longer matches the profile"
                        : draft_costs_.reason().c_str());
        }
        return;
    }
    /* Everything the last plan's rows will ever carry is in by now -- its verify at the paired
     * step's commit, its drafting from the publications after it -- and B should hold those labels
     * before it predicts this cycle. */
    draft_ring_close(draft_ring_cur_);
    draft_ring_cur_ = 0;
    const int32_t n = draft_fill_view();

    draft_view_.cycle_id       = ++draft_cycle_id_;
    draft_view_.policy_version = RAD_DRAFT_POLICY_VERSION;
    draft_view_.n_seq          = n;
    draft_view_.k_ceiling      = cfg_.n_spec;
    /* Publish-then-verify: the step paired with this plan verifies what the previous cycle
     * published, so this plan's draft distribution reaches a verify one cycle later. */
    draft_view_.pipeline_depth = 1;
    /* From the last bracket that completed, not from what the runtime *could* do: a recording
     * deployment whose tape stays warm plays, and only an observed state can say so. The initial
     * value is the capability answer (`draft_cycle_replay_`, set at init), and a bracket with no
     * evidence leaves it as it was rather than claiming a state nobody saw. */
    draft_view_.replay         = draft_cycle_replay_;
    /* The step this cycle pairs with may add the step budget; the KV lookahead may spend what the
     * pool has free. Both are bounds read here, not measurements (the step itself is planned
     * after this cycle opens). */
    draft_view_.token_budget   = cfg_.max_tok > INT32_MAX ? INT32_MAX : (int32_t)cfg_.max_tok;
    const int64_t kv_free = kv_ ? kv_->free_blocks(0) * kv_->block_tokens(0) : 0;
    draft_view_.kv_budget      = kv_free > INT32_MAX ? INT32_MAX : (int32_t)(kv_free > 0 ? kv_free : 0);
    /* Prompt work still pending at the open. The chunk that shares this step is planned later, so
     * this is what could share it: 0 is exact when nothing is left to prefill, and non-zero keeps
     * a history/prefill-bearing step from being priced as decode-only. */
    int64_t pending = 0;
    for (int32_t slot : running_) {
        const SchedReq& s = *reqs_[(size_t)slot];
        if (!s.req || !s.running) continue;
        const int64_t left = (int64_t)s.req->prompt.size() - s.req->n_computed;
        if (left > 0) pending += left;
    }
    if (pending > cfg_.max_tok) pending = cfg_.max_tok;
    draft_view_.prefill_tokens = pending > INT32_MAX ? INT32_MAX : (int32_t)pending;
    /* This runtime has no n-gram switch: the fallback is always available, and k[] gates it per
     * row (bars_fill). */
    draft_view_.ngram_allowed  = true;

    const int rc = draft_.open_cycle(draft_view_, &draft_preds_);
    /* Opened, planned or not: the five ends below (priced, fallback, calib_forced, empty, no plan)
     * sum to this. */
    ++m_.draft_cost_cycles;
    if (rc < 0) {
        /* Counted apart from both a priced cycle and a fallback, so "the controller refused this
         * view" never reads as "the policy chose something". The cycle itself runs legacy. */
        ++m_.draft_cost_no_plan;
        /* A coverage gap is not a broken hook: no later cycle of this deployment can price the
         * offered top, so running on without a plan would draft at the ceiling and present the
         * very depth the ladder never offers. Latch the existing fail-closed abort instead: step()
         * and promote_next() then refuse to draft, and the engine stops the run through
         * fail_loop(). */
        if (draft_.planner().coverage_gap_refused()) {
            draft_cost_aborted_ = true;
            return;
        }
        /* No plan this cycle: draft_cap() falls back to next_depth() and the cycle runs legacy.
         * Logged once, so a broken hook does not flood the log every step. */
        if (!draft_cost_failed_)
            RAD_ERR("draft policy: cycle %llu has no plan (%s); drafting at the ceiling %d",
                    (unsigned long long)draft_view_.cycle_id, rad_strerror(rc), cfg_.n_spec);
        draft_cost_failed_ = true;
    } else {
        draft_cost_failed_ = false;
        /* The plan the next step's verify and this cycle's publications belong to; copied now,
         * because the controller reuses its two plan slots. */
        const DraftPlan* p = draft_.plan();
        draft_ring_cur_ = p->cycle_id;
        draft_ring_open(*p);
        /* An empty planning point: every step opens a cycle, and a pure prefill or idle step has
         * no decode rows to plan over, so the controller issues the valid K0 plan for an empty
         * view. It is neither a decision nor a measurement -- no ring record, no bracket, no
         * sample -- so it is counted apart, leaves the last plan's ledger alone (the `last_*`
         * scalars and the priced-tier count) and spends no trace line. A valid all-K0 plan over
         * real rows is still a decision (`last_rows > 0`). */
        if (p->n_seq <= 0) {
            ++m_.draft_cost_empty;
            return;
        }
        /* What the policy decided, for /stats: whether this cycle was priced at all, what it asked
         * for and why, and under which candidate. The replay is the candidate the plan declared --
         * the last bracket's observed state, not this cycle's (see draft_cycle_replay_). */
        if (p->source == DraftPlanSource::Cost) {
            ++m_.draft_cost_priced;
            /* The comparison the ledger has to show: how many of the ladder's depths were priced
             * on this cycle (deepest plus shallower ones that priced), so "the shallow depth lost"
             * is distinguishable from "it was never measured". */
            m_.draft_cost_tiers_priced = draft_.planner().last_tiers_priced();
        } else if (p->source == DraftPlanSource::Fallback) {
            ++m_.draft_cost_fallback;
            m_.draft_cost_tiers_priced = 0;
        } else if (p->source == DraftPlanSource::Calibration) {
            /* A forced plan with rows, counted apart from both decision sources so a measurement
             * never reads as a cost choice. A forced plan over an empty view is an `empty`
             * planning point above: it has no depth to measure, and only a plan with rows
             * advances the sequence. */
            ++m_.draft_calib_forced;
            m_.draft_cost_tiers_priced = 0;
            /* A plan-time resource cut (cap_of or the token/KV budget) is not kDraftSampleShortened
             * -- that flag is the bracket stopping below its plan. It is visible here instead. */
            for (int32_t i = 0; i < p->n_seq; ++i)
                if (p->why[i] == DraftFallback::Resources) { ++m_.draft_calib_cut; break; }
        }
        m_.draft_cost_last_k        = p->max_k;
        m_.draft_cost_last_replay   = (int64_t)p->replay;
        m_.draft_cost_last_source   = (int64_t)p->source;
        m_.draft_cost_last_fallback = (int64_t)p->fallback;
        m_.draft_cost_last_rows     = p->n_seq;
        draft_trace_plan(*p);
    }
}

/* ------------------------------------------------------------------ the observation ring
 *
 * A cost plan is issued at the open and read row by row while the step is planned; `pipeline_depth`
 * 1 means the plan is chosen one step ahead of the step that spends its k[]. So a cycle's rows
 * carry two windows, exactly as DraftSeqObservation documents them:
 *
 *   - its drafting: what was published under this plan -- the head's proposals after this step's
 *     commit, which the next step verifies;
 *   - its paired verify: the step of this cycle, which verifies the frozen previous plan's window.
 *
 * B's predictor must be shown exactly one observation per completed cycle, and it must name the
 * plan the rows came from -- the controller hands out its plan storage again, so the record owns a
 * copy of the plan and its rows.
 *
 * Lifecycle: open at issuance, report the verify at the paired step's commit, publish the drafting
 * while the proposals are made, deliver at the next open. By then nothing can touch the record
 * again, and the labels reach the predictor before it predicts the cycle that follows. A row whose
 * request left first (preempted, cancelled) never reports, so the cycle is incomplete and is
 * counted rather than delivered (draft_observation_deliverable's `complete`). Delivery at the next
 * open also bounds a record's life to one cycle; the ring's eviction is only the backstop for a
 * cycle that never got another open.
 */
void Scheduler::draft_ring_open(const DraftPlan& p) {
    if (p.n_seq <= 0 || p.n_seq > p.cap) return;
    if ((size_t)p.n_seq > draft_view_seq_.size()) return;
    /* A plan that carries no request identity of its own -- the calibration's, which was built
     * for this one record -- is named by the view the cycle just built. The cost policy's plan
     * carries its own and is copied verbatim. */
    const bool view_identity = p.req_id == nullptr;
    DraftCycleRecord* r = nullptr;
    for (int32_t i = 0; i < (int32_t)kDraftRing; ++i)
        if (!draft_ring_[i].active) { r = &draft_ring_[i]; break; }
    if (!r) {
        r = &draft_ring_[0];
        for (int32_t i = 1; i < (int32_t)kDraftRing; ++i)
            if (draft_ring_[i].cycle_id < r->cycle_id) r = &draft_ring_[i];
        ++draft_cycles_dropped_;
    }
    r->active   = true;
    r->cycle_id = p.cycle_id;
    r->n_seq    = p.n_seq;
    /* The bracket is not established yet: the step that owns this cycle has not even been built,
     * and only step_finished() can say what it cost and what the tape did in it. */
    r->timed           = false;
    r->total_ms        = 0.0;
    r->prefill_tokens  = 0;
    r->admitted0       = m_.admitted;
    r->replay          = DraftReplay::Unknown;
    r->placed          = 0;
    /* Scalars first; the arrays below are the record's own storage and never move after init. */
    r->plan.cycle_id       = p.cycle_id;
    r->plan.policy_version = p.policy_version;
    r->plan.n_seq          = p.n_seq;
    r->plan.cap            = p.cap;
    r->plan.k_ceiling      = p.k_ceiling;
    r->plan.max_k          = p.max_k;
    r->plan.verify_k       = p.verify_k;
    r->plan.pipeline_depth = p.pipeline_depth;
    r->plan.source         = p.source;
    r->plan.fallback       = p.fallback;
    r->plan.probe          = p.probe;
    r->plan.replay         = p.replay;
    r->plan.timing_valid   = false;   /* the planner does not time; the bracket does */
    r->plan.prefill_tokens = p.prefill_tokens;
    for (int32_t i = 0; i < p.n_seq; ++i) {
        const DraftSeqView& v = draft_view_seq_[(size_t)i];
        r->id[i]             = view_identity ? v.req_id : p.req_id[i];
        r->slot[i]           = view_identity ? v.slot : p.slot[i];
        r->gen[i]            = view_identity ? v.slot_gen : p.slot_gen[i];
        r->k[i]              = p.k[i];
        r->fill[i]           = p.fill[i];
        r->ctx_class[i]      = view_identity ? v.ctx_class : p.ctx_class[i];
        r->history_tokens[i] = p.history_tokens ? p.history_tokens[i] : v.ctx_len;
        r->sampled[i]        = draft_view_seq_[(size_t)i].sampling == DraftSampling::Sampled ? 1 : 0;
        r->seq[i]            = DraftSeqObservation{};
        r->state[i]          = 0;
    }
    for (int32_t j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) r->rows[j] = p.rows[j];
}

Scheduler::DraftCycleRecord* Scheduler::draft_ring_cycle(uint64_t cycle_id) {
    for (int32_t i = 0; i < (int32_t)kDraftRing; ++i)
        if (draft_ring_[i].active && draft_ring_[i].cycle_id == cycle_id) return &draft_ring_[i];
    return nullptr;
}

/* The head's proposals for the step this plan governs, published after the step's commit. The
 * lookup fill is not published (see the n-gram site in step()): its verify still counts --
 * `verified` is not chained to `published` -- and B does not learn from a row with no drafting.
 * `generated` is what the head executed for the row, `published` what the paired step will verify:
 * under the adaptive window the commit can cut a row's verify below the window it drafted, and A
 * must price the work that ran (draft_cost_observe). */
void Scheduler::draft_ring_publish(const SchedReq& s, int32_t generated, int32_t published,
                                   DraftOrigin origin) {
    if (!draft_.cost_policy() && !draft_record_) return;
    if (!s.req) return;
    /* The controller names the cycle it planned; the calibration's record has no controller plan,
     * and draft_ring_cur_ is the cycle the recording just opened. */
    const DraftPlan* p = draft_.cost_policy() ? draft_.plan() : nullptr;
    DraftCycleRecord* r = draft_ring_cycle(p ? p->cycle_id : draft_ring_cur_);
    if (!r) return;
    if (generated < 0) generated = 0;
    if (generated > RAD_SCHED_MAX_SPEC) generated = RAD_SCHED_MAX_SPEC;
    if (published < 0) published = 0;
    if (published > generated) published = generated;
    for (int32_t j = 0; j < r->n_seq; ++j) {
        if (r->id[j] != s.req->id || r->slot[j] != s.slot || r->gen[j] != s.gen) continue;
        /* No `state` gate: the row's own verify was reported at this step's commit, and the
         * drafting is published after it -- both halves land in the same record. `state` answers
         * only whether the verify is in, which is what delivery waits for. */
        r->seq[j].generated = (uint8_t)generated;
        r->seq[j].published = (uint8_t)published;
        r->seq[j].origin    = published > 0 ? origin : DraftOrigin::None;
        break;
    }
}

void Scheduler::draft_ring_report(const SchedReq& s, const StepEntry& e, int32_t accepted,
                                  uint64_t raw_match) {
    if (!draft_.cost_policy() && !draft_record_) return;
    if (!e.req) return;
    int32_t v = e.n_spec;
    if (v < 0) v = 0;
    if (accepted < 0) accepted = 0;
    if (accepted > v) accepted = v;
    if (raw_match < (uint64_t)accepted) raw_match = (uint64_t)accepted;
    if (raw_match > (uint64_t)v) raw_match = (uint64_t)v;
    /* The verify paired with this cycle's plan: step n verifies the frozen window of plan n-1, and
     * its commits are plan n's `accepted` (DraftSeqObservation). */
    DraftCycleRecord* r = draft_ring_cycle(draft_ring_cur_);
    if (!r) return;
    for (int32_t j = 0; j < r->n_seq; ++j) {
        if (r->id[j] != e.req->id || r->slot[j] != s.slot || r->gen[j] != s.gen) continue;
        if (r->state[j] != 0) break;
        DraftSeqObservation& o = r->seq[j];
        o.verified  = (uint8_t)v;
        o.accepted  = (uint8_t)accepted;
        o.raw_match = raw_match;
        /* The labels the verify actually produced: the accepted prefix plus the first rejection.
         * A raw tally above the commits is a matched-but-uncommitted stop, which the tally rule
         * leaves uncensored (DraftSeqEnd::Stopped) rather than reading as a rejection. */
        if (raw_match == (uint64_t)accepted) {
            o.checked = (uint8_t)(accepted == v ? accepted : accepted + 1);
            o.end = accepted == v ? DraftSeqEnd::AllAccepted : DraftSeqEnd::Rejected;
        } else {
            o.checked = (uint8_t)accepted;
            o.end = DraftSeqEnd::Stopped;
        }
        /* The provenance of the window this verify actually consumed, which is NOT this plan's
         * own drafting (`origin`/`generated`, published after this commit): the step verifies the
         * head proposals the previous plan published, or the prompt-lookup fallback this step's
         * planning filled itself. B must be told which, or a row whose plan published nothing
         * would hide the head's labels -- and a row whose plan publishes next step would teach
         * from a lookup's matches (draft_predictor.cpp's learn() gates on this, not on origin). */
        if (v <= 0) {
            o.label_origin    = DraftOrigin::None;
            o.label_generated = 0;
        } else if (e.ngram) {
            o.label_origin    = DraftOrigin::Lookup;
            o.label_generated = (uint8_t)v;
        } else {
            o.label_origin    = DraftOrigin::Head;
            o.label_generated = (uint8_t)v;
        }
        r->state[j] = 1;
        return;
    }
}

/* A row that will never report: the request left the running set (preempted, cancelled, failed)
 * before the step paired with its cycle committed. The cycle it was in is no longer complete, so
 * it is dropped at the next open rather than delivered with a row nobody measured. */
void Scheduler::draft_ring_drop_row(uint64_t req_id, int32_t slot, uint32_t gen) {
    for (int32_t i = 0; i < (int32_t)kDraftRing; ++i) {
        DraftCycleRecord& r = draft_ring_[i];
        if (!r.active) continue;
        for (int32_t j = 0; j < r.n_seq; ++j) {
            if (r.id[j] != req_id || r.slot[j] != slot || r.gen[j] != gen) continue;
            if (r.state[j] == 0) r.state[j] = 2;
            break;
        }
    }
}

/* The cycle is over: every row has reported or will never. Delivered at most once, and only when
 * every row of the plan is in the record -- a cycle missing a row is not a smaller observation, it
 * is a different one. The record is retired whatever the predictor answers: a refused observation
 * is a dropped cycle, not a retried one.
 *
 * A's sample asks a different question: it prices what RAN, so a row that left the cycle belongs
 * in the measurement -- as kDraftSampleShortened, which keeps the bucket out of pricing -- while B
 * still receives only a cycle without a hole. */
void Scheduler::draft_ring_close(uint64_t cycle_id) {
    DraftCycleRecord* rp = draft_ring_cycle(cycle_id);
    if (!rp) return;
    DraftCycleRecord& r = *rp;
    bool complete = r.n_seq > 0, resolved = r.n_seq > 0;
    for (int32_t j = 0; j < r.n_seq; ++j) {
        if (r.state[j] == 0) complete = resolved = false;
        else if (r.state[j] == 2) complete = false;
    }
    if (resolved) draft_cost_observe(r);
    if (complete && draft_.cost_policy()) {
        DraftObservation o;
        o.cycle_id       = r.cycle_id;
        o.policy_version = r.plan.policy_version;
        o.n_seq          = r.n_seq;
        o.cap            = r.plan.cap;
        o.plan           = &r.plan;
        o.req_id         = r.id.data();
        o.slot           = r.slot.data();
        o.slot_gen       = r.gen.data();
        o.seq            = r.seq.data();
        o.complete       = true;
        const int rc = draft_predict_.observe(o);
        if (rc < 0) {
            /* Once, not once per cycle: a wiring gap (e.g. a plan whose fill[] cannot name the
             * drafter that really filled a row) would otherwise log on every step. The counter
             * carries the rest. */
            if (!draft_predict_refused_) {
                draft_predict_refused_ = true;
                RAD_WARN("draft predictor: cycle %llu refused (%s); further refusals are counted "
                         "only", (unsigned long long)r.cycle_id, rad_strerror(rc));
            }
            ++draft_cycles_dropped_;
        } else {
            ++draft_cycles_done_;
        }
    } else if (complete) {
        /* Recording: the cycle went to the table above, and B (the predictor) belongs to the cost
         * policy's runtime, not to the legacy run this calibration leaves untouched. */
        ++draft_cycles_done_;
    } else {
        ++draft_cycles_dropped_;
    }
    r.active = false;
    r.n_seq  = 0;
}

/* The last cycle has no next open: shutdown must deliver what is already fully measured before
 * the request identities are released, and count what never completed as dropped (visible in
 * SchedMetrics::draft_obs_discarded) rather than leave it in flight. Called at shutdown after the
 * requests are closed -- so a pending row has been marked dropped -- and again at fini, where
 * every record is already retired and the call is a no-op. */
void Scheduler::draft_ring_flush() {
    draft_ring_close(draft_ring_cur_);
    draft_ring_cur_ = 0;
    for (int32_t i = 0; i < (int32_t)kDraftRing; ++i)
        if (draft_ring_[i].active) draft_ring_close(draft_ring_[i].cycle_id);
}

/* The same close from outside, for an engine that is still up: serve() has joined the step and
 * rank threads before it exports, so no cycle is in flight and what is closed here is what
 * shutdown would have delivered. */
void Scheduler::flush_draft_observations() {
    std::lock_guard<std::mutex> lk(mu_);
    draft_ring_flush();
}

/* A's sample for one resolved cycle. The price is the cost of the distribution that RAN
 * (draft_cost.h): `rounds`/`draft_round_rows` come from what each row executed, the verify window
 * from what each row actually verified. `shape.draft_k`/`draft_rows` stay the plan's -- provenance
 * and the upper bound -- but they are not the key. */
void Scheduler::draft_cost_observe(DraftCycleRecord& r) {
    DraftCostSample s;
    s.cycle_id       = r.cycle_id;
    s.policy_version = r.plan.policy_version;
    DraftVerifyShape& sh = s.shape;
    sh.n_rows         = r.n_seq;
    /* The plan's shape takes the deepest row's class (draft_policy.cpp's shape_of); the price has
     * to be filed under that same class, or an estimate looked up for a long-context plan lands in
     * a bucket no measurement can ever reach. */
    sh.ctx_class      = 0;
    for (int32_t i = 0; i < r.n_seq; ++i)
        if (r.ctx_class[i] > sh.ctx_class) sh.ctx_class = r.ctx_class[i];
    sh.draft_k        = r.plan.max_k;
    for (int32_t j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) sh.draft_rows[j] = r.rows[j];
    sh.pipeline_depth = r.plan.pipeline_depth;
    sh.prefill_tokens = r.prefill_tokens;
    sh.replay         = r.replay;
    s.replay          = r.replay;   /* A keys on the sample's, and requires the shape's to agree */
    /* Every fill the policy produces is the serial head (draft_policy.cpp's finish()); the lookup
     * fill is never published to the ring and so never names a row here. */
    sh.drafter  = r.plan.k_ceiling > 0 ? DraftDrafter::Serial : DraftDrafter::None;
    sh.sampling = DraftSampling::Greedy;
    int32_t vk = 0;
    for (int32_t i = 0; i < r.n_seq; ++i) {
        if (r.sampled[i]) sh.sampling = DraftSampling::Sampled;
        if (r.state[i] == 1 && (int32_t)r.seq[i].verified > vk)
            vk = (int32_t)r.seq[i].verified;
    }
    /* A row that left did not verify, so only reported rows count here. n_rows stays the plan's
     * (draft_shape_matches_plan pins it), which leaves a dropped row's bonus token overstated --
     * the cycle carries Shortened and is never priced. */
    sh.verify_k = vk;
    for (int32_t i = 0; i < r.n_seq; ++i) {
        if (r.state[i] != 1) continue;
        for (int32_t j = 1; j <= (int32_t)r.seq[i].verified; ++j) ++sh.rows[j];
    }
    sh.verify_tokens = r.n_seq;
    for (int32_t j = 1; j <= vk; ++j) sh.verify_tokens += sh.rows[j];
    /* serial: one history row per verified position, plus the lead row. */
    sh.history_tokens = sh.verify_tokens;

    /* The extent that RAN: `generated` is the rounds the head executed for the row, not what the
     * paired step verified. Under the adaptive window the commit can cut a row's verify below the
     * window it drafted (expect_draft), and reconstructing the extent from the publication would
     * time rounds the bracket did not execute. A row that executed past its plan entry fails
     * executed_within_plan in observe() -- visibly, rather than priced at a window it did not run. */
    for (int32_t i = 0; i < r.n_seq; ++i)
        if ((int32_t)r.seq[i].generated > s.rounds) s.rounds = (int32_t)r.seq[i].generated;
    for (int32_t i = 0; i < r.n_seq; ++i)
        for (int32_t j = 1; j <= (int32_t)r.seq[i].generated; ++j) ++s.draft_round_rows[j];
    for (int32_t j = 1; j <= s.rounds; ++j) s.draft_rows_total += s.draft_round_rows[j];
    for (int32_t i = 0; i < r.n_seq; ++i) s.drafted += (int32_t)r.seq[i].generated;

    /* The provenance of each verify row: the state the verify window consumed, not the origin this
     * cycle published (which is its own drafting). Without it the writer would have to default
     * rows to none and price a head row as a lookup row. */
    if (r.n_seq <= RAD_SCHED_MAX_SPEC) {
        for (int32_t i = 0; i < r.n_seq; ++i) s.origins[i] = r.seq[i].label_origin;
        s.origins_valid = 1;
    }

    if (r.timed) s.flags |= kDraftSampleValid;
    /* The bracket timed less than the shape it would be filed under: the deepest planned round
     * never ran, or a row left before its verify. The global round count alone cannot tell -- one
     * row can leave while the deepest round still ran for the others. */
    if (s.rounds < r.plan.max_k) s.flags |= kDraftSampleShortened;
    for (int32_t i = 0; i < r.n_seq; ++i)
        if (r.state[i] == 2) s.flags |= kDraftSampleShortened;
    if (m_.admitted != r.admitted0) s.flags |= kDraftSampleAdmission;
    if (r.placed) s.flags |= kDraftSamplePlacement;
    if (r.replay == DraftReplay::Record) s.flags |= kDraftSampleRecord;
    else if (r.replay == DraftReplay::Audit) s.flags |= kDraftSampleAudit;
    s.total_ms = r.total_ms;

    DraftObservation o;
    o.cycle_id       = r.cycle_id;
    o.policy_version = r.plan.policy_version;
    o.n_seq          = r.n_seq;
    o.cap            = r.plan.cap;
    o.plan           = &r.plan;
    o.req_id         = r.id.data();
    o.slot           = r.slot.data();
    o.slot_gen       = r.gen.data();
    o.seq            = r.seq.data();
    o.complete       = true;
    const int rc = draft_costs_.observe(o, s);
    draft_trace_sample(s, rc);
    if (rc < 0) {
        if (!draft_cost_warned_) {
            draft_cost_warned_ = true;
            RAD_WARN("draft cost: cycle %llu not measured (%s); further refusals are counted only",
                     (unsigned long long)r.cycle_id, rad_strerror(rc));
        }
        ++draft_cost_refused_;
    }
}

/* ------------------------------------------------------------------ the next step */
void Scheduler::expect_draft(uint64_t id, int n) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return;
    SchedReq& s = *reqs_[(size_t)it->second];
    /* Under the adaptive window the rounds ran for the deepest sequence; this one verifies the
     * first of them up to its own window. The same holds for a cost plan, whose K per request is
     * the frozen choice. That is the cap serial_begin used: nothing commits between the two, and
     * D is the cycle's (draft_cycle). */
    if ((draft_.adaptive() || draft_.cost_policy()) && n > 0) {
        const int cap = draft_cap(s);
        if (n > cap) n = cap;
    }
    s.draft_next = n > 0 ? n : 0;
}

const RadBatch* Scheduler::step_next() {
    std::lock_guard<std::mutex> lk(mu_);
    const RadBatch* b = step_next_locked();
    if (!b)
        for (const StepEntry& e : plan_.e)
            if (e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot])
                reqs_[(size_t)e.slot]->draft_next = 0;
    return b;
}

const RadBatch* Scheduler::step_next_locked() {
    if (!step_open_ || next_open_ || plan_.e.empty()) return nullptr;
    /* An admission is a decision about the pool and the queue that the full step makes. */
    if (!waiting_.empty()) return nullptr;
    /* Once the hold is spent, the held chunk goes into the next full step rather than waiting
     * behind decode steps issued ahead. */
    if (held_prefill_ > 0 && interleave_left_ == 0) return nullptr;

    plan_next_.reset();
    for (const StepEntry& e : plan_.e) {
        if (e.slot < 0 || (size_t)e.slot >= reqs_.size()) return nullptr;
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req || s.req->state != ReqState::Running || !s.running) return nullptr;
        Request& r = *s.req;
        /* A mid-prompt chunk continues as a prefill from where the host knows it is. A grammar
         * that cannot rewind decides at commit whether the next step may speculate; one that can,
         * or no grammar, keeps its answer. */
        if (!e.produces_token || !r.speculation_fixed) return nullptr;

        /* The furthest context the row can reach: every position this step verifies committed,
         * counted from the last commit rather than this step's own bound (which would add a
         * window of slack per step issued ahead). The card knows the real position; this sizes
         * the pool, the block tables and every host-side ceiling. */
        const int64_t ctx_up = r.n_computed + e.n_tokens;
        const int32_t d      = s.draft_next;
        const int64_t want   = 1 + d;
        if (ctx_limit_ > 0 && ctx_up + want > ctx_limit_) return nullptr;
        if (kv_->ensure_from(r.id, ctx_up + want, r.n_computed) < 0) return nullptr;

        StepEntry n;
        n.req            = &r;
        n.seq            = r.id;
        n.slot           = e.slot;
        n.card           = e.card;
        n.n_tokens       = (int32_t)want;
        n.ctx_len        = (int32_t)ctx_up;
        n.n_spec         = d;
        n.n_accepted     = 1;          /* the card's: StepSlot::n_acc */
        n.conv_cursor    = s.conv_cursor;
        n.draft          = s.draft;    /* published before this step commits (set_draft) */
        n.n_draft        = d;
        n.is_prefill     = false;
        n.produces_token = true;
        n.dev            = true;
        plan_next_.e.push_back(n);
        plan_next_.n_tok += want;
        if (d > plan_next_.n_spec) plan_next_.n_spec = d;
    }
    plan_next_.phase        = RAD_PHASE_DECODE;
    plan_next_.n_seq_decode = (int64_t)plan_next_.e.size();
    plan_next_.n_tok_decode = plan_next_.n_tok;
    {
        int64_t k = 0;
        int32_t row = 0;
        for (StepEntry& e : plan_next_.e) {
            e.dev_tok = (int32_t)(k + e.n_tokens - 1 - e.n_spec);
            e.dev_out = row;
            e.dev_aux = e.n_spec;
            row += 1 + e.n_spec;
            k += e.n_tokens;
        }
    }
    const RadBatch* b = builder_.build(plan_next_, *kv_, step_);
    if (!b) { plan_next_.reset(); return nullptr; }
    for (size_t i = 0; i < plan_next_.e.size(); ++i)
        reqs_[(size_t)plan_next_.e[i].slot]->entry_next = (int32_t)i;
    next_open_ = true;
    return b;
}

void Scheduler::promote_next() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!next_open_) return;
    /* The other place a cycle opens: the step issued ahead becomes current without step(). Its
     * proposals were counted under the last cycle's D; the next ones are drafted under this one.
     * A refused cycle (a drifted residency) leaves the ahead plan unexecuted: what it was planned
     * under no longer describes the run, and the engine stops on the latch. */
    draft_cycle();
    if (draft_cost_aborted_) return;
    std::swap(plan_, plan_next_);
    plan_next_.reset();
    step_prefill_tok_ = 0;
    step_decode_tok_  = 0;
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        SchedReq& s = *reqs_[(size_t)plan_.e[i].slot];
        s.entry      = (int32_t)i;
        s.entry_next = -1;
        s.draft_next = 0;
        s.sched_tokens = plan_.e[i].n_tokens;
        if (s.req) { s.req->n_draft = plan_.e[i].n_spec; s.req->draft = s.draft; }
    }
    ++m_.steps_decode;
    /* A step issued ahead during a hold is one of its decode-only steps. */
    if (interleave_left_ > 0) {
        --interleave_left_;
        if (held_prefill_ > 0) ++m_.steps_held;
    }
    ++m_.steps_ahead;
    for (const StepEntry& e : plan_.e)
        if (e.n_spec == 0) ++m_.decode_rows_unspeculated;
    step_start_ = std::chrono::steady_clock::now();
    ++step_;
    step_open_ = true;
    next_open_ = false;
}

/* ------------------------------------------------------------------ commit */
int Scheduler::commit(const StepResult& res) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!step_open_) return RAD_E_STATE;
    if (res.n_seq != (int)plan_.e.size()) {
        RAD_ERR("commit of %d sequences against a step of %d", res.n_seq, (int)plan_.e.size());
        return RAD_E_INVAL;
    }
    step_open_ = false;

    /* What the serial bind reads after commit, copied here while the Requests are still held:
     * reap() may release a finished one before finish_serial_drafter runs. */
    committed_.clear();

    const int32_t W = conv_window();

    /* A mixed step leaves decode rows the same replay factors as a decode step (recurrent path
     * over [0, n_seq_decode), chunked over the rest), so their scratch is not invalidated; the
     * next verify needs it for rollback. This relies on step() sorting decode rows first: one
     * sorted into the prefill half would take the chunked path and leave stale factors. */
    for (int i = 0; i < res.n_seq; ++i) {
        const StepEntry& e = plan_.e[(size_t)i];
        SchedReq& s = *reqs_[(size_t)e.slot];
        Request& r = *s.req;
        if (r.state != ReqState::Running) {
            /* Cancelled between step and commit; its blocks were held until the device finished
             * with them. If the server also reaped it, the sweep at the end of this function
             * returns the slot once every entry is retired. */
            s.entry = -1;
            /* Unless the drafter's snapshot still names it, or the next step carries it: the
             * sweep below, or that step's commit, releases it once nothing does. */
            if (s.pending_free && !block_holds(r.id) && !(next_open_ && s.entry_next >= 0)) {
                free_kv(s);
                s.pending_free = false;
                --n_deferred_;
            }
            committed_.push_back({e.seq, (uint64_t)r.output.size(), false});
            continue;
        }

        /* `ignore_eos` drops the token's termination and nothing else: max_tokens and the stop
         * strings below still end the request. See Request::ignore_eos. */
        bool eos = res.eos && res.eos[i] && !r.ignore_eos;

        bool overflowed = false;
        /* A token whose requested logprobs could not be computed (TokenLogprobs::token -1) fails
         * the request with the reason; that token and everything after it is not sent. */
        bool lp_failed = false;

        /* The snapshot the engine just took is now readable. Marked here, not at reservation
         * (which happens at planning), so a lookup in between cannot adopt a slot still holding
         * the previous tenant's state. A later restore is enqueued on the same stream behind this
         * save, so issuing the copy is enough. */
        if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0 && pc_)
            pc_->commit_checkpoint(r, (int64_t)e.ctx_len + e.ckpt_tok + 1);

        if (e.is_prefill) {
            r.n_computed += e.n_tokens;
            /* The chunk is committed: a window group returns its blocks below the window now,
             * not at this sequence's next ensure(), which may be steps away. */
            kv_->trim_window(r.id, r.n_computed);
            step_prefill_tok_ += e.n_tokens;
            server::sink_prompt_progress(r.sink, r.n_computed, (int32_t)r.prompt.size(),
                                         r.n_cached);
            /* Once, when the prompt is fully placed: the usage block's admission accounting.
             * `n_cached` is what the prefix cache supplied. */
            if (r.n_computed >= (int64_t)r.prompt.size())
                server::sink_prompt_stats(r.sink, (int32_t)r.prompt.size(), r.n_cached);
            if (e.produces_token && res.token && r.embed) {
                /* The answer is the pooled vector the engine already put in the sink; there is
                 * no token to emit, and `eos` ends the request below. */
                if (!s.first_tok_us) s.first_tok_us = mono_us();
            } else if (e.produces_token && res.token) {
                if (!s.first_tok_us) s.first_tok_us = mono_us();
                r.output.push_back(res.token[i]);
                const TokenLogprobs* lp =
                    res.lp && res.first_row[i] >= 0 && r.sp.n_probs >= 0
                        ? &res.lp[res.first_row[i]] : nullptr;
                if (lp && lp->token < 0) lp_failed = true;
                else overflowed = !server::sink_push(r.sink, res.token[i], lp);
                /* The conv window absorbed this chunk; the cursor advances one committed
                 * position, as in decode. */
                s.conv_cursor = (int32_t)((s.conv_cursor + 1) % (W ? W : 1));
            }
        } else {
            int32_t a = res.n_accepted ? res.n_accepted[i] : 0;
            if (a < 0) a = 0;
            if (a > e.n_spec) a = e.n_spec;
            /* The raw tally is the comparison behind `a`, never below it: the EOG cut shortens what
             * was committed, not what matched. Clamped to the paired verify window this row was
             * compared over (`e.n_spec`), never to `a` or to the next cycle's draft K. */
            int32_t raw = res.n_raw ? res.n_raw[i] : a;
            if (raw < a) raw = a;
            if (raw > e.n_spec) raw = e.n_spec;
            /* The verify half of the cycle whose proposals this step ran. */
            draft_ring_report(s, e, a, (uint64_t)raw);
            /* A row issued ahead was staged at the furthest reachable context; it really
             * continued from the committed end. */
            const int64_t ctx0 = e.dev ? r.n_computed : (int64_t)e.ctx_len;

            /* Paged KV rolls back: blocks past the accepted position are freed, since KV written
             * for rejected tokens would be attended to later.
             *
             * Linear state does not. The recurrent state and conv window already absorbed the
             * rejected tokens; libr4d's conv update treats the state cache as a rolling window of
             * width-1 + num_spec entries and reads at the slot the last accepted token left, so a
             * rejection changes the read offset instead of recomputing (spec §10). */
            int64_t valid = ctx0 + 1 + a;
            /* Not while a step issued ahead writes past `valid`; that step's commit rolls back to
             * the end it reaches. */
            if (a < e.n_spec && !(next_open_ && s.entry_next >= 0)) {
                int st = kv_->rollback(r.id, valid);
                if (st < 0) RAD_WARN("kv rollback for request %llu: %s",
                                     (unsigned long long)r.id, rad_strerror(st));
            }
            s.conv_cursor = (int32_t)((s.conv_cursor + 1 + a) % (W ? W : 1));
            r.n_accepted = a;

            /* Accepted drafts first, then the verified token: the sink is a stream, so order here
             * is order in the client's text.
             *
             * The budget is checked per token, because a verify step emits 1 + a at once.
             * Trimming `output` afterwards would leave the overshoot already streamed, so the
             * same prompt at the same max_tokens would come back longer with speculation. */
            int64_t room = r.max_tokens > 0
                         ? r.max_tokens - (int64_t)r.output.size() : (int64_t)1 << 62;
            /* The context bounds it whatever max_tokens says, including no limit. */
            if (ctx_limit_ > 0 && ctx_limit_ - s.seq_len() < room) room = ctx_limit_ - s.seq_len();
            /* `j` is the token's place in the step's emission: its logprobs row is the entry's
             * j-th, when the request asked for them. */
            auto emit = [&](int32_t t, int32_t j) {
                if (room <= 0) return;
                --room;
                if (!s.first_tok_us) s.first_tok_us = mono_us();
                r.output.push_back(t);
                const TokenLogprobs* lp =
                    res.lp && res.first_row[i] >= 0 && r.sp.n_probs >= 0
                        ? &res.lp[res.first_row[i] + j] : nullptr;
                if (lp_failed || (lp && lp->token < 0)) { lp_failed = true; return; }
                if (!server::sink_push(r.sink, t, lp)) overflowed = true;
            };
            for (int32_t j = 0; j < a; ++j) emit(s.draft[j], j);
            if (res.token) emit(res.token[i], a);
            r.n_computed = valid;
            step_decode_tok_ += 1 + a;

            if (e.n_spec > 0) {
                draft_.observe(e.n_spec, a, raw);
                /* And this sequence's own estimate under the adaptive window, only when the head
                 * drafted: a prompt-lookup proposal is accepted on different text by a different
                 * rule. A move takes effect at the next draft; this step's drafter already ran. */
                if (draft_.adaptive() && !e.ngram) {
                    const int was = s.adapt.depth;
                    const int now = draft_.adapt_observe(s.adapt, e.n_spec, a);
                    if (now != was && was > 0 && (int)log_level() >= (int)Log::Debug) {
                        char est[8 * RAD_SCHED_MAX_SPEC];
                        int o = 0;
                        for (int j = 0; j < draft_.next_depth() && o < (int)sizeof est - 8; ++j)
                            o += std::snprintf(est + o, sizeof est - (size_t)o, "%s%.2f",
                                               j ? " " : "", (double)s.adapt.a[j]);
                        est[o] = 0;
                        RAD_DEBUG("draft window: request %llu stops at %d, was %d (expected "
                                  "tokens a verify %.3f -> %.3f; acceptance by position %s)",
                                  (unsigned long long)r.id, now, was, draft_.yield(s.adapt, was),
                                  draft_.yield(s.adapt, now), est);
                    }
                }
                m_.draft_tokens += e.n_spec;
                m_.draft_accepted += a;
                server::sink_drafts(r.sink, e.n_spec, a);
            }
        }
        s.n_draft = 0;
        r.n_draft = 0;
        s.entry = -1;

        /* A request the next step carries finishes now but keeps its blocks, which that step
         * writes; its commit finds the request not running and returns them. */
        const bool carried = next_open_ && s.entry_next >= 0;

        /* Finish. emit() already stops at max_tokens, so the resize below is a backstop: a >=
         * test that can only be == is easy to break quietly later. */
        if (overflowed) {
            /* The sink has already failed the request; stop generating and return its blocks. */
            close_request(s, ReqState::Failed, "sink_overflow", carried);
        } else if (lp_failed) {
            close_request(s, ReqState::Failed, "logprobs", carried, true, RAD_E_STATE,
                          "the logprobs of a token it generated could not be computed");
        } else if (server::sink_cancelled(r.sink)) {
            /* The client hung up: the blocks go back now (spec §14). */
            close_request(s, ReqState::Cancelled, "cancelled", carried);
        } else if (r.max_tokens > 0 && (int64_t)r.output.size() >= r.max_tokens) {
            r.output.resize((size_t)r.max_tokens);
            close_request(s, ReqState::Finished, "length", carried);
        } else if (ctx_limit_ > 0 && s.seq_len() >= ctx_limit_) {
            /* The context is full: the same finish as max_tokens, since the generation ran out
             * of room. */
            close_request(s, ReqState::Finished, "length", carried);
        } else if (eos) {
            close_request(s, ReqState::Finished, "stop", carried);
        }
        committed_.push_back({e.seq, (uint64_t)r.output.size(), r.state == ReqState::Running});
    }

    m_.prefill_tokens += step_prefill_tok_;
    m_.decode_tokens += step_decode_tok_;
    ++m_.steps;

    /* Every entry is retired, so anything deferred on this step, or on a drafter snapshot that
     * block_end() / serial_end() has since cleared, can be released: every pass naming them was
     * issued before this step's readback, which waited for the compute stream. */
    sweep_deferred();
    return RAD_OK;
}

/* ------------------------------------------------------------------ the end of a step */
/* The residency the profile was measured on, in one sticky decision. The evidence is cumulative
 * since the profile was bound; a move between two brackets -- the scheduler's own ticks run after
 * step_finished -- leaves no per-step stamp, so the engine hands the same count in before the next
 * cycle opens (draft_residency). Requires mu_. */
void Scheduler::draft_drift_note(int64_t moves_since_bind) {
    if (moves_since_bind <= 0 || draft_drift_) return;
    /* A loaded reference is an optimisation basis, not a recording of this run's placement. The
     * live run keeps its loans, flex moves, arena level and recall -- that is operation, not a
     * change to what the file measured (the plan's start placement with no borrowed copies), and
     * it neither invalidates the file nor teaches it. Nothing is deactivated or marked. */
    if (draft_costs_.reference()) return;
    if (!draft_.cost_policy() && !draft_record_) return;
    /* Sticky, because the run cannot move back to it: every later bracket is marked, and no
     * measurement after the move describes the start residency however clean it looks. */
    draft_drift_ = true;
    /* A record-only table is a calibration: it prices nothing (DraftCostTable::record_only), so a
     * move leaves no price of its own describing a placement the run has left. It keeps recording
     * -- the documented contract -- and every record after the move is marked `placed` and kept
     * out of the export. An online table stops pricing and stops learning instead. Neither is an
     * admit_fallback(): drift is no admission to price without the measurement the deployment was
     * admitted on, and the policy's own named baseline is not a substitute. */
    if (draft_.cost_policy() && draft_costs_.active() && !draft_costs_.records_only()) {
        draft_costs_.deactivate("residency drifted from the fingerprint's start placement");
        RAD_WARN("draft cost: %s; the loaded profile stops pricing and learning",
                 draft_costs_.reason().c_str());
    } else if (draft_record_) {
        RAD_WARN("draft cost: residency drifted from the fingerprint's start placement; "
                 "the calibration keeps recording and the export leaves every cycle after it "
                 "out");
    }
}

void Scheduler::draft_residency(int64_t moves_since_bind) {
    std::lock_guard<std::mutex> lk(mu_);
    draft_drift_note(moves_since_bind);
}

/* Closes the step's timing after the drafter has run. The next step's drafter runs after commit,
 * the first moment its inputs exist, and its rounds are real forward passes; timed at commit they
 * would be charged to nothing, so step time would read short and `decode_tps` high. A caller that
 * drives only commit() (the scheduler tests) gets counters and no timing.
 *
 * The rate is tokens over wall time across a decayed window, not an average of step rates: the
 * host does not wait on the card every step (a prefill chunk that samples nothing commits while
 * the card still runs it), so per-step rates are skewed by short intervals. Wall time runs from
 * the last step's end (so a step issued ahead is counted once), or from this step's start after
 * an idle gap. A step with no tokens of a kind still adds its time, or that rate would freeze.
 *
 * Host timing, always available. Per-op device timing adds synchronisations that change the
 * mover's slot supply, so profiled runs are not comparable (spec §16). */
void Scheduler::step_finished(const DraftStepEvidence& ev) {
    std::lock_guard<std::mutex> lk(mu_);
    if (step_start_.time_since_epoch().count() == 0) return;   /* no step has run */

    const auto step_end = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(step_end - step_start_).count();
    m_.step_ns += ns;
    if (plan_.phase == RAD_PHASE_DECODE) {
        m_.step_ns_decode += ns;
        /* The single-sequence floor over the step just timed (rad::DecodeFloor). A mixed step
         * never counts: its decode row shares the step with a prefill chunk. */
        if (plan_.n_seq_decode == 1) {
            decode_floor_.update(ns);
            m_.step_ns_decode_best  = decode_floor_.best();
            m_.step_ns_decode_recent = decode_floor_.recent();
        }
    }
    /* A's cycle bracket. This call is the only place that sees it: the ring's record is still
     * open -- the verify reported at commit, the drafting published after it -- and `ns` covers
     * the step and the draft rounds the engine ran after its commit. The plan's own copy stays
     * timing_valid = false; this one is the timed record observe() will read. */
    if (draft_.cost_policy() || draft_record_) {
        /* The profile's fingerprint names the residency the run started on. The move evidence the
         * rank froze covers the step's own bracket only; a move between two brackets -- the
         * scheduler's tier and arena ticks run after step_finished -- leaves no such stamp, so the
         * engine reports the cumulative count since the profile was bound, both here and before
         * each cycle open (draft_residency), and once it diverges every later bracket is marked:
         * no measurement after the move describes the start residency, however clean it looks. */
        draft_drift_note(ev.moves_since_bind);
        if (DraftCycleRecord* r = draft_ring_cycle(draft_ring_cur_)) {
            /* The candidate the next cycle may declare. This bracket is the newest observation
             * the planner has, and it is still only a prognosis for the next key: the sample
             * below stays the measurement, and the plan never becomes one (draft_cycle_replay_).
             * A bracket the ranks could not evidence leaves the candidate alone -- `Unknown` is
             * not a state they observed -- and so does a step with no cycle record, because no
             * sample came of it either. */
            if (ev.replay_known) draft_cycle_replay_ = ev.replay;
            draft_trace_bracket(ev);
            r->timed          = ns > 0;
            r->total_ms       = (double)ns / 1e6;
            r->plan.timing_valid = r->timed;
            r->replay         = ev.replay_known ? ev.replay : DraftReplay::Unknown;
            /* `moved` covers the bracket; `placement_unresolved` is the engine's state for it --
             * the start placement's fill had not landed, so the bracket measured a residency the
             * fingerprint does not name even when nothing retired inside it (the fill's copies
             * land over many, before and after this bracket). */
            r->placed         = (ev.staged || ev.moved || draft_drift_ || ev.placement_unresolved) ? 1 : 0;
            r->prefill_tokens = 0;
            for (const StepEntry& e : plan_.e)
                if (e.is_prefill) r->prefill_tokens += e.n_tokens;
            /* The plan keeps what it was priced on -- including the replay it declared -- and the
             * sample's shape carries what ran (draft_cost_observe). Only timing_valid is the
             * bracket's to say, because the planner cannot time its own cycle. */
        }
    }

    const auto   from = step_start_ - last_step_at_ < std::chrono::milliseconds(500)
                            ? last_step_at_ : step_start_;
    const double wall = std::chrono::duration<double>(step_end - from).count();
    /* The window ages by all elapsed time, idle included; only the busy part is added as time the
     * tokens took. */
    const double aged = std::chrono::duration<double>(step_end - last_step_at_).count();
    if (wall > 0) {
        constexpr double kWindowS = 1.0;
        const double f = std::exp(-aged / kWindowS);
        tps_win_prefill_ = tps_win_prefill_ * f + (double)step_prefill_tok_;
        tps_win_decode_  = tps_win_decode_  * f + (double)step_decode_tok_;
        tps_win_time_    = tps_win_time_    * f + wall;
        /* At least half a second of time under the rate. After an idle spell the window starts
         * over and a burst's first chunk commits as soon as it is issued, which would spike the
         * rate; with the floor the first reads are low instead, and the steady (~1 s) window is
         * unaffected. */
        const double t = std::max(tps_win_time_, 0.5);
        m_.prefill_tps = (float)(tps_win_prefill_ / t);
        m_.decode_tps  = (float)(tps_win_decode_  / t);
        /* The decay never reaches zero; below 0.05 tok/s it reads as zero. */
        if (m_.prefill_tps < 0.05f) m_.prefill_tps = 0.0f;
        if (m_.decode_tps  < 0.05f) m_.decode_tps  = 0.0f;
    }
    /* The step's cost at its depth, for the adaptive window (draft.h): the same wall time,
     * including the draft rounds after it, with a step issued ahead counted once. Decode steps
     * only. Keyed by the deepest window the step verified, which is the head depth it ran at
     * whenever D held from the step before. */
    if (draft_.adaptive() && plan_.phase == RAD_PHASE_DECODE && plan_.n_spec > 0 && wall > 0) {
        int rows = 0;
        for (const StepEntry& e : plan_.e) rows += (e.produces_token && !e.is_prefill) ? 1 : 0;
        draft_.step_time((int)plan_.n_spec, rows, wall * 1e3);
    }
    last_step_at_ = step_end;
    /* So a second call, or one on a loop iteration that ran no step, charges nothing. */
    step_start_ = {};
}

/* ------------------------------------------------------------------ the serial drafter */
/* Which sequences may take a serial draft after the step just run, and what each needs. Called
 * while plan() still describes that step, since everything here indexes into it: the head reads
 * the trunk's hidden states from the buffer the step filled.
 *
 * Head indexing (StepPlan::tok_shift): head index i reads hidden(i) and the token at i + 1,
 * writes K and V at position i, and predicts i + 2. After a step that committed `n_computed`
 * positions the newest index the head can serve is n_computed - 1; that is draft round 1, and
 * each later round is one position further. */
int Scheduler::serial_begin(int* rounds, SerialRow* out, int max) {
    std::lock_guard<std::mutex> lk(mu_);
    serial_.clear();
    serial_failed_ = false;   /* a new snapshot: the previous cycle's answer is spent */
    if (!rounds || *rounds <= 0 || !out || max <= 0) return 0;

    /* May anything draft? The controller sets depth from measured acceptance; 0 means drafting is
     * off. A step carrying a prefill chunk is no reason to refuse, since its decode rows still
     * take the recurrent path. Neither answer skips the history pass: the head's K and V must be
     * filled either way, e.g. for a sequence still in the middle of a long prompt. */
    const int depth = draft_.next_depth();
    bool may_draft = depth > 0;
    if (depth > 0 && *rounds > depth) *rounds = depth;
    if (*rounds > RAD_SCHED_MAX_SPEC) *rounds = RAD_SCHED_MAX_SPEC;

    /* Under the adaptive window, no more rounds than the deepest sequence wants. A round is one
     * head pass over every drafting row, and a sequence that wants fewer verifies the first of
     * them (expect_draft, set_draft); a round past the deepest window drafts for nobody. Only a
     * row that will draft counts. With none, the rounds stay as they are, because round 0 is the
     * history pass. The lookahead reservation below then matches the rounds that will run. */
    if (may_draft && (draft_.adaptive() || draft_.cost_policy())) {
        int want = 0;
        for (const StepEntry& e : plan_.e) {
            if (!e.produces_token || e.slot < 0 || (size_t)e.slot >= reqs_.size()) continue;
            SchedReq& s = *reqs_[(size_t)e.slot];
            if (!s.req || s.req->state != ReqState::Running || !s.running ||
                !s.req->may_speculate)
                continue;
            const int c = draft_cap(s);
            if (c > want) want = c;
        }
        if (want > 0 && *rounds > want) *rounds = want;
    }

    int n_draft_rows = 0;
    int64_t cu = 0;                     /* this entry's first token in the step just run */
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        const StepEntry& e = plan_.e[i];
        const int64_t base = cu;
        cu += e.n_tokens;

        if ((int)serial_.size() >= max) continue;
        if (e.slot < 0 || (size_t)e.slot >= reqs_.size()) continue;
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req || s.req->state != ReqState::Running || !s.running) continue;
        Request& r = *s.req;

        /* The verified positions are the entry's last 1 + n_spec. Round 1 continues from the last
         * committed one, which depends on acceptance the card knows and the host does not yet,
         * so the host bounds it by all of them. A mid-prompt chunk verified nothing. */
        const int64_t vstart = (int64_t)e.ctx_len + e.n_tokens - 1 - e.n_spec;
        const int64_t head_max = vstart + e.n_spec;
        bool can_draft = may_draft && e.produces_token;
        /* The rounds this row may draft: its own window (draft_cap) inside the cycle's. A window
         * of 0 is a decision -- a cost plan's K=0, or an adaptive stop -- so the row owes its
         * history (positions it already holds blocks for) and drafts nothing, which is the 0 -> 6
         * path without lost history; a shared head has no history to owe and is skipped below.
         * Without its own window every drafting row runs the cycle's deepest, and the bracket
         * times rounds no row's verify asks for. */
        const int32_t cap = can_draft ? draft_cap(s) : 0;
        if (can_draft && cap <= 0) can_draft = false;

        /* Lookahead KV: round k writes the head's K and V at head + k - 1, past the committed end,
         * so blocks must exist up to the furthest head acceptance could leave. The next verify
         * writes the trunk's K and V there, so it would reserve them anyway. A row refused
         * lookahead (or too near the context end) still owes its history, which writes positions
         * it already has blocks for; skipping it would leave them unwritten in the head's layer. */
        if (can_draft && ctx_limit_ > 0 && head_max + *rounds > ctx_limit_) can_draft = false;
        if (can_draft && kv_->ensure_from(r.id, head_max + *rounds, r.n_computed) < 0)
            can_draft = false;

        /* The history this step owes: first the positions whose next token the host holds (a
         * prefill chunk's prompt), from the watermark on, clamped to rows this step produced (a
         * prefix hit's adopted blocks have no hidden states here to redo them from); then all
         * 1 + n_spec verified positions, whose next tokens exist only on the card.
         *
         * The last row is draft round 1: the card fills rows past the last committed position
         * with that position again (advance.h, ADV_HIST), so the final row holds round 1's inputs
         * and a drafting row takes its proposal from it (RadBatch::draft_out_ids). The repeats
         * rewrite identical K and V and nothing past it is written, so no query can see a
         * rejected draft's state. */
        const int64_t owed_end = e.produces_token ? vstart : (int64_t)e.ctx_len + e.n_tokens;
        int64_t hs = s.draft_filled > e.ctx_len ? s.draft_filled : (int64_t)e.ctx_len;
        if (hs > owed_end) hs = owed_end;
        int64_t lead = owed_end - hs;
        const int64_t verified = e.produces_token ? e.n_spec + 1 : 0;
        int64_t hist_n = lead + verified;

        /* A head with no cache of its own owes no history (rad_drafter_shared_kv). Its one row is
         * round 1's at the committed position, and a non-drafting sequence runs nothing. */
        if (serial_shared_) {
            if (!can_draft) continue;
            hs = vstart;
            lead = 0;
            hist_n = 1;
        }

        if (!can_draft && hist_n <= 0) continue;   /* nothing owed and nothing to draft */

        SerialSeq m{};
        m.seq        = r.id;
        m.slot       = e.slot;
        m.card       = e.card;
        m.hist_row   = (int32_t)(base + (hs - e.ctx_len));
        m.hist_n     = (int32_t)hist_n;
        /* The shared head's one row sits where the card's acceptance puts it; the host stages
         * the furthest it can be, as for every round. */
        m.hist_pos   = serial_shared_ ? head_max : hs;
        m.dev        = e.produces_token;
        m.dev_tok    = (int32_t)(base + (vstart - e.ctx_len));
        m.dev_out    = e.dev_out;
        m.lead       = (int32_t)lead;
        m.base_pos   = head_max;
        m.draft_idx  = can_draft ? n_draft_rows : -1;
        /* The rounds this row actually runs: its own window, not the cycle's deepest. A shallower
         * row stops after its own depth; `deepest` below takes the snapshot's bound from these. */
        m.k = can_draft ? (cap < *rounds ? cap : *rounds) : 0;
        serial_.push_back(m);

        if (can_draft) ++n_draft_rows;
    }

    /* The rounds bound what the rows that survived this loop can fill. `*rounds` was taken from what
     * the rows wanted before it, but a row can lose its draft here: its plan says K0, its lookahead
     * crosses the context end, or the KV reservation was refused. A round past the deepest survivor
     * would have no drafting row and no history row, so serial_batch() would build nothing while
     * the driver still counted a pass. One round stays requested when no row drafts, because round
     * 0 is the history pass the remaining rows still owe. */
    int32_t deepest = 0;
    for (const SerialSeq& m : serial_)
        if (m.draft_idx >= 0 && m.k > deepest) deepest = m.k;
    *rounds = deepest > 0 ? deepest : 1;

    /* A round runs only the rows that still have a draft at its depth, and the driver freezes each
     * row's depth from the batches it saw. That needs the drafting rows to be a prefix for every
     * round, so the snapshot orders them by depth, deepest first, and among equal depths by their
     * distinct draft_idx, which is the plan's own order. History-only rows (draft_idx < 0) are
     * equivalent under this comparator: they run only in the history pass, and nothing may depend on
     * their order among themselves. Only this snapshot's own draft order changes; the issued plan
     * keeps its row order. */
    std::sort(serial_.begin(), serial_.end(), [](const SerialSeq& a, const SerialSeq& b) {
        const int32_t ka = a.draft_idx >= 0 ? a.k : -1;
        const int32_t kb = b.draft_idx >= 0 ? b.k : -1;
        if (ka != kb) return ka > kb;
        return a.draft_idx < b.draft_idx;
    });
    int n = 0;
    for (SerialSeq& m : serial_) {
        if (m.draft_idx < 0) continue;
        m.draft_idx = n;
        out[n].seq  = m.seq;
        out[n].slot = m.slot;
        out[n].card = m.card;
        ++n;
    }
    return n;
}

const RadBatch* Scheduler::serial_batch(int round, int step) {
    /* RADIANCE_DEBUG_STAGE, below. Read before the lock so the timer covers the wait for it. */
    static const bool stg = std::getenv("RADIANCE_DEBUG_STAGE") != nullptr;
    const auto t_stage0 = stg ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    std::lock_guard<std::mutex> lk(mu_);
    serial_failed_ = false;   /* this call's answer, not the snapshot's */
    if (round < 0 || serial_.empty()) return nullptr;
    /* Round 1 has no pass of its own: the history pass carries it (RadBatch::draft_out_ids). */
    if (round == 1) return nullptr;

    serial_plan_.reset();
    serial_plan_.phase      = RAD_PHASE_DECODE;
    /* Negative is the history pass; positive is draft round r. The sign is what the plugin reads
     * to decide whether the pass ends in the lm_head (rad_runtime.h). */
    serial_plan_.draft_pass = round == 0 ? -1 : round;
    serial_plan_.tok_shift  = 1;           /* the head's own indexing; see StepPlan::tok_shift */
    serial_plan_.e.reserve(serial_.size());

    /* Drafting sequences first, in draft order (deepest first, see serial_begin), then
     * history-only ones. Every round's rows are the drafting sequences that have a draft left at
     * that depth, so each round is a prefix of the one before, and at depth 1 the history pass is
     * the group's last pass, so its first rows are where the advance that takes the last proposal
     * (ADV_TAKE) reads them. */
    for (int part = 0; part < 2; ++part)
    for (size_t i = 0; i < serial_.size(); ++i) {
        const SerialSeq& m = serial_[i];
        if ((m.draft_idx >= 0) != (part == 0)) continue;
        if (round == 0 && m.hist_n <= 0) continue;      /* nothing owed for this sequence */
        if (round > 0 && (m.draft_idx < 0 || m.k < round))
            continue;      /* history only, or this row's depth ended before this round */
        const Request* rq = serial_req(m);
        /* An invariant, not a case: reap() defers on block_holds() while a snapshot names the row.
         * If it is ever broken the round cannot be built, and saying so beats a null the driver
         * would have to read as a round with nothing to do. */
        if (!rq) {
            RAD_ERR("draft: serial row %llu lost its request inside its own snapshot",
                    (unsigned long long)m.seq);
            serial_failed_ = true;
            return nullptr;
        }

        StepEntry e;
        e.req      = rq;
        e.seq      = m.seq;
        e.slot     = m.slot;
        e.card     = m.card;
        e.n_tokens = round == 0 ? m.hist_n : 1;
        /* A round's position is the card's; the host stages the furthest it can be, which every
         * host-side ceiling reads. History positions are exact. A head with no cache of its own
         * does not step forward, so neither does its ceiling. */
        e.ctx_len  = (int32_t)(round == 0 ? m.hist_pos
                                          : m.base_pos + (serial_shared_ ? 0 : round - 1));
        e.n_spec   = 0;
        /* One-based and constant: a head pass commits exactly the tokens it was given. The head
         * has no linear-state op to read it; it is set to a value the kernels accept. */
        e.n_accepted  = 1;
        e.conv_cursor = 0;
        /* No draft array: a history row's tokens past its lead, and every round's token, come
         * from the card (advance.h); token_at() stages 0 for them, and the positions before the
         * lead are the prompt, staged exactly. */
        e.draft   = nullptr;
        e.n_draft = 0;
        e.is_prefill     = false;
        e.produces_token = round > 0;
        /* The history pass gathers the trunk's hidden states, a whole run of them. Later rounds
         * read the head's own output, which it left in the same buffer. */
        e.hidden_row = round == 0 ? m.hist_row : -1;
        e.draft_out  = round == 0 && m.draft_idx >= 0;
        e.dev      = m.dev;
        e.dev_tok  = m.dev_tok;
        e.dev_out  = m.dev_out;
        e.dev_aux  = round == 0 ? m.lead : 0;
        serial_plan_.e.push_back(e);
        serial_plan_.n_tok += e.n_tokens;
    }
    /* Nothing owed: no drafting row reaches this round, and round 0 had no history row either.
     * Not a failure -- serial_begin() never asked for this round in the first place. */
    if (serial_plan_.e.empty()) return nullptr;

    /* RADIANCE_DEBUG_STAGE: splits a draft pass's host staging into the plan loop above and
     * `builder_.build`, reporting the average of each. Both are small next to issuing the pass
     * (engine_draft.cpp has that split). Off by default; two clock reads when on. */
    if (!stg) {
        const RadBatch* b = builder_.build(serial_plan_, *kv_, step);
        serial_failed_ = b == nullptr;
        return b;
    }

    static double acc_plan = 0.0, acc_build = 0.0;
    static long   n_calls = 0;
    const auto t1 = std::chrono::steady_clock::now();
    const RadBatch* b = builder_.build(serial_plan_, *kv_, step);
    const auto t2 = std::chrono::steady_clock::now();
    serial_failed_ = b == nullptr;
    acc_plan  += std::chrono::duration<double, std::micro>(t1 - t_stage0).count();
    acc_build += std::chrono::duration<double, std::micro>(t2 - t1).count();
    if (++n_calls % 400 == 0)
        RAD_INFO("stage: %ld draft batches; plan %.1f us, build %.1f us a call",
                 n_calls, acc_plan / (double)n_calls, acc_build / (double)n_calls);
    return b;
}

/* The last serial_batch() owed a pass and built none. Only serial_batch() sets it, so a reader
 * that calls the two together sees whether a null return was work that could not be built or work
 * that was never owed; see the declaration for why a row count cannot answer this. */
bool Scheduler::serial_batch_failed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return serial_failed_;
}

/* The watermark moves to what actually ran, read off the committed sequence: the history pass
 * wrote the head's K and V at n_computed - 1 from the trunk's hidden state and the committed
 * token, so the history is exact up to n_computed. Rounds past it write speculatively and are not
 * counted; redoing them next step costs one head pass over a few tokens. A history-only row is
 * exact up to where its history reached: n_computed if it covered the verified positions, its
 * host-known end otherwise.
 *
 * `history_ran` is the driver's completion signal for round 0, not this scheduler's view of it: a
 * built batch is not an executed pass (the engine may reject the cycle, or the host pass may have
 * failed), and only the caller sees which happened. A caller that cannot say passes false, which
 * costs a repeated history pass and never a wrong watermark. */
void Scheduler::serial_end(int rounds_done, bool history_ran) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const SerialSeq& m : serial_) {
        if (m.slot < 0 || (size_t)m.slot >= reqs_.size()) continue;
        SchedReq& s = *reqs_[(size_t)m.slot];
        if (!s.req || s.req->id != m.seq) continue;
        const int64_t filled = m.dev ? s.req->n_computed
                                     : m.hist_pos + m.hist_n;
        /* Every row's history comes from round 0, so nothing counts unless that pass ran.
         * A drafting row additionally needs a draft round to have run: without one its verified
         * positions were never written. A K0 row owes only the history pass, and a history-only
         * row owes nothing else. */
        const bool counts = history_ran && (m.dev ? (rounds_done >= 1 || m.draft_idx < 0) : true);
        if (counts && filled > s.draft_filled) s.draft_filled = filled;
    }
    /* The snapshot ends here: every pass naming these rows is built, and keeping them would keep
     * deferring the release of a request cancelled or reaped while the rounds ran. */
    serial_.clear();
}

int64_t Scheduler::draft_watermark(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& s : reqs_)
        if (s->req && s->req->id == id) return s->draft_filled;
    return -1;
}

/* ------------------------------------------------------------------- the block drafter */
/* Which sequences the drafter runs over after the step just committed, and which of them may
 * also propose. Called while plan() still describes that step, because the context pass mirrors
 * it exactly, rejected suffix included: the drafter reads hidden states positionally from its
 * own buffer, so batch row i must be buffer row i. The rejected positions lie in [n_computed,
 * n_computed + block), which the query pass overwrites before anything reads them, and the
 * ensure() below covers them since n_computed + block reaches the trunk step's own end.
 *
 * On the card the step has not committed yet. The step began at n_computed (a row issued ahead
 * was staged at its furthest reach; it really continued from the committed end, as commit()
 * reads it), and the anchor lands somewhere in [that + n_tokens - n_spec, that + n_tokens]
 * depending on acceptance: the gates take the end that binds them, the query rows the furthest. */
int Scheduler::block_begin(int block, int depth, int window, int32_t mask, BlockRow* out, int max,
                           bool on_card) {
    std::lock_guard<std::mutex> lk(mu_);
    block_.clear();
    block_len_ = 0;
    block_depth_ = 0;
    if (block < 2 || depth <= 0 || depth > block || !out || max <= 0) return 0;
    block_len_   = (int32_t)block;
    block_depth_ = (int32_t)depth;
    block_mask_.assign((size_t)(block - 1), mask);

    /* Not the drafter's declared `depth`: whether the draft policy wants any tokens this step
     * (0 when it has backed off). */
    const bool may_draft = draft_.next_depth() > 0;
    /* The query pass is `block` tokens per drafting sequence, and the builder's staging arrays
     * are sized at max_tok. Unlike a decode step it can put more rows than sequences in a batch,
     * which on a small token budget would overrun h_tok. Under --prefill-span a step of several
     * sequences is bounded by the shared budget instead (RadBuildCtx::max_mixed_tok), which is
     * max_tok without a span. */
    const int64_t mixed = geo_.shared_tok > 0 ? geo_.shared_tok : cfg_.max_tok;
    if ((int64_t)max * block > mixed) max = (int)(mixed / block);
    if (max <= 0) return 0;

    int n_draft_rows = 0;
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        const StepEntry& e = plan_.e[i];

        BlockSeq d{};
        d.seq      = e.seq;
        d.slot     = e.slot;
        d.card     = e.card;
        d.ctx_len  = e.ctx_len;
        d.n_tokens = e.n_tokens;

        /* Every entry of the trunk step gets a row, in order, running or not: the context pass's
         * row i must be the trunk's row i, so leaving out a sequence the commit just finished
         * would shift every later one onto its neighbour's hidden states. It stays as a
         * placeholder (BlockSeq::mirror). */
        SchedReq* sp = (e.slot >= 0 && (size_t)e.slot < reqs_.size()) ? reqs_[(size_t)e.slot].get()
                                                                        : nullptr;
        if (!sp || !sp->req || sp->req->id != e.seq || sp->req->state != ReqState::Running ||
            !sp->running) {
            d.mirror = true;
            block_.push_back(d);
            continue;
        }
        SchedReq& s = *sp;
        Request& r = *s.req;
        if (on_card && e.dev) d.ctx_len = (int32_t)r.n_computed;
        /* The anchor's position, the sequence's committed end: lowest and highest it can be. */
        const int64_t next_lo = on_card ? (int64_t)d.ctx_len + e.n_tokens - e.n_spec : r.n_computed;
        const int64_t next_hi = on_card ? (int64_t)d.ctx_len + e.n_tokens : r.n_computed;
        d.base_pos = next_hi;
        d.dev      = on_card;

        /* May it propose? It needs an anchor token (the one the sampler just chose, so the entry
         * produced one) and a draft cache covering what the block's attention reads. The high end
         * is this step's context pass, about to cover everything up to the trunk's end. The low
         * end is draft_lo, non-zero only after a prefix-cache hit that left a hole. On the card
         * the anchor is the card's, and a request that may not speculate is left out here rather
         * than at set_draft, since the step issued ahead verifies what this pass proposes. */
        const bool have_tok = on_card ? e.produces_token && r.may_speculate
                                      : e.produces_token && !r.output.empty() && r.n_computed >= 1;
        /* Where this sequence's draft cache starts once the context pass has run: its own floor,
         * or for a sequence that never had one, this step's first row (0 for an ordinary prefill,
         * the hit length after a prefix-cache hit). The binding query row is the lowest, at
         * n_computed, and it reads back to n_computed - window + 1. */
        const int64_t lo = draft_floor(s, (int64_t)d.ctx_len);
        const bool reach = lo <= 0 ? true
                                   : (window > 0 && next_lo - window + 1 >= lo);

        /* RADIANCE_DEBUG_DRAFT counts which gate declined. A drafter that silently stops
         * proposing looks like collapsed acceptance downstream (accept% is conditional on
         * drafting), and no other counter tells the five gates apart. */
        const bool ok_room = n_draft_rows < max;
        /* The query block writes positions n_computed .. n_computed + block - 1, and none of them
         * may be at or past the end of the context. */
        const bool ok_ctx  = ctx_limit_ <= 0 || next_hi + block <= ctx_limit_;
        const bool ok_kv   = ok_room && ok_ctx &&
                             kv_->ensure_from(r.id, next_hi + block, r.n_computed) >= 0;
        static const bool dbg = [] { const char* v = std::getenv("RADIANCE_DEBUG_DRAFT");
                                     return v && *v && *v != '0'; }();
        if (dbg) {
            static int64_t n_call = 0, n_depth = 0, n_tok = 0, n_reach = 0, n_room = 0, n_kv = 0, n_ok = 0;
            ++n_call;
            if (!may_draft)      ++n_depth;
            else if (!have_tok)  ++n_tok;
            else if (!reach)     ++n_reach;
            else if (!ok_room)   ++n_room;
            else if (!ok_kv)     ++n_kv;
            else                 ++n_ok;
            if ((n_call % 256) == 0)
                RAD_INFO("draft gates over %lld asks: ok %lld | declined depth %lld tok %lld "
                         "reach %lld room %lld kv %lld  (window %d, floor %lld, ctx_len %lld)",
                         (long long)n_call, (long long)n_ok, (long long)n_depth, (long long)n_tok,
                         (long long)n_reach, (long long)n_room, (long long)n_kv,
                         window, (long long)lo, (long long)e.ctx_len);
        }
        if (may_draft && have_tok && reach && ok_kv) {
            d.can_draft = true;
            out[n_draft_rows].seq  = d.seq;
            out[n_draft_rows].slot = d.slot;
            out[n_draft_rows].card = d.card;
            ++n_draft_rows;
        }
        block_.push_back(d);
    }
    return n_draft_rows;
}

const Request* Scheduler::block_req(const BlockSeq& d) const { return live_req(d.slot, d.seq); }
const Request* Scheduler::serial_req(const SerialSeq& m) const { return live_req(m.slot, m.seq); }

const RadBatch* Scheduler::block_context(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_.empty()) return nullptr;

    block_plan_.reset();
    block_plan_.phase      = RAD_PHASE_DECODE;
    block_plan_.draft_pass = -1;           /* the sign is what the plugin reads (rad_runtime.h) */
    block_plan_.e.reserve(block_.size());

    for (const BlockSeq& d : block_) {
        const Request* rq = block_req(d);
        /* Unreachable for a live row while reap() defers on block_holds(); if it happens the pass
         * is skipped, costing one step's draft. A placeholder row may have lost its Request
         * before block_begin (the server reaps a finished request as soon as its sink is done)
         * and needs none: this pass reads no token ids and the builder stages it as padding. */
        if (!rq && !d.mirror) return nullptr;
        StepEntry e;
        e.req        = rq;
        e.seq        = d.seq;
        e.slot       = d.slot;
        e.card       = d.card;
        e.n_tokens   = d.n_tokens;
        e.ctx_len    = d.ctx_len;
        e.n_spec     = 0;
        e.n_accepted = 1;               /* one-based; nothing in this pass reads it */
        /* The context pass reads tapped hidden states, not token ids: nothing to shift and no
         * draft. */
        e.produces_token = false;
        block_plan_.e.push_back(e);
        block_plan_.n_tok += e.n_tokens;
    }
    if (block_plan_.e.empty()) return nullptr;
    return builder_.build(block_plan_, *kv_, step);
}

/* One entry per drafting sequence, `block` tokens: the anchor at the sequence's next position,
 * then `block - 1` masks. `out_ids` names the last `depth` rows.
 *
 * At block = depth + 1 those are the mask rows: the anchor predicts nothing and each mask row
 * predicts the token at its own position (vLLM's _prepare_dflash_inputs_kernel: is_sample =
 * query_off >= 1, sample_pos = query_pos). At block = depth it is every row, anchor included,
 * each predicting the token after it: the ordinary next-token relation that a drafter whose
 * reference reads `hidden[:, -block_size:, :]` was trained in. rad_builder.h explains why the
 * plugin states which. */
const RadBatch* Scheduler::block_query(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_.empty() || block_len_ < 2 || block_depth_ <= 0) return nullptr;
    const int32_t block = block_len_, steps = block - 1, depth = block_depth_;

    block_plan_.reset();
    block_plan_.phase      = RAD_PHASE_DECODE;
    block_plan_.draft_pass = 1;
    block_plan_.tok_shift  = 0;            /* the anchor sits at its own row; see StepPlan */
    block_plan_.e.reserve(block_.size());

    int32_t row = 0;
    for (const BlockSeq& d : block_) {
        if (!d.can_draft) continue;
        const Request* rq = block_req(d);
        if (!rq) return nullptr;   /* see block_context */
        StepEntry e;
        e.req        = rq;
        e.seq        = d.seq;
        e.slot       = d.slot;
        e.card       = d.card;
        e.n_tokens   = block;
        e.ctx_len    = (int32_t)d.base_pos;
        e.n_spec     = 0;
        e.n_accepted = 1;
        /* token_at() walks prompt ++ output ++ draft: index base_pos is the token the sampler just
         * chose (the anchor) and everything past it falls into the mask array. */
        e.draft      = block_mask_.data();
        e.n_draft    = steps;
        e.produces_token = false;
        /* The last `depth` rows of the block: at block = depth + 1 this skips the anchor; at
         * block = depth it starts on it. */
        e.hidden_row = row + block - depth;
        e.hidden_n   = depth;
        /* On the card the anchor, its position and every mask come from the card (ADV_BLOCK),
         * which reads the mask id here; the host staged them at the furthest position. */
        if (d.dev) {
            e.dev     = true;
            e.dev_out = -1;
            e.dev_aux = block_mask_.empty() ? 0 : block_mask_[0];
        }
        block_plan_.e.push_back(e);
        block_plan_.n_tok += block;
        row += block;
    }
    if (block_plan_.e.empty()) return nullptr;
    return builder_.build(block_plan_, *kv_, step);
}

/* The filled range moves over what the context pass wrote. The query pass's K and V are
 * speculative and not counted; the next step's context pass rewrites them from the trunk's real
 * hidden states. */
void Scheduler::block_end() {
    std::lock_guard<std::mutex> lk(mu_);
    for (const BlockSeq& d : block_) {
        if (d.mirror) continue;                /* finished; it has no draft cache to track */
        if (d.slot < 0 || (size_t)d.slot >= reqs_.size() || !reqs_[(size_t)d.slot]) continue;
        SchedReq& s = *reqs_[(size_t)d.slot];
        if (!s.req || s.req->id != d.seq) continue;
        if (s.draft_hi == 0 && s.draft_lo == 0) s.draft_lo = draft_floor(s, d.ctx_len);
        const int64_t hi = (int64_t)d.ctx_len + d.n_tokens;
        if (hi > s.draft_hi) s.draft_hi = hi;
    }
    /* The snapshot ends here: both passes are built, and keeping the rows would keep deferring
     * the release of a request cancelled or reaped while the passes ran. */
    block_.clear();
}

int32_t Scheduler::conv_window() const {
    int32_t W = 1;
    for (const KVGroupInfo& g : prog_->kv_groups)
        if (g.decl.kind == RAD_KV_CONV && g.decl.conv_width > 0)
            W = (int32_t)conv_window_entries(g.decl, cfg_.n_spec);
    return W;
}

/* ------------------------------------------------------------------ metrics */
SchedMetrics Scheduler::metrics() const {
    std::lock_guard<std::mutex> lk(mu_);
    return metrics_locked();
}

void Scheduler::publish(bool idle) {
    const int64_t now = mono_us();
    if (idle && now - pub_at_us_ < 20000) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        scratch_m_ = metrics_locked();
        scratch_reqs_.resize(reqs_.size());
        scratch_reqs_.resize((size_t)requests_locked(scratch_reqs_.data(), (int)scratch_reqs_.size()));
    }
    if (!pub_mu_.try_lock()) return;
    pub_m_ = scratch_m_;
    pub_reqs_.swap(scratch_reqs_);
    pub_mu_.unlock();
    pub_at_us_ = now;
}

SchedMetrics Scheduler::observed_metrics() const {
    std::lock_guard<std::mutex> lk(pub_mu_);
    return pub_m_;
}

int Scheduler::observed_requests(ReqStat* out, int max) const {
    std::lock_guard<std::mutex> lk(pub_mu_);
    const int n = std::min(max, (int)pub_reqs_.size());
    std::copy(pub_reqs_.begin(), pub_reqs_.begin() + n, out);
    return n;
}

SchedMetrics Scheduler::metrics_locked() const {
    SchedMetrics m = m_;
    m.step = step_;
    m.running = (int64_t)running_.size();
    m.queue_depth = (int64_t)waiting_.size();
    m.preempted_waiting = 0;
    for (int32_t slot : waiting_)
        if (reqs_[(size_t)slot]->req->state == ReqState::Preempted) ++m.preempted_waiting;

    /* An idle engine runs no steps, so the window would hold the last rate forever. Decay it on
     * the way out (to 37% after half a second idle) without touching the stored window. Not while
     * a step is open: a long prefill chunk is not idleness. */
    const bool   open = step_start_.time_since_epoch().count() != 0;
    const double idle = open ? 0.0
        : std::chrono::duration<double>(std::chrono::steady_clock::now() - last_step_at_).count();
    if (idle > 0.05) {
        const float f = (float)std::exp(-idle / 0.5);
        m.prefill_tps *= f;
        m.decode_tps  *= f;
        if (m.prefill_tps < 0.05f) m.prefill_tps = 0.0f;
        if (m.decode_tps  < 0.05f) m.decode_tps  = 0.0f;
    }

    const std::vector<KVGroupInfo>& gs = prog_->kv_groups;
    m.n_kv_groups = (int)gs.size();
    if (m.n_kv_groups > RAD_SCHED_MAX_KV_GROUPS) m.n_kv_groups = RAD_SCHED_MAX_KV_GROUPS;
    for (int i = 0; i < m.n_kv_groups; ++i) {
        /* Against the carve and what is held of it, not the live mark. An elastic pool grows
         * back on demand, so a released block is neither occupied nor unavailable; counting it
         * would make an idle server read nearly full and turn the pool's own arbitration into
         * apparent workload pressure for anything watching kv_util. */
        const int64_t tot = kv_->total_blocks(i);
        const int64_t fr  = tot - kv_->held_blocks(i);
        m.kv[i].name = gs[(size_t)i].name.c_str();
        m.kv[i].blocks_total = tot;
        m.kv[i].blocks_free = fr;
        m.kv[i].utilisation = tot > 0 ? (float)(1.0 - (double)fr / (double)tot) : 0.0f;
        m.kv[i].block_tokens = kv_->block_tokens(i);
        m.kv[i].window       = gs[(size_t)i].decl.kind == RAD_KV_WINDOW;
        m.kv[i].bytes_total  = kv_->pool_bytes(i);
        m.kv[i].bytes_used   = kv_->pool_bytes_used(i);
        m.kv[i].bytes_carved = kv_->pool_bytes_carved(i);
    }

    if (pc_) { m.prefix_evictions = pc_->evictions();
               m.prefix_ckpt_evictions = pc_->ckpt_evictions();
               const IPrefixCache::CkptCounts cc = pc_->ckpt_counts();
               m.prefix_ckpt_retired = cc.retired;
               m.prefix_ckpt_lru = cc.lru;
               m.prefix_ckpt_refused = cc.refused;
               m.prefix_expired = pc_->expired();
               m.prefix_ckpt_expired = pc_->ckpt_expired();
               m.prefix_held_tokens = pc_->held_tokens(); }
    m.attn_hit_rate = m.prompt_tokens
        ? (float)((double)m.attn_cached_tokens / (double)m.prompt_tokens) : 0.0f;
    m.linear_hit_rate = m.prompt_tokens
        ? (float)((double)m.linear_cached_tokens / (double)m.prompt_tokens) : 0.0f;
    m.draft_acceptance = draft_.acceptance();
    m.draft_depth = draft_.next_depth();
    /* The loop bound is the array's extent, not RAD_SCHED_MAX_SPEC: the values are equal, but gcc
     * cannot prove the boundary index safe through the inlined accessors (which guard with
     * `j < RAD_SCHED_MAX_SPEC ? reach_[j] : 0`) and warns. */
    const int kPos = (int)(sizeof m.draft_pos_reached / sizeof m.draft_pos_reached[0]);
    m.draft_depth_seen = draft_.depth_seen();
    if (m.draft_depth_seen > kPos) m.draft_depth_seen = kPos;
    for (int j = 0; j < m.draft_depth_seen && j < kPos; ++j) {
        m.draft_pos_reached [j] = draft_.reached(j);
        m.draft_pos_accepted[j] = draft_.hits(j);
    }
    if (draft_.adaptive()) {
        m.draft_adaptive = true;
        /* Over the running sequences whose estimate has started; one that has not drafted yet
         * has no window to report. */
        int64_t sum = 0, n = 0;
        for (int32_t slot : running_) {
            const SchedReq& s = *reqs_[(size_t)slot];
            if (s.adapt.depth <= 0) continue;
            /* The window it drafts into: its stop under the step's head depth. */
            sum += draft_.window_of(s.adapt);
            ++n;
        }
        m.draft_depth_adaptive = n ? (float)((double)sum / (double)n) : 0.0f;
        const int kDepth = (int)(sizeof m.draft_verify_depth / sizeof m.draft_verify_depth[0]);
        for (int d = 0; d < kDepth; ++d) m.draft_verify_depth[d] = draft_.verified_at(d);
    }
    /* The delivery ledger: what the predictor was handed, what it applied, what it skipped, and
     * what never arrived. A predictor call that returns OK has not necessarily taught anything. */
    const DraftPredictModel::Counts& dc = draft_predict_.counts();
    m.draft_obs_attempted = (int64_t)dc.rows_attempted;
    m.draft_obs_applied   = (int64_t)dc.rows_applied;
    m.draft_obs_skipped   = (int64_t)dc.rows_skipped;
    m.draft_obs_discarded = draft_cycles_dropped_;
    m.draft_obs_inactive  = draft_costs_.inactive();
    /* The reference's own count: a no-loan table absorbs nothing, so a cycle reaching it leaves
     * this number and nothing else. */
    m.draft_ref_skipped   = draft_costs_.reference_skipped();
    return m;
}

}  /* namespace rad */
