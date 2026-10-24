/* draft_policy.cpp -- the cost-based draft-depth policy. See draft_policy.h.
 *
 * The objective is the issue's: expected output tokens of the whole batch over its expected
 * batch time -- the time of the stationary pass that verifies those tokens (the horizon score,
 * draft_policy.h), not of this cycle's frozen-window pass. A row that does not draft still emits
 * its one token, so it adds 1 to the yield and costs a verify row at K 0.
 */
#include "draft_policy.h"

#include <cmath>
#include <cstdint>

namespace rad {

namespace {

/* The dwell a depth change is priced over: `hold`, and at least two cycles -- the move's own cycle
 * plus the one that first earns the new depth's yield. A shorter dwell has no steady cycle to gain
 * in, so it would refuse every move. One definition, so the configure() log names the dwell the rule
 * really uses. */
double dwell_of(int32_t hold) { return hold > 2 ? (double)hold : 2.0; }

/* A request's own cap: no draft past its remaining output (the verified token and the bonus token
 * take one each), no draft past the declared ceiling, none at all when it may not draft. */
int32_t cap_of(const DraftSeqView& v, int32_t ceiling) {
    if (v.eligibility != DraftEligibility::Eligible) return 0;
    const int32_t cap = draft_budget_k(v.remaining, ceiling);
    return cap > RAD_SCHED_MAX_SPEC ? RAD_SCHED_MAX_SPEC : cap;
}

/* The one resource cut, shared by the cost search and the calibration sequence: a positive token or
 * KV budget caps the summed drafts, and scaling every K down proportionally is the cut that may
 * happen at once. Only the assignment is shared -- the Cost path's repricing of the reduced plan
 * stays in choose(), where the hysteresis needs the price of the plan that will actually be
 * issued. Returns whether anything was cut; a cut row is marked DraftFallback::Resources. */
bool apply_draft_budget(const DraftBatchView& view, int32_t* k, DraftFallback* why) {
    int32_t budget = -1;
    if (view.token_budget > 0) budget = view.token_budget;
    if (view.kv_budget > 0 && (budget < 0 || view.kv_budget < budget)) budget = view.kv_budget;
    if (budget < 0) return false;
    int32_t sum = 0;
    for (int32_t i = 0; i < view.n_seq; ++i) sum += k[i];
    if (sum <= budget || sum <= 0) return false;
    bool cut = false;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        if (k[i] <= 0) continue;
        const int32_t scaled = (int32_t)((int64_t)k[i] * budget / sum);
        if (scaled < k[i]) {
            k[i] = scaled;
            why[i] = DraftFallback::Resources;
            cut = true;
        }
    }
    return cut;
}

}  /* namespace */

int DraftPlanner::configure(const DraftPolicyConfig& c) {
    if (c.k_ceiling < 0 || c.k_ceiling > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft policy: ceiling %d is outside 0..%d", (int)c.k_ceiling,
                (int)RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    /* The tier list is the cost choice's input; a legacy plan never reads it, so a deployment
     * whose window is shallower than the default tiers is not refused for them. */
    if (c.mode == DraftPolicyMode::Cost) {
        if (c.n_tiers < 1 || c.n_tiers > RAD_DRAFT_MAX_TIERS) {
            RAD_ERR("draft policy: %d tiers, outside 1..%d", (int)c.n_tiers,
                    (int)RAD_DRAFT_MAX_TIERS);
            return RAD_E_INVAL;
        }
        for (int32_t i = 0; i < c.n_tiers; ++i) {
            const int32_t t = c.tiers[i];
            if (t < 0 || t > c.k_ceiling) {
                RAD_ERR("draft policy: tier %d is outside 0..%d", (int)t, (int)c.k_ceiling);
                return RAD_E_INVAL;
            }
            if (i > 0 && t <= c.tiers[i - 1]) {
                RAD_ERR("draft policy: tiers must be ascending and unique (tier %d is %d after "
                        "%d)", (int)i, (int)t, (int)c.tiers[i - 1]);
                return RAD_E_INVAL;
            }
        }
        if (c.fallback_k < 0 || c.fallback_k > c.k_ceiling) return RAD_E_INVAL;
        /* The calibration sequence runs only under Cost, and every value must be one of the
         * effective tiers: those are inside 0..ceiling already, and a depth the serving ladder
         * never offers must not be measured into a profile (the coverage invariant). Duplicates
         * and order are the sequence's, so they are not checked. */
        if (c.n_calib_k > 0) {
            if (c.n_calib_k < 2 || c.n_calib_k > RAD_DRAFT_MAX_TIERS) {
                RAD_ERR("draft policy: %d calibration depths, outside 2..%d", (int)c.n_calib_k,
                        (int)RAD_DRAFT_MAX_TIERS);
                return RAD_E_INVAL;
            }
            for (int32_t i = 0; i < c.n_calib_k; ++i) {
                const int32_t k = c.calib_k[i];
                bool offered = false;
                for (int32_t t = 0; t < c.n_tiers; ++t)
                    if (c.tiers[t] == k) { offered = true; break; }
                if (!offered) {
                    RAD_ERR("draft policy: calibration K%d is not one of the offered tiers "
                            "(%d..%d, %d values); a calibration measures only depths the serving "
                            "ladder offers", (int)k, (int)c.tiers[0],
                            (int)c.tiers[c.n_tiers - 1], (int)c.n_tiers);
                    return RAD_E_INVAL;
                }
            }
        }
    } else if (c.n_calib_k > 0) {
        RAD_ERR("draft policy: a calibration sequence needs --draft-policy cost");
        return RAD_E_INVAL;
    }
    if (c.passes < 0 || c.passes > 8 || c.max_evals < 1) return RAD_E_INVAL;
    if (!std::isfinite(c.hyst) || c.hyst < 0.0 || c.hold < 0) return RAD_E_INVAL;
    c_ = c;
    reset();
    /* Reserve the frozen window here, so no cycle allocates one. */
    if (c_.max_seqs > 0 && fr_id_.size() < (size_t)c_.max_seqs) {
        fr_id_.assign((size_t)c_.max_seqs, 0);
        fr_slot_.assign((size_t)c_.max_seqs, -1);
        fr_gen_.assign((size_t)c_.max_seqs, 0);
        fr_k_.assign((size_t)c_.max_seqs, 0);
    }
    if (c_.hold_reductions && c_.max_seqs > 0 && held_k_.size() < (size_t)c_.max_seqs)
        held_k_.assign((size_t)c_.max_seqs, 0);
    if (c_.mode == DraftPolicyMode::Cost)
        RAD_INFO("draft policy: cost, ceiling %d, tiers %d..%d (%d), baseline K%d, a depth change is "
                 "priced against the one cycle it costs over a %g-cycle dwell (%.0f%% net margin)%s",
                 (int)c_.k_ceiling, (int)c_.tiers[0], (int)c_.tiers[c_.n_tiers - 1],
                 (int)c_.n_tiers, (int)c_.fallback_k, dwell_of(c_.hold),
                 100.0 * c_.hyst, c_.hold_reductions ? ", and so is a reduction" : "");
    return RAD_OK;
}

void DraftPlanner::reset() {
    last_k_ = 0;
    chosen_ = false;
    gap_reported_ = false;
    gap_refused_  = false;
    calib_pos_ = 0;      /* the sequence starts over; configure() starts it here too */
    last_tiers_priced_ = 0;
    last_n_tier_price_ = 0;
    last_price_known_  = false;
    last_price_ = DraftTierPrice{};
    fr_n_   = 0;   /* the storage stays: configure() reserved it, or the first cycle grew it */
}

int32_t DraftPlanner::verify_k_of(const DraftBatchView& view, int32_t i, int32_t ki) const {
    if (view.pipeline_depth <= 0) return ki;
    const int32_t frozen = frozen_k_of(view, i);
    /* Only a step() cycle is built after this plan opened, so only there does Scheduler::draft_cap
     * re-cap the frozen window to the plan's own K. A step issued ahead was built under the plan
     * that published the drafts and verifies the frozen window as declared. */
    return view.direct_step && frozen > ki ? ki : frozen;
}

int32_t DraftPlanner::frozen_k_of(const DraftBatchView& view, int32_t i) const {
    const DraftSeqView& v = view.seq[i];
    for (size_t j = 0; j < fr_n_; ++j) {
        if (fr_id_[j] != v.req_id || fr_slot_[j] != v.slot || fr_gen_[j] != v.slot_gen) continue;
        return fr_k_[j];
    }
    return 0;
}

void DraftPlanner::shape_of(const DraftBatchView& view, const int32_t* k, int32_t clamp_to,
                            DraftVerifyShape* s, bool horizon) const {
    *s = DraftVerifyShape{};
    s->n_rows = view.n_seq;
    s->prefill_tokens = view.prefill_tokens;
    /* The loaded drafter, not who spends k[]: a head that is loaded keeps the history and KV of
     * a K0 row, so its K0 is a different bucket from a process with no head at all. */
    s->drafter = view.k_ceiling > 0 ? DraftDrafter::Serial : DraftDrafter::None;
    /* The view names the cycle's replay state only when the runtime evidenced it. Without that
     * evidence the dimension is missing, not a `None`: the field stays at its default and the flag
     * says not to read it, so draft_shape_priceable() refuses the shape instead of a model keying
     * it as measured. `DraftReplay::None` here would be the same claim with the evidence missing. */
    s->replay = view.replay;
    if (view.replay == DraftReplay::Unknown)
        s->unknown |= (uint32_t)kDraftShapeReplayUnknown;
    s->pipeline_depth = view.pipeline_depth;
    bool sampled = false;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        /* The draft side is what this cycle asks its rounds to run. */
        const int32_t want = k[i];
        const int32_t ki = clamp_to >= 0 && want > clamp_to ? clamp_to : want;
        if (ki > s->draft_k) s->draft_k = ki;
        for (int32_t j = 1; j <= ki; ++j) ++s->draft_rows[j];
        /* At pipeline_depth 0 the paired step verifies exactly what this cycle drafts; only when
         * the plan is issued ahead (publish-then-verify) does it verify the frozen previous
         * plan's window instead. The key names the window the paired step executes: the declared
         * frozen window on the ahead path, that window re-capped per row to this candidate's own
         * K on the direct path (Scheduler::draft_cap). The horizon shape verifies the row's own K:
         * the pass that will verify what this candidate drafts, if the next cycle keeps it. */
        const int32_t fk = horizon ? ki : verify_k_of(view, i, ki);
        if (fk > s->verify_k) s->verify_k = fk;
        s->verify_tokens += 1 + fk;
        for (int32_t j = 1; j <= fk; ++j) ++s->rows[j];
        /* The history the paired step owes this row: one row for the verified token, plus one per
         * draft it verifies (Scheduler::serial_begin's hist_n, lead zero on a decode row). With no
         * head loaded nothing is carried. */
        if (view.k_ceiling > 0) s->history_tokens += 1 + fk;
        if (view.seq[i].ctx_class > s->ctx_class) s->ctx_class = view.seq[i].ctx_class;
        if (view.seq[i].sampling == DraftSampling::Sampled) sampled = true;
    }
    /* The other dimensions are observed here: history from the frozen plan, prefill from the view,
     * both distributions from the cycle. Replay is not, and set its flag above. */
    s->sampling = sampled ? DraftSampling::Sampled : DraftSampling::Greedy;
}

void DraftPlanner::fit_frozen(const DraftBatchView& view) {
    const size_t n = (size_t)view.n_seq;
    if (fr_id_.size() < n) {   /* only when the caller did not size the storage at configure() */
        fr_id_.resize(n);
        fr_slot_.resize(n);
        fr_gen_.resize(n);
        fr_k_.resize(n);
    }
    for (size_t j = 0; j < n; ++j) {
        fr_id_[j]   = view.seq[j].req_id;
        fr_slot_[j] = view.seq[j].slot;
        fr_gen_[j]  = view.seq[j].slot_gen;
    }
    fr_n_ = n;
}

void DraftPlanner::remember(const DraftBatchView& view, const int32_t* k) {
    fit_frozen(view);
    for (size_t j = 0; j < fr_n_; ++j) fr_k_[j] = k[j];
}

void DraftPlanner::remember_published(uint64_t req_id, int32_t slot, uint32_t slot_gen, int32_t k) {
    if (k > c_.k_ceiling) k = c_.k_ceiling;
    if (k < 0) k = 0;
    for (size_t j = 0; j < fr_n_; ++j) {
        if (fr_id_[j] != req_id || fr_slot_[j] != slot || fr_gen_[j] != slot_gen) continue;
        fr_k_[j] = k;
        return;
    }
    if (fr_id_.size() <= fr_n_) {   /* only when configure() did not size the storage */
        fr_id_.resize(fr_n_ + 1);
        fr_slot_.resize(fr_n_ + 1);
        fr_gen_.resize(fr_n_ + 1);
        fr_k_.resize(fr_n_ + 1);
    }
    fr_id_[fr_n_]   = req_id;
    fr_slot_[fr_n_] = slot;
    fr_gen_[fr_n_]  = slot_gen;
    fr_k_[fr_n_]    = k;
    ++fr_n_;
}

double DraftPlanner::expect(const DraftPredictions& pred, int32_t i, int32_t cap, int32_t k) {
    if (k > cap) k = cap;
    const DraftPrediction& p = pred.seq[i];
    double e = 1.0, run = 1.0;
    for (int32_t d = 1; d <= k; ++d) {
        run *= (double)p.a[d - 1];
        e += run;
    }
    return e;
}

bool DraftPlanner::switch_pays(const Priced& cand, const Priced& inc) const {
    /* The dwell the new depth is assumed to run (dwell_of). */
    const double dwell = dwell_of(c_.hold);
    /* The move's own cycle is priced, not assumed free: this cycle pays the candidate shape's
     * execution price (`ms`, the pass it will run) while the tokens it commits are the frozen
     * window's, so the dwell's first cycle earns the incumbent's yield at the candidate's price and
     * the rest earn the candidate's own. That one cycle is the switching cost; over the dwell it is
     * what separates the switch from simply staying. */
    const double tokens = inc.yield + (dwell - 1.0) * cand.yield;
    const double ms     = cand.ms + (dwell - 1.0) * cand.hms;
    /* Not a margin on either score: the net rate of the move against the rate of staying, the
     * dwell's priced switch included. */
    return tokens * inc.hms > (1.0 + c_.hyst) * ms * inc.yield;
}

int32_t DraftPlanner::total_k(const int32_t* k, int32_t n, int32_t* max_k) {
    int32_t sum = 0, top = 0;
    for (int32_t i = 0; i < n; ++i) {
        sum += k[i];
        if (k[i] > top) top = k[i];
    }
    if (max_k) *max_k = top;
    return sum;
}

int DraftPlanner::price(const DraftBatchView& view, const DraftPredictions& pred,
                        const DraftCostModel& costs, const int32_t* k, int32_t clamp_to,
                        int limit, int* evals, Priced* out) const {
    *out = Priced{};

    /* The horizon cost (draft_policy.h): the exact stationary shape that verifies this candidate's
     * own drafts. One shape when every row already verifies its own k; otherwise both are asked or
     * neither -- a candidate is never scored on one lookup of two. The horizon is asked first, so
     * the cycle's last estimate call is still the issued plan's execution shape. */
    bool same = true;
    for (int32_t i = 0; same && i < view.n_seq; ++i) {
        const int32_t ki = clamp_to >= 0 && k[i] > clamp_to ? clamp_to : k[i];
        if (verify_k_of(view, i, ki) != ki) same = false;
    }
    if (*evals + (same ? 1 : 2) > limit) return RAD_OK;   /* not priced: no budget for it */

    DraftVerifyShape shape;
    bool   hpriced = same;
    double hms = 0.0;
    if (!same) {
        ++*evals;
        shape_of(view, k, clamp_to, &shape, /*horizon=*/true);
        DraftCostEstimate h;
        const int hrc = costs.estimate(costs.ctx, shape, &h);
        if (hrc < 0 && hrc != RAD_E_INVAL) return hrc;
        hpriced = hrc >= 0 && h.valid && h.ms > 0.0 && std::isfinite(h.ms);
        if (hpriced) hms = h.ms;
    }

    ++*evals;
    shape_of(view, k, clamp_to, &shape);
    DraftCostEstimate est;
    const int rc = costs.estimate(costs.ctx, shape, &est);
    /* A model refuses a shape it has no key for (draft_shape_priceable, e.g. the replay dimension
     * above). That is "not priced", not a broken cycle: the search then falls back to the named
     * baseline instead of failing the step. */
    if (rc == RAD_E_INVAL) return RAD_OK;
    if (rc < 0) return rc;
    if (!est.valid || !(est.ms > 0.0) || !std::isfinite(est.ms)) return RAD_OK;   /* not priced */
    /* No exact stationary price: not scored. The execution price is another shape's, not a
     * substitute for it, whichever way the verify window moved. */
    if (!hpriced) return RAD_OK;
    if (same) hms = est.ms;

    double y = 0.0;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        int32_t ki = k[i];
        if (clamp_to >= 0 && ki > clamp_to) ki = clamp_to;
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        y += expect(pred, i, c, ki);
    }
    out->yield = y;
    out->ms    = est.ms;
    out->unc   = est.unc_ms > 0.0 && std::isfinite(est.unc_ms) ? est.unc_ms : 0.0;
    out->hms   = hms;
    out->valid = true;
    return RAD_OK;
}

int DraftPlanner::finish(const DraftBatchView& view, int32_t max_k, DraftPlanSource src,
                         DraftFallback fb, DraftPlan* out) {
    out->cycle_id       = view.cycle_id;
    out->policy_version = view.policy_version ? view.policy_version : RAD_DRAFT_POLICY_VERSION;
    out->n_seq          = view.n_seq;
    out->k_ceiling      = view.k_ceiling;
    out->max_k          = max_k;
    out->source         = src;
    out->fallback       = fb;
    out->probe          = DraftProbe::None;
    /* The cycle's replay state as the view evidenced it; `Unknown` when the caller could not say,
     * so a plan never declares a plain run it cannot establish. */
    out->replay         = view.replay;
    out->timing_valid   = false;   /* the planner does not time the cycle */
    out->prefill_tokens = view.prefill_tokens;
    out->pipeline_depth = view.pipeline_depth;
    if (out->cap < view.n_seq) return RAD_E_INVAL;
    /* The window the paired step verifies, per row: this plan's own deepest draft when it verifies
     * what this cycle drafts, the frozen previous plan's otherwise -- re-capped to the plan's own K
     * on the direct path, where the step is built after the plan opened (Scheduler::draft_cap). */
    int32_t verify_top = 0;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        out->ctx_class[i]      = view.seq[i].ctx_class;
        const int32_t verifies = verify_k_of(view, i, out->k[i]);
        if (verifies > verify_top) verify_top = verifies;
        out->history_tokens[i] = view.k_ceiling > 0 ? 1 + verifies : 0;
        out->fill[i] = out->k[i] > 0 ? DraftDrafter::Serial : DraftDrafter::None;
    }
    out->verify_k = verify_top;
    if (draft_plan_rows(*out) < 0) return RAD_E_INVAL;
    const int rc = draft_plan_valid(*out);
    if (rc == RAD_OK) remember(view, out->k);
    return rc;
}

int DraftPlanner::choose(const DraftBatchView& view, const DraftPredictions& pred,
                         const DraftCostModel& costs, DraftPlan* out) {
    gap_refused_ = false;
    last_tiers_priced_ = 0;   /* only a Cost search below fills this */
    last_n_tier_price_ = 0;   /* and with it the per-depth prices */
    last_price_known_ = false;
    last_price_ = DraftTierPrice{};
    if (!out || !out->k || !out->rows || !out->req_id || !out->slot || !out->slot_gen ||
        !out->why || !out->fill || !out->ctx_class || !out->history_tokens)
        return RAD_E_INVAL;
    if (view.n_seq < 0 || view.n_seq > out->cap) return RAD_E_INVAL;
    if (pred.n_seq < view.n_seq || pred.cap < view.n_seq || !pred.seq) return RAD_E_INVAL;

    /* Identity first: a plan names its requests even where it drafts nothing. */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        out->req_id[i]   = view.seq[i].req_id;
        out->slot[i]     = view.seq[i].slot;
        out->slot_gen[i] = view.seq[i].slot_gen;
        out->why[i]      = DraftFallback::None;
        out->k[i]        = 0;
    }
    /* verify_k is set in finish(), once the issued k[] is final: it is this plan's own deepest
     * draft when the paired step verifies what this cycle drafts, the frozen plan's window
     * otherwise (re-capped to the plan's own K on the direct path). */

    /* --draft-calib-k: the operator's sequence, taken and not chosen. Nothing below this branch
     * runs for it -- no price, no ranking, no hysteresis -- and it holds no state beyond its own
     * position. The planner still reads no prediction, but the controller's open_cycle calls
     * predictor.predict for the cycle either way: the set belongs to the cycle, not to this plan.
     * The resource budget still applies: a plan may never ask for more drafts than the step or
     * the pool was given, calibration or not. */
    if (c_.n_calib_k > 0) {
        const int32_t want = c_.calib_k[calib_pos_ % c_.n_calib_k];
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = want < c ? want : c;
            if (view.seq[i].eligibility == DraftEligibility::Eligible && c < want)
                out->why[i] = DraftFallback::Resources;
        }
        apply_draft_budget(view, out->k, out->why);
        int32_t top = 0;
        total_k(out->k, view.n_seq, &top);
        /* Only a cycle with rows moves the sequence on: an empty view has no depth to measure,
         * and advancing here would shift every later depth off its rotation. */
        if (view.n_seq > 0) calib_pos_ = (calib_pos_ + 1) % c_.n_calib_k;
        return finish(view, top, DraftPlanSource::Calibration, DraftFallback::None, out);
    }

    if (c_.mode == DraftPolicyMode::Legacy) {
        const int32_t want = c_.legacy_k > 0 ? c_.legacy_k : c_.k_ceiling;
        int32_t top = 0;
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = want < c ? want : c;
            /* An eligible request bounded below the legacy depth was cut, by its remaining
             * output or the ceiling; an ineligible one drafts nothing by its own state. */
            if (view.seq[i].eligibility == DraftEligibility::Eligible && c < want)
                out->why[i] = DraftFallback::Resources;
            if (out->k[i] > top) top = out->k[i];
        }
        return finish(view, top, DraftPlanSource::Legacy, DraftFallback::None, out);
    }

    if (!costs.estimate) return RAD_E_UNSUPPORTED;

    int evals = 0;
    /* The budget after the search: the cut's re-price, the incumbent and the issued plan's own
     * price, at most two lookups each when pipelined. The alternatives below the deepest gate stop
     * short of that reserve; the gate itself may spend the whole budget, since without it nothing
     * is decided. A budget too small for a call leaves that call unpriced, never half-priced: the
     * cut plan is issued as a Resources fallback, the raise is refused, last_price() is invalid. */
    const int search_limit = c_.max_evals - 3 * (view.pipeline_depth > 0 ? 2 : 1);
    /* From here on a price was asked for: the trace may report a price or its absence, but never
     * as "no search ran" (which is what a calibration or a legacy plan says). */
    last_price_known_ = true;
    last_n_tier_price_ = c_.n_tiers;

    /* Stage 1: the homogeneous baselines. Every offered tier is a candidate on its own exact price;
     * an unpriced one is simply not compared. The deepest is asked first and may spend the whole
     * budget. A missing price for it says nothing about the shallower tiers either way, so it no
     * longer discards them: the deepest wins only when priced and not beaten, and with no price it
     * is out. Only a cycle that prices no tier at all takes the named baseline. Each candidate is
     * its tier clamped to each row's own cap. Ties keep the deepest when it is priced, the
     * shallowest priced tier otherwise. */
    const int32_t deepest = c_.tiers[c_.n_tiers - 1];
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        out->k[i] = deepest < c ? deepest : c;
    }
    Priced  best_p;
    int32_t best_tier = deepest;
    {
        const int rc = price(view, pred, costs, out->k, -1, c_.max_evals, &evals, &best_p);
        if (rc < 0) return rc;
        last_tier_price_[c_.n_tiers - 1] =
            DraftTierPrice{deepest, best_p.valid, best_p.ms, best_p.unc, best_p.hms};
    }

    /* Counted whether or not it wins: the question the ledger answers is "were both depths
     * compared", not "which one won" -- a Cost plan at one tier alone cannot say whether the other
     * lost or was never priced. It is a search metric: the resource cut and the re-price below
     * change what is issued, not what the search weighed, so the count is not the issued depth
     * (that is the plan's max_k, and the price of the plan as issued is last_price()). */
    int32_t tiers_priced = best_p.valid ? 1 : 0;
    for (int32_t t = 0; t + 1 < c_.n_tiers; ++t) {
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = c_.tiers[t] < c ? c_.tiers[t] : c;
        }
        Priced p;
        const int rc = price(view, pred, costs, out->k, -1, search_limit, &evals, &p);
        if (rc < 0) return rc;
        last_tier_price_[t] = DraftTierPrice{c_.tiers[t], p.valid, p.ms, p.unc, p.hms};
        if (!p.valid) continue;
        ++tiers_priced;
        if (!best_p.valid || p.yield / p.hms > best_p.yield / best_p.hms) {
            best_p    = p;
            best_tier = c_.tiers[t];
        }
    }

    if (!best_p.valid) {
        /* No offered tier has a measurement. R's named baseline is evidence only when it runs a
         * shape this ladder can reach: the baseline inside the ladder, or rows whose remaining
         * budget clamps both to the same k. When the baseline would run at a deeper k than the
         * ladder's top -- an explicit ladder below exec_depth, or a configured baseline above it,
         * which Scheduler::init refuses at startup -- its bucket can never be queried by this
         * ladder, so the run would stay in the fallback and never price an offered tier.
         * Diagnose that cycle and issue nothing: extending the ladder or moving the baseline is
         * the operator's call, and a sample must not be learnt at a depth the ladder never offers. */
        if (c_.fallback_k > deepest && !costs.records_only) {
            bool deeper_row = false;
            for (int32_t i = 0; i < view.n_seq; ++i)
                if (cap_of(view.seq[i], c_.k_ceiling) > deepest) {
                    deeper_row = true;
                    break;
                }
            if (deeper_row) {
                if (!gap_reported_) {
                    RAD_ERR("draft policy: the deepest offered tier K%d is unpriced for this "
                            "cycle's shape, and the named baseline would run at K%d; the ladder "
                            "never offers that depth, so the cycle is refused and nothing is "
                            "learned into it (end the ladder at K%d, or set the ceiling to K%d)",
                            (int)deepest, (int)c_.fallback_k, (int)c_.fallback_k, (int)deepest);
                    gap_reported_ = true;
                }
                gap_refused_ = true;
                return RAD_E_UNSUPPORTED;
            }
        }
        /* No tier priced: the named, configured baseline, not an invented price. It is the ladder's
         * middle tier by default (Scheduler: fallback_k, --draft-fallback-k), not the executable
         * depth: a profile that prices no shape of this batch would otherwise run every one of its
         * cycles at the maximum depth. It is not a policy choice, so the hysteresis incumbent does
         * not move. Every tier was asked, and the trace lists each as unpriced. A recording run
         * prices nothing at all by construction, which is why the gap rule above skips it. */
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = c_.fallback_k < c ? c_.fallback_k : c;
        }
        int32_t top = 0;
        for (int32_t i = 0; i < view.n_seq; ++i)
            if (out->k[i] > top) top = out->k[i];
        last_price_.k = top;
        return finish(view, top, DraftPlanSource::Fallback, DraftFallback::UnknownShape, out);
    }

    gap_reported_ = false;   /* a priced cycle re-arms the one diagnosis */
    last_tiers_priced_ = tiers_priced;

    /* Take the winning baseline as the search's starting assignment. */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        out->k[i] = best_tier < c ? best_tier : c;
    }
    double best_score = best_p.yield / best_p.hms;

    /* Stage 2: deterministic coordinate passes. A pass walks the requests in index order and
     * keeps a tier only when it improves the whole batch's tokens per millisecond. */
    for (int32_t pass = 0; pass < c_.passes; ++pass) {
        bool moved = false;
        for (int32_t i = 0; i < view.n_seq && evals < search_limit; ++i) {
            const int32_t cap_i = cap_of(view.seq[i], c_.k_ceiling);
            if (cap_i <= 0) continue;
            const int32_t keep = out->k[i];
            int32_t accepted = keep;
            for (int32_t t = 0; t < c_.n_tiers && evals < search_limit; ++t) {
                const int32_t cand = c_.tiers[t] < cap_i ? c_.tiers[t] : cap_i;
                if (cand == accepted) continue;
                out->k[i] = cand;
                Priced p;
                const int rc = price(view, pred, costs, out->k, -1, search_limit, &evals, &p);
                if (rc < 0) { out->k[i] = keep; return rc; }
                if (p.valid && p.yield / p.hms > best_score) {
                    best_score = p.yield / p.hms;
                    best_p     = p;
                    accepted   = cand;
                    moved      = true;
                }
            }
            out->k[i] = accepted;
        }
        if (!moved || evals >= search_limit) break;
    }

    /* --draft-hysteresis both: the search's reduction is judged by the priced switch rule, as a
     * raise is below. Taken at once, every dip of the acceptance estimate pulled the depth down
     * for at least `hold` cycles before a raise could undo it, so under a noisy estimate the plan
     * sat below the depth its mean asks for. A reduction is not a loss of cheap time though: the
     * dwell's first cycle still commits the deeper window's tokens at the shallower price, and the
     * later ones run the shallower window's own rate, so the rule compares the two rates with that
     * one cycle priced -- a reduction is refused only when the shallower window's throughput,
     * credited with that cycle, does not cover what it stops producing.
     * It is judged before the resource cut, which still applies at once, against the incumbent's
     * own assignment: a row its cap holds under the incumbent is no policy reduction, and an
     * incumbent with no price is no reason to stay. At most one incumbent is priced per cycle -- a
     * held reduction never reaches the raise. */
    if (c_.hold_reductions && chosen_ && view.n_seq > 0) {
        int32_t search_top = 0;
        total_k(out->k, view.n_seq, &search_top);
        if (search_top < last_k_) {
            if (held_k_.size() < (size_t)view.n_seq) held_k_.resize((size_t)view.n_seq);
            bool reduced = false;
            for (int32_t i = 0; i < view.n_seq; ++i) {
                const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
                held_k_[(size_t)i] = last_k_ < c ? last_k_ : c;
                if (out->k[i] < held_k_[(size_t)i]) reduced = true;
            }
            if (reduced) {
                Priced incumbent;
                const int rc = price(view, pred, costs, held_k_.data(), -1, c_.max_evals, &evals,
                                     &incumbent);
                if (rc < 0) return rc;
                if (incumbent.valid && !switch_pays(best_p, incumbent)) {
                    for (int32_t i = 0; i < view.n_seq; ++i) out->k[i] = held_k_[(size_t)i];
                    best_p = incumbent;
                }
            }
        }
    }

    /* Budgets: a positive token or KV budget caps the summed drafts. Scaling every K down is
     * the resource cut that may happen at once (apply_draft_budget, shared with calibration). */
    const bool cut = apply_draft_budget(view, out->k, out->why);

    int32_t top = 0;
    total_k(out->k, view.n_seq, &top);

    /* The cut changed the assignment `best_p` priced: price the plan as it stands, so the
     * hysteresis below compares the plan that will actually be issued. A reduced plan its own
     * shape cannot price is issued unpriced: the uncut plan's cost and yield are not its evidence,
     * and the raise below is refused rather than argued from them. */
    bool issued_priced = true;
    if (cut) {
        Priced p;
        const int rc = price(view, pred, costs, out->k, -1, c_.max_evals, &evals, &p);
        if (rc < 0) return rc;
        issued_priced = p.valid;
        if (p.valid) best_p = p;
    }

    /* The priced switch, both ways by the same rule. A depth change is taken when the yield it
     * gains over the dwell pays for the one cycle the switch spends (switch_pays): the plan is
     * frozen one cycle ahead, so the cycle after a move runs the new shape's price while the tokens
     * it commits are the frozen window's -- it earns the incumbent's yield at the candidate's
     * price, and only the cycles after it earn the candidate's own. A fixed percentage of the
     * incumbent's score was the wrong threshold: it ignored what the move costs and blocked a rung
     * of roughly 12% more time for eight cycles however large the yield gain was, so the run stayed
     * a rung below its class's optimum after every category switch. `unc` takes no part: it is the
     * largest deviation any one sample showed (DraftCostTable::record), 0 at n=1 and only growing
     * with n, so as a penalty it blocked raises to the depths measured most and spared the ones
     * measured least. Added to the raise alone it made every reduction permanent -- with a
     * reference's (3,3) at 30.3 +- 29.4 ms no acceptance could lift a K3 incumbent again. */
    if (chosen_ && top > last_k_) {
        Priced incumbent;
        const int rc = price(view, pred, costs, out->k, last_k_, c_.max_evals, &evals, &incumbent);
        if (rc < 0) return rc;
        if (!issued_priced || !incumbent.valid || !switch_pays(best_p, incumbent)) {
            for (int32_t i = 0; i < view.n_seq; ++i)
                if (out->k[i] > last_k_) out->k[i] = last_k_;
            top = last_k_;
        }
    }
    /* A reduction that only the rows' own caps forced (remaining output, eligibility) is not a
     * policy depth: every row runs at min(incumbent, its cap). Keeping the incumbent here stops a
     * request that ends at max_tokens from dragging the next request's raise down to a depth the
     * profile cannot price as incumbent (e.g. K0 with a frozen window 0), which held it forever. */
    bool forced = chosen_ && top < last_k_ && view.n_seq > 0;
    for (int32_t i = 0; forced && i < view.n_seq; ++i) {
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        if (out->k[i] < (last_k_ < c ? last_k_ : c)) forced = false;
    }
    if (!forced) last_k_ = top;
    chosen_ = true;

    /* The issued assignment's own estimate -- after the budget cut and the hysteresis, so it
     * describes the plan the step will run, not the search's winner. Diagnostic only: a model
     * error here must not refuse a plan the search already decided, so it only leaves the price
     * invalid. The extra estimate spends from the same budget, after every decision was made. */
    last_price_.k = top;
    {
        Priced p;
        if (price(view, pred, costs, out->k, -1, c_.max_evals, &evals, &p) >= 0) {
            last_price_.valid  = p.valid;
            last_price_.ms     = p.ms;
            last_price_.unc_ms = p.unc;
            last_price_.horizon_ms = p.hms;
        }
    }

    return finish(view, top, issued_priced ? DraftPlanSource::Cost : DraftPlanSource::Fallback,
                  issued_priced ? DraftFallback::None : DraftFallback::Resources, out);
}

}  /* namespace rad */
