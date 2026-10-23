/* draft_policy.cpp -- the cost-based draft-depth policy. See draft_policy.h.
 *
 * The objective is the issue's: expected output tokens of the whole batch over its expected
 * batch time. A row that does not draft still emits its one token, so it adds 1 to the yield and
 * costs a verify row at K 0.
 */
#include "draft_policy.h"

#include <cmath>
#include <cstdint>

namespace rad {

namespace {

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
    if (c_.mode == DraftPolicyMode::Cost)
        RAD_INFO("draft policy: cost, ceiling %d, tiers %d..%d (%d), a raise needs %.0f%% and %d "
                 "cycles", (int)c_.k_ceiling, (int)c_.tiers[0], (int)c_.tiers[c_.n_tiers - 1],
                 (int)c_.n_tiers, 100.0 * c_.hyst, (int)c_.hold);
    return RAD_OK;
}

void DraftPlanner::reset() {
    last_k_ = 0;
    held_   = 0;
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
                            DraftVerifyShape* s) const {
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
         * K on the direct path (Scheduler::draft_cap). */
        const int32_t fk = verify_k_of(view, i, ki);
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

void DraftPlanner::remember_legacy(const DraftBatchView& view, int32_t head_depth) {
    if (head_depth > c_.k_ceiling) head_depth = c_.k_ceiling;
    if (head_depth < 0) head_depth = 0;
    fit_frozen(view);
    for (size_t j = 0; j < fr_n_; ++j) fr_k_[j] = cap_of(view.seq[j], head_depth);
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
                        int* evals, Priced* out) const {
    *out = Priced{};
    if (*evals >= c_.max_evals) return RAD_OK;
    ++*evals;

    DraftVerifyShape shape;
    shape_of(view, k, clamp_to, &shape);

    DraftCostEstimate est;
    const int rc = costs.estimate(costs.ctx, shape, &est);
    /* A model refuses a shape it has no key for (draft_shape_priceable, e.g. the replay dimension
     * above). That is "not priced", not a broken cycle: the search then falls back to the named
     * baseline instead of failing the step. */
    if (rc == RAD_E_INVAL) return RAD_OK;
    if (rc < 0) return rc;
    if (!est.valid || !(est.ms > 0.0) || !std::isfinite(est.ms)) return RAD_OK;   /* not priced */

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
    /* From here on a price was asked for: the trace may report a price or its absence, but never
     * as "no search ran" (which is what a calibration or a legacy plan says). */
    last_price_known_ = true;
    last_n_tier_price_ = c_.n_tiers;

    /* Stage 1: the homogeneous baselines. The deepest offered plan is the coverage gate: if it
     * cannot be priced, the batch takes the named baseline -- never a shallower tier that happens
     * to be priced. A missing measurement for the deeper alternative is not evidence that the
     * shallow one is cheaper, and letting the shallow tier decide would present an uncovered
     * depth as a cost decision. The deepest candidate is the last tier clamped to each row's own
     * cap; every other tier must be priced on its own to beat it. */
    const int32_t deepest = c_.tiers[c_.n_tiers - 1];
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        out->k[i] = deepest < c ? deepest : c;
    }
    Priced  best_p;
    int32_t best_tier = deepest;
    {
        const int rc = price(view, pred, costs, out->k, -1, &evals, &best_p);
        if (rc < 0) return rc;
        last_tier_price_[c_.n_tiers - 1] =
            DraftTierPrice{deepest, best_p.valid, best_p.ms, best_p.unc};
    }

    if (!best_p.valid) {
        /* The deepest offered plan has no measurement. R's named baseline is evidence only when it
         * runs the offered top's own shape: the default ladder (fallback_k == the deepest tier) or
         * rows whose remaining budget clamps both to the same k. When the baseline would run at a
         * deeper k -- an explicit ladder below exec_depth -- its bucket can never be queried by
         * this ladder, so the run would stay in the fallback and never price the offered top.
         * Diagnose that cycle and issue nothing: extending the ladder or moving the baseline is
         * the operator's call, and a sample must not be learnt at a depth the ladder never offers. */
        if (c_.fallback_k > deepest) {
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
        /* The deepest offered plan has no measurement: a named baseline, not an invented price --
         * and specifically not the price of a shallower plan this batch did not offer as deep. */
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = c_.fallback_k < c ? c_.fallback_k : c;
        }
        int32_t top = 0;
        for (int32_t i = 0; i < view.n_seq; ++i)
            if (out->k[i] > top) top = out->k[i];
        last_price_.k = top;
        /* The gate stopped the search here: the deepest offered tier is the only depth anything
         * was asked about, and its price was not obtained. The shallower alternatives were never
         * compared, so the trace lists one depth, not a ladder of unasked ones. */
        last_tier_price_[0] = DraftTierPrice{deepest, best_p.valid, best_p.ms, best_p.unc};
        last_n_tier_price_  = 1;
        return finish(view, top, DraftPlanSource::Fallback, DraftFallback::UnknownShape, out);
    }

    gap_reported_ = false;   /* a priced cycle re-arms the one diagnosis */

    /* The deepest tier priced (or the code above would have returned), and every shallower one
     * that did. Counted whether or not it wins: the question the ledger answers is "were both
     * depths compared", not "which one won" -- a Cost plan at the deep tier alone cannot say
     * whether the shallow alternative lost or was never priced. It is a search metric: the
     * resource cut and the re-price below change what is issued, not what the search weighed, so
     * the count is not the issued depth (that is last_k, and the price of the plan as issued is
     * last_price()). */
    int32_t tiers_priced = 1;
    for (int32_t t = 0; t + 1 < c_.n_tiers; ++t) {
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = c_.tiers[t] < c ? c_.tiers[t] : c;
        }
        Priced p;
        const int rc = price(view, pred, costs, out->k, -1, &evals, &p);
        if (rc < 0) return rc;
        last_tier_price_[t] = DraftTierPrice{c_.tiers[t], p.valid, p.ms, p.unc};
        if (!p.valid) continue;
        ++tiers_priced;
        if (p.yield / p.ms > best_p.yield / best_p.ms) {
            best_p    = p;
            best_tier = c_.tiers[t];
        }
    }
    last_tiers_priced_ = tiers_priced;

    /* Take the winning baseline as the search's starting assignment. */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        out->k[i] = best_tier < c ? best_tier : c;
    }
    double best_score = best_p.yield / best_p.ms;

    /* Stage 2: deterministic coordinate passes. A pass walks the requests in index order and
     * keeps a tier only when it improves the whole batch's tokens per millisecond. */
    for (int32_t pass = 0; pass < c_.passes; ++pass) {
        bool moved = false;
        for (int32_t i = 0; i < view.n_seq && evals < c_.max_evals; ++i) {
            const int32_t cap_i = cap_of(view.seq[i], c_.k_ceiling);
            if (cap_i <= 0) continue;
            const int32_t keep = out->k[i];
            int32_t accepted = keep;
            for (int32_t t = 0; t < c_.n_tiers && evals < c_.max_evals; ++t) {
                const int32_t cand = c_.tiers[t] < cap_i ? c_.tiers[t] : cap_i;
                if (cand == accepted) continue;
                out->k[i] = cand;
                Priced p;
                const int rc = price(view, pred, costs, out->k, -1, &evals, &p);
                if (rc < 0) { out->k[i] = keep; return rc; }
                if (p.valid && p.yield / p.ms > best_score) {
                    best_score = p.yield / p.ms;
                    best_p     = p;
                    accepted   = cand;
                    moved      = true;
                }
            }
            out->k[i] = accepted;
        }
        if (!moved || evals >= c_.max_evals) break;
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
        if (evals < c_.max_evals) {
            const int rc = price(view, pred, costs, out->k, -1, &evals, &p);
            if (rc < 0) return rc;
        }
        issued_priced = p.valid;
        if (p.valid) best_p = p;
    }

    /* held_ counts the cycles that have run at last_k_; the plan issued here is not one of them
     * yet, and a cycle with no rows is not a cycle at the depth. */
    if (view.n_seq == 0) {
        /* nothing ran: the depth did not hold, it had nothing to hold over */
    } else if (chosen_) {
        if (held_ < INT32_MAX) ++held_;
    } else {
        held_ = 0;
    }

    /* Hysteresis and hold: a raise waits, a reduction is immediate. A raise is judged
     * pessimistically (`ms + unc`), the incumbent at its known cost; with no price for the
     * incumbent it is refused. `ms + unc` is a heuristic ordering of two measured means, not a
     * bound on a future cycle: at n=1 `unc` is 0, and the brackets behind a bucket cover only the
     * shapes that ran, so it neither proves dominance nor justifies a raise on its own. */
    if (chosen_ && top > last_k_) {
        Priced incumbent;
        const int rc = price(view, pred, costs, out->k, last_k_, &evals, &incumbent);
        if (rc < 0) return rc;
        const double raised = issued_priced ? best_p.yield / (best_p.ms + best_p.unc) : 0.0;
        if (!issued_priced || held_ < c_.hold || !incumbent.valid ||
            !(raised > incumbent.yield / incumbent.ms * (1.0 + c_.hyst))) {
            for (int32_t i = 0; i < view.n_seq; ++i)
                if (out->k[i] > last_k_) out->k[i] = last_k_;
            top = last_k_;
        }
    }
    if (top != last_k_) held_ = 0;   /* a new depth starts a new hold */
    last_k_ = top;
    chosen_ = true;

    /* The issued assignment's own estimate -- after the budget cut and the hysteresis, so it
     * describes the plan the step will run, not the search's winner. Diagnostic only: a model
     * error here must not refuse a plan the search already decided, so it only leaves the price
     * invalid. The extra estimate spends from the same budget, after every decision was made. */
    last_price_.k = top;
    if (evals < c_.max_evals) {
        Priced p;
        if (price(view, pred, costs, out->k, -1, &evals, &p) >= 0) {
            last_price_.valid  = p.valid;
            last_price_.ms     = p.ms;
            last_price_.unc_ms = p.unc;
        }
    }

    return finish(view, top, issued_priced ? DraftPlanSource::Cost : DraftPlanSource::Fallback,
                  issued_priced ? DraftFallback::None : DraftFallback::Resources, out);
}

}  /* namespace rad */
