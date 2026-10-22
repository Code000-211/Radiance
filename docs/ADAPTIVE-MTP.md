# Adaptive MTP: contracts, profile and ownership

This is the working agreement for the adaptive-MTP work on Qwen3.8-Flash-Next, TP2 on two Radeon
AI PRO R9700. `spec.md` remains the binding design; this file records the internal seam (P00), the
versioned profile and benchmark-manifest formats, the CLI names, and which files each strand owns.

**Status: host-wired, inactive by default.** The contracts, their fakes and the scheduler wiring
exist and are tested on the host: B's predictor and A's cost table are both fed from the cycle
ring, and the cost policy is opt-in (`--draft-policy cost`; the default stays `legacy`). Cost is
still inactive without a configured profile: `--draft-profile PATH` is read once at startup and
configured against this deployment's fingerprint before `Scheduler::init`, and without a matching
profile `--draft-policy cost` is refused there by name (the baseline lock: an active table **and**
either a bucket usable by the rule `estimate()` applies or an explicit `admit_fallback()`) and
every estimate takes its named fallback. The sample path only measures: it activates nothing. No
profile has been generated or measured yet and no M1/M2 or GPU result is claimed here.

## The seam

`core/sched/draft_types.h` declares four hooks and the data they exchange:

| Hook | Owner | Signature |
|------|-------|-----------|
| `predict(view, predictions)` | B | fills per-request, per-position `DraftPrediction` |
| `estimate(shape, out)` | A | prices one `DraftVerifyShape` as milliseconds |
| `choose(view, predictions, costs, plan)` | C | writes the cycle's `DraftPlan` once |
| `observe(observation, sample)` | A + B | learns from a completed cycle, after `complete` |

The integrator wires one `DraftHooks` into the scheduler and stamps `version` into every plan,
prediction and sample.

### Invariants every strand may rely on

- **Caller-provided memory, no allocation.** Every array is a pointer into storage the caller
  owns, sized at `max_seqs` (requests) or `RAD_SCHED_MAX_SPEC` (draft positions). A whole hook call
  is allocation-free. Profile I/O and its buffers stay outside the step path.
- **Arrays are bounded by `RAD_SCHED_MAX_SPEC` (64)**, libr4d's verify query-row band. K is
  counted after the committed token; the first target is K = 0..6. `draft_plan_rows`,
  `draft_plan_valid` and `draft_plan_identifies` refuse `n_seq > cap` before they read `k[]` or the
  identity arrays, so a caller cannot make a contract check read past the storage it sized -- the
  identity check included, which is the one a delivery path calls on a plan it did not issue.
- **A plan is immutable from issue until every pass it names is consumed.** `choose` writes it
  once; no other hook writes into it, and `observe` takes only const inputs. Historical plans stay
  identifiable while `issue_ahead()` uses them; the scheduler sizes its ring for the live cycles
  in advance, and a telemetry overflow drops and counts records rather than blocking the card.
- **Lifecycle: Preempt → Recycle → Resume, as a predicate, not as storage.** P00 supplies the
  *check*, `draft_observation_deliverable`, and the identity it applies; it does not supply a ring,
  an owner, an allocation or a free. The cycle ring -- a plan and its observation kept alive until
  the observation is delivered or abandoned, with copies of `req_id`, `slot` and `slot_gen` rather
  than pointers into live slot state -- is **C's** to build and to prove. A copied `DraftPlan`
  struct whose `req_id`/`slot`/`slot_gen` were copied but whose `k`, `fill`, `rows`, `ctx_class` and
  `history_tokens` still point into the issuing caller's storage is not ownership of the plan: the
  lifetime test does exactly that, so it exercises the predicate, and the doc claim stops here.
  Preemption stops the work but does not discard a cycle that already ran: its observation is
  delivered when `complete`. Recycling a slot resets slot state but must not free or rewrite a plan
  or observation still in flight; resume starts a new cycle with a new plan, and the old observation
  never merges into it. Delivery never consults the live slot -- so a recycled generation cannot
  drop a measured cycle -- and refuses a row whose origin does not fit its plan, and a plan that
  fails its own gate.
- **History belongs to the request, the slot generation is only the binding.** A prediction state
  (`a`, `n`, `ess`) survives preemption and resume with its `req_id`; a new slot generation for the
  same `req_id` is adopted, not reset, and only a different `req_id` starts from the prior. An
  observation row is matched by `req_id` for the same reason: waiting for the slot generation a
  preempted request no longer occupies would wait for a generation that never comes back. The
  scheduler copies the state out of the slot at preempt and back in at resume, or holds it
  off-slot; B cannot, because the table belongs to the slot. A smaller `max_seqs` may not be
  configured while an open cycle names a slot above it -- shrink is refused or applied after the
  drain, never by dropping the cycle.
- **Delivery and application are two acts, and application is per row and counted.** The
  scheduler delivers a completed cycle once (at the next cycle's open, or at shutdown); the
  observer answers `RAD_OK` even when it applied nothing, so that return value is not evidence of
  application. `DraftPredictModel::observe` counts the rows apart: `attempted` (every
  request/sequence row of a deliverable cycle -- one per row, not one per token position, so a
  wider verify window adds labels inside a row and does not raise this count), `applied` (a row
  that applied a learning update -- at least one position's learning state moved, which is not a
  promise that the numeric estimate changed) and `skipped` (a row with no teachable label, or one
  whose `(req_id, slot_gen)` no longer binds). A cycle the predictor refuses, and one that never
  completed, are `discarded` by the scheduler -- a refused observation is a dropped cycle, not a
  retried one. `SchedMetrics` exposes all four
  (`draft_obs_attempted`, `_applied`, `_skipped`, `_discarded`): a rising `attempted` with
  `applied` flat is a policy whose rows teach nothing, and a non-zero `discarded` is a cycle lost
  before it taught. A row that cannot bind is skipped, never applied partially, so one cycle's
  evidence cannot learn the same row twice.
- **The last cycle is delivered before its identities are lost.** Delivery at "the next open"
  leaves the final cycle of a run with no open to deliver at; `shutdown` and `fini` flush the ring
  before the requests' slot generations are released. A complete record is delivered there exactly
  once (`draft_ring_cur_` is cleared and the record retired, so a second call is a no-op); an
  incomplete one -- a row that left and never reported -- is dropped and counted in
  `draft_obs_discarded`, not delivered with a hole. The cycle's cost sample was already taken when
  it resolved, so the flush changes only label delivery.
- **The per-row windows are quantities of the cycle, not of one K.** A row's `generated` is bounded
  by its own `plan.k[i]`. Its `verified` is bounded by the window the paired step could have run:
  `plan.verify_k` (what a plan issued ahead verifies, i.e. the frozen previous plan's
  distribution) **or** this plan's own `k[i]` where the head published nothing and the step filled
  from the prompt lookup -- that fill is verified in the same step that made it, so a K raised
  above the frozen window is legitimate and a complete cycle must not be dropped for it. Which of
  the two the verify consumed is `label_origin`, not this bound. `draft_observation_deliverable`
  refuses two rows for the same `req_id` in one cycle: one cycle's evidence is applied to a state
  once.
- **Completion is evidence, not a rank count.** The rank handoff is a host barrier, which joins
  each rank's *enqueue*, and `step_end_` is a card-local event recorded on rank 0: neither says
  another rank's stream has finished. `Engine::await_step_end()` is therefore the step's one
  completion point -- it waits on rank 0's event and on every other rank's own end event, all
  recorded at the same place, before the step issued ahead. `collect_step` uses it wherever it
  already waits and, on the no-rows path, waits whenever the step issued a history pass: a
  row-less pass is enqueued but read back by nothing, so returning without that wait would leave
  "no row to collect" indistinguishable from "the pass completed". A step that owes no pass keeps
  its run-ahead. The outcome is one flag, `StepOut::complete` -- set only after the wait, cleared
  when the next step is prepared, and never set when the wait fails -- so a driver reads
  `history_ran = serial_hist_issued_ && so.complete` and never derives completion from `n_rows`, a
  non-null batch, a single rank or an enqueue. A failed, rejected or history-only batch has a
  distinct outcome from a completed one and never advances a water level.
- **Identity is `(cycle_id, policy_version, req_id, slot, slot_gen)`.** A recycled slot is another
  request, and `draft_plan_identifies(plan, observation)` is the check that the rows a hook is
  handed came from the plan they name — the observation carries both the identity arrays and the
  plan pointer, and either one alone can be stale. One memory set per live cycle: a plan is not
  rewritten while an observation or an issued-ahead pass still names it.
- **Version checks are two, and they are not interchangeable.** A prediction set describes the
  view it was computed for: `draft_predictions_current` requires `predictions.version` and the
  request rows to be the view's, so a set left over from an older policy is stale rather than
  silently reused. An observation describes the plan it names: its version must equal that plan's,
  not the hooks' current version, so a consistent plan issued under an older policy stays
  observable while it is alive.
- **Origin is part of the cycle, not a side list.** `DraftPlan::fill[]` is the permission (who may
  spend `k[]`); `DraftSeqObservation::origin` is the fact (who produced the row: `Head`, `Lookup`
  or `None`). `draft_origin_fits_plan` refuses a lookup row behind a head budget, a head row behind
  a lookup budget, any proposal on a K0 row the plan barred -- the K0/fallback leak -- and the
  amount: `Head`/`Lookup` require `0 < generated <= k[i]`, so a row reporting more proposals than
  its budget is not this plan's work, and a row that produced nothing is `None` whatever its budget
  was. An observation with no origin is not priced as head.
- **Learning labels carry the provenance of the window they measured, not of the plan.** A cycle
  record keeps two windows in one row: `origin`/`generated` is what this plan drafted and published
  after the commit, while `checked`/`verified` is what the paired step verified -- the head
  proposals the previous plan published, or the step's own prompt-lookup fill. `origin` cannot
  describe the labels: a cycle whose own plan publishes head proposals may have verified a lookup,
  and a cycle that publishes nothing may have verified the head. So the record stamps
  `label_origin`/`label_generated` from the verify itself (`StepEntry::n_spec` and `ngram`, at
  report time), `draft_seq_observation_valid` ties a `None` label origin to zero labels, and
  `DraftPredictModel::learn` gates on the labels only. Teaching from `origin` mislabels both
  directions: a lookup's matches would be filed as a head's, and a head's accepted prefix would be
  dropped because the following plan published nothing.
- **The current verify-K (`verify_k`) and the next draft-K (`max_k`) are separate fields:**
  reading one for the other is how a proposal from another generation leaks into a step.
- **`k[]` is the budget for every drafter.** The plan names per request who may fill it (`fill[]`):
  `None` exactly where K is 0, the serial head or the n-gram fallback where K is positive. A
  request the plan puts at K 0 stays empty; the scheduler's n-gram fallback is a drafter too and
  may not fill it (`draft_plan_bars_fill`). The view's batch-wide `ngram_allowed` is the gate the
  plan is chosen under, not a licence to override the plan.
- **The plan carries the shape it was priced on.** Per request `ctx_class` and `history_tokens`,
  cycle-wide `prefill_tokens`, `replay`, `timing_valid` and `pipeline_depth`. A historical plan
  kept alive by `issue_ahead()` can then still say what it priced, and a K0 plan with
  `history_tokens > 0` is "policy K0 with history", not "no work".
- **Verify distribution and planned draft distribution are separate keys.** `DraftVerifyShape`
  carries `rows[]` for the verify pass and `draft_rows[]`/`draft_k` for the draft rounds the plan
  asks for; under `pipeline_depth > 0` they belong to different cycles and may name different K.
  `rounds` plus one total cannot tell two shrinking distributions apart, so `DraftCostSample`
  carries the executed distribution per round (`draft_round_rows[]`). `draft_shape_matches_plan`
  checks the planned distribution against the plan's `k[]`; at `pipeline_depth == 0` it also binds
  `verify_k` and `rows[]` to the plan, because nothing is issued ahead and the plan verifies the
  window it drafts (`draft_plan_valid` binds `verify_k == max_k` there). A plan at K 0 is the same
  separation at plan level: `max_k == 0` with `verify_k > 0` drafts nothing and verifies the window
  a previous cycle published, which only `pipeline_depth > 0` admits. `draft_shape_priceable`
  requires both distributions to be monotone and complete, `verify_tokens` to equal
  `n_rows + sum_j rows[j]`, and the deepest round of each to be non-empty: `verify_k > 0` with
  `rows[verify_k] == 0` is a missing shape, not a measured zero.
- **The extent priced is the extent that ran, not the one published.** Under the adaptive window
  `serial_begin()` gives each row the rounds its own cap allows (`draft_cap`: its stop inside the
  cycle's D), and `serial_batch()` runs each round over the rows that still have one. A row that
  stops at 3 in a cycle whose deepest row runs 6 executes three rounds, and `expect_draft()` caps
  its verify to the same three. The ring keeps both facts separately: `generated` is what the head
  executed for the row and `published` what the paired step will verify. `draft_cost_observe`
  derives `rounds`/`draft_round_rows[]` from `generated`, so a commit that cut a row's verify
  below the window it drafted still prices the rounds the bracket timed: a step issued ahead for
  three that the head ran for six prices six executed rounds, with three published and verified,
  and the sample's `verify_k` is that three. A row that executed past
  its plan entry is refused by `executed_within_plan` (`RAD_E_INVAL`) and counted as a refusal --
  visibly unpriced, never priced at a window it did not run.
- **A missing measurement is not zero.** `DraftVerifyShape::unknown` names an absent history,
  prefill, replay or plan distribution; `draft_shape_priceable` refuses the shape, and the caller
  takes its named fallback. A zero is only a zero when it was measured as one.
- **Labels only where measured.** `draft_label` returns 1 on the accepted prefix and 0 at a real
  rejection, and nothing else. `DraftSeqEnd` says how the labelled span ends: `Rejected` (a real
  0 at `accepted`), `Stopped` (the target agreed on a stopping token that is not committed —
  compared, *not* a reject, no label), `AllAccepted`, or `Unset` (refused, never guessed). EOG,
  `max_tokens`, cancellation and the context end remain the `DraftCensor` reason; a censor does not
  erase a rejection measured before it. Raw match length, commit length, and the censor reason are
  separate fields, which is what lets F assert a correctly hit but uncommitted EOG without
  inventing a reject label. `accepted` is the committed draft count, not the emitted token count:
  the step emits those drafts plus the target's own token at the next position (`accepted + 1`
  tokens), which is the EOG when the cut hits one. Reading `accepted` as an emitted length
  understates every step by one and makes a hit EOG look like a lost token. The tally is still
  bounded where it protects a label:
  `draft_seq_observation_valid` requires `raw_match == accepted` for `Rejected` (otherwise the 0 at
  `accepted` would deny a token the drafter got right) and for `AllAccepted` (the row claims all it
  checked was committed), and `raw_match <= verified` always -- the paired verify window, not this
  plan's `generated`: a plan issued ahead drafts and verifies different windows, and `raw_match` is
  `accept_greedy`'s uncensored count taken before the end-of-generation cut, so the cut must not
  shrink it. The two windows are never chained: `verified` is not bounded by `published`, because a
  verify window above the draft window computes positions this plan did not publish and one below
  it stops short of them. `accepted` is the paired verify's commit count, not this plan's published
  count: a K0 plan issued ahead commits drafts a previously published plan produced, so
  `accepted > published` is a measured cycle rather than a contradiction, and neither refusing it
  nor zeroing `accepted` to fit a chain is admissible. `Stopped` carries no label at `accepted`, and
  the censor decides where commitment stopped -- `DraftController::observe` takes the committed count
  from the card -- so a taller tally is accepted there: refusing it would lose every real label in a
  censored cycle.
- **The output budget includes the bonus token.** `remaining` counts the tokens the request may
  still emit, and the verify step always emits the target's bonus token after the accepted prefix.
  Drafts are therefore bounded by `remaining - 1` (`draft_budget_k`); with one token left the
  request drafts nothing. A plan that clamps at `remaining` overruns the output budget by one
  token per verify step.
- **Units are milliseconds or nothing.** An unmeasured shape is `valid = false` or
  `DraftCostSource::ConservativeFallback` with a bound; a unitless prior is never reported as ms.
- **Only cleanly attributed samples update a measured play bucket** (`draft_sample_attributable`):
  the sample names the cycle and policy version the observation names, has a finite positive
  bracket, `kDraftSampleValid`, and no warmup, record, audit, admission, placement or shortened
  flag. A sample with `cycle_id = 0` is unattributed and never accepted; a non-finite or non-positive
  total is not a duration. Real DMA cost is routed away by its flag, not discarded as an outlier.
  An `ahead` sample stays flagged so the model can weigh it. The observation overload also requires
  a completed cycle over a plan that passes `draft_plan_valid`: an unfinished cycle has no bracket
  to fold in, and a plan that fails its own gate names no shape the sample could have been measured
  under.
- **The sample reports the run, the plan reports the intention.** `DraftCostSample::replay` is what
  the executing path did; `DraftPlan::replay` is what was decided at issue time and is never proof
  that a replay ran. `DraftReplay::None` is a claim of a plain run and may only be written from
  evidence -- a caller that cannot tell leaves `Unknown`, including across a mixed record/play
  pass. `draft_sample_attributable` enforces both halves: a sample whose replay state is `Unknown`
  fills no bucket, and the bucket's key must be the state the run established
  (`sample.shape.replay == sample.replay`), so a plan's declared `None` cannot key a sample the
  execution did not evidence. `kDraftSampleShortened` marks a cycle that ended before its planned
  extent (fewer rounds than `plan.max_k`, or a cancelled row): its bracket times less work than the
  shape it would be filed under, so it is kept as an online diagnostic, never folded into the
  bucket it would otherwise name. `shortened` and the unpriced bucket (`kDraftBucketUnpriced`)
  stay **online diagnosis**: the profile format gets no name for either — a new format for
  diagnostic buckets is not needed — and neither is fed back as a regular measurement or used as
  a fallback anchor. A truncated bracket prices nothing, not even another shape. `timing_valid`
  is the executing path's statement, never the planner's: the plan is issued with it false, and
  only the cycle ring's record of that plan takes it from a bracket `step_finished()` closed --
  one opened before the step's build and closed after its draft rounds. A cycle whose bracket
  never opened or never closed keeps it false, and `observe` refuses that sample (`RAD_E_INVAL`)
  rather than pricing it at zero; nothing flips the field afterwards to force a sample through.
- **The planner prices a replay state only where the deployment evidences it.** The estimate's
  key carries `replay`, and `DraftBatchView::replay` is how it enters the cycle: a runtime whose
  backend cannot record a pass runs every pass live and unkeyed, so the engine reports `None` --
  but only when every rank proves it (`Ctx::records_passes()` false on all of them, see
  `draft_replay_evidence`); one recording rank or no rank asked leaves the cycle `Unknown`. Then
  the cycle is priced against the buckets that runtime measured. That is necessary, not
  sufficient: a cycle whose shape the file does not hold still falls back -- a request's first
  cycle verifies nothing, so it is keyed `verify_k = 0`, and a legacy calibration files only cycles
  that verify the window it already published; a `cost` calibration (record-only) files the start
  cycle itself. A recording deployment stays `Unknown`, because which of play, record and audit a
  pass does exists only once that pass runs, after its plan was issued; the shape is then
  unpriceable and the named fallback is issued. The default is `Unknown`, so a caller that says
  nothing never asserts a plain run.
- **Non-play buckets have their own path.** A record or audit bracket is stored through
  `draft_sample_storable` with its flag, in the bucket that names it, and never folded into a play
  bucket; there is no flag combination that makes it attributable.
- **Evidence belongs to the step that produced it, not to the collector's window.** The cycle is
  closed when the next plan opens, but `step_loop` issues the next step's ahead pass before it
  collects the current one, so a counter read at collect time describes the ahead pass: its tape
  branch and its stager answer are in the delta, and a step with no pass at all is indistinguishable
  from an undisturbed one. `Ctx::run_step` freezes `{step, tape, staged}` on the rank thread as the
  pass returns (`Ctx::StepEvidence`), and that snapshot is merged per rank into a slot keyed by step
  number (`Rank::StepSnap`); a draft pass shares the step number and keeps the trunk pass's slot
  untouched, but its tape class is merged into the same slot (`Ctx::last_pass_tape`, Record > Audit
  > Play > None), so a recording or audited draft pass cannot leave its bracket filed as Play.
  `Engine::step_evidence(step)` merges the ranks the same way, and a step no rank ran a pass for
  yields `replay_known = false`, which the sample treats as `Unknown` and excludes. A per-step
  placement fact (`staged`) and heat movement inside the step's own window (`moved`, bounded by the
  counters `prepare_release` snapshots) travel the same way. `moved` is also frozen by a
  sampler-only release when its record still names the step: that release carries the tiering
  dispatch of a step issued ahead, and a step with no draft pass (a shared-KV head that drew K=0)
  has no later release to report it before the next step re-bases the counter. Both the snapshot
  and the comparison use `HeatStats::moves()`, so the initial fill retires into a bracket without
  stamping it: that bracket is the start placement being reached, and marking it would throw away
  the only clean bracket a calibration has.
- **A move between two brackets belongs to the bracket after it.** The per-step facts cover one
  bracket, but the scheduler's own tier and arena ticks run after `step_finished()` and move units
  with no step to stamp. The engine therefore reports `DraftStepEvidence::moves_since_bind`, the
  cumulative moves over every rank since the profile was bound -- swap promotions plus every
  demotion -- and compares it with the count at the loop's start. Once the count is non-zero the
  record's `placed` is sticky: every later cycle is marked placement-affected, including one whose
  own bracket saw no move, because the run cannot move back to the start placement. Dropping only
  the bucket the mover directly marked would leave a later clean-looking bracket to be priced as
  the start residency.
- **The start placement resolves once, and the fill that materialises it is exempt only before
  that.** A dynamic plan's fill promotes pooled units into slots the KV cache lends; the start
  placement is the plan plus that fill, so the profile may only be priced once the fill has
  landed. `Mover::fill_settled()` says so: the fill has run since the loan, no covered lent slot
  is free past the working reserve (the turnover slots are part of the start placement, and the
  balance keeps them free until turnover is held), and no copy it issued is still landing. A plan
  that never fills is settled from the start. The engine latches it
  (`Engine::start_placement_settled`, `Engine::settle_start_placement`) only between two
  brackets, so a bracket is never stamped with a placement it ran without: until the resolve,
  every record carries `DraftStepEvidence::placement_unresolved` and is marked, and
  `DraftCostTable::hold_prices` makes `estimate()` answer nothing, so the run executes its named
  baseline instead of pricing a residency it has not reached. The idle branch is not the only
  read: a run under continuous load may never take it, so `step_loop` reads the same predicate
  again after `publish(false)`, when the bracket is closed and the rank threads are parked --
  otherwise a fill that landed under load would keep the prices held for as long as requests kept
  arriving. That mark is per bracket, not sticky -- the first clean bracket after the fill has
  landed is a measurement of the start placement again.
- **The initial fill is provably the plan's, not a heat decision.** `Mover::fill()` marks a
  promotion `initial` when it is issued before the first observation (`HeatStats::
  initial_fill_promotions`); at zero dispatches every heat is zero, so the units are taken by layer
  and the amount is a function of the plan and the lent slots. The loan that bounds it is settled
  before serving, and the fill is completed there: `Engine::serve` runs one `loan_tick` with the
  cache still empty and then `Engine::complete_start_fill`, which holds turnover -- the production
  balance keeps those slots free and calls the loan settled that way -- runs the held balance at
  least once even if `Mover::fill_settled()` already reads true, and repeats it until
  `Mover::fill_settled()`. Every wave stays at zero dispatches, so the reserve is
  materialised as part of the start placement instead of being filled by heat in the first steps.
  A fill that cannot settle, and any device, stream or mover error on the way, is `serve`'s error:
  the run stops before the fingerprint, the loaded profile and the prices rather than measuring a
  placement it never reached.
  The mover sizes one record per lent slot (`Mover::lend_from`), so the wave split is the reserve,
  or a wave a drain cut short, not record exhaustion. The fingerprint names the loan and the slots
  it covers (`flex=`/`lend=`), which is what binds the resolved start placement to the identity. A
  fill issued after one observation, or any fill after the resolve (a loan growing again), is not
  initial: `Engine::residency_moves()` exempts the initial promotions only up to the per-rank count
  frozen at the resolve (`Engine::rank_fill_exempt`), so a later fill counts as a move even though
  it promotes rather than evicts. The same rule feeds the step's own snapshot and the bracket's
  count, so a fill that lands after the resolve is a move in all three. There is no re-baselining:
  a move made before the resolve keeps counting. That covers every way a lent slot changes hands
  after the start placement: a loan that grows back after a shrink, a loan that follows freed KV
  blocks, and a donor a migration moved out of a lent slot -- the slot it freed is refilled by a
  later `Mover::fill`, whose promotions are no longer initial.
- **Both abort gates distinguish a calibration.** In the online policy a divergence deactivates
  the loaded table (`DraftCostTable::deactivate`, reason "residency drifted from the fingerprint's
  start placement"): it stops pricing and stops learning, and it does not `admit_fallback()` --
  drift is no admission to price without the measurement the deployment was admitted on. An
  explicit record-only calibration (`DraftCostTable::record_only`, `--draft-policy cost` with
  `--draft-profile-out`) prices nothing by construction, so a move ends neither its recording nor
  the run: it keeps measuring the named baseline, marks every record after the move, and the
  export leaves those cycles out. Both gates read `records_only()` explicitly -- `draft_drift_note`
  does not deactivate it, `draft_cost_cycle` refuses no cycle for it, and `draft_cost_aborted()`
  stays false -- because a calibration that stopped at the first move would never see the
  post-move baseline it exists to record. There is no blanket drift exception: a loaded profile
  that does price keeps the fail-closed rule above, and a record-only table never gains a price it
  did not measure.
- **The online policy fails closed before the next planning, not inside the step it affects.**
  `step_finished` is not the only place the count is read: the engine reports the same cumulative
  evidence at the top of every `step_loop` iteration (`Scheduler::draft_residency`), before
  `promote_next()`/`step()` open the next cycle. The scheduler's idle ticks run after
  `step_finished`, so a move published there would otherwise be seen only by the cycle after the
  one it priced. Once the divergence is noted under `cost` with an online table,
  `draft_cost_cycle()` refuses to open a cycle at all: it sets a one-way latch
  (`draft_cost_aborted`), logs the deactivation reason, and
  leaves the last plan in force. `step()` returns no plan, and `promote_next()` does not make the
  step issued ahead current: that work was already issued to the ranks but is never collected or
  committed. The engine then ends the run on its existing error path
  (`Engine::fail_loop`), which stops the loop, aborts the barrier and calls `Scheduler::shutdown`
  for the work already in flight. It does **not** fall back to the ceiling: with the table inactive
  every price is "not priced", the planner's named fallback would draft at `cfg.n_spec`, and that
  is the implicit baseline the admission rule refuses (docs above, `admit_fallback()`). Drift is a
  stopped cost run, not a new policy and not a silent legacy run.
- **A measurement is filed under the class the planner priced.** `draft_cost_observe` derives the
  shape's `ctx_class` as the deepest frozen row's, exactly `DraftPlanner::shape_of`; a constant
  (class 0, or the plan's per-row field) would file a long-context sample where no class-1 estimate
  can look it up.
- **An inactive table and an absorbed sample are counted apart.** `DraftCostTable::observe` on a
  table that was never configured (or was cleared) increments `inactive()` -- surfaced as
  `SchedMetrics::draft_obs_inactive`, carried to the server as
  `ServerMetrics::draft_cost_observations_inactive` and exported as
  `radiance:draft_cost_observations_inactive_total` -- and returns `RAD_OK`; it is neither a
  refusal (the sample was well formed) nor an absorption, so "nothing was learned" is never
  reported as "nothing was offered". Its unit is the cost observation, one per complete,
  attributable cycle handed to an inactive table: not a row and not a cycle.
- **The host backend cannot exercise the tape branches.** Only `core/device/aql.cpp` implements
  `tape_begin`; on the host every pass is unkeyed, so a host regression can prove the step binding
  of the frozen snapshot, the placement bit and the aggregation rule (`RuntimeTest` checks
  `Ctx::merge_tape` over every class pair; `engine_step_test` injects the notes a rank thread would
  file (the peer calls `Rank::note_evidence` directly; no rank thread, device or `Ctx` pass runs)
  into an evidence slot (a draft pass's `pass_tape`, a repeated note, the next step, a
  shared slot and a second rank) and reads them back through `Engine::step_evidence`, so the slot
  merge, the alias reset and the cross-rank read are covered without a tape transition;
  `engine_loop_move_test` checks that a move injected into
  a sampler-only release is filed under the step that release names, on a drafter-free fixture
  where that release is the step's last one), not a real Play/Record/Audit transition. Those are
  covered by the `Ctx` contract and by fixtures that hand `step_finished` a replay state directly;
  a genuine tape run remains the missing proof that a Record/Audit draft pass reaches the bucket
  rule above.
- **The last cycle is delivered by the next `step()` or `promote_next()`, not by an extra idle
  close.** Every `Scheduler::step()` and `promote_next()` opens by closing the current record
  (`draft_cost_cycle()`), and `Engine::step_loop` calls one of them before its idle branch -- so a
  fully reported cycle is delivered on the idle step itself, and a record no later open can finish
  is counted as dropped there. There is no separate idle flush to rely on. `drain_completed()` on
  that path ends the finished requests' sampler state; it recycles no slot, which
  `Scheduler::reap()` frees later (`sweep_deferred()` when the free was deferred). A cycle measured
  by the truly last step of a run has no step after it, so `shutdown()`/`fini()` flush the ring
  before the requests' identities are released.
- **No fabricated success.** An unimplemented mode returns `RAD_E_UNSUPPORTED`; an unpriced shape
  is a named fallback (`DraftFallback::NoProfile`/`UnknownShape`) with the fixed baseline still
  visible in the plan. A hook never invents a plan or a price to look successful.

`tests/draft_fakes.h` holds the compiling fakes: a flat predictor, a cost model that is invalid
until given measured milliseconds, a policy that prices its fixed baseline and reports the
fallback, and an observer that refuses an incomplete cycle. `tests/draft_types_test.cpp` is the
executable form of the invariants above.

## File ownership

| Strand | Owns | First deliverable |
|--------|------|-------------------|
| Integrator | `core/engine.cpp`, `core/engine_priv.h`, `core/rad_core.h`, `core/config.cpp`, `core/sched/metrics.h`, server metrics/adapters, build files, `spec.md`, docs, shared contracts, telemetry wiring | P00; then telemetry/profile reader (inactive) and the shared commits |
| P00 (this change) | `core/sched/draft_types.h`, `tests/draft_fakes.h`, `tests/draft_types_test.cpp`, `docs/ADAPTIVE-MTP.md`, the `RAD_SCHED_MAX_SPEC` move in `core/sched/draft.h` | contract basis, fakes, tests |
| A — cost model | new `core/sched/draft_cost.{h,cpp}`, `tests/draft_cost_test.cpp` | profile validation/lookup/uncertainty/sample replay on synthetic data |
| B — predictor | new `core/sched/draft_predictor.{h,cpp}`, `tests/draft_predictor_test.cpp` | observation mask, EMA, uncertainty, offline replay |
| C — policy/scheduler | `core/sched/draft.{h,cpp}`, `scheduler.{h,cpp}`, `batch.{h,cpp}`; new `draft_policy.*`, `tests/draft_policy_test.cpp` | K0/immutable lifecycle/cost choice against A+B fakes |
| D — runtime/replay | `core/runtime/ctx.{h,cpp}`, `core/device/aql.cpp` when needed; `tests/draft_replay_test.cpp` | pass counters, replay coverage, no new host wait |
| E — draft execution | `core/engine_draft.cpp`, `sched/advance.{h,cpp,hip}`, `advance_impl.h`, Flash-Next MTP files; `tests/draft_rounds_test.cpp` | publishing, shrinking rounds, prefix rowmaps |
| F — correctness/benchmark | `tests/adaptive_mtp_test.cpp`, `scripts/mtp_matrix.sh`, a C++ SSE client under `tools/` if needed, fixtures/result format | deterministic fixtures, state matrix, harness |

Changes to another strand's files go back as an interface request to its owner; no large mixed
patch. New sources are picked up by the existing `CONFIGURE_DEPENDS` globs, and a new
`*_test.cpp` is a new ctest entry.

## CLI names

The window, policy and profile flags exist today and keep their meaning. `--draft-depth-tiers` is
accepted now that the cost choice reads it. `--draft-policy-shadow` and `--draft-explore-rate` are
**defined here but not accepted yet**: Welle 2 adds them when they have an effect. A flag that
parses and does nothing would be the fabricated success the contracts forbid.

| Flag | Values | Default | State |
|------|--------|---------|-------|
| `--num-speculative-tokens` | N / `auto` | `auto` | existing; 0 = no drafter process |
| `--draft-window` | `fixed` / `adaptive` | `fixed` | existing |
| `--draft-policy` | `legacy` / `cost` | `legacy` | accepted; `cost` needs a serial head, a positive ceiling and, for this runtime's fingerprint, an active profile with a measured bucket or an explicitly admitted fallback baseline (refused with `RAD_E_UNSUPPORTED` otherwise). B's predictor runs every cycle; with a matched profile A's table prices the shapes, otherwise every cycle is the named fallback at the configured depth |
| `--draft-policy-shadow` | switch | off | cost choice logged, not applied; with C |
| `--draft-profile` | path | none | accepted; read once before `Scheduler::init`. An unreadable or malformed file aborts startup; a fingerprint or execution mismatch disables the profile and `reason()` names the first difference. On a generator or embedding model the flag is a start error, because that path never initialises the scheduler that reads it |
| `--draft-depth-tiers` | ascending unique K list, e.g. `0,2,3,4,6` | `0,2,3,4,6` | accepted; the K values the cost choice weighs, at most eight. Each must be ≤ the ceiling (auto resolves it, so the range check is `DraftPlanner::configure`'s) and a list is never silently trimmed. A cost deployment whose ceiling is below a tier is refused by name |
| `--draft-explore-rate` | R in [0, 1] | `0.01` | at most 1% of eligible decode cycles |

`cost` is accepted only with a serial drafter, a valid ceiling contract and, for the runtime's
fingerprint, an active profile that has a measured bucket or an explicitly admitted fallback
baseline; unsupported combinations are refused by name, never silently mapped to `legacy`. Block
drafters are unchanged.
The K0 rules are in the plan: K0 with the drafter loaded stays a temporary operating state with
weights, buffers and arena geometry unchanged (history for the MTP KV/QSA keeps being maintained),
and it is measured separately from a process started with `--num-speculative-tokens 0` and no
drafter. Initialisation is explicitly not depth 0. A speculation-off plan is never filled by the
n-gram fallback: its `fill[]` is `None` for every request and `draft_plan_bars_fill` says so;
"MTP off, n-gram allowed" would be a separate state with its own costs and labels.

## Draft profile format, version 3

`core/sched/draft_cost.cpp` (A) reads and validates this; the parser never runs on the step path.
The format id is `rad-draft-profile/3` (`RAD_DRAFT_PROFILE_VERSION`). A file with another id is
refused by name, not reinterpreted under the current keys: `rad-draft-profile/1` had no draft
distribution, origin or pipeline depth, and `rad-draft-profile/2` had no resolved placement or
device equipment in its fingerprint, so it cannot be bound to a deployment. Older profiles are
regenerated, never rescaled.

```json
{
  "format": "rad-draft-profile/3",
  "fingerprint": {
    "model": "Qwen3.8-Flash-Next",
    "quant": "gguf-q4_k_m",
    "tp": 2,
    "tp_wire": "wht6",
    "gpus": "2x Radeon AI PRO R9700 (gfx1201) dev=0 cu=96 vram=25757220864",
    "kernels": "libr4d <commit>+gfx1201",
    "kv_format": "bf16",
    "placement": "all_vram vram=17179869184 host=0 file=0 dyn=1 res=9f2c41ab7e0d3355 flex=12582912 lend=98304"
  },
  "execution": "shrinking",
  "buckets": [
    {
      "ctx_class": 0,
      "shape": {
        "n_rows": 8,
        "verify_k": 3,
        "verify_tokens": 32,
        "history_tokens": 0,
        "prefill_tokens": 0,
        "pipeline_depth": 0,
        "rows": [0, 8, 8, 8],
        "draft_k": 3,
        "draft_rows": [0, 8, 8, 8],
        "origins": ["head", "head", "head", "head",
                    "head", "head", "head", "head"],
        "sampling": "greedy",
        "replay": "play",
        "drafter": "serial"
      },
      "extent": { "rounds": 3, "draft_rows_total": 24,
                  "draft_round_rows": [0, 8, 8, 8] },
      "time_ms": { "mean": 43.0, "unc": 0.6, "n": 5 },
      "components_ms": { "target": 25.9, "draft": 12.1, "host": 4.2, "valid": true },
      "flags": ["ahead"]
    }
  ]
}
```

- **Fingerprint.** A profile is valid only when model, quantisation, TP wire and rank count, GPUs,
  kernel/plugin stamp and KV format/placement match. The `kernels` stamp is the loaded libraries'
  `name@version`, the arch plugin's and the device arch; the engine carries no build id of its own,
  so a rebuild that changes no plugin version is not distinguished. `configure()` does not fail on
  a mismatch: it leaves the table inactive and names the first differing field in `reason()`. That
  is a visible fallback on a running `legacy` deployment, not a start abort -- `Cost` was never
  admitted. With
  `--draft-policy cost` the same mismatch is a start abort: `Scheduler::init` refuses by name (see
  below), so a Cost deployment never runs on the fixed baseline because its profile silently did
  not match.
  The runtime fields are derived from the deployment, never typed by the user: `model` is the
  container's name (its arch id when unnamed), `quant` its quantisation (`bf16` when none), `tp`
  the rank count, `tp_wire` what `engine_info` reports for `--tp-wire` (`exact`/`wht6`, `n/a` at
  TP1), `gpus` the ranks' devices in order -- product name, arch id and the equipment a same-named
  SKU can differ in (`dev=`, `cu=`, `vram=`; `Nx S` when all are identical, comma-separated
  otherwise), `kernels` the loaded libraries (`name@version`, selection order) plus the arch plugin
  and the device architecture, `kv_format` `--kv-cache-dtype`, and `placement` the **resolved**
  plan, not the flag: per rank in rank order, `strategy vram= host= file= dyn= res= flex= lend=`,
  where `res` is a digest of the plan's tier/site per weight, `flex` the bytes of the carved lent
  slots the start loan covers and `lend` that loan in tokens. The last two bind the start
  placement: a dynamic plan's fill promotes pooled units into exactly those slots, so two runs
  that lend differently reach different placements on the same plan. The loan is settled before
  serving's first step, so both fields are facts of the start, not a snapshot of something still
  moving. `auto` resolves against a budget that depends on
  the free VRAM at this start, so two runs that asked for `auto` can land on different strategies
  and different expert residency; the flag value alone would let both load the same profile. If a
  rank has no resolved plan (still `auto`, or no weights), the runtime is not derivable and the
  reader refuses the start instead of matching against a guess. `execution` is the backend's round
  model: `shrinking` for the serial head that runs one round per row (also in a cycle whose K are
  uniform), `rectangular` for a block drafter, a disabled drafter or drafting switched off. The
  reader derives it and the `--draft-profile-out` writer derives it from the same function, so
  read and write cannot drift. A `rectangular`-stamped profile for a serial deployment is refused
  by name and must be recalibrated, not restamped.
- **Writing a profile (`--draft-profile-out`).** A calibration run decodes under whichever policy
  the deployment really runs -- `legacy`, `adaptive` or `cost` -- with the recording armed. Under
  legacy/adaptive the controller's plan is never set, so `draft_cap()`/`bars_fill()` and the rest of
  the run are exactly what they are without the flag. Under `cost` the table is put in record-only
  mode: the run opens the same cycles any cost deployment opens -- the start cycle included, which
  is the one no legacy run can execute, because it verifies a window it did not draft -- issues the
  admitted fallback baseline, and prices nothing from the buckets it is still filling. That is the
  mode that measures the shape a later cost deployment prices first (`verify_k = 0`); without it a
  reload can price no request's first cycle. The depth a calibration measures is its ceiling, so a
  calibration below the default ladder passes the tiers it fits (`--draft-depth-tiers 0,2` at a
  ceiling of 2); the reload may then run a deeper ceiling and choose the measured tier by its price.
  Each cycle's record names the depth the step really
  caps each row at (the head's own depth, the adaptive window, 0 for a row that may not speculate),
  so a bucket describes a shape the run could execute. A cycle with more decode rows than
  `RAD_SCHED_MAX_SPEC` (64) is not recorded at all: naming a prefix would describe a shape shorter
  than the bracket `step_finished()` times. It is counted (`Scheduler::draft_cycles_dropped()`)
  and warned once, and the table learns nothing from it. When the engine exits -- and only when the
  step loop did not abort -- the ring is flushed and the measured buckets are written with the same
  fingerprint and execution the reader derives. The identity is derived at startup, before the
  scheduler runs: an incomplete fingerprint refuses the calibration up front instead of failing
  after a complete run. It also refuses `--draft-profile` (a calibration does not read its own
  input), a deployment without a serial draft head and a positive ceiling (the file's drafter
  dimension can only name that head), a generator or embedding deployment (no scheduler, so no
  cycle would ever be recorded) and a path that already exists. The file is written to a
  temporary in the same directory with `O_CREAT|O_EXCL`, fsynced, published with `link()` (so it
  appears whole or not at all and an existing file is never replaced) and its directory is
  synced. A `write()` that returns 0 is an error, not progress: the loop stops, the temporary is
  unlinked and the export reports `RAD_E_IO` with "the write made no progress". A run that
  measured no bucket is refused by the writer and never published: an empty
  file would read as a calibration of nothing. Nor is a file of diagnostics a calibration: the
  export refuses when no kept bucket is priceable by the same `draft_bucket_usable` rule
  `estimate()`/`has_usable_bucket()` apply, rather than publishing a profile that prices
  nothing. A bucket whose origin set was not captured, or
  whose rows do not fit its drafter, is not written at all: the serializer refuses rather than
  inventing `head`. The origin set follows the drafter that filled each row -- the lookup fallback
  fills rows the serial head left empty, so a serial bucket may carry `lookup` rows; a `head` row
  under a lookup-only bucket is still refused. A row holds the *set* of origins its window came
  from, not one: two cycles of one bucket can disagree (the head drops a row in one, the fallback
  fills it in the next) with the same executed distribution, and the file names both joined with
  `+` (`head+lookup`) instead of picking a sample or discarding the export. Diagnostic buckets
  whose flags the file cannot name (truncated and unpriced ones) are left out and counted in the
  log. A bucket measured while the mover was moving a unit (`kDraftSamplePlacement`) is left out
  as well: the fingerprint names the plan resolved at start, the file has no field that could
  name the residency the mover drifted to, and writing that bracket anyway would present it as a
  price of the start placement. The same flag marks a bracket measured before the start placement
  resolved, so a calibration whose fill had not landed when it measured writes nothing for those
  cycles either -- the exclusion follows the record's `placed`, not the reason it was set. Only
  the affected buckets go -- a clean cycle before the first
  move still measured the plan and is exported -- and a run whose every bucket was placement-
  affected refuses the export instead of publishing a file that prices nothing.
- **No cost policy without an admissible baseline.** `Scheduler::init` refuses
  `draft_policy = Cost` unless the cost table has a baseline for THIS runtime: `active()` (i.e.
  `configure()` matched a profile to this runtime's fingerprint) **and** a price it can actually
  quote -- at least one bucket usable by the same predicate `estimate()` filters with
  (`DraftCostTable::has_usable_bucket()`: no disturbing flag, a positive sample count, a finite
  positive mean), or `DraftCostTable::admit_fallback()`. A non-empty `buckets()` list is not that
  price: a profile of only warmup/record/audit/admission/placement records passes it and still
  prices nothing. The disturbance set is wider than what a profile file can name:
  `kDisturbing` (`core/sched/draft_cost.cpp`) also carries `kDraftSampleShortened` and
  `kDraftBucketUnpriced`, the runtime-only records of a cycle cut short or of a bracket whose
  deepest round never ran, so those never price either -- they are written with their flag and
  stay diagnostic. `active()` alone is
  not the admission: a matching profile with no bucket proves the reader ran, not that anyone
  accepted pricing every shape with the policy's named fixed baseline. The deployment must say so,
  in so many words, before `init`; otherwise `init` returns `RAD_E_UNSUPPORTED` with
  `no measured bucket` (or the profile's `reason()` when the table was never active), so a
  forgotten admission cannot look like a loaded baseline. The drafter shape alone (a serial head, a
  positive ceiling) is not a baseline and never was. A pre-configured table therefore survives
  `init` instead of being cleared; only an inactive one is cleared, and `clear()` drops the
  admission with the rest of the table. Configuring the table after `init` is not a bootstrap: the
  policy the scheduler admitted was already chosen under a fallback. The startup reader now fills
  the runtime fingerprint from the deployment; `admit_fallback()` is reachable from an embedding
  caller, not from a CLI flag, so a matched profile with no usable bucket is refused rather than
  admitted until such a flag exists.
- **`execution`.** The backend's round model: `shrinking` is a drafter that runs rounds per row
  (the serial head, `depth > 0`), so N_j falls where row depths differ -- and the profile stays
  `shrinking` in a cycle whose K are uniform; `rectangular` is a backend without per-row rounds
  (a block drafter's pass over `block` positions), a disabled drafter, or drafting off. The two
  run different kernels and times, so they are different profiles. The executed distribution
  itself is part of the bucket key (`rounds`, `draft_round_rows[]`, and the planned
  `draft_k`/`draft_rows[]` it is bounded by), so a uniform bucket never matches a query with a
  different distribution. Changing the backend invalidates the old file rather than rescaling it.
- **`shape` is the lookup key.** It is the verify token count and length distribution
  (`n_rows`, `verify_k`, `verify_tokens`), context class, history tokens, prefill tokens, `N_j`,
  sampling class, replay state and the loaded drafter. The planned draft distribution
  (`draft_k`, `draft_rows`) and the pipeline depth are part of the key too: the same `rounds` and
  total rows can shrink differently, and an issued-ahead cycle prices a different pair of
  distributions than a synchronous one. Origins name the mix of rows (`head`, `lookup`, or a
  `+`-joined set when the bucket's cycles differed); a head row does not price a lookup row.
  `T(B, maxK)` alone is not a key, and K0 with the head loaded is
  not the same bucket as K0 with no drafter process.
- **`replay` is stored by name, never by enum number.** `DraftReplay` carries `Unknown` at 0 in
  front of `None`, so a reader that cast the file's index would read a measured `none` as no
  evidence. The four file names (`none`, `play`, `record`, `audit`) are mapped one by one, and
  `unknown` is not a file value: a profile is a measurement.
- **Version 2 is not version 1 with new fields.** A version-1 file has no `draft_k`/`draft_rows`,
  no `pipeline_depth` and no origin mix, so its buckets cannot be mapped onto the current key by
  assuming zeros. The reader refuses it by format id; a profile is regenerated, never rescaled.
  The parser also refuses a bucket whose `draft_rows` disagrees with `rows` where the two
  distributions are the same execution, and a shape carrying an `unknown` dimension.
- **`ctx_class`.** 0 = below 4K, 1 = 4K–32K, 2 = 32K–128K, 3 = 128K and above
  (`draft_ctx_class`). Profiles only carry the classes the configured capacity can reach.
- **Finite table, exact keys.** The table has a fixed capacity and neither grows nor evicts: a shape
  that finds no room is dropped and counted, which the policy sees as a missing bucket and prices
  through the fallback, never as a fabricated number. Admission may be made stricter to protect
  those finite slots from one-off shapes, but only by refusing to admit a shape -- never by
  coarsening the key. `ctx_class` buckets the context length alone; `history_tokens` and
  `prefill_tokens` are compared exactly, as `anchor_class_matches` now does (A, `2c1a01d`). Two
  prompts carrying a different amount of history or prefill did a different amount of work, and
  filing them as one bucket makes one of them a bound on another measurement. Prefill work is the
  tokens actually processed, not the prompt's cap.
- **Coverage and uncertainty.** Each bucket carries its sample count and uncertainty. A missing
  bucket is priced by one documented interpolation step (a neighbouring context class or the
  documented N_j scaling) with `DraftCostSource::Interpolated` and the propagated uncertainty;
  otherwise it is a conservative bound with `coverage = 0`, never a unitless prior as ms.
- **Attribution.** Buckets are built only from samples `draft_sample_attributable` accepts.
  Recording, audit, warmup, admission and placement/DMA brackets are stored separately and never
  folded into the play bucket. Real DMA cost stays in the record with its flag.
- **What is optimised.** The measured amortised batch time at the same pipeline depth is the
  comparison number; target, history, draft and host components are for diagnosis and calibration
  and are not summed under overlap.

The measurement axes (C1/2/4/8/16 as memory allows, fixed K0..6, K0 with and without the drafter,
4K/32K/128K contexts within capacity, the homogeneous and mixed windows at least `[6,2]`,
`[6,2,0,0]` and a C8 mix, warm replay/record/audit/placement/DMA separated, decode-only versus
decode+prefill) are the manifest's business, not the reader's; the reader only accepts buckets
that name their axes.

## Benchmark manifest, version 1

`scripts/mtp_matrix.sh` (F) takes one immutable manifest. The format id is
`rad-bench-manifest/1`.

```json
{
  "format": "rad-bench-manifest/1",
  "id": "2026-10-22-mtp-wave1",
  "lock": "/tmp/radiance-tp2-bench.lock",
  "model": { "path": "/models/qwen3.8-flash-next.rad" },
  "server_args": ["--tp", "2", "--max-num-seqs", "16"],
  "arms": [
    { "name": "fixed0", "args": ["--num-speculative-tokens", "0"] },
    { "name": "fixed3", "args": ["--num-speculative-tokens", "3", "--draft-window", "fixed"] },
    { "name": "legacy5", "args": ["--num-speculative-tokens", "5", "--draft-window", "adaptive"] },
    { "name": "cost6", "enabled": false,
      "args": ["--num-speculative-tokens", "6", "--draft-window", "adaptive",
               "--draft-policy", "cost", "--draft-profile", "profiles/r9700-tp2-flashnext.json"] }
  ],
  "workloads": [
    { "name": "code", "prompts": "manifests/prompts/code.txt" },
    { "name": "prose", "prompts": "manifests/prompts/prose.txt" },
    { "name": "reasoning", "prompts": "manifests/prompts/reasoning.txt" },
    { "name": "json", "prompts": "manifests/prompts/json.txt" }
  ],
  "concurrency": [1, 2, 4, 8, 16],
  "repetitions": 5,
  "order": "alternate",
  "warmup_requests": 2,
  "outputs": "runs/2026-10-22-mtp-wave1"
}
```

- A manifest is immutable once used. An arm, workload, prompt set or repetition change is a new
  file with a new `id`; a typo fix before the first run is a new file too.
- Tuning, calibration and the final holdout are separate manifests. The best fixed depth is chosen
  on tuning data; holdouts are not re-tuned.
- `concurrency: 16` that cannot run is recorded as not run (`C16 OOM`), not relabelled as a smaller
  concurrency. A single card is not a representative TP2 cost basis; a second TP2 pair may measure
  independently.
- Every GPU consumer takes the same exclusive `flock` on the whole TP2 pair; CTest's
  `RESOURCE_LOCK` is only within one run. A productive server in the manifest is accounted for or
  released in order, never stopped by a blanket kill.
- `order` alternates or randomises the arms; repetitions are paired and at least five for decisive
  comparisons. Warmup and cache state are recorded.

## Verifying P00 on the host

**Compiler.** Every build and compiled test — P00 and strands A–F — uses ROCm's clang 23: C at
`/opt/rocm/llvm/bin/clang`, C++ at `/opt/rocm/llvm/bin/clang++`, HIP through
`${ROCM_PATH:-/opt/rocm}/bin/hipcc`, which drives the same clang. The two `llvm/bin` spellings are
not two toolchains: on the delivered ROCm 10.0.0 image `/opt/rocm/llvm/bin/{clang,clang++}` and
`/opt/rocm/lib/llvm/bin/{clang,clang++}` all resolve to
`/opt/rocm/core-10.0/lib/llvm/bin/clang-23` (one inode, `2103776`; sha256
`241bf4da7ec39bc00b68ed74f6be751516d9892ed990e8c7fa372bad18500247`), and `hipcc --version` names
the same `AMD clang version 23.0.0git (8f497e0…)`. ROCm clang 23 is the mandate; no ROCm 10.1
requirement follows from it. A system compiler is not a substitute; if the toolchain is missing,
the blocker is reported rather than worked around. The entry points name it: `./build.sh` passes
both compiler paths itself and refuses to configure without them, and the `default` CMake preset
sets the same pair. Name it explicitly, in a build directory of its own so a differently compiled
tree cannot be mistaken for this one:

```bash
/opt/rocm/llvm/bin/clang++ --version   # the proof: path and version beside the result

./build.sh -b build-mtp-clang --no-hip   # configure + build + ctest -L fast
ctest --test-dir build-mtp-clang --output-on-failure -LE 'gpu|oot' -j 8
```

**Build prerequisite, separate from P00.** The clang 23 prerequisite lives on branch
`build/clang23-libavx` (from `cb5a69a`) as three separately reviewable commits; this tree already
carries two of them by content:

| Commit | What | Here |
|--------|------|------|
| `dcb4e6f` | libavx: a duplicated `target` attribute, and a bf16 intrinsic in an OpenMP-outlined loop | present (`733e6f7`, `29e21bb`) |
| `c04006e` | `-Wl,--build-id`: kbench names a plugin by its ELF build-id note, which clang does not add itself | deliberately absent (`b6a9f1c`) |
| `1ef93d2` | `core/device/aql.cpp`: prefer ROCm's `kfd_ioctl.h` (1.16) over the host kernel's 1.11 | present (`f271478`) |

The two libavx hunks and the aql include are here byte for byte (`git diff 1ef93d2 HEAD -- libavx
core/device/aql.cpp` is empty), so clang 23's two compile breaks are fixed in this tree. The
`--build-id` line is not, and P00 stays contract-only: compile and link do not depend on it, but a
clang-linked library then carries no ELF build-id, `kb::code_identity()` names it 0 and caches
nothing (`tools/kbench_core.cpp`), and `kbench_test`'s `the_code_identity_follows_build_ids` check
(`build_id_of(...).size() >= 16`) cannot pass. Take the branch where a clang-linked tree must be
cachable. This table is a source comparison, not a build result: no device build has been run
against this tree.

That branch is reconciled with the `eval-main` handoff, not a parallel repair variant:
`git diff dcb4e6f <eval-main> -- libavx` is empty, so the two libavx commits carry exactly
eval-main's `8d46163` and `0bdecad`, and `1ef93d2` is eval-main's `ce42e4e` verbatim. The aql change
is visible only in a HIP build (`RAD_WITH_HIP=OFF` compiles the translation unit out), so the host
results below do not exercise it.

**Result (2026-10-22).** Measured with `/opt/rocm/llvm/bin/clang{,++}` (AMD clang 23.0.0git, ROCm
llvm-project `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, installed as ROCm 10.0.0), `-DRAD_WITH_HIP=OFF`,
Ninja and `RelWithDebInfo` (`-O2 -g -DNDEBUG`) on both sides — P00 plus the prerequisite above, and
`cb5a69a` plus the same prerequisite:

- The targeted suites `draft_types_test`, `sched_test`, `sample_test`, `runtime_test`, `config_test`
  pass 5/5 on the working tree committed at 02:14:35 as `8f54c57` (docs at 02:15:09 as `7a9cbd0`);
  `draft_types_test` reports 364 checks over 29 cases, built at 02:14:06 and run at 02:14:28, after a
  first 364/29 run of the same change at 02:13:09 and under ASan at 02:13:32. A later binary, built
  at 02:17:12, ran `draft_types_test` at 02:17:27 (367/29) and the other four suites at
  02:17:53-59, and was committed only afterwards, at 02:18:02, as `e0a053c`. Both binaries therefore
  ran on a tree that was still uncommitted, before the commit they are named by existed; the close
  times prove nothing about tree identity, and no rebuild was done to settle it. They are reported
  as runs on an unverified working tree — neither "`e0a053c` built" nor "that exact commit
  checked". `9bc94cd`, `8c91876`, `93d8797`,
  `28cb9da` and `f77f93a` add checks to `draft_types_test` and `sched_test` and are **unexecuted**.
  Every run here started after the 02:01:24 CEST pause order; nothing has run since 02:17:59. No
  test target here depends on `libavx` (`ninja -t inputs`), so the older libavx objects are not part
  of these binaries.
- The broad host suite (`-LE 'gpu|oot'`, 104 tests) passes 97/104 on P00. The same suite on the
  baseline tree registers 103 tests (`draft_types_test` is P00's) and passes 98/103. Exactly the
  five failures below fail on both.
- The two further broad-run failures on P00, `glm5next_test` (10.49 s CPU) and `quant_ggml_test`
  (25.84 s CPU), pass individually on the same tree (7.83 s and 6.01 s) and passed in the baseline
  broad run: they cross a wall-clock budget under the eight-way parallel run, not a check.

The five, each run alone (`ctest -R '^<name>$' -j1`) with the same invocation on both trees:

| Failure | P00 + prerequisite | `cb5a69a` + prerequisite |
|---------|--------------------|--------------------------|
| `gbnf_test_shard2` | 41715 checks over 7 cases in 20.94 s; `FAIL the binary spent 20.94 s of CPU, over its budget of 10 s (RAD_TEST_CPU_BUDGET)` | 41715 checks, 22.30 s CPU, same budget failure |
| `gbnf_test_shard3` | `FAIL json_schema_walks_write_only_valid_instances`, `tests/gbnf_test.cpp:1857: CHECK(ok)`, `seed 69 wrote {"n":\x0a 91,"s":"bion\xf0\x85\xaa\x8d","t":\x0a[] }`; 1 failure | identical failure, same seed and output |
| `images_test` | 9 failures, all `500 server_error: the picture could not be encoded: this build has no media decoder … (RAD_WITH_FFMPEG)` | same 9 failures |
| `tokenizer_test` | 907 checks over 23 cases in 25.98 s; `FAIL the binary spent 26.04 s of CPU, over its budget of 20 s` | 907 checks, 30.89 s CPU, same budget failure |
| `videos_test` | 8 failures: `tests/videos_test.cpp:238` `status` `"failed"` vs `"completed"`, `:239` `seconds` `"auto"` vs `"0.12"` | same 8 failures |

None of the five touches `core/sched/`; the budget overruns above and the two load cases are this
host's speed, not a check result. The earlier 99/104 and 98/104 figures were taken with
`CMAKE_BUILD_TYPE` unset (an unoptimised build) and are superseded by the RelWithDebInfo runs
above.

The test binary is picked up by the existing `*_test.cpp` glob; no CMake edit is needed.
