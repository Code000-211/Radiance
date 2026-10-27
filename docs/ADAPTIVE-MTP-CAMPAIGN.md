# Adaptive MTP: the #59 campaign plan

**Status: a plan, not a result.** This document records what the campaign of issue #59 can and
cannot express today, and pre-registers the decision rules for its measurements. Nothing in it was
measured on a card: the host this was written on has no `/dev/kfd`, no ROCm, no model container, no
BetterBench and no profile artifacts. The acceptance of #59 — the BetterBench median at ≥ 1.20x
fixed K3 in every class C1/C2/C4/C6/C8 — is settled only by the runs §3 prescribes, on the GPU
host, under its exclusive coordination contract. Until they run, the 20 % question is open, and
this document claims no step towards it except the preparation.

Two trees are named, and every `file:line` below was re-validated against the tree it names:

| tree | SHA | what it is |
|------|-----|------------|
| `main` | `39f957d` | `origin/main` at the time of writing |
| `strand` | `b1dbcf8` | `origin/integration/fallback-k-on-main`, the head of open PR #50: the adaptive-MTP strand integrated onto `main` |

## 1. What the campaign can stand on, and where it stands

#59 warns that the merged state of an original strand says nothing about presence in the
integration tree. This section is that record: each contract of the campaign, and the exact place
it lives. "External" means: not in this repository, not reachable from a checkout, and therefore
not auditable from a development host — only from the GPU deployment.

| #59 contract | where | evidence |
|---|---|---|
| fixed K3 comparator: legacy policy, fixed window, depth three | `main` | `--num-speculative-tokens 3`; the window default is `fixed` (`core/rad_core.h:147`), the depth default is `auto` (`core/config.cpp:154`). Not the tuning-selected best fixed depth — the start line names 3 |
| the documented rollback to fixed K3 | `main` + `strand` | the rollback is the K3 start line itself; on the `strand` the policy switch and the flags each policy refuses are written down (`docs/ADAPTIVE-MTP.md`, "The deployment start line") |
| native serial MTP, explicitly enabled coupled mode | `main` | `RADIANCE_MTP_COUPLED=1` (`core/engine_draft.cpp:62`) requires the linear serial chain and refuses a head that declared no coupled planes (`core/engine_draft.cpp:60-77`); it cannot serve uncoupled drafts silently (`README.md:14`) |
| coupled acceptance contract, RNG semantics, censoring | `main` | acceptance by agreement in `core/sample/accept.h`; the stateless counter-based RNG in `abi/rad_philox.h`; censor reasons in the draft observation (`spec.md` §10, §13) |
| the adaptive window (`--draft-window adaptive`) | `main` | per-sequence acceptance stop and step-time-driven head depth (`core/sched/draft.h:15-29`, `DraftAdapt` at `core/sched/draft.h:50`). Off by default; it is not the campaign's policy, but it is the fallback the cost policy must beat |
| the cost seam: predictor / estimate / choose / observe | `strand` | four hooks over caller-owned memory (`core/sched/draft_types.h`; `choose` at `core/sched/draft_policy.h:159`), wired into the scheduler, allocation-free |
| cost policy `--draft-policy cost`, opt-in, default `legacy` | `strand` | the default stays legacy (`core/config.cpp:49` on the `strand`); `cost` becomes a deployment's standard only by being written on its own start line, and a matched profile with no usable bucket is refused at start rather than admitted (`docs/ADAPTIVE-MTP.md`, "The deployment start line") |
| profile binding (model, quant, TP wire, GPUs, kernels, KV format, resolved placement, basis, arena) | `strand` | `rad-draft-profile/4` (`core/sched/draft_cost.h:5`), format gate at `core/sched/draft_cost.cpp:654`; the fingerprint is the "Fingerprint" entry of `docs/ADAPTIVE-MTP.md` §"Draft profile format, version 4" |
| the named K3 fallback for unpriced shapes | `strand` | `DraftFallback::UnknownShape` (`core/sched/draft_types.h:104`); `--draft-fallback-k` defaults to the ladder's middle tier, K3 of the default ladder `0,2,3,4,6` (`core/rad_core.h:928`, `core/sched/scheduler.cpp:414`) |
| calibration: forced rotation, no-loan reference, record-only export | `strand` | `--draft-calib-k` (the `3,2,2,3` rotation, `docs/ADAPTIVE-MTP.md` §"The calibration sequence (`--draft-calib-k`)"), `--draft-reference`, `--draft-profile-out` (`core/config.cpp:187-210` on the `strand`); warmup/record/audit/admission brackets are stored apart from priced samples (`kDraftSampleWarmup`, `core/sched/draft_types.h:232`) |
| per-class coverage, fallback shares, priced/unpriced ledgers | `strand` | the `/stats` `draft_cost` block: cycles, priced, fallback, empty, no-plan always summing to cycles (`core/sched/metrics.h:173-174`), plus the per-request ledger line |
| replay/telemetry, issue-ahead overlap, K0 gate | `strand` | pass counters and step-evidence snapshots; K0 with the drafter loaded is a measured state, not "no drafter" (`docs/ADAPTIVE-MTP.md` §"CLI names") |
| the paired-arm harness | partly | the manifest format `rad-bench-manifest/1` is specified (`docs/ADAPTIVE-MTP.md` §"Benchmark manifest, version 1") — arms, alternating order, ≥5 repetitions, one exclusive `flock` per TP2 pair, immutable manifests, tuning/holdout separation. The harness itself (`scripts/mtp_matrix.sh`, strand F's deliverable) is **not in the tree** |
| BetterBench, its Quick tier, its report fields | external | the measurement tool of the deployment owner; its exact median field and aggregation must be frozen before any baseline is interpreted (#59, story 5) |
| the IQ4R deployment (container, TP2 two-card group, patched toolchain) | external | `qwen3.8-next-flash-fp8-iq4r-moe.rad` (`README.md:50`); that family's production line is `--tp 2 --tp-wire wht6 … --num-speculative-tokens 3` (`README.md:218`) — the fixed-K3 comparator is the production start line, not a synthetic one |
| the profile artifacts 0006/0007/0008 | external | 0006 is known from the strand's own record: 47 buckets, every one `n_rows = 1`, the file of the one C8 run that reported −13.8 % — cited there as "a measurement artefact … not in this tree". 0007 stays blocked and 0008 is a failed attempt per #59; neither is in this repository |
| the calibration boundary protocol (16 warmup requests → one boundary POST acknowledged `accepted=true`, `epoch=2`, integer drop counters → 32 scored requests → export) | external | **no endpoint in this tree implements it.** The engine-side counterparts exist — warmup brackets kept apart from priced samples, the export gates (nothing written for a placement-affected or empty set, `O_EXCL` publish) — but the request counting, the boundary acknowledgement and the epoch identity are the external driver's contract. #61 must restate that protocol against the engine's actual seam before Stage 2 runs, or the "boundary/export epoch identity" check has nothing to check |
| the GPU coordination window | external | one exclusive lock over the TP2 pair; a running server is the window's owner. Not in this repository |
| the self-measurement mode (`--diagnose`) | not on `main` | open PR #32; an engine-side seam, not part of #59's acceptance. BetterBench over HTTP is the campaign's serving seam |

### 1.1 What this host verified of the two trees

Both trees configure, build and pass their host suites on a machine with no card, with the system
gcc 14.2.0 (`RAD_WITH_HIP=OFF`, `RelWithDebInfo`, Ninja). The exact commands and counts are in the
PR that carries this document; in summary:

- `main` @ `39f957d`: 632/632 targets, exit 0. Fast tier: 76/85 pass under `-j8`; the host tier:
  83/93. The failures are the host's, not the trees': `images_test`/`videos_test` need ffmpeg
  codecs this build was configured without; `ops_doc` needs GNU `gensub` (only mawk is installed);
  the rest (`chatparse`, `gbnf` shards, `quant_ggml`, `tokenizer`, `trigger`) pass when run alone —
  they cross `RAD_TEST_CPU_BUDGET` only under eight-way load on this 12-core host.
- `strand` @ `b1dbcf8`: 683/683 targets, exit 0. Fast tier: 93/101 under `-j8`, same environment
  failures. Every draft suite passes: `draft_cost` 646 checks / 52 cases, `draft_policy` 1770/43,
  `draft_predictor` 3926/22, `draft_types` 408/31, `draft_rounds` 54/3, `draft_reference` 89/7,
  `sched_draft` 4117/90.

This matters to the campaign for one reason beyond hygiene: #59's testing decisions keep the
tokenizer CPU-budget failures and skipped suites as **explicit open findings**, and a suite run
under load that was never re-run alone is exactly the "silently skipped test" that decision
forbids. The numbers above separate the trees' failures from this host's.

### 1.2 The gap inside the fingerprint

#59 demands pinned image, binary and kernel identities — "a generic image tag or plugin version
alone is insufficient evidence of source identity". The strand's profile fingerprint stamps the
loaded libraries' `name@version`, the arch plugin and the device arch, and **the engine carries no
build id of its own** (`docs/ADAPTIVE-MTP.md`, "Fingerprint"): a rebuild that changes no plugin
version is not distinguished. The campaign must close this itself: every sealed run line (§4)
carries the engine binary's hash beside the git SHA, so a profile's binding and a run's identity
can be compared as files, not as version strings.

## 2. Why this document is a plan and not a result

Every acceptance criterion of #59 and of its stage issues #60–#65 is a measurement on the GPU
deployment. This host cannot run one of them: no card, no container, no BetterBench. Code written
here without a measurement behind it would violate the issue's own rules — "no ineffective changes
kept for ticket fulfilment" (#62), "the spec authorises neither a claim that a suitable cost
profile already exists nor a claim of measured gains" (#59). What can be done from a development
host, this document does: the audit above, the host-side evidence, and the decision rules of §3,
written down before any measurement so that no rule is chosen after seeing its own result.

## 3. The stages, with their decision rules written first

Each stage names its issue, its precondition, its runs, the statistic, and the rule that decides.
The rules are pre-registered: a stage may not reinterpret them after its data exists. Where a rule
needs a quantity only the GPU host can produce, it says so and stops — that is the plan's own
discipline, not an omission.

### Stage 0 — before any run

1. **The measured tree is chosen and pinned.** PR #50 is open; the campaign measures the
   integration tree or its successor, never a moving branch. Every stage pins its tree by SHA and
   its engine binary by hash (§1.2). A rebase or a merge onto the measured tree after Stage 1
   invalidates the baseline: the fingerprint's `kernels` stamp will not catch an engine-only
   rebuild, so the sealed lines must.
2. **The BetterBench freeze.** One sample report is read and the exact median field, its unit, and
   its run-level aggregation are written down before any baseline number is interpreted (#59,
   story 5; #60's first checkbox). C1 single-stream decode and the concurrency-C1 median are named
   as two different fields, and C6's presence in the report is confirmed, not assumed. Until this
   freeze exists, no number from any Quick is comparable to anything.
3. **The effective Quick configuration is written down**, per the issue's Ergänzung: during
   optimisation, C1 + C8 and only the 64K prefill target tier; the targeted class added when it is
   being optimised; single-stream decode included; `--quick` alone is not assumed to reduce tiers —
   the configuration that actually ran is recorded with every run, and actual prompt tokens are
   reported beside the 64K target. The five baseline runs and the final acceptance use all of
   C1/C2/C4/C6/C8 with the full warmup/request parameters.
4. **The boundary protocol is restated against the engine's seam** (§1, the external row): what
   request the driver counts as a completed warmup, what the boundary POST is acknowledged by, and
   where the epoch identity of boundary and export is compared. If the driver's contract cannot be
   checked from the export alone, that is found here, before 16 warmup requests have been spent on
   a recording that cannot be scored.

### Stage 1 — the K3 baseline and the Quick spread (#60)

**Runs.** Five unchanged, complete, error-free BetterBench Quick runs of the fixed-K3 start line —
`--num-speculative-tokens 3`, coupled mode on, C1/C2/C4/C6/C8, one server lifetime per run or as
the deployment's coordination contract requires, same deployment identity for all five, under the
GPU lock. An already-launched Quick counts only if complete, error-free and configuration-
equivalent; otherwise it is retained and marked not-counted, with the difference named.

**Statistics, pre-registered.** Per class: the five medians are all reported; the class baseline
is the median of the five; the observed spread is the range (max − min) and the relative spread
(range / median) of the five. No run is discarded; the fastest run is not the baseline.

**Decision rules, pre-registered.**

- The class baseline of every later comparison is that median, and only that median.
- The noise floor of the Quick screen for a class is its relative spread from this stage. A later
  single-Quick delta within that spread is **inconclusive** — neither a regression nor a gain
  (#59: "an isolated gain within observed noise is inconclusive"). A change is not kept on
  in-spread evidence; it is re-run, or dropped.
- If the five runs of a class do not agree on the median within a factor the campaign owner
  accepts in advance, the Quick is not a usable screen for that class and Stage 1 says so — the
  fix is more runs or a tighter configuration, never a quieter statistic.

### Stage 2 — the valid coupled cost profile (#61)

**Audit first, then measure.** The historical artifacts are audited where they lie (external,
#59's own rule: 0007 blocked, 0008 a failed attempt, 0006 known to price only `n_rows = 1`). An
incompatible profile is neither adopted nor restamped; the binding check the engine already
carries (fingerprint, per-field mismatch naming) is the audit's instrument, and its refusal output
is quoted in the audit record.

**Runs.** Per class C1/C2/C4/C6/C8: one server lifetime, the calibration sequence of the issue —
16 successfully completed warmup requests, the one boundary POST with its epoch and integer drop
counters, 32 successful scored requests, orderly export of the same epoch — with the forced
rotation (`--draft-calib-k 3,2,2,3`, whose four-cycle period plus the period wrap measures
the four transitions of the K3/K2 pair) so neighbouring depths are measured where the choice
will be made, and the no-loan reference recorded once per #59's contract for it. Warmup observations are excluded by the engine's own bracket
(`kDraftSampleWarmup`), and the export's refusal gates (placement-affected, empty, unpriceable
sets) are the recording's quality gate — a refused export is a failed calibration, not a
half-valid profile.

**Statistics, pre-registered.** Per class: the priced-cycle share (`draft_cost_priced /
draft_cost_cycles` from the `/stats` ledger), the fallback share with its named reason
(`UnknownShape` vs `NoProfile`), the bucket count by `n_rows` band, and the coverage of the class's
actual serving shapes (the shapes BetterBench drove, not invented ones — #59's fourteenth story).

**Decision rules, pre-registered.**

- A class whose profile prices only a small share of its cycles runs the named K3 fallback for the
  rest **by design** — that is honest, but it bounds the class's achievable gain by the priced
  minority. Stage 2's product is this coverage table, and the campaign reads it as the first
  measurable fact about where gains can exist. Missing prices are measured next, not interpolated
  away: the engine already refuses to invent them, and so does this plan.
- The rollback is checked here, once, on the card: the same start line with the policy switched
  back serves fixed K3, and the flags the other policy refuses are removed with it.
- The profile binds, or the run does not start. A profile that binds after a restamp is a blocked
  profile wearing a new stamp; the start refusal is the correct outcome, and it is kept in the
  record with the field it named.

### Stage 3 — the first measured policy/algorithm change (#62)

One bounded change at a time, each with its bottleneck, hypothesis and expected effect written
down **before** the change, through the existing interfaces (the four hooks; no parallel
controller). Correctness cases run first (K0 transitions, mixed depths, first rejection and
censoring, generation identity, cancellation, EOG, output and context limits, history-only work,
issue-ahead overlap — the existing draft suites). Then one Quick as the direction check, against
the Stage 1 baseline, per class.

**Decision rule, pre-registered.** The change is kept only if its per-class delta leaves the
Stage 1 noise floor in the intended direction on the classes it targeted, and its correctness
suites pass; otherwise it is reverted and the record says which hypothesis failed. A kept change
is a candidate, not a result: only Stage 6 decides. A change whose effect is within the spread is
not kept for ticket fulfilment — that rule is #62's own acceptance criterion, restated here as
the pre-registered rule it already is.

### Stage 4 — the first kernel bottleneck, end to end (#63)

Relevance in the live path first (the per-op profile and the step records name the kernels the
class actually runs), then KBench before/after at representative shapes against the reference
oracle, then exactly one BetterBench Quick to confirm the end-to-end effect. The microbenchmark
figure and the serving figure are reported separately and never added; a KBench win that the
Quick does not carry is a local win and is recorded as one. If the change is evaluated on the cost
path, it uses the Stage 2 profile, and a changed measurement basis recalibrates first.

### Stage 5 — the integrated candidate (#64)

The kept changes are integrated, the joint state is correctness-checked, and all five classes are
evaluated against fixed K3 with the same paired discipline as Stage 6 but as a screening pass. The
outcome is one of two written artefacts: a documented acceptance candidate handed to Stage 6, or a
plateau report naming what was tried, what failed, and the next bounded experiments as their own
tickets. A plateau report alone does not open Stage 6 (#65's own precondition).

### Stage 6 — acceptance (#65)

**Runs.** At least five paired complete BetterBench comparisons: within each pair the arms see the
same prompts, sampling, and deployment inputs; the arm order alternates; warmup and cache
treatment are controlled; the evaluation data are held out from everything Stage 3–5 tuned on.

**Statistics and decision rules, pre-registered.**

- Per class, per pair: the ratio of the candidate's frozen median field to the K3 arm's. The
  class statistic is the **median of the five pair ratios**; the class passes only when that
  median is ≥ 1.20 and clears it by more than the class's Stage 1 noise floor — a median inside
  its own noise is inconclusive, and the answer is more evidence, not a favourable read. No
  cross-class netting, no averaging a class away (#59, story 24; #65's third checkbox). All
  pairs are reported, including the ones that lose.
- The uncertainty of the class result is reported from the Stage 1 noise floor and the spread of
  the five pair ratios. If the spread prevents a defensible conclusion, more evidence is
  collected; a marginal class is not accepted on a favourable read of a wide spread.
- TTFT and p95-TPOT are reported per class with no fixed rejection threshold — and no errors,
  hangs, starvation or broken lifecycle as the price of throughput (#59, story 23).
- Both TP ranks and coupled execution are verified on the card — a healthcheck and a non-empty
  answer are not state/sampling correctness proof (#59's testing decisions).
- Failure, missing class, or OOM is a failure/not-run finding; it is never a relabelled smaller
  concurrency.
- **The outcome that settles the issue by finding nothing:** if no candidate reaches 1.20 in every
  class, #65 stays open, the plateau report is the deliverable, and nothing about the target is
  quietly lowered. The remaining options the issue names (Tree-MTP, E4, DarTree — issues #66–#68)
  are out of this campaign's scope and are not started by a plateau.

## 4. The rules that hold for every stage

- **Sealed lines.** One line per run in the stage's record: git SHA, engine binary hash, container
  identity, start line verbatim, card clocks, the Quick configuration that actually ran, and the
  counters every figure came from. A figure without its sealed line did not happen.
- **One name per quantity.** The engine already publishes its ledger as `draft_cost_*`; the stage
  records read those names and invent no synonyms. The BetterBench field names are frozen once, in
  Stage 0, and every later document uses them.
- **Foreign numbers stay foreign.** The −13.8 % C8 figure and the 47-bucket profile are evidence
  from measuring runs elsewhere, cited as such, never folded into a local argument. The historical
  Docker results are a historical reference, not a baseline (the issue's Ergänzung).
- **The GPU window is single-tenant.** Every run takes the exclusive coordination lock; a running
  server is accounted as the window's owner or released in order, never killed by a blanket sweep.
- **No fabricated success.** A run that could not measure says so; a skipped suite is listed as
  not run with its reason; the tokenizer CPU-budget findings and the codec-dependent suites stay
  explicit open findings in every stage record, exactly as #59's testing decisions require.
