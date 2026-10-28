# Adaptive MTP: loading the recorded profile — the #61 execution plan

**Status: a plan, not a result.** This document is the Stage-2 execution companion of the #59
campaign plan (`docs/ADAPTIVE-MTP-CAMPAIGN.md`, PR #70): the recording of issue #61 exists — the
deployment owner's record is cited below — and what remains of the issue is the load of that
profile by a serving run, the proof that the engine prices from it, the rollback to fixed K3, the
per-class coverage report, and the measurement of the prices the profile does not carry. All of
that is work on the GPU host that owns the artifacts; this host has no `/dev/kfd`, no ROCm, no
model container, no BetterBench and no profile files, so nothing in §3–§6 was measured here. What
this document adds is what the campaign plan demanded before Stage 2 runs: the engine-side audit
re-validated at the tree the deployment actually runs (§2), and the decision rules for the
remaining runs written down before their data exists (§3–§6), so that no rule is chosen after
seeing its own result.

Two trees are named, and every `file:line` below was re-validated against the tree it names:

| tree | SHA | what it is |
|------|-----|------------|
| `main` | `39f957d` | `origin/main` at the time of writing |
| `strand` | `738dd49` | `origin/feat/draft-calib-boundary`, the deployment line: the calibration boundary, the no-loan reference basis, the load path, their suites, and the subsystem guide `docs/ADAPTIVE-MTP.md`. It supersedes the campaign plan's `strand` `b1dbcf8` (`b1dbcf8` is an ancestor), and the external recorder's artifact line `v6/artifacts-738dd49` names it |

External — not in this repository, not reachable from a checkout, cited as the deployment owner's
records and never folded into a local claim (the campaign plan's "foreign numbers stay foreign"):

- `runs/gh-60/20261026T050130Z-host-fixed3-calibration-r02` — the recorded cells: one owned
  server life per class, native binary `4ff00caa…`, coupled, `--draft-reference` (no-loan basis),
  45 min per cell, no retry; every cell passed `harness/scripts/calibration_verify.py` (16
  finished warmups → exactly one accepted boundary acknowledgement, epoch 2, integer drop counters
  → 32/32 scored requests → SIGTERM → export of the same epoch).
- `calibration/native-reference-profile.json` — the merged profile: `rad-draft-profile/4`,
  execution `shrinking`, 492 buckets, 9393 samples, sha256
  `b16b6f36f6ff561fd4855b24a4b0758dfe36e3c6c0b881ec6b09be15e383e3ab`.
- `harness/v6/rec2/record.py` and its CPU selftest — the external recorder; issue #61's first
  comment records the two trust-boundary fixes in it (boolean drop counters accepted as integers;
  an uncaught `.get` after a rejected non-dict acknowledgement) and the passing selftest run
  `runs/gh-61/20261025T231834Z-recorder-boundary-cpu-selftest-r01`.

## 1. Where issue #61 stands

The issue's five acceptance criteria, each with the record that stands for it today:

1. **Audit the historical profiles for provenance and binding; adopt or restamp neither a locked
   nor an incompatible one.** The historical artifacts stay external and audited where they lie:
   0006 is the 47-bucket, `n_rows = 1`-only artifact the strand's own record calls a measurement
   artefact; 0007 stays blocked; 0008 was a failed attempt (the campaign plan §1 keeps this
   record). No historical profile was adopted or restamped for this recording — the profile of
   the second bullet was recorded fresh. The engine's binding check is the audit's instrument;
   §2 names its refusal outputs and rule D1 fixes how they are read, so the audit record can
   quote them verbatim.
2. **The new recording, per the contract.** Done by the deployment owner per the record above:
   one server lifetime per class, 16 finished warmups, exactly one confirmed boundary POST with
   epoch 2 and integer drop counters, 32 scored requests, ordered export of the same epoch, no
   invented sample cap (the cells exported 4381 / 2213 / 1195 / 900 / 704 samples for
   C1 / C2 / C4 / C6 / C8 — 9393 preserved through the merge). The recorder-side trust boundaries
   were fixed and CPU-tested first (the first comment's record).
3. **Coverage and fallback shares per C1/C2/C4/C6/C8; missing prices measured.** The coverage of
   the recorded profile is known (the owner's honest numbers, restated in §5 because every later
   rule reads them); the per-class *serving* shares are not measured yet — they are the product
   of §5's runs. The missing prices (no `draft_k 0` bucket, context classes 1–3, thin k1/k5) are
   named and their measurement is pre-registered in §6, not interpolated away.
4. **`cost` loads the matching profile; the rollback to legacy with fixed K3 is checked.** Not
   run yet — it is §3 and §4 of this plan. The engine-side machinery it exercises is present and
   pinned by tests at the `strand` (§2), so what is missing is the run on the card, not the seam.
5. **Exclusive GPU coordination for the baseline measurements.** A rule of the deployment, not a
   ticket dependency (the issue's own words); restated in §7 because no run of this plan may
   bypass it.

The issue also names the guard on any claim: no acceptance before the load proves the engine
accepts the profile and prices from it, and before the rollback is exercised. This plan turns that
guard into the decision rules D1–D6 of §§3–4.

## 2. What this host verified of the deployment tree

The `strand` at `738dd49` configures, builds and passes its host suites on a machine with no
card (gcc 14.2.0, `RAD_WITH_HIP=OFF`, `RelWithDebInfo`, Ninja): 683/683 targets, exit 0. Every
draft suite passes; against the campaign plan's `b1dbcf8` audit no suite lost a check, and the
two the boundary work touched gained — checks were added, none weakened:

| suite | checks / cases at `b1dbcf8` | checks / cases at `738dd49` |
|---|---|---|
| `draft_cost` | 646 / 52 | 661 / 53 |
| `draft_policy` | 1770 / 43 | 1770 / 43 |
| `draft_predictor` | 3926 / 22 | 3926 / 22 |
| `draft_reference` | 89 / 7 | 89 / 7 |
| `draft_rounds` | 54 / 3 | 54 / 3 |
| `draft_types` | 408 / 31 | 408 / 31 |
| `sched_draft` | 4117 / 90 | 4336 / 102 |
| `config` | — | 2418 / 58 |
| `engine_e2e` | — | 657 / 42 (3 GPU cases skipped on this host) |
| `engine_cost_drift`, `engine_reference_basis`, `server` | — | 36 / 4, 188 / 8, 535965 / 179 |

The machinery the remaining bullets exercise, each fact re-read at `738dd49`:

- **The load is fail-closed by name.** `--draft-profile` (`core/config.cpp:196`) is read once at
  startup: `Engine::serve` derives the runtime first and refuses a deployment whose identity
  cannot be established (`core/engine.cpp:2412-2417`), then
  `Engine::load_draft_cost_profile` (`core/engine.cpp:1919`) parses and binds. A parse failure
  aborts the start with the reader's own refusal (`core/sched/draft_cost.cpp:654-665` names the
  format gate: a v1/v2/v3 file is refused as *"…(version 1 has no draft distribution … a profile
  is regenerated, not rescaled)"*). A fingerprint or execution mismatch does not abort — it
  leaves the table inactive with `reason()` naming the first differing field
  (`core/sched/draft_cost.cpp:1001-1023`), the engine warns *"--draft-profile '…' is not used for
  this deployment: <reason>"* (`core/engine.cpp:1939`), and `Scheduler::init` then refuses the
  start under `--draft-policy cost` — *"no admissible cost baseline -- <reason>"*
  (`core/sched/scheduler.cpp:309`). A started cost run is therefore itself evidence that the
  profile parsed, matched the fingerprint, and carries at least one usable bucket (the
  `has_usable_bucket()` gate, `core/sched/scheduler.cpp:305-312`).
- **The loaded profile is a reference and prices only measurements.** A file whose basis is
  no-loan binds as an immutable reference: `estimate()` answers only from exact `Measured`
  buckets — no interpolation, no heuristic anchor — and `observe()` counts `reference_skipped`
  without touching a bucket (`core/sched/draft_cost.h:326-335`,
  `core/sched/draft_cost.cpp:1040-1076`: step 1 is the exact shape; *"a reference stops after
  step 1"*). The success log names the basis: *"…: N buckets, basis no-loan -- an optimisation
  basis for the plan's start placement without borrowed copies… it is kept immutable and only
  exact measured buckets price"* (`core/engine.cpp:1943`).
- **The price key is exact and small enough to recur.** The key is the full shape — `n_rows`,
  `verify_k`, `verify_tokens`, the context class, the cycle's own history and prefill token rows,
  sampling, replay, drafter, pipeline depth, the verify distribution — plus the executed draft
  extent (`shape_same` + `extent_same`, `core/sched/draft_cost.cpp:102-118`). `history_tokens` is
  the MTP history/QSA rows the cycle carries, not the request's context length
  (`core/sched/draft_policy.cpp:207`), and `prefill_tokens` is zero on a decode-only step — so a
  steady decode cycle's key is a small, repeating combination, and a serving run *can* hit
  measured keys. The one documented interpolation step (neighbouring context class,
  `core/sched/draft_cost.cpp:1080-1092`) exists only for a non-reference table; this profile
  never uses it.
- **The search prices every offered tier on its own exact price.** Each tier of the ladder is a
  candidate; an unpriced tier is simply not compared, and a missing price for the deepest tier
  no longer discards the shallower ones (`core/sched/draft_policy.cpp:455-505`). Only a cycle
  that prices no tier at all takes the named baseline — the ladder's middle tier, K3 of the
  default ladder `0,2,3,4,6` (`core/rad_core.h:927`,
  `core/sched/scheduler.cpp:413-414`) — as `source=fallback`,
  `fallback=unknown_shape` (`core/sched/draft_policy.cpp:553`). The ladder-gap refusal
  (`core/sched/draft_policy.cpp:518`) cannot fire on the default ladder: the baseline K3 does not
  run deeper than the top tier K6.
- **Tier 0 is offered and can never be priced by this profile.** No `draft_k 0` bucket exists in
  the recording, an unpriced tier is not compared, so K0 cannot be chosen even where dropping
  drafts would pay — the issue's own analysis, and the reason §6 measures the bucket instead of
  deriving it. Every trace line of §3's runs will show `0:invalid` in `tiers=[…]`: that is this
  gap, visible by name.
- **The rollback is the start line, and its refusals are pinned.** The way back is the same line
  with `--draft-policy legacy` (the fixed depth; `kDraftPolicyDefault` stays `legacy`,
  `docs/ADAPTIVE-MTP.md`, "The deployment start line"). `--draft-hysteresis both` and
  `--draft-fallback-k` are refused by name under legacy
  (`core/sched/scheduler.cpp:448-459`, pinned by `tests/sched_draft_test.cpp:4230`),
  `--draft-calib-k` / `--draft-reference` are refused outside a calibration
  (`core/engine.cpp:1963-1983`), and `--draft-profile` is *not* refused under legacy — nothing
  prices from it there, so the rollback line drops it rather than name a policy that is not
  running.
- **The observables exist and are named.** `/stats` carries the `draft_cost` block — `cycles`,
  `priced`, `fallback`, `empty`, `no_plan`, `calib_forced`, `calib_cut`, `reference_skipped`, and
  for the last plan with rows `last_source`, `last_fallback`, `last_replay`, `tiers_priced`
  (`core/server/dashboard.cpp:287-306`); `/metrics` carries
  `radiance:draft_position_reached_total{position="k"}` — the verifies that evaluated each
  draft position, the observable the e2e load test itself reads
  (`core/server/metrics.cpp:511-514`); each request ends with the ledger line *"draft policy: req
  N over C cost cycle(s), U without a priced candidate; priced tiers per cycle k:count …"*
  (`core/sched/scheduler.cpp:1121`); and `RADIANCE_DRAFT_TRACE=1` writes up to 64 per-cycle plan
  lines with `source=`, `fallback=`, `price=ok:…|invalid|-` and the per-tier
  `tiers=[0:…,2:…,3:…,4:…,6:…]` (`core/sched/scheduler.cpp:48-95`).

## 3. Stage 2a — the load run, pre-registered

**Preconditions.**

- P1. The profile file is the recorded one, verified by its hash (`sha256 b16b6f36…`) before the
  run, and the sealed line of the run carries the git SHA, the engine binary's hash, the
  container identity, the start line verbatim, and the profile's hash (the campaign plan's §1.2
  and §4; the engine carries no build id of its own, so the binary hash is the only binding a
  rebuild cannot fake).
- P2. The GPU window is exclusively held: one coordination lock over the TP2 pair, the running
  server its owner (§7).
- P3. The start line is the deployment's own, with the policy and the profile added — the guide's
  minimal form: `--draft-policy cost --draft-profile /path/to/native-reference-profile.json`, on
  the production line (`--tp 2 --tp-wire wht6 …`), coupled mode on, the ladder and the fallback
  at their defaults (`0,2,3,4,6`, middle tier K3) unless the deployment line says otherwise.
  **The ceiling is not the comparator's.** The fixed-K3 line carries an explicit
  `--num-speculative-tokens 3`; carried over to the cost line it would not mean "draft up to 3"
  but silently reshape the policy: the default ladder clamps every tier to the ceiling
  (`core/sched/scheduler.cpp:370-375`), so the ladder becomes `0,2,3` and the named baseline —
  the clamped ladder's middle — becomes **K2**, not K3 (`core/sched/scheduler.cpp:414`). Under
  cost the ceiling is the executable head depth: leave `--num-speculative-tokens` at its `auto`
  default, which resolves to the drafter's declared depth
  (`core/engine_bringup.cpp:1016-1019`), or set it to that depth explicitly. The recorded k6
  buckets say this deployment's head depth is at least 6; the sealed line records the resolved
  ceiling with the rest of the line.
- P4. The serving traffic includes **short prompts** (context class 0, the only class the profile
  measured) beside the 64K prefill target tier — because of D3 below, a pure-64K load run cannot
  prove "prices from it" at all. The exact request mix is recorded on the sealed line.
- P5. The requests' sampling class is the one the profile measured. The sampling dimension is part
  of the price key; if the calibration drove greedy and the serving requests sample, every cycle
  is unpriced *even at class 0*, and that reading must not be mistaken for a binding failure.
  The profile's bucket shapes say which class they carry; the run's request parameters say which
  one it drives. They are compared before the run, not after.

**The run.** One server lifetime: start under the lock, serve the §3-P4 mix, read `/stats` and
`/metrics` after the window, stop. `RADIANCE_DRAFT_TRACE=1` on the environment for the first
window (the trace self-limits to 64 plan lines; the ledger line covers every request).

**Decision rules, pre-registered.**

- **D1 — the bind.** The run starts, or the load failed; there is no third outcome. The success
  line is *"--draft-profile '…': 492 buckets, basis no-loan -- an optimisation basis…"*; the
  budget-drift warning (same resolved plan, different measured start budget,
  `core/engine.cpp:1933`) still binds and is recorded with both budgets. A refusal is the
  correct fail-closed outcome and is kept in the record with the field it named — the campaign
  plan's own rule for a blocked profile: restamping it is not an option.
- **D2 — the fallback half.** At the 64K prefill target tier, every decode row is context class
  2 (`draft_ctx_class`, `core/sched/draft_types.h:66-71`), the profile measured only class 0, and
  a reference does not interpolate — so the prediction, from the coverage table and the estimate
  semantics alone: `priced` stays 0, `fallback` ≈ `cycles` minus `empty`, `last_source` reads
  `fallback`, `last_fallback` reads `unknown_shape`, `tiers_priced` reads 0, and the trace shows
  `price=invalid tiers=[0:invalid,2:invalid,3:invalid,4:invalid,6:invalid]`. **This is the
  honest expected result, not a failure of the load** — it is the coverage table read through
  serving, and §5 turns it into the per-class report. A run that instead *crashed*, hung or
  refused requests is a failure of the run and voids the cell.
- **D3 — the priced half.** The load run proves "prices from it" only if some cycle prices: at
  least one trace line with `price=ok:…` and a `/stats` `priced ≥ 1` in the same window, on
  class-0 traffic, after the start placement has settled (prices are held until the fill
  materialises, `core/sched/draft_cost.h:338-343` — early cycles are unpriced by design, and
  tape warmup cycles are record/audit-keyed, also by design). Two outcomes, both honest:
  - `priced ≥ 1`: the engine accepts the profile and prices from it — bullet 4's load claim is
    proven, and §5 reports the shares.
  - `priced = 0` on class-0 traffic that P5 says should price: the finding is *"the profile
    binds but prices none of the shapes the serving mix drove"* — recorded as that, with the
    trace's shape keys beside the profile's bucket keys, and §6's follow-up cells become the fix.
    The load claim is then only half-proven (bind yes, price no), and the record says so; it is
    never rounded up.
- **D4 — the K0 gap, visible.** `tiers=[0:invalid,…]` on every trace line is the missing `draft_k
  0` bucket showing by name. It is expected, not news; §6 measures it away.

## 4. Stage 2b — the rollback check, pre-registered

**The run.** The same start line with the policy switched back: `--draft-policy legacy
--num-speculative-tokens 3`, the profile path and every cost-only flag removed —
`--draft-fallback-k`, `--draft-hysteresis both`, `--draft-calib-k`, `--draft-reference` are
refused by name under legacy (§2), and a refusal here is the rollback failing loudly, which is
the design working; the line is fixed, not argued with.

**Decision rules, pre-registered.**

- **D5 — the rollback runs fixed K3.** The startup line names the fixed depth; the per-request
  ledger lines are absent (no cost policy ran); `/stats` carries no `draft_cost` block (a legacy
  run counts no cost cycles); the served generations are the fixed window's. A K3 window that
  serves, completes and reports no cost ledger is the rollback proven.
- **D6 — the rollback is total.** No `--draft-profile` on the line (nothing prices from it under
  legacy), no cost observables in the stats, and the run's generation identity unchanged. The
  rollback check is the campaign's safety line for every later stage: if a later cost run has to
  be abandoned mid-campaign, this line is where serving falls back to, so it is checked once
  here and cited thereafter.

## 5. Stage 2c — the coverage report, pre-registered

Stage 2's product is the coverage table (the campaign plan's own words), and the recorded
profile's coverage is already known — the owner's honest numbers:

- 116 of 492 buckets usable; 376 carry a disturbing flag (`record` 297, `admission` 109,
  `audit` 79) and are excluded — not smoothed, not merged into usable classes.
- The usable grid, all at context class 0: draft k2 / k3 / k4 / k6 measured across batch sizes
  0–8 (21 / 27 / 31 / 32 buckets, ~2100–2200 samples each); k1 thin (4 buckets, 11 samples),
  k5 a single sample; **no `draft_k 0` bucket**; **no context class 1, 2 or 3 bucket**.

What is *not* known is the per-class serving share: of C1/C2/C4/C6/C8's cycles, how many found a
price, how many took the named fallback, with which reason. That is measured by §3's runs and
reported per class:

| class | cycles | priced | fallback (unknown_shape) | empty | priced share |
|---|---|---|---|---|---|
| C1 | | | | | |
| C2 | | | | | |
| C4 | | | | | |
| C6 | | | | | |
| C8 | | | | | |

The counters are read from the `/stats` `draft_cost` block by their engine names — no synonyms
(the campaign plan's "one name per quantity"); the shares divide by `cycles` per class, and every
figure's sealed line names the run it came from.

**Reading rules, pre-registered.**

- **R1.** A class whose profile prices only a small share of its cycles runs the named K3
  fallback for the rest *by design* — honest, but it bounds the class's achievable gain by the
  priced minority (the campaign plan's own rule). With this profile, the 64K prefill target tier
  is expected to price nothing (D2); the report says so per class instead of averaging it away.
- **R2.** The bucket-coverage table (which shapes were measured) and the serving-share table
  (which cycles priced) are reported side by side and never merged: the first is a property of
  the profile, the second of the workload.
- **R3.** Missing prices are measured next (§6), not interpolated, not prorated, not smoothed.
  The engine already refuses to invent them; so does this plan.

## 6. Stage 2d — the missing prices, measured next

The gaps are named by the recording and each has one fix — a measurement, never a derivation:

- **The K0 bucket.** Tier 0 is offered by the default ladder and never priced: no `draft_k 0`
  bucket exists, and `DraftPolicy` does not compare unpriced candidates. The fix is a disclosed
  follow-up calibration whose forced sequence includes 0 — `--draft-calib-k` accepts it
  (`tests/config_test.cpp:867` pins `"0,6,0"`; K0 is a plain decode step with its own bucket,
  `tests/draft_cost_test.cpp`), and the rotation is recorded on the sealed line. Deriving a K0
  price from other buckets is not an option the engine offers, and this plan does not invent one.
- **Context classes 1–3.** The reference is Measured-only, so longer contexts are unpriced by
  construction until they are measured. The follow-up cells drive the serving stage's real
  shapes — the 64K prefill target tier's decode cycles — so the class 1/2 keys the cost policy
  will actually query are in the table, not wished into it.
- **k1/k5.** 11 samples and 1 sample do not stand as prices. The follow-up either densifies them
  or the ladder stops offering what was not measured — the operator's call, recorded either way.

Each follow-up cell runs under the same recording contract as the first recording (one server
lifetime per cell, 16 finished warmups, one confirmed boundary POST with epoch 2 and integer drop
counters, 32 scored requests, ordered export of the same epoch, no invented sample cap), the same
`--draft-reference` basis, and merges into a profile with the same fingerprint — more buckets,
same identity, sample counts preserved. A cell that fails the contract checker is a failed
calibration, not a half-valid profile.

## 7. What this plan cannot settle from a development host

- Every run of §3–§6 needs the GPU host: the card, the container, the model, BetterBench and the
  profile artifacts. This host verified the seam (§2) and nothing beyond it.
- The historical artifacts (0006/0007/0008) are audited where they lie, externally; this
  repository carries the record of why none was adopted, not the artifacts.
- The GPU window is single-tenant: every run takes the exclusive coordination lock over the TP2
  pair, a running server is the window's owner or is released in order, and no run of this plan
  is started by killing a blanket sweep's victim. The resource rule is not a ticket dependency —
  it binds this plan precisely because it bound the recording.
- The profile's own verification (sha256, fingerprint fields against the deployment identity,
  usable-bucket count) happens on the GPU host against the sealed lines, where the file and the
  deployment both are. The engine performs the binding check itself at startup; this plan's
  contribution is that its refusal outputs are named in advance, so a mismatch is recognised as
  the correct outcome and quoted, not debugged into a restamp.

## 8. The rules that hold

The campaign plan's §4 rules hold for every run of this plan, restated in one line each because
they are what make the runs readable later: **sealed lines** (a figure without its sealed line did
not happen); **one name per quantity** (the engine's `draft_cost_*` names, no synonyms);
**foreign numbers stay foreign** (the recording's numbers are the owner's records, cited as
such); **the GPU window is single-tenant**; **no fabricated success** — a run that could not
measure says so, a skipped suite is listed as not run with its reason, and a half-proven claim
(D3's second outcome) is recorded as half-proven.
