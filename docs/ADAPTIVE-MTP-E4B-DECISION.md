# Adaptive MTP: the #62 keep-or-discard decision — the E4-B protocol

**Status: a plan, not a result.** This document pre-registers the decision rules for the first
measured policy/algorithm change of the #59 campaign — the E4-B cycle stop, `--draft-cycle-stop`
— before the deciding pairs run. It extends PR #70's campaign plan (its Stage 3) and changes
neither that plan nor PR #71's #61 procedure. Nothing in it was measured on the host it was
written on: that host has no `/dev/kfd`, no ROCm, no model container and no BetterBench. Every
measured figure quoted here is the deployment thread's (the GH-60/GH-62 comments of 2026-10-26),
labelled **foreign**; every ratio derived from those figures is labelled **derived** and is a
division of the published numbers, checkable by hand. The E4-B code itself is not in this
repository — `git log --all` over every branch finds no `--draft-cycle-stop`, and its fix
`52252c3` is the deployment tree's — so this plan binds the *reading* of the deployment's
measurement, and the deployment owner applies it.

Two trees are named, and every `file:line` below was re-validated against the tree it names:

| tree | SHA | what it is |
|------|-----|------------|
| `main` | `39f957d` | `origin/main` at the time of writing |
| deployment | (not reachable) | the owner's serving tree: carries E4-B, the five-run basis runs, the paired harness |

## 1. Where issue #62 stands

The issue's four acceptance criteria, and what satisfies each today:

1. **Bottleneck, hypothesis and expected effect documented before the change.** The deployment
   thread documented all three before the first serving pair: the bottleneck (a draft round is
   paid for every row even where the row's accepted prefix will not survive the round), the
   hypothesis (stopping a row at its accepted-prefix boundary removes built and issued work —
   "the first change in this program that removes work instead of rearranging it"), and the
   expected effect (a throughput lead). This plan completes the expected-effect record with the
   quantitative envelope of §3 — written before any deciding pair exists, so the pre-registration
   keeps its substance: the envelope is derivable without the deciding data.
   **Met by the record as it stands; the envelope is completed here.**
2. **Existing policy/predictor/scheduler interfaces used, correctness cases checked.** E4-B runs
   through the existing coupled seam, all of which is on `main`: the coupled-mode gate that
   requires the linear serial chain (`core/engine_draft.cpp:62-77`), the per-round advance modes
   (`ADV_ROUND`/`ADV_TAKE`, `core/sched/advance.h:45-46`), the coupled state wire with its
   per-(round, row) states (`abi/rad_spechub.h:36-45`). The correctness record — 18/18 tests
   green after the row-renumbering fix `52252c3` — is **foreign**. **Open in this repository: the
   E4-B code is not here, so no correctness run of it can be shown from a checkout.**
3. **One BetterBench Quick as the direction check, against the K3 basis, result per load
   class.** One complete pair exists (**foreign**, all five classes, zero error lines). The
   deciding pairs — three alternating runs of the same binary — have not run. **Open.**
4. **The keep-or-discard decision documented; ineffective changes not kept just to fulfil the
   ticket.** Explicitly open in the thread's own words: "Lead, not result." **Open.** The rules
   below are what the decision will apply; they are fixed here, before the data.

#62 completes when the decision is made and recorded — a discard with the failed hypothesis
named satisfies the issue exactly as a keep does. The issue's own fourth criterion ("wirkungslose
Änderungen nicht allein zur Ticketerfüllung behalten") makes that explicit, and Stage 3 of the
campaign plan restates it: a change whose effect is within the spread is not kept for ticket
fulfilment.

## 2. The record this decision reads

### 2.1 The K3 basis and its noise floors (**foreign**)

Five unchanged fixed-K3 Quicks, one owned server, 48/48 requests per class, binary `aa00289`,
`runs/gh-60/20261026T035246Z-host-fixed3-five-baselines-provisional-r01`. The class baseline is
the **median of the five run medians** — the pre-registered Stage-1 rule — and the class noise
floor is its relative spread:

| class | basis (median) | min | max | spread (noise floor) |
|---|---|---|---|---|
| C1 | 193.92 | 191.80 | 199.96 | 4.21 % |
| C2 | 152.16 | 145.31 | 154.99 | 6.36 % |
| C4 | 110.00 | 107.00 | 113.13 | 5.58 % |
| C6 | 91.36 | 89.83 | 92.49 | 2.91 % |
| C8 | 78.93 | 77.93 | 81.65 | 4.71 % |

The metric is `concurrency_rows(results)[level].decode_med` per class (the frozen field, GH-60's
measurement contract). The thread's summary "rig noise ±5 %" is consistent with these floors but
the floors are the pre-registered quantities; the decision rules read them, not the summary.

### 2.2 The E4-B pair, read against the basis (**foreign** inputs, **derived** ratios)

One pair, same binary, only the flag differing (GH-60 comment of 2026-10-26 13:52):

| class | flag 0 | flag 2 | f2/f0 | flag 2 / basis | flag 0 / basis |
|---|---|---|---|---|---|
| C1 | 185.73 | 190.11 | 1.024 | 0.980 (−2.0 %) | 0.958 (−4.2 %) |
| C2 | 141.93 | 142.55 | 1.004 | 0.937 (−6.3 %) | 0.933 (−6.7 %) |
| C4 | 102.27 | 105.31 | 1.030 | 0.957 (−4.3 %) | 0.930 (−7.0 %) |
| C6 | 79.97 | 85.78 | 1.073 | 0.939 (−6.1 %) | 0.875 (−12.5 %) |
| C8 | 65.59 | 74.80 | 1.140 | 0.948 (−5.2 %) | 0.831 (−16.9 %) |

The reading this table forces, before any statistics: **both arms sit below the K3 basis in every
class** — the flag-on arm by 2.0–6.3 %, the flag-off arm by 4.2–16.9 %. The lead is within-binary,
against a control that is itself depressed. Under Stage 3's rule (the delta that matters is the
one against the Stage-1 basis, beyond the noise floor, in the intended direction) the present
record supports no keep on any class. The thread's own caveats — "one pair", "the control is
itself depressed against the canonical binary", "not yet identity-slow" — say the same. What the
record does not yet say is whether the depression is the binary's off path or the flag's
neighbourhood; that is what the identity gate (§4) separates, and why the gate precedes any
keep/discard.

### 2.3 The depth-curve restatement (record hygiene, **derived** from **foreign** tables)

The depth-curve table (GH-60 comment of 2026-10-26 06:50) prints its K3 column with the Stage-1
**maxima** for C6 (92.49) and C8 (81.65) where the pre-registered basis is the **medians**
(91.36, 78.93); C1/C2/C4 carry the medians. Under the Stage-1 rule the restated ratios are:

| class | K2/K3 printed | restated | K4/K3 printed | restated |
|---|---|---|---|---|
| C6 | 0.877 | 0.888 | 0.974 | 0.986 |
| C8 | 0.859 | 0.889 | 0.933 | 0.965 |

The reading survives the restatement — K2 loses beyond every floor, K4 is within noise of K3 and
better nowhere; the restated C8 K4/K3 moves *inside* the floor, which strengthens it. But the
table is the measured closure of the depth lever, and that closure is what made E4-B the
candidate, so the record should be restated once, under the medians, before the #62 decision
cites it. One statistic per named quantity; the campaign's own rule.

### 2.4 The per-position acceptance (**foreign**)

`/stats` per-position conditional acceptance (`runs/gh-60/20261026T120942Z-native-stats-acceptance-r01`;
the counters are `draft_pos_reached`/`draft_pos_accepted`, `core/sched/metrics.h:130-131`):

| class | position 1 | position 2 | position 3 |
|---|---|---|---|
| C1 | 0.750 (60/80) | 0.700 | 0.810 |
| C8 | 0.775 (562/725) | 0.712 | 0.733 |

These are the inputs of the derived envelope below.

## 3. The derived envelope (**derived**, not measured)

### 3.1 The stop channel

At flag 2, a row stops after round 2 when its accepted prefix is shorter than 2, so a row
continues to round 3 only when position 1 and position 2 were both accepted. Per row:

- P(continue) = p₁·p₂ — C1: 0.750 × 0.700 = 0.525; C8: 0.775 × 0.712 = 0.552
- P(stop) — C1: 0.475; C8: 0.448

Round 3 disappears entirely (is never built or issued) only when **every** row of the cycle has
stopped. Per cycle:

- C1 (one row): P ≈ 0.475 — round 3 is skipped in roughly every second cycle.
- C8 (eight rows): P ≈ 0.448⁸ ≈ 0.0016 — round 3 is skipped in roughly one cycle in six hundred.

Rows that stop while others continue leave round 3 issued with fewer rows — the row-shrinking
channel, a smaller pass rather than a missing one.

### 3.2 The C1 arithmetic

The stage split (**foreign**): host issue ≈ 674.0 µs per pass, ≈ K+1 = 4 passes per step at K3,
so ≈ 2.70 ms of host issue per step. The step rate implied by the basis and the acceptance
(**derived**): committed tokens per verify ≈ 1 + p₁ + p₁p₂ + p₁p₂p₃ ≈ 2.70 at C1, so the 193.92
tok/s basis implies ≈ 71.8 steps/s, ≈ 13.9 ms/step, of which host issue is ≈ 19 %. Skipping
round-3 issue in 47.5 % of cycles saves ≈ 0.475 × 674 µs ≈ 320 µs ≈ **2.3 % of the C1 step**,
before charging the added readback (§3.4).

The one-pair C1 lead is +2.4 % within-binary — the same order as the envelope. Two consequences:
the C1 lead is the documented mechanism at its documented size, and **a confirmed C1 lead of this
size still does not clear C1's 4.21 % floor** — under D3 it is not keepable on its own.

### 3.3 The C8 discrepancy

The full-round channel at C8 is bounded at ≈ 0.16 % of cycles ≈ 1 µs per step — three orders of
magnitude below the one-pair C8 lead of +14.0 %. If the pairs reproduce a C8 lead, it cannot be
the documented stop channel that produced it. The candidate channels:

1. **Row shrinking** — round 3 issued with fewer rows; the saving then rides on device work per
   row (a MoE streaming fewer experts), not on host issue, which is per-pass.
2. **Overlap** — the once-per-cycle readback changing when passes become ready relative to the
   card.
3. **Control recovery** — the depressed flag-off arm flattering the ratio. This channel is
   excluded from the *decision* by G1 + D3 — the decision reads against the basis, not the
   control — but it remains a live explanation of the one-pair number.

D5 makes the counters, not the throughput, name the channel.

### 3.4 The readback is a cost

The coupled design exists so the host does not read back per step: the advance's own header says
deciding acceptance on the host "costs a readback and a sync per step with the card idle", which
is why the card does it (`core/sched/advance.h`). E4-B reintroduces exactly one such readback
per cycle, at round K. That is within #67's bound — "keine Host-Abfrage oder zusätzliche
Synchronisation je Draft-Token"; once per cycle is not per draft token — but the bound permits
the mechanism, it does not pay for it: the readback's stall is part of the net the pairs measure,
and the ledger reports it rather than amortising it away.

## 4. The gates, pre-registered

**G1 — identity.** Before any flag-on number is read as the change's effect, the E4-B binary
with the flag **off** must reproduce the K3 basis within each measured class's noise floor:
|flag-0 / basis − 1| ≤ spread, per class. Today's record fails G1 on every class (§2.2). A G1
failure has one meaning: the binary, not the flag, is the difference — the finding is the off-path
defect, recorded as that, and no keep/discard on the flag is issued from behind it. The off path
is fixed first (the thread's own next step). A change whose off path cannot be brought within the
floor is discarded on the identity rule alone: the deployment's standard line is the flag-off
binary, and a standard that measures 4–17 % under the basis is not the standard, whatever the
flag adds on top.

**G2 — correctness.** The Stage-3 suite list — K0 transitions, mixed depths, first rejection and
censoring, generation identity, cancellation, EOG, output and context limits, history-only work,
issue-ahead overlap — plus the case the first version's abort exposed: the stop renumbering a
cycle's rows against the card's per-round state, which surfaced as
`MTP coupled: sequence … round … returned status … refusing the step` and was a row/state pairing
break, not a phase guard. Green on the binary the pairs measure, the aborting case as a named
test. The foreign record (18/18 after `52252c3`) is the fix's, not the decision's; the decision
needs the suites on the measured binary.

## 5. The deciding measurement, pre-registered

Three alternating pairs of the same binary, only the flag differing, on the exclusive GPU window,
one owned server per run (the five-run basis's lifetime policy). The first arm of the first pair
is fixed by the binary hash's parity — even hash: flag-off first — and alternates thereafter
(AB/BA/AB or BA/AB/BA); no coin toss, no discretion. Classes: C1 and C8 at minimum — the
Ergänzung's regular set; all five is welcome and is already the one pair's practice. Whichever
set runs, the decision names exactly the classes measured and no others; the omitted classes are
"not proven by this decision" and are first counted at Stage 5/6 — the shortened Quick replaces
neither acceptance nor the proof of the omitted classes (the Ergänzung's own rule).

Sealed line per run: git SHA, engine binary hash, container identity, start line verbatim, card
clocks, the effective full Quick configuration, actual prompt tokens at the 64K target tier, and
the counters every figure came from. The direction-check contract is unchanged: 48/48 requests
per class, no error lines, TTFT p50/p99 reported separately, raw values checked beyond the
report's own aggregation. A run that could not measure says so; the one-pair record stays one
pair and is never quietly promoted.

## 6. The decision rules, pre-registered

- **D1 — identity precedes effect.** No keep/discard on the flag is issued unless G1 holds on
  the measured classes. If G1 still fails after the off-path fix, the next honest move is a fresh
  five-run fixed-K3 series in the Stage-1 discipline to test host drift — not a re-read of the
  old basis, which stays in the record either way.
- **D2 — the flag's effect is read within-binary.** Per class, per pair: the ratio of the
  flag-on median field to the flag-off median field. The class statistic is the median of the
  three pair ratios. All pairs are reported, including the ones that lose.
- **D3 — the keep bar is the basis.** Per class, per pair: the ratio of the flag-on median field
  to the Stage-1 class median — the cross-binary read that G1 licenses. A class is keepable only
  when the median of its three ratios is ≥ 1 + the class noise floor. A within-binary lead with
  both arms below the basis is a lead against a depressed control; the present record (§2.2)
  supports no keep.
- **D4 — the mechanism shows its work.** The keep record carries the counters: rounds built and
  issued per cycle, flag 0 beside flag 2, per measured class, under the patch's own named
  counters (none exist in `/stats` on `main` or the strand; the patch names its own, and the
  one-name-per-quantity rule holds). #67's contracts apply although E4-B ran early under #62:
  discarding already-computed tokens is not work saving — the counters must show rounds never
  built or issued, not tokens dropped at commit — and the once-per-cycle readback's cost is in
  the ledger. A throughput lead without the counters is not attributable to the mechanism.
- **D5 — C8 is decided by the counters.** The full-round channel at C8 is bounded (§3.3) three
  orders below the one-pair lead. If the pairs reproduce a C8 lead, the keep record names the
  channel the counters show — row shrinking, overlap, or another named one. A lead of unexplained
  channel is recorded as unexplained; it is not kept as E4-B's mechanism, because the hypothesis
  record (bullet 1) is then wrong, and a wrong hypothesis is rewritten and re-measured, not waved
  through.
- **D6 — inconclusive is not kept.** A class whose median ratio sits inside its floor is not
  kept — the issue's own fourth criterion. Up to two further pairs (five, the Stage-6 count) may
  be run to try to clear it; a class still inside its floor at five is discarded, the record
  naming "effect within noise". #62 completes either way: its deliverable is the decision and
  the record, not a kept change.
- **D7 — keep means landed.** A kept change lands in this repository — the flag (default off),
  its tests including the renumbering case, its counters, its documentation — before it counts
  as kept for the campaign. Stage 5 integrates from the repository; a change that lives only in
  an unpushed tree is not kept, it is pending. A discard is recorded in #62 with the failed
  hypothesis named, and the E4 line returns to #67 with the verdict attached: it ran early under
  #62, and a discard answers its first bounded trial without closing the idea.

## 7. What this plan cannot settle from a development host

The host this was written on has no `/dev/kfd`, no ROCm, no model container, no BetterBench, and
no E4-B binary — the code is not in this repository (`git log --all` finds no
`--draft-cycle-stop`; the fix `52252c3` is the deployment tree's). What it could do, and did:
re-validate the engine vocabulary on `main` at `39f957d` — the coupled gate
(`core/engine_draft.cpp:62-77`), the per-round advance (`core/sched/advance.h:45-46`), the
coupled wire (`abi/rad_spechub.h:36-45`), the per-position counters
(`core/sched/metrics.h:130-131`), and the adaptive window's own per-sequence stop as the
precedent E4-B extends within the cycle (`core/sched/draft.h`) — and re-derive every ratio and
envelope of §2–3 from the quoted record. The three pairs, the counters, the gates and the
decision itself are the deployment's. This plan binds only their reading, and it was written
before the data.

## 8. The rules that hold

- **Sealed lines.** One line per run: git SHA, engine binary hash, container identity, start line
  verbatim, card clocks, the effective Quick configuration, actual prompt tokens, and the
  counters every figure came from. A figure without its sealed line did not happen.
- **One name per quantity.** The frozen BetterBench field per class; the Stage-1 medians as the
  only basis; the patch's own counter names for the work-saving evidence. No synonyms.
- **Foreign numbers stay foreign.** Every measured figure in this document is the deployment
  thread's, labelled as such; the derived ratios are divisions of those figures and invent no
  data. Nothing here was measured on this host, and no foreign figure is folded into a local
  claim.
- **The GPU window is single-tenant.** Every run takes the exclusive coordination lock; a
  running server is the window's owner or is released in order.
- **No fabricated success.** The one-pair record stays one pair; a skipped suite is listed as not
  run with its reason; a lead is a lead, a result is a result, and the difference is the rules
  above.
