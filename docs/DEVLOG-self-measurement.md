# Devlog: the engine measuring itself

The story behind `radiance --diagnose DIR`, `core/diagnose.*`, `core/sched/steplog.h` and
`Scheduler::step_quiet`. Written the day it was built, on the CPU backend, because the machine it
was written on has no ROCm and the host backend is a primary target (docs/IMPLEMENTATION.md). The
user-facing description is TOOLS.md §The self-measurement mode and spec.md §16.1; this is the log of
how the numbers got their definitions, including the three that were wrong at first.

## What shipped

`--diagnose DIR` starts a deployment exactly as a serve would, runs one fixed workload through the
engine's own scheduler instead of listening, and writes `DIR/diagnose.json` and `DIR/diagnose.md`.
`--diagnose-baseline FILE` compares a run with an earlier report: token parity (a difference fails
the run) and pooled draft acceptance by point, category and draft position. `--diagnose-salt N` is
the counter every prompt is salted with; `--diagnose-acceptance-drop FRAC` is how far acceptance has
to fall to be called a fall.

The scheduler writes one `StepRecord` per step — time, rows, tokens that reached the sinks, the wait
since the previous step — into a fixed ring that exists only while the mode runs. A serving process
pays nothing for it: no ring, no store, one branch.

Three host-only test binaries cover it: `diagnose_test` (the arithmetic against hand-computed
numbers, the report's own review, the workload table's self-consistency, the ring's overwrite
counting), `diagnose_e2e_test` (the real binary on a tiny synthetic checkpoint: two runs' token
parity, a baseline accepted, a baseline of another salt refused, absence reported as absence), and
the existing `config_test`, whose "every listed option parses and is reported as given" case now
carries the four flags.

## What it measured, on the machine it was written on

Host backend, `libref` only, 2 threads, 512 MiB of pretend VRAM, the tiny synthetic Qwen35
checkpoint `tests/tiny_model.h` writes. These numbers say nothing about a real card: the tiny model
has a couple of thousand weights and every op runs the reference implementation. They are here
because they are the numbers that were *checked* — they are what the tests assert the shape of.

Whole workload, 14 points, 29 requests, 1795 generated tokens: 4.3 s wall, of which bringup is most.
The same workload rerun against that report: 4.7 s, exit 0,
`token-parity: all 29 request(s) generated exactly the baseline's tokens`.

| point | conc | live | decode step p50 | token interval p50 | waiting p50 | tokens/step |
|---|---|---|---|---|---|---|
| `decode-floor` | 1 | 1 | 621 µs | 623 µs | 1 µs | 1 |
| `category:structured` | 1 | 1 | 556 µs | 557 µs | 1 µs | 1 |
| `category:code` | 1 | 1 | 611 µs | 613 µs | 1 µs | 1 |
| `category:long-context` | 1 | 1 | 1211 µs | 1217 µs | 2 µs | 1 |
| `concurrency:4` | 4 | 4 | 1579 µs | 1583 µs | 1 µs | 4 |
| `concurrency:8` | 8 | 8 | 2363 µs | 2368 µs | 3 µs | 8 |
| `prefill:2048` | 1 | 1 | — | — | — | 1 step of 2048 rows |
| `interference` | 4 | 5 | 1603 µs | 1607 µs | 3 µs | 4, and 5 after the arrival |

Two readings of the same series, and the difference between them is the point: at concurrency 8 the
step takes 2.36 ms and the engine waits 3 µs between steps, so the machine is saturated and the
client's interval *is* the step's length. Nothing in `/metrics` says that — its decaying means report
a throughput and a step time, and the reader has to guess which of the two moved and whether the
engine is starved or busy. The decode floor's 621 µs against concurrency:8's 2363 µs is the same
engine, same weights, same step, eight times the rows: that comparison is what the mode is for, and
on this machine it is a factor of 3.8 rather than 8, which is the host backend being two threads.

The interference point is the one that could not be measured any other way: streams decoding, one
prompt arriving once each had produced eight tokens. Its cost is read as the already-running
streams' token interval before and after the arrival — `interference:long` goes from 777 µs to
2049 µs, `interference` from 1513 µs to 2113 µs — and the arriving request's own timings say nothing
about it, because the arriving request is not the thing that got slower. `--live`'s draft-acceptance
tile and `/metrics` have no quantity in this shape at all.

## Three numbers that were wrong at first

**1. `tok_out` counted rows, so the burst interval meant the gap between prefills.** The first
record stored what a step *computed*. A 1024-row prefill chunk emits nothing, and the interval
between "bursts" came out as the gap between chunks — the very quantity the mode exists to separate.
Now `StepRecord::tok_out` is the tokens that reached sinks, counted where the scheduler pushes them,
and `tokens_per_step` is checked in the tests against the number of sequences the point had live.

**2. A step's emissions leaked into the next step's record.** The per-step counters are reset in two
places — the planner, and `promote_next()` for a step issued ahead — and the new counter was reset in
only the second. On any step planned through the first, `tok_out` carried its predecessors too: a
one-token point reported five tokens in its only step. `diagnose_e2e_test` now asserts
`tokens_per_step.emitted.max <= max_live` for every point, which is the invariant that was broken.

**3. A point's last step belonged to the next point.** A result is committed by the runtime thread
and its step recorded a moment later by `step_finished()`, under a second acquisition of the same
lock. The mode waits on sinks, so it saw the last token, drained the ring, and the step that
produced it landed in the *next* point's records. `decode-floor` had 96 tokens and 95 decode steps.
`Scheduler::step_quiet()` now says when no step is open, none is issued ahead and no committed step
awaits its record; the point's drain waits for quiet, and if quiet does not come it says the
distributions cover the steps before it rather than pretending otherwise. `steps_by_shape.decode`
equals the point's decode steps now, and the emitting-step maximum matches the point's live count.

Two smaller ones, both found by writing the report out and reading it: the step *counts* came from
the scheduler's phase counters while the *distributions* came from the records, and the two disagree
by one step — a prefill that completes is a decode step to the counters and a prefill row-count to
the records. They are reported apart (`steps_by_phase`, `steps_by_shape`) rather than reconciled by
force. And the interference split was taken at the last prompt of *every* point, not at the arriving
prompt of the two points that have one; the condition now says so.

## Absence, not zero

Every quantity is either measured or absent with the reason: `prefill:2048` has no token interval
("the point emitted tokens from one step only"), a model with no draft head has no acceptance
("no draft token was verified"), and a point whose KV the 512 MiB pool cannot hold says what would
have to change and does not fail the run. `review()` re-checks the finished report — a distribution
whose p50 is above its p90 cannot be written, a cache hit makes a point not a measurement, dropped
ring records are named — because a report that reads plausibly and means nothing is the failure mode
this whole mode exists to avoid.

## What is still missing

- The workload is one deployment's size. A 480-token prompt is long to a tiny model and short to the
  container in `GUIDE.md`; `category:long-context` would be a prefix-cache hit's anecdote on a
  machine that cannot hold it, reported as such. Point shapes scaled from the container's own
  context and KV budget are the obvious next step, and would make two machines' reports comparable.
- The host backend's step bracket covers host work and the reference kernels, so the numbers here are
  not comparable to a card's, and the report says which backend wrote them (`engine.backend`).
- A baseline compares tokens exactly and timings not at all. That is deliberate — timings on a busy
  box move by more than most changes — but it means the report answers "did this change the words, and
  what did it do to the shapes" and leaves "is it faster" to the distributions beside them.

## Addendum: what review found after this log was written

This log records the day the mode was built. Reviewing the merged change later, against the real
model and in sealed pairs, found two more defects; both are fixed on this branch and each carries
the test that would have caught it. They belong in this log because the log's third story above
claims the last-step question was settled, and it was only half settled.

**4. The acceptance-drop flag never reached the comparison.** `--diagnose-acceptance-drop` was
parsed into the config and echoed by the startup report, but `run_diagnose()` passed a hardcoded
`0.10` to `compare_acceptance()`, so the threshold the operator named was compared against nothing.
The flag's value is observable end to end in exactly one place: the note the comparison writes when
nothing fell ("no point's pooled acceptance fell by N% or more"). The e2e case added with the fix
runs the spec model against its own report with the flag at 0.50 and reads the note back; with the
hardcoded default restored, the note says 10% and the case fails.

**5. The counters were charged at the sink's fill, one step ahead of the drain.** Two sealed runs
of the same build, same salt, quiet box: `decode-floor` reported `steps: 96` in one and `steps: 97`
in the other, `steps_by_phase.decode` 95 and 96, while the record-based counts — the ones the
distributions are made of — said 97 both times. A count is not a timing; two identical runs
disagreeing on one was the sealed-pair protocol doing its job. The point's metrics were read before
the quiet wait, but the point is not over at the sink's fill: the step that filled it owes its
`step_finished()`, and behind it the drafter's issued-ahead step runs a full pass that emits
nothing, because its request is done. Those increments raced the read; the records, drained after
quiet, did not. The counters are now read at the same boundary the records are drained at, and the
e2e asserts the contract the header states — the phase families sum to the point's step count, and
so do the shapes. After the fix, the sealed pair agrees: 97 and 97, both families, both runs.

An honest word about the mutation control on that one: with the counters moved back to the fill,
the e2e's tiny model finishes its tail step faster than the diagnose thread wakes, the race
resolves the same way every time, and the suite cannot kill the mutant. The kill evidence is the
sealed pair itself — 96/97 before, 97/97 after. The invariant checks pin the boundary and catch the
flap wherever the timing allows it; the protocol catches it everywhere.

And one hole the mutation control found in the suite rather than in the mode: with
`Scheduler::step_quiet()` made to return true unconditionally, every e2e case still passed on this
host — the drain's wait resolves benignly when it is lied to. `sched_test` now drives the quiet
arms by hand, in the engine's own order, and the disabled mutant dies on four arms at once.
