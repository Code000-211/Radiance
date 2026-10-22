# Record: what "overlap" can mean for a prefill chunk

Issue #8 keeps open two questions: should a long prompt be served as overlapping chunks while the
same server decodes other requests, and should a request's own decode steps overlap the tail of its
own prefill. This document is the first acceptance item: what overlap means in terms the batch
builder, the arena and the per-sequence state can express, what the engine already runs, what it
cannot express at all, and the measurement that has to happen before any of the rest is believed.

It adds no policy. What it adds is the machinery to say, from the engine's own numbers, which rows
rode together and what that cost (`core/sched/metrics.h`, `/stats`, `/metrics`), the ordering check
that keeps a mixed step's recurrent state honest (`StepPlan::split_ordered()`, checked by
`BatchBuilder::build`), and the harness for the paired A/B (`scripts/fnmix.sh`). The measurement
itself needs a card and has not been run; what is settled here is which measurement answers it.

## Two things the word keeps mixing together

**Budget** is how many tokens one step may carry. `--max-num-batched-tokens`,
`--checkpoint-interval`, `--prefill-span` and `--busy-prefill-chunk` are all statements about
budget: they decide how a step is planned, and nothing else.

**Concurrency** is which rows are computed while which weights are in flight. That is a property of
one forward pass and of the staging around it, and no budget flag changes it.

`--prefill-span` is the flag where the difference bites hardest, because it looks like concurrency
and is not: a span carries several checkpoint intervals in one step, and it is taken only for a
prefill alone in that step (`core/sched/scheduler.cpp:1182`), with every other step planned at the
shared budget (`prefill_shared_tok`, `core/sched/geometry.cpp:35`). A span makes a step wider. It
never puts a decode row next to a chunk.

## What one step can say

Everything below is a field a kernel reads, not a figure of speech.

| term | what it is |
|---|---|
| **row** | one token of one sequence in one forward pass: a token id, a position, a KV slot |
| **entry** | one sequence's participation in one step (`StepEntry`, `core/sched/batch.h`): `n_tokens` rows from `ctx_len`, and whether it is a chunk or a decode row |
| **chunk** | an entry's rows when the sequence is still inside its prompt |
| **the split** | `RadBatch::n_seq_decode` / `n_tok_decode`: sequences `[0, n_seq_decode)` are decode rows, the rest are chunks (`abi/rad_runtime.h:116`) |
| **mixed step** | `RAD_PHASE_MIXED`: the step carries both. Not a third kind of pass; one pass, issued half recurrent and half chunked |
| **span** | how many checkpoint intervals a chunk may carry (`ChunkGeometry::span`), a budget |
| **staged step** | a step with at least `PrefillStager::min_rows()` rows, where the ring is armed and a layer's non-resident experts are copied ahead of the layer that reads them |

Two consequences of that table decide most of the issue.

A step carries **each sequence exactly once, in one entry**. The scheduler walks `running_` once per
step and admits each waiting request once, so there is no shape in which a sequence has two entries
in one plan.

The split is **by sequence, not by row**: a block issues over `cu_seqlens + n_seq_decode` and
`state_index + n_seq_decode * pitch`, with the token operands left at their base. A sequence is in
one half of a mixed step or the other; it cannot straddle it.

## What already overlaps

| what runs concurrently | turned on by | where |
|---|---|---|
| a chunk's weights with the layers that read them: the non-resident experts of the layers ahead are copied into the ring while earlier layers compute | any step of `min_rows` tokens or more | `PrefillStager::before_op`, `core/place/stager.cpp` |
| **the next chunk's first layers with this chunk's last layers**: when the prompt continues, `end_pass` stages the next trunk pass's first `kLead` layers, paced on this pass's last layers as if the passes were one | `--busy-prefill-chunk` keeping the next chunk at or above the stager's step, and a step at the arena's full plan | `PrefillStager::end_pass` (`core/place/stager.cpp:343-364`), `Scheduler::prefill_continues`, `Engine::probe_arena_levels` (`set_next_prefill`) |
| a chunk of one prompt with the decode rows of other requests, one pass, each half on its own path | `--busy-prefill-chunk auto` (default) or `N` | `Scheduler::busy_cap` (`core/sched/scheduler.cpp:868`), the split, `rad::batch_split` |
| two prompts' chunks in one step, each leaving the other a share of the budget | the same flag: `auto` reserves a share per prefill behind the one being planned | `Scheduler::busy_cap` and its `behind_pf` reserve, `core/sched/scheduler.cpp:1267` |
| a decode row's forward pass with the step before its commit | always, for a step that can be planned before the commit | `Scheduler::step_next` |
| decode-only steps between chunks, so the decoders keep a cadence | `--prefill-interleave N` | `Scheduler::step`'s hold, `core/sched/scheduler.cpp:1168` |

The second row is the honest answer to "serve a prompt as overlapping chunks". Two chunks of one
prompt cannot be computed at the same time: chunk N+1 reads the KV chunk N wrote and continues from
the recurrent state chunk N left, so the dependency is the model's, not the scheduler's. What can
overlap is the **weights**: chunk N+1's first layers are in flight across the link while chunk N's
last layers compute, and the engine already does that, one chunk boundary ahead.

## What cannot be expressed

A sequence's own decode row inside a step carrying its own prefill chunk. One sequence, one entry,
and the split is per sequence, so this is not a policy gap; it is not a shape the batch carries.

That is also the precise answer to the acceptance note that "a prefill chunk and a decode step touch
the same per-sequence state". They do when they belong to the same sequence, and one step never
holds both of those. When they belong to different sequences they touch different `state_index`
slots and different conv windows, and nothing is shared to order. So the requirement is not
"serialise two writers of one state"; it is the weaker and checkable one below: each sequence is
written by exactly one path per step.

## The ordering a mixed step depends on

1. One entry per sequence per step (the scheduler's walk).
2. Every decode row before the split, every chunk after it. `Scheduler::step` partitions the plan
   into two contiguous runs for exactly this (`core/sched/scheduler.cpp:1507`).
3. One path per sequence, chosen by which half of the split it is in: the recurrent path over the
   decode rows, the chunked scan over the prefill rows (`rad::batch_split`, and the GDN blocks,
   which keep decode-only sequences off the chunked path because it commits state without a
   rollback operand).
4. A checkpoint names a token of this step, and the builder indexes it from the entry's base row
   after the sort, so a mixed step snapshots the right token (`core/sched/batch.cpp:951`).
5. `commit()` leaves a mixed step's decode rows the replay factors a decode step gets, which is
   only true because they are in the decode half (`core/sched/scheduler.cpp:1841`).

Item 2 is what makes 3, 4 and 5 hold, and nothing downstream of the builder could repair it: a
decode row left in the prefill half is scanned without `num_accepted`, so its rejected drafts stay
folded into the recurrent state and the conv window, and it verifies one token instead of several.
A chunk moved into the decode half is stepped as one token. Neither shows afterwards in the batch's
shape, which is why the batch builder refuses a plan whose entries do not agree with its own split
(`StepPlan::split_ordered()`, `core/sched/batch.h:109`; the refusal
`core/sched/batch.cpp:821`). The scheduler satisfies the rule by construction; the check is there
so the next caller of `build()` cannot quietly not.

## The numbers a mixed-step argument needs

`steps_mixed` alone cannot carry an argument about sharing: it says a step carried both kinds and
nothing about how much of either. These joined it (`core/sched/metrics.h`, `/stats` under the same
names, `/metrics` with the `radiance:engine_` prefix and a `_total` suffix):

| counter | what it counts |
|---|---|
| `steps_multi_prefill` | steps carrying two or more chunks, with or without decode rows: prefill against prefill, the concurrency the issue asks about |
| `decode_rows_mixed` | decode rows carried by a step that also carried a chunk |
| `prefill_rows_mixed` | chunks carried by a step that also carried decode rows |
| `prefill_tok_mixed` | prompt tokens computed inside such a step, against `prefill_tokens_total` |
| `prefill_rows_unstaged` | chunks carried by a step too small for the stager to arm (`PrefillStager::begin_pass` arms on the step's whole row count), so every routed layer read its pooled experts across the link inside the GEMMs |

`prefill_rows_unstaged` is the one that explains a loss rather than recording it. Sharing costs a
prefill in two distinguishable ways: a smaller GEMM, which the harness's own control can bracket,
and a step that dropped out of staging, which is a link cost an order of magnitude larger
(`stager.h`: 1.53 ms against 4.34 ms for the same 2048-row gate_up GEMM). If a shared arm loses
prefill throughput with `prefill_rows_unstaged` at zero, the loss is the smaller GEMM. If it loses
it with that counter rising, the arms were never comparing the same thing.

## The A/B that answers it

`scripts/fnmix.sh` runs the arms alternately against one container and prints the figures the issue
names: decode at one stream and at eight, prefill, time to first token of a long prompt, p99
inter-token latency at one stream, the acceptance rate, and the five counters above as deltas over
each arm. Arms, all at the same container, same prompts and same seed:

| arm | flags | the claim it tests |
|---|---|---|
| `off` | `--busy-prefill-chunk off` | the comparator the issue asks for: the same work as separate steps, nothing sharing |
| `auto` | default | sharing by leaving room for whoever is waiting |
| `N` | `--busy-prefill-chunk N` | sharing at a stated ceiling, no floor |
| `isolated` | `--busy-prefill-chunk off --prefill-interleave 2` | chunk isolated, decoders paced, cost paid in TTFT |

Both outcomes close the issue, and the record says which to look for:

- If an arm that shares wins, it wins on decode (p99 at C1, C8 throughput) and its prefill loss has
  to be attributed, by `prefill_rows_unstaged`, to staging or to the smaller GEMM. The winner's
  `N` becomes the documented default and this file records the shape it was measured on.
- If `off` wins on everything the arm changes, the finding is that the overlap worth having is the
  one already there -- weight staging across the chunk boundary -- and the issue closes with that
  and the tables behind it.
- If the arms do not separate beyond run-to-run reproducibility (`scripts/ident.sh` for the noise
  floor), the question is settled for that box and not elsewhere, and this file says so.

## What the source line's sweep already settled

The FullDrafter line measured chunking alone -- no overlap, no shared step -- against a control of
score 541.0, decode 176.5, prefill 5888.8, C8 453.9 (`prefill-chunk-control-sweep.json` in that
campaign's reports; each arm a complete sealed run, preserved in the FullDrafter review pull
requests):

| chunk | score | decode tok/s | prefill tok/s | C8 tok/s | C1 p99 ITL ms |
|---:|---:|---:|---:|---:|---:|
| control | 541.0 | 176.5 | 5888.8 | 453.9 | — |
| 4096 | 528.8 | 180.2 | 5614.2 | 460.5 | 8.09 |
| 8192 | 521.8 | 172.3 | 4786.6 | 467.3 | 8.09 |
| 16384 | 470.4 | 170.2 | 3821.0 | 458.0 | 8.37 |

Prefill throughput fell as the chunk grew and the score fell with it. Two readings follow, and they
are the reason this issue stayed open rather than being implemented on the strength of the idea:

- Chunking without overlap is a cost, not a shortcut. Any overlap proposal has to pay for that cost
  out of what the overlap returns, and it has to be measured against the same control, not against
  an arm that also changed the chunk.
- Decode at C8 went *up* at 4096 and 8192 while prefill went down, which is what sharing predicts:
  the decoders stop queueing behind a long step. That is the effect to isolate. It is also the
  effect the sweep cannot attribute, because it had no mixed-step counters -- a chunk of 4096 in a
  step of its own and a chunk of 4096 sharing a step look identical from the outside. That is what
  `steps_multi_prefill`, `decode_rows_mixed` and `prefill_tok_mixed` are for.

## Status

Development item, unchanged: no policy in this repository makes a chunk overlap. What exists now is
the statement above, the ordering check, the counters, and `scripts/fnmix.sh`. The A/B has not been
run here -- it needs the card and the container -- so the question stays open until someone posts
the arms.
