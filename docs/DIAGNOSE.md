# the diagnose mode: the engine measures itself

`radiance --diagnose DIR` starts as a serve would — same placement, same kernels, same prefix
cache — runs a fixed workload on itself instead of listening, and writes `DIR/diagnose.json` and
`DIR/diagnose.md`. The flags are in [TOOLS.md](TOOLS.md#the-diagnose-mode); this file is what the
numbers mean, what makes a run a measurement, and what is not measured yet.

## What the run does

`core/engine_diagnose.cpp` is the driver; the arithmetic and the report format are
`core/diag/diagnose.{h,cpp}`, which know nothing about the engine and are tested on the host
(`tests/diagnose_test.cpp`). The driver builds requests with no client, hands them to
`Scheduler::add_all` and reads their tokens back from `server::Sink` — the same path the KL mode
takes. The four points, each run once unmeasured and then `--diagnose-runs` times:

| point | what it answers |
|---|---|
| decode by category | code, reasoning, prose, json, file_edit and summarization prompts to EOS: time per token by prompt shape, draft acceptance, tokens a step |
| concurrency sweep | 1, 2, 4, 8 streams at once (`--diagnose-concurrency`): how TTFT and per-token time give way as the batch fills |
| prefill sweep | a prompt of 1k, 4k, 16k, 64k tokens (`--diagnose-prefill`) then decode at that depth: prefill time, prefill tokens/s, and how deep context changes decode |
| interference | a prompt that arrives while streams are running: its TTFT, and the stall of the streams it joined |

A point the engine cannot run — a concurrency above `--max-num-seqs`, a prompt the context cannot
hold with its decode — is skipped and said so in the report. It is never reported as a zero.

Per phase the report carries, from the scheduler's step ring and its counters: steps, step time
(median, p95, worst), prefill/decode tokens a step, decode tokens/s, draft positions drafted and
accepted, TTFT, per-token emission interval and its jitter, and each request's own tokens. Every
statistic over an empty set is `null` in the JSON and `–` in the Markdown, with the count beside it.

## What makes a run a measurement

| what would spoil it | what the run does |
|---|---|
| a prefix-cache hit — the second run of a point would measure lookups, not work | every prompt carries a tag unique to its run and stream, the cache is emptied before every run, and each request's `n_cached` is checked afterwards. A hit, or step records the ring overwrote before they were read, makes the verdict `NOT A MEASUREMENT`, lists the reason, and exits non-zero |
| the first time a shape runs (tape recording, code objects, pool growth) | each point runs once unmeasured first, in the same shape |
| `ignore_eos` generating past EOS, where a drafter hits differently | categories stop at the model's EOS; only the concurrency and prefill sweeps use `ignore_eos`, so every stream decodes for the whole run |
| a client-side temperature that a deployment never serves | decode runs at temperature 1 — draft acceptance is tuned on it — with the server's top_k/top_p/min_p and a fixed seed per request. Parity does not need greedy: identical logits sample identical tokens at any temperature |
| the environment | the report lists the `RADIANCE_*`, `R4D_*`, `RAD_*`, `AVX_*`, `OMP_*`, `HIP_*`, `HSA_*` and `ROCR_*` variables **by name only** — a value may be a path or a token, and the report is pasted into issues |

The mode warns, in the report, about the settings that make its numbers those of a different
engine: `--profile-ops` (recorded passes off, barriers added), `--no-prefix-cache` (no lookups or
publishing to pay for), `RADIANCE_HEAT_RESIDENCY > 0` (experts over the link), no usable chat
template (raw text, so tokens a step and acceptance are not chat traffic's), and a busy host.

## Parity

`--diagnose-baseline FILE` compares every request's tokens with an earlier `diagnose.json` of the
same workload, and fails the run on any difference. Two reports are comparable only if their
`workload.fingerprint` — corpus, tokens, runs, sweeps, context, template, sampling — is the same;
`compare_outputs` refuses rather than compare a different workload. Points only one side ran are
reported as missing, not as equal. A drop in draft acceptance that every request shares is
reported beside the token comparison, since it is not a correctness difference.

## Sequence

1. **Done, host-tested.** The flag, the driver, the step ring, the report with categories, the
   concurrency and prefill sweeps, and interference.
1b. Placement, expert-streaming and link counters per phase; the dashboard's Diagnose tab reading
   the last report.
2. An issue census: every (op, band, domain, `n`) with calls per step and operand shapes, exported
   for `rad-kbench`.
3. Per-dispatch device time in the play path, joined to the op through a record-to-op table made at
   recording time, evaluated like `libr4d/isa/tsum.sh` and `libr4d/isa/gaps.sh`.
4. A correctness window: repeat and oracle for chosen ops over a step window rather than the whole
   process, with ops that have no `libref` line reported as NOT CHECKED.

Steps 1b to 4 are listed under `not_measured` in every report, so a number that is absent says so
in the artifact itself and never reads as a zero. The mode is host-buildable and its arithmetic is
tested on the host; the numbers it writes on a card are the measurement, and belong with the
hardware they came from.
