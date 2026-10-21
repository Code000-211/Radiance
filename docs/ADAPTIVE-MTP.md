# Adaptive MTP: contracts, profile and ownership

This is the working agreement for the adaptive-MTP work on Qwen3.8-Flash-Next, TP2 on two Radeon
AI PRO R9700. `spec.md` remains the binding design; this file records the internal seam (P00), the
versioned profile and benchmark-manifest formats, the CLI names, and which files each strand owns.

**Status: P00 only.** The contracts and their fakes exist and are tested on the host. No cost
model, predictor, policy, flag or telemetry is wired into a running engine, and no M1/M2 or GPU
result has been measured or is claimed here.

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
  counted after the committed token; the first target is K = 0..6. `draft_plan_rows` and
  `draft_plan_valid` refuse `n_seq > cap` before reading `k[]`, so a caller cannot make the
  contract check read past the storage it sized.
- **A plan is immutable from issue until every pass it names is consumed.** `choose` writes it
  once; no other hook writes into it, and `observe` takes only const inputs. Historical plans stay
  identifiable while `issue_ahead()` uses them; the scheduler sizes its ring for the live cycles
  in advance, and a telemetry overflow drops and counts records rather than blocking the card.
- **Identity is `(cycle_id, policy_version, req_id, slot, slot_gen)`.** A recycled slot is another
  request, and `draft_plan_identifies(plan, observation)` is the check that the rows a hook is
  handed came from the plan they name — the observation carries both the identity arrays and the
  plan pointer, and either one alone can be stale. One memory set per live cycle: a plan is not
  rewritten while an observation or an issued-ahead pass still names it.
- **The current verify-K (`verify_k`) and the next draft-K (`max_k`) are separate fields:**
  reading one for the other is how a proposal from another generation leaks into a step.
- **`k[]` is the budget for every drafter.** The plan names per request who may fill it (`fill[]`):
  `None` exactly where K is 0, the serial head or the n-gram fallback where K is positive. A
  request the plan puts at K 0 stays empty; the scheduler's n-gram fallback is a drafter too and
  may not fill it (`draft_plan_bars_fill`). The view's batch-wide `ngram_allowed` is the gate the
  plan is chosen under, not a licence to override the plan.
- **The plan carries the shape it was priced on.** Per request `ctx_class` and `history_tokens`,
  cycle-wide `prefill_tokens`, `replay` and `timing_valid`. A historical plan kept alive by
  `issue_ahead()` can then still say what it priced, and a K0 plan with `history_tokens > 0` is
  "policy K0 with history", not "no work".
- **Verify shape and draft extent are separate.** `DraftVerifyShape` keys the cost model on verify
  tokens, context class, history tokens and `N_j` (requests reaching round j), not just
  `(rows, max K)`. `DraftCostSample` carries the rounds and rows actually executed beside it.
- **Labels only where measured.** `draft_label` returns a label for the accepted prefix (1) and
  the first real rejection (0), and nothing after that: a position the target computed past the
  first rejection was not measured, even when `checked` covers it. EOG, `max_tokens`, cancellation
  and the context end are `DraftCensor` reasons: they cut `checked`, and the cut is what keeps the
  censored tail out — the reason names why, it is not itself a position. Raw match length, commit
  length, and the censor reason are separate fields.
- **Units are milliseconds or nothing.** An unmeasured shape is `valid = false` or
  `DraftCostSource::ConservativeFallback` with a bound; a unitless prior is never reported as ms.
- **Only cleanly attributed samples update a measured play bucket** (`draft_sample_attributable`):
  the sample names the cycle and policy version the observation names, has a finite positive
  bracket, `kDraftSampleValid`, and no warmup, record, audit, admission or placement flag. A
  sample with `cycle_id = 0` is unattributed and never accepted; a non-finite or non-positive
  total is not a duration. Real DMA cost is routed away by its flag, not discarded as an outlier.
  An `ahead` sample stays flagged so the model can weigh it.
- **Non-play buckets have their own path.** A record or audit bracket is stored through
  `draft_sample_storable` with its flag, in the bucket that names it, and never folded into a play
  bucket; there is no flag combination that makes it attributable.
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

The first two exist today and keep their meaning. The rest are **defined here but not accepted
yet**: Welle 2 adds them when they have an effect. A flag that parses and does nothing would be the
fabricated success the contracts forbid.

| Flag | Values | Default | State |
|------|--------|---------|-------|
| `--num-speculative-tokens` | N / `auto` | `auto` | existing; 0 = no drafter process |
| `--draft-window` | `fixed` / `adaptive` | `fixed` | existing |
| `--draft-policy` | `legacy` / `cost` | `legacy` | `cost` refused with `RAD_E_UNSUPPORTED` until A+C land |
| `--draft-policy-shadow` | switch | off | cost choice logged, not applied; with C |
| `--draft-profile` | path | none | profile reader lands with A |
| `--draft-depth-tiers` | ascending unique K list, e.g. `0,2,3,4,6` | `0,2,3,4,6` | must start at 0 if K0 is allowed, end ≤ ceiling |
| `--draft-explore-rate` | R in [0, 1] | `0.01` | at most 1% of eligible decode cycles |

`cost` is accepted only with a serial drafter and a valid ceiling contract; unsupported
combinations are refused by name, never silently mapped to `legacy`. Block drafters are unchanged.
The K0 rules are in the plan: K0 with the drafter loaded stays a temporary operating state with
weights, buffers and arena geometry unchanged (history for the MTP KV/QSA keeps being maintained),
and it is measured separately from a process started with `--num-speculative-tokens 0` and no
drafter. Initialisation is explicitly not depth 0. A speculation-off plan is never filled by the
n-gram fallback: its `fill[]` is `None` for every request and `draft_plan_bars_fill` says so;
"MTP off, n-gram allowed" would be a separate state with its own costs and labels.

## Draft profile format, version 1

`core/sched/draft_cost.cpp` (A) reads and validates this; the parser never runs on the step path.
The format id is `rad-draft-profile/1` (`RAD_DRAFT_PROFILE_VERSION`). A file with another id is
refused by name, not reinterpreted.

```json
{
  "format": "rad-draft-profile/1",
  "fingerprint": {
    "model": "Qwen3.8-Flash-Next",
    "quant": "gguf-q4_k_m",
    "tp": 2,
    "tp_wire": "wht6",
    "gpus": "2x Radeon AI PRO R9700",
    "kernels": "libr4d <commit>+gfx1201",
    "kv_format": "bf16",
    "placement": "all_vram"
  },
  "execution": "rectangular",
  "buckets": [
    {
      "ctx_class": 0,
      "shape": {
        "n_rows": 8,
        "verify_k": 3,
        "verify_tokens": 32,
        "history_tokens": 0,
        "prefill_tokens": 0,
        "rows": [0, 8, 8, 8],
        "sampling": "greedy",
        "replay": "play",
        "drafter": "serial"
      },
      "extent": { "rounds": 3, "draft_rows": 24 },
      "time_ms": { "mean": 43.0, "unc": 0.6, "n": 5 },
      "components_ms": { "target": 25.9, "draft": 12.1, "host": 4.2, "valid": true },
      "flags": ["ahead"]
    }
  ]
}
```

- **Fingerprint.** A profile is valid only when model, quantisation, TP wire and rank count, GPUs,
  kernel/build stamp and KV format/placement match. A mismatch disables the profile and the policy
  falls back to the known fixed configuration, naming the mismatch in the log and in the plan.
- **`execution`.** `rectangular` (all N_j equal) and `shrinking` (E2: per-request caps, N_j falling
  with j) produce different kernels and times, so they are different profiles. Changing the
  execution invalidates the old file rather than rescaling it.
- **`shape` is the lookup key.** It is the verify token count and length distribution
  (`n_rows`, `verify_k`, `verify_tokens`), context class, history tokens, prefill tokens, `N_j`,
  sampling class, replay state and the loaded drafter. `T(B, maxK)` alone is not a key, and K0
  with the head loaded is not the same bucket as K0 with no drafter process.
- **`ctx_class`.** 0 = below 4K, 1 = 4K–32K, 2 = 32K–128K, 3 = 128K and above
  (`draft_ctx_class`). Profiles only carry the classes the configured capacity can reach.
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

**Compiler.** Every build and compiled test — P00 and strands A–F — uses the compiler the engine
ships with, ROCm's clang at `${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++` (the compiler `hipcc`
drives). A system compiler is not a substitute; if the toolchain is missing, the blocker is
reported rather than worked around. The intended entry point names it: `./build.sh` passes both
compiler paths itself and refuses to configure without them, and the `default` CMake preset sets
the same pair. Name it explicitly, in a build directory of its own so a differently compiled tree
cannot be mistaken for this one:

```bash
/opt/rocm/llvm/bin/clang++ --version   # the proof: path and version beside the result

./build.sh -b build-mtp-clang --no-hip   # configure + build + ctest -L fast
ctest --test-dir build-mtp-clang --output-on-failure -LE 'gpu|oot' -j 8
```

The P00 result recorded on 2026-10-22 used `/opt/rocm/lib/llvm/bin/clang++` (AMD clang 23.0.0git,
ROCm llvm-project `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, installed as ROCm 10.0.0): the five
suites `draft_types_test`, `sched_test`, `sample_test`, `runtime_test`, `config_test` pass 5/5,
and the broad host suite (`-LE 'gpu|oot'`, 104 tests) passes 99/104. The five remaining failures
are named in the issue: two need FFmpeg (`images_test`, `videos_test` — this host has no
libavformat), two exceed a CPU budget on this slower host (`gbnf_test_shard2`, `tokenizer_test`),
and `gbnf_test_shard3` is a pre-existing grammar failure that reproduces in an independent clang
build of a later commit. None of them touches `core/sched/`.

The test binary is picked up by the existing `*_test.cpp` glob; no CMake edit is needed.

The linker now states `--build-id` explicitly, which Debian's GCC driver passed by itself and
clang's does not, so rad-kbench keeps a plugin identity whichever compiler links. The two libavx
incompatibilities clang 23 has with GCC-built kernel sources (a duplicated `target` attribute and a
bf16 intrinsic inside an OpenMP-outlined loop) are **not** part of P00; they are on branch
`build/clang23-libavx` (commit `dcb4e6f`, from `cb5a69a`) and must land for a whole-tree clang build.
