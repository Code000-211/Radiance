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
  window it drafts (`draft_plan_valid` binds `verify_k == max_k` there). `draft_shape_priceable`
  requires both distributions to be monotone and complete, `verify_tokens` to equal
  `n_rows + sum_j rows[j]`, and the deepest round of each to be non-empty: `verify_k > 0` with
  `rows[verify_k] == 0` is a missing shape, not a measured zero.
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
  inventing a reject label. The length and the classification are still one statement:
  `draft_seq_observation_valid` requires `raw_match == accepted` for `Rejected` and `AllAccepted`,
  `raw_match ∈ {accepted, accepted + 1}` for `Stopped`, and `raw_match <= generated` always -- a
  "rejection" whose own tally says the draft matched is refused, not labelled 0.
- **The output budget includes the bonus token.** `remaining` counts the tokens the request may
  still emit, and the verify step always emits the target's bonus token after the accepted prefix.
  Drafts are therefore bounded by `remaining - 1` (`draft_budget_k`); with one token left the
  request drafts nothing. A plan that clamps at `remaining` overruns the output budget by one
  token per verify step.
- **Units are milliseconds or nothing.** An unmeasured shape is `valid = false` or
  `DraftCostSource::ConservativeFallback` with a bound; a unitless prior is never reported as ms.
- **Only cleanly attributed samples update a measured play bucket** (`draft_sample_attributable`):
  the sample names the cycle and policy version the observation names, has a finite positive
  bracket, `kDraftSampleValid`, and no warmup, record, audit, admission or placement flag. A
  sample with `cycle_id = 0` is unattributed and never accepted; a non-finite or non-positive
  total is not a duration. Real DMA cost is routed away by its flag, not discarded as an outlier.
  An `ahead` sample stays flagged so the model can weigh it. The observation overload also requires
  a completed cycle over a plan that passes `draft_plan_valid`: an unfinished cycle has no bracket
  to fold in, and a plan that fails its own gate names no shape the sample could have been measured
  under.
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

## Draft profile format, version 2

`core/sched/draft_cost.cpp` (A) reads and validates this; the parser never runs on the step path.
The format id is `rad-draft-profile/2` (`RAD_DRAFT_PROFILE_VERSION`). A file with another id —
including `rad-draft-profile/1`, whose key had no draft distribution, origin or pipeline depth — is
refused by name, not reinterpreted under the current keys.

```json
{
  "format": "rad-draft-profile/2",
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
        "pipeline_depth": 0,
        "rows": [0, 8, 8, 8],
        "draft_k": 3,
        "draft_rows": [0, 8, 8, 8],
        "origins": ["head"],
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
  kernel/build stamp and KV format/placement match. A mismatch disables the profile and the policy
  falls back to the known fixed configuration, naming the mismatch in the log and in the plan.
- **`execution`.** `rectangular` (all N_j equal) and `shrinking` (E2: per-request caps, N_j falling
  with j) produce different kernels and times, so they are different profiles. Changing the
  execution invalidates the old file rather than rescaling it.
- **`shape` is the lookup key.** It is the verify token count and length distribution
  (`n_rows`, `verify_k`, `verify_tokens`), context class, history tokens, prefill tokens, `N_j`,
  sampling class, replay state and the loaded drafter. The planned draft distribution
  (`draft_k`, `draft_rows`) and the pipeline depth are part of the key too: the same `rounds` and
  total rows can shrink differently, and an issued-ahead cycle prices a different pair of
  distributions than a synchronous one. Origins name the mix of rows (`head`, `lookup`); a head
  row does not price a lookup row. `T(B, maxK)` alone is not a key, and K0 with the head loaded is
  not the same bucket as K0 with no drafter process.
- **Version 2 is not version 1 with new fields.** A version-1 file has no `draft_k`/`draft_rows`,
  no `pipeline_depth` and no origin mix, so its buckets cannot be mapped onto the current key by
  assuming zeros. The reader refuses it by format id; a profile is regenerated, never rescaled.
  The parser also refuses a bucket whose `draft_rows` disagrees with `rows` where the two
  distributions are the same execution, and a shape carrying an `unknown` dimension.
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

**Build prerequisite, separate from P00.** A whole-tree clang 23 build needs branch
`build/clang23-libavx` (from `cb5a69a`), which P00 does not contain:

| Commit | What |
|--------|------|
| `dcb4e6f` | libavx: a duplicated `target` attribute, and a bf16 intrinsic in an OpenMP-outlined loop |
| `c04006e` | `-Wl,--build-id`: kbench names a plugin by its ELF build-id note, which clang does not add itself |
| `1ef93d2` | `core/device/aql.cpp`: prefer ROCm's `kfd_ioctl.h` (1.16) over the host kernel's 1.11 |

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
  pass 5/5 (`draft_types_test`: 364 checks over 29 cases after the review corrections, 310/23
  before them).
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
