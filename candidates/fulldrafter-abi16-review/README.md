# FullDrafter ABI-16 source review

This is an **inactive review snapshot**, not an ABI-26 engine integration or a performance release. Nothing here changes Radiance's default build or serving path. The target repository is Radiance 1.10.0 / ABI 26; these sources come from the separate ABI-16 FullDrafter line, including its uncommitted experiments.

The user explicitly requested Draft PRs containing the existing implementation for review rather than a wholesale port of the old engine. Shared runtime and architecture changes remain together so they are not misrepresented as independently applicable patches.

## Contents and provenance

- `source/`: original implementation files, focused native comparators, small synthetic test fixtures, candidate profiles, and selected original measurement artifacts.
- `source.patch`: a generated patch against common ancestor `d0f639bd0620e7ba2cb3672af07b252ebca2005f`, **not** against current `main`.
- `manifest.json`: source revision, byte hashes, file inventory, omissions, recovered corrupt-file provenance, and the complete candidate-accounting matrix.
- Original source HEAD: `7eab082cf8c128a9546300324be4eb5a80e7f940`, plus recorded working-tree changes.

Production model checkpoints, training data, build products, worker transcripts, credentials, local agent configuration, and protected benchmark baselines are not included. The small Safetensors files under `tests/fixtures/*tiny/` are deterministic test fixtures, not a production model.

## Feature accounting

| Area | Existing implementation in this snapshot | Evidence and limitations |
|---|---|---|
| DLoop / FullDrafter | GPU kernels, reference implementations, loader, tree proposals, skinny GEMMs, elementwise/attention/head optimizations and integration sources | Historical source-line measurements and tests are preserved; target ABI-26 execution is NOT RUN. Real serving also needs the external drafter artifact, which is not included. |
| DFlash2 / DarTree | GPU selector, projection/convolution/KV fusions, fixed/scored/pruned MTP tree construction, tree attention and winner commit | Selector profiling showed a local gain; this is not a demonstrated overall BetterBench gain. |
| GDN tree path | Convolution/recurrence/activation-quantization fusion, bounded intermediates, tree state and commit handling | Existing rounding and state contracts must be reviewed against the new engine. Bole variants are experimental, not endorsed winners. |
| Linear MTP coupling | Proposal state, exact coupled verification, cooperative FP64 mass cache, host reference and regression sources | Source oracle tests and native timings exist. The serving comparison did not establish a reproducible throughput gain from the mass-cache change. Seeded trajectories are not equivalent to the ordinary picker. |
| Narrow candidate plan | Optional 64-wide plan with full-width fallback for ineligible request data | Implemented but not built/run successfully as a complete source package before work stopped. No target performance claim. |
| Adaptive bandit | Measured completed-step reward, incoming/outgoing depth attribution and diagnostics | Existing policy is arms 1..5 at most; startup refuses a configured depth of 6 or greater. The requested 2..6 extension was never implemented. No runtime gain is claimed. |
| Shared-expert down / HC up | Existing kernel modules and experimental native comparisons | Original and later variants have different results. Do not conflate the rejected runtime-M gate/up probe with the static-row down kernel. No engine gain established. |
| New private probes | Cooperative proposal, GDN scratch-clear/served-width comparators and other recorded native source experiments | Some were never compiled. They are review material, not production backends. |

The partial QSA `skip_dense_sel` member is preserved only because it occurs in an original integration file. It has no reader and is **not an implemented optimization**.

## Candidate metadata

The modern `gate.json` files are inputs to the old local lever-queue harness, not configuration files consumed by the native engine. Refer to `manifest.json` for all 29 candidate directories, including the historical candidates delivered in the separate legacy-review PR.

`candidates/dflash2-qwen-sorted-down/gate.json` was 77 NUL bytes. It is deliberately omitted, and no replacement behavior or performance result is invented.

The source working copy of `libr4d/r4d_moe.hip` was also corrupt. Its last clean Git-HEAD blob is preserved instead; the exact recovery is recorded in the manifest. This is not a claim that the recovered source reproduces a particular frozen binary.

## Reviewing or reconstructing the old line

Do **not** apply `source.patch` to current `main`. For a separate historical checkout:

```sh
git worktree add --detach /tmp/radiance-abi16-review d0f639bd0620e7ba2cb3672af07b252ebca2005f
git -C /tmp/radiance-abi16-review apply --check /absolute/path/to/source.patch
```

Only after reviewing the patch should it be applied in that disposable checkout. Successful patch application establishes textual reproducibility, not a successful build, kernel parity, model quality, or a gate PASS. Porting live behavior to ABI 26 requires deliberate reconciliation of the runtime, sampler, tree-state and architecture interfaces; copying these older engine files over current files would remove newer functionality.
