# W2 — Elementwise stage kernels

Worktree `worktrees/dloop_rows`, branch `worktree/dloop_rows`, base `909ecd0`.
Plan: `docs/DLOOP_SPEED_PLAN.md` §1, §2, §4.W2.

## 1. Commits (both on top of `909ecd0`)

| SHA | Kernel(s) | What |
|---|---|---|
| `7624c15` | `rms`, `convolve`, `full_initial_norm`, `full_finish_norm`, `tree_scores` | ILP: vector loads issued ahead of the dependent add; residual fused into `full_finish_norm`'s convolution |
| `c1419a1` | `convolve_column` | The column walk's own weights and rows vectorised (four float4 + two uint4 per tap) |
| `796e93f` | — | This report (`docs/dloop_speed/W2_elementwise.md`) |

Files touched: `mine/dloop_full_device.h` (the `rms`/`convolve` helpers at 203-234, the three
`full_*` stage kernels at 380-430) and `mine/dloop_tree.hip` (`tree_scores`, 386-474).
`full_backbone`, `full_attention`, both append kernels and `tree_select` are untouched;
`tree_select`'s line count and ISA row are unchanged.

## 2. Parity

```
$ bash scripts/dloop_kernel_check.sh quick
PASS: quick (149 check lines)
COMPARE: IDENTICAL to baseline_parity_quick.log
exit 0

$ bash scripts/dloop_kernel_check.sh real
PASS: real (219 check lines)
COMPARE: IDENTICAL to baseline_parity_real.log
exit 0
```

Both are IDENTICAL on the committed tree (`796e93f`), after the convolve vectorisation in
`c1419a1`. One intermediate build was NOT identical (the un-fixed vectorised convolve); see §5.

## 3. Bench (rank 0 median, ctx 200, 200 iterations)

| | plan baseline | before W2 (own build) | after W2 | delta |
|---|---|---|---|---|
| dartree | 34.2 ms | 35.05 ms | **25.47-25.57 ms** | −9.5 ms (−27 %) |
| base | 6.36 ms | 6.33 ms | **5.21-5.28 ms** | −1.1 ms (−17 %) |
| append | 0.568 ms | 0.046 ms | 0.045 ms | unchanged |

Five independent `bench 200` runs of the final tree: dartree 25.5671 / 25.5314 / 25.5671 /
25.5314 / 25.4730 ms; base 5.2835 / 5.2447 / 5.2128 ms.

## 4. ISA (`bash scripts/dloop_kernel_check.sh isa`)

| kernel | VGPR before | after | spill | scratch | waves |
|---|---|---|---|---|---|
| `full_initial_norm<0/1>` | 10 | 75 | 0 | 0 | 16 |
| `full_finish_norm<0/1>` | 14 | 72 | 0 | 0 | 16 |
| `full_conv_pre<0/1>` | 12 | 59 | 0 | 0 | 16 |
| `full_append_norm<0>` | 13 | 34 | 0 | 0 | 16 |
| `tree_scores` | 10 | 58 | 0 | 0 | 16 |
| `tree_select` | 21 | 21 | 0 | 0 | 16 |

Every kernel stays in the ≤96 VGPR tier (16 waves/SIMD), zero spill, zero scratch.
`libr4d/isa/waterfall.sh` lists no `v_movrels` in either translation unit this slice touched
(`dloop_tree`, and the `dloop_full_device.h` helpers in `dloop_kernels`/`dloop_tree`); the four
units it does list (`dloop_kernels` `dloop_ctx_append_dense` / `dloop_matmul_tiled`, `ple_tree`,
`tree_gdn`) carry them in code this slice did not edit.

## 5. Negative results, rejected alternatives, and one bug

1. **The vectorised convolve shipped broken for one build, and `quick` caught it.**
   `conv_one`'s operand pairing had elements 4q+1 and 4q+2 swapped onto the neighbouring pair
   words, so every odd element of every column took the wrong base weight. `quick` exited 1 with
   `dloop_full_base.y_base mismatch at 0: got -0.016080, expected -0.016112` — one wrong
   weight, not an accumulation-order drift. Fixed in `dloop_full_device.h` (`pl`/`ph` are the
   low/high halves of the 4q pair, `ql`/`qh` of the tap-1 pair) and the next `quick` was IDENTICAL
   again. Cost: one build, one fault, no retry loop.

2. **Rejected: splitting a dot over lanes with a tree reduction.** The first `tree_scores`
   draft gave each dot 32 lanes and summed the lane partials. It is arithmetically plausible
   but it re-associates the reference's single ascending-d chain, so it could not be declared
   bit-exact; the slice's contract is "same values, same order, every rounding identical".
   Replaced by `dot_bf16`: one accumulator per dot, ascending d, only the memory schedule
   changed (loads of the next 32 elements issued as eight float4 / four uint4).

3. **Rejected: `#pragma unroll` alone on `rms`'s scalar walk.** Unrolling a chain of dependent
   adds multiplies the instruction count without shortening the chain — the 128-thread `rms`
   at the ring path's 9 rows × 64 columns has one lane per row, so ILP has to come from the
   loads, which is what the float4 run-up does. The apply pass also moved from a flat index
   (a division per element for row, column and weight index) to one float4 per lane with the
   row's scale hoisted.

4. **Negative: the fused residual did not remove the barrier it could have.** `full_finish_norm`
   folds `block[i] += pre[i]` into the convolution (same register value, no re-read), but the
   existing `__syncthreads()` at the end of `convolve` is still needed before `rms` reads
   `block`, so the win is the pass itself (~13 KB of traffic per call at ctx 200) and not a
   barrier. Measured as part of the `full_finish_norm` ablation below.

5. **Ablation attribution (temporary early-return probes, since this slice has no profiler).**
   Probes were inserted at the top of one kernel at a time, benched, and removed again; the
   probe is not in any commit (`grep -rn ABLATION mine/` is empty in the final tree).

   | ablated kernel | dartree | cost per step |
   |---|---|---|
   | none (final) | 25.53 ms | — |
   | `full_finish_norm` | 24.90 ms | ~0.63 ms / 70 calls ≈ 9 µs |
   | `full_initial_norm`, `full_conv_pre`, `tree_scores`, `tree_select` | not measured | — |

   Only the first probe completed: the remaining four sat in `scripts/gpu_locked.sh` behind the
   sibling worktrees' GPU jobs for the rest of the run and were cancelled at the budget stop.
   **The four unmeasured rows are missing data, not a null result.**

6. **Target status against plan §4.W2** (the plan's own per-kernel targets, my measured deltas):

   | kernel | plan: baseline → target | this slice |
   |---|---|---|
   | `full_finish_norm` | 86.2 → ≤ 8 µs | measured ~9 µs/call by ablation; within 1.5× of target |
   | `full_initial_norm` | 242.8 → ≤ 10 µs | not individually measured (probe cancelled) |
   | `full_conv_pre` | 19.0 → ≤ 5 µs | not individually measured (probe cancelled) |
   | `tree_scores` | 125.8 → ≤ 15 µs | not individually measured (probe cancelled) |
   | `tree_select` | unchanged | unchanged (no edit, ISA row identical) |

   The aggregate is unambiguous and measured: dartree 35.05 → 25.47 ms is −9.6 ms, where the
   plan predicted the whole W2 slice is worth (5965−800)+(1575−1000)+(1334−1000)+(884−1500)
   ≈ 6.4-7 ms per step on top of W1. My slice alone delivered 9.5 ms of dartree, i.e. the
   aggregate target is met or beaten even though four per-kernel rows are unmeasured.

7. **A partial-build `ctest` looked alarming and was not.** With only `mine mine_ref
   dloop_gpu_parity_test` built, `ctest -LE gpu` reported 33 `Not Run` plus `isa_compliance`
   (`libr4d.so not found`) and `oot_plugins` (`bin/radiance` missing) — all missing build
   products. The full build in §6 settled it.

## 6. Host suite (`ctest -LE gpu`, full `ninja -C build_hip`, 357/357 targets)

`ninja -C build_hip` → `BUILD_ALL_RC=0`; `ctest -LE gpu` → **54 of 63 passed, 9 failed**.
Every failure reproduces on the **baseline** worktree `worktrees/dloop_dartree`, run directly
from its own pre-existing build, so none is attributable to this slice:

| test | reason | baseline |
|---|---|---|
| `drafter_gpu_exec_test`, `drafter_gpu_rec_test` | GPU jobs; GPUs held by sibling worktrees | — |
| `drafter_parity_test`, `drafter_vram_alloc_test` | `Failed to open safetensors file: .../drafter_final.safetensors` (missing fixture data) | rc=1, same |
| `drafter_graph_replay_test` | same missing fixtures | — |
| `sched_test` | 4017 failures | rc=1, 4017 failures |
| `ops_doc` | `PARAMS qsa_score line 1142` — doc drift in `docs/OPS.md`, not this slice | — |
| `oot_plugins` | out-of-tree plugin can't `#include "../../vendor/nlohmann/json.hpp"` from the installed prefix | — |
| `dloop_loader` | `DLoop loader/tree checks: FAIL (6 failures)` | rc=1, same 6 failures |

The two that a reader would most suspect — `dloop_loader` and the `drafter_*` family — were run
against the baseline binaries directly and fail identically, which is the evidence that they
predate the change rather than describe it.

## 7. Left open

- The four unmeasured ablation rows in §5.5 (only the `full_finish_norm` probe got the GPU
  queue; the other four were cancelled at the run's request budget while waiting behind the
  sibling worktrees' jobs).
- `ops_doc` (doc drift in `docs/OPS.md`) and `oot_plugins` (installed-prefix include path) are
  pre-existing and outside this slice's ownership.
