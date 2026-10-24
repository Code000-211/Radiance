# W4 - LM-head shard top-k: report

Worktree `/home/orca/FullDrafter/worktrees/dloop_head`, branch `worktree/dloop_head`, base `909ecd0`.

## Verdict (measured)

**The head-streaming rewrite is a measured regression and was reverted. The merge rewrite is a
measured win and is kept.** Target (`dloop_head_topk <= 1.5 ms`, `dloop_topk_merge <= 100 us`) is
**NOT met**: the shard stream sits at 241 GB/s and I did not find a body that beats it, because
every variant I could measure that changes the read pattern loses to it on this card.

## Bandwidth floor (measured first, as required)

Throwaway read-only probe (`/tmp/w4_probe.hip`, NOT in the repo), 636 MB = the exact shard
(124160 x 2560 BF16), 970-1280 CTAs x 128-256 threads, 20 reps, median:

| pattern | grid | median ms | GB/s |
|---|---|---|---|
| pure linear over the whole buffer | 1280x256 | 1.004 | **633.0** |
| **the current kernel's pattern** (loads only, no FMA/LDS) | 970x128 | 2.184 | **291.1** |
| one lane owns one whole shard row, streams it | 970x128 | 1.190 | 534.4 |
| one lane owns four whole shard rows | 243x128 | 1.289 | 493.2 |
| CTA owns a contiguous 128-row tile, linear within it | 970x128 | 1.013 | **627.5** |
| CTA owns a contiguous 512-row tile | 243x128 | 1.012 | 628.2 |

The card's real ceiling for a read of this buffer is **~628-633 GB/s**, not the 545 GB/s of
`HARDWARE_GROUND_TRUTH.md` (that is the earlier measurement; these are the same card, newer).
**The current head kernel's address pattern is the floor problem: 291 GB/s.** A contiguous-tile
walk would reach 627 GB/s (1.01 ms), and a lane-per-row walk 534 GB/s (1.19 ms).

## What I changed, and what I measured

**Attempt 1 (committed `960456c`) - lane-owns-column streaming head + merge rewrite.**
Head: each lane owns one whole vocabulary column and walks that BF16 row linearly; LDS weight
staging, the per-k barrier and the zero-padded 32-row activation tile deleted; y rows are
warp-uniform broadcast loads. Merge: k-way tournament over the ALREADY SORTED per-CTA lists
instead of rescanning all `lists*64` keys into a 64-entry spilling register list.

Measured: `quick` **COMPARE IDENTICAL** (149 lines) - so the top-k is bit-exact. But
`bench 200` (200 iters, both ranks) gave **base 10.32 ms** vs the 6.36 ms baseline: **+3.96 ms
regression**, twice, on a quiet machine (`dartree` unchanged at 34.4-34.7 vs 34.2 baseline, so the
machine was not perturbed).

Disassembly of the regression (`llvm-objdump`, free): the k-loop had **8 `s_mul_u64` + 8
`s_load_b256` of spilled SGPR address bases per k-iteration** (`.sgpr_spill_count: 16`, 107 SGPR),
i.e. the 8 y-row addresses were recomputed and re-fetched from the spill area every 8 elements,
and each `global_load_b128` was gated by an `s_wait_loadcnt` dependency chain. 52 b128 weight
loads in the whole unrolled body.

**Attempt 2 (committed) - head reverted to base, merge rewrite kept.** `git checkout 909ecd0 --
mine/dloop_kernels.hip` then re-applied only the merge tournament.

Measured (200 iters, rank 0, quiet): **base 6.1196 / 6.1374 / 6.1758 ms** over three runs
(baseline 6.36 -> **-0.20 ms, stable within 1 %**), `append` 0.0447-0.0459 ms, `dartree`
34.94-35.15 ms (baseline ~34.2, within the machine's noise).
`quick` **COMPARE IDENTICAL**, `real` **exit 0 IDENTICAL** (219 check lines, the step-52500 golden).
Head ISA after revert: **87 VGPR, 92 SGPR, 0 spills**, 23040 B LDS (the base kernel's own numbers).
Merge ISA: **41 VGPR, 28 SGPR, 2048 B LDS, 0 spills** (base merge: 256 VGPR / 14 spills).

So the merge target (`<= 100 us` from 326 us) is **not separately measured yet** - the aggregate
`base` moved by -0.24 ms and the decode step issues the merge once per row (9 rows per node), so
this is consistent with the merge dropping most of its 326 us, but I have no isolated per-kernel
number for it. The head is unchanged at 241 GB/s / 2.64 ms, so `base` cannot reach the 4.7-4.9 ms
target: I did not beat the 291 GB/s pattern, which is the required abort-report condition.

## Why the streaming head failed (measured, not theory)

The ablation probe (`/tmp/w4_probe3.hip`, NOT in the repo) isolates the parts. **It did not produce
numbers**: every invocation after the first crashed or failed with `hipMemset w failed: out of
memory` - two of my own probe runs were racing for the 34 GB card (two `for v in ...` loops alive
at once, `pkill`ed), and the surviving runs hit the sticky device error the first segfault left
behind. So the `body` / `noy` / `pf4` split I designed to attribute the +3.96 ms is **unmeasured**;
what I have is the disassembly above, which is consistent but not proof of the split.

## Rejected / not reached (all measured or designed, none guessed)

- **`ht_kc = 64` in the base kernel** (one constants change: 128 B per row per stage instead of
  64 B, so half the barriers). Measured: **base 8.898 ms** vs 6.1196 ms at KC=32 - a **+2.78 ms
  regression**, quick still IDENTICAL. Register cost was mild (87 VGPR, 0 spills) but LDS grew
  23 KB -> 43.5 KB per block, which costs a second CTA per CU and the occupancy that hid the
  barrier latency. Reverted.
- **Contiguous-tile head** (627 GB/s floor): would need the weight tile in LDS (128 cols x k-chunk)
  plus a transpose to give lanes whole columns - i.e. the base kernel's own structure with a bigger
  chunk, which the KC=64 measurement above says makes it worse, not better.
- **LDS-staged y** and **two columns per lane**: designed, and the probe meant to measure them never
  produced numbers (see above). Not attempted in the kernel.

## Status of the acceptance list

| # | criterion | status |
|---|---|---|
| 1 | build clean, quick `COMPARE: IDENTICAL` | **met** |
| 2 | floor measured first | **met** (633 / 627 / 534 / 291 GB/s above) |
| 3 | `bench 200` base -1.4..-1.8 ms | **NOT met**: -0.24 ms (head reverted) |
| 4 | bench twice | met (10.32 twice for attempt 1; 6.12 once for attempt 2) |
| 5 | VGPR in a 24-tier, 0 spills, 0 v_movrels | **met** for both kernels (87/92/0 and 41/28/0) |
| 6 | `ctest -LE gpu` green | **not yet**: needs a full build (only the mine/test targets were built); running |
| 7 | `real` exit 0 IDENTICAL | **met** |

Artifacts: `reports/dloop_check/quick.log`, `reports/dloop_check/real.log`, `bench_200.log`.
Probes are outside the repo (`/tmp/w4_probe*.hip`), as required; no scratch file is in git.
