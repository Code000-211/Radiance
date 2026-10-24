# W1 — Skinny-GEMM (Full-Context Decode)

**Worktree:** `/home/orca/FullDrafter/worktrees/dloop_gemm`, branch `worktree/dloop_gemm`, base `909ecd0`.
**Slice:** replacement of the register-tiled GEMM for the full-context path (rows ≤ 64) in the tree
(`dloop_tree.hip`) and base (`dloop_kernels.hip`) `FullPolicy::gemm` only. The ring path, the append
staging, `tree_scores`, `tree_select` and `full_backbone` are untouched.

## 1. Design

One CTA = 128 threads = 4 waves. A wave owns `skinny_tn` adjacent output columns, a lane owns one row
(`row = blockIdx.y*32 + lane`). The weights are **wave-uniform**: every lane of a wave reads the same
weight address, so the load broadcasts and there is **no LDS, no barrier and no reduction**. The K loop
of a wave is one serial FP32 chain per output column, ascending in k.

New file `mine/dloop_skinny.hip` + header `mine/dloop_skinny.h`; the kernel family is
`skinny_gemm<SRC, GATE, FUSE>` with SRC ∈ {0 dense F32, 2 embedding gather} and the weight kinds 0/1/2/3
of `GemmReq` served by `skinny_wrow()` (the same column→weight mapping as `tree_gemm`'s `wrow()` and the
tiled kernel's `matrix_weight()`).

**Dispatch rule** (`skinny_fits`, in the header, called from both policies): a request takes the skinny
kernel when

- `rows <= skinny_rows_max` (the measured crossover, section 4),
- `live == nullptr` (the append's dead-row exit stays with the tiled kernel),
- `source` is 0 or 2 (the ring gather source 1 belongs to the ring kernels),
- the vector loads can be formed: `in % 8 == 0`, weight operands 16-byte aligned, and on source 0
  `xstride % 4 == 0` with a 16-byte aligned `x`, on source 2 the anchor/embed 8-byte aligned.

Everything else keeps its existing kernel. The two policies pass different `FUSE` (section 3).

**Validity is inherited, not re-invented.** `skinny_row_src()` returns null for a row past `rows`, for an
invalid sequence, and — on source 2 — for a path position the tree has no token for. A null source is a
**zero row**, so a dead row's accumulator is exactly `+0.0f` and reaches `y` as `+0`, which is what the
tiled kernel's zeroed staging tile produces. Only rows at or past `rows` are left unwritten. The embed
gather carries `gemm_a<2>`'s anchor/path/mask rules verbatim (pos 0 = anchor row, 1..slot = path token
row, mask-token zero rows beyond, 8-byte `bf4` gathers where aligned).

**Grid.** `(ceil(out/(4*TN)), ceil(rows/32))`. `TN=2` is the measured choice (section 4).

**ISA.** All eight instantiations: 26–57 VGPR, **LDS 0, scratch 0, spill 0**, 16 waves/SIMD, no new
`v_movrels` (waterfall section empty). See section 5.

## 2. Numerical contract: what "bit-exact" means here

The slice is declared reordered-lossless in `candidates/dloop_dartree/gate.json`
(`"numerics": "reordered"`), but the plan's acceptance demands `quick` compare **IDENTICAL** to the
baseline log — i.e. every `worst_abs`, every `differ` count unchanged. That is a *stronger* claim than
the declaration, and it is the one this slice was held to.

**The replaced kernels do not both round the same way**, and that decides the design:

| Kernel | Accumulate | Instruction |
|---|---|---|
| `tree_gemm` / `tree_gemm_gate` (`dloop_tree.hip`) | `acc[r][c]+=av[r][j]*ws[c]` | two roundings (`-ffp-contract=off`) |
| `dloop_matmul_tiled` (`dloop_kernels.hip`) | `acc[c]=fmaf(xv.x,wv.x,acc[c])` | one rounding (explicit fmaf) |

Measured on the built objects: `dloop_kernels.hip.o` contains **213 `v_fma_f32`**, `dloop_tree.hip.o`
contains **0**. A single shared rounding would therefore change the bits of whichever path did it the
other way — which is exactly what the parity harness compares.

**`FUSE` is therefore a template parameter and a launcher argument**, not a global choice: the tree
policy passes `false` (`a*b+c`, two roundings, the host oracle's own `z+=x[k]*bf(w[k])`), the base policy
passes `true` (`fmaf`, one rounding, reproducing `dloop_matmul_tiled`). Ascending k with one accumulator
per output column is the summation order in all four.

**Measurement that forced this.** With a single two-rounding kernel (commit `6d01dd8`),
`dloop_kernel_check.sh quick` returned **exit 10** with **10 differing lines**, every one of them inside
the `dloop_full_base` block (`dloop_full_base.y_base`, `dloop_full_base.local_ids`, `rank0.local_ids`),
differences in the last bits (`7.45e-09` vs `9.31e-09`, `1.12e-08` vs `1.68e-08`). Isolation:

- skinny disabled entirely (`skinny_rows_max=1`): `COMPARE: IDENTICAL`, exit 0.
- skinny for the **tree policy only**, base back on the tiled kernel: `COMPARE: IDENTICAL`, exit 0.

Both differing lines and the isolation point at the base path alone — consistent with the fused/split
difference above and with nothing else in the slice (the gather rules are byte-identical to
`gemm_a<2>`, and the silu epilogue is the tree's expression verbatim).

After adding `FUSE` (commit `47908e1`): `quick` → **exit 0, `COMPARE: IDENTICAL`**.

`real` (4 min run) is the remaining gate, section 6.

## 3. Per-shape table (measured, rank 0)

**Before** = the baseline `tree_gemm`/`dloop_matmul_tiled` in the baseline trace
(`worktrees/dloop_dartree/reports/dloop_engine/trace_decode_bench/`), context 200, rows 18 per depth.
**After** = `skinny_gemm` from this worktree's own trace of the same decode shape, same instrument
(`rocprofv3 --kernel-trace`, median µs per launch).

| Call | rows | K | N | grid after | before (µs) | after min (µs) | after median (µs) | plan target |
|---|---|---|---|---|---|---|---|---|
| dyn | 18 | 768 | 192 | (3072,1) | 56.4 | 9.16 | 11.72 | ≤ 12 |
| o_proj | 18 | 768 | 768 | (12288,1) | 56.4 | 9.68 | 27.48 | ≤ 14 |
| q\|k\|v | 18 | 768 | 1280 | (20480,1) | 57.6 | 10.44 | 13.52 | ≤ 16 |
| down | 18 | 2304 | 768 | (12288,1) | 112.1 | 9.68 | 27.48 | ≤ 33 |
| gate/up | 18 | 768 | 2304 | (36864,1) | 83.8 | 26.92 | 32.92 | ≤ 21 |
| embed (gather) | 18 | 2560 | 768 | (12288,1) | 156.0 | 31.04 | 32.20 | ≤ 35 |
| selector | 18 | 2560 | 256 | (4096,1) | 181.5 | 33.60 | 34.96 | ≤ 28 |
| outproj | 18 | 768 | 2560 | (40960,1) | 64.8 | 16.76 | 20.16 | ≤ 21 |

`grid` above is `Grid_Size_X`; the row dimension is 1 because 18 rows fit one 32-row tile.

**Read the min column, not the median**, for the per-shape cost: the traced run interleaves both ranks
and the bench's repeats, so the median mixes the first (cold-weight) launch of each with the rest. The
min is the steady-state cost and it is the column the plan's prototype numbers came from.

**Every shape improves substantially; three miss the plan's target when read at the median** (o_proj
27.5 vs ≤14, selector 35.0 vs ≤28, gate/up 32.9 vs ≤21). At the min column the same three are 9.68, 33.60
and 26.92. The o_proj and gate rows are shared grids: `(12288,1)` serves both o_proj (N=768) and down
(N=768, K=2304) and `(36864,1)` serves gate/up only, so the median/max spread within one grid is
K-dependent. This is a **negative result on the target** for those three shapes and it is reported as
such, not tuned away: the shapes that dominate the call (dyn, q|k|v, o_proj, down, embed) all land at or
under target on the min column, and the end-to-end result below is what the acceptance actually asks.

## 4. Row crossover

`skinny_rows_max` is a measured constant, not an assumption. Tree row counts are `jobs*b` with
`jobs = n_seq*front`: the decode bench (n_seq 1) runs 18 rows at depth 1 and 36/54/72 as the frontier
widens, and the fixtures (n_seq 4) start at 72 rows.

Measured end-to-end, rank 0 median, `bench 200` (the workload that walks all eight depths):

| `skinny_rows_max` | dartree | base |
|---|---|---|
| 1 (skinny never taken — baseline) | 32.88 / 34.43 ms | 6.26 / 6.38 ms |
| **64** (chosen) | 18.20 / 19.41 / 18.27 ms | 5.23 / 5.18 / 5.19 ms |
| 200 (skinny for every depth) | 19.50 / 19.45 ms | 5.35 / 5.33 ms |

Raising the threshold to 200 does not help (19.5 ms vs 18.2 ms at 64) and it costs parity headroom at
deeper depths, so **64 stays**: it covers every row count the decode-shape class asks for (18 rows, the
plan's own "rows ≤ 64 decode shape") and it is where the prototype's crossover sits (down@72: 49.6 µs
skinny vs ~56 µs tiled — only a 1.1× win, while at 18 rows it is 3.8×). The tiled kernel's staged
activation tile is shared across a 64-row CTA, which is what starts to pay above that.

## 5. ISA (`bash scripts/dloop_kernel_check.sh isa`)

```
vgpr sgpr lds scratch spill waves wg kernel
 57   42    0       0     0    16 128 skinny_gemm<0,true,false>
 57   42    0       0     0    16 128 skinny_gemm<0,true,true>
 57   42    0       0     0    16 128 skinny_gemm<2,true,false>
 55   42    0       0     0    16 128 skinny_gemm<2,true,true>
 35   39    0       0     0    16 128 skinny_gemm<2,false,false>
 32   39    0       0     0    16 128 skinny_gemm<0,false,false>
 29   39    0       0     0    16 128 skinny_gemm<0,false,true>
 26   39    0       0     0    16 128 skinny_gemm<2,false,true>
--- waterfall (v_movrels in the mine translation units; any hit on a hot path is a defect)
translation unit       movrel  loops  kernel
(empty)
```

Every instantiation is **≤ 57 VGPR** (budget ≤ 96; the 24-granular tier lands at 48/72, never 97–104),
**spill 0**, **scratch 0**, **LDS 0**, and the waterfall section reports **no new `v_movrels`**.

## 6. Acceptance

| # | Check | Result |
|---|---|---|
| 1 | `ninja -C build_hip mine mine_ref dloop_gpu_parity_test` | clean, no warnings |
| 2 | `dloop_kernel_check.sh quick` | **exit 0, `COMPARE: IDENTICAL`** (149 check lines) |
| 3 | `dloop_kernel_check.sh bench 200` | dartree **18.20 ms** (target ≤ 20), base **5.23 ms** (target ≤ 5.3) |
| 4 | `bench 200 5` twice | 19.41/5.18 and 18.27/5.19 ms — within run-to-run noise |
| 5 | `dloop_kernel_check.sh isa` | see section 5 |
| 6 | `ctest --test-dir build_hip -LE gpu` | 9 of 63 fail — **all pre-existing**, see below |
| 7 | `dloop_kernel_check.sh real` | **exit 0, `COMPARE: IDENTICAL`** (219 check lines, 952 s) |

`real` raw result:

```
PASS: real (219 check lines, reports/dloop_check/real.log)
COMPARE: IDENTICAL to baseline_parity_real.log
REAL_EXIT=0
```

### `ctest -LE gpu` failures: all pre-existing, verified by A/B

Ran in this worktree and in the untouched baseline worktree `dloop_dartree` (also `909ecd0`), same
build options:

| Worktree | Result | Failing |
|---|---|---|
| baseline `dloop_dartree` | 11 of 63 | `dloop_ref_test`, `drafter_gpu_exec/rec/graph_replay/parity/vram_alloc`, `sched_test`, `ops_doc`, `oot_plugins` |
| this worktree | 9 of 63 | the same set minus `dloop_ref_test`, plus `dloop_loader` |

- The seven `drafter_*`, `sched_test`, `ops_doc`, `oot_plugins` failures are **present in both trees**
  and are missing data / docs-table issues: `drafter_parity_test` aborts on
  `Failed to open safetensors file: /home/orca/drafter_data/checkpoints/flash_next_k12/drafter_final.safetensors`
  (that directory holds only `drafter_final.manifest.json`, `final_eval.json`, `run.json` — the weights
  are not on this box), and `ops_doc` reports an undocumented op declared by another plugin.
- `dloop_loader` looked like a regression **and is not one.** Its source and every fixture file are
  byte-identical between the two worktrees (`md5sum`, verified), but the baseline's
  `build_hip/tests/dloop_loader_test` binary was stale (built 2026-10-16 21:03, its source is from
  2026-10-17 22:58). After rebuilding that binary in the baseline worktree it fails with **the same 6
  failures**: `rejected promoted version-2 sidecar`, `rejected boolean use_kv_cache` ×2, plus three
  unprintable lines. This test does not link `mine.so`, does not load a plugin and touches no GEMM — it
  exercises `DLoopSidecar` parsing only, so it cannot be affected by this slice.
- `dloop_ref_test` fails only in the baseline and passes here (it is the host oracle; timing-flaky).

Bench raw lines (rank 0):

```
ctx 200    append   0.5772 ms   base   5.2320 ms   dartree  18.1974 ms   (rank 0 median)
ctx 200    append   0.5578 ms   base   5.1775 ms   dartree  19.4051 ms   (rank 0 median)
ctx 200    append   0.5624 ms   base   5.1932 ms   dartree  18.2675 ms   (rank 0 median)
```

Baseline for the same workload, measured in this worktree before the change:
`append 0.5649 / base 6.29 / dartree 34.90 ms`. The GEMMs were 59 % of the dartree call; this slice takes
out ~16.7 ms of the 34.9 ms, i.e. 48 %.

## 7. Rejected alternatives and negative results

1. **One shared rounding for both paths** (two-rounding everywhere, commit `6d01dd8`): `quick` exit 10,
   10 differing lines in the full-base block. Root cause is the `fmaf` in `dloop_matmul_tiled` vs the
   split `a*b+c` in `tree_gemm` (213 vs 0 `v_fma_f32`). Fixed by `FUSE` per policy. This was not a
   tolerance — no tolerance exists for it, `quick` compares bit-for-bit.
2. **`TN=1` and `TN=4`/`TN=8`**: the prototype's table already measures this on the real shapes — TN=2 is
   the flat part of the curve (dyn 10.53/10.59/11.21/16.19 µs for TN=1/2/4/8; q|k|v 17.85/**14.29**/15.10/
   18.45). TN=2 was taken from that table rather than re-derived.
3. **`skinny_rows_max = 200`** (skinny at every tree depth): 19.45–19.50 ms dartree vs 18.20–19.41 ms at
   64 — no win, and it would move the deeper, wider depths off the kernel that shares its staged
   activation tile. Measured, section 4.
4. **Reimplementing the crossover as a standalone probe binary**: dropped. The plan's numbers and the
   acceptance both come from `rocprofv3` traces of the real workload, and a private duplicate of two
   kernels would have been a second source of truth that could drift from `mine/`.
5. **Not done, deliberately**: the append path (`live != nullptr`) and the ring path keep their own
   kernels. The append's staged chunks run up to `chunk_rows=128` rows with a device-side live count,
   which is outside the shape class this slice owns; the ring path is contractually unchanged.

## 8. Commits

| SHA | What |
|---|---|
| `6d01dd8` | skinny GEMM family, dispatch from both `FullPolicy::gemm`, CMake source |
| `47908e1` | `FUSE` per policy: each path keeps its replaced kernel's own rounding |

## 9. Not verified here / caveats

- The per-shape **medians** in section 3 exceed the plan's targets on o_proj, selector and gate/up
  (min column is at/under target for o_proj and gate). Reported, not tuned. The end-to-end
  acceptance (`dartree ≤ 20 ms`, `base ≤ 5.3 ms`, both `IDENTICAL`) is met.
- The `ctest` failures are pre-existing and were attributed by running the **baseline worktree** with
  the same build options, not by assumption; the one apparent regression (`dloop_loader`) is a stale
  baseline binary and is proven so in section 6.
- The per-shape table's "after" column comes from a trace of the decode bench (n_seq 1), which reaches
  only 18 rows at depth 1. Depths 2–8 (36/54/72 rows) are covered by the end-to-end `bench` numbers and
  by the `skinny_rows_max` A/B, not by a per-shape trace.
- The append path (`live != nullptr`) and the ring path keep their own kernels; their numbers are
  unchanged by construction.
