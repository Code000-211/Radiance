# DLoop device kernels

The current engine model is **L5/full verified prefix**, specified in
`docs/DLOOP_FULL_CONTEXT_PLAN.md`, `mine/dloop_full.h` and `spec.md`.
It uses `dloop_full_append`, `dloop_full_base` and `dloop_full_dartree`, sharing the
unchanged pool and matrix/top-k implementations below. The L2/32-ring contracts remain
independent oracle/fixture coverage, not a fallback for serving the L5 checkpoint.
All timing tables below describe the earlier **L2/32-ring** model and do not establish
L5/full-context latency or GPU parity.

## Original ring-op implementation and measurements

`dloop_rows.cpp` registers production and tiny rows for all four ops. No schema changed in the batching rewrite. Production geometry is H768/FF2304/L2/Q12/KV4/Dh64/F7680/W2560/M8/C32/R256/K64, convolution 2/group 16, vocabulary 248320 with 248077 active IDs, TP2 shards 124160. Tiny geometry is H64/FF128/L2/Q4/KV2/Dh16/F48/W32/M8/C32/R8/K64, vocabulary 512/509 active, shards 256. Selection constraints and launch validation reject other geometries or rank/shard mappings. Budget is 1–15; width is 1–12; fixed quotas ignore width.

## Tiled matrix kernel and launch geometry

All launches use 128 threads (4 waves) and `hipLaunchKernel` on the engine stream. There is no allocation, host readback, stream synchronization, cooperative launch, or software grid barrier.

The batched matrix op is **one kernel for every shape**. The grid is `(column blocks, row chunks)`: a CTA owns `BN = 4*C` output columns for one 32-row chunk and streams K in 32-wide LDS steps.

- A lane holds **one row** and `C` output columns in registers, so a weight read out of LDS is wave-uniform and one `LDS.128` of weights feeds 32 lanes × 4 k values (128 FMAs per weight quad).
- Both weight tiles are staged as `uint4` (16-byte) loads with each column's row pointer resolved **once per CTA** into a shared table before the K loop; `matrix_weight` has branches and 64-bit multiplies that cost more than the 16 bytes it feeds, so it is never called from the inner loop.
- The activation tile is staged from per-row source pointers resolved once per row chunk (`matrix_row`), which also folds the ring's wrapped offset, the masked prefix, and the embedding gather.
- Pitch 36 floats makes every staged and every consumed float4 bank-conflict free and 16-byte aligned.
- K streams through LDS, so **no LDS size depends on a matrix shape**. `C` widens with the output width: 16 for the 124160-column head and the 2304-column gate (13.5/22.5/40.5 KiB of LDS + row table), 8 for `out>=1024`, else 4.

FP32 activations and FP32 accumulators are unchanged. Each active weight element is read from global exactly **once per CTA**, and the grid covers every (column block, row chunk) pair. Adjacent threads cooperatively load contiguous weights; the head pass streams its 636 MB shard once per call with all `S*M` y-vectors resident.

The exact shard top-k is a **single pass** over the `S*M*Vs` logits plus a two-level list merge. Each thread keeps a descending 64-entry list of `(score, token)` rank keys in registers; a monotone key orders by descending score and, at equal scores, by the **lower** token id, which is the reference's `better`. The list stays sorted by a fully unrolled compare-and-shift, all at compile-time register indices, so no thread spills its list to local memory. Each wave merges its 32 lists by repeated warp argmax (the winner rotates its list), and wave 0 merges the four wave results the same way with a per-lane LDS cursor. The previous version rescanned the whole shard once per output — 64 latency-bound passes — and cost 14.7 ms of the 78 ms baseline.

Norm/convolution/attention work has one CTA per path. Context append has one CTA per sequence and copies each accepted tap row with `uint4` gathers (no per-element modulo, no barrier in the row loop). Pool has one CTA per sequence/slot, merges the two ranks' packets, sorts the pool by token ID, and gathers owned head/embed/pred/succ rows or BF16 zero. A slot CTA also gathers anchor predecessor rows. Engine all-gathers/all-reduces remain separate operations.

Each tree depth submits **one batched backbone pipeline for all frontier paths of all sequences**, with rows `S*frontier*(M+1)`. No sequence CTA executes a backbone or serializes its paths. Matrix CTAs span the entire path batch. Only quota/top-k selection and final pruning are serial per-sequence bookkeeping.

## Per-step launch sequence

1. Context append: one CTA per (sequence, tap row) copies accepted BF16 tap bits into its ring; the ring offset of a row is a closed form of its index, so no ordering is needed and only the tap-0 CTA publishes the ring head. Optional separate tap planes concatenate into the context row.
2. Base: initialize/guard sequence metadata; context projection; context RMSNorm; for each of two layers, one packed context K/V matrix call and one RoPE launch. These cached 32-row K/V tensors are reused throughout the base backbone.
3. Base backbone: token projection (embedding input is gathered directly by the matrix loader); mask replacement/input norm; eleven launches per layer; output projection of all eight proposal rows.
4. Head: one fused launch over a 128-column vocabulary tile for every `S*8` row streams each BF16 head row from DRAM once, keeps a per-row top-64 key list in registers, and writes only those candidates; a second launch merges the per-CTA lists with the exact reference tie rule (descending score, then the LOWER token id) and emits the top-64 ID/score pairs. No `[rows × shard]` FP32 logits tensor is materialized. Ties prefer lower global ID.
5. Pool: one CTA per sequence/slot reads one candidate per thread from both rank packets, bitonic-sorts the 128 rank keys into the reference's order, walks the sorted list once for the 64 best DISTINCT ids (reproducing the reference's duplicate and `n < K` handling), sorts those ascending by token id, and gathers owned head/embed/pred/succ rows or BF16 zero with `uint4` copies. A slot CTA also gathers anchor predecessor rows. Engine all-gathers/all-reduces remain separate operations. Ties at the cutoff are exact.
6. Tree: initialize/guard sequence metadata; project and rotate context K/V **once per layer into tree scratch**, then reuse those tensors at every depth. This duplicates the base's small context K/V computation rather than changing the op schema or relying on scratch surviving across ops.
7. Slot zero uses base Y. Slots 1–7 each issue one batched backbone pipeline. Each prefix backbone computes `output_proj` for **only the current slot**, not all eight output rows. Each depth then launches selector hidden projection, score/log-softmax, and per-sequence selection.
8. Final per-sequence pruning/output launch emits split implicit-root trees and optional root-inclusive packed trees.

The eleven layer launches are: attention dynamic projection; attention pre-convolution; packed block Q/K/V projection; RoPE+attention; O projection; attention post-convolution/residual/post norm; MLP dynamic projection; MLP pre-convolution; gate/up+SwiGLU; down projection; MLP post-convolution/residual plus next-layer input norm or final norm. Final norm covers all eight base rows, but only the current prefix row. Thus a backbone costs **25 launches**.

At budget 15, both fixed and pruned reach all eight depths: base costs **34** launches; tree costs **205** (5 initialization/context-cache + 7*25 prefix + 8*3 selector/selection + 1 output). Including append and pool, the total is **241 launches/step**, excluding engine collectives and anchor embedding gather. Launch count does not grow with S or frontier width. Fixed produces two nodes at seven depths and one at the last depth, with fourteen prefix paths total. Pruned width 12 evaluates 84 prefix paths in seven batched calls, then retains fifteen nodes.

[INFERENCE] The first GPU-time optimization target is fusing per-layer launch boundaries. The rewrite deliberately retains exact FP32 activation storage and ordered accumulations. BF16 WMMA is a separate numerical change and must not be introduced without measuring acceptance impact. No latency or speedup is claimed before GPU execution.

## Dartree device path (`dloop_tree.hip`)

The tree op is the only DLoop op that walks seven dependent depths, each with `S*frontier*(M+1)` backbone rows, so its matrix traffic and its launch count grow with the frontier while the generic four-column kernel's activation re-reads grow with `N/4`. `mine/dloop_tree.hip` therefore carries its own device path for `dloop_dartree` only; base, pool and ctx_append keep using `dloop_kernels.hip` unchanged. The op schema, operands, parameters, scratch arena layout and `rad_mine_dloop_tree_scratch` are identical to before, so nothing outside `mine/` sees the split.

**Register-tiled GEMM.** One CTA owns a 64-row x 64-column output tile with 256 threads and a 4x4 thread tile (16 FP32 accumulators per thread), K chunked by 32. Weight tiles are expanded to FP32 in LDS once per K chunk and read back as two `float4` per K coordinate; the K inner loop is deliberately not unrolled, because unrolling it raises register pressure from 174 to 183 VGPRs and drops occupancy from 16 to 8 waves per SIMD, which costs more than the loop overhead. Measured `; NumVgrs: 179, Occupancy: 8` on gfx1201 for the plain variant. The gate/up pair keeps one activation tile and two weight tiles in LDS and writes `silu(gate)*up` for 64 columns per CTA.

Traffic per matrix is `2*K*N*ceil(rows/64)` weight bytes plus `4*rows*K*ceil(N/64)` activation bytes, against `4*rows*K*ceil(N/4)` for the generic kernel; at width 12 the generic loader re-read its activation tile ~16x more often than it needed to.

Thread-tile size is an occupancy trade, not a free choice: the 4x8 tile needs 179 VGPRs and lands on the 8-wave occupancy tier, while the 4x4 tile needs 109 VGPRs and reaches the 12-wave tier. Measured on the real geometry, the 4x4 tile is 1.25x faster for fixed and 1.3x for pruned, so the smaller tile wins despite doing twice as many LDS weight reads. The 32 accumulators of the 4x8 tile were re-measured and rejected.

**Launch sequence.** Same depth loop and the same per-depth stage order, but each stage is one launch over all sequences and frontier paths:

1. `tree_initialize` — every sequence CTA validates its own slot/length/anchor and the token-sorted in-range pool, then writes the root node.
2. `context_cache` — per layer, one K|V matrix over the S*C context rows plus one RoPE launch; both layers run before the depth loop so the depth-invariant context K/V is computed once.
3. Per depth with frontier > 1: token projection (embedding rows gathered by the matrix loader with the same anchor/path bounds as the generic path), then per layer the two dynamic projections, two convolutions, QKV, attention, O projection, gate/up, down projection and two finish/norm stages, then the output projection of the **current slot's** row only.
4. Per depth: selector hidden projection, score/log-softmax, and the selection launch (block-wide argmax per quota over the whole frontier x pool).
5. `tree_write_tree` — pruning, remapping and packed-tree publication.

**Launch counts per call (real geometry, n_seq=4).** The launch sequence is not reduced — the tree op still issues 205 launches, the same count as before the split. What changed is the cost per launch:

| Piece | Launches |
| --- | ---: |
| `tree_initialize` + `context_cache` (2 layers x K\|V gemm + RoPE) | 5 |
| Seven depths x backbone (embed, initial norm, 2 layers x (dyn proj, conv, QKV, attention, O proj, gate/up, down proj, finish), output proj) | 7 * 25 = 175 |
| Eight depths x (selector projection, scores, selection) | 8 * 3 = 24 |
| `tree_write_tree` | 1 |
| **Total** | **205** |

The eleven layer launches are unchanged: one dynamic projection, one pre-convolution, the packed Q/K/V projection, one RoPE+attention, the O projection, one attention finish/post norm, the MLP dynamic projection, the MLP pre-convolution, gate/up+SwiGLU, the down projection and one MLP finish. Depth 0 has no backbone, because slot 0 reads `y_base` as its hidden state; the `frontier` quota loop and per-depth launch structure of the previous path are preserved exactly. Both fixed and pruned issue 205 launches; the differences are the rows carried per launch (fixed fronts 1,2,2,2,2,2,2,2, pruned fronts 1,12,12,12,12,12,12,12) and the per-depth quota.

**Numerics.** Unchanged and reorder-free relative to `rad_ref_dloop_forward`: one FP32 accumulator per output advanced in ascending K with `-ffp-contract=off`, so every multiply-add order matches the CPU reference. Weight expansion happens once per tile into LDS (`bf(w)` is `uint32_t(w) << 16`, exact), activations stay FP32 end to end, and no WMMA or BF16 rounding of activations is introduced. Discrete outputs — tree tokens, parents, depths, node counts and the packed tree — are bit-identical, which is what the golden comparison checks.

Fixed and pruned both reach all eight depths at budget 15, with the same quotas, the same lowest-index tie rule in every argmax, and the same priority/depth/original-order pruning rule as before.

## Scratch

Let `J=S*Fmax`, B=M+1, D=KV_heads*Dh. Fmax is 1 for base, `ceil(budget/M)` for fixed tree, and `min(width,K,budget)` for pruned tree. In four-byte elements, the device arena is:

`J * (4*B*H + B*(H+2*D) + B*FF + 4*B*(H/group) + M*H + W + R + K)`

`+ S * (L*C*2*D + 121*(5+M) + 15 + 2)`.

It includes both layers' context K/V, batched block/norm/pre/output/QKV/FFN/dynamic buffers, current-slot projection/selector scores, and per-sequence tree bookkeeping. The scratch hook takes the maximum of this arena and the shared reference minimum. Base additionally appends `S*M*Vs` FP32 shard logits. Append/pool require no global scratch.

Production bytes per sequence with the current shared minimum:

| Op/config | Bytes per sequence |
| --- | ---: |
| Base, either variant | 4788736 |
| Fixed tree, budget 15 | 815616 |
| Pruned tree, width 12/budget 15 | 3528920 |

Multiply by S. Engine allocation must use the hooks, not these constants.

## Bytes streamed per step

The matrix weight-load count is independent of S and frontier width because each CTA covers every row of its row chunk, and the row-chunk axis only appears when the batch exceeds 32 rows. For a full eight-depth step, BF16 matrix weights account for **953679872 bytes on rank 0**, or **952435712 bytes on rank 1** (inactive head IDs are not read). This includes:

- context projection once;
- context K/V for both layers twice, once in base and once in tree;
- token projection, both layers' matrices, and output projection once per depth;
- selector hidden projection once per depth;
- one active LM-head shard pass: 635699200 bytes on rank 0, 634455040 on rank 1.

This is a source-level matrix-weight count, **not measured VRAM traffic or total step traffic**. Norm/base-convolution weights and gathered candidate rows add smaller reads. Conditional gathered head rows add `14*S*64*2560*2 = 4587520*S` bytes for fixed, or `84*S*64*2560*2 = 27525120*S` for width-12 pruned, before cache effects.

For a plain cached matrix `(rows,N,K)`, additional activation reads are `4*rows*K*ceil(N/4)` bytes and stores are `4*rows*N`; context/embedding source reads use two bytes instead of four. Gate/up share activation reads but load two weight matrices. The source loader does not read padded context or masked suffix embeddings. At full context, rank-0 fixed budget 15 issues **4270194688*S bytes of matrix activation source reads**; pruned width 12 issues **11610226688*S**. Of this, `2542796800*S` bytes are repeated reads of the small teacher-head input vectors, which are chunked through LDS. These counts expose L2 reuse requirements rather than pretending all repeated activation loads hit VRAM. [INFERENCE] Actual DRAM traffic, cache reuse, launch cost, occupancy, and step latency require GPU counters/timing.

## Semantics and invalid-value guards

All weights/teacher tables are BF16; computed states and accumulators remain FP32. `-ffp-contract=off` preserves split base/dynamic convolution terms and scalar reduction order. GQA, allocated-context RoPE positions, bidirectional block attention, immutable token-sorted pools, parent-major stable ties, fixed quotas, and priority/depth/original-order pruning match the reference. Transition normalization is `(score-max)-log(sum_exp)` before adding the parent path score.

Device-side guards skip invalid sequence work without a host readback: slot, head, length, reset, accepted-path row/termination, optional anchor token, and pool ID bounds/order are checked before data-dependent accesses. Prefix source indices are separately bounded. Invalid base metadata produces zero model outputs and empty shard packets; invalid tree metadata produces empty trees. Duplicate append slots are ignored after the first owner. These are safety guards for trusted engine metadata, not synchronous reference-style error reporting. Incomplete pool packets emit an empty/zero-masked pool instead of enabling an out-of-bounds prefix gather. Comments explain why duplicate real TP2 packets and ancestor-dropping pruning are unreachable; final parent bounds are guarded anyway.

## Measured device time (2026-10-17, checkpoint step 185000, n_seq=4)

`RADIANCE_DLOOP_BENCH=10`, median of 5 batches, real geometry, ranks 0/1 on one card. Launch counts and
bytes are per call; the matrix counts are per launch of the tiled kernel. "before" is the pre-optimization
baseline of the same command.

| Op | before | round 1 | round 2 | target |
| --- | ---: | ---: | ---: | ---: |
| `dloop_base` (rank 0 / 1) | 78.2 / 78.8 | 8.15 / 8.24 | **6.55 / 6.65** | ≤ 2 |
| `dloop_pool` (rank 0 / 1) | 0.92 / 1.23 | 0.92 / 1.24 | **0.078 / 0.044** | ≤ 0.3 |
| `dloop_ctx_append` | 0.294 / 0.298 | 0.089 | **0.013** | ≤ 0.03 |
| `dloop_dartree` fixed / pruned | 86.8 / 350.7 | 22.4 / 54.2 | 24.8 / 54.4 | ≤ 3 / ≤ 6 |

Per-launch attribution of the 78.2 ms baseline (engine `hipStreamSynchronize` around each launch, so the
absolute numbers are inflated but the split is not): `top64_shard` **14.7 ms**, the head tiled matrix
**4.8 ms**, the remaining 32 launches ≈ 1.0–1.1 ms in total. The 64-pass rescan in the old top-k and the
63.5 KB LDS per CTA in the old head GEMM (1 CTA/CU, 12.5% occupancy) were the two dominant costs.

Round 2 fused the head GEMM with the top-k: one CTA owns a 128-column vocabulary tile for every batch
row, streams each BF16 head row once from DRAM with 16-byte coalesced loads, keeps a per-row top-64 key
list in registers and writes only those keys; a second kernel merges the per-CTA lists with the exact
reference tie rule. The `[rows × 124188]` FP32 logits round trip and the separate latency-bound rescan
are gone. Serialized launch timing of the fused pass is **4.5 ms** (2 ranks), so together with the merge
it accounts for essentially all of the remaining `dloop_base` time.

`dloop_base` still exceeds its 2 ms target. [INFERENCE] The fused head reads its 636 MB rank shard in a
single pass, so its floor at the measured 545 GB/s local-VRAM bandwidth is ≈ 1.17 ms per rank; the
measured 4.5 ms (serialized) is about 4× that. The gap is not launch count (two launches per rank): the
per-CTA top-64 argmax chain (64 warp-wide rounds against the register key list) and the row-major 32-row
chunking remain the candidates, but no round-2 measurement isolates them, so the next step is a counter
or a standalone bench of the kernel rather than a further blind rewrite.

Round 2 also replaced the pool merge (one CTA per row, one candidate per thread, bitonic sort, single
walk over DISTINCT ids) and vectorised its gathers, and split `ctx_append` into one CTA per
(sequence, tap row). Both are under their targets.

The earlier warp-merged pool attempt failed `dloop_pool.pool_ids` against the real golden because all
four warps scanned the same candidates, so the merge emitted 16 distinct keys four times; the round-2
version assigns candidates to threads disjointly and reproduces the reference's duplicate handling and
its `n < K` rejection.

## Integration verification (2026-10-17)

The integration branch combines Base round 2 (`454359f`), Tree (`761765b`), and GDN
(`acb237b`), retaining the separate `dloop_tree.hip` entry point and the bounded GPU-lock wrapper.
HIP targets `mine`, `mine_ref`, `dloop_gpu_parity_test`, and `radiance` build successfully.
Host checks `dloop_ref_test`, `gdn_tree_ops_test`, `ple_tree_ops_test`, `ops_doc`, and `arch_test`
pass (5/5). Logs: `reports/dloop_engine/integration/optimized_final_build.log` and
`reports/dloop_engine/integration/optimized_host_checks.log`.

Combined GPU parity and the budget-15 engine run are **not verified**. The parity attempt ended
with signal 15 before acquiring the queue lock and produced no parity output
(`reports/dloop_engine/integration/optimized_parity_bench.log`). The legitimate lock holder was
`record_dloop_context.py` (PID 2467390); KFD additionally listed PID 2368831 without a matching
live process. Neither the queue owner nor the GPU driver was disturbed. Worker measurements above
are not measurements of the combined integration build. Resume only after the queue owner releases
the devices and the KFD idle check passes; then run the locked parity benchmark followed by
`scripts/dloop_testrun.sh reports/dloop_engine/sidecar_185000`.

## GPU checks when the devices are free

`tests/dloop_gpu_parity_test.cpp` is the executable pipeline check. It loads `mine` and `mine_ref` through the ordinary registry, requires actual gfx1201 device rows for all ten ops, uploads fixture inputs, compares each device stage to the CPU row on identical inputs, and passes the downloaded GPU outputs to the next stage. Missing devices, skipped/missing rows, missing fixtures, or any mismatch exit nonzero; there is no skip exit code. CMake builds/registers it only for the HIP gfx1201 build, labels it `gpu;dloop`, and excludes its source from the ordinary host test glob.

The test always runs the tiny golden pipeline. With all three environment paths below, it then runs the current real step-185000 pipeline. Both logical vocabulary shards execute on **one physical card**. A single-contributor pool and both rank-local pool gathers are checked; all-gather packets and the exact disjoint-row sum are host-emulated between device calls. This tests the op contract, not the engine's two-card collective transport.

After the integrator authorizes GPU execution:

```sh
RADIANCE_DLOOP_GOLDEN="$PWD/reports/dloop_engine/golden_185000" \
RADIANCE_DLOOP_SIDECAR="$PWD/reports/dloop_engine/sidecar_185000" \
RADIANCE_TEACHER_MATRICES=/home/orca/drafter_data/teacher/teacher-matrices.safetensors \
  build_hip/tests/dloop_gpu_parity_test \
  build_hip/radiance_home/kernels/mine.so \
  build_hip/radiance_home/kernels/mine_ref.so tests/fixtures/dloop_tiny
```

Success requires `DLOOP_GPU_PARITY_TINY_PASS`, `DLOOP_GPU_PARITY_REAL_PASS`, and `DLOOP_GPU_PARITY_ALL_PASS`, emitted only after each required coverage group passes. Without the real-golden environment variable, only the tiny group is requested. Append reconstructs the golden context ring in two accepted-path batches; base checks both logical shards; pool compares IDs/unary and exact owned-row bytes; fixed/pruned tree checks root-inclusive golden token/parent/depth arrays exactly. Each resulting topology also drives device/reference convolution, recurrent verification, and winner-path GDN commit on deterministic state/history inputs. Float tolerances are the reference test's ctxh `atol=2e-5, rtol=1e-5` and Y/unary `atol=3e-5, rtol=2e-5`; discrete outputs and copied BF16 data are exact. GDN numerical tensors use the same float tolerances; GDN state/history goldens do not exist, so those numerical outputs use `mine_ref`, while their topology is golden-checked.

`rad-kbench --op dloop_ --ref mine_ref` alone is not a replacement: its recorded fixtures store geometry/seeds rather than live producer packets, and the generic fill ABI cannot construct the required sorted pools. Additional GPU coverage/timing still to run:

- Append: byte equality, reset/empty path/wrap/16 rows/multiple slots/three tap planes; invalid slot/path/reset/head/length must not touch guard pages.
- Base: ctxh/Y/logits vs reference and exported goldens; lengths 1/31/32, wrap, padded NaNs, tied logits, inactive head tail, and invalid metadata. Compare single-sequence vs batched results. Verify cached context K/V is independent of prefix depth.
- Pool: byte equality of token IDs/BF16 owned rows/non-owned zeros, tie cutoffs, full all-reduce reconstruction, invalid anchors, incomplete packets, and duplicate packets.
- Tree: fixed/pruned budgets 1/7/15, widths 1/12, unequal paths, stable ties, very large equal unary logits, and ancestor-first pruning. Require exact split/packed token-parent agreement, <=15 draft nodes, <=16 packed rows, root preservation, and -1 padding. Compare batched S=8 against independent S=1 runs.
- Safety: invalid slots/lengths/heads/anchors and malformed accepted paths must not read/write outside operands; wrong geometry, rank/shard, paired optional operands, strides, or scratch must be refused before launch.
- Timing/counters: report drafter-phase wall time for S=1/8, per-depth matrix occupancy, launch gaps, L2 hit rate, and head weight bytes. Verify one head weight pass per step, then compare the measured phase with the serving drafter budget. Do not infer a performance pass from compile success.

Authorized workers may run bounded parity benchmarks through `scripts/gpu_locked.sh`; the integrator owns engine runs. Both paths must respect the queue lock and device-idle checks.
