# libbolegdn — the Bole tree-GDN state update

An opt-in kernel plugin for `gfx1201` that ports the **Bole** tree state-update solver for the gated
delta net (arXiv:2608.01651v1, §IV-A) onto Radiance's ABI 26. It adds one op, `gdn_tree_bole`, with
two device rows selected by the `stage` constraint:

| `stage` | row | what it does | status |
|---|---|---|---|
| `verify` | `gdn_tree_bole_verify_k128_v128` | runs the Bole solver over a candidate tree: the root serial in FP32, every other node node-parallel in FP64 over its non-root ancestors; writes per-node factors and outputs | **reordered** accumulation below the root |
| `commit` | `gdn_tree_bole_commit_k128_v128` | replays the accepted path's factors onto the state cache and rewrites the conv history | **byte-exact** to a serial replay |

Nothing selects this op unless a caller names the plugin. It is not a drop-in replacement for
`libr4d`'s `gdn_conv_recurrent_update` and does not shadow it: the two rows name a different op, and
the engine has no tree-layout seam (`core/sched/batch.cpp` refuses tree-layout speculation).

## Why a new op is needed

`libr4d`'s GDN decode row is a serial recurrence: it walks a sequence's tokens in order, keeps one
FP32 state in registers, and writes one state per candidate. A speculative **tree** needs three
things that op has no operand for:

- a `parent` array naming each candidate's parent, so candidates sharing an ancestor can be
  evaluated against that ancestor rather than against each other;
- per-**node** factors, so the committing step can replay only the accepted path;
- a per-node output row, so the verifier can be scored per candidate.

Adding those operands to `gdn_conv_recurrent_update` would change an op the shipped models already
select, and its schema is fixed by the first plugin that declares it (`core/plugin/loader.cpp`). So
the tree path gets its own op, and the serial row stays exactly as it was.

## Which archived variant this is, and why

The source line (ABI 16) holds three archived Bole variants. This plugin ports **one**:

| variant | why not |
|---|---|
| `2026-10-20_bole-vb-exact-commit` | **ported.** Newest. The only variant whose *commit* was byte-exact to the serial recurrence in every tested case (48/48 commits, 173,890,816 state bytes identical), including the FP32-cancellation scene that defeated the earlier two. Its verifier still fails a cancellation **below the root**, which is why the `verify` row declares a reordered result instead of claiming bytes. |
| `2026-10-20_bole-adjoint` | superseded. Emitted its own `U` for the commit to replay; because the verifier's `U` bits differed from the serial path's, the replay left a residual in the committed anchor. Failed at round 2, node 0: group scale `4.2651e-36` against a serial zero group, anchor `1.2622e-29` against `0`. |
| `2026-10-20_bole-neumann` | superseded. The finite-Neumann readout adds terms of order `2^17 q` to reach `2^-40 q`, an algebraic-conditioning defect rather than a code bug. Failed at node 2: `U` `0xb5000000` against serial `0xb4ffffe4`, dequantized output error `1.5227e-11` against a bound of `9.7968e-13`. |

The decisive property of the ported variant is that it does **not** replay a stored `U`. Its commit
reads the original update data — normalized `k`, convolved `v`, `gamma`, `beta` — and recomputes `u`
with the recurrent kernel's own FP32 instructions in the serial order. The verifier may then use any
arithmetic it likes below the root and the committed bytes are still the serial bytes.

## What is the engine's arithmetic, unchanged

Bole changes the recurrence **solver**, not the precision. Everything else is `libr4d`'s, restated in
`bole_device.h` and cited line by line there:

- the conv tap order, bias first, one BF16 rounding of `cp_silu`;
- `gamma = exp(-exp(A_log) * softplus(a + dt_bias, thr))`, `beta = sigmoid(b)`;
- q/k L2 normalization with `l2_eps = 1e-6`, `q` scaled by `scale`;
- `h *= e^g; dot = h·k; u = (v - dot) * beta; h += u k^T; o = h·q`;
- the gated RMS norm over 128 channels with the engine's reduction order, one BF16 output rounding;
- the INT8 group-128 epilogue, which quantizes the **rounded BF16** row, never the FP32 value.

The state is FP32 and the intermediate boundaries are BF16, as in the serial row. A caller that
passes `l2_eps != 1e-6` is refused, exactly as `libr4d`'s row refuses it: the epsilon is compiled into
the exchange.

## Refusals

Both rows validate before launching and return a negative `RAD_E_*` rather than serving a geometry
they would compute differently:

- `head_k`/`head_v` other than 128, `conv_width` other than 4;
- `q_len` outside `1..16` (the node budget the LDS tables are sized for);
- `conv_depth` outside `1..64`, or disagreeing with the conv cache's extent;
- `l2_eps` other than `1e-6`;
- a state cache that is not FP32 (an FP16 cache is refused, not silently narrowed);
- `o_q` without `o_scale`, or the wrong dtypes for either;
- `cu`, `conv_idx`, `ctx_before` or `state_idx` inconsistent with the sequence count.

The device rows also re-check the metadata — including the parent array, where a parent must name an
earlier node — and publish that verdict to the whole block before any read or write, so a direct op
caller cannot drive a walk off the array or leave a half-committed state behind.

## What is NOT ported, and why

- **The scheduler and the tree-policy driver.** DarTree's adaptive depth, node budget and pruning
  live in the source's `core/sched`. Radiance ABI 26 has no tree seam; this plugin supplies the
  arithmetic only, and a caller is responsible for building the tree and its `parent` array.
- **MTP8 / tree-64 speculation.** The source's candidate profile `mtp8-tree64-bole` is not ported and
  is not endorsed by this plugin.
- **The pruned-DarTree selection and the tree-construction kernel.** Those are the source's tree
  policy, not the state update.
- **Any model wiring.** No architecture plugin here names this op. That is deliberate: the source
  campaign measured the MTP8/tree-64 path built on Bole as clearly negative end to end, so promoting
  it would be wrong. See "Measured status" below.
- **FP16 state caches**, the `anchor` state form, and the paged-anchor marker/factor-column machinery.
  The ported rows serve the linear per-sequence state layout only. The anchor form is where the
  source's verifier boundary was found, and re-deriving the page-shift columns for this engine was
  not done.

## Measured status

Read this before treating the plugin as a speedup. **It is not one.**

- The isolated source-line numbers are **ABI-16** evidence from a different engine, not results of
  this plugin: a 4-node case at 103.9 → 20.2 µs, a 16-node chain at 143.8 → 178.2 µs (slower), and an
  engine smoke at 122.25 tok/s against a serial 123.39 tok/s.
- The end-to-end MTP8/tree-64 path built on Bole measured **negative**: 229.7 / 238.0 / 267.1 tok/s
  against a 375.6 baseline.
- The Bole GDN path was never measured as a standalone engine gain in the source campaign.
- **This plugin has no speed measurement at all.** `tests/bole_gdn_gpu_test` measures correctness, not
  time. Do not quote a throughput figure for it.

What this plugin does establish, and what the test checks, is narrower and checkable: the commit
row's committed state and conv state are byte-identical to `libr4d`'s GDN row, the verify row's root
reproduces the serial value exactly, and the verify row's deeper nodes agree within a declared
reorder tolerance — with negative controls proving the comparison predicates can fail.

## Build

Built in radiance's tree, beside `libr4d`, because the device source restates `libr4d`'s rounding
boundaries and is developed against its headers:

```sh
cmake -S . -B build -GNinja -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201 ...
ninja -C build libbolegdn
```

It is installed with the other kernel plugins and loaded from `$RADIANCE_HOME/kernels`. Configure
without HIP and it is not built: the op would have rows with no kernels behind them.

## Test

```sh
RADIANCE_HOME=build/radiance_home ctest --test-dir build -R bole_gdn_gpu_card
```

One card, `RUN_SERIAL`. Without a visible device it reports SKIP (exit 77) rather than passing.
