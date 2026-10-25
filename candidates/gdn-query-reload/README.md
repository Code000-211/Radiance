# gdn-query-reload

An out-of-tree kernel plugin for radiance that serves the `gdn_conv_recurrent_update` op — the
folded GDN decode step (convolution + gated-delta-rule recurrence + gated RMS norm + quantiser
epilogue in one launch). It is a port of a candidate written against the older ABI 16 tree; this
copy is restructured for radiance 1.10.0 / `RAD_ABI_VERSION` 26. It **builds** there and the engine's
loader **accepts** it (see "Build" and "Load" below); it has not been gated against the frozen row
on a card. The source's register and speed numbers belong to an older query-only prototype, not to
the candidate this port copies — see "Original evidence" below.

## What it changes relative to the libr4d row

The arithmetic is the same kernel libr4d ships; three things differ, and only the first is a
numerical-path change — one argued from the compiled flags, not measured:

- **Query and key are re-read from LDS instead of held across the update.** Under the folded
  convolution the projection's q and k rows are already staged in shared memory. The shared l2
  norms are closed once per token by the first thread pair (the original ascending 64-term FP32
  sums and the lane-1 exchange), and the three post-norm consumers — the delta-rule inner product,
  the anchor factor store and the `h` update — re-read the same BF16 bytes from LDS and multiply by
  the same FP32 scale. Each element is therefore one `bf2f` widening of the same 16 bits followed by
  one FP32 multiply, and the unit is compiled `-ffp-contract=off`, so no multiply can fuse into the
  add after it. This is what removes the 64-register `qq`/`kk` arrays from the live set.
- **One payload owns the engine's scratch.** The arrival counters that let the last value head of a
  key head rewrite that head's conv state live in the scratch the engine hands the launch, and the
  engine makes no first-use-zero promise. The row dispatches a real 256-thread clear kernel on the
  caller's stream before every main dispatch, touching exactly `N * Hg` counters. There is no
  `hipMalloc`, no `hipMemset`, no synchronisation, and no allocation or initialisation pool. This
  dispatch is copied from the source candidate and has never been executed in either tree.
- **The scratch hook states a fixed 16384-byte ceiling** (4096 counters), which is exactly the
  `N * Hg` bound the entry point refuses past.

## Build

```sh
cmake -S candidates/gdn-query-reload -B /tmp/gdn-query-reload-build \
      -DCMAKE_PREFIX_PATH=<radiance install prefix> \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DGQRELOAD_LIBR4D_INCLUDE=<radiance tree>/libr4d
cmake --build /tmp/gdn-query-reload-build
```

- `CMAKE_HIP_COMPILER` must be ROCm's `clang++`, not `hipcc`; CMake refuses the hipcc driver as a
  HIP compiler.
- `CMAKE_PREFIX_PATH` must be the install prefix, not a build directory: the build tree's
  `radiance-config.cmake` computes its prefix from its own location and does not hold the headers
  or the targets file.
- `GQRELOAD_LIBR4D_INCLUDE` must name the libr4d **source** directory. The SDK exports exactly one
  target, `radiance::abi`, whose include directory holds `abi/` and `arch/` only; libr4d's private
  headers (`r4d.h`, `r4d_args.h`, `r4d_gdn_wmma.h`, `r4d_common.h`, `r4d_gdn_conv_tap.h`,
  `r4d_q8.h`, `r4d_emit.h`) are not installed. Configure refuses by name when the path is wrong.
- The build targets `gfx1201` only and skips itself with a message when the radiance being built
  against targets other architectures. It refuses at configure when that radiance has no device
  build at all, rather than producing a `.so` with no kernels behind it.

The plugin lands in `${RAD_PLUGIN_OUT}/kernels`, which defaults to this build directory's
`radiance_home`. To A/B it against the frozen libr4d row, point `-DRAD_PLUGIN_OUT` at the engine's
own home and name both libraries: `--kernels gdn_query_reload,libr4d`.

The project opts out of `rad_add_plugin`'s install rule (`RAD_PLUGIN_INSTALL OFF`): a candidate must
not be able to land in a real installation's `$RADIANCE_HOME/kernels` through `cmake --install`.
Copy the `.so` where you want it served from, deliberately.

### Built and loaded against radiance 1.10.0 / ABI 26 (measured)

This port was re-verified against the tree it is checked into, `b679448` (Radiance 1.10.0, ABI 26);
the earlier pass against `cb5a69a` is not what the numbers below are taken from. libr4d and
`abi/rad_abi.h` are unchanged between the two except for additive ABI symbols
(`RadKernelReferenceFn`, `rad_kernel_reference`) and a loader line that reads it, so nothing in the
source needed adapting: the row forks `libr4d/r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip`'s
arithmetic, and that file does not move between the two commits.

Against a staged SDK prefix built from `b679448`, with the ROCm clang++ named above and
`-DRAD_GPU_TARGETS=gfx1201`:

- configure: exit 0, build: exit 0, producing `radiance_home/kernels/gdn_query_reload.so`
  (121104 bytes) — the same size as the `cb5a69a` build, and a different `sha256` because
  `rad_hipify` names each code object after the source path (`-cuid`), so a build from another
  directory cannot be byte-identical by construction.
- `nm -D --defined-only` lists exactly the six ABI entries: `rad_plugin_abi_version`,
  `rad_plugin_info`, `rad_kernel_schema_count`, `rad_kernel_schema_at`, `rad_kernel_count`,
  `rad_kernel_at`. Nothing else is exported.
- `ldd` shows no radiance-linked dependency: `libamdhip64`, `libstdc++`, `libm`, `libgcc_s`, `libc`,
  `libdl`, and the ROCm runtime's own transitive libraries.
- All four instantiations are present in the binary: `ru_compact_cv<1,0>`, `<1,1>`, `<2,0>`, `<2,1>`
  (code format × state element type).
- `rad-info --plugins` with this plugin's home first on `RADIANCE_HOME`
  (`ROCR_VISIBLE_DEVICES=99`, no card touched) reports
  `gdn_query_reload 1.0.0  kernels  1 kernel(s), 0 schema(s), built for gfx1201` — i.e. the loader
  accepted the ABI version and read the row, and the row declares no schema by design. With
  `libr4d` in the same home and `RADIANCE_KERNELS=gdn_query_reload,libr4d` both load and the
  schema-owner complaint below is gone: `libr4d 0.5.0  kernels  448 kernel(s), 232 schema(s)`.
  Loaded on its own, the load fails — `kernel '…' in plugin 'gdn_query_reload' implements op
  'gdn_conv_recurrent_update', which no loaded plugin declares a schema for` — and that status is
  fatal to the whole load, not a per-row miss (see "Load").

Not run: `rad-kbench`, the comparator on a card, and any comparison against the frozen libr4d row;
selection of this row (which needs a declare pass over a model) is unverified for the same reason.

## The comparator

`gdn_query_reload_compare.cpp` drives both rows through radiance's own device layer
(`rad_dev_set` / `rad_dev_props` / `rad_stream_*`), which the installed SDK does not ship — it is
the engine's, not the plugin ABI's. So the harness is built only when this directory is built as
part of the radiance tree:

```sh
cmake -S <radiance tree> -B <build> -DRAD_BUILD_CANDIDATES=ON \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DRAD_GPU_TARGETS=gfx1201
cmake --build <build> --target gdn_query_reload gdn_query_reload_compare
<build>/bin/gdn_query_reload_compare --frozen <home>/kernels/libr4d.so \
      --candidate <home>/kernels/gdn_query_reload.so [--device N] [--steps N] [--poison-counters]
```

The in-tree build puts the plugin in the build tree's own `$RADIANCE_HOME/kernels`, so a later
engine run from that home loads it; a build without `-DRAD_BUILD_CANDIDATES=ON` never sees this
directory, and `RAD_PLUGIN_INSTALL OFF` keeps `cmake --install` from publishing it.

`rad-kbench` coverage of this op is not a small integration: kbench replays cases recorded from a
model (`data/kernels.rkb` has none for `gdn_conv_recurrent_update`), and `--record-rows` can only
move an existing case of the same op into a row's constraints, so the folded row stays unreached
until a fixture is recorded with a container on a card. The comparator is the coverage that does
not need one.

## Load

```sh
rad-info --plugins --radiance-home <home containing kernels/gdn_query_reload.so>
RADIANCE_KERNELS=gdn_query_reload,libr4d radiance --model <container> ...
```

The plugin declares no op schema. `gdn_conv_recurrent_update` is libr4d's schema — the first plugin
to declare an op fixes its vocabulary — and this DSO only implements it. Selection is first-match in
hierarchy order (`core/plugin/registry.h`): plugins are queried in the configured order and the first
one with any matching row supplies the op, so `--kernels gdn_query_reload,libr4d` is what puts this
row ahead of libr4d's for this op and leaves every other op to libr4d. A non-empty `--kernels` is
exclusive — kernel libraries it does not name are closed before registration
(`core/plugin/loader.cpp`) — so `libref` is not loaded at all under these flags, not merely ranked
below them.

Loading the plugin is not opt-in once its file sits in an environment's `kernels/`: the complete set
is checked for a schema per row (`Registry::check_complete`), a row whose op no loaded plugin declares
returns `RAD_E_NOSCHEMA`, and `Engine::load_plugins` aborts on that status
(`core/engine_bringup.cpp`). A home holding this `.so` and no libr4d therefore cannot start the
engine at all — `rad-info --plugins` itself exits non-zero on it. With libr4d in the same home and
both named, both load.

## Limits — what this row refuses

Declared as selection constraints, so a call outside them is served by another library rather than
failing:

| key | accepted |
|---|---|
| `head_k`, `head_v` | `128` |
| `conv_width` | `4` |
| `q_len` | `1 … 16` |
| `state_form` | `anchor` |

Because the arch block declares `state_form=anchor` only on steps that speculate
(`rad_block_gdn_fp8.h`), this row serves those steps and a plain decode step falls through to
libr4d's row. The alternative — serving `per_candidate` too — would mean carrying the second state
indexing scheme, which this fork does not have.

Refused at launch (they depend on the tensors, not the query):

- a state cache that is neither `f32` nor `f16`; a quantiser pair that is neither `i8` + `f32`
  scale nor `fp8_e4m3` + `f32` scale, or a pair with only one present;
- an `x` whose columns are not `H*V + 2*Hg*K` wide, a `head_k`/`head_v` parameter that disagrees with
  the tensors, or an `Hg` that does not divide `H`;
- a conv-state cache depth smaller than `conv_width - 2 + q_len`;
- a stride that does not fit `int32`, a `state_idx` pitch under 6 or too small to hold a head's
  factor block;
- a state flood larger than 16 MiB per launch (this row is instantiated with the non-temporal hint
  on, the variant the source's comments describe as the reached one);
- absent scratch, scratch smaller than `N * Hg * 4` bytes, or scratch not 4-byte aligned;
- any `state_form` other than `anchor`. This fork carries the anchor instantiation only; a
  `per_candidate` call is refused rather than served with the other form's state indexing.

`N == 0` returns success without dispatching.

**Per-sequence token count.** The folded conv staging holds `RU_CVT == 16` tokens per sequence and
the token loop indexes it by the sequence's own row count read from `cu`. A sequence longer than 16
rows in one step would overrun that staging, exactly as libr4d's own folded row would (it clamps
its staging loop to `RU_CVT`, not its token loop). The engine issues `q_len` equal to the step's
rows, so the `q_len <= 16` constraint is the guard that keeps this true; a caller driving the entry
point directly must uphold the same bound.

## Numerics

What follows is the **arithmetic contract this source states**, not a measurement of this build:

- FP32 accumulation, IEEE round-to-nearest-even, `-ffp-contract=off` and `-fno-fast-math` are set on
  the target. Every BF16 intermediate — the conv output, the output store, the quantiser's input —
  keeps its original rounding, and the gated RMS norm and the int8 quantiser epilogue (the
  `amax * (1/127)` scale and the magic-constant round-to-nearest-even code) are the source's.
- The non-temporal hint on the state load and store changes no byte; it is a cache-eviction hint.
- The two template axes this row instantiates beyond the source's single instantiation — the state
  cache element type (`f32`/`f16`) and the quantiser code format (`fp8_e4m3`/`i8`) — are the fork's
  own; they select the element type of the state traffic and the epilogue's conversion, and add no
  new numeric stage. An absent `o_q`/`o_scale` pair skips the quantiser entirely, which is what the
  target schema allows.

`gate.json` declares `numerics: bitexact`. That is the contract above, not a result: **no run of this
payload on a card exists in this tree**, so byte equality with the frozen libr4d row is unverified for
every form, including the fp32-state / int8 one the source was written for, and a fortiori for the
f16 and fp8-e4m3 instantiations, which have never executed anywhere. The harness that would decide it
is built and ready; running it is the open measurement.

### The comparator: what it checks, and that it has not run here

`gdn_query_reload_compare.cpp` dlopens the frozen `libr4d.so` and this plugin, finds both rows by
name, and calls each `launch(const RadArgs*, RadStream)` on radiance's own stream. There is no
describe path in ABI 26, so the comparison it performs is of the real dispatches.

**It has not been run on a card in this tree.** No invocation, timestamp or log of it exists here, so
this document states no result — not for byte equality, not for the poison/guard checks, not for the
refusals. `grep -c 'ok(' gdn_query_reload_compare.cpp` reports 13 lines: the helper's definition plus
twelve call sites, one of them inside the six-iteration refusal loop. That is what a complete run
asserts, not a count it has returned. An earlier version of this file reported "13 checks, 0
failures, on a card", which this port cannot substantiate: the statement came with the source
candidate, has no log in this repository, and is not carried over here.

When it runs, it asserts byte equality over a six-step trajectory, at the allocation's full extent, on
every plane: `o` (bf16), `o_q` (i8), `o_scale` (f32), the recurrent state/anchor/factor cache (f32)
and the conv state cache (bf16), with each step's acceptance count being what the previous step
actually wrote and neither arm reset to a snapshot, so every step reads what the step before left.

The equality is guarded by controls inside the same invocation:

- both frozen caches are checked to have moved (a comparison over untouched buffers would be
  vacuous);
- a one-bit input flip must move the frozen row's output, and the comparison must reject one changed
  consumer byte (negative control);
- with `--poison-counters`, the candidate's scratch is filled with `0x5b` (guards `0xa7`) before
  **every** launch, and after each launch all `N*Hg` counters must be zero and both guards untouched
  — the emitted 256-thread clear writing exactly its own counters on reused scratch;
- the scratch hook must answer a parameters-only declare with exactly `4096 * 4` bytes;
- six refusals must return their declared codes: integer state cache (`RAD_E_DTYPE`), an `x` pitch
  past int32 (`RAD_E_STRIDE`), scratch one byte short (`RAD_E_SCRATCH`), `q_len` 17
  (`RAD_E_SHAPE`), `state_form=per_candidate` (`RAD_E_UNSUPPORTED`), absent `num_accepted`
  (`RAD_E_INVAL`).

That run does not cover speed (no timing is part of it), the f16 state cache, the fp8-e4m3 code path,
the optional conv-bias operand, or any full-model step.

The open measurement, for whoever has a card. The build is CPU work and takes no GPU window, even
while another agent is measuring; only the run itself opens both cards, so only the run goes through
the shared agent test window:

```sh
# CPU only, no lock, no status transition:
cmake --build <build> --target gdn_query_reload gdn_query_reload_compare

# The run takes /tmp/radiance-tp2-bench.lock nonblocking, sets the executing issue to `testing`
# before it starts, and keeps the lock through the client and its cleanup. Exit 75 = the window is
# taken, nothing ran, the issue stays ready_for_testing; name the holder, do not retry in a loop.
python3 scripts/gpu_job.py --issue MAIN-<n> -- \
  <build>/bin/gdn_query_reload_compare --frozen <home>/kernels/libr4d.so \
    --candidate <home>/kernels/gdn_query_reload.so --steps 6 --poison-counters
```

`gpu_job.py` and `docs/GPU-COORDINATION.md` are not part of this tree; they live on the
coordination branch `agent/mika/496236b8c06d` (PR #47, closed without merge on request). A bare
`flock -n -x /tmp/radiance-tp2-bench.lock` around this command is the legacy form the runner
replaces: it records no `testing` transition, and around a build it blocks unrelated host tests and
reviews for the length of somebody else's measurement.

## Original evidence, and what it does and does not say about this target

The source tree's candidate carries this record: a 171-VGPR, 0-spill prototype register count, and
a set of decode-step timings taken on the same card. **None of it was produced by this payload
against radiance 1.10.0.**

Two distinct pieces of history must not be conflated:

- The **source candidate** (`candidates/gdn-query-reload` in the old tree) already carried the real
  emitted counter-clear kernel ahead of its main dispatch — that is the code this port copies. Its
  clear was **never run** in the source tree: the register/cost record there is from a query-only
  prototype whose caller zeroed the counters.
- The **older query-only prototype** had no clear at all and relied on caller zeroing. It is not
  the parent of this port.

Nothing in this tree supersedes that: **the clear has not run here either.** Its correctness is
exactly what the `--poison-counters` pass above would establish (guards intact, all `N*Hg` counters
zero on reused scratch), and that pass is open. Its cost is unknown in any tree — no timing has been
taken — so the second dispatch's price is not a number this port has.

Three things in this port are **not** covered by the source's evidence at all:

1. **The per-launch counter clear — neither correctness nor cost.** It has not run here; the
   poison/guard check is the one that would settle correctness, and no timing anywhere settles cost.
   Its ordering argument is stream order alone.
2. **The f16 state instantiation.** The source instantiated the fp32 state only. This row adds the
   f16 form because the target schema allows it and a refusal would abort a step the row was
   selected for. The arithmetic is `ru_state<1>`'s, unchanged from the source body, but it has
   never run — and there is **no measured bit-exactness claim** for it.
3. **The fp8-e4m3 code path.** The source instantiated the int8 epilogue only. The fp8 branch is the
   source body's own code (clamp to 448, `v_cvt_pk_fp8_f32`), compiled here for the first time, with
   **no measured bit-exactness claim**. If the served model's out projection takes fp8 rather than
   int8 codes, this is the path that runs and it is unvalidated.

Also unmeasured: the removed `qq`/`kk` live range and the added block barrier; the optional
conv-bias operand; the non-temporal hint, whose cost/benefit is only reported in the source's
comments (a step-level measurement, not this payload's); and any full-model step — `rad-kbench` was
not run, and no speed number of any kind was taken on this payload.

## Files

- `gdn_query_reload.hip` — the kernel, the entry point and the plugin exports.
- `gdn_query_reload_compare.cpp` — the ABI 26 comparison harness: it dlopens the frozen `libr4d.so`
  and this plugin, drives both rows over a seeded multi-step trajectory and compares every output
  and state plane byte for byte, with the controls and refusals listed under "Numerics".
- `CMakeLists.txt` — the plugin build (out-of-tree against a staged SDK prefix), and the comparator
  target, which exists only when the directory is built inside the radiance tree.
- `gate.json` — review metadata. Its `engine_env`/`engine_args` are the flags this engine accepts;
  the `optional_local_harness_profile` block records the source tree's own harness variables as
  history and is not read by this engine.
