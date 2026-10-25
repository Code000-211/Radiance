# shared-i8a8-down

An out-of-tree kernel plugin for radiance, ported from the `mine` model plugin. It serves the
shared expert's **down projection** — `gemm_nt_q` at dtype `i8a8` — at the speculative-verify
widths it measured a win on.

Opt-in: the row exists only when `RADIANCE_SHARED_I8A8_DOWN=1`. With the variable unset the plugin
declares its schema and no kernel, so selection is exactly the build's existing one.

## Scope

| axis | value |
|---|---|
| op | `gemm_nt_q` |
| dtype | `i8a8` |
| M | 1..3 |
| N | 2560 (`n_embd`) |
| K | 384 (`n_ff_shared`, per rank) |
| group | 128 |

**One geometry only.** The row is written and tuned for this exact `(N, K)` pair, and the
constraints are equalities on both. It does **not** serve the rank-1 `K=256` shape (a different
instantiation), nor the gate/up projection: that geometry is `N=768, K=2560`, it was measured and
it **loses** to the frozen row, so it is refused by the constraint table and again by the body.

The candidate covers one projection of one layer family. **No engine-level gain has been
measured**: what exists is the standalone kernel comparison recorded in the source tree
(`reports/betterbench-1200-campaign/shared-i8a8-probe/static-m-native-card{0,1}-timing.log`), which
timed AQL event submission for this body against the frozen row on the old ABI tree. Selection,
integration and any served-path number belong to the parent.

## Numerics

**Measured byte-identical at the served geometry; still declared `reordered`.** `compare.hip`
beside this file loads the frozen `gemm_i8a8_nt_m16` and this row, packs one weight with each
row's own `layout`/`relayout` hooks, and compares the two output buffers. See
[Measured](#measured) for the run. What is preserved:

- the signed INT8 dot per 128-K group, in INT32;
- every encoded byte and both scale planes (`i8` codes, `f32` activation scales per (row, 128),
  `bf16` weight scales per (row, 128));
- the frozen per-group product: `float(int32_dot) * (activation_scale * weight_scale)` in FP32,
  the same two factors as libr4d's `out += acc_val * (wsr * s)`;
- the BF16 output through the same round-to-nearest-even the frozen epilogue uses.

What could differ is the order of the FP32 fold across the three 128-K groups. At `K = 384` the
frozen narrow row does not split K at all — its split rule's first test is `K < 512` — so it folds
the groups ascending, which is this row's order. `compare.hip` confirms it on the card: zero
differing BF16 words at M = 1..4 under random, signed-extrema and all-zero inputs.

The declaration stays `reordered`, not `bitexact`: the measured equality is one weight and one
activation set per mode on one gfx1201 card, and it is an equality of this row against the row it
would displace, not a claim about any other build of libr4d.

No reduced precision, no requantisation, no copy, no scratch.

## Measured

Run on this tree's target build (`libr4d.so` built from the same commit as the candidate's
`RAD_LIBR4D_DIR`), AMD Radeon AI PRO R9700, gfx1201, under the GPU queue lock.
`compare.hip` is not part of the plugin and is not added to its CMake target; build it by hand:

```sh
/opt/rocm/llvm/bin/clang++ -x hip --rocm-path=/opt/rocm/core-10.0 \
  --rocm-device-lib-path=/opt/rocm/core-10.0/amdgcn/bitcode --offload-arch=gfx1201 \
  -std=c++17 -O3 -fno-fast-math -ffp-contract=off -I<tree>/abi -I<tree>/libr4d \
  compare.hip -o compare -L/opt/rocm/core-10.0/lib -lamdhip64 -ldl
```

```text
$ ./compare /path/to/build/radiance_home/kernels/libr4d.so \
            /path/to/build/radiance_home/kernels/shared_i8a8_down.so
device 0 AMD Radeon AI PRO R9700 arch=gfx1201 lanes=32
frozen_row=gemm_i8a8_nt_m16 candidate_row=shared_i8a8_down_m1_m3
mode=random tags frozen=("r4d.i8.w.frag64","r4d.i8.s.t16") candidate=("r4d.i8.w.frag64","r4d.i8.s.t16")
  codes_equal=1 scale_equal=1 bytes=983040/983040
M=1 bytes=5120  frozen_vs_candidate_diffs=0 first=-1 max_abs=0 frozen_vs_int_reference_bad=0
M=2 bytes=10240 frozen_vs_candidate_diffs=0 first=-1 max_abs=0 frozen_vs_int_reference_bad=0
M=3 bytes=15360 frozen_vs_candidate_diffs=0 first=-1 max_abs=0 frozen_vs_int_reference_bad=0
... the same for mode=signed-extremes and mode=zeros ...
negative_control changed_input_differs=1
gate/up geometry (N=768,K=2560,M=1) candidate launch status=-1 refused=1
gate/up geometry (N=768,K=2560,M=1) candidate opd_shape status=-2 refused=1
fourth row (N=2560,K=384,M=4)       candidate launch status=-1 refused=1
fourth row (N=2560,K=384,M=4)       candidate opd_shape status=-2 refused=1
RESULT: BYTE-IDENTICAL AND REFERENCE-CONSISTENT
```

What the harness establishes, and what it does not:

- the two rows' `layout` hooks agree on the tag and on the packed bytes (983,040 B of codes at
  2560 x 384, 15,360 B of BF16 scales at 2560 x 3), and each hook's `unrelayout` round-trips to
  the canonical plane;
- the candidate's BF16 output equals the frozen row's word for word at every M, and the frozen
  output itself matches an independent signed-integer reference at 4-BF16-ulp;
- changing the activation changes the output (the comparison has teeth);
- the gate/up geometry is refused, so the losing shape cannot be picked up.

**Not measured here:** speed. The correctness run above is one launch per case; the timed run
below is the speed question.

## Timed

`compare.hip --timing <out.csv>` is the speed half of the same harness and the same two rows. It
times `Rounds`=9 rounds of `Batches`=9 batches of `PerBatch`=256 launches, the codes rotating over
`Planes`=64 planes of 983,040 B (63 MiB, past this card's L2), one event pair per batch: the number
is launch-to-launch throughput, not a single launch's latency. The arms interleave **ABBA** inside
every round, so a drift over the round cancels instead of loading one arm, and the run reports the
median, the full range and how many paired rounds the candidate won.

Four independent runs, 36 rounds per arm and M, on a card with the GPU queue lock held:

```text
weights                     M     frozen      candidate   delta              rounds faster  spread    gain/spread
256 MiB, rotated            1     5.285 us    4.935 us    -0.343 us  -6.49%  36/36          0.079 us  4.3x
256 MiB, rotated            2     5.255 us    4.878 us    -0.381 us  -7.25%  36/36          0.066 us  5.1x
256 MiB, rotated            3     5.239 us    4.920 us    -0.311 us  -5.94%  36/36          0.073 us  4.7x
one copy, L2-resident       1                             -7.58%             36/36
one copy, L2-resident       2                            -10.03%             36/36
one copy, L2-resident       3                             -9.68%             36/36          0.307 us  1.4x
```

`spread` is the widest between-run range of the two arms at that M, and `gain/spread` the median
gain over it: a gain smaller than the spread would not be a win. Two earlier runs were thrown
away because a second tenant held the cards without taking the queue lock; the six rows above are
the clean four.

The gain is real at the kernel and it is small in absolute terms: ~0.3-0.4 us against a 5 us
launch. What it is worth to a served step depends on how often the serving path launches this
shape and on the step's own length, and **that is not measured**: no MoE container on this host
runs this row (`n_ff_shared`=640 per rank in the one container that exists, and its shared expert
is served block-fp8, not i8a8), so a serving A/B has no model to run on. `--timing --rank
frozen|cand <out.csv> <gofile>` is the harness for that case once there is one: two processes, one
per card, released together through `<gofile>`, so the gain has to survive the two ranks sharing
the memory system the way a `--tp 2` step does.

## How it hooks the weight

The stored planes are libr4d's `r4d.i8.w.frag64` codes and `r4d.i8.s.t16` tile-major scale plane.
The row publishes libr4d's own `layout` / `relayout` / `unrelayout` hooks — the same ones
`gemm_i8a8_nt_m16` publishes — and those come from `libr4d/r4d_fp8_layout.cpp` **compiled into this
DSO**, not linked from `libr4d.so`. Re-implementing the permutation here would drift from the bytes
libr4d's wider row reads for the same weight in the bands above M=4, so the implementation is
borrowed rather than restated.

Why compiled in and not resolved at load: the loader opens every plugin `RTLD_LOCAL`
(`core/plugin/loader.cpp`), so an unresolved `r4d_rad_layout_i8_block` would never bind. Why not
`dlopen`+`dlsym` at runtime: that is a load-time dependency on a specific library's private
symbols, which this candidate does not want.

The hooks are hidden inside this DSO (`-fvisibility=hidden` on the host sources), so they cannot
interpose on libr4d's copies. The plugin's exports are exactly the ABI's set:
`rad_plugin_abi_version`, `rad_plugin_info`, `rad_kernel_schema_count`, `rad_kernel_schema_at`,
`rad_kernel_count`, `rad_kernel_at`.

Because the layout implementation is not installed by the SDK, the build takes the radiance
**source** `libr4d/` directory as an input (`RAD_LIBR4D_DIR`). It is a build-time include, not a link
dependency: the resulting `.so` has no undefined symbol from `libr4d` and does not load it.

## Build

Against an installed radiance (ABI 26):

```sh
cmake -S . -B build -G Ninja \
      -DCMAKE_PREFIX_PATH=/path/to/radiance/prefix \
      -DRAD_LIBR4D_DIR=/path/to/radiance-source/libr4d \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j
```

`CMAKE_HIP_COMPILER` must be ROCm's clang, not `hipcc`. `RAD_GPU_TARGETS` defaults to the targets
the engine was built for; this plugin narrows them to `gfx1201` and skips the configure when none of
them match.

`RAD_LIBR4D_DIR` defaults to `../../libr4d` (the tree this candidate lives in) and must be given
explicitly when the candidate is built from outside that tree.

The result is `build/radiance_home/kernels/shared_i8a8_down.so`. `cmake --install build` puts it in
the found installation's `$RADIANCE_HOME/kernels`.

## Run

```sh
RADIANCE_KERNELS=shared_i8a8_down,libr4d RADIANCE_SHARED_I8A8_DOWN=1 radiance ...
```

`RADIANCE_KERNELS` is exclusive when set: name libr4d too, or nothing serves the ops this plugin
does not. Listing this library first makes it an override; it matches only its own geometry, so
every other op still resolves in libr4d.

`gate.json` beside this file records the run metadata. It is review metadata: nothing in radiance
reads it.

## What the row does at run time

- Reads **M from the operands** (`y->shape[0]`), not from the band's `M` parameter. Bands resolve
  at their upper bound, so the parameter reads 4 while a decode step issues one row; dispatching on
  it would run a 4-row template over a 1-row pass and write past the output.
- Refuses (negative status, never a no-op success): wrong dtypes, wrong rank, `M` outside 1..4, any
  `(N, K, group)` other than the pair above, a non-`i8a8` dtype string, a `pb1`/`pb2` partition, a
  non-tight row pitch, or a misaligned operand. It also refuses a call that passes `a_sum` or
  `b_ref`, which belong to other formats.
- Does not allocate, does not synchronise, and issues exactly one launch on the stream it is
  given.

## Known risks

- **Built, loaded and compared here; not timed.** The build, the loader (`rad-info --plugins`)
  and the byte comparison against `gemm_i8a8_nt_m16` all ran on this tree. No benchmark, no
  `rad-kbench`, no served-path number — see [Measured](#measured).
- **Layout bytes come from a source tree, not from the installed SDK.** If the engine's `libr4d`
  and `RAD_LIBR4D_DIR` come from different revisions, the tag would match while the bytes did not.
  The comparison above holds only because both were built from the same commit.
- **The bucket table gains an edge at M=3.** Any M-bounded row does this; it is a declare-time
  subdivision, not a change to the frozen bands' resolution. The edge sits at the band this row
  measured a win on, so a four-token verify step resolves into the band above and is served by
  libr4d's wider row (`gemm_i8a8_nt_m16`), which is faster there: 5.23 µs against 5.35 µs per
  launch, cold weights, 36/36 rounds (MAIN-42). M=1..3 win by 6-8 % in the same protocol.
- **No `RAD_R4D_TRACE` or `r4d_record_dispatches` participation.** The source row reached those
  through libr4d's `r4d_emit` sink, which does not exist in this ABI; this DSO packs its launch
  itself. Debugging this row in isolation means adding a trace locally.
- **The source row's `describe`/`RadLaunchDesc` half is gone, not stubbed.** ABI 26 has no
  `RadDescribeFn` and no `RadKernelInfo.describe`, so the launch is the only entry point, which is
  what a kernel row is here.
