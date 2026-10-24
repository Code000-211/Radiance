# hc_up_wmma -- an opt-in `hc_read` row for the M=3/4 band, ported from the ABI16 drafter work

A kernel plugin that answers `hc_read` for one geometry: `n=2560, hc=4, lowrank=320`, `M=3..4`,
`inject=1`, `group=128` -- the decode band of Qwen Flash-Next's drafter, where the up-dot reduction
dominates. It differs from libr4d's base row in **one** kernel: the up matvec runs as an unscaled
BF16 WMMA dot per 80-column slice with the FP32 mixing scale applied after the dot, instead of the
scalar reduction. Down, inject, the operand order, the mixing-weight rows and the fp8 layouts are
libr4d's, included rather than restated. The quantiser at the end is **not** included: it is this
library's own `quant` template (`hc_up_wmma.hip`), a different launch geometry from libr4d's
`hc_quant_rows_kernel` (one wave per row, one block per 128-channel group) doing the same amax,
scale and packing, so the codes it writes are the base row's.

Ported from `candidates/fulldrafter-abi16-review/source/mine/hc_up_wmma.hip` on
`pr/fulldrafter-source-review` (PR #1, `5c3bb1d`) to this tree's ABI (26). The archive is a snapshot
of ABI16 and is not merged wholesale; see "What changed in the port" below.

## Turning it on

Installing it changes nothing: `rad_kernel_count` is 0 unless `RADIANCE_HC_UP_WMMA=1` is set. On a
default install that switch alone is enough, because it is the load order -- not an extra option --
that puts this library first:

```sh
RADIANCE_HC_UP_WMMA=1 radiance --model drafter.rad ...
```

With no `RADIANCE_KERNELS` every library in the plugin directory loads, sorted by path
(`core/plugin/loader.cpp`): `hc_up_wmma.so` comes before `libavx.so` and `libr4d.so`. Selection is
by that hierarchy and is absolute (`core/plugin/select.cpp`): the first library that carries a row
for an op wins it, and a later library only serves rows the earlier one leaves unserved. The plugin
is built and installed by default (`cmake/RadAddPlugin.cmake`), so it is on that path already; the
`RADIANCE_HC_UP_WMMA=1` switch is the whole opt-in.

Naming the hierarchy explicitly pins the same order -- and that list is exclusive, so it has to
name every library, the host library (`libavx`, which serves every host-placed op, the tokeniser
and the server included) and the reference along with it (`docs/PLUGIN.md`, `docs/GUIDE.md`):

```sh
RADIANCE_HC_UP_WMMA=1 RADIANCE_KERNELS=hc_up_wmma,libavx,libr4d radiance --model drafter.rad ...
```

`hc_up_wmma` declares `hc_read` itself, so it may be listed first; the declaration is checked
against libr4d's field for field (`tests/hc_up_wmma_wiring_test.cpp`). A drifted copy is refused at
load with both plugins named -- and it is the **later** library the loader refuses, so with this one
first it is libr4d, plus every device row it carries, that disappears. That refusal is
unconditional: the schema is declared whether or not `RADIANCE_HC_UP_WMMA` is set.

When the row is offered, the library writes one line to stderr (`[hc_up_wmma] ...`). The engine's
`RAD_INFO` is core-internal and the plugin ABI has no log hook, so stderr is the channel a plugin
has; the line is written once, on the first `rad_kernel_count` call.

Every other shape -- `M=1,2`, other widths, a different `hc`/`lowrank`, `inject=0`, another
`dtype`/`mix` -- falls through to libr4d untouched. The row carries those limits as constraints, so
it is not even a candidate for them; there is no in-kernel fallback path and none is needed.

## What it does not change

- **No symbol interposition.** Everything the port pulls in from libr4d sits behind
  `#pragma GCC visibility push(hidden)`, as do its own non-exported definitions -- the dispatch-log
  globals (`r4d_dispatch_log`, `r4d_dispatch_cap`, `r4d_dispatch_n`, `exports.cpp`) and the device
  half's row accessors. The project sets no `-fvisibility=hidden`, so those pragmas are what keep
  it in. A card-less build (`-DRAD_WITH_HIP=OFF`) defines six dynamic symbols -- the plugin exports
  alone. The device build defines **18** (`nm -D --defined-only`): those six plus eleven
  `hc_up_wmma::*` names (`up_mix`, its `__device_stub__`, `quant<true>`, `quant<false>` and their
  stubs, `body`, `launch`, `row`, `scratch`, `dense`) and `__hip_cuid_*`. Cause: the `visibility
  pop` on `hc_up_wmma.hip:20` closes the pragma before the namespace at `:26`, so only the
  dispatch-log globals and the row accessors are pulled in; `__hip_cuid_*` is HIP's own. None of
  those names is a libr4d name, so nothing here can interpose: `libr4d`'s `r4d_rad_*` entries are
  not among them, both libraries load and the base row keeps working beside this one. Pulling the
  pragma over the namespace is the fix for the eleven; it is not made here (documentation-only
  run).
- **No new architecture, default or option.** This is the row only.
- **No numerics change in down/inject.** The `d` and scale buffers are the base row's, sized by the
  same arithmetic, so the up dot consumes exactly what libr4d's down kernel writes.

## Limits, beyond the shape constraints

- **Dense operands, aligned.** `h`, `x` and `inj` must have exactly their logical row pitch (`HN`,
  `n`, `hc`) and `h` must be 16-byte aligned; the base row accepts a padded pitch and needs no such
  alignment on `h`. These are not row constraints, so a caller that hands a padded `h` selects this
  row and gets `RAD_E_STRIDE` -- it does not fall back to libr4d.
- **`group` is required to be 128 even without `q`.** The row's constraint makes `group` part of
  selection; a call that leaves it unset (the parameter's default, 0) never reaches this row.

## What changed in the port

| ABI16 archive | here | why |
|---|---|---|
| `R4D_ENTRY(sym, body)` with the macro undefined around the include | not used | ABI26's device libraries define their entries directly; there is nothing to suppress, and the visibility pragma already keeps them in |
| `r.describe = ...` (a graph-node launch description) | dropped | ABI26 has no `RadKernelInfo::describe`; a row without it is submitted the ordinary way |
| `const R4dSink& sk` | `RadStream s` | `R4D_EMIT` takes the stream directly |
| row `mine_hc_read_up_wmma_m3_m4`, exports `rad_mine_hc_up_*` | `hc_read_up_wmma_m3_m4`, `rad_hc_up_wmma_*` | it is its own plugin, not a forked `mine` library |
| plugin exported its own whole kernel library, so `hc_read` was one row among many | this library declares `hc_read`'s schema and offers only this row | the ABI16 line forked libr4d; here libr4d is the base |

## Building

In the tree: `add_subdirectory(plugin/hc_up_wmma)` (already wired), so `hc_up_wmma` builds with
`libr4d`. Standalone against an installed prefix:

```sh
cmake -S plugin/hc_up_wmma -B build-hcw \
      -DCMAKE_PREFIX_PATH=<radiance prefix> \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DRAD_WITH_HIP=ON
cmake --build build-hcw
```

`CMAKE_HIP_COMPILER` must be ROCm's `clang++`, not `hipcc`, and `CMAKE_PREFIX_PATH` must be the
install prefix, not a build directory -- the same two rules every out-of-tree plugin follows.

Without a device build the library still compiles and loads, declares `hc_read`'s schema and offers
**no row**: there is no device `hc_read` base row on such a machine to override.

## What has been verified, and what has not

Verified here, no card involved:

- it compiles and links for `gfx1201` in this tree; the six plugin exports are the only dynamic
  symbols in a card-less build, and the device build adds eleven `hc_up_wmma::*` names plus
  `__hip_cuid_*` (18 in total, `nm -D --defined-only`);
- it is inert with the switch unset or `0`, and offers one device row with it set;
- that row claims only `M=3..4, n=2560, hc=4, lowrank=320, inject=1, group=128, dtype=bf16,
  mix=e4m3`, and its scratch hook answers `M*(lowrank*2 + 8*4)` and refuses every other band;
- its `hc_read` schema is libr4d's, field for field.

**Measured on a card, and not taken over.** `Radiance Testing` measured this row at head `39a6ef3`
over the base `b679448` on a physical R9700 (`gfx1201`, `HIP_VISIBLE_DEVICES=1`, serialized under
the benchmark lock); `up_mix` and `quant` are identical between `59d2a1a` and `39a6ef3`, so the
numbers hold for both. Numerics pass the tree's contract (`tools/kbench_core.cpp`): against `libref`
at `hc_read`'s 3,0e-2 e4m3 tolerance the port reads 6,95e-03 (M=4, `rotate=0`) and 1,30e-02
(M=4, `rotate=1`) -- at or below the base row's 7,76e-03 / 1,45e-02 on every seed, all six outputs
covered, zero memory violations, zero skips. Performance **fails**: 29,23 us against the base
row's 27,71 us at M=4 (+5,5 %, interleaved 3x) and 30,88 us against 27,69 us at M=3 (+11,6 %,
separate runs with identical case inputs). Cause: three launches per `hc_read` (down, `up_mix`,
`quant`) where the base row needs two, its up/mix/quant fused into one. Verdict: **no takeover** --
`RADIANCE_HC_UP_WMMA` stays off in serving. No serving A/B was run (C8, decode, prefill, composite,
TP2); the negative kernel delta could only be confirmed there, not reversed.
