# hc_up_wmma -- an opt-in `hc_read` row for the M=3/4 band, ported from the ABI16 drafter work

A kernel plugin that answers `hc_read` for one geometry: `n=2560, hc=4, lowrank=320`, `M=3..4`,
`inject=1`, `group=128` -- the decode band of Qwen Flash-Next's drafter, where the up-dot reduction
dominates. It differs from libr4d's base row in **one** kernel: the up matvec runs as an unscaled
BF16 WMMA dot per 80-column slice with the FP32 mixing scale applied after the dot, instead of the
scalar reduction. Down, inject, the operand order, the mixing-weight rows, the fp8 layouts and the
quantiser are libr4d's, included rather than restated.

Ported from `candidates/fulldrafter-abi16-review/source/mine/hc_up_wmma.hip` on
`pr/fulldrafter-source-review` (PR #1, `5c3bb1d`) to this tree's ABI (26). The archive is a snapshot
of ABI16 and is not merged wholesale; see "What changed in the port" below.

## Turning it on

Nothing happens by installing it. Two switches, both required:

```sh
RADIANCE_HC_UP_WMMA=1 RADIANCE_KERNELS=hc_up_wmma,libr4d radiance --model drafter.rad ...
```

1. `RADIANCE_HC_UP_WMMA=1` makes the library offer its row at all (`rad_kernel_count` is 0
   otherwise). An operator who installs the plugin and forgets this gets exactly the base kernel.
2. The library must come **before** libr4d in the kernel hierarchy. Op selection is by hierarchy,
   not by priority: the first library that carries a row for an op fixes the op's schema, and a
   later library only serves rows the earlier one leaves unserved. `hc_up_wmma` alone on the path
   therefore never wins the op -- it also *declares* `hc_read`, so it can be listed first, and that
   declaration is checked against libr4d's field for field (`tests/hc_up_wmma_wiring_test.cpp`); a
   drifted copy is refused at load with both plugins named.

Every other shape -- `M=1,2`, other widths, a different `hc`/`lowrank`, `inject=0`, another
`dtype`/`mix` -- falls through to libr4d untouched. The row carries those limits as constraints, so
it is not even a candidate for them; there is no in-kernel fallback path and none is needed.

## What it does not change

- **No symbol interposition.** Everything the port pulls in from libr4d sits behind
  `#pragma GCC visibility push(hidden)`, so the only dynamic symbols this library exports are the
  six plugin exports. `libr4d`'s `r4d_rad_*` entries are not among them; both libraries load and
  the base row keeps working beside this one.
- **No new architecture, default or option.** This is the row only.
- **No numerics change in down/inject.** The `d` and scale buffers are the base row's, sized by the
  same arithmetic, so the up dot consumes exactly what libr4d's down kernel writes.

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

- it compiles and links for `gfx1201` in this tree and exports exactly the six plugin symbols;
- it is inert with the switch unset or `0`, and offers one device row with it set;
- that row claims only `M=3..4, n=2560, hc=4, lowrank=320, inject=1, group=128, dtype=bf16,
  mix=e4m3`, and its scratch hook answers `M*(lowrank*2 + 8*4)` and refuses every other band;
- its `hc_read` schema is libr4d's, field for field.

**Not verified: the numbers on a card.** No GPU run, no benchmark and no C8 measurement was made
for this port; that belongs to `Radiance Testing`. Until then the row's outputs have not been
compared against libr4d's on real weights, and the switch should stay off in serving.
