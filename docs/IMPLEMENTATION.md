# Radiance: implementation brief

`spec.md` is the design and it is binding; read it first. This file covers the mechanical part:
where things live, how they build, and the conventions every component follows.

## The host backend is a primary target

Everything except the device kernels builds, tests and runs through the host device backend:
`cmake` with no ROCm must configure, build, test and run, and `RAD_HIP_FOUND` gates only the HIP
backend and the device kernels. A machine with no card is a complete development environment for
the core, the tools, the architecture plugins and `libref`, and the suite runs in full there.

## Layout

```
abi/              the ABI, and all a plugin compiles against. Frozen: change it deliberately
core/rad_internal.h   shared internals: logging, Geometry, constraint matching
core/util/        logging, error strings, dtype arithmetic, tensor helpers
core/device/      device backends: host, and aql (HSA queues the engine writes; HIP for memory)
core/plugin/      plugin loader, kernel registry, op schemas, selector, bucket tables
core/build/       the declare phase: builder, weight manifest, buffer plan, graph dump
core/runtime/     the run phase: RadCtx, rad_issue, the arena
core/mem/         pools, slab, KV block manager, prefix cache
core/sched/       scheduler, RadBatch construction, request lifecycle
core/place/       placement planner, the mover, the heat engine
core/sample/      device sampler dispatch, the host reference chain, grammar bitmask
core/text/        tokeniser, chat template, tool parsing, detokenisation
core/mm/          image and video decoding, and the processor that turns frames into patches
core/gen/         image generation: the request and the pass driver
core/format/      the .rad reader and writer, safetensors, recipes, the imatrix, the tune cache
core/server/      HTTP: the OpenAI, Anthropic and llama-server endpoints, /metrics, the dashboard
core/diag/        the startup report
core/kld.*        the KL mode: --kld-record / --kld-ref
core/engine*      the engine that ties them together; core/main.cpp is the `radiance` binary
libref/           every op, naively and correctly, host domain: the oracle and last resort
libr4d/           the RDNA4 (gfx12xx) kernel library: kernels, rows, layouts, fusion claims
libavx/           the same vocabulary in x86 SIMD, host domain: the production host kernels,
                  built on every x86-64 build because the engine refuses to serve an op that
                  resolves only to libref (without --debug-accept-reference-kernels)
libquant/         the built-in quantisers rad-convert's recipes name: grids, transforms, rtn, gptq
arch/<id>/        one architecture plugin per directory
tools/            rad-info, rad-kbench, rad-convert, rad-tune, rad-schemas, rad-tokcheck,
                  rad-chatparse, rad-gbnf-check
cmake/            rad_add_plugin, and the package config an out-of-tree plugin find_package()s
scripts/          harnesses for serving, tracing and profiling a running engine
data/             kernels.rkb and kernels_moe.rkb: recorded references rad-kbench replays with
                  no model. Two, because no one container declares the whole vocabulary.
                  recipes/: the quantisation recipes of the containers this tree serves
docs/KERNELS.md   what rad-kbench last measured, generated and checked in
docs/ADAPTIVE-MTP.md  the adaptive draft policy's contracts, profile/manifest formats and owners
vendor/           llama.cpp lifts tracked to an upstream commit (vendor/UPSTREAM), changed only
                  through vendor/patches/; nlohmann/json and cpp-httplib
tests/            ctest
```

Each directory has its own `CMakeLists.txt` and builds only what is inside it. A new top-level
directory adds its `add_subdirectory` line to the root `CMakeLists.txt`; `arch/CMakeLists.txt`
adds every architecture directory itself.

## Build

The compiler is the one the engine ships with: ROCm's clang, `${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++` — the compiler `hipcc` drives. Every build and every compiled test uses it, the host backend included. A distribution compiler on PATH (CMake would pick `/usr/bin/c++`) is not a substitute, and neither is a different system GCC: results from another compiler are not comparable, so they do not verify a change.

The entry points name it, so no build depends on what happens to be on PATH: `./build.sh` passes `-DCMAKE_C_COMPILER=${ROCM_PATH:-/opt/rocm}/llvm/bin/clang` and `-DCMAKE_CXX_COMPILER=.../clang++` and stops with a named blocker when they are missing, and `CMakePresets.json`'s `default` preset (which `asan`, `tsan` and the test presets inherit) sets the same pair. A direct configure must name both:

```sh
cmake -S . -B build -G Ninja -DRAD_WITH_HIP=OFF \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Each result states the compiler path and version it was built with (`/opt/rocm/llvm/bin/clang++ --version`). If the shipped toolchain is missing, report that blocker; do not build with another compiler to get a green result.

A change is done when its directory compiles and its tests pass.

## Conventions

**No Python in the engine, its tools, its build or its test suite.** `spec.md` opens with this
constraint. Tests are C++ and CTest. The Python files in the tree are run by hand and nothing in
the build or the suite calls them: `tests/mkgolden.py` and `tests/mklfm2golden.py` regenerate the
reference numbers in `tests/mm_golden.h` and `tests/lfm2_golden.h` from transformers;
`tools/pad_tp.py` and `tools/pad_verify.py` pad a checkpoint for a tensor-parallel width and check
the result; `scripts/cardclock.py` monitors card clocks.

**C ABI at plugin boundaries, C++20 inside.** `abi/` is C and must stay compilable as C.
Everything under `core/` is C++20 and may use the standard library freely.

**Errors are values.** Return `RAD_OK` or a negative `RAD_E_*`. Do not throw across a function
that a plugin or the run phase can call. `fatal()` is for broken invariants only.

**Nothing unimplemented returns success.** An unwritten function returns `RAD_E_UNSUPPORTED` and
logs once at Warn with its own name. A stub that returns 0 produces fluent wrong output with no way
to bisect it (`spec.md` §17).

**No allocation and no synchronisation on the step path.** The arena is sized at declare. A
`malloc` needed inside `rad_arch_step` or the scheduler's per-step work is a declare-phase bug.

**`rad_memset_async` refuses `RAD_MEM_HOST` on a card; the host backend cannot reproduce that.**
`RAD_MEM_HOST` is pageable memory the GPU has never been shown, so the device backend returns
`RAD_E_UNSUPPORTED` naming the rule (`core/device/aql.cpp`). On the host backend all memory kinds
are the same memory, so a memset on host memory passes in a no-ROCm build and fails on a card. Use
`RAD_MEM_HOST_PINNED` or `RAD_MEM_HOST_MAPPED`, or memset on the host. `rad_memcpy_async` accepts
pageable memory but bounces it through pinned memory and waits, so a pageable copy on the step path
is a host stall; `aql_pageable_bytes()` counts them.

**Kernel plugins launch with `hipLaunchKernel` and the engine dispatches them.** On a card every
stream is an HSA queue the engine owns and writes AQL packets into (`core/device/aql.cpp`). The
executable defines `hipLaunchKernel` and HIP's kernel registration hooks, so a plugin built by
hipcc needs nothing special. A plugin must not call any other HIP stream API on an engine stream:
`hipEventRecord` and `hipStreamWaitEvent` are refused by name, and anything else would hand HIP a
stream it does not own. Use `rad_event_*` and `rad_memcpy_async`.

**Comments explain the decision, not the syntax.** Say why this and not the obvious alternative,
and name the cost when there is one. `libr4d/r4d_rows.cpp` is the model.

**Naming.** `rad_` prefix on everything with external linkage. Files are named for what they
provide. `snake_case` for functions and variables, `PascalCase` for types.
