# Writing a plugin

Three kinds of plugin extend the engine, each a shared object that exports a few C symbols and
links nothing of radiance's:

- a **kernel library** implements ops for some hardware (a card, a family of cards, the CPU);
- an **architecture** declares a model: its weights, its buffers, and the ops it issues in order;
- a **quantiser** turns a checkpoint's weights into the planes of an encoding, for `rad-convert`.

A plugin is compiled against the installed `abi/` headers alone, so a third party can ship one
without building the engine. The in-tree plugins (libr4d, libref, libavx, libquant and every
architecture) use the same interface: each is built with `rad_add_plugin`, builds on its own
against an installed radiance, and is found and ranked by what it declares, never by its name.
`ctest -L oot` (`tests/oot_plugins.sh`) rebuilds them out of tree against an installed prefix
(libr4d only when given a HIP compiler) and requires the same result.

§1-§9 cover kernel libraries, §10 a library for a new card, §11 and §12 architectures and
quantisers, and §13 building, installing and the search path.

---

## 1. Kernel library exports

```c
#include "rad_abi.h"          /* required */
#include "rad_plugin.h"       /* optional helpers -- see §6 */

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo i = {
        RAD_PLUGIN_KERNEL, "mykernels", "0.1", "a GEMM or two", "gfx1100",
    };
    return &i;
}

int                rad_kernel_schema_count(void);   /* the ops you define, if any */
const RadOpSchema* rad_kernel_schema_at(int i);
int                rad_kernel_count(void);          /* the kernels you implement */
const RadKernelInfo* rad_kernel_at(int i);
```

A plugin whose `rad_plugin_abi_version` differs from the engine's is refused at load, by name, with
both numbers printed. Return the macro, not the integer, and rebuild when it changes; it changes
with any shape change in a header a plugin compiles against (`rad_abi.h` lists them).

`build_target` is printed, not trusted: the targets the library was built for (`"gfx1100"`,
`"gfx11-generic"`, several separated by spaces or commas), `"host"` for CPU kernels, or `""` for a
portable plugin. The engine checks the code objects the library carries instead (§10).

Five more exports are optional, each looked up by exactly this name:

| symbol | meaning |
|---|---|
| `int rad_plugin_open(void)` | called once after load. `RAD_E_UNSUPPORTED` declines this machine: the library is unloaded and the rest of the installation carries on. Any other negative status is an error |
| `void rad_plugin_close(void)` | called once at shutdown |
| `int rad_kernel_concurrent(int i)` | nonzero: kernel `i` touches nothing but its operands and scratch, so independent launches may overlap |
| `int rad_kernel_collective(int i)` | nonzero: kernel `i` exchanges data with the other ranks' cards, so the engine keeps its own bulk link traffic away while it runs. 2: each launch ends that hold itself once the other ranks' bytes have landed, through the `link_gate` parameter (`rad_abi.h`). Without this export only the `all_reduce` op is a collective |
| `int rad_plugin_reference(void)` | nonzero: a **reference implementation**, ranked below every unnamed library, refused for serving unless `--debug-accept-reference-kernels`, and what `rad-kbench` checks the others against. libref declares itself this way |

---

## 2. Kernel constraints

```c
static const RadConstraint c_gemm[] = {
    RAD_CLE("M", 64),              /* M <= 64            */
    RAD_CDIV("K", 128),            /* K divisible by 128 */
    RAD_CIN("dtype", "fp8a8 w4a8") /* dtype in that set  */
};

static const RadKernelInfo k[] = {{
    .name = "mygemm_m64", .op = "gemm_nt_q", .family = "gemm",
    .computes = "row-major A times column-major B, fp8 in, bf16 out",
    .shape    = "M<=64, K%128==0",
    .dtypes   = "fp8a8, w4a8",
    .domain   = RAD_DOMAIN_DEVICE,
    .priority = 50,
    .constraints = c_gemm, .n_constraints = 3,
    .launch = my_launch,
}};
```

A kernel declares what it can serve. The operators are `RAD_CEQ`, `RAD_CLE`, `RAD_CGE`, `RAD_CDIV`
and `RAD_CIN`. The engine takes bucket boundaries from the `LE`/`GE`/`EQ` values and resolves once
per bucket, since a GEMM sees one row at decode and thousands in a prefill chunk (`spec.md` §2.2).
`DIV` and `IN` are not intervals, so neither is a band edge.

Declare only the band you serve: the engine believes the claim, and a kernel that claims
`M <= 2048` but is correct only to 64 is wrong. A wrong band shows only at concurrency, where M is
larger, so test the whole concurrency ladder, not C=1 alone.

**Machine constraints.** A key starting with `device.` is matched against what the engine measured
about the ranks' cards at launch, not against the op's parameters. `RAD_CEQ(RAD_FACT_P2P, 1)`
(`device.p2p`) says a kernel accesses another card's memory. Where some pair of the ranks' cards
cannot reach each other, the kernel is not selected and the next matching row takes the op; with
none, startup stops and the selection table names the fact (`device.p2p == 1, have 0 -- this
machine, measured at launch`). Every libr4d collective carries it and has a `_host` twin requiring
`device.p2p == 0` that exchanges through pinned host memory, so exactly one of the pair matches any
machine. A machine fact changes no tuning key. A tool selecting without an engine has measured
nothing, so such a kernel is not selected there.

**Priority.** `priority` breaks ties within one plugin, highest first, then declaration order.
Between plugins the hierarchy is absolute: the first plugin in `--kernels` order with any matching
kernel supplies it, however specialised a later plugin's kernel is. That makes a user's library an
override rather than a competitor.

---

## 3. Operand extents

`RadArgs.t` is positional, in the op schema's order. Each `RadTensor` carries the extent the caller
narrowed it to, generally smaller than the buffer. Size loops and grids from those extents, never
from a declared maximum: otherwise a single-row decode runs over every row of an arena buffer,
reading uninitialised memory and overwriting the next op's input.

Where an op takes `cu_seqlens`, derive the sequence count from its own length (`n = len - 1`), not
from a sibling array, and each sequence's rows from `cu[n]` and `cu[n+1]`, which are absolute
offsets into the token operands. This lets the engine pass a sub-range of the step (§8).

An absent optional operand is a null `RadTensor.data`, not a shorter operand count; reading it as
a shorter array shifts every later operand by one. `rad_arg_in()` in `rad_plugin.h` fetches an
operand correctly.

`RadTensor::space` is where the operand's bytes lie, as the engine's own placement knows it
(`RAD_SPACE_DEVICE`, `RAD_SPACE_HOST`, or `RAD_SPACE_UNKNOWN` when the engine declines to say -- a
raw pointer it did not allocate, or a weight a mover relocated: it republishes the pointer, not the
site). A KV operand is device memory, the block manager's pool. Ask it before asking the
driver: the backend locks its host pool itself (`core/device/aql.cpp`), and `hipPointerGetAttributes`
reports those ranges as unclassifiable rather than as host memory. A kernel that accounts for what
it read can then tell a device read from a link read. Where neither the operand nor the driver can
say, report the figure as unmeasured rather than as 0.

---

## 4. Optional kernel hooks

None are required. Each is a claim the engine acts on.

| hook | meaning |
|---|---|
| `init` / `fini` | per-resolved-instance state; `RadArgs.instance` hands it back |
| `scratch` | bytes of scratch a launch needs; the engine provides at least that |
| `layout` | the kernel holds this weight in its own order, not as the container stores it |
| `relayout` | produces that order from the container's planes, run by the loader |
| `unrelayout` | the inverse, so a tool holding the stored bytes can read the planes back |
| `opd_shape` | how to size and fill every operand, so a tool can call the kernel cold |
| `replaces` | the chain of conventional ops the kernel computes in one launch |
| `tunables` / `tune_valid` | the axes of the instantiation space, and which combinations are legal at a geometry |

`opd_shape` and `unrelayout` make a kernel checkable; without them, tools skip it by name instead
of guessing.

`replaces` must be exact ("this kernel stands in for this chain; check it against it"). A kernel
that is only close publishes no hook. The claim lets the engine treat the fused kernel and the
chain as alternatives, which the Model view draws as one branch.

`tunables` declares the axes and their legal values (`RadTunable` in `rad_abi.h`), not a list of
named variants. The core takes the cross product, so a ladder has no hole that silently runs the
next tile up. `tune_valid` rejects combinations independent lists cannot express (a bound coupling
two axes, a `K % (sk * unit)` rule); a rejected candidate is reported as skipped, not failed.
`rad-tune` measures the survivors at the shapes the model issues and writes the cache the selector
reads at declare. The chosen point arrives as ordinary parameters, read with `rad_args_geti()` like
`M`, so the launch needs no variant index. `deflt` runs where this machine has no measurement and
must be launchable; the loader refuses the plugin by name if `deflt` is not in `values`.

---

## 5. What a launch may do

On a card the `RadStream` a kernel receives is a queue the engine writes packets into itself
(`core/device/aql.cpp`), not a `hipStream_t`. The engine binary defines these HIP entry points and
serves them on its own queues:

- `hipLaunchKernel`, and therefore `<<<>>>` and `hipLaunchKernelGGL`;
- `hipMemcpyAsync`, `hipMemcpy2DAsync`, `hipMemsetAsync`;
- `hipStreamSynchronize` (not from a launch, which must not synchronise);
- `hipMemset`, which runs on the engine's queue for the current device and waits;
- `hipGetLastError` / `hipPeekAtLastError`, with HIP's semantics.

**Nothing else in HIP takes an engine stream.** `hipModuleLaunchKernel`, `hipExtLaunchKernel`,
cooperative launches, `hipLaunchHostFunc`, and libraries that launch through them (rocBLAS,
hipBLASLt, MIOpen, hipRTC modules) would hand HIP a queue it does not know. Header-only device code
(Composable Kernel's device templates, rocPRIM's block primitives) compiled into your own `.hip`
launches through `hipLaunchKernel` and works. `hipEventRecord` and `hipStreamWaitEvent` on an engine
stream are refused by name; ordering between streams is the engine's.

The engine reads the code object too, which limits a kernel to:

- **no hostcall**: `printf` and `assert` in device code need a buffer the queues do not provide,
  and the kernel is refused by name at its first launch. `rad_add_plugin` compiles device code with
  `-DNDEBUG` outside Debug for this reason;
- **no dynamic stack** (recursion, a variable-length array), since nothing sizes its scratch;
- **kernel arguments up to 768 bytes**, and static plus dynamic LDS up to 64 KiB a dispatch;
- **an uncompressed fat binary**. `rad_add_plugin` passes `--no-offload-compress`; a compressed one
  is refused at the first launch.

Device memory a kernel needs for itself comes from the `scratch` hook (§4), not `hipMalloc`, since
the engine's allocator knows what the card has left.

---

## 6. `rad_plugin.h`

A plugin that includes only `rad_abi.h` is complete. The helper header holds what every plugin
needs: the optional-operand fetch, parameters with defaults, row pitch, pointer alignment and the
narrowing conversions.

Use its conversions: they define what a result is. The reference contract is f32 accumulation,
IEEE round-to-nearest-even, no FMA and no reassociation, and a plugin whose bf16 narrowing
truncates cannot be checked against the reference.

---

## 7. Defining a new op

Only when `docs/OPS.md` has no name for it. That document is the op vocabulary and its rules: what
a schema must declare, and that the first plugin in hierarchy order to declare an op fixes its
schema (a later plugin declaring it differently is refused at load, both named). `ctest -R ops_doc`
checks it against the registries, so the rules live there only.

An op in `docs/OPS.md` is describable by the reference, and so checkable, even when your kernel
cannot describe itself.

---

## 8. Mixed steps in an architecture plugin

An **architecture** plugin with a linear or recurrent block must follow `spec.md` §10 in its
`step()`.

Continuous batching produces many steps that carry a prefill chunk beside decode rows. A block that
issues one path for the whole step sends its decode rows down the chunked path, which commits the
recurrent state directly and takes no `num_accepted`. A draft verified there cannot be rolled back:
the rejected tokens stay in the state and the model is conditioned on text it never emitted. The
output stays fluent and drifts, so the symptom is concurrent requests disagreeing with the same
requests run alone.

Issue the recurrent path over the first `RadBatch::n_seq_decode` sequences and the chunked path
over the rest; the core orders every step decode-first, so each half is contiguous. No kernel
change is needed (§3): `cu_seqlens` holds absolute offsets, so the prefill half is addressed by
passing `cu + n_seq_decode` with the token operands left at their base.

---

## 9. Checking your work

`rad-kbench` checks a library for correctness, completeness and speed without a model:

```sh
rad-kbench --kernels mylib,libr4d,libref --report mylib.md --bench
```

It replays a recorded fixture of libref's answers over a few hundred real model geometries
(`kernels.rkb` unless `--fixture` names another; `data/` holds a dense one and a routed-MoE one).
Rows are found as the engine finds them, but every library with a row for a case is run, so yours
appear beside libr4d's. The report covers:

- **correctness** against the recording, which keeps every output's full norm plus 1024 sampled
  values, so a kernel right at the samples but wrong in total energy fails;
- **completeness**, the op-by-library matrix, read from the registries, so a missing op is a gap
  whether or not the fixture exercises it;
- **speed**, timed on the same bytes in the same pass.

It also checks:

- **red zones**: every device allocation has a poisoned 256-byte margin on each side, verified
  after the launch, since in a harness an out-of-bounds write otherwise passes. AMD's device
  AddressSanitizer needs `xnack+`, which RDNA lacks;
- **fusion**: a row with `replaces` is run against the chain it names, out of your own library,
  and the outputs compared byte for byte;
- **divergence**: an output your `opd_shape` describes differently from libref's is reported by
  name with both descriptions and counted apart from the failures.

A skipped kernel is reported with the reason, usually a missing `opd_shape` (§4). Other
diagnostics:

- `--debug-selection` prints the selection table: which kernel took each band, and the constraint
  that refused each other candidate;
- `/graph` on a running server returns the declared graph and which declarations the run phase has
  issued, which finds a band resolved for nothing;
- `--debug-accept-reference-kernels` is required to serve with a reference fallback; without it a
  fallback is fatal, because a reference kernel in production would look like a performance
  problem;
- `rad-info --plugins` prints every plugin on the search path as the engine loads it: kind, version,
  declarations, whether it is a reference, whether its device code runs on this card, and its file.

---

## 10. A library for a new card

The engine's own device code (the copies and fills every transfer runs, the step's bookkeeping,
the weight-table patches) is compiled for every AMD GPU family's generic target (`gfx9-generic`,
`gfx9-4-generic`, `gfx10-1-generic`, `gfx10-3-generic`, `gfx11-generic`, `gfx12-generic`) beside
the build's own targets, so one engine binary starts on any of them. What it computes there depends
on the kernel library installed.

**Targets.** Configure with `-DRAD_GPU_TARGETS=gfx942` or `=gfx11-generic`; the default is the
installed radiance's targets. For each code object the engine asks the runtime which targets the
card accepts, in order (the chip, then its family: `gfx1036`, then `gfx10-3-generic`), each with
the card's modes (`gfx942:sramecc+:xnack-`), and takes the best fit in the fat binary: the chip
over the family, a code object for the card's mode over one for either.

**A library with nothing for this card is left out, not failed.** At load, the fat binaries a
kernel library registered are checked against the ranks' card. If none fits, its device rows are
left out of selection, the log says what it carries and what the card runs, and its host rows stay.
One installation can therefore carry a library per card. A library can also decline from
`rad_plugin_open` with `RAD_E_UNSUPPORTED`.

**Devices.** The engine uses the discrete cards if there are any (those with the first one's
architecture), and integrated graphics only when there is nothing else. `ROCR_VISIBLE_DEVICES`
chooses explicitly. On integrated graphics device memory is host memory: the budget is bounded by
what the host has available and leaves an eighth of host memory to the host.

The core assumes nothing about a card's wave size, LDS, fp8 hardware or links. Three points matter:

- **The vocabulary is what you implement.** A model runs on your library when every op its
  architecture issues has a row there (`docs/OPS.md`; `--debug-graph` lists them). The bf16 models
  in the tree issue only ops libref implements (MiniCPM5-2B: 13 model ops and 11 sampler ops), so
  libref's host code is the specification and `rad-kbench` the check. Ops `docs/OPS.md` marks
  **device-only** are libr4d's and have no reference; the fp8 models and a drafter with a 2-bit
  head issue some.
- **fp8 bytes are OCP E4M3** (`e4m3fn`, bias 7, max 448) wherever the engine reads them: container
  weights, the KV cache at `--kv-cache-dtype fp8`, activations between ops. A card whose fp8 is
  `e4m3fnuz` (gfx942) converts weights in its `relayout` hook (the codes keep their bits except
  0x80 and decode to half the value, so the scale doubles) and keeps activations and KV in OCP
  bytes, or converts at its own boundaries.
- **Tensor parallelism is ops you supply.** `all_reduce`, `all_gather` and the fused collectives
  are kernel-library rows like any other; the engine inserts none and assumes no link.

---

## 11. Architecture plugins

An architecture plugin declares a model in two phases (`spec.md` §3): `rad_arch_declare` names
every weight, buffer and op once, against the model's metadata, and `rad_arch_step` issues the ops
for one step. The contract is `abi/rad_builder.h` and `abi/rad_runtime.h`. The helpers the in-tree
architectures use (weight declarations, the gated-attention and MoE blocks, the MTP and DFlash2
drafters) are installed as `<arch/rad_arch.h>` and its siblings, and the in-tree plugins include
them by that spelling.

```c
const char* rad_arch_id(void);          /* "llama" -- what a container's header names */
const char* rad_arch_quant(void);       /* optional: "" unquantised, or the descriptor served */
int         rad_arch_serves_imports(void);  /* optional: 0 declines an imported quantisation */
int         rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
void        rad_arch_step(struct RadCtx* c, const struct RadBatch* batch);
int         rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out);          /* optional */
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta);             /* optional */
```

`rad_plugin_info` reports `RAD_PLUGIN_ARCH`. The engine picks the plugin whose
`(rad_arch_id, rad_arch_quant)` pair equals the container's, with no fallback, so one model family
at two weight formats is two plugins. A checkpoint's id is its `config.json` `model_type` without
underscores; `rad-convert --arch ID` overrides it. Every configuration key reaches the plugin in
`RadModelMeta`, and the plugin's name map says which checkpoint tensor each declared weight comes
from, so `rad-convert` needs no knowledge of the model.

A checkpoint whose quantisation was imported weight by weight (GUIDE §3.2: Quark, ParoQuant, GPTQ,
NVFP4) names its producing tool, which no plugin claims; it goes to the family's one quantised
plugin, and two are refused by name. A second quantised plugin that reads only its own containers
says so with `RAD_ARCH_SERVES_IMPORTS(0)` (`qwen35_w8a8` does), and imports keep their plugin.

- **Ops from `docs/OPS.md` run on any library.** An op libref implements has a host fallback, an
  oracle and a specification to check kernel libraries against. An op you invent runs only on a
  library that implements it.
- **A fusion is an op name.** The engine does not rewrite your graph, and `replaces` declares an
  equivalence for checking, not a substitution. Ask for the fused op and fall back to its parts
  where it does not resolve: an op no library serves is declared as `RAD_NULL_HANDLE`
  (`rad_op_resolved`) and is an error only if issued. The in-tree blocks do this for
  `qk_norm_rope_gate`.
- **Read §8** if the model has a linear or recurrent block.
- **Media preprocessing is the core's** (`core/mm`): resizing, patching, placeholder-token
  expansion and positions, in three processor families (Qwen-VL, Gemma 4, LFM2-VL) chosen by the
  container's `vision_config.model_type`. A vision model with another pipeline needs a core change.
  Audio goes through one of two feature extractors, Gemma 4's and Qwen3-ASR's (Whisper's), chosen
  by `audio_config.model_type`.
- **A speech model says how it hears**, through the optional
  `const RadAudioFormat* rad_arch_audio_format(const RadModelMeta*)` (`RAD_ARCH_AUDIO_FORMAT`): the
  tokens around a clip when its config names none, the turn a transcription is asked in, how a
  language is forced and named, and where the transcript starts in the reply. That is all
  `/v1/audio/transcriptions` knows about a model, so a new speech model is a plugin, not a server
  change. Like `rad_arch_chat_format`, the struct grows only at its end, behind its `struct_size`.

---

## 12. Quantiser plugins

A quantiser turns a weight, handed to it in f32, into the planes of an **encoding**
(`abi/rad_encoding.h`): the numbers in one canonical row-major arrangement, which a kernel library
lays out its own way at load. The contract is `abi/rad_quant.h`:

```c
int                     rad_quant_count(void);
const RadQuantizerInfo* rad_quant_at(int i);   /* name, doc, options, encoding, row_block, quantize, decode */
```

`rad_plugin_info` reports `RAD_PLUGIN_QUANT`. `rad-convert` drives quantisers through a
**recipe**: ordered rules from a weight's logical name to a quantiser and its options.

```
# PATTERN                    QUANTISER  OPTIONS
blk.*.ffn_*_exps.*.weight    myquant    bits=3 group=64
*                            rtn        codes=fp8_e4m3 block=128x128 scale=bf16
```

`rad-convert --list-quantizers` prints every loaded quantiser and its options. A recipe option the
quantiser does not declare is refused before anything is quantised. Two plugins naming the same
quantiser are refused, both named.

The core decodes the `plain` and `affine` schemes. Your own scheme is decoded by your `decode`
hook, which the oracle, `rad-kbench` and `rad-convert`'s error report use; return
`RAD_E_UNSUPPORTED` for a scheme you did not write. Serving a new encoding also needs a kernel
library whose `layout` hook accepts it and an architecture that declares the weight at it. The
in-tree architectures declare only the encodings their blocks were written for, so another format
needs its own kernel library and architecture plugin.

---

## 13. Building, installing and the search path

```cmake
cmake_minimum_required(VERSION 3.21)
project(mykernels LANGUAGES CXX HIP)   # CXX alone for a host-only plugin
find_package(radiance REQUIRED)
rad_plugin_defaults()                  # C++20, an optimising build type unless one was asked for
rad_add_plugin(mykernels kernels SOURCES rows.cpp HIP_SOURCES kernels.hip)
```

The kind is `kernels`, `architectures` or `quantizers`, the directory the engine loads it from.
Only `HIP_SOURCES` are compiled as HIP: for `RAD_GPU_TARGETS` (§10) or a `GPU_TARGETS` list,
optimised with `-DNDEBUG` outside Debug, uncompressed, against the HIP runtime the installed
radiance found. radiance builds its own plugins with the same `rad_add_plugin`, and every plugin
directory in the tree opens with these lines guarded by `if(NOT COMMAND rad_add_plugin)`, so each
builds inside the tree or on its own.

`find_package` also sets `RADIANCE_ABI_VERSION` (to refuse at configure time rather than at
`dlopen`), `RADIANCE_HOME`, `RADIANCE_KERNELS_DIR`, `RADIANCE_ARCHITECTURES_DIR` and
`RADIANCE_QUANTIZERS_DIR`. `cmake --install` puts the plugin under the home of the radiance you
built against; `RAD_PLUGIN_INSTALL_HOME` overrides that. Before installing, the build tree's
`radiance_home/` is a working home for the plugin alone.

**`$RADIANCE_HOME` is a search path**: homes separated by `:`, the first winning. Each may hold
`kernels/`, `architectures/` and `quantizers/`, and a plugin file in an earlier home shadows one of
the same name later. Your plugins go ahead of an installation without copying it:

```sh
RADIANCE_HOME=$HOME/radiance-plugins:/opt/radiance/share/radiance radiance --model ...
```

The container image takes one the same way, mounted:
`-v /my/plugins:/plugins -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance`.

**The kernel hierarchy** is `--kernels` (or `$RADIANCE_KERNELS`), an ordered list of the kernel
libraries to use: `--kernels mykernels,libr4d` puts yours before libr4d. An entry matches a
plugin's name or its file (`libr4d`, `r4d`, `libr4d.so` and a path to it are one library); an entry
matching nothing is warned about by name. Separate entries with commas or colons
(`mykernels:libr4d` is the same list); the engine and every tool read `--kernels` and
`$RADIANCE_KERNELS` by this one rule, and an empty `$RADIANCE_KERNELS` counts as unset. A kernel library the list does not
name is not loaded, so name every library you want, including the host library (`libavx`) and the
reference (`libref`). With no list every library found loads, a reference library (§1) last.
