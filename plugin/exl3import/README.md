# RadianceEXL3Convert: an EXL3 checkpoint into a `.rad` container

This project writes an exllamav3 (EXL3) checkpoint, as it is downloaded, into a `.rad` container. Every weight
keeps its checkpoint name, nothing is quantised and nothing is lost. It exists so that a model's container can be
written **before** an architecture plugin that serves the model exists. The first use is DeepSeek-V4.1-Flash from
an EXL3 conversion (for example `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw`).

It has two plugins and two tools:

| piece | what it does |
|---|---|
| `quant/` -> `quantizers/exl3_quant.so` | the `exl3` importer: an EXL3 linear's trellis, `suh` and `svh` copied into the canonical exl3 encoding (`include/exl3_encoding.h`) at its own K and codebook |
| `arch/` -> `architectures/exl3_import.so` | a **convert-only** architecture: it declares every tensor of the checkpoint under its own name, and no buffers, ops or logits. It cannot serve |
| `tools/exl3_inventory.py` | what the checkpoint holds and every reason it cannot be written; writes the hole file |
| `tools/exl3_convert.sh` | the one command: inventory, hole file, space check, plan, convert |

The importer, the encoding and the EXL3 decode come from
[Dyluhn/radiance-exl3](https://github.com/Dyluhn/radiance-exl3) v1.3.0 (`c88ba2b`, Apache-2.0,
`NOTICE.radiance-exl3`). There they import Qwen3.8-Flash-Next's routed experts only. This copy adds the generic
form: any `<X>.weight` whose `<X>.trellis` the checkpoint holds, with K and codebook read per weight.

## Building

The plugins are host code only, and a convert needs no GPU. They are written against radiance **1.10.0, plugin
ABI 26**: the [Code000-211/Radiance](https://github.com/Code000-211/Radiance) fork at the commit `RADIANCE_VERSION`
names. A build against another ABI is refused at configure time, and a plugin of another ABI is refused at load.

**Standalone, against an installed radiance** (its headers, CMake package and `rad-convert`):

```sh
# a radiance install prefix, host-only: in the radiance checkout
cmake -S . -B build -G Ninja -DRAD_WITH_HIP=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build && cmake --install build --prefix $HOME/radiance-1.10.0

# this project
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/radiance-1.10.0 -DCMAKE_BUILD_TYPE=Release
cmake --build build && cmake --install build   # -> build/home/{architectures,quantizers}
ctest --test-dir build -L host --output-on-failure
```

`build/home` is this project's plugin home. Put it ahead of the engine's own on `RADIANCE_HOME`:
`RADIANCE_HOME=$PWD/build/home:$HOME/radiance-1.10.0/share/radiance`.

**In radiance's tree:** this repository is also `plugin/exl3import` on the fork's `claude/jolly-hamilton-n405gx`
branch, and there it is built with the rest of the tree. There `build/radiance_home` holds every plugin a convert loads.

Installed beside the serving plugins, `exl3_import` is inert. It claims the architecture `exl3import`, which no
checkpoint has, unless `EXL3_IMPORT_ARCH` names another id.

## Converting

```sh
export RAD_CONVERT=$HOME/radiance-1.10.0/bin/rad-convert
export RADIANCE_HOME=$PWD/build/home:$HOME/radiance-1.10.0/share/radiance
PLAN_ONLY=1 tools/exl3_convert.sh /models/DSV4.1-Flash-EXL3 /models/dsv4.1-flash-exl3.rad
tools/exl3_convert.sh /models/DSV4.1-Flash-EXL3 /models/dsv4.1-flash-exl3.rad
```

Run `PLAN_ONLY=1` first. It reads only headers (seconds), prints the inventory, writes the hole file and runs
`rad-convert --plan-only`. Look for these lines in the inventory:

- `arch deepseekv4 quant exl3`: the id the container will carry. A plugin that serves the container must claim
  exactly this id with the quantisation `exl3`. Set `EXL3_IMPORT_ARCH` to carry another id.
- `K=.. <codebook> N linears`: every bit width the checkpoint uses.
- `FAIL` lines. Each one stops the convert:
  - **half-integer K** (`trellis last dimension 16K+8`, exllamav3 1.5.2 and later): no decoder here reads it.
    A 2.50 bpw conversion may consist entirely of this. If so, the decode (`common/exl3_decode.c`) has to learn
    it first.
  - a layer whose experts have more than 8 weights each: a movement unit holds at most 8.
  - an EXL3 linear whose rows or columns are not a multiple of 128: both rotations are 128-wide.
  - a tensor dtype radiance's safetensors reader does not take (F64, F8_E8M0, U16, U64), or a malformed EXL3
    linear.
- a `mixed` line, for a checkpoint whose experts differ inside a layer, e.g. SAGE-style mixed bit widths or an
  expert that lacks a weight. This is not an error: see "Mixed experts" below. The line says how many experts
  and how many GiB end up outside the expert groups. That share is what expert placement and the host-pool
  offload cannot move, so it is the number to look at before serving.
- `WARN` lines: tensors of a linear the importer does not read (`.su`/`.sv`, an older layout), or a bias, which
  is written as a weight of its own.

The convert needs:

- **Disk:** about as much as the checkpoint for the container, which the inventory prints as `container about N
  GiB`. The script checks the free space first.
- **A filesystem with sparse files** for the hole file (ext4, xfs, btrfs; not exFAT). Its apparent size is 2 bytes
  a weight, hundreds of GiB for a model of this size, but it takes almost nothing on disk.
- **RAM:** rad-convert holds each weight it is converting in f32, one per worker thread. The largest is usually
  the lm_head, at vocabulary x hidden x 4 bytes. Not measured on a model of this size.

The script refuses an `OUT` that already exists. It writes `OUT.part` and renames it when rad-convert succeeds.
`OUT.inventory.json` (the full report) and `OUT.recipe` stay beside it. `OUT.holes.safetensors` can be deleted
afterwards.

### Mixed experts

The container addresses a layer's routed experts as an array: base + expert number x stride
(`core/format/radfile.cpp`, `index_experts`). So every expert in a layer's expert group has one layout: the same
encoding and shape at every weight. Radiance's own containers mix formats inside a layer anyway, by keeping the
odd ones out of the group: a 4-bit Qwen3.8-Flash-Next layer's protected bf16 experts are tensors of the layer
(`arch/common/rad_block_moe_fp8.h`). This project does the same for any checkpoint:

- per layer, the most common expert form (the encoding and shape of every weight) is the layer's expert group.
  A tie goes to the form of the lowest-numbered expert;
- the group's experts are numbered 0, 1, 2, ... in expert order;
- every other expert's weights are tensors of the layer (expert `-1`).

Nothing is converted or lost: every weight keeps its checkpoint name and its own K. What differs is serving.
Radiance places, prefetches and offloads experts unit by unit through the group only. An expert outside it stays
with its layer. Radiance itself keeps at most 8 experts a layer per rank this way. A pack where a large share of
experts falls outside the groups may therefore need its experts split into one group per bit width, or a
container format with an offset per expert. Both are a change to the serving side, not to this convert.

### Why a hole file

rad-convert plans every weight from a checkpoint tensor of that name, and an EXL3 checkpoint has no `<X>.weight`,
only `<X>.trellis`, `.suh` and `.svh`. `exl3_inventory.py --holes` writes one BF16 `<X>.weight [n, k]` for each
EXL3 linear, as a safetensors header followed by a hole. rad-convert merges it in with `--add`. The importer
ignores the rows it is handed and reads the planes from `src=`, so nothing ever reads a value from the holes. This
is how radiance-exl3 converts too (its `tools/skeleton_ckpt.py`).

## The contract: what the container holds

The plugin that serves the container is written later and declares against these names. It can read anything
here with `rad_weight_encoding`.

- **Names.** One entry per checkpoint tensor, under the checkpoint's own name (`model.layers.3.mlp.experts.17.
  down_proj.weight`, `model.embed_tokens.weight`, ...). An EXL3 linear `<X>` is one entry `<X>.weight`. Its
  `.trellis/.suh/.svh/.mul1/.mcg` are not entries of their own.
- **EXL3 entries.** Encoding `exl3:codes=uK[1x1],suh=f16[*x1],svh=f16[1x*]/<codebook>/h128` over the logical shape
  `[n, k]` (rows = output features, as a PyTorch `Linear` weight), at the weight's own K (2..8) and codebook
  (`mul1`, `mcg`, `3inst`). The codes plane holds trellis symbols in radiance's canonical order
  (`common/exl3_decode.h`, "RADIANCE'S VIEW"). The `exl3` quantiser's `decode` hook turns it back into f32.
  radiance-exl3 checks that decode bit for bit against a golden decode. There are no stacked
  (`suh.1`) entries: gate and up stay separate weights.
- **Every other entry** is `plain`, at the checkpoint's dtype, shape and bytes.
- **Movement groups** (`layer`, `expert`, `slot`):
  - A tensor under `<P>layers.<N>.` belongs to layer `N` when `<P>` is the trunk's prefix: the one with
    `num_hidden_layers` layers, otherwise the first by name. Every other prefix (an MTP head such as
    `mtp.layers.0.`, a vision tower) is numbered after the trunk, from `num_hidden_layers` up, prefixes in name
    order. This matches how radiance's qwen4exp declares its MTP head at layer `num_hidden_layers`.
  - A tensor under `.experts.<E>.` is routed expert `E` of its layer. If expert `E` is in the layer's expert
    group (see "Mixed experts"), its `expert` is its place among the group's members in expert order, which is
    `E` itself when no expert of the layer falls outside. Its slot is the rank of its name, with `E` taken out,
    among that layer's expert weight names. For `down_proj`, `gate_proj` and `up_proj` that gives `down=0,
    gate=1, up=2`. An expert outside the group has `expert = -1` and `slot = 0`.
  - The real expert number is always the one in the name.
  - Everything else has expert `-1`.
- **Metadata.** Every `config.json` key, as rad-convert carries it. The architecture is the checkpoint's id (or
  `EXL3_IMPORT_ARCH`), the quantisation is `exl3`, and `exl3_import.src` names the source directory, which is
  recorded relative to where the convert ran.

A later convert of the same checkpoint with the serving plugin can take this container as `--reuse`. Every weight
that plugin declares at the same name, shape, encoding and group, under the same recipe rule, is then copied
from it without reading the checkpoint again.

## Tests

`ctest -R exl3_import_convert` (label `host`, needs `python3` with `numpy`) runs
`tests/make_fixture.py` to build a two-layer, DeepSeek-shaped EXL3 checkpoint. It has:

- K 2, 3, 4 and 6 in all three codebooks;
- a bias on an EXL3 linear;
- plain F16, BF16 and F32 tensors of rank 1 to 3;
- routed experts, shared experts and a router;
- with `--mixed-experts`, experts that differ inside their layer: one at another K in the middle of a layer,
  one as the layer's expert 0, and one that lacks a weight;
- an MTP layer under its own prefix;
- two shards behind an index.

The test runs `tools/exl3_convert.sh` on that checkpoint and on its `--mixed-experts` form, then checks every container entry against the source
with its own reader of `abi/rad_format.h` (`tests/check_container.py`). The codes planes are compared with
`tests/exl3_ref.py`, a numpy reference written independently of the C importer. The test also checks three
refusals:

- a container with one codes byte flipped must fail the check;
- a checkpoint with a half-integer K and a linear not in 128s must be refused, each by name;
- no container may be written for that checkpoint.
