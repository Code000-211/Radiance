# exl3import: an EXL3 checkpoint into a `.rad` container

This directory writes an exllamav3 (EXL3) checkpoint, as it is downloaded, into a `.rad` container. Every weight
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

Build it with the rest of the tree. It is part of the top-level build:

```sh
cmake -S . -B build -DRAD_WITH_HIP=OFF -DCMAKE_BUILD_TYPE=Release   # a convert needs no GPU
cmake --build build --target rad-convert libref libquant exl3_quant exl3_import
```

`build/radiance_home` is then a plugin home that holds everything a convert loads. To build against an installed
radiance instead, use `cmake -S plugin/exl3import -B build-exl3 -DCMAKE_PREFIX_PATH=<prefix>` and put
`build-exl3/home` ahead of the engine's home on `RADIANCE_HOME`.

Installed beside the serving plugins, `exl3_import` is inert. It claims the architecture `exl3import`, which no
checkpoint has, unless `EXL3_IMPORT_ARCH` names another id.

## Converting

```sh
export RAD_CONVERT=$PWD/build/bin/rad-convert RADIANCE_HOME=$PWD/build/radiance_home
PLAN_ONLY=1 plugin/exl3import/tools/exl3_convert.sh /models/DSV4.1-Flash-EXL3 /models/dsv4.1-flash-exl3.rad
plugin/exl3import/tools/exl3_convert.sh /models/DSV4.1-Flash-EXL3 /models/dsv4.1-flash-exl3.rad
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
  - **a layer whose experts differ** in K, codebook or shape at one weight. The container addresses a layer's
    experts by a fixed stride and refuses this (`core/format/radfile.h`).
  - an EXL3 linear whose rows or columns are not a multiple of 128: both rotations are 128-wide.
  - a tensor dtype radiance's safetensors reader does not take (F64, F8_E8M0, U16, U64), or a malformed EXL3
    linear.
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
  - A tensor under `.experts.<E>.` is expert `E` of its layer. Its slot is the rank of its name, with `E` taken
    out, among that layer's expert weight names. For `down_proj`, `gate_proj` and `up_proj` that gives
    `down=0, gate=1, up=2`.
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
- an MTP layer under its own prefix;
- two shards behind an index.

The test runs `tools/exl3_convert.sh` on that checkpoint, then checks every container entry against the source
with its own reader of `abi/rad_format.h` (`tests/check_container.py`). The codes planes are compared with
`tests/exl3_ref.py`, a numpy reference written independently of the C importer. The test also checks three
refusals:

- a container with one codes byte flipped must fail the check;
- a checkpoint with a half-integer K, a layer of mixed experts and a linear not in 128s must be refused, each
  by name;
- no container may be written for that checkpoint.
