# Radiance — tools reference

Every flag of every tool, with examples. For getting started, read [GUIDE.md](GUIDE.md) first.

Six binaries are installed beside `radiance`, plus the shell harnesses in `scripts/`:

| | |
|---|---|
| [`rad-info`](#rad-info) | print what a `.rad` container holds. Loads no plugins |
| [`rad-convert`](#rad-convert) | safetensors, GGUF or `.rad` in, `.rad` out: quantised by a recipe, or extended in place |
| [`rad-kbench`](#rad-kbench) | correctness, completeness and speed of every kernel library |
| [`rad-tune`](#rad-tune) | benchmark each kernel's tunable space at this model's shapes, on this machine |
| [`rad-schemas`](#rad-schemas) | print and diff the op schemas plugins declare |
| [`rad-tokcheck`](#rad-tokcheck) | hold the tokeniser to its reference encoder and to HuggingFace's ids over real text, and time it |
| [`radiance --kld-*`](#the-kl-mode) | KL divergence of a quantised model against the bf16 one, at every position of a corpus |
| [`scripts/`](#the-shell-harnesses) | bring a model up, measure it, assert on what comes back |

Three of them — `rad-convert`, `rad-kbench`, `rad-tune` — **are the engine's startup with a
different last step**. They load the same plugins in the same hierarchy and run the same declare
phase. A tool that resolved kernels differently from the engine would be a tool whose answers are
about a different program.

Every tool that loads plugins honours `RADIANCE_HOME` and `RADIANCE_KERNELS`.

---

## rad-info

```
rad-info [options] <model.rad>

  -v, --verbose      list every tensor, not just the per-unit summary
      --meta         print the metadata table and stop
      --vocab        print the vocab summary and its normaliser chain, and stop
      --profile N    print the top N entries of the expert profile (default 20, 0 = all)
      --recipe       print the recipe the container was converted with, and stop
```

This is the tool you reach for when something is wrong, so it is written for reading: aligned
columns, byte counts in human units, and every number that could mislead labelled with what it
actually counts. **It opens the container read-only and loads no plugins** — a container names no
kernel library, so there is nothing to load.

```sh
rad-info flashnext-w4.rad
```

```
flashnext-w4.rad
  format        RAD2, 166.4 GiB, little-endian
  architecture  qwen4exp
  model         Qwen3.8-Flash-Next
  source quant  (none)
  recipe        14 line(s) -- --recipe prints it
  created by    rad-convert ...

  section          offset        bytes    count
  strings             256      1.2 MiB        -
  metadata        1258752      8.4 KiB      214
  directory       ...
  planes          ...
  encodings       ...
  data              ...        166.1 GiB    ...
```

Every weight is listed with its **encoding** — `fp8_e4m3*bf16[128x128]`, `i4*bf16[1x128]/fwht128`,
`bf16` — its logical shape, and what made it: the checkpoint as it stood, or the quantiser and
options of the recipe rule that matched. The totals are by encoding and by quantiser.

```sh
rad-info --meta m.rad                         # the metadata table alone
rad-info --vocab m.rad                        # vocab + normaliser chain
rad-info --recipe m.rad                       # the recipe, expanded, one rule a line
rad-info -v m.rad | head -40                  # every tensor, its encoding, shape and planes
rad-info --profile 0 m.rad                    # the whole expert-popularity profile
```

---

## rad-convert

```
rad-convert [options] <input> -o <model.rad>

  <input>              a .gguf, a .safetensors, an index.json, a checkpoint directory, or a
                       .rad container (extended, not re-encoded; see --in-place)
  -o, --out FILE       the container to write
      --recipe FILE    the quantisation recipe (below)
      --quant RULE     'PATTERN=QUANTISER:key=value,...', ahead of the recipe's rules. Repeatable
      --list-quantizers  every quantiser loaded and the options it takes
      --arch ID        override the architecture id from the checkpoint
      --home DIR       $RADIANCE_HOME
      --kernels A:B    kernel plugin hierarchy, first wins. Default $RADIANCE_KERNELS
      --tokenizer F    tokenizer.json; its declared chain is baked instead of being
                       reconstructed from a pre-tokenizer name
      --imatrix F      llama.cpp imatrix: expert profile, column importances, error weighting
      --set K=V        a metadata key, seen by declare AND written into the container. Repeatable
      --max-tok N      the max_tok declare is run at (default 8192)
      --max-ctx N      the context bound declare is run at. Default: the checkpoint's
                       training context
      --draft-model DIR  a SECOND checkpoint: a drafter that ships as its own repository,
                       merged under a `draft.` prefix
      --draft-head FILE  a safetensors file merged UNPREFIXED: the target matrices a drafter
                       borrows but does not ship (a bf16 lm_head, an embedding)
      --kv-cache-dtype T  bf16 | fp8, the cache width declare is run at. Default: the first
                       at which every op resolves; the container is the same at either
      --max-spec N     the max_spec declare is run at (default 4, or block_size - 1 for a
                       drafter that states it). 0 leaves the draft head out
      --reuse FILE     copy every weight an earlier container holds identically
      --in-place       with --reuse: extend that container instead of writing a new one.
                       A .rad input requires it, and then defaults --reuse and -o to itself;
                       entries keep their offsets and new weights go past the end
      --plan-only      declare, plan every weight's encoding, print the plan, then STOP
      --report N       the worst N quantisation errors to print (default 20)
  -v, --verbose
```

### What it does

A container holds each weight as the **canonical planes of its encoding** — codes, scales, zero
points, a codebook — row-major, and no kernel library is named anywhere in it. The kernels
rearrange those planes into their own layouts when the model loads, so a kernel library can change
any layout it has without a reconvert. What rad-convert decides is the *numbers*:

1. Reads the checkpoint's `config.json` (or GGUF metadata).
2. Loads the architecture, kernel and quantiser plugins and runs **declare** — the same phase the
   engine runs. The architecture plugin's name map says which checkpoint tensors make each declared
   weight; `rad_weight_encoding` answers each one with the encoding the **recipe** gives it.
3. Every op resolves, and every resolved kernel's `layout` hook is asked about the planned encoding.
   A recipe nothing can serve is refused here, before a byte is written.
4. Writes each logical weight once: the checkpoint's own bytes where no rule matches, or the planes
   the matching rule's quantiser writes, a block of rows at a time.
5. Measures every quantised weight against its source through the core's decoder and prints the
   worst, imatrix-weighted when `--imatrix` is given.
6. Bakes in the tokeniser, the chat template, the metadata, the recipe and any draft head.

A **`.rad` input** takes a different path through the same phases. Each entry it holds is the
encoding a quantiser already produced, so it is never decoded or requantised: it is carried at its
own offset with its own encoding, quantiser, options and shape. Only what the run adds — a merged
drafter, a supplied head — is planned, and steps 4 and 5 apply to those alone.

### Recipes

Ordered rules from a logical weight name to a quantiser and its options. **The first rule whose
pattern matches decides**; a weight no rule matches keeps the checkpoint's encoding, which has to be
one the engine serves as it is — bf16, f16, f32, or block-scaled FP8 (`X.weight` with
`X.weight_scale_inv`). Anything else is named and refused.

```
# PATTERN                      QUANTISER  OPTIONS
blk.*.ffn_*_exps.*.weight      gptq       codes=i4 group=128 scale=bf16 transform=fwht128 calib=$CALIB
blk.*.ple_ngram.weight         rtn        codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=1.99317932128906e-4
output.weight                  rtn        codes=fp8_e4m3 block=128x128 scale=bf16
blk.*.weight                   rtn        codes=fp8_e4m3 block=128x128 scale=bf16
```

A pattern is a glob over the whole name (`*` crosses dots, `?` is one character). `$NAME` in an
option is the environment's, and an unset one is refused. `--quant` adds rules on the command line,
ahead of the file's: `--quant 'output.weight=rtn:codes=fp8_e4m3,block=128x128,scale=bf16'`.
A quantiser that declines a weight — `rtn` declines a rank-1 tensor, so a pattern that happens to
match a norm keeps the norm as it is — leaves it to the checkpoint's encoding.

`rad-convert --list-quantizers` prints every quantiser and its options. The built-in set is
`libquant`:

| quantiser | |
|---|---|
| `rtn` | round to nearest on a grid; the scale of each block chosen by `rule` |
| `gptq` | error feedback against the layer's input Hessian (`calib=DIR`, damped by `damp`, default 0.01 of its mean diagonal), scales fixed from the weight first; `act_order=1` quantises the columns by descending input energy and stores the permutation as the encoding's `perm` transform; `cd=N` adds N sweeps of coordinate descent after the loop, every code re-chosen at its conditional optimum against the same Hessian |
| `awq` | each input column scaled by its activation magnitude to a searched power, then a per-block clip search, both against the Hessian; the column scales are the encoding's last scale level |
| `autoround` | sign-gradient descent on a rounding offset a value and a range factor a block, against the Hessian; `iters` steps of N·K² multiply-adds each |
| `paroquant` | a channel scaling and `stages` rounds of Givens pair rotations, learned against the Hessian, then AutoRound on the rotated weight; the encoding's `givens` transform |
| `ggml` | `type=` one of llama.cpp's formats -- q4_0, q4_1, q5_0, q5_1, q8_0, q2_K..q6_K, iq4_nl, iq4_xs, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m -- by llama.cpp's own search, imatrix-weighted, the codes byte for byte llama.cpp's, stored as each type's affine planes; iq2_xxs, iq2_xs and iq1_s need `--imatrix` |
| `cast` | a float weight in another float dtype (`dtype=` f32, bf16, f16, fp8_e4m3, fp8_e5m2), no scale |

and the grid options every quantiser but `ggml` and `cast` takes:

| option | |
|---|---|
| `codes` | `i2`..`i8`, `i16`, `i32` symmetric; `u1`..`u8`, `u16` with `zero=`; `fp8_e4m3`, `fp8_e5m2`, `fp6_e2m3`, `fp6_e3m2`, `fp4_e2m1` |
| `table` | a codebook the codes index instead: `nf4`, `iq4nl`, `w4nl`, `w5nl`, `binary` (one bit), `ternary`, or the values, `;`-separated |
| `group=N` / `block=RxC` | where the scales sit; `*` is a whole extent |
| `scale` | f32, bf16, f16, fp8_e4m3, e8m0; under `scale2` also an integer sub-scale, u4..u8, i6, i8 |
| `scale2`, `group2` / `block2` | a second scale level the first is relative to (default the whole tensor) |
| `scale2_value` | the second level fixed at this value, exact in `scale2`'s dtype, for every block of it |
| `zero` | an integer zero point a block, making the grid asymmetric |
| `transform=fwhtN` | rotate each row by the Hadamard first |
| `rule` | `absmax`, `mx` (the OCP shared exponent), `fixed` (`scale_value=`), `search` (least squared error over candidates and their refits, imatrix-weighted) |
| `clamp` | a signed code's range, `full` or `sym` |

So MXFP4 is `codes=fp4_e2m1 group=32 scale=e8m0`, NVFP4 `codes=fp4_e2m1 group=16 scale=fp8_e4m3
scale2=f32`, NF4 `table=nf4 group=64`, a K-quant-like grid `codes=i4 group=32 scale=u6 scale2=f16
group2=256`, and a ternary BitNet weight `table=ternary`.

**A container can hold only what some kernel reads.** The converter checks every planned encoding
against the layout hooks of the kernels declare resolved and refuses one nothing accepts, so a
grid libquant can write is a container's only once a kernel library takes it. libr4d reads the
plain floats; block-scaled fp8 and fp8 rows; int8 block rows (the W8A8 trunk and lm_head); the
routed experts as fp8, int4 (`i4*bf16[1x128]/fwht128`) or the `w4nl`/`w5nl` codebooks under E4M3
scales ([MOE-W4.md](MOE-W4.md)); the dense w4, w8 and mxfp4 GEMM rows; and the 2-bit draft head.
Every encoding decodes through the core all the same, which is what the converter's error report,
the oracle and these quantisers' tests read it with.

The recipes for the containers this tree serves are in `data/recipes/`, and the conversions of
record in `scripts/convfn.sh` and `scripts/convdspark.sh` use them.

### Examples

```sh
# Flash-Next as it is served (w4nl64-i8lm.rad): codebook experts by GPTQ against the calibration
# Grams in $CALIB, an int8 trunk (docs/MOE-W4.md, "What is served"). The recipe's expert rule
# reads calib=$CALIB, and an unset variable is refused
CALIB=<calibration dir> rad-convert <Qwen3.8-Flash-Next checkpoint> \
    -o w4nl64-i8lm.rad --recipe data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home $RADIANCE_HOME -v

# A rule given on the command line goes ahead of the recipe's: here the experts at plain int4 by
# GPTQ, under a recipe whose own expert rule rounds to nearest
CALIB=<calibration dir> rad-convert <Qwen3.8-Flash-Next checkpoint> -o flashnext-w4.rad \
    --quant 'blk.*.ffn_*_exps.*.weight=gptq:codes=i4,group=128,scale=bf16,transform=fwht128,calib=$CALIB' \
    --recipe data/recipes/qwen4exp-w4-hc8m.recipe --home $RADIANCE_HOME -v

# A block-fp8 checkpoint needs no recipe: every weight is kept as it is. It needs no container
# either -- `radiance --model` serves the checkpoint directly -- unless a tier only a container
# has is wanted (GUIDE §3.4)
rad-convert <Qwen3.8-27B-FP8 checkpoint> -o q38-27b-fp8.rad

# Check the plan without writing 166 GiB
rad-convert $SRC -o /dev/null --recipe r.recipe --plan-only -v

# Merge a drafter that ships as its own repository
rad-convert <MiniCPM5-2B checkpoint> -o minicpm5-2b-dspark.rad \
    --draft-model <MiniCPM5-2B-DSpark checkpoint> --recipe data/recipes/minicpm5-2b-dspark.recipe

# Add the original DFlash2 drafter to a container this tree already serves. The original target is
# never written: copy it once, then extend the copy. Every entry it holds keeps its offset,
# encoding and bytes; only the drafter's tensors and the supplied head go past its end
cp --reflink=auto w4nl64-i8lm.rad work.rad
rad-convert work.rad --in-place --draft-model <Qwen3.8-Flash-Next-DFlash2 checkpoint> \
    --draft-head teacher-matrices.safetensors --set draft.bf16=1 --max-spec 7

# The declared block is the default: block_size 8 is one anchor plus seven proposals, and
# --num-speculative-tokens auto resolves to 7. A SHORTER runtime block is served at exactly the
# rows it asks for; a longer one than the container declares is refused.
radiance --model work.rad --num-speculative-tokens 7
radiance --model work.rad --num-speculative-tokens 3   # 4 query rows, 3 proposals

# Re-quantise only the experts of an existing container
rad-convert $SRC -o new.rad --recipe new.recipe --reuse old.rad
```

### The flags that catch people

**`--tokenizer`.** Pass it for a GGUF. Without it the tokeniser chain is reconstructed from a
pre-tokeniser *name*, which is a guess; a safetensors checkpoint's `tokenizer.json` is found beside
it. Also make sure a `chat_template.jinja` sits beside `tokenizer_config.json` in the checkpoint —
newer repos moved the template out of the JSON, and the symptom of missing it is `chat is 501`.

**`--set K=V` is how a convert-time choice that must also hold at serve time is stated once.** The
key is handed to declare *and* written into the container. What a weight is stored as is not such a
choice: the encoding is in the container, and the plugin reads it from there.

**`--max-ctx`.** Some architectures' graphs depend on the context bound: Qwen4-Exp sizes its QSA
buffers by it, and refuses a `--max-tok` and context pair whose score plane would pass 2 GiB.
Default is the checkpoint's training context.

**`--max-spec`.** A *container* carries what the model can serve, so a draft head is written even
when this run would not have drafted. `0` leaves it out. A DFlash2 drafter that states
`draft.dflash_config.block_size` is declared at `block_size - 1` by default — **block 8 is one
anchor plus seven proposals** — so the conversion states `--max-spec 7` (block_size 8 reaches the
same 7 by default). That declared block is the DEFAULT and the source config, not proof of a
training length: the [model card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) states
"Speculation block size: 4 (3 draft tokens per verification step), limited by current QSA
implementation" for the deployed drafter, and its QuickStart names the same block size. A serve of
Qwen4Exp may therefore ask for a shorter runtime block with `--num-speculative-tokens N`.
The drafter then allocates and issues exactly `N + 1` query rows (`N` proposals). A block longer
than the declared one is refused by name as outside the geometry the container describes. A serve
never rewrites the stored config.

**`--draft-model` here is real** — unlike on the engine, where it is refused. This is where a
drafter gets merged: its tensors go in under a `draft.` prefix and its config under `draft.` keys.
It has no embedding and no `lm_head` of its own because it uses the target's.

**`--draft-head` supplies the target matrices a drafter borrows but does not ship.** A bf16 DFlash2
declares its draft head over the target's `lm_head.weight`, and a container whose recipe made that
head int8 holds no bf16 rows. **Give it the original BF16 head from the teacher checkpoint**
(`--draft-head teacher-matrices.safetensors`), holding the tensor under its checkpoint name —
`lm_head.weight`, not the container's logical `output.weight`. Its tensors merge *unprefixed*, so
the plugin's name map finds them exactly as it would in a safetensors target. Do not dequantise the
container's i8 head: that is the lossy wire the bf16 drafter exists to avoid. A name an entry
already serves is refused.

**`--set draft.bf16=1` writes the bf16-mode marker.** `draft.bf16` is the key
`rad_block_dflash2.h` documents for a drafter converted without a quantising recipe, and `--set`
hands it to declare *and* writes it into the container, so the conversion and the serve read the
same statement. The tensors cannot carry it: a bf16 drafter and the legacy fp8 one hold the same
ones and differ in whether a recipe ran.

**Numerics and parity of BF16 DFlash2:** `numerik: verlustfrei-nicht-bitgenau`.
Weights, stored activations and draft KV stay BF16. `hc_mean` sums in FP32 and rounds once.
The existing `dflash_select` kernel retains FP32 selector intermediates where the SGLang
reference rounds some intermediate results to BF16; candidate IDs can therefore differ.
This keeps intermediate precision equal or higher, but does not establish bit-identical
reference draft selection. Check target-token parity separately; BF16 mode implies no gate PASS.

**`--reuse` matches a weight on its encoding, quantiser, options and shape.** Whatever calibration
made the old planes is carried with them: pass `--reuse` only when the source and the calibration
are the same.

**`--in-place` is the route for an existing container, and it does not copy.** The file extended is
the file read, so **copy the original target first** (`cp --reflink=auto old.rad work.rad`, or plain
`cp` without a reflink filesystem) and point `rad-convert work.rad --in-place` at the copy; a
different `-o` is refused rather than duplicating 100 GiB. **Every entry the container already holds
is retained**: it is recorded at its own offset with its own encoding, quantiser, options and shape,
and no byte of it is decoded or rewritten. New weights, the drafter's included, go past the old end
and the header is rewritten last. `<file>.pre-append` holds the old header and length for a
byte-exact rollback. A weight that would change encoding and a *new expert* are refused — neither
can join the existing layout. A recipe rule that matches a carried entry is refused too: scope the
recipe to the weights this run adds.

**`--imatrix`** supplies the expert-popularity profile, which the placement planner ranks units by,
hands each quantiser its weight's column importances, and weights the error report.

**`calib=DIR`** (`gptq`, `awq`, `autoround`, `paroquant`) is a directory an engine run wrote with
`RADIANCE_CALIB_DIR` — each layer's Gram matrices, of the input in the domain the kernel multiplies
it in: a container that stores a weight rotated records the rotated input, and a recipe that rotates
the same weight reads it as it stands. A weight with nothing there is quantised the uncalibrated
way and says so; `awq`, `autoround` and `paroquant` take the imatrix's diagonal instead when
`--imatrix` gives one. The three-step calibrated recipe is in [MOE-W4.md](MOE-W4.md).

**`autoround` and `paroquant` are slow by construction.** AutoRound's 200 default steps each cost
N·K² multiply-adds — on a 16-core Zen 5 at ~2.5 TFLOP/s that is minutes for a 17408-wide down
projection and most of a working day for a 27B dense model. `iters=50` is a quarter of it.

---

## rad-kbench

It checks every kernel every loaded library supplies against `libref`'s answers, benchmarks them,
and writes a report. It is meant for out-of-tree authors as much as for this repo.

```
rad-kbench [options]                     replay the recorded fixture (no model)
rad-kbench -m <model.rad> [options]      check against a real container
rad-kbench -m <model.rad> --record FILE  record a new fixture
```

### The three modes

**Replay (no `-m`)** — seconds, on any machine, with no model and no weights present. The
geometries, the seeds the inputs are drawn from, and `libref`'s answers reduced to a norm and a
sample are checked in as `data/kernels.rkb`. This is what `ctest` runs.

```sh
rad-kbench --home $RADIANCE_HOME --max-cases 1
rad-kbench --home $RADIANCE_HOME --max-cases 1 --fixture $RADIANCE_HOME/kernels_moe.rkb
```

There are **two** fixtures because a recording is of *one* container and no container declares the
whole op vocabulary: `kernels.rkb` is a dense model with a block drafter and gqa 6; `kernels_moe.rkb`
is a routed MoE with an MTP head and gqa 8.

Note that the replay also regression-tests `libref` itself — which a live oracle never can, since a
live comparison moves both sides together and reports agreement.

**Against a container (`-m`)** — weight operands are read *from it*: each weight's canonical
planes, which the kernel under test gets through its own `layout`/`relayout` hooks and the oracle
gets decoded. That is a test of the artifact and of the layout hooks, and it is the only way to get
one.

```sh
rad-kbench -m flashnext-w4.rad --home $RADIANCE_HOME --bench --report kbench.md
```

**Recording (`--record`)** — runs the oracle once and checks in what it produced. Weights are
**drawn from a seed, not read from the container**, because a reference nothing can reproduce is
not one. The recording carries each weight's encoding and the planes the operand takes, so a replay
draws the same planes, every library lays them out its own way, and the reference reads what they
decode to.

```sh
rad-kbench -m model.rad --max-cases 1 --record data/kernels.rkb
rad-kbench -m moe.rad   --max-cases 1 --record data/kernels_moe.rkb
```

### Flags

| | |
|---|---|
| `--fixture FILE` | the recording to replay (default `$RADIANCE_HOME/kernels.rkb`) |
| `--record FILE` | with `-m`: write a fixture |
| `-m, --model FILE` | the `.rad` to read weight operands from |
| `--home DIR`, `--kernels A:B` | plugin home and hierarchy |
| `--op NAME` | check only this op (repeatable prefix match) |
| `--ref NAME` | the oracle plugin's name (default `libref`) |
| `--seed N` | base seed; every case derives its own, so changing a filter does not change any other case's inputs |
| `--tol X` | override every tolerance (for bisecting, **not** for passing) |
| `--max-ctx N` | the deepest context considered: declare runs at it and the attention sweep builds to it (default: the checkpoint's training context for declare, 16384 for the sweep). Attention is swept over depth as well as query length |
| `--max-tok N` | the `max_tok` declare runs at (default 512, small on purpose) |
| `--kv-cache-dtype T` | `bf16` or `fp8`, the cache width declare runs at, as on the engine (default bf16) |
| `--max-cases N` | runs per distinct (op, kernel, geometry, tuned point); 0 means every one (default 4) |
| `--max-bytes MIB` | **skip a case whose operands would need more than this**, by name (default: half the memory free at start, at most 8192 and at least 512) |
| `--no-fusion` | skip the fusion oracle |
| `--bench` | time each case that passes, on the same bytes |
| `--bench-iters N`, `--bench-reps N` | launches per rep (20), timed reps (3) |
| `--bench-warm-ms N` | untimed launches before each timed window (default 500) |
| `--report FILE` | write the markdown report |
| `--analysis FILE` | splice in a fragment from `tools/kbench-analysis.sh` |
| `--num-speculative-tokens N` | draft depth to declare at |
| `-v` | print every case, not only failures |

### Things that will bite you

**`--max-bytes` matters on a loaded machine.** One case holds every operand as host floats, again
as the kernel's dtype, and again per library on the device. Without the cap a small machine is
killed by the OOM killer with no indication which op it died on. The default is taken from the
memory free when the run starts, which a server started later does not respect; if a server is
running on the same box, pass something small explicitly:

```sh
rad-kbench --home $RADIANCE_HOME --max-cases 1 --max-bytes 1024
```

**`--bench` is cache-warm.** It cannot rank a bandwidth-bound kernel the way production runs it.
Read the caveat the report prints about the GB/s column.

**`--max-cases`.** A 62-layer model presents the same kernel at the same shape sixty times and only
the weights differ. The default of 4 is not a sample of anything interesting; `--max-cases 1` is
what the fixtures are recorded at.

**Fusion claims are checked byte for byte.** A kernel that declares the chain it replaces is run
against that chain, out of its own plugin, and the outputs compared with `memcmp`. That claim is
what makes a fused kernel *substitutable* rather than a second numerical path.

**A kernel that does not read a weight's encoding is a skip, not a failure.** Its layout hook
answers `RAD_E_DTYPE` for an encoding it does not take, and the report names the kernel, the
operand and the encoding. A fixture in another format version does not load at all — record it
again.

---

## rad-tune

```
rad-tune -m <model.rad> [options]

  -m, --model FILE   the .rad whose declared graph supplies the shapes
      --home DIR     $RADIANCE_HOME; the cache is written to <home>/tune/<machine>.tune
      --kernels A:B  kernel plugin hierarchy, first wins
      --op NAME      tune only ops whose name starts with NAME
      --machine KEY  override the machine key (for writing a cache for another box)
      --reps N       timing reps, median reported (default 5)
      --iters N      launches per rep (default 32)
      --max-tok N    the max_tok declare is run at (default 8192)
      --tp N         tensor-parallel width to declare at (default 1)
      --num-speculative-tokens N
      --dry-run      list what would be measured and stop
  -v, --verbose      print every candidate, not just the winner
```

Kernels declare **tunable axes** rather than named variants. `rad-tune` walks that space at this
model's actual shapes on this actual machine, and writes the winner per (op, geometry) to
`$RADIANCE_HOME/tune/<machine>.tune`. The engine's selector reads that cache at startup.

```sh
rad-tune -m model.rad --home $RADIANCE_HOME --tp 2
rad-tune -m model.rad --op gemm --dry-run        # what would be measured
rad-tune -m model.rad --op attn_paged -v         # every candidate
```

**Declare at the shapes you serve.** Two things decide that and both are flags here:

- **`--tp N`** — a sharded deployment halves every sharded N. A declare at world size 1 measures a
  program no `--tp 2` serve ever runs.
- **`--num-speculative-tokens N`** — this is what puts the *drafter's* kernels in the measured graph
  at all. The default is the container's own depth. The drafter's shapes appear in no other model's
  tuning table, so nothing else covers them.

At `--tp > 1` every rank declares at once, on its own thread and its own device, exactly as the
engine does: a collective's `init` is a rendezvous, so declaring rank 0 alone at world size 2 does
not fail — it *hangs*.

**Prove the variants beat the default before trusting a cache.** `RADIANCE_TUNE` pins a kernel's
axes for a comparison; it is an instrument, not a deployment setting.

---

## rad-schemas

```
rad-schemas libref.so                 print one plugin's vocabulary
rad-schemas libr4d.so libref.so       diff them, in hierarchy order
```

The first plugin in hierarchy order to declare an op **fixes its schema**, and a later plugin
declaring the same op differently is refused at load with both plugins named. That rule is right:
arguments are positional, so two disagreeing schemas make the same call mean different things
depending on which plugin won selection — silent numerical garbage rather than a diagnosable
failure.

But "refused at load" is a poor place to find out, because the message names one op and you have to
guess the rest. This prints all of them at once, so a new plugin can be conformed to the vocabulary
before it is ever loaded beside another.

**It links no core at all**, deliberately: it `dlopen`s plugins and reads their static schema
tables, so it works on a plugin the engine cannot even load — which is exactly when you need it.

```sh
rad-schemas $RADIANCE_HOME/kernels/libr4d.so $RADIANCE_HOME/kernels/libref.so
```

---

## rad-tokcheck

```
rad-tokcheck [options] <tokenizer.json> <corpus-file>...

  --hf-dir DIR    compare against HuggingFace ids in DIR/<corpus basename>.ids
                  (little-endian int32, encode(text, add_special_tokens=False))
  --no-ref        skip the reference encoder, which is slow
  --user          encode with parse_special off: a control token's spelling is text
  --turns N       turns in the multi-turn measurement (default 40, 0 skips it)
  --repeat N      timing repeats, best taken (default 3)
  --context N     ids shown either side of a difference (default 6)
```

Three questions, per corpus file: does the encoder produce the ids the reference encoder in
`tools/tokref` produces (llama.cpp's splitters and `std::regex`, a string-keyed merge); does it
produce the ids HuggingFace `tokenizers` produces, read from files made elsewhere so that nothing
here needs Python; and how fast is it -- MB/s for both encoders, and what re-tokenising a growing
conversation costs with the segment cache and without it. Any difference is printed with the ids
and the text around it, and makes the exit status non-zero, so it can gate a change.

---

## The KL mode

```
radiance --model BF16.rad ... --kld-record DIR --kld-corpus CORPUS.jsonl    the yardstick, once
radiance --model QUANT.rad ... --kld-ref DIR [--kld-out REPORT.json]        each candidate
```

How far a quantised model is from the one it was quantised from, as the distribution of the
per-position **KL(P_bf16 || Q_quant)** over the whole vocabulary: mean, median, 99th, 99.9th and
99.99th percentiles and maximum, the top-1 agreement (the candidate's argmax is the reference's),
and both models' perplexity on the corpus's own next tokens -- overall and per source. A mode of
the engine rather than a tool, like `--debug-graph`: it starts exactly as a serve would, with the
same placement and kernels, then runs the corpus through prefill instead of listening, and exits.

**The reference is computed once.** `--kld-record` writes the bf16 model's log-probabilities at
every scored position to DIR -- f16 over the whole vocabulary, the 16 most likely tokens exactly,
and the tokens themselves -- and `--kld-ref` runs a candidate over those stored tokens, scoring its
own f32 logits against the stored row. A candidate costs one prefill of the corpus. DIR is never
written over, and is read only once its `kld.json` exists, which the recording writes last.

**The corpus** is JSONL, one `{"prompt", "score_from", "source"}` a line: `prompt` is tokenised
with control tokens recognised, as a rendered chat template is; positions `[score_from, n - 1)`
are scored; `source` is the report's grouping. The corpus the Flash-Next numbers in
[MOE-W4.md](MOE-W4.md) were measured on is disjoint from the calibration corpus, so a calibrated
quantiser is not scored on the text it was fitted to. Positions are what the tail needs: the
99.99th percentile of 83K positions rests on eight of them.

**What the mode changes for itself.** Speculation and the prefix cache are off (a drafter writes
no scored row; a prefix hit would skip positions the reference holds), the step pipeline is off
(the scored rows are copied off the cards after the step), and the logits buffer holds a row for
every token of a step (`RadBuildCtx::max_out_rows`) -- `--max-num-batched-tokens 4096` is 2 GiB
of it a rank, plus the same in pinned host memory. Everything else is the command line's: run
every candidate with the same flags, and the reference with an exact wire and a bf16 KV cache.

`--kld-out FILE` writes the report as JSON and every position's KL, both NLLs and the argmax
agreement to `FILE.rows` (`[positions, 4]` f32, reference row order), so two candidates can be
compared position by position.

---

## The shell harnesses

`scripts/` drives a **running** engine: bring a model up, measure it, and assert on what comes back
over the wire. They are versioned with the engine because a measurement is only reproducible if the
thing that took it is — and because what each one encodes is an argument (why a window is sampled
the way it is, which counter is the honest one) that is expensive to reconstruct from the number.
[scripts/README.md](../scripts/README.md) has a line on each.

The scripts that start a server or a conversion expect the build at
`$HOME/workspace/radiance/build` and the containers under `$HOME/models/rad/`; every default is an
environment override, listed in each script's header. The ones that only attach to a live server
need nothing but `curl` and `jq`. Scripts resolve each other by `$(dirname "$0")`.

**Which ones restart the server.** `fnserve.sh`, `mcserve.sh`, `fnstop.sh`, `fntrace.sh`,
`dprofd.sh`, `mctrace.sh`, `mcagent.sh` and `mcdecl.sh` all `pkill` a running engine, whatever port
it is on. `oracle.sh` starts its own server and stops whatever listens on its port (`P`, default
8302). Everything else — `cstep2.sh`, `ident.sh`, `tierident.sh`, `tiercross.sh`, `mctools.sh`,
`mcstream.sh`, `hfget.sh`, `convfn.sh`, `convdspark.sh`, `fnpf.sh`, `fnconc.sh`, `fndrop.sh`,
`apikeys.sh`, `diskstall.py` — leaves a live serve alone.
`fndrop.sh` is the one to read the description of first: it leaves the PROCESS alone and
deliberately disconnects clients from it, which is not the same promise.

### Bringing a model up

| | knobs |
|---|---|
| `mcserve.sh` | MiniCPM5-2B + DSpark drafter (`minicpm5-2b-dspark.v2.rad`), left running. `P` `TP` `MLEN` `SEQS` `M` `KVD` `HR` `MNBT` `EXTRA` `EXTRA_ENV` |
| `fnserve.sh` | Qwen3.8-Flash-Next, the production container `w4nl64-i8lm.rad`, same contract. `P` `TP` `MLEN` `SEQS` `M` `RATIO` `NS` `HOSTPOOL` `KVD` `HR` `MNBT` `WIRE` `KVHOST` `KVDIR` `KVDISK` `EXTRA` `EXTRA_ENV` |
| `fnstop.sh` | stops whichever is running and **waits for the process to go** |
| `convdspark.sh`, `convfn.sh` | rebuild the containers those two serve. The conversions of record. `convfn.sh` needs `CALIB=<dir>`: the production recipe's expert rule runs GPTQ against the calibration Grams there |
| `hfget.sh` | fetch a checkpoint from Hugging Face with `curl` and `jq`. Resumable |

```sh
scripts/fnserve.sh                             # defaults: port 8100, tp 2, 200K ctx, ratio 0.82
P=8100 MLEN=131072 SEQS=4 scripts/fnserve.sh   # override
scripts/fnstop.sh
```

Both print the endpoints and exit with the server still up. Read their headers — `fnserve.sh` in
particular carries the whole `--expert-vs-cache-ratio` argument with the measurements behind it.

### Measuring a server that is already up

These attach to a running server and never restart one, so they can be pointed at a live
deployment. `P` selects the port.

| | what it answers | knobs |
|---|---|---|
| `fnpf.sh` | prefill throughput, with a fresh nonce per rep so the prefix cache cannot answer for work that was never done | `P` `N` `REPS` `BPT` |
| `fnconc.sh` | what a long prefill costs the requests already running beside it — both halves on one clock, alone then contended, with the warm pass discarded | `P` `C` `DEC` `PROMPT` |
| `fndrop.sh` | whether a client going away mid-request takes anything with it. Drop points are derived from one timed request, and every answer is checked against a reference | `P` `LONG` `GEN` |
| `apikeys.sh` | whether the server does what oai.cpp says about request keys: every rejected one refused by name, every inert one accepted, every accepted one honoured | `P` |
| `cstep2.sh` | steady-state decode ms/step at concurrency | `P` `LEVELS` `MT` `WIN` |
| `ident.sh` | run-to-run reproducibility at a fixed configuration: one hash per (temperature, question) | `P` |
| `diskstall.py` | what restoring a conversation from the disk tier costs: the returning request's time to first token, and the longest stall it puts on a stream decoding beside it | `--port` `--session` `--fill` |

`fndrop.sh` exists because a disconnect reaps a request at an arbitrary point in the step loop, and
every other harness here waits for its requests, so none of them can see what holds a reference
across that point. Surviving is not the whole test: a server that stays up and then answers
differently has a freed slot feeding a live sequence. Note its own caveat — one diverged hash is not
a finding, it is an instruction to run it again.

`ident.sh` does not expect a speculative run to match a non-speculative one byte for byte: the
split-K count is chosen from M, so the two take different reduction orders and a long answer can
flip a near-tie. A divergence within the first sentence or two is a defect.

```sh
P=8100 LEVELS="1 2 4 8" scripts/cstep2.sh
```

```
C        steps       tok   tok/step    ms/step   accept%
1          812       900      1.108     29.820      74.2
2         ...
```

`cstep2.sh` waits until exactly C sequences are running before sampling and re-checks at the end: a
window whose edges carry fewer than C describes no concurrency at all.

### Measuring with a server of its own

| | |
|---|---|
| `dprofd.sh` | per-op device timing at depth and concurrency, on the 27B dense model with its DFlash2 drafter. **Difference two dumps**, both taken after prefill ended |
| `fntrace.sh` | a Flash-Next prefill under `rocprofv3`, with the link counters from `/stats` across the same request |
| `mctrace.sh` | a MiniCPM5-2B decode trace under `rocprofv3`, formatted for `libr4d/isa/tsum.sh` |
| `oracle.sh` | every op compared against libref on the operands the model actually issued; reads for a MISMATCH and for the NOT CHECKED lines that name what cannot be compared. `OPS` `MAXN` `FROM`/`TO` `TOL` `M` |

### Regression checks against a live server

These cover what `tests/chat_test.cpp` structurally cannot: the server, the wire format, and a real
model's output.

| | |
|---|---|
| `tierident.sh` | does a session that left VRAM come back byte for byte, both halves? `P` `H` `N` `TO_HOST` `TO_DISK` `FILL` |
| `tiercross.sh` | the same with **two** sessions in flight — the only way to see one being served the other's context |
| `mctools.sh` | the tool-call shapes hardest to parse — an array argument, two calls in one reply, prose before a call, plain chat with tools available. Asserts `finish_reason` and that no `<function`/`<param`/CDATA marker reaches content |
| `mcstream.sh` | tool-call replies **streamed**, asserting on raw SSE fragments. Content cannot be retracted once sent, so a repair applied at the end is not a fix — this is the check that sees the difference |
| `mcagent.sh` | what speculation is worth on an agentic workload (chat + tools, a constrained decode) |
| `mcdecl.sh` | how often the drafter actually proposes. `draft_acceptance_ratio` is conditional on having drafted, so it reads healthy while tokens-per-step collapses; blocks-per-step separates the two |

> **They are not all read-only.** `tierident.sh`, `tiercross.sh`, `mctools.sh` and `mcstream.sh`
> talk to a server already running (port 8100 unless `P` says otherwise) and leave it alone.
> `mcagent.sh` and `mcdecl.sh` **start and stop servers of their own** — and `mcserve.sh`, which
> they call, begins with `pkill -f "bin/radianc[e]"`, so they will take down any radiance you have
> running. Check before you run one on a box that is serving.

```sh
scripts/mctools.sh && scripts/mcstream.sh
```

### One more, in `tools/`

`tools/kbench-analysis.sh` produces the static-and-dynamic-analysis fragment that
`rad-kbench --analysis` splices into its report. `tools/ops_doc_check.sh` is the `ops_doc` test:
it holds `docs/OPS.md` to what the registries actually declare.
