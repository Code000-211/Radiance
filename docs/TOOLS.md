# radiance tools reference

Every flag of every tool, with examples. Start with [GUIDE.md](GUIDE.md).

Six binaries and the `rad-export` script are installed beside `radiance`; the shell harnesses
live in `scripts/`:

| | |
|---|---|
| [`rad-info`](#rad-info) | print what a `.rad` container holds. Loads no plugins |
| [`rad-convert`](#rad-convert) | safetensors or GGUF in, `.rad` out, quantised by a recipe |
| [`rad-export`](#rad-export) | a finetune (merged checkpoint or LoRA adapter) to a `.rad`, with the base model's own recipe |
| [`rad-kbench`](#rad-kbench) | correctness, completeness and speed of every kernel library |
| [`rad-tune`](#rad-tune) | benchmark each kernel's tunable space at this model's shapes, on this machine |
| [`rad-schemas`](#rad-schemas) | print and diff the op schemas plugins declare |
| [`rad-tokcheck`](#rad-tokcheck) | compare the tokeniser with its reference encoder and with HuggingFace's ids over real text, and time it |
| [`radiance --kld-*`](#the-kl-mode) | KL divergence of a quantised model against the bf16 one, at every position of a corpus |
| [`radiance --diagnose`](#the-self-measurement-mode) | the engine measured against a fixed serving workload instead of serving: step time by shape, the interval between the steps that emitted tokens, acceptance by draft position, and every generated token |
| [`scripts/`](#the-shell-harnesses) | bring a model up, measure it, assert on what comes back |

[`rad-avx-check`](#rad-avx-check), which checks a host kernel library against libref without a GPU,
is built beside libavx rather than installed.

`rad-convert`, `rad-kbench` and `rad-tune` run the engine's startup with a different last step:
the same plugins, in the same hierarchy, through the same declare phase, so they resolve kernels
as the engine does.

Every tool that loads plugins honours `RADIANCE_HOME` and `RADIANCE_KERNELS`, and reads them, and
`--kernels`, as the engine does. A numeric option takes a whole number, as the engine's do: `12k`,
`auto` (where the option does not name it) or an empty value is refused with exit status 2.

---

## rad-info

```
rad-info [options] <model.rad>
rad-info --plugins [--home DIRS]

  -v, --verbose      list every tensor, not just the per-unit summary
      --meta         print the metadata table and stop
      --vocab        print the vocab summary and its normaliser chain, and stop
      --profile N    print the top N entries of the expert profile (default 20, 0 = all)
      --recipe       print the recipe the container was converted with, and stop
      --plugins      print every plugin the $RADIANCE_HOME search path offers, as the
                     engine loads it, and stop
      --home DIRS    the search path for --plugins (default $RADIANCE_HOME)
```

It opens the container read-only and, except for `--plugins`, loads no plugins: a container names
no kernel library. Output is aligned, byte counts are in binary units, and each count says what it
counts.

```sh
rad-info flashnext-w4.rad
```

```
flashnext-w4.rad
  format        RAD2, ... GiB, little-endian
  architecture  qwen4exp
  model         Qwen3.8-Flash-Next
  source quant  (none)
  recipe        14 line(s) -- --recipe prints it
  created by    rad-convert ...

  section          offset        bytes    count
  strings          ...
  metadata         ...
  directory        ...
  planes           ...
  encodings        ...
  vocab            ...
  profile          ...
  data             ...
```

The metadata table, the weight directory (grouped by layer and movement unit), the vocab and the
expert profile follow. With `-v`, every weight is listed with its encoding
(`fp8_e4m3*bf16[128x128]`, `i4*bf16[1x128]/fwht128`, `bf16`), its logical shape and what made it:
the checkpoint as it stood, or the quantiser and options of the recipe rule that matched. Totals
are given by encoding and by what made the weight.

```sh
rad-info --meta m.rad                         # the metadata table alone
rad-info --vocab m.rad                        # vocab + normaliser chain
rad-info --recipe m.rad                       # the recipe, expanded, one rule a line
rad-info -v m.rad | head -40                  # every tensor, its encoding, shape and planes
rad-info --profile 0 m.rad                    # the whole expert-popularity profile
rad-info --plugins                            # every plugin on the search path
```

---

## rad-convert

```
rad-convert [options] <input> -o <model.rad>

  <input>              a .gguf, a .safetensors, an index.json, or a checkpoint directory
  -o, --out FILE       the container to write
      --recipe FILE    the quantisation recipe (below)
      --quant RULE     'PATTERN=QUANTISER:key=value,...', ahead of the recipe's rules. Repeatable
      --list-quantizers  every quantiser loaded and the options it takes
      --arch ID        override the architecture id from the checkpoint
      --home DIR       $RADIANCE_HOME
      --kernels A:B    the kernel libraries to load, first wins. Default $RADIANCE_KERNELS,
                       else every library found
      --tokenizer F    tokenizer.json; its declared chain is baked instead of being
                       reconstructed from a pre-tokenizer name
      --tokenizer2 F   a second tokenizer.json, baked as the container's second vocab: a
                       generator that reads its prompt through two tokenizers
      --add FILE       one more .safetensors file merged in, its tensor names kept (a release
                       shipped as separate single files). Repeatable
      --config DIR     config.json, generation_config.json and tokenizer.json from DIR, for a
                       single-file checkpoint that ships none
      --imatrix F      llama.cpp imatrix: expert profile, column importances, error weighting
      --set K=V        a metadata key, seen by declare and written into the container. Repeatable
      --max-tok N      the max_tok declare is run at (default 8192)
      --max-ctx N      the context bound declare is run at. Default: the checkpoint's
                       training context
      --draft-model DIR  a drafter that ships as its own repository, merged under `draft.`
      --kv-cache-dtype T  bf16 | fp8, the cache width declare is run at. Default: the first
                       at which every op resolves; the container is the same at either
      --max-spec N     the max_spec declare is run at (default 4, or the block a drafter
                       states; a number given here is used as given). 0 leaves the draft
                       head out
      --reuse FILE     copy every weight an earlier container holds identically
      --changed FILE   with --reuse: checkpoint tensors (one name a line) that differ from
                       the --reuse container's source; every weight that reads one is converted
      --in-place       with --reuse: extend that container instead of writing a new one
      --plan-only      declare, plan every weight's encoding, print the plan, and stop
      --report N       the worst N quantisation errors to print (default 20)
  -v, --verbose
```

### What it does

A container holds each weight as the canonical planes of its encoding (codes, scales, zero points,
a codebook), row-major, and names no kernel library. Kernels rearrange the planes into their own
layouts at load, so a kernel library can change its layouts without a reconvert. rad-convert
decides the numbers:

1. Reads the checkpoint's `config.json` (or GGUF metadata).
2. Loads the architecture, kernel and quantiser plugins and runs declare, as the engine does. The
   architecture plugin's name map says which checkpoint tensors make each declared weight, and
   `rad_weight_encoding` answers each one with the encoding the recipe gives it.
3. Resolves every op and asks each resolved kernel's `layout` hook about the planned encoding. A
   recipe no kernel can serve is refused here, before a byte is written.
4. Writes each logical weight once: the checkpoint's own bytes where no rule matches, otherwise the
   planes the matching rule's quantiser writes, a block of rows at a time.
5. Measures every quantised weight against its source through the core's decoder and prints the
   worst, imatrix-weighted when `--imatrix` is given.
6. Bakes in the tokeniser, the chat template, the metadata, the recipe and any draft head.

### Recipes

A recipe is an ordered list of rules from a logical weight name to a quantiser and its options.
The first rule whose pattern matches decides. A weight no rule matches keeps the checkpoint's
encoding, which must be one the engine serves as it is: bf16, f16, f32, or block-scaled FP8
(`X.weight` with `X.weight_scale_inv`). Anything else is named and refused.

```
# PATTERN                      QUANTISER  OPTIONS
blk.*.ffn_*_exps.*.weight      gptq       codes=i4 group=128 scale=bf16 transform=fwht128 calib=$CALIB
blk.*.ple_ngram.weight         rtn        codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=1.99317932128906e-4
output.weight                  rtn        codes=fp8_e4m3 block=128x128 scale=bf16
blk.*.weight                   rtn        codes=fp8_e4m3 block=128x128 scale=bf16
```

A pattern is a glob over the whole name: `*` crosses dots, `?` is one character. `$NAME` (or
`${NAME}`) in an option is taken from the environment; an unset variable is refused. `--quant`
adds rules ahead of the file's:
`--quant 'output.weight=rtn:codes=fp8_e4m3,block=128x128,scale=bf16'`. A quantiser that declines a
weight leaves it in the checkpoint's encoding; `rtn` declines a rank-1 tensor, so a pattern that
also matches a norm leaves the norm as it is.

`rad-convert --list-quantizers` prints every quantiser and its options. The built-in set is
`libquant`:

| quantiser | |
|---|---|
| `rtn` | round to nearest on a grid; the scale of each block chosen by `rule` |
| `gptq` | error feedback against the layer's input Hessian (`calib=DIR`, damped by `damp`, default 0.01 of its mean diagonal), scales fixed from the weight first; `act_order=1` quantises the columns by descending input energy and stores the permutation as the encoding's `perm` transform; `cd=N` adds N sweeps of coordinate descent after the loop, every code re-chosen at its conditional optimum against the same Hessian |
| `awq` | each input column scaled by its activation magnitude to a searched power, then a per-block clip search, both against the Hessian; the column scales are the encoding's last scale level |
| `autoround` | sign-gradient descent on a rounding offset per value and a range factor per block, against the Hessian; `iters` steps of N·K² multiply-adds each |
| `paroquant` | a channel scaling and `stages` rounds of Givens pair rotations, learned against the Hessian, then AutoRound on the rotated weight; the encoding's `givens` transform |
| `ggml` | `type=` one of llama.cpp's formats (q4_0, q4_1, q5_0, q5_1, q8_0, q2_K..q6_K, iq4_nl, iq4_xs, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m), chosen by llama.cpp's own search, imatrix-weighted; the codes are byte for byte llama.cpp's, stored as each type's affine planes. iq2_xxs, iq2_xs and iq1_s need `--imatrix` |
| `cast` | a float weight in another float dtype (`dtype=` f32, bf16, f16, fp8_e4m3, fp8_e5m2), no scale |
| `import` | rad-convert's own, not a plugin: the numbers a quantised release ships, in the spelling the kernels read, instead of quantising again. `codes=uN` makes a symmetric grid's signed codes unsigned under a zero point of 2^(N-1); `zero=` is the zero plane's dtype (u4, u8); `scale=` the scale's (f16, bf16, f32), each shipped scale rounded to nearest; `block=RxC` repeats an FP8 scale shipped per row or per tensor over that block grid. It reads an imported integer grid (GPTQ, CT-INT4) or an FP8 weight with a scale per row or per tensor, refuses any other quantised form by name, and keeps a plain float weight as it is. The run reports per rule how many weights it re-expressed and how many scales the new dtype rounded |

Every quantiser except `ggml`, `cast` and `import` takes the grid options:

| option | |
|---|---|
| `codes` | `i2`..`i8`, `i16`, `i32` symmetric; `u1`..`u8`, `u16` with `zero=`; `fp8_e4m3`, `fp8_e5m2`, `fp6_e2m3`, `fp6_e3m2`, `fp4_e2m1` |
| `table` | a codebook the codes index instead: `nf4`, `iq4nl`, `w4nl`, `w5nl`, `binary` (one bit), `ternary`, or the values, `;`-separated |
| `group=N` / `block=RxC` | where the scales sit; `*` is a whole extent |
| `scale` | f32, bf16, f16, fp8_e4m3, e8m0; under `scale2` also an integer sub-scale, u4..u8, i6, i8 |
| `scale2`, `group2` / `block2` | a second scale level the first is relative to (default the whole tensor) |
| `scale2_value` | the second level fixed at this value, exact in `scale2`'s dtype, for every block of it |
| `zero` | an integer zero point per block, making the grid asymmetric |
| `transform=fwhtN` | rotate each row by the Hadamard first |
| `rule` | `absmax`, `mx` (the OCP shared exponent), `fixed` (`scale_value=`), `search` (least squared error over candidates and their refits, imatrix-weighted) |
| `clamp` | a signed code's range, `full` or `sym` |

Examples: MXFP4 is `codes=fp4_e2m1 group=32 scale=e8m0`; NVFP4 is `codes=fp4_e2m1 group=16
scale=fp8_e4m3 scale2=f32`; NF4 is `table=nf4 group=64`; a K-quant-like grid is `codes=i4
group=32 scale=u6 scale2=f16 group2=256`; a ternary BitNet weight is `table=ternary`.

A container can hold only what some kernel reads. The converter checks every planned encoding
against the layout hooks of the kernels declare resolved and refuses one that no kernel accepts,
so a grid libquant can write is usable only once a kernel library takes it. libr4d reads the plain
floats; block-scaled fp8 and fp8 rows; int8 block rows (the W8A8 trunk and lm_head), plain or
rotated (`i8*bf16[1x128]/fwht128`); the routed experts as fp8, int4 (`i4*bf16[1x128]/fwht128`) or
the `w4nl`/`w5nl` codebooks under E4M3 scales ([MOE-W4.md](MOE-W4.md)); the dense w4, w8 and mxfp4
GEMM rows; and the 2-bit draft head. Every encoding also decodes through the core, which the
converter's error report, the oracle and the quantisers' tests use.

The recipes for the containers this tree serves are in `data/recipes/`. `scripts/convfn.sh` and
`scripts/convdspark.sh` are the reference conversions.

### Examples

```sh
# Flash-Next as served (w4nl64-i8lm.rad): codebook experts by GPTQ against the calibration
# Grams in $CALIB, an int8 trunk (docs/MOE-W4.md, "What is served"). The recipe's expert rule
# reads calib=$CALIB, so CALIB must be set
CALIB=<calibration dir> rad-convert <Qwen3.8-Flash-Next checkpoint> \
    -o w4nl64-i8lm.rad --recipe data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home $RADIANCE_HOME -v

# A --quant rule goes ahead of the recipe's: the experts as plain int4 by GPTQ, under a recipe
# whose own expert rule rounds to nearest
CALIB=<calibration dir> rad-convert <Qwen3.8-Flash-Next checkpoint> -o flashnext-w4.rad \
    --quant 'blk.*.ffn_*_exps.*.weight=gptq:codes=i4,group=128,scale=bf16,transform=fwht128,calib=$CALIB' \
    --recipe data/recipes/qwen4exp-w4-hc8m.recipe --home $RADIANCE_HOME -v

# A block-fp8 checkpoint needs no recipe: every weight is kept as it is. It needs no container
# either (`radiance --model` serves the checkpoint directly) unless a tier only a container
# has is wanted (GUIDE §3.4)
rad-convert <Qwen3.8-27B-FP8 checkpoint> -o q38-27b-fp8.rad

# A bf16 checkpoint of the same model, as that release's block fp8 (the recipe's header has the
# --set keys that select the fp8 plugin)
rad-convert <Qwen3.8-27B or a finetune> -o q38-27b-fp8.rad --recipe data/recipes/q38-27b-fp8block-df2.recipe \
    --draft-model <DFlash2 checkpoint> --set quantization_config.quant_method=fp8 \
    --set quantization_config.fmt=e4m3 --set "quantization_config.weight_block_size=128 128" \
    --set quantization_config.activation_scheme=dynamic

# Check the plan without writing anything
rad-convert $SRC -o /dev/null --recipe r.recipe --plan-only -v

# Merge a drafter that ships as its own repository
rad-convert <MiniCPM5-2B checkpoint> -o minicpm5-2b-dspark.rad \
    --draft-model <MiniCPM5-2B-DSpark checkpoint> --recipe data/recipes/minicpm5-2b-dspark.recipe

# Re-quantise only the experts of an existing container
rad-convert $SRC -o new.rad --recipe new.recipe --reuse old.rad
```

### Flag notes

- `--tokenizer`: pass it for a GGUF. Without it the tokeniser chain is reconstructed from a
  pre-tokeniser name, which is a guess. A safetensors checkpoint's `tokenizer.json` is found beside
  it. A checkpoint that keeps its chat template in `chat_template.jinja` needs that file beside
  `tokenizer_config.json`; without a template, `/v1/chat/completions` answers 501.
- `--set K=V` states a convert-time choice that must also hold at serve time: the key is handed to
  declare and written into the container. A weight's storage format is not such a choice (the
  encoding is in the container and the plugin reads it there), but the serving plugin is: a
  `quantization_config.*` key re-derives the descriptor the plugin is chosen by. A bf16 checkpoint
  whose recipe writes Qwen's block fp8 is therefore declared as that format and converted for the
  fp8 plugin (`data/recipes/q38-27b-fp8block-df2.recipe` has the keys).
- `--max-ctx`: some graphs depend on the context bound. Qwen4-Exp sizes its QSA buffers by it and
  refuses a `--max-tok` and context pair whose score plane would pass 2 GiB.
- `--max-spec`: a container carries what the model can serve, so a draft head is written even when
  this run would not draft. `0` leaves it out.
- `--add`, `--config` and `--tokenizer2` read a release that ships as ComfyUI's single files
  (Anima: a diffusion model, a text encoder and a VAE in `split_files/`): the files merge into one
  namespace with their own tensor names (a name in two files is refused), the text encoder's
  repository supplies the config and tokenizer the plugin is sized by, and a second tokenizer.json
  (with its `tokenizer_config.json` beside it) becomes the container's second vocab section, which
  a generator declaring `aux_tokens` reads its prompt through. GUIDE §3.2 has the Anima command.
- `--draft-model` merges a drafter here; the engine refuses the flag. The drafter's tensors go in
  under a `draft.` prefix and its config under `draft.` keys. It has no embedding and no `lm_head`
  of its own; it uses the target's.
- `--imatrix` supplies the expert-popularity profile the placement planner ranks units by, hands
  each quantiser its weight's column importances, and weights the error report.
- `calib=DIR` (`gptq`, `awq`, `autoround`, `paroquant`) is a directory an engine run wrote with
  `RADIANCE_CALIB_DIR`: each layer's Gram matrices of the input, in the domain the kernel multiplies
  it in. A container that stores a weight rotated records the rotated input, and a recipe that
  rotates the same weight reads it as it is. A weight with no file there is quantised uncalibrated,
  and the converter says so; `awq`, `autoround` and `paroquant` use the imatrix's diagonal instead
  when `--imatrix` gives one. Under tensor parallelism a column-split linear (an attention output or
  down projection) sees only its rank's columns, so a tap may write each rank's Gram as
  `gram.r<rank>.<stem>.part.bin`: with no whole file, libquant places the parts on the diagonal in
  rank order (the cross-rank products, never formed, left zero). They must cover K and be in the
  model's own domain. The three-step calibrated conversion is in [MOE-W4.md](MOE-W4.md).
- `autoround` and `paroquant` are slow. AutoRound's 200 default steps each cost N·K² multiply-adds;
  on a 16-core Zen 5 at about 2.5 TFLOP/s that is minutes for a 17408-wide down projection and
  most of a working day for a 27B dense model. `iters=50` takes a quarter of the time.
- `--reuse` / `--in-place` match a weight on its encoding, quantiser, options and shape, not on its
  source bytes, and carry whatever calibration made the old planes. Use `--reuse` only when the
  source and the calibration are the same, or name the changed source tensors with `--changed`.
  `--in-place` writes new weights past the container's end and the header last; it refuses to
  change a weight the container holds or to add an expert, and keeps the old header and length in
  `<file>.pre-append`.

---

## rad-export

```
rad-export SRC [-o OUT.rad] [--recipe NAME|FILE|auto] [--base DIR] [--draft DIR|ORG/NAME]
           [--set K=V ...] [--image IMAGE | --rad-convert PATH] [--memory 24g] [--dry-run]
           [-- rad-convert options]
rad-export --list-recipes [--json]

  SRC                a merged 16-bit checkpoint (config.json + *.safetensors) or a PEFT LoRA
                     adapter (adapter_config.json + adapter_model.safetensors)
  -o, --out FILE     the container (default: SRC's folder name + .rad, beside SRC or in out_dir)
      --recipe R     a recipe name, a recipe file, or auto (default): the one recipe whose header
                     names the base model, takes a bf16 checkpoint and needs no calibration data
      --base DIR     the base checkpoint a LoRA adapter merges into. Default: the adapter's
                     base_model_name_or_path, from disk or the Hugging Face cache, else downloaded
      --draft D      the drafter, when the recipe merges one. Default: the config, else the Hub
                     repository the recipe's usage line names (`<org/name>`); a usage line with
                     a placeholder instead asks for --draft
      --image IMAGE  run rad-convert in this Radiance image (docker, CPU only, no network, not
                     pulled). Default $RADIANCE_IMAGE, then the config's image, then
                     stilldeadcode/radiance
      --rad-convert P  run this rad-convert instead of docker; recipes from its share/radiance
      --memory M     docker memory limit for the convert
      --dry-run      resolve recipe, drafter and paths, print the rad-convert line, write nothing
      --keep-merged  keep the merged checkpoint of an adapter beside OUT
```

A trainer -- Unsloth, PEFT/TRL, Axolotl, a plain torch loop -- leaves a merged 16-bit checkpoint or
a LoRA adapter. The container a finetune ships in should be encoded exactly like the stock model's,
so rad-export converts it with the **stock model's recipe**, its `--set` keys and its drafter; the
tuned `.rad` loads with the same serve line as the stock one.

1. **An adapter is merged first**, `W + scale · B·A` per weight as PEFT's `merge_and_unload`
   computes it on the CPU (`scale` = alpha/r, alpha/√r with rsLoRA; `rank_pattern` and
   `alpha_pattern` matched as PEFT matches them; the delta in fp32, rounded to the adapter's dtype
   when that is bf16 or fp16), so the merged weights equal PEFT's bit for bit. It streams tensor by
   tensor into 1 GiB shards beside OUT, so a 27B merge needs a few GiB of RAM, not the model twice,
   and the bf16 model's size free on OUT's disk until the convert ends. Plain linear LoRA only:
   embedding LoRA, trained biases, DoRA, saved modules, a convolution's LoRA, a rank other than the
   adapter config's, and a target the base lacks are refused before anything is written -- merge
   those with PEFT's `merge_and_unload` and pass the merged folder. The adapter's tokenizer and
   chat template, where it ships them, replace the base's; one with more tokens than the base's
   embedding has rows is refused.
2. **The recipe** comes from the image (`/opt/radiance/share/radiance/recipes`, copied out once
   per image) or from the installed `rad-convert`'s `share/radiance/recipes`. `auto` matches the
   base model's name against each recipe's header; when several fit (a bf16 and an fp8 recipe of
   the same model), it names them and asks. The recipe's own usage line supplies the `--set` keys
   and says whether a drafter is merged.
3. **rad-convert runs on the CPU.** It opens no GPU, so it is safe beside a running serve. Under
   docker it runs as the caller, with no network, and sees only the checkpoint, the drafter, the
   recipe and OUT's folder. Its output goes to `OUT.log`.

```bash
rad-export ~/outputs/my-run/checkpoint-400 --dry-run            # what it would run
rad-export ~/outputs/my-run/checkpoint-400                      # adapter -> my-run-checkpoint-400.rad
rad-export ~/exports/merged -o ~/models/mine.rad --recipe q38-27b-fp8block-df2 --draft ~/Qwen3.8-27B-DFlash2
```

`~/.config/rad-export/config.json` (or `--config`) pins what auto cannot know -- every key optional:

```json
{"image": "stilldeadcode/radiance", "out_dir": "~/models/rad",
 "recipes": {"Qwen3.8-27B": "q38-27b-fp8block-df2"},
 "drafts":  {"q38-27b-fp8block-df2": "~/models/Qwen3.8-27B-DFlash2"},
 "bases":   {"Qwen/Qwen3.8-27B": "~/models/Qwen3.8-27B"}}
```

A recipe that needs calibration data (`calib=`, `--imatrix`) is listed but never picked by auto:
a finetune's Hessians are not the stock model's, so those recipes need their calibration rerun on
the tuned checkpoint first. rad-export itself is Python 3 with no dependencies; merging an adapter
needs `torch` and `safetensors`, which every trainer's environment already has, and fetching a base
or drafter that is not in the Hub cache needs `huggingface_hub` (only the files a conversion reads:
`*.json`, `*.safetensors` and the tokenizer's). The image carries no Python; take the script out of
it with
`docker run --rm --entrypoint busybox stilldeadcode/radiance cat /opt/radiance/bin/rad-export > ~/.local/bin/rad-export`.

When it runs as a post-export command (`UNSLOTH_POST_EXPORT` set in its environment), its last line
is `UNSLOTH_HOOK_OUTPUT=<path of the .rad>`, which a trainer UI can list with the export.

---

## rad-kbench

Checks every kernel every loaded library supplies against the reference library's answers,
benchmarks them, and writes a report. It is meant for out-of-tree library authors as well as this
repository.

```
rad-kbench [options]                       replay the recorded fixture (no model)
rad-kbench -m <model.rad> [options]        check against a real container
rad-kbench -m <model.rad> --record FILE    record a new fixture
rad-kbench -m <model.rad> --skeleton FILE  write a metadata-only copy to record from
```

### Modes

Replay (no `-m`) needs no model, no weights and no card. The geometries, the seeds the inputs are
drawn from, and `libref`'s answers reduced to a norm, a sample and a projection sketch are checked
in as `data/kernels.rkb` and `data/kernels_moe.rkb`. `ctest` runs three replays: the other
libraries against each recording, and `libref` against both of its own at tolerance zero.

```sh
rad-kbench --home $RADIANCE_HOME --max-cases 1 --no-ref --strict           # kbench
rad-kbench --home $RADIANCE_HOME --max-cases 1 --no-ref --strict \
           --fixture $RADIANCE_HOME/kernels_moe.rkb                         # kbench_moe
rad-kbench --home $RADIANCE_HOME --kernels libref --max-cases 1 --strict \
           --fixture $RADIANCE_HOME/kernels.rkb \
           --fixture $RADIANCE_HOME/kernels_moe.rkb                         # kbench_ref
```

(The ctests also pass `--coverage data/kbench_rows`, below.) There are two fixtures because a
recording covers one container and no container declares the whole op vocabulary: `kernels.rkb`
is a dense model with a block drafter and gqa 6; `kernels_moe.rkb` is a routed MoE with an MTP
head and gqa 8. `--fixture` is repeatable, and the report then covers the union.

Each library is asked for its row as the engine would ask it with that library alone: the same
constraints, derived parameters and weight-encoding filter, so a row the engine would pass over
for a weight's encoding does not run here either (`--all-rows` runs every row whose constraints
hold).

The replay of `libref` is its regression test, which a live comparison cannot be: there both
sides move together and agree. `libref` is deterministic and machine-independent by construction,
so it is judged exactly. Its reduced answers are cached under `$RADIANCE_HOME/cache/kbench`, keyed
by the build-ids of `libref` and every object it links, the CPU, a hash of every input byte and
the case itself: an unchanged `libref` replays in seconds, a rebuilt one in full, and the
comparison with the recording runs either way. `--no-cache` runs it regardless.

Against a container (`-m`), weight operands are read from it: each weight's canonical planes,
which the kernel under test gets through its own `layout`/`relayout` hooks and the oracle gets
decoded, and every expert of a weight table. This is the only test of the artifact and of the
layout hooks.

```sh
rad-kbench -m flashnext-w4.rad --home $RADIANCE_HOME --bench --report kbench.md
```

Recording (`--record`) takes the reference's answer once at every point of every declared op the
reference serves, whichever library resolved it. Weights are drawn from a seed, not read from the
container, so any machine can reproduce the reference and no card is needed: a host with only
`libref` records the same fixture. The recording carries each weight's encoding (a weight table's
from its first expert) and the planes the operand takes, so a replay draws the same planes, each
library lays them out its own way, and the reference reads what they decode to. Only the
container's tables are read, so a skeleton of it (`--skeleton`: the tables, the data region a
hole) records the same fixture in a few megabytes.

```sh
rad-kbench -m model.rad --skeleton model.skel.rad
rad-kbench -m model.skel.rad --max-cases 1 --record data/kernels.rkb
rad-kbench -m moe.rad        --max-cases 1 --record data/kernels_moe.rkb
```

Inputs are drawn by the law the reference's description states for each operand (a sigmoid gate,
a monotone log-decay, a routing histogram, a sampler's parameter rows as requests over a token
history that repeats itself), and the recording keeps the law, so a replay draws what was
recorded. The sampler's two laws need format version 6: a sampler case in an older recording
draws its rows as numbers, on which the mask, DRY and XTC stages do nothing and the penalties act
on an eighth of the rows.

`--rerecord FILE` writes the replayed fixtures' cases again with the reference's current answers
in the current format, each case keeping the laws it was recorded with. `--record-rows FILE`
moves the fixtures' cases into the constraints of every row of the libraries under test (another
band, tuned point or dtype) and records the reference there, described afresh: a fixture that
reaches rows no container selects, to replay with `--all-rows`. A row whose geometry would change
a weight's shape needs a container, and is listed.

### Verdicts

- An output is compared on its sampled positions, its full-tensor norm (a correct first tile
  followed by zeros) and, from format version 5, 32 seeded random-sign projections whose
  difference estimates the full rel_l2 to about 25% (a permuted or sign-flipped tile no sample
  lands in). The largest of the three is the error.
- Tolerances are per op and activation dtype (`dtype`, else `q_dtype`, else `kv_dtype`). Copies,
  gathers, scatters and cache stores are exact. Every op in the fixtures has a row of its own
  (`tests/kbench_test.cpp` checks).
- Ids (token, expert and top-k indices) are compared exactly. Where an op's contract leaves a tie
  open (`sample_argmax`, `router_topk`, `sample_topk`), an id that indexes a value equal to the
  reference id's is accepted.
- A case whose reference and output are both identically zero checked nothing: it is counted as
  vacuous, not passed.
- A failure, a red-zone overrun, a fusion that differs from its chain, and a host kernel that
  wrote into an input it was handed all exit 1. Under `--strict` so does a stale fixture: an
  operand count or schema that moved, or a case the reference no longer serves or now describes
  differently. A case no library under test serves is named, never a failure.
- `--coverage DIR` keeps one file per fixture and library of the rows a replay reaches
  (`data/kbench_rows/`). A row listed and not reached fails the run, unless a memory skip kept
  every case that would reach it from running. `--coverage-write` writes the files from a run.
- Every pass ends with a line saying where its time went: draw, hashing, layout, upload, launch
  per library, read-back, comparison, cache, timing.

### Timing

`--bench` times every case that passes, on the bytes it was checked on, and the report's speed
section says how with the calibration it was checked against (`tools/kbench_timer.h`):

- **Device cases are timed after the checks**, operands still on the device, back to back after
  one sustained warm-up (`--bench-warm-ms`, again after correctness work long enough to drop the
  clock). Cases are held until the `--max-bytes` budget fills, then timed.
- **A time is device timestamps around a batch** of back-to-back launches, divided by the launch
  count. The batch is queued behind a gate (`rad_stream_gate`, `core/device/device.h`) the host
  opens once the whole batch is queued, so neither the host's submission rate nor a host sync is
  in the interval; a keep-alive copy runs ahead of the gate so the card does not idle and drop its
  clock while the host submits. Two untimed launches precede every batch. A batch lasts a
  millisecond and fifty launch floors; a case takes batches until its median's 95% interval is
  within 1% or `--bench-reps` is reached. The report gives median, minimum, p90, batch count and
  the interval. A launch that waits on its own stream cannot run behind a gate; it is timed
  without one and marked `ungated`.
- **Cold by rotation.** Weights, weight tables and inputs of 1/32 of the footprint or more are
  read from one of a set of copies spanning the footprint (`--bench-flush`, default 256 MiB, at
  least twice an R9700's 8 MiB L2 plus 64 MiB last-level cache), so every launch reads them from
  memory as the engine's streamed weights are; smaller inputs stay warm, as the previous op
  leaves them. A weight table is read through pointer tables that send each launch's routed
  experts to different slots, so the experts a launch reads are cold although it reads only a
  few. Nothing is flushed, so no dirty lines are left for the timed kernel to write back.
- **In-place operands are restored** from a pristine copy before every launch. The restores are
  timed alone in the same window and subtracted; their share is in the notes.
- **Realistic inputs where the time depends on them**: the timing pass draws expert offsets and
  routing ids skewed (Zipf, exponent 0.5) and block tables as permutations. The checked launch
  keeps the recorded draw, so fixtures replay byte for byte.
- **Calibration**, printed above the table: the launch floor (the backend's own 4-byte fill, back
  to back and between two events), and streaming-read bandwidth (ranged copies whose writes all
  land in one cache-resident ring) warm and rotated-cold at four sizes up to
  the footprint. If a smaller cold probe runs more than 1.5x the largest's rate the rotation was
  served from a cache, and device times are withheld. The clock is read from the card's
  `gpu_metrics` after every batch (v1.3 tables: RDNA3 and RDNA4 discrete cards); a batch during
  which a throttler came on that was off at idle and after the warm-up is repeated, up to three
  times. A throttler on under the warm-up's load binds throughout and is named in the header
  instead.
- **Host cases are timed where they are checked**, in their own column: steady_clock on the
  calling thread after an untimed launch, in-place operands restored outside the timed region,
  weights rotated over twice the CPU's last-level cache (at least 256 MiB, or `--bench-flush`),
  on the OpenMP threads the notes name, for at most two seconds a case. A reference answer the
  cache served launched nothing to time; `--no-cache` times it too.

### Flags

| | |
|---|---|
| `--fixture FILE` | the recording to replay (default `$RADIANCE_HOME/kernels.rkb`); repeatable |
| `--record FILE` | with `-m`: write a fixture |
| `--skeleton FILE` | with `-m`: write a sparse copy of the container, its tables intact and its data a hole |
| `--rerecord FILE` | replay: write the fixtures' cases again with the reference's answers |
| `--record-rows FILE` | replay: record a row-driven fixture from the fixtures' cases |
| `-m, --model FILE` | the `.rad` to read weight operands from |
| `--home DIR`, `--kernels A:B` | plugin home, and the kernel libraries to load (default every one found) |
| `--op NAME` | check only this op (repeatable prefix match) |
| `--ref NAME` | the oracle plugin's name (default: the loaded library that declares itself a reference implementation, `libref` in this tree) |
| `--no-ref` | replay: leave the reference out of the libraries under test |
| `--strict` | replay: a stale fixture fails the run |
| `--all-rows` | replay: run every row whose constraints hold, not only the selected one |
| `--coverage DIR`, `--coverage-write` | replay: the row coverage ratchet, and writing its files |
| `--no-cache`, `--cache-dir DIR` | replay: run the reference instead of reusing its cached answers; where the cache lives |
| `--seed N` | base seed; every case derives its own, so changing a filter does not change any other case's inputs |
| `--tol X` | override every tolerance (for bisecting, not for passing) |
| `--max-ctx N` | the deepest context considered: declare runs at it and the attention sweep builds to it (default: the checkpoint's training context for declare, 16384 for the sweep). Attention is swept over depth as well as query length |
| `--max-tok N` | the `max_tok` declare runs at (default 512, small on purpose) |
| `--mm-max-patches N` | the encoder pass declare runs at, so a media tower's kernels are in the graph (default 2048, rounded to whole output rows, when the container carries an encoder; 0 leaves it out) |
| `--kv-cache-dtype T` | `bf16` or `fp8`, the cache width declare runs at, as on the engine (default bf16) |
| `--num-speculative-tokens N` | draft depth to declare at (default: the container's own); puts the drafter's kernels in the checked graph |
| `--max-cases N` | runs per distinct (op, kernel, library, domain, geometry, operand shapes); 0 means every one (default 4) |
| `--max-bytes MIB` | skip a case whose operands would need more than this, by name (default: half the memory available at start, at most 8192 and at least 512) |
| `--no-fusion` | skip the fusion oracle |
| `--bench` | time each case that passes, on the same bytes (Timing, above) |
| `--bench-iters N` | launches in every timed batch; 0 (default) sizes each batch to a millisecond and fifty launch floors |
| `--bench-reps N` | the most batches a case takes (default 20); fewer once the median's 95% interval is within 1% after six |
| `--bench-warm-ms N` | sustained load before each timing pass, in ms (default 2000; 0 for none) |
| `--bench-flush MIB` | the rotation footprint: weights and inputs of MIB/32 or more are read from copies spanning MIB, so from memory (default 256 on a card; twice the last-level cache and at least 256 for host rows); 0 times warm, every launch rereading one set of operands. Implies `--bench` |
| `--timing-only` | time every device kernel and check nothing: no reference run, read-back, margins or fusion chain, host rows skipped. For comparing builds a checked run has already passed; refuses `--record`. Implies `--bench` |
| `--report FILE` | write the markdown report |
| `--analysis FILE` | splice in a fragment from `tools/kbench-analysis.sh` |
| `-v` | print every case, not only failures |

### Notes

- A replay costs the input draw (threaded, and nothing for a case every library caps or skips)
  plus the libraries under test. Without `libref` a host replay takes seconds to a minute;
  `kbench_ref` takes minutes after a `libref` rebuild and seconds otherwise.
- `--max-bytes` matters on a loaded machine. One case holds every operand as drawn, again as a
  kernel's copy of what it writes, and again on the device. Without the cap a small machine is
  killed by the OOM killer with no indication of the op. The default is taken from the memory
  available when the run starts, which a server started later does not respect; with a server on
  the same machine, pass a small value:

  ```sh
  rad-kbench --home $RADIANCE_HOME --max-cases 1 --max-bytes 1024
  ```

- `--bench` times cold, the way the engine reads its weights. On an R9700 a 52 MB bf16 decode
  weight read warm (`--bench-flush 0`) comes from the last-level cache at 1.45 TB/s against the
  card's 565 GB/s from memory, so warm times rank kernels at a shape but do not size a roofline.
  Warm and cold are both device time per launch, so the two are comparable.
- A time includes the launch floor the header prints, the gap between two dependent dispatches,
  which the engine pays too; a kernel close to the floor is launch-bound.
- `--timing-only` times the same way a checked run does, after the same warm-up, without the
  checks in between. Compare builds within one mode, interleaved.
- `--max-cases`: a 62-layer model presents the same kernel at the same shape sixty times with only
  the weights differing. The fixtures are recorded at `--max-cases 1`.
- Fusion claims are checked byte for byte. A kernel that declares the chain it replaces is run
  against that chain, out of its own plugin, and the outputs compared with `memcmp`. That is what
  makes a fused kernel substitutable instead of a second numerical path.
- A kernel that cannot lay out a weight's encoding is a skip, not a failure: its layout hook
  answers `RAD_E_DTYPE`, and the report names the kernel, the operand and the encoding. A fixture
  in a format version this tree cannot read does not load; record it again. Version 4 still loads:
  it has no sketch, and its weight tables were recorded undrawn, so its grouped-GEMM cases are
  vacuous until it is recorded again.

---

## rad-tune

```
rad-tune -m <model.rad> [options]

  -m, --model FILE   the .rad whose declared graph supplies the shapes
      --home DIR     $RADIANCE_HOME; the cache is written to <home>/tune/<machine>.tune
      --kernels A:B  the kernel libraries to load, first wins; default every library found
      --op NAME      tune only ops whose name starts with NAME
      --machine KEY  override the machine key (for writing a cache for another machine)
      --reps N       timing reps, median reported (default 5)
      --iters N      launches per rep (default 32)
      --max-tok N    the max_tok declare is run at (default 8192)
      --tp N         tensor-parallel width to declare at (default 1)
      --num-speculative-tokens N|auto
                     draft depth to declare at (default auto: the container's own)
      --dry-run      list what would be measured and stop
  -v, --verbose      print every candidate, not just the winner
```

Kernels declare tunable axes, not named variants. `rad-tune` walks that space at this model's
shapes on this machine and writes the winner per (op, geometry) to
`$RADIANCE_HOME/tune/<machine>.tune`, which the engine's selector reads at startup.

```sh
rad-tune -m model.rad --home $RADIANCE_HOME --tp 2
rad-tune -m model.rad --op gemm --dry-run        # what would be measured
rad-tune -m model.rad --op attn_paged -v         # every candidate
```

Declare at the shapes you serve:

- `--tp N`: a sharded deployment divides every sharded N by the world size. A declare at world size
  1 measures a program no `--tp 2` serve runs.
- `--num-speculative-tokens N` puts the drafter's kernels in the measured graph. The drafter's
  shapes appear in no other model's tuning table.

At `--tp > 1` every rank declares at once, each on its own thread and device, as the engine does:
a collective's `init` is a rendezvous, so declaring rank 0 alone at world size 2 hangs.

Before trusting a cache, check that the tuned variants beat the defaults. `RADIANCE_TUNE` pins a
kernel's axes for such a comparison; it is a measurement aid, not a deployment setting.

---

## rad-schemas

```
rad-schemas libref.so                 print one plugin's vocabulary
rad-schemas libr4d.so libref.so       diff them, in hierarchy order
```

The first plugin in hierarchy order to declare an op fixes its schema, and a later plugin that
declares the same op differently is refused at load, with both plugins named. Arguments are
positional, so two disagreeing schemas would make the same call mean different things depending on
which plugin won selection.

The loader names one op at a time. `rad-schemas` prints every disagreement at once, so a new
plugin can be conformed to the vocabulary before it is loaded beside another. It exits 1 when any
schema disagrees.

It links no core: it `dlopen`s plugins and reads their static schema tables, so it works on a
plugin the engine cannot load. It takes no options; run it with no arguments for its usage.

```sh
rad-schemas $RADIANCE_HOME/kernels/libr4d.so $RADIANCE_HOME/kernels/libref.so
```

---

## rad-avx-check

```
rad-avx-check --ref libref.so --avx LIB.so [options]

  --ref PATH     the oracle plugin      (default libref.so)
  --avx PATH     the plugin under test  (default libavx.so); any host library
  --op NAME      only this op
  --level N      only this ISA level (0 scalar 1 avx 2 avx2 3 avx512)
  --threads N    rerun every case with each parallel region on N threads and
                 require the same bytes (default 4, 0 to skip; libavx only)
  --tol X        override every tolerance (for bisecting, not for passing)
  -v             print every case, not only the failures
```

Checks a host kernel library op by op against libref, on the CPU, with neither the engine nor a
card: two `dlopen` calls, a geometry table (`libavx/avx_cases.cpp`), libref's operand descriptions
and hand-built cases for the operand shapes and formats a description cannot draw. It is built
into the build's `radiance_home/kernels/` with libavx (and by a standalone libavx build against an
installed radiance), and ctest runs it as `avx_check`.

For every case: schemas as the loader compares them, the outputs within rad-kbench's tolerances
(ids exactly), guard bytes around every buffer, inputs unchanged, and an oracle output that is not
just the draw. libavx is also checked at every ISA level the CPU executes and with every parallel
region forced onto several threads, and must cover every libref op but those `avx_check.cpp` lists
as not implemented yet.

A new host library is checked with the same command, pointing `--avx` at it:

```sh
K=<build>/radiance_home/kernels
$K/rad-avx-check --ref $K/libref.so --avx ./build/mylib.so -v
```

A library not named `libavx` is checked as a host library: the libref ops it has no host row for
are listed rather than failed, each case runs on the row the selector would choose for its
geometry (constraints, then priority), and the ISA sweep and threaded rerun run only if it exports
`avx_isa_force` and `avx_par_force`. A row that returns `RAD_E_UNSUPPORTED` for a geometry libref
serves fails in either case: the selector would resolve to it and the step would fail at issue.
`libavx/README.md` has the details. Device libraries are checked by `rad-kbench`.

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

Per corpus file it checks three things:

1. The ids match those of the reference encoder in `tools/tokref` (llama.cpp's splitters and
   `std::regex`, a string-keyed merge).
2. The ids match HuggingFace `tokenizers`, read from files made elsewhere so that no Python is
   needed here.
3. Speed: MB/s for both encoders, and the cost of re-tokenising a growing conversation with and
   without the segment cache.

Any difference is printed with the ids and the text around it and makes the exit status non-zero,
so the tool can gate a change.

---

## The KL mode

```
radiance --model BF16.rad ... --kld-record DIR --kld-corpus CORPUS.jsonl    the reference, once
radiance --model QUANT.rad ... --kld-ref DIR [--kld-out REPORT.json]        each candidate
```

Measures how far a quantised model is from the one it was quantised from, as the distribution of
the per-position KL(P_bf16 || Q_quant) over the whole vocabulary: mean, median, 99th, 99.9th and
99.99th percentiles and maximum, top-1 agreement (the candidate's argmax is the reference's), and
both models' perplexity on the corpus's own next tokens, overall and per source. It is a mode of
the engine, like `--debug-graph`: it starts as a serve would, with the same placement and kernels,
runs the corpus through prefill instead of listening, and exits.

The reference is computed once. `--kld-record` writes the bf16 model's log-probabilities at every
scored position to DIR (f16 over the whole vocabulary, the 16 most likely tokens exactly, and the
tokens themselves). `--kld-ref` runs a candidate over those stored tokens and scores its own f32
logits against the stored rows, so a candidate costs one prefill of the corpus. DIR is never
overwritten, and is read only once its `kld.json` exists, which the recording writes last.

The corpus is JSONL, one `{"prompt", "score_from", "source"}` object a line; only `prompt` is
required. `prompt` is tokenised with control tokens recognised, as a rendered chat template is.
Positions `[score_from, n - 1)` are scored (`score_from` defaults to 0), and `source` groups the
report. The corpus behind the Flash-Next numbers in [MOE-W4.md](MOE-W4.md) is disjoint from the
calibration corpus, so a calibrated quantiser is not scored on the text it was fitted to. The tail
needs many positions: the 99.99th percentile of 83K positions rests on eight of them.

The mode changes these settings itself:

- Speculation and the prefix cache are off: a drafter writes no scored row, and a prefix hit would
  skip positions the reference holds. An explicit `--num-speculative-tokens` above 0 is refused.
- The step pipeline is off, since the scored rows are copied off the cards after the step.
- The logits buffer holds a row for every token of a step (`RadBuildCtx::max_out_rows`). At
  `--max-num-batched-tokens 4096` with a 248320-token vocabulary over two ranks that is about
  2 GiB per rank, plus the same in pinned host memory.

Everything else comes from the command line: run every candidate with the same flags, and the
reference with the exact wire and a bf16 KV cache.

The control vector's switch is per request, and KLD mode serves none, so a deployment that has one
loaded is scored with it **off**. `--control-vector on` starts the deployment default on, which is how
the projection itself is scored (the switch still decides per request, and a request's false wins).

`--kld-out FILE` writes the report as JSON, and every position's KL, both NLLs and the argmax
agreement to `FILE.rows` (`[positions, 4]` f32, reference row order), so two candidates can be
compared position by position.

A runtime LoRA adapter (GUIDE.md §4.9.1) is scored the same way: `--lora NAME=DIR --kld-lora NAME`
runs every document through the adapter, against a reference recorded from the merged checkpoint
(or from PEFT itself). `--kld-lora-sources PREFIX` runs only the documents whose `source` starts
with `PREFIX` through it and the rest as the base, in the same steps. The steps are those of the
all-adapter and all-base runs with the same flags, so each document's rows should be byte for byte
theirs: that is how a mixed batch is checked.

---

## The self-measurement mode

```
radiance --model M.rad ... --diagnose DIR [--diagnose-salt N]                    the run
radiance --model M.rad ... --diagnose DIR --diagnose-baseline BEFORE.json        the same run, against BEFORE
```

Starts the deployment as a serve would and then, instead of listening, runs one fixed workload
through the engine's own scheduler and writes `DIR/diagnose.json` and `DIR/diagnose.md`. The
workload is in the engine (`core/diagnose.h`) and does not move with the change being tested:
prompts held as token-id spans, and points that combine them at a concurrency, a request count and a
generation length -- single-sequence decode, decode at 1, 4 and 8, prose, code, structured output,
recall, long-context, prefill at 128, 512 and 2048 tokens, and one prompt arriving while streams are
already decoding. Everything else comes from the command line, because the chunk geometry, the
drafter and the cache settings are what is under test. [spec.md §16.1](../spec.md) is the contract.

The numbers are the scheduler's and not a client's. Each step records its own time, the tokens it
emitted and the gap since the previous step that emitted any (`core/sched/steplog.h`), which is what
separates a long step from a starved scheduler, and what prices a draft pass and an arrival. The
report is step time by step shape; the interval from one step that emitted tokens to the next, with
the part of it the engine spent waiting; tokens and rows per step; throughput; cache hits; draft
acceptance pooled and by category and by draft position; preemptions; and every generated token id.

```sh
radiance --model w4nl64-i8lm.rad --tp 2 --num-speculative-tokens 4 \
         --diagnose /tmp/diag-before --diagnose-salt 11
# ... the change, and a rebuild ...
radiance --model w4nl64-i8lm.rad --tp 2 --num-speculative-tokens 4 \
         --diagnose /tmp/diag-after --diagnose-salt 11 \
         --diagnose-baseline /tmp/diag-before/diagnose.json
```

Both runs write the same prompts, greedily and to the same lengths, so the second one's tokens are
the first one's unless the build changed what the model says. `--diagnose-baseline` fails the run on
any token difference, naming the point, the request and the first token where the two part, and keeps
every point whose tokens still match. It reports pooled draft acceptance -- per point, per category,
per draft position -- where it has fallen by more than `--diagnose-acceptance-drop`, a fraction of
what it was, default `0.10`; below that acceptance moves with the machine. A baseline of another
workload or a different salt is refused with both values named, because its tokens are not the ones
this run would have written.

Nothing in the workload is served from the prefix cache: every prompt is salted by `--diagnose-salt`,
the cache is dropped before each point, and a point the cache served anyway is written up as *not a
measurement* rather than as a fast one. A quantity that could not be measured -- a decode rate with
no decode step, an acceptance with no drafter, a point whose KV the pool cannot hold -- is absent
with the reason and what would have to change, never a zero. `DIR/diagnose.md` is the same content in
tables, for reading; the JSON is for a script.

Exit status is 0 for a run that measured, 2 for a refused command line (`--diagnose-baseline` with
no `--diagnose`), and 1 for a deployment that did not boot, a report that could not be written, or a
baseline that disagrees about the tokens. A point this deployment cannot hold is a line in the report
and not a failure of the run.

This is not the startup report (`core/diag/diag.h`), which every start prints: that one describes the
machine, the configuration and the PCIe path, and it is in the log above a diagnose run's lines.
[DEVLOG-self-measurement.md](DEVLOG-self-measurement.md) is the log of how the report's quantities
got their definitions, what the mode last measured, and three numbers that were wrong at first.

---

## The shell harnesses

`scripts/` drives a running engine: bring a model up, measure it, and assert on what comes back
over the wire. The scripts are versioned with the engine so that a measurement can be reproduced,
and each one's header records why it measures the way it does (how a window is sampled, which
counter to trust). [scripts/README.md](../scripts/README.md) has a line on each.

The scripts that start a server or a conversion expect the source tree and its `build/` directory
at a fixed path under `$HOME`, set at the top of each script, and the containers under
`$HOME/models/rad/`. Most other defaults (port, container, sizes) take an environment override,
listed in each script's header. The scripts that only attach to a live server need nothing but
`curl` and `jq`. Scripts find each other through `$(dirname "$0")`.

Which ones stop a running server:

- `fnserve.sh`, `mcserve.sh`, `fnstop.sh`, `fntrace.sh`, `dprofd.sh`, `mctrace.sh`, `mcagent.sh`
  and `mcdecl.sh` `pkill` any running engine, whatever port it is on.
- `oracle.sh` starts its own server and stops whatever listens on its port (`P`, default 8302).
- `cstep2.sh`, `ident.sh`, `tierident.sh`, `tiercross.sh`, `mctools.sh`, `mcstream.sh`, `hfget.sh`,
  `convfn.sh`, `convdspark.sh`, `fnpf.sh`, `fnconc.sh`, `fndrop.sh` and `cardclock.py` leave a
  running server alone. `fndrop.sh` leaves the process alone but disconnects clients from it on
  purpose.

### Bringing a model up

| | knobs |
|---|---|
| `mcserve.sh` | MiniCPM5-2B + DSpark drafter (`minicpm5-2b-dspark.v2.rad`), left running. `P` `TP` `MLEN` `SEQS` `M` `KVD` `HR` `MNBT` `EXTRA` `EXTRA_ENV` |
| `fnserve.sh` | Qwen3.8-Flash-Next, the production container `w4nl64-i8lm.rad`, left running. `P` `TP` `MLEN` `SEQS` `M` `NS` `HOSTPOOL` `KVD` `HR` `MNBT` `WIRE` `KVHOST` `EXTRA` `EXTRA_ENV` |
| `fnstop.sh` | stops whichever is running and waits for the process to exit |
| `convdspark.sh`, `convfn.sh` | rebuild the containers those two serve; the reference conversions. `convfn.sh` needs `CALIB=<dir>`: the production recipe's expert rule runs GPTQ against the calibration Grams there |
| `hfget.sh` | fetch a checkpoint from Hugging Face with `curl` and `jq`. Resumable; a file changed at the same size is fetched again (`.hfget-manifest`). `radiance --model <owner/name>` runs it too |

```sh
scripts/fnserve.sh                             # defaults: port 8100, tp 2, 200K ctx
P=8100 MLEN=131072 SEQS=4 scripts/fnserve.sh   # override
scripts/fnstop.sh
```

Both serve scripts print the endpoints and exit with the server still up. `fnserve.sh`'s header
gives the reason for each production flag.

### Measuring a server that is already up

These attach to a running server and never restart it, so they can be pointed at a live
deployment. `P` selects the port.

| | what it measures | knobs |
|---|---|---|
| `fnpf.sh` | prefill throughput, with a fresh nonce per rep so the prefix cache cannot serve it | `P` `N` `REPS` `BPT` |
| `fnconc.sh` | what a long prefill costs the requests running beside it: both on one clock, alone and then contended, with the warm-up pass discarded | `P` `C` `DEC` `PROMPT` |
| `fndrop.sh` | whether a client disconnecting mid-request breaks anything. Drop points are derived from one timed request, and every answer is checked against a reference | `P` `LONG` `GEN` |
| `cstep2.sh` | steady-state decode ms/step at each concurrency | `P` `LEVELS` `MT` `WIN` |
| `ident.sh` | run-to-run reproducibility at a fixed configuration: one hash per (temperature, question) | `P` |
| `cardclock.py` | what holds each card's clock under sustained load (power limit, temperature or current limit, or the top of the voltage/frequency curve) and which card the tensor-parallel step waits for, with the engine's step counters per window. Drives its own greedy load; `--watch` only samples while other traffic drives the server. Never writes to a card | `--port` `--conc` `--gen` `--seconds` `--window` `--seed` `--watch` `--json` `--compare` |

`fndrop.sh`: a disconnect reaps a request at an arbitrary point in the step loop, and every other
harness waits for its requests, so none of them sees what holds a reference across that point.
Staying up is not enough; a server that stays up but answers differently has a freed slot feeding
a live sequence. One diverged hash can be a tie flipped by a different batch; rerun before
treating it as a defect.

`ident.sh` does not expect a speculative run to match a non-speculative one byte for byte: the
split-K count is chosen from M, so the two use different reduction orders and a long answer can
flip a near-tie. A divergence within the first sentence or two is a defect.

```sh
P=8100 LEVELS="1 2 4 8" scripts/cstep2.sh
```

```
C        steps       tok   tok/step    ms/step   accept%
1          ...
```

`cstep2.sh` waits until exactly C sequences are running before sampling and re-checks the count at
the end: a window with fewer than C at either edge measures no single concurrency.

`cardclock.py`: on workstation cards the same image can measure several percent apart from one
boot to the next. A tensor-parallel step runs at its slowest card's clock, and under sustained load
the firmware picks each card's clock. A throttler bit counts only when the card's clock is lower
while it is set. Some firmware keeps a bit on at idle (`TEMP_HOTSPOT` on the Radeon AI PRO R9700),
so a bit the idle baseline also shows is listed and ignored. When the script names a setting
(power cap, fan target, voltage offset) it prints the command and changes nothing. Apply such a
setting only while no process holds `/dev/kfd`: on gfx1201 an SMU write under load can reset the
machine.

```sh
scripts/cardclock.py --port 8100 --conc 8 --seconds 180 --seed 7 --json before.json
scripts/cardclock.py --compare before.json after.json
```

### Measuring with a server of its own

| | |
|---|---|
| `dprofd.sh` | per-op device timing at depth and concurrency, on the 27B dense model with its DFlash2 drafter. Take the difference of two dumps, both after prefill ended |
| `fntrace.sh` | a Flash-Next prefill under `rocprofv3`, with the link counters from `/stats` across the same request |
| `mctrace.sh` | a MiniCPM5-2B decode trace under `rocprofv3`, formatted for `libr4d/isa/tsum.sh` |
| `oracle.sh` | every op compared against libref on the operands the model issued. Read it for MISMATCH lines and for the NOT CHECKED lines that name what cannot be compared. `OPS` `MAXN` `FROM`/`TO` `TOL` `M` |

### Regression checks against a live server

These cover what `tests/chat_test.cpp` cannot: the server, the wire format and a real model's
output.

| | |
|---|---|
| `tierident.sh` | whether a session that left VRAM comes back byte for byte, attention blocks and recurrent state both, after a full host tier and (with `TTL`) after it expired. `P` `H` `N` `TO_HOST` `FILL` `TTL` |
| `tiercross.sh` | the same with two sessions in flight, the only way to see one served the other's context |
| `mctools.sh` | the tool-call shapes hardest to parse: an array argument, two calls in one reply, prose before a call, plain chat with tools available. Asserts `finish_reason` and that no `<function`/`<param`/CDATA marker reaches content |
| `mcstream.sh` | streamed tool-call replies, asserting on raw SSE fragments. Streamed content cannot be retracted, so a repair applied at the end of a reply does not fix it; this check sees the difference |
| `mcagent.sh` | what speculation is worth on an agentic workload (chat + tools, a constrained decode) |
| `mcdecl.sh` | how often the drafter proposes. Acceptance (`radiance:draft_acceptance_ratio`, and the script's accept%) is conditional on having drafted, so it reads healthy while tokens per step collapse; blocks per step separates the two |

> **Not all of them leave a server alone.** `tierident.sh`, `tiercross.sh`, `mctools.sh` and
> `mcstream.sh` talk to a server already running (port 8100 unless `P` says otherwise) and leave
> it running. `mcagent.sh` and `mcdecl.sh` start and stop servers of their own through
> `mcserve.sh`, which begins with `pkill -f "bin/radianc[e]"`, so they stop any radiance running
> on the machine.

```sh
scripts/mctools.sh && scripts/mcstream.sh
```

### In `tools/`

`tools/kbench-analysis.sh` produces the static- and dynamic-analysis fragment that
`rad-kbench --analysis` splices into its report. `tools/ops_doc_check.sh` is the `ops_doc` test: it
checks `docs/OPS.md` against the schemas the plugins declare.
