# The conventional op vocabulary

Op names are free strings (`spec.md` §2.3) and the core never interprets one. `libref` can only
implement ops it knows, so its op list is the conventional vocabulary: an architecture plugin that
stays inside it can run somewhere, and an op outside it has no fallback (`spec.md` §17).

This file describes that vocabulary for architecture and kernel plugin authors. The schemas
themselves are the `RadOpSchema` rows the kernel plugins declare (`libref/` for every op it
implements, `libr4d/r4d_rows.cpp` for the rest).

The tables are machine-checked. `tests/CMakeLists.txt` registers `ops_doc`, which dumps every
plugin's schemas with `rad-schemas` and compares each table row with them: op, parameters,
operands, optional markers and weight markers. By hand:

    tools/ops_doc_check.sh docs/OPS.md build/bin/rad-schemas build/radiance_home/kernels/*.so

A row with a fourth column is a proposal, not a description, and is exempt from the check. This is
the only way an op no plugin declares yet may appear here.

Write against this document, not a guess: `qk_norm_rope_gate` and `qk_norm_rope` differ
where a guess goes wrong (one takes a precomputed cos/sin table, the other `theta`).

A table or op marked **device-only** is declared by libr4d alone. It has no host fallback and no
oracle for `rad-kbench`, so an architecture that uses it runs on libr4d or not at all. Every fusion
here is such a choice.

## Conventions

- Operands are positional, in the order given. `→` separates inputs from outputs.
- `w` marks a weight operand and `wt` a weight table (see
  [Weight operands and weight tables](#weight-operands-and-weight-tables));
  `:inout` an operand that is read and written; `?` an optional operand.
- `/opt` marks an optional parameter (a kernel that does not read it ignores it), `/der` a derived
  one ([Parameters the kernel supplies](#parameters-the-kernel-supplies)), `:f64` a float
  parameter and `:str` a string one where the name does not make it obvious.
- `(range)` marks the parameter callers declare as a range (`RAD_RANGE`); it is `M` wherever a token
  count appears. At most one parameter per op is ranged, which keeps `rad_issue` to a single
  integer on the hot path.
- A tensor's leading dimension is tokens unless stated. Everything is contiguous unless an op says
  otherwise; strides are the kernel's business, not the schema's.
- Where an op takes `dtype`, it is a string naming the operand combination, so a weight format and
  an activation format share one key: `bf16`, `f16`/`fp16`, `f32`/`fp32`, `u8`, `fp8_e4m3`,
  `fp8a8`, `fp8a8f32`, `fp8a16`, `i8a8`, `i8a8h`, `w4`, `w4a8`, `w4a8_asym`, `w4a8h`, `w8a8`,
  `w4a16`, `w8a16`, `w2a8`, `mxfp4a8`, `w5a16`, `w4a16_asym`, `w5a8i8`, `w4a8i8_asym`, and the
  codebook expert forms `w4nla8h`, `w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h` (`docs/MOE-W4.md`).
  - `w4a8h` is `w4a8` with a `group`-wide Walsh–Hadamard along K baked into the stored weight
    (`docs/MOE-W4.md`).
  - `i8a8` is an i8 weight with a bf16 scale per (row, 128 of K) against an int8 activation on
    the same grid; `i8a8h` is that weight rotated by a 128-wide Walsh–Hadamard along K, its scale
    carrying the 1/128 (libquant's `transform=fwht128`), met by `had_quant_act_i8g`.
  - `w4a8_asym` is the unsigned-nibble-minus-zero-point grid instead of the symmetric one.
  - `fp8a8` and `fp8a16` say whether a block-scaled E4M3 weight meets an fp8 or a bf16 activation.
  - `fp8a8f32` is `fp8a8` with f32 weight scales on the 128x128 block grid.
  - `w5a16` and `w4a16_asym` are ParoQuant's int5 and z-lab's int4 (u5 / u4 codes, an f16 scale and
    a u8 zero point per 128 of K) against an f16 activation; `w5a8i8` and `w4a8i8_asym` are the same
    weights against an int8 activation with an f32 scale per (row, 128 of K).
  - `q_dtype` and `kv_dtype` are the same key split in two where attention reads two formats.
- Every op returns `RAD_OK` or a negative `RAD_E_*`. There is no partial success.
- **Zero rows is not a call.** An op issued over zero rows (zero along its ranged extent, or a
  `cu_seqlens` whose sequences are all empty) returns `RAD_E_SHAPE` and writes nothing. A caller
  with nothing to compute skips the issue, as the architectures' decode and prefill halves do.
  libref holds every op to this, so a host run names an unguarded issue.
- An implementation refuses what it cannot compute: an operand dtype it does not read
  (`RAD_E_DTYPE`) or an optional operand it does not implement (`RAD_E_UNSUPPORTED`). Returning
  `RAD_OK` with the answer to a different function is the one wrong response.
- libref accumulates in f32 (`libref/ref_common.h`). Its double-precision sums are part of the
  op's reference: `seg_mean`, `gram_accum`, `logit_rerank`, the normaliser of `sample_logprobs`,
  `givens_dense_quant`'s rotation, `ple_gate`'s dot, and the sums of squares of the norms in
  `hc_read`, `mtp_enter`, `ple_gate` and `qsa_block_key`.

## How a schema is written

`libref` and `libr4d` implement the same ops, and a schema they disagree on cannot be loaded: the
loader refuses the whole plugin on any one conflict, which makes all its other rows unreachable
and `rad-kbench` unrunnable. Four rules settle every case; they are why `gdn_conv_prep` and
`gdn_recurrent_update` carry long operand lists.

### The four rules

1. **Operands are the union**, and an operand only one implementation needs is optional. A schema
   that lacks an operand cannot express the call at all. An implementation ignores an optional
   operand it does not use; it cannot invent one it needs.
2. **Parameters are the union**, and a parameter only one implementation needs is optional. A
   constraint on an absent key does not hold, so a row that needs a parameter the caller did not
   state does not match.
3. **A float-valued parameter is `RAD_P_F64`**: `eps`, `scale`, `theta`, `softplus_thr`, `l2_eps`,
   `wadd`. A `"%.9g"` string would round-trip an IEEE single but is unreadable in a graph dump.
   Constraints never match on a float: a tolerance is not a predicate, and a kernel that depends on
   `eps` to that precision has a bug.
4. **An enumerated parameter is `RAD_P_STR` with named values.** `act` is `"silu"` or `"sigmoid"`,
   not `0` or `1`, so a graph dump says what runs. A plugin's launch adapter maps the name to
   whatever integer its kernel wants.

The union rule has two deliberate exceptions, both on the gated delta net: `cu` and `num_accepted`
are required ([Linear attention](#linear-attention-gated-delta-net)).

### Parameters the kernel supplies

`attn_paged.block_size`, `gdn_chunk_scan.chunk`, `gdn_kkt_solve.chunk` and `gdn_conv_prep.chunk`
(and the `block_size` of `attn_paged_qrow` and `attn_prefix_self`) are supplied by the kernel, not
the caller (`spec.md` §7.2, `RAD_DERIVED` in `rad_abi.h`). Constraints on them are skipped during
matching, and the winning kernel's `RAD_C_EQ` value is written into the resolved geometry
afterwards. Declared `RAD_REQUIRED` they would be unselectable: a kernel compiled for
`block_size == 16` could never match the query whose answer is to be read off it.

Nothing in `gdn_conv_prep`'s operands carries the chunk length, so `libref`, which has no
`RAD_C_EQ` to supply it, returns `RAD_E_SHAPE` when a caller omits `chunk` rather than treating the
whole sequence as one chunk. libr4d supplies 64.

`all_reduce.hops` is optional: a one-shot/two-shot preference the caller may state and usually
should not. `all_reduce.exact` is required: a caller that does not accept a lossy wire must get no
kernel rather than the quantised one.

### Weight operands and weight tables

- **`RAD_OPD_WEIGHT`** (`w`): the issue names one weight handle, and the kernel receives that
  weight's bytes, dtype and declared shape.
- **`RAD_OPD_WTAB`** (`wt`): the issue names the first handle of a run and the run's length, and
  the kernel receives a device array of pointers, one per entry. `RAD_WTAB(h, n)` in
  `abi/rad_runtime.h` is the issue-side spelling; `rad_decl_weight` hands out consecutive indices,
  so a declaration loop over experts produces such a run.

A table, not a base plus a stride, because a layer's experts are separate movement units
(`spec.md` §4.2): after placement one may be in VRAM, the next streamed over the link and the third
still on disk. `base + e * stride` describes only the case where nothing was offloaded. The core
resolves every handle of the run like any weight (residency table, mover event, use note) and
uploads a pointer array, patching it only when a pointer moves.

What the kernel sees: `RadTensor::data` points at the pointer array, `shape[0]` is the entry count
and `stride[0]` is 0, which marks a table. `dtype` and the trailing extents and strides are one
entry's, and every entry of a run must agree on them; the core refuses the issue by name if they
do not. A kernel may also accept the stacked form (an ordinary rank-3 tensor with a real leading
stride): `rad-kbench` replaying a fixture has nowhere to put a pointer array, and libref and libr4d
both read either form.

How the declared weights are divided: after each fixed weight position takes one weight, the rest
is split evenly across the table positions in declaration order, and where it does not divide the
first positions take one more each. So `moe_gemm_q_mod4`'s first `n_expert % 4` classes hold one
expert more than the rest, and their codes and scales are its first table positions. A list too
short to give every table an entry is refused at declare; a table whose declared run is not
contiguous is refused when the tables are bound, before the first step; entries that disagree on
shape, or an issue naming a run other than the declared one, are refused at issue.

### Weights of several planes

A weight may be declared over several planes of one stored tensor (`RadWeightDecl::planes`,
`spec.md` §4.3) and still be one operand: an fp8 or int8 lm head (codes and scale), Flash-Next's
E4M3 mixing and fc rows (codes and f32 group scales), a zero-point grid's `b_scale` (scale and
zero). A device kernel arranges such a weight in its layout hook. A host kernel reads the joined
form `abi/rad_planes.h` defines: a header naming each plane by role, then the planes as the
container holds them. libref and libavx join the same selections on the same operands and refuse
the others with `RAD_E_DTYPE`, so either may serve any band:

| op | operand | joined planes |
|---|---|---|
| logits_gemm, logit_rerank | `lm_head` / `head` | codes (E4M3 or i8, one a value) and one scale plane (bf16, f16 or f32, any block) |
| hc_read | `mix_down`, `mix_up` | the same (`mix` e4m3) |
| mtp_enter | `fc_h`, `fc_e` | the same (`fc` e4m3) |
| gemm_nt_q | `b_scale` | `scale` (f16) and `zero` (u8, u4 or u5), one each per (row, `group` of K), at the zero-point grids below |

The kernel finds the joined form by its header and its extents from the planes' entries, not the
operand's declared shape. A joined head or matrix is read element by element as the core's affine
decoder gives it (`core/format/encoding.h`): the code times the scale covering it, one f32
rounding, so libref's product over a joined head is bit-identical to its product over that
decoder's f32 head. A single plane is never joined, and the plain forms (a bf16 head, the logical
f32 head a tool describes) reach the same ops unchanged. Where a joined `hc_read` or `mtp_enter`
matrix rounds a gate, it rounds to `h`'s dtype, as a plain matrix rounds to its own.

## Loader checks

Every op a kernel row implements must have a schema declared by some loaded plugin. The check runs
once the whole hierarchy is open, because an override plugin may rely on the schema of a plugin
below it, and the refusal names the kernel, the plugin and the op. Without it a typo and an
unimplemented op give the same diagnostic.

The first plugin in hierarchy order to declare an op fixes its schema. A later plugin declaring the
same op with a different schema (parameter keys, types and requiredness; operand count, roles and
optionality) is refused at load with both plugins named; an identical declaration is expected.
Since arguments are positional, two disagreeing schemas would give one call two meanings depending
on which plugin won selection.

`build/bin/rad-schemas a.so b.so` prints what plugins declare and every disagreement between them
at once. Run it before adding a kernel plugin; the loader names only the first conflict.

---

## Elementwise and norms

| op | params | operands |
|---|---|---|
| `rmsnorm` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt `mult`:f64/opt | `x`, `w`w? → `y` (no gain operand: a gain of one) |
| `rmsnorm_p` | `M`(range) `n` `eps`:f64 `dtype` | `x` → `y` |
| `rmsnorm_add` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt | `x`, `residual`, `w`w → `y`, `residual_out` |
| `hc_enter` | `M`(range) `n` `hc` `dtype` | `x` → `h` |
| `mhc_pre` | `M`(range) `n` `hc` `iterations` `eps`:f64 `hc_eps`:f64 `dtype` | `h`, `fn`w, `scale`w, `base`w → `x`, `post`, `comb` |
| `mhc_pre_partial` | `M`(range) `n` `hc` `chunk` `first` `chunks` `dtype` | `h`, `fn`w → `part` |
| `mhc_pre_finish` | `M`(range) `n` `hc` `iterations` `eps`:f64 `hc_eps`:f64 `chunk` `dtype` | `part`, `h`, `scale`w, `base`w → `x`, `post`, `comb` |
| `mhc_post` | `M`(range) `n` `hc` `dtype` | `x`, `h`, `post`, `comb` → `out` |
| `mhc_contract` | `M`(range) `n` `hc` `dtype` | `h` → `x` |
| `hc_read` | `M`(range) `n` `hc` `lowrank` `eps`:f64 `dtype` `wadd`:f64/opt `inject`/opt `group`/opt `rotate`/opt `mix`/opt | `h`, `w`w, `mix_down`w, `mix_up`w, `inject_w`w? → `x`, `inj`?, `q`?, `scale`?, `rq`?, `rscale`? |
| `hc_write` | `M`(range) `n` `hc` `dtype` | `y`, `inj`, `h`:inout |
| `hc_project` | `M`(range) `n` `hc` `scale`:f64 `dtype` | `h`:inout, `v`, `on` |
| `layernorm` | `M`(range) `n` `eps`:f64 `dtype` | `x`, `w`w, `b`w? → `y` |
| `grid_embed` | `M`(range) `n` `side` `dtype` | `x`:inout, `table`w, `coord` |
| `grid_embed_aa` | `M`(range) `n` `side` `dtype` | `x`:inout, `table`w, `coord` |
| `add` | `M`(range) `n` `dtype` `div`:f64/opt | `a`, `b` → `y` |
| `mul` | `M`(range) `n` `dtype` | `a`, `b` → `y` |
| `silu_mul` | `M`(range) `n` `dtype` | `gate_up` → `y` (splits `[M,2n]` into gate·up) |
| `swiglu_clamped` | `M`(range) `n` `limit`:f64 `dtype` | `gate_up` → `y` |
| `mul_weight` | `M`(range) `n` `dtype` `scalar`/opt | `x`, `w`w → `y` (w one element or one row) |
| `act_mul` | `M`(range) `n` `act` `dtype` | `a`, `b` → `y` (act(a)·b, two operands) |
| `seg_mean` | `M`(range) `n` `dtype` | `x`, `cu_seqlens` → `y` (one mean row a segment) |
| `pool_rows` | `M`(range) `n` `k` `mult`:f64/opt `dtype` | `x` → `y` (one mean row a window of k rows) |
| `relu` | `M`(range) `n` `dtype` | `x` → `y` |
| `clamp` | `M`(range) `n` `dtype` | `x`, `lo`w, `hi`w → `y` |
| `conv2d` | `M`(range) `width` `cin` `cout` `kh` `kw` `stride` `pad` `cu_div`/opt `valid_stride`/opt `dtype` `t_last`/opt `act`/opt | `x`, `w`w, `cu_seqlens`, `valid`? → `y`; `b`w? |
| `conv2d_unfold` | `M`(range) `width` `cin` `kh` `kw` `stride` `pad` `cu_div`/opt `valid_stride`/opt `t_last`/opt `dtype` | `x`, `cu_seqlens`, `valid`? → `y` |
| `conv1d_dw` | `M`(range) `n` `k` `cu_div`/opt `dtype` | `x`, `w`w, `cu_seqlens` → `y`; `b`w? |
| `rel_sinusoid` | `M`(range) `n` `max_timescale`:f64/opt `dtype` | → `y` |
| `gelu_mul` | `M`(range) `n` `dtype` | `gate_up` → `y` (splits `[M,2n]` into gelu·up, tanh approx) |
| `seg_sinusoid` | `M`(range) `n` `period` `max_timescale`:f64/opt `cu_div`/opt `dtype` | `cu_seqlens` → `y` |
| `gelu` / `silu` | `M`(range) `n` `dtype` | `x` → `y` |
| `gelu_quick` | `M`(range) `n` `dtype` | `x` → `y` |
| `cast` | `M`(range) `n` `from` `to` | `x` → `y` |
| `softmax` | `M`(range) `n` `dtype` | `x` → `y` |
| `sigmoid` | `M`(range) `n` `dtype` | `x` → `y` |
| `logit_softcap` | `M`(range) `n` `cap`:f64 `dtype` | `x` → `y` |
| `scale` | `M`(range) `n` `dtype` | `x`, `s`w → `y` |
| `silu` | `M` `n` `dtype` | `x` → `y` |
| `fill` | `M`(range) `n` `dtype` `value`:f64/opt | → `y` |

`gelu` (exact erf) and `silu` share one schema; `silu` has a row of its own because a row names one
op.

**`gelu_quick`** is CLIP's `quick_gelu`, `x * sigmoid(1.702 x)`, rounded where torch stores each
tensor in a reduced dtype: `a = rd(1.702 x)`, `s = rd(1 / (1 + exp(-a)))`, `y = rd(x s)`, `rd` the
rounding to `y`'s dtype and the arithmetic f32 (CLIP-L's feed-forward in SDXL's text encoder).

**`rmsnorm`** is `y = x * rsqrt(mean(x^2) + eps) * (w + wadd)`; `wadd` is GemmaRMSNorm's `1 +`,
added in f32. `mult` scales the normed row before its one rounding, for a residual branch weighted
by a constant (Gemma 4's audio feed-forward adds half its output). For a power of two this equals
norm-then-scale exactly, since scaling by a power of two commutes with rounding.

**`rmsnorm_p`** is `rmsnorm` with no gain: `y = x * rsqrt(mean(x^2) + eps)`. Gemma's attention
normalises the value with a bare RMS while the key takes the `k_norm` gain, and the checkpoint has
no plane to feed `rmsnorm` as a gain.

**`rmsnorm_add`** is the residual add and the next norm in one pass, as every real implementation
does it. A plugin asks for it first and falls back to `add` + `rmsnorm` (`spec.md` §2.3).

**`gelu_mul`** is `silu_mul` with the tanh GELU: Gemma's gated FFN is trained with
`gelu_pytorch_tanh`, while the unary `gelu` is the exact erf form. Both split the fused `gate_up`
with the gate in the first half.

`swiglu_clamped` splits `[M, 2n]` as gate then up, as `silu_mul` does, and computes
`silu(min(gate, limit)) * clamp(up, -limit, limit)` in f32, rounded once to `y`'s dtype. `limit`
is positive and finite (GLM-5.3's is 10). The clip precedes the SiLU, and the gate has no lower
clip.

libr4d's `swiglu_clamped_bf16` serves bf16 in and out at any width, pitched rows, `M` up to
65535.

**`add` and `mul`** broadcast a `b` of exactly `n` elements along the rows of a taller output (a
bias add). Nothing else broadcasts, in these or any other elementwise op. `add`'s `div` divides
the sum after its rounding, `y = rd(rd(a + b) / div)` with `rd` `y`'s dtype (absent, 1: the plain
sum): the last step of a mean summed pairwise, as a vocoder averages its residual blocks,
`add(add(b0, b1), b2, div 3)` for `(b0 + b1 + b2) / 3`. A `div` of 0 is refused.

**`sigmoid`**, with `mul`, gates a shared expert (`ffn_gate_inp_shexp`). Without it the shared
expert runs ungated, which is a different model.

**`mul_weight`** is a learned gain on an activation: `w` is a weight of one element (a scalar for
the whole output, Gemma 4's per-layer `layer_scalar`) or one row broadcast down the rows (a
per-channel layer scale). It is its own op because the schema fixes whether an operand is a weight
or an activation, and `mul`'s `b` is an activation. Kernels read the form off the operand;
`scalar` 1 states the one-element form so a shape description (`rad-kbench`'s) builds the right
weight.

**`act_mul`** is `silu_mul` with its halves in two operands, so the multiplier can be any tensor:
two column slices of a fused `gate_up`, or a gate and a separate per-layer input (EmbeddingGemma 2's
per-layer-embedding block multiplies `gelu_tanh(gate(x))` by one). `act` is `silu`, `gelu` (exact
erf), `gelu_tanh` or `sigmoid` (a GLU: `a` the gate half, `b` the value half). The activation is
rounded to `y`'s dtype before the product, as the unfused pair stores it.

**`seg_mean`** is mean pooling over a ragged batch: row `s` of `y` is the mean of `x`'s rows
`[cu[s], cu[s+1])`, summed in double and rounded once. `M` is the step's token count and bands the
op; the segment count is `cu_seqlens`' length less one. An empty segment is a zero row.

**`pool_rows`** is a vision pooler over windows laid out consecutively: row `r` of `y` is the mean
of `x`'s rows `[r*k, r*k + k)`, summed in f32 and rounded to `y`'s dtype, then times `mult` and
rounded again (the unfused pair stores the mean). `M` is the input row count, a multiple of `k`.

**`fill`** writes `y = value` over the extent (`value` 0 when absent). A rank issues it where it
computes nothing but a downstream reader still expects a defined buffer: an attention-zero rank's
block (`docs/TP3.md`) is one `fill` of its output wire before the all-reduce. A wire shared across
layers cannot be zeroed once a step, and a memset knows neither the row pitch nor the dtype.

**`grid_embed`** adds a learned `side x side` position table resampled to each row's grid. `coord`
is `[4, M]` (row, column, grid height, grid width); sampling is bilinear with the grid's corners on
the table's (align_corners), edge taps clamped, the four taps summed in f32 and rounded to `x`'s
dtype before the add.

**`grid_embed_aa`** is the same addition with SigLIP2's resample, torch's
`F.interpolate(mode="bilinear", align_corners=False, antialias=True)`: per axis a triangle filter on
half-pixel centres, widened by the scale when the grid is smaller than the table, its in-table taps
normalised to sum to one; separable, f32, rounded once. A different function from `grid_embed`'s,
so a different op rather than a mode a library could ignore.

### Hyper-connections

`hc_project` is the control vector's projection (core/cvec.h): `h[s] -=
scale * (h[s] . v) * v` for each of the `hc` streams of a row whose `on` flag (an int32 per row) is
1, `v` being that layer's unit direction and `scale` the norm the file stored. It is llama.cpp's
`--cvec-mode project`, applied after a layer's write into the stream. A row whose flag is 0 is
neither read nor written, so an unprojected request's stream is the stock model's; the direction table is f32 and the op is issued only by an architecture that was
declared with one (`RadBuildCtx::cvec`).

`hc_read` and `hc_write` are the gated residual pair a hyper-connection model has instead of a
pre-norm and a residual add. The stream is `hc` times the model width; every block reads a mix of
the `hc` sub-streams and writes back into all of them, both gated by data:

    hc_read     N      = grouped_rmsnorm(h)          one rsqrt per n-wide stream, one hc*n gain
                g      = sigmoid(mix_up @ silu(mix_down @ N / hc))
                x      = mean over the hc streams of g * N
                inj[s] = 2 * sigmoid(inject_w[s] . N / hc)
    hc_write    h[s]  += inj[s] * y

Three facts a second implementation must match: the mix is a mean over the streams, not a sum;
each of the two gate arguments has its own `/ hc`; and `hc_write` adds into the unnormed `h`. The
normalisation feeds the block input and the gates and never enters the residual. Writing into `N`
instead re-normalises the stream at every connection and still produces fluent text, so the error
is easy to miss.

- `inject_w` absent is the final mixer, which collapses the wide stream to one and writes nothing
  back.
- `group` absent is a read whose consumer wants bf16. Otherwise the fp8 form of `x` (`q`, `scale`)
  comes out of the same pass, quantised from the narrowed `x` rather than the accumulator, so a
  fused kernel and the reference agree bit for bit.
- `rotate` adds `rq`/`rscale`: `had_quant_act_fp8` of the same stored `x`, each group rotated by an
  unnormalised Hadamard before its absmax, for a consumer whose weight carries the rotation (the
  `w4a8h` experts).
- `mix` is the mixing matrices' stored format: `bf16`, or `e4m3` (rows of E4M3 codes with f32
  scales in each row's own tail, one per 128 columns for `mix_down` and one per quarter row for
  `mix_up`). It changes the bytes read, not the arithmetic. The matrices are replicated on every
  rank and read once per connection; on Qwen3.8-Flash-Next (96 connections) `e4m3` halves their
  1.26 GB a step. The scales in the row's tail are libr4d's arrangement; the container holds codes
  and an f32 scale plane (a group of 128 for `mix_down`, a quarter row for `mix_up`), which a host
  kernel reads joined (Weights of several planes).

**mHC** (manifold-constrained hyper-connections) keeps `hc` residual streams in one `[M, hc*n]`
row and is a separate family from `hc_read`/`hc_write`. `mhc_pre` projects the whole row with the
f32 matrix `fn` `[hc*(hc+2), hc*n]` and multiplies each projection by
`1 / sqrt(sum(h^2) / (hc*n) + eps)`, giving coefficients `c`. `scale` holds three f32 scalars and
`base` `hc*(hc+2)` f32 biases: the pre gates are `sigmoid(c[j]*scale[0] + base[j]) + hc_eps` and
the post gates `2*sigmoid(c[hc+j]*scale[1] + base[hc+j])` for `j < hc`, and the remaining `hc*hc`
coefficients, times `scale[2]` plus their bias, are the mixing logits indexed
`[source, destination]`. The first Sinkhorn pass is a row softmax plus `hc_eps`, then column
normalization; every later pass normalizes the rows, then the columns; each normalization divides
by the sum plus `hc_eps`. `iterations` counts the passes, the first included. `x` is the sum of the
streams weighted by the pre gates, rounded once to `dtype`; `post` `[M, hc]` and `comb`
`[M, hc*hc]` stay f32. Transposing `comb` leaves every shape valid and changes the model.
`mhc_post` writes `out[dst, i] = post[dst]*x[i] + sum_src comb[src, dst]*h[src, i]` into a
separate buffer (`out` must not alias `h`), with no normalization of the stream. `mhc_contract` is
the mean over the streams, rounded once.

`mhc_pre_partial` and `mhc_pre_finish` are `mhc_pre` split by residual columns, so
tensor-parallel ranks share its projection instead of each computing it whole. Chunk `c` covers
columns `[c*chunk, (c+1)*chunk)` of the `hc*n` row (`chunk` divides it). `mhc_pre_partial`
computes chunks `[first, first+chunks)`: `part[m, (c-first)*S + j]` is the chunk's sum of
`h*fn[j]` for `j < hc*(hc+2)` and of `h*h` for `j = hc*(hc+2)`, with `S = hc*(hc+2)+1` rounded up
to a multiple of four and the slots past the last zero, so a gathered payload is defined. A
row-interleaved `all_gather` with `row = chunks*S` joins the ranks' matrices into
`[M, (hc*n/chunk)*S]`, which `mhc_pre_finish` reads: each coefficient and the sum of squares is
the sum of its chunks' partials, then `mhc_pre`'s gates, Sinkhorn mixing and collapse. The order
of that sum is the implementation's, so the pair matches `mhc_pre` to rounding, and bit for bit
only where an implementation says so.

libr4d serves `mhc_pre`, `mhc_post` and `mhc_contract` at `hc` 2 to 8, `n` up to 16384 and
`M` up to 65535 (`iterations` 1 to 256), bf16 activations with f32 `fn`, `scale`, `base`,
`post` and `comb`. At GLM's geometry (`n` 4096, `hc` 4) `mhc_pre` runs as eight 2048-column chunk
partials held in `x`'s rows, then a finishing kernel; other geometries project and finish in one
workgroup per row. `mhc_pre_partial` and `mhc_pre_finish` serve that geometry only (`n` 4096,
`hc` 4, `chunk` 2048, a 28-float chunk stride) and run `mhc_pre`'s own projection and finishing
kernels with its chunk-sum order, so the pair is byte-identical to `mhc_pre` (`r4d_selftest`,
`tests/r4d_emu/test_mhc.cpp`). The projection takes a four-slot form from two rows, one row a wave
and two from 32, where the weight rows are 16-byte and the residual rows 8-byte aligned; the
finish forms the 4x4 mixing on one wave and collapses the row over the workgroup; `mhc_post` at
`hc` 4 writes eight columns of every stream a thread on 16-byte rows. Every form gives the bytes
of the scalar kernels that serve the rows it cannot read.

### Gemma 4's audio tower

- **`relu`** is `max(x, 0)` at each operand's own row pitch.
- **`clamp`** is `min(max(x, lo), hi)` with `lo` and `hi` weights of one element each: a clipped
  linear's input or output range, stated per projection in the checkpoint.
- **`conv2d`** convolves over (time, width) with channels innermost: element `(t, w, c)` of `x` is
  at row `t`, column `w * cin + c`, and `w` is PyTorch's `[cout, cin, kh, kw]`. Each `cu_seqlens`
  segment (rows `[cu[s] / cu_div, cu[s+1] / cu_div)`) is convolved on its own with zero padding
  `pad` at its edges on both axes, so two clips in one pass never see each other, and its output
  rows start at `cu[s] / (cu_div * stride)` (a segment whose start does not divide is refused).
  With `valid`, input row `t` reads as zeros unless `valid[t * valid_stride] != 0`: the frame mask
  the subsampler applies before each convolution. f32 sum, one rounding.
- **`conv1d_dw`** is a causal depthwise convolution over time,
  `y[t, c] = sum_j w[c, 0, j] * x[t - (k - 1) + j, c] + b[c]`, with rows before the segment's start
  reading as zeros and `b` zero when unbound.
- **`rel_sinusoid`** writes the sinusoidal relative-position table, row `d` for distance `d`:
  `sin(d * inv[i])` over the first half and `cos(d * inv[i])` over the second,
  `inv[i] = exp(i * -log(max_timescale) / max(n/2 - 1, 1))`, each step in f32 as the reference's
  torch expressions are, then rounded.

### Gemma 4 fusions (device-only)

| op | params | operands |
|---|---|---|
| `norm_add_norm` | `M`(range) `n` `eps`:f64 `dtype` | `a`, `w_a`w, `x`:inout, `w_y`w → `y` |
| `norm_add_norm_route` | `M`(range) `n` `eps`:f64 `dtype` `mult`:f64 `group` | `a`, `w_a`w, `x`:inout, `w_y`w → `y`; `w_r`w → `r`; `w_e`w → `q`, `qs` |
| `join_add_scale` | `M`(range) `n` `eps`:f64 `dtype` | `d`, `w_d`w, `r`, `w_r`w, `w_j`w, `x`:inout, `s`w, `w_n`w? → `y`? |
| `expert_scale` | `M`(range) `n_expert` `top_k` `dtype` | `ids`, `w_e`w, `ew`:inout |
| `attn_paged_qrow` | `q_len`(range) `head_dim` `gqa` `block_size`/der `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out` |

Gemma's sandwich norms, each one kernel in place of a chain of `rmsnorm`, `add` and `mul_weight`:

- `norm_add_norm` is `t = rmsnorm(a) * w_a; x += t; y = rmsnorm(x) * w_y`: a post-norm, the
  residual add and the next pre-norm.
- `join_add_scale` is `x = (x + rmsnorm(rmsnorm(d) * w_d + rmsnorm(r) * w_r) * w_j) * s`: two arms'
  post-norms, their sum, its norm, the residual add and a one-element layer scalar (Gemma 4's MoE
  feed-forward). Given `w_n` and `y` (both or neither) it also writes `y = rmsnorm(x) * w_n` over
  the new `x`: the next layer's input norm, or the final norm after the last layer.
- `norm_add_norm_route` is `norm_add_norm` plus, from the same norm of the new `x`, a MoE layer's
  two routed inputs: `r = rmsnorm(x) * w_r * mult` (`rmsnorm`'s expression with `mult`), and `q`,
  `qs` = `rmsnorm_quant_fp8` of `x` under `w_e` (E4M3 codes of the bf16-rounded norm, an f32 scale
  per (row, `group` columns) of `amax / 448`, 1 for an all-zero group, the value over it clamped to
  ±448). `n` is whole groups and `group` is 128. In Gemma 4 it replaces the router's `rmsnorm` and
  the experts' `rmsnorm_quant_fp8`.

Every value the chain stores is rounded to the activation dtype at the same point, so the fusions
differ from the chain only in the order a sum of squares is added: with the same order they are
bit-identical, and against a chain summed in another block width a norm factor can differ in its
last bit.

`expert_scale` is `ew[t, k] = ew[t, k] * w_e[ids[t, k]]` over `[M, top_k]`, `w_e` an `[n_expert]`
table and an id outside it reading 0: Gemma 4's `per_expert_scale` on the renormalised routing
weights, in place of an `embed_lookup` and a `mul`. One product rounded once, so the pair's value
bit for bit.

`attn_paged_qrow` is `attn_paged` (same parameters, operands and values) with `q` read at its own
row pitch: the q part of a fused `[q|k|v]` projection, read where it lies instead of copied into a
dense block by a `cast`. It has its own name because other `attn_paged` kernels refuse a pitched `q`
(libr4d's with `RAD_E_STRIDE`, at the first step), and under `attn_paged` the first library in
`--kernels` wins the op, an order the architecture cannot see. Asked for by name, the op resolves to
a library that reads the pitch or to nothing, and the architecture copies `q` for `attn_paged` in
the second case. In libr4d the windowed kernels serve it at every height; the split-KV decode kernel
serves it at h256/gqa2 over its decode band (q_len up to 32, ahead of the windowed rows there), and
at h128/gqa8 up to q_len 16 by reading each kv head for two query groups of 4 (a drafter's cache
then holds each kv head once). Since no host library declares the op, a plugin that asks for it
learns whether a card serves the geometry.

libref has no row for any of them, deliberately. An architecture asks for a fusion and declares
the chain where it does not resolve (`rad_op_resolved`); a reference row would make it resolve
everywhere, so a model loaded without the serving library would resolve to the reference and be
refused at serve, instead of running the chain on libr4d or libavx. libr4d declares them with
device rows only, so the host backend runs the chains. With no oracle, its kernels are held bit
for bit to a chain summed in their own block order by `tests/r4d_emu`. No row declares `replaces`:
`rad-kbench`'s fusion oracle runs the replaced chain on the fusing library's own rows and compares
bytes, and libr4d's `rmsnorm` adds its sums of squares in another block order.

**The Whisper-style audio encoder's** (Qwen3-ASR's, `arch/qwen3asr_bf16`), which convolves each
second of a clip apart and attends in windows of eight seconds:

- **`conv2d`'s `t_last`, `b` and `act`.** `t_last` 1 reads the weight as PyTorch's `[cout, cin, kw,
  kh]`, time last: the kernel of a model that convolves a `[mel, time]` spectrogram, whose first
  spatial axis is the one this op calls width. `b`, after `y` so a call without it keeps its five
  operands, is a bias per output channel added to the f32 sum before the one rounding; `act`
  (`none`, `gelu` exact) is then applied to the rounded value and rounded again, as
  `gemm_nt_bias`'s epilogue does. A linear over a `(width, channel)` row whose weight is stored
  channel-major is a `1 x width` convolution with `kh` 1: its `[N, cin * width]` weight is
  PyTorch's `[N, cin, 1, width]`.
- **`conv2d_unfold`** is `conv2d`'s receptive fields as rows, so a convolution with many channels
  is a GEMM. Row `(to, wo)` (output time row `to` in conv2d's numbering, output column `wo`)
  holds, for every input channel and kernel tap, the value conv2d multiplies that tap's weight by,
  and zero where the tap falls in the padding or on a row `valid` masks. Its columns are in the
  weight's order, `(ci, i, j)` for `[cout, cin, kh, kw]` and `(ci, j, i)` with `t_last`, so `y`
  times the weight read as `[cout, cin * kh * kw]` is conv2d's output (`[rows * Wo, cout]` is the
  same bytes as conv2d's `[rows, Wo * cout]`). Segments, `cu_div` and `valid` are conv2d's; `M`
  counts input time rows.
- **`seg_sinusoid`** is `rel_sinusoid`'s table at absolute positions that restart: row `r` of
  segment `s` (rows `[cu[s] / cu_div, cu[s+1] / cu_div)`) takes position `(r - cu[s] / cu_div) mod
  period`. Qwen3-ASR adds it to each second's rows, which start again at position 0. A row no
  segment covers is left untouched.
- **`attn_dense`'s `block` and `cu_div`.** `block` cuts a segment into blocks of that many rows from
  its start, the last one short, each attending only within itself: a chunked encoder's window
  (Qwen3-ASR's 104 rows, eight seconds). `cu_div` reads `cu_seqlens` in units that many rows
  apart, as the other encoder ops do, so a tower can bound its rows by the pass's frame
  boundaries. A kernel that reads neither refuses a call that sets them.

## Quantisation and rotation

| op | params | operands |
|---|---|---|
| `quant_act_i8` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale`, `asum`? |
| `quant_act_i8g` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale` |
| `had_quant_act_i8` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale` |
| `rmsnorm_had_quant_i8` | `M`(range) `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gated_had_quant_i8` | `M`(range) `n` `group` `dtype` `act`:str/opt `mode`/opt | `a`, `b`? → `q`, `scale` |
| `dequant` | `M`(range) `n` `group` `dtype` | `q`, `scale` → `y` |
| `gram_accum` | `M`(range) `n` `group` `dtype` | `a`, `a_scale`?, `h`:inout |

**`gram_accum`** characterises a quantiser's output:
`h[k1][k2] += sum over rows of dequant(a)[m][k1] * dequant(a)[m][k2]`, the Hessian a GPTQ packer
needs over the operand the consuming GEMM contracts. `h` is f32 `[n, n]` and inout: a caller
zeroes it once and drives a whole calibration corpus through the graph, each launch adding its
block. It is not `gemm_nt` with a transposed operand: that op contracts the last axis of both sides
and this one the first. It is the one op here that no forward pass issues; it runs under a
calibration flag and its result is read at convert time, in another process, through a file.
`dtype` says how `a` is read: `fp8a8` is E4M3 codes with an f32 dequant scale per (row, `group`
columns) in `a_scale`; `bf16` is a plain activation with no `a_scale` and no grouping.

**Activation widths.** Which width a linear's activation takes follows from the weight. Where the
checkpoint ships E4M3 weights and a block scale plane, every linear is `fp8a8`
(`arch/common/rad_fp8.h`). libr4d's `fp8a16` matvec (`M <= 8`) reads bf16 directly, but it assumes
a weight layout the stored WMMA fragment order does not provide, so it is not used. Where the engine
quantises the activation for an integer weight it is int8: `quant_act_i8` for `w4a8`, `w8a8` and
`w2a8`, `quant_act_i8g` for `i8a8`.

`fp8a8f32` is `fp8a8`'s E4M3 activation against an E4M3 weight whose 128x128 block scales are
f32, read as the checkpoint stores them. A scale that bf16 cannot represent exactly keeps its
value, which `fp8a8`'s bf16 scale plane would round. It applies to `gemm_nt_q` and `moe_gemm_q`
with `group` 128.

libr4d reads both canonical planes in place, with no relayout: `gemm_fp8a8_f32scale` for
`gemm_nt_q` and `moe_gemm_fp8a8_f32scale` for `moe_gemm_q`, `N` and `K` multiples of 128, E4M3
WMMA with the f32 scale product folded per 128 of K, bf16 out. The dense row picks a decode form
(row tiles sharing each weight fragment, several waves on consecutive K blocks) or a row-tile
form by `M`, `N` and `K`; on rows that are not 16-byte aligned it takes the 16x64 tile. The forms
are byte-identical (`tests/r4d_emu/test_fp8f32.cpp`). The grouped row takes equal-width parity
tables (`N_odd = N`, `K_odd = K`), token or sorted activations, and stacked planes or pointer
tables, and runs expert-major: a block per (column span, expert) takes every row of its expert,
each element keeping the 16x64 tile's sequence.

**`group` has different meanings in this table.** A kernel that reads it the other way is wrong,
not slow. `libref/ref_quant.cpp` is the definition.

- **`quant_act_i8`**: one f32 scale per row, `max(amax, 1e-8) / 127`, and codes `x * (1 / scale)`
  rounded to nearest even, stored in the step order the int8 WMMA reads an A fragment in: the
  middle two dwords of every 16-byte step exchanged (column `c` at byte `c` with bits 2 and 3
  swapped), so `n` is a multiple of 16. `group` sizes `asum`, each row's code sums per `group`
  columns, group-major `[n/group][ceil(M/16)*16]`, which an asymmetric weight grid needs to remove
  its zero point. Rows past `M` are not written.
- **The `had_*` int8 forms**: `group` is the Walsh–Hadamard rotation width (libr4d's `had`), and
  the int8 scale is per row, carrying the `1/sqrt(group)` normalisation. The rotation is what makes
  one scale per row viable, so its width and the scale granularity cannot be the same number.
- **The `had_*` fp8 forms** (`had_quant_act_fp8`, `gated_had_quant_fp8`): `group` is both. They
  serve a four-bit MoE expert whose scale grid is already one bf16 per (row, 128 of K); a rotation
  of the same 128 lines the two up, so the scale is per group, `[M, n/group]` as for
  `quant_act_fp8`. They apply no normalisation: `<Hw, Ha> = group * <w, a>`, and the whole factor is
  folded into the stored weight scale at convert, where dividing a bf16 by a power of two is exact.
  Splitting it as the int8 forms do would round `amax/7/sqrt(group)` into bf16, a ~0.2% error per
  group. `n` must be a multiple of `group`.
- **`quant_act_i8g`, `had_quant_act_i8g`, `dequant`**: `quant_act_fp8`'s grouping at int8 codes:
  one f32 scale per `group` contiguous elements of a row, so `scale` is `[M, ceil(n/group)]`
  (`group <= 0` or `group >= n` is one per row). The scale is `amax / 127` (1 for a zero group) and
  each code the value over it rounded to nearest even and clamped to ±127, row-major. They feed the
  int8 trunk GEMM (`gemm_nt_q` at `dtype=i8a8`), which folds a scale per K block. The `had_` form
  rotates first and, like `had_quant_act_fp8`, folds nothing: its consumer is a weight at
  `dtype=i8a8h`, whose scale carries the whole 1/`group`. It is not interchangeable with the
  per-row `had_*` int8 forms, which fold 1/sqrt(`group`) into the activation's scale. `dequant`
  inverts `quant_act_i8g`: the scale stream's extent gives the group count and `group` each group's
  width.

The rotation is unnormalised butterflies at ascending strides `1, 2, 4, ... group/2`, pairing `i`
with `i + h` where `(i & h) == 0`. The `1/sqrt(group)` is folded into the scale rather than applied
to the values; dividing every element by a constant cannot change which int8 code it rounds to. A
rotation block may not cross a tensor-parallel shard boundary, so the width must divide the
per-rank `n`.

`gated_had_quant_i8` takes the gate and up as two operands (libr4d's form), or one fused `gate_up`
`[M, 2n]` with the gate first as `a` and `b` absent. `mode` is libr4d's integer spelling of `act`
and is optional.

### The fp8 fusions

| op | params | operands |
|---|---|---|
| `quant_act_fp8` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `quant_cols_fp8` | `M` `n` `dtype` | `x` → `y`, `scale` |
| `had_quant_act_fp8` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `givens_dense_quant` | `M` `n` `group` `dtype` `parts` `out`:str | `x`, `tdense`w, `rscale`w → `q`, `scale` |
| `givens_rotate` | `M` `n` `group` `dtype` `stages` `parts` | `x`, `pairs`w, `angle`w, `rscale`w → `y` |
| `rmsnorm_quant_fp8` | `M` `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gated_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `swiglu_clamped_quant_fp8` | `M`(range) `n` `limit`:f64 `group` `dtype` | `gate_up` → `q`, `scale`, `out_bf16`? |
| `gemm_nt_q_gated` | `M` `N` `K` `group` `dtype` `act` | `a`, `a_scale`, `b`w, `b_scale`w → `q`, `scale`, `out_bf16`? |

`quant_cols_fp8` is the same E4M3 quantiser with one scale per column over all `M` rows,
`scale[c] = amax_r |x[r, c]| * (1/448)` (1 for an all-zero column), `scale` `[n]` f32: an
attention's values for its fp8 P.V form, where a column is one dimension of one head and its scale
multiplies out of the sum over keys.

`gemm_nt_q_gated` is `gemm_nt_q` over a `[N, K]` gate_up plane (`N` is the weight's rows, twice the
fused width, gate half first) with `gated_quant_fp8`'s pass in the epilogue, so the `[M, N]` bf16
tensor between them is never written. It is a separate schema, not a flag on `gemm_nt_q`, because
it does not produce that op's `y`. `act` is required here, unlike on the quantiser: an
implementation may compile one gate only, and an optional parameter would let it resolve for
another.

`swiglu_clamped_quant_fp8` (device-only) is `swiglu_clamped` then `quant_act_fp8` over its
product in one pass. The product is rounded to bf16 before the quantiser reads it, as the pair
reads it from its bf16 tensor, so the codes, scales and `out_bf16` are the pair's bytes, NaN and
infinite inputs included (`r4d_selftest`). `n` is a multiple of 4, `group` 128, `M` up to 65535,
rows pitched.

`out_bf16` is part of the function, not an optimisation: the unfused pair leaves a bf16 tensor
behind and takes the codes over it, so a layer with a bf16 consumer of the post-norm value must ask
for it. `residual` present is the `fused_add_rms_norm` form.

### ParoQuant rotations

A weight stored with the `givens` transform (`abi/rad_encoding.h`) is `v = w . D . G_1 ... G_S`: a
channel scale `D` and `S` stages of disjoint pairwise rotations applied to each row before
quantisation (ParoQuant, z-lab). The GEMM contracts `<v, x'>` with `x' = x . D^-1 . G_1 ... G_S`,
so the activation takes the same rotation once a token.

- **`givens_dense_quant`** computes `x / rscale`, the rotation, then a block-scaled quantiser per
  `group` columns that `out` names: `fp8` is `quant_act_fp8`'s (scale `amax / 448`), `i8` is
  `quant_act_i8g`'s (scale `amax / 127`, round to nearest). The rotation is one dense 128 x 128 f16
  matrix per (part, group), `tdense` `[parts * n, 128]`, composed from that group's `stages` rounds
  of `a' = a cos t + b sin t, b' = b cos t - a sin t`. Each pair lies inside one scale group, so the
  matrices belong to the weight: rad-convert composes them once in f64 and stores them as the
  weight's `t.dense` table, which the layout hook (`r4d_rad_layout_gdense`) only moves. Consumers
  are `gemm_nt_q` at `mxfp4a8` (`out` fp8) and at `w4a8i8_asym` / `w5a8i8` (`out` i8).
- **`givens_rotate`** is the same rotation stage by stage with no quantiser, `y` in its own dtype
  (f16), for the 16-bit-activation weights `w5a16` and `w4a16_asym`, whose GEMM serves any `M`
  (rows past 64 run as chunks of 64).

A fused projection (gate|up, qkv|z) was rotated per checkpoint tensor, so the activation has
`parts` rotated copies side by side: `q` is `[M, parts * n]` and `scale` `[M, parts * n / group]`;
`pairs` and `angle` stack each part's `stages` rows and `rscale` has one row per part. `gemm_nt_q`'s
`pb1` and `pb2` are the row bounds of the weight's parts: rows `[0, pb1)`, `[pb1, pb2)` and
`[pb2, N)` meet copies 0, 1 and 2 of the wide `a`/`a_scale`. `pb1` absent is one part; `pb2` absent
ends the second part at `N`. The `mxfp4a8` rows and the asymmetric 4- and 5-bit rows read them;
host rows that do not (libavx) refuse a call that names one rather than contract against the wrong
copy.

### Int8 and gated fp8 fusions (device-only)

| op | params | operands |
|---|---|---|
| `had_quant_act_i8g` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `gated_had_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `gated_had_quant_i8g` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `rmsnorm_had_quant_i8g` | `M` `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gate_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate`, `x` → `q`, `scale`, `out_bf16`? |
| `rmsnorm_quant_i8` | `M` `n` `eps`:f64 `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`?; `w_post`w?, `scalar`w? |
| `gelu_mul_quant_i8` | `M` `n` `dtype` | `gate_up` → `q`, `scale`, `out_bf16`? |

`gelu_mul_quant_i8` is `gelu_mul` with `rmsnorm_quant_i8`'s quantiser on its pass, byte-identical
to the pair.

`gated_had_quant_i8g` and `rmsnorm_had_quant_i8g` are `gated_quant_fp8` and `rmsnorm_quant_fp8`
with `had_quant_act_i8g`'s quantiser on their pass: the `group`-wide rotation of the bf16-rounded
value, then int8 codes per (row, `group`), byte-identical to the producer followed by
`had_quant_act_i8g`. They feed a weight at `i8a8h` without the activation quantiser of its own:
one launch for a norm read by three projections (q, k and v), and none between the gated product and
the down projection. `out_bf16` receives the unrotated value, for a bf16 consumer. `n` is a
multiple of `group`. A rotated code read as a plain one, or the reverse, gives a model that runs
and is wrong, so these are separate ops, as `gated_had_quant_fp8` is.

`rmsnorm_quant_i8` is `rmsnorm` with `quant_act_i8`'s quantiser on its pass: step-order int8 codes
and one f32 scale per row, what `gemm_nt_q` reads at `dtype=w4a8` and `w8a8`. With `residual` the
norm is of `residual + x`, stored back; `w_post` normalises `x` before that add and `scalar` (`[1]`)
multiplies the sum after it, together a sandwich-norm model's residual junction (`rmsnorm` → `add`
→ `scale` → `rmsnorm` → `quant_act_i8`). Byte-identical to that chain: every intermediate the chain
stores is rounded to bf16, and both sums of squares take `rmsnorm`'s partition.

### `dequant_w4` (device-only)

| op | params | operands |
|---|---|---|
| `dequant_w4` | `N` `K` `group` `dtype` `twos` | `wq`w, `ws`w → `y` |

libr4d's own op: a packed 4-bit fragment-ordered weight back to row-major bf16. libref declares
`dequant`, which is activation-shaped, and nothing else declares `dequant_w4`.

`twos` selects the symmetric two's-complement nibble over the asymmetric unsigned-nibble-minus-
zero-point grid. The two differ in bit 3, so the wrong one is wrong output, not a refusal.

`wq` and `ws` must be weight operands. As plain inputs they would name the same bytes, but the core
would stop resolving their handles and stop waiting on the mover's event, reading weight bytes that
may not have arrived.

## GEMM

| op | params | operands |
|---|---|---|
| `gemm_nt` | `M`(range) `N` `K` `dtype` | `a`, `b`w → `y` |
| `gemm_nt_f32` | `M`(range) `N` `K` `dtype` | `a`, `b`w → `y` |
| `gemm_nt_bias` | `M`(range) `N` `K` `dtype` `act`/opt | `a`, `b`w, `bias`w, `res`? → `y` |
| `gemm_nt_q` | `M`(range) `N` `K` `group` `dtype` `pb1`/opt `pb2`/opt `split_n`/opt | `a`, `a_scale`?, `b`w, `b_scale`w → `y`; `a_sum`?, `b_ref`w? |
| `gemm_nt_q_bias` | `M`(range) `N` `K` `group` `dtype` | `a`, `a_scale`, `b`w, `b_scale`w, `bias`w → `y` |

`nt` is A row-major `[M,K]` times B transposed `[N,K]`, giving `[M,N]`: the layout every weight in
a transformer already has. `gemm_nt_q`'s `dtype` names the operand combination, so one op covers
every quantised format and an architecture plugin that changes quantisation changes one string, not
its op names.

`gemm_nt_q` takes the activation scales as an operand because the quantiser produced them; a kernel
that folds dequantisation into its epilogue still receives them. Being a quantiser's amax over its
range, an `a_scale` entry is never negative, and a kernel may treat a block whose scale is zero as
contributing nothing (libr4d's MXFP4 GEMMs carry their accumulators in the last positive scale).
`a_scale` is absent for the f16-activation grids. At `w4a8`, `w4a8_asym`, `w8a8` and `w2a8`, `a` and `a_scale` are
`quant_act_i8`'s codes, in its step order, and per-row scale, and `a_sum` is its `asum`.
At `i8a8` they are `quant_act_i8g`'s row-major codes and per-(row, 128 of K) scales against an int8
weight with a bf16 scale per (row, 128 of K); at `mxfp4a8`, `quant_act_fp8`'s pair against e2m1
codes with an E8M0 exponent per (row, 32 of K).

`a_sum` removes an asymmetric grid's zero point and `b_ref` is the per-row reference exponent an
MXFP4 kernel folds its block exponents against. `b_ref` is derived from `b_scale` and does not
change the product. Both host libraries accept `b_ref` beside an e2m1 weight and ignore it, and
refuse `a_sum` (`RAD_E_UNSUPPORTED`) except at the grids that remove their zero through it.

The zero-point grids read unsigned codes as `b` and their scale and zero joined as `b_scale` (above):
weight element (n, k) is `(code - zero) * scale`, one f16 scale and one integer zero per (row,
`group` of K). libref's reference forms each group's sub-sum on the integer grid and scales it once:

| dtype | codes | `a`, `a_scale` | group sub-sum `sub` | accumulated |
|---|---|---|---|---|
| w2a8, w4a8_asym | u2, u4 | `quant_act_i8`'s step-order codes, per-row scale; `a_sum` required | `sum(a * code) - zero * a_sum[g][m]`, exact | `(sub * a_scale[m]) * scale` |
| w4a8i8_asym, w5a8i8 | u4, u5 | row-major int8, f32 scale per (row, 128 of K) (`givens_dense_quant`) | `sum(a * (code - zero))`, exact | `(sub * a_scale[m][g]) * scale` |
| w4a16_asym, w5a16 | u4, u5 | a 16-bit activation (`givens_rotate`), no `a_scale` | `sum(a * (code - zero))` in f32, ascending k | `sub * scale` |

`a_sum` is `quant_act_i8`'s `asum`, group-major `[K/group][Mpad]` with `Mpad = ceil(M/16)*16`. The
ParoQuant grids (the four 4- and 5-bit ones) take parts (`pb1`, `pb2`); `w2a8` and `w4a8_asym` do
not, and none has a bias form.

`split_n` is the output width whose K split the call takes, when it is a row slice of a wider
linear (a rank's share of a weight whose rows do not divide among the ranks, issued as two slices).
libr4d's narrow fp8a8 and i8a8 kernels pick their K split from N alone, and the split sets the order
K partials are summed in, so a slice given the whole width sums each output as the whole linear
does and its outputs are the linear's to the bit. Absent, N. The host libraries never split K and
ignore it. libr4d's kernels compute the same products in another order (split-K
partials, the activation scale applied at the store); libavx folds the scales into its decoded rows,
each product rounding once where libref rounds each group's sub-sum. A `b_scale` that is not joined
at these grids has no zero to read and is refused (`RAD_E_DTYPE`). libref describes none of them
(`opd_shape` declines: no single tensor states a scale and a zero), so rad-kbench and the op oracle
leave them out and rad-avx-check builds them by hand.

`gemm_nt_bias`'s epilogue is a linear layer's: the biased product rounded to `y`'s dtype, then
`act` of it (`none`, `gelu` exact, `gelu_tanh`) rounded, then `res` added and rounded. Each step is
what a separate tensor would hold, so the fused form equals the unfused chain bit for bit. `res`
may be `y` itself, the residual stream updated in place (the encoder towers' layers do this), so an
implementation reads `res` before it stores the product over it.

`gemm_nt_f32` is `y[M, N] = a[M, K] @ b[N, K]^T` with a bf16 activation, an f32 weight, f32
accumulation and an f32 `y`; `dtype` (`bf16`) names the activation. It exists for the projections
whose weights and results must not narrow to bf16: GLM's router logits and its indexer's signed
head gates.

libr4d's `gemm_nt_f32_bf16` takes `N` up to 512, `K` up to 16384, `M` up to 65535 and pitched
rows. A wave computes a tile of rows by columns, chosen by `M` and `N` (a single column per wave
at one row, eight rows by four columns from 64 rows, four rows by one column at `N <= 8`); each
output keeps one order (lane `l` sums `k = l, l+32, ...` with fused multiply-add, then a fixed
butterfly), so every tile is byte-identical (`tests/r4d_emu/test_f32.cpp`).

### Runtime LoRA: `lora_shrink` and `lora_expand`

| op | params | operands |
|---|---|---|
| `lora_shrink` | `M`(range) `K` `R` `S` `group`/opt `dtype` | `a`, `a_scale`?, `slot`, `lora_a` → `t` |
| `lora_expand` | `M`(range) `N` `R` `S` `dtype` | `t`, `slot`, `lora_b`, `scale`, `y`:inout |
| `lora_expand_mm` | `M`(range) `N` `R` `S` `dtype` | `t`, `slot`, `lora_b`, `scale`, `y`:inout |

The two halves of `y += scale ⊙ (B · (A · x))` at a trunk GEMM, issued by the core right after the
GEMM in the GEMM's own row slice, with several adapters and no adapter mixed in one step; a step
with no adapter row issues neither. `slot` is
an i32 per row naming its adapter's pool slot -- the step batch's `RadBatch::lora_slot` from the
GEMM's first row on; **a row with `slot < 0 || slot >= S` is a base row and neither op touches
it**: `lora_shrink` writes nothing into its `t` row, and `lora_expand` does not read or write its
`y` row at all — so a base row stays bit for bit what the GEMM wrote, where `y + 0` would already
have turned a -0 into +0. On an image or video generator's pass every row runs the same adapter,
and `slot` is a run of the core's slot-row table holding that slot's index in every element, so the
GEMM's rows need not be the pass's token rows.

- `lora_shrink`: `t[m, j] = Σ_k x(m, k) · lora_a[s, j, k]`, `s = slot[m]`, f32 accumulate.
  `a`/`a_scale` are the GEMM's own first two operands, and `x(m, k)` is the value `gemm_nt_q` uses
  for that element: the code times its group's scale, read through the same scale-stream rule
  (`[rows, groups]` or `[rows]`, the group derived from the stream), or a `bf16`/`f16`/`f32`
  element as it stands. An `fp8_e4m3`/`fp8_e5m2`/`i8` `a` without `a_scale` (null data) is refused.
  `group` is a cross-check on the stream's grid -- `ceil(K / group)` groups along K -- and 0 means
  none; `dtype` is `a`'s tensor dtype name
  (`rad_dtype_name`: `bf16`, `f16`, `f32`, `fp8_e4m3`, `i8`, ...), not a combined quant string.
  `lora_a` is `[S, R, K]` bf16, `t` is `[M, R]` f32.
- `lora_expand`: `y[m, n] = y[m, n] + scale[s, n] · Σ_j t[m, j] · lora_b[s, n, j]`, f32
  accumulate, one rounding to `y`'s dtype (`bf16`/`f16`/`f32`, named by `dtype`). `lora_b` is
  `[S, N, R]` bf16 and `scale` `[S, N]` f32 — per output column, so a fused projection whose parts
  carry different `alpha / r` is one launch.

- `lora_expand_mm`: `lora_expand` with `t` rounded to bf16 and the sum over `j` in any order,
  `y[m, n] = y[m, n] + scale[s, n] · Σ_j bf16(t[m, j]) · lora_b[s, n, j]`, f32 accumulate, one
  rounding to `y`; base rows as `lora_expand`. A row's bytes may depend on `M`. The core declares
  it in place of `lora_expand` for an image or video generator, whose passes are thousands of rows
  of one adapter and where the exact row's scalar sum is most of the adapter's cost; libr4d runs
  it on the WMMA over 64 x 64 tiles.

`a`, `a_scale` and `y` may be row and column slices of wider buffers; every row stride is the
tensor's. The pools are ordinary inputs and not weights: the engine owns them as program buffers
and copies an adapter into a slot between steps, so a recorded pass replays the same pointers and
reads whatever adapters the slots hold. A row's result does not depend on `M` or on any other row
(no split of a reduction that moves with the batch), which is what lets a request's bytes be the
same alone and in a mixed batch. Implementations: libref, libavx, libr4d (scalar f32 rows, and
tiled rows over row tiles that outrank them: a WMMA shrink whose summation order is fixed by `K`
and `R`, and an expand that stages `lora_b[s]` in LDS for a tile of rows and computes libref's
bytes).

## Attention

| op | params | operands |
|---|---|---|
| `attn_paged` | `q_len`(range) `head_dim` `gqa` `block_size`/der `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out` |
| `attn_dense` | `M`(range) `head_dim` `gqa` `causal` `dtype` `scale`:f64/opt `max_seqlen`/opt `window`/opt `cu_div`/opt `block`/opt `window_right`/opt | `q`, `k`, `v`, `cu_seqlens` → `out` |
| `attn_cross` | `M`(range) `head_dim` `gqa` `causal` `dtype` `scale`:f64/opt | `q`, `k`, `v`, `cu_seqlens`, `cu_k` → `out` |
| `attn_na3d` | `M`(range) `head_dim` `kt` `kh` `kw` `dtype` `scale`:f64/opt | `q`, `k`, `v`, `geom` → `out` |
| `attn_local_rel` | `M`(range) `head_dim` `n_head` `left` `right` `softcap`:f64 `cu_div`/opt `dtype` | `q`, `k`, `v`, `relk`, `pds`w, `cu_seqlens` → `out` |
| `kv_store` | `M`(range) `head_dim` `n_head_kv` `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `k`, `v`, `slot_mapping`, `kv_cache`:inout |
| `rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `scale`:f64 `mode` `rotary_dim`/opt `sections`/opt `yarn_factor`:f64/opt `yarn_low`:f64/opt `yarn_high`:f64/opt `yarn_magnitude`:f64/opt `ladder`/opt `llama3_factor`:f64/opt `llama3_low`:f64/opt `llama3_high`:f64/opt `llama3_orig`:f64/opt | `qkv`:inout, `positions` |
| `rope_table` | `M`(range) `rot` `theta`:f64 `scale`:f64 `dtype` `yarn_factor`:f64/opt `yarn_low`:f64/opt `yarn_high`:f64/opt `yarn_magnitude`:f64/opt `head_dim`/opt `ladder`/opt | `positions` → `cos_sin` |
| `qk_norm_rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `eps`:f64 `scale`:f64/opt `mode`/opt `rotary_dim`/opt `wadd`:f64/opt `v_norm`/opt `sections`/opt `comps`/opt | `qkv`:inout, `positions`?, `q_w`w?, `k_w`w? |
| `qkv_norm_rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `eps`:f64 `scale`:f64/opt `mode`/opt `rotary_dim`/opt `ladder`/opt `wadd`:f64/opt `v_from_k`/opt `v_norm`/opt `dtype` | `q`:inout, `k`:inout, `v`:inout, `positions`, `q_w`w, `k_w`w |
| `qkv_norm_rope_kv_store` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `eps`:f64 `scale`:f64/opt `mode`/opt `rotary_dim`/opt `ladder`/opt `wadd`:f64/opt `v_from_k`/opt `dtype` `v_norm`/opt `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `q`:inout, `k`:inout, `v`:inout, `positions`, `q_w`w, `k_w`w, `slot_mapping`, `kv_cache`:inout |

**`attn_paged`**'s band is `q_len`, not `M`: one query row at decode, `1 + n_spec` at a speculative
verify, thousands in a prefill chunk, and libr4d has a different kernel for each band.
`block_size` is read off the resolved kernel, not chosen: declare the group, declare the op, then
ask `rad_kv_block_size()` (`spec.md` §7.2).

`cu_seqlens` is optional and makes the batch ragged: `[n_seq+1]` i32, sequence `s` owning query rows
`[cu[s], cu[s+1])` of the dense `q` and `out` blocks. Absent, every sequence has the same query
length and the total query rows must be a multiple of `n_seq`. Pass it: the scheduler builds mixed
steps (one sequence's prefill chunk beside other sequences' decode rows, each with its own verify
window) whenever prefill and decode coexist, and those are not uniform. `q_len` is then the batch
maximum, which selects the band; the per-sequence lengths come from `cu_seqlens` alone.

**`attn_dense`** serves vision towers and encoders: nothing paged, `cu_seqlens` bounding one image
or one input per entry so a whole batch is one launch. `window` is `attn_paged`'s rule over a
segment's own positions (causal: key `j` is visible to query `i` when `i - j < window`; non-causal:
when `|i - j| < window`), and 0 is no window. A bidirectional encoder's sliding layer of radius `r`
(`|i - j| <= r`) is `window = r + 1`; one that sees `l` keys behind and `r` ahead is `window = l + 1`
with `window_right = r + 1` (non-causal only: key `j` visible when `i - j < window` and
`j - i < window_right`). `q`, `k` and `v` may each be `[tokens, heads, head_dim]` or
`[tokens, heads * head_dim]`, and the second form may be a column slice of a fused projection read
at its row pitch, so a tower's `[q|k|v]` row needs no copy.

**`attn_cross`** is `attn_dense` with each entry's keys a run of their own: entry `s`'s queries
are rows `[cu_seqlens[s], cu_seqlens[s+1])` of `q`, its keys and values rows `[cu_k[s], cu_k[s+1])`
of `k` and `v`, and with `causal` a query sees key `j` of its run when `j <= i`, both counted from
their run's start. A query that sees no key is zero. It is a video's rows reading their prompt
(cross attention), and a padded prompt whose padding rows see only the prompt (`cu_k` the
prompt's rows, `cu_seqlens` the prompt and its padding: every padding row sees every prompt row
and none of the padding, as a masked padding token does in the reference).

**`attn_na3d`** is 3-D neighborhood attention, NATTEN's `na3d` (LTX-2.5's diffusion decoder). The
rows of `q`, `k`, `v` and `out` are the voxels of the volume `geom` = `(T, H, W)` (i32 on the
device), `(t, i, j)` row-major, each `[rows, heads head_dim]` at its own row pitch (column slices of
a fused projection). A query sees the keys of a `kt x kh x kw` window, each extent at most its axis,
centred on it and shifted inward at the border: along an axis of `L` and window `n` it starts at
`clamp(c - n / 2, 0, L - n)`, so it always holds `n` keys. Per head, in f32: each score the dot over
`d` ascending times `scale` (default `1 / sqrt(head_dim)`), the window's keys in raster order, the
softmax as `attn_cross` takes it, the output rounded to its dtype. Rows past `T H W` are not
written. libr4d serves `head_dim` 64.

**`attn_local_rel`** is Gemma 4's audio attention: local, with a relative-position term and a
softcap. Query row `i` of a segment sees key rows `j` of the same segment with
`0 <= i - j <= left` (`right` must be 0). Per head, `qf = q * (head_dim^-0.5 / ln 2) * softplus(pds)`
(softplus rounded to `pds`'s dtype, as the reference evaluates it on the parameter) and
`logit = qf . k * (ln(1 + e) / ln 2) + qf . relk[i - j]`, then `softcap * tanh(logit / softcap)`, a
softmax over the visible keys and the weighted sum of `v`, all in f32 and rounded once. `relk` is
`[left + 1, n_head * head_dim]`, row `d` the projected position for distance `d`. The upstream
reference computes this in 12-query blocks over 24-key windows with a Transformer-XL shift, giving
every key outside the window a `-1e9` logit, so the sliding form is the same function.

**`rope`'s multi-component modes.** `mrope`, `imrope` and `axial` rotate with NeoX pairing and let
several position components drive different frequencies. `positions` is then `[C, M]`,
component-major, and `sections` names how many of the `rotary_dim / 2` frequencies each component
owns:

- `mrope` gives contiguous runs of them to components 0, 1, ... (Qwen2-VL);
- `imrope` interleaves: component 1 takes frequency `i` where `i % 3 == 1` and `i < 3 * s1`,
  component 2 where `i % 3 == 2` and `i < 3 * s2`, component 0 the rest (Qwen3-VL);
- `axial` gives contiguous runs and restarts the frequency ladder in each, `theta^(-2j / (2 s_c))`
  for the run's `j`-th frequency (a vision tower's 2-D rotary over row and column).

With `[M]` positions every component is that one position, so `mrope` and `imrope` over a text batch
are exactly `neox`.

**Positions** are integers or floats, and the operand's dtype says which. An integer `positions`
(i32) is read as an integer; an f32 one (bf16 and f16 too) as the value it holds, a fractional
coordinate such as a video decoder's normalised grid already multiplied by `2 pi`. Either way the
angle is `f32(pos / scale) * inv` in f32. `[C, M]` positions are read at the operand's own strides,
so a `[C, M]` view of a wider `[C, max]` plane is read in place.

**`llama3_*`** applies Llama 3's frequency scaling (transformers' `rope_type` `llama3`) to `rope`'s
ladder: with wavelength `2 pi / inv`, a frequency shorter than `llama3_orig / llama3_high` is kept,
one longer than `llama3_orig / llama3_low` is divided by `llama3_factor`, and those between blend,
`s = (llama3_orig / wavelength - llama3_low) / (llama3_high - llama3_low)`,
`(1 - s) inv / llama3_factor + s inv`, in f32 (`abi/rad_yarn.h`, `RadLlama3`). It does not combine
with YaRN or a multi-component mode.

**`ladder`** names the width the base frequency ladder is measured in. Absent or `rotary`, the
exponent divides by `rotary_dim`: a partial rotary is a complete rotation of its own width. `head`
divides by `head_dim` instead (the proportional variant), so the `rotary_dim / 2` frequencies are a
prefix of the full-width table: at `head_dim 512, rotary_dim 128`, frequency `i = 1` is
`theta^(-1 / 256)`, not `theta^(-1 / 64)`. Only the base ladder moves (`axial` keeps its per-section
ladders), and the head past `rotary_dim` is untouched either way. `rope_table` takes `head_dim` for
the same reason: its angles must stay `rope`'s exactly.

**`qk_norm_rope`** RMS-norms each q and k head of a fused `[q | k | v]` row against its gain (plus
`wadd`), rounds, and rotates in place. `positions` is read as `rope` reads it, integer or f32.

- An absent `q_w` or `k_w` is a unit gain, without `wadd`: the head is `x rsqrt(mean(x^2) + eps)`,
  rounded (a video decoder's weightless q and k norms).
- `rotary_dim` 0 is the norm alone: nothing rotates, and `positions` may be absent (`M` is then
  `qkv`'s rows). A rotation with no positions is refused.
- `v_norm` 1 also RMS-norms every v head with a unit gain and does not rotate it: Gemma 4's value
  norm, which has no weight.
- `mode` `blocks` rotates each section as its own block of channels: section `c` is `sections[c]`
  pairs over the next `2 * sections[c]` channels, paired `(start + j, start + sections[c] + j)`
  (NeoX inside the block), turned by position component `comps[c]` (default `c`) at
  `theta^(-j / sections[c])`. The sections fill `rotary_dim` exactly and `positions` is `[C, M]`.
  This is Gemma 4's vision rotary: x over the first half of each head and y over the second, each a
  complete rotary; `rope`'s `axial` pairs across the halves instead.
- `mode` `gptj_blocks` is `blocks` with each section's pairs adjacent, `(start + 2j, start + 2j + 1)`:
  a 3-axis image rotary's complex form.
- `mode` `imrope` is `rope`'s `imrope` over NeoX pairs, reading three position planes (Qwen3-VL's
  text attention).
- `mode` `axial` is `rope`'s `axial` over NeoX pairs `(i, i + rotary_dim / 2)`: the
  `rotary_dim / 2` frequencies split into contiguous `sections` (summing to `rotary_dim / 2`),
  section `c`'s `j`-th frequency `theta^(-2j / (2 sections[c]))` turned by position component `c`.
  `positions` is `[C, M]` with `C` at least the section count, or `[M]`, every component then that
  one position on the same per-section ladders (unlike `rope`, whose `[M]` positions fall back to
  the plain ladder). MiniMax-H3's DiT rotates 96 of its 128 channels with sections `16 16 16`
  (theta 1e4) over f32 `(t, h, w)` positions; its video decoder 48 of 64 with `8 8 8` (theta 100)
  over positions `grid_pos3` writes.

**`qkv_norm_rope`** (device-only) is the same prologue over separate `q`, `k` and `v` projections
(Gemma 4's text attention, three GEMMs): the q and k norms, the unit-gain value norm and the rotary
in one dispatch. `v_from_k` 1 takes v from the raw k before the k norm (the k == v global class);
`v_norm` 0 leaves v unnormed (a Qwen3-shaped layer: two norms and two rotaries). `n_head` 0 is the
k (and v) heads alone, `q` and `q_w` passed but neither read nor written: a pass that writes only
keys and values (a block drafter's context pass).
The rotary parameters are `rope`'s, `ladder` included. Byte-identical to `rmsnorm_p`, two
`rmsnorm`s and two `rope`s (one each at `n_head` 0).

**`qkv_norm_rope_kv_store`** (device-only) is `qkv_norm_rope` followed by `kv_store` in one launch:
the block of each k head (and of each v head, or under `v_from_k` the k head's block for both) ends
holding its token's finished head, so it also writes it into the paged cache at the token's slot.
Its parameters are the two ops' own, its operands the prologue's followed by `slot_mapping` and
`kv_cache`. Byte-identical to the pair at either cache width: the store converts each element as
`kv_store` does, and q, k and v are still written in place as the prologue leaves them.

| op | params | operands |
|---|---|---|
| `attn_paged_gate_quant` | `q_len`(range) `head_dim` `gqa` `block_size` `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt:capacity `group` `act` | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out`; `gate` → `q8`, `q8_scale` |

**`attn_paged_gate_quant`** (device-only) is `attn_paged` followed by `gate_quant_fp8` over its
output, finished inside the split-KV merge. A merge row is one (token, head) pair of `head_dim`
columns, which is a `gate_quant_fp8` row, so the merged values are gated (`bf16(sigmoid(gate))`,
rounded before the multiply), rounded, written to `out` and quantised to E4M3 with one scale per 128
columns while still in registers. `r4d_selftest` checks it byte for byte against the pair at both
merge forms and over a bf16 and an E4M3 cache. `block_size` is required rather than derived because
the cache's page size is `attn_paged`'s to answer; the caller passes the value it read. The gated
attention block uses it on the QSA sparse path, where every launch is one query row per sequence
and the decode kernel serves the whole step in one launch. `max_seqs` is a capacity
(`RAD_PROLE_CAPACITY`): the block declares it at the step size, and the activation arena's smaller
declares bound it lower.

### The KV cache operand

`attn_paged` and `kv_store` take the group's cache as operand 1, passed as `RAD_KV(group, layer)`.
`RadKVGroupBatch` carries the slot mapping, the block table and the sequence lengths, but the pool
base belongs to the block manager, not the batch.

**`kv_dtype` `fp8_v4` and `k6_v4`** (device-only) are the quantised layouts of
`--kv-cache-dtype` (`RadBuildCtx::kv_quant`). A head's dims are grouped by 32, each group with an
f16 scale (its absmax over the largest code, clamped so every code decodes finite in f16),
codes two's complement. A token's row in a head, at
head_dim HD:

| bytes | `fp8_v4` (25 HD / 16) | `k6_v4` (22 HD / 16) |
|---|---|---|
| K | [0, HD): E4M3, as `fp8_e4m3` | [0, HD/2): the codes' low four bits; [HD/2, 3HD/4): their high two bits, a word per 16 dims whose byte j holds dims 2j, 2j+1 in bits 0-3 and 8+2j, 9+2j in bits 4-7 |
| V | [HD, 3HD/2): int4 codes, dim d in the nibble at bit 4 (d mod 2) of byte d/2 | [3HD/4, 5HD/4): the same |
| scales | [3HD/2, 25HD/16): V's (absmax / 7) | [5HD/4, 21HD/16): K's (absmax / 31); [21HD/16, 22HD/16): V's |

The group is declared at `RAD_F8E4M3` with head_dim the row's bytes / 2, so the core sizes pages
as it does for fp8, and the cache operand's last extent is the row. The ops take the model's
head_dim. libr4d's rows are `kv_store_fp8v4` / `kv_store_k6v4`,
`qk_norm_rope_gate_kv_store_fp8v4` / `_k6v4` (byte-identical to the prologue and the store, as at
the other widths), `attn_prefill_h256_gqa6_{fp8v4,k6v4}` and `attn_decode_h256_gqa6_{fp8v4,k6v4}`,
which widen the codes to f16 in a tile's staging, and the same kernels at Qwen3.8-Flash-Next's
ratios over 4-token pages: `attn_{prefill,decode,decode_gq}_h256_gqa12_{fp8v4,k6v4}` and
`attn_{prefill,decode,decode_gq}_h256_gqa6_bs4_{fp8v4,k6v4}`, the `decode_gq` rows with the output
gate and quantiser in the split-KV merge. `r4d_selftest` holds the stores to an exactly
representable set and the attention rows to its inline oracle.

### `qk_norm_rope_gate` (device-only)

| op | params | operands |
|---|---|---|
| `qk_norm_rope_gate` | `M` `head_dim` `n_head` `n_head_kv` `rot` `q_dtype` `eps`:f64/opt `wadd`:f64/opt `cs_f32`/opt `pos_i64`/opt | `q_gate`?, `k`, `cos_sin`, `positions`, `q_w`w?, `k_w`w → `q_out`?, `k_out`, `gate_out`? |

It splits an interleaved `[q|gate]` projection per head, RMS-norms q and k over the head, applies
partial NeoX rope and copies the gate: four launches a layer in one. It is not `qk_norm_rope`, which
takes `theta`: this op takes a precomputed `cos_sin` table (`rope_table` builds it) and a different
operand list. `rot` is the rotary width, `cs_f32` and `pos_i64` state the widths of the table and
the position vector, and `gate_out` is optional for a caller that reads the gate at its column
offset inside the projection.

**`n_head` 0 is the K-only form.** The `k` side reads one head per `head_dim` columns at the
operand's own row pitch and writes contiguously to `k_out`; only the q side adds the `[q|gate]`
split. A plain input asks for `n_head` 0 and passes no `q_gate`, `q_w` or `q_out`. The QSA
indexer's query heads have that shape: `heads` of them inside a `heads + 1` projection, a stride no
rank-2 view can express, which would otherwise take `heads` `rmsnorm` launches plus a `rope`.

| op | params | operands |
|---|---|---|
| `qk_norm_rope_gate_kv_store` | `M` `head_dim` `n_head` `n_head_kv` `rot` `q_dtype` `eps`:f64/opt `wadd`:f64/opt `cs_f32`/opt `pos_i64`/opt `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `q_gate`?, `k`, `cos_sin`, `positions`, `q_w`w?, `k_w`w → `q_out`?, `k_out`, `gate_out`?; `v`, `slot_mapping`, `kv_cache`:inout |

**`qk_norm_rope_gate_kv_store`** is the prologue followed by `kv_store` in one launch: each k head's
block already holds its token's rotated k one element a lane, so it also writes that element and
the head's v element into the cache at the token's slot. The store reads `k_out`, so the tail adds
only `v`, `slot_mapping` and `kv_cache`; every parameter is one of the two ops' own. It is
byte-identical to the pair at either cache width: the store converts the rotated k exactly as
`k_out` stores it, a bf16 cache takes the values as they are, and the E4M3 conversion rounds each
element on its own.

Both ops are byte-identical to the kernels they replace, checked in `r4d_selftest` for the
`[q|gate]` form and the K-only form on a strided projection. The check is byte equality, not a
tolerance: a sum of squares folded in another order differs by an ULP, which a relative bound
accepts, but a rotated element is a difference of two products, so where those nearly cancel the
ULP becomes a few percent and the transcript moves within a few dozen steps. The kernel reproduces
the reference's fold exactly: widths above the head dim contribute zeros, and adding 0.0f is exact.

## Image generation

The ops a diffusion image generator declares (`rad_declare_generator`, `abi/rad_builder.h`): a
single-stream diffusion transformer's adaptive norm and gated residual, the timestep embedding,
per-image latent lanes and the flow-matching sampler over them, attention over a cached
conditioning prefix, and an image decoder's convolution, channel norm, resampling shortcuts and
pixel output. Each rounds where the reference model materialises a bf16 tensor; the libref function
says which (`libref/ref_gen.cpp`).

| op | params | operands |
|---|---|---|
| `ln_mod` | `M`(range) `n` `eps`:f64 `dtype` | `x`, `s`, `cu_seqlens` → `y` |
| `gate_add` | `M`(range) `n` `dtype` `act`/opt | `y`, `g`, `cu_seqlens`, `x`:inout |
| `ln_mod_quant_fp8` | `M`(range) `n` `eps`:f64 `group` `gated` `dtype` | `y`?, `g`?, `s`, `cu_seqlens`, `x`:inout → `q`, `scale`, `z`? |
| `timestep_embed` | `M`(range) `n` `scale`:f64 `max_period`:f64 `round`/opt `dtype` | `t` → `y` |
| `gather_rows2` | `M`(range) `n` `dtype` | `a`, `b`, `idx` → `y` |
| `table_add` | `M`(range) `n` `L` `dtype` | `x`, `t`w → `y` |
| `mix_rows` | `M`(range) `n` `k` `dtype` | `x`, `w`w → `y` |
| `chan_stats` | `M`(range) `n` `mode` `var`/opt `eps`:f64/opt `dtype` | `x`, `mean`w, `s`w → `y` |
| `lane_gather` | `M`(range) `n` `stride` `dtype` | `src`, `lane`, `cu_seqlens` → `y` |
| `lane_store` | `M`(range) `n` `stride` `dtype` | `src`, `lane`, `dst`:inout |
| `randn_lanes` | `M`(range) `n` `stride` `dtype` | `flags`, `seed`, `lane`, `cu_seqlens`, `dst`:inout |
| `flow_euler` | `M`(range) `n` `stride` `dtype` | `v`, `sigma`, `sigma_next`, `cfg`, `pair`, `lane`, `flags`, `seed`, `cu_seqlens`, `inp_lat`?, `inp_mask`?, `state`:inout, `preview`:inout? |
| `kd_input` | `M`(range) `n` `stride` `dtype` | `sigma`, `pair`, `lane`, `flags`, `seed`, `cu_seqlens`, `state`:inout → `y` |
| `kd_step` | `M`(range) `n` `stride` `pred` `dtype` | `out`, `sigma`, `sigma_next`, `sigma_prev`, `cfg`, `pair`, `lane`, `flags`, `seed`, `sampler`, `step`, `cu_seqlens`, `state`:inout, `hist`:inout, `preview`:inout? |
| `chan_affine` | `M`(range) `n` `mul` `add` `mode` `dtype` `round`/opt | `x` → `y` |
| `conv2d_nhwc` | `M`(range) `cin` `cout` `k` `stride` `pad` `up` `dtype` | `x`, `w`w, `b`w → `y` |
| `im2col_nhwc` | `M`(range) `cin` `k` `stride` `pad` `up` `dtype` | `x` → `y` |
| `l2norm_chan` | `M`(range) `n` `act` `dtype` | `x`, `gamma`w → `y` |
| `d2s_add` | `M`(range) `cin` `cout` `ft` `dtype` | `x`, `y`:inout |
| `s2d_avg_add` | `M`(range) `cin` `cout` `ft` `dtype` | `x`, `y`:inout |
| `to_rgba8` | `M`(range) `n` `dtype` | `x` → `y` |
| `gelu_tanh` | `M`(range) `n` `dtype` | `x` → `y` |
| `scatter_add_rows` | `M`(range) `n` `dtype` | `v`, `idx`, `x`:inout |
| `attn_prefix_self` | `q_len` `head_dim` `gqa` `block_size`/der `causal` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt `qk`/opt `pv`/opt | `q`, `k`, `v`, `kv_cache`?, `block_table`?, `seqused`?, `cu_seqlens`, `k_descale`?, `v_descale`?, `k_i8`?, `k_scale`? → `out`, `v_scale`? |

**Sequences and lanes.** `cu_seqlens` partitions the rows into entries (`[n_seq+1]` i32, entry `e`
owning `[cu[e], cu[e+1])`; it need not start at 0). `ln_mod` and `gate_add` take each row's
modulation row from `s`/`g`, `[n_seq, >= n]` at its own row pitch (a column slice of one shared
modulation projection). A lane is one image's latent in a per-image state buffer, `stride` rows
apart: `lane_gather` copies entry `e`'s rows out of lane `lane[e]`, and `lane_store` copies one
source into every listed lane (a repainted picture's latent into each of its images).

- `ln_mod`: `y = bf16(bf16(LN(x)) * bf16(1 + s[seq]))`, a layer norm with no gain and no bias.
- `gate_add`: `x = bf16(x + bf16(bf16(tanh(g[seq])) * y))`. `act` names the gate's activation,
  `tanh` (the default) or `none`, the gate as it is and not rounded first:
  `x = bf16(x + bf16(g[seq] * y))` (MiniMax-H3's last block, whose gates have no tanh). The
  roundings are to `x`'s dtype.
- `ln_mod_quant_fp8`: with `gated` 1, `gate_add` first; then `z = ln_mod(x)` and
  `(q, scale) = quant_act_fp8(z, group)`, `z` written when bound. Each value is rounded as its own
  store would round it, so the bytes are the three ops'.
- `timestep_embed`: `[cos | sin]` of `t' * f_i`, `f_i = exp(-ln(max_period) * i / (n/2))` in f32,
  `t' = scale * t`; with `round` 1, `t' = scale * bf16(bf16(t * scale) / scale)`, the value
  diffusers' pipeline hands its transformer; with `round` 2, `t' = scale * bf16(t)` (a pipeline
  that hands the transformer `t` in bf16 and scales it inside, Krea 2's); with `round` 3,
  `t' = bf16(scale * bf16(t))` (the scaling also in bf16, FLUX's). `t` is f32 `[M]`.
- `gather_rows2`: `y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]]`, zeros past either source.
- `table_add`: `y[r, l n + c] = x[r, c] + t[l, c]` for each of `L` rows of the weight `t` `[L, n]`,
  the sum f32 and rounded once: a modulation shared by every block plus each block's own learned
  table (Krea 2, LTX-2), every block's rows in one launch, read at `l n` of each `y` row.
- `chan_stats`: a latent through per-channel statistics a VAE keeps as weights, `mean` and `s`
  `[n]`: `std` is `s`, or with `var` 1 `sqrt(s + eps)` (a BatchNorm's running variance, FLUX's VAE),
  each statistic rounded to `y`'s dtype as the reference casts it; `mode` `denorm` is
  `y = rd(rd(x std) + mean)` and `norm` is `y = rd(rd(x - mean) / std)`.
- `mix_rows`: `y[r, c] = sum_i w[i] x[r k + i, c]` over `i < k`, f32 in ascending `i` and rounded
  once: a `Linear(k, 1)` across `k` consecutive rows (Krea 2's text fusion folds a token's twelve
  tapped encoder layers, laid out as consecutive rows, into one).
- `randn_lanes`: for entries whose `flags` carry 1, the lane's rows become the entry's seed's
  standard-normal stream (`abi/rad_philox.h`), element `(k, c)` the stream's `k * n + c`. `seed` is
  `[n_seq, 2]` u32, low word first.
- `flow_euler`: for every entry with `pair != -2` (an entry that is a negative conditioning moves
  no lane), `v' = pair >= 0 ? bf16(neg + bf16(cfg * bf16(v - neg))) : v`, `neg` the paired entry's
  rows; flag 4 writes `bf16(x - sigma * v')` to the lane's `preview`; then
  `x' = bf16(x + bf16(bf16(sigma_next - sigma) * v'))`; flag 2 restores the kept region,
  `x' = bf16(m x' + (1 - m)((1 - sigma_next) src + sigma_next noise))` with `m` the lane's f32
  `inp_mask` row, `src` its `inp_lat` row and `noise` the seed's stream.

**The k-diffusion sampler.** A denoiser trained on discrete timesteps of a variance-preserving
schedule (SDXL; eps- or v-prediction) is sampled as k-diffusion samples it, in the
variance-exploding form `x = x0 + sigma eps`: the lane holds `x` in f32 (`state` and `hist` are f32
`[lanes * stride, n]`), the model reads `c_in x` with `c_in = 1 / sqrt(sigma^2 + 1)`, and its output
`out` (`dtype`, in the pass's compact image order bounded by `cu_seqlens`) becomes the clean
estimate. Entries, pairs, lanes, flags and seeds are `flow_euler`'s; `sigma`, `sigma_next`,
`sigma_prev` and `cfg` are f32 `[n_seq]`, `sampler` and `step` i32 `[n_seq]`. Every product and sum
is one f32 rounding; `log`, `expm1` and `sqrt` are the library's. A repaint (flag 2) is refused.

- `kd_input`: for every entry with `pair != -2`, a first step (flag 1) sets the lane to the seed's
  normal stream times `sqrt(1 + sigma^2)` (`rad_philox_normal(seed, k n + c)`, the start of a
  schedule at the model's largest sigma); then `y = rd(x / sqrt(sigma^2 + 1))` at the entry's rows
  and, when `pair >= 0`, at the pair's rows too, so the negative entry reads the same latent without
  a second writer.
- `kd_step`: for every entry with `pair != -2`, with `x` the lane's value,
  `D(o) = x - sigma o` (`pred` `eps`) or `x / (sigma^2 + 1) - o sigma / sqrt(sigma^2 + 1)` (`v`),
  guided as `D = D(o_u) + cfg (D(o_c) - D(o_u))` against the pair's rows (`D(o_c)` without one);
  flag 4 writes `D` to the lane's `preview`. Then by `sampler`:
  0, Euler: `x' = x + (sigma_next - sigma)(x - D) / sigma`;
  1, Euler ancestral (eta 1): `sigma_up = min(sigma_next, sqrt(sigma_next^2 (sigma^2 -
  sigma_next^2) / sigma^2))`, `sigma_down = sqrt(sigma_next^2 - sigma_up^2)`,
  `x' = x + (x - D) / sigma (sigma_down - sigma)`, plus `sigma_up` times the seed's stream
  `1 + step` (`rad_philox_normal_s`, element `k n + c`) while `sigma_next > 0`;
  2, DPM++ 2M: `x' = D` at `sigma_next` 0; otherwise with `t = -ln sigma`, `h = -ln sigma_next - t`,
  `x' = (sigma_next / sigma) x - expm1(-h) D'`, where `D' = D` on a first step and otherwise
  `(1 + 1/(2r)) D - (1/(2r)) hist` with `r = (t + ln sigma_prev) / h`.
  Every step stores `D` to `hist` for the next. Any other sampler is refused.

**Images** are `[height, width * channels]`, dense, and the ops read both sizes off the operand
shapes: an image's size belongs to a request, and a parameter is fixed at declare.

- `conv2d_nhwc`: `w` is PyTorch's `[cout, cin, k, k]` (a kernel may store it permuted through its
  layout hook), `b` `[cout]`; `up` 2 reads the input upsampled 2x nearest-neighbour; `pad` `same`
  is `k/2` zeros a side, `down` is none above and left and one below and right
  (`ZeroPad2d((0,1,0,1))` before a stride-2 conv), `causal` is `k - 1` rows above, none below and
  `k/2` columns a side (causal along the rows: LTX-2's audio VAE). f32 accumulation, the bias added, one rounding.
- `im2col_nhwc`: `conv2d_nhwc`'s receptive fields as rows, so a convolution is a GEMM against the
  checkpoint's own weight. `x` is the image `[Hin, Win * cin]` (rows at their own pitch), read
  through `up`, `pad` and `stride` as `conv2d_nhwc` reads it; `y` is `[Ho * Wo, cin * k * k]`, its
  rows exactly the output pixels, and row `oy * Wo + ox`, column `(ci * k + i) * k + j` holds the
  value `conv2d_nhwc` multiplies `w[o, ci, i, j]` by, 0 in the padding. That column order is the
  weight's own `[cout, cin, k, k]` flattened, so `y @ w.reshape(cout, cin k k)^T + b` (a
  `gemm_nt_bias`) is the convolution with the weight as stored, and a runtime LoRA over that GEMM
  reads a conv adapter's down projection `[rank, cin, k, k]` flattened the same way. A copy: no
  arithmetic, the bits moved. libr4d stages a block's k input rows in LDS so its reads and its
  stores are both coalesced; it serves `k` up to 7 and `stride` and `up` 1 or 2.
- `l2norm_chan`: each pixel's channels to unit L2 (eps 1e-12), times `sqrt(n)`, times `gamma`
  (three roundings), then `act` `none` or `silu`.
- `chan_affine`: `mul` and `add` are space-separated lists of `n` values, each rounded to bf16;
  `fma` is `bf16(bf16(x * mul) + add)`, `inv` is `bf16(bf16(x - add) / mul)`, the roundings to
  `y`'s dtype. With `round` 0 (default 1) the lists stay f32 and so does the arithmetic, rounded
  once into `y`: `fma` is `rd(f32(x * mul) + add)`, `inv` `rd(f32(x - add) / mul)` (MiniMax-H3
  denormalises its f32 audio latents with an f32 deviation and mean).
- `d2s_add` (Wan's DupUp3D on one frame): `y[2r+i, 2c+j, o] += x[r, c, (o F + off + 2i + j) / rep]`,
  `F = 4 ft`, `off = 4` when `ft` is 2, `rep = cout F / cin`.
- `s2d_avg_add` (AvgDown3D on one frame): `y` gains the mean of each output channel's group of
  `cin F / cout` dealt values, a dealt channel being `(c, t, i, j)` at `c F + 4t + 2i + j`; with
  `ft` 2 the frame `t = 0` is the zero frame the reference pads in, which the mean still counts.
- `to_rgba8`: `round(clamp(bf16(bf16(clamp(x, -1, 1) / 2) + 0.5), 0, 1) * 255)`, u8 out.

**`attn_prefix_self`** attends over two key sources: entry `s`'s first `seqused[s]` keys in the
paged cache, all visible, and the pass's own rows `k`/`v` beside `q` (dense, at their own pitch,
indexed as `q`'s rows), of which row `i` sees `j <= i` with `causal` and all of them without. It is
an image's latent rows against its conditioning, or one segment of a conditioning against the
segments before it. `qk` `"i8"` is the int8 score form and takes `k_i8` `[tokens, kv_heads *
head_dim]` and `k_scale` `[tokens, kv_heads]` f32, the own keys as `quant_act_i8g` writes them at
group `head_dim`. The query and every prefix key are quantised inside the op by the same rule
(scale `amax/127`, 1 for an all-zero row; code `rint(clamp(x * (1/scale), ±127))`), a score is the
exact integer dot product times both scales, and `k` is not read for the own rows. The prefix's
three operands come together or not at all: without them every entry's prefix is empty and the op
is self attention over each entry's rows (`max_ctx` 0 declares that form), which is how a single
long sequence takes the int8 score form.

`pv` `"fp8"` is the fp8 P.V form, prefix-free only: `v` holds E4M3 codes and `v_scale`
`[kv_heads * head_dim]` their column scales (`quant_cols_fp8`). Over a row's visible keys, with
`p = exp(s - max)` against the row's max, each weight is taken to E4M3 as `w = e4m3(448 p)` and

    o = (sum w v) v_scale / (448 sum p)                        f32 sums in key order

A kernel may take `p` against its running max in its own key order as long as every `p <= 1`
(E4M3's top is 448), which differs from this only by where each weight was rounded.

## Audio generation

The ops an audio generator declares (`rad_declare_audio_generator`, `abi/rad_builder.h`) beyond a
language model's and an image generator's: a frame sampler with guidance between branches over
residual codebooks, the frame embeddings, the window states of a latent denoiser, and a
vocoder's 1-D convolutions. Each rounds where the reference model materialises a bf16 tensor; the
libref function says which (`libref/ref_audio.cpp`).

| op | params | operands |
|---|---|---|
| `conv1d` | `M`(range) `cin` `cout` `k` `dil` `act`/opt `dtype` `causal`/opt | `x`, `w`w, `cu_seqlens`, `res`? → `y`; `b`w? |
| `conv_t1d` | `M`(range) `cin` `cout` `stride` `dtype` `pad`/opt `k`/opt | `x`, `w`w, `b`w, `cu_seqlens` → `y` |
| `snake` | `M`(range) `n` `dtype` `log`/opt | `x`, `alpha`w → `y`; `beta`w? |
| `snake_aa` | `M`(range) `n` `ratio` `k` `log`/opt `dtype` | `x`, `alpha`w, `beta`w, `f_up`w, `f_down`w, `cu_seqlens` → `y` |
| `cfg_topk_sample` | `M`(range) `n_vocab` `k` `rep` `cond_k`/opt `cfg`:f64 `temp`:f64/opt `stream` `dtype` `sup_lo`/opt `sup_hi`/opt `penalty`:f64/opt | `logits`, `seed`, `frame` → `ids`; `cfg`?, `lane`?, `hist`:inout? |
| `rvq_embed` | `M`(range) `n_embd` `n_codes` `vocab` `off0` `scale`:f64 `dtype` `round`/opt | `codes`, `t0`w, `t1`w → `y` |
| `frame_mix_store` | `M`(range) `n_embd` `L` `stride` `rep`/opt `dtype` | `hid`, `logits`w, `scale`w, `lane`, `frame`, `status`, `dst`:inout |
| `end_status` | `M`(range) `end` `rep` | `ids`, `lane`, `status`:inout |
| `frame_store` | `M`(range) `n_codes` `stride` `rep`/opt `skip`/opt | `codes`, `lane`, `frame`, `status`, `dst`:inout |
| `resample_nearest` | `M`(range) `n` `dtype` | `x` → `y` |
| `fourier_embed` | `M`(range) `n` `round`/opt `dtype` | `t`, `w`w → `y` |
| `win_input` | `M`(range) `n_lat` `n_zero` `n_cond` `stride` `dtype` | `lat`, `cond`, `lat_cu`, `win_lane`, `pair` → `y` |
| `win_prep` | `M`(range) `n` `stride` `ov` `dtype` | `flags`, `t`, `seed`, `win_index`, `win_lane`, `pair`, `lat_cu`, `prev`, `noise`:inout, `lat`:inout |
| `win_finish` | `M`(range) `n` `n_cond` `stride` `ov` `dtype` | `flags`, `pair`, `win_lane`, `lat_cu`, `cond`, `lat`:inout, `prev`:inout, `prev_cond`:inout |
| `chan_split` | `M`(range) `n` `parts` `dtype` | `x` → `y` |
| `interleave2` | `M`(range) `dtype` | `a`, `b` → `y` |
| `iota` | `M`(range) `start` `step` `period`/opt | → `y` |
| `cu_repeat` | `M`(range) `mult` | `len` → `y` |
| `int_affine` | `M`(range) `src` `mul` `add` | `x` → `y` |

**Frames.** A lane is one piece; a frame is one step of its language model, whose branches (the
conditional first, then the unconditional) are consecutive rows, `rep` of them a lane.

- `cfg_topk_sample`: per lane, `g = u + cfg (c - u)` between its first two branch rows (`c` alone
  at `rep` 1), `cfg` the operand's value for the lane when it is bound. With `hist`
  `[lanes, n_vocab]` i32 and `lane` `[n]`, the lane's history row is cleared at frame 0 and every id
  it holds is penalised, `g > 0 ? g / penalty : g * penalty` (a repetition penalty over the lane's
  earlier picks); the pick is recorded in it. Ids in `[sup_lo, sup_hi)` are never candidates; with
  `cond_k` > 0 only the entries where `c` reaches its `cond_k`-th largest value are. The `k`
  largest `g` (ties to the lower index) are kept, divided by `temp`, softmaxed in f32 and drawn by
  inverse CDF in descending `g`, `u = rad_philox_uniform_s(seed, 2^32 + frame, stream)`
  (`abi/rad_philox.h`); the pick goes to every one of the lane's `rep` slots. `logits`
  `[rep n, n_vocab]` f32 or bf16, `seed` `[n, 2]` u32 low word first, `frame` `[n]`.
- `end_status`: `status[lane[i]] = ids[rep i] == end ? -1 : ids[rep i]`.
- `rvq_embed`: `codes` is code-major `[n_codes, rows]`, read at the operand's own strides (a block
  of a wider plane is a view);
  `y = bf16(bf16(t0[c_0 + off0] + bf16(sum_k t1[c_k + (k - 1) vocab])) * scale)`, or with `round` 1
  one f32 sum of every codebook's row in codebook order, times `scale`, rounded once to `y`'s dtype.
  A first code outside `t0` leaves its row unwritten; a later code outside `t1` adds nothing.
- `frame_store`: `dst[(lane n_codes + k) stride + frame - skip] = codes[k, rep i]` for every `k`,
  when `frame >= skip` and `status[lane] >= 0`: a frame's codes into its lane's history, each
  codebook a row over the lane's frames, so a run of a lane's frames is a block of `n_codes`
  rows.
- `frame_mix_store`: `m = bf16(bf16(sum_l w_l h_l) * bf16(scale))`, `w = bf16(softmax(bf16(logits)))`
  over the `L` planes of `hid` `[L, rows, n_embd]`, row `rep i` of each plane being lane `i`'s
  conditional row; stored at `dst[lane * stride + frame - 1]` for `frame >= 1` and
  `status[lane] >= 0` (frame 0 only moves past the prompt; a negative status ended the piece).
- `iota`: `y[i] = start + i * step`, or `start + (i % period) * step` with `period` (the
  positions of equal sequences laid end to end). `cu_repeat`: `y[j] = j * len[0] * mult`, the bounds of equal
  sequences whose length lives on the device. `int_affine`:
  `y[j] = (src[j] < 0 ? 0 : x[src[j]]) mul[j] + add[j]` over three lists of integers of one length
  (`src` -1 a constant): the shapes and bounds a model derives from a pass's device geometry, such
  as a decoder's volume at each of its scales.

**Windows.** The denoiser runs over windows of frames, each entry of a window step a lane's latent
in a state buffer `stride` rows a lane (`lat_cu` bounds each entry's rows, `win_lane` names its
lane, `pair` is -2 for an unconditional branch, which moves no lane). `flags` carries 1 (the
window's first step), 8 (its last) and 16 (it overlaps the previous window by `ov` rows).

- `win_input`: `y = [lat | zeros(n_zero) | cond]` from the entry's lane; an entry with `pair` -2
  takes zeros for `cond`.
- `win_prep`: flag 1 fills the lane with `rad_philox_normal_s(seed, 1 + win_index, k n + c)` as
  bf16, keeping its first `ov` rows in `noise` with flag 16; flag 16 then binds the first
  `min(ov, rows)` rows to the previous window's end,
  `bf16(bf16(1 - bf16((1 - 1e-6) t)) noise + bf16(t prev))`, `t = bf16(t[e])`.
- `win_finish` (flag 8): with flag 16 the first `min(ov, rows)` rows become `prev` exactly; then
  rows `[max(0, rows - 2 ov), max(.., rows - ov))` of the latent and of `cond` are kept as the next
  window's `prev` and `prev_cond`.
- `fourier_embed`: `[cos(2 pi t w) | sin(2 pi t w)]`, `t` f32 `[M]`, `w` `[n / 2]`; with `round` 1
  `t`, `2 pi t`, the product and each cosine and sine are rounded to bf16.
- `resample_nearest`: output row `j` reads input row `min(floor(j * in / out), in - 1)`, the row
  counts read off the operands (torch's `interpolate(mode="nearest")`).

**The vocoder.** Rows are samples, channels innermost; `cu_seqlens` bounds each sequence and the
input reads zero outside it.

- `conv1d`: PyTorch's `[cout, cin, k]` weight, `k` odd, dilation `dil` and padding
  `dil (k - 1) / 2` a side, or with `causal` 1 (any `k`) `dil (k - 1)` on the left; `res` is added
  after the bias, then `act` `none`, `tanh` or `clamp`, `min(max(y, -1), 1)` (BigVGAN's output
  clamp; a NaN stays NaN). f32 accumulation, one rounding.
- `conv_t1d`: PyTorch's `[cin, cout, k]` transposed weight, any `k` of at least a stride (two
  strides by default), padding `pad` (`(k - stride) / 2` by default, which needs `k - stride` even;
  0 is the causal form, whose full output's last `k - stride` rows are dropped; it may exceed a
  stride): output row `o` of a sequence is `b + sum_i x[i] w[(o + pad) - i stride]` over the
  sequence's input rows `i` with `0 <= (o + pad) - i stride < k`, summed in f32 as
  `q = (o + pad) / stride`, `r = (o + pad) % stride` and taps `t = 0, 1, ..` while
  `r + t stride < k` reading `x[q - t] w[r + t stride]`; one rounding. Sequence `e`'s rows
  `[cu[e], cu[e+1])` become `[stride cu[e], stride cu[e+1])`, `stride L` rows from `L`. BigVGAN
  upsamples with `(k 9, stride 5, pad 2)` and `(k 4, stride 2)`.
- `snake`: `x + sin(alpha x)^2 / (alpha + 1e-9)`, `alpha` per channel; with `beta` the SnakeBeta
  form `x + (1 / (beta + 1e-9)) sin(alpha x)^2`. `log` 1 reads both as logarithms, exponentiated
  first.
- `snake_aa`: BigVGAN's alias-free SnakeBeta (diffusers' `MiniMaxH3AudioActivation1d`: an
  upsampler, the activation, a low-pass downsampler) over each sequence, in f32. For channel `c`
  of a sequence of `L` rows, `P = k / ratio - 1`:
  - `xp` is the sequence replicate-padded by `P` rows on each side;
  - `u_full[o] = sum_i f_up[o - i ratio] xp[i]` over the rows `i` with `0 <= o - i ratio < k`, in
    ascending `i` (a depthwise `conv_transpose1d` at stride `ratio`), and
    `u = ratio u_full[P ratio + (k - ratio) / 2 + m]` for `m < L ratio` (the full output cropped by
    `P ratio + (k - ratio) / 2` rows at the start and `P ratio + (k - ratio + 1) / 2` at the end);
  - `a = u + (1 / (B + 1e-9)) sin(A u)^2` as `snake`'s `beta` form evaluates it, `A = exp(alpha[c])`
    and `B = exp(beta[c])` with `log` 1, the values themselves otherwise;
  - `ap` is `a` replicate-padded by `k / 2` rows on the right and `k / 2` on the left less one for
    an even `k`, and `y[o] = sum_j f_down[j] ap[o ratio + j]` over `j < k` in ascending `j`, for
    `o < L`.

  `f_up` and `f_down` are the checkpoint's `[1, 1, k]` Kaiser filters, one for every channel
  (MiniMax-H3: `k` 12, `ratio` 2). A sequence keeps its length.
- `chan_split`: `y[p rows + r] = x[r, p n .. p n + n)`, a stereo latent's halves as two streams.
- `interleave2`: `y [rows, 2] = (a[r], b[r])`, f32.

## Video generation

The ops a video generator (MiniMax-H3) needs beyond a language model's, an image generator's and
an audio generator's: its diffusion transformer's modulated RMS norm and its sampler step, and the
video decoder's patch unfold, the tiles it runs over, their stitch back into frames and their
rotary positions. The transformer's attention is `qk_norm_rope` `axial` (its step-distilled form's
is [video sparse attention](#video-sparse-attention)) and its gated residual
`gate_add` `act` none; the soundtrack's vocoder is `conv_t1d`, `snake_aa` and `conv1d` `clamp`
([Audio generation](#audio-generation)). Each op rounds where the reference model materialises a
bf16 tensor and keeps f32 where it computes in f32; the libref function says which
(`libref/ref_gen.cpp`, `libref/ref_video.cpp`). Its latent upscaler's ops are
[below](#the-latent-upscaler).

| op | params | operands |
|---|---|---|
| `rms_mod` | `M`(range) `n` `eps`:f64 `gated`/opt `dtype` `wadd`:f64/opt `norm_once`/opt `norm`/opt | `y`?, `g`?, `s`, `w`w?, `cu_seqlens`, `x`:inout → `z` |
| `rms_mod_quant_fp8` | `M`(range) `n` `eps`:f64 `group` `gated` `dtype` `wadd`:f64/opt `norm_once`/opt `norm`/opt | `y`?, `g`?, `s`, `w`w?, `cu_seqlens`, `x`:inout → `q`, `scale`, `z`? |
| `flow_x0_step` | `M`(range) `n` `dtype` | `v`, `t`, `sigma`, `sigma_next`, `x`:inout |
| `patch_unfold` | `M`(range) `C` `pt` `ph` `pw` `mul` `add` `dtype` | `x`, `geom` → `y` |
| `vtile_gather` | `M`(range) `n` `tile` `overlap` `frames` `period` `n_reg` `dtype` | `x`, `reg`w, `geom` → `y` |
| `vtile_stitch` | `M`(range) `tile` `overlap` `scale` `pt` `skip` `keep` `blend` `period` `mean` `std` | `x`, `geom`, `prev`:inout → `y` |
| `grid_pos3` | `M`(range) `T` `H` `W` `period` `scale`:f64 | → `y` |

**The transformer.** `cu_seqlens` partitions the rows into entries as for `ln_mod`, and each row
reads its entry's rows of `s` and `g`.

- `rms_mod` is `norm(h) (1 + scale) + shift` after the branch before it is added: with `gated` 1
  first `x = bf16(x + bf16(g[e] y))`, no tanh, stored back to `x`; then
  `n1 = bf16(x rsqrt(mean(x^2) + eps))` (the sum of squares f32 in ascending order, over the
  stored `x`), `n2 = bf16(n1 w)`, `m = bf16(n2 bf16(1 + scale[e]))`, `z = bf16(m + shift[e])`.
  `s` is `[n_seq, >= 2n]` at its own row pitch, `shift = s[e, 0:n]` and `scale = s[e, n:2n]` (a
  column slice of the modulation table); `g` is `[n_seq, >= n]` at its own pitch; `w` is `[n]`.
  The residual rounds to `x`'s dtype and the norm chain to `z`'s. `wadd` is added to the gain in
  f32 (a zero-centred norm's `1 + w`), and with `norm_once` 1 the norm and its gain round together,
  `n2 = bf16(x rsqrt(mean(x^2) + eps) (w + wadd))`, as an f32 RMS norm with a gain stores its
  output (Krea 2's). `norm` `layer` (default `rms`) is a LayerNorm instead: `x` centred by its
  mean (an f32 sum in ascending order) and `rsqrt` of the centred values' biased variance plus
  `eps`. Without `w` the norm has no gain and rounds once, `n2 = bf16((x - mean) rsqrt(var +
  eps))` (FLUX.3's LayerNorm without an affine). `norm` `none` takes no norm and no gain,
  `n2 = x`, and refuses `w` (LTX-2's text keys and values, modulated as they are).
- `rms_mod_quant_fp8` is `rms_mod` with the norm chain in bf16, then `quant_act_fp8` of that bf16
  row per `group` as `ln_mod_quant_fp8` takes it; `z` written when bound.
- `flow_x0_step` is MiniMax-H3's sampler step over a lane's rows, all f32, each product and sum
  one rounding: the velocity points at the data, so the clean estimate is `d = x + (1 - t) v` and
  the step blends toward it, `st = f32(1 - t)`, `d = f32(x + f32(st v))`,
  `r = f32(sigma_next / sigma)`, `x = f32(f32(r x) + f32(f32(1 - r) d))`. `t` (the timestep the
  transformer was conditioned on), `sigma` and `sigma_next` (the schedule's entries) are one f32
  value each; `v` is f32 or bf16, `x` f32.

**The decoder** runs over a request's latent geometry, which `geom`, i32 on the device, carries
(the parameters are fixed at declare). `mul`, `add`, `mean` and `std` are space-separated lists
kept in f32. An output holds at least the rows the geometry writes; rows past them are not
written.

- `patch_unfold`: patch rows `x [F Hp Wp, C pt ph pw]` (row `(f, i, j)` frame-major then
  row-major, column `(c, a, b, d)` at `c pt ph pw + a ph pw + b pw + d`, `geom` = `(F, Hp, Wp)`)
  to voxel rows `y [(F pt) (Hp ph) (Wp pw), C]`, row `(t, yy, xx)` row-major:
  `y[(t, yy, xx), c] = f32(f32(x[(t/pt, yy/ph, xx/pw), (c, t%pt, yy%ph, xx%pw)] mul[c]) + add[c])`,
  the latent denormalised with f32 statistics. `M` is the output's rows.
- The tiles of an axis of `L` latent units are diffusers' `_split_tiles` in latent units: a `tile`
  at least as long as the axis is one tile `[0, L)`, refused unless `L == tile`; otherwise
  `n = ceil(L / tile)`, raised while `tile n - overlap (n - 1) < L`; `ov = [overlap] * (n - 1)`,
  the slack `tile n - sum(ov) - L` added one unit at a time to `ov[i % (n - 1)]` for
  `i = 0, 1, ..`; `start[0] = 0`, `start[i + 1] = start[i] + tile - ov[i]`. `overlap` must be
  below `tile`. A frame's tiles are row-major over (row band, column band).
- `vtile_gather`: the tiles out of voxel rows `x [frames H W, n]` (`(t, yy, xx)` row-major,
  `geom` = `(frames, H, W)`, its frames the parameter's). Tile `k` is rows
  `[k period, (k + 1) period)` of `y`: its `frames tile tile` voxels `(t, ty, tx)` row-major from
  `x`'s row `(t, start_y + ty, start_x + tx)`, then `reg`'s first `n_reg` rows in `y`'s dtype (a
  decoder's register tokens), then zero rows to the period, which holds at least
  `frames tile tile + n_reg` rows. `M` is `y`'s rows, `n_tiles period`.
- `vtile_stitch`: a decoded chunk's tiles into RGB bytes. `x [>= n_tiles period, 3 pt scale^2]`:
  tile `k`'s row `k period + (lt tile + ty) tile + tx`, column
  `c pt scale^2 + a scale^2 + b scale + d`, is colour `c` of pixel `(ty scale + b, tx scale + d)`
  of clip frame `lt pt + a`. `geom` = `(frames, H, W, first, last)`: the chunk's latent geometry,
  and whether it is its video's first and its last chunk (nonzero). Pixel `(Y, X)` belongs to the
  tile `(i, j)` whose kept range holds it, `[start_i scale, start_{i+1} scale)` on each axis (the
  last tile to its end), at `(ly, lx)` within it, `T(i, j, ly, lx)` that tile's decoded value. With
  every product and sum rounded to `x`'s dtype (`rd`), `wb(p, e) = rd(p / e)` and
  `wa(p, e) = rd(1 - wb(p, e))`, the spatial stitch blends with the tile above, then with the tile
  to the left, both as decoded:

      V = rd(rd(T(i-1, j, ly + tile scale - ev, lx) wa(ly, ev)) + rd(T(i, j, ly, lx) wb(ly, ev)))
          when i > 0 and ly < ev = ov_y[i-1] scale, else T(i, j, ly, lx)
      S = rd(rd(T(i, j-1, ly, lx + tile scale - eh) wa(lx, eh)) + rd(V wb(lx, eh)))
          when j > 0 and lx < eh = ov_x[j-1] scale, else V

  Output frame `k < keep` is clip frame `skip + k`, its first `blend` frames blended with the
  previous chunk's tail unless the chunk is first,
  `v = rd(rd(prev[k] wa(k, blend)) + rd(S wb(k, blend)))`; then `prev[k'] = S(2 skip + keep + k')`
  for `k' < blend` (`prev` `[blend, H scale, W scale, 3]`), which a last chunk also writes as
  frames `keep + k'`. A byte is `round_half_even(255 clamp(f32(f32(v std[c]) + mean[c]), 0, 1))`
  (a NaN is 0). `y` is u8 `[frames_out, H scale, W scale, 3]`, `frames_out` `keep` or
  `keep + blend`; `M` is its pixels. Refused: `blend` past `keep`, clip frames (`frames pt`) short
  of `2 skip + keep + blend`, a `period` short of `frames tile tile`. MiniMax-H3: `tile` 16,
  `overlap` 4, `scale` 16, `pt` 4, `skip` 3, `keep` 17, `blend` 5, `period` 1797, ImageNet's
  `mean` and `std`.
- `grid_pos3`: a ViT decoder's rotary positions over consecutive tiles of `period` rows, the
  tile's `T x H x W` tokens first, then the period's other rows (register tokens). Row `m` is
  token `k = m % period`; for `k < T H W`, `(t, i, j) = (k / (H W), (k / W) % H, k % W)` and
  `y[0][m] = f32(f32(scale) g(t, T))`, `y[1][m] = f32(f32(scale) g(i, H))`,
  `y[2][m] = f32(f32(scale) g(j, W))`, `g(c, n) = f32(f32(2 f32((c + 0.5) / n)) - 1)` (torch's
  `2 * (arange(0.5, n) / n) - 1`); otherwise all three are 0. `y` is f32 `[3, >= M]`,
  component-major, at its own row pitch: a `[3, M]` view of a wider plane, which `qk_norm_rope`
  `axial` then reads as f32 positions. MiniMax-H3: `T` 7, `H` 16, `W` 16, `period` 1797,
  `scale` `2 pi`.

### Video sparse attention

MiniMax-H3's step-distilled form (FastH3) was trained with FastVideo's video sparse attention: a
video query sees the dense rows and only the video tiles it scores highest, scored per head from
tile means, and a coarse branch adds attention over every tile's mean, gated per element.
`vsa_select` makes the selection and the coarse term, `attn_vsa` the attention over them. They
share their parameters; `attn_vsa` recomputes `k` from `sparsity`.

| op | params | operands |
|---|---|---|
| `vsa_select` | `M`(range) `head_dim` `tile_t` `tile_h` `tile_w` `patch` `sparsity`:f64 `scale`:f64/opt | `q`, `k`, `v`?, `segs`, `geom` → `sel`, `coarse`? |
| `attn_vsa` | `M`(range) `head_dim` `tile_t` `tile_h` `tile_w` `patch` `sparsity`:f64 `scale`:f64/opt `qk`/opt `pv`/opt | `q`, `k`, `v`, `segs`, `geom`, `sel`, `coarse`?, `gate`? → `out`, `k_i8`?, `k_scale`?, `v_scale`? |

**The tiles.** A pass's `R` rows (`M` is their band) are one packed sequence. `segs`, i32
`[n_seg + 1]`, cuts it into segments: `segs[0] = 0`, non-decreasing, `segs[n_seg] = R`; every
segment but the last is dense (the text, the soundtrack), the last is the video grid of
`F x Hp x Wp` rows in `(t, h, w)` order, `w` fastest. `geom` is the request's geometry on the
device, i32 `[RAD_VGEOM_N]` (`abi/rad_runtime.h`): `F = geom[RAD_VGEOM_LAT_FRAMES]`,
`Hp = geom[RAD_VGEOM_LAT_H] / patch`, `Wp = geom[RAD_VGEOM_LAT_W] / patch`. With
`TT = tile_t tile_h tile_w`:

- each dense segment, on its own, is cut into chunks of `TT` rows from its start, the last chunk
  shorter, an empty segment none: the prefix tiles `0 .. P-1`, in segment order;
- the grid is cut into `ceil(F / tile_t) x ceil(Hp / tile_h) x ceil(Wp / tile_w)` boxes. Box
  `(a, b, c)` holds `t` in `[a tile_t, min(F, (a + 1) tile_t))` and likewise `h` and `w`, edge boxes
  smaller, and is tile `P + (a nb + b) nc + c` (`nb = ceil(Hp / tile_h)`, `nc = ceil(Wp / tile_w)`):
  `V` video tiles, `N = P + V` in all;
- a tile's rows are taken in its `(t, h, w)` raster order (a prefix tile's in row order), which
  fixes the order of every f32 sum over a tile. `size(i)` is tile `i`'s row count.

`k = max(1, min(ceil((1 - sparsity) V), V))`, computed in f64 from the f64 `sparsity`, as the
trained schedule computes it: `(1 - 0.8) 660` is `131.99999999999997`, so MiniMax-H3's 660 video
tiles at 1344 x 768 keep 132, and `(1 - 0.7) 10` keeps 4.

Both ops refuse with `RAD_E_SHAPE` a `segs` that does not start at 0, decreases, or covers no row
or more rows than the operands hold; a last segment of other than `F Hp Wp` rows; a box extent or
`patch` below 1; a `sparsity` outside `[0, 1)`; and an optional operand without its partner. `sel`
must be i32 (`RAD_E_DTYPE`). Rows past `segs[n_seg]` are neither read nor written. `q`, `k`, `v`,
`gate` and `out` are `[R, n_head head_dim]`, a column slice of a wider projection read at its own
row pitch, or `[R, n_head, head_dim]`; the head count is read off `q`.

**`vsa_select`.** Per head `h` and tile `i`, the tile means and the scores, in f32:

    qp[h][i] = (sum over the rows r of tile i, in raster order, of q[r, h, :]) / size(i)
    kp[h][i], and vp[h][i] (read for `coarse` alone), likewise from k and v
    S[h][i][j] = (qp[h][i] . kp[h][j]) scale        the dot a sum over d ascending, then the product

For a video query tile `i >= P`, `sel[h][i][0 .. k)` is the `k` video tiles `j` (`P <= j < N`) of
largest `S[h][i][j]`, written in ascending tile order; equal scores go to the lower tile, and a NaN
ranks above every number (`torch.topk`'s ranking). The prefix query tiles' rows of `sel` and its
slots past `k` are not written. `sel` is `[n_head, tiles_cap >= N, k_cap >= k]`. With `coarse`
(and `v`: both or neither), for every tile `i`, prefix tiles too:

    e_j = exp(S[h][i][j] - max over all N tiles of S[h][i][:])
    p[h][i][j] = e_j / (sum over all N tiles, j ascending, of e_j)
    coarse[h][i][:] = (sum over j ascending of p[h][i][j] vp[h][j][:]), rounded once to its dtype

`coarse` is `[n_head, tiles_cap >= N, head_dim]`. At `sparsity` 0, `k = V` and every video tile is
selected.

**`attn_vsa`.** A query row of tile `i` sees the rows of every tile when `i < P` (dense prefix
queries), else of the prefix tiles and of the tiles `sel[h][i][0 .. k)`. Each such row of `sel` must
hold `k` ascending video tiles, `RAD_E_INVAL` otherwise, checked before anything is written; `sel`'s
other entries are not read. Per head (`q`, `k` and `v` have the same heads), `o` is the softmax over
the visible keys of `(q . k) scale` times `v`, accumulated in f32, and

    out[r, h, :] = o                                           without coarse and gate
    out[r, h, :] = rd(rd(o) + coarse[h][i][:] gate[r, h, :])   with them, in f32

`rd` rounding to `out`'s dtype: FastVideo's inference epilogue, the sparse output in bf16, then
`out.addcmul_(coarse, gate)` in bf16 with f32 arithmetic. Two bf16 values multiply exactly in f32,
so that is one f32 add between the two roundings. `gate` comes with `coarse`, both or neither. At
`sparsity` 0 (with `vsa_select`'s selection) `attn_vsa` is `attn_dense` over the whole sequence.
libref takes the keys by ascending tile, each tile in raster order, in `attn_dense`'s two passes.
`qk` `"i8"` is `attn_prefix_self`'s int8 score form: `k_i8` `[M, n_head * head_dim]` and `k_scale`
`[M, n_head]` f32 are k as `quant_act_i8g` writes it at group `head_dim`, each query row is
quantised inside the op by the same rule, a score is `((dot * q_scale) * k_scale) * scale` in f32
with `dot` the codes' exact integer sum, and `k` is not read. `pv` `"fp8"` is `attn_prefix_self`'s
fp8 P.V form (`v` E4M3 codes, `v_scale` `[n_head * head_dim]`), before the epilogue.

MiniMax-H3: `tile_t`, `tile_h` and `tile_w` 4, `patch` 2, `sparsity` 0.8, `head_dim` 128, the
default `scale` `1 / sqrt(head_dim)`.

### The latent upscaler

MiniMax-H3's latent upscaler (`LatentResizer3D`) is a 3-D convolutional network over the video
latent: its 24 channels to 512 by a 3 x 3 x 3 convolution, residual blocks
(`t = silu(GN(x)); h = conv3(t); t = silu(GN(h) (1 + scale) + shift); x = x + conv3(t)`, the
scale and shift from the upscale factor's embedding) and temporal blocks
(`t = silu(GN(x)); t = dwconv_t5(t); x = x + pwconv(t)`, the 1 x 1 x 1 `pwconv` a `gemm_nt_bias`)
at the source size, a spatial bilinear resize, the same blocks at the target size, and
`GroupNorm(32)`, SiLU and a 3 x 3 x 3 convolution back to 24 channels. `patch_fold` turns its output
back into the transformer's patch rows and `flow_renoise` takes them to the noise level a refining
pass starts from. Each op rounds once into its output (`libref/ref_video.cpp`).

| op | params | operands |
|---|---|---|
| `conv3d_ndhwc` | `M`(range) `cin` `cout` `k` `kt` `dtype` `tpad`/opt `tpre`/opt | `x`, `w`w, `b`w, `geom`, `res`? → `y` |
| `group_norm` | `M`(range) `n` `groups` `eps`:f64 `act` `dtype` | `x`, `w`w, `b`w, `mod`?, `stats`:inout → `y` |
| `tconv_dw` | `M`(range) `n` `k` `dtype` | `x`, `w`w, `b`w, `geom` → `y` |
| `resize_bilinear` | `M`(range) `n` `dtype` | `x`, `geom` → `y` |
| `patch_fold` | `M`(range) `C` `pt` `ph` `pw` `mul` `add` `dtype` | `x`, `geom` → `y` |
| `flow_renoise` | `M`(range) `n` `stream` `dtype` | `sigma`, `seed`, `x`:inout |

Activations are voxel rows: row `(t, i, j)` of a `T x H x W` volume is `(t H + i) W + j`, channels
innermost (NDHWC). `geom` is the request's volume, i32 on the device (the parameters are fixed at
declare). An output holds at least the rows the volume writes; rows past them are not written.
Every op honours its activation operands' row pitches. `rd` is a rounding into the output's dtype.

- `conv3d_ndhwc`: `x [>= T H W, cin]`, `y` and `res` `[>= T H W, cout]`, `geom = (T, H, W)`; `w` is
  PyTorch's Conv3d weight `[cout, cin, kt, k, k]` (a kernel may relayout it at load, as
  `conv2d_nhwc`'s layout hook does) and `b [cout]`. Stride 1, zero padding `kt/2` frames and `k/2`
  rows and columns on each side (`same`; `k` and `kt` odd):
  `y[(t,i,j), o] = rd(f32(sum over (dt, kh, kw, ci) of w[o, ci, dt, kh, kw]
  x[(t+dt-kt/2, i+kh-k/2, j+kw-k/2), ci]) + b[o])`, terms outside the volume zero, f32 accumulation
  in an order of the implementation's (libref: the taps in `(dt, kh, kw)` order, each tap's
  channels ascending). `tpad` (optional) pads the frames instead with `replicate` (the edge frame
  repeated, `kt/2` a side) or `causal` (the first frame repeated `kt - 1` times in front, nothing
  after; with `tpre` that many more copies in front, and y then has `T + tpre` frames): LTX-2's
  decoder and encoder. With `res`, `y = rd(rd(acc + b) + res)`: the convolution's output rounded,
  then the residual added and rounded, as a block's `skip(x) + h` in bf16. `res` may be the same
  memory as `y` (the residual added in place); `x` never aliases `y`. x, y, res and w bf16 (also
  f32), b bf16 or f32. MiniMax-H3: `k` = `kt` = 3, `(cin, cout)` `(24, 512)`, `(512, 512)`,
  `(512, 24)`.
- `group_norm` is PyTorch's `nn.GroupNorm(groups, n)` on `[1, n, T, H, W]`: `x` and `y`
  `[>= M, n]` (`M` is `y`'s rows), group `g` the channels `[g n/groups, (g+1) n/groups)` of all
  `M` rows. `mean_g` and the biased `var_g` are over the group's `M n/groups` values, accumulated in
  f32 or better (libref: f64, two passes, row-major); `rstd_g = 1 / sqrt(f32(var_g) + eps)` in f32.
  Then, in f32, `v = (x - mean_g) rstd_g w[c] + b[c]`; with `mod` (`[>= 2n]`, f32 or bf16, one row:
  `scale = mod[0:n]`, `shift = mod[n:2n]`) `v = v (1 + scale[c]) + shift[c]`; `act` `none` or
  `silu` (`v / (1 + exp(-v))`); one rounding into `y`. `stats` is f32 of at least
  `groups (2 + 3 1024)` values: afterwards `stats[2g]` and `stats[2g+1]` hold group `g`'s `mean_g`
  and `rstd_g`, and the rest is the implementation's scratch (partial sums of up to 1024 parts a
  group; libref leaves it untouched). A kernel may take several launches but must be deterministic,
  a fixed reduction order with no float atomics. `y` may be the same memory as `x`. `w` and `b`
  `[n]`, bf16 or f32. MiniMax-H3: `n` 512, `groups` 32, `eps` 1e-5; SDXL's UNet 320 to 2560
  channels in 32 groups (10 to 80 a group), its VAE 128 to 512. libr4d serves `n` a multiple of 8
  up to 8192 with an even group width.
- `tconv_dw`: a depthwise convolution along the frames (a Conv3d of kernel `(k, 1, 1)`, one group a
  channel): `geom = (T, H, W)`, `P = H W` rows a frame, `w` PyTorch's `[n, 1, k, 1, 1]` (tap `j` of
  channel `c` at `c k + j`), `b [n]`, `k` odd:
  `y[(t, p), c] = rd(f32(sum_j w[c, j] x[(t + j - k/2, p), c]) + b[c])`, frames outside `[0, T)`
  zero, the sum f32 in ascending `j`. `x` never aliases `y`. MiniMax-H3: `n` 512, `k` 5.
- `resize_bilinear`: `geom = (T, h, w, H, W)`, `x [>= T h w, n]`, `y [>= T H W, n]`; each frame
  resized on its own, as torch's `upsample_bilinear2d` with `align_corners=False` computes it for an
  output size (not a scale factor): `sh = f32(h) / f32(H)`, `src = max(sh (f32(i) + 0.5) - 0.5, 0)`
  in f32, `i0 = (int)src`, `i1 = i0 + (i0 < h - 1)`, `l1 = src - i0`, `l0 = 1 - l1`, the same for
  columns, then
  `v = l0y (l0x x[i0,j0] + l1x x[i0,j1]) + l1y (l0x x[i1,j0] + l1x x[i1,j1])` in f32 (each product
  and sum one rounding), rounded once into `y`. Any sizes, shrinking too. The upscaler's
  `F.interpolate(mode="trilinear")` at an unchanged frame count weighs each frame by exactly 1 and
  its neighbour by 0, so this is that resize. `x` never aliases `y`.
- `patch_fold` is `patch_unfold`'s inverse, normalising: voxel rows
  `x [(F pt) (Hp ph) (Wp pw), C]` to patch rows `y [F Hp Wp, C pt ph pw]`, `geom = (F, Hp, Wp)`,
  `y[(f,i,j), c pt ph pw + a ph pw + b pw + d] = f32(f32(x[(f pt + a, i ph + b, j pw + d), c]
  mul[c]) + add[c])`, `mul` and `add` lists of `C` values kept in f32. `x` bf16 or f32, `y` f32.
  `M` is `y`'s rows. MiniMax-H3: `C` 24, `pt` 1, `ph` = `pw` = 2.
- `flow_renoise` re-noises a clean latent to noise level `s` with a fresh stream, over `x`'s rows:
  `x[k, c] = f32(f32(f32(1 - s) x[k, c]) + f32(s z(k n + c)))`, `s` = `sigma` (f32 `[1]`),
  `z(e) = rad_philox_normal_s(seed, stream, e)` (`abi/rad_philox.h`), `seed` u32 `[2]` low word
  first, `stream` the parameter. `x` f32 `[M, n]` at its row pitch. MiniMax-H3: `n` 96 (the video's
  patch rows) and 32 (the soundtrack's), `stream` 1.

## LTX-2

The ops LTX-2's audio-video transformer adds (`libref/ref_ltx.cpp`): its rotary over fractional
coordinates, the per-head gate on every attention's output, and the sampler step that combines a
step's guidance passes in the clean-estimate space. Each rounds where the reference rounds
(diffusers' `transformer_ltx2.py` and `pipeline_ltx2.py`): the transformer in bf16, the rotary,
guidance and sampler in f32.

| op | params | operands |
|---|---|---|
| `rope_split` | `M`(range) `n_head` `head_dim` `dim` `n_dims` `theta`:f64 `max` `head0`/opt `dtype` | `x`:inout, `positions` |
| `head_gate` | `M`(range) `n_head` `head_dim` `dtype` | `g`, `x`:inout |
| `ltx_guide` | `M`(range) `n` `dtype` | `vc`, `vu`?, `vs`?, `vm`?, `scales`, `sched`, `stats`:inout, `x`:inout |
| `pixel_norm` | `M`(range) `n` `eps`:f64 `form` `act`/opt `ileave`/opt `dtype` `mult`:f64/opt | `x` → `y` |
| `d2s3d` | `M`(range) `C` `st` `sh` `sw` `drop`/opt `dtype` | `x`, `geom` → `y` |
| `s2d3d` | `M`(range) `C` `st` `sh` `sw` `tpre`/opt `group`/opt `mode` `add`/opt `dtype` | `x`, `geom`, `y`:inout |
| `pix_patch` | `M`(range) `p` `C` `order`/opt `dtype` | `x` → `y` |
| `patch_pix8` | `M`(range) `p` `C` `order`/opt `dtype` | `x` → `y` |
| `stft_logmel` | `M`(range) `hop` `nfft` `n_mel` `ch` `dtype` | `x`, `basis`w, `mel`w → `y` |
| `bwe_up` | `M`(range) `ratio` `ch` `dtype` | `x`, `res` → `y` |
| `vol_index` | `M`(range) | `geom` → `y` |
| `tile_grid` | `M`(range) `tile` `stride` `rule`/opt `min`/opt `tile_t`/opt `stride_t`/opt `min_t`/opt `tail`/opt | `geom` → `y` |
| `tile_crop` | `M`(range) `C` `dtype` | `x`, `geom`, `box` → `y` |
| `tile_stitch` | `M`(range) `tile` `stride` `rule`/opt `min`/opt `tile_t`/opt `stride_t`/opt `min_t`/opt `scale` `scale_t`/opt `p` `C` `order`/opt `dtype` | `x`, `geom`, `first` → `y` |

- `rope_split` is LTX2AudioVideoRotaryPosEmbed's "split" rotary. Over the full width `dim` of all
  heads there are `P = dim / 2` pairs: the first `pad = P - steps n_dims` are not rotated, and the
  rest are a frequency ladder `freq[k] = f32(theta^linspace(0, 1, steps)[k] pi / 2)` (f64 before
  the cast, torch's linspace), `steps = P / n_dims`, its pairs interleaving the coordinates: pair
  `pad + k n_dims + d` turns by coordinate `d` at `freq[k]`. A coordinate `c` gives the angle
  `a = f32(f32(2 f32(c / max[d])) - 1) freq[k]`, `max` a list of `n_dims` values. Head `h` owns
  pairs `[h hd/2, (h+1) hd/2)` and turns its halves against each other,
  `o1 = f32(f32(x1 cos a) - f32(x2 sin a))`, `o2 = f32(f32(x2 cos a) + f32(x1 sin a))`, rounded to
  `x`'s dtype. `head0` is the first head the operand holds (a rank's share). `positions` is
  `[n_dims, M]`, f32 or integer, at its own strides. LTX-2.5: the video's self-attention `dim`
  4096, `n_dims` 3, `max` `20 2048 2048` (two padded pairs); the soundtrack's and both cross
  attentions' `dim` 2048, `n_dims` 1, `max` `20`; the text connectors' `max` 4096 over the row
  index.
- `head_gate`: `x = rd(x rd(2 rd(sigmoid(g))))` for each head's `head_dim` values, `g`
  `[M, >= n_head]` the head's logit (the gated attention's `to_gate_logits`), each operand at its
  own row pitch.
- `ltx_guide` is one sampler step over a modality's latent `x` (f32 `[M, n]`, all of it: the
  rescale's deviations are over every value) from its guidance passes' velocities (`[M, n]`, read
  as f32), all in f32:

      x0_p = f32(x - f32(v_p sigma))                                         each pass p
      g    = x0_c [+ f32(f32(gs - 1) f32(x0_c - x0_u))] [+ f32(stg f32(x0_c - x0_s))]
                  [+ f32(f32(ms - 1) f32(x0_c - x0_m))]                      left to right
      r > 0:  g = f32(f32(r f32(g f32(std(x0_c) / std(g)))) + f32(f32(1 - r) g))
      v    = f32(f32(x - g) / sigma),  x = f32(x + f32(f32(sigma_next - sigma) v))

  A bracketed term is taken when its pass is bound and its coefficient is not zero. `std` is the
  unbiased deviation over all `M n` values, accumulated in f64. `scales` is f32 `[4]`
  `(gs, stg, ms, r)` and `sched` f32 `[2]` `(sigma, sigma_next)`, both read on the device; `stats`
  is f32 of at least 8200 values, the implementation's scratch.

Its VAE, audio VAE and vocoder run in bf16 tensors, and these ops round where those tensors are
stored (`rd`, the output's dtype unless said).

- `pixel_norm` is an RMS norm over each row's `n` values: `m = rd(sum rd(x^2) / n)` (f32 sum in
  ascending order), then `form` `rsqrt` `z = rd(x rd(1 / sqrt(rd(m + eps))))` or `div`
  `z = rd(x / rd(sqrt(rd(m + eps))))`, with `act` silu `z = rd(z / (1 + exp(-z)))`, and with
  `mult` `z = rd(z mult)` (the connectors' scale to the DiT's width). The VAE's
  PerChannelRMSNorm and the audio VAE's PixelNorm are the div form (with the SiLU after them
  fused); the text connectors' per-layer norm is the rsqrt form, whose output `ileave` L
  interleaves: x's row `t L + l` goes to columns `c L + l` of y's row `t`, as the reference
  flattens `[C, layers]`.
- `d2s3d` and `s2d3d` resample voxel rows (`geom` = x's `(T, H, W)`, i32 on the device), the
  packed channel of `(c, a, b, d)` being `((c st + a) sh + b) sw + d` (a the frame offset, b the
  row's, d the column's). `d2s3d` is the upsampler's depth to space,
  `y[(t st + a - drop, i sh + b, j sw + d), c] = x[(t, i, j), packed]`, the first `drop` frames
  dropped (by default `st - 1`, the causal mapping; 0 keeps them, as a decoder's tile that does not
  start the video does). `s2d3d` is the downsampler's space to depth over the volume with `tpre` leading
  frames prepended (frames `0 .. tpre - 1` again): `mode` copy writes the packed row, `mean`
  averages each run of `group` packed channels (f32 sum, divided, one rounding); with `add` 1
  `y = rd(y + v)`. `M` is x's rows.
- `pix_patch` and `patch_pix8` are the patchifier both ways over dense 2-D operands whose shapes
  give the sizes: a `p x p` patch of `C` channels is one row of `C p p` values, channel `c`'s
  `(ph, pw)` at `c p p + pw p + ph` (`order` wh, LTX-2's) or `c p p + ph p + pw` (hw).
  `pix_patch` takes an image `[H, W C]` to `[H / p, (W / p) C p p]`; `patch_pix8` takes patch rows
  `[R, w C p p]` to RGB bytes `[R p, w p C]` by `to_rgba8`'s rule,
  `round(255 clamp(rd(rd(clamp(v, -1, 1) / 2) + 0.5), 0, 1))`, rd x's dtype.
- `vol_index` writes a volume's voxel coordinates, the rotary positions of attention over it:
  `geom` = `(T, H, W)`, `y` i32 `[3, n]` component-major at its own row pitch, voxel
  `m = (t H + i) W + j`'s `(t, i, j)` at column `m`, for `m < min(n, T H W)`.
- `tile_grid`, `tile_crop` and `tile_stitch` run a decoder in tiles as diffusers does: LTX-2's VAE
  (`tiled_decode`, `rule` `stride`, the default) and LTX-2.5's diffusion decoder
  (`_tile_intervals`, `rule` `intervals`). Along an axis of `L` cells a tile starts at every
  multiple of `stride` below `L`; under `intervals` an axis of `tile` cells or fewer is one tile, and
  while more than one start is left and the last leaves fewer than `min` cells (default 1) that
  start is dropped. A tile but the last is `min(tile, L - start)` long, the last runs to `L`. The
  same cut with `tile_t`, `stride_t` and `min_t` makes time bands (without `tile_t`, one band of
  every frame); tiles are raster-ordered over (time band, row band, column band). `tile_grid`
  writes the tiles of the grid `geom` = `(T, H, W)` to `y` i32 `[M, 8]`, row k
  `(t, h, w, t0, y0, x0, k, count)`, the last time band `tail` cells longer than its cut (frames a
  decoder carries past `T`); rows past the count are zero. A row's `(t, h, w)` is a decoder's volume
  operand and its first six entries `tile_crop`'s `box`: `tile_crop` copies the box
  `y[(a, i, j)] = x[(t0 + a, y0 + i, x0 + j)]` out of the volume `x` (`geom` its `(T, H, W)`).
  `tile_stitch` takes every tile's decoder output back to back in tile order: a cell is `scale`
  patches of `p x p` pixels a side and `scale_t` frames (default 1), the video's first cell one
  frame, so tile `(kt, i, j)` with cells `[t0, t1)` in time holds `scale_t (t1 - t0)` frames (less
  `scale_t - 1` when `t0` is 0) of `(h scale) x (w scale)` rows of patches (`pix_patch`'s columns).
  `geom` is the grid the tiles cut, `first` the first frame `y` holds, and `y` RGB bytes
  `[n (H scale p), (W scale p) C]`. A frame is the time band whose kept frames hold it (a band but
  the last keeps `stride_t scale_t`, the first `scale_t - 1` fewer), a pixel `(Y, X)` the tile
  `(Y / s, X / s)` of that band, `s = stride scale p`, the last band of an axis keeping the rest.
  With every product and sum rounded to `x`'s dtype and
  `mix(a, b, k, e) = rd(rd(a f32(1 - k / e)) + rd(b f32(k / e)))`, a band's picture is diffusers'
  in-place `blend_v` with the tile above and then `blend_h` with the tile to the left over
  `min(their extents, (tile - stride) scale p)` pixels, the neighbours read as their own blends
  left them (libref/ref_ltx.cpp writes the four-tile corner out), and a frame of a band within
  `min(their frames, (tile_t - stride_t) scale_t)` of its start is `blend_t` of the band before and
  this one; the byte is `patch_pix8`'s.
- `stft_logmel` is each channel's causal log-mel spectrogram (the bandwidth extender's MelSTFT):
  `x [S, ch]`, `basis [2F, 1, nfft]` (the windowed DFT, real then imaginary rows,
  `F = nfft / 2 + 1`), `mel [n_mel, F]`; frame `t` reads samples `t hop + j - (nfft - hop)`, zero
  before the start; `s_k = rd(sum_j basis[k, j] x)`,
  `mag_k = rd(sqrt(rd(rd(re_k^2) + rd(im_k^2))))`,
  `y[t, h n_mel + m] = rd(log(max(rd(sum_k mel[m, k] mag_k), rd(1e-5))))`; y is `[S / hop, ch n_mel]`.
- `bwe_up` is the extender's output: x `[S, ch]` raised `ratio` times by its Hann-windowed sinc
  resampler (width `w = ceil(6 / 0.99)`, `2 w ratio + 1` taps
  `rd(sinc(u) cos^2(clamp(u, -6, 6) pi / 12) 0.99 / ratio)` with `u = f32(n / ratio - w) 0.99`,
  the input's edges replicated), `up = rd(ratio rd(sum taps x))`, then
  `y = clamp(rd(res + up), -1, 1)`; rd is x's dtype, y f32 or x's dtype.

The VAEs' convolutions are `conv3d_ndhwc` with `tpad` and the audio VAE's `conv2d_nhwc` with
`pad` causal.

## Mixture of experts

| op | params | operands |
|---|---|---|
| `router_topk` | `M`(range) `n_expert` `top_k` `norm` `dtype` | `logits` → `expert_ids`, `expert_w` |
| `moe_scatter` | `M`(range) `n_expert` `top_k` `expert_base`/opt `protect`:str/opt | `expert_ids` → `sorted_tok`, `expert_offset`, `expert_count` |
| `router_topk_scatter` | `M`(range) `n_expert` `n_local`/opt `top_k` `norm` `dtype` `expert_base`/opt `protect`:str/opt | `logits` → `expert_ids`, `expert_w`, `sorted_tok`, `expert_offset`, `expert_count` |
| `router_gemm_topk_scatter` | `M`(range) `K` `n_expert` `n_local`/opt `top_k` `norm` `dtype` `expert_base`/opt `protect`:str/opt | `a`, `b`w, `gate_b`w? → `logits`, `expert_ids`, `expert_w`, `sorted_tok`, `expert_offset`, `expert_count`, `gate`? |
| `moe_gemm` | `M`(range) `N` `K` `n_expert` `top_k` `a_order` `dtype` | `a`, `w`wt, `sorted_tok`, `expert_offset` → `y` |
| `router_sigmoid_topk` | `M`(range) `n_expert` `top_k` `norm` `route_scale`:f64 `dtype` | `logits`, `bias`w → `ids`, `weights` |
| `router_sigmoid_topk_scatter` | `M`(range) `n_expert` `n_local`/opt `top_k` `norm` `route_scale`:f64 `dtype` `expert_base`/opt `protect`:str/opt | `logits`, `bias`w → `ids`, `weights`, `sorted_tok`, `expert_offset`, `expert_count`, `gate`?, `inverse`? |
| `moe_scatter_inv` | `M`(range) `n_expert` `top_k` `expert_base`/opt `protect`:str/opt | `expert_ids` → `sorted_tok`, `expert_offset`, `expert_count`, `inverse` |
| `moe_gather_inv` | `M`(range) `n` `top_k` `dtype` `act`:str/opt | `y_expert`, `expert_w`, `inverse`, `shared`?, `shared_gate`? → `y` |
| `moe_gather` | `M`(range) `n` `top_k` `dtype` `act`:str/opt | `y_expert`, `expert_w`, `sorted_tok`, `shared`?, `shared_gate`? → `y` |
| `row_topk` | `M`(range) `N` `R` `vocab_off`/opt `dtype` | `x` → `idx`, `val`, `pairs`? |
| `row_topk_merge` | `M` `R` `world_size` `dtype` | `gathered` → `idx`, `val` |
| `pair_argmax` | `M`(range) `n` `dtype` | `pairs` → `idx`, `val`? |
| `logit_rerank` | `M`(range) `R` `n_vocab` `n_embd` `vocab_off`/opt `dtype` | `x`, `head`w, `idx_in` → `idx`, `val`, `pairs`? |

**`row_topk` and `row_topk_merge`** are a pair for a vocab-sharded head, whose logits cover only its
own slice. `vocab_off` shifts the reported columns into global ids, and `pairs` writes the same R
entries as `[M, R, 2]` (an int32 id in the first float slot, its value in the second): the plane
`core/sample/sampler.h` holds as `buf_pairs_` and the sampler's vocab-parallel path gathers. An
`all_gather` with `row = R*2` concatenates them into `[M, world_size * R, 2]`, rank r's at
`[r*R, (r+1)*R)`, and `row_topk_merge` reduces that to the global top-R. The merge is exact: a
candidate in the global top-R is in its own shard's top-R, and both stages break ties on the lower
id, so the result equals a top-R over the unsharded plane. A slot a shard could not fill carries id
-1 in `pairs` and is dropped; a sentinel id rather than a -inf value, because a masked logit can be
-inf legitimately.

**`pair_argmax`** is the merge at R = 1, for a vocab-sharded draft head that wants each position's
winner: `idx` gets the id of the largest value among a row's `n` gathered pairs, ties to the lower
id, negative ids skipped, -1 (and -inf in `val`) when nothing is valid, written one element a row at
the output's own pitch (a block drafter's contiguous proposal). It is not `row_topk_merge` at R = 1
because a rank still sends R >= 2 pairs: a row-interleaved `all_gather` moves whole 16-byte words
and a pair is 8 bytes, so `n` is `world_size * R`.

**`logit_rerank`** is the second half of a coarse top-R. A cheap, coarsely quantised draft head
proposes R candidates, and this op rescores those R against the full-precision head the trunk
already carries: a few hundred KB a row instead of a few hundred MB, and an answer that agrees with
the exact argmax on almost every row. A worse draft costs acceptance, never a token, since the
target verifies every proposal. `idx_in` carries global ids as `row_topk` reports them and
`vocab_off` is this rank's first column, so under vocab-parallel drafting each rank rescores the
candidates it owns and the existing `all_gather` + `row_topk_merge` finds the global winner. Ties
break on the lower id, as in the other two.

**`router_topk_scatter`** (device-only) is `router_topk` and `moe_scatter` in one launch. At a
decode shape each is a single 256-thread workgroup ranking about a kilobyte, so each costs the
one-workgroup dispatch floor and almost nothing else. The chain is closed (the sort reads only
`expert_ids`, which the selection writes), so a launch between them buys only a barrier.

- **Two expert counts.** `n_expert` is the router's: it is replicated and every rank picks from
  every expert. `n_local` is this rank's slice, which the sort runs over and which sizes
  `expert_offset` and `expert_count`, as when a calibration run divides Qwen3.8-Flash-Next's experts
  between two ranks. `n_local` absent means the slice is the whole router, as in every deployment
  that gives each rank a slice of every expert.
- **Bounded, with no fallback.** Only `moe_scatter`'s single-workgroup form is fusible
  (`M * top_k <= 1024`); above that the sort is three launches, which keep it stable without an
  atomic deciding the order. That bound covers the decode band (eight sequences at depth 3 and
  top_k 10 is 320 slots). A prefill chunk is past it, where the pair is amortised over real work.
  A caller past the bound gets a refusal, not a silent change of cost.

**`router_gemm_topk_scatter`** (device-only) adds the router's GEMM in front, in the same launch:
`gemm_nt` of `a` `[M, K]` against the router `b` `[n_expert, K]` into `logits`, then
`router_topk_scatter` over those logits. With `gate_b` `[1, K]` and `gate` `[M, 1]` (both or
neither) it is also the shared expert's one-column `gemm_nt` over the same `a`. The selection needs
every logit of a row, so it runs in the workgroup that finishes the GEMM last (a self-resetting
arrival counter per op instance, which libr4d takes in the row's `init`, so it is in the VRAM
budget; a launch without one is refused). It is byte-identical to the launches it replaces: an
element of the skinny bf16 GEMM depends only on its split-K count, which this op takes from the same
law at the same N, and the selection and sort are the same code. Decode only (`M <= 64`, past which
the router is the tiled GEMM), and bounded like the pair. The logits stay an output.

The router GEMM inside it always takes the split the untuned law picks, so a tune-cache entry for
the unfused router's `gemm_nt` does not reach it, and on a tuned install the two forms may differ in
the last bit; no shipped configuration tunes these GEMMs. An architecture asks for this op wherever
it declares the pair, and issues the router and the pair itself where it does not resolve or a pass
is past its rows.

**`moe_scatter`** is a stable counting sort of the (token, slot) pairs by expert. `sorted_tok` holds
the flattened index `token * top_k + slot`. A slot whose expert is outside
`[expert_base, expert_base + n_expert)` is dropped and the tail filled with -1, which is how expert
parallelism is expressed. Its output is what `rad_route_report` hands the heat engine; the histogram
is read back asynchronously and consumed on the next step, so nothing blocks the compute stream
(`spec.md` §5.5).

**`protect`** sorts a few experts last. A layer that keeps some experts at a higher precision serves
them with a second grouped GEMM, and a grouped GEMM takes one contiguous run of the offsets. So
`protect` lists local experts (ascending, at most eight) that the sort places after every other
expert, in the order listed, with the rest closed up: `expert_offset` and `expert_count` are in that
order, the regular experts are `[0, n - p)` of the offsets and the protected ones `[n - p, n)`. A
`moe_gemm` handed the trailing run (offsets that begin above zero) serves those rows and leaves the
ones below untouched.

**`moe_gather`** scatters the sorted expert output back to token order and weights it, summing a
token's slots in ascending slot order. With `shared` and `shared_gate` it also folds the shared
expert in: `y = bf16(shared * act(shared_gate)) + bf16(gathered)`, `scale_rows`' fused
gate-and-add form one op further back, so the bf16 plane between them never exists. Both narrowings
are part of the contract: the unfused pair writes the gather to a bf16 buffer and narrows the
product before the sum, so carrying either in f32 moves the last bit. `act` is `"sigmoid"` or
`"none"` and applies to the scalar, as in `scale_rows`. A shared arm that cannot be width-sharded
runs on one rank, so the op this replaces is a dispatch every other rank waits out inside the
block's all-reduce.

With bf16 expert outputs, `expert_w` may be bf16 or f32. f32 mixture weights stay f32 through
the sum; only the gathered result narrows to bf16, as with bf16 weights.

libr4d's `moe_gather_bf16` takes f32 weights in its column-per-thread kernel, with the self-scan
up to 512 slots (and `top_k` up to 64) and the two prepass launches above. The wave-per-token
form past 512 slots reads bf16 weights only, so f32 weights stay on the column-per-thread kernel
at every size.

### Quantised experts and the row gate

| op | params | operands |
|---|---|---|
| `moe_gemm_q` | `M`(range) `N` `K` `n_expert` `top_k` `group` `a_order` `dtype` `N_odd`/opt `K_odd`/opt `parts`/opt | `a`, `a_scale`?, `w`wt, `w_scale`wt, `sorted_tok`, `expert_offset`, `w_odd`wt, `w_odd_scale`wt → `y` |
| `moe_gemm_q_mod4` | `M`(range) `N` `K` `n_expert` `top_k` `group` `a_order` `dtype` `N_1`/opt `K_1`/opt `N_2`/opt `K_2`/opt `N_3`/opt `K_3`/opt `parts`/opt | `a`, `a_scale`?, `w`wt, `w_scale`wt, `sorted_tok`, `expert_offset`, `w_1`wt, `w_1_scale`wt, `w_2`wt, `w_2_scale`wt, `w_3`wt, `w_3_scale`wt → `y` |
| `scale_rows` | `M`(range) `n` `act` `dtype` | `x`, `s`, `add`? → `y` |

The arithmetic is the same whether a kernel is handed the tables or the stacked form
([Weight operands and weight tables](#weight-operands-and-weight-tables)).

**`moe_gemm_q`'s experts are two tables by parity.** `w`/`w_scale` hold the even-numbered experts at
`[N, K]` and `w_odd`/`w_odd_scale` the odd ones at `[N_odd, K_odd]` (defaulting to `N`, `K`); expert
`e` is entry `e / 2` of its class. Every entry of one table shares a geometry, and a rank that holds
an uneven slice of every expert's width (three 128-column blocks of one parity's experts and two of
the other's, which is how a five-block expert splits evenly across two ranks on average) has two.
`parts` stacked parts run along both the weight rows and the output columns (two for a fused
gate/up); the output is `max(N, N_odd)` wide, and a narrower class fills the leading columns of each
output part and writes zeros after them, so the gated quantiser behind it reads one layout for every
row. The weight list is the even class's codes, its scales, then the odd class's codes and scales.

**`router_sigmoid_topk`** is single-group correction-bias routing, all in f32. The scores are
`p = sigmoid(logits)`; the selection takes the `top_k` largest `p + bias`, ties to the lower
expert, and the weights are the selected `p`, never the biased ranks. With `norm` 1 a weight is
`p * (route_scale / (sum + 1e-20))` over the row's selected `p`, with `norm` 0 `p * route_scale`;
`route_scale` is positive and finite. A rank that is NaN or `-inf` is never selected, and a row
with fewer than `top_k` selectable experts has no defined selection (libref refuses it).
Grouped selection (`n_group > 1`) is not this op. GLM-5.3 routes 8 of 288 experts in one group
with `route_scale` 2.5.

libr4d's `router_sigmoid_topk_f32` takes up to 512 experts and `top_k` up to 32, f32 logits, bias
and weights, i32 ids, pitched rows, `M` up to 65535. It selects in one wave per row: each lane
holds its experts as order-preserving keys (`-0` as `+0`, ties to the lower expert), and round `j`
takes the largest key below round `j-1`'s winner. A row without NaN ranks gives the bytes of a
serial scan with a strict `>`. A NaN rank is never selected, and a row with fewer than `top_k`
other ranks names expert 0 in its empty slots, weighted by expert 0's `p`.

**`router_sigmoid_topk_scatter`** (device-only) is `router_sigmoid_topk` and `moe_scatter` over its
`ids` in one launch, and with `gate` `[M, 1]` also the `fill` of an ungated shared arm's gate with
1 (GLM-5.3's shared expert has no gate, and the gather reads one). `n_expert`, `n_local`,
`expert_base` and `protect` are as in `router_topk_scatter`. Every output is the bytes the
separate launches write, including a row with fewer than `top_k` non-NaN ranks (NaN logits),
whose empty slots carry expert 0 into the sort as `ids` does (`r4d_selftest`). Bounded like
`router_topk_scatter` (`M * top_k <= 1024`, one workgroup) and refused past it. With `inverse` the
sort is `moe_scatter_inv`'s.

**`moe_scatter_inv`** and **`moe_gather_inv`** (device-only) are `moe_scatter` and `moe_gather`
with the gather's slot table passed between them. `inverse` `[M * top_k]` holds, for slot
`token * top_k + slot`, its row in `sorted_tok`, or -1 when the slot was dropped; `moe_gather`
builds exactly that table from `sorted_tok` before it sums (a scan of the sort per token up to 512
slots, two prepass launches above), and `moe_gather_inv` reads it instead. They are ops of their
own because `moe_scatter`'s and `moe_gather`'s operand lists are the shared vocabulary's.
`moe_scatter_inv` is the same sort (one launch up to 1024 slots, three above, `n_expert` up to
512), and `moe_gather_inv` runs the kernel `moe_gather` picks at the same size, bf16 or f32
weights; both are byte-identical to the pair they replace, at GLM's shape with f32 routing weights
and the shared arm folded in (`r4d_selftest`).

At `dtype=mxfp4a8`, `w` and `w_odd` are OCP e2m1 codes and `w_scale` and `w_odd_scale` hold one
E8M0 exponent per (row, 32 of K), typed `e8m0` or raw `u8`; either way a byte is an exponent, not
an integer multiplier. `a` and `a_scale` are `quant_act_fp8`'s pair, and `group` is that
activation's scale group, independent of the weight's 32-wide grid.

libr4d's `moe_gemm_mxfp4a8` keeps the canonical bytes, including pointer tables of device or
device-mapped host addresses: a wave decodes the packed e2m1 words into exact E4M3 bytes and runs
fp8 WMMA, folding each 32-wide E8M0 exponent into the f32 accumulator. Per 128 of K, each 32-wide
group is a fresh WMMA pair, folded as raw * activation scale * E8M0 scale. It serves `group` 128,
`K` a multiple of 128, 2 to 512 experts, `top_k` up to 32, token or sorted activations, `parts`,
and an odd class narrower in `N` or `K`, with `M` up to 32767; there is no `moe_gemm_q_mod4` row
at this dtype. The form follows the routed rows. The expert-major form, a block per (column block,
expert) that reads each expert slice once, takes up to 511 rows, and at GLM's TP8 slices (288
experts, top 8) gate_up up to 768 and down up to 6143. At those slices the wide prefill form (P64:
jobs of 32 or 64 rows of one expert) takes gate_up from 769 rows and down from 5760, ahead of the
expert-major band. At GLM's TP4 slices the GLM prefill form (jobs of 32 rows of one expert, its
loads a stage ahead) takes from 512 rows; elsewhere the row groups (16 sorted rows a block) serve.
Every form gives the bytes of one per-element sequence, restated on the host by
`tests/r4d_emu/test_moemx.cpp`.

**`moe_gemm_q_mod4`** is the same GEMM over four classes by the expert's number mod 4. `w`/`w_scale`
hold class 0 at `[N, K]` and `w_q`/`w_q_scale` class `q` at `[N_q, K_q]` (defaulting to `N`, `K`);
expert `e` is entry `e / 4` of class `e % 4`, so class `q` holds `ceil((n_expert - q) / 4)` of them.
Four ranks splitting a five-block expert cannot share the extra block evenly over two classes, but
over four they can: rank `r` takes two blocks of class `r` and one of each other, five of every four
experts on every rank. It is a separate op, not a parameter on `moe_gemm_q`, because the declaration
divides a table op's weights over its table operands by count alone, so the operand list is the
class count. Everything else (`parts`, the zero fill, `a_order`) is `moe_gemm_q`'s, and the weight
list is class 0's codes and scales, then class 1's, 2's and 3's.

**The codebook forms** (`w4nla8h`, `w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h`; `docs/MOE-W4.md`) hold
u4 or u5 codes into an f32 table of 16 or 32 levels, declared as one view over the planes
`codes,table`, and a scale per (row, group of K): bf16 per 128, or E4M3 per 64 or 32 declared with
its fixed f32 level as `scale,scale.1`. Element (n, k) of an expert is `table[code] * scale`, times
`scale.1` where there is one. The encoding's `fwht128` is in the stored weight and in the
activation `had_quant_act_fp8` writes, so the GEMM applies none. On the host, a view of several
planes is stored in the joined form of `abi/rad_planes.h`; libref's and libavx's hooks accept
exactly the planes `dtype` names (code width and table length, scale dtype and group, `fwht128`)
and refuse any other selection at declare with `RAD_E_DTYPE`. libref computes the form as
`gemm_nt_q` does: per weight group the sub-sum of table values times activation codes, then
`(sub * sa) * sb` with `sb` the scale times its level. The logical form the oracle and rad-kbench
pass (table entries for codes, scale products for scales) goes through the same chain, bit for
bit. libr4d sums each 64- or 32-wide group under its scale before the activation's scale per 128,
and libavx folds both scales into the operands before one sum over K; both differ from libref in
the last bits only.

**`moe_gemm_q` is a separate op, not a `dtype` on `moe_gemm`**, for the reason `gemm_nt_q` is
separate from `gemm_nt`: the operand list differs and operands are positional. The group sub-sum is
accumulated from the raw codes and scaled once per group, as `gemm_nt_q` does; the two
references must agree in the last bits or `rad-kbench` blames a kernel for a choice made in
`libref`.

**`moe_gemm`** is what a model served at bf16 runs, and libr4d serves it at `dtype` bf16: one weight
table of the checkpoint's own row-major `[N, K]` planes, read with no layout hook, so a container
stores an expert as the safetensors file holds it. It has no parity classes and no `parts` (those
exist because a quantised expert's width must fall on its scale blocks), and its rows go through
the same run grouping and the same three forms as `moe_gemm_q`'s kernels, byte-identical to each
other.

**`scale_rows`** is the per-token scalar gate: `y[m, i] = x[m, i] * act(s[m])`, one scalar per row
broadcast across it. `mul` broadcasts one row down the rows, and one op with both rules would be
ambiguous when `rows == n`. `act` is `"sigmoid"` or `"none"` and applies to the scalar. Qwen3.5-MoE's
`shared_expert_gate` is a `[1, n_embd]` projection whose one output per token gates the shared
expert's contribution, and this is that multiply. With `add` present it is
`y = bf16(x * act(s)) + add`, the product narrowed before the sum as the unfused pair stores it.

### `a_order`

A routed block runs two `moe_gemm_q` ops, and they disagree about what a row of `a` is, so this is
a parameter, not a convention.

- The **gate_up** projection reads the block's normed input, one row per token. Sorted row `i`
  belongs to the pair `sorted_tok[i] = token * top_k + slot`, so it gathers `a[sorted_tok[i] / top_k]`:
  `a_order = "token"`, the default.
- The **down** projection reads gate_up's output through the gated quantiser, already one row per
  (token, slot) in the scatter's order, so row `i` is row `i`: `a_order = "sorted"`.

`a_scale` has a row for every row `a` has, either way.

It is stated, not derived from the row count (`M` against `M * top_k`), because no harness would
catch a wrong derivation: a kernel and a reference that both gather agree on every case, and
`rad-kbench` sizes each operand from the op's own shape hook, so a hook describing `a` at the extent
the gather wants builds a self-consistent, wrong case. Served, the down projection would hand every
slot of a token the first slot's intermediate, and the model would answer fluently and get facts
wrong.

## Linear attention (gated delta net)

| op | params | operands |
|---|---|---|
| `kda_update` | `M`(range) `heads` `head_k` `head_v` `lower_bound`:f64 `scale`:f64 `l2_eps`:f64 `mode` `dtype` `state_form`/opt | `q`, `k`, `v`, `a`, `b`, `A_log`w, `dt_bias`w, `state`:inout, `state_idx`, `cu`, `num_accepted`?, `has_init`? → `o` |
| `gdn_conv_prep` | `M`(range) `head_k` `head_v` `chunk`/der `conv_width` `l2_eps`:f64/opt `softplus_thr`:f64/opt | `x`, `w`w, `b`w?, `A_log`w, `dt_bias`w, `conv_state`:inout, `cu`, `a`?, `b_gate`?, `cache_idx`?, `has_init`? → `q`,`k`,`v`,`g`,`beta` |
| `gdn_conv_update` | `q_len`(range) `head_k` `head_v` `conv_width` `max_query_len`/opt | `x`, `w`w, `b`w?, `conv_state`:inout, `state_idx`, `num_accepted`, `cu` → `q`,`k`,`v` |
| `gdn_kkt_solve` | `M`(range) `head_k` `chunk`/der | `k`, `beta`, `g`, `cu` → `A` |
| `gdn_chunk_scan` | `M`(range) `head_k` `head_v` `chunk`/der `scale`:f64/opt `state_fp16`/opt | `q`,`k`,`v`,`A`,`g`,`beta`,`h0`?,`cu` → `o`, `ht`?; `state_idx`? |
| `gdn_recurrent_update` | `q_len`(range) `head_k` `head_v` `n_head_v`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`:str/opt `state_fp16`/opt `state_form`:str/opt | `q`,`k`,`v`,`a`,`b`,`A_log`w,`dt_bias`w,`state`:inout,`state_idx`,`num_accepted`,`cu`,`z`?,`norm_w`w? → `o`,`o_q`?,`o_scale`? |
| `gdn_conv_recurrent_update` | `q_len`(range) `head_k` `head_v` `conv_width` `n_head_v`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`:str/opt `state_fp16`/opt `state_form`:str/opt | `x`, `conv_w`w, `conv_b`w?, `conv_state`:inout, `conv_idx`, `a`,`b`,`A_log`w,`dt_bias`w,`state`:inout,`state_idx`,`num_accepted`,`cu`,`z`,`norm_w`w → `o`,`o_q`?,`o_scale`? |
| `gdn_gated_rmsnorm` | `M`(range) `channels` `eps`:f64 `act`:str | `x`, `z`, `w`w → `o` |

**`kda_update` is Kimi delta attention with per-channel decay.** Per token and head it
L2-normalizes `q` and `k` with `l2_eps`, scales `q` by `scale`, and forms the decay
`exp(lower_bound * sigmoid(exp(A_log[h]) * (a[d] + dt_bias[h, d])))` per K channel `d`
(`lower_bound` negative and finite) and `beta = sigmoid(b[h])`. For each value row `e` of the
state it decays the row by channel, computes `delta = beta * (v[e] - dot(S[e], k))`, adds
`delta * k`, and writes `o[e] = dot(S[e], q)`. `state` is f32 `[slots, heads, head_v, head_k]`;
`q`, `k`, `v` and `a` may be `[M, heads, width]` or flattened `[M, heads*width]`, and `b` is one
value per (token, head). `o` excludes the model's gated RMS norm. `cu` holds absolute token
offsets, so a call can cover a slice of a mixed batch. `mode=verify` reads column
`num_accepted - 1` of the sequence's `state_idx` row (`num_accepted` absent is 1) and writes the
state after token `t` of the window to column `t`; `mode=prefill` reads column zero and writes only
the final state there. `has_init` 0 starts from a zero state, a negative read slot gives zero
output, and a negative write slot is skipped.

`state_form` is `gdn_recurrent_update`'s (below): `per_candidate` when absent, and with `anchor`
one committed state and one scratch slot per sequence. Under `anchor`, column zero of `state_idx`
is the committed state and column `width/2` the scratch; the two must differ, and both forms need
`head_k = head_v = 128`. The scratch holds, per head, one f32 replay count and then one record of
385 f32 per token: the normalized key, the raw value, the per-channel decay and beta, 32 records
at most, which fits one 128 x 128 head block. `mode=verify` first replays the previous window's
accepted prefix (`num_accepted` records, at most the stored count) onto the committed state and
stores it back, then runs the window from it, writing each token's record and the window length
as the new count; the committed state does not advance past the accepted prefix, and a window has
at most 32 tokens. `mode=prefill` commits its final state and sets the count to zero. A zeroed
scratch slot replays nothing.

libr4d's `kda_update_k128_v128_bf16` takes `head_k = head_v = 128`, 1 to 65535 heads, bf16
activations, f32 `A_log`, `dt_bias` and state, both modes and both state forms. A wave holds four
value rows of a head, loads the next token's operands while the current one reduces and runs the
sequence's tokens in order (held to libref by `r4d_selftest` and `tests/r4d_emu/test_kda.cpp`).
Under `anchor`, `verify` replays the accepted prefix in a separate dispatch before the window, so
no value row overwrites factors another is still reading. A sequence whose window or slots are
out of bounds is skipped on the device, not refused.

**Speculative rollback is part of the schema** (`spec.md` §10). The recurrent state and the conv
window have already absorbed rejected tokens, so the state cache is a rolling window of
`conv_width-1 + n_spec` entries read at the slot the last accepted token left, and a rejection is a
change of read offset, not a recompute.

**`num_accepted` is required and one-based.** It counts the tokens the previous step committed: the
one it was given plus however many of its drafts survived. 1 is an ordinary decode step that drafted
nothing, and 0 is not a value either kernel can read (`gdn_conv_update` reads its window at
`num_accepted - 1`). A non-speculative decode passes all ones. It is one of the two exceptions to
the union rule: an op that lets the caller omit `num_accepted` has an optional rollback contract.

**`cu` is required by every gdn kernel, decode included**, the other exception. `gdn_conv_update`
and `gdn_recurrent_update` read `cu[n]` and `cu[n+1]` before any null test, and the sequence count
is `cu`'s length minus one; nothing else in the argument list carries it. A caller working from a
schema that made `cu` optional would pass a null and fault rather than be refused.

**`state_idx` is `[n_seq, W]`, and W is the group's, not the op's.** Column 0 is the sequence's
state. A kernel that needs somewhere to put what it cannot commit yet may ask for more columns, and
those are its scratch: the caller stages the row, zeroes every slot it names when the sequence is
admitted, zeroes the scratch again after any step that ran a different path over the sequence (a
chunked prefill commits the state directly and writes no scratch), and otherwise does not look
inside. `gdn_chunk_scan` takes the same row and reads column 0.

**`state_form` says what the state cache means.** `per_candidate`, the default, writes one full
state per candidate token and needs `1 + n_spec` distinct slots in `state_idx`. `anchor` keeps one
committed state per sequence and, beside it, the few hundred floats per head each token's rank-1
update needs; the next step replays the accepted prefix onto the state out of that scratch slot. That
is two slots whatever the draft depth, against `1 + n_spec` full states (1.3 GiB a sequence on
Qwen3.8-27B), and less write traffic. A zeroed scratch block replays as the identity, so "zero it
and pass `num_accepted` 1" is the correct handover out of a prefill.

The two forms produce the same `o` from a different `state`, so an implementation handed the other
form computes a correct output and disagrees with a checker on every element of the cache. The index
row's width does not say which form is meant, so it is declared, not inferred: the engine passes
`anchor` when the deployment speculates and `per_candidate` otherwise (the condition that sizes the
state pool), and an implementation that serves one form refuses the other.

libref serves both, the anchor form in libr4d's cache layout (`libref/ref_gdn.cpp`,
`AnchorLayout`): `state_idx` has six columns, the anchor in 0-2 and the factor block in 3-5. A paged
caller's window may shift by -1, 0 or +1 pages a step, so the advanced anchor is published at column
2 and each factor copy is stamped with its column; the next step reads the stamp at column 4 and the
anchor at the column it implies. The factors are each token's normalised k and its decay exponent
and beta (once per 32-row group of `head_v`, each group's rows replaying from their own copy, as
libr4d reads them), and v, at most 16 tokens a step, with the stamp just past them. A
head whose `[head_v, head_k]` block cannot hold 16 tokens' factors (under 64 x 64) holds as many as
fit, so a 16 x 16 head speculates at most 6 tokens deep. With an f32 state the replayed state is bit
for bit the per-candidate one, so `o` matches the per-candidate form exactly. libavx serves both
forms in the same layout, its replay bit for bit its own per-candidate state (rad-avx-check runs the
two over a four-step schedule). The end-to-end check is that greedy speculation emits the same bytes
as greedy without it.

**`gdn_conv_recurrent_update`** (`gdn_conv_update` then `gdn_recurrent_update`, in one launch)
computes the convolution where the recurrence reads it, so `q`, `k` and `v` are never written. The
operands are the convolution's inputs (`conv_idx` is its `state_idx`) followed by the recurrence's
without those three; the parameters are the union of the two ops' under their own names, and both
ops' rollback and state contracts hold unchanged. `z` and `norm_w` are required, because the fusion
serves the decode shape, where the fused row norm already puts one workgroup on each (sequence,
value head). It is byte-identical to the pair. libr4d rewrites a key head's conv state once every
value head reading it has read it, by an arrival count, so no workgroup waits on another.

**The output gate `z` may be a column slice**, on both `gdn_recurrent_update` and
`gdn_gated_rmsnorm`. The fp8 delta-net block projects `[q|k|v|z]` in one GEMM
(`arch/common/rad_block_gdn_fp8.h`), so the gate is the right-hand `n_head_v * head_v` columns of a
wider row and its `(token, head)` rows are not evenly spaced: heads are `head_v` apart inside a
token and the next token is the slice's own stride away. An implementation that reads one pitch off
`z` walks into the next token after `n_head_v` rows and is wrong without going out of bounds. A
caller may hand over `[tokens, heads, head_v]`, a flat `[tokens, heads*head_v]` buffer of its own,
or that slice; only the slice needs the token stride carried separately.

## Collectives

| op | params | operands |
|---|---|---|
| `all_reduce` | `world_size` `numel`(range) `dtype` `exact` `min_bytes`/opt `hops`/opt `overlap`/opt | `x`:inout, `y`:out? |
| `all_gather` | `world_size` `numel` `dtype` `row`/opt | `x` → `y` |

`exact` separates the exact sum from a lossy wire, and a caller must state it: a request that does
not accept a lossy wire gets no kernel rather than the quantised one.

`hops` (1 or 2) is a caller's request for one-shot or two-shot; libref accepts and ignores it, and
no libr4d row reads it. libr4d chooses by message size where a width has both: its two-shot rows
start above 524288 elements (8192 at eight ranks), a band edge of the ranged `numel`. Every
all-reduce kernel serves `y` absent, in place. `overlap` 1 says the caller issues the collective on
its second lane, beside compute; libr4d then picks rows with a narrow launch (8 blocks of 256
threads at two ranks, 4 of 1024 for the four-rank two-shots, exact and tiered-int8) that compute the
same bits and leave the CUs to the kernels running beside them. libref ignores it.

libr4d serves these at 2, 3, 4 and 8 ranks:

- Two ranks have their own kernels (one flag word a block), the rotated 6-bit lossy wire and the
  fused norm-and-reduce forms below.
- The wider widths share one template on the world size. 3, 4 and 8 have both one-shot and
  two-shot (the three-rank two-shot shards unevenly), and 3 and 4 have a tiered int8 lossy two-shot.
  8 has no lossy wire.
- `all_gather` at those widths rides the same instance as the wide `all_reduce` (`ar_gather_3rank`,
  `_4rank`, `_8rank`).

With `row`, `all_gather`'s `y` may be the leading `world_size * row` columns of a wider buffer, and
`x` a `row`-wide window of a wider one (column views whose row pitch is whole 16-byte words): rows
are read and land at those pitches, and the rest of each row is left as it was. A linear split by
rows whose rank shares cut its scale blocks has each rank compute the whole blocks covering its
share and hand over the share's window, gathered into a tight output (arch/common's DFlash2 fc).

`min_bytes` is the byte floor under which a lossy row serves the message exactly. It is a parameter,
not a library constant, because it is a property of the link: compression pays only once the
transfer is bandwidth-bound, and for a decode step's few residual rows the rotate, pack and unpack
cost more than the bytes they remove. A row selected for `exact=0` therefore decides the row once,
at declare, and the wire per launch. Absent, the plugin uses its own default.

`all_gather.row` is one row of a rank's contribution, in elements. Absent or `0`, `y` is the plain
concatenation, rank `r`'s slice at `r * numel`. Present, the placement is row-interleaved: `x` is
`[rows][row]` and `y` is `[rows][world_size][row]`, so a vocab-parallel candidate merge reads one
row of gathered candidates per sampled position at a single row stride.

**`-ffp-contract=off` is mandatory for any quantised cross-rank op.** Two ranks see the two
dequantised products in opposite order, and contracting either into an FMA makes them disagree by
~1 ULP, which breaks the replicated-state invariant. `cmake/RadAddPlugin.cmake` sets it for all
device code.

### Collectives with a norm (device-only)

| op | params | operands |
|---|---|---|
| `ar_rmsnorm_quant_fp8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`:inout, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `ar_rmsnorm_post_quant_fp8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`:inout, `residual`:inout, `w`w → `q`, `scale`, `out_bf16`?; `w_post`w |
| `ar_rmsnorm_had_quant_i8g` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`:inout, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `ar_rmsnorm_quant_i8` | `M` `n` `eps`:f64 `world_size` `dtype` `wadd`:f64/opt | `x`:inout, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`?; `w_post`w?, `scalar`w? |
| `ar_hc_write` | `M`(range) `n` `hc` `world_size` `dtype` `wire`/opt | `y`:inout, `inj`, `h`:inout |
| `ar_gather_hc_write` | `M`(range) `n` `hc` `top_k` `world_size` `dtype` `act`/opt `wire`/opt | `y_expert`, `expert_w`, `sorted_tok`, `shared`?, `shared_gate`?, `y`:inout, `inj`, `h`:inout |
| `ar_ln_had_quant_i8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gdn_gated_norm_had_quant_i8` | `M` `n` `head_dim` `group` `eps`:f64 `dtype` `zgrp`/opt `zblk`/opt | `x`, `z`, `w`w → `q`, `scale` |

`wire` is the payload width on the link: 0 is exact bf16, 6 the rotated six-bit form. It is a
parameter for the same reason as `min_bytes`. `x` is inout on `ar_rmsnorm_quant_fp8`: it is both
the message and the input, and the reduced sum lands in it.
`ar_rmsnorm_had_quant_i8g` is `ar_rmsnorm_quant_fp8` with `rmsnorm_had_quant_i8g`'s quantiser:
`x` and the residual are the same bytes, and the codes are int8 for a weight at `i8a8h`.

The norm-carrying forms are two-rank only. `ar_hc_write` and `ar_gather_hc_write` also have an
exact-wire four-rank form. At other widths the arch blocks issue the plain ops.

`ar_hc_write` is `all_reduce` of `y` followed by `hc_write` of the reduced `y`, for a block whose
output feeds a gated residual rather than a norm; `y` is inout as `x` is above. It serves both
wires, one op per wire: a caller declares the rotated form beside the exact one and issues whichever
the step's message goes out on, and `wire` 6 is 6 at every size. A caller states `wire` 0 on the
exact form rather than omitting it: a row that serves one wire says so with a constraint on `wire`,
and a constraint on an absent key does not hold.

`ar_gather_hc_write` is `moe_gather` followed by `ar_hc_write`: the gather's output is the message,
so the gather half writes `y` before it is reduced in place, and the five gather operands and `act`
mean what they mean on `moe_gather`. `wire` is `ar_hc_write`'s. On either wire it is byte-identical
to the three kernels it replaces.

`ar_rmsnorm_quant_i8` is the exact `all_reduce` of `x` followed by `rmsnorm_quant_i8`'s junction
over the reduced sum, at two ranks and at four. At four it is the exact one-shot's transport with
the junction as its epilogue: a block owns whole rows, and the fold is element by element in the
norm's partition, each element the one-shot's arithmetic (fp32 from zero, ascending rank order,
one rounding to bf16), so it is byte-identical to `all_reduce` + `rmsnorm_quant_i8`. A one-shot
sends (ranks - 1) times the message a rank, so it is for decode heights; above them a caller
issues the pair, whose `all_reduce` goes two-shot.

From 5 to 32 rows (the `tail` axis, on by default) the four-rank row takes a third form: the
standalone exact two-shot over words, on the blocks the standalone `all_reduce` takes for that
message, with the junction as its tail -- one more block a row, which waits on a card-local count
of the transport blocks (no cross-card handshake), acquires, and runs `rmsnorm_quant_i8`'s row at
its own width. The bytes are the pair's; the launch the junction would take is saved.

## Vocabulary edges

| op | params | operands |
|---|---|---|
| `embed_lookup` | `M`(range) `n_embd` `n_vocab` `dtype` `vocab_offset`/opt `wscale`:f64/opt `mult`:f64/opt | `tokens`, `wte`w → `x` |
| `embed_lookup_q` | `M`(range) `n_embd` `n_vocab` `dtype` `vocab_offset`/opt `wscale`:f64/opt | `tokens`, `wte`w, `scale`w? → `x`, `ahead`? |
| `logits_gemm` | `M`(range) `n_vocab` `n_embd` `dtype` | `x`, `lm_head`w → `logits` |
| `ngram_ids` | `M`(range) `heads` `ngram` `eos` | `tok`, `state`:inout, `cu`, `mult`w, `vocab_sizes`w, `offsets`w, `cache_idx`?, `has_init`?, `num_accepted`? → `ids`, `ahead_ids`? |
| `ple_gate` | `M`(range) `n` `hc` `eps`:f64 `dtype` `wadd`:f64/opt `gate_eps`:f64/opt | `k`, `q`, `v`, `w_key`w, `w_query`w, `w_conv`w → `gv`, `gvn` |
| `ple_conv` | `M`(range) `n` `width` `dilation` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `resid`?, `cache_idx`?, `has_init`?, `num_accepted`? → `y` |
| `causal_conv1d` | `M`(range) `n` `width` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `cache_idx`?, `has_init`?, `num_accepted`? → `y` |

`logits_gemm` is separate from `gemm_nt` because `lm_head` is its own access class
(`RAD_ACCESS_VOCAB`) and because the sampler fuses against it: under the distributed top-k a rank
never materialises a full logits row (`spec.md` §9, §13). `M` is one row per sampled position,
`gather_rows` having run first. A quantised head (block fp8, int8 per row and 128 of K) reaches a
host kernel joined (Weights of several planes, above).

### `gather_rows`

| op | params | operands |
|---|---|---|
| `gather_rows` | `M`(range) `n` `dtype` | `x`, `idx` → `y` |

`y[i, :] = x[idx[i], :]`; a negative index writes a zero row. Without it a plugin computes logits for
every token of a prefill chunk instead of the last one per sequence: `n_tok / n_seq` times the work,
and at `max_tok` 8192 against a 248K vocab the logits buffer alone is 8 GiB.

### `scatter_rows`

| op | params | operands |
|---|---|---|
| `scatter_rows` | `M`(range) `n` `dtype` | `v`, `idx`, `x`:inout |

`x[idx[i], :] = v[i, :]`, the mirror of `gather_rows`. `x` is inout because the write is partial:
rows no index names keep what they held, so it can target a pool other sequences also use. Two
rules differ from the gather, because the destination is live memory:

- **A negative index writes nothing.** `gather_rows` reads a zero row for one (the padding a short
  prefill chunk leaves); here a zero fill would be data loss.
- **Indices must be distinct**, declared with `RAD_OPD_F_IDX_UNIQUE`. Two rows landing on one
  destination have no defined winner on a parallel machine, so the reference and a kernel could
  legitimately disagree.

The caller is the KV tier traffic (`core/mem/kvtier.h`), where a row is a few hundred bytes and `M`
separate transfers would be all dispatch cost. A paged pool is layer-major, so within one layer a
group's blocks are dense rows of `layer_bytes_per_block()`: packing any set of them is a gather and
restoring them a scatter, one launch per (group, layer) however many blocks move.

At `dtype` u8 the rows are opaque bytes, and libr4d supplies that width as its own kernel pair
rather than widening the bf16 one. What moves is whatever the attention kernel wrote. Calling it
bf16 would matter: `libref` is the oracle for every op here and its bf16 path loads through float,
which need not return a NaN payload unchanged. At u8 every value is 0..255, which float carries
exactly, so oracle and kernel agree bit for bit on any payload.

### `embed_lookup`'s `mult`

`mult` (optional, default 1) multiplies the rows that are found, `x[m] = wte[id] * mult`, rounded
once to `x`'s dtype; a missed row (negative padding or another rank's token) writes zero. Gemma
scales its embedding by `sqrt(hidden)` held in a bf16 buffer, so the plugin passes that value
rounded to bf16 (22.625 at a hidden of 512, not 22.6274) and the product is the reference's bit for
bit. Scaling at the lookup leaves the table plain and tied to the head, as the checkpoint stores
it.

`wscale` is a different thing: the scale `embed_lookup_q`'s layout quantises the stored table with,
read back as the dequant. `embed_lookup` reads none, and `embed_lookup_q` takes no `mult`.

### `embed_lookup`'s `vocab_offset`

`vocab_offset` (optional, default 0) is the first vocabulary row this rank holds. With it, `n_vocab`
is this rank's row count (as it is for `logits_gemm`) and the table covers global ids
`[vocab_offset, vocab_offset + n_vocab)`. Token ids stay global: a rank subtracts the offset and
writes a zero row for an id that belongs to another rank, as it does for the negative padding id, so
summing the ranks reconstructs `x`. Every token lands on exactly one rank, so the `all_reduce` that
follows is declared `exact` rather than taking the model's wire setting: it sums disjoint rows, and
a lossy wire would lose precision on values that are already final.

Without it the embedding would be replicated while `lm_head` is row-sharded. The table is bf16
even in an fp8 checkpoint, so replicating a 248320 x 5120 table costs 2.37 GiB on every card that
the split gives to the KV cache instead.

Under a split the kernel cannot tell a garbage id from another rank's token, so "an id at or above
`n_vocab` fails the launch" holds only where the rank holds the whole vocabulary (world size 1, not
merely `vocab_offset` 0: rank 0 of a split has offset 0 and sees ids past its slice on most steps).
Bounding ids is the caller's job, and the tokeniser and sampler are already bounded.

### `embed_lookup_q`, the n-gram gather

A gather from a table quantised with one scale for all of it: `x[m] = wte[id] * scale[0]`, an absent
`scale` being 1. Padding and bounds behave as `embed_lookup`'s, except that an id past the table
fails the launch in the host kernels and writes a zero row in the device one, which cannot read its
ids before it runs.

One scale suffices because E4M3 carries its own exponent: on Qwen3.8-Flash-Next's table a row's
dynamic range is 2.3x and a per-row scale would improve the error by only 2.9%. The FP8 checkpoint
ships it this way, E4M3 rows with a single bf16 `weight_scale`.

It is a separate op, not a `dtype` on `embed_lookup`, because the operand list differs and
operands are positional; an optional scale operand on `embed_lookup` would also move `x` from slot
2 to slot 3 for every caller.

Where it runs is where `--ngram-placement` puts the table: the planner runs an op on the host when
every weight operand of it is `Site::Host`, and takes that from the table's placement. The table is
51.2 GB and is never staged a unit at a time (moving a whole unit to use 5 KB is the wrong shape),
so it is read where it lies:

| placement | the table | the gather |
|---|---|---|
| `disk` | the container's mapping | libavx's host kernel |
| `ram`, libr4d loaded | a pinned copy every card maps at its host address | libr4d's device kernel, over the link |
| `ram`, no device kernel | a locked copy (`core/format/table_copy.h`) | libavx's host kernel |
| `vram` | the card, every rank holding all of it | libr4d's device kernel |

`hipHostRegister` on the container's mapping is not an option: it would force the whole file
resident.

**Host gather.** libavx keeps the rows a gather has touched in a row cache of its own and reads a
miss from the file with `O_DIRECT` (`libavx/avx_ngram.cpp`); the page cache would map a 4 KiB page
per 160-byte row. Reads keep the alignment the filesystem reports for direct I/O, or its block size
where it reports none (as btrfs does); one read of the table proves it before the first row is
served. A filesystem without direct I/O cannot hold a container with this table on `disk`.

**Device gather.** libr4d's `embed_lookup_q_bf16` handles four entries a thread, decodes each E4M3
code exactly, multiplies by the scale in f32 and rounds to nearest even: the value libavx's lookup
table holds, bit for bit. It reads the ids and writes `x` in the host arena, where the architecture
declares them for the host gather, so the step has no host op left in it.

The table must be a weight operand for the host site to be reachable at all: `gather_rows` takes a
plain input and would never be placed there. The host kernel lives in a production plugin (libavx)
because `rad_gate_reference_kernels` refuses to serve on libref.

### `ngram_ids`

Qwen4-Exp's Per-Layer Embedding gives one layer a second embedding table addressed by a hash of the
last `ngram` tokens. Qwen3.8-Flash-Next's has 320,001,536 rows of 160 (102 GB at bf16, 51 of the
model's 180 billion parameters), read sixteen rows a token. `ngram_ids` computes those row ids:

    mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]
    ids[h]   = remainder(mixed(b+1), vocab_sizes[h]) + offsets[h]

where the `heads` are `ngram - 1` equal blocks and block `b` uses the `(b+2)`-gram. `t_s` is the
token `s` positions back, or the EOS token if the shift would cross one, and once a shift falls
back to EOS every deeper shift does too. `mult`, `vocab_sizes` and `offsets` are weights because the
checkpoint carries them; re-deriving them from the reference's seed and prime search would be a
second source.

**`state` is a rolling window of committed token ids**, `ngram - 1 + n_spec` deep, read at
`num_accepted - 1` and rewritten in `ple_conv`'s layouts (below), including the two that
`num_accepted`'s presence chooses between. A cold window is EOS, not zero, since zero is a real
token id. It is a state rather than a plain input because the contract is already paid for: the
dilated convolution downstream carries a window with this rollback contract, so PLE needs
`num_accepted`, `cache_idx` and `has_init` regardless, while nothing in the engine derives "the ids
before this step" (`DeriveInput` carries positions and cu_seqlens and no token ids, and an input
would need a new core derivation and a wider `DeriveFn`). `RAD_KV_CONV` is already "per-sequence, a
rolling window of width-1 + num_spec entries", and `conv_state_index` already derives its read
cursor.

The gather is `embed_lookup_q`, not `gather_rows`, because the table must be a weight operand for
`Site::Host` to be reachable.

**`ahead_ids` and `ahead` fetch the next chunk's rows a step early.** A prefill chunk gathers `heads`
rows a token, and with the table on `disk` and text the row cache has not seen, most are random
reads of the container file while the card idles behind the host-site op. The next chunk of the
prompt is already known (`RadBatch::n_ahead`, stored in `token_ids` after the step's own tokens), so:

- `ngram_ids` with `ahead_ids` present reads that many tokens past the step's in `tok`, as a
  continuation of the last sequence, and writes their ids there: the same ids the step that runs
  those tokens will compute. The window it writes back is the step's own.
- `embed_lookup_q` with `ahead` present may start reading those rows once its own are gathered. No
  output depends on it, so a kernel may ignore it; libavx's row hands them to a reader thread that
  fills its row cache while the step runs.

### `ple_gate`

PLE's mixing coefficient is a dot product between the n-gram embedding and the residual stream, one
scalar per hyper-connection stream per token:

    K[c] = grouped_rmsnorm(k)[c] * (w_key + wadd)      k is key_proj(embedding)
    Q[c] = grouped_rmsnorm(q)[c] * (w_query + wadd)    q is the incoming hc stream
    s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
    s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))
    gv[c] = sigmoid(s[c]) * v                          v is value_proj(embedding), one row
    gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)

- The signed square root keeps the sign, so a stream whose embedding opposes the residual gets a
  gate below one half rather than zero. The clamp is on the magnitude and the sign comes from the
  original, which matters at one value: torch's `sign(0)` is 0, so a dot product of exactly zero
  stays zero rather than becoming `sqrt(gate_eps)`.
- Both outputs leave the op: the dilated convolution reads the normed copy and the residual add the
  un-normed one, and recomputing either outside would be a second definition of the norm.
- The convolution branch does not see the gate. `gv[c]` is a scalar times one shared `v` row and the
  norm's rms is per `n`-wide stream, so the gate divides back out: `gvn[c][d]` is `v[d]/rms(v)`
  times its gain whatever the gate did. A fused kernel whose `gvn` varies with the gate is wrong. It
  is not a shortcut, though: `gv` is narrowed to its dtype before the third norm reads it, so
  computing `gvn` from `v` directly differs in the last bits.

It is one op rather than six (three grouped norms, a reduction, a scalar nonlinearity and a broadcast
multiply, all over `[M, hc*n]`, 10240 wide on Qwen3.8-Flash-Next), because six memory-bound launches
would each pay the per-kernel floor; `hc_read` is the same argument at the same width.

### `ple_conv`

    y[t] = resid[t] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation] )

The dilation sets the history depth. At Qwen4-Exp's width 4 and dilation 3 the taps are 9, 6, 3 and
0 timesteps back, so a sequence carries `(width-1)*dilation = 9` timesteps, not `width-1 = 3`. A
cache sized for the undilated width reads zeros where the deepest tap belongs, and the model still
generates fluently from a slightly wrong context.

The state is `[n_slots, n, state_len]`, slot `i` holding the value at relative position
`i - (width-1)*dilation` (`gdn_conv_prep`'s convention with the dilation folded in). `has_init` 0
means that window is zeros rather than whatever the slot last held; without it a fresh sequence on a
recycled slot convolves the previous sequence's tail into its first nine tokens.

`num_accepted` shifts the read offset, and its presence picks what the step leaves behind. The
manager sizes the state `(width-1)*dilation + n_spec` deep so a speculative step can read at an
offset inside it. The next step's offset must find the history ending at the last token this step
committed:

- **Present: a decode step**, whose drafts a verify may reject. The window is rewritten as
  `gdn_conv_update` rewrites its own: the history read, shifted down by one, then every input of
  the step, so slot `i` holds position `i + 1` of (window read ++ step). A next step that kept `k`
  of these tokens reads at `k - 1` and finds the history ending at the `k`-th. It needs
  `(width-1)*dilation - 1` slots plus one a token, which a step of `1 + n_spec` tokens gets.
- **Absent: every token commits**, as in a prefill chunk. The read is at offset zero and the last
  `(width-1)*dilation` inputs land at offset zero, where the next step's `num_accepted` of 1 reads
  them.

The two agree for a one-token step, and no single layout serves both at several tokens: at offset
zero one needs the history ending at the step's first token and the other at its last. An
architecture therefore issues the op once over its decode rows and once over its prefill rows, as it
does the two `gdn` convolutions.

`gdn_conv_prep` and `gdn_conv_update` are separate because the work around their convolutions
differs (one splits a fused projection and takes an l2 norm). PLE's convolution is the same
arithmetic at prefill and decode, so one op takes the union of both contracts: `has_init` is
prefill's question and `num_accepted` decode's.

The residual add is folded in because the reference is `gv + silu(conv(gvn))` and both tensors are
`[M, hc*n]`. `resid` absent leaves the plain convolution.

### `causal_conv1d`

    y[t] = sum_j w[c][j] * x[t - (width-1-j)]

LFM2's short convolution: depthwise, causal, undilated, with no activation and no residual (the
gates around it are separate `mul`s in the architecture). The window and its three optional operands
are `ple_conv`'s at dilation one, so an architecture issues it the same way, once over its decode
rows with `num_accepted` and once over its prefill rows without. libr4d, libref and libavx each
serve both ops from one implementation.

## Sampling

Every sampler is a kernel, not a host lift. `rad-kbench` checks the device samplers against a host
reference that follows llama.cpp's chain (`spec.md` §13).

**Per-request settings are operands, not parameters.** `params` is `[M]` rows of `RadSampleParams`
(`abi/rad_sample.h`), uploaded once a step, carrying `temp`, `top_k`, `top_p`, `min_p`,
`typical_p`, the three penalties, the DRY and XTC settings, the seed and position, the history
window and the grammar's `mask_row`. A parameter is geometry: the selector matches on it and freezes
it at declare (`spec.md` §2.2), so two requests in one batch with different `top_p` would need two
resolutions of one op. The one exception is `sample_chain` (below).

| op | params | operands |
|---|---|---|
| `sample_penalties` | `M`(range) `n_vocab` `vocab_off`/opt | `logits`:inout, `params`, `history` |
| `sample_temp` | `M`(range) `n_vocab` | `logits`:inout, `params` |
| `sample_topk` | `M`(range) `n_vocab` `vocab_off`/opt | `logits`, `params` → `cand_idx`, `cand_val`, `pairs`? |
| `sample_topp` | `M`(range) `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_minp` | `M`(range) `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_mask` | `M`(range) `n_vocab` | `logits`:inout, `params`, `bitmask` |
| `sample_pick` | `M`(range) `n_cand` | `cand_idx`, `cand_val`, `params` → `token` |
| `sample_argmax` | `M`(range) `n_vocab` | `logits`, `params` → `token` |
| `sample_merge_topk` | `M`(range) `n_cand` | `gathered`, `params` → `cand_idx`, `cand_val` |
| `sample_logprobs` | `M`(range) `n_vocab` `k` `vocab_off`/opt | `logits`, `params` → `stats` |
| `mtp_couple_propose` | `M`(range) `n_vocab` `round` `max_rounds` | `idx`:inout, `val`, `seq_ids`, `draw` → `state`:inout |
| `mtp_couple_verify` | `M`(range) `n_cand` `n_vocab` | `cand_idx`, `cand_val`, `params`, `state` → `token`:inout |

`mtp_couple_propose` and `mtp_couple_verify` are the linear-MTP coupling (`RADIANCE_MTP_COUPLED=1`),
and nothing else in the engine knows about coupling. `mtp_couple_propose` runs on a draft round's
final reranked top-8 and samples the proposal `x ~ Q`, where `Q` is the FP64 softmax of those
scores at temperature 1; it writes `x` into column 0 of `idx` and one state row per round. A row
whose `seq_ids` does not name the card its draw was staged for, or whose shortlist is malformed,
leaves column 0 alone with `status = -1`. `mtp_couple_verify` runs after `sample_pick`, over the same
final candidate plane, and applies the maximal coupling: the draft is accepted when it equals the
token the target itself drew at that position, and otherwise the token is a draw from the FP64
residual `(P - Q)+`. Both use 53-bit uniforms at `params.pos`; a row whose state is not bound to
that seed and position keeps the ordinary token. `P` is FP64-normalised where `sample_pick` is
FP32/24-bit, so a coupled row realizes the ideal filtered target law and is **not** seed-identical
to the ordinary pick.

`sample_mask` is where a GBNF grammar lands: the state machine stays on the host, and what crosses
is a token bitmask per sequence (~19 KB at a 151K vocab), cheap enough to upload every step. A set
bit means allowed.

`sample_penalties` and `sample_dry` read global token ids from `history` and write this rank's
logits, so under a split vocabulary they take `vocab_off`, as `sample_topk` does, and a token outside
the shard is another rank's to penalise. Their `penalty_last_n` / `dry_penalty_last_n` window is
llama.cpp's: 0 turns the stage off for the row, and a negative value takes the whole history.

`sample_logprobs` serves an OpenAI `logprobs` field or llama-server's `n_probs`. It runs on the raw
logits, before any stage has touched them, and only for a row whose flags carry `RAD_SP_LOGPROBS`;
other rows are not written. `stats` is `[M, 2 + 2k]` f32: the row's maximum, the sum of
`exp(l - max)` over it, and its `k` largest as (id, logit) pairs, descending, ties to the lower id,
the id written as a float (exact below 2^24) and global under a split vocabulary (`vocab_off`
added). A slot past the vocabulary is id -1, logit -infinity. `k` is geometry, the one width every
request's `top_logprobs` fits in: the engine declares 21, OpenAI's limit of 20 plus one so a row is
44 floats, whole 16-byte words, which the vocab-parallel `all_gather` requires. A token's logprob is
`l - max - log(sum)`; under a split vocabulary the host merges the ranks' rows (maxima and sums by
the usual rescale, pairs by value).

### Further sampling stages

| op | params | operands |
|---|---|---|
| `sample_dry` | `M` `n_vocab` `vocab_off`/opt | `logits`:inout, `params`, `history`, `breakers`? |
| `sample_typical` | `M` `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_xtc` | `M` `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_chain` | `M` `n_vocab` `temp`:f64/opt `p`:f64/opt `min_p`:f64/opt `k`/opt | `logits`, `query`?, `u` → `draw` |

`sample_dry` is llama.cpp's DRY: it penalises, once, each token that would extend a repeated suffix
of the history, by its longest such repeat. Sequence breakers are strings, so they are resolved
against the vocabulary on the host: the row's `dry_rep_limit` carries the repeat limit the most
recent breaker imposes, and the optional `breakers` plane (one byte beside every history token) marks
tokens that are whole breakers and are never penalised. Absent, no token is exempt.

`sample_typical` is locally typical sampling. It is not a prefix of the descending candidate order
(that order and the typicality order disagree), so it narrows in place rather than truncating.

`sample_chain` (device-only) is the whole chain in one kernel for a speculative verify: temperature,
softmax, top-k, top-p, min-p, renormalise and draw over an L2-resident row. `query` is the drafted
token per position (absent at the last), `u` is two uniforms per position, and `draw` is
`{u32 tok, u32 tok_excl, f32 p_query}`: a draw, a draw with `query` excluded, and `query`'s filtered
probability, the three values a rejection-sampling verify needs. Its settings are launch-time
parameters, never constraints. libr4d serves it; the engine's sampler issues the decomposed stages
and does not declare it.

### Vocabulary-parallel sampling

When `lm_head` is row-sharded, a rank's logits row covers only its slice of the vocabulary.
`sample_argmax` is then unusable (its answer is a local column index, a different token), so a greedy
request is staged at `top_k = 1` and takes the chain. `sample_topk` writes the optional `pairs`
output: its candidates as (u32 global id, f32 logit), with the rank's `vocab_off` added, which is
what `all_gather` moves and `sample_merge_topk` reads.

`sample_merge_topk` re-applies the row's `top_k`, which is why it takes `params`. Every rank ran its
own top-k before the gather, so the merged set holds up to `world_size * k` candidates and no later
stage narrows again. At `k = 1` that matters: `sample_pick` is a draw, so a greedy row with two live
candidates would return whichever one the seed selects.

`gathered` is `world_size` sorted runs, and the merge relies on it. Each rank's `sample_topk` writes
its `n_cand` pairs in descending (logit, lower id first) order with its holes at the tail, and the
all-gather lays a row's runs side by side. libr4d's merge therefore runs only the merge stages of
its sorting network when the runs are a power of two long and fill the row (11 passes over two runs
of 1024, where a full sort is 66), and the whole network otherwise. Unsorted runs give a wrong
order, not a slow one.

## Sparse attention (QSA)

| op | params | operands |
|---|---|---|
| `qsa_block_key` | `M`(range) `n` `ratio` `rotary_dim` `pos0`/opt `block_size`/opt `eps`:f64 `theta`:f64 `scale`:f64/opt `wadd`:f64/opt `dtype` `mode`/opt `sections`/opt `yarn_factor`:f64/opt `yarn_low`:f64/opt `yarn_high`:f64/opt `yarn_magnitude`:f64/opt | `k`, `w`w, `page`?, `pos`? → `out` |
| `dsa_pool` | `M`(range) `n` `ratio` `dtype` `block_size`/opt | `kg`, `ape`w → `q`, `scale`; `page`?, `cache`:inout? |
| `dsa_query` | `M`(range) `n` `dtype` | `x` → `q`, `scale` |
| `dsa_score` | `M`(range) `n` `heads` `blocks` `seqs` `dtype` | `q`, `q_scale`, `gate`, `k`, `k_scale`?, `bt`, `cu`, `nc` → `score` |
| `mla_attn` | `M`(range) `latent_dim` `heads` `blocks` `block_size` `scale`:f64 `dtype` | `q`, `kv`, `bt`, `length` → `out` |
| `mla_project` | `M`(range) `heads` `head_dim` `latent_dim` `mode` `dtype` | `a`, `b`w → `y` |
| `qsa_score` | `M`(range) `n` `heads` `blocks` `seqs` `dtype` | `q`, `bk`, `bt`, `cu`, `nc` → `score` |
| `qsa_select` | `M`(range) `blocks` `topk` `ratio` `seqs` `signed`/opt | `score`, `bt`, `nc`, `pos`, `cu` → `sel`, `nsel`, `seqused` |
| `qsa_tail_store` | `M`(range) `n` `ratio` `ring` `seqs` `dtype` | `k`, `pos`, `cu`, `state`:inout, `sidx` |
| `qsa_work` | `M`(range) `n` `ratio` `ring` `seqs` `work` `dtype` | `k`, `pos`, `cu`, `state`, `sidx`, `bt`, `rope`? → `stage`, `page`, `bpos`, `nc` |

Five ops decompose one indexer, which scores blocks of key history against the query and hands
`attn_paged` a shorter block table. `docs/QSA.md` describes the decomposition and
`arch/common/rad_qsa.h` issues all five. libref implements all five (the oracle for libr4d's
kernels), and libavx all five for the host path.

### `qsa_block_key`

The compressed key: one vector per `ratio` tokens of key history, against which the indexer's block
scores are taken.

```
out[b] = rope( rmsnorm( mean over j<ratio of k[b*ratio + j] ),  at position pos0 + b*ratio )
```

The rotation is `rope`'s NeoX ladder over the first `rotary_dim` lanes, with `rope`'s YaRN
coefficients when given, so a compressed key and the keys it summarises are rotated alike. With
YaRN, `scale` must be 1.

- **`M` is blocks and `k` is `ratio` times taller.** Every other op here has one output row per
  input row. A caller that sized `k` at the block count would pool rows of the next block into each
  one, so libr4d's entry point checks it.
- **The mean is accumulated in f32 and rounded to the activation dtype before the norm**, the
  reference's order (`key_groups.float().mean(dim=1).to(raw_keys.dtype)`). Carrying f32 into the
  norm gives a small, systematic difference on every block of every layer.
- **`block_size` equals `ratio` and exists to size a KV group.** `Builder::find_kv_consumer` binds a
  group to the first unclaimed op declared after it that carries the key `block_size`, and the
  block-key store's page is the compress block. libr4d refuses a disagreement.
- **`out` may be a KV cache, and is then rank four.** The block-key store's geometry is a container,
  not an attention shape (`kv_heads` 1, `block_size` the compress ratio, `head_dim` `n/(2*ratio)`),
  so a page holds exactly `n` contiguous values and the manager allocates one per `ratio` tokens.
  libr4d checks the product of the extents rather than the rank, which is the engine's business.
  `qsa_score`'s `bk` is the same object, read the same way.
- **`page` changes where the answer goes, not what it is.** The block-key store is a paged KV group
  (`docs/QSA.md`), so consecutive blocks are not consecutive pages. Absent, the write is contiguous
  (a caller with its own scratch, or a test). Present, `M` comes off `page`, not `out`, because
  `out` is then as tall as the pool. A `-1` entry is a block with no page and is skipped.
- **`pos` turns a launch from a range into a list; `pos0` is the range.** Within one sequence block
  `i`'s first token is at `pos0 + i*ratio`, which describes a prefill chunk. At decode the blocks
  completing in one launch belong to different sequences at different positions, so with `pos`
  given a launch is a set of work items and one launch a layer a step serves any mix.
- **Under a multi-component rotary** (M-RoPE) a block key rotates at its first token's three
  components. `qsa_work` takes the step's `rope` planes (`RadBatch::rope_pos`, `[3, T]`) and writes
  `bpos` as `[work, 3]`, a block's three components side by side: a first token in the step reads its
  own components, and one committed in an earlier step (always text, since a prefill chunk ends on a
  whole block and a prompt ends in text) takes its index plus the shift that sequence's first row in
  the step carries. `qsa_block_key` with `mode` (`mrope`, `imrope`) and `sections` then reads the
  component each frequency's section names, as `rope` does.

A block key is query-independent and permanent once its `ratio` tokens exist. The reference
recomputes it inside its per-query loop; the engine computes it when the block's last token arrives
and caches it, so the indexer costs 64 bytes a token a layer instead of a pass over the whole
history every step.

### `qsa_score`

```
score[m][b] = ( sum over heads h of relu( q[m][h] . bk[bt[seq(m)][b]] ) ) / sqrt(n)
```

- **`m` is a query and `bt` is per sequence; `cu` joins them.** At decode there is one query per
  sequence, but a prefill chunk is many queries over one history, each selecting its own set.
  `cu_seqlens` is the only operand that says which sequence a token belongs to.
- **`nc` bounds the block loop and is required.** `blocks` is a declared width sized for
  `max_ctx / ratio`, so a scorer that trusted it would read the whole declared width of padding on
  a short prompt. `nc[seq]` is the sequence's live complete-block count: an upper bound for every
  query in the chunk, within `T / ratio` of each one's own. `qsa_select` narrows it per query, since
  it has `pos`.
- **The caller narrows `score` to the live reach**, because a grid cannot read `nc`: both ops size
  their launch from the declared width (the live count is device data, the grid a host number). The
  architecture narrows the score operand's column count to
  `(max_ctx_len + n_tok + n_spec + ratio) / ratio + 1`, which the batch carries on the host, and the
  pitch stays the declaration's. Every term is deliberate slack: an underestimate drops blocks from
  the selection silently, and the `+ 1` keeps `n < blocks`, which makes `qsa_select`'s partial page
  `bt[n]` readable.
- **The relu is per (block, head) and the sum over heads**, in that order: independent votes, each
  of which can only argue for a block. Summing first would let one head's large negative cancel
  another's positive, a different model that still ranks blocks plausibly.
- **`bt` holds page ids, not block indices.** The attention KV group is paged at the compress ratio,
  so a block and a page are the same thing: the sequence's block table is already the list to score,
  and the top-k of the output is already the table the attention reads.
- **A negative table entry scores `-inf`, not zero.** A table is padded to its declared width and a
  real score can be zero (every head vetoed the block), so zero would let padding outrank it.

### `qsa_select`

Turns the scores into the table the attention reads: the `topk` highest-scoring blocks as page ids,
with the partial page appended.

- **`nc` is separate from the table's width** because the two differ by the partial page, which the
  op treats specially. `bt[n]`, when not `-1`, is the partial page: the tokens after the last complete
  block, which the reference attends unconditionally and which the query itself lives in. It is
  appended after the selection and never a candidate for it, so `sel` is `topk + 1` wide; sized at
  `topk` it would drop the newest tokens on three queries in four.
- **The bound is the query's**: `n = min(nc[seq(m)], (pos[m] + 1) / ratio)`, because block `b` is
  entirely in this query's past exactly when `b*ratio + ratio - 1 <= pos`. At decode the two
  bounds agree. `cu` is the join, as in `qsa_score`.
- **The partial page exists exactly when `(pos + 1) % ratio != 0`.** This is a causality rule. A
  decode query on a block boundary has `n == nc`, and `bt[nc]` is the page after the sequence's
  last, which the KV manager leaves `-1`, so testing the page would work at decode. Mid-chunk it
  would not: the same query finds later tokens of its own chunk already stored in `bt[n]`, a page
  entirely in its future, which `seqused` would then cover. The model would attend to its own
  future, which shows up only as an implausibly good prefill.
- **The partial page goes last.** Every selected complete page is entirely in the query's past, so
  the only causal masking left is inside the final page, and placing it last lets the attention's
  own `kk <= q` bound do it with no change to the attention kernel. The effective query position
  the caller passes is `(nsel - 1) * ratio + pos % ratio`.
- **Ties break to the lower block index, through a prefix count.** A block score is a sum of relus
  over a shared grid, so equal scores are ordinary, and a top-k placed by whichever atomic landed
  first would not reproduce across runs; the tree checks generated text byte for byte.
- **`signed` chooses the ranking.** Absent or `0`, a block is live when its page id is `>= 0` and
  its score `> -1e30`, and ranks by `max(score, 0)`: QSA's scores are sums of relus, so the clamp
  changes no ranking there. `1` is for indexers whose scores can be negative (GLM's DSA weights
  its heads by signed gates): a block is live when its page id is `>= 0` and its score finite, and
  ranks by the score itself, `-0` tying `+0`. The two rankings pick different sets whenever a live
  score is negative, so an architecture states the one its indexer needs. libr4d's
  `qsa_select_bf16` serves both; the signed ranking keys a score by its float's total order over
  all 32 bits.
- **`seqused` is why QSA needs no new attention kernel.** `attn_paged` takes its block table and
  per-sequence length as ordinary operands, so the sparse attention is that op issued with `sel` in
  place of the sequence's table and this number in place of its length. The KV it walks is `nsel`
  pages of `ratio` with the partial page last, so the query sits at `(nsel-1)*ratio + pos%ratio`
  and the length is one past it. Without a tail, `pos%ratio` is `ratio-1` and this collapses to
  `nsel*ratio`, every selected page fully visible. The op emits it because it is one store off a
  number it already has.

libr4d's kernel takes scratch (sized by the row's scratch hook; it is not an operand). With it, a
launch of at most 32 query rows cuts the block list into up to 16 chunks, takes each chunk's own
top-k and merges them, so a decode step uses more than `1 + n_spec` workgroups. The result is exact,
because a chunk's local top-k contains every global top-k member of that chunk, and identical in
output layout too: the attention sums the selected pages in table order and float addition is not
associative. Without scratch, or above 32 rows, one workgroup per query walks the whole list.

### `qsa_tail_store`

Parks the raw index keys of the tokens that have not completed a block yet:
`state[sidx[s][0]][pos[t] % ring] = k[t]`. A block key is the mean of `ratio` consecutive raw keys,
so at decode most of a block's keys come from earlier steps and something must carry them.

- **The tail is a linear KV state, not a plane indexed by the batch row.** It is read on a later step
  than the one that wrote it, and it has two axes a buffer cannot express: the layer (the keys are
  `index_qk_proj`'s output, a per-layer weight) and the sequence. The batch row is step-local (the
  scheduler builds a step from the running set in priority order and stable-sorts decode first), so a
  request that outlives an earlier one would shift down a row and inherit its keys. A linear group
  gives each sequence a slot for its life, zeroes it when handed out, and saves and restores it with
  the rest of the linear state when the prefix cache checkpoints. `state` is the group's cache for
  one bound layer, `[slots, 1, ring, n]`; `sidx` is the group's `state_index` row, column 0 the
  committed slot.
- **`cu` finds a token's row**; `slot_mapping` is a flat KV slot and carries no sequence. Which state
  that row belongs to is `sidx`, a different question.
- **The ring is `ratio + n_spec`.** A verify step runs `1 + n_spec` positions and the next step
  restarts at the first rejected one, so a position this step has already overwritten may be owed to
  the next. At `ratio` 4 and depth 3, the step covering positions 9..12 writes rows 1, 2, 3, 0, and
  row 0 is what position 8 is still owed, because the block holding 8..11 is rebuilt as soon as the
  following step starts at 10. The widest window a step can still owe is
  `[p0 - ratio + 1, p0 + n_spec]`, so a ring that long overwrites a row only with a position past
  the end of any step that could ask for it. With `n_spec` 0 the ring is the ratio.
- **Only the last `ring` tokens of a sequence's chunk write.** Positions `p` and `p + ring` share a
  row, so writing every token of a prefill chunk would have workgroups racing for one row. With the
  restriction every row has at most one writer, and the result is what a serial last-writer-wins
  loop leaves.
- **`state` is inout, and rows nothing targets keep what they held**: at decode a handful of rows
  are written and the rest are what the next block key needs.

### `qsa_work`

The gather that turns a batch into the flat block list `qsa_block_key` reads. Nothing in a batch
provides that list: the keys of a block completing at decode are mostly rows of the tail state plus
one of this step, those of a block completing inside a prefill chunk are rows of the chunk, and a
block straddling the chunk's start is some of each, while the blocks belong to different sequences
at different positions.

So this op walks the sequences, works out which blocks became complete, copies each one's `ratio`
raw keys into consecutive rows of `stage`, and writes its destination `page` and first-token position
beside it. The block-key kernel never learns that there were two sources. The copy is a kilobyte
per completing block (512 KiB per layer for a 2048-token prefill chunk); the alternative is a
block-key kernel with two sources, a per-key source selector and a tail layout.

- **`M` is tokens and `work` is blocks**: one block per `ratio` tokens plus at most one straddling
  block per sequence. `work` is a host bound on how many blocks can complete, not a count of how many
  did (that is device data); slots past the count get `page = -1`, which `qsa_block_key` skips, so
  the bound costs empty workgroups, never a wrong answer.
- **`bt` is the block-key group's table**, not the attention group's (`docs/QSA.md` explains the
  two).
- **`state` and `sidx` are the tail**, read at row `q % ring` of slot `sidx[s][0]` (see
  `qsa_tail_store`). A sequence with no slot (`sidx[s][0] < 0`) gets `page = -1` on its blocks.
- **`nc` is complete blocks per sequence**, `(last position + 1) / ratio`, which the scorer and the
  selection both need and this op computes on the way. It is not the number of blocks that completed
  this step; the two agree only for a sequence that starts at zero.

### GLM-5.3's indexer and latent attention

GLM's sparse attention keeps QSA's selection (`qsa_select` with `signed` 1) and replaces the
block keys, the queries and the score. Its indexer applies no RoPE, and the layer norm of the raw
keys precedes `dsa_pool`.

`dsa_pool` compresses `ratio` key rows into one. `kg` is `[M*ratio, 2n]` bf16, each row a
normalized key followed by its channel-wise gate logits. The learned f32 `ape` `[ratio, n]` is
added to the gate logits, which are softmaxed over the `ratio` axis per channel; the keys are
pooled with those probabilities and rounded to bf16, then rotated by the ascending-stride
Walsh-Hadamard transform, multiplied by `1/sqrt(n)` and rounded to bf16 again. Each row is
quantised to E4M3 with `scale = 2^ceil(log2(max(amax, 1e-4) / 448))`, codes rounded to nearest
even: `q` `[M, n]`, `scale` `[M, 1]` f32. `n` is a power of two up to 256, `ratio` 2 to 16.

`page` and `cache`, both or neither, also store the result into the indexer's paged key cache, at
`n` 128 and `ratio` 4 only. `page[M]` names each row's destination page, which must be unique
within the call; a negative or out-of-range id is skipped. `cache` is u8 `[pages, 1, 4, 34]` with
contiguous inner rows and a four-byte-aligned page pitch of at least 136 bytes: a page holds the
128 codes, then the f32 scale at byte 128, and its last four bytes and any pitch padding are left
untouched. That is one block-size-4 group of the indexer's KV, with no separate scale allocation.
`block_size`, when given, is 4 and names the consuming layout to the cache planner.

libr4d's `dsa_pool_fp8` serves `n` 128 and `ratio` 4, the cache form included: one wave per pooled
row, the softmax channels and the Hadamard in registers, pitched rows.

`dsa_query` rotates each bf16 query head by the same normalized Walsh-Hadamard transform, rounds
to bf16, and quantises to E4M3 with the same power-of-two scale: `x` `[M, n]`, `q` `[M, n]`,
`scale` `[M, 1]`, token and head flattened into `M`. There is no RoPE and no RMS norm. Unlike
`had_quant_act_fp8`, both the normalization of the rotation and the power-of-two scale are part of
the op. `n` is a power of two up to 256.

libr4d's `dsa_query_fp8` serves `n` 128, one wave per query head, pitched rows.

`dsa_score` takes E4M3 queries `[M, heads, n]` with f32 scales and signed f32 gates `[M, heads]`,
E4M3 pooled keys `[pages, n]` with f32 scales `[pages]`. `bt[seqs, blocks]` names key pages,
`cu[seqs+1]` holds absolute query offsets and `nc[seqs]` bounds the live blocks. The score of block
`b` is `k_scale * sum_h(gate_h * q_scale_h * relu(dot(q_h, k))) / sqrt(heads*n)`: the ReLU comes
before the head weighting, so a negative gate can make a block's score negative. A block past
`nc`, a negative page id or one past the key rows scores `-inf`. `score` is f32 `[M, blocks]` and
may have fewer columns than `bt` with its own row pitch, so an architecture can bound it by the
current history. `qsa_select` applies each query's causal bound, so scoring a chunk's shared
history makes no future block eligible.

`k` may instead be the packed u8 cache `dsa_pool` writes, with `k_scale` absent: the codes and the
embedded scale are read in place. `q` may be flattened `[M*heads, n]` with `q_scale`
`[M*heads, 1]`, each with its own row pitch, which is `dsa_query`'s output as written.

libr4d's `dsa_score_fp8` serves `n` 128 and 32 heads, `M` up to 65535, both key forms and both
query forms. A block of two waves covers the 32 heads of one query row against one or two 16-key
tiles (two past 32 blocks); the dot products run on fp8 WMMA, and scales and gates fold only after
each head's ReLU.

`mla_attn` is absorbed latent attention: one latent vector per token is both key and value.
`q` and `out` are `[M, heads, latent_dim]`, `kv` is `[pages, 1, block_size, latent_dim]`, and
`bt[M, blocks]` is each query's own page table, the selected complete pages followed by its tail.
`length[M]` counts the table's visible tokens (every selected complete page and the visible part of
the tail) and is the causal bound: no slot at or past it is read. A missing or out-of-range page
contributes nothing, and a row with no visible key outputs zero. The softmax is f32 and `out` is
rounded once. `scale` is positive and finite and comes from the original query/key head width,
not the latent width (GLM: `1/sqrt(256)` against a latent of 512).

libr4d has two `mla_attn` rows, both at `latent_dim` 512, `block_size` 4 and up to 64 heads. The
matrix row, `mla_attn_matrix_bf16`, ranks above the wave row and takes heads in multiples of 8: the
heads are the WMMA tile's rows, so all of them share each latent load, and the probabilities enter
it as bf16 high and low parts with f32 accumulation. At a multiple of 16 a tile holds 16 heads of
one query; otherwise it pairs eight heads of two neighbouring queries, loading a key once where
both tables name the page, so GLM-5.3's 8 heads a rank fill the tile. The pairing walks the union
of the two tables, so each chunk runs once, for tables of up to 520 entries; past that it pairs
by table position, with a second pass where the two differ. Other head counts take the wave row,
`mla_attn_bf16`: a wave per head with an online softmax. Both split a short batch's history across
workgroups and merge the f32 partials before narrowing, into the fixed scratch their hooks
reserve; without scratch they compute unsplit.

`mla_project` moves a query into the latent and a latent result out of it, through the bf16
weight `b` `[heads*2*head_dim, latent_dim]` that stores each head's key rows, then its value rows.
`mode=absorb` multiplies `a` `[M, heads, head_dim]` by the head's key block as stored,
`[head_dim, latent_dim]`, giving `[M, heads, latent_dim]`; `mode=expand` multiplies
`a` `[M, heads, latent_dim]` by the transposed value block, giving `[M, heads, head_dim]`. Both
accumulate in f32 and round once to bf16. The KV cache stays latent; only this weight carries the
per-head width. Its rows are head-major, so a tensor-parallel slice of whole heads is a row range.
`a` and `y` may also be `[M, heads*width]`, heads contiguous, with their own token pitch.

libr4d's `mla_project_bf16` serves `head_dim` 256 and `latent_dim` 512, up to 64 heads, pitched
tensors. It takes up to eight rows a thread (`absorb`) or a wave (`expand`), loading each weight
element once for them, with every row's sum in the same order whatever the group, so a row's bytes
do not depend on the batch.

## Draft heads

A drafter is an ordinary graph over ordinary ops except at its two ends: what proposes the
candidates, and what enters the head from the trunk.

### DFlash2

| op | params | operands |
|---|---|---|
| `dflash_select` | `M` `steps` `top_k` `rank` `n_vocab` `anchor_stride`/opt `dtype` | `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w → `tokens` |

`dflash_conv` is the drafter's grouped convolution (coefficient add and every tap in one pass):

| op | params | operands |
|---|---|---|
| `dflash_conv` | `T` `hidden_size` `taps` `group_size` `NG` `block_size` `dtype` | `x`, `delta`, `base`w → `out` |

    out[t][h] = sum over taps j with j == 0 or t % block_size >= j of
                (base[j][h] + delta[t][j][h / group_size]) * x[t - j][h]

Blocks are `block_size` rows starting at row 0, so a tap never reaches into the previous block.
`delta` is `[T, taps, NG]` read at its own row pitch: the architecture passes a slice of the
`[T, 2, taps, NG]` coefficient projection. `out` must not alias `x`. libr4d's kernel serves
`taps` 2, `group_size` 16 and a power-of-two `block_size`; libref and libavx any.

`add_rmsnorm` (device-only) is the drafter's residual add and the norm after it in one pass:

| op | params | operands |
|---|---|---|
| `add_rmsnorm` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt `mult`:f64/opt | `x`:inout, `a`, `w`w → `y` |

It writes `x + a` into `x`, rounded as `add` rounds it, then `y = rmsnorm(x) * (w + wadd) * mult`
over the `x` it wrote, at `rmsnorm`'s width and column partition: byte-identical to `add` followed
by `rmsnorm`, on both outputs. That is where it differs from `rmsnorm_add`, which normalises the
unrounded sum: a drafter whose trained graph is the pair keeps its bits.

`dflash_select` is the greedy walk over the candidate lattice: `cand`/`unary` are `row_topk`'s two
output planes at one row per draft position, `hp` the selector's rank-256 query, and `anchor` the
step's own token ids (`anchor_stride` is the block width, because a draft pass carries `block` ids a
sequence and the anchor is the first). `tokens` is the draft, which the engine reads by name rather
than through the sampler. Only the K edges leaving the previous pick are scored, because the walk is
serial.

`hp` absent is a plane of ones, which turns the edge score
`unary + Σ_r pred[prev][r]·hp[l][r]·succ[cand][r]` into the plain low-rank bigram
`⟨pred[prev], succ[cand]⟩`. That is DSpark's Markov head: a rank-256 learned bigram bias added to
the draft logits, decoded greedily, with the predecessor read after the previous step's argmax
inside one workgroup. Written literally it is `steps` full-vocabulary `[rank, n_vocab]` matrix
products; through this operand it is a codebook gather. The only approximation is the top-K
candidate set, which can cost acceptance and nothing else.

### `mtp_enter`

| op | params | operands |
|---|---|---|
| `mtp_enter` | `M`(range) `n` `hc` `eps`:f64 `dtype` `wadd`:f64/opt `ngroup`/opt `fc`/opt | `h`, `e`, `w_h`w, `w_e`w, `fc_h`w, `fc_e`w → `x` |

    ne[j]    = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
    fe[i]    = fc_e[i] . ne
    rs[k]    = rsqrt(mean(h[k*n .. k*n+n)^2) + eps)
    nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
    x[k*n+i] = fc_h[i] . nh[k] + fe[i]                         for each of the hc sub-streams

The MTP head's entry when the residual stream is wide. A pre-norm MTP head (Qwen3.5, Qwen3-Next,
DeepSeek-V3) joins the trunk's hidden state and the next token's embedding with one `fc` over their
concatenation, and `arch/common/rad_block_mtp_fp8.h` runs that as two `rmsnorm`s into the two
halves of one buffer and one `gemm_nt`. Qwen4-Exp's head cannot: its hidden state is the wide
stream, and its checkpoint carries `mtp.fc_hidden` and `mtp.fc_embedding` as two square `[n, n]`
matrices beside a `[hc*n]` gain and an `[n]` one. So the hidden half applies the same matrix to each
sub-stream (the reference's `.unflatten(-1, (hc_count, hidden_size))`), and the embedding half is
one `n`-wide product added into all `hc` of them, the broadcast the trunk enters the stream with
(`hidden_states.repeat(1, 1, hc_count)`).

It is one op because the unfused form cannot be written: every intermediate is a different view of
the same bytes (`nh` is `[M*hc, n]` to the GEMM and `[M, hc*n]` to the norm), and
`core/runtime/issue.cpp` narrows only dim 0 or the last dimension with every stride left alone, so a
row-pitch change is inexpressible. It also saves launches in a draft round, whose latency the head
exists to hide.

`fc` is the two square matrices' stored format: `bf16`, or `e4m3` rows with f32 128-column group
scales in each row's own tail (the geometry of `hc_read`'s `mix_down`). Only the drafts see it (the
head proposes and the trunk verifies), so it can cost acceptance, never a token. A host kernel
reads the codes and f32 scale planes joined (Weights of several planes).

`ngroup` is the one thing the checkpoint does not say. A `[hc*n]` gain fits both a grouped norm
(`hc` rsqrts, one per sub-stream, as `Qwen4ExpTextGatedResidual` does at every connection) and a
flat one (one rsqrt over the whole wide row). Both readings are self-consistent, libref shares
whichever the graph picks, and no per-op oracle can tell them apart. The empirical answer is cheap:
a head wired correctly accepts most of what it proposes, and one wired wrongly proposes noise while
the model's own output stays correct. The head declares the grouped reading (`ngroup = hc`), and the
parameter makes a flat reading (`ngroup = 1`) one declaration away rather than a different kernel.

---

## Adding an op

Add the schema row to `libref/`, write the naive host implementation beside it, and add a row to
this document (`ops_doc` fails on an op a plugin declares that no table lists). No core file
changes. An op added without a ref implementation has no fallback and no oracle; say so in the
commit, because that is a decision, not an oversight.
