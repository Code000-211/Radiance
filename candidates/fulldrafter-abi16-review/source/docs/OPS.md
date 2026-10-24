# The conventional op vocabulary

Op names are free strings (`spec.md` §2.3) and the core never learns one. But `libref`
cannot implement an op it has never heard of, so **ref's op list is the conventional vocabulary**:
an architecture plugin that stays inside it is guaranteed to run somewhere, and one that invents
an op has no fallback for it (`spec.md` §17).

This file is that list. `libref/` holds the authoritative `RadOpSchema` rows; this is what
they say, in a form an architecture plugin author reads first.

**AND IT IS CHECKED.** `tests/CMakeLists.txt` registers `ops_doc`, which dumps every plugin's
schemas with `rad-schemas` and holds every table row below to them — op for op, parameter for
parameter, operand for operand, including the optional markers and the `w` that makes an operand a
weight. Run it by hand as:

    tools/ops_doc_check.sh docs/OPS.md build/bin/rad-schemas build/radiance_home/kernels/*.so

A row that is a PROPOSAL rather than a description carries a FOURTH COLUMN saying so and is exempt
from the check. That is how an op the vocabulary has agreed on but no plugin declares yet stays in
this document without being a lie, and it is the only way one may.

An op that a kernel library serves and this document omits gets written against a GUESS at its
schema, and no fallback absorbs a wrong guess — `qk_norm_rope_gate` and `qk_norm_rope` differ in
exactly the way a guess gets wrong, one taking a `theta` and the other a precomputed cos/sin table.
**A document that is WRONG is worse than one that is missing**, because it is read instead of the
source.

**A ROW HERE DOES NOT MEAN THERE IS A HOST KERNEL.** A table marked **device-only** is declared by
libr4d and by nothing else, so its ops have no oracle for `rad-kbench` and no host fallback: an
architecture that reaches for one is choosing to run on this library or not at all. That is a
legitimate choice — every fusion in this document is one — but it is a choice, and it should be
made knowingly.

## Conventions

- Operands are **positional**, in the order given. `w` marks a weight operand, `→` the outputs.
- A tensor's leading dimension is tokens unless stated. Everything is contiguous unless an op
  says otherwise; strides are the kernel's business, not the schema's.
- `dtype` is always present and always a string. It names the **operand combination**, which is
  why a weight format and an activation format share one key: `bf16`, `f16`/`fp16`, `f32`/`fp32`,
  `fp8_e4m3`, `fp8a8`, `fp8a16`, `w4`, `w4a8`, `w4a8_asym`, `w4a8h`, `w8a8`, `w4a16`, `w8a16`,
  `w2a8`, `mxfp4a8`. `w4a8h` is `w4a8` with a `group`-wide Walsh–Hadamard along K baked into the
  stored weight (`docs/MOE-W4.md`), `w4a8_asym` the unsigned-nibble-minus-zero-point grid rather
  than the symmetric one, and `fp8a8`/`fp8a16` say whether a block-scaled E4M3 weight is met by
  an fp8 activation or a bf16 one. `q_dtype` and `kv_dtype` are the same key split in two where
  attention reads two formats at once.
- `M` is the ranged parameter wherever a token count appears. **At most one ranged parameter per
  op** — that is what keeps `rad_issue` a single integer on the hot path.
- Every op below returns `RAD_OK` or a negative `RAD_E_*`. There is no partial success.
- In a row, `?` marks an optional operand, `/opt` an optional parameter and `/der` a derived one,
  and none of the three is decoration: `/der` says the KERNEL supplies the value rather than the
  caller, an omitted optional parameter is one a kernel will ignore when a caller passes it, and a
  constraint on an `:f64` key never matches, because a tolerance is not a predicate.

## How a schema is written

`libref` and `libr4d` implement the same ops from opposite ends, and a schema they do not agree on
cannot be loaded at all — the loader refuses the whole plugin on any one conflict, which makes
every other row of it unreachable and `rad-kbench` unrunnable. Four rules settle every case, and
they are why `gdn_conv_prep` and `gdn_recurrent_update` carry the long operand lists they do.

### The four rules that resolve every case

1. **Operands are the UNION, and anything only one implementation needs is `optional`.** A schema
   that lacks an operand cannot express the call at all, so the narrower reading always loses. An
   implementation ignores an optional operand it does not use; it cannot invent one it needs.
2. **Parameters are the UNION, and anything only one implementation needs is `optional`.** Same
   argument. A constraint on an absent key simply does not hold, which is the correct answer: a
   row that needs a parameter the caller did not state is not a match.
3. **A float-valued parameter is `RAD_P_F64`.** `eps`, `scale`, `theta`, `softplus_thr`, `l2_eps`,
   `wadd`. A `"%.9g"` string round-trips an IEEE single exactly and is unreadable in a graph dump,
   which is the whole of the argument against spelling them as strings. **Constraints never match
   on a float** — a tolerance is not a predicate, and a kernel that cares about `eps` to that
   precision has a bug.
4. **An enumerated parameter is `RAD_P_STR` with named values.** `act` is `"silu"` or `"sigmoid"`,
   not `0` or `1`. A graph dump reading `act=1` is precisely the "unclear what actually runs"
   complaint this engine exists to answer; a plugin's launch adapter maps the name to whatever
   integer its kernel wants.

The union rule has exactly two deliberate exceptions, both on the gated delta net and both argued
where those ops are described: `cu` and `num_accepted` are REQUIRED, because an op that lets the
caller omit either is an op whose contract is optional.

### Parameters the kernel supplies

`attn_paged.block_size`, `gdn_chunk_scan.chunk`, `gdn_kkt_solve.chunk` and `gdn_conv_prep.chunk`
are supplied by the **kernel**, not the caller (`spec.md` §7.2, and `RAD_DERIVED` in `rad_abi.h`).
Constraints on them are skipped during matching and the winning kernel's `RAD_C_EQ` value is
written into the resolved geometry afterwards. Declaring them `RAD_REQUIRED` makes them
unselectable: a constraint whose key the geometry lacks does not hold, so a kernel compiled for
`block_size == 16` could never be reached by the query whose answer was to be read off it.

`gdn_conv_prep.chunk` has a cost the other three do not: nothing in conv_prep's operands carries
the chunk length, so `libref`, which declares no `RAD_C_EQ` to answer with, returns `RAD_E_SHAPE`
by name when a caller omits it rather than quietly treating the whole sequence as one chunk.
libr4d supplies 64.

`all_reduce.hops` is `RAD_OPTIONAL` — a one-shot/two-shot distinction the caller may state and
usually should not. `all_reduce.exact` stays `RAD_REQUIRED`: a caller that does not say it will
accept a lossy wire gets no kernel rather than the quantised one.

### Weight operands, and the weight TABLE

- **`RAD_OPD_WEIGHT`** — the issue names one weight handle, and the kernel receives that weight's
  bytes, its dtype and its declared shape.
- **`RAD_OPD_WTAB`** — the issue names the FIRST handle of a run and the run's length, and the
  kernel receives a device array of `n` pointers, one per entry.

`RAD_WTAB(h, n)` in `abi/rad_runtime.h` is the issue-side spelling, and `rad_decl_weight` handing
out consecutive indices is what makes a declaration loop over experts produce such a run.

**Why it is a table and not a base plus a stride.** A layer's experts are separate movement units
(`spec.md` §4.1): after placement one is in VRAM, the next is streamed over the link and the third
is still on disk. `base + e * stride` describes exactly one of those cases — the one where nothing
was offloaded — and expert offload is the reason the placement engine exists. So the core resolves
every handle in the run the way it resolves any weight (residency table, mover event, use note) and
uploads a pointer array, rebuilding it only when a pointer actually moved.

**What the kernel sees.** `RadTensor::data` points at the pointer array; `shape[0]` is the entry
count and **`stride[0]` is 0**, which is the marker. `dtype` and the trailing extents and strides
are ONE entry's, and every entry of a run must agree about them — the core refuses the issue by
name if they do not, because a table whose entries have different shapes is a table the kernel
cannot walk. A kernel may also accept the STACKED form (an ordinary rank-3 tensor with a real
leading stride): `rad-kbench` replaying a fixture has nowhere to put a pointer array, and both
libref and libr4d read either, so a described case runs against the same bytes the engine would
hand it.

**How the declaration divides.** An op may carry several table operands — `moe_gemm_q` has four,
the codes and the scale planes of its even experts and of its odd ones. Whatever is left after the fixed weight positions have taken one each
is split EVENLY across the table positions, in declaration order, and where it does not come out
whole the first positions take one more each: `moe_gemm_q_mod4`'s first `n_expert % 4` classes hold
one expert more than the rest, and their codes and scales are its first table positions. A list too
short to give every table an entry is refused at declare. A table whose declared run is not
contiguous is refused when the tables are bound, before the first step; one whose entries disagree
about their shape, or an issue naming a run other than the declared one, is refused at issue. That
is exactly right for the case tables exist for: every table of one MoE op takes one weight per
expert of its class.

## What the loader enforces

**A kernel plugin must declare a `RadOpSchema` for every op it has a `RadKernelInfo` row for.**
The loader checks this after the whole hierarchy is open — deferred to the end because an override
plugin may legitimately rely on the schema of the plugin below it — and refuses the load naming the
plugin and the op. Without the rule, a typo and an unimplemented op are the same diagnostic.

The first plugin in hierarchy order to declare an op **fixes its schema**. A later plugin declaring
the same op name with a *different* schema is refused at load with both plugins named; declaring it
identically is fine and is what every plugin implementing a shared op does. Since arguments are
positional, two disagreeing schemas would make the same call mean different things depending on
which plugin won selection — silent numerical garbage rather than a diagnosable failure.

`build/bin/rad-schemas a.so b.so` prints and diffs what plugins actually declare. Run it before
adding a kernel plugin; "refused at load" names one op and leaves you to find the rest.

---

## Elementwise and norms

| op | params | operands |
|---|---|---|
| `rmsnorm` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt | `x`, `w`w → `y` |
| `rmsnorm_add` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt | `x`, `residual`, `w`w → `y`, `residual_out` |
| `hc_enter` | `M`(range) `n` `hc` `dtype` | `x` → `h` |
| `hc_read` | `M`(range) `n` `hc` `lowrank` `eps`:f64 `dtype` `wadd`:f64/opt `inject`/opt `group`/opt `rotate`/opt `mix`/opt | `h`, `w`w, `mix_down`w, `mix_up`w, `inject_w`w? → `x`, `inj`?, `q`?, `scale`?, `rq`?, `rscale`? |
| `hc_write` | `M`(range) `n` `hc` `dtype` | `y`, `inj`, `h`:inout |
| `hc_mean` | `M`(range) `N` `hc` `dtype` | `h` → `x` |
| `layernorm` | `M`(range) `n` `eps`:f64 `dtype` | `x`, `w`w, `b`w? → `y` |
| `grid_embed` | `M`(range) `n` `side` `dtype` | `x`:inout, `table`w, `coord` |
| `add` | `M`(range) `n` `dtype` | `a`, `b` → `y` |
| `mul` | `M`(range) `n` `dtype` | `a`, `b` → `y` |
| `silu_mul` | `M`(range) `n` `dtype` | `gate_up` → `y` (splits `[M,2n]` into gate·up) |
| `gelu` / `silu` | `M`(range) `n` `dtype` | `x` → `y` |
| `cast` | `M`(range) `n` `from` `to` | `x` → `y` |
| `softmax` | `M`(range) `n` `dtype` | `x` → `y` |
| `sigmoid` | `M`(range) `n` `dtype` | `x` → `y` |
| `silu` | `M` `n` `dtype` | `x` → `y` |

`gelu` and `silu` are two ops with one schema, and `silu` carries a row of its own because a row
names one op.

`rmsnorm_add` exists because the residual add and the next norm are one pass in every real
implementation. A plugin asks for it first and falls back to `add` + `rmsnorm` (`spec.md` §2.3).

`hc_read` / `hc_write` are the GATED RESIDUAL pair, which is what a hyper-connection model has
instead of a pre-norm and a residual add. The stream is `hc` times the model width and every block
reads a mix of the `hc` sub-streams and writes back into all of them, both gated by data:

    hc_read     N      = grouped_rmsnorm(h)          one rsqrt per n-wide stream, one hc*n gain
                g      = sigmoid(mix_up @ silu(mix_down @ N / hc))
                x      = mean over the hc streams of g * N
                inj[s] = 2 * sigmoid(inject_w[s] . N / hc)
    hc_write    h[s]  += inj[s] * y

Three facts that are not derivable from the others and that a second implementation has to match:
the mix is a MEAN over the streams and not a sum; there is a separate `/ hc` inside each of the two
gate arguments; and **`hc_write` adds into the UNNORMED `h`** -- the normalisation is consumed by
the block input and the gates and never enters the residual. Writing into `N` instead re-normalises
the stream 96 times over and still produces fluent text.

`inject_w` absent is the final mixer, which collapses the wide stream to one and has nothing to
write back; `group` absent is a read whose consumer wants bf16. Otherwise the fp8 form of `x` comes
out of the same pass, because a model with 96 connections cannot afford a separate quantise launch
at each of them -- and it is quantised from the NARROWED `x`, not from the accumulator, so a fused
kernel and the reference agree bit for bit. `rotate` adds `rq`/`rscale`: `had_quant_act_fp8` of the
same stored `x`, each group rotated by an unnormalised Hadamard before its absmax, for a consumer
whose weight plane carries the rotation (the w4a8h experts) -- which then needs no quantise launch
of its own either.

`mix` is the mixing matrices' STORED format: `bf16`, or `e4m3` -- rows of E4M3 codes with f32
scales in each row's own tail, a 128-column group for `mix_down` and a quarter row for `mix_up`.
It changes the bytes a step reads, not the arithmetic above; the two matrices are replicated on
every rank and read once a connection, so at 96 connections the format is worth half of 1.26 GB a
step on Qwen3.8-Flash-Next.

`hc_mean` is NOT a third member of that pair, and substituting `hc_read` for it is the mistake it
exists to rule out. It is the DFlash2 capture's own reduction and nothing else:

    hc_mean     x[m,i] = mean over the hc streams of h[m, c*N+i]

Summed in f32, divided by `hc`, stored once in the output's dtype. No rsqrt, no gate, no weight, no
read of a block input -- `hc_read` would normalise each stream, multiply by a data-dependent gain
and divide by the same `hc` a second time, so every tap would differ. It is bf16 in and bf16 out;
any other dtype is refused rather than converted.

**`x` is normally a COLUMN VIEW of a wider tap plane.** A DFlash2 step records one tap per named
layer into `[max_tok, n_taps * N]`, issue t writing width `N` at column `t*N` with the plane's
`n_taps * N` kept as the row stride. The op therefore writes through `x`'s own row pitch and never
touches a neighbouring tap; a tool that allocates its own output gets a packed `[M, N]` plane, which
is the same call with the width as its pitch. The input's pitch is honoured the same way, so a
column slice of a wider buffer is a legal `h` too. An output aliasing the input is refused.

`mul`'s `b` operand may have extent 1 on any dimension where `a` does not, and is broadcast along
it. Every other elementwise op keeps the strict same-shape rule.

`sigmoid`, with a broadcast-capable `mul`, is what a gated shared expert needs
(`ffn_gate_inp_shexp`). Without it the shared expert runs ungated, which is a different model.

## Quantisation and rotation

| op | params | operands |
|---|---|---|
| `quant_act_i8` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale`, `asum`? |
| `had_quant_act_i8` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale` |
| `rmsnorm_had_quant_i8` | `M`(range) `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gated_had_quant_i8` | `M`(range) `n` `group` `dtype` `act`:str/opt `mode`/opt | `a`, `b`? → `q`, `scale` |
| `dequant` | `M`(range) `n` `group` `dtype` | `q`, `scale` → `y` |
| `gram_accum` | `M`(range) `n` `group` `dtype` | `a`, `a_scale`?, `h`:inout |

`gram_accum` is the odd one here and belongs in this table anyway, because it characterises a
quantiser's output: `h[k1][k2] += sum over rows of dequant(a)[m][k1] * dequant(a)[m][k2]`, in f32.
That is the Hessian a GPTQ packer needs, over exactly the operand the consuming GEMM contracts.
**`h` is `[n, n]` and INOUT** -- a caller zeroes it once and drives a whole calibration corpus
through the graph, so each launch adds its own block; an implementation that stored instead of
added would agree on one call and disagree on a corpus. It is **not** `gemm_nt` with a transposed
operand: that op contracts the LAST axis of both sides and this one contracts the FIRST, so
expressing it as a GEMM would need `x` materialised transposed. It is also the one op in this
document that **no forward pass issues** -- it runs under a calibration flag and its result is
read at convert time, in another process, through a file.

**`a_scale` is optional, and `dtype` says which reading `a` takes.** `fp8a8` is E4M3 codes with
an f32 dequant scale per (row, `group` columns): the tap of a model whose expert GEMM contracts a
quantised activation. `bf16` is a plain activation read as it stands, with no `a_scale` and no
grouping: the tap of a model served at bf16, whose expert GEMM contracts exactly that.

**Two activation widths are in use, and which one a linear takes is the WEIGHT's question rather
than a preference.** Where the checkpoint already ships E4M3 weights and a block scale plane, the
activation follows the weight: `fp8a8` is the shipped path (`arch/common/rad_fp8.h`), because the
`fp8a16` form that would read a bf16 activation directly and skip the quantise launch has a kernel
only at `M <= 1` — a single-sequence decode and nothing else. Where the engine quantises the
activation itself, it is **rotated int8**: the rotation is a per-group Walsh–Hadamard, and it is
what makes a single scale per row viable at all, which is what an int8 row buys over a row of
floats each carrying its own exponent.

**`group` MEANS TWO DIFFERENT THINGS IN THIS TABLE**, and a kernel that reads it the other way is
wrong rather than slow. `libref/ref_quant.cpp` is the definition; this is what it says.

- **`quant_act_i8`, `dequant`** — `group` is the number of **contiguous elements sharing one f32
  scale**, so `scale` is `[M, ceil(n/group)]`. `group <= 0` or `group >= n` means one scale per
  ROW. For `dequant` the same reading, inverted: the scale stream's extent says how many groups
  there are and `group` says how wide each one is.
- **every `had_*` INT8 form** — `group` is the **Walsh–Hadamard rotation width** (libr4d spells it
  `had`), and the int8 scale is per **ROW**, carrying the `1/sqrt(group)` normalisation. That is
  the reading the prose above forces: a per-group rotation is what makes a single scale per row
  viable, so the rotation width and the scale granularity cannot be the same number.
- **the `had_*` FP8 forms** (`had_quant_act_fp8`, `gated_had_quant_fp8`) — `group` is **both**, and
  that is not an inconsistency but a different bargain. They exist for a four-bit MoE expert whose
  scale grid is already one bf16 per (row, 128 of K); making the rotation the same 128 lines the
  two up, so the scale is per group here and shaped `[M, n/group]` exactly as `quant_act_fp8`'s is.
  They also do **no normalisation at all**: `<Hw, Ha> = group * <w, a>`, and the whole residual
  factor is folded into the **stored weight scale** at convert, where dividing a bf16 by a power of
  two is exact. Splitting it as the int8 forms do would round `amax/7/sqrt(group)` into bf16 and
  land a ~0.2% multiplicative error on every group. `n` must be a multiple of `group`: a rotation
  block cannot straddle the end of a row.
- **`quant_act_i8g`, `had_quant_act_i8g`** — `quant_act_fp8`'s grouping and scale shape at INT8
  codes: one f32 scale per `group` of a row, `amax / 127` (1 for a zero group), each code the value
  over it rounded to nearest even and clamped to +-127. They are the activation side of the int8
  W8A8 trunk (`gemm_nt_q` at `dtype=i8a8`), which is why they are not `quant_act_i8`: that op's scale
  is per row unless asked otherwise, and an int8 GEMM folding a scale per K block needs the block.
  The `had_` form rotates first, as `had_quant_act_fp8` does.

The rotation is unnormalised butterflies at ascending strides `1, 2, 4, ... group/2`, pairing `i`
with `i + h` where `(i & h) == 0`. The `1/sqrt(group)` is folded into the scale rather than applied
to the values -- dividing every element by a constant cannot change which int8 code it rounds to,
and folding it costs nothing. A rotation block may not cross a tensor-parallel shard boundary, so
the width has to divide the PER-RANK `n`.

`gated_had_quant_i8` serves two readings of its input: the gate and up as two operands, which is
libr4d's, and one fused `gate_up` `[M, 2n]` with the gate first, which is what a fused gate_up
projection produces. Two operands with the second optional expresses both — a caller with a fused
buffer passes it as `a` and omits `b` — and `mode` is optional because it is libr4d's integer
spelling of `act`, not the vocabulary's.

### The fp8 fusions

| op | params | operands |
|---|---|---|
| `quant_act_fp8` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `had_quant_act_fp8` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `rmsnorm_quant_fp8` | `M` `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gated_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `gemm_nt_q_gated` | `M` `N` `K` `group` `dtype` `act` | `a`, `a_scale`, `b`w, `b_scale`w → `q`, `scale`, `out_bf16`? |

### The int8 and gated fp8 fusions — device-only

| op | params | operands |
|---|---|---|
| `quant_act_i8g` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `had_quant_act_i8g` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `gated_had_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `gate_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate`, `x` → `q`, `scale`, `out_bf16`? |

`gemm_nt_q_gated` is the row above pulled into the projection that feeds it: `gemm_nt_q` over a
`[N, K]` gate_up plane — `N` is the WEIGHT's rows, twice the fused width, gate half first — with
`gated_quant_fp8`'s pass in the epilogue, so the `[M, N]` bf16 tensor between them is never
written. It is a separate schema and not a flag on `gemm_nt_q` because it does not produce that
op's `y`, and an architecture that resolved it into `gemm_nt_q`'s operand list would hand the GEMM
a buffer of half the width it expects. `act` is REQUIRED here, unlike the quantiser's: an
implementation is free to compile one gate and an optional parameter would let it resolve for
another.

`out_bf16` is not an optimisation: the unfused pair leaves a bf16 tensor behind and the codes are
taken over *that*, so a layer with a bf16 consumer of the post-norm value must ask for it or the
fusion is a different function. `residual` present is the `fused_add_rms_norm` form.

### `dequant_w4` — device-only

`libref` declares `dequant` and nothing else declares `dequant_w4`, so this op is not negotiated
between plugins the way the shared ones are: it is libr4d's own, spelled libr4d's way.

| op | params | operands |
|---|---|---|
| `dequant_w4` | `N` `K` `group` `dtype` `twos` | `wq`w, `ws`w → `y` |

`twos` selects the symmetric two's-complement nibble over the asymmetric unsigned-nibble-minus-
zero-point grid. The two differ by bit 3, so the wrong one is not a refusal, it is wrong output.

**`wq` and `ws` are WEIGHT operands**, and the `w` marker is the load-bearing part of that. Spelt
as plain inputs they name the same bytes, and the core stops resolving their handles and stops
waiting on the mover's event — a read of weight bytes that may not have arrived yet.

## GEMM

| op | params | operands |
|---|---|---|
| `gemm_nt` | `M`(range) `N` `K` `dtype` | `a`, `b`w → `y` |
| `gemm_nt_bias` | `M`(range) `N` `K` `dtype` `act`/opt | `a`, `b`w, `bias`w, `res`? → `y` |
| `gemm_nt_q` | `M`(range) `N` `K` `group` `dtype` | `a`, `a_scale`?, `b`w, `b_scale`w → `y`; `a_sum`?, `b_ref`w? |
| `gemm_nt_q_bias` | `M`(range) `N` `K` `group` `dtype` | `a`, `a_scale`, `b`w, `b_scale`w, `bias`w → `y` |

`nt` is A row-major `[M,K]` times B **transposed** `[N,K]`, giving `[M,N]` — the layout every
weight in a transformer already has. `dtype` distinguishes `bf16` from `w4a8`, so one op name
covers the family and the selector picks by the quant string. That is deliberate: an architecture
plugin that switches quant changes one string, not its op names.

`gemm_nt_q` takes activation scales as an operand because the quantiser produced them; a kernel
that folds dequantisation into its epilogue still receives them.

## Attention

| op | params | operands |
|---|---|---|
| `attn_paged` | `q_len`(range) `head_dim` `gqa` `block_size`/der `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out`, `tree_ancestors`?:in |
| `attn_dense` | `M`(range) `head_dim` `gqa` `causal` `dtype` `scale`:f64/opt `max_seqlen`/opt | `q`, `k`, `v`, `cu_seqlens` → `out` |
| `kv_store` | `M`(range) `head_dim` `n_head_kv` `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `k`, `v`, `slot_mapping`, `kv_cache`:inout |
| `rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `scale`:f64 `mode` `rotary_dim`/opt `sections`/opt | `qkv`:inout, `positions` |
| `rope_table` | `M`(range) `rot` `theta`:f64 `scale`:f64 `dtype` | `positions` → `cos_sin` |
| `qk_norm_rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `eps`:f64 `scale`:f64/opt `mode`/opt `rotary_dim`/opt `wadd`:f64/opt | `qkv`:inout, `positions`, `q_w`w, `k_w`w |

`attn_paged` is the ranged op whose band is `q_len`, not `M`: one query row at decode, up to 64 at
a speculative verify, thousands in a prefill chunk, and libr4d has a different kernel for each
band. `block_size` is **read off the resolved kernel**, not chosen — declare the group, declare
the op, then ask `rad_kv_block_size()` (`spec.md` §7.2).

`cu_seqlens` is optional and makes the batch **ragged**: `[n_seq+1]` i32, sequence `s` owning query
rows `[cu[s], cu[s+1])` of the dense `q` and `out` blocks. Absent, every sequence must present the
same query length and `total_q` must divide `n_seq`. Pass it: a mixed step — one sequence's prefill
chunk beside other sequences' decode rows, which carry their own verify windows and so are not one
row each — is what the scheduler builds whenever prefill and decode coexist, and it does not
satisfy the uniform form. `q_len` is then the batch MAXIMUM, which is what selects the band; the
per-sequence lengths come from `cu_seqlens` alone.

`attn_dense` is the vision tower's: non-causal, nothing paged, `cu_seqlens` bounding one image per
entry so a whole batch is one launch. `q`, `k` and `v` may each be `[tokens, heads, head_dim]` or
`[tokens, heads * head_dim]`, and the second form may be a column slice of a fused projection read
at that projection's row pitch -- which is how a tower's `[q|k|v]` row reaches it without a copy.

**`rope`'s multi-component modes.** `mrope`, `imrope` and `axial` rotate with NeoX pairing and let
several position components drive different frequencies. `positions` is then `[C, M]`,
component-major, and `sections` names how many of the `rotary_dim / 2` frequencies each component
owns: `mrope` gives contiguous runs of them to components 0, 1, ... (Qwen2-VL); `imrope` interleaves
-- component 1 takes frequency `i` where `i % 3 == 1` and `i < 3 * s1`, component 2 where
`i % 3 == 2` and `i < 3 * s2`, component 0 the rest (Qwen3-VL); `axial` gives contiguous runs AND
restarts the frequency ladder in each, `theta^(-2j / (2 s_c))` for the run's `j`-th frequency (a
vision tower's 2-D rotary over row and column). With `[M]` positions every component is that one
position, so `mrope` and `imrope` over a text batch are `neox` exactly.

**`gemm_nt_bias`'s epilogue** is a linear layer's: the biased product rounded to `y`'s dtype, then
`act` of it (`none`, `gelu` exact, `gelu_tanh`) rounded, then `res` added and rounded -- each step
what a separate tensor would hold, so the fused form is the unfused chain to the bit.

**`grid_embed`** adds a learned `side x side` position table resampled to each row's grid: `coord`
is `[4, M]` (row, column, grid height, grid width), sampled bilinearly with the grid's corners on the
table's (align_corners), edge taps clamped, the four taps summed in f32 and rounded to `x`'s dtype
before the add.

| op | params | operands |
|---|---|---|
| `attn_paged_gate_quant` | `q_len`(range) `head_dim` `gqa` `block_size` `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt:capacity `group` `act` | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out`; `gate` → `q8`, `q8_scale` |

**`attn_paged_gate_quant` is `attn_paged` followed by `gate_quant_fp8` over its output**, the
second op finished inside the first's split-KV merge. A merge row is one (token, head) pair of
`head_dim` columns, which is exactly a `gate_quant_fp8` row, so the merged values are gated
(`bf16(sigmoid(gate))`, rounded before the multiply), rounded, written to `out` and quantised to
E4M3 with one scale per 128 columns while they are still in registers. It is byte-for-byte the
pair, checked by `r4d_selftest` at both merge forms and over a bf16 and an E4M3 cache. `block_size` is required rather than derived
because the cache's page size is `attn_paged`'s to answer; the caller passes the value it read. The
gated attention block takes it on the QSA sparse path, where every launch is one query row a
sequence and the decode kernel serves every chunk -- the whole step in one launch, which is why
`max_seqs` is a capacity: the block declares it at the step size, and the activation arena's
smaller declares bound it lower.

### The KV cache is an operand

`attn_paged` and `kv_store` take the group's cache as operand 1, passed as `RAD_KV(group, layer)`.
`RadKVGroupBatch` carries the slot mapping, the block table and the sequence lengths, but the pool
base belongs to the block manager and not to the batch.

### `qk_norm_rope_gate` — device-only

| op | params | operands |
|---|---|---|
| `qk_norm_rope_gate` | `M` `head_dim` `n_head` `n_head_kv` `rot` `q_dtype` `eps`:f64/opt `wadd`:f64/opt `cs_f32`/opt `pos_i64`/opt | `q_gate`?, `k`, `cos_sin`, `positions`, `q_w`w?, `k_w`w → `q_out`?, `k_out`, `gate_out`? |

It splits an interleaved `[q|gate]` projection per head, RMS-norms q and k over the
head, applies partial NeoX rope and copies the gate — four launches a layer in one. **It is NOT
`qk_norm_rope`** in the attention table above, which takes `theta`: this one takes a
PRECOMPUTED `cos_sin` table (`rope_table` builds it) and a different operand list, which is why it
carries its own name. `rot` is the rotary width, `cs_f32` and `pos_i64` state the widths of the
table and the position vector, and `gate_out` is optional — a caller that already reads the gate at
its column offset inside the projection passes none.

**`n_head` 0 IS THE K-ONLY FORM**, and it is the half of this op that has nothing to do with
gating. The `k` side is the plain reader — one head per `head_dim` columns, taken at the
OPERAND's own row pitch, written contiguously to `k_out` — and the only thing the q side adds is
the `[q|gate]` split. So an input that is already plain asks for `n_head` 0 and passes no
`q_gate`, no `q_w` and no `q_out`. The QSA indexer's query heads are that shape: `heads` of them
inside a `heads + 1` projection, which is a stride no rank-2 view can express and would otherwise
be `heads` separate `rmsnorm` launches plus a `rope`. At 4 heads over 12 trunk layers and 4 MTP
passes that is 80 launches a step for ~7 µs of work each; as one op it is 16.

| op | params | operands |
|---|---|---|
| `qk_norm_rope_gate_kv_store` | `M` `head_dim` `n_head` `n_head_kv` `rot` `q_dtype` `eps`:f64/opt `wadd`:f64/opt `cs_f32`/opt `pos_i64`/opt `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `q_gate`?, `k`, `cos_sin`, `positions`, `q_w`w?, `k_w`w → `q_out`?, `k_out`, `gate_out`?; `v`, `slot_mapping`, `kv_cache`:inout |

**`qk_norm_rope_gate_kv_store` is the prologue followed by `kv_store`**, in one launch. A k head's
block already holds its token's rotated k one element a lane, so it writes that element and the
head's v element into the cache at the token's slot as well. The `k` the store reads is `k_out`
and is not an operand of its own, which is why the tail adds only `v`, `slot_mapping` and
`kv_cache`; every parameter is one of the two ops' own. It is byte-for-byte the pair at either
cache width: the store converts the rotated k exactly as `k_out` stores it, a bf16 cache takes the
two values as they are, and the E4M3 conversion rounds each element on its own.

**It IS byte-identical to the kernels it replaces**, and the check is in `r4d_selftest` rather than
in an argument — two rows, one for the `[q|gate]` form and one for the K-only form on a strided
projection. Byte equality and not a tolerance, and the difference between the two is the whole
reason: fold the sum of squares over eight consecutive elements a lane where `rmsnorm` folds it
over one and the two agree to a ULP, which a RELATIVE bound calls ok — but a ULP in the mean square
is a ULP in the normed value, and a rotated element is a difference of two products, so where those
nearly cancel the ULP comes out as a few percent and the transcript moves after a few dozen steps.
The kernel reproduces the reference's fold exactly: the widths above the head dim contribute zeros
and adding 0.0f is exact, so a block sum over 8 wave partials and over 32 are the same number.

## Mixture of experts

| op | params | operands |
|---|---|---|
| `router_topk` | `M`(range) `n_expert` `top_k` `norm` `dtype` | `logits` → `expert_ids`, `expert_w` |
| `moe_scatter` | `M`(range) `n_expert` `top_k` `expert_base`/opt `protect`:str/opt | `expert_ids` → `sorted_tok`, `expert_offset`, `expert_count` |
| `router_topk_scatter` | `M`(range) `n_expert` `n_local`/opt `top_k` `norm` `dtype` `expert_base`/opt `protect`:str/opt | `logits` → `expert_ids`, `expert_w`, `sorted_tok`, `expert_offset`, `expert_count` |
| `moe_gemm` | `M`(range) `N` `K` `n_expert` `top_k` `a_order` `dtype` | `a`, `w`wt, `sorted_tok`, `expert_offset` → `y` |
| `moe_gather` | `M`(range) `n` `top_k` `dtype` `act`:str/opt | `y_expert`, `expert_w`, `sorted_tok`, `shared`?, `shared_gate`? → `y` |
| `row_topk` | `M`(range) `N` `R` `vocab_off`/opt `dtype` | `x` → `idx`, `val`, `pairs`? |
| `row_topk_merge` | `M` `R` `world_size` `dtype` | `gathered` → `idx`, `val` |
| `logit_rerank` | `M`(range) `R` `n_vocab` `n_embd` `vocab_off`/opt `dtype` | `x`, `head`w, `idx_in` → `idx`, `val`, `pairs`? |

`row_topk` and `row_topk_merge` are a pair, and the second only means anything with the first's
optional half. A vocab-sharded head produces logits for its own slice and nothing else, so its
top-R is over that slice: `vocab_off` shifts the columns reported into GLOBAL ids, and `pairs`
writes the same R entries as `[M, R, 2]` (an int32 id in the first float slot, its value in the
second) -- the plane `core/sample/sampler.h` holds as `buf_pairs_`, and the one the
sampler's own vocab-parallel path gathers. An `all_gather` with `row = R*2` concatenates those
into `[M, world_size * R, 2]`, rank r's at `[r*R, (r+1)*R)`, and `row_topk_merge` reduces it to
the global top-R. **The merge is exact, not approximate**: a candidate in the global top-R is in
its own shard's top-R, and both stages break ties on the lower id, so the result equals a top-R
over the unsharded plane element for element. A slot a shard could not fill carries id -1 in
`pairs` and is dropped -- a sentinel rather than a -inf value, because a masked logit takes that
value legitimately.

`logit_rerank` is the second half of a COARSE top-R. A speculative step can afford a cheap draft
head -- the target verifies every proposal, so a worse draft costs acceptance and never a token --
but a head quantised hard enough to be cheap also picks a different top-1 often enough to matter.
So the coarse head proposes R and this op scores those R, and only those R, against the
full-precision head the trunk already carries: a few hundred KB a row against a few hundred MB, for
an answer that agrees with the exact argmax on almost every row. `idx_in` carries GLOBAL ids as
`row_topk` reports them and `vocab_off` is this rank's first column, so under vocab-parallel
drafting each rank rescores the candidates it owns and the existing `all_gather` + `row_topk_merge`
still finds the global winner -- it needs no collective of its own. Ties break on the LOWER id, the
rule the other two use, so a tie inside a shard and a tie between shards resolve alike.

**`router_topk_scatter` is those two in one launch**, and it exists because neither of them is
work. At a decode shape each is a SINGLE 256-thread workgroup ranking about a kilobyte — the
selection is a wave a row and four rows is one workgroup; the sort is one workgroup by
construction — so each costs the one-workgroup dispatch floor and almost nothing else. At 52 of
each per step that is ~1.0 ms, 3.6% of the decode step, for two dispatches. The chain is closed:
the sort reads `expert_ids` and nothing else, and `expert_ids` is exactly what the selection
writes, so a launch between them buys only a barrier and `__syncthreads()` is that barrier.

**Two expert counts, and they are not the same number.** `n_expert` is the ROUTER's — it is
replicated and every rank picks from every expert — while `n_local` is THIS RANK's slice, which is
what the sort runs over and what sizes `expert_offset` and `expert_count` — half of the router's
on Qwen3.8-Flash-Next when its experts are divided between two ranks, as a calibration run divides
them. `n_local` absent means the slice is the router, which is every deployment that gives each
rank a slice of every expert instead.

**Bounded, and it refuses rather than falling back.** Only `moe_scatter`'s single-workgroup form is
fusible (`M * top_k <= 1024`); above it the sort is three launches, which exist to keep it stable
without an atomic deciding the order. That bound is the decode band by construction — eight
sequences at depth 3 and top_k 10 is 320 slots — and a prefill chunk is past it, which is also
where the pair is amortised over real work and a fusion would buy nothing. A caller past the bound
gets a refusal, not a silent change of cost.

`moe_scatter`'s output is what `rad_route_report` hands the heat engine. The histogram is read
back asynchronously and consumed on the next step; nothing here blocks the compute stream
(`spec.md` §5.5).

**`protect` sorts a few experts last.** A layer that keeps some experts at a higher precision than
the rest serves them with a second grouped GEMM, and a grouped GEMM takes one contiguous run of the
offsets. So `protect` lists local experts (ascending, at most eight) that the sort places after
every other expert, in the order listed, with the rest closed up: `expert_offset` and
`expert_count` are then in that order, the regular experts are `[0, n - p)` of the offsets and the
protected ones `[n - p, n)`. A `moe_gemm` handed the trailing run (offsets that begin above zero)
serves those rows and leaves the ones below untouched.

**`moe_gather` folds the shared expert in when `shared` and `shared_gate` are given**, computing
`y = bf16(shared * act(shared_gate)) + bf16(gathered)` in one pass -- which is `scale_rows`' own
fused gate-and-add form, taken one op further back so the bf16 plane between the two never
exists. BOTH NARROWS ARE PART OF THE CONTRACT: the unfused pair writes the gather to a bf16
buffer and narrows the product before the sum, so an implementation that carried either in f32
would move the last bit of every channel. `act` is "sigmoid" or "none" and applies to the
SCALAR, the same spelling `scale_rows` uses. It is worth a pair of operands because a shared arm
that cannot be width-sharded runs on ONE RANK, so the op it replaces is a dispatch a layer that
every other rank waits out inside the block's all-reduce.

### The quantised and gated arms

| op | params | operands |
|---|---|---|
| `moe_gemm_q` | `M`(range) `N` `K` `n_expert` `top_k` `group` `a_order` `dtype` `N_odd`/opt `K_odd`/opt `parts`/opt | `a`, `a_scale`?, `w`wt, `w_scale`wt, `sorted_tok`, `expert_offset`, `w_odd`wt, `w_odd_scale`wt → `y` |
| `moe_gemm_q_mod4` | `M`(range) `N` `K` `n_expert` `top_k` `group` `a_order` `dtype` `N_1`/opt `K_1`/opt `N_2`/opt `K_2`/opt `N_3`/opt `K_3`/opt `parts`/opt | `a`, `a_scale`?, `w`wt, `w_scale`wt, `sorted_tok`, `expert_offset`, `w_1`wt, `w_1_scale`wt, `w_2`wt, `w_2_scale`wt, `w_3`wt, `w_3_scale`wt → `y` |
| `scale_rows` | `M`(range) `n` `act` `dtype` | `x`, `s`, `add`? → `y` |

The `wt` suffix is the table form and `w` alone is an ordinary weight operand; `moe_gemm`'s row
further up carries the same suffix, because its weight list is a per-expert pointer table. The
arithmetic is unchanged by that — a kernel may also be handed the stacked reading, an ordinary
rank-3 tensor with a real leading stride, and computes the same function from it.

**`moe_gemm_q`'s experts are two tables by parity.** `w`/`w_scale` hold the even-numbered experts
at `[N, K]` and `w_odd`/`w_odd_scale` the odd ones at `[N_odd, K_odd]` (defaulting to `N`, `K`);
expert `e` is entry `e / 2` of its class. Every entry of one table shares a geometry, and a rank
that holds an uneven slice of every expert's width — three 128-column blocks of one parity's
experts and two of the other's, which is how a five-block expert splits across two ranks evenly on
average — has two. `parts` stacked parts run along both the weight rows and the output columns (two
for a fused gate/up); the output is `max(N, N_odd)` wide, and a narrower class fills the leading
columns of each output part and writes zeros after them, so the gated quantiser behind it reads
one layout for every row. Under the equal-division rule above the op's weight list is the even
class's codes, its scales, then the odd class's codes and scales.

**`moe_gemm_q_mod4` is the same GEMM over four classes, by the expert's number mod 4.**
`w`/`w_scale` hold class 0 at `[N, K]` and `w_q`/`w_q_scale` class `q` at `[N_q, K_q]` (defaulting
to `N`, `K`); expert `e` is entry `e / 4` of class `e % 4`, so class `q` holds `ceil((n_expert - q) /
4)` of them. Four ranks splitting a five-block expert cannot share the extra block evenly over two
classes -- whichever two ranks take it carry three blocks of every even/odd pair against the other
two's two -- and over four they can: rank `r` takes two blocks of class `r` and one of each other,
five of every four experts on every rank. It is a separate op and not a parameter on `moe_gemm_q`
because the declaration divides a table op's weights over its table operands by count alone: the
operand list is the class count. Everything else -- `parts`, the zero fill of a narrower class, `a_order`
-- is `moe_gemm_q`'s, and the weight list is class 0's codes and scales, then class 1's, 2's and
3's.

### `a_order`: which rows a grouped GEMM's activation has

A routed block runs two `moe_gemm_q` ops and **they disagree about what a row of `a` is**, which is
why this is a parameter and not a convention.

- The **gate_up** projection reads the block's normed input: one row per **token**. A sorted row
  `i` belongs to the pair `sorted_tok[i] = token * top_k + slot`, so it must gather
  `a[sorted_tok[i] / top_k]`. That is `a_order = "token"`, the default.
- The **down** projection reads gate_up's output through the gated quantiser, which is already one
  row per **(token, slot)** in the scatter's own order. Row `i` is row `i`. That is
  `a_order = "sorted"`.

`a_scale` has a row for every row `a` has, either way.

**Why it is said and not derived.** The row count gives it away — `M` against `M * top_k` — and
deriving it is the tempting shortcut. It is also the shortcut no harness catches: a kernel and a
reference that *both* gather agree to 1e-4 on every case, and `rad-kbench` builds each operand from
the op's own shape hook, so a hook describing `a` at the extent the gather wants builds a case that
is self-consistent and wrong. Served, the down projection hands every slot of a token the *first*
slot's intermediate — at decode, where a step carries one token, all `top_k` experts read slot 0 —
and the model runs, and answers fluently, and gets facts wrong. An operand whose meaning changes
with its extent cannot be checked by a harness that sizes the operand from the same reading.

**`moe_gemm_q` is a separate op and not a `dtype` on `moe_gemm`**, for the same reason `gemm_nt_q`
is separate from `gemm_nt`: the operand LIST differs, operands are positional, and an op whose
count depends on a string means different things to different kernels. The group sub-sum is
accumulated from the raw codes and scaled once per group, exactly as `gemm_nt_q` does — the two
references have to agree in the last bits or `rad-kbench` blames a kernel for a choice made in
`libref`.

**`moe_gemm` itself is what a model served at bf16 runs**, and libr4d serves it at `dtype` bf16:
one weight table of the checkpoint's own row-major `[N, K]` planes, read with no layout hook, so a
container stores an expert as the safetensors file holds it. It has no parity classes and no
`parts` — those exist because a quantised expert's width must fall on its scale blocks, and an
unscaled plane has none — and its rows go through the same run grouping and the same three forms
as `moe_gemm_q`'s kernels, byte-identical to each other.

**`scale_rows` is the per-token scalar gate**, and it exists because `mul`'s broadcast is the wrong
one. `mul` lets `b` carry a single ROW broadcast down the rows; a shared-expert gate is a single
COLUMN broadcast across them — `y[m, i] = x[m, i] * act(s[m])` — and adding a second broadcast rule
to `mul` would make the two ambiguous exactly when `rows == n`. `act` is `"sigmoid"` or `"none"`.
Qwen3.5-MoE's `shared_expert_gate` is a `[1, n_embd]` projection whose one output per token gates
the shared expert's contribution, and this is that multiply.

## Linear attention (gated delta net)

| op | params | operands |
|---|---|---|
| `gdn_conv_prep` | `M`(range) `head_k` `head_v` `chunk`/der `conv_width` `l2_eps`:f64/opt `softplus_thr`:f64/opt | `x`, `w`w, `b`w?, `A_log`w, `dt_bias`w, `conv_state`:inout, `cu`, `a`?, `b_gate`?, `cache_idx`?, `has_init`? → `q`,`k`,`v`,`g`,`beta` |
| `gdn_conv_update` | `q_len`(range) `head_k` `head_v` `conv_width` `max_query_len`/opt | `x`, `w`w, `b`w?, `conv_state`:inout, `state_idx`, `num_accepted`, `cu` → `q`,`k`,`v` |
| `gdn_kkt_solve` | `M`(range) `head_k` `chunk`/der | `k`, `beta`, `g`, `cu` → `A` |
| `gdn_chunk_scan` | `M`(range) `head_k` `head_v` `chunk`/der `scale`:f64/opt `state_fp16`/opt | `q`,`k`,`v`,`A`,`g`,`beta`,`h0`?,`cu` → `o`, `ht`?; `state_idx`? |
| `gdn_recurrent_update` | `q_len`(range) `head_k` `head_v` `n_head_v`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`:str/opt `state_fp16`/opt `state_form`:str/opt | `q`,`k`,`v`,`a`,`b`,`A_log`w,`dt_bias`w,`state`:inout,`state_idx`,`num_accepted`,`cu`,`z`?,`norm_w`w? → `o`,`o_q`?,`o_scale`? |
| `gdn_conv_recurrent_update` | `q_len`(range) `head_k` `head_v` `conv_width` `n_head_v`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`:str/opt `state_fp16`/opt `state_form`:str/opt | `x`, `conv_w`w, `conv_b`w?, `conv_state`:inout, `conv_idx`, `a`,`b`,`A_log`w,`dt_bias`w,`state`:inout,`state_idx`,`num_accepted`,`cu`,`z`,`norm_w`w → `o`,`o_q`?,`o_scale`? |
| `gdn_gated_rmsnorm` | `M`(range) `channels` `eps`:f64 `act`:str | `x`, `z`, `w`w → `o` |

**A linear-attention kernel pair must support speculative rollback.** The recurrent state and the
conv window have already absorbed rejected tokens, so `num_accepted` is a required operand and the
state cache is a rolling window of `conv_width-1 + n_spec` entries read at the slot the last
accepted token left. A rejection is a change of read offset, not a recompute. This is part of the
schema, not an assumption (`spec.md` §10).

**`num_accepted` is ONE-BASED.** It counts the tokens the previous step committed: the one it was
given, plus however many of its drafts survived. So 1 is an ordinary decode step that drafted
nothing, which is exactly what these kernels assume when the operand is absent, and 0 is not a
value either of them can read -- `gdn_conv_update` reads its window at `num_accepted - 1`.

**`state_form` says what the state cache MEANS, and there are two answers.** `per_candidate` --
the default, and what this table's operand description above says -- writes one full state per
candidate token and needs `1 + n_spec` distinct slots in `state_idx`. `anchor` keeps ONE committed
state a sequence and replays the accepted prefix onto it out of a scratch slot, which is two slots
whatever the draft depth. The two produce the SAME `o` from a COMPLETELY DIFFERENT `state`, so an
implementation that serves one and is handed the other computes a correct answer to the other
question and disagrees with a checker on every element of the cache while agreeing on every
element of the output. That is why it is declared and not inferred: the index row's width does not
say which, and neither does anything else a kernel can see.

A caller declares it; an implementation that serves only one form REFUSES the other by name.
libref serves `per_candidate`. The engine picks the form from whether the deployment speculates,
which is the same condition that sizes the state pool.

It is also REQUIRED rather than optional, which is one of the two places the union rule is
deliberately not applied: an op that lets the caller omit `num_accepted` is an op whose rollback
contract is optional. A non-speculative decode passes all ones.

**`state_idx` is `[n_seq, W]`, and W is the group's, not the op's.** One column is the sequence's
state. A kernel that needs somewhere to put what it cannot commit yet may ask for more, and the
columns after the first are ITS scratch: the caller stages the row, zeroes every slot it names when
the sequence is admitted, zeroes the scratch again after any step that ran a different path over
the same sequence (a chunked prefill commits the state directly and writes no scratch), and
otherwise does not look inside. `gdn_chunk_scan` takes the same row and reads column 0.

The point of that freedom is what libr4d does with it. One state per candidate token is the plain
reading of "the next step reads the slot the last accepted token left", and on Qwen3.8-27B it is
1.3 GiB a sequence. Instead libr4d keeps ONE committed state and, beside it, the few hundred floats
a head that each token's rank-1 update needs; the next step replays the accepted prefix onto the
state out of that block and starts from a state carrying exactly the tokens that survived. Two
slots a sequence instead of `1 + n_spec`, and less write traffic than the per-candidate form.

Two consequences worth naming. A zeroed scratch block replays as the identity, which is what makes
"zero it and pass `num_accepted` 1" the correct handover out of a prefill. And **libref is
not a meaningful oracle for `gdn_recurrent_update` while that layout is in use** -- ref implements
the per-candidate reading, so it would decode another plugin's scratch as its own. The end-to-end
check that does hold is stronger anyway: greedy speculation must emit the same bytes as greedy
without it.

**Every gdn kernel takes `cu`, decode included, and that is the other place the union rule is not
applied.** `gdn_conv_update` and `gdn_recurrent_update` run off the one cumulative-lengths tensor
on the one step, and both open by reading `cu[n]` and `cu[n+1]` before any null test — the number
of sequences is `cu`'s length minus one and nothing else in the argument list carries it. A caller
working from a schema that made `cu` optional passes a null and faults on the dereference rather
than being refused.

**`gdn_conv_recurrent_update` is the decode pair as one launch**: `gdn_conv_update` followed by
`gdn_recurrent_update`, with the convolution computed where the recurrence reads it, so `q`, `k`
and `v` are never written. The operands are the convolution's inputs -- `conv_idx` is its
`state_idx` -- and then the recurrence's without those three; the parameters are the union of the
two ops' under their own names, and both ops' rollback and state contracts hold unchanged. `z` and
`norm_w` are required, because the fold serves the decode shape, where the fused row norm already
puts one workgroup on each (sequence, value head). It is byte-identical to the pair. libr4d
rewrites a key head's conv state once all the value heads reading it have read it, by an arrival
count, so nothing waits on another workgroup.

**The output gate `z` may be a COLUMN SLICE, on both `gdn_recurrent_update` and
`gdn_gated_rmsnorm`.** The fp8 delta-net block projects `[q|k|v|z]` in one GEMM
(`arch/common/rad_block_gdn_fp8.h`), so the gate is the right-hand `n_head_v * head_v` columns of a
wider row and its `(token, head)` rows are NOT evenly spaced: heads are `head_v` apart inside a
token and the next token is the slice's own stride away. An implementation that reads one pitch off
`z` walks into the next token after `n_head_v` rows and is wrong without being out of bounds. The
three spellings a caller may hand over are `[tokens, heads, head_v]`, a flat
`[tokens, heads*head_v]` buffer of its own, and that slice; the first two are the same arithmetic
and the third is what needs the token stride carried separately.

## Collectives

| op | params | operands |
|---|---|---|
| `all_reduce` | `world_size` `numel`(range) `dtype` `exact` `min_bytes`/opt `hops`/opt | `x`:inout, `y`:out? |
| `all_gather` | `world_size` `numel` `dtype` `row`/opt | `x` → `y` |

`exact` separates the exact sum from the quantised wire, and a caller has to say which it wants: a
request that does not declare it will accept a lossy wire gets no kernel rather than the quantised
one. That is libr4d's rule and it is right.

libr4d separates one-shot from two-shot, at a width that has both, by message size: its two-shot
rows start above 524288 elements, which makes that a band edge of the ranged `numel`, so a caller
states nothing for it. `hops` is a caller's request for one or the other (1 or 2); libref accepts
and ignores it, and no libr4d row reads it. Every all-reduce kernel serves `y` absent, in place.

libr4d serves these at **2, 3, 4 and 8 ranks**. Two ranks have their own kernels (one flag word a
block, the rotated 6-bit wire, the fused norm-and-reduce forms). The wide family is one template on
the world size; 4 and 8 have both hops, 3 has the one-shot only -- the two-shot shards the message
`world` ways and a residual width such as 2560 does not divide by 3 -- and `all_gather` at width
rides the same instance as the wide `all_reduce` (`ar_gather_3rank`, `_4rank`, `_8rank`). The
fused forms are declared only at two ranks by the arch blocks, which fall back to the plain pair at
any other width.

`min_bytes` is the byte floor under which a LOSSY row must serve the message exactly anyway, and it
is a parameter rather than a constant in the library because it is a property of the link and not
of the kernel. Compression only pays once the transfer is bandwidth-bound: a decode step
all-reduces a handful of residual rows and the exact kernel already moves that at most of what the
interconnect delivers, so the rotate, the pack and the unpack cost more than the bytes they remove.
A row selected for `exact=0` therefore decides the ROW once, at declare, and the WIRE per launch.
Absent, the plugin uses its own measured default.

`all_gather.row` is one row of a rank's contribution, in elements. Absent or `0`, `y` is the plain
concatenation — rank `r`'s slice at `r * numel`. Present, the placement is row-interleaved: `x` is
`[rows][row]` and `y` is `[rows][world_size][row]`, which is what a vocab-parallel candidate merge
reads, because it takes one row of gathered candidates per sampled position at a single row stride.

**`-ffp-contract=off` is mandatory for any quantised cross-rank op.** Two ranks see the two
dequantised products in the opposite order, and contracting either into an FMA makes them disagree
by ~1 ULP, which breaks the replicated-state invariant. The build sets it globally for device code.

### The collectives that carry a norm — device-only

| op | params | operands |
|---|---|---|
| `ar_rmsnorm_quant_fp8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`:inout, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `ar_hc_write` | `M`(range) `n` `hc` `world_size` `dtype` `wire`/opt | `y`:inout, `inj`, `h`:inout |
| `ar_gather_hc_write` | `M`(range) `n` `hc` `top_k` `world_size` `dtype` `act`/opt `wire`/opt | `y_expert`, `expert_w`, `sorted_tok`, `shared`?, `shared_gate`?, `y`:inout, `inj`, `h`:inout |
| `ar_ln_had_quant_i8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gdn_gated_norm_had_quant_i8` | `M` `n` `head_dim` `group` `eps`:f64 `dtype` `zgrp`/opt `zblk`/opt | `x`, `z`, `w`w → `q`, `scale` |

`wire` is the payload width on the link: 0 is exact bf16, 6 the rotated six-bit form. It is a
property of the LINK and not of the kernel, which is why it is a parameter — see `all_reduce`'s
`min_bytes` for the same argument. `x` is INOUT on `ar_rmsnorm_quant_fp8`: it is both the message
and the input, and the reduced sum lands in it.

`ar_hc_write` is `all_reduce` of `y` followed by `hc_write` of the reduced `y`, for a block whose
output feeds a gated residual rather than a norm. `y` is INOUT for the same reason `x` is above.
It serves both wires, one op a wire: a caller declares the rotated form beside the exact one and
issues whichever the step's message goes out on, and a `wire` of 6 is 6 at every size. At four
ranks only the exact form exists. A caller states `wire` 0 on the exact form rather than leaving it
out: a row that serves one wire only says so with a constraint on `wire`, and a constraint on an
absent key does not hold.

`ar_gather_hc_write` is `moe_gather` followed by `ar_hc_write`: the gather's output IS the
message, so `y` is written by the gather half before it is reduced in place, and the five gather
operands and `act` mean exactly what they mean on `moe_gather`. `wire` is `ar_hc_write`'s. On
either wire, byte-identical to the three kernels it stands in for. At four ranks only the exact
form exists, as for `ar_hc_write`.

## Vocabulary edges

| op | params | operands |
|---|---|---|
| `embed_lookup` | `M`(range) `n_embd` `n_vocab` `dtype` `vocab_offset`/opt `wscale`:f64/opt | `tokens`, `wte`w → `x` |
| `embed_lookup_q` | `M`(range) `n_embd` `n_vocab` `dtype` `vocab_offset`/opt `wscale`:f64/opt | `tokens`, `wte`w, `scale`w? → `x`, `ahead`? |
| `logits_gemm` | `M`(range) `n_vocab` `n_embd` `dtype` | `x`, `lm_head`w → `logits` |
| `ngram_ids` | `M`(range) `heads` `ngram` `eos` | `tok`, `state`:inout, `cu`, `mult`w, `vocab_sizes`w, `offsets`w, `cache_idx`?, `has_init`?, `num_accepted`? → `ids`, `ahead_ids`? |
| `ple_gate` | `M`(range) `n` `hc` `eps`:f64 `dtype` `wadd`:f64/opt `gate_eps`:f64/opt | `k`, `q`, `v`, `w_key`w, `w_query`w, `w_conv`w → `gv`, `gvn` |
| `ple_conv` | `M`(range) `n` `width` `dilation` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `resid`?, `cache_idx`?, `has_init`?, `num_accepted`? → `y` |

Separate from `gemm_nt` because `lm_head` is its own access class (`RAD_ACCESS_VOCAB`) and because
the sampler wants to fuse against it: at a 151K vocab and batch 256 the logits are ~77 MiB a step,
and the distributed top-k means a rank never materialises a full row (`spec.md` §9, §13).

### `gather_rows`

| op | params | operands |
|---|---|---|
| `gather_rows` | `M`(range) `n` `dtype` | `x`, `idx` → `y` |

`gather_rows` is not a convenience. Without it a plugin computes logits for **every** token in a
prefill chunk instead of the last one per sequence — `n_tok / n_seq` times too much work, and at
`max_tok` 8192 against a 248K vocab the logits buffer alone is 8 GiB. It is the op that turns
"correct" into "affordable".


### `scatter_rows`

| op | params | operands |
|---|---|---|
| `scatter_rows` | `M`(range) `n` `dtype` | `v`, `idx`, `x`:inout |

`x[idx[i], :] = v[i, :]`, the mirror of `gather_rows`, and `x` is **inout** because the write is
partial: rows no index names keep what they held. That is what makes it usable against a pool other
sequences are also in — an op that returned a fresh `x` would have to copy everything it was not
asked to change.

Two rules differ from the gather, and both follow from the destination being live memory rather
than a fresh buffer:

- **A negative index writes nothing.** `gather_rows` reads a zero row for one, which is the padding
  convention a short prefill chunk leaves behind. On this side a zero fill would be data loss.
- **Indices must be distinct**, declared with `RAD_OPD_F_IDX_UNIQUE` rather than left to prose. Two
  rows landing on one destination has no defined winner on a parallel machine, so permitting it
  would make this an op whose reference implementation and whose real kernel are allowed to
  disagree — which is exactly what the oracle exists to rule out.

It exists for the same reason `gather_rows` does: the alternative is `M` separate transfers, and
where this op is used a row is a few hundred bytes, so that is dispatch cost with the data as a
rounding error. The caller is the KV tier traffic (`core/mem/kvtier.h`). A paged pool is
**layer-major**, so within one layer a group's blocks are dense rows of `layer_bytes_per_block()`
— which makes packing an arbitrary set of them exactly a gather and restoring them exactly a
scatter, one launch per (group, layer) however many blocks are moving.

**At `dtype` u8 the rows are opaque bytes**, and libr4d supplies that width as its own kernel pair
rather than widening the bf16 one. What moves is whatever the attention kernel wrote; calling it
bf16 would matter, because `libref` is the oracle for every op here and its bf16 path loads through
float, which is not guaranteed to return a NaN payload unchanged. At u8 every value is 0..255,
float carries it exactly, and oracle and kernel agree bit for bit on any payload at all.

### `embed_lookup`'s `vocab_offset`

`vocab_offset` (optional, default 0) is the first vocabulary row this rank holds. With it,
`n_vocab` is THIS RANK'S row count -- the meaning it already carries for `logits_gemm` -- and the
table covers global ids `[vocab_offset, vocab_offset + n_vocab)`. **Token ids stay GLOBAL**: a rank
subtracts the offset and writes a zero row for an id belonging to another rank, exactly as it does
for the negative padding id, so summing the ranks reconstructs `x`. Every token lands on exactly
one rank, which is why the `all_reduce` that follows is declared `exact` rather than taking the
model's wire setting: it sums disjoint rows, so a lossy wire would be loss on a value that is
already the answer.

Without it the embedding must be REPLICATED while `lm_head` is row-sharded, and asymmetric sharding
of the two ends of the same vocabulary is not a design, it is an omission. The table is bf16 even
in the fp8 checkpoint, so replicating it costs 248320 x 5120 x 2 = 2.43 GiB on every card --
memory the split hands to the KV cache instead.

One thing is given up and it is worth naming: under a split this kernel can no longer tell a
garbage id from another rank's token, so the "an id at or above `n_vocab` fails the launch" check
holds only where the rank holds the WHOLE vocabulary -- `world_size` 1, not merely `vocab_offset`
0. **Rank 0 of a vocab-parallel split has offset 0** and legitimately sees ids past its own slice
on most steps, so a check keyed on the offset alone refuses correct calls. Bounding ids is the
caller's, and the tokeniser and the sampler both already are.

### `embed_lookup_q`, the n-gram gather

**A gather from a table quantised with ONE scale for all of it**: `x[m] = wte[id] * scale[0]`.
Padding and bounds behave exactly as `embed_lookup`'s — a negative id writes a zero row, an id past
the table fails the launch in the host kernels and writes a zero row in the device one, which
cannot read its ids before it runs (libr4d's divergence note).

**Why one scale is enough**, measured on the bf16 original: E4M3 carries its own exponent, a row's
dynamic range in this table is only 2.3×, and the scale sits on a **three-octave plateau** where
2⁻¹³ through 2⁻¹⁰ all give rel_l2 0.02662 to five digits. A per-row scale buys 2.9%.
Qwen3.8-Flash-Next-FP8 ships exactly this: 128 E4M3 shards and a single bf16 `weight_scale` of
1.9932e-4.

**A separate op and not a `dtype` on `embed_lookup`.** The operand list differs — there is a scale
— and operands are positional, so an op whose count depends on a string means different things to
different kernels. It is `gemm_nt_q`'s argument against being a `dtype` on `gemm_nt`. Adding the
scale to `embed_lookup` as an optional operand would also have moved `x` from slot 2 to slot 3 for
every caller in the tree.

**Where it runs is where `--ngram-placement` puts the table**, because the planner runs an op on
the host when **every weight operand of it** is `Site::Host` and takes that from the table's
placement. The table is 51.2 GB and is never staged a unit at a time — moving a whole unit to use
5 KB is the wrong shape — so it is read where it lies:

| placement | the table | the gather |
|---|---|---|
| `disk` | the container's mapping | libavx's host kernel |
| `ram`, libr4d loaded | a pinned copy every card maps at its host address | libr4d's device kernel, over the link |
| `ram`, no device kernel | a locked copy (`core/format/table_copy.h`) | libavx's host kernel |
| `vram` | the card, every rank holding all of it | libr4d's device kernel |

`hipHostRegister` on the container's mapping is not one of them: it would force all of the file
resident.

**The host gather keeps its own row cache.** The container is mmapped, so the table is at a host
address on open. libavx keeps the rows a gather has touched in a row cache of its own and reads a
miss straight from the file with O_DIRECT (`libavx/avx_ngram.cpp`): the page cache would map a
4 KiB page per 160-byte row. The reads keep the alignment the filesystem reports for direct I/O, or
its block size where it reports none, as btrfs does; one read of the table proves it before the
first row is served. A filesystem with no direct I/O at all cannot hold a container with this table
on `disk`.

**The device gather is libr4d's `embed_lookup_q_bf16`**: four entries a thread, each E4M3 code
decoded exactly, multiplied by the scale in f32 and rounded to nearest even — the value libavx's
lookup table holds, to the bit. It reads the ids and writes `e` in the host arena, where the
architecture declares them for the host gather, so the step has no host op left in it.

The table has to be a weight operand at all for the host site to be reachable: `gather_rows` takes
a plain `IN` operand and would never be placed there. And the host kernel has to live in a
production plugin — libavx, the host kernel library — because `rad_gate_reference_kernels`
refuses to serve on libref by design.

### `ngram_ids`, and the gather that is not an op

**Qwen4-Exp's Per-Layer Embedding gives one layer a second embedding table addressed by a HASH of
the last `ngram` tokens.** Qwen3.8-Flash-Next's is 320,001,536 rows of 160 — 102 GB, 51 of the
model's 180 billion parameters — read sixteen rows a token. `ngram_ids` computes those row ids:

    mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]
    ids[h]   = remainder(mixed(b+1), vocab_sizes[h]) + offsets[h]

where `heads` is `ngram - 1` equal blocks and block `b` uses the `(b+2)`-gram. `t_s` is the token
`s` positions back, **or the EOS token if the shift would cross one** — and once a shift falls back
to EOS every deeper shift does too. `mult`, `vocab_sizes` and `offsets` are weights because the
checkpoint carries them; the reference builds them from a seed and a prime search at construction
time, and re-deriving sixteen numbers the container already holds would be a second source for them.

**`state` is a rolling window of committed ids**, `ngram - 1 + n_spec` deep, read at
`num_accepted - 1` and rewritten in the layout `ple_conv` describes below — operand for operand,
including the two layouts `num_accepted`'s presence chooses between. **A cold window is EOS and not
zero**, which is the one place it differs from a convolution window: zero is a real token id.

**The ids are a STATE and not a plain input**, and the reason is that the contract is already paid
for. A state needs a rollback contract and an input does not, which reads like an argument for the
input — but the dilated convolution downstream carries a 9-timestep window with exactly this
contract, so PLE pays for `num_accepted`, `cache_idx` and `has_init` regardless, and against that a
plain input costs *more*. Nothing in the engine derives "the ids before this step": `DeriveInput`
carries positions and cu_seqlens and no token ids at all, so an input would mean a new core
derivation and a widened `DeriveFn` signature to serve one op. `RAD_KV_CONV` is already
"per-sequence, a rolling window of width-1 + num_spec entries", and `conv_state_index` already
derives the read cursor for it.

**The gather is `embed_lookup_q`, not `gather_rows`.** The table has to be a WEIGHT operand or
`Site::Host` is unreachable, because the planner puts an op on the host only when every weight
operand of it is `Site::Host`, and `gather_rows` takes a plain `IN` operand.

**`ahead_ids` and `ahead`: the next chunk's rows, a step early.** A prefill chunk gathers `heads`
rows a token and, with the table on `disk` and text the row cache has not seen, most of them are
random reads of the container file — ~30 ms a 2048-token chunk on an NVMe, with the card idle behind
the host-site op.
The next chunk of the prompt is already known (`RadBatch::n_ahead`, which stores it in `token_ids`
after the step's own tokens), so:

- `ngram_ids` with `ahead_ids` present reads `tok` that many tokens past the step's, as a
  continuation of the LAST sequence, and writes their ids there — exactly the ids the step that
  runs those tokens will compute, since it hashes the same history. The window it writes back is
  the step's own and never sees them.
- `embed_lookup_q` with `ahead` present may start reading those rows once its own are gathered.
  No output depends on it, so a kernel may ignore it; libavx's row hands them to a reader thread
  that fills its row cache while the step runs, and the next call finds them there.

### `ple_gate`, and what the convolution branch does not see

PLE's mixing coefficient is a **dot product between the n-gram embedding and the residual stream** —
one scalar per hyper-connection stream, per token:

    K[c] = grouped_rmsnorm(k)[c] * (w_key + wadd)      k is key_proj(embedding)
    Q[c] = grouped_rmsnorm(q)[c] * (w_query + wadd)    q is the INCOMING hc stream
    s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
    s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))
    gv[c] = sigmoid(s[c]) * v                          v is value_proj(embedding), ONE row
    gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)

**The signed square root keeps the sign**, so a stream whose embedding opposes the residual gets a
gate below a half rather than a gate of zero. The clamp is on the **magnitude** and the sign comes
from the **original**, which matters at exactly one value: torch's `sign(0)` is 0, so a dot product
of exactly zero comes out as zero and not as `sqrt(gate_eps)`. Clamping first and re-signing the
clamped value would give 1e-3 there.

**Both outputs leave the op.** The dilated convolution that follows reads the *normed* copy and the
residual add reads the *un-normed* one — two different tensors, and recomputing either outside
would be a second definition of the norm.

**AND THE CONVOLUTION BRANCH DOES NOT SEE THE GATE.** `gv[c]` is a scalar times one shared `v` row,
and `norm_conv`'s rms is taken per `n`-wide stream — so the gate appears in both the numerator and
the rms and divides straight back out. `gvn[c][d]` is `v[d]/rms(v)` times its gain, whatever the
gate did. This is a sharp invariant to check a kernel against: a fused implementation whose `gvn`
varies with the gate has folded something it should not have. It is *not* a shortcut to take —
`gv` is narrowed to its dtype before the third norm reads it, so computing `gvn` from `v` directly
would differ in the last bits.

**One op and not six.** Three grouped norms, a reduction, a scalar nonlinearity and a broadcast
multiply over tensors that are all `[M, hc*n]` — 10240 wide here — would be six launches against a
~4.5 µs per-kernel floor, every one of them memory-bound. `hc_read` is the same argument at the
same width.

### `ple_conv`: a dilated convolution cache is not a conv cache

    y[t] = resid[t] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation] )

**The dilation changes how deep the history is, and that is the whole trap.** At Qwen4-Exp's width
4 and dilation 3 the taps are **9, 6, 3 and 0** timesteps back, so a sequence carries
`(width-1)*dilation = 9` timesteps — not `width-1 = 3`. A cache sized for the undilated width keeps
only the three most recent values and reads zeros where the deepest tap belongs, which is a model
that generates fluently from a slightly wrong context.

The state is `[n_slots, n, state_len]` with slot `i` holding the value at relative position
`i - (width-1)*dilation`, which is `gdn_conv_prep`'s convention with the dilation folded in.
`has_init` 0 means that window is **zeros** rather than whatever the slot last held — without it a
fresh sequence landing on a recycled slot convolves the previous sequence's tail into its first
nine tokens.

**`num_accepted` shifts the READ offset, and its presence picks what the step leaves behind.** The
manager sizes the state `(width-1)*dilation + n_spec` deep so a speculative step can read at an
offset inside it. What the next step's offset must find is the history ending at the last token
this step *committed*, and which token that is depends on the kind of step:

- **Present — a decode step**, any of whose drafts a verify may reject. The window is rewritten the
  way `gdn_conv_update` rewrites its own: the history read, shifted down by one, then every input of
  the step, so slot `i` holds position `i + 1` of (window read ++ step). A next step that kept `k`
  of these tokens reads at `k - 1` and finds the history ending at the `k`-th. It needs
  `(width-1)*dilation - 1` slots plus one a token, which a step of `1 + n_spec` tokens gets.
- **Absent — every token commits**, which is the prefill chunk. The read is at offset zero and the
  last `(width-1)*dilation` inputs land at offset zero, where the next step's `num_accepted` of 1
  reads them.

The two agree when the step has one token, and no single layout can serve both at several: at
offset zero one must hold the history ending at the step's first token and the other the history
ending at its last. An architecture therefore issues the op once over its decode rows and once over
its prefill rows, as it already does the two `gdn` convolutions.

**One op where the gated delta net has two.** `gdn_conv_prep` and `gdn_conv_update` are separate
because the work *around* their convolutions differs — one splits a fused projection and takes an
l2 norm, the other does not. PLE's convolution is the same arithmetic at prefill and at decode, so
it takes the union of the two contracts instead: `has_init` is prefill's question and
`num_accepted` is decode's.

The residual add is folded in because the reference is `gv + silu(conv(gvn))` and the two tensors
are both `[M, hc*n]` — 10240 wide here. `resid` absent leaves the plain convolution.

## Sampling

Every sampler is a kernel, not a host lift. `llama.cpp`'s chain is kept as a **host reference**
that `rad-kbench` compares the device samplers against (`spec.md` §13).

**EVERY PER-REQUEST SCALAR IS AN OPERAND, NOT A PARAMETER.** `params` is `[M]` rows of
`RadSampleParams` (`abi/rad_sample.h`), uploaded once a step, and it carries `temp`,
`top_k`, `top_p`, `min_p`, `typical_p`, the three penalties, the DRY and XTC settings, the seed and
position, the history window and the grammar's `mask_row`. Not one of them is in a params cell
below, and the reason is that a parameter is GEOMETRY: the selector matches on it and freezes it at
declare (`spec.md` §2.2), so two requests in one batch with different `top_p` would be two
resolutions of one op — which is not something a batch can contain.

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

`sample_mask` is where a GBNF grammar lands: the state machine stays on host and what crosses is a
token bitmask per sequence, 19 KiB at a 151K vocab, cheap enough to upload every step.

### Stages the decomposed chain does not carry

| op | params | operands |
|---|---|---|
| `sample_dry` | `M` `n_vocab` `vocab_off`/opt | `logits`:inout, `params`, `history`, `breakers`? |
| `sample_typical` | `M` `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_xtc` | `M` `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_chain` | `M` `n_vocab` `temp`:f64/opt `p`:f64/opt `min_p`:f64/opt `k`/opt | `logits`, `query`?, `u` → `draw` |

`sample_dry` penalises each token that would extend a repeated suffix of the history ONCE, by its
longest such repeat — llama.cpp's DRY. The sequence breakers are strings, so they are resolved
against the vocabulary on the host: the row's `dry_rep_limit` carries the repeat limit the most
recent breaker imposes, and the optional `breakers` plane, one byte beside every history token,
marks the tokens that are whole breakers and are never penalised. Absent, no token is exempt.

`sample_penalties` and `sample_dry` read GLOBAL token ids from `history` and write this rank's
logits, so under a split vocabulary they take `vocab_off`, as `sample_topk` does, and a token
outside the shard is another rank's to penalise. Their `penalty_last_n` / `dry_penalty_last_n`
window is llama.cpp's: 0 turns the stage off for the row, and a negative value takes the whole
history.

`sample_typical` is locally-typical sampling and is NOT a prefix of the descending candidate order —
that order and the probability order disagree, which is why it narrows in place rather than
truncating. `sample_chain` is the whole chain in one kernel for the speculative verify: `query` is
the drafted token per position (absent at the last one), `u` is TWO uniforms a position, and `draw`
is `{u32 tok, u32 tok_excl, f32 p_query}` — the rejection test's three values. It stands beside the
decomposed stages rather than replacing them; both are declared and the band decides.

### The vocabulary-parallel chain

When `lm_head` is row-sharded, a rank's logits row covers only its own slice of the vocabulary, and
two things follow. `sample_argmax` becomes unusable — its answer is a LOCAL column index, which is
a different token — so a greedy request is staged at `top_k = 1` and takes the chain instead. And
`sample_topk` writes an optional third output, `pairs`: its candidates as (u32 GLOBAL id, f32
logit), with the rank's `vocab_off` already added, which is what `all_gather` moves and
`sample_merge_topk` reads.

`sample_merge_topk` **re-applies the row's `top_k`**, and that is why it takes `params`. Every rank
ran its own top-k before the gather, so the merged set holds up to `world_size * k` candidates
where the request asked for `k`, and no stage after it narrows again. At `k = 1` the difference is
not a slightly wider support: `sample_pick` is a DRAW, so a greedy row with two live candidates
returns whichever one the request's seed happens to select.

**`gathered` is `world_size` SORTED RUNS, and the merge relies on it.** Each rank's `sample_topk`
writes its `n_cand` pairs in descending (logit, lower id first) order with its holes at the tail,
and the all-gather lays a row's runs side by side. libr4d's merge therefore runs only the merge
stages of its sorting network when the runs are a power of two long and fill the row -- 11 passes
over two runs of 1024 where a full sort is 66 -- and the whole network otherwise. A producer that
hands it unsorted runs gets a wrong order, not a slow one.

## Sparse attention (QSA) — device-only

| op | params | operands |
|---|---|---|
| `qsa_block_key` | `M`(range) `n` `ratio` `rotary_dim` `pos0`/opt `block_size`/opt `eps`:f64 `theta`:f64 `scale`:f64/opt `wadd`:f64/opt `dtype` `mode`/opt `sections`/opt | `k`, `w`w, `page`?, `pos`? → `out` |
| `qsa_score` | `M`(range) `n` `heads` `blocks` `seqs` `dtype` `tp_partition`/opt `m_offset`/opt | `q`, `bk`, `bt`, `cu`, `nc` → `score` |
| `qsa_select` | `M`(range) `blocks` `topk` `ratio` `seqs` `tp_partition`/opt `m_offset`/opt | `score`, `bt`, `nc`, `pos`, `cu` → `sel`, `nsel`, `seqused` |
| `qsa_tail_store` | `M`(range) `n` `ratio` `ring` `seqs` `dtype` | `k`, `pos`, `cu`, `state`:inout, `sidx` |
| `qsa_work` | `M`(range) `n` `ratio` `ring` `seqs` `work` `dtype` | `k`, `pos`, `cu`, `state`, `sidx`, `bt`, `rope`? → `stage`, `page`, `bpos`, `nc` |

Five ops decompose one thing: an indexer that scores blocks of key history against the query and
hands `attn_paged` a shorter block table. `docs/QSA.md` is that decomposition; these are the
schemas, and `arch/common/rad_qsa.h` issues all five.

`qsa_block_key` builds the **compressed key**: one vector per `ratio` tokens of key history, which
is what the indexer's block scores are taken against.

```
out[b] = rope( rmsnorm( mean over j<ratio of k[b*ratio + j] ),  at position pos0 + b*ratio )
```

**`M` is BLOCKS and `k` is `ratio` times taller.** Every other op in this document has one output
row per input row. A caller that sized `k` at the block count would have every block pool three
rows of the next one — plausible numbers and a different model — so libr4d's entry point checks it.

**The mean is accumulated in f32 and rounded to the activation dtype BEFORE the norm.** That is
the reference's order (`key_groups.float().mean(dim=1).to(raw_keys.dtype)`), and carrying f32
through into the norm is a different number: small, and systematic on every block of every layer.

**`block_size` is the same number as `ratio` and exists to size a KV group.**
`Builder::find_kv_consumer` binds a group to the first op declared after it that carries the
literal key `block_size`, so an op that wants to decide a group's page size has to spell it that
way -- and the block-key store's page *is* the compress block. libr4d refuses a disagreement
rather than picking one.

**`out` may be a KV cache, and is then rank four.** The block-key store's geometry is a container
and not an attention shape -- `kv_heads` 1, `block_size` the compress ratio, `head_dim`
`n/(2*ratio)` -- so a page holds exactly `n` contiguous values and the manager allocates one per
`ratio` tokens. Read that way it *is* `[pages, n]`, and libr4d checks the **product** rather
than the rank, because the rank is the engine's business and the width is the op's. `qsa_score`'s
`bk` is the same object and takes the same reading.

**`page` is optional and changes where the answer goes, not what it is.** The block-key store is
the paged KV group described in `docs/QSA.md`, so block `b` of a sequence lives at whatever page
the manager gave it and consecutive blocks are not consecutive pages. Absent, the write is
contiguous — a caller with its own scratch, or a test. **Present, `M` comes off `page` and not off
`out`**, because `out` is then as tall as the pool and reading the block count from it would write
the whole store on every call. A `-1` entry is a block with no page and is skipped rather than
written somewhere arbitrary.

**`pos` turns a launch from a RANGE into a LIST, and `pos0` is the range.** Blocks tile the token
stream, so within one sequence block `i`'s first token is at `pos0 + i*ratio` and the parameter is
enough — which is what a prefill chunk looks like. It is *not* what a decode step looks like: the
sequences in a batch are at different positions, so the blocks completing in one launch belong to
different sequences and are not a run at all. With `pos` given, a launch is a set of work items
and one of them a layer a step serves any mix.

**A block key is query-independent and permanent** once its `ratio` tokens exist. The reference
recomputes it inside its per-query loop for clarity; an engine computes it when the last token of
the block arrives and caches it. That is the difference between the indexer costing 64 bytes a
token a layer and costing a pass over the whole history every step.

`qsa_score` is the indexer's vote:

```
score[m][b] = ( sum over heads h of relu( q[m][h] . bk[bt[seq(m)][b]] ) ) / sqrt(n)
```

**`m` is a QUERY and `bt` is the SEQUENCE's, and `cu` is the join.** At decode the two are the same
list — one query a sequence — but a prefill chunk is many queries over one history and each of
them selects its own set. `cu_seqlens` is the only operand in a batch that says which sequence a
token belongs to, so it is what turns a query row into a table row.

**`nc` bounds the block loop and is required.** `blocks` is a *declared* width, sized for
`max_ctx / ratio`, so a scorer that trusted it would read 65536 padding entries a query on a
200-token prompt. `nc[seq]` is the sequence's live complete-block count — an upper bound for every
query in the chunk, and within `T / ratio` of each one's own. `qsa_select` narrows it to the exact
per-query bound, which it can because it has `pos`.

**And `score` is narrowed by the CALLER to the live reach, because a grid cannot read `nc`.** Both
ops size their launch from the declared width -- the live count is device data and the grid is a
host-side number -- so an op issued at the full `blocks` launches for 200K of context whatever the
prompt is. The architecture narrows the score operand's column count to
`(max_ctx_len + n_tok + n_spec + ratio) / ratio + 1`, which the batch already carries on the host,
and the pitch stays the declaration's. Every term of that is slack on purpose: an UNDER-estimate
drops blocks from the selection and nothing anywhere would say so, and the `+ 1` is what keeps
`n < blocks`, which is what makes `qsa_select`'s partial page -- `bt[n]` -- readable at all.

**The relu is per (block, head) and the sum is over heads, in that order.** Four independent
votes, each of which can only ever argue *for* a block. Summing first and clamping after would let
one head's large negative cancel another's positive — a different model that still ranks blocks
plausibly, which is the worst kind of difference.

**`bt` holds PAGE IDS, not block indices.** The attention KV group is paged at the compress ratio,
so a block and a page are the same thing: the sequence's existing block table is already the list
to score, and the top-k over the output is already the block table the attention reads. That
identity is the design, and an op that took block indices would put a translation on both sides
of it.

**A negative table entry scores `-inf`, not zero.** A block table is padded to its declared width
and a real score can be zero — every head vetoed the block — so zero would let padding outrank a
block the model deliberately ignored.

`qsa_select` turns those scores into the table the attention reads: the `topk` highest-scoring
blocks as **page ids**, with the partial page appended.

**`nc` is the complete-block count and is a separate operand from the table's width**, because the
two differ by exactly the thing the op treats specially. `bt[n]`, when it is not `-1`, is the
**partial page** — the tokens after the last complete block, which the reference attends
unconditionally and which the query itself lives in. It is appended after the selection and is
never a candidate for it, so `sel` is `topk + 1` wide and the last slot is not spare: sized at
`topk` it would drop the newest tokens on three queries in four.

**The bound `n` is the QUERY's, not the sequence's**: `n = min(nc[seq(m)], (pos[m] + 1) / ratio)`,
because block `b` is entirely in this query's past exactly when `b*ratio + ratio - 1 <= pos`. At
decode the two agree by construction. `cu` is the join, as in `qsa_score`.

**`qsa_select` takes OPTIONAL scratch, and it is what lets the selection use the device.** One
workgroup a query is the natural grid -- a selection is a reduction over the whole block list --
and at decode that is `1 + n_spec` workgroups on a 64-CU part, 6% of it, walking 50000 blocks five
times. Given an arena the kernel instead cuts the list into chunks, takes each chunk's own top-k,
and merges: **exact**, because a chunk's local top-k contains every global top-k member of that
chunk, and bit-identical in the OUTPUT LAYOUT too -- the attention sums the selected pages in table
order and float addition is not associative, so "the same set" is not the same text. Without the
arena the op runs the single-workgroup kernel, which is what a caller with no scratch, or a launch
wider than a decode step, gets.

**The partial page exists exactly when `(pos + 1) % ratio != 0`, and that is a causality rule.** A
decode query on a block boundary has `n == nc` and `bt[nc]` is the page after the sequence's last,
which the KV manager leaves `-1` — so testing the page instead of the position would be right for
the only case decode presents, and wrong mid-chunk, where the same query has *later tokens of the
same chunk* already stored in `bt[n]`: a live page entirely in its future, which `seqused` would
then cover in full. A model that attends to its own future is not a wrong ranking; it is a model
that cheats, and it would show up as an implausibly good prefill and nothing else.

**Last is load-bearing.** Every selected *complete* page is entirely in the query's past, so the
only causal masking left is inside the final page — and putting it at the end of the table makes
the attention's own `kk <= q` bound do exactly that, with no change to the attention kernel. The
effective query position the caller passes is then `(nsel - 1) * ratio + pos % ratio`.

**Ties break to the lower block index, through a prefix count.** A block score is a sum of four
relus over a shared grid, so equal scores are ordinary rather than pathological, and a top-k
placed by whichever atomic landed first would not reproduce across runs — which this tree checks
by comparing generated text byte for byte.

**`seqused` is why QSA needs no new attention kernel.** `attn_paged` takes its block table and its
per-sequence length as *ordinary operands* — the architecture reads them off the KV group batch
and passes them — so the sparse attention is that same op issued with `sel` in place of the
sequence's table and this number in place of its length. The flat KV it then walks is `nsel` pages
of `ratio` with the partial page last, so the query sits at `(nsel-1)*ratio + pos%ratio` and the
length is one past it. One formula covers both cases: with a tail the query is inside that last
page, and without one `pos%ratio` is `ratio-1` and it collapses to `nsel*ratio`, which is every
selected page fully visible — correct, because a complete page is entirely in the query's past.
It is emitted here rather than by an op of its own because it is one store off a number this
kernel already has.

`qsa_tail_store` parks the raw index keys of the tokens that have not completed a block yet:
`state[sidx[s][0]][pos[t] % ring] = k[t]`. A block key is the mean of `ratio` consecutive raw
keys, so at decode three of the four are from earlier steps and something has to carry them.

**The tail is a LINEAR KV STATE, not a plane indexed by the batch row.** It is read on a later
step than the one that wrote it, which makes it state and not scratch, and state has two axes a
buffer cannot express: the LAYER (the keys are `index_qk_proj`'s output, and that weight is per
layer) and the SEQUENCE. The batch row is step-local — the scheduler builds a step from the
running set in priority order and stable-sorts decode first — so a request that outlives an
earlier one shifts down a row and inherits its keys. A linear group gives a slot that is the
sequence's for its life, zeroes it when it hands one out, and saves and restores it with the rest
of the linear state when the prefix cache checkpoints. `state` is the group's cache for one bound
layer, `[slots, 1, ring, n]`; `sidx` is that group's `state_index` row and column 0 is the
committed slot.

**`cu` is how a token finds its ROW**, and it is the only per-token index in the batch that says
so — `slot_mapping` is a flat KV slot and carries no sequence at all. With a handful of sequences
a linear scan of the boundary list costs nothing beside the copy it guards. Which STATE that row
belongs to is `sidx`, and the two are different questions.

**The ring is `ratio + n_spec`, and speculation is the whole of the difference.** A verify step
runs `1 + n_spec` positions and the next step restarts at the first REJECTED one, so a position
this step has already overwritten is owed to the next. At `ratio` 4 and depth 3 the step covering
positions 9..12 writes rows 1, 2, 3, 0 — and row 0 is the one position 8 is still owed, because
the block holding 8..11 is rebuilt as soon as the following step starts at 10. The widest window
a step can still owe is `[p0 - ratio + 1, p0 + n_spec]`, so a ring that long means a row is only
ever overwritten by a position past the end of any step that could ask for it. With `n_spec` 0 the
ring is the ratio.

**Only the last `ring` tokens of a sequence's chunk write, and that is not an optimisation.**
Positions `p` and `p + ring` land on the same row, so a launch that wrote every token of a prefill
chunk would have several workgroups racing for one row and the winner would depend on the
schedule. Restricting the write makes every row the target of at most one token — their positions
differ by less than `ring`, so their rows differ — and the result is what a serial
last-writer-wins loop leaves. Removing the restriction makes the case report `rows DIFFER` **and**
`two runs identical NO`, which is the race saying so out loud.

**`state` is INOUT and rows nothing targets keep what they held.** That is the whole point across
steps: at decode a handful of rows are written and the rest are what the next block key needs.
The selftest counts the untouched rows of the slots the batch does NOT name separately, because
that is the count a batch-row index destroys.

**Under a multi-component rotary** (M-RoPE) a block key rotates at its first token's three
components, not at its index. `qsa_work` takes the step's `rope` planes (`RadBatch::rope_pos`,
`[3, T]`) and then writes `bpos` as `[work, 3]`, a block's three components side by side (rows first,
so a smaller step is fewer rows of the same buffer): a first token in the step reads its
own components, and one committed in an earlier step -- always text, since a prefill chunk ends on a
whole block and a prompt ends in text -- takes its index plus the shift the step's first row of that
sequence carries. `qsa_block_key` with `mode` (`mrope`, `imrope`) and `sections` then reads the
component each frequency's section names, as `rope` does.

`qsa_work` is the gather that turns a batch into the flat block list `qsa_block_key` reads.
Nothing in a batch hands it that list: the keys of a block completing at **decode** are three rows
of the tail state and one of this step, the keys of a block completing inside a **prefill chunk** are
four rows of the chunk, and a block straddling the chunk's start is some of each — while the
blocks themselves belong to different sequences at different positions.

So this op walks the sequences, works out which blocks became complete, copies each one's `ratio`
raw keys into consecutive rows of `stage`, and writes its destination `page` and first-token
position beside it. **The block-key kernel never learns that two sources existed.**

**`M` is tokens and `work` is blocks**, and they are different numbers: one block per `ratio`
tokens plus at most one straddling block per sequence. **`work` is a HOST bound on how many blocks
can complete, not a count of how many did** — the count is device data, and slots past it get
`page = -1`, which is the sentinel `qsa_block_key` already skips. So the bound costs empty
workgroups that exit on their first load and never a wrong answer.

**`bt` is the BLOCK-KEY group's table**, not the attention group's — see `docs/QSA.md` on why
there are two and which op takes which.

**`state` and `sidx` are the tail**, read at row `q % ring` of slot `sidx[s][0]` — see
`qsa_tail_store` just above for why it is a KV state and not a plane. A sequence with no slot
(`sidx[s][0] < 0`) gets `page = -1` on its blocks rather than a page built out of whatever the
pointer arithmetic reached.

**`nc` is complete blocks per sequence**, which the scorer and the selection both need and which
this op has already computed on its way to the work list. It is `(last position + 1) / ratio` and
*not* the number of blocks that completed this step — the two are the same only on a sequence that
starts at zero. A separate op for one division per sequence would be a launch a layer a step for
four instructions.

The copy is a kilobyte per completing block: 512 KiB for a full 2048-token prefill chunk a layer,
four kilobytes for a decode step. The alternative is a block-key kernel that takes two sources, a
per-key source selector and a tail layout — three more things to get wrong inside the kernel that
does the arithmetic.

## Draft heads

A drafter is an ordinary graph over ordinary ops except at its two ends: what proposes the
candidates, and what enters the head from the trunk. Those two ends are what these ops are.

| op | params | operands |
|---|---|---|
| `fulldrafter_two_stage` | `M` `block_stride`/opt `dtype` | `ktap0`, `ktap1`, `ktap2`, `ktap_anc`, `tokens_in`, `gpu_params` → `tokens_out` |

`fulldrafter_two_stage` is the discrete two-stage block drafter (K=12, 2x6) with Markov correction and Top-16 Selector.

### DFlash2

| op | params | operands |
|---|---|---|
| `dflash_select` | `M` `steps` `top_k` `rank` `n_vocab` `anchor_stride`/opt `dtype` | `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w → `tokens` |

`dflash_conv` is the drafter's grouped convolution, and device-only:

| op | params | operands |
|---|---|---|
| `dflash_conv` | `T` `hidden_size` `taps` `group_size` `NG` `block_size` `dtype` | `x`, `delta`, `base`w → `out` |
| `dflash_conv_add` | `T` `hidden_size` `taps` `group_size` `NG` `block_size` `dtype` | `x`, `delta`, `base`w, `residual` → `out` |

`dflash_select` is the greedy walk over the candidate lattice: `cand`/`unary` are `row_topk`'s two
output planes at one row per draft position, `hp` the selector's rank-256 query, and `anchor` the
step's own token ids — `anchor_stride` is the block width, because a draft pass carries `block` ids
a sequence and the anchor is the first of them. `tokens` is THE DRAFT, and the engine reads it by
name rather than through the sampler.

**`hp` IS OPTIONAL, AND ABSENT IT IS A PLANE OF ONES** — which turns the edge score
`unary + Σ_r pred[prev][r]·hp[l][r]·succ[cand][r]` into the plain low-rank BIGRAM
`⟨pred[prev], succ[cand]⟩`. That is exactly DSpark's Markov head: a rank-256 learned bigram bias
added to the draft logits, decoded greedily, with the predecessor read after the previous step's
argmax inside one workgroup. Written literally it is `steps` full-vocabulary [rank, n_vocab]
matrix products — 466 MiB a step to use seven rows — and through this operand it is 8 KiB of
codebook gather. The only approximation is the top-K candidate set, which can cost acceptance and
nothing else.

`dflash_tree_select` is the SAME edge arithmetic widened into a bounded tree, and it is a separate
op rather than a mode of `dflash_select` so the linear path stays untouched and unselected:

| op | params | operands |
|---|---|---|
| `dflash_tree_select` | `M` `steps` `top_k` `rank` `n_vocab` `anchor_stride` `tree_budget` `tree_width` `depth_penalty`:f64 `dtype` | `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w → `packed_tree` |

The first six operands are `dflash_select`'s, in its order, and `hp` is optional at index 2 with the
same meaning (absent is a plane of ones). `packed_tree` is the root-inclusive layout
`mine/dloop_tree.hip` writes and `dloop_tree_from_packed` reads, `I32[M, 1 + 2*capacity]` — 33
columns at the declared capacity of 16: `[0]` count (1 + draft nodes; 0 when the row is invalid),
`[1+i]` node `i`'s token with node 0 the anchor, `[1+capacity+i]` node `i`'s root-inclusive parent
(`0` = the anchor), and `[1+capacity]` the anchor's parent — the **`-1` sentinel the validator
requires**, which at capacity 16 is column 17. Unused slots are `-1` in both blocks.

`capacity` is the DECLARED tree size (16 node slots), not the selector's `tree_budget`; they are
different numbers and the pitch is derived from the former.

    edge[level][c] = unary[level][c] + Σ_r pred[pid][r] · hp[row][r] · succ[cand[level][c]][r]
    node[child]     = node[parent] + (edge[child] - max_over_that_parent's_candidates(edge))

Every parent of a level's frontier scores every candidate; each parent's set is then shifted by
that parent's own maximum, so the increment is `<= 0` and a child can never outrank its parent. A
level keeps its best `tree_width` pairs, the tree keeps its best `tree_budget` nodes by
`(node + depth_penalty·depth, depth asc, old id asc)`, and pruning refuses a node whose parent it
did not keep — so the emitted set is always prefix-closed. `steps` is `<= 3` and `tree_budget` is
`<= 15`; outside those bounds the call is refused, never clamped.

**THE RANKING ACROSS PARENTS IS A HEURISTIC, NOT A PROBABILITY.** Shifting each parent's candidates
by that parent's local maximum is a normalisation by local maximum: it puts a parent's best at zero
so that accumulated node scores are comparable between branches, which a chain's raw logits are
not. It is not a log-softmax and `node` is not a log probability; the log-sum-exp term is dropped
because it is a per-parent constant that changes no ordering within the parent, and because
`logf`/`expf` need not agree bit for bit between the device and the host oracle, which this op's
checks compare byte for byte.

THE EDGE ARITHMETIC IS THE SHIPPING KERNEL'S, OPERATION FOR OPERATION: a 32-lane accumulation over
`r = lane, lane+32, …` seeded at zero, the xor-butterfly fold (`16, 8, 4, 2, 1`), then `unary` added
last, all FP32. Both `libr4d` and this tree compile with `-ffp-contract=off`, so nothing contracts
into an FMA on either side. A single-path call (`tree_width` 1, `tree_budget` >= `steps`)
reproduces `dflash_select`'s walk token for token; that equivalence is checked against the real
`libr4d` kernel in `tests/dflash_tree_gpu_test.cpp`, and against an independent host implementation
of the same walk in `tests/dflash_tree_select_test.cpp`.

Two divergences from `dflash_select` are deliberate and documented in `mine/dflash_tree_select.h`:
a candidate id outside `[0, n_vocab)` is not offered (a chain may emit such an INDEX, but a tree
proposes a TOKEN the target must have), and an anchor outside `[0, n_vocab)` writes `count = 0`
(a tree cannot have an unverifiable root). A THIRD is forced by the consumer: a repeated candidate
id under one parent becomes ONE node, keeping the highest raw edge score and, on an exact tie, the
lower candidate index. `set_draft_tree` refuses a tree whose siblings propose the same token, since
under sample-and-match that is one continuation claimed twice; retaining the best-scoring instance
is the linear walk's own argmax rule, so a `tree_width = 1` pick is unchanged.

Fused prologue and projection ops for DFlash2:

| op | params | operands |
|---|---|---|
| `dflash_k_norm_rope_store` | `M` `n_head_kv` `head_dim` `eps`:f64 `wadd`:f64 `theta`:f64 `scale`:f64 `block_size` `kv_dtype` | `k`:inout, `v`, `w`w, `positions`, `slot_mapping`, `kv_cache`:inout |
| `dflash_gated_bf16` | `M` `N` `K` `dtype` | `a`, `b`w → `y` |

`dflash_k_norm_rope_store` fuses RMS norm, RoPE and paged KV store in one launch. `dflash_gated_bf16` fuses the BF16 gate/up projection and SiLU multiply into the GEMM epilogue.

### `mtp_enter`: the MTP head's entry, when the residual stream is wide

| op | params | operands |
|---|---|---|
| `mtp_enter` | `M`(range) `n` `hc` `eps`:f64 `dtype` `wadd`:f64/opt `ngroup`/opt `fc`/opt | `h`, `e`, `w_h`w, `w_e`w, `fc_h`w, `fc_e`w → `x` |

    ne[j]    = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
    fe[i]    = fc_e[i] . ne
    rs[k]    = rsqrt(mean(h[k*n .. k*n+n)^2) + eps)
    nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
    x[k*n+i] = fc_h[i] . nh[k] + fe[i]                         for each of the hc sub-streams

A pre-norm MTP head (Qwen3.5, Qwen3-Next, DeepSeek-V3) joins the trunk's hidden state and the next
token's embedding with **one** `fc` over their concatenation, and `arch/common/rad_block_mtp_fp8.h`
runs that as two `rmsnorm`s into the two halves of one buffer and a single `gemm_nt`. Qwen4-Exp's
head cannot: its hidden state is the **wide** stream and its checkpoint carries `mtp.fc_hidden` and
`mtp.fc_embedding` as two **square** `[n, n]` matrices beside a `[hc*n]` gain and an `[n]` one. So
the hidden half is the same matrix applied to **each** sub-stream — the reference's own
`.unflatten(-1, (hc_count, hidden_size))` idiom, which is how every other 10240-wide tensor in that
architecture is handled — and the embedding half is one `n`-wide product added into all `hc` of
them, the same broadcast the trunk enters the stream with (`hidden_states.repeat(1, 1, hc_count)`).

**One op and not six, and here that is a statement about the ABI rather than about launch count.**
Every intermediate above is a different *view* of the same bytes: `nh` is `[M*hc, n]` to the GEMM
and `[M, hc*n]` to the norm. `core/runtime/issue.cpp` narrows **dim 0 or the last dimension with
every stride left alone**, which is the right rule and makes a row-pitch change inexpressible — so
the unfused spelling cannot be written at all, not merely at a cost. It is also six launches off a
draft round, and a draft round is the latency the head exists to hide.

**`fc` is the two square matrices' STORED format**: `bf16`, or `e4m3` rows with f32 128-column
group scales in each row's own tail, the geometry `hc_read`'s `mix_down` uses. Only the drafts see
it -- the head proposes and the trunk verifies -- so it can cost acceptance and never a token.

**`ngroup` is the one thing the checkpoint does not say.** A `[hc*n]` gain fits both a grouped norm
(`hc` rsqrts, one per sub-stream — what `Qwen4ExpTextGatedResidual` does at every one of its 97
connections) and a flat one (a single rsqrt over the whole wide row). Both readings are
self-consistent, libref shares whichever the graph picks, and **no per-op oracle can reach it** —
exactly the shape of question `rad_block_mtp_fp8.h`'s concat order is. It has the same cheap
empirical answer: a head wired the right way round accepts most of what it proposes, and one wired
the wrong way round proposes noise while the model's own output stays perfectly correct. The head
declares the grouped reading (`ngroup = hc`), and `ngroup` carries it as a parameter so a flat
reading (`ngroup = 1`) is one declaration away rather than a different kernel.

---

## DLoop draft trees and parent-aware GDN/PLE

This section documents **model-specific extensions**, not the conventional
vocabulary above. These ops are served on the device by **`mine`** and on
the host by **`mine_ref`**. Neither `libref` nor `libavx` implements them.
`mine_ref` publishes its own schemas and declares itself a reference plugin, so
`rad-kbench --ref mine_ref` selects its host oracle without changing the
conventional vocabulary or requiring model code in upstream kernel libraries.

The DLoop ops share BF16 weights and F32 activations. `mine/dloop_ops.h` specifies
the original ring contracts and the shared pool; `mine/dloop_full.h` specifies
the five-layer full-prefix contracts. The full-context ops add `context_capacity`
to the parameter list and require `context_window=0`. No launch allocates.

| op | params | operands |
|---|---|---|
| `dloop_ctx_append` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `taps`, `path`, `slots`, `reset`, `ring`:inout, `head`:inout, `length`:inout, `tap1`?, `tap2`? |
| `dloop_base` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `ring`, `head`, `length`, `slots`, `anchor_embed`, `weights`, `head_shard` → `ctxh`, `y_base`, `local_ids`, `local_logits` |
| `dloop_pool` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `gathered_ids`, `gathered_logits`, `anchor`, `head_shard`, `embed_shard`, `pred_shard`, `succ_shard` → `pool_ids`, `unary`, `pool_head`, `pool_embed`, `pool_pred`, `pool_succ`, `anchor_pred` |
| `dloop_dartree` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `ctxh`, `length`, `slots`, `anchor_embed`, `weights`, `y_base`, `pool_ids`, `unary`, `pool_head`, `pool_embed`, `pool_pred`, `pool_succ`, `anchor_pred` → `tree_tok`, `tree_parent`, `n_nodes`, `anchor`?:in, `packed_tree`? |
| `dloop_full_append` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `context_capacity` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `tap0`, `tap1`, `tap2`, `path`, `span`, `slots`, `reset`, `weights`, `cache`:inout, `length`:inout |
| `dloop_full_base` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `context_capacity` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `cache`, `length`, `slots`, `anchor_embed`, `weights`, `head_shard` → `y_base`, `local_ids`, `local_logits` |
| `dloop_full_dartree` | `n_seq` `max_seqs` `tap_rows` `world_size` `hidden_size` `intermediate_size` `num_layers` `num_q_heads` `num_kv_heads` `head_dim` `context_features_dim` `target_width` `proposal_length` `context_window` `active_vocab` `vocab_size` `selector_rank` `conv_kernel_size` `conv_group_size` `vocab_lo` `vocab_size_shard` `pool_size` `budget` `width` `context_capacity` `variant` `eps`:f64 `rope_theta`:f64 `depth_bonus`:f64 `dtype` | `cache`, `length`, `slots`, `anchor_embed`, `weights`, `y_base`, `pool_ids`, `unary`, `pool_head`, `pool_embed`, `pool_pred`, `pool_succ`, `anchor_pred` → `tree_tok`, `tree_parent`, `n_nodes`, `anchor`?:in, `packed_tree`? |
| `gdn_conv_tree` | `q_len` `head_k` `head_v` `n_head_k` `n_head_v` `conv_width` `node_capacity` `history_ring` `n_seq`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`/opt `state_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt `factor_buffer`/opt | `x`, `weights`w, `bias`w?, `history`, `history_idx`, `cu`, `parent`?, `ctx_before` → `q`, `k`, `v`, `saved_raw_x` |
| `gdn_recurrent_tree` | `q_len` `head_k` `head_v` `n_head_k` `n_head_v` `conv_width` `node_capacity` `history_ring` `n_seq`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`/opt `state_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt `factor_buffer`/opt | `q`, `k`, `v`, `a`, `b`, `A_log`w, `dt_bias`w, `state`, `state_idx`, `cu`, `parent`?, `z`, `norm_w`w → `o`, `factors` |
| `gdn_conv_recurrent_tree_i8g` | `q_len` `head_k` `head_v` `n_head_k` `n_head_v` `conv_width` `node_capacity` `history_ring` `n_seq`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`/opt `state_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt `factor_buffer`/opt `max_tokens` | `x`, `weights`w, `bias`w?, `history`, `history_idx`, `cu`, `parent`?, `ctx_before`, `a`, `b`, `A_log`w, `dt_bias`w, `state`, `state_idx`, `z`, `norm_w`w → `saved_raw_x`, `factors`, `o_q`, `o_s` |
| `gdn_tree_commit` | `q_len` `head_k` `head_v` `n_head_k` `n_head_v` `conv_width` `node_capacity` `history_ring` `n_seq`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`/opt `state_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt `factor_buffer`/opt | `state`:inout, `state_idx`, `factors`, `history`:inout, `history_idx`, `saved_raw_x`, `path_nodes`, `path_len`, `ctx_before`, `cu`, `parent`? |
| `ngram_ids_tree` | `q_len` `n` `heads` `ngram` `eos` `width` `dilation` `node_capacity` `history_ring` `n_seq`/opt `mode`/opt `tok_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt | `tokens`, `tok_history`, `tok_idx`, `cu`, `parent`?, `ctx_before`, `mult`w, `vocab_sizes`w, `offsets`w → `ids` |
| `ple_conv_tree` | `q_len` `n` `heads` `ngram` `eos` `width` `dilation` `node_capacity` `history_ring` `n_seq`/opt `mode`/opt `tok_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt | `x`, `weights`w, `conv_history`:inout, `conv_idx`, `cu`, `parent`?, `ctx_before`, `residual`? → `y`, `saved_raw`? |
| `ple_tree_commit` | `q_len` `n` `heads` `ngram` `eos` `width` `dilation` `node_capacity` `history_ring` `n_seq`/opt `mode`/opt `tok_group`/opt `conv_group`/opt `layer`/opt `raw_buffer`/opt | `tok_history`:inout, `tok_idx`, `conv_history`:inout, `conv_idx`, `tokens`, `saved_raw`, `cu`, `parent`?, `ctx_before`, `path_nodes`, `path_len` |
| `mtp_tree_accum` | `M`(range) `pitch` `capacity` `max_rounds` `n_vocab` `round` | `cand`, `token_ids`?, `draft_out_ids`?, `history`:inout, `packed`:inout |
| `mtp_merge_tree_accum` | `M`(range) `pitch` `capacity` `max_rounds` `n_vocab` `round` `world_size` | `gathered`, `idx`:out, `val`:out, `token_ids`?, `draft_out_ids`?, `history`:inout, `packed`:inout |
| `mtp_hybrid_lse` | `M`(range) `N` `n_vocab` `vocab_off` | `coarse`, `idx`, `val` → `stats` |
| `mtp_scored_merge_tree_accum` | `M`(range) `pitch` `capacity` `max_rounds` `n_vocab` `round` `world_size` `policy` `cost0`:f64/opt `cost1`:f64/opt `cost2`:f64/opt `cost3`:f64/opt `cost4`:f64/opt `cost5`:f64/opt `cost6`:f64/opt `cost7`:f64/opt | `gathered`, `stats`?, `idx`:out, `val`:out, `token_ids`?, `draft_out_ids`?, `history`:inout, `log_history`?:inout, `packed`:inout |
| `mtp_pruned_expand` | `M` `round` `n_vocab` | `gathered`, `request_ids`?, `tokens`?, `anchor_rows`?, `positions`?, `meta`:inout, `score`:inout, `packed`:inout |
| `mtp_pruned_prepare` | `M` `round` `n_vocab` | `request_ids`, `meta` → `token_ids`, `positions`, `gather_ids`, `ancestors`, `slots`, `prefix` |
| `mtp_pruned_pack` | `M` `beta`:f64 | `request_ids`, `meta`, `score` → `packed` |
| `mtp_overlay_attn` | `M` `head_dim` `n_head` `n_head_kv` `block_size` `depth` `q_dtype` `kv_dtype` `max_ctx` `N` | `q`, `committed_kv`, `committed_bt`, `prefix`, `provisional_kv`, `ancestors`, `out`:out, `meta`:inout, `request_ids` |
| `attn_paged_tree` | `q_len` `max_tokens` `max_seqs` `max_ctx` `head_dim` `gqa` `n_head` `block_size` `causal` `window` `q_dtype` `kv_dtype` | `q`, `kv`, `block_table`, `seqused`, `cu_seqlens`, `spec_parent` → `out` |
| `qsa_tree_commit` | `M` `n` `ratio` `ring` `rotary_dim` `block_size` `max_tokens` `eps`:f64/opt `wadd`:f64/opt `theta`:f64/opt `scale`:f64/opt | `k`, `pos`, `paths`, `lengths`, `starts`, `tail`:inout, `sidx`, `bt`, `weight`w, `block_keys`:inout |
| `spechub_propose` | `M`(range) `n_vocab` `world_size` | `gathered`, `draw`, `token_ids`, `draft_out_ids` → `state`, `packed`, `idx`, `val` |
| `spechub_verify` | `M`(range) `n_cand` `n_vocab` `transport`/opt | `cand_idx`, `cand_val`, `params`, `state`, `token`:inout |
| `mtp_couple_propose` | `M`(range) `n_vocab` `round` `max_rounds` | `idx`:inout, `val`, `seq_ids`, `draw`, `state`:inout |
| `mtp_couple_verify` | `M`(range) `n_cand` `n_vocab` | `cand_idx`, `cand_val`, `params`, `state`, `token`:inout |

`attn_paged_tree` serves BF16 queries/output and FP8 E4M3 KV with head dimension
256, twelve query heads, GQA12 and four-token pages. Each sequence contains
one to 65 ordered nodes; `spec_parent` uses sequence-local I32 indices.
Attention sees the committed prefix, self and ancestors, never sibling nodes.
The device keeps dot products, softmax, PV and split partials in FP32; the
registered `mine_ref` oracle uses independent two-pass FP64 arithmetic.
This is a reordered path, not bit-exact to the ordinary F16/PF16 rows.
Invalid device metadata zeroes the affected sequence; invalid tensor shapes
and unsupported geometry return errors before launch.

The Qwen TP2 block also declares this op for small trees: ordinary prefill
attention ignores ancestor masks, including when a longer prefill suffix
selects its band for a mixed batch. Pure decode through five rows retains
the existing correct masked decode kernel. Small-tree declarations reserve
at most sixteen rows per sequence; wide declarations reserve 65.

DLoop candidate pools use only active-vocabulary logits. Equal cutoff logits
keep the lower token ID, and pools store IDs ascending. This matches the
Drafter's repaired `_candidate_pool`; plain `torch.topk` does not guarantee a
tie order. Tree selection matches stable descending argsort on parent-major
pool order. Pruning breaks equal priorities by depth, then original node order.
The block attention is bidirectional; context padding is masked before projection.
The packed backbone follows manifest tensor order excluding the two codebooks,
without padding. Student weights and teacher inputs are BF16; intermediate
activations and accumulation remain F32, including convolution's separate base
and dynamic products. Reordered F32 reductions can differ from PyTorch.
`n_seq` may be ranged; launches derive the actual count from narrowed sequence
operands. The optional trailing `anchor` remains an input despite its position
after the split outputs. Pool packets also accept dense `[world_size,S*M*K]`.
Pool and search inputs require producer-consistent IDs, scores and gathered rows;
generic independent random fills cannot express those cross-operand invariants.

Full-context cache layout is F32 `[max_seqs,num_layers,context_capacity,2*KV_heads*head_dim]`;
keys are already rotated at absolute committed positions, followed by values.
Append consumes three BF16 teacher-tap planes. A nonnegative span count selects a
contiguous prefill range; count -1 selects the terminated accepted-path row list.
Only slot initialization resets the length. The cache never wraps and never receives
rejected siblings. Base/tree query and block-key positions start at the slot's actual
committed length, not the allocation capacity. Both ops read historical cache without
mutating it or reconstructing it at each tree depth.

GDN verification keeps committed recurrent/convolution state immutable. Commit
replays only the selected root-inclusive path; no bonus row enters either state.
The optional group/layer/buffer parameters bind that transaction to the declared
architecture state. Parents are sequence-local, root `-1`, and precede children.

`gdn_conv_recurrent_tree_i8g` combines tree convolution, recurrence and the existing
128-channel INT8 activation quantizer in one device launch. It requires 128-wide
key/value heads and convolution width four. Convolution and gated-output BF16
rounding remain explicit; recurrent state and replay factors remain F32.
Q/K/V stay inside the kernel. Outputs are I8 codes and F32 scales, plus the
unchanged raw-input and replay-factor records. Winner commit remains separate.
`max_tokens` bounds operand rows and sizes the host oracle's declared scratch;
the host oracle composes the existing convolution and recurrence without
allocating during launch. Other geometries retain the unfused operations.

`RADIANCE_SPECHUB=1` selects a separate one-position, two-sibling experiment.
It requires serial Qwen MTP with the TP2 reranked candidate head, two proposal
slots, `RADIANCE_DARTREE=1`, adaptive drafting off, and
`RADIANCE_NO_PROMPT_LOOKUP=1`. It does not combine with score/cost/pruned policies.
The GPU merges eight global candidates, forms their temperature-one softmax
`q`, and samples `Q(x,a)=q[x]`, `Q(a,x)=q[a]*r[x]`, where `a` is the hub
and `r` normalizes the nonhub tail explicitly. It builds the root-inclusive
tree on the device. True pointmass support emits one deduplicated child;
malformed inputs and finite exponential underflow fail rather than silently
changing the proposal law.

`spechub_verify` consumes **all** surviving global target candidates after
the existing filters, not only the proposal support. The sparse transport
allocates hub mass to `(a,x)` capacity before `(x,a)` capacity and samples
the full target residual. FP64 target normalization is a declared numerical
reordering of the ordinary FP32 picker. Requests require positive temperature,
explicit bounded `top_k`, and no grammar, XTC, or thinking budget. Independent
proposal, transport, and residual RNG streams remain distinct from ordinary
target PICK. `RADIANCE_SPECHUB_VERIFY=match` retains the same proposal law,
topology, and FP64 target distribution but independently samples the root;
`transport` is the experimental default.

Metadata binds the sequence, seed, output position, root, and retained
siblings. Only the root sample is replaced: existing acceptance, winner
compaction, EOG truncation, and root-plus-accepted state commit remain
authoritative. Bonus tokens do not enter commit before being forwarded.
One remaining output slot bypasses proposal; other truncated windows retain
the complete pair law while accepting only evaluated slots. These operations
use actual tensor rows within the resolved `M` band; metadata planes require
eight-byte alignment. Standalone probability checks do not replace live
engine lifecycle or full-model numerical validation.

`RADIANCE_MTP_COUPLED=1` (default off) is a separate experiment for the
**linear** builtin MTP chain only: no `RADIANCE_DARTREE`, DLoop, FullDrafter,
DFlash2 or SpecHub, `max_spec` 1..8, a reranked draft head (world 1 or a
vocabulary-sharded head), active drafting (`RADIANCE_NO_DRAFT` unset), and
`RADIANCE_NO_PROMPT_LOOKUP=1` so every draft came from MTP's own `set_draft`.

`mtp_couple_propose` runs once a temporal round (`round` is frozen per op),
after the final rerank and merge, on the unchanged shortlist `idx`/`val` whose
column 0 is the existing greedy proposal. A row whose `draw.mode` is 1 forms
`Q`, the FP64 temperature-one softmax of the FP32 scores (holes and `-inf`
carry no mass; entries whose FP64 weight or normalized probability underflows
are dropped, so the stored `q` is exactly what is sampled), draws `x` with the
53-bit uniform `(seed ^ (KEY_DRAFT + round), base)`, where `base` is the
request's pre-commit output size, saves ids and `q` in `state[round-1][row]`,
then writes `x` to column 0. Mode 0 (temperature 0 or `top_k` 1) keeps column 0
with status 0. A row whose `seq_ids[row]` is not `draw.card`, or whose shortlist
holds an out-of-range id, NaN/`+inf` score or duplicate id, keeps column 0 with
status -1. The engine reads the state back beside the step slots, refuses the
step for any stochastic row that is not status 1, whose draw is stale, or whose
`x` is not the card's own draft token, and binds `pos` to the committed output
position after commit.

`mtp_couple_verify` runs after `sample_pick` over every surviving target
candidate. For a bound state (seed, `pos`, round and draft token all match; the
sampler refuses a running stochastic draft row without one) it accepts `x` when
`u_accept*Q(x) < P(x)` (always when `P(x) >= Q(x)`), otherwise samples the
residual `(P-Q)+` with `r(x)=0`, using `RAD_SAMPLE_KEY_ACCEPT` and
`RAD_SAMPLE_KEY_RESID` at the verify position. `P` is the FP64 softmax of the
final FP32 candidate logits, with duplicate candidates aggregated. The bonus
row and every row without a bound state keep the ordinary pick. An empty or
invalid target also keeps it. If FP64 cancellation leaves zero residual while
some `Q` token has `P<Q`, the verifier instead samples `P` directly in FP64
using the independent 53-bit residual uniform. This numerical fallback is
chosen from `(P,Q)` alone, never from `x` or a drawn uniform; it does not
restore the FP32/24-bit pick. Acceptance, commit and KV handling are unchanged.

Numerics: the coupled rows realize the filtered target law computed in FP64
from the same FP32 candidate logits, to FP64 rounding and 53-bit uniforms;
the ordinary row realizes the FP32-CDF, 24-bit-uniform law. They are different
finite-precision approximations of one distribution, so flag-on output is not
seed-identical to flag-off output, and no bitwise equivalence with the
ordinary sampler is claimed.

`mtp_tree_accum` constructs the serial MTP tree on the device after each of
three possible top-eight proposal rounds. Round one reads the root from
`token_ids[draft_out_ids[i]]` and resets the live history row. Later rounds reuse
that root. Each call reserves the complete available spine before adding
alternates in round/candidate order, suppressing duplicate siblings.
The packed I32 row is `[count, tokens[capacity], parents[capacity]]`, including
the root with parent `-1`; capacity is 2 through 16. Invalid token IDs produce
an empty tree. Only the final packed rows cross to the host for validation and
scheduler publication; the engine neither reads each round's candidates nor
constructs the topology on the CPU. A declared serial tree requires a real
packed buffer, just like a block tree.

`mtp_merge_tree_accum` combines the TP2 top-eight merge with that same tree
construction. It preserves the merge's unsigned bit-key ordering, float payloads,
padding and duplicate-key removal. The gathered operand is F32 `[M,16,2]`;
the merged IDs and values remain available to existing consumers. Other tensor
parallel sizes retain the separate merge and accumulation operations.

`mtp_hybrid_lse` computes a stable shard-local maximum and scaled exponential
sum over the full valid coarse BF16 vocabulary, replacing local top-eight
entries with their reranked F32 logits. Padding beyond `n_vocab` is excluded.
Inputs are `coarse BF16[M,N]`, global `idx I32[M,8]` and `val F32[M,8]`;
output `stats F32[M,4]` contains `[maximum,sum,0,0]`. Stats rows require
16-byte alignment. TP metadata uses a separate collective, not candidate-pair
padding. This defines a hybrid proposal proxy, not an exact full-head softmax
and not a softmax restricted to the eight candidates.

`mtp_scored_merge_tree_accum` retains the TP2 bit-key top-eight merge and
constructs the topology on the device. It requires `pitch=8`, `world_size=2`,
`capacity=2..8` root-inclusive nodes, and `max_rounds=3` or `5`.
`gathered` is F32 `[M,16,2]`; `idx` and `val` are I32/F32 `[M,8]`.
`history` is I32 `[max_rounds,M,8]`. Policies 1 and 2 additionally require
F32 `log_history` of that shape and gathered F32 `stats[M,8]`
(two aligned four-value shard records). Round one reads
`token_ids[draft_out_ids[i]]` and resets history; later rounds reuse the root.

Policy 0 preserves spine-first selection and does not access `stats` or
`log_history`. Policy 1 ranks cumulative log proxy probabilities while
requiring each selected node's conditioning spine ancestor to be selected.
Equal scores prefer shallower depth, then lower token ID, then candidate order.
Only evaluated spine parents receive children; side-leaf descendants are not
invented, and duplicate siblings are suppressed. Policy 2 considers prefixes
of that selection, including the root-only prefix, and maximizes
`(1 + sum(exp(log_probability))) / cost[n]`, where `n` excludes the root.
Equal objectives retain the shorter prefix. All eight `cost0` through `cost7`
must be finite and positive for policy 2.

The packed I32 output remains `[count,tokens[capacity],parents[capacity]]`.
Count 1 is a valid root-only, no-draft proposal; count 0 signals invalid
input or normalization and makes the serial consumer fail the step.
Final count determines actual verification rows, not the configured capacity.
See `spec.md` §10, “Scored MTP proposal policy”, for opt-in flags and limits.

`mtp_pruned_expand`, `mtp_pruned_prepare`, and `mtp_pruned_pack` implement the
fixed W8/K8/D5/B7 policy independently of the spine accumulators. `M` is the
total request capacity, not the active tile size. Persistent `meta I32[M,85]`
holds status, base head position, anchor, forty-one tokens and forty-one parents;
`score F32[M,41]` includes root zero. Round one consumes one gathered TP2
sixteen-pair row per request, resets status/output, and obtains the anchor and
base position through history row indices. Later rounds consume eight parent
rows per active request and use `request_ids` to address persistent rows.

Each parent merges eight candidates and normalizes only that candidate set.
Global frontier ranking selects eight edges per level without a mandatory spine.
`prepare` emits actual parent hidden-row indices, sibling positions, unique
provisional slots, and chronological ancestors; the normal `gather_rows` operation
then gathers hidden carry before the real MTP transformer runs. Round five is
followed by deferred top-seven pruning with finite nonpositive beta and old-ID
ancestor-first ties. The final packed row is I32 `[count,tokens[8],parents[8]]`;
count eight is the complete result, count zero is fatal.

`mtp_overlay_attn` is model-owned, not a modified target/libr4d operation.
Queries/output are BF16 `[R,n_head*head_dim]`, `R=8*active_requests`.
Committed KV uses the public rank-four paged layout and one block-table row per
real request. The provisional rank-four KV has thirty-two token slots per
tiled request. `ancestors I32[R,4]` names chronological provisional slots;
`depth=1..4` selects the valid prefix. Each query reads its committed prefix
followed only by its own ancestor chain, including itself. Both caches are read-only.
Supported KV dtypes are BF16 and E4M3, block size four, head dimensions 1..256,
and any positive integral GQA ratio. Invalid device metadata invalidates the
request's shared status before safely zeroing output, so packing fails closed.

The device uses FP32 products, a zero-padded 256-lane QK reduction, chronological
online FP32 softmax/PV, and final BF16 round-to-nearest-even. Its independent
host reference uses scalar FP32 QK and two-pass softmax/PV. These are reordered
MTP numerics, not a bitwise target-parity claim. Ordinary same-op host/device
reference rows are registered; no fictitious exact-fusion `replaces` hook is declared.


`qsa_tree_commit` replays each accepted root-inclusive path in order, combining
`qsa_work`, `qsa_block_key` and `qsa_tail_store`. One device block owns each
sequence. Only completed pooled blocks produce block keys; pooled BF16,
RMS-normalized BF16 and rotary-output BF16 boundaries remain explicit.
Invalid metadata or duplicate live state/page destinations leave the batch
unchanged. The supported geometry is `n<=256`, `ratio<=ring<=32`,
`paths[seqs,16|65]`, and scalar-position NeoX rotation. The logical path width
sets the internal node capacity independently of the physical row stride.
Accepted lengths must fit that capacity; node IDs must fit both that capacity
and their sequence's token span. The original parameter schema and sequence-count
`M` dispatch remain unchanged. Both path widths use one fused launch per indexer,
replacing `3 * sum(accepted lengths)` replay launches; this is a launch-count
reduction, not a throughput claim.
`max_tokens` is a `RAD_PROLE_CAPACITY` bound, so sizing declarations may shrink
it. The operand's actual token extent still controls metadata validation;
the sequence extent controls the launch grid. Neither uses the capacity as
an actual row count.
An `mrope`/`imrope` configuration may use the fusion only when the batch has no
rotary-position planes; equal but shifted components are not scalar positions.
Unsupported geometries and plane-bearing batches retain the existing replay.
The no-plane unfused replay passes a one-column position view, preventing reads
of unwritten extra components in a three-column backing allocation.
The host oracle uses declared scratch, not launch-time allocation; GPU bit
parity is checked against the corrected unfused libr4d sequence, not inferred
from host transcendental math.

PLE operands use full-step absolute token rows `T`; narrowed sequence metadata
retains absolute `cu[S+1]` offsets. `ngram_ids_tree` reads `tokens I32[T]`,
canonical `tok_history I32[slots,1,Rtok]` (rank2 `[slots,Rtok]` also accepted),
`tok_idx I32[S]` (or `[S,pitch]`, column zero), `ctx_before I32[S]`,
`mult I64[ngram]`, `vocab_sizes/offsets I64[heads]`, and writes `ids I32[T,heads]`.
It follows ancestors then committed predecessors; EOS cuts every deeper shift.
Hash products wrap as signed I64 and remainder floors. Each vocabulary is
positive, each offset is nonnegative, and every resulting row ID fits I32.

`ple_conv_tree` reads `x/residual BF16[T,n]`, `weights BF16[n,width]`, and
`conv_history BF16[slots,n,Rconv]`, indexed by `conv_idx I32[S]` or `[S,pitch]`.
It writes `y BF16[T,n]` and exact input bits into `saved_raw BF16[T,n]`.
`mode=verify` (default) requires `saved_raw` and never mutates either history.
Ascending taps use lags `(width-1-j)*dilation`, separate F32 multiply/add rounding,
then `residual + acc/(1+exp(-acc))` and BF16 round-to-nearest-even.
`mode=prefill` is valid only for convolution, forbids a non-null parent,
permits arbitrary positive linear lengths, and makes `saved_raw` optional.
It materializes all inputs into convolution history only after its history reads.

All three schemas require `n>=1`, `ngram=2..8`, positive `heads` divisible by
`ngram-1`, nonnegative I32 `eos`, `width=2..8`, positive `dilation`,
`hist=(width-1)*dilation<=64`, `node_capacity=16`, and `history_ring=0|1`.
`q_len` bounds each actual sequence length; verify/commit require `1..16`.
`n_seq` only sizes shape-hook fixtures. Parent indices are local, root `-1`,
with each later parent in `[0,node)`. Outputs outside the selected rows survive.
Output/input aliases are rejected, including overlapping views.

`ple_tree_commit` reads `path_nodes I32[S,16]` and `path_len I32[S]`, with
nonempty root-inclusive ancestry paths, then commits only those tokens/raw rows.
Token history uses its first `ngram-1` cells, oldest to newest; absent logical
context is EOS. Canonical convolution (`history_ring=0`) uses its first `hist`
cells, oldest to newest, with zero for absent context. Position-ring convolution
(`history_ring=1`) stores accepted row `j` at `(ctx_before+j)%Rconv`.
`Rtok>=ngram-1` and `Rconv>=hist`; unused slots, canonical tails and padding
survive. Host commit validates every sequence/path and disjoint token/convolution
slots before any write. Prefill also requires disjoint convolution slots.
Optional group/layer/raw-buffer bindings connect all three ops to one geometry.

## Adding an op

Add the schema row to `libref/`, write the naive host implementation beside it, and it exists.
No core file changes. If you add an op **without** a ref implementation, say so in your commit: you
have just created something with no fallback and no oracle, and that is a decision, not an
oversight.
