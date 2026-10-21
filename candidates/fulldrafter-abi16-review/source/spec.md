# Radiance Inference Engine Specification

C++, with no Python anywhere in the engine or its tooling. A monolithic core with `.so`-based
plugins for the two things that actually change per model: the architecture, and the kernels.

Device code is a plugin's business and not the core's. `RAD_WITH_HIP` is tri-state — `ON | OFF |
AUTO` — and a host-only build with no device kernels is a supported configuration rather than a
degraded one: everything except the device kernels is exercised by it. Three kernel libraries are
in the tree and only one of them is HIP — `libr4d` for AMD RDNA4, `libavx` in hand-written x86
SIMD, `libref` in plain C++ — which is the reason that configuration is not a special case.

## Why

vLLM has the right core design — continuous batching, paged KV, prefix caching — and an
implementation that makes every one of those ideas expensive to reach. Everything is dynamic,
nothing states what it will run before it runs, the hooking points are numerous and
under-specified, and the Python tax is charged per step, per layer, per KV-cache group.

The measure of that is a patch set over vLLM: twenty-four monkey-patches to serve four models on
one card. Sorted by what they are fighting, they are the requirements list for this engine.

| what the patch class fights | count | radiance's answer |
|---|---|---|
| kernel substitution | 7 | kernel plugin hierarchy (§2.1) |
| fusion coverage | 1 | fused ops are ops you can select on (§2.3) |
| per-step host cost and metadata sharing | 4 | core-owned step batch, built once (§8) |
| KV group sizing for hybrid models | 1 | hybrid KV groups are first-class (§7.2) |
| speculative decode plumbing | 5 | a drafter is declared by the model's own plugin (§10) |
| chat template and tool parsing | 2 | lifted and rewritten from llama.cpp (§12) |
| platform detection | 2 | does not arise |

The goal is that none of those has an analogue here. Every one of them exists because vLLM's
extension points did not reach where the work was, so the only remaining move was to edit its
source. An engine you have to fork to change is not extensible; it just has a patch queue.

## Scope

**In, v1:** continuous batching, paged KV, prefix caching, tensor parallel, speculative decoding,
hybrid KV (full + sliding-window + linear-attention state in one model), weight placement across
VRAM / RAM / SSD with runtime re-placement, an OpenAI-compatible server, a quantiser.

**Multimodal: pictures and video, on the Qwen plugins (§11).** The vision tower is declared by the
language model's own architecture plugin, its embeddings travel in the step batch, and an image is
a prefix-cache hit by content. Audio is not served.

**Large MoE models.** The heat-driven expert placement path (§5.4) and the 4-bit expert formats
carry Qwen3.8-Flash-Next (`docs/MOE-W4.md`). DeepSeek-V4-Flash is the next architecture they are
meant for; it is not ported.

**Out:** training, LoRA, pipeline parallelism, CUDA, disaggregated prefill. None are designed
out; none are designed for either.

**Hardware:** no particular card. Everything above the device layer learns the machine at run time
-- architecture, compute units, wave width, LDS, VRAM -- and everything that cannot be written
once is a kernel library, which is a plugin selected per op against declared constraints (§2.1).
The device library shipped here, `libr4d`, covers AMD RDNA4 (`gfx12xx`); that is the hardware this
engine has been measured on, and it is the part of the tree that would be written again for
another card. Serving a card no installed library covers is a refusal that names the unserved ops
(§17), not a crash and not a silent slow path.

---

## 1. Shape of the engine

One process. One thread per tensor-parallel rank, each owning one device — no IPC, no
shared-memory handshake, no serialisation between ranks, which is most of what a Python engine
spends its multiprocessing budget on. HTTP threads feed a request queue; one scheduler thread
produces one `RadBatch` per step (§8) and releases the rank threads against it at a barrier.
Because the ranks share an address space, the batch is shared by pointer rather than broadcast.

Startup, in order:

1. Load kernel plugins from `$RADIANCE_HOME/kernels/`, in the configured hierarchy order.
2. Load the model's metadata only — architecture id, dimensions, quantisation descriptor, vocab.
3. Look up the architecture id in `$RADIANCE_HOME/architectures/`. No match, no model.
4. **Declare.** The architecture plugin enumerates every weight, buffer and op it will ever issue.
   The core resolves each op against the kernel hierarchy. This phase always completes; if
   anything failed to resolve, the full list is reported at the end rather than the first failure
   at the top.
5. **Plan.** The placement planner assigns every weight a tier and an execution site, against
   explicit pool budgets.
6. **Load.** Weights are mapped and staged according to the plan.
7. Serve.

Between 4 and 5 the whole model is known statically: every op, every kernel that will service it,
every weight it touches, and the order. That is the property the rest of this document spends.

---

## 2. Plugins

Three flavours, loaded from `$RADIANCE_HOME`: kernels (§2.1), architectures (§2.4) and
quantisers (§4.5). All are C ABI only. C++ across an `.so` boundary is
a standing invitation to a mismatch that manifests as corruption three frames later, and the
boundary here is exactly where a user is expected to substitute their own build.

Every plugin exports a version symbol and is refused on mismatch:

```c
uint32_t rad_plugin_abi_version(void);   /* must equal RAD_ABI_VERSION */
const RadPluginInfo* rad_plugin_info(void);
```

### 2.1 Kernel plugins

`$RADIANCE_HOME/kernels/*.so`. Each provides any number of kernels as **rows**: the plugin
exports `rad_kernel_count` / `rad_kernel_at` and the core reads the table, alongside the op
schemas (§2.3) it exports the same way.

**A plugin does not supply a selector.** The constraint selector is core-owned
(`core/plugin/select.cpp`), and that is the point rather than an economy: if every plugin brought
its own, "which kernel ran, and why" would have one answer per plugin, and the hierarchy below
could not be absolute because nothing above the plugins would be enforcing it. A plugin declares
what it can do; the core decides what runs.

```c
enum { RAD_C_EQ, RAD_C_LE, RAD_C_GE, RAD_C_DIV, RAD_C_IN };

struct RadConstraint {
    const char* key;    /* parameter name, e.g. "head_dim" */
    int         op;
    long long   ival;   /* EQ / LE / GE / DIV */
    const char* sval;   /* IN: space-separated set */
};

enum { RAD_DOMAIN_DEVICE, RAD_DOMAIN_HOST };

/* One AXIS of this kernel's instantiation space -- see §15. Null/0 means nothing to tune. */
struct RadTunable {
    const char*    key;       /* the kernel's own name for the axis: "sk", "bm", "npw" */
    const int64_t* values;    /* legal values, ascending; the core tries exactly these */
    int            n_values;
    int64_t        deflt;     /* what runs where this machine has no measurement */
    const char*    doc;
};

struct RadKernelInfo {
    const char* name;       /* entry point */
    const char* op;         /* the op it implements -- the key callers select on */
    const char* family;
    const char* computes;   /* prose, for a human */
    const char* shape;      /* the geometry it is compiled for, prose */
    const char* dtypes;
    int         domain;     /* RAD_DOMAIN_DEVICE or RAD_DOMAIN_HOST */
    int         priority;   /* higher wins within this plugin; ties break on declaration order */

    const RadConstraint* constraints;  int n_constraints;
    const RadTunable*    tunables;     int n_tunables;
    RadTuneValidFn       tune_valid;   /* cross-axis legality at a geometry, or null */

    RadInitFn      init;       /* per resolved instance: P2P handshake, persistent workspace */
    RadFiniFn      fini;
    RadLaunchFn    launch;     /* int (*)(const RadArgs*, RadStream) */
    RadScratchFn   scratch;    /* int64_t (*)(const RadArgs*), or null */
    RadLayoutFn    layout;     /* what this kernel stores for a weight of a given encoding -- §4.3 */
    RadRelayoutFn  relayout;   /* host code producing it from the canonical planes, at load */
    RadShapeFn     opd_shape;  /* how to size and fill each operand, for a tool calling it cold */
    RadUnrelayoutFn unrelayout;/* the inverse, so a stored weight can be read back */
    RadFuseFn      replaces;   /* the chain of conventional ops this kernel does in one launch */
    RadDescribeFn  describe;   /* the same dispatch, described rather than performed */
};
```

`init` and `fini` exist because some kernels have per-instance state that must be established
once and not per launch — libr4d's all-reduce needs its peer signal buffers and IPC handles set up
across ranks before the first call, and a kernel that JITs or loads a config table wants somewhere
to do it that is not the hot path.

**The six hooks below `scratch` are what make a kernel legible to everything that is not the
step.** `layout` and `relayout` are what the loader calls to turn a weight's canonical planes into
the bytes this kernel wants (§4.3), and `unrelayout` turns them back, so a stored weight can be
compared against a dense oracle instead of being unfalsifiable. None of them quantises: a kernel
states which encodings it reads by refusing the rest in `layout`, and producing an encoding is a
quantiser's job (§4.5). `opd_shape` says how big each operand is and what may be in
it, so a tool can call the kernel cold; a kernel that supplies none is skipped **by name** rather
than guessed at, and a guessed extent tests the kernel's bounds handling and calls the result
correctness. `replaces` is a fusion declaring the chain of conventional ops it computes in one
launch, which is what turns "byte-identical to the two ops it replaces" from prose into something
checkable (§2.3, §17). `describe` reports a dispatch instead of performing it (§3.2). All six are
optional; `launch` is not.

**There is no variant index.** A kernel declares tunable **axes** and the core picks a point in
that space, appending the chosen values to `RadArgs::p` as ordinary parameters — so a kernel reads
its tuning with the same call it reads its geometry with. §15 is the whole of the mechanism.

**Constraints are a necessary condition, not a sufficient one.** They settle whether a kernel
exists for a geometry. Strides, contiguity, alignment and buffer sizes are properties of a call
rather than of a model; they stay in the entry point, which still rejects what it cannot run.
libr4d's registry comments make this point at length.

**Selection.** Plugins are queried in hierarchy order, and the hierarchy is absolute: the first
plugin with any matching kernel supplies it, and a lower-priority plugin never outbids a higher
one no matter how specialised its kernel. That is what makes a user override an override.

Within one plugin, the highest `priority` among matching rows wins, ties broken by declaration
order. Specificity is **declared, not derived** — counting matched constraints gets it backwards,
since `{head_dim<=1024, gqa>=1, block%16==0, causal in {0,1}}` matches four constraints and
`{head_dim==256, gqa==6}` matches two, and the second is by far the more specialised. libr4d's
all-reduce rows are the example: the two-shot rows must outrank the one-shot ones or nothing could
ever reach them, and an integer states that intent instead of leaving it to the order of the rows.

**Domains.** A CPU kernel is an ordinary kernel with `domain = RAD_DOMAIN_HOST`, resolved through
the same hierarchy and the same constraints. Declare resolves both domains where both exist, so
the core can change execution site at runtime without re-resolving. Absence is reported
asymmetrically: no device kernel is fatal; no host kernel is a warning naming the placement
options it just removed.

**Arguments** are positional, per the op's schema:

```c
struct RadTensor {
    void*    data;
    uint32_t dtype;
    uint32_t rank;
    int64_t  shape [RAD_MAX_RANK];
    int64_t  stride[RAD_MAX_RANK];   /* elements, not bytes */
};

struct RadArgs {
    const RadTensor* t;  int n_t;      /* operands, in the op schema's order */
    const RadParam*  p;  int n_p;      /* the resolved geometry: every param, ranges collapsed,
                                        * with the tuned axes appended (§15) */
    void*            scratch;
    int64_t          scratch_bytes;
    void*            instance;         /* whatever init() returned for this resolved instance */
    int              rank;             /* tensor-parallel rank and world size, for collectives */
    int              world_size;
};
```

`instance` is why `init` is worth having at all: the state it established is handed back on every
launch without a lookup. `rank` and `world_size` are here rather than as parameters because a
collective needs them and nothing else does, and a kernel that asked its geometry for them would
be reading a key the architecture plugin had to remember to write.

### 2.2 Shape domains and the bucket table

A parameter may be declared as a **range** rather than a value, because some of them are
properties of the batch and not of the model. `M` is the obvious one: a GEMM sees one row at
decode, sixty-four at a speculative verify, and thousands in a prefill chunk, and libr4d has a
different kernel for each band. Resolving once at `M = max_tok` would pick the prefill kernel and
then run it at `M = 1`, which is exactly the loss a skinny-GEMM patch to vLLM exists to recover.

```c
h = decl_op(b, "gemm_nt_q",
            RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("N", 8192), RAD_INT("K", 5120),
                       RAD_INT("group", 128), RAD_STR("dtype", "w4a8")));
```

The core resolves the range into a **bucket table**: one resolution per band, built at declare and
indexed by the actual value at issue.

**The bucket boundaries are the constraint values themselves.** Take the union of the `LE`, `GE`
and `EQ` values that candidate kernels place on the ranged parameter, and those are the bands.
Three kernels constraining `M<=16`, `M<=64` and nothing produce `(0,16]`, `(16,64]`, `(64,max]`
and no policy was chosen by anyone. A kernel whose fast band sits at an odd boundary gets that
boundary, because it said so.

`--debug-graph` prints the table:

```
gemm_nt_q  N=8192 K=5120 group=128 dtype=w4a8
    M <=   16  ->  gemm_w4a8_nt_m64        (libr4d, prio 20)
    M <=   64  ->  gemm_w4a8_nt_m64        (libr4d, prio 20)
    M <= 8192  ->  gemm_w4a8_prefill       (libr4d, prio 10)
```

The missing-kernel report is per band. A range with a hole in it — a model whose prefill shape
nothing serves — is reported as that band, not as the whole op.

### 2.3 The op vocabulary

Op names are free strings. Kernel plugins declare the ops they implement **and the schema for
each** — a `RadOpSchema` table exported through `rad_kernel_schema_count` / `rad_kernel_schema_at`
— and the core cross-checks an architecture plugin's declared parameters against the union of
schemas.

```c
static const RadParamSpec pGemmNtQ[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },   { "N", RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },   { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGemmNtQ[] = {
    { "a", RAD_OPD_IN, 0 },  { "a_scale", RAD_OPD_IN, /*optional*/ 1 },
    { "b", RAD_OPD_WEIGHT, 0 },  { "b_scale", RAD_OPD_WEIGHT, 0 },  { "y", RAD_OPD_OUT, 0 },
};
SCHEMA("gemm_nt_q", pGemmNtQ, oGemmNtQ, "C[M,N] = A[M,K] @ dequant(Bq)[N,K]^T ...")
```

**A schema names operands as well as parameters, and the roles are the part the core needs.** It
has to know which operands are weights, so issue can resolve a handle and wait on the mover's
event, and which are written, so the buffer planner can compute liveness. It needs nothing else
about them — which is exactly what keeps a third party able to add an op.

Without the schema, a typo and an unimplemented op are the same diagnostic. With it, a misspelled
op fails declare with the near-miss named. New ops need no core change, which is the requirement:
the core must never be the thing you edit to add a kernel.

**There are three requirednesses, not two.** A parameter is supplied by the caller
(`RAD_REQUIRED`), may be (`RAD_OPTIONAL`), or is supplied by the **kernel** (`RAD_DERIVED`). The
third exists because of a circularity §7.2 makes concrete: the paged block size comes from the
resolved attention kernel and not from a core constant — but a constraint whose key the geometry
does not carry does not hold, so a kernel constrained to `block_size == 16` could never be
selected by the query that was going to read the answer off it. So constraints on a derived
parameter are **skipped** during matching, and once a kernel wins, its `RAD_C_EQ` value for that
key is written into the resolved geometry. A caller that supplies one anyway pins it and it
matches normally, which is how an operator forces a block size and gets a refusal by name rather
than a silent substitution.

**Schemas are fixed by first declaration.** The first plugin in hierarchy order to declare an op
fixes its schema; a later plugin declaring the same op name with a different schema is refused at
load with both plugins named. Since arguments are positional, two disagreeing schemas would make
the same call mean different things depending on which plugin won selection, and that is not a
diagnosable failure — it is silent numerical garbage.

**Fusion is a selection, not a compiler pass.** A plugin asks for the fused op first and emits the
unfused sequence when it does not resolve:

```c
rad_op h = decl_op(b, "rmsnorm_had_quant_i8", {{"n", d_model}, {"dtype", "bf16"}}, ...);
if (!h) {
    h_norm = decl_op(b, "rmsnorm",           ...);
    h_hq   = decl_op(b, "had_quant_act_i8",  ...);
}
```

`rad_op_resolved` is the shipped spelling of that test: a handle that did not resolve is
`RAD_NULL_HANDLE`, and that is **not an error yet** — it becomes one only if the plugin issues it.
Explicit, no pattern matcher, and the plugin author decides. A fusion rewriter is a compiler, and
a compiler is the thing whose output nobody can predict — which is the complaint this engine
exists to answer.

A kernel may also **declare** the chain it replaces (`RadFuseFn`, §2.1), and that is a
declaration, not a rewriter: it makes the equivalence machine-readable so a fused kernel can be
checked against the same plugin's own unfused rows — where the answer should be byte-identical,
which is a stronger test than a tolerance against `libref` — and it is what `--debug-graph`'s
fusion advisory reads. Nothing consumes it to transform the graph, and the declare phase keeps
nothing a rewriter could work on in any case: the buffer planner collapses each buffer's whole
use-set to a `first_def`/`last_use` pair, so "op A's output is op B's only input" is not
recoverable afterwards.

### 2.4 Architecture plugins

`$RADIANCE_HOME/architectures/*.so`. One plugin implements one architecture id for one
quantisation scheme, and **the selection key is the pair**: `qwen35_bf16.so` and `qwen35_fp8.so`
both answer `"qwen35"` from `rad_arch_id()` and are told apart by `rad_arch_quant()`, against the
descriptor the container already carries. No two plugins may claim the same *pair* — the second
is refused at load with both named — and one id at several quantisations is the normal case rather
than a collision, so the uniqueness rule is on the pair and not on the id. The match is exact and
there is no fallback: an unquantised container and an unquantised plugin both say `""`. Falling
back from a quantised container to a plugin that declares dense weights would fail later, at the
name map, with a far worse message than *"no plugin claims qwen35 at fp8_e4m3; qwen35 is served
at: (unquantised)"*.

The combination is deliberate. Extreme fusion means the graph genuinely differs by quant: a w4a8
model wants `rmsnorm_had_quant_i8` where a bf16 model wants a plain `rmsnorm`, and pretending
otherwise means a plugin full of branches on a quant descriptor — which is how you get back to
"unclear what actually runs". A declare function with no branches in it is a readable list of
what this model does, in order, and that readability is the point of the project.

Dimensions come from model metadata, so one plugin covers every size in a family: `qwen35_fp8.so`
serves Qwen3.5 / 3.6 / 3.8 dense from 0.8B to 27B. The file-count mitigation for the quant axis is
a header-only helper library of composable blocks (attention layer, MoE layer, GDN layer), so a
plugin is composition rather than copy-paste.

The mitigation along the other axis is that two `.so`s can be built from one source.
`qwen35moe_fp8` is `qwen35_fp8` with the feed-forward routed: three macros and an include, with
the half-dozen places a routed FFN differs behind one `#if`. They stay two plugins because the id
is the selection key — a container converted from the MoE checkpoint reports `qwen35moe` and the
dense one reports `qwen35`, one `.so` cannot claim both, and an operator looking at what is
installed should see two plugins for two architectures.

The plugin also owns **the map from checkpoint tensor names to declared weight names**, including
the fusions — q/k/v into a single qkv, gate and up into gate_up, one expert out of a stacked
tensor. It is the only component that knows both sides, and both readers of a checkpoint use it:
`rad-convert` (§4.4) and the engine loading one directly (§4.3).

**It asks what a weight holds instead of being told by a key.** `rad_weight_encoding(b, name, &enc)`
answers with the logical weight's encoding (§4.1) — from the container's entry, from the checkpoint
tensors the name map resolves it to, or, inside `rad-convert`, from the recipe — and the plugin
declares the ops that read it: fp8 codes and their scale plane to `gemm_nt_q`, int4 rotated
experts behind `had_quant_act_fp8`, a bf16 weight to the bf16 GEMM, and a refusal by name for an
encoding it has no path for. One plugin per (architecture, quantisation) still holds; what the
encoding query removes is every container key that existed only to tell a plugin how some of its
weights had been packed.

Each plugin exports these, two of them optional:

```c
const char* rad_arch_id(void);
const char* rad_arch_quant(void);     /* the other half of the selection key; "" if unquantised */
int  rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out);          /* optional */
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta);       /* optional, §12 */
int  rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
void rad_arch_step(RadCtx* c, const RadBatch* batch);
```

`rad_arch_probe` is the one thing a plugin can say **before** it declares anything, and it exists
for the one fact that is an *input* to declaration and a property of the model: the draft depth.
`RadArchProbe{draft_depth, draft_depth_fixed}` is what resolves `--num-speculative-tokens auto`
(§10), and it has to happen first because `RadBuildCtx::max_spec` sizes buffers, bands kernels and
sets the scheduler's window. The alternative is a core that reads a drafter's own configuration
keys itself, by name, with an architecture's prefix in the string. A plugin that does not export
it answers zero to everything: no draft depth, so `auto` resolves to 0.

`RadBuildCtx` carries `rank` and `world_size`, so declare runs once per rank and the plugin
declares dimensions already divided (§9). It also carries the bounds the deployment was
configured for — `max_tok`, `max_seqs`, `max_ctx`, `max_spec` — so a plugin sizes its buffers
against the worst step rather than guessing; the tensor-parallel wire choice `tp_wire_lossy` and
its `tp_wire_min_bytes` cutover (§9); `kv_dtype`; and `is_draft` / `scope`, which say whether this
builder scope is the model, a drafter, or an encoder.

---

## 3. Two-phase construction

### 3.1 Declare

The plugin states everything it will ever do. Weights first:

```c
RadWeightDecl wd = {};
wd.dtype  = RAD_I4;                     /* the LOGICAL dtype; the kernel owns the stored layout */
wd.rank   = 2;
wd.shape[0] = 8192; wd.shape[1] = 5120;
wd.access = RAD_ACCESS_CONDITIONAL;     /* see §5 */
wd.group  = rad_group_expert(12, e);
wd.shard  = RAD_SHARD_ROW;
rad_weight w = rad_decl_weight(b, "blk.12.ffn_gate.weight", &wd);
```

Activations are declared too, or the buffer plan below has nothing to analyse:

```c
RadBufDecl bd = {};
bd.dtype = RAD_I8; bd.rank = 2;
bd.shape[0] = max_tok; bd.shape[1] = d_model;
bd.kind = RAD_BUF_TRANSIENT;
rad_buf tmp = rad_decl_buffer(b, "ffn.in", &bd);
```

Then ops, which name their weight operands:

```c
rad_op o = RAD_OP(b, "gemm_nt_q",
                  RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("N", 8192),
                             RAD_INT("K", 5120), RAD_INT("group", 128),
                             RAD_STR("dtype", "w4a8")),
                  RAD_WEIGHTS(w));
```

...and their **buffer** operands, which are a separate call and not optional if the paragraph
above is to mean anything. `rad_decl_op` names weights, because weights are what placement moves;
liveness is which ops read and write which buffers, and an undeclared transient has no declared
use and must be assumed live for the whole program.

```c
rad_op_reads (b, o, RAD_BUFS(tmp));
rad_op_writes(b, o, RAD_BUFS(out));
```

Every name above is the shipped spelling — `abi/rad_builder.h` is the authority, and
`arch/common/rad_arch.h` wraps these five calls in the `decl_w`/`decl_b`/`rw` helpers that the
in-tree plugins actually read like. An architecture plugin is C++ and uses designated
initialisers where it likes; the field-at-a-time form is written out here only because
`.shape = { 8192, 5120 }` hides that `rank` is a separate field and is not derived from it.

Declare produces five things:

- **the resolved op list** — every op with the kernel(s) that will service it, per band, per domain;
- **the missing list** — every op and band that resolved to nothing, reported whole;
- **the weight manifest** — every weight with bytes, alignment, access class and grouping keys;
- **the buffer plan** — activation liveness over the op list, so scratch and intermediates come
  from one arena sized once instead of an allocator called per step;
- **the tuning requests** — every (kernel, shape) pair `rad-tune` would need to benchmark (§2.1, §15).

The arena is sized for `max_tok`, and liveness is computed over the declared list, which is a
superset of what any given step issues. Both are conservative on purpose: a plan that is correct
for the worst step is correct for every step, and the alternative is an allocator on the hot path.

### 3.2 Run

```c
void rad_arch_step(RadCtx* c, const RadBatch* batch) {
    RAD_ISSUE(c, h_norm, RAD_B(x), RAD_W(w_norm), RAD_B(tmp));
    RAD_ISSUE(c, h_gemm, RAD_B(tmp), RAD_W(w), RAD_B(y));   /* core indexes the bucket table */
    RAD_ISSUE(c, h_moe,  RAD_B(y), RAD_WTAB(w_expert, n_expert), RAD_B(out));
}
```

The call underneath is `int rad_issue(RadCtx*, rad_op, const RadOperand*, int n_opd, int64_t n)`.
`RAD_OPD` packs the operand array; `RAD_ISSUE` passes `RAD_N_BATCH` for the op's ranged parameter
and `RAD_ISSUE_N` names a value instead. Operands are **positional, per the op's schema**, so
there is no in/out tagging at the call site — the schema already carries the roles, and a second
statement of them is a second place for them to disagree. `RAD_B` is a buffer, `RAD_W` a weight,
`RAD_WTAB` a whole run of them.

Note what the last line is *not*: a host-side loop over routed experts. Routing is resolved on
device against the current residency table (§5.5), so the issue names the entire expert table and
the kernel indexes it. A `for (int e : route(...))` here would be a device round-trip in the
middle of the step, which is the thing §5.5 exists to avoid.

Ordinary C++, with one rule: **it may only issue handles obtained during declare.** Nothing else
is constrained. There is no graph capture: every dispatch is submitted individually, and the
accepted cost is the per-dispatch launch floor times the dispatch count — on the order of 1850
dispatches in a decode step at a couple of microseconds apiece — against a Python engine paying
that same floor with an interpreter on top of it.

**The ABI already carries the hook capture would need, and it is not the same hook as `launch`.**
`RadDescribeFn describe` reports the dispatch a call *would* perform instead of performing it: the
same dispatch, described rather than performed, so a repeatedly issued sequence can be replayed as
a graph. `launch` cannot support that, because it marshals its own arguments — so nothing above it
knows which argument slot holds which operand, and a graph node update needs exactly that slot.
`describe` is optional and `launch` remains required; a kernel that implements only `launch` never
becomes a graph node and is submitted the ordinary way. The one trap is that the two must dispatch
the same kernel with the same arguments and nothing can check that for you, so the shape
arithmetic is written once and emitted into a sink that is either a stream or a descriptor.

**Handles, not pointers.** `rad_issue` takes weight handles. The core resolves the handle to a
pointer at issue time, waits on the mover's event if the weight is in flight, and dispatches to
the device or host implementation according to the current execution site. If an op received a
`void*`, the weight's location would be fixed at declare and placement (§5) could not be a core
feature at all. This is the single ABI decision the whole placement design rests on.

---

## 4. Model format

### 4.1 Encodings and layouts

Two things decide the bytes a kernel reads, and they change for different reasons, so they are
kept apart:

- An **encoding** is what the numbers are: the code type, how scales and zero points are grouped
  over the weight, and whether the weight was rotated before it was quantised. It is a property of
  the *model*. Choosing one is expensive and lossy — GPTQ on Qwen3.8-Flash-Next's experts needs
  Hessians from a calibration run — and changing one means quantising again.
- A **layout** is how a kernel wants those numbers arranged: WMMA fragment order, scales carried
  in each row's tail, nibbles permuted to suit an unpack. It is a property of the *kernel*. It is a
  lossless rearrangement and it is cheap.

The container stores encodings and only encodings, in one canonical arrangement, and the kernel
library rearranges them when the model loads (§4.3). A kernel library may change every layout it
has without touching a container; the only thing that ever requires converting again is a change
in the numbers.

The split is not a matter of taste. In a Qwen3.8-Flash-Next container with int4 experts, 94% of
the bytes (the experts and the fp8 n-gram table) are canonical as they stand and only ~3 GiB are
kernel-permuted, while 82% of Qwen3.8-27B-FP8's bytes are fp8 that the checkpoint holds row-major
and its kernels read fragment-ordered. What makes a container kernel-specific is a small fraction
of its bytes; what makes it expensive is the quantisation, and that is the part worth storing.

**An encoding is data, not a name in an enum** (`abi/rad_encoding.h`), so a quantiser can invent
one without the core learning about it:

```c
typedef struct RadEncPlane {
    char     role[24];        /* "codes", "scale", "scale.1", "zero", "min", "grid", "t.perm" */
    uint32_t dtype;           /* a core dtype (rad_types.h), never a plugin-private one */
    uint32_t kind;            /* TILED: covers the weight by blocks | TABLE: its own extents */
    int64_t  block[2];        /* TILED: logical rows x cols one element covers; 0 = the whole */
    int64_t  extent[2];       /* TABLE: rows x cols */
} RadEncPlane;

typedef struct RadEncoding {
    char        scheme[24];     /* "plain" | "affine" | a quantiser's own (§4.5) */
    char        transform[24];  /* "" | "fwhtN" | "perm" | "givens" */
    int32_t     n_planes;
    RadEncPlane plane[8];
} RadEncoding;
```

Every plane is a dense row-major array. A TILED plane covers the weight seen as `[rows, cols]` —
every leading axis a row, the last the column — so a block of `[1 x 32]` is a per-group scale, `[0 x
0]` one per tensor and `[0 x 1]` one per column. A TABLE plane does not cover the weight: a codebook,
a lattice grid, a transform's parameters, carried whole by every slice. Elements narrower than a
byte or not a power of two wide are a bit stream along the row, lowest bits first, every row on a
byte — the order a reference implementation and a hex dump assume. The dtypes are every float a
model ships in (f32, bf16, f16, fp8 e4m3 and e5m2, fp6 e2m3 and e3m2, fp4 e2m1, e8m0), signed and
unsigned integers of every width from one bit to eight plus 16 and 32, and 9- to 12-bit unsigned
for lattice indices.

`plain` is a single `codes` plane whose value is the code. **`affine` is the scheme the core
decodes, and it is wide enough for every format in common use**:

```
q = code                      one code a value, or one a V-vector of a `grid` (TABLE [entries x V])
q = table[code]               a scalar codebook (TABLE [1 x entries])
q = -q where `signs` is set   one bit a value
q = q - zero                  an offset in the code's domain
v = q * scale * scale.1 ...   every scale level, multiplied
v = v - min * min.1 ...       a minimum in the value's domain
w = v . T^-1                  the transform, per row (a kernel transforms its activation instead)
```

`fwhtN` is the block-diagonal unnormalised Hadamard with its 1/N in the stored scale (docs/MOE-W4.md),
`perm` a column permutation, and `givens` stages of disjoint pair rotations under a channel scale,
their tables beside the weight. Each encoding has a canonical spelling, which is what people, logs
and `rad-info` read; kernels read the struct:

| encoding | what carries it |
|---|---|
| `bf16` | a dense checkpoint's weight |
| `fp8_e4m3*bf16[128x128]` | a block-FP8 checkpoint: `X.weight` with `X.weight_scale_inv` |
| `i4*bf16[1x128]/fwht128` | Qwen3.8-Flash-Next's int4 experts (`data/recipes/qwen4exp-w4.recipe`) |
| codes `u4` into `table f32{1x16}`, `scale fp8_e4m3[1x64]` . `f32[*x*]`, `/fwht128` | Qwen3.8-Flash-Next's served experts: the `w4nl` codebook (`docs/MOE-W4.md`) |
| `fp8_e4m3*bf16[*x*]` | the n-gram table: one scale for 51 GB |
| `u2*f16[1x128]-u8[1x128]` | a 2-bit draft head, asymmetric |
| `fp4_e2m1*e8m0[1x32]` | MXFP4 |
| `fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]` | NVFP4: two scale levels |
| codes `u4`, `scale u6[1x32] . f16[1x256]`, `min u6[1x32] . f16[1x256]` | llama.cpp's Q4_K |
| codes `u4`, `table i8{1x16}`, two scale levels | IQ4_XS |
| codes `u8[1x8]` into `grid u8{256x8}`, `signs`, two scale levels | IQ2_XXS |
| codes `u4` scaled, `scale f16[0x1]` beside the group scale | AWQ's column scale, unfolded |
| `perm` with `t.perm i32{1xK}` | GPTQ's act-order |
| `givens` with `t.pairs`, `t.angle`, `t.scale` | ParoQuant's scaled pairwise rotation |

So a GGUF file's K- and I-quants are as representable as anything else: reading one is a lossless
rearrangement into planes, and the only thing such a file lacks is a kernel that reads it.

**The core decodes every `plain` and `affine` encoding itself**, to f32: for the oracle, for
`rad-convert`'s error report, and as the input to quantising something again. A quantiser that
invents a scheme the core cannot decode supplies the decoder with it (§4.5), so nothing in the
core has to learn a codebook to check a kernel that reads one.

### 4.2 `.rad`

The engine loads `.rad` containers, safetensors checkpoints and GGUF files (§4.3). `.rad` is the
one built for the load path:

- **canonical, not kernel-shaped.** An entry is a logical weight holding the planes of its
  encoding, row-major. No kernel library is named anywhere in the file.
- **grouped by movement unit.** An expert's entries are adjacent, because the unit of transfer is
  an expert, not a tensor. Addressing within a layer is `base + id * stride`, O(1) arithmetic and
  no table walk on the critical path, and the reader re-derives the stride at open and refuses a
  file where it does not hold.
- **aligned.** Pool 2 MiB (huge-page mappable), movement unit 4 KiB, which is `O_DIRECT`'s
  alignment and so what lets the file tier read a unit with no bounce buffer; planes 256 B.
- **self-describing.** Each entry carries its encoding, its logical shape and the quantiser and
  options that made it, so `rad-info` can say exactly what a file holds and `--reuse` can tell
  which weights another conversion would have written identically.

Little-endian only. The container also carries the checkpoint's configuration as metadata, the
vocab (§12), the recipe it was converted with, and the expert-popularity profile the placement
planner warm-starts from. There is one version: a file in any other is refused with
`RAD_E_FORMAT` and the instruction to convert again.

### 4.3 Loading

The path is the same whatever the source:

1. **Declare** (§3.1) names each weight and says which planes of which logical weight it takes —
   `RadWeightDecl::source` and `::planes`, so a linear's codes and its scale plane can be two
   operands of one op, or one operand a kernel interleaves.
2. **Slicing** for tensor parallelism happens on the canonical form, where it is always a
   rectangle: a ROW shard is rows of every plane and a COL shard is columns, each cut on the
   plane's own block boundary and refused where a block would be split.
3. **Relayout.** The resolved kernel's `layout` hook says what it stores for the weight, given its
   encoding; `relayout` produces that from the rank's slices of the planes. A hook that answers
   `RAD_E_UNSUPPORTED` keeps the plane as it is, and those bytes go from the file to their
   destination with nothing in between.
4. **Agreement.** Every kernel that reads a weight — every band, both domains — must ask for the
   same layout, or declare refuses and names both.

**Selection reads the encoding.** A kernel whose `layout` hook answers `RAD_E_DTYPE` for a weight
an op takes — it does not read that encoding — is not a candidate for the op, and neither is a
device kernel with no hook for a weight of several planes. So one declaration serves a bf16 head
with the bf16 GEMM and an fp8 head with the fp8 one, and a checkpoint and the container quantised
from it select by what each holds.

Relayout is host code, run by a pool behind the load sweep, and it has to keep pace with the disk:
Qwen3.8-27B-FP8 rearranges 23 GiB of its 29 at every load. The sweep only reads — `O_DIRECT`, into
a ring of pinned windows — and the pool makes every host copy out of them as well as the relayouts,
out of memory it reuses from one weight to the next.

**Two tiers read the file while serving and so cannot relayout.** The file tier stages a unit
from the container with `O_DIRECT`, and the mapped tier reads the container's own mapping. A
weight whose layout is not the identity is placed on neither; the planner keeps it in the host
pools and says so by name when that is what fills them. The weight these tiers exist for — the
n-gram table, which is mapped — is identity.

**A checkpoint loads directly when its encoding is trivial.** A safetensors directory or a GGUF
file is a source like a container: the architecture plugin's name map (§2.4) resolves each declared
weight to checkpoint tensors, concatenations and expert slices included, the reader states their
encodings, and the same path runs. Trivial means the checkpoint's numbers are served as they are:

- `bf16`, `f16` and `f32`, and the exact widenings between them (a bf16 norm the plugin reads as
  f32);
- block-scaled FP8 (`quantization_config.quant_method = fp8`), where `X.weight_scale_inv` is the
  scale plane of `X.weight` — the encoding `fp8_e4m3*bf16[128x128]`, byte for byte;
- GGUF `F32`, `F16` and `BF16`.

**Nothing is quantised at load.** A model whose served form needs quantisation work — Flash-Next's
int4 experts, an fp8 `lm_head` made from a checkpoint's bf16 one — is converted ahead of time
(§4.4), and a checkpoint loaded directly is served at its own precision. A direct load also has no
file tier and no mapped tier (a checkpoint is neither grouped by movement unit nor aligned for
`O_DIRECT`) and no expert profile, so the planner keeps everything in VRAM and the host pools and
says so.

### 4.4 `rad-convert`

Reads safetensors or GGUF and writes `.rad`. It runs the architecture plugin's declare phase for
the declared weights and the name map, and it quantises according to a **recipe**: ordered rules
from a logical weight name to a quantiser and its options. The first rule whose pattern matches a
weight decides it; a weight no rule matches keeps the checkpoint's encoding.

```
# data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe -- Qwen3.8-Flash-Next as it is served (excerpt)
# PATTERN                        QUANTISER  OPTIONS
blk.34.ffn_*_exps.407.weight     cast       dtype=bf16
blk.*.ffn_*_exps.*.weight        gptq       table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* scale2_value=0.0001220703125 transform=fwht128 rule=search cd=3 calib=$CALIB
*_hc_down.weight                 rtn        codes=fp8_e4m3 group=128 scale=f32
blk.*.attn_qg.weight             rtn        codes=i8 group=128 scale=bf16 clamp=sym rule=search
output.weight                    rtn        codes=i8 group=128 scale=bf16 clamp=sym rule=search
blk.*.ple_ngram.weight           rtn        codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=1.99317932128906e-4
mtp.draft_head.weight            rtn        codes=u2 zero=u8 group=128 scale=f16
...
```

A rule names an **algorithm** (`rtn`, `gptq`, `cast`) and its options name the **grid** (`codes` or
`table`, `group` or `block`, `scale`, a second level `scale2`, `zero`, `rule`) and the **transform**
(`transform=fwht128`) (§4.5). The first match deciding is what lets a protected expert's `cast`
rule sit above the rule every other expert takes. A quantiser that declines a weight -- `rtn`
declines a rank-1 tensor -- leaves it to the checkpoint's encoding, so a broad pattern does not
have to step around every norm.

`--recipe FILE` names a file and `--quant 'PATTERN=QUANTISER:k=v,...'` adds rules ahead of it.
`$NAME` in an option is the environment's, so one recipe serves every machine.

**The architecture plugin learns a weight's encoding the same way at convert and at load** —
`rad_weight_encoding` (§2.4) — and at convert the answer is the recipe's. So the declare that runs
in the converter is the declare the engine will run, every op still has to resolve, and every
resolved kernel's `layout` hook has to accept the planned encoding: a recipe nothing can serve is
refused before a byte is written. The kernel plugins check the recipe and play no other part; the
converter writes planes, and which library rearranges them later is not its concern.

It **reports the error of every weight it quantised** against the source — relative Frobenius
norm, and imatrix-weighted when `--imatrix` is given, worst first — through the core's decoder, so
a quantiser is measured the moment it exists. A format change that cannot be ranked is a format
change nobody can argue about.

`--reuse C` copies from an earlier container every weight this run would write identically — same
encoding, same quantiser and options, same source shape — and `--in-place` extends `C` instead of
copying it, so adding a vision tower to an existing container is about a gigabyte of I/O however
large the container is (§11).

### 4.5 Quantiser plugins

`$RADIANCE_HOME/quantizers/*.so`, the third plugin kind. A quantiser turns a weight in f32 into
the planes of an encoding:

```c
typedef struct RadQuantizerInfo {
    const char* name;                  /* what a recipe names: "rtn", "gptq" */
    const char* doc;
    const RadQuantOption* options;  int n_options;
    /* The encoding it writes for this weight under these options; RAD_E_UNSUPPORTED declines. */
    int     (*encoding)(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out);
    /* Rows one call may take, so a 51 GB table streams; 0 is the whole weight at once. */
    int64_t (*row_block)(const RadParam* o, int n_o, const RadQuantWeight* w);
    int     (*quantize)(const RadParam* o, int n_o, const RadQuantWeight* w,
                        const float* src, int64_t row0, int64_t rows, void* const* planes);
    /* Only for a scheme the core does not decode. */
    int     (*decode)(const RadEncoding* e, const RadTensor* planes, int n_planes, float* out);
} RadQuantizerInfo;
```

`RadQuantWeight` names the weight, gives its logical shape and movement group, and hands over the
weight's imatrix column importances when the converter has them. Calibration that is not an
imatrix — GPTQ's Hessians — is an option the quantiser reads itself (`calib=DIR`): what a Hessian
is, and how it is stored, is the quantiser's business, exactly as a layout is the kernel's.

**`libquant` is the built-in set**, and every quantised plane a served container holds is its
work. It is organised by what is independent: a **grid** (the code type —
any integer width, fp8, fp6, fp4, a non-linear table, a lattice — and the scale and zero
structure over it, multilevel included, which is what makes MXFP4, NVFP4 and the K- and I-quants
grids rather than algorithms), a **transform** applied first (FWHT, a GPTQ act-order permutation,
ParoQuant's rotations; AWQ's column scales are a scale level of their own), and an **algorithm**
that chooses the codes on that grid: round-to-nearest with absmax, MSE or imatrix-searched scales;
GPTQ's error feedback; AutoRound's signed-gradient rounding; the lattice searches the I-quants
need. Each pairs with any grid it makes sense for, and a recipe names the three.

What it ships:

| quantiser | algorithm | grids | writes |
|---|---|---|---|
| `rtn` | round to nearest; `rule` absmax, mx, fixed or search | every one below | the grid |
| `gptq` | error feedback against H; `act_order=1`; `cd=N` sweeps of coordinate descent after it | every one | the grid, `perm` under act order |
| `awq` | per-column activation scales searched against H, then a per-block clip | every untransformed one | the grid and a last scale level `[* x 1]` |
| `autoround` | sign-gradient descent on rounding offsets and block ranges against H | integer codes, row blocks, one level | the grid |
| `paroquant` | Givens pair rotations and channel scales fitted against H, then AutoRound | every untransformed one | the grid and a `givens` transform |
| `ggml` | llama.cpp's own searches, imatrix-weighted | its legacy, K- and I-quant types | each type's affine form |
| `cast` | a float in another float dtype | — | `plain` |

The grid is `codes` (integers to 32 bits, fp8/6/4) or `table` (a codebook: NF4, IQ4_NL's values,
the `w4nl` table Flash-Next's experts are served in and its 32-level `w5nl` form, binary,
ternary, or given), a scale block and dtype, an optional second scale level the first is relative
to — an E4M3 or integer sub-scale under an f32 or f16, which is NVFP4's and the K-quants' shape —
an optional integer zero and an optional Hadamard. The calibrated algorithms read the same
H the engine's tap writes (below), in the domain the weight is quantised in.

Adding a quant is a quantiser plugin and a kernel that accepts its encoding; the core does not
change.

The Hessians come from the engine: `RADIANCE_CALIB_DIR` makes the architecture's calibration tap
write each layer's Gram matrices while a container serves a corpus. docs/MOE-W4.md walks through
both ways to get them: a bootstrap — convert with absmax, serve that and calibrate on it, convert
again — or a calibration run on the unquantised model itself, which is where the served
Flash-Next container's Hessians come from.
The tap records the input as the kernel multiplies it, rotated where the container stores the
weight rotated, so a Hessian lives in the domain the weight is quantised in. A layer-local
algorithm's objective is then the quadratic form sum_r (w_hat_r - w_r) H (w_hat_r - w_r)^T, which
decomposes over rows; AWQ, AutoRound and ParoQuant spend their time in D·H, and libquant sums every
element of it in a fixed order so a weight converts to the same bytes on any machine and thread
count.

---

## 5. Placement

All-in-VRAM, layer offload to RAM/SSD, and expert offload to RAM/SSD are one mechanism at three
granularities. What separates them is not the tiering but whether the access is predictable:

- **Layer offload is a scheduling problem.** Declare already enumerated every op with its weight
  operands in order, so the core knows the entire future access sequence statically. Exact
  prefetch, no policy, no predictor.
- **Expert offload is a prediction problem.** Routing is data-dependent, so there is no schedule
  to emit and the only thing left is a policy. This is why this engine needs a heat engine (§5.4)
  and llama.cpp never did.
- **All-in-VRAM is the degenerate case** where the planner assigns everything to tier 0 and the
  mover never runs.

### 5.1 Tier and site are independent axes

**Tier** is where the bytes live. **Site** is where the arithmetic happens.

| tier | site | what it is |
|---|---|---|
| `VRAM` | device | everything resident: vLLM's only mode |
| `HostPinned` / `SSD` | device, staged into a slab slot | the offload path: exact or predicted |
| `HostPinned` | device, zero-copy over the link | read-once weights, no slab slot |
| `HostPageable` | host | llama.cpp `-ngl`: the bytes and the arithmetic both stay off the card |
| `Mapped` | host | the container's own mmap — no pool, no copy |

All five are core behaviour. Host-site execution needs a `RAD_DOMAIN_HOST` kernel from a kernel
plugin (§2.1); where none resolves, that site is unavailable for that op and the planner is told
so at declare.

**`Mapped` is not "host memory", it is the container's own mapping, and that is the whole of the
difference.** The other off-device tiers all copy: pinned and pageable allocate an arena and
memcpy the weight in at load, and the file tier `pread`s it a slot at a time. A mapped weight is
never copied and occupies no pool — the loader publishes a pointer into the mmap the container
already holds, and the page cache decides what is resident. It exists for the one shape every
copying tier is wrong for: a weight of tens of gigabytes read *a handful of rows a token*, which
is what an n-gram embedding table is. Staging moves the whole table to use a few kilobytes of it,
pinning forces all of it resident on a host whose RAM is the same order as the table, and an arena
copy wants a second copy beside the mapping that already holds one; a sparse Zipfian gather of a
dozen rows out of hundreds of millions is exactly what a page cache is good at. Only a host kernel
can read one — the bytes are pageable and at no device address — so the planner assigns this tier
where, and only where, no device kernel for the op exists at all. And only to a weight whose
layout is the identity: the mapping holds the canonical planes (§4.2), so a weight a kernel
rearranges at load has nowhere in the file that already looks the way the kernel reads it. The
file tier has the same rule for the same reason (§4.3).

### 5.2 Access classes

Declared per weight, because it is information only the architecture plugin has:

| class | meaning | prefetch |
|---|---|---|
| `PER_TOKEN` | read every forward pass — dense weights | exact, from the declared order |
| `PER_REQUEST` | read once per request — an encoder tower, with an image | exact, on admission |
| `CONDITIONAL` | read only if selected — routed experts | predicted (§5.4) |
| `RARE` | may not be read at all in a given deployment | on demand |
| `VOCAB` | `lm_head` and the embedding table | exact, from the declared order |

`VOCAB` earns its own row because neither of its neighbours describes it. It is read every token,
like `PER_TOKEN`, but it is enormous at a large vocabulary and one of the two members is a gather
rather than a GEMM — so what the planner wants to know about it is its size and its access
*pattern*, and `PER_TOKEN`'s exact prefetch and `CONDITIONAL`'s prediction each answer only half
of that.

### 5.3 The planner

Runs once, after declare, before load. Takes the manifest plus explicit pool budgets and assigns
each weight a tier and a site. Strategies: `all_vram`, `layer_offload`, `expert_tiered`, `auto`.

Under tensor parallel the placement unit is a **shard** of a weight, not the whole weight — each
rank holds its own half of every resident expert and plans against its own card's budget.

Three objectives beyond fitting:

**Contiguous host runs.** A host-site op leaves its activation in host memory, so a host layer
between two device layers costs two link crossings. A few hundred KB is nothing; alternating
per-layer is not. llama.cpp gets this free because `-ngl` assigns a contiguous suffix. Here it is
an explicit objective: host-site work is assigned in runs, so activations cross once per run
boundary rather than once per layer.

**Prefill and decode want opposite plans.** A prefill chunk of a few thousand tokens touches
essentially every expert, so residency buys nothing and streaming amortises across the chunk;
decode touches eight per layer per token and residency is the whole game. The planner produces
one plan; what reconciles it with whichever phase the server is actually spending its steps in is
the heat engine (§5.4), which re-ranks from realised routing rather than from a declared intent.

**Warm start from the profile.** The container's expert-popularity profile — derived from the
imatrix's activation counts — ranks which experts begin resident. Worth about a point of hit rate
against nothing, and honestly not more: imatrix popularity is not decode popularity, and the heat
engine re-ranks from real routing within a few dispatches regardless. It is a warm start, not a
policy.

### 5.4 Residency, the mover, and heat

Fixed-stride VRAM slab slots per weight class, a pinned host pool, and a file tier read with
`O_DIRECT` into pinned buffers. Predictable latency, no page-cache double-copy, and reads land
where the mover can DMA from them without a bounce. `O_DIRECT` wants offsets, lengths and buffers
all on the logical block size, so every slot and every sub-region is a multiple of it and there is
no unaligned tail to handle anywhere.

The mover runs on a dedicated stream. The compute stream waits on its events. **No host
synchronisation is involved in a transfer**, which is what lets placement be dynamic without
serialising the step.

**Dynamic re-placement is in v1.** Heat tracking on conditional weights, promotion and demotion
against hysteresis and minimum-gain thresholds, decay per dispatch, shadow slab slots for
in-flight promotions, and a bounded move budget per layer dispatch. This is the largest single
piece of core complexity in the engine and it is also the part with no equivalent anywhere else:
it is the difference between MoE offload being unusable and being fast.

Link traffic is U-shaped in the minimum-gain threshold. Raising it moves more bytes, not fewer.
That is counterintuitive and it is measured, so it belongs in the spec rather than in a comment.

### 5.5 The dispatch contract

Routing is computed on device and **never blocks the compute stream.** Dispatch is resolved on
device against the current residency table; the routing histogram is copied back asynchronously
and the heat engine consumes it on the *next* step.

Placement decisions are therefore always one step stale, which for heat-based promotion is
irrelevant — a hot expert is hot across many steps. The real cost is different and worth being
honest about: **an expert that is not resident this step cannot be fetched in time, so it is
served from wherever it is.** Hit rate is a property of the placement policy, not of a
last-moment rescue. That is the trade, and it buys a run phase with no device round-trip in it.

---

## 6. Memory and budgets

Every pool is budgeted explicitly, and where the operator does not state a budget the engine
derives one **from the card and prints every term of it**. There is still no profiling pass, no
utilisation fraction and no inference about what the workload will be:

| | |
|---|---|
| `--gpu-headroom-mib` | VRAM left unclaimed on every card (default 96) |
| `--expert-vs-cache-ratio` | weights vs KV split of what is claimed (default 0.75) |
| `--vram-weights-mib` | the resident weight slab; 0 derives it from the card |
| `--vram-kv-mib` | the paged KV pool; 0 derives it from the card |
| `--host-pool-mib` | pinned host tier |
| `--weights-disk-tier` | file tier for weights, read from the container; needs `O_DIRECT` |

**Deriving is not inferring, and the distinction is the whole of why this is allowed.** What makes
a derived budget unreproducible is deriving it from the *free VRAM at startup*: the same command
on the same machine then measures differently depending on what else is open. Every term of this
one is either a fixed property, a computed quantity, or a measurement printed on its own line —
and what is left is spent in **two steps**, because only one of the two is a trade:

    card total          31.86 GiB   the part's totalGlobalMem
    already held       442.84 MiB   MEASURED once, after every plugin's code object is resident
    activation arena   144.47 MiB   computed from the buffer plan, by the call that allocates it
    step staging        18.02 MiB   the step batch's block tables and media staging, the same walk
    --gpu-headroom-mib  96.00 MiB   stated
    ------------------------------
    claimable           31.19 GiB
    static weights       3.94 GiB   off the top: every rank's own shard, not a share of anything
    ------------------------------
    elastic             27.25 GiB   what --expert-vs-cache-ratio divides
      experts           26.18 GiB
      kv cache           1.08 GiB   a full cache for 8 sequences at 2048 tokens
    ==============================
    vram_weights        30.11 GiB   static + experts
    vram_kv              1.08 GiB

**The static weights are a floor, not a share.** Everything that is not an expert — the attention
projections, the norms, the embedding, the head — has to be resident for the model to run at a
sensible speed at all, and under tensor parallel each rank already holds its own even share of
them (declare runs per rank with every dimension divided, §9). Dividing the weights *as a whole*
by a ratio would let `--expert-vs-cache-ratio 0.3` demote an attention projection to host memory
to make room for a cache nothing will fill. So they come out whole first, and what remains is the
one genuinely two-sided trade this engine has: the expert plane against the KV cache.

**Both sides are then capped at what they could use, and the slack crosses over.** The ceilings
are upper bounds and not targets. An expert plane that is entirely resident has no use for another
byte. A paged cache cannot have more than `max-num-seqs` × `max-model-len` tokens live at one
moment however large the pool is — but landing *under* that is the normal case, not a shortfall,
and it is the difference between this design and a static one:

> A sequence holds only the blocks its tokens actually occupy. `KVManager::ensure` grows it by
> `ceil_div(n_tokens, block_size)` as it runs, drops them at completion, and returns `RAD_E_FULL`
> for the scheduler to preempt rather than failing the request. Every sequence reaching the
> context limit simultaneously is the worst case, not the workload; production runs well under
> half backing routinely.

So the engine **reports** the backing fraction and does not warn about it. Sizing for the worst
case spends VRAM that, on a model whose experts do not fit, is about 1.4 ms a decode step per GiB.

What *is* a hard limit, and is still checked: the pool must hold **one** max-length sequence. A
context no single request can ever complete is a configuration error however much slack the pool
has, because the API accepts that request and admission then refuses it.

The trade between KV blocks and resident weights is real and workload-dependent — a long-context
low-concurrency server and a short-context high-concurrency one want opposite splits — and stating
it is still the operator's to do: `--vram-weights-mib` and `--vram-kv-mib` win outright wherever
they are given, and nothing re-optimises a split that was stated. An engine that quietly
re-optimises is an engine whose benchmarks do not reproduce.
A fixed weight-slab budget also means dynamic re-placement (§5.4) never competes with the KV pool.
The mover works inside its own slab and cannot evict a KV block to promote an expert.

**What the budget cannot compute, it measures and reports.** `Engine::start` reads the card back
once everything has allocated and says what settled there against what `--gpu-headroom-mib` asked
to keep. The engine's own device allocations are ledgered (`rad_dev_alloc_totals`) and are, on a
served model, exactly the two pools, the activation arena and the step staging; the difference between that and the
card is the driver's context, its code objects, and any kernel plugin that calls `hipMalloc` for
itself. That difference is the one thing the headroom exists to cover, and it is printed rather
than assumed.

---

## 7. Scheduling

vLLM's core design, which is the part of vLLM that is right.

### 7.1 Batching

Continuous batching, chunked prefill, priority queue, preemption by recompute. One scheduler
decision per step, producing one `RadBatch` (§8).

Chunk boundaries are the scheduler's, and it uses that: chunks split at linear-state checkpoint
intervals (§7.3), and the chunk size reconciles GDN state geometry with the attention block size
rather than being a free number. That reconciliation is what forces the vLLM patch set to require
`--max-num-batched-tokens >= 2240` on the 35B-A3B; here it is computed from the resolved kernels
instead of being a documented minimum the operator has to know.

### 7.2 Paged KV and hybrid groups

**Block size comes from the resolved attention kernel, not from a core constant.** libr4d's paged
attention is compiled for block size 16 and rejects anything else; the block manager reads that
off the resolved kernel's constraints at declare and sizes itself accordingly. A core that picked
its own block size would be a core that has to be edited when a kernel changes.

Hybrid models are the normal case, not an extension: Qwen3-Next is 48 linear-attention + 16 full
layers, Gemma-4 is 50 sliding-window + 10 global. Each distinct state kind is a **KV group** with
its own block geometry, its own pool, and its own slot mapping in the step batch. Group sizing
minimises wasted memory rather than taking the smallest bundle — vLLM pads each bundle up to a
multiple of the group size and allocates the padding, which is what the KV-group-sizing patch in
the table at the top exists to stop.

Linear-attention state (GDN recurrent state, conv state) is a KV group too. It is not paged the
same way — the state is per-sequence and fixed-size rather than per-token and growing — but it is
allocated, mapped, and freed by the same manager, because otherwise two allocators disagree about
what a sequence owns.

### 7.3 Prefix caching

Full-attention layers: content-hashed blocks with a chained hash, so a block's identity includes
its whole prefix; LRU eviction; copy-on-write on fork. Multimodal inputs hash by content, so an
image is a cache hit.

Linear-attention layers cannot work that way. A GDN layer's recurrent state at position N depends
on all of 0..N and is one fixed-size object per sequence rather than one per token — 32 heads of
128×128 fp32 is 2 MiB a layer, so roughly 96 MiB across the 48 linear layers of a Qwen3-Next.
Snapshotting that per block is not affordable.

vLLM's answer is a state checkpoint at block boundaries: `--mamba-cache-mode align` keeps the
state of the last token of a scheduler step when that token sits at `i * block_size`. We take the
mechanism and change two things about it, both of which are forced or nearly free.

**The checkpoint interval is decoupled from the attention block size.** vLLM inflates the
attention block to match the mamba page — 528 tokens for Qwen3.5, 2240 for Qwen3.6-35B-A3B — and
since prefix caching is block-granular, every prompt shorter than that gets a 0% hit rate on
*attention* as well, which is a standing complaint. We cannot inflate the attention block even if
we wanted to: §7.2 takes it from the resolved kernel and libr4d's paged attention is compiled for
16. So attention caches at 16-token granularity and the linear checkpoint interval is a separate,
much larger number. A hit landing between checkpoints reuses the attention blocks and replays
only the linear layers forward from the last checkpoint.

**Checkpoints are taken deliberately, not opportunistically.** vLLM writes one only when a
scheduler step happens to end on a boundary, which is why a checkpoint landing in request-unique
tokens silently drops caching to zero. Our scheduler owns chunk boundaries (§7.1), so it splits a
chunk at the interval and always writes one. The cost is an occasional short chunk; the benefit is
that hit rate is a property of the workload rather than of scheduling luck.

Retention is budgeted, since the snapshots are large — at a 2048-token interval a 128K session is
64 checkpoints and about 6 GiB. The session tip is always kept, because that is what an appending
agent transcript actually needs.

**Beyond the tip, retention is geometric backoff and that is the default.** `--checkpoint-policy`
names it: `geometric-backoff` keeps the tip and then checkpoints 1×, 2×, 4×, 8× back, which bounds
the snapshot count at log(n) and the recompute at one interval-doubling — the two things keeping
everything and keeping only the tip each get wrong in one direction. `tip-only` and `keep-all` are
the two ends of that trade and ship beside it, because a default nobody can run an alternative
against is a default nobody can falsify. Distances are counted in **checkpoints**, not tokens:
the scheduler splits chunks at the interval so the spacing is uniform, and counting that way stays
correct if it ever is not. Retention is a policy object rather than an if-statement, so replacing
it is writing a class (§19.1).

`--checkpoint-slots` bounds the snapshots **in total**, not per sequence: they come out of one
global free list, so an operator reading it the other way and setting 2 gets two for the whole
server rather than two per sequence.

### Idle session tiers

A finished turn leaves its prefix in the cache holding KV blocks, which is what makes the next turn
cheap. Those blocks are the most valuable thing in the pool while the session is live and the least
valuable thing in it once the session stops. So a finished entry is **copied** to host memory at
once and keeps its VRAM copy; once the copy has landed, giving the VRAM up costs nothing, and it is
given up the moment something wants it — the expert slab's loan, or a request the pool cannot
otherwise serve. A restore keeps the host copy, so the next turn copies only what it added:

    published --(copy)--> clean in VRAM --(memory wanted)--> host
    host copy --(copy)--> clean in host memory --(slot wanted)--> disk

The disk tier is a copy for the same reason: every host copy is written on to disk in the
background as soon as it lands, and keeps its host slot. Once the write has landed the slot is as
free to give up as a clean entry's VRAM, and a copy or a restore that finds the arena full takes
the oldest such slots — host entries first, then the host copies of clean VRAM entries, which stay
clean through their disk copy. Moving an entry to disk only when its slot is wanted would put the
write on the path of whatever wanted it, and a restore out of disk, which needs a host slot to read
through, could find an arena full of host-only entries with nothing to give.

`--prefix-cache-host-mib` and `--prefix-cache-disk-mib` size the two tiers, and `--prefix-cache-dir` says where
the disk one lives. Both caps default to zero: a tier of no bytes holds nothing, so the feature is
off until an operator says how much memory it may have. With no disk tier a full host tier drops
its oldest entries; the disk store evicts its own oldest entries, and a restore that finds one gone
drops it from the cache. A hit in host memory promotes straight back to the device.

A hit on disk is read into host memory first, on the tiers' IO thread, and the request waits for it
outside the step: the scheduler asks the next request in the queue instead, and admits this one on
the step after its read has landed. The restore itself then copies host memory to the device and
nothing else. Reading inside the restore would run the reads one block at a time on the thread that
builds every step, so a conversation coming back off disk would stall every other conversation for
as long as its reads took. The read is parallel -- many blocks in flight at once, each checked
against the key it was asked for -- and the slots it fills are held for the restore that asked:
neither the rest of its own chain nor anything else may take them back until that restore has run,
or for ten seconds if it never does. The request is not admitted with a shorter hit while its read
is out, because the hit length decides how its prefill is chunked, and a prompt must run the same
shapes however far a read had got. A chain longer than the arena is read as far as the arena reaches
and the rest re-prefilled.

The trade is worth making because the alternative is a re-prefill. Dropping a 200K-token prefix
costs minutes; moving it costs a copy over PCIe.

**The pool is layer-major, and that is what makes this affordable.** A logical block is `n_layers`
fragments of a few hundred bytes, so moving one block at a time would be dispatch cost with the
data as a rounding error. Turned around, *within* a layer the blocks are dense rows, which makes
packing an arbitrary set of them exactly `gather_rows` and restoring them exactly `scatter_rows`
(docs/OPS.md): one launch per (group, layer) for a whole batch, however many blocks are moving.

**On a hybrid model the blocks are only half of a session.** The other half is the recurrent state,
which lives in a checkpoint slot — one contiguous `checkpoint_bytes()` extent, of which there are
`--checkpoint-slots` for the whole server. Eight by default, against the six or seven a single 200K
transcript retains under geometric backoff, so a snapshot is the scarcest thing in the pool.
Tiering the blocks and leaving the snapshot behind gets the trade backwards: it frees the plentiful
resource and holds the scarce one, and the session comes back with its attention hit intact, finds
no checkpoint at any position its chain can reach, and replays every recurrent layer from token
zero. So a snapshot moves with the blocks whose hash it is keyed by, and the checkpoint slot goes
back to the pool for the duration. There is no gather on this path — the slot is already
contiguous — so it is one copy each way per rank.

In this implementation the snapshot is not the smaller half — it decides whether there is a hit at
all. The scheduler clamps a hybrid session's reuse to the last linear checkpoint it can reach,
because `RadBatch` carries one query length for every KV group and the linear layers cannot be
replayed over a longer range than the attention ones, so a session restored without a reachable
snapshot reuses ZERO tokens and re-prefills its whole transcript. The two halves are still reported
separately, because a snapshot whose copy failed is still correct in its device slot and costs
nothing to leave there, while one that is dropped costs the session everything. A snapshot has no
LRU of its own — it ages with the blocks it belongs to, because one key on two lists gives one entry
two lifetimes that can disagree.

The snapshot store's share of each tier is **derived from the session the operator sized the server
for**, and carved out of the cap they already set rather than added to it. At `--max-model-len` M a
session costs `(M / block_size) * slot_bytes` in blocks and `(log2(M / interval) + 1) * ck_bytes`
in snapshots, and splitting the cap in that ratio makes both halves hold the same NUMBER OF
SESSIONS — the only sense in which they can be balanced. The ratio is strongly length dependent,
because the retained set grows logarithmically while the blocks grow linearly: snapshots want about
a fifth of the tier at 200K and more than half at 32K, which is why a fixed fraction is the wrong
shape. Two thirds is the ceiling, not because the arithmetic asks for one but because the clamp
above is a property of this implementation rather than of the design. The startup line says how
many slots it came to.

**What can go wrong here is silent, so it is tested by byte comparison.** Corrupt KV does not read
as babble — it reads as fluent, deterministic, plausible text that simply is not what the model
would have produced. `scripts/tierident.sh` asks one prompt cold, again from VRAM, again after the
session has reached RAM, and again after it has reached disk, and compares output hashes. The VRAM
repeat is the control: without it a differing hash proves nothing, because a cache hit could have
been legitimately tie-divergent. A run in which the session never left VRAM is a failure, not a
pass.

**Half of that test cannot be a hash, and this is the part that is easy to get wrong.** A lost
snapshot *is not visible in the text*: the engine replays the linear layers from further back and
produces exactly the same answer, slower. So the snapshot checks are counters — how many reached
each tier, how many came back, and how many prompt tokens a lookup then *covered* with restored
state. Those last two are different claims: a restore that puts the state somewhere the chain cannot
reach scores perfectly on one and zero on the other.

The disk tier holds conversation content unencrypted and says so at startup; it is fingerprinted on
the container, layout, KV width and rank count, because a block read back under any other is a
lossless function of somebody else's weights. `/sessions` reports what is stored, where, and what
the tiers have actually done — a session table that cannot say "it tried and failed" looks the same
whether the tiers are working, off, or failing every pass.

---

## 8. The step batch

`RadBatch` is built **once per step by the core** and shared by every layer and every KV group.
Four of the vLLM patches in the table at the top exist because vLLM builds this per KV-cache group,
per layer, in Python, where it made 13% of a decode step host time with the GPU idle.

It carries: token ids and positions; per-group slot mappings and block tables; sequence boundaries
(`cu_seqlens`); query and context lengths with their upper bounds; the speculative window layout
(§10); multimodal embedding spans (§11); the phase, which the run phase uses to index bucket tables
(§2.2); and the DECODE/PREFILL SPLIT.

The split is `n_seq_decode` and `n_tok_decode`: the core orders every step so its decode rows come
first, and those two say where they end. A block that has to treat the two kinds of row
differently — a linear block must, see §10 — then issues over each as a contiguous sub-range
rather than choosing one path for the whole step. `n_seq_decode` equals `n_seq` on a pure decode
step and is 0 on a pure prefill one, so a block that does not care can ignore it and one that does
needs no phase test.

An architecture plugin that needs derived per-step metadata — GDN chunk boundaries, conv-state
indices — declares it at build time as a derived buffer, and the core computes it once per step
into the arena. It is not recomputed per layer, and layers that share a geometry share the buffer.

---

## 9. Tensor parallel

Declare runs once per rank with `rank` and `world_size` in `RadBuildCtx`. The plugin declares
dimensions already divided, and emits collectives as ordinary ops resolved through the same
kernel hierarchy:

```c
decl_op(b, "all_reduce", RAD_PARAMS({"world_size", ctx->world_size},
                                    {"dtype", "bf16"},
                                    {"exact", 1}), ...);
```

The core does not reason about sharding propagation and does not insert collectives. That is
deliberate: a sharding inference pass is a compiler pass, and its failure mode is a model that
produces fluent wrong text on a rank count nobody tested. Here the graph dump shows every
collective exactly where it runs, because the plugin put it there.

`lm_head` is vocab-sharded like any other column-parallel weight, so the sampler (§13) runs a
distributed top-k: each rank reduces its own vocab shard, and only the candidate sets are
combined. Full logits are never gathered anywhere.

libr4d supplies the all-reduce at 2, 4 and 8 ranks, in an exact form and a quantised-wire one.
Which of the two a deployment accepts is `RadBuildCtx::tp_wire_lossy`, and the plugin inverts it
into `all_reduce`'s `exact` parameter. **The polarity is that way round deliberately:** a zeroed
context means exact, because the sum a rank receives over a lossy wire is not the sum it would
have received, and a field that defaults to lossy when someone forgets to fill it is precisely the
failure §17 exists to prevent. `tp_wire_min_bytes` is where the lossy wire starts paying — below
it the exact kernel wins and the plugin must use it even when lossy is set, because compression
only pays once the transfer is bandwidth-bound and a decode step's message is not.

---

## 10. Speculative decoding

**A drafter is not a separate architecture plugin.** The target model's own plugin declares it,
into the same builder under a draft scope, so its kernels resolve through the same hierarchy and
its weights are placed by the same planner:

```c
RadDrafterDecl d = {};
d.name = "mtp";  d.kind = RAD_DRAFT_SERIAL;  d.depth = ctx->max_spec;
d.proposal = prop;  d.proposal_pitch = pitch;   /* 0 = the sampler's own token buffer */
d.mask_token = mask;  d.window = w;  d.block = n;   /* RAD_DRAFT_BLOCK only */
rad_declare_drafter(b, &d);
```

It replaces a weight-name match, and that is the whole argument for it. The alternative is an
engine that decides whether a container drafts by scanning declared weights for names like
`mtp.fc.weight`, and then re-reads the drafter's own configuration keys — block size, mask token,
sliding window, predicted-layer count — every one of them a key the plugin has already read to
build the thing. Two readers of one fact is two places for it to disagree, and the second reader
would be the core, which is the one component that must not know.

**Kind is a shape, not a model.** `RAD_DRAFT_SERIAL` runs `depth` dependent passes, each embedding
what the last one proposed, so tokens arrive one at a time and a pass that declines ends the
chain. `RAD_DRAFT_BLOCK` runs a context pass and then one pass that fills every drafted position
at once. MTP, EAGLE and Medusa-serial are all serial; a block-diffusion drafter is block. A plugin
picks the shape that describes it and needs no core change to do it. A block drafter that reads the
trunk's hidden states at the last accepted position finds that row on its query pass in
`RadBatch::draft_out_ids` — a device index, so the pass's operands do not change between steps and
its recording replays; host-side row numbers baked into operand offsets would not.

Separate draft models would load as a second plugin, which is a design and not a feature. A
drafter is part of the model's source instead: `rad-convert --draft-model` merges its checkpoint
into the container under the `draft.` prefix, and an MTP head that ships inside the target
checkpoint needs no flag at all. The server's `--draft-model` refuses, naming the conversion that
does it — loudly, rather than quietly serving the container's own drafter instead.

Verification is a declared op. Acceptance is core sampler logic. **Draft length is static** — it is
`--num-speculative-tokens` and nothing moves it per step. There is no per-step controller, and its
absence is a decision rather than an omission: the depth that wins is a constant, a controller
that can only pick a worse one contributes nothing but the risk that it does, and the gate such a
controller needs reads the drafter's own confidences AFTER the drafter has run — which is after
the scheduler has had to choose. `core/sched/draft.h` carries the structural argument and the
acceptance accounting.

**The flag defaults to `auto`, and merging a drafter into a container is the statement of
intent.** `auto` is `n_spec = -1`, resolved before declare from what `rad_arch_probe()` reports:
a serial head names a measured throughput trade-off, a block drafter names the block length its
checkpoint was trained at, and `draft_depth_fixed` says which of the two it is. A container with
no draft head resolves to 0, which is the right answer to "speculate as deeply as this model can"
on a model that cannot. Defaulting to off instead would leave a deployment that forgot the flag
paying for a draft head at convert time, paying for it again in VRAM, and then running one token
a step.

**The scheduler carries a drafter of its own, underneath the declared one.** A prompt-lookup
n-gram matcher — patterns of 2 to 4 tokens, longest match first, most recent occurrence first —
fills any proposal the declared head left empty, and is the whole of speculation on a plugin that
declares no drafter at all, which is not an error and is the normal case. It costs nothing to
have: it is host string matching over tokens the scheduler is already holding, and because it only
ever extends a pattern the sequence has already produced, it proposes nothing where there is
nothing to repeat. It sits behind the same eligibility gate as the declared head — refusing a
draft upstream and stopping there would hand the request a *different* draft rather than none.
`RADIANCE_NO_PROMPT_LOOKUP=1` disables only this fallback, leaving the declared
neural drafter active. It defaults off and is independent of the model's n-gram
embedding tables and `--ngram-placement`. Use the same setting on all matched
arms when attributing proposal counts or acceptance to MTP alone.
The environment key must include the `RADIANCE_` prefix; `NO_PROMPT_LOOKUP`
alone is not a control. The startup banner reflects the canonical flag.
`RADIANCE_NO_DRAFT=1` remains the control that disables both proposal sources.

The speculative window is part of `RadBatch`, which is why attention kernels take a query length
rather than assuming one: libr4d's decode attention serves up to 64 query rows per KV head
(`q_len * gqa <= 64`) precisely because a verify step falls in that band.

**A MIXED STEP VERIFIES LIKE ANY OTHER, AND THAT IS A CONTRACT ON THE ARCHITECTURE PLUGIN.** A step
carrying a prefill chunk beside decode rows is what continuous batching produces constantly — on an
agentic load 28.5% of steps, and none of them prefill-only, because every chunk rides with decode
rows. The obvious linear block issues one path for the whole step, and then its decode rows are on
the chunked path, which commits the state directly and takes no `num_accepted`: a draft verified
there cannot be rolled back, the rejected tokens stay folded into the recurrent state, and the
model is conditioned on text it never emitted. The answer stays fluent and drifts.

So a linear block **must issue the recurrent path over the decode rows and the chunked path over
the prefill ones**, as two issues against two ranges. `RadBatch` carries the boundary and the core
orders every step decode-first so each half is a contiguous run of sequences (§8). It costs no
kernel change: these kernels take `cu_seqlens`, derive their grid from its length and their rows
from `cu[n]`, which is an absolute offset — so the prefill half is addressed by passing `cu + D`
with the token operands left at their base, and the decode half by narrowing both. Only the ops
that take no `cu_seqlens` — a per-token gate or quantiser — need a row slice.

Refusing to draft on a mixed step is the other way to be correct, and it costs a quarter of all
decode rows their draft.

**Rejection rolls back state, and the two kinds roll back differently.** Paged KV is a slot-table
edit — free the blocks past the accepted position. Linear state is not: the recurrent state and
the conv window have already absorbed the rejected tokens. libr4d's conv update handles this by
treating the state cache as a rolling window of `width-1 + num_spec` entries and reading at the
slot the last accepted token left, so a rejection is a change of read offset rather than a
recompute. The engine's contract is that a linear-attention kernel pair must support this, and it
is part of the op's schema rather than an assumption.

**A declared tree proposal is additive and optional beside the linear chain.** `RadDrafterDecl`
remains unchanged and carries linear proposals (`RadDrafterDecl::proposal`). An architecture plugin
with a block or serial drafter that can propose tree candidates declares a separate output buffer:

```c
rad_declare_drafter_tree(b, tree_buf, node_capacity);
```

The declared `tree` buffer is a `RAD_I32` rank-2 tensor with fixed shape
`[max_seqs, 1 + 2 * node_capacity]` (row width `1 + 2 * node_capacity`), laid out per sequence row as:

```text
[count, token[node_capacity], parent[node_capacity]]
```

`node_capacity` is root-inclusive, bounded to `1..16` (`RAD_TREE_MAX_NODES = 16`). Node 0 is the
root, included in `count`, with parent `-1`. Every non-root node has parent `0 <= parent < node_idx`.
Unused trailing row entries are ignored.

At declare time, `rad_declare_drafter_tree()` performs strict validation before any state is mutated:
- Validates that declare is not finished (`RAD_E_STATE`).
- Validates that no tree buffer was previously declared (`RAD_E_STATE` on duplicate).
- Validates that an existing `RAD_DRAFT_BLOCK` or `RAD_DRAFT_SERIAL` drafter is declared (`RAD_E_INVAL`).
- Validates `1 <= node_capacity <= 16` (`RAD_E_INVAL`).
- Validates the buffer handle, verifying it is a declared buffer different from the linear proposal buffer and logits buffer (`RAD_E_INVAL`).
- Validates that the buffer has `dtype == RAD_I32`, `rank == 2`, row width exactly `1 + 2 * node_capacity`, row count `shape[0] >= max_seqs` (and `> 0`), and total byte size does not overflow `int64` (`RAD_E_INVAL`).

The logits/tree distinction is independent of declaration order: `rad_declare_logits()` rejects
an already-declared tree buffer with `RAD_E_INVAL`, without changing Program metadata. Its existing
`RAD_E_STATE` checks for finished declaration or a second, different logits buffer take precedence.

Declaring a tree records Program metadata (`drafter_tree_buf` and `drafter_tree_capacity`) and claims
no branching capability. `prog.branched_tree_supported` is untouched by declaration; unsupported
recurrence/attention topologies continue to veto branching monotonically.

**A serial drafter constructs its tree on the device.** Its architecture writes the declared
packed tree. The legacy MTP policies evaluate one spine: round `r+1` embeds round `r`'s pick,
so round `r`'s other candidates are legal children only of the spine node at depth `r-1`.
The pruned MTP policy below instead evaluates a fixed-width frontier with branch-specific
hidden carry and KV. Both serial and block declarations require a real, shape-checked tree
buffer; `RAD_NULL_HANDLE` is invalid. The engine reads final packed rows once for validation
and publication, never constructing the proposal topology on the host.

### Bounded DarTree for the built-in drafters (`RADIANCE_DARTREE`)

`RADIANCE_DARTREE=1` selects a TREE proposal for whichever built-in drafter the program declares.
It is opt-in by name and orthogonal to the drafter choice: unset, or `0`, declares exactly the graph
the deployment had before. It does not apply to DLoop, which is a tree drafter of its own.

**What it changes, and what it does not.** For DFlash2 the change is the SELECTOR only:
`dflash_select` walks one path through the candidate lattice the query pass already produced, and
`dflash_tree_select` keeps exactly that edge arithmetic while growing a bounded, pruned tree over
the same candidate planes. The backbone is untouched — a query pass still runs `block` rows a
sequence, `block = 4` (three proposal levels) is the trained depth, and the tree adds no row, no
pass and no level. `NUM_SPECULATIVE_TOKENS` is the VERIFICATION row budget the tree is pruned to,
never a depth: `RadBuildCtx::max_spec + 1` nodes root-included, capped at the verifier's 16.

For legacy MTP policies, `mtp_tree_accum` constructs the topology after each serial head round.
A serial head proposes one token a round and each round is conditioned on the round before it,
so round `r`'s remaining candidates are legal children of the spine node at depth `r-1`
and of nothing else — no pass ever ran under them. The device accumulator reserves the complete
available spine first, then adds each round's alternates under the node that conditioned them.
Descendants are never invented for a side leaf, and a
candidate repeating an id already placed under that parent is dropped, because
`Scheduler::set_draft_tree` refuses duplicate siblings.

**The verifier is the existing one.** A program that may verify a branched window declares the
tree-state ops — `gdn_tree_commit`, `qsa_tree_*`, `ple_tree_commit` — and `core/sched/tree_state.h`
answers from the DECLARED program, so `RADIANCE_DARTREE` turns those declarations on for the target
as well as the selector on for the drafter. A deployment that declared the proposal without them
would have every branched window downgraded to its greedy chain: correct, and no faster than the
linear walk it replaced. The long-context QSA path keeps its linear fallback, and a branched window
past the exactness bound is served as the proposal's greedy chain, exactly as DLoop already does.

Winning-branch paged KV compaction uses the core device seam, selected by the runtime backend
rather than by HIP build availability. AQL submits one kernel per pool and rank, with the bounded
move list carried by value; the host backend queues the same sequential copies on its stream. Each layer/head
executes moves in list order, including write-to-later-read dependencies. Copies include the entire
declared token width, including packed metadata, unaligned rows and byte tails; padding outside
that width stays unchanged. Empty and identity move lists submit nothing. Enqueue errors abort
acceptance before scheduler commit.

#### Scored MTP proposal policy

The qwen4exp built-in MTP path has a startup-only experimental selector:

| Environment variable | Values and default |
|---|---|
| `RADIANCE_MTP_TREE_POLICY` | `off` (default), `score`, `cost`, `pruned` |
| `RADIANCE_MTP_TREE_ROUNDS` | `3` (default), `5` |
| `RADIANCE_MTP_TREE_COSTS` | Required only for `cost`: exactly eight comma-separated finite positive values |

Unset flags preserve the existing three-round, spine-first tree. `off` does
not compute proxy normalization, gather score metadata, or retain score history.
`off` with five rounds uses the extended accumulator without scoring.
These flags affect MTP, not DFlash2 or DLoop; five rounds are serial MTP head
passes, not permission to extend a diffusion drafter's trained proposal length.
For every serial-tree policy, the startup pass count uses the same configured
round clamp as the driver; it does not confuse declared verification capacity
with actual serial rounds.

`score` or five rounds requires `RADIANCE_DARTREE=1`, TP2 with a sharded
vocabulary, the two-bit draft head with FP32 reranking, and
`--num-speculative-tokens` between the configured round count and seven.
Unsupported declarations fail rather than silently selecting another policy.
The token budget bounds proposal nodes; packed capacity includes one extra root.

`score` selects ancestor-closed nodes by cumulative log proposal probability.
The proposal distribution is a proxy: full coarse BF16 vocabulary logits with
the local top-eight reranked FP32 logits substituted, normalized across both
shards. It is neither an exact full-head distribution nor a top-eight softmax.
Only the greedy spine has evaluated descendants. The selector never constructs
descendants below unevaluated side leaves. `mine` constructs the packed tree
on the GPU; `mine_ref` provides the host oracle for the custom ops documented
in `docs/OPS.md`.

`cost` chooses a prefix of the scored selection by probability mass per
complete decode-step cost, including the root-only choice. Cost entry `n`
measures a complete step verifying root plus `n` proposals, for `n=0..7`;
it is not an isolated verifier-kernel duration. Use consistent units and
measure costs for the deployment and workload being evaluated. Costs from
another concurrency, context, model, or runtime configuration are not a
general calibration. There are no built-in measured costs. Missing or malformed
costs fail startup; providing costs with `off` or `score` also fails.

Packed count selects actual target verification rows. Count one means a valid
root-only, no-draft step; count zero is invalid and aborts the step.
These controls establish no acceptance, speedup, parity, or gate result.
`candidates/mtp-scored-tree/gate.json` enables the three-round `score` candidate
with the gate's default strict parity requirement; it does not declare
`reordered` numerics or configure unmeasured `cost` values.

#### Genuine Pruned MTP policy

`RADIANCE_MTP_TREE_POLICY=pruned`, `RADIANCE_MTP_TREE_ROUNDS=5`,
`RADIANCE_DARTREE=1`, and `--num-speculative-tokens 7` select the opt-in
W8/K8/D5/B7 trial. TP2, a sharded two-bit candidate head with exact reranking,
and adaptive drafting off are required. Other policies and defaults are unchanged.
`candidates/mtp-pruned-tree/gate.json` declares this configuration and `reordered`
numerics; it establishes no parity, acceptance, or speed result.
The shared drafter ABI still declares `depth=7`, matching `max_spec=7` and
the seven-proposal verification capacity. The serial driver independently
executes five actual MTP rounds; it does not declare a depth-five program.

The history pass produces eight root candidates. Four subsequent real MTP
transformer rounds each evaluate eight selected branches per request, gathering
the selected parent's wide hidden carry and attending to its own ancestor KV.
Each parent normalizes its merged eight candidates with candidate-set logsumexp,
not the scored policy's hybrid full-vocabulary normalization. Each level retains
the global best eight parent/token edges. K8 is an explicit adaptation from
the pinned official DARTree default K64; unevaluated side-leaf descendants and
target-hidden interpolation are not used.

The GPU retains all forty proposal nodes until depth five, then selects seven
by cumulative log probability minus `0.2 * depth`. Ancestor-first old-ID ties
preserve parent closure, including negative-infinity ties. The eight-node
root-inclusive result uses the existing target verifier and winner commit.
There is no reserved spine and no early top-seven pruning.

Committed MTP KV remains in its existing `RAD_KV_FULL` group. Frontier rounds
leave that cache and the committed QSA state read-only, using thirty-two
provisional KV token slots per tiled request plus four ancestry indices/query.
The additional provisional state does not grow with context length; QSA's
2048-token selection budget is not a sliding-window allocation claim. After
acceptance, MTP history is rebuilt from the accepted target hidden rows before
another frontier runs. Request tiles respect existing activation/head row limits.
Pruned keeps five actual rounds and a seven-proposal verification budget at
every supported concurrency; it does not inherit the legacy four-node
concurrent-tree ceiling. Context/output eligibility still bounds each request.

The model-owned `mtp_overlay_attn` keeps BF16 queries/output and the existing KV
dtype but uses FP32 products, softmax and accumulation with a different reduction
order from the normal MTP attention. Its numerics are explicitly reordered;
target precision and target arithmetic are unchanged. Neither obligatory W-times
weight bandwidth nor one DRAM weight stream per batch is established.

Pruned proposals require the existing target's dense-exact QSA route. Incompatible
QSA bounds, sampler branching, or a whole mixed batch switching sparse fail
explicitly instead of silently converting this policy to a chain. A request
near its normal context limit becomes history-only when depth five no longer
fits; other eligible requests continue drafting. Existing legacy fallback
behavior is unchanged. Invalid proposal/attention metadata invalidates the
packed count and aborts publication.

`mtp_pruned_tree_test` and `mtp_pruned_gpu_test` exercise the operation seams.
Matched end-to-end serving must additionally prove the real frontier row counts,
branched target verification, repeated-step lifecycle, and measured acceptance
and speed; the configuration alone is not verification.

#### One-position SpecHub sampling and verification

`RADIANCE_SPECHUB=1` is an independent k=2 sibling experiment, not a
five-round MTP tree policy. It requires serial Qwen MTP, the TP2 reranked
candidate head, two proposal slots, DarTree on, adaptive drafting off, and
prompt lookup explicitly disabled. The declared two-slot capacity remains
separate from its single actual MTP history round. Explicit
`RADIANCE_SPECHUB=0`, like an unset flag, keeps the ordinary path; verifier
options are valid only when SpecHub is enabled.

The device samples the paper's sparse ordered-pair law from a normalized
top-eight candidate distribution and constructs the root-plus-siblings tree.
Verification consumes the complete filtered target candidate distribution,
including residual mass outside proposal support. Request filters require
positive temperature and explicit bounded top-k; grammar, XTC, and thinking
budgets are refused. `RADIANCE_SPECHUB_VERIFY=match` is the same-proposal,
same-topology independent-target control; `transport` selects the coupling.
FP64 normalization and separate RNG streams do not promise per-seed token
identity with the ordinary picker. The candidate declares reordered numerics.

Sequence/seed/output-position/root/sibling metadata binds proposal and
verification. Only the root sample changes; existing device/host acceptance,
winner compaction, EOG truncation, and selected-path state commit remain
authoritative. A bonus token is not committed before it is forwarded.
One remaining output slot bypasses proposal. Other truncated windows retain
the full pair law but accept only evaluated children. `docs/OPS.md` defines
the wire operations; probability tests alone do not prove engine lifecycle.

#### Linear MTP maximal coupling

`RADIANCE_MTP_COUPLED=1` (default off; explicit `0` equals unset) makes the
linear builtin MTP chain propose a sample `x ~ Q` in every temporal round,
where `Q` is the FP64 temperature-one softmax of the unchanged final reranked
top-eight, and makes the target sampler verify it by maximal coupling: emit
`x` with probability `min(1, P(x)/Q(x))`, otherwise sample `(P-Q)+`, where `P`
is the actual filtered target candidate distribution after every existing
filter. Equality acceptance, commit and the bonus row are unchanged. It
excludes DarTree, DLoop, FullDrafter, DFlash2 and SpecHub, and requires
prompt lookup disabled and `RADIANCE_NO_DRAFT` unset. Temperature-0 and
top-k-1 requests keep the greedy proposal and the ordinary sampler.

Exactness, given an ideal independent RNG: for every `y`,
`Pr(emit y) = min(P(y),Q(y)) + Pr(reject) (P(y)-Q(y))+ / R = P(y)`, because
`Pr(reject) = sum_x (Q(x)-P(x))+ = R`, and `(P(x)-Q(x))+ = 0` whenever `x`
can be rejected. Each verify row is conditioned on the same draft prefix as
its `Q`, so the chain argument of speculative sampling applies row by row. It
requires: the stored `q` is the law `x` was drawn from (entries are dropped,
never renormalized, after underflow); `x`'s stream `(seed, pre-commit output
size, round)` never repeats for a request; acceptance and residual streams
are distinct keys at the verify position; and numerical cancellation falls
back to direct FP64 `P` sampling with the independent residual uniform.
That fallback is chosen only from `(P,Q)`, never from `x` or a drawn uniform. The limits are
finite precision: FP64 normalization, the 53-bit uniform's granularity,
fall-through to the last live entry when rounded mass falls short, and the
counter-based splitmix streams standing in for independent ones. Coupled rows
follow the FP64 law and ordinary rows the FP32 pick's, so neither tokens per
seed nor the FP32 finite-precision law of the ordinary sampler are preserved.

Lifecycle: draws are staged per drafting row from an explicit card/sequence
map before the serial passes; every pass also checks the row's card against
`RadBatch::seq_ids`. After commit the engine binds each round to the verify
output position and refuses a step whose stochastic rows did not come back
sampled or disagree with the card's draft token; the sampler stores the states
per request id (erased with the request) and refuses a running stochastic
draft row without a state bound to its seed, position, round and token.
`spechub_test`, `sample_test` and `drafter_integration_regression_test` cover
the shared math, the sampler binding and the engine binding; live serving
must still show acceptance and lifecycle under mixed, cancelled and TP2 load.




### DLoop DARTree drafter (qwen4exp)

`RADIANCE_DLOOP_DRAFTER=1` selects the DLoop diffusion drafter instead of MTP.
It is mutually exclusive with FullDrafter, which retains its existing selection and implementation.
`RADIANCE_DLOOP_SIDECAR` names a directory containing `dloop_sidecar.json` and
`dloop_sidecar.safetensors`. Version 2 requires the complete five-layer, full-context DLoop config,
exact tensor names, shapes, BF16 dtypes and manifest order. The loader rejects mismatches before serving.
`context_window=0` retains the entire verified prefix; it does not select a sliding window.
`config.use_kv_cache` is a required boolean in the current checkpoint config. The engine maintains
request-slot-owned per-layer FP32 K/V regardless of the Python implementation's caching switch.
Context keys use absolute positions. Only new verified target taps are projected and appended;
base and tree backbones read the same immutable historical cache.
Backbone weights form one dense BF16 blob in manifest order; selector codebooks are vocabulary
sharded. `RADIANCE_TEACHER_MATRICES` supplies the unfused BF16 teacher matrices from the same
file as FullDrafter. The DLoop-only loader uploads the full embedding and only this rank's LM-head
shard; neither matrix is duplicated in the student sidecar. FullDrafter's loader is unchanged.
The candidate gate names the development export `reports/dloop_engine/sidecar_full_l5/`.
That export must be generated from a validated checkpoint before a GPU gate can run; merely naming
the directory does not promote a training checkpoint or establish serving readiness.
`scripts/dloop_testrun.sh SIDECAR_DIR` is the manual first-GPU-run driver, not a build/test hook.
It holds the queue lock, refuses occupied GPUs, requires tiny and matching real pipeline parity,
checks plain/schema-constrained HTTP output and separate mine_ref oracle coverage, then runs
one warmup and one single-stream BetterBench decode pass. Reports and stage exit codes go to
`reports/dloop_engine/testrun_<timestamp>/`. A sibling `golden_*` directory is selected from the
`sidecar_*` name, or explicitly through `DLOOP_GOLDEN_DIR`. Its EXIT trap stops its managed engine
before releasing the queue lock; an oracle-enabled engine is never used for the benchmark.

The supported teacher geometry is width 2560, vocabulary 248320 (248077 active), TP2.
The student uses width 768, five layers, 12 query/4 KV heads of width 64, FFN 2304, eight
proposal slots, rank-256 selector and two-tap grouped dynamic convolution. Computed activations
and sums are FP32; BF16 is the weight and recorded-tap storage format.
`RADIANCE_DARTREE_VARIANT=fixed|pruned` defaults to `fixed`;
`RADIANCE_DARTREE_BUDGET=1..15` defaults to 15 and must match the configured speculative depth.
The root is the sampled anchor. At most 15 draft nodes plus that root fit the 16-row verifier.

Each sequence slot owns a full-prefix cache `[max_seqs,5,context_capacity,512]` in FP32.
For each layer and token, the first 256 elements are rotated K and the next 256 are V.
`context_capacity` follows the engine's declared context bound; exceeding it is an error, never a
reason to truncate or wrap history. Three teacher-tap planes concatenate layer inputs 1 and 23
and the LM-head input. All target prefill rows, including chunks with no sampled output, contribute.
`dloop_full_append` resets only on a new slot occupant and appends subsequent prefill chunks.
After verification it appends only root-first accepted-path rows, excluding rejected siblings.
Trunk prefix-cache reuse is disabled for this drafter: a trunk-only snapshot does not contain
the corresponding drafter K/V. Every newly admitted prompt therefore runs from position zero.
Other drafters retain their existing prefix-cache policy. Restoring a trunk snapshot without its
full drafter context is not a supported shortcut.

The query pass issues `dloop_full_base`, two byte-preserving candidate all-gathers, `dloop_pool`,
five exact BF16 all-reduce sums of disjoint pool rows, then `dloop_full_dartree`. The teacher
head and selector codebooks remain sharded; only candidate packets and selected pool rows cross
ranks. DARTree's split outputs use `[max_seqs,16]` token/parent arrays with an implicit anchor.
Its declared packed output uses `[max_seqs,33]`: root-inclusive count, 16 tokens, 16 parents.
The engine validates the parent array and submits `CandidateTree` through `set_draft_tree`.
This path never uses preissued drafts or device acceptance; canonical recurrent state commits
from the host's accepted path before subsequent drafts.

Single-stream tree-state verification retains the full published tree, capped by the smaller
of `NUM_SPECULATIVE_TOKENS`, the declared drafter depth and **15 draft nodes** (16 rows with
the anchor). It does not apply FullDrafter's 8-node single-stream submit ceiling. **Known
multi-sequence limit:** the scheduler still caps each sequence at four draft nodes for the
verify-row budget. This is a scheduling limit, not a restriction on DLoop's tree construction.
Non-tree-state programs, including MTP, retain their existing depth policy; the old FullDrafter
keeps its 8/4 single-/multi-sequence submit ceilings.

QSA keeps its trained sparse semantics. At publication, if
`root_pos + 1 + max_tree_depth > topk * indexer_compress_ratio + indexer_compress_ratio - 1`,
the engine submits only a greedy chain from the DARTree. Each node chooses its highest
path-score child (first child in DLoop's stable score order); a debug message marks the first
fallback for that sequence. The scheduler rejects branching past the same exactness bound and
also linearizes branches when another row forces a sparse batch route. Branching verification
therefore runs only where dense attention is exact; long contexts continue through ordinary QSA.
The engine reads this bound through the scheduler's `tree_qsa_exact_to(program)` helper, so
non-divisible indexer budgets cannot cause divergent publication and admission limits.

**Static VRAM accounting.** Let `Q=max_seqs` and `C=context_capacity`. The persistent FP32
historical cache alone requires `Q * 5 * C * 2 * 4 * 64 * 4 = 10,240 * Q * C` bytes per rank.
It is not a 32-entry ring and is not copied into base/tree scratch. Append scratch covers only
new target rows; base/tree scratch covers block/frontier computation and candidate selection.
Scratch hooks are authoritative, and the engine shares scratch regions according to its planner.
The three retained target-tap planes require `15,360 * max_tok` bytes when all prefill rows are retained.
Student/selector storage follows the validated sidecar tensor sizes. The former two-layer/ring
totals no longer apply. These are source-level sizes, not measured residency or available headroom.

The DLoop-only teacher loader allocates exactly one BF16 head shard (635,699,200 bytes)
and the full BF16 embedding (1,271,398,400 bytes) per rank: **1,907,097,600 bytes / 1,818.75 MiB**.
It allocates no peer BF16 head, Q8 weights, quantization scales or FullDrafter parameter structure.
Compared with calling the unchanged FullDrafter teacher loader, this removes
**1,272,402,392 bytes / 1,213.46 MiB per rank**. Upload uses a reusable 1 MiB host-pinned
staging buffer, released before serving, with no device staging allocation.
Total residency additionally includes student weights, selected pool rows, target taps, scratch-region
count, trunk state and allocation alignment. A full-prefix cache must fit the configured memory budget;
lowering activation/KV precision or silently shortening the prefix is not an allowed fallback.

---

## 11. Multimodal

**Pictures and video are served on every Qwen container that carries the tower** — Qwen3.8-27B
(bf16, fp8), Qwen3.6-35B-A3B and Qwen3.8-Flash-Next — through the chat API's `image_url` /
`video_url` parts (and the Responses spellings `input_image` / `input_video`), as `data:` URLs
only: fetching a caller's URL would make the server an SSRF proxy. Audio is not served.

**The processor is `core/mm/`, held to transformers number for number** (`tests/mm_test.cpp`, goldens
generated by `tests/mkgolden.py` from transformers' own functions). ffmpeg decodes, with a format
whitelist and no protocols, and converts at the frame's own colour matrix and range. `smart_resize`
fits a picture to multiples of 32 in its pixel band, and Pillow's antialiased bicubic — transcribed
bit for bit, 8-bit intermediate and all — resamples it; the patches go out normalised, in merge-block
order, a picture repeated to the temporal depth. A video keeps `fps` 2 frames a second (4 to 768),
chosen as numpy's rounded linspace, and each frame is capped the way qwen-vl-utils does it (and
transformers will from 5.22): at most `max_video_tokens` rows and the budget's even share. Its
expansion is a timestamp, `<T seconds>`, then the group's pads between vision_start and vision_end,
standing for the template's whole vision triple as the model's own processor wrote it (transformers
4.57); transformers 5.x keeps the outer pair and writes two tokens more. A media part reaches the chat
template as a `media_marker`, which the part join leaves unpadded, so an image prompt tokenises to the
reference's ids exactly.

**The encoder is the language model's plugin.** The Qwen3-VL tower (`arch/common/rad_block_vit.h`) is
declared last by `qwen35_bf16`, `qwen35_fp8`, `qwen35moe_fp8` and `qwen4exp_fp8` when the container
carries its weights (`radiance.encoder`) and the deployment asks for an encoder (`--mm-max-patches`,
16384 patches a pass by default — a 2048 × 2048 picture; a larger one is resized to fit, since a
segment is never split): replicated weights at `RAD_ACCESS_PER_REQUEST` in groups past every text
layer, one `rad_declare_encoder`, and a pass (`RadBatch::enc`) that issues the tower and nothing else.
Its output matches transformers' at bf16 parity on every checkpoint's tower.

**Encoding happens between steps, never inside one.** The scheduler holds a request back until its
items are encoded (`Scheduler::media_ready`); the engine packs whole segments — a picture, or a
video's temporal group — into passes, reads the rows back into host memory with the item, and the
batch builder copies a chunk's rows in with the chunk that reaches them (`RadBatch::mm_rows` /
`mm_embd`). Rank 0 gets the rows and every other rank zeros, and the plugin scatters them over the
token embeddings BEFORE the vocab-sharded embedding's reduction, so the sum counts each once. The
placeholders stay the model's own pad ids, which a hashed n-gram embedding (Flash-Next's PLE) reads
like any other token.

**Positions are M-RoPE.** `RadBatch::rope_pos` carries a pass's [3, T] (t, h, w) planes — the image
grid at the run's base, the next text at base + max(rows), a video's groups each a run of their own
— and `rope_mixed` says whether any token's components differ; the fused one-component rope path runs
only when none do, so a text-only step is the graph it always was. Decode rows continue at their index
plus the prompt's final shift, moved on device with the position. QSA block keys rotate at their
first token's three components (`qsa_work`'s `rope` operand, a `[work, 3]` `bpos`).

**The prefix cache keys a media run by its content** — a hash of the item's patches and the run's row
offset — never by its placeholder ids, which are the same for every picture. Two pictures never alias;
the same picture hits. Not done: a hit that covers a whole item still encodes it before admission, and
the same picture in two requests is encoded twice — both correct, both encoder time a cache of the
tower's output would save.

**Containers gain the tower in place.** `rad-convert --reuse C --in-place` writes only the new weights
past the end of `C` with fresh tables and rewrites the header last (`<C>.pre-append` keeps the old
header and length): about a gigabyte of I/O for a 120 GB Flash-Next container.

---

## 12. Text frontend

Written once, in the core, and mostly not written by us. llama.cpp is MIT, is already C++, and
has solved every one of these problems in production. The line to draw is: **take the text
frontend and the tooling, not the engine.**

**A lift and a rewrite are different things, and the table says which.** A lift is upstream's file
at upstream's path under `vendor/`, with local changes as a patch series. A rewrite is a file we
would have had to modify so heavily that it belongs in `core/` under our own name instead, with a
comment at its head naming what it was read from. `vendor/UPSTREAM` records the commit, the split,
and the reason for every row here.

| need | source | how |
|---|---|---|
| Jinja engine | `common/jinja/` | lifted, 5.0k |
| template capability probe | `common/jinja/caps.cpp` | lifted, 0.5k |
| reply format analysis | `chat.cpp`, `chat-auto-parser*.cpp`, `chat-diff-analyzer.cpp` | lifted, most of the patch series; the markers only |
| reply parsing | `chat-peg-parser.cpp`, `peg-parser.cpp` | **rewritten** → `core/text/chatparse.{h,cpp}`, one streaming pass |
| JSON-schema → grammar | `json-schema-to-grammar.cpp` | lifted, 1.2k |
| unicode and regex splitting | `common/unicode.cpp`, `src/unicode*` | lifted |
| HTTP and JSON | `vendor/cpp-httplib`, `vendor/nlohmann` | lifted, unmodified |
| GBNF grammar state machine | `src/llama-grammar.cpp` | **rewritten** → `core/sample/gbnf.{h,cpp}`, token masks in `gbnf_mask.cpp` |
| tokenizer merge machinery | `src/llama-vocab.cpp` | **rewritten** → `core/text/tokenizer.{h,cpp}` |
| GGUF reading | `ggml/src/gguf.cpp` | **rewritten** → `core/text/gguf.{h,cpp}` |
| sampling chain | `src/llama-sampler.cpp` | **rewritten** → `core/sample/host_ref.{h,cpp}`; reference only, §13 |
| `tokenizer.json` interpretation | `convert_hf_to_gguf.py` | Python, unliftable → `core/text/tokenizer_json.cpp` |
| NFC/NFKC normalisation | — | no upstream to take: llama.cpp has none |
| OAI schema translation | — | original: `core/server/oai.cpp` includes nothing of llama.cpp |
| multimodal preprocessing | — | original: `core/mm/`, held to transformers (§11); nothing of `mtmd` is vendored |

Not taken: ggml, their graph and scheduler, `llama-context` / `llama-batch`, the server's slot
model. Those are what we are replacing.

The rewrites each have the same story behind them, and it is worth one example: `llama-vocab.cpp`
is welded to `llama_model_loader`, `GGML_ASSERT` and the `llama_vocab` class, and organised around
a pre-tokeniser identified by *hashing* `tokenizer.json` against a hardcoded enum — which is the
behaviour the rest of this section exists to replace. Taking it whole would mean taking all of
that; so the BPE bigram queue and the UGM Viterbi are re-expressed with attribution, and the enum
is gone. The rule is mechanical: anything that cannot be expressed as a patch against upstream
gets rewritten, so `vendor/` stays updatable.

**A reply is read by one streaming parser, driven by a description of the model's format.** How a
model spells a turn — the reasoning tags, the call opener, the function and argument delimiters,
how a value is encoded — is plain data (`core/text/chatfmt.h`), and everything that depends on the
spelling is derived from it: the parser that reads the reply back, the lazy tool-call grammar and
the literal words that trigger it, and the control tokens the decoder must render as text.

There are two producers, in order of precedence. An architecture plugin may declare the format
through the optional `rad_arch_chat_format` export (`abi/rad_builder.h`) — a new model whose
template the analysis cannot read is supported by stating its format once. Otherwise the format
comes from llama.cpp's differential analysis of the chat template — render with and without
tools, with and without a call, diff the outputs to find the delimiters. Only the analyser's derived
markers are used; the PEG parser it would generate is not. A format neither producer can describe
is not guessed at: the model still chats, and a request that sends tools is refused, naming why.

The parser (`core/text/chatparse.h`) reads each reply exactly once, as it streams: an Aho-Corasick
automaton over the delimiters that can end the current region, skipping plain text with memchr; a
transducer over the tag structure; an incremental scanner for JSON-typed arguments. A long reply
costs its length, not its length squared, and nothing on that path is a regular expression. Where
a reply is within the format's grammar it reads exactly as the grammar says — the derived grammar
is lenient on purpose where models are: an argument the schema does not declare, a non-string
argument written as Python, a reasoning block the model never closed, prose before, between and
after calls. Where a reply leaves the grammar — a function that is not a tool, text inside a call —
the parser recovers locally and says so, and the call still reaches the agent: an unknown name
arriving as a proper error is a turn the model recovers from, and the same name echoed back as
prose is a turn it loses.

**What has been streamed is final.** A delta cannot be taken back, so the parser holds a byte only
while its meaning is undecided — a possible delimiter prefix, a trailing whitespace run that may be
trimmed, a JSON-typed value whose type its closing tag decides — and the deltas a client
concatenates are the answer the buffered path returns, by construction.

**The tokenizer is the one place llama.cpp's answer is Python.** Their vocab is GGUF-shaped and
the `tokenizer.json` → vocab mapping lives in `convert_hf_to_gguf.py`; at runtime they identify
the pre-tokenizer by *hashing tokenizer.json* against a hardcoded enum, which is why an
unrecognised model gets a default split.

`rad-convert` interprets `tokenizer.json` properly instead: the normalizer / pre-tokenizer /
decoder chain as declared — `Sequence`, `ByteLevel`, `Split` with its regex, `Metaspace`,
`Replace`, NFC/NFKC — resolved and baked into `.rad`. llama-vocab's BPE and unigram merge
machinery is lifted underneath. This is real work: the pre-tokenizer regexes need a correct
engine and ByteLevel's alphabet mapping has to be exact. It removes a failure class whose symptom
is plausible text with wrong token boundaries, which is the hardest kind to notice.

Detokenisation is incremental and streaming-safe: byte-level BPE emits partial UTF-8 sequences,
and a stop string may straddle a chunk boundary, so both are buffered until they resolve.

---

## 13. Sampling and structured output

**Sampling runs on device.** At a 151K vocab and batch 256, logits are ~77 MiB a step; moving that
across the link to run a host chain costs both the bandwidth and a synchronisation, on the step
path, at exactly the concurrency the engine exists to serve.

The consequence is stated plainly: every sampler is a kernel, not a lift. Temperature, top-k,
top-p, min-p, typical, repetition and presence penalties, DRY, XTC, seeds. llama.cpp's chain is
kept as a **host reference implementation**, which is what `rad-kbench` compares the device
samplers against — the same relationship §16 sets up for every other op.

Two things in that reference deviate from upstream on purpose, and both are load-bearing. The
**chain order** is vLLM's and this spec's — penalties, temperature, grammar mask, top-k, then
top-p / min-p / typical, then the pick — where llama.cpp puts temperature last; the difference is
not cosmetic, since top-p, min-p and typical all cut on probabilities and whether temperature ran
first decides which tokens survive. And the **RNG is stateless**: a counter-based function of
(seed, position, stream) rather than upstream's per-sampler Mersenne twister, because a kernel has
nowhere to keep 624 words of state per sequence and a stateful generator makes the answer depend
on how many draws happened earlier — which makes a comparison irreproducible the moment the batch
composition changes.

Structured output is GBNF, with JSON-schema compiled to a grammar. The grammar state machine stays
on host, where it belongs; what crosses to the device is a token bitmask per sequence, which at
151K tokens is 19 KiB — cheap enough to upload every step. A grammar backend is a small interface
so an alternative can be dropped in, but GBNF is the shipped one and there is no reason to carry
xgrammar for it: the masks are served from a plan precomputed when a grammar is compiled —
xgrammar's adaptive token-mask cache, keyed on stack suffixes — so a mask costs the same whether
its state was seen before or not, and compiling happens where the request is admitted.

---

## 14. Server

`/v1/chat/completions` (streaming and not), `/v1/completions`, `/v1/models` and `/v1/models/:id`,
a real `/v1/embeddings`, tool calls in and out, and a Prometheus `/metrics` endpoint with
vLLM-compatible metric names so existing dashboards work unchanged.

Beside those, the operational set: `/health`, `/ping`, `/load`, `/version`, `/server_info`,
`/tokenize`, `/detokenize`, `/get_tokenizer_info`, `/reset_prefix_cache`. Method spellings follow
vLLM's, including the ones that look wrong — `/tokenize` and `/detokenize` are POST because their
input is a body, `/reset_prefix_cache` is POST because it changes state, and `/ping` answers both
GET and POST because different load balancers send different ones and neither is worth a 405 on a
health probe.
There is one route table and `/server_info` answers out of it rather than out of a copy, because a
list kept beside the routes is a list that goes stale on the next route added.

An HTML dashboard is served at `/dashboard`, with `/stats`, `/graph` and `/sessions` (the idle
session tiers, §7.3) beneath it — and at `/`,
because a person holding the host and port of a running engine types exactly that, and answering
it with a 404 answers "is this thing alive" with "no".

Request cancellation propagates to the scheduler and frees blocks immediately rather than at
completion.

---

## 15. Tuning

Constraint matching finds a kernel that is *correct* for a geometry. Within one kernel there are
usually many tile configurations, and the right one is a property of the shape and the machine —
the vLLM patch set ships `moe-configs/` and `fp8-configs/` for exactly this, hand-tuned per shape.

**A kernel declares AXES, not a list of pre-named combinations.** `RadTunable` (§2.1) names one
axis — `sk`, `bm`, `npw` — its legal values ascending, and a default; `tune_valid` answers whether
a combination is legal at a given geometry, which is exactly what a product of independent lists
cannot say (libr4d's skinny GEMM needs `wv * sk * 32` inside the block bound *and* `K % (sk *
unit) == 0`, and neither is a property of `sk` alone). Hand-enumerating the interesting points of
a multi-axis space — `wv1_sk8_mb3_npw4` and its twenty-odd siblings — leaves **holes**, and a hole
costs step time silently: a ladder that skips a value runs the next tile up, so a shape with forty
rows of work gets a sixty-four-row tile. A cross product has no holes. It also keeps the tuning
*law* out of the launch adapter, where it would be C++ — ladders, caps, a hand-written picker —
that an out-of-tree author cannot write.

**The chosen values arrive as ordinary parameters.** There is no variant index: once the core has
picked a point it appends the choices to `RadArgs::p`, and the kernel reads `sk` with the same
`rad_args_geti()` it reads `M` with. One way in, for a value the caller supplied and a value the
tuner chose.

Every axis carries an explicit `deflt` — what runs on a machine with no measurement for this
instantiation, which is a fresh install, a shape nobody tuned, or a cache from another card. There
is nothing beneath it, so the loader checks that the default appears in `values` and **refuses the
plugin by name** if it does not. An untuned install is therefore fast rather than broken.

The core owns the cache and the policy: `rad-tune` walks the declared graph's actual shapes,
benchmarks each legal candidate on this machine, and writes **one file per machine** —
`$RADIANCE_HOME/tune/<machine>.tune` — that the selector reads at declare. The lookup key is
(kernel, plugin, geometry), not the op: two kernels can serve one op and they tune independently.
Ranges are already collapsed to the band's upper bound, because a bucket table row is what was
measured and so a bucket table row is what is keyed. The winning point is written as sorted
`key=value` pairs — `mb=3,npw=4,sk=8` — rather than a name, since a name would tie the cache to a
spelling the library happens to use this release: an axis the reader no longer declares is ignored
on lookup, and one it declares that the row omits falls back to that axis's default, so a cache
written before an axis existed still helps. It is UTF-8 text, because a human has to be able to
read it, diff it, delete one line of it, and check it into a repo beside a deployment.

One mechanism and one cache format, which is the point — the alternative is every plugin
inventing its own tuning tool and its own file, and the engine being unable to say why a shape is
slow. `--debug-graph` names the chosen point and where it came from: the kernel's own defaults, or
a cache row and when it was measured. The age of a measurement — taken before a driver or a
library changed underneath it — answers a large share of the questions a slow shape produces.

---

## 16. Observability

Normal operation is a few lines. Everything below is off by default.

- **`--debug-graph`** — the entire declared graph: every op, its bucket table, the kernel that
  resolved each band and the tuned point it will run at, the constraints that matched, the weights
  it touches, the buffer it writes. This is the answer to "it is unclear what even runs". It also
  carries the fusion advisory: these N declared ops are a chain some kernel says it can do in one
  launch (§2.1), and here is why it did not fire.
- **The selection table** (`--debug-selection`) — every distinct question the selector was asked,
  in the order first asked, with what it resolved to and, for a miss, the constraint that refused
  it. Repeats are folded and counted, because a 64-layer model asks the same question 64 times and
  "asked 64 times" is how much of the model resolved to that kernel. It is core-owned, like the
  selector itself, and it is the record of why a fallback is running.
- **The placement plan** — every weight's tier and site, the pool totals, and what did not fit.
- **Per-op device timing**, opt-in, with a caveat that has to travel with the numbers: enabling
  profiling restores synchronisations that change the mover's slot supply, so a profiled run is
  not comparable against an unprofiled one.
- **The live view** — per-card utilisation and VRAM, throughput, draft acceptance, the expert
  plane coloured by tier, residency per tier in units and bytes, and the interconnect split
  between the tensor-parallel all-reduce and the expert mover's promotions and demotions. Piped
  or redirected it prints ordinary lines instead.

---

## 17. Correctness and failure

**`libref` is a kernel plugin that implements the conventional op vocabulary in plain C++** —
correct, slow, generic over shape. It sits last in the hierarchy and does two jobs.

As an **oracle**, it is what `rad-kbench` runs the resolved kernels against, per op, on the real
container, and what the in-engine oracle below runs beside every device issue.

As a **fallback**, it is opt-in and the engine refuses by default. An op that resolves to libref
means the model runs orders of magnitude below the machine with nothing in the output to say so,
so startup stops with `RAD_E_NOKERNEL`, naming every op that fell through;
`--debug-accept-reference-kernels` accepts the cost deliberately and warns on every start. That is
a deliberate reversal of the obvious policy: the point of a fallback is that it is *reachable*,
not that it is automatic, and a silent slow path is the failure this section exists to prevent.

One artifact, both jobs, and the second is why the first is safe: a fallback you cannot test
against is just an untested slow path.

**Its coverage is honest rather than total, which is worth stating where a reader would assume
otherwise.** Of the 79 ops libr4d serves, 19 have no libref row (docs/KERNELS.md has the current
counts), and almost all of those are
fusions — which is not a hole so much as the shape of the thing: a fused kernel's claim is that it
equals a chain of conventional ops, and the chain is what ref implements, so `replaces` (§2.1) is
what checks it. Nor is ref constraint-free. `kv_store` and `attn_paged` carry a `block_size >= 1`
constraint, and the entry points refuse with `RAD_E_UNSUPPORTED` what they cannot represent: a
head dim above 1024, more than 8 ranks in a collective, a convolution history wider than 16. A
reference that quietly computed something else at those sizes would be worse than one that stops.

**The fallback guarantee has a boundary and it is worth naming.** Op names are free strings
(§2.3), so `libref` cannot implement an op it has never heard of. Therefore **ref's op list
is the conventional vocabulary**: an architecture plugin that stays inside it always has an
implementation available — behind the opt-in above — and one that invents an op has nothing to
fall back to and no oracle either. Free strings stay free; the guarantee is scoped, not silently
absent.

**A second oracle runs inside the engine, on the engine's own shapes, and it is the stronger of
the two.** `rad-kbench` falsifies one op at a time against operands *it* invented: dense, rank-2,
freshly filled. The engine's operands are row slices of a `max_tok` buffer, column slices of a
fused projection, rank-3 views a scan kernel required, and weights that came off a file. That gap
is structural rather than a gap in coverage — an op can be correct at every shape `rad-kbench`
knows and wrong at the one shape the model issues, and then the build passes nearly everything
while the engine answers the same token to every prompt.

So `RADIANCE_OP_ORACLE` names ops, and for each matching issue the runtime copies every operand to
the host before the launch, lets the device kernel run, copies the results back, and runs
**libref's host row for the same (op, band)** on the untouched copy — same shapes, same strides,
same parameters, only the pointers differ. That host row costs nothing to obtain: it is already
sitting in the bucket table beside the device plan, resolved by the same selector and never used
because the planner put the op on the device (§2.1's "declare resolves both domains" is what pays
for this). What it costs to *run* is everything — two device-to-host copies of every operand, a
host synchronisation per issue, a step that takes minutes — so it is a debugging instrument and it
is off unless asked.

`RADIANCE_OP_REPEAT` asks a different question with the same machinery: not "does this kernel
agree with libref" but "does this kernel agree with **itself**". It snapshots the operands, runs
the kernel, restores them exactly, runs the same kernel again, and byte-compares. That is the one
fault class every other instrument here is structurally blind to — an unordered reduction, a
workgroup reading what another is writing, a workspace assumed zeroed — because they all run an op
once per side and so pass at exactly the rate the kernel happens to be right. Byte-exact, no
tolerance: the same kernel on the same bytes has no licence to move, and one mantissa bit is a
different token a few hundred greedy steps later.

This section is small and it is the most important one in the document. A from-scratch engine
whose kernels cannot be individually falsified is a project that ends with fluent wrong output and
no way to bisect it: every component tests clean, the whole emits degenerate text, and the cause
is never found. Refusing to serve a configuration beats shipping one whose output is fluent and
wrong — but per-op oracles are how you avoid having to refuse.

**Failure model.** A kernel returning negative at issue is a bug in selection, not a runtime
condition, and it aborts the step and fails the requests in it with the kernel and geometry named
— it does not silently fall back, because a silent fallback is how you ship a slow path nobody
knows about. KV pool exhaustion preempts by recompute, and if the smallest running request cannot
fit it is failed rather than the engine deadlocking. A device fault is not recoverable in-process;
the engine exits non-zero rather than serving from a wedged queue.

**Determinism.** Dynamic placement sets which experts are resident, and residency sets the
summation order, so identical inputs do not give identical outputs. `--deterministic` pins
**placement**: the plan is static, the heat engine does not run, and nothing moves underneath a
comparison. It is the mode anything being compared across builds has to run in.

**It does not buy bit-exactness, and the remainder is named rather than quietly dropped.** The
fp8 GEMM's split-K finisher sums its planes in index order, so a run repeats exactly — but the
split *count* is chosen from `M`, so the same row computed at `M = 1` and at `M = 8` is not
bit-identical. That is why a speculative decode diverges from a non-speculative one on a near-tie,
and why the same request at a different batch position can take a different reduction order.
Nothing pins it; §19.7 is where that sits.

---

## 18. Build and layout

```
radiance/
  core/            engine, scheduler, placement, sampler, server
  abi/             the C ABI -- the only thing plugins compile against
  arch/            architecture plugins, one directory each
  libr4d/          the RDNA4 (gfx12xx) kernel library -- ships with the engine
  libref/          the conventional vocabulary in plain C++: the oracle, an opt-in last resort
  libavx/          the same vocabulary in x86 SIMD, dispatched by CPU feature: the production host
                   kernels, built on every x86-64 build
  libquant/        the built-in quantisers (§4.5), host C++, a plugin like any other
  tools/           rad-convert, rad-kbench, rad-tune, rad-info, rad-schemas, rad-tokcheck
  tests/           the suite; runs in full on the host backend
  scripts/         shell harnesses for serving, tracing and profiling a running engine
  data/            the conversion recipes, and recorded kernel references rad-kbench replays with
                   no model present
  docs/            the user guide, the tools reference, the op vocabulary, the plugin contract,
                   the kernel report
  cmake/           the find_package config and rad_add_plugin, shared with out-of-tree plugins
  vendor/          llama.cpp lifts, tracked to an upstream commit, patched not edited
```

CMake, C++20, hipcc for anything device-side — and `RAD_WITH_HIP` (`ON | OFF | AUTO`) decides
whether there is any device side at all. It is a cache string rather than an `option()` for a
reason worth knowing: `option()` is boolean-only, so a default of `AUTO` collapses to `OFF` and
the detection never runs, which is how a build on a machine with two cards silently produces the
host backend. `RAD_GPU_TARGETS` is the set of offload architectures device kernels are compiled
for; left unset it is detected from the cards present, and with a toolchain but no card the device
build is declined by name rather than guessed at, because emitting kernels for hardware nobody
named is worse than saying so.
The top-level build holds no opinion about which architectures are acceptable: a kernel library
declares the ones it serves and removes itself from a build targeting anything else. So a card no
library covers is a configure that succeeds and an engine that refuses to serve (§17), not a
project that will not build. The core's own device code — the copies and fills of the device layer,
the step's bookkeeping, the weight-table patches — is built for every AMD GPU family's generic target
besides, so the engine starts on a card its build machine never saw, and serves it with whatever
kernel library is installed for it. A library whose code objects hold nothing for the card is left
out of that card's selection at load rather than failing at its first launch.

`$RADIANCE_HOME` defaults to `share/radiance` under the install prefix, and is a search path of homes
separated by `:`, the first winning; `architectures/`, `kernels/` and `quantizers/` under each are
scanned at startup as one set. Kernel hierarchy is configured, not discovered — an ordered list
(`--kernels`, or `RADIANCE_KERNELS`). A library named there takes that position, one present but
unnamed loads after every named one, and a library that declares itself a reference implementation
(`rad_plugin_reference`) sits below every unnamed one, because it matches every op and would
otherwise shadow whatever ranked behind it. So adding a kernel library is dropping an `.so` in a
home, and substituting one for another is naming it ahead of the other.

The plugins that ship are plugins like any other. Each directory builds inside the tree or on its
own against an installed radiance through `find_package(radiance)` and the same `rad_add_plugin`, and
nothing in the core names one: `ctest -L oot` installs the build, rebuilds every shipped plugin out
of tree against it, and requires the same plugins, schemas and registration back.

---

## 19. Open questions

**The numbering here is stable and answered questions keep their number.** Several of these are
cited from the tree by number, and a citation that silently comes to point at a different question
is worse than a list with answers in it.

1. **Linear checkpoint retention beyond the tip. ANSWERED IN CODE, NOT IN MEASUREMENT.** Geometric
   backoff ships and is the default (§7.3), with `tip-only` and `keep-all` beside it — as policy
   objects rather than an if-statement, which is the shape the answer has to keep until there is a
   workload that says what the right curve is.
2. **`RadWeightDecl.access` for `lm_head` and embeddings. ANSWERED:** they have their own class,
   `RAD_ACCESS_VOCAB` (§5.2).
3. **Zero-copy site selection.** Still open, and the shipped criterion is not the one this asks
   for. The planner cuts over on the **declared access class** — `CONDITIONAL` streams over the
   link out of the pinned pool, `PER_TOKEN` with no host kernel stages into a slab slot — rather
   than on a measured read-count cutover. "Read once per step" remains the principle; where it
   sits against staging is a measurement nobody has taken, and it will differ between prefill and
   decode, which the single plan the planner produces (§5.3) does not distinguish at all.
4. **Draft model placement. ANSWERED BY CONSTRUCTION:** there is no `vram`/`ram` choice to make.
   The drafter is part of the model's source — merged into the container at convert time, or
   shipped inside the checkpoint as an MTP head (§10) — and the planner places its weights like any
   other weight, against the same budgets.
5. **Prefix-cache and weight tier contention on one SSD.** Still open, and explicitly so. Both
   want `O_DIRECT` bandwidth and under load they fight. `TokenBucket` bounds the fight with one
   shared byte budget and per-consumer counters, and it is labelled a placeholder and off by
   default — an unmeasured rate limit throttling a machine nobody profiled is worse than none. The
   actual question is who should *win*: a weight-tier read blocks a layer that is about to
   execute, a prefix-cache read only makes a prefill shorter, so a priority scheme is probably the
   right shape of answer and a flat rate limit certainly is not.
6. **Whether `RadTensor` needs a dtype the core understands. ANSWERED: it does** — `RAD_F32`
   through `RAD_I64`, parsed by `rad_dtype_parse`, with `rad_load_f32` / `rad_store_f32` switching
   on it. `libref` has to interpret a tensor to compute on it, `rad-kbench` has to compare two of
   them elementwise, the buffer planner has to size a buffer from its shape, and `rad-convert` has
   to write bytes; an opaque dtype pushes all four back into the plugin, which is the opposite of
   the point. The enum stays open at the top — values at or above `RAD_DT_PLUGIN_BASE` are
   plugin-private, sized by the plugin's own layout hook and never interpreted — but only for what
   a kernel STORES. A container holds core dtypes and nothing else (§4.1), so a format invented for
   one kernel is an encoding plus a relayout, and the file stays readable without that kernel.
7. **Bit-exact reproducibility is not achieved.** `--deterministic` pins placement; it does not
   pin the fp8 GEMM's split-K count, which is chosen from `M`, so the same request at a different
   batch position can take a different reduction order (§17). Whether that should become a pinned
   kernel axis — §15's mechanism already has the shape for one — or an accepted property of a
   batched engine, is open. Pinning it is not free: the count varies with `M` because that is
   where the speed is, and fixing it costs the narrow shapes their tuning.

8. **`RadTransformFn` cannot see a source tensor's companion planes. ANSWERED by §4:** the hook
   is gone, and `relayout` is handed every plane the declared weight takes. What follows is the
   question as it stood. It is handed one `src`, so
   a kernel whose stored layout has to be derived from a quantised weight AND its scale grid --
   the two arrive as separate checkpoint tensors -- cannot compute it: the transform holds the
   codes and not the scales. Every layout in the tree today is derivable from one plane, so
   nothing is blocked, but the signature is the reason a future one could not be. The shape of the
   answer is a `const RadTensor* const* src_aux` beside `src`, which is an ABI break and therefore
   wants to ride with the next one rather than on its own.
