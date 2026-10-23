# radiance engine specification

The engine is C++: a monolithic core with `.so` plugins for the two things that change per model,
the architecture and the kernels. Python appears only in a few offline helpers
(`tools/pad_tp.py`, `tests/mkgolden.py`); neither the engine nor its tools run it.

Device code belongs to plugins, not to the core. `RAD_WITH_HIP` is `ON | OFF | AUTO`, and a
host-only build with no device kernels is a supported configuration that exercises everything
except the device kernels. Three kernel libraries are in the tree and only one is HIP: `libr4d` for
AMD RDNA4, `libavx` in hand-written x86 SIMD, and `libref` in plain C++.

## Why

vLLM's core design (continuous batching, paged KV, prefix caching) is the model this engine
follows. Its implementation is dynamic throughout: nothing states what it will run before it runs,
its extension points are many and loosely specified, and Python costs host time per step, per
layer and per KV-cache group.

Serving four models on one card with vLLM took twenty-four monkey-patches. Grouped by what they
change, they are this engine's requirements:

| patch class | count | requirement here |
|---|---|---|
| kernel substitution | 7 | kernel plugin hierarchy (§2.1) |
| fusion coverage | 1 | fused ops are ops that selection can choose (§2.3) |
| per-step host cost and metadata sharing | 4 | core-owned step batch, built once (§8) |
| KV group sizing for hybrid models | 1 | hybrid KV groups are first-class (§7.2) |
| speculative decode plumbing | 5 | a drafter is declared by the model's own plugin (§10) |
| chat template and tool parsing | 2 | lifted and rewritten from llama.cpp (§12) |
| platform detection | 2 | not applicable |

Each class exists because no extension point reached the code that needed changing, so the only
option was to patch the source. Here each has a designed extension point.

## Scope

**In:** continuous batching, paged KV, prefix caching, tensor parallelism, speculative decoding,
hybrid KV (full, sliding-window and linear-attention state in one model), weight placement across
VRAM, RAM and SSD with runtime re-placement, a server for OpenAI, Anthropic and llama-server
clients, embeddings, image generation, a quantiser, and PEFT LoRA adapters served beside the base
model and picked per request (`docs/GUIDE.md` §4.9.1).

**Multimodal (§11).** Pictures and video on the Qwen and Gemma 4 plugins, pictures on LFM2-VL,
and pictures, video and audio as EmbeddingGemma 2 input. The encoder is declared by the language
model's own architecture plugin, its output travels in the step batch, and a picture is a
prefix-cache hit by content.

**Large MoE models.** Heat-driven expert placement (§5.4) and the 4-bit expert formats
(`docs/MOE-W4.md`) serve Qwen3.8-Flash-Next. DeepSeek-V4-Flash is not ported.

**Out:** training, pipeline parallelism, CUDA, disaggregated prefill. None is designed out and
none is designed for.

**Hardware.** Everything above the device layer learns the machine at run time (architecture,
compute units, wave width, LDS, VRAM). Everything that must be written per card is a kernel
library, a plugin selected per op against declared constraints (§2.1). The shipped device
library, `libr4d`, covers AMD RDNA4 (`gfx12xx`), the hardware the engine is measured on. A card no
installed library covers is refused at startup with the unserved ops named (§17).

---

## 1. Shape of the engine

One process. One thread per tensor-parallel rank, each owning one device, so ranks need no IPC
and no serialisation. HTTP threads feed a request queue; one scheduler thread produces one
`RadBatch` per step (§8) and releases the rank threads against it at a barrier. The ranks share an
address space, so the batch is shared by pointer.

Startup, in order:

1. Open the devices.
2. Read the model's metadata: architecture id, dimensions, quantisation descriptor, vocabulary.
3. Load the kernel plugins from `$RADIANCE_HOME/kernels/` in hierarchy order, and the architecture
   plugin for the model's id from `$RADIANCE_HOME/architectures/`. No match, no model.
4. **Declare.** The architecture plugin enumerates every weight, buffer and op it will issue, and
   the core resolves each op against the kernel hierarchy. This phase always completes and
   reports every unresolved op together.
5. **Plan.** The placement planner assigns every weight a tier and an execution site against
   explicit pool budgets.
6. **Load.** Weights are read and staged according to the plan.
7. Serve.

After step 4 the whole model is known statically: every op, the kernel that serves it, every
weight it touches, and the order. The rest of this document relies on that.

---

## 2. Plugins

Three kinds, loaded from `$RADIANCE_HOME`: kernels (§2.1), architectures (§2.4) and quantisers
(§4.5). All are C ABI only: C++ across an `.so` boundary invites mismatches that surface as
corruption far from their cause, and this boundary is where users substitute their own builds.

Every plugin exports a version symbol and is refused on mismatch:

```c
uint32_t rad_plugin_abi_version(void);   /* must equal RAD_ABI_VERSION */
const RadPluginInfo* rad_plugin_info(void);
```

### 2.1 Kernel plugins

`$RADIANCE_HOME/kernels/*.so`. Each provides any number of kernels as **rows**, exported through
`rad_kernel_count` / `rad_kernel_at`, alongside the op schemas (§2.3) exported the same way.

A plugin does not supply a selector. The constraint selector is core-owned
(`core/plugin/select.cpp`), so "which kernel ran, and why" has one answer and the hierarchy below
is enforced above every plugin. A plugin declares what it can do; the core decides what runs.

```c
enum { RAD_C_EQ, RAD_C_LE, RAD_C_GE, RAD_C_DIV, RAD_C_IN };

struct RadConstraint {
    const char* key;    /* parameter name, e.g. "head_dim" */
    int         op;
    long long   ival;   /* EQ / LE / GE / DIV */
    const char* sval;   /* IN: space-separated set */
};

enum { RAD_DOMAIN_DEVICE, RAD_DOMAIN_HOST };

/* One axis of this kernel's instantiation space (§15). Null/0 means nothing to tune. */
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
    RadLayoutFn    layout;     /* what this kernel stores for a weight of a given encoding (§4.3) */
    RadRelayoutFn  relayout;   /* host code producing it from the canonical planes, at load */
    RadShapeFn     opd_shape;  /* how to size and fill each operand, for a tool calling it cold */
    RadUnrelayoutFn unrelayout;/* the inverse, so a stored weight can be read back */
    RadFuseFn      replaces;   /* the chain of conventional ops this kernel does in one launch */
};
```

`init` and `fini` hold per-instance state set up once, off the hot path: libr4d's all-reduce
exchanges peer signal buffers and IPC handles there; a kernel may JIT or load a config table.

The hooks after `scratch` serve tools as well as the step. All are optional; `launch` is not.

- `layout` and `relayout` turn a weight's canonical planes into the bytes the kernel reads (§4.3);
  `unrelayout` turns them back, so a stored weight can be checked against a dense oracle. None of
  them quantises: a kernel refuses in `layout` the encodings it does not read (§4.5).
- `opd_shape` gives each operand's size and value domain, so a tool can call the kernel cold. A
  kernel without it is skipped by name, never called with guessed extents.
- `replaces` declares the chain of conventional ops a fusion computes, so "byte-identical to the
  ops it replaces" is checkable (§2.3, §17).

A kernel declares tunable axes, not variants; the core appends the chosen point to `RadArgs::p` as
ordinary parameters (§15).

**Constraints are necessary, not sufficient.** They decide whether a kernel exists for a geometry.
Strides, contiguity, alignment and buffer sizes belong to a call, not a model; the entry point
checks them and rejects what it cannot run.

**Selection.** Plugins are queried in hierarchy order, and the hierarchy is absolute: the first
plugin with any matching kernel supplies it, whatever a lower plugin offers. That is what makes a
user override reliable.

Within one plugin the highest `priority` among matching rows wins, ties broken by declaration
order. Specificity is declared, not derived: counting matched constraints would rank
`{head_dim<=1024, gqa>=1, block%16==0, causal in {0,1}}` (four) above the more specialised
`{head_dim==256, gqa==6}` (two). libr4d's two-shot all-reduce rows must outrank its one-shot rows
or nothing could reach them.

**Domains.** A CPU kernel is an ordinary kernel with `domain = RAD_DOMAIN_HOST`. Declare resolves
both domains where both exist, so the core can change execution site at runtime without
re-resolving. Misses are reported by class:

- a device miss with no host kernel to stand in is a hole, and the engine refuses to start;
- a device miss a host kernel covers runs that op on the host, with its weights in the
  container's mapping (§5.1); an op with no weights has nothing to place, so its host kernel
  cannot stand in;
- a host miss only removes a placement option, and is reported on one line.

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

`instance` returns `init`'s state on every launch without a lookup. `rank` and `world_size` are
fields rather than parameters because only collectives need them, and as parameters every
architecture plugin would have to write them.

### 2.2 Shape domains and the bucket table

A parameter may be declared as a **range**, because some parameters belong to the batch, not the
model. A GEMM's `M` is one row at decode, up to sixty-four at a speculative verify and thousands in
a prefill chunk, and libr4d has a kernel for each band; resolving once at `M = max_tok` would run
the prefill kernel at `M = 1`.

```c
h = decl_op(b, "gemm_nt_q",
            RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("N", 8192), RAD_INT("K", 5120),
                       RAD_INT("group", 128), RAD_STR("dtype", "w4a8")));
```

The core resolves the range into a **bucket table**: one resolution per band, built at declare and
indexed by the actual value at issue. The band boundaries are the union of the `LE`, `GE` and `EQ`
values the candidate kernels place on the parameter: kernels constraining `M<=16`, `M<=64` and
nothing give `(0,16]`, `(16,64]` and `(64,max]`.

`--debug-graph` prints the table:

```
gemm_nt_q  N=8192 K=5120 group=128 dtype=w4a8
    M <=   16  ->  gemm_w4a8_nt_m64        (libr4d, prio 20)
    M <=   64  ->  gemm_w4a8_nt_m64        (libr4d, prio 20)
    M <= 8192  ->  gemm_w4a8_prefill       (libr4d, prio 10)
```

The missing-kernel report is per band: a range with a hole (a prefill shape nothing serves) is
reported as that band, not as the whole op.

### 2.3 The op vocabulary

Op names are free strings. Kernel plugins declare the ops they implement and a schema for each, a
`RadOpSchema` table exported through `rad_kernel_schema_count` / `rad_kernel_schema_at`, and the
core checks an architecture plugin's declared parameters against the union of schemas. An excerpt:

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

The core needs only the operand roles: which operands are weights (issue resolves a handle and
waits on the mover's event) and which are written (the buffer planner computes liveness). Knowing
nothing else is what lets a third party add an op without a core change. A misspelled op fails
declare with the near miss named.

**Requiredness.** A parameter is supplied by the caller (`RAD_REQUIRED`), may be supplied
(`RAD_OPTIONAL`), or is supplied by the kernel (`RAD_DERIVED`). The third breaks a circularity:
the paged block size comes from the resolved attention kernel (§7.2), but a constraint on a key
the geometry lacks does not hold, so a kernel constrained to `block_size == 16` could never be
selected. Constraints on a derived parameter are skipped during matching, and the winner's
`RAD_C_EQ` value is written into the resolved geometry. A caller that supplies the parameter pins
it and it matches normally, so a forced block size is refused by name, not silently replaced.

**Schemas are fixed by first declaration.** The first plugin in hierarchy order to declare an op
fixes its schema; a later plugin declaring it differently is refused at load, both named.
Arguments are positional, so disagreeing schemas would give one call different meanings depending
on which plugin won selection.

**Fusion is a selection, not a compiler pass.** A plugin asks for the fused op first and emits the
unfused sequence when it does not resolve:

```c
rad_op h = decl_op(b, "rmsnorm_had_quant_i8", {{"n", d_model}, {"dtype", "bf16"}}, ...);
if (!h) {
    h_norm = decl_op(b, "rmsnorm",           ...);
    h_hq   = decl_op(b, "had_quant_act_i8",  ...);
}
```

`rad_op_resolved` is the shipped form of that test: an unresolved handle is `RAD_NULL_HANDLE`,
an error only if issued. `rad_op_declared` asks whether any loaded library declares an op, for a
fused op only an optional library defines. There is no pattern matcher; the plugin author decides.

A kernel may also declare the chain it replaces (`RadFuseFn`, §2.1). That lets a fused kernel be
checked against the same plugin's unfused rows, byte for byte, and feeds `--debug-graph`'s fusion
advisory. Nothing uses it to rewrite the graph, and declare keeps nothing a rewriter could use: the
buffer planner reduces each buffer's uses to a `first_def`/`last_use` pair, so "op A's output is op
B's only input" is not recoverable.

### 2.4 Architecture plugins

`$RADIANCE_HOME/architectures/*.so`. One plugin implements one architecture id for one
quantisation scheme, and the selection key is the pair: `qwen35_bf16.so` and `qwen35_fp8.so` both
answer `"qwen35"` from `rad_arch_id()` and are told apart by `rad_arch_quant()`, matched against the
model's quantisation descriptor. Two plugins claiming one pair are refused at load, both named. The
match is exact (unquantised is `""` on both sides), with no fallback: a dense plugin given a
quantised model would fail later at the name map, with a worse message than *"no plugin claims
qwen35 at fp8_e4m3; qwen35 is served at: (unquantised)"*.

The graph differs by quantisation (a w4a8 model wants `rmsnorm_had_quant_i8` where a bf16 model
wants `rmsnorm`), and one plugin per quantisation keeps the declare function free of branches on
it, so it reads as a list of what the model does, in order.

Dimensions come from metadata, so one plugin covers a family: `qwen35_fp8.so` serves Qwen3.5, 3.6
and 3.8 dense models. Header-only composable blocks in `arch/common/` (attention, MoE, GDN and
other layers) keep plugins from copying each other. Two `.so`s can share a source: `qwen35moe_fp8`
is `qwen35_fp8` with a routed feed-forward, three macros and an include. They stay two plugins
because the id is the selection key and a container reports `qwen35moe` or `qwen35`.

The plugin owns the map from checkpoint tensor names to declared weight names, fusions included
(q/k/v into one qkv, gate and up into gate_up, one expert out of a stacked tensor). Both readers of
a checkpoint use it: `rad-convert` (§4.4) and the engine loading a checkpoint directly (§4.3).

**A plugin asks what a weight holds.** `rad_weight_encoding` returns the logical weight's encoding
(§4.1), from the container entry, the checkpoint tensors the name map resolves, or, in
`rad-convert`, the recipe. The plugin declares the ops that read it (fp8 codes and scale plane to
`gemm_nt_q`, int4 rotated experts behind `had_quant_act_fp8`, bf16 to the bf16 GEMM) and refuses by
name an encoding it has no path for. No container key exists only to say how a weight was packed.

Each plugin exports these; `rad_arch_quant`, `rad_arch_probe` and `rad_arch_chat_format` are
optional:

```c
const char* rad_arch_id(void);
const char* rad_arch_quant(void);     /* the other half of the selection key; "" if unquantised */
int  rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out);
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta);       /* §12 */
int  rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
void rad_arch_step(RadCtx* c, const RadBatch* batch);
```

`rad_arch_probe` reports the model properties that are inputs to declaration. The main one is the
draft depth: `RadArchProbe{draft_depth, draft_depth_fixed}` resolves `--num-speculative-tokens
auto` (§10) before declare, because `RadBuildCtx::max_spec` sizes buffers, bands kernels and sets
the scheduler's window; otherwise the core would read the drafter's configuration keys itself. The
probe also reports sizing-declare support (`shape_probe_ok`, §3.1), the encoder's merge, the audio
encoder's longest clip and an image generator's pass sizes. Without the export every answer is
zero, so `auto` resolves to 0.

`RadBuildCtx` carries `rank` and `world_size` (declare runs per rank with dimensions already
divided, §9); the deployment's bounds `max_tok`, `max_seqs`, `max_ctx`, `max_spec`,
`max_enc_patches` and `max_audio_frames`, so buffers are sized for the worst step; the
tensor-parallel wire choice `tp_wire_lossy` and its `tp_wire_min_bytes` cutover (§9); `kv_dtype`;
`shape_probe` (§3.1); and `is_draft` / `scope`, which say whether this builder scope is the model,
a drafter or an encoder.

---

## 3. Two-phase construction

### 3.1 Declare

The plugin states everything it will do. Weights first:

```c
RadWeightDecl wd = {};
wd.dtype  = RAD_I4;                     /* the logical dtype; the kernel owns the stored layout */
wd.rank   = 2;
wd.shape[0] = 8192; wd.shape[1] = 5120;
wd.access = RAD_ACCESS_CONDITIONAL;     /* see §5 */
wd.group  = rad_group_expert(12, e);
wd.shard  = RAD_SHARD_ROW;
rad_weight w = rad_decl_weight(b, "blk.12.ffn_gate.weight", &wd);
```

Activations are declared too, so the buffer plan has something to analyse:

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

Buffer operands are a separate call, since liveness needs which ops read and write which buffers;
a transient with no declared use is assumed live for the whole program.

```c
rad_op_reads (b, o, RAD_BUFS(tmp));
rad_op_writes(b, o, RAD_BUFS(out));
```

`abi/rad_builder.h` is the authority for these names; `arch/common/rad_arch.h` wraps them in the
`decl_w` / `decl_b` / `rw` helpers the in-tree plugins use. `rank` is a separate field, not derived
from `shape`.

Declare produces:

- **the resolved op list**: every op with the kernels that serve it, per band, per domain;
- **the missing list**: every op and band that resolved to nothing, reported together;
- **the weight manifest**: every weight with bytes, alignment, access class and grouping keys;
- **the buffer plan**: activation liveness over the op list, so scratch and intermediates come
  from one arena sized once instead of an allocator called per step;
- **the tuning requests**: every (kernel, shape) pair `rad-tune` would benchmark (§15).

The arena is sized for `max_tok` and liveness is computed over the declared list, a superset of
what any step issues: a plan correct for the worst step is correct for every step, with no
allocator on the hot path. A plugin that sets `RadArchProbe::shape_probe_ok` is declared again at
smaller `max_tok` values (`RadBuildCtx::shape_probe`). Each result is an arena level packing the
same buffers for smaller steps; while steps are that small, the arena above the level is lent to
the expert slab.

### 3.2 Run

```c
void rad_arch_step(RadCtx* c, const RadBatch* batch) {
    RAD_ISSUE(c, h_norm, RAD_B(x), RAD_W(w_norm), RAD_B(tmp));
    RAD_ISSUE(c, h_gemm, RAD_B(tmp), RAD_W(w), RAD_B(y));   /* core indexes the bucket table */
    RAD_ISSUE(c, h_moe,  RAD_B(y), RAD_WTAB(w_expert, n_expert), RAD_B(out));
}
```

The call underneath is `int rad_issue(RadCtx*, rad_op, const RadOperand*, int n_opd, int64_t n)`.
`RAD_OPD` packs the operands; `RAD_ISSUE` passes `RAD_N_BATCH` for the ranged parameter and
`RAD_ISSUE_N` names a value. Operands are positional per the schema, which carries their roles.
`RAD_B` is a buffer, `RAD_W` a weight, `RAD_WTAB` a run of weights.

The last line is not a host loop over routed experts: routing is resolved on device against the
residency table (§5.5), so the issue names the whole expert table and the kernel indexes it. A host
loop would put a device round-trip in the middle of the step.

The step function is ordinary C++ with one rule: it may only issue handles obtained during
declare.

**Dispatch.** With a device present, the device backend (`core/device/aql.cpp`) owns its queues
and writes each kernel launch as an AQL packet: about 6 ns of host time against 0.5-1.3 us for
`hipLaunchKernel`. At about 1600 dispatches per rank per decode step, that is about 10 us a step
instead of 1-2 ms. A decode pass repeats step after step, so the runtime records its packets once
and replays them (recorded passes, `core/runtime/ctx.h`). The key covers every batch field and
runtime state the pass's shape or arguments depend on. A key is recorded on its second sighting
(first, for decode), recorded again and compared, and only then played; every 1024th play is
checked against a fresh issue, and a difference fails the step naming the first differing
dispatch. Per-op profiling and the in-engine
oracle (§16, §17) turn recording off, since they must see every issue.

**Handles, not pointers.** `rad_issue` takes weight handles. The core resolves each to a pointer at
issue time, waits on the mover's event if the weight is in flight, and dispatches to the device or
host implementation by the current execution site. With a `void*`, a weight's location would be
fixed at declare and placement (§5) could not be a core feature.

---

## 4. Model format

### 4.1 Encodings and layouts

Two things decide the bytes a kernel reads, and they change for different reasons:

- An **encoding** is what the numbers are: the code type, how scales and zero points are grouped,
  and whether the weight was rotated before quantisation. It belongs to the model; choosing one is
  expensive and lossy (GPTQ on Qwen3.8-Flash-Next's experts needs calibration Hessians).
- A **layout** is how a kernel wants the numbers arranged: WMMA fragment order, scales in each
  row's tail, nibbles permuted for an unpack. It belongs to the kernel and is a cheap, lossless
  rearrangement.

The container stores encodings only, in one canonical arrangement, and the kernel library
rearranges them at load (§4.3). A library may change every layout without touching a container;
only a change in the numbers requires converting again. The kernel-specific part is small: in a
Qwen3.8-Flash-Next container with int4 experts, 94% of the bytes (the experts and the fp8 n-gram
table) are canonical as stored and about 3 GiB are kernel-permuted, while 82% of Qwen3.8-27B-FP8's
bytes are fp8 the checkpoint holds row-major and its kernels read fragment-ordered.

**An encoding is data, not an enum value** (`abi/rad_encoding.h`), so a quantiser can define one
without a core change:

```c
typedef struct RadEncPlane {
    char     role[RAD_ENC_STR];   /* "codes", "scale", "scale.1", "zero", "min", "grid", "t.perm" */
    uint32_t dtype;               /* a core dtype -- never a plugin-private one */
    uint32_t kind;                /* RAD_PLANE_TILED | RAD_PLANE_TABLE */
    int64_t  block[2];            /* TILED: logical rows x cols one element covers; 0 = whole */
    int64_t  extent[2];           /* TABLE: the plane's own rows x cols */
} RadEncPlane;

typedef struct RadEncoding {
    char        scheme[RAD_ENC_STR];      /* "plain" | "affine" | a quantiser's own (§4.5) */
    char        transform[RAD_ENC_STR];   /* "" | "fwhtN" | "perm" | "givens" */
    int32_t     n_planes;
    int32_t     reserved;
    RadEncPlane plane[RAD_ENC_MAX_PLANES];   /* 8 */
} RadEncoding;
```

Every plane is a dense row-major array. A TILED plane covers the weight seen as `[rows, cols]`
(every leading axis a row, the last the column): a block of `[1 x 32]` is a per-group scale,
`[0 x 0]` one per tensor and `[0 x 1]` one per column. A TABLE plane does not cover the weight: a
codebook, a lattice grid or a transform's parameters, carried whole by every slice. Elements
narrower than a byte or not a power of two wide are a bit stream along the row, lowest bits first,
each row starting on a byte. The dtypes are f32, bf16, f16, fp8 e4m3 and e5m2, fp6 e2m3 and e3m2,
fp4 e2m1, e8m0, signed and unsigned integers of every width from one bit to eight plus 16 and 32,
and 9- to 12-bit unsigned for lattice indices.

`plain` is a single `codes` plane whose value is the code. `affine` is the scheme the core decodes,
and it covers every format in common use:

```
q = code                      one code a value, or one a V-vector of a `grid` (TABLE [entries x V])
q = table[code]               a scalar codebook (TABLE [1 x entries])
q = -q where `signs` is set   one bit a value
q = q - zero                  an offset in the code's domain
v = q * scale * scale.1 ...   every scale level, multiplied
v = v - min * min.1 ...       a minimum in the value's domain
w = v . T^-1                  the transform, per row (a kernel transforms its activation instead)
```

`fwhtN` is the block-diagonal unnormalised Hadamard with its 1/N in the stored scale
(`docs/MOE-W4.md`), `perm` a column permutation, and `givens` stages of disjoint pair rotations
under a channel scale, their tables beside the weight. Each encoding has a canonical spelling,
which people, logs and `rad-info` read; kernels read the struct:

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

A GGUF file's K- and I-quants are therefore a lossless rearrangement into planes; serving one needs
only a kernel that reads it.

The core decodes every `plain` and `affine` encoding to f32 (for the oracle, `rad-convert`'s error
report, and requantising). A quantiser whose scheme the core cannot decode supplies the decoder
(§4.5).

### 4.2 `.rad`

The engine loads `.rad` containers, safetensors checkpoints and GGUF files (§4.3). `.rad` is the
format built for the load path (`abi/rad_format.h`):

- **Canonical, not kernel-shaped.** An entry is a logical weight holding the planes of its
  encoding, row-major. No kernel library is named in the file.
- **Grouped by movement unit.** An expert's entries are adjacent, because the unit of transfer is
  an expert, not a tensor. Addressing within a layer is `base + id * stride`, with no table walk on
  the critical path; the reader re-derives the stride at open and refuses a file where it does not
  hold.
- **Aligned.** Pool 2 MiB (huge-page mappable), movement unit 4 KiB (`O_DIRECT`'s alignment, so
  the file tier reads a unit with no bounce buffer), planes 256 B.
- **Self-describing.** Each entry carries its encoding, its logical shape and the quantiser and
  options that made it, so `rad-info` can say what a file holds and `--reuse` can tell which
  weights another conversion would write identically.

Little-endian only. The container also carries the checkpoint's configuration as metadata, the
vocabulary (§12), the recipe it was converted with, and the expert-popularity profile the
placement planner starts from. There is one format version; a file in any other is refused with
`RAD_E_FORMAT` and an instruction to convert again.

### 4.3 Loading

The path is the same whatever the source:

1. **Declare** (§3.1) names each weight and the planes of a logical weight it takes
   (`RadWeightDecl::source` and `::planes`), so a linear's codes and scale plane can be two
   operands of one op, or one operand a kernel interleaves.
2. **Slicing** for tensor parallelism happens on the canonical form, where it is always a
   rectangle: a ROW shard is rows of every plane and a COL shard is columns, each cut on the
   plane's own block boundary and refused where a block would be split.
3. **Relayout.** The resolved kernel's `layout` hook says what it stores for the weight, given its
   encoding; `relayout` produces that from the rank's slices. A hook that answers
   `RAD_E_UNSUPPORTED` keeps the plane as it is, and those bytes go from the file to their
   destination unchanged.
4. **Agreement.** Every kernel that reads a weight (every band, both domains) must ask for the same
   layout, or declare refuses and names both. Device rows decide first: a host row that reads a
   weight differently from the device row claiming it is no error, only a reason its op cannot
   move to the host. Where no device row reads a weight (the host backend), the host rows' hooks
   decide, and a weight of several planes is joined into the form `abi/rad_planes.h` defines: a
   header naming each plane, then the planes unchanged. libref and libavx both read it, so either
   may serve any band.

**Selection reads the encoding.** A kernel whose `layout` hook answers `RAD_E_DTYPE` for a weight
an op takes is not a candidate for the op, and neither is a device kernel with no hook for a weight
of several planes. So one declaration serves a bf16 head with the bf16 GEMM and an fp8 head with
the fp8 one.

Relayout is host code on a thread pool behind the load sweep, and must keep pace with the disk
(Qwen3.8-27B-FP8 rearranges about 24 GiB of its 29 at every load). The sweep only reads, with
`O_DIRECT` into a ring of pinned windows; the pool makes the host copies out of the windows and the
relayouts, in memory reused from weight to weight.

**The file and mapped tiers cannot relayout**, because they read the container while serving. A
weight whose layout is not the identity goes to neither; the planner keeps it in the host pools and
says so by name. The n-gram table, the weight these tiers exist for, has the identity layout.

**A checkpoint loads directly when its encoding is trivial.** A safetensors directory or GGUF file
is a source like a container: the name map (§2.4) resolves each declared weight to checkpoint
tensors, concatenations and expert slices included, the reader states their encodings, and the
same path runs. Trivial means served as stored:

- `bf16`, `f16` and `f32`, and the exact widenings between them (a bf16 norm the plugin reads as
  f32);
- block-scaled FP8 (`quantization_config.quant_method = fp8`), where `X.weight_scale_inv` is the
  scale plane of `X.weight`: the encoding `fp8_e4m3*bf16[128x128]`, byte for byte;
- MXFP4 as AMD Quark ships it (`quant_method = quark`, fp4 per group of 32 with e8m0 scales
  throughout), where `X.weight` holds the codes and `X.weight_scale` the exponents: the encoding
  `fp4_e2m1*e8m0[1x32]`, byte for byte;
- GGUF `F32`, `F16` and `BF16`.

**Nothing is quantised at load.** A served form that needs quantisation (Flash-Next's int4
experts, an fp8 `lm_head` from a bf16 one) is converted ahead of time (§4.4). A direct load has no
file tier, no mapped tier (a checkpoint is neither grouped by movement unit nor aligned for
`O_DIRECT`) and no expert profile, so the planner keeps everything in VRAM and the host pools and
says so.

### 4.4 `rad-convert`

Reads safetensors or GGUF and writes `.rad`. It runs the architecture plugin's declare phase for
the declared weights and the name map, and quantises by a **recipe**: ordered rules from a logical
weight name to a quantiser and its options. The first matching rule decides; a weight no rule
matches keeps the checkpoint's encoding.

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

A rule names an algorithm (`rtn`, `gptq`, `cast`); its options name the grid (`codes` or `table`,
`group` or `block`, `scale`, a second level `scale2`, `zero`, `rule`) and the transform
(`transform=fwht128`) (§4.5). First match wins, so a protected expert's `cast` rule sits above the
rule every other expert takes. A quantiser that declines a weight (`rtn` declines a rank-1 tensor)
leaves it to the checkpoint's encoding, so a broad pattern need not exclude every norm.

`--recipe FILE` names a file and `--quant 'PATTERN=QUANTISER:k=v,...'` adds rules ahead of it.
`$NAME` in an option is read from the environment.

The plugin learns a weight's encoding the same way at convert and at load (`rad_weight_encoding`,
§2.4); at convert the answer is the recipe's. So the converter runs the engine's declare: every op
must resolve and every resolved `layout` hook must accept the planned encoding, and a recipe
nothing can serve is refused before a byte is written. Kernel plugins only check the recipe; which
library rearranges the planes later is not the converter's concern.

It reports the error of every weight it quantised against the source (relative Frobenius norm,
imatrix-weighted with `--imatrix`, worst first), through the core's decoder.

`--reuse C` copies from an earlier container every weight this run would write identically (same
encoding, quantiser, options and source shape), and `--in-place` extends `C` instead of copying
it, so adding a vision tower is about a gigabyte of I/O however large the container (§11).

### 4.5 Quantiser plugins

`$RADIANCE_HOME/quantizers/*.so`, the third plugin kind (`abi/rad_quant.h`). A quantiser turns a
weight in f32 into the planes of an encoding:

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
                        float* src, int64_t row0, int64_t rows, void* const* planes);
    /* Only for a scheme the core does not decode. */
    int     (*decode)(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes,
                      int64_t row0, int64_t rows, float* out);
} RadQuantizerInfo;
```

`RadQuantWeight` names the weight, gives its logical shape and movement group, and carries its
imatrix column importances when the converter has them. Other calibration (GPTQ's Hessians) is an
option the quantiser reads itself (`calib=DIR`), in its own format.

**libquant.** The built-in set, `libquant`, is organised by what is independent:

- a **grid**: the code type (any integer width, fp8, fp6, fp4, a non-linear table, a lattice) and
  the scale and zero structure over it, multilevel included, which makes MXFP4, NVFP4 and the K-
  and I-quants grids rather than algorithms;
- a **transform** applied first: FWHT, GPTQ's act-order permutation, ParoQuant's rotations (AWQ's
  column scales are a scale level of their own);
- an **algorithm** that chooses the codes on that grid.

Each algorithm pairs with any grid it makes sense for, and a recipe names all three:

| quantiser | algorithm | grids | writes |
|---|---|---|---|
| `rtn` | round to nearest; `rule` absmax, mx, fixed or search | every one below | the grid |
| `gptq` | error feedback against H; `act_order=1`; `cd=N` sweeps of coordinate descent after it | every one | the grid, `perm` under act order |
| `awq` | per-column activation scales searched against H, then a per-block clip | every untransformed one | the grid and a last scale level `[* x 1]` |
| `autoround` | sign-gradient descent on rounding offsets and block ranges against H | integer codes, row blocks, one level | the grid |
| `paroquant` | Givens pair rotations and channel scales fitted against H, then AutoRound | every untransformed one | the grid and a `givens` transform |
| `ggml` | llama.cpp's own searches, imatrix-weighted | its legacy, K- and I-quant types | each type's affine form |
| `cast` | a float in another float dtype | none | `plain` |

The grid is `codes` (integers to 32 bits, fp8/6/4) or `table` (a codebook: NF4, IQ4_NL's values,
the `w4nl` table Flash-Next's experts are served in and its 32-level `w5nl` form, binary, ternary,
or given), a scale block and dtype, an optional second scale level the first is relative to (an
E4M3 or integer sub-scale under an f32 or f16, as in NVFP4 and the K-quants), an optional integer
zero and an optional Hadamard.

Adding a quantisation format is a quantiser plugin plus a kernel that accepts its encoding; the
core does not change.

The Hessians come from the engine: `RADIANCE_CALIB_DIR` makes the architecture's calibration tap
write each layer's Gram matrices while a container serves a corpus, either a bootstrap container
(convert with absmax, calibrate, convert again) or the unquantised model, where the served
Flash-Next container's Hessians come from (`docs/MOE-W4.md`). The tap records the input as the
kernel multiplies it, rotated where the weight is stored rotated, so a Hessian is in the domain the
weight is quantised in. A layer-local objective is sum_r (w_hat_r - w_r) H (w_hat_r - w_r)^T, which
decomposes over rows. AWQ, AutoRound and ParoQuant spend their time in D·H, and libquant sums it in
a fixed order, so a weight converts to the same bytes on any machine and thread count.

---

## 5. Placement

All-in-VRAM, layer offload to RAM/SSD and expert offload to RAM/SSD are one mechanism at three
granularities. They differ in whether access is predictable:

- **Layer offload is a scheduling problem.** Declare enumerated every op and its weight operands
  in order, so the core knows the whole access sequence statically: exact prefetch, no predictor.
- **Expert offload is a prediction problem.** Routing is data-dependent, so there is no schedule,
  only a policy: the heat engine (§5.4).
- **All-in-VRAM is the degenerate case**: every weight on tier 0 and the mover never runs.

### 5.1 Tier and site

**Tier** is where the bytes live. **Site** is where the arithmetic happens. They are independent.

| tier | site | what it is |
|---|---|---|
| `VRAM` | device | resident |
| `HostPinned` / `SSD` | device, staged into a slab slot | the offload path: exact or predicted |
| `HostPinned` | device, zero-copy over the link | a routed expert streamed by its GEMM, or a table read in place; no slab slot |
| `HostPageable` | host | llama.cpp `-ngl`: bytes and arithmetic both off the card |
| `Mapped` | host, or device zero-copy | the container's own mapping, no pool |

All are core behaviour. Host-site execution needs a `RAD_DOMAIN_HOST` kernel (§2.1); where none
resolves, that site is unavailable for that op and the planner is told so at declare.

**The mapped tier.** `Mapped` is the container's own mapping. The other off-device tiers copy
(pinned and pageable memory at load, the file tier a slot at a time). A mapped weight occupies no
pool: the loader publishes a pointer into the container's mapping and the page cache decides what is
resident. It suits a weight of tens of gigabytes read a few rows a token, such as an n-gram
embedding table, which staging, pinning or a pool copy would each move or duplicate whole. The
planner maps:

- a weight no device kernel can read, run on the host;
- an n-gram table that `--ngram-placement` keeps off the card: `disk` gathers rows on the host from
  the mapping; `ram` copies the table at load into locked or pinned memory that the host gather or
  the card reads in place.

Only an identity-layout weight can be mapped, since the mapping holds the canonical planes (§4.2);
the file tier has the same rule (§4.3).

`--embedding-placement ram` and `--vision-tower-placement ram` put the token embedding and the
encoder towers in pinned host memory beside the host pool, read over the link in place.

### 5.2 Access classes

Declared per weight, because only the architecture plugin knows them:

| class | meaning | prefetch |
|---|---|---|
| `PER_TOKEN` | read every forward pass: dense weights | exact, from the declared order |
| `PER_REQUEST` | read once per request: an encoder tower, with an image | exact, on admission |
| `CONDITIONAL` | read only if selected: routed experts | predicted (§5.4) |
| `RARE` | may not be read at all in a given deployment | on demand |
| `VOCAB` | `lm_head` and the embedding table | exact, from the declared order |

`VOCAB` has its own class because it is read every token like `PER_TOKEN` but is very large at a
large vocabulary, and one of its members is a gather rather than a GEMM; neither `PER_TOKEN`'s
prefetch nor `CONDITIONAL`'s prediction fits it.

### 5.3 The planner

Runs once, after declare and before load (`core/place/planner.h`), and assigns each weight a tier
and a site against explicit pool budgets. Strategies (`--placement`): `all_vram`, `layer_offload`,
`expert_tiered`, `auto`.

The unit of placement is a weight group: an expert's gate, up and down move together, and the
container stores them adjacently (§4.2). Under tensor parallelism it is this rank's shard, planned
against this card's budget. The resident set must be identical on every rank, so the ranking uses
only the container's profile and each unit's identity.

Beyond fitting, the planner has three objectives:

**Contiguous host runs.** A host-site op leaves its activation in host memory, so a host layer
between device layers costs two link crossings. Host-site work is assigned in runs (layer offload
spills a suffix of the execution order), so activations cross once per run boundary.

**Prefill and decode want opposite plans.** A prefill chunk touches nearly every expert, so
residency gains little; decode touches a few experts per layer per token and residency decides its
speed. The planner produces one plan; the heat engine re-ranks it from realised routing and the
prefill stager covers prefill (§5.4).

**Warm start from the profile.** The container's expert-popularity profile, derived from the
imatrix's activation counts, ranks which experts begin resident. It is only a starting point:
imatrix popularity is not decode popularity, and the heat engine re-ranks within a few steps.

### 5.4 Residency, the mover, and heat

Fixed-stride VRAM slab slots per weight class, a pinned host pool, and a file tier read with
`O_DIRECT` into pinned buffers (`core/place/mover.h`). Slots within a class are interchangeable,
so a promotion is an integer swap and the slab never fragments. The file tier has predictable
latency and no page-cache copy, and lands reads where the mover can DMA them without a bounce;
every slot and sub-region is a multiple of the logical block size `O_DIRECT` requires.

The mover runs on its own stream; each dispatch's swap copies go to the card's DMA engine as one
list, and the compute stream waits on the mover's events. No transfer involves a host
synchronisation, so placement changes without serialising the step.

**Dynamic re-placement.** The heat engine (`core/place/heat.h`) accumulates heat on conditional
weights from the routing histograms, decays each layer's heat once per step, and promotes and
demotes against a hysteresis ratio and a minimum gain, up to 64 swaps per slab class per step
(`RAD_PLACE_MOVES_PER_STEP`). A promotion copies into a shadow slot before publishing the new
pointer, and a freed slot is quarantined until every launch that could name it has retired, so no
slot is overwritten while a kernel may read it. Link traffic is U-shaped in the minimum gain: a
declined promotion returns as a miss over the same link, so a higher threshold loses hit rate
without saving bytes.

**Prefill stager** (`core/place/stager.h`). A prefill chunk routes to nearly every expert, so
pinned-pool experts would cross the link inside each layer's GEMMs. For large steps the stager
copies them, for the layers ahead, into a VRAM ring at the top of the activation arena while
earlier layers run, and points each layer's ops at the copies. Output is bit-identical.

**Decode prefetch** (`core/runtime/xprefetch.h`). Under expert-tiered placement, each routed layer
applies the next layer's router to its own router input, copies the likeliest pinned-pool experts
onto the card by DMA, and redirects the next layer's weight tables to copies that have landed. A
late copy goes unused; nothing waits for it. The prefetch pauses when too few copies are used.

### 5.5 The dispatch contract

Routing is computed on device and never blocks the compute stream. Dispatch is resolved on device
against the current residency table; the routing histogram is copied back asynchronously and the
heat engine consumes it on the next step.

Placement decisions are one step stale, which does not hurt ranking: a hot expert stays hot across
many steps. A non-resident expert is not fetched on demand, which would have to fit between the
router's output and the expert GEMM's first block; it is read where it lies, streamed from the
pinned pool or staged from the file tier. The stager and the decode prefetch copy ahead on a
prediction, never on demand. Hit rate is a property of the placement policy, and the run phase has
no device round-trip.

---

## 6. Memory and budgets

Every pool is budgeted explicitly. The engine derives each budget from the card and prints every
term; the operator moves the inputs, never a pool size. There is no profiling pass and no
utilisation fraction:

| flag | |
|---|---|
| `--gpu-headroom-mib` | VRAM left unclaimed on every card (default 96) |
| `--host-pool-mib` | the pinned host tier |
| `--host-headroom-mib` | host RAM left free for the system at startup (default 4096) |
| `--weights-disk-tier` | the file tier for weights, read from the container; needs `O_DIRECT` |
| `--max-model-len`, `--max-num-seqs` | the KV cache's bound |

Nothing is inferred from the free VRAM at startup, which would size pools differently depending on
what else is open. Each term is a fixed property, a computed quantity, or a measurement printed on
its own line (`core/mem/vram_budget.h`). The startup report lists them in the order spent (sizes
omitted here):

```
VRAM budget, per card:
  card total           the part's totalGlobalMem
  already held         measured once, after every plugin's code objects are resident
  driver drift         what that moves by after it is read: kept free
  activation arena     computed from the buffer plan
  step staging         block tables, media staging and the KV tiers' copy staging
  hardware queues      the engine's streams, at a measured cost each
  expert prefetch      the decode prefetch's ring and buffers
  --gpu-headroom-mib   left unclaimed
  ------------------------------
  claimable
  static weights       off the top: every rank's own shard
  ------------------------------
  elastic              what the experts and the KV cache share
    experts
    kv cache
    prefill staging    the stager's region, out of the cache's share (or the experts')
  ==============================
  vram_weights         static + experts
  vram_kv              the cache's share
  unclaimed            the headroom, anything neither side could use, and MiB rounding
```

**Static weights are a floor, not a share.** Every non-expert weight (attention projections,
norms, embedding, head) must be resident for useful speed; under tensor parallelism each rank holds
its own share (§9). If they were split against the cache, the cache could push an attention
projection to host memory for blocks nothing fills. So they come out first (except a token
embedding `--embedding-placement ram` puts in host memory), and what remains is one trade: the
expert plane against the KV cache. Static weights that do not fit are refused with the shortfall
named.

**When the cache lends.** Under `--placement auto` or `expert_tiered`, without `--deterministic`,
on a device with virtual memory management, the cache lends every block no live request holds to
the expert slab. The experts then keep on the card what the pinned host pool cannot hold plus the
slab's spare slots (`expert_vram_floor`), and the cache takes the rest up to its ceiling, so it is
as large as the work needs at any moment. `--host-pool-mib` moves the split: a larger pool takes
experts off the card and gives the cache more. Without a lender an unused block would sit empty, so
the experts come first and the cache keeps one `--max-model-len` sequence; the cache never gets
less than that.

The mover never takes a block a live request holds. When the cache needs lent memory back, the
mover drops its coldest lent units, which keep host copies.

**Ceilings.** Both sides are capped at what they can use, and the excess goes to the other. A fully
resident expert plane needs no more bytes; a paged cache cannot have more than `max-num-seqs` x
`max-model-len` tokens live. Landing under that ceiling is normal: a sequence holds only the blocks
its tokens occupy, `KVManager::ensure` grows it as it runs and returns `RAD_E_FULL` for the
scheduler to preempt rather than failing the request, and blocks are dropped at completion. So the
engine reports the backing fraction without warning about it. The hard limit is one max-length
sequence: a context no single request can complete is a configuration error, since the API would
accept a request admission then refuses.

The split is computed once from printed numbers and nothing re-optimises it afterwards, so
benchmarks reproduce.

**What the budget cannot compute, it measures.** At startup the engine reads the card back once
everything has allocated and reports what settled there against what `--gpu-headroom-mib` asked to
keep. The engine's own device allocations are ledgered (`rad_dev_alloc_totals`); the rest of the
card is the driver's context, code objects, and any kernel plugin's own `hipMalloc`. That remainder
is what the headroom covers, and it is printed.

---

## 7. Scheduling

The scheduling model follows vLLM's.

### 7.1 Batching

Continuous batching, chunked prefill, priority queue, preemption by recompute. One scheduler
decision per step, producing one `RadBatch` (§8).

The scheduler owns chunk boundaries. Chunks split at linear-state checkpoint intervals (§7.3), and
the chunk quantum is the least common multiple of the attention block size and the linear-state
scan tile, both from the resolved kernels (`core/sched/geometry.h`, `RAD_PROLE_SEQ_CHUNK`). A
`--max-num-batched-tokens` below what that requires is refused at startup with the minimum printed.

### 7.2 Paged KV and hybrid groups

**Block size comes from the resolved attention kernel.** libr4d's paged attention is compiled for
16; the block manager reads the size from the resolved kernel at declare (§2.3, `RAD_DERIVED`), so
a kernel change needs no core edit.

Hybrid models are the normal case: Qwen3.5-27B has 48 linear-attention and 16 full-attention
layers, and Gemma 4 interleaves sliding-window and global layers. Each state kind is a **KV group**
(`RAD_KV_FULL`, `RAD_KV_WINDOW`, `RAD_KV_LINEAR`, `RAD_KV_CONV`) with its own block geometry, pool
and slot mapping in the step batch, so there is no common page size to pad layer bundles to (the
padding vLLM's KV-group-sizing patch in the table at the top works around).

Linear-attention state (GDN recurrent state, conv state) is per-sequence and fixed-size rather
than per-token, but it is a KV group too, allocated, mapped and freed by the same manager, so one
allocator decides what a sequence owns.

### 7.3 Prefix caching

Full-attention layers: content-hashed blocks with a chained hash, so a block's identity includes
its whole prefix; LRU eviction; copy-on-write on fork. Multimodal inputs hash by content (§11).

A GDN layer's recurrent state at position N depends on all of 0..N and is one fixed-size object
per sequence: on Qwen3.5-27B, 48 heads of 128x128 f32, 3 MiB a layer and 144 MiB across the 48
linear layers. A snapshot per block is not affordable, so the state is checkpointed. vLLM does this
at block boundaries (`--mamba-cache-mode align`), keeping the state of a scheduler step's last
token when it sits at `i * block_size`. This engine changes two things:

- **Independent interval.** The checkpoint interval does not depend on the attention block size.
  vLLM inflates the attention block to the mamba page size (528 tokens for Qwen3.5, 2240 for
  Qwen3.6-35B-A3B), so every prompt shorter than that misses on attention too. Here attention caches
  at the kernel's block size (16 in libr4d, §7.2), and the linear checkpoint interval is separate
  (`--checkpoint-interval`, default 2048, rounded up to the chunk quantum). A hit between
  checkpoints reuses the attention blocks and replays only the linear layers from the last
  checkpoint.
- **Deliberate checkpoints.** vLLM writes one only when a step happens to end on a boundary.
  The scheduler here splits a chunk at the interval (§7.1) and always writes one, at the
  cost of an occasional short chunk.

On a model with sliding-window attention and no linear state, a checkpoint snapshots the window
behind a prompt's tail instead, and chunks are not split at the interval.

Snapshots are large (at a 2048-token interval a 128K session on Qwen3.5-27B is 64 checkpoints,
about 9 GiB), so retention is budgeted. Retention keeps the session tip, which an appending agent
transcript resumes from, and any snapshot a lookup resumed from since the tip was written, which a
client rewriting its last message resumes from again. `--checkpoint-policy` chooses the rest: `geometric-backoff` (the
default) keeps, at every scale S = interval x 2^k, the shallowest checkpoint in the tip's S-token
bucket and in the bucket before it. Depths behind the tip grow geometrically, a session keeps about
log2(n) + 2 snapshots, and a turn that diverges d tokens behind the tip resumes less than
max(2d, two intervals) below the divergence. The buckets are fixed token ranges, so applying the
rule again each time the store fills keeps the same ladder instead of thinning it towards the tip.
`tip-only` and `keep-all` are the two ends of that trade, for comparison. Retention is a policy
class (`core/mem/prefix.h`), so a new policy is a new class (§19.1).

`--checkpoint-slots` bounds the snapshots for the whole server, not per sequence: they come from
one free list in pinned host memory. The default, `auto`, gives every sequence `--max-num-seqs`
admits what the policy keeps for one session at `--max-model-len`, plus one for the checkpoint a
running turn writes (geometric backoff keeps 9 for a 200K-token session at a 2048-token interval,
so 10 a sequence), within an eighth of host memory and what the host can pin after the rest of
startup; a count is taken as given. The store is shared by more conversations than run at once:
a parent waits while its subagent runs, and in a smaller store the subagent's snapshots displace the
ones the parent resumes from.

A session is a request id, and a conversation's snapshots belong to its latest turn: the
admission's lookup takes over every snapshot on the chain it hit, and the publish every snapshot on
the chain it published. A running turn and the snapshots its conversation already held are then one
session, retained as one ladder, rather than two ladders beside each other while the turn runs.

When retention leaves the store full, the least recently used snapshot goes, where a use is its
reservation or a lookup resuming from it. A tip has no further protection: a rewritten last message
leaves one tip per turn that no later turn reaches, while the snapshot every turn resumes from is
used every turn.

### Idle session tiers

A finished turn's prefix stays in the cache, which makes the next turn cheap; its blocks are dead
weight once the session stops (`core/mem/kvtier.h`). A finished entry is copied to host memory at
once and keeps its VRAM copy. Once the copy has landed, the VRAM is given up as soon as something
wants it (the expert slab's loan, or a request the pool cannot otherwise serve). A restore keeps
the host copy, so the next turn copies only what it added:

    published --(copy)--> clean in VRAM --(memory wanted)--> host

Host memory is the last tier. `--prefix-cache-mib` sizes it (default 0, off), and a full tier
drops its least recently used entries, for a copy worth more than they are. A chain is stamped
with one clock, and among equals the deeper block is worth less, in VRAM and on the host, so a
conversation larger than the tier loses its end and keeps its start, the part a lookup reaches.
A host hit is restored to the device. Nothing is written to
a drive: a disk tier would be written at the prefill rate to save re-prefills only for what
outgrows VRAM and host memory together. Restoring a 200K-token prefix over PCIe takes a fraction of
a second; re-prefilling it takes minutes.

**Every entry expires.** An entry unused for `--prefix-cache-ttl` seconds (default 3600, `0` for
never) leaves whichever tier holds it. A hit or a publish stamps every block of its chain, so a
conversation expires from its tip inwards and a shared system prompt lives as long as any user of
it. A snapshot expires with the entry it is keyed by.

**The pool is layer-major.** A logical block is `n_layers` fragments of a few hundred bytes, so
moving blocks one at a time would be all dispatch cost. Within a layer blocks are dense rows, so
packing a set is `gather_rows` and restoring is `scatter_rows` (`docs/OPS.md`): one launch per
(group, layer) for a whole batch.

**Recurrent state.** On a hybrid model a session is blocks plus recurrent state. The state lives in
a slot of the snapshot store (pinned host memory, `--checkpoint-slots`), copied off the card after
the step that produced it and back before the step that resumes from it, so it never holds VRAM.

The snapshot decides whether there is a hit at all: the scheduler clamps a hybrid session's reuse
to the last reachable linear checkpoint, because `RadBatch` carries one query length for every KV
group. A session restored without a reachable snapshot re-prefills its transcript. The two halves
are reported separately, since blocks restored without their snapshot lose the hit while every
block counter looks healthy.

**Testing.** Corrupt KV produces fluent, deterministic, plausible text that is not what the model
would have produced. `scripts/tierident.sh` asks one prompt cold, again from VRAM, and again after
the session has reached host memory, and compares output bytes. The VRAM repeat is the control,
since a cache hit can legitimately diverge on a near-tie; a run whose session never left VRAM is a
failure. A lost snapshot does not change the text (the linear layers replay from further back,
slower), so the snapshot check is a counter of prompt tokens covered with restored state.

`/sessions` reports what is stored, where, and what the tiers have done, failures included.

---

## 8. The step batch

`RadBatch` (`abi/rad_runtime.h`) is built once per step by the core and shared by every layer and
every KV group, rather than per group and per layer.

It carries: token ids and positions; per-group slot mappings and block tables; sequence boundaries
(`cu_seqlens`); query and context lengths with their host-side upper bounds; the speculative window
layout (§10); encoder rows and M-RoPE positions (§11); checkpoint destinations (§7.3); the phase,
which the run phase uses to index bucket tables (§2.2); the decode/prefill split; and each row's
runtime LoRA adapter slot (`lora_slot`, null on a step with no adapter row).

The split is `n_seq_decode` and `n_tok_decode`: the core orders every step so its decode rows come
first, and those two say where they end. A block that treats the two kinds of row differently (a
linear block must, §10) issues over each as a contiguous sub-range. `n_seq_decode` equals `n_seq`
on a pure decode step and 0 on a pure prefill one, so a block that does not care can ignore it.

An architecture plugin that needs derived per-step metadata (GDN chunk boundaries, conv-state
indices) declares it as a derived buffer (`RAD_BUF_DERIVED`), and the core computes it once per
step into the arena. Layers that share a geometry share the buffer.

---

## 9. Tensor parallel

Declare runs once per rank with `rank` and `world_size` in `RadBuildCtx`. The plugin declares
dimensions already divided and emits collectives as ordinary ops, resolved through the same kernel
hierarchy:

```c
decl_op(b, "all_reduce", RAD_PARAMS({"world_size", ctx->world_size},
                                    {"dtype", "bf16"},
                                    {"exact", 1}), ...);
```

The core neither infers sharding nor inserts collectives. A sharding inference pass is a compiler
pass whose failure mode is fluent wrong text at a rank count nobody tested; here the graph dump
shows every collective where the plugin put it.

The residual stream is replicated: every rank holds the whole hidden state between blocks. Weights
that act only on the stream, such as hyper-connection mixing (`arch/common/rad_block_hc.h`), are
replicated too, since sharding them would add a collective at each use.

`lm_head` is vocabulary-sharded like any column-parallel weight, so the sampler (§13) runs a
distributed top-k: each rank reduces its own shard and only the candidate sets are combined. Full
logits are never gathered.

libr4d supplies the exact all-reduce at 2, 3, 4 and 8 ranks, and a lossy wire (`--tp-wire`) at 2
ranks (Walsh-Hadamard-rotated 6-bit) and at 3 and 4 (tiered int8). With `--p2p off` the
collectives exchange through pinned host memory instead of peer access.

`RadBuildCtx::tp_wire_lossy` says whether the deployment accepts the lossy wire, and the plugin
inverts it into `all_reduce`'s `exact` parameter, so a zeroed context means exact.
`tp_wire_min_bytes` (default 128 KiB) is where the lossy wire starts paying: below it the plugin
must use the exact kernel, because compression pays only once the transfer is bandwidth-bound and
a decode step's message is not.

---

## 10. Speculative decoding

**Declaration.** A drafter is declared by the target model's own plugin, into the same builder under
a draft scope, so its kernels resolve through the same hierarchy and its weights are placed by the
same planner:

```c
RadDrafterDecl d = {};
d.name = "mtp";  d.kind = RAD_DRAFT_SERIAL;  d.depth = ctx->max_spec;
d.proposal = prop;  d.proposal_pitch = pitch;   /* 0 = the sampler's own token buffer */
d.mask_token = mask;  d.window = w;  d.block = n;   /* RAD_DRAFT_BLOCK only */
rad_declare_drafter(b, &d);
```

Otherwise the core would decide whether a container drafts by scanning weight names like
`mtp.fc.weight` and read the drafter's configuration keys (block size, mask token, sliding window,
predicted-layer count) that the plugin already reads: two readers of one fact, one of them the
core.

**Kind is a shape, not a model.** `RAD_DRAFT_SERIAL` runs `depth` dependent passes, each embedding
what the last proposed. `RAD_DRAFT_BLOCK` runs a context pass, then one pass that fills every
drafted position. MTP and EAGLE-style heads are serial; block-diffusion drafters (DFlash2, DSpark)
are block. Both run on the card's acceptance (`core/sched/advance.h`): the verify's acceptance, the
positions of every pass after it and the proposals stay on the card, so a step's passes and the
next step are issued before the step is read back, and the host's commit runs beside the card.

Separate draft models are not supported; a drafter is part of the model's source.
`rad-convert --draft-model` merges a drafter that ships as its own repository into the container
under the `draft.` prefix, and an MTP head inside the target checkpoint needs no flag. The server's
`--draft-model` refuses and names the conversion.

**Verification and acceptance.** Verification is a declared op. Acceptance is core sampler logic
(`core/sample/accept.h`) by agreement: a drafted token is accepted when it equals the target's own
token at that position (the argmax at temperature 0, the target's sampled draw above), so every
emitted token is the target's.

**Draft length** is `--num-speculative-tokens`, fixed per step by default. With a draft head whose
cost is nearly flat in depth, the best depth is a property of the model pair, and a gate on the
drafter's confidences would need them before the drafter has run. `--draft-window adaptive`
(serial drafters only) makes the number a ceiling: each sequence's stop follows its measured
per-position acceptance, and the step's head depth maximises expected tokens per measured step time
(`core/sched/draft.h`). It is off by default because it loses throughput on a busy server.

**Default depth.** `--num-speculative-tokens` defaults to `auto`, resolved before declare from
`rad_arch_probe()`: a serial head's throughput operating point, or the block length a block drafter
was trained at (`draft_depth_fixed` says which). A container with no draft head resolves to 0. A
merged drafter is used without a flag, so a deployment does not pay for a draft head it never runs.

**Prompt lookup.** Beneath the declared drafter, the scheduler runs an n-gram matcher that finds
where the last few committed tokens last occurred in the sequence. It fills any proposal the
declared head left empty, and is all of speculation for a plugin with no drafter (the normal case).
It is host matching over tokens the scheduler already holds, and proposes nothing where there is
nothing to repeat. It sits behind the declared head's eligibility gate, so a draft refused upstream
is never replaced by a different one.

The speculative window is part of `RadBatch`, which is why attention kernels take a query length:
libr4d's decode attention serves up to 64 query rows per KV head (`q_len * gqa <= 64`), the band a
verify step falls in.

**Mixed steps.** A mixed step verifies like any other, and this is a contract on the architecture
plugin. Continuous batching often puts a prefill chunk beside decode rows. A linear block issuing
one path for the whole step would put its decode rows on the chunked path, which commits state
directly and takes no `num_accepted`: rejected drafts stay folded into the recurrent state, and the
output drifts while staying fluent.

So a linear block issues the recurrent path over the decode rows and the chunked path over the
prefill rows, as two issues over two ranges; the core orders every step decode-first so each half
is contiguous (§8). These kernels take `cu_seqlens`, derive their grid from its length and their
rows from `cu[n]`, an absolute offset, so the prefill half is addressed by passing `cu + D` with
the token operands at their base and the decode half by narrowing both. Only ops without
`cu_seqlens` (a per-token gate or quantiser) need a row slice. Refusing to draft on mixed steps
would also be correct, but would cost many decode rows their draft.

**Rejection rolls back state, differently per kind.** Paged KV is a slot-table edit: free the
blocks past the accepted position. Linear state has already absorbed the rejected tokens: libr4d
keeps the conv state as a rolling window of `conv_width - 1 + n_spec` entries read at the slot the
last accepted token left, so a rejection changes a read offset. The recurrent state is kept either
per candidate token or as one committed anchor with the accepted prefix replayed onto it, as the
op's `state_form` parameter declares. Rollback support, with `num_accepted` as an operand, is part
of a linear-attention kernel pair's schema (`docs/OPS.md`).

---

## 11. Multimodal

**Coverage.** Pictures and video on every Qwen container that carries the vision tower
(Qwen3.8-27B in bf16 and fp8, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next) and on Gemma 4 31B
(`gemma4_bf16`); pictures on LFM2-VL; pictures, video and audio as EmbeddingGemma 2 input. Media
arrive as `image_url` / `video_url` chat parts (`input_image` / `input_video` in the Responses
API) and audio as `input_audio` / `audio_url`, as `data:` URLs only: fetching a caller's URL would
make the server an SSRF proxy.

**The processor is `core/mm/`.** One decoder (libavformat/libavcodec) reads every image and video
format from a memory buffer, with an empty protocol whitelist, and converts at the frame's own
colour matrix and range. The family comes from the container (`core/mm/processor.h`):

- **Qwen-VL.** `smart_resize` fits a picture to multiples of patch x merge within its pixel band,
  resampled with PIL's antialiased bicubic; patches are normalised, in merge-block order, a picture
  repeated to the temporal depth. A video is sampled at `fps` (default 2, 4 to 768 frames), each
  frame capped at `video_preprocessor.max_video_tokens` rows, and each temporal group is preceded by
  `<T seconds>`, since the groups share one temporal rotary component.
- **Gemma 4.** The largest size whose patch count fits the soft-token budget (70, 140, 280, 560 or
  1120); a video is frames at `fps`, each its own segment at the per-frame budget.
- **LFM2-VL.** A picture within its token budget is resized whole; a larger one is cut into tiles
  followed by a thumbnail. Images only.

`tests/mm_test.cpp` checks the processor against goldens generated by `tests/mkgolden.py` from
transformers' own functions. A media part reaches the chat template as a marker that the part join
leaves unpadded, so an image prompt tokenises to the reference's ids.

**The encoder is the language model's plugin.** The tower is declared last by the language model's
plugin (for Qwen, `arch/common/rad_block_vit.h` in `qwen35_bf16`, `qwen35_fp8`, `qwen35moe_fp8` and
`qwen4exp_fp8`) when the container carries its weights and the deployment asks for an encoder
(`--mm-max-patches`; `auto` is 16384 patches a pass when the container has a tower, a 2048 x 2048
picture at 16x16 patches; a larger picture is resized to fit, since a segment is never split). Its
weights are `RAD_ACCESS_PER_REQUEST` in groups past every text layer, declared with one
`rad_declare_encoder`, and an encoder pass (`RadBatch::enc`) issues only the tower. A program
declares at most two encoders: one for images and video, one for audio.

**Encoding happens between steps, never inside one** (`core/engine_mm.cpp`). The scheduler holds a
request back until its items are encoded (`Scheduler::media_ready`). The engine packs whole
segments (a picture, or a video's temporal group) into passes, every rank running the whole tower,
and reads the rows back into host memory. The batch builder copies them in with the chunk that
reaches them (`RadBatch::mm_rows` / `mm_embd`). Rank 0 gets the rows and every other rank zeros,
and the plugin writes them over the token embeddings before the vocab-sharded embedding's
reduction, so the sum counts each once. The placeholders stay the model's own pad ids, which a
hashed n-gram embedding (Flash-Next's PLE) reads like any other token.

**Positions are M-RoPE** on the Qwen family. `RadBatch::rope_pos` carries a pass's [3, T]
(t, h, w) planes: the image grid at the run's base, the next text at base + max(rows), each video
group a run of its own. The one-component rope path runs only when `rope_mixed` says no token's
components differ, so a text-only step's graph is unchanged. Decode rows continue at their index
plus the prompt's final shift. QSA block keys rotate at their first token's three components
(`qsa_work`'s `rope` operand, a `[work, 3]` `bpos`).

**The prefix cache keys a media run by its content** (a hash of the item's patches and the run's
row offset), never by its placeholder ids, which are the same for every picture. A hit that covers
a whole item still encodes it before admission, and the same picture in two requests is encoded
twice; a cache of the tower's output would avoid both.

**Containers gain the tower in place.** `rad-convert --reuse C --in-place` writes only the new
weights past the end of `C` with fresh tables and rewrites the header last (`<C>.pre-append` keeps
the old header and length): about a gigabyte of I/O for a 120 GB Flash-Next container.

---

## 12. Text frontend

Written once, in the core, mostly from llama.cpp, which is MIT, C++ and handles these problems in
production. The rule: take the text frontend and the tooling, not the engine.

A **lift** is upstream's file at upstream's path under `vendor/`, with local changes as a patch
series. A **rewrite** is a file that would need such heavy modification that it lives in `core/`
under its own name, with a comment at its head naming its source. `vendor/UPSTREAM` records the
commit, the split and the reason for every row:

| need | source | how |
|---|---|---|
| Jinja engine | `common/jinja/` | lifted, 5.0k |
| template capability probe | `common/jinja/caps.cpp` | lifted, 0.5k |
| reply format analysis | `chat.cpp`, `chat-auto-parser*.cpp`, `chat-diff-analyzer.cpp` | lifted, most of the patch series; the markers only |
| reply parsing | `chat-peg-parser.cpp`, `peg-parser.cpp` | rewritten → `core/text/chatparse.{h,cpp}`, one streaming pass |
| JSON-schema → grammar | `json-schema-to-grammar.cpp` | lifted, 1.2k |
| unicode and regex splitting | `common/unicode.cpp`, `src/unicode*` | lifted |
| HTTP and JSON | `vendor/cpp-httplib`, `vendor/nlohmann` | lifted, unmodified |
| GBNF grammar state machine | `src/llama-grammar.cpp` | rewritten → `core/sample/gbnf.{h,cpp}`, token masks in `gbnf_mask.cpp` |
| tokenizer merge machinery | `src/llama-vocab.cpp` | rewritten → `core/text/tokenizer.{h,cpp}` |
| GGUF reading | `ggml/src/gguf.cpp` | rewritten → `core/text/gguf.{h,cpp}` |
| sampling chain | `src/llama-sampler.cpp` | rewritten → `core/sample/host_ref.{h,cpp}`; reference only, §13 |
| `tokenizer.json` interpretation | `convert_hf_to_gguf.py` | Python, so rewritten → `core/text/tokenizer_json.cpp` |
| NFC/NFKC normalisation | none | llama.cpp has none → `core/text/unicode_norm.{h,cpp}` |
| OAI schema translation | none | original: `core/server/oai.cpp` |
| multimodal preprocessing | none | original: `core/mm/`, checked against transformers (§11); nothing of `mtmd` is vendored |

Not taken: ggml, its graph and scheduler, `llama-context` / `llama-batch`, the server's slot model.
Those are what this engine replaces.

Anything that cannot be expressed as a patch against upstream is rewritten, so `vendor/` stays
updatable. `llama-vocab.cpp`, for example, depends on `llama_model_loader`, `GGML_ASSERT` and the
`llama_vocab` class; its BPE bigram queue and UGM Viterbi are re-expressed with attribution.

**Reply parsing.** A reply is read by one streaming parser driven by a description of the model's
format. How a model spells a turn (reasoning tags, call opener, function and argument delimiters,
value encoding) is data (`core/text/chatfmt.h`), and everything that depends on the spelling is
derived from it: the reply parser, the lazy tool-call grammar and its trigger words, and the control
tokens the decoder must render as text.

There are two families of call syntax. In the tagged family each argument has its own tags (Qwen's
`<parameter=NAME>`, MiniCPM's `<param name="NAME">`) and is typed by the tool's schema. In the
object family a call is its name and one object holding the arguments: JSON, or a dialect with
strings between declared delimiters and bare keys, as Gemma 4 writes
`call:NAME{path:<|"|>/a<|"|>,limit:3}`; values carry their own types.

The format comes from the architecture plugin's optional `rad_arch_chat_format` export
(`abi/rad_builder.h`) when it has one, which covers a template the analysis cannot read. Otherwise
it comes from llama.cpp's differential analysis of the chat template: render with and without
tools and calls, and diff the outputs to find the delimiters. Only the analyser's markers are used,
not the PEG parser it would generate. A format neither source describes is not guessed: the model
still chats, and a request with tools is refused with the reason.

The parser (`core/text/chatparse.h`) reads each reply once as it streams: an Aho-Corasick automaton
over the delimiters that can end the current region, skipping plain text with `memchr`; a
transducer over the tag structure; an incremental scanner for JSON-typed arguments. Cost is linear
in the reply, and nothing on that path is a regular expression. The grammar is lenient where models
are: undeclared arguments, non-string arguments written as Python, an unclosed reasoning block,
prose around calls. Where a reply leaves the grammar (a function that is not a tool, text inside a
call) the parser recovers locally, says so, and still delivers the call, so the agent sees an
unknown name as an error it can recover from rather than as prose.

**What has been streamed is final.** The parser holds a byte only while its meaning is undecided (a
possible delimiter prefix, trailing whitespace that may be trimmed, a JSON-typed value whose type
its closing tag decides). The concatenated deltas equal the non-streaming answer.

**Tokenisation.** llama.cpp maps `tokenizer.json` in Python (`convert_hf_to_gguf.py`) and at
runtime identifies the pre-tokeniser by hashing `tokenizer.json` against a hardcoded enum, so an
unrecognised model gets a default split. `rad-convert` interprets `tokenizer.json` instead: the
declared normalizer / pre-tokenizer / decoder chain (`Sequence`, `ByteLevel`, `Split` with its
regex, `Metaspace`, `Replace`, NFC/NFKC) is resolved and stored in `.rad`, and a component it
cannot express is refused by name. This removes a failure class whose symptom, plausible text with
wrong token boundaries, is hard to notice.

Detokenisation is incremental and streaming-safe: byte-level BPE emits partial UTF-8 sequences and
a stop string may straddle a chunk boundary, so both are buffered until they resolve.

---

## 13. Sampling and structured output

**Sampling runs on device.** At a 151K vocabulary and batch 256 the f32 logits are about 148 MiB a
step; a host chain would add that transfer and a synchronisation to the step path.

Every sampler is a kernel: temperature, top-k, top-p, min-p, typical, repetition, frequency and
presence penalties, DRY, XTC, seeds. Sampler ops are declared and resolved like any other op
(`core/sample/sampler.h`), with libref's rows as `rad-kbench`'s oracle (§17). llama.cpp's chain is
ported to `core/sample/host_ref` as a host reference the test suite checks the chain against.

Two deviations from upstream are deliberate:

- **Chain order.** Penalties, temperature, the grammar mask, top-k, then top-p / min-p / typical,
  then XTC, then the pick (vLLM's order). llama.cpp applies temperature last. Top-p, min-p and
  typical cut on probabilities, so whether temperature ran first decides which tokens survive. A
  request may ask for llama.cpp's order with `samplers`.
- **The RNG is stateless**: a counter-based function of (seed, position, stream)
  (`abi/rad_philox.h`) instead of upstream's per-sampler Mersenne twister. A kernel cannot keep 624
  words of state per sequence, and a stateful generator makes the answer depend on how many draws
  happened before, which changes with batch composition.

Structured output is GBNF, with JSON schema compiled to a grammar. The grammar state machine runs on
the host, and a token bitmask per sequence (19 KiB at 151K tokens) is uploaded every step. The
backend is an interface (`GrammarBackend`); GBNF is the shipped one. Masks are assembled from a plan
precomputed when the grammar is compiled at admission (`core/sample/gbnf_mask.cpp`; XGrammar's
adaptive token-mask cache, keyed on stack suffixes), so a mask costs the same for new and seen
states.

---

## 14. Server

OpenAI's API: `/v1/chat/completions` (streaming and not), `/v1/completions`, `/v1/models` and
`/v1/models/:id`, `/v1/embeddings`, `/v1/images/generations` and `/v1/images/edits`, with tool
calls in and out and logprobs. `/v1/systemone` serves the decisions of LiquidAI's d1 when the
container declares it. A Prometheus `/metrics` endpoint uses vLLM's metric names, so existing
dashboards work.

Other clients' APIs are translated onto the same chat request rather than served by a second path:
Anthropic's `/v1/messages` (and `/v1/messages/count_tokens`), OpenAI's `/v1/responses`, and
llama-server's native `/completion` and `/infill`, with `/props` and `/apply-template` for
front-ends built against llama-server. Chat, embeddings, images, models and responses are also
served without the `/v1` prefix, as llama-server does; `/completions` without it is llama-server's
native endpoint. A request field this server does not implement is ignored when it cannot change
the output and refused by name when it would, unless it carries its off value (GUIDE §5.8).

`/v1/messages` lowers Anthropic's request onto the chat request: `tool_use` to tool calls,
`tool_result` to tool messages, thinking blocks to `reasoning_content`, the `system` field and the
system messages at the head of the conversation to one system message, and the text of a later
system message into the user message beside it, so the roles still alternate (templates read a
system message only at the head, and some refuse two user turns in a row). The answer,
its event stream and every refusal on the path, the transport's own included, are in Anthropic's
shapes. Anthropic's API refuses a field it does not define, and so does this endpoint: its clients
send beta fields that change what a request means (a `thread` continuation carries only the tail of
the conversation), so a field is read, taken because it cannot change the answer here, or refused
by name (GUIDE §5.11).

The operational set: `/health`, `/ping`, `/load`, `/version`, `/server_info`, `/diagnostics`,
`/tokenize`, `/detokenize`, `/get_tokenizer_info`, `/reset_prefix_cache`. Methods follow vLLM's:
`/tokenize` and `/detokenize` are POST (their input is a body), `/reset_prefix_cache` is POST (it
changes state), and `/ping` takes GET and POST because load balancers differ. `/server_info`
answers from the one route table.

An HTML dashboard is served at `/dashboard` and at `/`, with `/stats`, `/graph` and `/sessions`
(the idle session tiers, §7.3) beneath it.

Cancelling a request frees its blocks at once (or once no in-flight pass names them), not at
completion.

---

## 15. Tuning

Constraint matching finds a kernel that is correct for a geometry. Within one kernel there are
usually many tile configurations, and the best depends on the shape and the machine.

**A kernel declares axes, not named combinations.** `RadTunable` (§2.1) names one axis (`sk`,
`bm`, `npw`), its legal values ascending, and a default. `tune_valid` says whether a combination is
legal at a geometry, which independent lists cannot express (libr4d's skinny GEMM needs
`wv * sk * 32` inside the block bound and `K % (sk * unit) == 0`). Hand-enumerated points in a
multi-axis space leave holes that cost step time silently (a shape with forty rows of work runs a
sixty-four-row tile); a cross product has none, and keeps the tuning policy out of the launch
adapter, where an out-of-tree author could not write it.

The chosen values are appended to `RadArgs::p`, and the kernel reads `sk` with the same
`rad_args_geti()` it reads `M` with.

Every axis has an explicit `deflt`, used where this machine has no measurement (a fresh install, an
untuned shape, a cache from another card). The loader refuses a plugin by name if a default is not
among its axis's `values`, so an untuned install always runs.

The core owns the cache and the policy (`core/format/tunecache.h`). `rad-tune` benchmarks each
legal candidate at the declared graph's actual shapes on this machine and writes one file per
machine, `$RADIANCE_HOME/tune/<machine>.tune`, which the selector reads at declare. The key is
(kernel, plugin, geometry), so kernels serving one op tune independently, with ranges collapsed to
the band's upper bound (a bucket table row is what was measured). The winning point is stored as
sorted `key=value` pairs (`mb=3,npw=4,sk=8`): an axis no longer declared is ignored on lookup, and
a declared axis the row omits takes its default. The file is UTF-8 text, to be read, diffed, edited
and versioned beside a deployment. `RADIANCE_TUNE=op:key=value,...` pins axes for debugging; an
axis or value the kernel does not offer is an error.

`--debug-graph` names each chosen point and its source: the kernel's defaults, or a cache row and
when it was measured, since a measurement may predate a driver or library change.

---

## 16. Observability

Normal operation prints a few lines. Everything below is off by default.

- `--debug-graph`: every op, its bucket table, the kernel and tuned point for each band, the
  constraints that matched, the weights it touches and the buffers it writes; and the fusion
  advisory: declared ops some kernel could run in one launch (§2.1), and why it did not fire.
- `--debug-selection`: every distinct question the selector was asked, in first-asked order,
  with its answer and, for a miss, the refusing constraint. Repeats are folded and counted, so the
  count shows how much of the model resolved to a kernel. This is the record of why a fallback is
  running.
- `--debug-placement`: every weight's tier and site, the pool totals, and what did not fit.
- `--profile-ops`: per-op device timing. Profiling adds synchronisations that change the mover's
  slot supply and turns off recorded passes (§3.2), so a profiled run is not comparable with an
  unprofiled one.
- `--live`: per-card utilisation and VRAM, throughput, draft acceptance, the expert plane
  coloured by tier, residency per tier in units and bytes, and the interconnect split between the
  tensor-parallel all-reduce and the expert mover's traffic. Redirected, it prints one greppable
  `key=value` line instead.
- `--diagnose DIR`: run a fixed serving workload through the engine instead of listening, and write
  the measurement to `DIR` (§16.1).

### 16.1 The self-measurement mode

`radiance --diagnose DIR` starts the deployment as a serve would -- the same plugins, the same
placement, the same scheduler -- then runs one fixed workload through its own scheduler without
listening on a port, writes `DIR/diagnose.json` and `DIR/diagnose.md`, and exits. It measures the
engine rather than a client's view of it: the numbers come from the scheduler's own step records, so
the wait behind a prefill chunk, the interval between the steps that put tokens on the wire and how
much of it the engine spent waiting, and the cost of a draft pass are all in them, which a client's
throughput figure cannot separate.

The workload belongs to the engine and not to a script, because a workload that moves with the
change being tested measures nothing. It is a fixed table of prompts, held as token-id spans so that
what ran does not depend on a tokeniser, and a fixed table of points that combine them at a
concurrency, a request count and a generation length: single-sequence decode, decode at several
concurrencies, prose, code, structured output, long-context recall and long-form generation, prefill
at several lengths, and the cost an arrival adds to streams already decoding. Everything else --
chunk geometry, speculation, cache settings -- comes from the command line, because that is what is
under test.

The report gives, per point and pooled: step time by step shape; the interval from one step that
emitted tokens to the next, and the part of it the engine spent not running a step; tokens and rows
per step; throughput; prefix-cache hits and reuse; draft acceptance pooled, per request category and
per draft position; preemption counts; the step counts as the scheduler phased them and as the
records shaped them; and every generated token id.

- A quantity that could not be measured is absent, with the reason it is absent, and never zero. A
  point this deployment cannot hold says what would have to change to hold it, and does not fail the
  run.
- Nothing is served from the prefix cache. Every prompt is salted from `--diagnose-salt`, the cache
  is dropped before each point, and a point whose prompt was nonetheless a cache hit is reported as
  not a measurement rather than as a fast one.
- Sampling is greedy with `ignore_eos` and every request generates to its point's length, so two runs
  of one build write the same tokens: a build that says different words shows as a token difference
  instead of hiding inside a timing difference.
- `--diagnose-baseline FILE` compares the run against an earlier report. The workload's identity and
  its salt have to match, or the comparison is refused with both values named. Token parity is a
  failure, reported as the point, the request and the first token where the two disagree, with every
  point whose tokens still match kept in the report. Pooled draft acceptance -- per point, per
  category and per draft position -- is reported when it falls by more than
  `--diagnose-acceptance-drop` of what it was.
- The report reviews itself before it is written: a run that generated no token, a point that was not
  a measurement, step records dropped because the ring was too small, a cache hit, or a distribution
  whose percentiles do not order are printed to the log, and the ones that mean nothing was measured
  exit non-zero.

---

## 17. Correctness and failure

`libref` is a kernel plugin that implements the conventional op vocabulary in plain C++:
correct, slow, generic over shape. It declares itself a reference implementation
(`rad_plugin_reference`), which places it below every other library, and it has two jobs.

As an **oracle**, it is what `rad-kbench` checks the other libraries' kernels against, per op, on
a real container or a recorded fixture, and what the in-engine oracle below runs beside a device
issue. The other libraries are held to per-op tolerances, and ids exactly. libref is
deterministic and machine-independent, so its replay of its own recording is held to tolerance
zero: a difference is a change in libref or in the input draw, which a looser check would let
into the next recording as the target. A recording needs no card: it reads only a container's
tables and records libref's answer at every point, whichever library resolved it.

As a **fallback**, it is opt-in. An op that resolves only to libref runs orders of magnitude below
the machine with nothing in the output to say so, so by default startup stops with
`RAD_E_NOKERNEL`, naming every such op; `--debug-accept-reference-kernels` accepts the cost and
warns on every start. Because the fallback is also the oracle, it is tested.

**Coverage is not total.** Some libr4d ops have no libref row (docs/KERNELS.md lists them), mostly
fusions, which `replaces` (§2.1) checks against the conventional chain libref does implement.
libref also has limits: `kv_store` and `attn_paged` carry a `block_size >= 1` constraint, and the
entry points refuse with `RAD_E_UNSUPPORTED` what they cannot represent (a head dim above 2048, more
than 8 ranks in a collective, a convolution history wider than 16) rather than compute something
else.

Op names are free strings (§2.3), so the fallback and the oracle exist only for the conventional
vocabulary: an architecture plugin that invents an op has neither for it.

**In-engine oracle** (`core/runtime/oracle.cpp`). A second oracle runs inside the engine, on the
engine's own shapes. `rad-kbench` tests ops with operands it invents (dense, rank-2, freshly
filled); the engine's are row slices of a `max_tok` buffer, column slices of a fused projection,
rank-3 views and weights read from a file. An op can be correct at every shape `rad-kbench` knows
and wrong at the one the model issues.

`RADIANCE_OP_ORACLE` names ops. For each matching issue the runtime copies every operand to the
host before the launch, lets the device kernel run, copies the results back, and runs the reference
library's (libref's) host row for the same (op, band) on the untouched copy, with the same shapes,
strides and parameters. That is the band's host row when libref won it; when another host library
did (libavx), the runtime resolves libref's row at the band's geometry instead. Each issue
costs two device-to-host copies of every operand and a host synchronisation, so a step takes
minutes; it is a debugging instrument, off unless asked.

`RADIANCE_OP_REPEAT` asks whether a kernel agrees with itself: it snapshots the operands, runs the
kernel, restores them, runs it again and byte-compares. This catches what a single run per side
cannot: an unordered reduction, a workgroup reading what another writes, a workspace assumed
zeroed. There is no tolerance, since one mantissa bit becomes a different token a few hundred
greedy steps later.

Without per-op oracles, an engine whose output turns fluent but wrong cannot be bisected: every
component tests clean and the whole does not. Each kernel must be individually falsifiable.

**Refusal.** What the engine cannot do correctly it refuses, naming the cause: an
unimplemented path returns an error, never success, and a missing or unknown configuration (a
drafter's `draft.` keys, say) is refused by name, never defaulted. A guess yields output that is
fluent but wrong, or slower, and no error reports it. For the same reason, given a visible cost or
silent corruption, the engine takes the cost: at shutdown it leaks memory a running step may still
write rather than free it.

**Failure model.** A kernel returning negative at issue is a selection bug, not a runtime
condition: it aborts the step and fails its requests, naming the kernel and geometry, with no
fallback. KV pool exhaustion preempts by recompute, and a smallest running request that cannot
fit is failed rather than deadlocking. A device fault is not recoverable in-process; the engine
reports it and exits non-zero.

**Determinism.** Residency decides the summation order, so with dynamic placement identical inputs
need not give identical outputs. `--deterministic` pins placement: the plan is static, the heat
engine does not run, and the cache does not lend. Run anything compared across builds in this mode.

It does not give bit-exactness. The fp8 GEMM's split-K finisher sums in index order, so a run
repeats, but the split count is chosen from `M`, so a row computed at `M = 1` and at `M = 8` can
differ in the last bits. A speculative decode can therefore diverge from a non-speculative one on a
near-tie, and a request at a different batch position can take a different reduction order. Nothing
pins it (§19.7).

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
  tools/           rad-convert, rad-kbench, rad-tune, rad-info, rad-schemas, rad-tokcheck,
                   rad-chatparse, rad-gbnf-check
  tests/           the suite; runs in full on the host backend
  scripts/         shell harnesses for serving, tracing and profiling a running engine
  data/            the conversion recipes, and recorded kernel references rad-kbench replays with
                   no model present
  docs/            the user guide, the tools reference, the op vocabulary, the plugin contract,
                   the kernel report
  deploy/          compose files and the Grafana dashboard
  docker/          the container image's build files
  cmake/           the find_package config and rad_add_plugin, shared with out-of-tree plugins
  vendor/          llama.cpp lifts, tracked to an upstream commit, patched not edited
```

CMake, C++20, hipcc for device code. `RAD_WITH_HIP` (`ON | OFF | AUTO`, default `AUTO`) decides
whether there is any device code; it is a cache string because a boolean `option()` would collapse
`AUTO` to `OFF`. `RAD_GPU_TARGETS` lists the offload architectures device kernels are compiled for.
Unset, it is detected from the cards present; with a toolchain but no card, the device build is
declined by name.

The top-level build has no opinion about architectures: a kernel library declares the ones it
serves and leaves itself out of a build targeting anything else, so a card no library covers is a
configure that succeeds and an engine that refuses to serve (§17). The core's own device code (the
device layer's copies and fills, the step's bookkeeping, the weight-table patches) is also built for
every AMD GPU family's generic target, so the engine starts on a card its build machine never saw
and serves it with whatever kernel library is installed for it. A library whose code objects hold
nothing for the card is left out of that card's selection at load.

`$RADIANCE_HOME` (or `--radiance-home`) defaults to `share/radiance` under the install prefix and
is a `:`-separated search path, the first home winning; `architectures/`, `kernels/` and
`quantizers/` under each are scanned as one set. The kernel hierarchy is configured: `--kernels` (or
`RADIANCE_KERNELS`) is an ordered list, and a library present but unnamed is not loaded, since
selection runs per domain and an order alone could not keep a library out of a domain the named
ones leave empty. With no list every library found loads, and a reference library
(`rad_plugin_reference`) sits below every other, because it matches every op and would shadow the
libraries behind it. Adding a kernel library is dropping an `.so` into a home; substituting one is
naming it first.

The shipped plugins are plugins like any other: each builds inside the tree or against an installed
radiance through `find_package(radiance)` and `rad_add_plugin`, and nothing in the core names one.
`ctest -L oot` installs the build, rebuilds every shipped plugin out of tree against it, and
requires the same plugins, schemas and registration back.

---

## 19. Open questions

The numbers are stable because source comments cite them. A closed question keeps its number and
points to its answer.

1. **Linear checkpoint retention beyond the tip.** Geometric backoff is the default (§7.3), with
   `tip-only` and `keep-all` beside it as policy classes. No workload measurement has yet shown
   which retention curve is right.
2. Closed: `lm_head` and the embeddings have their own access class, `RAD_ACCESS_VOCAB` (§5.2).
3. **Zero-copy site selection.** The planner chooses the site by declared access class:
   `CONDITIONAL` streams over the link from the pinned pool, `PER_TOKEN` with no host kernel is
   staged into a slab slot. The principle is "zero-copy for what is read once per step", but the
   cutover against staging has not been measured, and it will differ between prefill and decode,
   which the single plan (§5.3) does not distinguish.
4. Closed: a drafter is part of the model's source and the planner places its weights like any
   other (§10).
5. Closed: the prefix cache writes nothing to a drive (§7.3), so it does not compete with the file
   tier for `O_DIRECT` bandwidth.
6. Closed: `RadTensor` carries a dtype the core understands (`abi/rad_types.h`): libref computes on
   tensors, `rad-kbench` compares them elementwise, the buffer planner sizes buffers from shapes and
   `rad-convert` writes bytes. Values at or above `RAD_DT_PLUGIN_BASE` are plugin-private, sized by
   the plugin's layout hook and never interpreted, and only for what a kernel stores; a container
   holds core dtypes only (§4.1).
7. **Bit-exact reproducibility.** `--deterministic` pins placement but not the fp8 GEMM's split-K
   count, which is chosen from `M` (§17). Whether that should become a pinned kernel axis (§15's
   mechanism could express one) or remain an accepted property of a batched engine is open.
   Pinning it costs the narrow shapes their tuning, since the count varies with `M` for speed.
8. Closed: `relayout` receives every plane the declared weight takes (§4.3), so a stored layout can
   be derived from a quantised weight and its scale grid together.
