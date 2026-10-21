# libavx

The conventional op vocabulary in hand-written x86 SIMD, runtime-dispatched by CPU feature.

A third kernel plugin beside `libref` and `libr4d`, for a domain whose only other implementation is
the reference one, which says in its own header that it is slow. Same op set as libref,
same schemas, same answers to a tolerance — the difference is the instruction stream underneath.

    cmake -S libavx -B build-avx -DCMAKE_PREFIX_PATH=<radiance prefix> && cmake --build build-avx -j4
    K=build-avx/radiance_home/kernels; R=<radiance prefix>/share/radiance/kernels
    $K/rad-avx-check --ref $R/libref.so --avx $K/libavx.so
    $K/rad-avx-bench --ref $R/libref.so --avx $K/libavx.so --all

It builds on its own exactly as a third party's plugin does -- `find_package(radiance)` and the
same `rad_add_plugin` -- against an installed radiance, with no ROCm and no core build. A
full-tree build on x86-64 always builds it: it is where the engine's HOST kernels come from. libref is the
oracle and the engine refuses to serve on it, so an op the placement runs on the host — the n-gram
embedding's gather, every step — needs this plugin.

## The embedding gather

`embed_lookup_q` over an E4M3 table into bf16 is the one host op a GPU deployment runs, wherever
`--ngram-placement` leaves the n-gram table for the host gather: in the container's mapping
(`disk`), or in a locked copy in RAM when no device kernel for the op is loaded (`ram`). Where
libr4d is loaded, `ram` is a pinned copy its kernel reads over the link and this one does not run.
Its rows come through `avx_ngram.cpp`, compiled baseline so the process has one of each: a row
cache shared by every thread (2^19 rows, eight ways, CLOCK), an O_DIRECT read through io_uring for
each miss, the ranks dividing a call's misses rather than duplicating them, and a reader thread
that fetches the next prefill chunk's rows while the current one runs. The level's kernel decodes
the rows: at AVX-512 the call's 256-entry bf16 table sits in eight registers and four `vpermt2w`
look up 32 codes; below it, one code at a time. The row's layout hook only checks the table's
encoding — E4M3 with one bf16 scale, or the checkpoint's plain bf16 — and asks for no re-layout,
because the gather reads the rows from the container file in the layout they are stored in.

`rad-avx-check` runs the op over a file-backed table with two ranks and the reader racing, at
every level, and compares every row with the file's bytes.

## The build is the dispatch

Every kernel source is compiled **four times** into four object libraries — scalar / AVX /
AVX2+FMA / AVX-512 — with `AVX_LEVEL` defined and the matching `-m` flags. `avx_vec.h` wraps each
level in its own inline namespace so the four symbol sets cannot collide; without that the linker
picks one definition of `avx::row_to_f32` for all four tables and the *scalar* path executes
AVX-512, which is a SIGILL from the fallback that exists to prevent SIGILLs.

`avx_dispatch.cpp` and `avx_registry.cpp` are compiled **baseline, with no `-m` flags**, because
they are what decides whether the others may run. That is a property of the flags and not of the
source: a compiler given `-mavx512f` may auto-vectorise anything, so "contains no intrinsics" is
not the same claim. `-march=native` appears nowhere.

Detection is `__builtin_cpu_supports`, not raw CPUID, because the OS has a say — a CPU may
enumerate AVX-512 while the kernel has left the ZMM state disabled, and using it then faults.
AVX-512 is checked as the conjunction F + BW + DQ + VL, all four of which the kernels use.

`avx_isa_force()` is an exported symbol, not an environment variable, so the checker can sweep all
four levels and the engine can never reach it.

## Coverage

75 of 75 ops. Every op `libref` implements, `libavx` implements; `rad-avx-check` fails if that
stops being true, and reports a row that returns `RAD_E_UNSUPPORTED` separately from one that is
absent — a row that declines is worse than no row, because the selector resolves to it and the step
fails at issue rather than at declare.

Two things are deliberately narrower than libref and say so at the call site:

- **`all_reduce` / `all_gather` serve world size 1** and refuse anything wider by name. libref's
  real implementation is a rendezvous over shared mutable state with a barrier and a generation
  counter, and there is exactly one of it per process; two kernel plugins each carrying their own
  would be two rendezvous groups for one world, and which one a rank joined would depend on which
  plugin won selection. libref sits below this plugin in the hierarchy, so a tensor-parallel host
  deployment resolves the collectives there and everything else here.
- **The rows carry no `opd_shape`.** libref publishes one for the whole conventional vocabulary and
  the ABI says an op inside `docs/OPS.md` is describable even when the kernel under test is not.
  A second copy would be a second set of judgement calls about the same ops.

## Correctness

`rad-avx-check` opens both plugins with `dlopen` and links neither — the claim is about the `.so`
the engine would load. It checks four things in order: schema agreement (the loader's own
`schema_same`, run before any arithmetic, because the loader refuses the *whole* plugin on one
disagreement), coverage, the numbers, and every ISA level.

    233 checked, 233 passed, 0 failed, 0 declined, 4 skipped, 0 missing.
    worst rel_l2 = 3.556e-04   [rmsnorm M=3 n=645 bf16 wadd=1]
    hand-built: 22 checked, 0 failed

at every level, against `rad-kbench`'s tolerance table (the worst case at AVX2 is silu_mul's
4.038e-04 rather than rmsnorm's). Every conversion is
additionally proven **exhaustively** — all 256 or 65536 bit patterns of each narrow format, in both
directions, bit-identical to libref including NaN, subnormals and negative zero. That sweep is not
redundant with the geometry table: a draw of normals reaches none of those patterns, so a
conversion can be wrong by octaves at the bottom of its range while every geometry still passes.

**The four skipped cases are limits of libref's descriptions, not of libavx.**
`sample_merge_topk`'s shape hook returns `RAD_E_UNSUPPORTED` unconditionally, so it publishes no
description at all. `gemm_nt_q_bias` at `fp8a16` is declined by name (the bias form takes `a_scale`
as required and fp8a16 has none). `ngram_ids` describes `vocab_sizes` as normal-filled i64, which
its own launch then refuses because a vocabulary size must be positive — so that op cannot be
checked cold until libref's hook describes it as an index fill with a positive range.

A cold description cannot reach every form of every op, and the **hand-built cases** are the rest:
operand views and index patterns a description does not draw, and the forms an optional operand
selects. `dflash_select`'s are the sharpest example: a description draws its codebooks as weights,
at 0.02, where the pairwise term is a thousandth of the unary score and never decides a pick — so
the walk is checked cold and the form it computes is checked by hand, with unit-scale codebooks, once
with `hp` and once without.

## The measurement platform

The numbers below are one thread on a **Ryzen 9 9950X3D** (Zen 5, 16 cores, 128 MiB L3), taken with
`rad-avx-bench --peak` rather than quoted from a data sheet. All four ISA levels execute natively on
that part, so all four are measured and none is inferred. Re-run the commands above to get the
figures for another one; the ratios are what the rest of this document reasons about.

    FMA scalar     29.3 GFLOP/s      triad L1     129 GB/s   (48 KiB)
    FMA avx       135.0 GFLOP/s      triad L2     157 GB/s   (1.5 MiB)
    FMA avx2      175.0 GFLOP/s      triad DRAM    42 GB/s   (768 MiB)
    FMA avx512    351.4 GFLOP/s

The avx2 and avx512 figures are exactly 2 FMA/cycle at 8 and 16 lanes; avx (no FMA) reaches 4
vector FP ops/cycle on Zen 5's four FP pipes. The triad counts three arrays — two read, one
written — and does not count a read-for-ownership the compiler may have elided.

## Conversion, MB/s of source bytes

`rad-avx-bench --cvt`, through the plugin's own `cast`.

| pair | scalar | avx | avx2 | avx512 | libref |
|---|---|---|---|---|---|
| f32 → bf16 | 17885 | 31739 | 40436 | 78707 | 1725 |
| bf16 → f32 | 28598 | 23382 | 34531 | 40154 | 990 |
| f32 → f16 | 1245 | 1246 | 138027 | 159337 | 670 |
| f16 → f32 | 2706 | 2706 | 34122 | 39493 | 717 |
| f32 → fp8_e4m3 | 1102 | 1165 | 20619 | 26463 | 623 |
| fp8_e4m3 → f32 | 4512 | 4486 | 8294 | 12545 | 360 |
| f32 → i8 | 4247 | 10699 | 99319 | 164889 | 1071 |
| i8 → f32 | 12088 | 11182 | 17324 | 20526 | 384 |

f16 is software below AVX2 because F16C is its own CPUID bit that arrived one generation after
AVX — a level called `avx` that required it would fault on the machines it exists for.

## Per op, at model shapes

`rad-avx-bench`, microseconds per launch, median of 7 reps, **one thread**, cache-warm. Geometry is
a ~1.5B dense model: n_embd 2048, n_ff 5632, n_vocab 32000, 16 heads of 128. M = 1 is decode, 64 a
speculative verify window, 512 a prefill chunk.

| op | geometry | libref | scalar | avx | avx2 | avx512 | rate (avx512) |
|---|---|---|---|---|---|---|---|
| gemm_nt | M=1 N=2048 K=2048 | 1589 | 537 | 345 | 282 | **241** | 34.8 GFLOP/s |
| gemm_nt | M=64 N=2048 K=2048 | 101770 | 30355 | 6958 | 3846 | **2860** | 187.7 GFLOP/s |
| gemm_nt | M=512 N=2048 K=2048 | 816412 | 242612 | 55778 | 30745 | **22870** | 187.8 GFLOP/s |
| gemm_nt | M=512 N=5632 K=2048 | 2244318 | 672900 | 156165 | 87061 | **65176** | 181.2 GFLOP/s |
| logits_gemm | M=1 32000×2048 | 25055 | 8861 | 6808 | 6866 | **5525** | 23.7 GB/s |
| rmsnorm | M=512 n=2048 | 2351 | 433 | 248 | 200 | **87** | 48.2 GB/s |
| silu_mul | M=512 n=5632 | 9971 | 5426 | 1874 | 1281 | **740** | 23.4 GB/s |
| softmax | M=512 n=2048 | 3236 | 2618 | 475 | 306 | **204** | 41.0 GB/s |
| quant_act_fp8 | M=512 n=2048 g=128 | 7745 | 5254 | 4767 | 344 | **251** | 12.5 GB/s |
| had_quant_act_i8 | M=512 n=2048 g=128 | 7362 | 3633 | 731 | 367 | **261** | 12.1 GB/s |
| sample_argmax | M=1 n_vocab=32000 | 23.7 | 14.3 | 2.02 | 2.02 | **1.15** | 111 GB/s |
| sample_topk | M=1 n_vocab=32000 | 2287 | 1067 | 174 | 139 | **77** | |
| attn_paged | q_len=1 hd=128 gqa=4 ctx=2048 | 6.73 | 1.53 | 1.18 | 1.11 | **0.93** | |

**How to read this.** The libref column is a statement about libref, which is a scalar loop with a
per-element dtype switch and says so in its own header; the ratio is what says an op is wired at
all. The rate column against the ceilings above is the statement about libavx:

- `gemm_nt` at prefill reaches **188 GFLOP/s against a 351 GFLOP/s ceiling — 54%**.
- `sample_argmax` reaches **111 GB/s against a 129 GB/s L1 ceiling — 86%**; it is a single pass
  over the logits and is essentially at the memory system.
- `softmax` at 41 GB/s and `logits_gemm` at 24 GB/s are **DRAM-bound, not flop-bound** —
  `logits_gemm`'s weight is 128 MiB in bf16 and fits in no cache, so whatever the flop count says
  it is a memory benchmark wearing a GEMM's name, and 24 against a 42 GB/s single-thread triad is
  the honest way to read it.
- `silu_mul` at 23 GB/s and `rmsnorm` at 48 GB/s are the two that have room left.

These are **single-thread** numbers throughout. The plugin parallelises over the outer dimension
with OpenMP above a work threshold; the bench pins one thread because that is what ranks a kernel
change, and a 32-thread number would be measuring the memory controller.

## Why the code is shaped the way it is

Each choice below is measured rather than reasoned about, and the ones that did not pay are recorded
beside the code, because a null result is worth as much as a win.

**The register block is swept, not reasoned about** (`gemm_nt` bf16, M=512 N=2048 K=2048):

    AVX-512, 32 ZMM   4x4 23.2 ms   4x6 22.1 (chosen)   6x4 24.2   8x4 23.8   8x2 35.2   12x2 34.1
    AVX2/AVX, 16 YMM  3x3 34.4 ms   2x6 30.7 (chosen)   2x5 31.9   2x4 34.8   3x4 58.4   4x3 58.1

Two things in that table matter more than the choice. The **cliff**: 3×4 and 4×3 at AVX2 are 58 ms
against 31–34 around them, because 12 accumulators plus 3–4 A vectors plus one B is 16–17 live in a
16-register file, and the spill lands *inside* the k loop. And **wider in N beat wider in M at equal
arithmetic intensity** — 2×6 and 3×3 both load 8 vectors, for 12 and 9 FMAs, intensity 1.5 either
way, and 2×6 is 11% faster. Nothing about the flop count predicts that.

**NULL RESULT — cache-blocking the A panel is worth 6% on this part**, not the several-fold the
traffic argument predicts. Swept: 128 KiB 26.0 ms, 256 KiB 23.4, 1 MiB 24.0, 4 MiB 25.4, 16 MiB
25.4, unblocked 25.0; and at M=64 blocking is *worse* (2.91 against 2.85). The reason is the part:
128 MiB of L3, so the traffic the blocking removes is served at L3 speed and not DRAM's. Kept,
because it is never much worse and is the real fix on a part with 8 MiB of L3 — but it is worth six
percent where there is this much cache, not the several-fold the traffic argument suggests.

**NULL RESULT — the fp8 decode table gather loses to scalar.** Every fp8 byte is one of 256 floats,
so a lookup is exact and obvious; `vpgatherdd` measured 3268 MB/s at AVX2 and 3815 at AVX-512
against **4512 scalar**. A gather issues one cache line per lane however small the table is.
The decode is a branchless exponent reconstruction instead: 8294 and 12545 MB/s, 2.5× and 3.3×.

**A "fast path" whose epilogue is a per-element branch is not a fast path.** Computing sixteen lanes
of an fp8 encode with vector integer arithmetic and then spilling them to a stack array to run a
scalar loop testing one mask bit each costs roughly twice what the vector form does: on a normalised
activation the mask is all-ones every time, so that loop runs on exactly the path it was written to
avoid. The encode finishes in vector registers.

**An unvectorised level is not a half-speed level.** A level that falls through to the ABI's scalar
converter for a conversion costs an order of magnitude more than the vector form, not the 2× its
width would suggest — so every conversion is written at AVX2 as well as at AVX-512, and AVX2 is the
level most machines in the world run.

**A fallback that loses to the thing it falls back from is not a fallback.** The top-R selection's
two-pass form (vector max, then find the index) pays only because the first pass is vector work; at
the scalar level it is strictly two passes where a single-pass loop does one, and it measures slower
than libref's. The scalar level therefore uses the single-pass loop. The confirmation pass is
vectorised at every level for the same reason: scalar, it reads n/2 elements on average and is most
of what `sample_argmax` costs.

**A ceiling has to be compiled at the level it describes.** A peak loop compiled baseline measures
SSE2 whatever level it is nominally for, and every kernel looks excellent against a ceiling an order
of magnitude too low — so the peak kernels live in the plugin and are compiled per level like every
other kernel here. The accumulator count matters as much as the flags: sixteen accumulators plus two
constants is eighteen live vectors and spills a 16-register file, which costs more than the FMA
saves, so the AVX and AVX2 peaks use ten.

## Layout

    avx_isa.h          the ISA levels and AVX_OP_LIST -- the only place the op set is written
    avx_vec.h          the vector abstraction and the dtype conversions; compiled 4x
    avx_common.h       libref's addressing contract, the scratch arena, the threading rule
    avx_dispatch.cpp   BASELINE: CPUID, the level tables, the thunks, the fp8 table
    avx_registry.cpp   BASELINE: the schemas (libref's, slot for slot) and the kernel rows
    avx_ngram.cpp      BASELINE: the embedding gather's row cache, direct reads and reader
    avx_rows.h         the gather's two halves: the row source and the per-level decode
    avx_support.cpp    the generic row, the reductions, the top-R step; compiled 4x
    avx_had.h          the Hadamard rotation, the fp8 group, the fp8 fusions' shared epilogue
    avx_{elementwise,norm,hc,quant,gemm,attn,moe,gdn,sample,misc}.cpp   the kernels; compiled 4x
    avx_harness.h      plugin loading, the operand draw, the geometry table -- shared by both tools
    avx_cases.cpp      the geometries; one table so the check and the bench cannot disagree
    avx_check.cpp      rad-avx-check
    avx_check_rows.cpp rad-avx-check's file-backed embed_lookup_q case
    avx_bench.cpp      rad-avx-bench
