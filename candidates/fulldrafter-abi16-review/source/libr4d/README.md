# libr4d

R4D is a library of specialized HIP kernels for **gfx1201 (RDNA4, Radeon AI PRO R9700)**.

It is built by the radiance engine, which is its only consumer: `libr4d/CMakeLists.txt` names
every translation unit and the per-unit flags each one needs.

## Naming

An entry point is

```
<family>_<op>_<the geometry it is compiled for>
```

and the geometry is not decoration. These are specialised kernels: every dimension in the name is a
compile-time constant, and the entry point **rejects a mismatch rather than running**.
`attn_decode_h256_gqa6_fp8kv` serves head_dim 256 with 6 queries per KV head and an fp8-e4m3 paged
cache, and nothing else. A model with a different head size needs a new instantiation, which gets
its own name beside this one -- no existing name has to change to make room for it, and no caller
can pick up a kernel that does not fit by accident.

Dimensions that are the same across the whole library discriminate nothing between entry points and
so stay out of the names: the paged block size (16), the query dtype (bf16 everywhere) and the
target architecture. They are reported by the `r4d_*_dims()` accessors in
[`r4d.h`](r4d.h) and by the `RadKernelInfo` rows in `r4d_rows.cpp`.

| entry point | what it is |
| --- | --- |
| `attn_prefill_h256_gqa6_fp8kv` / `..._bf16kv` | paged causal attention, query-tiled; head_dim 256, 6 queries per KV head, fp8-e4m3 or bf16 KV cache |
| `attn_decode_h256_gqa6_fp8kv` / `..._bf16kv` | the same geometry, split-KV, up to 64 query rows |
| `attn_decode_h256_gqa6_scratch_bytes` / `attn_decode_h128_gqa4_scratch_bytes` | how many bytes of split-KV scratch a given launch needs — ask before allocating; the split count is chosen from the geometry |
| `attn_vit_h72_bf16` | dense non-causal vision-encoder attention, head_dim 72 native, varlen over images |
| `gdn_conv_prep_w4_h128_bf16` | the gdn prefill preamble in one kernel: causal conv (width 4, silu, with its state cache), q/k/v split, qk l2norm, gating and the per-chunk gate cumsum |
| `gdn_kkt_solve_k128_c64_bf16` | `A = (I + strict_lower(diag(beta) K K^T e^dg))^-1` per chunk of 64 -- the gram and its triangular inverse, without the fp32 gram ever reaching memory |
| `gdn_chunk_scan_k128_v128_c64_bf16` | gated-delta-net chunked scan -- WY recompute, state recurrence and output in one kernel; head_k 128, head_v 128, chunk 64 |
| `gdn_conv_update_w4_h128_bf16` | the same convolution for a decode step: a rolling speculative window over the conv state cache, writing q/k/v directly |
| `gdn_recurrent_update_k128_v128_bf16_fp32state` | the decode-time recurrent delta-rule update against the paged fp32 state cache, gating and qk l2norm included |
| `gdn_gated_rmsnorm_h128_bf16` | gated rms norm over a 128-channel row, `out = rms(x) . w . act(z)` |
| `ar_oneshot_2rank_exact` | one-shot push all-reduce over P2P, exactly 2 ranks, exact bf16/fp16/fp32 |
| `ar_oneshot_2rank_wht6` | the same handshake with a Walsh-Hadamard-rotated 6-bit wire payload (lossy) |
| `gemm_bf16_nt_m16` | skinny bf16 GEMM, `C[M,N] = A[M,K] @ W[N,K]^T`, M ≤ 16, split-K. **Superseded on every shape measured by the WMMA row below**, which the shim forwards to — its accumulators are registers indexed by row, so it falls off a cliff past 8 rows and is several times slower at M = 16. `R4D_BF16_M16=1` reaches it. |
| `gemm_bf16_nt_m64` | the same GEMM for M ≤ 64, one 16x16x16 WMMA per row tile per 16 of K |
| `gemm_w4a16_nt_m64` | the same GEMM with a **4-bit weight** — asymmetric scale and integer zero per 128 K, pre-permuted offline into WMMA fragment order so the weight path is one `global_load_b128` per lane per four k steps, f16 activations, bf16 out |
| `gemm_w4a8_nt_m64` | the 4-bit weight against an **int8 activation** through `v_wmma_i32_16x16x16_iu8` — twice the matrix rate of the f16 form. Symmetric scale per 128 K; the skinny decode kernel, M ≤ 64, per-shape (WV, SK, MB, NPW, NT) bands |
| `gemm_w4a8_asym_nt_m64` | the same with an asymmetric grid: unsigned B fragment, per-group zero from the scale dword's high half, minus the activation's per-group row sums |
| `gemm_w4a8_tiled` | the prefill/large-M sibling, block tile 128x256, BK=64 double-buffered through registers; variant-indexed tile table |
| `gemm_w4a8_asym_tiled` | the asymmetric grid at the tiled shape |
| `gemm_w4a8_prefill` | a second large-M W4A8 kernel, outranking `gemm_w4a8_tiled` in the M > 64 band; a tuned pick that names no prefill variant falls back to the tiled one |
| `gemm_w8a8_nt_m64` | int8 weight against int8 activation, skinny; the promoted (8-bit) linears of a mixed-precision artifact |
| `gemm_w8a8_tiled` | the prefill sibling of the above |
| `gemm_w8a16_nt_m64` | 8-bit weight, f16 activation, skinny |
| `gemm_w2a8_nt` | 2-bit weight against int8 activation — the fast draft head |
| `r4d_fulldrafter_lmhead_gemv12_kernel` | coarse teacher head: SmoothQuant-balanced W4A8 with interleaved unsigned nibbles, 158,924,800 weight bytes per 124160×2560 shard plus row/column scales; exact BF16 rescoring of the selected top 16 retains the original hidden-slot indexing |
| `gemm_mxfp4a8_nt_m64` | OCP-MXFP4 weight, fp8 activation, skinny |
| `attn_decode_h128_gqa4_fp8kv` / `..._bf16kv` | head_dim 128, 4 queries per KV head, sliding window — a block-diffusion drafter's geometry. Decode only; `attn_h128_gqa4_have_prefill()` reports 0 so a caller routes long queries to its own fallback |
| `rowtopk_bf16` / `rowtopk_bf16_chunks` | exact per-row top-K over a bf16 matrix, two-stage |
| `dflash_conv_t2_g16_bf16` | the drafter's causal conv, two taps, 16 groups |
| `dflash_select_bf16` | the drafter's candidate selector: edge scores and the greedy path walk |
| `dequant_w4_bf16` | unpack a 4-bit fragment-ordered weight back to bf16 |
| `quant_act_i8` / `quant_act_i8_asum` | per-row int8 activation quantiser, **fragment-ordered** for the r4d GEMMs (see the warning below); the `_asum` form also emits the per-group row sums the asymmetric GEMMs need |
| `had_quant_act_i8` | the same with a block Walsh-Hadamard rotation first — the rotation is what makes one scale per row viable |
| `rmsnorm_had_quant_i8` | residual add, RMS norm, rotate and quantise in one kernel |
| `ar_ln_had_quant_i8` | **the all-reduce and the norm that always follows it, fused**: the peer's contribution is a second operand on the norm's load, so the all-reduce's bf16 output never exists. Exact bf16 or rotated 6-bit wire (`wire`) |
| `gated_had_quant_i8` | `silu(a)*b` or `sigmoid(a)*b`, then the same rotation and quantiser |
| `gdn_gated_norm_had_quant_i8` | the gdn gated norm with the rotation and quantiser fused onto it |
| `qk_norm_rope_gate` | q/k rms-norm, rope and the attention gate in one kernel |
| `ar_oneshot_{4,8}rank_exact`, `ar_twoshot_{4,8}rank_exact`, `ar_twoshot_4rank_ti8` | the wider-world all-reduces (TP4/TP8), exact and int8-wire |
| `p2p_copy2d`, `p2p_copy2d_kernel`, `p2p_fence` | strided peer device-to-device copy, through the copy engines or a kernel |

## ALoDLM 2x4 draft window

With `alodlm_max_depth == 1`, FullDrafter uses two four-slot Transformer passes.
The second pass starts from the normalized hidden state and selected token of
slot 3; it is not launched when the requested depth is at most four. The engine
passes that depth in the opaque GPU-parameter tensor's host extent (1–12 means
a requested depth; the legacy full-struct extent uses `alodlm_pad`): 3 for concurrent decoding, 4 for
normal single-stream decoding, or 7 after a high-acceptance streak. Seven
proposals plus the teacher anchor fit one eight-row verification tile.

Standalone callers using the legacy parameter extent and `alodlm_pad == 0`
request all eight proposals, regardless of batch size. Unsubmitted slots through slot 11 receive the sentinel token.
Markov pruning retains the exact Cauchy–Schwarz certificate and Top-16 selector;
this path does not use a quantized shortlist.

## Asking the library instead of remembering

A geometry could be written down three times: in the entry point's name, in the check the entry
point rejects a mismatch on, and again in whatever caller decided it was allowed to make the call.
The third copy is the one that drifts, so the constraints are DATA rather than prose, and they live
where the answers are used rather than in this library's own registry.

`libr4d/r4d_rows.cpp` carries one `RadKernelInfo` row per kernel, with its geometry written as
`RadConstraint`s as well as in prose, and the engine's selector reads those rows to decide which
kernel serves a band — so a caller never has to know a geometry in order to ask for one.
`rad-kbench` tests the rows against the reference kernels, and `r4d_selftest` calls every launch
with a minimal valid input, which is how a transcription error in a constraint is found.

The kernels themselves are plain HIP behind the C ABI in [`r4d.h`](r4d.h) and know nothing about
any of that.

## Building

There is no build here. `libr4d/CMakeLists.txt` is it -- see the engine's `build.sh`.

Translation units are compiled separately and linked, so per-kernel target features stay scoped:
`-mcumode` applies to `r4d_gdn_chunk_scan_k128_v128_c64_bf16.o` alone, and `-ffp-contract=off` is
global because the rotated 6-bit all-reduce requires it (the two ranks fuse different products
otherwise and diverge by ~1 ULP).

## Adding a kernel

1. Write the `.hip`; name the entry point for what it computes and the geometry it is compiled for.
2. Declare it in `r4d.h`, and add a `_dims()` accessor if it has compiled-in geometry a caller
   should be able to read.
3. Name the file after the entry point, geometry included; only shared machinery (`r4d_common.h`,
   `r4d_dt16.h`, `r4d_gdn_wmma.h`) is named for what it is, because it provides no kernel.
4. Add the translation unit to `R4D_UNITS` in `libr4d/CMakeLists.txt`, with any per-unit flags.
5. Add a `RadKernelInfo` row in `libr4d/r4d_rows.cpp` and a shim beside it -- that row is what
   the engine's selector reads, so a kernel that is not in it is one no model can resolve to. Give
   it the `op` of the operation it implements, which it may share with other rows, and write its
   geometry as constraints as well as prose.
6. Add a case to `libr4d/r4d_selftest.cpp` so the row is called at least once.

## Notes for anyone reading the kernels

Three properties of this architecture are assumed throughout and are not restated per kernel:

* The wave32 WMMA fragment layout is `idx = lane % 16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`. Both
  A and B want K-contiguous rows, so swapping the operands of a WMMA transposes the result for free.
* gfx1201 has no `v_cvt_pk_bf16_f32`, no direct-to-LDS, and no `ds_read_b64_tr_b16`. Packing bf16 is
  software, and the cheapest form is two adds plus a `v_perm_b32`.
* LLVM single-buffers LDS and drains before every WMMA unless you tell it not to.
  `__builtin_amdgcn_sched_group_barrier` is the fix, and the granularity of the groups matters far
  more than their contents -- coarser is better, up to the point where the pattern asks for more
  outstanding loads than the hardware can hold.
