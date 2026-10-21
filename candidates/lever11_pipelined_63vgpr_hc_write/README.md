# Lever 11 (review material): pipelined double-buffered HC write

## Status in this target

- **Port status: ported to `pr/hc-write-pipeline`.** The change is in `libr4d/r4d_ar_hc.h`,
  function `r4d_ar_hc_write_word`.
- **Byte-exactness on the target: PASS.** `verify/` builds the base commit's header and this
  tree's header into two kernels of identical launch shell and compares the whole output buffer
  byte for byte on a real gfx1201 card (cards 0 and 1), 42 shape cases plus guards, input
  immutability, a known-value write and a negative control. See "Evidence" below.
- **Tree oracle: PASS.** `r4d_selftest --case ar_hc_write` passes before and after the change,
  including the `hc=2` and `hc=4` shapes it already covered.
- **Register usage: measured.** The 63-VGPR figure in `source_README.md` is the source candidate's
  claim and does **not** reproduce on this toolchain. Real numbers are in "Resource usage".
- **Throughput: NOT RUN.** No engine-level benchmark, profiler trace or bandwidth measurement was
  made for this change. The `+8%..+12%` bandwidth and the `392 -> 485 GB/s` table in
  `source_README.md` are source claims, not measurements here.

## Files here

- `lever.patch`: the original patch, byte-identical to the source candidate's. Its base blob
  (`b845120`) is the source tree's `libr4d/r4d_ar_hc.h`, which differs from this target's
  version of that function, so the patch does not apply to this tree. It is review material
  only, not an instruction to run it. Apply the port from this branch instead.
- `source_README.md`: the source candidate's README, unedited except that the absolute path in
  its apply snippet is now repo-relative. Every figure and every kernel name in it belongs to
  the source tree. It documents two further kernels (`hc_write_vec_kernel`,
  `gather_vec_kernel`) that this target does not have and that this port does not bring over.
  Its `git apply` line targets the source tree and does not apply here.
- `verify/`: the on-card byte-comparison harness (`hc-baseline.hip`, `hc-candidate.hip`,
  `compare.hip`, `run.sh`) and `mutation.sh`, which mutates the base header and requires the
  comparison to reject it.

## The ported change, in this target

Before (`r4d_ar_hc_write_word`, base commit):

```cpp
for (int c = 0; c < hc; ++c) {
  const float g = bf2f(inj[(long)row * irow + c]);
  int4* hp = reinterpret_cast<int4*>(h + (long)row * hrow + (long)c * n + i0);
  int4 hv = *hp;
  unsigned short* he = reinterpret_cast<unsigned short*>(&hv);
  for (int j = 0; j < 8; ++j) he[j] = f2bf(bf2f(he[j]) + g * v[j]);
  *hp = hv;
}
```

After:

```cpp
if (hc <= 0) return;
float g_curr = bf2f(inj[(long)row * irow]);
int4 hv_curr = *reinterpret_cast<const int4*>(h + (long)row * hrow + i0);
for (int c = 0; c < hc; ++c) {
  const bool more = c + 1 < hc;
  float g_next;
  int4 hv_next;
  if (more) {
    g_next = bf2f(inj[(long)row * irow + c + 1]);
    hv_next = *reinterpret_cast<const int4*>(h + (long)row * hrow + (long)(c + 1) * n + i0);
  }
  unsigned short* he = reinterpret_cast<unsigned short*>(&hv_curr);
  for (int j = 0; j < 8; ++j) he[j] = f2bf(bf2f(he[j]) + g_curr * v[j]);
  *reinterpret_cast<int4*>(h + (long)row * hrow + (long)c * n + i0) = hv_curr;
  if (more) { g_curr = g_next; hv_curr = hv_next; }
}
```

- Load scheduling only. The per-stream expression, its order, the dtype and every intermediate
  format are identical: the same `g = bf2f(inj[...])`, the same `f2bf(bf2f(he[j]) + g * v[j])`,
  the same one 128-bit store per stream in ascending `c`, the same store addresses.
- The `hc <= 0` early return keeps the "no memory touched, nothing stored" boundary; the
  `c + 1 < hc` guard bounds the prefetch and the carry, so the last stream neither loads nor
  copies a dead next value.
- No ABI change, no signature change, no caller edit.

## Callers

- `libr4d/r4d_ar_oneshot_2rank_exact.hip`
- `libr4d/r4d_ar_oneshot_2rank_wht6.hip`
- `libr4d/r4d_ar_oneshot_Nrank_exact.hip`

## Evidence

`bash candidates/lever11_pipelined_63vgpr_hc_write/verify/run.sh <card>`, gfx1201, this
toolchain (`/opt/rocm/llvm/bin/clang++`, `-O3 -ffp-contract=off`):

```
hc-baseline compile rc=0
hc-candidate compile rc=0
compare compile rc=0
link rc=0
device=0 arch=gfx1201 baseline=hc_baseline candidate=hc_candidate
PASS cases=42 byte_exact guards=PASS immutable_inputs=PASS known_write=PASS negative_control=PASS
compare rc=0
```

Same output on card 1 (`device=1`, exit 0). The 42 cases are rows 1/4 x n 8/24/2560 x
hc -1/0/1/2/4/8/16; the negative control changes one injection value and requires the buffers to
differ.

`verify/mutation.sh` mutates the base header so the last stream is skipped when `hc > 1`, and
requires the comparison to fail. It does:

```
== mutated base: last stream not written when hc > 1; comparison must FAIL
FAIL candidate differs from independent baseline header
compare rc=1
mutated run rc=1 (expected nonzero)
== pass: the comparison rejected the mutated header
```

`r4d_selftest <libr4d.so> --case ar_hc_write`, base header and ported header, both exit 0:

```
ok    ar_hc_write M=8 n=1024 hc=4: every rank holds the folded y and the written h of the unfused chain, byte for bit
ok    ar_hc_write M=64 n=2560 hc=2: every rank holds the folded y and the written h of the unfused chain, byte for bit
```

## Resource usage

Measured with this tree's own compile command per one-shot unit plus
`-Rpass-analysis=kernel-resource-usage`, `gfx1201`, `-O3` (target numbers, not source claims):

| kernel | base VGPR | ported VGPR | occ [waves/SIMD] | spills |
|---|---|---|---|---|
| `r4d_ar_hc_write_kernel` (2rank exact) | 36 | 30 | 16 -> 16 | 0 |
| `r4d_ar_gather_hc_write_kernel<false>` | 43 | 43 | 16 -> 16 | 0 |
| `r4d_ar_gather_hc_write_kernel<true>` | 47 | 47 | 16 -> 16 | 0 |
| `r4d_ar_hc_write_wht6_kernel<false,8>` | 68 | 69 | 16 -> 16 | 0 |
| `r4d_ar_hc_write_wht6_kernel<true,8>` | 87 | 87 | 16 -> 16 | 0 |
| `r4d_ar_hc_write_wide_kernel<4>` | 38 | 38 | 16 -> 16 | 0 |
| `r4d_ar_gather_hc_write_wide_kernel<4,false>` | 43 | 43 | 16 -> 16 | 0 |
| `r4d_ar_gather_hc_write_wide_kernel<4,true>` | 47 | 47 | 16 -> 16 | 0 |

No kernel changes its occupancy tier and none spills. The largest kernel in each unit is
unchanged (2rank exact 88, wht6 127, Nrank exact 76), so the write is not the register-pressure
limiter anywhere. The source's "88 -> 63 VGPR" story does not apply to this tree: the base write
loop here costs 36 VGPRs, not 88, and the ported form 30.

**Not measured:** wall time, effective bandwidth, achieved occupancy at run time, and any
engine-level effect. This change alters load scheduling only; whether it helps on the target is
unproven.

## Building on this branch

```bash
cmake --build build_hip --target libr4d -j"$(nproc)"
```

The port is already in the working tree of `pr/hc-write-pipeline`; nothing else to apply.
