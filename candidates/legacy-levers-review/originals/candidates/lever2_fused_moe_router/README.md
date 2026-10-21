# Lever 2: Fused MoE Router (`op_router_fused` / `router_gemm_topk_scatter`)

**Status**: Verified Candidate Lever  
**Target Commit**: Opt 146 (`1a4455d26871100a6a97e1dd3b5c83d4431764b6`)  
**Source Branch**: `feat/task2_fused_router` (commit `38e870d2fb12b1892165e1aae9b08f184ef706db`) & `moe_pipeline_fusion_blueprint.md`  
**Kernel Name**: `router_gemm_topk_scatter_bf16`  
**Gating Flag**: `R4D_ROUTER_FUSED` (default `0` = safe opt-in)

---

## 1. Architectural Purpose

In the standard decode pass across all 48 MoE layers of Qwen3.8 / Qwen4, routing requires three sequential, highly dependent HIP kernel launches per layer:
1. `op_router`: Router GEMM $[M \times 2560] \times [2560 \times 512]^T \to \text{logits} [M \times 512]$
2. `op_sgate`: Shared Expert Gate $[M \times 2560] \times [2560 \times 1]^T \to \text{sgate} [M \times 1]$
3. `op_topk_scatter`: Top-10 Softmax selection, routing weights $w$, and token slot scatter (`sorted_tok`, `expert_offset`, `expert_count`)

Across 48 layers, this consumes **144 individual kernel dispatches** per decode step, saturating the Command Processor (CP) queue and incurring repeated L2 cache invalidations and round-trips.

**Lever 2 integrates the router GEMM, shared gate GEMM, and top-$k$ counting sort into a single fused dispatch (`router_gemm_topk_scatter`)**, reducing 144 dispatches to 48 dispatches at decode ($M \le 64$).

---

## 2. Autopsy: Why Upstream Fused Router Suffered Atomic Arrival Stalls

In upstream v1.2.0 (commit `0eeee5b`), an initial prototype of `router_gemm_topk_scatter` was benchmarked in Opt 189. It exhibited catastrophic performance collapse:
- **C8 Concurrency**: Dropped from **419 tok/s to 240 tok/s** ($-42.7\%$ regression).
- **64k Prefill**: Dropped from **5993 tok/s to 3659 tok/s** ($-38.9\%$ regression).
- Forced upstream maintainers to immediately gate the kernel behind `R4D_ROUTER_FUSED=0`.

Detailed architectural analysis on AMD RDNA4 Wave32 (`gfx1201`) reveals the four fatal failure mechanisms:

### 2.1 Failure Mechanism A: Global Device-Wide Atomic Arrival Counter
In `libr4d/r4d_moe.hip` (lines 3926–3938):
```cpp
__threadfence();
__syncthreads();
__shared__ unsigned int s_arrived;
if (tid == 0)
    s_arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
__syncthreads();
if (s_arrived + 1u != gridDim.x * gridDim.y) return;
if (tid == 0) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
__threadfence();

/* router_topk_scatter_kernel executed ONLY by the last arriving block */
```

1. **Agent-Scope Serialization Across 64 CUs**:
   The grid dimension had 16 to 32 blocks. Every block finished its GEMM tile and hammered a single global memory address `ctr` using `__HIP_MEMORY_SCOPE_AGENT`. On RDNA4 (gfx1201), agent-scope atomics traverse the crossbar/Infinity Fabric directly to the L2 cache controllers. Severe contention from 32+ workgroups caused complete L2 pipeline serialization.
2. **Double Device-Wide Memory Fences (`__threadfence`)**:
   `__threadfence()` on RDNA4 forces all vector write buffers to drain to L2 and invalidates all Vector L1 (TCP) caches across the entire GPU. Calling this before and after the counter increment completely stalled the execution pipeline.
3. **GPU Occupancy Collapse & Severe Tail Stalls**:
   31 of the 32 workgroups hit early termination (`if (s_arrived + 1u != gridDim.x * gridDim.y) return;`) and immediately exited. Whichever workgroup arrived last was left entirely alone on a single CU to execute the Top-10 selection and small sort (`scatter_small_place`).
   **GPU occupancy instantly collapsed from 64 CUs down to 1 CU (1.5% occupancy)**! If 31 workgroups completed in $4\,\mu\text{s}$ but one workgroup was delayed by $3\,\mu\text{s}$ due to memory bank collision or memory refresh, the entire GPU sat completely idle waiting for that single workgroup to arrive, and then waited another $5\,\mu\text{s}$ for that single workgroup to complete the sort.

### 2.2 Failure Mechanism B: False Fusion (Logits Round-Tripped Through VRAM Anyway)
Because the upstream GEMM split the 512 experts across multiple column workgroups (`nrb`), no single workgroup possessed all 512 logits for a given token. Every workgroup was forced to write its partial logits out to global VRAM (`lg`), and then the last arriving block had to read all 512 logits back from VRAM over the memory bus. **Zero memory bandwidth was saved**.

### 2.3 Failure Mechanism C: Fragile Sort Limit ($T \le 1024$)
The scatter logic was hardcoded to `kScatterSmall = 1024`. For $\text{top\_k} = 10$, $T = M \times 10 \le 1024 \implies M \le 102$ tokens. During prefill chunks where $M > 102$, the single-workgroup scatter failed or fell back to dynamic allocation routines, inducing pipeline stalls.

---

## 3. The Solution: Architectural Blueprint for Clean Atomic-Free Router Fusion

To eliminate the atomic arrival stall and unlock full theoretical speedup on RDNA4, the implementation must adhere to the following architectural design:

```
+-----------------------------------------------------------------------------------------+
|                  CLEAN FUSED ROUTER ARCHITECTURE (gfx1201, Wave32)                      |
|                                                                                         |
| Phase 1: GEMM (Parallel over 16 WGs on 16 WGPs / 32 CUs)                                |
|   - Workgroup b computes 32 logits per token in LDS (32 experts * 16 tokens * 4B = 2KB) |
|   - Workgroup 0 computes shared gate scalar                                             |
|   - Each WG extracts local Top-10 candidates in LDS (zero VRAM writes)                   |
|   - Zero global atomics, zero agent-scope counters, zero __threadfence()                |
|                                                                                         |
| Phase 2: Distributed Top-10 Reduction & DPP Bitonic Merge                               |
|   - Decode fast-path (M in [1, 4]): Single workgroup computes full GEMM + Top-K in LDS  |
|   - Multi-token (M in [5, 16]): Two-stage hierarchical reduction                        |
|   - Single Wave32 performs final Top-10 reduction across candidates in 28 clock cycles  |
+-----------------------------------------------------------------------------------------+
```

### Key Principles:
1. **Zero Global Atomics**: No shared atomic counter `ctr`. All synchronizations are local to workgroups (`__syncthreads()`) or handled via clean multi-stream event dependencies.
2. **In-LDS Logit Accumulation**: For $M \le 16$, logits reside strictly in LDS ($<8\,\text{KiB}$ total allocation), completely eliminating the VRAM round-trip.
3. **Single-Workgroup Decode Fast-Path ($M \le 4$)**: For decode steps with small batch or speculative drafts ($M \in [1, 4]$), dispatch a single 8-wave workgroup (256 threads) that holds all weights in L1 and performs the entire GEMM and Top-10 selection without any cross-block synchronization.

---

## 4. Safety Gating & Environment Flags

Lever 2 is completely opt-in and safe by default:
- **`R4D_ROUTER_FUSED=0`** (or unset): Default safe mode. Retains separate `op_router`, `op_topk_scatter`, and `op_sgate` kernel launches.
- **`R4D_ROUTER_FUSED=1`**: Enables fused router GEMM + Top-10 scatter + shared gate in a single launch.
- **`R4D_ROUTER_FUSED=2`**: Enables fused router GEMM + Top-10 scatter in a single launch, leaving shared gate separate.

The runtime verifies bit-exact numerical parity of all outputs (`logits`, `sgate`, `expert_ids`, `expert_w`, `sorted_tok`, `expert_offset`, `expert_count`).

---

## 5. Exact Diff Against Opt 146

```
 arch/common/rad_block_moe_fp8.h | 112 +++++++++++++-
 docs/OPS.md                     |  12 ++
 libr4d/r4d_moe.hip              | 325 ++++++++++++++++++++++++++++++++++++++++
 libr4d/r4d_plugin.h             |   6 +
 libr4d/r4d_rows.cpp             |  49 ++++++
 libr4d/r4d_selftest.cpp         | 195 ++++++++++++++++++++++++
 libr4d/r4d_shapes.cpp           |  27 ++++
 tests/arch_test.cpp             | 115 ++++++++++++++
 8 files changed, 837 insertions(+), 4 deletions(-)
```

---

## 6. Application Instructions

To apply this candidate lever directly to a clean Opt 146 checkout:

```bash
cd /path/to/radiance
git apply /home/orca/projects/radiance_candidates_for_opt146/lever2_fused_moe_router/lever.patch
```

To build and verify:

```bash
# Configure & compile
cmake --build build_hip --target arch_test -j$(nproc)

# Run the unit test validating single-dispatch execution at decode
./build_hip/tests/arch_test qwen4exp_fused_router_single_dispatch_at_decode
```

Expected output:
```
  ok   qwen4exp_fused_router_single_dispatch_at_decode
76220 check(s) over 40 case(s)
```
