# Radiance Candidate Levers Catalog (Opt 146 & Production Recovery)

**Target Baseline**: Radiance Opt 146 (`1a4455d26871100a6a97e1dd3b5c83d4431764b6`) / ARM1–ARM14  
**Hardware Environment**: Dual AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 64 CUs, Wave32, 576 GB/s GDDR6, PCIe 4.0 x16 `--tp-wire wht6`)  
**Target Architecture**: Qwen3.8-Flash-Next / Qwen4Exp (113B MoE / 512 Experts, Top-10 Routing, GDN Linear Recurrence, MTP Speculative Decoding)  
**Package Repository**: `/home/orca/RadianceMine/candidates/`  

---

## 1. Executive Summary

This directory provides a cleanly isolated, fully packaged, and standalone repository of the 12 primary optimization levers excavated across the full project trajectory from ARM1 to ARM14 up to OPT191.

Every candidate lever is physically isolated in its own self-contained folder containing:
1. **`README.md`**: Architectural and mathematical principles, hardware latency equations, safety gating flags, and step-by-step verification commands.
2. **`lever.patch`**: Standalone unified patch diff generated and verified directly against clean code. Can be applied via `git apply lever.patch`.
3. **`src/`**: Standalone source files for inspection, side-by-side diffing, or cherry-picking.

---

## 2. Master Candidate Lever Matrix (12 Levers)

| Lever ID & Directory | Target Subsystem | Innovation & Mathematical Principle | Historical Demonstrated Gain | Gating Flag / Safeguard | Priority Tier |
|---|---|---|---|---|:---:|
| **[Lever 1](file:///home/orca/RadianceMine/candidates/lever1_gdn_qsa_fusion/)** | GDN & Full Attention | Unified $[q\|k\|v\|z\|a\|b]$ in-projection GEMM ($8240 \times 2560$) + Fused `op_qprep`. | **$-14.0\,\text{ms/step}$** decode latency reduction | Default safe (layer layout fusion) | **Tier 1** |
| **[Lever 2](file:///home/orca/RadianceMine/candidates/lever2_fused_moe_router/)** | MoE Routing | Single-launch router GEMM + Top-10 counting sort (`router_gemm_topk_scatter`). | Eliminates **96 dispatches/step** | `R4D_ROUTER_FUSED=1` (default `0`) | **Tier 2** |
| **[Lever 3](file:///home/orca/RadianceMine/candidates/lever3_qsa_offload_kvarn/)** | KV-Cache Compression | Involutory Fast Walsh-Hadamard (FWHT-256) rotation for 1:1 INT4 KV cache without outliers. | **$4\times$ KV compression**, enables 64k+ context | `--kvarn` (`RADIANCE_KVARN=1`) | **Tier 2** |
| **[Lever 4](file:///home/orca/RadianceMine/candidates/lever4_fr_spec_draft_head/)** | MTP Speculative Drafter | FR-Spec reduced-vocabulary draft head (52k tokens, $>99.2\%$ token coverage). | **$3\times$ faster draft head** ($1.8 \to 0.58\,\text{ms}$) | Auto-discovery of safetensors | **Tier 1** |
| **[Lever 5](file:///home/orca/RadianceMine/candidates/lever5_adaptive_bandit_controller/)** | Speculative Controller | SOTA Multi-Armed Bandit (LCB) 4-pillar controller ($K \in [1, 6]$). | **$+15\%$ to $+28\%$ TPS** on mixed entropy | `RADIANCE_ADAPTIVE_SPEC=1` | **Tier 2** |
| **[Lever 6](file:///home/orca/RadianceMine/candidates/lever6_fast_store_drain_p2p_ar/)** | Dual-GPU P2P AR | `s_wait_storecnt 0x0` fast hardware store drain replacing `__threadfence_system()`. | **$+180\,\text{tok/s}$ C8 Concurrency** ($262 \to 451+\,\text{t/s}$) | Fine-grained PCIe memory invariant | **Tier 1** |
| **[Lever 7](file:///home/orca/RadianceMine/candidates/lever7_swizzled_prefill_gemm_vopd/)** | Prefill W4A8 GEMM | Swizzled prefill GEMM Arm 34 with Wave32 dual-issue VOPD, b128 A-loads & conflict-free LDS. | **$+2,500\,\text{tok/s}$ Cold Prefill** (enabled 9,353 t/s peak) | Default tile arm 34 in `tGemmPrefill` | **Tier 1** |
| **[Lever 8](file:///home/orca/RadianceMine/candidates/lever8_stager_wrap_prefetch/)** | Prefill Host Staging | Stager chunk-to-chunk wrap prefetch (`start_wrap`) in 3-buffer ring pipeline. | **$+2,311\,\text{tok/s}$ Cold Prefill** (+62.8% throughput) | Bound to `is_chunked_prefill_` | **Tier 1** |
| **[Lever 9](file:///home/orca/RadianceMine/candidates/lever9_4way_rlfu_vram_cache/)** | PLE N-Gram Cache | 4-Way Set-Associative RLFU L1 VRAM Cache with 1-wave-per-row `__shfl` dispatch. | **$116,472\,\text{tok/s}$ Cached Prefill** ($16.9\times$ speedup) | Uses 85 MiB unclaimed VRAM slack | **Tier 2** |
| **[Lever 10](file:///home/orca/RadianceMine/candidates/lever10_moe_decode_grid_clamp_wave0/)** | MoE Decode Kernel | Decode launch grid clamping `gx = (tiles<=1) ? T : tiles*16u` + Wave-0 binary search. | **$+18\,\text{tok/s}$ Decode**, $-340\,\mu\text{s/step}$ | Active when $T \le 16$ | **Tier 1** |
| **[Lever 11](file:///home/orca/RadianceMine/candidates/lever11_pipelined_63vgpr_hc_write/)** | HyperConnections & AR | Pipelined 63-VGPR double buffering for HC write & vectorized `uint4` gathering. | Preserves **Tier-64 occupancy** (8 waves/CU) | Hardware register constraint | **Tier 1** |
| **[Lever 12](file:///home/orca/RadianceMine/candidates/lever12_zero_cpu_wait_pipelining/)** | CPU-GPU Overlap | Zero-CPU-wait pipelining overlap (`ahead != nullptr`) + Recurrent slot scrubbing race fix. | **$+115\,\text{tok/s}$ C8 Concurrency**, eradicates `bdbdbd` loop | Moves `state_clears()` before `step_next` | **Tier 1** |

---

## 3. Recommended Sequential Integration Order

1. **Step 1 (Safe Micro-Wins & Synchronization Restoration)**:
   - Apply **Lever 6** (`s_wait_storecnt 0x0` P2P AR drain).
   - Apply **Lever 12** (Pipelining overlap + Recurrent scrubbing race fix).
   - Apply **Lever 10** (MoE decode grid clamping & Wave-0 binary search).
   - Apply **Lever 11** (Pipelined 63-VGPR HC write).
2. **Step 2 (Prefill Pipeline & GEMM Acceleration)**:
   - Apply **Lever 7** (Swizzled prefill GEMM Arm 34 with VOPD).
   - Apply **Lever 8** (Stager wrap prefetch `start_wrap`).
   - Apply **Lever 9** (4-Way RLFU L1 VRAM cache).
3. **Step 3 (Trunk Layer & Architecture Fusion)**:
   - Apply **Lever 1** (GDN in-projection concatenation & QSA `op_qprep` fusion).
   - Apply **Lever 4** (FR-Spec 52k reduced-vocabulary draft head).
4. **Step 4 (Advanced Dynamic Control & Compression)**:
   - Apply **Lever 5** (SOTA MTP adaptive bandit controller).
   - Apply **Lever 2** (Fused MoE router with single-workgroup top-k sort).
   - Apply **Lever 3** (KVarN 1:1 INT4 KV compression & QSA PCIe offloading).
