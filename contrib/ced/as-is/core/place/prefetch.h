/* prefetch.h -- CED (Conditional / Clustered Expert Distribution) Projections
 * for Predictive MoE Prefetching in Radiance.
 *
 * ---- THE PROBLEM: PCIE LATENCY VS GFX1201 LAYER COMPUTE ---------------------------------------
 *
 * In a 512-expert MoE architecture (such as Qwen3.8-Flash-Next at --tp 2), an offloaded expert chunk
 * (~2.5–3.5 MiB) takes ~40–60 µs to transfer over PCIe Gen5 x16 (effective DMA ~50–55 GB/s).
 * On AMD RDNA4 (gfx1201, 64 CUs, Dual SIMD Wave32), an optimized decoder layer executes in only
 * ~10–18 µs.
 *
 * A reactive placement engine (relocating between steps or upon a cache miss) produces severe
 * bubbles: the compute stream must stall for ~40–50 µs while the missing expert is pulled across
 * the bus.
 *
 * ---- THE MECHANISM: EARLY CED LOOKAHEAD (ΔL >= 4–8 LAYERS) -----------------------------------
 *
 * Early layers (layers 0–23 or early linear attention/GDN blocks) process hidden state h_t.
 * By evaluating a lightweight low-rank Conditional / Clustered Expert Distribution (CED) projection
 * \hat{P}^{l+\Delta L}(e | h_t), the engine estimates the routing distribution 4 to 8 layers ahead.
 *
 * Over this 50–90 µs window (4–6 layers of intervening compute), the prefetcher issues asynchronous
 * DMA (hipMemcpyAsync or AQL packet) on the mover transfer stream (avoiding creating a 4th hardware
 * queue that causes RDNA4 queue performance cliffs). The predicted expert arrives in VRAM before the
 * compute stream reaches layer l + ΔL.
 *
 * ---- LEASE SAFETY: THE 30-DISPATCH HOLD-DOWN LEASE -------------------------------------------
 *
 * In core/place/heat.cpp, newly promoted units receive a 30-dispatch hold-down lease
 * (min_residency_dispatches = 30). CED prefetching guarantees:
 *   1. Prefetched units only overwrite FREE slab slots or slots with EXPIRED leases (>= 30 dispatches).
 *   2. Units with ACTIVE leases (< 30 dispatches) are strictly protected. If no free slot or expired
 *      lease exists, prefetch safely throttles (prefetch_skipped_leased) to prevent thrashing.
 *   3. Prefetched units receive an immediate hold-down lease upon prefetch registration.
 *
 * ---- PRECISION & TP2 BANDWIDTH PROTECTION ----------------------------------------------------
 *
 * Over-prefetching saturates PCIe and contends with TP2 All-Reduce over --tp-wire wht6.
 * Candidates are filtered by probability threshold P(e) > tau (e.g. 0.15) and top-k cluster
 * (max 1–2 candidates per lookahead dispatch), enforcing strict PCIe transfer quotas.
 */
#pragma once

#include "../rad_core.h"
#include "../runtime/stager.h"
#include "../runtime/rad_ridgefill_wrapper.h"
#include "mover.h"
#include "heat.h"
#include "planner.h"
#include "ced_gemv.h"
#include <arch/rad_rdna4_isa.h>

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>

namespace rad {

/* ------------------------------------------------------------------ CED Config & Stats */

struct CedConfig {
    bool    enabled = true;
    int     lookahead = 4;           /* ΔL: lookahead layers ahead (default 4, range 4..8) */
    float   threshold = 0.15f;       /* tau: probability threshold for candidate filtering */
    int     top_k = 2;               /* max candidates to prefetch per lookahead dispatch */
    int64_t max_prefetch_bytes = 8 * 1024 * 1024; /* max bytes per lookahead window (8 MiB) */
    int     rank = 32;               /* low-rank projection bottleneck dimension r */
    int     n_clusters = 32;         /* clustered expert distribution clusters */
    int     n_experts = 512;         /* total expert count */
    int     hidden_dim = 2560;       /* hidden state dimension d */
    bool    reuse_ridgefill = false; /* bridge mode: reuse RidgeFill projector weights when L >= 24 */
    int     cluster_top_k = 4;       /* top clusters retained for clustered expert distribution */
    bool    filter_by_cluster = true;/* whether to enforce top-k cluster filtering */
};

struct CedStats {
    uint64_t predictions = 0;
    uint64_t prefetch_issued = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_skipped_resident = 0;
    uint64_t prefetch_skipped_leased = 0;
    uint64_t prefetch_bytes = 0;
    uint64_t pcie_throttled = 0;
    uint64_t cluster_filtered = 0;
};

struct CedCandidate {
    int32_t expert_id = -1;  /* expert index in layer (0..n_experts-1) */
    int32_t layer = -1;      /* target layer (l + ΔL) */
    int32_t unit_id = -1;    /* global unit index in Plan */
    float   probability = 0.0f;
};

/* ------------------------------------------------------------------ CED Projector */

class CedProjector {
public:
    CedProjector() = default;

    /* Initialize low-rank matrices or clustered centroid maps */
    int init(const CedConfig& cfg);

    /* Dedicated low-rank CED projection:
     *   z = V * h_t  (V: r x d)
     *   logits = U * z + b  (U: E x r)
     *   probs = Softmax(logits)
     * Aggregates cluster mass and filters candidates where P(e) > tau in top active clusters. */
    int project(int target_layer, const float* h_t, int d,
                std::vector<CedCandidate>& out_candidates,
                float tau, int top_k) const;

    /* RidgeFill bridge projection:
     * When RidgeFill is loaded and target_layer >= 24, reuses the INT8/BF16 RidgeFill projector
     * matrix W_RF: R^10240 -> R^2560, followed by layer MoE gate projection to R^512. */
    int project_with_ridgefill(int target_layer, const float* h_t, int d,
                               const ridgefill::ProjectorBundle* bundle,
                               const float* gate_weights,
                               std::vector<CedCandidate>& out_candidates,
                               float tau, int top_k) const;

    /* RDNA4 Wave32 device GEMV launch */
    int project_device(const float* d_h_t, float* d_out_logits, void* stream) const;

    /* Low-rank matrix accessors */
    const std::vector<float>& down_proj() const { return v_down_; }
    const std::vector<float>& up_proj() const { return u_up_; }
    const std::vector<float>& bias() const { return bias_; }
    int rank() const { return cfg_.rank; }

    /* Set custom trained projection weights */
    void set_weights(const std::vector<float>& v_down,
                     const std::vector<float>& u_up,
                     const std::vector<float>& bias);

    /* Cluster mapping for clustered expert distribution */
    int cluster_of_expert(int expert) const {
        if (expert < 0 || expert >= cfg_.n_experts || cluster_map_.empty()) return 0;
        return cluster_map_[(size_t)expert];
    }

    void compute_cluster_mass(const std::vector<float>& probs, std::vector<float>& cluster_mass) const;

private:
    CedConfig cfg_;
    std::vector<float> v_down_;    /* [rank x hidden_dim] */
    std::vector<float> u_up_;      /* [n_experts x rank] */
    std::vector<float> bias_;      /* [n_experts] */
    std::vector<int>   cluster_map_; /* [n_experts] -> cluster id */
};

/* ------------------------------------------------------------------ CED Prefetcher */

class CedPrefetcher {
public:
    CedPrefetcher() = default;
    ~CedPrefetcher() = default;

    int init(const Program& prog, const Plan& plan, Mover* mover, HeatEngine* heat,
             const CedConfig& cfg);

    void set_ridgefill(const ridgefill::ProjectorBundle* bundle) {
        rf_bundle_ = bundle;
    }

    /* Invoked at layer l during forward pass with hidden state h_t:
     * Estimates \hat{P}^{l+\Delta L}(e | h_t), checks residency and hold-down leases,
     * and pipelines asynchronous DMA for candidate experts before compute arrives at l + ΔL. */
    int on_layer_dispatch(int current_layer, const float* hidden_state, RadStream stream = nullptr);

    /* Invoked at step completion / tiering_dispatch:
     * Dispatches CED cluster-predictive prefetch for layer l + ΔL based on active
     * expert routing observed in layer l. */
    int on_step_routing(int current_layer, const int32_t* counts, int32_t n_experts);

    /* Synchronous evaluation helper for tests and verification */
    int evaluate_and_prefetch(int current_layer, const float* hidden_state);

    bool  enabled() const { return cfg_.enabled; }
    const CedStats&  stats()  const { return stats_; }
    const CedConfig& config() const { return cfg_; }
    void  set_config(const CedConfig& cfg) { cfg_ = cfg; }
    void  reset_stats();
    std::string report() const;

    /* Unit mapping queries */
    int32_t find_unit_id(int32_t layer, int32_t expert) const {
        auto it = expert_to_unit_.find({layer, expert});
        return it != expert_to_unit_.end() ? it->second : -1;
    }

    const CedProjector& projector() const { return projector_; }
    CedProjector&       projector()       { return projector_; }

private:
    CedConfig   cfg_;
    CedStats    stats_;
    const Program* prog_ = nullptr;
    const Plan*    plan_ = nullptr;
    Mover*         mover_ = nullptr;
    HeatEngine*    heat_ = nullptr;
    const ridgefill::ProjectorBundle* rf_bundle_ = nullptr;

    CedProjector projector_;

    /* Map (layer, expert) -> global MoveUnit index in Plan */
    std::map<std::pair<int32_t, int32_t>, int32_t> expert_to_unit_;
    int total_layers_ = 48;
};

/* ------------------------------------------------------------------ RDNA4 Wave32 Kernel Standards
 * The following constants and structural contracts govern the execution of CED projection
 * kernels on AMD RDNA4 (gfx1201):
 *   Rule 1: VGPR <= 96 for 100% occupancy (16 waves/SIMD)
 *   Rule 2: Compile-time constant bounds, zero waterfall loops
 *   Rule 3: Dual-issue VOPD destination parity
 *   Rule 5: TemporalHint::NT_RT for streaming projection matrices, LU for intermediate activations
 *   Rule 6: Independent hardware wait counters (s_wait_loadcnt, s_wait_dscnt)
 *   Rule 7: Epilogue early deallocation (s_sendmsg MSG_DEALLOC_VGPRS) before wait
 */
constexpr int kCedWaveSize = 32;
constexpr int kCedWorkgroupThreads = 128; // 4 Wave32 waves
constexpr int kCedMaxVgprs = 96;          // Strict upper bound for 100% SIMD occupancy
constexpr int kCedMaxLdsBytes = 4096;     // 4 KiB LDS per workgroup (within 64 KiB CU LDS)

}  /* namespace rad */
