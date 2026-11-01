/* prefetch.cpp -- CED (Conditional / Clustered Expert Distribution) Projections
 * for Predictive MoE Prefetching implementation.
 */
#include "prefetch.h"
#include "rad_internal.h"
#include "rad_plugin.h"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rad {

/* ------------------------------------------------------------------ CedProjector */

int CedProjector::init(const CedConfig& cfg) {
    cfg_ = cfg;
    const int r = cfg_.rank;
    const int d = cfg_.hidden_dim;
    const int E = cfg_.n_experts;

    if (r <= 0 || d <= 0 || E <= 0) return RAD_E_INVAL;

    v_down_.resize((size_t)(r * d), 0.0f);
    u_up_.resize((size_t)(E * r), 0.0f);
    bias_.resize((size_t)E, 0.0f);

    /* Generate normalized orthogonal projection basis for reproducible lookahead routing */
    for (int k = 0; k < r; ++k) {
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            float val = std::sin(static_cast<float>((k + 1) * (j + 1)) * 0.0174532925f);
            v_down_[(size_t)(k * d + j)] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (int j = 0; j < d; ++j) v_down_[(size_t)(k * d + j)] /= norm;
        }
    }

    for (int e = 0; e < E; ++e) {
        float norm = 0.0f;
        for (int k = 0; k < r; ++k) {
            float val = std::cos(static_cast<float>((e + 1) * (k + 1)) * 0.034906585f);
            u_up_[(size_t)(e * r + k)] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (int k = 0; k < r; ++k) u_up_[(size_t)(e * r + k)] /= norm;
        }
        bias_[(size_t)e] = -0.5f;
    }

    /* Assign experts to clusters (e.g. 512 experts into 32 clusters of 16 experts each) */
    cluster_map_.resize((size_t)E, 0);
    int cluster_size = std::max(1, E / std::max(1, cfg_.n_clusters));
    for (int e = 0; e < E; ++e) {
        cluster_map_[(size_t)e] = std::min(cfg_.n_clusters - 1, e / cluster_size);
    }

    return RAD_OK;
}

void CedProjector::set_weights(const std::vector<float>& v_down,
                               const std::vector<float>& u_up,
                               const std::vector<float>& bias) {
    if (v_down.size() == v_down_.size()) v_down_ = v_down;
    if (u_up.size() == u_up_.size())     u_up_ = u_up;
    if (bias.size() == bias_.size())     bias_ = bias;
}

int CedProjector::project(int target_layer, const float* h_t, int d,
                          std::vector<CedCandidate>& out_candidates,
                          float tau, int top_k) const {
    if (!h_t || d != cfg_.hidden_dim) return RAD_E_INVAL;
    out_candidates.clear();

    const int r = cfg_.rank;
    const int E = cfg_.n_experts;

    /* 1. Down-projection: z = V * h_t (R^r) */
    std::vector<float> z((size_t)r, 0.0f);
    for (int k = 0; k < r; ++k) {
        float sum = 0.0f;
        const float* v_row = &v_down_[(size_t)(k * d)];
        for (int j = 0; j < d; ++j) {
            sum += v_row[j] * h_t[j];
        }
        z[(size_t)k] = sum;
    }

    /* 2. Up-projection: logits = U * z + bias (R^E) */
    std::vector<float> logits((size_t)E, 0.0f);
    float max_logit = -1e30f;
    for (int e = 0; e < E; ++e) {
        float sum = bias_[(size_t)e];
        const float* u_row = &u_up_[(size_t)(e * r)];
        for (int k = 0; k < r; ++k) {
            sum += u_row[k] * z[(size_t)k];
        }
        logits[(size_t)e] = sum;
        if (sum > max_logit) max_logit = sum;
    }

    /* 3. Numerically stable Softmax */
    std::vector<float> probs((size_t)E, 0.0f);
    float exp_sum = 0.0f;
    for (int e = 0; e < E; ++e) {
        float p = std::exp(logits[(size_t)e] - max_logit);
        probs[(size_t)e] = p;
        exp_sum += p;
    }
    if (exp_sum > 0.0f) {
        for (int e = 0; e < E; ++e) probs[(size_t)e] /= exp_sum;
    }

    /* 4. Clustered Expert Distribution (CED) aggregation and candidate filtering */
    std::vector<float> cluster_mass;
    compute_cluster_mass(probs, cluster_mass);

    std::vector<std::pair<float, int>> ranked_clusters;
    ranked_clusters.reserve((size_t)cfg_.n_clusters);
    for (int c = 0; c < cfg_.n_clusters; ++c) {
        if (cluster_mass[(size_t)c] > 0.0f) {
            ranked_clusters.emplace_back(cluster_mass[(size_t)c], c);
        }
    }
    std::sort(ranked_clusters.begin(), ranked_clusters.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<bool> active_cluster((size_t)cfg_.n_clusters, false);
    const int effective_top_clusters = std::min((int)ranked_clusters.size(),
                                                std::max(1, cfg_.cluster_top_k));
    for (int i = 0; i < effective_top_clusters; ++i) {
        active_cluster[(size_t)ranked_clusters[i].second] = true;
    }

    std::vector<std::pair<float, int32_t>> filtered;
    filtered.reserve((size_t)E);
    for (int e = 0; e < E; ++e) {
        if (probs[(size_t)e] >= tau) {
            int c = cluster_of_expert(e);
            if (!cfg_.filter_by_cluster || (c >= 0 && c < cfg_.n_clusters && active_cluster[(size_t)c])) {
                filtered.emplace_back(probs[(size_t)e], static_cast<int32_t>(e));
            }
        }
    }

    std::sort(filtered.begin(), filtered.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second; // Tie-break by lower expert id
              });

    if (top_k > 0 && (int)filtered.size() > top_k) {
        filtered.resize((size_t)top_k);
    }

    out_candidates.reserve(filtered.size());
    for (const auto& f : filtered) {
        CedCandidate cand;
        cand.expert_id = f.second;
        cand.layer = target_layer;
        cand.unit_id = -1; // to be resolved by CedPrefetcher
        cand.probability = f.first;
        out_candidates.push_back(cand);
    }

    return RAD_OK;
}

void CedProjector::compute_cluster_mass(const std::vector<float>& probs,
                                        std::vector<float>& cluster_mass) const {
    cluster_mass.assign((size_t)cfg_.n_clusters, 0.0f);
    const int E = cfg_.n_experts;
    for (int e = 0; e < E && e < (int)probs.size(); ++e) {
        int c = cluster_of_expert(e);
        if (c >= 0 && c < cfg_.n_clusters) {
            cluster_mass[(size_t)c] += probs[(size_t)e];
        }
    }
}

int CedProjector::project_device(const float* d_h_t, float* d_out_logits, void* stream) const {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    CedGemvArgs args{};
    args.h_t = d_h_t;
    args.v_down = v_down_.data();
    args.u_up = u_up_.data();
    args.bias = bias_.data();
    args.logits = d_out_logits;
    args.d = cfg_.hidden_dim;
    args.r = cfg_.rank;
    args.n_experts = cfg_.n_experts;
    return rad_ced_gemv_launch(args, stream);
#else
    /* ced_gemv.hip is only built with HIP; the host backend must still link and must not report
     * a projection it did not compute (docs/IMPLEMENTATION.md). */
    (void)d_h_t; (void)d_out_logits; (void)stream;
    return RAD_E_UNSUPPORTED;
#endif
}

int CedProjector::project_with_ridgefill(int target_layer, const float* h_t, int d,
                                         const ridgefill::ProjectorBundle* bundle,
                                         const float* gate_weights,
                                         std::vector<CedCandidate>& out_candidates,
                                         float tau, int top_k) const {
    if (!bundle || target_layer < bundle->split() || !h_t) {
        return project(target_layer, h_t, d, out_candidates, tau, top_k);
    }

    const int8_t* codes = nullptr;
    const uint16_t* scale = nullptr;
    const uint16_t* bias = nullptr;
    int64_t n = 0, k = 0;

    if (!bundle->get_layer_i8(target_layer, &codes, &scale, &bias, &n, &k) || !codes || !scale) {
        return project(target_layer, h_t, d, out_candidates, tau, top_k);
    }

    /* RidgeFill map: R^10240 -> R^2560
     * Predict intermediate x_hat = W_RF * h_t + bias_RF */
    std::vector<float> x_hat((size_t)n, 0.0f);
    for (int64_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        const int8_t* row_codes = codes + row * k;
        const uint16_t* row_scale = scale + row * (k / 128);
        for (int64_t col = 0; col < k && col < d; ++col) {
            float sc = rad_bf16_to_f32(row_scale[col / 128]);
            float weight = static_cast<float>(row_codes[col]) * sc;
            sum += weight * h_t[col];
        }
        if (bias) sum += rad_bf16_to_f32(bias[row]);
        x_hat[(size_t)row] = sum;
    }

    /* Router gate projection: R^2560 -> R^512 logits */
    const int E = cfg_.n_experts;
    std::vector<float> logits((size_t)E, 0.0f);
    float max_logit = -1e30f;

    if (gate_weights) {
        for (int e = 0; e < E; ++e) {
            float sum = 0.0f;
            const float* g_row = gate_weights + e * n;
            for (int64_t j = 0; j < n; ++j) {
                sum += g_row[j] * x_hat[(size_t)j];
            }
            logits[(size_t)e] = sum;
            if (sum > max_logit) max_logit = sum;
        }
    } else {
        /* Fallback: use first E components of x_hat */
        for (int e = 0; e < E; ++e) {
            float val = (e < n) ? x_hat[(size_t)e] : 0.0f;
            logits[(size_t)e] = val;
            if (val > max_logit) max_logit = val;
        }
    }

    /* Softmax and thresholding */
    std::vector<float> probs((size_t)E, 0.0f);
    float exp_sum = 0.0f;
    for (int e = 0; e < E; ++e) {
        float p = std::exp(logits[(size_t)e] - max_logit);
        probs[(size_t)e] = p;
        exp_sum += p;
    }
    if (exp_sum > 0.0f) {
        for (int e = 0; e < E; ++e) probs[(size_t)e] /= exp_sum;
    }

    std::vector<float> cluster_mass;
    compute_cluster_mass(probs, cluster_mass);

    std::vector<std::pair<float, int>> ranked_clusters;
    ranked_clusters.reserve((size_t)cfg_.n_clusters);
    for (int c = 0; c < cfg_.n_clusters; ++c) {
        if (cluster_mass[(size_t)c] > 0.0f) {
            ranked_clusters.emplace_back(cluster_mass[(size_t)c], c);
        }
    }
    std::sort(ranked_clusters.begin(), ranked_clusters.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<bool> active_cluster((size_t)cfg_.n_clusters, false);
    const int effective_top_clusters = std::min((int)ranked_clusters.size(),
                                                std::max(1, cfg_.cluster_top_k));
    for (int i = 0; i < effective_top_clusters; ++i) {
        active_cluster[(size_t)ranked_clusters[i].second] = true;
    }

    std::vector<std::pair<float, int32_t>> filtered;
    for (int e = 0; e < E; ++e) {
        if (probs[(size_t)e] >= tau) {
            int c = cluster_of_expert(e);
            if (!cfg_.filter_by_cluster || (c >= 0 && c < cfg_.n_clusters && active_cluster[(size_t)c])) {
                filtered.emplace_back(probs[(size_t)e], static_cast<int32_t>(e));
            }
        }
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const auto& a, const auto& b) {
                  return (a.first != b.first) ? (a.first > b.first) : (a.second < b.second);
              });

    if (top_k > 0 && (int)filtered.size() > top_k) {
        filtered.resize((size_t)top_k);
    }

    out_candidates.clear();
    out_candidates.reserve(filtered.size());
    for (const auto& f : filtered) {
        CedCandidate cand;
        cand.expert_id = f.second;
        cand.layer = target_layer;
        cand.unit_id = -1;
        cand.probability = f.first;
        out_candidates.push_back(cand);
    }

    return RAD_OK;
}

/* ------------------------------------------------------------------ CedPrefetcher */

int CedPrefetcher::init(const Program& prog, const Plan& plan, Mover* mover, HeatEngine* heat,
                        const CedConfig& cfg) {
    prog_ = &prog;
    plan_ = &plan;
    mover_ = mover;
    heat_ = heat;
    cfg_ = cfg;

    expert_to_unit_.clear();
    total_layers_ = 0;

    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.layer >= 0 && mu.expert >= 0) {
            expert_to_unit_[{mu.layer, mu.expert}] = static_cast<int32_t>(u);
            if (mu.layer + 1 > total_layers_) total_layers_ = mu.layer + 1;
        }
    }

    reset_stats();
    return projector_.init(cfg_);
}

void CedPrefetcher::reset_stats() {
    stats_ = CedStats{};
}

int CedPrefetcher::evaluate_and_prefetch(int current_layer, const float* hidden_state) {
    return on_layer_dispatch(current_layer, hidden_state, nullptr);
}

int CedPrefetcher::on_layer_dispatch(int current_layer, const float* hidden_state, RadStream stream) {
    (void)stream;
    if (!cfg_.enabled || !hidden_state) return RAD_OK;

    const int target_layer = current_layer + cfg_.lookahead;
    if (target_layer >= total_layers_) return RAD_OK;

    std::vector<CedCandidate> candidates;
    int rc = RAD_OK;

    if (cfg_.reuse_ridgefill && rf_bundle_) {
        rc = projector_.project_with_ridgefill(target_layer, hidden_state, cfg_.hidden_dim,
                                               rf_bundle_, nullptr, candidates,
                                               cfg_.threshold, cfg_.top_k);
    } else {
        rc = projector_.project(target_layer, hidden_state, cfg_.hidden_dim,
                                candidates, cfg_.threshold, cfg_.top_k);
    }
    if (rc < 0) return rc;

    stats_.predictions += candidates.size();

    int64_t window_bytes = 0;

    for (auto& cand : candidates) {
        cand.unit_id = find_unit_id(target_layer, cand.expert_id);
        if (cand.unit_id < 0) continue;

        /* 1. Check if unit is already resident in VRAM */
        if ((mover_ && mover_->is_unit_resident(cand.unit_id)) ||
            (heat_ && heat_->tier(cand.unit_id) == Tier::VRAM)) {
            ++stats_.prefetch_hits;
            ++stats_.prefetch_skipped_resident;
            continue;
        }

        /* 2. Bandwidth throttling check to protect PCIe and TP2 All-Reduce */
        const int64_t bytes = mover_ ? mover_->unit_bytes(cand.unit_id) : 3145728;
        if (window_bytes + bytes > cfg_.max_prefetch_bytes) {
            ++stats_.pcie_throttled;
            break;
        }

        /* 3. Check slab slot availability: speculative prefetch is purely opportunistic.
         * It must NEVER evict resident units and must leave working reserve free for RLFU. */
        const int32_t klass = mover_ ? mover_->unit_klass(cand.unit_id) : 0;
        const bool has_free = mover_ ? mover_->has_free_slab_slot((size_t)klass) : true;
        if (!has_free) {
            ++stats_.prefetch_skipped_leased;
            continue;
        }

        /* 4. Issue asynchronous lookahead DMA prefetch */
        if (mover_) {
            int prc = mover_->prefetch_unit_async(cand.unit_id, -1, heat_, 2);
            if (prc == RAD_OK) {
                ++stats_.prefetch_issued;
                stats_.prefetch_bytes += (uint64_t)bytes;
                window_bytes += bytes;
            }
        } else {
            /* Standalone / mock path */
            if (heat_) heat_->register_prefetch(cand.unit_id);
            ++stats_.prefetch_issued;
            stats_.prefetch_bytes += (uint64_t)bytes;
            window_bytes += bytes;
        }
    }

    return RAD_OK;
}

int CedPrefetcher::on_step_routing(int current_layer, const int32_t* counts, int32_t n_experts) {
    if (!cfg_.enabled || !counts || n_experts <= 0) return RAD_OK;

    const int target_layer = current_layer + cfg_.lookahead;
    if (target_layer >= total_layers_) return RAD_OK;

    /* Compute cluster mass from observed routing counts */
    std::vector<float> cluster_mass((size_t)cfg_.n_clusters, 0.0f);
    int64_t total_hits = 0;
    for (int32_t e = 0; e < n_experts && e < cfg_.n_experts; ++e) {
        if (counts[e] > 0) {
            total_hits += counts[e];
            int c = projector_.cluster_of_expert(e);
            if (c >= 0 && c < cfg_.n_clusters) {
                cluster_mass[(size_t)c] += static_cast<float>(counts[e]);
            }
        }
    }
    if (total_hits == 0) return RAD_OK;

    /* Rank active clusters by relative mass */
    std::vector<std::pair<float, int>> ranked_clusters;
    ranked_clusters.reserve((size_t)cfg_.n_clusters);
    for (int c = 0; c < cfg_.n_clusters; ++c) {
        if (cluster_mass[(size_t)c] > 0.0f) {
            ranked_clusters.emplace_back(cluster_mass[(size_t)c] / static_cast<float>(total_hits), c);
        }
    }
    std::sort(ranked_clusters.begin(), ranked_clusters.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    /* Entropy Gate: If cluster mass is diffuse (top cluster < 30%), routing entropy is high
     * (e.g. creative prose, summarization, random tokens). Skip speculative prefetching
     * to avoid locking slab slots with false positives and saturating PCIe All-Reduce. */
    if (ranked_clusters.empty() || ranked_clusters[0].first < 0.30f) {
        return RAD_OK;
    }

    std::vector<bool> active_cluster((size_t)cfg_.n_clusters, false);
    const int effective_top_clusters = std::min((int)ranked_clusters.size(),
                                                std::max(1, cfg_.cluster_top_k));
    for (int i = 0; i < effective_top_clusters; ++i) {
        active_cluster[(size_t)ranked_clusters[i].second] = true;
    }

    /* Candidate expert selection for target_layer:
     * Experts in active clusters with probability above threshold */
    std::vector<CedCandidate> candidates;
    for (int e = 0; e < cfg_.n_experts; ++e) {
        int c = projector_.cluster_of_expert(e);
        if (c >= 0 && c < cfg_.n_clusters && active_cluster[(size_t)c]) {
            float prob = cluster_mass[(size_t)c] / static_cast<float>(total_hits);
            if (prob >= cfg_.threshold) {
                CedCandidate cand;
                cand.expert_id = e;
                cand.layer = target_layer;
                cand.unit_id = find_unit_id(target_layer, e);
                cand.probability = prob;
                if (cand.unit_id >= 0) {
                    candidates.push_back(cand);
                }
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return (a.probability != b.probability) ? (a.probability > b.probability)
                                                          : (a.expert_id < b.expert_id);
              });

    if (cfg_.top_k > 0 && (int)candidates.size() > cfg_.top_k) {
        candidates.resize((size_t)cfg_.top_k);
    }

    stats_.predictions += candidates.size();
    int64_t window_bytes = 0;

    for (auto& cand : candidates) {
        if (cand.unit_id < 0) continue;

        /* 1. Check if unit is already resident in VRAM */
        if ((mover_ && mover_->is_unit_resident(cand.unit_id)) ||
            (heat_ && heat_->tier(cand.unit_id) == Tier::VRAM)) {
            ++stats_.prefetch_skipped_resident;
            continue;
        }

        /* 2. Bandwidth throttling check to protect PCIe and TP2 All-Reduce */
        const int64_t bytes = mover_ ? mover_->unit_bytes(cand.unit_id) : 3145728;
        if (window_bytes + bytes > cfg_.max_prefetch_bytes) {
            ++stats_.pcie_throttled;
            break;
        }

        /* 3. Check slab slot availability: speculative prefetch is purely opportunistic.
         * It must NEVER evict resident units (which causes ping-pong loops and quarantine violations)
         * and must leave the shadow working reserve free for RLFU turnover. */
        const int32_t klass = mover_ ? mover_->unit_klass(cand.unit_id) : 0;
        const bool has_free = mover_ ? mover_->has_free_slab_slot((size_t)klass) : true;
        if (!has_free) {
            ++stats_.prefetch_skipped_leased;
            continue;
        }

        /* 4. Issue asynchronous lookahead DMA prefetch with transient lease (2 dispatches) */
        if (mover_) {
            int prc = mover_->prefetch_unit_async(cand.unit_id, -1, heat_, 2);
            if (prc == RAD_OK) {
                ++stats_.prefetch_issued;
                stats_.prefetch_bytes += (uint64_t)bytes;
                window_bytes += bytes;
            }
        } else {
            if (heat_) heat_->register_prefetch(cand.unit_id);
            ++stats_.prefetch_issued;
            stats_.prefetch_bytes += (uint64_t)bytes;
            window_bytes += bytes;
        }
    }

    return RAD_OK;
}

std::string CedPrefetcher::report() const {
    char buf[512];
    double hit_rate = (stats_.predictions > 0)
        ? (100.0 * (double)stats_.prefetch_hits / (double)stats_.predictions) : 0.0;
    snprintf(buf, sizeof(buf),
             "CED MoE Prefetcher: lookahead=%d tau=%.2f top_k=%d cluster_k=%d | "
             "preds=%llu issued=%llu hits=%llu (%.1f%%) skipped_res=%llu "
             "skipped_leased=%llu throttled=%llu bytes=%s",
             cfg_.lookahead, cfg_.threshold, cfg_.top_k, cfg_.cluster_top_k,
             (unsigned long long)stats_.predictions,
             (unsigned long long)stats_.prefetch_issued,
             (unsigned long long)stats_.prefetch_hits,
             hit_rate,
             (unsigned long long)stats_.prefetch_skipped_resident,
             (unsigned long long)stats_.prefetch_skipped_leased,
             (unsigned long long)stats_.pcie_throttled,
             humanb(static_cast<int64_t>(stats_.prefetch_bytes)).c_str());
    return std::string(buf);
}

}  /* namespace rad */
