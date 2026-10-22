/* tiny_moe.h -- a tiny Gemma4 MoE checkpoint, for the start-time tests that need a model which can
 * lend at all.
 *
 * tests/tiny_model.h's checkpoints are dense: every weight is a static unit, the planner marks
 * nothing movable, and Engine::start registers no lender (core/engine_bringup.cpp). The reference
 * basis' start gate can only be told apart on a model that can lend at all: without conditional
 * units a start with the gate on and a start with it off look the same. Those units are what
 * `--placement expert_tiered` can tier between the card and the pinned host pool, which is what
 * makes `Mover::lend_from` carve anything.
 *
 * This header adds exactly that routed arm to tiny_model.h's Gemma4 trunk, by rewriting the
 * checkpoint and config write_gemma4 wrote. It duplicates no trunk tensor and changes neither
 * tiny_model.h nor any plugin: the arm is the one arch/gemma4 already declares behind
 * `enable_moe_block`.
 *
 * Not a model for numerics: it exists to make the host start path reach the lender with a positive
 * offered capacity. */
#pragma once

#include "tiny_model.h"

namespace tiny {

/* Rewrites `dir` (a checkpoint write_gemma4(dir, s) just wrote, with s.assistant false) into a MoE
 * release: `experts` experts of width `moe_ffn` on every layer, `top_k` routed per token.
 *
 * The stacked per-layer tensors are the ones arch/gemma4/gemma4.cpp reads:
 *   experts.gate_up_proj [E, 2 * moe_ffn, hidden]   (map_slice along the expert axis)
 *   experts.down_proj    [E, hidden, moe_ffn]
 *   router.proj.weight   [E, hidden], router.per_expert_scale [E]
 *   router.scale, pre_feedforward_layernorm_2, post_feedforward_layernorm_{1,2} [hidden]
 * The dense arm stays: the plugin runs both, as the 31B does. */
inline bool write_gemma4_moe(const std::string& dir, const Gemma4& s, int64_t experts,
                             int64_t top_k, int64_t moe_ffn, uint64_t seed = 0x3EED) {
    if (experts <= 0 || top_k <= 0 || top_k > experts || moe_ffn <= 0) return false;
    /* write_gemma4(assistant) selects the plugin that has the serial head, and that plugin refuses
     * any checkpoint with experts; this fixture stays on the routed arch, which declares none. */
    if (s.assistant) return false;
    if (!write_gemma4(dir, s)) return false;

    nlohmann::ordered_json c =
        nlohmann::ordered_json::parse(read_text(dir + "/config.json"), nullptr, false);
    if (!c.is_object() || !c.contains("text_config")) return false;
    c["text_config"]["enable_moe_block"]     = true;
    c["text_config"]["num_experts"]          = experts;
    c["text_config"]["top_k_experts"]        = top_k;
    c["text_config"]["moe_intermediate_size"] = moe_ffn;

    std::vector<Tensor> ts = read_safetensors(dir + "/model.safetensors");
    if (ts.empty()) return false;
    Rng r(seed ^ s.seed);
    const int64_t H = s.hidden;
    const float gu_amp = std::sqrt(3.0f / (float)H);
    const float dn_amp = std::sqrt(3.0f / (float)moe_ffn);
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
        ts.push_back(noise(p + "pre_feedforward_layernorm_2.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "router.scale", { H }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "post_feedforward_layernorm_1.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "post_feedforward_layernorm_2.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "router.proj.weight", experts, H, r));
        ts.push_back(noise(p + "router.per_expert_scale", { experts }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "experts.gate_up_proj", { experts, 2 * moe_ffn, H }, r, gu_amp));
        ts.push_back(noise(p + "experts.down_proj", { experts, H, moe_ffn }, r, dn_amp));
    }
    return write_text(dir + "/config.json", c.dump(2)) &&
           write_safetensors(dir + "/model.safetensors", ts);
}

}  /* namespace tiny */
