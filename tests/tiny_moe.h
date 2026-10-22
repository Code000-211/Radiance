/* tiny_moe.h -- tiny routed checkpoints, for the start-time tests that need a model which can lend
 * at all, and for the ones that need a serial drafter on top.
 *
 * tests/tiny_model.h's checkpoints are dense: every weight is a static unit, the planner marks
 * nothing movable, and Engine::start registers no lender (core/engine_bringup.cpp). The reference
 * basis' start gate can only be told apart on a model that can lend at all: without conditional
 * units a start with the gate on and a start with it off look the same. Those units are what
 * `--placement expert_tiered` can tier between the card and the pinned host pool, which is what
 * makes `Mover::lend_from` carve anything.
 *
 * This header adds exactly that routed arm to tiny_model.h's checkpoints, by rewriting the
 * checkpoint and config write_gemma4 / write_qwen35 wrote. It duplicates no trunk tensor and
 * changes neither tiny_model.h nor any plugin: each arm is the one the plugin already declares.
 *
 * Not models for numerics: they exist to make the host start path reach the lender with a positive
 * offered capacity, and (write_qwen35_moe) to give the reference basis a serial head to profile. */
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

/* ------------------------------------------------------------------ qwen35moe: experts + MTP */

/* Appends `w` as the block-scaled fp8 release of itself: `fp8_e4m3*bf16[128x128]`, the encoding
 * arch/qwen35_fp8 declares for every linear it serves. The codes are one byte a value and the
 * plane `<name>_scale_inv` holds each 128x128 block's scale as `[rows / 128, cols / 128]`, which is
 * where core/format/checkpoint.cpp looks for it. A fixture weight comes from linear(), so no block
 * is all zeros and the division is safe. */
inline void push_fp8(std::vector<Tensor>& out, const Tensor& w) {
    const int64_t N = w.shape[0], K = w.shape[1];
    const int64_t nb = (N + 127) / 128, kb = (K + 127) / 128;
    Tensor q{ w.name, "F8_E4M3", w.shape, {} };
    Tensor s{ w.name + "_scale_inv", "BF16", { nb, kb }, {} };
    q.bytes.assign((size_t)(N * K), 0);
    s.bytes.assign((size_t)(nb * kb * 2), 0);
    for (int64_t n = 0; n < N; n += 128)
        for (int64_t k = 0; k < K; k += 128) {
            float amax = 0.0f;
            for (int64_t i = n; i < n + 128 && i < N; ++i)
                for (int64_t j = k; j < k + 128 && j < K; ++j)
                    amax = std::max(amax,
                                    std::fabs(rad_load_f32(w.bytes.data(), RAD_BF16, i * K + j)));
            const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
            rad_store_f32(s.bytes.data(), RAD_BF16, (n / 128) * kb + k / 128, scale);
            for (int64_t i = n; i < n + 128 && i < N; ++i)
                for (int64_t j = k; j < k + 128 && j < K; ++j)
                    rad_store_f32(q.bytes.data(), RAD_F8E4M3, i * K + j,
                                  rad_load_f32(w.bytes.data(), RAD_BF16, i * K + j) / scale);
        }
    out.push_back(std::move(q));
    out.push_back(std::move(s));
}

/* Qwen3.5 MoE with its MTP head (`qwen3_5_moe`, served by arch/qwen35moe_fp8) -- the only plugin a
 * tiny checkpoint can reach that declares BOTH the routed feed-forward (one RAD_ACCESS_CONDITIONAL
 * unit an expert, arch/common/rad_block_moe_fp8.h) and a serial drafter (`dd.kind =
 * RAD_DRAFT_SERIAL` over the `mtp.` block, arch/qwen35_fp8/qwen35_fp8.cpp:1067). A reference-basis
 * run needs both at once: the serial head is what `--draft-reference` profiles, and the
 * conditional units are what makes the live twin lend.
 *
 * Built on write_qwen35's trunk (same attention, gated delta net, names and geometry) and rewritten
 * the way write_gemma4_moe rewrites write_gemma4:
 *
 *   config      model_type `qwen3_5_moe`; `moe_intermediate_size`, `num_experts`,
 *               `num_experts_per_tok`; `mtp_num_hidden_layers` (the head is declared only when
 *               `nextn > 0 && max_spec > 0`); and the fp8 quantisation the plugin serves.
 *   per layer   `mlp.gate.weight` [E, H], the router, bf16; and per expert
 *               `mlp.experts.N.{gate,up,down}_proj.weight` in fp8 -- MoeFP8 refuses plain bf16
 *               experts by name -- as separate tensors, which moe_map_expert joins into gate_up.
 *   the head    `mtp.fc` [H, 2H] bf16, the three f32 norms, one gated-attention layer on the
 *               trunk's geometry and its own routed feed-forward (`mtp.layers.0.mlp.*`), as
 *               arch/common/rad_block_mtp_fp8.h declares them.
 *
 * The dense `mlp.gate_proj/up_proj/down_proj` write_qwen35 wrote are dropped: this release routes.
 * Trunk linears and the attention head are quantised; the router, the delta net's a/b projections,
 * the head's `fc` and every gain stay as the checkpoint holds them, which is the shape
 * data/recipes/qwen4exp-w4.recipe gives a routed release. */
struct Qwen35Moe {
    /* Every linear's K a whole 128-block, so the fp8 codes and the scales line up: hidden and the
     * expert width a multiple of 128, and head_dim / kv-head dim 128. */
    Qwen35  trunk{};
    int64_t experts = 8, top_k = 2, moe_ffn = 128;
    int64_t mtp_layers = 1;                    /* `mtp_num_hidden_layers`; below trunk.layers */
    float   branch = 0.5f;                    /* the o_proj / down_proj gain, as Qwen35's */
    uint64_t seed = 0x35;
};

inline bool write_qwen35_moe(const std::string& dir, const Qwen35Moe& m) {
    const Qwen35& s = m.trunk;
    if (s.layers < 2 || m.experts <= 0 || m.top_k <= 0 || m.top_k > m.experts || m.moe_ffn <= 0 ||
        m.mtp_layers <= 0 || m.mtp_layers >= s.layers)
        return false;
    if (!write_qwen35(dir, s)) return false;

    nlohmann::ordered_json c =
        nlohmann::ordered_json::parse(read_text(dir + "/config.json"), nullptr, false);
    if (!c.is_object()) return false;
    c["architectures"]         = { "Qwen3_5MoeForConditionalGeneration" };
    c["model_type"]            = "qwen3_5_moe";
    c["intermediate_size"]     = m.moe_ffn;
    c["moe_intermediate_size"] = m.moe_ffn;
    c["num_experts"]           = m.experts;
    c["num_experts_per_tok"]   = m.top_k;
    c["mtp_num_hidden_layers"] = m.mtp_layers;
    /* check_format holds the plugin to `fp8` / `e4m3` / a 128 block, which push_fp8 writes. */
    c["quantization_config"] = { { "quant_method", "fp8" }, { "fmt", "e4m3" },
                                 { "weight_block_size", { 128, 128 } } };

    std::vector<Tensor> in = read_safetensors(dir + "/model.safetensors");
    if (in.empty()) return false;
    Rng r(m.seed ^ s.seed);

    const int64_t H = s.hidden, q = s.heads * s.head_dim, kv = s.kv_heads * s.head_dim;
    const float gu_amp = std::sqrt(3.0f / (float)H);
    const float dn_amp = std::sqrt(3.0f / (float)m.moe_ffn);

    /* The routed feed-forward of one layer: the router bf16, every expert its own fp8 tensors. */
    auto moe_ffn = [&](const std::string& p, std::vector<Tensor>& out) {
        out.push_back(noise(p + "mlp.gate.weight", { m.experts, H }, r, 0.5f));
        for (int64_t e = 0; e < m.experts; ++e) {
            const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            push_fp8(out, linear(ep + "gate_proj.weight", m.moe_ffn, H, r, gu_amp));
            push_fp8(out, linear(ep + "up_proj.weight", m.moe_ffn, H, r, gu_amp));
            push_fp8(out, linear(ep + "down_proj.weight", H, m.moe_ffn, r, dn_amp));
        }
    };
    /* True for the trunk linears the fp8 plugin serves as block-scaled: the attention and delta
     * net projections. `in_proj_a`/`in_proj_b` keep bf16 (`gdn_ab`, its own buffer), as the
     * release holds them. */
    auto is_fp8_trunk = [](const std::string& n) {
        static const char* kSuffix[] = { "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                         "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                         "linear_attn.in_proj_qkv.weight",
                                         "linear_attn.in_proj_z.weight",
                                         "linear_attn.out_proj.weight" };
        for (const char* sfx : kSuffix) {
            const size_t n_sfx = std::strlen(sfx);
            if (n.size() >= n_sfx && n.compare(n.size() - n_sfx, n_sfx, sfx) == 0) return true;
        }
        return false;
    };
    auto is_dense_mlp = [](const std::string& n) {
        return n.find(".mlp.gate_proj.weight") != std::string::npos ||
               n.find(".mlp.up_proj.weight") != std::string::npos ||
               n.find(".mlp.down_proj.weight") != std::string::npos;
    };

    std::vector<Tensor> out;
    for (const Tensor& t : in) {
        if (is_dense_mlp(t.name)) continue;         /* a routed release has no dense arm */
        if (t.shape.size() == 2 && is_fp8_trunk(t.name)) {
            push_fp8(out, t);
            continue;
        }
        out.push_back(t);
    }

    const std::string lm = "model.language_model.";
    for (int64_t l = 0; l < s.layers; ++l)
        moe_ffn(lm + "layers." + std::to_string(l) + ".", out);

    /* The MTP head. `fc` is bf16 and the three norms f32 by the block's own declaration
     * (rad_block_mtp_fp8.h: w_fc RAD_BF16, w_pre_h/w_pre_e/w_norm RAD_F32). */
    out.push_back(noise("mtp.pre_fc_norm_hidden.weight", { H }, r, 0.1f, 1.0f, "F32"));
    out.push_back(noise("mtp.pre_fc_norm_embedding.weight", { H }, r, 0.1f, 1.0f, "F32"));
    out.push_back(noise("mtp.norm.weight", { H }, r, 0.1f, 1.0f, "F32"));
    out.push_back(linear("mtp.fc.weight", H, 2 * H, r));
    for (int64_t l = 0; l < m.mtp_layers; ++l) {
        const std::string p = "mtp.layers." + std::to_string(l) + ".";
        out.push_back(noise(p + "input_layernorm.weight", { H }, r, 0.1f));
        push_fp8(out, linear(p + "self_attn.q_proj.weight", 2 * q, H, r));
        push_fp8(out, linear(p + "self_attn.k_proj.weight", kv, H, r));
        push_fp8(out, linear(p + "self_attn.v_proj.weight", kv, H, r));
        push_fp8(out, linear(p + "self_attn.o_proj.weight", H, q, r, m.branch));
        out.push_back(noise(p + "self_attn.q_norm.weight", { s.head_dim }, r, 0.1f));
        out.push_back(noise(p + "self_attn.k_norm.weight", { s.head_dim }, r, 0.1f));
        out.push_back(noise(p + "post_attention_layernorm.weight", { H }, r, 0.1f));
        moe_ffn(p, out);
    }

    return write_text(dir + "/config.json", c.dump(2)) &&
           write_safetensors(dir + "/model.safetensors", out);
}

}  /* namespace tiny */
