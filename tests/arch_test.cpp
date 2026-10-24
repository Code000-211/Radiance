/* arch_test.cpp -- the declared graph of the architecture plugins, asserted statically.
 *
 * No GPU, no model file, no core. With two-phase construction (spec §3) the whole model is known
 * between declare and plan, so a unit test can check what a plugin declares. A fake builder
 * records every decl_weight, decl_buffer, decl_op, buffer use and name-map call, and a fake RadCtx
 * records every issue, so the run phase's order and operands are checked too.
 *
 * The fake is deliberately not core/build/'s RadBuilder: a defect there must not fail this test,
 * and a plugin that only works against one builder has an undeclared dependency.
 */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include <array>
#include <dlfcn.h>
#include <thread>

/* ==================================================================== the plugin under test */
#include "../arch/qwen35_bf16/qwen35_bf16.cpp"
/* The production architecture, for its sizing declare (RadBuildCtx::shape_probe). */
#include "../arch/qwen4exp_fp8/qwen4exp_fp8.cpp"
/* The LLaMA family, for how it reads its rotary scaling. */
#include "../arch/llama_fp8/llama_fp8.cpp"
/* LFM2, for its own layer schedule and short convolution. */
#include "../arch/lfm2_bf16/lfm2_bf16.cpp"
/* Gemma 4 dense: two attention classes (sliding-window + proportional-RoPE global) on one stream. */
#include "../arch/gemma4_bf16/gemma4_bf16.cpp"
/* The bf16 / fp8-expert Gemma 4 plugin: the 26B-A4B's dense and routed arms side by side. */
#include "../arch/gemma3/gemma3.cpp"
#include "../arch/gemma4/gemma4.cpp"
/* Qwen3-ASR: a Whisper-style audio tower over a dense Qwen3 decoder. */
#include "../arch/qwen3asr_bf16/qwen3asr_bf16.cpp"
/* The PLE block alone: no architecture here declares it, and what its step() hands the two
 * windowed ops is its own contract (rad_block_ple.h). */
#include <arch/rad_block_ple.h>
/* Qwen3.5 / 3.6 / 3.8 with block-fp8 weights, dense and routed: one source built twice under
 * three macros, as arch/qwen35moe_fp8 builds it. */
#include "../arch/qwen35_fp8/qwen35_fp8.cpp"
#undef QWEN35_MOE
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35moe_fp8/qwen35moe_fp8.cpp"
/* The same source a third time, as qwen35_w8a8: the int8 trunk. */
#undef QWEN35_MOE
#undef QWEN35_W8A8
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35_w8a8/qwen35_w8a8.cpp"
/* The LLaMA family from a dense checkpoint: llama_fp8 under its other descriptor. */
#undef LLAMA_DENSE_SRC
#undef LLAMA_NS
#undef LLAMA_QUANT
#include "../arch/llama_dense_fp8/llama_dense_fp8.cpp"
/* EmbeddingGemma 2: a pooled encoder with no logits. */
#include "../arch/embgemma2/embgemma2.cpp"
/* Qwen-Image 2.1: a generator conditioned by Qwen3-VL. */
#include "../arch/qwenimage21/qwenimage21.cpp"
/* Anima: a cross-attention generator conditioned through two tokenizers. */
#include "../arch/anima/anima.cpp"

/* ==================================================================== fixtures and helpers */
namespace {

/* Qwen3.8-27B's shape cut to eight layers, so failures stay readable. Eight is the smallest count
 * that exercises the alternation twice: with full_attention_interval 4, layers 3 and 7 are full
 * attention and the other six gated delta net, the 64-layer model's 1:3 ratio.
 *
 * head_dim is 256 while n_embd / n_head is 213, so using the latter would be caught. */
const char* k_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
};
const char* k_vals[] = { "4", "64", "128", "16", "48", "6144", "4" };

RadModelMeta qwen35_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen35";
    m.name        = "test-qwen35-27b";
    m.quant       = "bf16";
    m.n_layers    = 8;
    m.n_embd      = 5120;
    m.n_head      = 24;
    m.n_head_kv   = 4;
    m.head_dim    = 256;
    m.n_ff        = 17408;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_keys) / sizeof(k_keys[0]));
    m.kv_key      = k_keys;
    m.kv_val      = k_vals;
    return m;
}

/* Gemma 4 31B's text geometry cut to twelve layers, the smallest count with two global layers at
 * (l + 1) % 6 == 0 (layers 5 and 11) over ten sliding ones, the model's 5:1 ratio. head_dim 256 is
 * the sliding width; the global class's 512 width and 4 heads are header keys the plugin must
 * read, not RadModelMeta fields. */
const char* k_gem4_keys[] = {
    "sliding_window",
    "global_head_dim",
    "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta",
    "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor",
    "num_kv_shared_layers",
    "final_logit_softcapping",
    "full_attention_interval",
    "attention_k_eq_v",
};
const char* k_gem4_vals[] = { "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1" };

RadModelMeta gemma4_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-31b";
    m.quant       = "bf16";
    m.n_layers    = 12;
    m.n_embd      = 5376;
    m.n_head      = 32;
    m.n_head_kv   = 16;
    m.head_dim    = 256;
    m.n_ff        = 21504;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_gem4_keys) / sizeof(k_gem4_keys[0]));
    m.kv_key      = k_gem4_keys;
    m.kv_val      = k_gem4_vals;
    return m;
}

/* The same twelve layers with Gemma's assistant head merged under `draft.` as
 * `rad-convert --draft-model` writes it: the head's config flattened, its text config under
 * `draft.text_config.`. The 31B's head: width 1024, four layers (three sliding, one full), the
 * trunk's head geometry and window, and its own 262144-row tied table. */
const char* k_gem4d_keys[] = {
    "sliding_window", "global_head_dim", "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v",
    "draft.architectures", "draft.backbone_hidden_size", "draft.use_ordered_embeddings",
    "draft.text_config.hidden_size", "draft.text_config.num_hidden_layers",
    "draft.text_config.layer_types", "draft.text_config.num_attention_heads",
    "draft.text_config.num_key_value_heads", "draft.text_config.num_global_key_value_heads",
    "draft.text_config.head_dim", "draft.text_config.global_head_dim",
    "draft.text_config.intermediate_size", "draft.text_config.sliding_window",
    "draft.text_config.vocab_size", "draft.text_config.rms_norm_eps",
    "draft.text_config.rope_parameters.sliding_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.partial_rotary_factor",
};
const char* k_gem4d_vals[] = {
    "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1",
    "Gemma4AssistantForCausalLM", "5376", "0",
    "1024", "4",
    "sliding_attention sliding_attention sliding_attention full_attention", "32",
    "16", "4",
    "256", "512",
    "8192", "1024",
    "262144", "1e-06",
    "10000", "1000000", "0.25",
};

RadModelMeta gemma4_draft_meta() {
    RadModelMeta m = gemma4_meta();
    m.n_kv   = (int)(sizeof(k_gem4d_keys) / sizeof(k_gem4d_keys[0]));
    m.kv_key = k_gem4d_keys;
    m.kv_val = k_gem4d_vals;
    return m;
}

/* The gated attention layer's ops, listed once for every test.
 *
 * The fused qk_norm_rope_gate is not declared: its schema requires a precomputed cos/sin table
 * rather than theta, and this plugin has none (see rad_block_attn_gated.h). Declaring it with
 * parameters its schema lacks would fail declare outright rather than fall back to the four ops it
 * stands for. */
std::vector<std::string> attn_layer_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt" };
    v.push_back("rmsnorm"); v.push_back("rmsnorm");
    v.push_back("rope");    v.push_back("rope");
    v.push_back("attn_paged");
    v.push_back("kv_store");
    v.push_back("sigmoid");
    v.push_back("mul");
    v.push_back("gemm_nt");
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

std::vector<std::string> gdn_layer_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt",
                                   "gdn_conv_prep", "gdn_kkt_solve", "gdn_chunk_scan",
                                   "gdn_gated_rmsnorm",
                                   "gdn_conv_update", "gdn_recurrent_update",
                                   "gemm_nt" };
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

std::vector<std::string> mlp_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt" };
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

void append(std::vector<std::string>& v, const std::vector<std::string>& t) {
    v.insert(v.end(), t.begin(), t.end());
}

/* The whole declared program for the 8-layer fixture: layers 3 and 7 are full attention. */
std::vector<std::string> program_ops(bool tp) {
    std::vector<std::string> v = { "embed_lookup" };
    /* The embedding table is vocab-sharded: a rank gathers only ids in its slice and zero-fills
     * the rest, and this collective sums the ranks into x. Hence 2 all_reduces per layer plus
     * one. */
    if (tp) v.push_back("all_reduce");
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) append(v, attn_layer_ops(tp));
        else                  append(v, gdn_layer_ops(tp));
        append(v, mlp_ops(tp));
    }
    v.push_back("rmsnorm");
    /* gather_rows before the lm_head: the sampler needs one row per sequence, and the widest GEMM
     * over every prefill token would be n_tok/n_seq times the work and an 8 GiB logits buffer
     * (docs/OPS.md). */
    v.push_back("gather_rows");
    v.push_back("logits_gemm");
    return v;
}

}  /* namespace */

/* ==================================================================== tests */

TEST(declare_is_a_qwen35_trunk) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* Six gated delta net layers to two full attention ones, the interval's 1:3 ratio. */
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "attn_paged"), 2);
    /* The fused gate op is not declared (see attn_layer_ops), so each attention layer runs two
     * per-head rmsnorms and two ropes instead. */
    CHECK_EQ(count_op(b, "qk_norm_rope_gate"), 0);
    CHECK_EQ(count_op(b, "rope"), 4);
    /* One MLP per layer, the same block for both layer kinds. */
    CHECK_EQ(count_op(b, "silu_mul"), 8);
}

TEST(qwen35_bf16_yarn_reaches_every_text_rotary_op) {
    RadModelMeta meta = qwen35_meta();
    std::vector<const char*> keys(meta.kv_key, meta.kv_key + meta.n_kv);
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    keys.insert(keys.end(), {"rope_parameters.rope_type", "rope_parameters.factor",
                            "rope_parameters.original_max_position_embeddings"});
    vals.insert(vals.end(), {"yarn", "2", "262144"});
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuildCtx ctx = build_ctx();
    RadBuilder b;
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
    int rotary = 0;
    for (const auto& op : b.ops) if (op.op == "rope") {
        ++rotary;
        int found = 0;
        for (const auto& p : op.p) {
            if (p.key == "yarn_factor") { CHECK_EQ(p.dval, 2.0); ++found; }
            if (p.key == "yarn_magnitude") { CHECK_NEAR(p.dval, 1.069314718, 1e-7); ++found; }
            if (p.key == "scale") CHECK_EQ(p.dval, 1.0);
        }
        CHECK_EQ(found, 2);
    }
    CHECK_EQ(rotary, 4);
    vals[vals.size()-2] = "nan";
    RadBuilder invalid;
    CHECK_EQ(qwen35_bf16::declare(&invalid, &meta, &ctx), RAD_E_INVAL);
}

/* The LLaMA family reads rotary scaling through the same reader. Llama 3.1's `rope_type: llama3`
 * with factor 8, read as a bare factor, would be a linear position divide by 8, so it is refused
 * by name at declare. Linear stays linear and YaRN reaches the rope op. */
TEST(llama_reads_its_rotary_scaling_by_type) {
    auto declare = [](const char* type, const char* factor, RadBuilder* b) {
        const char* keys[] = {"rope_scaling.rope_type", "rope_scaling.factor",
                              "rope_scaling.original_max_position_embeddings"};
        const char* vals[] = {type, factor, "8192"};
        RadModelMeta m{};
        m.arch_id = "llama"; m.name = "test-llama"; m.quant = "fp8_e4m3";
        m.n_layers = 2; m.n_embd = 1024; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
        m.n_ff = 2048; m.n_vocab = 32000; m.n_ctx_train = 8192;
        m.rms_eps = 1e-5f; m.rope_theta = 5e5f; m.rope_scale = 1.0f;
        m.n_kv = type ? 3 : 0; m.kv_key = keys; m.kv_val = vals;
        RadBuildCtx c = build_ctx();
        return llama_fp8::declare(b, &m, &c);
    };
    for (const char* refused : {"llama3", "longrope", "dynamic"}) {
        RadBuilder b;
        CHECK_EQ(declare(refused, "8", &b), RAD_E_UNSUPPORTED);
    }
    {
        RadBuilder b;
        CHECK_OK(declare("linear", "2", &b));
        CHECK(count_op(b, "rope") > 0);
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 2.0);
            CHECK(!param(&op, "yarn_factor"));
        }
    }
    {
        RadBuilder b;
        CHECK_OK(declare("yarn", "4", &b));
        CHECK(count_op(b, "rope") > 0);
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 1.0);
            CHECK_EQ(pflt(&op, "yarn_factor"), 4.0);
        }
    }
    {
        RadBuilder b;
        CHECK_OK(declare(nullptr, "", &b));
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 1.0);
            CHECK(!param(&op, "yarn_factor"));
        }
    }
}

TEST(the_alternation_puts_full_attention_at_3_and_7) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The weight manifest shows a layer's kind: full attention has a q|gate projection, linear
     * attention an in-projection, never both. */
    for (int l = 0; l < 8; ++l) {
        const bool full = ((l + 1) % 4 == 0);
        const std::string qg  = "blk." + std::to_string(l) + ".attn_qg.weight";
        const std::string in  = "blk." + std::to_string(l) + ".ssm_in.weight";
        CHECK_EQ(find_w(b, qg) != nullptr, full);
        CHECK_EQ(find_w(b, in) != nullptr, !full);
        /* Both kinds share the pre-norm name and feed-forward. */
        CHECK(find_w(b, "blk." + std::to_string(l) + ".attn_norm.weight") != nullptr);
        CHECK(find_w(b, "blk." + std::to_string(l) + ".ffn_gate_up.weight") != nullptr);
    }
}

TEST(attention_shapes_at_the_27b_dimensions) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* q_dim = 24 * 256 = 6144; the projection is twice that, query and gate interleaved. */
    const RecWeight* qg = find_w(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->d.rank, 2u);
    CHECK_EQ(qg->d.shape[0], 12288);
    CHECK_EQ(qg->d.shape[1], 5120);
    CHECK_EQ(qg->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(qg->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(qg->d.access, RAD_ACCESS_PER_TOKEN);

    /* k and v are separate projections, 4 kv heads of 256. */
    const RecWeight* kw = find_w(b, "blk.3.attn_k.weight");
    const RecWeight* vw = find_w(b, "blk.3.attn_v.weight");
    CHECK(kw != nullptr && vw != nullptr);
    CHECK_EQ(kw->d.shape[0], 1024);
    CHECK_EQ(vw->d.shape[0], 1024);
    CHECK_EQ(kw->d.shard, RAD_SHARD_ROW);

    const RecWeight* o = find_w(b, "blk.3.attn_output.weight");
    CHECK(o != nullptr);
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 6144);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* Per-head gains are head_dim wide and replicated, one vector shared by all heads. */
    const RecWeight* qn = find_w(b, "blk.3.attn_q_norm.weight");
    CHECK(qn != nullptr);
    CHECK_EQ(qn->d.shape[0], 256);
    CHECK_EQ(qn->d.shard, RAD_SHARD_NONE);
    CHECK_EQ(qn->d.dtype, (uint32_t)RAD_F32);

    /* The interleaved projection buffer is rank 3, [tokens, heads, 2*head_dim], so query and gate
     * are addressable as column slices. */
    const RecBuf* qgb = find_b(b, "attn_qg");
    CHECK(qgb != nullptr);
    CHECK_EQ(qgb->d.rank, 3u);
    CHECK_EQ(qgb->d.shape[0], 256);
    CHECK_EQ(qgb->d.shape[1], 24);
    CHECK_EQ(qgb->d.shape[2], 512);

    /* qk_norm_rope_gate must stay undeclared until something builds its cos/sin table (see
     * attn_layer_ops); the four ops it stands for carry the same numbers. */
    CHECK(find_op(b, "qk_norm_rope_gate") == nullptr);

    const RecOp* qn2 = find_op(b, "rope");
    CHECK(qn2 != nullptr);
    if (qn2) {
        CHECK_EQ(pint(qn2, "head_dim"), 256);
        CHECK_EQ(pint(qn2, "rotary_dim"), 64);   /* partial_rotary_factor 0.25 of 256 */
        CHECK_EQ(pstr(qn2, "mode"), std::string("mrope"));
        CHECK_NEAR(pflt(qn2, "theta"), 1e7, 1.0);
    }
    const RecOp* at = find_op(b, "attn_paged");
    CHECK_EQ(pint(at, "head_dim"), 256);
    CHECK_EQ(pint(at, "gqa"), 6);
    CHECK_EQ(pint(at, "causal"), 1);
    CHECK_EQ(pint(at, "window"), 0);
    /* block_size is declared deliberately. It is RAD_DERIVED (kernel-supplied) and every libr4d
     * attention row pins C_EQ("block_size", 16), so 16 changes nothing on the accelerated path.
     * But a geometry no accelerated row serves (e.g. head_dim 256 at gqa 4) resolves to libref,
     * which accepts any block size and names none, leaving the KV group unsizable. The block size
     * belongs to the cache the model asks for, so the model declares it. */
    CHECK_EQ(pint(at, "block_size"), 16);

    /* ...and it reaches kv_store, which needs it as a parameter. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ(pint(ks, "block_size"), 16);
    CHECK_EQ(pint(ks, "n_head_kv"), 4);

    /* The gate: a per-head sigmoid over head_dim, then an elementwise multiply over the attention
     * output, `out = attn * sigmoid(gate)`. */
    const RecOp* sg = find_op(b, "sigmoid");
    CHECK_EQ(pint(sg, "n"), 256);
    CHECK_EQ(param(sg, "M")->ihi, 256LL * 24);
    const RecOp* mu = find_op(b, "mul");
    CHECK_EQ(pint(mu, "n"), 6144);
}

TEST(gdn_shapes_and_both_paths) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* conv_dim = 2*16*128 + 48*128 = 10240, followed by one `a` and one `b` per value head:
     * 10240 + 96 = 10336. The fusion lets gdn_conv_prep read the decay and beta inputs from `x`. */
    const RecWeight* in = find_w(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->d.shape[0], 10336);
    CHECK_EQ(in->d.shape[1], 5120);
    CHECK_EQ(in->d.shard, RAD_SHARD_ROW);

    const RecWeight* z = find_w(b, "blk.0.ssm_z.weight");
    CHECK(z != nullptr);
    CHECK_EQ(z->d.shape[0], 6144);            /* 48 value heads of 128 */

    const RecWeight* cv = find_w(b, "blk.0.ssm_conv1d.weight");
    CHECK(cv != nullptr);
    CHECK_EQ(cv->d.rank, 2u);
    CHECK_EQ(cv->d.shape[0], 10240);          /* depthwise: one filter per convolved channel */
    CHECK_EQ(cv->d.shape[1], 4);
    CHECK_EQ(cv->d.shard, RAD_SHARD_ROW);
    /* Must be BF16, and nothing else checks it: libr4d's conv reads this pointer as bf16, so an
     * f32 plane would be misread on the device path only (libref reads the operand's own dtype). */
    CHECK_EQ(cv->d.dtype, (uint32_t)RAD_BF16);
    /* Qwen3.5's conv1d has no bias, so none is declared; the operand is passed absent. */
    CHECK(find_w(b, "blk.0.ssm_conv1d.bias") == nullptr);

    /* The decay weight is named for its convention: it is a log and the kernels take exp(). */
    const RecWeight* al = find_w(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->d.shape[0], 48);
    CHECK_EQ(al->d.dtype, (uint32_t)RAD_F32);
    CHECK(find_w(b, "blk.0.ssm_a") == nullptr);

    const RecWeight* on = find_w(b, "blk.0.ssm_norm.weight");
    CHECK(on != nullptr);
    CHECK_EQ(on->d.shape[0], 128);            /* over head_v, shared by every head */
    CHECK_EQ(on->d.shard, RAD_SHARD_NONE);

    const RecWeight* ow = find_w(b, "blk.0.ssm_out.weight");
    CHECK(ow != nullptr);
    CHECK_EQ(ow->d.shape[0], 5120);
    CHECK_EQ(ow->d.shape[1], 6144);
    CHECK_EQ(ow->d.shard, RAD_SHARD_COL);

    /* Both the chunked and recurrent paths are declared, so both appear in the graph dump and
     * resolve at declare. */
    CHECK_EQ(count_op(b, "gdn_conv_prep"), 6);
    CHECK_EQ(count_op(b, "gdn_kkt_solve"), 6);
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "gdn_gated_rmsnorm"), 6);
    CHECK_EQ(count_op(b, "gdn_conv_update"), 6);
    CHECK_EQ(count_op(b, "gdn_recurrent_update"), 6);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ(pint(cp, "head_k"), 128);
    CHECK_EQ(pint(cp, "head_v"), 128);
    CHECK_EQ(pint(cp, "conv_width"), 4);
    CHECK_EQ(pint(cp, "chunk"), 64);
    CHECK_EQ(param(cp, "M")->kind, RAD_P_RANGE);
    /* Weight operands: the conv weight and the two per-head vectors; no bias. */
    CHECK_EQ((int)cp->w.size(), 3);

    /* chunk is RAD_DERIVED (kernel-chosen) on the two ops that tile on it. */
    CHECK(param(find_op(b, "gdn_kkt_solve"), "chunk") == nullptr);
    CHECK(param(find_op(b, "gdn_chunk_scan"), "chunk") == nullptr);

    /* The recurrent path is ranged on q_len, since a speculative verify carries n_spec+1 rows per
     * sequence, and states n_head_v explicitly: with 48 value heads against 16 key heads, inferring
     * it from a flattened view gives a plausible wrong answer. */
    const RecOp* ru = find_op(b, "gdn_recurrent_update");
    CHECK_EQ(param(ru, "q_len")->kind, RAD_P_RANGE);
    CHECK_EQ(pint(ru, "n_head_v"), 48);
    CHECK_EQ(pstr(ru, "act"), std::string("silu"));
    CHECK_EQ(pint(find_op(b, "gdn_conv_update"), "conv_width"), 4);

    /* q and k have 16 heads and v 48: the 3:1 sharing is in the buffer shapes. */
    const RecBuf* gq = find_b(b, "gdn_q");
    const RecBuf* gv = find_b(b, "gdn_v");
    CHECK(gq != nullptr && gv != nullptr);
    CHECK_EQ(gq->d.shape[1], 16);
    CHECK_EQ(gq->d.shape[2], 128);
    CHECK_EQ(gv->d.shape[1], 48);
    CHECK_EQ(gv->d.shape[2], 128);
    /* Decay and beta are f32: the gate is a cumulative sum spanning hundreds of nats, and every
     * decay factor is an exponent of it.
     *
     * The chunk inverse is bf16 by contract: libr4d's gdn_kkt_solve writes it and gdn_chunk_scan
     * reads it as bf16, so f32 would double its size and misread every element. libref reads each
     * operand's own dtype, so only the device path would show it. */
    CHECK_EQ(find_b(b, "gdn_g")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_beta")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.shape[2], 64);
}

TEST(three_kv_groups_with_every_layer_bound) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_attn"));
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_gdn_state"));
    CHECK_EQ(b.kv_groups[2].name, std::string("kv_gdn_conv"));

    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.kind, RAD_KV_FULL);
    CHECK_EQ(f.n_head_kv, 4);
    CHECK_EQ(f.head_dim, 256);
    CHECK_EQ(f.dtype, (uint32_t)RAD_BF16);

    /* 48 heads of 128x128 fp32: 3 MiB per layer, 144 MiB per sequence across the real model's 48
     * linear layers. fp32 because that is the state width libr4d has a kernel for. */
    const RadKVGroupDecl& s = b.kv_groups[1].d;
    CHECK_EQ(s.kind, RAD_KV_LINEAR);
    CHECK_EQ(s.n_head_kv, 48);
    CHECK_EQ(s.state_dim[0], 128);
    CHECK_EQ(s.state_dim[1], 128);
    CHECK_EQ(s.dtype, (uint32_t)RAD_F32);

    /* The conv window is counted in heads (2*16 + 48 of width 128 = 10240 channels), so a
     * tensor-parallel split never straddles a q/k/v boundary. */
    const RadKVGroupDecl& c = b.kv_groups[2].d;
    CHECK_EQ(c.kind, RAD_KV_CONV);
    CHECK_EQ(c.n_head_kv, 80);
    CHECK_EQ(c.head_dim, 128);
    CHECK_EQ(c.conv_width, 4);

    /* Every layer is bound to the groups it owns state in; a linear layer owns two. */
    std::map<rad_kvgroup, std::set<int>> bound;
    for (const auto& p : b.binds) bound[p.second].insert(p.first);
    CHECK_EQ((int)bound[1].size(), 2);          /* full attention: layers 3 and 7 */
    CHECK_EQ((int)bound[2].size(), 6);
    CHECK_EQ((int)bound[3].size(), 6);
    CHECK(bound[1].count(3) && bound[1].count(7));
    CHECK(bound[2].count(0) && bound[2].count(6));
    CHECK(bound[2] == bound[3]);                /* the two GDN groups cover the same layers */
    for (int l = 0; l < 8; ++l)
        CHECK_EQ(bound[1].count(l) + bound[2].count(l), (size_t)1);
}

TEST(the_name_map_owns_both_sides) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The interleaved query-and-gate projection is one checkpoint tensor and one declared weight. */
    const RecMap* qg = find_map(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->mode, RAD_MAP_COPY);
    CHECK_EQ((int)qg->src.size(), 1);
    CHECK_EQ(qg->src[0],
             std::string("model.language_model.layers.3.self_attn.q_proj.weight"));

    /* The GDN in-projection is a convert-time fusion of qkv, a and b, in gdn_conv_prep's row
     * order. */
    const RecMap* in = find_map(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)in->src.size(), 3);
    CHECK_EQ(in->src[0],
             std::string("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    CHECK_EQ(in->src[1],
             std::string("model.language_model.layers.0.linear_attn.in_proj_a.weight"));
    CHECK_EQ(in->src[2],
             std::string("model.language_model.layers.0.linear_attn.in_proj_b.weight"));
    CHECK_EQ(in->concat_dim, 0);

    /* A_log maps straight through; the checkpoint tensor is already a log. */
    const RecMap* al = find_map(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->src[0], std::string("model.language_model.layers.0.linear_attn.A_log"));

    /* gate and up are one weight and one GEMM. */
    const RecMap* gu = find_map(b, "blk.1.ffn_gate_up.weight");
    CHECK(gu != nullptr);
    CHECK_EQ(gu->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)gu->src.size(), 2);
    CHECK_EQ(gu->src[0], std::string("model.language_model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(gu->src[1], std::string("model.language_model.layers.1.mlp.up_proj.weight"));

    /* The feed-forward's norm is the post-attention norm, not the input norm. */
    const RecMap* fn = find_map(b, "blk.1.ffn_norm.weight");
    CHECK(fn != nullptr);
    CHECK_EQ(fn->src[0],
             std::string("model.language_model.layers.1.post_attention_layernorm.weight"));

    /* This model does not tie its lm_head; the second source serves family members that do. */
    const RecMap* lm = find_map(b, "output.weight");
    CHECK(lm != nullptr);
    CHECK_EQ(lm->mode, RAD_MAP_COPY);
    CHECK_EQ((int)lm->src.size(), 2);
    CHECK_EQ(lm->src[0], std::string("lm_head.weight"));
    CHECK_EQ(lm->src[1], std::string("model.language_model.embed_tokens.weight"));

    /* Every declared weight has a map entry, so rad-convert can fill all of them. */
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A transient without declared uses has no liveness, so the planner must keep it live for the
     * whole program. A GDN layer has eleven intermediates, so this decides the arena size. */
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ((int)cp->reads.size(), 1);      /* the fused projection row */
    CHECK_EQ((int)cp->writes.size(), 5);     /* q, k, v, g, beta */

    /* kv_store writes the group's pool, not a declared buffer, so its write set is empty by
     * design; it reads the two projections it stores. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ((int)ks->reads.size(), 2);
    CHECK_EQ((int)ks->writes.size(), 0);

    /* The residual add reads the stream and the block output and writes the stream. */
    const RecOp* ad = find_op(b, "add");
    CHECK_EQ((int)ad->reads.size(), 2);
    CHECK_EQ((int)ad->writes.size(), 1);
    CHECK_EQ(ad->writes[0], ad->reads[0]);
}

/* The gated attention layer runs the four ops the undeclared fused op stands for; no refusal in
 * the fake hierarchy is needed to reach this sequence. */
TEST(the_gated_attention_layer_runs_the_unfused_sequence) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* The per-head norms are rmsnorm over head_dim with M ranged over tokens x heads. The q norm
     * reads the strided interleaved projection and writes the contiguous q buffer, de-interleaving
     * and normalising in one pass. */
    const RecOp* an = nullptr;      /* the first attention layer's q norm */
    for (size_t i = 0; i + 1 < b.ops.size(); ++i)
        if (b.ops[i].op == "rope") { an = &b.ops[i - 2]; break; }
    CHECK(an != nullptr);
    if (!an) return;
    CHECK_EQ(an->op, std::string("rmsnorm"));
    CHECK_EQ(pint(an, "n"), 256);
    CHECK_EQ(param(an, "M")->ihi, 256LL * 24);
    CHECK_EQ((an + 1)->op, std::string("rmsnorm"));
    CHECK_EQ(param(an + 1, "M")->ihi, 256LL * 4);

    const RecOp* rq = an + 2;
    CHECK_EQ(rq->op, std::string("rope"));
    CHECK_EQ(pint(rq, "head_dim"), 256);
    CHECK_EQ(pint(rq, "n_head"), 24);
    CHECK_EQ(pint(rq, "n_head_kv"), 0);      /* q only: no k or v heads */
    CHECK_EQ(pint(rq, "rotary_dim"), 64);
    CHECK_EQ(pstr(rq, "mode"), std::string("mrope"));
    CHECK_NEAR(pflt(rq, "theta"), 1e7, 1.0);

    const RecOp* rk = an + 3;
    CHECK_EQ(rk->op, std::string("rope"));
    CHECK_EQ(pint(rk, "n_head"), 4);         /* the k row, 4 heads */
    CHECK_EQ(pint(rk, "n_head_kv"), 0);

    /* The step issues them, once per full-attention layer. */
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    int n_rope = 0;
    for (const auto& i : c.issues)
        if (b.op_of(i.op) && b.op_of(i.op)->op == "rope") ++n_rope;
    CHECK_EQ(n_rope, 4);                     /* two per attention layer, two layers */

    /* The sigmoid reads the interleaved projection: a column slice at offset head_dim, width
     * head_dim, of the buffer whose declared width is the pitch. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "sigmoid") continue;
        CHECK_EQ(i.opd[0].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(i.opd[0].offset, 256);      /* the gate half of the head's pair */
        CHECK_EQ(i.opd[0].cols, 256);
        CHECK_EQ(i.opd[0].rows, batch.n_tok);
    }
}

TEST(tensor_parallel_divides_dimensions_and_places_the_collectives) {
    RadBuilder b0, b1;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  c0 = build_ctx(0, 2), c1 = build_ctx(1, 2);
    CHECK_OK(qwen35_bf16::declare(&b0, &meta, &c0));
    CHECK_OK(qwen35_bf16::declare(&b1, &meta, &c1));

    /* Declare runs once per rank, and the plugin declares already-divided dimensions. */
    const RecWeight* qg = find_w(b0, "blk.3.attn_qg.weight");
    CHECK_EQ(qg->d.shape[0], 6144);           /* 12 heads x 256 x 2 */
    CHECK_EQ(qg->d.shape[1], 5120);           /* the residual stream is not sharded */
    CHECK_EQ(find_w(b0, "blk.3.attn_k.weight")->d.shape[0], 512);      /* 2 kv heads */
    const RecWeight* o = find_w(b0, "blk.3.attn_output.weight");
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 3072);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* The GDN row halves by head: 8 key and 24 value heads give 2*8*128 + 24*128 = 5120 convolved
     * channels, plus 48 for a and b. */
    const RecWeight* in = find_w(b0, "blk.0.ssm_in.weight");
    CHECK_EQ(in->d.shape[0], 5168);
    CHECK_EQ(find_w(b0, "blk.0.ssm_z.weight")->d.shape[0], 3072);
    CHECK_EQ(find_w(b0, "blk.0.ssm_conv1d.weight")->d.shape[0], 5120);
    CHECK_EQ(find_w(b0, "blk.0.ssm_a_log")->d.shape[0], 24);
    CHECK_EQ(find_w(b0, "blk.0.ssm_out.weight")->d.shape[1], 3072);
    /* The gain vectors are head-width and stay whole. */
    CHECK_EQ(find_w(b0, "blk.0.ssm_norm.weight")->d.shape[0], 128);
    CHECK_EQ(find_w(b0, "blk.3.attn_q_norm.weight")->d.shape[0], 256);

    const RecWeight* gu = find_w(b0, "blk.0.ffn_gate_up.weight");
    CHECK_EQ(gu->d.shape[0], 17408);          /* 2 * (17408 / 2) */
    CHECK_EQ(find_w(b0, "blk.0.ffn_down.weight")->d.shape[1], 8704);

    /* gqa and head width are rank-invariant. */
    CHECK_EQ(pint(find_op(b0, "attn_paged"), "gqa"), 6);
    CHECK_EQ(pint(find_op(b0, "gdn_conv_prep"), "head_k"), 128);
    CHECK_EQ(pint(find_op(b0, "gdn_recurrent_update"), "n_head_v"), 24);

    /* The KV groups halve with the heads they hold. */
    CHECK_EQ(b0.kv_groups[0].d.n_head_kv, 2);
    CHECK_EQ(b0.kv_groups[1].d.n_head_kv, 24);
    CHECK_EQ(b0.kv_groups[2].d.n_head_kv, 40);   /* 2*8 + 24 */

    /* Two collectives per layer (the block's and the feed-forward's column-parallel projections)
     * plus one for the vocab-sharded embedding, and no others. */
    CHECK_EQ(count_op(b0, "all_reduce"), 17);
    CHECK_EQ(join(op_names(b0)), join(program_ops(/*tp=*/true)));

    const RecOp* ar = find_op(b0, "all_reduce");
    CHECK_EQ(pint(ar, "world_size"), 2);
    CHECK_EQ(pint(ar, "exact"), 1);           /* must be stated, or no kernel matches */
    CHECK_EQ(param(ar, "numel")->kind, RAD_P_RANGE);
    CHECK_EQ(param(ar, "numel")->ihi, 256LL * 5120);

    /* The vocabulary split is ragged: ceil(total/world) rows on every rank but the last, which
     * takes the rest. This must match the rule core/engine.cpp restates and checks; since 248320
     * divides by two, the sum and shard are asserted rather than a value "floor" would also give. */
    const RecWeight* lm0 = find_w(b0, "output.weight");
    const RecWeight* lm1 = find_w(b1, "output.weight");
    CHECK_EQ(lm0->d.shape[0] + lm1->d.shape[0], 248320);
    CHECK_EQ(lm0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(lm0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(pint(find_op(b0, "logits_gemm"), "n_vocab"), lm0->d.shape[0]);
    /* The draft head stays whole on every rank: its argmax is a row_topk over one rank's plane
     * that bypasses the sampler's merge, so a sharded plane would only propose tokens from that
     * rank's half. */
    if (const RecWeight* dh = find_w(b0, "mtp.draft_head.weight")) {
        CHECK_EQ(dh->d.shape[0], 248320);
        CHECK_EQ(dh->d.shard, RAD_SHARD_NONE);
    }
    /* The embedding uses the same ragged split as the lm_head (whole, it would be gigabytes of
     * replicated bf16 per card). The shards sum to the vocabulary and each rank's embed_lookup gets
     * its slice's vocab_offset. */
    const RecWeight* te0 = find_w(b0, "token_embd.weight");
    const RecWeight* te1 = find_w(b1, "token_embd.weight");
    CHECK_EQ(te0->d.shape[0] + te1->d.shape[0], 248320);
    CHECK_EQ(te0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(te0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te0->d.access, RAD_ACCESS_VOCAB);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "n_vocab"), te0->d.shape[0]);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "vocab_offset"), 0);
    CHECK_EQ(pint(find_op(b1, "embed_lookup"), "vocab_offset"), te0->d.shape[0]);

    /* A rank count that does not divide the heads is refused, not rounded. */
    RadBuilder b5;
    RadBuildCtx c5 = build_ctx(0, 5);
    CHECK_EQ(qwen35_bf16::declare(&b5, &meta, &c5), RAD_E_INVAL);

    /* With one rank there are no collectives. */
    RadBuilder bs;
    RadBuildCtx cs = build_ctx(0, 1);
    CHECK_OK(qwen35_bf16::declare(&bs, &meta, &cs));
    CHECK_EQ(count_op(bs, "all_reduce"), 0);
}

TEST(a_decode_step_issues_both_layer_types_in_order) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) {
            /* Issue order differs from declare order in one place: kv_store writes the cache
             * before attn_paged reads it. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                  "rmsnorm", "rmsnorm", "rope", "rope",
                                  "kv_store", "attn_paged",
                                  "sigmoid", "mul", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        } else {
            /* Decode takes the recurrent path, which folds in the gated norm. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt",
                                  "gdn_conv_update", "gdn_recurrent_update", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt", "add" })
            want.push_back(s);
    }
    want.push_back("rmsnorm");
    want.push_back("gather_rows");
    want.push_back("logits_gemm");
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The ranged value: RAD_N_BATCH for token-ranged ops, the query length for q_len-banded ops,
     * an explicit row count where rows are tokens x heads, and n_out for the two vocabulary ops,
     * whose band is the number of sampled rows, not n_tok. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged" || o->op == "gdn_conv_update" ||
            o->op == "gdn_recurrent_update")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "sigmoid")
            CHECK_EQ(i.n, batch.n_tok * 24);
        else if (o->op == "rmsnorm")
            /* Three rmsnorm row counts: layer pre-norms over tokens (RAD_N_BATCH), and the
             * per-head q and k norms at tokens x n_head and tokens x n_head_kv. */
            CHECK(i.n == RAD_N_BATCH || i.n == batch.n_tok * 24 || i.n == batch.n_tok * 4);
        else if (o->op == "gather_rows" || o->op == "logits_gemm")
            CHECK_EQ(i.n, batch.n_out);
        else
            CHECK_EQ(i.n, RAD_N_BATCH);
    }

    /* Every operand is a declared handle or a batch pointer, and every buffer operand carries the
     * step's row count, not the declared extent: kernels read extents from operands, so an
     * un-narrowed operand would compute a one-row decode over the whole arena. */
    for (const auto& i : c.issues)
        for (const auto& od : i.opd) {
            if (od.kind == RAD_OPK_BUF) {
                CHECK(od.handle >= 1 && od.handle <= b.bufs.size());
                CHECK_EQ(od.rows, batch.n_tok);
            }
            if (od.kind == RAD_OPK_WEIGHT) CHECK(od.handle >= 1 && od.handle <= b.weights.size());
        }
}

TEST(a_prefill_step_takes_the_chunked_gdn_path) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = prefill_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> got = issued_ops(b, c);
    int n_prep = 0, n_scan = 0, n_recur = 0, n_gnorm = 0;
    for (const auto& s : got) {
        if (s == "gdn_conv_prep")        ++n_prep;
        if (s == "gdn_chunk_scan")       ++n_scan;
        if (s == "gdn_recurrent_update") ++n_recur;
        if (s == "gdn_gated_rmsnorm")    ++n_gnorm;
    }
    CHECK_EQ(n_prep, 6);
    CHECK_EQ(n_scan, 6);
    CHECK_EQ(n_gnorm, 6);
    CHECK_EQ(n_recur, 0);      /* declared but not issued in prefill */

    /* The decay and beta inputs are the last 2*n_head_v columns of the projection row: read
     * inside gdn_conv_prep on the chunked path, as strided views on the recurrent one. */
    const RecOp* cp = nullptr;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_conv_prep") {
            cp = o;
            /* Sixteen operands, inputs before outputs (docs/OPS.md, "Linear attention").
             * `a` and `b_gate` (the recurrent path's strided views) are passed although optional:
             * libr4d dereferences them and `cache_idx` unconditionally, so omitting them would
             * silently resolve the layer to libref. */
            CHECK_EQ((int)i.opd.size(), 16);
            CHECK_EQ(i.opd[2].kind, (uint8_t)RAD_OPK_NONE);   /* no conv bias on this model */
            CHECK_EQ(i.opd[5].kind, (uint8_t)RAD_OPK_KV);     /* the conv state */
            CHECK_EQ(i.opd[6].rows, batch.n_seq + 1);         /* cu_seqlens */
            CHECK_EQ(i.opd[7].offset, 10240);                 /* a */
            CHECK_EQ(i.opd[7].cols, 48);
            CHECK_EQ(i.opd[8].offset, 10288);                 /* b_gate */
            CHECK_EQ(i.opd[10].rows, batch.n_seq);            /* has_init == ctx_lens */
            break;
        }
    }
    CHECK(cp != nullptr);

    /* The scan reads and writes the recurrent state group, as h0 and ht. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_chunk_scan") {
            /* Eleven: the state index after the two KV operands makes the scan write the
             * sequence's manager-assigned slot, not its batch row. */
            CHECK_EQ((int)i.opd.size(), 11);
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[9].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[10].rows, batch.n_seq);
            break;
        }
    }
}

TEST(the_recurrent_path_slices_a_and_b_out_of_the_projection_row) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    bool seen = false;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "gdn_recurrent_update") continue;
        seen = true;
        /* Sixteen, as both kernel registries declare for gdn_recurrent_update and
         * core/runtime/issue.cpp requires. The count comes from the schema, since this harness
         * only records operands. The last two (the folded out-projection quantiser) are unused by
         * the bf16 block; operands are positional, so an absent optional operand is a null
         * operand, not a shorter list. */
        CHECK_EQ((int)i.opd.size(), 16);
        CHECK_EQ(i.opd[14].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[15].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[3].offset, 10240);      /* a: first column after the conv row */
        CHECK_EQ(i.opd[3].cols, 48);
        CHECK_EQ(i.opd[4].offset, 10288);      /* b: one per value head after a */
        CHECK_EQ(i.opd[4].cols, 48);
        CHECK_EQ(i.opd[7].kind, (uint8_t)RAD_OPK_KV);
        break;
    }
    CHECK(seen);
}

TEST(what_the_plugin_refuses) {
    RadModelMeta meta = qwen35_meta();

    /* A drafter would need a second set of per-rank handles and name-map namespace; nothing
     * unimplemented returns success (spec §17). */
    RadBuilder bd;
    RadBuildCtx cd = build_ctx();
    cd.is_draft = 1;
    CHECK_EQ(qwen35_bf16::declare(&bd, &meta, &cd), RAD_E_UNSUPPORTED);

    /* A routed (MoE) checkpoint is a different graph. */
    RadBuilder bm;
    RadModelMeta moe = meta;
    moe.n_expert = 128;
    moe.n_expert_used = 8;
    RadBuildCtx cm = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bm, &moe, &cm), RAD_E_UNSUPPORTED);

    /* A container that does not describe the linear layers is refused, not defaulted; a guessed
     * head count would load and produce wrong text. */
    RadBuilder bn;
    RadModelMeta bare = meta;
    bare.n_kv = 2;                 /* interval and rope width only; no ssm keys */
    RadBuildCtx cn = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bn, &bare, &cn), RAD_E_INVAL);

    /* So is one that omits the rotary width. */
    RadBuilder br;
    RadModelMeta norot = meta;
    static const char* rk[] = { "qwen35.full_attention_interval", "qwen35.ssm.state_size",
                                "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank",
                                "qwen35.ssm.inner_size", "qwen35.ssm.conv_kernel" };
    static const char* rv[] = { "4", "128", "16", "48", "6144", "4" };
    norot.n_kv = 6; norot.kv_key = rk; norot.kv_val = rv;
    RadBuildCtx cr = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&br, &norot, &cr), RAD_E_INVAL);
}

TEST(the_mtp_layer_sits_beside_the_trunk_and_not_inside_it) {
    /* `nextn_predict_layers` is not part of the layer count in this family: Qwen3.8-27B-FP8 sets
     * num_hidden_layers = 64, lists 64 layer_types, ships layers 0..63 under
     * `model.language_model.layers`, and keeps its MTP block in a separate file under `mtp.`.
     * Subtracting it would drop the last (full-attention) trunk layer and silently run one layer
     * short. So every declared layer is a trunk layer and `nextn` does not change the graph. */
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    static const char* mk[] = { "qwen35.full_attention_interval", "qwen35.rope.dimension_count",
                                "qwen35.ssm.state_size", "qwen35.ssm.group_count",
                                "qwen35.ssm.time_step_rank", "qwen35.ssm.inner_size",
                                "qwen35.ssm.conv_kernel", "qwen35.nextn_predict_layers" };
    static const char* mv[] = { "4", "64", "128", "16", "48", "6144", "4", "1" };
    meta.n_kv = 8; meta.kv_key = mk; meta.kv_val = mv;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The same graph as without nextn. */
    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));
    /* The last trunk layer, blk.7, is declared; (7 + 1) % 4 == 0 makes it full attention, the
     * layer a subtraction would drop. */
    CHECK(find_w(b, "blk.7.attn_norm.weight") != nullptr);
    CHECK(find_w(b, "blk.8.attn_norm.weight") == nullptr);
}

/* The PLE windowed ops are issued once per half of the step, and the halves differ in whether the
 * ops see `num_accepted`, which picks their write-back layout. Decode rows may reject drafts, so
 * `ngram_ids` and `ple_conv` must see it; a prefill chunk commits everything, so they must not, and
 * the chunk's tail stays at offset zero for the scheduler's `num_accepted` of 1 next step. A mixed
 * step therefore needs two issues; a pure step issues one. */
TEST(the_ple_windows_are_issued_once_per_half_of_a_step) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_layer = 1; g.n_embd = 64; g.max_tok = 64; g.max_seqs = 4; g.max_spec = 3;
    RadBufDecl hd{};
    hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = g.max_tok; hd.shape[1] = 4 * g.n_embd;
    PleLayer::Wire w{};
    w.h = rad_decl_buffer(&b, "h", &hd);
    PleLayer::Config pc{};
    pc.hc = 4; pc.embed_dim = 32; pc.heads = 4; pc.ngram = 3; pc.eos = 7; pc.rows = 1000;
    pc.shards = 2; pc.conv_width = 4; pc.dilation = 3;
    PleLayer::Src src{};
    src.shard_prefix = "ple.shard_"; src.shard_suffix = ".weight"; src.mult = "ple.mult";
    src.vocab_sizes = "ple.vsz"; src.offsets = "ple.off"; src.key_proj = "ple.k";
    src.value_proj = "ple.v"; src.norm_key = "ple.nk"; src.norm_query = "ple.nq";
    src.norm_conv = "ple.nc"; src.conv1d = "ple.conv";
    PleLayer ple;
    CHECK_OK(ple.declare(&b, nm, g, pc, 0, w, src));

    /* Two decode rows (a verify of 4 and a plain decode of 1) and one prefill chunk of 6. */
    static const int32_t tok[11] = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15 };
    static const int32_t cu[4]   = { 0, 4, 5, 11 };
    static const int32_t ctxl[3] = { 40, 30, 12 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static int32_t tk_idx[3] = { 5, 6, 7 }, cv_idx[3] = { 1, 2, 3 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = ple.kv_tok;  kv[0].state_index = tk_idx;
    kv[1].group = ple.kv_conv; kv[1].state_index = cv_idx;

    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 11; mixed.n_seq = 3;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 5;
    mixed.token_ids = tok; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c{ .b = &b, .batch = &mixed };
    /* The hash is issued at the top of the step and the rest at the layer. */
    ple.ids(&c, &mixed);
    ple.step(&c, &mixed);

    const char* want[] = { "ngram_ids", "ngram_ids", "embed_lookup_q", "cast", "gemm_nt",
                           "gemm_nt", "ple_gate", "ple_conv", "ple_conv", "add" };
    std::vector<std::string> wv(want, want + 10);
    CHECK_EQ(join(issued_ops(b, c)), join(wv));
    REQUIRE(c.issues.size() == 10);

    /* The projections read the gather's rows from the card: the copy reads the gather's output
     * and both GEMMs read the copy, never the host buffer. */
    {
        const RecIssue& gth = c.issues[2];
        const RecIssue& cp  = c.issues[3];
        REQUIRE(gth.opd.size() == 5 && cp.opd.size() == 2);
        CHECK_EQ(cp.opd[0].handle, gth.opd[3].handle);
        CHECK_EQ(cp.opd[0].rows, (int64_t)11);
        CHECK(cp.opd[1].handle != gth.opd[3].handle);
        CHECK_EQ(c.issues[4].opd[0].handle, cp.opd[1].handle);
        CHECK_EQ(c.issues[5].opd[0].handle, cp.opd[1].handle);
    }

    /* ngram_ids: tok, state, cu, 3 weights, cache_idx, has_init, num_accepted, ids, ahead_ids. */
    {
        const RecIssue& d = c.issues[0];
        const RecIssue& p = c.issues[1];
        REQUIRE(d.opd.size() == 11 && p.opd.size() == 11);
        /* No following prompt (n_ahead 0); a decode half never has one. */
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[2].raw == (void*)cu);             CHECK_EQ(d.opd[2].rows, (int64_t)3);
        CHECK(d.opd[6].raw == (void*)tk_idx);         CHECK_EQ(d.opd[6].rows, (int64_t)2);
        CHECK(d.opd[7].raw == (void*)ctxl);
        CHECK_EQ(d.opd[8].kind, (uint8_t)RAD_OPK_RAW);
        CHECK(d.opd[8].raw == (void*)nacc);           CHECK_EQ(d.opd[8].rows, (int64_t)2);
        CHECK_EQ(d.opd[9].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);         /* token operands stay at their base */
        CHECK(p.opd[2].raw == (void*)(cu + 2));       CHECK_EQ(p.opd[2].rows, (int64_t)2);
        CHECK(p.opd[6].raw == (void*)(tk_idx + 2));   CHECK_EQ(p.opd[6].rows, (int64_t)1);
        CHECK(p.opd[7].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[8].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
    }
    /* ple_conv: x, w, state, cu, resid, cache_idx, has_init, num_accepted, y. */
    {
        const RecIssue& d = c.issues[7];
        const RecIssue& p = c.issues[8];
        REQUIRE(d.opd.size() == 9 && p.opd.size() == 9);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[3].raw == (void*)cu);
        CHECK(d.opd[5].raw == (void*)cv_idx);
        CHECK(d.opd[7].raw == (void*)nacc);
        CHECK_EQ(d.opd[8].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);
        CHECK(p.opd[3].raw == (void*)(cu + 2));
        CHECK(p.opd[5].raw == (void*)(cv_idx + 2));
        CHECK(p.opd[6].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[7].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[8].rows, (int64_t)11);
    }

    /* A pure step issues one half: a decode step passes `num_accepted`, a prefill step does not. */
    for (int phase : { (int)RAD_PHASE_DECODE, (int)RAD_PHASE_PREFILL }) {
        RadBatch pure = mixed;
        pure.phase = phase;
        pure.n_seq_decode = 0; pure.n_tok_decode = 0;   /* the phase decides, not these counts */
        RadCtx cp{ .b = &b, .batch = &pure };
        ple.ids(&cp, &pure);
        ple.step(&cp, &pure);
        int n_ids = 0, n_conv = 0;
        for (const auto& i : cp.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            const bool ids = o->op == "ngram_ids", conv = o->op == "ple_conv";
            if (!ids && !conv) continue;
            n_ids += ids; n_conv += conv;
            CHECK_EQ(i.n, (int64_t)11);
            const RadOperand& na = i.opd[ids ? 8 : 7];
            if (phase == RAD_PHASE_DECODE) CHECK(na.kind == RAD_OPK_RAW && na.raw == (void*)nacc);
            else                           CHECK_EQ(na.kind, (uint8_t)RAD_OPK_NONE);
        }
        CHECK_EQ(n_ids, 1);
        CHECK_EQ(n_conv, 1);
    }

    /* A following prompt rides the prefill half only: its ngram_ids reads `tok` that many tokens
     * further and writes their ids into the `ple_ids` rows after the step's, which the gather also
     * gets. The decode half never sees them. */
    {
        RadBatch ahead = mixed;
        ahead.n_ahead = 7;
        RadCtx ca{ .b = &b, .batch = &ahead };
        ple.ids(&ca, &ahead);
        ple.step(&ca, &ahead);
        REQUIRE(ca.issues.size() == 10);
        const RecIssue& d = ca.issues[0];
        const RecIssue& p = ca.issues[1];
        const RecIssue& gth = ca.issues[2];
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)(11 + 7));
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(p.opd[10].rows, (int64_t)7);
        CHECK_EQ(p.opd[10].offset, (int64_t)(11 * pc.heads));
        REQUIRE(gth.opd.size() == 5);
        CHECK_EQ(gth.opd[4].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(gth.opd[4].handle, p.opd[10].handle);
        CHECK_EQ(gth.opd[4].rows, (int64_t)7);
        CHECK_EQ(gth.opd[4].offset, (int64_t)(11 * pc.heads));
    }
}

/* ==================================================================== the sizing declare
 *
 * The activation arena is lent out by level, and a level is the plugin declaring again at a
 * smaller max_tok. The plugin must keep this safe: the sizing declare must not replace the state
 * its step reads (written at the real size), and must describe the same buffers with only their
 * rows smaller, since every other extent is a stride the real kernels were chosen for.
 *
 * Qwen3.8-Flash-Next's metadata, cut to eight layers. */
namespace {
const char* q4_keys[] = {
    "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram",
    "hidden_act", "indexer_budget", "indexer_compress_ratio", "indexer_head_dim",
    "indexer_n_heads", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "make_ngram_vocab_size_divisible_by",
    "moe_intermediate_size", "mtp_num_hidden_layers", "ngram_size", "ngram_vocab_size_base",
    "num_experts", "num_experts_per_tok", "output_gate_type", "partial_rotary_factor",
    "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids", "shared_expert_intermediate_size",
    "split_ngram_parts", "rope_parameters.partial_rotary_factor", "eos_token_id",
};
const char* q4_vals[] = {
    "4", "4", "320", "8",
    "silu", "2048", "4", "128",
    "4",
    "linear_attention linear_attention linear_attention full_attention "
    "linear_attention linear_attention linear_attention full_attention",
    "4", "128",
    "16", "48", "128", "128",
    "640", "1", "3", "20000000",
    "512", "10", "sigmoid", "0.25",
    "4", "2560", "2", "640",
    "128", "0.25", "248044",
};

RadModelMeta qwen4exp_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen4exp";
    m.name        = "test-q38-flashnext";
    m.quant       = "";
    m.n_layers    = 8;
    m.n_embd      = 2560;
    m.n_head      = 24;
    m.n_head_kv   = 2;
    m.head_dim    = 256;
    m.n_ff        = 640;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(q4_keys) / sizeof(q4_keys[0]));
    m.kv_key      = q4_keys;
    m.kv_val      = q4_vals;
    return m;
}
}  /* namespace */

TEST(qwen4exp_yarn_reaches_attention_indexer_and_compressed_keys) {
    RadModelMeta meta = qwen4exp_meta();
    std::vector<const char*> keys(meta.kv_key, meta.kv_key + meta.n_kv);
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    keys.insert(keys.end(), {"text_config.rope_parameters.rope_type", "text_config.rope_parameters.factor",
                            "text_config.rope_parameters.original_max_position_embeddings"});
    vals.insert(vals.end(), {"yarn", "2", "262144"});
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 524288;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int direct = 0, table = 0, compressed = 0;
    for (const auto& op : b.ops) {
        if (op.op == "rope") ++direct;
        else if (op.op == "rope_table") ++table;
        else if (op.op == "qsa_block_key") ++compressed;
        else continue;
        int found = 0;
        for (const auto& p : op.p) {
            if (p.key == "yarn_factor") { CHECK_EQ(p.dval, 2.0); ++found; }
            if (p.key == "yarn_magnitude") { CHECK_NEAR(p.dval, 1.069314718, 1e-7); ++found; }
            if (p.key == "scale") CHECK_EQ(p.dval, 1.0);
        }
        CHECK_EQ(found, 2);
    }
    CHECK(direct > 0); CHECK(table > 0); CHECK(compressed > 0);
}

/* The plugin declares by each weight's stored encoding. Flash-Next's experts, gated-residual
 * mixing matrices, MTP fc and 2-bit draft head each have a recipe-chosen format the plugin reads
 * from rad_weight_encoding, at convert and at load alike. With int4-rotated, E4M3-row and u2
 * encodings it declares the rotated expert kernels, the E4M3 mix and the 2-bit head; with no model
 * behind it, the fp8 experts and the bf16 mix. */
TEST(the_encodings_the_model_holds_choose_the_flash_next_kernels) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding row8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&row8) && rad_enc_valid(&u2));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 },
               { "_hc_down.weight", row8 }, { "_hc_up.weight", row8 },
               { "fc_hidden", row8 }, { "fc_embedding", row8 }, { "draft_head", u2 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    CHECK_EQ(m.moecfg.expert_rot, 1);
    CHECK(m.layers[0].hc_mix.mix_e4m3);
    int rotated = 0, fp8_experts = 0, head_views = 0, scale_views = 0;
    for (const RecOp& o : b.ops)
        if (o.op.rfind("moe_gemm", 0) == 0)
            for (const auto& p : o.p) {
                if (p.key == "dtype" && p.sval == "w4a8h") ++rotated;
                if (p.key == "dtype" && p.sval == "fp8a8") ++fp8_experts;
            }
    for (const RecWeight& w : b.weights) {
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") ++head_views;
        /* every expert's scale is a view of its logical weight, not a separate weight */
        if (w.name.find("_exps.") != std::string::npos &&
            w.name.size() > 6 && w.name.compare(w.name.size() - 6, 6, ".scale") == 0) {
            ++scale_views;
            CHECK(w.d.source != nullptr && w.d.planes && std::string(w.d.planes) == "scale");
        }
    }
    CHECK(rotated > 0);
    CHECK_EQ(fp8_experts, 0);
    CHECK_EQ(head_views, 1);
    CHECK(scale_views > 0);

    RadBuilder plain;
    REQUIRE_EQ(qwen4exp_fp8::declare(&plain, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& p = qwen4exp_fp8::g_model[0];
    CHECK_EQ(p.moecfg.expert_w4, 0);
    CHECK(!p.layers[0].hc_mix.mix_e4m3);
}

TEST(mxfp4_expert_declarations_preserve_tp4_and_tp8_scale_geometry_and_movement_groups) {
    using namespace rad::arch;
    const RadEncoding mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    REQUIRE(rad_enc_valid(&mx));
    CHECK_EQ(MoeFP8::format_of(mx), 8);
    for (int world : { 4, 8 })
        for (bool glm : { false, true })
            for (int rank = 0; rank < world; ++rank) {
                const int64_t F = 2048 / world;
                RadBuilder b;
                b.encs = { { "ffn_gate_up_exps", mx },
                           { "ffn_down_exps", mx },
                           { "ffn_gate_up_shexp", mx },
                           { "ffn_down_shexp", mx } };
                Names nm("");
                Geom g{};
                g.n_embd = 4096;
                g.max_tok = 32;
                g.world = world;
                g.rank = rank;
                MoeFP8::Src src{};
                src.router = "mlp.gate.weight";
                src.experts = "mlp.experts";
                src.cfg.n_expert = src.cfg.n_expert_all = 288;
                src.cfg.top_k = 8;
                src.cfg.n_ff_exp = F;
                src.cfg.rows = 32;
                if (glm) {
                    src.router_bias = "mlp.gate.e_score_correction_bias";
                    src.shared = "mlp.shared_experts";
                    src.cfg.sigmoid_router = 1;
                    src.cfg.route_scale = 2.5;
                    src.cfg.swiglu_limit = 10;
                    src.cfg.shared_ungated = 1;
                    src.cfg.n_ff_shared = F;
                }
                for (int q = 0; q < 2; ++q) {
                    src.cfg.ff_lo[q] = rank * F;
                    src.cfg.ff_hi[q] = (rank + 1) * F;
                }
                REQUIRE_EQ(MoeFP8::probe(&b, nm, 3, src, &src.cfg), RAD_OK);
                CHECK_EQ(src.cfg.expert_mx, 1);
                CHECK_EQ(src.cfg.expert_w4, 0);
                MoeFP8::Wire w{};
                auto buf = [&](const char* name, uint32_t dt, int64_t rows, int64_t cols) {
                    return decl_b(&b, name, dt, { rows, cols });
                };
                w.h.x = buf("h", RAD_BF16, 32, 4096);
                REQUIRE_EQ(w.h.declare_qs(&b, nm, g, "h", 4096), RAD_OK);
                w.logits = buf("logits", glm ? RAD_F32 : RAD_BF16, 32, 288);
                w.ids = buf("ids", RAD_I32, 32, 8);
                w.ew = buf("ew", glm ? RAD_F32 : RAD_BF16, 32, 8);
                w.eoff = decl_b(&b, "offset", RAD_I32, { 289 });
                w.ecnt = decl_b(&b, "count", RAD_I32, { 288 });
                w.sorted = decl_b(&b, "sorted", RAD_I32, { 256 });
                w.egu = buf("gate_up", RAD_BF16, 256, 2 * F);
                w.eff.x = buf("ff", RAD_BF16, 256, F);
                REQUIRE_EQ(w.eff.declare_qs(&b, nm, g, "ff", F, 256), RAD_OK);
                w.edn = buf("down", RAD_BF16, 256, 4096);
                if (glm) {
                    w.sgu = buf("shared_gu", RAD_BF16, 32, 2 * F);
                    w.sff.x = buf("shared_ff", RAD_BF16, 32, F);
                    REQUIRE_EQ(w.sff.declare_qs(&b, nm, g, "shared_ff", F), RAD_OK);
                    w.sout = buf("shared_out", RAD_BF16, 32, 4096);
                    w.sgate = buf("shared_one", RAD_BF16, 32, 1);
                }
                MoeFP8 moe;
                REQUIRE_EQ(moe.declare(&b, nm, g, 3, w, src, false, false), RAD_OK);
                CHECK_EQ(moe.ncls, 2);
                if (glm) {
                    CHECK_EQ(b.weights[moe.w_router - 1].d.dtype, RAD_F32);
                    CHECK_EQ(b.ops[moe.op_topk - 1].op, std::string("router_sigmoid_topk"));
                    CHECK_EQ(moe.op_topk_scatter, 0u);
                    CHECK_EQ(moe.op_router_fused, 0u);
                    CHECK_EQ(b.ops[moe.gq_exp.op - 1].op, std::string("swiglu_clamped"));
                    CHECK_EQ(b.ops[moe.gq_sh.op - 1].op, std::string("swiglu_clamped"));
                    CHECK_EQ(b.ops[moe.op_sgate - 1].op, std::string("fill"));
                    CHECK_EQ(moe.w_sgate, 0u);
                    bool plain_shared = false;
                    for (const auto& p : b.ops[moe.op_gather - 1].p)
                        if (p.key == "act") plain_shared = p.sval == "none";
                    CHECK(plain_shared);
                    RadCtx ctx;
                    ctx.b = &b;
                    ctx.rank = rank;
                    ctx.world = world;
                    moe.pass(&ctx, 8, 4, 4);
                    const auto ops = issued_ops(b, ctx);
                    CHECK(std::find(ops.begin(), ops.end(), "router_sigmoid_topk") != ops.end());
                    CHECK(std::find(ops.begin(), ops.end(), "router_topk") == ops.end());
                    CHECK_EQ(std::count(ops.begin(), ops.end(), "swiglu_clamped"), 2);
                    CHECK_EQ(std::count(ops.begin(), ops.end(), "fill"), 1);
                    for (const auto& issue : ctx.issues)
                        if (issue.op == moe.op_sgate) {
                            REQUIRE_EQ(issue.opd.size(), (size_t)1);
                            CHECK_EQ(issue.opd[0].handle, w.sgate);
                            CHECK_EQ(issue.opd[0].offset, 4);
                        }
                }
                for (int e = 0; e < 288; ++e) {
                    const auto& gu = b.weights[moe.w_gu[e] - 1];
                    const auto& gs = b.weights[moe.w_gus[e] - 1];
                    const auto& dn = b.weights[moe.w_dn[e] - 1];
                    const auto& ds = b.weights[moe.w_dns[e] - 1];
                    CHECK_EQ(gu.d.dtype, RAD_FP4E2M1);
                    CHECK_EQ(dn.d.dtype, RAD_FP4E2M1);
                    CHECK_EQ(gs.d.dtype, RAD_E8M0);
                    CHECK_EQ(ds.d.dtype, RAD_E8M0);
                    CHECK_EQ(gs.d.shape[0], 2 * F);
                    CHECK_EQ(gs.d.shape[1], 128);
                    CHECK_EQ(ds.d.shape[0], 4096);
                    CHECK_EQ(ds.d.shape[1], F / 32);
                    CHECK_EQ(gu.span_lo, rank * F);
                    CHECK_EQ(gs.span_lo, rank * F);
                    CHECK_EQ(dn.span_lo, rank * F);
                    CHECK_EQ(ds.span_lo, rank * (F / 32));
                    CHECK_EQ(gu.d.access, RAD_ACCESS_CONDITIONAL);
                    CHECK_EQ(dn.d.access, RAD_ACCESS_CONDITIONAL);
                }
                for (rad_op h : { moe.op_gu, moe.op_dn }) {
                    const RecOp* op = b.op_of(h);
                    REQUIRE(op != nullptr);
                    CHECK_EQ(op->op, std::string("moe_gemm_q"));
                    bool found = false;
                    for (const auto& p : op->p)
                        if (p.key == "dtype") {
                            CHECK_EQ(p.sval, std::string("mxfp4a8"));
                            found = true;
                        }
                    CHECK(found);
                }
            }
}

/* A quantised container may keep one linear as the checkpoint holds it. Flash-Next's recipe leaves
 * the QSA indexer's projection plain bf16 (its top-k is a hard choice between blocks), so in a
 * four-bit model with fp8 linears it is declared bf16 and read by `gemm_nt` over its input's bf16
 * plane, while the attention projections beside it stay block fp8. */
TEST(a_quantised_flash_next_reads_its_plain_indexer_as_bf16) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 }, { "qsa_qk", plain } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int qk = 0, qk_bf16 = 0, attn_k_fp8 = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.find(".qsa_qk.weight") != std::string::npos) {
            ++qk;
            qk_bf16 += w.d.dtype == RAD_BF16 && !w.d.planes;
        }
        if (w.name.find(".attn_k.weight") != std::string::npos) attn_k_fp8 += w.d.dtype == RAD_F8E4M3;
    }
    CHECK(qk > 0);
    CHECK_EQ(qk_bf16, qk);
    CHECK(attn_k_fp8 > 0);
    /* every op that reads an indexer weight is the plain GEMM */
    int qk_ops = 0, qk_plain_ops = 0;
    for (const RecOp& o : b.ops)
        for (rad_weight wh : o.w)
            if (wh && b.weights[wh - 1].name.find(".qsa_qk.weight") != std::string::npos) {
                ++qk_ops;
                qk_plain_ops += o.op == "gemm_nt";
            }
    CHECK(qk_ops > 0);
    CHECK_EQ(qk_plain_ops, qk_ops);
}

/* A quantised layer may keep a few experts plain bf16. Layer 1 of a four-bit Flash-Next keeps
 * experts 3 and 10: that layer's sort places them last, its four-bit GEMMs serve the others in
 * slots 0..n-3 with no gap (a layer's experts are consecutive fixed-size units and routing counts
 * are in sort order), and the two are static bf16 weights behind a bf16 grouped GEMM pair, one of
 * which writes the same routed output as the four-bit down GEMM. Other layers are ordinary. */
TEST(a_four_bit_layer_serves_its_plain_experts_beside_the_rest) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
               { "blk.1.ffn_gate_up_exps.10.weight", plain }, { "blk.1.ffn_down_exps.10.weight", plain },
               { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    const int64_t ne = m.moecfg.n_expert;

    auto param = [](const RecOp& o, const char* k) -> const RecParam* {
        for (const RecParam& p : o.p) if (p.key == k) return &p;
        return nullptr;
    };
    auto layer_of = [&](const RecOp& o) -> int {
        for (rad_weight wh : o.w)
            if (wh) {
                const std::string& n = b.weights[wh - 1].name;
                if (n.rfind("blk.", 0) == 0) return std::atoi(n.c_str() + 4);
            }
        return -1;
    };
    /* the sorts: only layer 1's names the two experts */
    int prot_sorts = 0, plain_sorts = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "moe_scatter" || o.op == "router_topk_scatter") {
            const RecParam* p = param(o, "protect");
            if (p) { ++prot_sorts; CHECK_EQ(p->sval, std::string("3,10")); }
            else ++plain_sorts;
        }
    CHECK(prot_sorts >= 1);
    CHECK(plain_sorts > 0);
    /* the grouped GEMMs of layer 1 */
    int q_ops = 0, bf_ops = 0;
    for (const RecOp& o : b.ops) {
        if (layer_of(o) != 1) continue;
        if (o.op == "moe_gemm_q") {
            ++q_ops;
            CHECK_EQ(param(o, "n_expert")->ival, ne - 2);
            for (rad_weight wh : o.w) {
                const std::string& n = b.weights[wh - 1].name;
                CHECK(n.find("exps.3.") == std::string::npos && n.find("exps.10.") == std::string::npos);
            }
        }
        if (o.op == "moe_gemm") {
            ++bf_ops;
            CHECK_EQ(param(o, "n_expert")->ival, 2);
            CHECK_EQ(param(o, "dtype")->sval, std::string("bf16"));
            REQUIRE_EQ(o.w.size(), (size_t)2);
            const std::string& n0 = b.weights[o.w[0] - 1].name;
            const std::string& n1 = b.weights[o.w[1] - 1].name;
            CHECK(n0.find("exps.3.weight") != std::string::npos);
            CHECK(n1.find("exps.10.weight") != std::string::npos);
        }
    }
    CHECK_EQ(q_ops, 2);
    CHECK_EQ(bf_ops, 2);
    /* layer 1's four-bit experts fill slots 0..ne-3 with none missing; the plain ones are static
     * bf16 weights */
    std::vector<int> slot_seen((size_t)ne, 0);
    int plain_w = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.name.find("_exps.") == std::string::npos) continue;
        const bool prot_name = w.name.find("exps.3.") != std::string::npos ||
                               w.name.find("exps.10.") != std::string::npos;
        if (prot_name) {
            ++plain_w;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(w.d.group.expert, -1);
            CHECK_EQ(w.d.access, RAD_ACCESS_PER_TOKEN);
        } else if (w.d.group.expert >= 0 && w.d.group.expert < ne) {
            slot_seen[(size_t)w.d.group.expert] = 1;
        }
    }
    CHECK_EQ(plain_w, 4);
    int filled = 0;
    for (int64_t s = 0; s < ne; ++s) filled += slot_seen[(size_t)s];
    CHECK_EQ(filled, (int)(ne - 2));
    CHECK_EQ(slot_seen[(size_t)(ne - 1)] + slot_seen[(size_t)(ne - 2)], 0);

    /* An odd plain count leaves the rest unsplittable into two equal parity tables: refused. */
    RadBuilder odd;
    odd.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
                 { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    CHECK(qwen4exp_fp8::declare(&odd, &meta, &c) != RAD_OK);

    /* With two ranks each holding a slice of every expert, a plain expert is sliced evenly too
     * (gate_up 2 x 320 rows, down 320 columns) and its GEMMs are that wide. */
    {
        RadBuildCtx c2 = build_ctx(0, 2);
        c2.max_tok = 2048; c2.max_seqs = 8; c2.max_spec = 3; c2.max_ctx = 200000;
        RadBuilder sb;
        sb.encs = b.encs;
        REQUIRE_EQ(qwen4exp_fp8::declare(&sb, &meta, &c2), RAD_OK);
        int sliced = 0;
        for (const RecWeight& w : sb.weights) {
            if (w.name != "blk.1.ffn_gate_up_exps.3.weight" && w.name != "blk.1.ffn_down_exps.3.weight")
                continue;
            ++sliced;
            if (w.name.find("gate_up") != std::string::npos) {
                CHECK_EQ(w.d.shape[0], 640);
                CHECK_EQ(w.d.shard, RAD_SHARD_ROW);
                CHECK_EQ(w.d.n_row_parts, 2);
            } else {
                CHECK_EQ(w.d.shape[1], 320);
                CHECK_EQ(w.d.shard, RAD_SHARD_COL);
            }
        }
        CHECK_EQ(sliced, 2);
        int widths_ok = 0;
        for (const RecOp& o : sb.ops)
            if (o.op == "moe_gemm" && param(o, "n_expert")->ival == 2)
                widths_ok += (param(o, "N")->ival == 640 && param(o, "K")->ival == 2560) ||
                             (param(o, "N")->ival == 2560 && param(o, "K")->ival == 320);
        CHECK_EQ(widths_ok, 2);
    }

    /* The expert the format is probed from may be a plain one; the probe reads on to the first
     * four-bit expert, so the model is four-bit, not bf16. */
    RadBuilder first;
    first.encs = { { "blk.0.ffn_gate_up_exps.0.weight", plain }, { "blk.0.ffn_down_exps.0.weight", plain },
                   { "blk.0.ffn_gate_up_exps.1.weight", plain }, { "blk.0.ffn_down_exps.1.weight", plain },
                   { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&first, &meta, &c), RAD_OK);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_w4, 1);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_bf16, 0);
}

TEST(a_sizing_declare_leaves_the_step_state_alone_and_shrinks_only_rows) {
    (void)&qwen4exp_fp8::step;     /* included for its declare */
    RadModelMeta meta = qwen4exp_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen4exp_fp8::probe(&meta, &pb));
    CHECK(pb.shape_probe_ok);

    /* A 4096 budget under a span (RadBuildCtx::max_mixed_tok 2048: a step wider than 2048 is one
     * sequence) reads in pass-wide pieces and scores a pass-tall plane in windows, and at four
     * ranks pipelines its prefill (step_pipelined), declaring an all-reduce at `overlap` 1 a
     * block; the sizing declare, at a max_tok no wider than a pass, has to declare the same ones.
     * At two ranks the gated residual's fused write takes the all-reduce, so nothing pipelines.
     * Without the span a 4096 budget declares none of it. */
    struct Shape { int64_t tok, mixed; };
    for (const Shape sh : { Shape{ 2048, 0 }, Shape{ 4096, 0 }, Shape{ 4096, 2048 } })
    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            const int64_t real_tok = sh.tok;
            const bool mixed = sh.mixed > 0;
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = real_tok;
            c.max_mixed_tok = sh.mixed;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            const int64_t max_tok = m.g.max_tok;
            const int64_t max_work = m.qsacfg.max_work;
            /* The score plane: a pass tall where a wider step is one sequence, max_tok rows
             * otherwise. */
            if (const RecBuf* sc = find_b(full, "qsa_score")) {
                const int64_t want = mixed ? 2048 : c.max_tok;
                CHECK_EQ(sc->d.shape[0], want);
                CHECK_EQ(m.qsacfg.plane_rows, want < c.max_tok ? want : 0);
            }
            /* And under a span every connection read is declared no wider than a pass: a wider
             * step reads in pieces, and the read's scratch follows its band. */
            {
                const int64_t want = (mixed && c.max_tok > 2048) ? 2048 : c.max_tok;
                int reads = 0, wide = 0;
                for (const auto& o : full.ops) {
                    if (o.op != "hc_read") continue;
                    ++reads;
                    const RecParam* pm = param(&o, "M");
                    if (!pm || pm->ihi != want) ++wide;
                }
                CHECK(reads > 0);
                CHECK_EQ(wide, 0);
            }
            /* The pipelined step's all-reduces at `overlap` 1, a block each, only where it runs,
             * with the buffer its second lane touches named concurrent. */
            {
                int lane = 0;
                for (const auto& o : full.ops)
                    if (o.op == "all_reduce" && pint(&o, "overlap") == 1) ++lane;
                const bool pipe = mixed && world > 2;
                CHECK_EQ(lane > 0, pipe);
                const RecBuf* x = find_b(full, "x");
                REQUIRE(x != nullptr);
                const rad_buf xh = (rad_buf)(x - full.bufs.data()) + 1;
                CHECK_EQ(std::count(full.concurrent.begin(), full.concurrent.end(), xh) > 0, pipe);
                CHECK_EQ(find_b(full, "pipe_cu") != nullptr, mixed);
            }
            const rad_op last_op = (rad_op)full.ops.size();
            CHECK_EQ(max_tok, real_tok);

            RadBuildCtx p = c;
            p.max_tok = 256;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);

            /* The step's state is still the real declare's. */
            CHECK_EQ(m.g.max_tok, max_tok);
            CHECK_EQ(m.qsacfg.max_work, max_work);
            CHECK_EQ((rad_op)full.ops.size(), last_op);

            /* The same graph, and the same buffers with only their rows smaller. */
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                CHECK_EQ(small.ops[i].op, full.ops[i].op);
                /* Only the ranged parameter may differ; a fixed one following max_tok would run
                 * the kernel at the real size against a level's buffers. */
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    /* Capacities in libr4d's schema (RAD_PROLE_CAPACITY) may shrink: qsa_work
                     * launches as wide as its `page` operand regardless of `work`, and
                     * attn_paged_gate_quant by its block table's rows regardless of `max_seqs`. */
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld at 2048, %lld at 256\n",
                                i, full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            int shrank = 0;
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(small.bufs[i].name, full.bufs[i].name);
                CHECK_EQ(q.rank, f.rank);
                CHECK_EQ(q.kind, f.kind);
                for (uint32_t d = 1; d < f.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld at 2048, %lld at 256\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d]);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
                shrank += q.shape[0] < f.shape[0];
            }
            CHECK(shrank > 10);
        }
    }
}

/* Four ranks on Flash-Next: KV heads replicate, expert widths split, most fused collectives drop.
 *
 *   - KV heads cannot be cut four ways, so they are replicated in pairs: one KV head per rank,
 *     ranks {0,1} reading rows [0,256) of the checkpoint's k/v projection and {2,3} [256,512).
 *     This is a span, not a shard index, and the only place a rank names the container rows it
 *     wants, so it is asserted as agreeing spans whose union is the whole projection.
 *   - An expert is 640 columns, five 128-column scale blocks, which four ranks cannot cut evenly.
 *     So the width divides, unevenly by class, not the expert set: every rank routes all 512
 *     experts and computes [lo, hi) of each, the odd block rotating with expert number mod 4, five
 *     blocks per four experts on every rank. Sharding the experts would not work: protection is
 *     per-rank, so a rank could own one plain bf16 expert and an odd count of quantised ones, while
 *     the grouped GEMM takes two equal parity tables (the recipe protects experts in pairs).
 *   - Fused all-reduces are world-2 kernels (`ar_wire_is_exact` and `ar_fused_ok` return false
 *     otherwise), except the gated residual's write: `ar_hc_write` and `ar_gather_hc_write` also
 *     exist at four, on the exact wire only (`wire` 0) and over a decode or verify step's rows,
 *     max_seqs * (1 + max_spec). Every block still declares its standalone `all_reduce`, and every
 *     MoE block its `moe_gather`, for the steps the fused write declines. The world-2 contrast
 *     below catches a fused declare gated on the wrong condition.
 *
 * Expected numbers derive from this file's meta (n_head 24, n_head_kv 2, head_dim 256,
 * num_experts 512); the production container's unsharded k_proj height (512 rows) is asserted as a
 * relation, not a literal. */
TEST(flashnext_tp4_replicates_kv_heads_and_splits_expert_widths) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world    = 4;
    const int64_t  n_head   = meta.n_head;        /* 24 query heads */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2 KV heads, fewer than the ranks */
    const int64_t  head_dim = meta.head_dim;      /* 256 */
    const int64_t  ne_all   = atoll(rad_meta_gets(&meta, "num_experts", "0"));
    REQUIRE(n_head % world == 0);
    REQUIRE_EQ(world % n_head_kv, (int64_t)0);   /* replication needs the multiple */
    REQUIRE(ne_all % world == 0);

    /* Full-attention layers of the eight-layer fixture: 3 and 7 (interval 4). */
    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        /* geom_from must accept a world larger than the KV head count. */
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    for (int r = 0; r < world; ++r) {
        /* (1) A quarter of the query heads and one KV head, so gqa is 6, and the page is the QSA
         * compress block of 4 (qwen4exp_fp8.cpp's lay.attn.kv_block must agree with the row's
         * C_EQ("block_size", 4) or nothing resolves). */
        const RecOp* at = find_op(b[r], "attn_paged");
        REQUIRE(at != nullptr);
        CHECK_EQ(pint(at, "n_head"), n_head / world);
        CHECK_EQ(pint(at, "gqa"), n_head / world);   /* one KV head, so gqa == the rank's heads */
        CHECK_EQ(pint(at, "block_size"), 4);
        /* The full-attention KV group (group 0, kv_attn, declared first) holds this rank's one
         * head. */
        REQUIRE(b[r].kv_groups.size() > 0);
        CHECK_EQ(b[r].kv_groups[0].name, std::string("kv_attn"));
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        CHECK_EQ(b[r].kv_groups[0].d.head_dim, head_dim);

        /* (2) One KV head's rows (256 per rank) and the span locating them in the entry. Each pair
         * of ranks declares the same span, and the two spans cover the checkpoint's whole k_proj
         * (asserted as a relation below). */
        for (const std::string& wn : {kq, kv}) {
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);          /* one head */
            CHECK_EQ(w->d.shape[1], meta.n_embd);       /* K is not sharded */
            CHECK_EQ(w->d.shard, RAD_SHARD_ROW);
            CHECK(w->span_set);
            CHECK_EQ(w->span_hi - w->span_lo, w->d.shape[0]);
            CHECK_EQ(w->span_lo, (int64_t)(r / (world / n_head_kv)) * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        /* (3) Every rank routes every expert and computes a slice of each. The router and expert
         * set are replicated (n_expert 512, base 0); ranks divide each expert's width in whole
         * 128-column scale blocks. */
        const RecOp* sc = find_op(b[r], "moe_scatter");
        REQUIRE(sc != nullptr);
        CHECK_EQ(pint(sc, "n_expert"), ne_all);
        CHECK_EQ(pint(sc, "expert_base"), (int64_t)0);
        const RecOp* tk = find_op(b[r], "router_topk");
        REQUIRE(tk != nullptr);
        CHECK_EQ(pint(tk, "n_expert"), ne_all);         /* replicated routing */
        if (const RecOp* ts = find_op(b[r], "router_topk_scatter")) {
            CHECK_EQ(pint(ts, "n_expert"), ne_all);
            CHECK_EQ(pint(ts, "expert_base"), (int64_t)0);
        }
        /* Each rank's width per class, as a span on every expert's down projection (no expert is
         * protected here, so class = expert number mod 4). These literals are the TP4 MoE contract:
         * five 128-blocks over four ranks, the odd block rotating with the class. They are written
         * out rather than recomputed from the plugin's formula on purpose. */
        const int64_t blk = 128;                    /* RAD_FP8_BLOCK, the scale block width */
        const int64_t nblk = meta.n_ff / blk;        /* five at the fixture's 640 */
        const int64_t cls_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                       { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t cls_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                       { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        REQUIRE_EQ(nblk, (int64_t)5);
        int spanned[4] = { 0, 0, 0, 0 };
        for (const RecWeight& w : b[r].weights) {
            if (w.name.rfind("blk.0.ffn_down_exps.", 0) != 0) continue;
            if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
            const int q = atoi(w.name.c_str() + 20) % 4;      /* the expert's class */
            CHECK(w.span_set);
            CHECK_EQ(w.span_lo, cls_lo[r][q]);
            CHECK_EQ(w.span_hi, cls_hi[r][q]);
            ++spanned[q];
        }
        for (int q = 0; q < 4; ++q)
            CHECK_EQ(spanned[q], ne_all / 4);            /* every expert of each class declared */
        /* The four ranks' slices of one class tile its width exactly: no block unserved and none
         * claimed twice (the all-reduce would double it). */
        for (int q = 0; q < 4; ++q) {
            int64_t covered = 0;
            for (int r2 = 0; r2 < world; ++r2)
                for (int64_t i = 0; i < nblk; ++i) {
                    const int64_t lo = cls_lo[r2][q], hi = cls_hi[r2][q];
                    if (lo <= i * blk && i * blk < hi) ++covered;
                }
            CHECK_EQ(covered, nblk);   /* per class, exactly once each */
        }
        /* Every rank takes the same width, five blocks per four experts; that is why there are
         * four classes rather than two. */
        {
            int64_t sum = 0;
            for (int q = 0; q < 4; ++q) sum += cls_hi[r][q] - cls_lo[r][q];
            CHECK_EQ(sum, nblk * blk);
        }
        /* The gate/up GEMM's four classes follow the same split, doubled since the operand stacks
         * gate and up; buffers take the widest class. No parity op remains. */
        const RecOp* gu = find_op(b[r], "moe_gemm_q_mod4");
        REQUIRE(gu != nullptr);
        CHECK_EQ(pint(gu, "N"), 2 * (cls_hi[r][0] - cls_lo[r][0]));
        CHECK_EQ(pint(gu, "N_1"), 2 * (cls_hi[r][1] - cls_lo[r][1]));
        CHECK_EQ(pint(gu, "N_2"), 2 * (cls_hi[r][2] - cls_lo[r][2]));
        CHECK_EQ(pint(gu, "N_3"), 2 * (cls_hi[r][3] - cls_lo[r][3]));
        CHECK_EQ(pint(gu, "n_expert"), ne_all);
        CHECK_EQ(gu->w.size(), (size_t)(2 * ne_all));   /* codes and scales, each expert once */
        CHECK_EQ(count_op(b[r], "moe_gemm_q"), 0);

        /* (4) A standalone collective per reduction, plus the exact four-rank ar_hc_write and
         * ar_gather_hc_write (the gather form on each MoE connection), but no fused norm. Both
         * fused forms state the exact wire and range over a decode or verify step's rows,
         * 8 * (1 + 3), the architecture's own bound. */
        CHECK(count_op(b[r], "all_reduce") > 0);
        CHECK_EQ(pint(find_op(b[r], "all_reduce"), "world_size"), world);
        CHECK(count_op(b[r], "ar_hc_write") > 0);
        CHECK(count_op(b[r], "moe_gather") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") <= count_op(b[r], "moe_gather"));
        for (const RecOp& o : b[r].ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "world_size"), world);
            CHECK_EQ(pint(&o, "wire"), 0);                 /* exact only, stated */
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->kind, (int)RAD_P_RANGE);
            CHECK_EQ(m->ihi, c[r].max_seqs * (1 + c[r].max_spec));
        }
        CHECK_EQ(count_op(b[r], "ar_rmsnorm_quant_fp8"), 0);
    }

    /* The pairs agree and their union is the whole projection, n_head_kv * head_dim rows (a
     * container fact, so asserted as a relation). */
    for (const std::string& wn : {kq, kv}) {
        CHECK_EQ(find_w(b[0], wn)->span_hi, find_w(b[1], wn)->span_hi);
        CHECK_EQ(find_w(b[2], wn)->span_lo, find_w(b[3], wn)->span_lo);
        CHECK_EQ(find_w(b[0], wn)->span_lo, find_w(b[2], wn)->span_hi - n_head_kv * head_dim);
        const int64_t lo = find_w(b[0], wn)->span_lo;
        const int64_t hi = find_w(b[2], wn)->span_hi;
        CHECK_EQ(hi - lo, n_head_kv * head_dim);
        CHECK_EQ(hi - lo, world * find_w(b[0], wn)->d.shape[0] / (world / n_head_kv));
    }

    /* (5) All four ranks declare the same ops; shards differ only in parameters and spans, since
     * the plan must not vary by rank. */
    for (int r = 1; r < world; ++r) {
        if (op_names(b[r]) != op_names(b[0])) {
            fprintf(stderr, "    rank %d differs from rank 0 in its op list (%zu vs %zu)\n", r,
                    b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size() && i < b[r].ops.size(); ++i)
                if (b[0].ops[i].op != b[r].ops[i].op) {
                    fprintf(stderr, "    first at %zu: '%s' vs '%s'\n", i,
                            b[0].ops[i].op.c_str(), b[r].ops[i].op.c_str());
                    break;
                }
        }
        CHECK_EQ(join(op_names(b[r])), join(op_names(b[0])));
    }

    /* The shared expert is split by whole blocks, unevenly at four: 640 is five 128-blocks, and
     * rank 2 takes the extra one, after the plain experts' extra tiles on ranks 0 and 1
     * (qwen4exp_fp8.cpp's shared_place). Ranks 2, 3, 0, 1 cut [0,256) [256,384) [384,512)
     * [512,640). At two ranks, rank 0 takes three of five. */
    const int64_t sh_lo[] = { 384, 512, 0,   256 };
    const int64_t sh_hi[] = { 512, 640, 256, 384 };
    for (int r = 0; r < world; ++r) {
        const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
        REQUIRE(sg != nullptr);
        CHECK(sg->span_set);
        CHECK_EQ(sg->span_lo, sh_lo[r]);
        CHECK_EQ(sg->span_hi, sh_hi[r]);
        CHECK_EQ(sg->d.shape[0], 2 * (sh_hi[r] - sh_lo[r]));   /* gate's and up's rows */
    }
    /* Every block of the shared expert has exactly one owner. */
    for (int64_t i = 0; i < 640; i += 128) {
        int owners = 0;
        for (int r = 0; r < world; ++r) {
            const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
            owners += sg->span_lo <= i && i < sg->span_hi;
        }
        CHECK_EQ(owners, 1);
    }

    /* At two ranks the KV heads split evenly, so the even shard selects the rows and no span is
     * declared. The fused forms resolve at world 2 over every step. */
    {
        RadBuilder h[2];
        for (int r = 0; r < 2; ++r) {
            RadBuildCtx c2 = build_ctx(r, 2);
            c2.max_tok  = 2048;
            c2.max_seqs = 8;
            c2.max_spec = 3;
            c2.max_ctx  = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&h[r], &meta, &c2), RAD_OK);
            const RecWeight* w = find_w(h[r], kq);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], n_head_kv / 2 * head_dim);   /* one of two heads */
            CHECK(!w->span_set);
            CHECK_EQ(h[r].kv_groups[0].d.n_head_kv, n_head_kv / 2);
            CHECK_EQ(pint(find_op(h[r], "attn_paged"), "gqa"), (n_head / 2) / (n_head_kv / 2));
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "n_expert"), ne_all);
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "expert_base"), 0);
            CHECK(count_op(h[r], "ar_hc_write") > 0);
            CHECK(count_op(h[r], "ar_gather_hc_write") > 0);
            /* Two ranks keep the exact write over every step, prefill chunks included. */
            for (const RecOp& o : h[r].ops) {
                if (o.op != "ar_hc_write") continue;
                CHECK_EQ(pint(&o, "wire"), 0);
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, c2.max_tok);
            }
            /* No ar_rmsnorm_quant_fp8 at either world: qwen4exp hands every block's norm
             * `kArNone`, since the reduction rides in the gated residual's write instead
             * (qwen4exp_fp8.cpp's block declares). */
            CHECK_EQ(count_op(h[r], "ar_rmsnorm_quant_fp8"), 0);
        }
    }

    /* Which steps the four-rank fused write takes: decode and verify steps, up to
     * max_seqs * (1 + max_spec) rows and never more than max_tok, whose message goes out exact (on
     * the lossy wire: under its floor or off its 128-element tile). The bound is the architecture's,
     * not a message size: 64 sequences at depth 3 are 256 rows (1.25 MiB) and still take it, while
     * a prefill chunk never does. The preceding block asks the same predicate. At two ranks it
     * equals ar_wire_is_exact, which itself stays false at four. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)32);
        CHECK_EQ(rad::arch::ar_exact_write_rows(g4), (int64_t)32);
        CHECK(!rad::arch::ar_exact_write_takes(g4, 0));
        CHECK(rad::arch::ar_exact_write_takes(g4, 1));
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));      /* a decode step */
        CHECK(rad::arch::ar_exact_write_takes(g4, 32));     /* every sequence verifying 1 + 3 */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 33));    /* more: a prefill chunk */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 2048));
        CHECK(!rad::arch::ar_wire_is_exact(g4, 8));
        g4.wire_exact = 0;
        CHECK(rad::arch::ar_exact_write_takes(g4, 25));     /* 125 KiB, under the floor */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 26));    /* 130 KiB, goes on the ti8 wire */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 32));
        g4.wire_exact = 1;
        g4.max_spec = 0;                                    /* one row per sequence */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)8);
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 9));
        g4.max_seqs = 64; g4.max_spec = 3;
        CHECK(rad::arch::ar_exact_write_takes(g4, 256));    /* 1.25 MiB */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 257));
        g4.max_tok = 128;                                   /* caps a step's rows */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)128);
        CHECK(rad::arch::ar_exact_write_takes(g4, 128));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 129));
        rad::arch::Geom g8 = g4;
        g8.world = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g8), (int64_t)0);
        CHECK(!rad::arch::ar_exact_write_takes(g8, 8));
        rad::arch::Geom g2 = g4;
        g2.world = 2; g2.max_tok = 4096; g2.max_seqs = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g2), g2.max_tok);
        for (int exact : { 0, 1 }) {
            g2.wire_exact = exact;
            for (int64_t T : { 1, 8, 25, 26, 33, 205, 4096 })
                CHECK(rad::arch::ar_exact_write_takes(g2, T) == rad::arch::ar_wire_is_exact(g2, T));
        }
    }
    /* The four-rank fused gather takes the fused write's steps, within one MoE pass. Only the
     * exact bound exists at four (the rotated form is never declared, so `rows6` is 0), and a step
     * the write declines keeps its plain gather. At two ranks ar_wire_is_exact picks the bound. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK(rad::arch::gather_taken(g4, 8, 32, 0));        /* a decode step */
        CHECK(rad::arch::gather_taken(g4, 32, 32, 0));       /* a verify step, one pass */
        CHECK(!rad::arch::gather_taken(g4, 33, 2048, 0));    /* the write declines it */
        CHECK(!rad::arch::gather_taken(g4, 8, 0, 0));        /* not declared */
        CHECK(!rad::arch::gather_taken(g4, 20, 16, 0));      /* two passes */
        g4.wire_exact = 0;
        CHECK(rad::arch::gather_taken(g4, 25, 32, 0));       /* under the floor */
        CHECK(!rad::arch::gather_taken(g4, 26, 32, 0));      /* goes on the ti8 wire */
        rad::arch::Geom g2 = g4;
        g2.world = 2;
        for (int64_t T : { 1, 8, 25, 26, 205, 4096 })
            for (int64_t r6 : { 0, 2048 }) {
                const int64_t r = rad::arch::ar_wire_is_exact(g2, T) ? 2048 : r6;
                CHECK(rad::arch::gather_taken(g2, T, 2048, r6) == (r > 0 && T <= r));
            }
    }
    /* At four, the lossy wire declares no rotated write or gather: only the exact forms, stating
     * `wire` 0, for decode and verify steps under the floor. */
    {
        RadBuilder b4;
        RadBuildCtx c4 = build_ctx(0, world);
        c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
        c4.tp_wire_lossy = 1;
        c4.tp_wire_min_bytes = 128 * 1024;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b4, &meta, &c4), RAD_OK);
        CHECK(count_op(b4, "ar_hc_write") > 0);
        CHECK(count_op(b4, "ar_gather_hc_write") > 0);
        for (const RecOp& o : b4.ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "wire"), 0);
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->ihi, c4.max_seqs * (1 + c4.max_spec));
        }
    }

    /* A world that is not a multiple of the KV head count still declares: qwen4exp opts in via
     * geom_from's `attn_world_ok`, so attention runs on the largest multiple below the world and the
     * remaining ranks compute none (see the tp3 test). Nothing is rounded into a half head. A
     * plugin that does not opt in is refused at this world by geom_from. */
    {
        RadBuilder b3;
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK_EQ(qwen4exp_fp8::declare(&b3, &meta, &c3), RAD_OK);
    }
    {
        rad::arch::Geom g3{};
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK_EQ(rad::arch::geom_from(g3, &meta, &c3, "bf16", RAD_BF16, /*arch_splits_ff=*/true), RAD_E_INVAL);
        CHECK_EQ(rad::arch::geom_from(g3, &meta, &c3, "bf16", RAD_BF16, /*arch_splits_ff=*/true,
                           /*attn_world_ok=*/true), RAD_OK);
        CHECK(g3.attn_zero == false);
        CHECK_EQ(g3.attn_world, 2);
    }

    /* ---- A layer that keeps experts plain stays at four classes. Two plain experts leave 510
     * quantised ones, four leave 508; either way `moe_gemm_q_mod4` serves the layer, the class being
     * the expert's place in the sort mod 4 (plain ones sorted last). At 510, classes 0 and 1 hold
     * 128 experts and 2 and 3 hold 127 (the first tables take the remainder). The plain experts'
     * extra tiles go to ranks 0 and 1, so every rank holds the same routed width; the shared
     * expert's extra block is rank 2's. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* The four-class slices, [lo, hi) by rank and place mod 4. */
        const int64_t c4_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                      { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t c4_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                      { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        for (int nprot : { 2, 4 }) {
            const std::vector<int> pe = nprot == 2 ? std::vector<int>{ 3, 10 }
                                                   : std::vector<int>{ 3, 10, 21, 40 };
            for (int r = 0; r < world; ++r) {
                RadBuilder bp;
                for (int e : pe) {
                    bp.encs.push_back({ "blk.1.ffn_gate_up_exps." + std::to_string(e) + ".weight",
                                        plain });
                    bp.encs.push_back({ "blk.1.ffn_down_exps." + std::to_string(e) + ".weight",
                                        plain });
                }
                bp.encs.push_back({ "ffn_gate_up_exps", w4p });
                bp.encs.push_back({ "ffn_down_exps", w4p });
                RadBuildCtx cp = build_ctx(r, world);
                cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
                REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
                const int64_t n_reg = ne_all - nprot;
                /* layer 1's quantised GEMMs */
                int l1_ops = 0, other_ops = 0;
                for (const RecOp& o : bp.ops) {
                    if (o.op != "moe_gemm_q" && o.op != "moe_gemm_q_mod4") continue;
                    bool l1 = false;
                    for (rad_weight wh : o.w)
                        l1 = l1 || bp.weights[wh - 1].name.rfind("blk.1.", 0) == 0;
                    if (!l1) { CHECK_EQ(o.op, std::string("moe_gemm_q_mod4")); ++other_ops; continue; }
                    ++l1_ops;
                    CHECK_EQ(o.op, std::string("moe_gemm_q_mod4"));
                    CHECK_EQ(pint(&o, "n_expert"), n_reg);
                    CHECK_EQ(o.w.size(), (size_t)(2 * n_reg));
                    /* four runs, codes then scales, class by class; class q holds places q,
                     * q + 4, ... (128, 128, 127, 127 at 510) */
                    size_t at = 0;
                    for (int q = 0; q < 4; ++q) {
                        const int64_t nq = (n_reg + 3 - q) / 4;
                        for (int half = 0; half < 2; ++half)
                            for (int64_t j = 0; j < nq && at < o.w.size(); ++j, ++at) {
                                const std::string& wn = bp.weights[o.w[at] - 1].name;
                                const int e = atoi(wn.c_str() + wn.find("_exps.") + 6);
                                int place = e;
                                for (int x : pe) place -= x < e ? 1 : 0;
                                CHECK_EQ(place, (int)(q + 4 * j));
                                CHECK((wn.find(".scale") != std::string::npos) == (half == 1));
                            }
                    }
                    CHECK_EQ(at, o.w.size());
                }
                CHECK_EQ(l1_ops, 2);
                CHECK(other_ops > 0);
                /* each quantised expert of layer 1 gets its class's slice */
                int seen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    if (w.d.dtype == RAD_BF16) continue;          /* plain ones checked below */
                    const int e = atoi(w.name.c_str() + 20);
                    int place = e;
                    for (int x : pe) place -= x < e ? 1 : 0;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, c4_lo[r][place % 4]);
                    CHECK_EQ(w.span_hi, c4_hi[r][place % 4]);
                    ++seen;
                }
                CHECK_EQ(seen, (int)n_reg);
                /* the plain experts' tiles by rank: 3, 3, 2, 2 */
                const int64_t p4_lo[4] = { 0, 192, 384, 512 },   p4_hi[4] = { 192, 384, 512, 640 };
                int pseen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.d.dtype != RAD_BF16) continue;
                    if (w.name.find("_exps.") == std::string::npos) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, p4_lo[r]);
                    CHECK_EQ(w.span_hi, p4_hi[r]);
                    ++pseen;
                }
                CHECK_EQ(pseen, 2 * nprot);
            }
        }
    }

    /* ---- The plain bf16 experts at four ranks.
     *
     * A second grouped GEMM serves them over their own slice of the same width: whole 64-wide
     * output tiles, one range per rank, uniform across a layer's protected experts (their GEMM
     * takes one table, so every entry has one geometry; they do not alternate by parity). The
     * width must be the expert's own, not derived from this rank's parity slices, which only match
     * at two ranks.
     *
     * Ten tiles over four ranks is 3, 3, 2, 2: uneven, whole tiles, each owned once. Ranks 0 and 1
     * compute 192 columns of every protected expert, ranks 2 and 3 compute 128; every rank holds
     * the same routed width, and rank 2 the shared expert's extra block (shared_place). Two
     * protected experts, as the production recipe pairs them. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* [lo, hi) of the plain experts' columns per rank, written out: 3, 3, 2, 2 tiles. */
        const int64_t p_lo[4] = { 0,   192, 384, 512 };
        const int64_t p_hi[4] = { 192, 384, 512, 640 };
        for (int r = 0; r < world; ++r) {
            RadBuilder bp;
            bp.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain },
                        { "blk.1.ffn_down_exps.3.weight", plain },
                        { "blk.1.ffn_gate_up_exps.10.weight", plain },
                        { "blk.1.ffn_down_exps.10.weight", plain },
                        { "ffn_gate_up_exps", w4p }, { "ffn_down_exps", w4p } };
            RadBuildCtx cp = build_ctx(r, world);
            cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
            const int64_t wp = p_hi[r] - p_lo[r];
            int seen = 0;
            for (const RecWeight& w : bp.weights) {
                if (w.name.rfind("blk.1.ffn_gate_up_exps.", 0) != 0 &&
                    w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                if (w.d.dtype != RAD_BF16) continue;      /* plain ones only */
                if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                CHECK(w.span_set);
                CHECK_EQ(w.span_lo, p_lo[r]);
                CHECK_EQ(w.span_hi, p_hi[r]);
                ++seen;
            }
            /* Both projections of both protected experts. */
            CHECK_EQ(seen, 4);
            /* The gate/up GEMM's N is twice this rank's share (stacked pair) and the down GEMM's
             * K is the share. */
            int gu_ok = 0, dn_ok = 0;
            for (const RecOp& o : bp.ops) {
                if (o.op != "moe_gemm") continue;
                if (pint(&o, "N") == 2 * wp && pint(&o, "n_expert") == 2) ++gu_ok;
                if (pint(&o, "K") == wp && pint(&o, "n_expert") == 2) ++dn_ok;
            }
            CHECK_EQ(gu_ok, 1);
            CHECK_EQ(dn_ok, 1);
        }
    }
}

/* ==================================================================== media */
namespace {

/* The fixture above plus what a Qwen3.8 container adds for pictures: the text config's
 * interleaved M-RoPE and a vision tower cut to two blocks at the real widths, enough to see a
 * block repeat. */
const char* k_vl_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
    "rope_parameters.mrope_section",
    "rope_parameters.mrope_interleaved",
    "vision_config.depth",
    "vision_config.hidden_size",
    "vision_config.num_heads",
    "vision_config.intermediate_size",
    "vision_config.patch_size",
    "vision_config.temporal_patch_size",
    "vision_config.in_channels",
    "vision_config.spatial_merge_size",
    "vision_config.out_hidden_size",
    "vision_config.num_position_embeddings",
    "vision_config.deepstack_visual_indexes",
    "vision_config.hidden_act",
};
const char* k_vl_vals[] = { "4", "64", "128", "16", "48", "6144", "4",
                            "11 11 10", "1",
                            "2", "1152", "16", "4304", "16", "2", "3", "2", "5120", "2304", "",
                            "gelu_pytorch_tanh" };

RadModelMeta qwen35_vl_meta() {
    RadModelMeta m = qwen35_meta();
    m.n_kv   = (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0]));
    m.kv_key = k_vl_keys;
    m.kv_val = k_vl_vals;
    return m;
}

RadBuildCtx vl_ctx(int64_t patches) {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = patches;
    return c;
}

const RecOp* issued_op(const RadBuilder& b, const RadCtx& c, const std::string& name, int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return o;
    }
    return nullptr;
}

}  /* namespace */

/* Three ranks over two KV heads: the attention world is 2 (rad_arch.h), so rank 2 computes no
 * attention (a `fill` of its output wire and a zero KV group). The delta net's 16 K heads go 5, 5,
 * 6 with the V heads at the same multiple, the extra head on the rank without attention, declared
 * as container spans with no padding. */
TEST(flashnext_tp3_zeroes_the_third_ranks_attention_and_splits_gdn_heads_unevenly) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world     = 3;
    const int64_t  n_head    = meta.n_head;       /* 24 */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2: does not divide 3 */
    const int64_t  head_dim  = meta.head_dim;     /* 256 */
    const int64_t  nk = atoll(rad_meta_gets(&meta, "linear_num_key_heads", "0"));     /* 16 */
    const int64_t  nv = atoll(rad_meta_gets(&meta, "linear_num_value_heads", "0"));   /* 48 */
    const int64_t  hk = atoll(rad_meta_gets(&meta, "linear_key_head_dim", "0"));      /* 128 */
    const int64_t  hv = atoll(rad_meta_gets(&meta, "linear_value_head_dim", "0"));    /* 128 */
    REQUIRE(n_head_kv < world && world % n_head_kv != 0);
    REQUIRE(nk % world != 0);
    const int attn_world = (int)(n_head_kv * (world / n_head_kv));   /* 2 */

    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";
    const std::string qg = "blk." + std::to_string(lfull) + ".attn_qg.weight";
    const std::string oo = "blk." + std::to_string(lfull) + ".attn_output.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    /* Every rank declares the same KV groups in the same order, since the scheduler indexes groups
     * by position and one block table serves all ranks. Ops may differ per rank (issue walks each
     * rank's own program), but the attention ranks declare the same ops. */
    for (int r = 1; r < world; ++r) {
        REQUIRE_EQ(b[r].kv_groups.size(), b[0].kv_groups.size());
        for (size_t g = 0; g < b[0].kv_groups.size(); ++g) {
            CHECK_EQ(b[r].kv_groups[g].name, b[0].kv_groups[g].name);
            CHECK_EQ(b[r].kv_groups[g].d.kind, b[0].kv_groups[g].d.kind);
        }
        if (r < attn_world) {
            REQUIRE_EQ(b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size(); ++i) CHECK_EQ(b[r].ops[i].op, b[0].ops[i].op);
        }
    }

    std::vector<std::pair<int64_t, int64_t>> k_span((size_t)world);   /* each rank's K heads [lo, hi) */
    for (int r = 0; r < world; ++r) {
        const bool zero = r >= attn_world;
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        /* The attention-zero rank has no attention op or projection weight, fills its output
         * wire, and carves no cache for the attention group. */
        if (zero) {
            CHECK(find_op(b[r], "attn_paged") == nullptr);
            CHECK(find_op(b[r], "kv_store") == nullptr);
            CHECK(find_op(b[r], "fill") != nullptr);
            CHECK(find_w(b[r], kq) == nullptr);
            CHECK(find_w(b[r], qg) == nullptr);
            CHECK(find_w(b[r], oo) == nullptr);
            CHECK(b[r].kv_groups[0].zero);
            /* It still joins the all-reduce and runs the indexer. */
            CHECK(find_op(b[r], "qsa_select") != nullptr);
        } else {
            CHECK(!b[r].kv_groups[0].zero);
            CHECK(find_op(b[r], "fill") == nullptr);
        }
        /* (1) Each attention rank has the two-rank geometry: 12 heads, one KV head, gqa 12. */
        const RecOp* at = find_op(b[r], "attn_paged");
        if (!zero) {
            REQUIRE(at != nullptr);
            CHECK_EQ(pint(at, "n_head"), n_head / attn_world);
            CHECK_EQ(pint(at, "gqa"), n_head / attn_world);
        }
        /* (2) The projections: the span of this rank's heads in the container. */
        for (const std::string& wn : {kq, kv}) {
            if (zero) break;
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);
            CHECK(w->span_set);
            const int ar = zero ? attn_world - 1 : r;
            CHECK_EQ(w->span_lo, (int64_t)ar * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        if (!zero) {
            const RecWeight* w = find_w(b[r], qg);
            REQUIRE(w != nullptr);
            const int ar = zero ? attn_world - 1 : r;
            const int64_t qg_rows = 2 * (n_head / attn_world) * head_dim;
            CHECK_EQ(w->d.shape[0], qg_rows);
            CHECK(w->span_set);
            CHECK_EQ(w->span_lo, (int64_t)ar * qg_rows);
            const RecWeight* o = find_w(b[r], oo);
            REQUIRE(o != nullptr);
            CHECK_EQ(o->d.shard, RAD_SHARD_COL);
            CHECK(o->span_set);
            CHECK_EQ(o->span_lo, (int64_t)ar * (n_head / attn_world) * head_dim);
        }
        /* (3) The delta net: 5, 5, 6 K heads and 15, 15, 18 V heads on every rank, as per-part
         * spans of [q|k|v|z] and of the conv channels, and a single span of A_log. */
        const int64_t base = nk / world, rem = nk % world;
        /* The extra head goes to the rank without attention first (rad_arch.h remainder_place):
         * rank 2 takes 6, ranks 0 and 1 take 5. */
        const int64_t pl   = r >= attn_world ? r - attn_world : (world - attn_world) + r % attn_world;
        const int64_t my_k = base + (pl < rem ? 1 : 0);
        const int64_t k0   = pl * base + (pl < rem ? pl : rem);
        const int64_t my_v = (nv / nk) * my_k, v0 = (nv / nk) * k0;
        const RecWeight* in = find_w(b[r], "blk.0.ssm_inz.weight");
        REQUIRE(in != nullptr);
        REQUIRE_EQ(in->d.n_row_parts, 4);
        CHECK_EQ(in->d.row_parts[0], my_k * hk);
        CHECK_EQ(in->d.row_parts[2], my_v * hv);
        REQUIRE_EQ(in->n_part_spans, 4);
        CHECK_EQ(in->part_lo[0], k0 * hk);
        CHECK_EQ(in->part_lo[1], k0 * hk);
        CHECK_EQ(in->part_lo[2], v0 * hv);
        CHECK_EQ(in->part_lo[3], v0 * hv);
        CHECK_EQ(in->part_total[0], nk * hk);
        CHECK_EQ(in->part_total[2], nv * hv);
        k_span[(size_t)r] = { in->part_lo[0] / hk, in->part_hi[0] / hk };
        const RecWeight* cv = find_w(b[r], "blk.0.ssm_conv1d.weight");
        REQUIRE(cv != nullptr);
        REQUIRE_EQ(cv->n_part_spans, 3);
        CHECK_EQ(cv->part_lo[2], v0 * hv);
        CHECK_EQ(cv->part_hi[2], (v0 + my_v) * hv);
        const RecWeight* al = find_w(b[r], "blk.0.ssm_a_log");
        REQUIRE(al != nullptr);
        CHECK_EQ(al->d.shape[0], my_v);
        CHECK(al->span_set);
        CHECK_EQ(al->span_lo, v0);
        CHECK_EQ(al->span_hi, v0 + my_v);
        const RecWeight* ow = find_w(b[r], "blk.0.ssm_out.weight");
        REQUIRE(ow != nullptr);
        CHECK_EQ(ow->d.shard, RAD_SHARD_COL);
        CHECK(ow->span_set);
        CHECK_EQ(ow->span_lo, v0 * hv);
        CHECK_EQ(ow->span_hi, (v0 + my_v) * hv);
        /* The GDN state group holds this rank's V heads. */
        bool seen = false;
        for (const auto& g : b[r].kv_groups)
            if (g.name == "kv_gdn_state") { seen = true; CHECK_EQ(g.d.n_head_kv, (int)my_v); }
        CHECK(seen);
    }
    /* The three ranks' K head spans cover the container's 16 heads exactly once. */
    {
        std::vector<int> owner((size_t)nk, -1);
        for (int r = 0; r < world; ++r)
            for (int64_t h = k_span[(size_t)r].first; h < k_span[(size_t)r].second; ++h) {
                REQUIRE(h >= 0 && h < nk);
                CHECK_EQ(owner[(size_t)h], -1);
                owner[(size_t)h] = r;
            }
        for (int64_t h = 0; h < nk; ++h) CHECK(owner[(size_t)h] >= 0);
    }
}

TEST(a_vision_container_declares_the_tower_when_the_engine_asks_for_it) {
    /* With no encoder budget the tower is not declared, whatever the container carries. */
    {
        RadBuilder b;
        RadModelMeta meta = qwen35_vl_meta();
        RadBuildCtx  ctx  = vl_ctx(0);
        CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK(find_w(b, "v.patch.weight") == nullptr);
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.patch_dim, 3 * 2 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 5120);
    const RecBuf* out = find_b(b, "v.out");
    REQUIRE(out != nullptr);
    CHECK_EQ(out->d.shape[0], 256);
    CHECK_EQ(out->d.shape[1], 5120);

    /* 3 stem + 12 per block + 6 merger, each mapped from the checkpoint's model.visual.*. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
            CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
        }
    CHECK_EQ(nv, 3 + 12 * 2 + 6);
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)1);
    CHECK_EQ(qkv->src[0], std::string("model.visual.blocks.1.attn.qkv.weight"));

    /* The tower's rotary is 2-D, half of the 72-wide head per axis. */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "axial") {
            ++axial;
            CHECK_EQ(pstr(&o, "sections"), std::string("18 18"));
            CHECK_EQ(pint(&o, "head_dim"), 72);
        }
    CHECK_EQ(axial, 2);
    /* The language model's is the interleaved M-RoPE with the checkpoint's sections. */
    int im = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "imrope") {
            ++im;
            CHECK_EQ(pstr(&o, "sections"), std::string("11 11 10"));
            CHECK_EQ(pint(&o, "rotary_dim"), 64);
        }
    CHECK_EQ(im, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* A sizing declare sizes the tower without declaring it to the core. */
    RadBuilder bp;
    RadBuildCtx cp = ctx;
    cp.shape_probe = 1;
    CHECK_OK(qwen35_bf16::declare(&bp, &meta, &cp));
    CHECK(bp.encoder_name.empty());
    CHECK(find_w(bp, "v.patch.weight") != nullptr);
}

TEST(an_encoder_pass_issues_the_tower_and_nothing_else) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* Two segments of 8 x 8 patches: two pictures or two temporal groups of one video. */
    static uint16_t pix[128 * 1536];
    static int32_t  coord[4 * 128];
    static const int32_t cu[] = { 0, 64, 128 };
    RadBatch batch{};
    batch.enc         = 1;
    batch.enc_n_patch = 128;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "rope", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The merger runs over merged rows, a quarter of the patches. */
    const RecIssue* last = &c.issues.back();
    CHECK_EQ(last->n, 32);
    /* The attention reads the segment boundaries, and q, k, v as column slices. */
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    REQUIRE_EQ(at->opd.size(), (size_t)5);
    CHECK_EQ(at->opd[0].cols, 1152);
    CHECK_EQ(at->opd[1].offset, 1152);
    CHECK_EQ(at->opd[2].offset, 2304);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(at->opd[3].rows, 3);
    /* The rotary reads the row and column planes; the grid embedding all four. */
    const RecIssue* ro = issue_of(b, c, "rope");
    REQUIRE(ro != nullptr);
    CHECK(ro->opd[1].raw == coord);
    CHECK_EQ(ro->opd[1].rows, 2);
    CHECK_EQ(ro->opd[1].cols, 128);
    const RecIssue* ge = issue_of(b, c, "grid_embed");
    REQUIRE(ge != nullptr);
    CHECK_EQ(ge->opd[2].rows, 4);

    /* A pass larger than the declared tower fails the step rather than overrunning. */
    RadBatch big = batch;
    big.enc_n_patch = 2048;
    RadCtx c2{ .b = &b, .batch = &big };
    qwen35_bf16::step(&c2, &big);
    CHECK(!c2.step_fail.empty());
    CHECK(c2.issues.empty());
}

TEST(a_step_with_pictures_scatters_their_rows_and_rotates_by_three_components) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A prefill chunk whose tokens 4..11 are one picture's rows. */
    static int32_t rope[3 * 64];
    static const int32_t rows[] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    static uint16_t embd[8 * 5120];
    RadBatch batch = prefill_batch();
    batch.rope_pos   = rope;
    batch.rope_mixed = 1;
    batch.n_mm_rows  = 8;
    batch.mm_rows    = rows;
    batch.mm_embd    = embd;
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    const std::vector<std::string> ops = issued_ops(b, c);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
    const RecIssue* sc = issue_of(b, c, "scatter_rows");
    REQUIRE(sc != nullptr);
    CHECK_EQ(sc->n, 8);
    CHECK(sc->opd[0].raw == embd);
    CHECK_EQ(sc->opd[0].rows, 8);
    CHECK_EQ(sc->opd[0].cols, 5120);
    CHECK(sc->opd[1].raw == rows);
    CHECK_EQ(sc->opd[2].rows, 64);

    /* Every language-model rotary takes the [3, T] planes. */
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope" || pstr(o, "mode") != "imrope") continue;
        ++n;
        CHECK(i.opd[1].raw == rope);
        CHECK_EQ(i.opd[1].rows, 3);
        CHECK_EQ(i.opd[1].cols, 64);
    }
    CHECK_EQ(n, 4);

    /* A text-only step on the same program: no scatter, one position per token. */
    static const int32_t pos[] = { 90, 91, 92, 93 };
    RadBatch text = decode_batch();
    text.positions = pos;
    RadCtx ct{ .b = &b, .batch = &text };
    qwen35_bf16::step(&ct, &text);
    CHECK_LINT(b, &ct);
    CHECK(ct.step_fail.empty());
    CHECK(issued_op(b, ct, "scatter_rows") == nullptr);
    for (const auto& i : ct.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope") continue;
        CHECK(i.opd[1].raw == pos);
        CHECK_EQ(i.opd[1].cols, 0);
    }
}

namespace {
/* The Flash-Next fixture with its vision tower and its interleaved M-RoPE. */
std::vector<const char*> q4vl_keys, q4vl_vals;
RadModelMeta qwen4exp_vl_meta() {
    RadModelMeta m = qwen4exp_meta();
    if (q4vl_keys.empty()) {
        q4vl_keys.assign(m.kv_key, m.kv_key + m.n_kv);
        q4vl_vals.assign(m.kv_val, m.kv_val + m.n_kv);
        for (int i = 7; i < (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0])); ++i) {
            q4vl_keys.push_back(k_vl_keys[i]);
            /* Flash-Next's tower emits rows at its width 2560, not the 27B's 5120. */
            q4vl_vals.push_back(std::strcmp(k_vl_keys[i], "vision_config.out_hidden_size") ? k_vl_vals[i]
                                                                                           : "2560");
        }
    }
    m.n_kv   = (int)q4vl_keys.size();
    m.kv_key = q4vl_keys.data();
    m.kv_val = q4vl_vals.data();
    return m;
}
}  /* namespace */

TEST(a_sizing_declare_with_a_vision_tower_is_the_same_graph) {
    RadModelMeta meta = qwen4exp_vl_meta();
    /* A serving configuration (fp8 cache, lossy wire, three MTP rounds) at levels below a
     * 2048-token chunk. */
    for (int world : { 1, 2 }) for (int lossy : { 0, 1 }) for (int64_t R : { 256, 1024 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = 2048;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            c.max_enc_patches = 16384;
            c.kv_dtype = RAD_F8E4M3;
            c.tp_wire_lossy = lossy;
            c.tp_wire_min_bytes = 64 * 1024;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            CHECK_EQ(full.encoder_name, std::string("vision"));
            RadBuildCtx p = c;
            p.max_tok = R;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);
            CHECK(small.encoder_name.empty());
            size_t first = 0;
            while (first < full.ops.size() && first < small.ops.size() &&
                   full.ops[first].op == small.ops[first].op)
                ++first;
            if (first < full.ops.size() || first < small.ops.size())
                fprintf(stderr, "    world %d rank %d lossy %d R %lld: %zu ops vs %zu, first "
                                "difference at %zu ('%s' vs '%s')\n", world, rank, lossy,
                        (long long)R, full.ops.size(), small.ops.size(),
                        first, first < full.ops.size() ? full.ops[first].op.c_str() : "-",
                        first < small.ops.size() ? small.ops[first].op.c_str() : "-");
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            CHECK_EQ(first, full.ops.size());
            /* Every fixed parameter matches the real declare's, the tower's included: a level runs
             * the real declare's kernels, so only ranged row counts may differ. */
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld real, %lld at %lld\n", i,
                                full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival,
                                (long long)R);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            /* A level shrinks only a buffer's rows; other extents stay the real declare's. */
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(q.rank, f.rank);
                for (uint32_t d = 1; d < f.rank && d < q.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld real, %lld at %lld\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d], (long long)R);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
            }
        }
    }
}

/* CalibExperts stores each expert's Gram as a packed upper triangle; a packing slip could move
 * entries between experts or transpose them while every result stays k x k and symmetric. Two
 * passes of distinct symmetric Grams go in, and each expert must come back as the sum of its own
 * two, entry for entry, at an awkward width. */
TEST(the_per_expert_tap_keeps_each_experts_gram_whole) {
    const int64_t ne = 5, k = 7;
    rad::arch::CalibExperts ce;
    ce.init(ne, k);
    std::vector<float> want((size_t)(ne * k * k), 0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> full((size_t)(ne * k * k));
        for (int64_t e = 0; e < ne; ++e)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t lo = std::min(i, j), hi = std::max(i, j);
                    const float v = (float)(1000 * e + 31 * lo + hi + 7 * pass);
                    full[(size_t)((e * k + i) * k + j)] = v;
                    want[(size_t)((e * k + i) * k + j)] += v;
                }
        ce.add(full.data());
    }
    std::vector<float> got((size_t)(k * k));
    int bad = 0;
    for (int64_t e = 0; e < ne; ++e) {
        ce.unpack(e, got.data());
        for (int64_t x = 0; x < k * k; ++x) bad += got[(size_t)x] != want[(size_t)(e * k * k + x)];
    }
    CHECK_EQ(bad, 0);
}

/* ==================================================================== LFM2 */
namespace {

/* d1-3B's graph cut to four layers: conv, conv, attention, conv, an irregular schedule the plugin
 * must read, not derive. A VL checkpoint (it has a text_config) with a tied head. */
const char* k_lfm2_keys[] = { "text_config.num_hidden_layers", "layer_types", "conv_L_cache",
                              "tie_word_embeddings" };
const char* k_lfm2_vals[] = { "4", "conv conv full_attention conv", "3", "true" };

RadModelMeta lfm2_meta(int n_kv = 4) {
    RadModelMeta m{};
    m.arch_id = "lfm2"; m.name = "test-lfm2"; m.quant = "";
    m.n_layers = 4; m.n_embd = 64; m.n_head = 4; m.n_head_kv = 1; m.head_dim = 16;
    m.n_ff = 128; m.n_vocab = 512; m.n_ctx_train = 32768;
    m.rms_eps = 1e-5f; m.rope_theta = 1e6f; m.rope_scale = 1.0f;
    m.n_kv = n_kv; m.kv_key = k_lfm2_keys; m.kv_val = k_lfm2_vals;
    return m;
}

std::vector<std::string> lfm2_layer_ops(bool attention) {
    std::vector<std::string> v = { "rmsnorm" };
    if (attention) {
        for (const char* s : { "gemm_nt", "gemm_nt", "gemm_nt", "rmsnorm", "rmsnorm", "rope",
                               "rope", "attn_paged", "kv_store", "gemm_nt" })
            v.push_back(s);
    } else {
        for (const char* s : { "gemm_nt", "mul", "causal_conv1d", "mul", "gemm_nt" })
            v.push_back(s);
    }
    v.push_back("add");
    append(v, mlp_ops(false));
    return v;
}

}  /* namespace */

TEST(lfm2_declares_the_checkpoints_own_schedule) {
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    std::vector<std::string> want = { "embed_lookup" };
    for (bool att : { false, false, true, false }) append(want, lfm2_layer_ops(att));
    for (const char* s : { "gather_rows", "rmsnorm", "logits_gemm" }) want.push_back(s);
    CHECK_EQ(join(op_names(b)), join(want));

    /* Two state groups: paged KV bound to the attention layer, and a conv window (n_embd channels,
     * the checkpoint's width) bound to the other three. */
    REQUIRE(b.kv_groups.size() == 2);
    CHECK_EQ(b.kv_groups[0].d.kind, (int)RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.kind, (int)RAD_KV_CONV);
    CHECK_EQ(b.kv_groups[1].d.conv_width, (int64_t)3);
    CHECK_EQ(b.kv_groups[1].d.n_head_kv * b.kv_groups[1].d.head_dim, (int64_t)64);
    REQUIRE(b.binds.size() == 4);
    for (const auto& bd : b.binds)
        CHECK_EQ(bd.second, bd.first == 2 ? (rad_kvgroup)1 : (rad_kvgroup)2);

    /* A plain RMSNorm at the checkpoint's epsilon, NeoX over the whole head. */
    const RecOp* norm = find_op(b, "rmsnorm");
    CHECK_NEAR(pflt(norm, "eps"), 1e-5, 1e-9);
    CHECK_EQ(param(norm, "wadd") ? pflt(norm, "wadd") : 0.0, 0.0);
    CHECK_EQ(pstr(find_op(b, "rope"), "mode"), "neox");
    CHECK_EQ(pint(find_op(b, "rope"), "rotary_dim"), 16);
    CHECK_EQ(pint(find_op(b, "attn_paged"), "gqa"), 4);
    CHECK_EQ(pint(find_op(b, "causal_conv1d"), "width"), 3);

    /* The VL checkpoint's names, and a tied head read from the embedding when no lm_head
     * exists. */
    const RecMap* in = find_map(b, "blk.0.conv_in.weight");
    REQUIRE(in != nullptr);
    CHECK_EQ(in->src[0], "model.language_model.layers.0.conv.in_proj.weight");
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr && head->src.size() == 2);
    CHECK_EQ(head->src[0], "lm_head.weight");
    CHECK_EQ(head->src[1], "model.language_model.embed_tokens.weight");
}

TEST(lfm2_text_checkpoint_keeps_its_names_under_model) {
    /* No text_config: a text-only LFM2, with layers under `model.` */
    static const char* keys[] = { "layer_types", "conv_L_cache" };
    static const char* vals[] = { "conv conv full_attention conv", "3" };
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    meta.n_kv = 2; meta.kv_key = keys; meta.kv_val = vals;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
    const RecMap* q = find_map(b, "blk.2.attn_q.weight");
    REQUIRE(q != nullptr);
    CHECK_EQ(q->src[0], "model.layers.2.self_attn.q_proj.weight");
}

TEST(lfm2_refuses_what_it_does_not_serve) {
    {
        RadBuilder b;
        RadModelMeta meta = lfm2_meta(1);   /* no layer_types */
        RadBuildCtx ctx = build_ctx();
        CHECK(lfm2_bf16::declare(&b, &meta, &ctx) < 0);
    }
    {
        RadBuilder b;
        RadModelMeta meta = lfm2_meta();
        RadBuildCtx ctx = build_ctx(0, 2);
        CHECK_EQ(lfm2_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    }
    {
        static const char* keys[] = { "layer_types", "conv_L_cache" };
        static const char* vals[] = { "conv sliding_attention conv conv", "3" };
        RadBuilder b;
        RadModelMeta meta = lfm2_meta();
        meta.n_kv = 2; meta.kv_key = keys; meta.kv_val = vals;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(lfm2_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    }
}

TEST(lfm2_issues_the_convolution_once_per_half_of_a_mixed_step) {
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    /* Two decode rows (a verify of 3 and a plain decode) and one prefill chunk of 5. */
    static const int32_t tok[9]  = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    static const int32_t pos[9]  = { 40, 41, 42, 30, 0, 1, 2, 3, 4 };
    static const int32_t cu[4]   = { 0, 3, 4, 9 };
    static const int32_t ctxl[3] = { 40, 30, 0 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static const int32_t out[3]  = { 2, 3, 8 };
    static int32_t cv_idx[3] = { 4, 5, 6 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = lfm2_bf16::g_model[0].kv_attn;
    kv[1].group = lfm2_bf16::g_model[0].kv_conv; kv[1].state_index = cv_idx;
    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 9; mixed.n_seq = 3; mixed.max_q_len = 5;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 4;
    mixed.token_ids = tok; mixed.positions = pos; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc; mixed.n_out = 3; mixed.out_ids = out;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c{ .b = &b, .batch = &mixed };
    lfm2_bf16::step(&c, &mixed);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    int convs = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "mul" && i.opd.size() == 3 && i.opd[0].cols > 0) {
            /* conv(B * x): the B and x thirds of the [B | C | x] row as column slices. */
            if (i.opd[1].cols > 0) {
                CHECK_EQ(i.opd[0].offset, (int64_t)0);
                CHECK_EQ(i.opd[1].offset, (int64_t)128);
            } else {
                CHECK_EQ(i.opd[0].offset, (int64_t)64);    /* C * conv(...) */
            }
            CHECK_EQ(i.opd[0].cols, (int64_t)64);
        }
        if (o->op != "causal_conv1d") continue;
        REQUIRE(i.opd.size() == 8);
        if (convs++ % 2 == 0) {      /* the decode half */
            CHECK_EQ(i.n, (int64_t)4);
            CHECK_EQ(i.opd[0].rows, (int64_t)4);
            CHECK(i.opd[3].raw == (void*)cu);         CHECK_EQ(i.opd[3].rows, (int64_t)3);
            CHECK(i.opd[4].raw == (void*)cv_idx);
            CHECK(i.opd[6].raw == (void*)nacc);
        } else {                     /* prefill half: per-seq arrays past the decode seqs */
            CHECK_EQ(i.n, (int64_t)9);
            CHECK_EQ(i.opd[0].rows, (int64_t)9);
            CHECK(i.opd[3].raw == (void*)(cu + 2));   CHECK_EQ(i.opd[3].rows, (int64_t)2);
            CHECK(i.opd[4].raw == (void*)(cv_idx + 2));
            CHECK(i.opd[5].raw == (void*)(ctxl + 2));
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_NONE);
        }
    }
    CHECK_EQ(convs, 6);   /* three conv layers, two halves each */
}

/* ==================================================================== LFM2-VL's tower */
namespace {

/* lfm2_meta's text model with a SigLIP2 tower cut to two layers of 64 (four heads of 16), d1's
 * patch, position table and unshuffle, and a 96-wide projector. */
const char* k_lfm2vl_keys[] = { "text_config.num_hidden_layers", "layer_types", "conv_L_cache",
                                "tie_word_embeddings", "vision_config.model_type",
                                "vision_config.num_hidden_layers", "vision_config.hidden_size",
                                "vision_config.num_attention_heads", "vision_config.intermediate_size",
                                "vision_config.patch_size", "vision_config.num_patches",
                                "vision_config.hidden_act", "vision_config.vision_use_head",
                                "downsample_factor", "projector_hidden_size",
                                "projector_hidden_act", "projector_bias",
                                "projector_use_layernorm", "max_num_patches" };
const char* k_lfm2vl_vals[] = { "4", "conv conv full_attention conv", "3", "true",
                                "siglip2_vision_model", "2", "64", "4", "128", "16", "256",
                                "gelu_pytorch_tanh", "false", "2", "96", "gelu", "true", "false",
                                "1024" };
const char* k_lfm2vl_norm_vals[] = { "4", "conv conv full_attention conv", "3", "true",
                                     "siglip2_vision_model", "2", "64", "4", "128", "16", "256",
                                     "gelu_pytorch_tanh", "false", "2", "96", "gelu", "true", "true",
                                     "1024" };

RadModelMeta lfm2vl_meta(bool proj_norm = false) {
    RadModelMeta m = lfm2_meta();
    m.n_kv = (int)(sizeof(k_lfm2vl_keys) / sizeof(k_lfm2vl_keys[0]));
    m.kv_key = k_lfm2vl_keys;
    m.kv_val = proj_norm ? k_lfm2vl_norm_vals : k_lfm2vl_vals;
    return m;
}

}  /* namespace */

TEST(lfm2_vl_declares_the_siglip2_tower_and_its_projector) {
    {   /* No encoder budget: text only, whatever the container carries. */
        RadBuilder b;
        RadModelMeta meta = lfm2vl_meta();
        RadBuildCtx ctx = vl_ctx(0);
        CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = lfm2vl_meta();
    RadBuildCtx ctx = vl_ctx(1024);
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)RAD_MM_IMAGE);
    CHECK_EQ(b.encoder.patch_dim, 3 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 64);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* 3 stem + 12 per layer + 2 post-norm + 4 projector, each mapped from the checkpoint. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
        }
    CHECK_EQ(nv, 3 + 2 * 12 + 2 + 4);
    /* q, k and v are concatenated at convert into one QKV weight and bias. */
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[0],
             std::string("model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight"));
    CHECK_EQ(qkv->src[2],
             std::string("model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.weight"));
    const RecMap* l1 = find_map(b, "v.proj.fc1.weight");
    REQUIRE(l1 != nullptr);
    CHECK_EQ(l1->src[0], std::string("model.multi_modal_projector.linear_1.weight"));
    /* The projector reads four patches as one row; it uses erf GELU, the tower tanh GELU. */
    const RecWeight* w1 = find_w(b, "v.proj.fc1.weight");
    REQUIRE(w1 != nullptr);
    CHECK_EQ(w1->d.shape[0], 96);
    CHECK_EQ(w1->d.shape[1], 4 * 64);
    int tanh = 0, erf = 0;
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt_bias") {
            tanh += pstr(&o, "act") == "gelu_tanh";
            erf += pstr(&o, "act") == "gelu";
        }
    CHECK_EQ(tanh, 2);
    CHECK_EQ(erf, 1);
    /* No rotary in the tower; the position table is resampled as torch's antialiased bilinear,
     * and attention's block height is bounded by the longest segment the processor makes. */
    CHECK_EQ(count_op(b, "grid_embed_aa"), 1);
    CHECK_EQ(count_op(b, "grid_embed"), 0);
    CHECK_EQ(pint(find_op(b, "attn_dense"), "max_seqlen"), 1024);
    CHECK_EQ(pint(find_op(b, "attn_dense"), "head_dim"), 16);

    /* A projector with a layer norm declares it over the merged rows. */
    RadBuilder bn;
    RadModelMeta mn = lfm2vl_meta(true);
    CHECK_OK(lfm2_bf16::declare(&bn, &mn, &ctx));
    CHECK(find_w(bn, "v.proj.norm.weight") != nullptr);
    CHECK_EQ(count_op(bn, "layernorm"), count_op(b, "layernorm") + 1);
}

TEST(lfm2_vl_encoder_pass_and_the_rows_it_scatters) {
    RadBuilder b;
    RadModelMeta meta = lfm2vl_meta();
    RadBuildCtx ctx = vl_ctx(1024);
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    /* A tile of 8 x 8 patches and a thumbnail of 4 x 8. */
    static uint16_t pix[96 * 768];
    static int32_t  coord[4 * 96];
    static const int32_t cu[] = { 0, 64, 96 };
    RadBatch batch{};
    batch.enc         = RAD_MM_IMAGE;
    batch.enc_n_patch = 96;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c{ .b = &b, .batch = &batch };
    lfm2_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed_aa" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));
    /* The position resample reads all four coordinate planes; the projector runs on merged rows. */
    const RecIssue* ge = issue_of(b, c, "grid_embed_aa");
    REQUIRE(ge != nullptr);
    CHECK(ge->opd[2].raw == coord);
    CHECK_EQ(ge->opd[2].rows, 4);
    CHECK_EQ(c.issues.back().n, 24);
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    CHECK_EQ(at->opd[1].offset, 64);
    CHECK_EQ(at->opd[2].offset, 128);
    CHECK(at->opd[3].raw == cu);

    /* A text step with media rows scatters them over the lookup before the first layer. */
    static const int32_t tok[4] = { 1, 2, 3, 4 }, pos[4] = { 0, 1, 2, 3 }, cus[2] = { 0, 4 };
    static const int32_t ctxl[1] = { 0 }, out[1] = { 3 }, rows[2] = { 1, 2 };
    static uint16_t embd[2 * 64];
    static int32_t cv_idx[1] = { 0 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = lfm2_bf16::g_model[0].kv_attn;
    kv[1].group = lfm2_bf16::g_model[0].kv_conv; kv[1].state_index = cv_idx;
    RadBatch tb{};
    tb.phase = RAD_PHASE_PREFILL;
    tb.n_tok = 4; tb.n_seq = 1; tb.max_q_len = 4;
    tb.token_ids = tok; tb.positions = pos; tb.cu_seqlens = cus; tb.ctx_lens = ctxl;
    tb.n_out = 1; tb.out_ids = out; tb.n_kv_groups = 2; tb.kv = kv;
    tb.n_mm_rows = 2; tb.mm_rows = rows; tb.mm_embd = embd;
    RadCtx ct{ .b = &b, .batch = &tb };
    lfm2_bf16::step(&ct, &tb);
    CHECK_LINT(b, &ct);
    CHECK(ct.step_fail.empty());
    const std::vector<std::string> ops = issued_ops(b, ct);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
}

/* =============================== gemma4 dense: two attention classes =========================== */

TEST(declare_is_a_gemma4_trunk) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    /* 12 layers: 10 sliding + 2 global (layers 5 and 11). */
    CHECK_EQ(count_op(b, "attn_paged"),  12);
    /* the cache store rides the fused prologue where it resolves (qkv_norm_rope_kv_store) */
    CHECK_EQ(count_op(b, "qkv_norm_rope_kv_store"), 12);
    CHECK_EQ(count_op(b, "kv_store"),     0);
    CHECK_EQ(count_op(b, "rmsnorm_p"),   12);   /* weightless value norm, one per layer */
    CHECK_EQ(count_op(b, "rope"),        24);   /* q + k */
    CHECK_EQ(count_op(b, "gelu_mul"),    12);   /* Gemma's FFN activation, not SwiGLU */
    CHECK_EQ(count_op(b, "silu_mul"),     0);
    CHECK_EQ(count_op(b, "gemm_nt"),     70);   /* 10x6 sliding (q,k,v,o,gu,down) + 2x5 global (no v) */
    CHECK_EQ(count_op(b, "rmsnorm"),     73);   /* 4 attn + 2 ffn per layer, plus the final norm */
    CHECK_EQ(count_op(b, "add"),         24);   /* the two GemmaRMSNorm residuals per layer */
    CHECK_EQ(count_op(b, "embed_lookup"),  1);
    CHECK_EQ(count_op(b, "gather_rows"),   1);
    CHECK_EQ(count_op(b, "logits_gemm"),   1);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);
    CHECK_EQ(count_op(b, "all_reduce"),    0);  /* world 1 */
}

/* The plugin states the reply format, since template analysis cannot read Gemma 4's template.
 * Calls are the object family (a name and one object in Gemma's dialect) and the thought channel
 * may reopen after the answer. chatparse_gemma4_test covers parsing; this checks the export. */
TEST(gemma4_declares_its_reply_format) {
    RadModelMeta meta = gemma4_meta();
    const RadChatFormat* f = gemma4_bf16::chat_format(&meta);
    REQUIRE(f != nullptr);
    CHECK_EQ(f->struct_size, (uint32_t)sizeof(RadChatFormat));
    CHECK_EQ(f->args_form, (uint32_t)RAD_CHAT_ARGS_OBJECT);
    CHECK_EQ(std::string(f->call_open), std::string("<|tool_call>"));
    CHECK_EQ(std::string(f->call_close), std::string("<tool_call|>"));
    CHECK_EQ(std::string(f->name_prefix), std::string("call:"));
    CHECK_EQ(std::string(f->name_suffix), std::string(""));
    CHECK_EQ(std::string(f->object_string_open), std::string("<|\"|>"));
    CHECK_EQ(std::string(f->object_string_close), std::string("<|\"|>"));
    CHECK(f->object_bare_keys != 0);
    CHECK_EQ(std::string(f->reasoning_open), std::string("<|channel>thought\n"));
    CHECK_EQ(std::string(f->reasoning_close), std::string("<channel|>"));
    CHECK(f->reasoning_anywhere != 0);
    REQUIRE(f->n_triggers == 1);
    CHECK_EQ(std::string(f->triggers[0]), std::string("<|tool_call>"));
    CHECK(f->arg_prefix == nullptr && f->arg_value_suffix == nullptr);
}

TEST(gemma4_declares_two_kv_groups_and_binds_the_schedule) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 2);
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);   /* sliding, declared first */
    CHECK_EQ(b.kv_groups[0].d.window,    1024);
    CHECK_EQ(b.kv_groups[0].d.n_head_kv, 16);
    CHECK_EQ(b.kv_groups[0].d.head_dim,  256);
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);     /* global */
    CHECK_EQ(b.kv_groups[1].d.window,    0);
    CHECK_EQ(b.kv_groups[1].d.n_head_kv, 4);
    CHECK_EQ(b.kv_groups[1].d.head_dim,  512);

    /* (l + 1) % 6 == 0 gives global layers 5 and 11; group 1 is sliding (declared first), 2
     * global. */
    CHECK_EQ((int)b.binds.size(), 12);
    for (int l = 0; l < 12; ++l) {
        const bool want_global = ((l + 1) % 6 == 0);
        CHECK_EQ((int)b.binds[(size_t)l].first, l);
        CHECK_EQ((int)b.binds[(size_t)l].second, want_global ? 2 : 1);
    }
}

TEST(gemma4_attention_shapes_for_both_classes) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    /* layer 0 (sliding): q 32x256, k/v 16x256, windowed. */
    {
        const RecOp* at = find_op(b, "attn_paged", 0);
        CHECK(at != nullptr);
        CHECK_EQ(pint(at, "head_dim"), 256);
        CHECK_EQ(pint(at, "gqa"),      2);
        CHECK_EQ(pint(at, "window"),   1024);
        CHECK_EQ(pint(at, "n_head"),   32);
        CHECK_EQ(pint(at, "block_size"), 16);
        CHECK_EQ(pstr(at, "q_dtype"), std::string("bf16"));
    }
    /* layer 5 (the first global, (5+1)%6==0): q 32x512, k 4x512, windowless. */
    {
        const RecOp* at = find_op(b, "attn_paged", 5);
        CHECK(at != nullptr);
        CHECK_EQ(pint(at, "head_dim"), 512);
        CHECK_EQ(pint(at, "gqa"),      8);
        CHECK_EQ(pint(at, "window"),   0);
        CHECK_EQ(pint(at, "n_head"),   32);
    }

    /* rope: sliding is standard over the full 256 (theta 1e4); global is proportional over 128 of
     * 512 (the "head" ladder, theta 1e6). Ropes are declared q, k per layer, so index =
     * 2*layer + (q ? 0 : 1). */
    {
        const RecOp* r  = find_op(b, "rope", 0);   /* layer 0, q */
        CHECK(r != nullptr);
        CHECK_EQ(pint(r, "head_dim"),   256);
        CHECK_EQ(pint(r, "rotary_dim"), 256);
        CHECK_NEAR(pflt(r, "theta"),    1e4, 1.0);
        CHECK_EQ(pstr(r, "ladder"), std::string("rotary"));
        CHECK_EQ(pint(r, "n_head"),    32);

        const RecOp* rg = find_op(b, "rope", 10);  /* layer 5, q */
        CHECK(rg != nullptr);
        CHECK_EQ(pint(rg, "head_dim"),   512);
        CHECK_EQ(pint(rg, "rotary_dim"), 128);
        CHECK_NEAR(pflt(rg, "theta"),    1e6, 1.0);
        CHECK_EQ(pstr(rg, "ladder"), std::string("head"));
        CHECK_EQ(pint(rg, "n_head"),    32);

        /* the k rope is a separate buffer, run as a q-only form over the k heads: n_head is the
         * count to rotate (n_head_kv = 4) and n_head_kv is 0. */
        const RecOp* rkg = find_op(b, "rope", 11); /* layer 5, k */
        CHECK(rkg != nullptr);
        CHECK_EQ(pint(rkg, "n_head"),    4);
        CHECK_EQ(pint(rkg, "n_head_kv"), 0);
    }

    /* the weightless value norm reads 256 dims on a sliding layer and 512 on a global one, and
     * the global layer has no v_proj (v is k). */
    {
        const RecOp* vn  = find_op(b, "rmsnorm_p", 0);   /* layer 0 */
        CHECK(vn != nullptr);
        CHECK_EQ(pint(vn, "n"), 256);
        const RecOp* vng = find_op(b, "rmsnorm_p", 5);   /* layer 5 */
        CHECK(vng != nullptr);
        CHECK_EQ(pint(vng, "n"), 512);
    }
    CHECK(find_w(b, "blk.0.attn_v.weight") != nullptr);  /* sliding has a v projection */
    CHECK(find_w(b, "blk.5.attn_v.weight") == nullptr);  /* global: k == v */
    CHECK(find_w(b, "blk.5.attn_q.weight") != nullptr);
    CHECK(find_w(b, "blk.5.attn_o.weight") != nullptr);
}

TEST(gemma4_final_norm_softcap_and_vocab) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const RecOp* cap = find_op(b, "logit_softcap", 0);
    CHECK(cap != nullptr);
    CHECK_EQ(pint(cap, "n"), 262144);
    CHECK_NEAR(pflt(cap, "cap"), 30.0, 1e-9);
    CHECK_EQ(pstr(cap, "dtype"), std::string("f32"));

    const RecOp* lg = find_op(b, "logits_gemm", 0);
    CHECK(lg != nullptr);
    CHECK_EQ(pint(lg, "n_vocab"), 262144);
    CHECK_EQ(pint(lg, "n_embd"),  5376);

    const RecWeight* emb = find_w(b, "token_embd.weight");
    CHECK(emb != nullptr);
    CHECK_EQ(emb->d.shape[0], 262144);
    CHECK_EQ(emb->d.shape[1], 5376);
    CHECK_EQ(emb->d.shard, RAD_SHARD_ROW);

    /* lm_head is tied to the embedding. */
    CHECK(find_map(b, "output.weight") != nullptr);

    /* The embedding is scaled by sqrt(n_embd) at lookup (as the reference's build_inp_embd and
     * HF's ScaledWordEmbedding do), not baked into the table, so the table stays tied to the head.
     * HF multiplies by `embed_scale.to(bf16)`, so sqrt(5376) = 73.32 is 73.5. */
    const RecOp* el = find_op(b, "embed_lookup", 0);
    CHECK(el != nullptr);
    CHECK_NEAR(pflt(el, "mult"), 73.5, 1e-9);
}

TEST(gemma4_every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
}

/* attention_k_eq_v decides whether global layers have a v_proj: 1 (31B) reads V from K, 0 declares
 * one, and absence is refused. */
TEST(gemma4_global_v_follows_attention_k_eq_v) {
    for (const char* v : { "1", "0", "" }) {
        std::vector<const char*> keys(std::begin(k_gem4_keys), std::end(k_gem4_keys));
        std::vector<const char*> vals(std::begin(k_gem4_vals), std::end(k_gem4_vals));
        if (*v) vals.back() = v;
        else { keys.pop_back(); vals.pop_back(); }
        RadModelMeta meta = gemma4_meta();
        meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        const int rc = gemma4_bf16::declare(&b, &meta, &ctx);
        if (!*v) { CHECK(rc != RAD_OK); continue; }
        CHECK_OK(rc);
        CHECK(find_w(b, "blk.0.attn_v.weight") != nullptr);              /* sliding: always */
        CHECK_EQ(find_w(b, "blk.5.attn_v.weight") != nullptr, *v == '0');  /* global */
    }
}

TEST(gemma4_every_weight_has_a_name_map) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(a_gemma4_decode_step_runs_both_classes_in_order) {
    RadBuilder b;
    b.refuse.insert("qkv_norm_rope");   /* force the unfused prologue; fused is checked below */
    b.refuse.insert("qkv_norm_rope_kv_store");
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 12; ++l) {
        if ((l + 1) % 6 == 0) {
            /* global: k == v, so no v gemm; the weightless v norm derives from k. */
            for (const char* s : { "rmsnorm", "gemm_nt", "gemm_nt",
                                   "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope",
                                   "kv_store", "attn_paged", "gemm_nt", "rmsnorm", "add" })
                want.push_back(s);
        } else {
            /* sliding: a v gemm feeds the weightless v norm. */
            for (const char* s : { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                   "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope",
                                   "kv_store", "attn_paged", "gemm_nt", "rmsnorm", "add" })
                want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "gelu_mul", "gemm_nt", "rmsnorm", "add" })
            want.push_back(s);
        want.push_back("scale");   /* per-layer scalar over the residual stream */
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The sampled-row ops band on n_out; the attention bands on q_len. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "gather_rows" || o->op == "logits_gemm" || o->op == "logit_softcap")
            CHECK_EQ(i.n, (int64_t)batch.n_out);
    }
}

/* The w4a8 trunk fuses its residual junctions: the first block's entry, each attention ->
 * feed-forward and each feed-forward -> next attention is one `rmsnorm_quant_i8` (post-norm, add,
 * layer scalar, norm and quantiser), and q, k and v read one quantised pair. The last layer goes to
 * the final norm unfused. Without the fused op the whole chain stays unfused, still with the shared
 * q/k/v pair. */
static std::vector<std::string> gemma4_w4_step(const std::set<std::string>& refuse, int world = 1) {
    RadBuilder b;
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
               { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
    for (const auto& r : refuse) b.refuse.insert(r);
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx(0, world);
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch, .world = world };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    return issued_ops(b, c);
}

/* At two ranks the all-reduce before each junction is folded into it; only the embedding's and
 * the last feed-forward's (which feeds the final norm, not a quantiser) remain standalone. Without
 * the fused collective, junctions keep their preceding all-reduces. */
TEST(a_gemma4_w4a8_step_at_two_ranks_folds_the_all_reduce_into_the_junction) {
    std::vector<std::string> want = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "ar_rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            want.push_back(s);
        if (l < 11) want.push_back("ar_rmsnorm_quant_i8");
        else for (const char* s : { "all_reduce", "rmsnorm", "add", "scale" }) want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({}, 2)), join(want));

    std::vector<std::string> plain = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) plain.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "all_reduce", "rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8",
                               "gemm_nt_q", "all_reduce" })
            plain.push_back(s);
        if (l < 11) plain.push_back("rmsnorm_quant_i8");
        else for (const char* s : { "rmsnorm", "add", "scale" }) plain.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        plain.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({ "ar_rmsnorm_quant_i8" }, 2)), join(plain));
}

TEST(a_gemma4_w4a8_step_fuses_its_residual_junctions) {
    std::vector<std::string> want = { "embed_lookup", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope_kv_store",
                               "attn_paged", "quant_act_i8", "gemm_nt_q", "rmsnorm_quant_i8",
                               "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            want.push_back(s);
        if (l < 11) want.push_back("rmsnorm_quant_i8");
        else for (const char* s : { "rmsnorm", "add", "scale" }) want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({})), join(want));
}

TEST(a_gemma4_w4a8_step_without_the_fused_ops_keeps_the_chain_and_one_qkv_quantiser) {
    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        want.push_back("rmsnorm");
        want.push_back("quant_act_i8");
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope", "kv_store",
                               "attn_paged", "quant_act_i8", "gemm_nt_q", "rmsnorm", "add",
                               "rmsnorm", "quant_act_i8", "gemm_nt_q", "gelu_mul", "quant_act_i8",
                               "gemm_nt_q", "rmsnorm", "add", "scale" })
            want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({ "rmsnorm_quant_i8", "qkv_norm_rope", "qkv_norm_rope_kv_store",
                                   "gelu_mul_quant_i8" })), join(want));
}

/* =============================== gemma4: the assistant (MTP) head =============================== */

namespace {
/* The draft passes the core stages for a head with no cache of its own: one row per drafting
 * sequence. Four sequences; the history pass marks every row as a draft row. */
static const int32_t kHeadRows[] = { 0, 1, 2, 3 };
RadBatch head_batch(int pass) {
    RadBatch b{};
    b.phase         = RAD_PHASE_DECODE;
    b.n_tok         = 4;
    b.n_seq         = 4;
    b.max_q_len     = 1;
    b.max_ctx_len   = 100;
    b.draft_pass    = pass;
    if (pass < 0) {
        b.n_out         = 4;
        b.out_ids       = kHeadRows;
        b.n_draft_out   = 4;
        b.draft_out_ids = kHeadRows;
    }
    return b;
}
}  /* namespace */

TEST(gemma4_declares_no_head_without_one_or_without_speculation) {
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_meta();
        RadBuildCtx  ctx  = build_ctx();
        ctx.max_spec = 3;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "mtp.pre_proj.weight") == nullptr);
    }
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_draft_meta();
        RadBuildCtx  ctx  = build_ctx();      /* max_spec 0 */
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "mtp.pre_proj.weight") == nullptr);
        CHECK_EQ(count_op(b, "attn_paged"), 12);
    }
}

TEST(gemma4_assistant_is_a_serial_head_with_no_cache_of_its_own) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(b.drafter.depth, 3);
    CHECK_EQ(b.drafter_name, std::string("mtp"));
    CHECK_EQ((int)b.drafter.proposal, 0);         /* proposes through the sampler chain */
    CHECK(b.drafter_shared);
    CHECK_EQ(b.drafter_pos, gemma4_bf16::kHeadPosOffset);

    /* It binds no layer and writes no K or V; only the trunk's twelve bindings exist. */
    CHECK_EQ((int)b.binds.size(), 12);
    CHECK_EQ((int)b.kv_groups.size(), 2);

    /* Four head attention layers beside the trunk's twelve, none storing. */
    CHECK_EQ(count_op(b, "attn_paged"), 16);
    CHECK_EQ(count_op(b, "kv_store") + count_op(b, "qkv_norm_rope_kv_store"), 12);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);    /* trunk only; head logits are not capped */
    CHECK_EQ(count_op(b, "logits_gemm"), 2);

    for (int l = 0; l < 4; ++l) {
        const RecOp* at = find_op(b, "attn_paged", 12 + l);
        REQUIRE(at != nullptr);
        const bool full = (l == 3);
        CHECK_EQ(pint(at, "head_dim"), full ? 512 : 256);
        CHECK_EQ(pint(at, "gqa"), full ? 8 : 2);
        CHECK_EQ(pint(at, "window"), full ? 0 : 1024);
        CHECK_EQ(pint(at, "causal"), 1);
        CHECK_NEAR(pflt(at, "scale"), 1.0, 1e-12);
        const RecParam* ql = param(at, "q_len");
        REQUIRE(ql != nullptr);
        CHECK_EQ(ql->ival, 1);                        /* one row per sequence: decode band only */
        CHECK_EQ(ql->ihi, 1);
    }

    /* Weights: q-only attention, the two projections and the head's own tied table. */
    const RecWeight* wq0 = find_w(b, "mtp.blk.0.attn_q.weight");
    const RecWeight* wq3 = find_w(b, "mtp.blk.3.attn_q.weight");
    REQUIRE(wq0 != nullptr); REQUIRE(wq3 != nullptr);
    CHECK_EQ(wq0->d.shape[0], 32 * 256); CHECK_EQ(wq0->d.shape[1], 1024);
    CHECK_EQ(wq3->d.shape[0], 32 * 512); CHECK_EQ(wq3->d.shape[1], 1024);
    const RecWeight* wo3 = find_w(b, "mtp.blk.3.attn_o.weight");
    REQUIRE(wo3 != nullptr);
    CHECK_EQ(wo3->d.shape[0], 1024); CHECK_EQ(wo3->d.shape[1], 32 * 512);
    CHECK(find_w(b, "mtp.blk.0.attn_k.weight") == nullptr);
    CHECK(find_w(b, "mtp.blk.0.attn_v.weight") == nullptr);
    const RecWeight* gu = find_w(b, "mtp.blk.1.ffn_gate_up.weight");
    REQUIRE(gu != nullptr);
    CHECK_EQ(gu->d.shape[0], 2 * 8192); CHECK_EQ(gu->d.shape[1], 1024);
    const RecWeight* pre  = find_w(b, "mtp.pre_proj.weight");
    const RecWeight* post = find_w(b, "mtp.post_proj.weight");
    REQUIRE(pre != nullptr); REQUIRE(post != nullptr);
    CHECK_EQ(pre->d.shape[0], 1024);  CHECK_EQ(pre->d.shape[1], 2 * 5376);
    CHECK_EQ(post->d.shape[0], 5376); CHECK_EQ(post->d.shape[1], 1024);
    CHECK_EQ(pre->d.shard, RAD_SHARD_NONE);
    const RecWeight* lm = find_w(b, "mtp.output.weight");
    REQUIRE(lm != nullptr);
    CHECK_EQ(lm->d.shape[0], 262144); CHECK_EQ(lm->d.shape[1], 1024);
    CHECK_EQ(lm->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm->d.access, RAD_ACCESS_VOCAB);

    /* Checkpoint names merged under `draft.`: the head's table is its own embedding (no
     * fallback), and gate and up are fused like the trunk's. */
    const RecMap* mlm = find_map(b, "mtp.output.weight");
    REQUIRE(mlm != nullptr);
    CHECK_EQ(mlm->src[0], std::string("draft.model.embed_tokens.weight"));
    const RecMap* mgu = find_map(b, "mtp.blk.1.ffn_gate_up.weight");
    REQUIRE(mgu != nullptr);
    CHECK_EQ(mgu->mode, RAD_MAP_CONCAT);
    CHECK_EQ(mgu->src[0], std::string("draft.model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(mgu->src[1], std::string("draft.model.layers.1.mlp.up_proj.weight"));
    const RecMap* mpre = find_map(b, "mtp.pre_proj.weight");
    REQUIRE(mpre != nullptr);
    CHECK_EQ(mpre->src[0], std::string("draft.pre_projection.weight"));

    /* The entry embeds through the trunk's table with the trunk's scale, sqrt(5376) rounded to
     * bf16 as the reference's embedding module holds it. */
    const RecOp* el = find_op(b, "embed_lookup", 1);
    REQUIRE(el != nullptr);
    CHECK_EQ(pint(el, "n_embd"), 5376);
    CHECK_NEAR(pflt(el, "mult"), 73.5, 1e-9);
    CHECK(el->w.size() == 1 && find_op(b, "embed_lookup", 0)->w == el->w);

    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(gemma4_head_passes_read_the_last_trunk_layer_of_their_class) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    for (int pass : { -1, 2 }) {
        RadBatch batch = head_batch(pass);
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());

        std::vector<std::string> want;
        want.push_back(pass < 0 ? "gather_rows" : "cast");   /* the trunk's h or the head's h' */
        for (const char* s : { "embed_lookup", "cast", "gemm_nt" }) want.push_back(s);
        for (int l = 0; l < 4; ++l)
            for (const char* s : { "rmsnorm", "gemm_nt", "rmsnorm", "rope", "attn_paged", "gemm_nt",
                                   "rmsnorm", "add", "rmsnorm", "gemm_nt", "gelu_mul", "gemm_nt",
                                   "rmsnorm", "add", "scale" })
                want.push_back(s);
        for (const char* s : { "rmsnorm", "logits_gemm", "gemm_nt" }) want.push_back(s);
        CHECK_EQ(join(issued_ops(b, c)), join(want));

        /* Attention reads the trunk's cache: head layers 0..2 the last sliding trunk layer (10),
         * layer 3 the last full one (11). Groups: 1 sliding, 2 full. */
        int seen = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "attn_paged") continue;
            REQUIRE(i.opd.size() > 1);
            CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_KV);
            CHECK_EQ((int)i.opd[1].handle, seen < 3 ? 1 : 2);
            CHECK_EQ((long long)i.opd[1].offset, seen < 3 ? 10LL : 11LL);
            CHECK_EQ(i.n, 1);
            ++seen;
        }
        CHECK_EQ(seen, 4);
    }
}

TEST(gemma4_head_at_two_ranks_reduces_before_each_post_norm) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx(0, 2);
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const RecWeight* wq0 = find_w(b, "mtp.blk.0.attn_q.weight");
    REQUIRE(wq0 != nullptr);
    CHECK_EQ(wq0->d.shape[0], 16 * 256);
    const RecWeight* lm = find_w(b, "mtp.output.weight");
    REQUIRE(lm != nullptr);
    CHECK_EQ(lm->d.shape[0], 131072);

    RadBatch batch = head_batch(-1);
    RadCtx c{ .b = &b, .batch = &batch, .world = 2 };
    gemma4_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());
    const std::vector<std::string> got = issued_ops(b, c);
    std::vector<std::string> want = { "gather_rows", "embed_lookup", "all_reduce", "cast", "gemm_nt" };
    for (int l = 0; l < 4; ++l)
        for (const char* s : { "rmsnorm", "gemm_nt", "rmsnorm", "rope", "attn_paged", "gemm_nt",
                               "all_reduce", "rmsnorm", "add", "rmsnorm", "gemm_nt", "gelu_mul",
                               "gemm_nt", "all_reduce", "rmsnorm", "add", "scale" })
            want.push_back(s);
    for (const char* s : { "rmsnorm", "logits_gemm", "gemm_nt" }) want.push_back(s);
    CHECK_EQ(join(got), join(want));
}

TEST(gemma4_head_refuses_a_history_pass_shaped_for_a_cache_of_its_own) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    RadBatch batch = head_batch(-1);
    batch.n_draft_out = 1;                 /* a history pass with non-drafting rows */
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4_bf16::step(&c, &batch);
    CHECK(!c.step_fail.empty());
    CHECK(c.issues.empty());
}

TEST(gemma4_probe_names_the_heads_operating_depth) {
    RadArchProbe p{};
    RadModelMeta with = gemma4_draft_meta();
    CHECK_OK(gemma4_bf16::probe(&with, &p));
    CHECK_EQ(p.draft_depth, gemma4_bf16::kHeadDepth);
    CHECK_EQ(p.draft_depth_fixed, 0);
    RadArchProbe q{};
    RadModelMeta without = gemma4_meta();
    CHECK_OK(gemma4_bf16::probe(&without, &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* ===================== gemma4: the 26B-A4B, a dense and a routed arm side by side ===================== */

namespace {

/* gemma-4-26B-A4B-it's text geometry cut to six layers (five sliding then one full, the model's
 * 5:1) with the routed arm on. Keys are the checkpoint's flattened text_config, as arch/gemma4
 * reads them; quant is empty, the container this plugin claims. */
const char* k_g4moe_keys[] = {
    "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim",
    "num_global_key_value_heads", "attention_k_eq_v", "vocab_size", "intermediate_size",
    "rms_norm_eps", "sliding_window", "final_logit_softcapping", "hidden_activation", "layer_types",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.sliding_attention.rope_type",
    "rope_parameters.full_attention.rope_theta", "rope_parameters.full_attention.rope_type",
    "rope_parameters.full_attention.partial_rotary_factor", "enable_moe_block", "num_experts",
    "top_k_experts", "moe_intermediate_size",
};
const char* k_g4moe_vals[] = {
    "2816", "16", "8", "256", "512",
    "2", "true", "262144", "2112",
    "1e-06", "1024", "30.0", "gelu_pytorch_tanh",
    "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
    "\"sliding_attention\", \"full_attention\"]",
    "10000.0", "default",
    "1000000.0", "proportional",
    "0.25", "true", "128",
    "8", "704",
};
static_assert(sizeof(k_g4moe_keys) == sizeof(k_g4moe_vals), "one value a key");

RadModelMeta gemma4_moe_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-26b-a4b";
    m.quant       = "";
    m.n_layers    = 6;
    m.n_embd      = 2816;
    m.n_head      = 16;
    m.n_head_kv   = 8;
    m.head_dim    = 256;
    m.n_ff        = 2112;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.n_kv        = (int)(sizeof(k_g4moe_keys) / sizeof(k_g4moe_keys[0]));
    m.kv_key      = k_g4moe_keys;
    m.kv_val      = k_g4moe_vals;
    return m;
}

int count_resolved(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) if (o.op == name && o.resolved) ++n;
    return n;
}

}  /* namespace */

TEST(a_gemma4_moe_trunk_declares_both_arms_and_the_joins) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    /* five sliding layers read q from the fused row (attn_paged_qrow); the full one attn_paged */
    CHECK_EQ(count_op(b, "attn_paged_qrow"), 5);
    CHECK_EQ(count_op(b, "attn_paged"),      1);
    CHECK_EQ(count_op(b, "kv_store"),   6);
    /* Per layer, one join after attention and one after the two feed-forward arms. */
    CHECK_EQ(count_resolved(b, "norm_add_norm"),  6);
    CHECK_EQ(count_resolved(b, "join_add_scale"), 6);
    /* The routed arm: experts are bf16 where the container has no encoding for them. */
    CHECK(count_op(b, "moe_gemm") > 0);
    CHECK_EQ(count_op(b, "moe_gemm_q"), 0);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);
    CHECK_EQ(count_op(b, "all_reduce"),    0);   /* world 1 */

    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

/* The sliding group is RAD_KV_FULL with the window on the attention op, since a window group is
 * not prefix-cached (arch/gemma4/gemma4.cpp, the KV groups). */
TEST(gemma4_moe_kv_groups_are_both_full_and_the_window_rides_the_op) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    REQUIRE((int)b.kv_groups.size() == 2);
    for (const auto& g : b.kv_groups) {
        CHECK_EQ((int)g.d.kind, RAD_KV_FULL);
        CHECK_EQ(g.d.window, 0);
    }
    const RecOp* sl = find_op(b, "attn_paged_qrow", 0);
    REQUIRE(sl != nullptr);
    CHECK_EQ(pint(sl, "head_dim"), 256);
    CHECK_EQ(pint(sl, "window"),   1024);
    const RecOp* fu = find_op(b, "attn_paged", 0);
    REQUIRE(fu != nullptr);
    CHECK_EQ(pint(fu, "head_dim"), 512);
    CHECK_EQ(pint(fu, "gqa"),      8);
    CHECK_EQ(pint(fu, "window"),   0);
}

/* Proportional RoPE is not a partial rotary: full layers rotate the whole 512-wide head as NeoX
 * pairs (j, j + 256), with frequencies past pair 63 removed by the YaRN ramp. rotary_dim 128 would
 * pair (j, j + 64), which is wrong. */
TEST(gemma4_full_rope_is_the_whole_head_with_the_frequencies_past_64_pairs_cut) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    int full = 0;
    for (const auto& o : b.ops) {
        if (o.op != "rope" || pint(&o, "head_dim") != 512) continue;
        ++full;
        CHECK_EQ(pint(&o, "rotary_dim"), 512);
        CHECK_EQ(pstr(&o, "mode"), std::string("neox"));
        CHECK_NEAR(pflt(&o, "theta"),     1e6, 1.0);
        CHECK_NEAR(pflt(&o, "yarn_low"),  63.5, 1e-9);
        CHECK_NEAR(pflt(&o, "yarn_high"), 64.0, 1e-9);
        CHECK(pflt(&o, "yarn_factor") >= 1e30);
    }
    CHECK(full > 0);
}

/* Where no kernel library serves a join, the plugin declares and issues the rmsnorm / add chain;
 * libref deliberately has no row for the joins so this path stays reachable (docs/OPS.md). */
TEST(gemma4_falls_back_to_the_chain_where_no_join_resolves) {
    std::vector<std::string> fused, chain;
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) { b.refuse.insert("norm_add_norm"); b.refuse.insert("join_add_scale"); }
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        (refuse ? chain : fused) = issued_ops(b, c);
    }
    CHECK_EQ(count_issued(fused, "norm_add_norm"),  6);
    CHECK_EQ(count_issued(fused, "join_add_scale"), 6);
    CHECK_EQ(count_issued(chain, "norm_add_norm"),  0);
    CHECK_EQ(count_issued(chain, "join_add_scale"), 0);
    for (const auto* v : { &fused, &chain }) {
        CHECK_EQ(count_issued(*v, "attn_paged_qrow"), 5);
        CHECK_EQ(count_issued(*v, "attn_paged"),      1);
    }
    /* The chain is longer by the norms and adds the joins fold in. */
    CHECK(count_issued(chain, "rmsnorm") > count_issued(fused, "rmsnorm"));
    CHECK(count_issued(chain, "add") > count_issued(fused, "add"));
}

/* Both Gemma 4 plugins export the same reply-format declaration. */
TEST(both_gemma4_plugins_export_one_reply_format) {
    RadModelMeta moe = gemma4_moe_meta(), dense = gemma4_meta();
    CHECK(gemma4::chat_format(&moe) == rad::arch::gemma4_chat_format());
    CHECK(gemma4_bf16::chat_format(&dense) == rad::arch::gemma4_chat_format());
}

/* ================== gemma4 + DFlash: the drafter's taps, live where they are written ================== */

namespace {

/* The MoE trunk above with z-lab's DFlash drafter merged: five layers (four sliding, one full)
 * reading the trunk at layers 1, 3 and 5. */
std::vector<const char*> g4df_keys() {
    std::vector<const char*> k(std::begin(k_g4moe_keys), std::end(k_g4moe_keys));
    for (const char* s : { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim",
                           "draft.num_attention_heads", "draft.num_key_value_heads",
                           "draft.intermediate_size", "draft.rms_norm_eps", "draft.rope_theta",
                           "draft.hidden_size", "draft.vocab_size", "draft.block_size",
                           "draft.dflash_config.target_layer_ids",
                           "draft.dflash_config.mask_token_id", "draft.layer_types",
                           "draft.sliding_window" })
        k.push_back(s);
    return k;
}
std::vector<const char*> g4df_vals() {
    std::vector<const char*> v(std::begin(k_g4moe_vals), std::end(k_g4moe_vals));
    for (const char* s : { "[\"DFlashDraftModel\"]", "5", "128", "32", "8", "5632", "1e-06",
                           "1000000.0", "2816", "262144", "16", "[1, 3, 5]", "4",
                           "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                           "\"sliding_attention\", \"full_attention\"]",
                           "2048" })
        v.push_back(s);
    return v;
}

int buf_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == name) return (int)i + 1;
    return 0;
}

}  /* namespace */

/* DFlash proposes a block in one pass, so the probe's depth is the trained block size minus the
 * anchor, and fixed. Without a drafter, or with DFlash2 (not served by this plugin), it is 0. */
TEST(gemma4_probe_names_the_dflash_blocks_depth) {
    std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    RadModelMeta meta = gemma4_moe_meta();
    meta.n_kv   = (int)keys.size();
    meta.kv_key = keys.data();
    meta.kv_val = vals.data();
    RadArchProbe p{};
    CHECK_OK(gemma4::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 15);
    CHECK_EQ(p.draft_depth_fixed, 1);

    for (size_t i = 0; i < keys.size(); ++i)
        if (!std::strcmp(keys[i], "draft.architectures")) vals[i] = "[\"DFlash2DraftModel\"]";
    meta.kv_val = vals.data();
    RadArchProbe q{};
    CHECK_OK(gemma4::probe(&meta, &q));
    CHECK_EQ(q.draft_depth, 0);

    RadModelMeta plain = gemma4_moe_meta();
    RadArchProbe r{};
    CHECK_OK(gemma4::probe(&plain, &r));
    CHECK_EQ(r.draft_depth, 0);
}

/* Each tap is declared where it is issued. The buffer planner takes liveness from declaration
 * order, so `dflash_src` is live from its first declared write; a tap declared after the trunk but
 * issued inside it would let the plan alias it with `moe_edn`, which every MoE layer writes,
 * silently corrupting what the drafter reads. The two live ranges must overlap. */
TEST(gemma4_dflash_taps_are_live_from_the_layer_that_writes_them) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    RadModelMeta meta = gemma4_moe_meta();
    meta.n_kv   = (int)keys.size();
    meta.kv_key = keys.data();
    meta.kv_val = vals.data();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    const int src = buf_named(b, "dflash_src"), edn = buf_named(b, "moe_edn");
    REQUIRE(src > 0 && edn > 0);
    int first_src = -1, last_edn = -1;
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (rad_buf w : b.ops[i].writes) {
            if ((int)w == src && first_src < 0) first_src = (int)i;
            if ((int)w == edn) last_edn = (int)i;
        }
    REQUIRE(first_src >= 0 && last_edn >= 0);
    CHECK(first_src < last_edn);

    /* A trunk step writes each tap once: three copies into `dflash_src`. */
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    int taps = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && std::find(o->writes.begin(), o->writes.end(), (rad_buf)src) != o->writes.end())
            ++taps;
    }
    CHECK_EQ(taps, 3);
}

/* Sliding attention reads q in place in the fused projection row where attn_paged_qrow is served,
 * and lifts q into its own buffer for attn_paged where it is not. */
TEST(gemma4_sliding_attention_reads_q_in_the_fused_row) {
    /* Served: attn_paged_qrow reads the slice and nothing is lifted. Refused: q is lifted into q_s
     * for attn_paged. attn_paged never gets the slice, since its kernel may read q only dense. */
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) b.refuse.insert("attn_paged_qrow");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        const int qkv = buf_named(b, "qkv_s"), qs = buf_named(b, "q_s");
        REQUIRE(qkv > 0);
        CHECK_EQ(qs > 0, refuse == 1);
        auto reads = [](const RecOp& o, int buf) {
            return buf > 0 && std::find(o.reads.begin(), o.reads.end(), (rad_buf)buf) != o.reads.end();
        };
        int qrow = 0, paged_slice = 0, paged_lifted = 0;
        for (const auto& o : b.ops) {
            if (!o.resolved) continue;
            if (o.op == "attn_paged_qrow" && reads(o, qkv)) ++qrow;
            if (o.op == "attn_paged" && reads(o, qkv)) ++paged_slice;
            if (o.op == "attn_paged" && reads(o, qs)) ++paged_lifted;
        }
        const int sliding = refuse ? paged_lifted : qrow;
        CHECK(sliding > 0);
        CHECK_EQ(refuse ? qrow : paged_lifted, 0);
        CHECK_EQ(paged_slice, 0);
        CHECK_EQ(count_op(b, "cast"), refuse ? sliding : 0);   /* the lifts (no drafter taps) */

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "attn_paged_qrow"), refuse ? 0 : sliding);
        CHECK_EQ(count_issued(issued, "cast"), refuse ? sliding : 0);
        for (const auto& o : b.ops)
            if (o.resolved) CHECK(o.uses_declared);
    }
}

/* The feed-forward join writes the next layer's input norm. With joins served, the only rmsnorm
 * from residual `x` into `h` is layer 0's, the last join produces the final norm, and every join
 * writes `h`. Refused, the chain declares them all. */
TEST(gemma4_feed_forward_join_writes_the_next_input_norm) {
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) { b.refuse.insert("norm_add_norm"); b.refuse.insert("join_add_scale"); }
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        const int x = buf_named(b, "x"), h = buf_named(b, "h");
        REQUIRE(x > 0 && h > 0);
        int x_to_h = 0, joins_to_h = 0;
        for (const auto& o : b.ops) {
            if (!o.resolved) continue;
            const bool rx = std::find(o.reads.begin(), o.reads.end(), (rad_buf)x) != o.reads.end();
            const bool wh = std::find(o.writes.begin(), o.writes.end(), (rad_buf)h) != o.writes.end();
            if (o.op == "rmsnorm" && rx && wh) ++x_to_h;
            if (o.op == "join_add_scale" && wh) ++joins_to_h;
        }
        /* fused: layer 0's input norm. Chained: six input norms, six pre-feedforward norms and the
         * final norm. */
        CHECK_EQ(x_to_h, refuse ? 13 : 1);
        CHECK_EQ(joins_to_h, refuse ? 0 : 6);

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "join_add_scale"), refuse ? 0 : 6);
        CHECK_EQ(count_issued(issued, "logits_gemm"), 1);
    }
}

/* At decode heights the router GEMM, top-k and sort are one launch (libr4d's
 * router_gemm_topk_scatter); if refused, router_topk_scatter follows the router GEMM; if both are
 * refused, three separate ops. The routing buffers are the same in all cases. */
TEST(gemma4_routes_a_decode_step_in_one_launch) {
    for (int refuse = 0; refuse < 3; ++refuse) {
        RadBuilder b;
        if (refuse >= 1) b.refuse.insert("router_gemm_topk_scatter");
        if (refuse >= 2) b.refuse.insert("router_topk_scatter");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "router_gemm_topk_scatter"), refuse == 0 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "router_topk_scatter"), refuse == 1 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "router_topk"), refuse == 2 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "moe_scatter"), refuse == 2 ? 6 : 0);
        /* the per-expert scale and gather run whichever routing form was used */
        CHECK_EQ(count_issued(issued, "expert_scale"), 6);
        CHECK_EQ(count_issued(issued, "moe_gather"), 6);
    }
}

/* The per-expert scale is one launch where `expert_scale` is served, else a table lookup and a
 * multiply over the same buffers. */
TEST(gemma4_scales_the_routing_weights_in_one_launch) {
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) b.refuse.insert("expert_scale");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "expert_scale"), refuse ? 0 : 6);
        CHECK_EQ(count_issued(issued, "embed_lookup"), refuse ? 1 + 6 : 1);   /* 1: the token embedding */
        CHECK_EQ(count_issued(issued, "mul"), refuse ? 6 : 0);
        for (const auto& o : b.ops)
            if (o.resolved) CHECK(o.uses_declared);
    }
}

/* The drafter's head reads the container's block-fp8 copy of the target's table
 * (`dflash.draft_head.weight`) through quant_act_fp8 and gemm_nt_q where present, else the bf16
 * table through gemm_nt. The copy is only declared when present, so older containers are served
 * unchanged. */
TEST(gemma4_dflash_head_reads_the_fp8_copy_where_the_container_holds_one) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    for (int fp8 = 0; fp8 < 2; ++fp8) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuilder b;
        if (fp8)
            b.encs = { { "dflash.draft_head", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) } };
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK_EQ(find_w(b, "dflash.draft_head.weight") != nullptr, fp8 == 1);

        const int dlog = buf_named(b, "df_dlog"), selh = buf_named(b, "df_selh");
        REQUIRE(dlog > 0 && selh > 0);
        const RecOp* head = nullptr;
        for (const auto& o : b.ops)
            if (o.resolved &&
                std::find(o.writes.begin(), o.writes.end(), (rad_buf)dlog) != o.writes.end())
                head = &o;
        REQUIRE(head != nullptr);
        CHECK_EQ(head->op, std::string(fp8 ? "gemm_nt_q" : "gemm_nt"));
        const bool reads_selh =
            std::find(head->reads.begin(), head->reads.end(), (rad_buf)selh) != head->reads.end();
        CHECK_EQ(reads_selh, fp8 == 0);

        /* a draft pass issues it */
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        int head_issues = 0;
        for (const auto& i : c.issues)
            if (b.op_of(i.op) == head) ++head_issues;
        CHECK_EQ(head_issues, 1);
    }
}

/* The drafter fuses residual add, norm and quantiser into `rmsnorm_quant_fp8`: ten per draft pass
 * over five layers (each layer's two inputs; layer 0's attention input takes x as the embedding
 * left it), leaving one `add`, the last layer's, before the final norm. Refused: two adds and two
 * quantisers of x per layer. */
TEST(gemma4_dflash_folds_the_residual_add_norm_and_quantiser_into_one_launch) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuilder b;
        if (refuse) b.refuse.insert("rmsnorm_quant_fp8");
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));

        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "rmsnorm_quant_fp8"), refuse ? 0 : 10);
        CHECK_EQ(count_issued(ops, "add"), refuse ? 10 : 1);
        /* chain: x quantised twice per layer; attention output quantised once per layer always */
        CHECK_EQ(count_issued(ops, "quant_act_fp8"), refuse ? 15 : 5);
        if (refuse) continue;

        /* the delta then the residual x past layer 0; layer 0 reads x and adds nothing */
        const uint32_t x = (uint32_t)buf_named(b, "df_x"), h = (uint32_t)buf_named(b, "df_h");
        int k = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "rmsnorm_quant_fp8") continue;
            REQUIRE(i.opd.size() == 6);
            if (k++ == 0) {
                CHECK_EQ(i.opd[0].handle, x);
                CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_NONE);
            } else {
                CHECK_EQ(i.opd[0].handle, h);
                CHECK_EQ(i.opd[1].handle, x);
            }
        }
    }
}

/* ============ gemma4: the prefill overlap -- two slices, the reductions on lane 1 ============ */

namespace {

/* The elements [lo, hi) of its buffer an operand can reach, in whole rows at the declared width.
 * False for a non-buffer operand. */
bool reach(const RadBuilder& b, const RadOperand& o, rad_buf* buf, int64_t* lo, int64_t* hi) {
    if (o.kind != RAD_OPK_BUF || o.handle == 0 || o.handle > b.bufs.size()) return false;
    const RadBufDecl& d = b.bufs[o.handle - 1].d;
    int64_t pitch = 1;
    for (uint32_t i = 1; i < d.rank; ++i) pitch *= d.shape[i];
    *buf = o.handle;
    *lo  = o.offset - o.offset % pitch;
    *hi  = *lo + (o.rows > 0 ? o.rows : d.shape[0]) * pitch;
    return true;
}

/* Whether issue a runs before issue b (a < b): on one lane by issue order; across lanes only via a
 * join from a's lane to b's issued between them. */
bool ordered(const RadCtx& c, size_t a, size_t b) {
    const int la = c.issues[a].lane, lb = c.issues[b].lane;
    if (la == lb) return true;
    for (const RecJoin& j : c.joins)
        if (j.from == la && j.to == lb && j.at > a && j.at <= b) return true;
    return false;
}

/* Pairs of unordered issues on different lanes that reach a common element. Every operand counts
 * as read and written, so this can only over-count. */
int races(const RadBuilder& b, const RadCtx& c) {
    int n = 0;
    for (size_t i = 0; i < c.issues.size(); ++i)
        for (size_t j = i + 1; j < c.issues.size(); ++j) {
            if (ordered(c, i, j)) continue;
            bool meet = false;
            for (const RadOperand& p : c.issues[i].opd)
                for (const RadOperand& q : c.issues[j].opd) {
                    rad_buf bp = 0, bq = 0;
                    int64_t pl = 0, ph = 0, ql = 0, qh = 0;
                    if (reach(b, p, &bp, &pl, &ph) && reach(b, q, &bq, &ql, &qh) && bp == bq &&
                        pl < qh && ql < ph)
                        meet = true;
                }
            n += meet ? 1 : 0;
        }
    return n;
}

/* One prompt of T tokens, sampling its last. */
RadBatch one_prompt(int64_t T, const int32_t* out) {
    RadBatch b{};
    b.phase       = RAD_PHASE_PREFILL;
    b.n_tok       = T;
    b.n_seq       = 1;
    b.max_q_len   = T;
    b.max_ctx_len = T;
    b.n_out       = 1;
    b.out_ids     = out;
    return b;
}

/* Rows issued, summed by op: a slice's n is its rows; RAD_N_BATCH counts as T. */
std::map<std::string, int64_t> rows_by_op(const RadBuilder& b, const RadCtx& c, int64_t T) {
    std::map<std::string, int64_t> w;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        w[o ? o->op : "<null>"] += i.n < 0 ? T : i.n;
    }
    return w;
}

/* The n of each issue of one op, in issue order. */
std::vector<int64_t> ns_of(const RadBuilder& b, const RadCtx& c, const char* op) {
    std::vector<int64_t> v;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == op) v.push_back(i.n);
    }
    return v;
}

bool noted(const RadBuilder& b, const char* text) {
    for (const auto& n : b.notes)
        if (n.find(text) != std::string::npos) return true;
    return false;
}

}  /* namespace */

/* A prefill chunk in two slices (arch/gemma4, trunk_overlap). Checked: every reduction runs on
 * lane 1 after its producer and before its consumer, and nothing else does; the cut is a whole
 * number of routed passes; and the slices issue exactly the whole step's rows (same routed
 * passes, same drafter taps), so the output matches. A chunk under two passes, and a decode step,
 * run whole: reductions on lane 0 and the dense arm on lane 1 beside the routed arm, all ordered. */
TEST(a_gemma4_prefill_chunk_runs_as_two_slices_with_its_reductions_on_the_second_lane) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    const int64_t E = 2816, L = 6;
    auto model = [&](RadBuilder& b, bool whole) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok  = 8192;
        ctx.max_spec = 15;
        /* The lossy wire runs every chunk whole, since a slice is not shown to get the same bits
         * as the whole step. */
        if (whole) ctx.tp_wire_lossy = 1;
        return gemma4::declare(&b, &meta, &ctx);
    };
    auto run = [&](RadBuilder& b, RadCtx& c, RadBatch& batch) {
        c = RadCtx{ .b = &b, .batch = &batch, .world = 4 };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
    };

    struct Case { int64_t T, cut; };
    for (const Case k : { Case{8192, 4096}, Case{6144, 4096}, Case{6143, 2048}, Case{4096, 2048},
                          Case{4095, 0} }) {
        const int32_t out[1] = { (int32_t)(k.T - 1) };
        RadBuilder b;
        CHECK_OK(model(b, false));
        CHECK(noted(b, "prefill overlap: a chunk of two routed passes"));
        CHECK_EQ((int)b.concurrent.size(), 5);
        for (const char* name : { "o", "ff2", "h", "gate_up", "ffn" })
            CHECK(std::count(b.concurrent.begin(), b.concurrent.end(),
                             (rad_buf)buf_named(b, name)) == 1);
        RadBatch batch = one_prompt(k.T, out);
        RadCtx c;
        run(b, c, batch);
        CHECK(c.step_fail.empty());
        CHECK_EQ(c.lane, 0);
        CHECK_EQ(races(b, c), 0);

        std::vector<int64_t> lane1;   /* reductions on lane 1 */
        int dense1 = 0;               /* dense-arm launches on lane 1 */
        for (const auto& i : c.issues) {
            if (i.lane != 1) continue;
            const RecOp* o = b.op_of(i.op);
            REQUIRE(o != nullptr);
            if (o->op == "all_reduce") lane1.push_back(i.n);
            else ++dense1;
        }
        if (k.cut == 0) {
            CHECK(lane1.empty());
            CHECK_EQ(dense1, 3 * (int)L);    /* gate_up, act_mul, down beside the routed arm */
        } else {
            CHECK_EQ(dense1, 0);
            REQUIRE(lane1.size() == (size_t)(4 * L));
            for (int64_t l = 0; l < L; ++l) {
                CHECK_EQ(lane1[4 * l + 0], k.cut * E);               /* attention, slice A */
                CHECK_EQ(lane1[4 * l + 1], (k.T - k.cut) * E);       /* attention, slice B */
                CHECK_EQ(lane1[4 * l + 2], 2 * k.cut * E);           /* both arms, slice A */
                CHECK_EQ(lane1[4 * l + 3], 2 * (k.T - k.cut) * E);   /* both arms, slice B */
            }
        }

        /* The whole step on the lossy wire: the same rows by op and routed passes, reductions on
         * lane 0, only the dense arm on lane 1. */
        RadBuilder w;
        CHECK_OK(model(w, true));
        CHECK(noted(w, "prefill overlap: off (the lossy all-reduce wire)"));
        CHECK(std::count(w.concurrent.begin(), w.concurrent.end(), (rad_buf)buf_named(w, "o")) == 0);
        RadBatch wbatch = one_prompt(k.T, out);
        RadCtx wc;
        run(w, wc, wbatch);
        CHECK(wc.step_fail.empty());
        CHECK_EQ(races(w, wc), 0);
        for (const auto& i : wc.issues) {
            const RecOp* o = w.op_of(i.op);
            if (i.lane == 1) CHECK(o == nullptr || o->op != "all_reduce");
        }
        CHECK(rows_by_op(b, c, k.T) == rows_by_op(w, wc, k.T));
        CHECK(ns_of(b, c, "moe_gather") == ns_of(w, wc, "moe_gather"));
        CHECK_EQ((int)ns_of(b, c, "cast").size(), k.cut ? 6 : 3);   /* three taps, per slice */
    }

    /* A decode step: reductions on lane 0, the dense arm on lane 1 joined before the feed-forward
     * reduction, all ordered. */
    RadBuilder b;
    CHECK_OK(model(b, false));
    CHECK(noted(b, "the two arms: in a step that runs whole"));
    RadBatch d = decode_batch();
    RadCtx c;
    run(b, c, d);
    CHECK(c.step_fail.empty());
    CHECK_EQ(c.lane, 0);
    CHECK_EQ(races(b, c), 0);
    int dense1 = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        REQUIRE(o != nullptr);
        if (o->op == "all_reduce") CHECK_EQ(i.lane, 0);
        if (i.lane == 1) {
            ++dense1;
            CHECK(o->op == "gemm_nt" || o->op == "act_mul");
        }
    }
    CHECK_EQ(dense1, 3 * (int)L);
    CHECK_EQ((int)c.joins.size(), 2 * (int)L);
}

/* Where no loaded library declares the fused ops (CPU-only, or no libr4d), asking for one would
 * stop the engine, so the plugin checks rad_op_declared first and declares the chains: rmsnorm /
 * add joins, embed_lookup and mul for the expert scale, and q lifted for attn_paged. */
TEST(gemma4_without_the_fused_ops_declares_the_chains_and_asks_for_no_undeclared_op) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", fp8 } };
    for (const char* op : { "norm_add_norm", "norm_add_norm_route", "join_add_scale",
                            "expert_scale", "attn_paged_qrow" })
        b.undeclared.insert(op);
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));
    CHECK(b.asked_undeclared.empty());
    CHECK(noted(b, "sandwich joins: 0 of 12 fused"));
    CHECK(noted(b, "routed inputs: 0 of 6"));
    CHECK(noted(b, "sliding attention: q lifted"));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    const std::vector<std::string> ops = issued_ops(b, c);
    CHECK_EQ(count_issued(ops, "rmsnorm_quant_fp8"), 6);
    CHECK_EQ(count_issued(ops, "cast"), 5);
    CHECK(count_issued(ops, "add") > 0);
}

/* On a layer whose gate_up is block fp8 the attention join (norm_add_norm_route) also writes the
 * routed arm's inputs: the router's norm into moe_rn and the experts' codes into moe_hq / moe_hqs,
 * so neither the router's rmsnorm nor rmsnorm_quant_fp8 is issued. Refused: norm_add_norm plus
 * those two launches. A tp-4 prefill through the overlap keeps every reduction ordered too. */
TEST(gemma4_attention_join_writes_the_routed_inputs_on_fp8_experts) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    std::map<std::string, int> issued[2];
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        b.encs = { { "ffn_gate_up_exps", fp8 } };
        if (refuse) b.refuse.insert("norm_add_norm_route");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK_EQ(count_resolved(b, "norm_add_norm_route"), refuse ? 0 : 6);
        CHECK_EQ(count_resolved(b, "norm_add_norm"), refuse ? 6 : 0);
        CHECK(noted(b, refuse ? "routed inputs: 0 of 6" : "routed inputs: 6 of 6"));
        if (!refuse) {
            const RecOp* r = find_op(b, "norm_add_norm_route");
            REQUIRE(r != nullptr);
            for (const char* name : { "moe_rn", "moe_hq", "moe_hqs", "x", "h" })
                CHECK(std::count(r->writes.begin(), r->writes.end(),
                                 (rad_buf)buf_named(b, name)) == 1);
        }

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        for (const auto& n : issued_ops(b, c)) ++issued[refuse][n];
    }
    CHECK_EQ(issued[0]["norm_add_norm_route"], 6);
    CHECK_EQ(issued[0]["norm_add_norm"], 0);
    CHECK_EQ(issued[0]["rmsnorm_quant_fp8"], 0);
    CHECK_EQ(issued[1]["norm_add_norm_route"], 0);
    CHECK_EQ(issued[1]["norm_add_norm"], 6);
    CHECK_EQ(issued[1]["rmsnorm_quant_fp8"], 6);
    CHECK_EQ(issued[1]["rmsnorm"] - issued[0]["rmsnorm"], 6);   /* the router's input norm */
    CHECK_EQ(issued[0]["moe_gemm_q"], 6);

    /* tp 4, an 8192-token chunk in two slices */
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", fp8 } };
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx(0, 4);
    ctx.max_tok = 8192;
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));
    CHECK(noted(b, "prefill overlap: a chunk of two routed passes"));
    const int32_t out[1] = { 8191 };
    RadBatch batch = one_prompt(8192, out);
    RadCtx c{ .b = &b, .batch = &batch, .world = 4 };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    CHECK_EQ(races(b, c), 0);
    CHECK_EQ(count_issued(issued_ops(b, c), "norm_add_norm_route"), 12);
    CHECK_EQ(count_issued(issued_ops(b, c), "rmsnorm_quant_fp8"), 0);
}

/* At four ranks the junctions carry their reductions at decode heights, in the same issue order as
 * at two ranks (only the embedding's and the last layer's feed-forward reduction stand alone),
 * through ar_rmsnorm_quant_i8 declared at world_size 4 up to 16 rows (kAr4FoldRows: a DFlash
 * verify's rows, which libr4d folds in its tail form, inside its one-shot band of 524288 / 5376 =
 * 97). A taller step issues each reduction and its junction apart, and where no library declares
 * the fold every step issues the pair. */
TEST(a_gemma4_w4a8_step_at_four_ranks_folds_the_all_reduce_into_the_junction_at_decode_heights) {
    const int L = 12;
    std::vector<std::string> folded = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    std::vector<std::string> apart  = folded;
    for (int l = 0; l < L; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) { folded.push_back("gemm_nt_q"); apart.push_back("gemm_nt_q"); }
        for (const char* s : { "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "ar_rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            folded.push_back(s);
        for (const char* s : { "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "all_reduce", "rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8",
                               "gemm_nt_q", "all_reduce" })
            apart.push_back(s);
        if (l < L - 1) {
            folded.push_back("ar_rmsnorm_quant_i8");
            apart.push_back("rmsnorm_quant_i8");
        } else {
            for (const char* s : { "all_reduce", "rmsnorm", "add", "scale" }) folded.push_back(s);
            for (const char* s : { "rmsnorm", "add", "scale" }) apart.push_back(s);
        }
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" }) {
        folded.push_back(s);
        apart.push_back(s);
    }
    /* a decode step folds */
    CHECK_EQ(join(gemma4_w4_step({}, 4)), join(folded));

    auto declare4 = [](RadBuilder& b) {
        const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
        b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
                   { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
        RadModelMeta meta = gemma4_meta();
        RadBuildCtx  ctx  = build_ctx(0, 4);
        return gemma4_bf16::declare(&b, &meta, &ctx);
    };
    RadBuilder b;
    CHECK_OK(declare4(b));
    CHECK(noted(b, "folded in at steps of 16 rows or fewer"));
    auto fold_hi = [](const RadBuilder& bb, int* n) {
        int64_t hi = -1;
        *n = 0;
        for (const auto& o : bb.ops) {
            if (o.op != "ar_rmsnorm_quant_i8") continue;
            ++*n;
            CHECK_EQ(pint(&o, "world_size"), 4);
            for (const auto& p : o.p)
                if (p.key == "M") hi = p.ihi;
        }
        return hi;
    };
    int n_ar = 0;
    CHECK_EQ(fold_hi(b, &n_ar), 16);
    CHECK_EQ(n_ar, 2 * L - 1);

    /* a verify's height folds (libr4d's tail form) and a step past 16 rows issues the pair */
    const int32_t out[1] = { 127 };
    for (int rows : { 16, 128 }) {
        RadBatch tall = one_prompt(rows, out);
        RadCtx c; c.b = &b; c.rank = 0; c.world = 4; c.batch = &tall;
        gemma4_bf16::step(&c, &tall);
        CHECK(c.step_fail.empty());
        CHECK_EQ(join(issued_ops(b, c)), join(rows <= 16 ? folded : apart));
    }


    /* no library declares the fold: none declared, every step issues the pair */
    RadBuilder off;
    off.undeclared.insert("ar_rmsnorm_quant_i8");
    CHECK_OK(declare4(off));
    CHECK(noted(off, "issued apart"));
    for (const auto& o : off.ops) CHECK(o.op != "ar_rmsnorm_quant_i8");
}

/* ============ gemma4 31B + DFlash: z-lab's block drafter on the w4a8 / w8a8 trunk ============ */

namespace {

/* The twelve-layer 31B above with z-lab/gemma-4-31B-it-DFlash merged under `draft.` as
 * `rad-convert --draft-model` writes it: its own geometry (5376 wide, 64 heads / 8 kv x 128, ffn
 * 10752, five layers -- four sliding over 2048, one full -- at block 16), reading the trunk at three
 * of the fixture's twelve layers where the real one reads six of sixty. */
std::vector<const char*> g431df_keys() {
    std::vector<const char*> k(std::begin(k_gem4_keys), std::end(k_gem4_keys));
    for (const char* s : { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim",
                           "draft.num_attention_heads", "draft.num_key_value_heads",
                           "draft.intermediate_size", "draft.rms_norm_eps", "draft.rope_theta",
                           "draft.hidden_size", "draft.vocab_size", "draft.block_size",
                           "draft.dflash_config.target_layer_ids",
                           "draft.dflash_config.mask_token_id", "draft.layer_types",
                           "draft.sliding_window" })
        k.push_back(s);
    return k;
}
std::vector<const char*> g431df_vals() {
    std::vector<const char*> v(std::begin(k_gem4_vals), std::end(k_gem4_vals));
    for (const char* s : { "[\"DFlashDraftModel\"]", "5", "128", "64", "8", "10752", "1e-06",
                           "1000000", "5376", "262144", "16", "[1, 5, 9]", "4",
                           "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                           "\"sliding_attention\", \"full_attention\"]",
                           "2048" })
        v.push_back(s);
    return v;
}

struct G431Df {
    std::vector<const char*> keys = g431df_keys(), vals = g431df_vals();
    RadModelMeta meta() const {
        RadModelMeta m = gemma4_meta();
        m.n_kv   = (int)keys.size();
        m.kv_key = keys.data();
        m.kv_val = vals.data();
        return m;
    }
};

/* The trunk at w4a8 as the -df recipe makes it, the drafter's linears on one 8-bit grid -- the
 * E4M3 one (block fp8) or the int8 one (w8a8) -- and, with `head`, the table's 8-bit copy in the
 * same grid. The drafter's keys come FIRST: the fake answers the first key a name contains, and
 * `attn_q.` is in `dflash.blk.0.attn_q.weight` too. With `fused`, the trunk's q, k and v are one
 * linear, as a recipe naming `blk.*.attn_qkv.weight` converts them. */
void g431df_encs(RadBuilder& b, bool i8, bool head, bool fused = false) {
    const RadEncoding w4  = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    const RadEncoding q8  = rad_enc_affine(RAD_I8, RAD_F16, 1, 128);
    const RadEncoding& d  = i8 ? q8 : fp8;
    b.encs = { { "dflash.fc.", d }, { "dflash.blk.", d } };
    if (head) b.encs.push_back({ "dflash.draft_head.", d });
    if (fused) b.encs.push_back({ "attn_qkv.", w4 });
    else
        for (const char* s : { "attn_q.", "attn_k.", "attn_v." }) b.encs.push_back({ s, w4 });
    for (const char* s : { "attn_o.", "ffn_gate_up.", "ffn_down." })
        b.encs.push_back({ s, w4 });
}

int count_writes_to(const RadBuilder& b, const RadCtx& c, int buf) {
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && std::find(o->writes.begin(), o->writes.end(), (rad_buf)buf) != o->writes.end())
            ++n;
    }
    return n;
}

rad_kvgroup kv_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.kv_groups.size(); ++i)
        if (b.kv_groups[i].name == name) return (rad_kvgroup)(i + 1);
    return 0;
}

}  /* namespace */

/* The 31B declares z-lab's drafter as a block drafter past its trunk: five layers at 12..16 on a KV
 * group of their own, the trunk's taps in one plane, its embedding read at the reference's scale,
 * and the depth the block was trained at, the only one it serves, which probe names for rad-convert
 * and `--num-speculative-tokens auto`. Without speculation none of it is declared. */
TEST(gemma4_31b_declares_dflash_as_a_block_drafter_past_its_trunk) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (const bool served : { true, false }) {
    RadBuilder b;
    g431df_encs(b, false, false);
    if (!served) b.refuse.insert("attn_paged_qrow");
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.drafter_name, std::string("dflash"));
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 15);
    CHECK_EQ((int)b.drafter.block, 16);
    CHECK_EQ(b.drafter.mask_token, 4);
    CHECK(!b.drafter_shared);
    CHECK_EQ(b.drafter.proposal, (rad_buf)buf_named(b, "dflash_tokens"));

    /* its own cache, the five layers past the trunk's twelve bound to it */
    const rad_kvgroup kvd = kv_named(b, "kv_draft");
    REQUIRE(kvd != 0);
    CHECK_EQ((int)b.kv_groups[kvd - 1].d.kind, (int)RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[kvd - 1].d.head_dim, 128);
    /* 64 heads over 8 kv heads is gqa 8. Where a card serves it under attn_paged_qrow (libr4d's
     * head-128 kernel with two query groups a kv head), the cache holds each kv head once and
     * attention runs at gqa 8. Where not, libr4d's head-128 attention takes 4 queries per kv head
     * at a block of 16: the cache holds each kv head twice, the projections stay 8 heads wide, and
     * every attention and cache write of the drafter sees 16 kv heads at gqa 4. */
    const int64_t heads = served ? 8 : 16;
    CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, heads);
    CHECK_EQ(b.bufs[buf_named(b, "df_k") - 1].d.shape[1], 8 * 128);
    if (served) CHECK_EQ(buf_named(b, "df_k4"), 0);
    else        CHECK_EQ(b.bufs[buf_named(b, "df_k4") - 1].d.shape[1], 16 * 128);
    {
        int n_attn = 0, n_store = 0;
        for (const auto& o : b.ops) {
            /* the asking ops (qrow_serves) name no buffer */
            if (o.op == (served ? "attn_paged_qrow" : "attn_paged") && pint(&o, "head_dim") == 128 &&
                !o.reads.empty()) {
                ++n_attn;
                CHECK_EQ(pint(&o, "gqa"), served ? 8 : 4);
                CHECK_EQ(pint(&o, "n_head"), 64);
            }
            if (o.op == "kv_store" && pint(&o, "head_dim") == 128) {
                ++n_store;
                CHECK_EQ(pint(&o, "n_head_kv"), heads);
            }
        }
        CHECK_EQ(n_attn, 2);    /* the sliding layers' and the full layer's */
        CHECK_EQ(n_store, 2);   /* the context pass's and the query pass's */
    }
    std::vector<int> bound;
    for (const auto& [layer, g] : b.binds) if (g == kvd) bound.push_back(layer);
    CHECK(bound == (std::vector<int>{ 12, 13, 14, 15, 16 }));

    /* three taps of 5376 a row */
    const int src = buf_named(b, "dflash_src");
    REQUIRE(src > 0);
    CHECK_EQ(b.bufs[src - 1].d.shape[1], 3 * 5376);

    /* the drafter embeds the anchor and the masks through the TRUNK's table at 73.5 */
    const RecOp* te = find_op(b, "embed_lookup", 0);
    const RecOp* de = find_op(b, "embed_lookup", 1);
    REQUIRE(te != nullptr && de != nullptr);
    CHECK(te->w == de->w);
    CHECK_NEAR(pflt(de, "mult"), 73.5, 1e-9);

    /* one name map a declared weight: a COPY map repeated is refused by the real builder */
    std::map<std::string, int> maps;
    for (const auto& mp : b.maps) if (mp.mode == RAD_MAP_COPY) ++maps[mp.declared];
    for (const auto& [name, n] : maps) CHECK_EQ(name + " x" + std::to_string(n), name + " x1");

    RadArchProbe p{};
    CHECK_OK(gemma4_bf16::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 15);
    CHECK_EQ(p.draft_depth_fixed, 1);

    /* no speculation: no drafter, no taps */
    RadBuilder b0;
    g431df_encs(b0, false, false);
    RadBuildCtx c0 = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b0, &meta, &c0));
    CHECK_EQ(buf_named(b0, "dflash_src"), 0);
    CHECK(b0.drafter_name.empty());

    /* and a depth the block was not trained at is refused by name */
    RadBuilder b4;
    g431df_encs(b4, false, false);
    RadBuildCtx c4 = build_ctx();
    c4.max_spec = 4;
    CHECK_EQ(gemma4_bf16::declare(&b4, &meta, &c4), RAD_E_UNSUPPORTED);
    }
}

/* The taps on the w4a8 trunk are declared in the layer they copy, so `dflash_src` is live from its
 * first write on (the arch/gemma4 trap that cost 98% of the drafts). Each is issued once its
 * layer's output is whole, after the junction that adds the feed-forward, applies the layer scalar
 * and quantises the next layer's input, so the copy is HF's hidden_states[l + 1]. */
TEST(gemma4_31b_dflash_taps_the_stream_the_junction_leaves) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    g431df_encs(b, false, false);
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const int src = buf_named(b, "dflash_src"), gu = buf_named(b, "gate_up");
    REQUIRE(src > 0 && gu > 0);
    int first_src = -1, last_gu = -1;
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (rad_buf w : b.ops[i].writes) {
            if ((int)w == src && first_src < 0) first_src = (int)i;
            if ((int)w == gu) last_gu = (int)i;
        }
    REQUIRE(first_src >= 0 && last_gu >= 0);
    CHECK(first_src < last_gu);

    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch batch = decode_batch();
    c.batch = &batch;
    gemma4_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());
    CHECK_EQ(count_writes_to(b, c, src), 3);
    int t = 0;
    for (size_t i = 0; i < c.issues.size(); ++i) {
        const RecOp* o = b.op_of(c.issues[i].op);
        if (!o || std::find(o->writes.begin(), o->writes.end(), (rad_buf)src) == o->writes.end())
            continue;
        REQUIRE(i > 0);
        CHECK_EQ(b.op_of(c.issues[i - 1].op)->op, std::string("rmsnorm_quant_i8"));
        REQUIRE(c.issues[i].opd.size() == 2);
        CHECK_EQ(c.issues[i].opd[1].handle, (uint32_t)src);
        CHECK_EQ(c.issues[i].opd[1].offset, (int64_t)t * 5376);   /* the tap's own column slice */
        ++t;
    }
}

/* A draft pass runs only the drafter: the context pass from the taps into the drafter's cache,
 * the query pass over the block to the proposal. */
TEST(gemma4_31b_dflash_passes_run_the_drafter_and_nothing_else) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    g431df_encs(b, false, true);
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    const int src = buf_named(b, "dflash_src"), tok = buf_named(b, "dflash_tokens");

    for (int pass : { -1, 1 }) {
        RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
        RadBatch batch = decode_batch();
        batch.draft_pass = pass;
        c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        REQUIRE(!ops.empty());
        CHECK_EQ(count_issued(ops, "gelu_mul_quant_i8"), 0);   /* no trunk layer */
        CHECK_EQ(count_issued(ops, "kv_store"), 5);
        if (pass < 0) {
            CHECK_EQ(ops.front(), std::string("quant_act_fp8"));
            const RecOp* o = b.op_of(c.issues.front().op);
            CHECK(std::find(o->reads.begin(), o->reads.end(), (rad_buf)src) != o->reads.end());
            CHECK_EQ(count_issued(ops, "attn_paged") + count_issued(ops, "attn_paged_qrow"), 0);
        } else {
            CHECK_EQ(ops.front(), std::string("embed_lookup"));
            CHECK_EQ(count_issued(ops, "attn_paged") + count_issued(ops, "attn_paged_qrow"), 5);
            CHECK_EQ(count_writes_to(b, c, tok), 1);
        }
    }
}

/* Q8, the drafter on the int8 grid. Every linear is gemm_nt_q at w8a8 over quant_act_i8's per-row
 * activation, each residual add, norm and quantiser is one rmsnorm_quant_i8 (ten per draft pass, as
 * rmsnorm_quant_fp8 is on the E4M3 grid), and nothing on the E4M3 grid is declared. The
 * gate_up|silu|quant fold is fp8a8 only, so the int8 grid keeps silu_mul and the quantiser. Where
 * rmsnorm_quant_i8 is refused, the chain runs. */
TEST(gemma4_31b_dflash_on_the_int8_grid_reads_w8a8_and_folds_its_norms) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        g431df_encs(b, true, true);
        if (refuse) b.refuse.insert("rmsnorm_quant_i8");
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK(b.bufs[buf_named(b, "df_hq") - 1].d.dtype == RAD_I8);
        CHECK_EQ(b.bufs[buf_named(b, "df_hs") - 1].d.shape[1], 1);   /* a scale a row */
        const RecWeight* wq = find_w(b, "dflash.blk.0.attn_q.weight");
        const RecWeight* sq = find_w(b, "dflash.blk.0.attn_q.scale");
        REQUIRE(wq != nullptr && sq != nullptr);
        CHECK(wq->d.dtype == RAD_I8 && sq->d.dtype == RAD_F16);
        CHECK_EQ(sq->d.shape[1], 5376 / 128);

        RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "rmsnorm_quant_i8"), refuse ? 0 : 10);
        CHECK_EQ(count_issued(ops, "add"), refuse ? 10 : 1);
        /* the attention output's and the gated product's a layer, x's twice a layer in the chain,
         * and the head's copy of the gathered rows */
        CHECK_EQ(count_issued(ops, "quant_act_i8"), (refuse ? 20 : 10) + 1);
        CHECK_EQ(count_issued(ops, "silu_mul"), 5);
        for (const char* fp : { "quant_act_fp8", "rmsnorm_quant_fp8", "gemm_nt_q_gated" })
            CHECK_EQ(count_issued(ops, fp), 0);
        int q = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "gemm_nt_q") continue;
            CHECK_EQ(pstr(o, "dtype"), std::string("w8a8"));
            ++q;
        }
        CHECK_EQ(q, 5 * 6 + 1);   /* six linears a layer, and the head */
        if (refuse) continue;

        /* the fused form's operands: rmsnorm_quant_i8's eight, the delta and the residual past
         * layer 0, and no junction gains */
        const uint32_t x = (uint32_t)buf_named(b, "df_x"), h = (uint32_t)buf_named(b, "df_h");
        int k = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "rmsnorm_quant_i8") continue;
            REQUIRE(i.opd.size() == 8);
            CHECK_EQ(i.opd[0].handle, k++ == 0 ? x : h);
            CHECK_EQ((int)i.opd[6].kind, (int)RAD_OPK_NONE);
            CHECK_EQ((int)i.opd[7].kind, (int)RAD_OPK_NONE);
        }

        /* and the context pass: two quantisers, eleven w8a8 GEMMs (fc, then k and v a layer) */
        RadCtx cc; cc.b = &b; cc.rank = 0; cc.world = 1;
        RadBatch cb = decode_batch();
        cb.draft_pass = -1;
        cc.batch = &cb;
        gemma4_bf16::step(&cc, &cb);
        CHECK(cc.step_fail.empty());
        const std::vector<std::string> cops = issued_ops(b, cc);
        CHECK_EQ(count_issued(cops, "quant_act_i8"), 2);
        CHECK_EQ(count_issued(cops, "gemm_nt_q"), 11);
    }
}

/* The drafter's linears take one grid: one quantised activation feeds several of them, so an fc on
 * the E4M3 grid and layers on the int8 one would read pairs nothing writes. Refused by name. */
TEST(gemma4_31b_dflash_refuses_linears_on_two_grids) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    g431df_encs(b, true, false);
    b.encs[0].second = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);   /* fc on E4M3 */
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_E_DTYPE);
}

/* The head's int8 copy: where the container holds one, the gathered rows are quantised a row at a
 * time and read by gemm_nt_q at w8a8 over it; otherwise the head reads the tied bf16 table. */
TEST(gemma4_31b_dflash_head_reads_the_int8_copy_where_the_container_holds_one) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int copy = 0; copy < 2; ++copy) {
        RadBuilder b;
        g431df_encs(b, true, copy == 1);
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        const int dlog = buf_named(b, "df_dlog");
        const RecOp* head = nullptr;
        for (const auto& o : b.ops)
            if (o.resolved &&
                std::find(o.writes.begin(), o.writes.end(), (rad_buf)dlog) != o.writes.end())
                head = &o;
        REQUIRE(head != nullptr);
        CHECK_EQ(head->op, std::string(copy ? "gemm_nt_q" : "gemm_nt"));
        if (!copy) continue;
        CHECK_EQ(pstr(head, "dtype"), std::string("w8a8"));
        const int selq = buf_named(b, "df_selq"), sels = buf_named(b, "df_sels");
        REQUIRE(selq > 0 && sels > 0);
        CHECK(b.bufs[selq - 1].d.dtype == RAD_I8);
        CHECK_EQ(b.bufs[sels - 1].d.shape[1], 1);
        const RecWeight* hs = find_w(b, "dflash.draft_head.scale");
        REQUIRE(hs != nullptr);
        CHECK(hs->d.dtype == RAD_F16);
        CHECK_EQ(hs->d.shape[0], 262144);
        CHECK_EQ(hs->d.shape[1], 5376 / 128);
    }
}

/* At four ranks, as the R9700 box serves it: the drafter's heads split 16 / 2 kv per rank, its head
 * is vocab-sharded with the trunk's and merged by pair_argmax, and both 8-bit grids declare (the
 * int8 one's prefill GEMM takes N in 256s, which a 256-row k and v still are). */
TEST(gemma4_31b_dflash_at_four_ranks) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int i8 = 0; i8 < 2; ++i8)
    for (const bool served : { true, false }) {
        RadBuilder b;
        g431df_encs(b, i8 == 1, true);
        if (!served) b.refuse.insert("attn_paged_qrow");
        RadBuildCtx ctx = build_ctx(1, 4);
        ctx.max_spec = 15;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        const rad_kvgroup kvd = kv_named(b, "kv_draft");
        REQUIRE(kvd != 0);
        /* two kv heads a rank: once at gqa 8 where attn_paged_qrow serves it, else each held twice
         * at gqa 4 (the declares test says why) */
        CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, served ? 2 : 4);
        CHECK_EQ(b.bufs[buf_named(b, "df_q") - 1].d.shape[1], 16 * 128);
        CHECK_EQ(b.bufs[buf_named(b, "df_k") - 1].d.shape[1], 2 * 128);
        CHECK(buf_named(b, "df_dgath") > 0);
        for (const auto& o : b.ops)
            if ((o.op == "attn_paged" || o.op == "attn_paged_qrow") && pint(&o, "head_dim") == 128 &&
                !o.reads.empty())
                CHECK_EQ(pint(&o, "gqa"), served ? 8 : 4);

        RadCtx c; c.b = &b; c.rank = 1; c.world = 4;
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "pair_argmax"), 1);
        CHECK_EQ(count_issued(ops, "all_gather"), 1);
        /* the embedding's reduction and two a layer */
        CHECK_EQ(count_issued(ops, "all_reduce"), 1 + 2 * 5);
        /* k and v into the cache's layout: two heads, two copies each, a layer -- or none */
        CHECK_EQ(count_issued(ops, "cast"), served ? 0 : 5 * 2 * (2 * 2));
    }
}

/* The 31B's prefill in two slices (gemma4_bf16, trunk_overlap). The order and the work: every
 * trunk reduction rides lane 1 behind its producer and ahead of its consumer and nothing else does;
 * no two issues on different lanes reach the same rows unordered (`h` carries both reductions); the
 * cut leaves both slices at least 512 rows; and the slices issue exactly the rows the whole step
 * issues, the drafter's taps a slice each. Only `h` is taken out of the arena's sharing. A step
 * under two slices, and a decode step, run whole with their reductions on lane 0. At two ranks the
 * slices take the plain junctions, the whole step the ones with the all-reduce folded in. */
TEST(a_gemma4_31b_prefill_step_runs_as_two_slices_with_its_reductions_on_the_second_lane) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    const int64_t E = 5376, L = 12;
    for (int world : { 4, 2 }) {
        auto model = [&](RadBuilder& b, bool lossy) {
            g431df_encs(b, false, true);
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_tok  = 4096;
            ctx.max_spec = 15;
            if (lossy) ctx.tp_wire_lossy = 1;   /* nothing shows a slice its bits on that wire */
            return gemma4_bf16::declare(&b, &meta, &ctx);
        };
        auto run = [&](RadBuilder& b, RadCtx& c, RadBatch& batch) {
            c.b = &b; c.rank = 0; c.world = world; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
        };

        struct Case { int64_t T, cut; };
        for (const Case k : { Case{4096, 2048}, Case{1100, 512}, Case{1024, 512}, Case{1023, 0} }) {
            const int32_t out[1] = { (int32_t)(k.T - 1) };
            RadBuilder b;
            CHECK_OK(model(b, false));
            CHECK(noted(b, "prefill overlap: a step of 1024 rows or more runs as two slices"));
            REQUIRE(b.concurrent.size() == 1);
            CHECK_EQ(b.concurrent[0], (rad_buf)buf_named(b, "h"));
            RadBatch batch = one_prompt(k.T, out);
            RadCtx c;
            run(b, c, batch);
            CHECK(c.step_fail.empty());
            CHECK_EQ(c.lane, 0);
            CHECK_EQ(races(b, c), 0);

            std::vector<int64_t> lane1;
            int other1 = 0;
            for (const auto& i : c.issues) {
                if (i.lane != 1) continue;
                const RecOp* o = b.op_of(i.op);
                REQUIRE(o != nullptr);
                if (o->op == "all_reduce") lane1.push_back(i.n);
                else ++other1;
            }
            CHECK_EQ(other1, 0);
            if (k.cut == 0) {
                CHECK(lane1.empty());
            } else {
                REQUIRE(lane1.size() == (size_t)(4 * L));
                for (int64_t l = 0; l < L; ++l) {
                    CHECK_EQ(lane1[4 * l + 0], k.cut * E);              /* attention, slice A */
                    CHECK_EQ(lane1[4 * l + 1], (k.T - k.cut) * E);      /* attention, slice B */
                    CHECK_EQ(lane1[4 * l + 2], k.cut * E);              /* feed-forward, slice A */
                    CHECK_EQ(lane1[4 * l + 3], (k.T - k.cut) * E);      /* feed-forward, slice B */
                }
                CHECK_EQ(count_issued(issued_ops(b, c), "ar_rmsnorm_quant_i8"), 0);
            }

            /* The whole step, on the lossy wire: the same rows by op -- every junction, GEMM and
             * quantiser over the same rows, the reductions over the same elements. */
            RadBuilder w;
            CHECK_OK(model(w, true));
            CHECK(noted(w, "prefill overlap: off (the lossy all-reduce wire)"));
            CHECK(w.concurrent.empty());
            RadBatch wbatch = one_prompt(k.T, out);
            RadCtx wc;
            run(w, wc, wbatch);
            CHECK(wc.step_fail.empty());
            for (const auto& i : wc.issues) CHECK_EQ(i.lane, 0);
            std::map<std::string, int64_t> sliced = rows_by_op(b, c, k.T), whole = rows_by_op(w, wc, k.T);
            if (world == 2 && k.cut == 0) {
                /* At two ranks on the exact wire a step that runs whole folds all but two of its
                 * reductions -- the embedding's and the last layer's feed-forward -- into its
                 * 2L - 1 junctions; the lossy wire folds none. */
                CHECK_EQ(sliced["ar_rmsnorm_quant_i8"], (2 * L - 1) * k.T);
                sliced.erase("ar_rmsnorm_quant_i8");
                sliced["all_reduce"] += (2 * L - 1) * k.T * E;
                sliced["rmsnorm_quant_i8"] += (2 * L - 1) * k.T;
            }
            CHECK(sliced == whole);
            CHECK_EQ((int)ns_of(b, c, "cast").size(), k.cut ? 6 : 3);   /* three taps, a slice each */
        }

        /* a decode step runs whole: every reduction on lane 0, nothing on lane 1 */
        RadBuilder b;
        CHECK_OK(model(b, false));
        RadBatch d = decode_batch();
        RadCtx c;
        run(b, c, d);
        CHECK(c.step_fail.empty());
        CHECK(c.joins.empty());
        for (const auto& i : c.issues) CHECK_EQ(i.lane, 0);
        /* folded where the plugin declares the fold at this rank count (two ranks; four with the
         * four-rank junction), every junction but the last layer's feed-forward one */
        bool fold = false;
        for (const auto& o : b.ops) fold = fold || o.op == "ar_rmsnorm_quant_i8";
        CHECK(fold || world == 4);
        CHECK_EQ(count_issued(issued_ops(b, c), "ar_rmsnorm_quant_i8"), fold ? 2 * (int)L - 1 : 0);
    }

    /* the overlap needs the fused junctions: a bf16 trunk keeps its chain and runs whole */
    RadBuilder b;
    RadModelMeta plain = gemma4_meta();
    RadBuildCtx ctx = build_ctx(0, 4);
    ctx.max_tok = 4096;
    CHECK_OK(gemma4_bf16::declare(&b, &plain, &ctx));
    CHECK(noted(b, "prefill overlap: off (the residual junctions are not fused)"));
    CHECK(b.concurrent.empty());

    /* and the switch: every step whole, nothing on the second lane */
    setenv("RADIANCE_GEMMA4_OVERLAP", "0", 1);
    RadBuilder sw;
    g431df_encs(sw, false, true);
    RadBuildCtx sctx = build_ctx(0, 4);
    sctx.max_tok  = 4096;
    sctx.max_spec = 15;
    const int rc = gemma4_bf16::declare(&sw, &meta, &sctx);
    unsetenv("RADIANCE_GEMMA4_OVERLAP");
    CHECK_OK(rc);
    CHECK(noted(sw, "prefill overlap: off (RADIANCE_GEMMA4_OVERLAP=0)"));
    CHECK(sw.concurrent.empty());
    const int32_t out[1] = { 4095 };
    RadBatch big = one_prompt(4096, out);
    RadCtx sc; sc.b = &sw; sc.rank = 0; sc.world = 4; sc.batch = &big;
    gemma4_bf16::step(&sc, &big);
    CHECK(sc.step_fail.empty());
    for (const auto& i : sc.issues) CHECK_EQ(i.lane, 0);
}

/* Where no loaded library declares libr4d's fused ops (a CPU-only build, where rad-convert and the
 * host backend run), asking for one is an error, so the 31B with its DFlash drafter asks first and
 * declares the chains: the junctions, the attention prologue, the gated quantiser, the folded
 * all-reduce, and the drafter's fused norm on the int8 grid. It declares on both grids, at one rank
 * and two, and never asks for an undeclared op. */
TEST(gemma4_31b_without_libr4d_declares_the_chains_and_asks_for_no_undeclared_op) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int i8 = 0; i8 < 2; ++i8)
        for (int world : { 1, 2 }) {
            RadBuilder b;
            g431df_encs(b, i8 == 1, true);
            for (const char* op : { "rmsnorm_quant_i8", "gelu_mul_quant_i8", "qkv_norm_rope",
                                    "qkv_norm_rope_kv_store", "ar_rmsnorm_quant_i8" })
                b.undeclared.insert(op);
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_spec = 15;
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            CHECK(b.asked_undeclared.empty());
            RadCtx c; c.b = &b; c.rank = 0; c.world = world;
            RadBatch batch = decode_batch();
            c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK(c.step_fail.empty());
            const std::vector<std::string> ops = issued_ops(b, c);
            CHECK_EQ(count_issued(ops, "rmsnorm_p"), 12);    /* the prologue's value norm, unfused */
            CHECK_EQ(count_issued(ops, "gelu_mul"), 12);
            batch.draft_pass = 1;
            RadCtx d; d.b = &b; d.rank = 0; d.world = world; d.batch = &batch;
            gemma4_bf16::step(&d, &batch);
            CHECK(d.step_fail.empty());
            CHECK_EQ(count_issued(issued_ops(b, d), "rmsnorm_quant_i8"), 0);
        }
}

/* The 31B's engine steps lint clean with z-lab's drafter, on both of the drafter's grids, at one,
 * two and four ranks (the first and last rank of each): a decode step, a mixed step, a prefill
 * chunk tall enough to run as two slices, and a verify step with the drafter's context and query
 * passes behind it. The lint holds the buffers each pass reads to what an earlier op of the step
 * wrote and the planner kept alive, across the passes, so a tap whose plane the planner let another
 * buffer share, or a drafter buffer read after its arena slot was reused, fails here. Eight
 * sequences at a block of 16 is 128 query rows, past every fused band the drafter has. Both trunk
 * layouts: q, k and v apart, and as one linear, where a one-row decode reads q in the row and a
 * taller step copies it out. */
TEST(gemma4_31b_steps_lint_clean_with_a_dflash_drafter) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int fused = 0; fused < 2; ++fused)
    for (int i8 = 0; i8 < 2; ++i8)
        for (int world : { 1, 2, 4 })
            for (int rank : { 0, world - 1 }) {
                if (world == 1 && rank > 0) continue;
                RadBuilder b;
                g431df_encs(b, i8 == 1, true, fused == 1);
                RadBuildCtx ctx = build_ctx(rank, world);
                ctx.max_tok = 2048;
                ctx.max_spec = 15;
                REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
                REQUIRE_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
                for (const auto& k : { kit(b, { 1 }, {}), kit(b, { 1, 1, 1 }, {}),
                                       kit(b, { 8, 1 }, { 5 }), kit(b, {}, { 1500 }) }) {
                    const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, rank, world);
                    CHECK_PASSES(b, v);
                }
                for (int64_t n_seq : { 2, 8 }) {
                    const auto ver = kit(b, std::vector<int>((size_t)n_seq, 16), {});
                    const auto ctxp = context_pass(b, *ver);
                    const auto qry = query_pass(b, n_seq, 16, 15);
                    const auto v = run_passes(b, gemma4_bf16::step,
                                              { ver.get(), ctxp.get(), qry.get() }, rank, world);
                    CHECK_PASSES(b, v);
                }
            }
}

/* ============ gemma4 31B + DFlash2: RedHatAI's speculators-format drafter ============ */
namespace {
/* The 31B fixture with a DFlash2 drafter in vLLM's speculators layout, as rad-convert merges
 * RedHatAI/gemma-4-31B-it-speculator.dflash2's config.json: the speculator's own keys at the top of
 * `draft.`, the decoder layers' config under `draft.transformer_layer_config.`, booleans as 1/0.
 * Taps at aux ids 1 5 9 (the residual entering those layers) for the fixture's twelve layers. */
struct G431Df2 {
    std::vector<const char*> keys, vals;
    G431Df2() {
        keys.assign(std::begin(k_gem4_keys), std::end(k_gem4_keys));
        vals.assign(std::begin(k_gem4_vals), std::end(k_gem4_vals));
        const char* kv[][2] = {
            { "draft.architectures", "[\"DFlash2DraftModel\"]" },
            { "draft.speculators_model_type", "dflash2" },
            { "draft.block_size", "8" },
            { "draft.mask_token_id", "4" },
            { "draft.aux_hidden_state_layer_ids", "[1, 5, 9]" },
            { "draft.selector_rank", "256" },
            { "draft.selector_top_k", "16" },
            { "draft.conv_kernel_size", "2" },
            { "draft.conv_group_size", "16" },
            { "draft.sample_from_anchor", "0" },
            { "draft.sliding_window_non_causal", "1" },
            { "draft.draft_vocab_size", "262144" },
            { "draft.transformer_layer_config.num_hidden_layers", "5" },
            { "draft.transformer_layer_config.head_dim", "256" },
            { "draft.transformer_layer_config.num_attention_heads", "32" },
            { "draft.transformer_layer_config.num_key_value_heads", "16" },
            { "draft.transformer_layer_config.intermediate_size", "21504" },
            { "draft.transformer_layer_config.rms_norm_eps", "1e-06" },
            { "draft.transformer_layer_config.rope_parameters.rope_theta", "10000.0" },
            { "draft.transformer_layer_config.hidden_size", "5376" },
            { "draft.transformer_layer_config.vocab_size", "262144" },
            { "draft.transformer_layer_config.hidden_act", "silu" },
            { "draft.transformer_layer_config.sliding_window", "2048" },
            { "draft.transformer_layer_config.use_sliding_window", "1" },
            { "draft.transformer_layer_config.layer_types",
              "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
              "\"sliding_attention\", \"sliding_attention\"]" },
        };
        for (const auto& p : kv) { keys.push_back(p[0]); vals.push_back(p[1]); }
    }
    RadModelMeta meta() const {
        RadModelMeta m = gemma4_meta();
        m.n_kv   = (int)keys.size();
        m.kv_key = keys.data();
        m.kv_val = vals.data();
        return m;
    }
};

/* The -df2 recipe's grids: the drafter's linears at block fp8, its head copy int8 (or, without
 * `head8`, no 8-bit encoding, which leaves the 2-bit grid), and the trunk at w4a8 with q|k|v fused. */
void g431df2_encs(RadBuilder& b, bool head8) {
    g431df_encs(b, false, false, true);
    if (head8) b.encs.insert(b.encs.begin(), { "dflash.draft_head.", rad_enc_affine(RAD_I8, RAD_F16, 1, 128) });
}
}  /* namespace */

/* The 31B declares RedHatAI's DFlash2 drafter as a block drafter past its trunk, read from the
 * speculators layout: depth 7 (block 8, the anchor predicting nothing), mask 4, every layer on a
 * 2048-token window, non-causal, its own windowed KV group of 16 kv heads x 256, the taps on the
 * layers BEFORE the aux ids (1 5 9 -> 0 4 8), the trunk's table read unscaled, and its head on the
 * 8-bit copy, the candidates merged across ranks where the head is vocab-sharded. */
TEST(gemma4_31b_declares_redhats_dflash2_from_the_speculators_layout) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    RadBuildCtx pc = build_ctx();
    pc.max_spec = 7;
    int taps[8] = {};
    CHECK_EQ(gemma4::dflash2_taps(&meta, &pc, 12, taps, 8), 3);
    CHECK_EQ(taps[0], 0);
    CHECK_EQ(taps[1], 4);
    CHECK_EQ(taps[2], 8);
    RadArchProbe p{};
    CHECK_OK(gemma4_bf16::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);

    for (int world : { 1, 4 })
    for (int head8 = 0; head8 < 2; ++head8) {
        RadBuilder b;
        g431df2_encs(b, head8 == 1);
        RadBuildCtx ctx = build_ctx(0, world);
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);

        CHECK_EQ(b.drafter_name, std::string("dflash2"));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(b.drafter.depth, 7);
        CHECK_EQ((int)b.drafter.block, 8);
        CHECK_EQ(b.drafter.mask_token, 4);
        CHECK_EQ(b.drafter.window, 2048);
        CHECK_EQ(b.drafter.proposal, (rad_buf)buf_named(b, "dflash_tokens"));

        const rad_kvgroup kvd = kv_named(b, "kv_draft");
        REQUIRE(kvd != 0);
        CHECK_EQ((int)b.kv_groups[kvd - 1].d.kind, (int)RAD_KV_WINDOW);
        CHECK_EQ(b.kv_groups[kvd - 1].d.head_dim, 256);
        CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, 16 / world);

        const int src = buf_named(b, "dflash_src");
        REQUIRE(src > 0);
        CHECK_EQ(b.bufs[src - 1].d.shape[1], 3 * 5376);

        /* attention: head 256, gqa 2, non-causal, the window */
        int n_attn = 0;
        for (const auto& o : b.ops)
            if (o.op == "attn_paged" && pint(&o, "head_dim") == 256 && pint(&o, "window") == 2048 &&
                pint(&o, "causal") == 0) {
                ++n_attn;
                CHECK_EQ(pint(&o, "gqa"), 2);
            }
        CHECK_EQ(n_attn, 1);

        /* the drafter's lookup reads the trunk's table without Gemma's scale */
        const RecOp* te = find_op(b, "embed_lookup", 0);
        const RecOp* de = find_op(b, "embed_lookup", 1);
        REQUIRE(te != nullptr && de != nullptr);
        CHECK(te->w == de->w);
        CHECK_NEAR(pflt(te, "mult"), 73.5, 1e-9);
        CHECK(param(de, "mult") == nullptr);

        /* the head: 8-bit where the copy is int8, else the 2-bit grid; merged across ranks */
        bool head_seen = false;
        for (const auto& o : b.ops)
            if (o.op == "gemm_nt_q" && pint(&o, "K") == 5376 && pint(&o, "N") == 262144 / world) {
                head_seen = true;
                CHECK_EQ(pstr(&o, "dtype"), std::string(head8 ? "w8a8" : "w2a8"));
            }
        CHECK(head_seen);
        CHECK_EQ(count_op(b, "row_topk_merge"), world > 1 ? 1 : 0);
        CHECK_EQ(count_op(b, "dflash_select"), 1);
    }
}

/* DFlash2's two passes each run their attention prologue as one launch a layer where
 * qkv_norm_rope_kv_store resolves: the query pass with its q heads (8 a rank at four ranks; v_norm
 * 0), the context pass with none (n_head 0: the keys alone, q passed as k's buffer, which it does
 * not read). With the fold refused, both chains return: the norms, the rotaries and kv_store. */
TEST(gemma4_31b_dflash2_folds_both_passes_prologues) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    const int L = 5;
    for (int off = 0; off < 2; ++off) {
        RadBuilder b;
        g431df2_encs(b, true);
        if (off) b.refuse.insert("qkv_norm_rope_kv_store");
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok = 2048;
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const auto ver = kit(b, { 8 }, {});
        const auto ctxp = context_pass(b, *ver);
        const auto qry = query_pass(b, 1, 8, 7);
        const auto v = run_passes(b, gemma4_bf16::step, { ctxp.get(), qry.get() }, 0, 4);
        REQUIRE(steps_ok(v));
        for (int p = 0; p < 2; ++p) {
            const RadCtx& c = *v[(size_t)p];
            int fused = 0;
            for (const RecIssue& is : c.issues) {
                const RecOp* o = b.op_of(is.op);
                if (!o || o->op != "qkv_norm_rope_kv_store") continue;
                ++fused;
                CHECK_EQ(pint(o, "n_head"), p == 0 ? 0 : 8);
                CHECK_EQ(pint(o, "v_norm"), 0);
                if (p == 0) {
                    REQUIRE(is.opd.size() >= 2);
                    CHECK_EQ(is.opd[0].handle, is.opd[1].handle);
                }
            }
            const std::vector<std::string> ops = issued_ops(b, c);
            CHECK_EQ(fused, off ? 0 : L);
            CHECK_EQ(count_issued(ops, "kv_store"), off ? L : 0);
            CHECK_EQ(count_issued(ops, "rope"), off ? (p == 0 ? L : 2 * L) : 0);
        }
    }
}

/* DFlash2's query pass issues each residual add with the norm after it, as one add_rmsnorm where it
 * resolves: after every attention (the feed-forward norm) and every feed-forward (the next layer's
 * attention norm, after the last the final norm), ten on the 31B's five layers, no add left and
 * one plain norm (layer 0's, after the embedding). With add_rmsnorm refused, the pairs. */
TEST(gemma4_31b_dflash2_folds_each_residual_add_into_the_next_norm) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    const int L = 5;
    for (int off = 0; off < 2; ++off) {
        RadBuilder b;
        g431df2_encs(b, true);
        if (off) b.refuse.insert("add_rmsnorm");
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok = 2048;
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const auto qry = query_pass(b, 1, 8, 7);
        const auto v = run_passes(b, gemma4_bf16::step, { qry.get() }, 0, 4);
        REQUIRE(steps_ok(v));
        const std::vector<std::string> ops = issued_ops(b, *v[0]);
        CHECK_EQ(count_issued(ops, "add_rmsnorm"), off ? 0 : 2 * L);
        CHECK_EQ(count_issued(ops, "add"), off ? 2 * L : 0);
        CHECK_EQ(count_issued(ops, "rmsnorm"), off ? 2 * L + 1 : 1);
    }
}

/* Every engine step with RedHatAI's DFlash2 behind the trunk lints clean at one, two and four ranks:
 * decode, a mixed step, a prefill chunk, and a verify step with the drafter's context and query
 * passes behind it (block 8). */
TEST(gemma4_31b_steps_lint_clean_with_a_dflash2_drafter) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    for (int world : { 1, 2, 4 })
        for (int rank : { 0, world - 1 }) {
            if (world == 1 && rank > 0) continue;
            RadBuilder b;
            g431df2_encs(b, true);
            RadBuildCtx ctx = build_ctx(rank, world);
            ctx.max_tok = 2048;
            ctx.max_spec = 7;
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            for (const auto& k : { kit(b, { 1 }, {}), kit(b, { 1, 1, 1 }, {}),
                                   kit(b, { 8, 1 }, { 5 }), kit(b, {}, { 1500 }) }) {
                const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            for (int64_t n_seq : { 2, 8 }) {
                const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
                const auto ctxp = context_pass(b, *ver);
                const auto qry = query_pass(b, n_seq, 8, 7);
                const auto v = run_passes(b, gemma4_bf16::step,
                                          { ver.get(), ctxp.get(), qry.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
        }
}

/* DFlash2's fc, split by rows over the ranks and kept bit for bit. At four ranks each rank hands
 * over 1344 of c's 5376 columns and computes the 11 whole scale blocks covering them (1408 rows: a
 * span of the one stored weight, so a container converted before the split serves it), its window
 * starting 0 or 64 rows in; the GEMM takes its K split from the whole 5376 rows (split_n), and the
 * all_gather reads each window at the GEMM output's pitch and writes c whole. At two ranks a share
 * is 21 whole blocks; at one rank the fc is whole, and so is an MXFP4 fc at four (its GEMM picks
 * its own K split, so only block fp8 splits). */
TEST(gemma4_31b_dflash2_fc_splits_by_rows_without_changing_a_bit) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    const int64_t E = 5376, K = 3 * E;   /* the fixture taps three trunk layers */
    struct Case { int world; int rank; int64_t lo, hi, share; };
    for (const Case cs : { Case{ 4, 0, 0, 1408, 1344 }, Case{ 4, 1, 1280, 2688, 1344 },
                           Case{ 4, 2, 2688, 4096, 1344 }, Case{ 4, 3, 3968, 5376, 1344 },
                           Case{ 2, 0, 0, 2688, 2688 }, Case{ 2, 1, 2688, 5376, 2688 },
                           Case{ 1, 0, 0, 0, 0 } }) {
        RadBuilder b;
        g431df2_encs(b, true);
        RadBuildCtx ctx = build_ctx(cs.rank, cs.world);
        ctx.max_tok = 2048;
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const RecWeight* fw = find_w(b, "dflash.fc.weight");
        const RecWeight* fs = find_w(b, "dflash.fc.scale");
        REQUIRE(fw != nullptr);
        REQUIRE(fs != nullptr);
        CHECK_EQ(fw->d.shape[1], K);
        CHECK(find_w(b, "dflash.fc_tail.weight") == nullptr);
        int split_gemms = 0, fc_gathers = 0;
        for (const RecOp& o : b.ops) {
            if (o.op == "gemm_nt_q" && pint(&o, "split_n") == E) {
                ++split_gemms;
                CHECK_EQ(pint(&o, "N"), (long long)(cs.hi - cs.lo));
            }
            if (o.op == "all_gather" && pint(&o, "row") == cs.share) {
                ++fc_gathers;
                CHECK_EQ(pint(&o, "numel"), (long long)(2048 * E));   /* the trunk's instance */
                CHECK_EQ(pint(&o, "world_size"), (long long)cs.world);
            }
        }
        if (!cs.share) {
            CHECK_EQ(fw->d.shape[0], E);
            CHECK_EQ(fw->d.shard, RAD_SHARD_NONE);
            CHECK(!fw->span_set);
            CHECK_EQ(split_gemms, 0);
            continue;
        }
        CHECK_EQ(fw->d.shape[0], cs.hi - cs.lo);
        CHECK_EQ(fw->d.shard, RAD_SHARD_ROW);
        CHECK(fw->span_set);
        CHECK_EQ(fw->span_lo, cs.lo);
        CHECK_EQ(fw->span_hi, cs.hi);
        CHECK(fs->span_set);   /* the scale plane's span is in 128-row blocks */
        CHECK_EQ(fs->span_lo, cs.lo / 128);
        CHECK_EQ(fs->span_hi, cs.hi / 128);
        /* the window covers exactly this rank's share */
        CHECK(cs.lo <= cs.rank * cs.share && (cs.rank + 1) * cs.share <= cs.hi);
        CHECK_EQ(split_gemms, 1);
        CHECK_EQ(fc_gathers, 1);
    }

    /* An MXFP4 drafter: fc whole on every rank. */
    {
        RadBuilder b;
        g431df2_encs(b, true);
        for (auto& e : b.encs)
            if (e.first == "dflash.fc." || e.first == "dflash.blk.")
                e.second = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
        RadBuildCtx ctx = build_ctx(1, 4);
        ctx.max_tok = 2048;
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const RecWeight* fw = find_w(b, "dflash.fc.weight");
        REQUIRE(fw != nullptr);
        CHECK_EQ(fw->d.shape[0], E);
        CHECK_EQ(fw->d.shard, RAD_SHARD_NONE);
        CHECK(find_w(b, "dflash.fc_tail.weight") == nullptr);
        for (const RecOp& o : b.ops) CHECK(pint(&o, "split_n") != E);
    }
}

/* Where the container holds q, k and v as one linear (a recipe naming blk.*.attn_qkv.weight), the
 * 31B declares one projection a layer -- q | k | v on a sliding layer, q | k on a global one, each
 * rank its slice of every part -- and none of the three apart, and the junction writes that
 * linear's int8 pair. The step: four GEMMs a layer instead of six (five on a global layer). A
 * global layer's attention reads q in the row (attn_paged_qrow); a sliding layer's reads it there
 * at one row and from attn_qs, copied out, at sixteen. */
TEST(gemma4_31b_takes_q_k_and_v_as_one_linear_where_the_container_holds_them_so) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    const int64_t E = 5376, L = 12;
    for (int world : { 1, 4 }) {
        RadBuilder b;
        g431df_encs(b, false, true, true);
        RadBuildCtx ctx = build_ctx(0, world);
        ctx.max_tok = 2048;
        ctx.max_spec = 15;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);

        for (int64_t l = 0; l < L; ++l) {
            const bool global = (l + 1) % 6 == 0;
            const std::string pre = "blk." + std::to_string(l) + ".";
            const RecWeight* qkv = nullptr;
            for (const RecWeight& w : b.weights) {
                if (w.name == pre + "attn_qkv.weight") qkv = &w;
                for (const char* sep : { "attn_q.weight", "attn_k.weight", "attn_v.weight" })
                    CHECK(w.name != pre + sep);
            }
            REQUIRE(qkv);
            CHECK_EQ(qkv->d.shape[0], (global ? 32 * 512 + 4 * 512 : 32 * 256 + 2 * 16 * 256) / world);
            CHECK_EQ(qkv->d.shape[1], E);
            CHECK_EQ((int)qkv->d.n_row_parts, global ? 2 : 3);
        }

        for (const int q : { 1, 16, 40 }) {
            const auto k = kit(b, { q }, {});
            const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(b, v);
            const std::vector<std::string> ops = issued_ops(b, *v[0]);
            CHECK_EQ(count_issued(ops, "gemm_nt_q"), (int)(4 * L));
            /* the sliding layers read the row up to 32 rows a sequence (gqa 2), copy q out above */
            const bool dec = q <= 32;
            CHECK_EQ(count_issued(ops, "attn_paged_qrow"), dec ? 12 : 2);
            CHECK_EQ(count_issued(ops, "attn_paged"), dec ? 0 : 10);
            CHECK_EQ(count_issued(ops, "quant_act_i8"), (int)L);   /* o_proj's input only */
            CHECK_EQ(count_writes_to(b, *v[0], buf_named(b, "attn_qs")), dec ? 0 : 10);
            /* the GEMM, then qkv_norm_rope in place */
            CHECK_EQ(count_writes_to(b, *v[0], buf_named(b, "attn_qkvs")), 2 * 10);
            CHECK_EQ(count_writes_to(b, *v[0], buf_named(b, "attn_qkvg")), 2 * 2);
        }

        /* Where attn_paged_qrow is not served: attn_paged everywhere, q in the row at one row and
         * copied out above it. */
        RadBuilder r;
        g431df_encs(r, false, true, true);
        r.refuse.insert("attn_paged_qrow");
        REQUIRE_EQ(gemma4_bf16::declare(&r, &meta, &ctx), RAD_OK);
        for (const int q : { 1, 16 }) {
            const auto k = kit(r, { q }, {});
            const auto v = run_passes(r, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(r, v);
            const std::vector<std::string> ops = issued_ops(r, *v[0]);
            CHECK_EQ(count_issued(ops, "attn_paged_qrow"), 0);
            CHECK_EQ(count_issued(ops, "attn_paged"), 12);
            CHECK_EQ(count_writes_to(r, *v[0], buf_named(r, "attn_qs")), q == 1 ? 0 : 10);
            CHECK_EQ(count_writes_to(r, *v[0], buf_named(r, "attn_qg")), q == 1 ? 0 : 2);
        }
    }
}

/* ==================================================================== the step lint, checked */

namespace {

/* A hand-built program for the lint's own checks: two transients of one arena, A and X, and four
 * ops declared in the order write A, read A, write X, read X, so A lives over ops 1..2 and X over
 * 3..4 and the plan may give them the same bytes. A KV group binds layer 0 only. */
struct LintProgram {
    RadBuilder b;
    rad_buf    A = 0, X = 0;
    rad_op     wA = 0, rA = 0, wX = 0, rX = 0;
    rad_kvgroup kv = 0;
    LintProgram() {
        RadBufDecl d{};
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 8; d.shape[1] = 16;
        A = rad_decl_buffer(&b, "A", &d);
        X = rad_decl_buffer(&b, "X", &d);
        wA = rw(b, "fill", {}, { A });
        rA = rw(b, "copy", { A }, {});
        wX = rw(b, "fill", {}, { X });
        rX = rw(b, "copy", { X }, {});
        RadKVGroupDecl g{};
        g.kind = RAD_KV_FULL; g.dtype = RAD_BF16; g.n_head_kv = 1; g.head_dim = 16;
        kv = rad_decl_kv_group(&b, "kv", &g);
        rad_bind_layer_kv(&b, 0, kv);
    }
    static rad_op rw(RadBuilder& b, const char* name, std::initializer_list<rad_buf> r,
                     std::initializer_list<rad_buf> w) {
        const rad_op h = rad_decl_op(&b, name, nullptr, 0, nullptr, 0);
        rad_op_reads(&b, h, r.begin(), (int)r.size());
        rad_op_writes(&b, h, w.begin(), (int)w.size());
        return h;
    }
    std::vector<std::string> lint(std::initializer_list<std::pair<rad_op, RadOperand>> issues) {
        RadCtx c{ .b = &b };
        for (const auto& [op, o] : issues) rad_issue(&c, op, &o, 1, RAD_N_BATCH);
        return lint_step(b, { &c });
    }
};

bool any_has(const std::vector<std::string>& v, const char* s) {
    for (const auto& x : v) if (has(x, s)) return true;
    return false;
}

}  /* namespace */

/* The lint's rules, each on a program built to break exactly one of them. A lint that cannot
 * fire would let every plugin test below pass vacuously. */
TEST(the_step_lint_reports_each_rule_it_checks) {
    {   /* Issued in declaration order: nothing to report. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.rA, RAD_B(p.A) },
                                { p.wX, RAD_B(p.X) }, { p.rX, RAD_B(p.X) } });
        CHECK_EQ(report_lint(f), (size_t)0);
    }
    {   /* X written between A's write and its read, with disjoint planned ranges. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.X) },
                                { p.rA, RAD_B(p.A) }, { p.rX, RAD_B(p.X) } });
        REQUIRE_EQ(f.size(), (size_t)1);
        CHECK(has(f[0], "reads 'A' (ops 1..2) after a write of 'X' (ops 3..4)"));
    }
    {   /* The same order is safe when A stays live through X's ops: the plan keeps them apart. */
        LintProgram p;
        rad_op_reads(&p.b, p.rX, &p.A, 1);
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.X) },
                                { p.rA, RAD_B(p.A) }, { p.rX, RAD_B(p.X) } });
        CHECK_EQ(report_lint(f), (size_t)0);
    }
    {   /* An op issued with a buffer outside the buffer's planned ops. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.A) } });
        CHECK(any_has(f, "undeclared use of 'A' (planned ops 1..2)"));
    }
    {   /* Rows past the declared extent, and an offset that runs a slice off the end. */
        LintProgram p;
        CHECK(any_has(p.lint({ { p.wA, RAD_B_N(p.A, 0, 9) } }), "span past 'A'"));
        CHECK(any_has(p.lint({ { p.wA, RAD_B_N(p.A, 16 * 7 + 1, 1) } }), "span past 'A'"));
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_B_N(p.A, 16 * 7, 1) } })), (size_t)0);
        /* A column slice keeps the declared pitch: 8 rows of the last 4 columns fit. */
        RadOperand col = RAD_B_COL(p.A, 12, 4);
        col.rows = 8;
        CHECK_EQ(report_lint(p.lint({ { p.wA, col } })), (size_t)0);
        col.offset = 13;
        CHECK(any_has(p.lint({ { p.wA, col } }), "span past 'A'"));
    }
    {   /* A fused op that did not resolve, issued anyway. */
        LintProgram p;
        CHECK(any_has(p.lint({ { RAD_NULL_HANDLE, RAD_B(p.A) } }), "null op issued"));
    }
    {   /* A transient read before anything in the step wrote it. */
        LintProgram p;
        CHECK(any_has(p.lint({ { p.rA, RAD_B(p.A) } }),
                      "reads 'A' before any issue of this step wrote it"));
    }
    {   /* A cache operand naming a layer no binding gave the group. */
        LintProgram p;
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_KV(p.kv, 0) } })), (size_t)0);
        CHECK(any_has(p.lint({ { p.wA, RAD_KV(p.kv, 1) } }), "layer 1 is not bound"));
    }
    {   /* A GEMM fed a column slice narrower than its buffer; the whole width is fine. */
        LintProgram p;
        const rad_op mm = LintProgram::rw(p.b, "gemm_nt", { p.A }, { p.X });
        const RadOperand half = RAD_B_COL(p.A, 0, 8), whole = RAD_B_COL(p.A, 0, 16);
        CHECK(any_has(p.lint({ { p.wA, RAD_B(p.A) }, { mm, half } }), "not contiguous"));
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_B(p.A) }, { mm, whole } })), (size_t)0);
    }
    {   /* A raw pointer into a buffer counts as that buffer. */
        LintProgram p;
        RadCtx c{ .b = &p.b };
        const RadOperand o = RAD_P_T((char*)rad_buf_ptr(&c, p.A) + 32, RAD_BF16);
        CHECK(any_has(p.lint({ { p.wX, o } }), "undeclared use of 'A' through a raw pointer"));
    }
}

/* ====================================================== qwen35_fp8 and qwen35moe_fp8 */

namespace {

/* Qwen3.8-27B-FP8's shape cut to the eight-layer fixture, its MTP head declared. */
MetaKV q35fp8_meta() {
    MetaKV k(qwen35_meta());
    k.m.quant = "fp8_e4m3";
    k.m.name  = "test-qwen35-27b-fp8";
    k.set("qwen35.nextn_predict_layers", "1");
    return k;
}

/* The 35B-A3B's routed feed-forward on the same trunk: 16 experts of 512, 4 a token, and a shared
 * expert of 512 (cut from 256 / 8 at 512). `n_ff` is the expert width, as rad-convert writes it. */
MetaKV q35moe_meta() {
    MetaKV k = q35fp8_meta();
    k.m.name = "test-qwen35-35b-a3b";
    k.m.n_ff = 512;
    k.m.n_expert = 16;
    k.m.n_expert_used = 4;
    k.set("shared_expert_intermediate_size", "512");
    return k;
}

/* A DFlash2 drafter merged under `draft.`: two layers reading trunk layers 1, 3 and 5, a block
 * of 8 (seven drafts), the selector of rank 64 over 8 candidates. */
MetaKV with_dflash2(MetaKV k) {
    k.set("draft.architectures", "DFlash2DraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "32")
     .set("draft.num_key_value_heads", "8")
     .set("draft.intermediate_size", "8192")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "10000000")
     .set("draft.vocab_size", "248320")
     .set("draft.dflash_config.target_layer_ids", "1 3 5")
     .set("draft.dflash_config.block_size", "8")
     .set("draft.dflash_config.mask_token_id", "248070")
     .set("draft.dflash_config.selector_rank", "64")
     .set("draft.dflash_config.selector_top_k", "8")
     .set("draft.dflash_config.conv_kernel_size", "2")
     .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

/* Every gemm_nt handle a pass issues is issued once: the bf16 GEMMs take a whole pass in one
 * launch (libr4d's bf16 row serves M to 65536), so a slice loop would be a stale band. */
bool each_gemm_nt_once(const RadBuilder& b, const RadCtx& c) {
    std::map<rad_op, int> n;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gemm_nt") ++n[i.op];
    }
    for (const auto& [op, k] : n)
        if (k != 1) {
            fprintf(stderr, "    gemm_nt op %u issued %d times in one pass\n", (unsigned)op, k);
            return false;
        }
    return true;
}

/* The engine steps a serial-drafter or block-drafter model runs, each linted whole: decode,
 * prefill, a mixed step, and a verify followed by the drafter's passes. */
template <class Fn>
void lint_qwen35_fp8_family(Fn declare, StepFn step, const RadModelMeta* meta, int max_spec,
                            bool block_drafter, int64_t block) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, max_spec);
            REQUIRE_EQ(declare(&b, meta, &ctx), RAD_OK);
            const auto dec = kit(b, { 1, 1, 1, 1 }, {});
            const auto pre = kit(b, {}, { 32, 32 });
            const auto mix = kit(b, { 1, 1 + max_spec }, { 6 });
            const auto ver = kit(b, std::vector<int>(4, 1 + max_spec), {});
            for (const Kit* k : { dec.get(), pre.get(), mix.get() }) {
                const auto v = run_passes(b, step, { k }, rank, world);
                CHECK_PASSES(b, v);
            }
            std::vector<std::unique_ptr<Kit>> passes;
            if (block_drafter) {
                passes.push_back(context_pass(b, *ver));
                passes.push_back(query_pass(b, 4, block, block - 1));
            } else {
                passes.push_back(history_pass(b, *ver));
                for (int r = 2; r <= max_spec; ++r) passes.push_back(round_pass(b, 4, r));
            }
            std::vector<const Kit*> all = { ver.get() };
            for (const auto& p : passes) all.push_back(p.get());
            const auto v = run_passes(b, step, all, rank, world);
            CHECK_PASSES(b, v);
            for (size_t i = 1; i < v.size(); ++i) CHECK(each_gemm_nt_once(b, *v[i]));
        }
}

}  /* namespace */

/* The fp8 sibling of qwen35_bf16: the same layers with every large linear a block-fp8 gemm_nt_q
 * read at fp8a8, one quantiser per activation, and the attention's output gate, multiply and
 * quantiser fused into gate_quant_fp8 (the sigmoid/mul pair is the bf16-linear form only). The
 * delta net's a|b projection stays bf16: its 48 rows are not a whole 128-row scale block. */
TEST(qwen35_fp8_declares_fp8_linears_and_the_fused_output_gate) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    int q = 0;
    for (const auto& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        ++q;
        CHECK_EQ(pstr(&o, "dtype"), std::string("fp8a8"));
        CHECK_EQ(pint(&o, "group"), 128);
    }
    /* q|gate, k, v, o for 2 attention layers; qkv|z and out for 6 delta-net layers; gate_up and
     * down for 8 feed-forwards. */
    CHECK_EQ(q, 2 * 4 + 6 * 2 + 8 * 2);
    CHECK_EQ(count_op(b, "gate_quant_fp8"), 2);
    CHECK_EQ(count_op(b, "sigmoid"), 0);
    CHECK_EQ(count_op(b, "mul"), 0);
    CHECK_EQ(count_op(b, "gemm_nt"), 6);
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt") CHECK(in(o.writes, (rad_buf)buf_named(b, "gdn_ab")));
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
    /* No speculation: no head, no drafter. */
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "mtp_x") == nullptr);
}

/* Every engine step the dense fp8 model runs lints clean at one and two ranks, the MTP head's
 * history pass and rounds included. */
TEST(qwen35_fp8_steps_lint_clean_with_its_mtp_head) {
    MetaKV mk = q35fp8_meta();
    lint_qwen35_fp8_family(qwen35_fp8::declare, qwen35_fp8::step, mk.meta(), 3, false, 0);
}

namespace {

/* Every trunk and MTP linear of a qwen35 container in the 4-bit form an int4 W4A16 release
 * imports to: u4 codes, an f16 scale and a u4 zero point per 128 of K. */
std::vector<std::pair<std::string, RadEncoding>> int4_linears() {
    RadEncoding w4 = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&w4, "zero", RAD_U4, 1, 128);
    std::vector<std::pair<std::string, RadEncoding>> e;
    for (const char* k : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        e.emplace_back(k, w4);
    return e;
}

int qwen35_fp8_int4_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    b->encs = int4_linears();
    return qwen35_fp8::declare(b, m, c);
}

}  /* namespace */

/* An int4 container's linears are read by the 4-bit GEMMs over their input's bf16 plane: at or
 * below 64 rows a `cast` to f16 and w4a16_asym, above that quant_act_i8g's int8 codes and
 * w4a8i8_asym, both over the codes and the joined (scale, zero) view. No linear keeps an fp8a8
 * GEMM, and the gated fold (an fp8a8 tile) is not declared over a 4-bit gate_up. */
TEST(qwen35_fp8_reads_an_int4_container_with_the_4bit_gemms) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8_int4_declare(&b, mk.meta(), &ctx), RAD_OK);
    int f16 = 0, i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        const std::string dt = pstr(&o, "dtype");
        const RecParam* m = param(&o, "M");
        REQUIRE(m != nullptr);
        if (dt == "w4a16_asym") {
            ++f16;
            CHECK_EQ(m->ival, 1LL);
            CHECK_EQ(m->ihi, 64LL);
        } else if (dt == "w4a8i8_asym") {
            ++i8;
            CHECK_EQ(m->ival, 65LL);
            CHECK_EQ(m->ihi, 256LL);
        } else {
            ++other;
        }
        CHECK_EQ(pint(&o, "group"), 128);
    }
    /* q|gate, k, v, o for 2 attention layers; qkv|z and out for 6 delta-net layers; gate_up and
     * down for 8 feed-forwards: 36 linears, each with both GEMMs, a cast and a quantiser. */
    CHECK_EQ(f16, 36);
    CHECK_EQ(i8, 36);
    CHECK_EQ(other, 0);
    int casts = 0;
    for (const auto& o : b.ops)
        if (o.op == "cast" && pstr(&o, "to") == "f16") {
            ++casts;
            CHECK_EQ(pstr(&o, "from"), std::string("bf16"));
            CHECK_EQ(param(&o, "M")->ihi, 64LL);
        }
    CHECK_EQ(casts, 36);
    CHECK_EQ(count_op(b, "quant_act_i8g"), 36);
    CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 0);
    int joined = 0;
    for (const RecWeight& w : b.weights)
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") {
            ++joined;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_F16);
        }
    CHECK_EQ(joined, 36);
}

/* Every engine step an int4 container runs lints clean at one and two ranks, the MTP head's
 * history pass and rounds included: the 4-bit linears' row parts and shards are cut like the
 * block-fp8 ones. */
TEST(qwen35_fp8_int4_steps_lint_clean_with_its_mtp_head) {
    MetaKV mk = q35fp8_meta();
    lint_qwen35_fp8_family(qwen35_fp8_int4_declare, qwen35_fp8::step, mk.meta(), 3, false, 0);
}

/* The head is a serial drafter at the deployment's depth, declared after the trunk's lm_head:
 * the planner reads declaration order as liveness, and the head's passes run after the trunk. */
TEST(qwen35_fp8_declares_its_mtp_head_after_the_trunk) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 3);
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(b.drafter.depth, 3);
    CHECK_EQ(b.drafter_name, std::string("mtp"));
    size_t logits = 0, first_head = 0;
    for (size_t i = 0; i < b.ops.size() && !logits; ++i)
        if (b.ops[i].op == "logits_gemm") logits = i + 1;
    const rad_buf hx = (rad_buf)buf_named(b, "mtp_x");
    for (size_t i = 0; i < b.ops.size() && !first_head; ++i)
        if (in(b.ops[i].reads, hx) || in(b.ops[i].writes, hx)) first_head = i + 1;
    REQUIRE(logits > 0 && first_head > 0);
    CHECK(first_head > logits);
    /* The head binds one more layer of the trunk's attention group, at index n_layers. */
    bool bound = false;
    for (const auto& [layer, g] : b.binds) bound |= layer == 8 && g == (rad_kvgroup)1;
    CHECK(bound);
    /* A draft round-1 pass is refused: the history pass carries it. */
    const auto one = round_pass(b, 4, 1);
    const auto v = run_passes(b, qwen35_fp8::step, { one.get() });
    CHECK(!v[0]->step_fail.empty());
}

/* DFlash2 replaces the MTP head when the container carries one: a block drafter whose block is
 * one longer than the depth, a power of two (its convolution masks positions with block - 1),
 * reading trunk taps copied out after layers 1, 3 and 5. */
TEST(qwen35_fp8_dflash2_drafter_takes_a_power_of_two_block) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 7);
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(b.drafter_name, std::string("dflash2"));
        CHECK_EQ(b.drafter.depth, 7);
        CHECK_EQ(b.drafter.block, 8);
        CHECK_EQ(b.drafter.mask_token, 248070);
        CHECK(find_b(b, "mtp_x") == nullptr);           /* no MTP head beside it */
        CHECK(noted(b, "the DFlash2 drafter replaces the 1-layer MTP head"));
        const RecBuf* src = find_b(b, "dflash_src");
        REQUIRE(src != nullptr);
        CHECK_EQ(src->d.shape[1], 3 * 5120);
        /* The drafter's layers follow the trunk's in its own group. */
        std::set<int> drafted;
        for (const auto& [layer, g] : b.binds)
            if (g == (rad_kvgroup)b.kv_groups.size()) drafted.insert(layer);
        CHECK(drafted == (std::set<int>{ 8, 9 }));
    }
    {   /* Six drafts make a block of seven, which the convolution cannot mask. */
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 6);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {   /* Another power of two runs, with a note that it was trained at 8. */
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 15);
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter.block, 16);
        CHECK(noted(b, "DFlash2 was trained at block_size 8"));
    }
    {   /* The probe fixes the depth at the trained block, so the engine asks for 7. */
        RadArchProbe p{};
        CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
        CHECK_EQ(p.draft_depth, 7);
        CHECK_EQ(p.draft_depth_fixed, 1);
    }
}

/* The block drafter's engine step (verify, context pass, query pass) lints clean at one and two
 * ranks, and its bf16 GEMMs run once a pass. */
TEST(qwen35_fp8_steps_lint_clean_with_a_dflash2_drafter) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    lint_qwen35_fp8_family(qwen35_fp8::declare, qwen35_fp8::step, mk.meta(), 7, true, 8);
}

/* Qwen3.8's DFlash2 drafter has head_dim 128, outside every qkv_norm_rope_kv_store a library serves
 * (libr4d's: 256 to 512), and an op declared for a shape that resolves to no device kernel refuses
 * startup whether or not it is issued. So the block declares no prologue fold for it, and each pass
 * keeps its norm, rope and kv_store chain; the residual adds, which add_rmsnorm serves at n_embd,
 * still fold, two a layer. */
TEST(qwen35_fp8_dflash2_head_128_drafter_declares_no_prologue_fold) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    int folds_128 = 0;
    for (const auto& o : b.ops)
        if (o.op == "qkv_norm_rope_kv_store" && pint(&o, "head_dim") == 128) ++folds_128;
    CHECK_EQ(folds_128, 0);
    CHECK_EQ(count_op(b, "add_rmsnorm"), 2 * 2);
}

/* =============================== gemma3: TranslateGemma 27B's trunk =========================== */

namespace {

/* TranslateGemma 27B's text geometry cut to twelve layers, two global (5 and 11) over ten sliding
 * ones by the legacy (l + 1) % 6 rule. One head geometry for both classes (head_dim 128, 32 / 16),
 * scores scaled by query_pre_attn_scalar^-0.5, linear rotary scaling on the global class only. */
MetaKV gemma3_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma3";
    m.name        = "test-translategemma-27b";
    m.quant       = "";
    m.n_layers    = 12;
    m.n_embd      = 5376;
    m.n_head      = 32;
    m.n_head_kv   = 16;
    m.head_dim    = 128;
    m.n_ff        = 21504;
    m.n_vocab     = 262208;
    m.n_ctx_train = 131072;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    MetaKV k(m);
    k.set("sliding_window", "1024")
     .set("_sliding_window_pattern", "6")
     .set("rope_local_base_freq", "10000")
     .set("rope_theta", "1000000")
     .set("query_pre_attn_scalar", "168")
     .set("rope_scaling.rope_type", "linear")
     .set("rope_scaling.factor", "8");
    return k;
}

MetaKV gemma3_dflash2_meta() {
    MetaKV k = gemma3_meta();
    k.set("draft.architectures", "DFlash2DraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "32")
     .set("draft.num_key_value_heads", "8")
     .set("draft.hidden_size", "5376")
     .set("draft.intermediate_size", "8192")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "1000000")
     .set("draft.vocab_size", "262208")
     .set("draft.use_sliding_window", "1")
     .set("draft.sliding_window", "2048")
     .set("draft.dflash_config.input_embedding_scale", "73.5")
     .set("draft.dflash_config.target_layer_ids", "1 5 10")
     .set("draft.dflash_config.block_size", "8")
     .set("draft.dflash_config.mask_token_id", "6")
     .set("draft.dflash_config.selector_rank", "64")
     .set("draft.dflash_config.selector_top_k", "8")
     .set("draft.dflash_config.conv_kernel_size", "2")
     .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

}  /* namespace */

/* Sliding layers in a window group declared first, the global ones in a full group, both at the
 * one head geometry; layers bound by the (l + 1) % 6 rule. */
TEST(gemma3_declares_two_kv_groups_and_binds_the_schedule) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE_EQ((int)b.kv_groups.size(), 2);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_sliding"));
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);
    CHECK_EQ(b.kv_groups[0].d.window, 1024);
    CHECK_EQ(b.kv_groups[0].d.n_head_kv, 16);
    CHECK_EQ(b.kv_groups[0].d.head_dim, 128);
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_global"));
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.window, 0);
    CHECK_EQ(b.kv_groups[1].d.head_dim, 128);
    CHECK_EQ((int)b.binds.size(), 12);
    for (int l = 0; l < 12; ++l) {
        CHECK_EQ((int)b.binds[(size_t)l].first, l);
        CHECK_EQ((int)b.binds[(size_t)l].second, (l + 1) % 6 == 0 ? 2 : 1);
    }
}

/* layer_types, where the config has it, names the schedule instead of the legacy pattern, and a
 * count that is not the model's is refused. */
TEST(gemma3_layer_types_name_the_global_layers) {
    MetaKV mk = gemma3_meta();
    mk.set("layer_types", "[\"full_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"full_attention\"]");
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    for (int l = 0; l < 12; ++l)
        CHECK_EQ((int)b.binds[(size_t)l].second, (l == 0 || l == 11) ? 2 : 1);

    mk.set("layer_types", "[\"full_attention\", \"sliding_attention\"]");
    RadBuilder c;
    CHECK(stderr_of_rc([&] { return gemma3::declare(&c, mk.meta(), &ctx); }) != RAD_OK);
}

/* Both classes attend at 1/sqrt(query_pre_attn_scalar), not 1/sqrt(head_dim); the sliding class
 * is windowed and rotates at rope_local_base_freq unscaled, the global class at rope_theta with
 * the linear factor; every norm is GemmaRMSNorm (wadd 1). */
TEST(gemma3_attention_scale_rope_and_norms_per_class) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    std::vector<const RecOp*> at;
    for (const auto& o : b.ops) if (o.op == "attn_paged") at.push_back(&o);
    REQUIRE(!at.empty());
    bool windowed = false, full = false;
    for (const RecOp* o : at) {
        CHECK_EQ(pint(o, "head_dim"), 128);
        CHECK_EQ(pint(o, "gqa"), 2);
        CHECK(std::fabs(pflt(o, "scale") - 1.0 / std::sqrt(168.0)) < 1e-9);
        if (pint(o, "window") == 1024) windowed = true;
        if (pint(o, "window") == 0) full = true;
    }
    CHECK(windowed && full);
    std::set<std::pair<double, double>> rot;   /* (theta, scale) of every rotary op */
    for (const auto& o : b.ops)
        if (param(&o, "theta")) rot.insert({ pflt(&o, "theta"), pflt(&o, "scale") });
    CHECK(rot.count({ 10000.0, 1.0 }) == 1);
    CHECK(rot.count({ 1000000.0, 0.125 }) == 1 || rot.count({ 1000000.0, 8.0 }) == 1);
    for (const auto& o : b.ops)
        if (param(&o, "wadd")) CHECK_EQ(pflt(&o, "wadd"), 1.0);
}

/* Gemma 2's soft-capping and a YaRN scaling are refused rather than served without them. */
TEST(gemma3_refuses_softcapping_and_yarn) {
    RadBuildCtx ctx = build_ctx();
    {
        MetaKV mk = gemma3_meta();
        mk.set("final_logit_softcapping", "30");
        RadBuilder b;
        CHECK_EQ(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }), RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = gemma3_meta();
        mk.set("rope_scaling.rope_type", "yarn").set("rope_scaling.original_max_position_embeddings", "8192");
        RadBuilder b;
        CHECK(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }) != RAD_OK);
    }
}

TEST(gemma3_every_op_declares_its_buffers_and_every_weight_has_a_name_map) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
}

/* A decode step and a prefill step lint clean at one and two ranks. */
TEST(gemma3_steps_lint_clean_at_one_and_two_ranks) {
    MetaKV mk = gemma3_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
            RadBatch batch = decode_batch();
            RadCtx c{ .b = &b, .batch = &batch };
            gemma3::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
            CHECK_EQ(count_op(b, "all_reduce") + count_op(b, "ar_rmsnorm_quant_fp8") +
                     count_op(b, "ar_rmsnorm_post_quant_fp8") > 0, world == 2);
        }
}

/* The DFlash2 drafter: a block of 8 (depth 7, fixed), taps copied out after the named layers into
 * one column-sliced buffer, the mask token from the config, and the drafter's own KV group after
 * the trunk's two. Its input_embedding_scale must be the trunk's sqrt(hidden) held in bf16
 * (73.5 on the 27B), which the embedding lookup applies. */
TEST(gemma3_dflash2_drafter_taps_and_block) {
    MetaKV mk = gemma3_dflash2_meta();
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 7);
    CHECK_EQ(b.drafter.block, 8);
    CHECK_EQ(b.drafter.mask_token, 6);
    const RecBuf* src = find_b(b, "dflash_src");
    REQUIRE(src != nullptr);
    CHECK_EQ(src->d.shape[1], 3 * 5376);
    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK(!noted(b, "input_embedding_scale"));
    {   /* A drafter trained at another embedding scale still serves, with a note. */
        MetaKV other = gemma3_dflash2_meta();
        other.set("draft.dflash_config.input_embedding_scale", "64");
        RadBuilder o;
        REQUIRE_EQ(gemma3::declare(&o, other.meta(), &ctx), RAD_OK);
        CHECK(noted(o, "input_embedding_scale"));
    }

    RadArchProbe p{};
    CHECK_OK(gemma3::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    MetaKV plain = gemma3_meta();
    RadArchProbe q{};
    CHECK_OK(gemma3::probe(plain.meta(), &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* --kv-cache-dtype fp8_v4: the trunk's attention cache is E4M3 K (256 bytes), int4 V (128) and
 * eight f16 scales, 400 bytes a token and head, declared as 25 * 256 / 32 = 200 E4M3 columns so the
 * core sizes its pages; every trunk op on it bands on kv_dtype fp8_v4. The DFlash2 drafter's own
 * group stays E4M3 at its head_dim. Without kv_quant the same build is plain fp8. */
TEST(qwen35_fp8_fp8_v4_cache_is_400_bytes_a_token_and_head) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    auto count = [](const RadBuilder& b, const char* kd) {
        int n = 0;
        for (const auto& o : b.ops) if (pstr(&o, "kv_dtype") == kd) ++n;
        return n;
    };
    RadBuilder f8;
    RadBuildCtx c8 = spec_ctx(0, 1, 7);
    c8.kv_dtype = RAD_F8E4M3;
    REQUIRE_EQ(qwen35_fp8::declare(&f8, mk.meta(), &c8), RAD_OK);
    CHECK_EQ(f8.kv_groups[0].d.head_dim, 256);
    CHECK_EQ(count(f8, "fp8_v4"), 0);
    const int n8 = count(f8, "fp8_e4m3");

    RadBuilder b;
    RadBuildCtx ctx = c8;
    ctx.kv_quant = RAD_KVQ_V4;
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_attn"));
    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.kind, RAD_KV_FULL);
    CHECK_EQ(f.dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(f.n_head_kv, 4);
    CHECK_EQ(f.head_dim, 200);
    const RadKVGroupDecl& d = b.kv_groups.back().d;
    CHECK_EQ(d.dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(d.head_dim, 128);
    const int v4 = count(b, "fp8_v4"), e4 = count(b, "fp8_e4m3");
    CHECK(v4 > 0);
    CHECK(e4 > 0);                       /* the drafter's attention */
    CHECK_EQ(v4 + e4, n8);
    CHECK(noted(b, "kv cache fp8_v4"));

    RadArchProbe p{};
    CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.kv_quant_ok, (1 << RAD_KVQ_V4) | (1 << RAD_KVQ_K6V4));
}

/* --kv-cache-dtype k6_v4: int6 K and int4 V, each in groups of 32 with an f16 scale, 352 bytes a
 * token and head, declared as 176 E4M3 columns; the trunk's ops band on kv_dtype k6_v4 and the
 * drafter's group stays E4M3. */
TEST(qwen35_fp8_k6_v4_cache_is_352_bytes_a_token_and_head) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    ctx.kv_dtype = RAD_F8E4M3;
    ctx.kv_quant = RAD_KVQ_K6V4;
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(f.head_dim, 176);
    CHECK_EQ(b.kv_groups.back().d.head_dim, 128);
    int k6 = 0, v4 = 0, e4 = 0;
    for (const auto& o : b.ops) {
        const std::string kd = pstr(&o, "kv_dtype");
        k6 += kd == "k6_v4"; v4 += kd == "fp8_v4"; e4 += kd == "fp8_e4m3";
    }
    CHECK(k6 > 0);
    CHECK(e4 > 0);
    CHECK_EQ(v4, 0);
    CHECK(noted(b, "kv cache k6_v4"));
}

/* libr4d's fp8_v4 and k6_v4 rows are h256/gqa6 only: the probe answers no for another geometry, so the
 * engine refuses the flag at startup, and declare refuses it too if asked anyway. */
TEST(qwen35_fp8_quantised_kv_is_refused_outside_h256_gqa6) {
    MetaKV mk = q35fp8_meta();
    mk.m.n_head_kv = 2;                  /* gqa 12 */
    RadArchProbe p{};
    CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.kv_quant_ok, 0);
    for (int q : { (int)RAD_KVQ_V4, (int)RAD_KVQ_K6V4 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        ctx.kv_dtype = RAD_F8E4M3;
        ctx.kv_quant = q;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* The 35B-A3B: a replicated router over every expert, a normalised top-k (norm_topk_prob
 * defaults on in this family; without it the weights sum to about 0.4), fp8a8 grouped GEMMs over
 * the expert planes, and the shared expert folded into the gather behind a sigmoid gate. */
TEST(qwen35moe_fp8_routes_every_expert_and_folds_the_shared_arm) {
    MetaKV mk = q35moe_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_op(b, "router_topk"), 8);
        for (const auto& o : b.ops) {
            if (o.op == "router_topk") {
                CHECK_EQ(pint(&o, "n_expert"), 16);
                CHECK_EQ(pint(&o, "top_k"), 4);
                CHECK_EQ(pint(&o, "norm"), 1);
            }
            if (o.op == "moe_gemm_q") {
                CHECK_EQ(pstr(&o, "dtype"), std::string("fp8a8"));
                CHECK_EQ(pint(&o, "n_expert"), 16);
                const long long N = pint(&o, "N"), K = pint(&o, "K");
                CHECK((N == 2 * 512 / world && K == 5120) || (N == 5120 && K == 512 / world));
            }
            if (o.op == "moe_gather") CHECK_EQ(pstr(&o, "act"), std::string("sigmoid"));
        }
        CHECK_EQ(count_op(b, "moe_gather"), 8);
        CHECK(count_op(b, "moe_gemm_q") >= 16);
        /* The shared arm's buffers sit on the concurrent list, out of the linear packing. */
        CHECK(std::count(b.concurrent.begin(), b.concurrent.end(),
                         (rad_buf)buf_named(b, "shexp_gate_up")) > 0);
    }
    /* Without experts this is the dense plugin's checkpoint, and the reverse. */
    {
        MetaKV dense = q35fp8_meta();
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(qwen35moe_fp8::declare(&b, dense.meta(), &ctx), RAD_E_UNSUPPORTED);
        RadBuilder b2;
        CHECK_EQ(qwen35_fp8::declare(&b2, mk.meta(), &ctx), RAD_E_UNSUPPORTED);
    }
}

/* The routed model's engine steps, its routed MTP head included, lint clean at one and two
 * ranks. */
TEST(qwen35moe_fp8_steps_lint_clean_with_its_routed_mtp_head) {
    MetaKV mk = q35moe_meta();
    lint_qwen35_fp8_family(qwen35moe_fp8::declare, qwen35moe_fp8::step, mk.meta(), 3, false, 0);
}

/* `layer_types` is checked against the interval rule, not ignored: a model built with the wrong
 * layers linear still generates fluent text. A disagreeing, short or unknown schedule is refused
 * by every plugin of the family; an agreeing one declares. */
TEST(the_qwen35_family_refuses_a_layer_schedule_that_contradicts_its_interval) {
    const char* agree = "linear_attention linear_attention linear_attention full_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    const char* wrong = "linear_attention linear_attention full_attention linear_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    const char* shrt  = "linear_attention linear_attention linear_attention full_attention";
    const char* odd   = "linear_attention linear_attention linear_attention sliding_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    for (const char* s : { agree, wrong, shrt, odd }) {
        const int want = s == agree ? RAD_OK : RAD_E_UNSUPPORTED;
        MetaKV b16(qwen35_meta());
        b16.set("layer_types", s);
        MetaKV f8 = q35fp8_meta();
        f8.set("layer_types", s);
        RadBuildCtx ctx = build_ctx();
        RadBuilder x, y;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_bf16::declare(&x, b16.meta(), &ctx); }), want);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&y, f8.meta(), &ctx); }), want);
    }
    /* Flash-Next reads the same key under its own spelling. */
    MetaKV q4(qwen4exp_meta());
    q4.set("layer_types", wrong);
    RadBuildCtx c4 = build_ctx(0, 1);
    c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
    RadBuilder z;
    CHECK_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&z, q4.meta(), &c4); }),
             RAD_E_UNSUPPORTED);
}

/* ==================================================================== qwen35_w8a8 */

namespace {

/* The dense fixture as data/recipes/q38-27b-w8a8-df2.recipe states it: w8a8 / fwht128. */
MetaKV q35w8a8_meta() {
    MetaKV k = q35fp8_meta();
    k.m.quant = "w8a8_fwht128";
    k.m.name  = "test-qwen35-27b-w8a8";
    k.set("quantization_config.quant_method", "w8a8").set("quantization_config.fmt", "fwht128");
    return k;
}

/* The recipe's encodings by declared name: the eight trunk linears int8 (`/fwht128` unless
 * `rotated` is false, which the plugin refuses), an int8 lm_head, and the DFlash2 drafter's MXFP4
 * linears with its 2-bit head. The drafter's keys come first because its names contain the
 * trunk's (dflash.blk.0.attn_k.weight), and the head's last because `output.weight` is in
 * `attn_output.weight`. */
void w8a8_encs(RadBuilder* b, bool rotated = true) {
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    if (rotated) rad_enc_copy_str(i8.transform, "fwht128");
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    b->encs = { { "dflash.draft_head", u2 },
                { "dflash.", rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32) } };
    for (const char* t : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        b->encs.push_back({ t, i8 });
    b->encs.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
}

/* qwen35_w8a8::declare with the recipe's encodings, in lint_qwen35_fp8_family's shape. */
int w8a8_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    w8a8_encs(b);
    return qwen35_w8a8::declare(b, meta, ctx);
}

}  /* namespace */

/* The `_had_quant_i8g` producers, which libr4d serves and libref does not. */
const char* const kW8a8Fused[] = { "rmsnorm_had_quant_i8g", "ar_rmsnorm_had_quant_i8g",
                                   "gated_had_quant_i8g" };

/* Where the `_had_quant_i8g` producers are not served, every trunk linear of a rotated int8
 * container is an i8a8h GEMM right behind its own had_quant_act_i8g, at one rank and two: 2
 * attention layers of four linears, 6 delta-net layers of two and 8 feed-forwards of two. None is
 * left at fp8, which is what a linear that missed the encoding falls back to, and the MLP declares
 * no gated fold (an fp8a8 tile over E4M3 codes). The four trunk activations carry an int8 pair;
 * the delta net's `gdn_in` feeds no int8 linear. */
TEST(qwen35_w8a8_serves_every_trunk_linear_rotated_int8) {
    MetaKV mk = q35w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        for (const char* op : kW8a8Fused) b.undeclared.insert(op);
        w8a8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        int rot = 0, paired = 0;
        for (size_t i = 0; i < b.ops.size(); ++i) {
            const RecOp& o = b.ops[i];
            if (o.op != "gemm_nt_q") continue;
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
            if (pstr(&o, "dtype") != "i8a8h") continue;
            ++rot;
            paired += i > 0 && b.ops[i - 1].op == "had_quant_act_i8g";
        }
        CHECK_EQ(rot, 2 * 4 + 6 * 2 + 8 * 2);
        CHECK_EQ(paired, rot);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
        CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 0);
        for (const char* a : { "h.q8", "attn_out.q8", "gdn_o.q8", "ffn.q8" })
            CHECK(find_b(b, a) != nullptr);
        CHECK(find_b(b, "gdn_in.q8") == nullptr);
    }
}

/* Where they are served, the norms and the gated product write had_quant_act_i8g's codes into the
 * int8 pair, and the projections reading them declare no quantiser: had_quant_act_i8g is left only
 * before the attention and delta-net output projections (2 + 6), every norm is the rotated twin (at
 * two ranks with its fused all-reduce form beside it), and every feed-forward's product is
 * gated_had_quant_i8g. No E4M3 norm or gated quantiser is left, and the 36 GEMMs are unchanged. */
TEST(qwen35_w8a8_feeds_its_rotated_codes_from_the_producers_where_libr4d_serves_them) {
    MetaKV mk = q35w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        w8a8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        int rot = 0;
        for (const RecOp& o : b.ops) rot += o.op == "gemm_nt_q" && pstr(&o, "dtype") == "i8a8h";
        CHECK_EQ(rot, 2 * 4 + 6 * 2 + 8 * 2);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), 2 + 6);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 8 + 8);
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 8);
        for (const char* op : { "rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8", "gated_quant_fp8",
                                "quant_act_i8g" })
            CHECK_EQ(count_op(b, op), 0);
        CHECK_EQ(count_op(b, "ar_rmsnorm_had_quant_i8g") > 0, world == 2);
    }
}

/* What the int8 build refuses: a trunk that is not int8 (no stated encoding, which a block-fp8
 * plugin reads as fp8), a plain int8 trunk (the descriptor names the rotation, and a plain weight
 * served behind the rotated quantiser would be wrong), and a container stated for another format
 * or without the rotation's `fmt`. */
TEST(qwen35_w8a8_refuses_what_is_not_a_rotated_int8_trunk) {
    RadBuildCtx ctx = build_ctx();
    {
        MetaKV mk = q35w8a8_meta();
        RadBuilder b;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_DTYPE);
    }
    {
        MetaKV mk = q35w8a8_meta();
        RadBuilder b;
        w8a8_encs(&b, false);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_DTYPE);
    }
    {
        MetaKV mk = q35w8a8_meta();
        mk.set("quantization_config.quant_method", "fp8").set("quantization_config.fmt", "e4m3");
        RadBuilder b;
        w8a8_encs(&b);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = q35w8a8_meta();
        mk.drop("quantization_config.fmt");
        RadBuilder b;
        w8a8_encs(&b);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* Every engine step lints clean at one and two ranks, with the MTP head's passes and with the
 * DFlash2 drafter's, whose depth the probe fixes at its trained block as qwen35_fp8's does. */
TEST(qwen35_w8a8_steps_lint_clean_with_its_mtp_head_and_a_dflash2_drafter) {
    MetaKV mtp = q35w8a8_meta();
    lint_qwen35_fp8_family(w8a8_declare, qwen35_w8a8::step, mtp.meta(), 3, false, 0);
    MetaKV df = with_dflash2(q35w8a8_meta());
    lint_qwen35_fp8_family(w8a8_declare, qwen35_w8a8::step, df.meta(), 7, true, 8);
    RadArchProbe p{};
    CHECK_OK(qwen35_w8a8::probe(df.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
}

/* ==================================================================== llama_fp8 and DSpark */

namespace {

/* MiniCPM5-2B's graph cut to four layers at a quarter of its width, with the DSpark drafter
 * MiniCPM5-2B-DSpark ships merged under `draft.`: two layers reading trunk layers 0, 2 and 3, a
 * block of seven, the rank-64 bigram head. */
MetaKV llama_meta(const char* quant = "fp8_e4m3") {
    RadModelMeta m{};
    m.arch_id = "llama"; m.name = "test-minicpm5"; m.quant = quant;
    m.n_layers = 4; m.n_embd = 1024; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 2048; m.n_vocab = 32000; m.n_ctx_train = 32768;
    m.rms_eps = 1e-5f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    return MetaKV(m);
}

MetaKV with_dspark(MetaKV k) {
    k.set("draft.architectures", "DSparkDraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "8")
     .set("draft.num_key_value_heads", "2")
     .set("draft.intermediate_size", "2048")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "1000000")
     .set("draft.vocab_size", "32000")
     .set("draft.target_layer_ids", "0 2 3")
     .set("draft.block_size", "7")
     .set("draft.mask_token_id", "31999")
     .set("draft.markov_rank", "64")
     .set("draft.markov_head_type", "vanilla");
    return k;
}

RadBuildCtx dspark_ctx(int rank, int world) {
    RadBuildCtx c = spec_ctx(rank, world, 7);
    c.max_seqs = 16;     /* room for a query pass past 64 rows */
    return c;
}

/* Moves the declarations of `moved`, in their order, to just before `before`, as a plugin that
 * declared them there would have, and renumbers the recorded issues to match. */
void redeclare_before(RadBuilder& b, const std::vector<std::unique_ptr<RadCtx>>& ctxs,
                      const std::vector<rad_op>& moved, rad_op before) {
    std::set<rad_op> mv(moved.begin(), moved.end());
    std::vector<rad_op> order;
    for (rad_op h = 1; h <= (rad_op)b.ops.size(); ++h) {
        if (mv.count(h)) continue;
        if (h == before) order.insert(order.end(), moved.begin(), moved.end());
        order.push_back(h);
    }
    std::vector<rad_op> to(b.ops.size() + 1, 0);
    std::vector<RecOp> ops;
    for (size_t i = 0; i < order.size(); ++i) {
        to[order[i]] = (rad_op)(i + 1);
        ops.push_back(b.ops[order[i] - 1]);
    }
    b.ops = ops;
    for (const auto& c : ctxs)
        for (RecIssue& is : c->issues) is.op = to[is.op];
}

}  /* namespace */

/* The plain LLaMA graph's engine steps lint clean at one and two ranks, with the DSpark drafter's
 * context and query passes behind a verify. Ten sequences at a block of seven is 70 query rows,
 * past the 64 the fused gate_up GEMM serves, so the unfused feed-forward path runs too. */
TEST(llama_fp8_steps_lint_clean_with_a_dspark_drafter) {
    MetaKV mk = with_dspark(llama_meta());
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = dspark_ctx(rank, world);
            REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
            CHECK_EQ(b.drafter.depth, 7);
            CHECK_EQ(b.drafter.block, 7);       /* every row predicts, the anchor included */
            CHECK_EQ(b.drafter.mask_token, 31999);
            const int64_t gated = llama_fp8::g_model[rank].ds.gated_rows;
            for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 40 }),
                                   kit(b, { 8, 1 }, { 5 }) }) {
                const auto v = run_passes(b, llama_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            for (int64_t n_seq : { 4, 10 }) {
                const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
                const auto ctxp = context_pass(b, *ver);
                const auto qry = query_pass(b, n_seq, 7, 7);
                const auto v = run_passes(b, llama_fp8::step, { ver.get(), ctxp.get(), qry.get() },
                                          rank, world);
                CHECK_PASSES(b, v);
                /* The feed-forward of the query pass takes the fold up to its band, the pair past
                 * it. */
                const bool fold = gated > 0 && n_seq * 7 <= gated;
                const auto ops = issued_ops(b, *v[2]);
                CHECK_EQ(count_issued(ops, "silu_mul") > 0, !fold);
            }
        }
}

/* Declaration order is liveness, and DSpark's norms are declared where its step issues them. Moved
 * back to where they once were, all ahead of the layers, the residual stream's planned life ends
 * before the first layer's temporaries begin; past the fused GEMM's 64 rows the unfused
 * feed-forward writes those temporaries, and the lint reports the stream being read after them. */
TEST(the_step_lint_catches_dspark_norms_declared_ahead_of_their_layers) {
    MetaKV mk = with_dspark(llama_meta());
    for (int64_t n_seq : { 4, 10 }) {
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        const rad::arch::Dspark& ds = llama_fp8::g_model[0].ds;
        const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
        const auto ctxp = context_pass(b, *ver);
        const auto qry = query_pass(b, n_seq, 7, 7);
        const auto v = run_passes(b, llama_fp8::step, { ver.get(), ctxp.get(), qry.get() });
        CHECK_PASSES(b, v);

        std::vector<rad_op> norms;
        for (int64_t l = 0; l < ds.g.n_layer; ++l) {
            norms.push_back(ds.op_anorm[l]);
            norms.push_back(ds.op_fnorm[l]);
        }
        norms.push_back(ds.op_outnorm);
        redeclare_before(b, v, norms, ds.op_qat);
        const auto f = lint_step(b, view(v));
        bool stream = false;
        for (const auto& s : f)
            stream |= has(s, "reads 'ds_x'") && (has(s, "'ds_gate_up'") || has(s, "'ds_ffn'"));
        if (n_seq * 7 > ds.gated_rows) CHECK(stream);
        else                           CHECK(!stream);
    }
}

/* A drafter's taps name trunk layers. Out-of-range ids are dropped, not clamped (the drafter
 * belongs to another target), parsing stops at the first non-number, and a drafter naming no
 * layer of this model is refused. So are a trained block other than the serve's depth, a Markov
 * rank the walk cannot apply, and a head type other than the bigram. */
TEST(llama_reads_its_drafters_taps_and_refuses_what_dspark_does_not_implement) {
    {
        MetaKV mk = with_dspark(llama_meta());
        mk.set("draft.target_layer_ids", "[1, 99, 3, -2]");
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(llama_fp8::g_model[0].n_taps, 2);
        CHECK_EQ(llama_fp8::g_model[0].tap_layer[0], 1);
        CHECK_EQ(llama_fp8::g_model[0].tap_layer[1], 3);
        CHECK_EQ(find_b(b, "draft_src")->d.shape[1], 2 * 1024);
    }
    auto refused = [](MetaKV mk, int max_spec = 7) {
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        ctx.max_spec = max_spec;
        return stderr_of_rc([&] { return llama_fp8::declare(&b, mk.meta(), &ctx); });
    };
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.target_layer_ids", "7 9")),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()), 6), RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.markov_rank", "48")), RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.markov_head_type", "mlp")),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).drop("draft.block_size")), RAD_E_FORMAT);
    CHECK_EQ(refused(with_dspark(llama_meta()).drop("draft.mask_token_id")), RAD_E_FORMAT);
    /* Without speculation the drafter is ignored and the trunk declares alone. */
    RadBuilder b;
    MetaKV mk = with_dspark(llama_meta());
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "draft_src") == nullptr);
    /* The probe reports the trained block, fixed. */
    RadArchProbe p{};
    CHECK_OK(llama_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
}

/* One graph, two descriptors: llama_dense_fp8 serves the dense checkpoint a recipe quantises
 * (descriptor ""), llama_fp8 the fp8 one, each refusing the other's, and both declare the same
 * ops and weights. */
TEST(llama_dense_fp8_declares_the_fp8_graph_for_the_dense_descriptor) {
    MetaKV fp8 = with_dspark(llama_meta());
    MetaKV dense = with_dspark(llama_meta(""));
    RadBuildCtx ctx = dspark_ctx(0, 1);
    RadBuilder a, d;
    REQUIRE_EQ(llama_fp8::declare(&a, fp8.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(llama_dense_fp8::declare(&d, dense.meta(), &ctx), RAD_OK);
    CHECK_EQ(join(op_names(a)), join(op_names(d)));
    REQUIRE_EQ(a.weights.size(), d.weights.size());
    for (size_t i = 0; i < a.weights.size(); ++i) {
        CHECK_EQ(a.weights[i].name, d.weights[i].name);
        CHECK_EQ(a.weights[i].d.dtype, d.weights[i].d.dtype);
    }
    RadBuilder x, y;
    CHECK_EQ(stderr_of_rc([&] { return llama_fp8::declare(&x, dense.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(stderr_of_rc([&] { return llama_dense_fp8::declare(&y, fp8.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    /* Its steps are llama_fp8's. A plugin keeps one model per rank, so declare again after the
     * refusal above replaced it. */
    d = RadBuilder{};
    REQUIRE_EQ(llama_dense_fp8::declare(&d, dense.meta(), &ctx), RAD_OK);
    const auto ver = kit(d, std::vector<int>(4, 8), {});
    const auto ctxp = context_pass(d, *ver);
    const auto qry = query_pass(d, 4, 7, 7);
    const auto v = run_passes(d, llama_dense_fp8::step, { ver.get(), ctxp.get(), qry.get() });
    CHECK_PASSES(d, v);
}

/* ==================================================================== qwen4exp_fp8 stepped */

namespace {

RadBuildCtx q4_ctx(int rank, int world, int64_t max_seqs = 8) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_tok = 2048; c.max_seqs = max_seqs; c.max_spec = 3; c.max_ctx = 200000;
    return c;
}

/* The 2-bit draft head's encoding (u2 codes, an f16 scale and a u8 zero per 128), as the recipe
 * writes `mtp.draft_head.weight`. */
RadEncoding u2_head() {
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    return u2;
}

/* The Flash-Next fixture with 32 experts instead of 512: the step and its lint do not depend on
 * the expert count, and declaring 512 experts on every rank is most of a test's time. */
MetaKV q4_small() {
    MetaKV k(qwen4exp_meta());
    k.set("num_experts", "32");
    return k;
}

/* The issues of op `name` in a pass. */
std::vector<const RecIssue*> issues_of(const RadBuilder& b, const RadCtx& c, const char* name) {
    std::vector<const RecIssue*> v;
    for (const auto& i : c.issues)
        if (const RecOp* o = b.op_of(i.op); o && o->op == name) v.push_back(&i);
    return v;
}

}  /* namespace */

/* Flash-Next's engine steps lint clean at one, two and four ranks: decode, a prefill chunk after
 * a long history (the sparse attention), a mixed step, and a verify followed by the MTP head's
 * history pass and rounds, the 2-bit draft head and its vocab-parallel gather included. */
TEST(flashnext_steps_lint_clean_at_one_two_and_four_ranks) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = { { "draft_head", u2_head() } };
            RadBuildCtx ctx = q4_ctx(rank, world);
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
            REQUIRE_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
            for (const auto& k : { kit(b, { 1, 1, 1, 1 }, {}), kit(b, {}, { 64, 64 }, 9000),
                                   kit(b, { 1, 4 }, { 6 }, 300) }) {
                const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            const auto his = history_pass(b, *ver);
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, qwen4exp_fp8::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, rank, world);
            CHECK_PASSES(b, v);
        }
}

/* Sparse attention issues one row per query, chunked by the split-KV scratch it declared, never
 * by `max_seqs`: with four sequences allowed, a 1024-token chunk after a long history is one
 * launch of the gated form, or four of 256 rows where only the plain form resolves. Rows of four
 * would be 256 launches a layer. */
TEST(flashnext_sparse_attention_chunks_by_its_scratch_not_by_max_seqs) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int gated = 0; gated < 2; ++gated) {
        RadBuilder b;
        if (!gated) b.refuse.insert("attn_paged_gate_quant");
        RadBuildCtx ctx = q4_ctx(0, 1, 4);
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
        const auto k = kit(b, {}, { 1024 }, 9000);
        const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        const auto at = issues_of(b, *v[0], gated ? "attn_paged_gate_quant" : "attn_paged");
        /* Two full-attention layers. */
        REQUIRE_EQ(at.size(), (size_t)(gated ? 2 : 2 * 4));
        for (const RecIssue* i : at) {
            CHECK_EQ(i->n, 1);                               /* q_len 1: a row is a query */
            CHECK_EQ(i->opd[0].rows, gated ? 1024 : 256);    /* queries in this launch */
            CHECK_EQ(i->opd[2].rows, i->opd[0].rows);        /* one selection row a query */
        }
    }
}

/* The shared-table read path, the prototype behind RADIANCE_QSA_TABLE=1: with the switch on, the
 * per-tile planes are declared and a sparse step issues `qsa_table` over its rows and `qsa_attn`
 * once for the step, and `qsa_select` grows the block-index operand the table is built from. With
 * the switch unset, not one buffer or op of it is declared -- the control arm of the measurement is
 * the graph this branch shipped before, because both arms have to run in one build. */
TEST(flashnext_reads_attention_through_the_shared_table_only_behind_its_switch) {
    for (int on = 0; on < 2; ++on) {
        if (on) setenv("RADIANCE_QSA_TABLE", "1", 1);
        else unsetenv("RADIANCE_QSA_TABLE");
        RadBuilder b;
        RadBuildCtx ctx = q4_ctx(0, 1, 4);
        MetaKV mk = q4_small();
        const RadModelMeta& meta = *mk.meta();
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
        const bool declared = find_b(b, "qsa_tab") != nullptr;
        CHECK_EQ(declared, on != 0);
        const auto k = kit(b, {}, { 1024 }, 9000);
        const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        /* Two full-attention layers. With the switch on, the step's 1024 rows go through one table
         * per tile of 16 rows -- 64 tiles, one launch -- and one read over all 1024 rows. Not a read
         * per chunk: `cu_seqlens` is the batch's, so a chunk would look its first row up in the first
         * sequence's range, and a declared op's parameters are baked at declare, so the chunk's
         * offset cannot be one. With the switch off, the control arm is what the branch already
         * shipped: the fused gated op over the whole step. */
        const auto tab = issues_of(b, *v[0], "qsa_table");
        const auto rd = issues_of(b, *v[0], "qsa_attn");
        const auto row = issues_of(b, *v[0], "attn_paged");
        const auto gq = issues_of(b, *v[0], "attn_paged_gate_quant");
        CHECK_EQ(tab.size(), on ? (size_t)2 : (size_t)0);
        CHECK_EQ(rd.size(), on ? (size_t)2 : (size_t)0);
        CHECK_EQ(row.size(), (size_t)0);
        CHECK_EQ(gq.size(), on ? (size_t)0 : (size_t)2);
        for (const RecIssue* i : tab) {
            CHECK_EQ(i->opd[0].rows, 1024);                  /* selection rows of the step */
            CHECK_EQ(i->opd[3].rows, 1024 / 16);             /* one table row a tile of 16 */
            CHECK_EQ(i->opd[6].rows, 1024);                  /* every row carries a row_off */
        }
        for (const RecIssue* i : rd) {
            CHECK_EQ(i->opd[0].rows, 1024);                  /* queries of the step */
            CHECK_EQ(i->opd[2].rows, 1024 / 16);             /* the whole tile plane, unsliced */
            CHECK_EQ(i->opd[2].cols, i->opd[4].cols);        /* the mask's u_max is the table's */
        }
        for (const RecIssue* i : gq) CHECK_EQ(i->opd[0].rows, 1024);
        /* The selection's operand list is positional, so the block indices are the ninth operand
         * whether or not the path is on: RAD_NONE keeps the count, and a dropped operand would be
         * a silent shift of everything after it. */
        for (const RecIssue* i : issues_of(b, *v[0], "qsa_select")) {
            REQUIRE_EQ(i->opd.size(), (size_t)9);
            CHECK_EQ(i->opd[8].kind, on ? RAD_OPK_BUF : RAD_OPK_NONE);
        }
        if (on) unsetenv("RADIANCE_QSA_TABLE");
    }
}

/* The indexer's unfinished block is state, not scratch: each QSA layer stores it in the linear
 * `kv_qsa_tail` group at its own layer, so two layers never share one tail and a slot survives
 * across steps. */
TEST(flashnext_qsa_tail_is_per_layer_state) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    rad_kvgroup tail = 0;
    for (size_t g = 0; g < b.kv_groups.size(); ++g)
        if (b.kv_groups[g].name == "kv_qsa_tail") tail = (rad_kvgroup)(g + 1);
    REQUIRE(tail != 0);
    CHECK_EQ((int)b.kv_groups[tail - 1].d.kind, (int)RAD_KV_LINEAR);
    const auto k = kit(b, { 1, 1 }, {});
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    std::set<int64_t> layers;
    for (const RecIssue* i : issues_of(b, *v[0], "qsa_tail_store"))
        for (const RadOperand& o : i->opd)
            if (o.kind == RAD_OPK_KV) {
                CHECK_EQ(o.handle, tail);
                layers.insert(o.offset);
            }
    CHECK(layers == (std::set<int64_t>{ 3, 7 }));
}

/* The rope table has a row for every position a step can name: a verify and the draft rounds run
 * up to max_spec past max_ctx, so a table ending at max_ctx would rotate them by unwritten rows.
 * Every draft pass fills its own rows before anything reads them, since a round's positions are
 * ones the trunk step never wrote. */
TEST(flashnext_rope_table_reaches_past_the_context_and_draft_passes_fill_it_first) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const RecBuf* cs = find_b(b, "rope_cs");
    REQUIRE(cs != nullptr);
    CHECK(cs->d.shape[0] >= ctx.max_ctx + ctx.max_spec);
    const rad_buf h = (rad_buf)buf_named(b, "rope_cs");
    const auto ver = kit(b, std::vector<int>(4, 4), {});
    const auto his = history_pass(b, *ver);
    const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
    const auto v = run_passes(b, qwen4exp_fp8::step, { ver.get(), his.get(), r2.get(), r3.get() });
    CHECK_PASSES(b, v);
    for (size_t p = 1; p < v.size(); ++p) {
        int first_write = -1, first_read = -1;
        for (size_t i = 0; i < v[p]->issues.size(); ++i) {
            const RecIssue& is = v[p]->issues[i];
            const RecOp* o = b.op_of(is.op);
            bool names = false;
            for (const RadOperand& od : is.opd) names |= od.kind == RAD_OPK_BUF && od.handle == h;
            if (!names) continue;
            if (in(o->writes, h) && first_write < 0) first_write = (int)i;
            if (in(o->reads, h) && first_read < 0) first_read = (int)i;
        }
        CHECK(first_write >= 0);
        CHECK(first_read < 0 || first_write < first_read);
        CHECK_EQ(b.op_of(v[p]->issues[0].op)->op, std::string("rope_table"));
    }
}

/* A draft pass refills the rope table before the head's first op, while the carry `mtp_hin` holds
 * the rows the previous pass left, so the table's planned life must reach the carry's or the
 * planner may give the two one set of bytes. Without qk_norm_rope_gate (the fused attention
 * prologue and the indexer's) no head op reads the table, so only the head's own declaration of
 * the refill carries its life there. At one rank and two, with the fused prologue and without, the
 * table is live at the carry's first op and a verify, the history pass and two rounds lint clean. */
TEST(flashnext_draft_passes_refill_the_rope_table_where_the_carry_is_live) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int fused = 0; fused < 2; ++fused)
        for (int world : { 1, 2 }) {
            RadBuilder b;
            if (!fused) b.undeclared.insert("qk_norm_rope_gate");
            RadBuildCtx ctx = q4_ctx(0, world);
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
            CHECK_EQ(count_op(b, "qk_norm_rope_gate") > 0, fused == 1);
            const std::vector<Live> L = liveness(b);
            const rad_buf cs = (rad_buf)buf_named(b, "rope_cs");
            const rad_buf hin = (rad_buf)buf_named(b, "mtp_hin");
            REQUIRE(cs && hin);
            CHECK(L[cs].planned && L[hin].planned);
            CHECK(L[cs].lo <= L[hin].lo && L[hin].lo <= L[cs].hi);
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            const auto his = history_pass(b, *ver);
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, qwen4exp_fp8::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, 0, world);
            CHECK_PASSES(b, v);
        }
}

/* The vocab-parallel draft head gathers each rank's top-R as (id, value) pairs: a row is 2R
 * floats, whole 16-byte words, and the gather moves every draft row's. A row of one pair (8 bytes)
 * is what the two-rank gather cannot take. */
TEST(flashnext_draft_head_gathers_whole_candidate_rows_at_two_ranks) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    b.encs = { { "draft_head", u2_head() } };
    RadBuildCtx ctx = q4_ctx(0, 2);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const RecOp* topk = nullptr;
    for (const auto& o : b.ops)
        if (o.op == "row_topk" && param(&o, "vocab_off")) topk = &o;
    const RecOp* ag = find_op(b, "all_gather");
    REQUIRE(topk != nullptr && ag != nullptr);
    const long long R = pint(topk, "R");
    CHECK(R >= 2);
    CHECK_EQ(pint(ag, "row"), 2 * R);
    CHECK_EQ(pint(ag, "row") * 4 % 16, 0);
    CHECK_EQ(pint(ag, "world_size"), 2);
    const long long rows = ctx.max_seqs * (1 + ctx.max_spec);
    CHECK_EQ(pint(ag, "numel"), rows * 2 * R);
    const RecBuf* pair = find_b(b, "mtp_dpair");
    const RecBuf* gath = find_b(b, "mtp_dgath");
    REQUIRE(pair != nullptr && gath != nullptr);
    CHECK_EQ(pair->d.shape[1] * pair->d.shape[2], 2 * R);
    CHECK_EQ(gath->d.shape[1] * gath->d.shape[2], 2 * 2 * R);
}

/* ==================================================================== embgemma2 */

namespace {

/* EmbeddingGemma 2 cut to six layers (five sliding, one whole-input, its 5:1), with its vision
 * tower and audio encoder cut to two layers each. */
MetaKV eg2_meta() {
    RadModelMeta m{};
    m.arch_id = "embeddinggemma2"; m.name = "test-embgemma2"; m.quant = "";
    m.n_layers = 6; m.n_embd = 512; m.n_head = 4; m.n_head_kv = 2; m.head_dim = 256;
    m.n_ff = 2048; m.n_vocab = 262144; m.n_ctx_train = 2048;
    m.rms_eps = 1e-6f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    MetaKV k(m);
    k.set("layer_types", "sliding_attention sliding_attention sliding_attention sliding_attention "
                         "sliding_attention full_attention")
     .set("rope_parameters.sliding_attention.rope_theta", "10000")
     .set("rope_parameters.full_attention.rope_theta", "1000000")
     .set("sliding_window", "512")
     .set("per_layer_config.05.head_dim", "512")
     .set("per_layer_config.05.num_key_value_heads", "1")
     .set("hidden_size_per_layer_input", "512")
     .set("embedding_dim", "768")
     .set("vision_config.model_type", "gemma4_vision")
     .set("vision_config.hidden_size", "768")
     .set("vision_config.num_attention_heads", "12")
     .set("vision_config.head_dim", "64")
     .set("vision_config.intermediate_size", "3072")
     .set("vision_config.pooling_kernel_size", "3")
     .set("vision_config.position_embedding_size", "10240")
     .set("vision_config.num_hidden_layers", "2")
     .set("vision_config.patch_size", "16")
     .set("vision_config.rope_parameters.rope_theta", "100")
     .set("vision_config.rope_parameters.rope_type", "default")
     .set("vision_config.hidden_activation", "gelu_pytorch_tanh")
     .set("audio_config.model_type", "gemma4_audio")
     .set("audio_config.hidden_size", "1024")
     .set("audio_config.num_attention_heads", "8")
     .set("audio_config.output_proj_dims", "1536")
     .set("audio_config.attention_chunk_size", "12")
     .set("audio_config.attention_context_left", "13")
     .set("audio_config.attention_context_right", "0")
     .set("audio_config.attention_logit_cap", "50")
     .set("audio_config.num_hidden_layers", "2")
     .set("audio_config.conv_kernel_size", "5")
     .set("audio_config.residual_weight", "0.5")
     .set("audio_config.hidden_act", "silu")
     .set("audio_config.subsampling_conv_channels", "128 32")
     .set("preprocessor.feature_size", "128");
    return k;
}

RadBuildCtx eg2_ctx() {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = 9 * 64;
    c.max_audio_frames = 400;
    return c;
}

/* An encoder pass of `n` patches in one segment, every coordinate plane filled. */
struct EncKit {
    std::vector<uint16_t> pix;
    std::vector<int32_t>  coord, cu;
    RadBatch b{};
};
std::unique_ptr<EncKit> enc_pass(int32_t mod, int64_t n, int64_t patch_dim) {
    auto k = std::make_unique<EncKit>();
    k->pix.assign((size_t)(n * patch_dim), 0);
    k->coord.assign((size_t)(4 * n), 1);
    k->cu = { 0, (int32_t)n };
    k->b.enc = mod;
    k->b.enc_n_patch = n;
    k->b.enc_pixels = k->pix.data();
    k->b.enc_coord = k->coord.data();
    k->b.enc_n_seg = 1;
    k->b.enc_cu = k->cu.data();
    k->b.enc_max_seg = (int32_t)n;
    return k;
}

}  /* namespace */

/* An embedding model: a pooled f32 output and no logits, attention bidirectional on every layer
 * (a window on the sliding ones, one more than the reference's radius since attn_dense's bound is
 * strict), each layer's residual gated by its slice of the per-layer inputs and scaled by its
 * learned scalar, and both media encoders declared. */
TEST(embgemma2_declares_a_pooled_encoder_with_both_media_towers) {
    MetaKV mk = eg2_meta();
    RadBuilder b;
    RadBuildCtx ctx = eg2_ctx();
    REQUIRE_EQ(embgemma2::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    REQUIRE(b.pooled != 0);
    const RadBufDecl& p = b.bufs[b.pooled - 1].d;
    CHECK_EQ(p.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(p.shape[0], ctx.max_seqs);
    CHECK_EQ(p.shape[1], 768);
    CHECK_EQ(count_op(b, "logits_gemm"), 0);
    std::vector<long long> win;
    for (const auto& o : b.ops)
        if (o.op == "attn_dense" && pint(&o, "causal") == 0 && pint(&o, "head_dim") >= 256) {
            win.push_back(pint(&o, "window"));
            CHECK_NEAR(pflt(&o, "scale"), 1.0, 0.0);
        }
    CHECK(win == (std::vector<long long>{ 513, 513, 513, 513, 513, 0 }));
    CHECK_EQ(count_op(b, "mul_weight"), 6);       /* the layer scalars */
    CHECK_EQ(count_op(b, "seg_mean"), 1);
    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.max_patches, 400);
    CHECK_EQ(b.audio_encoder.merge, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights) CHECK(find_map(b, w.name) != nullptr);
}

/* What it refuses: a second rank, a container without layer_types or with a kind it does not
 * know or a count that is not the model's, and an audio tower that is not the conformer it
 * implements. */
TEST(embgemma2_refuses_what_it_does_not_serve) {
    RadBuildCtx ctx = eg2_ctx();
    auto rc = [&](MetaKV mk, const RadBuildCtx& c) {
        RadBuilder b;
        return stderr_of_rc([&] { return embgemma2::declare(&b, mk.meta(), &c); });
    };
    CHECK_EQ(rc(eg2_meta(), build_ctx(0, 2)), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().drop("layer_types"), ctx), RAD_E_INVAL);
    CHECK_EQ(rc(eg2_meta().set("layer_types", "sliding_attention sliding_attention "
                                              "sliding_attention sliding_attention "
                                              "sliding_attention chunked_attention"), ctx),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().set("layer_types", "sliding_attention full_attention"), ctx),
             RAD_E_INVAL);
    CHECK_EQ(rc(eg2_meta().set("audio_config.residual_weight", "1.0"), ctx), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().drop("embedding_dim"), ctx), RAD_E_INVAL);
}

/* The text pass over three whole inputs (one with an image's rows scattered in), the vision pass
 * and the audio pass each lint clean, and every GEMM reads whole rows: the audio encoder's clamp
 * copy feeding a GEMM is a buffer of its own width, not a slice of a wider one. */
TEST(embgemma2_text_and_media_passes_lint_clean_and_feed_gemms_whole_rows) {
    MetaKV mk = eg2_meta();
    RadBuilder b;
    RadBuildCtx ctx = eg2_ctx();
    REQUIRE_EQ(embgemma2::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto text = kit(b, {}, { 12, 7, 30 });
    static const int32_t rows[] = { 2, 3, 4 };
    static uint16_t embd[3 * 512];
    text->b.n_mm_rows = 3;
    text->b.mm_rows = rows;
    text->b.mm_embd = embd;
    const auto vis = enc_pass(RAD_MM_IMAGE, 9 * 16, 3 * 16 * 16);
    const auto aud = enc_pass(RAD_MM_AUDIO, 400, 128);
    std::vector<const RadBatch*> passes = { &text->b, &vis->b, &aud->b };
    for (const RadBatch* pb : passes) {
        RadCtx c{ .b = &b, .batch = pb };
        embgemma2::step(&c, pb);
        CHECK(c.step_fail.empty());
        CHECK(!c.issues.empty());
        CHECK_LINT(b, &c);
        if (pb == &text->b) {
            /* the pooled rows, one per input */
            const RecIssue& last = c.issues.back();
            CHECK_EQ(last.n, 3);
            CHECK_EQ(last.opd.back().handle, b.pooled);
        }
    }
    /* A pass past the declared audio frames, or not whole output rows, is refused. */
    for (int64_t n : { 404, 398 }) {
        const auto bad = enc_pass(RAD_MM_AUDIO, n, 128);
        RadCtx c{ .b = &b, .batch = &bad->b };
        embgemma2::step(&c, &bad->b);
        CHECK(!c.step_fail.empty());
        CHECK(c.issues.empty());
    }
}

/* ==================================================================== qwenimage21 */

namespace {

/* Qwen-Image 2.1 with every network cut: a two-layer conditioning language model of 512, a
 * two-block DiT of four 128-wide heads (rotary axes 16, 56, 56), and the residual VAE's own
 * shape. */
MetaKV qimg_meta() {
    RadModelMeta m{};
    m.arch_id = "qwenimage21"; m.name = "test-qwen-image"; m.quant = "";
    m.n_layers = 2; m.n_embd = 512; m.n_head = 4; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 1024; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    std::string mean, sd;
    for (int i = 0; i < 64; ++i) {
        mean += (i ? " " : "") + std::to_string(0.01 * i);
        sd += (i ? " " : "") + std::to_string(1.0 + 0.01 * i);
    }
    k.set("pipeline._class_name", "QwenImage21Pipeline")
     .set("transformer.patch_size", "1")
     .set("transformer.causal_condition", "1")
     .set("transformer.num_layers", "2")
     .set("transformer.attention_head_dim", "128")
     .set("transformer.num_attention_heads", "4")
     .set("transformer.mlp_ratio", "4")
     .set("transformer.in_channels", "64")
     .set("transformer.context_in_dim", "512")
     .set("transformer.axes_dims_rope", "16 56 56")
     .set("vae.is_residual", "1")
     .set("vae.z_dim", "64")
     .set("vae.latents_mean", mean.c_str())
     .set("vae.latents_std", sd.c_str())
     .set("vae.dim_mult", "1 2 4 4 4")
     .set("vae.temperal_downsample", "0 1 1 0");
    return k;
}

/* A generator pass: COND over a text prompt of `n` tokens, DENOISE over `entries` latents of `n`
 * rows each, or DECODE of a 64 x 64 picture. */
struct GenKit {
    std::vector<int32_t> tok, rope, cu, src, seg, seg_h, seg_bidi, seg_cu, seg_start, slot;
    std::vector<int32_t> pair, lane, flags, table, used;
    std::vector<uint32_t> seed;
    std::vector<float> sigma, sigma_next, cfg;
    RadKVGroupBatch kv{};
    RadGenBatch g{};
    RadBatch b{};
};

std::unique_ptr<GenKit> gen_pass(const RadBuilder& bld, int pass, int64_t n, int64_t entries = 1) {
    auto k = std::make_unique<GenKit>();
    const int64_t S = pass == RAD_GEN_PASS_DENOISE ? entries : 1, N = n * S;
    for (int64_t i = 0; i < N; ++i) {
        k->tok.push_back((int32_t)(3 + i % 50));
        k->src.push_back((int32_t)i);
        k->slot.push_back((int32_t)i);
    }
    k->rope.assign((size_t)(3 * N), 0);
    for (int64_t s = 0; s <= S; ++s) k->cu.push_back((int32_t)(s * n));
    k->seg = { 0, (int32_t)n };
    k->seg_h = k->seg;
    k->seg_bidi = { 0 };
    k->seg_cu = { 0, (int32_t)n };
    k->seg_start = { 0 };
    k->sigma.assign((size_t)S, pass == RAD_GEN_PASS_COND ? 0.0f : 0.5f);
    k->sigma_next.assign((size_t)S, 0.25f);
    k->cfg.assign((size_t)S, 1.0f);
    k->pair.assign((size_t)S, -1);
    for (int64_t s = 0; s < S; ++s) k->lane.push_back((int32_t)s);
    k->flags.assign((size_t)S, RAD_GEN_F_FIRST);
    k->seed.assign((size_t)(2 * S), 7u);
    k->table.assign((size_t)(S * 16), 0);
    k->used.assign((size_t)S, (int32_t)n);
    k->kv.group = 1;
    k->kv.slot_mapping = k->slot.data();
    k->kv.block_table = k->table.data();
    k->kv.block_table_pitch = 16;
    k->kv.seqused = k->used.data();
    k->kv.block_size = bld.kv_block;
    k->kv.max_blocks = 16;
    RadGenBatch& g = k->g;
    g.rows = n;
    g.src = k->src.data();
    g.rope = k->rope.data();
    g.n_seg = 1;
    g.seg = k->seg.data();
    g.seg_h = k->seg_h.data();
    g.seg_bidi_h = k->seg_bidi.data();
    g.seg_cu = k->seg_cu.data();
    g.seg_start = k->seg_start.data();
    g.slot_mapping = k->slot.data();
    g.sigma = k->sigma.data();
    g.sigma_next = k->sigma_next.data();
    g.cfg = k->cfg.data();
    g.pair = k->pair.data();
    g.lane = k->lane.data();
    g.lane_dh = k->lane.data();
    g.n_lane = S;
    g.flags = k->flags.data();
    g.seed = k->seed.data();
    g.dec_lane = 0;
    g.dec_height = g.dec_width = 64;
    RadBatch& b = k->b;
    b.gen = pass;
    b.gen_b = &k->g;
    b.n_tok = pass == RAD_GEN_PASS_DECODE ? 16 : N;
    b.n_seq = S;
    b.token_ids = k->tok.data();
    b.rope_pos = k->rope.data();
    b.cu_seqlens = k->cu.data();
    b.n_kv_groups = 1;
    b.kv = &k->kv;
    return k;
}

RadBuildCtx qimg_ctx(int rank, int world, int lossy = 0) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_tok = 512;
    c.max_seqs = 2;
    c.tp_wire_lossy = lossy;
    c.tp_wire_min_bytes = 64 * 1024;
    return c;
}

}  /* namespace */

/* An image generator, not a language model: no logits, the generator declared with the model's
 * prompt and sizes. One modulation projection is shared by every block; the DiT rotates three
 * sections of adjacent pairs from axes 16, 56, 56; the prefix is cached in a full KV group, read
 * causally by text and both ways by a picture. A Turbo container's fixed schedule names it. */
TEST(qwenimage21_declares_a_generator_with_one_shared_modulation) {
    MetaKV mk = qimg_meta();
    RadBuilder b;
    RadBuildCtx ctx = qimg_ctx(0, 1);
    REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK_EQ(b.generator_name, std::string("qwen-image-2.1"));
    CHECK_EQ(b.generator.layout, (int32_t)RAD_GEN_LAYOUT_JOINT_GRID);
    CHECK_EQ(b.generator.latent_channels, 64);
    CHECK_EQ(b.generator.n_sigmas, 0);
    CHECK(b.generator.image_out != 0);
    int mods = 0;
    for (const auto& w : b.weights) mods += has(w.name, "dit.mod.weight");
    CHECK_EQ(mods, 1);
    const rad_buf mod = (rad_buf)buf_named(b, "dit.mod");
    int readers = 0;
    for (const auto& o : b.ops)
        if ((o.op == "ln_mod" || o.op == "gate_add" || o.op == "ln_mod_quant_fp8") &&
            in(o.reads, mod))
            ++readers;
    CHECK(readers >= 2 * 2);          /* both blocks' norms read the one modulation */
    bool dit_rope = false;
    for (const auto& o : b.ops)
        if (pstr(&o, "sections") == "8 28 28") dit_rope = true;
    CHECK(dit_rope);
    /* Per block: the causal and the bidirectional form, and (fp8 linears) the denoising pass's
     * bidirectional form with int8 scores. */
    int causal = 0, bidi = 0, bidi8 = 0;
    for (const auto& o : b.ops)
        if (o.op == "attn_prefix_self")
            (pint(&o, "causal") ? causal : pstr(&o, "qk") == "i8" ? bidi8 : bidi)++;
    CHECK_EQ(causal, 2);
    CHECK_EQ(bidi, 2);
    CHECK_EQ(bidi8, 2);
    REQUIRE_EQ(b.kv_groups.size(), (size_t)1);
    CHECK_EQ((int)b.kv_groups[0].d.kind, (int)RAD_KV_FULL);

    MetaKV turbo = qimg_meta();
    turbo.set("pipeline.sample_sigmas", "1.0 0.9 0.75 0.6 0.45 0.3 0.15 0.05");
    RadBuilder bt;
    REQUIRE_EQ(qwenimage21::declare(&bt, turbo.meta(), &ctx), RAD_OK);
    CHECK_EQ(bt.generator_name, std::string("qwen-image-2.1-turbo"));
    CHECK_EQ(bt.generator.n_sigmas, 8);
    CHECK_EQ(bt.generator.default_steps, 8);
}

/* Every collective follows --tp-wire: exact unless the lossy wire was accepted, and the floor
 * under which a message stays exact is the deployment's. */
/* The DiT's adapter names: one root before any GEMM, and every alias lands on a checkpoint tensor
 * the plugin maps, with the gate half of ComfyUI's `gate_up` on `gate_layer`. */
TEST(qwenimage21_names_its_adapter_keys_onto_mapped_tensors) {
    MetaKV mk = qimg_meta();
    RadBuilder b;
    const RadBuildCtx ctx = qimg_ctx(0, 1);
    REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(b.lora_roots.size(), (size_t)1);
    CHECK_EQ(b.lora_roots[0].first, std::string("transformer."));
    CHECK(b.lora_roots[0].second.find("lora_unet_") != std::string::npos);
    std::set<std::string> srcs;
    for (const RecMap& m : b.maps) for (const std::string& x : m.src) srcs.insert(x);
    REQUIRE_EQ(b.lora_aliases.size(), (size_t)2);
    for (const auto& a : b.lora_aliases) {
        std::string to = a.native + a.to;
        const size_t h = to.find('#');
        if (h != std::string::npos) to.replace(h, 1, "0");
        CHECK(srcs.count(to + ".weight"));
    }
    CHECK(b.lora_aliases[0].to.find("gate_layer") != std::string::npos);
    CHECK_EQ(b.lora_aliases[0].from_lo, (int64_t)0);
    CHECK_EQ(b.lora_aliases[1].from_lo, b.lora_aliases[0].from_hi);
}

TEST(qwenimage21_collectives_follow_the_tp_wire) {
    MetaKV mk = qimg_meta();
    for (int lossy : { 0, 1 }) {
        RadBuilder b;
        RadBuildCtx ctx = qimg_ctx(0, 2, lossy);
        REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
        int n = 0;
        for (const auto& o : b.ops) {
            if (o.op != "all_reduce") continue;
            ++n;
            CHECK_EQ(pint(&o, "exact"), lossy ? 0 : 1);
            CHECK_EQ(pint(&o, "min_bytes"), 64 * 1024);
        }
        CHECK(n >= 2 * 2 + 2 * 2);      /* two a layer in each network */
    }
}

/* The three passes the core drives (a text prompt's conditioning, denoising one image or two at
 * once, decoding one) lint clean at one and two ranks, the split all-reduces included. */
TEST(qwenimage21_generator_passes_lint_clean) {
    MetaKV mk = qimg_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = qimg_ctx(rank, world);
            REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
            for (const auto& k : { gen_pass(b, RAD_GEN_PASS_COND, 12),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16, 2),
                                   gen_pass(b, RAD_GEN_PASS_DECODE, 0) }) {
                RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = world };
                qwenimage21::step(&c, &k->b);
                CHECK(c.step_fail.empty());
                CHECK(!c.issues.empty());
                CHECK_EQ(c.lane, 0);
                CHECK_LINT(b, &c);
            }
            /* A token step is not a generator pass. */
            const auto tok = kit(b, { 1 }, {});
            RadCtx c{ .b = &b, .batch = &tok->b, .rank = rank, .world = world };
            qwenimage21::step(&c, &tok->b);
            CHECK(!c.step_fail.empty());
        }
}

/* ==================================================================== anima */

namespace {

/* Anima's text encoder (Qwen3-0.6B-Base) as the container's config states it; the DiT, adapter
 * and VAE are the plugin's own constants. */
MetaKV anima_meta(const char* variant) {
    RadModelMeta m{};
    m.arch_id = "anima"; m.name = "test-anima"; m.quant = "";
    m.n_layers = 28; m.n_embd = 1024; m.n_head = 16; m.n_head_kv = 8; m.head_dim = 128;
    m.n_ff = 3072; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 1e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    k.set("num_hidden_layers", "28").set("hidden_size", "1024").set("anima.variant", variant);
    return k;
}

RadBuildCtx anima_ctx() {
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2 * (4096 + 512);
    c.max_seqs = 4;
    return c;
}

/* A JOINT_STEP pass as the driver lays one out for Anima: COND over a prompt of `nq` Qwen3 tokens
 * (padded to 512) and `nt` T5 ids, DENOISE over `entries` entries of 512 context rows and `ni`
 * image rows each, or DECODE of a 64 x 64 picture. */
struct AnimaKit {
    std::vector<int32_t> tok, pos, cu, src, lane, flags, pair, aux, aux_cu, valid_cu;
    std::vector<int32_t> img_cu, img_rows, mode_cu, mode_rows, mode_src, mode_ccu;
    std::vector<int64_t> n_mode;
    std::vector<uint32_t> seed;
    std::vector<float> sigma, sigma_next, cfg, rope_f, mode_rope_f;
    RadGenBatch g{};
    RadBatch b{};
};

std::unique_ptr<AnimaKit> anima_pass(int pass, int64_t nq = 12, int64_t nt = 9, int64_t ni = 16,
                                     int64_t entries = 1) {
    auto k = std::make_unique<AnimaKit>();
    RadGenBatch& g = k->g;
    RadBatch& b = k->b;
    b.gen = pass;
    b.gen_b = &k->g;
    if (pass == RAD_GEN_PASS_COND) {
        for (int64_t i = 0; i < 512; ++i) {
            k->tok.push_back(i < nq ? (int32_t)(100 + i) : 151643);
            k->pos.push_back(i < nq ? (int32_t)i : 1);
        }
        for (int64_t i = 0; i < nt; ++i) k->aux.push_back((int32_t)(3 + i));
        k->aux_cu = { 0, (int32_t)nt };
        k->valid_cu = { 0, (int32_t)nq };
        k->cu = { 0, 512 };
        k->lane = { 0, 1 };
        k->sigma = { 0.0f };
        b.n_tok = 512;
        b.n_seq = 1;
        b.token_ids = k->tok.data();
        b.positions = k->pos.data();
        b.cu_seqlens = k->cu.data();
        g.rows = 512;
        g.n_pad = (int32_t)(512 - nq);
        g.valid_cu = k->valid_cu.data();
        g.n_lane = 2;
        g.lane = k->lane.data();
        g.lane_h = k->lane.data();
        g.branch = 1;
        g.sigma = k->sigma.data();
        g.n_aux = nt;
        g.aux_ids = k->aux.data();
        g.aux_cu = k->aux_cu.data();
        return k;
    }
    if (pass == RAD_GEN_PASS_DECODE) {
        k->cu = { 0, 16 };
        b.n_tok = 16;
        b.n_seq = 1;
        b.cu_seqlens = k->cu.data();
        g.dec_lane = 1;
        g.dec_height = g.dec_width = 64;
        return k;
    }
    const int64_t S = entries, N = S * (512 + ni);
    k->cu.push_back(0);
    k->img_cu.push_back(0);
    for (int64_t s = 0; s < S; ++s) {
        k->mode_cu.push_back((int32_t)(s * (512 + ni)));
        k->mode_cu.push_back((int32_t)(s * (512 + ni) + 512));
        k->mode_cu.push_back((int32_t)(s * (512 + ni) + 512));
        for (int64_t r = 0; r < 512; ++r) k->src.push_back((int32_t)(s * 512 + r));
        for (int64_t r = 0; r < ni; ++r) {
            k->src.push_back((int32_t)(-1 - (s * ni + r)));
            k->img_rows.push_back((int32_t)(s * (512 + ni) + 512 + r));
        }
        k->cu.push_back((int32_t)((s + 1) * (512 + ni)));
        k->img_cu.push_back((int32_t)((s + 1) * ni));
        k->lane.push_back((int32_t)(s / 2));
        k->pair.push_back(S > 1 ? (s % 2 ? -2 : (int32_t)(s + 1)) : -1);
    }
    k->mode_cu.push_back((int32_t)N);
    for (int64_t s = 0; s < S; ++s)
        for (int64_t r = 0; r < 512; ++r) {
            k->mode_rows.push_back((int32_t)(s * (512 + ni) + r));
            k->mode_src.push_back((int32_t)(s * 512 + r));
        }
    for (int64_t s = 0; s < S; ++s)
        for (int64_t r = 0; r < ni; ++r) {
            k->mode_rows.push_back((int32_t)(s * (512 + ni) + 512 + r));
            k->mode_src.push_back((int32_t)(-1 - (s * ni + r)));
        }
    k->n_mode = { 512 * S, 0, ni * S };
    for (int m = 0; m < 3; ++m)
        for (int64_t s = 0; s <= S; ++s)
            k->mode_ccu.push_back((int32_t)(s * (m == 0 ? 512 : m == 1 ? 0 : ni)));
    k->rope_f.assign((size_t)(4 * N), 0.0f);
    k->mode_rope_f.assign((size_t)(4 * N), 0.0f);
    k->tok.assign((size_t)N, 0);
    k->pos.assign((size_t)N, 0);
    k->sigma.assign((size_t)S, 0.75f);
    k->sigma_next.assign((size_t)S, 0.5f);
    k->cfg.assign((size_t)S, 4.0f);
    k->flags.assign((size_t)S, RAD_GEN_F_FIRST);
    k->seed.assign((size_t)(2 * S), 7u);
    b.n_tok = N;
    b.n_seq = S;
    b.token_ids = k->tok.data();
    b.positions = k->pos.data();
    b.cu_seqlens = k->cu.data();
    g.src = k->src.data();
    g.n_img = ni * S;
    g.img_cu = k->img_cu.data();
    g.img_rows = k->img_rows.data();
    g.mode_cu = k->mode_cu.data();
    g.rope_f = k->rope_f.data();
    g.mode_rows = k->mode_rows.data();
    g.mode_src = k->mode_src.data();
    g.mode_ccu = k->mode_ccu.data();
    g.n_mode_h = k->n_mode.data();
    g.mode_rope_f = k->mode_rope_f.data();
    g.sigma = k->sigma.data();
    g.sigma_next = k->sigma_next.data();
    g.cfg = k->cfg.data();
    g.pair = k->pair.data();
    g.lane = k->lane.data();
    g.lane_dh = k->lane.data();
    g.flags = k->flags.data();
    g.seed = k->seed.data();
    return k;
}

/* The linears a LoRA for Anima names: every DiT block's attention, MLP and modulation linears
 * and the text encoder's. */
bool anima_lora_target(const std::string& w) {
    for (const char* t : { "attn_qkv", "attn_output", "cross_q", "cross_k", "cross_v",
                           "cross_output", "mlp_layer1", "mlp_layer2", ".mod_", "ffn_gate_up",
                           "ffn_down" })
        if (has(w, t) && !has(w, ".scale")) return true;
    return false;
}

}  /* namespace */

/* A JOINT_STEP generator that reads a second tokenizer, pads an empty prompt to one token and
 * samples on ComfyUI's simple schedule; the variant sets its defaults and the DiT keys' prefix is
 * the checkpoint's, `model.diffusion_model.` here and `net.` for Base. */
TEST(anima_declares_a_two_tokenizer_generator_under_the_checkpoints_prefix) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("aesthetic");
    RadBuilder b;
    b.encs = { { "dit.t_norm.mdm", plain }, { "dit.blk.0.attn_qkv", plain } };
    REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK_EQ(b.generator_name, std::string("anima-aesthetic"));
    const RadGeneratorDecl& d = b.generator;
    CHECK_EQ(d.layout, (int32_t)RAD_GEN_LAYOUT_JOINT_STEP);
    CHECK_EQ(d.aux_tokens, 1);
    CHECK_EQ(d.aux_max_tokens, (int64_t)512);
    CHECK_EQ(d.min_tokens, 1);
    CHECK_EQ(d.pad_multiple, 512);
    CHECK_EQ(d.cond_rows, (int64_t)512);
    CHECK_EQ(d.sigma_grid, 1000);
    CHECK_EQ(d.shift, 3.0);
    CHECK_EQ(d.dyn_shift, 0);
    CHECK_EQ(d.cfg_original, 0);
    CHECK_EQ(d.default_steps, (int64_t)30);
    CHECK_EQ(d.default_cfg, 4.0);
    CHECK_EQ(d.latent_channels, (int64_t)64);
    CHECK(b.kv_groups.empty());
    const RecMap* q = find_map(b, "dit.blk.27.attn_qkv.weight");
    REQUIRE(q != nullptr);
    REQUIRE_EQ(q->src.size(), (size_t)3);
    CHECK_EQ(q->src[0], std::string("model.diffusion_model.blocks.27.self_attn.q_proj.weight"));
    const RecMap* te = find_map(b, "te.blk.3.ffn_gate_up.weight");
    REQUIRE(te != nullptr);
    CHECK_EQ(te->src[1], std::string("model.layers.3.mlp.up_proj.weight"));
    /* The patch embedding over its first 64 columns, the final linear's rows in eight pieces. */
    const RecMap* xe = find_map(b, "dit.x_embed.weight");
    REQUIRE(xe != nullptr);
    CHECK_EQ(xe->mode, (int)RAD_MAP_COLS);
    int final_maps = 0;
    for (const auto& m : b.maps)
        if (m.declared == "dit.final.weight") { ++final_maps; CHECK_EQ(m.mode, (int)RAD_MAP_ROWS); }
    CHECK_EQ(final_maps, 8);
    /* The DiT's rotary: 22, 21 and 21 pairs, NeoX, over (frame, row, column). */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "qk_norm_rope" && pstr(&o, "mode") == "axial") {
            CHECK_EQ(pstr(&o, "sections"), std::string("22 21 21"));
            ++axial;
        }
    CHECK_EQ(axial, 28);

    MetaKV turbo = anima_meta("turbo");
    RadBuilder bt;
    bt.encs = { { "dit.t_norm.mdm", plain } };
    REQUIRE_EQ(anima::declare(&bt, turbo.meta(), &ctx), RAD_OK);
    CHECK_EQ(bt.generator_name, std::string("anima-turbo"));
    CHECK_EQ(bt.generator.default_steps, (int64_t)10);
    CHECK_EQ(bt.generator.default_cfg, 1.0);

    MetaKV base = anima_meta("base");
    RadBuilder bb;
    bb.encs = { { "dit.t_norm.net", plain } };
    REQUIRE_EQ(anima::declare(&bb, base.meta(), &ctx), RAD_OK);
    const RecMap* nq = find_map(bb, "dit.blk.0.mlp_layer1.weight");
    REQUIRE(nq != nullptr);
    CHECK_EQ(nq->src[0], std::string("net.blocks.0.mlp.layer1.weight"));

    /* Neither spelling, or an unknown variant, is refused. */
    RadBuilder none;
    CHECK(anima::declare(&none, mk.meta(), &ctx) < 0);
    MetaKV odd = anima_meta("deluxe");
    RadBuilder bo;
    bo.encs = { { "dit.t_norm.mdm", plain } };
    CHECK(anima::declare(&bo, odd.meta(), &ctx) < 0);
}

/* Every linear a LoRA for Anima names is read by a plain GEMM a runtime LoRA can follow: gemm_nt
 * in a bf16 container, gemm_nt_q without rotated parts in a block-fp8 one, its activation a
 * separate op. */
TEST(anima_lora_targets_are_plain_gemms) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("base");
    for (bool q : { false, true }) {
        RadBuilder b;
        b.encs = { { "dit.t_norm.net", plain } };
        if (!q) b.encs.push_back({ "dit.blk.0.attn_qkv", plain });
        if (q)
            for (const char* t : { "attn_qkv", "attn_output", "cross_q", "cross_k", "cross_v",
                                   "cross_output", "mlp_layer1", "mlp_layer2", "ffn_gate_up",
                                   "ffn_down" })
                b.encs.push_back({ t, fp8 });
        REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
        int targets = 0;
        for (const auto& o : b.ops)
            for (rad_weight w : o.w) {
                const std::string& n = b.weights[w - 1].name;
                if (!anima_lora_target(n)) continue;
                ++targets;
                const bool ok = o.op == "gemm_nt" || (q && o.op == "gemm_nt_q" &&
                                                      pint(&o, "pb1") <= 0 && pint(&o, "pb2") <= 0);
                if (!ok) fprintf(stderr, "    '%s' read by %s\n", n.c_str(), o.op.c_str());
                CHECK(ok);
                CHECK(o.op != "gemm_nt_bias");
            }
        /* 28 blocks of 8 linears (two views a block-fp8 one) and 6 modulation linears; 28 text
         * encoder layers of 4. */
        CHECK(targets >= 28 * (8 + 6) + 28 * 4);
    }
}

/* The three passes the core drives (a conditioning with its T5 ids, denoising one image or a
 * guided pair, decoding one) lint clean in a bf16 and a block-fp8 container. */
TEST(anima_generator_passes_lint_clean) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("aesthetic");
    for (bool q : { false, true }) {
        RadBuilder b;
        b.encs = { { "dit.t_norm.mdm", plain } };
        if (!q) b.encs.push_back({ "dit.blk.0.attn_qkv", plain });
        if (q) b.encs.push_back({ "attn_qkv", fp8 }), b.encs.push_back({ "attn_output", fp8 }),
               b.encs.push_back({ "cross_", fp8 }), b.encs.push_back({ "mlp_layer", fp8 }),
               b.encs.push_back({ "ffn_", fp8 });
        REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
        for (const auto& k : { anima_pass(RAD_GEN_PASS_COND, 12, 9),
                               anima_pass(RAD_GEN_PASS_COND, 3, 40),
                               anima_pass(RAD_GEN_PASS_DENOISE, 12, 9, 16, 1),
                               anima_pass(RAD_GEN_PASS_DENOISE, 12, 9, 16, 2),
                               anima_pass(RAD_GEN_PASS_DECODE) }) {
            RadCtx c{ .b = &b, .batch = &k->b, .rank = 0, .world = 1 };
            anima::step(&c, &k->b);
            CHECK(c.step_fail.empty());
            if (!c.step_fail.empty()) fprintf(stderr, "    %s\n", c.step_fail.c_str());
            CHECK(!c.issues.empty());
            CHECK_LINT(b, &c);
        }
        const auto tok = kit(b, { 1 }, {});
        RadCtx c{ .b = &b, .batch = &tok->b, .rank = 0, .world = 1 };
        anima::step(&c, &tok->b);
        CHECK(!c.step_fail.empty());
    }
}

/* ======================================================== the conventional vocabulary */

namespace {

/* libref's vocabulary: every op its schemas name. A CPU-only engine, or a new hardware library
 * that implements only what libref does, declares exactly these (libavx's are a subset). Empty
 * when libref.so is not under $RADIANCE_HOME/kernels (ctest sets it to the build's plugin home). */
const std::set<std::string>& libref_vocab() {
    static std::set<std::string> v;
    static bool tried = false;
    if (tried) return v;
    tried = true;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home) return v;
    const std::string path = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "    cannot load %s: %s\n", path.c_str(), dlerror());
        return v;
    }
    auto count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    auto at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    if (count && at)
        for (int i = 0; i < count(); ++i)
            if (const RadOpSchema* s = at(i); s && s->op) v.insert(s->op);
    return v;
}

/* One architecture and configuration. `pinned` lists the device-only ops it may still ask for on
 * libref's vocabulary: an upper bound that may only shrink. An op libref gains drops out of the
 * expectation by itself; one the plugin stops asking for must be removed here. */
struct ArchCase {
    std::string name;
    std::function<int(RadBuilder*)> declare;
    StepFn step = nullptr;
    int max_spec = 0;
    std::vector<std::string> pinned;
    /* The declared graph's fingerprint with every op declared (graph_hash). */
    uint64_t graph = 0;
};

/* Every architecture in the tree, at the fixtures above. */
struct ArchCases {
    MetaKV q35 = q35fp8_meta(), q35df = with_dflash2(q35fp8_meta()), q35moe = q35moe_meta();
    MetaKV q35w8df = with_dflash2(q35w8a8_meta());
    MetaKV lla = with_dspark(llama_meta()), eg2 = eg2_meta(), qimg = qimg_meta();
    MetaKV q4s = q4_small();
    RadModelMeta q35b = qwen35_meta(), q4 = *q4s.meta(), lfm = lfm2_meta();
    RadModelMeta g4 = gemma4_meta(), g4m = gemma4_moe_meta(), g4df = gemma4_moe_meta();
    std::vector<const char*> g4dk = g4df_keys(), g4dv = g4df_vals();
    RadEncoding w4{}, w4g{}, fp8{};
    std::vector<ArchCase> list;

    ArchCases() {
        g4df.n_kv = (int)g4dk.size(); g4df.kv_key = g4dk.data(); g4df.kv_val = g4dv.data();
        w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
        rad_enc_copy_str(w4.transform, "fwht128");
        w4g = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
        fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
        auto sc = [](int world, int max_spec) { return spec_ctx(0, world, max_spec); };
        auto g4w4 = [this](RadBuilder* b) {
            b->encs = { { "attn_q.", w4g }, { "attn_k.", w4g }, { "attn_v.", w4g },
                        { "attn_o.", w4g }, { "ffn_gate_up.", w4g }, { "ffn_down.", w4g } };
        };
        /* The same with q, k and v as one linear, as a recipe naming attn_qkv converts it. */
        auto g4qkv = [this](RadBuilder* b) {
            b->encs = { { "attn_qkv.", w4g }, { "attn_o.", w4g }, { "ffn_gate_up.", w4g },
                        { "ffn_down.", w4g } };
        };
        /* The ops libref lacks that each still asks for, both two-rank fusions with no chain:
         * qwen35_fp8's all-reduce norm (layer_fold assumes it at two ranks) and Flash-Next's
         * all-reduce into the wide stream. */
        auto add = [&](const char* name, std::function<int(RadBuilder*)> d, StepFn step,
                       int max_spec, std::vector<std::string> pinned, uint64_t graph) {
            list.push_back({ name, std::move(d), step, max_spec, std::move(pinned), graph });
        };
        add("qwen35_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return qwen35_bf16::declare(b, &q35b, &c); },
            qwen35_bf16::step, 0, {}, 0xf82117051023e608ull);
        add("qwen35_fp8 + MTP", [this, sc](RadBuilder* b) {
                auto c = sc(1, 3); return qwen35_fp8::declare(b, q35.meta(), &c); },
            qwen35_fp8::step, 3, {}, 0x74fd4495028c1525ull);
        add("qwen35_fp8 + MTP, two ranks", [this, sc](RadBuilder* b) {
                auto c = sc(2, 3); return qwen35_fp8::declare(b, q35.meta(), &c); },
            nullptr, 3, { "ar_rmsnorm_quant_fp8" }, 0x1c28909125c12f52ull);
        add("qwen35_fp8 + DFlash2", [this, sc](RadBuilder* b) {
                auto c = sc(1, 7); return qwen35_fp8::declare(b, q35df.meta(), &c); },
            qwen35_fp8::step, 7, {}, 0x76bc2f92b452df49ull);
        add("qwen35_w8a8 + DFlash2", [this, sc](RadBuilder* b) {
                auto c = sc(1, 7); return w8a8_declare(b, q35w8df.meta(), &c); },
            qwen35_w8a8::step, 7, {}, 0x5570932de623edcaull);
        add("qwen35_w8a8 + DFlash2, two ranks", [this, sc](RadBuilder* b) {
                auto c = sc(2, 7); return w8a8_declare(b, q35w8df.meta(), &c); },
            nullptr, 7, { "ar_rmsnorm_quant_fp8" }, 0xec1ca3ad670311bbull);
        add("qwen35moe_fp8 + MTP", [this, sc](RadBuilder* b) {
                auto c = sc(1, 3); return qwen35moe_fp8::declare(b, q35moe.meta(), &c); },
            qwen35moe_fp8::step, 3, {}, 0x1bd3b53543cd06eeull);
        add("qwen4exp_fp8", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 1); return qwen4exp_fp8::declare(b, &q4, &c); },
            qwen4exp_fp8::step, 3, {}, 0xbe74d476c7fd04b8ull);
        add("qwen4exp_fp8, two ranks", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 2); return qwen4exp_fp8::declare(b, &q4, &c); },
            nullptr, 3, { "ar_hc_write" }, 0x9d7df7914590d6d8ull);
        add("qwen4exp_fp8, four-bit experts", [this](RadBuilder* b) {
                b->encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
                auto c = q4_ctx(0, 1); return qwen4exp_fp8::declare(b, &q4, &c); },
            qwen4exp_fp8::step, 3, {}, 0xb308d06b168a61f1ull);
        add("llama_fp8 + DSpark", [this](RadBuilder* b) {
                auto c = dspark_ctx(0, 1); return llama_fp8::declare(b, lla.meta(), &c); },
            llama_fp8::step, 7, {}, 0xdcf36a1d31aba73cull);
        add("lfm2_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return lfm2_bf16::declare(b, &lfm, &c); },
            lfm2_bf16::step, 0, {}, 0x1775b188dc4a5120ull);
        /* The first three moved only by the attn_qkv name map each layer declares to ask whether
         * the container holds q, k and v as one linear: with those maps left out of the hash they
         * are the earlier pins, 0xafc23087c634a537, 0x719fa07788f8b98a and 0x93b174ab52cc0d31. */
        add("gemma4_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x2ba2f4943031468full);
        add("gemma4_bf16, w4a8", [this, sc, g4w4](RadBuilder* b) {
                g4w4(b); auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x2e4c6e3235dd3802ull);
        add("gemma4_bf16, w4a8, two ranks", [this, sc, g4w4](RadBuilder* b) {
                g4w4(b); auto c = sc(2, 0); return gemma4_bf16::declare(b, &g4, &c); },
            nullptr, 0, {}, 0xade9c0ccfad1fd47ull);
        add("gemma4_bf16, w4a8, fused qkv", [this, sc, g4qkv](RadBuilder* b) {
                g4qkv(b); auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x3913aca59733d0e2ull);
        add("gemma4_bf16, w4a8, fused qkv, two ranks", [this, sc, g4qkv](RadBuilder* b) {
                g4qkv(b); auto c = sc(2, 0); return gemma4_bf16::declare(b, &g4, &c); },
            nullptr, 0, {}, 0x8a9935e773f546efull);
        add("gemma4", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return gemma4::declare(b, &g4m, &c); },
            gemma4::step, 0, {}, 0x0f227f5853b6488cull);
        add("gemma4, fp8 experts", [this, sc](RadBuilder* b) {
                b->encs = { { "ffn_gate_up_exps", fp8 } };
                auto c = sc(1, 0); return gemma4::declare(b, &g4m, &c); },
            gemma4::step, 0, {}, 0x532d06c46f2e7488ull);
        add("gemma4 + DFlash", [this, sc](RadBuilder* b) {
                auto c = sc(1, 15); return gemma4::declare(b, &g4df, &c); },
            gemma4::step, 15, {}, 0x37bffcc9bd749b88ull);
        add("embgemma2", [this](RadBuilder* b) {
                auto c = eg2_ctx(); return embgemma2::declare(b, eg2.meta(), &c); },
            embgemma2::step, 0, {}, 0xac72c705bc97d2b3ull);
        add("qwenimage21", [this](RadBuilder* b) {
                auto c = qimg_ctx(0, 1); return qwenimage21::declare(b, qimg.meta(), &c); },
            nullptr, 0, {}, 0xd7990ce44671b171ull);
    }
};

ArchCases& arch_cases() {
    static ArchCases c;
    return c;
}

/* The engine steps a model runs: decode, a prefill chunk, a mixed step, and with a drafter a
 * verify followed by its passes, each run on its own contexts. */
std::vector<std::vector<std::unique_ptr<RadCtx>>> model_steps(RadBuilder& b, StepFn step,
                                                               int max_spec) {
    std::vector<std::vector<std::unique_ptr<RadCtx>>> out;
    for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 24 }), kit(b, { 1 }, { 6 }) })
        out.push_back(run_passes(b, step, { k.get() }));
    if (b.drafter.kind == RAD_DRAFT_NONE) return out;
    const auto ver = kit(b, std::vector<int>(3, 1 + max_spec), {});
    std::vector<std::unique_ptr<Kit>> ps;
    if (b.drafter.kind == RAD_DRAFT_SERIAL) {
        ps.push_back(history_pass(b, *ver));
        for (int r = 2; r <= b.drafter.depth; ++r) ps.push_back(round_pass(b, 3, r));
    } else {
        ps.push_back(context_pass(b, *ver));
        ps.push_back(query_pass(b, 3, b.drafter.block, b.drafter.proposal_pitch));
    }
    std::vector<const Kit*> all = { ver.get() };
    for (const auto& p : ps) all.push_back(p.get());
    out.push_back(run_passes(b, step, all));
    return out;
}

std::string joined(const std::set<std::string>& s) {
    std::string r;
    for (const auto& x : s) r += (r.empty() ? "" : " ") + x;
    return r.empty() ? std::string("(none)") : r;
}

}  /* namespace */

/* For every architecture, declaring with only libref's vocabulary either asks for nothing else and
 * then issues only libref's ops (a CPU-only engine serves it), or asks for exactly the pinned
 * device-only ops libref lacks. A new fused op declared without a chain, or a block that starts
 * calling a device-only op, fails here rather than at a CPU-only engine's startup, where a
 * request for an op no library declares stops the declare. */
TEST(each_architecture_declares_on_librefs_vocabulary_or_names_its_device_only_ops) {
    const std::set<std::string>& vocab = libref_vocab();
    if (vocab.empty()) {
        fprintf(stderr, "    SKIP: libref.so not found under $RADIANCE_HOME/kernels\n");
        return;
    }
    for (const ArchCase& ac : arch_cases().list) {
        RadBuilder b;
        b.vocab = &vocab;
        const int rc = stderr_of_rc([&] { return ac.declare(&b); });
        const std::set<std::string> asked(b.asked_undeclared.begin(), b.asked_undeclared.end());
        std::set<std::string> want;
        for (const auto& op : ac.pinned) {
            if (vocab.count(op))
                fprintf(stderr, "    note: %s: libref now declares '%s'; drop it from the pin\n",
                        ac.name.c_str(), op.c_str());
            else
                want.insert(op);
        }
        if (asked != want)
            fprintf(stderr, "    %s asks for: %s\n    pinned: %s\n", ac.name.c_str(),
                    joined(asked).c_str(), joined(want).c_str());
        CHECK(asked == want);
        if (!asked.empty()) continue;
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK || !ac.step) continue;
        for (const auto& v : model_steps(b, ac.step, ac.max_spec)) {
            CHECK_PASSES(b, v);
            for (const auto& c : v)
                for (const auto& op : issued_ops(b, *c))
                    if (!vocab.count(op)) {
                        fprintf(stderr, "    %s issues '%s'\n", ac.name.c_str(), op.c_str());
                        CHECK(vocab.count(op));
                    }
        }
    }
}

/* With every op declared, as an engine with libr4d sees it, each architecture declares its pinned
 * graph: the conventional-vocabulary fallbacks are taken only where a fused op is not declared or
 * does not resolve, so they leave these values alone. A snapshot (the fingerprint of the recorded
 * declare), since nothing independent describes a whole graph. */
TEST(with_every_op_declared_each_architecture_declares_its_unchanged_graph) {
    for (const ArchCase& ac : arch_cases().list) {
        RadBuilder b;
        const int rc = stderr_of_rc([&] { return ac.declare(&b); });
        CHECK_EQ(rc, RAD_OK);
        const uint64_t h = graph_hash(b);
        if (h != ac.graph)
            fprintf(stderr, "    %s: graph 0x%016llx, pinned 0x%016llx\n", ac.name.c_str(),
                    (unsigned long long)h, (unsigned long long)ac.graph);
        CHECK_EQ(h, ac.graph);
    }
}

/* Where gate_quant_fp8 is not declared, or declared and not resolved, the fp8 gated attention runs
 * the chain it fuses: the per-head sigmoid into a gate buffer of the layer's own, the multiply
 * into the attention output, and quant_act_fp8 into the codes the output projection reads, the
 * three buffers the fused op writes. */
TEST(the_fp8_output_gate_runs_its_chain_where_gate_quant_fp8_is_not_served) {
    MetaKV mk = q35fp8_meta();
    for (int how = 0; how < 2; ++how) {
        RadBuilder b;
        if (how == 0) b.undeclared.insert("gate_quant_fp8");
        else          b.refuse.insert("gate_quant_fp8");
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_resolved(b, "gate_quant_fp8"), 0);
        CHECK_EQ(count_op(b, "sigmoid"), 2);
        CHECK_EQ(count_op(b, "mul"), 2);
        const rad_buf ax = (rad_buf)buf_named(b, "attn_out");
        const rad_buf aq = (rad_buf)buf_named(b, "attn_out.q");
        const rad_buf as = (rad_buf)buf_named(b, "attn_out.s");
        REQUIRE(ax && aq && as);
        CHECK(find_b(b, "blk.3.attn_gate") != nullptr);
        CHECK(find_b(b, "blk.7.attn_gate") != nullptr);
        const auto k = kit(b, { 1, 1 }, {});
        const auto v = run_passes(b, qwen35_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        /* attention, then gate, multiply and quantiser, then the out projection, per layer */
        const std::vector<std::string> ops = issued_ops(b, *v[0]);
        int chains = 0;
        for (size_t i = 0; i + 4 < ops.size(); ++i)
            if (ops[i] == "attn_paged") {
                CHECK_EQ(ops[i + 1], std::string("sigmoid"));
                CHECK_EQ(ops[i + 2], std::string("mul"));
                CHECK_EQ(ops[i + 3], std::string("quant_act_fp8"));
                CHECK_EQ(ops[i + 4], std::string("gemm_nt_q"));
                const RecIssue& q = v[0]->issues[i + 3];
                CHECK_EQ(q.opd[0].handle, ax);
                CHECK_EQ(q.opd[1].handle, aq);
                CHECK_EQ(q.opd[2].handle, as);
                ++chains;
            }
        CHECK_EQ(chains, 2);
    }
}

/* Where gated_had_quant_fp8 is not served, a four-bit routed down projection's input is made by
 * silu_mul into the bf16 plane and had_quant_act_fp8 from it, into the rotated codes the fused op
 * writes. */
TEST(the_rotated_expert_gate_runs_its_chain_where_gated_had_quant_fp8_is_not_served) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    const RadEncoding w4 = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                rad_enc_copy_str(e.transform, "fwht128"); return e; }();
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    b.undeclared.insert("gated_had_quant_fp8");
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(b.asked_undeclared.empty());
    CHECK_EQ(count_op(b, "gated_had_quant_fp8"), 0);
    /* One a routed layer (eight trunk layers and the MTP head), from the product into the down
     * projection's codes. */
    const rad_buf eff = (rad_buf)buf_named(b, "moe_eff"), effq = (rad_buf)buf_named(b, "moe_eff.q");
    int chained = 0;
    for (const auto& o : b.ops)
        if (o.op == "had_quant_act_fp8" && in(o.reads, eff)) {
            ++chained;
            CHECK_EQ(pint(&o, "group"), 128);
            CHECK(in(o.writes, effq));
        }
    CHECK_EQ(chained, 9);
    const auto k = kit(b, { 1, 1 }, { 6 });
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    const std::vector<std::string> ops = issued_ops(b, *v[0]);
    int pairs = 0;
    for (size_t i = 0; i + 1 < ops.size(); ++i)
        pairs += ops[i] == "silu_mul" && ops[i + 1] == "had_quant_act_fp8";
    CHECK(pairs >= 8);
}

/* What each plugin's probe tells the engine before declare: a drafter's depth (an MTP head's
 * operating point, a block drafter's trained block, fixed), an encoder's pooling window and audio
 * budget, a generator's widest pass. */
TEST(the_probes_size_what_declare_needs) {
    {   /* the routed and dense fp8 plugins share the MTP head's depth of 3, not fixed */
        MetaKV moe = q35moe_meta();
        RadArchProbe p{};
        CHECK_OK(qwen35moe_fp8::probe(moe.meta(), &p));
        CHECK_EQ(p.draft_depth, 3);
        CHECK_EQ(p.draft_depth_fixed, 0);
        MetaKV none = q35moe_meta();
        none.drop("qwen35.nextn_predict_layers");
        RadArchProbe q{};
        CHECK_OK(qwen35moe_fp8::probe(none.meta(), &q));
        CHECK_EQ(q.draft_depth, 0);
    }
    {   /* the dense llama build reads DSpark's trained block like llama_fp8 */
        MetaKV d = with_dspark(llama_meta(""));
        RadArchProbe p{};
        CHECK_OK(llama_dense_fp8::probe(d.meta(), &p));
        CHECK_EQ(p.draft_depth, 7);
        CHECK_EQ(p.draft_depth_fixed, 1);
    }
    {   /* the vision tower pools 3 x 3; 30 s of audio at a 10 ms hop is 3000 frames, whole rows of
         * four */
        MetaKV e = eg2_meta();
        RadArchProbe p{};
        CHECK_OK(embgemma2::probe(e.meta(), &p));
        CHECK_EQ(p.enc_merge, 9);
        CHECK_EQ(p.audio_frames, 3000);
    }
    {   /* two entries (an image and its guidance) of the largest image's 16800 latent rows; ten
         * references of 1024 x 1024 pixels at 16 x 16 a patch, with a tenth for rounding */
        MetaKV q = qimg_meta();
        RadArchProbe p{};
        CHECK_OK(qwenimage21::probe(q.meta(), &p));
        CHECK_EQ(p.gen_rows, 2 * 16800);
        CHECK_EQ(p.gen_patches, 10 * 4096 * 11 / 10);
    }
}

/* ==================================================================== rad_arch.h's helpers */

/* The vocabulary split, swept: every rank's slice starts where the previous ends, the slices sum
 * to the vocabulary, a split that divides is even, and one that does not cuts only at whole
 * 128-row tiles (a block-fp8 lm_head's tile), the remainder going to the last rank with rows. A
 * rank can be left empty (301 rows over four ranks: 128, 128, 45, 0), and geom_from refuses that
 * world rather than declare a zero-row head. */
TEST(the_vocabulary_split_tiles_the_vocabulary) {
    int bad = 0;
    for (int64_t total : { 1, 7, 127, 128, 129, 300, 301, 1000, 32000, 151669, 248320, 262144 })
        for (int world = 1; world <= 8; ++world) {
            int64_t at = 0;
            for (int r = 0; r < world; ++r) {
                const int64_t lo = rad::arch::shard_begin(total, world, r);
                const int64_t n = rad::arch::shard_extent(total, world, r);
                const bool ok = lo == at && n >= 0 && lo + n <= total &&
                                (total % world == 0 ? n == total / world
                                                    : (n % 128 == 0 || lo + n == total));
                if (!ok) {
                    ++bad;
                    fprintf(stderr, "    total %lld world %d rank %d: [%lld, +%lld)\n",
                            (long long)total, world, r, (long long)lo, (long long)n);
                }
                at = lo + n;
            }
            if (at != total) ++bad;
        }
    CHECK_EQ(bad, 0);
    CHECK_EQ(rad::arch::shard_extent(301, 4, 2), 45);
    CHECK_EQ(rad::arch::shard_extent(301, 4, 3), 0);
    CHECK_EQ(rad::arch::shard_extent(300, 4, 3), 75);
    CHECK_EQ(rad::arch::shard_extent(248320, 3, 2), 82688);
    rad::arch::Geom g{};
    RadModelMeta m = lfm2_meta();
    m.n_vocab = 301;
    m.n_head = 4; m.n_head_kv = 4;
    RadBuildCtx c = build_ctx(3, 4);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::geom_from(g, &m, &c, "bf16", RAD_BF16); }),
             RAD_E_INVAL);
}

/* A layer schedule: names map to full (1) or linear (0), an unknown name is refused by its
 * index, a list shorter than the trunk is refused, and entries past the trunk (an MTP block's)
 * are not compared. */
TEST(the_layer_schedule_readers) {
    std::vector<int> k;
    CHECK_OK(rad::arch::parse_layer_kinds("t", "[\"a\", \"b\", \"a\"]", { "b" }, "a", &k));
    CHECK(k == (std::vector<int>{ 0, 1, 0 }));
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::parse_layer_kinds("t", "a c", { "b" }, "a", &k); }),
             RAD_E_UNSUPPORTED);
    CHECK(rad::arch::is_full_attention(3, 4) && rad::arch::is_full_attention(7, 4));
    CHECK(!rad::arch::is_full_attention(4, 4) && !rad::arch::is_full_attention(3, 0));
    const char* lin = "linear_attention", * ful = "full_attention";
    const std::string four = std::string(lin) + " " + lin + " " + lin + " " + ful;
    CHECK_OK(rad::arch::check_layer_schedule("t", four.c_str(), 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", (four + " " + ful).c_str(), 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", nullptr, 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", "", 4, 4, { ful }));
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::check_layer_schedule("t", four.c_str(), 8, 4, { ful }); }),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::check_layer_schedule("t", four.c_str(), 4, 2, { ful }); }),
             RAD_E_UNSUPPORTED);
}

/* A drafter's taps: out-of-range ids are dropped, not clamped; parsing stops at the first token
 * that is not a number; at most `max` are kept. */
TEST(parse_layer_ids_keeps_the_trunks_layers_in_order) {
    int out[4] = {};
    CHECK_EQ(rad::arch::parse_layer_ids("1 10 20 30 39", 40, out, 4), 4);
    CHECK(out[0] == 1 && out[1] == 10 && out[2] == 20 && out[3] == 30);
    CHECK_EQ(rad::arch::parse_layer_ids("[3, 64, -1, 5]", 40, out, 4), 2);
    CHECK(out[0] == 3 && out[1] == 5);
    CHECK_EQ(rad::arch::parse_layer_ids("2 x 5", 40, out, 4), 1);
    CHECK_EQ(rad::arch::parse_layer_ids(nullptr, 40, out, 4), 0);
    CHECK_EQ(rad::arch::parse_layer_ids("", 40, out, 4), 0);
}

/* The remainder order is a permutation of the ranks for every rotation: attention-zero ranks
 * take places 0.. first, then the attention ranks rotated by `parity`. */
TEST(remainder_place_orders_every_rank_once) {
    for (int world : { 1, 2, 3, 4, 6 })
        for (int attn_world : { world, world - 1 }) {
            if (attn_world < 1) continue;
            rad::arch::Geom g{};
            g.world = world;
            g.attn_world = attn_world;
            for (int parity = 0; parity < world; ++parity) {
                std::set<int64_t> seen;
                for (int r = 0; r < world; ++r) {
                    const int64_t p = rad::arch::remainder_place(g, r, parity);
                    seen.insert(p);
                    if (r >= attn_world) CHECK_EQ(p, (int64_t)(r - attn_world));
                }
                CHECK_EQ((int)seen.size(), world);
                CHECK_EQ(*seen.begin(), 0);
                CHECK_EQ(*seen.rbegin(), (int64_t)(world - 1));
            }
        }
}

/* divide_or_fail refuses rather than rounds, a world of zero included; batch_split takes a pure
 * step's split from its phase and a mixed step's from its fields. */
TEST(divide_or_fail_and_batch_split) {
    int64_t q = -1;
    CHECK_OK(rad::arch::divide_or_fail(48, 4, "x", &q));
    CHECK_EQ(q, 12);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::divide_or_fail(48, 5, "x", &q); }), RAD_E_INVAL);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::divide_or_fail(48, 0, "x", &q); }), RAD_E_INVAL);
    RadBatch b{};
    b.n_seq = 5; b.n_tok = 40; b.n_seq_decode = 2; b.n_tok_decode = 7;
    int64_t s = -1, t = -1;
    b.phase = RAD_PHASE_PREFILL; rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 0 && t == 0);
    b.phase = RAD_PHASE_DECODE;  rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 5 && t == 40);
    b.phase = RAD_PHASE_MIXED;   rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 2 && t == 7);
}

/* Gemma 4's assistant head keeps no cache of its own: its history pass is one row per drafting
 * sequence, gathered from the trunk's verify rows, and each later round one row per sequence. The
 * trunk's verify and the head's passes, linted as the one engine step they are, at one and two
 * ranks. */
TEST(gemma4_assistant_head_engine_step_lints_clean) {
    RadModelMeta meta = gemma4_draft_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, 3);
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            auto his = kit(b, { 1, 1, 1, 1 }, {});
            his->b.draft_pass = -1;
            his->out = { 3, 7, 11, 15 };               /* each sequence's last verify row */
            his->b.out_ids = his->out.data();
            his->b.n_out = 4;
            his->dout = { 0, 1, 2, 3 };
            his->b.draft_out_ids = his->dout.data();
            his->b.n_draft_out = 4;
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, gemma4_bf16::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, rank, world);
            CHECK_PASSES(b, v);
        }
}

/* gemma4_bf16 asks for its fused prologue and int8 junctions only where a library declares them:
 * on libref's and libavx's vocabulary a w4a8 (compressed-tensors) model declares the unfused
 * chains and asks for no op outside it, at one rank and at two, issuing what it issues when the
 * fused ops are declared and do not resolve. With them declared nothing changes (the graph
 * fingerprints above). */
TEST(gemma4_bf16_without_the_fused_ops_declares_the_chains_and_asks_for_no_undeclared_op) {
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    const std::set<std::string> fused = { "qkv_norm_rope", "qkv_norm_rope_kv_store", "rmsnorm_quant_i8",
                                          "gelu_mul_quant_i8", "ar_rmsnorm_quant_i8" };
    for (int world : { 1, 2 }) {
        std::vector<std::string> seq[2];
        for (int undeclared = 0; undeclared < 2; ++undeclared) {
            RadBuilder b;
            b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
                       { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
            (undeclared ? b.undeclared : b.refuse) = fused;
            RadModelMeta meta = gemma4_meta();
            RadBuildCtx ctx = build_ctx(0, world);
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            CHECK(b.asked_undeclared.empty());
            for (const auto& op : fused) CHECK_EQ(count_resolved(b, op), 0);
            const auto k = kit(b, { 1, 1, 1, 1 }, {});
            const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(b, v);
            seq[undeclared] = issued_ops(b, *v[0]);
        }
        CHECK_EQ(join(seq[1]), join(seq[0]));
        CHECK(count_issued(seq[1], "quant_act_i8") > 0);
    }
}

/* ==================================================================== qwen3asr */

/* Qwen3-ASR 1.7B's shape, its decoder and its audio tower each cut to TWO layers. */
const char* k_asr_keys[] = {
    "audio_config.model_type", "audio_config.d_model", "audio_config.encoder_attention_heads",
    "audio_config.encoder_ffn_dim", "audio_config.downsample_hidden_size",
    "audio_config.num_mel_bins", "audio_config.encoder_layers", "audio_config.n_window",
    "audio_config.n_window_infer", "audio_config.output_dim", "audio_config.num_key_value_heads",
    "audio_config.activation_function", "audio_token_id",
};
const char* k_asr_vals[] = { "qwen3_asr_encoder", "1024", "16", "4096", "480", "128", "2", "50",
                             "800", "2048", "16", "gelu", "151676" };

RadModelMeta qwen3asr_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen3asr";
    m.name        = "test-qwen3-asr-1.7b";
    m.quant       = "bf16";
    m.n_layers    = 2;
    m.n_embd      = 2048;
    m.n_head      = 16;
    m.n_head_kv   = 8;
    m.head_dim    = 128;
    m.n_ff        = 6144;
    m.n_vocab     = 151936;
    m.n_ctx_train = 65536;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_asr_keys) / sizeof(k_asr_keys[0]));
    m.kv_key      = k_asr_keys;
    m.kv_val      = k_asr_vals;
    return m;
}

/* The decoder is plain Qwen3 and the tower is declared as the audio encoder: GQA 2 paged attention
 * with NeoX rotary over the whole head and a tied head; the two 480-channel convolutions as unfold
 * plus GEMM against the checkpoint's kernels read as matrices; positions a second at a time and
 * attention in blocks of 104 rows read at cu_div 8. */
TEST(qwen3asr_declares_a_qwen3_decoder_and_its_audio_tower) {
    RadModelMeta meta = qwen3asr_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen3asr_bf16::probe(&meta, &pb));
    CHECK_EQ(pb.audio_frames, (int64_t)3120);

    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = pb.audio_frames;
    CHECK_OK(qwen3asr_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.modalities, (uint32_t)RAD_MM_AUDIO);
    CHECK_EQ(b.audio_encoder.patch_dim, 128);
    CHECK_EQ(b.audio_encoder.merge, 8);
    CHECK_EQ(b.audio_encoder.max_patches, 3120);
    CHECK_EQ(b.audio_encoder.n_embd, 2048);

    for (const auto& o : b.ops)
        if (o.op == "attn_paged") {
            CHECK_EQ(pint(&o, "gqa"), 2);
            CHECK_EQ(pint(&o, "head_dim"), 128);
            CHECK_EQ(pint(&o, "causal"), 1);
        }
    int rope = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope") {
            ++rope;
            CHECK_EQ(pstr(&o, "mode"), std::string("neox"));
            CHECK_EQ(pint(&o, "rotary_dim"), 128);
        }
    CHECK_EQ(rope, 4);
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr);
    REQUIRE_EQ(head->src.size(), (size_t)2);
    CHECK_EQ(head->src[1], std::string("model.language_model.embed_tokens.weight"));

    CHECK_EQ(count_op(b, "conv2d"), 1);
    CHECK_EQ(count_op(b, "conv2d_unfold"), 3);
    CHECK_EQ(count_op(b, "seg_sinusoid"), 1);
    const RecWeight* c2 = find_w(b, "a.conv2.w");
    REQUIRE(c2 != nullptr);
    CHECK_EQ(c2->d.rank, 2u);
    CHECK_EQ(c2->d.shape[0], 480);
    CHECK_EQ(c2->d.shape[1], 4320);
    const RecMap* qkv = find_map(b, "a.blk.1.qkv.w");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[2], std::string("model.audio_tower.layers.1.self_attn.v_proj.weight"));
    int dense = 0;
    for (const auto& o : b.ops)
        if (o.op == "attn_dense") {
            ++dense;
            CHECK_EQ(pint(&o, "block"), 104);
            CHECK_EQ(pint(&o, "cu_div"), 8);
            CHECK_EQ(pint(&o, "head_dim"), 64);
        }
    CHECK_EQ(dense, 2);
    for (const auto& o : b.ops)
        if (o.op == "conv2d_unfold" || o.op == "conv2d")
            CHECK_EQ(pint(&o, "t_last"), pint(&o, "kh") == 3 ? 1 : 0);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* How it hears: the clip's markers and a transcription turn with every slot. */
    const RadAudioFormat* f = qwen3asr_bf16::audio_format(&meta);
    REQUIRE(f != nullptr);
    CHECK_EQ(f->struct_size, (uint32_t)sizeof(RadAudioFormat));
    CHECK_EQ(std::string(f->audio_open), std::string("<|audio_start|>"));
    CHECK(std::string(f->transcribe_prompt).find("{audio}") != std::string::npos);
    CHECK(std::string(f->transcribe_prompt).find("{context}") != std::string::npos);
    CHECK_EQ(f->n_languages, 30u);
}

/* An audio pass runs the tower, in order, and reads the core's planes: the first convolution and the
 * unfolds take plane 1 as their mask, the positions and the attention the pass's clip bounds. */
TEST(a_qwen3asr_audio_pass_issues_the_tower_and_nothing_else) {
    RadModelMeta meta = qwen3asr_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 3120;
    CHECK_OK(qwen3asr_bf16::declare(&b, &meta, &ctx));

    /* One clip of two whole seconds: 208 staged frames, 26 rows. */
    static uint16_t frames[208 * 128];
    static int32_t  coord[4 * 208];
    static const int32_t cu[] = { 0, 208 };
    RadBatch batch{};
    batch.enc         = RAD_MM_AUDIO;
    batch.enc_n_patch = 208;
    batch.enc_pixels  = frames;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 1;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 208;
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
    qwen3asr_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "conv2d", "conv2d_unfold", "gemm_nt_bias", "conv2d_unfold",
                                      "gemm_nt_bias", "conv2d_unfold", "gemm_nt", "seg_sinusoid",
                                      "add" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    const RecIssue* c1 = issue_of(b, c, "conv2d");
    REQUIRE(c1 != nullptr);
    CHECK_EQ(c1->n, 208);
    CHECK(c1->opd[3].raw == coord + 208);
    /* The second convolution's GEMM runs over 52 time rows of 32 positions. */
    const RecIssue* g2 = issue_of(b, c, "gemm_nt_bias");
    REQUIRE(g2 != nullptr);
    CHECK_EQ(g2->n, 52 * 32);
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    CHECK_EQ(at->n, 26);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(c.issues.back().n, 26);
}

/* What this tower does not carry is refused by name. */
TEST(qwen3asr_refuses_a_tower_it_does_not_serve) {
    static const char* vals[] = { "qwen3_asr_encoder", "1024", "16", "4096", "480", "128", "2",
                                  "50", "800", "2048", "16", "silu", "151676" };
    RadModelMeta meta = qwen3asr_meta();
    meta.kv_val = vals;
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 3120;
    CHECK_EQ(qwen3asr_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    /* No audio budget: the decoder alone, and no tower. */
    RadModelMeta ok = qwen3asr_meta();
    RadBuilder bt;
    RadBuildCtx text = build_ctx();
    CHECK_OK(qwen3asr_bf16::declare(&bt, &ok, &text));
    CHECK(bt.audio_encoder_name.empty());
    CHECK_EQ(count_op(bt, "conv2d"), 0);
}

RAD_TEST_MAIN()

/* Without --prefill-span a 4096 budget declares and steps the pinned graph and launches: for
 * Flash-Next and the other plugins built on the same fp8 blocks, at one, two and four ranks, taken
 * from the plugins as they were before the span. */
TEST(a_4096_budget_without_a_span_is_the_pinned_graph_and_steps) {
    MetaKV q4s = q4_small(), q35 = q35fp8_meta(), q35moe = q35moe_meta();
    struct Case {
        const char* name;
        int world, rank;
        uint64_t graph, steps;
    };
    const Case cases[] = {
        /* The steps moved once, when `qsa_select` grew its ninth operand (`selb`, the same entries
         * as block indices, RAD_NONE without the shared-table path): operands are positional, so
         * the issued list is one longer at every issue. The graph is unchanged. */
        { "qwen4exp_fp8", 1, 0, 0x14b9cab0bea2591full, 0x81e3b5c2d26a3062ull },
        { "qwen4exp_fp8", 2, 0, 0xc9bcbf1b538b597full, 0xf16ad463d8e97197ull },
        { "qwen4exp_fp8", 2, 1, 0x8b2f1786a4ac12b8ull, 0x56edd03c1af81c97ull },
        { "qwen4exp_fp8", 4, 0, 0x1e97a26b759d29dfull, 0x664a14802019ed6bull },
        { "qwen4exp_fp8", 4, 3, 0x6669aa0573931943ull, 0x4238f06998784f93ull },
        { "qwen35_fp8", 1, 0, 0xcfc82ca915ec549dull, 0x4d6fca823d1e97efull },
        { "qwen35_fp8", 2, 1, 0x36da8d762194e24eull, 0xe4022daff02ec8bdull },
        { "qwen35moe_fp8", 1, 0, 0x7fa7b015e1bc8873ull, 0x3fcf71676e558d79ull },
        { "qwen35moe_fp8", 2, 0, 0xf60c9508dd197370ull, 0x05e0778848bf416cull },
    };
    for (const Case& cs : cases) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(cs.rank, cs.world, 3);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 200000;
        StepFn step = nullptr;
        int rc = RAD_E_INVAL;
        if (!std::strcmp(cs.name, "qwen4exp_fp8")) {
            b.encs = { { "draft_head", u2_head() } };
            rc = stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, q4s.meta(), &ctx); });
            step = qwen4exp_fp8::step;
        } else if (!std::strcmp(cs.name, "qwen35_fp8")) {
            rc = stderr_of_rc([&] { return qwen35_fp8::declare(&b, q35.meta(), &ctx); });
            step = qwen35_fp8::step;
        } else {
            rc = stderr_of_rc([&] { return qwen35moe_fp8::declare(&b, q35moe.meta(), &ctx); });
            step = qwen35moe_fp8::step;
        }
        REQUIRE_EQ(rc, RAD_OK);
        const uint64_t g = graph_hash(b);
        const uint64_t s = pinned_steps(b, step, cs.rank, cs.world);
        fprintf(stderr, "    pin: %s world %d rank %d: graph 0x%016llxull, steps 0x%016llxull\n",
                cs.name, cs.world, cs.rank, (unsigned long long)g, (unsigned long long)s);
        CHECK_EQ(g, cs.graph);
        CHECK_EQ(s, cs.steps);
    }
}

/* Under a span a Flash-Next prefill of two intervals runs a pass at a time. At every world its
 * connection reads take a pass each and the indexer scores its pass-tall plane in windows, the MTP
 * head's pass included. At four ranks the step is pipelined (step_pipelined): every layer's
 * all-reduce carries one pass's rows on the second lane at `overlap` 1 and the delta net runs each
 * piece over its own bounds, while the embedding's collective stays whole, on the second lane at
 * no overlap. At two ranks the gated residual's fused write takes the layers' all-reduces and the
 * step runs on one lane. */
TEST(flashnext_a_spanned_prefill_runs_a_pass_at_a_time_with_its_all_reduces_beside_it) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = { { "draft_head", u2_head() } };
            RadBuildCtx ctx = q4_ctx(rank, world);
            ctx.max_tok = 4096;
            ctx.max_mixed_tok = 2048;
            REQUIRE_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, &meta, &ctx); }),
                       RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            CHECK(m.span_on);
            CHECK_EQ(m.pipe_on, world > 2);
            CHECK_EQ(m.qsacfg.plane_rows, 2048);
            const auto pf = kit(b, {}, { 4096 }, 4096);
            const auto his = history_pass(b, *pf);
            const auto v = run_passes(b, qwen4exp_fp8::step, { pf.get(), his.get() }, rank, world);
            CHECK_PASSES(b, v);
            const RadCtx& c = *v[0];
            CHECK(c.step_fail.empty());
            /* The MTP head's pass over the same rows reads and scores a pass at a time too. */
            CHECK(v[1]->step_fail.empty());
            for (const RecIssue& i : v[1]->issues)
                if (const RecOp* o = b.op_of(i.op); o && (o->op == "hc_read" || o->op == "qsa_score"))
                    CHECK(i.n <= 2048);
            const int64_t piece = 2048 * m.g.n_embd;
            int reads = 0, wide_reads = 0, ar = 0, ar_lane0 = 0, ar_piece = 0, ar_overlap = 0;
            int score = 0, wide_score = 0, conv = 0, wide_conv = 0;
            for (const RecIssue& i : c.issues) {
                const RecOp* o = b.op_of(i.op);
                if (!o) continue;
                if (o->op == "hc_read") { ++reads; wide_reads += i.n > 2048; }
                if (o->op == "qsa_score") { ++score; wide_score += i.n > 2048; }
                if (o->op == "gdn_conv_prep") { ++conv; wide_conv += i.n > 2048; }
                if (o->op == "all_reduce") {
                    ++ar;
                    ar_lane0 += i.lane == 0;
                    ar_piece += i.n == piece;
                    ar_overlap += pint(o, "overlap") == 1;
                }
            }
            CHECK(reads > 0);
            CHECK_EQ(wide_reads, 0);
            CHECK(score > 0);
            CHECK_EQ(wide_score, 0);
            CHECK(conv > 0);
            if (world < 4) {
                CHECK_EQ(ar, world > 1 ? 1 : 0);
                CHECK_EQ(ar_lane0, ar);
                CHECK(c.joins.empty());
                continue;
            }
            CHECK_EQ(wide_conv, 0);
            CHECK(ar > 2);
            CHECK_EQ(ar_lane0, 0);
            CHECK_EQ(ar_piece, ar - 1);
            CHECK_EQ(ar_overlap, ar - 1);
            CHECK(!c.joins.empty());
        }
}
