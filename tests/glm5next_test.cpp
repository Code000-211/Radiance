/* glm5next_test.cpp -- the declared graph and the steps of GLM-5.3-Flash's plugin and its blocks,
 * asserted statically against the recording builder of arch_fakes.h. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include <array>
#include <dlfcn.h>
#include <map>
#include <thread>

/* ==================================================================== the plugin under test */

/* GLM-5.3-Flash: its blocks (through the plugin's includes) and the plugin. */
#include "../arch/glm5next/glm5next.cpp"

/* ==================================================================== the blocks */

TEST(manifold_connections_preserve_saved_coefficients_and_use_a_separate_residual_destination) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.max_tok = 32;
    g.world = 4;
    ManifoldConn::Config cfg;
    ManifoldConn::Src src{ "model.layers.3.hc_attn_fn", "model.layers.3.hc_attn_scale",
                           "model.layers.3.hc_attn_base" };
    ManifoldConn::Wire w;
    w.h = decl_b(&b, "h", RAD_BF16, { 32, 16384 });
    w.x = decl_b(&b, "x", RAD_BF16, { 32, 4096 });
    w.y = decl_b(&b, "y", RAD_BF16, { 32, 4096 });
    w.out = decl_b(&b, "out", RAD_BF16, { 32, 16384 });
    w.post = decl_b(&b, "post", RAD_F32, { 32, 4 });
    w.comb = decl_b(&b, "comb", RAD_F32, { 32, 16 });
    ManifoldConn conn;
    REQUIRE_EQ(conn.declare(&b, nm, g, 3, "blk.3.mhc_attn", cfg, w, src), RAD_OK);
    REQUIRE_EQ(b.weights.size(), (size_t)3);
    CHECK_EQ(b.weights[0].d.shape[0], 24);
    CHECK_EQ(b.weights[0].d.shape[1], 16384);
    CHECK_EQ(b.weights[1].d.shape[0], 3);
    CHECK_EQ(b.weights[2].d.shape[0], 24);
    for (const auto& weight : b.weights) {
        CHECK_EQ(weight.d.dtype, RAD_F32);
        CHECK_EQ(weight.d.shard, RAD_SHARD_NONE);
    }
    CHECK(b.ops[conn.op_pre - 1].reads == std::vector<rad_buf>({ w.h }));
    CHECK(b.ops[conn.op_pre - 1].writes == std::vector<rad_buf>({ w.x, w.post, w.comb }));
    CHECK(b.ops[conn.op_post - 1].reads == std::vector<rad_buf>({ w.y, w.h, w.post, w.comb }));
    CHECK(b.ops[conn.op_post - 1].writes == std::vector<rad_buf>({ w.out }));
    RadCtx ctx;
    conn.read(&ctx, 4);
    conn.write(&ctx, 4);
    REQUIRE_EQ(ctx.issues.size(), (size_t)2);
    CHECK_EQ(ctx.issues[0].op, conn.op_pre);
    CHECK_EQ(ctx.issues[1].op, conn.op_post);
    CHECK_EQ(ctx.issues[0].n, 4);
    CHECK_EQ(ctx.issues[1].n, 4);
    REQUIRE_EQ(ctx.issues[0].opd.size(), (size_t)7);
    CHECK_EQ(ctx.issues[0].opd[0].handle, w.h);
    const rad_weight coefficients[] = { conn.w_fn, conn.w_scale, conn.w_base };
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(ctx.issues[0].opd[i + 1].kind, RAD_OPK_WEIGHT);
        CHECK_EQ(ctx.issues[0].opd[i + 1].handle, coefficients[i]);
    }
    CHECK_EQ(ctx.issues[0].opd[4].handle, w.x);
    CHECK_EQ(ctx.issues[0].opd[5].handle, w.post);
    CHECK_EQ(ctx.issues[0].opd[6].handle, w.comb);
    REQUIRE_EQ(ctx.issues[1].opd.size(), (size_t)5);
    const rad_buf expected[] = { w.y, w.h, w.post, w.comb, w.out };
    for (int i = 0; i < 5; ++i) {
        CHECK_EQ(ctx.issues[1].opd[i].handle, expected[i]);
        CHECK_EQ(ctx.issues[1].opd[i].rows, 4);
    }
    auto invalid = w;
    invalid.out = w.h;
    ManifoldConn refused;
    CHECK_EQ(refused.declare(&b, nm, g, 3, "bad", cfg, invalid, src), RAD_E_INVAL);
}

TEST(manifold_read_splits_its_projection_by_chunks_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.world = world;
            g.rank = rank;
            ManifoldConn::Config cfg;
            ManifoldConn::Src src{ "model.layers.3.hc_attn_fn", "model.layers.3.hc_attn_scale",
                                   "model.layers.3.hc_attn_base" };
            ManifoldConn::Wire w;
            w.h = decl_b(&b, "h", RAD_BF16, { 512, 16384 });
            w.x = decl_b(&b, "x", RAD_BF16, { 512, 4096 });
            w.y = decl_b(&b, "y", RAD_BF16, { 512, 4096 });
            w.out = decl_b(&b, "out", RAD_BF16, { 512, 16384 });
            w.post = decl_b(&b, "post", RAD_F32, { 512, 4 });
            w.comb = decl_b(&b, "comb", RAD_F32, { 512, 16 });
            ManifoldConn conn;
            REQUIRE_EQ(conn.declare(&b, nm, g, 3, "blk.3.mhc_attn", cfg, w, src), RAD_OK);
            const bool split = world > 1;
            const int64_t mine = 8 / world;
            CHECK_EQ(count_op(b, "mhc_pre"), 1);
            CHECK_EQ(count_op(b, "mhc_pre_partial"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "all_gather"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "mhc_pre_finish"), split ? 1 : 0);
            CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_NONE);
            if (split) {
                const RecOp* part = find_op(b, "mhc_pre_partial");
                const RecOp* gather = find_op(b, "all_gather");
                const RecOp* finish = find_op(b, "mhc_pre_finish");
                REQUIRE(part && gather && finish);
                CHECK_EQ(pint(part, "chunk"), 2048);
                CHECK_EQ(pint(part, "first"), rank * mine);
                CHECK_EQ(pint(part, "chunks"), mine);
                CHECK(part->w == std::vector<rad_weight>({ conn.w_fn }));
                CHECK(part->reads == std::vector<rad_buf>({ w.h }));
                CHECK(part->writes == std::vector<rad_buf>({ conn.part }));
                CHECK_EQ(pint(gather, "world_size"), world);
                CHECK_EQ(pint(gather, "row"), mine * 28);
                CHECK_EQ(param(gather, "numel")->kind, RAD_P_RANGE);
                CHECK_EQ(param(gather, "numel")->ival, mine * 28);
                CHECK_EQ(param(gather, "numel")->ihi, 512 * mine * 28);
                CHECK_EQ(pstr(gather, "dtype"), "f32");
                CHECK(gather->reads == std::vector<rad_buf>({ conn.part }));
                CHECK(gather->writes == std::vector<rad_buf>({ conn.parts }));
                CHECK_EQ(pint(finish, "chunk"), 2048);
                CHECK_EQ(pint(finish, "iterations"), 20);
                CHECK(finish->w == std::vector<rad_weight>({ conn.w_scale, conn.w_base }));
                CHECK(finish->reads == std::vector<rad_buf>({ conn.parts, w.h }));
                CHECK(finish->writes == std::vector<rad_buf>({ w.x, w.post, w.comb }));
                CHECK_EQ(find_b(b, "blk.3.mhc_attn.parts")->d.shape[1], 8 * 28);
            }
            RadCtx ctx;
            conn.read(&ctx, 300);
            conn.read(&ctx, 255);
            if (split) {
                REQUIRE_EQ(ctx.issues.size(), (size_t)4);
                CHECK_EQ(ctx.issues[0].op, conn.op_partial);
                CHECK_EQ(ctx.issues[0].opd[0].handle, w.h);
                CHECK_EQ(ctx.issues[0].opd[0].rows, 300);
                CHECK_EQ(ctx.issues[0].opd[1].handle, conn.w_fn);
                CHECK_EQ(ctx.issues[0].opd[2].handle, conn.part);
                CHECK_EQ(ctx.issues[1].op, conn.op_gather);
                CHECK_EQ(ctx.issues[1].n, 300 * mine * 28);
                CHECK_EQ(ctx.issues[1].opd[1].handle, conn.parts);
                CHECK_EQ(ctx.issues[1].opd[1].rows, 300);
                CHECK_EQ(ctx.issues[2].op, conn.op_finish);
                REQUIRE_EQ(ctx.issues[2].opd.size(), (size_t)7);
                const uint32_t expected[] = { conn.parts, w.h,    conn.w_scale, conn.w_base,
                                              w.x,        w.post, w.comb };
                for (int i = 0; i < 7; ++i) CHECK_EQ(ctx.issues[2].opd[i].handle, expected[i]);
                CHECK_EQ(ctx.issues[3].op, conn.op_pre);
            } else {
                REQUIRE_EQ(ctx.issues.size(), (size_t)2);
                CHECK_EQ(ctx.issues[0].op, conn.op_pre);
                CHECK_EQ(ctx.issues[1].op, conn.op_pre);
            }
        }
}

TEST(glm_routing_keeps_f32_logits_and_correction_replicated_under_tp4) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 128;
        g.world = 4;
        g.rank = rank;
        GlmRouter::Config cfg;
        GlmRouter::Wire w;
        w.input = decl_b(&b, "hidden", RAD_BF16, { 128, 4096 });
        w.logits = decl_b(&b, "logits", RAD_F32, { 32, 288 });
        w.ids = decl_b(&b, "ids", RAD_I32, { 32, 8 });
        w.weights = decl_b(&b, "weights", RAD_F32, { 32, 8 });
        GlmRouter::Src src{ "model.layers.3.mlp.gate.weight",
                            "model.layers.3.mlp.gate.e_score_correction_bias" };
        GlmRouter router;
        REQUIRE_EQ(router.declare(&b, nm, g, 3, "blk.3.router", cfg, w, src), RAD_OK);
        REQUIRE_EQ(b.weights.size(), (size_t)2);
        for (const auto& weight : b.weights) {
            CHECK_EQ(weight.d.dtype, RAD_F32);
            CHECK_EQ(weight.d.shard, RAD_SHARD_NONE);
        }
        CHECK_EQ(b.weights[0].d.shape[0], 288);
        CHECK_EQ(b.weights[0].d.shape[1], 4096);
        CHECK_EQ(b.ops[router.op_project - 1].op, std::string("gemm_nt_f32"));
        CHECK_EQ(b.ops[router.op_select - 1].op, std::string("router_sigmoid_topk"));
        RadCtx ctx;
        router.step(&ctx, 7, 4);
        REQUIRE_EQ(ctx.issues.size(), (size_t)2);
        CHECK_EQ(ctx.issues[0].opd[0].handle, w.input);
        CHECK_EQ(ctx.issues[0].opd[0].offset, 7 * 4096);
        CHECK_EQ(ctx.issues[0].opd[0].rows, 4);
        CHECK_EQ(ctx.issues[1].opd[1].handle, router.correction);
        CHECK_EQ(ctx.issues[1].opd[2].handle, w.ids);
        CHECK_EQ(ctx.issues[1].opd[3].handle, w.weights);
    }
}

TEST(glm_router_and_index_head_split_columns_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.world = world;
            g.rank = rank;
            GlmRouter::Config cfg;
            cfg.rows = 512;
            GlmRouter::Wire w;
            w.input = decl_b(&b, "hidden", RAD_BF16, { 512, 4096 });
            w.logits = decl_b(&b, "logits", RAD_F32, { 512, 288 });
            w.ids = decl_b(&b, "ids", RAD_I32, { 512, 8 });
            w.weights = decl_b(&b, "weights", RAD_F32, { 512, 8 });
            GlmRouter::Src src{ "model.layers.3.mlp.gate.weight",
                                "model.layers.3.mlp.gate.e_score_correction_bias" };
            GlmRouter router;
            REQUIRE_EQ(router.declare(&b, nm, g, 3, "blk.3.router", cfg, w, src), RAD_OK);
            const bool split = world > 1;
            const int64_t cols = 288 / world;
            CHECK_EQ(count_op(b, "all_gather"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "gemm_nt_f32"), split ? 2 : 1);
            // The weight stays whole and replicated: decode keeps the unsplit projection.
            CHECK_EQ(b.weights[0].d.shape[0], 288);
            CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_NONE);
            if (split) {
                const RecOp* part = find_op(b, "gemm_nt_f32", 1);
                const RecOp* gather = find_op(b, "all_gather");
                REQUIRE(part && gather);
                CHECK_EQ(pint(part, "N"), cols);
                CHECK_EQ(pint(part, "K"), 4096);
                CHECK(part->w == std::vector<rad_weight>({ router.projection }));
                CHECK(part->reads == std::vector<rad_buf>({ w.input }));
                CHECK(part->writes == std::vector<rad_buf>({ router.split.part }));
                CHECK_EQ(pint(gather, "world_size"), world);
                CHECK_EQ(pint(gather, "row"), cols);
                CHECK_EQ(param(gather, "numel")->kind, RAD_P_RANGE);
                CHECK_EQ(param(gather, "numel")->ival, cols);
                CHECK_EQ(param(gather, "numel")->ihi, 512 * cols);
                CHECK_EQ(pstr(gather, "dtype"), "f32");
                CHECK(gather->reads == std::vector<rad_buf>({ router.split.part }));
                CHECK(gather->writes == std::vector<rad_buf>({ w.logits }));
            }
            RadCtx ctx;
            router.step(&ctx, 7, 300, 7);
            if (split) {
                REQUIRE_EQ(ctx.issues.size(), (size_t)3);
                CHECK_EQ(ctx.issues[0].op, router.split.op_part);
                CHECK_EQ(ctx.issues[0].n, 300);
                CHECK_EQ(ctx.issues[0].opd[0].handle, w.input);
                CHECK_EQ(ctx.issues[0].opd[0].offset, 7 * 4096);
                CHECK_EQ(ctx.issues[0].opd[0].rows, 300);
                CHECK_EQ(ctx.issues[0].opd[1].kind, RAD_OPK_WEIGHT);
                CHECK_EQ(ctx.issues[0].opd[1].handle, router.projection);
                CHECK_EQ(ctx.issues[0].opd[1].offset, rank * cols * 4096);
                CHECK_EQ(ctx.issues[0].opd[1].rows, cols);
                CHECK_EQ(ctx.issues[0].opd[2].handle, router.split.part);
                CHECK_EQ(ctx.issues[0].opd[2].rows, 300);
                CHECK_EQ(ctx.issues[1].op, router.split.op_gather);
                CHECK_EQ(ctx.issues[1].n, 300 * cols);
                CHECK_EQ(ctx.issues[1].opd[0].handle, router.split.part);
                CHECK_EQ(ctx.issues[1].opd[1].handle, w.logits);
                CHECK_EQ(ctx.issues[1].opd[1].offset, 7 * 288);
                CHECK_EQ(ctx.issues[1].opd[1].rows, 300);
                CHECK_EQ(ctx.issues[2].op, router.op_select);
            } else {
                REQUIRE_EQ(ctx.issues.size(), (size_t)2);
                CHECK_EQ(ctx.issues[0].op, router.op_project);
            }
            ctx.issues.clear();
            router.step(&ctx, 0, 255);
            REQUIRE_EQ(ctx.issues.size(), (size_t)2);
            CHECK_EQ(ctx.issues[0].op, router.op_project);
        }
}

TEST(absorbed_mla_shards_complete_heads_and_reads_one_latent_cache) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = RAD_BF16;
        kd.n_head_kv = 1;
        kd.head_dim = 256;
        const rad_kvgroup kv = rad_decl_kv_group(&b, "latent", &kd);
        MlaLatent::Config cfg;
        MlaLatent::Wire w;
        w.q = decl_b(&b, "q", RAD_BF16, { 32, 16, 256 });
        w.q_latent = decl_b(&b, "ql", RAD_BF16, { 32, 16, 512 });
        w.attn_latent = decl_b(&b, "al", RAD_BF16, { 32, 16, 512 });
        w.out = decl_b(&b, "out", RAD_BF16, { 32, 16, 256 });
        w.selected = decl_b(&b, "selected", RAD_I32, { 32, 513 });
        w.length = decl_b(&b, "length", RAD_I32, { 32 });
        MlaLatent block;
        REQUIRE_EQ(
            block.declare(&b, nm, g, 3, kv, cfg, w, "model.layers.3.self_attn.kv_b_proj.weight"),
            RAD_OK);
        REQUIRE_EQ(b.weights.size(), (size_t)1);
        CHECK_EQ(b.weights[0].d.dtype, RAD_BF16);
        CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[0].d.shape[0], 8192);
        CHECK_EQ(b.weights[0].d.shape[1], 512);
        RadCtx ctx;
        block.step(&ctx, 4);
        REQUIRE_EQ(ctx.issues.size(), (size_t)3);
        CHECK_EQ(b.ops[ctx.issues[0].op - 1].op, std::string("mla_project"));
        CHECK_EQ(b.ops[ctx.issues[1].op - 1].op, std::string("mla_attn"));
        CHECK_EQ(b.ops[ctx.issues[2].op - 1].op, std::string("mla_project"));
        CHECK_EQ(ctx.issues[1].opd[1].kind, RAD_OPK_KV);
        CHECK_EQ(ctx.issues[1].opd[2].handle, w.selected);
        CHECK_EQ(ctx.issues[1].opd[3].handle, w.length);
        CHECK_EQ(ctx.issues[2].opd[1].handle, block.projection);
    }
}

TEST(
    glm53_native_graph_declares_all_layers_preserves_mixed_formats_and_runs_mtp_only_in_draft_passes) {
    using namespace rad::arch;
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    for (int world : { 1, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
            RadBuildCtx build{};
            build.rank = rank;
            build.world_size = world;
            build.max_tok = rank == 0 ? 512 : 16;
            build.max_seqs = 4;
            build.max_ctx = 4096;
            build.max_spec = 3;
            REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
            const auto& m = *glm5next::models[rank];
            CHECK(m.have_mtp);
            CHECK_EQ(b.drafter.kind, RAD_DRAFT_SERIAL);
            CHECK_EQ(b.drafter.depth, 3);
            CHECK_EQ(m.layers.size(), (size_t)45);
            int full = 0;
            for (int l = 0; l < 45; ++l) {
                const auto& layer = m.layers[l];
                full += layer.full;
                CHECK(layer.attn_hc.op_pre < layer.norm.op);
                CHECK(layer.attn_hc.op_post < layer.ffn_hc.op_pre);
                if (layer.full) {
                    CHECK(layer.index.cache.op_select < layer.mla.core.op_attn);
                    CHECK(layer.mla.output.op < layer.attn_hc.op_post);
                } else
                    CHECK(layer.kda.output.op < layer.attn_hc.op_post);
                if (l >= 3) {
                    CHECK_EQ(layer.moe.c.expert_mx, 1);
                    CHECK(layer.moe.op_gather < layer.ffn_hc.op_post);
                }
            }
            CHECK_EQ(full, 11);
            CHECK_EQ(m.draft.moe.c.expert_f32_scale, 1);
            for (const auto& map : b.maps)
                if (map.declared.starts_with("mtp."))
                    for (const auto& source : map.src) CHECK(!source.starts_with("mtp.model."));
            int32_t cu[] = { 0, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 0 }, ids[] = { 7, 9 },
                    slots[] = { 0, 1 }, out[] = { 1 };
            int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[32]{};
            std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
            for (size_t i = 0; i < kv.size(); ++i) {
                kv[i].group = (rad_kvgroup)i + 1;
                kv[i].slot_mapping = slots;
                kv[i].block_table = tables;
                kv[i].block_table_pitch = 32;
                kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
                kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
            }
            RadBatch batch{};
            batch.phase = RAD_PHASE_PREFILL;
            batch.n_tok = 2;
            batch.n_seq = 1;
            batch.token_ids = ids;
            batch.positions = positions;
            batch.cu_seqlens = cu;
            batch.ctx_lens = ctxlens;
            batch.kv = kv.data();
            batch.n_kv_groups = (int)kv.size();
            batch.n_out = 1;
            batch.out_ids = out;
            RadCtx ctx;
            ctx.b = &b;
            ctx.rank = rank;
            ctx.world = world;
            ctx.batch = &batch;
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            const auto ops = issued_ops(b, ctx);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "mhc_pre"), 90);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "kda_update"), 34);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "mla_attn"), 11);
            // Two rows stay under the split threshold: 42 routers and 11 index heads, replicated.
            CHECK_EQ(std::count(ops.begin(), ops.end(), "gemm_nt_f32"), 53);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "all_gather"), 0);
            // The split is declared on every rank whatever its step size; only rank 0 issues it here.
            const bool split = world > 1;
            // 43 routers and 12 index heads (MTP included), and the 90 mHC reads.
            CHECK_EQ(count_op(b, "all_gather"), split ? 55 + 90 : 0);
            CHECK_EQ(count_op(b, "mhc_pre_partial"), split ? 90 : 0);
            ctx.issues.clear();
            batch.draft_pass = -1;
            batch.n_draft_out = 1;
            batch.draft_out_ids = out;
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            const auto draft_ops = issued_ops(b, ctx);
            CHECK_EQ(std::count(draft_ops.begin(), draft_ops.end(), "mhc_pre"), 0);
            CHECK_EQ(std::count(draft_ops.begin(), draft_ops.end(), "mla_attn"), 1);
            CHECK_EQ(ctx.issues.back().op, m.mtp.op_head);
            CHECK(m.mtp.op_select < m.draft.moe.op_router);
            for (const auto& issue : ctx.issues) {
                if (issue.op == m.draft.moe.op_router || issue.op == m.mtp.op_norm)
                    CHECK_EQ(issue.opd[0].rows, 1);   // Only the row producing a proposal.
                if (issue.op == m.mtp.op_residual)
                    CHECK_EQ(issue.opd[0].rows, 2);   // Full history still fills attention caches.
            }
            ctx.issues.clear();
            batch.n_draft_out = 0;
            glm5next::step(&ctx, &batch);
            const auto history = issued_ops(b, ctx);
            CHECK(std::find(history.begin(), history.end(), "kv_store") != history.end());
            CHECK(std::find(history.begin(), history.end(), "moe_gemm_q") == history.end());
            if (rank == 0) {
                std::vector<int32_t> long_ids(512, 7), long_positions(512), long_slots(512),
                    long_tables(1024);
                for (int i = 0; i < 512; ++i) long_positions[i] = long_slots[i] = i;
                int32_t long_cu[] = { 0, 512 }, last = 511;
                for (auto& group : kv) {
                    group.slot_mapping = long_slots.data();
                    group.block_table = long_tables.data();
                    group.block_table_pitch = 1024;
                }
                batch.n_tok = 512;
                batch.token_ids = long_ids.data();
                batch.positions = long_positions.data();
                batch.cu_seqlens = long_cu;
                batch.out_ids = &last;
                batch.draft_pass = 0;
                batch.max_q_len = 512;
                ctx.issues.clear();
                glm5next::step(&ctx, &batch);
                CHECK(ctx.step_fail.empty());
                {
                    // 512 rows reach the split: each projection runs on its column slice and each
                    // mHC read on its chunks, both gathered; at TP1 nothing changes.
                    const auto long_ops = issued_ops(b, ctx);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "gemm_nt_f32"), 53);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "all_gather"),
                             world > 1 ? 53 + 90 : 0);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mhc_pre"),
                             world > 1 ? 0 : 90);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mhc_pre_finish"),
                             world > 1 ? 90 : 0);
                    for (const auto& issue : ctx.issues) {
                        if (b.ops[issue.op - 1].op != "gemm_nt_f32") continue;
                        const int64_t n = pint(&b.ops[issue.op - 1], "N");
                        if (world == 1)
                            CHECK(n == 288 || n == 32);
                        else {
                            CHECK(n == 288 / world || n == 32 / world);
                            CHECK_EQ(issue.opd[1].rows, n);
                        }
                    }
                }
                for (int l = 3; l < 45; ++l) {
                    // A prefill routes its full batch once, reusing each expert's
                    // weights across its rows instead of re-reading them in chunks.
                    CHECK_EQ(std::count_if(ctx.issues.begin(), ctx.issues.end(),
                                           [&](const auto& issue) {
                                               const auto& moe = m.layers[l].moe;
                                               return issue.op == moe.op_router ||
                                                      issue.op == moe.glm_router.split.op_part;
                                           }),
                             1);
                }
                // Arena-lending probes must not replace the model the serving
                // graph's handles refer to, even when they declare fewer rows.
                const auto* live = glm5next::models[rank].get();
                RadBuilder pb;
                pb.kv_block = 4;
                pb.encs = b.encs;
                auto small = build;
                small.shape_probe = 1;
                small.max_tok = 1;
                REQUIRE_EQ(glm5next::declare(&pb, &meta, &small), RAD_OK);
                CHECK_EQ(glm5next::models[rank].get(), live);
                CHECK_EQ(glm5next::models[rank]->g.max_tok, 512);
                RadArchProbe capabilities{};
                REQUIRE_EQ(glm5next::probe(&meta, &capabilities), RAD_OK);
                CHECK_EQ(capabilities.shape_probe_ok, 1);
                std::array<int, 8> results{};
                std::vector<std::thread> probes;
                for (int i = 0; i < 8; ++i)
                    probes.emplace_back([&, i] {
                        RadBuilder parallel;
                        parallel.kv_block = 4;
                        parallel.encs = b.encs;
                        auto level = build;
                        level.rank = i % 4;
                        level.max_tok = i < 4 ? 256 : 1024;
                        level.shape_probe = 1;
                        results[i] = glm5next::declare(&parallel, &meta, &level);
                    });
                for (auto& thread : probes) thread.join();
                for (int status : results) CHECK_EQ(status, RAD_OK);
                CHECK_EQ(glm5next::models[rank].get(), live);
            }
        }
}

/* The engine's arena levels declare again at a smaller max_tok (RadBuildCtx::shape_probe) and
 * must get the real declare's graph: the same ops with the same fixed parameters, only ranged row
 * counts and buffer rows smaller. The TP split declares its ops whatever the step size and its
 * gathers' message as a range. */
TEST(glm53_sizing_declares_match_the_real_graph_at_every_arena_level) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    const int world = 8, rank = 3;
    for (int64_t R : { 256, 3072 }) {
        auto encs = [&](RadBuilder& b) {
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
        };
        RadBuildCtx c{};
        c.rank = rank;
        c.world_size = world;
        c.max_tok = 4096;
        c.max_seqs = 8;
        c.max_ctx = 32768;
        c.max_spec = 3;
        RadBuilder full;
        encs(full);
        REQUIRE_EQ(glm5next::declare(&full, &meta, &c), RAD_OK);
        RadBuildCtx p = c;
        p.max_tok = R;
        p.shape_probe = 1;
        RadBuilder small;
        encs(small);
        REQUIRE_EQ(glm5next::declare(&small, &meta, &p), RAD_OK);
        REQUIRE_EQ(small.ops.size(), full.ops.size());
        int differ = 0;
        for (size_t i = 0; i < full.ops.size(); ++i) {
            if (small.ops[i].op != full.ops[i].op) {
                ++differ;
                continue;
            }
            REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
            for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                const RecParam& a = full.ops[i].p[k];
                const RecParam& q = small.ops[i].p[k];
                if (a.kind == RAD_P_RANGE) continue;
                // A capacity may shrink: its kernel writes by its operands' extents.
                if (full.ops[i].op == "qsa_work" && a.key == "work" && q.ival <= a.ival)
                    continue;
                if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                    fprintf(stderr,
                            "    R %lld: op %zu '%s' param '%s': "
                            "%lld real, %lld small\n",
                            (long long)R, i, full.ops[i].op.c_str(), a.key.c_str(),
                            a.ival, q.ival);
                    ++differ;
                }
            }
        }
        CHECK_EQ(differ, 0);
        REQUIRE_EQ(small.bufs.size(), full.bufs.size());
        for (size_t i = 0; i < full.bufs.size(); ++i)
            for (uint32_t d = 1; d < full.bufs[i].d.rank; ++d)
                CHECK_EQ(small.bufs[i].d.shape[d], full.bufs[i].d.shape[d]);
    }
}

/* The MoE block's decode fusions: over a two-row trunk step, its draft pass and a 512-row prefill
 * (past the router fusion's one-workgroup sort), each fused launch stands where it fits and the
 * launches it replaces are gone, with the step lint clean throughout. The sort writes each slot's
 * row (`moe_scatter_inv`) and the gather reads it (`moe_gather_inv`) in every MoE pass. */
TEST(glm53_moe_decode_fusions_replace_their_launches_where_they_fit) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    using Counts = std::map<std::string, int>;
    /* Launches per step: trunk, draft pass, prefill. The trunk has 42 MoE layers, each a routed
     * and a shared clamped SwiGLU, and three dense ones. */
    struct Want {
        const char* op;
        int trunk, draft, prefill;
    };
    const Want wants[] = {
        { "router_sigmoid_topk_scatter", 42, 1, 0 }, { "router_sigmoid_topk", 0, 0, 42 },
        { "swiglu_clamped_quant_fp8", 87, 2, 87 },   { "swiglu_clamped", 0, 0, 0 },
        { "moe_scatter", 0, 0, 0 },                  { "moe_gather", 0, 0, 0 },
    };
    for (int world : { 1, 8 }) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx build{};
        build.rank = 0;
        build.world_size = world;
        build.max_tok = 512;
        build.max_seqs = 4;
        build.max_ctx = 4096;
        build.max_spec = 3;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
        int32_t cu[] = { 0, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 0 }, ids[] = { 7, 9 },
                slots[] = { 0, 1 }, out[] = { 1 };
        int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[32]{};
        std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
        for (size_t i = 0; i < kv.size(); ++i) {
            kv[i].group = (rad_kvgroup)i + 1;
            kv[i].slot_mapping = slots;
            kv[i].block_table = tables;
            kv[i].block_table_pitch = 32;
            kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
            kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
        }
        RadBatch batch{};
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 2;
        batch.n_seq = 1;
        batch.token_ids = ids;
        batch.positions = positions;
        batch.cu_seqlens = cu;
        batch.ctx_lens = ctxlens;
        batch.kv = kv.data();
        batch.n_kv_groups = (int)kv.size();
        batch.n_out = 1;
        batch.out_ids = out;
        RadCtx ctx;
        ctx.b = &b;
        ctx.rank = 0;
        ctx.world = world;
        ctx.batch = &batch;
        Counts got[3];
        /* With the inverse, the fused selection writes it as its ninth operand and the gather
         * reads it as its third. */
        const rad_buf inv = find_b(b, "expert_inverse")
                                ? (rad_buf)(find_b(b, "expert_inverse") - &b.bufs[0] + 1)
                                : 0;
        REQUIRE(inv != 0);
        auto run = [&](int k) {
            ctx.issues.clear();
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            CHECK_LINT(b, &ctx);
            for (const auto& op : issued_ops(b, ctx)) ++got[k][op];
            for (const auto& is : ctx.issues) {
                const RecOp* o = b.op_of(is.op);
                if (!o) continue;
                if (o->op == "router_sigmoid_topk_scatter")
                    CHECK(is.opd.size() > 8 && is.opd[8].kind == RAD_OPK_BUF &&
                          is.opd[8].handle == inv);
                if (o->op == "moe_gather_inv")
                    CHECK(is.opd.size() > 2 && is.opd[2].kind == RAD_OPK_BUF &&
                          is.opd[2].handle == inv);
            }
        };
        run(0);
        batch.draft_pass = -1;
        batch.n_draft_out = 1;
        batch.draft_out_ids = out;
        run(1);
        std::vector<int32_t> long_ids(512, 7), long_positions(512), long_slots(512),
            long_tables(1024);
        for (int i = 0; i < 512; ++i) long_positions[i] = long_slots[i] = i;
        int32_t long_cu[] = { 0, 512 }, last = 511;
        for (auto& group : kv) {
            group.slot_mapping = long_slots.data();
            group.block_table = long_tables.data();
            group.block_table_pitch = 1024;
        }
        batch.n_tok = 512;
        batch.token_ids = long_ids.data();
        batch.positions = long_positions.data();
        batch.cu_seqlens = long_cu;
        batch.out_ids = &last;
        batch.draft_pass = 0;
        batch.n_draft_out = 0;
        batch.max_q_len = 512;
        run(2);
        for (int k = 0; k < 3; ++k)
            for (const Want& w : wants) {
                const int want = k == 0 ? w.trunk : k == 1 ? w.draft : w.prefill;
                if (got[k][w.op] != want)
                    fprintf(stderr, "    world %d step %d: %s %d (want %d)\n", world, k, w.op,
                            got[k][w.op], want);
                CHECK_EQ(got[k][w.op], want);
            }
    }
}

TEST(dsa_cache_uses_distinct_index_and_attention_pages_and_a_speculative_tail_ring) {
    using namespace rad::arch;
    for (bool front : { false, true }) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.max_seqs = 4;
        g.max_spec = 3;
        RadKVGroupDecl ad{};
        ad.kind = RAD_KV_FULL;
        ad.dtype = RAD_BF16;
        ad.n_head_kv = 1;
        ad.head_dim = 256;
        const auto attn = rad_decl_kv_group(&b, "attn", &ad);
        // Claim the attention group's consumer before declaring the index group.
        RAD_OP(&b, "mla_attn", RAD_PARAMS(RAD_INT("block_size", 4)), RAD_NOWEIGHTS);
        RadKVGroupDecl td{};
        td.kind = RAD_KV_LINEAR;
        td.dtype = RAD_BF16;
        td.n_head_kv = 1;
        td.state_dim[0] = 7;
        td.state_dim[1] = 256;
        const auto tail = rad_decl_kv_group(&b, "tail", &td);
        DsaCache::Wire w;
        w.query = decl_b(&b, "query", RAD_BF16, { 1024, 128 });
        w.codes = decl_b(&b, "codes", RAD_F8E4M3, { 1024, 128 });
        w.scales = decl_b(&b, "scales", RAD_F32, { 1024, 1 });
        w.gate = decl_b(&b, "gates", RAD_F32, { 32, 32 });
        w.kg = decl_b(&b, "kg", RAD_BF16, { 32, 256 });
        w.stage = decl_b(&b, "stage", RAD_BF16, { 48, 256 });
        w.page = decl_b(&b, "page", RAD_I32, { 12 });
        w.bpos = decl_b(&b, "bpos", RAD_I32, { 12 });
        w.nc = decl_b(&b, "nc", RAD_I32, { 4 });
        w.pooled = decl_b(&b, "pooled", RAD_F8E4M3, { 12, 128 });
        w.pool_scale = decl_b(&b, "pool_scale", RAD_F32, { 12, 1 });
        w.score = decl_b(&b, "score", RAD_F32, { 32, 1024 });
        w.selected = decl_b(&b, "sel", RAD_I32, { 32, 513 });
        w.nsel = decl_b(&b, "nsel", RAD_I32, { 32 });
        w.length = decl_b(&b, "len", RAD_I32, { 32 });
        DsaCache block;
        DsaCache::Config cfg;
        cfg.blocks = 1024;
        cfg.work = 12;
        cfg.ring = 7;
        rad_kvgroup index = 0;
        GlmIndexer indexer;
        if (front) {
            GlmIndexer::Wire iw;
            iw.cache = w;
            iw.input = decl_b(&b, "input", RAD_BF16, { 32, 4096 });
            iw.q_low = decl_b(&b, "q_low", RAD_BF16, { 32, 1536 });
            iw.q_wide = decl_b(&b, "q_wide", RAD_BF16, { 32, 4096 });
            iw.kg_raw = decl_b(&b, "kg_raw", RAD_BF16, { 32, 256 });
            REQUIRE_EQ(indexer.declare(&b, nm, g, 3, iw, "indexer", cfg, attn, tail, &index),
                       RAD_OK);
            block = indexer.cache;
            CHECK_EQ(b.weights[indexer.head_weight - 1].d.dtype, RAD_F32);
            CHECK_EQ(b.weights[indexer.query.w - 1].d.shard, RAD_SHARD_NONE);
            CHECK_EQ(b.weights[indexer.key_gate.w - 1].d.shard, RAD_SHARD_NONE);
        } else
            REQUIRE_EQ(block.declare(&b, nm, g, 3, cfg, w, attn, tail, &index, "indexer.pool.ape"),
                       RAD_OK);
        CHECK_EQ(b.kv_groups[index - 1].d.dtype, RAD_U8);
        CHECK_EQ(b.kv_groups[index - 1].d.head_dim, 17);
        int32_t ibt[16]{}, abt[16]{}, sidx[12]{}, cu[] = { 0, 3, 8 }, pos[8]{};
        RadKVGroupBatch kv[3]{};
        kv[0].group = index;
        kv[0].block_table = ibt;
        kv[0].block_table_pitch = 8;
        kv[1].group = tail;
        kv[1].state_index = sidx;
        kv[1].state_index_pitch = 6;
        kv[2].group = attn;
        kv[2].block_table = abt;
        kv[2].block_table_pitch = 8;
        RadBatch batch{};
        batch.n_tok = 8;
        batch.n_seq = 2;
        batch.n_spec = 2;
        batch.kv = kv;
        batch.n_kv_groups = 3;
        batch.positions = pos;
        batch.cu_seqlens = cu;
        batch.max_ctx_len = 64;   // Mixed-row bound exceeds the eight physical table columns.
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        if (front) {
            indexer.step(&ctx, &batch);
            REQUIRE_EQ(ctx.issues.size(), (size_t)12);
            CHECK_EQ(ctx.issues[1].op, indexer.op_query_copy);
            CHECK_EQ(ctx.issues[1].opd[1].rows, 256);
            CHECK_EQ(ctx.issues[3].op, indexer.op_norm);
            CHECK_EQ(ctx.issues[4].opd[0].offset, 128);
            CHECK_EQ(ctx.issues[5].op, indexer.op_head);
            ctx.issues.erase(ctx.issues.begin(), ctx.issues.begin() + 6);
        } else
            block.step(&ctx, &batch);
        REQUIRE_EQ(ctx.issues.size(), (size_t)6);
        CHECK_EQ(ctx.issues[0].n, 256);
        CHECK(ctx.issues[1].opd[5].raw == ibt);
        CHECK_EQ(ctx.issues[1].opd[4].cols, 6);
        CHECK_EQ(ctx.issues[2].op, block.op_pool);
        CHECK_EQ(ctx.issues[2].opd[5].handle, index);
        CHECK_EQ(ctx.issues[3].op, block.op_tail);
        CHECK_EQ(ctx.issues[4].opd[0].rows, 256);
        CHECK_EQ(ctx.issues[4].opd[3].handle, index);
        CHECK_EQ(ctx.issues[4].opd[4].kind, RAD_OPK_NONE);
        CHECK(ctx.issues[4].opd[5].raw == ibt);
        CHECK_EQ(ctx.issues[4].opd[8].cols, 8);
        CHECK_EQ(ctx.issues[5].opd[0].cols, 8);
        CHECK(ctx.issues[5].opd[1].raw == abt);
    }
}

TEST(glm_dense_ffn_keeps_clipping_and_disables_the_unclipped_fp8_epilogue) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.n_ff = 3072;
    g.max_tok = 32;
    g.world = 4;
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    b.encs = { { "ffn_gate_up", fp8 }, { "ffn_down", fp8 } };
    MlpFP8::Wire w;
    w.x = decl_b(&b, "x", RAD_BF16, { 32, 4096 });
    w.h.x = decl_b(&b, "h", RAD_BF16, { 32, 4096 });
    REQUIRE_EQ(w.h.declare_qs(&b, nm, g, "h", 4096), RAD_OK);
    w.gate_up = decl_b(&b, "gate_up", RAD_BF16, { 32, 6144 });
    w.ffn.x = decl_b(&b, "ffn", RAD_BF16, { 32, 3072 });
    REQUIRE_EQ(w.ffn.declare_qs(&b, nm, g, "ffn", 3072), RAD_OK);
    MlpFP8 block;
    MlpFP8::Src src{ "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                     "mlp.up_proj.weight", "mlp.down_proj.weight" };
    REQUIRE_EQ(block.declare(&b, nm, g, 0, w, src, false, false, kArNone, false, 10), RAD_OK);
    CHECK_EQ(block.op_gated, 0u);
    CHECK_EQ(b.ops[block.gq_ffn.op - 1].op, std::string("swiglu_clamped"));
    RadCtx ctx;
    ctx.b = &b;
    block.slice(&ctx, 8, 4, 4);
    const auto ops = issued_ops(b, ctx);
    CHECK_EQ(std::count(ops.begin(), ops.end(), "swiglu_clamped"), 1);
    CHECK(std::find(ops.begin(), ops.end(), "gemm_nt_q_gated") == ops.end());
    CHECK(std::find(ops.begin(), ops.end(), "quant_act_fp8") != ops.end());
}

TEST(glm_mla_stores_one_normalized_latent_and_exposes_q_lora_before_sparse_attention) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = RAD_BF16;
        kd.n_head_kv = 1;
        kd.head_dim = 256;
        const auto kv = rad_decl_kv_group(&b, "latent", &kd);
        auto buf = [&](const char* name, int64_t n) {
            return decl_b(&b, name, RAD_BF16, { 32, n });
        };
        GlmMla::Wire w;
        w.input.x = buf("input", 4096);
        w.q_raw = buf("q_raw", 1536);
        w.kv_raw = buf("kv_raw", 512);
        w.q_low.x = buf("q_low", 1536);
        w.latent = buf("latent_out", 512);
        w.out = buf("out", 4096);
        REQUIRE_EQ(w.input.declare_qs(&b, nm, g, "input", 4096), RAD_OK);
        REQUIRE_EQ(w.q_low.declare_qs(&b, nm, g, "q_low", 1536), RAD_OK);
        w.core.q = buf("q", 4096);
        w.core.q_latent = decl_b(&b, "q_latent", RAD_BF16, { 32, 16, 512 });
        w.core.attn_latent = decl_b(&b, "attn_latent", RAD_BF16, { 32, 16, 512 });
        w.core.out = buf("expanded", 4096);
        w.attn.x = w.core.out;
        REQUIRE_EQ(w.attn.declare_qs(&b, nm, g, "expanded", 4096), RAD_OK);
        w.core.selected = decl_b(&b, "selected", RAD_I32, { 32, 513 });
        w.core.length = decl_b(&b, "length", RAD_I32, { 32 });
        GlmMla block;
        REQUIRE_EQ(block.declare(&b, nm, g, 3, w, "attn", kv), RAD_OK);
        CHECK_EQ(b.weights[block.q_a.w - 1].d.shard, RAD_SHARD_NONE);
        CHECK_EQ(b.weights[block.q_b.w - 1].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[block.output.w - 1].d.shard, RAD_SHARD_COL);
        int32_t slots[4]{};
        RadKVGroupBatch kb{};
        kb.group = kv;
        kb.slot_mapping = slots;
        RadBatch batch{};
        batch.n_tok = 4;
        batch.kv = &kb;
        batch.n_kv_groups = 1;
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        block.prepare(&ctx, &batch);
        auto find_issue = [&](rad_op op) -> const RecIssue* {
            for (const auto& issue : ctx.issues)
                if (issue.op == op) return &issue;
            return nullptr;
        };
        const auto* store = find_issue(block.op_store);
        REQUIRE(store);
        CHECK_EQ(store->opd[0].handle, w.latent);
        CHECK_EQ(store->opd[0].cols, 256);
        CHECK_EQ(store->opd[1].offset, 256);
        CHECK_EQ(store->opd[1].cols, 256);
        CHECK(store->opd[2].raw == slots);
        const auto before = issued_ops(b, ctx);
        CHECK(std::find(before.begin(), before.end(), "mla_attn") == before.end());
        block.finish(&ctx, &batch);
        const auto* attn = find_issue(block.core.op_attn);
        REQUIRE(attn);
        CHECK_EQ(attn->opd[2].handle, w.core.selected);
        CHECK_EQ(ctx.issues.back().op, block.op_ar);
    }
}

TEST(glm_kda_projects_local_heads_and_reduces_only_after_the_output_projection) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        g.max_spec = 3;
        RadKVGroupDecl sd{};
        sd.kind = RAD_KV_LINEAR;
        sd.dtype = RAD_F32;
        sd.n_head_kv = 16;
        sd.state_dim[0] = sd.state_dim[1] = 128;
        RadKVGroupDecl cd{};
        cd.kind = RAD_KV_CONV;
        cd.dtype = RAD_BF16;
        cd.n_head_kv = 1;
        cd.head_dim = 6144;
        cd.conv_width = 4;
        const auto state = rad_decl_kv_group(&b, "state", &sd),
                   history = rad_decl_kv_group(&b, "conv", &cd);
        auto buf = [&](const char* name, int64_t width) {
            return decl_b(&b, name, RAD_BF16, { 32, width });
        };
        GlmKda::Wire w;
        w.input = buf("input", 4096);
        w.forget_low = buf("forget_low", 128);
        w.gate_low = buf("gate_low", 128);
        w.out = buf("out", 4096);
        w.core.qkv = buf("qkv", 6144);
        w.core.conv = buf("conv_out", 6144);
        w.core.a = buf("a", 2048);
        w.core.beta = buf("beta", 16);
        w.core.gate = buf("gate", 2048);
        w.core.out = buf("core_out", 2048);
        KdaState::Src src{ "attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight",
                           "attn.A_log",           "attn.dt_bias",         "attn.o_norm.weight" };
        GlmKda block;
        REQUIRE_EQ(block.declare(&b, nm, g, 0, w, "attn", src, state, history), RAD_OK);
        CHECK_EQ(b.weights[block.qkv.w - 1].d.n_row_parts, 3);
        CHECK_EQ(b.weights[block.forget_a.w - 1].d.shard, RAD_SHARD_NONE);
        CHECK_EQ(b.weights[block.forget_b.w - 1].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[block.output.w - 1].d.shard, RAD_SHARD_COL);
        int32_t cu[] = { 0, 4 }, ctxlens[] = { 0 }, indices[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 };
        RadKVGroupBatch kv[2]{};
        kv[0].group = state;
        kv[0].state_index = indices;
        kv[0].state_index_pitch = 6;
        kv[1].group = history;
        kv[1].state_index = ci;
        RadBatch batch{};
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 4;
        batch.n_seq = 1;
        batch.cu_seqlens = cu;
        batch.ctx_lens = ctxlens;
        batch.kv = kv;
        batch.n_kv_groups = 2;
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        block.step(&ctx, &batch);
        const auto ops = issued_ops(b, ctx);
        CHECK_EQ(std::count(ops.begin(), ops.end(), "gemm_nt"), 7);
        CHECK_EQ(std::count(ops.begin(), ops.end(), "silu"), 0);
        REQUIRE_EQ(ctx.issues.size(), (size_t)11);
        CHECK_EQ(ctx.issues[9].op, block.output.op);
        CHECK_EQ(ctx.issues[10].op, block.op_ar);
        CHECK_EQ(ctx.issues[10].opd[0].handle, w.out);
    }
}

TEST(kda_state_core_splits_mixed_batches_and_preserves_absolute_prefill_offsets) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.max_tok = 32;
    g.world = 4;
    g.max_spec = 3;
    RadKVGroupDecl sd{};
    sd.kind = RAD_KV_LINEAR;
    sd.dtype = RAD_F32;
    sd.n_head_kv = 16;
    sd.state_dim[0] = sd.state_dim[1] = 128;
    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV;
    cd.dtype = RAD_BF16;
    cd.n_head_kv = 1;
    cd.head_dim = 6144;
    cd.conv_width = 4;
    const rad_kvgroup st = rad_decl_kv_group(&b, "state", &sd),
                      cv = rad_decl_kv_group(&b, "conv", &cd);
    KdaState::Wire w;
    w.qkv = decl_b(&b, "qkv", RAD_BF16, { 32, 6144 });
    w.conv = decl_b(&b, "filtered", RAD_BF16, { 32, 6144 });
    w.a = decl_b(&b, "a", RAD_BF16, { 32, 2048 });
    w.beta = decl_b(&b, "beta", RAD_BF16, { 32, 16 });
    w.gate = decl_b(&b, "gate", RAD_BF16, { 32, 2048 });
    w.out = decl_b(&b, "out", RAD_BF16, { 32, 16, 128 });
    KdaState::Src src{ "attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight",
                       "attn.A_log",           "attn.dt_bias",         "attn.o_norm.weight" };
    KdaState block;
    KdaState::Config cfg;
    REQUIRE_EQ(block.declare(&b, nm, g, 0, st, cv, cfg, w, src), RAD_OK);
    CHECK_EQ(b.weights[block.taps - 1].d.n_row_parts, 3);
    for (int p = 0; p < 3; ++p) CHECK_EQ(b.weights[block.taps - 1].d.row_parts[p], 2048);
    CHECK_EQ(b.weights[block.bias - 1].d.shape[0], 2048);
    for (rad_op op : { block.op_prefill, block.op_verify }) {
        bool found = false;
        for (const auto& p : b.ops[op - 1].p)
            if (p.key == "state_form") {
                CHECK_EQ(p.sval, std::string("anchor"));
                found = true;
            }
        CHECK(found);
    }
    int32_t cu[] = { 0, 3, 8 }, ctxlens[] = { 11, 0 }, accepted[] = { 2, 1 };
    int32_t indices[] = { 0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3 }, conv_indices[] = { 0, 1 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = st;
    kv[0].state_index = indices;
    kv[0].state_index_pitch = 6;
    kv[1].group = cv;
    kv[1].state_index = conv_indices;
    kv[1].state_index_pitch = 1;
    RadBatch batch{};
    batch.phase = RAD_PHASE_MIXED;
    batch.n_tok = 8;
    batch.n_seq = 2;
    batch.n_seq_decode = 1;
    batch.n_tok_decode = 3;
    batch.cu_seqlens = cu;
    batch.ctx_lens = ctxlens;
    batch.num_accepted = accepted;
    batch.kv = kv;
    batch.n_kv_groups = 2;
    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &batch;
    block.step(&ctx, &batch);
    REQUIRE_EQ(ctx.issues.size(), (size_t)5);
    CHECK_EQ(ctx.issues[0].op, block.op_conv);
    CHECK_EQ(ctx.issues[0].n, 3);
    CHECK_EQ(ctx.issues[1].op, block.op_verify);
    CHECK_EQ(ctx.issues[1].opd[0].cols, 2048);
    CHECK_EQ(ctx.issues[1].opd[1].offset, 2048);
    CHECK_EQ(ctx.issues[1].opd[2].offset, 4096);
    CHECK_EQ(ctx.issues[2].op, block.op_conv);
    CHECK_EQ(ctx.issues[2].n, 8);
    CHECK(ctx.issues[2].opd[3].raw == cu + 1);
    CHECK_EQ(ctx.issues[2].opd[7].kind, RAD_OPK_NONE);
    CHECK_EQ(ctx.issues[3].op, block.op_prefill);
    CHECK(ctx.issues[3].opd[8].raw == indices + 6);
    CHECK(ctx.issues[3].opd[9].raw == cu + 1);
    CHECK_EQ(ctx.issues[3].opd[0].offset, 0);
    CHECK_EQ(ctx.issues[3].opd[0].rows, 8);
    CHECK_EQ(ctx.issues[4].op, block.op_norm);
}

/* ==================================================================== the pinned graph and steps */

/* GLM declares and steps the pinned graph and launches at one and eight ranks: a 4096-row prefill
 * after history, a mixed step, a decode step, a verify and its draft passes. */
TEST(glm53_is_the_pinned_graph_and_steps) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    struct Case {
        int world;
        uint64_t graph, steps;
    };
    /* The step hashes moved once when `qsa_select` grew its ninth operand (`selb`, issued RAD_NONE
     * by the DSA indexer); the graph is unchanged. */
    const Case cases[] = { { 1, 0xed3fc26c486ea1b5ull, 0xb1c07ecf787e0fb5ull },
                           { 8, 0x8204e073efdeee0cull, 0x9576fc1f22894272ull } };
    for (const Case& cs : cases) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx ctx = spec_ctx(0, cs.world, 3);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 200000;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &ctx), RAD_OK);
        const uint64_t g = graph_hash(b);
        const uint64_t s = pinned_steps(b, glm5next::step, 0, cs.world);
        fprintf(stderr, "    pin: glm5next world %d: graph 0x%016llxull, steps 0x%016llxull\n",
                cs.world, (unsigned long long)g, (unsigned long long)s);
        CHECK_EQ(g, cs.graph);
        CHECK_EQ(s, cs.steps);
    }
}

/* ==================================================================== fixtures */
namespace {

/* GLM-5.3-Flash's container keys, the four the plugin reads. */
MetaKV glm53_meta() {
    RadModelMeta m{};
    m.n_layers = 45;
    m.n_embd = 4096;
    m.n_vocab = 154880;
    m.rms_eps = 1e-5f;
    MetaKV k(m);
    k.set("num_nextn_predict_layers", "1").set("hc_mult", "4").set("n_routed_experts", "288")
     .set("index_topk", "2048");
    return k;
}

/* The served container's encodings: block fp8 at f32 scales for the attention and dense linears
 * and the MTP layer's experts, MXFP4 for the trunk's experts. */
void glm53_encs(RadBuilder& b) {
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    b.kv_block = 4;
    b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
               { "mtp.blk.45.ffn_down_exps", fp8 },
               { "ffn_gate_up_exps", mx },
               { "ffn_down_exps", mx },
               { "ffn_gate_up_shexp", mx },
               { "ffn_down_shexp", mx },
               { "ffn_gate_up.weight", fp8 },
               { "ffn_down.weight", fp8 } };
}

/* A deployment of 512 rows a step, four sequences, 4k of context. */
RadBuildCtx glm53_ctx(int rank, int world, int max_spec = 3) {
    RadBuildCtx c = spec_ctx(rank, world, max_spec);
    c.max_tok = 512;
    c.max_seqs = 4;
    c.max_ctx = 4096;
    return c;
}

}  /* namespace */

RAD_TEST_MAIN()

/* ==================================================================== the op schemas */
namespace {

/* The op schemas the kernel libraries under $RADIANCE_HOME/kernels declare: libr4d's first, the
 * served vocabulary, then libref's for what it lacks. Empty without a home or a library, as
 * arch_test's libref vocabulary is. */
struct OpSpec {
    int n_operands = 0, w_req = 0, w_max = 0, n_tab = 0;
};
struct LoadedSchemas {
    std::map<std::string, OpSpec> ops;
    bool r4d = false;
};
const LoadedSchemas& loaded_schemas() {
    static LoadedSchemas s;
    static bool tried = false;
    if (tried) return s;
    tried = true;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home) return s;
    for (const char* lib : { "libr4d.so", "libref.so" }) {
        const std::string path = std::string(home) + "/kernels/" + lib;
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "    cannot load %s: %s\n", path.c_str(), dlerror());
            continue;
        }
        auto count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
        auto at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
        if (!count || !at) continue;
        for (int i = 0; i < count(); ++i) {
            const RadOpSchema* sc = at(i);
            if (!sc || !sc->op || s.ops.count(sc->op)) continue;
            OpSpec spec;
            spec.n_operands = sc->n_operands;
            for (int k = 0; sc->operands && k < sc->n_operands; ++k) {
                if (sc->operands[k].role == RAD_OPD_WTAB) {
                    ++spec.n_tab;
                } else if (sc->operands[k].role == RAD_OPD_WEIGHT) {
                    ++spec.w_max;
                    if (!sc->operands[k].optional) ++spec.w_req;
                }
            }
            s.ops[sc->op] = spec;
        }
        if (!std::strcmp(lib, "libr4d.so")) s.r4d = true;
    }
    return s;
}

}  /* namespace */

/* Every op the plugin issues carries as many operands as its schema declares, counted as
 * core/runtime/issue.cpp counts them (an absent optional operand is one of them), and every op it
 * declares names as many weights as rad_builder.cpp admits. The steps: at eight ranks and at
 * one, a mixed step over the split threshold (two decode rows beside a 498-row chunk), a step
 * under it, a six-row verify and six single-row decodes, each with its history pass and a draft
 * round, and a verify of eight sequences at 32k of context. The engine refuses a step at the
 * first mismatch, and the fakes alone cannot see one: an issue short of an optional operand
 * passes every host test and stops a served engine at its first step through that op. */
TEST(glm53_issues_every_op_with_the_operand_count_its_schema_declares) {
    const LoadedSchemas& lib = loaded_schemas();
    if (lib.ops.empty()) {
        fprintf(stderr, "    no kernel library under $RADIANCE_HOME/kernels; nothing to count\n");
        return;
    }
    MetaKV mk = glm53_meta();
    std::set<std::string> uncovered, wrong;
    for (int world : { 8, 1 }) {
        const int rank = world == 8 ? 3 : 0;
        RadBuilder b;
        glm53_encs(b);
        RadBuildCtx ctx = glm53_ctx(rank, world, 5);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 65536;
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        for (const RecOp& r : b.ops) {
            auto it = lib.ops.find(r.op);
            if (it == lib.ops.end()) {
                uncovered.insert(r.op);
                continue;
            }
            const OpSpec& s = it->second;
            const int n_w = (int)r.w.size();
            const bool ok = s.n_tab > 0 ? n_w - s.w_max >= s.n_tab
                                        : n_w >= s.w_req && n_w <= s.w_max;
            if (!ok && wrong.insert(r.op + " (declare)").second)
                fprintf(stderr,
                        "    world %d: '%s' declares %d weights; the schema has %d (%d required) "
                        "and %d tables\n",
                        world, r.op.c_str(), n_w, s.w_max, s.w_req, s.n_tab);
        }
        std::vector<std::unique_ptr<Kit>> kits;
        auto with_drafts = [&](std::unique_ptr<Kit> k) {
            auto his = history_pass(b, *k);
            auto round = round_pass(b, k->b.n_seq, 2);
            kits.push_back(std::move(k));
            kits.push_back(std::move(his));
            kits.push_back(std::move(round));
        };
        with_drafts(kit(b, { 1, 1 }, { 498 }, 100));
        with_drafts(kit(b, {}, { 255 }));
        with_drafts(kit(b, { 6 }, {}));
        with_drafts(kit(b, std::vector<int>(6, 1), {}));
        {
            auto dec = kit(b, std::vector<int>(8, 6), {});
            const int64_t pitch = 8192;
            dec->table.resize((size_t)(8 * pitch));
            for (int64_t i = 0; i < 8 * pitch; ++i) dec->table[(size_t)i] = (int32_t)i;
            for (auto& g : dec->kv) {
                g.block_table = dec->table.data();
                g.block_table_pitch = g.max_blocks = pitch;
            }
            dec->b.max_ctx_len = 32000;
            kits.push_back(std::move(dec));
        }
        std::vector<const Kit*> passes;
        for (const auto& k : kits) passes.push_back(k.get());
        auto v = run_passes(b, glm5next::step, passes, rank, world);
        CHECK_PASSES(b, v);
        for (size_t p = 0; p < v.size(); ++p)
            for (const RecIssue& issue : v[p]->issues) {
                const RecOp* op = b.op_of(issue.op);
                REQUIRE(op);
                auto it = lib.ops.find(op->op);
                if (it == lib.ops.end()) {
                    uncovered.insert(op->op);
                    continue;
                }
                if ((int)issue.opd.size() != it->second.n_operands && wrong.insert(op->op).second)
                    fprintf(stderr,
                            "    world %d pass %zu: '%s' issued with %zu operands; the schema "
                            "declares %d\n",
                            world, p, op->op.c_str(), issue.opd.size(), it->second.n_operands);
            }
    }
    CHECK(wrong.empty());
    for (const auto& o : uncovered) fprintf(stderr, "    no schema loaded for '%s'\n", o.c_str());
    if (lib.r4d) CHECK(uncovered.empty());
}

/* ==================================================================== the container's names */
namespace {

/* The served container, GLM-5.3-Flash-Quark-MXFP4.rad, as `rad-info -v` lists it: each weight's
 * name and stored encoding, for the model, one KDA layer (0, with the dense feed-forward of
 * layers 0 to 2), one MLA layer (3, with its experts 0 and 287) and the MTP layer under its
 * `mtp.` scope (the fp8 linears of its layer under `mtp.mtp.`, as the converter wrote them). The
 * other layers of a kind carry the same names at their own index. */
struct ContainerWeight {
    const char* name;
    const char* enc;
};
const ContainerWeight kGlm53Container[] = {
    { "output.weight", "bf16" },
    { "output_norm.weight", "bf16" },
    { "token_embd.weight", "bf16" },
    { "blk.0.attn_beta.weight", "bf16" },
    { "blk.0.attn_forget_a.weight", "bf16" },
    { "blk.0.attn_forget_b.weight", "bf16" },
    { "blk.0.attn_gate_a.weight", "bf16" },
    { "blk.0.attn_gate_b.weight", "bf16" },
    { "blk.0.attn_norm.weight", "bf16" },
    { "blk.0.attn_output.weight", "bf16" },
    { "blk.0.attn_qkv.weight", "bf16" },
    { "blk.0.ffn_down.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.0.ffn_gate_up.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.0.ffn_norm.weight", "bf16" },
    { "blk.0.hc_attn.base", "f32" },
    { "blk.0.hc_attn.fn", "bf16" },
    { "blk.0.hc_attn.scale", "f32" },
    { "blk.0.hc_ffn.base", "f32" },
    { "blk.0.hc_ffn.fn", "bf16" },
    { "blk.0.hc_ffn.scale", "f32" },
    { "blk.0.ssm_a", "f32" },
    { "blk.0.ssm_conv.weight", "bf16" },
    { "blk.0.ssm_dt_bias", "f32" },
    { "blk.0.ssm_norm.weight", "bf16" },
    { "blk.3.attn_kv_a.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.attn_kv_a_norm.weight", "bf16" },
    { "blk.3.attn_kv_b.weight", "bf16" },
    { "blk.3.attn_norm.weight", "bf16" },
    { "blk.3.attn_output.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.attn_q_a.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.attn_q_a_norm.weight", "bf16" },
    { "blk.3.attn_q_b.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.ffn_down_shexp.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_gate_up_shexp.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_norm.weight", "bf16" },
    { "blk.3.glm_router.correction", "f32" },
    { "blk.3.glm_router.weight", "bf16" },
    { "blk.3.hc_attn.base", "f32" },
    { "blk.3.hc_attn.fn", "bf16" },
    { "blk.3.hc_attn.scale", "f32" },
    { "blk.3.hc_ffn.base", "f32" },
    { "blk.3.hc_ffn.fn", "bf16" },
    { "blk.3.hc_ffn.scale", "f32" },
    { "blk.3.index_ape", "bf16" },
    { "blk.3.index_head.weight", "bf16" },
    { "blk.3.index_key_gate.weight", "bf16" },
    { "blk.3.index_norm.bias", "bf16" },
    { "blk.3.index_norm.weight", "bf16" },
    { "blk.3.index_query.weight", "bf16" },
    { "blk.3.ffn_gate_up_exps.0.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_down_exps.0.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_gate_up_exps.287.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_down_exps.287.weight", "fp4_e2m1*e8m0[1x32]" },
    { "mtp.blk.45.attn_kv_a_norm.weight", "bf16" },
    { "mtp.blk.45.attn_kv_b.weight", "bf16" },
    { "mtp.blk.45.attn_norm.weight", "bf16" },
    { "mtp.blk.45.attn_q_a_norm.weight", "bf16" },
    { "mtp.blk.45.ffn_norm.weight", "bf16" },
    { "mtp.blk.45.index_ape", "bf16" },
    { "mtp.blk.45.index_head.weight", "bf16" },
    { "mtp.blk.45.index_norm.bias", "bf16" },
    { "mtp.blk.45.index_norm.weight", "bf16" },
    { "mtp.eh_proj.weight", "bf16" },
    { "mtp.enorm.weight", "bf16" },
    { "mtp.hnorm.weight", "bf16" },
    { "mtp.mtp.blk.45.attn_kv_a.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.attn_output.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.attn_q_a.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.attn_q_b.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.ffn_down_shexp.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.ffn_gate_up_shexp.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.glm_router.correction", "f32" },
    { "mtp.mtp.blk.45.glm_router.weight", "bf16" },
    { "mtp.mtp.blk.45.index_key_gate.weight", "bf16" },
    { "mtp.mtp.blk.45.index_query.weight", "bf16" },
    { "mtp.output_norm.weight", "bf16" },
    { "mtp.blk.45.ffn_gate_up_exps.0.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.blk.45.ffn_down_exps.0.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.blk.45.ffn_gate_up_exps.287.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.blk.45.ffn_down_exps.287.weight", "fp8_e4m3*f32[128x128]" },
};

RadEncoding container_enc(const std::string& e) {
    if (e == "fp8_e4m3*f32[128x128]") return rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    if (e == "fp4_e2m1*e8m0[1x32]") return rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    if (e == "f32") return rad_enc_plain(RAD_F32);
    if (e != "bf16") fprintf(stderr, "    unknown container encoding '%s'\n", e.c_str());
    return rad_enc_plain(RAD_BF16);
}

/* The table above expanded to every layer: a KDA layer's names from layer 0 (its dense
 * feed-forward in layers 0 to 2 only), an MLA layer's from layer 3, the MoE's from layer 3 into
 * every layer from 3. `encs` answers rad_weight_encoding as
 * the container would; an expert's key stands without its index, so every expert matches it,
 * and the keys run longest first, because the fake matches a key anywhere in the name asked
 * (`output.weight` is inside `blk.3.attn_output.weight`). */
struct Glm53Container {
    std::set<std::string> names;
    std::map<std::string, std::string> enc;
    std::vector<std::pair<std::string, RadEncoding>> encs;
};
const Glm53Container& glm53_container() {
    static Glm53Container c;
    static bool built = false;
    if (built) return c;
    built = true;
    std::set<std::string> keys;
    auto put = [&](const std::string& name, const char* e) {
        if (!c.names.insert(name).second) return;
        c.enc[name] = e;
        std::string key = name;
        const size_t ex = key.find("_exps.");
        if (ex != std::string::npos) key = key.substr(0, ex + 5);
        if (keys.insert(key).second) c.encs.push_back({ key, container_enc(e) });
    };
    auto at_layer = [](const std::string& n, size_t prefix, int l) {
        return "blk." + std::to_string(l) + "." + n.substr(prefix);
    };
    for (const ContainerWeight& w : kGlm53Container) {
        const std::string n = w.name;
        if (n.rfind("blk.0.", 0) == 0) {
            const bool dense = n.find(".ffn_") != std::string::npos;
            for (int l = 0; l < 45; ++l)
                if (l % 4 != 3 && (!dense || l < 3)) put(at_layer(n, 6, l), w.enc);
        } else if (n.rfind("blk.3.", 0) == 0) {
            const bool moe = n.find(".ffn_") != std::string::npos ||
                             n.find(".glm_router.") != std::string::npos;
            for (int l = 3; l < 45; ++l)
                if (moe || l % 4 == 3) put(at_layer(n, 6, l), w.enc);
        } else {
            put(n, w.enc);
        }
    }
    std::stable_sort(c.encs.begin(), c.encs.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });
    return c;
}

/* An expert's entry stands for its siblings: any index reads as expert 0's. */
std::string as_expert_zero(std::string n) {
    const size_t ex = n.find("_exps.");
    if (ex == std::string::npos) return n;
    const size_t dot = n.find('.', ex + 6);
    return dot == std::string::npos ? n : n.substr(0, ex + 6) + "0" + n.substr(dot);
}

}  /* namespace */

/* Every weight the plugin declares reads an entry the served container holds: under its own name,
 * or through `source` for a view (a scale plane), and every GEMM names the dtype its weight's
 * stored encoding calls for: `fp8a8f32` on the f32-scaled block FP8, `mxfp4a8` on the MXFP4
 * experts, `bf16` on the plain linears. The encodings come from the table, as the container would
 * answer. A weight declared under a name the container does not hold, or a GEMM at a dtype its
 * weight's encoding does not call for, passes every host test and leaves bands with no kernel at
 * the eight-card start. */
TEST(glm53_declares_the_served_containers_names_and_their_encodings) {
    const Glm53Container& c = glm53_container();
    MetaKV mk = glm53_meta();
    for (int world : { 8, 1 }) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = c.encs;
        RadBuildCtx ctx = glm53_ctx(world == 8 ? 5 : 0, world);
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        std::map<std::string, const RecWeight*> declared;
        for (const RecWeight& w : b.weights) declared[w.name] = &w;
        // The stored entry a declared weight reads: itself, or its source, followed through views.
        auto stored = [&](const RecWeight& w) {
            std::string n = w.name;
            for (int hops = 0; hops < 4; ++hops) {
                auto it = declared.find(n);
                if (it == declared.end() || !it->second->d.source || n == it->second->d.source)
                    break;
                n = it->second->d.source;
            }
            return n;
        };
        int missing = 0, optional_absent = 0;
        for (const RecWeight& w : b.weights) {
            const std::string n = as_expert_zero(stored(w));
            if (c.names.count(n)) continue;
            if (w.d.optional) {
                ++optional_absent;
                continue;
            }
            if (++missing <= 24)
                fprintf(stderr, "    world %d: '%s' reads '%s', which the container does not hold\n",
                        world, w.name.c_str(), n.c_str());
        }
        CHECK_EQ(missing, 0);
        if (optional_absent)
            fprintf(stderr, "    world %d: %d optional weights the container does not hold\n", world,
                    optional_absent);
        int wrong = 0;
        for (const RecOp& r : b.ops) {
            if (r.w.empty() || !r.w[0]) continue;
            const bool quant = r.op == "gemm_nt_q" || r.op == "gemm_nt_q_gated" ||
                               r.op == "gemm_nt_q_bias" || r.op == "moe_gemm_q" ||
                               r.op == "moe_gemm_q_mod4";
            const bool plain = r.op == "gemm_nt" || r.op == "gemm_nt_bias" ||
                               r.op == "logits_gemm" || r.op == "embed_lookup";
            if (!quant && !plain) continue;
            const std::string n = as_expert_zero(stored(b.weights[r.w[0] - 1]));
            auto e = c.enc.find(n);
            if (e == c.enc.end()) continue;
            const char* want = e->second == "fp8_e4m3*f32[128x128]" ? "fp8a8f32"
                             : e->second == "fp4_e2m1*e8m0[1x32]"   ? "mxfp4a8"
                             : e->second == "bf16"                  ? "bf16"
                                                                    : nullptr;
            if (!want) continue;
            if (quant == (e->second == "bf16") && ++wrong <= 24) {
                fprintf(stderr, "    world %d: '%s' over '%s' (%s)\n", world, r.op.c_str(), n.c_str(),
                        e->second.c_str());
                continue;
            }
            if (pstr(&r, "dtype") != want && ++wrong <= 24)
                fprintf(stderr, "    world %d: '%s' over '%s' (%s) at dtype %s, not %s\n", world,
                        r.op.c_str(), n.c_str(), e->second.c_str(), pstr(&r, "dtype").c_str(), want);
        }
        CHECK_EQ(wrong, 0);
    }
}
