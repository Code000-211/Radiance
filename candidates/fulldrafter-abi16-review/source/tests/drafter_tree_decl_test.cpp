/* drafter_tree_decl_test.cpp -- validation and contracts for rad_declare_drafter_tree.
 *
 * Verifies:
 * 1. Additive public C-ABI declaration of rad_declare_drafter_tree.
 * 2. Unchanged linear drafter declaration behavior (SERIAL and BLOCK).
 * 3. Validation before mutation: handles, dtype (RAD_I32), rank (2), row width
 *    (1 + 2 * capacity), seq capacity (>= max_seqs), existing SERIAL/BLOCK drafter,
 *    duplicate tree declaration, capacity limits (1..65), and finished state.
 * 4. Program metadata publication without mutating branched_tree_supported.
 * 5. C-ABI compatibility and function signatures.
 * 6. Pruned MTP's five rounds do not replace the max_spec=7 declaration capacity.
 */

#include "rad_test.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "abi/rad_builder.h"
#include "arch/common/rad_mtp_tree_config.h"
#include "build/rad_build.h"
#include "plugin/registry.h"
#include "rad_core.h"

using namespace rad;

namespace {

RadModelMeta make_meta() {
    RadModelMeta m{};
    m.arch_id  = "test_arch";
    m.name     = "test_model";
    m.quant    = "bf16";
    m.n_layers = 2;
    m.n_embd   = 4096;
    return m;
}

RadBuildCtx make_ctx(int64_t max_seqs = 16, int max_spec = 7) {
    RadBuildCtx c{};
    c.rank = 0;
    c.world_size = 1;
    c.max_tok  = 8192;
    c.max_seqs = max_seqs;
    c.max_ctx  = 4096;
    c.max_spec = max_spec;
    c.scope = "";
    return c;
}

rad_buf decl_test_buf(Builder& b, const char* name, uint32_t dtype, uint32_t rank,
                      int64_t d0, int64_t d1 = 0, int64_t d2 = 0) {
    RadBufDecl bd{};
    bd.dtype = dtype;
    bd.rank = rank;
    bd.shape[0] = d0;
    bd.shape[1] = d1;
    bd.shape[2] = d2;
    bd.kind = RAD_BUF_TRANSIENT;
    bd.domain = RAD_DOMAIN_DEVICE;
    return b.decl_buffer(name, &bd);
}

// Declares a standard block drafter for tests requiring a prerequisite drafter.
rad_buf setup_block_drafter(Builder& b, int depth = 7, int block = 8) {
    rad_buf prop = decl_test_buf(b, "linear_proposal", RAD_I32, 2, b.program().ctx.max_seqs, depth);
    RadDrafterDecl d{};
    d.kind = RAD_DRAFT_BLOCK;
    d.name = "block_drafter";
    d.depth = depth;
    d.block = block;
    d.proposal = prop;
    d.mask_token = 151643;
    int st = rad_declare_drafter(static_cast<RadBuilder*>(&b), &d);
    (void)st;
    return prop;
}

}  // namespace

TEST(valid_drafter_tree_declaration) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));

    rad_buf linear_prop = setup_block_drafter(b, 7, 8);
    REQUIRE(linear_prop != RAD_NULL_HANDLE);

    const int64_t node_cap = 4;
    const int64_t expected_width = 1 + 2 * node_cap;  // 9
    rad_buf tree_buf = decl_test_buf(b, "drafter_tree_out", RAD_I32, 2, 16, expected_width);
    REQUIRE(tree_buf != RAD_NULL_HANDLE);

    CHECK_OK(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, node_cap));

    const Program& p = b.program();
    CHECK_EQ(p.drafter_tree_buf, tree_buf);
    CHECK_EQ(p.drafter_tree_capacity, node_cap);
    // Declaring tree must not alter branched_tree_supported
    CHECK(p.branched_tree_supported);
}

TEST(wide_drafter_tree_declaration_preserves_packed_format) {
    for (int64_t capacity : {int64_t{17}, int64_t{65}, int64_t{66}}) {
        Registry reg;
        Builder b(reg, make_meta(), make_ctx(1, 64));
        REQUIRE(setup_block_drafter(b, 64, 65) != RAD_NULL_HANDLE);
        REQUIRE_EQ(b.program().drafter.kind, RAD_DRAFT_BLOCK);
        REQUIRE_EQ(b.program().drafter.depth, 64);
        const rad_buf packed = decl_test_buf(b, "wide_tree", RAD_I32, 2, 1, 1 + 2 * capacity);
        REQUIRE(packed != RAD_NULL_HANDLE);
        const int status = rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), packed, capacity);
        CHECK_EQ(status, capacity <= 65 ? RAD_OK : RAD_E_INVAL);
        CHECK_EQ(b.program().drafter_tree_capacity, capacity <= 65 ? capacity : int64_t{0});
        CHECK_EQ(b.program().drafter_tree_buf, capacity <= 65 ? packed : RAD_NULL_HANDLE);
        CHECK(b.program().branched_tree_supported);
    }
}

TEST(unchanged_linear_serial_declaration) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 3));

    rad_buf prop = decl_test_buf(b, "serial_prop", RAD_I32, 2, 16, 3);
    RadDrafterDecl d{};
    d.kind = RAD_DRAFT_SERIAL;
    d.name = "serial_drafter";
    d.depth = 3;
    d.proposal = prop;

    CHECK_OK(rad_declare_drafter(static_cast<RadBuilder*>(&b), &d));

    const Program& p = b.program();
    CHECK_EQ(p.drafter.kind, static_cast<int>(RAD_DRAFT_SERIAL));
    CHECK_EQ(p.drafter.depth, 3);
    CHECK_EQ(p.drafter_tree_buf, static_cast<rad_buf>(0));
    CHECK_EQ(p.drafter_tree_capacity, static_cast<int64_t>(0));
}

TEST(pruned_rounds_do_not_replace_declared_spec_capacity) {
    // Parse the real startup policy, without leaking environment changes to other cases.
    const char* names[] = {
        "RADIANCE_MTP_TREE_POLICY", "RADIANCE_MTP_TREE_ROUNDS", "RADIANCE_MTP_TREE_COSTS"
    };
    std::string saved[3];
    bool present[3];
    for (int i = 0; i < 3; ++i) {
        const char* value = std::getenv(names[i]);
        present[i] = value != nullptr;
        if (value) saved[i] = value;
    }
    CHECK_EQ(setenv(names[0], "pruned", 1), 0);
    CHECK_EQ(setenv(names[1], "5", 1), 0);
    CHECK_EQ(unsetenv(names[2]), 0);
    MtpTreeConfig config;
    const bool parsed = parse_mtp_tree_config(config);
    for (int i = 0; i < 3; ++i)
        CHECK_EQ(present[i] ? setenv(names[i], saved[i].c_str(), 1) : unsetenv(names[i]), 0);
    REQUIRE(parsed);
    REQUIRE(config.pruned());
    CHECK_EQ(config.rounds, 5);
    CHECK_EQ(MtpTreeConfig::depth, 5);

    Registry reg;
    const RadBuildCtx ctx = make_ctx(16, 7);
    Builder rejected(reg, make_meta(), ctx);
    RadDrafterDecl d{};
    d.kind = RAD_DRAFT_SERIAL;
    d.name = "mtp";
    d.depth = config.rounds;
    d.proposal_pitch = 8;
    d.mask_token = -1;
    // This was the startup failure: actual rounds cannot replace declared capacity.
    CHECK_EQ(rad_declare_drafter(static_cast<RadBuilder*>(&rejected), &d), RAD_E_INVAL);
    CHECK_EQ(rejected.status(), RAD_E_INVAL);
    CHECK_EQ(rejected.program().drafter.kind, static_cast<int>(RAD_DRAFT_NONE));
    CHECK_EQ(rejected.program().drafter.depth, 0);
    CHECK(rejected.program().drafter_name.empty());
    CHECK_EQ(rejected.program().drafter_tree_buf, RAD_NULL_HANDLE);
    CHECK_EQ(rejected.program().drafter_tree_capacity, int64_t{0});

    // A rejected builder retains its error, so validate the corrected declaration separately.
    Builder accepted(reg, make_meta(), ctx);
    d.depth = ctx.max_spec;
    d.proposal = decl_test_buf(accepted, "mtp_didx", RAD_I32, 2, 16, 8);
    REQUIRE(d.proposal != RAD_NULL_HANDLE);
    REQUIRE_EQ(rad_declare_drafter(static_cast<RadBuilder*>(&accepted), &d), RAD_OK);
    const rad_buf tree = decl_test_buf(accepted, "mtp_tree", RAD_I32, 2, 16, 17);
    REQUIRE(tree != RAD_NULL_HANDLE);
    REQUIRE_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&accepted), tree, 8), RAD_OK);
    const Program& p = accepted.program();
    CHECK_EQ(p.ctx.max_spec, 7);
    CHECK_EQ(p.drafter.kind, static_cast<int>(RAD_DRAFT_SERIAL));
    CHECK_EQ(p.drafter.depth, 7);
    CHECK_EQ(p.drafter.proposal, d.proposal);
    CHECK_EQ(p.drafter_tree_buf, tree);
    CHECK_EQ(p.drafter_tree_capacity, int64_t{8});
    CHECK_EQ(config.rounds, 5);
    CHECK_OK(accepted.finish());
}

TEST(unchanged_linear_block_declaration) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));

    rad_buf prop = setup_block_drafter(b, 7, 8);
    REQUIRE(prop != RAD_NULL_HANDLE);

    const Program& p = b.program();
    CHECK_EQ(p.drafter.kind, static_cast<int>(RAD_DRAFT_BLOCK));
    CHECK_EQ(p.drafter.depth, 7);
    CHECK_EQ(p.drafter_tree_buf, static_cast<rad_buf>(0));
    CHECK_EQ(p.drafter_tree_capacity, static_cast<int64_t>(0));
}

TEST(reject_null_builder) {
    CHECK_EQ(rad_declare_drafter_tree(nullptr, 1, 4), RAD_E_INVAL);
}

TEST(reject_before_block_drafter) {
    Registry reg;

    // Case 1: No drafter declared at all
    {
        Builder b(reg, make_meta(), make_ctx(16, 7));
        rad_buf tree_buf = decl_test_buf(b, "tree_buf", RAD_I32, 2, 16, 9);
        CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, 4), RAD_E_INVAL);
        CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
    }

    // Case 2: SERIAL drafters must publish a packed device buffer too.
    {
        Builder b(reg, make_meta(), make_ctx(16, 3));
        rad_buf prop = decl_test_buf(b, "serial_prop", RAD_I32, 2, 16, 3);
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL;
        d.name = "serial_drafter";
        d.depth = 3;
        d.proposal = prop;
        CHECK_OK(rad_declare_drafter(static_cast<RadBuilder*>(&b), &d));

        CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), RAD_NULL_HANDLE, 4),
                 RAD_E_INVAL);
        CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
        CHECK_EQ(b.program().drafter_tree_capacity, static_cast<int64_t>(0));
        rad_buf tree = decl_test_buf(b, "serial_tree", RAD_I32, 2, 16, 9);
        CHECK_OK(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree, 4));
        CHECK_EQ(b.program().drafter_tree_buf, tree);
        CHECK_EQ(b.program().drafter_tree_capacity, static_cast<int64_t>(4));
    }

    // Case 3: SERIAL drafter naming a buffer is shape-validated like a block one, so a row that
    // is not 1 + 2*capacity is refused rather than silently mis-staged.
    {
        Builder b(reg, make_meta(), make_ctx(16, 3));
        rad_buf prop = decl_test_buf(b, "serial_prop", RAD_I32, 2, 16, 3);
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL;
        d.name = "serial_drafter";
        d.depth = 3;
        d.proposal = prop;
        CHECK_OK(rad_declare_drafter(static_cast<RadBuilder*>(&b), &d));

        rad_buf wrong = decl_test_buf(b, "tree_wrong", RAD_I32, 2, 16, 8);
        CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), wrong, 4), RAD_E_INVAL);
    }
}

TEST(reject_duplicate_tree_declaration) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);

    rad_buf tree1 = decl_test_buf(b, "tree1", RAD_I32, 2, 16, 9);
    rad_buf tree2 = decl_test_buf(b, "tree2", RAD_I32, 2, 16, 9);

    CHECK_OK(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree1, 4));
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree2, 4), RAD_E_STATE);

    // Metadata remains from the first declaration
    const Program& p = b.program();
    CHECK_EQ(p.drafter_tree_buf, tree1);
    CHECK_EQ(p.drafter_tree_capacity, static_cast<int64_t>(4));
}

TEST(reject_capacity_out_of_bounds) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);

    rad_buf tree_buf = decl_test_buf(b, "tree_buf", RAD_I32, 2, 16, 9);

    // Root-inclusive capacity must be in [1, 65].
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, 0), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, -1), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, -100), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, 66), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, 100), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree_buf, INT64_MAX), RAD_E_INVAL);

    // No metadata mutated on invalid calls
    CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
    CHECK_EQ(b.program().drafter_tree_capacity, static_cast<int64_t>(0));
}

TEST(reject_invalid_handles_and_aliases) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    rad_buf linear_prop = setup_block_drafter(b, 7, 8);

    rad_buf logits = decl_test_buf(b, "logits", RAD_BF16, 2, 16, 32000);
    CHECK_OK(rad_declare_logits(static_cast<RadBuilder*>(&b), logits));

    // Null handle
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), RAD_NULL_HANDLE, 4), RAD_E_INVAL);

    // Non-existent buffer handle
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), 9999, 4), RAD_E_INVAL);

    // Reusing the linear proposal buffer as tree buffer is rejected
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), linear_prop, 4), RAD_E_INVAL);

    // Reusing the logits buffer is rejected
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), logits, 4), RAD_E_INVAL);

    CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
}

TEST(reject_logits_tree_alias_in_both_declaration_orders) {
    Registry reg;
    for (bool tree_first : {false, true}) {
        Builder b(reg, make_meta(), make_ctx(16, 7));
        rad_buf linear_prop = setup_block_drafter(b, 7, 8);
        REQUIRE(b.program().drafter.kind == RAD_DRAFT_BLOCK);
        rad_buf tree = decl_test_buf(b, "tree", RAD_I32, 2, 16, 9);
        rad_buf logits = decl_test_buf(b, "logits", RAD_BF16, 2, 16, 32000);
        REQUIRE(tree != RAD_NULL_HANDLE);
        REQUIRE(logits != RAD_NULL_HANDLE);
        REQUIRE(rad_disable_branched_tree(static_cast<RadBuilder*>(&b)) == RAD_OK);

        if (tree_first) {
            REQUIRE(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree, 4) == RAD_OK);
            CHECK_EQ(rad_declare_logits(static_cast<RadBuilder*>(&b), tree), RAD_E_INVAL);
        } else {
            // A valid tree shape ensures rejection is for aliasing, not dtype or extent.
            REQUIRE(rad_declare_logits(static_cast<RadBuilder*>(&b), tree) == RAD_OK);
            CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree, 4), RAD_E_INVAL);
        }

        const Program& p = b.program();
        CHECK_EQ(p.logits_buf, tree_first ? RAD_NULL_HANDLE : tree);
        CHECK_EQ(p.drafter_tree_buf, tree_first ? tree : RAD_NULL_HANDLE);
        CHECK_EQ(p.drafter_tree_capacity, static_cast<int64_t>(tree_first ? 4 : 0));
        CHECK_EQ(p.drafter.proposal, linear_prop);
        CHECK_EQ(p.drafter.kind, static_cast<int>(RAD_DRAFT_BLOCK));
        CHECK_EQ(p.drafter.depth, 7);
        CHECK_EQ(p.drafter.block, static_cast<int64_t>(8));
        CHECK(!p.branched_tree_supported);

        if (tree_first) {
            // A failed alias declaration does not prevent a distinct logits declaration.
            CHECK_OK(rad_declare_logits(static_cast<RadBuilder*>(&b), logits));
            CHECK_OK(rad_declare_logits(static_cast<RadBuilder*>(&b), logits));
            // Preserve the existing error for a second, different logits buffer.
            CHECK_EQ(rad_declare_logits(static_cast<RadBuilder*>(&b), tree), RAD_E_STATE);
            CHECK_EQ(p.logits_buf, logits);
            CHECK_EQ(p.drafter_tree_buf, tree);
            CHECK_EQ(p.drafter_tree_capacity, static_cast<int64_t>(4));
            CHECK(!p.branched_tree_supported);
        }
    }
}

TEST(reject_wrong_dtype) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);

    rad_buf bf16_buf = decl_test_buf(b, "bf16_buf", RAD_BF16, 2, 16, 9);
    rad_buf f32_buf  = decl_test_buf(b, "f32_buf",  RAD_F32,  2, 16, 9);
    rad_buf i64_buf  = decl_test_buf(b, "i64_buf",  RAD_I64,  2, 16, 9);
    rad_buf i8_buf   = decl_test_buf(b, "i8_buf",   RAD_I8,   2, 16, 9);

    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), bf16_buf, 4), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), f32_buf,  4), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), i64_buf,  4), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), i8_buf,   4), RAD_E_INVAL);

    CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
}

TEST(reject_wrong_rank) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);

    rad_buf rank1 = decl_test_buf(b, "rank1", RAD_I32, 1, 16 * 9);
    rad_buf rank3 = decl_test_buf(b, "rank3", RAD_I32, 3, 16, 3, 3);

    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), rank1, 4), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), rank3, 4), RAD_E_INVAL);

    CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
}

TEST(reject_wrong_row_width) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);

    // For capacity = 4, expected width = 1 + 2 * 4 = 9.
    rad_buf w8  = decl_test_buf(b, "w8",  RAD_I32, 2, 16, 8);
    rad_buf w10 = decl_test_buf(b, "w10", RAD_I32, 2, 16, 10);
    rad_buf w1  = decl_test_buf(b, "w1",  RAD_I32, 2, 16, 1);

    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), w8,  4), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), w10, 4), RAD_E_INVAL);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), w1,  4), RAD_E_INVAL);

    CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
}

TEST(reject_insufficient_or_invalid_sequence_rows) {
    Registry reg;
    // max_seqs = 16
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);

    // shape[0] = 15 < max_seqs (16)
    rad_buf small_seq = decl_test_buf(b, "small_seq", RAD_I32, 2, 15, 9);
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), small_seq, 4), RAD_E_INVAL);

    CHECK_EQ(b.program().drafter_tree_buf, static_cast<rad_buf>(0));
}

TEST(reject_after_finished) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);
    rad_buf tree = decl_test_buf(b, "tree", RAD_I32, 2, 16, 9);

    b.finish();
    CHECK_EQ(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree, 4), RAD_E_STATE);
}

TEST(branching_monotonicity) {
    Registry reg;
    Builder b(reg, make_meta(), make_ctx(16, 7));
    setup_block_drafter(b, 7, 8);
    rad_buf tree = decl_test_buf(b, "tree", RAD_I32, 2, 16, 9);

    // Program starts with branched_tree_supported = true
    CHECK(b.program().branched_tree_supported);

    // Explicit veto disables branched tree
    rad_disable_branched_tree(static_cast<RadBuilder*>(&b));
    CHECK(!b.program().branched_tree_supported);

    // Declaring a tree does not re-enable it; veto remains monotonic
    CHECK_OK(rad_declare_drafter_tree(static_cast<RadBuilder*>(&b), tree, 4));
    CHECK(!b.program().branched_tree_supported);
}

TEST(c_header_compatibility) {
    // Verify C function pointer signature and invocation
    int (*c_fn)(RadBuilder*, rad_buf, int64_t) = &rad_declare_drafter_tree;
    CHECK(c_fn != nullptr);

    // Test calling via C function pointer with null builder
    int res = c_fn(nullptr, 0, 4);
    CHECK_EQ(res, RAD_E_INVAL);
}

RAD_TEST_MAIN()
