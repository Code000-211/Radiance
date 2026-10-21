#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <random>

#include "rad_test.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "mem/prefix.h"
#include "mem/vram_budget.h"
#include "device/device.h"
#include "rad_internal.h"

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

#if __has_include("arch/common/rad_kv_compact.h")
#include "arch/common/rad_kv_compact.h"
#elif __has_include("../arch/common/rad_kv_compact.h")
#include "../arch/common/rad_kv_compact.h"
#endif

#undef assert
#define RAD_ASSERT(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)
#define assert(cond) RAD_ASSERT(cond)

using namespace rad;
using namespace radiance;

static Config test_config() {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = 1;
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 32;
    c.checkpoint_interval = 64;
    c.prefix_cache = true;
    return c;
}

static KVGroupInfo null_group() { return KVGroupInfo{}; }

enum : int32_t { G_NULL = 0, G_FULL = 1, G_GDN = 2 };

static std::vector<KVGroupInfo> test_groups(int block_sz = 4) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    {
        KVGroupInfo k;
        k.name = "full";
        k.decl.kind = RAD_KV_FULL;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 128;
        k.block_size = block_sz;
        k.layers = { 0, 1 };
        g.push_back(k);
    }
    {
        KVGroupInfo k;
        k.name = "gdn";
        k.decl.kind = RAD_KV_LINEAR;
        k.decl.dtype = RAD_F32;
        k.decl.n_head_kv = 4;
        k.decl.state_dim[0] = 16;
        k.decl.state_dim[1] = 16;
        k.layers = { 2, 3 };
        g.push_back(k);
    }
    return g;
}

struct TestFixture {
    Pools pools;
    KVManager kv;
    PrefixCache pc;
    Config cfg = test_config();

    int build(int block_sz = 4) {
        cfg.checkpoint_slots = 4;
        int s = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
        if (s < 0) return s;
        s = kv.configure(test_groups(block_sz), cfg, pools.kv());
        if (s < 0) return s;
        return pc.configure(cfg, &kv, cfg.checkpoint_interval);
    }
};

static inline int64_t ceil_div(int64_t a, int64_t b) {
    return b > 0 ? (a + b - 1) / b : 0;
}

// =============================================================================
// TEST 1: Paged KV slot allocation for tree candidates (N <= 16) across block boundaries
// =============================================================================
static void test_paged_kv_slot_allocation_tree_candidates() {
    std::cout << "[TEST 1] Testing Paged KV slot allocation for tree candidates (N <= 16)...\n";
    TestFixture f;
    assert(f.build(4) == RAD_OK);

    uint64_t seq_id = 1001;
    assert(f.kv.add_sequence(seq_id) == RAD_OK);

    const int64_t ctx_samples[] = { 0, 1, 2, 3, 4, 7, 8, 15, 16, 25, 31, 32, 63, 64, 101, 103, 104 };
    const int32_t bs = 4;

    for (int64_t ctx0 : ctx_samples) {
        for (int32_t N = 1; N <= 16; ++N) {
            // Context reaches ctx0
            assert(f.kv.ensure(seq_id, ctx0) == RAD_OK);

            // Speculative reservation for N tree candidate tokens
            assert(f.kv.ensure(seq_id, ctx0 + N) == RAD_OK);

            const auto& bt = f.kv.block_table(seq_id, G_FULL);
            const int64_t first_pos = f.kv.first_block_pos(seq_id, G_FULL);

            int32_t slots[16];
            for (int32_t j = 0; j < N; ++j) {
                const int64_t pos = ctx0 + j;
                const int64_t bi = (pos - first_pos) / bs;
                assert(bi >= 0 && bi < static_cast<int64_t>(bt.size()));
                const int32_t off = static_cast<int32_t>(pos % bs);
                slots[j] = bt[static_cast<size_t>(bi)] * bs + off;
                assert(slots[j] >= 0);
            }

            // Assert uniqueness: all candidate slots must be strictly distinct
            for (int32_t j = 0; j < N; ++j) {
                for (int32_t k = j + 1; k < N; ++k) {
                    assert(slots[j] != slots[k]);
                }
            }

            // Verify block boundary transitions:
            for (int32_t j = 0; j < N - 1; ++j) {
                const int64_t pos_j = ctx0 + j;
                const int64_t pos_next = ctx0 + j + 1;
                const int64_t bi_j = (pos_j - first_pos) / bs;
                const int64_t bi_next = (pos_next - first_pos) / bs;

                if (pos_next % bs == 0) {
                    // Crossed block boundary: bi must increment by exactly 1
                    assert(bi_next == bi_j + 1);
                    assert(bt[static_cast<size_t>(bi_next)] != bt[static_cast<size_t>(bi_j)]);
                } else {
                    // Intra-block contiguous offset
                    assert(bi_next == bi_j);
                    assert(slots[j + 1] == slots[j] + 1);
                }
            }

            // Rollback to ctx0
            assert(f.kv.rollback(seq_id, ctx0) == RAD_OK);
        }
    }

    f.kv.free_sequence(seq_id);
    std::cout << "  PASS: Paged KV slot allocation across block boundaries verified for all N <= 16.\n";
}

// =============================================================================
// TEST 2: In-place winning branch compaction across branching topologies asserting exact byte parity
// =============================================================================
static void test_winning_branch_compaction_byte_parity() {
    std::cout << "[TEST 2] Testing In-place winning branch compaction byte parity across topologies...\n";

    const int32_t n_layers = 4;
    const int32_t n_heads = 2;
    const int32_t bs = 4;
    const int32_t head_dim = 128;
    const int32_t elem_size = 2; // BF16
    const int32_t tok_bytes = 2 * head_dim * elem_size; // 512 bytes per token per head
    const int64_t kv_head_stride = static_cast<int64_t>(bs) * tok_bytes;
    const int64_t kv_block_stride = static_cast<int64_t>(n_heads) * kv_head_stride;
    const int32_t total_blocks = 32;
    const int64_t layer_stride = static_cast<int64_t>(total_blocks) * kv_block_stride;
    const size_t total_bytes = static_cast<size_t>(n_layers) * layer_stride;

    std::vector<uint8_t> pool(total_bytes, 0xAA); // Fill with canary byte 0xAA

    auto fill_token_slot = [&](int32_t slot, uint8_t seed_val) {
        const size_t blk = static_cast<size_t>(slot / bs);
        const size_t off = static_cast<size_t>(slot % bs);
        for (int l = 0; l < n_layers; ++l) {
            uint8_t* layer_base = pool.data() + static_cast<size_t>(l) * layer_stride;
            for (int h = 0; h < n_heads; ++h) {
                const size_t byte_off = blk * static_cast<size_t>(kv_block_stride)
                                      + static_cast<size_t>(h) * static_cast<size_t>(kv_head_stride)
                                      + off * static_cast<size_t>(tok_bytes);
                uint8_t* ptr = layer_base + byte_off;
                for (int b = 0; b < tok_bytes; ++b) {
                    ptr[b] = static_cast<uint8_t>(seed_val + (b * 13) + (l * 37) + (h * 7));
                }
            }
        }
    };

    auto verify_token_slot = [&](int32_t slot, uint8_t seed_val) -> bool {
        const size_t blk = static_cast<size_t>(slot / bs);
        const size_t off = static_cast<size_t>(slot % bs);
        for (int l = 0; l < n_layers; ++l) {
            const uint8_t* layer_base = pool.data() + static_cast<size_t>(l) * layer_stride;
            for (int h = 0; h < n_heads; ++h) {
                const size_t byte_off = blk * static_cast<size_t>(kv_block_stride)
                                      + static_cast<size_t>(h) * static_cast<size_t>(kv_head_stride)
                                      + off * static_cast<size_t>(tok_bytes);
                const uint8_t* ptr = layer_base + byte_off;
                for (int b = 0; b < tok_bytes; ++b) {
                    const uint8_t expected = static_cast<uint8_t>(seed_val + (b * 13) + (l * 37) + (h * 7));
                    if (ptr[b] != expected) return false;
                }
            }
        }
        return true;
    };

    // --- Scenario A: Mode A Top-2 (Winning path: 0 -> 2 -> 5) ---
    // Candidate nodes: 0, 1, 2, 3, 4, 5, 6
    // ctx0 = 101, bs = 4, first_pos = 100
    // block_table maps block indices 0, 1, 2, 3, 4
    {
        std::vector<int32_t> bt = { 5, 6, 7, 8, 9 };
        const int64_t ctx0 = 101;
        const int64_t first_pos = 100;
        const int32_t best_nodes[] = { 0, 2, 5 };
        const int32_t acc = 2;

        // Populate candidates 0..6
        for (int32_t node = 0; node <= 6; ++node) {
            const int64_t pos = ctx0 + node;
            const int64_t bi = (pos - first_pos) / bs;
            const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
            fill_token_slot(slot, static_cast<uint8_t>(10 + node));
        }

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        const int32_t n_moves = rad_kv_compact_build_moves(best_nodes, acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES);
        assert(n_moves == 2);
        // Move 0: src node 2 -> dst position 1
        // Move 1: src node 5 -> dst position 2
        assert(moves[0].dst_slot != moves[0].src_slot);
        assert(moves[1].dst_slot != moves[1].src_slot);
        assert(moves[0].dst_slot < moves[0].src_slot);
        assert(moves[1].dst_slot < moves[1].src_slot);

        // Execute compaction
        assert(rad_kv_compact_winning_branch(
            pool.data(), layer_stride, kv_block_stride, kv_head_stride,
            bs, tok_bytes, n_layers, n_heads, moves, n_moves, nullptr
        ) == 0);

        // Verify exact byte parity:
        // Position 0 (ctx0 + 0) still contains node 0
        const int32_t s0 = bt[static_cast<size_t>((ctx0 + 0 - first_pos) / bs)] * bs + static_cast<int32_t>((ctx0 + 0) % bs);
        assert(verify_token_slot(s0, 10));

        // Position 1 (ctx0 + 1) now contains node 2
        const int32_t s1 = bt[static_cast<size_t>((ctx0 + 1 - first_pos) / bs)] * bs + static_cast<int32_t>((ctx0 + 1) % bs);
        assert(verify_token_slot(s1, 12));

        // Position 2 (ctx0 + 2) now contains node 5
        const int32_t s2 = bt[static_cast<size_t>((ctx0 + 2 - first_pos) / bs)] * bs + static_cast<int32_t>((ctx0 + 2) % bs);
        assert(verify_token_slot(s2, 15));
    }

    // --- Scenario B: Top-4 Branch Alternative (Winning path: 0 -> 3 -> 14) ---
    {
        std::vector<int32_t> bt = { 10, 11, 12, 13, 14, 15 };
        const int64_t ctx0 = 200;
        const int64_t first_pos = 200;
        const int32_t best_nodes[] = { 0, 3, 14 };
        const int32_t acc = 2;

        for (int32_t node = 0; node <= 15; ++node) {
            const int64_t pos = ctx0 + node;
            const int64_t bi = (pos - first_pos) / bs;
            const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
            fill_token_slot(slot, static_cast<uint8_t>(50 + node));
        }

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        const int32_t n_moves = rad_kv_compact_build_moves(best_nodes, acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES);
        assert(n_moves == 2);

        assert(rad_kv_compact_winning_branch(
            pool.data(), layer_stride, kv_block_stride, kv_head_stride,
            bs, tok_bytes, n_layers, n_heads, moves, n_moves, nullptr
        ) == 0);

        const int32_t s0 = bt[static_cast<size_t>((ctx0 + 0 - first_pos) / bs)] * bs + static_cast<int32_t>((ctx0 + 0) % bs);
        const int32_t s1 = bt[static_cast<size_t>((ctx0 + 1 - first_pos) / bs)] * bs + static_cast<int32_t>((ctx0 + 1) % bs);
        const int32_t s2 = bt[static_cast<size_t>((ctx0 + 2 - first_pos) / bs)] * bs + static_cast<int32_t>((ctx0 + 2) % bs);

        assert(verify_token_slot(s0, 50));     // Node 0
        assert(verify_token_slot(s1, 50 + 3));  // Node 3
        assert(verify_token_slot(s2, 50 + 14)); // Node 14
    }

    // --- Scenario C: Deep Multi-Hop Branch (0 -> 2 -> 6 -> 12) ---
    {
        std::vector<int32_t> bt = { 20, 21, 22, 23, 24 };
        const int64_t ctx0 = 402;
        const int64_t first_pos = 400;
        const int32_t best_nodes[] = { 0, 2, 6, 12 };
        const int32_t acc = 3;

        for (int32_t node = 0; node <= 15; ++node) {
            const int64_t pos = ctx0 + node;
            const int64_t bi = (pos - first_pos) / bs;
            const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
            fill_token_slot(slot, static_cast<uint8_t>(100 + node));
        }

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        const int32_t n_moves = rad_kv_compact_build_moves(best_nodes, acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES);
        assert(n_moves == 3);

        assert(rad_kv_compact_winning_branch(
            pool.data(), layer_stride, kv_block_stride, kv_head_stride,
            bs, tok_bytes, n_layers, n_heads, moves, n_moves, nullptr
        ) == 0);

        for (int32_t k = 0; k <= acc; ++k) {
            const int32_t node_expected = best_nodes[k];
            const int64_t pos = ctx0 + k;
            const int32_t s = bt[static_cast<size_t>((pos - first_pos) / bs)] * bs + static_cast<int32_t>(pos % bs);
            assert(verify_token_slot(s, static_cast<uint8_t>(100 + node_expected)));
        }
    }

    std::cout << "  PASS: Winning branch compaction verified with 100% exact byte parity.\n";
}

// =============================================================================
// TEST 3: Fast-path bypass when n_k == k verifying 0 GPU moves
// =============================================================================
static void test_fast_path_bypass_zero_gpu_moves() {
    std::cout << "[TEST 3] Testing Fast-path bypass (0 GPU moves when n_k == k)...\n";

    std::vector<int32_t> bt = { 1, 2, 3, 4 };
    RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];

    // Case 1: Pure linear acceptance [0, 1, 2, 3, 4]
    {
        const int32_t linear_nodes[] = { 0, 1, 2, 3, 4 };
        int32_t n_moves = rad_kv_compact_build_moves(linear_nodes, 4, 10, bt, 8, 4, moves, RAD_KV_COMPACT_MAX_MOVES);
        assert(n_moves == 0);
        // Dispatch wrapper returns 0 immediately with zero moves
        assert(rad_kv_compact_winning_branch(nullptr, 0, 0, 0, 4, 512, 2, 1, moves, n_moves, nullptr) == 0);
    }

    // Case 2: Root-only acceptance (acc == 0)
    {
        const int32_t root_nodes[] = { 0 };
        int32_t n_moves = rad_kv_compact_build_moves(root_nodes, 0, 10, bt, 8, 4, moves, RAD_KV_COMPACT_MAX_MOVES);
        assert(n_moves == 0);
        assert(rad_kv_compact_winning_branch(nullptr, 0, 0, 0, 4, 512, 2, 1, moves, 0, nullptr) == 0);
    }

    // Case 3: Identity single proposal [0, 1]
    {
        const int32_t single_nodes[] = { 0, 1 };
        int32_t n_moves = rad_kv_compact_build_moves(single_nodes, 1, 10, bt, 8, 4, moves, RAD_KV_COMPACT_MAX_MOVES);
        assert(n_moves == 0);
        assert(rad_kv_compact_winning_branch(nullptr, 0, 0, 0, 4, 512, 2, 1, moves, n_moves, nullptr) == 0);
    }

    std::cout << "  PASS: Fast-path bypass successfully verified (0 relocations dispatched).\n";
}

// =============================================================================
// TEST 4: KVManager::rollback verification asserting free block conservation invariant
// =============================================================================
static void test_kv_manager_rollback_conservation_invariant() {
    std::cout << "[TEST 4] Testing KVManager::rollback free block conservation invariant...\n";
    TestFixture f;
    assert(f.build(4) == RAD_OK);

    const int64_t f0 = f.kv.free_blocks(G_FULL);
    uint64_t seq = 2001;
    assert(f.kv.add_sequence(seq) == RAD_OK);

    int test_count = 0;
    for (int64_t ctx0 = 0; ctx0 <= 48; ++ctx0) {
        for (int64_t N = 1; N <= 16; ++N) {
            for (int64_t m = 0; m <= N; ++m) {
                // Reach base context
                assert(f.kv.ensure(seq, ctx0) == RAD_OK);
                int64_t free_base = f.kv.free_blocks(G_FULL);
                int64_t b_base = static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size());
                assert(b_base == ceil_div(ctx0, 4));

                // Speculatively reserve N candidate slots
                assert(f.kv.ensure(seq, ctx0 + N) == RAD_OK);
                int64_t b_spec = static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size());
                assert(b_spec == ceil_div(ctx0 + N, 4));

                // Rollback to ctx0 + m accepted tokens
                assert(f.kv.rollback(seq, ctx0 + m) == RAD_OK);
                int64_t b_final = static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size());
                int64_t free_final = f.kv.free_blocks(G_FULL);

                // Conservation Invariant: FreeBlocks_final == FreeBlocks_0 - Delta_B_retained
                int64_t expected_blocks = ceil_div(ctx0 + m, 4);
                assert(b_final == expected_blocks);
                assert(free_base - free_final == b_final - b_base);

                // Reset back to ctx0 for next cycle
                assert(f.kv.rollback(seq, ctx0) == RAD_OK);
                assert(f.kv.free_blocks(G_FULL) == free_base);
                test_count++;
            }
        }
    }

    f.kv.free_sequence(seq);
    assert(f.kv.free_blocks(G_FULL) == f0);
    std::cout << "  PASS: Exact conservation invariant verified across " << test_count << " configurations.\n";
}

// =============================================================================
// TEST 5: Edge cases: m = 0 (root divergence), m = N (full acceptance), boundary crossings, preemption
// =============================================================================
static void test_edge_cases_m0_mN_boundary_preemption() {
    std::cout << "[TEST 5] Testing Edge cases (m = 0, m = N, block boundary crossings, preemption)...\n";
    TestFixture f;
    assert(f.build(4) == RAD_OK);
    const int64_t f0 = f.kv.free_blocks(G_FULL);

    // Edge Case 1: m = 0 (Total rejection, divergence at root)
    {
        uint64_t s_m0 = 5001;
        assert(f.kv.add_sequence(s_m0) == RAD_OK);
        assert(f.kv.ensure(s_m0, 17) == RAD_OK); // 5 blocks
        int64_t free_17 = f.kv.free_blocks(G_FULL);

        for (int n = 1; n <= 16; ++n) {
            assert(f.kv.ensure(s_m0, 17 + n) == RAD_OK);
            assert(f.kv.rollback(s_m0, 17) == RAD_OK);
            assert(f.kv.free_blocks(G_FULL) == free_17);
            assert(static_cast<int64_t>(f.kv.block_table(s_m0, G_FULL).size()) == ceil_div(17, 4));
        }
        f.kv.free_sequence(s_m0);
        assert(f.kv.free_blocks(G_FULL) == f0);
    }

    // Edge Case 2: m = N (100% full acceptance)
    {
        uint64_t s_mN = 5002;
        assert(f.kv.add_sequence(s_mN) == RAD_OK);
        assert(f.kv.ensure(s_mN, 15) == RAD_OK);
        int64_t free_15 = f.kv.free_blocks(G_FULL);

        for (int n = 1; n <= 16; ++n) {
            assert(f.kv.ensure(s_mN, 15 + n) == RAD_OK);
            // Rollback to 15 + n (full acceptance)
            assert(f.kv.rollback(s_mN, 15 + n) == RAD_OK);
            assert(static_cast<int64_t>(f.kv.block_table(s_mN, G_FULL).size()) == ceil_div(15 + n, 4));
            // Rollback back to 15
            assert(f.kv.rollback(s_mN, 15) == RAD_OK);
            assert(f.kv.free_blocks(G_FULL) == free_15);
        }
        f.kv.free_sequence(s_mN);
        assert(f.kv.free_blocks(G_FULL) == f0);
    }

    // Edge Case 3: Block boundary alignments (ctx0 % 4 == 0 vs ctx0 % 4 == 3)
    {
        uint64_t s_bound = 5003;
        assert(f.kv.add_sequence(s_bound) == RAD_OK);

        // Starts exactly on boundary: ctx0 = 16
        assert(f.kv.ensure(s_bound, 16) == RAD_OK);
        assert(static_cast<int64_t>(f.kv.block_table(s_bound, G_FULL).size()) == 4);
        assert(f.kv.ensure(s_bound, 17) == RAD_OK); // cross by 1 token -> block 5 allocated
        assert(static_cast<int64_t>(f.kv.block_table(s_bound, G_FULL).size()) == 5);
        assert(f.kv.rollback(s_bound, 16) == RAD_OK); // pruned block 5
        assert(static_cast<int64_t>(f.kv.block_table(s_bound, G_FULL).size()) == 4);

        // Starts with 1 token left in block: ctx0 = 15
        assert(f.kv.ensure(s_bound, 15) == RAD_OK);
        assert(static_cast<int64_t>(f.kv.block_table(s_bound, G_FULL).size()) == 4);
        assert(f.kv.ensure(s_bound, 20) == RAD_OK); // 15..19 -> blocks 4 and 5
        assert(static_cast<int64_t>(f.kv.block_table(s_bound, G_FULL).size()) == 5);
        assert(f.kv.rollback(s_bound, 16) == RAD_OK); // 1 token accepted, fills block 4
        assert(static_cast<int64_t>(f.kv.block_table(s_bound, G_FULL).size()) == 4);

        f.kv.free_sequence(s_bound);
        assert(f.kv.free_blocks(G_FULL) == f0);
    }

    // Edge Case 4: Sudden sequence preemption mid-speculation
    {
        for (int i = 0; i < 50; ++i) {
            uint64_t s_preempt = 6000 + i;
            assert(f.kv.add_sequence(s_preempt) == RAD_OK);
            assert(f.kv.ensure(s_preempt, 25) == RAD_OK);
            assert(f.kv.ensure(s_preempt, 25 + 16) == RAD_OK);
            assert(f.kv.free_blocks(G_FULL) < f0);

            // Preempt sequence
            assert(f.kv.preempt(s_preempt) == RAD_OK);
            assert(f.kv.free_blocks(G_FULL) == f0);
            assert(f.kv.seq_tokens(s_preempt) == 0);
            assert(static_cast<int64_t>(f.kv.block_table(s_preempt, G_FULL).size()) == 0);

            // Clean sequence
            f.kv.free_sequence(s_preempt);
            assert(f.kv.free_blocks(G_FULL) == f0);
        }
    }

    std::cout << "  PASS: Edge cases (m = 0, m = N, boundary crossings, preemption) verified cleanly.\n";
}

// =============================================================================
// TEST 6: 1,000+ repeated spec verify -> compact -> rollback cycles asserting 0 leaks
// =============================================================================
static void test_1000_cycles_stress_verify_compact_rollback() {
    std::cout << "[TEST 6] Testing 1,000+ repeated spec verify -> compact -> rollback cycles...\n";
    TestFixture f;
    assert(f.build(4) == RAD_OK);
    const int64_t f0 = f.kv.free_blocks(G_FULL);

    const int32_t bs = 4;
    const int32_t tok_bytes = 256;
    const int64_t layer_stride = 64 * 4 * tok_bytes;
    const int64_t kv_block_stride = bs * tok_bytes;
    std::vector<uint8_t> mock_pool(layer_stride * 2, 0);

    uint64_t seq = 7777;
    assert(f.kv.add_sequence(seq) == RAD_OK);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_ctx(0, 50);
    std::uniform_int_distribution<int> dist_n(1, 16);
    std::uniform_int_distribution<int> dist_topo(0, 3); // 0=linear, 1=top2, 2=top4, 3=random

    for (int cycle = 0; cycle < 1200; ++cycle) {
        int64_t ctx0 = dist_ctx(rng);
        int32_t N = dist_n(rng);
        int topo = dist_topo(rng);

        assert(f.kv.ensure(seq, ctx0) == RAD_OK);
        assert(f.kv.ensure(seq, ctx0 + N) == RAD_OK);

        // Synthesize best_nodes topology
        int32_t best_nodes[17] = { 0 };
        int32_t acc = 0;
        if (N > 1) {
            std::uniform_int_distribution<int> dist_acc(0, N - 1);
            acc = dist_acc(rng);
        }

        if (topo == 0) {
            // Linear
            for (int k = 0; k <= acc; ++k) best_nodes[k] = k;
        } else if (topo == 1) {
            // Top-2 Mode A
            best_nodes[0] = 0;
            for (int k = 1; k <= acc; ++k) {
                best_nodes[k] = std::min(k * 2, N - 1);
            }
        } else if (topo == 2) {
            // Top-4
            best_nodes[0] = 0;
            if (acc >= 1) best_nodes[1] = std::min(3, N - 1);
            for (int k = 2; k <= acc; ++k) {
                best_nodes[k] = std::min(k + 4, N - 1);
            }
        } else {
            // Monotonic non-decreasing
            best_nodes[0] = 0;
            for (int k = 1; k <= acc; ++k) {
                best_nodes[k] = std::max(best_nodes[k - 1] + 1, k);
                if (best_nodes[k] >= N) best_nodes[k] = N - 1;
            }
        }

        const auto& bt = f.kv.block_table(seq, G_FULL);
        const int64_t first_pos = f.kv.first_block_pos(seq, G_FULL);

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        int32_t n_moves = rad_kv_compact_build_moves(
            best_nodes, acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES
        );

        if (n_moves > 0) {
            assert(rad_kv_compact_winning_branch(
                mock_pool.data(), layer_stride, kv_block_stride, 0,
                bs, tok_bytes, 2, 1, moves, n_moves, nullptr
            ) == 0);
        }

        // Rollback unaccepted slots
        assert(f.kv.rollback(seq, ctx0 + 1 + acc) == RAD_OK);

        const int64_t expected_blocks = ceil_div(ctx0 + 1 + acc, 4);
        assert(static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size()) == expected_blocks);

        // Reset for next cycle
        assert(f.kv.rollback(seq, 0) == RAD_OK);
        assert(static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size()) == 0);
    }

    f.kv.free_sequence(seq);
    assert(f.kv.free_blocks(G_FULL) == f0);
    std::cout << "  PASS: 1,200 repeated spec verify -> compact -> rollback cycles completed with 0 leaks.\n";
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " Radiance Milestone 3: KV-Cache Tree Rollback & Compaction Test Suite\n";
    std::cout << "===================================================================\n";

    test_paged_kv_slot_allocation_tree_candidates();
    test_winning_branch_compaction_byte_parity();
    test_fast_path_bypass_zero_gpu_moves();
    test_kv_manager_rollback_conservation_invariant();
    test_edge_cases_m0_mN_boundary_preemption();
    test_1000_cycles_stress_verify_compact_rollback();

    std::cout << "===================================================================\n";
    std::cout << " ALL 6 MILESTONE 3 KV ROLLBACK & COMPACTION TESTS PASSED!\n";
    std::cout << "===================================================================\n";
    return 0;
}
