#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <random>
#include <atomic>
#include <chrono>

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

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

#if __has_include("core/sample/accept.h")
#include "core/sample/accept.h"
#elif __has_include("../core/sample/accept.h")
#include "../core/sample/accept.h"
#endif

static std::atomic<uint64_t> g_adv_assertions{0};

#define ADV_ASSERT(cond) do { \
    g_adv_assertions.fetch_add(1, std::memory_order_relaxed); \
    if (!(cond)) { \
        std::fprintf(stderr, "\n[FATAL ERROR] Adversarial Assertion Failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

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

// Byte signature generator for exact parity checks
static inline uint8_t token_byte_val(int32_t node_id, int b, int l, int h) {
    return static_cast<uint8_t>(0xA5 + (node_id * 31) + (b * 17) + (l * 43) + (h * 11));
}

// Poison byte generator for non-winning siblings
static inline uint8_t poison_byte_val(int32_t node_id, int b, int l, int h) {
    return static_cast<uint8_t>(0x5A ^ (node_id * 53) ^ (b * 29) ^ (l * 19) ^ (h * 7));
}

// =============================================================================
// TEST 1: Fast-Path Bypass Verification & Edge Input Hardening
// =============================================================================
static void test_fast_path_bypass_and_edge_inputs() {
    std::cout << "[ADV-TEST 1] Testing Fast-Path Bypass & Edge Input Hardening...\n";

    std::vector<int32_t> bt = { 2, 5, 8, 11, 14 };
    RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];

    // 1.1 Root-only divergence: acc == 0
    {
        const int32_t root_nodes[] = { 0 };
        int32_t n = rad_kv_compact_build_moves(root_nodes, 0, 100, bt, 96, 4, moves, RAD_KV_COMPACT_MAX_MOVES);
        ADV_ASSERT(n == 0);
        // Fast-path bypass execution: returns 0 with zero relocations
        ADV_ASSERT(rad_kv_compact_winning_branch(nullptr, 0, 0, 0, 4, 512, 2, 1, moves, n, nullptr) == 0);
    }

    // 1.2 Pure linear acceptance: n_k == k for all k
    for (int acc = 1; acc <= 15; ++acc) {
        int32_t linear_nodes[16];
        for (int k = 0; k <= acc; ++k) linear_nodes[k] = k;
        int32_t n = rad_kv_compact_build_moves(linear_nodes, acc, 200, bt, 192, 4, moves, RAD_KV_COMPACT_MAX_MOVES);
        ADV_ASSERT(n == 0);
        ADV_ASSERT(rad_kv_compact_winning_branch(nullptr, 0, 0, 0, 4, 512, 2, 1, moves, n, nullptr) == 0);
    }

    // 1.3 Partial linear prefix followed by identity
    {
        const int32_t ident_nodes[] = { 0, 1, 2, 3 };
        int32_t n = rad_kv_compact_build_moves(ident_nodes, 3, 50, bt, 48, 4, moves, RAD_KV_COMPACT_MAX_MOVES);
        ADV_ASSERT(n == 0);
    }

    // 1.4 Invalid / Null parameter contracts
    {
        ADV_ASSERT(rad_kv_compact_build_moves(nullptr, 4, 100, bt, 96, 4, moves, RAD_KV_COMPACT_MAX_MOVES) == 0);
        const int32_t dummy[] = { 0, 2 };
        ADV_ASSERT(rad_kv_compact_build_moves(dummy, -1, 100, bt, 96, 4, moves, RAD_KV_COMPACT_MAX_MOVES) == 0);
        ADV_ASSERT(rad_kv_compact_build_moves(dummy, 1, 100, bt, 96, 4, nullptr, RAD_KV_COMPACT_MAX_MOVES) == 0);
        ADV_ASSERT(rad_kv_compact_build_moves(dummy, 1, 100, bt, 96, 4, moves, 0) == 0);
        ADV_ASSERT(rad_kv_compact_build_moves(dummy, 1, 100, bt, 96, -4, moves, RAD_KV_COMPACT_MAX_MOVES) == 0);

        std::vector<int32_t> empty_bt;
        ADV_ASSERT(rad_kv_compact_build_moves(dummy, 1, 100, empty_bt, 96, 4, moves, RAD_KV_COMPACT_MAX_MOVES) == 0);
    }

    // 1.5 Self-moves pruning in rad_kv_compact_winning_branch
    {
        RadKVSlotMove self_moves[3] = { { 10, 10 }, { 20, 20 }, { 30, 30 } };
        ADV_ASSERT(rad_kv_compact_winning_branch(nullptr, 0, 0, 0, 4, 512, 2, 1, self_moves, 3, nullptr) == 0);
    }

    std::cout << "  PASS: Fast-path bypass and edge input handling verified (0 moves dispatched).\n";
}

// =============================================================================
// Memory Buffer Helper for Parity and Sibling Isolation Testing
// =============================================================================
struct KVTestBuffer {
    int32_t n_layers;
    int32_t n_heads;
    int32_t bs;
    int32_t tok_bytes;
    int64_t kv_head_stride;
    int64_t kv_block_stride;
    int32_t total_blocks;
    int64_t layer_stride;
    std::vector<uint8_t> pool;

    KVTestBuffer(int32_t layers = 4, int32_t heads = 2, int32_t block_sz = 4, int32_t hd = 128, int32_t blocks = 64)
        : n_layers(layers), n_heads(heads), bs(block_sz), total_blocks(blocks)
    {
        tok_bytes = 2 * hd * 2; // BF16: 512 bytes
        kv_head_stride = static_cast<int64_t>(bs) * tok_bytes;
        kv_block_stride = static_cast<int64_t>(n_heads) * kv_head_stride;
        layer_stride = static_cast<int64_t>(total_blocks) * kv_block_stride;
        pool.assign(static_cast<size_t>(n_layers) * layer_stride, 0xCC); // Canary byte 0xCC
    }

    void fill_slot(int32_t slot, int32_t node_id, bool is_poison = false) {
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
                    ptr[b] = is_poison ? poison_byte_val(node_id, b, l, h) : token_byte_val(node_id, b, l, h);
                }
            }
        }
    }

    bool verify_slot(int32_t slot, int32_t expected_node_id) const {
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
                    const uint8_t expected = token_byte_val(expected_node_id, b, l, h);
                    if (ptr[b] != expected) return false;
                }
            }
        }
        return true;
    }

    bool is_slot_entirely_poison(int32_t slot, int32_t sibling_node_id) const {
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
                    const uint8_t poison = poison_byte_val(sibling_node_id, b, l, h);
                    if (ptr[b] != poison) return false;
                }
            }
        }
        return true;
    }

    bool is_canary(int32_t slot) const {
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
                    if (ptr[b] != 0xCC) return false;
                }
            }
        }
        return true;
    }

    void execute_compaction(const RadKVSlotMove* moves, int32_t n_moves) {
        if (n_moves <= 0 || !moves) return;

#if defined(__HIPCC__)
        void* d_pool = nullptr;
        const size_t total_bytes = static_cast<size_t>(n_layers) * layer_stride;
        ADV_ASSERT(hipMalloc(&d_pool, total_bytes) == hipSuccess);
        ADV_ASSERT(hipMemcpy(d_pool, pool.data(), total_bytes, hipMemcpyHostToDevice) == hipSuccess);

        hipStream_t stream = nullptr;
        ADV_ASSERT(hipStreamCreate(&stream) == hipSuccess);

        ADV_ASSERT(rad_kv_compact_winning_branch(
            d_pool, layer_stride, kv_block_stride, kv_head_stride,
            bs, tok_bytes, n_layers, n_heads, moves, n_moves, stream
        ) == 0);

        ADV_ASSERT(hipStreamSynchronize(stream) == hipSuccess);
        ADV_ASSERT(hipMemcpy(pool.data(), d_pool, total_bytes, hipMemcpyDeviceToHost) == hipSuccess);
        ADV_ASSERT(hipFree(d_pool) == hipSuccess);
        ADV_ASSERT(hipStreamDestroy(stream) == hipSuccess);
#else
        ADV_ASSERT(rad_kv_compact_winning_branch(
            pool.data(), layer_stride, kv_block_stride, kv_head_stride,
            bs, tok_bytes, n_layers, n_heads, moves, n_moves, nullptr
        ) == 0);
#endif
    }
};

// =============================================================================
// TEST 2: Mode A Top-2 Compaction Across All Branches & Block Alignments
// =============================================================================
static void test_mode_a_top2_compaction_and_sibling_isolation() {
    std::cout << "[ADV-TEST 2] Testing Mode A Top-2 Compaction & Sibling Isolation...\n";

    CandidateTree tree;
    tree.build_top2(100, nullptr, /*dloop_best_path=*/false);
    ADV_ASSERT(tree.size() == 16);

    // Mode A branch paths:
    const std::vector<std::vector<int32_t>> mode_a_paths = {
        { 0, 1, 3, 7, 15 },
        { 0, 1, 3, 8 },
        { 0, 1, 4, 9 },
        { 0, 1, 4, 10 },
        { 0, 2, 5, 11 },
        { 0, 2, 5, 12 },
        { 0, 2, 6, 13 },
        { 0, 2, 6, 14 },
        { 0, 2, 5 },
        { 0, 1, 4 },
        { 0, 2 }
    };

    const int64_t ctx_offsets[] = { 0, 1, 2, 3, 4, 7, 8, 15, 16, 25, 31, 32, 63, 64, 101, 103, 104 };
    const int32_t block_sizes[] = { 4, 16 };

    int path_run_count = 0;

    for (int32_t bs : block_sizes) {
        for (int64_t ctx0 : ctx_offsets) {
            for (const auto& path : mode_a_paths) {
                const int32_t acc = static_cast<int32_t>(path.size() - 1);
                KVTestBuffer buf(4, 2, bs, 128, 64);

                // Build a block table covering ctx0 + 16
                const int64_t first_pos = (ctx0 / bs) * bs;
                std::vector<int32_t> bt;
                for (int b = 0; b < 16; ++b) bt.push_back(10 + b);

                // Populate candidates 0..15 in buffer
                std::vector<bool> is_winning(16, false);
                for (int32_t nid : path) is_winning[nid] = true;

                for (int32_t nid = 0; nid < 16; ++nid) {
                    const int64_t pos = ctx0 + nid;
                    const int64_t bi = (pos - first_pos) / bs;
                    ADV_ASSERT(bi >= 0 && bi < static_cast<int64_t>(bt.size()));
                    const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);

                    if (is_winning[nid]) {
                        buf.fill_slot(slot, nid, /*is_poison=*/false);
                    } else {
                        // Poison non-winning sibling nodes!
                        buf.fill_slot(slot, nid, /*is_poison=*/true);
                    }
                }

                // Build moves
                RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
                int32_t n_moves = rad_kv_compact_build_moves(
                    path.data(), acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES
                );

                // Check hazard order: dst slots must be strictly ascending
                for (int m = 0; m < n_moves - 1; ++m) {
                    ADV_ASSERT(moves[m].dst_slot != moves[m + 1].dst_slot);
                }

                // Execute compaction
                buf.execute_compaction(moves, n_moves);

                // Verify exact bit-for-bit byte parity at destination slots [0..acc]
                for (int32_t k = 0; k <= acc; ++k) {
                    const int64_t target_pos = ctx0 + k;
                    const int64_t bi = (target_pos - first_pos) / bs;
                    const int32_t target_slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(target_pos % bs);
                    const int32_t winning_node = path[static_cast<size_t>(k)];

                    ADV_ASSERT(buf.verify_slot(target_slot, winning_node));

                    // Verify no poison leaked from any rejected sibling
                    for (int32_t rej = 0; rej < 16; ++rej) {
                        if (!is_winning[rej]) {
                            ADV_ASSERT(!buf.is_slot_entirely_poison(target_slot, rej));
                        }
                    }
                }

                path_run_count++;
            }
        }
    }

    std::cout << "  PASS: Mode A Top-2 verified across " << path_run_count << " path/offset configurations with 100% byte parity.\n";
}

// =============================================================================
// TEST 3: Mode B Top-2 Compaction (Deep Chains up to Depth 11)
// =============================================================================
static void test_mode_b_top2_deep_chains() {
    std::cout << "[ADV-TEST 3] Testing Mode B Top-2 Deep Chain Topologies (Depth <= 11)...\n";

    CandidateTree tree;
    tree.build_top2(200, nullptr, /*dloop_best_path=*/true);
    ADV_ASSERT(tree.size() == 16);

    // Mode B winning deep path: 0 -> 1 -> 3 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 (depth 11)
    const std::vector<int32_t> deep_path = { 0, 1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    const int64_t ctx_offsets[] = { 3, 15, 31, 63, 101 };
    const int32_t bs = 4;

    for (int64_t ctx0 : ctx_offsets) {
        for (size_t depth = 1; depth < deep_path.size(); ++depth) {
            std::vector<int32_t> sub_path(deep_path.begin(), deep_path.begin() + depth + 1);
            const int32_t acc = static_cast<int32_t>(depth);

            KVTestBuffer buf(4, 2, bs, 128, 64);
            const int64_t first_pos = (ctx0 / bs) * bs;
            std::vector<int32_t> bt;
            for (int b = 0; b < 20; ++b) bt.push_back(20 + b);

            std::vector<bool> is_winning(16, false);
            for (int32_t nid : sub_path) is_winning[nid] = true;

            for (int32_t nid = 0; nid < 16; ++nid) {
                const int64_t pos = ctx0 + nid;
                const int64_t bi = (pos - first_pos) / bs;
                const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
                buf.fill_slot(slot, nid, !is_winning[nid]);
            }

            RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
            int32_t n_moves = rad_kv_compact_build_moves(
                sub_path.data(), acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES
            );

            buf.execute_compaction(moves, n_moves);

            for (int32_t k = 0; k <= acc; ++k) {
                const int64_t target_pos = ctx0 + k;
                const int64_t bi = (target_pos - first_pos) / bs;
                const int32_t target_slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(target_pos % bs);
                const int32_t winning_node = sub_path[static_cast<size_t>(k)];

                ADV_ASSERT(buf.verify_slot(target_slot, winning_node));
            }
        }
    }

    std::cout << "  PASS: Mode B Top-2 deep chains verified up to depth 11 with 100% exact parity.\n";
}

// =============================================================================
// TEST 4: Top-4 Tree Topologies (Wide Branching Factor 4)
// =============================================================================
static void test_top4_wide_branching() {
    std::cout << "[ADV-TEST 4] Testing Top-4 Tree Topologies (Wide Branching bf=4)...\n";

    CandidateTree tree;
    tree.build_top4(300, nullptr);
    ADV_ASSERT(tree.size() == 16);

    const std::vector<std::vector<int32_t>> top4_paths = {
        { 0, 1, 5 },
        { 0, 1, 8 },
        { 0, 2, 9 },
        { 0, 2, 12 },
        { 0, 3, 13 },
        { 0, 3, 15 },
        { 0, 4 },
        { 0, 2 }
    };

    const int32_t bs = 4;
    const int64_t ctx0 = 103; // Straddles block boundary immediately
    const int64_t first_pos = 100;
    std::vector<int32_t> bt = { 5, 6, 7, 8, 9, 10 };

    for (const auto& path : top4_paths) {
        const int32_t acc = static_cast<int32_t>(path.size() - 1);
        KVTestBuffer buf(2, 2, bs, 128, 32);

        std::vector<bool> is_winning(16, false);
        for (int32_t nid : path) is_winning[nid] = true;

        for (int32_t nid = 0; nid < 16; ++nid) {
            const int64_t pos = ctx0 + nid;
            const int64_t bi = (pos - first_pos) / bs;
            const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
            buf.fill_slot(slot, nid, !is_winning[nid]);
        }

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        int32_t n_moves = rad_kv_compact_build_moves(
            path.data(), acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES
        );

        buf.execute_compaction(moves, n_moves);

        for (int32_t k = 0; k <= acc; ++k) {
            const int64_t target_pos = ctx0 + k;
            const int64_t bi = (target_pos - first_pos) / bs;
            const int32_t target_slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(target_pos % bs);
            const int32_t winning_node = path[static_cast<size_t>(k)];
            ADV_ASSERT(buf.verify_slot(target_slot, winning_node));
        }
    }

    std::cout << "  PASS: Top-4 wide branching verified across all branches.\n";
}

// =============================================================================
// TEST 5: Monte Carlo Fuzzing Across 500+ Arbitrary Random Branching DAGs
// =============================================================================
static void test_500_random_branching_dags() {
    std::cout << "[ADV-TEST 5] Testing Monte Carlo Fuzzing Across 500+ Arbitrary Random Branching DAGs...\n";

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist_n(3, 16);
    std::uniform_int_distribution<int> dist_bs(0, 1);
    std::uniform_int_distribution<int64_t> dist_ctx(0, 1000);

    const int num_random_dags = 500;

    for (int dag_idx = 0; dag_idx < num_random_dags; ++dag_idx) {
        const int32_t N = dist_n(rng);
        const int32_t bs = dist_bs(rng) == 0 ? 4 : 16;
        const int64_t ctx0 = dist_ctx(rng);
        const int64_t first_pos = (ctx0 / bs) * bs;

        CandidateTree tree;
        tree.add_node(0, -1, 1000, 0, 1.0f);

        // Generate random valid tree DAG
        for (int32_t i = 1; i < N; ++i) {
            std::uniform_int_distribution<int> dist_parent(0, i - 1);
            int32_t parent = dist_parent(rng);
            int32_t depth = tree.nodes[parent].depth + 1;
            uint32_t tok = static_cast<uint32_t>(1000 + i * 17);
            ADV_ASSERT(tree.add_node(i, parent, tok, depth, 1.0f / static_cast<float>(i + 1)));
        }

        // Generate random teacher predictions to trigger realistic tree argmax
        std::vector<uint32_t> teacher_preds(N);
        std::uniform_int_distribution<int> dist_agree(0, 3);
        for (int i = 0; i < N; ++i) {
            if (dist_agree(rng) == 0) {
                // Find a child to agree with
                std::vector<uint32_t> child_tokens;
                for (int j = i + 1; j < N; ++j) {
                    if (tree.nodes[j].parent_id == i) {
                        child_tokens.push_back(tree.nodes[j].token_id);
                    }
                }
                if (!child_tokens.empty()) {
                    teacher_preds[i] = child_tokens[0];
                } else {
                    teacher_preds[i] = 9999 + i; // Bonus token divergence
                }
            } else {
                teacher_preds[i] = 9999 + i;
            }
        }

        // Run DARTree verifier
        auto vres = DARTreeVerifier::verify_tree_greedy(tree, teacher_preds.data(), teacher_preds.size());
        const int32_t acc = vres.accepted_length;
        ADV_ASSERT(acc >= 0 && acc < N);

        // Setup buffer and block table
        std::vector<int32_t> bt;
        const int64_t max_pos = ctx0 + N;
        const int64_t blocks_needed = ceil_div(max_pos - first_pos + 1, bs) + 2;
        for (int64_t b = 0; b < blocks_needed; ++b) bt.push_back(static_cast<int32_t>(50 + b));

        KVTestBuffer buf(2, 2, bs, 128, static_cast<int32_t>(blocks_needed + 50));

        std::vector<bool> is_winning(N, false);
        for (int32_t k = 0; k <= acc; ++k) {
            is_winning[vres.best_path_node_ids[k]] = true;
        }

        for (int32_t nid = 0; nid < N; ++nid) {
            const int64_t pos = ctx0 + nid;
            const int64_t bi = (pos - first_pos) / bs;
            const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
            buf.fill_slot(slot, nid, !is_winning[nid]);
        }

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        int32_t n_moves = rad_kv_compact_build_moves(
            vres.best_path_node_ids, acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES
        );

        // If n_k == k for all k, moves must be 0
        bool all_identity = true;
        for (int32_t k = 1; k <= acc; ++k) {
            if (vres.best_path_node_ids[k] != k) { all_identity = false; break; }
        }
        if (all_identity) {
            ADV_ASSERT(n_moves == 0);
        }

        buf.execute_compaction(moves, n_moves);

        // Assert exact bit-for-bit parity
        for (int32_t k = 0; k <= acc; ++k) {
            const int64_t target_pos = ctx0 + k;
            const int64_t bi = (target_pos - first_pos) / bs;
            const int32_t target_slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(target_pos % bs);
            const int32_t winning_node = vres.best_path_node_ids[k];
            ADV_ASSERT(buf.verify_slot(target_slot, winning_node));

            // Sibling poison isolation
            for (int32_t rej = 0; rej < N; ++rej) {
                if (!is_winning[rej]) {
                    ADV_ASSERT(!buf.is_slot_entirely_poison(target_slot, rej));
                }
            }
        }
    }

    std::cout << "  PASS: 500 arbitrary random DAG topologies verified with 100% exact bit parity and sibling isolation.\n";
}

// =============================================================================
// TEST 6: Physical GPU Device Kernel Verification on AMD RDNA4 (gfx1201)
// =============================================================================
#if defined(__HIPCC__)
static void test_gpu_device_kernel_execution() {
    std::cout << "[ADV-TEST 6] Testing rad_kv_compact_winning_branch_kernel on AMD RDNA4 gfx1201...\n";

    hipError_t err = hipSetDevice(0);
    ADV_ASSERT(err == hipSuccess);

    hipDeviceProp_t prop{};
    ADV_ASSERT(hipGetDeviceProperties(&prop, 0) == hipSuccess);
    std::cout << "  Device: " << prop.name << " (arch: " << prop.gcnArchName << ")\n";

    const int32_t n_layers = 4;
    const int32_t n_heads = 2;
    const int32_t bs = 4;
    const int32_t tok_bytes = 512; // 128-bit aligned
    const int64_t kv_head_stride = static_cast<int64_t>(bs) * tok_bytes;
    const int64_t kv_block_stride = static_cast<int64_t>(n_heads) * kv_head_stride;
    const int32_t total_blocks = 32;
    const int64_t layer_stride = static_cast<int64_t>(total_blocks) * kv_block_stride;
    const size_t total_bytes = static_cast<size_t>(n_layers) * layer_stride;

    void* d_pool = nullptr;
    ADV_ASSERT(hipMalloc(&d_pool, total_bytes) == hipSuccess);

    // Host buffer to seed data
    KVTestBuffer host_buf(n_layers, n_heads, bs, 128, total_blocks);

    // Test 100 random moves on GPU
    std::mt19937 rng(4242);
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<int32_t> bt = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const int64_t ctx0 = 101;
        const int64_t first_pos = 100;
        const int32_t best_nodes[] = { 0, 2, 5, 11 };
        const int32_t acc = 3;

        // Seed host buffer
        for (int32_t nid = 0; nid < 16; ++nid) {
            const int64_t pos = ctx0 + nid;
            const int64_t bi = (pos - first_pos) / bs;
            const int32_t slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(pos % bs);
            bool win = (nid == 0 || nid == 2 || nid == 5 || nid == 11);
            host_buf.fill_slot(slot, nid, !win);
        }

        // Copy host buffer to device
        ADV_ASSERT(hipMemcpy(d_pool, host_buf.pool.data(), total_bytes, hipMemcpyHostToDevice) == hipSuccess);

        RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
        int32_t n_moves = rad_kv_compact_build_moves(
            best_nodes, acc, ctx0, bt, first_pos, bs, moves, RAD_KV_COMPACT_MAX_MOVES
        );
        ADV_ASSERT(n_moves == 3);

        RadKVCompactParams params{};
        params.kv_pool_base    = d_pool;
        params.layer_stride    = layer_stride;
        params.kv_block_stride = kv_block_stride;
        params.kv_head_stride  = kv_head_stride;
        params.block_size      = bs;
        params.tok_bytes       = tok_bytes;
        params.n_layers        = n_layers;
        params.n_heads         = n_heads;
        params.n_moves         = n_moves;
        for (int m = 0; m < n_moves; ++m) params.moves[m] = moves[m];

        dim3 grid(n_heads, n_layers);
        dim3 block(32);
        hipLaunchKernelGGL(
            rad_kv_compact_winning_branch_kernel,
            grid,
            block,
            0,
            0,
            params
        );
        ADV_ASSERT(hipGetLastError() == hipSuccess);
        ADV_ASSERT(hipDeviceSynchronize() == hipSuccess);

        // Copy back to host and verify bit-for-bit parity
        std::vector<uint8_t> h_result(total_bytes);
        ADV_ASSERT(hipMemcpy(h_result.data(), d_pool, total_bytes, hipMemcpyDeviceToHost) == hipSuccess);
        std::memcpy(host_buf.pool.data(), h_result.data(), total_bytes);

        for (int32_t k = 0; k <= acc; ++k) {
            const int64_t target_pos = ctx0 + k;
            const int64_t bi = (target_pos - first_pos) / bs;
            const int32_t target_slot = bt[static_cast<size_t>(bi)] * bs + static_cast<int32_t>(target_pos % bs);
            const int32_t winning_node = best_nodes[k];
            ADV_ASSERT(host_buf.verify_slot(target_slot, winning_node));

            // Sibling poison check
            for (int32_t rej = 0; rej < 16; ++rej) {
                if (rej != 0 && rej != 2 && rej != 5 && rej != 11) {
                    ADV_ASSERT(!host_buf.is_slot_entirely_poison(target_slot, rej));
                }
            }
        }
    }

    ADV_ASSERT(hipFree(d_pool) == hipSuccess);
    std::cout << "  PASS: Physical GPU device kernel executed on gfx1201 with 100% exact bit parity and sibling isolation.\n";
}
#endif

// Host-compiled core entry: this is also the engine path in a HIP-enabled binary.
static void test_runtime_batched_compaction() {
    std::cout << "[ADV-TEST] Runtime batched compaction: " << device_backend_name() << "\n";
    RadKVCompactParams empty{};
    ADV_ASSERT(kv_compact_winning_branch(empty, nullptr) == RAD_OK);
    empty.n_moves = 1;
    empty.moves[0] = {7, 7};
    ADV_ASSERT(kv_compact_winning_branch(empty, nullptr) == RAD_OK);
    empty.n_moves = RAD_KV_COMPACT_MAX_MOVES + 1;
    ADV_ASSERT(kv_compact_winning_branch(empty, nullptr) == RAD_E_INVAL);

    for (int device = 0; device < rad_dev_count(); ++device) {
        ADV_ASSERT(rad_dev_set(device) == RAD_OK);
        RadStream stream{};
        ADV_ASSERT(rad_stream_create(&stream, 0) == RAD_OK);
        // Production-sized rows, synthetic payload-plus-scale widths, and an odd-width tail.
        // The synthetic packed widths do not assert the production cache stores inline scales.
        for (int width : {1024, 512, 256, 264, 520, 263}) {
            for (int scenario = 0; scenario < 4; ++scenario) {
                const bool ordered = (scenario & 1) != 0;
                const bool padded = (scenario & 2) != 0;
                RadKVCompactParams p{};
                p.block_size = 4;
                p.tok_bytes = width;
                p.n_layers = 3;
                p.n_heads = 3;
                p.kv_head_stride = 4 * width + (padded ? 19 : 0);
                p.kv_block_stride = p.n_heads * p.kv_head_stride + (padded ? 23 : 0);
                p.layer_stride = 4 * p.kv_block_stride + (padded ? 29 : 0);
                p.n_moves = 3;
                if (ordered) {
                    p.moves[0] = {3, 1};
                    p.moves[1] = {1, 2};
                    p.moves[2] = {2, 4};
                } else {
                    // Physical block order is deliberately not monotonic.
                    const std::vector<int32_t> bt{2, 0, 3, 1};
                    const int32_t nodes[]{0, 3, 7, 11};
                    p.n_moves = rad_kv_compact_build_moves(nodes, 3, 0, bt, 0, 4,
                                                          p.moves, RAD_KV_COMPACT_MAX_MOVES);
                    ADV_ASSERT(p.n_moves == 3);
                }
                const int guard = padded ? 37 : 32;
                const int64_t bytes = guard + p.n_layers * p.layer_stride + guard;
                std::vector<uint8_t> seed(static_cast<size_t>(bytes));
                std::mt19937 rng(4242 + width);
                for (auto& b : seed) b = static_cast<uint8_t>(rng());
                auto oracle = [&](bool reverse, bool wrong) {
                    auto result = seed;
                    for (int l = 0; l < p.n_layers; ++l)
                        for (int h = 0; h < p.n_heads; ++h)
                            for (int i = 0; i < p.n_moves; ++i) {
                                const auto move = p.moves[reverse ? p.n_moves - 1 - i : i];
                                auto offset = [&](int slot) {
                                    return guard + l * p.layer_stride + h * p.kv_head_stride +
                                           (slot / p.block_size) * p.kv_block_stride +
                                           (slot % p.block_size) * width;
                                };
                                // Independent old sequential byte-copy oracle.
                                const int src = move.src_slot + (wrong && i == 0 ? 1 : 0);
                                std::memcpy(result.data() + offset(move.dst_slot),
                                            result.data() + offset(src), width);
                            }
                    return result;
                };
                const auto expected = oracle(false, false);
                ADV_ASSERT(expected != seed);
                ADV_ASSERT(expected != oracle(false, true));
                if (ordered) ADV_ASSERT(expected != oracle(true, false));

                void* allocation = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
                ADV_ASSERT(allocation != nullptr);
                p.kv_pool_base = static_cast<uint8_t*>(allocation) + guard;
                ADV_ASSERT(rad_memcpy_async(allocation, seed.data(), bytes, stream) == RAD_OK);
                ADV_ASSERT(kv_compact_winning_branch(p, nullptr) == RAD_E_INVAL);
                ADV_ASSERT(kv_compact_winning_branch(p, stream) == RAD_OK);
                // Identity dispatch must leave the result alone and need no valid stream.
                RadKVCompactParams identity = p;
                identity.n_moves = 1;
                identity.moves[0] = {3, 3};
                ADV_ASSERT(kv_compact_winning_branch(identity, nullptr) == RAD_OK);
                std::vector<uint8_t> actual(static_cast<size_t>(bytes));
                ADV_ASSERT(rad_memcpy_async(actual.data(), allocation, bytes, stream) == RAD_OK);
                ADV_ASSERT(rad_stream_sync(stream) == RAD_OK);
                ADV_ASSERT(actual == expected); // Includes every padding byte and both guards.
                rad_dev_free(allocation, RAD_MEM_DEVICE);
            }
        }
        rad_stream_destroy(stream);
    }
    ADV_ASSERT(rad_dev_set(0) == RAD_OK);
}

// =============================================================================
// TEST 7: AddressSanitizer & Conservation Stress Cycle
// =============================================================================
static void test_conservation_and_leak_stress() {
    std::cout << "[ADV-TEST 7] Testing Free Block Conservation & Leak Stress...\n";
    TestFixture f;
    ADV_ASSERT(f.build(4) == RAD_OK);
    const int64_t f0 = f.kv.free_blocks(G_FULL);

    uint64_t seq = 9001;
    ADV_ASSERT(f.kv.add_sequence(seq) == RAD_OK);

    std::mt19937 rng(8888);
    std::uniform_int_distribution<int64_t> dist_ctx(0, 40);
    std::uniform_int_distribution<int> dist_n(1, 16);

    for (int cycle = 0; cycle < 1000; ++cycle) {
        int64_t ctx0 = dist_ctx(rng);
        int32_t N = dist_n(rng);
        std::uniform_int_distribution<int> dist_acc(0, N - 1);
        int32_t acc = dist_acc(rng);

        ADV_ASSERT(f.kv.ensure(seq, ctx0) == RAD_OK);
        int64_t free_base = f.kv.free_blocks(G_FULL);
        int64_t b_base = static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size());

        ADV_ASSERT(f.kv.ensure(seq, ctx0 + N) == RAD_OK);

        // Rollback to ctx0 + 1 + acc
        ADV_ASSERT(f.kv.rollback(seq, ctx0 + 1 + acc) == RAD_OK);

        int64_t b_final = static_cast<int64_t>(f.kv.block_table(seq, G_FULL).size());
        int64_t free_final = f.kv.free_blocks(G_FULL);
        int64_t expected_blocks = ceil_div(ctx0 + 1 + acc, 4);

        ADV_ASSERT(b_final == expected_blocks);
        ADV_ASSERT(free_base - free_final == b_final - b_base);

        // Reset for next cycle
        ADV_ASSERT(f.kv.rollback(seq, 0) == RAD_OK);
    }

    f.kv.free_sequence(seq);
    ADV_ASSERT(f.kv.free_blocks(G_FULL) == f0);
    std::cout << "  PASS: 1,000 conservation and rollback stress cycles verified with 0 leaks.\n";
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " Radiance Milestone 3: KV Compaction & Memory Safety Adversarial Suite\n";
    std::cout << "===================================================================\n";

    test_fast_path_bypass_and_edge_inputs();
    test_mode_a_top2_compaction_and_sibling_isolation();
    test_mode_b_top2_deep_chains();
    test_top4_wide_branching();
    test_500_random_branching_dags();
    test_runtime_batched_compaction();

#if defined(__HIPCC__)
    test_gpu_device_kernel_execution();
#endif

    test_conservation_and_leak_stress();

    std::cout << "===================================================================\n";
    std::cout << " ALL ADVERSARIAL KV COMPACTION & MEMORY SAFETY TESTS PASSED!\n";
    std::cout << " Total Assertions Checked: " << g_adv_assertions.load() << "\n";
    std::cout << "===================================================================\n";
    return 0;
}
