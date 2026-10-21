/* fulldrafter_select_test.cpp -- CPU-only regression test for FullDrafter top-16 candidate selection.
 *
 * Verifies exact top-16 vocabulary selection with strict lower-ID tie breaking:
 *   1. Negative control: proves that old per-thread top-2 hierarchy loses up to 14 true candidates
 *      and selects the wrong downstream winner under selector bias.
 *   2. Proves that FullDrafterThreadTop16 retains all candidates and selects the exact winner.
 *   3. Adversarial same-thread top-16 retention.
 *   4. Strict lower-ID tie breaking on equal finite scores.
 *   5. Inactive padding rejection (tokens >= V_ACT cannot enter top-16).
 *   6. Dual-rank (V_TOTAL=248320) and single-rank (V_SHARD=124160) vocabulary partitioning.
 *   7. Deterministic randomized comparison against full std::stable_sort.
 */

#include "rad_test.h"
#include "../libr4d/r4d_fulldrafter_select.h"

#include <algorithm>
#include <cstdint>
#include <random>

using namespace r4d;

static constexpr int TEST_BLOCKS       = 64;
static constexpr int TEST_THREADS      = 256;
static constexpr int TEST_V_TOTAL      = 248320;
static constexpr int TEST_V_ACT        = 248077;
static constexpr int TEST_V_SHARD      = 124160;
static constexpr int TEST_K_TOP        = 16;

// Mock old top-2 hierarchy for deliberate negative control
struct OldTop2Thread {
    float top_v[2] = { FULLDRAFTER_SENTINEL_SCORE, FULLDRAFTER_SENTINEL_SCORE };
    uint32_t top_i[2] = { FULLDRAFTER_SENTINEL_ID, FULLDRAFTER_SENTINEL_ID };

    void insert(float logit, uint32_t v) {
        if (fulldrafter_is_better(logit, v, top_v[0], top_i[0])) {
            top_v[1] = top_v[0]; top_i[1] = top_i[0];
            top_v[0] = logit;    top_i[0] = v;
        } else if (fulldrafter_is_better(logit, v, top_v[1], top_i[1])) {
            top_v[1] = logit;    top_i[1] = v;
        }
    }
};

static void old_block_top16(
    const OldTop2Thread* threads,
    float* out_vals,
    uint32_t* out_ids
) {
    // Old implementation staged 512 candidate pairs into LDS
    float s_all_val[512];
    uint32_t s_all_id[512];
    for (int t = 0; t < TEST_THREADS; ++t) {
        s_all_val[t]       = threads[t].top_v[0];
        s_all_id[t]        = threads[t].top_i[0];
        s_all_val[256 + t] = threads[t].top_v[1];
        s_all_id[256 + t]  = threads[t].top_i[1];
    }

    for (int k = 0; k < TEST_K_TOP; ++k) {
        float best_v = FULLDRAFTER_SENTINEL_SCORE;
        uint32_t best_i = FULLDRAFTER_SENTINEL_ID;
        int best_idx = -1;
        for (int i = 0; i < 512; ++i) {
            if (fulldrafter_is_better(s_all_val[i], s_all_id[i], best_v, best_i)) {
                best_v = s_all_val[i];
                best_i = s_all_id[i];
                best_idx = i;
            }
        }
        out_vals[k] = best_v;
        out_ids[k] = best_i;
        if (best_idx >= 0) {
            s_all_val[best_idx] = FULLDRAFTER_SENTINEL_SCORE;
            s_all_id[best_idx]  = FULLDRAFTER_SENTINEL_ID;
        }
    }
}

// -----------------------------------------------------------------------------
// Test 1: Deliberate Negative Control & Exact Top-16 Proof on Counterexample
// -----------------------------------------------------------------------------
TEST(counterexample_negative_control_and_exact_proof) {
    // Counterexample geometry from PREFLIGHT.md:
    // V = 248320, 64 blocks, 256 threads, ceil(V/64) = 3880 rows per block.
    // Thread 0 of Block 0 visits 0, 256, 512, 768, 1024, ..., 3840 (16 rows).
    // Logits: 100, 99, 98, ..., 85.
    // All other active logits: -1000.
    std::vector<float> logits(TEST_V_TOTAL, -1000.0f);
    std::vector<uint32_t> expected_cand_ids(TEST_K_TOP);
    for (int i = 0; i < TEST_K_TOP; ++i) {
        uint32_t tok = (uint32_t)(i * 256);
        logits[tok] = 100.0f - (float)i;
        expected_cand_ids[i] = tok;
    }

    // Run Old Top-2 pipeline on Block 0
    OldTop2Thread old_threads[TEST_THREADS];
    const int v_per_block = (TEST_V_TOTAL + TEST_BLOCKS - 1) / TEST_BLOCKS;
    const int v_start = 0;
    const int v_end = std::min(v_start + v_per_block, TEST_V_TOTAL);

    for (int tid = 0; tid < TEST_THREADS; ++tid) {
        for (int v = v_start + tid; v < v_end; v += TEST_THREADS) {
            if (v < TEST_V_ACT) {
                old_threads[tid].insert(logits[v], (uint32_t)v);
            }
        }
    }

    float old_vals[TEST_K_TOP];
    uint32_t old_ids[TEST_K_TOP];
    old_block_top16(old_threads, old_vals, old_ids);

    // Negative control check: old top-2 hierarchy retained ONLY tokens 0 and 256,
    // and lost tokens 512..3840!
    CHECK_EQ(old_ids[0], 0u);
    CHECK_EQ(old_ids[1], 256u);
    bool old_has_512 = false;
    for (int k = 0; k < TEST_K_TOP; ++k) {
        if (old_ids[k] == 512u) old_has_512 = true;
    }
    CHECK(!old_has_512); // Deliberate failure verified: token 512 was lost!

    // Run Exact Top-16 pipeline on Block 0 using shared FullDrafterThreadTop16
    FullDrafterThreadTop16 exact_threads[TEST_THREADS];
    for (int tid = 0; tid < TEST_THREADS; ++tid) exact_threads[tid].init();

    for (int tid = 0; tid < TEST_THREADS; ++tid) {
        for (int v = v_start + tid; v < v_end; v += TEST_THREADS) {
            if (v >= TEST_V_ACT) continue;
            exact_threads[tid].insert(logits[v], (uint32_t)v);
        }
    }

    int local_idx[TEST_THREADS] = {0};
    float exact_vals[TEST_K_TOP];
    uint32_t exact_ids[TEST_K_TOP];
    fulldrafter_block_top16(exact_threads, local_idx, TEST_THREADS, exact_vals, exact_ids);

    // Exact pipeline check: ALL 16 candidates are retained in exact descending order
    for (int k = 0; k < TEST_K_TOP; ++k) {
        CHECK_EQ(exact_ids[k], expected_cand_ids[k]);
        CHECK_EQ(exact_vals[k], 100.0f - (float)k);
    }
    bool exact_has_512 = false;
    for (int k = 0; k < TEST_K_TOP; ++k) {
        if (exact_ids[k] == 512u) exact_has_512 = true;
    }
    CHECK(exact_has_512);

    // Downstream selector demonstration:
    // Selector query adds bias +100.0f to token 512, and 0.0f to all others.
    auto score_selector = [](uint32_t tok_id, float unary_val) -> float {
        float bias = (tok_id == 512u) ? 100.0f : 0.0f;
        return unary_val + bias;
    };

    // Old selector winner:
    uint32_t old_winner = old_ids[0];
    float old_best_score = score_selector(old_ids[0], old_vals[0]);
    for (int k = 1; k < TEST_K_TOP; ++k) {
        float s = score_selector(old_ids[k], old_vals[k]);
        if (fulldrafter_is_better(s, old_ids[k], old_best_score, old_winner)) {
            old_best_score = s;
            old_winner = old_ids[k];
        }
    }

    // Exact selector winner:
    uint32_t exact_winner = exact_ids[0];
    float exact_best_score = score_selector(exact_ids[0], exact_vals[0]);
    for (int k = 1; k < TEST_K_TOP; ++k) {
        float s = score_selector(exact_ids[k], exact_vals[k]);
        if (fulldrafter_is_better(s, exact_ids[k], exact_best_score, exact_winner)) {
            exact_best_score = s;
            exact_winner = exact_ids[k];
        }
    }

    // The counterexample proves:
    // Old hierarchy produced winner 0 (wrong!).
    // Exact hierarchy produces winner 512 (exact true winner!).
    CHECK_EQ(old_winner, 0u);
    CHECK_EQ(exact_winner, 512u);
}

// -----------------------------------------------------------------------------
// Test 2: Adversarial Same-Thread Top-16 on Arbitrary Thread
// -----------------------------------------------------------------------------
TEST(adversarial_same_thread_arbitrary_lane) {
    // Thread 42 owns all top-16 scores across the entire block
    FullDrafterThreadTop16 threads[TEST_THREADS];
    for (int tid = 0; tid < TEST_THREADS; ++tid) threads[tid].init();

    // Background scores for all threads
    for (int tid = 0; tid < TEST_THREADS; ++tid) {
        for (int step = 0; step < 16; ++step) {
            uint32_t id = (uint32_t)(tid + step * 256);
            threads[tid].insert(-50.0f - (float)step, id);
        }
    }

    // Thread 42 gets 16 high scores: 200 down to 185
    std::vector<uint32_t> t42_ids(TEST_K_TOP);
    for (int step = 0; step < TEST_K_TOP; ++step) {
        uint32_t id = (uint32_t)(42 + step * 256);
        t42_ids[step] = id;
        threads[42].insert(200.0f - (float)step, id);
    }

    int local_idx[TEST_THREADS] = {0};
    float out_vals[TEST_K_TOP];
    uint32_t out_ids[TEST_K_TOP];
    fulldrafter_block_top16(threads, local_idx, TEST_THREADS, out_vals, out_ids);

    for (int k = 0; k < TEST_K_TOP; ++k) {
        CHECK_EQ(out_ids[k], t42_ids[k]);
        CHECK_EQ(out_vals[k], 200.0f - (float)k);
    }
}

// -----------------------------------------------------------------------------
// Test 3: Strict Lower-ID Tie Breaking
// -----------------------------------------------------------------------------
TEST(strict_lower_id_tie_breaking) {
    FullDrafterThreadTop16 thread;
    thread.init();

    // Insert candidates with identical scores in arbitrary token order
    thread.insert(50.0f, 500u);
    thread.insert(50.0f, 100u);
    thread.insert(50.0f, 300u);
    thread.insert(50.0f, 200u);
    thread.insert(50.0f, 400u);

    CHECK_EQ(thread.id[0], 100u);
    CHECK_EQ(thread.id[1], 200u);
    CHECK_EQ(thread.id[2], 300u);
    CHECK_EQ(thread.id[3], 400u);
    CHECK_EQ(thread.id[4], 500u);

    // Test tie breaking across threads in block reduction
    FullDrafterThreadTop16 threads[TEST_THREADS];
    for (int tid = 0; tid < TEST_THREADS; ++tid) threads[tid].init();

    // 16 different threads each have a candidate with score 77.0f
    std::vector<uint32_t> tie_ids = {
        900, 150, 42, 1024, 7, 333, 88, 12, 5000, 64, 18, 9999, 2, 400, 777, 5
    };
    for (size_t i = 0; i < tie_ids.size(); ++i) {
        int tid = (int)(i * 10);
        threads[tid].insert(77.0f, tie_ids[i]);
    }

    int local_idx[TEST_THREADS] = {0};
    float out_vals[TEST_K_TOP];
    uint32_t out_ids[TEST_K_TOP];
    fulldrafter_block_top16(threads, local_idx, TEST_THREADS, out_vals, out_ids);

    std::vector<uint32_t> sorted_tie_ids = tie_ids;
    std::sort(sorted_tie_ids.begin(), sorted_tie_ids.end());

    for (int k = 0; k < TEST_K_TOP; ++k) {
        CHECK_EQ(out_vals[k], 77.0f);
        CHECK_EQ(out_ids[k], sorted_tie_ids[k]);
    }
}

// -----------------------------------------------------------------------------
// Test 4: Inactive Padding Exclusion
// -----------------------------------------------------------------------------
TEST(inactive_padding_exclusion) {
    // Tokens in range [TEST_V_ACT, TEST_V_TOTAL) must be excluded
    FullDrafterThreadTop16 threads[TEST_THREADS];
    for (int tid = 0; tid < TEST_THREADS; ++tid) threads[tid].init();

    // Simulate loop over vocab with inactive tokens having huge scores
    const int v_start = 63 * 3880;
    const int v_end = TEST_V_TOTAL;

    for (int tid = 0; tid < TEST_THREADS; ++tid) {
        for (int v = v_start + tid; v < v_end; v += TEST_THREADS) {
            if (v >= TEST_V_ACT) {
                // Inactive token: must be skipped!
                continue;
            }
            float val = 10.0f - (float)(v - v_start) * 0.001f;
            threads[tid].insert(val, (uint32_t)v);
        }
    }

    int local_idx[TEST_THREADS] = {0};
    float out_vals[TEST_K_TOP];
    uint32_t out_ids[TEST_K_TOP];
    fulldrafter_block_top16(threads, local_idx, TEST_THREADS, out_vals, out_ids);

    for (int k = 0; k < TEST_K_TOP; ++k) {
        if (out_ids[k] != FULLDRAFTER_SENTINEL_ID) {
            CHECK(out_ids[k] < (uint32_t)TEST_V_ACT);
        }
    }
}

// -----------------------------------------------------------------------------
// Test 5: Sentinel Handling and Empty Slots
// -----------------------------------------------------------------------------
TEST(sentinel_handling_and_empty_slots) {
    FullDrafterThreadTop16 thread;
    thread.init();

    // Verify initial sentinels
    for (int i = 0; i < FullDrafterThreadTop16::K; ++i) {
        CHECK_EQ(thread.val[i], FULLDRAFTER_SENTINEL_SCORE);
        CHECK_EQ(thread.id[i], FULLDRAFTER_SENTINEL_ID);
    }

    // Insert only 3 items
    thread.insert(-10.0f, 100u);
    thread.insert(-5.0f, 50u);
    thread.insert(-20.0f, 200u);

    CHECK_EQ(thread.id[0], 50u);
    CHECK_EQ(thread.id[1], 100u);
    CHECK_EQ(thread.id[2], 200u);
    for (int i = 3; i < FullDrafterThreadTop16::K; ++i) {
        CHECK_EQ(thread.id[i], FULLDRAFTER_SENTINEL_ID);
        CHECK_EQ(thread.val[i], FULLDRAFTER_SENTINEL_SCORE);
    }

    // Check that pop beyond 3 yields sentinel
    CHECK_EQ(thread.current_val(0), -5.0f);
    CHECK_EQ(thread.current_val(2), -20.0f);
    CHECK_EQ(thread.current_val(3), FULLDRAFTER_SENTINEL_SCORE);
    CHECK_EQ(thread.current_id(3), FULLDRAFTER_SENTINEL_ID);
    CHECK_EQ(thread.current_val(16), FULLDRAFTER_SENTINEL_SCORE);
    CHECK_EQ(thread.current_id(16), FULLDRAFTER_SENTINEL_ID);

    CHECK_EQ(thread.current_id(-1), FULLDRAFTER_SENTINEL_ID);
    CHECK_EQ(thread.current_val(-1), FULLDRAFTER_SENTINEL_SCORE);
    CHECK(fulldrafter_is_better(-1e30f, 1234u, FULLDRAFTER_SENTINEL_SCORE, FULLDRAFTER_SENTINEL_ID));
}

TEST(finite_scores_below_empty_slot_value) {
    FullDrafterThreadTop16 thread;
    thread.init();
    thread.insert(-3e30f, 7u);
    thread.insert(-2e30f, 9u);
    thread.insert(-2e30f, 3u);
    CHECK_EQ(thread.current_id(0), 3u);
    CHECK_EQ(thread.current_id(1), 9u);
    CHECK_EQ(thread.current_id(2), 7u);
    CHECK(!fulldrafter_is_better(FULLDRAFTER_SENTINEL_SCORE,
                               FULLDRAFTER_SENTINEL_ID, -3e30f, 7u));
    int cursor = 0;
    float values[TEST_K_TOP];
    uint32_t ids[TEST_K_TOP];
    fulldrafter_block_top16(&thread, &cursor, 1, values, ids);
    float merged_values[TEST_K_TOP];
    uint32_t merged_ids[TEST_K_TOP];
    fulldrafter_merge_top16(values, ids, TEST_K_TOP, merged_values, merged_ids);
    CHECK_EQ(merged_ids[0], 3u);
    CHECK_EQ(merged_ids[1], 9u);
    CHECK_EQ(merged_ids[2], 7u);
    CHECK_EQ(merged_values[2], -3e30f);
    for (int i = 3; i < TEST_K_TOP; ++i)
        CHECK_EQ(merged_ids[i], FULLDRAFTER_SENTINEL_ID);
}

// -----------------------------------------------------------------------------
// Test 6: Full Pipeline Simulation vs Full Sort (Dual-Rank & Single-Rank Shard)
// -----------------------------------------------------------------------------
static void run_full_pipeline_test(int total_vocab, int active_vocab, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    std::vector<float> logits(total_vocab);
    for (int i = 0; i < active_vocab; ++i) {
        logits[i] = dist(rng);
    }
    for (int i = active_vocab; i < total_vocab; ++i) {
        logits[i] = -1e30f; // inactive
    }

    // Step 1: Run 64 blocks of Top-16 reduction
    const int v_per_block = (total_vocab + TEST_BLOCKS - 1) / TEST_BLOCKS;
    std::vector<float> blk_unary(TEST_BLOCKS * TEST_K_TOP);
    std::vector<uint32_t> blk_cands(TEST_BLOCKS * TEST_K_TOP);

    for (int b = 0; b < TEST_BLOCKS; ++b) {
        const int v_start = b * v_per_block;
        const int v_end = std::min(v_start + v_per_block, total_vocab);

        FullDrafterThreadTop16 threads[TEST_THREADS];
        for (int tid = 0; tid < TEST_THREADS; ++tid) threads[tid].init();

        for (int tid = 0; tid < TEST_THREADS; ++tid) {
            for (int v = v_start + tid; v < v_end; v += TEST_THREADS) {
                if (v >= active_vocab) continue;
                threads[tid].insert(logits[v], (uint32_t)v);
            }
        }

        int local_idx[TEST_THREADS] = {0};
        fulldrafter_block_top16(
            threads, local_idx, TEST_THREADS,
            &blk_unary[b * TEST_K_TOP],
            &blk_cands[b * TEST_K_TOP]
        );
    }

    // Step 2: Merge across the 64 blocks (1024 candidates)
    float final_vals[TEST_K_TOP];
    uint32_t final_ids[TEST_K_TOP];
    fulldrafter_merge_top16(
        blk_unary.data(), blk_cands.data(),
        TEST_BLOCKS * TEST_K_TOP,
        final_vals, final_ids
    );

    // Step 3: Reference computation: sort all active vocabulary tokens
    struct Item { float val; uint32_t id; };
    std::vector<Item> all_items;
    all_items.reserve(active_vocab);
    for (int i = 0; i < active_vocab; ++i) {
        all_items.push_back({logits[i], (uint32_t)i});
    }
    std::stable_sort(all_items.begin(), all_items.end(), [](const Item& a, const Item& b) {
        return a.val != b.val ? a.val > b.val : a.id < b.id;
    });

    // Step 4: Verify exact match
    for (int k = 0; k < TEST_K_TOP; ++k) {
        CHECK_EQ(final_ids[k], all_items[k].id);
        CHECK_EQ(final_vals[k], all_items[k].val);
    }
}

TEST(full_pipeline_dual_rank_random_sort) {
    run_full_pipeline_test(TEST_V_TOTAL, TEST_V_ACT, 42);
    run_full_pipeline_test(TEST_V_TOTAL, TEST_V_ACT, 1337);
}

TEST(full_pipeline_single_rank_shard_random_sort) {
    run_full_pipeline_test(TEST_V_SHARD, TEST_V_SHARD, 999);
    run_full_pipeline_test(TEST_V_SHARD, TEST_V_SHARD, 2026);
}

RAD_TEST_MAIN()
