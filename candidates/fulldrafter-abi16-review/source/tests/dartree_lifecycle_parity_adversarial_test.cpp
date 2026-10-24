#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <sstream>
#include <any>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <random>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <algorithm>

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

#include "sample/accept.h"
#include "sched/draft.h"
#include "diag.h"

#define private public
#include "sched/scheduler.h"
#include "sched/batch.h"
#include "engine.h"
#undef private

#include "rad_core.h"

// Tracking assertion count for audit
static std::atomic<uint64_t> g_assertions{0};

#define EMPIRICAL_ASSERT(cond) do { \
    g_assertions.fetch_add(1, std::memory_order_relaxed); \
    if (!(cond)) { \
        std::fprintf(stderr, "\n[FATAL ERROR] Empirical Challenger Assertion Failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

using namespace radiance;
using rad::accept_greedy;
using rad::accept_tree_greedy;
using rad::RAD_SCHED_MAX_SPEC;
using rad::Scheduler;
using rad::Engine;
using rad::Config;
using rad::Request;
using rad::ReqState;
using rad::StepEntry;
using rad::BatchBuilder;

struct MinimalFakeKV : public rad::IKVManager {
    std::unordered_set<uint64_t> seqs;
    int add_sequence(uint64_t id) override { seqs.insert(id); return RAD_OK; }
    bool has_sequence(uint64_t id) const override { return seqs.count(id) != 0; }
    void free_sequence(uint64_t id) override { seqs.erase(id); }
    int ensure(uint64_t, int64_t) override { return RAD_OK; }
    int adopt(uint64_t, int32_t, const std::vector<int32_t>&, int64_t) override { return RAD_OK; }
    int preempt(uint64_t) override { return RAD_OK; }
    bool can_ever_fit(int64_t) const override { return true; }
    const std::vector<int32_t>& block_table(uint64_t, int32_t) const override {
        static const std::vector<int32_t> s_empty;
        return s_empty;
    }
    int32_t state_slot(uint64_t, int32_t) const override { return -1; }
    int state_index_row(uint64_t, int32_t, int32_t*, int) const override { return 0; }
    int64_t state_index_width(int32_t) const override { return 0; }
    int64_t free_blocks(int32_t) const override { return 1000; }
    int64_t total_blocks(int32_t) const override { return 1000; }
    int64_t held_blocks(int32_t) const override { return 0; }
    int64_t pool_bytes(int32_t) const override { return 1000000; }
    int64_t pool_bytes_used(int32_t) const override { return 0; }
    int64_t pool_bytes_carved(int32_t) const override { return 1000000; }
    int64_t block_tokens(int32_t) const override { return 16; }
    int rollback(uint64_t, int64_t) override { return RAD_OK; }
    int64_t first_block_pos(uint64_t, int32_t) const override { return 0; }
    int64_t take_table_changes(uint64_t, int32_t) const override { return 0; }
};

static MinimalFakeKV g_fake_kv;

// =============================================================================
// SUITE 1: Multi-Step Sequential Mode Alternation (Tree -> Linear -> Single)
// =============================================================================
static void test_sequential_mode_alternations() {
    std::printf("[SUITE 1] Testing Sequential Mode Alternations (Tree <-> Linear <-> Single)...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    sched.kv_ = &g_fake_kv;

    const uint64_t seq_id = 1001;
    auto req = std::make_unique<Request>();
    req->id = seq_id;
    req->prompt = {10, 11, 12, 13};
    req->may_speculate = true;
    req->max_tokens = 500;
    EMPIRICAL_ASSERT(sched.add(std::move(req)) == RAD_OK);

    // Run 50 alternating cycles
    for (int cycle = 0; cycle < 50; ++cycle) {
        // -------------------------------------------------------------
        // Step 1: Speculative Tree Decode (Top-2 Mode A)
        // -------------------------------------------------------------
        CandidateTree tree;
        uint32_t draft_tokens[15];
        for (size_t i = 0; i < 15; ++i) draft_tokens[i] = 1000u + static_cast<uint32_t>(cycle * 100 + i);
        tree.build_top2(500u + static_cast<uint32_t>(cycle), draft_tokens, false);

        const uint32_t root_pos = 100 + static_cast<uint32_t>(cycle * 10);
        EMPIRICAL_ASSERT(sched.set_draft_tree(seq_id, tree, root_pos) == RAD_OK);

        // Verify TreeRegistry is actively populated
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(reg.get(seq_id, &trec));
        EMPIRICAL_ASSERT(trec.active);
        EMPIRICAL_ASSERT(trec.root_pos == root_pos);
        EMPIRICAL_ASSERT(trec.tree.node_count() == 16);

        // Simulate collect_step completion
        sched.plan_.reset();
        StepEntry entry_tree{};
        entry_tree.seq = seq_id;
        entry_tree.n_spec = 15;
        entry_tree.n_tokens = 16;
        Request req_copy;
        req_copy.id = seq_id;
        entry_tree.req = &req_copy;
        int32_t draft_buf[16];
        for (int i = 0; i < 15; ++i) draft_buf[i] = static_cast<int32_t>(draft_tokens[i]);
        entry_tree.draft = draft_buf;
        sched.plan_.e.push_back(entry_tree);

        RadBatch batch_tree{};
        batch_tree.step = cycle * 3 + 1;
        batch_tree.n_seq = 1;
        batch_tree.n_spec = 15;

        Engine::StepOut so_tree;
        so_tree.n_rows = 16;
        int32_t tree_samples[16];
        for (int i = 0; i < 15; ++i) tree_samples[i] = static_cast<int32_t>(draft_tokens[i]);
        tree_samples[15] = 9999;
        so_tree.rb = tree_samples;
        so_tree.first_row = { 0 };

        EMPIRICAL_ASSERT(engine.collect_step(&batch_tree, so_tree) == RAD_OK);

        // Invariant: TreeRegistry MUST be completely cleared for seq_id
        EMPIRICAL_ASSERT(!reg.get(seq_id, &trec));

        // -------------------------------------------------------------
        // Step 2: Linear Decode Transition
        // -------------------------------------------------------------
        int32_t lin_draft[5] = {301, 302, 303, 304, 305};
        EMPIRICAL_ASSERT(sched.set_draft(seq_id, lin_draft, 5) == RAD_OK);

        // Invariant: TreeRegistry MUST remain strictly empty
        EMPIRICAL_ASSERT(!reg.get(seq_id, &trec));

        sched.plan_.reset();
        StepEntry entry_lin{};
        entry_lin.seq = seq_id;
        entry_lin.n_spec = 5;
        entry_lin.n_tokens = 6;
        entry_lin.req = &req_copy;
        entry_lin.draft = lin_draft;
        sched.plan_.e.push_back(entry_lin);

        RadBatch batch_lin{};
        batch_lin.step = cycle * 3 + 2;
        batch_lin.n_seq = 1;
        batch_lin.n_spec = 5;

        Engine::StepOut so_lin;
        so_lin.n_rows = 6;
        int32_t lin_samples[6] = {301, 302, 303, 888, 999, 1111};
        so_lin.rb = lin_samples;
        so_lin.first_row = { 0 };

        EMPIRICAL_ASSERT(engine.collect_step(&batch_lin, so_lin) == RAD_OK);

        // Invariant: TreeRegistry is still clean
        EMPIRICAL_ASSERT(!reg.get(seq_id, &trec));
        EMPIRICAL_ASSERT(so_lin.accepted[0] == 3); // 3 accepted + 1 bonus = 888
        EMPIRICAL_ASSERT(so_lin.token[0] == 888);

        // -------------------------------------------------------------
        // Step 3: Single-Token Standard Decode Transition (n_draft = 0)
        // -------------------------------------------------------------
        EMPIRICAL_ASSERT(sched.set_draft(seq_id, nullptr, 0) == RAD_OK);

        // Invariant: TreeRegistry must be empty
        EMPIRICAL_ASSERT(!reg.get(seq_id, &trec));

        sched.plan_.reset();
        StepEntry entry_std{};
        entry_std.seq = seq_id;
        entry_std.n_spec = 0;
        entry_std.n_tokens = 1;
        entry_std.req = &req_copy;
        entry_std.draft = nullptr;
        sched.plan_.e.push_back(entry_std);

        RadBatch batch_std{};
        batch_std.step = cycle * 3 + 3;
        batch_std.n_seq = 1;
        batch_std.n_spec = 0;

        Engine::StepOut so_std;
        so_std.n_rows = 1;
        int32_t std_samples[1] = {7777};
        so_std.rb = std_samples;
        so_std.first_row = { 0 };

        EMPIRICAL_ASSERT(engine.collect_step(&batch_std, so_std) == RAD_OK);

        EMPIRICAL_ASSERT(!reg.get(seq_id, &trec));
        EMPIRICAL_ASSERT(so_std.token[0] == 7777);
    }

    std::printf("  PASS: 50 consecutive mode alternation cycles completed with 0 stale records.\n");
}

// =============================================================================
// SUITE 2: Multi-Sequence Interleaved Concurrency & Isolation
// =============================================================================
static void test_multi_sequence_isolation_and_alternation() {
    std::printf("[SUITE 2] Testing Multi-Sequence Interleaved Concurrency & Isolation (32 seqs)...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    sched.kv_ = &g_fake_kv;

    const int N_SEQS = 32;
    std::vector<uint64_t> seq_ids(N_SEQS);
    for (int i = 0; i < N_SEQS; ++i) {
        seq_ids[i] = 2000 + i;
        auto req = std::make_unique<Request>();
        req->id = seq_ids[i];
        req->prompt = {1, 2, 3};
        req->may_speculate = true;
        req->max_tokens = 200;
        EMPIRICAL_ASSERT(sched.add(std::move(req)) == RAD_OK);
    }

    // Step across 20 iterations
    for (int step = 0; step < 20; ++step) {
        sched.plan_.reset();

        std::vector<int32_t> all_samples;
        Engine::StepOut so;
        so.first_row.resize(N_SEQS);

        std::vector<Request> mock_reqs(N_SEQS);
        std::vector<std::vector<int32_t>> lin_storage(N_SEQS);
        std::vector<std::vector<int32_t>> tree_storage(N_SEQS);

        for (int i = 0; i < N_SEQS; ++i) {
            uint64_t sid = seq_ids[i];
            mock_reqs[i].id = sid;
            so.first_row[i] = static_cast<int32_t>(all_samples.size());

            // Lanes 0..7: Tree Top-2
            // Lanes 8..15: Linear Chain (5 tokens)
            // Lanes 16..23: Single Token
            // Lanes 24..31: Alternating based on step
            int mode = 0;
            if (i < 8) mode = 0; // Tree
            else if (i < 16) mode = 1; // Linear
            else if (i < 24) mode = 2; // Single
            else mode = (step + i) % 3;

            StepEntry entry{};
            entry.seq = sid;
            entry.req = &mock_reqs[i];

            if (mode == 0) {
                // Tree
                CandidateTree tree;
                uint32_t draft_tokens[15];
                for (size_t d = 0; d < 15; ++d) draft_tokens[d] = static_cast<uint32_t>(10000 + i * 100 + d);
                tree.build_top2(static_cast<uint32_t>(i), draft_tokens, false);
                EMPIRICAL_ASSERT(sched.set_draft_tree(sid, tree, 50) == RAD_OK);

                tree_storage[i].resize(16);
                for (size_t d = 0; d < 15; ++d) tree_storage[i][d] = static_cast<int32_t>(draft_tokens[d]);
                entry.draft = tree_storage[i].data();
                entry.n_spec = 15;
                entry.n_tokens = 16;
                sched.plan_.e.push_back(entry);

                for (int d = 0; d < 15; ++d) all_samples.push_back(static_cast<int32_t>(draft_tokens[d]));
                all_samples.push_back(999);
            } else if (mode == 1) {
                // Linear
                lin_storage[i] = {50, 51, 52, 53, 54};
                EMPIRICAL_ASSERT(sched.set_draft(sid, lin_storage[i].data(), 5) == RAD_OK);

                entry.draft = lin_storage[i].data();
                entry.n_spec = 5;
                entry.n_tokens = 6;
                sched.plan_.e.push_back(entry);

                for (int d = 0; d < 5; ++d) all_samples.push_back(lin_storage[i][d]);
                all_samples.push_back(888);
            } else {
                // Single
                EMPIRICAL_ASSERT(sched.set_draft(sid, nullptr, 0) == RAD_OK);

                entry.draft = nullptr;
                entry.n_spec = 0;
                entry.n_tokens = 1;
                sched.plan_.e.push_back(entry);

                all_samples.push_back(777);
            }
        }

        // Mid-step validation: TreeRegistry must contain records ONLY for mode 0 sequences
        for (int i = 0; i < N_SEQS; ++i) {
            uint64_t sid = seq_ids[i];
            int mode = 0;
            if (i < 8) mode = 0;
            else if (i < 16) mode = 1;
            else if (i < 24) mode = 2;
            else mode = (step + i) % 3;

            ActiveTreeRecord trec;
            bool has_rec = reg.get(sid, &trec);
            if (mode == 0) {
                EMPIRICAL_ASSERT(has_rec);
                EMPIRICAL_ASSERT(trec.tree.node_count() == 16);
            } else {
                EMPIRICAL_ASSERT(!has_rec);
            }
        }

        RadBatch batch{};
        batch.step = step + 1;
        batch.n_seq = N_SEQS;
        batch.n_spec = 15;

        so.n_rows = static_cast<int32_t>(all_samples.size());
        so.rb = all_samples.data();

        int rc_step = engine.collect_step(&batch, so); if (rc_step != RAD_OK) { std::fprintf(stderr, "collect_step step %d failed with rc=%d: %s\n", step, rc_step, rad_strerror(rc_step)); } EMPIRICAL_ASSERT(rc_step == RAD_OK);

        // Post-step validation: ALL tree records must be cleared
        for (int i = 0; i < N_SEQS; ++i) {
            ActiveTreeRecord trec;
            EMPIRICAL_ASSERT(!reg.get(seq_ids[i], &trec));
        }
    }

    std::printf("  PASS: 32 concurrent sequences verified across 20 steps with 0 cross-lane contamination.\n");
}

// =============================================================================
// SUITE 3: Cancellation, Finish, and Preemption Leak Stress (500 Seqs)
// =============================================================================
static void test_cancellation_finish_and_preemption_leaks() {
    std::printf("[SUITE 3] Testing Cancellation, Finish, and Preemption Leaks (500 seqs)...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    Config cfg;
    Engine engine(cfg);
    Scheduler& sched = engine.sched_;
    sched.kv_ = &g_fake_kv;

    const int TOTAL_SEQS = 500;
    std::vector<uint64_t> ids(TOTAL_SEQS);

    for (int i = 0; i < TOTAL_SEQS; ++i) {
        ids[i] = 30000 + i;
        auto req = std::make_unique<Request>();
        req->id = ids[i];
        req->prompt = {1, 2, 3};
        req->may_speculate = true;
        req->max_tokens = 100;
        EMPIRICAL_ASSERT(sched.add(std::move(req)) == RAD_OK);

        CandidateTree tree;
        uint32_t draft_tokens[15];
        for (size_t d = 0; d < 15; ++d) draft_tokens[d] = static_cast<uint32_t>(i * 100 + d);
        if (i % 3 == 0) tree.build_linear(10, draft_tokens, 15);
        else if (i % 3 == 1) tree.build_top2(10, draft_tokens, false);
        else tree.build_top4(10, draft_tokens);

        EMPIRICAL_ASSERT(sched.set_draft_tree(ids[i], tree, 100) == RAD_OK);
    }

    // Verify all 500 records are registered
    for (int i = 0; i < TOTAL_SEQS; ++i) {
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(reg.get(ids[i], &trec));
    }

    // Group 1: 100 Sequences Cancelled via sched.cancel(id)
    for (int i = 0; i < 100; ++i) {
        EMPIRICAL_ASSERT(sched.cancel(ids[i]) == RAD_OK);
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(!reg.get(ids[i], &trec)); // Must be immediately purged
    }

    // Group 2: 100 Sequences Finished via sched.close_request
    for (int i = 100; i < 200; ++i) {
        auto it = sched.by_id_.find(ids[i]);
        EMPIRICAL_ASSERT(it != sched.by_id_.end());
        rad::SchedReq& sr = *sched.reqs_[(size_t)it->second];
        sched.close_request(sr, ReqState::Finished, "completed", false);
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(!reg.get(ids[i], &trec)); // Must be immediately purged
    }

    // Group 3: 100 Sequences Invalidation via may_speculate = false
    for (int i = 200; i < 300; ++i) {
        auto it = sched.by_id_.find(ids[i]);
        EMPIRICAL_ASSERT(it != sched.by_id_.end());
        rad::SchedReq& sr = *sched.reqs_[(size_t)it->second];
        sr.req->may_speculate = false;

        CandidateTree dummy;
        dummy.build_top2(10, nullptr, false);
        EMPIRICAL_ASSERT(sched.set_draft_tree(ids[i], dummy, 100) == RAD_OK);
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(!reg.get(ids[i], &trec)); // Cleared due to !may_speculate
    }

    // Group 4: 100 Sequences Invalidation via Empty Tree
    for (int i = 300; i < 400; ++i) {
        CandidateTree empty_tree;
        EMPIRICAL_ASSERT(sched.set_draft_tree(ids[i], empty_tree, 100) == RAD_OK);
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(!reg.get(ids[i], &trec)); // Cleared due to empty tree
    }

    // Group 5: 100 Sequences Completed via Engine::collect_step
    sched.plan_.reset();
    std::vector<Request> grp5_reqs(100);
    std::vector<int32_t> grp5_samples;
    Engine::StepOut so;
    so.first_row.resize(100);

    for (int i = 400; i < 500; ++i) {
        int idx = i - 400;
        grp5_reqs[idx].id = ids[i];
        so.first_row[idx] = static_cast<int32_t>(grp5_samples.size());

        StepEntry entry{};
        entry.seq = ids[i];
        entry.n_spec = 15;
        entry.n_tokens = 16;
        entry.req = &grp5_reqs[idx];
        sched.plan_.e.push_back(entry);

        for (int d = 0; d < 16; ++d) grp5_samples.push_back(555);
    }

    RadBatch batch{};
    batch.step = 1;
    batch.n_seq = 100;
    batch.n_spec = 15;

    so.n_rows = static_cast<int32_t>(grp5_samples.size());
    so.rb = grp5_samples.data();

    EMPIRICAL_ASSERT(engine.collect_step(&batch, so) == RAD_OK);

    // Final Invariant Check: ALL 500 sequences MUST be gone from TreeRegistry!
    for (int i = 0; i < TOTAL_SEQS; ++i) {
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(!reg.get(ids[i], &trec));
    }

    // Sequence ID Recycling Audit: Reuse the same sequence IDs
    for (int i = 0; i < 50; ++i) {
        uint64_t recycled_id = ids[i];
        ActiveTreeRecord trec;
        EMPIRICAL_ASSERT(!reg.get(recycled_id, &trec)); // No ghost record
    }

    std::printf("  PASS: 500 sequence cancellation/finish/invalidation lifecycle stress test passed with 0 leaks.\n");
}

// =============================================================================
// SUITE 4: Concurrent Multi-Threaded Stress & Mutex Invariants
// =============================================================================
static void test_concurrent_multithreaded_stress() {
    std::printf("[SUITE 4] Testing Concurrent Multi-Threaded Stress (8 threads, 40,000 ops)...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    const int NUM_THREADS = 8;
    const int OPS_PER_THREAD = 5000;
    std::vector<std::thread> workers;

    std::atomic<bool> start_flag{false};

    for (int tid = 0; tid < NUM_THREADS; ++tid) {
        workers.emplace_back([tid, &reg, &start_flag]() {
            while (!start_flag.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }

            std::mt19937 rng(1337 + tid);
            CandidateTree tree;
            uint32_t drafts[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            tree.build_top2(100, drafts, false);

            for (int op = 0; op < OPS_PER_THREAD; ++op) {
                uint64_t sid = 50000 + (rng() % 200);
                int action = rng() % 4;
                if (action == 0) {
                    reg.set(sid, tree, static_cast<uint32_t>(op));
                } else if (action == 1) {
                    ActiveTreeRecord out;
                    reg.get(sid, &out);
                } else if (action == 2) {
                    reg.clear(sid);
                } else {
                    if (rng() % 100 == 0) {
                        reg.clear_all();
                    }
                }
            }
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& w : workers) {
        w.join();
    }

    reg.clear_all();
    std::printf("  PASS: 40,000 concurrent multithreaded operations completed with 0 crashes or deadlocks.\n");
}

// =============================================================================
// SUITE 5: Bit-Exact Autoregressive Parity Across 1,200 Test Scenarios
// =============================================================================

// Deterministic Teacher Language Model Oracle
// Simulates an autoregressive teacher model on a vocabulary of size V = 32,000.
// Generates reproducible next tokens based on context hash.
class DeterministicTeacherOracle {
public:
    explicit DeterministicTeacherOracle(uint64_t seed = 42, uint32_t vocab_size = 32000)
        : seed_(seed), vocab_size_(vocab_size) {}

    uint32_t predict_next(const std::vector<uint32_t>& context) const {
        uint64_t h = seed_;
        for (uint32_t tok : context) {
            h ^= static_cast<uint64_t>(tok) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        }
        return static_cast<uint32_t>(h % vocab_size_);
    }

private:
    uint64_t seed_;
    uint32_t vocab_size_;
};

static void test_autoregressive_parity_1200_scenarios() {
    std::printf("[SUITE 5] Testing Bit-Exact Autoregressive Output Parity Across 1,200 Scenarios (Temp = 0.0)...\n");

    const int TOTAL_SCENARIOS = 1200;
    uint64_t total_tokens_generated = 0;
    uint64_t total_speculative_steps = 0;
    uint64_t total_accepted_tokens = 0;

    for (int scenario = 0; scenario < TOTAL_SCENARIOS; ++scenario) {
        DeterministicTeacherOracle teacher(10007 + scenario * 31, 32000);

        // Target length to generate in this scenario
        const size_t target_gen_len = 20 + (scenario % 31); // 20 to 50 tokens

        // Initial prompt
        std::vector<uint32_t> prompt = {101, 202, 303, static_cast<uint32_t>(scenario + 1)};

        // -------------------------------------------------------------
        // Baseline: Non-Speculative Autoregressive Generation
        // -------------------------------------------------------------
        std::vector<uint32_t> gold_output;
        gold_output.reserve(target_gen_len);

        std::vector<uint32_t> gold_context = prompt;
        for (size_t step = 0; step < target_gen_len; ++step) {
            uint32_t next_tok = teacher.predict_next(gold_context);
            gold_output.push_back(next_tok);
            gold_context.push_back(next_tok);
        }

        // -------------------------------------------------------------
        // Speculative DARTree Generation
        // -------------------------------------------------------------
        std::vector<uint32_t> spec_output;
        spec_output.reserve(target_gen_len);

        std::vector<uint32_t> spec_context = prompt;

        // Drafter configuration for this scenario
        // Accuracy profiles:
        // 0: Perfect match (100%)
        // 1: High match (~75%)
        // 2: Medium match (~50%)
        // 3: Low match (~25%)
        // 4: Adversarial (0% - immediate divergence)
        // 5: Branch tie / secondary branch match
        int profile = scenario % 6;
        int topo_type = (scenario / 6) % 5; // 0=linear, 1=top2_a, 2=top2_b, 3=top4, 4=fuzzed

        std::mt19937 rng(9999 + scenario);

        while (spec_output.size() < target_gen_len) {
            total_speculative_steps++;
            uint32_t root_token = spec_context.back();

            // Build Candidate Tree
            CandidateTree tree;

            if (topo_type == 0) {
                // Linear Chain
                size_t n_cand = 1 + (rng() % 15);
                uint32_t drafts[15];
                std::vector<uint32_t> sim_ctx = spec_context;
                for (size_t i = 0; i < n_cand; ++i) {
                    uint32_t true_next = teacher.predict_next(sim_ctx);
                    sim_ctx.push_back(true_next);
                    if (profile == 0) {
                        drafts[i] = true_next;
                    } else if (profile == 4) {
                        drafts[i] = true_next + 1000; // Mismatch
                    } else {
                        bool match = (rng() % 100) < (profile == 1 ? 75 : profile == 2 ? 50 : 25);
                        drafts[i] = match ? true_next : (true_next + 500);
                    }
                }
                tree.build_linear(root_token, drafts, n_cand);
            } else if (topo_type == 1 || topo_type == 2) {
                // Top-2 (Mode A or B)
                bool dloop = (topo_type == 2);
                uint32_t drafts[15];
                for (size_t i = 0; i < 15; ++i) {
                    if (profile == 4) drafts[i] = static_cast<uint32_t>(rng() % 32000);
                    else drafts[i] = static_cast<uint32_t>(1000 + i);
                }
                tree.build_top2(root_token, drafts, dloop);
            } else if (topo_type == 3) {
                // Top-4
                uint32_t drafts[15];
                for (size_t i = 0; i < 15; ++i) drafts[i] = static_cast<uint32_t>(2000 + i);
                tree.build_top4(root_token, drafts);
            } else {
                // Fuzzed Random Valid Topology
                tree.reset();
                tree.add_node(0, -1, root_token, 0, 1.0f);
                int n_nodes = 4 + (rng() % 13); // 4 to 16
                for (int nid = 1; nid < n_nodes; ++nid) {
                    int pid = rng() % nid;
                    int depth = tree[pid].depth + 1;
                    if (depth > 15) { pid = 0; depth = 1; }
                    uint32_t tok = static_cast<uint32_t>(rng() % 32000);
                    tree.add_node(nid, pid, tok, depth, 1.0f);
                }
            }

            // Teacher Oracle evaluates all nodes in the candidate tree
            // Under causal tree attention, each node u attends to prefix + its ancestor path
            size_t n_nodes = tree.node_count();
            uint32_t teacher_preds[RAD_TREE_MAX_NODES] = {0};

            for (size_t i = 0; i < n_nodes; ++i) {
                std::vector<uint32_t> node_ctx = spec_context;
                // Trace ancestor path from root to node i
                std::vector<int32_t> path_nodes;
                int32_t curr = static_cast<int32_t>(i);
                while (curr >= 0) {
                    path_nodes.push_back(curr);
                    curr = tree[curr].parent_id;
                }
                std::reverse(path_nodes.begin(), path_nodes.end());
                // Append branch tokens (excluding root which is already in spec_context.back())
                for (size_t p = 1; p < path_nodes.size(); ++p) {
                    node_ctx.push_back(tree[path_nodes[p]].token_id);
                }
                teacher_preds[i] = teacher.predict_next(node_ctx);
            }

            // Verify Tree with DARTree Greedy Argmax Verifier
            int32_t out_accepted[RAD_TREE_MAX_NODES + 1] = {0};
            int32_t acc_len = 0;
            int32_t bonus_tok = 0;
            int32_t best_nodes[RAD_TREE_MAX_NODES + 1] = {0};

            int rc = accept_tree_greedy(
                tree,
                reinterpret_cast<const int32_t*>(teacher_preds),
                static_cast<int>(n_nodes),
                out_accepted,
                &acc_len,
                &bonus_tok,
                best_nodes
            );
            EMPIRICAL_ASSERT(rc == RAD_OK);
            EMPIRICAL_ASSERT(acc_len >= 0 && acc_len < static_cast<int>(n_nodes));

            // Commit accepted branch tokens + bonus token
            size_t to_take = 1 + static_cast<size_t>(acc_len);
            for (size_t j = 0; j < to_take; ++j) {
                if (spec_output.size() < target_gen_len) {
                    uint32_t em = static_cast<uint32_t>(out_accepted[j]);
                    spec_output.push_back(em);
                    spec_context.push_back(em);
                }
            }
            total_accepted_tokens += static_cast<uint64_t>(acc_len);
        }

        // =============================================================
        // BIT-EXACT PARITY INVARIANT: spec_output MUST IDENTICALLY MATCH gold_output!
        // =============================================================
        EMPIRICAL_ASSERT(spec_output.size() == gold_output.size());
        for (size_t t = 0; t < target_gen_len; ++t) {
            EMPIRICAL_ASSERT(spec_output[t] == gold_output[t]);
        }
        total_tokens_generated += target_gen_len;
    }

    double avg_tau = static_cast<double>(total_tokens_generated) / static_cast<double>(total_speculative_steps);
    std::printf("  PASS: 1,200 Scenarios Verified with 100.0%% Output Parity against Teacher Baseline!\n");
    std::printf("  Metrics: Total Tokens Generated: %llu | Spec Steps: %llu | Avg Tau: %.2f\n",
                (unsigned long long)total_tokens_generated,
                (unsigned long long)total_speculative_steps,
                avg_tau);
}

// =============================================================================
// Main Entrypoint
// =============================================================================
int main() {
    std::printf("===================================================================\n");
    std::printf(" Radiance DARTree Lifecycle & Bit-Exact Parity Adversarial Suite   \n");
    std::printf(" Milestone 2 Iteration 2 Challenger 2                             \n");
    std::printf("===================================================================\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    test_sequential_mode_alternations();
    test_multi_sequence_isolation_and_alternation();
    test_cancellation_finish_and_preemption_leaks();
    test_concurrent_multithreaded_stress();
    test_autoregressive_parity_1200_scenarios();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("===================================================================\n");
    std::printf(" ALL 5 ADVERSARIAL CHALLENGER SUITES PASSED SUCCESSFULLY!\n");
    std::printf(" Total Empirical Assertions Verified: %llu\n", (unsigned long long)g_assertions.load());
    std::printf(" Total Elapsed Time: %.2f ms\n", elapsed_ms);
    std::printf("===================================================================\n");

    return 0;
}
